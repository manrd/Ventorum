# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""3D source-doublet panel method solver family integration (Ventorum-3DP).

This module connects the 3D panel method to the Ventorum solver architecture.
It takes an Aircraft, FlightCondition (or sweep), and SolverSettings, and
returns standard SolverResult objects with integrated totals, spanwise strip
loads, trust scoring, and execution dispatch records.

Published sources
-----------------
* Morino, L. and Kuo, C. C., 1974, Subsonic potential aerodynamics for
  complex configurations: A general theory, AIAA Journal, Vol. 12, No. 2,
  pp. 191-197.
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, section 9.4 and chapter 12.
* Wieselsberger, C., 1921, Ueber den Fluegelwiderstand in der Naehe des Bodens,
  Zeitschrift fuer Flugtechnik und Motorluftschiffahrt, Vol. 12, No. 10.
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from collections.abc import Sequence

import numpy as np

from ventorum import gpu
from ventorum.aero.system import (
    GroundPlane,
    freestream_direction,
    is_symmetric_condition,
    lift_direction,
)
from ventorum.airfoil import airfoil
from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    IntegratedResult,
    LiftingSurface,
    SolverResult,
    SolverSettings,
    SpanwiseResult,
)
from ventorum.core.errors import ValidityError
from ventorum.core.trust import evaluate_aerodynamic_trust
from ventorum.geometry.lattice import (
    _AirfoilBlender,
    _sorted_sections,
    _surface_eta,
    airfoil_linear_properties,
    resolve_spacing,
    surface_reference_line,
)
from ventorum.panel3d.images import (
    check_ground_clearance,
    ground_plane_parameters,
)
from ventorum.panel3d.mesh import (
    PanelMesh,
    check_mesh,
    mesh_lifting_surface,
)
from ventorum.panel3d.solver import PanelResult, PanelSolver
from ventorum.solvers.base import BaseSolver
from ventorum.solvers.lattice_base import as_aircraft
from ventorum.utils.validation import validate_aircraft

__all__ = ["PanelMethodSolver"]

_DEFAULT_N_CHORD = 20


def _estimate_panel_mesh_error(aircraft: Aircraft, condition: FlightCondition, settings: SolverSettings):
    """Run a three-level panel convergence study for the trust notes.

    Parameters
    ----------
    aircraft : Aircraft
        Case geometry.
    condition : FlightCondition
        Flight condition of the study.
    settings : SolverSettings
        Solver settings of the base (coarsest) mesh.

    Returns
    -------
    ConvergenceReport
        Values per level, observed order, Richardson value and relative
        error of CL, CDi and Cm.
    """
    from ventorum.panel3d.convergence import panel_mesh_convergence

    return panel_mesh_convergence(
        aircraft, condition, settings, levels=3, ratio=1.5, quantities=("CL", "CDi", "Cm")
    )


def _build_surface_strip_panel_map(
    surf: LiftingSurface,
    n_span: int,
    n_chord: int,
    half: bool,
    sharp_te: np.ndarray,
) -> tuple[list[list[int]], list[list[int]] | None]:
    """Map panels to spanwise strips for one lifting surface.

    Parameters
    ----------
    surf : LiftingSurface
        Lifting surface definition.
    n_span : int
        Spanwise panels on the semi-span.
    n_chord : int
        Chordwise panels per surface.
    half : bool
        True if meshed as right half only.
    sharp_te : numpy.ndarray
        Boolean array of shape (n_span + 1,) indicating sharp trailing edge.

    Returns
    -------
    right_strip_panels : list of list of int
        List of panel indices for each strip on the right half (root to tip).
    left_strip_panels : list of list of int or None
        List of panel indices for each strip on the left half (tip to root),
        or None if half is True or surface is not symmetric.
    """
    p_idx = 0
    right_strip_panels: list[list[int]] = [[] for _ in range(n_span)]

    for j in range(n_span):
        for _ in range(n_chord):
            right_strip_panels[j].append(p_idx)
            p_idx += 1
            right_strip_panels[j].append(p_idx)
            p_idx += 1
        if not (sharp_te[j] and sharp_te[j + 1]):
            right_strip_panels[j].append(p_idx)
            p_idx += 1

    # Tip cap panels at tip station
    n_cap_panels = 1 if n_chord == 1 else n_chord
    for _ in range(n_cap_panels):
        right_strip_panels[-1].append(p_idx)
        p_idx += 1

    symmetric = bool(surf.is_symmetric)
    if not symmetric:
        # Root cap panels at root station
        for _ in range(n_cap_panels):
            right_strip_panels[0].append(p_idx)
            p_idx += 1

    right_n_panels = p_idx

    if symmetric and not half:
        left_strip_panels: list[list[int]] = [[] for _ in range(n_span)]
        shift = right_n_panels
        # Left strips run from left tip to root (k = 0 at left tip, corresponding
        # to right strip n_span - 1 - k).
        for k in range(n_span):
            j = n_span - 1 - k
            left_strip_panels[k] = [shift + p for p in right_strip_panels[j]]
    else:
        left_strip_panels = None

    return right_strip_panels, left_strip_panels


class PanelMethodSolver(BaseSolver):
    """Inviscid three-dimensional potential flow panel solver.

    Follows the BaseSolver interface for full integration with Aircraft,
    FlightCondition, SolverSettings, and SolverResult.
    """

    name: str = "panel3d"

    def _resolve_backend(self, settings: SolverSettings) -> tuple[str, str, str]:
        """Resolve backend, device, and arithmetic dtype for 3D panel solve."""
        user_backend = getattr(settings, "backend", None)
        if user_backend is not None:
            backend = str(user_backend).strip().lower()
            if backend not in ("numpy", "numba", "gpu"):
                raise ValueError(
                    f"Unknown panel backend {backend!r}; use 'numpy', 'numba' or 'gpu'."
                )
            device = "gpu" if backend == "gpu" else "cpu"
            dtype = "float32" if device == "gpu" else "float64"
            return backend, device, dtype

        if gpu.get_device() == "gpu" and gpu.available():
            return "gpu", "gpu", "float32"

        return "numpy", "cpu", "float64"

    def solve(
        self,
        aircraft: Aircraft | LiftingSurface,
        condition: FlightCondition | Sequence[FlightCondition],
        settings: SolverSettings,
        *,
        ground: GroundPlane | None = None,
        ref_point: np.ndarray | None = None,
    ) -> SolverResult | list[SolverResult]:
        """Solve one flight condition or a sequence of conditions.

        Parameters
        ----------
        aircraft : Aircraft or LiftingSurface
            Geometry definition.
        condition : FlightCondition or sequence of FlightCondition
            Operating free-stream condition(s).
        settings : SolverSettings
            Discretization and solution knobs.
        ground : GroundPlane or None, optional
            Ground plane definition.
        ref_point : numpy.ndarray or None, optional
            Moment reference point [m].

        Returns
        -------
        SolverResult or list of SolverResult

        Mesh error estimate. Set the ``estimate_mesh_error`` attribute of
        the settings to True to run a three-level panel mesh convergence
        study (ratio 1.5 in both directions) for each condition. The study
        appends its notes to the trust warnings and stores the report in
        ``details["mesh_convergence"]``. It costs about 25 to 30 times one
        base solve, so it is off by default.
        """
        is_single = isinstance(condition, FlightCondition)
        cond_list: list[FlightCondition] = [condition] if is_single else list(condition)

        ac = as_aircraft(aircraft)
        validate_aircraft(ac)
        ac.compute_reference_values()

        s_ref = ac.S_ref
        b_ref = ac.b_ref
        c_ref = ac.c_ref
        rp = ac.moment_reference() if ref_point is None else np.asarray(ref_point, dtype=float)

        backend, device, dtype = self._resolve_backend(settings)
        kutta = getattr(settings, "kutta", "morino")
        wake_length = getattr(settings, "wake_length", None)
        n_chord = int(settings.n_chord) if getattr(settings, "n_chord", None) is not None else _DEFAULT_N_CHORD
        chord_spacing = getattr(settings, "chord_spacing", "cosine")
        use_sym_setting = getattr(settings, "use_symmetry", True)
        estimate_mesh_error = bool(getattr(settings, "estimate_mesh_error", False))

        all_surfs_sym = all(s.is_symmetric for s in ac.surfaces)
        all_conds_sym = all(is_symmetric_condition(c, ground) for c in cond_list)
        all_phi_zero = all(getattr(c, "phi", 0.0) == 0.0 for c in cond_list)
        use_half = bool(use_sym_setting and all_surfs_sym and all_conds_sym and all_phi_zero)

        # 1. Section shape extraction and coordinate preparation
        notes: list[str] = []
        section_coords_by_surface: list[list[np.ndarray]] = []
        for surf in ac.surfaces:
            surf_coords: list[np.ndarray] = []
            for sec in surf.sections:
                shape_spec = getattr(sec, "shape", None)
                if shape_spec is None:
                    shape_obj = airfoil("0012")
                    notes.append(
                        f"Section shape not specified on surface '{surf.name}' section at "
                        f"y_frac={sec.y_frac:.2f}; default NACA 0012 used."
                    )
                else:
                    shape_obj = airfoil(shape_spec)
                surf_coords.append(shape_obj.coords)
            section_coords_by_surface.append(surf_coords)

        # 2. Build panel mesh and track strip-to-panel associations
        node_parts = []
        panel_parts = []
        kind_parts = []
        comp_parts = []
        wake_parts = []
        n_nodes_total = 0
        n_panels_total = 0

        surface_strip_maps: list[tuple[list[list[int]], list[list[int]] | None]] = []
        surface_geom_meta: list[dict] = []

        ref_semi = max((s.semi_span for s in ac.surfaces), default=1.0)
        for s_idx, (surf, coords) in enumerate(zip(ac.surfaces, section_coords_by_surface)):
            if getattr(settings, "proportional_panels", False) and surf.n_panels is None:
                from ventorum.geometry.lattice import compute_surface_n_panels

                n_span = compute_surface_n_panels(
                    surf,
                    base_n_panels=settings.n_panels,
                    reference_semi_span=ref_semi,
                    min_panels=getattr(settings, "min_panels", 8),
                )
            else:
                n_span = surf.n_panels if surf.n_panels is not None else settings.n_panels

            sub = mesh_lifting_surface(
                surf,
                coords,
                n_chord=n_chord,
                chord_spacing=chord_spacing,
                tip="flat",
                half=use_half,
                n_span=n_span,
            )

            # Determine sharp trailing edge array
            xt = np.linspace(0.0, 1.0, n_chord + 1)
            fr = np.array([float(s.y_frac) for s in _sorted_sections(surf)])
            spacing = resolve_spacing(surf.spacing if surf.spacing is not None else settings.spacing, surf, "vlm")
            eta_e, eta_mid = _surface_eta(surf, n_span, spacing)
            yu_list = []
            yl_list = []
            for c in coords:
                from ventorum.panel3d.mesh import _resample_section

                yu, yl = _resample_section(c, xt)
                yu_list.append(yu)
                yl_list.append(yl)
            from ventorum.panel3d.mesh import _interp_rows

            yu_e = _interp_rows(fr, np.vstack(yu_list), eta_e)
            yl_e = _interp_rows(fr, np.vstack(yl_list), eta_e)
            sharp_te = np.abs(yu_e[:, -1] - yl_e[:, -1]) <= 1e-6

            r_map, l_map = _build_surface_strip_panel_map(surf, n_span, n_chord, use_half, sharp_te)
            # Offset panel indices for combined mesh
            r_map_offset = [[p + n_panels_total for p in strip_p] for strip_p in r_map]
            l_map_offset = (
                [[p + n_panels_total for p in strip_p] for strip_p in l_map]
                if l_map is not None
                else None
            )
            surface_strip_maps.append((r_map_offset, l_map_offset))

            # Store geometry stations and properties for strip integration
            x_ref, y_ref, z_ref = surface_reference_line(surf, eta_mid)
            offset = np.asarray(surf.position, dtype=float).reshape(3)
            pos_mid = np.column_stack([x_ref, y_ref, z_ref]) + offset

            chords_def = np.array([float(s.chord) for s in _sorted_sections(surf)])
            strip_chords = np.interp(eta_mid, fr, chords_def)
            x_e, y_e, z_e = surface_reference_line(surf, eta_e)
            pos_e = np.column_stack([x_e, y_e, z_e]) + offset
            dl = pos_e[1:] - pos_e[:-1]
            strip_width = np.sqrt(dl[:, 1] ** 2 + dl[:, 2] ** 2)
            edge_ch = np.interp(eta_e, fr, chords_def)
            strip_area = 0.5 * (edge_ch[:-1] + edge_ch[1:]) * strip_width

            # Linear airfoil properties for alpha_eff
            blender = _AirfoilBlender(_sorted_sections(surf))
            afs = [blender.at(float(e)) for e in eta_mid]
            props = [airfoil_linear_properties(af) for af in afs]
            a0 = np.array([p[0] for p in props], dtype=float)
            alpha_L0 = np.array([p[1] for p in props], dtype=float)

            surface_geom_meta.append({
                "name": surf.name,
                "is_symmetric": bool(surf.is_symmetric),
                "pos_mid": pos_mid,
                "chord": strip_chords,
                "width": strip_width,
                "area": strip_area,
                "a0": a0,
                "alpha_L0": alpha_L0,
            })

            node_parts.append(np.asarray(sub.nodes, dtype=float))
            panel_parts.append(np.asarray(sub.panels, dtype=int) + n_nodes_total)
            kind_parts.append(np.asarray(sub.kind))
            comp_parts.append(np.full(np.asarray(sub.panels).shape[0], s_idx, dtype=int))
            w = np.asarray(sub.wake_edges, dtype=int).copy()
            if w.size:
                w[:, 0:2] += n_panels_total
                w[:, 2:4] += n_nodes_total
            wake_parts.append(w.reshape(-1, 4))
            n_nodes_total += int(np.asarray(sub.nodes).shape[0])
            n_panels_total += int(np.asarray(sub.panels).shape[0])

        mesh = PanelMesh(
            nodes=np.vstack(node_parts),
            panels=np.vstack(panel_parts),
            component=np.concatenate(comp_parts),
            kind=np.concatenate(kind_parts),
            wake_edges=np.vstack(wake_parts).reshape(-1, 4),
            half=use_half,
        )

        # 3. Check mesh quality
        mesh_stats = check_mesh(mesh)

        # 4. Solve potential flow with PanelSolver
        t0 = time.perf_counter()
        solver = PanelSolver(
            mesh,
            wake_length=wake_length,
            S_ref=s_ref,
            b_ref=b_ref,
            c_ref=c_ref,
            ref_point=rp,
            kutta=kutta,
            backend=backend,
        )
        panel_results: list[PanelResult] = solver.solve_conditions(cond_list)

        # Factorization count
        ground_groups: set[tuple] = set()
        for c in cond_list:
            if c.h is None:
                ground_groups.add(("free",))
            else:
                ground_groups.add(("ground", round(float(c.alpha), 11), round(float(c.h), 11)))
        factorization_count = max(1, len(ground_groups))

        dispatch = {
            "backend": backend,
            "device": device,
            "dtype": dtype,
            "factorization_count": factorization_count,
            "precision": dtype,
            "backends": {"kernel": backend, "solve": "torch" if device == "gpu" else "scipy"},
            "threads": 1,
            "workers": 1,
            "plan_source": "default",
            "profile": None,
            "reason": "CPU path selected" if device == "cpu" else "GPU path selected",
        }

        # 5. Build SolverResult per condition with strip integration
        x_hat = np.array([1.0, 0.0, 0.0])
        dt_per_case = (time.perf_counter() - t0) / max(1, len(cond_list))
        results: list[SolverResult] = []

        for cond, p_res in zip(cond_list, panel_results):
            alpha = float(cond.alpha)
            beta = float(cond.beta)
            v_inf = float(cond.V_inf)
            q_inf = 0.5 * float(cond.rho) * v_inf**2
            l_hat = lift_direction(alpha)
            d_hat = freestream_direction(alpha, beta)

            # Ground clearance check
            h_min = None
            if cond.h is not None:
                normal, offset = ground_plane_parameters(cond, rp)
                h_min = check_ground_clearance(solver._full_mesh.nodes, normal, offset)
                if h_min <= 0.0:
                    raise ValidityError(
                        f"Ground strike: minimum clearance is {h_min:.3g} m (<= 0)."
                    )

            # Evaluate forces on mesh panels
            # p_res.cp corresponds to solver.mesh panels (half mesh if use_half)
            cp_arr = p_res.cp
            areas = solver.areas
            normals = solver.normals
            centroids = solver.centroids

            # Panel pressure forces in geometry axes: dF = -Cp * q_inf * Area * n
            dF_panel = -cp_arr[:, None] * areas[:, None] * normals * q_inf

            # Build SpanwiseResult for each surface
            spanwise_list: list[SpanwiseResult] = []
            for s_idx, meta in enumerate(surface_geom_meta):
                r_map, l_map = surface_strip_maps[s_idx]
                n_sp = len(r_map)

                r_mid = meta["pos_mid"]
                r_chord = meta["chord"]
                r_width = meta["width"]
                r_area = meta["area"]

                # Right half strip forces and moments
                r_lift = np.empty(n_sp)
                r_drag = np.empty(n_sp)
                r_cm = np.empty(n_sp)
                for j in range(n_sp):
                    p_ids = r_map[j]
                    fj = np.sum(dF_panel[p_ids], axis=0) if p_ids else np.zeros(3)
                    r_lift[j] = float(np.dot(fj, l_hat))
                    r_drag[j] = float(np.dot(fj, d_hat))

                    # Quarter-chord point
                    qc_j = r_mid[j] + 0.25 * r_chord[j] * x_hat
                    arm = centroids[p_ids] - qc_j[None, :] if p_ids else np.zeros((0, 3))
                    mj = np.sum(np.cross(arm, dF_panel[p_ids]), axis=0) if p_ids else np.zeros(3)
                    # Pitching moment (nose up): My in body axes
                    r_cm[j] = float(mj[1]) / max(q_inf * r_area[j] * r_chord[j], 1e-300)

                r_cl = r_lift / np.maximum(q_inf * r_area, 1e-300)
                r_cdi = r_drag / np.maximum(q_inf * r_area, 1e-300)
                r_local_lift = r_lift / np.maximum(r_width, 1e-300)
                r_gamma = r_local_lift / max(float(cond.rho) * v_inf, 1e-300)
                r_alpha_eff = meta["alpha_L0"] + r_cl / np.maximum(meta["a0"], 1e-9)
                r_alpha_i = alpha - r_alpha_eff

                if meta["is_symmetric"]:
                    if l_map is not None:
                        # Direct integration on full mesh left strips
                        l_lift = np.empty(n_sp)
                        l_drag = np.empty(n_sp)
                        l_cm = np.empty(n_sp)
                        for k in range(n_sp):
                            p_ids = l_map[k]
                            j = n_sp - 1 - k
                            fk = np.sum(dF_panel[p_ids], axis=0) if p_ids else np.zeros(3)
                            l_lift[k] = float(np.dot(fk, l_hat))
                            l_drag[k] = float(np.dot(fk, d_hat))

                            qc_k = np.array([r_mid[j, 0], -r_mid[j, 1], r_mid[j, 2]]) + 0.25 * r_chord[j] * x_hat
                            arm = centroids[p_ids] - qc_k[None, :] if p_ids else np.zeros((0, 3))
                            mk = np.sum(np.cross(arm, dF_panel[p_ids]), axis=0) if p_ids else np.zeros(3)
                            l_cm[k] = float(mk[1]) / max(q_inf * r_area[j] * r_chord[j], 1e-300)

                        l_cl = l_lift / np.maximum(q_inf * r_area[::-1], 1e-300)
                        l_cdi = l_drag / np.maximum(q_inf * r_area[::-1], 1e-300)
                        l_local_lift = l_lift / np.maximum(r_width[::-1], 1e-300)
                        l_gamma = l_local_lift / max(float(cond.rho) * v_inf, 1e-300)
                        l_alpha_eff = meta["alpha_L0"][::-1] + l_cl / np.maximum(meta["a0"][::-1], 1e-9)
                        l_alpha_i = alpha - l_alpha_eff
                    else:
                        # Symmetric flow reflection on half mesh
                        l_lift = r_lift[::-1]
                        l_cl = r_cl[::-1]
                        l_cdi = r_cdi[::-1]
                        l_cm = r_cm[::-1]
                        l_local_lift = r_local_lift[::-1]
                        l_gamma = r_gamma[::-1]
                        l_alpha_eff = r_alpha_eff[::-1]
                        l_alpha_i = r_alpha_i[::-1]

                    y_full = np.concatenate([-r_mid[::-1, 1], r_mid[:, 1]])
                    chord_full = np.concatenate([r_chord[::-1], r_chord])
                    cl_full = np.concatenate([l_cl, r_cl])
                    cdi_full = np.concatenate([l_cdi, r_cdi])
                    cm_full = np.concatenate([l_cm, r_cm])
                    local_lift_full = np.concatenate([l_local_lift, r_local_lift])
                    gamma_full = np.concatenate([l_gamma, r_gamma])
                    alpha_eff_full = np.concatenate([l_alpha_eff, r_alpha_eff])
                    alpha_i_full = np.concatenate([l_alpha_i, r_alpha_i])
                else:
                    y_full = r_mid[:, 1]
                    chord_full = r_chord
                    cl_full = r_cl
                    cdi_full = r_cdi
                    cm_full = r_cm
                    local_lift_full = r_local_lift
                    gamma_full = r_gamma
                    alpha_eff_full = r_alpha_eff
                    alpha_i_full = r_alpha_i

                spanwise_list.append(
                    SpanwiseResult(
                        y=y_full,
                        gamma=gamma_full,
                        Cl=cl_full,
                        Cd_i=cdi_full,
                        alpha_eff=alpha_eff_full,
                        alpha_i=alpha_i_full,
                        local_lift=local_lift_full,
                        surface_name=meta["name"],
                        chord=chord_full,
                        Cm_section=cm_full,
                    )
                )

            # Integrated totals
            ar = float(b_ref**2 / s_ref)
            cdi = float(p_res.CDi_trefftz)
            cl = float(p_res.CL)
            e_factor = cl**2 / (np.pi * ar * cdi) if cdi > 1e-12 else float("nan")

            totals = IntegratedResult(
                CL=cl,
                CDi=cdi,
                CDp=None,
                CD_total=cdi,
                e=e_factor,
                AR=ar,
                Cl=float(p_res.Cl),
                Cm=float(p_res.Cm),
                Cn=float(p_res.Cn),
                CY=float(p_res.CY),
                CDi_nearfield=float(p_res.CD_pressure),
            )

            # Trust score evaluation
            all_notes = list(notes)
            if p_res.trust_note:
                all_notes.append(p_res.trust_note)
            conv_details = None
            if estimate_mesh_error:
                conv_report = _estimate_panel_mesh_error(ac, cond, settings)
                all_notes.extend(conv_report.notes)
                conv_details = conv_report.to_dict()

            main_surf_n_span = ac.surfaces[0].n_panels or settings.n_panels
            h_over_c = (h_min / c_ref) if (h_min is not None and c_ref) else None

            totals.trust = evaluate_aerodynamic_trust(
                AR=ar,
                condition=cond,
                spanwise_list=spanwise_list,
                CL=totals.CL,
                CDi=totals.CDi,
                CD_total=totals.CD_total,
                converged=p_res.kutta_converged,
                n_panels=int(main_surf_n_span),
                solver_type="panel3d",
                h_over_c=h_over_c,
                notes=all_notes,
                mesh_stats=mesh_stats,
                CD_nearfield=totals.CDi_nearfield,
            )

            result = SolverResult(
                spanwise=spanwise_list,
                totals=totals,
                solver_type="panel3d",
                converged=p_res.kutta_converged,
                iterations=p_res.kutta_iters,
                residual_history=[p_res.kutta_residual],
                condition=cond,
                execution_time=dt_per_case,
                symmetry_used=use_half,
                b_ref=float(b_ref),
                c_ref=float(c_ref),
            )

            result.details = {
                "cp": p_res.cp,
                "panel_cp": p_res.cp,
                "mesh": mesh,
                "panel_result": p_res,
                "mesh_stats": mesh_stats,
                "h_min": h_min,
                "notes": all_notes,
                "dispatch": dispatch,
            }
            if conv_details is not None:
                result.details["mesh_convergence"] = conv_details

            for msg in all_notes:
                warnings.warn(msg, RuntimeWarning, stacklevel=2)

            results.append(result)

        return results[0] if is_single else results

    def solve_sweep(
        self,
        aircraft: Aircraft | LiftingSurface,
        condition: FlightCondition,
        settings: SolverSettings,
        alpha_range: np.ndarray,
        ref_point: np.ndarray | None = None,
    ) -> list[SolverResult]:
        """Solve an angle-of-attack sweep.

        Parameters
        ----------
        aircraft : Aircraft or LiftingSurface
            Geometry definition.
        condition : FlightCondition
            Base free-stream condition.
        settings : SolverSettings
            Discretization and solution parameters.
        alpha_range : numpy.ndarray
            Array of angles of attack [rad].
        ref_point : numpy.ndarray or None, optional
            Moment reference center [m].

        Returns
        -------
        list of SolverResult
        """
        conds = [dataclasses.replace(condition, alpha=float(a)) for a in np.asarray(alpha_range, dtype=float)]
        res_list = self.solve(aircraft, conds, settings, ref_point=ref_point)
        return res_list if isinstance(res_list, list) else [res_list]
