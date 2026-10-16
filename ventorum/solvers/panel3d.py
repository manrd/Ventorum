# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""3D source-doublet panel method solver family integration (Ventorum-3DP).

This module connects the 3D panel method to the Ventorum solver architecture.
It takes an Aircraft, FlightCondition (or sweep), and SolverSettings, and
returns standard SolverResult objects with integrated totals, spanwise strip
loads, trust scoring, and execution dispatch records.

Settings
--------
The panel solver reads these fields of :class:`SolverSettings`:
``n_panels`` (spanwise panels per semi-span, as the lattice), ``n_chord``
(chordwise panels per surface, 20 when None), ``spacing``,
``proportional_panels``, ``min_panels``, ``use_symmetry``,
``panel_kutta``, ``panel_backend``, ``panel_wake_length``,
``panel_chord_spacing`` (cosine by default),
``panel_estimate_mesh_error`` and ``panel_scale_root_thickness`` (True
by default: the root section of a symmetric surface with dihedral keeps
its thickness normal to the local surface; see
:mod:`ventorum.panel3d.mesh`).

Backend selection
-----------------
``panel_backend="auto"`` follows the device and precision selection of
:mod:`ventorum.gpu`, as the lattice solvers do: the device setting
``"cpu"`` keeps the solve on the CPU; inside a CPU worker thread the
``"auto"`` device stays on the CPU; otherwise :func:`ventorum.gpu.use_gpu`
decides with the work of the dense systems (influence evaluations,
``N^2`` per system, twice in ground effect). The panel method has no cost
model of its own yet, so the decision uses the thresholds of the vortex
lattice. On the GPU the precision is :func:`ventorum.gpu.get_precision`.
On the CPU the numpy reference kernels are used.

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

#: Accepted values of the panel options of :class:`SolverSettings`.
_KUTTA_OPTIONS = ("morino", "pressure")
_BACKEND_OPTIONS = ("auto", "numpy", "numba", "gpu")
_CHORD_SPACINGS = ("cosine", "uniform")

#: Names of a per-panel strip index that a mesh may carry (attribute of
#: PanelMesh). When present and consistent with the lattice strips, it
#: replaces the geometric strip assignment.
_STRIP_METADATA_NAMES = ("strip", "strip_index")

#: Per-panel and per-node metadata of the mesh builder that the combined
#: mesh carries when the PanelMesh class has these fields.
_PANEL_METADATA = ("chord_index", "side")
_NODE_METADATA = ("node_station",)


def _estimate_panel_mesh_error(aircraft: Aircraft, condition: FlightCondition, settings: SolverSettings):
    """Run a three-level panel convergence study for the trust notes.

    Parameters
    ----------
    aircraft : Aircraft
        Case geometry.
    condition : FlightCondition
        Flight condition of the study.
    settings : SolverSettings
        Solver settings of the base (coarsest) mesh, the mesh of the user.

    Returns
    -------
    ConvergenceReport
        Values per level, observed order, Richardson value and the error
        estimate of the base mesh for CL, CDi and Cm.
    """
    from ventorum.panel3d.convergence import panel_mesh_convergence

    return panel_mesh_convergence(
        aircraft, condition, settings, levels=3, ratio=1.5, quantities=("CL", "CDi", "Cm")
    )


def _option(settings: SolverSettings, name: str, default, allowed: tuple[str, ...]) -> str:
    """Return a string option of the settings, checked against the allowed values."""
    value = getattr(settings, name, default)
    key = str(default if value is None else value).strip().lower()
    if key not in allowed:
        raise ValueError(f"settings.{name}={value!r} must be one of {allowed}.")
    return key


def _has_antisymmetric_control(aircraft: Aircraft) -> bool:
    """Return True if a surface has a deflected antisymmetric control (geometry not symmetric)."""
    for surf in aircraft.surfaces:
        for ctrl in getattr(surf, "controls", None) or ():
            if float(getattr(ctrl, "deflection", 0.0)) != 0.0 and not bool(getattr(ctrl, "symmetric", True)):
                return True
    return False


def _combined_metadata(subs: list[PanelMesh]) -> dict:
    """Return the builder metadata of the combined mesh of several surfaces.

    The strips are offset per surface, as :func:`ventorum.panel3d.mesh.mesh_aircraft`
    does; the other fields are concatenated. Only the fields of the
    PanelMesh class that all surface meshes carry are returned.
    """
    names = {f.name for f in dataclasses.fields(PanelMesh)}
    out: dict = {}
    if "tip" in names:
        out["tip"] = "flat"
    if "strip" in names and all(getattr(m, "strip", None) is not None for m in subs):
        parts = []
        n_strips = 0
        for m in subs:
            st = np.asarray(m.strip, dtype=int)
            parts.append(np.where(st >= 0, st + n_strips, -1))
            n_strips += int(np.max(st)) + 1 if np.any(st >= 0) else 0
        out["strip"] = np.concatenate(parts)
    for name in _PANEL_METADATA + _NODE_METADATA:
        if name in names and all(getattr(m, name, None) is not None for m in subs):
            out[name] = np.concatenate([np.asarray(getattr(m, name), dtype=int) for m in subs])
    return out


def _ground_of(condition: FlightCondition, ground: GroundPlane | None, ref_point: np.ndarray):
    """Return the ground plane of one condition (explicit plane, else condition.h), or None."""
    if ground is not None:
        return GroundPlane(normal=np.asarray(ground.normal, dtype=float), offset=float(ground.offset))
    gp = ground_plane_parameters(condition, ref_point)
    if gp is None:
        return None
    return GroundPlane(normal=np.asarray(gp[0], dtype=float), offset=float(gp[1]))


def _strip_metadata(mesh: PanelMesh) -> np.ndarray | None:
    """Return a per-panel strip index of a mesh, or None when the mesh has none."""
    for name in _STRIP_METADATA_NAMES:
        value = getattr(mesh, name, None)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.shape[0] == np.asarray(mesh.panels).shape[0]:
            return arr.astype(int)
    return None


def _assign_strips(
    centroids: np.ndarray,
    surf: LiftingSurface,
    eta_e: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Return the lattice strip index of each panel from its centroid.

    The strips of the lattice are bounded by the station planes at the
    spanwise stations *eta_e*. A station plane holds the station point of
    the reference line and is normal to the span direction in the y-z
    plane (the same plane as the mesh sections). A panel belongs to the
    strip whose station planes enclose its centroid. Panels in a station
    plane (tip caps, control side closures) go to the inboard strip, and a
    cap at the last station to the last strip.

    Parameters
    ----------
    centroids : numpy.ndarray
        Panel centroids of the right side of the surface with shape (n, 3)
        in m.
    surf : LiftingSurface
        Surface definition.
    eta_e : numpy.ndarray
        Spanwise stations of the lattice strips with shape (n_strips + 1,)
        [-].
    offset : numpy.ndarray
        Position offset of the surface with shape (3,) in m.

    Returns
    -------
    numpy.ndarray
        Strip index of each panel with shape (n,).
    """
    _, y_ref, z_ref = surface_reference_line(surf, eta_e)
    pts = np.column_stack([y_ref + offset[1], z_ref + offset[2]])
    n_st = pts.shape[0]
    axis = np.column_stack([np.gradient(y_ref, eta_e), np.gradient(z_ref, eta_e)])
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-300)
    if surf.is_symmetric:
        axis[np.asarray(eta_e) <= 1e-12] = np.array([1.0, 0.0])
    c_yz = np.asarray(centroids, dtype=float)[:, 1:3]
    tol = 1e-9 * max(1.0, float(surf.semi_span))
    inner = range(1, n_st - 1)
    strip = np.zeros(c_yz.shape[0], dtype=int)
    for j in inner:
        side = (c_yz - pts[j]) @ axis[j]
        strip += (side > tol).astype(int)
    return np.clip(strip, 0, max(n_st - 2, 0))


def _surface_strip_maps(
    sub: PanelMesh,
    surf: LiftingSurface,
    n_span: int,
    eta_e: np.ndarray,
    first_panel: int,
    half: bool,
) -> tuple[list[list[int]], list[list[int]] | None]:
    """Map the panels of one surface to the lattice strips.

    Parameters
    ----------
    sub : PanelMesh
        Mesh of the surface alone.
    surf : LiftingSurface
        Surface definition.
    n_span : int
        Spanwise strips on the semi-span [-].
    eta_e : numpy.ndarray
        Spanwise stations of the strips with shape (n_span + 1,) [-].
    first_panel : int
        Index of the first panel of the surface in the combined mesh.
    half : bool
        True if the surface is meshed as the right half only.

    Returns
    -------
    right_strip_panels : list of list of int
        Panel indices (combined mesh) of each strip on the right side, root
        to tip.
    left_strip_panels : list of list of int or None
        Panel indices of each strip on the left side, left tip to root, or
        None on a half mesh or for a surface without a mirror side.
    """
    nodes = np.asarray(sub.nodes, dtype=float)
    panels = np.asarray(sub.panels, dtype=int)
    centroids = np.mean(nodes[panels], axis=1)
    # Repeated corners of triangles: use the mean of the distinct corners.
    for i, row in enumerate(panels):
        uniq = list(dict.fromkeys(int(a) for a in row))
        if len(uniq) < 4:
            centroids[i] = np.mean(nodes[uniq], axis=0)
    offset = np.asarray(surf.position, dtype=float).reshape(3)
    symmetric = bool(surf.is_symmetric)
    mirror_side = symmetric and not half
    left = np.zeros(panels.shape[0], dtype=bool)
    c_right = centroids.copy()
    if mirror_side:
        # The mirror side of a symmetric surface lies at y < 0.
        left = centroids[:, 1] < 0.0
        c_right[left, 1] = -c_right[left, 1]
    elif bool(getattr(surf, "mirror_y", False)) and not symmetric:
        c_right[:, 1] = -c_right[:, 1]

    strip = _assign_strips(c_right, surf, eta_e, offset)
    meta = _strip_metadata(sub)
    if meta is not None:
        # Strip index of the mesh builder: a full symmetric surface numbers
        # its strips from the left tip (0) to the right tip (2 n - 1), a half
        # mesh or a one-sided surface from the root (0) to the tip (n - 1);
        # tip and side panels have -1. It is used when its strips are the
        # lattice strips (the stations of a control can add mesh strips).
        valid = meta >= 0
        n_meta = int(np.max(meta[valid])) + 1 if np.any(valid) else 0
        if mirror_side and n_meta == 2 * n_span:
            left = np.where(valid, meta < n_span, left)
            strip = np.where(
                valid, np.where(meta < n_span, n_span - 1 - meta, meta - n_span), strip
            )
        elif not mirror_side and n_meta == n_span:
            strip = np.where(valid, meta, strip)

    right: list[list[int]] = [[] for _ in range(n_span)]
    left_map: list[list[int]] = [[] for _ in range(n_span)]
    for i in range(panels.shape[0]):
        j = int(strip[i])
        if mirror_side and left[i]:
            left_map[n_span - 1 - j].append(first_panel + i)
        else:
            right[j].append(first_panel + i)
    return right, (left_map if mirror_side else None)


class PanelMethodSolver(BaseSolver):
    """Inviscid three-dimensional potential flow panel solver.

    Follows the BaseSolver interface for full integration with Aircraft,
    FlightCondition, SolverSettings, and SolverResult.
    """

    name: str = "panel3d"

    def _resolve_backend(
        self,
        settings: SolverSettings,
        n_panels: int = 0,
        n_configs: int = 1,
        in_ground: bool = False,
    ) -> tuple[str, str, str, str]:
        """Resolve the backend, device, precision and reason of one solve.

        Parameters
        ----------
        settings : SolverSettings
            Settings with ``panel_backend``.
        n_panels : int, default=0
            Number of solved panels [-].
        n_configs : int, default=1
            Number of systems to factorize (configurations) [-].
        in_ground : bool, default=False
            True when a condition is in ground effect.

        Returns
        -------
        tuple of str
            (backend, device, precision, reason): backend "numpy", "numba"
            or "gpu"; device "cpu" or "gpu"; precision "float32" or
            "float64"; one sentence that gives the reason.
        """
        choice = _option(settings, "panel_backend", "auto", _BACKEND_OPTIONS)
        if choice in ("numpy", "numba"):
            return choice, "cpu", "float64", f"panel_backend setting is {choice}"
        if choice == "gpu":
            return "gpu", "gpu", gpu.get_precision(), "panel_backend setting is gpu"
        if gpu.get_device() == "cpu":
            return "numpy", "cpu", "float64", gpu._REASON_DEVICE_CPU
        from ventorum.utils.parallel import _local as _par_local

        if gpu.get_device() == "auto" and getattr(_par_local, "worker_threads", None) is not None:
            return "numpy", "cpu", "float64", "called inside a CPU worker thread"
        n = int(n_panels)
        work = float(max(n_configs, 1)) * n * n * (2.0 if in_ground else 1.0)
        use, reason = gpu.use_gpu(work, "vlm", max(int(n_configs), 1), None, return_reason=True)
        if use:
            return "gpu", "gpu", gpu.get_precision(), reason
        return "numpy", "cpu", "float64", reason

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
            Ground plane of all conditions (normal and offset in m). If
            None, ``condition.h`` sets the ground, as for the vortex
            lattice.
        ref_point : numpy.ndarray or None, optional
            Moment reference point [m].

        Returns
        -------
        SolverResult or list of SolverResult
            ``details["cp"]`` holds the panel pressure coefficients of the
            solved mesh, ``details["dispatch"]`` the backend, device,
            precision and the counted factorizations, and
            ``details["panel"]`` the panel options used (Kutta option,
            chordwise spacing and count, wake length in m, half mesh,
            root-thickness scaling).

        Notes
        -----
        Mesh error estimate. ``settings.panel_estimate_mesh_error=True``
        runs a three-level panel mesh convergence study (ratio 1.5 in both
        directions) for each condition. The study adds the error estimate
        of the given mesh to the trust warnings and stores the report in
        ``details["mesh_convergence"]``. The dense LU work grows with the
        cube of the panel count, so the study costs about
        ``1 + 1.5**6 + 1.5**12`` (about 142) times one base solve; it is
        off by default.
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

        kutta = _option(settings, "panel_kutta", "morino", _KUTTA_OPTIONS)
        chord_spacing = _option(settings, "panel_chord_spacing", "cosine", _CHORD_SPACINGS)
        wake_length = getattr(settings, "panel_wake_length", None)
        if wake_length is not None:
            wake_length = float(wake_length)
            if not np.isfinite(wake_length) or wake_length <= 0.0:
                raise ValueError(f"settings.panel_wake_length={wake_length} must be positive.")
        n_chord = int(settings.n_chord) if getattr(settings, "n_chord", None) is not None else _DEFAULT_N_CHORD
        use_sym_setting = bool(getattr(settings, "use_symmetry", True))
        estimate_mesh_error = bool(getattr(settings, "panel_estimate_mesh_error", False))
        scale_root_thickness = bool(getattr(settings, "panel_scale_root_thickness", True))

        grounds = [_ground_of(c, ground, rp) for c in cond_list]
        all_surfs_sym = all(s.is_symmetric for s in ac.surfaces) and not _has_antisymmetric_control(ac)
        all_conds_sym = all(is_symmetric_condition(c, g) for c, g in zip(cond_list, grounds))
        use_half = bool(use_sym_setting and all_surfs_sym and all_conds_sym)

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
        sub_meshes: list[PanelMesh] = []

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
                scale_root_thickness=scale_root_thickness,
            )

            fr = np.array([float(s.y_frac) for s in _sorted_sections(surf)])
            spacing = resolve_spacing(surf.spacing if surf.spacing is not None else settings.spacing, surf, "vlm")
            eta_e, eta_mid = _surface_eta(surf, n_span, spacing)
            eta_e = np.asarray(eta_e, dtype=float)

            surface_strip_maps.append(
                _surface_strip_maps(sub, surf, int(n_span), eta_e, n_panels_total, use_half)
            )

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

            sub_meshes.append(sub)
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
            **_combined_metadata(sub_meshes),
        )

        # 3. Check mesh quality
        mesh_stats = check_mesh(mesh)

        # 4. Solve potential flow with PanelSolver
        ground_keys = {
            (tuple(np.round(g.normal, 12)), round(g.offset, 12), round(float(c.alpha), 12), round(float(c.beta), 12))
            for c, g in zip(cond_list, grounds)
            if g is not None
        }
        n_configs = len(ground_keys) + (1 if any(g is None for g in grounds) else 0)
        backend, device, precision, reason = self._resolve_backend(
            settings, n_panels_total, n_configs, bool(ground_keys)
        )
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
            dtype=precision if backend == "gpu" else None,
        )
        panel_results: list[PanelResult] = solver.solve_conditions(
            cond_list, grounds=None if ground is None else [ground] * len(cond_list)
        )

        dispatch = {
            "backend": backend,
            "device": device,
            "dtype": precision,
            "factorization_count": int(solver.factorization_count),
            "precision": precision,
            "backends": {"kernel": backend, "solve": "torch" if device == "gpu" else "scipy"},
            "threads": 1,
            "workers": 1,
            "plan_source": "default",
            "profile": None,
            "reason": reason,
        }

        panel_options = {
            "kutta": kutta,
            "chord_spacing": chord_spacing,
            "n_chord": int(n_chord),
            "wake_length": float(solver.wake_length),
            "half_mesh": bool(use_half),
            "scale_root_thickness": scale_root_thickness,
        }

        # 5. Build SolverResult per condition with strip integration
        x_hat = np.array([1.0, 0.0, 0.0])
        dt_per_case = (time.perf_counter() - t0) / max(1, len(cond_list))
        results: list[SolverResult] = []

        for cond, gnd, p_res in zip(cond_list, grounds, panel_results):
            alpha = float(cond.alpha)
            beta = float(cond.beta)
            v_inf = float(cond.V_inf)
            q_inf = 0.5 * float(cond.rho) * v_inf**2
            l_hat = lift_direction(alpha)
            d_hat = freestream_direction(alpha, beta)

            # Ground clearance (a ground strike raises in the solve)
            h_min = None
            if gnd is not None:
                h_min = check_ground_clearance(solver._full_mesh.nodes, gnd.normal, gnd.offset)

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
                "panel": panel_options,
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
