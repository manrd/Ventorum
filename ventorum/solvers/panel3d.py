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
(chordwise panels per surface, 40 when None), ``spacing``,
``proportional_panels``, ``min_panels``, ``use_symmetry``,
``panel_kutta``, ``panel_backend``, ``panel_ground_solver``, ``panel_wake_length``,
``panel_chord_spacing`` (cosine by default),
``panel_estimate_mesh_error``, ``panel_scale_root_thickness`` (True
by default: the root section of a symmetric surface with dihedral keeps
its thickness normal to the local surface; see
:mod:`ventorum.panel3d.mesh`) and ``panel_cache``.

Solver cache
------------
With ``panel_cache=True`` (default) a solve keeps its mesh, its mesh check
and its :class:`ventorum.panel3d.solver.PanelSolver` with the factorized
systems in the cache of :mod:`ventorum.panel3d.cache`. A later solve of the
same geometry with the same mesh and solver settings uses them again: a
new angle of attack, sideslip or body rate in free air needs no new
factorization. The results are the same bits as without the cache.
``ventorum.panel3d.clear_cache()`` frees the cache.

The mesh uses the defaults of :func:`ventorum.panel3d.mesh.mesh_lifting_surface`:
an open (blunt) trailing edge is closed, the side edges of a deflected
control are sealed transitions, and the fold guard keeps the root strip
of a wing with dihedral open. ``details["panel"]`` records these choices.

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

import copy
import dataclasses
import math
import time
import warnings
from collections.abc import Sequence

import numpy as np
import scipy.linalg
import scipy.sparse

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
from ventorum.core.trust import (
    PANEL3D_FAR_FIELD_FREE_AIR_NOTE,
    PANEL3D_FAR_FIELD_GROUND_NOTE,
    PANEL3D_OUTER_STRIP_GROUND_NOTE,
    PANEL3D_ROOT_SCALE_WARNING,
    evaluate_aerodynamic_trust,
)
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
    _check_no_node_displacements,
    check_mesh,
    mesh_lifting_surface,
)
from ventorum.panel3d import cache as panel_cache
from ventorum.panel3d import solver as panel_solver_module
from ventorum.panel3d.cache import CachedGeometry
from ventorum.panel3d.post import _condition_arrays
from ventorum.panel3d.solver import PanelResult, PanelSolver
from ventorum.solvers.base import BaseSolver
from ventorum.solvers.lattice_base import as_aircraft
from ventorum.utils.validation import validate_aircraft

__all__ = ["PanelMethodSolver"]

_TRUST_NOTES_NO_WARN = {
    PANEL3D_FAR_FIELD_GROUND_NOTE,
    PANEL3D_FAR_FIELD_FREE_AIR_NOTE,
    PANEL3D_OUTER_STRIP_GROUND_NOTE,
}

#: Default chordwise panel count per surface [-]. A mesh-error study of the
#: method chose the smallest count for which no test wing gets the mesh
#: warning of the trust score.
_DEFAULT_N_CHORD = 40

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


#: Module functions that build the mesh data of a solve. The objects
#: themselves are part of the key of the solver cache, so that code which
#: replaces one (for example a test that counts calls) never gets mesh data
#: built with the original. Add each new helper of :func:`_build_geometry`.
_GEOMETRY_HELPERS = (
    "_build_geometry", "mesh_lifting_surface", "check_mesh", "_surface_strip_maps", "_assign_strips",
    "_strip_metadata", "_combined_metadata", "_sorted_sections", "_surface_eta", "resolve_spacing",
    "surface_reference_line", "_AirfoilBlender", "airfoil_linear_properties",
)


def _geometry_helpers() -> tuple:
    """Return the helper objects of the mesh data (this module and the lattice module)."""
    from ventorum.geometry import lattice as lattice_module

    module = globals()
    return tuple(module.get(name) for name in _GEOMETRY_HELPERS) + (
        getattr(lattice_module, "compute_surface_n_panels", None),
    )


#: Functions whose output a cached solver keeps (geometry, wakes, gradient
#: operator, influence matrices, factorizations and unit solutions), per
#: module. As for the mesh helpers, the objects themselves enter the key of
#: the solver cache: code that replaces one (for example a test that counts
#: the factorizations) never gets a solver built with the original.
_SOLVER_HELPERS = {
    "ventorum.panel3d.solver": (
        "_standardize_panels", "_precompute_geometry", "largest_streamwise_chord", "build_wake",
        "reconstruct_full_mesh", "build_surface_gradient_operator", "mirror_panels_y", "reflect_panels_plane",
        "influence_matrices", "doublet_potential", "strip_influence_matrix", "_fold_wake", "surface_gradient",
        "_trefftz_factors",
    ),
    "scipy.linalg": ("lu_factor", "lu_solve"),
    "scipy.sparse.linalg": ("gmres",),
}
_SOLVER_HELPERS_NUMBA = {"ventorum.panel3d.kernels_numba": ("influence_matrices_numba",)}
_SOLVER_HELPERS_GPU = {
    "ventorum.panel3d.gpu": (
        "influence_matrices_device", "doublet_potential_device", "strip_sum", "lu_factor_device",
        "lu_solve_device",
    ),
}


def _solver_helpers(backend: str) -> tuple:
    """Return the helper objects of a solver with the given backend (see ``_SOLVER_HELPERS``)."""
    import importlib

    table = dict(_SOLVER_HELPERS)
    if backend == "numba":
        table.update(_SOLVER_HELPERS_NUMBA)
    elif backend == "gpu":
        table.update(_SOLVER_HELPERS_GPU)
    out = []
    for module_name, names in table.items():
        module = importlib.import_module(module_name)
        out.extend(getattr(module, name, None) for name in names)
    return tuple(out)


def _cuda_device_index(backend: str) -> int | None:
    """Return the current CUDA device index of the gpu backend, or None for a CPU backend."""
    if backend != "gpu":
        return None
    import torch

    return int(torch.cuda.current_device()) if torch.cuda.is_available() else None


def _thread_settings(backend: str) -> tuple:
    """Return the thread settings that can change the bits of a factorization.

    The Numba thread count of the calling thread (numba backend) and the
    BLAS limit of :func:`ventorum.utils.parallel.blas_single_thread` (CPU
    backends) enter the key of the solver cache, so that a cached solver
    has the bits of an uncached solve with the same thread settings.
    """
    if backend == "gpu":
        return (None, None)
    from ventorum.utils import parallel

    numba_threads = parallel.kernel_threads() if backend == "numba" else None
    return (numba_threads, parallel._blas_depth > 0)


def _estimate_panel_mesh_error(
    aircraft: Aircraft,
    condition: Sequence[FlightCondition],
    settings: SolverSettings,
    *,
    base_results: Sequence[SolverResult] | None,
    ground: GroundPlane | None = None,
    ref_point: np.ndarray | None = None,
):
    """Run a directional panel convergence study for the trust notes.

    The study refines the chordwise and the spanwise panel count
    separately (three levels each at ratio 1.5, the base mesh first) and
    adds the two directional error estimates (see
    :mod:`ventorum.panel3d.convergence`).

    Parameters
    ----------
    aircraft : Aircraft
        Case geometry.
    condition : sequence of FlightCondition
        Flight conditions of the study, solved together on each mesh.
    settings : SolverSettings
        Solver settings of the base (coarsest) mesh, the mesh of the user.
    base_results : sequence of SolverResult or None
        Factory results of the base mesh, in condition order, before the
        study adds its trust notes, or None to solve the base mesh again.
    ground : GroundPlane or None, optional
        Explicit ground plane (unit normal [-], offset [m]). None uses
        the ground from each condition.h.
    ref_point : numpy.ndarray or None, optional
        Moment reference point [m]. None uses the aircraft reference.

    Returns
    -------
    list of ConvergenceReport
        Per quantity (CL, CDi and Cm) the error estimate of the base mesh,
        with the values per level, the observed order and the Richardson
        value of each direction, one report per condition.
    """
    from ventorum.panel3d.convergence import panel_mesh_convergence

    return panel_mesh_convergence(
        aircraft, condition, settings, levels=3, ratio=1.5, quantities=("CL", "CDi", "Cm"),
        base_results=base_results, ground=ground, ref_point=ref_point,
    )


def _auto_cpu_backend() -> tuple[str, str]:
    """Return the CPU kernel backend of ``panel_backend="auto"`` and a reason suffix.

    The compiled Numba kernels agree with the numpy reference kernels to
    round-off and run on all kernel threads, so "auto" uses them on the
    CPU. Without Numba it uses the numpy reference kernels.

    Returns
    -------
    tuple of str
        ("numba", "; CPU kernels: numba") or ("numpy", "; CPU kernels:
        numpy (Numba is not available)").
    """
    try:
        import numba  # noqa: F401
    except ImportError:
        return "numpy", "; CPU kernels: numpy (Numba is not available)"
    return "numba", "; CPU kernels: numba"


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
    if "root_scale_warning" in names:
        out["root_scale_warning"] = any(getattr(m, "root_scale_warning", False) for m in subs)
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


def _spanwise_results_batch(solver, panel_results, conditions, strip_maps, metadata):
    """Integrate all strips and conditions with one shared incidence matrix.

    Solver geometry is in m and m^2. Pressures are dimensionless. Return
    per-condition SpanwiseResult objects with their documented SI units.
    Quarter-chord arms are formed before reduction, as in the scalar path.
    """
    maps, quarter_chords, surfaces = [], [], []
    for (right, left), meta in zip(strip_maps, metadata):
        first = len(maps)
        qc = meta["pos_mid"].copy()
        qc[:, 0] += 0.25 * meta["chord"]
        maps.extend(right)
        quarter_chords.extend(qc)
        if left is not None:
            qc_left = qc[::-1].copy()
            qc_left[:, 1] *= -1
            maps.extend(left)
            quarter_chords.extend(qc_left)
        surfaces.append((first, len(right), left is not None))
    lengths = np.fromiter(map(len, maps), dtype=int)
    row = np.repeat(np.arange(len(maps)), lengths)
    col = np.concatenate([np.asarray(ids, dtype=int) for ids in maps])
    incidence = scipy.sparse.csr_matrix(
        (np.ones(len(col)), (row, col)), shape=(len(maps), len(solver.areas)),
    )
    owner = np.zeros(len(solver.areas), dtype=int)
    owner[col] = row
    arm = solver.centroids - np.asarray(quarter_chords)[owner]
    out = [[] for _ in conditions]
    size = max(1, panel_solver_module._CONDITION_BATCH_BYTES // solver._condition_bytes())
    for start in range(0, len(conditions), size):
        chunk = conditions[start:start + size]
        _, speed, rho, lift_axis, drag_axis = _condition_arrays(chunk)
        q = 0.5 * rho * speed**2
        cp = np.array([r.cp for r in panel_results[start:start + size]])
        force = -cp[..., None] * solver.areas[:, None] * solver.normals * q[:, None, None]
        moment = np.cross(arm, force)

        def reduce_panels(values):
            flat = values.transpose(1, 0, 2).reshape(len(solver.areas), -1)
            return (incidence @ flat).reshape(len(maps), values.shape[0], 3).transpose(1, 0, 2)

        strip_force = reduce_panels(force)
        strip_moment = reduce_panels(moment)
        lift = np.sum(strip_force * lift_axis[:, None, :], axis=-1)
        drag = np.sum(strip_force * drag_axis[:, None, :], axis=-1)
        alpha = np.array([c.alpha for c in chunk])[:, None]
        for meta, (first, n, has_left) in zip(metadata, surfaces):
            stop = first + n * (2 if has_left else 1)
            ids = np.arange(first, stop)
            area, chord, width = meta["area"], meta["chord"], meta["width"]
            a0, alpha0 = meta["a0"], meta["alpha_L0"]
            y = meta["pos_mid"][:, 1]
            if has_left:
                # The incidence rows are right then left. Public strips
                # are left tip to right tip.
                ids = np.concatenate([ids[n:], ids[:n]])
                area = np.concatenate([area[::-1], area])
                chord = np.concatenate([chord[::-1], chord])
                width = np.concatenate([width[::-1], width])
                a0 = np.concatenate([a0[::-1], a0])
                alpha0 = np.concatenate([alpha0[::-1], alpha0])
                y = np.concatenate([-y[::-1], y])
            cl = lift[:, ids] / np.maximum(q[:, None] * area, 1e-300)
            cdi = drag[:, ids] / np.maximum(q[:, None] * area, 1e-300)
            cm = strip_moment[:, ids, 1] / np.maximum(q[:, None] * area * chord, 1e-300)
            local = lift[:, ids] / np.maximum(width, 1e-300)
            gamma = local / np.maximum(rho * speed, 1e-300)[:, None]
            effective = alpha0 + cl / np.maximum(a0, 1e-9)
            induced = alpha - effective
            arrays = dict(Cl=cl, Cd_i=cdi, Cm_section=cm, local_lift=local,
                          gamma=gamma, alpha_eff=effective, alpha_i=induced)
            if meta["is_symmetric"] and not has_left:
                arrays = {key: np.concatenate([a[:, ::-1], a], axis=1)
                          for key, a in arrays.items()}
                y = np.concatenate([-y[::-1], y])
                chord = np.concatenate([chord[::-1], chord])
            for i in range(len(chunk)):
                out[start + i].append(SpanwiseResult(
                    y=y, chord=chord, surface_name=meta["name"],
                    **{key: a[i] for key, a in arrays.items()},
                ))
    return out


def _spanwise_reference(solver, p_res, cond, surface_strip_maps, surface_geom_meta):
    """Retain the scalar strip integration for parity tests (SI units)."""
    alpha, beta, v_inf = float(cond.alpha), float(cond.beta), float(cond.V_inf)
    q_inf = 0.5 * float(cond.rho) * v_inf**2
    l_hat, d_hat = lift_direction(alpha), freestream_direction(alpha, beta)
    x_hat = np.array([1.0, 0.0, 0.0])
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

    return spanwise_list


def _build_geometry(
    ac: Aircraft,
    settings: SolverSettings,
    section_coords_by_surface: list[list[np.ndarray]],
    n_chord: int,
    chord_spacing: str,
    use_half: bool,
    scale_root_thickness: bool,
) -> CachedGeometry:
    """Build the panel mesh, the strip maps and the strip data, and check the mesh.

    The data depend only on the geometry, the mesh settings and the half-mesh
    flag, so the solver cache keeps them with the solvers (see
    :mod:`ventorum.panel3d.cache`).

    Parameters
    ----------
    ac : Aircraft
        Case geometry.
    settings : SolverSettings
        Settings with the spanwise mesh fields (``n_panels``, ``spacing``,
        ``proportional_panels``, ``min_panels``).
    section_coords_by_surface : list of list of numpy.ndarray
        Unit-chord coordinates of each section of each surface [-].
    n_chord : int
        Chordwise panels per surface [-].
    chord_spacing : str
        Chordwise panel distribution ("cosine" or "uniform").
    use_half : bool
        True to mesh the right half only (symmetry plane y = 0).
    scale_root_thickness : bool
        Root-thickness scaling of the mesh (see
        :func:`ventorum.panel3d.mesh.mesh_lifting_surface`).

    Returns
    -------
    CachedGeometry
        Mesh, strip maps, strip data (positions in m, chords, widths in m,
        areas in m^2, lift slopes in 1/rad, zero-lift angles in rad), the
        number of solved panels and the mesh statistics.
    """
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
            spacing=settings.spacing,
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
    mesh_stats["root_scale_warning"] = bool(getattr(mesh, "root_scale_warning", False))
    return CachedGeometry(
        mesh=mesh,
        strip_maps=surface_strip_maps,
        geom_meta=surface_geom_meta,
        n_panels=n_panels_total,
        mesh_stats=mesh_stats,
    )


def _morino_section_solution(polygon, ground=None):
    """Solve a unit-chord polygon for the two unit onsets.

    The constant source/doublet Dirichlet method uses the two-dimensional
    logarithmic Green function and its normal derivative. The wake is a
    semi-infinite doublet sheet along +x (along the ground line with a
    ground). Its strength is the upper minus lower trailing-edge doublet.
    See Katz and Plotkin (2001), chapters 11 and 12, and Morino and Kuo
    (1974). Coordinates are in chord units in upper-TE, LE, lower-TE
    order. Two unit onsets share one factorization. An optional ground
    line has a unit normal and an offset in chord units. Equal-strength
    source and reflected doublet images impose zero normal velocity there
    (Katz and Plotkin, method of images).

    Returns
    -------
    tuple
        The lift coefficient [-] as a function of the onset angle [rad]
        from the chord, and coefficients [-] of
        Cm = c0 + cxx cos(alpha)^2 + cxz cos(alpha) sin(alpha)
        + czz sin(alpha)^2 about the clean quarter chord. With a ground,
        the onset of a physical solution is parallel to the ground line.
    """
    points = np.asarray(polygon, dtype=float)
    edges = np.diff(points, axis=0)
    length = np.linalg.norm(edges, axis=1)
    tangent = edges / length[:, None]
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    center = 0.5 * (points[:-1] + points[1:])
    def matrices(vertices, tangents, normals):
        rel = center[:, None, :] - vertices[None, :-1, :]
        x = np.sum(rel * tangents, axis=-1)
        h = np.sum(rel * normals, axis=-1)
        ah = np.abs(h)

        def primitive(u):
            return 0.5 * u * np.log(np.maximum(u * u + h * h, 1e-300)) - u + ah * np.arctan2(u, ah)

        b = (primitive(x) - primitive(x - length)) / (2 * np.pi)
        c = -np.sign(h) * (np.arctan2(x, ah) - np.arctan2(x - length, ah)) / (2 * np.pi)
        return b, c

    def wake_column(origin, direction, wake_normal):
        relative = center - origin
        x, h = relative @ direction, relative @ wake_normal
        return -np.sign(h) * (np.arctan2(x, np.abs(h)) + np.pi / 2) / (2 * np.pi)

    b, c = matrices(points, tangent, normal)
    np.fill_diagonal(c, 0.5)
    direction = np.array([1.0, 0.0])
    if ground is not None:
        ng, offset = ground
        reflection = np.eye(2) - 2 * np.outer(ng, ng)
        image = points @ reflection + 2 * offset * ng
        bi, ci = matrices(image, tangent @ reflection, normal @ reflection)
        b += bi
        c += ci
        direction -= (direction @ ng) * ng
        direction /= np.linalg.norm(direction)
    wake_normal = np.array([-direction[1], direction[0]])
    wake = wake_column(points[0], direction, wake_normal)
    if ground is not None:
        wake += wake_column(image[0], direction @ reflection, wake_normal @ reflection)
    c[:, 0] += wake
    c[:, -1] -= wake
    mu = scipy.linalg.lu_solve(scipy.linalg.lu_factor(c), b @ normal)
    # Interior nodes use inverse-distance interpolation, as the panel
    # surface gradient. The trailing edge cuts the upper/lower connection.
    node = (length[1:, None] * mu[:-1] + length[:-1, None] * mu[1:]) / (
        length[:-1] + length[1:]
    )[:, None]
    gradient = np.empty_like(mu)
    gradient[1:-1] = (node[1:] - node[:-1]) / length[1:-1, None]
    gradient[0] = (mu[1] - mu[0]) / np.dot(center[1] - center[0], tangent[0])
    gradient[-1] = (mu[-1] - mu[-2]) / np.dot(center[-1] - center[-2], tangent[-1])
    velocity = tangent - gradient

    def lift(angle):
        direction = np.array([np.cos(angle), np.sin(angle)])
        cp = 1 - (velocity @ direction)**2
        force = -np.sum(cp[:, None] * length[:, None] * normal, axis=0)
        return force @ np.array([-np.sin(angle), np.cos(angle)])

    # Positive pitch is z Fx - x Fz in the geometry x-z plane.
    arm = center - np.array([0.25, 0.0])
    weight = -length * (arm[:, 1] * normal[:, 0] - arm[:, 0] * normal[:, 1])
    moment = np.array([np.sum(weight), -np.sum(weight * velocity[:, 0]**2),
                       -2 * np.sum(weight * velocity[:, 0] * velocity[:, 1]),
                       -np.sum(weight * velocity[:, 1]**2)])
    return lift, moment


def _section_moment(moment, angle):
    """Return Cm [-] of a moment quadratic at the onset angle *angle* [rad]."""
    co, si = np.cos(angle), np.sin(angle)
    return moment[0] + moment[1] * co**2 + moment[2] * co * si + moment[3] * si**2


def _morino_section_line(polygon, ground=None):
    """Return the lift line and moment quadratic of a unit-chord polygon.

    The line comes from the onsets 0 and 1 deg of one factorization of
    :func:`_morino_section_solution`. With a fixed ground line these
    onsets cross the ground unless the ground is parallel to the chord;
    :func:`_morino_section_ground_line` gives the reference of a section
    in ground effect.

    Returns
    -------
    tuple
        Lift slope [1/rad], zero-lift angle [rad], and coefficients [-]
        of Cm = c0 + cxx cos(alpha)^2 + cxz cos(alpha) sin(alpha)
        + czz sin(alpha)^2. The moment reference is the clean quarter chord.
    """
    lift, moment = _morino_section_solution(polygon, ground)
    step = np.radians(1.0)
    cl0 = lift(0.0)
    slope = (lift(step) - cl0) / step
    return float(slope), float(-cl0 / slope), moment


#: Half step [rad] of the central difference of the ground section line.
GROUND_SECTION_STEP = np.radians(0.5)


def _morino_section_ground_line(polygon, ground, step=GROUND_SECTION_STEP):
    """Return the lift line and moment quadratic of a section in ground effect.

    The ground line fixes the attitude of the section: the onset is
    parallel to the ground, at the angle ``theta = atan2(-n_x, n_z)`` [rad]
    from the chord, where ``n`` is the unit ground normal in the chord
    frame. The quarter-chord height ``h`` [chord] above the ground stays
    fixed. The section pitches about its quarter chord to the attitudes
    ``theta - step``, ``theta`` and ``theta + step``. Each attitude has its
    own ground line and an onset parallel to it, so each needs one
    factorization. This is the pitch curve ``Cl(alpha; h)`` of a 2D polar
    with a height axis. The line is its tangent at the attitude: the slope
    ``(Cl(theta + step) - Cl(theta - step)) / (2 step)`` and the zero-lift
    angle ``theta - Cl(theta) / slope``. The moment quadratic
    ``A + B cos(2 alpha) + C sin(2 alpha)`` goes through the three
    pitch-curve moments.

    Parameters
    ----------
    polygon : numpy.ndarray
        Section coordinates in chord units, upper-TE, LE, lower-TE order.
    ground : tuple
        Unit ground normal, shape (2,), and offset [chord] of the ground
        line ``n . x = offset``.
    step : float
        Half step of the central difference [rad].

    Returns
    -------
    tuple
        Lift slope [1/rad], zero-lift angle [rad], and the moment
        coefficients [-] as in :func:`_morino_section_line`.
    """
    normal, offset = ground
    normal = np.asarray(normal, dtype=float)
    attitude = float(np.arctan2(-normal[0], normal[1]))
    quarter = np.array([0.25, 0.0])
    height = float(quarter @ normal - offset)
    angles = attitude + np.array([-step, 0.0, step])
    lifts, moments = [], []
    for angle in angles:
        tilted = np.array([-np.sin(angle), np.cos(angle)])
        lift, moment = _morino_section_solution(polygon, (tilted, float(quarter @ tilted) - height))
        lifts.append(lift(angle))
        moments.append(_section_moment(moment, angle))
    slope = (lifts[2] - lifts[0]) / (2.0 * step)
    fit = np.linalg.solve(np.column_stack([np.ones(3), np.cos(2 * angles), np.sin(2 * angles)]),
                          np.asarray(moments))
    # A + B cos(2a) + C sin(2a) = A + B cos^2 + 2 C cos sin - B sin^2.
    moment = np.array([fit[0], fit[1], 2.0 * fit[2], -fit[1]])
    return float(slope), float(attitude - lifts[1] / slope), moment


def _panel_section_lines(mesh, maps, lattice, strips, ground=None, with_images=None):
    """Solve the actual midspan section polygons of reporting strips in 2D.

    Panel metadata selects the upper and lower chordwise segments. The
    intersection with the reporting section plane gives its polygon. Coordinates
    are projected into the clean section frame and divided by chord [m].
    A strip with images uses the ground line of its section plane. Its line
    is the tangent of the pitch curve at the attitude of the strip (onset
    parallel to the ground, see :func:`_morino_section_ground_line`).
    Return slopes [1/rad], zero-lift angles [rad], and moment quadratics [-].
    """
    slopes, zeros, moments = [], [], []
    memo = {}
    for index, (ids, strip) in enumerate(zip(maps, strips)):
        ids = np.asarray(ids, dtype=int)
        skin = ids[np.asarray(mesh.side)[ids] != 0]
        rows = np.unique(np.asarray(mesh.strip)[skin])
        target = lattice.qc_mid[strip]
        axis = np.cross(lattice.normal[strip], lattice.chord_dir[strip])
        distances = []
        for row in rows:
            vertices = mesh.nodes[mesh.panels[skin[mesh.strip[skin] == row]]]
            span = (vertices - target) @ axis
            distances.append(max(float(np.min(span)), -float(np.max(span)), 0.0))
        row = rows[np.argmin(distances)]
        skin = skin[mesh.strip[skin] == row]
        chains = []
        for side in (1, -1):
            panels = skin[mesh.side[skin] == side]
            panels = panels[np.argsort(mesh.chord_index[panels])]
            chain = []
            for pid in panels:
                vertices = mesh.panels[pid]
                stations = mesh.node_station[vertices]
                segments = []
                for station in np.unique(stations):
                    p = mesh.nodes[vertices[stations == station]]
                    p = p[np.argsort(p @ lattice.chord_dir[strip])]
                    segments.append(p)
                mids = np.mean(segments, axis=1)
                fraction = np.dot(target - mids[0], axis) / np.dot(mids[1] - mids[0], axis)
                segment = segments[0] + fraction * (segments[1] - segments[0])
                chain.append(segment[0])
            chain.append(segment[-1])
            chains.append(np.asarray(chain))
        polygon = np.vstack([chains[0][::-1], chains[1][1:]])
        origin = 0.5 * (lattice.le_left[strip] + lattice.le_right[strip])
        relative = (polygon - origin) / lattice.chord[strip]
        coords = np.column_stack([relative @ lattice.chord_dir[strip], relative @ lattice.normal[strip]])
        section_ground = None
        if ground is not None and with_images[index]:
            ng = np.array([ground.normal @ lattice.chord_dir[strip], ground.normal @ lattice.normal[strip]])
            norm = np.linalg.norm(ng)
            if norm < 1e-12:
                raise ValueError("The ground plane has no normal in this 2D section plane.")
            offset = (ground.offset - ground.normal @ origin) / (lattice.chord[strip] * norm)
            section_ground = ng / norm, offset
        key = (np.round(coords, 12).tobytes(), None if section_ground is None
               else tuple(np.round(np.r_[section_ground[0], section_ground[1]], 12)))
        if key not in memo:
            memo[key] = (_morino_section_line(coords) if section_ground is None
                         else _morino_section_ground_line(coords, section_ground))
        a, a0, cm = memo[key]
        slopes.append(a)
        zeros.append(a0)
        moments.append(cm)
    return np.asarray(slopes), np.asarray(zeros), np.asarray(moments)


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
        cpu_backend, cpu_note = _auto_cpu_backend()
        if gpu.get_device() == "cpu":
            return cpu_backend, "cpu", "float64", gpu._REASON_DEVICE_CPU + cpu_note
        from ventorum.utils.parallel import _local as _par_local

        if gpu.get_device() == "auto" and getattr(_par_local, "worker_threads", None) is not None:
            # A worker thread keeps its own Numba thread count (set when the
            # worker starts), so the compiled kernels use the share of that worker.
            return cpu_backend, "cpu", "float64", "called inside a CPU worker thread" + cpu_note
        n = int(n_panels)
        # Each ground configuration needs its own assembly and factorization.
        # Count it once, as in the cold single-configuration panel cost model.
        work = float(max(n_configs, 1)) * n * n
        use, reason = gpu.use_gpu(work, "panel3d", max(int(n_configs), 1), n, return_reason=True)
        if use:
            return "gpu", "gpu", gpu.get_precision(), reason
        return cpu_backend, "cpu", "float64", reason + cpu_note

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
            Discretization and solution knobs. The chordwise count
            ``n_chord`` [-] (40 when None) also goes to the trust score:
            it sets the chordwise rule of
            :func:`ventorum.core.trust.panel3d_mesh_checks` and, in ground
            effect, the rule for chordwise panels longer than the gap.
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
            precision, the factorizations of this call
            (``factorization_count``) and whether the call used a cached
            solver (``cached_solver``), and
            ``details["panel"]`` the panel options used (Kutta option,
            chordwise spacing and count, wake length in m, half mesh,
            root-thickness scaling, trailing-edge closure, control side
            edges and fold guard of the mesh). With the solver cache
            the arrays of ``details["mesh"]`` are read-only when a cached
            solver uses that mesh; the other arrays belong to the call.

        Notes
        -----
        Mesh error estimate. ``settings.panel_estimate_mesh_error=True``
        runs a panel mesh convergence study for each condition: a
        chordwise and a spanwise refinement of three levels each (ratio
        1.5, the given mesh first), each with a Richardson extrapolation,
        added with a separable error model (the error estimate of the
        given mesh is the sum of the two directional estimates; see
        :mod:`ventorum.panel3d.convergence`). The study adds the error
        estimate of the given mesh to the trust warnings and stores the
        report in ``details["mesh_convergence"]``. It solves five meshes;
        the dense LU work grows with the cube of the panel count, so the
        study costs about ``1 + 2 (1.5**3 + 1.5**6)`` (about 30) times one
        base solve. It is off by default.
        """
        is_single = isinstance(condition, FlightCondition)
        cond_list: list[FlightCondition] = [condition] if is_single else list(condition)

        ac = as_aircraft(aircraft)
        validate_aircraft(ac)
        for surf in ac.surfaces:
            _check_no_node_displacements(surf)
        ac.compute_reference_values()

        s_ref = ac.S_ref
        b_ref = ac.b_ref
        c_ref = ac.c_ref
        rp = ac.moment_reference() if ref_point is None else np.asarray(ref_point, dtype=float)

        kutta = _option(settings, "panel_kutta", "morino", _KUTTA_OPTIONS)
        ground_solver = _option(settings, "panel_ground_solver", "auto", ("auto", "direct", "krylov"))
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

        # 2. Mesh, strip data and mesh check (from the solver cache when it has them)
        use_cache = bool(getattr(settings, "panel_cache", True))
        coupling = getattr(self, "_viscous_coupling", None)
        if coupling is not None:
            use_cache = False
        geom_key = None
        geometry = None
        if use_cache:
            mesh_settings = (
                int(n_chord), chord_spacing, settings.n_panels, settings.spacing,
                bool(getattr(settings, "proportional_panels", False)), getattr(settings, "min_panels", 8),
                scale_root_thickness, use_half,
            )
            geom_key = panel_cache.geometry_key(ac, section_coords_by_surface, mesh_settings, _geometry_helpers())
            geometry = panel_cache.find_geometry(geom_key)
        if geometry is None:
            geometry = _build_geometry(
                ac, settings, section_coords_by_surface, n_chord, chord_spacing, use_half, scale_root_thickness
            )
        mesh = geometry.mesh
        n_panels_total = geometry.n_panels
        surface_strip_maps = geometry.strip_maps
        # Each call gets its own strip data and mesh statistics: the results
        # hold these arrays, and the cached ones must not change.
        surface_geom_meta = [
            {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in meta.items()}
            for meta in geometry.geom_meta
        ]
        mesh_stats = copy.deepcopy(geometry.mesh_stats) if use_cache else geometry.mesh_stats

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
        solver_class = PanelSolver

        def build_solver():
            panel_solver = solver_class(
                mesh,
                wake_length=wake_length,
                S_ref=s_ref,
                b_ref=b_ref,
                c_ref=c_ref,
                ref_point=rp,
                kutta=kutta,
                backend=backend,
                ground_solver=ground_solver,
                dtype=precision if backend == "gpu" else None,
            )
            if coupling is not None:
                coupling.prepare(panel_solver, geometry, ac, settings, cond_list)
            return panel_solver

        key = None
        if use_cache:
            key = panel_cache.solver_key(geom_key, (
                solver_class, wake_length, kutta, backend, _cuda_device_index(backend), precision, ground_solver,
                float(s_ref), float(b_ref), float(c_ref), np.asarray(rp, dtype=float).tobytes(),
            ) + _thread_settings(backend) + _solver_helpers(backend))
        # A solver in use by another thread is not shared: acquire builds a
        # second one for this call (see ventorum.panel3d.cache).
        lease = panel_cache.acquire(key, geom_key, geometry, build_solver, gpu=backend == "gpu")
        solver = lease.solver
        factorizations_before = int(solver.factorization_count)
        solve_ok = False
        try:
            panel_results: list[PanelResult] = solver.solve_conditions(
                cond_list, grounds=None if ground is None else [ground] * len(cond_list)
            )

            dispatch = {
                "backend": backend,
                "device": device,
                "dtype": precision,
                "factorization_count": int(solver.factorization_count) - factorizations_before,
                "factors_dropped": int(solver._factors_dropped),
                "configurations_evicted": int(solver._configurations_evicted),
                "cached_solver": bool(lease.cached),
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
                "trailing_edge": "close",
                "control_edge": "transition",
                "fold_guard": True,
            }

            # Largest strip chord over the reference chord [-] (ground rule of the trust).
            max_chord = max((float(np.max(m["chord"])) for m in surface_geom_meta if np.size(m["chord"])), default=0.0)
            max_chord_over_c = max_chord / float(c_ref) if (c_ref and max_chord > 0.0) else 1.0

            # 5. Build SolverResult per condition with strip integration
            dt_per_case = (time.perf_counter() - t0) / max(1, len(cond_list))
            spanwise_batches = _spanwise_results_batch(
                solver, panel_results, cond_list, surface_strip_maps, surface_geom_meta,
            )
            clearance = {}
            for gnd in grounds:
                if gnd is not None:
                    key = (tuple(gnd.normal), gnd.offset)
                    if key not in clearance:
                        clearance[key] = check_ground_clearance(
                            solver._full_mesh.nodes, gnd.normal, gnd.offset,
                        )
            results: list[SolverResult] = []

            for case_index, (cond, gnd, p_res) in enumerate(zip(cond_list, grounds, panel_results)):

                # Ground clearance (a ground strike raises in the solve)
                h_min = None
                if gnd is not None:
                    h_min = clearance[(tuple(gnd.normal), gnd.offset)]

                spanwise_list = spanwise_batches[case_index]

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
                main_surf_n_span = ac.surfaces[0].n_panels or settings.n_panels
                far_ratio_active = solver.far_ratio
                if math.isfinite(far_ratio_active):
                    if gnd is not None:
                        all_notes.append(PANEL3D_FAR_FIELD_GROUND_NOTE)
                    elif main_surf_n_span > 80:
                        all_notes.append(PANEL3D_FAR_FIELD_FREE_AIR_NOTE)

                if gnd is not None and getattr(mesh, "tip", "flat") != "none":
                    all_notes.append(PANEL3D_OUTER_STRIP_GROUND_NOTE)

                if getattr(mesh, "root_scale_warning", False):
                    all_notes.append(PANEL3D_ROOT_SCALE_WARNING)
                h_over_c = (h_min / c_ref) if (h_min is not None and c_ref) else None

                # The chordwise count turns on the chordwise rule of the mesh
                # checks and the ground rule "chordwise panels longer than the
                # gap"; the largest strip chord scales that ground rule, as for
                # the vortex lattice.
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
                    n_chord=int(n_chord),
                    notes=all_notes,
                    max_chord_over_c=max_chord_over_c,
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
                    "dispatch": dict(dispatch),
                    "panel": panel_options,
                }
                config_solver = solver
                config_key, wake_dir, gp = solver._config_key(cond, gnd)
                if solver.mesh.half and not solver._symmetric_config(wake_dir, gp):
                    config_solver = solver._full_solver
                cfg = config_solver._configs[config_key]
                result.details["dispatch"].update({
                    "ground_solver": cfg.ground_solver,
                    "ground_solver_reason": cfg.ground_solver_reason,
                    "iterations": dict(cfg.iterations),
                    "matrix_passes": cfg.matrix_passes,
                    "wake_shared": cfg.wake_shared,
                    "relative_residuals": dict(cfg.relative_residuals),
                    "fallbacks": copy.deepcopy(cfg.fallbacks),
                })
                if coupling is not None:
                    coupling.finish_result(result, solver, geometry, ac)
                for msg in all_notes:
                    if msg in _TRUST_NOTES_NO_WARN:
                        continue
                    if msg == PANEL3D_ROOT_SCALE_WARNING:
                        continue
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)

                results.append(result)
            solve_ok = True
        finally:
            panel_cache.release(lease, solve_ok)

        if estimate_mesh_error:
            # Keep the default base reuse and pass explicit inputs to every level.
            reuse = ground is None and ref_point is None
            estimate_inputs = {}
            if ground is not None:
                estimate_inputs["ground"] = ground
            if ref_point is not None:
                estimate_inputs["ref_point"] = ref_point
            conv_reports = _estimate_panel_mesh_error(
                ac, cond_list, settings, base_results=results if reuse else None, **estimate_inputs,
            )
            for result, conv_report in zip(results, conv_reports):
                result.details["mesh_convergence"] = conv_report.to_dict()
                result.details["notes"].extend(conv_report.notes)
                # Study notes add warnings only, as in evaluate_aerodynamic_trust.
                trust = result.totals.trust
                for msg in conv_report.notes:
                    if msg not in trust.warnings:
                        trust.warnings.append(msg)
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)

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
