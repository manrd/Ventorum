# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Deformed geometry utilities and node displacement helpers."""

from __future__ import annotations

import numpy as np

from ventorum.core.datatypes import Aircraft, NodeDisplacements, SolverSettings
from ventorum.geometry.lattice import (
    _rotate,
    _surface_mesh_eta,
    surface_edge_geometry,
)

#: Collocation rule of each canonical solver name that accepts node displacements.
_COLLOCATION = {"vlm": "vlm", "linear": "llt", "nonlinear": "llt"}


def undeformed_nodes(
    aircraft: Aircraft,
    settings: SolverSettings | None = None,
    *,
    solver: str = "vlm",
) -> dict[str, dict[str, np.ndarray]]:
    """Return undeformed lattice node coordinates and strip edges for each surface.

    The strip edges come from the same function that
    :func:`ventorum.geometry.lattice.build_lattice` uses, so they are the
    edges of the mesh that the solver builds. Use the same settings and the
    same solver here and for the analysis.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft model. Surface names must be unique.
    settings : SolverSettings or None
        Discretisation settings. Defaults are used if None.
    solver : str
        Solver name. All the names that ``SolverSettings.solver_type``
        accepts are valid, except the Fourier solver. ``"vlm"`` uses the
        vortex-lattice collocation; ``"linear"`` and ``"nonlinear"`` use the
        lifting-line collocation (the ``"auto"`` spacing depends on it).

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Mapping of surface name to a dict with keys 'eta' (n_edges,) [-],
        'le' (n_edges, 3) [m], and 'te' (n_edges, 3) [m].

    Raises
    ------
    ValueError
        If surface names are not unique, if the solver name is unknown, or
        if the solver is the Fourier solver.
    """
    from ventorum.solvers.factory import resolve_solver_type

    canonical = resolve_solver_type(solver)
    if canonical not in _COLLOCATION:
        raise ValueError(
            "The Fourier solver does not accept node displacements; use 'vlm', 'linear' or 'nonlinear'."
        )
    collocation = _COLLOCATION[canonical]

    names = [s.name for s in aircraft.surfaces]
    if len(names) != len(set(names)):
        raise ValueError(f"Surface names must be unique; found duplicate names in {names}.")

    st = settings or SolverSettings()
    ref_semi = max((s.semi_span for s in aircraft.surfaces), default=1.0)
    out: dict[str, dict[str, np.ndarray]] = {}

    for surf in aircraft.surfaces:
        eta_e, _ = _surface_mesh_eta(surf, st, collocation, ref_semi)
        geo = surface_edge_geometry(surf, eta_e)
        out[surf.name] = {
            "eta": np.asarray(eta_e, dtype=float).copy(),
            "le": np.asarray(geo["le"], dtype=float).copy(),
            "te": np.asarray(geo["te"], dtype=float).copy(),
        }

    return out


def displacements_from_section_motion(
    nodes: dict[str, np.ndarray],
    *,
    heave: np.ndarray,
    twist: np.ndarray,
    pivot_x_c: float = 0.25,
) -> NodeDisplacements:
    """Compute node displacements for beam-like section heave and twist.

    Parameters
    ----------
    nodes : dict[str, np.ndarray]
        Dictionary containing 'le' and 'te' node arrays of shape (n_edges, 3) [m],
        as :func:`undeformed_nodes` gives them. If it also contains 'eta',
        the result keeps a copy of it, so the lattice checks that the mesh
        has the same strip edges.
    heave : np.ndarray
        Spanwise vertical displacement (+z) at each edge [m], shape (n_edges,).
    twist : np.ndarray
        Nose-up twist angle at each edge [rad], shape (n_edges,).
    pivot_x_c : float
        Chord fraction from the leading edge where the section rotation occurs [-].

    Returns
    -------
    NodeDisplacements
        Node displacements for the leading edge and trailing edge.

    Raises
    ------
    ValueError
        If the surface is not flat in the x-y plane or array shapes do not match.
    """
    le0 = np.asarray(nodes["le"], dtype=float)
    te0 = np.asarray(nodes["te"], dtype=float)
    if le0.ndim != 2 or le0.shape[1] != 3 or le0.shape != te0.shape:
        raise ValueError(
            f"Expected 'le' and 'te' arrays of shape (n_edges, 3), got {le0.shape} and {te0.shape}."
        )

    n_edges = le0.shape[0]
    h = np.asarray(heave, dtype=float)
    tw = np.asarray(twist, dtype=float)
    if h.shape != (n_edges,):
        raise ValueError(f"heave must have shape ({n_edges},), got {h.shape}.")
    if tw.shape != (n_edges,):
        raise ValueError(f"twist must have shape ({n_edges},), got {tw.shape}.")

    span = float(np.max(le0[:, 1]) - np.min(le0[:, 1]))
    if span <= 0.0:
        span = float(np.linalg.norm(le0[-1] - le0[0]))
    delta_z = float(np.max(le0[:, 2]) - np.min(le0[:, 2]))
    if delta_z > 1e-9 * max(span, 1e-12):
        raise ValueError(
            f"Surface is not flat in the x-y plane: leading-edge z changes by {delta_z:.4g} m "
            f"over span {span:.4g} m (limit is 1e-9 times the span)."
        )

    p_pivot = le0 + float(pivot_x_c) * (te0 - le0)
    r_le = le0 - p_pivot
    r_te = te0 - p_pivot

    y_axis = np.tile(np.array([0.0, 1.0, 0.0]), (n_edges, 1))
    r_le_rot = _rotate(r_le, y_axis, tw)
    r_te_rot = _rotate(r_te, y_axis, tw)

    heave_vec = np.column_stack([np.zeros(n_edges), np.zeros(n_edges), h])
    le_new = p_pivot + r_le_rot + heave_vec
    te_new = p_pivot + r_te_rot + heave_vec

    d_le = le_new - le0
    d_te = te_new - te0

    eta = nodes.get("eta")
    if eta is not None:
        eta = np.asarray(eta, dtype=float).copy()
    return NodeDisplacements(le=d_le, te=d_te, eta=eta)
