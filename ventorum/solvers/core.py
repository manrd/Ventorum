# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Solver kernels shared by the public solver classes.

* :func:`solve_vlm`: vortex-lattice method. Flow tangency at the control
  points: ``(V_inf + v) . n_bc = 0``.
* :func:`solve_llt_linear`: linear numerical lifting line of Phillips &
  Snyder (2000). At the middle of each bound vortex the Kutta-Joukowski lift
  equals the section lift ``a0 (alpha - alpha_L0)``.
* :func:`solve_llt_nonlinear`: the same lifting-line equations with the
  section lift taken from the polar, solved by Newton's method with a line
  search (Phillips & Snyder 2000, Phillips 2004).

References
----------
W. F. Phillips and D. O. Snyder, "Modern adaptation of Prandtl's classic
lifting-line theory", Journal of Aircraft 37(4), 2000, pp. 662-670.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ventorum.aero.loads import group_strips_by_airfoil
from ventorum.aero.system import (
    GroundPlane,
    UnknownMap,
    build_sources,
    freestream_direction,
    is_symmetric_condition,
    make_unknown_map,
    panel_targets,
)
from ventorum.aero.vortex import influence_matrix, velocity_tensor_unknowns
from ventorum.core.datatypes import FlightCondition
from ventorum.geometry.lattice import VortexLattice, section_cl, section_cl_slope
from ventorum.utils.vec import cross3


@dataclass
class SolveInfo:
    """Hold the convergence information of a circulation solve."""

    converged: bool = True
    iterations: int = 1
    residual_history: list[float] = field(default_factory=list)
    symmetric: bool = False
    condition_number: float | None = None
    # Induced velocity [m/s] at the control point of every panel, shape
    # (n_panels, 3), when the solver has it (lifting line). The loads then
    # use it instead of a second kernel evaluation at the same points.
    v_control: np.ndarray | None = None


_MIRROR_Y = np.array([1.0, -1.0, 1.0])


def _panel_velocity(lattice: VortexLattice, umap: UnknownMap, v_unknown: np.ndarray) -> np.ndarray:
    """Velocity at the control point of every panel from the values at the unknown panels.

    On a symmetric map the velocity of a left panel is the mirror image of the
    velocity of its mirror panel (the same rule as the symmetric fold of the loads).
    """
    v = np.empty((lattice.n_panels, 3))
    v[umap.unknown_panels] = v_unknown
    if umap.symmetric:
        left = lattice.geom_cache.get("left_panels")
        if left is None:
            left = np.setdiff1d(np.arange(lattice.n_panels), umap.unknown_panels)
            lattice.geom_cache["left_panels"] = left
        v[left] = v[lattice.panel_mirror[left]] * _MIRROR_Y
    return v


def _unknown_map(lattice: VortexLattice, condition: FlightCondition,
                 ground: GroundPlane | None, use_symmetry: bool) -> UnknownMap:
    sym = bool(use_symmetry) and is_symmetric_condition(condition, ground)
    return make_unknown_map(lattice, sym)


def assemble_vlm(
    lattice: VortexLattice,
    condition: FlightCondition,
    ground: GroundPlane | None = None,
    use_symmetry: bool = True,
    wake_dir: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, UnknownMap]:
    """Matrix and right-hand side of the vortex-lattice system."""
    if lattice.collocation != "vlm":
        raise ValueError("assemble_vlm needs a lattice built with collocation='vlm'.")
    d = freestream_direction(condition.alpha, condition.beta)
    wd = d if wake_dir is None else np.asarray(wake_dir, dtype=float)
    umap = _unknown_map(lattice, condition, ground, use_symmetry)
    src = build_sources(lattice, wd, umap, ground)
    P = lattice.cp[umap.unknown_panels]
    tg = panel_targets(lattice, umap.unknown_panels)
    n_bc = lattice.normal_bc[umap.unknown_panels]
    cache = lattice.kernel_cache if ground is None else None
    A = influence_matrix(P, n_bc, src, umap.n, tg, cache=cache, cache_key=("vlm", umap.symmetric))
    rhs = -condition.V_inf * (n_bc @ d)
    return A, rhs, umap


def solve_vlm(
    lattice: VortexLattice,
    condition: FlightCondition,
    ground: GroundPlane | None = None,
    use_symmetry: bool = True,
    wake_dir: np.ndarray | None = None,
) -> tuple[np.ndarray, SolveInfo]:
    """Solve the vortex lattice. Returns the circulation of every panel."""
    A, rhs, umap = assemble_vlm(lattice, condition, ground, use_symmetry, wake_dir)
    g = np.linalg.solve(A, rhs)
    return g[umap.panel_column], SolveInfo(symmetric=umap.symmetric)


def _llt_strip_data(lattice: VortexLattice, umap: UnknownMap) -> dict[str, np.ndarray]:
    if lattice.collocation != "llt":
        raise ValueError("Lifting-line solvers need a lattice built with collocation='llt'.")
    s = lattice.panel_strip[umap.unknown_panels]
    return {
        "strip": s,
        "dl": lattice.dl[s],
        "area": lattice.area[s],
        "normal": lattice.normal[s],
        "chord_dir": lattice.chord_dir[s],
        "a0": np.maximum(lattice.a0[s], 1e-6),
        "alpha_L0": lattice.alpha_L0[s],
    }


def assemble_llt_linear(
    lattice: VortexLattice,
    condition: FlightCondition,
    ground: GroundPlane | None = None,
    use_symmetry: bool = True,
    wake_dir: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, UnknownMap]:
    """Linear Phillips & Snyder system ``M G = r``.

    ``[2 |u x dl_i| / (a0_i dA_i)] G_i - sum_j (v_ij . n_i) G_j = V (u . n_i - alpha_L0_i)``
    """
    M, rhs, umap, _ = _assemble_llt_linear(lattice, condition, ground, use_symmetry, wake_dir)
    return M, rhs, umap


def _linear_llt_matrix(lattice, condition, sd, A_ind):
    """Matrix and right-hand side of the linear lifting line from the normal influence *A_ind*."""
    d = freestream_direction(condition.alpha, condition.beta)
    diag = 2.0 * np.linalg.norm(cross3(d[None, :], sd["dl"]), axis=1) / (sd["a0"] * sd["area"])
    M = np.diag(diag) - A_ind
    rhs = condition.V_inf * (sd["normal"] @ d - sd["alpha_L0"])
    return M, rhs


def _assemble_llt_linear(lattice, condition, ground, use_symmetry, wake_dir):
    """Linear lifting-line system, and the velocity tensor per unknown when it was computed.

    Outside sweeps (no kernel cache) the system comes from the velocity
    tensor per unknown, which the loads reuse for the induced velocity at
    the control points (one kernel evaluation instead of two). In sweeps the
    influence matrix with the kernel cache is used.
    """
    d = freestream_direction(condition.alpha, condition.beta)
    wd = d if wake_dir is None else np.asarray(wake_dir, dtype=float)
    umap = _unknown_map(lattice, condition, ground, use_symmetry)
    sd = _llt_strip_data(lattice, umap)
    src = build_sources(lattice, wd, umap, ground)
    P = lattice.cp[umap.unknown_panels]
    tg = panel_targets(lattice, umap.unknown_panels)
    cache = lattice.kernel_cache if ground is None else None
    if cache is None:
        Vt = velocity_tensor_unknowns(P, src, umap.n, tg)
        A_ind = np.einsum("ijk,ik->ij", Vt, sd["normal"])
    else:
        Vt = None
        A_ind = influence_matrix(P, sd["normal"], src, umap.n, tg, cache=cache, cache_key=("llt", umap.symmetric))
    M, rhs = _linear_llt_matrix(lattice, condition, sd, A_ind)
    return M, rhs, umap, Vt


def solve_llt_linear(
    lattice: VortexLattice,
    condition: FlightCondition,
    ground: GroundPlane | None = None,
    use_symmetry: bool = True,
    wake_dir: np.ndarray | None = None,
) -> tuple[np.ndarray, SolveInfo]:
    """Solve the linear lifting line. Returns the circulation of every strip."""
    M, rhs, umap, Vt = _assemble_llt_linear(lattice, condition, ground, use_symmetry, wake_dir)
    g = np.linalg.solve(M, rhs)
    info = SolveInfo(symmetric=umap.symmetric)
    if Vt is not None:
        info.v_control = _panel_velocity(lattice, umap, np.einsum("ijk,j->ik", Vt, g))
    return g[umap.panel_column], info


def solve_llt_nonlinear(
    lattice: VortexLattice,
    condition: FlightCondition,
    ground: GroundPlane | None = None,
    use_symmetry: bool = True,
    max_iterations: int = 100,
    tolerance: float = 1.0e-8,
    gamma0: np.ndarray | None = None,
    wake_dir: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, SolveInfo]:
    """Solve the nonlinear lifting line with section polars (Newton's method).

    The residual of strip i (Phillips & Snyder 2000, eq. 15 in dimensional
    form) is::

        R_i = 2 |W_i x dl_i| G_i - V_inf^2 dA_i Cl_i(alpha_i)

    with the local velocity ``W_i = V_inf + sum_j v_ij G_j`` at the middle of
    the bound vortex and ``alpha_i = atan2(W_i . n_i, W_i . a_i)``.

    The convergence test uses ``max_i |R_i| / (V_inf^2 dA_i)``, the error in
    the section lift coefficient.

    Returns
    -------
    gamma_strip : numpy.ndarray
        Circulation of every strip [m^2/s], shape (n_strips,).
    alpha_eff : numpy.ndarray
        Section angle of attack at the solution [rad], shape (n_strips,).
    info : SolveInfo
        Convergence information.
    """
    V = condition.V_inf
    d = freestream_direction(condition.alpha, condition.beta)
    wd = d if wake_dir is None else np.asarray(wake_dir, dtype=float)
    umap = _unknown_map(lattice, condition, ground, use_symmetry)
    sd = _llt_strip_data(lattice, umap)
    src = build_sources(lattice, wd, umap, ground)
    P = lattice.cp[umap.unknown_panels]
    Vt = velocity_tensor_unknowns(P, src, umap.n, panel_targets(lattice, umap.unknown_panels))  # (n, n, 3)
    dl, dA, n_vec, a_vec = sd["dl"], sd["area"], sd["normal"], sd["chord_dir"]
    strips = sd["strip"]

    afs = [lattice.airfoils[s] for s in strips]
    groups = group_strips_by_airfoil(afs)

    def section_cl_all(alpha: np.ndarray) -> np.ndarray:
        cl = np.empty_like(alpha)
        for af, idx in groups:
            cl[idx] = section_cl(af, alpha[idx])
        return cl

    def cl_and_slope(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cl = np.empty_like(alpha)
        dcl = np.empty_like(alpha)
        for af, idx in groups:
            cl[idx] = section_cl(af, alpha[idx])
            dcl[idx] = section_cl_slope(af, alpha[idx])
        return cl, dcl

    Vinf = V * d
    scale = V * V * dA

    def evaluate(g: np.ndarray):
        W = Vinf[None, :] + np.einsum("ijk,j->ik", Vt, g)
        wx = cross3(W, dl)
        wxn = np.linalg.norm(wx, axis=1)
        wn = np.sum(W * n_vec, axis=1)
        wa = np.sum(W * a_vec, axis=1)
        alpha = np.arctan2(wn, wa)
        cl, dcl = cl_and_slope(alpha)
        R = 2.0 * wxn * g - scale * cl
        return R, W, wx, wxn, wn, wa, alpha, dcl

    if gamma0 is None:
        # Linear start from the same velocity tensor (no second kernel pass).
        M_lin, rhs_lin = _linear_llt_matrix(lattice, condition, sd, np.einsum("ijk,ik->ij", Vt, sd["normal"]))
        g = np.linalg.solve(M_lin, rhs_lin)
    else:
        g = np.asarray(gamma0, dtype=float)[umap.unknown_panels].copy()

    history: list[float] = []
    R, W, wx, wxn, wn, wa, alpha, dcl = evaluate(g)
    res = float(np.max(np.abs(R) / scale))
    history.append(res)
    converged = res < tolerance
    vn = np.einsum("ijk,ik->ij", Vt, n_vec)
    va = np.einsum("ijk,ik->ij", Vt, a_vec)

    def merit(r: np.ndarray) -> float:
        return 0.5 * float(np.sum((r / scale) ** 2))

    it = 0
    while not converged and it < max_iterations:
        it += 1
        f0 = merit(R)
        # Jacobian dR_i/dG_j
        # d|W_i x dl_i|/dG_j = u_i . (v_ij x dl_i) with u_i = (W_i x dl_i)/|W_i x dl_i|;
        # by the scalar triple product this is v_ij . (dl_i x u_i): one cross
        # product per strip instead of one per (i, j) pair.
        u = wx / np.maximum(wxn, 1e-300)[:, None]
        dwxn = np.einsum("ijk,ik->ij", Vt, cross3(dl, u))
        denom = np.maximum(wa ** 2 + wn ** 2, 1e-300)
        dalpha = (wa[:, None] * vn - wn[:, None] * va) / denom[:, None]
        J = 2.0 * g[:, None] * dwxn - (scale * dcl)[:, None] * dalpha
        J[np.diag_indices_from(J)] += 2.0 * wxn
        try:
            step = -np.linalg.solve(J, R)
        except np.linalg.LinAlgError:
            step = None
        accepted = False
        if step is not None:
            # Backtracking line search with a sufficient-decrease test.
            lam = 1.0
            while lam >= 1.0 / 1024.0:
                out = evaluate(g + lam * step)
                if merit(out[0]) <= (1.0 - 1.0e-4 * lam) * f0:
                    g = g + lam * step
                    R, W, wx, wxn, wn, wa, alpha, dcl = out
                    accepted = True
                    break
                lam *= 0.5
        if not accepted:
            # Past the maximum lift the Newton direction can fail (negative
            # section slope). Fall back to a damped fixed-point step towards
            # G = V^2 dA Cl / (2 |W x dl|), with the largest damping that
            # lowers the residual.
            g_fp = scale * section_cl_all(alpha) / (2.0 * np.maximum(wxn, 1e-300))
            for omega in (0.5, 0.25, 0.1, 0.05, 0.02):
                out = evaluate(g + omega * (g_fp - g))
                if merit(out[0]) < f0:
                    g = g + omega * (g_fp - g)
                    R, W, wx, wxn, wn, wa, alpha, dcl = out
                    accepted = True
                    break
        if not accepted:
            # No step lowers the residual. The state did not change, so a new
            # iteration repeats the same steps: stop (the caller can restart
            # from another first guess).
            break
        res = float(np.max(np.abs(R) / scale))
        history.append(res)
        converged = res < tolerance

    gamma_full = g[umap.panel_column]
    alpha_full = np.empty(lattice.n_strips)
    alpha_full[strips] = alpha
    if umap.symmetric:
        left = np.flatnonzero(~lattice.strip_is_right)
        alpha_full[left] = alpha_full[lattice.strip_mirror[left]]
    info = SolveInfo(converged=converged, iterations=it, residual_history=history, symmetric=umap.symmetric)
    info.v_control = _panel_velocity(lattice, umap, W - Vinf[None, :])
    return gamma_full, alpha_full, info
