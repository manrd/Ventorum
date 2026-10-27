# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Iterative pressure Kutta condition for the 3D panel method.

The linear Morino Kutta condition sets each wake strip strength from the
trailing-edge doublet jump. A thick trailing edge then keeps a pressure jump
between the upper and lower trailing-edge panels. The pressure Kutta
condition removes that jump by adjusting the wake strip strengths. The
adjustment is nonlinear because the surface pressure depends on the square
of the surface speed.

The unknowns are the wake strip strength corrections. The residual at each
trailing-edge panel pair is the pressure jump. The iteration starts from
the Morino solution. Each Newton step uses an exact Jacobian built from the
linear sensitivity of the body doublets to the wake strengths. Each wake
strength enters the solved linear system on the right-hand side only, so one
triangular solve per wake strip gives one sensitivity column. The body
matrix factorization is reused. Broyden rank updates are allowed after the
first exact Jacobian; this module recomputes the exact Jacobian at each
step, which is the stronger option. The surface gradient is a linear
operator (see :func:`ventorum.panel3d.post.build_surface_gradient_operator`),
so the gradients of all sensitivity columns come from one sparse product.

On a half mesh the solver gives the sensitivity of the full-mesh doublets
to each unknown: for a symmetric flow one correction per right strip that
also applies to its mirror strip; for an asymmetric flow one correction per
strip of the full wake, from the symmetric and the antisymmetric
factorizations (a correction on one side is half a symmetric plus half an
antisymmetric correction).

Published sources
-----------------
* Kinnas, S. A. and Hsin, C. Y., 1992, Boundary element method for the
  analysis of the unsteady flow around extreme propeller geometries,
  AIAA Journal, Vol. 30, No. 3 (iterative pressure Kutta).
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, section 12.4 (the Kutta condition).
* Maskew, B., 1987, Program VSAERO Theory Document, NASA CR-4023.

Conventions
-----------
* Geometry axes: x aft [m], y right [m], z up [m].
* Doublet strengths in [m^2/s]. Source strengths in [m/s].
* Pressure coefficient [-]. Residual is a pressure jump [-].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ventorum.panel3d.post import compute_surface_velocity_and_cp, surface_gradient

__all__ = [
    "PressureKuttaResult",
    "strip_influence_matrix",
    "trailing_edge_residual",
    "solve_pressure_kutta",
]

#: Largest number of Newton steps of the pressure Kutta iteration.
MAX_ITERATIONS = 20

#: Convergence threshold on the largest trailing-edge pressure jump [-].
TOLERANCE = 1.0e-6


def trailing_edge_residual(
    cp: np.ndarray,
    wake_up: np.ndarray,
    wake_lo: np.ndarray,
) -> np.ndarray:
    """Compute the trailing-edge pressure jump at each strip.

    Parameters
    ----------
    cp : numpy.ndarray
        Pressure coefficients on body panels with shape (Np,) [-].
    wake_up : numpy.ndarray
        Upper trailing-edge panel indices with shape (Ns,).
    wake_lo : numpy.ndarray
        Lower trailing-edge panel indices with shape (Ns,).

    Returns
    -------
    numpy.ndarray
        Residual vector with shape (Ns,) [-].
    """
    c = np.asarray(cp, dtype=float)
    up_idx = np.asarray(wake_up, dtype=int)
    lo_idx = np.asarray(wake_lo, dtype=int)
    return c[up_idx] - c[lo_idx]


def strip_influence_matrix(
    cw: np.ndarray,
    strip_panel_indices: list[list[int]],
) -> np.ndarray:
    """Sum wake panel potentials into one column per wake strip.

    Parameters
    ----------
    cw : numpy.ndarray
        Doublet potentials of wake panels with shape (Np, Nw) [-].
    strip_panel_indices : list of list of int
        Wake panel indices of each strip.

    Returns
    -------
    numpy.ndarray
        Strip influence matrix with shape (Np, Ns) [-].
    """
    c = np.asarray(cw, dtype=float)
    n_s = len(strip_panel_indices)
    d_mat = np.zeros((c.shape[0], n_s), dtype=float)
    lengths = np.fromiter(map(len, strip_panel_indices), dtype=np.intp, count=n_s)
    active = lengths > 0
    if np.any(active):
        cols = np.concatenate(strip_panel_indices).astype(np.intp, copy=False)
        starts = np.cumsum(lengths) - lengths
        d_mat[:, active] = np.add.reduceat(c[:, cols], starts[active], axis=1)
    return d_mat


@dataclass(slots=True)
class PressureKuttaResult:
    """Result of the pressure Kutta iteration.

    Attributes
    ----------
    delta : numpy.ndarray
        Wake strength corrections of the unknowns with shape (Nu,) in m^2/s.
    mu_full : numpy.ndarray
        Corrected body doublets on the full mesh with shape (Nf,) in m^2/s.
    n_iters : int
        Number of Newton updates performed.
    residual : float
        Final largest trailing-edge pressure jump [-].
    converged : bool
        True when the residual falls below the tolerance.
    """

    delta: np.ndarray
    mu_full: np.ndarray
    n_iters: int
    residual: float
    converged: bool


def _surface_cp(grad_op, mu_full, v_onset, normals, v_inf, rates=False):
    """Return the surface velocity (Nf, 3) in m/s and Cp (Nf,) of a doublet vector."""
    grad = surface_gradient(grad_op, mu_full)
    return compute_surface_velocity_and_cp(v_onset, normals, grad, v_inf, rates)


def solve_pressure_kutta(
    mu_full0: np.ndarray,
    z_full: np.ndarray,
    grad_op,
    wake_up: np.ndarray,
    wake_lo: np.ndarray,
    normals_full: np.ndarray,
    v_onset_full: np.ndarray,
    v_inf: float,
    max_iter: int = MAX_ITERATIONS,
    tol: float = TOLERANCE,
    rates: bool = False,
) -> PressureKuttaResult:
    """Iterate the wake strengths to zero trailing-edge pressure jump.

    The body doublets follow the wake corrections through the sensitivity
    matrix *z_full*: ``mu_full = mu_full0 + z_full @ delta``. The corrections
    start at zero (the Morino solution). Each Newton step solves
    ``J d = -R`` with the exact Jacobian
    ``J[s, t] = 2 (V_up(s) . g_t,up(s) - V_lo(s) . g_t,lo(s)) / V_inf^2``,
    where ``g_t`` is the surface gradient of column t of *z_full*, then
    halves the step (at most 5 times) until the largest residual falls.

    Parameters
    ----------
    mu_full0 : numpy.ndarray
        Morino body doublets on the full mesh with shape (Nf,) in m^2/s.
    z_full : numpy.ndarray
        Sensitivity of the full-mesh doublets to the unknown wake
        corrections with shape (Nf, Nu) [-].
    grad_op : scipy.sparse.spmatrix
        Surface gradient operator of the full mesh with shape (3 Nf, Nf)
        in 1/m.
    wake_up : numpy.ndarray
        Upper trailing-edge panel indices of the residual with shape (Nu,).
    wake_lo : numpy.ndarray
        Lower trailing-edge panel indices of the residual with shape (Nu,).
    normals_full : numpy.ndarray
        Full mesh unit normals with shape (Nf, 3).
    v_onset_full : numpy.ndarray
        Onset velocity at full centroids with shape (Nf, 3) in m/s.
    v_inf : float
        Free-stream speed in m/s.
    max_iter : int, default=20
        Maximum Newton steps.
    tol : float, default=1e-6
        Convergence threshold on max |delta Cp| [-].
    rates : bool, default=False
        True when the condition has a body rate. Cp then includes the
        kinematic term of the rotating body (see
        :func:`ventorum.panel3d.post.compute_surface_velocity_and_cp`).
        The term does not depend on the doublets, so the Jacobian does not
        change.

    Returns
    -------
    PressureKuttaResult
        Corrections, corrected doublets, step count, residual and the
        convergence flag.
    """
    mu_f = np.array(mu_full0, dtype=float, copy=True).reshape(-1)
    z_f = np.asarray(z_full, dtype=float)
    n_u = z_f.shape[1]
    up_idx = np.asarray(wake_up, dtype=int).reshape(-1)
    lo_idx = np.asarray(wake_lo, dtype=int).reshape(-1)
    v_on = np.asarray(v_onset_full, dtype=float)
    v_inf_f = float(v_inf)
    delta = np.zeros(n_u, dtype=float)

    if n_u == 0 or v_inf_f <= 1.0e-12:
        return PressureKuttaResult(delta, mu_f, 0, 0.0, True)
    if up_idx.shape[0] != n_u or lo_idx.shape[0] != n_u:
        raise ValueError(
            f"The pressure Kutta system needs one residual per unknown: "
            f"{up_idx.shape[0]} residuals for {n_u} unknowns."
        )

    # Gradients of all sensitivity columns: shape (Nf, 3, Nu).
    g_z = surface_gradient(grad_op, z_f)
    g_up = g_z[up_idx]
    g_lo = g_z[lo_idx]

    v_surf, cp = _surface_cp(grad_op, mu_f, v_on, normals_full, v_inf_f, rates)
    resid = trailing_edge_residual(cp, up_idx, lo_idx)
    max_res = float(np.max(np.abs(resid)))
    if max_res < tol:
        return PressureKuttaResult(delta, mu_f, 0, max_res, True)

    n_iters = 0
    converged = False
    for _ in range(int(max_iter)):
        # Exact Jacobian from the linearized pressure at the current speed.
        jac = 2.0 * (
            np.einsum("sd,sdt->st", v_surf[up_idx], g_up)
            - np.einsum("sd,sdt->st", v_surf[lo_idx], g_lo)
        ) / v_inf_f**2
        try:
            step = np.linalg.solve(jac, -resid)
        except np.linalg.LinAlgError:
            step, _, _, _ = np.linalg.lstsq(jac, -resid, rcond=None)
        n_iters += 1

        # Backtracking: accept the first halving that lowers the residual,
        # else take the full step.
        frac_used = 1.0
        for halve in range(6):
            frac = 0.5**halve
            trial = mu_f + z_f @ (frac * step)
            _, cp_try = _surface_cp(grad_op, trial, v_on, normals_full, v_inf_f, rates)
            m_try = float(np.max(np.abs(trailing_edge_residual(cp_try, up_idx, lo_idx))))
            if m_try < max_res:
                frac_used = frac
                break
        delta = delta + frac_used * step
        mu_f = mu_f + z_f @ (frac_used * step)
        v_surf, cp = _surface_cp(grad_op, mu_f, v_on, normals_full, v_inf_f, rates)
        resid = trailing_edge_residual(cp, up_idx, lo_idx)
        max_res = float(np.max(np.abs(resid)))
        if max_res < tol:
            converged = True
            break

    return PressureKuttaResult(delta, mu_f, int(n_iters), float(max_res), bool(converged))
