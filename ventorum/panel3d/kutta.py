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
triangular solve per wake strip gives one Jacobian column. The body matrix
factorization is reused. Broyden rank updates are allowed after the first
exact Jacobian; this module recomputes the exact Jacobian at each step,
which is the stronger option.

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

import numpy as np
import scipy.linalg

from ventorum.panel3d.post import compute_surface_gradient, compute_surface_velocity_and_cp

__all__ = [
    "strip_influence_matrix",
    "trailing_edge_residual",
    "solve_pressure_kutta",
]


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
    for s, idx in enumerate(strip_panel_indices):
        cols = np.asarray(idx, dtype=int)
        if cols.size:
            d_mat[:, s] = np.sum(c[:, cols], axis=1)
    return d_mat


def solve_pressure_kutta(
    mu_solve0: np.ndarray,
    mu_full0: np.ndarray,
    lu_and_piv,
    d_solve: np.ndarray,
    wake_up: np.ndarray,
    wake_lo: np.ndarray,
    centroids_full: np.ndarray,
    normals_full: np.ndarray,
    neighbours_full: list[list[int]],
    v_onset_full: np.ndarray,
    v_inf: float,
    max_iter: int = 20,
    tol: float = 1e-6,
    linear_solve=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, bool]:
    """Iterate the wake strengths to zero trailing-edge pressure jump.

    The body solution follows the wake correction through the factored body
    matrix. The correction starts at zero (the Morino solution).

    Parameters
    ----------
    mu_solve0 : numpy.ndarray
        Morino body doublets on the solved mesh with shape (N,) in m^2/s.
    mu_full0 : numpy.ndarray
        Morino body doublets on the full mesh with shape (Nf,) in m^2/s.
    lu_and_piv
        LU factorization of the solved body matrix from ``scipy.linalg.lu_factor``.
    d_solve : numpy.ndarray
        Strip influence matrix on the solved mesh with shape (N, Ns) [-].
    wake_up : numpy.ndarray
        Upper trailing-edge panel indices with shape (Ns,).
    wake_lo : numpy.ndarray
        Lower trailing-edge panel indices with shape (Ns,).
    centroids_full : numpy.ndarray
        Full mesh panel centroids with shape (Nf, 3) in m.
    normals_full : numpy.ndarray
        Full mesh unit normals with shape (Nf, 3).
    neighbours_full : list of list of int
        Full mesh edge neighbours without the wake shedding line.
    v_onset_full : numpy.ndarray
        Onset velocity at full centroids with shape (Nf, 3) in m/s.
    v_inf : float
        Free-stream speed in m/s.
    max_iter : int, default=20
        Maximum Newton steps.
    tol : float, default=1e-6
        Convergence threshold on max |delta Cp| [-].
    linear_solve : callable or None, optional
        Solve of one right-hand side column through the factored body
        matrix: ``linear_solve(rhs)`` returns ``x`` with ``A x = rhs``.
        None uses ``scipy.linalg.lu_solve`` on ``lu_and_piv`` (numpy and
        numba backends). The gpu backend passes a solve through the torch
        factorization on the device.

    Returns
    -------
    mu_solve : numpy.ndarray
        Corrected body doublets on the solved mesh with shape (N,) in m^2/s.
    mu_full : numpy.ndarray
        Corrected body doublets on the full mesh with shape (Nf,) in m^2/s.
    delta : numpy.ndarray
        Wake strength corrections with shape (Ns,) in m^2/s.
    n_iters : int
        Number of Newton updates performed.
    residual : float
        Final max |delta Cp| [-].
    converged : bool
        True when the residual falls below ``tol`` within ``max_iter`` steps.
    """
    mu_s = np.array(mu_solve0, dtype=float, copy=True).reshape(-1)
    mu_f = np.array(mu_full0, dtype=float, copy=True).reshape(-1)
    d_mat = np.asarray(d_solve, dtype=float)
    n_s = d_mat.shape[1]
    up_idx = np.asarray(wake_up, dtype=int).reshape(-1)
    lo_idx = np.asarray(wake_lo, dtype=int).reshape(-1)
    v_on = np.asarray(v_onset_full, dtype=float)
    v_inf_f = float(v_inf)

    if n_s == 0 or v_inf_f <= 1.0e-12:
        return mu_s, mu_f, np.zeros(n_s, dtype=float), 0, 0.0, True

    # Linear sensitivity of the solved doublets to each wake correction.
    z_solve = np.zeros((mu_s.shape[0], n_s), dtype=float)
    for t in range(n_s):
        col = -d_mat[:, t]
        if linear_solve is None:
            z_solve[:, t] = scipy.linalg.lu_solve(lu_and_piv, col)
        else:
            z_solve[:, t] = np.asarray(linear_solve(col), dtype=float).reshape(-1)

    # Full mesh sensitivity: equal correction on the mirror side.
    if mu_f.shape[0] == mu_s.shape[0]:
        z_full = z_solve
    else:
        z_full = np.vstack([z_solve, z_solve])

    delta = np.zeros(n_s, dtype=float)

    grad = compute_surface_gradient(centroids_full, normals_full, mu_f, neighbours_full)
    v_surf, cp = compute_surface_velocity_and_cp(v_on, normals_full, grad, v_inf_f)
    resid = trailing_edge_residual(cp, up_idx, lo_idx)
    max_res = float(np.max(np.abs(resid)))
    if max_res < tol:
        return mu_s, mu_f, delta, 0, max_res, True

    n_iters = 0
    converged = False
    for _ in range(int(max_iter)):
        # Exact Jacobian from the linearized pressure at the current speed.
        jac = np.zeros((n_s, n_s), dtype=float)
        for t in range(n_s):
            g_t = compute_surface_gradient(
                centroids_full, normals_full, z_full[:, t], neighbours_full
            )
            v_up = v_surf[up_idx]
            v_lo = v_surf[lo_idx]
            g_up = g_t[up_idx]
            g_lo = g_t[lo_idx]
            jac[:, t] = 2.0 * (
                np.sum(v_up * g_up, axis=1) - np.sum(v_lo * g_lo, axis=1)
            ) / v_inf_f**2
        try:
            step = np.linalg.solve(jac, -resid)
        except np.linalg.LinAlgError:
            step, _, _, _ = np.linalg.lstsq(jac, -resid, rcond=None)
        n_iters += 1

        # Backtracking: accept the first halving that lowers the residual.
        best_mu_s = mu_s + z_solve @ step
        best_mu_f = mu_f + z_full @ step
        best_delta = delta + step
        best_resid = resid
        best_max = max_res
        found = False
        for halve in range(6):
            frac = 0.5**halve
            trial_delta = delta + frac * step
            trial_mu_s = mu_s + z_solve @ (frac * step)
            trial_mu_f = mu_f + z_full @ (frac * step)
            g_try = compute_surface_gradient(
                centroids_full, normals_full, trial_mu_f, neighbours_full
            )
            _, cp_try = compute_surface_velocity_and_cp(v_on, normals_full, g_try, v_inf_f)
            r_try = trailing_edge_residual(cp_try, up_idx, lo_idx)
            m_try = float(np.max(np.abs(r_try)))
            if m_try < max_res:
                best_mu_s = trial_mu_s
                best_mu_f = trial_mu_f
                best_delta = trial_delta
                best_resid = r_try
                best_max = m_try
                found = True
                break
        if not found:
            g_full = compute_surface_gradient(
                centroids_full, normals_full, best_mu_f, neighbours_full
            )
            _, cp_full = compute_surface_velocity_and_cp(v_on, normals_full, g_full, v_inf_f)
            best_resid = trailing_edge_residual(cp_full, up_idx, lo_idx)
            best_max = float(np.max(np.abs(best_resid)))

        mu_s = best_mu_s
        mu_f = best_mu_f
        delta = best_delta
        resid = best_resid
        max_res = best_max
        grad = compute_surface_gradient(centroids_full, normals_full, mu_f, neighbours_full)
        v_surf, cp = compute_surface_velocity_and_cp(v_on, normals_full, grad, v_inf_f)
        resid = trailing_edge_residual(cp, up_idx, lo_idx)
        max_res = float(np.max(np.abs(resid)))
        if max_res < tol:
            converged = True
            break

    return mu_s, mu_f, delta, int(n_iters), float(max_res), bool(converged)
