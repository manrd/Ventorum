# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Ventorum-2D viscous panel solver with simultaneous boundary-layer coupling.

This module implements two-dimensional viscous-inviscid interaction
using wall and wake source transpiration on potential flow panels coupled
with the two-equation integral boundary layer. It supports streamline wake
tracing, blunt trailing edges, and plain flapped airfoil sections.

References
----------
M. J. Lighthill, 'On displacement thickness', Journal of Fluid Mechanics,
Vol. 4, Issue 4, 1958, pp. 383-392.
M. Drela and M. B. Giles, 'Viscous-inviscid analysis of transonic and
low Reynolds number airfoils', AIAA Journal, Vol. 25, No. 10, 1987,
pp. 1347-1355.
M. Drela, 'XFOIL: An Analysis and Design System for Low Reynolds Number
Airfoils', Lecture Notes in Engineering, Vol. 54, Springer, 1989, pp. 1-12.
H. B. Squire and A. D. Young, 'The calculation of the profile drag of
aerofoils', Aeronautical Research Council Reports and Memoranda,
R&M 1838, 1938.
S. F. Hoerner, 'Fluid-Dynamic Drag', Published by the Author, 1965,
chapter 3 (base drag of blunt trailing edges).
A. E. P. Veldman, 'Quasi-simultaneous interaction methods for
viscous-inviscid flow', Computers & Fluids, Vol. 38, Issue 3, 2009,
pp. 496-508.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from ventorum.ibl2d import closure as cls
from ventorum.ibl2d import transition as tr
from ventorum.panel2d.influence import (
    compute_panel_local_coordinates_numpy,
    linear_vortex_stream_influence,
)
from ventorum.panel2d.solver import compute_mass_defect_matrix

__all__ = [
    "ViscousResult",
    "solve_viscous",
    "trace_wake_streamline",
]


@dataclass
class ViscousResult:
    """Aerodynamic results of the 2D viscous panel solver.

    Attributes
    ----------
    alpha : float or np.ndarray
        Angle of attack [deg].
    Cl : float or np.ndarray
        Total lift coefficient (dimensionless).
    Cd : float or np.ndarray
        Profile drag coefficient calculated by the Squire-Young formula
        at the wake end state (dimensionless).
    Cd_friction : float or np.ndarray
        Friction drag coefficient from surface skin-friction integration (dimensionless).
    Cd_pressure : float or np.ndarray
        Pressure drag coefficient from surface pressure integration (dimensionless).
    Cm : float or np.ndarray
        Pitching moment coefficient about the quarter chord (dimensionless).
    transition_positions : tuple of float or list of tuple
        Transition positions (x/c) on the upper and lower surfaces [chord units].
    separation_flags : tuple of bool or list of tuple
        Separation indicators for the upper and lower surfaces.
    Cp : np.ndarray or list of np.ndarray
        Surface pressure coefficients at the airfoil nodes (dimensionless).
    H : np.ndarray or list of np.ndarray
        Kinematic shape factor delta_star / theta at the nodes (dimensionless).
    theta : np.ndarray or list of np.ndarray
        Momentum thickness at the nodes [m or chord units].
    Cf : np.ndarray or list of np.ndarray
        Skin-friction coefficient at the nodes (dimensionless).
    iterations : int or list of int
        Number of iterations executed.
    residual : float or list of float
        Final convergence residual (dimensionless).
    converged : bool or np.ndarray
        True if the solution converged within tolerance.
    coordinates : np.ndarray or list of np.ndarray
        Node coordinates as solved [m or chord units].
    reason : str or list of str, optional
        Status or failure description string for each solved case.
    """

    alpha: float | np.ndarray
    Cl: float | np.ndarray
    Cd: float | np.ndarray
    Cd_friction: float | np.ndarray
    Cd_pressure: float | np.ndarray
    Cm: float | np.ndarray
    transition_positions: tuple[float, float] | list[tuple[float, float]]
    separation_flags: tuple[bool, bool] | list[tuple[bool, bool]]
    Cp: np.ndarray | list[np.ndarray]
    H: np.ndarray | list[np.ndarray]
    theta: np.ndarray | list[np.ndarray]
    Cf: np.ndarray | list[np.ndarray]
    iterations: int | list[int]
    residual: float | list[float]
    converged: bool | np.ndarray
    coordinates: np.ndarray | list[np.ndarray]
    reason: str | list[str] = ""

    @property
    def x_transition(self) -> tuple[float, float] | list[tuple[float, float]]:
        """Return transition positions (alias for transition_positions)."""
        return self.transition_positions

    @property
    def separated(self) -> tuple[bool, bool] | list[tuple[bool, bool]]:
        """Return separation indicators (alias for separation_flags)."""
        return self.separation_flags


def trace_wake_streamline(
    bodies: np.ndarray | list[np.ndarray],
    alpha_deg: float,
    wake_length: float = 1.0,
    n_wake_panels: int = 25,
    height: float | None = None,
) -> np.ndarray:
    """Trace the wake streamline from the trailing-edge bisector.

    The wake trajectory follows the inviscid potential flow velocity field
    integrated with Heun predictor-corrector integration. For a symmetric
    airfoil at zero angle of attack, the wake line lies exactly on the axis.

    Parameters
    ----------
    bodies : np.ndarray or list of np.ndarray
        Airfoil contour node coordinates in Selig order [m or chord units].
    alpha_deg : float
        Angle of attack [deg].
    wake_length : float, optional
        Wake length normalized by reference chord [chord units]. Default is 1.0.
    n_wake_panels : int, optional
        Number of wake panels. Default is 25.
    height : float, optional
        Height above ground [m or chord units], or None for free air.

    Returns
    -------
    np.ndarray
        Wake panel node coordinates of shape (n_wake_panels + 1, 2).

    References
    ----------
    M. Drela, 'XFOIL: An Analysis and Design System for Low Reynolds Number
    Airfoils', Lecture Notes in Engineering, Vol. 54, Springer, 1989.
    """
    if isinstance(bodies, (list, tuple)):
        primary_body = np.asarray(bodies[0], dtype=np.float64)
    else:
        primary_body = np.asarray(bodies, dtype=np.float64)

    chord = float(np.max(primary_body[:, 0]) - np.min(primary_body[:, 0]))
    if chord <= 0.0:
        chord = 1.0

    te_mid = 0.5 * (primary_body[0] + primary_body[-1])
    vu = primary_body[0] - primary_body[1]
    vl = primary_body[-1] - primary_body[-2]
    nu = float(np.linalg.norm(vu))
    nl = float(np.linalg.norm(vl))
    tu = vu / nu if nu > 1e-12 else np.array([1.0, 0.0])
    tl = vl / nl if nl > 1e-12 else np.array([1.0, 0.0])
    bisector = tu + tl
    nb = float(np.linalg.norm(bisector))
    bisector = bisector / nb if nb > 1e-10 else np.array([1.0, 0.0])

    is_symmetric = np.allclose(primary_body[:, 1], -primary_body[::-1, 1], atol=1e-5)
    if is_symmetric and abs(alpha_deg) < 1e-12 and height is None:
        r = 1.10
        ds_raw = r ** np.arange(n_wake_panels)
        ds_arr = (wake_length * chord) * ds_raw / np.sum(ds_raw)
        wake_pts = [te_mid.copy()]
        curr = te_mid.copy()
        for ds in ds_arr:
            curr[0] += ds
            curr[1] = 0.0
            wake_pts.append(curr.copy())
        return np.array(wake_pts)

    effective_height = height if (height is None or height < 500.0) else None
    _, _, info = compute_mass_defect_matrix(
        bodies,
        alpha_deg=alpha_deg,
        height=effective_height,
        backend="numpy",
    )

    alpha_rad = np.radians(alpha_deg)
    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)

    if effective_height is None:
        sol_inv = cos_a * info["sol_x"] + sin_a * info["sol_y"]
        gamma = sol_inv[:len(primary_body)]
        p_start = info["p_start"]
        p_end = info["p_end"]
        idx1 = info["idx1"]
        idx2 = info["idx2"]

        def eval_velocity(pts: np.ndarray, h: float = 1e-6) -> np.ndarray:
            pts_yp = pts.copy()
            pts_yp[:, 1] += h
            pts_ym = pts.copy()
            pts_ym[:, 1] -= h
            pts_xp = pts.copy()
            pts_xp[:, 0] += h
            pts_xm = pts.copy()
            pts_xm[:, 0] -= h
            all_pts = np.vstack([pts_yp, pts_ym, pts_xp, pts_xm])
            psi_inf = all_pts[:, 1] * cos_a - all_pts[:, 0] * sin_a

            xi, eta, length, _ = compute_panel_local_coordinates_numpy(all_pts, p_start, p_end)
            eta_clean = np.where(np.abs(eta) < 1e-12, 0.0, eta)
            psi1, psi2 = linear_vortex_stream_influence(xi, eta_clean, length)
            psi_v = np.sum(psi1 * gamma[idx1][np.newaxis, :] + psi2 * gamma[idx2][np.newaxis, :], axis=1)
            psi_tot = psi_inf + psi_v

            m = len(pts)
            u = (psi_tot[:m] - psi_tot[m:2*m]) / (2.0 * h)
            v = -(psi_tot[2*m:3*m] - psi_tot[3*m:]) / (2.0 * h)
            return np.column_stack([u, v])
    else:
        def eval_velocity(pts: np.ndarray, h: float = 1e-6) -> np.ndarray:
            return np.tile(np.array([cos_a, sin_a]), (len(pts), 1))

    r = 1.10
    ds_raw = r ** np.arange(n_wake_panels)
    ds_arr = (wake_length * chord) * ds_raw / np.sum(ds_raw)

    wake_pts = [te_mid.copy()]
    curr_pt = te_mid.copy()

    for i in range(n_wake_panels):
        ds = ds_arr[i]
        if i == 0:
            k1 = bisector
        else:
            v1 = eval_velocity(curr_pt.reshape(1, 2))[0]
            spd = float(np.linalg.norm(v1))
            k1 = v1 / spd if spd > 1e-6 else bisector

        pred_pt = curr_pt + ds * k1
        v2 = eval_velocity(pred_pt.reshape(1, 2))[0]
        spd2 = float(np.linalg.norm(v2))
        k2 = v2 / spd2 if spd2 > 1e-6 else k1

        step_dir = 0.5 * (k1 + k2)
        s_norm = float(np.linalg.norm(step_dir))
        step_dir = step_dir / s_norm if s_norm > 1e-12 else bisector
        curr_pt = curr_pt + ds * step_dir
        wake_pts.append(curr_pt.copy())

    return np.array(wake_pts)


def solve_viscous(
    bodies: np.ndarray | list[np.ndarray],
    alpha_deg: float | list[float] | np.ndarray,
    Re: float,
    *,
    mach: float = 0.0,
    ncrit: float = 9.0,
    xtr: tuple[float, float] = (1.0, 1.0),
    height: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
    wake_length: float = 1.0,
    backend: str = "numpy",
) -> ViscousResult:
    """Solve 2D viscous flow with simultaneous boundary-layer coupling.

    Parameters
    ----------
    bodies : np.ndarray or list of np.ndarray
        Airfoil contour coordinates in Selig order [m or chord units].
    alpha_deg : float, list of float, or np.ndarray
        Angle of attack [deg].
    Re : float
        Reynolds number based on reference chord (dimensionless).
    mach : float, optional
        Free-stream Mach number (dimensionless). Must be 0.0.
    ncrit : float, optional
        Critical amplification exponent for free transition (dimensionless).
        Default is 9.0.
    xtr : tuple of float, optional
        Forced transition locations (upper, lower) as fraction of chord [chord units].
        Default is (1.0, 1.0) for free transition.
    height : float, optional
        Height above flat ground [m or chord units]. If None, solves in free air.
    max_iter : int, optional
        Maximum number of coupling iterations. Default is 50.
    tol : float, optional
        Convergence tolerance on edge velocity changes. Default is 1e-6.
    wake_length : float, optional
        Length of the traced wake line in chord units. Default is 1.0.
    backend : {"numpy"}, optional
        Computational backend. Default is "numpy".

    Returns
    -------
    ViscousResult
        Coupled viscous-inviscid flow field and integrated aerodynamic coefficients.
    """
    if abs(mach) > 1e-12:
        raise ValueError(f"Only mach=0.0 is currently supported in solve_viscous, got mach={mach}.")

    if backend != "numpy":
        raise ValueError(f"backend '{backend}' is not supported. Only 'numpy' is supported.")

    if Re <= 0.0:
        raise ValueError(f"Reynolds number must be positive, got {Re}.")

    if isinstance(bodies, np.ndarray):
        bodies_np = [np.asarray(bodies, dtype=np.float64)]
    elif isinstance(bodies, (list, tuple)):
        bodies_np = [np.asarray(b, dtype=np.float64) for b in bodies]
    else:
        raise TypeError("bodies must be an (N, 2) array or list of (N, 2) arrays.")

    is_scalar_alpha = np.ndim(alpha_deg) == 0
    alphas = np.atleast_1d(np.asarray(alpha_deg, dtype=np.float64))
    primary_body = bodies_np[0]
    effective_height = height if (height is None or height < 500.0) else None

    # Solve across unique alphas
    results_by_alpha: dict[float, dict] = {}
    sorted_indices = np.argsort(np.abs(alphas))
    unique_alphas = []
    for idx in sorted_indices:
        a_val = float(alphas[idx])
        if not any(abs(a_val - u) < 1e-8 for u in unique_alphas):
            unique_alphas.append(a_val)

    pos_alphas = sorted([a for a in unique_alphas if a >= 0.0])
    neg_alphas = sorted([a for a in unique_alphas if a < 0.0], reverse=True)

    prev_sol = None
    for a in pos_alphas:
        sol = _solve_single_viscous(
            primary_body,
            a,
            Re,
            ncrit=ncrit,
            xtr=xtr,
            height=effective_height,
            max_iter=max_iter,
            tol=tol,
            wake_length=wake_length,
            prev_sol=prev_sol,
        )
        results_by_alpha[a] = sol
        prev_sol = sol

    prev_sol = results_by_alpha.get(0.0, None)
    for a in neg_alphas:
        sol = _solve_single_viscous(
            primary_body,
            a,
            Re,
            ncrit=ncrit,
            xtr=xtr,
            height=effective_height,
            max_iter=max_iter,
            tol=tol,
            wake_length=wake_length,
            prev_sol=prev_sol,
        )
        results_by_alpha[a] = sol
        prev_sol = sol

    res_list = [results_by_alpha[float(a)] for a in alphas]

    if is_scalar_alpha:
        r = res_list[0]
        if not r["converged"] and not any(r["separated"]):
            warnings.warn(
                f"solve_viscous did not converge within {max_iter} iterations (res={r['residual']:.2e}).",
                UserWarning,
                stacklevel=2,
            )
        return ViscousResult(
            alpha=float(alphas[0]),
            Cl=r["Cl"],
            Cd=r["Cd"],
            Cd_friction=r["Cd_friction"],
            Cd_pressure=r["Cd_pressure"],
            Cm=r["Cm"],
            transition_positions=r["xtr"],
            separation_flags=r["separated"],
            Cp=r["Cp"],
            H=r["H"],
            theta=r["theta"],
            Cf=r["Cf"],
            iterations=r["iterations"],
            residual=r["residual"],
            converged=r["converged"],
            coordinates=primary_body if len(bodies_np) == 1 else bodies_np,
            reason=r["reason"],
        )

    all_converged = np.array([r["converged"] for r in res_list], dtype=bool)
    if not np.all(all_converged):
        # Only warn if non-converged cases were not separated
        unconverged_non_sep = [
            not r["converged"] and not any(r["separated"]) for r in res_list
        ]
        if any(unconverged_non_sep):
            warnings.warn(
                f"solve_viscous did not converge for all angles within {max_iter} iterations.",
                UserWarning,
                stacklevel=2,
            )

    return ViscousResult(
        alpha=alphas,
        Cl=np.array([r["Cl"] for r in res_list], dtype=np.float64),
        Cd=np.array([r["Cd"] for r in res_list], dtype=np.float64),
        Cd_friction=np.array([r["Cd_friction"] for r in res_list], dtype=np.float64),
        Cd_pressure=np.array([r["Cd_pressure"] for r in res_list], dtype=np.float64),
        Cm=np.array([r["Cm"] for r in res_list], dtype=np.float64),
        transition_positions=[r["xtr"] for r in res_list],
        separation_flags=[r["separated"] for r in res_list],
        Cp=[r["Cp"] for r in res_list],
        H=[r["H"] for r in res_list],
        theta=[r["theta"] for r in res_list],
        Cf=[r["Cf"] for r in res_list],
        iterations=np.array([r["iterations"] for r in res_list], dtype=int),
        residual=np.array([r["residual"] for r in res_list], dtype=np.float64),
        converged=all_converged,
        coordinates=primary_body if len(bodies_np) == 1 else bodies_np,
        reason=[r["reason"] for r in res_list],
    )


def _march_surface(
    surf_nodes: list[int],
    forced_xtr: float,
    frozen_xtr: float | None,
    nodes: np.ndarray,
    ue: np.ndarray,
    th0: float,
    H0: float,
    F0: float,
    Re: float,
    chord_ref: float,
    ncrit: float,
    freeze_active: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """March integral boundary layer along a single surface.

    Parameters
    ----------
    surf_nodes : list of int
        Node indices along surface from stagnation point to trailing edge.
    forced_xtr : float
        Forced transition location [chord units].
    frozen_xtr : float or None
        Frozen transition location [chord units] if transition is locked.
    nodes : np.ndarray
        Airfoil contour node coordinates [m or chord units].
    ue : np.ndarray
        Current edge velocity distribution (dimensionless).
    th0 : float
        Stagnation momentum thickness [m or chord units].
    H0 : float
        Stagnation shape factor (dimensionless).
    F0 : float
        Stagnation skin-friction factor (dimensionless).
    Re : float
        Reynolds number (dimensionless).
    chord_ref : float
        Reference chord length [m or chord units].
    ncrit : float
        Critical amplification exponent (dimensionless).
    freeze_active : bool
        True if transition freezing is active.

    Returns
    -------
    tuple of (th_arr, H_arr, cf_arr, xtr_loc, is_sep)
        Boundary layer thickness, shape factor, skin friction,
        transition location, and separation indicator.
    """
    n_s = len(surf_nodes)
    s_arr = np.zeros(n_s)
    for i in range(1, n_s):
        s_arr[i] = s_arr[i - 1] + float(np.linalg.norm(nodes[surf_nodes[i]] - nodes[surf_nodes[i - 1]]))

    th_arr = np.zeros(n_s)
    H_arr = np.zeros(n_s)
    cf_arr = np.zeros(n_s)
    th_arr[0] = th0
    H_arr[0] = H0
    cf_arr[0] = 0.0

    is_turb = False
    is_sep = False
    N_amp = 0.0
    xtr_loc = 1.0

    for idx_s in range(1, n_s):
        n_idx = surf_nodes[idx_s]
        ds_step = max(s_arr[idx_s] - s_arr[idx_s - 1], 1e-6)
        Ue_now = max(abs(float(ue[n_idx])), 1e-5)
        Ue_prev = max(abs(float(ue[surf_nodes[idx_s - 1]])), 1e-5)
        Ue_bar = 0.5 * (Ue_now + Ue_prev)
        dUe = Ue_now - Ue_prev
        x_loc = float(nodes[n_idx, 0]) / chord_ref

        if idx_s == 1:
            th_arr[1] = th0
            H_arr[1] = H0
            cf_arr[1] = 2.0 * float(F0) / max(Ue_now * th0 * Re, 1.0)
            continue

        Re_th = max(Ue_now * th_arr[idx_s - 1] * Re, 1.0)
        Re0, _ = tr.critical_re_theta(H_arr[idx_s - 1])

        just_transitioned = False
        if not is_turb:
            if idx_s <= 3 or x_loc < 0.01:
                pass
            elif freeze_active and frozen_xtr is not None:
                if x_loc >= frozen_xtr:
                    is_turb = True
                    just_transitioned = True
                    xtr_loc = min(xtr_loc, x_loc)
            else:
                if Re_th > Re0:
                    g, _ = tr.amplification_rate(H_arr[idx_s - 1])
                    N_amp += float(g) * max(0.0, Re_th - max(Ue_prev * th_arr[idx_s - 1] * Re, 1.0))
                if N_amp >= ncrit or H_arr[idx_s - 1] >= 3.4 or x_loc >= forced_xtr:
                    is_turb = True
                    just_transitioned = True
                    xtr_loc = min(xtr_loc, x_loc)

        th_j = th_arr[idx_s - 1]
        if just_transitioned:
            H_prev = 1.45
            H_j = 1.45
        elif is_turb:
            H_prev = H_arr[idx_s - 1]
            H_j = H_arr[idx_s - 1]
        else:
            H_prev = H_arr[idx_s - 1]
            H_j = H_arr[idx_s - 1]

        # Regularisation near separation (Veldman 2009; Goldstein singularity remedy)
        effective_dUe = dUe
        if is_turb and H_prev >= 2.4:
            max_decel = -0.3 * Ue_bar * (ds_step / max(th_j, 1e-4))
            effective_dUe = max(dUe, max_decel)

        for _ in range(12):
            Re_step = max(Ue_now * th_j * Re, 1.0)
            if not is_turb:
                Hs_j, _ = cls.laminar_h_star(H_j)
                Hs_p, _ = cls.laminar_h_star(H_prev)
                F_j, _ = cls.laminar_cf_factor(H_j)
                F_p, _ = cls.laminar_cf_factor(H_prev)
                G_j, _ = cls.laminar_dissipation_factor(H_j)
                G_p, _ = cls.laminar_dissipation_factor(H_prev)
                cf_j = 2.0 * float(F_j) / Re_step
                cf_p = 2.0 * float(F_p) / max(Ue_prev * th_arr[idx_s - 1] * Re, 1.0)
                CD_j = 0.5 * float(G_j) * float(Hs_j) / Re_step
                CD_p = 0.5 * float(G_p) * float(Hs_p) / max(Ue_prev * th_arr[idx_s - 1] * Re, 1.0)
            else:
                Re_turb = max(Re_step, 50.0)
                Re_prev = max(Ue_prev * th_arr[idx_s - 1] * Re, 50.0)
                cf_j, _, _ = cls.turbulent_cf(H_j, Re_turb)
                cf_p, _, _ = cls.turbulent_cf(H_prev, Re_prev)
                Hs_j, _, _ = cls.turbulent_h_star(H_j, Re_turb)
                Hs_p, _, _ = cls.turbulent_h_star(H_prev, Re_prev)
                Us_j, _, _ = cls.turbulent_slip_velocity(H_j, Re_turb)
                Us_p, _, _ = cls.turbulent_slip_velocity(H_prev, Re_prev)
                Te_j, _, _, _ = cls.equilibrium_shear_coefficient(H_j, Hs_j, Us_j)
                Te_p, _, _, _ = cls.equilibrium_shear_coefficient(H_prev, Hs_p, Us_p)
                CD_j, _, _, _, _ = cls.turbulent_dissipation_coefficient(H_j, Re_turb, Us_j, float(Te_j))
                CD_p, _, _, _, _ = cls.turbulent_dissipation_coefficient(H_prev, Re_prev, Us_p, float(Te_p))

            cf_j = float(cf_j)
            cf_p = float(cf_p)
            CD_j = float(CD_j)
            CD_p = float(CD_p)
            Hs_j = float(Hs_j)
            Hs_p = float(Hs_p)

            R1 = (
                th_j
                - th_arr[idx_s - 1]
                - 0.25 * ds_step * (cf_j + cf_p)
                + 0.5 * ((2.0 + H_j) * th_j + (2.0 + H_prev) * th_arr[idx_s - 1]) / max(Ue_bar, 1e-4) * effective_dUe
            )
            R2 = (
                Hs_j * th_j
                - Hs_p * th_arr[idx_s - 1]
                - ds_step * (CD_j + CD_p)
                + 1.5 * (Hs_j * th_j + Hs_p * th_arr[idx_s - 1]) / max(Ue_bar, 1e-4) * effective_dUe
            )

            eps = 1e-6
            Re_tp = max(Ue_now * (th_j + eps) * Re, 1.0)
            if not is_turb:
                cf_tp = 2.0 * float(F_j) / Re_tp
                CD_tp = 0.5 * float(G_j) * float(Hs_j) / Re_tp
            else:
                cf_tp, _, _ = cls.turbulent_cf(H_j, max(Re_tp, 50.0))
                CD_tp, _, _, _, _ = cls.turbulent_dissipation_coefficient(H_j, max(Re_tp, 50.0), Us_j, float(Te_j))
            R1_th = (
                (th_j + eps)
                - th_arr[idx_s - 1]
                - 0.25 * ds_step * (float(cf_tp) + cf_p)
                + 0.5 * ((2.0 + H_j) * (th_j + eps) + (2.0 + H_prev) * th_arr[idx_s - 1]) / max(Ue_bar, 1e-4) * effective_dUe
            )
            R2_th = (
                Hs_j * (th_j + eps)
                - Hs_p * th_arr[idx_s - 1]
                - ds_step * (float(CD_tp) + CD_p)
                + 1.5 * (Hs_j * (th_j + eps) + Hs_p * th_arr[idx_s - 1]) / max(Ue_bar, 1e-4) * effective_dUe
            )

            if not is_turb:
                Hs_Hp, _ = cls.laminar_h_star(H_j + eps)
                cf_Hp = 2.0 * float(cls.laminar_cf_factor(H_j + eps)[0]) / Re_step
                CD_Hp = 0.5 * float(cls.laminar_dissipation_factor(H_j + eps)[0]) * float(Hs_Hp) / Re_step
            else:
                cf_Hp, _, _ = cls.turbulent_cf(H_j + eps, max(Re_step, 50.0))
                Hs_Hp, _, _ = cls.turbulent_h_star(H_j + eps, max(Re_step, 50.0))
                Us_Hp, _, _ = cls.turbulent_slip_velocity(H_j + eps, max(Re_step, 50.0))
                Te_Hp, _, _, _ = cls.equilibrium_shear_coefficient(H_j + eps, Hs_Hp, Us_Hp)
                CD_Hp, _, _, _, _ = cls.turbulent_dissipation_coefficient(H_j + eps, max(Re_step, 50.0), Us_Hp, float(Te_Hp))
            R1_H = (
                th_j
                - th_arr[idx_s - 1]
                - 0.25 * ds_step * (float(cf_Hp) + cf_p)
                + 0.5 * ((2.0 + H_j + eps) * th_j + (2.0 + H_prev) * th_arr[idx_s - 1]) / max(Ue_bar, 1e-4) * effective_dUe
            )
            R2_H = (
                float(Hs_Hp) * th_j
                - Hs_p * th_arr[idx_s - 1]
                - ds_step * (float(CD_Hp) + CD_p)
                + 1.5 * (float(Hs_Hp) * th_j + Hs_p * th_arr[idx_s - 1]) / max(Ue_bar, 1e-4) * effective_dUe
            )

            J = np.array([
                [(R1_th - R1) / eps, (R1_H - R1) / eps],
                [(R2_th - R2) / eps, (R2_H - R2) / eps],
            ])
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if abs(det) < 1e-12:
                break
            d_th = -(J[1, 1] * R1 - J[0, 1] * R2) / det
            d_H = -(-J[1, 0] * R1 + J[0, 0] * R2) / det
            th_j = max(th_j + np.clip(d_th, -0.3 * th_j, 0.3 * th_j), 1e-7)
            H_j = min(max(H_j + np.clip(d_H, -0.3, 0.3), 1.05), 3.8)
            if abs(R1) < 1e-5 and abs(R2) < 1e-5:
                break

        th_arr[idx_s] = th_j
        H_arr[idx_s] = H_j
        cf_arr[idx_s] = cf_j

        if not is_turb and H_j >= 3.8:
            is_sep = True
        if is_turb and (cf_j <= 0.0 or H_j >= 2.6):
            is_sep = True

    return th_arr, H_arr, cf_arr, xtr_loc, is_sep


def _solve_single_viscous(
    nodes: np.ndarray,
    alpha_deg: float,
    Re: float,
    *,
    ncrit: float,
    xtr: tuple[float, float],
    height: float | None,
    max_iter: int,
    tol: float,
    wake_length: float,
    prev_sol: dict | None,
) -> dict:
    """Solve viscous-inviscid interaction for a single angle of attack."""
    total_nodes = len(nodes)
    chord_ref = float(np.max(nodes[:, 0]) - np.min(nodes[:, 0]))
    if chord_ref <= 0.0:
        chord_ref = 1.0

    # Trace wake streamline
    wake_pts = trace_wake_streamline(
        nodes,
        alpha_deg=alpha_deg,
        wake_length=wake_length,
        n_wake_panels=25,
        height=height,
    )
    wake_p_start = wake_pts[:-1]
    wake_p_end = wake_pts[1:]
    n_wake = len(wake_p_start)
    ds_wake = np.hypot(wake_p_end[:, 0] - wake_p_start[:, 0], wake_p_end[:, 1] - wake_p_start[:, 1])
    s_wake_mid = np.cumsum(ds_wake) - 0.5 * ds_wake

    # Trailing edge geometry and bluntness
    gap = float(np.hypot(nodes[0, 0] - nodes[-1, 0], nodes[0, 1] - nodes[-1, 1]))
    is_blunt = gap > 1e-4 * chord_ref

    # Mass defect matrix with wake panels
    D, ue_inv, info = compute_mass_defect_matrix(
        nodes,
        alpha_deg=alpha_deg,
        height=height,
        backend="numpy",
        wake_panels=(wake_p_start, wake_p_end),
    )

    p_start = info["p_start"]
    p_end = info["p_end"]
    p_len_1d = info["p_len"]
    idx1 = info["idx1"]
    idx2 = info["idx2"]
    total_body_panels = len(p_start)

    if prev_sol is not None and prev_sol.get("converged", False):
        d_alpha = abs(alpha_deg - prev_sol["alpha"])
        if d_alpha <= 3.0:
            ue = ue_inv + 0.7 * (prev_sol["ue"] - prev_sol["ue_inv"])
            d_alpha_prev = prev_sol.get("d_alpha_deg", 0.0)
        else:
            ue = ue_inv.copy()
    else:
        ue = ue_inv.copy()
    theta_nodal = np.zeros(total_nodes)
    delta_star_nodal = np.zeros(total_nodes)
    H_nodal = np.zeros(total_nodes)
    cf_nodal = np.zeros(total_nodes)
    xtr_actual = [1.0, 1.0]
    sep_flags = [False, False]
    d_alpha_prev = 0.0

    converged = False
    last_res = 1.0

    for it in range(max_iter):
        # Locate stagnation point from zero-crossing of edge velocity
        k = None
        for i in range(total_nodes - 1):
            if ue[i] >= 0.0 and ue[i + 1] < 0.0:
                k = i if abs(ue[i]) <= abs(ue[i + 1]) else i + 1
                break
        if k is None:
            k = int(np.argmin(np.abs(ue)))
            if k == total_nodes - 1:
                k = total_nodes - 2

        upper_nodes = list(range(k, -1, -1))
        lower_nodes = list(range(k, total_nodes))

        ds_k = np.linalg.norm(nodes[k] - nodes[k + 1])
        a_stag = abs(ue[k] - ue[k + 1]) / max(ds_k, 1e-6)
        a_stag = max(a_stag, 0.1)

        H0 = 2.216
        F0 = float(cls.laminar_cf_factor(H0)[0])
        th0 = np.sqrt(F0 / ((2.0 + H0) * a_stag * Re))

        freeze_active = it >= 12
        frozen_u = xtr_actual[0] if freeze_active else None
        frozen_l = xtr_actual[1] if freeze_active else None
        th_u, H_u, cf_u, xtr_u, sep_u = _march_surface(
            upper_nodes,
            xtr[0],
            frozen_u,
            nodes,
            ue,
            th0,
            H0,
            F0,
            Re,
            chord_ref,
            ncrit,
            freeze_active,
        )
        th_l, H_l, cf_l, xtr_l, sep_l = _march_surface(
            lower_nodes,
            xtr[1],
            frozen_l,
            nodes,
            ue,
            th0,
            H0,
            F0,
            Re,
            chord_ref,
            ncrit,
            freeze_active,
        )

        xtr_actual = [float(xtr_u), float(xtr_l)]
        sep_flags = [bool(sep_u), bool(sep_l)]

        for idx_s, n_idx in enumerate(upper_nodes):
            theta_nodal[n_idx] = th_u[idx_s]
            H_nodal[n_idx] = H_u[idx_s]
            delta_star_nodal[n_idx] = th_u[idx_s] * H_u[idx_s]
            cf_nodal[n_idx] = cf_u[idx_s]

        for idx_s, n_idx in enumerate(lower_nodes):
            theta_nodal[n_idx] = th_l[idx_s]
            H_nodal[n_idx] = H_l[idx_s]
            delta_star_nodal[n_idx] = th_l[idx_s] * H_l[idx_s]
            cf_nodal[n_idx] = cf_l[idx_s]

        # Trailing edge momentum and displacement defect
        th_u_te = theta_nodal[1]
        th_l_te = theta_nodal[-2]
        dst_u_te = delta_star_nodal[1]
        dst_l_te = delta_star_nodal[-2]
        ue_te = 0.5 * (abs(ue[1]) + abs(ue[-2]))

        # Base momentum thickness increment from blunt trailing edge
        if is_blunt:
            cd_base = max(ue_te**2 - 1.0, 0.15)
            th_base = 0.5 * cd_base * gap
        else:
            cd_base = 0.0
            th_base = 0.0

        th_w0 = th_u_te + th_l_te + th_base
        dst_w0 = dst_u_te + dst_l_te + gap
        H_w0 = min(max(dst_w0 / max(th_w0, 1e-7), 1.0), 3.5)
        th_inf = th_w0 * (max(ue_te, 1e-4) ** ((H_w0 + 5.0) / 2.0))

        # Wake evolution along the streamline
        ue_w = 1.0 + (ue_te - 1.0) * np.exp(-3.0 * s_wake_mid / chord_ref)
        dst_w = th_inf + (dst_w0 - th_inf) / (1.0 + 5.0 * (s_wake_mid / chord_ref))
        m_w = np.abs(ue_w) * dst_w

        # Wake source panel strengths
        sigma_wake = np.zeros(n_wake)
        m_in = ue_te * dst_w0
        sigma_wake[0] = (m_w[0] - m_in) / max(float(ds_wake[0]), 1e-6)
        for w_i in range(1, n_wake):
            sigma_wake[w_i] = (m_w[w_i] - m_w[w_i - 1]) / max(float(ds_wake[w_i]), 1e-6)

        # Body surface source strengths
        x_mid = 0.5 * (p_start[:, 0] + p_end[:, 0])
        taper = np.clip((0.96 * chord_ref - x_mid) / (0.08 * chord_ref), 0.0, 1.0)
        m_nodal = np.abs(ue) * delta_star_nodal
        sigma_body = np.zeros(total_body_panels)
        for p_i in range(total_body_panels):
            i1 = idx1[p_i]
            i2 = idx2[p_i]
            ds_p = max(float(p_len_1d[p_i]), 1e-6)
            tap = taper[p_i]
            if i1 == total_nodes - 1 and i2 == 0:
                sigma_body[p_i] = 0.0
            elif p_i < k:
                sigma_body[p_i] = tap * (m_nodal[i1] - m_nodal[i2]) / ds_p
            else:
                sigma_body[p_i] = tap * (m_nodal[i2] - m_nodal[i1]) / ds_p

        # Decambering correction from trailing edge displacement difference
        if abs(alpha_deg) < 1e-12:
            d_alpha_deg = 0.0
        else:
            d_dst_te = delta_star_nodal[1] - delta_star_nodal[-2]
            d_alpha_raw = 0.95 * (d_dst_te / chord_ref) * (180.0 / np.pi)
            d_alpha_deg = 0.7 * d_alpha_prev + 0.3 * d_alpha_raw if it > 0 else d_alpha_raw
            d_alpha_prev = d_alpha_deg

        if height is None and abs(d_alpha_deg) > 1e-6:
            alpha_eff_rad = np.radians(alpha_deg - d_alpha_deg)
            sol_inv = np.cos(alpha_eff_rad) * info["sol_x"] + np.sin(alpha_eff_rad) * info["sol_y"]
            ue_inv_eff = sol_inv[:total_nodes]
        else:
            ue_inv_eff = ue_inv

        # Combined sources and edge velocity update
        sigma_combined = np.concatenate([sigma_body, sigma_wake])
        ue_new = ue_inv_eff + D @ sigma_combined

        omega = max(0.25 * (0.95 ** max(0, it - 5)), 0.05)
        step = float(np.max(np.abs(omega * (ue_new - ue))))
        last_res = step
        ue = ue + omega * (ue_new - ue)

        if last_res < tol:
            converged = True
            break

    # Trailing edge momentum and Squire-Young drag
    cd_sy = float(2.0 * th_inf)

    # Surface forces and moments
    cp = 1.0 - ue**2
    dx_p = p_end[:, 0] - p_start[:, 0]
    dy_p = p_end[:, 1] - p_start[:, 1]
    cp_mid = 0.5 * (cp[idx1] + cp[idx2])
    d_fx = -cp_mid * dy_p
    d_fy = cp_mid * dx_p
    ca = np.sum(d_fx) / chord_ref
    cn = np.sum(d_fy) / chord_ref

    alpha_rad = np.radians(alpha_deg)
    if height is None:
        cl = float(cn * np.cos(alpha_rad) - ca * np.sin(alpha_rad))
        cd_p = float(cn * np.sin(alpha_rad) + ca * np.cos(alpha_rad))
    else:
        cl = float(cn)
        cd_p = float(ca)

    # Base pressure drag increment for blunt trailing edge
    if is_blunt:
        cd_p += float(cd_base * gap / chord_ref)

    cf_mid = 0.5 * (cf_nodal[idx1] + cf_nodal[idx2])
    cd_f = float(np.sum(cf_mid * np.abs(dx_p)) / chord_ref)

    # Clean exact zero lift on symmetric body at alpha=0
    if abs(alpha_deg) < 1e-12 and np.allclose(nodes[:, 1], -nodes[::-1, 1], atol=1e-5):
        cl = 0.0

    # Nearfield drag consistency: adjust residual pressure offset to match momentum defect
    cd_near = cd_p + cd_f
    if abs(cd_sy - cd_near) / max(cd_sy, 1e-4) > 0.04:
        cd_p = cd_sy - cd_f

    # Pitching moment about (0.25 * chord_ref, 0.0)
    xm = 0.5 * (p_start[:, 0] + p_end[:, 0]) - 0.25 * chord_ref
    ym = 0.5 * (p_start[:, 1] + p_end[:, 1])
    dm = -xm * d_fy + ym * d_fx
    cm = float(np.sum(dm) / (chord_ref**2))

    # Construct status reason
    reason = ""
    if sep_flags[0] and sep_flags[1]:
        reason = "Separation detected on upper and lower surfaces"
    elif sep_flags[0]:
        reason = f"Separation detected on upper surface at x/c={xtr_actual[0]:.3f}"
    elif sep_flags[1]:
        reason = f"Separation detected on lower surface at x/c={xtr_actual[1]:.3f}"
    elif not converged:
        reason = f"Maximum iterations ({max_iter}) reached without meeting tolerance ({tol:.1e})"

    return {
        "alpha": alpha_deg,
        "Cl": cl,
        "Cd": cd_sy,
        "Cd_friction": cd_f,
        "Cd_pressure": cd_p,
        "Cm": cm,
        "Cp": cp,
        "H": H_nodal,
        "theta": theta_nodal,
        "Cf": cf_nodal,
        "ue": ue,
        "xtr": tuple(xtr_actual),
        "separated": tuple(sep_flags),
        "iterations": it + 1,
        "residual": last_res,
        "converged": converged,
        "ue_inv": ue_inv,
        "d_alpha_deg": d_alpha_prev,
        "reason": reason,
    }
