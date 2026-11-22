# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Closed-form 3D semi-infinite harmonic far-wake tail.

When a discrete harmonic wake of ``N_wx`` rows ends at distance ``L_w = N_wx dx_w``
behind the trailing-edge ring closure, this module computes the exact velocity
and normal-velocity influence of the discarded semi-infinite harmonic wake
sheet on ``xi in [0, inf)`` behind each strip.

Mathematical derivation
-----------------------
Consider one wake strip starting at corners ``(p_L, p_R)`` [m] and extending
along the unit wake direction ``d`` [-] for ``xi in [0, inf)`` with harmonic
ring (doublet) circulation

    Gamma_tail(xi) = Gamma_0 exp(-i kappa xi),    kappa = omega / V_inf = 2 k / c_ref [1/m].

Matching the discrete wake ring midpoints ``x_m = x_TE + (m - 0.5) dx_w`` with
circulations ``Gamma_TE exp(-i omega m dt)`` sets the phase anchor at ``xi = 0``
to ``Gamma_0 = Gamma_TE exp(-i omega (N_wx + 0.5) dt)``.

By Helmholtz vorticity conservation (Stokes' theorem on the semi-infinite
parallelogram ``X(xi, eta) = p_L + xi d + eta (p_R - p_L)``, ``eta in [0, 1]``),
this harmonic doublet sheet decomposes without far-field approximation into
three parts (per unit ``Gamma_0``):

1. **Closing spanwise segment** at ``xi = 0`` from ``p_L`` to ``p_R`` with
   unit strength, evaluated by the finite-segment Biot-Savart law
   (:func:`ventorum.aero.vortex.segment_velocity`). At ``k = 0``, ``Gamma_0 = Gamma_TE``
   and this segment cancels the trailing segment of the last discrete wake ring.
2. **Two semi-infinite oscillatory trailing filaments** along ``+d`` from ``p_R``
   (strength ``+exp(-i kappa xi)``) and ``p_L`` (strength ``-exp(-i kappa xi)``).
   With ``r_e = P - p_e``, ``a_e = -(r_e . d)`` and ``rho_e^2 = |d x r_e|^2``:

       v_trail,e(P) = +/- (1 / 4 pi) (d x r_e) int_0^inf exp(-i kappa xi) / ((xi + a_e)^2 + rho_e^2)^(3/2) dxi.

   At ``kappa = 0`` this equals the exact semi-infinite steady vortex ray.
3. **Continuous oscillatory shed vorticity sheet** of density
   ``gamma_shed(xi) = dGamma_tail/dxi = -i kappa exp(-i kappa xi)`` along the
   unit span vector ``u_s = (p_R - p_L) / |p_R - p_L|``. Integrating along the
   span segment analytically at fixed ``xi`` yields

       v_span(P) = (-i kappa / 4 pi) int_0^inf exp(-i kappa xi) (v_0 - xi v_1)
                   * [ (-t_R + xi (d . u_s)) / rad_R(xi) - (-t_L + xi (d . u_s)) / rad_L(xi) ]
                   / q_perp^2(xi) dxi,

   where ``v_0 = u_s x r_L``, ``v_1 = u_s x d``, ``t_L = r_L . u_s``,
   ``t_R = r_R . u_s``, ``rad_e(xi) = sqrt((xi + a_e)^2 + rho_e^2)``, and
   ``q_perp^2(xi) = |v_0|^2 - 2 xi (v_0 . v_1) + xi^2 |v_1|^2``.

Contour rotation
----------------
When the strip start is strictly downstream of ``P`` along ``d``
(``a_min = min(a_L, a_R, -(v_0 . v_1) / |v_1|^2) > 0``), all poles and
square-root branch points lie strictly in the left half-plane ``Re(xi) < 0``.
The fourth quadrant ``Re(xi) >= 0, Im(xi) <= 0`` is analytic and the principal
square root never crosses its branch cut. Rotating the path to
``xi = -i u`` (``u in [0, inf)``, ``dxi = -i du``) turns ``exp(-i kappa xi)``
into the real decaying exponential ``exp(-kappa u) in (0, 1]``, eliminating
oscillation and overflow at all reduced frequencies and ground heights. The
substitution ``u = L tan(theta)`` with ``L = min(a_min, 2 / kappa)`` maps
``[0, inf)`` to ``[0, pi/2)`` for 32-point Gauss-Legendre quadrature.

For the rare case where a downstream evaluation point lies aft of the tail
start (``a_min <= 0``), the integral splits into a finite real segment
``[0, xi_0]`` and a contour-rotated tail on ``[xi_0, inf)`` with
``xi_0 = -a_min + max(0.5, 1 / kappa)``.

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, chapters 10 and 13 (Biot-Savart law and unsteady
  vortex-ring wakes).
* K. C. Hall, "Eigenanalysis of unsteady flows about airfoils, cascades, and
  wings", AIAA Journal 32(12), 1994, pp. 2426-2432 (harmonic wake relation).
"""

from __future__ import annotations

import math

import numpy as np

from ventorum.aero.vortex import segment_velocity
from ventorum.aero.vortex_ring import steady_wake_velocity
from ventorum.utils.vec import cross3

_INV_4PI = 1.0 / (4.0 * math.pi)
# Target byte budget for temporary (chunk_m, n_strips, Q) complex arrays.
_TAIL_CHUNK_BYTES = 4_000_000

# 32-point Gauss-Legendre quadrature on [0, pi/2] (for contour u = L tan(theta))
# and on [0, 1] (for the finite real segment when a_min <= 0).
_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)
_TH_32 = 0.25 * np.pi * (_GL_X + 1.0)
_W_TH_32 = 0.25 * np.pi * _GL_W
_TAN_32 = np.tan(_TH_32)
_SEC2_W_32 = _W_TH_32 / (np.cos(_TH_32) ** 2)
_U01_32 = 0.5 * (_GL_X + 1.0)
_W01_32 = 0.5 * _GL_W


def _broadcast_wake_dir(wake_dir: np.ndarray, n: int) -> np.ndarray:
    """Validate and broadcast *wake_dir* to unit vectors of shape ``(n, 3)``."""
    wd = np.asarray(wake_dir, dtype=np.float64)
    if wd.ndim == 1:
        if wd.shape != (3,):
            raise ValueError(f"Expected wake_dir of shape (3,) or ({n}, 3), got {wd.shape}.")
        wd = np.broadcast_to(wd, (n, 3))
    elif wd.ndim != 2 or wd.shape != (n, 3):
        raise ValueError(f"Expected wake_dir of shape (3,) or ({n}, 3), got {wd.shape}.")
    norms = np.linalg.norm(wd, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("wake_dir must have non-zero length for every strip.")
    return np.ascontiguousarray(wd / norms)


def _slice_rc(rc: np.ndarray | float, i0: int, i1: int) -> np.ndarray | float:
    """Slice *rc* along the evaluation-point axis when it is 2-D."""
    if np.ndim(rc) == 2:
        return np.asarray(rc, dtype=np.float64)[i0:i1]
    return rc


def _contour_integrals(
    a_L: np.ndarray,
    a_R: np.ndarray,
    rho2_L: np.ndarray,
    rho2_R: np.ndarray,
    v0_sq: np.ndarray,
    v0_dot_v1: np.ndarray,
    v1_sq: np.ndarray,
    tL_0: np.ndarray,
    tR_0: np.ndarray,
    d_dot_us: np.ndarray,
    L_scale: np.ndarray,
    kappa: float,
    xi_shift: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the four scalar tail integrals on ``xi = xi_shift - i u``."""
    u = L_scale[..., None] * _TAN_32
    du_w = L_scale[..., None] * _SEC2_W_32
    exp_arg = np.minimum(kappa * u, 700.0)
    wt = (-1j) * du_w * np.exp(-exp_arg)
    if np.ndim(xi_shift) > 0 or float(xi_shift) != 0.0:
        xi_s = np.asarray(xi_shift, dtype=np.float64)[..., None]
        wt = wt * np.exp(-1j * kappa * xi_s)
        xi = xi_s - 1j * u
    else:
        xi = -1j * u

    sL = xi + a_L[..., None]
    sR = xi + a_R[..., None]
    r2L = sL * sL + rho2_L[..., None]
    r2R = sR * sR + rho2_R[..., None]
    radL = np.sqrt(r2L)
    radR = np.sqrt(r2R)

    I_tr_R = np.sum(wt / (radR * r2R), axis=-1)
    I_tr_L = np.sum(wt / (radL * r2L), axis=-1)

    q_perp2 = v0_sq[..., None] - 2.0 * xi * v0_dot_v1[..., None] + (xi * xi) * v1_sq[..., None]
    xi_dus = xi * d_dot_us[..., None]
    num_R = -tR_0[..., None] + xi_dus
    num_L = -tL_0[..., None] + xi_dus
    span_br = (num_R / radR - num_L / radL) / q_perp2

    I_sp0 = np.sum(wt * span_br, axis=-1)
    I_sp1 = np.sum((wt * xi) * span_br, axis=-1)
    return I_tr_R, I_tr_L, I_sp0, I_sp1


def _real_segment_integrals(
    a_L: np.ndarray,
    a_R: np.ndarray,
    rho2_L: np.ndarray,
    rho2_R: np.ndarray,
    v0_sq: np.ndarray,
    v0_dot_v1: np.ndarray,
    v1_sq: np.ndarray,
    tL_0: np.ndarray,
    tR_0: np.ndarray,
    d_dot_us: np.ndarray,
    xi_0: np.ndarray,
    kappa: float,
    rc2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the four scalar tail integrals on the real interval ``[0, xi_0]``."""
    xi = xi_0[..., None] * _U01_32
    wt = (xi_0[..., None] * _W01_32) * np.exp(-1j * kappa * xi)

    sL = xi + a_L[..., None]
    sR = xi + a_R[..., None]
    r2L = sL * sL + rho2_L[..., None] + rc2[..., None]
    r2R = sR * sR + rho2_R[..., None] + rc2[..., None]
    radL = np.sqrt(r2L)
    radR = np.sqrt(r2R)

    I_tr_R = np.sum(wt / (radR * r2R), axis=-1)
    I_tr_L = np.sum(wt / (radL * r2L), axis=-1)

    q_perp2 = (
        v0_sq[..., None]
        - 2.0 * xi * v0_dot_v1[..., None]
        + (xi * xi) * v1_sq[..., None]
        + rc2[..., None]
    )
    xi_dus = xi * d_dot_us[..., None]
    num_R = -tR_0[..., None] + xi_dus
    num_L = -tL_0[..., None] + xi_dus
    span_br = (num_R / radR - num_L / radL) / q_perp2

    I_sp0 = np.sum(wt * span_br, axis=-1)
    I_sp1 = np.sum((wt * xi) * span_br, axis=-1)
    return I_tr_R, I_tr_L, I_sp0, I_sp1


def harmonic_tail_velocity(
    P: np.ndarray,
    p_left: np.ndarray,
    p_right: np.ndarray,
    wake_dir: np.ndarray,
    kappa: float,
    rc: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Compute velocity at *P* from unit-strength semi-infinite harmonic wake strips.

    Each strip ``s`` starts at ``(p_left[s], p_right[s])`` [m], extends along
    ``wake_dir[s]`` to infinity, and carries doublet circulation
    ``exp(-i kappa xi)`` [m^2/s] at downstream distance ``xi >= 0`` [m].
    Multiply the returned tensor by the strip phase anchor
    ``exp(-i omega (N_wx + 0.5) dt)`` to attach the tail after ``N_wx``
    discrete wake rows.

    Parameters
    ----------
    P : numpy.ndarray
        Evaluation points [m], shape ``(M, 3)``.
    p_left, p_right : numpy.ndarray
        Left and right start corners of each tail strip [m], shape ``(N_s, 3)``.
    wake_dir : numpy.ndarray
        Unit wake direction [-], shape ``(3,)`` or ``(N_s, 3)``.
    kappa : float
        Spatial wavenumber ``omega / V_inf = 2 k / c_ref`` [1/m], must be ``>= 0``.
    rc : numpy.ndarray or float, optional
        Core radius [m], scalar, shape ``(N_s,)`` or shape ``(M, N_s)``.

    Returns
    -------
    numpy.ndarray
        Complex induced velocity per unit circulation at ``xi = 0`` [1/m],
        shape ``(M, N_s, 3)``.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    pl = np.asarray(p_left, dtype=np.float64)
    pr = np.asarray(p_right, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[1] != 3:
        raise ValueError(f"Expected P of shape (M, 3), got {P_arr.shape}.")
    if pl.ndim != 2 or pl.shape[1] != 3 or pr.shape != pl.shape:
        raise ValueError(f"Expected p_left and p_right of shape (N_s, 3), got {pl.shape} and {pr.shape}.")
    kap = float(kappa)
    if not math.isfinite(kap) or kap < 0.0:
        raise ValueError(f"kappa must be finite and >= 0, got {kappa!r}.")

    m, n = P_arr.shape[0], pl.shape[0]
    wd = _broadcast_wake_dir(wake_dir, n)
    if m == 0 or n == 0:
        return np.empty((m, n, 3), dtype=np.complex128)

    if kap == 0.0:
        return steady_wake_velocity(P_arr, pl, pr, wd, rc).astype(np.complex128)

    l_span = pr - pl
    span_len = np.linalg.norm(l_span, axis=1)
    u_s = l_span / np.maximum(span_len[:, None], 1e-300)
    v1 = cross3(u_s, wd)                          # (n, 3)
    v1_sq = np.einsum("si,si->s", v1, v1)         # (n,)
    d_dot_us = np.einsum("si,si->s", wd, u_s)     # (n,)

    out = np.empty((m, n, 3), dtype=np.complex128)
    step = max(1, min(m, _TAIL_CHUNK_BYTES // max(1, n * 32 * 16)))
    span_pref = (-1j * kap) * _INV_4PI

    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        Pc = P_arr[i0:i1]
        rc_c = _slice_rc(rc, i0, i1)

        # Part 1: closing spanwise segment at xi = 0 (p_left -> p_right).
        v_tot = segment_velocity(Pc, pl, pr, rc_c).astype(np.complex128)

        r_L = Pc[:, None, :] - pl[None, :, :]
        r_R = Pc[:, None, :] - pr[None, :, :]
        a_L = -np.einsum("msi,si->ms", r_L, wd)
        a_R = -np.einsum("msi,si->ms", r_R, wd)

        d_x_rL = cross3(wd[None, :, :], r_L)
        d_x_rR = cross3(wd[None, :, :], r_R)
        rho2_L = np.einsum("msi,msi->ms", d_x_rL, d_x_rL)
        rho2_R = np.einsum("msi,msi->ms", d_x_rR, d_x_rR)

        v0 = cross3(u_s[None, :, :], r_L)
        v0_sq = np.einsum("msi,msi->ms", v0, v0)
        v0_dot_v1 = np.einsum("msi,si->ms", v0, v1)
        xi_min = v0_dot_v1 / np.maximum(v1_sq[None, :], 1e-300)
        tL_0 = np.einsum("msi,si->ms", r_L, u_s)
        tR_0 = np.einsum("msi,si->ms", r_R, u_s)

        a_min = np.minimum(np.minimum(a_L, a_R), -xi_min)
        v1_sq_2d = np.broadcast_to(v1_sq[None, :], a_L.shape)
        dus_2d = np.broadcast_to(d_dot_us[None, :], a_L.shape)

        if bool(np.all(a_min > 0.0)):
            L_scale = np.minimum(a_min, 2.0 / kap)
            I_tr_R, I_tr_L, I_sp0, I_sp1 = _contour_integrals(
                a_L, a_R, rho2_L, rho2_R, v0_sq, v0_dot_v1, v1_sq_2d,
                tL_0, tR_0, dus_2d, L_scale, kap,
            )
        else:
            I_tr_R = np.empty(a_L.shape, dtype=np.complex128)
            I_tr_L = np.empty(a_L.shape, dtype=np.complex128)
            I_sp0 = np.empty(a_L.shape, dtype=np.complex128)
            I_sp1 = np.empty(a_L.shape, dtype=np.complex128)
            pos = a_min > 0.0
            if np.any(pos):
                L_p = np.minimum(a_min[pos], 2.0 / kap)
                rR, rL, s0, s1 = _contour_integrals(
                    a_L[pos], a_R[pos], rho2_L[pos], rho2_R[pos],
                    v0_sq[pos], v0_dot_v1[pos], v1_sq_2d[pos],
                    tL_0[pos], tR_0[pos], dus_2d[pos], L_p, kap,
                )
                I_tr_R[pos], I_tr_L[pos], I_sp0[pos], I_sp1[pos] = rR, rL, s0, s1
            neg = ~pos
            if np.any(neg):
                pad = max(0.5, 1.0 / kap)
                xi_0 = -a_min[neg] + pad
                rc_mat = np.broadcast_to(np.asarray(rc_c, dtype=np.float64), a_L.shape)
                rc2_n = np.maximum(rc_mat[neg] ** 2, 1e-24)
                rR_1, rL_1, s0_1, s1_1 = _real_segment_integrals(
                    a_L[neg], a_R[neg], rho2_L[neg], rho2_R[neg],
                    v0_sq[neg], v0_dot_v1[neg], v1_sq_2d[neg],
                    tL_0[neg], tR_0[neg], dus_2d[neg], xi_0, kap, rc2_n,
                )
                L_n = np.minimum(a_min[neg] + xi_0, 2.0 / kap)
                rR_2, rL_2, s0_2, s1_2 = _contour_integrals(
                    a_L[neg], a_R[neg], rho2_L[neg], rho2_R[neg],
                    v0_sq[neg], v0_dot_v1[neg], v1_sq_2d[neg],
                    tL_0[neg], tR_0[neg], dus_2d[neg], L_n, kap, xi_shift=xi_0,
                )
                I_tr_R[neg] = rR_1 + rR_2
                I_tr_L[neg] = rL_1 + rL_2
                I_sp0[neg] = s0_1 + s0_2
                I_sp1[neg] = s1_1 + s1_2

        # Part 2: oscillatory semi-infinite trailing filaments (+right, -left).
        v_tot += _INV_4PI * (I_tr_R[:, :, None] * d_x_rR - I_tr_L[:, :, None] * d_x_rL)
        # Part 3: continuous oscillatory spanwise shed vorticity sheet.
        v_tot += span_pref * (I_sp0[:, :, None] * v0 - I_sp1[:, :, None] * v1[None, :, :])
        out[i0:i1] = v_tot

    return out


def harmonic_tail_normal_influence(
    P: np.ndarray,
    normals: np.ndarray,
    p_left: np.ndarray,
    p_right: np.ndarray,
    wake_dir: np.ndarray,
    kappa: float,
    rc: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Compute normal-velocity influence of unit-strength semi-infinite harmonic wake strips.

    Entry ``(i, s)`` is the dot product of :func:`harmonic_tail_velocity` at
    ``P[i]`` from strip ``s`` with ``normals[i]``.

    Parameters
    ----------
    P : numpy.ndarray
        Evaluation points [m], shape ``(M, 3)``.
    normals : numpy.ndarray
        Unit normal vectors at *P* [-], shape ``(M, 3)``.
    p_left, p_right : numpy.ndarray
        Left and right start corners of each tail strip [m], shape ``(N_s, 3)``.
    wake_dir : numpy.ndarray
        Unit wake direction [-], shape ``(3,)`` or ``(N_s, 3)``.
    kappa : float
        Spatial wavenumber ``omega / V_inf = 2 k / c_ref`` [1/m], must be ``>= 0``.
    rc : numpy.ndarray or float, optional
        Core radius [m], scalar, shape ``(N_s,)`` or shape ``(M, N_s)``.

    Returns
    -------
    numpy.ndarray
        Complex normal-velocity influence matrix [1/m], shape ``(M, N_s)``.
    """
    n_arr = np.asarray(normals, dtype=np.float64)
    P_arr = np.asarray(P, dtype=np.float64)
    if n_arr.shape != P_arr.shape:
        raise ValueError(f"Expected normals of shape {P_arr.shape}, got {n_arr.shape}.")
    v = harmonic_tail_velocity(P_arr, p_left, p_right, wake_dir, kappa, rc)
    return np.einsum("msi,mi->ms", v, n_arr)
