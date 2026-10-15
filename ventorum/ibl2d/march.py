# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Direct integral boundary-layer march for Ventorum-2D.

The march integrates the two-equation integral boundary layer in direct
mode (edge velocity given) along one surface, from the leading edge or
the stagnation point to the trailing edge, in incompressible flow only.
It is the reference for the later viscous-inviscid coupling: it gives
the momentum thickness, the shape factor, the skin friction and the
transpiration velocity for a prescribed edge velocity.

The state has the momentum thickness ``theta`` [m] and the kinematic
shape factor ``H`` (dimensionless). Laminar flow adds the amplification
exponent ``N`` (dimensionless) of the e^N envelope; turbulent flow adds
the maximum shear-stress coefficient ``Ctau`` (dimensionless). The
momentum equation and the kinetic-energy shape-factor equation close
with :mod:`ventorum.ibl2d.closure`; transition closes with
:mod:`ventorum.ibl2d.transition`.

Sources (papers only; no program code was read or used):

* M. Drela and M. B. Giles, "Viscous-inviscid analysis of transonic and
  low Reynolds number airfoils", AIAA Journal 25(10), 1987, pp.
  1347-1355: the two-equation integral formulation with the lag equation
  (Eqs. 9-11 in the paper) and the e^N transition model.
* M. Drela, doctoral thesis, MIT Gas Turbine Laboratory Report 187,
  1986: the Falkner-Skan closure fits and the critical Reynolds number
  and envelope slope of the transition model.
* E. Hiemenz, "Die Grenzschicht an einem in den gleichfoermigen
  Fluessigkeitsstrom eingetauchten geraden Kreiszylinder", Dinglers
  Polytechnisches Journal 326, 1911: the stagnation-point similarity
  solution used for the start.
* H. Blasius, "Grenzschichten in Fluessigkeiten mit kleiner Reibung",
  Zeitschrift fuer Mathematik und Physik 56, 1908: the flat-plate
  similarity solution used for the start.
* F. R. Houwink and A. E. P. Veldman, "Steady and unsteady separated
  flow computations for transonic airfoils", AIAA paper 83-1790, 1983:
  the interpolation of the transition point inside a step (used here
  for both natural and forced transition).

The equation labels (M1), (M2), (M3) below point to
``docs/theory/ibl2d_march.md``. All lengths use the same unit as the
input arc length ``s``; the reference Reynolds number is
``Re_ref = V_ref L / nu`` with ``L`` equal to one unit of ``s``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

H_BLASIUS = 2.591
H_LAM_SEP = 4.0


@dataclass
class BoundaryLayerResult:
    """Result of the direct march on one surface.

    Parameters
    ----------
    s : array_like
        Arc length stations [m or chord units], length N.
    ue : array_like
        Edge velocity ratio (dimensionless), length N.
    theta : array_like
        Momentum thickness [m or chord units], length N.
    delta_star : array_like
        Displacement thickness [m or chord units], length N.
    H : array_like
        Kinematic shape factor (dimensionless), length N.
    Cf : array_like
        Skin-friction coefficient (dimensionless), length N.
    Ctau : array_like
        Maximum shear-stress coefficient (dimensionless), length N.
    N : array_like
        Amplification exponent (dimensionless), length N.
    transp : array_like
        Transpiration velocity ``d(ue delta_star)/ds`` (dimensionless).
    x_transition : float or None
        Transition position [same unit as ``s``] or None if laminar.
    x_separation : float or None
        Separation position [same unit as ``s``] or None if attached.
    separated : bool
        True when the march stopped at separation.
    n_valid : int
        Number of valid stations (entries after it hold NaN).
    """

    s: Any = None
    ue: Any = None
    theta: Any = None
    delta_star: Any = None
    H: Any = None
    Cf: Any = None
    Ctau: Any = None
    N: Any = None
    transp: Any = None
    x_transition: Any = None
    x_separation: Any = None
    separated: bool = False
    n_valid: int = 0


def _is_torch(xp) -> bool:
    """Return True when ``xp`` is the torch module."""
    return getattr(xp, "__name__", "") == "torch"


def _to_xp(xp, values, dtype=None, device=None):
    """Return ``values`` as an array of module ``xp``."""
    import numpy as np

    if xp is None or xp is np:
        return np.asarray(values, dtype=float)
    torch = xp
    if isinstance(values, torch.Tensor):
        out = values
        if dtype is not None:
            out = out.to(dtype)
        if device is not None:
            out = out.to(device)
        return out
    arr = np.asarray(values, dtype=float)
    kw = {}
    if dtype is not None:
        kw["dtype"] = dtype
    if device is not None:
        kw["device"] = device
    return torch.as_tensor(arr, **kw)


def _full(xp, shape, value, like=None, dtype=None, device=None):
    """Return a full array of module ``xp``."""
    import numpy as np

    if xp is None or xp is np:
        return np.full(shape, value, dtype=float)
    torch = xp
    if dtype is None:
        dtype = like.dtype if isinstance(like, torch.Tensor) else torch.float64
    if device is None:
        device = like.device if isinstance(like, torch.Tensor) else "cpu"
    return torch.full(shape, value, dtype=dtype, device=device)


def _gradient(xp, y, x):
    """Return dy/dx with central differences inside, one-sided at ends.

    Parameters
    ----------
    y : array_like
        Field with shape (B, N) (quantity with any unit).
    x : array_like
        Stations [m or chord units] with shape (N,).

    Returns
    -------
    array_like
        Derivative with the unit of ``y`` per [m or chord units].
    """
    dydx = xp.empty_like(y)
    dx1 = x[1] - x[0]
    dxn = x[-1] - x[-2]
    dydx[..., 0] = (y[..., 1] - y[..., 0]) / dx1
    dydx[..., -1] = (y[..., -1] - y[..., -2]) / dxn
    dx = x[2:] - x[:-2]
    dydx[..., 1:-1] = (y[..., 2:] - y[..., :-2]) / dx
    return dydx


def _hiemenz_state():
    """Return (H, F, G) of the stagnation similarity point.

    Returns
    -------
    tuple
        ``(H, F, G)`` with ``H`` dimensionless, ``F = Re_theta Cf / 2``
        dimensionless and ``G = Re_theta 2 CD / Hstar`` dimensionless.
    """
    import numpy as np

    from ventorum.ibl2d import closure as cls

    def ratio(h):
        f = float(np.asarray(cls.laminar_cf_factor(h)[0]))
        g = float(np.asarray(cls.laminar_dissipation_factor(h)[0]))
        return g / f - 3.0 / (2.0 + h)

    lo, hi = 1.8, 3.0
    flo = ratio(lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fmid = ratio(mid)
        if flo * fmid <= 0.0:
            hi = mid
        else:
            lo, fmid = mid, fmid
            flo = fmid
    h = 0.5 * (lo + hi)
    f = float(np.asarray(cls.laminar_cf_factor(h)[0]))
    g = float(np.asarray(cls.laminar_dissipation_factor(h)[0]))
    return h, f, g


def _solve2(xp, j11, j12, j21, j22, r1, r2):
    """Solve the 2 by 2 Newton system for the correction.

    Parameters
    ----------
    j11, j12, j21, j22 : array_like
        Jacobian entries (with mixed units).
    r1, r2 : array_like
        Residuals (with mixed units).

    Returns
    -------
    tuple
        ``(d1, d2)`` corrections with the unit of the unknowns.
    """
    det = j11 * j22 - j12 * j21
    d1 = (-r1 * j22 + r2 * j12) / det
    d2 = (-j11 * r2 + j21 * r1) / det
    return d1, d2


def _solve3(xp, mat, rhs):
    """Solve a batched 3 by 3 system ``mat x = rhs``.

    Parameters
    ----------
    mat : array_like
        Matrix with shape (L, 3, 3) (mixed units).
    rhs : array_like
        Right-hand side with shape (L, 3) (mixed units).

    Returns
    -------
    array_like
        Solution with shape (L, 3).
    """
    import numpy as np

    if xp is None or xp is np:
        return np.linalg.solve(mat, rhs[..., None])[..., 0]
    torch = xp
    out = torch.linalg.solve(mat, rhs[..., None])[..., 0]
    return out


def _tol_for(xp, arr) -> float:
    """Return the Newton tolerance for the dtype of ``arr``."""
    import numpy as np

    if xp is None or xp is np:
        dt = np.asarray(arr).dtype
        if dt == np.float32:
            return 1e-5
        return 1e-10
    try:
        import torch

        if isinstance(arr, torch.Tensor) and arr.dtype == torch.float32:
            return 1e-5
    except Exception:
        pass
    return 1e-10


def _laminar_rhs(xp, theta, H, ue, m, re_ref):
    """Return laminar momentum/energy right-hand sides and closures.

    Parameters
    ----------
    theta : array_like
        Momentum thickness [m or chord units].
    H : array_like
        Shape factor (dimensionless).
    ue : array_like
        Edge velocity ratio (dimensionless).
    m : array_like
        Edge velocity gradient [1/m or 1/chord units].
    re_ref : array_like
        Reference Reynolds number (dimensionless).

    Returns
    -------
    tuple
        ``(Fm, Fe, tstar, Hs, F, G)`` with rates [dimensionless] and
        thicknesses [m or chord units].
    """
    from ventorum.ibl2d import closure as cls

    re = ue * theta * re_ref
    Hs, _ = cls.laminar_h_star(H, xp)
    F, _ = cls.laminar_cf_factor(H, xp)
    G, _ = cls.laminar_dissipation_factor(H, xp)
    inv = xp.where(re > 0.0, 1.0 / xp.where(re > 0.0, re, 1.0), 0.0)
    cf2 = F * inv
    press = (2.0 + H) * theta * m / xp.where(ue != 0.0, ue, 1.0)
    Fm = cf2 - press
    cd2 = G * Hs * inv
    estar = Hs * theta
    Fe = cd2 - 3.0 * estar * m / xp.where(ue != 0.0, ue, 1.0)
    return Fm, Fe, estar, Hs, F, G


def _newton_laminar(xp, th_p, H_p, ue_p, m_p, ue_i, m_i, ds, re_ref,
                    th_g, H_g, zero_prev=False):
    """Solve one laminar step with Newton iteration.

    Parameters
    ----------
    th_p, H_p : array_like
        Previous theta [m or chord units] and H (dimensionless).
    ue_p, m_p, ue_i, m_i : array_like
        Edge velocity (dimensionless) and gradient [1/m] at both ends.
    ds : float
        Step size [m or chord units].
    re_ref : array_like
        Reference Reynolds number (dimensionless).
    th_g, H_g : array_like
        Initial guess of theta [m or chord units] and H.
    zero_prev : bool
        True at the stagnation point where the previous rate is zero.

    Returns
    -------
    tuple
        ``(theta, H, Hs, F, G, ok)`` with the converged state and a
        boolean convergence flag.
    """
    from ventorum.ibl2d import closure as cls

    import numpy as np

    th = th_g + 0.0
    H = H_g + 0.0
    if zero_prev:
        Fm_p = 0.0 * th_p
        Fe_p = 0.0 * th_p
        ts_p = th_p + 0.0
        try:
            Hs_p, _ = cls.laminar_h_star(H_p, xp)
            ts_p = Hs_p * th_p
        except Exception:
            pass
    else:
        Fm_p, Fe_p, ts_p, _, _, _ = _laminar_rhs(xp, th_p, H_p, ue_p, m_p, re_ref)
    tol = _tol_for(xp, th)
    if _is_torch(xp):
        L = int(th_p.numel())
    else:
        L = int(np.asarray(th_p).size)
    ok = np.ones((L,), dtype=bool)
    for _ in range(15):
        Fm, Fe, ts, Hs, F, G = _laminar_rhs(xp, th, H, ue_i, m_i, re_ref)
        r1 = (th - th_p) / ds - 0.5 * (Fm + Fm_p)
        Hs_d, dHs = cls.laminar_h_star(H, xp)
        r2 = (ts - ts_p) / ds - 0.5 * (Fe + Fe_p)
        re = ue_i * th * re_ref
        inv = 1.0 / xp.where(re > 0.0, re, 1.0)
        _, dF = cls.laminar_cf_factor(H, xp)
        _, dG = cls.laminar_dissipation_factor(H, xp)
        ue_s = xp.where(ue_i != 0.0, ue_i, 1.0)
        dFm_dth = -F * inv / xp.where(th > 0.0, th, 1.0) - (2.0 + H) * m_i / ue_s
        dFm_dH = dF * inv - th * m_i / ue_s
        dFe_dth = -G * Hs * inv / xp.where(th > 0.0, th, 1.0) - 3.0 * Hs * m_i / ue_s
        dFe_dH = (dG * Hs + G * dHs) * inv - 3.0 * dHs * th * m_i / ue_s
        j11 = 1.0 / ds - 0.5 * dFm_dth
        j12 = -0.5 * dFm_dH
        j21 = Hs / ds - 0.5 * dFe_dth
        j22 = th * dHs / ds - 0.5 * dFe_dH
        d1, d2 = _solve2(xp, j11, j12, j21, j22, r1, r2)
        bad = ~(xp.isfinite(d1) & xp.isfinite(d2))
        if _is_torch(xp):
            import torch

            badn = bad.detach().cpu().numpy()
        else:
            badn = np.asarray(bad)
        ok = ok & (~badn)
        step_t = xp.where(bad, 0.0, d1)
        step_h = xp.where(bad, 0.0, d2)
        th = th + step_t
        H = H + step_h
        th = xp.where(th > 1e-14, th, 1e-14)
        H = xp.where(H > 1.02, H, 1.02)
        H = xp.where(H < 7.49, H, 7.49)
        if _is_torch(xp):
            import torch

            c1 = (step_t / xp.where(th > 0.0, th, torch.ones_like(th))).detach()
            c1 = c1.cpu().numpy()
            c2 = (step_h / xp.where(H > 0.0, H, torch.ones_like(H))).detach()
            c2 = c2.cpu().numpy()
        else:
            c1 = np.asarray(step_t / np.where(np.asarray(th) > 0.0, np.asarray(th), 1.0))
            c2 = np.asarray(step_h / np.where(np.asarray(H) > 0.0, np.asarray(H), 1.0))
        if np.all(~np.isfinite(c1) | (np.abs(c1) < tol)) and np.all(
                ~np.isfinite(c2) | (np.abs(c2) < tol)):
            break
    Fm, Fe, ts, Hs, F, G = _laminar_rhs(xp, th, H, ue_i, m_i, re_ref)
    r1 = (th - th_p) / ds - 0.5 * (Fm + Fm_p)
    r2 = (ts - ts_p) / ds - 0.5 * (Fe + Fe_p)
    if _is_torch(xp):
        r1n = r1.detach().cpu().numpy()
        r2n = r2.detach().cpu().numpy()
        thn = th.detach().cpu().numpy()
    else:
        r1n = np.asarray(r1, dtype=float)
        r2n = np.asarray(r2, dtype=float)
        thn = np.asarray(th, dtype=float)
    scale = np.abs(thn) / max(float(ds), 1e-300) + 1.0
    good = ok & np.isfinite(r1n) & np.isfinite(r2n)
    good = good & (np.abs(r1n) < 1e-6 * scale + 1e-9)
    good = good & (np.abs(r2n) < 1e-6 * scale + 1e-9)
    return th, H, Hs, F, G, good


def _turb_state(xp, theta, H, Ct, ue, re_ref):
    """Return turbulent closures and rates at one state.

    Parameters
    ----------
    theta : array_like
        Momentum thickness [m or chord units].
    H : array_like
        Shape factor (dimensionless).
    Ct : array_like
        Shear coefficient (dimensionless).
    ue : array_like
        Edge velocity ratio (dimensionless).
    re_ref : array_like
        Reference Reynolds number (dimensionless).

    Returns
    -------
    tuple
        ``(Fm, Fe, Fl, tstar, Cf, Hs, Us, Te, CD)``.
    """
    from ventorum.ibl2d import closure as cls

    re = ue * theta * re_ref
    Cf, _, _ = cls.turbulent_cf(H, re, xp)
    Hs, _, _ = cls.turbulent_h_star(H, re, xp)
    Us, _, _ = cls.turbulent_slip_velocity(H, re, xp)
    Te, _, _, _ = cls.equilibrium_shear_coefficient(H, Hs, Us, xp)
    CD, _, _, _, _ = cls.turbulent_dissipation_coefficient(H, re, Us, Ct, xp)
    return Cf, Hs, Us, Te, CD


def _newton_turbulent(xp, th_p, H_p, C_p, ue_p, m_p, ue_i, m_i, ds,
                      re_ref, th_g, H_g, C_g):
    """Solve one turbulent step with Newton iteration.

    Parameters
    ----------
    th_p, H_p, C_p : array_like
        Previous theta [m or chord units], H and Ctau (dimensionless).
    ue_p, m_p, ue_i, m_i : array_like
        Edge data at both ends of the step.
    ds : float
        Step size [m or chord units].
    re_ref : array_like
        Reference Reynolds number (dimensionless).
    th_g, H_g, C_g : array_like
        Initial guess of the new state.

    Returns
    -------
    tuple
        ``(theta, H, Ctau, Cf, ok)`` with the converged state and flag.
    """
    from ventorum.ibl2d import closure as cls

    import numpy as np

    th = th_g + 0.0
    H = H_g + 0.0
    Ct = C_g + 0.0
    tol = _tol_for(xp, th)
    if _is_torch(xp):
        L = int(th_p.numel())
    else:
        L = int(np.asarray(th_p).size)
    ok = np.ones((L,), dtype=bool)
    for _ in range(20):
        Cf, Hs, Us, Te, CD = _turb_state(xp, th, H, Ct, ue_i, re_ref)
        Cf_p, Hs_p, Us_p, Te_p, CD_p = _turb_state(xp, th_p, H_p, C_p, ue_p, re_ref)
        ue_s = xp.where(ue_i != 0.0, ue_i, 1.0)
        ue_ps = xp.where(ue_p != 0.0, ue_p, 1.0)
        Fm = 0.5 * Cf - (2.0 + H) * th * m_i / ue_s
        Fm_p = 0.5 * Cf_p - (2.0 + H_p) * th_p * m_p / ue_ps
        ts = Hs * th
        ts_p = Hs_p * th_p
        Fe = 2.0 * CD - 3.0 * ts * m_i / ue_s
        Fe_p = 2.0 * CD_p - 3.0 * ts_p * m_p / ue_ps
        sq = xp.sqrt(xp.where(Ct > 1e-300, Ct, 1e-300))
        sqe = xp.sqrt(xp.where(Te > 1e-300, Te, 1e-300))
        dst = H * th
        Fl = 5.6 * sq * (sqe - sq) / xp.where(dst > 1e-300, dst, 1.0)
        sq_p = xp.sqrt(xp.where(C_p > 1e-300, C_p, 1e-300))
        sqe_p = xp.sqrt(xp.where(Te_p > 1e-300, Te_p, 1e-300))
        dst_p = H_p * th_p
        Fl_p = 5.6 * sq_p * (sqe_p - sq_p) / xp.where(dst_p > 1e-300, dst_p, 1.0)
        r1 = (th - th_p) / ds - 0.5 * (Fm + Fm_p)
        r2 = (ts - ts_p) / ds - 0.5 * (Fe + Fe_p)
        r3 = (Ct - C_p) / ds - 0.5 * (Fl + Fl_p)
        re = ue_i * th * re_ref
        dRe = ue_i * re_ref
        _, dCfH, dCfRe = cls.turbulent_cf(H, re, xp)
        _, dHsH, dHsRe = cls.turbulent_h_star(H, re, xp)
        _, dUsH, dUsRe = cls.turbulent_slip_velocity(H, re, xp)
        _, _, dTeHs, dTeUs = cls.equilibrium_shear_coefficient(H, Hs, Us, xp)
        dTeH = dTeHs * dHsH + dTeUs * dUsH
        dTeRe = dTeHs * dHsRe + dTeUs * dUsRe
        dTe_dth = dTeRe * dRe
        dCf_dth = dCfRe * dRe
        dHs_dth = dHsRe * dRe
        dCD_dH = 0.5 * (dCfH * Us + Cf * dUsH) - 0.5 * (dTeHs * dHsH + dTeUs * dUsH)
        dCD_dRe = (0.5 * (dCfRe * Us + Cf * dUsRe)
                   - 0.5 * (dTeHs * dHsRe + dTeUs * dUsRe))
        dCD_dth = dCD_dRe * dRe
        dFm_dth = 0.5 * dCf_dth - (2.0 + H) * m_i / ue_s
        dFm_dH = 0.5 * dCfH - th * m_i / ue_s
        dFe_dth = (2.0 * dCD_dth
                   - 3.0 * m_i / ue_s * (dHs_dth * th + Hs))
        dFe_dH = 2.0 * dCD_dH - 3.0 * m_i / ue_s * th * dHsH
        dFe_dC = xp.full_like(th + 0.0, 1.0)
        ulo = xp.where(dst > 1e-300, dst, 1.0)
        dFl_dth = (-Fl / xp.where(th > 1e-300, th, 1.0)
                   + 5.6 * sq / ulo * 0.5 / xp.where(sqe > 1e-300, sqe, 1.0)
                   * dTe_dth)
        dFl_dH = (-Fl / xp.where(H > 1e-300, H, 1.0)
                  + 5.6 * sq / ulo * 0.5 / xp.where(sqe > 1e-300, sqe, 1.0)
                  * dTeH)
        dFl_dC = (2.8 * (sqe - 2.0 * sq) / ulo
                  / xp.where(sq > 1e-300, sq, 1.0))
        j11 = 1.0 / ds - 0.5 * dFm_dth
        j12 = -0.5 * dFm_dH
        j21 = Hs / ds + th * dHs_dth / ds - 0.5 * dFe_dth
        j22 = th * dHsH / ds - 0.5 * dFe_dH
        j23 = -0.5 * dFe_dC
        j31 = -0.5 * dFl_dth
        j32 = -0.5 * dFl_dH
        j33 = 1.0 / ds - 0.5 * dFl_dC
        o = xp.zeros_like(j11)
        mat = xp.stack([
            xp.stack([j11, j12, o], axis=-1),
            xp.stack([j21, j22, j23], axis=-1),
            xp.stack([j31, j32, j33], axis=-1)], axis=-2)
        rhs = xp.stack([-r1, -r2, -r3], axis=-1)
        step = _solve3(xp, mat, rhs)
        s1 = step[..., 0]
        s2 = step[..., 1]
        s3 = step[..., 2]
        bad = ~(xp.isfinite(s1) & xp.isfinite(s2) & xp.isfinite(s3))
        if _is_torch(xp):
            badn = bad.detach().cpu().numpy()
        else:
            badn = np.asarray(bad)
        ok = ok & (~badn)
        s1 = xp.where(bad, 0.0, s1)
        s2 = xp.where(bad, 0.0, s2)
        s3 = xp.where(bad, 0.0, s3)
        th = th + s1
        H = H + s2
        Ct = Ct + s3
        th = xp.where(th > 1e-14, th, 1e-14)
        H = xp.where(H > 1.02, H, 1.02)
        H = xp.where(H < 7.49, H, 7.49)
        Ct = xp.where(Ct > 1e-12, Ct, 1e-12)
        if _is_torch(xp):
            c1 = (s1 / th).detach().cpu().numpy()
            c2 = (s2 / H).detach().cpu().numpy()
            c3 = (s3 / Ct).detach().cpu().numpy()
        else:
            c1 = np.asarray(s1 / np.asarray(th))
            c2 = np.asarray(s2 / np.asarray(H))
            c3 = np.asarray(s3 / np.asarray(Ct))
        if (np.all(~np.isfinite(c1) | (np.abs(c1) < tol))
                and np.all(~np.isfinite(c2) | (np.abs(c2) < tol))
                and np.all(~np.isfinite(c3) | (np.abs(c3) < tol))):
            break
    Cf, Hs, Us, Te, CD = _turb_state(xp, th, H, Ct, ue_i, re_ref)
    return th, H, Ct, Cf, ok


def _advance_N(xp, N_p, H_p, re_p, H_i, re_i):
    """Advance the envelope exponent over one step.

    Parameters
    ----------
    N_p : array_like
        Previous exponent (dimensionless).
    H_p, re_p : array_like
        Previous shape factor and Re_theta (dimensionless).
    H_i, re_i : array_like
        New shape factor and Re_theta (dimensionless).

    Returns
    -------
    array_like
        New exponent (dimensionless), never below ``N_p``.
    """
    from ventorum.ibl2d import transition as tr

    re0_p, _ = tr.critical_re_theta(H_p, xp)
    re0_i, _ = tr.critical_re_theta(H_i, xp)
    g_p, _ = tr.amplification_rate(H_p, xp)
    g_i, _ = tr.amplification_rate(H_i, xp)
    above_p = re_p >= re0_p
    above_i = re_i >= re0_i
    both = above_p & above_i
    cross = (~above_p) & above_i
    inc_both = 0.5 * (g_p + g_i) * (re_i - re_p)
    inc_cross = g_i * (re_i - re0_i)
    inc = xp.where(both, inc_both, 0.0)
    inc = xp.where(cross, inc_cross, inc)
    inc = xp.where(inc > 0.0, inc, 0.0)
    return N_p + inc


def march_direct_batch(s, ue_batch, Re_ref_batch, xp=None, *,
                       dtype=None, device=None, ncrit=9.0,
                       forced_transition_s=None, theta0=None):
    """March many independent surfaces in direct mode at once.

    Parameters
    ----------
    s : array_like
        Arc length stations [m or chord units], shape (N,), increasing.
    ue_batch : array_like
        Edge velocity ratio (dimensionless), shape (B, N).
    Re_ref_batch : array_like
        Reference Reynolds number (dimensionless), shape (B,) or scalar.
    xp : module or None
        Array module (default numpy); pass torch for CPU or CUDA tensors.
    dtype : object or None
        Torch dtype for a torch march (default float64 on CPU).
    device : object or None
        Torch device for a torch march (default CPU).
    ncrit : float
        Critical exponent (dimensionless).
    forced_transition_s : float or None
        Forced transition position [same unit as ``s``] or None.
    theta0 : float, array_like or None
        Initial momentum thickness [m or chord units] for a Blasius
        start, or None for the similarity value.

    Returns
    -------
    list of BoundaryLayerResult
        One result per batch row, with arrays of the ``xp`` module.
    """
    import numpy as np

    from ventorum.ibl2d import closure as cls

    if xp is None:
        import numpy as xp_mod

        xp = xp_mod
    is_torch = _is_torch(xp)
    if is_torch:
        import torch

        if dtype is None:
            dtype = torch.float64
        elif isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if device is None:
            device = "cpu"
    s_np = np.asarray(s, dtype=float).ravel()
    ue_np = np.asarray(ue_batch, dtype=float)
    if ue_np.ndim == 1:
        ue_np = ue_np[None, :]
    npts = s_np.size
    nrows = ue_np.shape[0]
    assert ue_np.shape[1] == npts
    assert np.all(np.diff(s_np) > 0.0)
    re_np = np.asarray(Re_ref_batch, dtype=float).ravel()
    if re_np.size == 1:
        re_np = np.full((nrows,), float(re_np[0]))
    assert re_np.size == nrows
    if theta0 is None:
        th0_np = np.full((nrows,), np.nan)
    else:
        t0 = np.asarray(theta0, dtype=float).ravel()
        if t0.size == 1:
            t0 = np.full((nrows,), float(t0[0]))
        assert t0.size == nrows
        th0_np = t0
    s_x = _to_xp(xp, s_np, dtype, device)
    ue_x = _to_xp(xp, ue_np, dtype, device)
    re_x = _to_xp(xp, re_np, dtype, device)
    m_x = _gradient(xp, ue_x, s_x)
    h_stag, f_stag, _ = _hiemenz_state()
    f_blas = float(np.asarray(cls.laminar_cf_factor(H_BLASIUS)[0]))
    stag = np.zeros((nrows,), dtype=bool)
    if abs(float(s_np[0])) < 1e-12 and npts >= 3:
        for b in range(nrows):
            u0 = abs(float(ue_np[b, 0]))
            umax = float(np.max(np.abs(ue_np[b, :]))) + 1e-300
            if u0 > 1e-9 * umax:
                continue
            k1 = float(ue_np[b, 1] / (s_np[1] - s_np[0]))
            k2 = float(ue_np[b, 2] / (s_np[2] - s_np[0]))
            if k1 <= 0.0 or k2 <= 0.0:
                continue
            if abs(k2 - k1) <= 0.05 * (abs(k1) + abs(k2)):
                stag[b] = True
    th = _full(xp, (nrows, npts), float("nan"), like=ue_x, dtype=dtype, device=device)
    H = _full(xp, (nrows, npts), float("nan"), like=ue_x, dtype=dtype, device=device)
    Cf = _full(xp, (nrows, npts), float("nan"), like=ue_x, dtype=dtype, device=device)
    Ct = _full(xp, (nrows, npts), 0.0, like=ue_x, dtype=dtype, device=device)
    Namp = _full(xp, (nrows, npts), 0.0, like=ue_x, dtype=dtype, device=device)
    mode = np.full((nrows,), 0, dtype=int)
    alive = np.ones((nrows,), dtype=bool)
    xtr = np.full((nrows,), np.nan)
    xsep = np.full((nrows,), np.nan)
    nvalid = np.ones((nrows,), dtype=int)
    start_idx = np.ones((nrows,), dtype=int)
    if is_torch:
        import torch as _torch

        th_np = np.full((nrows, npts), np.nan)
        H_np = np.full((nrows, npts), np.nan)
    else:
        th_np = None
    for b in range(nrows):
        if stag[b]:
            k = 0.5 * (ue_np[b, 1] / (s_np[1] - s_np[0])
                       + ue_np[b, 2] / (s_np[2] - s_np[0]))
            t0 = float(np.sqrt(f_stag / ((2.0 + h_stag) * k * re_np[b])))
            if is_torch:
                th[b, 0] = float(t0)
                H[b, 0] = float(h_stag)
                th_np[b, 0] = t0
                H_np[b, 0] = h_stag
            else:
                th[b, 0] = t0
                H[b, 0] = h_stag
            start_idx[b] = 1
        elif abs(float(s_np[0])) < 1e-12:
            if np.isfinite(th0_np[b]):
                if is_torch:
                    th[b, 0] = float(th0_np[b])
                    H[b, 0] = H_BLASIUS
                else:
                    th[b, 0] = float(th0_np[b])
                    H[b, 0] = H_BLASIUS
                start_idx[b] = 1
            else:
                if is_torch:
                    th[b, 0] = 0.0
                    H[b, 0] = H_BLASIUS
                    s1 = float(s_np[1])
                    u1 = float(ue_np[b, 1])
                    t1 = float(np.sqrt(2.0 * f_blas * s1 / (u1 * re_np[b])))
                    th[b, 1] = t1
                    H[b, 1] = H_BLASIUS
                    Namp[b, 0] = 0.0
                    Namp[b, 1] = 0.0
                else:
                    th[b, 0] = 0.0
                    H[b, 0] = H_BLASIUS
                    s1 = float(s_np[1])
                    u1 = float(ue_np[b, 1])
                    t1 = float(np.sqrt(2.0 * f_blas * s1 / (u1 * re_np[b])))
                    th[b, 1] = t1
                    H[b, 1] = H_BLASIUS
                    Namp[b, 0] = 0.0
                    Namp[b, 1] = 0.0
                start_idx[b] = 2
                nvalid[b] = 2
        elif np.isfinite(th0_np[b]):
            if is_torch:
                th[b, 0] = float(th0_np[b])
            else:
                th[b, 0] = float(th0_np[b])
            if is_torch:
                H[b, 0] = H_BLASIUS
            else:
                H[b, 0] = H_BLASIUS
            start_idx[b] = 1
        else:
            s0 = float(s_np[0])
            u0 = float(ue_np[b, 0])
            t0 = float(np.sqrt(2.0 * f_blas * s0 / (u0 * re_np[b])))
            if is_torch:
                th[b, 0] = t0
                H[b, 0] = H_BLASIUS
            else:
                th[b, 0] = t0
                H[b, 0] = H_BLASIUS
            start_idx[b] = 1
    if is_torch:
        import torch as _torch

        for b in range(nrows):
            if stag[b]:
                continue
            if abs(float(s_np[0])) < 1e-12 and not np.isfinite(th0_np[b]):
                continue
            th_np[b, 0] = float(th[b, 0].detach().cpu().numpy())
            H_np[b, 0] = float(H[b, 0].detach().cpu().numpy())
        th_np[:, 1:] = np.nan
        H_np[:, 1:] = np.nan
        for b in range(nrows):
            if abs(float(s_np[0])) < 1e-12 and not stag[b] and not np.isfinite(th0_np[b]):
                th_np[b, 1] = float(th[b, 1].detach().cpu().numpy())
                H_np[b, 1] = H_BLASIUS
    for i in range(1, npts):
        ds = float(s_np[i] - s_np[i - 1])
        for b in range(nrows):
            if not alive[b]:
                continue
            if i < start_idx[b]:
                continue
            if i == 1 and start_idx[b] == 2 and not stag[b]:
                nvalid[b] = 2
                continue
            prev = i - 1
            if mode[b] == 1:
                th_p = th[b, prev:prev + 1]
                H_p = H[b, prev:prev + 1]
                C_p = Ct[b, prev:prev + 1]
                ue_p = ue_x[b, prev:prev + 1]
                m_p = m_x[b, prev:prev + 1]
                ue_i = ue_x[b, i:i + 1]
                m_i = m_x[b, i:i + 1]
                re_b = re_x[b:b + 1]
                thn, Hn, Cn, Cfn, ok = _newton_turbulent(
                    xp, th_p, H_p, C_p, ue_p, m_p, ue_i, m_i,
                    ds, re_b, th_p, H_p, C_p)
                if not bool(np.asarray(ok)[0]):
                    alive[b] = False
                    xsep[b] = float(s_np[i])
                    nvalid[b] = i
                    continue
                th[b, i] = thn[0]
                H[b, i] = Hn[0]
                Ct[b, i] = Cn[0]
                Cf[b, i] = Cfn[0]
                Namp[b, i] = Namp[b, prev]
                cfi = float(Cfn.detach().cpu().numpy()[0]) if is_torch else float(
                    np.asarray(Cfn)[0])
                if cfi <= 0.0:
                    cfp = float(Cf[b, prev].detach().cpu().numpy()) if is_torch else float(
                        Cf[b, prev])
                    if np.isfinite(cfp) and cfp > 0.0 and cfi != cfp:
                        frac = (0.0 - cfp) / (cfi - cfp)
                        xsep[b] = float(s_np[prev] + frac * ds)
                    else:
                        xsep[b] = float(s_np[i])
                    alive[b] = False
                    mode[b] = 2
                    nvalid[b] = i + 1
                else:
                    nvalid[b] = i + 1
                continue
            s_prev = float(s_np[prev])
            s_now = float(s_np[i])
            forced_in = (forced_transition_s is not None
                         and s_prev < float(forced_transition_s) <= s_now)
            th_p = th[b, prev:prev + 1]
            H_p = H[b, prev:prev + 1]
            ue_p = ue_x[b, prev:prev + 1]
            m_p = m_x[b, prev:prev + 1]
            ue_i = ue_x[b, i:i + 1]
            m_i = m_x[b, i:i + 1]
            re_b = re_x[b:b + 1]
            zero_prev = bool(stag[b] and prev == 0)
            thn, Hn, Hsn, Fn, Gn, ok = _newton_laminar(
                xp, th_p, H_p, ue_p, m_p, ue_i, m_i, ds, re_b,
                th_p, H_p, zero_prev=zero_prev)
            if not bool(np.asarray(ok)[0]):
                alive[b] = False
                xsep[b] = float(s_np[i])
                nvalid[b] = i
                continue
            Hn_f = float(Hn.detach().cpu().numpy()[0]) if is_torch else float(
                np.asarray(Hn)[0])
            Np = Namp[b, prev:prev + 1]
            Hp = H[b, prev:prev + 1]
            Ni = _advance_N(xp, Np, Hp,
                            ue_p * th_p * re_b, Hn, ue_i * thn * re_b)
            Ni_f = float(Ni.detach().cpu().numpy()[0]) if is_torch else float(
                np.asarray(Ni)[0])
            s_nat = None
            if Ni_f >= float(ncrit):
                Np_f = float(Np.detach().cpu().numpy()[0]) if is_torch else float(
                    np.asarray(Np)[0])
                if Ni_f > Np_f:
                    frac = (float(ncrit) - Np_f) / (Ni_f - Np_f)
                    frac = min(max(frac, 0.0), 1.0)
                    s_nat = s_prev + frac * ds
            s_tr = None
            if forced_in:
                s_f = float(forced_transition_s)
                if s_nat is not None and s_nat < s_f:
                    s_tr = s_nat
                else:
                    s_tr = s_f
            elif s_nat is not None:
                s_tr = s_nat
            if s_tr is None:
                th[b, i] = thn[0]
                H[b, i] = Hn[0]
                Namp[b, i] = Ni[0]
                re_i_x = ue_i * thn * re_b
                lam_cf = 2.0 * Fn / xp.where(re_i_x > 0.0, re_i_x, 1.0)
                Cf[b, i] = lam_cf[0]
                Ct[b, i] = 0.0 if not is_torch else Ct[b, i]
                Hn_c = Hn_f
                if Hn_c >= H_LAM_SEP:
                    Hp_f = float(H[b, prev].detach().cpu().numpy()) if is_torch else float(
                        H[b, prev])
                    if np.isfinite(Hp_f) and Hn_c != Hp_f:
                        frac = (H_LAM_SEP - Hp_f) / (Hn_c - Hp_f)
                        xsep[b] = float(s_prev + frac * ds)
                    else:
                        xsep[b] = float(s_now)
                    alive[b] = False
                    mode[b] = 2
                    nvalid[b] = i + 1
                else:
                    nvalid[b] = i + 1
                continue
            frac = (s_tr - s_prev) / ds if ds > 0.0 else 0.0
            Hp_f = float(H[b, prev].detach().cpu().numpy()) if is_torch else float(
                H[b, prev])
            ds1 = float(s_tr - s_prev)
            ds2 = float(s_now - s_tr)
            if ds1 > 1e-14:
                ue_tr = ue_x[b, prev:prev + 1] * (1.0 - frac) + ue_i * frac
                m_tr = m_x[b, prev:prev + 1] * (1.0 - frac) + m_i * frac
                th_tr_g = th_p * (1.0 - frac) + thn * frac
                H_tr_g = Hp * (1.0 - frac) + Hn * frac
                th_tr, H_tr, _, _, _, ok1 = _newton_laminar(
                    xp, th_p, Hp, ue_p, m_p, ue_tr, m_tr, ds1, re_b,
                    th_tr_g, H_tr_g, zero_prev=zero_prev)
                if not bool(np.asarray(ok1)[0]):
                    th_tr, H_tr = th_tr_g, H_tr_g
            else:
                th_tr, H_tr = th_p + 0.0, Hp + 0.0
                ue_tr = ue_p + 0.0
                m_tr = m_p + 0.0
            re_tr = ue_tr * th_tr * re_b if ds1 > 1e-14 else ue_p * th_p * re_b
            Hs_t, _, _ = cls.turbulent_h_star(H_tr, re_tr, xp)
            Us_t, _, _ = cls.turbulent_slip_velocity(H_tr, re_tr, xp)
            Te_t, _, _, _ = cls.equilibrium_shear_coefficient(H_tr, Hs_t, Us_t, xp)
            C_tr = xp.where(Te_t > 1e-12, Te_t, 1e-12)
            ue_i2 = ue_x[b, i:i + 1]
            m_i2 = m_x[b, i:i + 1]
            if ds2 > 1e-14:
                thn2, Hn2, Cn2, Cfn2, ok2 = _newton_turbulent(
                    xp, th_tr, H_tr, C_tr, ue_tr, m_tr, ue_i2, m_i2,
                    ds2, re_b, th_tr, H_tr, C_tr)
                if not bool(np.asarray(ok2)[0]):
                    thn2, Hn2, Cn2 = th_tr + 0.0, H_tr + 0.0, C_tr + 0.0
                    Cfn2, _, _, _, _ = _turb_state(xp, thn2, Hn2, Cn2, ue_i2, re_b)
                    Cfn2 = Cfn2[0:1] if hasattr(Cfn2, "__getitem__") else Cfn2
                    try:
                        cft, _, _ = cls.turbulent_cf(Hn2, ue_i2 * thn2 * re_b, xp)
                        Cfn2 = cft
                    except Exception:
                        pass
            else:
                thn2, Hn2, Cn2 = th_tr, H_tr, C_tr
                Cfn2, _, _, _, _ = _turb_state(xp, thn2, Hn2, Cn2, ue_i2, re_b)
                try:
                    cft, _, _ = cls.turbulent_cf(Hn2, ue_i2 * thn2 * re_b, xp)
                    Cfn2 = cft
                except Exception:
                    pass
            th[b, i] = thn2[0]
            H[b, i] = Hn2[0]
            Ct[b, i] = Cn2[0]
            Cf[b, i] = Cfn2[0]
            Namp[b, i] = float(ncrit)
            xtr[b] = float(s_tr)
            mode[b] = 1
            nvalid[b] = i + 1
    results = []
    for b in range(nrows):
        nv = int(nvalid[b])
        sep = bool(mode[b] == 2 or not alive[b])
        if np.isnan(xsep[b]) and sep and nv < npts:
            xsep[b] = float(s_np[min(nv, npts - 1)])
        if is_torch:
            import torch as _torch

            s_r = _to_xp(xp, s_np, dtype, device)
            ue_r = ue_x[b, :].clone()
            th_r = th[b, :].clone()
            H_r = H[b, :].clone()
            Cf_r = Cf[b, :].clone()
            Ct_r = Ct[b, :].clone()
            N_r = Namp[b, :].clone()
            ds_r = H_r * th_r
            prod = ue_r * ds_r
            tra = _gradient(xp, prod[None, :], s_x)[0, :]
        else:
            s_r = s_np.copy()
            ue_r = np.asarray(ue_x[b, :]).copy()
            th_r = np.asarray(th[b, :]).copy()
            H_r = np.asarray(H[b, :]).copy()
            Cf_r = np.asarray(Cf[b, :]).copy()
            Ct_r = np.asarray(Ct[b, :]).copy()
            N_r = np.asarray(Namp[b, :]).copy()
            prod = ue_r * H_r * th_r
            tra = np.empty_like(prod)
            tra[0] = (prod[1] - prod[0]) / (s_np[1] - s_np[0])
            tra[-1] = (prod[-1] - prod[-2]) / (s_np[-1] - s_np[-2])
            tra[1:-1] = (prod[2:] - prod[:-2]) / (s_np[2:] - s_np[:-2])
        if sep:
            if is_torch:
                import torch as _torch

                if nv < npts:
                    th_r[nv:] = float("nan")
                    H_r[nv:] = float("nan")
                    Cf_r[nv:] = float("nan")
                    Ct_r[nv:] = float("nan")
                    N_r[nv:] = float("nan")
                    tra[nv:] = float("nan")
            else:
                if nv < npts:
                    th_r[nv:] = np.nan
                    H_r[nv:] = np.nan
                    Cf_r[nv:] = np.nan
                    Ct_r[nv:] = np.nan
                    N_r[nv:] = np.nan
                    tra[nv:] = np.nan
        else:
            if np.isnan(xsep[b]):
                pass
        if is_torch:
            import torch as _torch

            valid = _torch.arange(npts, device=tra.device) < nv
            tra = _torch.where(valid, tra, float("nan"))
        ds_out = H_r * th_r
        res = BoundaryLayerResult(
            s=s_r,
            ue=ue_r,
            theta=th_r,
            delta_star=ds_out,
            H=H_r,
            Cf=Cf_r,
            Ctau=Ct_r,
            N=N_r,
            transp=tra,
            x_transition=None if np.isnan(xtr[b]) else float(xtr[b]),
            x_separation=None if np.isnan(xsep[b]) else float(xsep[b]),
            separated=sep,
            n_valid=nv,
        )
        results.append(res)
    return results


def march_direct(s, ue, Re_ref, *, ncrit=9.0, forced_transition_s=None,
                 theta0=None):
    """March one surface in direct mode.

    Parameters
    ----------
    s : array_like
        Arc length stations [m or chord units], increasing.
    ue : array_like
        Edge velocity ratio (dimensionless).
    Re_ref : float
        Reference Reynolds number (dimensionless).
    ncrit : float
        Critical exponent (dimensionless).
    forced_transition_s : float or None
        Forced transition position [same unit as ``s``] or None.
    theta0 : float or None
        Initial momentum thickness [m or chord units] for a Blasius
        start, or None for the similarity value.

    Returns
    -------
    BoundaryLayerResult
        Boundary-layer state along ``s``.
    """
    import numpy as np

    s_np = np.asarray(s, dtype=float).ravel()
    ue_np = np.asarray(ue, dtype=float).ravel()
    outs = march_direct_batch(s_np, ue_np[None, :], np.array([float(Re_ref)]),
                              xp=np, ncrit=ncrit,
                              forced_transition_s=forced_transition_s,
                              theta0=theta0)
    return outs[0]
