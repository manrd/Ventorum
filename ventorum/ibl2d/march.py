# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Direct march of the integral boundary layer for a given edge velocity.

The march solves the discrete equations of :mod:`ventorum.ibl2d.equations`
station by station along one surface for a given edge velocity ue(s)
(direct mode). It uses the same equations and closures as the coupled
viscous-inviscid solver of :mod:`ventorum.panel2d.vii`; the coupled solver
uses it for its first estimate.

* Start: the stagnation-point (Hiemenz) similarity solution when ue(s[0])
  is zero (ue proportional to s near s[0]), else the flat-plate (Blasius)
  similarity solution, or a given momentum thickness theta0 with the
  Blasius shape factor of the closure.
* Each station: Newton iteration on (theta, delta_star, c) with the exact
  Jacobian of the discrete equations.
* Transition: envelope e^N (free, critical exponent ncrit) or forced at a
  given position, at the interpolated point inside the interval.
* Separation: in direct mode the kinetic-energy equation is singular where
  dH*/dH = 0 (Goldstein singularity; Drela and Giles 1987): H = 4 for the
  laminar closure and H = H0(Re_theta) for the turbulent closure. When a
  station reaches this shape factor, or the station Newton has no
  solution, the march stops, marks the station as separated and returns
  the partial result. There is no inverse mode here; the coupled solver
  handles separated flow.

Sources: M. Drela and M. B. Giles, AIAA Journal 25(10), 1987, pp.
1347-1355; M. Drela, MIT GTL Report 187, 1986; K. Hiemenz (1911) and
H. Blasius (1908) for the similarity solutions (as given in H. Schlichting,
Boundary-Layer Theory). No program source code was read or used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ventorum.ibl2d import closure as cl
from ventorum.ibl2d import equations as eq
from ventorum.ibl2d.dual import constant, variables

__all__ = [
    "BoundaryLayerResult",
    "march_direct",
    "march_direct_batch",
    "stagnation_shape_factor",
    "blasius_shape_factor",
    "station_newton",
]


@dataclass
class BoundaryLayerResult:
    """Result of a direct boundary-layer march.

    For :func:`march_direct` the arrays have shape (n,); for
    :func:`march_direct_batch` they have shape (B, n) and the scalars have
    shape (B,). Values after a separation point are NaN.

    Attributes
    ----------
    s : array
        Surface coordinate of the stations [m].
    ue : array
        Edge velocity over the reference speed [-].
    theta : array
        Momentum thickness [m].
    delta_star : array
        Displacement thickness [m].
    H : array
        Shape factor [-].
    Cf : array
        Skin-friction coefficient based on ue [-] (NaN at a stagnation
        point, where ue = 0).
    C_tau : array
        Shear-stress coefficient [-] (NaN at laminar stations).
    N : array
        Amplification exponent [-] (NaN at turbulent stations).
    turbulent : array of bool
        True at turbulent stations.
    transition_s : float or array
        Transition position [m], NaN when the layer stays laminar.
    separation_s : float or array
        Position of the station where the march stopped by separation [m],
        NaN when the layer stays attached.
    separated : bool or array
        True when the march stopped by separation.
    transpiration : array
        Wall transpiration velocity d(ue delta_star)/ds [-] (over the
        reference speed), the source strength for the coupling.
    """

    s: np.ndarray
    ue: np.ndarray
    theta: np.ndarray
    delta_star: np.ndarray
    H: np.ndarray
    Cf: np.ndarray
    C_tau: np.ndarray
    N: np.ndarray
    turbulent: np.ndarray
    transition_s: float | np.ndarray
    separation_s: float | np.ndarray
    separated: bool | np.ndarray
    transpiration: np.ndarray


def _scalar_root(fun, x0: float) -> float:
    """Return a root of a scalar function by the secant method."""
    x1 = x0 + 1e-3
    f0, f1 = fun(x0), fun(x1)
    for _ in range(100):
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, f0, x1, f1 = x1, f1, x2, fun(x2)
        if abs(x1 - x0) < 1e-15:
            break
    return float(x1)


def stagnation_shape_factor() -> float:
    """Return the closure shape factor of the stagnation-point flow.

    Root of 3 f(H) = (2 + H) g(H) with the laminar closure (Hiemenz
    similarity; see :func:`ventorum.ibl2d.equations.similarity_residuals`).

    Returns
    -------
    float
        Shape factor [-] (about 2.24; the exact Hiemenz value is 2.216).
    """

    def fun(h):
        f = float(cl.laminar_cf_factor(h)[0])
        g = float(cl.laminar_dissipation_factor(h)[0])
        return 3.0 * f - (2.0 + h) * g

    return _scalar_root(fun, 2.2)


def blasius_shape_factor() -> float:
    """Return the closure shape factor of the flat-plate flow.

    Root of g(H) = f(H) with the laminar closure (Blasius similarity).

    Returns
    -------
    float
        Shape factor [-] (about 2.59).
    """

    def fun(h):
        return float(cl.laminar_dissipation_factor(h)[0]) - float(cl.laminar_cf_factor(h)[0])

    return _scalar_root(fun, 2.59)


def _xp_of(a):
    """Return the array module of ``a``."""
    if isinstance(a, np.ndarray):
        return np
    import torch

    return torch


def _rowmax(xp, R):
    """Return the largest absolute value of each row of R."""
    if xp is np:
        return np.max(np.abs(R), axis=-1)
    return R.abs().max(-1).values


def _solve3(xp, J, R):
    """Solve the batched 3x3 systems J d = -R; return d and a validity mask."""
    if xp is np:
        det = np.linalg.det(J)
        ok = np.isfinite(det) & (np.abs(det) > 1e-300) & np.all(np.isfinite(R), axis=-1)
        Js = np.where(ok[:, None, None], J, np.eye(3)[None])
        Rs = np.where(ok[:, None], R, 0.0)
        d = np.linalg.solve(Js, -Rs[..., None])[..., 0]
        return d, ok
    det = xp.linalg.det(J)
    ok = xp.isfinite(det) & (det.abs() > 1e-300) & xp.all(xp.isfinite(R), dim=-1)
    eye = xp.eye(3, dtype=J.dtype, device=J.device).expand_as(J)
    Js = xp.where(ok[:, None, None], J, eye)
    Rs = xp.where(ok[:, None], R, xp.zeros_like(R))
    d = xp.linalg.solve(Js, -Rs[..., None])[..., 0]
    return d, ok


def station_newton(kind, st1, ue2, dxi, re, guess, *, ncrit=None, w_forced=None, h_fix=None,
                   max_iter=40, tol=None):
    """Solve the equations of one interval for the downstream station.

    Parameters
    ----------
    kind : int
        Interval kind of :mod:`ventorum.ibl2d.equations` (LAMINAR,
        TURBULENT, WAKE or TRANSITION).
    st1 : tuple of array
        ``(theta, delta_star, c, ue)`` of the upstream station, shape (B,).
    ue2 : array
        Edge velocity of the downstream station [-], shape (B,).
    dxi : array
        Interval length [m], shape (B,).
    re : float or array
        Reynolds number [-].
    guess : tuple of array
        First estimate of ``(theta, delta_star, c)`` downstream.
    ncrit, w_forced : array, optional
        Transition data for the TRANSITION kind.
    h_fix : array, optional
        When given, the kinetic-energy equation is replaced by H = h_fix
        (used only to build a first estimate past a separation point).
    max_iter : int
        Largest number of Newton iterations.
    tol : float or None
        Convergence limit of the largest absolute residual [-]. None selects
        1e-10 in float64 and 1e-5 (about 100 times the unit round-off) in
        float32.

    Returns
    -------
    tuple
        ``(theta, delta_star, c, ok, w)``: the downstream state, a mask of
        the converged stations and, for the TRANSITION kind, the fraction w
        of the transition point (else None).
    """
    xp = _xp_of(ue2)
    if tol is None:
        tol = 1e-10 if (xp is np or ue2.dtype == xp.float64) else 1e-5
    th, ds, c = (g * 1.0 for g in guess)
    ok = None
    w_out = None
    for _ in range(max_iter):
        v_th, v_ds, v_c = variables([th, ds, c], xp=xp)
        like = v_th
        s1 = tuple(constant(a, like) for a in st1)
        u2 = constant(ue2, like)
        s2 = (v_th, v_ds, v_c, u2)
        if kind == eq.TRANSITION:
            r1, r2, r3, w = eq.transition_residuals(s1, s2, dxi, re, ncrit, w_forced)
            w_out = w.val
        else:
            r1, r2, r3 = eq.interval_residuals(kind, s1, s2, dxi, re)
        if h_fix is not None:
            r2 = v_ds / v_th / h_fix - 1.0
        R = xp.stack([r1.val, r2.val, r3.val], -1)
        J = xp.stack([r1.der, r2.der, r3.der], -2)
        d, ok = _solve3(xp, J, R)
        res = _rowmax(xp, R)
        # Step limit: relative changes at most 0.5, H stays above 1.
        lam = 1.0 + 0.0 * th
        rel = xp.maximum(abs(d[:, 0]) / th, abs(d[:, 1]) / ds)
        if kind != eq.LAMINAR:
            rel = xp.maximum(rel, abs(d[:, 2]) / c)
        lam = xp.where(rel > 0.5, 0.5 / rel, lam)
        h_old = ds / th
        for _k in range(30):
            h_new = (ds + lam * d[:, 1]) / (th + lam * d[:, 0])
            bad = h_new < 1.0 + 0.5 * (h_old - 1.0)
            if not bool(xp.any(bad)):
                break
            lam = xp.where(bad, 0.5 * lam, lam)
        th = th + lam * d[:, 0]
        ds = ds + lam * d[:, 1]
        c = c + lam * d[:, 2]
        if bool(xp.all((res < tol) | ~ok)):
            break
    # Final residual check.
    v_th, v_ds, v_c = variables([th, ds, c], xp=xp)
    s1 = tuple(constant(a, v_th) for a in st1)
    s2 = (v_th, v_ds, v_c, constant(ue2, v_th))
    if kind == eq.TRANSITION:
        r1, r2, r3, w = eq.transition_residuals(s1, s2, dxi, re, ncrit, w_forced)
        w_out = w.val
    else:
        r1, r2, r3 = eq.interval_residuals(kind, s1, s2, dxi, re)
    if h_fix is not None:
        r2 = v_ds / v_th / h_fix - 1.0
    R = xp.stack([r1.val, r2.val, r3.val], -1)
    res = _rowmax(xp, R)
    ok = ok & (res < 1e3 * tol) & xp.isfinite(res)
    return th, ds, c, ok, w_out


def _turbulent_ceq_sqrt(xp, th, ds, ue, re):
    """Return sqrt(TRANSITION_CTAU_FACTOR * Ctau_eq) of a state."""
    h = ds / th
    rt = ue * th * re
    hs = cl.turbulent_h_star(h, rt, xp=xp)[0]
    us = cl.slip_velocity(h, hs, xp=xp)[0]
    ceq = cl.equilibrium_shear_coefficient(h, hs, us, xp=xp)[0]
    return xp.sqrt(eq.TRANSITION_CTAU_FACTOR * ceq)


def march_direct_batch(s, ue_batch, Re_ref_batch, xp=None, *, ncrit=9.0, forced_transition_s=None,
                       theta0=None, device=None, dtype=None):
    """March many independent surfaces at once in direct mode.

    The station loop is sequential; the work at each station is vectorized
    over the batch. With ``xp=torch`` the arrays are torch tensors on the
    given device (CPU or CUDA).

    Parameters
    ----------
    s : array_like
        Surface coordinate [m], shape (n,), increasing.
    ue_batch : array_like
        Edge velocity over the reference speed [-], shape (B, n).
    Re_ref_batch : array_like
        Reynolds number V_ref L / nu [-] per surface, shape (B,) (L is the
        unit of s).
    xp : module, optional
        numpy (default) or torch.
    ncrit : float or array_like
        Critical amplification exponent [-]. Default 9.
    forced_transition_s : float, array_like or None
        Forced transition position [m] per surface, or None (free only).
    theta0 : float, array_like or None
        Momentum thickness at s[0] [m] for the Blasius start; None for the
        similarity start.
    device, dtype : optional
        Torch device and dtype (only with ``xp=torch``).

    Returns
    -------
    BoundaryLayerResult
        Arrays of shape (B, n).
    """
    xp = np if xp is None else xp
    if xp is np:
        def arr(a):
            return np.asarray(a, dtype=np.float64)
    else:
        dt = xp.float64 if dtype is None else dtype

        def arr(a):
            return xp.as_tensor(np.asarray(a, dtype=np.float64) if not isinstance(a, xp.Tensor) else a,
                                dtype=dt, device=device)
    sv = arr(s)
    ue = arr(ue_batch)
    if ue.ndim == 1:
        ue = ue[None, :]
    nb, n = ue.shape
    re = arr(Re_ref_batch) * (1.0 + 0.0 * ue[:, 0])
    nc = arr(ncrit) * (1.0 + 0.0 * ue[:, 0])
    big = 1e300
    sf = arr(big if forced_transition_s is None else forced_transition_s) * (1.0 + 0.0 * ue[:, 0])
    nan = float("nan")

    def full(v):
        return v + 0.0 * ue

    TH = full(nan)
    DS = full(nan)
    C = full(nan)
    TURB = (0.0 * ue) > 1.0
    xtr = 0.0 * re + nan
    xsep = 0.0 * re + nan
    stop = (0.0 * re) > 1.0

    # First station.
    if theta0 is None and bool(xp.all(ue[:, 0] <= 0.0)):
        hs0 = stagnation_shape_factor()
        f0 = float(cl.laminar_cf_factor(hs0)[0])
        a = ue[:, 1] / (sv[1] - sv[0])
        th1 = xp.sqrt(f0 / ((2.0 + hs0) * re * a))
        TH[:, 0] = th1
        TH[:, 1] = th1
        DS[:, 0] = hs0 * th1
        DS[:, 1] = hs0 * th1
        C[:, 0] = 0.0
        C[:, 1] = 0.0
        start = 2
    else:
        hb = blasius_shape_factor()
        fb = float(cl.laminar_cf_factor(hb)[0])
        if theta0 is not None:
            TH[:, 0] = arr(theta0) * (1.0 + 0.0 * re)
            DS[:, 0] = hb * TH[:, 0]
            C[:, 0] = 0.0
            start = 1
        elif float(sv[0]) > 0.0:
            TH[:, 0] = xp.sqrt(2.0 * fb * sv[0] / (re * ue[:, 0]))
            DS[:, 0] = hb * TH[:, 0]
            C[:, 0] = 0.0
            start = 1
        else:
            th1 = xp.sqrt(2.0 * fb * (sv[1] - sv[0]) / (re * ue[:, 1]))
            TH[:, 0] = 0.0
            DS[:, 0] = 0.0
            C[:, 0] = 0.0
            TH[:, 1] = th1
            DS[:, 1] = hb * th1
            C[:, 1] = 0.0
            start = 2
    for i in range(start, n):
        th1, ds1, c1, u1 = TH[:, i - 1], DS[:, i - 1], C[:, i - 1], ue[:, i - 1]
        u2 = ue[:, i]
        dx = (sv[i] - sv[i - 1]) * (1.0 + 0.0 * re)
        act = ~stop
        lam = act & ~TURB[:, i - 1]
        trb = act & TURB[:, i - 1]
        new_th = 0.0 * re + nan
        new_ds = 0.0 * re + nan
        new_c = 0.0 * re + nan
        new_turb = TURB[:, i - 1] | False
        failed = (0.0 * re) > 1.0
        sub_s = (th1, ds1, c1, u1)
        if bool(xp.any(lam)):
            idx = xp.nonzero(lam)[0] if xp is np else xp.nonzero(lam)[:, 0]
            st = tuple(a[idx] for a in sub_s)
            s1v = sv[i - 1]
            wf = (sf[idx] - s1v) / (sv[i] - s1v)
            wf = xp.where(wf < 0.0, 0.0 * wf, wf)
            wf = xp.where(sf[idx] < sv[i], wf, 2.0 + 0.0 * wf)
            # Laminar step first; the transition interval is the one where the
            # laminar N reaches ncrit (or that holds the forced point).
            t2, d2, n2, ok, _ = station_newton(eq.LAMINAR, st, u2[idx], dx[idx], re[idx], (st[0], st[1], st[2]))
            new_th[idx] = t2
            new_ds[idx] = d2
            new_c[idx] = n2
            failed[idx] = ~ok
            go = ok & ((n2 >= nc[idx]) | (wf <= 1.0))
            if bool(xp.any(go)):
                jdx = idx[go]
                stj = tuple(a[jdx] for a in sub_s)
                g_c = _turbulent_ceq_sqrt(xp, t2[go], d2[go], u2[jdx], re[jdx])
                t3, d3, c3, ok3, w3 = station_newton(eq.TRANSITION, stj, u2[jdx], dx[jdx], re[jdx],
                                                     (t2[go], d2[go], g_c), ncrit=nc[jdx],
                                                     w_forced=wf[go])
                new_th[jdx] = t3
                new_ds[jdx] = d3
                new_c[jdx] = c3
                new_turb[jdx] = True
                failed[jdx] = ~ok3
                wcl = xp.where(w3 < 0.0, 0.0 * w3, xp.where(w3 > 1.0, 1.0 + 0.0 * w3, w3))
                xtr[jdx] = s1v + wcl * (sv[i] - s1v)
        if bool(xp.any(trb)):
            idx = xp.nonzero(trb)[0] if xp is np else xp.nonzero(trb)[:, 0]
            st = tuple(a[idx] for a in sub_s)
            t2, d2, c2, ok, _ = station_newton(eq.TURBULENT, st, u2[idx], dx[idx], re[idx],
                                               (st[0], st[1], st[2]))
            new_th[idx] = t2
            new_ds[idx] = d2
            new_c[idx] = c2
            failed[idx] = ~ok
        h2 = new_ds / new_th
        hsep_t = cl.separation_shape_factor(u2 * new_th * re, xp=xp)[0]
        hsep = xp.where(new_turb, hsep_t, 4.0 + 0.0 * h2)
        sep = act & (failed | ~(h2 < hsep))
        TH[:, i] = xp.where(sep, nan + 0.0 * h2, new_th)
        DS[:, i] = xp.where(sep, nan + 0.0 * h2, new_ds)
        C[:, i] = xp.where(sep, nan + 0.0 * h2, new_c)
        TURB[:, i] = new_turb & act
        xsep = xp.where(sep, sv[i] + 0.0 * re, xsep)
        stop = stop | sep
    lead = TH == 0.0
    H = xp.where(lead, blasius_shape_factor() + 0.0 * DS, DS / xp.where(lead, 1.0 + 0.0 * TH, TH))
    rt = re[:, None] * ue * TH
    rt = xp.where(lead, 1.0 + 0.0 * rt, rt)
    stag = rt <= 0.0
    rt = xp.where(stag, 1.0 + 0.0 * rt, rt)
    cf_l = 2.0 * cl.laminar_cf_factor(xp.where(TURB, 2.6 + 0.0 * H, H), xp=xp)[0] / rt
    cf_t = cl.turbulent_cf(xp.where(TURB, H, 1.4 + 0.0 * H), xp.where(TURB, rt, 1e4 + 0.0 * rt), xp=xp)[0]
    Cf = xp.where(TURB, cf_t, cf_l)
    Cf = xp.where(stag, nan + 0.0 * Cf, Cf)
    ctau = xp.where(TURB, C * C, nan + 0.0 * C)
    nn = xp.where(TURB, nan + 0.0 * C, C)
    m = ue * DS
    if xp is np:
        trans = np.gradient(m, sv, axis=1) if n > 2 else (m[:, 1:] - m[:, :-1]) / (sv[1:] - sv[:-1])
    else:
        trans = xp.gradient(m, spacing=(sv,), dim=1)[0] if n > 2 else (m[:, 1:] - m[:, :-1]) / (sv[1:] - sv[:-1])
    return BoundaryLayerResult(
        s=sv,
        ue=ue,
        theta=TH,
        delta_star=DS,
        H=H,
        Cf=Cf,
        C_tau=ctau,
        N=nn,
        turbulent=TURB,
        transition_s=xtr,
        separation_s=xsep,
        separated=stop,
        transpiration=trans,
    )


def march_direct(s, ue, Re_ref, *, ncrit=9.0, forced_transition_s=None, theta0=None):
    """March one surface in direct mode (edge velocity given).

    Parameters
    ----------
    s : array_like
        Surface coordinate [m] (or chord units), shape (n,), increasing.
    ue : array_like
        Edge velocity over the reference speed [-], shape (n,). A zero at
        s[0] selects the stagnation-point start (ue proportional to s).
    Re_ref : float
        Reynolds number V_ref L / nu [-] with L the unit of s.
    ncrit : float
        Critical amplification exponent [-]. Default 9.
    forced_transition_s : float or None
        Forced transition position [m], or None for free transition only.
    theta0 : float or None
        Momentum thickness at s[0] [m] for the Blasius start, or None.

    Returns
    -------
    BoundaryLayerResult
        Arrays of shape (n,).
    """
    r = march_direct_batch(
        s,
        np.asarray(ue, dtype=np.float64)[None, :],
        np.array([float(Re_ref)]),
        ncrit=ncrit,
        forced_transition_s=None if forced_transition_s is None else [float(forced_transition_s)],
        theta0=None if theta0 is None else [float(theta0)],
    )
    return BoundaryLayerResult(
        s=r.s,
        ue=r.ue[0],
        theta=r.theta[0],
        delta_star=r.delta_star[0],
        H=r.H[0],
        Cf=r.Cf[0],
        C_tau=r.C_tau[0],
        N=r.N[0],
        turbulent=r.turbulent[0],
        transition_s=float(r.transition_s[0]),
        separation_s=float(r.separation_s[0]),
        separated=bool(r.separated[0]),
        transpiration=r.transpiration[0],
    )
