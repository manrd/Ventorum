# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Discrete equations of the integral boundary layer (shared engine).

This module holds the only implementation of the discrete boundary-layer
equations of Ventorum-2D. The direct march of :mod:`ventorum.ibl2d.march`
and the coupled viscous-inviscid Newton solver of
:mod:`ventorum.panel2d.vii` call the same functions, so both solve the same
equations with the same closures.

Equations (incompressible, Drela and Giles 1987)::

    d(theta)/dxi + (2 + H) (theta/ue) due/dxi = Cf/2                      (momentum)
    theta dH*/dxi + H* (1 - H) (theta/ue) due/dxi = 2 CD - H* Cf/2        (kinetic energy)
    (delta/Ctau) dCtau/dxi = 5.6 (Ctau_eq^1/2 - Ctau^1/2)
        + 2 delta {(4/(3 delta*)) [Cf/2 - ((H - 1)/(6.7 H))^2] - (1/ue) due/dxi}   (lag)
    dN/dxi = (dN/dRe_theta)(H) (theta dRe_theta/dxi)(H) / theta               (e^N envelope)

Discretization between two stations 1 and 2 (interval length dxi): the
derivatives are logarithmic differences, for example
``ln(theta2/theta1)``, and the source terms are trapezoidal averages. The
source terms of the momentum and kinetic-energy equations hold a factor
1/ue; it is integrated with the logarithmic mean of ue over the interval,
``dxi ln(ue2/ue1) / (ue2 - ue1)``, which is exact for the linear edge
velocity of a stagnation-point flow and is second-order accurate in
general. The scheme is second order and implicit.

Variables per station: theta [m], delta_star [m], the third variable c
(the amplification exponent N in laminar flow, the square root of the
shear-stress coefficient Ctau in turbulent flow and in the wake) and the
edge velocity ue [-] (over the reference speed).

The functions take :class:`ventorum.ibl2d.dual.Dual` inputs and return
Dual residuals, so the Newton Jacobians are exact.
"""

from __future__ import annotations

from ventorum.ibl2d import closure as cl
from ventorum.ibl2d import transition as tr
from ventorum.ibl2d.dual import Dual, constant, lift, where

__all__ = [
    "LAMINAR",
    "TURBULENT",
    "WAKE",
    "TRANSITION",
    "POST",
    "TRANSITION_CTAU_FACTOR",
    "transition_fraction",
    "post_transition_residuals",
    "laminar_terms",
    "turbulent_terms",
    "interval_residuals",
    "transition_residuals",
    "similarity_residuals",
    "trailing_edge_residuals",
    "laminar_n_prediction",
]

#: Interval kind codes (POST: the turbulent interval after the transition interval).
LAMINAR, TURBULENT, WAKE, TRANSITION, POST = 0, 1, 2, 3, 4

#: Initial shear-stress coefficient after transition, as a fraction of the
#: equilibrium value Ctau_eq [-]. Best reading of Drela and Giles (1987);
#: to be checked against the paper (see docs/theory/ibl2d_march.md).
TRANSITION_CTAU_FACTOR = 0.7


def _inv_log_mean(u1: Dual, u2: Dual) -> Dual:
    """Return ln(u2/u1) / (u2 - u1), the inverse logarithmic mean of u.

    The series ln(1 + x)/x = 1 - x/2 + x^2/3 - x^3/4 + x^4/5 is used for
    |x| < 1e-3 with x = u2/u1 - 1.
    """
    xp = u1.xp
    a = u1.val
    x = u2.val / a - 1.0
    small = abs(x) < 1e-3
    xs = xp.where(small, 1.0 + 0.0 * x, x)
    lg = xp.log(1.0 + xs)
    phi_big = lg / xs
    dphi_big = (xs / (1.0 + xs) - lg) / (xs * xs)
    phi_small = 1.0 - x / 2.0 + x * x / 3.0 - x**3 / 4.0 + x**4 / 5.0
    dphi_small = -0.5 + 2.0 * x / 3.0 - 0.75 * x * x + 0.8 * x**3
    phi = xp.where(small, phi_small, phi_big)
    dphi = xp.where(small, dphi_small, dphi_big)
    q = phi / a
    dq_du1 = -(phi + dphi * (1.0 + x)) / (a * a)
    dq_du2 = dphi / (a * a)
    return lift((q, dq_du1, dq_du2), [u1, u2])


def laminar_terms(th: Dual, ds: Dual, nn: Dual, ue: Dual, re: float) -> dict:
    """Return the station terms of the laminar equations.

    Parameters
    ----------
    th, ds, nn, ue : Dual
        Momentum thickness [m], displacement thickness [m], amplification
        exponent N [-] and edge velocity [-].
    re : float
        Reynolds number of the reference length and speed [-].

    Returns
    -------
    dict
        ``H``, ``Hs`` (H*), ``amom`` (ue Cf / (2 theta)), ``ake``
        (ue (2 CD/H* - Cf/2) / theta), ``src3`` (dN/dxi), ``cf``, ``rt``.
    """
    xp = th.xp
    h = ds / th
    rt = ue * th * re
    hs = lift(cl.laminar_h_star(h.val, xp=xp), [h])
    f = lift(cl.laminar_cf_factor(h.val, xp=xp), [h])
    g = lift(cl.laminar_dissipation_factor(h.val, xp=xp), [h])
    den = th * th * re
    rate = lift(tr.transition_growth_rate(h.val, th.val, rt.val, xp=xp), [h, th, rt])
    dndr = lift(tr.amplification_rate(h.val, xp=xp), [h])
    rho = lift(tr.growth_factor(h.val, xp=xp), [h])
    r0 = lift(tr.critical_re_theta(h.val, xp=xp), [h])
    return {
        "H": h,
        "Hs": hs,
        "amom": f / den,
        "ake": (g - f) / den,
        "src3": rate,
        "rtil": dndr * rho / th,
        "exc": rt - r0,
        "cf": 2.0 * f / rt,
        "rt": rt,
    }


def turbulent_terms(th: Dual, ds: Dual, c: Dual, ue: Dual, re: float, wake: bool = False) -> dict:
    """Return the station terms of the turbulent or wake equations.

    In the wake (``wake=True``) the skin friction is zero and the
    dissipation coefficient is doubled, because the wake has two shear
    layers (Drela and Giles 1987).

    Parameters
    ----------
    th, ds, c, ue : Dual
        Momentum thickness [m], displacement thickness [m], square root of
        the shear-stress coefficient [-] and edge velocity [-].
    re : float
        Reynolds number of the reference length and speed [-].
    wake : bool
        True for a wake station.

    Returns
    -------
    dict
        ``H``, ``Hs``, ``amom``, ``ake``, ``src3`` (the lag-equation source
        d ln(c)/dxi + d ln(ue)/dxi), ``cf``, ``rt``, ``ceq`` (square root
        of Ctau_eq) and ``us``.
    """
    xp = th.xp
    h = ds / th
    rt = ue * th * re
    hs = lift(cl.turbulent_h_star(h.val, rt.val, xp=xp), [h, rt])
    us = lift(cl.slip_velocity(h.val, hs.val, xp=xp), [h, hs])
    ct = c * c
    if wake:
        cf = constant(0.0, th)
        cd = 2.0 * (ct * (1.0 - us))
    else:
        cf = lift(cl.turbulent_cf(h.val, rt.val, xp=xp), [h, rt])
        cd = lift(cl.turbulent_dissipation(cf.val, us.val, ct.val, xp=xp), [cf, us, ct])
    ceq2 = lift(cl.equilibrium_shear_coefficient(h.val, hs.val, us.val, xp=xp), [h, hs, us])
    ceq = ceq2.sqrt()
    delta = lift(cl.layer_thickness(th.val, h.val, xp=xp), [th, h])
    g = (h - 1.0) / (cl.EQUILIBRIUM_A * h)
    lag = cl.LAG_CONSTANT * (ceq - c) / (2.0 * delta) + (4.0 / 3.0) * (0.5 * cf - g * g) / ds
    return {
        "H": h,
        "Hs": hs,
        "amom": ue * cf / (2.0 * th),
        "ake": ue * (2.0 * cd / hs - 0.5 * cf) / th,
        "src3": lag,
        "cf": cf,
        "rt": rt,
        "ceq": ceq,
        "us": us,
    }


def _terms(kind: int, st, re: float) -> dict:
    """Return the station terms of one interval kind."""
    th, ds, c, ue = st
    if kind == LAMINAR:
        return laminar_terms(th, ds, c, ue, re)
    return turbulent_terms(th, ds, c, ue, re, wake=(kind == WAKE))


def _pair(kind: int, ta: dict, tb: dict, sa, sb, dxi, wb: float = 0.5):
    """Return the three residuals between states a and b of one kind.

    ``wb`` is the weight of state b in the averages of the source terms and
    of H: 0.5 gives the trapezoidal rule, 1 the implicit (backward Euler)
    rule.
    """
    wa = 1.0 - wb
    tha, _, ca, ua = sa
    # wb may be a Dual (post-transition interval); the products then carry its derivatives.
    thb, _, cb, ub = sb
    lu = (ub / ua).log()
    q = _inv_log_mean(ua, ub)
    hbar = wa * ta["H"] + wb * tb["H"]
    r1 = (thb / tha).log() + (2.0 + hbar) * lu - dxi * q * (wa * ta["amom"] + wb * tb["amom"])
    r2 = (tb["Hs"] / ta["Hs"]).log() + (1.0 - hbar) * lu - dxi * q * (wa * ta["ake"] + wb * tb["ake"])
    if kind == LAMINAR:
        r3 = cb - ca - _amplification_increase(ta, tb, dxi)
    else:
        r3 = (cb / ca).log() + lu - dxi * (wa * ta["src3"] + wb * tb["src3"])
    return r1, r2, r3


def _amplification_increase(ta: dict, tb: dict, dxi):
    """Return the increase of N over a laminar interval.

    The growth rate is rtil for Re_theta above Re_theta0 and zero below it
    (the published switch of the envelope method). The increase is
    dN = dxi phi rtil_1: the rate of the upstream station times the
    fraction phi of the interval where the excess e = Re_theta - Re_theta0
    (taken linear between the stations) is positive. The rule uses the
    upstream rate so that the transition interval (where the downstream
    station is turbulent) and the laminar intervals use the same increase:
    the transition point then moves continuously from one interval to the
    next. The increase is continuous in the station states, which the Newton
    iteration needs. The rule is first order in dxi.
    """
    e1, e2 = ta["exc"], tb["exc"]
    p1 = e1.val > 0.0
    p2 = e2.val > 0.0
    den = e2 - e1
    den = where(abs(den.val) > 1e-300, den, constant(1.0, den))
    one = constant(1.0, e1)
    zero = constant(0.0, e1)
    phi = where(p1 & p2, one, where(p2, e2 / den, where(p1, -(e1 / den), zero)))
    return dxi * (phi * ta["rtil"])


def interval_residuals(kind: int, s1, s2, dxi, re: float):
    """Return the residuals of a laminar, turbulent or wake interval.

    Parameters
    ----------
    kind : int
        LAMINAR, TURBULENT or WAKE.
    s1, s2 : tuple of Dual
        ``(theta, delta_star, c, ue)`` at the upstream and the downstream
        station.
    dxi : array
        Interval length [m].
    re : float
        Reynolds number [-].

    Returns
    -------
    tuple of Dual
        Momentum, kinetic-energy and third-equation residuals [-].
    """
    t1 = _terms(kind, s1, re)
    t2 = _terms(kind, s2, re)
    return _pair(kind, t1, t2, s1, s2, dxi)


def laminar_n_prediction(s1, s2, dxi, re: float) -> Dual:
    """Return N at station 2 predicted by the laminar envelope equation.

    The increase over the interval is that of the laminar N equation (see
    :func:`_amplification_increase`), with station 2 evaluated with the
    laminar closure.

    Parameters
    ----------
    s1, s2 : tuple of Dual
        ``(theta, delta_star, c, ue)``; ``c`` of station 1 is N1.
    dxi : array
        Interval length [m].
    re : float
        Reynolds number [-].

    Returns
    -------
    Dual
        Predicted N at station 2 [-].
    """
    t1 = laminar_terms(*s1, re)
    t2 = laminar_terms(s2[0], s2[1], s2[2], s2[3], re)
    return s1[2] + _amplification_increase(t1, t2, dxi)


def _clip01(w: Dual) -> Dual:
    """Return w clipped to [0, 1]."""
    return where(w.val < 0.0, constant(0.0, w), where(w.val > 1.0, constant(1.0, w), w))


def transition_fraction(s1, dxi, re: float, ncrit, w_forced) -> Dual:
    """Return the fraction w of an interval at the transition point (not clipped).

    For free transition N grows linearly from N1 by the increase of the
    laminar N equation with the rate of station 1 (the whole interval grows
    when Re_theta > Re_theta0 at station 1: the onset of amplification lies
    upstream of the transition interval), so w = (ncrit - N1) / dN; w = 2
    when N does not grow. For forced transition w is the given geometric
    fraction; the smaller value applies.

    Parameters
    ----------
    s1 : tuple of Dual
        ``(theta, delta_star, N, ue)`` of the laminar station 1.
    dxi : array
        Interval length [m].
    re : float
        Reynolds number [-].
    ncrit : array
        Critical amplification exponent [-].
    w_forced : array
        Fraction of the forced transition point, or a value above 1.

    Returns
    -------
    Dual
        The fraction w [-].
    """
    th1, ds1, n1, u1 = s1
    ta = laminar_terms(th1, ds1, n1, u1, re)
    inc = where(ta["exc"].val > 0.0, dxi * ta["rtil"], constant(0.0, th1))
    grows = inc.val > 0.0
    incs = where(grows, inc, constant(1.0, inc))
    w_free = where(grows, (ncrit - n1) / incs, constant(2.0, inc))
    wf = constant(w_forced, w_free)
    return where(wf.val < w_free.val, wf, w_free)


def post_transition_residuals(s0, s1, s2, dxi0, dxi, re: float, ncrit, w_forced0):
    """Return the residuals of the turbulent interval that follows the transition interval.

    The transition interval (stations 0 and 1, transition at the fraction w)
    integrates its turbulent part with the implicit rule. So that the
    solution changes continuously when the transition point passes station
    1 (w = 1 here and w = 0 in the next interval give the same equations),
    this interval uses the downstream weight wb = (1 + w) / 2: implicit at
    w = 1, trapezoidal at w = 0.

    Parameters
    ----------
    s0 : tuple of Dual
        ``(theta, delta_star, N, ue)`` of the last laminar station.
    s1, s2 : tuple of Dual
        ``(theta, delta_star, c, ue)`` of the two turbulent stations.
    dxi0, dxi : array
        Lengths of the transition interval and of this interval [m].
    re : float
        Reynolds number [-].
    ncrit, w_forced0 : array
        Transition data of the transition interval.

    Returns
    -------
    tuple of Dual
        Momentum, kinetic-energy and lag residuals [-].
    """
    wc = _clip01(transition_fraction(s0, dxi0, re, ncrit, w_forced0))
    t1 = turbulent_terms(*s1, re)
    t2 = turbulent_terms(*s2, re)
    return _pair(TURBULENT, t1, t2, s1, s2, dxi, wb=0.5 + 0.5 * wc)


def transition_residuals(s1, s2, dxi, re: float, ncrit, w_forced):
    """Return the residuals of the interval that holds the transition point.

    Station 1 is laminar (third variable N), station 2 is turbulent (third
    variable sqrt(Ctau)). The transition point divides the interval at the
    fraction w. For free transition, N grows linearly over the interval by
    the increase of the laminar N equation (:func:`_amplification_increase`,
    which uses the growth rate of station 1; the whole interval grows when
    Re_theta > Re_theta0 at station 1), so w = (ncrit - N1) / dN;
    the laminar state of station 2 does not enter, which keeps the coupled
    Newton iteration stable when the transition point is close to station
    2, and the transition point moves continuously between intervals. For
    forced
    transition, w is the given geometric fraction; the smaller value
    applies. w is clipped to [0, 1] (w = 0 when N1 already exceeds ncrit).
    The state at the transition point is the linear interpolation of theta,
    delta_star and ue. The laminar equations hold on the part before the
    transition point and the turbulent equations on the part after it; the
    momentum and kinetic-energy residuals of the two parts are added. The
    shear stress at the transition point is
    Ctau = TRANSITION_CTAU_FACTOR * Ctau_eq. The closures jump at the
    transition point (laminar to turbulent), and the turbulent layer then
    relaxes over a few thicknesses; a central average over this jump makes
    the stiff relaxation overshoot. The turbulent part therefore uses the
    implicit (backward Euler) rule, which damps it. All other intervals use
    the trapezoidal rule.

    Parameters
    ----------
    s1, s2 : tuple of Dual
        ``(theta, delta_star, c, ue)`` of the laminar and the turbulent
        station.
    dxi : array
        Interval length [m].
    re : float
        Reynolds number [-].
    ncrit : array
        Critical amplification exponent [-].
    w_forced : array
        Fraction of the interval at the forced transition point, or a value
        above 1 when no forced transition point is in the interval.

    Returns
    -------
    tuple
        ``(r1, r2, r3, w)``: the three residuals (Dual) and the fraction w
        (Dual, not clipped; above 1 when N does not reach ncrit in the
        interval and the point is not forced).
    """
    xp = s1[0].xp
    th1, ds1, n1, u1 = s1
    th2, ds2, c2, u2 = s2
    w = transition_fraction(s1, dxi, re, ncrit, w_forced)
    wc = _clip01(w)
    tht = th1 + wc * (th2 - th1)
    dst = ds1 + wc * (ds2 - ds1)
    ut = u1 + wc * (u2 - u1)
    # Laminar part from station 1 to the transition point.
    nt = constant(0.0, n1) + ncrit
    sa = (th1, ds1, n1, u1)
    sb = (tht, dst, nt, ut)
    ta = laminar_terms(*sa, re)
    tb = laminar_terms(*sb, re)
    ra1, ra2, _ = _pair(LAMINAR, ta, tb, sa, sb, wc * dxi)
    # Turbulent part from the transition point to station 2.
    ht = dst / tht
    rtt = ut * tht * re
    hst = lift(cl.turbulent_h_star(ht.val, rtt.val, xp=xp), [ht, rtt])
    ust = lift(cl.slip_velocity(ht.val, hst.val, xp=xp), [ht, hst])
    ceq2 = lift(cl.equilibrium_shear_coefficient(ht.val, hst.val, ust.val, xp=xp), [ht, hst, ust])
    ct = (TRANSITION_CTAU_FACTOR * ceq2).sqrt()
    sc = (tht, dst, ct, ut)
    tc = turbulent_terms(*sc, re)
    td = turbulent_terms(th2, ds2, c2, u2, re)
    rb1, rb2, rb3 = _pair(TURBULENT, tc, td, sc, s2, (1.0 - wc) * dxi, wb=1.0)
    return ra1 + rb1, ra2 + rb2, rb3, w


def similarity_residuals(th: Dual, ds: Dual, nn: Dual, a: Dual, re: float):
    """Return the residuals of the first station (stagnation-point similarity).

    Near a stagnation point ue = a xi. The similarity solution has constant
    theta and H; the momentum and kinetic-energy equations give
    (2 + H) Re a theta^2 = f(H) and 3 f(H) = (2 + H) g(H), with
    f = Re_theta Cf/2 and g = Re_theta 2 CD / H* of the laminar closure
    (Hiemenz flow; the closure gives H = 2.24 against the exact 2.216).

    Parameters
    ----------
    th, ds, nn : Dual
        Momentum thickness [m], displacement thickness [m] and N [-].
    a : Dual
        Edge-velocity gradient due/dxi at the stagnation point [1/m].
    re : float
        Reynolds number [-].

    Returns
    -------
    tuple of Dual
        Three residuals [-]; the third sets N = 0.
    """
    xp = th.xp
    h = ds / th
    f = lift(cl.laminar_cf_factor(h.val, xp=xp), [h])
    g = lift(cl.laminar_dissipation_factor(h.val, xp=xp), [h])
    r1 = 1.0 - f / ((2.0 + h) * re * a * th * th)
    r2 = 3.0 * f / ((2.0 + h) * g) - 1.0
    return r1, r2, nn


def flat_plate_residuals(th: Dual, ds: Dual, nn: Dual, xi, ue, re: float):
    """Return the residuals of a first station on a flat plate (Blasius).

    For ue constant the similarity solution has theta^2 = 2 f(H) xi /
    (Re ue) and g(H) = f(H) (closure value of the Blasius shape factor,
    H = 2.59).

    Parameters
    ----------
    th, ds, nn : Dual
        Momentum thickness [m], displacement thickness [m] and N [-].
    xi : array
        Distance from the leading edge [m].
    ue : array
        Edge velocity [-].
    re : float
        Reynolds number [-].

    Returns
    -------
    tuple of Dual
        Three residuals [-]; the third sets N = 0.
    """
    xp = th.xp
    h = ds / th
    f = lift(cl.laminar_cf_factor(h.val, xp=xp), [h])
    g = lift(cl.laminar_dissipation_factor(h.val, xp=xp), [h])
    r1 = 1.0 - 2.0 * f * xi / (re * ue * th * th)
    r2 = g / f - 1.0
    return r1, r2, nn


def trailing_edge_residuals(up, lo, wk, gap, re: float, up_turb, lo_turb):
    """Return the residuals that start the wake at the trailing edge.

    The wake starts with theta_w = theta_upper + theta_lower and
    delta_star_w = delta_star_upper + delta_star_lower + h, with h the
    trailing-edge gap normal to the bisector (Drela 1989). The initial
    shear-stress coefficient of the wake is the theta-weighted mean of the
    two sides; a side that is still laminar at the trailing edge enters
    with TRANSITION_CTAU_FACTOR * Ctau_eq (it transitions at the edge).

    Parameters
    ----------
    up, lo : tuple of Dual
        ``(theta, delta_star, c, ue)`` of the upper and lower trailing-edge
        stations.
    wk : tuple of Dual
        ``(theta, delta_star, c)`` of the first wake station.
    gap : array
        Trailing-edge gap [m].
    re : float
        Reynolds number [-].
    up_turb, lo_turb : array of bool
        True where the side is turbulent at the trailing edge.

    Returns
    -------
    tuple of Dual
        Three relative residuals [-].
    """
    xp = up[0].xp

    def side_ctau(st, turb):
        th, ds, c, ue = st
        h = ds / th
        rt = ue * th * re
        hs = lift(cl.turbulent_h_star(h.val, rt.val, xp=xp), [h, rt])
        us = lift(cl.slip_velocity(h.val, hs.val, xp=xp), [h, hs])
        ceq2 = lift(cl.equilibrium_shear_coefficient(h.val, hs.val, us.val, xp=xp), [h, hs, us])
        return where(turb, c * c, TRANSITION_CTAU_FACTOR * ceq2)

    cu = side_ctau(up, up_turb)
    cl_ = side_ctau(lo, lo_turb)
    thw, dsw, cw = wk
    th_sum = up[0] + lo[0]
    cmix = ((up[0] * cu + lo[0] * cl_) / th_sum).sqrt()
    r1 = 1.0 - th_sum / thw
    r2 = 1.0 - (up[1] + lo[1] + gap) / dsw
    r3 = 1.0 - cmix / cw
    return r1, r2, r3
