# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Envelope e^N transition relations of the integral boundary layer.

The relations give the critical momentum-thickness Reynolds number, the
envelope amplification rate per unit momentum-thickness Reynolds number,
and the growth rate of the amplification exponent N along the surface, for
incompressible flow. They are written from the published papers only:

* M. Drela and M. B. Giles, "Viscous-inviscid analysis of transonic and low
  Reynolds number airfoils", AIAA Journal 25(10), 1987, pp. 1347-1355
  (envelope fits to the Orr-Sommerfeld results of the Falkner-Skan family,
  and the conversion from dN/dRe_theta to dN/dxi).
* M. Drela, doctoral thesis, MIT Gas Turbine Laboratory Report 187, 1986.

No program source code was read or used. The labels (E1) to (E4) point to
``docs/theory/ibl2d_closure.md``.

Each function returns the value and its partial derivatives as a tuple and
takes an optional ``xp`` argument (numpy or torch).
"""

from __future__ import annotations

import math


from ventorum.ibl2d.closure import _arr, _mod

__all__ = [
    "critical_re_theta",
    "amplification_rate",
    "growth_factor",
    "transition_growth_rate",
]

_LN10 = math.log(10.0)


def critical_re_theta(H, xp=None):
    """Return the critical Reynolds number Re_theta0 (E1) and its derivative.

    Relation of Drela and Giles (1987)::

        log10 Re_theta0 = (1.415 / (H - 1) - 0.489) tanh(20 / (H - 1) - 12.9)
                          + 3.295 / (H - 1) + 0.44

    Parameters
    ----------
    H : array_like
        Shape factor [-], H > 1.
    xp : module, optional
        Array module (numpy or torch). Default numpy.

    Returns
    -------
    tuple
        ``(Re_theta0, dRe_theta0_dH)`` [-].
    """
    xp = _mod(xp)
    H = _arr(xp, H)
    u = 1.0 / (H - 1.0)
    du = -u * u
    a = 1.415 * u - 0.489
    t = xp.tanh(20.0 * u - 12.9)
    lg = a * t + 3.295 * u + 0.44
    dlg = (1.415 * t + a * (1.0 - t * t) * 20.0 + 3.295) * du
    r0 = xp.exp(_LN10 * lg)
    return r0, r0 * _LN10 * dlg


def amplification_rate(H, xp=None):
    """Return the envelope rate dN/dRe_theta (E2) and its derivative.

    Relation of Drela and Giles (1987)::

        dN/dRe_theta = 0.01 sqrt([2.4 H - 3.7 + 2.5 tanh(1.5 H - 4.65)]^2 + 0.25)

    Parameters
    ----------
    H : array_like
        Shape factor [-].
    xp : module, optional
        Array module (numpy or torch). Default numpy.

    Returns
    -------
    tuple
        ``(dN_dRe_theta, derivative_dH)`` [-].
    """
    xp = _mod(xp)
    H = _arr(xp, H)
    t = xp.tanh(1.5 * H - 4.65)
    b = 2.4 * H - 3.7 + 2.5 * t
    db = 2.4 + 3.75 * (1.0 - t * t)
    s = xp.sqrt(b * b + 0.25)
    return 0.01 * s, 0.01 * b * db / s


def growth_factor(H, xp=None):
    """Return theta dRe_theta/dxi of the Falkner-Skan family (E3).

    Drela and Giles (1987) convert dN/dRe_theta to dN/dxi with the
    Falkner-Skan relations::

        l(H) = (6.54 H - 14.07) / H^2
        m(H) = (0.058 (H - 4)^2 / (H - 1) - 0.068) / l(H)
        theta dRe_theta/dxi = (m + 1) l / 2

    The product (m + 1) l / 2 is evaluated in the form
    (0.058 (H - 4)^2 / (H - 1) - 0.068 + l) / 2, which has no singular point
    where l = 0.

    Parameters
    ----------
    H : array_like
        Shape factor [-], H > 1.
    xp : module, optional
        Array module (numpy or torch). Default numpy.

    Returns
    -------
    tuple
        ``(rho, drho_dH)`` with rho = theta dRe_theta/dxi [-].
    """
    xp = _mod(xp)
    H = _arr(xp, H)
    hm1 = H - 1.0
    h4 = H - 4.0
    p = 0.058 * h4 * h4 / hm1 - 0.068
    dp = 0.058 * (2.0 * h4 * hm1 - h4 * h4) / (hm1 * hm1)
    ell = (6.54 * H - 14.07) / (H * H)
    dell = (6.54 * H - 2.0 * (6.54 * H - 14.07)) / (H * H * H)
    return 0.5 * (p + ell), 0.5 * (dp + dell)


def transition_growth_rate(H, theta, Re_theta, xp=None):
    """Return the envelope growth rate dN/dxi (E4) and its derivatives.

    dN/dxi = (dN/dRe_theta)(H) rho(H) / theta for Re_theta > Re_theta0(H),
    and zero otherwise (Drela and Giles 1987). The switch at Re_theta0 is a
    jump of the source relation; it is kept as published.

    Parameters
    ----------
    H : array_like
        Shape factor [-].
    theta : array_like
        Momentum thickness [m].
    Re_theta : array_like
        Momentum-thickness Reynolds number [-].
    xp : module, optional
        Array module (numpy or torch). Default numpy.

    Returns
    -------
    tuple
        ``(dN_dxi, d_dH, d_dtheta, d_dRe_theta)``: the rate [1/m] and its
        partial derivatives. The derivative with respect to Re_theta is
        zero away from the switch.
    """
    xp = _mod(xp)
    H = _arr(xp, H)
    th = _arr(xp, theta)
    rt = _arr(xp, Re_theta)
    a, da = amplification_rate(H, xp=xp)
    rho, drho = growth_factor(H, xp=xp)
    r0, _ = critical_re_theta(H, xp=xp)
    on = rt > r0
    rate = a * rho / th
    zero = 0.0 * rate
    d_h = (da * rho + a * drho) / th
    return (
        xp.where(on, rate, zero),
        xp.where(on, d_h, zero),
        xp.where(on, -rate / th, zero),
        zero,
    )
