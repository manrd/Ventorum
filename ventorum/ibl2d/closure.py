# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Closure relations of the two-equation integral boundary layer.

The relations close the integral momentum and kinetic-energy equations and
the shear-stress lag equation for incompressible flow (edge Mach number 0,
so the kinematic shape factor equals the shape factor H). They are written
from the published papers only:

* M. Drela and M. B. Giles, "Viscous-inviscid analysis of transonic and low
  Reynolds number airfoils", AIAA Journal 25(10), 1987, pp. 1347-1355
  (closure relations, lag equation, boundary-layer thickness).
* M. Drela, "Two-dimensional transonic aerodynamic design and analysis using
  the Euler equations", doctoral thesis, MIT Gas Turbine Laboratory Report
  187, 1986 (the same relations with their derivation).
* T. J. Swafford, "Analytical approximation of two-dimensional separated
  turbulent boundary-layer velocity profiles", AIAA Journal 21(6), 1983,
  pp. 923-926 (the turbulent profile family behind the skin-friction fit).
* P. D. Green, D. G. Weeks and J. W. F. Brooman, "Prediction of turbulent
  boundary layers and wakes in compressible flow by a lag-entrainment
  method", ARC R and M 3791, 1977 (the lag-entrainment idea).

No program source code was read or used. The labels (C1) to (C10) point to
``docs/theory/ibl2d_closure.md``, which gives each relation, its source,
its range of validity and the points that are to be checked against the
paper (the equation numbers of the paper).

Each function is a pure function of arrays. It returns the value and its
partial derivatives with respect to every input, as a tuple. Each function
takes an optional ``xp`` argument (default numpy); with ``xp=torch`` the
same code runs on torch tensors on the CPU or on CUDA. Where a relation has
two branches, the code keeps the branches of the source; all branch points
are continuous in value and in first derivative.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "LAG_CONSTANT",
    "EQUILIBRIUM_A",
    "laminar_h_star",
    "laminar_cf_factor",
    "laminar_dissipation_factor",
    "separation_shape_factor",
    "turbulent_h_star",
    "turbulent_cf",
    "slip_velocity",
    "equilibrium_shear_coefficient",
    "turbulent_dissipation",
    "layer_thickness",
]

#: Rate constant of the shear-stress lag equation (C9) [-].
LAG_CONSTANT = 5.6
#: Constant A of the equilibrium locus G = A sqrt(1 + B beta), used in (C9) [-].
EQUILIBRIUM_A = 6.7

_LN10 = math.log(10.0)


def _mod(xp):
    """Return the array module (numpy by default)."""
    return np if xp is None else xp


def _arr(xp, x):
    """Return ``x`` as a floating-point array of the ``xp`` module."""
    if xp is np:
        return np.asarray(x, dtype=np.float64)
    if isinstance(x, xp.Tensor):
        return x if x.is_floating_point() else x.double()
    return xp.as_tensor(x, dtype=xp.float64)


def _pos(xp, x):
    """Return ``max(x, 0)`` elementwise."""
    return xp.where(x > 0.0, x, 0.0 * x)


def laminar_h_star(H, xp=None):
    """Return the laminar energy shape factor (C1) and its derivative.

    Fit to the Falkner-Skan profile family (Drela and Giles 1987; Drela
    1986)::

        H* = 1.515 + 0.076 (4 - H)^2 / H    for H < 4
        H* = 1.515 + 0.040 (H - 4)^2 / H    for H >= 4

    Parameters
    ----------
    H : array_like
        Shape factor delta_star / theta [-].
    xp : module, optional
        Array module (numpy or torch). Default numpy.

    Returns
    -------
    tuple
        ``(H_star, dH_star_dH)`` [-].
    """
    xp = _mod(xp)
    H = _arr(xp, H)
    a = _pos(xp, 4.0 - H)
    b = _pos(xp, H - 4.0)
    low = H < 4.0
    hs = 1.515 + xp.where(low, 0.076 * a * a / H, 0.040 * b * b / H)
    d = xp.where(low, -0.076 * a * (H + 4.0) / (H * H), 0.040 * b * (H + 4.0) / (H * H))
    return hs, d


def laminar_cf_factor(H, xp=None):
    """Return the laminar skin-friction factor Re_theta Cf / 2 (C2).

    Fit to the Falkner-Skan profile family (Drela and Giles 1987; Drela
    1986)::

        Re_theta Cf/2 = -0.067 + 0.01977 (7.4 - H)^2 / (H - 1)      for H < 7.4
        Re_theta Cf/2 = -0.067 + 0.022 (1 - 1.4 / (H - 6))^2        for H >= 7.4

    Parameters
    ----------
    H : array_like
        Shape factor [-], H > 1.
    xp : module, optional
        Array module (numpy or torch). Default numpy.

    Returns
    -------
    tuple
        ``(f, df_dH)`` with f = Re_theta Cf / 2 [-].
    """
    xp = _mod(xp)
    H = _arr(xp, H)
    low = H < 7.4
    a = _pos(xp, 7.4 - H)
    hm1 = H - 1.0
    f_low = -0.067 + 0.01977 * a * a / hm1
    d_low = -0.01977 * a * (H + 5.4) / (hm1 * hm1)
    hm6 = xp.where(low, 1.4 + 0.0 * H, H - 6.0)
    q = 1.0 - 1.4 / hm6
    f_high = -0.067 + 0.022 * q * q
    d_high = 0.044 * q * 1.4 / (hm6 * hm6)
    return xp.where(low, f_low, f_high), xp.where(low, d_low, d_high)


def laminar_dissipation_factor(H, xp=None):
    """Return the laminar dissipation factor Re_theta 2 CD / H* (C3).

    Fit to the Falkner-Skan profile family (Drela and Giles 1987; Drela
    1986)::

        Re_theta 2CD/H* = 0.207 + 0.00205 (4 - H)^5.5                          for H < 4
        Re_theta 2CD/H* = 0.207 - 0.003 (H - 4)^2 / (1 + 0.02 (H - 4)^2)       for H >= 4

    Parameters
    ----------
    H : array_like
        Shape factor [-].
    xp : module, optional
        Array module (numpy or torch). Default numpy.

    Returns
    -------
    tuple
        ``(g, dg_dH)`` with g = Re_theta 2 CD / H* [-].
    """
    xp = _mod(xp)
    H = _arr(xp, H)
    low = H < 4.0
    a = _pos(xp, 4.0 - H)
    b = _pos(xp, H - 4.0)
    den = 1.0 + 0.02 * b * b
    g = xp.where(low, 0.207 + 0.00205 * a**5.5, 0.207 - 0.003 * b * b / den)
    d = xp.where(low, -0.00205 * 5.5 * a**4.5, -0.006 * b / (den * den))
    return g, d


def _h0(xp, Rt):
    """Return the shape factor H0 of the turbulent H* relation and dH0/dRt."""
    big = Rt > 400.0
    safe = xp.where(big, Rt, 400.0 + 0.0 * Rt)
    h0 = xp.where(big, 3.0 + 400.0 / safe, 4.0 + 0.0 * Rt)
    dh0 = xp.where(big, -400.0 / (safe * safe), 0.0 * Rt)
    return h0, dh0


def separation_shape_factor(Re_theta, xp=None):
    """Return the turbulent shape factor H0 at the minimum of H* (C4).

    H0 = 3 + 400 / Re_theta for Re_theta > 400, else 4 (Drela and Giles
    1987). In a direct march (edge velocity given) the kinetic-energy
    equation is singular where dH*/dH = 0, which is at H = H0 for the
    turbulent relation and at H = 4 for the laminar relation (C1).

    Parameters
    ----------
    Re_theta : array_like
        Momentum-thickness Reynolds number [-].
    xp : module, optional
        Array module (numpy or torch). Default numpy.

    Returns
    -------
    tuple
        ``(H0, dH0_dRe_theta)``.
    """
    xp = _mod(xp)
    return _h0(xp, _arr(xp, Re_theta))


def turbulent_h_star(H, Re_theta, xp=None):
    """Return the turbulent energy shape factor (C4) and its derivatives.

    Relation of Drela and Giles (1987), incompressible form, with
    H0 = 3 + 400 / Re_theta for Re_theta > 400 and H0 = 4 otherwise::

        H* = 1.505 + 4/Re_theta + (0.165 - 1.6/sqrt(Re_theta)) (H0 - H)^1.6 / H        for H < H0
        H* = 1.505 + 4/Re_theta + (H - H0)^2 [0.04/H + 0.007 ln(Re_theta)
                                               / (H - H0 + 4/ln(Re_theta))^2]       for H >= H0

    Parameters
    ----------
    H : array_like
        Shape factor [-].
    Re_theta : array_like
        Momentum-thickness Reynolds number [-], Re_theta > 1.
    xp : module, optional
        Array module (numpy or torch). Default numpy.

    Returns
    -------
    tuple
        ``(H_star, dH_star_dH, dH_star_dRe_theta)``.
    """
    xp = _mod(xp)
    H = _arr(xp, H)
    Rt = _arr(xp, Re_theta)
    h0, dh0 = _h0(xp, Rt)
    low = H < h0
    sq = xp.sqrt(Rt)
    base = 1.505 + 4.0 / Rt
    dbase = -4.0 / (Rt * Rt)
    # Branch H < H0.
    k = 0.165 - 1.6 / sq
    dk = 0.8 / (Rt * sq)
    a = _pos(xp, h0 - H)
    a06 = a**0.6
    a16 = a06 * a
    hs_low = base + k * a16 / H
    dh_low = -k * a06 * (0.6 * H + h0) / (H * H)
    dr_low = dbase + dk * a16 / H + k * 1.6 * a06 / H * dh0
    # Branch H >= H0.
    d = _pos(xp, H - h0)
    lr = xp.log(Rt)
    q = d + 4.0 / lr
    br = 0.04 / H + 0.007 * lr / (q * q)
    hs_high = base + d * d * br
    dbr_dh = -0.04 / (H * H) - 0.014 * lr / (q * q * q)
    dh_high = 2.0 * d * br + d * d * dbr_dh
    dd_dr = -dh0
    dq_dr = -dh0 - 4.0 / (lr * lr * Rt)
    dbr_dr = 0.007 / (Rt * q * q) - 0.014 * lr / (q * q * q) * dq_dr
    dr_high = dbase + 2.0 * d * dd_dr * br + d * d * dbr_dr
    return (
        xp.where(low, hs_low, hs_high),
        xp.where(low, dh_low, dh_high),
        xp.where(low, dr_low, dr_high),
    )


def turbulent_cf(H, Re_theta, xp=None):
    """Return the turbulent skin-friction coefficient (C5) and its derivatives.

    Fit of Drela and Giles (1987) to the Swafford (1983) profile family,
    incompressible form::

        Cf = 0.3 exp(-1.33 H) (log10 Re_theta)^(-1.74 - 0.31 H)
             + 0.00011 [tanh(4 - H / 0.875) - 1]

    Parameters
    ----------
    H : array_like
        Shape factor [-].
    Re_theta : array_like
        Momentum-thickness Reynolds number [-], Re_theta > 1.
    xp : module, optional
        Array module (numpy or torch). Default numpy.

    Returns
    -------
    tuple
        ``(Cf, dCf_dH, dCf_dRe_theta)``.
    """
    xp = _mod(xp)
    H = _arr(xp, H)
    Rt = _arr(xp, Re_theta)
    l10 = xp.log(Rt) / _LN10
    p = -1.74 - 0.31 * H
    a = 0.3 * xp.exp(-1.33 * H) * xp.exp(p * xp.log(l10))
    da_dh = a * (-1.33 - 0.31 * xp.log(l10))
    da_dr = a * p / (l10 * Rt * _LN10)
    t = xp.tanh(4.0 - H / 0.875)
    b = 0.00011 * (t - 1.0)
    db_dh = -0.00011 * (1.0 - t * t) / 0.875
    return a + b, da_dh + db_dh, da_dr


def slip_velocity(H, H_star, xp=None):
    """Return the normalized slip velocity Us (C6) and its derivatives.

    Relation of Drela and Giles (1987), incompressible form::

        Us = (H* / 2) (1 - (4/3) (H - 1) / H)

    Parameters
    ----------
    H : array_like
        Shape factor [-].
    H_star : array_like
        Energy shape factor [-].
    xp : module, optional
        Array module (numpy or torch). Default numpy.

    Returns
    -------
    tuple
        ``(Us, dUs_dH, dUs_dH_star)`` [-].
    """
    xp = _mod(xp)
    H = _arr(xp, H)
    hs = _arr(xp, H_star)
    fac = 1.0 - (4.0 / 3.0) * (H - 1.0) / H
    us = 0.5 * hs * fac
    return us, -2.0 * hs / (3.0 * H * H), 0.5 * fac


def equilibrium_shear_coefficient(H, H_star, U_s, xp=None):
    """Return the equilibrium shear-stress coefficient (C7) and its derivatives.

    Relation of Drela and Giles (1987), from the equilibrium locus
    G = A sqrt(1 + B beta), incompressible form::

        Ctau_eq = H* (0.015 / (1 - Us)) (H - 1)^3 / H^3

    Parameters
    ----------
    H : array_like
        Shape factor [-], H > 1.
    H_star : array_like
        Energy shape factor [-].
    U_s : array_like
        Normalized slip velocity [-], Us < 1.
    xp : module, optional
        Array module (numpy or torch). Default numpy.

    Returns
    -------
    tuple
        ``(Ctau_eq, dCtau_eq_dH, dCtau_eq_dH_star, dCtau_eq_dUs)`` [-].
    """
    xp = _mod(xp)
    H = _arr(xp, H)
    hs = _arr(xp, H_star)
    us = _arr(xp, U_s)
    hm1 = H - 1.0
    c = hs * 0.015 / (1.0 - us) * hm1**3 / H**3
    return c, c * 3.0 / (H * hm1), c / hs, c / (1.0 - us)


def turbulent_dissipation(Cf, U_s, C_tau, xp=None):
    """Return the turbulent dissipation coefficient (C8) and its derivatives.

    Relation of Drela and Giles (1987): wall-layer part plus outer-layer
    part, with the shear-stress coefficient C_tau of the lag equation::

        CD = (Cf / 2) Us + C_tau (1 - Us)

    Parameters
    ----------
    Cf : array_like
        Skin-friction coefficient [-] (zero in a wake).
    U_s : array_like
        Normalized slip velocity [-].
    C_tau : array_like
        Shear-stress coefficient [-].
    xp : module, optional
        Array module (numpy or torch). Default numpy.

    Returns
    -------
    tuple
        ``(CD, dCD_dCf, dCD_dUs, dCD_dC_tau)`` [-].
    """
    xp = _mod(xp)
    cf = _arr(xp, Cf)
    us = _arr(xp, U_s)
    ct = _arr(xp, C_tau)
    cd = 0.5 * cf * us + ct * (1.0 - us)
    return cd, 0.5 * us, 0.5 * cf - ct, 1.0 - us


def layer_thickness(theta, H, xp=None):
    """Return the boundary-layer thickness delta (C10) and its derivatives.

    Correlation of Drela and Giles (1987), used in the lag equation::

        delta = theta (3.15 + 1.72 / (H - 1)) + delta_star,   delta_star = H theta

    Parameters
    ----------
    theta : array_like
        Momentum thickness [m].
    H : array_like
        Shape factor [-], H > 1.
    xp : module, optional
        Array module (numpy or torch). Default numpy.

    Returns
    -------
    tuple
        ``(delta, ddelta_dtheta, ddelta_dH)``: delta [m], the derivative
        with respect to theta at constant H [-], and the derivative with
        respect to H at constant theta [m].
    """
    xp = _mod(xp)
    th = _arr(xp, theta)
    H = _arr(xp, H)
    hm1 = H - 1.0
    fac = 3.15 + 1.72 / hm1 + H
    return th * fac, fac, th * (1.0 - 1.72 / (hm1 * hm1))
