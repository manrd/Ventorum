# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Two-equation integral boundary-layer closures for Ventorum-2D.

The closures give the shape factors, the skin friction, the dissipation and
the shear-stress relations of the laminar and turbulent boundary layers, in
incompressible flow only. A later viscous-inviscid interaction solver uses
them, together with the transition model of
:mod:`ventorum.ibl2d.transition`, in the momentum, kinetic-energy and
shear-lag equations.

Each closure is a pure function of arrays. It returns the value and its
partial derivatives with respect to every input, as a tuple, with analytic
derivatives for Newton solves. Each function takes an optional ``xp``
argument (default numpy); with ``xp=torch`` the same code runs on torch
tensors on the CPU or on CUDA. There are no Python loops over stations.

Sources (papers only; no program code was read or used):

* M. Drela and M. B. Giles, "Viscous-inviscid analysis of transonic and low
  Reynolds number airfoils", AIAA Journal 25(10), 1987, pp. 1347-1355: the
  two-equation integral formulation with the lag equation and the e^N
  transition model.
* M. Drela, doctoral thesis, MIT Gas Turbine Laboratory Report 187, 1986:
  the closure relations fitted to the Falkner-Skan profile family.
* T. J. Swafford, "Analytical approximation of two-dimensional separated
  turbulent boundary-layer velocity profiles", AIAA Journal 21(6), 1983:
  the turbulent skin-friction fit.
* P. D. Green, D. G. Weeks and J. W. F. Brooman, "Prediction of turbulent
  boundary layers and wakes in compressible flow by a lag-entrainment
  method", ARC R and M 3791, 1977: the lag-entrainment idea behind the
  shear-stress rate equation.

The equation labels below (L1) to (T5) and (G1) point to
``docs/theory/ibl2d_closure.md``, which states every relation, its source
and its range of validity. The fit coefficients are Ventorum fits to the
profile families above; their values live in the code below.
"""

from __future__ import annotations

_LN10 = 2.302585092994046


def _mod(xp):
    """Return the array module (numpy by default)."""
    if xp is None:
        import numpy as np

        return np
    return xp


def _f(xp, x):
    """Return ``x`` as a float array of the ``xp`` module."""
    return xp.asarray(x) * 1.0


def laminar_h_star(H, xp=None):
    """Return the laminar energy shape factor and its derivative.

    Energy shape factor ``Hstar = theta_star / theta`` (dimensionless) from
    the Falkner-Skan profile family, theory page Eq. (L1). The fit has two
    branches that meet at ``H = 4`` with the same value and slope.

    Parameters
    ----------
    H : array_like
        Kinematic shape factor ``delta_star / theta`` (dimensionless).
    xp : module or None
        Array module (default numpy); pass torch for torch tensors.

    Returns
    -------
    tuple
        ``(Hstar, dHstar_dH)``, both dimensionless.
    """
    m = _mod(xp)
    H = _f(m, H)
    attached = H <= 4.0
    back = 4.0 - H
    fwd = H - 4.0
    val = m.where(attached, 1.515 + 0.076 * back * back / H,
                  1.515 - 0.147 * fwd * fwd / H)
    der = m.where(attached, -0.076 * back * (H + 4.0) / (H * H),
                  -0.147 * fwd * (H + 4.0) / (H * H))
    return val, der


def laminar_cf_factor(H, xp=None):
    """Return the laminar skin-friction factor and its derivative.

    Factor ``F = Re_theta * Cf / 2`` (dimensionless) from the Falkner-Skan
    profile family, theory page Eq. (L2). ``F`` is zero near ``H = 4``
    (laminar separation) and negative above it.

    Parameters
    ----------
    H : array_like
        Kinematic shape factor ``delta_star / theta`` (dimensionless).
    xp : module or None
        Array module (default numpy); pass torch for torch tensors.

    Returns
    -------
    tuple
        ``(F, dF_dH)``, both dimensionless.
    """
    m = _mod(xp)
    H = _f(m, H)
    t = 7.4 - H
    den = H - 1.0
    val = -0.067 + 0.01977 * t * t / den
    der = -0.01977 * t * (H + 5.4) / (den * den)
    return val, der


def laminar_dissipation_factor(H, xp=None):
    """Return the laminar dissipation factor and its derivative.

    Factor ``G = Re_theta * 2 * CD / Hstar`` (dimensionless) from the
    Falkner-Skan profile family, theory page Eq. (L3). The fit has two
    branches that meet at ``H = 4`` with the same value and slope.

    Parameters
    ----------
    H : array_like
        Kinematic shape factor ``delta_star / theta`` (dimensionless).
    xp : module or None
        Array module (default numpy); pass torch for torch tensors.

    Returns
    -------
    tuple
        ``(G, dG_dH)``, both dimensionless.
    """
    m = _mod(xp)
    H = _f(m, H)
    attached = H <= 4.0
    dp = m.where(attached, 4.0 - H, 0.0)
    fwd = H - 4.0
    u = fwd * fwd
    den = 1.0 + 0.02 * u
    val = m.where(attached, 0.207 + 0.00205 * dp ** 5.5,
                  0.207 - 0.003 * u / den)
    der = m.where(attached, -0.011275 * dp ** 4.5,
                  -0.003 * fwd * (2.0 * den - 0.04 * u) / (den * den))
    return val, der


def turbulent_cf(H, Re_theta, xp=None):
    """Return the turbulent skin-friction coefficient and its derivatives.

    Coefficient ``Cf`` (dimensionless) from the Swafford profile fit,
    theory page Eq. (T1). The first term is the attached skin friction from
    the wall-wake profile family; the second term lowers ``Cf`` through
    separation and makes it negative in reversed flow.

    Parameters
    ----------
    H : array_like
        Kinematic shape factor ``delta_star / theta`` (dimensionless).
    Re_theta : array_like
        Momentum-thickness Reynolds number (dimensionless).
    xp : module or None
        Array module (default numpy); pass torch for torch tensors.

    Returns
    -------
    tuple
        ``(Cf, dCf_dH, dCf_dRe)``, all dimensionless.
    """
    m = _mod(xp)
    H = _f(m, H)
    Re = _f(m, Re_theta)
    L = m.log10(Re)
    A = 0.3 * m.exp(-1.33 * H) * L ** (-1.74 - 0.31 * H)
    t = m.tanh(4.0 - H)
    B = 1.1e-4 * (t - 0.5)
    val = A + B
    dH = A * (-1.33 - 0.31 * m.log(L)) - 1.1e-4 * (1.0 - t * t)
    p = -1.74 - 0.31 * H
    dRe = A * p / (L * Re * _LN10)
    return val, dH, dRe


def turbulent_h_star(H, Re_theta, xp=None):
    """Return the turbulent energy shape factor and its derivatives.

    Energy shape factor ``Hstar = theta_star / theta`` (dimensionless) from
    the turbulent wall-wake profile family, theory page Eq. (T2). ``Hstar``
    grows with ``H`` and falls slowly with ``Re_theta``.

    Parameters
    ----------
    H : array_like
        Kinematic shape factor ``delta_star / theta`` (dimensionless).
    Re_theta : array_like
        Momentum-thickness Reynolds number (dimensionless).
    xp : module or None
        Array module (default numpy); pass torch for torch tensors.

    Returns
    -------
    tuple
        ``(Hstar, dHstar_dH, dHstar_dRe)``, all dimensionless.
    """
    m = _mod(xp)
    H = _f(m, H)
    Re = _f(m, Re_theta)
    hm = H - 1.0
    val = 1.56 + 0.44 * hm + 0.20 * hm * hm - 0.06 * m.log10(Re / 1e4)
    dH = 0.44 + 0.40 * hm
    dRe = -0.06 / (Re * _LN10)
    return val, dH, dRe


def turbulent_slip_velocity(H, Re_theta, xp=None):
    """Return the turbulent slip velocity and its derivatives.

    Slip velocity ``Us = u_s / u_e`` (dimensionless) of the outer wake
    profile, theory page Eq. (T3). ``Us`` grows with the wake strength
    (with ``H``) and falls slowly with ``Re_theta``.

    Parameters
    ----------
    H : array_like
        Kinematic shape factor ``delta_star / theta`` (dimensionless).
    Re_theta : array_like
        Momentum-thickness Reynolds number (dimensionless).
    xp : module or None
        Array module (default numpy); pass torch for torch tensors.

    Returns
    -------
    tuple
        ``(Us, dUs_dH, dUs_dRe)``, all dimensionless.
    """
    m = _mod(xp)
    H = _f(m, H)
    Re = _f(m, Re_theta)
    hm = H - 1.0
    val = 1.0 + 0.55 * hm + 0.10 * hm * hm - 0.02 * m.log10(Re / 1e4)
    dH = 0.55 + 0.20 * hm
    dRe = -0.02 / (Re * _LN10)
    return val, dH, dRe


def equilibrium_shear_coefficient(H, H_star, U_s, xp=None):
    """Return the equilibrium shear-stress coefficient and its derivatives.

    Coefficient ``Ctau_eq = tau_max / (rho_e * u_e ** 2)`` (dimensionless)
    on the equilibrium locus, theory page Eq. (T4). At a zero-pressure
    gradient flat plate it equals half the local skin-friction coefficient
    (the maximum shear is the wall shear there). ``H`` sets the shape only;
    it enters through ``H_star`` in a coupled solve, so the direct partial
    with respect to ``H`` is zero.

    Parameters
    ----------
    H : array_like
        Kinematic shape factor ``delta_star / theta`` (dimensionless).
    H_star : array_like
        Energy shape factor ``theta_star / theta`` (dimensionless).
    U_s : array_like
        Slip velocity ``u_s / u_e`` (dimensionless).
    xp : module or None
        Array module (default numpy); pass torch for torch tensors.

    Returns
    -------
    tuple
        ``(Ctau_eq, d_dH, d_dHstar, d_dUs)``, all dimensionless.
    """
    m = _mod(xp)
    H = _f(m, H)
    Hs = _f(m, H_star)
    Us = _f(m, U_s)
    q = (Hs - 1.0) / (Hs * Us)
    val = 0.0095 * q * q
    dH = m.zeros_like(val + 0.0 * H)
    dHs = 2.0 * 0.0095 * q / (Hs * Hs * Us)
    dUs = -2.0 * 0.0095 * q * q / Us
    return val, dH, dHs, dUs


def turbulent_dissipation_coefficient(H, Re_theta, U_s, C_tau, xp=None):
    """Return the turbulent dissipation coefficient and its derivatives.

    Coefficient ``CD`` (dimensionless) from the shear work of the profile,
    theory page Eq. (T5). At equilibrium (``C_tau`` equal to its equilibrium
    value) the dissipation is the wall shear work ``(Cf / 2) * Us``; off
    equilibrium it rises linearly with the shear excess.

    Parameters
    ----------
    H : array_like
        Kinematic shape factor ``delta_star / theta`` (dimensionless).
    Re_theta : array_like
        Momentum-thickness Reynolds number (dimensionless).
    U_s : array_like
        Slip velocity ``u_s / u_e`` (dimensionless).
    C_tau : array_like
        Maximum shear-stress coefficient (dimensionless).
    xp : module or None
        Array module (default numpy); pass torch for torch tensors.

    Returns
    -------
    tuple
        ``(CD, dCD_dH, dCD_dRe, dCD_dUs, dCD_dCtau)``, all dimensionless.
    """
    m = _mod(xp)
    H = _f(m, H)
    Re = _f(m, Re_theta)
    Us = _f(m, U_s)
    Ct = _f(m, C_tau)
    Cf, dCf_H, dCf_Re = turbulent_cf(H, Re, xp)
    Hs, dHs_H, dHs_Re = turbulent_h_star(H, Re, xp)
    Te, _, dTe_Hs, dTe_Us = equilibrium_shear_coefficient(H, Hs, Us, xp)
    F2 = 0.5 * Cf
    val = F2 * Us + 0.5 * (Ct - Te)
    dH = 0.5 * dCf_H * Us - 0.5 * dTe_Hs * dHs_H
    dRe = 0.5 * dCf_Re * Us - 0.5 * dTe_Hs * dHs_Re
    dUs = F2 - 0.5 * dTe_Us
    dCt = m.full_like(val, 0.5)
    return val, dH, dRe, dUs, dCt


def shear_lag_coefficient(C_tau, xp=None):
    """Return the shear-lag rate coefficient and its derivative.

    Dimensionless coefficient ``K`` of the shear-stress rate equation
    ``delta_star * dC_tau/ds = K * (sqrt(Ctau_eq) - sqrt(C_tau))`` with
    ``delta_star`` [m] and ``s`` [m], theory page Eq. (G1). The rate scales
    with the turbulence level ``sqrt(C_tau)`` (the lag-entrainment idea of
    Green, Weeks and Brooman); ``Kc = 5.6`` is the lag constant.

    Parameters
    ----------
    C_tau : array_like
        Maximum shear-stress coefficient (dimensionless).
    xp : module or None
        Array module (default numpy); pass torch for torch tensors.

    Returns
    -------
    tuple
        ``(K, dK_dCtau)``, both dimensionless.
    """
    m = _mod(xp)
    Ct = _f(m, C_tau)
    s = m.sqrt(Ct)
    val = 5.6 * s
    der = 2.8 / s
    return val, der
