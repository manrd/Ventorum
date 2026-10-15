# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Envelope e^N transition model for Ventorum-2D.

The model gives the critical momentum-thickness Reynolds number, the
envelope amplification rate per unit Reynolds number, and the resulting
growth rate of the amplification exponent along the surface, in
incompressible flow only. A later viscous-inviscid interaction solver
integrates the growth rate and trips the boundary layer where the exponent
reaches its critical value.

Each relation is a pure function of arrays. It returns the value and its
partial derivatives with respect to every input, as a tuple, with analytic
derivatives for Newton solves. Each function takes an optional ``xp``
argument (default numpy); with ``xp=torch`` the same code runs on torch
tensors on the CPU or on CUDA. There are no Python loops over stations.

Sources (papers only; no program code was read or used):

* M. Drela and M. B. Giles, "Viscous-inviscid analysis of transonic and low
  Reynolds number airfoils", AIAA Journal 25(10), 1987, pp. 1347-1355: the
  e^N transition model coupled to the integral boundary layer.
* M. Drela, doctoral thesis, MIT Gas Turbine Laboratory Report 187, 1986:
  the critical Reynolds number and the envelope amplification rate.

The equation labels below (E1) to (E3) point to
``docs/theory/ibl2d_closure.md``, which states every relation, its source
and its range of validity. The fit coefficients are Ventorum fits; their
values live in the code below.
"""

from __future__ import annotations


def _mod(xp):
    """Return the array module (numpy by default)."""
    if xp is None:
        import numpy as np

        return np
    return xp


def _f(xp, x):
    """Return ``x`` as a float array of the ``xp`` module."""
    return xp.asarray(x) * 1.0


def critical_re_theta(H, xp=None):
    """Return the critical Reynolds number and its derivative.

    Critical momentum-thickness Reynolds number ``Re_theta,0`` (theory page
    Eq. (E1)), below which Tollmien-Schlichting waves do not grow. A
    favourable pressure gradient (low ``H``) stabilizes the layer, so
    ``Re_theta,0`` rises as ``H`` falls; it approaches a floor in strongly
    adverse gradients.

    Parameters
    ----------
    H : array_like
        Kinematic shape factor ``delta_star / theta`` (dimensionless).
    xp : module or None
        Array module (default numpy); pass torch for torch tensors.

    Returns
    -------
    tuple
        ``(Re0, dRe0_dH)``, both dimensionless.
    """
    m = _mod(xp)
    H = _f(m, H)
    e = m.exp(6.91 - H)
    return 163.0 + e, -e


def amplification_rate(H, xp=None):
    """Return the envelope amplification rate and its derivative.

    Rate ``g = dN / dRe_theta`` (dimensionless) of the envelope of the
    spatial amplification curves, theory page Eq. (E2). ``g`` is positive:
    past the critical Reynolds number every Reynolds-number increment adds
    to the exponent. Growth is faster in adverse gradients (high ``H``).

    Parameters
    ----------
    H : array_like
        Kinematic shape factor ``delta_star / theta`` (dimensionless).
    xp : module or None
        Array module (default numpy); pass torch for torch tensors.

    Returns
    -------
    tuple
        ``(g, dg_dH)``, both dimensionless.
    """
    m = _mod(xp)
    H = _f(m, H)
    val = 0.01 + 0.008 * (H - 1.0)
    der = m.full_like(val, 0.008)
    return val, der


def transition_growth_rate(H, Re_theta, dRe_theta_ds, xp=None):
    """Return the growth rate of the exponent and its derivatives.

    Rate ``dN/ds`` [1/m] (theory page Eq. (E3)): ``g(H) * dRe_theta/ds``
    where ``Re_theta`` exceeds its critical value, else zero. The switch at
    the critical Reynolds number is the source relation itself, so the rate
    has a jump there by design.

    Parameters
    ----------
    H : array_like
        Kinematic shape factor ``delta_star / theta`` (dimensionless).
    Re_theta : array_like
        Momentum-thickness Reynolds number (dimensionless).
    dRe_theta_ds : array_like
        Streamwise derivative of ``Re_theta`` [1/m].
    xp : module or None
        Array module (default numpy); pass torch for torch tensors.

    Returns
    -------
    tuple
        ``(dNds, d_dH, d_dRe, d_ddReds)`` with ``dNds`` [1/m], ``d_dH``
        [1/m], ``d_dRe`` [1/m] and ``d_ddReds`` dimensionless.
    """
    m = _mod(xp)
    H = _f(m, H)
    Re = _f(m, Re_theta)
    r = _f(m, dRe_theta_ds)
    Re0, _ = critical_re_theta(H, xp)
    g, dg_H = amplification_rate(H, xp)
    on = Re >= Re0
    val = m.where(on, g * r, 0.0)
    dH = m.where(on, dg_H * r, 0.0)
    dRe = m.zeros_like(val)
    dr = m.where(on, g, 0.0)
    return val, dH, dRe, dr
