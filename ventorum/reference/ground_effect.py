# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Induced drag of an elliptic loading near flat ground (Trefftz plane with an image wake).

This is an independent reference for the ground-effect solvers. It does not
use the production vortex kernels, the lattice or the Trefftz-plane code.

Theory
------
A wing of span ``b`` [m] carries the elliptic circulation
``Gamma(y) = Gamma_0 sqrt(1 - (y/a)^2)`` with ``a = b/2`` [m]. Its wake is a
flat vortex sheet at the height ``h`` [m] above the ground, parallel to the
ground and to the free stream. The ground is replaced by the mirror image of
the sheet, with the circulation ``-Gamma``, at the distance ``H = 2 h`` [m]
below the sheet (method of images; C. Wieselsberger 1921).

In the Trefftz plane, with ``z = y + i z_v`` [m], the elliptic sheet alone has
the complex velocity

    W(z) = u - i v = i w_0 (1 - z / sqrt(z^2 - a^2)),    w_0 = Gamma_0 / b,

with the branch of ``sqrt(z^2 - a^2)`` that is ``z`` at infinity and has its
cut on the sheet. This is the flow of a flat plate of half-width ``a`` that
moves normal to itself. Three properties identify it as the flow of the
sheet: ``W`` goes to zero at infinity; on the sheet ``v = -w_0`` (the
uniform downwash of the elliptic loading); and the potential jump across the
sheet is ``2 w_0 sqrt(a^2 - y^2) = Gamma(y)``.

The image sheet has the opposite circulation, thus its velocity at the real
sheet is ``-W(zeta)`` with ``zeta = y + i H``. The vertical velocity of the
image on the real sheet is ``v_img = w_0 (1 - Re[zeta / sqrt(zeta^2 - a^2)])``
(an upwash). The Trefftz-plane induced drag is
``D_i = (rho / 2) int Gamma(y) w(y) dy`` with the downwash ``w = w_0 - v_img``.
At the same circulation (the same lift), the ratio of the induced drag in
ground effect to the induced drag in free air is

    R(h/b) = D_i,GE / D_i,free = k_GE / k_free
           = (2 / (pi a^2)) int_{-a}^{a} sqrt(a^2 - y^2) Re[zeta / sqrt(zeta^2 - a^2)] dy,

where ``k = C_Di / C_L^2``. With ``y = a cos(theta)`` the integrand is a
smooth periodic function of ``theta`` for ``h > 0``, and the midpoint rule in
``theta`` converges exponentially. ``R`` is
``1 - sigma`` with the influence coefficient ``sigma`` of Wieselsberger
(equation 3 of NACA TM 77).

A second, independent form follows from the Fourier transform of the
loading. The transform of ``sqrt(a^2 - y^2)`` is ``pi a J_1(k a) / |k|``, and
the interaction kernel of two parallel sheets at the distance ``H`` has the
transform ``pi |k| exp(-H |k|)``. With ``beta = H / a = 4 h / b``:

    R = 1 - 2 int_0^inf J_1(x)^2 exp(-beta x) / x dx.

The free-air limit is ``R -> 1`` for ``h -> inf``. Far from the ground the
sheet acts on its image as a vortex pair: ``1 - z / sqrt(z^2 - a^2)`` is
``-a^2 / (2 z^2)`` at large ``|z|``, thus ``1 - R ~ a^2 / (2 H^2) = (b/h)^2 / 32``.
For small ``beta``, ``R ~ (2 beta / pi) (ln(8 / beta) - 1)``. The two
asymptotes were derived for this module and are checked against the
quadrature in the tests.

The reference holds for an elliptic loading. A real wing near the ground
changes its loading, so the reference is valid only at heights where the
loading stays near elliptic.
"""

from __future__ import annotations

import numpy as np

REFERENCE = (
    "C. Wieselsberger, 'Ueber den Fluegelwiderstand in der Naehe des Bodens', Zeitschrift fuer "
    "Flugtechnik und Motorluftschiffahrt 12(10), 1921; English translation 'Wing resistance near the "
    "ground', NACA TM 77, 1922 (image wing, elliptic loading, influence coefficient sigma and "
    "Prandtl's approximation, equations 3 and 4). Exact Trefftz-plane evaluation derived here."
)


def _check_height(h_over_b: float) -> float:
    h = float(h_over_b)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h_over_b must be positive and finite.")
    return h


def elliptic_ground_drag_ratio(h_over_b: float, n_points: int | None = None) -> float:
    """Induced-drag ratio of an elliptic loading at a height above flat ground.

    The ratio is ``k_GE / k_free`` with ``k = C_Di / C_L^2``: the induced drag
    in ground effect divided by the induced drag in free air, at the same
    circulation. The evaluation uses the exact Trefftz-plane velocity of the
    image sheet and the midpoint rule in the angle ``theta``
    (``y = (b/2) cos(theta)``). See the module docstring for the derivation.

    Parameters
    ----------
    h_over_b : float
        Height of the wake sheet above the ground divided by the span [-].
        Must be positive.
    n_points : int or None
        Number of quadrature points [-]. None selects a number that gives
        round-off accuracy: the integrand has a peak of width about
        ``sqrt(4 h / b)`` [rad] near the tips.

    Returns
    -------
    float
        ``k_GE / k_free`` [-], between 0 and 1.
    """
    h = _check_height(h_over_b)
    beta = 4.0 * h
    if n_points is None:
        n_points = int(min(200 + 60.0 / np.sqrt(beta), 200_000))
    n = int(n_points)
    if n < 2:
        raise ValueError("n_points must be at least 2.")
    a = 1.0
    theta = (np.arange(n) + 0.5) * np.pi / n
    zeta = a * np.cos(theta) + 1j * (2.0 * h * 2.0 * a)  # H = 2 h, with b = 2 a
    ratio = zeta / (np.sqrt(zeta - a) * np.sqrt(zeta + a))
    # (2 / (pi a^2)) int a sin(t) Re(.) a sin(t) dt, midpoint rule with dt = pi / n.
    return float(2.0 / n * np.sum(np.sin(theta) ** 2 * ratio.real))


def elliptic_ground_drag_ratio_spectral(h_over_b: float) -> float:
    """Induced-drag ratio of an elliptic loading from the Fourier (Bessel) form.

    ``R = 1 - 2 int_0^inf J_1(x)^2 exp(-beta x) / x dx`` with
    ``beta = 4 h / b``. This is a second evaluation, independent of
    :func:`elliptic_ground_drag_ratio`, for the tests. It uses adaptive
    quadrature (``scipy.integrate.quad``) on intervals of length ``pi``.

    Parameters
    ----------
    h_over_b : float
        Height of the wake sheet above the ground divided by the span [-].
        Must be positive. Small values need many intervals and are slow.

    Returns
    -------
    float
        ``k_GE / k_free`` [-].
    """
    from scipy import integrate, special

    beta = 4.0 * _check_height(h_over_b)
    x_end = 40.0 / beta + 50.0  # exp(-beta x) < 5e-18 after x_end, and J_1^2 / x < 1 / x^2

    def f(x: float) -> float:
        return special.j1(x) ** 2 * np.exp(-beta * x) / x if x > 0.0 else 0.0

    edges = np.arange(0.0, x_end + np.pi, np.pi)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        total += integrate.quad(f, lo, hi, epsabs=1e-16, epsrel=1e-13)[0]
    return 1.0 - 2.0 * total


def elliptic_ground_drag_ratio_asymptote(h_over_b: float) -> float:
    """Small-height asymptote of the induced-drag ratio of an elliptic loading.

    ``R ~ (2 beta / pi) (ln(8 / beta) - 1)`` with ``beta = 4 h / b``. The
    remainder is of higher order than ``beta`` (the tests check it).

    Parameters
    ----------
    h_over_b : float
        Height of the wake sheet above the ground divided by the span [-].

    Returns
    -------
    float
        Asymptotic ``k_GE / k_free`` [-].
    """
    beta = 4.0 * _check_height(h_over_b)
    return 2.0 * beta / np.pi * (np.log(8.0 / beta) - 1.0)


def wieselsberger_prandtl_drag_ratio(h_over_b: float) -> float:
    """Prandtl's approximation of the ground-effect drag ratio, as given by Wieselsberger.

    Wieselsberger (NACA TM 77, equation 4) gives the influence coefficient
    ``sigma = (1 - 0.66 g/b) / (1.05 + 3.7 g/b)``, where ``g = 2 h`` is the
    distance between the wing and its image. He states that it holds for
    ``g/b`` from 1/15 to 1/2, thus ``h/b`` from 1/30 to 1/4. The induced-drag
    ratio is ``1 - sigma`` (his equation 3 divided by the free-air induced
    drag ``A^2 / (pi q b^2)``).

    Parameters
    ----------
    h_over_b : float
        Height above the ground divided by the span [-].

    Returns
    -------
    float
        Approximate ``k_GE / k_free`` [-].
    """
    g = 2.0 * _check_height(h_over_b)
    return 1.0 - (1.0 - 0.66 * g) / (1.05 + 3.7 * g)
