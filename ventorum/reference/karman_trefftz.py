# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Karman-Trefftz and Joukowski conformal mapping reference solutions.

This module provides exact coordinates and analytical potential flow solutions
for Karman-Trefftz and Joukowski airfoils using conformal mapping.

References
----------
L. M. Milne-Thomson, 'Theoretical Aerodynamics', 4th ed., Macmillan, 1958.
J. Katz and A. Plotkin, 'Low-Speed Aerodynamics', 2nd ed., Cambridge University
Press, 2001, section 6.6.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

#: Number of points on the circle that bracket the leading edge before the
#: 1D minimisation in :func:`_leading_edge_x`. It is fixed, so the chord does
#: not depend on the number of contour points of the caller.
_LE_BRACKET_POINTS = 2001


def _leading_edge_x(zc: complex, radius: float, theta_te: float, k: float, b: float = 1.0) -> float:
    """Return the x coordinate of the leading edge of the mapped section.

    The leading edge is the point of the section contour with the smallest
    x. A grid of ``_LE_BRACKET_POINTS`` points on the circle gives the
    discrete minimum. A bounded 1D minimisation over the circle angle
    between the two neighbours of that point then gives the minimum of the
    smooth contour. The result does not depend on the contour points of the
    caller.

    Parameters
    ----------
    zc : complex
        Circle centre in the circle plane.
    radius : float
        Circle radius.
    theta_te : float
        Circle angle of the trailing-edge point zeta = b [rad].
    k : float
        Karman-Trefftz exponent, 2 - tau / pi.
    b : float, optional
        Position of the trailing-edge singular point (default 1.0).

    Returns
    -------
    float
        x coordinate of the leading edge, in the units of the mapped plane
        (before the unit-chord normalisation).
    """
    beta = np.linspace(0.0, 2.0 * np.pi, _LE_BRACKET_POINTS, endpoint=True)[1:-1]
    zeta = zc + radius * np.exp(1j * (theta_te + beta))
    lam = (zeta - b) / (zeta + b)
    phi = np.unwrap(np.angle(lam))
    s = (np.abs(lam) ** k) * np.exp(1j * k * phi)
    x = (k * b * (1.0 + s) / (1.0 - s)).real
    i = int(np.argmin(x))
    if i == 0 or i == x.size - 1:
        return float(x[i])
    lam_i, phi_i = lam[i], phi[i]

    def x_of(angle: float) -> float:
        zeta_a = zc + radius * np.exp(1j * (theta_te + angle))
        lam_a = (zeta_a - b) / (zeta_a + b)
        # Continue the phase from the bracket point, so that the branch of
        # lam**k is the branch of the contour.
        phi_a = phi_i + np.angle(lam_a / lam_i)
        s_a = (abs(lam_a) ** k) * np.exp(1j * k * phi_a)
        return float((k * b * (1.0 + s_a) / (1.0 - s_a)).real)

    res = minimize_scalar(x_of, bounds=(float(beta[i - 1]), float(beta[i + 1])), method="bounded",
                          options={"xatol": 1e-12})
    return float(min(res.fun, x[i]))


def karman_trefftz_airfoil(
    center: tuple[float, float] | complex,
    trailing_edge_angle_deg: float,
    n_points: int,
) -> np.ndarray:
    """Generate Karman-Trefftz airfoil coordinates in Selig order.

    Parameters
    ----------
    center : tuple of float or complex
        Circle center (mu_x, mu_y) in the circle plane. Negative mu_x gives
        thickness; positive mu_y gives positive camber.
    trailing_edge_angle_deg : float
        Interior trailing-edge angle [deg]. Zero corresponds to a cusped
        Joukowski trailing edge.
    n_points : int
        Number of discretization points along the airfoil surface contour.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, 2) in Selig order (counterclockwise,
        from upper trailing edge to leading edge, then along lower surface
        back to trailing edge). Normalized to unit chord [m].
    """
    if isinstance(center, complex):
        mux = center.real
        muy = center.imag
    else:
        mux, muy = float(center[0]), float(center[1])

    zc = complex(mux, muy)
    b = 1.0
    radius = abs(b - zc)
    theta_te = np.angle(b - zc)

    tau = np.radians(trailing_edge_angle_deg)
    k = 2.0 - tau / np.pi

    # Discretize circle in Selig order
    beta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=True)
    theta = theta_te + beta
    zeta = zc + radius * np.exp(1j * theta)
    zeta[0] = b
    zeta[-1] = b

    # Karman-Trefftz transformation
    # (z - k*b) / (z + k*b) = ((zeta - b) / (zeta + b))^k
    lam = (zeta - b) / (zeta + b)

    z = np.zeros(n_points, dtype=complex)
    z[0] = k * b
    z[-1] = k * b

    if n_points > 2:
        r_lam = np.abs(lam[1:-1])
        phi_lam = np.unwrap(np.angle(lam[1:-1]))
        s = (r_lam**k) * np.exp(1j * k * phi_lam)
        z[1:-1] = k * b * (1.0 + s) / (1.0 - s)

    # Unit chord normalization
    x_le = float(np.min(z.real))
    x_te = float(z[0].real)
    chord = x_te - x_le

    x = (z.real - x_le) / chord
    y = z.imag / chord

    # Force exact 0.0 leading edge and 1.0 trailing edge
    x[0] = 1.0
    x[-1] = 1.0
    y[0] = 0.0
    y[-1] = 0.0

    return np.column_stack([x, y]).astype(np.float64)


def karman_trefftz_solution(
    center: tuple[float, float] | complex,
    trailing_edge_angle_deg: float,
    n_points: int = 201,
    *,
    alpha_deg: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Calculate exact surface Cp and lift coefficient from conformal mapping.

    Parameters
    ----------
    center : tuple of float or complex
        Circle center (mu_x, mu_y).
    trailing_edge_angle_deg : float
        Interior trailing-edge angle [deg].
    n_points : int, optional
        Number of discretization points along the airfoil contour (default 201).
    alpha_deg : float, optional
        Angle of attack [deg] (default 0.0).

    Returns
    -------
    tuple of (np.ndarray, float)
        Surface pressure coefficients Cp at the nodes, and exact lift
        coefficient Cl.

    Notes
    -----
    Cl is normalised by the chord of the smooth section: the distance from
    the trailing edge to the leading edge, the point of the contour with the
    smallest x (see :func:`_leading_edge_x`). Thus Cl does not depend on
    *n_points*; only the Cp nodes do.
    """
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
        raise TypeError(f"n_points must be an integer, got {n_points!r}.")
    n_points = int(n_points)
    alpha_deg = float(alpha_deg)

    if isinstance(center, complex):
        mux = center.real
        muy = center.imag
    else:
        mux, muy = float(center[0]), float(center[1])

    zc = complex(mux, muy)
    b = 1.0
    radius = abs(b - zc)
    theta_te = np.angle(b - zc)

    alpha = np.radians(alpha_deg)
    tau = np.radians(trailing_edge_angle_deg)
    k = 2.0 - tau / np.pi
    v_inf = 1.0

    # Circulation from Kutta condition at zeta = b
    gamma = 4.0 * np.pi * radius * v_inf * np.sin(alpha - theta_te)

    beta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=True)
    theta = theta_te + beta
    zeta = zc + radius * np.exp(1j * theta)
    zeta[0] = b
    zeta[-1] = b

    lam = (zeta - b) / (zeta + b)

    z = np.zeros(n_points, dtype=complex)
    z[0] = k * b
    z[-1] = k * b

    if n_points > 2:
        r_lam = np.abs(lam[1:-1])
        phi_lam = np.unwrap(np.angle(lam[1:-1]))
        s = (r_lam**k) * np.exp(1j * k * phi_lam)
        z[1:-1] = k * b * (1.0 + s) / (1.0 - s)

    # The chord comes from the leading edge of the smooth contour, not from
    # the contour points, so Cl does not depend on n_points.
    x_le = _leading_edge_x(zc, radius, theta_te, k, b)
    x_te = float(z[0].real)
    chord = x_te - x_le

    # Lift coefficient
    cl = float(2.0 * gamma / (v_inf * chord))

    # Complex velocity in circle plane dW/dzeta
    dw_dzeta = -1j * np.exp(-1j * theta) * 2.0 * v_inf * (np.sin(theta - alpha) - np.sin(theta_te - alpha))

    # Derivative dz/dzeta
    dz_dzeta = np.zeros(n_points, dtype=complex)
    if n_points > 2:
        lam_mid = lam[1:-1]
        lam_km1 = (r_lam ** (k - 1.0)) * np.exp(1j * (k - 1.0) * phi_lam)
        dz_dzeta[1:-1] = (k**2) * lam_km1 * ((1.0 - lam_mid) / (1.0 - s)) ** 2

    # Physical surface speed V
    v_surface = np.zeros(n_points, dtype=np.float64)
    if n_points > 2:
        v_surface[1:-1] = np.abs(dw_dzeta[1:-1] / dz_dzeta[1:-1])

    if trailing_edge_angle_deg > 0.0:
        v_surface[0] = 0.0
        v_surface[-1] = 0.0
    else:
        # Joukowski trailing-edge limit. Both dW/dzeta and dz/dzeta are zero
        # at zeta = b. L'Hopital's rule with d2W/dzeta2 =
        # (2 V / R) exp(-2 i theta_te) cos(alpha - theta_te) and
        # d2z/dzeta2 = 2 / b gives |dW/dz| = V b |cos(alpha - theta_te)| / R.
        te_speed = v_inf * b * abs(np.cos(alpha - theta_te)) / radius
        v_surface[0] = te_speed
        v_surface[-1] = te_speed

    cp = 1.0 - (v_surface / v_inf) ** 2
    return cp, cl
