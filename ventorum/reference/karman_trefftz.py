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
    *args,
    alpha_deg: float | None = None,
    **kwargs,
) -> tuple[np.ndarray, float]:
    """Calculate exact surface Cp and lift coefficient from conformal mapping.

    Parameters
    ----------
    center : tuple of float or complex
        Circle center (mu_x, mu_y).
    trailing_edge_angle_deg : float
        Interior trailing-edge angle [deg].
    *args
        Positional arguments supporting (n_points, alpha_deg) or (alpha_deg, n_points).
    alpha_deg : float, optional
        Angle of attack [deg].
    **kwargs
        Keyword arguments, including n_points or alpha_deg.

    Returns
    -------
    tuple of (np.ndarray, float)
        Surface pressure coefficients Cp at the nodes, and exact lift
        coefficient Cl.
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

    # Parse args for n_points and alpha_deg
    n_points = kwargs.get("n_points", None)
    if alpha_deg is None:
        alpha_deg = kwargs.get("alpha_deg", None)

    for a in args:
        if isinstance(a, (int, np.integer)):
            if n_points is None:
                n_points = int(a)
            elif alpha_deg is None:
                alpha_deg = float(a)
        elif isinstance(a, (float, np.floating)):
            if alpha_deg is None:
                alpha_deg = float(a)
            elif n_points is None:
                n_points = int(a)

    if n_points is None:
        n_points = 201
    if alpha_deg is None:
        alpha_deg = 0.0

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

    x_le = float(np.min(z.real))
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
