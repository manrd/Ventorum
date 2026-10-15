# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the Karman-Trefftz exact conformal mapping reference solutions.

This module tests the generation of Karman-Trefftz airfoils and their exact
potential flow solutions from conformal mapping (Milne-Thomson 1958, Katz and
Plotkin 2001).
"""

from __future__ import annotations

import numpy as np

from ventorum.reference.karman_trefftz import (
    karman_trefftz_airfoil,
    karman_trefftz_solution,
)


def _polygon_signed_area(xy: np.ndarray) -> float:
    """Compute the signed area of a 2D polygon [m^2]."""
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def test_karman_trefftz_airfoil_geometry():
    """Verify coordinate conventions and properties of the generated airfoil."""
    center = (-0.07, 0.04)
    tau_deg = 10.0
    n_points = 201

    coords = karman_trefftz_airfoil(center, tau_deg, n_points)

    assert isinstance(coords, np.ndarray)
    assert coords.shape == (n_points, 2)
    assert coords.dtype == np.float64

    # Selig order check: counterclockwise orientation (positive signed area)
    area = _polygon_signed_area(coords)
    assert area > 0.0, "Airfoil coordinates must follow counterclockwise Selig order"

    # Unit chord check: leading edge at x=0, trailing edge at x=1
    x_min = float(np.min(coords[:, 0]))
    x_max = float(np.max(coords[:, 0]))
    assert abs(x_min) < 1e-12, f"Leading edge x should be 0, got {x_min}"
    assert abs(x_max - 1.0) < 1e-12, f"Trailing edge x should be 1, got {x_max}"

    # Trailing edge closed check
    assert np.allclose(coords[0], coords[-1], atol=1e-12)

    # Thickness check: about 12% thick
    thickness = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))
    assert 0.10 <= thickness <= 0.14, f"Thickness should be about 12%, got {thickness:.3f}"


def test_symmetric_joukowski_zero_alpha():
    """Verify symmetric Joukowski airfoil at zero angle of attack gives zero lift."""
    center = (-0.1, 0.0)
    tau_deg = 0.0
    n_points = 201

    cp, cl = karman_trefftz_solution(center, tau_deg, n_points, alpha_deg=0.0)

    # Exact lift must be zero by symmetry
    assert abs(cl) < 1e-12, f"Symmetric Joukowski at alpha 0 must have Cl=0, got {cl}"

    # Cp must be symmetric between upper and lower surfaces
    # Node 0 is TE, node n_points//2 is LE, node -1 is TE
    n_mid = n_points // 2
    upper_cp = cp[: n_mid + 1]
    lower_cp = cp[n_mid:][::-1]
    assert np.allclose(upper_cp, lower_cp, atol=1e-6)

    # Stagnation point at leading edge has Cp = 1.0
    assert abs(cp[n_mid] - 1.0) < 1e-4


def test_karman_trefftz_solution_stagnation():
    """Verify stagnation points and pressure coefficient limits."""
    center = (-0.07, 0.04)
    tau_deg = 10.0
    n_points = 201
    alpha_deg = 4.0

    cp, cl = karman_trefftz_solution(center, tau_deg, n_points, alpha_deg=alpha_deg)

    # A finite trailing edge angle (tau > 0) has a stagnation point at the trailing edge (Cp = 1)
    assert abs(cp[0] - 1.0) < 1e-8
    assert abs(cp[-1] - 1.0) < 1e-8

    # Maximum Cp on the body is the stagnation point (Cp = 1)
    assert abs(float(np.max(cp)) - 1.0) < 1e-8

    # Lift coefficient must be positive for positive alpha and positive camber
    assert cl > 0.5, f"Expected positive lift, got {cl}"


def test_karman_trefftz_lift_slope():
    """Verify that lift scales with sin(alpha - theta_te)."""
    center = (-0.07, 0.04)
    tau_deg = 10.0
    n_points = 151

    alphas = [0.0, 4.0, 8.0]
    cls = []
    for a in alphas:
        _, cl = karman_trefftz_solution(center, tau_deg, n_points, alpha_deg=a)
        cls.append(cl)

    # Check that lift increases monotonically with angle of attack
    assert cls[0] < cls[1] < cls[2]
    # Check approximate linearity of Cl with alpha for small angles
    dcl_da1 = (cls[1] - cls[0]) / 4.0
    dcl_da2 = (cls[2] - cls[1]) / 4.0
    assert abs(dcl_da1 - dcl_da2) < 0.01
