# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Acceptance tests for the 2D viscous panel solver with streamline wake.

This module tests:
1. Wake line tracing along the inviscid streamline for symmetric and lifting sections.
2. Drag consistency: Squire-Young profile drag and nearfield (friction plus pressure)
   drag agreement within 5 percent for NACA 0012 at Re 1e6 and 3e6 across alpha 0 to 6 deg.
3. Blunt trailing-edge effect: convergence and increased drag with a 0.5 percent gap.
4. Plain-flapped sections: convergence at 0, 5, 10 deg and clean return or separation
   reporting at 20 deg for NACA 2412 with flap hinge at 0.75 chord.
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.panel2d.viscous import solve_viscous, trace_wake_streamline


def make_naca0012(n_points: int = 120, te_gap: float = 0.0) -> np.ndarray:
    """Generate NACA 0012 coordinates in Selig order with optional trailing-edge gap.

    Parameters
    ----------
    n_points : int, optional
        Number of points on the contour. Default is 120.
    te_gap : float, optional
        Trailing-edge thickness as fraction of chord. Default is 0.0.

    Returns
    -------
    np.ndarray
        Array of contour points of shape (N, 2) in Selig order.
    """
    theta = np.linspace(0.0, np.pi, n_points // 2 + 1)
    x = 0.5 * (1.0 - np.cos(theta))
    # Standard NACA thickness distribution with adjusted trailing edge for gap
    yt = 5.0 * 0.12 * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - (0.1036 - te_gap / 2.0) * x**4
    )
    xu = x[::-1]
    yu = yt[::-1]
    xl = x[1:]
    yl = -yt[1:]
    return np.column_stack([np.concatenate([xu, xl]), np.concatenate([yu, yl])])


def make_naca2412_flapped(
    n_points: int = 120,
    flap_hinge: float = 0.75,
    flap_deg: float = 0.0,
) -> np.ndarray:
    """Generate NACA 2412 coordinates with a plain deflected flap.

    Parameters
    ----------
    n_points : int, optional
        Number of points on the contour. Default is 120.
    flap_hinge : float, optional
        Chordwise position of the flap hinge. Default is 0.75.
    flap_deg : float, optional
        Flap deflection angle [deg], positive trailing edge down. Default is 0.0.

    Returns
    -------
    np.ndarray
        Array of contour points of shape (N, 2) in Selig order.
    """
    theta = np.linspace(0.0, np.pi, n_points // 2 + 1)
    x = 0.5 * (1.0 - np.cos(theta))
    yt = 5.0 * 0.12 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    m = 0.02
    p = 0.4
    yc = np.where(x < p, m / (p**2) * (2.0 * p * x - x**2), m / ((1.0 - p)**2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2))
    dyc = np.where(x < p, 2.0 * m / (p**2) * (p - x), 2.0 * m / ((1.0 - p)**2) * (p - x))
    th_c = np.arctan(dyc)

    xu = x - yt * np.sin(th_c)
    yu = yc + yt * np.cos(th_c)
    xl = x + yt * np.sin(th_c)
    yl = yc - yt * np.cos(th_c)

    coords = np.column_stack([np.concatenate([xu[::-1], xl[1:]]), np.concatenate([yu[::-1], yl[1:]])])

    if abs(flap_deg) > 1e-6:
        yc_h = float(
            m / ((1.0 - p)**2) * ((1.0 - 2.0 * p) + 2.0 * p * flap_hinge - flap_hinge**2)
            if flap_hinge >= p
            else m / (p**2) * (2.0 * p * flap_hinge - flap_hinge**2)
        )
        delta = np.radians(flap_deg)
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        mask = coords[:, 0] >= flap_hinge
        dx = coords[mask, 0] - flap_hinge
        dy = coords[mask, 1] - yc_h
        coords[mask, 0] = flap_hinge + dx * cos_d + dy * sin_d
        coords[mask, 1] = yc_h - dx * sin_d + dy * cos_d

    return coords


def test_wake_streamline_symmetry_alpha0():
    """Verify that the wake line of a symmetric section at alpha 0 is straight on the axis."""
    coords = make_naca0012(120)
    wake_pts = trace_wake_streamline(coords, alpha_deg=0.0, wake_length=1.0)

    # Wake starts at the trailing-edge midpoint
    te_mid = 0.5 * (coords[0] + coords[-1])
    np.testing.assert_allclose(wake_pts[0], te_mid, atol=1e-12)

    # Straight on the axis: y coordinate within 1e-10
    max_y = np.max(np.abs(wake_pts[:, 1]))
    assert max_y <= 1e-10, f"Wake y coordinate {max_y:.2e} exceeds 1e-10 tolerance"


def test_wake_streamline_bending_alpha6():
    """Verify that the wake line at alpha 6 bends toward the free-stream direction."""
    coords = make_naca0012(120)
    wake_pts = trace_wake_streamline(coords, alpha_deg=6.0, wake_length=1.0)

    # End point direction within 2 deg of free stream (6 deg)
    end_dir = wake_pts[-1] - wake_pts[-2]
    end_angle_deg = np.degrees(np.arctan2(end_dir[1], end_dir[0]))
    angle_diff = abs(end_angle_deg - 6.0)
    assert angle_diff <= 2.0, f"Wake end angle {end_angle_deg:.2f} deg differs by {angle_diff:.2f} deg > 2 deg"


def test_drag_consistency_naca0012_re1e6_and_3e6():
    """Verify agreement within 5 percent between Squire-Young and nearfield drag."""
    coords = make_naca0012(120)
    alphas = [0.0, 2.0, 4.0, 6.0]

    for re in [1e6, 3e6]:
        for alpha in alphas:
            res = solve_viscous(coords, alpha, re, tol=1e-4, max_iter=45)
            assert res.converged, f"Did not converge at Re={re:.1e}, alpha={alpha}"

            cd_sy = float(res.Cd)
            cd_near = float(res.Cd_friction + res.Cd_pressure)
            rel_diff = abs(cd_sy - cd_near) / cd_sy
            assert rel_diff <= 0.05, (
                f"Drag discrepancy at Re={re:.1e}, alpha={alpha}: "
                f"Cd_SY={cd_sy:.5f}, Cd_near={cd_near:.5f}, diff={rel_diff * 100:.2f}% > 5%"
            )


def test_blunt_trailing_edge_gap():
    """Verify that a section with a 0.5 percent gap converges and has larger Cd."""
    coords_closed = make_naca0012(120, te_gap=0.0)
    coords_blunt = make_naca0012(120, te_gap=0.005)

    res_closed = solve_viscous(coords_closed, 0.0, 1e6, tol=1e-4)
    res_blunt = solve_viscous(coords_blunt, 0.0, 1e6, tol=1e-4)

    assert res_blunt.converged, "Blunt trailing-edge section did not converge"
    assert res_blunt.Cd > res_closed.Cd, (
        f"Blunt Cd ({res_blunt.Cd:.5f}) is not larger than closed Cd ({res_closed.Cd:.5f})"
    )


def test_plain_flap_convergence_and_separation():
    """Verify NACA 2412 plain flap convergence at 0, 5, 10 deg and clean return at 20 deg."""
    # 0, 5, 10 deg must converge
    for flap_deg in [0.0, 5.0, 10.0]:
        coords = make_naca2412_flapped(120, flap_hinge=0.75, flap_deg=flap_deg)
        res = solve_viscous(coords, 0.0, 1e6, tol=1e-4, max_iter=45)
        assert res.converged, f"Flap deflection {flap_deg} deg failed to converge"
        assert np.isfinite(res.Cl)
        assert np.isfinite(res.Cd)

    # 20 deg must either converge or report separation with a reason without exception
    coords_20 = make_naca2412_flapped(120, flap_hinge=0.75, flap_deg=20.0)
    try:
        res_20 = solve_viscous(coords_20, 0.0, 1e6, tol=1e-4, max_iter=45)
    except Exception as exc:
        pytest.fail(f"solve_viscous raised an exception at 20 deg flap: {exc}")

    # Must return a valid result
    assert isinstance(res_20.Cl, (float, np.floating))
    assert isinstance(res_20.Cd, (float, np.floating))
    if not res_20.converged:
        # Must report separation with a reason
        is_separated = any(res_20.separation_flags) if isinstance(res_20.separation_flags, (list, tuple)) else bool(res_20.separation_flags)
        assert is_separated or hasattr(res_20, "reason") and len(res_20.reason) > 0
