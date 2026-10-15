# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the NACA 4-digit and 5-digit airfoil generators."""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.airfoil.naca import naca, naca4, naca4_camber, naca4_thickness, naca5, naca5_camber


def _shoelace_area(points: np.ndarray) -> float:
    """Return the signed polygon area of a point loop."""
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _circle_radius(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Return the circumradius of three points."""
    a = float(np.linalg.norm(p2 - p3))
    b = float(np.linalg.norm(p1 - p3))
    c = float(np.linalg.norm(p1 - p2))
    area2 = abs(
        p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
    )
    assert area2 > 0.0
    return a * b * c / (2.0 * area2)


def test_thickness_maximum_of_0012_is_012():
    """Maximum of 2*y_t of NACA 0012 at x 0.30 is 0.12 within 1e-3."""
    x = np.array([0.30])
    yt = naca4_thickness(x, 0.12)
    assert abs(2.0 * float(yt[0]) - 0.12) < 1e-3


def test_open_te_thickness_of_0012():
    """Open trailing edge thickness of NACA 0012 is 0.00252 within 1e-5."""
    x = np.array([1.0])
    yt = naca4_thickness(x, 0.12, closed_te=False)
    assert abs(2.0 * float(yt[0]) - 0.00252) < 1e-5


def test_closed_te_thickness_is_zero():
    """Closed trailing edge thickness of NACA 0012 is zero."""
    x = np.array([1.0])
    yt = naca4_thickness(x, 0.12, closed_te=True)
    assert abs(float(yt[0])) < 1e-15


def test_leading_edge_radius_of_0012():
    """Circle fit near the leading edge matches 1.1019*t^2 within 2 percent."""
    t = 0.12
    points = naca4("0012", n_points=801, spacing="cosine")
    nose = int(np.argmin(points[:, 0]))
    p1 = points[nose - 1]
    p2 = points[nose]
    p3 = points[nose + 1]
    radius = _circle_radius(p1, p2, p3)
    expected = 1.1019 * t * t
    assert abs(radius - expected) / expected < 0.02


def test_2412_camber_maximum():
    """Mean line maximum of NACA 2412 is 0.02 at x 0.4 to 1e-12."""
    x = np.linspace(0.0, 1.0, 10001)
    yc, _ = naca4_camber(x, 0.02, 0.4)
    assert abs(float(np.max(yc)) - 0.02) < 1e-12
    assert abs(float(x[int(np.argmax(yc))]) - 0.4) < 1e-3


def test_2412_slope_continuous_at_p():
    """Mean line slope of NACA 2412 is continuous at x = p."""
    eps = 1e-7
    p = 0.4
    _, d_left = naca4_camber(np.array([p - eps]), 0.02, p)
    _, d_mid = naca4_camber(np.array([p]), 0.02, p)
    _, d_right = naca4_camber(np.array([p + eps]), 0.02, p)
    assert abs(float(d_mid[0])) < 1e-15
    assert abs(float(d_left[0]) - float(d_right[0])) < 1e-6


def test_23012_camber_maximum_position_and_table_value():
    """Mean line maximum of 23012 is near x 0.15 with the table value."""
    x = np.linspace(0.0, 1.0, 20001)
    yc, _ = naca5_camber(x, "23012")
    peak = float(x[int(np.argmax(yc))])
    assert abs(peak - 0.15) < 0.005
    m = 0.2025
    k1 = 15.957
    expected_at_m = k1 * m**3 * (1.0 - m) / 6.0
    yc_m, _ = naca5_camber(np.array([m]), "23012")
    assert abs(float(yc_m[0]) - expected_at_m) < 1e-12
    assert float(np.max(yc)) > expected_at_m


def test_symmetric_airfoils_are_symmetric():
    """Symmetric airfoils mirror about y = 0 to 1e-15."""
    for name in ("0012", "0009"):
        points = naca4(name, n_points=161)
        n = points.shape[0]
        half = (n + 1) // 2
        upper = points[:half][::-1]
        lower = points[half - 1 :]
        assert upper.shape == lower.shape
        assert np.max(np.abs(upper[:, 0] - lower[:, 0])) < 1e-15
        assert np.max(np.abs(upper[:, 1] + lower[:, 1])) < 1e-15


def test_selig_order_has_positive_area():
    """All generators return loops with positive shoelace area."""
    cases = [
        naca4("2412", n_points=161),
        naca4("0012", n_points=161),
        naca4("2412", n_points=161, closed_te=True),
        naca5("23012", n_points=161),
        naca5("23012", n_points=161, closed_te=True),
        naca4("2412", n_points=51, spacing="uniform"),
    ]
    for points in cases:
        assert _shoelace_area(points) > 0.0


def test_point_count_and_single_leading_edge():
    """Odd point counts give one leading edge station point at (0, 0)."""
    for n_points in (51, 161):
        for fun in (lambda n: naca4("2412", n_points=n), lambda n: naca5("23012", n_points=n)):
            points = fun(n_points)
            assert points.shape == (n_points, 2)
            nose = (n_points + 1) // 2 - 1
            assert abs(float(points[nose, 0])) < 1e-15
            assert abs(float(points[nose, 1])) < 1e-15
            d2 = (points[:, 0] - points[nose, 0]) ** 2 + (points[:, 1] - points[nose, 1]) ** 2
            assert int(np.sum(d2 == 0.0)) == 1


def test_even_point_count_raises():
    """Even point counts raise ValueError."""
    with pytest.raises(ValueError):
        naca4("2412", n_points=160)
    with pytest.raises(ValueError):
        naca5("23012", n_points=160)
    with pytest.raises(ValueError):
        naca("2412", n_points=100)


def test_designation_formats_give_same_points():
    """Every designation format of naca() gives the same points."""
    ref4 = naca("2412")
    for name in ("NACA 2412", "naca2412", "2412", 2412, "NACA2412"):
        assert np.array_equal(naca(name), ref4)
    assert np.array_equal(naca("2412"), naca4("2412"))
    ref5 = naca("23012")
    for name in ("NACA 23012", "naca23012", "23012", 23012):
        assert np.array_equal(naca(name), ref5)
    assert np.array_equal(naca("23012"), naca5("23012"))


def test_closed_te_repeats_end_point():
    """A closed trailing edge repeats the same point at start and end."""
    points = naca4("2412", n_points=161, closed_te=True)
    assert np.array_equal(points[0], points[-1])
    assert abs(float(points[0, 0]) - 1.0) < 1e-12
    assert abs(float(points[0, 1])) < 1e-12


def test_open_te_gap_matches_thickness():
    """The open trailing edge gap equals twice the half thickness."""
    points = naca4("0012", n_points=161, closed_te=False)
    gap = float(points[0, 1] - points[-1, 1])
    assert abs(gap - 0.00252) < 1e-5


def test_invalid_designations_raise():
    """Invalid designations raise ValueError."""
    bad = [
        "12",
        "123",
        "12345",
        "123456",
        "NACA 123",
        "2400",
        "23000",
        "1012",
        "ab12",
        "",
    ]
    for name in bad:
        with pytest.raises(ValueError):
            naca(name)
    with pytest.raises(ValueError):
        naca4("1012")
    with pytest.raises(ValueError):
        naca4("2400")
    with pytest.raises(ValueError):
        naca5("23000")
    with pytest.raises(ValueError):
        naca4("2412", spacing="sine")


def test_reflex_mean_line_raises():
    """Reflex mean lines raise ValueError with a clear message."""
    with pytest.raises(ValueError, match="eflex"):
        naca5("23112")
    with pytest.raises(ValueError, match="eflex"):
        naca5_camber(np.array([0.5]), "23112")


def test_unknown_5digit_position_raises():
    """Unknown 5-digit camber positions raise ValueError."""
    with pytest.raises(ValueError):
        naca5("26012")
    with pytest.raises(ValueError):
        naca5_camber(np.array([0.5]), "26012")
