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


def test_23012_published_maximum_camber():
    """The 230 mean line has its published maximum camber of about 0.0184 at x 0.15.

    The reference value comes from the published 5-digit mean-line
    equation with m and k1 of the 230 line written here from the
    five-digit mean-line table of Abbott and von Doenhoff (1959, section
    6.5): m = 0.2025, k1 = 15.957. The maximum of the front cubic lies
    where its slope is zero, at x = m (1 - sqrt(m / 3)). The test does
    not read the constants of the code.
    """
    m = 0.2025
    k1 = 15.957
    x_peak = m * (1.0 - np.sqrt(m / 3.0))
    y_peak = k1 / 6.0 * (x_peak**3 - 3.0 * m * x_peak**2 + m**2 * (3.0 - m) * x_peak)
    # Published design: maximum camber near 1.84 percent chord at 15 percent chord.
    assert abs(x_peak - 0.15) < 1e-3
    assert abs(y_peak - 0.0184) < 1e-4
    yc_code, slope_code = naca5_camber(np.array([x_peak]), "23012")
    assert abs(float(yc_code[0]) - y_peak) < 1e-12
    assert abs(float(slope_code[0])) < 1e-12
    x = np.linspace(0.0, 1.0, 200001)
    yc, _ = naca5_camber(x, "23012")
    assert float(np.max(yc)) <= y_peak + 1e-15
    assert abs(float(x[int(np.argmax(yc))]) - x_peak) < 1e-5


def test_thickness_is_perpendicular_to_the_mean_line():
    """On a cambered section the thickness acts normal to the mean line.

    The test builds NACA 2412 from the published formulas of Abbott and
    von Doenhoff (1959, section 6.4), written here, and compares every
    point with the generator. It also checks the geometry of each pair of
    upper and lower points: the midpoint lies on the mean line, the
    distance is 2 y_t, and the line between them is normal to the mean
    line.
    """
    n_points = 41
    m, p, t = 0.02, 0.4, 0.12
    beta = np.linspace(0.0, np.pi, (n_points + 1) // 2)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    front = x < p
    yc = np.where(front, m / p**2 * (2.0 * p * x - x**2),
                  m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2))
    slope = np.where(front, 2.0 * m / p**2 * (p - x), 2.0 * m / (1.0 - p) ** 2 * (p - x))
    theta = np.arctan(slope)
    upper = np.column_stack([x - yt * np.sin(theta), yc + yt * np.cos(theta)])
    lower = np.column_stack([x + yt * np.sin(theta), yc - yt * np.cos(theta)])
    points = naca4("2412", n_points=n_points)
    half = (n_points + 1) // 2
    gen_upper = points[:half][::-1]
    gen_lower = points[half - 1:]
    np.testing.assert_allclose(gen_upper, upper, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(gen_lower, lower, rtol=0.0, atol=1e-15)
    mid = 0.5 * (gen_upper + gen_lower)
    np.testing.assert_allclose(mid[:, 0], x, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(mid[:, 1], yc, rtol=0.0, atol=1e-15)
    gap = gen_upper - gen_lower
    np.testing.assert_allclose(np.linalg.norm(gap, axis=1), 2.0 * yt, rtol=0.0, atol=1e-15)
    tangent = np.column_stack([np.ones_like(slope), slope])
    assert np.max(np.abs(np.sum(gap * tangent, axis=1))) < 1e-15
    # The section is cambered, so the perpendicular offset moves x.
    assert np.max(np.abs(gen_upper[:, 0] - x)) > 1e-4


def test_first_digit_scales_k1_linearly():
    """The first digit scales the 5-digit mean line linearly (k1 times d / 2)."""
    x = np.linspace(0.0, 1.0, 501)
    yc2, d2 = naca5_camber(x, "23012")
    yc4, d4 = naca5_camber(x, "43012")
    yc1, d1 = naca5_camber(x, "13012")
    np.testing.assert_allclose(yc4, 2.0 * yc2, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(d4, 2.0 * d2, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(yc1, 0.5 * yc2, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(d1, 0.5 * d2, rtol=1e-15, atol=0.0)
    assert float(np.max(yc4)) > float(np.max(yc2)) > float(np.max(yc1)) > 0.0


@pytest.mark.parametrize("name", ["01012", "03012", "NACA 01012"])
def test_zero_design_lift_gives_flat_mean_line_with_warning(name):
    """A first digit 0 (k1 = 0) gives the symmetric section and a warning."""
    with pytest.warns(UserWarning, match="zero camber"):
        points = naca(name, n_points=61)
    np.testing.assert_array_equal(points, naca4("0012", n_points=61))
    with pytest.warns(UserWarning, match="zero camber"):
        yc, slope = naca5_camber(np.linspace(0.0, 1.0, 11), name)
    assert np.all(yc == 0.0)
    assert np.all(slope == 0.0)


def test_zero_camber_with_position_digit_warns_for_4_digit():
    """A 4-digit name with zero camber and a position digit warns."""
    with pytest.warns(UserWarning, match="zero camber"):
        points = naca("0412", n_points=61)
    np.testing.assert_array_equal(points, naca4("0012", n_points=61))


@pytest.mark.parametrize("name", ["24-12", "2_4_1_2", "NACA-2412", "NACA_2412", "24 12", "230-12"])
def test_undocumented_separators_raise(name):
    """Hyphens, underscores and inner spaces are not designation forms."""
    with pytest.raises(ValueError, match="Invalid NACA designation"):
        naca(name)


def test_is_naca_designation_checks_the_form():
    """The form test accepts 4 and 5 digits with an optional prefix only."""
    from ventorum.airfoil.naca import is_naca_designation

    for good in ("2412", "NACA 2412", "naca2412", " NACA  23012 ", "1012", 2412, 23012):
        assert is_naca_designation(good)
    for bad in ("12", "123456", "24-12", "naca2412.dat", "2412.dat", "airfoil", 12, True, 2.0, None):
        assert not is_naca_designation(bad)
