# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the airfoil spline and the repaneling functions."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ventorum.airfoil.paneling import AirfoilSpline, repanel, repanel_cosine

CHORD = 1.0
SEMI_MAJOR = 0.5
SEMI_MINOR = 0.06
THICKNESS = 0.12
N_PER_SURFACE = 61
N_PANELS = 160


def ellipse_coords(n_angle: int = 361) -> np.ndarray:
    """Ellipse of thickness ratio 0.12 in Selig order.

    The points are uniform in eccentric angle, which resolves the cubic
    spline near the ends. A coarse uniform-in-x input cannot meet the
    exactness bounds below with any not-a-knot cubic spline, so the
    exactness tests use this resolved sampling.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, 2 * n_angle - 1)
    x = SEMI_MAJOR + SEMI_MAJOR * np.cos(theta)
    y = SEMI_MINOR * np.sin(theta)
    return np.column_stack([x, y])


def naca_symmetric_coords(
    n: int = N_PER_SURFACE, thickness: float = THICKNESS
) -> np.ndarray:
    """Symmetric NACA section with open trailing edge in Selig order.

    The section comes from the analytic thickness formula, sampled with 61
    uniform-in-x points per surface.
    """
    x = np.linspace(0.0, CHORD, n)
    yt = (
        5.0
        * thickness
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
    )
    upper = np.column_stack([x[::-1], yt[::-1]])
    lower = np.column_stack([x[1:], -yt[1:]])
    return np.vstack([upper, lower])


def panel_lengths(nodes: np.ndarray) -> np.ndarray:
    """Euclidean length of each panel in m."""
    return np.linalg.norm(np.diff(nodes, axis=0), axis=1)


def max_adjacent_ratio(nodes: np.ndarray) -> float:
    """Largest length ratio of two adjacent panels."""
    lengths = panel_lengths(nodes)
    both = np.maximum(lengths[:-1] / lengths[1:], lengths[1:] / lengths[:-1])
    return float(np.max(both))


def ellipse_distance(nodes: np.ndarray) -> np.ndarray:
    """First-order distance of nodes to the exact ellipse in m."""
    xn = (nodes[:, 0] - SEMI_MAJOR) / SEMI_MAJOR
    yn = nodes[:, 1] / SEMI_MINOR
    f = xn**2 + yn**2 - 1.0
    grad = np.column_stack([2.0 * xn / SEMI_MAJOR, 2.0 * yn / SEMI_MINOR])
    return np.abs(f) / np.linalg.norm(grad, axis=1)


def polygon_area(nodes: np.ndarray) -> float:
    """Absolute polygon area in m^2 (shoelace formula)."""
    x = nodes[:, 0]
    y = nodes[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def leading_edge_index(nodes: np.ndarray) -> int:
    """Index of the node nearest the leading edge (minimum x in m)."""
    return int(np.argmin(nodes[:, 0]))


def test_repanel_node_count() -> None:
    for coords in (ellipse_coords(), naca_symmetric_coords()):
        assert repanel(coords, n_panels=N_PANELS).shape == (N_PANELS + 1, 2)
        assert repanel(coords, n_panels=40).shape == (41, 2)


def test_repanel_keeps_trailing_edge_points() -> None:
    for coords in (ellipse_coords(), naca_symmetric_coords()):
        for nodes in (repanel(coords), repanel_cosine(coords, 80)):
            np.testing.assert_allclose(nodes[0], coords[0], atol=1e-14)
            np.testing.assert_allclose(nodes[-1], coords[-1], atol=1e-14)


def test_closed_stays_closed_and_open_stays_open() -> None:
    closed = repanel(ellipse_coords())
    assert np.linalg.norm(closed[0] - closed[-1]) < 1e-12
    coords = naca_symmetric_coords()
    gap_in = float(np.linalg.norm(coords[0] - coords[-1]))
    assert gap_in > 0.0
    nodes = repanel(coords)
    gap_out = float(np.linalg.norm(nodes[0] - nodes[-1]))
    assert gap_out == pytest.approx(gap_in, abs=1e-12)


def test_ellipse_leading_edge_node() -> None:
    coords = ellipse_coords()
    spline = AirfoilSpline(coords)
    point = np.asarray(spline(spline.s_leading_edge))
    assert np.linalg.norm(point) < 1e-8
    nodes = repanel(coords)
    assert np.min(np.linalg.norm(nodes, axis=1)) < 1e-8


def test_ellipse_nodes_on_exact_curve() -> None:
    nodes = repanel(ellipse_coords())
    assert np.max(ellipse_distance(nodes)) <= 1e-6


def test_adjacent_panel_ratio() -> None:
    # The 1.5 bound holds for repanel on both shapes. Cosine spacing clusters
    # both ends of each surface, so its first two panels differ by up to a
    # factor of three; that distribution carries no such bound.
    for coords in (ellipse_coords(), naca_symmetric_coords()):
        assert max_adjacent_ratio(repanel(coords)) <= 1.5
    # Strong refinement triggers the weight smoothing path.
    assert max_adjacent_ratio(repanel(naca_symmetric_coords(), curvature_weight=8.0)) <= 1.5


def test_leading_edge_panel_is_smallest() -> None:
    for coords in (ellipse_coords(), naca_symmetric_coords()):
        nodes = repanel(coords)
        lengths = panel_lengths(nodes)
        le = leading_edge_index(nodes)
        assert 0 < le < len(nodes) - 1
        mid_points = 0.5 * (nodes[:-1] + nodes[1:])
        upper = np.arange(le)
        mid = upper[int(np.argmin(np.abs(mid_points[upper, 0] - 0.5 * CHORD)))]
        assert lengths[le - 1] < lengths[mid]
        assert lengths[le] < lengths[mid]


def test_ellipse_area() -> None:
    # Straight panels undercut the curve, so the polygon area needs enough
    # panels to reach the 1e-4 bound (a 160-gon circle is already off by
    # 2.6e-4 relative).
    nodes = repanel(ellipse_coords(), n_panels=640)
    exact = np.pi * SEMI_MAJOR * SEMI_MINOR
    assert polygon_area(nodes) == pytest.approx(exact, rel=1e-4)


def test_repanel_twice() -> None:
    once = repanel(ellipse_coords())
    twice = repanel(once)
    assert twice.shape == once.shape
    assert np.max(np.linalg.norm(twice - once, axis=1)) <= 1e-6


def test_reverse_order_raises() -> None:
    for coords in (ellipse_coords(), naca_symmetric_coords()):
        reversed_coords = coords[::-1].copy()
        with pytest.raises(ValueError):
            AirfoilSpline(reversed_coords)
        with pytest.raises(ValueError):
            repanel(reversed_coords)
        with pytest.raises(ValueError):
            repanel_cosine(reversed_coords, 80)


def test_circle_curvature() -> None:
    # Spline second derivatives converge slowly, so this formula check uses
    # a resolved sampling to isolate the formula from interpolation error.
    radius = 1.0
    theta = np.linspace(0.0, 2.0 * np.pi, 6401)
    coords = np.column_stack(
        [1.0 + radius * np.cos(theta), radius * np.sin(theta)]
    )
    spline = AirfoilSpline(coords)
    s = np.linspace(0.0, spline.length, 65)[1:-1]
    np.testing.assert_allclose(spline.curvature(s), 1.0 / radius, atol=1e-6)


def test_spline_call_derivative_and_length() -> None:
    coords = naca_symmetric_coords()
    spline = AirfoilSpline(coords)
    np.testing.assert_allclose(spline(0.0), coords[0], atol=1e-14)
    np.testing.assert_allclose(spline(spline.length), coords[-1], atol=1e-14)
    s = np.linspace(0.0, spline.length, 25)
    assert spline(s).shape == (25, 2)
    assert spline.derivative(s, 1).shape == (25, 2)
    assert spline.derivative(s, 2).shape == (25, 2)
    with pytest.raises(ValueError):
        spline.derivative(s, 4)
    assert 0.0 < spline.s_leading_edge < spline.length
    a, b = SEMI_MAJOR, SEMI_MINOR
    h = ((a - b) / (a + b)) ** 2
    perimeter = np.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h)))
    assert AirfoilSpline(ellipse_coords()).length == pytest.approx(perimeter, rel=0.02)


def test_invalid_inputs_raise() -> None:
    coords = naca_symmetric_coords()
    with pytest.raises(ValueError):
        repanel(coords, n_panels=1)
    with pytest.raises(ValueError):
        repanel_cosine(coords, 0)
    with pytest.raises(ValueError):
        repanel(coords, curvature_weight=-1.0)
    bad = coords.copy()
    bad[3, 0] = np.nan
    with pytest.raises(ValueError):
        repanel(bad)


def test_repanel_cosine_counts() -> None:
    for coords in (ellipse_coords(), naca_symmetric_coords()):
        nodes = repanel_cosine(coords, 80)
        assert nodes.shape == (161, 2)
        le = leading_edge_index(nodes)
        assert 0 < le < len(nodes) - 1


def coarse_ellipse_coords(n: int = N_PER_SURFACE) -> np.ndarray:
    """Ellipse of thickness ratio 0.12 with 61 uniform-in-x points per surface.

    This is the coarse input of the paneling specification. The trailing
    edge is the blunt end of the ellipse at (1, 0); it is closed.
    """
    x = np.linspace(0.0, CHORD, n)
    y = SEMI_MINOR * np.sqrt(np.clip(1.0 - ((x - SEMI_MAJOR) / SEMI_MAJOR) ** 2, 0.0, None))
    upper = np.column_stack([x[::-1], y[::-1]])
    lower = np.column_stack([x[1:], -y[1:]])
    return np.vstack([upper, lower])


def test_coarse_ellipse_accuracy_limit() -> None:
    """Repaneling the coarse ellipse input meets a looser, measured bound.

    The fine-input checks above (1e-6 chord) need the resolved sampling.
    With 61 uniform-in-x points per surface, the spline cannot follow the
    blunt ends of the ellipse, where the slope is infinite and the input
    points are far apart along the curve. Measured node distances to the
    exact ellipse for n_panels 160: 1.3e-3 chord near the blunt trailing
    end (x > 0.95), 3.0e-4 chord near the leading edge (x < 0.05) and
    1.2e-5 chord between. The bounds below are 1.5e-3, 4e-4 and 2e-5
    chord. The leading-edge node stays at (0, 0) within 1e-8, and the
    polygon area of 640 panels is within 5e-4 relative of pi a b
    (measured 4.1e-4).
    """
    coords = coarse_ellipse_coords()
    assert coords.shape == (2 * N_PER_SURFACE - 1, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nodes = repanel(coords, n_panels=N_PANELS)
    dist = ellipse_distance(nodes)
    x = nodes[:, 0]
    assert np.max(dist) <= 1.5e-3
    assert np.max(dist[x < 0.05]) <= 4e-4
    assert np.max(dist[(x >= 0.05) & (x <= 0.95)]) <= 2e-5
    assert np.min(np.linalg.norm(nodes, axis=1)) < 1e-8
    assert max_adjacent_ratio(nodes) <= 1.5
    fine = repanel(coords, n_panels=640)
    exact = np.pi * SEMI_MAJOR * SEMI_MINOR
    assert polygon_area(fine) == pytest.approx(exact, rel=5e-4)


def test_leading_edge_search_raises_without_a_distance_maximum() -> None:
    """A curve whose distance to the trailing-edge midpoint has no maximum.

    Before the fix, the search accepted any sign change of the tangency
    function and silently returned the distance minimum (the bottom of
    this bowl) as the leading edge.
    """
    x = np.linspace(0.0, 1.0, 21)
    y = -0.3 + 0.5 * (x - 0.5) ** 2
    bowl = np.column_stack([x, y])
    with pytest.raises(ValueError, match="leading edge"):
        AirfoilSpline(bowl)


def test_ratio_warning_when_smoothing_cannot_reach_the_bound() -> None:
    """Very few panels cannot meet the 1.5 ratio: the function warns.

    Before the fix, the nodes came back with a ratio above 1.5 and no
    signal.
    """
    coords = naca_symmetric_coords()
    with pytest.warns(UserWarning, match="adjacent panel length ratio"):
        nodes = repanel(coords, n_panels=3)
    assert nodes.shape == (4, 2)
    assert max_adjacent_ratio(nodes) > 1.5


def test_default_repanel_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for coords in (ellipse_coords(), naca_symmetric_coords()):
            repanel(coords)


def test_two_panels_raise_a_clear_error() -> None:
    """Two panels give three nodes, too few for the spline of the next pass."""
    with pytest.raises(ValueError, match="at least 3"):
        repanel(naca_symmetric_coords(), n_panels=2)
