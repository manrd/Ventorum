# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for plain-flap deflection of airfoil coordinates."""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from ventorum.airfoil.flap import deflect_plain_flap, flap_hinge_point

HINGES = [0.6, 0.7, 0.8]
DEFLECTIONS = [-30.0, -10.0, 5.0, 10.0, 20.0, 30.0, 45.0]


def _naca4_coords(m, p, t, n=81, closed=True):
    """NACA 4-digit section in Selig order from the analytic formulas."""
    beta = np.linspace(0.0, math.pi, n)
    xc = 0.5 * (1.0 - np.cos(beta))
    if m == 0.0:
        yc = np.zeros_like(xc)
        dyc = np.zeros_like(xc)
    else:
        fore = xc < p
        yc = np.where(fore,
                      m / p**2 * (2.0 * p * xc - xc**2),
                      m / (1.0 - p)**2 * ((1.0 - 2.0 * p) + 2.0 * p * xc - xc**2))
        dyc = np.where(fore,
                       2.0 * m / p**2 * (p - xc),
                       2.0 * m / (1.0 - p)**2 * (p - xc))
    last = -0.1036 if closed else -0.1015
    yt = 5.0 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2
                    + 0.2843 * xc**3 + last * xc**4)
    theta = np.arctan(dyc)
    upper = np.column_stack([xc - yt * np.sin(theta), yc + yt * np.cos(theta)])[::-1]
    lower = np.column_stack([xc + yt * np.sin(theta), yc - yt * np.cos(theta)])[1:]
    return np.vstack([upper, lower])


def _sections():
    return {
        "naca0012": _naca4_coords(0.0, 0.4, 0.12),
        "naca4412": _naca4_coords(0.04, 0.4, 0.12),
    }


def _signed_area(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _rotate_about(pts, hinge, angle_deg):
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    d = np.asarray(pts, dtype=float) - hinge
    return hinge + np.column_stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]])


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segs_touch(p1, p2, p3, p4, eps=1e-12):
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    if ((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0)) and \
       ((d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0)):
        return True
    for d, a, b, q in ((d1, p3, p4, p1), (d2, p3, p4, p2), (d3, p1, p2, p3), (d4, p1, p2, p4)):
        if abs(d) <= eps and min(a[0], b[0]) <= q[0] <= max(a[0], b[0]) \
                and min(a[1], b[1]) <= q[1] <= max(a[1], b[1]):
            return True
    return False


def _as_ring(pts):
    if np.linalg.norm(pts[0] - pts[-1]) <= 1e-12:
        return pts[:-1]
    return pts


def _has_self_intersection(pts):
    q = _as_ring(pts)
    m = q.shape[0]
    a = q
    b = np.roll(q, -1, axis=0)
    x0 = np.minimum(a[:, 0], b[:, 0])
    x1 = np.maximum(a[:, 0], b[:, 0])
    y0 = np.minimum(a[:, 1], b[:, 1])
    y1 = np.maximum(a[:, 1], b[:, 1])
    overlap = (x0[:, None] <= x1[None, :]) & (x0[None, :] <= x1[:, None]) \
        & (y0[:, None] <= y1[None, :]) & (y0[None, :] <= y1[:, None])
    for i, j in zip(*np.nonzero(np.triu(overlap, k=1))):
        if j == i + 1 or (i == 0 and j == m - 1):
            continue
        if _segs_touch(a[int(i)], b[int(i)], a[int(j)], b[int(j)]):
            return True
    return False


def _check_valid_polygon(out, base, hinge_x, delta, y_hinge=0.5):
    assert out.shape[1] == 2
    assert out.dtype == np.float64
    assert _signed_area(out) > 0.0
    assert not _has_self_intersection(out)
    hinge = flap_hinge_point(base, hinge_x, y_hinge)
    expected_te = _rotate_about(base[[0, -1]], hinge, -delta)
    np.testing.assert_allclose(out[[0, -1]], expected_te, atol=1e-12, rtol=0.0)


def _check_hinge_angle(out, base, hinge_x, delta, y_hinge=0.5):
    hinge = flap_hinge_point(base, hinge_x, y_hinge)
    mid_in = 0.5 * (base[0] + base[-1])
    mid_out = 0.5 * (out[0] + out[-1])
    a_in = math.degrees(math.atan2(mid_in[1] - hinge[1], mid_in[0] - hinge[0]))
    a_out = math.degrees(math.atan2(mid_out[1] - hinge[1], mid_out[0] - hinge[0]))
    change = (a_out - a_in + 180.0) % 360.0 - 180.0
    assert change == pytest.approx(-delta, abs=1e-9)


def _check_fixed_part(out, base, hinge_x):
    sel_out = out[out[:, 0] < hinge_x - 0.05]
    sel_in = base[base[:, 0] < hinge_x - 0.05]
    assert sel_out.shape[0] > 0 and sel_in.shape[0] > 0
    dist = np.sqrt(((sel_out[:, None, :] - sel_in[None, :, :]) ** 2).sum(-1))
    assert dist.min(axis=1).max() < 1e-15
    assert dist.min(axis=0).max() < 1e-15


@pytest.mark.parametrize("name", ["naca0012", "naca4412"])
def test_base_sections_are_clean(name):
    base = _sections()[name]
    assert _signed_area(base) > 0.0
    assert not _has_self_intersection(base)


@pytest.mark.parametrize("name", ["naca0012", "naca4412"])
def test_zero_deflection_returns_same_points(name):
    # The treatment "none" adds no node, so a zero deflection returns the
    # input points. "merge" and "repanel" (the default) add the hinge node
    # also at a zero deflection (tests/test_hinge_zero_deflection.py).
    base = _sections()[name]
    out = deflect_plain_flap(base, 0.7, 0.0, hinge_treatment="none")
    assert out is not base
    np.testing.assert_array_equal(out, base)


@pytest.mark.parametrize("name", ["naca0012", "naca4412"])
@pytest.mark.parametrize("hinge_x", HINGES)
@pytest.mark.parametrize("delta", DEFLECTIONS)
def test_deflected_section_stays_valid(name, hinge_x, delta):
    base = _sections()[name]
    out = deflect_plain_flap(base, hinge_x, delta, hinge_treatment="none")
    _check_valid_polygon(out, base, hinge_x, delta)
    _check_hinge_angle(out, base, hinge_x, delta)
    _check_fixed_part(out, base, hinge_x)
    if abs(delta) == 10.0:
        area0 = _signed_area(base)
        assert abs(_signed_area(out) - area0) / area0 < 0.02


@pytest.mark.parametrize("name", ["naca0012", "naca4412"])
def test_hinge_point_uses_thickness_fraction(name):
    base = _sections()[name]
    xh = 0.7
    le = int(np.argmin(base[:, 0]))
    upper = base[:le + 1][::-1]
    lower = base[le:]
    y_up = float(np.interp(xh, upper[:, 0], upper[:, 1]))
    y_lo = float(np.interp(xh, lower[:, 0], lower[:, 1]))
    hinge = flap_hinge_point(base, xh, 0.5)
    assert hinge.shape == (2,)
    assert hinge[0] == pytest.approx(xh)
    assert hinge[1] == pytest.approx(0.5 * (y_up + y_lo))
    np.testing.assert_allclose(flap_hinge_point(base, xh, 0.0), [xh, y_lo], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(flap_hinge_point(base, xh, 1.0), [xh, y_up], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("y_hinge", [0.0, 1.0])
@pytest.mark.parametrize("delta", [10.0, -10.0])
def test_hinge_on_surface_stays_valid(y_hinge, delta):
    base = _sections()["naca4412"]
    out = deflect_plain_flap(base, 0.7, delta, y_hinge=y_hinge)
    _check_valid_polygon(out, base, 0.7, delta, y_hinge)
    _check_hinge_angle(out, base, 0.7, delta, y_hinge)


@pytest.mark.parametrize("delta", [60.0, -60.0])
def test_limit_deflection_runs(delta):
    base = _sections()["naca0012"]
    out = deflect_plain_flap(base, 0.7, delta, hinge_treatment="none")
    _check_valid_polygon(out, base, 0.7, delta)


def test_explicit_arc_count_runs():
    base = _sections()["naca4412"]
    out = deflect_plain_flap(base, 0.7, 20.0, n_arc=6)
    _check_valid_polygon(out, base, 0.7, 20.0)
    _check_hinge_angle(out, base, 0.7, 20.0)


@pytest.mark.parametrize("delta", [10.0, -20.0, 30.0])
def test_open_trailing_edge_stays_valid(delta):
    base = _naca4_coords(0.04, 0.4, 0.12, closed=False)
    assert not np.array_equal(base[0], base[-1])
    out = deflect_plain_flap(base, 0.7, delta, hinge_treatment="none")
    _check_valid_polygon(out, base, 0.7, delta)
    _check_hinge_angle(out, base, 0.7, delta)
    _check_fixed_part(out, base, 0.7)


@pytest.mark.parametrize("xh", [0.05, 0.04, 0.0, 0.98, 0.99, 1.5, -0.2, float("nan")])
def test_bad_hinge_station_raises(xh):
    base = _sections()["naca0012"]
    with pytest.raises(ValueError):
        deflect_plain_flap(base, xh, 10.0)


@pytest.mark.parametrize("delta", [60.0001, -60.1, 90.0, float("nan"), float("inf")])
def test_bad_deflection_raises(delta):
    base = _sections()["naca0012"]
    with pytest.raises(ValueError):
        deflect_plain_flap(base, 0.7, delta)


@pytest.mark.parametrize("yh", [-0.1, 1.1, float("nan")])
def test_bad_hinge_height_raises(yh):
    base = _sections()["naca0012"]
    with pytest.raises(ValueError):
        deflect_plain_flap(base, 0.7, 10.0, y_hinge=yh)
    with pytest.raises(ValueError):
        flap_hinge_point(base, 0.7, yh)


@pytest.mark.parametrize("n_arc", [0, 1, -3, 2.5, "4"])
def test_bad_arc_count_raises(n_arc):
    base = _sections()["naca0012"]
    with pytest.raises(ValueError):
        deflect_plain_flap(base, 0.7, 10.0, n_arc=n_arc)


def test_bad_coords_raise():
    base = _sections()["naca0012"]
    with pytest.raises(ValueError):
        deflect_plain_flap(base[:, 0], 0.7, 10.0)
    with pytest.raises(ValueError):
        deflect_plain_flap(base[:3], 0.7, 10.0)
    bad = base.copy()
    bad[5, 0] = float("nan")
    with pytest.raises(ValueError):
        deflect_plain_flap(bad, 0.7, 10.0)
    with pytest.raises(ValueError):
        deflect_plain_flap([[0.0, 0.0], [0.5, 0.05], [1.0, 0.0]], 0.7, 10.0)


SWEEP_SHAPES = {"0006": (0.0, 0.4, 0.06), "0012": (0.0, 0.4, 0.12),
                "2412": (0.02, 0.4, 0.12), "4415": (0.04, 0.4, 0.15)}
SWEEP_N_POINTS = [31, 61, 161]


def _sweep_section(name, n_points):
    """Open trailing-edge NACA section with n_points in total (Selig order)."""
    m, p, t = SWEEP_SHAPES[name]
    return _naca4_coords(m, p, t, n=(n_points + 1) // 2, closed=False)


@pytest.mark.parametrize(
    "name, n_points, hinge_x, delta, y_hinge",
    [
        ("2412", 31, 0.9, -20.0, 0.0),
        ("2412", 31, 0.1, 15.0, 0.5),
        ("0006", 31, 0.1, -10.0, 0.0),
        ("0006", 161, 0.1, 10.0, 1.0),
        ("0012", 31, 0.1, -15.0, 0.0),
        ("0012", 61, 0.1, 20.0, 0.5),
        ("2412", 31, 0.25, -10.0, 0.0),
        ("2412", 31, 0.1, 10.0, 0.25),
    ],
)
def test_coarse_input_regression_cases_are_simple(name, n_points, hinge_x, delta, y_hinge):
    """Coarse inputs that gave a self-intersecting polygon before the fix.

    The old construction added the filler arc on any side where the fixed
    and the moved polylines did not cross. On coarse input the moved
    surface can pass the fixed corner on the compressed side, so the arc
    folded the contour over itself.
    """
    base = _sweep_section(name, n_points)
    out = deflect_plain_flap(base, hinge_x, delta, y_hinge=y_hinge)
    _check_valid_polygon(out, base, hinge_x, delta, y_hinge)
    _check_hinge_angle(out, base, hinge_x, delta, y_hinge)


def _sweep(name, n_points, hinges, deltas, heights):
    base = _sweep_section(name, n_points)
    for hinge_x in hinges:
        for delta in deltas:
            for y_hinge in heights:
                out = deflect_plain_flap(base, hinge_x, delta, y_hinge=y_hinge)
                _check_valid_polygon(out, base, hinge_x, delta, y_hinge)


@pytest.mark.parametrize("n_points", SWEEP_N_POINTS)
@pytest.mark.parametrize("name", list(SWEEP_SHAPES))
def test_flap_sweep_gives_simple_polygons(name, n_points):
    """Every hinge, deflection and hinge height gives a simple CCW polygon.

    Grid: hinges 0.1 to 0.97, deflections -60 to 60 deg, hinge heights 0
    to 1, on NACA 0006, 0012, 2412 and 4415 with 31, 61 and 161 points.
    """
    _sweep(name, n_points,
           hinges=[0.1, 0.25, 0.5, 0.75, 0.9, 0.97],
           deltas=[-60.0, -20.0, -10.0, 10.0, 15.0, 60.0],
           heights=[0.0, 0.5, 1.0])


@pytest.mark.slow
@pytest.mark.parametrize("n_points", SWEEP_N_POINTS)
@pytest.mark.parametrize("name", list(SWEEP_SHAPES))
def test_flap_dense_sweep_gives_simple_polygons(name, n_points):
    """The dense grid of the same sweep (12 hinges, 18 deflections, 5 heights)."""
    _sweep(name, n_points,
           hinges=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97],
           deltas=[-60.0, -45.0, -30.0, -20.0, -15.0, -10.0, -5.0, -2.0, -1.0,
                   1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0],
           heights=[0.0, 0.25, 0.5, 0.75, 1.0])


def test_compressed_side_follows_the_deflection_sign():
    """Trailing edge down puts the filler arc on the upper surface.

    With the hinge at 0.3 of the thickness, the arc about the hinge has
    the radius of the upper corner for trailing edge down (upper surface
    stretched) and the radius of the lower corner for trailing edge up
    (lower surface stretched). The arc holds n_arc points.
    """
    base = _sweep_section("2412", 61)
    xh, yh, n_arc = 0.7, 0.3, 8
    hinge = flap_hinge_point(base, xh, yh)
    r_up = float(flap_hinge_point(base, xh, 1.0)[1] - hinge[1])
    r_lo = float(hinge[1] - flap_hinge_point(base, xh, 0.0)[1])
    assert abs(r_up - r_lo) > 1e-3
    for delta, r_arc, r_cut in ((20.0, r_up, r_lo), (-20.0, r_lo, r_up)):
        out = deflect_plain_flap(base, xh, delta, y_hinge=yh, n_arc=n_arc, hinge_treatment="none")
        dist = np.linalg.norm(out - hinge, axis=1)
        assert int(np.sum(np.abs(dist - r_arc) < 1e-12)) == n_arc
        assert int(np.sum(np.abs(dist - r_cut) < 1e-12)) == 0


def test_flapped_shape_leading_edge_is_the_package_definition():
    """On a deflected section the flap split uses the shared leading edge.

    A deflected section moves the trailing-edge midpoint, so the point
    farthest from it (the package leading edge) is no longer the point of
    minimum x. The flap functions split the surfaces at the farthest
    point, the same point as AirfoilShape.leading_edge_index and the
    node nearest to the spline leading edge.
    """
    from ventorum.airfoil import AirfoilShape, AirfoilSpline
    from ventorum.airfoil.flap import _surfaces
    from ventorum.airfoil.geometry import leading_edge_index

    flapped = AirfoilShape.naca("2412").with_flap(0.7, 30.0)
    coords = flapped.coords
    far = leading_edge_index(coords)
    assert far == flapped.leading_edge_index
    assert far != int(np.argmin(coords[:, 0]))
    spline = AirfoilSpline(coords)
    tip = spline(spline.s_leading_edge)
    assert int(np.argmin(np.linalg.norm(coords - tip, axis=1))) == far
    upper, lower = _surfaces(coords)
    np.testing.assert_array_equal(upper[0], coords[far])
    np.testing.assert_array_equal(lower[0], coords[far])
    np.testing.assert_array_equal(upper, flapped.upper())
    np.testing.assert_array_equal(lower, flapped.lower())
    # A second flap on the deflected section works on the same split.
    twice = deflect_plain_flap(coords, 0.3, 10.0)
    _check_valid_polygon(twice, coords, 0.3, 10.0)


def test_clockwise_input_raises():
    base = _sections()["naca0012"]
    with pytest.raises(ValueError, match="Selig order"):
        deflect_plain_flap(base[::-1].copy(), 0.7, 10.0)


def test_output_check_raises_for_a_crossing_polygon():
    from ventorum.airfoil.flap import _check_output
    from ventorum.airfoil.geometry import AirfoilGeometryError

    t = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    eight = np.column_stack([0.5 + 0.5 * np.sin(t), 0.2 * np.sin(t) * np.cos(t)])
    with pytest.raises(AirfoilGeometryError, match="not a simple polygon"):
        _check_output(eight, 0.7, 10.0, 0.5)


def test_hinge_line_cut_branches():
    from ventorum.airfoil.flap import _hinge_line_cut
    from ventorum.airfoil.geometry import AirfoilGeometryError

    moved = np.array([[0.6, 0.0], [0.8, -0.1], [1.0, -0.2]])
    j, point = _hinge_line_cut(moved, 0.7)
    assert j == 0
    np.testing.assert_allclose(point, [0.7, -0.05], rtol=0.0, atol=1e-15)
    j, point = _hinge_line_cut(moved, 0.6)
    assert j == -1
    np.testing.assert_array_equal(point, moved[0])
    with pytest.raises(AirfoilGeometryError, match="hinge line"):
        _hinge_line_cut(moved, 1.5)


def test_non_monotonic_nose_is_accepted_forward_of_the_hinge():
    """A short decreasing-x run at the nose does not block the flap."""
    flapped = deflect_plain_flap(_sweep_section("2412", 161), 0.7, 30.0)
    upper_like = flapped[: int(np.argmax(np.linalg.norm(
        flapped - 0.5 * (flapped[0] + flapped[-1]), axis=1))) + 1]
    assert upper_like.shape[0] > 0
    out = deflect_plain_flap(flapped, 0.4, -10.0)
    _check_valid_polygon(out, flapped, 0.4, -10.0)


def test_decreasing_x_aft_of_the_hinge_raises():
    base = _sections()["naca0012"].copy()
    # Swap two upper-surface points near x = 0.8 so that x decreases there.
    k = int(np.argmin(np.abs(base[:40, 0] - 0.8)))
    base[[k, k + 1]] = base[[k + 1, k]]
    with pytest.raises(ValueError, match="non-decreasing"):
        deflect_plain_flap(base, 0.7, 10.0)


# Hinge treatments ---------------------------------------------------------

TREATMENT_NAMES = ["0012", "2412", "4415"]
TREATMENT_HINGES = [0.6, 0.7, 0.75, 0.8, 0.85]
TREATMENT_DELTAS = [-20.0, -15.0, -10.0, -5.0, -2.0, 2.0, 5.0, 10.0, 15.0, 20.0]
#: Largest distance [chord units] from a "repanel" node to the deflected
#: outline of "none" on these inputs (60 to 240 panels).
REPANEL_DEVIATION_BOUND = 2.0e-3


@functools.cache
def _package_section(name, n_panels):
    """Package NACA section, repaneled to n_panels (None keeps the generator points)."""
    from ventorum.airfoil import airfoil

    shape = airfoil(name)
    if n_panels is not None:
        shape = shape.repanel(n_panels=n_panels)
    coords = np.array(shape.coords, dtype=float)
    coords.setflags(write=False)
    return coords


def _distance_to_outline(pts, outline):
    """Distance [chord units] of each point to the closed polyline outline."""
    a = outline
    b = np.roll(outline, -1, axis=0)
    d = b - a
    l2 = np.maximum((d ** 2).sum(1), 1e-300)
    t = np.clip(((pts[:, None, :] - a[None]) * d[None]).sum(-1) / l2, 0.0, 1.0)
    q = a[None] + t[..., None] * d[None]
    return np.sqrt(((pts[:, None, :] - q) ** 2).sum(-1)).min(1)


def _zone_panel_ratios(out, hinge_x):
    """Panel length over the mean of its neighbours, for panels with an end in the hinge zone."""
    from ventorum.airfoil.geometry import leading_edge_index

    lengths = np.linalg.norm(np.diff(out, axis=0), axis=1)
    keep = np.abs(out[:, 0] - hinge_x) < 0.05
    keep[[0, out.shape[0] - 1, leading_edge_index(out)]] = False
    ratios = []
    for i in range(lengths.size):
        if keep[i] or keep[i + 1]:
            near = [lengths[j] for j in (i - 1, i + 1) if 0 <= j < lengths.size]
            ratios.append(lengths[i] / np.mean(near))
    return np.array(ratios)


def _zone_max_turn(out, hinge_x):
    """Largest turning angle [deg] at the nodes with |x - hinge_x| < 0.05."""
    d = np.diff(out, axis=0)
    a = np.arctan2(d[:, 1], d[:, 0])
    turn = np.degrees(np.abs((np.diff(a) + np.pi) % (2.0 * np.pi) - np.pi))
    zone = np.abs(out[1:-1, 0] - hinge_x) < 0.05
    return float(np.max(turn[zone]))


def test_default_treatment_is_repanel():
    base = _package_section("2412", 160)
    for delta in (5.0, -10.0, 20.0):
        np.testing.assert_array_equal(
            deflect_plain_flap(base, 0.75, delta),
            deflect_plain_flap(base, 0.75, delta, hinge_treatment="repanel"),
        )


@pytest.mark.parametrize("treatment", ["repanel", "merge"])
@pytest.mark.parametrize("name", TREATMENT_NAMES)
@pytest.mark.parametrize("hinge_x", TREATMENT_HINGES)
def test_treated_sections_stay_valid(treatment, name, hinge_x):
    """Simple CCW polygon, exact trailing-edge rotation, exact flap angle (analytic input)."""
    m, p, t = SWEEP_SHAPES[name]
    base = _naca4_coords(m, p, t, n=81, closed=False)
    for delta in TREATMENT_DELTAS:
        out = deflect_plain_flap(base, hinge_x, delta, hinge_treatment=treatment)
        _check_valid_polygon(out, base, hinge_x, delta)
        _check_hinge_angle(out, base, hinge_x, delta)
        if treatment == "merge":
            _check_fixed_part(out, base, hinge_x)
        else:
            assert out.shape == base.shape


@pytest.mark.parametrize("name", TREATMENT_NAMES)
@pytest.mark.parametrize("n_panels", [None, 120, 160, 240])
def test_merge_keeps_outline_nodes_and_removes_short_panels(name, n_panels):
    """The kept nodes are nodes of the deflected outline; no short panel stays near the hinge."""
    base = _package_section(name, n_panels)
    for hinge_x in TREATMENT_HINGES:
        for delta in TREATMENT_DELTAS:
            none = deflect_plain_flap(base, hinge_x, delta, hinge_treatment="none")
            out = deflect_plain_flap(base, hinge_x, delta, hinge_treatment="merge")
            _check_valid_polygon(out, base, hinge_x, delta)
            _check_fixed_part(out, base, hinge_x)
            outline = {tuple(q) for q in none}
            assert all(tuple(q) in outline for q in out)
            assert out.shape[0] <= none.shape[0]
            assert np.min(_zone_panel_ratios(out, hinge_x)) >= 0.5


@pytest.mark.parametrize("name", TREATMENT_NAMES)
@pytest.mark.parametrize("n_panels", [None, 120, 160, 240])
def test_repanel_stays_near_the_deflected_outline(name, n_panels):
    """Panel count kept, hinge and trailing-edge nodes kept, deviation below the bound."""
    base = _package_section(name, n_panels)
    for hinge_x in TREATMENT_HINGES:
        for delta in TREATMENT_DELTAS:
            none = deflect_plain_flap(base, hinge_x, delta)
            out = deflect_plain_flap(base, hinge_x, delta, hinge_treatment="repanel")
            _check_valid_polygon(out, base, hinge_x, delta)
            _check_hinge_angle(out, base, hinge_x, delta)
            assert out.shape == base.shape
            assert np.max(_distance_to_outline(out, none)) < REPANEL_DEVIATION_BOUND
            le = int(np.argmax(np.linalg.norm(out - 0.5 * (out[0] + out[-1]), axis=1)))
            outline = {tuple(q) for q in none if abs(q[0] - hinge_x) < 0.02}
            for half in (out[: le + 1], out[le:]):
                assert any(tuple(q) in outline for q in half)


def test_repanel_with_a_panel_count():
    base = _package_section("2412", 160)
    out = deflect_plain_flap(base, 0.75, 10.0, hinge_treatment="repanel", n_panels=200)
    assert out.shape == (201, 2)
    _check_valid_polygon(out, base, 0.75, 10.0)


def test_treatments_at_zero_deflection():
    """Zero deflection: "merge" keeps the outline, "repanel" stays on it."""
    base = _package_section("4415", 160)
    merged = deflect_plain_flap(base, 0.7, 0.0, hinge_treatment="merge")
    assert {tuple(q) for q in base} <= {tuple(q) for q in merged}
    assert np.max(_distance_to_outline(merged, base)) < 1e-15
    rep = deflect_plain_flap(base, 0.7, 0.0, hinge_treatment="repanel")
    assert rep.shape == base.shape
    assert np.max(_distance_to_outline(rep, base)) < 1e-4
    np.testing.assert_array_equal(rep[[0, -1]], base[[0, -1]])


def test_small_deflection_step_panel_is_removed():
    """The hinge-line step of "none" (NACA 2412, hinge 0.75, 5 deg) is gone with both treatments."""
    base = _package_section("2412", 160)
    none = deflect_plain_flap(base, 0.75, 5.0, hinge_treatment="none")
    assert np.min(_zone_panel_ratios(none, 0.75)) < 0.01
    assert _zone_max_turn(none, 0.75) > 80.0
    merged = deflect_plain_flap(base, 0.75, 5.0, hinge_treatment="merge")
    assert np.min(_zone_panel_ratios(merged, 0.75)) >= 0.5
    assert _zone_max_turn(merged, 0.75) < 10.0
    rep = deflect_plain_flap(base, 0.75, 5.0, hinge_treatment="repanel")
    lengths = np.linalg.norm(np.diff(rep, axis=0), axis=1)
    assert np.max(np.maximum(lengths[1:] / lengths[:-1], lengths[:-1] / lengths[1:])) <= 1.5
    assert _zone_max_turn(rep, 0.75) < 10.0


def test_merge_reduces_the_arc_end_corner():
    """At 10 deg the largest corner near the hinge gets smaller with "merge"."""
    base = _package_section("2412", 160)
    none = deflect_plain_flap(base, 0.75, 10.0)
    merged = deflect_plain_flap(base, 0.75, 10.0, hinge_treatment="merge")
    assert _zone_max_turn(merged, 0.75) < _zone_max_turn(none, 0.75)


def test_bad_treatment_arguments_raise():
    base = _sections()["naca0012"]
    with pytest.raises(ValueError, match="hinge_treatment"):
        deflect_plain_flap(base, 0.7, 10.0, hinge_treatment="smooth")
    with pytest.raises(ValueError, match="hinge_treatment"):
        deflect_plain_flap(base, 0.7, 10.0, hinge_treatment=None)
    for treatment in ("none", "merge"):
        with pytest.raises(ValueError, match="n_panels"):
            deflect_plain_flap(base, 0.7, 10.0, hinge_treatment=treatment, n_panels=100)
    for bad in (3, 0, -5, 2.5, "100"):
        with pytest.raises(ValueError, match="n_panels"):
            deflect_plain_flap(base, 0.7, 10.0, hinge_treatment="repanel", n_panels=bad)
