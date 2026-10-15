# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for plain-flap deflection of airfoil coordinates."""

from __future__ import annotations

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
    base = _sections()[name]
    out = deflect_plain_flap(base, 0.7, 0.0)
    assert out is not base
    np.testing.assert_array_equal(out, base)


@pytest.mark.parametrize("name", ["naca0012", "naca4412"])
@pytest.mark.parametrize("hinge_x", HINGES)
@pytest.mark.parametrize("delta", DEFLECTIONS)
def test_deflected_section_stays_valid(name, hinge_x, delta):
    base = _sections()[name]
    out = deflect_plain_flap(base, hinge_x, delta)
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
    out = deflect_plain_flap(base, 0.7, delta)
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
    out = deflect_plain_flap(base, 0.7, delta)
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
