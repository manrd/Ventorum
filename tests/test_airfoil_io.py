# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the airfoil shape type and the Selig and Lednicer readers and writers."""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.airfoil.geometry import AirfoilGeometryError, AirfoilShape
from ventorum.airfoil.io import parse_airfoil, read_airfoil, write_lednicer, write_selig


def _naca_symmetric_coords(thickness_ratio: float = 0.12, n_each: int = 101) -> np.ndarray:
    """Symmetric NACA 4-digit section in Selig order (closed trailing edge)."""
    beta = np.linspace(0.0, np.pi, n_each)
    x = 0.5 * (1.0 - np.cos(beta))
    t = float(thickness_ratio)
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    upper = np.column_stack([x, yt])
    lower = np.column_stack([x, -yt])
    upper_le_to_te = upper
    lower_le_to_te = lower
    selig = np.vstack([upper_le_to_te[::-1], lower_le_to_te[1:]])
    return selig


def _ellipse_coords(n: int = 60) -> np.ndarray:
    """Ellipse section in Selig order (closed trailing edge)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    x = 0.5 * (1.0 - np.cos(theta))
    y = 0.06 * np.sin(theta)
    upper = np.column_stack([x[: n // 2 + 1][::-1], y[: n // 2 + 1][::-1]])
    lower = np.column_stack([x[n // 2 :], y[n // 2 :]])
    selig = np.vstack([upper, lower[1:]])
    return selig


def test_round_trip_selig(tmp_path):
    coords = _naca_symmetric_coords()
    shape = AirfoilShape(name="NACA0012", coords=coords)
    path = tmp_path / "naca0012.dat"
    write_selig(shape, path, precision=6)
    back = read_airfoil(path)
    assert back.coords.shape == shape.coords.shape
    assert np.allclose(back.coords, shape.coords, atol=1e-6)


def test_round_trip_lednicer(tmp_path):
    coords = _naca_symmetric_coords()
    shape = AirfoilShape(name="NACA0012", coords=coords)
    path = tmp_path / "naca0012.led"
    write_lednicer(shape, path, precision=6)
    back = read_airfoil(path)
    assert back.coords.shape == shape.coords.shape
    assert np.allclose(back.coords, shape.coords, atol=1e-6)


def test_lednicer_and_selig_give_same_shape(tmp_path):
    coords = _naca_symmetric_coords()
    shape = AirfoilShape(name="NACA0012", coords=coords)
    selig_path = tmp_path / "a.dat"
    led_path = tmp_path / "b.dat"
    write_selig(shape, selig_path, precision=6)
    write_lednicer(shape, led_path, precision=6)
    from_selig = read_airfoil(selig_path)
    from_led = read_airfoil(led_path)
    assert from_led.coords.shape == from_selig.coords.shape
    assert np.allclose(from_led.coords, from_selig.coords, atol=1e-6)


def test_reversed_order_is_corrected():
    coords = _naca_symmetric_coords()
    shape = AirfoilShape(name="fwd", coords=coords)
    rev = AirfoilShape(name="rev", coords=coords[::-1].copy())
    assert rev.area() < 0.0
    text = "REV\n" + "\n".join(f"{x} {y}" for x, y in coords[::-1])
    parsed = parse_airfoil(text)
    assert parsed.area() > 0.0
    assert np.allclose(parsed.coords, shape.coords, atol=1e-12)


def test_comma_separated_and_commented_files_parse():
    coords = _naca_symmetric_coords()
    lines = ["# a comment line", "COMMA", "# another comment"]
    lines += [f"{x},{y}" for x, y in coords]
    parsed = parse_airfoil("\n".join(lines))
    assert parsed.name == "COMMA"
    assert np.allclose(parsed.coords, coords, atol=1e-12)
    tabbed = "TAB\n" + "\n".join(f"{x}\t{y}" for x, y in coords)
    assert np.allclose(parse_airfoil(tabbed).coords, coords, atol=1e-12)


def test_normalized_recovers_original():
    coords = _naca_symmetric_coords()
    shape = AirfoilShape(name="base", coords=coords)
    angle = np.radians(7.0)
    scale = 2.5
    shift = np.array([3.0, -1.5])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = (coords @ rot.T) * scale + shift
    moved_shape = AirfoilShape(name="moved", coords=moved)
    back = moved_shape.normalized()
    assert np.allclose(back.coords, shape.coords, atol=1e-12)


def test_max_thickness_of_12_percent_section():
    coords = _naca_symmetric_coords(thickness_ratio=0.12, n_each=101)
    shape = AirfoilShape(name="NACA0012", coords=coords)
    assert shape.n_points == 201
    value, x = shape.max_thickness()
    assert value == pytest.approx(0.12, abs=2e-3)
    assert x == pytest.approx(0.30, abs=2e-3)


def test_figure_eight_fails_validate():
    t = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    x = 0.5 + 0.5 * np.sin(t)
    y = 0.2 * np.sin(t) * np.cos(t)
    coords = np.column_stack([x, y])
    shape = AirfoilShape(name="eight", coords=coords)
    with pytest.raises(AirfoilGeometryError):
        shape.validate()


def test_clockwise_fails_validate():
    coords = _naca_symmetric_coords()
    assert AirfoilShape(name="ccw", coords=coords).area() > 0.0
    cw = AirfoilShape(name="cw", coords=coords[::-1].copy())
    assert cw.area() < 0.0
    with pytest.raises(AirfoilGeometryError):
        cw.validate()


def test_te_gap_open_and_closed():
    coords = _naca_symmetric_coords()
    expected_open_gap = float(np.linalg.norm(coords[0] - coords[-1]))
    assert expected_open_gap > 1e-6
    opened = AirfoilShape(name="open", coords=coords)
    assert not opened.is_closed
    assert opened.te_gap == pytest.approx(expected_open_gap, abs=1e-12)
    closed_coords = coords.copy()
    closed_coords[-1] = closed_coords[0]
    closed = AirfoilShape(name="closed", coords=closed_coords)
    assert closed.is_closed
    assert closed.te_gap == pytest.approx(0.0, abs=1e-12)


def test_parse_errors_are_clear():
    with pytest.raises(AirfoilGeometryError, match="one number"):
        parse_airfoil("BAD\n1.0 0.0\n0.5\n0.0 0.0")
    with pytest.raises(AirfoilGeometryError, match="text"):
        parse_airfoil("BAD\n1.0 0.0\nabc def\n0.0 0.0")
    with pytest.raises(AirfoilGeometryError, match="fewer than 3"):
        parse_airfoil("ONLY\n1.0 0.0\n0.5 0.01")
