# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the airfoil shape type and the Selig and Lednicer readers and writers."""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.airfoil.geometry import AirfoilGeometryError, AirfoilShape
from ventorum.airfoil.io import parse_airfoil, read_airfoil, write_lednicer, write_selig


def _naca_symmetric_coords(thickness_ratio: float = 0.12, n_each: int = 101) -> np.ndarray:
    """Symmetric NACA 4-digit section in Selig order (open trailing edge).

    The last thickness coefficient -0.1015 leaves a finite trailing-edge
    thickness, so the first and the last point differ.
    """
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
    """Ellipse section of thickness ratio 0.12 in Selig order (closed trailing edge).

    The eccentric angle runs from 0 at the trailing edge (1, 0) over the
    upper surface to the leading edge (0, 0) at pi and back over the lower
    surface. The last point repeats the first point exactly.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n + 1)
    x = 0.5 * (1.0 + np.cos(theta))
    y = 0.06 * np.sin(theta)
    selig = np.column_stack([x, y])
    selig[-1] = selig[0]
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


def test_closed_ellipse_round_trip_and_checks(tmp_path):
    coords = _ellipse_coords()
    shape = AirfoilShape(name="ELLIPSE", coords=coords)
    assert shape.is_closed
    assert shape.te_gap == 0.0
    assert shape.problems() == []
    shape.validate()
    assert shape.area() > 0.0
    path = tmp_path / "ellipse.dat"
    write_selig(shape, path, precision=9)
    back = read_airfoil(path)
    assert back.name == "ELLIPSE"
    assert back.is_closed
    assert np.allclose(back.coords, coords, atol=1e-9)


def test_lednicer_file_without_name_line(tmp_path):
    """A Lednicer file without the name line starts with its counts line.

    Before the fix, the counts line became the data point (n_up, n_low).
    """
    coords = _naca_symmetric_coords(n_each=31)
    shape = AirfoilShape(name="NACA0012", coords=coords)
    path = tmp_path / "with_name.dat"
    write_lednicer(shape, path, precision=8)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[1].split() == ["31", "31"]
    no_name = "\n".join(lines[1:]) + "\n"
    parsed = parse_airfoil(no_name)
    assert parsed.n_points == shape.n_points
    assert np.allclose(parsed.coords, coords, atol=1e-8)
    assert parsed.coords.max() <= 1.0 + 1e-12
    named = parse_airfoil(no_name, name="given")
    assert named.name == "given"
    nameless_path = tmp_path / "nameless_lednicer.dat"
    nameless_path.write_text(no_name, encoding="utf-8")
    from_file = read_airfoil(nameless_path)
    assert from_file.name == "nameless_lednicer"
    assert np.allclose(from_file.coords, coords, atol=1e-8)


def test_first_point_like_counts_stays_data():
    """Two whole numbers that do not count the following lines stay a point."""
    pts = [(6.0, 6.0), (3.0, 1.0), (0.0, 0.0), (3.0, -1.0), (6.0, -6.0)]
    text = "\n".join(f"{x} {y}" for x, y in pts)
    parsed = parse_airfoil(text)
    assert parsed.n_points == 5
    assert np.allclose(parsed.coords[0], [6.0, 6.0])


def test_problems_reports_few_points():
    coords = _naca_symmetric_coords(n_each=4)
    shape = AirfoilShape(name="few", coords=coords)
    assert shape.n_points == 7
    found = shape.problems()
    assert any("fewer than 10 points" in item for item in found)
    with pytest.raises(AirfoilGeometryError, match="fewer than 10 points"):
        shape.validate()


def test_problems_reports_repeated_points():
    coords = _naca_symmetric_coords(n_each=21)
    repeated = np.vstack([coords[:5], coords[4:]])
    shape = AirfoilShape(name="repeat", coords=repeated)
    found = shape.problems()
    assert any("repeated consecutive points at index 4 and 5" in item for item in found)
    with pytest.raises(AirfoilGeometryError, match="repeated"):
        shape.validate()


def test_problems_non_monotonic_note_is_not_an_error():
    coords = _naca_symmetric_coords(n_each=41)
    wavy = coords.copy()
    # Move one upper-surface point aft of its aft neighbour, so x decreases
    # once from the leading edge to the trailing edge.
    k = 10
    wavy[k, 0] = wavy[k - 1, 0] + 0.25 * (wavy[k - 2, 0] - wavy[k - 1, 0])
    shape = AirfoilShape(name="wavy", coords=wavy)
    found = shape.problems()
    assert found == ["note: non-monotonic x on the upper surface"]
    shape.validate()
