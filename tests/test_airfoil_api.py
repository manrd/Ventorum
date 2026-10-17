# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the unified airfoil interface (shape methods and factory)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ventorum.airfoil import AirfoilShape, airfoil
from ventorum.airfoil.flap import deflect_plain_flap
from ventorum.airfoil.io import read_airfoil
from ventorum.airfoil.naca import naca
from ventorum.airfoil.paneling import repanel, repanel_cosine

NAMES_4 = ["0012", "2412"]
NAMES_5 = ["23012"]
ALL_NAMES = NAMES_4 + NAMES_5
DELTAS = [10.0, 25.0]


def test_factory_naca_string():
    for designation in ("2412", "NACA 2412", "naca2412", "NACA2412"):
        shape = airfoil(designation)
        assert isinstance(shape, AirfoilShape)
        assert np.array_equal(shape.coords, naca(designation))
        assert shape.area() > 0.0


def test_factory_naca_int():
    shape = airfoil(2412)
    assert isinstance(shape, AirfoilShape)
    assert np.array_equal(shape.coords, naca("2412"))


def test_factory_five_digit():
    shape = airfoil("23012")
    assert isinstance(shape, AirfoilShape)
    assert np.array_equal(shape.coords, naca("23012"))


def test_factory_array():
    coords = naca("2412", n_points=61)
    shape = airfoil(coords)
    assert isinstance(shape, AirfoilShape)
    assert np.array_equal(shape.coords, coords)
    assert shape.coords is not coords


def test_factory_shape_returns_copy():
    first = AirfoilShape.naca("2412", n_points=61)
    second = airfoil(first)
    assert isinstance(second, AirfoilShape)
    assert second is not first
    assert np.array_equal(second.coords, first.coords)
    assert second.name == first.name


def test_factory_file_path(tmp_path):
    shape = AirfoilShape.naca("2412", n_points=61)
    path = tmp_path / "naca2412.dat"
    shape.to_file(path, fmt="selig")
    from_str = airfoil(str(path))
    from_path = airfoil(Path(path))
    assert np.allclose(from_str.coords, shape.coords, atol=1e-6, rtol=0.0)
    assert np.allclose(from_path.coords, shape.coords, atol=1e-6, rtol=0.0)


def test_factory_invalid_raises():
    with pytest.raises(ValueError):
        airfoil("not an airfoil designation 999")


def test_shape_naca_matches_array_function():
    for designation in ALL_NAMES:
        shape = AirfoilShape.naca(designation, n_points=61)
        assert np.array_equal(shape.coords, naca(designation, n_points=61))
    shape = AirfoilShape.naca("2412", n_points=51, spacing="uniform", closed_te=True)
    assert np.array_equal(
        shape.coords, naca("2412", n_points=51, spacing="uniform", closed_te=True)
    )


def test_naca_2412_file_roundtrip(tmp_path):
    shape = AirfoilShape.naca("2412")
    selig_path = tmp_path / "naca2412_selig.dat"
    lednicer_path = tmp_path / "naca2412_led.dat"
    shape.to_file(selig_path, fmt="selig")
    shape.to_file(lednicer_path, fmt="lednicer")
    from_selig = read_airfoil(selig_path)
    from_lednicer = read_airfoil(lednicer_path)
    assert from_selig.coords.shape == shape.coords.shape
    assert from_lednicer.coords.shape == shape.coords.shape
    assert np.allclose(from_selig.coords, shape.coords, atol=1e-6, rtol=0.0)
    assert np.allclose(from_lednicer.coords, shape.coords, atol=1e-6, rtol=0.0)
    assert np.allclose(from_selig.coords, from_lednicer.coords, atol=1e-6, rtol=0.0)


def test_from_file_matches_read_airfoil(tmp_path):
    shape = AirfoilShape.naca("0012", n_points=61)
    path = tmp_path / "naca0012.dat"
    shape.to_file(path)
    back = AirfoilShape.from_file(path)
    direct = read_airfoil(path)
    assert np.array_equal(back.coords, direct.coords)
    assert back.name == direct.name


@pytest.mark.parametrize("designation", ALL_NAMES)
@pytest.mark.parametrize("delta", DELTAS)
def test_repanel_then_flap_then_validate(designation, delta):
    shape = AirfoilShape.naca(designation)
    repaneled = shape.repanel()
    assert repaneled.coords.shape == (161, 2)
    deflected = repaneled.with_flap(0.7, delta)
    deflected.validate()
    assert deflected.area() > 0.0


def test_shape_repanel_matches_array_function():
    shape = AirfoilShape.naca("2412", n_points=61)
    assert np.array_equal(shape.repanel(n_panels=40).coords, repanel(shape.coords, n_panels=40))


def test_shape_repanel_cosine_matches_array_function():
    shape = AirfoilShape.naca("2412", n_points=61)
    assert np.array_equal(
        shape.repanel_cosine(20).coords, repanel_cosine(shape.coords, 20)
    )


def test_shape_with_flap_matches_array_function():
    shape = AirfoilShape.naca("2412", n_points=61).repanel(n_panels=40)
    assert np.array_equal(
        shape.with_flap(0.7, 10.0).coords,
        deflect_plain_flap(shape.coords, 0.7, 10.0),
    )
    assert np.array_equal(
        shape.with_flap(0.7, 10.0, 0.5).coords,
        shape.with_flap(0.7, 10.0).coords,
    )


@pytest.mark.parametrize("treatment", ["none", "repanel", "merge"])
def test_shape_with_flap_passes_the_hinge_treatment(treatment):
    shape = AirfoilShape.naca("2412", n_points=61).repanel(n_panels=80)
    assert np.array_equal(
        shape.with_flap(0.75, 5.0, hinge_treatment=treatment).coords,
        deflect_plain_flap(shape.coords, 0.75, 5.0, hinge_treatment=treatment),
    )
    assert np.array_equal(
        shape.with_flap(0.75, 5.0, hinge_treatment="none").coords,
        shape.with_flap(0.75, 5.0).coords,
    )


def test_shape_with_flap_repanel_count():
    shape = AirfoilShape.naca("2412", n_points=61).repanel(n_panels=80)
    out = shape.with_flap(0.75, 5.0, hinge_treatment="repanel", n_panels=100)
    assert out.coords.shape == (101, 2)
    assert np.array_equal(
        out.coords,
        deflect_plain_flap(shape.coords, 0.75, 5.0, hinge_treatment="repanel", n_panels=100),
    )
    out.validate()
    with pytest.raises(ValueError, match="n_panels"):
        shape.with_flap(0.75, 5.0, n_panels=100)


def test_to_file_matches_writers(tmp_path):
    from ventorum.airfoil.io import write_lednicer, write_selig

    shape = AirfoilShape.naca("2412", n_points=61)
    selig_a = tmp_path / "a.dat"
    selig_b = tmp_path / "b.dat"
    shape.to_file(selig_a, fmt="selig")
    write_selig(shape, selig_b)
    assert selig_a.read_text(encoding="utf-8") == selig_b.read_text(encoding="utf-8")
    led_a = tmp_path / "c.dat"
    led_b = tmp_path / "d.dat"
    shape.to_file(led_a, fmt="lednicer")
    write_lednicer(shape, led_b)
    assert led_a.read_text(encoding="utf-8") == led_b.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        shape.to_file(tmp_path / "e.dat", fmt="plain")


def test_leading_edge_definitions_agree_on_clean_shapes():
    for designation in ALL_NAMES:
        shape = AirfoilShape.naca(designation).repanel()
        farthest = shape.leading_edge_index
        min_x = int(np.argmin(shape.coords[:, 0]))
        assert farthest == min_x


def test_factory_naca_designation_wins_over_a_file_of_that_name(tmp_path, monkeypatch):
    """A file named "2412" in the current folder does not shadow NACA 2412.

    Before the fix, airfoil("2412") read the file first.
    """
    monkeypatch.chdir(tmp_path)
    other = AirfoilShape.naca("0012", n_points=31)
    other.to_file(tmp_path / "2412")
    other.to_file(tmp_path / "NACA 2412")
    for spec in ("2412", "NACA 2412"):
        shape = airfoil(spec)
        assert np.array_equal(shape.coords, naca("2412"))
    # A Path object always names a file, also without an extension.
    from_path = airfoil(Path("2412"))
    assert np.allclose(from_path.coords, other.coords, atol=1e-6, rtol=0.0)


def test_factory_string_file_needs_an_extension(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shape = AirfoilShape.naca("2412", n_points=61)
    shape.to_file(tmp_path / "section")
    shape.to_file(tmp_path / "section.dat")
    with pytest.raises(ValueError, match="extension"):
        airfoil("section")
    assert np.allclose(airfoil("section.dat").coords, shape.coords, atol=1e-6, rtol=0.0)
    with pytest.raises(ValueError, match="extension"):
        airfoil("missing.dat")
    with pytest.raises(ValueError, match="not found"):
        airfoil(Path("missing.dat"))


@pytest.mark.parametrize("keyword", [{"n_points": 61}, {"spacing": "cosine"}, {"closed_te": True}])
def test_factory_rejects_naca_keywords_for_other_inputs(tmp_path, keyword):
    """Generator keywords do not apply to arrays, files and shapes.

    Before the fix, the factory ignored them without a signal.
    """
    shape = AirfoilShape.naca("2412", n_points=61)
    path = tmp_path / "naca2412.dat"
    shape.to_file(path)
    for spec in (shape.coords, shape, path, str(path)):
        with pytest.raises(TypeError, match="do not apply"):
            airfoil(spec, **keyword)


def test_factory_rejects_unknown_keywords_and_keeps_name():
    with pytest.raises(TypeError, match="do not apply"):
        airfoil("2412", n_panels=40)
    shape = AirfoilShape.naca("2412", n_points=61)
    assert airfoil(shape, name="copy").name == "copy"
    assert airfoil(shape.coords, name="points").name == "points"
    assert airfoil("2412", n_points=61, name="wing root").name == "wing root"
    assert np.array_equal(airfoil("2412", n_points=61, spacing="uniform").coords,
                          naca("2412", n_points=61, spacing="uniform"))
