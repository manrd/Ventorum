# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the Ventorum-2D section polar source.

The sweep tests run the viscous panel solver on NACA 0012 and are marked
slow. The small tests check the protocol, the cache and the input
validation without long solves.
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.aero.polar_cache import PolarCache
from ventorum.aero.polar_source import SectionPolarSource
from ventorum.aero.polar_source_v2d import Ventorum2DPolarSource
from ventorum.airfoil import airfoil

ALPHAS_MINUS4_TO_8 = np.arange(-4.0, 8.1, 2.0)


def test_source_implements_the_polar_protocol():
    source = Ventorum2DPolarSource(cache=False)
    assert isinstance(source, SectionPolarSource)


@pytest.mark.slow
def test_naca0012_re1e6_sweep_minus4_to_8():
    source = Ventorum2DPolarSource(cache=False)
    polar = source.polar("naca 0012", 1e6, ALPHAS_MINUS4_TO_8)
    deg = np.degrees(np.asarray(polar.alpha))
    assert deg.size == 7
    np.testing.assert_allclose(deg, ALPHAS_MINUS4_TO_8, atol=1e-12)
    assert "0 of 7 alpha dropped" in polar.name
    assert "h=free air" in polar.name
    zero = int(np.flatnonzero(np.isclose(deg, 0.0))[0])
    assert abs(float(polar.Cl_data[zero])) < 1e-6
    assert np.all(np.asarray(polar.Cd_data) > 0.0)
    assert float(polar.Cl_data[-1]) > 0.0 > float(polar.Cl_data[0])


def test_same_call_uses_the_cache(tmp_path, monkeypatch):
    import ventorum.aero.polar_source_v2d as module

    real = module.solve_viscous
    calls = {"count": 0}

    def counting_solve(*args, **kwargs):
        calls["count"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "solve_viscous", counting_solve)
    source = Ventorum2DPolarSource(cache=PolarCache(tmp_path), n_panels=80)
    first = source.polar("naca 0012", 1e6, [0.0, 4.0])
    assert calls["count"] == 1
    second = source.polar("naca 0012", 1e6, [0.0, 4.0])
    assert calls["count"] == 1
    np.testing.assert_array_equal(second.alpha, first.alpha)
    np.testing.assert_array_equal(second.Cl_data, first.Cl_data)
    np.testing.assert_array_equal(second.Cd_data, first.Cd_data)
    # The height is part of the key, so another height solves again.
    third = source.polar("naca 0012", 1e6, [0.0, 4.0], height=1000.0)
    assert calls["count"] == 2
    np.testing.assert_allclose(third.Cl_data, first.Cl_data, atol=1e-12)
    # The options are part of the key too.
    fourth = source.polar("naca 0012", 1e6, [0.0, 4.0], ncrit=7.0)
    assert fourth.Cl_data is not None
    assert calls["count"] == 3
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_nonconverged_points_are_absent():
    source = Ventorum2DPolarSource(cache=False)
    polar = source.polar("naca 0012", 1e6, [0.0, 4.0], max_iter=1)
    assert len(polar.alpha) == 0
    assert "2 of 2 alpha dropped" in polar.name


@pytest.mark.slow
def test_flap_deflection_adds_lift():
    source = Ventorum2DPolarSource(cache=False)
    clean = source.polar("naca 0012", 1e6, [0.0])
    flapped = source.polar("naca 0012", 1e6, [0.0], flap=(0.75, 0.5, 10.0))
    assert abs(float(clean.Cl_data[0])) < 1e-6
    assert float(flapped.Cl_data[0]) > 0.4
    assert float(flapped.Cd_data[0]) > 0.0


@pytest.mark.slow
def test_section_kinds_agree():
    shape = airfoil("naca 0012")
    source = Ventorum2DPolarSource(cache=False)
    from_name = source.polar("naca 0012", 1e6, [2.0])
    from_array = source.polar(shape.coords, 1e6, [2.0])
    from_shape = source.polar(shape, 1e6, [2.0])
    np.testing.assert_array_equal(from_array.Cl_data, from_name.Cl_data)
    np.testing.assert_array_equal(from_array.Cd_data, from_name.Cd_data)
    np.testing.assert_array_equal(from_shape.Cl_data, from_name.Cl_data)
    np.testing.assert_array_equal(from_shape.Cd_data, from_name.Cd_data)


@pytest.mark.slow
def test_height_polars():
    source = Ventorum2DPolarSource(cache=False)
    free = source.polar("naca 0012", 1e6, [2.0])
    far = source.polar("naca 0012", 1e6, [2.0], height=1000.0)
    near = source.polar("naca 0012", 1e6, [2.0], height=0.2)
    # The solver treats heights of 500 chord or more as free air.
    np.testing.assert_array_equal(far.Cl_data, free.Cl_data)
    assert "h=free air" in free.name
    assert "h=1000" in far.name
    assert "h=0.2" in near.name
    assert np.isfinite(near.Cl_data[0])
    assert float(near.Cd_data[0]) > 0.0


def test_input_validation():
    source = Ventorum2DPolarSource(cache=False)
    with pytest.raises(ValueError, match="mach"):
        source.polar("naca 0012", 1e6, [0.0], mach=0.2)
    with pytest.raises(ValueError, match="Re"):
        source.polar("naca 0012", 0.0, [0.0])
    with pytest.raises(ValueError, match="alphas_deg"):
        source.polar("naca 0012", 1e6, [])
    with pytest.raises(ValueError, match="clash"):
        source.polar("naca 0012", 1e6, [0.0], Re=2e6)
    with pytest.raises(ValueError, match="height"):
        source.polar("naca 0012", 1e6, [0.0], height=0.0)
    with pytest.raises(ValueError, match="x_hinge"):
        source.polar("naca 0012", 1e6, [0.0], flap=(0.01, 0.5, 10.0))
    with pytest.raises(TypeError, match="cache"):
        Ventorum2DPolarSource(cache="yes")
    with pytest.raises(ValueError, match="n_panels"):
        Ventorum2DPolarSource(cache=False, n_panels=1)
    with pytest.raises(ValueError, match="ncrit"):
        Ventorum2DPolarSource(cache=False, ncrit=0.0)
    with pytest.raises(ValueError, match="xtr"):
        Ventorum2DPolarSource(cache=False, xtr=(1.5, 1.0))
    with pytest.raises(ValueError, match="backend"):
        Ventorum2DPolarSource(cache=False, backend="torch")
