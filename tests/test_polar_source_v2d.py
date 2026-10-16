# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the Ventorum-2D section polar source.

The sweep tests run the viscous panel solver on NACA 0012 and are marked
slow. The small tests check the protocol, the cache and the input
validation without long solves.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

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


def _fake_viscous_result(alpha_deg, height=None):
    """Canned solver result: thin-airfoil lift, constant drag, all converged."""
    alphas = np.atleast_1d(np.asarray(alpha_deg, dtype=float))
    shift = 0.0 if height is None else 0.01 / float(height)
    return SimpleNamespace(
        alpha=alphas,
        Cl=2.0 * np.pi * np.radians(alphas) + shift,
        Cd=np.full(alphas.shape, 0.008),
        Cm=np.zeros(alphas.shape),
        converged=np.ones(alphas.shape, dtype=bool),
    )


@pytest.fixture
def fake_solver(monkeypatch):
    """Replace the 2D solver by a recorder, so the tests need no solve."""
    import ventorum.aero.polar_source_v2d as module

    calls: list[dict] = []

    def fake_solve(bodies, alpha_deg, Re, **kwargs):
        calls.append({"bodies": np.asarray(bodies), "alpha_deg": np.asarray(alpha_deg),
                      "Re": Re, **kwargs})
        return _fake_viscous_result(alpha_deg, kwargs.get("height"))

    monkeypatch.setattr(module, "solve_viscous", fake_solve)
    return calls


def test_protocol_keyword_call_works(fake_solver):
    """Regression: polar(*args) refused section=, Re= and alphas_deg= keywords."""
    source = Ventorum2DPolarSource(cache=False)
    polar = source.polar(section="naca 0012", Re=1e6, alphas_deg=[0.0, 2.0])
    np.testing.assert_allclose(np.degrees(polar.alpha), [0.0, 2.0])
    params = inspect.signature(source.polar).parameters
    assert list(params)[:3] == ["section", "Re", "alphas_deg"]
    for name in ("mach", "flap", "height"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_same_call_uses_the_cache(tmp_path, fake_solver):
    source = Ventorum2DPolarSource(cache=PolarCache(tmp_path), n_panels=80)
    first = source.polar("naca 0012", 1e6, [0.0, 4.0])
    assert len(fake_solver) == 1
    second = source.polar("naca 0012", 1e6, [0.0, 4.0])
    assert len(fake_solver) == 1
    np.testing.assert_array_equal(second.alpha, first.alpha)
    np.testing.assert_array_equal(second.Cl_data, first.Cl_data)
    np.testing.assert_array_equal(second.Cd_data, first.Cd_data)
    # The height is part of the key, so another height solves again.
    third = source.polar("naca 0012", 1e6, [0.0, 4.0], height=50.0)
    assert len(fake_solver) == 2
    assert not np.array_equal(third.Cl_data, first.Cl_data)
    # The options are part of the key too.
    source.polar("naca 0012", 1e6, [0.0, 4.0], ncrit=7.0)
    assert len(fake_solver) == 3
    source.polar("naca 0012", 1e6, [0.0, 4.0], flap=(0.75, 0.5, 10.0))
    assert len(fake_solver) == 4
    assert len(list(tmp_path.glob("*.json"))) == 4


def test_cache_key_holds_package_version_and_solver_revision(tmp_path, fake_solver, monkeypatch):
    """Regression: the key had no solver version, so a new solver read old polars."""
    import ventorum
    import ventorum.aero.polar_source_v2d as module
    import ventorum.panel2d as panel2d

    source = Ventorum2DPolarSource(cache=PolarCache(tmp_path))
    source.polar("naca 0012", 1e6, [0.0, 2.0])
    source.polar("naca 0012", 1e6, [0.0, 2.0])
    assert len(fake_solver) == 1
    monkeypatch.setattr(panel2d, "SOLVER_REVISION", "next")
    source.polar("naca 0012", 1e6, [0.0, 2.0])
    assert len(fake_solver) == 2
    monkeypatch.setattr(ventorum, "__version__", "99.0.0")
    source.polar("naca 0012", 1e6, [0.0, 2.0])
    assert len(fake_solver) == 3
    monkeypatch.setattr(module, "solver_fingerprint", lambda: "another solver source")
    source.polar("naca 0012", 1e6, [0.0, 2.0])
    assert len(fake_solver) == 4
    info = source.describe()
    assert info["source"] == "Ventorum2DPolarSource"
    assert "99.0.0" in info["version"] and "next" in info["version"]
    assert info["options"]["n_panels"] == 160 and info["options"]["ncrit"] == 9.0
    assert len(module.solver_fingerprint()) > 0


def test_height_and_flap_pass_through(fake_solver):
    source = Ventorum2DPolarSource(cache=False, n_panels=100)
    source.polar("naca 0012", 1e6, [2.0], height=0.4, flap=(0.7, 0.25, 8.0))
    call = fake_solver[-1]
    assert call["height"] == pytest.approx(0.4)
    # The flapped geometry goes to the solver: the trailing edge moved down.
    te_y = 0.5 * (call["bodies"][0, 1] + call["bodies"][-1, 1])
    # 0.3 chord of flap turned 8 deg down: about -0.042 chord.
    assert te_y < -0.03
    source.polar("naca 0012", 1e6, [2.0])
    assert fake_solver[-1]["height"] is None


def test_flap_tuple_is_the_common_three_tuple(fake_solver):
    """Owner decision: (x_hinge, y_hinge, deflection_deg) only, no 2-tuple alias."""
    source = Ventorum2DPolarSource(cache=False)
    with pytest.raises(ValueError, match="3 numbers"):
        source.polar("naca 0012", 1e6, [0.0], flap=(0.75, 10.0))
    with pytest.raises(ValueError, match="thickness"):
        source.polar("naca 0012", 1e6, [0.0], flap=(0.75, 2.0, 10.0))
    with pytest.raises(ValueError, match="Unknown Ventorum-2D option"):
        source.polar("naca 0012", 1e6, [0.0], bogus=1)
    assert fake_solver == []


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
    near = source.polar("naca 0012", 1e6, [2.0], height=0.2)
    assert "h=free air" in free.name
    assert "h=0.2" in near.name
    assert np.isfinite(near.Cl_data[0])
    assert float(near.Cd_data[0]) > 0.0


#: Largest |Cl(h/c = 50) - Cl(free air)| [-] at alpha 2 deg. The image of
#: the bound vortex lowers the onset speed by Gamma / (4 pi h), which
#: changes Cl by about Cl^2 c / (4 pi h) = 8e-5 here (the inviscid 2D
#: solver gives 4e-5). The bound adds the coupling tolerance of the
#: viscous solve with margin.
HEIGHT_50_CL_TOLERANCE = 2e-3


@pytest.mark.slow
def test_far_ground_matches_free_air():
    """At h/c 50 the polar is the free-air polar within a stated tolerance.

    The former check (h/c 1000 equal to free air) passed only because the
    solver treated heights of 500 chords or more as free air.
    """
    source = Ventorum2DPolarSource(cache=False)
    free = source.polar("naca 0012", 1e6, [2.0])
    far = source.polar("naca 0012", 1e6, [2.0], height=50.0)
    assert "h=50" in far.name
    assert abs(float(far.Cl_data[0]) - float(free.Cl_data[0])) < HEIGHT_50_CL_TOLERANCE


def test_input_validation():
    source = Ventorum2DPolarSource(cache=False)
    with pytest.raises(ValueError, match="mach"):
        source.polar("naca 0012", 1e6, [0.0], mach=0.2)
    with pytest.raises(ValueError, match="Re"):
        source.polar("naca 0012", 0.0, [0.0])
    with pytest.raises(ValueError, match="alphas_deg"):
        source.polar("naca 0012", 1e6, [])
    # The protocol signature names Re: a second Re is refused by the call.
    with pytest.raises(TypeError, match="multiple values"):
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


def test_coordinate_file_section_is_keyed_by_its_points(tmp_path, fake_solver):
    """Regression: a file path got generator options (TypeError) and a key by name.

    A coordinate file must work as a section, and the cache key must use the
    points, so that an edited file with the same name solves again.
    """
    from ventorum.airfoil import airfoil
    from ventorum.airfoil.io import write_selig

    path = tmp_path / "section.dat"
    write_selig(airfoil("0012"), path)
    source = Ventorum2DPolarSource(cache=PolarCache(tmp_path / "cache"), n_panels=80)
    source.polar(str(path), 1e6, [0.0, 2.0])
    source.polar(str(path), 1e6, [0.0, 2.0])
    assert len(fake_solver) == 1
    write_selig(airfoil("2412"), path)
    source.polar(str(path), 1e6, [0.0, 2.0])
    assert len(fake_solver) == 2
