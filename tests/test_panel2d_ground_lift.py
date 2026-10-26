# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for 2D panel lift in ground effect and in free air.

Checks the lift coefficient Cl against surface pressure integration and
against the lumped-vortex reference with ground images.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ventorum.aero.polar_source_v2d import Ventorum2DPolarSource
from ventorum.airfoil import airfoil
from ventorum.airfoil.naca import naca4
from ventorum.panel2d import vii
from ventorum.panel2d.solver import solve_inviscid
from ventorum.panel2d.viscous import solve_viscous
from ventorum.panel2d.viscous_batch import ViscousCase, solve_viscous_batch
from ventorum.reference.cambered_plate import cambered_plate
from ventorum.reference.karman_trefftz import karman_trefftz_airfoil


def test_free_air_bits_unchanged():
    """In free air Cl keeps the circulation lift and equals Cl_circulation."""
    coords = karman_trefftz_airfoil((-0.07, 0.04), 10.0, n_points=81)
    alpha_scalar = 4.0
    r_scalar = solve_inviscid(coords, alpha_scalar, chord_ref=1.0)
    cl_circ_scalar = 2.0 * float(r_scalar.circulation)
    assert r_scalar.Cl == cl_circ_scalar
    assert r_scalar.Cl_circulation == r_scalar.Cl

    alphas = np.array([-2.0, 0.0, 4.0, 8.0])
    r_vec = solve_inviscid(coords, alphas, chord_ref=1.0)
    cl_circ_vec = 2.0 * np.asarray(r_vec.circulation)
    assert np.array_equal(r_vec.Cl, cl_circ_vec)
    assert np.array_equal(r_vec.Cl_circulation, r_vec.Cl)


def test_ground_effect_cl_equals_cl_pressure():
    """In ground effect Cl equals Cl_pressure for scalar and vector angles."""
    coords = karman_trefftz_airfoil((-0.07, 0.04), 10.0, n_points=81)
    height = 0.3

    r_scalar = solve_inviscid(coords, 4.0, height=height)
    assert r_scalar.Cl == r_scalar.Cl_pressure
    assert r_scalar.Cl != r_scalar.Cl_circulation

    alphas = np.array([0.0, 4.0, 8.0])
    r_vec = solve_inviscid(coords, alphas, height=height)
    assert np.array_equal(r_vec.Cl, r_vec.Cl_pressure)
    assert not np.array_equal(r_vec.Cl, r_vec.Cl_circulation)


def test_ground_effect_torch_backend():
    """Torch backend reports Cl as Cl_pressure in ground effect."""
    torch = pytest.importorskip("torch")
    coords = karman_trefftz_airfoil((-0.07, 0.04), 10.0, n_points=81)
    alphas = np.array([0.0, 4.0])
    r_torch = solve_inviscid(coords, alphas, height=0.4, backend="torch", dtype=torch.float64)
    assert torch.equal(r_torch.Cl, r_torch.Cl_pressure)
    r_np = solve_inviscid(coords, alphas, height=0.4, backend="numpy")
    assert np.allclose(r_torch.Cl.numpy(), r_np.Cl, atol=1e-10, rtol=0.0)
    assert np.allclose(r_torch.Cl_circulation.numpy(), r_np.Cl_circulation, atol=1e-10, rtol=0.0)


def test_cl_circulation_keeps_circulation_lift():
    """Cl_circulation keeps 2 Gamma / c in free air and in ground effect."""
    coords = naca4("0012", n_points=101, closed_te=True)
    chord = 1.0

    for h in [None, 0.5, 0.2]:
        r = solve_inviscid(coords, 3.0, height=h)
        expected_circ_cl = 2.0 * float(r.circulation) / chord
        assert abs(r.Cl_circulation - expected_circ_cl) < 1e-14


def test_thin_flat_plate_matches_discrete_vortex_reference():
    """A thin flat plate at h/c 0.2 matches the 2D discrete-vortex reference within 1 %."""
    alpha_deg = 4.0
    alpha_rad = np.radians(alpha_deg)
    h_c = 0.2

    zero_line = lambda x: 0.0 * np.asarray(x)  # noqa: E731
    ref = cambered_plate(zero_line, zero_line, alpha=alpha_rad, h=h_c, n_panels=512)
    ref_cl = ref["CL"]

    plate_coords = naca4("0001", n_points=241, closed_te=True)
    res = solve_inviscid(plate_coords, alpha_deg, height=h_c)

    rel_diff = abs(res.Cl - ref_cl) / ref_cl
    assert rel_diff < 0.01

    circ_diff = abs(res.Cl_circulation - ref_cl) / ref_cl
    assert circ_diff > 0.08


@pytest.fixture
def viscous_loads(monkeypatch):
    """Record the circulation and pressure loads before result assembly."""
    outputs = vii.case_outputs
    loads = {}

    def record(setup, state, ue):
        result = outputs(setup, state, ue)
        loads[(setup.sys.alpha, setup.sys.nodes.tobytes())] = result
        return result

    monkeypatch.setattr(vii, "case_outputs", record)
    return loads


def _check_viscous_lift(result, height, loads):
    """Check the reported lift against the unchanged internal load outputs."""
    assert np.all(result.converged)
    ground = height is not None
    coordinates = [result.coordinates] if np.ndim(result.alpha) == 0 else result.coordinates
    expected = [loads[(a, nodes.tobytes())]
                for a, nodes in zip(np.atleast_1d(result.alpha), coordinates)]
    circulation = np.array([o["Cl"] for o in expected])
    pressure = np.array([o["Cl_pressure"] for o in expected])
    assert np.array_equal(np.atleast_1d(result.Cl_circulation), circulation)
    assert np.array_equal(np.atleast_1d(result.Cl_pressure), pressure)
    assert np.array_equal(np.atleast_1d(result.Cl), pressure if ground else circulation)
    assert not np.array_equal(circulation, pressure)


@pytest.mark.slow
@pytest.mark.parametrize("height", [None, 0.3, 0.2])
@pytest.mark.parametrize("alphas", [4.0, [4.0, 5.0]], ids=["scalar", "sweep"])
def test_viscous_lift_field_choice(height, alphas, viscous_loads):
    """Keep circulation lift exactly, and report pressure lift near the ground."""
    shape = airfoil("4412", n_points=161)
    result = solve_viscous(shape, alphas, 1e6, height=height)
    _check_viscous_lift(result, height, viscous_loads)


@pytest.mark.slow
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_viscous_batch_lift_field_choice(device, viscous_loads):
    """Both batch paths use the same lift rule, including a warm start."""
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    shape = airfoil("4412", n_points=161)
    cases = [ViscousCase(shape, a, 1e6, height=h)
             for h in (None, 0.3, 0.2) for a in (4.0, 5.0)]
    results = solve_viscous_batch(cases, device=device, dtype=torch.float64)
    for case, result in zip(cases, results):
        _check_viscous_lift(result, case.height, viscous_loads)


@pytest.mark.parametrize("mode", ["scalar", "sweep", "batch"])
def test_failed_viscous_circulation_is_nan(mode):
    """Failed results keep all lift fields as NaN on every assembly path."""
    shape = airfoil("4412", n_points=81)
    if mode == "batch":
        result = solve_viscous_batch(
            [ViscousCase(shape, 4.0, 1e6, height=0.3, work_budget=1)], device="cpu"
        )[0]
    else:
        with pytest.warns(UserWarning, match="did not converge"):
            result = solve_viscous(shape, 4.0 if mode == "scalar" else [4.0],
                                   1e6, height=0.3, work_budget=1)
    assert not np.any(result.converged)
    assert np.all(np.isnan(result.Cl))
    assert np.all(np.isnan(result.Cl_pressure))
    assert np.all(np.isnan(result.Cl_circulation))


@pytest.mark.parametrize("height", [None, 0.3, 0.2])
def test_polar_source_uses_reported_viscous_lift(monkeypatch, height):
    """The polar source reads Cl from the solver for every height."""
    def solve(*args, **kwargs):
        return SimpleNamespace(alpha=np.array([4.0]), Cl=np.array([0.5]),
                               Cl_pressure=np.array([0.4]), Cd=np.array([0.01]),
                               Cm=np.array([0.0]), converged=np.array([True]))

    monkeypatch.setattr("ventorum.aero.polar_source_v2d.solve_viscous", solve)
    polar = Ventorum2DPolarSource(cache=False).polar("4412", 1e6, [4.0], height=height)
    assert np.array_equal(polar.Cl_data, [0.5])
