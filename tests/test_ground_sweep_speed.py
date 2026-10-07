# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Parity and correctness tests for ground sweeps and Trefftz loads on CPU."""

from __future__ import annotations

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero import loads as loads_module
from ventorum.geometry import lattice_cache
from ventorum.ground_effect import solver as ge_solver
from ventorum.ground_effect import sweep as ge_sweep
from ventorum.ground_effect.sweep import GroundEffectSweep
from ventorum.solvers.factory import make_solver


@pytest.fixture
def wing_tail_aircraft():
    """Create a wing and tail geometry for ground sweep testing."""
    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    tail = vt.LiftingSurface(
        name="Tail",
        semi_span=1.5,
        position=np.array([3.5, 0.0, 0.5]),
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.6),
            vt.WingSection(y_frac=1.0, chord=0.4),
        ],
    )
    return vt.Aircraft(name="WingTail", surfaces=[wing, tail])


@pytest.fixture
def rectangular_wing():
    """Create a single rectangular wing."""
    return vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )


@pytest.mark.parametrize("solver_type", ["vlm", "linear", "nonlinear"])
def test_cpu_batch_sweep_equals_case_by_case(wing_tail_aircraft, solver_type, monkeypatch):
    """CPU batch sweep gives the same rows as the case-by-case path."""
    sett = vt.SolverSettings(solver_type=solver_type, n_panels=16)
    sweep = GroundEffectSweep(wing_tail_aircraft, settings=sett, backend="serial")

    # Grid: 3 heights above h_min/c >= 1.0, plus 1 height where some cases strike.
    heights = [0.1, 1.5, 2.0, 3.0]
    alphas = [0.0, 2.0, 4.0, 6.0]
    phis = [0.0, 3.0]

    # Run batch sweep on CPU
    res_batch = sweep.run_sweep(heights, alphas, phis)

    # Force the case-by-case path by returning None from _cpu_batch
    monkeypatch.setattr(ge_sweep, "_cpu_batch", lambda *args, **kwargs: None)
    res_serial = sweep.run_sweep(heights, alphas, phis)

    # Verify all grid quantities match within relative 1e-12
    grid_keys = (
        "CL", "CDi", "CD", "CY", "Cl", "Cm", "Cn",
        "L_over_D", "e", "h_min", "h_ref_grid", "phi_strike_limit",
    )
    for k in grid_keys:
        val_batch = getattr(res_batch, k)
        val_serial = getattr(res_serial, k)
        np.testing.assert_allclose(
            val_batch, val_serial, rtol=1e-12, atol=1e-12, equal_nan=True,
            err_msg=f"Mismatch in grid {k} for solver {solver_type}",
        )

    np.testing.assert_array_equal(res_batch.is_strike, res_serial.is_strike)
    np.testing.assert_array_equal(res_batch.refused, res_serial.refused)
    assert res_batch.errors == res_serial.errors


def test_cpu_batch_chunks_give_the_same_rows(wing_tail_aircraft, monkeypatch):
    """Chunk limit of 1 gives the same rows as a single chunk."""
    sett = vt.SolverSettings(solver_type="vlm", n_panels=16)
    sweep = GroundEffectSweep(wing_tail_aircraft, settings=sett, backend="serial")

    heights = [1.5, 2.0]
    alphas = [0.0, 4.0]
    phis = [0.0, 3.0]

    res_default = sweep.run_sweep(heights, alphas, phis)

    # Force 1 case per chunk
    monkeypatch.setattr(ge_sweep, "_chunk_size", lambda canonical, n_unknowns: 1)
    res_chunked = sweep.run_sweep(heights, alphas, phis)

    grid_keys = ("CL", "CDi", "CD", "CY", "Cl", "Cm", "Cn", "L_over_D", "e", "h_min", "h_ref_grid")
    for k in grid_keys:
        np.testing.assert_allclose(
            getattr(res_chunked, k), getattr(res_default, k),
            rtol=1e-12, atol=1e-12, equal_nan=True,
        )
    np.testing.assert_array_equal(res_chunked.is_strike, res_default.is_strike)
    np.testing.assert_array_equal(res_chunked.refused, res_default.refused)
    assert res_chunked.errors == res_default.errors


def test_probe_lattice_is_built_once(wing_tail_aircraft, monkeypatch):
    """The probe lattice is built once and reused across identical sweeps."""
    lattice_cache.clear()

    probe_build_calls = 0
    orig_build_solver = ge_solver.build_lattice
    orig_build_sweep = ge_sweep.build_lattice

    def counting_build_solver(*args, **kwargs):
        nonlocal probe_build_calls
        if kwargs.get("n_chord") == 1 and kwargs.get("collocation") == "vlm":
            probe_build_calls += 1
        return orig_build_solver(*args, **kwargs)

    def counting_build_sweep(*args, **kwargs):
        nonlocal probe_build_calls
        if kwargs.get("n_chord") == 1 and kwargs.get("collocation") == "vlm":
            probe_build_calls += 1
        return orig_build_sweep(*args, **kwargs)

    monkeypatch.setattr(ge_solver, "build_lattice", counting_build_solver)
    monkeypatch.setattr(ge_sweep, "build_lattice", counting_build_sweep)

    sett = vt.SolverSettings(solver_type="vlm", n_panels=16)
    sweep = GroundEffectSweep(wing_tail_aircraft, settings=sett, backend="serial")

    heights = [1.5, 2.0]
    alphas = [0.0, 4.0]
    phis = [0.0]

    res1 = sweep.run_sweep(heights, alphas, phis)
    calls_after_first = probe_build_calls
    assert calls_after_first > 0

    res2 = sweep.run_sweep(heights, alphas, phis)
    calls_during_second = probe_build_calls - calls_after_first

    assert calls_during_second == 0
    np.testing.assert_array_equal(res1.CL, res2.CL)
    np.testing.assert_array_equal(res1.CDi, res2.CDi)


def test_trefftz_half_equals_full(rectangular_wing, wing_tail_aircraft, monkeypatch):
    """Trefftz half evaluation equals full evaluation in symmetric cases."""
    geometries = [
        vt.Aircraft(name="Wing", surfaces=[rectangular_wing]),
        wing_tail_aircraft,
    ]
    solver = make_solver("vlm")
    sett = vt.SolverSettings(solver_type="vlm", n_panels=20)

    for geom in geometries:
        # 1. Symmetric free air
        cond_fa = vt.FlightCondition(V_inf=50.0, alpha=np.radians(4.0))
        res_half_fa = solver.solve(geom, cond_fa, sett)

        monkeypatch.setattr(loads_module, "_FORCE_FULL_TREFFTZ", True)
        res_full_fa = solver.solve(geom, cond_fa, sett)
        monkeypatch.setattr(loads_module, "_FORCE_FULL_TREFFTZ", False)

        np.testing.assert_allclose(
            res_half_fa.totals.CDi, res_full_fa.totals.CDi, rtol=1e-12, atol=1e-12,
        )

        # 2. Symmetric ground effect (phi = 0)
        cond_ge = vt.FlightCondition(V_inf=50.0, alpha=np.radians(4.0), h=1.5)
        res_half_ge = solver.solve(geom, cond_ge, sett)

        monkeypatch.setattr(loads_module, "_FORCE_FULL_TREFFTZ", True)
        res_full_ge = solver.solve(geom, cond_ge, sett)
        monkeypatch.setattr(loads_module, "_FORCE_FULL_TREFFTZ", False)

        np.testing.assert_allclose(
            res_half_ge.totals.CDi, res_full_ge.totals.CDi, rtol=1e-12, atol=1e-12,
        )

    # 3. Check that asymmetric cases (beta = 2 deg or phi = 3 deg) use full evaluation
    half_eval_called = False
    orig_core_data = loads_module._trefftz_core_data

    def spy_core_data(lattice, with_images, right_only=False):
        nonlocal half_eval_called
        if right_only:
            half_eval_called = True
        return orig_core_data(lattice, with_images, right_only=right_only)

    monkeypatch.setattr(loads_module, "_trefftz_core_data", spy_core_data)

    geom = geometries[0]
    # Free air with sideslip beta = 2 deg
    half_eval_called = False
    cond_beta = vt.FlightCondition(V_inf=50.0, alpha=np.radians(4.0), beta=np.radians(2.0))
    solver.solve(geom, cond_beta, sett)
    assert not half_eval_called

    # Ground effect with bank phi = 3 deg
    half_eval_called = False
    cond_phi = vt.FlightCondition(V_inf=50.0, alpha=np.radians(4.0), phi=np.radians(3.0), h=1.5)
    solver.solve(geom, cond_phi, sett)
    assert not half_eval_called


def test_asymmetric_circulation_uses_full_trefftz(rectangular_wing, monkeypatch):
    """Symmetric lattice with asymmetric circulation uses the full Trefftz evaluation."""
    ac = vt.Aircraft(name="Wing", surfaces=[rectangular_wing])
    solver = make_solver("vlm")
    sett = vt.SolverSettings(solver_type="vlm", n_panels=20)
    lat = solver.build(ac, sett, vt.FlightCondition(), None, np.zeros(3))
    assert lat.can_fold_symmetry() is True

    # Asymmetric circulation distribution
    gamma = np.ones(lat.n_strips)
    gamma[0] = 5.0
    wake_dirs = np.array([[1.0, 0.0, 0.0]])
    rho = np.array([1.225])

    half_eval_called = False
    orig_core_data = loads_module._trefftz_core_data

    def spy_core_data(lattice, with_images, right_only=False):
        nonlocal half_eval_called
        if right_only:
            half_eval_called = True
        return orig_core_data(lattice, with_images, right_only=right_only)

    monkeypatch.setattr(loads_module, "_trefftz_core_data", spy_core_data)

    loads_module.trefftz_induced_drag_batch(lat, gamma[None, :], wake_dirs, rho, [None])
    assert not half_eval_called
