# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Parity and correctness tests for ground sweeps and Trefftz loads on CPU."""

from __future__ import annotations

import threading

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.aero import loads as loads_module
from ventorum.geometry import lattice_cache
from ventorum.ground_effect import solver as ge_solver
from ventorum.ground_effect import sweep as ge_sweep
from ventorum.ground_effect.sweep import GroundEffectSweep
from ventorum.solvers import core as solver_core
from ventorum.solvers import horseshoe as horseshoe_module
from ventorum.solvers import linear as linear_module
from ventorum.solvers.factory import make_solver
from ventorum.solvers.lattice_base import LatticeSolver

# The lifting line warns below h_min/c = 2 in ground effect. The sweeps of
# these tests are there on purpose: do not show these expected warnings.
pytestmark = pytest.mark.filterwarnings("ignore:Lifting line in ground effect:RuntimeWarning")

_GRID_KEYS = (
    "CL", "CDi", "CD", "CY", "Cl", "Cm", "Cn",
    "L_over_D", "e", "h_min", "h_ref_grid", "phi_strike_limit",
)


def _assert_same_rows(res_a, res_b, rtol=1e-12, label=""):
    """Assert that two sweep results have the same grids, flags and errors."""
    for k in _GRID_KEYS:
        np.testing.assert_allclose(
            getattr(res_a, k), getattr(res_b, k), rtol=rtol, atol=1e-12 if rtol else 0.0, equal_nan=True,
            err_msg=f"Mismatch in grid {k} {label}",
        )
    np.testing.assert_array_equal(res_a.is_strike, res_b.is_strike)
    np.testing.assert_array_equal(res_a.refused, res_b.refused)
    assert res_a.errors == res_b.errors


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


@pytest.mark.slow
@pytest.mark.parametrize("phis", [(0.0,), (3.0,), (0.0, 3.0)], ids=["phi0", "phi3", "phi0_3"])
@pytest.mark.parametrize("solver_type", ["vlm", "linear", "nonlinear"])
def test_cpu_batch_sweep_equals_case_by_case(wing_tail_aircraft, solver_type, phis, monkeypatch):
    """CPU batch sweep gives the same rows as the case-by-case path.

    With one bank angle all cases have the same unknown map, so the batch
    solves several cases in one call (K > 1). With two bank angles the
    symmetric and the not symmetric cases form two groups.
    """
    sett = vt.SolverSettings(solver_type=solver_type, n_panels=16)
    sweep = GroundEffectSweep(wing_tail_aircraft, settings=sett, backend="serial")

    # Grid: 3 heights above h_min/c >= 1.0, plus 1 height where some cases strike.
    heights = [0.1, 1.5, 2.0, 3.0]
    alphas = [0.0, 2.0, 4.0, 6.0]
    phis = list(phis)

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


def _serial_sweep(ac, solver_type, n_panels=8):
    """Return a sweep of *ac* in serial mode with a small mesh."""
    sett = vt.SolverSettings(solver_type=solver_type, n_panels=n_panels)
    return GroundEffectSweep(ac, settings=sett, backend="serial")


def test_cpu_batch_does_not_change_the_device(wing_tail_aircraft, monkeypatch):
    """The CPU batch never calls gpu.set_device and never enters the GPU pipeline.

    Regression test: the CPU batch set the device to "cpu" for each chunk
    and restored it after the chunk. That global change is not thread-safe.
    """
    monkeypatch.setattr(gpu, "_device", "auto")
    monkeypatch.setattr(ge_sweep, "_gpu_batch", lambda *args, **kwargs: None)
    set_calls = []
    monkeypatch.setattr(gpu, "set_device", lambda name: set_calls.append(name))
    pipeline_calls = []

    def no_gpu(*args, **kwargs):
        pipeline_calls.append(1)
        return None

    monkeypatch.setattr("ventorum.gpu.pipeline.solve_batch", no_gpu)
    cpu_batch_calls = []
    orig_cpu_batch = ge_sweep._cpu_batch

    def spy_cpu_batch(*args, **kwargs):
        out = orig_cpu_batch(*args, **kwargs)
        cpu_batch_calls.append(out is not None)
        return out

    monkeypatch.setattr(ge_sweep, "_cpu_batch", spy_cpu_batch)

    res = _serial_sweep(wing_tail_aircraft, "vlm").run_sweep([1.5, 2.0], [0.0, 4.0], [0.0, 3.0])

    assert cpu_batch_calls == [True]
    assert set_calls == []
    assert pipeline_calls == []
    assert gpu.get_device() == "auto"
    assert np.all(np.isfinite(res.CL))


def _sweeps_in_threads(jobs):
    """Run each job ``(aircraft, solver, grid)`` as a sweep in its own thread; all start together."""
    barrier = threading.Barrier(len(jobs))
    results: list = [None] * len(jobs)
    failures: list = []

    def work(i, ac, solver_type, grid):
        try:
            barrier.wait()
            results[i] = _serial_sweep(ac, solver_type).run_sweep(*grid)
        except Exception as exc:  # noqa: BLE001 - the main thread reports it
            failures.append(exc)

    threads = [threading.Thread(target=work, args=(i, *job)) for i, job in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, failures


def test_parallel_sweeps_keep_the_device_and_the_results(wing_tail_aircraft, rectangular_wing, monkeypatch):
    """Two sweeps in parallel threads give the serial results and keep the device setting."""
    monkeypatch.setattr(gpu, "_device", "auto")
    monkeypatch.setattr(ge_sweep, "_gpu_batch", lambda *args, **kwargs: None)
    wing_ac = vt.Aircraft(name="Wing", surfaces=[rectangular_wing])
    jobs = [
        (wing_tail_aircraft, "vlm", ([1.5, 2.0], [0.0, 2.0, 4.0], [0.0, 3.0])),
        (wing_ac, "linear", ([1.5, 2.5], [0.0, 2.0, 4.0], [0.0, 3.0])),
    ]

    serial = [_serial_sweep(ac, s).run_sweep(*grid) for ac, s, grid in jobs]
    assert gpu.get_device() == "auto"

    for _ in range(3):
        parallel, failures = _sweeps_in_threads(jobs)
        assert failures == []
        assert gpu.get_device() == "auto"
        for res_p, res_s in zip(parallel, serial):
            _assert_same_rows(res_p, res_s, rtol=0.0, label="(parallel threads)")


@pytest.mark.parametrize("solver_type", ["vlm", "linear", "nonlinear"])
def test_mixed_bank_sweep_has_no_batch_fallback(wing_tail_aircraft, solver_type, monkeypatch):
    """A sweep with phi = 0 and phi = 3 deg solves each chunk as one batch.

    Regression test: the symmetric cases (phi = 0) have half the unknowns of
    the others. A chunk with both kinds made the batch solvers return None,
    and the cases were solved one by one.
    """
    batch_returns = []

    def spy(fn):
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            batch_returns.append(out is not None)
            return out
        return wrapped

    monkeypatch.setattr(horseshoe_module, "solve_vlm_batch", spy(horseshoe_module.solve_vlm_batch))
    monkeypatch.setattr(linear_module, "solve_llt_linear_batch", spy(linear_module.solve_llt_linear_batch))
    fallback_calls = []
    orig_one_by_one = LatticeSolver.solve_circulation_batch

    def spy_one_by_one(self, *args, **kwargs):
        fallback_calls.append(len(args[1]))
        return orig_one_by_one(self, *args, **kwargs)

    # The subclasses call the one-by-one solve of the base class only when
    # their batch cannot take the cases.
    monkeypatch.setattr(LatticeSolver, "solve_circulation_batch", spy_one_by_one)

    sweep = _serial_sweep(wing_tail_aircraft, solver_type, n_panels=12)
    grid = ([1.5, 2.0], [0.0, 4.0], [0.0, 3.0])
    res_batch = sweep.run_sweep(*grid)

    assert fallback_calls == []
    assert all(batch_returns)
    if solver_type in ("vlm", "linear"):
        assert len(batch_returns) == 2   # one call per group (symmetric, not symmetric)

    monkeypatch.setattr(ge_sweep, "_cpu_batch", lambda *args, **kwargs: None)
    res_single = sweep.run_sweep(*grid)
    _assert_same_rows(res_batch, res_single, label=f"for solver {solver_type}")


@pytest.mark.parametrize("solver_type", ["vlm", "linear", "nonlinear"])
def test_chunk_size_uses_the_unknowns_of_its_cases(wing_tail_aircraft, solver_type, monkeypatch):
    """Each chunk is sized with the unknown count of the cases that it contains.

    Regression test: the chunk size came from the first case only. A first
    symmetric case (half the unknowns) gave chunks of cases with twice the
    unknowns, so up to 4 times the memory budget.
    """
    events = []
    orig_chunk_size = ge_sweep._chunk_size

    def spy_chunk_size(canonical, n_unknowns):
        events.append(("chunk", n_unknowns))
        return orig_chunk_size(canonical, n_unknowns)

    monkeypatch.setattr(ge_sweep, "_chunk_size", spy_chunk_size)
    solver_cls = type(make_solver(solver_type))
    orig_batch = solver_cls.solve_circulation_batch

    def spy_batch(self, lattice, conditions, settings, grounds, wake_dirs, continuation=True):
        use_sym = getattr(settings, "use_symmetry", True)
        ns = {solver_core._unknown_map(lattice, c, g, use_sym).n for c, g in zip(conditions, grounds)}
        events.append(("solve", ns))
        return orig_batch(self, lattice, conditions, settings, grounds, wake_dirs, continuation)

    monkeypatch.setattr(solver_cls, "solve_circulation_batch", spy_batch)

    # The first case is symmetric (phi = 0), the next one is not (phi = 3 deg).
    _serial_sweep(wing_tail_aircraft, solver_type, n_panels=12).run_sweep([1.5, 2.0], [0.0, 4.0], [0.0, 3.0])

    last_chunk_n = None
    for kind, value in events:
        if kind == "chunk":
            last_chunk_n = value
        else:
            assert len(value) == 1, f"one chunk mixes unknown counts {sorted(value)}"
            assert value == {last_chunk_n}
    assert len([e for e in events if e[0] == "solve"]) == 2   # one chunk per group
