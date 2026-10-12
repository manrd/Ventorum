# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the dispatch record in solver results.

Verifies that every lattice solver result records how it was computed and why
in result.details["dispatch"], covering device, precision, backends, threads,
workers, plan source, profile, and reason.
"""

from __future__ import annotations

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.agent.tools import wing_analysis
from ventorum.core.datatypes import FlightCondition, SolverSettings
from ventorum.ground_effect import GroundEffectSweep
from ventorum.hardware import profile
from ventorum.solvers.horseshoe import VortexLatticeSolver
from ventorum.solvers.linear import LinearLLTSolver
from ventorum.utils import parallel

DISPATCH_KEYS = {
    "device",
    "precision",
    "backends",
    "threads",
    "workers",
    "plan_source",
    "profile",
    "reason",
}

KERNEL_NAMES = {"tensor", "influence", "induced", "trefftz"}


def _rect_aircraft() -> vt.Aircraft:
    """Return a simple rectangular wing aircraft."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing])
    ac.compute_reference_values()
    return ac


def _assert_dispatch(dispatch: dict, expected_device: str | None = None) -> None:
    """Assert that a dispatch dictionary has all required keys and valid types."""
    assert isinstance(dispatch, dict)
    assert set(dispatch.keys()) == DISPATCH_KEYS
    assert dispatch["device"] in ("cpu", "gpu")
    if expected_device is not None:
        assert dispatch["device"] == expected_device
    assert dispatch["precision"] in ("float32", "float64")
    assert isinstance(dispatch["backends"], dict)
    assert set(dispatch["backends"].keys()) == KERNEL_NAMES
    assert isinstance(dispatch["threads"], int)
    assert dispatch["threads"] >= 1
    assert isinstance(dispatch["workers"], int)
    assert dispatch["workers"] >= 1
    assert dispatch["plan_source"] in ("user", "profile", "default", "nested")
    assert dispatch["profile"] is None or isinstance(dispatch["profile"], str)
    assert isinstance(dispatch["reason"], str)
    assert len(dispatch["reason"]) > 0


def test_single_solve_cpu_dispatch():
    """Single solve on CPU includes dispatch record with all keys."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        ac = _rect_aircraft()
        cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
        st = SolverSettings(solver_type="vlm", n_panels=16)
        res = VortexLatticeSolver().solve(ac, cond, st)
        assert "dispatch" in res.details
        _assert_dispatch(res.details["dispatch"], expected_device="cpu")
        assert res.details["dispatch"]["precision"] == "float64"
        # Existing top-level keys should remain absent for CPU
        assert "device" not in res.details
        assert "precision" not in res.details
    finally:
        gpu.set_device(old_dev)


def test_sweep_cpu_dispatch():
    """Angle sweep on CPU includes dispatch record on each result."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        ac = _rect_aircraft()
        cond = FlightCondition(V_inf=30.0)
        st = SolverSettings(solver_type="linear", n_panels=16)
        alphas = np.radians([0.0, 2.0, 4.0])
        results = LinearLLTSolver().solve_sweep(ac, cond, st, alphas)
        assert len(results) == len(alphas)
        for r in results:
            assert "dispatch" in r.details
            _assert_dispatch(r.details["dispatch"], expected_device="cpu")
    finally:
        gpu.set_device(old_dev)


def test_batch_cpu_dispatch():
    """Batch solve on CPU includes dispatch record on each result."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        ac = _rect_aircraft()
        solver = VortexLatticeSolver()
        st = SolverSettings(solver_type="vlm", n_panels=16)
        conds = [
            FlightCondition(V_inf=30.0, alpha=np.radians(a))
            for a in (0.0, 3.0, 6.0)
        ]
        lat = solver.build(ac, st, conds[0])
        results = solver.solve_batch(
            lat, conds, st, ac.S_ref, ac.b_ref, ac.c_ref
        )
        assert len(results) == len(conds)
        for r in results:
            assert "dispatch" in r.details
            _assert_dispatch(r.details["dispatch"], expected_device="cpu")
    finally:
        gpu.set_device(old_dev)


def test_ground_effect_sweep_cpu_dispatch():
    """Ground effect sweep on CPU includes dispatch record."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        wing = vt.LiftingSurface(
            name="wing",
            semi_span=3.0,
            sections=[
                vt.WingSection(y_frac=0.0, chord=1.0),
                vt.WingSection(y_frac=1.0, chord=0.7),
            ],
        )
        st = SolverSettings(solver_type="vlm", n_panels=12, n_chord=2)
        sweep = GroundEffectSweep(wing, settings=st)
        res = sweep.run_sweep(
            heights=[0.3, 0.6],
            alphas_deg=[2.0, 4.0],
            V_inf=30.0,
        )
        for r in res.results:
            if r is not None and r.solver_result is not None:
                assert "dispatch" in r.solver_result.details
                _assert_dispatch(
                    r.solver_result.details["dispatch"], expected_device="cpu"
                )
    finally:
        gpu.set_device(old_dev)


def test_fallback_reason_device_cpu():
    """GPU fallback reason identifies that device setting is cpu."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        ac = _rect_aircraft()
        cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
        st = SolverSettings(solver_type="vlm", n_panels=16)
        res = VortexLatticeSolver().solve(ac, cond, st)
        assert res.details["dispatch"]["reason"] == "device setting is cpu"
    finally:
        gpu.set_device(old_dev)


def test_fallback_reason_worker_thread(monkeypatch):
    """GPU fallback reason identifies when called inside a CPU worker thread."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("auto")
        monkeypatch.setattr(parallel._local, "worker_threads", 2, raising=False)
        ac = _rect_aircraft()
        cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
        st = SolverSettings(solver_type="vlm", n_panels=16)
        res = VortexLatticeSolver().solve(ac, cond, st)
        assert res.details["dispatch"]["reason"] == "called inside a CPU worker thread"
        assert res.details["dispatch"]["plan_source"] == "nested"
    finally:
        gpu.set_device(old_dev)


def test_fallback_reason_cost_model(monkeypatch):
    """GPU fallback reason gives the two cost-model times when the model rejects the GPU."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("auto")
        monkeypatch.setattr(gpu, "available", lambda: True)
        # Cost model where CPU is faster than GPU
        fake_model = {
            "n": [100, 1000],
            "cpu": {"k": [1, 33], "t": [[0.001, 0.033], [0.01, 0.33]]},
            "gpu": {"k": [1, 33], "t": [[0.05, 0.1], [0.5, 1.0]]},
        }
        monkeypatch.setattr(gpu, "cost_model", lambda family: fake_model)
        ac = _rect_aircraft()
        cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
        st = SolverSettings(solver_type="vlm", n_panels=100)
        solver = VortexLatticeSolver()
        lat = solver.build(ac, st, cond)
        res = solver.solve_batch(lat, [cond], st, ac.S_ref, ac.b_ref, ac.c_ref)[0]
        t_gpu = gpu.estimate_time(fake_model, "gpu", lat.n_panels, 1)
        t_cpu = gpu.estimate_time(fake_model, "cpu", lat.n_panels, 1)
        assert t_gpu >= t_cpu
        expected = f"auto: GPU cost model {t_gpu * 1e3:.1f} ms >= CPU {t_cpu * 1e3:.1f} ms"
        assert res.details["dispatch"]["reason"] == expected
    finally:
        gpu.set_device(old_dev)


def test_fallback_reason_min_work_rule(monkeypatch):
    """Without a cost model, the reason gives the work and the case count against the built-in rule."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("auto")
        monkeypatch.setattr(gpu, "available", lambda: True)
        monkeypatch.setattr(gpu, "cost_model", lambda family: None)
        use, reason = gpu.use_gpu(10.0, "vlm", 2, 40, return_reason=True)
        assert use is False
        mw, mc = gpu.min_work("vlm"), gpu.min_cases("vlm")
        assert reason == f"auto: work 10 < min_work {mw:.0f} and 2 cases < min_cases {mc}"
        use, reason = gpu.use_gpu(mw, "vlm", 1, 40, return_reason=True)
        assert use is True
        assert reason == f"auto: work {mw:.0f} >= min_work {mw:.0f}"
        use, reason = gpu.use_gpu(10.0, "vlm", mc, 40, return_reason=True)
        assert use is True
        assert reason == f"auto: {mc} cases >= min_cases {mc}"
    finally:
        gpu.set_device(old_dev)


def test_fallback_reason_gpu_unavailable(monkeypatch):
    """The reason names the missing GPU for the device settings gpu and auto."""
    old_dev = gpu.get_device()
    try:
        monkeypatch.setattr(gpu, "available", lambda: False)
        monkeypatch.setattr(gpu, "unavailable_reason", lambda: "no CUDA device")
        # set_device("gpu") refuses a missing GPU: a device setting from
        # before the GPU was lost gives this state.
        monkeypatch.setattr(gpu, "_device", "gpu")
        assert gpu.use_gpu(1e9, "vlm", 100, 400, return_reason=True) == (
            False, "GPU requested but unavailable: no CUDA device")
        monkeypatch.setattr(gpu, "_device", "auto")
        assert gpu.use_gpu(1e9, "vlm", 100, 400, return_reason=True) == (
            False, "GPU is not available: no CUDA device")
        assert gpu.use_gpu(1e9, "vlm", 100, 400) is False
    finally:
        gpu.set_device(old_dev)


def test_fallback_reason_device_cpu_same_wording():
    """The solver, the GPU batch and use_gpu give the same reason for the device setting cpu."""
    from ventorum.gpu.pipeline import solve_batch as gpu_solve_batch

    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        ac = _rect_aircraft()
        cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
        st = SolverSettings(solver_type="vlm", n_panels=16)
        solver = VortexLatticeSolver()
        lat = solver.build(ac, st, cond)
        res = solver.solve_batch(lat, [cond, cond], st, ac.S_ref, ac.b_ref, ac.c_ref)
        assert [r.details["dispatch"]["reason"] for r in res] == ["device setting is cpu"] * 2
        out = gpu_solve_batch(solver, lat, [cond], st, ac.S_ref, ac.b_ref, ac.c_ref, None, 0, False, 0.0,
                              return_reason=True)
        assert out == (None, "device setting is cpu")
        assert gpu.use_gpu(1e9, "vlm", 100, 400, return_reason=True) == (False, "device setting is cpu")
    finally:
        gpu.set_device(old_dev)


def test_fallback_reason_unsupported_solver():
    """The GPU batch names a solver family that it does not support."""
    from ventorum.gpu.pipeline import solve_batch as gpu_solve_batch

    class _Other:
        name = "other"

    old_dev = gpu.get_device()
    try:
        gpu.set_device("auto")
        out = gpu_solve_batch(_Other(), None, [], None, 1.0, 1.0, 1.0, None, 0, False, 0.0, return_reason=True)
        assert out == (None, "solver 'other' is not supported on GPU")
    finally:
        gpu.set_device(old_dev)


def test_fallback_reason_cpu_only_batch():
    """A batch with the private CPU-only flag records the reason of the flag."""
    ac = _rect_aircraft()
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = SolverSettings(solver_type="vlm", n_panels=16)
    solver = VortexLatticeSolver()
    lat = solver.build(ac, st, cond)
    res = solver.solve_batch(lat, [cond], st, ac.S_ref, ac.b_ref, ac.c_ref, _cpu_only=True)
    assert res[0].details["dispatch"]["reason"] == "caller requested cpu only"


@pytest.mark.gpu
def test_body_rates_on_gpu_record_gpu():
    """A case with body rates on device gpu runs on the GPU and records device gpu."""
    if not gpu.available():
        pytest.skip(f"GPU unavailable: {gpu.unavailable_reason()}")
    old_dev = gpu.get_device()
    try:
        gpu.set_device("gpu")
        ac = _rect_aircraft()
        cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), p=0.2, q=0.1)
        st = SolverSettings(solver_type="vlm", n_panels=16)
        res = VortexLatticeSolver().solve(ac, cond, st)
        assert res.details["device"] == "gpu"
        _assert_dispatch(res.details["dispatch"], expected_device="gpu")
        assert res.details["dispatch"]["reason"] == "device setting is gpu"
    finally:
        gpu.set_device(old_dev)


def test_profile_fingerprint_in_dispatch(monkeypatch):
    """Active profile fingerprint appears in dispatch record when profile is active."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        monkeypatch.setattr(
            profile,
            "active_profile",
            lambda: {"fingerprint": "test-fp-12345", "schema": 2, "settings": {}},
        )
        ac = _rect_aircraft()
        cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
        st = SolverSettings(solver_type="vlm", n_panels=16)
        res = VortexLatticeSolver().solve(ac, cond, st)
        assert res.details["dispatch"]["profile"] == "test-fp-12345"
    finally:
        gpu.set_device(old_dev)


def test_profile_none_when_autotune_disabled(monkeypatch):
    """Profile is None when autotuning is disabled."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        monkeypatch.setattr(profile, "active_profile", lambda: None)
        ac = _rect_aircraft()
        cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
        st = SolverSettings(solver_type="vlm", n_panels=16)
        res = VortexLatticeSolver().solve(ac, cond, st)
        assert res.details["dispatch"]["profile"] is None
    finally:
        gpu.set_device(old_dev)


def test_solver_result_to_dict_detail_levels():
    """SolverResult.to_dict gives dispatch only when detail_level is full."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        ac = _rect_aircraft()
        cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
        st = SolverSettings(solver_type="vlm", n_panels=16)
        res = VortexLatticeSolver().solve(ac, cond, st)

        d_std = res.to_dict(detail_level="standard")
        assert "dispatch" not in d_std

        d_sum = res.to_dict(detail_level="summary")
        assert "dispatch" not in d_sum

        d_full = res.to_dict(detail_level="full")
        assert "dispatch" in d_full
        _assert_dispatch(d_full["dispatch"])
    finally:
        gpu.set_device(old_dev)


def test_agent_tool_default_output_unchanged_and_full_has_dispatch():
    """Agent tool output is unchanged at standard level, and includes dispatch at full level."""
    wing = {"span_m": 10.0, "chord_m": 1.0}
    fast = {"n_panels": 12}

    out_std = wing_analysis(wing=wing, settings=fast, detail_level="standard")
    assert "dispatch" not in out_std

    out_full = wing_analysis(wing=wing, settings=fast, detail_level="full")
    assert "dispatch" in out_full
    _assert_dispatch(out_full["dispatch"])


@pytest.mark.gpu
def test_gpu_solve_dispatch():
    """GPU solve records dispatch with device gpu and warp backends."""
    if not gpu.available():
        pytest.skip(f"GPU unavailable: {gpu.unavailable_reason()}")
    old_dev = gpu.get_device()
    old_prec = gpu._precision
    try:
        gpu.set_device("gpu")
        gpu.set_precision("float64")
        ac = _rect_aircraft()
        cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
        st = SolverSettings(solver_type="vlm", n_panels=16)
        res = VortexLatticeSolver().solve(ac, cond, st)
        assert "dispatch" in res.details
        disp = res.details["dispatch"]
        _assert_dispatch(disp, expected_device="gpu")
        assert disp["device"] == "gpu"
        assert disp["precision"] == "float64"
        assert all(b == "warp" for b in disp["backends"].values())
        # Existing top-level details keys stay intact
        assert res.details["device"] == "gpu"
        assert res.details["precision"] == "float64"
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)


def _expected_solve_threads(n: int) -> int | None:
    """Return the kernel threads that solve_threads sets for a solve that runs alone, or None."""
    from ventorum.aero.vortex import get_kernel_backend

    if get_kernel_backend() not in ("numba", "cython"):
        return None
    import numba

    return max(1, min(int(n), numba.config.NUMBA_NUM_THREADS))


def test_batch_cpu_dispatch_records_plan_that_ran():
    """A 33-case CPU batch runs in the calling thread: one worker and the batch thread count."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        ac = _rect_aircraft()
        solver = VortexLatticeSolver()
        st = SolverSettings(solver_type="vlm", n_panels=16)
        conds = [FlightCondition(V_inf=30.0, alpha=np.radians(a)) for a in np.linspace(-4.0, 12.0, 33)]
        lat = solver.build(ac, st, conds[0])
        results = solver.solve_batch(lat, conds, st, ac.S_ref, ac.b_ref, ac.c_ref)
        expected = _expected_solve_threads(parallel.single_case_threads(lat.n_panels, batch=33))
        for r in results:
            d = r.details["dispatch"]
            assert d["workers"] == 1
            assert d["plan_source"] == "default"
            if expected is not None:
                assert d["threads"] == expected
    finally:
        gpu.set_device(old_dev)


def test_single_solve_profile_threads_record_profile(monkeypatch):
    """A single solve with the tuned single-case threads of the profile records plan_source profile."""
    from ventorum.hardware.profile import size_class

    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        ac = _rect_aircraft()
        cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
        st = SolverSettings(solver_type="vlm", n_panels=16)
        solver = VortexLatticeSolver()
        lat = solver.build(ac, st, cond)
        fake = {"fingerprint": "fp-single", "settings": {"single": {size_class(lat.n_panels): {"threads": 2}}}}
        monkeypatch.setattr(profile, "active_profile", lambda: fake)
        res = solver.solve_lattice(lat, cond, st, ac.S_ref, ac.b_ref, ac.c_ref)
        d = res.details["dispatch"]
        assert d["plan_source"] == "profile"
        assert d["workers"] == 1
        assert d["profile"] == "fp-single"
        expected = _expected_solve_threads(2)
        if expected is not None:
            assert d["threads"] == expected
    finally:
        gpu.set_device(old_dev)


def test_ground_sweep_user_jobs_record_user_plan():
    """A ground-effect sweep with n_jobs=3 on a lattice that is not small records the pool of 3 workers."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        wing = vt.LiftingSurface(
            name="wing",
            semi_span=3.0,
            sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
        )
        st = SolverSettings(solver_type="vlm", n_panels=60, n_chord=8)
        res = GroundEffectSweep(wing, settings=st, n_jobs=3).run_sweep(
            heights=[0.5, 1.0], alphas_deg=[2.0, 4.0], V_inf=30.0,
        )
        plan = parallel.plan_parallel(4, 480, 3)
        assert plan.workers == 3
        rs = [r for r in res.results if r is not None]
        assert len(rs) == 4
        for r in rs:
            d = r.solver_result.details["dispatch"]
            assert d["workers"] == 3
            assert d["plan_source"] == "user"
            expected = _expected_solve_threads(plan.threads_per_worker)
            if expected is not None:
                assert d["threads"] == expected
    finally:
        gpu.set_device(old_dev)


def test_serial_run_cases_with_user_jobs_record_user():
    """A serial run of cases with n_jobs=1 records one worker and plan_source user."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        ac = _rect_aircraft()
        st = SolverSettings(solver_type="vlm", n_panels=16)
        solver = VortexLatticeSolver()
        conds = [FlightCondition(V_inf=30.0, alpha=np.radians(a)) for a in (0.0, 4.0)]
        results = parallel.run_cases(lambda c: solver.solve(ac, c, st), conds, 64, n_jobs=1)
        for r in results:
            assert r.details["dispatch"]["workers"] == 1
            assert r.details["dispatch"]["plan_source"] == "user"
        assert getattr(parallel._local, "plan", None) is None
    finally:
        gpu.set_device(old_dev)
