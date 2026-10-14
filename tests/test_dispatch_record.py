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
from ventorum.solvers.factory import make_solver
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
    if dispatch["device"] == "gpu":
        # The GPU record: the kernel groups that ran and no CPU thread plan.
        assert dispatch["backends"]["dense_solve"] == "torch"
        assert {"trefftz", "loads"} <= set(dispatch["backends"])
        assert all(v == "warp" for k, v in dispatch["backends"].items() if k != "dense_solve")
        assert dispatch["threads"] is None
        assert dispatch["workers"] is None
        assert dispatch["plan_source"] is None
    else:
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


def _small_ground_sweep():
    """Run a small ground-effect sweep (4 cases, small lattice: the CPU batch path)."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.7)],
    )
    st = SolverSettings(solver_type="vlm", n_panels=12, n_chord=2)
    res = GroundEffectSweep(wing, settings=st).run_sweep(heights=[0.3, 0.6], alphas_deg=[2.0, 4.0], V_inf=30.0)
    rs = [r for r in res.results if r is not None]
    assert len(rs) == 4
    return rs


def test_ground_sweep_cpu_batch_reason_device_cpu():
    """The CPU batch of a ground-effect sweep on device cpu records the device reason."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        for r in _small_ground_sweep():
            assert r.solver_result.details["dispatch"]["reason"] == "device setting is cpu"
    finally:
        gpu.set_device(old_dev)


def test_ground_sweep_cpu_batch_reason_min_work(monkeypatch):
    """The CPU batch of a small ground-effect sweep on device auto records the min_work rule."""
    import re

    old_dev = gpu.get_device()
    try:
        gpu.set_device("auto")
        monkeypatch.setattr(gpu, "available", lambda: True)
        monkeypatch.setattr(gpu, "cost_model", lambda family: None)
        mw, mc = gpu.min_work("vlm"), gpu.min_cases("vlm")
        pattern = rf"auto: work \d+ < min_work {mw:.0f} and 4 cases < min_cases {mc}"
        for r in _small_ground_sweep():
            assert "device" not in r.solver_result.details
            assert re.fullmatch(pattern, r.solver_result.details["dispatch"]["reason"])
    finally:
        gpu.set_device(old_dev)


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
        assert disp["backends"] == {"influence": "warp", "induced": "warp", "trefftz": "warp",
                                    "loads": "warp", "dense_solve": "torch"}
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
        expected = _expected_solve_threads(parallel.single_case_threads(lat.n_panels, batch=33, family="vlm"))
        for r in results:
            d = r.details["dispatch"]
            assert d["workers"] == 1
            # A tuned batch_threads entry of the machine profile makes the source "profile".
            assert d["plan_source"] == ("profile" if parallel._tuned_batch_threads(lat.n_panels, "vlm") else "default")
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


def _count_record_calls(monkeypatch) -> dict:
    """Count the calls of the backend and plan helpers of the dispatch record."""
    from ventorum.aero import vortex

    calls = {"backends": 0, "plan": 0}
    real_backends = vortex._kernel_backends_for
    real_plan = parallel._dispatch_plan

    def backends(*args, **kwargs):
        calls["backends"] += 1
        return real_backends(*args, **kwargs)

    def plan(*args, **kwargs):
        calls["plan"] += 1
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(vortex, "_kernel_backends_for", backends)
    monkeypatch.setattr(parallel, "_dispatch_plan", plan)
    return calls


def test_cpu_batch_builds_dispatch_record_once(monkeypatch):
    """A CPU batch makes the dispatch record once, and each case gets its own copy."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        ac = _rect_aircraft()
        solver = VortexLatticeSolver()
        st = SolverSettings(solver_type="vlm", n_panels=16)
        conds = [FlightCondition(V_inf=30.0, alpha=np.radians(a)) for a in np.linspace(-4.0, 12.0, 33)]
        lat = solver.build(ac, st, conds[0])
        calls = _count_record_calls(monkeypatch)
        results = solver.solve_batch(lat, conds, st, ac.S_ref, ac.b_ref, ac.c_ref)
        assert calls == {"backends": 1, "plan": 1}
        d0, d1 = results[0].details["dispatch"], results[1].details["dispatch"]
        assert d0 == d1
        assert d0 is not d1 and d0["backends"] is not d1["backends"]
    finally:
        gpu.set_device(old_dev)


@pytest.mark.gpu
def test_gpu_batch_does_not_build_cpu_record(monkeypatch):
    """A GPU batch does not make a CPU dispatch record."""
    if not gpu.available():
        pytest.skip(f"GPU unavailable: {gpu.unavailable_reason()}")
    old_dev = gpu.get_device()
    try:
        gpu.set_device("gpu")
        ac = _rect_aircraft()
        solver = VortexLatticeSolver()
        st = SolverSettings(solver_type="vlm", n_panels=16)
        conds = [FlightCondition(V_inf=30.0, alpha=np.radians(a)) for a in (0.0, 4.0, 8.0)]
        lat = solver.build(ac, st, conds[0])
        calls = _count_record_calls(monkeypatch)
        results = solver.solve_batch(lat, conds, st, ac.S_ref, ac.b_ref, ac.c_ref)
        assert calls == {"backends": 0, "plan": 0}
        for r in results:
            _assert_dispatch(r.details["dispatch"], expected_device="gpu")
        assert results[0].details["dispatch"] is not results[1].details["dispatch"]
    finally:
        gpu.set_device(old_dev)


def test_to_dict_refuses_unknown_detail_level():
    """SolverResult.to_dict refuses a detail level that it does not know."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        res = VortexLatticeSolver().solve(
            _rect_aircraft(), FlightCondition(V_inf=30.0, alpha=np.radians(4.0)),
            SolverSettings(solver_type="vlm", n_panels=16),
        )
        for bad in ("Full", "all", "", None):
            with pytest.raises(ValueError, match="detail_level"):
                res.to_dict(detail_level=bad)
    finally:
        gpu.set_device(old_dev)


@pytest.mark.parametrize("solver_type", ["linear", "nonlinear"])
def test_lifting_line_cpu_dispatch(solver_type):
    """The linear and nonlinear lifting lines record the CPU dispatch of a single solve and a sweep."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        ac = _rect_aircraft()
        st = SolverSettings(solver_type=solver_type, n_panels=16)
        single = vt.analyze(ac, FlightCondition(V_inf=30.0, alpha=np.radians(4.0)), st)
        sweep = make_solver(solver_type).solve_sweep(ac, FlightCondition(V_inf=30.0), st,
                                                         np.radians([0.0, 4.0]))
        assert single.solver_type == solver_type
        for r in [single, *sweep]:
            d = r.details["dispatch"]
            _assert_dispatch(d, expected_device="cpu")
            assert d["precision"] == "float64"
            assert d["workers"] == 1
            assert d["reason"] == "device setting is cpu"
        assert single.details["dispatch"]["plan_source"] in ("profile", "default")
        assert sweep[0].details["dispatch"]["plan_source"] in ("profile", "default")
    finally:
        gpu.set_device(old_dev)


@pytest.mark.gpu
@pytest.mark.parametrize("solver_type", ["linear", "nonlinear"])
def test_lifting_line_gpu_dispatch(solver_type):
    """The linear and nonlinear lifting lines on device gpu record the GPU dispatch."""
    if not gpu.available():
        pytest.skip(f"GPU unavailable: {gpu.unavailable_reason()}")
    old_dev = gpu.get_device()
    try:
        gpu.set_device("gpu")
        ac = _rect_aircraft()
        st = SolverSettings(solver_type=solver_type, n_panels=16)
        res = vt.analyze(ac, FlightCondition(V_inf=30.0, alpha=np.radians(4.0)), st)
        assert res.details["device"] == "gpu"
        _assert_dispatch(res.details["dispatch"], expected_device="gpu")
        assert res.details["dispatch"]["reason"] == "device setting is gpu"
    finally:
        gpu.set_device(old_dev)


@pytest.mark.gpu
def test_fallback_reason_nonlinear_groups():
    """A nonlinear batch with continuation and several case groups records the continuation reason."""
    if not gpu.available():
        pytest.skip(f"GPU unavailable: {gpu.unavailable_reason()}")
    old_dev = gpu.get_device()
    try:
        gpu.set_device("gpu")
        ac = _rect_aircraft()
        st = SolverSettings(solver_type="nonlinear", n_panels=16)
        solver = make_solver("nonlinear")
        # Symmetric and side-slip cases: two groups of unknowns.
        conds = [FlightCondition(V_inf=30.0, alpha=np.radians(a), beta=np.radians(b))
                 for a, b in ((2.0, 0.0), (4.0, 3.0), (6.0, 0.0))]
        lat = solver.build(ac, st, conds[0])
        res = solver.solve_batch(lat, conds, st, ac.S_ref, ac.b_ref, ac.c_ref, continuation=True)
        for r in res:
            assert "device" not in r.details
            _assert_dispatch(r.details["dispatch"], expected_device="cpu")
            assert r.details["dispatch"]["reason"] == "continuation needs the cases in order: the CPU path"
    finally:
        gpu.set_device(old_dev)


@pytest.mark.gpu
def test_fallback_reason_unsupported_polars(monkeypatch):
    """Section polars that the GPU pipelines refuse record the airfoil reason."""
    if not gpu.available():
        pytest.skip(f"GPU unavailable: {gpu.unavailable_reason()}")
    from ventorum.gpu import engine

    def refuse(dl, lattice):
        raise TypeError("polar not supported")

    monkeypatch.setattr(engine, "check_polars", refuse)
    old_dev = gpu.get_device()
    try:
        gpu.set_device("gpu")
        res = vt.analyze(_rect_aircraft(), FlightCondition(V_inf=30.0, alpha=np.radians(4.0)),
                         SolverSettings(solver_type="vlm", n_panels=16))
        assert "device" not in res.details
        _assert_dispatch(res.details["dispatch"], expected_device="cpu")
        assert res.details["dispatch"]["reason"] == "airfoils are not supported on GPU"
    finally:
        gpu.set_device(old_dev)


@pytest.mark.gpu
def test_fallback_reason_unknown_map(monkeypatch):
    """An unknown map that the GPU pipelines refuse records the unknown-map reason."""
    if not gpu.available():
        pytest.skip(f"GPU unavailable: {gpu.unavailable_reason()}")
    from ventorum.gpu import engine

    monkeypatch.setattr(engine.DeviceLattice, "unknown_map", lambda self, lattice, umap: None)
    old_dev = gpu.get_device()
    try:
        gpu.set_device("gpu")
        res = vt.analyze(_rect_aircraft(), FlightCondition(V_inf=30.0, alpha=np.radians(4.0)),
                         SolverSettings(solver_type="vlm", n_panels=16))
        assert "device" not in res.details
        assert res.details["dispatch"]["reason"] == "unknown map is not supported on GPU"
    finally:
        gpu.set_device(old_dev)


def test_spanwise_payload_does_not_change_caller_locals():
    """spanwise_payload does not write into a local dict 'out' of its caller."""
    from ventorum.agent.response import spanwise_payload

    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        res = VortexLatticeSolver().solve(
            _rect_aircraft(), FlightCondition(V_inf=30.0, alpha=np.radians(4.0)),
            SolverSettings(solver_type="vlm", n_panels=16),
        )
        assert "dispatch" in res.details

        def unrelated_caller():
            out = {"my": "data"}
            spanwise_payload(res)
            return out

        assert unrelated_caller() == {"my": "data"}
    finally:
        gpu.set_device(old_dev)


_TOOL_RECT = {"span_m": 10.0, "chord_m": 1.0}
_TOOL_WING_TAIL = {
    "surfaces": [
        {"name": "wing", "span_m": 12.0, "root_chord_m": 1.8, "tip_chord_m": 1.0},
        {"name": "tail", "span_m": 4.0, "chord_m": 0.7, "position_m": [4.6, 0.0, 0.3],
         "controls": [{"name": "elevator", "hinge_x_c": 0.7}]},
    ],
    "ref_point_m": [0.45, 0.0, 0.0],
}


def _tool_call(name: str, dl: str) -> dict:
    """Run the agent tool *name* on a small case at detail level *dl*."""
    from ventorum.agent import tools as T

    fast = {"n_panels": 8}
    if name == "wing_analysis":
        return T.wing_analysis(wing=_TOOL_RECT, settings=fast, detail_level=dl)
    if name == "polar_sweep":
        return T.polar_sweep(wing=_TOOL_RECT, alpha_start_deg=0, alpha_end_deg=4, alpha_step_deg=2,
                             settings=fast, detail_level=dl)
    if name == "ground_effect":
        return T.ground_effect(wing=_TOOL_RECT, heights_m=[0.5, 1.0], settings=fast, detail_level=dl)
    if name == "stability_derivatives":
        return T.stability_derivatives(wing=_TOOL_RECT, settings=fast, detail_level=dl)
    if name == "mesh_convergence":
        return T.mesh_convergence(wing=_TOOL_RECT, flight_condition={"alpha_deg": 4.0}, panel_counts=[8, 10],
                                  ref_n_panels=24, spacing_schemes=["cosine"], detail_level=dl)
    if name == "trim":
        return T.trim(aircraft=_TOOL_WING_TAIL, CL_target=0.5, pitch_control="elevator", settings=fast,
                      detail_level=dl)
    if name == "error_bars":
        return T.error_bars(wing=_TOOL_RECT, flight_condition={"alpha_deg": 4.0}, settings={"n_panels": 16},
                            detail_level=dl)
    if name == "derivative_matrix":
        return T.derivative_matrix(aircraft=_TOOL_WING_TAIL, settings=fast, detail_level=dl)
    raise AssertionError(name)


def _tool_records(name: str, disp) -> list:
    """Return the dispatch records of the full output of tool *name*, and check its shape."""
    if name in ("wing_analysis", "trim", "error_bars", "derivative_matrix"):
        return [disp]
    if name == "polar_sweep":
        assert isinstance(disp, list)
        return disp
    if name == "ground_effect":
        assert set(disp) == {"rows", "free_air"}
        return [*disp["rows"], disp["free_air"]]
    if name == "stability_derivatives":
        assert set(disp) == {"base", "a+", "a-", "b+", "b-"}
        return list(disp.values())
    if name == "mesh_convergence":
        assert set(disp) == {"points", "reference"}
        return [*disp["points"], disp["reference"]]
    raise AssertionError(name)


@pytest.mark.parametrize("name", [
    "wing_analysis", "polar_sweep", "ground_effect", "stability_derivatives", "mesh_convergence",
    "trim", "error_bars", "derivative_matrix",
])
def test_agent_tools_dispatch_only_at_full(name):
    """Every agent tool that runs a lattice solve gives dispatch at full detail only."""
    for dl in ("summary", "standard"):
        out = _tool_call(name, dl)
        assert out["status"] == "success", out
        assert "dispatch" not in out
    out = _tool_call(name, "full")
    assert out["status"] == "success", out
    assert "dispatch" in out
    records = _tool_records(name, out["dispatch"])
    assert records
    for rec in records:
        _assert_dispatch(rec)
    if name == "polar_sweep":
        assert len(out["dispatch"]) == len(out["polar_table"])
    if name == "ground_effect":
        assert len(out["dispatch"]["rows"]) == len(out["rows"])
    if name == "mesh_convergence":
        assert len(out["dispatch"]["points"]) == len(out["all_evaluated_points"])


def test_agent_tool_full_dispatch_none_for_fourier():
    """A tool that runs the Fourier solver gives dispatch None at full detail."""
    out = wing_analysis(wing=_TOOL_RECT, settings={"solver": "fourier", "n_panels": 12}, detail_level="full")
    assert out["status"] == "success"
    assert out["dispatch"] is None


@pytest.mark.gpu
@pytest.mark.parametrize("solver_type", ["vlm", "linear", "nonlinear"])
def test_gpu_record_shape(solver_type):
    """A GPU result names the kernel groups of its family and has no CPU thread plan."""
    if not gpu.available():
        pytest.skip(f"GPU unavailable: {gpu.unavailable_reason()}")
    expected = {
        "vlm": {"influence": "warp", "induced": "warp", "trefftz": "warp", "loads": "warp",
                "dense_solve": "torch"},
        "linear": {"tensor": "warp", "trefftz": "warp", "loads": "warp", "dense_solve": "torch"},
        "nonlinear": {"tensor": "warp", "newton": "warp", "trefftz": "warp", "loads": "warp",
                      "dense_solve": "torch"},
    }[solver_type]
    old_dev = gpu.get_device()
    try:
        gpu.set_device("gpu")
        ac = _rect_aircraft()
        st = SolverSettings(solver_type=solver_type, n_panels=16)
        solver = make_solver(solver_type)
        conds = [FlightCondition(V_inf=30.0, alpha=np.radians(a)) for a in (2.0, 4.0)]
        lat = solver.build(ac, st, conds[0])
        for r in solver.solve_batch(lat, conds, st, ac.S_ref, ac.b_ref, ac.c_ref, continuation=False):
            d = r.details["dispatch"]
            assert r.details["device"] == "gpu"
            assert d["device"] == "gpu"
            assert d["backends"] == expected
            assert d["threads"] is None and d["workers"] is None and d["plan_source"] is None
            assert d["precision"] == r.details["precision"]
    finally:
        gpu.set_device(old_dev)


@pytest.mark.gpu
def test_gpu_nonlinear_case_solved_again_on_cpu_records_cpu(monkeypatch):
    """A nonlinear case that does not converge on the GPU and is solved again on the CPU records the CPU."""
    if not gpu.available():
        pytest.skip(f"GPU unavailable: {gpu.unavailable_reason()}")
    from ventorum.gpu import engine

    real = engine.llt_nonlinear

    def one_case_fails(*args, **kwargs):
        out = real(*args, **kwargs)
        conv = out["converged"].clone()
        conv[1] = False
        out["converged"] = conv
        return out

    monkeypatch.setattr(engine, "llt_nonlinear", one_case_fails)
    old_dev = gpu.get_device()
    try:
        gpu.set_device("gpu")
        ac = _rect_aircraft()
        st = SolverSettings(solver_type="nonlinear", n_panels=16)
        solver = make_solver("nonlinear")
        conds = [FlightCondition(V_inf=30.0, alpha=np.radians(a)) for a in (2.0, 4.0, 6.0)]
        lat = solver.build(ac, st, conds[0])
        res = solver.solve_batch(lat, conds, st, ac.S_ref, ac.b_ref, ac.c_ref, continuation=False)
        for k in (0, 2):
            assert res[k].details["device"] == "gpu"
            _assert_dispatch(res[k].details["dispatch"], expected_device="gpu")
        d = res[1].details["dispatch"]
        _assert_dispatch(d, expected_device="cpu")
        assert d["precision"] == "float64"
        assert d["workers"] == 1
        assert d["reason"] == "gpu: the nonlinear solve did not converge; solved again on the cpu"
        assert "device" not in res[1].details
        assert "precision" not in res[1].details
        assert res[1].converged
    finally:
        gpu.set_device(old_dev)
