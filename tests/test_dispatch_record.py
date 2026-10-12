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
        reason = res.details["dispatch"]["reason"]
        assert "cpu" in reason.lower() and "device" in reason.lower()
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
        reason = res.details["dispatch"]["reason"]
        assert "worker thread" in reason.lower()
        assert res.details["dispatch"]["plan_source"] == "nested"
    finally:
        gpu.set_device(old_dev)


def test_fallback_reason_cost_model(monkeypatch):
    """GPU fallback reason identifies when cost model rejects GPU."""
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
        res = VortexLatticeSolver().solve(ac, cond, st)
        reason = res.details["dispatch"]["reason"]
        assert "cost model" in reason.lower()
    finally:
        gpu.set_device(old_dev)


def test_fallback_reason_body_rates(monkeypatch):
    """GPU fallback reason identifies when body rates need the CPU path."""
    old_dev = gpu.get_device()
    try:
        gpu.set_device("auto")
        solver = VortexLatticeSolver()
        # Simulate solver without GPU rate support
        monkeypatch.setattr(solver, "supports_gpu_rates", False, raising=False)
        ac = _rect_aircraft()
        cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), p=0.2)
        st = SolverSettings(solver_type="vlm", n_panels=16)
        res = solver.solve(ac, cond, st)
        reason = res.details["dispatch"]["reason"]
        assert "body rates" in reason.lower()
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
