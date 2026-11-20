"""Tuning gate of the agent tools.

The tuner sets the device, the precision and the machine profile for the
whole process while it measures. So the tools do not run a tuning run and
other tool calls at the same time.
"""

from __future__ import annotations

import threading

import pytest

import ventorum.agent.tools as tools
from ventorum.agent import wing_analysis

FC = {"V_inf_m_s": 40.0, "alpha_deg": 4.0}


def test_tool_calls_are_refused_while_a_tuning_run_is_in_progress(monkeypatch):
    """Regression: the tuner set the device and the profile while other solves ran in the process."""
    import ventorum.hardware.tuner as tuner

    started = threading.Event()
    release = threading.Event()

    def fake_tuner(quick, save, verbose):
        started.set()
        assert release.wait(30.0)
        return {"settings": {"kernels": {}}, "tuning_seconds": 0.0, "quick": quick}

    monkeypatch.setattr(tuner, "tune_machine", fake_tuner)
    out: dict = {}
    t = threading.Thread(target=lambda: out.setdefault("tune", tools.tune_machine(quick=True, save=False)))
    t.start()
    try:
        assert started.wait(30.0)
        p = wing_analysis(wing={"span_m": 10.0, "chord_m": 1.0}, flight_condition=dict(FC),
                          detail_level="summary")
        assert p["status"] == "error", p
        assert p["error"]["type"] == "invalid_input", p
        assert "tuning run is in progress" in p["error"]["message"], p
    finally:
        release.set()
        t.join(30.0)
    assert out["tune"]["status"] == "success", out
    assert tools._GATE_STATE == {"calls": 0, "tuning": False}
    # After the run, the calls work again.
    p = wing_analysis(wing={"span_m": 10.0, "chord_m": 1.0}, flight_condition=dict(FC), detail_level="summary")
    assert p["status"] == "success", p


def test_tuning_is_refused_while_other_tool_calls_run(monkeypatch):
    """A tuning run is refused while another tool call runs; the counter returns to zero."""
    import ventorum.hardware.tuner as tuner

    def no_tuner(*args, **kwargs):
        raise AssertionError("the tuner must not run while other calls run")

    monkeypatch.setattr(tuner, "tune_machine", no_tuner)
    monkeypatch.setitem(tools._GATE_STATE, "calls", 1)
    p = tools.tune_machine(quick=True, save=False)
    assert p["status"] == "error", p
    assert p["error"]["type"] == "invalid_input", p
    assert "other tool call" in p["error"]["message"], p
    assert tools._GATE_STATE["tuning"] is False
    monkeypatch.setitem(tools._GATE_STATE, "calls", 0)
    p = wing_analysis(wing={"span_m": 10.0, "chord_m": 1.0}, flight_condition=dict(FC), detail_level="summary")
    assert p["status"] == "success", p
    assert tools._GATE_STATE["calls"] == 0


@pytest.mark.parametrize("fail", [False, True])
def test_gate_counts_an_outermost_call_once(monkeypatch, fail):
    """Nested tool calls count once, and the count returns to zero also after an error."""
    depths = []

    @tools._tool
    def inner() -> dict:
        depths.append(tools._GATE_STATE["calls"])
        return {"status": "success"}

    @tools._tool
    def outer() -> dict:
        depths.append(tools._GATE_STATE["calls"])
        inner()
        if fail:
            raise ValueError("fails on purpose")
        return {"status": "success"}

    outer()
    assert depths == [1, 1]
    assert tools._GATE_STATE["calls"] == 0

