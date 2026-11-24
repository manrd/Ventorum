# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Regression tests for jumps in callable motion and control inputs.

A callable input gets its rate from a tiny central difference. Before this
fix, a jump exactly at a sample time gave a rate of about jump / (2e-7 dt),
and a free-flight run diverged; a jump between two sample times gave no rate
at all. Now the sampler finds such jumps and uses the grid rate of the
sampled values there, as for an array input, and it adds a note. Smooth
callables and continuous callables with a kink keep their rates bit for bit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import ventorum as vt
from ventorum.agent import call_tool
from ventorum.unsteady.timedomain import _grid_rates, _sample_motion

DT = 0.01  # s
TIME = np.arange(21, dtype=np.float64) * DT  # s


def _tiny_rates(fn, time: np.ndarray, dt: float) -> np.ndarray:
    """Return the tiny-difference rates of the previous rule (no jump test)."""
    eps = 1.0e-7 * dt
    out = np.empty(time.size)
    for i, t in enumerate(time):
        t = float(t)
        if i == 0:
            out[i] = (-3.0 * fn(t) + 4.0 * fn(t + eps) - fn(t + 2.0 * eps)) / (2.0 * eps)
        else:
            h = min(eps, 0.25 * t)
            out[i] = (fn(t + h) - fn(t - h)) / (2.0 * h)
    return out


@pytest.mark.parametrize(
    "fn",
    [
        lambda t: 0.03 * math.sin(2.0 * math.pi * 5.0 * t),
        lambda t: 0.02 * math.cos(2.0 * math.pi * 3.0 * t) + 0.01,
        lambda t: 0.0 if t <= 0.035 else min(0.04, 0.5 * (t - 0.035)),  # ramp, kinks inside intervals
        lambda t: 0.0 if t <= 0.05 else min(0.04, 0.5 * (t - 0.05)),  # ramp, kink at a sample time
        lambda t: 0.07,
        lambda t: 0.0,
    ],
    ids=["sine", "cosine", "ramp-off-grid", "ramp-on-grid", "constant", "zero"],
)
def test_smooth_and_kinked_callables_keep_their_rates(fn):
    """Resolved smooth inputs and ramps keep the tiny-difference rates exactly, with no note."""
    notes: list[str] = []
    vals, dots = _sample_motion(fn, TIME, DT, "test", notes)
    assert np.array_equal(dots, _tiny_rates(fn, TIME, DT))
    assert np.array_equal(vals, np.array([fn(float(t)) for t in TIME]))
    assert notes == []


@pytest.mark.parametrize("t_jump", [float(TIME[7]), 0.5 * float(TIME[7] + TIME[8])], ids=["on-sample", "between"])
def test_jump_gives_the_array_rates_and_a_note(t_jump):
    """A step at a sample time or between two samples gives the rates of the sampled array."""
    amp = 0.035

    def step(t: float) -> float:
        return amp if t >= t_jump else 0.0

    notes: list[str] = []
    vals, dots = _sample_motion(step, TIME, DT, "flap", notes)
    _, dots_arr = _sample_motion(vals, TIME, DT, "flap-array")
    assert np.array_equal(dots, dots_arr)
    assert np.max(np.abs(dots)) == pytest.approx(amp / (2.0 * DT))
    # The trapezoidal integral of the rate equals the jump.
    assert np.trapezoid(dots, TIME) == pytest.approx(amp, rel=1e-12)
    assert len(notes) == 1 and "jump" in notes[0] and "flap" in notes[0]


def test_grid_rates_match_the_array_rule():
    """The shared grid rule is second order inside and at the ends, and first order for two samples."""
    arr = TIME**2
    d = _grid_rates(arr, DT)
    assert d == pytest.approx(2.0 * TIME, abs=1e-12)
    assert np.array_equal(_grid_rates(np.array([1.0, 3.0]), 0.5), np.array([4.0, 4.0]))


def _wing_tail() -> vt.Aircraft:
    """Build a small wing-tail aircraft with a tail elevator."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    tail = vt.LiftingSurface(
        name="tail",
        semi_span=1.0,
        position=np.array([1.8, 0.0, 0.0]),
        controls=[vt.ControlSurface(name="elevator", eta_start=0.0, eta_end=1.0, hinge_x_c=0.5, deflection=0.0)],
        sections=[vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.5)],
    )
    ac = vt.Aircraft(surfaces=[wing, tail], ref_point=np.array([-0.15, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


_SETTINGS = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
_COND = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=math.radians(2.0))


def test_time_response_step_callable_equals_array_input():
    """A flap step callable at a sample time gives the run of its sampled array, with a note."""
    ac = _wing_tail()
    n_steps = 12
    probe = vt.time_response(ac, _COND, _SETTINGS, n_steps=1)
    t_jump = float(np.arange(n_steps + 1)[4] * probe.dt)
    amp = math.radians(2.0)

    def step(t: float) -> float:
        return amp if t >= t_jump else 0.0

    res_fn = vt.time_response(ac, _COND, _SETTINGS, n_steps=n_steps, controls={"elevator": step})
    samples = np.array([step(float(t)) for t in res_fn.time])
    res_arr = vt.time_response(ac, _COND, _SETTINGS, n_steps=n_steps, controls={"elevator": samples})
    for key in ("CL", "Cm"):
        assert np.allclose(getattr(res_fn, key), getattr(res_arr, key), rtol=0.0, atol=1e-12), key
    assert any("jump" in n and "elevator" in n for n in res_fn.notes)
    assert not any("jump" in n for n in res_arr.notes)


def test_free_flight_step_callable_at_a_sample_does_not_diverge():
    """The free-flight case that diverged before the fix now equals the array input run."""
    ac = _wing_tail()
    n_steps = 20
    mass, inertia = 400.0, np.diag([80.0, 120.0, 180.0])
    probe = vt.free_flight(ac, _COND, _SETTINGS, mass=mass, inertia=inertia, dofs=(), n_steps=1)
    t_jump = float(5 * probe.dt)
    amp = math.radians(2.0)

    def step(t: float) -> float:
        return amp if t >= t_jump else 0.0

    kw = dict(mass=mass, inertia=inertia, dofs=("z", "pitch"), n_steps=n_steps)
    res_fn = vt.free_flight(ac, _COND, _SETTINGS, controls={"elevator": step}, **kw)
    samples = np.array([step(float(t)) for t in res_fn.time])
    res_arr = vt.free_flight(ac, _COND, _SETTINGS, controls={"elevator": samples}, **kw)
    assert np.all(np.abs(np.degrees(res_fn.pitch)) < 5.0)
    assert np.allclose(res_fn.pitch, res_arr.pitch, rtol=0.0, atol=1e-12)
    assert np.allclose(res_fn.z, res_arr.z, rtol=0.0, atol=1e-12)
    assert any("jump" in n and "elevator" in n for n in res_fn.notes)


def test_agent_step_waveform_on_a_sample_time_gets_a_note():
    """The agent step waveform with t_start_s on a sample time runs and reports the jump note."""
    wing = {
        "span_m": 6.0,
        "chord_m": 1.0,
        "controls": [{"name": "flap", "eta_start": 0.0, "eta_end": 1.0, "hinge_x_c": 0.75, "deflection_deg": 0.0}],
    }
    base = {"wing": wing, "flight_condition": {"V_inf_m_s": 20.0, "alpha_deg": 2.0}, "n_steps": 6,
            "settings": {"n_panels": 4, "n_chord": 4}}
    probe = call_tool("ventorum_time_response", base)
    assert probe["status"] == "success", probe
    t_start = probe["time"][2]
    out = call_tool(
        "ventorum_time_response",
        {**base, "controls": {"flap": {"kind": "step", "amplitude": 2.0, "t_start_s": t_start}}},
    )
    assert out["status"] == "success", out
    assert any("jump" in n for n in out["notes"])
    assert max(abs(c) for c in out["CL"]) < 1.0
