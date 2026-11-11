# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Regression tests for a bounded retry of the cold quasi-simultaneous estimate."""

from __future__ import annotations

import functools
from types import SimpleNamespace

import numpy as np
import pytest

from ventorum.airfoil import airfoil
from ventorum.panel2d import vii
from ventorum.panel2d import viscous as vs


@pytest.fixture
def newton_case():
    """Return a small case and its cold estimate at zero incidence [deg]."""
    setup = vs.prepare_case(
        airfoil("0012", n_points=61), 0.0, 1e6, ncrit=9.0,
        xtr=(1.0, 1.0), height=None, height_reference="pivot",
        wake_length=1.0, n_wake=30,
    )
    return setup, vii.initial_state(setup)


def _force_window_stop(monkeypatch, later=None):
    """Force one window stop and record whether the window stays off."""
    calls = []
    early_stop = vii._early_stop

    def stop(*args, **kwargs):
        calls.append(kwargs.get("window_stop", True))
        if len(calls) == 2:
            return vii._EarlyStop("stagnation", "stagnation: the smallest residual did not fall (test stop)", True)
        if len(calls) > 2 and later is not None:
            return later
        return early_stop(*args, **kwargs)

    monkeypatch.setattr(vii, "_early_stop", stop)
    return calls


@pytest.mark.slow
def test_retry_keeps_the_state_path_and_counts(monkeypatch, newton_case):
    """A retry resumes the same path and counts all iterations and linear solves."""
    setup, start = newton_case
    reference = vii.newton_solve([setup], [start.copy()])[0]
    assert reference.converged, reference.reason
    calls = _force_window_stop(monkeypatch)
    ctl = vs._AngleWork(0.0, vs.StopRules())
    result = vs._solve_one(setup, start.copy(), 1e-6, 50, 1.0 / 64.0, ctl=ctl, retry_window=True)
    assert result.converged, result.reason
    assert "converged after one retry" in result.reason
    assert "first stop: stagnation: the smallest residual" in result.reason
    assert calls[:2] == [True, True]
    assert calls[2:] and not any(calls[2:])
    assert np.array_equal(result.state.x, reference.state.x)
    assert np.array_equal(result.ue, reference.ue)
    assert result.iterations == reference.iterations == ctl.iterations
    assert result.linear_solves == reference.linear_solves == ctl.linear_solves
    assert ctl.solves == 2


@pytest.mark.slow
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_torch_batch_retries_only_the_stopped_case(monkeypatch, newton_case, device):
    """A torch batch keeps each case path and applies the retry per case."""
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    setup, start = newton_case
    kwargs = dict(xp=torch, device=device, dtype=torch.float64)
    reference = vii.newton_solve([setup, setup], [start.copy(), start.copy()], **kwargs)
    assert all(r.converged for r in reference), [r.reason for r in reference]
    _force_window_stop(monkeypatch)
    results = vii.newton_solve([setup, setup], [start.copy(), start.copy()], retry_window=True, **kwargs)
    assert not results[0]._first_stop
    assert results[1]._first_stop
    for result, ref in zip(results, reference):
        assert result.converged, result.reason
        assert np.array_equal(result.state.x, ref.state.x)
        assert np.array_equal(result.ue, ref.ue)
        assert result.iterations == ref.iterations
        assert result.linear_solves == ref.linear_solves


@pytest.mark.parametrize("status,reason", [
    ("stagnation", "stagnation: neither the Newton line search nor the Levenberg-Marquardt fallback reduces the residual"),
    ("divergence", "divergence: the residual grew"),
])
def test_retry_failure_names_both_stops(monkeypatch, newton_case, status, reason):
    """The retry is used once; a second stop retains both reasons."""
    setup, start = newton_case
    calls = _force_window_stop(monkeypatch, (status, reason))
    result = vii.newton_solve([setup], [start], retry_window=True)[0]
    assert not result.converged
    assert result.status == status
    assert reason in result.reason
    assert "first stop: stagnation: the smallest residual" in result.reason
    assert calls == [True, True, False]


@pytest.mark.parametrize("status,reason", [
    ("stagnation", "stagnation: neither the Newton line search nor the Levenberg-Marquardt fallback reduces the residual"),
    ("divergence", "divergence: the residual grew"),
])
def test_other_stops_do_not_retry(monkeypatch, newton_case, status, reason):
    """A failed-step stop or divergence does not start a retry."""
    setup, start = newton_case
    monkeypatch.setattr(vii, "_early_stop", lambda *args, **kwargs: (status, reason))
    result = vii.newton_solve([setup], [start], retry_window=True)[0]
    assert not result.converged
    assert result.status == status
    assert result.reason == reason
    assert result.iterations == 0


def test_non_finite_residual_does_not_retry(monkeypatch, newton_case):
    """A non-finite residual takes precedence over a residual-window stop."""
    setup, start = newton_case
    evaluate = vii._residual_m

    def non_finite(*args):
        residual, jacobian, ue = evaluate(*args)
        residual[:] = np.nan
        return residual, jacobian, ue

    monkeypatch.setattr(vii, "_residual_m", non_finite)
    monkeypatch.setattr(vii, "_early_stop", lambda *args, **kwargs: vii._EarlyStop(
        "stagnation", "stagnation: the smallest residual did not fall (test stop)", True))
    result = vii.newton_solve([setup], [start], retry_window=True)[0]
    assert not result.converged
    assert result.status == "not_finite"
    assert result.reason == "the residual is not finite (a closure left its range)"
    assert not result._first_stop
    assert result.iterations == 0


def test_retry_preserves_other_stop_rules():
    """Turning off the window keeps the failed-step and divergence rules."""
    args = (2.0, [2.0] * 14, 1.0, 2, True, 12, 0.5, 1e4)
    assert vii._early_stop(*args, window_stop=False)[0] == "stagnation"
    args = (2e4, [2e4] * 14, 1.0, 0, True, 12, 0.5, 1e4)
    assert vii._early_stop(*args, window_stop=False)[0] == "divergence"
    args = (2.0, [2.0] * 14, 1.0, 0, True, 12, 0.5, 1e4)
    assert vii._early_stop(*args, window_stop=False) is None


def test_retry_stays_inside_the_angle_budget(monkeypatch, newton_case):
    """A retry cannot extend the stage limit or the remaining angle budget."""
    setup, start = newton_case
    _force_window_stop(monkeypatch)
    ctl = vs._AngleWork(0.0, vs.StopRules(work_budget=5))
    ctl.iterations = 2
    result = vs._solve_one(setup, start, 1e-6, 50, 1.0 / 64.0, ctl=ctl, retry_window=True)
    assert not result.converged
    assert result.status == "budget"
    assert result.iterations == 3
    assert ctl.iterations == 5
    assert ctl.solves == 2
    assert "work budget of 5" in result.reason
    assert "first stop: stagnation: the smallest residual" in result.reason


def test_budget_before_target_keeps_retry_reasons(monkeypatch):
    """A budget stop before the requested angle keeps the intermediate retry notes."""
    first_stop = "stagnation: the smallest residual did not fall (test stop)"
    retry_reason = f"failed (iteration limit) after one retry (first stop: {first_stop})"

    def cold_path(*args, ctl, **kwargs):
        setup = SimpleNamespace(sys=SimpleNamespace(alpha=-2.0))
        result = vii.NewtonResult(None, None, False, 3, 2.0, retry_reason, 3, "max_iter")
        result._first_stop = first_stop
        ctl.add(setup, result, 3, 50)
        raise vs._BudgetUsedUp()

    monkeypatch.setattr(vs, "_cold_path", cold_path)
    row, state = vs.finish_angle(np.zeros((3, 2)), 4.0, 1e6, None, 1e-6, 50, {},
                                 rules=vs.StopRules(work_budget=3))
    assert state is None
    assert not row["converged"]
    assert np.isnan(row["Cl"])
    assert row["newton_iterations"] == 3
    assert row["newton_solves"] == 2
    assert "work budget of 3" in row["reason"]
    assert retry_reason in row["reason"]


def test_default_newton_stage_does_not_retry(monkeypatch, newton_case):
    """An ordinary stage stops at the first residual-window stop."""
    setup, start = newton_case
    calls = _force_window_stop(monkeypatch)
    result = vii.newton_solve([setup], [start])[0]
    assert not result.converged
    assert result.status == "stagnation"
    assert result.iterations == 1
    assert not result._first_stop
    assert calls == [True, True]


def test_failed_target_keeps_retry_and_early_stop_notes(monkeypatch):
    """A later cold-start stop does not replace an earlier retry reason."""
    retry_note = "stage at alpha = -2 deg: failed after one retry (first stop: stagnation)"
    early_note = "cold start ended after a stop"
    setup = SimpleNamespace(sys=SimpleNamespace(alpha=4.0, nodes=np.zeros((3, 2)), wake_nodes=np.zeros((2, 2))))
    result = vii.NewtonResult(None, None, False, 3, 2.0, "iteration limit", 3, "max_iter")

    def cold_path(*args, ctl, **kwargs):
        ctl.retry_notes.append(retry_note)
        ctl.note(early_note)
        return setup, result

    monkeypatch.setattr(vs, "_cold_path", cold_path)
    row, _ = vs.finish_angle(np.zeros((3, 2)), 4.0, 1e6, None, 1e-6, 50, {})
    assert retry_note in row["reason"]
    assert early_note in row["reason"]
    assert result.reason in row["reason"]


def test_only_cold_quasi_simultaneous_estimate_can_retry(monkeypatch):
    """The direct estimates do not retry, including the one at omega = 1."""
    calls = []
    monkeypatch.setattr(vii, "initial_state", lambda setup, sweeps, relax=None: sweeps)

    def solve(setup, start, *args, retry_window=False):
        calls.append((start, retry_window))
        return vii.NewtonResult(None, None, False, 1, 1.0, "iteration limit", 1, "max_iter")

    monkeypatch.setattr(vs, "_solve_one", solve)
    vs._cold_solve(None, 1e-6, 50)
    assert calls == [(0, False), (vii.QS_SWEEPS, True), (0, False)]


def test_zero_lift_start_does_not_retry(monkeypatch):
    """Only the cold start at the requested angle may retry, not the zero-lift start."""
    calls = []
    failed = vii.NewtonResult(None, None, False, 1, 1.0, "iteration limit", 1, "max_iter")

    def cold_solve(setup, tol, max_iter, bk=None, ctl=None, *, retry=True):
        calls.append((setup, retry))
        return failed

    monkeypatch.setattr(vs, "zero_lift_angle", lambda body, kw: -4.0)
    monkeypatch.setattr(vs, "prepare_case", lambda body, alpha, re, **kw: alpha)
    monkeypatch.setattr(vs, "_cold_solve", cold_solve)
    # The angle probe before the continuation path fails here (see test_angle_probe_*).
    monkeypatch.setattr(vs, "_angle_probe", lambda body, alpha, *args: (alpha, failed))
    setup, result = vs._cold_path(None, 4.0, 1e6, 1e-6, 50, {})
    assert result is failed
    assert calls == [(-4.0, False), (4.0, True)]


def _probe_mocks(monkeypatch, converged):
    """Mock the solves of the cold path; return the record of the calls."""
    calls = []
    result = vii.NewtonResult(None, None, converged, 1, 1e-9 if converged else 1.0,
                              "converged" if converged else "iteration limit", 1,
                              "converged" if converged else "max_iter")

    def initial_state(setup, sweeps=0, relax=None):
        calls.append(("initial_state", setup, sweeps, relax))
        return "start"

    def solve_one(setup, start, tol, max_iter, omega0, bk=None, ctl=None, *, retry_window=False):
        calls.append(("solve", setup, start, omega0, retry_window))
        return result

    def cold_solve(setup, tol, max_iter, bk=None, ctl=None, *, retry=True):
        calls.append(("cold", setup, retry))
        return vii.NewtonResult(None, None, False, 1, 1.0, "iteration limit", 1, "max_iter")

    monkeypatch.setattr(vs, "zero_lift_angle", lambda body, kw: -4.0)
    monkeypatch.setattr(vs, "prepare_case", lambda body, alpha, re, **kw: alpha)
    monkeypatch.setattr(vii, "initial_state", initial_state)
    monkeypatch.setattr(vs, "_solve_one", solve_one)
    monkeypatch.setattr(vs, "_cold_solve", cold_solve)
    return calls, result


def test_angle_probe_skips_the_continuation_path(monkeypatch):
    """A converged probe at the angle ends the cold path: no zero-lift start, no continuation steps.

    Regression: with the relaxed quasi-simultaneous sweeps the continuation path of the ground grid
    (NACA 4412, 4 deg) converged at the zero-lift angle, stopped in a late step, and the cold start at
    the angle converged after it (115 to 360 Newton iterations per height against 10 to 15 for the probe).
    """
    calls, result = _probe_mocks(monkeypatch, converged=True)
    setup, res = vs._cold_path(None, 4.0, 1e6, 1e-6, 50, {})
    assert res is result and setup == 4.0
    assert calls == [("initial_state", 4.0, vii.QS_SWEEPS, vii.QS_RELAX), ("solve", 4.0, "start", 1.0, False)]


def test_angle_probe_failure_keeps_the_continuation_path(monkeypatch):
    """A failed probe is followed by the former path: the zero-lift start, then the cold start at the angle."""
    calls, _ = _probe_mocks(monkeypatch, converged=False)
    vs._cold_path(None, 4.0, 1e6, 1e-6, 50, {})
    assert calls == [("initial_state", 4.0, vii.QS_SWEEPS, vii.QS_RELAX), ("solve", 4.0, "start", 1.0, False),
                     ("cold", -4.0, False), ("cold", 4.0, True)]


@pytest.mark.parametrize("alpha,kw", [(-2.5, {}), (4.0, {"wake_weights": "first_mean"})])
def test_no_angle_probe_without_a_path_or_with_the_mean_weights(monkeypatch, alpha, kw):
    """No probe near the zero-lift angle (no continuation path) and in the wake-weight continuation."""
    calls, _ = _probe_mocks(monkeypatch, converged=True)
    vs._cold_path(None, alpha, 1e6, 1e-6, 50, kw)
    assert not [c for c in calls if c[0] != "cold"]


@pytest.mark.slow
@pytest.mark.parametrize("height,alphas,expected", [
    (0.5, 0.0, {
        "Cl": -0.043692190463068326, "Cd": 0.006227655458166425,
        "Cm": 0.007746622772972753, "iterations": 19,
        "newton_iterations": 19, "linear_solves": 19, "newton_solves": 1,
    }),
    (None, [0.0, 2.0, 4.0], {
        "Cl": [-6.375108774214766e-16, 0.20651733529175564, 0.41447571631957375],
        "Cd": [0.005991106907818574, 0.0064649960462908425, 0.007888126934412442],
        "Cm": [1.3720578492804059e-16, 0.004291754868394739, 0.008347585741137413],
        "iterations": [16, 17, 18], "newton_iterations": [16, 17, 18],
        "linear_solves": [16, 17, 18], "newton_solves": [1, 1, 1],
    }),
])
def test_unstopped_cold_and_warm_results_keep_their_values(height, alphas, expected):
    """Stored cold and warm NACA 0012 loads and work do not change.

    The values are bit-identical on the machine that made them. Other numpy
    and BLAS builds (CI) take another Newton path, so the loads agree to the
    level of the convergence tolerance and the work counts may differ by a
    few iterations.
    """
    # Generated with the length weights of the interior wake stations (solver revision 2026-11-11.1).
    # NACA 0012: 61 nodes, Re 1e6, default solver settings, one-thread environment.
    result = vs.solve_viscous(airfoil("0012", n_points=61), alphas, 1e6, height=height)
    assert np.all(result.converged), result.reason
    for name, values in expected.items():
        got = np.asarray(getattr(result, name), dtype=float)
        if name in ("Cl", "Cd", "Cm"):
            assert np.allclose(got, values, rtol=1e-5, atol=1e-9), name
        elif name == "newton_solves":
            assert np.array_equal(got, values), name
        else:
            assert np.all(np.abs(got - np.asarray(values)) <= np.maximum(5, 0.25 * np.asarray(values))), name
    assert all(reason == "converged" for reason in np.atleast_1d(result.reason))


@pytest.mark.slow
def test_ground_repro_converges_with_monotonic_lift():
    """NACA 4412 at 4 deg and Re 1e6 converges across the failed heights [m].

    The heights 0.8 and 1.0 failed before the retry of the quasi-simultaneous first estimate, and the
    retry rescued them. With the length weights of the interior wake stations and the relaxed
    quasi-simultaneous sweeps they converge without the retry (measured: no height of the grid of
    test_ground_height_grid_converges uses the retry now), so the test requires convergence, the
    residual and the work, not the retry. The retry itself is tested by the other tests of this file.
    """
    shape = airfoil("4412", n_points=161)
    results = [vs.solve_viscous(shape, [4.0], 1e6, height=h) for h in (0.5, 0.8, 1.0, 1.5)]
    assert all(np.all(r.converged) for r in results), [r.reason for r in results]
    assert np.all(np.diff([r.Cl[0] for r in results]) < 0.0)
    for result in results[1:3]:
        assert result.residual[0] < 1e-6
        assert result.newton_iterations[0] <= vs.WORK_BUDGET


# Generated with the length weights of the interior wake stations and the start strategies of the
# solver revision 2026-11-11.3 (angle probe before the continuation path; one-thread environment).
# NACA 4412, 161 nodes, 4 deg, Re 1e6. Each tuple holds h/c [-], Cl [-], Cd [-] and all Newton
# iterations [-].
_BASELINE_GROUND = [
    (0.3, 0.9531249635683076, 0.006623102391113474, 12),
    (0.4, 0.9240978837383023, 0.006739222655572624, 15),
    (0.5, 0.9076812429241111, 0.006829822636608692, 15),
    (0.6, 0.8978250388711114, 0.006903295187759113, 14),
    (0.7, None, None, 13),
    (0.8, None, None, 12),
    (0.9, 0.885760591500639, 0.007056882327130597, 11),
    (1.0, None, None, 12),
    (1.1, None, None, 12),
    (1.2, 0.883340905189164, 0.007156948100037195, 11),
    (1.3, None, None, 10),
    (1.4, 0.8834170648212244, 0.007206394545050116, 10),
    (1.5, 0.8836723548291316, 0.007227430027446213, 10),
    (1.6, 0.8840099651655114, 0.007246464062543222, 10),
    (1.7, 0.8844006647011214, 0.0072637646890125595, 10),
    (1.8, None, None, 10),
    (1.9, 0.8852656094844145, 0.007294021562758571, 10),
    (2.0, None, None, 10),
    (2.1, 0.8861661231181058, 0.007319591190268501, 10),
    (2.2, None, None, 10),
    (2.3, 0.8870523327316073, 0.007341474308113431, 10),
    (2.4, 0.8874819915902379, 0.007351271685967537, 10),
    (2.5, 0.8879003702649196, 0.007360408175106519, 10),
    (2.6, 0.8883065131144754, 0.00736894795379221, 10),
    (2.7, 0.888699886000922, 0.007376947213781249, 10),
    (2.8, 0.889080259729121, 0.007384455355328283, 10),
    (2.9, 0.8894476246199263, 0.007391515976719019, 10),
    (3.0, 0.8898021276104037, 0.00739816769691257, 10),
]


# Heights that were above WORK_BUDGET with the continuation path after the relaxed quasi-simultaneous
# sweeps (273, 306 and 319 iterations; they were inside it before those sweeps). The angle probe of
# the cold path converges there in 10 iterations; test_ground_height_grid_work_budget keeps the check.
_OVER_BUDGET_HEIGHTS = (1.9, 2.1, 2.7)


@functools.cache
def _ground_grid_solve(height):
    """Solve NACA 4412 (161 nodes) at 4 deg and Re 1e6 at the ground height h/c [-], once per height."""
    return vs.solve_viscous(airfoil("4412", n_points=161), 4.0, 1e6, height=height)


@pytest.mark.slow
@pytest.mark.parametrize("height,cl,cd,iterations", _BASELINE_GROUND)
def test_ground_height_grid_converges(height, cl, cd, iterations):
    """Keep baseline convergence and rescue first-estimate stagnation stops."""
    result = _ground_grid_solve(height)
    assert result.converged, (height, result.reason)
    assert result.residual < 1e-6
    # The work agrees with the stored count on every height (also where the loads are not stored).
    # Other numpy and BLAS builds (CI) take another Newton path: up to 10 of about 120 iterations
    # differ there.
    assert abs(int(result.newton_iterations) - iterations) <= max(5, 0.25 * iterations)
    assert result.newton_iterations <= vs.WORK_BUDGET
    if cl is None:
        # These heights failed before the retry of the quasi-simultaneous
        # first estimate. Now they converge without it on this machine;
        # other numpy and BLAS builds (CI) can take other paths.
        # Convergence is the requirement.
        pass
    elif "retry" not in result.reason:
        # Bit-identical on the machine that made the baseline. Other numpy and
        # BLAS builds (CI) take another Newton path: the loads agree to the
        # level of the convergence tolerance and the work differs by a few
        # iterations (up to 10 of about 120 seen on CI).
        assert np.allclose([result.Cl, result.Cd], [cl, cd], rtol=1e-5, atol=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("height", _OVER_BUDGET_HEIGHTS)
def test_ground_height_grid_work_budget(height):
    """Keep the work of the grid heights that left WORK_BUDGET inside the budget.

    Regression: with the relaxed quasi-simultaneous sweeps, the continuation path used 273 to 319
    Newton iterations at these heights. The angle probe of the cold path removes that extra work.
    """
    result = _ground_grid_solve(height)
    assert result.converged, (height, result.reason)
    assert result.newton_iterations <= vs.WORK_BUDGET
