# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Regression tests for a bounded retry of the cold quasi-simultaneous estimate."""

from __future__ import annotations

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
    setup, result = vs._cold_path(None, 4.0, 1e6, 1e-6, 50, {})
    assert result is failed
    assert calls == [(-4.0, False), (4.0, True)]


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
# solver revision 2026-11-11.1 (one-thread environment). NACA 4412, 161 nodes, 4 deg, Re 1e6.
# Each tuple holds h/c [-], Cl [-], Cd [-] and all Newton iterations [-].
_BASELINE_GROUND = [
    (0.3, 0.9531226948196488, 0.006623107045503646, 134),
    (0.4, 0.9241018086448973, 0.006739203326902924, 142),
    (0.5, 0.9076812429241111, 0.006829822636608692, 202),
    (0.6, 0.8978250388711114, 0.006903295187759113, 201),
    (0.7, None, None, 218),
    (0.8, None, None, 131),
    (0.9, 0.8857672507166842, 0.007056817085067056, 139),
    (1.0, None, None, 126),
    (1.1, None, None, 115),
    (1.2, 0.8833407694628899, 0.007156947077280827, 136),
    (1.3, None, None, 134),
    (1.4, 0.8834170789099528, 0.007206394516815438, 150),
    (1.5, 0.8836749658843072, 0.00722744593443187, 214),
    (1.6, 0.8840099651655114, 0.007246464062543222, 214),
    (1.7, 0.8844006647011214, 0.0072637646890125595, 185),
    (1.8, None, None, 279),
    (1.9, 0.885265472925404, 0.007294020275162975, 273),
    (2.0, None, None, 293),
    (2.1, 0.8861659803618196, 0.007319589834427724, 306),
    (2.2, None, None, 360),
    (2.3, 0.8870523327316073, 0.007341474308113431, 211),
    (2.4, 0.8874819915902379, 0.007351271685967537, 199),
    (2.5, 0.8879003862696289, 0.007360407940784891, 121),
    (2.6, 0.8883065131144754, 0.00736894795379221, 245),
    (2.7, 0.8886999026061962, 0.007376946977316944, 319),
    (2.8, 0.8890802766766231, 0.007384455119416496, 149),
    (2.9, 0.8894476419726463, 0.00739151574102193, 148),
    (3.0, 0.8898021452507265, 0.007398167460515863, 140),
]


@pytest.mark.slow
@pytest.mark.parametrize("height,cl,cd,iterations", _BASELINE_GROUND)
def test_ground_height_grid_converges(height, cl, cd, iterations):
    """Keep baseline convergence and rescue first-estimate stagnation stops."""
    result = vs.solve_viscous(airfoil("4412", n_points=161), 4.0, 1e6, height=height)
    assert result.converged, (height, result.reason)
    assert result.residual < 1e-6
    # The work agrees with the stored count on every height (also where the loads are not stored).
    # Other numpy and BLAS builds (CI) take another Newton path: up to 10 of about 120 iterations
    # differ there.
    assert abs(int(result.newton_iterations) - iterations) <= max(5, 0.25 * iterations)
    if iterations <= vs.WORK_BUDGET:
        assert result.newton_iterations <= vs.WORK_BUDGET
    # Else newton_iterations also counts the work of the continuation path (the cold start at the
    # zero-lift angle and the continuation steps), which has its own budgets (COLD_START_BUDGET and
    # STEP_BUDGET). With the relaxed quasi-simultaneous sweeps the cold start at the zero-lift angle
    # converges more often; a continuation step can then stop before the cold start at the angle
    # converges (traced at h/c 0.5: 202 iterations against 85 before). Measured: 273 to 360 here.
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
