# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the start strategies of the viscous solver: quasi-simultaneous first estimate, wake-weight continuation."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from ventorum.airfoil import airfoil
from ventorum.panel2d import vii
from ventorum.panel2d import viscous as vs

_KW = dict(ncrit=9.0, xtr=(1.0, 1.0), height=None, height_reference="pivot", wake_length=1.0, n_wake=30)


def _flap10_setup():
    """Return the setup of the NACA 2412 with a 10 deg flap (outline as constructed, 160 panels) at alpha 0."""
    coords = airfoil("2412").repanel(n_panels=160).with_flap(x_hinge=0.75, deflection=10.0,
                                                             hinge_treatment="none").coords
    return vs.prepare_case(coords, 0.0, 1e6, **_KW)


def test_first_mean_weights_change_only_the_first_interior_wake_station():
    """The map "first_mean" has the weights 0.5 and 0.5 at the first interior wake station and no other change."""
    coords = airfoil("0012").repanel(n_panels=80).coords
    length = vs.prepare_case(coords, 2.0, 1e6, **_KW)
    mean = vs.prepare_case(coords, 2.0, 1e6, wake_weights="first_mean", **_KW)
    assert length.wake_weights == "length" and mean.wake_weights == "first_mean"
    n = length.n
    assert mean._p_wake[n + 1, n] == 0.5 and mean._p_wake[n + 1, n + 1] == 0.5
    assert length._p_wake[n + 1, n] != 0.5
    diff = np.argwhere(mean._p_wake != length._p_wake)
    assert {int(r) for r in diff[:, 0]} == {n + 1}
    with pytest.raises(ValueError, match="wake_weights"):
        vs.prepare_case(coords, 2.0, 1e6, wake_weights="mean", **_KW)


def test_quasi_simultaneous_start_is_the_sweep_with_the_smallest_residual(monkeypatch):
    """The quasi-simultaneous first estimate keeps the sweep with the smallest residual at omega = 1.

    Regression: the plain sweeps oscillate on flapped sections (period two), and the last sweep was the
    start even when it was the worst one (10 deg flap: ue < 0 at the trailing edge after the third
    sweep). The sweeps are now under-relaxed, and the start is the best sweep.
    """
    setup = _flap10_setup()
    seen = []
    original = vii.start_residual

    def record(s, st):
        r = original(s, st)
        seen.append(r)
        return r

    monkeypatch.setattr(vii, "start_residual", record)
    st = vii.initial_state(setup, sweeps=vii.QS_SWEEPS)
    monkeypatch.setattr(vii, "start_residual", original)
    assert len(seen) == vii.QS_SWEEPS
    assert np.isfinite(min(seen))
    assert vii.start_residual(setup, st) == min(seen)
    # The direct marches (no sweeps) have no selection.
    seen.clear()
    monkeypatch.setattr(vii, "start_residual", record)
    vii.initial_state(setup, sweeps=0)
    assert seen == []


def test_quasi_simultaneous_sweeps_are_under_relaxed(monkeypatch):
    """The mass defects between sweeps are under-relaxed: with QS_RELAX = 1 the sweeps are the plain sweeps.

    The first two sweeps do not depend on the factor (the second sweep uses the mass defects of the first,
    the first starts from zero); the third sweep uses m2 + QS_RELAX (m2' - m2), so its state changes with it.
    """
    setup = _flap10_setup()
    original = vii.start_residual
    seen = {}
    for relax in (vii.QS_RELAX, 1.0, "arg"):
        rec = []

        def record(s, st, rec=rec):
            rec.append(st.x.copy())
            return original(s, st)

        monkeypatch.setattr(vii, "start_residual", record)
        if relax == "arg":
            # The argument relax = 1 of initial_state gives the same plain sweeps.
            monkeypatch.setattr(vii, "QS_RELAX", 0.5)
            vii.initial_state(setup, sweeps=3, relax=1.0)
        else:
            monkeypatch.setattr(vii, "QS_RELAX", relax)
            vii.initial_state(setup, sweeps=3)
        seen[relax] = rec
    monkeypatch.undo()
    assert vii.QS_RELAX == 0.5
    relaxed, plain = seen[0.5], seen[1.0]
    assert np.array_equal(relaxed[0], plain[0]) and np.array_equal(relaxed[1], plain[1])
    assert not np.array_equal(relaxed[2], plain[2])
    assert all(np.array_equal(a, b) for a, b in zip(seen["arg"], plain, strict=True))


def test_best_attempt_of_an_angle_is_a_length_weight_solve():
    """A solve with the mean weights at the first interior wake station is never the result of the angle."""
    ctl = vs._AngleWork(2.0, vs.StopRules())
    res = SimpleNamespace(status="max_iter", iterations=5, linear_solves=5, _first_stop=None, residual=0.2,
                          reason="maximum number of iterations")
    mean = SimpleNamespace(sys=SimpleNamespace(alpha=2.0), wake_weights="first_mean")
    ctl.add(mean, res, 50, 50)
    assert ctl.best is None and ctl.iterations == 5
    length = SimpleNamespace(sys=SimpleNamespace(alpha=2.0), wake_weights="length")
    worse = SimpleNamespace(**{**res.__dict__, "residual": 0.5})
    ctl.add(length, worse, 50, 50)
    assert ctl.best[0] is length


def test_wake_weight_continuation_gives_the_length_weight_solution(monkeypatch):
    """When the attempts with the length weights fail, the continuation in the wake weights solves the angle.

    The attempts with the length weights are made to fail here. The angle is then solved with the mean
    weights at the first interior wake station, and that solution starts the solve with the length
    weights. The result must be the direct solution with the length weights (to the Newton tolerance), and
    the reason names the continuation.
    """
    coords = airfoil("0012").repanel(n_panels=80).coords
    kw = dict(_KW, chord_ref=None, moment_ref=None, pivot=None, reference=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct, _ = vs.finish_angle(coords, 2.0, 1e6, None, 1e-6, 50, kw)
    assert direct["converged"] and vs.WEIGHT_CONTINUATION_NOTE not in direct["reason"]
    original = vs._cold_path
    weights = []

    def cold_path(body, alpha, re, tol, max_iter, kw, bk=None, ctl=None, cache=None):
        weights.append(kw.get("wake_weights", "length"))
        if kw.get("wake_weights", "length") == "length":
            setup = vs.prepare_case(body, alpha, re, **kw)
            return setup, vii.NewtonResult(None, None, False, 0, 1.0, "made to fail", 0, "max_iter")
        return original(body, alpha, re, tol, max_iter, kw, bk, ctl=ctl, cache=cache)

    monkeypatch.setattr(vs, "_cold_path", cold_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row, (setup, _) = vs.finish_angle(coords, 2.0, 1e6, None, 1e-6, 50, kw)
    assert weights == ["length", "first_mean"]
    assert row["converged"], row["reason"]
    assert vs.WEIGHT_CONTINUATION_NOTE in row["reason"]
    assert setup.wake_weights == "length"
    # Both are converged solutions of the same equations: they agree to the order of the Newton tolerance.
    assert abs(row["Cl"] - direct["Cl"]) < 1e-6
    assert abs(row["Cm"] - direct["Cm"]) < 1e-6
    assert abs(row["Cd"] / direct["Cd"] - 1.0) < 1e-5


def test_failed_wake_weight_continuation_keeps_the_failed_result(monkeypatch):
    """When the solve with the mean weights also fails, the angle fails with a note of the continuation."""
    coords = airfoil("0012").repanel(n_panels=80).coords
    kw = dict(_KW, chord_ref=None, moment_ref=None, pivot=None, reference=None)

    def cold_path(body, alpha, re, tol, max_iter, kw, bk=None, ctl=None, cache=None):
        setup = vs.prepare_case(body, alpha, re, **kw)
        return setup, vii.NewtonResult(None, None, False, 0, 1.0, "made to fail", 0, "max_iter")

    monkeypatch.setattr(vs, "_cold_path", cold_path)
    row, _ = vs.finish_angle(coords, 2.0, 1e6, None, 1e-6, 50, kw)
    assert not row["converged"]
    assert "wake-weight continuation: the solve with the mean weights" in row["reason"]


def test_mean_weight_solve_starts_from_the_plain_quasi_simultaneous_sweeps(monkeypatch):
    """A cold solve with the "first_mean" weights uses the plain sweeps of the former first estimate.

    Regression: with the under-relaxed sweeps in the mean-weight solves, the wake-weight continuation did
    not rescue a NACA 2412 with a 15 deg flap (robustness set 20 of 25 runs; with the plain sweeps 25 of
    25). The solves with the length weights keep the relaxed sweeps.
    """
    coords = airfoil("0012").repanel(n_panels=80).coords
    calls = []
    original = vii.initial_state

    def record(setup, sweeps=0, relax=None):
        calls.append((setup.wake_weights, sweeps, relax))
        return original(setup, sweeps=sweeps, relax=relax)

    monkeypatch.setattr(vii, "initial_state", record)
    monkeypatch.setattr(vs, "_solve_one", lambda *a, **k: vii.NewtonResult(None, None, False, 0, 1.0, "made to fail",
                                                                           0, "max_iter"))
    for weights in ("length", "first_mean"):
        setup = vs.prepare_case(coords, 2.0, 1e6, wake_weights=weights, **_KW)
        vs._cold_solve(setup, 1e-6, 50)
    qs = [c for c in calls if c[1] > 0]
    assert qs == [("length", vii.QS_SWEEPS, vii.QS_RELAX), ("first_mean", vii.QS_PLAIN_SWEEPS, 1.0)]
    assert vii.QS_PLAIN_SWEEPS == 3

