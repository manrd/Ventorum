# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the transition rule of the viscous solver: start independence, the tie band and the rule note."""

from __future__ import annotations

import math
import warnings

import pytest

from ventorum.airfoil import airfoil
from ventorum.panel2d import vii
from ventorum.panel2d.viscous import (_cold_solve, _solve_one, prepare_case, resolve_body, transition_rule_note,
                                      warm_state, zero_lift_angle)
from ventorum.utils.parallel import blas_single_thread

KW0 = dict(ncrit=9.0, xtr=(1.0, 1.0), height=None, height_reference="pivot", wake_length=1.0, n_wake=30)


def _case(closed_te, alpha, mode):
    """Return (setup, result) of a cold start at the angle ("cold") or two warm steps from zero lift ("path")."""
    body, chord, moment, pivot = resolve_body(airfoil("0012", closed_te=closed_te).repanel(n_panels=160))
    kw = dict(KW0, chord_ref=chord, moment_ref=moment, pivot=pivot)
    with warnings.catch_warnings(), blas_single_thread():
        warnings.simplefilter("ignore")
        if mode == "cold":
            s = prepare_case(body, alpha, 1e6, **kw)
            return s, _cold_solve(s, 1e-6, 50)
        a0 = zero_lift_angle(body, kw)
        s = prepare_case(body, a0, 1e6, **kw)
        r = _cold_solve(s, 1e-6, 50)
        for k in (1, 2):
            s2 = prepare_case(body, a0 + (alpha - a0) * k / 2, 1e6, **kw)
            r = _solve_one(s2, warm_state(s2, s, r.state), 1e-6, 50, 1.0)
            s = s2
        return s, r


def _rule_intervals(setup, state):
    """Return the transition intervals that the structure update gives for a copy of a state."""
    t = state.copy()
    vii._update_structure(setup, t)
    return tuple(int(v) for v in t.itr)


def test_transition_rule_note():
    """The note appears only for a fraction outside [-EPS_W, 1 + EPS_W]; NaN (no transition) is ignored."""
    e = vii.EPS_W
    assert transition_rule_note((0.0, 1.0)) is None
    assert transition_rule_note((-0.5 * e, 1.0 + 0.5 * e)) is None
    assert transition_rule_note((math.nan, math.nan)) is None
    note = transition_rule_note((0.3, -4.9))
    assert note.startswith("transition rule:")
    assert "lower side w = -4.9" in note and "upper" not in note
    assert "upper side" in transition_rule_note((1.0 + 2.0 * e, 0.5))


@pytest.mark.slow
@pytest.mark.parametrize("closed_te", [False, True])
def test_start_independent_and_consistent_with_the_transition_rule(closed_te):
    """Regression: the continuation path and the cold start give one solution (NACA 0012, 2 deg, Re 1e6).

    Before, a cycle of the transition interval far from convergence froze the side for the rest of
    the solve. The cold start then held the upper side one interval downstream of the transition
    rule, and the loads differed from the continuation path by up to 1.6e-4 in Cl and 9.4e-4 in Cd
    (relative). Now the intervals follow the iterate: both starts end in the same state, and that
    state satisfies the transition rule.
    """
    out = {}
    for mode in ("path", "cold"):
        s, r = _case(closed_te, 2.0, mode)
        assert r.converged, r.reason
        assert _rule_intervals(s, r.state) == tuple(int(v) for v in r.state.itr)
        o = vii.case_outputs(s, r.state, r.ue)
        assert all(-vii.EPS_W <= w <= 1.0 + vii.EPS_W for w in o["transition_fraction"])
        out[mode] = (o["Cl"], o["Cd"], o["Cm"])
    (l1, d1, m1), (l2, d2, m2) = out["path"], out["cold"]
    assert abs(l1 - l2) < 1e-6 * abs(l2)
    assert abs(d1 - d2) < 1e-6 * d2
    assert abs(m1 - m2) < 1e-7


@pytest.mark.slow
def test_tie_band_keeps_the_current_interval():
    """At w = 1 + EPS_W / 2 the rule keeps the interval; at w = 1 + 3 EPS_W it moves it downstream."""
    s, r = _case(False, 2.0, "cold")
    assert r.converged, r.reason
    up, _ = vii._sides(s, r.state)
    itr = r.state.itr[0]
    g = up[itr]
    ue, ds, _ = vii.host_edge(s, r.state)
    dxi, _ = vii._side_geometry(s, up)
    rate = float(vii._laminar_rate(s, r.state.x[g, 0], ds[g], ue[g]))
    assert rate > 0.0
    for w, moves in ((1.0 + 0.5 * vii.EPS_W, False), (1.0 + 3.0 * vii.EPS_W, True)):
        t = r.state.copy()
        t.x[g, 2] = s.ncrit - w * dxi[itr] * rate
        vii._update_structure(s, t)
        assert (t.itr[0] > itr) is moves
