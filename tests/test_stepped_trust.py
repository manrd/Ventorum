# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Trust of the stepped cases of the derivative tools.

The derivative tools report the trust of the base case. A stepped case on
the other side of a trust threshold (here the ground proximity limit
h_min/c = 0.1) has a lower trust; the tools give it in the field
``stepped_trust_min`` and in a note, and keep the base trust.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.datatypes import TrustScore
from ventorum.core.trust import stepped_trust_minimum, stepped_trust_note


def _fake(rating, score):
    return SimpleNamespace(totals=SimpleNamespace(trust=TrustScore(score=score, rating=rating)))


def test_stepped_trust_minimum_picks_the_lowest_case():
    res = [_fake("HIGH", 1.0), _fake("HIGH", 1.0), _fake("MODERATE", 0.8), _fake("LOW", 0.55), _fake("LOW", 0.6)]
    info = stepped_trust_minimum(res, ["base", "a +", "a -", "h +", "h -"])
    assert info == {"case": "h +", "rating": "LOW", "score": 0.55, "base_rating": "HIGH", "base_score": 1.0}
    note = stepped_trust_note(info)
    assert "'h +'" in note and "base case" in note and "stepped_trust_min" in note


def test_stepped_trust_minimum_none_when_no_case_is_lower():
    res = [_fake("MODERATE", 0.8), _fake("MODERATE", 0.8), _fake("HIGH", 1.0)]
    assert stepped_trust_minimum(res, ["base", "x +", "x -"]) is None
    # A lower score at the same rating counts.
    res = [_fake("HIGH", 1.0), _fake("HIGH", 0.9)]
    assert stepped_trust_minimum(res, ["base", "x +"])["case"] == "x +"
    # No trust on the base case: nothing to compare.
    assert stepped_trust_minimum([SimpleNamespace(totals=SimpleNamespace(trust=None)), _fake("LOW", 0.5)],
                                 ["base", "x +"]) is None


def _wing_pivot_at_te():
    """Rectangular wing; the pivot at the trailing edge, so the alpha steps do not move the lowest point."""
    w = vt.LiftingSurface(name="w", semi_span=3.0, n_panels=12,
                          sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)])
    ac = vt.Aircraft(name="r", surfaces=[w], ref_point=np.array([1.0, 0.0, 0.0]))
    return ac, vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=8)


# Base clearance 0.102 c (above the limit 0.1); the h- step (5 % of the clearance) goes below it.
_H = 0.102


def test_height_derivatives_names_the_h_minus_case_below_the_proximity_limit():
    ac, st = _wing_pivot_at_te()
    fc = vt.FlightCondition(V_inf=50.0, alpha=math.radians(4.0), h=_H)
    res = vt.height_derivatives(ac, fc, st, phi_step_deg=0.01)
    assert res.h_min_m / res.c_ref == pytest.approx(_H, abs=1e-6)
    info = res.stepped_trust_min
    assert info is not None and info["case"].startswith("h -")
    base_trust = res.base.totals.trust
    # The base trust stays the result trust, and it is the trust of a single solve of the base state.
    single = vt.analyze(ac, fc, st.clone()).totals.trust
    assert (base_trust.rating, base_trust.score) == (single.rating, single.score)
    assert (info["base_rating"], info["base_score"]) == (base_trust.rating, base_trust.score)
    assert info["score"] < base_trust.score
    assert any("lower trust" in n and info["case"] in n for n in res.notes)
    assert res.to_dict()["stepped_trust_min"] == info

    # The alpha and beta steps about the trailing edge keep the clearance: no stepped case is lower.
    s_res = vt.static_derivatives(ac, fc, st)
    assert s_res.stepped_trust_min is None
    assert not any("lower trust" in n for n in s_res.notes)
    assert s_res.to_dict()["stepped_trust_min"] is None


def test_derivative_matrix_gives_the_lowest_stepped_case_of_its_groups():
    ac, st = _wing_pivot_at_te()
    fc = vt.FlightCondition(V_inf=50.0, alpha=math.radians(4.0), h=_H)
    dm = vt.derivative_matrix(ac, fc, st, groups=("static", "height"))
    # derivative_matrix uses the fixed bank step 0.5 deg: the bank cases lower a tip by more than the h step.
    info = dm.stepped_trust_min
    assert info is not None and info["group"] == "height"
    assert info["score"] < dm.base.totals.trust.score
    assert dm.to_dict()["stepped_trust_min"] == info
    every = vt.derivative_matrix(ac, fc, st, groups=("static", "height"), axes="all")
    assert every.stepped_trust_min == info
    assert every.results["body"].stepped_trust_min == info


def test_control_derivatives_name_the_flap_down_case_below_the_proximity_limit():
    """A trailing-edge flap step of +1 deg lowers the trailing edge below h_min/c 0.1; -1 deg does not."""
    flap = vt.ControlSurface("flap", 0.0, 1.0, 0.7, 0.0)
    w = vt.LiftingSurface(name="w", semi_span=3.0, n_panels=12, controls=[flap],
                          sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)])
    ac = vt.Aircraft(name="r", surfaces=[w], ref_point=np.array([1.0, 0.0, 0.0]))
    st = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=10)
    fc = vt.FlightCondition(V_inf=50.0, alpha=math.radians(4.0), h=_H)
    res = vt.control_derivatives(ac, fc, st, hinge_alpha=False)
    info = res.stepped_trust_min
    assert info is not None and info["case"] == "flap +"
    assert info["score"] < res.base.totals.trust.score
    assert any("lower trust" in n and "'flap +'" in n for n in res.notes)
    assert res.to_dict()["stepped_trust_min"] == info


def test_free_air_derivatives_have_no_stepped_trust_note():
    ac, st = _wing_pivot_at_te()
    fc = vt.FlightCondition(V_inf=50.0, alpha=math.radians(4.0))
    assert vt.static_derivatives(ac, fc, st).stepped_trust_min is None
    assert vt.rate_derivatives(ac, fc, st).stepped_trust_min is None
