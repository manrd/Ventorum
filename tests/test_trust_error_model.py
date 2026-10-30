# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the calibrated error model of the trust rating.

The rating of the lifting line, the vortex lattice and the 3D panel method
comes from a predicted error ``k * E`` (``ventorum.core.trust.predicted_error``);
the envelope rules stay as caps. The expected values of the formulas below
are computed by hand from the published coefficients of the model.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.core import trust as tr
from ventorum.core.datatypes import FlightCondition
from ventorum.core.trust import evaluate_aerodynamic_trust

_RANK = {"HIGH": 3, "MODERATE": 2, "LOW": 1, "UNRELIABLE": 0}


# ----------------------------------------------------------------------------- formulas
@pytest.mark.parametrize("AR, sweep_deg, taper, expected", [
    (4.0, 0.0, 1.0, 0.13149095382310738),
    (8.0, 10.0, 0.5, 0.10297944920662531),
    (20.0, 30.0, 0.3, 0.16676983957859007),
])
def test_lifting_line_formula(AR, sweep_deg, taper, expected):
    """E = exp(-1.1) AR^-0.67 exp(0.063 sweep_qc_deg) exp(-0.82 (1 - taper))."""
    assert tr.lifting_line_error(AR, sweep_deg, taper) == pytest.approx(expected, rel=1e-12)
    e, k, _ = tr.predicted_error("linear", AR, sweep_qc_deg=sweep_deg, taper=taper)
    assert e == pytest.approx(expected, rel=1e-12)
    assert k == 1.3


@pytest.mark.parametrize("AR, nc, ns, sweep_deg, hc, controls, expected", [
    (8.0, 1, 24, 0.0, None, (), 0.00887148936456368),
    (6.0, 8, 16, 35.0, None, (), 0.005025927876233701),
    (10.0, 4, 20, 5.0, 0.2, ((math.radians(10.0), 0.75),), 0.04600738666647607),
])
def test_vortex_lattice_formula(AR, nc, ns, sweep_deg, hc, controls, expected):
    """E = chordwise + spanwise (+ ground) (+ control) terms of the vortex lattice."""
    terms = tr.vlm_error_terms(AR, nc, ns, math.radians(sweep_deg), hc, controls)
    assert sum(terms.values()) == pytest.approx(expected, rel=1e-12)
    assert ("ground" in terms) == (hc is not None)
    assert ("control" in terms) == bool(controls)
    e, k, _ = tr.predicted_error("vlm", AR, sweep_qc_deg=sweep_deg, n_chord=nc, n_span=ns, h_over_c=hc,
                                 controls=controls)
    assert e == pytest.approx(expected, rel=1e-12)
    assert k == 1.8


@pytest.mark.parametrize("nc, ns, expected", [
    (16, 12, 0.052491255599769346),
    (40, 20, 0.025087278061575635),
    (8, 40, 0.08479950654189354),
])
def test_panel3d_formula(nc, ns, expected):
    """E = exp(-0.46) n_chord^-1 + exp(-2.7) n_span^-0.66."""
    assert sum(tr.panel3d_error_terms(nc, ns).values()) == pytest.approx(expected, rel=1e-12)
    e, k, _ = tr.predicted_error("panel3d", 8.0, n_chord=nc, n_span=ns)
    assert e == pytest.approx(expected, rel=1e-12)
    assert k == 1.5


# ----------------------------------------------------------------------------- score function
def test_score_passes_through_the_rating_thresholds():
    """The score is 1 at 0.2 %, and the smallest score of each rating at the limit of its band."""
    assert tr.error_score(0.0) == 1.0
    assert tr.error_score(0.002) == 1.0
    assert tr.error_score(0.02) == pytest.approx(tr.RATING_SCORE_HIGH)
    assert tr.error_score(0.05) == pytest.approx(tr.RATING_SCORE_MODERATE)
    assert tr.error_score(0.15) == pytest.approx(tr.RATING_SCORE_LOW)
    assert tr.error_score(math.inf) == 0.0
    assert tr.error_score(100.0) == 0.0
    grid = np.geomspace(1e-4, 10.0, 2001)
    scores = np.array([tr.error_score(e) for e in grid])
    assert np.all(np.diff(scores) <= 0.0)
    assert np.all(np.diff(scores[(grid > 0.002) & (scores > 0.0)]) < 0.0)
    assert np.max(np.abs(np.diff(scores))) < 0.01  # continuous: no jump on a fine grid


@pytest.mark.parametrize("k_error", [0.001, 0.01, 0.019, 0.021, 0.04, 0.06, 0.12, 0.16, 0.5])
def test_rating_from_score_agrees_with_band(k_error):
    """The rating from the score is the band of k * E."""
    s = tr.error_score(k_error)
    rating = ("HIGH" if s >= tr.RATING_SCORE_HIGH else "MODERATE" if s >= tr.RATING_SCORE_MODERATE
              else "LOW" if s >= tr.RATING_SCORE_LOW else "UNRELIABLE")
    assert rating == tr.error_band(k_error)


@pytest.mark.parametrize("n_chord", [2, 8, 30, 60, 200])
def test_rating_of_a_result_without_cap_is_the_band(n_chord):
    """Without a cap the score of a result is the score of its predicted error."""
    t = evaluate_aerodynamic_trust(AR=8.0, CL=0.4, CDi=0.006, solver_type="panel3d", n_chord=n_chord,
                                   n_panels=24, n_span=None)
    # n_span None takes n_panels: compare with the full model of the result.
    e, k, _ = tr.predicted_error("panel3d", 8.0, n_chord=n_chord, n_span=24)
    assert t.score == pytest.approx(tr.error_score(k * e))
    assert t.rating == tr.error_band(k * e)


# ----------------------------------------------------------------------------- ratings of solves
def _rect(semi_span, chord=1.0, controls=None):
    return vt.LiftingSurface(name="Wing", semi_span=semi_span, controls=controls or [],
                             sections=[vt.WingSection(y_frac=0.0, chord=chord),
                                       vt.WingSection(y_frac=1.0, chord=chord)])


def test_lifting_line_ar4_rectangular_is_not_high():
    """A rectangular AR 4 wing on the lifting line (true error about 11 %) is not HIGH."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.analyze(_rect(2.0), alpha_deg=4.0, V_inf=30.0, solver="linear", n_panels=40)
    t = res.totals.trust
    assert res.totals.AR == pytest.approx(4.0)
    assert t.rating != "HIGH"
    assert t.factors["error_model"] == pytest.approx(1.0 - tr.error_score(1.3 * 0.13149095382310738), rel=1e-9)
    assert any("Predicted error about 17.1 %" in w for w in t.warnings)
    assert any("VLM" in r for r in t.recommendations)


def test_vlm_ar8_single_chordwise_panel_is_not_high_above_two_percent():
    """A VLM AR 8 wing with nc 1 is not HIGH when the predicted error E is above 2 %."""
    sweep = math.radians(50.0)
    e, k, _ = tr.predicted_error("vlm", 8.0, sweep_qc_deg=50.0, n_chord=1, n_span=8)
    assert e > 0.02
    t = evaluate_aerodynamic_trust(AR=8.0, CL=0.4, CDi=0.006, solver_type="vlm", n_panels=8, n_chord=1,
                                   max_sweep_rad=sweep)
    assert t.rating != "HIGH"
    assert t.rating == tr.error_band(k * e)
    # The solve: a rectangular AR 8 wing with nc 1 and few strips at 50 deg sweep.
    wing = vt.LiftingSurface(name="Wing", semi_span=4.0, sweep_le=sweep,
                             sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)])
    res = vt.analyze(wing, condition=FlightCondition(V_inf=30.0, alpha=math.radians(4.0)),
                     settings=vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1))
    lat = res.details["lattice"]
    e_solve = tr.predicted_error("vlm", res.totals.AR, sweep_qc_deg=res.details["sweep_deg"], n_chord=1,
                                 **{n: v for n, v in tr.lattice_error_inputs(lat).items() if n != "taper"})[0]
    assert e_solve > 0.02
    assert res.totals.trust.rating != "HIGH"


@pytest.mark.parametrize("solver", ["linear", "vlm", "panel3d"])
def test_score_is_continuous_at_the_former_thresholds(solver):
    """AR 2.9999 and 3.0001 give scores within 1e-3 (the former lifting-line rule jumped at AR 3)."""
    kw = dict(CL=0.4, CDi=0.006, solver_type=solver, n_panels=24, n_chord=8)
    a = evaluate_aerodynamic_trust(AR=2.9999, **kw).score
    b = evaluate_aerodynamic_trust(AR=3.0001, **kw).score
    assert abs(a - b) < 1e-3
    # Also at the former sweep threshold of the lifting line (2.5 deg) and the coarse-mesh rule (8 strips).
    s1 = evaluate_aerodynamic_trust(AR=8.0, max_sweep_rad=math.radians(2.4999), **kw).score
    s2 = evaluate_aerodynamic_trust(AR=8.0, max_sweep_rad=math.radians(2.5001), **kw).score
    assert abs(s1 - s2) < 1e-3
    kw.pop("n_panels")
    n1 = evaluate_aerodynamic_trust(AR=8.0, n_panels=8, n_span=7.9999, **kw).score
    n2 = evaluate_aerodynamic_trust(AR=8.0, n_panels=8, n_span=8.0001, **kw).score
    assert abs(n1 - n2) < 1e-3


def test_ground_clearance_and_deflection_are_continuous():
    """The predicted error is continuous in h/c and in the deflection."""
    kw = dict(AR=8.0, CL=0.4, CDi=0.006, solver_type="vlm", n_panels=24, n_chord=8)
    for hc in (0.1, 0.3, 0.5):
        a = evaluate_aerodynamic_trust(h_over_c=hc - 1e-6, **kw)
        b = evaluate_aerodynamic_trust(h_over_c=hc + 1e-6, **kw)
        assert abs(a.factors["error_model"] - b.factors["error_model"]) < 1e-3
    zero = evaluate_aerodynamic_trust(controls=[(0.0, 0.75)], **kw).factors["error_model"]
    small = evaluate_aerodynamic_trust(controls=[(1e-6, 0.75)], **kw).factors["error_model"]
    assert abs(zero - small) < 1e-3


# ----------------------------------------------------------------------------- caps
_COND = FlightCondition(V_inf=30.0, alpha=math.radians(4.0))
_BASE = dict(AR=8.0, CL=0.4, CDi=0.0065, solver_type="vlm", n_panels=24, n_chord=8, condition=_COND)
_STATS = {"max_local_cl": 0.5, "max_alpha_eff_deg": 4.0, "sign_change_run": 0}
_POLAR = {"n_strips": 24, "n_tabulated": 24, "n_out_of_table": 0, "max_excess_deg": 0.0, "n_past_stall": 0,
          "stall_ratio": 0.5, "peak_cl": 0.5, "cl_max_at_peak": 1.0, "all_with_stall": True}


def _panel_stats(**kw):
    stats = {"max_aspect_ratio": 3.0, "max_skew": 0.1, "max_warp": 0.0, "watertight": True, "n_nonconvex": 0}
    stats.update(kw)
    return stats


_CAPS = {
    "stall (fixed limits)": (dict(spanwise_stats={**_STATS, "max_local_cl": 1.7}), "stall_proximity"),
    "stall (polar)": (dict(polar={**_POLAR, "n_past_stall": 3, "stall_ratio": 1.1}), "stall_proximity"),
    "polar range": (dict(polar={**_POLAR, "n_out_of_table": 2, "max_excess_deg": 3.0}), "polar_range"),
    "side edge": (dict(AR=1.5, condition=FlightCondition(V_inf=30.0, alpha=math.radians(15.0))),
                  "side_edge_vortex"),
    "Mach": (dict(condition=FlightCondition(V_inf=150.0, alpha=math.radians(4.0))), "compressibility"),
    "non-finite": (dict(CL=float("nan")), "non_finite_result"),
    "non-convergence": (dict(converged=False), "numerical"),
    "saw-tooth": (dict(spanwise_stats={**_STATS, "sign_change_run": 5}), "numerical"),
    "small h/c (thickness, viscous)": (dict(h_over_c=0.25, n_chord=64), "ground_proximity"),
    "very small h/c": (dict(h_over_c=0.05, n_chord=64), "ground_proximity"),
    "post-stall branches": (dict(post_stall_branch=True), "post_stall_branch"),
    "non-positive AR": (dict(AR=0.0), "aspect_ratio"),
}


@pytest.mark.parametrize("name", list(_CAPS))
def test_every_cap_rule_lowers_the_rating(name):
    """Each envelope rule that stays as a cap lowers a HIGH rating of the error model."""
    change, factor = _CAPS[name]
    base_kw = dict(_BASE, spanwise_stats=_STATS)
    if "polar" in change:
        base_kw["polar"] = _POLAR
    if "n_chord" in change:
        base_kw["n_chord"] = change["n_chord"]
    if name == "side edge":
        base_kw["AR"] = change["AR"]
    base = evaluate_aerodynamic_trust(**base_kw)
    assert base.rating == "HIGH", (name, base.score, base.warnings)
    capped = evaluate_aerodynamic_trust(**{**base_kw, **change})
    assert capped.factors[factor] > 0.0
    assert _RANK[capped.rating] < _RANK[base.rating], (name, capped.rating)
    # The cap does not change the error model (the ground term at n_chord 64 is below 1e-3).
    if name != "non-positive AR":
        assert capped.factors["error_model"] == pytest.approx(base.factors["error_model"], abs=1e-3)


@pytest.mark.parametrize("change, factor", [
    (dict(mesh_stats=_panel_stats(n_nonconvex=2)), "mesh_adequacy"),
    (dict(mesh_stats=_panel_stats(watertight=False)), "mesh_adequacy"),
    (dict(mesh_stats=_panel_stats(max_warp=0.5)), "mesh_adequacy"),
    (dict(CD_nearfield=0.0065 * 1.6), "mesh_resolution"),
])
def test_panel3d_cap_rules_lower_the_rating(change, factor):
    """The panel validity rules and the near-field/far-field drag rule of the 3D panel method stay caps."""
    kw = dict(AR=8.0, CL=0.4, CDi=0.0065, solver_type="panel3d", n_panels=100, n_chord=100, condition=_COND,
              mesh_stats=_panel_stats())
    base = evaluate_aerodynamic_trust(**kw)
    assert base.rating == "HIGH"
    capped = evaluate_aerodynamic_trust(**{**kw, **change})
    assert capped.factors[factor] > 0.0
    assert _RANK[capped.rating] < _RANK[base.rating]


def test_cap_does_not_raise_a_low_model_rating():
    """A cap is a limit: a result of the error model rated LOW stays LOW under a weaker cap."""
    kw = dict(AR=8.0, CL=0.4, CDi=0.0065, solver_type="linear", n_panels=24, condition=_COND)
    plain = evaluate_aerodynamic_trust(**kw)
    capped = evaluate_aerodynamic_trust(post_stall_branch=True, **kw)
    assert plain.rating == "LOW"
    assert capped.score == pytest.approx(min(plain.score, 1.0 - tr.POST_STALL_BRANCH_PENALTY))


# ----------------------------------------------------------------------------- messages and inputs
def test_dominant_term_gives_the_recommendation():
    """The recommendation follows the term that dominates the predicted error."""
    kw = dict(CL=0.4, CDi=0.006, solver_type="vlm", condition=_COND)
    chord = evaluate_aerodynamic_trust(AR=2.0, n_panels=40, n_chord=1, **kw)
    assert any("chordwise term dominates" in r for r in chord.recommendations)
    span = evaluate_aerodynamic_trust(AR=8.0, n_panels=4, n_chord=8, max_sweep_rad=math.radians(45.0), **kw)
    assert any("spanwise term dominates" in r for r in span.recommendations)
    ground = evaluate_aerodynamic_trust(AR=8.0, n_panels=40, n_chord=2, h_over_c=0.05, **kw)
    assert "Increase n_chord so that c/n_chord < h_min." in ground.recommendations
    p3 = evaluate_aerodynamic_trust(AR=8.0, CL=0.4, CDi=0.006, solver_type="panel3d", n_panels=40, n_chord=8)
    assert any(r.startswith("Increase n_chord: the chordwise panel count controls") for r in p3.recommendations)


def test_slender_wing_rule_is_replaced():
    """The slender-wing rule (mean sweep above 55 deg at AR below 2.5) is replaced by the error model."""
    t = evaluate_aerodynamic_trust(AR=1.5, CL=0.3, CDi=0.02, solver_type="vlm", n_panels=24, n_chord=8,
                                   max_sweep_rad=math.radians(60.0))
    assert not any("Slender wing" in w for w in t.warnings)
    assert "sweep" not in t.factors
    e, k, terms = tr.predicted_error("vlm", 1.5, sweep_qc_deg=60.0, n_chord=8, n_span=24)
    assert t.score == pytest.approx(tr.error_score(k * e))


def test_lattice_error_inputs_of_a_solve():
    """The lattice gives the taper, the smallest strip count and the deflected chordwise controls."""
    flap = vt.ControlSurface("flap", eta_start=0.2, eta_end=0.8, hinge_x_c=0.7, deflection=math.radians(-8.0))
    wing = vt.LiftingSurface(name="Wing", semi_span=4.0, controls=[flap],
                             sections=[vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.6)])
    tail = vt.LiftingSurface(name="Tail", semi_span=1.2, position=np.array([4.0, 0.0, 0.3]), n_panels=6,
                             sections=[vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.4)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.analyze(vt.Aircraft(surfaces=[wing, tail]), condition=_COND,
                         settings=vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=10))
    inp = tr.lattice_error_inputs(res.details["lattice"])
    assert inp["taper"] == pytest.approx(0.5)
    assert inp["n_span"] == 6
    assert inp["controls"] == [(pytest.approx(math.radians(8.0)), pytest.approx(0.7))]
    e, k, terms = tr.predicted_error("vlm", res.totals.AR, sweep_qc_deg=res.details["sweep_deg"], n_chord=10,
                                     n_span=6, controls=inp["controls"])
    assert "control" in terms
    assert res.totals.trust.factors["error_model"] == pytest.approx(1.0 - tr.error_score(k * e))


def test_fourier_uses_the_lifting_line_model():
    """The Fourier lifting line gets the lifting-line error model with the taper of the surface."""
    res = vt.analyze(_rect(4.0), alpha_deg=4.0, V_inf=30.0, solver="fourier", n_panels=40)
    e = tr.lifting_line_error(8.0, 0.0, 1.0)
    assert res.totals.trust.score == pytest.approx(tr.error_score(1.3 * e))
    assert res.totals.trust.rating == "LOW"


# ----------------------------------------------------------------------------- GPU and CPU
@pytest.fixture
def gpu_device():
    """Run the test with the GPU device selected; restore the settings after it."""
    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    old_dev, old_prec = gpu.get_device(), gpu._precision
    yield
    gpu.set_device(old_dev)
    gpu.set_precision(old_prec)


@pytest.mark.gpu
@pytest.mark.parametrize("solver_cls, n_chord", [("VortexLatticeSolver", 8), ("LinearSolver", 1)])
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_gpu_and_cpu_give_the_same_trust(gpu_device, solver_cls, n_chord, precision):
    """The GPU and the CPU paths give the same rating, error model and messages."""
    from ventorum.solvers.horseshoe import VortexLatticeSolver
    from ventorum.solvers.linear import LinearSolver

    flap = vt.ControlSurface("flap", eta_start=0.1, eta_end=0.7, hinge_x_c=0.75, deflection=math.radians(6.0))
    wing = vt.LiftingSurface(name="Wing", semi_span=4.0, sweep_le=math.radians(10.0), controls=[flap],
                             sections=[vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.6)])
    ac = vt.Aircraft(surfaces=[wing])
    st = vt.SolverSettings(solver_type="vlm" if n_chord > 1 else "linear", n_panels=16, n_chord=n_chord)
    alphas = np.radians([-2.0, 3.0, 8.0])
    solver = {"VortexLatticeSolver": VortexLatticeSolver, "LinearSolver": LinearSolver}[solver_cls]()
    out = {}
    for device in ("cpu", "gpu"):
        gpu.set_device(device)
        gpu.set_precision(precision)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[device] = solver.solve_sweep(ac, FlightCondition(V_inf=25.0, h=1.5), st, alphas)
    for c, g in zip(out["cpu"], out["gpu"]):
        assert g.details.get("device") == "gpu"
        tc, tg = c.totals.trust, g.totals.trust
        assert tc.rating == tg.rating
        assert tc.factors["error_model"] == tg.factors["error_model"]
        assert [w for w in tc.warnings if "Predicted error" in w] == [w for w in tg.warnings if "Predicted error" in w]
        assert tc.score == pytest.approx(tg.score, abs=1e-6)
