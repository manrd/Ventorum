# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Unit tests for the aerodynamic trust scoring and uncertainty engine in Ventorum.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

import ventorum as vt
from ventorum.core.datatypes import FlightCondition, LiftingSurface, WingSection
from ventorum.core.trust import (
    TrustScore,
    _count_extrema,
    _count_extrema_rows,
    evaluate_aerodynamic_trust,
    spanwise_stats_batch,
)


def test_trust_score_high_aspect_ratio_straight_wing():
    """Verify high-AR straight wing achieves HIGH trust score."""
    wing = LiftingSurface(
        name="GliderWing",
        semi_span=10.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
        sweep_le=0.0,
        dihedral=0.0,
    )
    result = vt.analyze(wing, alpha_deg=4.0, V_inf=30.0)

    trust = result.totals.trust
    assert trust is not None
    assert isinstance(trust, TrustScore)
    assert trust.score >= 0.85
    assert trust.rating == "HIGH"
    assert trust.factors["aspect_ratio"] == 0.0
    assert trust.factors["sweep"] == 0.0
    assert trust.factors["compressibility"] == 0.0
    assert trust.uncertainty_CL > 0.0
    assert trust.uncertainty_CDi > 0.0
    assert "Trust:" in trust.summary_str()


def test_trust_score_low_aspect_ratio_penalty():
    """Verify low-AR wing incurs aspect ratio penalty and warning."""
    wing = LiftingSurface(
        name="LowARWing",
        semi_span=1.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    # AR = b^2 / S = 4.0 / 2.0 = 2.0: a lifting-line method is penalised...
    result = vt.analyze(wing, alpha_deg=4.0, V_inf=30.0, solver="linear")

    trust = result.totals.trust
    assert trust is not None
    assert trust.factors["aspect_ratio"] > 0.40
    assert trust.score < 0.65
    assert any("low aspect ratio" in w.lower() for w in trust.warnings)
    assert any("VLM" in r for r in trust.recommendations)

    # ...the vortex-lattice method (the default) is not.
    vlm = vt.analyze(wing, alpha_deg=4.0, V_inf=30.0).totals.trust
    assert vlm.factors["aspect_ratio"] == 0.0


def test_trust_score_sweep_angle_penalty():
    """Verify swept wing incurs sweep penalty and warning."""
    wing = LiftingSurface(
        name="SweptWing",
        semi_span=6.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.5),
            WingSection(y_frac=1.0, chord=0.6),
        ],
        sweep_le=np.radians(35.0),
    )
    with pytest.warns(RuntimeWarning):
        result = vt.analyze(wing, alpha_deg=4.0, V_inf=40.0, solver="linear")

    trust = result.totals.trust
    assert trust is not None
    assert trust.factors["sweep"] > 0.45
    assert any("sweep angle" in w.lower() for w in trust.warnings)

    # The vortex-lattice method handles sweep: no penalty.
    vlm = vt.analyze(wing, alpha_deg=4.0, V_inf=40.0).totals.trust
    assert vlm.factors["sweep"] == 0.0


def test_trust_score_high_alpha_stall_proximity():
    """Verify high angle of attack triggers stall proximity penalty."""
    wing = LiftingSurface(
        name="StalledWing",
        semi_span=5.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    # At alpha = 18 deg, linear lift exceeds 1.8
    result = vt.analyze(wing, alpha_deg=18.0, V_inf=40.0)

    trust = result.totals.trust
    assert trust is not None
    assert trust.factors["stall_proximity"] > 0.40
    assert any("stall" in w.lower() for w in trust.warnings)


def test_trust_score_compressibility_penalty():
    """Above Mach 0.3 the case is out of the valid envelope: warning and low trust."""
    wing = LiftingSurface(
        name="FastWing",
        semi_span=5.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    # V_inf = 240 m/s -> Mach ~ 0.70
    with pytest.warns(RuntimeWarning, match="out of the valid envelope"):
        result = vt.analyze(wing, alpha_deg=2.0, V_inf=240.0)

    trust = result.totals.trust
    assert trust is not None
    assert trust.factors["compressibility"] > 0.50
    assert trust.rating in ("LOW", "UNRELIABLE")
    assert any("out of the valid envelope" in w.lower() for w in trust.warnings)
    # Mach 0.25 is inside the envelope: no compressibility penalty.
    assert vt.analyze(wing, alpha_deg=2.0, V_inf=85.0).totals.trust.factors["compressibility"] == 0.0


def test_trust_score_ground_proximity_penalty():
    """Verify low ground clearance incurs proximity penalty."""
    wing = LiftingSurface(
        name="GroundEffectWing",
        semi_span=10.0,  # span = 20.0 m
        sections=[
            WingSection(y_frac=0.0, chord=2.0),
            WingSection(y_frac=1.0, chord=2.0),
        ],
    )
    # h = 0.3 m at the reference point (the origin, root leading edge); the trailing edge is lower:
    # h_min = 0.3 - 2 sin(4 deg) = 0.16 m, h_min/c = 0.08 (< 0.1).
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(4.0), h=0.3)
    # Few panels: the trust factor depends on the geometry, not on the mesh.
    result = vt.analyze(wing, condition=cond, settings=vt.SolverSettings(n_panels=16, n_chord=4))

    trust = result.totals.trust
    assert trust is not None
    assert trust.factors["ground_proximity"] > 0.35
    assert any("ground proximity" in w.lower() for w in trust.warnings)


def test_trust_score_serialization():
    """Verify TrustScore to_dict method produces complete JSON structure."""
    trust = evaluate_aerodynamic_trust(
        AR=8.0,
        CL=0.5,
        CDi=0.012,
        max_sweep_rad=0.0,
    )
    d = trust.to_dict()
    assert "score" in d
    assert "rating" in d
    assert "uncertainty_CL" in d
    assert "uncertainty_CDi" in d
    assert "factors" in d
    assert isinstance(d["factors"], dict)
    assert isinstance(d["warnings"], list)


def test_trust_evaluation_is_cheap():
    """Trust evaluation is cheap compared with a solve. The limit (5 ms per call)
    is far above the measured value, so a loaded test machine does not fail it."""
    t0 = time.perf_counter()
    n_runs = 200
    for _ in range(n_runs):
        _ = evaluate_aerodynamic_trust(
            AR=7.5,
            CL=0.45,
            CDi=0.011,
            max_sweep_rad=np.radians(10.0),
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / n_runs
    assert elapsed_ms < 5.0, f"Trust evaluation too slow: {elapsed_ms:.4f} ms per call"


def test_extrema_count_of_rows_equals_the_count_per_row():
    rng = np.random.default_rng(3)
    for _ in range(300):
        m = int(rng.integers(1, 25))
        G = rng.normal(size=(4, m)).cumsum(axis=1)
        if rng.random() < 0.5:
            G = np.round(G, 1)
        assert np.array_equal(_count_extrema_rows(G), [_count_extrema(g) for g in G])


def test_batch_trust_statistics_give_the_same_trust_score():
    wing = LiftingSurface(name="wing", semi_span=5.0, sweep_le=np.radians(2.0),
                          sections=[WingSection(y_frac=0.0, chord=1.25), WingSection(y_frac=1.0, chord=0.9)])
    tail = LiftingSurface(name="tail", semi_span=1.5, position=np.array([3.5, 0.0, 0.5]),
                          sections=[WingSection(y_frac=0.0, chord=0.6), WingSection(y_frac=1.0, chord=0.4)])
    ac = vt.Aircraft(surfaces=[wing, tail])
    st = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=2)
    res = vt.HorseshoeSolver().solve_sweep(ac, FlightCondition(V_inf=25.0), st, np.radians([-2.0, 4.0, 14.0]))
    lat = res[0].details["lattice"]
    arrays = {}
    for name in ("gamma", "Cl", "Cd_i", "alpha_eff", "alpha_i", "local_lift", "Cm_section"):
        arrays[name] = np.array([np.concatenate([getattr(sw, name) for sw in r.spanwise]) for r in res])
    stats = spanwise_stats_batch([surf.strips for surf in lat.surfaces], arrays)
    for r, s in zip(res, stats):
        t = r.totals
        kw = dict(AR=t.AR, condition=r.condition, CL=t.CL, CDi=t.CDi, CD_total=t.CD_total, n_panels=12,
                  solver_type="vlm", n_chord=2)
        a = evaluate_aerodynamic_trust(spanwise_list=r.spanwise, **kw)
        b = evaluate_aerodynamic_trust(spanwise_stats=s, **kw)
        assert a.to_dict() == b.to_dict()
