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
    res = vt.VortexLatticeSolver().solve_sweep(ac, FlightCondition(V_inf=25.0), st, np.radians([-2.0, 4.0, 14.0]))
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


# ---------------------------------------------------------------------------
# Review round 5 (A3 to A6): polar limits, side-edge vortex lift, ISA Mach
# ---------------------------------------------------------------------------

def _tab_wing(tab):
    return vt.LiftingSurface(semi_span=4.0, sections=[vt.WingSection(y_frac=0.0, chord=1.0, airfoil=tab),
                                                      vt.WingSection(y_frac=1.0, chord=1.0, airfoil=tab)])


def _stall_polar():
    """Polar with the stall at 8 deg (Cl_max about 0.88), below the fixed limits (Cl 1.55, 16 deg)."""
    a = np.radians(np.arange(-8.0, 25.1, 1.0))
    st_a = np.radians(8.0)
    cl = np.where(a < st_a, 2 * np.pi * a, 2 * np.pi * st_a - 1.5 * (a - st_a))
    return vt.TabulatedAirfoil(name="lowRe", alpha=a, Cl_data=cl, Cd_data=0.01 + 0.05 * np.maximum(a - st_a, 0))


def test_polar_out_of_range_warns():
    """A section angle outside the polar table gives a warning and a lower score (A3).

    The polar covers -4 to +4 deg only. At alpha 10 deg the nonlinear
    lifting line uses the end values of the table: CL 0.43 against 0.84
    with the full linear section. The trust must not stay HIGH.
    """
    a = np.radians(np.arange(-4.0, 4.01, 0.5))
    tab = vt.TabulatedAirfoil(name="short", alpha=a, Cl_data=2 * np.pi * a, Cd_data=0.008 + 0 * a)
    inside = vt.analyze(_tab_wing(tab), alpha_deg=3.0, solver="nonlinear", n_panels=30).totals.trust
    outside = vt.analyze(_tab_wing(tab), alpha_deg=10.0, solver="nonlinear", n_panels=30).totals.trust
    assert inside.rating == "HIGH"
    assert not any("Outside the polar data" in w for w in inside.warnings)
    assert outside.rating != "HIGH"
    assert outside.factors["polar_range"] > 0.0
    assert any("Outside the polar data" in w for w in outside.warnings)


@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_polar_stall_lowers_trust(solver):
    """Past the stall of the polar, the linear-section solvers are not HIGH (A4)."""
    tab = _stall_polar()
    low = vt.analyze(_tab_wing(tab), alpha_deg=4.0, solver=solver, n_panels=30).totals.trust
    high = vt.analyze(_tab_wing(tab), alpha_deg=11.0, solver=solver, n_panels=30).totals.trust
    assert low.rating == "HIGH"
    assert high.rating in ("LOW", "UNRELIABLE")
    assert high.factors["stall_proximity"] >= 0.5
    assert any("past the stall of their polar" in w for w in high.warnings)


def test_polar_stall_nonlinear_warns():
    tab = _stall_polar()
    t = vt.analyze(_tab_wing(tab), alpha_deg=11.0, solver="nonlinear", n_panels=30).totals.trust
    assert any("Post-stall sections" in w for w in t.warnings)


def test_linear_sections_keep_fixed_stall_limits():
    """Linear sections: the polar rule is off and the fixed limits stay (no change)."""
    w = vt.LiftingSurface(semi_span=4.0, sections=[vt.WingSection(y_frac=0.0, chord=1.0),
                                                   vt.WingSection(y_frac=1.0, chord=1.0)])
    t = vt.analyze(w, alpha_deg=20.0, solver="vlm", n_panels=30).totals.trust
    assert t.factors["polar_range"] == 0.0
    assert t.factors["stall_proximity"] >= 0.5


def test_polar_stats_end_maximum_is_not_stall():
    """A table whose largest Cl is at its end has no stall information."""
    from ventorum.core.trust import polar_limits, polar_stats

    a = np.radians(np.arange(-4.0, 4.01, 1.0))
    tab = vt.TabulatedAirfoil(name="short", alpha=a, Cl_data=2 * np.pi * a, Cd_data=0.01 + 0 * a)
    lim = polar_limits([tab, tab])
    assert np.isnan(lim["cl_max"]).all()
    st = polar_stats(lim, np.array([0.3, 0.5]), np.radians([2.0, 6.0]))
    assert st["n_out_of_table"] == 1
    assert st["max_excess_deg"] == pytest.approx(2.0)
    assert st["n_past_stall"] == 0
    assert st["all_with_stall"] is False
    assert polar_limits([vt.LinearAirfoil()]) is None


def test_low_ar_high_alpha():
    """Unswept AR 1 wing at high alpha: side-edge vortex lift is not modelled (A5)."""
    w = vt.LiftingSurface(semi_span=0.5, sections=[vt.WingSection(y_frac=0.0, chord=1.0),
                                                   vt.WingSection(y_frac=1.0, chord=1.0)])
    t10 = vt.analyze(w, alpha_deg=8.0, solver="vlm", n_panels=20).totals.trust
    t20 = vt.analyze(w, alpha_deg=20.0, solver="vlm", n_panels=20).totals.trust
    assert t10.factors["side_edge_vortex"] == 0.0
    assert t20.factors["side_edge_vortex"] > 0.0
    assert t20.rating != "HIGH"
    assert any("Polhamus 1966" in m for m in t20.warnings)


def test_mach_uses_altitude():
    """The Mach check uses the ISA speed of sound of the air density (A6)."""
    from ventorum.core.constants import A_SL, RHO_SL
    from ventorum.core.trust import isa_speed_of_sound

    assert isa_speed_of_sound(RHO_SL) == A_SL
    assert isa_speed_of_sound(0.36392) == pytest.approx(295.07, abs=0.05)  # 11 000 m
    rho_5km = RHO_SL * (1.0 - 2.25577e-5 * 5000.0) ** 4.25588
    assert isa_speed_of_sound(rho_5km) == pytest.approx(320.5, abs=0.2)
    w = vt.LiftingSurface(semi_span=4.0, sections=[vt.WingSection(y_frac=0.0, chord=1.0),
                                                   vt.WingSection(y_frac=1.0, chord=1.0)])
    sea = vt.analyze(w, condition=vt.FlightCondition(V_inf=100.0, alpha=np.radians(4.0)),
                     settings=vt.SolverSettings(solver_type="vlm", n_panels=20)).totals.trust
    high = vt.analyze(w, condition=vt.FlightCondition(V_inf=100.0, alpha=np.radians(4.0), rho=0.36392),
                      settings=vt.SolverSettings(solver_type="vlm", n_panels=20)).totals.trust
    assert sea.factors["compressibility"] == 0.0
    assert high.factors["compressibility"] > 0.0


# ---------------------------------------------------------------------------
# 3D panel method: mesh rules calibrated by a mesh-error study
# ---------------------------------------------------------------------------

def _panel_stats(ar_median: float = 4.0, skew_median: float = 0.0, n: int = 200, n_special: int = 12,
                 n_nonconvex: int = 0, with_arrays: bool = True) -> dict:
    """Mesh statistics like check_mesh: regular panels plus a few special panels.

    The special panels (trailing-edge base panels, tip-cap and control-edge
    triangles) have a very large aspect ratio and a skew near 1, as on a
    normal wing mesh.
    """
    ar = np.full(n, float(ar_median))
    ar[:n_special] = 800.0
    sk = np.full(n, float(skew_median))
    sk[:n_special] = 0.99
    stats = {"max_aspect_ratio": float(ar.max()), "max_skew": float(sk.max()), "max_warp": 0.0,
             "watertight": True, "n_nonconvex": int(n_nonconvex)}
    if with_arrays:
        stats.update({"aspect_ratio": ar, "skew": sk})
    return stats


def _panel_trust(**kw):
    """Trust of a 3D panel result of a wing with a lift of CL 0.43 and CDi 0.0076."""
    args = dict(AR=8.0, CL=0.43, CDi=0.0076, solver_type="panel3d", n_panels=16)
    args.update(kw)
    return evaluate_aerodynamic_trust(**args)


def test_panel3d_special_panels_do_not_give_a_mesh_warning():
    """A few very elongated or skewed panels (base, caps, control edges) do not warn.

    The largest aspect ratio and skew of a normal wing mesh come from these
    panels; the rules use the median panel.
    """
    t = _panel_trust(mesh_stats=_panel_stats(), CD_nearfield=0.0076 * 0.95)
    assert t.factors["mesh_adequacy"] == 0.0
    assert t.factors["mesh_resolution"] == 0.0
    assert not any("Elongated" in w or "Skewed" in w or "Near-field" in w for w in t.warnings)
    assert t.rating == "HIGH"


def test_panel3d_drag_rule_levels_use_the_named_thresholds():
    """The near/far-field drag levels follow PANEL3D_DRAG_WARN_REL and PANEL3D_DRAG_STRONG_REL."""
    from ventorum.core import trust as tr

    cdi = 0.0076

    def level_of(rel):
        t = _panel_trust(CD_nearfield=cdi * (1.0 - rel))
        return t, t.factors["mesh_resolution"]

    t, pen = level_of(0.98 * tr.PANEL3D_DRAG_WARN_REL)
    assert pen == 0.0 and t.rating == "HIGH"
    t, pen = level_of(1.02 * tr.PANEL3D_DRAG_WARN_REL)
    assert pen == tr.PANEL3D_PENALTY_RESOLUTION_WARN
    assert t.rating == "MODERATE"
    assert any("2 % in CL or 4 % in CDi" in w for w in t.warnings)
    t, pen = level_of(1.02 * tr.PANEL3D_DRAG_STRONG_REL)
    assert pen == tr.PANEL3D_PENALTY_RESOLUTION_STRONG
    assert t.rating == "LOW"
    assert any("5 % in CL or 10 % in CDi" in w for w in t.warnings)


def test_panel3d_drag_rule_floor_at_small_lift():
    """Below the absolute floor the relative drag difference gives no warning (zero lift)."""
    from ventorum.core import trust as tr

    small = _panel_trust(CL=0.0, CDi=0.0, CD_nearfield=0.9 * tr.PANEL3D_DRAG_ABSOLUTE_FLOOR)
    assert small.factors["mesh_resolution"] == 0.0
    large = _panel_trust(CL=0.0, CDi=0.0, CD_nearfield=1.1 * tr.PANEL3D_DRAG_ABSOLUTE_FLOOR)
    assert large.factors["mesh_resolution"] == tr.PANEL3D_PENALTY_RESOLUTION_STRONG


def test_panel3d_chordwise_count_levels_and_worse_level_counts_once():
    """The chordwise rule follows the named counts; the worse level of the two indicators counts once."""
    from ventorum.core import trust as tr

    def pen(n_chord, rel=0.0):
        return _panel_trust(n_chord=n_chord, CD_nearfield=0.0076 * (1.0 - rel)).factors["mesh_resolution"]

    assert pen(tr.PANEL3D_N_CHORD_WARN) == 0.0
    assert pen(tr.PANEL3D_N_CHORD_WARN - 1) == tr.PANEL3D_PENALTY_RESOLUTION_WARN
    assert pen(tr.PANEL3D_N_CHORD_STRONG) == tr.PANEL3D_PENALTY_RESOLUTION_WARN
    assert pen(tr.PANEL3D_N_CHORD_STRONG - 1) == tr.PANEL3D_PENALTY_RESOLUTION_STRONG
    # A drag warning and a strong chordwise level: the strong penalty, not the sum.
    both = pen(tr.PANEL3D_N_CHORD_STRONG - 1, rel=1.02 * tr.PANEL3D_DRAG_WARN_REL)
    assert both == tr.PANEL3D_PENALTY_RESOLUTION_STRONG
    # Without n_chord the chordwise rule does not act.
    assert _panel_trust(CD_nearfield=0.0076).factors["mesh_resolution"] == 0.0


def test_panel3d_pressure_drag_note_below_the_named_chordwise_count():
    """Below PANEL3D_CD_PRESSURE_MIN_N_CHORD a note says to use the Trefftz CDi, with no penalty.

    With least-squares node values in the surface gradient, the near-field
    pressure drag overshoots the Trefftz CDi on coarse chordwise meshes. The
    note must not change the penalty of the chordwise rule, and no value is
    removed from the result.
    """
    from ventorum.core import trust as tr

    n_min = tr.PANEL3D_CD_PRESSURE_MIN_N_CHORD
    assert n_min == 16

    def notes(t):
        return [w for w in t.warnings if "CD_pressure" in w and "Trefftz-plane CDi" in w]

    coarse = _panel_trust(n_chord=n_min - 1, CD_nearfield=0.0076)
    assert len(notes(coarse)) == 1
    assert f"{n_min - 1} chordwise panels" in notes(coarse)[0]
    assert coarse.factors["mesh_resolution"] == tr.PANEL3D_PENALTY_RESOLUTION_WARN
    fine = _panel_trust(n_chord=n_min, CD_nearfield=0.0076)
    assert notes(fine) == []
    assert fine.factors["mesh_resolution"] == coarse.factors["mesh_resolution"]
    # The note acts also without a near-field drag value, and not without n_chord.
    assert len(notes(_panel_trust(n_chord=8))) == 1
    assert notes(_panel_trust(CD_nearfield=0.0076)) == []
    # The note is for the 3D panel method only.
    assert notes(_panel_trust(n_chord=8, solver_type="vlm")) == []


def test_panel3d_median_aspect_ratio_and_folded_panels():
    """The median aspect ratio and folded (non-convex) panels add to mesh_adequacy."""
    from ventorum.core import trust as tr

    ok = _panel_trust(mesh_stats=_panel_stats(ar_median=0.9 * tr.PANEL3D_MAX_ASPECT_RATIO))
    assert ok.factors["mesh_adequacy"] == 0.0
    long_panels = _panel_trust(mesh_stats=_panel_stats(ar_median=1.1 * tr.PANEL3D_MAX_ASPECT_RATIO))
    assert long_panels.factors["mesh_adequacy"] == pytest.approx(tr.PANEL3D_PENALTY_ASPECT_RATIO)
    assert any("Elongated panels: the median panel aspect ratio" in w for w in long_panels.warnings)
    folded = _panel_trust(mesh_stats=_panel_stats(n_nonconvex=2))
    assert folded.factors["mesh_adequacy"] == pytest.approx(tr.PANEL3D_PENALTY_NONCONVEX)
    assert any("Folded panels" in w for w in folded.warnings)
    # Statistics without the per-panel arrays: the largest values are used.
    no_arrays = _panel_trust(mesh_stats=_panel_stats(with_arrays=False))
    assert any("Elongated panels: the panel aspect ratio" in w for w in no_arrays.warnings)


def test_panel3d_thresholds_are_used_by_name(monkeypatch):
    """Changing a named threshold changes the result: the rules read the module constants."""
    from ventorum.core import trust as tr

    stats = _panel_stats(ar_median=10.0)
    base = _panel_trust(mesh_stats=stats, CD_nearfield=0.0076 * 0.9, n_chord=20)
    assert base.factors["mesh_adequacy"] == 0.0
    assert base.factors["mesh_resolution"] == tr.PANEL3D_PENALTY_RESOLUTION_WARN  # n_chord 20
    monkeypatch.setattr(tr, "PANEL3D_MAX_ASPECT_RATIO", 5.0)
    monkeypatch.setattr(tr, "PANEL3D_N_CHORD_WARN", 16)
    monkeypatch.setattr(tr, "PANEL3D_DRAG_WARN_REL", 0.05)
    monkeypatch.setattr(tr, "PANEL3D_PENALTY_RESOLUTION_WARN", 0.3)
    changed = _panel_trust(mesh_stats=stats, CD_nearfield=0.0076 * 0.9, n_chord=20)
    assert changed.factors["mesh_adequacy"] == pytest.approx(tr.PANEL3D_PENALTY_ASPECT_RATIO)
    assert changed.factors["mesh_resolution"] == 0.3  # the drag rule now, at the new penalty
    assert any("above 5%" in w for w in changed.warnings)


@pytest.mark.parametrize("solver", ["vlm", "linear", "nonlinear", "fourier"])
def test_panel3d_rules_do_not_change_other_solvers(solver, monkeypatch):
    """The panel inputs and constants do not change the trust of the other solvers."""
    from ventorum.core import trust as tr

    kw = dict(AR=8.0, CL=0.43, CDi=0.0076, solver_type=solver, n_panels=16)
    plain = evaluate_aerodynamic_trust(**kw).to_dict()
    monkeypatch.setattr(tr, "PANEL3D_MAX_ASPECT_RATIO", 0.5)
    monkeypatch.setattr(tr, "PANEL3D_DRAG_WARN_REL", 0.0)
    monkeypatch.setattr(tr, "PANEL3D_DRAG_ABSOLUTE_FLOOR", 0.0)
    with_panel = evaluate_aerodynamic_trust(mesh_stats=_panel_stats(n_nonconvex=3), CD_nearfield=0.0,
                                            **kw).to_dict()
    assert with_panel == plain
    assert "mesh_resolution" not in plain["factors"] and "mesh_adequacy" not in plain["factors"]


def _shaped_wing(shape="4412"):
    return LiftingSurface(
        name="ShapedWing",
        semi_span=4.0,
        sections=[WingSection(y_frac=0.0, chord=1.0, shape=shape), WingSection(y_frac=1.0, chord=1.0, shape=shape)],
    )


@pytest.mark.parametrize("solver", ["linear", "nonlinear"])
def test_lifting_line_notes_an_unused_section_shape(solver):
    """The lifting line ignores WingSection.shape; the result says so in a note with no penalty.

    Regression: a NACA 4412 shape with the default LinearAirfoil gave the
    LLT result of an uncambered wing with HIGH trust and no note.
    """
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    settings = vt.SolverSettings(solver_type=solver, n_panels=24)
    with pytest.warns(RuntimeWarning, match="lifting line does not use it"):
        res = vt.analyze(_shaped_wing(), cond, settings)
    plain = vt.analyze(_shaped_wing(shape=None), cond, settings)
    msgs = [w for w in res.totals.trust.warnings if "lifting line does not use it" in w]
    assert len(msgs) == 1
    assert "section polar only" in msgs[0]
    assert any("section polar only" in n for n in res.details["notes"])
    # A note only: no score penalty and the same loads as without the shape.
    assert res.totals.trust.score == pytest.approx(plain.totals.trust.score)
    assert res.totals.CL == pytest.approx(plain.totals.CL)
    assert not any("lifting line does not use it" in w for w in plain.totals.trust.warnings)


def test_vlm_mean_line_mismatch_is_a_trust_warning():
    """The VLM warning about a mean line alpha_L0 far from the polar alpha_L0 is also a trust warning.

    Regression: it was only a Python warning.
    """
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    settings = vt.SolverSettings(solver_type="vlm", n_panels=16, n_chord=4)
    with pytest.warns(RuntimeWarning, match="mean line and polar alpha_L0"):
        res = vt.analyze(_shaped_wing(), cond, settings)
    msgs = [w for w in res.totals.trust.warnings if "mean line and polar alpha_L0" in w]
    assert len(msgs) == 2  # root and tip sections
    assert not any("lifting line does not use it" in w for w in res.totals.trust.warnings)


def test_fourier_notes_an_unused_section_shape():
    """The Fourier lifting line ignores WingSection.shape; the result says so in a note with no penalty.

    Regression: the Fourier solver ignored the shape with no note.
    """
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    settings = vt.SolverSettings(solver_type="fourier", n_panels=24)
    with pytest.warns(RuntimeWarning, match="Fourier lifting line does not use it"):
        res = vt.analyze(_shaped_wing(), cond, settings)
    plain = vt.analyze(_shaped_wing(shape=None), cond, settings)
    msgs = [w for w in res.totals.trust.warnings if "Fourier lifting line does not use it" in w]
    assert len(msgs) == 1
    assert "section polar only" in msgs[0]
    assert any("Fourier lifting line does not use it" in n for n in res.details["notes"])
    # A note only: no score penalty and the same loads as without the shape.
    assert res.totals.trust.score == pytest.approx(plain.totals.trust.score)
    assert res.totals.CL == plain.totals.CL
    assert res.totals.CDi == plain.totals.CDi
    assert res.totals.Cm == plain.totals.Cm
    assert not any("does not use it" in w for w in plain.totals.trust.warnings)
    assert not any("does not use it" in n for n in plain.details.get("notes", []))


def test_fourier_sweep_notes_an_unused_section_shape():
    """Each result of a Fourier sweep carries the note of an unused section shape."""
    from ventorum.solvers.fourier import FourierSolver

    cond = FlightCondition(V_inf=30.0, alpha=0.0)
    settings = vt.SolverSettings(solver_type="fourier", n_panels=24)
    alphas = np.radians([0.0, 4.0])
    with pytest.warns(RuntimeWarning, match="Fourier lifting line does not use it"):
        res = FourierSolver().solve_sweep(_shaped_wing(), cond, settings, alphas)
    plain = FourierSolver().solve_sweep(_shaped_wing(shape=None), cond, settings, alphas)
    for r, p in zip(res, plain):
        assert any("Fourier lifting line does not use it" in w for w in r.totals.trust.warnings)
        assert r.totals.CL == p.totals.CL
        assert not any("does not use it" in w for w in p.totals.trust.warnings)


def test_vlm_single_chord_panel_notes_an_unused_section_shape():
    """The VLM with n_chord = 1 ignores WingSection.shape; the result says so in a note with no penalty.

    The mean line needs two or more chordwise panels. With one panel the
    lattice keeps the polar alpha_L0 (turned normal) and the polar Cm0.
    Regression: it ignored the shape with no note.
    """
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    settings = vt.SolverSettings(solver_type="vlm", n_panels=16, n_chord=1)
    with pytest.warns(RuntimeWarning, match="n_chord = 1 does not use it"):
        res = vt.analyze(_shaped_wing(), cond, settings)
    plain = vt.analyze(_shaped_wing(shape=None), cond, settings)
    msgs = [w for w in res.totals.trust.warnings if "n_chord = 1 does not use it" in w]
    assert len(msgs) == 1
    assert "polar alpha_L0" in msgs[0]
    assert any("n_chord = 1 does not use it" in n for n in res.details["notes"])
    # A note only: no score penalty and the same loads as without the shape.
    assert res.totals.trust.score == pytest.approx(plain.totals.trust.score)
    assert res.totals.CL == plain.totals.CL
    assert res.totals.CDi == plain.totals.CDi
    assert res.totals.Cm == plain.totals.Cm
    assert not any("does not use it" in w for w in plain.totals.trust.warnings)
    assert not any("does not use it" in n for n in plain.details["notes"])
    # No mean-line warning: the mean line is not used.
    assert not any("mean line and polar alpha_L0" in w for w in res.totals.trust.warnings)


def test_vlm_with_chordwise_panels_gives_no_single_panel_note():
    """The VLM with n_chord > 1 uses the shape, so the note of one chordwise panel does not appear."""
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    settings = vt.SolverSettings(solver_type="vlm", n_panels=16, n_chord=4)
    with pytest.warns(RuntimeWarning, match="mean line and polar alpha_L0"):
        res = vt.analyze(_shaped_wing(), cond, settings)
    assert not any("does not use it" in w for w in res.totals.trust.warnings)


def test_one_panel_lattice_build_gives_no_shape_warning():
    """A one-panel lattice build alone issues no warning about the shape; only a solve does.

    Many callers (mesh-size checks, ground clearance probes, the strip data
    of the panel-method coupling) build a lattice with one chordwise panel
    and use the shape elsewhere. They must not get the note.
    """
    import warnings

    from ventorum.core.datatypes import Aircraft
    from ventorum.geometry.lattice import build_lattice

    ac = Aircraft(surfaces=[_shaped_wing()])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lat = build_lattice(ac, vt.SolverSettings(n_panels=8), collocation="vlm", n_chord=1)
    assert not any("does not use it" in m for m in lat.mesh_warnings)
    assert sum("n_chord = 1 does not use it" in m for m in lat.join_warnings) == 1
