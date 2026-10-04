# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Comprehensive tests for automated mesh convergence study, discretization optimization,
and dimensionless generalization guidelines in Ventorum.
"""

from __future__ import annotations

import json
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless test runs
import matplotlib.pyplot as plt

import ventorum as vt
from ventorum.geometry.mesh_convergence import (
    run_mesh_convergence_study,
    MeshConvergenceResult,
    GeneralizationGuideline,
)
from ventorum.agent.dispatcher import call_tool


@pytest.fixture
def planar_rectangular_wing() -> vt.LiftingSurface:
    """Standard planar rectangular benchmark wing (AR=8, span=10m, chord=1.25m)."""
    return vt.LiftingSurface(
        name="PlanarBenchWing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.25),
            vt.WingSection(y_frac=1.0, chord=1.25),
        ],
        sweep_le=0.0,
        dihedral=0.0,
        is_symmetric=True,
    )


@pytest.fixture
def multi_surface_aircraft() -> vt.Aircraft:
    """Two-surface aircraft: primary wing + smaller inverted V-tail."""
    main_wing = vt.LiftingSurface(
        name="MainWing",
        semi_span=1.5,
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.25),
            vt.WingSection(y_frac=1.0, chord=0.15),
        ],
        sweep_le=np.radians(1.5),
        dihedral=np.radians(2.0),
        is_symmetric=True,
    )
    v_tail = vt.LiftingSurface(
        name="VTail",
        semi_span=0.4,
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.14),
            vt.WingSection(y_frac=1.0, chord=0.09),
        ],
        sweep_le=np.radians(8.0),
        dihedral=np.radians(-38.0),
        position=np.array([0.9, 0.0, 0.05]),
        is_symmetric=True,
    )
    ac = vt.Aircraft(name="BenchUAV", surfaces=[main_wing, v_tail])
    ac.compute_reference_values()
    return ac


def test_planar_wing_mesh_convergence(planar_rectangular_wing):
    """Verify that mesh convergence study correctly identifies minimal and recommended meshes."""
    cond = vt.FlightCondition(V_inf=40.0, alpha=np.radians(5.0))
    study = run_mesh_convergence_study(
        case=planar_rectangular_wing,
        condition=cond,
        tolerance_pct=0.5,
        panel_counts=(10, 15, 20, 30, 40),
        ref_n_panels=80,
        spacing_schemes=("half-cosine", "uniform"),
        progress=False,
    )

    assert isinstance(study, MeshConvergenceResult)
    assert len(study.points) == 10  # 5 panel counts * 2 spacing schemes
    assert study.reference_point.n_panels == 80
    assert study.reference_point.CL > 0.0
    assert study.reference_point.CDi > 0.0

    # Minimal and recommended meshes should be identified
    min_mesh = study.minimal_mesh
    rec_mesh = study.recommended_mesh
    assert min_mesh is not None
    assert rec_mesh is not None

    # Minimal mesh must be accurate within tolerance
    assert min_mesh.is_accurate
    assert min_mesh.error_cl_pct <= study.tolerance_pct
    assert min_mesh.error_cdi_pct <= study.tolerance_pct

    # Recommended mesh panel count >= minimal mesh panel count (or higher safety margin)
    assert rec_mesh.total_panels >= min_mesh.total_panels
    assert rec_mesh.is_accurate

    # Generalization guidelines
    guideline = study.generalization
    assert isinstance(guideline, GeneralizationGuideline)
    assert guideline.panels_per_ar > 0.0
    assert guideline.aspect_ratio == pytest.approx(8.0, rel=1e-3)
    assert "half-cosine" in guideline.spacing_selection_rule.lower()
    assert guideline.recommended_n_panels == rec_mesh.n_panels


def test_spacing_scheme_efficiency_comparison(planar_rectangular_wing):
    """Verify that tip-clustered half-cosine spacing is more efficient than uniform spacing."""
    cond = vt.FlightCondition(V_inf=50.0, alpha=np.radians(4.0))
    study = run_mesh_convergence_study(
        case=planar_rectangular_wing,
        condition=cond,
        tolerance_pct=0.5,
        panel_counts=(15, 30),
        ref_n_panels=80,
        spacing_schemes=("half-cosine", "uniform"),
        progress=False,
    )

    # Find N=15 points for half-cosine and uniform
    hc_15 = next(p for p in study.points if p.n_panels == 15 and p.spacing == "half-cosine")
    u_15 = next(p for p in study.points if p.n_panels == 15 and p.spacing == "uniform")

    # Half-cosine tip clustering accurately captures tip circulation dropoff,
    # giving lower lift error at N=15
    assert hc_15.error_cl_pct < u_15.error_cl_pct
    assert hc_15.is_safe and u_15.is_safe


def test_multi_surface_aircraft_with_proportional_panels(multi_surface_aircraft):
    """Verify convergence study on multi-surface aircraft with proportional panel allocation."""
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    study = run_mesh_convergence_study(
        case=multi_surface_aircraft,
        condition=cond,
        tolerance_pct=0.5,
        panel_counts=(15, 25, 35),
        ref_n_panels=70,
        spacing_schemes="auto",
        proportional_panels=True,
        progress=False,
    )

    assert len(study.points) == 3
    # Check that total panels properly includes both surfaces with proportional scaling
    for pt in study.points:
        # On symmetry half-mesh: Wing (N) + Tail (clamped proportional) > N
        assert pt.total_panels > pt.n_panels
        assert pt.CL > 0
        assert pt.CDi > 0
        assert pt.cond_num < 1e8  # Well conditioned

    # Summary table should format cleanly
    summary_txt = study.summary(as_markdown=False)
    assert "Ventorum MESH CONVERGENCE STUDY" in summary_txt
    assert "[+] MINIMAL MESH:" in summary_txt
    assert "[+] RECOMMENDED MESH:" in summary_txt

    summary_md = study.summary(as_markdown=True)
    assert "| N_ref | Spacing | N_tot |" in summary_md


def test_mesh_convergence_across_alpha_sweep(planar_rectangular_wing):
    """Verify that mesh convergence evaluates accuracy across the entire flight envelope."""
    study = run_mesh_convergence_study(
        case=planar_rectangular_wing,
        tolerance_pct=0.8,
        panel_counts=(12, 20, 30),
        ref_n_panels=60,
        spacing_schemes="auto",
        evaluate_sweep=True,
        alpha_sweep_deg=[-2.0, 2.0, 6.0, 10.0],
        progress=False,
    )

    assert len(study.alpha_tested_deg) == 4
    for pt in study.points:
        # Sweep errors must be computed
        assert pt.max_sweep_error_pct >= 0.0
        assert pt.mean_sweep_error_pct >= 0.0
        assert pt.max_sweep_error_pct >= pt.mean_sweep_error_pct

    # Recommended mesh must pass across the sweep
    assert study.recommended_mesh.max_sweep_error_pct <= study.tolerance_pct


def test_ventorum_instance_integration(planar_rectangular_wing):
    """Verify calling mesh convergence directly on an Ventorum case instance."""
    case = vt.Ventorum(
        name="StudyCase",
        geometry=planar_rectangular_wing,
        alpha_deg=4.0,
        V_inf=45.0,
        n_panels=80,
    )

    study = case.run_mesh_convergence(
        tolerance_pct=0.5,
        panel_counts=(15, 25, 35),
        ref_n_panels=60,
        spacing_schemes="auto",
        apply_to_case=True,
    )

    assert isinstance(study, MeshConvergenceResult)
    # Check that settings were applied to case
    assert case.settings.n_panels == study.recommended_settings.n_panels
    assert case.settings.spacing == study.recommended_settings.spacing

    # Test explicit apply_mesh_recommendation
    case.settings.n_panels = 999
    case.apply_mesh_recommendation(study)
    assert case.settings.n_panels == study.recommended_settings.n_panels


def test_target_metrics_and_circulation_l2_error(planar_rectangular_wing):
    """Verify selective target metrics (CL only, CDi only, circulation L2 norm)."""
    cond = vt.FlightCondition(V_inf=50.0, alpha=np.radians(5.0))

    # 1. Target CL
    study_cl = run_mesh_convergence_study(
        case=planar_rectangular_wing,
        condition=cond,
        tolerance_pct=0.3,
        target_metric="CL",
        panel_counts=(15, 25),
        ref_n_panels=60,
        spacing_schemes="auto",
    )
    assert study_cl.target_metric == "CL"
    assert study_cl.minimal_mesh.error_cl_pct <= 0.3

    # 2. Target circulation L2 error
    study_gamma = run_mesh_convergence_study(
        case=planar_rectangular_wing,
        condition=cond,
        tolerance_pct=1.0,
        target_metric="circulation",
        panel_counts=(15, 25),
        ref_n_panels=60,
        spacing_schemes="auto",
    )
    assert study_gamma.target_metric == "circulation"
    assert study_gamma.minimal_mesh.error_gamma_l2_pct <= 1.0


def test_study_serialization_and_dict(planar_rectangular_wing):
    """Verify that MeshConvergenceResult serializes to valid JSON dictionary."""
    study = run_mesh_convergence_study(
        case=planar_rectangular_wing,
        tolerance_pct=0.5,
        panel_counts=(10, 20),
        ref_n_panels=40,
        spacing_schemes="auto",
    )

    data = study.to_dict()
    assert isinstance(data, dict)
    assert "recommended_mesh" in data
    assert "minimal_mesh" in data
    assert "generalization" in data
    assert "points" in data
    assert len(data["points"]) == 2

    # Verify JSON serializability
    json_str = json.dumps(data)
    assert len(json_str) > 0


def test_mesh_convergence_plot_generation(planar_rectangular_wing, tmp_path):
    """Verify that plot() creates a valid matplotlib figure with 4 subplots and exports to file."""
    study = run_mesh_convergence_study(
        case=planar_rectangular_wing,
        tolerance_pct=0.5,
        panel_counts=(10, 20),
        ref_n_panels=40,
        spacing_schemes="auto",
    )

    fig_path = tmp_path / "test_convergence.png"
    fig = study.plot(save_path=fig_path, show=False)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 4  # 4 subplots (plus possible twinx)
    assert fig_path.exists()
    assert fig_path.stat().st_size > 0
    plt.close(fig)


def test_agent_tool_mesh_convergence():
    """Agent tool 'ventorum_mesh_convergence' through the dispatcher, with strict inputs."""
    wing_spec = {
        "span_m": 10.0,
        "root_chord_m": 1.5,
        "tip_chord_m": 1.0,
        "sweep_le_deg": 0.0,
        "dihedral_deg": 0.0,
    }
    flight_cond = {"V_inf_m_s": 45.0, "alpha_deg": 4.0}

    res = call_tool(
        "ventorum_mesh_convergence",
        {
            "wing": wing_spec,
            "flight_condition": flight_cond,
            "tolerance_pct": 0.5,
            "panel_counts": [12, 20, 30],
            "spacing_schemes": ["half-cosine", "uniform"],
            "ref_n_panels": 60,
            "detail_level": "standard",
        },
    )

    assert res["status"] == "success", res
    json.dumps(res, allow_nan=False)
    assert "minimal_mesh" in res
    assert "recommended_mesh" in res
    assert "recommended_settings" in res
    assert "generalization" in res
    assert "summary_markdown" in res
    assert res["recommended_settings"]["n_panels"] in [12, 20, 30]
    assert res["executive_summary"].startswith("[Ventorum RESULT]")

    # Strict inputs: an unknown key and a too-small reference mesh are refused.
    bad = call_tool("ventorum_mesh_convergence", {"wing": wing_spec, "panels": [12, 20]})
    assert bad["status"] == "error" and bad["error"]["type"] == "invalid_input"
    bad = call_tool("ventorum_mesh_convergence",
                    {"wing": wing_spec, "panel_counts": [12, 20], "ref_n_panels": 20})
    assert bad["status"] == "error" and "ref_n_panels" in bad["error"]["message"]


# ── Regression tests: per-surface overrides, settings, span coordinate, reuse ──

def _wing_and_tail(tail_n_panels: int | None = 10, tail_spacing: str | None = None) -> vt.Aircraft:
    wing = vt.LiftingSurface(
        name="Wing", semi_span=3.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.6)],
    )
    tail = vt.LiftingSurface(
        name="Tail", semi_span=1.0,
        sections=[vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.4)],
        position=np.array([3.0, 0.0, 0.3]), n_panels=tail_n_panels, spacing=tail_spacing,
    )
    return vt.Aircraft(name="WingTail", surfaces=[wing, tail])


def test_surface_n_panels_override_is_refined_with_level():
    """A surface with its own n_panels is refined at each level (no false convergence)."""
    ac = _wing_and_tail(tail_n_panels=40, tail_spacing="uniform")
    study = run_mesh_convergence_study(
        ac, tolerance_pct=1.0, panel_counts=(10, 20), ref_n_panels=40,
        spacing_schemes="auto", proportional_panels=False,
    )
    # Base n_panels is 80, so the tail count is scaled by n / 80 (at least 4).
    tail_strips = [len(p.result.spanwise[1].gamma) for p in study.points]
    ref_tail = len(study.reference_result.spanwise[1].gamma)
    assert tail_strips == [2 * 5, 2 * 10]
    assert ref_tail == 2 * 20
    assert study.recommended_surface_n_panels[0] is None
    assert study.recommended_surface_n_panels[1] in (5, 10)
    # The input aircraft is not changed.
    assert ac.surfaces[1].n_panels == 40 and ac.surfaces[1].spacing == "uniform"


def test_surface_override_counts_in_total_panels():
    ac = _wing_and_tail(tail_n_panels=40)
    study = run_mesh_convergence_study(
        ac, tolerance_pct=1.0, panel_counts=(10, 20), ref_n_panels=40,
        spacing_schemes="auto", proportional_panels=False,
    )
    # Tail: round(40 * n / 80) panels per half; with symmetry one half is counted.
    assert [p.total_panels for p in study.points] == [10 + 5, 20 + 10]
    assert study.reference_point.total_panels == 40 + 20


def test_recommended_settings_keep_base_settings():
    case = vt.Ventorum(
        name="Keep",
        geometry=_wing_and_tail(tail_n_panels=None),
        alpha_deg=4.0,
        settings=vt.SolverSettings(n_panels=40, n_chord=2, chord_spacing="cosine", min_panels=5,
                                    wake_alignment="body"),
    )
    study = case.run_mesh_convergence(
        tolerance_pct=2.0, panel_counts=(10, 20), ref_n_panels=30, spacing_schemes="auto",
        apply_to_case=True,
    )
    rs = study.recommended_settings
    assert rs.n_chord == 2 and rs.chord_spacing == "cosine"
    assert rs.min_panels == 5 and rs.wake_alignment == "body"
    assert rs.n_panels == study.recommended_mesh.n_panels
    assert case.settings.n_chord == 2
    assert study.to_dict()["recommended_settings"]["n_chord"] == 2


def test_circulation_error_uses_span_coordinate_on_fin():
    """The circulation error of a vertical fin is taken along its span, not along y."""
    from ventorum.geometry.mesh_convergence import _gamma_l2_error_pct, _span_coordinate

    wing = vt.LiftingSurface(name="Wing", semi_span=3.0,
                              sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 0.8)])
    fin = vt.LiftingSurface(name="Fin", semi_span=1.0,
                             sections=[vt.WingSection(0.0, 0.8), vt.WingSection(1.0, 0.5)],
                             is_symmetric=False, dihedral=np.radians(90.0), position=np.array([3.0, 0.0, 0.0]))
    ac = vt.Aircraft(surfaces=[wing, fin])
    cond = vt.FlightCondition(alpha=np.radians(4.0), beta=np.radians(5.0))
    coarse = vt.analyze(ac, cond, vt.SolverSettings(n_panels=10))
    fine = vt.analyze(ac, cond, vt.SolverSettings(n_panels=40))
    eta = _span_coordinate(coarse, 1)
    assert np.ptp(coarse.spanwise[1].y) < 1e-9  # y does not change along the fin
    assert np.ptp(eta) > 0.8
    assert _gamma_l2_error_pct(fine, fine) == 0.0
    assert _gamma_l2_error_pct(coarse, fine) < 10.0


def test_levels_with_same_mesh_and_primary_alpha_are_solved_once(planar_rectangular_wing, monkeypatch):
    """'auto' gives 'half-cosine' on a planar wing: that level is solved once, and the
    sweep angle equal to the primary condition is not solved again."""
    calls = {"n": 0}
    real = vt.analyze

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(vt, "analyze", counting)
    cond = vt.FlightCondition(alpha=np.radians(5.0))
    study = run_mesh_convergence_study(
        planar_rectangular_wing, condition=cond, tolerance_pct=1.0, panel_counts=(10, 20),
        ref_n_panels=40, spacing_schemes=("auto", "half-cosine"),
        evaluate_sweep=True, alpha_sweep_deg=(0.0, 5.0),
    )
    # Reference: 2 solves (alpha 0 and 5). Two distinct levels: 2 solves each.
    assert calls["n"] == 2 + 2 * 2
    by_n = {}
    for p in study.points:
        by_n.setdefault(p.n_panels, []).append(p)
    for pts in by_n.values():
        assert len(pts) == 2 and pts[0].result is pts[1].result


def test_discarded_spacing_call_is_removed():
    import ventorum.geometry.mesh_convergence as mc

    assert not hasattr(mc, "determine_optimal_spacing")
