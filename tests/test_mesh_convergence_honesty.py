# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Honesty checks for mesh-convergence reports (T-0008)."""

from __future__ import annotations

import numpy as np

import ventorum as vt
from ventorum.agent.dispatcher import call_tool
from ventorum.geometry.mesh_convergence import run_mesh_convergence_study


def _bench_wing() -> vt.LiftingSurface:
    """Return the planar rectangular benchmark wing (AR=8)."""
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


def _wing_and_tail() -> vt.Aircraft:
    """Return a two-surface aircraft (wing plus tail)."""
    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=3.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=0.6),
        ],
    )
    tail = vt.LiftingSurface(
        name="Tail",
        semi_span=1.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.5),
            vt.WingSection(y_frac=1.0, chord=0.4),
        ],
        position=np.array([3.0, 0.0, 0.3]),
    )
    ac = vt.Aircraft(name="WingTail", surfaces=[wing, tail])
    ac.compute_reference_values()
    return ac


def test_not_converged_is_reported():
    """A study with a tolerance that no level meets reports converged False."""
    cond = vt.FlightCondition(V_inf=40.0, alpha=np.radians(5.0))
    study = run_mesh_convergence_study(
        case=_bench_wing(),
        condition=cond,
        tolerance_pct=0.01,
        panel_counts=(10, 15, 20),
        ref_n_panels=80,
        spacing_schemes=("uniform",),
        progress=False,
    )
    assert study.converged is False
    finest = max(study.points, key=lambda p: (p.panels_solved, p.total_panels))
    assert study.recommended_mesh.panels_solved == finest.panels_solved
    text = study.summary(as_markdown=False).lower()
    assert "not converged within tolerance" in text
    text_md = study.summary(as_markdown=True).lower()
    assert "not converged within tolerance" in text_md

    res = call_tool(
        "ventorum_mesh_convergence",
        {
            "wing": {
                "span_m": 10.0,
                "root_chord_m": 1.25,
                "tip_chord_m": 1.25,
                "sweep_le_deg": 0.0,
                "dihedral_deg": 0.0,
            },
            "flight_condition": {"V_inf_m_s": 40.0, "alpha_deg": 5.0},
            "tolerance_pct": 0.01,
            "panel_counts": [10, 15, 20],
            "spacing_schemes": ["uniform"],
            "ref_n_panels": 80,
            "detail_level": "standard",
        },
    )
    assert res["status"] == "success", res
    assert res["converged"] is False
    assert "not converged" in res["executive_summary"].lower()


def test_panels_solved_counts_all_surfaces():
    """Each level reports the panel count of the lattice that was solved."""
    study = run_mesh_convergence_study(
        case=_wing_and_tail(),
        tolerance_pct=1.0,
        panel_counts=(10, 20),
        ref_n_panels=40,
        spacing_schemes="auto",
        progress=False,
    )
    assert len(study.points) == 2
    for pt in study.points:
        lat = pt.result.details.get("lattice")
        assert lat is not None
        assert pt.panels_solved == int(lat.n_panels)
        # The solved lattice holds both halves and all chordwise panels.
        assert pt.panels_solved > pt.n_panels
        assert pt.panels_solved >= pt.total_panels
    ref_lat = study.reference_result.details.get("lattice")
    assert ref_lat is not None
    assert study.reference_point.panels_solved == int(ref_lat.n_panels)


def test_converged_study_unchanged():
    """A study that converges today keeps the same recommendation."""
    cond = vt.FlightCondition(V_inf=40.0, alpha=np.radians(5.0))
    study = run_mesh_convergence_study(
        case=_bench_wing(),
        condition=cond,
        tolerance_pct=0.5,
        panel_counts=(10, 15, 20, 30, 40),
        ref_n_panels=80,
        spacing_schemes=("half-cosine", "uniform"),
        progress=False,
    )
    assert study.converged is True
    assert (study.minimal_mesh.n_panels, study.minimal_mesh.spacing) == (10, "half-cosine")
    assert (study.recommended_mesh.n_panels, study.recommended_mesh.spacing) == (15, "half-cosine")
