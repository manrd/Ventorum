# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests that an Ventorum instance returns the results of its last run only.

The instance has two run modes: one flight condition (``analyze``) and an
angle-of-attack sweep (``analyze_sweep``). A new run replaces the results of
the other mode, so the getters never mix a single point with a sweep.
"""

import numpy as np
import pytest

import ventorum as vt
from ventorum.instance import Ventorum


@pytest.fixture
def rect_wing():
    """Rectangular wing, simple enough for a fast test."""
    return vt.LiftingSurface(
        name="RectangularWing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.5),
            vt.WingSection(y_frac=1.0, chord=1.5),
        ],
    )


def test_summary_after_sweep_then_single(rect_wing):
    """After a sweep, a single-point run leaves only the single-point results."""
    inst = Ventorum("State", geometry=rect_wing, n_panels=12)
    inst.analyze_sweep([0.0, 2.0, 4.0])
    assert inst.last_run_mode == "sweep"
    assert inst.sweep_results is not None
    assert "sweep" in inst.get_summary()

    inst.analyze(alpha_deg=6.0)
    assert inst.last_run_mode == "single"
    assert inst.sweep_results is None
    assert inst.sweep_alphas_deg is None

    ref = vt.analyze(rect_wing, alpha_deg=6.0, n_panels=12)
    summary = inst.get_summary()
    assert "sweep" not in summary
    assert summary["single_point"]["CL"] == pytest.approx(ref.totals.CL, rel=1e-12)
    assert summary["single_point"]["CDi"] == pytest.approx(ref.totals.CDi, rel=1e-12)

    ld = inst.get_L_over_D()
    assert isinstance(ld, float)
    assert ld == pytest.approx(ref.totals.CL / ref.totals.CDi, rel=1e-12)


def test_polar_after_single_then_sweep(rect_wing):
    """After a single-point run, a sweep leaves only the sweep results."""
    inst = Ventorum("State", geometry=rect_wing, n_panels=12)
    inst.analyze(alpha_deg=2.0)
    assert inst.last_run_mode == "single"
    assert inst.result is not None

    inst.analyze_sweep([0.0, 2.0, 4.0])
    assert inst.last_run_mode == "sweep"
    assert inst.result is None

    alphas, cls, cdis = inst.get_polar()
    assert alphas.tolist() == [0.0, 2.0, 4.0]
    assert len(cls) == 3 and len(cdis) == 3
    assert np.all(np.diff(cls) > 0.0)

    summary = inst.get_summary()
    assert "single_point" not in summary
    assert summary["sweep"]["n_alphas"] == 3

    ld = inst.get_L_over_D()
    assert isinstance(ld, np.ndarray)
    assert len(ld) == 3


def test_polar_after_single_run_has_one_point(rect_wing):
    """A single-point run gives a polar of one point, at the angle that was solved."""
    inst = Ventorum("OnePoint", geometry=rect_wing, n_panels=12)
    inst.analyze(alpha_deg=3.0)
    ref = vt.analyze(rect_wing, alpha_deg=3.0, n_panels=12)

    alphas, cls, cdis = inst.get_polar()
    assert alphas == pytest.approx([3.0])  # rad to deg round trip
    assert cls[0] == pytest.approx(ref.totals.CL, rel=1e-12)
    assert cdis[0] == pytest.approx(ref.totals.CDi, rel=1e-12)


def test_mesh_convergence_without_reference_values(rect_wing):
    """A mesh convergence study runs on an aircraft without reference values.

    The study reads ``S_ref``, ``b_ref`` and ``c_ref`` from a copy of the
    aircraft of the instance. A solve fills them, a study needs them.
    """
    wing = rect_wing.clone()
    wing.n_panels = 15  # a surface with its own count, so the study clones the aircraft
    inst = Ventorum("NoRef", geometry=wing, n_panels=20, alpha_deg=5.0)
    # An aircraft whose reference values were never computed.
    inst.aircraft.S_ref = None
    inst.aircraft.b_ref = None
    inst.aircraft.c_ref = None

    study = inst.run_mesh_convergence(
        tolerance_pct=2.0,
        panel_counts=(10, 20),
        ref_n_panels=30,
        spacing_schemes="auto",
    )
    assert study.recommended_mesh.n_panels > 0
    # The reference values are now the ones of the geometry.
    assert inst.aircraft.S_ref == pytest.approx(15.0, rel=1e-9)
    assert inst.aircraft.b_ref == pytest.approx(10.0, rel=1e-9)
    assert inst.aircraft.c_ref == pytest.approx(1.5, rel=1e-9)


def test_getters_before_any_run(rect_wing):
    """Before the first run the result getters raise RuntimeError."""
    inst = Ventorum("Fresh", geometry=rect_wing, n_panels=12)
    assert inst.last_run_mode is None
    with pytest.raises(RuntimeError):
        inst.get_polar()
    with pytest.raises(RuntimeError):
        inst.get_L_over_D()
    # The summary reports the state of the instance, also before a run.
    summary = inst.get_summary()
    assert summary["status"] == "idle"
    assert "single_point" not in summary
    assert "sweep" not in summary


def test_run_selects_the_mode_of_the_case(rect_wing):
    """run() runs a sweep when the case has one, and a single point otherwise."""
    single = Ventorum("Single", geometry=rect_wing, n_panels=12, alpha_deg=4.0)
    single.run()
    assert single.last_run_mode == "single"
    assert single.result is not None and single.sweep_results is None

    swept = Ventorum("Swept", geometry=rect_wing, n_panels=12, alpha_sweep_deg=[0.0, 2.0, 4.0])
    swept.run()
    assert swept.last_run_mode == "sweep"
    assert swept.sweep_results is not None and swept.result is None


def test_failed_run_keeps_the_results_of_the_last_success(rect_wing):
    """A run that fails leaves the results of the last successful run."""
    inst = Ventorum("Failure", geometry=rect_wing, n_panels=12)
    inst.analyze(alpha_deg=2.0)
    cl = inst.get_summary()["single_point"]["CL"]

    with pytest.raises(ValueError):
        inst.analyze(settings=vt.SolverSettings(n_panels=0))

    assert inst.last_run_mode == "single"
    assert inst.get_summary()["single_point"]["CL"] == cl
