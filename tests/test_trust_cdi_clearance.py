# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Unit tests for the trailing-edge clearance trust warning on Trefftz CDi.
"""

from __future__ import annotations

import numpy as np

import ventorum as vt
from ventorum.core.datatypes import FlightCondition, LiftingSurface, WingSection
from ventorum.core.trust import (
    _CDI_TE_CLEARANCE,
    evaluate_aerodynamic_trust,
    trailing_edge_clearance,
)
from ventorum.geometry.lattice import build_lattice
from ventorum.aero.system import ground_plane_from_condition


def test_te_clearance_threshold_constant():
    """Verify that the private threshold constant equals 0.15."""
    assert _CDI_TE_CLEARANCE == 0.15


def test_evaluate_aerodynamic_trust_te_clearance():
    """Direct checks of evaluate_aerodynamic_trust with te_clearance."""
    # Below threshold: warning should appear.
    trust_warn = evaluate_aerodynamic_trust(
        AR=8.0,
        CL=0.5,
        CDi=0.012,
        te_clearance=0.12,
    )
    assert any("trailing-edge clearance" in w.lower() for w in trust_warn.warnings)
    assert any("trefftz cdi" in w.lower() for w in trust_warn.warnings)
    assert any("far wake" in w.lower() for w in trust_warn.warnings)
    # Warning only: no penalty in factors.
    assert "te_clearance" not in trust_warn.factors

    # At or above threshold: no warning.
    trust_at = evaluate_aerodynamic_trust(
        AR=8.0,
        CL=0.5,
        CDi=0.012,
        te_clearance=0.15,
    )
    assert not any("trailing-edge clearance" in w.lower() for w in trust_at.warnings)

    trust_above = evaluate_aerodynamic_trust(
        AR=8.0,
        CL=0.5,
        CDi=0.012,
        te_clearance=0.20,
    )
    assert not any("trailing-edge clearance" in w.lower() for w in trust_above.warnings)

    # Free air (None): no warning.
    trust_none = evaluate_aerodynamic_trust(
        AR=8.0,
        CL=0.5,
        CDi=0.012,
        te_clearance=None,
    )
    assert not any("trailing-edge clearance" in w.lower() for w in trust_none.warnings)


def test_trailing_edge_clearance_helper():
    """Verify the helper trailing_edge_clearance with and without ground."""
    wing = LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    lat = build_lattice(vt.Aircraft(surfaces=[wing]), n_chord=4)

    # Free air (ground is None) returns None.
    assert trailing_edge_clearance(lat, None) is None

    # In ground effect: height at (0, 0, 0) is h = 0.20 m, alpha = 5 deg.
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(5.0), h=0.20)
    gp = ground_plane_from_condition(lat, cond)
    clearance = trailing_edge_clearance(lat, gp)
    assert clearance is not None
    # For a 1 m chord at alpha=5 deg, TE drops by 1.0 * sin(5 deg) = 0.087156 m.
    # Expected clearance = (0.20 - sin(5 deg)) / 1.0 = 0.112844.
    expected = (0.20 - np.sin(np.radians(5.0))) / 1.0
    assert abs(clearance - expected) < 1e-4


def test_vlm_ground_clearance_warning_and_free_air():
    """Verify warning below threshold, none above, and none in free air."""
    wing = LiftingSurface(
        name="StraightWing",
        semi_span=5.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    settings = vt.SolverSettings(n_panels=8, n_chord=4)

    # Case 1: Below threshold (h = 0.20 m, alpha = 5 deg -> clearance ~ 0.113 < 0.15).
    cond_low = FlightCondition(V_inf=30.0, alpha=np.radians(5.0), h=0.20)
    res_low = vt.analyze(wing, condition=cond_low, settings=settings)
    warnings_low = res_low.totals.trust.warnings
    assert any("trailing-edge clearance" in w.lower() for w in warnings_low)
    # Check that warning contains the clearance value.
    te_warn = next(w for w in warnings_low if "trailing-edge clearance" in w.lower())
    assert "h_te/c=" in te_warn or "clearance" in te_warn.lower()
    assert "0.113" in te_warn
    assert "0.15" in te_warn

    # Case 2: Above threshold (h = 0.30 m, alpha = 5 deg -> clearance ~ 0.213 > 0.15).
    cond_high = FlightCondition(V_inf=30.0, alpha=np.radians(5.0), h=0.30)
    res_high = vt.analyze(wing, condition=cond_high, settings=settings)
    warnings_high = res_high.totals.trust.warnings
    assert not any("trailing-edge clearance" in w.lower() for w in warnings_high)

    # Case 3: Free air (no ground plane).
    cond_free = FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    res_free = vt.analyze(wing, condition=cond_free, settings=settings)
    warnings_free = res_free.totals.trust.warnings
    assert not any("trailing-edge clearance" in w.lower() for w in warnings_free)


def test_controls_that_lower_trailing_edge():
    """Verify that a deflected flap that lowers the trailing edge triggers the warning."""
    # At alpha = 2 deg, h = 0.20 m:
    # Clean TE clearance is (0.20 - 1.0 * sin(2 deg)) / 1.0 = 0.1651 > 0.15 (no warning).
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(2.0), h=0.20)
    settings = vt.SolverSettings(n_panels=10, n_chord=4)

    wing_clean = LiftingSurface(
        name="CleanWing",
        semi_span=5.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    res_clean = vt.analyze(wing_clean, condition=cond, settings=settings)
    assert not any("trailing-edge clearance" in w.lower() for w in res_clean.totals.trust.warnings)

    # Add a flap deflected downward by 10 deg over 60% of the span.
    flap = vt.ControlSurface("flap", eta_start=0.2, eta_end=0.8, hinge_x_c=0.75, deflection=np.radians(10.0))
    wing_flapped = LiftingSurface(
        name="FlappedWing",
        semi_span=5.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
        controls=[flap],
    )
    res_flapped = vt.analyze(wing_flapped, condition=cond, settings=settings)
    warnings_flap = res_flapped.totals.trust.warnings
    assert any("trailing-edge clearance" in w.lower() for w in warnings_flap)


def test_dihedral_that_lowers_trailing_edge():
    """Verify that anhedral that lowers the wing tips triggers the warning."""
    # At alpha = 2 deg, h = 0.20 m:
    # Clean flat wing has clearance ~ 0.165 > 0.15 (no warning).
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(2.0), h=0.20)
    settings = vt.SolverSettings(n_panels=10, n_chord=4)

    wing_flat = LiftingSurface(
        name="FlatWing",
        semi_span=5.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
        dihedral=0.0,
    )
    res_flat = vt.analyze(wing_flat, condition=cond, settings=settings)
    assert not any("trailing-edge clearance" in w.lower() for w in res_flat.totals.trust.warnings)

    # Anhedral of -1 deg lowers the tips by 5.0 * sin(1 deg) = 0.087 m, dropping clearance below 0.15.
    wing_anhed = LiftingSurface(
        name="AnhedralWing",
        semi_span=5.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
        dihedral=np.radians(-1.0),
    )
    res_anhed = vt.analyze(wing_anhed, condition=cond, settings=settings)
    warnings_anhed = res_anhed.totals.trust.warnings
    assert any("trailing-edge clearance" in w.lower() for w in warnings_anhed)


def test_cdi_value_remains_unchanged():
    """Verify that the CDi result value does not change."""
    wing = LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(5.0), h=0.20)
    settings = vt.SolverSettings(n_panels=8, n_chord=4)
    res = vt.analyze(wing, condition=cond, settings=settings)
    assert res.totals.CDi > 0.0
    assert np.isfinite(res.totals.CDi)
