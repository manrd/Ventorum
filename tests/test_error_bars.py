# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the layer-1 numerical error bars (GCI from three mesh levels)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    SolverSettings,
    TabulatedAirfoil,
    WingSection,
)
from ventorum.core.error_bars import (
    DIVERGENT_NOTE,
    DRAG_BASIS_NOTE,
    LAYER_NOTE,
    NON_FINITE_NOTE,
    NOT_CONVERGED_ORDER_NOTE,
    ErrorBar,
    coefficient_error_bar,
    mesh_levels,
    numerical_error_bars,
    observed_order,
    refinement_ratios,
)


def _power_values(a, c, p, levels):
    # f(h) = a + c * h**p with h = 1 / N.
    return [a + c * (1.0 / n) ** p for n in levels]


def test_observed_order_recovers_p_with_unequal_ratios():
    # f(h) = 1.7 + 0.3 * h**p on N = (40, 28, 20), unequal ratios.
    levels = [40, 28, 20]
    r21, r32 = refinement_ratios(levels)
    assert r21 != pytest.approx(r32)
    for p in (1.0, 2.0, 2.7):
        f1, f2, f3 = _power_values(1.7, 0.3, p, levels)
        order = observed_order(f1, f2, f3, r21, r32)
        assert order is not None
        assert order == pytest.approx(p, abs=1e-8)
        bar = coefficient_error_bar("CL", [f1, f2, f3], levels)
        layer = bar.layers["numerical"]
        assert layer.state == "monotonic"
        assert layer.safety_factor == pytest.approx(1.25)
        assert layer.f_extrapolated == pytest.approx(1.7, abs=1e-10)


def test_gci_formula():
    # Hand-computed case: f1 = 1.0, f2 = 1.02, f3 = 1.06, N = (40, 28, 20).
    # eps21 = 0.02, eps32 = 0.04, R = 0.5 (monotonic).
    # r21 = 40/28, r32 = 28/20 = 1.4, s = +1.
    # Hand steps: ln|eps32/eps21| = ln 2 = 0.69314718; ln(r21) = 0.35667494.
    # p0 = 0.69314718 / 0.35667494 = 1.94335821 (q = 0).
    # q(p0) = ln((2 - 1) / (1.4^1.94335821 - 1)) = ln(1 / 0.922892) = 0.08012684
    #   (r21^p0 = 2 by construction of p0).
    # p1 = (0.69314718 + 0.08012684) / 0.35667494 = 2.16800770.
    # The fixed point is p = 2.17614283 (change below 1e-10 after 8 steps).
    # With 0.5 <= p <= 4.0: Fs = 1.25,
    # U = 1.25 * 0.02 / ((40/28)^2.17614283 - 1) = 0.025 / 1.17311906
    #   = 0.02131023.
    f1, f2, f3 = 1.0, 1.02, 1.06
    levels = [40, 28, 20]
    r21, r32 = refinement_ratios(levels)
    assert r21 == pytest.approx(40.0 / 28.0)
    assert r32 == pytest.approx(28.0 / 20.0)
    order = observed_order(f1, f2, f3, r21, r32)
    assert order is not None
    expected_u = 1.25 * abs(f2 - f1) / (r21**order - 1.0)
    bar = coefficient_error_bar("CL", [f1, f2, f3], levels)
    layer = bar.layers["numerical"]
    assert layer.state == "monotonic"
    assert layer.half_width == pytest.approx(expected_u)
    assert layer.half_width == pytest.approx(0.021310, abs=5e-6)
    assert bar.interval_low == pytest.approx(f1 - expected_u)
    assert bar.interval_high == pytest.approx(f1 + expected_u)


def test_oscillatory_state():
    bar = coefficient_error_bar("CL", [1.0, 1.1, 0.95], [40, 28, 20])
    layer = bar.layers["numerical"]
    assert layer.state == "oscillatory"
    assert layer.half_width == pytest.approx(3.0 * 0.5 * 0.15)
    assert layer.observed_order is None
    assert layer.f_extrapolated is None
    assert bar.interval_low == pytest.approx(1.0 - 3.0 * 0.5 * 0.15)
    assert bar.interval_high == pytest.approx(1.0 + 3.0 * 0.5 * 0.15)


def test_divergent_state():
    # The change grows towards the fine mesh: |eps21| > |eps32|.
    bar = coefficient_error_bar("CL", [1.0, 0.9, 0.85], [40, 28, 20])
    layer = bar.layers["numerical"]
    assert layer.state == "divergent"
    assert layer.half_width is None
    assert bar.interval_low is None
    assert bar.interval_high is None
    assert bar.status == "numerical_only"
    assert any("diverges with mesh refinement" in n for n in bar.notes)


def test_roundoff_state():
    bar = coefficient_error_bar("CY", [0.0, 1e-17, -1e-17], [40, 28, 20])
    layer = bar.layers["numerical"]
    assert layer.state == "roundoff"
    assert layer.half_width == pytest.approx(3.0 * 2e-17)


def test_order_outside_limits_uses_safety_factor_3():
    # f(h) = 1.7 + 0.3 * h**6 gives an observed order near 6.
    levels = [40, 28, 20]
    r21, r32 = refinement_ratios(levels)
    f1, f2, f3 = _power_values(1.7, 0.3, 6.0, levels)
    order = observed_order(f1, f2, f3, r21, r32)
    assert order is not None
    assert order == pytest.approx(6.0, abs=1e-6)
    bar = coefficient_error_bar("CL", [f1, f2, f3], levels)
    layer = bar.layers["numerical"]
    assert layer.state == "monotonic"
    assert layer.safety_factor == pytest.approx(3.0)
    assert layer.observed_order == pytest.approx(6.0, abs=1e-6)
    expected_u = 3.0 * abs(f2 - f1) / (r21**4.0 - 1.0)
    assert layer.half_width == pytest.approx(expected_u)


def test_status_values_are_checked():
    with pytest.raises(ValueError):
        ErrorBar(name="CL", value=1.0, status="calibrated")


def test_to_dict_is_json():
    bar = coefficient_error_bar("CL", [1.0, 1.02, 1.06], [40, 28, 20])
    d = bar.to_dict()
    text = json.dumps(d)
    assert json.loads(text)["layers"]["numerical"]["state"] == "monotonic"
    assert math.isfinite(json.loads(text)["value"])


def _rectangular_wing_ar8():
    # Rectangular wing of aspect ratio 8: span 10 m, chord 1.25 m.
    return LiftingSurface(
        name="rect",
        semi_span=5.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.25),
            WingSection(y_frac=1.0, chord=1.25),
        ],
    )


def test_rectangular_wing_bars_contain_the_fine_reference():
    wing = _rectangular_wing_ar8()
    condition = FlightCondition(alpha=np.radians(5.0))
    settings = SolverSettings(solver_type="vlm", n_panels=40, spacing="uniform")
    result = numerical_error_bars(wing, condition=condition, settings=settings)
    assert result.levels == [40, 28, 20]
    assert LAYER_NOTE in result.notes
    assert result.bars["CL"].state == "monotonic"
    assert result.bars["CD"].state == "monotonic"
    for name in ("CY", "Cl", "Cn"):
        assert result.bars[name].state == "roundoff"
    for bar in result.bars.values():
        assert bar.status == "numerical_only"
        assert bar.coverage is None
    ref = vt.analyze(
        wing,
        condition=condition,
        settings=SolverSettings(solver_type="vlm", n_panels=320, spacing="uniform"),
    )
    for name, ref_value in (("CL", ref.totals.CL), ("CD", ref.totals.CDi)):
        bar = result.bars[name]
        assert bar.interval_low is not None and bar.interval_high is not None
        assert bar.interval_low <= ref_value <= bar.interval_high


def test_axes_change_only_moments():
    wing = _rectangular_wing_ar8()
    condition = FlightCondition(alpha=np.radians(5.0), beta=np.radians(3.0))
    settings = SolverSettings(solver_type="vlm", n_panels=24)
    body = numerical_error_bars(wing, condition=condition, settings=settings, axes="body")
    stab = numerical_error_bars(wing, condition=condition, settings=settings, axes="stability")
    assert stab.axes == "stability"
    for name in ("CL", "CD", "CY", "Cm"):
        first, second = body.bars[name], stab.bars[name]
        assert (first.value, first.interval_low, first.interval_high) == (
            second.value,
            second.interval_low,
            second.interval_high,
        )
        assert first.state == second.state
    assert body.bars["Cl"].value != stab.bars["Cl"].value
    assert body.bars["Cn"].value != stab.bars["Cn"].value
    fine_moments = stab.results[0].moments("stability")
    assert stab.bars["Cl"].value == pytest.approx(fine_moments["Cl"])
    assert stab.bars["Cm"].value == pytest.approx(fine_moments["Cm"])
    assert stab.bars["Cn"].value == pytest.approx(fine_moments["Cn"])


def _tabulated_wing():
    # Rectangular wing AR 8 with a tabulated airfoil that has profile drag.
    deg = np.array([-5.0, 0.0, 5.0, 10.0])
    tabulated = TabulatedAirfoil(
        name="tab",
        alpha=np.radians(deg),
        Cl_data=np.array([-0.5, 0.0, 0.5, 1.0]),
        Cd_data=np.array([0.010, 0.008, 0.010, 0.015]),
    )
    return LiftingSurface(
        name="tab",
        semi_span=5.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.25, airfoil=tabulated),
            WingSection(y_frac=1.0, chord=1.25, airfoil=tabulated),
        ],
    )


def test_profile_drag_basis():
    wing_tab = _tabulated_wing()
    condition = FlightCondition(alpha=np.radians(5.0))
    with_tab = numerical_error_bars(
        wing_tab, condition=condition, settings=SolverSettings(solver_type="vlm", n_panels=20)
    )
    assert with_tab.drag_basis.startswith("CD_total")
    without_tab = numerical_error_bars(
        _rectangular_wing_ar8(),
        condition=condition,
        settings=SolverSettings(solver_type="vlm", n_panels=20),
    )
    assert without_tab.drag_basis.startswith("CDi only")


def test_small_mesh_is_refused():
    with pytest.raises(ValueError):
        numerical_error_bars(
            _rectangular_wing_ar8(), settings=SolverSettings(n_panels=12)
        )


def test_levels_use_real_integers():
    assert mesh_levels(41) == [41, 29, 21]
    assert refinement_ratios([41, 29, 21])[0] == pytest.approx(41.0 / 29.0)
    result = numerical_error_bars(
        _rectangular_wing_ar8(),
        condition=FlightCondition(alpha=np.radians(5.0)),
        settings=SolverSettings(solver_type="vlm", n_panels=41),
    )
    assert result.levels == [41, 29, 21]
    for bar in result.bars.values():
        assert bar.layers["numerical"].levels == [41, 29, 21]


def test_lifting_line_and_ground_effect_run():
    wing = _rectangular_wing_ar8()
    linear = numerical_error_bars(
        wing,
        condition=FlightCondition(alpha=np.radians(5.0)),
        settings=SolverSettings(solver_type="linear", n_panels=24),
    )
    assert linear.solver == "linear"
    ground = numerical_error_bars(
        wing,
        condition=FlightCondition(alpha=np.radians(5.0), h=1.25),
        settings=SolverSettings(solver_type="vlm", n_panels=24),
    )
    for result in (linear, ground):
        for bar in result.bars.values():
            assert math.isfinite(bar.value)
            if bar.interval_low is not None:
                assert math.isfinite(bar.interval_low)
                assert math.isfinite(bar.interval_high)


def _wing_and_tail():
    # Wing (semi-span 5 m) and a small tail (semi-span 1 m) at x = 6 m.
    wing = _rectangular_wing_ar8()
    tail = LiftingSurface(
        name="tail",
        semi_span=1.0,
        position=np.array([6.0, 0.0, 0.0]),
        sections=[
            WingSection(y_frac=0.0, chord=0.5),
            WingSection(y_frac=1.0, chord=0.5),
        ],
    )
    return Aircraft(name="wing_tail", surfaces=[wing, tail])


def test_min_panels_note_names_the_small_surface():
    # Regression: the strip count of a surface is a slice, and len() of a
    # slice raised a TypeError that hid the note.
    condition = FlightCondition(alpha=np.radians(5.0))
    settings = SolverSettings(
        solver_type="vlm", n_panels=24, proportional_panels=True, min_panels=8
    )
    result = numerical_error_bars(_wing_and_tail(), condition=condition, settings=settings)
    clip_notes = [n for n in result.notes if "min_panels=8" in n]
    assert len(clip_notes) == 1
    assert "'tail'" in clip_notes[0]
    assert "not refined with the others" in clip_notes[0]
    single = numerical_error_bars(
        _rectangular_wing_ar8(), condition=condition, settings=settings
    )
    assert not any("min_panels" in n for n in single.notes)


@pytest.mark.parametrize(
    "values",
    [(1.0, 1.1, 1.1), (1.0, 0.9, 0.9), (-1.0, -1.1, -1.1), (-1.0, -0.9, -0.9)],
)
def test_zero_coarse_change_is_divergent_for_both_signs(values):
    # Regression: with eps32 = 0 and eps21 != 0 the state depended on the
    # sign of eps21 (divergent or oscillatory). Index 1 is the fine mesh.
    bar = coefficient_error_bar("CL", list(values), [40, 28, 20])
    layer = bar.layers["numerical"]
    assert layer.state == "divergent"
    assert layer.half_width is None
    assert bar.interval_low is None
    assert bar.interval_high is None
    assert DIVERGENT_NOTE in bar.notes


def test_zero_fine_change_gives_nonzero_half_width():
    # Regression: R = 0 (f1 = f2, f3 different) gave a zero half-width.
    # eps21 = 0, eps32 = 0.05: U = 3 * max(0, 0.05) = 0.15.
    bar = coefficient_error_bar("CL", [1.0, 1.0, 1.05], [40, 28, 20])
    layer = bar.layers["numerical"]
    assert layer.state == "not_converged_order"
    assert layer.half_width == pytest.approx(0.15)
    assert bar.interval_low == pytest.approx(0.85)
    assert bar.interval_high == pytest.approx(1.15)
    assert NOT_CONVERGED_ORDER_NOTE in bar.notes


def test_not_converged_order_keeps_eps21_rule_for_positive_r():
    # For 0 < R < 1 the rule stays U = 3 * |eps21|. f2 - f1 = 1e-3 and
    # f3 - f2 = 1e300 give an order that overflows, so the iteration fails.
    bar = coefficient_error_bar("CL", [1.0, 1.001, 1e300], [40, 28, 20])
    layer = bar.layers["numerical"]
    assert layer.state == "not_converged_order"
    assert layer.half_width == pytest.approx(3.0 * 1e-3, rel=1e-9)


def test_levels_must_be_strictly_decreasing():
    with pytest.raises(ValueError):
        coefficient_error_bar("CL", [1.0, 1.02, 1.06], [40, 40, 28])
    with pytest.raises(ValueError):
        coefficient_error_bar("CL", [1.0, 1.02, 1.06], [20, 28, 40])


def test_non_finite_value_note_and_strict_json():
    # Regression: a non-finite value got the divergent note, and to_dict()
    # gave NaN, which is not strict JSON.
    bar = coefficient_error_bar("CL", [math.nan, 1.0, 1.1], [40, 28, 20])
    assert bar.notes == [NON_FINITE_NOTE]
    assert DIVERGENT_NOTE not in bar.notes
    assert bar.interval_low is None and bar.interval_high is None
    text = json.dumps(bar.to_dict(), allow_nan=False)
    data = json.loads(text)
    assert data["value"] is None
    assert data["layers"]["numerical"]["values"] == [None, 1.0, 1.1]


def test_drag_basis_mixing_gives_no_cd_bar(monkeypatch):
    # Regression: when a coarser level had no CD_total, its CD fell back to
    # CDi, so the CD bar mixed two drag bases.
    original = vt.analyze

    def analyze_without_total_on_coarse(*args, **kwargs):
        res = original(*args, **kwargs)
        if kwargs["settings"].n_panels == 14:
            res.totals.CD_total = None
        return res

    monkeypatch.setattr(vt, "analyze", analyze_without_total_on_coarse)
    result = numerical_error_bars(
        _tabulated_wing(),
        condition=FlightCondition(alpha=np.radians(5.0)),
        settings=SolverSettings(solver_type="vlm", n_panels=20),
    )
    assert result.levels == [20, 14, 10]
    assert result.drag_basis.startswith("CD_total")
    assert "CD" not in result.bars
    assert DRAG_BASIS_NOTE in result.notes
    assert "CL" in result.bars
    json.dumps(result.to_dict(), allow_nan=False)
    assert "| CD |" not in result.summary()
