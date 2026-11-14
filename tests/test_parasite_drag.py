# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the parasite-drag build-up (skin friction, form factors, components)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ventorum.aero.parasite import (
    MU_SL,
    RE_TRANSITION_MIN,
    BluffComponent,
    DragComponent,
    ParasiteDragResult,
    flat_plate_cf,
    form_factor_body,
    form_factor_lifting,
    parasite_drag,
    wetted_area_lifting_surface,
)
from ventorum.core.datatypes import FlightCondition, LiftingSurface, WingSection


def _rectangular_wing(span: float, chord: float) -> LiftingSurface:
    return LiftingSurface(
        name="RectWing",
        semi_span=span / 2.0,
        sections=[
            WingSection(y_frac=0.0, chord=chord),
            WingSection(y_frac=1.0, chord=chord),
        ],
        is_symmetric=True,
    )


def test_laminar_cf_at_re_1e6() -> None:
    cf = flat_plate_cf(1.0e6, x_transition=1.0)
    assert cf == pytest.approx(1.328e-3, rel=1e-12)


def test_turbulent_cf_at_re_1e7() -> None:
    expected = 0.455 / (math.log10(1.0e7)) ** 2.58
    assert flat_plate_cf(1.0e7) == pytest.approx(expected, rel=1e-12)


def test_mixed_cf_between_laminar_and_turbulent_and_monotone() -> None:
    re = 5.0e6
    cf_lam = flat_plate_cf(re, x_transition=1.0)
    cf_turb = flat_plate_cf(re, x_transition=0.0)
    assert cf_lam < cf_turb
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    values = [flat_plate_cf(re, x_transition=x) for x in grid]
    for lo, hi in zip(values[:-1], values[1:]):
        assert hi < lo
    assert values[0] == pytest.approx(cf_turb, rel=1e-12)
    assert values[-1] == pytest.approx(cf_lam, rel=1e-12)


def test_lifting_form_factor_raymer_against_closed_form() -> None:
    t_c = 0.12
    x_t = 0.3
    expected = (1.0 + 0.6 / x_t * t_c + 100.0 * t_c**4) * (math.cos(0.0)) ** 0.28
    assert form_factor_lifting(t_c, x_t, 0.0, method="raymer") == pytest.approx(expected, rel=1e-12)


def test_lifting_form_factor_torenbeek_against_closed_form() -> None:
    t_c = 0.12
    expected = 1.0 + 2.7 * t_c + 100.0 * t_c**4
    assert form_factor_lifting(t_c, 0.3, 0.0, method="torenbeek") == pytest.approx(expected, rel=1e-12)


def test_lifting_form_factor_sweep_trend() -> None:
    # Raymer keeps the sweep factor (cos sweep)^0.28, which falls with
    # sweep. The Torenbeek form 1 + 2.7 t/c + 100 (t/c)^4 has no sweep
    # term (owner decision: the normal-thickness variant could not be
    # cited precisely), so sweep does not change it.
    base = form_factor_lifting(0.12, 0.3, 0.0, method="raymer")
    swept = form_factor_lifting(0.12, 0.3, math.radians(30.0), method="raymer")
    assert swept < base
    assert swept == pytest.approx(base * math.cos(math.radians(30.0)) ** 0.28, rel=1e-12)
    base_t = form_factor_lifting(0.12, 0.3, 0.0, method="torenbeek")
    swept_t = form_factor_lifting(0.12, 0.3, math.radians(30.0), method="torenbeek")
    assert swept_t == base_t


def test_body_form_factor_decreases_with_fineness() -> None:
    fineness = [3.0, 4.0, 6.0, 8.0, 10.0]
    for method in ("raymer", "hoerner"):
        values = [form_factor_body(f, method=method) for f in fineness]
        for lo, hi in zip(values[:-1], values[1:]):
            assert hi < lo


def test_body_form_factor_raymer_against_closed_form() -> None:
    f = 8.0
    assert form_factor_body(f, method="raymer") == pytest.approx(1.0 + 60.0 / f**3 + f / 400.0, rel=1e-12)


def test_body_form_factor_hoerner_against_closed_form() -> None:
    f = 8.0
    expected = 1.0 + 1.5 * (1.0 / f) ** 1.5 + 7.0 * (1.0 / f) ** 3
    assert form_factor_body(f, method="hoerner") == pytest.approx(expected, rel=1e-12)


def test_wetted_area_rectangular_wing() -> None:
    span, chord, t_c = 8.0, 1.5, 0.12
    s_wet = wetted_area_lifting_surface(_rectangular_wing(span, chord), t_c)
    assert s_wet == pytest.approx(2.0 * span * chord * 1.024, rel=1e-12)


def test_wetted_area_half_wing() -> None:
    span, chord, t_c = 8.0, 1.5, 0.10
    surf = _rectangular_wing(span, chord)
    surf.is_symmetric = False
    s_wet = wetted_area_lifting_surface(surf, t_c)
    assert s_wet == pytest.approx(span * chord * (1.0 + 0.2 * t_c), rel=1e-12)


def test_roughness_raises_cf() -> None:
    re = 1.0e7
    smooth = flat_plate_cf(re)
    rough = flat_plate_cf(re, roughness=50.0e-6, length=2.0)
    assert rough > smooth


def test_component_sum_equals_total() -> None:
    condition = FlightCondition(V_inf=60.0, rho=1.225)
    s_ref = 12.0
    comps = [
        DragComponent(name="wing", wetted_area=20.0, reference_length=1.5, form_factor=1.26),
        DragComponent(name="fuselage", wetted_area=15.0, reference_length=6.0, form_factor=1.10),
    ]
    result = parasite_drag(comps, condition, s_ref)
    assert isinstance(result, ParasiteDragResult)
    assert result.cd0_total == pytest.approx(sum(result.cd0_by_component.values()), rel=1e-12)
    assert set(result.cd0_by_component) == {"wing", "fuselage"}
    assert result.cd0_total > 0.0


def test_interference_factor_scales_component() -> None:
    condition = FlightCondition(V_inf=60.0, rho=1.225)
    base = DragComponent(name="a", wetted_area=10.0, reference_length=2.0, form_factor=1.2)
    scaled = DragComponent(
        name="a", wetted_area=10.0, reference_length=2.0, form_factor=1.2, interference=1.2
    )
    cd_base = parasite_drag([base], condition, 10.0).cd0_total
    cd_scaled = parasite_drag([scaled], condition, 10.0).cd0_total
    assert cd_scaled == pytest.approx(1.2 * cd_base, rel=1e-12)


def test_laminar_fraction_lowers_component_drag() -> None:
    condition = FlightCondition(V_inf=60.0, rho=1.225)
    turb = DragComponent(name="a", wetted_area=10.0, reference_length=2.0, form_factor=1.2)
    lam = DragComponent(
        name="a", wetted_area=10.0, reference_length=2.0, form_factor=1.2, laminar_fraction=0.5
    )
    cd_turb = parasite_drag([turb], condition, 10.0).cd0_total
    cd_lam = parasite_drag([lam], condition, 10.0).cd0_total
    assert cd_lam < cd_turb


def test_total_uncertainty_is_root_sum_square() -> None:
    condition = FlightCondition(V_inf=60.0, rho=1.225)
    comps = [
        DragComponent(name="a", wetted_area=10.0, reference_length=2.0, form_factor=1.2,
                      uncertainty=0.10),
        DragComponent(name="b", wetted_area=5.0, reference_length=1.0, form_factor=1.3,
                      uncertainty=0.20),
    ]
    result = parasite_drag(comps, condition, 10.0)
    expected = math.sqrt(sum((result.relative_uncertainty_by_component[c.name]
                              * result.cd0_by_component[c.name]) ** 2 for c in comps))
    assert result.cd0_total_uncertainty == pytest.approx(expected, rel=1e-12)
    assert result.relative_uncertainty_by_component == {"a": 0.10, "b": 0.20}


def test_out_of_range_inputs_raise() -> None:
    with pytest.raises(ValueError):
        DragComponent(name="a", wetted_area=-1.0, reference_length=2.0, form_factor=1.2)
    with pytest.raises(ValueError):
        form_factor_lifting(0.5, 0.3, 0.0)
    with pytest.raises(ValueError):
        wetted_area_lifting_surface(_rectangular_wing(8.0, 1.5), 0.5)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        flat_plate_cf(0.0)
    with pytest.raises(ValueError):
        flat_plate_cf(1.0e6, x_transition=1.5)
    with pytest.raises(ValueError):
        flat_plate_cf(1.0e6, roughness=50.0e-6, length=None)
    with pytest.raises(ValueError):
        form_factor_lifting(0.0, 0.3, 0.0)
    with pytest.raises(ValueError):
        form_factor_lifting(0.12, 0.3, 0.0, method="unknown")
    with pytest.raises(ValueError):
        form_factor_body(0.0)
    with pytest.raises(ValueError):
        form_factor_body(5.0, method="unknown")
    with pytest.raises(ValueError):
        DragComponent(name="a", wetted_area=1.0, reference_length=0.0, form_factor=1.2)
    with pytest.raises(ValueError):
        DragComponent(name="a", wetted_area=1.0, reference_length=1.0, form_factor=0.9)
    with pytest.raises(ValueError):
        DragComponent(name="a", wetted_area=1.0, reference_length=1.0, form_factor=1.2,
                      laminar_fraction=2.0)
    condition = FlightCondition(V_inf=60.0, rho=1.225)
    with pytest.raises(ValueError):
        parasite_drag([], condition, 10.0)
    with pytest.raises(ValueError):
        parasite_drag(
            [DragComponent(name="a", wetted_area=1.0, reference_length=1.0, form_factor=1.2)],
            condition, 0.0,
        )
    with pytest.raises(ValueError):
        parasite_drag(
            [
                DragComponent(name="a", wetted_area=1.0, reference_length=1.0, form_factor=1.2),
                DragComponent(name="a", wetted_area=2.0, reference_length=1.0, form_factor=1.2),
            ],
            condition, 10.0,
        )


def _cf_turb(re: float) -> float:
    return 0.455 / math.log10(re) ** 2.58


def _cf_lam(re: float) -> float:
    return 1.328 / math.sqrt(re)


@pytest.mark.parametrize("x_tr", [0.0, 0.3, 0.6, 0.95])
def test_rough_cf_is_continuous_across_the_cutoff(x_tr: float) -> None:
    """The cutoff Reynolds number acts on the turbulent part only.

    Before the fix, above the cutoff the laminar fraction was dropped and
    Cf jumped by about 70 percent for a mixed-flow plate.
    """
    length, k = 2.0, 50.0e-6
    re_co = 38.21 * (length / k) ** 1.053
    below = flat_plate_cf(re_co * (1.0 - 1e-9), x_tr, roughness=k, length=length)
    above = flat_plate_cf(re_co * (1.0 + 1e-9), x_tr, roughness=k, length=length)
    assert abs(above - below) / below < 1e-6


def test_rough_cf_keeps_the_laminar_term_above_the_cutoff() -> None:
    length, k, x_tr = 2.0, 50.0e-6, 0.4
    re_co = 38.21 * (length / k) ** 1.053
    re = 3.0 * re_co
    smooth = flat_plate_cf(re, x_tr)
    rough = flat_plate_cf(re, x_tr, roughness=k, length=length)
    expected = smooth + (1.0 - x_tr) * (_cf_turb(re_co) - _cf_turb(re))
    assert rough == pytest.approx(expected, rel=1e-12)
    rough_turbulent = flat_plate_cf(re, 0.0, roughness=k, length=length)
    assert rough_turbulent == pytest.approx(_cf_turb(re_co), rel=1e-12)
    assert rough < rough_turbulent
    assert flat_plate_cf(re, 1.0, roughness=k, length=length) == pytest.approx(
        _cf_lam(re), rel=1e-12
    )


@pytest.mark.parametrize("re", [1.0e6, 5.0e6])
def test_mixed_cf_is_monotone_and_bounded_for_any_transition(re: float) -> None:
    """Cf does not rise with the laminar run and stays between the limits.

    Before the fix, a transition Reynolds number just above 1 gave a
    negative Cf, and short laminar runs gave more than the fully
    turbulent value.
    """
    x_grid = np.concatenate([[0.0], np.logspace(-8.0, 0.0, 801)])
    values = np.array([flat_plate_cf(re, x_tr) for x_tr in x_grid])
    cf_turb = _cf_turb(re)
    cf_lam = _cf_lam(re)
    assert np.all(np.diff(values) <= 1e-15 * cf_turb)
    assert np.all(values <= cf_turb * (1.0 + 1e-14))
    assert np.all(values >= cf_lam * (1.0 - 1e-14))
    assert values[0] == pytest.approx(cf_turb, rel=1e-14)
    assert values[-1] == pytest.approx(cf_lam, rel=1e-14)


def test_transition_guard_is_the_equal_friction_reynolds_number() -> None:
    assert 1.0e4 < RE_TRANSITION_MIN < 1.5e4
    assert _cf_turb(RE_TRANSITION_MIN) == pytest.approx(_cf_lam(RE_TRANSITION_MIN), rel=1e-12)
    re = 1.0e6
    x0 = RE_TRANSITION_MIN / re
    below = flat_plate_cf(re, x0 * (1.0 - 1e-9))
    above = flat_plate_cf(re, x0 * (1.0 + 1e-9))
    assert below == pytest.approx(_cf_turb(re), rel=1e-15)
    assert abs(above - below) / below < 1e-9


def test_viscosity_constant_and_argument() -> None:
    from ventorum.core import constants
    from ventorum.viscous import alpha_correction

    # One viscosity constant in the package (ventorum.core.constants.MU_SL).
    assert MU_SL is constants.MU_SL
    assert alpha_correction.MU_SL is constants.MU_SL
    condition = FlightCondition(V_inf=60.0, rho=1.225)
    comp = DragComponent(name="a", wetted_area=10.0, reference_length=2.0, form_factor=1.2)
    default = parasite_drag([comp], condition, 10.0).cd0_total
    explicit = parasite_drag([comp], condition, 10.0, mu=MU_SL).cd0_total
    assert default == explicit
    re = 1.225 * 60.0 * 2.0 / (2.0 * MU_SL)
    expected = _cf_turb(re) * 1.2 * 10.0 / 10.0
    doubled = parasite_drag([comp], condition, 10.0, mu=2.0 * MU_SL).cd0_total
    assert doubled == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError):
        parasite_drag([comp], condition, 10.0, mu=0.0)


def test_wetted_area_needs_no_sweep_or_dihedral_factor() -> None:
    """semi_span runs along the dihedral line and the chords are streamwise."""
    span, chord, t_c = 8.0, 1.5, 0.12
    plain = wetted_area_lifting_surface(_rectangular_wing(span, chord), t_c)
    swept = _rectangular_wing(span, chord)
    swept.sweep_le = math.radians(35.0)
    tilted = _rectangular_wing(span, chord)
    tilted.dihedral = math.radians(20.0)
    assert wetted_area_lifting_surface(swept, t_c) == pytest.approx(plain, rel=1e-12)
    assert wetted_area_lifting_surface(tilted, t_c) == pytest.approx(plain, rel=1e-12)


def test_wetted_area_uses_section_z_offsets() -> None:
    """With z_le on the sections, semi_span is the y extent; the span length grows.

    Before the fix, the area used the projected span only.
    """
    semi, chord, t_c, rise = 4.0, 1.5, 0.12, 1.0
    surf = LiftingSurface(
        name="Gull",
        semi_span=semi,
        sections=[
            WingSection(y_frac=0.0, chord=chord, z_le=0.0),
            WingSection(y_frac=1.0, chord=chord, z_le=rise),
        ],
        is_symmetric=True,
    )
    expected = 2.0 * (2.0 * chord * math.hypot(semi, rise)) * (1.0 + 0.2 * t_c)
    assert wetted_area_lifting_surface(surf, t_c) == pytest.approx(expected, rel=1e-12)


def test_wetted_area_of_the_exposed_part() -> None:
    """The covered root span is left out on each half (tapered wing)."""
    semi, c_root, c_tip, t_c, covered = 5.0, 2.0, 1.0, 0.12, 0.6
    surf = LiftingSurface(
        name="Taper",
        semi_span=semi,
        sections=[WingSection(y_frac=0.0, chord=c_root), WingSection(y_frac=1.0, chord=c_tip)],
        is_symmetric=True,
    )
    c_cut = c_root + (c_tip - c_root) * covered / semi
    exposed = 2.0 * 0.5 * (c_cut + c_tip) * (semi - covered)
    expected = 2.0 * exposed * (1.0 + 0.2 * t_c)
    got = wetted_area_lifting_surface(surf, t_c, covered_root_span=covered)
    assert got == pytest.approx(expected, rel=1e-12)
    gross = wetted_area_lifting_surface(surf, t_c)
    assert gross == pytest.approx(2.0 * 2.0 * 0.5 * (c_root + c_tip) * semi * (1.0 + 0.2 * t_c),
                                  rel=1e-12)
    assert got < gross
    for bad in (-0.1, semi, 2.0 * semi, float("nan")):
        with pytest.raises(ValueError):
            wetted_area_lifting_surface(surf, t_c, covered_root_span=bad)


def test_bluff_component_drag_and_reference_area_scaling() -> None:
    """Bluff component drag follows CD_pi * Q * S_pi / S_ref and scales with 1 / S_ref."""
    condition = FlightCondition(V_inf=55.0, rho=1.225)
    gear = BluffComponent(name="gear", frontal_area=0.18, cd_frontal=0.25, interference=1.15)
    assert gear.uncertainty == 0.20
    default_bluff = BluffComponent(name="mast", frontal_area=0.05, cd_frontal=1.0)
    assert default_bluff.interference == 1.0
    assert default_bluff.uncertainty == 0.20

    s_ref_1, s_ref_2 = 10.0, 16.0
    res_1 = parasite_drag([gear], condition, s_ref_1)
    res_2 = parasite_drag([gear], condition, s_ref_2)

    expected_1 = 0.25 * 1.15 * 0.18 / s_ref_1
    expected_2 = 0.25 * 1.15 * 0.18 / s_ref_2
    assert res_1.cd0_by_component["gear"] == pytest.approx(expected_1, rel=1e-15)
    assert res_2.cd0_by_component["gear"] == pytest.approx(expected_2, rel=1e-15)
    assert res_1.cd0_total * s_ref_1 == pytest.approx(res_2.cd0_total * s_ref_2, rel=1e-15)
    assert res_1.cd0_base_by_component == {"gear": 0.0}
    assert res_2.cd0_base_by_component == {"gear": 0.0}


def test_base_drag_term_and_reference_area_scaling() -> None:
    """Base term adds cd_base * base_area / S_ref while keeping friction drag unchanged."""
    condition = FlightCondition(V_inf=60.0, rho=1.225)
    base_area, cd_base = 0.12, 0.15
    body_plain = DragComponent(
        name="fuselage",
        wetted_area=18.0,
        reference_length=6.0,
        form_factor=1.15,
        interference=1.05,
    )
    body_with_base = DragComponent(
        name="fuselage",
        wetted_area=18.0,
        reference_length=6.0,
        form_factor=1.15,
        interference=1.05,
        base_area=base_area,
        cd_base=cd_base,
    )

    for s_ref in (10.0, 15.0):
        res_plain = parasite_drag([body_plain], condition, s_ref)
        res_base = parasite_drag([body_with_base], condition, s_ref)
        assert res_plain.cd0_base_by_component == {"fuselage": 0.0}
        assert res_base.cd0_base_by_component["fuselage"] == pytest.approx(
            cd_base * base_area / s_ref, rel=1e-15
        )
        assert res_base.cd0_base_by_component["fuselage"] * s_ref == pytest.approx(
            cd_base * base_area, rel=1e-15
        )
        friction_part = res_base.cd0_by_component["fuselage"] - res_base.cd0_base_by_component["fuselage"]
        assert friction_part == pytest.approx(res_plain.cd0_by_component["fuselage"], rel=1e-15)


def test_mixed_buildup_total_and_rss_uncertainty() -> None:
    """Mixed friction, base, and bluff build-up sums components and applies RSS uncertainty."""
    condition = FlightCondition(V_inf=60.0, rho=1.225)
    s_ref = 12.0
    wing = DragComponent(
        name="wing",
        wetted_area=20.0,
        reference_length=1.5,
        form_factor=1.26,
        uncertainty=0.10,
    )
    fuselage = DragComponent(
        name="fuselage",
        wetted_area=15.0,
        reference_length=6.0,
        form_factor=1.12,
        interference=1.02,
        uncertainty=0.12,
        base_area=0.10,
        cd_base=0.18,
    )
    wheel = BluffComponent(
        name="wheel",
        frontal_area=0.08,
        cd_frontal=0.25,
        interference=1.10,
        uncertainty=0.20,
    )
    result = parasite_drag([wing, fuselage, wheel], condition, s_ref)

    re_wing = condition.rho * condition.V_inf * wing.reference_length / MU_SL
    cd_wing = flat_plate_cf(re_wing) * wing.form_factor * wing.interference * wing.wetted_area / s_ref
    re_fuse = condition.rho * condition.V_inf * fuselage.reference_length / MU_SL
    cd_fuse_fric = (
        flat_plate_cf(re_fuse)
        * fuselage.form_factor
        * fuselage.interference
        * fuselage.wetted_area
        / s_ref
    )
    cd_fuse_base = 0.18 * 0.10 / s_ref
    cd_fuse = cd_fuse_fric + cd_fuse_base
    cd_wheel = 0.25 * 1.10 * 0.08 / s_ref
    expected_total = cd_wing + cd_fuse + cd_wheel
    expected_unc = math.sqrt((0.10 * cd_wing) ** 2 + (0.12 * cd_fuse) ** 2 + (0.20 * cd_wheel) ** 2)

    assert result.cd0_by_component["wing"] == pytest.approx(cd_wing, rel=1e-15)
    assert result.cd0_by_component["fuselage"] == pytest.approx(cd_fuse, rel=1e-15)
    assert result.cd0_by_component["wheel"] == pytest.approx(cd_wheel, rel=1e-15)
    assert result.cd0_base_by_component == {
        "wing": 0.0,
        "fuselage": pytest.approx(cd_fuse_base, rel=1e-15),
        "wheel": 0.0,
    }
    assert result.cd0_total == pytest.approx(sum(result.cd0_by_component.values()), rel=1e-15)
    assert result.cd0_total == pytest.approx(expected_total, rel=1e-15)
    assert result.relative_uncertainty_by_component == {"wing": 0.10, "fuselage": 0.12, "wheel": 0.20}
    assert result.cd0_total_uncertainty == pytest.approx(expected_unc, rel=1e-15)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "frontal_area": 0.1, "cd_frontal": 0.2},
        {"name": 123, "frontal_area": 0.1, "cd_frontal": 0.2},
        {"name": "strut", "frontal_area": 0.0, "cd_frontal": 0.2},
        {"name": "strut", "frontal_area": -0.1, "cd_frontal": 0.2},
        {"name": "strut", "frontal_area": float("nan"), "cd_frontal": 0.2},
        {"name": "strut", "frontal_area": float("inf"), "cd_frontal": 0.2},
        {"name": "strut", "frontal_area": 0.1, "cd_frontal": 0.0},
        {"name": "strut", "frontal_area": 0.1, "cd_frontal": -0.2},
        {"name": "strut", "frontal_area": 0.1, "cd_frontal": float("nan")},
        {"name": "strut", "frontal_area": 0.1, "cd_frontal": float("inf")},
        {"name": "strut", "frontal_area": 0.1, "cd_frontal": 0.2, "interference": 0.0},
        {"name": "strut", "frontal_area": 0.1, "cd_frontal": 0.2, "interference": -1.0},
        {"name": "strut", "frontal_area": 0.1, "cd_frontal": 0.2, "interference": float("nan")},
        {"name": "strut", "frontal_area": 0.1, "cd_frontal": 0.2, "interference": float("inf")},
        {"name": "strut", "frontal_area": 0.1, "cd_frontal": 0.2, "uncertainty": -0.01},
        {"name": "strut", "frontal_area": 0.1, "cd_frontal": 0.2, "uncertainty": float("nan")},
        {"name": "strut", "frontal_area": 0.1, "cd_frontal": 0.2, "uncertainty": float("inf")},
    ],
)
def test_bluff_component_invalid_inputs_raise(kwargs: dict) -> None:
    """Every validation rule on BluffComponent raises ValueError."""
    with pytest.raises(ValueError):
        BluffComponent(**kwargs)


@pytest.mark.parametrize(
    "base_kwargs",
    [
        {"base_area": 0.1, "cd_base": None},
        {"base_area": None, "cd_base": 0.2},
        {"base_area": 0.0, "cd_base": 0.2},
        {"base_area": -0.1, "cd_base": 0.2},
        {"base_area": float("nan"), "cd_base": 0.2},
        {"base_area": float("inf"), "cd_base": 0.2},
        {"base_area": 0.1, "cd_base": 0.0},
        {"base_area": 0.1, "cd_base": -0.2},
        {"base_area": 0.1, "cd_base": float("nan")},
        {"base_area": 0.1, "cd_base": float("inf")},
    ],
)
def test_drag_component_base_invalid_inputs_raise(base_kwargs: dict) -> None:
    """Every validation rule on DragComponent base-drag fields raises ValueError."""
    with pytest.raises(ValueError):
        DragComponent(
            name="fuselage",
            wetted_area=10.0,
            reference_length=4.0,
            form_factor=1.15,
            **base_kwargs,
        )


def test_duplicate_name_between_friction_and_bluff_raises() -> None:
    """Component names must be unique across DragComponent and BluffComponent."""
    condition = FlightCondition(V_inf=60.0, rho=1.225)
    friction = DragComponent(name="nacelle", wetted_area=3.0, reference_length=1.5, form_factor=1.2)
    bluff = BluffComponent(name="nacelle", frontal_area=0.1, cd_frontal=0.2)
    with pytest.raises(ValueError):
        parasite_drag([friction, bluff], condition, 10.0)


def test_existing_buildup_is_bit_identical() -> None:
    """An existing build-up without base drag keeps its exact bit-identical cd0_total."""
    condition = FlightCondition(V_inf=60.0, rho=1.225)
    comps = [
        DragComponent(name="wing", wetted_area=20.0, reference_length=1.5, form_factor=1.26),
        DragComponent(name="fuselage", wetted_area=15.0, reference_length=6.0, form_factor=1.10),
    ]
    result = parasite_drag(comps, condition, 12.0)
    assert result.cd0_total == float.fromhex("0x1.5537c4c97b36ap-7")
    assert result.cd0_base_by_component == {"wing": 0.0, "fuselage": 0.0}
    # Existing 5-argument ParasiteDragResult construction still works.
    legacy_shape = ParasiteDragResult(
        result.cd0_by_component,
        result.cd0_total,
        result.relative_uncertainty_by_component,
        result.cd0_total_uncertainty,
        result.s_ref,
    )
    assert legacy_shape.cd0_base_by_component == {}
