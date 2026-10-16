# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the parasite-drag build-up (skin friction, form factors, components)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ventorum.aero.parasite import (
    MU_SL,
    RE_TRANSITION_MIN,
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
    from ventorum.viscous.alpha_correction import MU_AIR_15C

    assert MU_SL == MU_AIR_15C
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
