# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for V1 viscous coupling ground kappa trust warning and trailing-edge guard."""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pytest

from ventorum.aero.polar_table import PolarTable
from ventorum.aero.system import ground_plane_from_condition
from ventorum.core.datatypes import (
    Aircraft,
    ControlSurface,
    FlightCondition,
    LiftingSurface,
    SolverSettings,
    TabulatedAirfoil,
    WingSection,
)
from ventorum.solvers.factory import make_solver
from ventorum.viscous.alpha_correction import _THICK_GROUND_FLAP_NOTE, _section_flap_shift, solve_viscous


@contextlib.contextmanager
def warnings_recorder():
    """Context manager to record warnings."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        yield recorded


def _make_flapped_wing(shape: str | None = "0012", deflection_deg: float = 10.0, hinge: float = 0.75) -> LiftingSurface:
    """Return an aspect ratio 8 rectangular wing with an optional control surface."""
    controls = []
    if deflection_deg != 0.0:
        controls = [ControlSurface("flap", 0.0, 1.0, hinge, np.radians(deflection_deg))]
    return LiftingSurface(
        name="TestWing",
        semi_span=4.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, shape=shape),
            WingSection(y_frac=1.0, chord=1.0, shape=shape),
        ],
        controls=controls,
    )


def _thin_polar() -> TabulatedAirfoil:
    """Return a synthetic thin-airfoil polar."""
    alpha = np.radians(np.linspace(-20.0, 25.0, 91))
    return TabulatedAirfoil(
        name="thin",
        alpha=alpha,
        Cl_data=2.0 * np.pi * alpha,
        Cd_data=np.full(alpha.shape, 0.01),
        Cm_data=np.zeros(alpha.shape),
        Re=1e6,
    )


def test_warning_appears_for_thick_polar_with_flap_at_hc_025():
    """Verify that the warning appears for a thick section with a flap at h/c = 0.25."""
    wing = _make_flapped_wing(shape="0012", deflection_deg=10.0)
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(2.0), h=0.25)
    settings = SolverSettings(n_chord=4, n_panels=24)
    polar = _thin_polar()

    with pytest.warns(RuntimeWarning, match="ground factor of the flap increment") as recorded:
        res = solve_viscous(wing, cond, settings, polar, model="vlm")

    assert res.converged
    # Only one RuntimeWarning per run.
    matching_warnings = [w for w in recorded if "ground factor of the flap increment" in str(w.message)]
    assert len(matching_warnings) == 1
    # Trust warning in trust.warnings.
    assert any("ground factor of the flap increment" in w for w in res.totals.trust.warnings)
    assert _THICK_GROUND_FLAP_NOTE in res.totals.trust.warnings
    assert _THICK_GROUND_FLAP_NOTE in res.details["notes"]


def test_no_warning_at_hc_035():
    """Verify that no warning appears at h/c = 0.35 above the 0.30 threshold."""
    wing = _make_flapped_wing(shape="0012", deflection_deg=10.0)
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(2.0), h=0.35)
    settings = SolverSettings(n_chord=4, n_panels=24)
    polar = _thin_polar()

    with warnings_recorder() as recorded:
        res = solve_viscous(wing, cond, settings, polar, model="vlm")

    assert res.converged
    matching = [w for w in recorded if "ground factor of the flap increment" in str(w.message)]
    assert len(matching) == 0
    assert not any("ground factor of the flap increment" in w for w in res.totals.trust.warnings)


def test_no_warning_without_flap():
    """Verify that no warning appears for a clean wing without a flap deflection."""
    wing = _make_flapped_wing(shape="0012", deflection_deg=0.0)
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(2.0), h=0.25)
    settings = SolverSettings(n_chord=4, n_panels=24)
    polar = _thin_polar()

    with warnings_recorder() as recorded:
        res = solve_viscous(wing, cond, settings, polar, model="vlm")

    assert res.converged
    matching = [w for w in recorded if "ground factor of the flap increment" in str(w.message)]
    assert len(matching) == 0
    assert not any("ground factor of the flap increment" in w for w in res.totals.trust.warnings)


def test_no_warning_for_thin_polar():
    """Verify that no warning appears for a thin section with t/c <= 0.04."""
    # NACA 0002 has t/c = 0.02 <= 0.04.
    wing = _make_flapped_wing(shape="0002", deflection_deg=10.0)
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(2.0), h=0.25)
    settings = SolverSettings(n_chord=4, n_panels=24)
    polar = _thin_polar()

    with warnings_recorder() as recorded:
        res = solve_viscous(wing, cond, settings, polar, model="vlm")

    assert res.converged
    matching = [w for w in recorded if "ground factor of the flap increment" in str(w.message)]
    assert len(matching) == 0
    assert not any("ground factor of the flap increment" in w for w in res.totals.trust.warnings)


def test_guard_raises_for_trailing_edge_under_ground_study_case():
    """Verify that the guard in _section_flap_shift raises when trailing edge is below ground."""
    # Study case: flap 0.3 c (hinge 0.7), h/c 0.1, alpha 10 deg.
    wing = _make_flapped_wing(shape="0012", deflection_deg=10.0, hinge=0.7)
    ac = Aircraft(name="StudyCase", surfaces=[wing])
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(10.0), h=0.1)
    solver = make_solver("vlm")
    settings = SolverSettings(n_chord=4, n_panels=24)
    lat = solver.build(ac, settings, cond, None, np.zeros(3))
    ground = ground_plane_from_condition(lat, cond, np.zeros(3))
    assert ground is not None

    with pytest.raises(ValueError, match=r"(?i)trailing edge.*below the ground") as exc_info:
        _section_flap_shift(lat, 0, 0.7, ground)

    msg = str(exc_info.value)
    # Must report strip index, height, and advice.
    assert "0" in msg
    assert "height" in msg.lower()
    assert "increase the height" in msg.lower() or "reduce the angle" in msg.lower()


def test_guard_passes_valid_case_bit_identical():
    """A valid case gives the flap shift of the section on the undeflected chord line.

    The guard does not change a valid case. The section is on the
    undeflected chord line, so the ground value is that of the clean
    strip (the section on the deflected trailing edge gave 0.5725625381466319).
    """
    wing = _make_flapped_wing(shape="0012", deflection_deg=10.0, hinge=0.75)
    ac = Aircraft(name="ValidCase", surfaces=[wing])
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(2.0), h=0.5)
    solver = make_solver("vlm")
    settings = SolverSettings(n_chord=4, n_panels=24)
    lat = solver.build(ac, settings, cond, None, np.zeros(3))
    ground = ground_plane_from_condition(lat, cond, np.zeros(3))
    assert ground is not None
    # Values of this code, same case. They are bit-identical on the machine
    # that made them; other numpy and BLAS builds (CI) change the last bits,
    # so the check uses a round-off tolerance.
    assert _section_flap_shift(lat, 0, 0.75, ground) == pytest.approx(0.5738308930925224, rel=1e-12)
    assert _section_flap_shift(lat, 0, 0.75, None) == pytest.approx(0.607922146145465, rel=1e-12)


def test_warning_from_polar_source_section():
    """A sweep with a thick polar source section warns once, also without a section shape."""
    wing = _make_flapped_wing(shape=None, deflection_deg=10.0)
    alpha = np.linspace(-20.0, 25.0, 91)
    cl = np.broadcast_to(2.0 * np.pi * np.radians(alpha), (2, 3, alpha.size)).copy()
    table = PolarTable(
        Re=np.array([1e5, 1e7]), flap_deg=np.array([-20.0, 0.0, 20.0]), alpha_deg=alpha,
        Cl=cl, Cd=np.full(cl.shape, 0.01), Cm=np.zeros(cl.shape),
        source_info={"source": "test", "section": "0012"},
    )
    conds = [FlightCondition(V_inf=40.0, alpha=np.radians(a), h=0.25) for a in (1.0, 2.0, 3.0)]
    with warnings_recorder() as recorded:
        results = solve_viscous(wing, conds, SolverSettings(n_chord=4, n_panels=24), table, model="vlm")
    matching = [w for w in recorded if "ground factor of the flap increment" in str(w.message)]
    assert len(matching) == 1
    assert all(_THICK_GROUND_FLAP_NOTE in r.totals.trust.warnings for r in results)


def _flap_lattice(deflection_deg: float, h: float):
    """Return the lattice and the ground plane of the flapped wing at alpha 2 deg."""
    ac = Aircraft(name="KappaCase", surfaces=[_make_flapped_wing(shape="0012", deflection_deg=deflection_deg)])
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(2.0), h=h)
    lat = make_solver("vlm").build(ac, SolverSettings(n_chord=4, n_panels=24), cond, None, np.zeros(3))
    return lat, ground_plane_from_condition(lat, cond, np.zeros(3))


@pytest.mark.parametrize("h", [0.5, 0.25])
def test_ground_factor_uses_the_undeflected_chord_line(h):
    """The ground factor kappa of the flap shift does not depend on the flap deflection.

    The 2D section of kappa is on the undeflected chord line of the strip.
    A section on the deflected trailing edge gave kappa 0.941835 (10 deg)
    against 0.943922 (no deflection) at h/c 0.5, and 0.883337 against
    0.884470 at h/c 0.25. The clean wing (no control) gives the reference
    value.
    """
    lat0, ground0 = _flap_lattice(0.0, h)
    free = _section_flap_shift(lat0, 0, 0.75, None)
    reference = _section_flap_shift(lat0, 0, 0.75, ground0) / free
    expected = {0.5: 0.943922, 0.25: 0.884470}[h]
    assert reference == pytest.approx(expected, abs=1e-6)
    for deflection in (10.0, -10.0, 20.0):
        lat, ground = _flap_lattice(deflection, h)
        te = 0.5 * (lat.te_left[0] + lat.te_right[0])
        # The trailing edge of the strip is deflected: the old section would see it.
        assert abs(float((te - lat.le_left[0]) @ lat.normal[0])) > 1e-2
        kappa = _section_flap_shift(lat, 0, 0.75, ground) / _section_flap_shift(lat, 0, 0.75, None)
        assert kappa == pytest.approx(reference, rel=1e-12)
