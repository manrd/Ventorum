"""The ground clearance comes from the geometry of the solve.

A vortex lattice with two or more chordwise panels puts a surface with a
shape at every section on the mean line of the shapes. A lattice with one
chordwise panel is flat. The ground placement, the clearance, the strike
check, the automatic chordwise count and the height step must use the
lattice of the solve, not a flat probe. The test case is a NACA 4412 with
its y coordinates negated (an inverted cambered wing): at alpha 0 its mean
line is about 0.038 c below the chord line.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from ventorum import gpu
from ventorum.airfoil import airfoil
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, LinearAirfoil, SolverSettings, WingSection
from ventorum.core.errors import GroundStrikeError
from ventorum.ground_effect import analyze_ground_effect
from ventorum.ground_effect.sweep import _fixed_n_chord
from ventorum.solvers.derivative_matrix import mesh_probe_conditions
from ventorum.solvers.height_derivatives import height_derivatives
from ventorum.solvers.lattice_base import DEFAULT_N_CHORD, MAX_AUTO_N_CHORD, ground_probe_count

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(autouse=True)
def _cpu():
    old_device, old_precision = gpu.get_device(), gpu._precision
    gpu.set_device("cpu")
    yield
    gpu.set_device(old_device)
    gpu.set_precision(old_precision)


def _inverted_4412():
    coords = airfoil("4412").coords.copy()
    coords[:, 1] *= -1.0
    return coords


def _wing(shape) -> Aircraft:
    return Aircraft(surfaces=[LiftingSurface(
        name="wing", semi_span=4.0,
        sections=[WingSection(y_frac=e, chord=1.0, shape=shape, airfoil=LinearAirfoil()) for e in (0.0, 1.0)],
    )], ref_point=np.array([0.25, 0.0, 0.0]))


def _mean_line_depth() -> float:
    """Return the depth [m] of the mean line below the chord line (chord 1 m)."""
    shape = airfoil(_inverted_4412()).normalized()
    x = np.linspace(0.0, 1.0, 2001)
    return -float(np.min(shape.camber(x)))


def test_probe_count_rule():
    sett = SolverSettings()
    assert ground_probe_count(_wing(None), sett) == 1
    assert ground_probe_count(_wing(_inverted_4412()), sett) == DEFAULT_N_CHORD
    assert ground_probe_count(_wing(_inverted_4412()), SolverSettings(n_chord=12)) == 12
    # The lifting line does not use the mean line.
    assert ground_probe_count(_wing(_inverted_4412()), SolverSettings(solver_type="linear")) == 1


def test_ground_effect_clearance_from_mean_line():
    """h_min is the clearance of the solve lattice (before: 0.05, the flat probe)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = analyze_ground_effect(_wing(_inverted_4412()), h=0.05, alpha_deg=0.0, n_panels=12)
    lattice = res.solver_result.details["lattice"]
    assert res.h_min == pytest.approx(res.solver_result.details["h_min"], rel=0, abs=1e-12)
    assert res.h_min == pytest.approx(0.05 - _mean_line_depth(), abs=5e-4)
    assert res.h_min_over_c == pytest.approx(res.h_min / res.c_ref, rel=1e-12)
    assert lattice.n_chord == res.n_chord == MAX_AUTO_N_CHORD
    # The strike limit uses the mean line too: the lowest point is about
    # 0.012 m above the ground and 4 m out, so the limit is below 0.2 deg.
    assert res.phi_strike_limit < 0.2


def test_ground_effect_n_chord_follows_rule():
    """n_chord = ceil(c / h_min) with h_min of the mean line (before: 10, from the flat probe)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = analyze_ground_effect(_wing(_inverted_4412()), h=0.1, alpha_deg=0.0, n_panels=12)
    expected = math.ceil(1.0 / (0.1 - _mean_line_depth()))
    assert DEFAULT_N_CHORD < expected < MAX_AUTO_N_CHORD
    assert res.n_chord == expected


def test_ground_strike_refused_before_solve():
    """At h 0.035 the mean line is below the ground: the case check refuses it (before: the solve did)."""
    with pytest.raises(GroundStrikeError, match="Ground strike: minimum clearance"):
        analyze_ground_effect(_wing(_inverted_4412()), h=0.035, alpha_deg=0.0, n_panels=12)


def test_sweep_count_follows_mean_line():
    sett = SolverSettings(solver_type="vlm", n_panels=12)
    ac = _wing(_inverted_4412())
    ac.compute_reference_values()
    n = _fixed_n_chord(ac, sett, np.array([0.1, 0.2]), np.array([0.0]), np.array([0.0]),
                       np.array([0.25, 0.0, 0.0]), "ref")
    assert n == math.ceil(1.0 / (0.1 - _mean_line_depth()))


def test_height_derivatives_step_from_mean_line():
    cond = FlightCondition(V_inf=30.0, alpha=0.0, h=0.1)
    res = height_derivatives(_wing(_inverted_4412()), cond, SolverSettings(n_chord=8, n_panels=12))
    # With a fixed count the probe is the solve lattice.
    assert res.h_min_m == pytest.approx(res.base.details["h_min"], rel=0, abs=1e-12)
    assert res.h_min_m == pytest.approx(0.1 - _mean_line_depth(), abs=1e-3)
    assert res.steps["h_m"] == pytest.approx(0.05 * res.h_min_m, rel=1e-12)


def test_height_derivatives_strike_refused_on_base_state():
    cond = FlightCondition(V_inf=30.0, alpha=0.0, h=0.035)
    with pytest.raises(GroundStrikeError, match="The base state touches the ground"):
        height_derivatives(_wing(_inverted_4412()), cond, SolverSettings(n_panels=12))


def test_derivative_matrix_height_step_from_mean_line():
    ac = _wing(_inverted_4412())
    ac.compute_reference_values()
    cond = FlightCondition(V_inf=30.0, alpha=0.0, h=0.1)
    conds = mesh_probe_conditions(ac, cond, SolverSettings(n_panels=12), ["static", "height"],
                                  np.array([0.25, 0.0, 0.0]))
    d_h = conds[-2].h - cond.h
    assert d_h == pytest.approx(0.05 * (0.1 - _mean_line_depth()), rel=0.02)


def test_flat_wing_unchanged():
    """A wing without a shape keeps the flat probe with one chordwise panel."""
    res = analyze_ground_effect(_wing(None), h=0.05, alpha_deg=0.0, n_panels=12)
    assert res.h_min == 0.05
    assert res.n_chord == 20
    assert res.solver_result.details["h_min"] == 0.05


def test_input_uncertainty_height_step_uses_the_mean_line_clearance():
    """The input-uncertainty height step uses the clearance of the solve lattice.

    The height step of the central differences is a fraction of the
    smallest clearance. With a flat probe the clearance of the inverted
    NACA 4412 at h 0.05 m is 0.05 m; the mean-line lattice of the solve
    is about 0.038 m lower, so the step was about four times too large.
    """
    from ventorum.core.input_uncertainty import _smallest_clearance

    ac = _wing(_inverted_4412())
    cond = FlightCondition(V_inf=30.0, alpha=0.0, h=0.05)
    h_min = _smallest_clearance(ac, cond, SolverSettings(solver_type="vlm", n_panels=12, n_chord=8))
    assert h_min < 0.05 - 0.5 * _mean_line_depth()
    flat = _smallest_clearance(_wing(None), cond, SolverSettings(solver_type="vlm", n_panels=12, n_chord=8))
    assert flat == pytest.approx(0.05, abs=1e-12)
