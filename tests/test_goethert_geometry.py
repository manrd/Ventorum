# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Geometry-level check of the Goethert form of the Prandtl-Glauert rule.

The vortex lattice applies the rule inside the solver (docs/theory/compressibility.md;
B. H. Goethert, NACA TM 1105, 1946). These tests do not use that path. They
build the stretched wing by hand with the public geometry types, solve it
incompressibly, and transform the coefficients back.

The rule follows from the transformation on the theory page. The stretch
x* = x / beta along the free stream keeps the circulation and the physical
loads of each strip (to first order in the angle of attack). The span does not
change. The stretched wing has the area S / beta, the chord c / beta and the
moment arms x / beta. Thus every force and moment coefficient of the stretched
wing, with its own reference values, is beta times the coefficient of the
physical wing:

    C(M, alpha; wing) = C(0, alpha; stretched wing) / beta.

The rule is a small-perturbation rule. The solver applies the stretch along the
free-stream axis and keeps the physical normal on the right-hand side. These
tests stretch along the geometry x axis. The two sides thus differ by terms
that go to zero with alpha (the largest part is of order M^2 alpha^2). Measured with the wing of these tests at
M = 0.5 (n_panels 16, n_chord 4), the largest relative difference is:

    alpha [deg]   CL        CDi       Cm
    0.25          2.1e-5    6.9e-5    3.9e-5
    1             8.2e-5    2.7e-4    1.5e-4
    2             2.7e-4    7.2e-4    1.7e-4
    4             9.9e-4    2.3e-3    1.6e-4

The mesh error is larger: from (n_panels 16, n_chord 4) to (32, 8) at
alpha 2 deg, CL changes by 7e-4, CDi by 2.6e-3 and Cm by 2.5e-3 (relative).
At M = 1e-4 the two sides agree to 4e-12, so the hand-built stretch is exact.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.datatypes import FlightCondition, LiftingSurface, SolverSettings, WingSection

MACH = 0.5
X_REF = 0.4  # moment reference point x [m] of the physical wing
SETTINGS = SolverSettings(solver_type="vlm", n_panels=16, n_chord=4)
FIELDS = ("CL", "CDi", "Cm")

# Relative tolerances. About two times the largest measured difference (see
# the module docstring): the second-order term of the rule, not round-off.
RTOL_SMALL_ALPHA = 2.0e-4  # alpha 0.25 deg; measured 6.9e-5
RTOL = 5.0e-3  # alpha 1 to 4 deg; measured 2.3e-3


def _wing(stretch: float) -> LiftingSurface:
    """Swept, tapered wing of aspect ratio 6, stretched in x by *stretch*."""
    return LiftingSurface(
        name="Wing",
        semi_span=3.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.25 * stretch),
            WingSection(y_frac=1.0, chord=0.75 * stretch),
        ],
        # The leading edge x grows with the span by tan(sweep): stretch it.
        sweep_le=math.atan(math.tan(math.radians(30.0)) * stretch),
        dihedral=math.radians(3.0),
    )


def _pair(mach: float, alpha_deg: float) -> tuple[dict, dict]:
    """Coefficients of the solver at *mach* and of the rule on the stretched wing."""
    beta = math.sqrt(1.0 - mach * mach)
    physical = vt.Aircraft(name="Physical", surfaces=[_wing(1.0)], ref_point=np.array([X_REF, 0.0, 0.0]))
    stretched = vt.Aircraft(
        name="Stretched", surfaces=[_wing(1.0 / beta)], ref_point=np.array([X_REF / beta, 0.0, 0.0])
    )
    alpha = math.radians(alpha_deg)
    with warnings.catch_warnings():
        # The stated Mach number does not agree with V_inf / a: intended here.
        warnings.simplefilter("ignore", RuntimeWarning)
        res_m = vt.analyze(physical, condition=FlightCondition(V_inf=50.0, alpha=alpha, mach=mach), settings=SETTINGS)
    res_0 = vt.analyze(stretched, condition=FlightCondition(V_inf=50.0, alpha=alpha), settings=SETTINGS)
    solver = {f: float(getattr(res_m.totals, f)) for f in FIELDS}
    rule = {f: float(getattr(res_0.totals, f)) / beta for f in FIELDS}
    return solver, rule


def _rel(solver: dict, rule: dict) -> dict:
    return {f: abs(rule[f] - solver[f]) / abs(solver[f]) for f in FIELDS}


@pytest.mark.parametrize("alpha_deg", [1.0, 2.0, 4.0])
def test_goethert_stretched_geometry(alpha_deg):
    """Solver at M 0.5 agrees with the incompressible stretched wing times 1/beta."""
    solver, rule = _pair(MACH, alpha_deg)
    err = _rel(solver, rule)
    for f in FIELDS:
        assert err[f] < RTOL, f"alpha {alpha_deg} deg, {f}: solver {solver[f]:.6g}, rule {rule[f]:.6g}"


def test_goethert_difference_vanishes_at_small_alpha():
    """The difference is a small-perturbation term: it goes to zero with alpha."""
    solver, rule = _pair(MACH, 0.25)
    err = _rel(solver, rule)
    for f in FIELDS:
        assert err[f] < RTOL_SMALL_ALPHA, f"{f}: solver {solver[f]:.6g}, rule {rule[f]:.6g}"


def test_goethert_stretch_is_exact_at_zero_mach():
    """With beta near 1 the hand-built stretch gives the same lattice."""
    solver, rule = _pair(1.0e-4, 2.0)
    err = _rel(solver, rule)
    for f in FIELDS:
        assert err[f] < 1.0e-10


def test_goethert_check_is_not_trivial():
    """The compressible result differs clearly from the unstretched incompressible one."""
    solver, _ = _pair(MACH, 2.0)
    ac = vt.Aircraft(name="Physical", surfaces=[_wing(1.0)], ref_point=np.array([X_REF, 0.0, 0.0]))
    res = vt.analyze(ac, condition=FlightCondition(V_inf=50.0, alpha=math.radians(2.0)), settings=SETTINGS)
    assert solver["CL"] / float(res.totals.CL) - 1.0 > 0.05
