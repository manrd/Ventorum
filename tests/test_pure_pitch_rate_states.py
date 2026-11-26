# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Pure pitch-rate derivatives at unloaded states with sideslip or bank.

The shortcut ``H_theta + H_z / (2 i k)`` holds only when the heave axis is
``cos(beta) L_0``. With sideslip, or banked in ground effect, the unloaded
state must give the same rate derivatives as a nearly unloaded state
(alpha 1e-6 deg), which uses the direct pure pitch-rate solve.
"""

from __future__ import annotations

import math

import pytest

import ventorum as vt


def _wing() -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="wing", semi_span=3.0, n_panels=8,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )


@pytest.mark.parametrize(
    "extra",
    [{"beta": math.radians(20.0)}, {"phi": math.radians(20.0), "h": 1.5}],
    ids=["sideslip", "banked_ground"],
)
def test_unloaded_rate_derivatives_have_no_jump(extra):
    """CL_q, Cm_q, CL_alpha_dot and Cm_alpha_dot are continuous at alpha = 0."""
    sett = vt.SolverSettings(n_chord=4)
    r0 = vt.dynamic_derivatives(_wing(), vt.FlightCondition(V_inf=30.0, alpha=0.0, **extra), sett, k=(0.05, 0.2))
    r1 = vt.dynamic_derivatives(
        _wing(), vt.FlightCondition(V_inf=30.0, alpha=math.radians(1e-6), **extra), sett, k=(0.05, 0.2),
    )
    for key in ("CL_q", "Cm_q", "CL_alpha_dot", "Cm_alpha_dot"):
        for i in range(2):
            assert r0.derivatives[key][i] == pytest.approx(r1.derivatives[key][i], rel=1e-4, abs=1e-8), key
