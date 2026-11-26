# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Long-time limit of the indicial alpha step.

The step tilts the onset at fixed geometry by
``V_inf cos(beta) lift_direction(alpha) amplitude``, and the forces are on the
fixed mean wind axes. The reported steady values must be the long-time
limit of that input: at a loaded mean state (``CD`` includes ``-CL0``), in
sideslip (the ``cos(beta)`` factor) and in ground effect (the wing does not
rotate against the ground).
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


_CASES = {
    "loaded": (vt.FlightCondition(V_inf=20.0, alpha=math.radians(6.0)), {"CL": 0.01, "CD": 0.01, "Cm": 0.01}),
    "sideslip": (vt.FlightCondition(V_inf=30.0, alpha=0.0, beta=math.radians(20.0)), {"CL": 0.01, "Cm": 0.01}),
    "ground": (
        vt.FlightCondition(V_inf=20.0, alpha=math.radians(6.0), h=0.3),
        {"CL": 0.015, "CD": 0.015, "Cm": 0.02},
    ),
    "banked_ground": (
        vt.FlightCondition(V_inf=30.0, alpha=0.0, phi=math.radians(20.0), h=1.5),
        {"CL": 0.01, "Cm": 0.01},
    ),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_indicial_alpha_tends_to_its_steady_values(case):
    """At s = 80 the response is at the reported steady values."""
    cond, tols = _CASES[case]
    res = vt.indicial_response(_wing(), cond, vt.SolverSettings(n_chord=4), kind="alpha", s_end=80.0)
    for key, tol in tols.items():
        assert float(getattr(res, key)[-1]) == pytest.approx(res.steady[key], rel=tol), key
    if case == "loaded":
        # The fixed-axes drag response is dCDi/dalpha - CL0 < 0 at this state.
        assert res.steady["CD"] < 0.0
