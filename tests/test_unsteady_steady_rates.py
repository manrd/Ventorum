# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""The unsteady solvers refuse steady body rates in the flight condition.

The steady base solve uses the rates ``p``, ``q``, ``r`` of the condition, but
the time march and the harmonic loads do not. The functions must refuse the
rates instead of returning an inconsistent result.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.errors import InputError


def _wing() -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="wing", semi_span=3.0, n_panels=4,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )


def _call(name: str, cond: vt.FlightCondition):
    sett = vt.SolverSettings(n_chord=4)
    if name == "time_response":
        return vt.time_response(_wing(), cond, sett, n_steps=4)
    if name == "indicial_response":
        return vt.indicial_response(_wing(), cond, sett, kind="alpha", s_end=1.0)
    if name == "free_flight":
        return vt.free_flight(_wing(), cond, sett, mass=10.0, inertia=np.eye(3), n_steps=4)
    return vt.dynamic_derivatives(_wing(), cond, sett, k=(0.0,))


@pytest.mark.parametrize("name", ["time_response", "indicial_response", "free_flight", "dynamic_derivatives"])
@pytest.mark.parametrize("rate", ["p", "q", "r"])
def test_unsteady_functions_refuse_steady_rates(name, rate):
    """Each unsteady function refuses a nonzero p, q or r with InputError."""
    cond = vt.FlightCondition(V_inf=20.0, alpha=math.radians(4.0), **{rate: 0.5})
    with pytest.raises(InputError, match=f"{rate}=0.5 rad/s"):
        _call(name, cond)
