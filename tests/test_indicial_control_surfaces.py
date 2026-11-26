# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Indicial control step on surfaces with different built deflections.

One control name may be on several surfaces with different built
deflections. The step must add the amplitude to the deflection of each
surface. It used the deflection of the first surface as the base of all of
them, so the second surface got a step of (delta0_first - delta0_second +
amplitude) and the response divided by the amplitude was wrong by a large
factor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import ventorum as vt


def _two_surfaces(tail_deflection_deg: float) -> vt.Aircraft:
    flap_w = vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=0.0)
    flap_t = vt.ControlSurface(
        name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=math.radians(tail_deflection_deg),
    )
    wing = vt.LiftingSurface(
        name="wing", semi_span=3.0, n_panels=4, controls=[flap_w],
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    tail = vt.LiftingSurface(
        name="tail", semi_span=1.2, n_panels=4, position=np.array([3.0, 0.0, 0.3]), controls=[flap_t],
        sections=[vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.5)],
    )
    return vt.Aircraft(name="two_flaps", surfaces=[wing, tail], ref_point=np.array([0.25, 0.0, 0.0]))


def test_control_step_adds_the_amplitude_to_each_surface():
    """With a 4 deg tail flap the step response is near the one with a 0 deg tail flap."""
    cond = vt.FlightCondition(V_inf=20.0, alpha=0.0)
    sett = vt.SolverSettings(n_chord=4)
    offset = vt.indicial_response(_two_surfaces(4.0), cond, sett, kind="control", control="flap", s_end=20.0)
    plain = vt.indicial_response(_two_surfaces(0.0), cond, sett, kind="control", control="flap", s_end=20.0)
    # The built tail deflection changes the mean geometry a little, not the order of the response.
    assert float(offset.CL[-1]) == pytest.approx(float(plain.CL[-1]), rel=0.05)
    assert float(offset.CL[-1]) == pytest.approx(offset.steady["CL"], rel=0.05)
