# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Steady trim imbalance of free flight.

The steady imbalance is the trim input of free flight. It must be the
steady load of the state that the march flies: with the control values of
``controls`` at t = 0 (absolute deflections), not the geometry deflection.
Its force must also equal the steady aerodynamic force plus the weight.
"""

from __future__ import annotations

import math

import numpy as np

import ventorum as vt
from ventorum.aero.system import freestream_direction, lift_direction, side_direction

_SETT = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=8, chord_spacing="uniform")
_INERTIA = np.diag([16.0, 24.0, 36.0])
_COND = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=math.radians(4.0))


def _wing_tail(deflection: float = 0.0) -> vt.Aircraft:
    wing = vt.LiftingSurface(
        name="wing", semi_span=3.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    elevator = vt.ControlSurface(name="elevator", eta_start=0.0, eta_end=1.0, hinge_x_c=0.5, deflection=deflection)
    tail = vt.LiftingSurface(
        name="tail", semi_span=1.0, position=np.array([1.8, 0.0, 0.0]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.5)],
        controls=[elevator],
    )
    ac = vt.Aircraft(name="wing_tail", surfaces=[wing, tail], ref_point=np.array([-0.15, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def test_steady_imbalance_uses_the_control_values_at_t0():
    """controls = 0 on a -3 deg elevator gives the steady imbalance of the 0 deg state, with a note."""
    run = vt.free_flight(
        _wing_tail(math.radians(-3.0)), _COND, _SETT, mass=80.0, inertia=_INERTIA, dofs=(), n_steps=1,
        controls={"elevator": lambda t: 0.0},
    )
    ref = vt.free_flight(_wing_tail(0.0), _COND, _SETT, mass=80.0, inertia=_INERTIA, dofs=(), n_steps=1)
    np.testing.assert_allclose(run.steady_moment_imbalance_body, ref.steady_moment_imbalance_body, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(run.steady_force_imbalance_body, ref.steady_force_imbalance_body, rtol=1e-9, atol=1e-9)
    assert any("control values of controls at t = 0" in note for note in run.notes)
    assert not any("control values of controls at t = 0" in note for note in ref.notes)


def test_steady_imbalance_without_controls_uses_the_geometry_deflection():
    """Without controls the steady imbalance is that of the geometry deflection, without a note."""
    defl = vt.free_flight(_wing_tail(math.radians(-3.0)), _COND, _SETT, mass=80.0, inertia=_INERTIA, dofs=(), n_steps=1)
    flat = vt.free_flight(_wing_tail(0.0), _COND, _SETT, mass=80.0, inertia=_INERTIA, dofs=(), n_steps=1)
    assert abs(defl.steady_moment_imbalance_body[1] - flat.steady_moment_imbalance_body[1]) > 1.0
    assert not any("control values of controls at t = 0" in note for note in defl.notes)


def test_steady_force_imbalance_is_aerodynamic_force_plus_weight():
    """At alpha 4 deg the body-axes steady force is qS (CD d0 + CY S0 + CL L0) - m g e_z."""
    mass, g = 80.0, 9.80665
    res = vt.free_flight(_wing_tail(), _COND, _SETT, mass=mass, inertia=_INERTIA, dofs=(), n_steps=1, gravity=g)
    tot = res.base.totals
    a, b = _COND.alpha, _COND.beta
    qS = 0.5 * _COND.rho * _COND.V_inf**2 * float(res.S_ref)
    aero = qS * (
        float(tot.CDi) * freestream_direction(a, b)
        + float(tot.CY) * side_direction(a, b)
        + float(tot.CL) * lift_direction(a)
    )
    expected = aero - mass * g * lift_direction(a)
    np.testing.assert_allclose(res.steady_force_imbalance_body, expected, rtol=1e-12, atol=1e-9)
