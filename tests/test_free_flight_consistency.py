# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Free flight: recorded state, controls and loads belong together.

A step that does not reach coupling_tol must record the state of its last
evaluated iterate, so that state-dependent controls and the loads match the
recorded state. A run with all degrees of freedom frozen and state-dependent
controls must check the ground clearance at step 0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.errors import InputError


def _flapped_wing(hinge_x_c: float) -> vt.Aircraft:
    wing = vt.LiftingSurface(
        name="wing", semi_span=3.0, n_panels=4,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
        controls=[vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=hinge_x_c)],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


_SETT = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")


@pytest.mark.parametrize("h", [None, 2.0], ids=["free_air", "ground"])
def test_non_converged_step_records_the_evaluated_state(h):
    """With max_subiterations = 1 the recorded controls follow the recorded state."""
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=math.radians(4.0), h=h)
    res = vt.free_flight(
        _flapped_wing(0.75), cond, _SETT, mass=5.0, inertia=np.diag([2.0, 1.0, 3.0]),
        dofs=("z", "pitch"), n_steps=4, controls={"flap": lambda t, state: 0.5 * state.pitch},
        max_subiterations=1,
    )
    assert any("did not reach coupling_tol" in note for note in res.notes)
    np.testing.assert_allclose(res.controls["flap"], 0.5 * np.asarray(res.pitch), rtol=0.0, atol=1e-15)


def test_frozen_run_with_state_controls_checks_ground_at_step_0():
    """A flap below the ground at t = 0 is refused also with dofs = ()."""
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0, h=0.12)

    def strike_at_start(t, state):
        _ = state
        return math.radians(30.0) if t == 0.0 else 0.0

    with pytest.raises(InputError, match="step 0"):
        vt.free_flight(
            _flapped_wing(0.5), cond, _SETT, mass=100.0, inertia=np.diag([20.0, 30.0, 40.0]),
            dofs=(), n_steps=2, controls={"flap": strike_at_start},
        )
