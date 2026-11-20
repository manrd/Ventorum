"""One InputError class for invalid user input.

The class lives in ``ventorum.core.errors``. It is a ``VentorumError`` and a
``ValueError``, so code that catches ``ValueError`` still catches it. The
agent module keeps its name, the input checks of the solvers raise it, and
the package exports it.
"""

import numpy as np
import pytest

import ventorum as vt
from ventorum.agent.response import error_type_of
from ventorum.core.errors import InputError, ValidityError, VentorumError
from ventorum.utils.validation import (
    check_flight_condition_values,
    validate_solver_settings,
    validate_surface,
)


def _wing() -> vt.LiftingSurface:
    return vt.LiftingSurface(name="W", semi_span=4.0, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])


def test_input_error_is_a_value_error_and_a_ventorum_error():
    err = InputError("bad value")
    assert isinstance(err, ValueError)
    assert isinstance(err, VentorumError)
    assert not isinstance(err, ValidityError)


def test_one_class_for_the_package_and_the_agent():
    from ventorum.agent import InputError as agent_init_error
    from ventorum.agent.schemas import InputError as agent_error
    from ventorum.agent.tools import InputError as tools_error

    assert agent_error is InputError
    assert agent_init_error is InputError
    assert tools_error is InputError
    assert vt.InputError is InputError


def test_agent_maps_input_error_to_invalid_input():
    assert error_type_of(InputError("bad key")) == "invalid_input"


@pytest.mark.parametrize("make", [
    lambda: check_flight_condition_values(vt.FlightCondition(V_inf=-1.0)),
    lambda: check_flight_condition_values(vt.FlightCondition(alpha=10.0)),
    lambda: validate_solver_settings(vt.SolverSettings(n_panels=2)),
    lambda: validate_surface(vt.LiftingSurface(name="W", semi_span=-1.0, sections=[
        vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])),
])
def test_validation_checks_raise_input_error(make):
    with pytest.raises(InputError):
        make()


def test_solver_input_check_raises_input_error():
    """A direct solver call refuses a zero speed with InputError (also a ValueError)."""
    solver = vt.VortexLatticeSolver()
    settings = vt.SolverSettings(solver_type="vlm", n_panels=8)
    with pytest.raises(InputError, match="V_inf"):
        solver.solve(_wing(), vt.FlightCondition(V_inf=0.0, alpha=np.radians(4.0)), settings)
    with pytest.raises(ValueError, match="V_inf"):
        vt.analyze(_wing(), condition=vt.FlightCondition(V_inf=0.0, alpha=np.radians(4.0)), settings=settings)


def test_mach_refusal_stays_a_validity_error():
    """A Mach number at or above 0.7 is a method limit, not an input error."""
    with pytest.raises(ValidityError):
        check_flight_condition_values(vt.FlightCondition(V_inf=50.0, mach=0.75))
