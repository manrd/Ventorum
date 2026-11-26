# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the flight-condition checks of angle-of-attack sweeps.

Regression: a lattice sweep checked each case three times
(:func:`ventorum.visualization.drag_polar.alpha_sweep`,
:meth:`LatticeSolver.solve_sweep` and the per-case set-up). That cost about
13 ms of host time per 257-case sweep. Now the first case gets all checks,
the other cases the check of ``alpha`` only, and the solver does not check
the cases of a sweep again. The errors and their messages stay the same.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.errors import InputError, ValidityError
from ventorum.solvers import lattice_base
from ventorum.utils import validation
from ventorum.utils.validation import (
    check_alpha_sweep_values,
    check_flight_condition_values,
    checked_conditions,
    condition_is_checked,
)


def _per_case_error(condition, alphas):
    """Return the error of the old per-case loop, or None."""
    try:
        for a in np.asarray(alphas, dtype=float).ravel():
            check_flight_condition_values(dataclasses.replace(condition, alpha=float(a)))
    except (InputError, ValidityError) as exc:
        return type(exc), str(exc)
    return None


def _sweep_error(condition, alphas):
    try:
        check_alpha_sweep_values(condition, alphas)
    except (InputError, ValidityError) as exc:
        return type(exc), str(exc)
    return None


@pytest.mark.parametrize("kw, alphas", [
    ({}, [0.0, 0.1, 0.2]),
    ({}, [0.0, math.nan, 0.2]),
    ({}, [0.0, 0.1, 4.0]),
    ({}, [math.inf, 0.1]),
    ({}, []),
    ({"V_inf": 0.0}, [0.0, math.nan]),
    ({"rho": -1.0}, [0.1, 5.0]),
    ({"beta": 4.0}, [0.0]),
    ({"h": -0.5}, [0.0, 0.1]),
    ({"mach": 0.8}, [0.0, 0.1]),
])
def test_sweep_check_gives_the_error_of_the_per_case_loop(kw, alphas):
    cond = dataclasses.replace(vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=0.05), **kw)
    assert _sweep_error(cond, alphas) == _per_case_error(cond, alphas)


def test_checked_conditions_scope():
    a = vt.FlightCondition(V_inf=30.0)
    b = vt.FlightCondition(V_inf=31.0)
    assert not condition_is_checked(a)
    with checked_conditions([a]):
        assert condition_is_checked(a) and not condition_is_checked(b)
        with checked_conditions([b]):
            assert condition_is_checked(a) and condition_is_checked(b)
        assert not condition_is_checked(b)
    assert not condition_is_checked(a)
    with pytest.raises(RuntimeError), checked_conditions([a]):
        raise RuntimeError("leave the block")
    assert not condition_is_checked(a)


def _count(monkeypatch):
    calls = []
    orig = validation.check_flight_condition_values

    def counting(fc):
        calls.append(fc.alpha)
        return orig(fc)

    monkeypatch.setattr(lattice_base, "check_flight_condition_values", counting)
    monkeypatch.setattr(validation, "check_flight_condition_values", counting)
    return calls


def _wing():
    return vt.LiftingSurface(name="w", semi_span=4.0,
                             sections=[vt.WingSection(y_frac=0.0, chord=1.0),
                                       vt.WingSection(y_frac=1.0, chord=1.0)])


@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_lattice_sweep_checks_the_full_condition_once_per_entry(monkeypatch, solver):
    _check_sweep_count(monkeypatch, solver, "cpu")


@pytest.mark.gpu
@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_gpu_lattice_sweep_checks_the_full_condition_once_per_entry(monkeypatch, solver):
    from ventorum import gpu

    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    _check_sweep_count(monkeypatch, solver, "gpu")


def _check_sweep_count(monkeypatch, solver, device):
    from ventorum import gpu

    old = gpu.get_device()
    gpu.set_device(device)
    try:
        calls = _count(monkeypatch)
        res = vt.analyze_sweep(_wing(), np.linspace(-2.0, 8.0, 11), solver=solver, n_panels=12, n_jobs=1)
    finally:
        gpu.set_device(old)
    assert len(res) == 11
    # analyze_sweep (base condition), alpha_sweep and solve_sweep: the first case
    # only. Before the change: 1 + 3 per case.
    assert len(calls) <= 3


def test_single_solve_still_checks_its_condition(monkeypatch):
    calls = _count(monkeypatch)
    vt.analyze(_wing(), vt.FlightCondition(V_inf=30.0, alpha=0.05), solver="vlm", n_panels=12)
    assert len(calls) >= 1


def test_direct_solve_lattice_still_refuses_a_bad_condition():
    ac = vt.Aircraft(surfaces=[_wing()])
    ac.compute_reference_values()
    from ventorum.solvers.factory import make_solver

    solver = make_solver("vlm")
    st = vt.SolverSettings(solver_type="vlm", n_panels=12)
    cond = vt.FlightCondition(V_inf=30.0, alpha=0.05)
    lattice = solver.build(ac, st, cond, None, ac.moment_reference())
    bad = dataclasses.replace(cond, V_inf=-1.0)
    with pytest.raises(InputError, match="V_inf"):
        solver.solve_lattice(lattice, bad, st, ac.S_ref, ac.b_ref, ac.c_ref)
