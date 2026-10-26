# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the viscous 2D solver with flap-deflection continuation."""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.airfoil import airfoil
from ventorum.panel2d import solve_viscous_flapped
from ventorum.panel2d.viscous import ViscousResult, solve_viscous


@pytest.mark.slow
def test_zero_deflection_equals_solve_viscous():
    """Deflection 0 gives the same result as solve_viscous on the base section."""
    base = airfoil("2412").repanel(n_panels=160)
    r_base = solve_viscous(base, 0.0, 1e6)
    r_flapped = solve_viscous_flapped(base, 0.0, 1e6, x_hinge=0.75, deflection_deg=0.0)

    assert r_base.converged
    assert r_flapped.converged
    assert np.array_equal(r_flapped.Cl, r_base.Cl)
    assert np.array_equal(r_flapped.Cd, r_base.Cd)
    assert np.array_equal(r_flapped.Cm, r_base.Cm)
    assert r_flapped.reached_deflection == 0.0
    assert r_flapped.details["reached_deflection"] == 0.0
    assert r_flapped.step_history == [(0.0, True, int(r_flapped.iterations))]


@pytest.mark.slow
def test_continuation_converges_to_15_deg():
    """NACA 2412 at alpha 0 and Re 1e6 converges at 5, 10 and 15 deg; Cl rises with deflection."""
    base = airfoil("2412").repanel(n_panels=160)
    r5 = solve_viscous_flapped(base, 0.0, 1e6, x_hinge=0.75, deflection_deg=5.0)
    r10 = solve_viscous_flapped(base, 0.0, 1e6, x_hinge=0.75, deflection_deg=10.0)
    r15 = solve_viscous_flapped(base, 0.0, 1e6, x_hinge=0.75, deflection_deg=15.0)

    assert r5.converged
    assert r10.converged
    assert r15.converged
    assert r5.Cl < r10.Cl < r15.Cl
    assert r5.reached_deflection == 5.0
    assert r10.reached_deflection == 10.0
    assert r15.reached_deflection == 15.0
    assert r5.details["reached_deflection"] == 5.0
    assert len(r5.step_history) >= 2
    assert all(conv for _, conv, _ in r5.step_history)


@pytest.mark.slow
def test_continuation_matches_cold_where_cold_converges():
    """At 2.5 deg where cold start converges, continuation and cold agree within 0.5 % in Cl and 2 % in Cd."""
    base = airfoil("2412").repanel(n_panels=160)
    f25 = base.with_flap(0.75, 2.5)
    r_cold = solve_viscous(f25, 0.0, 1e6)
    r_cont = solve_viscous_flapped(base, 0.0, 1e6, x_hinge=0.75, deflection_deg=2.5)

    assert r_cold.converged
    assert r_cont.converged

    diff_cl = abs(r_cont.Cl - r_cold.Cl) / abs(r_cold.Cl)
    diff_cd = abs(r_cont.Cd - r_cold.Cd) / abs(r_cold.Cd)
    print(f"\n[continuation vs cold at 2.5 deg] diff Cl: {diff_cl * 100:.4f} %, diff Cd: {diff_cd * 100:.4f} %")
    assert diff_cl < 0.005
    assert diff_cd < 0.02


@pytest.mark.slow
def test_failed_step_is_reported():
    """An impossible case returns not converged with reached deflection and no exception."""
    base = airfoil("2412").repanel(n_panels=160)
    r40 = solve_viscous_flapped(base, 0.0, 1e6, x_hinge=0.75, deflection_deg=40.0)

    assert not r40.converged
    assert np.isnan(r40.Cl)
    assert np.isnan(r40.Cd)
    assert "reached" in r40.reason.lower()
    assert "deg" in r40.reason.lower()
    assert r40.reached_deflection == 38.75
    assert r40.details["reached_deflection"] == 38.75
    assert len(r40.step_history) > 0
    assert r40.step_history[-1][1] is False
    assert isinstance(r40.step_history[-1][2], (int, np.integer))


@pytest.mark.slow
def test_step_independence():
    """Steps 2.5 and 1.25 deg give the same 10 deg result within 0.2 % in Cl."""
    base = airfoil("2412").repanel(n_panels=160)
    r25 = solve_viscous_flapped(base, 0.0, 1e6, x_hinge=0.75, deflection_deg=10.0, step_deg=2.5)
    r125 = solve_viscous_flapped(base, 0.0, 1e6, x_hinge=0.75, deflection_deg=10.0, step_deg=1.25)

    assert r25.converged
    assert r125.converged

    diff_cl = abs(r25.Cl - r125.Cl) / abs(r25.Cl)
    print(f"\n[step independence at 10 deg] diff Cl: {diff_cl * 100:.4f} %")
    assert diff_cl < 0.002


def test_continuation_work_budget():
    """Each step receives its own work budget and exhausted budget is reported."""
    base = airfoil("2412").repanel(n_panels=60)
    r_exhausted = solve_viscous_flapped(base, 0.0, 1e6, x_hinge=0.75, deflection_deg=5.0, work_budget=2)
    assert not r_exhausted.converged
    assert np.isnan(r_exhausted.Cl)
    assert "work budget used up" in r_exhausted.reason
    assert "reached 0 deg" in r_exhausted.reason
    assert r_exhausted.reached_deflection == 0.0
    assert r_exhausted.details["reached_deflection"] == 0.0
    assert len(r_exhausted.step_history) >= 1
    assert r_exhausted.step_history[0][0] == 0.0
    assert r_exhausted.step_history[0][1] is False
    assert isinstance(r_exhausted.step_history[0][2], (int, np.integer))


def test_failure_at_zero_deflection_reports_details():
    """A failure at 0 deg step reports reached 0 deg and populates step history."""
    base = airfoil("2412").repanel(n_panels=60)
    r = solve_viscous_flapped(base, 0.0, 1e6, x_hinge=0.75, deflection_deg=10.0, work_budget=1)
    assert not r.converged
    assert "reached 0 deg" in r.reason
    assert r.reached_deflection == 0.0
    assert r.details["reached_deflection"] == 0.0
    assert isinstance(r.step_history, list)
    assert len(r.step_history) == 1
    assert r.step_history[0] == (0.0, False, r.step_history[0][2])
    assert isinstance(r.step_history[0][2], (int, np.integer))


@pytest.mark.slow
def test_hinge_treatment_none_does_not_raise():
    """With hinge_treatment='none' the node count changes between steps: no exception.

    The flapped shape has more nodes than the clean one, so each step starts
    cold. The steps beyond 0 deg are really tried (default iteration limit),
    and a failure is reported with the reached deflection.
    """
    base = airfoil("2412").repanel(n_panels=60)
    r = solve_viscous_flapped(base, 0.0, 1e6, x_hinge=0.75, deflection_deg=5.0, hinge_treatment="none")
    assert isinstance(r, ViscousResult)
    history = r.details["step_history"]  # (deflection [deg], converged, Newton iterations)
    assert len(history) >= 2
    assert history[0][0] == 0.0
    assert any(step[0] > 0.0 for step in history)
    if not np.all(r.converged):
        assert r.details["reached_deflection"] < 5.0
        assert "reached" in str(r.reason)
