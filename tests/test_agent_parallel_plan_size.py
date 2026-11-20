"""Parallel plan size of the agent tools.

The parallel plan of a call gets the panel count of the solved mesh (with
the automatic n_chord of the case condition), not the free-air default of
4 chordwise panels.
"""

from __future__ import annotations

import numpy as np

import ventorum.agent.tools as tools
from ventorum.agent import polar_sweep, stability_derivatives
from ventorum.core.datatypes import FlightCondition, SolverSettings
from ventorum.solvers.lattice_base import CONTROL_MIN_AUTO_N_CHORD
from ventorum.utils.parallel import estimate_panels

# A wing with a control: the automatic n_chord is at least 16.
FLAP_WING = {"span_m": 10.0, "chord_m": 1.0,
             "controls": [{"name": "flap", "eta_start": 0.0, "eta_end": 0.6, "hinge_x_c": 0.75}]}
FC = {"V_inf_m_s": 40.0, "alpha_deg": 4.0}


def _record_plans(monkeypatch):
    """Record the panel count that each run_cases call of the tools gets."""
    seen: list[int | None] = []
    real = tools.run_cases

    def wrapped(func, items, n_panels=None, n_jobs="auto"):
        seen.append(n_panels)
        return real(func, items, n_panels, n_jobs)

    monkeypatch.setattr(tools, "run_cases", wrapped)
    return seen


def test_parallel_plan_gets_the_control_chord_count(monkeypatch):
    """Regression: the plan of a polar sweep and of the stability cases used n_chord 4.

    With a control the solve uses 16 chordwise panels, so the size class of the
    plan was 4 times too small.
    """
    monkeypatch.setattr(tools, "_gpu_takes", lambda n_cases, n_panels: False)
    seen = _record_plans(monkeypatch)
    settings = {"n_panels": 20}
    ac = tools.build_aircraft_from_spec(wing=dict(FLAP_WING))
    strips = 40
    assert estimate_panels(ac, SolverSettings(n_panels=20)) == strips * 4
    p = polar_sweep(wing=dict(FLAP_WING), alpha_start_deg=0.0, alpha_end_deg=2.0, alpha_step_deg=2.0,
                    flight_condition={"V_inf_m_s": 40.0}, settings=settings)
    assert p["status"] == "success", p
    assert seen == [strips * CONTROL_MIN_AUTO_N_CHORD], seen
    seen.clear()
    p = stability_derivatives(wing=dict(FLAP_WING), flight_condition=dict(FC), settings=settings)
    assert p["status"] == "success", p
    assert seen and seen[0] == strips * CONTROL_MIN_AUTO_N_CHORD, seen


def test_plan_panels_includes_the_ground_chord_count():
    """In ground effect the plan count is that of the solved mesh (more chordwise panels)."""
    ac = tools.build_aircraft_from_spec(wing={"span_m": 10.0, "chord_m": 1.0})
    sett = SolverSettings(n_panels=20)
    free = FlightCondition(V_inf=40.0, alpha=np.radians(2.0))
    ground = FlightCondition(V_inf=40.0, alpha=np.radians(2.0), h=0.15)
    assert tools._plan_panels(ac, sett, free) == estimate_panels(ac, sett) == 40 * 4
    n_ground = tools._plan_panels(ac, sett, ground)
    assert n_ground == tools._planned_panels(ac, sett, ground)
    assert n_ground > estimate_panels(ac, sett)
