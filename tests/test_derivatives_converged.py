# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""The derivative tools report the convergence of each case of their differences.

Regression: static_derivatives, rate_derivatives and height_derivatives did
not read ``converged`` of their cases. A case that did not converge (for
example a nonlinear lifting-line case past the maximum lift) gave
derivatives with no flag in the result. The result now has
``case_converged`` and a note that names the failed cases.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import ventorum as vt
from ventorum.solvers.lattice_base import LatticeSolver

SETT = vt.SolverSettings(solver_type="vlm", n_chord=2)


def _ac() -> vt.Aircraft:
    wing = vt.LiftingSurface(name="wing", semi_span=2.0, n_panels=6, sections=[
        vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.5)])
    return vt.Aircraft(surfaces=[wing], ref_point=np.array([0.125, 0.0, 0.0]))


TOOLS = {
    "static": (lambda: vt.static_derivatives(_ac(), vt.FlightCondition(V_inf=30.0, alpha=0.05), SETT), 5,
               "alpha -"),
    "rate": (lambda: vt.rate_derivatives(_ac(), vt.FlightCondition(V_inf=30.0, alpha=0.05), SETT), 7, "p -"),
    "height": (lambda: vt.height_derivatives(_ac(), vt.FlightCondition(V_inf=30.0, alpha=0.05, h=0.3), SETT), 9,
               "alpha -"),
}


@pytest.mark.parametrize("tool", list(TOOLS))
def test_converged_cases_are_reported(tool):
    """All cases converge: every flag is True and no note names a failed case."""
    run, n_cases, _ = TOOLS[tool]
    res = run()
    assert res.case_converged == [True] * n_cases
    assert not any("did not converge" in n or "Not all cases" in n for n in res.notes)
    d = res.to_dict()
    assert d["case_converged"] == [True] * n_cases
    json.dumps(d)


@pytest.mark.parametrize("tool", list(TOOLS))
def test_a_case_that_did_not_converge_is_reported(tool, monkeypatch):
    """A case with converged=False gives False in case_converged and a note with its name."""
    run, n_cases, label = TOOLS[tool]
    real_solve_batch = LatticeSolver.solve_batch

    def one_case_fails(self, *args, **kwargs):
        results = real_solve_batch(self, *args, **kwargs)
        results[2].converged = False
        return results

    monkeypatch.setattr(LatticeSolver, "solve_batch", one_case_fails)
    res = run()
    expected = [True] * n_cases
    expected[2] = False
    assert res.case_converged == expected
    assert any("Not all cases" in n and f"({label}" in n for n in res.notes)
    assert res.to_dict()["case_converged"] == expected
