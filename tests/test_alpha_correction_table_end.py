"""Strip flags of the alpha-correction coupling at the end of a polar table.

A polar table can stop before the stall (for example a ground polar that
stops where the section touches the ground). The table gives its end values
outside its range, so the section slope there is zero. Such strips are
outside the table, not post-stall.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from test_alpha_correction import make_rectangular_wing, make_stalling_polar
from ventorum.core.datatypes import FlightCondition, SolverSettings, TabulatedAirfoil
from ventorum.viscous.alpha_correction import _alpha_cl_max, _strip_stall_state, solve_viscous


def _linear_table(hi_deg: float) -> TabulatedAirfoil:
    """Return a 2 pi polar from -15 deg to *hi_deg* [deg] (no stall in the table)."""
    a = np.radians(np.arange(-15.0, hi_deg + 1e-9, 0.5))
    return TabulatedAirfoil(name=f"lin_to_{hi_deg:g}", alpha=a, Cl_data=2.0 * np.pi * a,
                            Cd_data=0.01 + 0.005 * a**2, Re=1e6)


def test_strips_past_the_end_of_a_cut_table_are_outside_the_table_not_post_stall():
    """Regression: strips past the end of a table that stops before stall were labelled post-stall.

    The strips had m <= 0 and the flag "post-stall, non-unique", and the branch
    search took the table end as the Cl maximum. Now they get the flag
    "outside polar table", no strip is post-stall, and the trust score keeps its
    out-of-table warning.
    """
    wing = make_rectangular_wing()
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(10.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cut = solve_viscous(wing, cond, SolverSettings(), _linear_table(6.0))
    d = cut.details
    assert cut.converged
    outside = np.asarray(d["outside_table_strips"])
    assert outside.any()
    assert not d["post_stall"] and not np.any(d["post_stall_strips"])
    assert not d["non_unique"]
    assert set(d["strip_flags"]) == {"attached", "outside polar table"}
    assert [f == "outside polar table" for f in d["strip_flags"]] == list(outside)
    np.testing.assert_array_equal(outside, np.degrees(d["alpha_eff"]) > 6.0)
    assert any("outside their polar table" in w for w in cut.totals.trust.warnings)


def test_end_maximum_of_a_table_is_not_a_stall():
    """The branch starts use an interior Cl maximum only."""
    assert _alpha_cl_max(_linear_table(6.0)) is None
    stall = make_stalling_polar(alpha_stall_deg=10.0)
    alpha, cl, _, _ = stall.tables()
    assert _alpha_cl_max(stall) == pytest.approx(float(alpha[int(np.argmax(cl))]), abs=0.0)
    assert 0.0 < _alpha_cl_max(stall) < float(alpha[-1])


def test_strip_past_an_interior_stall_and_outside_the_table_stays_post_stall():
    """A strip past the Cl maximum of its table is post-stall, also outside the table."""
    stall = make_stalling_polar(alpha_stall_deg=10.0)
    lin = _linear_table(6.0)
    hi = float(np.max(stall.tables()[0]))
    alpha_v = np.array([hi + 0.1, np.radians(8.0), np.radians(3.0), np.radians(-20.0), np.radians(15.0)])
    polars = [stall, lin, lin, lin, stall]
    m = np.array([0.0, 0.0, 2.0 * np.pi, 0.0, -1.5])
    post, out = _strip_stall_state(polars, alpha_v, m)
    np.testing.assert_array_equal(post, [True, False, False, False, True])
    np.testing.assert_array_equal(out, [True, True, False, True, False])
