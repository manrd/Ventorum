# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""The 3D panel method refuses a compressible condition."""

from __future__ import annotations

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.errors import ValidityError
from ventorum.solvers.factory import make_solver


def _wing():
    return vt.LiftingSurface(
        name="w", semi_span=3.0,
        sections=[vt.WingSection(y_frac=e, chord=1.0, shape="0012") for e in (0.0, 1.0)],
    )


def test_mach_is_refused():
    """Regression: mach > 0 was ignored and the trust said the correction was applied."""
    settings = vt.SolverSettings(solver_type="panel3d", n_panels=4, n_chord=6)
    with pytest.raises(ValidityError, match="compressibility"):
        make_solver("panel3d").solve(_wing(), vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), mach=0.5),
                                     settings)


def test_mach_none_and_zero_still_solve():
    """mach None and mach 0 are incompressible and still solve."""
    settings = vt.SolverSettings(solver_type="panel3d", n_panels=4, n_chord=6)
    solver = make_solver("panel3d")
    a = solver.solve(_wing(), vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0)), settings)
    b = solver.solve(_wing(), vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), mach=0.0), settings)
    assert np.isfinite(a.totals.CL) and a.totals.CL == b.totals.CL
