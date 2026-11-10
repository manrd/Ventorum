# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for ``SpanwiseResult.Cl_freestream`` of the 3D panel solver.

The 3D panel strips give ``Cl_freestream = 2 Gamma / (V_inf c)`` [-] with
``Gamma = local_lift / (rho V_inf)`` [m^2/s], so
``Cl_freestream * q_inf * c`` is the lift per unit span [N/m], as for the
lattice solvers. The strip ``Cl`` of the 3D panel solver is the strip
pressure lift on the free-stream dynamic pressure and the strip area, also
with body rates (it does not use the local onset speed of the lattice
rule). These tests check both definitions, with and without rates.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.solvers.factory import make_solver


def _aircraft(taper: float = 1.0) -> Aircraft:
    """Wing of semi-span 3 m, root chord 1 m, NACA 2412."""
    surf = LiftingSurface(
        name="Wing",
        semi_span=3.0,
        sections=[WingSection(y_frac=0.0, chord=1.0, shape="2412"),
                  WingSection(y_frac=1.0, chord=taper, shape="2412")],
        is_symmetric=True,
    )
    return Aircraft(name="W", surfaces=[surf])


def _solve(ac, cond, **kwargs):
    """Solve with the numpy backend; ignore the trust warnings of the coarse mesh."""
    settings = SolverSettings(n_panels=6, n_chord=10, panel_backend="numpy", panel_cache=False, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_solver("panel3d").solve(ac, cond, settings)


@pytest.mark.parametrize("rates", [{}, {"p": 0.5}, {"r": 0.5}], ids=["none", "roll", "yaw"])
def test_cl_freestream_is_two_gamma_over_v_inf_c(rates: dict) -> None:
    """Cl_freestream = 2 Gamma / (V_inf c) = local_lift / (q_inf c) on every strip."""
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(5.0), **rates)
    res = _solve(_aircraft(0.5), cond)
    q_inf = 0.5 * cond.rho * cond.V_inf**2
    for sw in res.spanwise:
        assert sw.Cl_freestream is not None
        np.testing.assert_allclose(sw.Cl_freestream, 2.0 * sw.gamma / (cond.V_inf * sw.chord), rtol=1e-13)
        np.testing.assert_allclose(sw.Cl_freestream * q_inf * sw.chord, sw.local_lift, rtol=1e-12)
    item = res.to_dict()["spanwise"][0]
    assert item["Cl_freestream"] == pytest.approx(res.spanwise[0].Cl_freestream.tolist(), rel=1e-15)


@pytest.mark.parametrize("rates", [{}, {"p": 0.5}, {"r": 0.5}], ids=["none", "roll", "yaw"])
def test_panel_strip_cl_stays_on_the_free_stream_dynamic_pressure(rates: dict) -> None:
    """On a rectangular wing (strip area = width x chord) Cl equals Cl_freestream, also with rates.

    The 3D panel strip Cl is the strip pressure lift over ``q_inf`` and the
    strip area. With body rates it does not change to the local onset
    speed (the lattice rule ``2 Gamma / (|U_qc| c)``). This test records
    that normalisation.
    """
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(5.0), **rates)
    sw = _solve(_aircraft(1.0), cond).spanwise[0]
    np.testing.assert_allclose(sw.Cl, sw.Cl_freestream, rtol=1e-12, atol=1e-14)
