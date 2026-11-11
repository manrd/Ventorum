# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for ``SpanwiseResult.Cl_freestream`` of the 3D panel solver.

The 3D panel strips give ``Cl_freestream = 2 Gamma / (V_inf c)`` [-] with
``Gamma = local_lift / (rho V_inf)`` [m^2/s], so
``Cl_freestream * q_inf * c`` is the lift per unit span [N/m], as for the
lattice solvers. The strip ``Cl`` of the 3D panel solver is the strip
pressure lift on the strip area and the local dynamic pressure of the
onset at the strip quarter-chord point (the lattice rule); without body
rates that is ``q_inf``. These tests check both definitions, with and
without rates.
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


@pytest.mark.parametrize("rates", [{}, {"p": 0.5}, {"r": 0.5}, {"q": 0.3}], ids=["none", "roll", "yaw", "pitch"])
def test_panel_strip_cl_uses_the_local_dynamic_pressure(rates: dict) -> None:
    """Cl is the strip pressure lift on the local dynamic pressure of the strip (the lattice rule).

    Regression: with body rates the 3D panel strip Cl stayed on ``q_inf``.
    Now the dynamic pressure is that of the onset (free stream and body
    rates, no induced velocity) at the strip quarter-chord point, as for the
    lattice strips. On a rectangular wing (strip area = width x chord)
    ``Cl = Cl_freestream (V_inf / |U_qc|)^2``; without rates the two are equal.
    """
    from ventorum.aero.system import freestream_direction, rotation_velocity

    ac = _aircraft(1.0)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(5.0), **rates)
    sw = _solve(ac, cond).spanwise[0]
    ac.compute_reference_values()
    qc = np.column_stack([0.25 * sw.chord, sw.y, np.zeros_like(sw.y)])
    u = cond.V_inf * freestream_direction(cond.alpha, cond.beta)[None, :]
    rot = rotation_velocity(cond, qc, ac.moment_reference())
    if rot is not None:
        u = u + rot
    ratio = (cond.V_inf / np.linalg.norm(u, axis=1)) ** 2
    np.testing.assert_allclose(sw.Cl, sw.Cl_freestream * ratio, rtol=1e-12, atol=1e-14)
    if rates:
        assert np.max(np.abs(ratio - 1.0)) > 1e-4
    else:
        np.testing.assert_allclose(sw.Cl, sw.Cl_freestream, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("rates", [{"p": 0.5}, {"r": 0.5}, {"q": 0.3}], ids=["roll", "yaw", "pitch"])
@pytest.mark.parametrize("path", ["batch", "scalar"])
def test_panel_strip_cm_section_uses_the_local_dynamic_pressure(rates: dict, path: str, monkeypatch) -> None:
    """Cm_section is on the local dynamic pressure of the strip, Cd_i on q_inf, with body rates.

    Regression: with body rates the 3D panel strip Cm_section stayed on
    ``q_inf``, while the strip Cl used the local onset. The lattice section
    moment uses the local onset too. The reference here is the same panel
    solve with the strip rules of a case without rates (``has_rates``
    patched to False in the strip integration only): the strip pressure
    moment and lift do not change, so ``Cm_section`` and ``Cl`` change by
    the factor ``(V_inf / |U_qc|)^2`` and ``Cd_i`` does not change.
    """
    import ventorum.solvers.panel3d as factory_module
    from ventorum.aero.system import freestream_direction, rotation_velocity

    if path == "scalar":
        def scalar_strips(solver, results, conditions, maps, meta):
            return [factory_module._spanwise_reference(solver, r, c, maps, meta)
                    for r, c in zip(results, conditions)]

        monkeypatch.setattr(factory_module, "_spanwise_results_batch", scalar_strips)
    ac = _aircraft(0.5)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(5.0), **rates)
    sw = _solve(ac, cond).spanwise[0]
    monkeypatch.setattr(factory_module, "has_rates", lambda c: False)
    ref = _solve(ac, cond).spanwise[0]

    ac.compute_reference_values()
    qc = np.column_stack([0.25 * sw.chord, sw.y, np.zeros_like(sw.y)])
    u = cond.V_inf * freestream_direction(cond.alpha, cond.beta)[None, :]
    u = u + rotation_velocity(cond, qc, ac.moment_reference())
    ratio = (cond.V_inf / np.linalg.norm(u, axis=1)) ** 2
    assert np.max(np.abs(ratio - 1.0)) > 1e-4
    np.testing.assert_allclose(sw.Cm_section, ref.Cm_section * ratio, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(sw.Cl, ref.Cl * ratio, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(sw.Cd_i, ref.Cd_i)
    np.testing.assert_array_equal(sw.local_lift, ref.local_lift)


def test_batch_and_scalar_strip_cl_agree_with_rates(monkeypatch) -> None:
    """The batch strip integration and the scalar reference give the same strips with body rates."""
    from dataclasses import fields

    import ventorum.solvers.panel3d as factory_module

    ac = _aircraft(0.5)
    conds = [FlightCondition(V_inf=30.0, alpha=np.radians(5.0), p=0.4, q=-0.2, r=0.3),
             FlightCondition(V_inf=25.0, alpha=np.radians(3.0), beta=0.02, r=-0.5),
             FlightCondition(V_inf=30.0, alpha=np.radians(4.0))]
    settings = SolverSettings(n_panels=6, n_chord=10, panel_backend="numpy", panel_cache=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batch = make_solver("panel3d").solve(ac, conds, settings)

        def scalar_strips(solver, results, conditions, maps, meta):
            return [factory_module._spanwise_reference(solver, r, c, maps, meta)
                    for r, c in zip(results, conditions)]

        monkeypatch.setattr(factory_module, "_spanwise_results_batch", scalar_strips)
        # The same batch of panel solves, with the scalar strip integration.
        singles = make_solver("panel3d").solve(ac, conds, settings)
    for a, b in zip(batch, singles):
        for x, y in zip(a.spanwise, b.spanwise):
            for f in fields(y):
                va, vb = getattr(x, f.name), getattr(y, f.name)
                if isinstance(vb, np.ndarray):
                    scale = max(float(np.max(np.abs(vb))), 1e-12)
                    assert np.max(np.abs(va - vb)) <= 1e-12 * scale, f.name
