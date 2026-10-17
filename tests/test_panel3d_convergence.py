# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for mesh convergence and Richardson error estimates of Ventorum-3DP."""

from __future__ import annotations

import json

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.error_bars import coefficient_error_bar
from ventorum.panel3d.convergence import (
    ConvergenceReport,
    level_counts,
    order_and_extrapolation,
    panel_mesh_convergence,
)
from ventorum.panel3d.mesh import PanelMesh, check_mesh
from ventorum.panel3d.solver import PanelSolver
from ventorum.reference.analytic3d import sphere_quad_mesh


def _sphere_mesh(n_lat: int, n_lon: int) -> PanelMesh:
    """Closed unit sphere mesh of the reference module (poles on z)."""
    nodes, panels = sphere_quad_mesh(1.0, n_lat, n_lon)
    return PanelMesh(
        nodes=nodes,
        panels=panels,
        component=np.zeros(len(panels), dtype=int),
        kind=np.full(len(panels), "surface"),
        wake_edges=np.empty((0, 4), dtype=int),
    )


@pytest.fixture(scope="module")
def rect_wing() -> vt.Aircraft:
    """Rectangular wing (span 8 m, chord 1 m, AR 8) with a thick section."""
    surf = vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0, shape="0012"),
            vt.WingSection(y_frac=1.0, chord=1.0, shape="0012"),
        ],
        is_symmetric=True,
    )
    ac = vt.Aircraft(name="RectWing", surfaces=[surf])
    ac.compute_reference_values()
    return ac


@pytest.fixture(scope="module")
def cruise() -> vt.FlightCondition:
    """Flight condition of the wing convergence study."""
    return vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))


# Base mesh of the wing study: 6 spanwise and 16 chordwise panels. With 6
# chordwise panels the pressure lift of the coarsest levels is not yet
# monotonic (it changes by less than 1 percent between the levels). The
# mesh builder closes the open trailing edge of the library section; for
# the closed edge the combined refinement from 12 chordwise panels gives
# an observed CL order below 0.5 (outside the asymptotic range), and from
# 16 chordwise panels an order near 1 in CL, CDi and Cm.
_WING_BASE = {"n_panels": 6, "n_chord": 16}


@pytest.fixture(scope="module")
def wing_report(rect_wing, cruise) -> ConvergenceReport:
    """Convergence report of the rectangular wing from the base mesh up."""
    settings = vt.SolverSettings(**_WING_BASE)
    return panel_mesh_convergence(
        rect_wing, cruise, settings, levels=3, ratio=1.5,
    )


@pytest.fixture(scope="module")
def wing_next(rect_wing, cruise):
    """One more refinement level beyond the finest level of the report."""
    n_span = level_counts(_WING_BASE["n_panels"], 1.5, 4)[-1]
    n_chord = level_counts(_WING_BASE["n_chord"], 1.5, 4)[-1]
    settings = vt.SolverSettings(n_panels=n_span, n_chord=n_chord, solver_type="panel3d")
    return vt.analyze(rect_wing, condition=cruise, settings=settings)


@pytest.fixture(scope="module")
def sphere_cpmax():
    """Maximum Cp and lift of a unit sphere at three mesh levels."""
    cpmax: list[float] = []
    cl: list[float] = []
    n_lats: list[int] = []
    for n_lat, n_lon in ((8, 16), (12, 24), (18, 36)):
        mesh = _sphere_mesh(n_lat, n_lon)
        res = PanelSolver(mesh).solve(vt.FlightCondition(V_inf=20.0, alpha=0.0))
        cpmax.append(float(np.max(res.cp)))
        cl.append(float(res.CL))
        n_lats.append(n_lat)
    return {"cpmax": cpmax, "cl": cl, "n_lat": n_lats}


def test_level_counts_scale_both_directions():
    """Level counts multiply the base count by the ratio at each level."""
    assert level_counts(6, 1.5, 3) == [6, 9, 13]
    assert level_counts(8, 1.5, 3) == [8, 12, 18]
    counts = level_counts(1, 1.5, 4)
    assert all(n >= 2 for n in counts)
    assert all(b > a for a, b in zip(counts[:-1], counts[1:]))


def test_known_order_two_recovered():
    """A constructed order-2 sequence recovers order 2 within 1e-6."""
    levels = [4, 6, 9]  # ratio 1.5, coarse to fine
    values = [float(1.0 + 0.5 / n**2) for n in levels]
    conv = order_and_extrapolation(values, levels, quantity="CL")
    assert conv.observed_order == pytest.approx(2.0, abs=1e-6)
    assert conv.f_extrapolated == pytest.approx(1.0, abs=1e-9)
    assert conv.in_asymptotic_range is True
    # Base mesh estimate: safety factor 1.25 times the distance to the
    # Richardson value.
    assert conv.base_abs_error == pytest.approx(1.25 * abs(values[0] - 1.0), rel=1e-6)


def test_order_and_extrapolation_calls_the_layer_one_error_bar():
    """The pure function gives the classification of coefficient_error_bar."""
    cases = (
        ([1.0, 1.1, 1.3], [4, 6, 9]),       # divergent
        ([1.0, 1.2, 1.1], [4, 6, 9]),       # oscillatory
        ([1.0, 1.02, 1.0201], [5, 8, 13]),  # monotonic, high order
        ([2.0, 2.0, 2.0], [4, 6, 9]),       # round-off
    )
    for values, levels in cases:
        conv = order_and_extrapolation(values, levels, quantity="CL")
        bar = coefficient_error_bar("CL", values[::-1], levels[::-1])
        layer = bar.layers["numerical"]
        assert conv.state == layer.state
        assert conv.abs_error == layer.half_width
        assert conv.observed_order == layer.observed_order
        assert conv.f_extrapolated == layer.f_extrapolated


def test_sphere_cp_max_order(sphere_cpmax):
    """Sphere Cp max error falls with refinement at order 1 to 2.5."""
    cpmax = sphere_cpmax["cpmax"]
    errors = [abs(v - 1.0) for v in cpmax]
    assert errors[0] > errors[1] > errors[2]
    conv = order_and_extrapolation(cpmax, sphere_cpmax["n_lat"], quantity="Cpmax")
    assert conv.observed_order is not None
    assert 1.0 <= conv.observed_order <= 2.5
    # Exact answers: zero lift and Cp max of 1 at the stagnation point.
    assert all(abs(v) < 1e-9 for v in sphere_cpmax["cl"])


def test_reference_sphere_mesh_is_closed_and_outward():
    """The reference sphere mesh used by the study is watertight with outward panels."""
    stats = check_mesh(_sphere_mesh(8, 16))
    assert stats["watertight"] is True
    assert stats["outward"] is True


def test_rectangular_wing_finite_order(wing_report):
    """The rectangular wing converges with a finite observed order."""
    assert isinstance(wing_report, ConvergenceReport)
    assert wing_report.levels == 3
    assert len(wing_report.n_span_levels) == 3
    assert len(wing_report.n_chord_levels) == 3
    cl = wing_report.quantities["CL"]
    assert len(cl.values) == 3
    assert cl.observed_order is not None
    assert np.isfinite(cl.observed_order)
    assert cl.in_asymptotic_range is True
    assert cl.f_extrapolated is not None
    assert np.isfinite(cl.f_extrapolated)
    assert cl.relative_error is not None
    assert cl.relative_error >= 0.0
    data = wing_report.to_dict()
    assert set(data["quantities"]) == {"CL", "CDi", "Cm"}
    json.dumps(data)


def test_error_estimate_is_for_the_base_mesh(wing_report):
    """The reported estimate belongs to the base mesh (the mesh of the user).

    The base mesh is the coarsest level, so its estimate is larger than the
    bar of the finest level, and the trust note states the base value.
    """
    cl = wing_report.quantities["CL"]
    assert cl.state == "monotonic"
    assert cl.base_value == cl.values[0]
    assert cl.base_abs_error == pytest.approx(1.25 * abs(cl.f_extrapolated - cl.values[0]), rel=1e-12)
    assert cl.base_abs_error > cl.abs_error
    assert f"{cl.values[0]:.6g}" in cl.note
    assert "given mesh" in cl.note


def test_cost_factor_uses_cubic_scaling(wing_report):
    """The cost factor follows the N^3 work of the dense LU factorization."""
    n = np.asarray(wing_report.total_panels, dtype=float)
    assert wing_report.cost_factor == pytest.approx(float(np.sum((n / n[0]) ** 3)), rel=1e-12)


def test_error_covers_next_refinement(wing_report, wing_next):
    """The error estimates cover the change of one more refinement level.

    The estimate of the base mesh (the value that the trust notes give to
    the user) covers the distance from the base mesh to one more level
    for CL, CDi and Cm. The bar of the finest level covers the change of
    one more level for CL and CDi. For Cm of this wing the change from
    the finest level to one more level is larger than the finest-level
    bar: Cm turns back at finer meshes, which three levels cannot show, so
    the finest-level bar of Cm is not a bound here.
    """
    totals = wing_next.totals
    finer = {"CL": totals.CL, "CDi": totals.CDi, "Cm": totals.Cm}
    for name in ("CL", "CDi", "Cm"):
        conv = wing_report.quantities[name]
        assert conv.base_abs_error is not None
        assert abs(finer[name] - conv.values[0]) <= conv.base_abs_error, name
    for name in ("CL", "CDi"):
        conv = wing_report.quantities[name]
        assert conv.abs_error is not None
        change = abs(finer[name] - conv.values[-1])
        assert change <= conv.abs_error, name


def test_out_of_range_order_is_flagged():
    """A divergent sequence gives no estimate and raises the range flag."""
    conv = order_and_extrapolation([1.0, 1.1, 1.3], [4, 6, 9], quantity="CL")
    assert conv.observed_order is None
    assert conv.f_extrapolated is None
    assert conv.in_asymptotic_range is False
    assert conv.state == "divergent"
    assert conv.base_abs_error is None


def test_trust_score_gets_mesh_estimate(rect_wing, cruise):
    """The solver option attaches the mesh error estimate to the result."""
    on = vt.SolverSettings(n_panels=4, n_chord=4, solver_type="panel3d", panel_estimate_mesh_error=True)
    off = vt.SolverSettings(n_panels=4, n_chord=4, solver_type="panel3d")
    res_on = vt.analyze(rect_wing, condition=cruise, settings=on)
    res_off = vt.analyze(rect_wing, condition=cruise, settings=off)
    assert res_on.totals.CL == pytest.approx(res_off.totals.CL)
    assert "mesh_convergence" in res_on.details
    assert "mesh_convergence" not in res_off.details
    assert res_on.totals.trust is not None
    assert any("mesh convergence" in w.lower() for w in res_on.totals.trust.warnings)
    assert all("mesh convergence" not in w.lower() for w in res_off.totals.trust.warnings)
    base_cl = res_on.details["mesh_convergence"]["quantities"]["CL"]["base_value"]
    assert base_cl == pytest.approx(res_on.totals.CL, rel=1e-12)


def test_invalid_inputs(rect_wing, cruise):
    """Bad level counts, ratios and quantity names are refused."""
    settings = vt.SolverSettings(**_WING_BASE)
    with pytest.raises(ValueError):
        panel_mesh_convergence(rect_wing, cruise, settings, levels=2)
    with pytest.raises(ValueError):
        panel_mesh_convergence(rect_wing, cruise, settings, ratio=1.0)
    with pytest.raises(ValueError):
        panel_mesh_convergence(rect_wing, cruise, settings, quantities=("CX",))
    with pytest.raises(ValueError):
        order_and_extrapolation([1.0, 1.1, 1.2], [6, 6, 9])
