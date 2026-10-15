# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for mesh convergence and Richardson error estimates of Ventorum-3DP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import ventorum as vt
from ventorum.panel3d.convergence import (
    ConvergenceReport,
    build_sphere_mesh,
    level_counts,
    order_and_extrapolation,
    panel_mesh_convergence,
)
from ventorum.panel3d.mesh import check_mesh
from ventorum.panel3d.solver import PanelSolver


@dataclass
class _FlagSettings(vt.SolverSettings):
    """Solver settings with the mesh-error option set by the user."""

    estimate_mesh_error: bool = False


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


@pytest.fixture(scope="module")
def wing_report(rect_wing, cruise) -> ConvergenceReport:
    """Convergence report of the rectangular wing from the base mesh up."""
    settings = vt.SolverSettings(n_panels=6, n_chord=6)
    return panel_mesh_convergence(
        rect_wing, cruise, settings, levels=3, ratio=1.5,
    )


@pytest.fixture(scope="module")
def wing_next(rect_wing, cruise):
    """One more refinement level beyond the finest level of the report."""
    settings = vt.SolverSettings(n_panels=14, n_chord=14, solver_type="panel3d")
    return vt.analyze(rect_wing, condition=cruise, settings=settings)


@pytest.fixture(scope="module")
def sphere_cpmax():
    """Maximum Cp and lift of a unit sphere at three mesh levels."""
    cpmax: list[float] = []
    cl: list[float] = []
    npan: list[int] = []
    for n_lat, n_lon in ((8, 16), (12, 24), (18, 36)):
        mesh = build_sphere_mesh(1.0, n_lat, n_lon)
        res = PanelSolver(mesh).solve(vt.FlightCondition(V_inf=20.0, alpha=0.0))
        cpmax.append(float(np.max(res.cp)))
        cl.append(float(res.CL))
        npan.append(int(len(mesh.panels)))
    return {"cpmax": cpmax, "cl": cl, "npan": npan}


def test_level_counts_scale_both_directions():
    """Level counts multiply the base count by the ratio at each level."""
    assert level_counts(6, 1.5, 3) == [6, 9, 13]
    assert level_counts(8, 1.5, 3) == [8, 12, 18]
    counts = level_counts(1, 1.5, 4)
    assert all(n >= 2 for n in counts)
    assert all(b > a for a, b in zip(counts[:-1], counts[1:]))


def test_known_order_two_recovered():
    """A constructed order-2 sequence recovers order 2 within 1e-6."""
    r = 1.5
    h = np.array([1.0, 1.0 / r, 1.0 / r**2])  # coarse to fine
    values = [float(1.0 + 0.5 * hh**2) for hh in h]
    conv = order_and_extrapolation(values, r, r, quantity="CL")
    assert conv.observed_order == pytest.approx(2.0, abs=1e-6)
    assert conv.f_extrapolated == pytest.approx(1.0, abs=1e-9)
    assert conv.in_asymptotic_range is True


def test_sphere_cp_max_order(sphere_cpmax):
    """Sphere Cp max error falls with refinement at order 1 to 2.5."""
    cpmax = sphere_cpmax["cpmax"]
    npan = sphere_cpmax["npan"]
    errors = [abs(v - 1.0) for v in cpmax]
    assert errors[0] > errors[1] > errors[2]
    h = [1.0 / np.sqrt(n) for n in npan]
    conv = order_and_extrapolation(cpmax, h[1] / h[2], h[0] / h[1], quantity="Cpmax")
    assert conv.observed_order is not None
    assert 1.0 <= conv.observed_order <= 2.5
    # Exact answers: zero lift and Cp max of 1 at the stagnation point.
    assert all(abs(v) < 1e-9 for v in sphere_cpmax["cl"])


def test_sphere_mesh_is_closed_and_outward():
    """The verification sphere mesh is watertight with outward panels."""
    stats = check_mesh(build_sphere_mesh(1.0, 8, 16))
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
    import json

    json.dumps(data)


def test_error_covers_next_refinement(wing_report, wing_next):
    """The error bar of each quantity covers the change of one more level."""
    totals = wing_next.totals
    finer = {"CL": totals.CL, "CDi": totals.CDi, "Cm": totals.Cm}
    for name in ("CL", "CDi", "Cm"):
        conv = wing_report.quantities[name]
        assert conv.abs_error is not None
        change = abs(finer[name] - conv.values[-1])
        assert change <= conv.abs_error


def test_out_of_range_order_is_flagged():
    """A divergent sequence gives no estimate and raises the range flag."""
    conv = order_and_extrapolation([1.0, 1.1, 1.3], 1.5, 1.5, quantity="CL")
    assert conv.observed_order is None
    assert conv.f_extrapolated is None
    assert conv.in_asymptotic_range is False
    assert conv.state == "divergent"


def test_trust_score_gets_mesh_estimate(rect_wing, cruise):
    """The solver option attaches the mesh error estimate to the result."""
    on = _FlagSettings(n_panels=4, n_chord=4, solver_type="panel3d", estimate_mesh_error=True)
    off = vt.SolverSettings(n_panels=4, n_chord=4, solver_type="panel3d")
    res_on = vt.analyze(rect_wing, condition=cruise, settings=on)
    res_off = vt.analyze(rect_wing, condition=cruise, settings=off)
    assert res_on.totals.CL == pytest.approx(res_off.totals.CL)
    assert "mesh_convergence" in res_on.details
    assert "mesh_convergence" not in res_off.details
    assert res_on.totals.trust is not None
    assert any("mesh convergence" in w.lower() for w in res_on.totals.trust.warnings)
    assert all("mesh convergence" not in w.lower() for w in res_off.totals.trust.warnings)


def test_invalid_inputs(rect_wing, cruise):
    """Bad level counts, ratios and quantity names are refused."""
    settings = vt.SolverSettings(n_panels=6, n_chord=6)
    with pytest.raises(ValueError):
        panel_mesh_convergence(rect_wing, cruise, settings, levels=2)
    with pytest.raises(ValueError):
        panel_mesh_convergence(rect_wing, cruise, settings, ratio=1.0)
    with pytest.raises(ValueError):
        panel_mesh_convergence(rect_wing, cruise, settings, quantities=("CX",))
