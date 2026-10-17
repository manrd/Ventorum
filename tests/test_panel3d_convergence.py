# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for mesh convergence and Richardson error estimates of Ventorum-3DP.

The mesh-error estimate refines the chordwise and the spanwise panel
count separately and adds the two directional estimates (separable error
model, see :mod:`ventorum.panel3d.convergence`). The known answers are the
exact potential flow about a sphere (Anderson, Fundamentals of
Aerodynamics, section 4.5) and constructed sequences with a known order.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.error_bars import coefficient_error_bar
from ventorum.panel3d.convergence import (
    ConvergenceReport,
    MeshErrorEstimate,
    directional_estimate,
    level_counts,
    order_and_extrapolation,
    panel_mesh_convergence,
)
from ventorum.panel3d.mesh import PanelMesh, check_mesh
from ventorum.panel3d.solver import PanelSolver
from ventorum.reference.analytic3d import sphere_cp, sphere_quad_mesh


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


# Base mesh of the wing study: 12 spanwise and 16 chordwise panels. The
# chordwise refinement (16, 24, 36 panels) converges with an observed
# order near 1 in CL, CDi and Cm. The spanwise CL of this wing converges
# slowly: from 8 spanwise panels the spanwise CL sequence (8, 12, 18) is
# not yet monotonic (the change does not fall), so it gives no estimate;
# from 12 spanwise panels it is monotonic with an order below 0.5 (outside
# the asymptotic range, safety factor 3).
_WING_BASE = {"n_panels": 12, "n_chord": 16}


def _wing_settings(n_span: int, n_chord: int) -> vt.SolverSettings:
    return vt.SolverSettings(n_panels=n_span, n_chord=n_chord, solver_type="panel3d")


@pytest.fixture(scope="module")
def wing_report(rect_wing, cruise) -> ConvergenceReport:
    """Directional convergence report of the rectangular wing from the base mesh up."""
    settings = vt.SolverSettings(**_WING_BASE)
    return panel_mesh_convergence(
        rect_wing, cruise, settings, levels=3, ratio=1.5,
    )


def _next_counts() -> tuple[int, int]:
    """Spanwise and chordwise counts one level beyond the finest level of the report."""
    n_span = level_counts(_WING_BASE["n_panels"], 1.5, 4)[-1]
    n_chord = level_counts(_WING_BASE["n_chord"], 1.5, 4)[-1]
    return n_span, n_chord


@pytest.fixture(scope="module")
def wing_next(rect_wing, cruise):
    """One more refinement level in each direction (at the base count of the other direction)."""
    n_span, n_chord = _next_counts()
    chord = vt.analyze(rect_wing, condition=cruise, settings=_wing_settings(_WING_BASE["n_panels"], n_chord))
    span = vt.analyze(rect_wing, condition=cruise, settings=_wing_settings(n_span, _WING_BASE["n_chord"]))
    return {"chordwise": chord, "spanwise": span}


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


def _totals(result) -> dict[str, float]:
    t = result.totals
    return {"CL": t.CL, "CDi": t.CDi, "Cm": t.Cm}


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


def test_separable_errors_of_opposite_sign_are_recovered():
    """A separable function with opposite directional errors gives its limit and a bound.

    f(ns, nc) = 1 + 0.5 / ns**2 - 0.3 / nc: the spanwise error is positive
    (order 2), the chordwise error negative (order 1), as for the lift of a
    thick wing. A refinement in both directions at once mixes the two
    orders; the directional estimate recovers each order, the limit 1 and
    an error estimate of the base mesh that covers its true error.
    """
    def f(ns, nc):
        return 1.0 + 0.5 / ns**2 - 0.3 / nc

    spans = [4, 6, 9]
    chords = [8, 12, 18]
    est = directional_estimate(
        [f(spans[0], nc) for nc in chords], chords, [f(ns, chords[0]) for ns in spans], spans, quantity="CL",
    )
    assert isinstance(est, MeshErrorEstimate)
    assert est.chordwise.observed_order == pytest.approx(1.0, abs=1e-6)
    assert est.spanwise.observed_order == pytest.approx(2.0, abs=1e-6)
    assert est.f_extrapolated == pytest.approx(1.0, abs=1e-9)
    assert est.state == "monotonic"
    assert est.in_asymptotic_range is True
    f0 = f(spans[0], chords[0])
    assert est.base_value == f0
    true_error = abs(f0 - 1.0)
    assert est.base_abs_error >= true_error
    # Sum of the two directional estimates (safety factor 1.25 each).
    expected = 1.25 * (0.3 / chords[0]) + 1.25 * (0.5 / spans[0] ** 2)
    assert est.base_abs_error == pytest.approx(expected, rel=1e-6)
    assert est.base_abs_error == pytest.approx(
        est.chordwise.base_abs_error + est.spanwise.base_abs_error, rel=1e-12
    )
    # A refinement in both directions at once does not see the opposite signs.
    both = order_and_extrapolation([f(ns, nc) for ns, nc in zip(spans, chords)], spans, quantity="CL")
    assert both.f_extrapolated is None or abs(both.f_extrapolated - 1.0) > 1e-3
    json.dumps(est.to_dict())


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


@pytest.mark.slow
def test_sphere_directional_estimate_against_the_exact_solution():
    """The directional estimate on a sphere covers the exact error and its value is near the exact one.

    The two mesh directions of the sphere are the latitude and the
    longitude counts. From the base mesh (12 latitude, 24 longitude
    panels) each direction is refined by 2 (the planes x = 0 stay mesh
    lines). Exact values from the exact surface pressure of
    ``ventorum.reference.analytic3d.sphere_cp`` (Anderson, section 4.5):
    Cp max 1 at the stagnation point and the axial force of the upstream
    half, ``C_front = -1/8`` (over q pi a^2, by Gauss-Legendre quadrature
    of the exact Cp). Each direction converges at order
    near 2, the Richardson value of the separable model is closer to the
    exact value than the base value by more than ten times, and the error
    estimate of the base mesh covers the exact error without being more
    than twice as large.
    """
    cond = vt.FlightCondition(V_inf=1.0, alpha=0.0)
    lats = level_counts(12, 2.0, 3)
    lons = level_counts(24, 2.0, 3)

    def solve(n_lat, n_lon):
        solver = PanelSolver(_sphere_mesh(n_lat, n_lon))
        res = solver.solve(cond)
        front = solver.centroids[:, 0] < 0.0
        fx = -np.sum(res.cp[front] * solver.areas[front] * solver.normals[front, 0])
        return {"C_front": float(fx / np.pi), "Cp_max": float(np.max(res.cp))}

    base = solve(lats[0], lons[0])
    lat_seq = [base] + [solve(n, lons[0]) for n in lats[1:]]
    lon_seq = [base] + [solve(lats[0], n) for n in lons[1:]]
    # Exact values: the flow is axisymmetric about x; psi is the angle from +x.
    xg, wg = np.polynomial.legendre.leggauss(64)
    psi = 0.25 * np.pi * (xg + 3.0)  # (pi/2, pi): the upstream half
    w_psi = 0.25 * np.pi * wg
    ring = np.column_stack([np.cos(psi), np.sin(psi), np.zeros_like(psi)])
    cp_ring = sphere_cp(ring, 1.0, 1.0)
    c_front = -float(np.sum(cp_ring * np.cos(psi) * 2.0 * np.pi * np.sin(psi) * w_psi)) / np.pi
    assert c_front == pytest.approx(-0.125, abs=1e-12)
    cp_stag = float(sphere_cp(np.array([[-1.0, 0.0, 0.0]]), 1.0, 1.0)[0])
    exact = {"C_front": c_front, "Cp_max": cp_stag}
    for name, value in exact.items():
        est = directional_estimate(
            [r[name] for r in lat_seq], lats, [r[name] for r in lon_seq], lons, quantity=name,
        )
        assert est.state == "monotonic", name
        assert est.in_asymptotic_range, name
        for conv in (est.chordwise, est.spanwise):
            assert 1.5 <= conv.observed_order <= 2.5, name
        base_error = abs(est.base_value - value)
        assert abs(est.f_extrapolated - value) < 0.1 * base_error, name
        assert base_error <= est.base_abs_error <= 2.0 * base_error, name


def test_directional_estimate_rejects_bad_input():
    """Sequences that do not share the base mesh, or have fewer than three levels, are refused."""
    with pytest.raises(ValueError):
        directional_estimate([1.0, 1.1, 1.15], [4, 6, 9], [1.01, 0.99, 0.98], [4, 6, 9])
    with pytest.raises(ValueError):
        directional_estimate([1.0, 1.1], [4, 6], [1.0, 0.99, 0.98], [4, 6, 9])
    with pytest.raises(ValueError):
        directional_estimate([1.0, 1.1, 1.15], [4, 6, 6], [1.0, 0.99, 0.98], [4, 6, 9])


def test_divergent_direction_gives_no_estimate():
    """A direction without an estimate leaves the combined estimate empty, with a note."""
    est = directional_estimate([1.0, 1.1, 1.3], [4, 6, 9], [1.0, 0.99, 0.985], [4, 6, 9], quantity="CL")
    assert est.chordwise.state == "divergent"
    assert est.state == "no_estimate"
    assert est.base_abs_error is None
    assert est.f_extrapolated is None
    assert "chordwise refinement gives none" in est.note


def test_reference_sphere_mesh_is_closed_and_outward():
    """The reference sphere mesh used by the study is watertight with outward panels."""
    stats = check_mesh(_sphere_mesh(8, 16))
    assert stats["watertight"] is True
    assert stats["outward"] is True


def test_rectangular_wing_directional_orders(wing_report):
    """The rectangular wing converges in each direction with a finite observed order."""
    assert isinstance(wing_report, ConvergenceReport)
    assert wing_report.levels == 3
    assert wing_report.n_chord_levels == [16, 24, 36]
    assert wing_report.n_span_levels == [12, 18, 27]
    assert wing_report.meshes == [(12, 16), (12, 24), (12, 36), (18, 16), (27, 16)]
    assert len(wing_report.results) == len(wing_report.total_panels) == 5
    for name in ("CL", "CDi", "Cm"):
        est = wing_report.quantities[name]
        assert est.state == "monotonic", name
        for conv in (est.chordwise, est.spanwise):
            assert len(conv.values) == 3
            assert conv.state == "monotonic", name
            assert conv.observed_order is not None and np.isfinite(conv.observed_order)
            assert conv.observed_order > 0.0
            assert conv.f_extrapolated is not None and np.isfinite(conv.f_extrapolated)
            assert conv.relative_error is not None and conv.relative_error >= 0.0
        # The chordwise refinement is in the asymptotic range (order near 1).
        assert est.chordwise.in_asymptotic_range is True, name
        assert 0.5 <= est.chordwise.observed_order <= 1.5, name
        assert est.f_extrapolated is not None and np.isfinite(est.f_extrapolated)
    # Thick wing: the chordwise refinement raises CL, the spanwise refinement lowers it.
    cl = wing_report.quantities["CL"]
    assert cl.chordwise.values[-1] > cl.base_value > cl.spanwise.values[-1]
    data = wing_report.to_dict()
    assert data["method"] == "directional"
    assert set(data["quantities"]) == {"CL", "CDi", "Cm"}
    assert set(data["quantities"]["CL"]["chordwise"]) >= {"values", "observed_order", "base_abs_error"}
    json.dumps(data)
    assert "chordwise" in wing_report.summary()


def test_error_estimate_is_for_the_base_mesh(wing_report):
    """The reported estimate belongs to the base mesh (the mesh of the user).

    The two directions start at the base mesh. The estimate is the sum of
    the two directional estimates, each the safety factor times the
    distance from the base value to the Richardson value of its
    direction, and the Richardson value is that of the separable model.
    The base mesh is the coarsest level of both directions, so its
    estimate is larger than the bar of the finest level of each direction.
    """
    for name in ("CL", "CDi", "Cm"):
        est = wing_report.quantities[name]
        f0 = est.base_value
        assert f0 == est.chordwise.values[0] == est.spanwise.values[0]
        assert f0 == pytest.approx(_totals(wing_report.results[0])[name], rel=1e-12)
        chord, span = est.chordwise, est.spanwise
        assert chord.base_abs_error == pytest.approx(1.25 * abs(chord.f_extrapolated - f0), rel=1e-12)
        fs_span = 1.25 if span.in_asymptotic_range else 3.0
        assert span.base_abs_error == pytest.approx(fs_span * abs(span.f_extrapolated - f0), rel=1e-12)
        assert est.base_abs_error == pytest.approx(chord.base_abs_error + span.base_abs_error, rel=1e-12)
        assert est.f_extrapolated == pytest.approx(chord.f_extrapolated + span.f_extrapolated - f0, rel=1e-12)
        assert est.base_abs_error > chord.abs_error
        assert est.base_abs_error > span.abs_error
        assert f"{f0:.6g}" in est.note
        assert "given mesh" in est.note


def test_cost_factor_uses_cubic_scaling(wing_report):
    """The cost factor follows the N^3 work of the dense LU factorization over the five solved meshes.

    Each direction multiplies the panel count by about the ratio per
    level, so the cost is about 1 + 2 (1.5**3 + 1.5**6) (about 30) base
    solves, far below the cost of a refinement in both directions at once
    (about 1 + 1.5**6 + 1.5**12).
    """
    n = np.asarray(wing_report.total_panels, dtype=float)
    assert wing_report.cost_factor == pytest.approx(float(np.sum((n / n[0]) ** 3)), rel=1e-12)
    assert wing_report.cost_factor == pytest.approx(1.0 + 2.0 * (1.5**3 + 1.5**6), rel=0.1)
    assert wing_report.cost_factor < 0.25 * (1.0 + 1.5**6 + 1.5**12)


def test_error_covers_next_refinement(wing_report, wing_next):
    """The error estimates cover the change of one more refinement level.

    The bar of the finest level of each direction covers the change of
    one more level in that direction, for CL, CDi and Cm. The estimate of
    the base mesh (the value that the trust notes give to the user)
    covers the sum of the distances from the base mesh to the next level
    of each direction, which bounds the distance to the mesh refined in
    both directions under the separable model.
    """
    nxt_chord = _totals(wing_next["chordwise"])
    nxt_span = _totals(wing_next["spanwise"])
    for name in ("CL", "CDi", "Cm"):
        est = wing_report.quantities[name]
        f0 = est.base_value
        assert abs(nxt_chord[name] - est.chordwise.values[-1]) <= est.chordwise.abs_error, name
        assert abs(nxt_span[name] - est.spanwise.values[-1]) <= est.spanwise.abs_error, name
        assert abs(nxt_chord[name] - f0) + abs(nxt_span[name] - f0) <= est.base_abs_error, name


@pytest.mark.slow
def test_error_covers_the_mesh_refined_in_both_directions(rect_wing, cruise, wing_report):
    """The estimate of the base mesh covers the change to the mesh refined once more in both directions.

    The mesh has one level more than the finest level of each direction
    in both counts at once (40 spanwise, 54 chordwise panels).
    """
    n_span, n_chord = _next_counts()
    res = vt.analyze(rect_wing, condition=cruise, settings=_wing_settings(n_span, n_chord))
    finer = _totals(res)
    for name in ("CL", "CDi", "Cm"):
        est = wing_report.quantities[name]
        assert abs(finer[name] - est.base_value) <= est.base_abs_error, name


def test_out_of_range_order_is_flagged():
    """A divergent sequence gives no estimate and raises the range flag."""
    conv = order_and_extrapolation([1.0, 1.1, 1.3], [4, 6, 9], quantity="CL")
    assert conv.observed_order is None
    assert conv.f_extrapolated is None
    assert conv.in_asymptotic_range is False
    assert conv.state == "divergent"
    assert conv.base_abs_error is None


def test_trust_score_gets_mesh_estimate(rect_wing, cruise):
    """The solver option attaches the directional mesh error estimate to the result."""
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
    report = res_on.details["mesh_convergence"]
    assert report["method"] == "directional"
    assert report["meshes"] == [[4, 4], [4, 6], [4, 9], [6, 4], [9, 4]]
    cl = report["quantities"]["CL"]
    assert cl["base_value"] == pytest.approx(res_on.totals.CL, rel=1e-12)
    assert cl["chordwise"]["values"][0] == cl["spanwise"]["values"][0] == cl["base_value"]


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
