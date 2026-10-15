# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the 3D panel method solver family integration.

Published sources
-----------------
* Morino, L. and Kuo, C. C., 1974, Subsonic potential aerodynamics for
  complex configurations: A general theory, AIAA Journal, Vol. 12, No. 2,
  pp. 191-197.
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, section 9.4 and chapter 12.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from ventorum.airfoil import airfoil
from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    SolverResult,
    SolverSettings,
    WingSection,
    aircraft_from_json,
    aircraft_to_json,
)
from ventorum.solvers.factory import make_solver, resolve_solver_type
from ventorum.solvers.panel3d import PanelMethodSolver


def _naca0012_coords(n_points: int = 101) -> np.ndarray:
    """Return unit-chord Selig coordinates of a NACA 0012 airfoil."""
    return airfoil("0012", n_points=n_points).coords


def _simple_wing(
    shape: str | np.ndarray | None = "0012",
    semi_span: float = 4.0,
    chord: float = 1.0,
    sweep_le: float = 0.0,
) -> Aircraft:
    """Return a rectangular wing aircraft with the given section shape."""
    surf = LiftingSurface(
        name="Wing",
        semi_span=semi_span,
        sections=[
            WingSection(y_frac=0.0, chord=chord, shape=shape),
            WingSection(y_frac=1.0, chord=chord, shape=shape),
        ],
        sweep_le=sweep_le,
        is_symmetric=True,
    )
    ac = Aircraft(name="SimpleWing", surfaces=[surf])
    ac.compute_reference_values()
    return ac


def test_factory_resolves_and_creates_panel3d():
    """Factory resolves 'panel3d' and builds PanelMethodSolver."""
    canonical = resolve_solver_type("panel3d")
    assert canonical == "panel3d"
    solver = make_solver("panel3d")
    assert isinstance(solver, PanelMethodSolver)
    assert solver.name == "panel3d"


def test_wing_with_shapes_solve():
    """Factory builds and solves a wing with airfoil section shapes."""
    solver = make_solver("panel3d")
    ac = _simple_wing(shape="2412", semi_span=3.0, chord=1.0)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(3.0))
    st = SolverSettings(n_panels=8, n_chord=8)

    res = solver.solve(ac, cond, st)
    assert isinstance(res, SolverResult)
    assert res.converged
    assert res.solver_type == "panel3d"
    assert res.totals.CL > 0.1
    assert res.totals.CDi >= 0.0
    assert len(res.spanwise) == 1

    sw = res.spanwise[0]
    assert sw.surface_name == "Wing"
    assert len(sw.Cl) > 0
    assert len(sw.Cd_i) > 0
    assert len(sw.Cm_section) > 0

    assert "cp" in res.details
    assert isinstance(res.details["cp"], np.ndarray)
    assert len(res.details["cp"]) > 0


def test_default_shape_warning_when_none():
    """Section with shape=None defaults to NACA 0012 with a warning note."""
    solver = make_solver("panel3d")
    ac = _simple_wing(shape=None, semi_span=2.0, chord=1.0)
    cond = FlightCondition(V_inf=25.0, alpha=np.radians(2.0))
    st = SolverSettings(n_panels=6, n_chord=6)

    with pytest.warns(RuntimeWarning, match="default NACA 0012"):
        res = solver.solve(ac, cond, st)

    assert any("default NACA 0012" in n for n in res.details.get("notes", []))
    assert res.converged


def test_wing_plus_tail_solve():
    """Factory solves an aircraft configuration with wing and tail surfaces."""
    solver = make_solver("panel3d")
    wing = LiftingSurface(
        name="Wing",
        semi_span=3.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, shape="0012"),
            WingSection(y_frac=1.0, chord=0.8, shape="0012"),
        ],
        is_symmetric=True,
    )
    tail = LiftingSurface(
        name="Tail",
        semi_span=1.2,
        sections=[
            WingSection(y_frac=0.0, chord=0.5, shape="0012"),
            WingSection(y_frac=1.0, chord=0.4, shape="0012"),
        ],
        position=np.array([2.5, 0.0, 0.2]),
        is_symmetric=True,
    )
    ac = Aircraft(name="WingTail", surfaces=[wing, tail])
    ac.compute_reference_values()

    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = SolverSettings(n_panels=6, n_chord=6)

    res = solver.solve(ac, cond, st)
    assert res.converged
    assert len(res.spanwise) == 2
    assert res.spanwise[0].surface_name == "Wing"
    assert res.spanwise[1].surface_name == "Tail"


def test_spanwise_strip_lift_sums_to_total_cl():
    """Spanwise strip lift sum matches total CL within 1e-3."""
    solver = make_solver("panel3d")
    ac = _simple_wing(shape="0012", semi_span=4.0, chord=1.0)
    cond = FlightCondition(V_inf=35.0, alpha=np.radians(4.0))
    st = SolverSettings(n_panels=8, n_chord=8)

    res = solver.solve(ac, cond, st)
    q_inf = 0.5 * cond.rho * cond.V_inf**2
    s_ref = ac.S_ref

    strip_lift_sum = 0.0
    for sw in res.spanwise:
        y_pts = sw.y
        n_s = len(y_pts)
        if n_s > 1:
            dy = np.diff(y_pts)
            widths = np.empty(n_s)
            widths[0] = dy[0]
            widths[-1] = dy[-1]
            widths[1:-1] = 0.5 * (dy[:-1] + dy[1:])
        else:
            widths = np.array([ac.b_ref])
        strip_lift_sum += float(np.sum(sw.local_lift * np.abs(widths)))

    cl_from_strips = strip_lift_sum / (q_inf * s_ref)
    assert abs(cl_from_strips - res.totals.CL) < 1.0e-3


def test_ground_effect_case():
    """Case in ground effect solves with image method and increases lift."""
    solver = make_solver("panel3d")
    ac = _simple_wing(shape="0012", semi_span=3.0, chord=1.0)
    cond_free = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    cond_ge = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=1.0)
    st = SolverSettings(n_panels=6, n_chord=6)

    res_free = solver.solve(ac, cond_free, st)
    res_ge = solver.solve(ac, cond_ge, st)

    assert res_ge.converged
    assert res_ge.totals.CL > res_free.totals.CL
    assert res_ge.details["h_min"] is not None


def test_five_angle_sweep_one_factorization():
    """Five-angle sweep performs and counts exactly one factorization in free air."""
    solver = make_solver("panel3d")
    ac = _simple_wing(shape="0012", semi_span=3.0, chord=1.0)
    cond = FlightCondition(V_inf=25.0, alpha=0.0)
    st = SolverSettings(n_panels=6, n_chord=6)

    angles = np.radians([-2.0, 0.0, 2.0, 4.0, 6.0])
    results = solver.solve_sweep(ac, cond, st, angles)

    assert len(results) == 5
    cls = [r.totals.CL for r in results]
    # Lift increases monotonically with alpha
    for i in range(4):
        assert cls[i + 1] > cls[i]

    for r in results:
        disp = r.details.get("dispatch", {})
        assert disp.get("factorization_count") == 1
        assert disp.get("backend") == "numpy"
        assert disp.get("device") == "cpu"
        assert disp.get("dtype") == "float64"


def test_json_round_trip_with_and_without_shape():
    """JSON serialization round trip preserves shape and supports backward compatibility."""
    # 1. Round trip with shape string
    ac_with_shape = _simple_wing(shape="2412", semi_span=3.0, chord=1.0)
    json_str = aircraft_to_json(ac_with_shape)
    loaded = aircraft_from_json(json_str)
    assert loaded.surfaces[0].sections[0].shape == "2412"

    # 2. Round trip with shape coordinate array
    coords = _naca0012_coords(21)
    ac_with_coords = _simple_wing(shape=coords, semi_span=3.0, chord=1.0)
    json_coords = aircraft_to_json(ac_with_coords)
    loaded_coords = aircraft_from_json(json_coords)
    assert np.allclose(loaded_coords.surfaces[0].sections[0].shape, coords)

    # 3. Round trip without shape (shape=None)
    ac_no_shape = _simple_wing(shape=None, semi_span=3.0, chord=1.0)
    json_none = aircraft_to_json(ac_no_shape)
    loaded_none = aircraft_from_json(json_none)
    assert loaded_none.surfaces[0].sections[0].shape is None

    # 4. Backward compatibility: JSON string without 'shape' key
    raw_dict = json.loads(json_none)
    for s in raw_dict["surfaces"]:
        for sec in s["sections"]:
            sec.pop("shape", None)
    old_json = json.dumps(raw_dict)
    loaded_old = aircraft_from_json(old_json)
    assert loaded_old.surfaces[0].sections[0].shape is None


def test_trust_score_lower_for_skewed_mesh():
    """Trust score is present and is lower for a deliberately skewed mesh."""
    solver = make_solver("panel3d")
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = SolverSettings(n_panels=6, n_chord=6)

    # Straight regular wing
    ac_clean = _simple_wing(shape="0012", semi_span=4.0, chord=1.0, sweep_le=0.0)
    res_clean = solver.solve(ac_clean, cond, st)

    # Deliberately skewed wing: large sweep and skewed sections
    surf_skewed = LiftingSurface(
        name="SkewedWing",
        semi_span=2.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, shape="0012", x_le=0.0),
            WingSection(y_frac=1.0, chord=1.0, shape="0012", x_le=4.0),
        ],
        is_symmetric=True,
    )
    ac_skewed = Aircraft(name="SkewedAircraft", surfaces=[surf_skewed])
    ac_skewed.compute_reference_values()
    res_skewed = solver.solve(ac_skewed, cond, st)

    assert res_clean.totals.trust is not None
    assert res_skewed.totals.trust is not None
    assert res_skewed.totals.trust.score < res_clean.totals.trust.score
    assert res_skewed.totals.trust.factors.get("mesh_adequacy", 0.0) > 0.0
