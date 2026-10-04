# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Comprehensive tests for Ventorum meshing schemes, local refinement, and auto-criteria.
"""

from __future__ import annotations

import numpy as np
import pytest

import ventorum as vt
from ventorum.geometry.discretization import (
    cosine_spacing,
    half_cosine_spacing,
    root_cosine_spacing,
    uniform_spacing,
    power_spacing,
    determine_optimal_spacing,
    compute_surface_n_panels,
    get_spacing,
)
from ventorum.geometry.processing import discretize_surface, discretize_aircraft_surfaces
from ventorum.solvers.horseshoe import HorseshoeSolver
from ventorum.core.config import aircraft_to_json, aircraft_from_json


def test_spacing_mathematical_properties():
    """Verify bounds, monotonicity, and clustering gradients of each spacing scheme."""
    N = 20

    # 1. Full cosine: clustered at both ends
    edges_cos, mids_cos = cosine_spacing(N)
    assert len(edges_cos) == N + 1
    assert len(mids_cos) == N
    assert np.isclose(edges_cos[0], 0.0)
    assert np.isclose(edges_cos[-1], 1.0)
    assert np.all(np.diff(edges_cos) > 0)
    # Root dy and tip dy should be small and symmetric
    dy_cos = np.diff(edges_cos)
    np.testing.assert_allclose(dy_cos[0], dy_cos[-1], atol=1e-12)
    assert dy_cos[0] < dy_cos[N // 2]  # clustered at ends relative to middle

    # 2. Half-cosine: clustered at tip, coarse at root
    edges_hc, mids_hc = half_cosine_spacing(N)
    assert len(edges_hc) == N + 1
    assert np.isclose(edges_hc[0], 0.0)
    assert np.isclose(edges_hc[-1], 1.0)
    assert np.all(np.diff(edges_hc) > 0)
    dy_hc = np.diff(edges_hc)
    assert dy_hc[-1] < dy_hc[0]  # tip panel is significantly narrower than root panel

    # 3. Root-cosine: clustered at root, coarse at tip
    edges_rc, mids_rc = root_cosine_spacing(N)
    assert len(edges_rc) == N + 1
    assert np.isclose(edges_rc[0], 0.0)
    assert np.isclose(edges_rc[-1], 1.0)
    assert np.all(np.diff(edges_rc) > 0)
    dy_rc = np.diff(edges_rc)
    assert dy_rc[0] < dy_rc[-1]  # root panel is narrower than tip panel

    # 4. Uniform: constant panel width
    edges_u, mids_u = uniform_spacing(N)
    dy_u = np.diff(edges_u)
    np.testing.assert_allclose(dy_u, 1.0 / N, atol=1e-14)

    # 5. Power spacing: controlled clustering
    edges_p, mids_p = power_spacing(N, power=1.5, clustering="tip")
    assert np.isclose(edges_p[0], 0.0)
    assert np.isclose(edges_p[-1], 1.0)
    assert np.all(np.diff(edges_p) > 0)


def test_get_spacing_dispatch():
    """Verify get_spacing dispatches alias names correctly and raises for invalid names."""
    N = 10
    e_cos, _ = get_spacing("cosine", N)
    e_full, _ = get_spacing("full-cosine", N)
    np.testing.assert_allclose(e_cos, e_full)

    e_tip, _ = get_spacing("tip", N)
    e_hc, _ = get_spacing("half-cosine", N)
    np.testing.assert_allclose(e_tip, e_hc)

    e_root, _ = get_spacing("root", N)
    e_rc, _ = get_spacing("root-cosine", N)
    np.testing.assert_allclose(e_root, e_rc)

    with pytest.raises(ValueError, match="Unknown spacing method"):
        get_spacing("invalid_scheme", N)


def test_determine_optimal_spacing_criteria():
    """Verify geometry-based criteria select the correct spacing."""
    # 1. Planar wing (dihedral = 0, sweep = 0, symmetric) -> half-cosine (dGamma/dy = 0 at root)
    planar_wing = vt.LiftingSurface(
        name="PlanarWing",
        semi_span=5.0,
        dihedral=0.0,
        sweep_le=0.0,
        is_symmetric=True,
    )
    assert determine_optimal_spacing(planar_wing) == "half-cosine"

    # 2. Near-planar wing (dihedral = 2 deg <= 5 deg, sweep = 1.5 deg <= 15 deg) -> half-cosine
    near_planar = vt.LiftingSurface(
        name="AerosondeMainWing",
        semi_span=1.45,
        dihedral=np.radians(2.0),
        sweep_le=np.radians(1.5),
        is_symmetric=True,
    )
    assert determine_optimal_spacing(near_planar) == "half-cosine"

    # 3. Inverted V-tail with angled junction (dihedral = -38 deg > 5 deg threshold) -> cosine
    v_tail = vt.LiftingSurface(
        name="AerosondeVTail",
        semi_span=0.42,
        dihedral=np.radians(-38.0),
        sweep_le=np.radians(8.0),
        is_symmetric=True,
    )
    assert determine_optimal_spacing(v_tail) == "cosine"

    # 4. Swept wing (dihedral = 0, sweep = 25 deg > 15 deg threshold) -> cosine (root sweep effect)
    swept_wing = vt.LiftingSurface(
        name="SweptWing",
        semi_span=5.0,
        dihedral=0.0,
        sweep_le=np.radians(25.0),
        is_symmetric=True,
    )
    assert determine_optimal_spacing(swept_wing) == "cosine"

    # 5. Asymmetric vertical tail -> half-cosine
    v_fin = vt.LiftingSurface(
        name="VerticalFin",
        semi_span=1.2,
        is_symmetric=False,
    )
    assert determine_optimal_spacing(v_fin) == "half-cosine"


def test_compute_surface_n_panels():
    """Verify span-proportional panel count determination and clamping."""
    vt.LiftingSurface(semi_span=2.0)
    tail = vt.LiftingSurface(semi_span=0.5)

    # Proportional scaling relative to reference semi-span (2.0 m)
    # sqrt(0.5 / 2.0) = 0.5 -> 40 * 0.5 = 20 panels
    n_tail = compute_surface_n_panels(tail, base_n_panels=40, reference_semi_span=2.0)
    assert n_tail == 20

    # Explicit override takes priority
    tail_override = vt.LiftingSurface(semi_span=0.5, n_panels=14)
    assert compute_surface_n_panels(tail_override, base_n_panels=40, reference_semi_span=2.0) == 14

    # Min/max bounds
    tiny_surf = vt.LiftingSurface(semi_span=0.01)
    n_tiny = compute_surface_n_panels(tiny_surf, base_n_panels=40, reference_semi_span=2.0, min_panels=8)
    assert n_tiny == 8


def test_surface_level_mesh_overrides_in_discretize_surface():
    """Verify that discretize_surface prioritizes surface-level n_panels and spacing."""
    wing = vt.LiftingSurface(
        name="CustomWing",
        semi_span=3.0,
        n_panels=25,
        spacing="half-cosine",
    )

    # Calling with solver settings defaults (80, "cosine") should be overridden by wing's attributes
    ds = discretize_surface(wing, n_panels=80, spacing="cosine")
    # For symmetric surface, panels = 2 * n_panels
    assert len(ds.dy_panels) == 50  # 2 * 25
    # Verify spacing is half-cosine (tip dy < root dy for semi-span)
    # Positive semi-span is the right half: indices [25:50]
    dy_right = ds.dy_panels[25:]
    assert dy_right[-1] < dy_right[0]


def test_discretize_aircraft_surfaces_with_proportional_and_auto():
    """Verify discretize_aircraft_surfaces with auto spacing and proportional panel allocation."""
    main_wing = vt.LiftingSurface(
        name="Wing",
        semi_span=2.0,
        dihedral=0.0,
        is_symmetric=True,
    )
    v_tail = vt.LiftingSurface(
        name="Tail",
        semi_span=0.5,
        dihedral=np.radians(-35.0),
        is_symmetric=True,
    )
    ac = vt.Aircraft(surfaces=[main_wing, v_tail])
    settings = vt.SolverSettings(
        n_panels=40,
        spacing="auto",
        proportional_panels=True,
        min_panels=10,
    )

    disc_surfaces = discretize_aircraft_surfaces(ac, settings)
    assert len(disc_surfaces) == 2

    # Main wing: auto -> half-cosine, n_panels = 40 (total = 80)
    ds_wing = disc_surfaces[0]
    assert len(ds_wing.dy_panels) == 80
    dy_w_right = ds_wing.dy_panels[40:]
    assert dy_w_right[-1] < dy_w_right[0]  # tip clustered

    # V-tail: auto -> cosine, n_panels = round(40 * sqrt(0.5/2.0)) = 20 (total = 40)
    ds_tail = disc_surfaces[1]
    assert len(ds_tail.dy_panels) == 40
    dy_t_right = ds_tail.dy_panels[20:]
    np.testing.assert_allclose(dy_t_right[0], dy_t_right[-1], rtol=1e-5)  # symmetric cosine


def test_horseshoe_solver_with_auto_and_proportional():
    """Verify HorseshoeSolver end-to-end execution with auto-spacing and proportional allocation."""
    main_wing = vt.LiftingSurface(
        name="Wing",
        semi_span=1.5,
        sections=[vt.WingSection(y_frac=0.0, chord=0.25), vt.WingSection(y_frac=1.0, chord=0.15)],
        dihedral=0.0,
        is_symmetric=True,
    )
    v_tail = vt.LiftingSurface(
        name="Tail",
        semi_span=0.4,
        sections=[vt.WingSection(y_frac=0.0, chord=0.15), vt.WingSection(y_frac=1.0, chord=0.10)],
        dihedral=np.radians(-38.0),
        is_symmetric=True,
    )
    ac = vt.Aircraft(surfaces=[main_wing, v_tail])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))

    solver = HorseshoeSolver()
    settings = vt.SolverSettings(
        n_panels=40,
        spacing="auto",
        proportional_panels=True,
    )
    res = solver.solve(ac, cond, settings)
    assert res.converged
    assert res.totals.CL > 0
    assert res.totals.CDi > 0
    assert np.isfinite(res.totals.CL)
    assert np.isfinite(res.totals.CDi)
    # Wing should have 80 panels, tail should have round(40 * sqrt(0.4/1.5)) = 21 -> 42 panels
    assert len(res.spanwise[0].y) == 80
    assert len(res.spanwise[1].y) == 42


def test_json_roundtrip_with_meshing_overrides():
    """Verify that per-surface n_panels and spacing serialize and deserialize perfectly."""
    surf = vt.LiftingSurface(
        name="CustomSurface",
        semi_span=2.5,
        n_panels=35,
        spacing="half-cosine",
    )
    ac = vt.Aircraft(name="TestAircraft", surfaces=[surf])
    ac.compute_reference_values()

    json_str = aircraft_to_json(ac)
    assert '"n_panels": 35' in json_str
    assert '"spacing": "half-cosine"' in json_str

    ac_loaded = aircraft_from_json(json_str)
    surf_loaded = ac_loaded.surfaces[0]
    assert surf_loaded.n_panels == 35
    assert surf_loaded.spacing == "half-cosine"
