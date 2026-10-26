# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for node displacements in the 3D panel method."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ventorum.airfoil import airfoil
from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    NodeDisplacements,
    SolverSettings,
    WingSection,
)
from ventorum.panel3d.mesh import mesh_aircraft, mesh_lifting_surface
from ventorum.solvers.panel3d import PanelMethodSolver


def _test_wing(displacements: NodeDisplacements | None = None) -> LiftingSurface:
    """Return a test lifting surface with optional node displacements."""
    return LiftingSurface(
        name="wing",
        semi_span=2.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, shape="0012"),
            WingSection(y_frac=1.0, chord=0.6, shape="0012"),
        ],
        is_symmetric=True,
        node_displacements=displacements,
    )


def _coords() -> np.ndarray:
    """Return unit-chord coordinates of a NACA 0012 section."""
    return airfoil("0012").coords


def _settings() -> SolverSettings:
    """Return test settings with a small mesh count."""
    return SolverSettings(n_panels=4, n_chord=6, panel_backend="numpy")


def _condition() -> FlightCondition:
    """Return a symmetric test flight condition."""
    return FlightCondition(V_inf=30.0, alpha=float(np.radians(3.0)))


@pytest.mark.parametrize("le_disp,te_disp", [
    (np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.01]]), np.zeros((2, 3))),
    (np.zeros((2, 3)), np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.01]])),
    (np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]])),
])
def test_mesh_lifting_surface_refuses_nonzero_node_displacements(le_disp, te_disp):
    """mesh_lifting_surface raises ValueError when a surface has non-zero displacements."""
    disp = NodeDisplacements(le=le_disp, te=te_disp)
    surf = _test_wing(disp)
    coords = _coords()

    with pytest.raises(ValueError) as excinfo:
        mesh_lifting_surface(surf, coords, n_chord=6, n_span=4)

    msg = str(excinfo.value)
    assert "3D panel method does not yet support" in msg
    assert "vortex lattice" in msg


def test_mesh_aircraft_refuses_nonzero_node_displacements():
    """mesh_aircraft raises ValueError when any surface has non-zero displacements."""
    le = np.zeros((3, 3))
    te = np.zeros((3, 3))
    te[1, 2] = 0.02
    disp = NodeDisplacements(le=le, te=te)
    ac = Aircraft(name="plane", surfaces=[_test_wing(disp)])

    with pytest.raises(ValueError) as excinfo:
        mesh_aircraft(ac, {"wing": _coords()}, n_chord=6, n_span=4)

    msg = str(excinfo.value)
    assert "3D panel method does not yet support" in msg
    assert "vortex lattice" in msg


def test_factory_refuses_nonzero_node_displacements():
    """PanelMethodSolver.solve raises ValueError for non-zero node displacements."""
    le = np.zeros((3, 3))
    te = np.zeros((3, 3))
    le[2, 0] = 0.01
    disp = NodeDisplacements(le=le, te=te)
    ac = Aircraft(name="plane", surfaces=[_test_wing(disp)])
    solver = PanelMethodSolver()

    with pytest.raises(ValueError) as excinfo:
        solver.solve(ac, _condition(), _settings())

    msg = str(excinfo.value)
    assert "3D panel method does not yet support" in msg
    assert "vortex lattice" in msg


def test_factory_solve_sweep_refuses_nonzero_node_displacements():
    """PanelMethodSolver.solve_sweep raises ValueError for non-zero node displacements."""
    le = np.zeros((3, 3))
    te = np.zeros((3, 3))
    te[0, 2] = 0.01
    disp = NodeDisplacements(le=le, te=te)
    ac = Aircraft(name="plane", surfaces=[_test_wing(disp)])
    solver = PanelMethodSolver()

    with pytest.raises(ValueError) as excinfo:
        solver.solve_sweep(ac, _condition(), _settings(), np.radians([0.0, 2.0]))

    msg = str(excinfo.value)
    assert "3D panel method does not yet support" in msg
    assert "vortex lattice" in msg


def test_mesh_builders_accept_all_zero_node_displacements():
    """Mesh builders accept all-zero node displacements with bit-identical output."""
    n_edges = 5
    zeros_disp = NodeDisplacements(
        le=np.zeros((n_edges, 3)),
        te=np.zeros((n_edges, 3)),
        eta=np.linspace(0.0, 1.0, n_edges),
    )
    surf_none = _test_wing(None)
    surf_zero = _test_wing(zeros_disp)
    coords = _coords()

    mesh_none = mesh_lifting_surface(surf_none, coords, n_chord=6, n_span=4)
    mesh_zero = mesh_lifting_surface(surf_zero, coords, n_chord=6, n_span=4)

    assert np.array_equal(mesh_zero.nodes, mesh_none.nodes)
    assert np.array_equal(mesh_zero.panels, mesh_none.panels)
    assert np.array_equal(mesh_zero.component, mesh_none.component)
    assert np.array_equal(mesh_zero.kind, mesh_none.kind)
    assert np.array_equal(mesh_zero.wake_edges, mesh_none.wake_edges)
    assert np.array_equal(mesh_zero.strip, mesh_none.strip)
    assert np.array_equal(mesh_zero.chord_index, mesh_none.chord_index)
    assert np.array_equal(mesh_zero.side, mesh_none.side)
    assert np.array_equal(mesh_zero.node_station, mesh_none.node_station)

    ac_none = Aircraft(name="plane", surfaces=[surf_none])
    ac_zero = Aircraft(name="plane", surfaces=[surf_zero])
    ac_mesh_none = mesh_aircraft(ac_none, {"wing": coords}, n_chord=6, n_span=4)
    ac_mesh_zero = mesh_aircraft(ac_zero, {"wing": coords}, n_chord=6, n_span=4)

    assert np.array_equal(ac_mesh_zero.nodes, ac_mesh_none.nodes)
    assert np.array_equal(ac_mesh_zero.panels, ac_mesh_none.panels)
    assert np.array_equal(ac_mesh_zero.component, ac_mesh_none.component)
    assert np.array_equal(ac_mesh_zero.wake_edges, ac_mesh_none.wake_edges)


def test_factory_accepts_all_zero_node_displacements():
    """PanelMethodSolver accepts all-zero displacements with bit-identical results."""
    n_edges = 5
    zeros_disp = NodeDisplacements(
        le=np.zeros((n_edges, 3)),
        te=np.zeros((n_edges, 3)),
        eta=np.linspace(0.0, 1.0, n_edges),
    )
    ac_none = Aircraft(name="plane", surfaces=[_test_wing(None)])
    ac_zero = Aircraft(name="plane", surfaces=[_test_wing(zeros_disp)])

    solver = PanelMethodSolver()
    st = _settings()
    cond = _condition()

    res_none = solver.solve(ac_none, cond, st)
    res_zero = solver.solve(ac_zero, cond, st)

    for field in ("CL", "CDi", "CY", "Cl", "Cm", "Cn", "CDi_nearfield", "e"):
        val_none = getattr(res_none.totals, field)
        val_zero = getattr(res_zero.totals, field)
        assert np.array_equal(np.asarray(val_zero), np.asarray(val_none)), field

    pa = res_none.details["panel_result"]
    pb = res_zero.details["panel_result"]
    for field in ("cp", "mu", "sigma", "velocity", "mu_w"):
        assert np.array_equal(getattr(pb, field), getattr(pa, field)), field

    assert np.array_equal(res_zero.details["cp"], res_none.details["cp"])
    for s_none, s_zero in zip(res_none.spanwise, res_zero.spanwise, strict=True):
        for f in dataclasses.fields(s_none):
            v_none = getattr(s_none, f.name)
            v_zero = getattr(s_zero, f.name)
            if isinstance(v_none, np.ndarray):
                assert np.array_equal(v_zero, v_none)
            else:
                assert v_zero == v_none
