# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for closed panel meshes of thick lifting surfaces."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from ventorum.core.datatypes import Aircraft, LiftingSurface, SolverSettings, WingSection
from ventorum.geometry.lattice import build_lattice
from ventorum.panel3d.mesh import PanelMesh, check_mesh, mesh_aircraft, mesh_lifting_surface


# Analytic NACA 4-digit sections (textbook formulas of Abbott and von Doenhoff).
def _naca4(m, p, t, n=401, closed=True):
    """Selig ordered unit-chord coordinates of a NACA 4-digit section."""
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    a4 = -0.1036 if closed else -0.1015
    yt = 5.0 * t * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + a4 * x**4
    )
    if m == 0.0:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            m / p**2 * (2.0 * p * x - x**2),
            m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2),
        )
        dyc = np.where(
            x < p, 2.0 * m / p**2 * (p - x), 2.0 * m / (1.0 - p) ** 2 * (p - x)
        )
    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    upper = np.column_stack([xu, yu])[::-1]
    lower = np.column_stack([xl, yl])[1:]
    return np.vstack([upper, lower])


def _naca_half_thickness(x, t, closed=True):
    a4 = -0.1036 if closed else -0.1015
    return 5.0 * t * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + a4 * x**4
    )


def _symmetric_area(t, closed=True):
    """Section area of a symmetric section by manual trapezoidal integration."""
    xs = np.linspace(0.0, 1.0, 4001)
    yt = _naca_half_thickness(xs[1:], t, closed)
    yt0 = _naca_half_thickness(np.array([0.0]), t, closed)[0]
    f = 2.0 * np.concatenate([[yt0], yt])
    dx = xs[1:] - xs[:-1]
    return float(np.sum(0.5 * (f[:-1] + f[1:]) * dx))


def _rect_wing(semi_span=3.0, chord=1.0):
    return LiftingSurface(
        name="Wing",
        semi_span=semi_span,
        sections=[
            WingSection(y_frac=0.0, chord=chord, twist=0.0),
            WingSection(y_frac=1.0, chord=chord, twist=0.0),
        ],
        is_symmetric=True,
    )


def _twisted_wing():
    return LiftingSurface(
        name="Wing",
        semi_span=3.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.5, twist=np.radians(2.0)),
            WingSection(y_frac=1.0, chord=0.8, twist=np.radians(-1.0)),
        ],
        is_symmetric=True,
        sweep_le=np.radians(10.0),
        dihedral=np.radians(5.0),
    )


def _edge_counts(mesh):
    """Map each undirected edge to the number of panels that use it."""
    counts = {}
    for row in np.asarray(mesh.panels):
        for a, b in ((row[0], row[1]), (row[1], row[2]), (row[2], row[3]), (row[3], row[0])):
            if int(a) == int(b):
                continue
            key = (min(int(a), int(b)), max(int(a), int(b)))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _open_edges(mesh):
    return [e for e, c in _edge_counts(mesh).items() if c == 1]


def _wake_points(mesh):
    we = np.asarray(mesh.wake_edges)
    pts = np.asarray(mesh.nodes)[np.unique(we[:, 2:4].ravel())]
    return pts[np.argsort(pts[:, 1])]


def test_rectangular_0012_volume():
    coords = _naca4(0.0, 0.0, 0.12, closed=True)
    surf = _rect_wing()
    mesh = mesh_lifting_surface(surf, coords, n_chord=60, n_span=40)
    info = check_mesh(mesh)
    assert info["watertight"]
    assert info["outward"]
    expected = _symmetric_area(0.12, closed=True) * 2.0 * surf.semi_span
    assert abs(info["volume"] - expected) / expected < 1e-3


def test_tapered_twisted_watertight():
    root = _naca4(0.02, 0.4, 0.12, closed=True)
    tip = _naca4(0.0, 0.0, 0.12, closed=True)
    surf = _twisted_wing()
    mesh = mesh_lifting_surface(surf, [root, tip], n_chord=16, n_span=12)
    info = check_mesh(mesh)
    assert info["watertight"]
    assert info["outward"]


def test_half_mesh_volume_and_symmetry():
    coords = _naca4(0.0, 0.0, 0.12, closed=True)
    surf = _rect_wing()
    full = mesh_lifting_surface(surf, coords, n_chord=24, n_span=12)
    half = mesh_lifting_surface(surf, coords, n_chord=24, n_span=12, half=True)
    assert half.half
    assert not full.half
    vf = check_mesh(full)["volume"]
    vh = check_mesh(half)["volume"]
    assert abs(vh - 0.5 * vf) / abs(vf) < 1e-12
    nodes = np.asarray(full.nodes)
    mirrored = nodes * np.array([1.0, -1.0, 1.0])
    dist, _ = cKDTree(nodes).query(mirrored)
    assert float(np.max(dist)) < 1e-14
    y = nodes[:, 1]
    for a, b in _open_edges(half):
        assert abs(y[a]) < 1e-9 and abs(y[b]) < 1e-9


def test_trailing_edge_matches_vlm_and_wake_count():
    root = _naca4(0.02, 0.4, 0.12, closed=True)
    tip = _naca4(0.0, 0.0, 0.12, closed=True)
    surf = _twisted_wing()
    n_span = 12
    mesh = mesh_lifting_surface(surf, [root, tip], n_chord=12, n_span=n_span)
    assert len(mesh.wake_edges) == 2 * n_span
    lattice = build_lattice(Aircraft(name="T", surfaces=[surf]), SolverSettings(n_panels=n_span))
    te = np.asarray(lattice.surfaces[0].edge_te)
    te = te[np.argsort(te[:, 1])]
    wake = _wake_points(mesh)
    assert wake.shape[0] == te.shape[0]
    np.testing.assert_allclose(wake, te, rtol=0.0, atol=1e-12)


def test_blunt_trailing_edge_base_panels():
    coords = _naca4(0.0, 0.0, 0.12, closed=False)
    surf = _rect_wing()
    n_span = 10
    mesh = mesh_lifting_surface(surf, coords, n_chord=12, n_span=n_span)
    kind = np.asarray(mesh.kind)
    assert int(np.sum(kind == "base")) == 2 * n_span
    assert len(mesh.wake_edges) == 2 * n_span
    info = check_mesh(mesh)
    assert info["watertight"]
    assert info["outward"]


def test_mesh_aircraft_two_components():
    wing = _rect_wing()
    wing.name = "Wing"
    tail = LiftingSurface(
        name="Stab",
        semi_span=1.2,
        sections=[
            WingSection(y_frac=0.0, chord=0.6, twist=0.0),
            WingSection(y_frac=1.0, chord=0.6, twist=0.0),
        ],
        is_symmetric=True,
        position=np.array([3.0, 0.0, 0.2]),
    )
    aircraft = Aircraft(name="T", surfaces=[wing, tail])
    coords = {
        "Wing": _naca4(0.0, 0.0, 0.12, closed=True),
        "Stab": _naca4(0.0, 0.0, 0.10, closed=True),
    }
    mesh = mesh_aircraft(aircraft, coords, n_chord=10, n_span=8)
    assert sorted(np.unique(np.asarray(mesh.component)).tolist()) == [0, 1]
    for comp in (0, 1):
        mask = np.asarray(mesh.component) == comp
        assert int(np.sum(mask)) > 0
        sub = PanelMesh(
            nodes=np.asarray(mesh.nodes),
            panels=np.asarray(mesh.panels)[mask],
            component=np.zeros(int(np.sum(mask)), dtype=int),
            kind=np.asarray(mesh.kind)[mask],
            wake_edges=np.empty((0, 4), dtype=int),
            half=False,
        )
        info = check_mesh(sub)
        assert info["watertight"]
        assert info["outward"]


def test_tip_none_is_open_only_at_tip():
    coords = _naca4(0.0, 0.0, 0.12, closed=True)
    surf = _rect_wing()
    mesh = mesh_lifting_surface(surf, coords, n_chord=10, n_span=8, tip="none")
    assert "tip" not in set(np.asarray(mesh.kind).tolist())
    info = check_mesh(mesh)
    assert info["watertight"]
    opened = _open_edges(mesh)
    assert len(opened) > 0
    max_y = float(np.max(np.abs(np.asarray(mesh.nodes)[:, 1])))
    for a, b in opened:
        y = np.asarray(mesh.nodes)[[a, b], 1]
        assert float(np.min(np.abs(y))) >= max_y - 1e-3 * max_y - 1e-9


def test_uniform_chord_spacing_and_defaults():
    coords = _naca4(0.0, 0.0, 0.12, closed=True)
    surf = _rect_wing()
    mesh = mesh_lifting_surface(surf, coords, n_chord=10, n_span=8, chord_spacing="uniform")
    info = check_mesh(mesh)
    assert info["watertight"]
    assert info["outward"]
    mesh2 = mesh_lifting_surface(surf, coords)
    info2 = check_mesh(mesh2)
    assert info2["watertight"]
    assert info2["outward"]
    with pytest.raises(ValueError):
        mesh_lifting_surface(surf, coords, n_chord=10, n_span=8, chord_spacing="linear")


def test_vertical_fin_closed():
    fin = LiftingSurface(
        name="Fin",
        semi_span=1.5,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, twist=0.0),
            WingSection(y_frac=1.0, chord=0.7, twist=0.0),
        ],
        is_symmetric=False,
        dihedral=np.pi / 2.0,
    )
    coords = _naca4(0.0, 0.0, 0.10, closed=True)
    mesh = mesh_lifting_surface(fin, coords, n_chord=10, n_span=8)
    info = check_mesh(mesh)
    assert info["watertight"]
    assert info["outward"]
    left = fin.mirrored("FinL")
    mesh_left = mesh_lifting_surface(left, coords, n_chord=10, n_span=8)
    info_left = check_mesh(mesh_left)
    assert info_left["watertight"]
    assert float(np.max(np.asarray(mesh_left.nodes)[:, 1])) <= 0.0
