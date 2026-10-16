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
    assert info_left["outward"]
    # The fin in the plane y = 0 has its thickness along y, so its mirror
    # image is the same node set with y negated (an earlier version of this
    # test required y <= 0 for all nodes, which held only because the fin
    # had no thickness).
    mirrored = np.asarray(mesh.nodes) * np.array([1.0, -1.0, 1.0])
    dist, _ = cKDTree(np.asarray(mesh_left.nodes)).query(mirrored)
    assert float(np.max(dist)) < 1e-14
    assert abs(info_left["volume"] - info["volume"]) <= 1e-12 * info["volume"]


def _section_area(coords):
    """Area of a Selig polygon [chord units^2] (shoelace formula)."""
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def test_vertical_fin_has_thickness_along_y():
    """A vertical fin (dihedral 90 deg) has its thickness along y and the right volume.

    Before the dihedral fix the section thickness lay along z, so all nodes
    of the fin were at y = 0 and the volume was zero.
    """
    fin = LiftingSurface(
        name="Fin",
        semi_span=1.5,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, twist=0.0),
            WingSection(y_frac=1.0, chord=1.0, twist=0.0),
        ],
        is_symmetric=False,
        dihedral=np.pi / 2.0,
    )
    coords = _naca4(0.0, 0.0, 0.12, closed=True)
    mesh = mesh_lifting_surface(fin, coords, n_chord=60, n_span=20)
    info = check_mesh(mesh)
    assert info["watertight"]
    assert info["outward"]
    nodes = np.asarray(mesh.nodes)
    half_t = float(np.max(_naca_half_thickness(np.linspace(0.0, 1.0, 2001), 0.12)))
    assert abs(float(np.max(nodes[:, 1])) - half_t) < 2e-3 * half_t
    assert abs(float(np.min(nodes[:, 1])) + half_t) < 2e-3 * half_t
    assert float(np.ptp(nodes[:, 2])) == pytest.approx(1.5, rel=1e-12)
    expected = _symmetric_area(0.12, closed=True) * fin.semi_span
    assert abs(info["volume"] - expected) / expected < 1e-3


def test_dihedral_sections_lie_normal_to_the_span():
    """With 30 deg dihedral each section lies in the plane normal to the span axis.

    The thickness follows the local lattice normal (0, -sin G, cos G), the
    volume is the section area times the true span, and the leading- and
    trailing-edge nodes stay on the vortex-lattice points.
    """
    gam = np.radians(30.0)
    surf = LiftingSurface(
        name="Wing",
        semi_span=2.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, twist=0.0),
            WingSection(y_frac=1.0, chord=1.0, twist=0.0),
        ],
        is_symmetric=True,
        dihedral=gam,
    )
    coords = _naca4(0.0, 0.0, 0.12, closed=True)
    mesh = mesh_lifting_surface(surf, coords, n_chord=60, n_span=16, half=True)
    info = check_mesh(mesh)
    assert info["watertight"]
    assert info["outward"]
    nodes = np.asarray(mesh.nodes)
    station = np.asarray(mesh.node_station)
    span_axis = np.array([0.0, np.cos(gam), np.sin(gam)])
    normal = np.array([0.0, -np.sin(gam), np.cos(gam)])
    for j in range(1, int(station.max()) + 1):
        pts = nodes[station == j]
        le = pts[np.argmin(pts[:, 0])]
        assert float(np.max(np.abs((pts - le) @ span_axis))) < 1e-12
        off = (pts - le) - np.outer((pts - le) @ np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        resid = off - np.outer(off @ normal, normal)
        assert float(np.max(np.abs(resid))) < 1e-12
    # One-sided surface (no root shared with a left half): every section is
    # normal to the span, so the volume is the section area times the span.
    # At the root of a symmetric surface the lattice twists about the y axis,
    # so the root section lies in y = 0 (see the module notes).
    one_side = LiftingSurface(
        name="Panel",
        semi_span=2.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, twist=0.0),
            WingSection(y_frac=1.0, chord=1.0, twist=0.0),
        ],
        is_symmetric=False,
        dihedral=gam,
    )
    info = check_mesh(mesh_lifting_surface(one_side, coords, n_chord=60, n_span=16))
    assert info["watertight"] and info["outward"]
    expected = _symmetric_area(0.12, closed=True) * one_side.semi_span
    assert abs(info["volume"] - expected) / expected < 1e-3
    full = mesh_lifting_surface(surf, coords, n_chord=12, n_span=8)
    lattice = build_lattice(Aircraft(name="T", surfaces=[surf]), SolverSettings(n_panels=8))
    te = np.asarray(lattice.surfaces[0].edge_te)
    te = te[np.argsort(te[:, 1])]
    np.testing.assert_allclose(_wake_points(full), te, rtol=0.0, atol=1e-12)


def _drop_panels(mesh, keep):
    """Return a copy of a mesh with only the panels in *keep*."""
    keep = np.asarray(keep)
    return PanelMesh(
        nodes=np.asarray(mesh.nodes),
        panels=np.asarray(mesh.panels)[keep],
        component=np.asarray(mesh.component)[keep],
        kind=np.asarray(mesh.kind)[keep],
        wake_edges=np.empty((0, 4), dtype=int),
        half=mesh.half,
        tip=mesh.tip,
        node_station=mesh.node_station,
    )


def test_check_mesh_rejects_removed_tip_and_interior_panels():
    """Open edges are accepted only at a half-mesh root and at the tips of tip 'none'."""
    coords = _naca4(0.0, 0.0, 0.12, closed=True)
    wing = mesh_lifting_surface(_rect_wing(), coords, n_chord=10, n_span=8)
    assert check_mesh(wing)["watertight"]
    no_caps = _drop_panels(wing, np.flatnonzero(np.asarray(wing.kind) != "tip"))
    assert not check_mesh(no_caps)["watertight"]
    fin = LiftingSurface(
        name="Fin",
        semi_span=1.5,
        sections=[WingSection(y_frac=0.0, chord=1.0), WingSection(y_frac=1.0, chord=0.7)],
        is_symmetric=False,
        dihedral=np.pi / 2.0,
    )
    fin_mesh = mesh_lifting_surface(fin, coords, n_chord=10, n_span=8)
    assert check_mesh(fin_mesh)["watertight"]
    interior = int(np.flatnonzero(np.asarray(fin_mesh.kind) == "surface")[17])
    holed = _drop_panels(fin_mesh, np.setdiff1d(np.arange(len(fin_mesh.panels)), [interior]))
    assert not check_mesh(holed)["watertight"]
    open_tip = mesh_lifting_surface(_rect_wing(), coords, n_chord=10, n_span=8, tip="none")
    assert check_mesh(open_tip)["watertight"]
    interior = int(np.flatnonzero(np.asarray(open_tip.kind) == "surface")[25])
    holed = _drop_panels(open_tip, np.setdiff1d(np.arange(len(open_tip.panels)), [interior]))
    assert not check_mesh(holed)["watertight"]
    half = mesh_lifting_surface(_rect_wing(), coords, n_chord=10, n_span=8, half=True)
    assert check_mesh(half)["watertight"]
    moved = PanelMesh(
        nodes=np.asarray(half.nodes) + np.array([0.0, 0.05, 0.0]),
        panels=np.asarray(half.panels),
        component=np.asarray(half.component),
        kind=np.asarray(half.kind),
        wake_edges=np.asarray(half.wake_edges),
        half=True,
    )
    assert not check_mesh(moved)["watertight"]


def test_check_mesh_orientation_and_volume_tolerance():
    """Outward needs consistent edge orientation and a volume above a relative tolerance."""
    coords = _naca4(0.0, 0.0, 0.12, closed=True)
    wing = mesh_lifting_surface(_rect_wing(), coords, n_chord=10, n_span=8)
    info = check_mesh(wing)
    assert info["outward"] and info["orientation_consistent"]
    panels = np.asarray(wing.panels).copy()
    panels[5] = panels[5][[0, 3, 2, 1]]
    one_flipped = PanelMesh(
        nodes=np.asarray(wing.nodes), panels=panels, component=np.asarray(wing.component),
        kind=np.asarray(wing.kind), wake_edges=np.asarray(wing.wake_edges),
    )
    info = check_mesh(one_flipped)
    assert not info["orientation_consistent"]
    assert not info["outward"]
    all_flipped = PanelMesh(
        nodes=np.asarray(wing.nodes), panels=np.asarray(wing.panels)[:, [0, 3, 2, 1]],
        component=np.asarray(wing.component), kind=np.asarray(wing.kind),
        wake_edges=np.asarray(wing.wake_edges),
    )
    info = check_mesh(all_flipped)
    assert info["orientation_consistent"]
    assert not info["outward"]
    flat = np.array([[1.0, 0.0], [0.5, 0.0], [0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    plate = mesh_lifting_surface(_rect_wing(), flat, n_chord=4, n_span=4)
    info = check_mesh(plate)
    assert abs(info["volume"]) < 1e-12
    assert not info["outward"]


def test_check_mesh_flags_nonconvex_panels():
    """A bow-tie panel is not convex and makes the panel set invalid."""
    coords = _naca4(0.0, 0.0, 0.12, closed=True)
    wing = mesh_lifting_surface(_rect_wing(), coords, n_chord=10, n_span=8)
    info = check_mesh(wing)
    assert info["n_nonconvex"] == 0
    assert info["valid_panels"]
    panels = np.asarray(wing.panels).copy()
    panels[7] = panels[7][[0, 2, 1, 3]]
    bad = PanelMesh(
        nodes=np.asarray(wing.nodes), panels=panels, component=np.asarray(wing.component),
        kind=np.asarray(wing.kind), wake_edges=np.asarray(wing.wake_edges),
    )
    info = check_mesh(bad)
    assert not bool(info["convex"][7])
    assert info["n_nonconvex"] >= 1
    assert not info["valid_panels"]


def test_strip_metadata_plain_meshes():
    """Strip, chordwise index, side and node station metadata of plain meshes."""
    coords = _naca4(0.0, 0.0, 0.12, closed=False)
    n_chord, n_span = 8, 6
    full = mesh_lifting_surface(_rect_wing(), coords, n_chord=n_chord, n_span=n_span)
    half = mesh_lifting_surface(_rect_wing(), coords, n_chord=n_chord, n_span=n_span, half=True)
    for mesh, n_strips in ((full, 2 * n_span), (half, n_span)):
        strip = np.asarray(mesh.strip)
        chord = np.asarray(mesh.chord_index)
        side = np.asarray(mesh.side)
        kind = np.asarray(mesh.kind)
        assert strip.shape == chord.shape == side.shape == kind.shape
        assert np.asarray(mesh.node_station).shape == (np.asarray(mesh.nodes).shape[0],)
        assert sorted(set(strip[strip >= 0].tolist())) == list(range(n_strips))
        assert np.all(strip[kind == "tip"] == -1)
        nodes = np.asarray(mesh.nodes)
        cen = nodes[np.asarray(mesh.panels)].mean(axis=1)
        y_mid = [float(np.mean(cen[strip == s, 1])) for s in range(n_strips)]
        assert np.all(np.diff(y_mid) > 0.0)
        for s in range(n_strips):
            m = strip == s
            for sd in (1, -1):
                assert sorted(chord[m & (side == sd)].tolist()) == list(range(n_chord))
            assert np.all(kind[m & (side == 0)] == "base")
            assert np.all(chord[m & (kind == "base")] == n_chord)
        for w in np.asarray(mesh.wake_edges):
            assert strip[w[0]] == strip[w[1]]
            assert side[w[0]] == 1 and side[w[1]] == -1
            assert chord[w[0]] == n_chord - 1 and chord[w[1]] == n_chord - 1
    station = np.asarray(full.node_station)
    y = np.asarray(full.nodes)[:, 1]
    assert np.allclose(y[station == station.min()], -3.0) and np.allclose(y[station == station.max()], 3.0)
    wing = _rect_wing()
    tail = LiftingSurface(
        name="Stab",
        semi_span=1.2,
        sections=[WingSection(y_frac=0.0, chord=0.6), WingSection(y_frac=1.0, chord=0.6)],
        is_symmetric=True,
        position=np.array([3.0, 0.0, 0.2]),
    )
    both = mesh_aircraft(Aircraft(name="T", surfaces=[wing, tail]), [coords, coords], n_chord=n_chord, n_span=n_span)
    strip = np.asarray(both.strip)
    comp = np.asarray(both.component)
    assert sorted(set(strip[(comp == 0) & (strip >= 0)].tolist())) == list(range(2 * n_span))
    assert sorted(set(strip[(comp == 1) & (strip >= 0)].tolist())) == list(range(2 * n_span, 4 * n_span))
