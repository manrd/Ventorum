# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for closed panel meshes of thick lifting surfaces."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from ventorum.core.datatypes import Aircraft, ControlSurface, LiftingSurface, SolverSettings, WingSection
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
    """The option trailing_edge='base' keeps the base panels of an open trailing edge."""
    coords = _naca4(0.0, 0.0, 0.12, closed=False)
    surf = _rect_wing()
    n_span = 10
    mesh = mesh_lifting_surface(surf, coords, n_chord=12, n_span=n_span, trailing_edge="base")
    kind = np.asarray(mesh.kind)
    assert int(np.sum(kind == "base")) == 2 * n_span
    assert len(mesh.wake_edges) == 2 * n_span
    info = check_mesh(mesh)
    assert info["watertight"]
    assert info["outward"]


def test_open_trailing_edge_is_closed_by_default():
    """Regression: the default mesh closes an open trailing edge with the quartic thickness change.

    The open NACA 0012 (x^4 coefficient -0.1015) becomes the closed NACA
    0012 of Ladson et al. (1996) (coefficient -0.1036): the two meshes
    agree to the interpolation error of the section points. There are no
    base panels, and the camber line (the mean of the upper and the lower
    ordinate) does not move. The wake edges stay one per strip.
    """
    surf = _rect_wing()
    n_span = 6
    open_mesh = mesh_lifting_surface(surf, _naca4(0.0, 0.0, 0.12, closed=False, n=401), n_chord=24, n_span=n_span)
    closed_mesh = mesh_lifting_surface(surf, _naca4(0.0, 0.0, 0.12, closed=True, n=401), n_chord=24, n_span=n_span)
    assert "base" not in set(np.asarray(open_mesh.kind).tolist())
    assert len(open_mesh.wake_edges) == 2 * n_span
    assert np.asarray(open_mesh.nodes).shape == np.asarray(closed_mesh.nodes).shape
    np.testing.assert_allclose(np.asarray(open_mesh.nodes), np.asarray(closed_mesh.nodes), rtol=0.0, atol=2e-6)
    info = check_mesh(open_mesh)
    assert info["watertight"] and info["outward"] and info["valid_panels"]
    # Cambered section: the camber line stays, the trailing edge closes at its mid point.
    from ventorum.panel3d.mesh import _close_trailing_edge

    xt, yu, yl = _resampled_section(_naca4(0.02, 0.4, 0.12, closed=False), 24)
    cu, cl = _close_trailing_edge(xt, yu, yl)
    np.testing.assert_allclose(0.5 * (cu + cl), 0.5 * (yu + yl), rtol=0.0, atol=1e-15)
    assert cu[-1] == pytest.approx(cl[-1], abs=1e-15)
    assert float(np.max(np.abs((cu - cl) - (yu - yl)))) <= float(yu[-1] - yl[-1]) + 1e-15
    with pytest.raises(ValueError):
        mesh_lifting_surface(surf, _naca4(0.0, 0.0, 0.12, closed=False), n_chord=8, n_span=4, trailing_edge="open")


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
    trailing-edge nodes stay on the vortex-lattice points. With 16
    spanwise panels the root strip of this wing is narrower than
    t tan(G), so these planes fold the root strip: the fold guard (on by
    default) turns the planes of the first stations, and the property
    holds for the stations that the guard keeps and for every station
    with ``fold_guard=False`` (a folded mesh, which check_mesh flags).
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
    span_axis = np.array([0.0, np.cos(gam), np.sin(gam)])
    normal = np.array([0.0, -np.sin(gam), np.cos(gam)])

    def normal_to_span(pts):
        le = pts[np.argmin(pts[:, 0])]
        along = float(np.max(np.abs((pts - le) @ span_axis)))
        off = (pts - le) - np.outer((pts - le) @ np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        resid = off - np.outer(off @ normal, normal)
        return along < 1e-12 and float(np.max(np.abs(resid))) < 1e-12

    raw = mesh_lifting_surface(surf, coords, n_chord=60, n_span=16, half=True, fold_guard=False)
    info = check_mesh(raw)
    assert info["watertight"]
    assert info["outward"]
    assert info["n_folded"] > 0 and not info["valid_panels"]
    nodes = np.asarray(raw.nodes)
    station = np.asarray(raw.node_station)
    for j in range(1, int(station.max()) + 1):
        assert normal_to_span(nodes[station == j])
    mesh = mesh_lifting_surface(surf, coords, n_chord=60, n_span=16, half=True)
    info = check_mesh(mesh)
    assert info["watertight"] and info["outward"] and info["valid_panels"] and info["n_folded"] == 0
    nodes = np.asarray(mesh.nodes)
    station = np.asarray(mesh.node_station)
    assert float(np.min(nodes[:, 1])) >= 0.0
    kept = [j for j in range(1, int(station.max()) + 1) if normal_to_span(nodes[station == j])]
    assert len(kept) >= int(station.max()) - 4
    assert kept == list(range(int(station.max()) + 1 - len(kept), int(station.max()) + 1))
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


def _straight_wing(dihedral_deg, symmetric=True):
    """Rectangular unit-chord wing of semi-span 2 m without twist."""
    return LiftingSurface(
        name="Wing",
        semi_span=2.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, twist=0.0),
            WingSection(y_frac=1.0, chord=1.0, twist=0.0),
        ],
        is_symmetric=symmetric,
        dihedral=np.radians(dihedral_deg),
    )


def _same_mesh(a, b):
    """True when two meshes are equal bit for bit (nodes, panels and metadata)."""
    for name in ("nodes", "panels", "component", "kind", "wake_edges", "strip", "chord_index", "side",
                 "node_station"):
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        if x.shape != y.shape or not np.array_equal(x, y):
            return False
    return a.half == b.half and a.tip == b.tip


def _resampled_section(coords, n_chord):
    """Upper and lower ordinates of the mesh stations of one section [chord units]."""
    from ventorum.panel3d.mesh import _resample_section, _target_x

    xt = _target_x(n_chord, "cosine")
    yu, yl = _resample_section(coords, xt)
    return xt, yu, yl


def test_root_thickness_scaling_keeps_the_half_wing_volume():
    """Regression: the root section of a 30 deg half wing was thinner normal to the surface.

    The root section lies in the plane y = 0 (the miter of the two halves).
    Its thickness along z was t cos(30 deg) normal to the first panel row,
    so the half wing had less volume than the same wing at 0 deg (0.26 %
    with 8 spanwise panels). With the root offsets from the camber line
    scaled by 1 / cos(dihedral) the first panel row of a symmetric section
    is an exact prism: the volumes agree to round-off (requirement 1e-3).
    """
    coords = _naca4(0.0, 0.0, 0.12, closed=True)
    gam = np.radians(30.0)
    v_flat = check_mesh(mesh_lifting_surface(_straight_wing(0.0), coords, n_chord=60, n_span=8, half=True))
    mesh = mesh_lifting_surface(_straight_wing(30.0), coords, n_chord=60, n_span=8, half=True)
    info = check_mesh(mesh)
    assert info["watertight"] and info["outward"]
    rel = info["volume"] / v_flat["volume"] - 1.0
    assert abs(rel) < 1e-3
    assert abs(rel) < 1e-12
    old = mesh_lifting_surface(_straight_wing(30.0), coords, n_chord=60, n_span=8, half=True,
                               scale_root_thickness=False)
    assert check_mesh(old)["volume"] / v_flat["volume"] - 1.0 < -2e-3
    # Thickness normal to the first panel row: the root equals the next station.
    nodes = np.asarray(mesh.nodes)
    station = np.asarray(mesh.node_station)
    normal = np.array([0.0, -np.sin(gam), np.cos(gam)])
    t_root = float(np.ptp(nodes[station == 0] @ normal))
    t_next = float(np.ptp(nodes[station == 1] @ normal))
    assert t_root == pytest.approx(t_next, rel=1e-12)
    old_nodes = np.asarray(old.nodes)
    t_old = float(np.ptp(old_nodes[np.asarray(old.node_station) == 0] @ normal))
    assert t_old == pytest.approx(np.cos(gam) * t_next, rel=1e-12)


def test_root_thickness_scaling_of_a_cambered_section_leaves_the_miter_wedge():
    """A cambered root section keeps a known volume difference: the miter wedge.

    The camber line of the root section does not move, so the miter plane
    y = 0 cuts the cambered slab at an angle. The volume difference to the
    0 deg wing is -(sin G + tan G (1 - cos G) / 2) Q, with Q the first
    moment of the section area about the chord line (about -0.42 % for a
    NACA 2412 half wing of semi-span 2 chords at 30 deg). It does not
    depend on the spanwise panel count. The meshes keep every section
    plane normal to the span (``fold_guard=False``): with 16 spanwise
    panels the root strip is narrower than t tan(30 deg), the guard would
    turn the first planes, and the first panel row is then no prism.
    """
    coords = _naca4(0.02, 0.4, 0.12, closed=True)
    gam = np.radians(30.0)
    xt, yu, yl = _resampled_section(coords, 60)
    area = float(np.trapezoid(yu - yl, xt))
    moment = float(0.5 * np.trapezoid(yu**2 - yl**2, xt))
    expected = -(np.sin(gam) + 0.5 * np.tan(gam) * (1.0 - np.cos(gam))) * moment / (area * 2.0)
    for n_span in (8, 16):
        v_flat = check_mesh(mesh_lifting_surface(_straight_wing(0.0), coords, n_chord=60, n_span=n_span,
                                                 half=True, fold_guard=False))["volume"]
        info = check_mesh(mesh_lifting_surface(_straight_wing(30.0), coords, n_chord=60, n_span=n_span,
                                               half=True, fold_guard=False))
        assert info["watertight"] and info["outward"]
        assert abs((info["volume"] / v_flat - 1.0) - expected) < 1e-6


def test_root_thickness_option_off_and_unaffected_surfaces():
    """Option off keeps the former mesh; surfaces without a miter do not change.

    With the option off only the root section differs from the option on;
    its thickness lies along z as before. A flat wing (factor 1), a
    one-sided surface (its root section is normal to its span axis), a
    vertical fin and a control at the root of a flat wing give the same
    mesh bit for bit with the option on and off.
    """
    coords = _naca4(0.02, 0.4, 0.12, closed=True)
    on = mesh_lifting_surface(_straight_wing(30.0), coords, n_chord=12, n_span=8)
    off = mesh_lifting_surface(_straight_wing(30.0), coords, n_chord=12, n_span=8, scale_root_thickness=False)
    for name in ("panels", "kind", "wake_edges", "strip", "chord_index", "side", "node_station"):
        assert np.array_equal(np.asarray(getattr(on, name)), np.asarray(getattr(off, name)))
    station = np.asarray(on.node_station)
    root = station == int(np.median(station[np.abs(np.asarray(on.nodes)[:, 1]) < 1e-12]))
    assert np.array_equal(np.asarray(on.nodes)[~root], np.asarray(off.nodes)[~root])
    assert not np.array_equal(np.asarray(on.nodes)[root], np.asarray(off.nodes)[root])
    assert np.all(np.abs(np.asarray(on.nodes)[root, 1]) < 1e-15)
    # Option off: the root offsets lie along z and equal the section ordinates.
    xt, yu, yl = _resampled_section(coords, 12)
    z_root = np.sort(np.asarray(off.nodes)[root, 2])
    np.testing.assert_allclose(z_root, np.sort(np.concatenate([yu[1:-1], yl[1:-1], [0.5 * (yu[0] + yl[0])],
                                                               [0.5 * (yu[-1] + yl[-1])]])), atol=1e-15)
    flapped = _straight_wing(0.0)
    flapped.controls = [ControlSurface("flap", 0.0, 0.5, 0.7, np.radians(10.0))]
    fin = _straight_wing(90.0, symmetric=False)
    one_side = _straight_wing(30.0, symmetric=False)
    for surf in (_straight_wing(0.0), flapped, fin, one_side):
        for half in (False, True):
            a = mesh_lifting_surface(surf, coords, n_chord=12, n_span=8, half=half)
            b = mesh_lifting_surface(surf, coords, n_chord=12, n_span=8, half=half, scale_root_thickness=False)
            assert _same_mesh(a, b)
    # mesh_aircraft passes the option to every surface.
    tail = _straight_wing(30.0)
    tail.name = "Tail"
    tail.position = np.array([4.0, 0.0, 0.5])
    ac = Aircraft(name="T", surfaces=[_straight_wing(30.0), tail])
    for flag in (True, False):
        both = mesh_aircraft(ac, [coords, coords], n_chord=12, n_span=8, scale_root_thickness=flag)
        single = mesh_lifting_surface(ac.surfaces[0], coords, n_chord=12, n_span=8, scale_root_thickness=flag)
        n0 = np.asarray(single.nodes).shape[0]
        assert np.array_equal(np.asarray(both.nodes)[:n0], np.asarray(single.nodes))


def test_root_thickness_scaling_keeps_edges_on_the_lattice_and_closure():
    """With the scaled root the leading and trailing edges stay on the lattice points.

    Tapered, swept and twisted wing with 30 deg dihedral. The trailing-edge
    nodes are the lattice points. The leading-edge nodes are the lattice
    points for symmetric sections; for a cambered section the leading-edge
    node is the mean of the resampled ordinates at x = 0, which the scaling
    about the camber line does not move. The same wing with a plain flap
    at the root stays watertight and outward.
    """
    root = _naca4(0.02, 0.4, 0.12, closed=True)
    tip = _naca4(0.0, 0.0, 0.12, closed=True)
    surf = _twisted_wing()
    surf.dihedral = np.radians(30.0)
    n_span = 8
    lattice = build_lattice(Aircraft(name="T", surfaces=[surf]), SolverSettings(n_panels=n_span))
    te = np.asarray(lattice.surfaces[0].edge_te)
    te = te[np.argsort(te[:, 1])]
    le = np.asarray(lattice.surfaces[0].edge_le)
    symmetric = mesh_lifting_surface(surf, tip, n_chord=12, n_span=n_span)
    dist, _ = cKDTree(np.asarray(symmetric.nodes)).query(le)
    assert float(np.max(dist)) < 1e-12
    np.testing.assert_allclose(_wake_points(symmetric), te, rtol=0.0, atol=1e-12)
    mesh = mesh_lifting_surface(surf, [root, tip], n_chord=12, n_span=n_span)
    info = check_mesh(mesh)
    assert info["watertight"] and info["outward"]
    np.testing.assert_allclose(_wake_points(mesh), te, rtol=0.0, atol=1e-12)
    old = mesh_lifting_surface(surf, [root, tip], n_chord=12, n_span=n_span, scale_root_thickness=False)
    # The root nodes changed (the option is on by default), the leading-edge nodes did not.
    assert not np.array_equal(np.asarray(mesh.nodes), np.asarray(old.nodes))
    _, i_new = cKDTree(np.asarray(mesh.nodes)).query(le)
    _, i_old = cKDTree(np.asarray(old.nodes)).query(le)
    np.testing.assert_allclose(np.asarray(mesh.nodes)[i_new], np.asarray(old.nodes)[i_old], rtol=0.0, atol=1e-15)
    surf.controls = [ControlSurface("flap", 0.0, 0.4, 0.7, np.radians(15.0))]
    for half in (False, True):
        flapped = mesh_lifting_surface(surf, [root, tip], n_chord=12, n_span=n_span, half=half)
        info = check_mesh(flapped)
        assert info["watertight"] and info["outward"]


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


def _dihedral_wing(dihedral_deg=6.0, semi_span=4.0):
    """Rectangular unit-chord wing with dihedral and the default (cosine) spanwise spacing."""
    return LiftingSurface(
        name="Wing",
        semi_span=semi_span,
        sections=[WingSection(y_frac=0.0, chord=1.0), WingSection(y_frac=1.0, chord=1.0)],
        is_symmetric=True,
        dihedral=np.radians(dihedral_deg),
    )


@pytest.mark.parametrize("n_span", [48, pytest.param(96, marks=pytest.mark.slow)])
def test_fold_guard_keeps_the_root_strip_of_a_dihedral_wing_open(n_span):
    """Regression: a 6 deg dihedral wing folded its root strip from 48 spanwise panels on.

    With the cosine spacing the root strip becomes narrower than t tan(G)
    (t the largest offset of the section from the chord line), so the next
    section plane, normal to the span, crosses the root plane y = 0 inside
    the section: nodes of the second station lay at y < 0 and the strip
    folded. check_mesh saw this only on some meshes (as non-convex
    panels). Now the fold guard turns the first station planes, no node
    of the half mesh lies at y < 0, and check_mesh flags the folded mesh
    of ``fold_guard=False`` on every chordwise count.
    """
    coords = _naca4(0.0, 0.0, 0.12, closed=True)
    surf = _dihedral_wing()
    for n_chord in (8, 16, 32):
        folded = mesh_lifting_surface(surf, coords, n_chord=n_chord, n_span=n_span, half=True, fold_guard=False)
        info = check_mesh(folded)
        assert float(np.min(np.asarray(folded.nodes)[:, 1])) < 0.0
        assert info["n_folded"] > 0 and info["n_nonconvex"] >= info["n_folded"]
        assert not info["valid_panels"]
        mesh = mesh_lifting_surface(surf, coords, n_chord=n_chord, n_span=n_span, half=True)
        info = check_mesh(mesh)
        assert info["watertight"] and info["outward"] and info["valid_panels"]
        assert info["n_folded"] == 0
        assert float(np.min(np.asarray(mesh.nodes)[:, 1])) >= 0.0
    # The full mesh too, and the guarded stations keep the lattice points.
    full = mesh_lifting_surface(surf, coords, n_chord=16, n_span=n_span)
    assert check_mesh(full)["n_folded"] == 0
    lattice = build_lattice(Aircraft(name="T", surfaces=[surf]), SolverSettings(n_panels=n_span))
    te = np.asarray(lattice.surfaces[0].edge_te)
    te = te[np.argsort(te[:, 1])]
    np.testing.assert_allclose(_wake_points(full), te, rtol=0.0, atol=1e-12)
    le = np.asarray(lattice.surfaces[0].edge_le)
    dist, _ = cKDTree(np.asarray(full.nodes)).query(le)
    assert float(np.max(dist)) < 1e-12


def test_fold_guard_margin_and_unchanged_meshes():
    """Every node keeps the fold margin from the neighbour section plane; meshes that do not fold stay.

    The fold guard keeps each node of a guarded station at least
    FOLD_MARGIN times the station spacing on the outboard side of the
    plane of the station before it (the guard measures the spacing between
    the quarter-chord points; the test uses the centroid distance of the
    two sections, so it asks for half the margin). A mesh that does not fold (wider root
    strip, no dihedral, one-sided surface) is the same with and without
    the guard, bit for bit.
    """
    from ventorum.panel3d.mesh import FOLD_MARGIN

    coords = _naca4(0.02, 0.4, 0.12, closed=True)
    surf = _dihedral_wing(dihedral_deg=10.0)
    mesh = mesh_lifting_surface(surf, coords, n_chord=16, n_span=64, half=True)
    nodes = np.asarray(mesh.nodes)
    station = np.asarray(mesh.node_station)
    for j in range(1, 8):
        prev = nodes[station == j - 1]
        cur = nodes[station == j]
        cen = prev.mean(axis=0)
        _, _, vt = np.linalg.svd(prev - cen)
        nrm = vt[-1] if float(np.dot(cur.mean(axis=0) - cen, vt[-1])) > 0.0 else -vt[-1]
        spacing = float(np.dot(cur.mean(axis=0) - cen, nrm))
        assert float(np.min((cur - cen) @ nrm)) >= 0.5 * FOLD_MARGIN * spacing
    for surf_k, n_span in ((_dihedral_wing(6.0), 24), (_dihedral_wing(0.0), 96), (_straight_wing(30.0, symmetric=False), 32)):
        a = mesh_lifting_surface(surf_k, coords, n_chord=12, n_span=n_span)
        b = mesh_lifting_surface(surf_k, coords, n_chord=12, n_span=n_span, fold_guard=False)
        assert _same_mesh(a, b)


def _root_scaling_warnings(surf, coords, **kw):
    """Return the strong root-scaling warnings of one mesh build."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh_lifting_surface(surf, coords, n_chord=12, n_span=8, **kw)
    return [w for w in caught if issubclass(w.category, UserWarning) and "strongly scaled" in str(w.message)]


def test_strong_root_thickness_scaling_warns_above_60_deg():
    """The root-thickness factor 1 / cos(G) above G = 60 deg gives a UserWarning, without a cap.

    The scaling is not capped (owner decision), so a steep first panel row
    gives a large factor (above 2 past 60 deg). The builder then warns that
    the root section is strongly scaled and that the result needs a spanwise convergence study.
    60.5 deg warns, 59.5 deg does not, the option off never warns, and the
    mesh with a control at the root (the control path of the builder)
    warns too. The warned mesh still has the full factor.
    """
    from ventorum.panel3d.mesh import ROOT_SCALE_WARN_DIHEDRAL_DEG

    assert ROOT_SCALE_WARN_DIHEDRAL_DEG == 60.0
    coords = _naca4(0.0, 0.0, 0.12, closed=True)
    for half in (False, True):
        warned = _root_scaling_warnings(_straight_wing(60.5), coords, half=half)
        assert len(warned) == 1
        text = str(warned[0].message)
        assert "60.5 deg" in text and "spanwise convergence study" in text
        assert f"{1.0 / np.cos(np.radians(60.5)):.2f}" in text
        assert _root_scaling_warnings(_straight_wing(59.5), coords, half=half) == []
        for deg in (59.5, 60.5, 75.0):
            assert _root_scaling_warnings(_straight_wing(deg), coords, half=half, scale_root_thickness=False) == []
    flapped = _straight_wing(60.5)
    flapped.controls = [ControlSurface("flap", 0.0, 0.5, 0.7, np.radians(10.0))]
    assert len(_root_scaling_warnings(flapped, coords)) == 1
    flapped_low = _straight_wing(59.5)
    flapped_low.controls = [ControlSurface("flap", 0.0, 0.5, 0.7, np.radians(10.0))]
    assert _root_scaling_warnings(flapped_low, coords) == []
    # No cap: the root offsets grow by the full factor 1 / cos(60.5 deg).
    with pytest.warns(UserWarning, match="strongly scaled"):
        on = mesh_lifting_surface(_straight_wing(60.5), coords, n_chord=12, n_span=8, half=True)
    off = mesh_lifting_surface(_straight_wing(60.5), coords, n_chord=12, n_span=8, half=True,
                               scale_root_thickness=False)
    root_on = np.asarray(on.nodes)[np.asarray(on.node_station) == 0]
    root_off = np.asarray(off.nodes)[np.asarray(off.node_station) == 0]
    assert float(np.ptp(root_on[:, 2])) == pytest.approx(float(np.ptp(root_off[:, 2])) / np.cos(np.radians(60.5)),
                                                         rel=1e-12)
