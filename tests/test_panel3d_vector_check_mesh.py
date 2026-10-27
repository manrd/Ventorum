# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Parity tests of the vectorized mesh check against the loop reference.

``check_mesh`` computes the per-panel geometry, the edge bookkeeping and
the fold detection with array operations. The former loop implementation
stays as the private reference ``_check_mesh_reference`` (with
``_panel_geometry`` and ``_station_folds``). These tests compare the two
on plain and on damaged meshes: every boolean and every count identical,
every float metric equal to round-off (relative 1e-12 of the largest
magnitude of the array). One test compares ``_station_folds_batch``
against ``_station_folds`` directly.
"""

from __future__ import annotations

import numpy as np

from ventorum.core.datatypes import Aircraft, ControlSurface, LiftingSurface, WingSection
from ventorum.panel3d.mesh import (
    PanelMesh,
    _check_mesh_reference,
    _station_folds,
    _station_folds_batch,
    check_mesh,
    mesh_aircraft,
    mesh_lifting_surface,
)

#: Keys that the two implementations must return equal bit for bit.
BOOL_KEYS = ("watertight", "orientation_consistent", "outward", "planar", "valid_panels")
ARRAY_BOOL_KEYS = ("convex", "folded")
COUNT_KEYS = ("n_folded", "n_nonconvex", "n_open_edges", "n_self_intersections")
#: Keys that the two implementations must return equal to round-off.
FLOAT_SCALAR_KEYS = ("volume", "min_area", "max_aspect_ratio", "max_skew", "max_warp")
FLOAT_ARRAY_KEYS = ("aspect_ratio", "skew", "warp")


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


def _rect_wing(semi_span=3.0, chord=1.0):
    """Symmetric rectangular wing of unit section thickness."""
    return LiftingSurface(
        name="Wing",
        semi_span=semi_span,
        sections=[
            WingSection(y_frac=0.0, chord=chord, twist=0.0),
            WingSection(y_frac=1.0, chord=chord, twist=0.0),
        ],
        is_symmetric=True,
    )


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


def _taper_swept_wing(antisymmetric=False):
    """Tapered swept wing with 4 deg dihedral and a 10 deg control at 50 to 90 % span."""
    ctrl = ControlSurface(
        name="aileron" if antisymmetric else "flap",
        eta_start=0.5,
        eta_end=0.9,
        hinge_x_c=0.75,
        deflection=np.radians(10.0),
        symmetric=not antisymmetric,
    )
    return LiftingSurface(
        name="Wing",
        semi_span=3.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, twist=np.radians(1.0)),
            WingSection(y_frac=1.0, chord=0.5, twist=np.radians(-1.0)),
        ],
        is_symmetric=True,
        sweep_le=np.radians(15.0),
        dihedral=np.radians(4.0),
        controls=[ctrl],
    )


def _panel_mesh(nodes, panels):
    """Return a PanelMesh from raw node and panel arrays (one component)."""
    nodes = np.asarray(nodes, dtype=float).reshape(-1, 3)
    panels = np.asarray(panels, dtype=int).reshape(-1, 4)
    n_p = panels.shape[0]
    return PanelMesh(
        nodes=nodes,
        panels=panels,
        component=np.zeros(n_p, dtype=int),
        kind=np.full(n_p, "surface", dtype="<U7"),
        wake_edges=np.empty((0, 4), dtype=int),
        half=False,
        tip="flat",
    )


def _wing_tail_mesh():
    """Full mesh of a wing and a tail (two components)."""
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
        "Wing": _naca4(0.0, 0.0, 0.12),
        "Stab": _naca4(0.0, 0.0, 0.10),
    }
    return mesh_aircraft(aircraft, coords, n_chord=10, n_span=8)


def _wing_meshes():
    """Named meshes of wings, controls, two components, open tips and root folds."""
    rect = _naca4(0.0, 0.0, 0.12)
    root = _naca4(0.02, 0.4, 0.12)
    tip = _naca4(0.0, 0.0, 0.12)
    return [
        ("rect_half", mesh_lifting_surface(_rect_wing(), rect, n_chord=10, n_span=8, half=True)),
        ("rect_full", mesh_lifting_surface(_rect_wing(), rect, n_chord=10, n_span=8)),
        ("taper_flap_half", mesh_lifting_surface(_taper_swept_wing(), [root, tip], n_chord=12, n_span=8, half=True)),
        ("taper_flap_full", mesh_lifting_surface(_taper_swept_wing(), [root, tip], n_chord=12, n_span=8)),
        ("aileron_full", mesh_lifting_surface(_taper_swept_wing(True), [root, tip], n_chord=12, n_span=8)),
        ("wing_tail_two_components", _wing_tail_mesh()),
        ("open_tip_full", mesh_lifting_surface(_rect_wing(), rect, n_chord=10, n_span=8, tip="none")),
        ("open_tip_half", mesh_lifting_surface(_rect_wing(), rect, n_chord=10, n_span=8, tip="none", half=True)),
        # Root-fold meshes of tests/test_panel3d_mesh.py: every section plane
        # normal to the span, so the root strip folds and the check flags it.
        ("root_fold_30deg", mesh_lifting_surface(_straight_wing(30.0), rect, n_chord=16, n_span=16,
                                                 half=True, fold_guard=False)),
        ("root_fold_6deg", mesh_lifting_surface(_straight_wing(6.0), rect, n_chord=8, n_span=48,
                                                half=True, fold_guard=False)),
    ]


def _damaged_meshes():
    """Named meshes with one defect each, on a rectangular wing or on a raw panel set."""
    rect = _naca4(0.0, 0.0, 0.12)
    base = mesh_lifting_surface(_rect_wing(), rect, n_chord=10, n_span=8)
    nodes = np.asarray(base.nodes)
    panels = np.asarray(base.panels)
    component = np.asarray(base.component)
    kind = np.asarray(base.kind)
    n_all = panels.shape[0]

    def with_rows(rows, comp, kinds):
        """Mesh with the nodes of the base mesh and the given panel rows."""
        rows = np.asarray(rows, dtype=int).reshape(-1, 4)
        return PanelMesh(
            nodes=nodes.copy(),
            panels=rows,
            component=np.asarray(comp, dtype=int).reshape(-1),
            kind=np.asarray(kinds),
            wake_edges=np.empty((0, 4), dtype=int),
            half=False,
            tip=base.tip,
            node_station=np.asarray(base.node_station),
        )

    flipped = panels.copy()
    flipped[5] = flipped[5][[0, 3, 2, 1]]
    interior = int(np.flatnonzero(kind == "surface")[17])
    dup = int(np.flatnonzero(kind == "surface")[7])
    a = int(panels[dup][0])
    b = int(panels[dup][1])
    out = [
        ("one_panel_flipped", with_rows(flipped, component, kind)),
        ("one_panel_removed", with_rows(panels[np.setdiff1d(np.arange(n_all), [interior])],
                                        component[np.setdiff1d(np.arange(n_all), [interior])],
                                        kind[np.setdiff1d(np.arange(n_all), [interior])])),
        ("panel_duplicated", with_rows(np.vstack([panels, panels[dup:dup + 1]]),
                                       np.concatenate([component, component[dup:dup + 1]]),
                                       np.concatenate([kind, kind[dup:dup + 1]]))),
        ("degenerate_zero_area", with_rows(np.vstack([panels, [[a, b, a, b]]]),
                                           np.concatenate([component, component[dup:dup + 1]]),
                                           np.concatenate([kind, kind[dup:dup + 1]]))),
        # One non-planar quad above the warp limit: corner 2 lifts off the plane.
        ("non_planar_panel", _panel_mesh(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.3], [0.0, 1.0, 0.0]], [[0, 1, 2, 3]])),
        # One dart: corner 2 is a reflex corner in the mean plane.
        ("dart_quad", _panel_mesh(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.2, 0.0]], [[0, 1, 2, 3]])),
        # One triangle stored with the repeated corner in each of the four positions.
        ("triangle_repeat_positions", _panel_mesh(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.8, 0.0]],
            [[0, 1, 2, 0], [0, 0, 1, 2], [0, 1, 0, 2], [0, 1, 2, 2]])),
    ]
    return out


#: Quality metrics that are ratios with a natural scale of 1 [-]. On a flat
#: panel the warp is round-off noise (about 1e-16), so a bound relative to its
#: own magnitude compares noise with noise; these keys use the scale 1.
DIMENSIONLESS_KEYS = ("aspect_ratio", "skew", "warp", "max_aspect_ratio", "max_skew", "max_warp")


def _assert_close_roundoff(name, key, new_val, ref_val):
    """Assert that two float scalars or arrays agree to round-off (relative 1e-12).

    Dimensionless quality metrics use at least the scale 1, so that round-off
    noise on flat panels does not count as a difference.
    """
    a = np.asarray(ref_val, dtype=float)
    b = np.asarray(new_val, dtype=float)
    assert a.shape == b.shape, f"{name}: {key} shape {b.shape} != {a.shape}"
    if a.size == 0:
        return
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if key in DIMENSIONLESS_KEYS:
        scale = max(scale, 1.0)
    diff = float(np.max(np.abs(b - a)))
    assert diff <= 1e-12 * scale, f"{name}: {key} differs by {diff:.3e} (scale {scale:.3e})"


def _assert_full_parity(name, mesh):
    """Assert every key, type, boolean, count and float rule of the parity contract."""
    new = check_mesh(mesh)
    ref = _check_mesh_reference(mesh)
    assert set(new) == set(ref), f"{name}: different keys"
    for key in BOOL_KEYS + COUNT_KEYS:
        assert type(new[key]) is type(ref[key]), f"{name}: {key} type"
    for key in ARRAY_BOOL_KEYS:
        assert np.asarray(new[key]).dtype == np.asarray(ref[key]).dtype, f"{name}: {key} dtype"
        assert np.asarray(new[key]).shape == np.asarray(ref[key]).shape, f"{name}: {key} shape"
    for key in FLOAT_SCALAR_KEYS + FLOAT_ARRAY_KEYS:
        _assert_close_roundoff(name, key, new[key], ref[key])
    assert set(new["component_volume"]) == set(ref["component_volume"]), f"{name}: component keys"
    for c in sorted(ref["component_volume"]):
        _assert_close_roundoff(name, f"component_volume[{c}]", np.array([new["component_volume"][c]]),
                               np.array([ref["component_volume"][c]]))
    return new, ref


def test_check_mesh_matches_reference_on_wing_meshes():
    """The vectorized check equals the loop reference on every plain wing mesh."""
    for name, mesh in _wing_meshes():
        new, ref = _assert_full_parity(name, mesh)
        assert ref["watertight"], name
        assert ref["outward"], name
        if name.startswith("root_fold"):
            assert ref["n_folded"] > 0, name
            assert not ref["valid_panels"], name
        else:
            assert new["valid_panels"], name


def test_check_mesh_matches_reference_on_damaged_meshes():
    """The vectorized check equals the loop reference on every damaged mesh."""
    for name, mesh in _damaged_meshes():
        _, ref = _assert_full_parity(name, mesh)
        if name == "one_panel_removed":
            assert ref["n_open_edges"] > 0
            assert not ref["watertight"]
        if name == "panel_duplicated":
            # The copy is geometrically valid, but its four edges are then
            # used three times, so the mesh is not watertight and none is open.
            assert not ref["watertight"]
            assert ref["n_open_edges"] == 0
        if name == "degenerate_zero_area":
            assert ref["min_area"] == 0.0
            assert not ref["valid_panels"]
        if name == "non_planar_panel":
            assert not ref["planar"]
        if name == "dart_quad":
            assert int(np.sum(~np.asarray(ref["convex"]))) >= 1
            assert not ref["valid_panels"]


def test_station_folds_batch_matches_loop():
    """The batched fold detection equals the loop on every mesh and on raw rows."""
    folds_seen = 0
    for name, mesh in _wing_meshes():
        if mesh.node_station is None:
            continue
        nodes = np.asarray(mesh.nodes)
        panels = np.asarray(mesh.panels)
        station = np.asarray(mesh.node_station)
        comp_of = np.asarray(mesh.component)
        loop = _station_folds(nodes, panels, station, comp_of)
        batch = _station_folds_batch(nodes, panels, station, comp_of)
        assert np.array_equal(loop, batch), name
        folds_seen += int(np.sum(loop))
    assert folds_seen > 0
    # Raw rows: one row with two points, one row of collinear points, and
    # one node below the first plane. The two rows with a bad plane must be
    # skipped, and the low node must fold its panel.
    nodes = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        [0.0, 2.0, 0.0], [1.0, 2.0, 0.0],
        [0.0, 3.0, 0.0], [1.0, 3.0, 0.0], [2.0, 3.0, 0.0],
        [0.0, 4.0, 1.0], [1.0, 4.0, 1.0], [2.0, 4.0, -0.5],
    ])
    station = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3])
    panels = np.array([
        [0, 1, 5, 4],
        [1, 2, 7, 6],
        [2, 3, 8, 7],
        [3, 0, 9, 10],
        [9, 10, 11, 0],
    ])
    comp_of = np.zeros(panels.shape[0], dtype=int)
    loop = _station_folds(nodes, panels, station, comp_of)
    assert np.array_equal(loop, [False, False, False, False, True])
    batch = _station_folds_batch(nodes, panels, station, comp_of)
    assert np.array_equal(loop, batch)


def test_check_mesh_booleans_identical():
    """Every boolean and count is identical on all the meshes."""
    for name, mesh in _wing_meshes() + _damaged_meshes():
        new = check_mesh(mesh)
        ref = _check_mesh_reference(mesh)
        for key in BOOL_KEYS:
            assert new[key] == ref[key], f"{name}: {key}"
            assert isinstance(new[key], bool), f"{name}: {key} not bool"
        for key in ARRAY_BOOL_KEYS:
            assert np.array_equal(np.asarray(new[key]), np.asarray(ref[key])), f"{name}: {key}"
        for key in COUNT_KEYS:
            assert new[key] == ref[key], f"{name}: {key}"
            assert isinstance(new[key], int), f"{name}: {key} not int"


def test_check_mesh_root_rule_needs_both_nodes_on_the_plane():
    """An open edge of a half mesh is a root edge only when both nodes lie on y = 0.

    The triangle has one edge on y = 0 and two edges with one node on
    y = 0 and one node at y > 0: those two edges are open edges that are
    not root edges, so the mesh is not watertight. A rule that tests one
    node only would accept them.
    """
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    mesh = PanelMesh(
        nodes=nodes,
        panels=np.array([[0, 1, 2, 2]]),
        component=np.array([0]),
        kind=np.array(["surface"]),
        wake_edges=np.empty((0, 4), dtype=int),
        half=True,
    )
    new = check_mesh(mesh)
    ref = _check_mesh_reference(mesh)
    assert ref["watertight"] is False
    assert new["watertight"] is ref["watertight"]
    assert new["n_open_edges"] == ref["n_open_edges"] == 3
