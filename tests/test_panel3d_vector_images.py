# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Exact parity of array operations and the panel geometry loop references."""

from dataclasses import dataclass, fields, replace

import numpy as np
import pytest

from ventorum.airfoil import airfoil
from ventorum.core.datatypes import Aircraft, ControlSurface, LiftingSurface, WingSection
from ventorum.panel3d.images import (
    _reconstruct_full_mesh_reference,
    _standardize_panel_corners_reference,
    mirror_panels_y,
    mirror_points_y,
    reconstruct_full_mesh,
    reflect_panels_plane,
    reflect_points_plane,
    standardize_panel_corners,
)
from ventorum.panel3d.mesh import PanelMesh, mesh_aircraft, mesh_lifting_surface
from ventorum.panel3d.solver import _standardize_panels, _standardize_panels_reference


def _surface(control=False):
    return LiftingSurface(
        name="Wing", semi_span=3.0, is_symmetric=True,
        sections=[WingSection(y_frac=0.0, chord=1.0),
                  WingSection(y_frac=1.0, chord=0.5 if control else 1.0)],
        sweep_le=np.radians(15.0) if control else 0.0,
        dihedral=np.radians(4.0) if control else 0.0,
        controls=[ControlSurface(name="flap", eta_start=0.4, eta_end=0.8,
                                 hinge_x_c=0.75, deflection=np.radians(10.0),
                                 symmetric=True)] if control else [],
    )


def _same(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected, equal_nan=actual.dtype.kind in "fc")
    # Check NaN payloads and signed zeros as well as numerical equality.
    if actual.dtype.kind in "fiu":
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("control", [False, True])
@pytest.mark.parametrize("half", [False, True])
def test_standardize_matches_reference_on_wing_meshes(control, half):
    mesh = mesh_lifting_surface(_surface(control), airfoil("0012").coords,
                                n_chord=6, n_span=6, half=half, control_edge="gap")
    for nodes in (None, mesh.nodes):
        _same(_standardize_panels(mesh.panels, nodes),
              _standardize_panels_reference(mesh.panels, nodes))


def _random_rows():
    rng = np.random.default_rng(1207)
    corners = rng.normal(size=(2000, 4, 3))
    panels = np.arange(8000).reshape(-1, 4)
    # Adjacent repeats in every position, opposite repeats, two pairs, no repeats.
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
    for k, (a, b) in enumerate(pairs):
        panels[k::10, b] = panels[k::10, a]
    panels[6::10, 1] = panels[6::10, 0]
    panels[6::10, 3] = panels[6::10, 2]
    panels[7::10, :] = panels[7::10, :1]
    # Distinct indices with equal coordinates, including two matching pairs.
    corners[8::10, 0] = corners[8::10, 3]
    corners[8::10, 1] = corners[8::10, 2]
    nodes = corners.reshape(-1, 3)
    return panels, nodes


def _limit_rows():
    rows = []
    # Both sides and the exact limit of the absolute and relative tests.
    for b in (0.0, 1e-9, 1.0, -1.0, 1e8):
        limit = 1e-12 + 1e-5 * abs(b)
        a = b + limit
        for value in (np.nextafter(a, b), a, np.nextafter(a, np.inf)):
            for shift in range(4):
                row = np.array([[7.0, 8.0, 9.0], [-4.0, 2.0, 3.0],
                                [value, 0.0, 0.0], [b, 0.0, 0.0]])
                rows.append(np.roll(row, shift, axis=0))
                # allclose is asymmetric: keep both argument orders covered.
                rows.append(np.roll(row[[0, 1, 3, 2]], shift, axis=0))
    for value in (np.nan, np.inf, -np.inf):
        row = np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0],
                        [value, 0.0, 0.0], [value, 0.0, 0.0]])
        rows.extend(np.roll(row, shift, axis=0) for shift in range(4))
    # A coordinate match at shift 0 must beat an index match at shift 2.
    rows.append(np.array([[8., 9., 10.], [8., 9., 10.], [0., 0., 0.], [0., 0., 0.]]))
    return np.asarray(rows)


def test_standardize_matches_reference_on_random_rows():
    panels, nodes = _random_rows()
    for coords in (None, nodes):
        _same(_standardize_panels(panels, coords), _standardize_panels_reference(panels, coords))
    corners = _limit_rows()
    indices = np.arange(corners.shape[0] * 4).reshape(-1, 4)
    indices[-1, 1] = indices[-1, 0]
    nodes = corners.reshape(-1, 3)
    _same(_standardize_panels(indices, nodes), _standardize_panels_reference(indices, nodes))
    # Once shift 0 matches, the unused out-of-range corner must not be read.
    indices = np.array([[len(nodes) + 1, 1, 2, 2]])
    _same(_standardize_panels(indices, nodes), _standardize_panels_reference(indices, nodes))
    # Index equality still matches when the repeated node has a NaN coordinate.
    indices = np.array([[0, 1, 2, 2], [2, 2, 0, 1]])
    nodes = np.array([[1., 2., 3.], [4., 5., 6.], [np.nan, 0., 0.]])
    _same(_standardize_panels(indices, nodes), _standardize_panels_reference(indices, nodes))


def test_standardize_corners_matches_reference():
    panels, nodes = _random_rows()
    corners = np.concatenate([nodes[panels], _limit_rows()])
    original = corners.copy()
    _same(standardize_panel_corners(corners), _standardize_panel_corners_reference(corners))
    normal = np.array([0.2, -0.3, 0.8])
    normal /= np.linalg.norm(normal)
    with np.errstate(invalid="ignore"):
        _same(mirror_panels_y(corners),
              _standardize_panel_corners_reference(mirror_points_y(corners)[:, [0, 3, 2, 1]]))
        _same(reflect_panels_plane(corners, normal, -0.6),
              _standardize_panel_corners_reference(
                  reflect_points_plane(corners, normal, -0.6)[:, [0, 3, 2, 1]]))
    _same(corners, original)


def _same_mesh(mesh):
    actual = reconstruct_full_mesh(mesh)
    expected = _reconstruct_full_mesh_reference(mesh)
    assert type(actual) is type(expected) is type(mesh)
    for fld in fields(mesh):
        a, b = getattr(actual, fld.name), getattr(expected, fld.name)
        if isinstance(b, np.ndarray):
            _same(a, b)
        else:
            assert a == b
    return actual


@pytest.mark.parametrize("case", ["wing", "wing_tail", "flap"])
def test_reconstruct_full_mesh_matches_reference(case):
    wing = _surface(case == "flap")
    surfaces = [wing]
    if case == "wing_tail":
        surfaces.append(LiftingSurface(
            name="Tail", semi_span=1.2, is_symmetric=True,
            position=np.array([3.0, 0.0, 0.3]),
            sections=[WingSection(y_frac=0.0, chord=0.5), WingSection(y_frac=1.0, chord=0.3)],
        ))
    mesh = mesh_aircraft(Aircraft(surfaces=surfaces), [airfoil("0012").coords] * len(surfaces),
                         n_chord=6, n_span=6, half=True, control_edge="gap")
    full = _same_mesh(mesh)
    assert reconstruct_full_mesh(full) is full
    _same_mesh(replace(mesh, strip=None, node_station=None, chord_index=None, side=None))


@dataclass
class _MetadataMesh(PanelMesh):
    panel_data: np.ndarray | None = None
    node_data: np.ndarray | None = None
    scalar: float = 1.0


def test_reconstruct_metadata_and_symmetry_boundary():
    mesh = _MetadataMesh(
        nodes=np.array([[0., 0., 0.], [1., -0., 0.], [0., 1e-12, 0.],
                        [1., -1e-12, 0.], [0., np.nextafter(1e-12, 0.), 0.],
                        [1., 2., 0.], [0., 4., 0.]]),
        panels=np.array([[0, 1, 2, 2], [2, 3, 4, 5], [0, 2, 5, 5]]),
        component=np.array([7, 2, 7]), kind=np.array(["tip", "surface", "tip"]),
        wake_edges=np.empty((0, 4), dtype=int), half=True,
        strip=np.array([-1, 10, -1]), node_station=np.arange(7),
        panel_data=np.arange(6, dtype=np.int16).reshape(3, 2),
        node_data=np.arange(14, dtype=np.float32).reshape(7, 2),
    )
    # Shared nodes use the component of the last panel that contains them.
    _same_mesh(mesh)
    _same_mesh(replace(mesh, component=np.empty(0, dtype=int), strip=None))


def test_empty_arrays_and_input_dtypes():
    for dtype in (np.int32, np.int64, np.float32):
        panels = np.empty((0, 4), dtype=dtype)
        _same(_standardize_panels(panels), _standardize_panels_reference(panels))
        corners = np.empty((0, 4, 3), dtype=dtype)
        _same(standardize_panel_corners(corners), _standardize_panel_corners_reference(corners))
        rows = np.array([[2, 2, 0, 1], [0, 1, 2, 3]], dtype=dtype)
        _same(_standardize_panels(rows), _standardize_panels_reference(rows))
    mesh = PanelMesh(nodes=np.empty((0, 3)), panels=np.empty((0, 4), dtype=int),
                     component=np.empty(0, dtype=int), kind=np.empty(0, dtype="U7"),
                     wake_edges=np.empty((0, 4), dtype=int), half=True)
    _same_mesh(mesh)


def _criterion_rows():
    """Corner rows near the repeated-corner limit (1e-12 of the panel diameter)."""
    rows = []
    base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    for offset in (0.0, 500.0, -1e4):
        for gap in (0.0, 5e-13, 1e-12, 2e-12, 1e-9, 4.9e-3):
            row = base + offset
            row[3] = row[2] + np.array([0.0, 0.0, gap])
            for shift in range(4):
                rows.append(np.roll(row, shift, axis=0))
    return np.asarray(rows)


def test_repeated_corner_criterion_has_no_relative_part():
    """A short real edge far from the origin is not a repeated corner.

    The old test (numpy.isclose with its default relative tolerance 1e-5)
    took two corners 4.9 mm apart at z = 500 m as one corner. The criterion
    is now a distance of at most 1e-12 of the panel diameter, the limit of
    a degenerate edge in the kernels, with a floor of a few ulp of the
    coordinates (8 eps times the largest absolute corner coordinate).
    """
    # Tip cap panel of a wing (chord 1 m) moved by 500 m in x, y and z.
    cap = np.array([[[0.9330127018922099, 4.0, 0.009329859751005642],
                     [0.9829629131445472, 4.0, 0.002448781896930541],
                     [0.9829629131445472, 4.0, -0.002448781896930541],
                     [0.9330127018922099, 4.0, -0.009329859751005642]]]) + 500.0
    _same(standardize_panel_corners(cap), cap)
    nodes = cap[0]
    _same(_standardize_panels(np.array([[0, 1, 2, 3]]), nodes), np.array([[0, 1, 2, 3]]))
    # Repeated corners within the limit are found at every position and offset.
    corners = _criterion_rows()
    out = standardize_panel_corners(corners)
    diameter = np.sqrt(2.0)
    # The stored gap after rounding of the coordinates (the closing edge of each row).
    stored = np.repeat(np.linalg.norm(corners[::4, 3] - corners[::4, 2], axis=-1), 4)
    # The limit: 1e-12 of the diameter, with a floor of 8 eps times the
    # largest absolute coordinate (the round-off of the coordinates).
    floor = 8.0 * np.finfo(float).eps * np.max(np.abs(corners), axis=(1, 2))
    repeated = stored <= np.maximum(1e-12 * diameter, floor)
    assert repeated.any() and not repeated.all()
    gaps = np.linalg.norm(out[:, 2] - out[:, 3], axis=-1)
    assert np.all(gaps[repeated] == stored[repeated])
    # A row that is not repeated keeps its corner order.
    _same(out[~repeated], corners[~repeated])
    _same(out, _standardize_panel_corners_reference(corners))
    indices = np.arange(corners.shape[0] * 4).reshape(-1, 4)
    flat = corners.reshape(-1, 3)
    _same(_standardize_panels(indices, flat), _standardize_panels_reference(indices, flat))


def _shift_rows():
    """Unit-square rows whose closing edge along x is near the repeated-corner limit [m].

    The gaps 5e-13 m and 9.5e-13 m are repeated corners at the origin
    (below 1e-12 of the diameter). Moved to x + 1e4 m, the coordinate
    round-off (one ulp of 1e4 m is 1.8e-12 m) can make the stored gap one
    ulp, above 1e-12 of the diameter.
    """
    rows = []
    base = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    for gap in (0.0, 5e-13, 9.5e-13, 1e-9, 1e-3):
        row = base.copy()
        row[3] = row[2] + np.array([gap, 0.0, 0.0])
        for shift in range(4):
            rows.append(np.roll(row, shift, axis=0))
    return np.asarray(rows)


def test_repeated_corner_classification_does_not_change_far_from_the_origin():
    """A mesh moved to x + 1e4 m gives the same repeated corners and degenerate edges.

    The limit has a floor of a few ulp of the largest absolute corner
    coordinate, so the round-off of the moved coordinates does not change
    the classification of the standardizations and of the kernels.
    """
    from ventorum.panel3d.kernels import _precompute_geometry

    corners = _shift_rows()
    moved = corners + np.array([1e4, 0.0, 0.0])
    # The round-off of the move opens at least one repeated corner to one ulp.
    gap0 = np.linalg.norm(corners[::4, 3] - corners[::4, 2], axis=-1)
    gap1 = np.linalg.norm(moved[::4, 3] - moved[::4, 2], axis=-1)
    assert np.any((gap0 <= 1e-12) & (gap1 > 1e-12 * np.sqrt(2.0)))

    # Another corner order moves a corner by about 1 m; the round-off of the
    # move is about 1e-12 m.
    back = standardize_panel_corners(moved) - np.array([1e4, 0.0, 0.0])
    np.testing.assert_allclose(back, standardize_panel_corners(corners), rtol=0, atol=1e-9)
    _same(_standardize_panel_corners_reference(moved), standardize_panel_corners(moved))
    indices = np.arange(corners.shape[0] * 4).reshape(-1, 4)
    expected = _standardize_panels(indices, corners.reshape(-1, 3))
    _same(_standardize_panels(indices, moved.reshape(-1, 3)), expected)
    _same(_standardize_panels_reference(indices, moved.reshape(-1, 3)), expected)
    # The kernels: the same degenerate edges (the closing edge of a triangle).
    standard = standardize_panel_corners(corners)
    standard_moved = standardize_panel_corners(moved)
    active = _precompute_geometry(standard).edge_active
    assert not active.all() and active.any()
    np.testing.assert_array_equal(_precompute_geometry(standard_moved).edge_active, active)
    from ventorum.panel3d import kernels_numba

    if kernels_numba.HAVE_NUMBA:
        act = kernels_numba._geometry_nb(np.ascontiguousarray(standard_moved), float(np.finfo(float).eps))[9]
        np.testing.assert_array_equal(act, active)
