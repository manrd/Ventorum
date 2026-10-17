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
