# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Parity tests of the array build of the 3D panel surface-gradient operator.

The array build :func:`ventorum.panel3d.post.build_surface_gradient_operator`
must give the operator of the loop build
``_build_surface_gradient_operator_reference``: the same sparsity pattern
(also the explicit zeros) and values equal to round-off.

Published sources
-----------------
* Shepard, D., 1968, A two-dimensional interpolation function for
  irregularly-spaced data, Proceedings of the 23rd ACM National Conference,
  pp. 517-524 (inverse distance weights).
* Blazek, J., 2015, Computational Fluid Dynamics: Principles and
  Applications, 3rd ed., Butterworth-Heinemann, chapter 5 (Green-Gauss
  gradient).
"""

from __future__ import annotations

import numpy as np
import pytest

import ventorum.panel3d.solver as solver_module
from ventorum.core.datatypes import Aircraft, ControlSurface, FlightCondition, LiftingSurface, WingSection
from ventorum.panel3d.images import reconstruct_full_mesh
from ventorum.panel3d.kernels import _precompute_geometry
from ventorum.panel3d.mesh import mesh_aircraft, mesh_lifting_surface
from ventorum.panel3d.post import (
    _build_surface_gradient_operator_reference,
    build_surface_gradient_operator,
    surface_gradient,
)
from ventorum.panel3d.solver import PanelSolver, _standardize_panels
from ventorum.reference.analytic3d import ellipsoid_quad_mesh, sphere_quad_mesh


def _naca4(m: float, p: float, t: float, n: int = 41) -> np.ndarray:
    """Selig ordered unit-chord coordinates of a NACA 4-digit section."""
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(x, 1e-12)) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )
    if m > 0.0:
        yc = np.where(x < p, m / p**2 * (2.0 * p * x - x**2), m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2))
    else:
        yc = np.zeros_like(x)
    upper = np.column_stack([x, yc + yt])[::-1]
    lower = np.column_stack([x, yc - yt])[1:]
    return np.vstack([upper, lower])


def _rect_wing(semi_span: float = 4.0, dihedral_deg: float = 0.0) -> LiftingSurface:
    return LiftingSurface(
        name="Wing",
        semi_span=semi_span,
        sections=[WingSection(y_frac=0.0, chord=1.0), WingSection(y_frac=1.0, chord=1.0)],
        is_symmetric=True,
        dihedral=float(np.radians(dihedral_deg)),
    )


def _taper_wing(antisymmetric: bool) -> LiftingSurface:
    ctrl = ControlSurface(
        name="aileron" if antisymmetric else "flap",
        eta_start=0.5,
        eta_end=0.9,
        hinge_x_c=0.75,
        deflection=float(np.radians(10.0)),
        symmetric=not antisymmetric,
    )
    return LiftingSurface(
        name="Wing",
        semi_span=3.0,
        is_symmetric=True,
        sweep_le=float(np.radians(15.0)),
        dihedral=float(np.radians(4.0)),
        sections=[WingSection(y_frac=0.0, chord=1.0), WingSection(y_frac=1.0, chord=0.5)],
        controls=[ctrl],
    )


def _wing_meshes() -> dict:
    """Full meshes as the solver sees them (half meshes made full by mirror)."""
    c12 = _naca4(0.0, 0.0, 0.12)
    c2412 = _naca4(0.02, 0.4, 0.12)
    tail = LiftingSurface(
        name="Tail",
        semi_span=1.2,
        is_symmetric=True,
        position=np.array([3.0, 0.0, 0.3]),
        sections=[WingSection(y_frac=0.0, chord=0.5), WingSection(y_frac=1.0, chord=0.5)],
    )
    wing_tail = mesh_aircraft(
        Aircraft(name="WT", surfaces=[_rect_wing(), tail]), [c12, c12], n_chord=6, n_span=6, half=True
    )
    return {
        "rectangular": reconstruct_full_mesh(mesh_lifting_surface(_rect_wing(), c12, n_chord=8, n_span=8, half=True)),
        "taper_flap": reconstruct_full_mesh(mesh_lifting_surface(_taper_wing(False), c2412, n_chord=8, n_span=12, half=True)),
        "taper_aileron": mesh_lifting_surface(_taper_wing(True), c2412, n_chord=8, n_span=12),
        "wing_tail": reconstruct_full_mesh(wing_tail),
        "tip_none": mesh_lifting_surface(_rect_wing(), c12, n_chord=6, n_span=6, tip="none"),
        "root_fold_guard": reconstruct_full_mesh(
            mesh_lifting_surface(_rect_wing(dihedral_deg=6.0), c12, n_chord=8, n_span=48, half=True)
        ),
    }


def _inputs(nodes, panels) -> tuple:
    """Standardized panels and their geometry, as in the solver."""
    nodes = np.asarray(nodes, dtype=float)
    std = _standardize_panels(panels, nodes)
    geom = _precompute_geometry(nodes[std])
    return nodes, std, geom.centroid, geom.normal, geom.area


def _assert_same_operator(op, ref) -> None:
    """Same pattern after sort_indices, data equal to round-off."""
    op = op.tocsr().copy()
    ref = ref.tocsr().copy()
    op.sort_indices()
    ref.sort_indices()
    assert op.shape == ref.shape
    np.testing.assert_array_equal(op.indptr, ref.indptr)
    np.testing.assert_array_equal(op.indices, ref.indices)
    scale = float(np.max(np.abs(ref.data))) if ref.data.size else 0.0
    diff = float(np.max(np.abs(op.data - ref.data))) if ref.data.size else 0.0
    assert diff <= 1e-12 * scale, (diff, scale)


def _check(nodes, panels, centroids, normals, areas, wake_edges=None, **kw) -> None:
    op = build_surface_gradient_operator(nodes, panels, centroids, normals, areas, wake_edges, **kw)
    ref = _build_surface_gradient_operator_reference(nodes, panels, centroids, normals, areas, wake_edges, **kw)
    _assert_same_operator(op, ref)


@pytest.fixture(scope="module")
def wing_meshes() -> dict:
    return _wing_meshes()


def test_operator_matches_reference_on_wing_meshes(wing_meshes):
    for name, mesh in wing_meshes.items():
        nodes, std, cen, nrm, area = _inputs(mesh.nodes, mesh.panels)
        assert std.shape[0] <= 2000, name
        _check(nodes, std, cen, nrm, area, mesh.wake_edges, kinds=mesh.kind)


def test_operator_matches_reference_without_kinds(wing_meshes):
    for mesh in wing_meshes.values():
        nodes, std, cen, nrm, area = _inputs(mesh.nodes, mesh.panels)
        _check(nodes, std, cen, nrm, area, mesh.wake_edges, kinds=None)
        _check(nodes, std, cen, nrm, area, mesh.wake_edges, kinds=None, fold_angle_deg=30.0)


def test_operator_matches_reference_on_closed_bodies():
    meshes = [
        sphere_quad_mesh(1.0, 10, 16),
        ellipsoid_quad_mesh(2.0, 1.0, 0.5, 12, 16),
        ellipsoid_quad_mesh(1.5, 1.0, 0.8, 9, 11, spacing="uniform"),
    ]
    for nodes, panels in meshes:
        nodes, std, cen, nrm, area = _inputs(nodes, panels)
        _check(nodes, std, cen, nrm, area, None)
        _check(nodes, std, cen, nrm, area, None, kinds=np.full(len(std), "surface"))


def _flat_plate(nx: int = 12, ny: int = 10):
    """Flat uniform grid in z = 0 and its panel geometry."""
    xs = np.linspace(0.0, 1.2, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    panels = np.array(
        [[i * (ny + 1) + j, (i + 1) * (ny + 1) + j, (i + 1) * (ny + 1) + j + 1, i * (ny + 1) + j + 1]
         for i in range(nx) for j in range(ny)]
    )
    centroids = nodes[panels].mean(axis=1)
    normals = np.tile([0.0, 0.0, 1.0], (len(panels), 1))
    areas = np.full(len(panels), (xs[1] - xs[0]) * (ys[1] - ys[0]))
    return nodes, panels, centroids, normals, areas


def test_operator_is_exact_for_linear_field():
    nx, ny = 12, 10
    nodes, panels, cen, nrm, area = _flat_plate(nx, ny)
    # A wake cut line at x = 0.6 for 2 <= j < 8 (left column 5, right column 6).
    cut = np.array([[5 * ny + j, 6 * ny + j, 6 * (ny + 1) + j, 6 * (ny + 1) + j + 1] for j in range(2, 8)])
    g_true = np.array([0.7, -1.3, 0.0])
    mu = cen @ g_true
    interior = [i * ny + j for i in range(1, nx - 1) for j in range(1, ny - 1)]
    for wake in (None, cut):
        op = build_surface_gradient_operator(nodes, panels, cen, nrm, area, wake)
        ref = _build_surface_gradient_operator_reference(nodes, panels, cen, nrm, area, wake)
        _assert_same_operator(op, ref)
        grad = surface_gradient(op, mu)
        assert np.max(np.abs(grad[interior] - g_true)) < 1e-12
        assert np.max(np.abs(surface_gradient(ref, mu)[interior] - g_true)) < 1e-12


def _special_mesh(duplicate: bool):
    """Flat grid with degenerate panels, a fin, an isolated panel and (optional) a duplicate panel.

    Node (i, j) is 6 i + j and panel (i, j) is 5 i + j.
    """
    nodes, panels, _, _, _ = _flat_plate(6, 5)
    n_n = len(nodes)
    n_grid = len(panels)
    extra_nodes = np.array(
        [
            [0.2, 0.7, 0.3],  # apex of a fin on the interior edge 9-10
            [5.0, 5.0, 0.0],  # isolated panel
            [6.0, 5.0, 0.0],
            [6.0, 6.0, 0.0],
            [5.0, 6.0, 0.0],
            [7.0, 5.0, 0.0],  # free panel with a repeated corner
            [8.0, 5.0, 0.0],
            [8.0, 6.0, 0.0],
        ]
    )
    nodes = np.vstack([nodes, extra_nodes])
    extra = [
        [9, 10, n_n, n_n],  # fin triangle: three panels on the edge 9-10
        [n_n + 1, n_n + 2, n_n + 3, n_n + 4],  # isolated panel: no neighbours
        [27, 28, 28, 27],  # two distinct corners: no row, but on the edge 27-28 two times
        [14, 15, 21, 15],  # node 15 is a corner two times (not adjacent)
        [24, 25, 31, 30],  # reversed copy of panel (4, 0) with zero area: no row
        [n_n + 5, n_n + 6, n_n + 7, n_n + 6],  # free panel: each edge two times, no second panel
    ]
    if duplicate:
        extra.append(list(panels[6]))  # copy of panel (1, 1)
    panels = np.vstack([panels, np.asarray(extra)])
    verts = nodes[panels]
    cen = verts.mean(axis=1)
    raw = np.cross(verts[:, 2] - verts[:, 0], verts[:, 3] - verts[:, 1])
    norm = np.linalg.norm(raw, axis=1)
    nrm = np.where(norm[:, None] > 0.0, raw / np.maximum(norm, 1e-300)[:, None], 0.0)
    area = 0.5 * norm
    nrm[n_grid + 3] = [0.0, 0.0, 1.0]
    area[n_grid + 3] = 0.01
    nrm[n_grid + 5] = [0.0, 0.0, 1.0]
    area[n_grid + 5] = 0.3
    area[n_grid + 4] = 0.0
    area[0] = 0.25 * area[0]  # an area that is not the polygon area
    kinds = np.array(["surface"] * len(panels), dtype=object)
    kinds[n_grid] = "tip"
    return nodes, panels, cen, nrm, area, kinds


def test_operator_matches_reference_on_special_cases():
    """Degenerate panels, three panels on one edge, an isolated panel, and wake rows.

    The wake rows include rows that pair a panel with a panel that is not
    across a mesh edge. At a wake node such a row removes a panel from a
    closed group: without a duplicate panel the group opens (the panel gets
    the least-squares row), with a duplicate panel it stays closed with
    other node weights.
    """
    cut_rows = [[1, 6, 7, 8], [0, 5, 6, 7], [3, -1, 9, 10], [10_000, 0, 20, 26]]
    for duplicate in (False, True):
        nodes, panels, cen, nrm, area, kinds = _special_mesh(duplicate)
        partner = len(panels) - 1 if duplicate else 6
        wakes = (None, np.array(cut_rows), np.array([[12, partner, 14, 0]]))
        for kw in ({}, {"kinds": kinds}, {"fold_angle_deg": 30.0}, {"fold_angle_deg": 90.0}, {"fold_angle_deg": 179.0}):
            for wake in wakes:
                _check(nodes, panels, cen, nrm, area, wake, **kw)


@pytest.mark.parametrize("tilt_deg", [90.0, 40.0])
def test_fold_at_the_limit_angle(tilt_deg):
    """Two flat faces with a fold, and fold limits at and near the fold angle.

    The last limit comes from the dot product of two normals at the fold,
    so the fold test sits at round-off: there numpy.dot decides, in the
    array build and in the loop build.
    """
    n = 4
    h = 1.0 / n
    t = np.radians(tilt_deg)
    nodes, index = [], {}
    for i in range(-n, n + 1):
        for j in range(n + 1):
            index[(i, j)] = len(nodes)
            nodes.append([i * h, j * h, 0.0] if i <= 0 else [i * h * np.cos(t), j * h, i * h * np.sin(t)])
    nodes = np.asarray(nodes)
    panels = np.array(
        [[index[(i, j)], index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]] for i in range(-n, n) for j in range(n)]
    )
    nodes, std, cen, nrm, area = _inputs(nodes, panels)
    d = float(np.dot(nrm[(n - 1) * n], nrm[n * n]))
    for angle in (tilt_deg - 1.0, tilt_deg, tilt_deg + 1.0, float(np.degrees(np.arccos(d)))):
        _check(nodes, std, cen, nrm, area, None, kinds=None, fold_angle_deg=angle)


def test_empty_mesh():
    op = build_surface_gradient_operator(np.zeros((0, 3)), np.zeros((0, 4), dtype=int), np.zeros((0, 3)),
                                         np.zeros((0, 3)), np.zeros(0))
    ref = _build_surface_gradient_operator_reference(np.zeros((0, 3)), np.zeros((0, 4), dtype=int), np.zeros((0, 3)),
                                                     np.zeros((0, 3)), np.zeros(0))
    assert op.shape == ref.shape == (0, 0)
    assert op.nnz == ref.nnz == 0


def _small_wing_mesh():
    surf = _rect_wing()
    return mesh_lifting_surface(surf, _naca4(0.0, 0.0, 0.12), n_chord=8, n_span=6, half=True)


@pytest.mark.parametrize("kutta", ["morino", "pressure"])
def test_solver_results_unchanged(monkeypatch, kutta):
    mesh = _small_wing_mesh()
    cond = FlightCondition(V_inf=30.0, alpha=float(np.radians(5.0)))
    new = PanelSolver(mesh, far_ratio=np.inf, kutta=kutta).solve(cond)
    monkeypatch.setattr(solver_module, "build_surface_gradient_operator", _build_surface_gradient_operator_reference)
    old = PanelSolver(mesh, far_ratio=np.inf, kutta=kutta).solve(cond)
    for name in ("CL", "CDi_trefftz", "CD_pressure", "Cm"):
        a, b = float(getattr(new, name)), float(getattr(old, name))
        assert abs(a - b) <= 1e-12 * max(abs(b), 1e-300), (name, a, b)
    cp_new, cp_old = np.asarray(new.cp), np.asarray(old.cp)
    assert np.max(np.abs(cp_new - cp_old)) <= 1e-12 * np.max(np.abs(cp_old))
