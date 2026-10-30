# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Linear exactness and convergence of the 3D panel surface-gradient operator.

On an interior panel the operator reconstructs the doublet values at the
corner nodes with a least-squares linear fit in the node tangent plane,
and then applies the Green-Gauss theorem on the panel polygon. The node
values are exact for every field that is linear in the tangent plane, so
the gradient is exact for a linear field on a flat mesh of any shape:
nodes with three panels, planform kinks and cosine end strips. On a
curved surface the error of a 3D linear field falls with refinement. The
exact tangential gradient of ``mu = g . x`` on a panel with unit normal n
is ``g - (g . n) n``.

The previous operator (inverse-distance node values, Shepard 1968) is not
exact at irregular nodes, and its error there does not decrease with
refinement. The private function ``_build_surface_gradient_operator_shepard``
keeps it for comparison.

Published sources
-----------------
* Blazek, J., 2015, Computational Fluid Dynamics: Principles and
  Applications, 3rd ed., Butterworth-Heinemann, chapter 5 (least-squares
  reconstruction and Green-Gauss gradient).
* Shepard, D., 1968, A two-dimensional interpolation function for
  irregularly-spaced data, Proceedings of the 23rd ACM National Conference,
  pp. 517-524 (inverse distance weights).
* Anderson, J. D., Fundamentals of Aerodynamics, section 4.5 (sphere in a
  uniform flow: mu = -(a / 2) V . n, Cp = 1 - 9/4 sin^2 theta).
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.core.datatypes import LiftingSurface, WingSection
from ventorum.panel3d.kernels import _precompute_geometry
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.post import (
    NODE_WEIGHT_LIMIT,
    _build_surface_gradient_operator_reference,
    _build_surface_gradient_operator_shepard,
    _linear_node_weights,
    build_surface_gradient_operator,
    surface_gradient,
)
from ventorum.panel3d.solver import _standardize_panels
from ventorum.reference.analytic3d import sphere_quad_mesh

G_VECTORS = (np.array([0.3, -0.7, 0.5]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))


def _geometry(nodes: np.ndarray, panels: np.ndarray) -> tuple:
    """Standardized panels and their geometry, as in the solver."""
    nodes = np.asarray(nodes, dtype=float)
    std = _standardize_panels(panels, nodes)
    geom = _precompute_geometry(nodes[std])
    return nodes, std, geom.centroid, geom.normal, geom.area


def _linear_error(op, cen: np.ndarray, nrm: np.ndarray) -> np.ndarray:
    """Largest relative error of each panel over the linear fields of G_VECTORS."""
    err = np.zeros(len(cen))
    for g in G_VECTORS:
        grad = surface_gradient(op, cen @ g)
        exact = g[None, :] - (nrm @ g)[:, None] * nrm
        err = np.maximum(err, np.linalg.norm(grad - exact, axis=1) / np.linalg.norm(g))
    return err


def _orders(values: list[float], counts: list[int]) -> np.ndarray:
    """Observed orders against h ~ N^(-1/2)."""
    v = np.asarray(values)
    h = 1.0 / np.sqrt(np.asarray(counts, dtype=float))
    return np.log(v[:-1] / v[1:]) / np.log(h[:-1] / h[1:])


def _cubed_sphere(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Equiangular cubed-sphere mesh of the unit sphere, n x n quads per face.

    The eight cube corners are nodes with three panels.
    """
    t = np.tan(np.linspace(-0.25 * np.pi, 0.25 * np.pi, n + 1))
    nodes: list[np.ndarray] = []
    index: dict[tuple, int] = {}
    panels = []
    for ax in range(3):
        for sgn in (-1.0, 1.0):
            o1, o2 = (ax + 1) % 3, (ax + 2) % 3
            grid = np.empty((n + 1, n + 1), dtype=int)
            for i in range(n + 1):
                for j in range(n + 1):
                    v = np.zeros(3)
                    v[ax], v[o1], v[o2] = sgn, t[i], t[j]
                    v /= np.linalg.norm(v)
                    key = tuple(np.round(v, 12))
                    if key not in index:
                        index[key] = len(nodes)
                        nodes.append(v)
                    grid[i, j] = index[key]
            for i in range(n):
                for j in range(n):
                    panels.append([grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]])
    nodes_arr = np.asarray(nodes)
    pan = np.asarray(panels)
    v = nodes_arr[pan]
    raw = np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 1])
    inward = np.sum(raw * v.mean(axis=1), axis=1) < 0.0
    pan[inward] = pan[inward][:, ::-1]
    return nodes_arr, pan


def _sheet(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Open quad sheet of a wing planform with irregular nodes.

    * "crease": two flat halves with 10 deg dihedral that meet at y = 0.
    * "kink": a flat planform whose leading-edge sweep changes at y = 1.
    * "cosine": a flat planform with cosine chord and span spacing (end
      strips with the width ratio 1:3:5).
    """
    nx, ny = 10, 12
    xi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, nx + 1)))
    if kind == "crease":
        eta = np.linspace(-2.0, 2.0, ny + 1)
        eta = np.sign(eta) * np.abs(eta) ** 1.3 * 2.0**-0.3
    elif kind == "kink":
        eta = np.linspace(0.0, 3.0, ny + 1) ** 1.1 * 3.0**-0.1
    else:
        eta = 1.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ny + 1)))
    nodes = []
    for x in xi:
        for y in eta:
            if kind == "kink":
                x_le = 0.2 * y if y <= 1.0 else 0.2 + 0.7 * (y - 1.0)
                chord = 1.0 - 0.15 * y
                nodes.append([x_le + chord * x, y, 0.0])
            elif kind == "crease":
                ang = np.radians(10.0)
                nodes.append([x, y * np.cos(ang), abs(y) * np.sin(ang)])
            else:
                nodes.append([x * (1.0 - 0.1 * y), y, 0.0])
    m = ny + 1
    panels = np.array([[i * m + j, (i + 1) * m + j, (i + 1) * m + j + 1, i * m + j + 1] for i in range(nx) for j in range(ny)])
    return np.asarray(nodes), panels


def _cube_corner(n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Flat mesh of three quad sectors around one node (a cube corner laid flat).

    The center node has three panels. A smooth in-plane map makes the other
    nodes irregular too.
    """
    axes = [np.array([np.cos(a), np.sin(a)]) for a in np.radians([90.0, 210.0, 330.0])]
    nodes: list[np.ndarray] = []
    index: dict[tuple, int] = {}
    panels = []
    for k in range(3):
        e1, e2 = axes[k], axes[(k + 1) % 3]
        grid = np.empty((n + 1, n + 1), dtype=int)
        for i in range(n + 1):
            for j in range(n + 1):
                p = (i * e1 + j * e2) / n
                key = tuple(np.round(p, 12))
                if key not in index:
                    index[key] = len(nodes)
                    nodes.append(p)
                grid[i, j] = index[key]
        for i in range(n):
            for j in range(n):
                panels.append([grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]])
    xy = np.asarray(nodes)
    xy = xy + 0.06 * np.column_stack([np.sin(2.0 * xy[:, 1]), np.cos(3.0 * xy[:, 0]) - 1.0])
    pan = np.asarray(panels)
    v = xy[pan]
    turn = (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 3, 1] - v[:, 1, 1]) - (v[:, 2, 1] - v[:, 0, 1]) * (v[:, 3, 0] - v[:, 1, 0])
    pan[turn < 0.0] = pan[turn < 0.0][:, ::-1]
    return np.column_stack([xy, np.zeros(len(xy))]), pan


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
    return np.vstack([np.column_stack([x, yc + yt])[::-1], np.column_stack([x, yc - yt])[1:]])


def _wing_mesh(n_chord: int, n_span: int):
    """Closed full mesh of a tapered, swept wing with a dihedral crease at the root and a kink."""
    surf = LiftingSurface(
        name="Wing",
        semi_span=3.0,
        is_symmetric=True,
        sweep_le=float(np.radians(25.0)),
        dihedral=float(np.radians(8.0)),
        sections=[
            WingSection(y_frac=0.0, chord=1.2),
            WingSection(y_frac=0.4, chord=0.9),
            WingSection(y_frac=1.0, chord=0.45),
        ],
    )
    return mesh_lifting_surface(surf, _naca4(0.02, 0.4, 0.12), n_chord=n_chord, n_span=n_span, half=False)


@pytest.mark.parametrize("kind", ["kink", "cosine", "corner"])
def test_flat_irregular_mesh_linear_field_is_exact(kind):
    mesh = _cube_corner() if kind == "corner" else _sheet(kind)
    nodes, std, cen, nrm, area = _geometry(*mesh)
    op = build_surface_gradient_operator(nodes, std, cen, nrm, area)
    assert _linear_error(op, cen, nrm).max() < 1e-12
    # The inverse-distance node values give errors of several percent.
    old = _build_surface_gradient_operator_shepard(nodes, std, cen, nrm, area)
    assert _linear_error(old, cen, nrm).max() > 0.05


def _sheet_interior(panels: np.ndarray, n_nodes: int) -> np.ndarray:
    """True for each panel of a quad sheet with no corner on the sheet border."""
    count = np.bincount(np.asarray(panels).ravel(), minlength=n_nodes)
    # A node inside a quad sheet is a corner of four panels.
    return np.all(count[panels] == 4, axis=1)


def test_crease_developed_linear_field_is_exact():
    """A field linear in the developed (unfolded) coordinates of a dihedral crease.

    The two halves have the same dihedral angle, so the node tangent plane
    of a crease node is the bisector plane, and the node fit sees the
    developed coordinates times one constant factor. The interior panels
    then get the exact gradient in their own plane. A border panel uses
    the edge-neighbour fit in the panel plane, which shortens a
    displacement across the crease by the cosine of the fold, so the test
    uses the interior panels only.
    """
    nodes, panels = _sheet("crease")
    nodes, std, cen, nrm, area = _geometry(nodes, panels)
    ang = np.radians(10.0)
    s = cen[:, 1] / np.cos(ang)  # signed arc length from the crease
    a, b = 0.7, -1.3
    t_span = np.column_stack([np.zeros(len(cen)), np.full(len(cen), np.cos(ang)), np.sign(cen[:, 1]) * np.sin(ang)])
    exact = a * np.array([1.0, 0.0, 0.0])[None, :] + b * t_span
    inner = _sheet_interior(panels, len(nodes))
    mu = a * cen[:, 0] + b * s
    new = surface_gradient(build_surface_gradient_operator(nodes, std, cen, nrm, area), mu)
    old = surface_gradient(_build_surface_gradient_operator_shepard(nodes, std, cen, nrm, area), mu)
    scale = np.hypot(a, b)
    assert np.max(np.linalg.norm(new - exact, axis=1)[inner]) < 1e-12 * scale
    assert np.max(np.linalg.norm(old - exact, axis=1)[inner]) > 0.01 * scale


def test_cubed_sphere_linear_field_error_falls():
    """The 3D linear field on the cubed sphere: the error falls (observed order 1.2 to 1.9).

    The node fit is linear in the node tangent plane, so the normal offsets
    of the centroids (of the order of the squared panel size times the
    curvature) give a small error that falls with the panel size. The
    previous operator stays at an error of about 0.13 at the cube corners.
    """
    new, old, counts = [], [], []
    for n in (4, 6, 9, 13):
        nodes, std, cen, nrm, area = _geometry(*_cubed_sphere(n))
        counts.append(len(std))
        new.append(_linear_error(build_surface_gradient_operator(nodes, std, cen, nrm, area), cen, nrm).max())
        old.append(_linear_error(_build_surface_gradient_operator_shepard(nodes, std, cen, nrm, area), cen, nrm).max())
    assert np.all(_orders(new, counts) > 1.0), _orders(new, counts)
    assert min(old) > 0.1


def test_wing_linear_field_error_is_smaller():
    """Wing with sweep, taper, a planform kink and a dihedral crease at the root.

    The median of the linear-field error over the surface panels falls to
    less than a third of the value of the previous operator, and the
    largest error does not grow with refinement as it does with the
    previous operator. The percentiles near the top do not fall on every
    mesh: the largest errors of both operators sit at the folds of the
    leading edge (more than 90 deg on the coarse mesh at the root, where
    the crease meets the leading edge) and at the border panels
    (edge-neighbour fit in the panel plane, not changed). There the 3D
    linear field has a component normal to the node tangent plane, which a
    doublet distribution on the surface does not have.
    """
    largest = []
    for n_chord, n_span in ((10, 6), (20, 12), (40, 24)):
        mesh = _wing_mesh(n_chord, n_span)
        nodes, std, cen, nrm, area = _geometry(mesh.nodes, mesh.panels)
        args = (nodes, std, cen, nrm, area, mesh.wake_edges)
        surface = np.asarray(mesh.kind) == "surface"
        new = _linear_error(build_surface_gradient_operator(*args, kinds=mesh.kind), cen, nrm)[surface]
        old = _linear_error(_build_surface_gradient_operator_shepard(*args, kinds=mesh.kind), cen, nrm)[surface]
        assert np.median(new) < 0.3 * np.median(old)
        largest.append(new.max())
    assert largest[-1] < largest[0]


def test_node_weights_rules():
    """The node weights sum to 1, reproduce a linear field, and flag a degenerate group."""
    rng = np.random.default_rng(5)
    dd = np.column_stack([rng.normal(size=5), rng.normal(size=5), 0.01 * rng.normal(size=5)])
    nrm = np.tile([0.0, 0.0, 1.0], (5, 1))
    w, valid = _linear_node_weights(dd, nrm)
    assert valid
    assert np.sum(w) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(w @ dd[:, :2], 0.0, atol=1e-12)
    # Collinear centroids: rank 2, the node keeps the inverse-distance weights.
    line = np.column_stack([np.linspace(-1.0, 1.0, 4), 0.5 * np.linspace(-1.0, 1.0, 4), np.zeros(4)])
    assert not _linear_node_weights(line, nrm[:4])[1]
    # A node far outside its centroids: weights larger than the limit.
    far = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0]])
    w_far, valid_far = _linear_node_weights(far, nrm[:3])
    assert np.max(np.abs(w_far)) > NODE_WEIGHT_LIMIT
    assert not valid_far


def _sphere_cp_error(nodes, panels, builder) -> tuple[float, float]:
    """Max and area-weighted L2 Cp error of the exact sphere doublet (V = x, radius 1)."""
    nodes, std, cen, nrm, area = _geometry(nodes, panels)
    op = builder(nodes, std, cen, nrm, area)
    v = np.array([1.0, 0.0, 0.0])
    u = cen / np.linalg.norm(cen, axis=1)[:, None]
    grad = surface_gradient(op, -0.5 * (u @ v))
    v_t = v[None, :] - (nrm @ v)[:, None] * nrm
    cp = 1.0 - np.sum((v_t - grad) ** 2, axis=1)
    err = cp - (1.0 - 2.25 * (1.0 - (u @ v) ** 2))
    w = area / area.sum()
    return float(np.max(np.abs(err))), float(np.sqrt(np.sum(w * err**2)))


def test_cubed_sphere_cp_converges():
    """With the exact doublet, the max Cp error falls (observed order 1.4 to 1.5).

    The previous operator stays at a max error of about 0.1 from 96 to
    1014 panels (the error at the cube corners does not fall).
    """
    meshes = [_cubed_sphere(n) for n in (4, 6, 9, 13)]
    counts = [len(p) for _, p in meshes]
    new = [_sphere_cp_error(nd, pn, build_surface_gradient_operator)[0] for nd, pn in meshes]
    old = [_sphere_cp_error(nd, pn, _build_surface_gradient_operator_shepard)[0] for nd, pn in meshes]
    order = _orders(new, counts)
    assert np.all(order > 1.2), order
    assert new[-1] < 0.02
    assert min(old) > 0.09


def test_latlon_sphere_cp_l2_order_is_two():
    """Latitude-longitude sphere with pole fans: L2 order about 2 (observed 2.0)."""
    meshes = [sphere_quad_mesh(1.0, n, 2 * n) for n in (8, 12, 16, 24)]
    counts = [len(p) for _, p in meshes]
    l2 = [_sphere_cp_error(nd, pn, build_surface_gradient_operator)[1] for nd, pn in meshes]
    order = _orders(l2, counts)
    assert np.all((order > 1.8) & (order < 2.3)), order


def test_no_values_across_a_sharp_edge():
    """Two flat faces with a 90 deg fold: the values on one face do not change the gradient on the other."""
    n = 5
    nodes, index = [], {}
    for i in range(-n, n + 1):
        for j in range(n + 1):
            index[(i, j)] = len(nodes)
            x = 0.2 * i
            nodes.append([x, 0.2 * j, 0.0] if i <= 0 else [0.0, 0.2 * j, x])
    panels = np.array(
        [[index[(i, j)], index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]] for i in range(-n, n) for j in range(n)]
    )
    nodes, std, cen, nrm, area = _geometry(np.asarray(nodes), panels)
    op = build_surface_gradient_operator(nodes, std, cen, nrm, area)
    face_a = cen[:, 2] < 1e-12
    g = np.array([0.4, -0.9, 0.0])
    rng = np.random.default_rng(3)
    for other in (np.zeros(int((~face_a).sum())), rng.normal(size=int((~face_a).sum()))):
        mu = cen @ g
        mu[~face_a] = other
        grad = surface_gradient(op, mu)
        np.testing.assert_allclose(grad[face_a], np.tile(g, (int(face_a.sum()), 1)), rtol=0.0, atol=1e-12)


def test_array_build_matches_loop_reference():
    """Array and loop builds agree on the cubed sphere and on the wing, for both operators."""
    mesh = _wing_mesh(8, 6)
    cases = [
        (_geometry(*_cubed_sphere(5)), {}),
        (_geometry(*_cube_corner(4)), {}),
        (_geometry(mesh.nodes, mesh.panels), {"wake_edges": mesh.wake_edges, "kinds": mesh.kind}),
    ]
    for args, kw in cases:
        for shepard in (False, True):
            build = _build_surface_gradient_operator_shepard if shepard else build_surface_gradient_operator
            op = build(*args, **kw).tocsr()
            ref = _build_surface_gradient_operator_reference(*args, **kw, shepard=shepard).tocsr()
            op.sort_indices()
            ref.sort_indices()
            np.testing.assert_array_equal(op.indptr, ref.indptr)
            np.testing.assert_array_equal(op.indices, ref.indices)
            assert np.max(np.abs(op.data - ref.data)) <= 1e-12 * np.max(np.abs(ref.data))
