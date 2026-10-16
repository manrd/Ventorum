# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the exact 3D potential-flow references (sphere, ellipsoid) and their meshes."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from ventorum.reference.analytic3d import (
    ellipsoid_coefficients,
    ellipsoid_cp,
    ellipsoid_quad_mesh,
    ellipsoid_surface_velocity,
    prolate_spheroid_coefficients,
    sphere_cp,
    sphere_quad_mesh,
    sphere_surface_velocity,
)


def _ellipsoid_points(a, b, c, n_theta=9, n_phi=16):
    """Deterministic points on the ellipsoid surface [m]."""
    theta = np.linspace(0.05, np.pi - 0.05, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    return np.column_stack(
        [a * np.sin(t.ravel()) * np.cos(p.ravel()),
         b * np.sin(t.ravel()) * np.sin(p.ravel()),
         c * np.cos(t.ravel())]
    )


def _ellipsoid_normals(points, a, b, c):
    """Exact outward unit normals of the ellipsoid at surface points."""
    grad = np.column_stack(
        [points[:, 0] / a**2, points[:, 1] / b**2, points[:, 2] / c**2]
    )
    return grad / np.linalg.norm(grad, axis=1)[:, None]


def _panel_area_and_volume(nodes, panels):
    """Flat-panel surface area [m^2] and signed volume [m^3] of a quad mesh."""
    area = 0.0
    volume = 0.0
    for quad in panels:
        tris = [(quad[0], quad[1], quad[2])]
        if quad[2] != quad[3]:
            tris.append((quad[0], quad[2], quad[3]))
        for i, j, k in tris:
            pa, pb, pc = nodes[i], nodes[j], nodes[k]
            cross = np.cross(pb - pa, pc - pa)
            area += 0.5 * float(np.linalg.norm(cross))
            volume += float(np.dot(pa, np.cross(pb, pc))) / 6.0
    return area, volume


def _edge_counts(panels):
    """Count of each undirected non-degenerate edge over all panels."""
    counts = Counter()
    for quad in panels:
        for k in range(4):
            a, b = int(quad[k]), int(quad[(k + 1) % 4])
            if a != b:
                counts[(min(a, b), max(a, b))] += 1
    return counts


def _check_diagonal_rule(nodes, panels, center):
    """Diagonal cross product of each panel points out of the body."""
    dots = []
    for quad in panels:
        p1, p2, p3, p4 = nodes[quad[0]], nodes[quad[1]], nodes[quad[2]], nodes[quad[3]]
        diag = np.cross(p3 - p1, p4 - p2)
        centroid = (p1 + p2 + p3) / 3.0
        dots.append(float(np.dot(diag, centroid - center)))
    return np.array(dots)


# Coefficient identities.


def test_coefficient_sum_is_two():
    triples = [
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0),
        (1.0, 2.0, 1.0),
        (0.5, 1.0, 1.0),
        (2.0, 1.0, 0.7),
        (3.0, 2.0, 1.0),
        (0.4, 0.9, 1.6),
    ]
    for a, b, c in triples:
        alpha0, beta0, gamma0 = ellipsoid_coefficients(a, b, c)
        assert alpha0 + beta0 + gamma0 == pytest.approx(2.0, abs=1e-12)


def test_sphere_limit_coefficients():
    for radius in (0.5, 1.0, 2.5):
        alpha0, beta0, gamma0 = ellipsoid_coefficients(radius, radius, radius)
        assert alpha0 == pytest.approx(2.0 / 3.0, abs=1e-12)
        assert beta0 == pytest.approx(2.0 / 3.0, abs=1e-12)
        assert gamma0 == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_prolate_closed_form_matches_general():
    for a, b in [(2.0, 1.0), (3.0, 1.0), (1.5, 1.2), (5.0, 0.5)]:
        closed = prolate_spheroid_coefficients(a, b)
        general = ellipsoid_coefficients(a, b, b)
        for left, right in zip(closed, general):
            assert left == pytest.approx(right, abs=1e-12)


def test_prolate_sphere_limit():
    alpha0, beta0, gamma0 = prolate_spheroid_coefficients(1.0, 1.0)
    assert alpha0 == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert beta0 == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert gamma0 == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_oblate_closed_form_matches_general():
    for a, b in [(0.5, 1.0), (1.0, 2.0)]:
        closed = prolate_spheroid_coefficients(a, b)
        general = ellipsoid_coefficients(a, b, b)
        for left, right in zip(closed, general):
            assert left == pytest.approx(right, abs=1e-12)


def test_coefficients_reject_bad_axes():
    with pytest.raises(ValueError):
        ellipsoid_coefficients(-1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        ellipsoid_coefficients(1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        prolate_spheroid_coefficients(0.0, 1.0)


# Surface velocities and pressure coefficients.


def test_ellipsoid_matches_sphere_for_equal_axes():
    points = _ellipsoid_points(1.5, 1.5, 1.5)
    stream = np.array([4.0, -1.5, 2.0])
    got = ellipsoid_surface_velocity(points, 1.5, 1.5, 1.5, stream)
    expected = sphere_surface_velocity(points, 1.5, stream)
    assert np.max(np.abs(got - expected)) < 1e-12
    assert np.max(np.abs(ellipsoid_cp(points, 1.5, 1.5, 1.5, stream)
                         - sphere_cp(points, 1.5, stream))) < 1e-12


def test_ellipsoid_velocity_is_tangent_for_oblique_stream():
    a, b, c = 2.0, 1.0, 0.7
    points = _ellipsoid_points(a, b, c)
    stream = np.array([10.0, 3.0, -2.0])
    velocity = ellipsoid_surface_velocity(points, a, b, c, stream)
    normals = _ellipsoid_normals(points, a, b, c)
    assert np.max(np.abs(np.einsum("ij,ij->i", velocity, normals))) < 1e-12


def test_sphere_velocity_is_tangent_and_matches_scalar_stream():
    radius = 1.5
    points = _ellipsoid_points(radius, radius, radius)
    speed = 12.0
    from_vector = sphere_surface_velocity(points, radius, np.array([speed, 0.0, 0.0]))
    from_scalar = sphere_surface_velocity(points, radius, speed)
    assert np.max(np.abs(from_vector - from_scalar)) < 1e-12
    normals = points / np.linalg.norm(points, axis=1)[:, None]
    assert np.max(np.abs(np.einsum("ij,ij->i", from_scalar, normals))) < 1e-12


def test_sphere_velocity_with_offset_center():
    center = np.array([1.0, -2.0, 0.5])
    points = _ellipsoid_points(1.0, 1.0, 1.0) + center
    stream = np.array([6.0, 1.0, -1.0])
    got = sphere_surface_velocity(points, 1.0, stream, center=center)
    expected = sphere_surface_velocity(points - center, 1.0, stream)
    assert np.max(np.abs(got - expected)) < 1e-12


def test_sphere_maximum_speed():
    speed = 8.0
    points = _ellipsoid_points(1.0, 1.0, 1.0)
    velocity = sphere_surface_velocity(points, 1.0, speed)
    peak = float(np.max(np.linalg.norm(velocity, axis=1)))
    assert peak == pytest.approx(1.5 * speed, rel=1e-12)


def test_sphere_cp_stagnation_and_equator():
    radius, speed = 2.0, 5.0
    nose = np.array([[radius, 0.0, 0.0], [-radius, 0.0, 0.0]])
    side = np.array([[0.0, radius, 0.0], [0.0, 0.0, radius]])
    assert np.allclose(sphere_cp(nose, radius, speed), 1.0, atol=1e-12)
    assert np.allclose(sphere_cp(side, radius, speed), 1.0 - 9.0 / 4.0, atol=1e-12)


def test_cp_matches_velocity_magnitude():
    a, b, c = 2.0, 1.0, 0.7
    points = _ellipsoid_points(a, b, c)
    stream = np.array([10.0, 3.0, -2.0])
    velocity = ellipsoid_surface_velocity(points, a, b, c, stream)
    expected = 1.0 - np.sum(velocity**2, axis=1) / float(np.dot(stream, stream))
    assert np.allclose(ellipsoid_cp(points, a, b, c, stream), expected, atol=1e-12)


def test_velocity_rejects_bad_input():
    points = _ellipsoid_points(1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        sphere_surface_velocity(points, -1.0, 5.0)
    with pytest.raises(ValueError):
        sphere_surface_velocity(np.zeros((4, 2)), 1.0, 5.0)
    with pytest.raises(ValueError):
        sphere_cp(points, 1.0, 0.0)
    zero = ellipsoid_surface_velocity(points, 2.0, 1.0, 0.7, np.zeros(3))
    assert np.allclose(zero, 0.0, atol=0.0)
    with pytest.raises(ValueError):
        ellipsoid_cp(points, 2.0, 1.0, 0.7, np.zeros(3))
    with pytest.raises(ValueError):
        ellipsoid_cp(points, 2.0, 1.0, 0.7, 0.0)


# Meshes.


def test_sphere_mesh_counts_and_no_duplicates():
    radius, n_lat, n_lon = 1.25, 8, 16
    nodes, panels = sphere_quad_mesh(radius, n_lat, n_lon)
    assert nodes.shape == (2 + (n_lat - 1) * n_lon, 3)
    assert panels.shape == (n_lat * n_lon, 4)
    assert len(np.unique(nodes, axis=0)) == nodes.shape[0]
    assert np.all((panels >= 0) & (panels < nodes.shape[0]))


def test_ellipsoid_mesh_counts_and_spacings():
    for spacing in ("cosine", "uniform"):
        nodes, panels = ellipsoid_quad_mesh(2.0, 1.0, 0.7, 6, 12, spacing=spacing)
        assert nodes.shape == (2 + 5 * 12, 3)
        assert panels.shape == (6 * 12, 4)
        assert len(np.unique(nodes, axis=0)) == nodes.shape[0]
    with pytest.raises(ValueError):
        ellipsoid_quad_mesh(2.0, 1.0, 0.7, 6, 12, spacing="linear")


def test_mesh_rejects_bad_input():
    with pytest.raises(ValueError):
        sphere_quad_mesh(-1.0, 8, 16)
    with pytest.raises(ValueError):
        sphere_quad_mesh(1.0, 1, 16)
    with pytest.raises(ValueError):
        sphere_quad_mesh(1.0, 8, 2)
    with pytest.raises(ValueError):
        ellipsoid_quad_mesh(2.0, -1.0, 0.7, 8, 16)


def test_meshes_are_watertight():
    meshes = (
        sphere_quad_mesh(1.0, 8, 16),
        ellipsoid_quad_mesh(2.0, 1.0, 0.7, 8, 16),
    )
    for _, panels in meshes:
        counts = _edge_counts(panels)
        assert len(counts) > 0
        assert set(counts.values()) == {2}


def test_meshes_point_outward():
    nodes, panels = sphere_quad_mesh(2.0, 8, 16)
    _, volume = _panel_area_and_volume(nodes, panels)
    assert volume == pytest.approx(4.0 / 3.0 * np.pi * 2.0**3, rel=0.1)
    assert volume > 0.0
    assert np.all(_check_diagonal_rule(nodes, panels, np.zeros(3)) > 0.0)
    nodes, panels = ellipsoid_quad_mesh(2.0, 1.0, 0.7, 8, 16)
    _, volume = _panel_area_and_volume(nodes, panels)
    assert volume == pytest.approx(4.0 / 3.0 * np.pi * 2.0 * 1.0 * 0.7, rel=0.1)
    assert volume > 0.0
    assert np.all(_check_diagonal_rule(nodes, panels, np.zeros(3)) > 0.0)


def test_sphere_mesh_area_converges_second_order():
    radius = 2.0
    exact_area = 4.0 * np.pi * radius**2
    exact_volume = 4.0 / 3.0 * np.pi * radius**3
    area_errors = []
    volume_errors = []
    for n_lat in (8, 16, 32):
        nodes, panels = sphere_quad_mesh(radius, n_lat, 2 * n_lat)
        area, volume = _panel_area_and_volume(nodes, panels)
        area_errors.append(abs(area - exact_area) / exact_area)
        volume_errors.append(abs(volume - exact_volume) / exact_volume)
    assert volume_errors[0] > volume_errors[1] > volume_errors[2]
    assert volume_errors[2] < 0.01
    order_1 = np.log2(area_errors[0] / area_errors[1])
    order_2 = np.log2(area_errors[1] / area_errors[2])
    assert order_1 == pytest.approx(2.0, abs=0.2)
    assert order_2 == pytest.approx(2.0, abs=0.2)


def test_prolate_spheroid_axial_flow_maximum_speed():
    """A 2:1 prolate spheroid in axial flow has the maximum surface speed of about 1.210 U.

    Independent value (Lamb 1932, art. 105): for a prolate spheroid with
    eccentricity e = sqrt(1 - (b/a)^2), alpha0 = 2 (1 - e^2) / e^3
    (artanh(e) - e), and the axial flow has the largest surface speed
    2 U / (2 - alpha0) on the equator. The test computes this value here
    with artanh (not with the module) and checks it against the known
    value 1.210 U and against the module velocity, both to 1e-3.
    """
    a, b, speed = 2.0, 1.0, 1.0
    ecc = np.sqrt(1.0 - (b / a) ** 2)
    alpha0 = 2.0 * (1.0 - ecc**2) / ecc**3 * (np.arctanh(ecc) - ecc)
    v_max = 2.0 * speed / (2.0 - alpha0)
    assert abs(v_max - 1.210) < 1e-3
    phi = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    equator = np.column_stack([np.zeros_like(phi), b * np.cos(phi), b * np.sin(phi)])
    vel = ellipsoid_surface_velocity(equator, a, b, b, np.array([speed, 0.0, 0.0]))
    got = np.linalg.norm(vel, axis=1)
    assert np.all(np.abs(got - v_max) <= 1e-3 * v_max)
    theta = np.linspace(0.05, np.pi - 0.05, 41)
    meridian = np.column_stack([a * np.cos(theta), b * np.sin(theta), np.zeros_like(theta)])
    along = np.linalg.norm(ellipsoid_surface_velocity(meridian, a, b, b, np.array([speed, 0.0, 0.0])), axis=1)
    assert int(np.argmax(along)) == 20
    assert float(np.max(along)) <= v_max * (1.0 + 1e-12)
