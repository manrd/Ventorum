# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for Ventorum-3DP source and doublet influence kernels.

Verifies:
* Exact potentials and velocities against 40 x 40 Gauss-Legendre quadrature.
* Point source and doublet asymptotic behavior in the far field.
* Physical relation V = +grad(Phi) via central finite differences.
* Gauss flux theorem on closed cube and sphere panel meshes.
* Continuity and finite values on the panel plane, edge extensions, and corners.
* Symmetry with respect to the y = 0 reflection plane.
* Single-precision float32 dtype preservation and numerical accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.panel3d.kernels import (
    doublet_potential,
    doublet_velocity,
    influence_matrices,
    source_potential,
    source_velocity,
)


def _gauss_legendre_quad(
    panel: np.ndarray,
    targets: np.ndarray,
    order: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute potentials and velocities via order x order Gauss-Legendre quadrature.

    Parameters
    ----------
    panel : numpy.ndarray
        Planar panel vertices with shape (4, 3) in m.
    targets : numpy.ndarray
        Evaluation targets with shape (P, 3) in m.
    order : int
        Gauss-Legendre quadrature order along each parameter axis.

    Returns
    -------
    tuple of numpy.ndarray
        phi_s with shape (P,), v_s with shape (P, 3),
        phi_d with shape (P,), v_d with shape (P, 3).
    """
    xi, w_xi = np.polynomial.legendre.leggauss(order)
    eta, w_eta = np.polynomial.legendre.leggauss(order)
    XI, ETA = np.meshgrid(xi, eta, indexing="ij")
    W = np.outer(w_xi, w_eta).ravel()

    p1, p2, p3, p4 = panel[0], panel[1], panel[2], panel[3]
    N1 = (0.25 * (1.0 - XI) * (1.0 - ETA)).ravel()
    N2 = (0.25 * (1.0 + XI) * (1.0 - ETA)).ravel()
    N3 = (0.25 * (1.0 + XI) * (1.0 + ETA)).ravel()
    N4 = (0.25 * (1.0 - XI) * (1.0 + ETA)).ravel()

    pts = N1[:, None] * p1 + N2[:, None] * p2 + N3[:, None] * p3 + N4[:, None] * p4

    dN1_dxi = (-0.25 * (1.0 - ETA)).ravel()
    dN2_dxi = (0.25 * (1.0 - ETA)).ravel()
    dN3_dxi = (0.25 * (1.0 + ETA)).ravel()
    dN4_dxi = (-0.25 * (1.0 + ETA)).ravel()

    dN1_deta = (-0.25 * (1.0 - XI)).ravel()
    dN2_deta = (-0.25 * (1.0 + XI)).ravel()
    dN3_deta = (0.25 * (1.0 + XI)).ravel()
    dN4_deta = (0.25 * (1.0 - XI)).ravel()

    dX_dxi = dN1_dxi[:, None] * p1 + dN2_dxi[:, None] * p2 + dN3_dxi[:, None] * p3 + dN4_dxi[:, None] * p4
    dX_deta = dN1_deta[:, None] * p1 + dN2_deta[:, None] * p2 + dN3_deta[:, None] * p3 + dN4_deta[:, None] * p4

    cross_prod = np.cross(dX_dxi, dX_deta)
    dA = np.linalg.norm(cross_prod, axis=-1) * W

    n_raw = np.cross(p3 - p1, p4 - p2)
    n_hat = n_raw / np.linalg.norm(n_raw)

    inv_4pi = 1.0 / (4.0 * np.pi)

    P = targets.shape[0]
    phi_s = np.zeros(P, dtype=np.float64)
    v_s = np.zeros((P, 3), dtype=np.float64)
    phi_d = np.zeros(P, dtype=np.float64)
    v_d = np.zeros((P, 3), dtype=np.float64)

    for i in range(P):
        R = targets[i] - pts
        dist = np.linalg.norm(R, axis=-1)
        dist3 = dist**3
        dist5 = dist**5

        phi_s[i] = -inv_4pi * np.sum(dA / dist)
        # V = +grad(Phi): grad of -1/(4 pi r) with respect to the target is +R/(4 pi r^3)
        v_s[i] = inv_4pi * np.sum((dA / dist3)[:, None] * R, axis=0)

        R_dot_n = np.sum(R * n_hat, axis=-1)
        phi_d[i] = -inv_4pi * np.sum(dA * R_dot_n / dist3)

        integrand_vd = n_hat[None, :] / dist3[:, None] - 3.0 * (R_dot_n[:, None] * R) / dist5[:, None]
        v_d[i] = -inv_4pi * np.sum(dA[:, None] * integrand_vd, axis=0)

    return phi_s, v_s, phi_d, v_d


def _panel_diameter(panel: np.ndarray) -> float:
    """Return maximum distance between corner points in m."""
    return float(max(np.linalg.norm(panel[i] - panel[j]) for i in range(4) for j in range(4)))


def _panel_centroid(panel: np.ndarray) -> np.ndarray:
    """Return panel centroid in m (area centroid for triangles)."""
    if np.linalg.norm(panel[3] - panel[2]) < 1e-12:
        return (panel[0] + panel[1] + panel[2]) / 3.0
    return 0.25 * (panel[0] + panel[1] + panel[2] + panel[3])


def _make_cube_mesh(n: int = 4) -> np.ndarray:
    """Construct closed cube mesh on [-1, 1]^3 with 6 x n x n panels."""
    panels = []
    u = np.linspace(-1.0, 1.0, n + 1)
    for i in range(n):
        for j in range(n):
            panels.append([[u[i], u[j], 1.0], [u[i + 1], u[j], 1.0], [u[i + 1], u[j + 1], 1.0], [u[i], u[j + 1], 1.0]])
    for i in range(n):
        for j in range(n):
            panels.append([
                [u[i], u[j + 1], -1.0],
                [u[i + 1], u[j + 1], -1.0],
                [u[i + 1], u[j], -1.0],
                [u[i], u[j], -1.0],
            ])
    for i in range(n):
        for j in range(n):
            panels.append([[1.0, u[i], u[j]], [1.0, u[i + 1], u[j]], [1.0, u[i + 1], u[j + 1]], [1.0, u[i], u[j + 1]]])
    for i in range(n):
        for j in range(n):
            panels.append([
                [-1.0, u[i + 1], u[j]],
                [-1.0, u[i], u[j]],
                [-1.0, u[i], u[j + 1]],
                [-1.0, u[i + 1], u[j + 1]],
            ])
    for i in range(n):
        for j in range(n):
            panels.append([
                [u[i], 1.0, u[j + 1]],
                [u[i + 1], 1.0, u[j + 1]],
                [u[i + 1], 1.0, u[j]],
                [u[i], 1.0, u[j]],
            ])
    for i in range(n):
        for j in range(n):
            panels.append([
                [u[i], -1.0, u[j]],
                [u[i + 1], -1.0, u[j]],
                [u[i + 1], -1.0, u[j + 1]],
                [u[i], -1.0, u[j + 1]],
            ])
    return np.array(panels, dtype=np.float64)


def _make_sphere_mesh(n_lat: int = 16, n_lon: int = 32, r: float = 1.0) -> np.ndarray:
    """Construct closed sphere mesh with triangle poles."""
    theta = np.linspace(0.0, np.pi, n_lat + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, n_lon + 1)
    panels = []
    for i in range(n_lat):
        th0, th1 = theta[i], theta[i + 1]
        for j in range(n_lon):
            ph0, ph1 = phi[j], phi[j + 1]
            p1 = [r * np.sin(th0) * np.cos(ph0), r * np.sin(th0) * np.sin(ph0), r * np.cos(th0)]
            p2 = [r * np.sin(th1) * np.cos(ph0), r * np.sin(th1) * np.sin(ph0), r * np.cos(th1)]
            p3 = [r * np.sin(th1) * np.cos(ph1), r * np.sin(th1) * np.sin(ph1), r * np.cos(th1)]
            p4 = [r * np.sin(th0) * np.cos(ph1), r * np.sin(th0) * np.sin(ph1), r * np.cos(th0)]
            if i == 0:
                panels.append([p2, p3, p1, p1])
            elif i == n_lat - 1:
                panels.append([p4, p1, p2, p2])
            else:
                panels.append([p1, p2, p3, p4])
    return np.array(panels, dtype=np.float64)


@pytest.fixture
def test_panels() -> dict[str, np.ndarray]:
    """Collection of benchmark panel geometries."""
    return {
        "square": np.array([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ], dtype=np.float64),
        "skewed": np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.5, 0.0],
            [2.5, 2.0, 0.0],
            [0.5, 1.5, 0.0],
        ], dtype=np.float64),
        "high_ar": np.array([
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [20.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64),
        "triangle": np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.5, 0.0],
            [1.0, 1.5, 0.0],
        ], dtype=np.float64),
    }


def test_quadrature_exactness(test_panels):
    """Potential and velocity match 40x40 Gauss-Legendre quadrature to 1e-9 relative."""
    for name, p in test_panels.items():
        diam = _panel_diameter(p)
        p_in = p[None, ...]
        centroid = _panel_centroid(p)

        # Generate target points at 0.3 to 4.0 diameters in diverse spatial directions
        distances = [0.35 * diam, 0.7 * diam, 1.5 * diam, 3.2 * diam]
        directions = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0.6, 0.8, 0.5]) / np.linalg.norm([0.6, 0.8, 0.5]),
            np.array([-0.7, 0.3, -0.65]) / np.linalg.norm([-0.7, 0.3, -0.65]),
            np.array([0.2, -0.9, 0.38]) / np.linalg.norm([0.2, -0.9, 0.38]),
        ]

        targets = np.array([centroid + d * u for d in distances for u in directions])

        # Exact kernels with far_ratio = inf (force exact calculation)
        phi_s_exact = source_potential(targets, p_in, far_ratio=np.inf)[:, 0]
        v_s_exact = source_velocity(targets, p_in, far_ratio=np.inf)[:, 0, :]
        phi_d_exact = doublet_potential(targets, p_in, far_ratio=np.inf)[:, 0]
        v_d_exact = doublet_velocity(targets, p_in, far_ratio=np.inf)[:, 0, :]

        # Gauss-Legendre reference (order=60 ensures 1e-14 quadrature precision on triangles)
        phi_s_ref, v_s_ref, phi_d_ref, v_d_ref = _gauss_legendre_quad(p, targets, order=60)

        assert np.allclose(phi_s_exact, phi_s_ref, rtol=1e-9, atol=1e-12), f"phi_s failed for {name}"
        assert np.allclose(v_s_exact, v_s_ref, rtol=1e-9, atol=1e-12), f"v_s failed for {name}"
        assert np.allclose(phi_d_exact, phi_d_ref, rtol=1e-9, atol=1e-12), f"phi_d failed for {name}"
        assert np.allclose(v_d_exact, v_d_ref, rtol=1e-9, atol=1e-12), f"v_d failed for {name}"


def test_far_field_asymptotics(test_panels):
    """At 10 diameters, the point formulas agree with the exact ones.

    The documented accuracy of the point approximation at 10 panel
    diameters is 1e-3 relative for the source potential, 2e-3 for the
    doublet potential and 3e-3 for the velocities (the quadrupole term of
    order (D / r)^2 that the point formulas leave out). The value 1e-3
    holds for all four quantities and all test panels at 25 diameters;
    see ``test_far_field_rule_at_25_diameters``.
    """
    for name, p in test_panels.items():
        diam = _panel_diameter(p)
        centroid = _panel_centroid(p)
        p_in = p[None, ...]

        # Targets at 10 panel diameters
        dirs = [
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.57735, 0.57735, 0.57735]),
            np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8]),
        ]
        targets = np.array([centroid + 10.0 * diam * u for u in dirs])

        # Exact computation
        phi_s_exact = source_potential(targets, p_in, far_ratio=np.inf)[:, 0]
        v_s_exact = source_velocity(targets, p_in, far_ratio=np.inf)[:, 0, :]
        phi_d_exact = doublet_potential(targets, p_in, far_ratio=np.inf)[:, 0]
        v_d_exact = doublet_velocity(targets, p_in, far_ratio=np.inf)[:, 0, :]

        # Far field computation (far_ratio=5.0 triggers far-field at 10.0 diameters)
        phi_s_far = source_potential(targets, p_in, far_ratio=5.0)[:, 0]
        v_s_far = source_velocity(targets, p_in, far_ratio=5.0)[:, 0, :]
        phi_d_far = doublet_potential(targets, p_in, far_ratio=5.0)[:, 0]
        v_d_far = doublet_velocity(targets, p_in, far_ratio=5.0)[:, 0, :]

        assert np.allclose(phi_s_far, phi_s_exact, rtol=1e-3), f"phi_s far field failed for {name}"
        assert np.allclose(v_s_far, v_s_exact, rtol=3e-3, atol=1e-7), f"v_s far field failed for {name}"

        # Doublet potential is zero on the panel plane (u_z = 0), check with atol
        non_zero_d = np.abs(phi_d_exact) > 1e-12
        if np.any(non_zero_d):
            assert np.allclose(
                phi_d_far[non_zero_d], phi_d_exact[non_zero_d], rtol=2e-3
            ), f"phi_d far field failed for {name}"
        assert np.allclose(v_d_far, v_d_exact, rtol=3e-3, atol=1e-7), f"v_d far field failed for {name}"


def test_gradient_consistency(test_panels):
    """Velocity equals plus the numerical gradient of the potential to 1e-6 relative."""
    for name, p in test_panels.items():
        diam = _panel_diameter(p)
        centroid = _panel_centroid(p)
        p_in = p[None, ...]

        targets = np.array([
            centroid + np.array([0.4, 0.5, 0.7]) * diam,
            centroid + np.array([-0.6, 0.8, -1.2]) * diam,
            centroid + np.array([1.5, -0.3, 0.9]) * diam,
        ])

        v_s_kernel = source_velocity(targets, p_in, far_ratio=np.inf)[:, 0, :]
        v_d_kernel = doublet_velocity(targets, p_in, far_ratio=np.inf)[:, 0, :]

        h = 1e-6 * diam
        v_s_num = np.zeros_like(v_s_kernel)
        v_d_num = np.zeros_like(v_d_kernel)

        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = h

            tg_plus = targets + delta
            tg_minus = targets - delta

            phi_s_plus = source_potential(tg_plus, p_in, far_ratio=np.inf)[:, 0]
            phi_s_minus = source_potential(tg_minus, p_in, far_ratio=np.inf)[:, 0]
            v_s_num[:, axis] = (phi_s_plus - phi_s_minus) / (2.0 * h)

            phi_d_plus = doublet_potential(tg_plus, p_in, far_ratio=np.inf)[:, 0]
            phi_d_minus = doublet_potential(tg_minus, p_in, far_ratio=np.inf)[:, 0]
            v_d_num[:, axis] = (phi_d_plus - phi_d_minus) / (2.0 * h)

        assert np.allclose(v_s_kernel, v_s_num, rtol=1e-6), f"source gradient failed for {name}"
        assert np.allclose(v_d_kernel, v_d_num, rtol=1e-6), f"doublet gradient failed for {name}"


def test_gauss_law_cube():
    """Gauss law on closed cube: interior sum is 1, exterior sum is 0, centroid limits consistent."""
    cube = _make_cube_mesh(n=4)

    # Interior point
    x_int = np.array([[0.2, -0.3, 0.15]])
    phi_d_int = doublet_potential(x_int, cube, far_ratio=np.inf)[0]
    assert np.isclose(np.sum(phi_d_int), 1.0, rtol=0.0, atol=1e-10)

    # Exterior point
    x_ext = np.array([[2.5, 3.2, 4.0]])
    phi_d_ext = doublet_potential(x_ext, cube, far_ratio=np.inf)[0]
    assert np.isclose(np.sum(phi_d_ext), 0.0, rtol=0.0, atol=1e-10)

    # Panel centroids collocation
    centroids = np.mean(cube, axis=1)
    B, C = influence_matrices(centroids, cube, far_ratio=np.inf)

    # Diagonal self-influence is limit from body side: +1/2
    assert np.allclose(np.diag(C), 0.5, atol=1e-12)

    # Off-diagonal row sum is Cauchy principal value: +1/2
    off_diag_sum = np.sum(C - np.diag(np.diag(C)), axis=1)
    assert np.allclose(off_diag_sum, 0.5, atol=1e-10)

    # Total row sum from interior limit is 1.0
    row_sum = np.sum(C, axis=1)
    assert np.allclose(row_sum, 1.0, atol=1e-10)


def test_gauss_law_sphere():
    """Gauss law on closed sphere: interior sum is 1, exterior sum is 0 to 1e-3."""
    sphere = _make_sphere_mesh(n_lat=16, n_lon=32, r=1.0)

    x_int = np.array([[0.1, -0.2, 0.3]])
    phi_d_int = doublet_potential(x_int, sphere, far_ratio=np.inf)[0]
    assert np.isclose(np.sum(phi_d_int), 1.0, rtol=1e-3, atol=1e-3)

    x_ext = np.array([[2.0, 3.0, 4.0]])
    phi_d_ext = doublet_potential(x_ext, sphere, far_ratio=np.inf)[0]
    assert np.isclose(np.sum(phi_d_ext), 0.0, atol=1e-3)

    centroids = np.mean(sphere, axis=1)
    B, C = influence_matrices(centroids, sphere, far_ratio=np.inf)
    assert np.allclose(np.diag(C), 0.5, atol=1e-12)
    assert np.allclose(np.sum(C, axis=1), 1.0, atol=1e-3)


def test_panel_plane_continuity_and_limits(test_panels):
    """Targets in plane outside panel, on edge extensions, and at corners are finite and continuous."""
    p = test_panels["square"]
    diam = _panel_diameter(p)
    p_in = p[None, ...]

    # Points outside panel on the plane, on an edge extension, and near a corner
    delta = 1e-4 * diam
    targets_on_plane = np.array([
        [1.0 + delta, 1.0 + delta, 0.0],  # Outside near corner
        [2.0, -1.0, 0.0],                 # Edge extension
        [3.0, 2.0, 0.0],                  # Plane outside panel
    ])

    eps = 1e-9 * diam
    targets_off_plane = targets_on_plane + np.array([[0.0, 0.0, eps]])

    phi_s_on = source_potential(targets_on_plane, p_in, far_ratio=np.inf)[:, 0]
    phi_s_off = source_potential(targets_off_plane, p_in, far_ratio=np.inf)[:, 0]

    assert np.all(np.isfinite(phi_s_on))
    assert np.all(np.isfinite(phi_s_off))
    assert np.allclose(phi_s_on, phi_s_off, atol=1e-6)

    phi_d_on = doublet_potential(targets_on_plane, p_in, far_ratio=np.inf)[:, 0]
    phi_d_off = doublet_potential(targets_off_plane, p_in, far_ratio=np.inf)[:, 0]

    assert np.all(np.isfinite(phi_d_on))
    assert np.all(np.isfinite(phi_d_off))
    assert np.allclose(phi_d_on, phi_d_off, atol=1e-6)

    # Exact corner: source potential is finite and continuous
    corner_on = np.array([[1.0, 1.0, 0.0]])
    corner_off = np.array([[1.0, 1.0, eps]])
    phi_s_corner_on = source_potential(corner_on, p_in, far_ratio=np.inf)[0, 0]
    phi_s_corner_off = source_potential(corner_off, p_in, far_ratio=np.inf)[0, 0]
    assert np.isfinite(phi_s_corner_on)
    assert np.isclose(phi_s_corner_on, phi_s_corner_off, atol=1e-6)


def test_mirror_symmetry(test_panels):
    """Mirroring panel and target about y = 0 preserves potential to 1e-14."""
    for name, p in test_panels.items():
        diam = _panel_diameter(p)
        centroid = np.mean(p, axis=0)
        target = centroid + np.array([0.5, 0.8, 1.2]) * diam

        # Reflection across y = 0: reverse vertex ordering to preserve outward normal
        p_refl = (p * np.array([1.0, -1.0, 1.0]))[[0, 3, 2, 1]]
        target_refl = target * np.array([1.0, -1.0, 1.0])

        phi_s = source_potential(target[None, :], p[None, ...], far_ratio=np.inf)[0, 0]
        phi_s_refl = source_potential(target_refl[None, :], p_refl[None, ...], far_ratio=np.inf)[0, 0]
        assert np.isclose(phi_s, phi_s_refl, rtol=0.0, atol=1e-14), f"mirror phi_s failed for {name}"

        phi_d = doublet_potential(target[None, :], p[None, ...], far_ratio=np.inf)[0, 0]
        phi_d_refl = doublet_potential(target_refl[None, :], p_refl[None, ...], far_ratio=np.inf)[0, 0]
        assert np.isclose(phi_d, phi_d_refl, rtol=0.0, atol=1e-14), f"mirror phi_d failed for {name}"


def test_float32_inputs(test_panels):
    """float32 inputs preserve dtype and match float64 to 1e-5 relative."""
    p_f64 = test_panels["skewed"]
    diam = _panel_diameter(p_f64)
    centroid = np.mean(p_f64, axis=0)
    targets_f64 = np.array([
        centroid + np.array([0.5, 0.8, 1.5]) * diam,
        centroid + np.array([-1.2, 0.4, 2.0]) * diam,
    ])

    p_f32 = p_f64.astype(np.float32)
    targets_f32 = targets_f64.astype(np.float32)

    phi_s_32 = source_potential(targets_f32, p_f32[None, ...], far_ratio=np.inf)
    phi_s_64 = source_potential(targets_f64, p_f64[None, ...], far_ratio=np.inf)
    assert phi_s_32.dtype == np.float32
    assert np.allclose(phi_s_32, phi_s_64, rtol=1e-5)

    v_s_32 = source_velocity(targets_f32, p_f32[None, ...], far_ratio=np.inf)
    v_s_64 = source_velocity(targets_f64, p_f64[None, ...], far_ratio=np.inf)
    assert v_s_32.dtype == np.float32
    assert np.allclose(v_s_32, v_s_64, rtol=1e-5)

    phi_d_32 = doublet_potential(targets_f32, p_f32[None, ...], far_ratio=np.inf)
    phi_d_64 = doublet_potential(targets_f64, p_f64[None, ...], far_ratio=np.inf)
    assert phi_d_32.dtype == np.float32
    assert np.allclose(phi_d_32, phi_d_64, rtol=1e-5)

    v_d_32 = doublet_velocity(targets_f32, p_f32[None, ...], far_ratio=np.inf)
    v_d_64 = doublet_velocity(targets_f64, p_f64[None, ...], far_ratio=np.inf)
    assert v_d_32.dtype == np.float32
    assert np.allclose(v_d_32, v_d_64, rtol=1e-5)


def test_influence_matrices_interface(test_panels):
    """influence_matrices returns (B, C) matching individual potential calls."""
    panels = np.array([test_panels["square"], test_panels["triangle"]])
    collocation = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.5, 2.0],
        [-0.5, 0.2, -1.0],
    ])

    B, C = influence_matrices(collocation, panels, far_ratio=5.0)
    assert B.shape == (3, 2)
    assert C.shape == (3, 2)

    B_ref = source_potential(collocation, panels, far_ratio=5.0)
    C_ref = doublet_potential(collocation, panels, far_ratio=5.0)
    assert np.array_equal(B, B_ref)
    assert np.array_equal(C, C_ref)


def test_empty_inputs():
    """Empty target or panel arrays return empty outputs with correct shape and dtype."""
    empty_pts = np.empty((0, 3), dtype=np.float64)
    empty_pans = np.empty((0, 4, 3), dtype=np.float64)
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    pans = np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]], dtype=np.float64)

    assert source_potential(empty_pts, pans).shape == (0, 1)
    assert source_potential(pts, empty_pans).shape == (1, 0)
    assert doublet_velocity(empty_pts, pans).shape == (0, 1, 3)
    assert doublet_velocity(pts, empty_pans).shape == (1, 0, 3)


def _rotation(seed: int) -> np.ndarray:
    """Return a random proper rotation matrix (det +1)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0.0:
        q[:, 0] = -q[:, 0]
    return q


def _unique_corners(panel: np.ndarray) -> np.ndarray:
    """Return the corners of a panel without the repeated triangle corner."""
    if np.linalg.norm(panel[3] - panel[2]) < 1e-12:
        return panel[:3]
    return panel


def test_far_field_rule_at_25_diameters(test_panels):
    """At 25 diameters the point formulas agree with the exact ones to 1e-3 relative.

    Potentials: relative error per target. Velocities: error norm over the
    norm of the exact vector. No absolute floor.
    """
    dirs = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
        np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8]),
    ]
    for name, p in test_panels.items():
        diam = _panel_diameter(p)
        targets = np.array([_panel_centroid(p) + 25.0 * diam * u for u in dirs])
        p_in = p[None, ...]
        for func in (source_potential, doublet_potential):
            exact = func(targets, p_in, far_ratio=np.inf)[:, 0]
            far = func(targets, p_in, far_ratio=5.0)[:, 0]
            mask = np.abs(exact) > 1e-14
            err = np.abs(far[mask] - exact[mask]) / np.abs(exact[mask])
            assert np.all(err <= 1e-3), f"{func.__name__} far rule failed for {name}: {err.max():.2e}"
        for func in (source_velocity, doublet_velocity):
            exact = func(targets, p_in, far_ratio=np.inf)[:, 0, :]
            far = func(targets, p_in, far_ratio=5.0)[:, 0, :]
            err = np.linalg.norm(far - exact, axis=1) / np.linalg.norm(exact, axis=1)
            assert np.all(err <= 1e-3), f"{func.__name__} far rule failed for {name}: {err.max():.2e}"


def test_on_plane_limits_inside_edge_corner_outside(test_panels):
    """In-plane targets get the body-side limits of the doublet potential.

    Inside 1/2, on an edge 1/4, at a corner theta / (4 pi) with the
    interior angle theta, outside 0. The same values hold for a rotated
    copy far from the origin and for float32 inputs, and they agree with
    a target 1e-9 diameters on the body side.
    """
    rot = _rotation(5)
    offset = np.array([310.0, -125.0, 47.0])
    for name, p in test_panels.items():
        corners = _unique_corners(p)
        n_c = corners.shape[0]
        diam = _panel_diameter(p)
        targets = [_panel_centroid(p)]
        expected = [0.5]
        for k in range(n_c):
            a = corners[k]
            b = corners[(k + 1) % n_c]
            prev = corners[k - 1]
            targets.append(0.5 * (a + b))
            expected.append(0.25)
            u = prev - a
            v = b - a
            theta = np.arccos(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
            targets.append(a.copy())
            expected.append(theta / (4.0 * np.pi))
            targets.append(b + 0.5 * (b - a))
            expected.append(0.0)
        targets = np.array(targets)
        expected = np.array(expected)
        n_hat = np.cross(p[2] - p[0], p[3] - p[1])
        n_hat /= np.linalg.norm(n_hat)
        for pts, pans, label in (
            (targets, p[None, ...], "base"),
            (targets @ rot.T + offset, (p @ rot.T + offset)[None, ...], "rotated"),
            (targets.astype(np.float32), p[None, ...].astype(np.float32), "float32"),
        ):
            got = doublet_potential(pts, pans, far_ratio=np.inf)[:, 0]
            assert np.allclose(got, expected, rtol=0.0, atol=1e-6), f"{name} {label}: {got} vs {expected}"
        body = targets - 1e-9 * diam * n_hat
        got_body = doublet_potential(body, p[None, ...], far_ratio=np.inf)[:, 0]
        assert np.allclose(got_body, expected, rtol=0.0, atol=1e-6), f"{name} body side: {got_body}"
        assert np.all(np.isfinite(source_potential(targets, p[None, ...], far_ratio=np.inf)))
        assert np.all(np.isfinite(source_velocity(targets, p[None, ...], far_ratio=np.inf)))
        assert np.all(np.isfinite(doublet_velocity(targets, p[None, ...], far_ratio=np.inf)))


def test_self_terms_relative_plane_band():
    """Self terms stay +1/2 for rotated panels far from the origin.

    The collocation points come from the unrotated cube and are rotated and
    moved on their own, so they carry round-off of the order of the
    coordinate magnitude, far above 1e-14 m. The on-plane band is relative
    to the panel size and the coordinate magnitude, so no self term flips
    to the fluid-side value -1/2. The Gauss law row sums (1 for each
    collocation point of the closed cube) hold as well. The float32 copy
    uses collocation points rounded to float32 on their own.
    """
    cube = _make_cube_mesh(n=4)
    centroids = np.mean(cube, axis=1)
    for seed, offset in ((1, np.array([1.0e3, -7.0e2, 4.0e2])), (2, np.array([-3.0e4, 2.0e4, 1.0e4]))):
        rot = _rotation(seed)
        panels = cube @ rot.T + offset
        colloc = (centroids @ rot.T) + offset
        _, C = influence_matrices(colloc, panels, far_ratio=np.inf)
        assert np.allclose(np.diag(C), 0.5, rtol=0.0, atol=1e-12), f"self terms flipped at offset {offset}"
        assert np.allclose(np.sum(C, axis=1), 1.0, rtol=0.0, atol=1e-6)
    rot = _rotation(3)
    panels32 = (cube @ rot.T + np.array([12.0, -5.0, 3.0])).astype(np.float32)
    colloc32 = ((centroids @ rot.T) + np.array([12.0, -5.0, 3.0])).astype(np.float32)
    _, C32 = influence_matrices(colloc32, panels32, far_ratio=np.inf)
    assert C32.dtype == np.float32
    assert np.allclose(np.diag(C32), 0.5, rtol=0.0, atol=1e-6)
    assert np.allclose(np.sum(C32.astype(np.float64), axis=1), 1.0, rtol=0.0, atol=1e-4)


def test_velocities_on_edge_extensions_of_rotated_panels(test_panels):
    """In-plane targets on edge extensions of rotated panels: exact velocities.

    The doublet velocity (vortex ring) on the line of an edge, outside the
    edge, must have no spurious term from that segment. The test compares
    both velocities with Gauss-Legendre quadrature and with the numerical
    gradient of the potentials, for rotated and moved panels.
    """
    rot = _rotation(9)
    offset = np.array([2.5, -1.0, 0.75])
    for name, p in test_panels.items():
        corners = _unique_corners(p)
        n_c = corners.shape[0]
        targets = []
        for k in range(n_c):
            a = corners[k]
            b = corners[(k + 1) % n_c]
            targets.append(b + 0.6 * (b - a))
            targets.append(a - 0.45 * (b - a))
        targets = np.array(targets) @ rot.T + offset
        panel = p @ rot.T + offset
        p_in = panel[None, ...]
        _, v_s_ref, _, v_d_ref = _gauss_legendre_quad(panel, targets, order=60)
        v_s = source_velocity(targets, p_in, far_ratio=np.inf)[:, 0, :]
        v_d = doublet_velocity(targets, p_in, far_ratio=np.inf)[:, 0, :]
        scale_s = np.max(np.linalg.norm(v_s_ref, axis=1))
        scale_d = np.max(np.linalg.norm(v_d_ref, axis=1))
        assert np.max(np.abs(v_s - v_s_ref)) <= 1e-7 * scale_s, f"source velocity on extension, {name}"
        assert np.max(np.abs(v_d - v_d_ref)) <= 1e-7 * scale_d, f"doublet velocity on extension, {name}"
        h = 1e-6 * _panel_diameter(p)
        grad = np.zeros_like(v_d)
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = h
            up = doublet_potential(targets + step, p_in, far_ratio=np.inf)[:, 0]
            dn = doublet_potential(targets - step, p_in, far_ratio=np.inf)[:, 0]
            grad[:, axis] = (up - dn) / (2.0 * h)
        assert np.max(np.abs(v_d - grad)) <= 1e-5 * scale_d, f"doublet gradient on extension, {name}"


def test_doublet_velocity_on_edge_is_finite():
    """A target on a panel edge gives a finite doublet velocity (singular segment dropped)."""
    p = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    rot = _rotation(4)
    panel = p @ rot.T
    on_edge = np.array([[1.0, 0.0, 0.0], [2.0, 0.5, 0.0], [0.0, 0.0, 0.0]]) @ rot.T
    v = doublet_velocity(on_edge, panel, far_ratio=np.inf)
    assert np.all(np.isfinite(v))
    assert np.max(np.abs(v)) < 1.0

