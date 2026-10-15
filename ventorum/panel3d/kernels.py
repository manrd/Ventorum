# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Influence kernels for three-dimensional constant-strength source and doublet panels.

This module provides the float64 reference implementation of the potential and
velocity kernels for constant-strength planar quadrilateral and triangular panels.
All kernels vectorize over target points and panels. Memory use is bounded by
chunking the target points.

Conventions
-----------
* Geometry: Panels are specified by four corner points in SI units [m] with shape
  (M, 4, 3). The unit normal vector points into the fluid (out of the body),
  defined by the right-hand rule from the cross product of the diagonals:
  n_raw = (p3 - p1) x (p4 - p2).
* Triangular panels: Specified with the fourth corner equal to the third corner.
* Planar projection: Non-planar quadrilaterals are projected onto the mean plane
  passing through the centroid with normal n_raw.
* Source potential: Unit-strength source potential in the far field is -1 / (4 pi r).
* Doublet potential: Unit-strength doublet with axis along the panel unit normal.
  The potential approaching the panel centroid from the fluid side tends to -1/2,
  and from the body side tends to +1/2.
* Targets on the panel plane: The self-influence at the own centroid evaluates to the
  limit from the body (interior) side (+1/2), consistent with the internal Dirichlet
  boundary condition. On the plane outside the panel, the doublet potential is 0.
* Velocity: The fluid velocity is defined as the negative gradient of the potential,
  V = -grad(Phi).
* Far field: Beyond far_ratio times the panel diameter (default 5.0), point source
  and point doublet expansions are used. Setting far_ratio=np.inf disables the
  far-field switch and uses exact integration everywhere.

References
----------
* J. L. Hess and A. M. O. Smith, "Calculation of Nonlifting Potential Flow About
  Arbitrary Three-Dimensional Bodies", Progress in Aeronautical Sciences 8 (1967) 1-138.
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge University
  Press, 2001, section 10.4 and chapter 12.
* J. N. Newman, "Distributions of sources and normal dipoles over a quadrilateral
  panel", Journal of Engineering Mathematics 20 (1986) 113-126.
* A. van Oosterom and J. Strackee, "The solid angle of a plane triangle", IEEE
  Transactions on Biomedical Engineering 30 (1983) 125-126.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "source_potential",
    "doublet_potential",
    "source_velocity",
    "doublet_velocity",
    "influence_matrices",
]

_INV_4PI = 1.0 / (4.0 * np.pi)
_CHUNK_BUDGET = 2_000_000


@dataclass(slots=True)
class _PanelGeometry:
    """Precomputed geometric properties of planar projected panels."""

    centroid: np.ndarray        # (M, 3) in m
    normal: np.ndarray          # (M, 3) unit normal
    l_axis: np.ndarray          # (M, 3) unit in-plane x-axis
    m_axis: np.ndarray          # (M, 3) unit in-plane y-axis
    corners_proj: np.ndarray    # (M, 4, 3) in m
    x_loc: np.ndarray           # (M, 4) in m
    y_loc: np.ndarray           # (M, 4) in m
    dx: np.ndarray              # (M, 4) in m
    dy: np.ndarray              # (M, 4) in m
    edge_len: np.ndarray        # (M, 4) in m
    area: np.ndarray            # (M,) in m^2
    diameter: np.ndarray        # (M,) in m


def _precompute_geometry(panels: np.ndarray) -> _PanelGeometry:
    """Precompute frame, projection, and edge parameters for panels.

    Parameters
    ----------
    panels : numpy.ndarray
        Array of corner coordinates with shape (M, 4, 3) in m.

    Returns
    -------
    _PanelGeometry
        Precomputed geometric parameters.
    """
    p1 = panels[:, 0, :]
    p2 = panels[:, 1, :]
    p3 = panels[:, 2, :]
    p4 = panels[:, 3, :]

    d13 = p3 - p1
    d24 = p4 - p2
    n_raw = np.cross(d13, d24)
    n_norm = np.linalg.norm(n_raw, axis=-1, keepdims=True)
    normal = n_raw / np.maximum(n_norm, 1e-300)

    # Project corners onto the mean plane passing through the corner average
    p_avg = 0.25 * (p1 + p2 + p3 + p4)
    dist_p1 = np.sum((p1 - p_avg) * normal, axis=-1, keepdims=True)
    dist_p2 = np.sum((p2 - p_avg) * normal, axis=-1, keepdims=True)
    dist_p3 = np.sum((p3 - p_avg) * normal, axis=-1, keepdims=True)
    dist_p4 = np.sum((p4 - p_avg) * normal, axis=-1, keepdims=True)

    p1_p = p1 - dist_p1 * normal
    p2_p = p2 - dist_p2 * normal
    p3_p = p3 - dist_p3 * normal
    p4_p = p4 - dist_p4 * normal
    corners_p = np.stack([p1_p, p2_p, p3_p, p4_p], axis=1)

    # Panel centroid: true area centroid (average of 3 vertices for triangle where p4 == p3)
    is_tri = np.linalg.norm(p4 - p3, axis=-1) < 1e-12
    centroid = np.where(is_tri[:, None], (p1_p + p2_p + p3_p) / 3.0, 0.25 * (p1_p + p2_p + p3_p + p4_p))

    # In-plane orthonormal axes
    v12 = p2_p - p1_p
    len12 = np.linalg.norm(v12, axis=-1, keepdims=True)
    # Fallback to p3 - p1 if side 1-2 has zero length
    v13 = p3_p - p1_p
    len13 = np.linalg.norm(v13, axis=-1, keepdims=True)
    use_13 = len12 < 1e-12
    l_raw = np.where(use_13, v13, v12)
    l_norm = np.where(use_13, len13, len12)
    l_axis = l_raw / np.maximum(l_norm, 1e-300)
    m_axis = np.cross(normal, l_axis)

    # Local 2D coordinates of corners relative to centroid
    disp = corners_p - centroid[:, None, :]
    x_loc = np.sum(disp * l_axis[:, None, :], axis=-1)
    y_loc = np.sum(disp * m_axis[:, None, :], axis=-1)

    # Edge vectors in local plane
    dx = np.empty_like(x_loc)
    dy = np.empty_like(y_loc)
    for k in range(4):
        k_next = (k + 1) % 4
        dx[:, k] = x_loc[:, k_next] - x_loc[:, k]
        dy[:, k] = y_loc[:, k_next] - y_loc[:, k]

    edge_len = np.hypot(dx, dy)
    area = 0.5 * n_norm[:, 0]

    # Panel diameter: maximum distance between corners
    diam_sq = np.zeros(panels.shape[0], dtype=panels.dtype)
    for i in range(4):
        for j in range(i + 1, 4):
            d_sq = np.sum((panels[:, i, :] - panels[:, j, :]) ** 2, axis=-1)
            diam_sq = np.maximum(diam_sq, d_sq)
    diameter = np.sqrt(diam_sq)

    return _PanelGeometry(
        centroid=centroid,
        normal=normal,
        l_axis=l_axis,
        m_axis=m_axis,
        corners_proj=corners_p,
        x_loc=x_loc,
        y_loc=y_loc,
        dx=dx,
        dy=dy,
        edge_len=edge_len,
        area=area,
        diameter=diameter,
    )


def _solid_angle_triangles(
    x_rel: np.ndarray,
    y_rel: np.ndarray,
    z_rel: np.ndarray,
    geom: _PanelGeometry,
) -> np.ndarray:
    """Compute solid angle subtended by quadrilateral panels at target points.

    Uses the van Oosterom and Strackee (1983) formulation for each of the two
    triangles forming the panel.

    Parameters
    ----------
    x_rel, y_rel, z_rel : numpy.ndarray
        Local coordinates of target points with shape (P_chunk, M) in m.
    geom : _PanelGeometry
        Precomputed panel geometry.

    Returns
    -------
    numpy.ndarray
        Signed solid angle with shape (P_chunk, M) in steradians.
    """
    xk = geom.x_loc
    yk = geom.y_loc

    # Corner distances: r_k with shape (P_chunk, M)
    r = []
    for k in range(4):
        r.append(np.sqrt((x_rel - xk[None, :, k]) ** 2 + (y_rel - yk[None, :, k]) ** 2 + z_rel**2))

    # Triangle 1: corners 0, 1, 2
    # Dot products: a . b where a = p_a - x, b = p_b - x
    def tri_angle(k0: int, k1: int, k2: int) -> np.ndarray:
        dx0 = xk[None, :, k0] - x_rel
        dy0 = yk[None, :, k0] - y_rel
        dx1 = xk[None, :, k1] - x_rel
        dy1 = yk[None, :, k1] - y_rel
        dx2 = xk[None, :, k2] - x_rel
        dy2 = yk[None, :, k2] - y_rel

        # Determinant [a, b, c] = -2 * Area_triangle * z_rel
        # Area = 0.5 * ((x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0))
        area2 = (xk[:, k1] - xk[:, k0]) * (yk[:, k2] - yk[:, k0]) - (yk[:, k1] - yk[:, k0]) * (xk[:, k2] - xk[:, k0])
        num = -area2[None, :] * z_rel

        r0, r1, r2 = r[k0], r[k1], r[k2]
        dot01 = dx0 * dx1 + dy0 * dy1 + z_rel**2
        dot02 = dx0 * dx2 + dy0 * dy2 + z_rel**2
        dot12 = dx1 * dx2 + dy1 * dy2 + z_rel**2

        den = r0 * r1 * r2 + dot01 * r2 + dot02 * r1 + dot12 * r0
        return 2.0 * np.arctan2(num, den)

    om1 = tri_angle(0, 1, 2)
    # Triangle 2: corners 0, 2, 3 (contributes 0 if p4 == p3)
    has_tri2 = geom.edge_len[:, 3] > 1e-14
    om2 = np.where(has_tri2[None, :], tri_angle(0, 2, 3), 0.0)

    total_om = om1 + om2

    # Handle points exactly on the panel plane (z_rel == 0)
    on_plane = np.abs(z_rel) < 1e-14
    if np.any(on_plane):
        # Point in polygon test via 2D cross products
        # A point is inside if cross = dx_k * (y - y_k) - dy_k * (x - x_k) >= 0 for all edges
        inside = np.ones_like(on_plane)
        for k in range(4):
            cross = geom.dx[None, :, k] * (y_rel - yk[None, :, k]) - geom.dy[None, :, k] * (x_rel - xk[None, :, k])
            active_edge = geom.edge_len[None, :, k] > 1e-14
            inside = inside & ((cross >= -1e-12 * geom.area[None, :]) | (~active_edge))

        # Body side limit: +2 pi if inside, 0 if outside
        body_side_om = np.where(inside, 2.0 * np.pi, 0.0)
        total_om = np.where(on_plane, body_side_om, total_om)

    return total_om


def _segment_velocity_Chunk(
    target_pts: np.ndarray,
    p_start: np.ndarray,
    p_end: np.ndarray,
) -> np.ndarray:
    """Compute Biot-Savart induced velocity from straight vortex segments.

    Parameters
    ----------
    target_pts : numpy.ndarray
        Target coordinates with shape (P_chunk, 1, 3) in m.
    p_start, p_end : numpy.ndarray
        Segment endpoints with shape (1, M, 3) in m.

    Returns
    -------
    numpy.ndarray
        Induced velocity vector with shape (P_chunk, M, 3) in m/s for unit circulation.
    """
    r1 = target_pts - p_start
    r2 = target_pts - p_end
    r0 = p_end - p_start

    # r1 x r2
    cx = r1[..., 1] * r2[..., 2] - r1[..., 2] * r2[..., 1]
    cy = r1[..., 2] * r2[..., 0] - r1[..., 0] * r2[..., 2]
    cz = r1[..., 0] * r2[..., 1] - r1[..., 1] * r2[..., 0]
    cr2 = cx * cx + cy * cy + cz * cz

    d1 = np.sum(r1 * r1, axis=-1)
    d2 = np.sum(r2 * r2, axis=-1)
    n1 = np.sqrt(d1)
    n2 = np.sqrt(d2)

    r0_sq = np.sum(r0 * r0, axis=-1)
    len_r0 = np.sqrt(r0_sq)

    # Project r0 . (r1/n1 - r2/n2)
    inv1 = 1.0 / np.maximum(n1, 1e-300)
    inv2 = 1.0 / np.maximum(n2, 1e-300)
    proj = (
        r0[..., 0] * (r1[..., 0] * inv1 - r2[..., 0] * inv2)
        + r0[..., 1] * (r1[..., 1] * inv1 - r2[..., 1] * inv2)
        + r0[..., 2] * (r1[..., 2] * inv1 - r2[..., 2] * inv2)
    )

    valid = (cr2 > 1e-280) & (len_r0 > 1e-14)
    factor = np.zeros_like(cr2)
    factor[valid] = _INV_4PI * proj[valid] / cr2[valid]

    out = np.empty((target_pts.shape[0], p_start.shape[1], 3), dtype=np.float64)
    out[..., 0] = cx * factor
    out[..., 1] = cy * factor
    out[..., 2] = cz * factor
    return out


def source_potential(
    points: np.ndarray,
    panels: np.ndarray,
    far_ratio: float = 5.0,
) -> np.ndarray:
    """Compute potential per unit source strength at target points.

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio beyond which point source expansion is used.
        Setting to numpy.inf forces exact integration everywhere.

    Returns
    -------
    numpy.ndarray
        Source potential with shape (P, M). Dtype matches input points and panels.
    """
    pts = np.asarray(points)
    pans = np.asarray(panels)
    target_dtype = np.result_type(pts.dtype, pans.dtype)

    pts_f64 = pts.astype(np.float64, copy=False)
    pans_f64 = pans.astype(np.float64, copy=False)

    P = pts_f64.shape[0]
    M = pans_f64.shape[0]
    out = np.empty((P, M), dtype=np.float64)

    if P == 0 or M == 0:
        return out.astype(target_dtype)

    geom = _precompute_geometry(pans_f64)
    chunk_size = max(1, _CHUNK_BUDGET // M)

    for p0 in range(0, P, chunk_size):
        p1 = min(p0 + chunk_size, P)
        pts_chunk = pts_f64[p0:p1]

        # Target relative position vectors: (P_chunk, M, 3)
        R_vec = pts_chunk[:, None, :] - geom.centroid[None, :, :]
        r_dist = np.linalg.norm(R_vec, axis=-1)

        # Far-field selection
        if np.isneginf(far_ratio):
            far_mask = np.zeros((p1 - p0, M), dtype=bool)
        elif np.isposinf(far_ratio):
            far_mask = np.zeros((p1 - p0, M), dtype=bool)
        else:
            far_mask = r_dist > (far_ratio * geom.diameter[None, :])

        # Local coordinates
        x_loc = np.sum(R_vec * geom.l_axis[None, :, :], axis=-1)
        y_loc = np.sum(R_vec * geom.m_axis[None, :, :], axis=-1)
        z_loc = np.sum(R_vec * geom.normal[None, :, :], axis=-1)

        # Near-field exact integration
        poly_om = _solid_angle_triangles(x_loc, y_loc, z_loc, geom)

        sum_edge = np.zeros((p1 - p0, M), dtype=np.float64)
        for k in range(4):
            active = geom.edge_len[None, :, k] > 1e-14
            xk = geom.x_loc[None, :, k]
            yk = geom.y_loc[None, :, k]
            xkp1 = geom.x_loc[None, :, (k + 1) % 4]
            ykp1 = geom.y_loc[None, :, (k + 1) % 4]
            dx = geom.dx[None, :, k]
            dy = geom.dy[None, :, k]
            dk = geom.edge_len[None, :, k]

            rk = np.sqrt((x_loc - xk) ** 2 + (y_loc - yk) ** 2 + z_loc**2)
            rkp1 = np.sqrt((x_loc - xkp1) ** 2 + (y_loc - ykp1) ** 2 + z_loc**2)
            r_perp = ((x_loc - xk) * dy - (y_loc - yk) * dx) / np.maximum(dk, 1e-300)

            num_L = rk + rkp1 + dk
            den_L = rk + rkp1 - dk
            safe_den = den_L > 1e-15
            Lk = np.zeros_like(den_L)
            Lk[safe_den] = np.log(num_L[safe_den] / den_L[safe_den])

            term = r_perp * Lk
            sum_edge += np.where(active, term, 0.0)

        # Exact formula: (I - z * Omega) / (4 pi)
        phi_exact = _INV_4PI * (sum_edge - z_loc * poly_om)

        # Far-field expansion: -A / (4 pi r)
        phi_far = -_INV_4PI * geom.area[None, :] / np.maximum(r_dist, 1e-300)

        out[p0:p1] = np.where(far_mask, phi_far, phi_exact)

    return out.astype(target_dtype)


def doublet_potential(
    points: np.ndarray,
    panels: np.ndarray,
    far_ratio: float = 5.0,
) -> np.ndarray:
    """Compute potential per unit doublet strength at target points.

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio beyond which point doublet expansion is used.
        Setting to numpy.inf forces exact integration everywhere.

    Returns
    -------
    numpy.ndarray
        Doublet potential with shape (P, M). Dtype matches input points and panels.
    """
    pts = np.asarray(points)
    pans = np.asarray(panels)
    target_dtype = np.result_type(pts.dtype, pans.dtype)

    pts_f64 = pts.astype(np.float64, copy=False)
    pans_f64 = pans.astype(np.float64, copy=False)

    P = pts_f64.shape[0]
    M = pans_f64.shape[0]
    out = np.empty((P, M), dtype=np.float64)

    if P == 0 or M == 0:
        return out.astype(target_dtype)

    geom = _precompute_geometry(pans_f64)
    chunk_size = max(1, _CHUNK_BUDGET // M)

    for p0 in range(0, P, chunk_size):
        p1 = min(p0 + chunk_size, P)
        pts_chunk = pts_f64[p0:p1]

        R_vec = pts_chunk[:, None, :] - geom.centroid[None, :, :]
        r_dist = np.linalg.norm(R_vec, axis=-1)

        if np.isneginf(far_ratio) or np.isposinf(far_ratio):
            far_mask = np.zeros((p1 - p0, M), dtype=bool)
        else:
            far_mask = r_dist > (far_ratio * geom.diameter[None, :])

        x_loc = np.sum(R_vec * geom.l_axis[None, :, :], axis=-1)
        y_loc = np.sum(R_vec * geom.m_axis[None, :, :], axis=-1)
        z_loc = np.sum(R_vec * geom.normal[None, :, :], axis=-1)

        poly_om = _solid_angle_triangles(x_loc, y_loc, z_loc, geom)
        phi_exact = _INV_4PI * poly_om

        # Far-field point doublet: -A * (R . n) / (4 pi r^3)
        r_dot_n = np.sum(R_vec * geom.normal[None, :, :], axis=-1)
        r_cube = np.maximum(r_dist**3, 1e-300)
        phi_far = -_INV_4PI * geom.area[None, :] * r_dot_n / r_cube

        out[p0:p1] = np.where(far_mask, phi_far, phi_exact)

    return out.astype(target_dtype)


def source_velocity(
    points: np.ndarray,
    panels: np.ndarray,
    far_ratio: float = 5.0,
) -> np.ndarray:
    """Compute induced velocity per unit source strength at target points.

    Velocity is defined as the negative gradient of potential, V = -grad(Phi).

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio beyond which point source expansion is used.
        Setting to numpy.inf forces exact integration everywhere.

    Returns
    -------
    numpy.ndarray
        Induced velocity vectors with shape (P, M, 3) in m/s. Dtype matches input.
    """
    pts = np.asarray(points)
    pans = np.asarray(panels)
    target_dtype = np.result_type(pts.dtype, pans.dtype)

    pts_f64 = pts.astype(np.float64, copy=False)
    pans_f64 = pans.astype(np.float64, copy=False)

    P = pts_f64.shape[0]
    M = pans_f64.shape[0]
    out = np.empty((P, M, 3), dtype=np.float64)

    if P == 0 or M == 0:
        return out.astype(target_dtype)

    geom = _precompute_geometry(pans_f64)
    chunk_size = max(1, _CHUNK_BUDGET // M)

    for p0 in range(0, P, chunk_size):
        p1 = min(p0 + chunk_size, P)
        pts_chunk = pts_f64[p0:p1]

        R_vec = pts_chunk[:, None, :] - geom.centroid[None, :, :]
        r_dist = np.linalg.norm(R_vec, axis=-1)

        if np.isneginf(far_ratio) or np.isposinf(far_ratio):
            far_mask = np.zeros((p1 - p0, M), dtype=bool)
        else:
            far_mask = r_dist > (far_ratio * geom.diameter[None, :])

        x_loc = np.sum(R_vec * geom.l_axis[None, :, :], axis=-1)
        y_loc = np.sum(R_vec * geom.m_axis[None, :, :], axis=-1)
        z_loc = np.sum(R_vec * geom.normal[None, :, :], axis=-1)

        poly_om = _solid_angle_triangles(x_loc, y_loc, z_loc, geom)

        sum_vx = np.zeros((p1 - p0, M), dtype=np.float64)
        sum_vy = np.zeros((p1 - p0, M), dtype=np.float64)

        for k in range(4):
            active = geom.edge_len[None, :, k] > 1e-14
            xk = geom.x_loc[None, :, k]
            yk = geom.y_loc[None, :, k]
            xkp1 = geom.x_loc[None, :, (k + 1) % 4]
            ykp1 = geom.y_loc[None, :, (k + 1) % 4]
            dx = geom.dx[None, :, k]
            dy = geom.dy[None, :, k]
            dk = geom.edge_len[None, :, k]

            rk = np.sqrt((x_loc - xk) ** 2 + (y_loc - yk) ** 2 + z_loc**2)
            rkp1 = np.sqrt((x_loc - xkp1) ** 2 + (y_loc - ykp1) ** 2 + z_loc**2)

            num_L = rk + rkp1 + dk
            den_L = rk + rkp1 - dk
            safe_den = den_L > 1e-15
            Lk = np.zeros_like(den_L)
            Lk[safe_den] = np.log(num_L[safe_den] / den_L[safe_den])

            sum_vx += np.where(active, -(dy / np.maximum(dk, 1e-300)) * Lk, 0.0)
            sum_vy += np.where(active, +(dx / np.maximum(dk, 1e-300)) * Lk, 0.0)

        # Local velocity components:
        # vx = -dPhi/dx = -_INV_4PI * sum_k (dy/dk)*Lk
        # vy = -dPhi/dy = +_INV_4PI * sum_k (dx/dk)*Lk
        # vz = -dPhi/dz = +_INV_4PI * Omega
        vx = _INV_4PI * sum_vx
        vy = _INV_4PI * sum_vy
        vz = _INV_4PI * poly_om

        # Transform to global coordinates
        v_exact = (
            vx[..., None] * geom.l_axis[None, :, :]
            + vy[..., None] * geom.m_axis[None, :, :]
            + vz[..., None] * geom.normal[None, :, :]
        )

        # Far-field expansion: -A * R / (4 pi r^3)
        r_cube = np.maximum(r_dist**3, 1e-300)
        v_far = -_INV_4PI * geom.area[None, :, None] * R_vec / r_cube[..., None]

        out[p0:p1] = np.where(far_mask[..., None], v_far, v_exact)

    return out.astype(target_dtype)


def doublet_velocity(
    points: np.ndarray,
    panels: np.ndarray,
    far_ratio: float = 5.0,
) -> np.ndarray:
    """Compute induced velocity per unit doublet strength at target points.

    Velocity is defined as the negative gradient of potential, V = -grad(Phi).
    Uses the mathematically equivalent vortex ring along the panel perimeter.

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio beyond which point doublet expansion is used.
        Setting to numpy.inf forces exact integration everywhere.

    Returns
    -------
    numpy.ndarray
        Induced velocity vectors with shape (P, M, 3) in m/s. Dtype matches input.
    """
    pts = np.asarray(points)
    pans = np.asarray(panels)
    target_dtype = np.result_type(pts.dtype, pans.dtype)

    pts_f64 = pts.astype(np.float64, copy=False)
    pans_f64 = pans.astype(np.float64, copy=False)

    P = pts_f64.shape[0]
    M = pans_f64.shape[0]
    out = np.empty((P, M, 3), dtype=np.float64)

    if P == 0 or M == 0:
        return out.astype(target_dtype)

    geom = _precompute_geometry(pans_f64)
    chunk_size = max(1, _CHUNK_BUDGET // M)

    for p0 in range(0, P, chunk_size):
        p1 = min(p0 + chunk_size, P)
        pts_chunk = pts_f64[p0:p1]

        R_vec = pts_chunk[:, None, :] - geom.centroid[None, :, :]
        r_dist = np.linalg.norm(R_vec, axis=-1)

        if np.isneginf(far_ratio) or np.isposinf(far_ratio):
            far_mask = np.zeros((p1 - p0, M), dtype=bool)
        else:
            far_mask = r_dist > (far_ratio * geom.diameter[None, :])

        # Equivalent vortex ring along 4 perimeter segments
        v_ring = np.zeros((p1 - p0, M, 3), dtype=np.float64)
        for k in range(4):
            active = geom.edge_len[None, :, k] > 1e-14
            p_start = geom.corners_proj[None, :, k, :]
            p_end = geom.corners_proj[None, :, (k + 1) % 4, :]
            v_seg = _segment_velocity_Chunk(pts_chunk[:, None, :], p_start, p_end)
            v_ring += np.where(active[..., None], v_seg, 0.0)

        # V_D = -v_ring
        v_exact = -v_ring

        # Far-field expansion: (A / 4 pi) * [n/r^3 - 3 (R . n) R / r^5]
        r_dot_n = np.sum(R_vec * geom.normal[None, :, :], axis=-1, keepdims=True)
        r_cube = np.maximum(r_dist**3, 1e-300)[..., None]
        r_fifth = np.maximum(r_dist**5, 1e-300)[..., None]

        term1 = geom.normal[None, :, :] / r_cube
        term2 = 3.0 * r_dot_n * R_vec / r_fifth
        v_far = _INV_4PI * geom.area[None, :, None] * (term1 - term2)

        out[p0:p1] = np.where(far_mask[..., None], v_far, v_exact)

    return out.astype(target_dtype)


def influence_matrices(
    collocation: np.ndarray,
    panels: np.ndarray,
    far_ratio: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute aerodynamic influence matrices B and C.

    Parameters
    ----------
    collocation : numpy.ndarray
        Collocation point coordinates with shape (P, 3) in m.
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio for point far-field transition.

    Returns
    -------
    tuple of numpy.ndarray
        (B, C) where B has shape (P, M) for source potentials and C has shape
        (P, M) for doublet potentials.
    """
    B = source_potential(collocation, panels, far_ratio=far_ratio)
    C = doublet_potential(collocation, panels, far_ratio=far_ratio)
    return B, C
