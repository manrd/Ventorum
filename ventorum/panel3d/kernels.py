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
  passing through the corner average with normal n_raw.
* Collocation point: the corner average of the projected corners (the average
  of the three corners for a triangle). For a general quadrilateral this point
  is not the area centroid.
* Source potential: Unit-strength source potential in the far field is -1 / (4 pi r).
* Doublet potential: Unit-strength doublet with axis along the panel unit normal.
  The potential approaching the panel centroid from the fluid side tends to -1/2,
  and from the body side tends to +1/2.
* Targets on the panel plane: a target is on the plane when its distance from
  the plane is at most a tolerance relative to the panel size and to the
  coordinate magnitude (32 times the machine epsilon of the input dtype times
  the sum of the panel diameter and the distance of the panel centroid from
  the origin). The doublet potential there is the limit from the body side:
  +1/2 inside the panel, 0 outside the panel, +1/4 on an edge and
  theta / (4 pi) at a corner with the interior angle theta [rad]. The
  self-influence at the own centroid is therefore +1/2, consistent with the
  internal Dirichlet boundary condition.
* Velocity: the kernels return the gradient of the potential, V = +grad(Phi),
  in m/s per unit singularity strength. This is the convention of the solver
  and its post-processing (the perturbation velocity is +grad of the
  perturbation potential).
* Singular target positions: a target on a panel edge or corner gives an
  infinite velocity. The kernels drop the singular edge term there (the
  vortex segment or the edge logarithm of that edge) and keep the finite rest.
* Far field: when the distance of the target from the area centroid of the
  panel is more than far_ratio times the panel diameter (default 5.0), a
  multipole expansion about the area centroid replaces the exact integrals.
  It keeps the area and the second moments of area of the panel, so the
  relative error of one pair is O((D / r)^3), and O((D / r)^4) for a
  parallelogram (Jackson 1999, section 4.1). The collocation point stays
  at the corner average. Setting far_ratio=np.inf disables the far-field
  switch and uses exact integration everywhere.
* Pair kernels: a half mesh with symmetry about y = 0 (and the ground images)
  solves systems of the form ``C_a + sign * C_b``, where ``b`` is the mirror
  (or image) of the panel set ``a`` and ``sign`` is +1 (symmetric) or -1
  (antisymmetric). :func:`doublet_pair_matrix` returns this sum without the
  two separate matrices, and :func:`source_pair_rhs` returns the source
  right-hand sides ``(B_a + sign * B_b) @ sigma`` without the source matrix.
  :func:`pair_system` returns both from one pass over the targets.
* Precision: the kernels accept float64 (and integer) inputs only, and the
  Warp kernels float64 arithmetic only (a float32 output dtype after float64
  arithmetic is allowed). The on-plane band follows the input precision: with
  float32 inputs it is about 1e-5 of the panel scale, so near a long panel
  (a 50 m wake panel: about 0.3 mm) real off-plane targets count as on-plane
  and their influence becomes zero. The kernels raise a ``ValueError`` for
  float32 inputs and float32 arithmetic. The developer context manager
  :func:`unsafe_float32_kernels` keeps the float32 behaviour runnable for the
  investigation of this band (see ``docs/design/performance_architecture.md``).

References
----------
* J. L. Hess and A. M. O. Smith, "Calculation of non-lifting potential flow
  about arbitrary three-dimensional bodies", Journal of Ship Research 8 (1964)
  22-44.
* J. L. Hess and A. M. O. Smith, "Calculation of potential flow about arbitrary
  bodies", Progress in Aeronautical Sciences 8 (1967) 1-138, Pergamon Press.
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge University
  Press, 2001, section 10.4 and chapter 12.
* J. N. Newman, "Distributions of sources and normal dipoles over a quadrilateral
  panel", Journal of Engineering Mathematics 20 (1986) 113-126.
* A. van Oosterom and J. Strackee, "The solid angle of a plane triangle", IEEE
  Transactions on Biomedical Engineering 30 (1983) 125-126.
* J. D. Jackson, "Classical Electrodynamics", 3rd ed., John Wiley & Sons,
  1999, section 4.1 (Cartesian multipole expansion).
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

__all__ = [
    "source_potential",
    "doublet_potential",
    "source_velocity",
    "doublet_velocity",
    "influence_matrices",
    "influence_matrices_backend",
    "doublet_pair_matrix",
    "source_rhs",
    "source_pair_rhs",
    "pair_system",
]

_INV_4PI = 1.0 / (4.0 * np.pi)
_CHUNK_BUDGET = 2_000_000
#: On-plane band: this factor times the input machine epsilon times the
#: panel scale (diameter plus distance of the centroid from the origin).
_PLANE_TOL_FACTOR = 32.0
#: Edge length [-] (fraction of the panel diameter) below which an edge
#: counts as degenerate (the closing edge of a triangle).
_DEGENERATE_EDGE = 1e-12


#: Environment variable of the developer path for float32 kernels. The path
#: needs this variable set to "1" and the context manager
#: :func:`unsafe_float32_kernels`; either one alone does not open it.
UNSAFE_FLOAT32_ENV = "VENTORUM_DEV_UNSAFE_FLOAT32_PANEL_KERNELS"

_unsafe_float32 = contextvars.ContextVar("ventorum_unsafe_float32_panel_kernels", default=False)

_FLOAT32_REFUSAL = (
    "The 3D panel kernels refuse {what}. The on-plane tolerance of the kernels "
    "follows the precision, and near long panels (for example the wake panels) "
    "a float32 tolerance sets real influences to zero. Pass float64 arrays and "
    "use float64 kernel arithmetic."
)


@contextlib.contextmanager
def unsafe_float32_kernels(reason: str) -> Iterator[None]:
    """Run the panel kernels with float32 inputs or arithmetic (developer path).

    The float32 kernels give wrong results: near a long panel the float32
    on-plane band sets real influences to zero (a 50 m wake panel zeroes the
    doublet potential at targets within about 0.3 mm of its plane). The
    kernels refuse float32 for this reason. This context manager keeps the
    float32 behaviour runnable for the investigation of that band; it is not
    a user setting.

    It opens only when the environment variable ``UNSAFE_FLOAT32_ENV``
    (``VENTORUM_DEV_UNSAFE_FLOAT32_PANEL_KERNELS``) is ``"1"``, and it warns on
    every entry. It holds for the calling thread and context only.

    Parameters
    ----------
    reason : str
        Why the float32 behaviour is needed (written to the warning). An empty
        reason is refused.

    Raises
    ------
    RuntimeError
        If the environment variable is not set to "1".
    ValueError
        If *reason* is empty.

    Warns
    -----
    RuntimeWarning
        On every entry, with the reason.
    """
    if os.environ.get(UNSAFE_FLOAT32_ENV) != "1":
        raise RuntimeError(
            f"The unsafe float32 panel kernels need the environment variable {UNSAFE_FLOAT32_ENV}=1."
        )
    if not str(reason).strip():
        raise ValueError("unsafe_float32_kernels needs a reason.")
    warnings.warn(
        f"Unsafe float32 panel kernels enabled ({reason}). Influences near long panels are wrong.",
        RuntimeWarning,
        stacklevel=3,
    )
    token = _unsafe_float32.set(True)
    try:
        yield
    finally:
        _unsafe_float32.reset(token)


def _require_float64(what: str) -> None:
    """Raise ValueError for a float32 use outside the developer path."""
    if not _unsafe_float32.get():
        raise ValueError(_FLOAT32_REFUSAL.format(what=what))


def _input_eps(*arrays) -> float:
    """Return the largest machine epsilon of the floating dtypes of the inputs.

    Integer and other non-floating inputs count as float64. A floating input
    coarser than float64 is refused (see the module notes on precision),
    except inside :func:`unsafe_float32_kernels`.

    Parameters
    ----------
    *arrays : array_like
        Input arrays (target points, panel corners) [m].

    Returns
    -------
    float
        Machine epsilon [-] of the coarsest floating input dtype, at least
        the float64 epsilon.

    Raises
    ------
    ValueError
        If an input has a floating dtype coarser than float64.
    """
    f64 = float(np.finfo(np.float64).eps)
    eps = f64
    for a in arrays:
        dt = np.asarray(a).dtype
        if np.issubdtype(dt, np.floating):
            eps = max(eps, float(np.finfo(dt).eps))
    if eps > f64:
        _require_float64("float32 (or coarser) input arrays")
    return eps


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
    edge_active: np.ndarray     # (M, 4) bool, False for a degenerate edge
    plane_tol: np.ndarray       # (M,) in m, on-plane band half width
    far_centre: np.ndarray      # (M, 3) in m, area centroid (centre of the far-field expansion)
    q_xx: np.ndarray            # (M,) in m^4, second moment of area about far_centre, x x
    q_xy: np.ndarray            # (M,) in m^4, second moment of area about far_centre, x y
    q_yy: np.ndarray            # (M,) in m^4, second moment of area about far_centre, y y


def _precompute_geometry(panels: np.ndarray, eps: float | None = None) -> _PanelGeometry:
    """Precompute frame, projection, and edge parameters for panels.

    Parameters
    ----------
    panels : numpy.ndarray
        Array of corner coordinates with shape (M, 4, 3) in m.
    eps : float or None
        Machine epsilon [-] of the input data. It sets the on-plane band
        ``plane_tol``. None uses the epsilon of the dtype of *panels*.

    Returns
    -------
    _PanelGeometry
        Precomputed geometric parameters.
    """
    if eps is None:
        eps = _input_eps(panels)
    p1 = panels[:, 0, :]
    p2 = panels[:, 1, :]
    p3 = panels[:, 2, :]
    p4 = panels[:, 3, :]

    # Panel diameter: maximum distance between corners
    diam_sq = np.zeros(panels.shape[0], dtype=panels.dtype)
    for i in range(4):
        for j in range(i + 1, 4):
            d_sq = np.sum((panels[:, i, :] - panels[:, j, :]) ** 2, axis=-1)
            diam_sq = np.maximum(diam_sq, d_sq)
    diameter = np.sqrt(diam_sq)
    tiny = _DEGENERATE_EDGE * diameter

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

    # Collocation point: corner average (average of 3 corners for a triangle)
    is_tri = np.linalg.norm(p4 - p3, axis=-1) <= tiny
    centroid = np.where(is_tri[:, None], (p1_p + p2_p + p3_p) / 3.0, 0.25 * (p1_p + p2_p + p3_p + p4_p))

    # In-plane orthonormal axes
    v12 = p2_p - p1_p
    len12 = np.linalg.norm(v12, axis=-1, keepdims=True)
    # Fallback to p3 - p1 if side 1-2 has zero length
    v13 = p3_p - p1_p
    len13 = np.linalg.norm(v13, axis=-1, keepdims=True)
    use_13 = len12 <= tiny[:, None]
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
    edge_active = edge_len > tiny[:, None]
    area = 0.5 * n_norm[:, 0]
    plane_tol = _PLANE_TOL_FACTOR * float(eps) * (np.linalg.norm(centroid, axis=-1) + diameter)
    far_centre, q_xx, q_xy, q_yy = _area_moments(x_loc, y_loc, centroid, l_axis, m_axis)

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
        edge_active=edge_active,
        plane_tol=plane_tol,
        far_centre=far_centre,
        q_xx=q_xx,
        q_xy=q_xy,
        q_yy=q_yy,
    )


def _area_moments(x_loc, y_loc, centroid, l_axis, m_axis):
    """Return the area centroid and the second moments of area of the panels.

    Green's theorem in the plane turns the integrals of 1, x, y, x^2, x y
    and y^2 over the projected panel into sums over its four edges (the
    polygon or shoelace formulas). The parallel-axis rule moves the second
    moments to the area centroid. A triangle (fourth corner equal to the
    third) has one edge of zero length, which adds zero.

    Parameters
    ----------
    x_loc, y_loc : numpy.ndarray
        Local corner coordinates with shape (M, 4) in m, relative to the
        corner average *centroid*.
    centroid : numpy.ndarray
        Corner average of the projected corners with shape (M, 3) in m.
    l_axis, m_axis : numpy.ndarray
        Unit in-plane axes with shape (M, 3).

    Returns
    -------
    tuple of numpy.ndarray
        (area centroid with shape (M, 3) in m, q_xx, q_xy, q_yy with shape
        (M,) in m^4). The moments are about the area centroid, in the local
        axes: q_xx is the integral of x^2, q_xy of x y and q_yy of y^2.
    """
    xn = np.roll(x_loc, -1, axis=1)
    yn = np.roll(y_loc, -1, axis=1)
    cross = x_loc * yn - xn * y_loc
    a_poly = 0.5 * np.sum(cross, axis=1)
    inv = 1.0 / np.where(np.abs(a_poly) > 0.0, a_poly, 1.0)
    cx = np.sum((x_loc + xn) * cross, axis=1) * inv / 6.0
    cy = np.sum((y_loc + yn) * cross, axis=1) * inv / 6.0
    i_xx = np.sum((x_loc * x_loc + x_loc * xn + xn * xn) * cross, axis=1) / 12.0
    i_yy = np.sum((y_loc * y_loc + y_loc * yn + yn * yn) * cross, axis=1) / 12.0
    i_xy = np.sum((x_loc * yn + 2.0 * x_loc * y_loc + 2.0 * xn * yn + xn * y_loc) * cross, axis=1) / 24.0
    # The sums carry the sign of the polygon orientation; the moments do not.
    sgn = np.sign(a_poly)
    q_xx = sgn * (i_xx - a_poly * cx * cx)
    q_xy = sgn * (i_xy - a_poly * cx * cy)
    q_yy = sgn * (i_yy - a_poly * cy * cy)
    far_centre = centroid + cx[:, None] * l_axis + cy[:, None] * m_axis
    return far_centre, q_xx, q_xy, q_yy


def _plane_winding(
    x: np.ndarray,
    y: np.ndarray,
    j: np.ndarray,
    geom: _PanelGeometry,
) -> np.ndarray:
    """Return the body-side solid-angle limit of in-plane targets [sr].

    The limit equals the in-plane angle [rad] that the panel subtends at the
    target: 2 pi inside, 0 outside, pi on an edge and the interior angle at
    a corner. An edge that holds the target (within the on-plane band) adds
    nothing.

    Parameters
    ----------
    x, y : numpy.ndarray
        Local in-plane coordinates of the targets with shape (K,) in m.
    j : numpy.ndarray
        Panel index of each target with shape (K,).
    geom : _PanelGeometry
        Precomputed panel geometry.

    Returns
    -------
    numpy.ndarray
        Body-side limit of the solid angle with shape (K,) in steradians.
    """
    tol = geom.plane_tol[j]
    total = np.zeros_like(x)
    for k in range(4):
        kn = (k + 1) % 4
        ax = geom.x_loc[j, k] - x
        ay = geom.y_loc[j, k] - y
        bx = geom.x_loc[j, kn] - x
        by = geom.y_loc[j, kn] - y
        cr = ax * by - ay * bx
        dt = ax * bx + ay * by
        ra = np.hypot(ax, ay)
        rb = np.hypot(bx, by)
        on_edge = ((np.abs(cr) <= tol * geom.edge_len[j, k]) & (dt <= 0.0)) | (ra <= tol) | (rb <= tol)
        use = geom.edge_active[j, k] & ~on_edge
        total += np.where(use, np.arctan2(cr, dt), 0.0)
    return total


def _solid_angle_triangles(
    x_rel: np.ndarray,
    y_rel: np.ndarray,
    z_rel: np.ndarray,
    geom: _PanelGeometry,
) -> np.ndarray:
    """Compute solid angle subtended by quadrilateral panels at target points.

    Uses the van Oosterom and Strackee (1983) formulation for each of the two
    triangles forming the panel. Targets in the on-plane band get the
    body-side limit (see :func:`_plane_winding`).

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

    def tri_angle(k0: int, k1: int, k2: int) -> np.ndarray:
        dx0 = xk[None, :, k0] - x_rel
        dy0 = yk[None, :, k0] - y_rel
        dx1 = xk[None, :, k1] - x_rel
        dy1 = yk[None, :, k1] - y_rel
        dx2 = xk[None, :, k2] - x_rel
        dy2 = yk[None, :, k2] - y_rel

        # Determinant [a, b, c] = -2 * Area_triangle * z_rel
        area2 = (xk[:, k1] - xk[:, k0]) * (yk[:, k2] - yk[:, k0]) - (yk[:, k1] - yk[:, k0]) * (xk[:, k2] - xk[:, k0])
        num = -area2[None, :] * z_rel

        r0, r1, r2 = r[k0], r[k1], r[k2]
        dot01 = dx0 * dx1 + dy0 * dy1 + z_rel**2
        dot02 = dx0 * dx2 + dy0 * dy2 + z_rel**2
        dot12 = dx1 * dx2 + dy1 * dy2 + z_rel**2

        den = r0 * r1 * r2 + dot01 * r2 + dot02 * r1 + dot12 * r0
        return 2.0 * np.arctan2(num, den)

    om1 = tri_angle(0, 1, 2)
    # Triangle 2: corners 0, 2, 3 (no contribution if p4 == p3)
    om2 = np.where(geom.edge_active[None, :, 3], tri_angle(0, 2, 3), 0.0)
    total_om = om1 + om2

    # Targets in the on-plane band: body-side limit
    on_plane = np.abs(z_rel) <= geom.plane_tol[None, :]
    if np.any(on_plane):
        i_idx, j_idx = np.nonzero(on_plane)
        total_om[i_idx, j_idx] = _plane_winding(x_rel[i_idx, j_idx], y_rel[i_idx, j_idx], j_idx, geom)
    return total_om


def _edge_log(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xa: np.ndarray,
    ya: np.ndarray,
    xb: np.ndarray,
    yb: np.ndarray,
    dk: np.ndarray,
) -> np.ndarray:
    """Return the edge logarithm L = ln((ra + rb + d) / (ra + rb - d)) [-].

    The denominator ra + rb - d is evaluated without cancellation:
    ra + rb - d = 2 (ra rb + a.b) / (ra + rb + d), and for a.b < 0 the term
    ra rb + a.b is |a x b|^2 / (ra rb - a.b). Here a and b are the vectors
    from the target to the two edge ends. A target on the edge segment or on
    an edge end gives the singular value; it returns 0 (the finite part).

    Parameters
    ----------
    x, y, z : numpy.ndarray
        Local target coordinates in m.
    xa, ya, xb, yb : numpy.ndarray
        Local coordinates of the edge ends in m.
    dk : numpy.ndarray
        Edge length in m.

    Returns
    -------
    numpy.ndarray
        Edge logarithm [-], broadcast shape of the inputs.
    """
    ax = xa - x
    ay = ya - y
    bx = xb - x
    by = yb - y
    z2 = z * z
    ra = np.sqrt(ax * ax + ay * ay + z2)
    rb = np.sqrt(bx * bx + by * by + z2)
    dot = ax * bx + ay * by + z2
    cr = ax * by - ay * bx
    rr = ra * rb
    cross2 = cr * cr + z2 * dk * dk
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(dot >= 0.0, rr + dot, cross2 / np.maximum(rr - dot, 1e-300))
        good = s > 0.0
        val = 2.0 * np.log(ra + rb + dk) - np.log(2.0 * np.where(good, s, 1.0))
    return np.where(good, val, 0.0)


def _segment_velocity_chunk(
    target_pts: np.ndarray,
    p_start: np.ndarray,
    p_end: np.ndarray,
    tol: np.ndarray,
) -> np.ndarray:
    """Compute Biot-Savart induced velocity from straight vortex segments.

    The form (r1 x r2) (|r1| + |r2|) / (|r1| |r2| (|r1| |r2| + r1.r2)) / (4 pi)
    has no cancellation on the extension of the segment line, where the
    velocity is zero. For r1.r2 < 0 the term |r1| |r2| + r1.r2 is evaluated
    as |r1 x r2|^2 / (|r1| |r2| - r1.r2). A target on the segment (closer
    than *tol* to its line, between the ends) or on an end gives the
    singular value; it returns zero.

    Parameters
    ----------
    target_pts : numpy.ndarray
        Target coordinates with shape (P_chunk, 1, 3) in m.
    p_start, p_end : numpy.ndarray
        Segment endpoints with shape (1, M, 3) in m.
    tol : numpy.ndarray
        Core radius of each segment with shape (1, M) in m.

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

    n1 = np.sqrt(np.sum(r1 * r1, axis=-1))
    n2 = np.sqrt(np.sum(r2 * r2, axis=-1))
    dot = np.sum(r1 * r2, axis=-1)
    nn = n1 * n2
    r0_sq = np.sum(r0 * r0, axis=-1)

    core = (n1 <= tol) | (n2 <= tol) | ((dot < 0.0) & (cr2 <= tol * tol * r0_sq))
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.where(dot >= 0.0, nn + dot, cr2 / np.maximum(nn - dot, 1e-300))
        factor = _INV_4PI * (n1 + n2) / (nn * den)
    factor = np.where(core | ~np.isfinite(factor), 0.0, factor)

    out = np.empty((target_pts.shape[0], p_start.shape[1], 3), dtype=np.float64)
    out[..., 0] = cx * factor
    out[..., 1] = cy * factor
    out[..., 2] = cz * factor
    return out


def _prepare(points: np.ndarray, panels: np.ndarray):
    """Return the output dtype, float64 work arrays and the input epsilon."""
    pts = np.asarray(points)
    pans = np.asarray(panels)
    target_dtype = np.result_type(pts.dtype, pans.dtype)
    eps = _input_eps(pts, pans)
    return target_dtype, pts.astype(np.float64, copy=False), pans.astype(np.float64, copy=False), eps


def _far_mask(pts_chunk: np.ndarray, geom: _PanelGeometry, far_ratio: float) -> np.ndarray:
    """Return the mask of target-panel pairs that use the far-field formulas.

    A pair is far when the distance of the target from the expansion centre
    (the area centroid of the panel) is larger than *far_ratio* times the
    panel diameter. An infinite *far_ratio* gives no far pair.
    """
    if np.isinf(far_ratio):
        return np.zeros((pts_chunk.shape[0], geom.area.shape[0]), dtype=bool)
    r_far = np.linalg.norm(pts_chunk[:, None, :] - geom.far_centre[None, :, :], axis=-1)
    return r_far > (far_ratio * geom.diameter[None, :])


# Far-field multipole expansion about the area centroid p_c of the panel, up
# to the second moments of area Q (in the panel plane). With R = x - p_c,
# a = R^T Q R and q = tr Q, the integral of 1 / |R - s| over the panel is
# S = A / r + (3 a - r^2 q) / (2 r^5) + O(A D^3 / r^4) (the Cartesian
# multipole expansion, J. D. Jackson, "Classical Electrodynamics", 3rd ed.,
# Wiley, 1999, section 4.1). The source potential is -S / (4 pi), the
# doublet potential n . grad(S) / (4 pi), and the velocities are their
# gradients. The relative error of one pair is O((D / r)^3), and
# O((D / r)^4) for a parallelogram. See docs/theory/ventorum_3dp.md.


def _far_frame(pts_chunk: np.ndarray, geom: _PanelGeometry):
    """Return the far-field frame of one chunk of targets.

    Returns
    -------
    tuple of numpy.ndarray
        (R with shape (P, M, 3) in m from the expansion centre, r with shape
        (P, M) in m, local X, Y, Z with shape (P, M) in m).
    """
    R = pts_chunk[:, None, :] - geom.far_centre[None, :, :]
    r = np.linalg.norm(R, axis=-1)
    X = np.sum(R * geom.l_axis[None, :, :], axis=-1)
    Y = np.sum(R * geom.m_axis[None, :, :], axis=-1)
    Z = np.sum(R * geom.normal[None, :, :], axis=-1)
    return R, r, X, Y, Z


def _far_terms(X: np.ndarray, Y: np.ndarray, geom: _PanelGeometry):
    """Return the second-moment scalars a = R^T Q R [m^6] and q = tr Q [m^4]."""
    a = geom.q_xx[None, :] * X * X + 2.0 * geom.q_xy[None, :] * X * Y + geom.q_yy[None, :] * Y * Y
    q = (geom.q_xx + geom.q_yy)[None, :]
    return a, q


def _far_source_potential(pts_chunk: np.ndarray, geom: _PanelGeometry) -> np.ndarray:
    """Return the far-field source potentials [m] with shape (P, M) of one chunk of targets."""
    R, r, X, Y, Z = _far_frame(pts_chunk, geom)
    a, q = _far_terms(X, Y, geom)
    r = np.maximum(r, 1e-300)
    inv_r2 = 1.0 / (r * r)
    t = (1.5 * a * inv_r2 - 0.5 * q) / (r * r * r)
    return -_INV_4PI * (geom.area[None, :] / r + t)


def _far_doublet_potential(pts_chunk: np.ndarray, geom: _PanelGeometry) -> np.ndarray:
    """Return the far-field doublet potentials [-] with shape (P, M) of one chunk of targets."""
    R, r, X, Y, Z = _far_frame(pts_chunk, geom)
    a, q = _far_terms(X, Y, geom)
    r = np.maximum(r, 1e-300)
    inv_r2 = 1.0 / (r * r)
    r3 = r * r * r
    f = (1.5 * q - 7.5 * a * inv_r2) * inv_r2 / r3
    return _INV_4PI * Z * (f - geom.area[None, :] / r3)


def _far_source_velocity(pts_chunk: np.ndarray, geom: _PanelGeometry) -> np.ndarray:
    """Return the far-field source velocities [m/s] with shape (P, M, 3) of one chunk of targets."""
    R, r, X, Y, Z = _far_frame(pts_chunk, geom)
    a, q = _far_terms(X, Y, geom)
    r = np.maximum(r, 1e-300)
    inv_r2 = 1.0 / (r * r)
    r3 = r * r * r
    r5 = r3 * r * r
    qr_x = geom.q_xx[None, :] * X + geom.q_xy[None, :] * Y
    qr_y = geom.q_xy[None, :] * X + geom.q_yy[None, :] * Y
    radial = geom.area[None, :] / r3 - (1.5 * q - 7.5 * a * inv_r2) / r5
    v = radial[..., None] * R - (3.0 / r5)[..., None] * (
        qr_x[..., None] * geom.l_axis[None, :, :] + qr_y[..., None] * geom.m_axis[None, :, :]
    )
    return _INV_4PI * v


def _far_doublet_velocity(pts_chunk: np.ndarray, geom: _PanelGeometry) -> np.ndarray:
    """Return the far-field doublet velocities [m/s] with shape (P, M, 3) of one chunk of targets."""
    R, r, X, Y, Z = _far_frame(pts_chunk, geom)
    a, q = _far_terms(X, Y, geom)
    r = np.maximum(r, 1e-300)
    inv_r2 = 1.0 / (r * r)
    r3 = r * r * r
    r5 = r3 * r * r
    r7 = r5 * r * r
    qr_x = geom.q_xx[None, :] * X + geom.q_xy[None, :] * Y
    qr_y = geom.q_xy[None, :] * X + geom.q_yy[None, :] * Y
    area = geom.area[None, :]
    c_n = (1.5 * q - 7.5 * a * inv_r2) / r5 - area / r3
    c_r = Z * ((52.5 * a * inv_r2 - 7.5 * q) / r7 + 3.0 * area / r5)
    c_q = -15.0 * Z / r7
    v = (
        c_n[..., None] * geom.normal[None, :, :]
        + c_r[..., None] * R
        + c_q[..., None] * (qr_x[..., None] * geom.l_axis[None, :, :] + qr_y[..., None] * geom.m_axis[None, :, :])
    )
    return _INV_4PI * v


def _source_block(pts_chunk: np.ndarray, geom: _PanelGeometry, far_ratio: float) -> np.ndarray:
    """Return the source potentials of one chunk of targets [m].

    Parameters
    ----------
    pts_chunk : numpy.ndarray
        Float64 target coordinates with shape (P_chunk, 3) in m.
    geom : _PanelGeometry
        Precomputed geometry of the M panels.
    far_ratio : float
        Distance-to-diameter ratio of the far-field expansion [-].

    Returns
    -------
    numpy.ndarray
        Source potential per unit source strength with shape (P_chunk, M)
        in m.
    """
    M = geom.area.shape[0]

    # Target relative position vectors: (P_chunk, M, 3)
    R_vec = pts_chunk[:, None, :] - geom.centroid[None, :, :]
    far_mask = _far_mask(pts_chunk, geom, far_ratio)

    # Local coordinates
    x_loc = np.sum(R_vec * geom.l_axis[None, :, :], axis=-1)
    y_loc = np.sum(R_vec * geom.m_axis[None, :, :], axis=-1)
    z_loc = np.sum(R_vec * geom.normal[None, :, :], axis=-1)

    # Near-field exact integration
    poly_om = _solid_angle_triangles(x_loc, y_loc, z_loc, geom)

    sum_edge = np.zeros((pts_chunk.shape[0], M), dtype=np.float64)
    for k in range(4):
        active = geom.edge_active[None, :, k]
        kn = (k + 1) % 4
        xk = geom.x_loc[None, :, k]
        yk = geom.y_loc[None, :, k]
        dx = geom.dx[None, :, k]
        dy = geom.dy[None, :, k]
        dk = geom.edge_len[None, :, k]

        Lk = _edge_log(x_loc, y_loc, z_loc, xk, yk, geom.x_loc[None, :, kn], geom.y_loc[None, :, kn], dk)
        r_perp = ((x_loc - xk) * dy - (y_loc - yk) * dx) / np.maximum(dk, 1e-300)
        sum_edge += np.where(active, r_perp * Lk, 0.0)

    # Exact formula: (I - z * Omega) / (4 pi)
    phi_exact = _INV_4PI * (sum_edge - z_loc * poly_om)

    if not far_mask.any():
        return phi_exact
    # Near pairs (also the own collocation point, r = 0) are evaluated and then
    # dropped by the mask; their divisions by zero are harmless.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        phi_far = _far_source_potential(pts_chunk, geom)
    return np.where(far_mask, phi_far, phi_exact)


def _doublet_block(pts_chunk: np.ndarray, geom: _PanelGeometry, far_ratio: float) -> np.ndarray:
    """Return the doublet potentials of one chunk of targets [-].

    Parameters
    ----------
    pts_chunk : numpy.ndarray
        Float64 target coordinates with shape (P_chunk, 3) in m.
    geom : _PanelGeometry
        Precomputed geometry of the M panels.
    far_ratio : float
        Distance-to-diameter ratio of the far-field expansion [-].

    Returns
    -------
    numpy.ndarray
        Doublet potential per unit doublet strength with shape
        (P_chunk, M) [-].
    """
    R_vec = pts_chunk[:, None, :] - geom.centroid[None, :, :]
    far_mask = _far_mask(pts_chunk, geom, far_ratio)

    x_loc = np.sum(R_vec * geom.l_axis[None, :, :], axis=-1)
    y_loc = np.sum(R_vec * geom.m_axis[None, :, :], axis=-1)
    z_loc = np.sum(R_vec * geom.normal[None, :, :], axis=-1)

    poly_om = _solid_angle_triangles(x_loc, y_loc, z_loc, geom)
    phi_exact = _INV_4PI * poly_om

    if not far_mask.any():
        return phi_exact
    # Near pairs (also the own collocation point, r = 0) are evaluated and then
    # dropped by the mask; their divisions by zero are harmless.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        phi_far = _far_doublet_potential(pts_chunk, geom)
    return np.where(far_mask, phi_far, phi_exact)


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
        Distance-to-diameter ratio beyond which the far-field source expansion is used.
        Setting to numpy.inf forces exact integration everywhere.

    Returns
    -------
    numpy.ndarray
        Source potential with shape (P, M) in m per unit source strength
        [m/s]. Dtype matches input points and panels.
    """
    target_dtype, pts_f64, pans_f64, eps = _prepare(points, panels)

    P = pts_f64.shape[0]
    M = pans_f64.shape[0]
    out = np.empty((P, M), dtype=np.float64)

    if P == 0 or M == 0:
        return out.astype(target_dtype)

    geom = _precompute_geometry(pans_f64, eps)
    chunk_size = max(1, _CHUNK_BUDGET // M)

    for p0 in range(0, P, chunk_size):
        p1 = min(p0 + chunk_size, P)
        out[p0:p1] = _source_block(pts_f64[p0:p1], geom, far_ratio)

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
        Distance-to-diameter ratio beyond which the far-field doublet expansion is used.
        Setting to numpy.inf forces exact integration everywhere.

    Returns
    -------
    numpy.ndarray
        Doublet potential [-] per unit doublet strength with shape (P, M).
        Dtype matches input points and panels. Targets in the on-plane band
        get the body-side limit (+1/2 inside, 0 outside, +1/4 on an edge,
        theta / (4 pi) at a corner with interior angle theta [rad]).
    """
    target_dtype, pts_f64, pans_f64, eps = _prepare(points, panels)

    P = pts_f64.shape[0]
    M = pans_f64.shape[0]
    out = np.empty((P, M), dtype=np.float64)

    if P == 0 or M == 0:
        return out.astype(target_dtype)

    geom = _precompute_geometry(pans_f64, eps)
    chunk_size = max(1, _CHUNK_BUDGET // M)

    for p0 in range(0, P, chunk_size):
        p1 = min(p0 + chunk_size, P)
        out[p0:p1] = _doublet_block(pts_f64[p0:p1], geom, far_ratio)

    return out.astype(target_dtype)


def source_velocity(
    points: np.ndarray,
    panels: np.ndarray,
    far_ratio: float = 5.0,
) -> np.ndarray:
    """Compute induced velocity per unit source strength at target points.

    The velocity is the gradient of the source potential, V = +grad(Phi).
    A unit source with positive strength pushes the fluid away from the
    panel on both sides.

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio beyond which the far-field source expansion is used.
        Setting to numpy.inf forces exact integration everywhere.

    Returns
    -------
    numpy.ndarray
        Induced velocity vectors with shape (P, M, 3) in m/s per unit source
        strength [m/s]. Dtype matches input. Targets in the on-plane band
        get the body-side limit of the normal component.
    """
    target_dtype, pts_f64, pans_f64, eps = _prepare(points, panels)

    P = pts_f64.shape[0]
    M = pans_f64.shape[0]
    out = np.empty((P, M, 3), dtype=np.float64)

    if P == 0 or M == 0:
        return out.astype(target_dtype)

    geom = _precompute_geometry(pans_f64, eps)
    chunk_size = max(1, _CHUNK_BUDGET // M)

    for p0 in range(0, P, chunk_size):
        p1 = min(p0 + chunk_size, P)
        pts_chunk = pts_f64[p0:p1]

        R_vec = pts_chunk[:, None, :] - geom.centroid[None, :, :]
        far_mask = _far_mask(pts_chunk, geom, far_ratio)

        x_loc = np.sum(R_vec * geom.l_axis[None, :, :], axis=-1)
        y_loc = np.sum(R_vec * geom.m_axis[None, :, :], axis=-1)
        z_loc = np.sum(R_vec * geom.normal[None, :, :], axis=-1)

        poly_om = _solid_angle_triangles(x_loc, y_loc, z_loc, geom)

        sum_vx = np.zeros((p1 - p0, M), dtype=np.float64)
        sum_vy = np.zeros((p1 - p0, M), dtype=np.float64)

        for k in range(4):
            active = geom.edge_active[None, :, k]
            kn = (k + 1) % 4
            dk = geom.edge_len[None, :, k]
            Lk = _edge_log(
                x_loc, y_loc, z_loc,
                geom.x_loc[None, :, k], geom.y_loc[None, :, k],
                geom.x_loc[None, :, kn], geom.y_loc[None, :, kn], dk,
            )
            inv_dk = 1.0 / np.maximum(dk, 1e-300)
            sum_vx += np.where(active, geom.dy[None, :, k] * inv_dk * Lk, 0.0)
            sum_vy += np.where(active, -geom.dx[None, :, k] * inv_dk * Lk, 0.0)

        # Local velocity components (V = +grad Phi):
        # vx = dPhi/dx = +_INV_4PI * sum_k (dy/dk) Lk
        # vy = dPhi/dy = -_INV_4PI * sum_k (dx/dk) Lk
        # vz = dPhi/dz = -_INV_4PI * Omega
        vx = _INV_4PI * sum_vx
        vy = _INV_4PI * sum_vy
        vz = -_INV_4PI * poly_om

        # Transform to global coordinates
        v_exact = (
            vx[..., None] * geom.l_axis[None, :, :]
            + vy[..., None] * geom.m_axis[None, :, :]
            + vz[..., None] * geom.normal[None, :, :]
        )

        if far_mask.any():
            # Near pairs (also the own collocation point, r = 0) are evaluated and then
            # dropped by the mask; their divisions by zero are harmless.
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                v_far = _far_source_velocity(pts_chunk, geom)
            v_exact = np.where(far_mask[..., None], v_far, v_exact)
        out[p0:p1] = v_exact

    return out.astype(target_dtype)


def doublet_velocity(
    points: np.ndarray,
    panels: np.ndarray,
    far_ratio: float = 5.0,
) -> np.ndarray:
    """Compute induced velocity per unit doublet strength at target points.

    The velocity is the gradient of the doublet potential, V = +grad(Phi).
    It equals the velocity of the equivalent vortex ring of unit
    circulation along the panel perimeter, in the corner order.

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio beyond which the far-field doublet expansion is used.
        Setting to numpy.inf forces exact integration everywhere.

    Returns
    -------
    numpy.ndarray
        Induced velocity vectors with shape (P, M, 3) in m/s per unit doublet
        strength [m^2/s]. Dtype matches input. A target on a panel edge
        gets the finite part (the singular segment is dropped).
    """
    target_dtype, pts_f64, pans_f64, eps = _prepare(points, panels)

    P = pts_f64.shape[0]
    M = pans_f64.shape[0]
    out = np.empty((P, M, 3), dtype=np.float64)

    if P == 0 or M == 0:
        return out.astype(target_dtype)

    geom = _precompute_geometry(pans_f64, eps)
    chunk_size = max(1, _CHUNK_BUDGET // M)

    for p0 in range(0, P, chunk_size):
        p1 = min(p0 + chunk_size, P)
        pts_chunk = pts_f64[p0:p1]

        far_mask = _far_mask(pts_chunk, geom, far_ratio)

        # Equivalent vortex ring along 4 perimeter segments
        v_ring = np.zeros((p1 - p0, M, 3), dtype=np.float64)
        for k in range(4):
            active = geom.edge_active[None, :, k]
            p_start = geom.corners_proj[None, :, k, :]
            p_end = geom.corners_proj[None, :, (k + 1) % 4, :]
            v_seg = _segment_velocity_chunk(pts_chunk[:, None, :], p_start, p_end, geom.plane_tol[None, :])
            v_ring += np.where(active[..., None], v_seg, 0.0)

        # V_D = +grad Phi_D = v_ring
        v_exact = v_ring

        if far_mask.any():
            # Near pairs (also the own collocation point, r = 0) are evaluated and then
            # dropped by the mask; their divisions by zero are harmless.
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                v_far = _far_doublet_velocity(pts_chunk, geom)
            v_exact = np.where(far_mask[..., None], v_far, v_exact)
        out[p0:p1] = v_exact

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
        Distance-to-diameter ratio of the far-field transition.

    Returns
    -------
    tuple of numpy.ndarray
        (B, C) where B has shape (P, M) for source potentials and C has shape
        (P, M) for doublet potentials.
    """
    B = source_potential(collocation, panels, far_ratio=far_ratio)
    C = doublet_potential(collocation, panels, far_ratio=far_ratio)
    return B, C


#: Largest number of right-hand-side columns of the source kernels [-]
#: (the six unit onsets u, v, w, p, q, r).
MAX_RHS_COLUMNS = 6


def _pair_inputs(points, panels_a, panels_b, sigma=None):
    """Check and convert the inputs of the pair kernels.

    Parameters
    ----------
    points : array_like
        Target coordinates with shape (P, 3) in m.
    panels_a : array_like
        First panel set with shape (M, 4, 3) in m.
    panels_b : array_like or None
        Second panel set with shape (M, 4, 3) in m, or None.
    sigma : array_like or None, optional
        Source strengths with shape (M, K) or (M,) in m/s, or None.

    Returns
    -------
    tuple
        (target dtype, float64 points, float64 panels_a, float64 panels_b or
        None, float64 sigma with shape (M, K) or None, True when *sigma* was
        one-dimensional, the right-hand-side dtype, input epsilon [-]).
    """
    pts = np.asarray(points)
    pa = np.asarray(panels_a)
    arrays = [pts, pa]
    pb = None
    if panels_b is not None:
        pb = np.asarray(panels_b)
        if pb.shape != pa.shape:
            raise ValueError(
                "The two panel sets of a pair kernel must have the same shape, "
                f"got {pa.shape} and {pb.shape}."
            )
        arrays.append(pb)
    target_dtype = np.result_type(*[a.dtype for a in arrays])
    eps = _input_eps(*arrays)
    sig = None
    flat = False
    rhs_dtype = target_dtype
    if sigma is not None:
        raw = np.asarray(sigma)
        rhs_dtype = np.result_type(target_dtype, raw.dtype)
        sig = raw.astype(np.float64, copy=False)
        flat = sig.ndim == 1
        if flat:
            sig = sig.reshape(-1, 1)
        if sig.ndim != 2 or sig.shape[0] != pa.shape[0]:
            raise ValueError(
                f"sigma must have shape (M, K) with M = {pa.shape[0]} panels, got {raw.shape}."
            )
        if sig.shape[1] > MAX_RHS_COLUMNS:
            raise ValueError(f"sigma may have at most {MAX_RHS_COLUMNS} columns, got {sig.shape[1]}.")
    return (
        target_dtype,
        pts.astype(np.float64, copy=False),
        pa.astype(np.float64, copy=False),
        None if pb is None else pb.astype(np.float64, copy=False),
        sig,
        flat,
        rhs_dtype,
        eps,
    )


def _pair_reference(points, panels_a, panels_b, sign, sigma, far_ratio, want_c, want_rhs):
    """Return the doublet pair matrix and the source pair right-hand sides.

    One loop over chunks of targets evaluates both panel sets for the same
    targets. No full source matrix is kept: each chunk of
    ``B_a + sign * B_b`` is multiplied by *sigma* and dropped.

    Returns
    -------
    tuple
        (C_a + sign * C_b with shape (P, M) or None, (B_a + sign * B_b) @
        sigma with shape (P, K) or (P,) or None, the target dtype, the
        right-hand-side dtype).
    """
    target_dtype, pts, pa, pb, sig, flat, rhs_dtype, eps = _pair_inputs(
        points, panels_a, panels_b, sigma if want_rhs else None
    )
    sign = float(sign)
    P = pts.shape[0]
    M = pa.shape[0]
    mat = np.empty((P, M), dtype=np.float64) if want_c else None
    rhs = np.zeros((P, sig.shape[1]), dtype=np.float64) if want_rhs else None
    if P > 0 and M > 0:
        geom_a = _precompute_geometry(pa, eps)
        geom_b = None if pb is None else _precompute_geometry(pb, eps)
        chunk_size = max(1, _CHUNK_BUDGET // M)
        for p0 in range(0, P, chunk_size):
            p1 = min(p0 + chunk_size, P)
            chunk = pts[p0:p1]
            if want_c:
                block = _doublet_block(chunk, geom_a, far_ratio)
                if geom_b is not None:
                    block = block + sign * _doublet_block(chunk, geom_b, far_ratio)
                mat[p0:p1] = block
            if want_rhs:
                block = _source_block(chunk, geom_a, far_ratio)
                if geom_b is not None:
                    block = block + sign * _source_block(chunk, geom_b, far_ratio)
                rhs[p0:p1] = block @ sig
    if rhs is not None and flat:
        rhs = rhs.reshape(-1)
    return mat, rhs, target_dtype, rhs_dtype


def doublet_pair_matrix(
    points: np.ndarray,
    panels_a: np.ndarray,
    panels_b: np.ndarray | None,
    sign: float = 1.0,
    far_ratio: float = 5.0,
) -> np.ndarray:
    """Compute the doublet potentials of a panel set and its mirror in one pass.

    The result is ``C(points, panels_a) + sign * C(points, panels_b)``,
    without the two separate matrices. On a half mesh with symmetry about
    y = 0, ``panels_b`` is the mirror of ``panels_a`` and ``sign`` is +1
    for the symmetric system and -1 for the antisymmetric system (Katz and
    Plotkin 2001, chapter 12). The ground images use the same form with
    the reflected panels.

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels_a : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    panels_b : numpy.ndarray or None
        Second panel set with shape (M, 4, 3) in m. None gives
        ``C(points, panels_a)``.
    sign : float, default=1.0
        Factor of the second panel set [-] (+1 or -1).
    far_ratio : float, default=5.0
        Distance-to-diameter ratio beyond which the far-field doublet expansion is used.

    Returns
    -------
    numpy.ndarray
        Doublet potential sum [-] per unit doublet strength with shape
        (P, M). Dtype matches the inputs.
    """
    mat, _, target_dtype, _ = _pair_reference(
        points, panels_a, panels_b, sign, None, far_ratio, want_c=True, want_rhs=False
    )
    return mat.astype(target_dtype, copy=False)


def source_pair_rhs(
    points: np.ndarray,
    panels_a: np.ndarray,
    panels_b: np.ndarray | None,
    sign: float,
    sigma: np.ndarray,
    far_ratio: float = 5.0,
) -> np.ndarray:
    """Compute the source potential products of a panel set and its mirror.

    The result is ``(B(points, panels_a) + sign * B(points, panels_b)) @
    sigma``. The source matrices are never stored: each chunk of target
    rows is multiplied by *sigma* and dropped. The solver uses the result
    as the known part of the internal Dirichlet boundary condition
    (Morino and Kuo 1974; Katz and Plotkin 2001, chapter 12).

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels_a : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    panels_b : numpy.ndarray or None
        Second panel set with shape (M, 4, 3) in m. None uses only
        *panels_a*.
    sign : float
        Factor of the second panel set [-] (+1 or -1).
    sigma : numpy.ndarray
        Source strengths with shape (M, K), K at most 6, or (M,) in m/s.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio beyond which the far-field source expansion is used.

    Returns
    -------
    numpy.ndarray
        Source potentials with shape (P, K), or (P,) for a one-dimensional
        *sigma*, in m^2/s.
    """
    _, rhs, _, rhs_dtype = _pair_reference(
        points, panels_a, panels_b, sign, sigma, far_ratio, want_c=False, want_rhs=True
    )
    return rhs.astype(rhs_dtype, copy=False)


def source_rhs(
    points: np.ndarray,
    panels: np.ndarray,
    sigma: np.ndarray,
    far_ratio: float = 5.0,
) -> np.ndarray:
    """Compute the source potential products ``B(points, panels) @ sigma``.

    The source matrix is never stored (see :func:`source_pair_rhs`).

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    sigma : numpy.ndarray
        Source strengths with shape (M, K), K at most 6, or (M,) in m/s.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio beyond which the far-field source expansion is used.

    Returns
    -------
    numpy.ndarray
        Source potentials with shape (P, K), or (P,) for a one-dimensional
        *sigma*, in m^2/s.
    """
    return source_pair_rhs(points, panels, None, 1.0, sigma, far_ratio=far_ratio)


def pair_system(
    points: np.ndarray,
    panels_a: np.ndarray,
    panels_b: np.ndarray | None,
    sign: float,
    sigma: np.ndarray,
    far_ratio: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the doublet pair matrix and the source pair right-hand sides.

    Returns the results of :func:`doublet_pair_matrix` and
    :func:`source_pair_rhs` for the same inputs from one loop over the
    targets.

    Parameters
    ----------
    points : numpy.ndarray
        Target coordinates with shape (P, 3) in m.
    panels_a : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    panels_b : numpy.ndarray or None
        Second panel set with shape (M, 4, 3) in m, or None.
    sign : float
        Factor of the second panel set [-] (+1 or -1).
    sigma : numpy.ndarray
        Source strengths with shape (M, K), K at most 6, or (M,) in m/s.
    far_ratio : float, default=5.0
        Distance-to-diameter ratio of the far-field expansions.

    Returns
    -------
    tuple of numpy.ndarray
        (C, R): C with shape (P, M) [-] is ``C_a + sign * C_b`` and R with
        shape (P, K) in m^2/s is ``(B_a + sign * B_b) @ sigma``.
    """
    mat, rhs, target_dtype, rhs_dtype = _pair_reference(
        points, panels_a, panels_b, sign, sigma, far_ratio, want_c=True, want_rhs=True
    )
    return mat.astype(target_dtype, copy=False), rhs.astype(rhs_dtype, copy=False)


def influence_matrices_backend(
    collocation: np.ndarray,
    panels: np.ndarray,
    backend: str = "numpy",
    device=None,
    dtype=None,
) -> tuple:
    """Compute aerodynamic influence matrices B and C with a chosen backend.

    The numpy backend is the reference. The numba backend runs the same
    formulas in compiled code on the CPU. The warp backend runs them on a
    CUDA GPU and returns PyTorch tensors on the device.

    Parameters
    ----------
    collocation : numpy.ndarray
        Collocation point coordinates with shape (P, 3) in m.
    panels : numpy.ndarray
        Panel corner coordinates with shape (M, 4, 3) in m.
    backend : str, default="numpy"
        Kernel backend: ``"numpy"``, ``"numba"`` or ``"warp"``.
    device : optional
        CUDA device of the warp backend (for example ``"cuda:0"``).
        None selects the current CUDA device. Other backends ignore it.
    dtype : optional
        Target dtype of the outputs. None keeps the dtype of the inputs
        (numpy and numba) or reads the inputs (warp).

    Returns
    -------
    tuple
        (B, C) where B has shape (P, M) for source potentials and C has shape
        (P, M) for doublet potentials. Numpy arrays for ``"numpy"`` and
        ``"numba"``, PyTorch tensors on the device for ``"warp"``.
    """
    key = str(backend).strip().lower()
    if key == "numpy":
        if dtype is not None:
            collocation = np.asarray(collocation, dtype=dtype)
            panels = np.asarray(panels, dtype=dtype)
        return influence_matrices(collocation, panels)
    if key == "numba":
        from ventorum.panel3d.kernels_numba import influence_matrices_numba

        if dtype is not None:
            collocation = np.asarray(collocation, dtype=dtype)
            panels = np.asarray(panels, dtype=dtype)
        return influence_matrices_numba(collocation, panels)
    if key == "warp":
        from ventorum.panel3d.kernels_warp import influence_matrices_warp

        return influence_matrices_warp(collocation, panels, device=device, dtype=dtype)
    raise ValueError(f"Unknown panel backend {backend!r}; use 'numpy', 'numba' or 'warp'.")
