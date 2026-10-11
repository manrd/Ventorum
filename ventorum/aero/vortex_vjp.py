# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Vector-Jacobian products of the bent-horseshoe kernel (numpy reference).

The adjoint gradients of :mod:`ventorum.sensitivity` need, for given adjoint
weights on the induced velocity, the derivatives of the cored bent-horseshoe
kernel with respect to the target points, the corner points of the sources,
the wake directions, the core radii and the circulation. This module gives
these vector-Jacobian products (VJP) in numpy. It is the reference of the
compiled VJP kernels (Numba, Cython, NVIDIA Warp): the tests require all of
them to agree with it and with the PyTorch reference copy
(:mod:`ventorum.sensitivity.reference`) to round-off.

Kernel
------
The kernel is the one of the reference copy: the bound segment ``a -> b``,
the chordwise legs ``a_te -> a`` and ``b -> b_te``, and the semi-infinite
wake rays from ``b_te`` and ``a_te`` along ``d``, each with the core term
``rc^2`` (Katz and Plotkin 2001, section 10.4; the core as in
:mod:`ventorum.aero.vortex`). The segment uses the stable form of the
projection where the target is not over the segment, the ray the stable
form of ``1 + cos(t)`` behind its start point. Pairs whose target lies on a
source segment by the mesh topology (the own panel of a load point, the
edge line of a leg mid-point) are zero, with zero derivatives.

The pair core is ``rc2 = rc2_src[j] + tg_rc2[i]`` for a target and a source
of different core groups (``use_tg``), else ``rc2_src[j]``.

The derivatives are derived by hand in reverse mode from the kernel formula
(Griewank and Walther 2008, chapter 3: the adjoint of each elementary
operation in reverse order). The derivation is in the theory manual
(``docs/theory/gradients.md``).

Products
--------
Each source j has a weight ``s[j]`` (sign times circulation) and a column
``col[j]`` (its unknown). Two products use the kernel:

* ``"normal"`` (mode 0): ``A[i, col[j]] += s[j] X[i] . v_ij`` (the system
  rows; X are the normals).
* ``"induced"`` (mode 1): ``V[i] += s[j] v_ij`` (the induced velocity).

The VJP takes the adjoint weights on the output: ``W`` (shape (m, n_u)) for
the system rows, or ``X`` (shape (m, 3)) for the induced velocity with
``W`` None. The adjoint of the pair velocity is ``u_ij = s[j] W[i, col[j]]
X[i]`` (``W`` taken as 1 when it is None).

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, sections 2.11 and 10.4.
* A. Griewank and A. Walther, "Evaluating Derivatives: Principles and
  Techniques of Algorithmic Differentiation", 2nd ed., SIAM, 2008.
* J. R. R. A. Martins and J. T. Hwang, "Review and unification of methods
  for computing derivatives of multidisciplinary computational models",
  AIAA Journal 51(11), 2013, pp. 2582-2599.
"""

from __future__ import annotations

import math

import numpy as np

_INV_4PI = 1.0 / (4.0 * math.pi)
_INV_2PI = 1.0 / (2.0 * math.pi)

# Floors of the squared norms and of the denominators, per precision. The
# float64 floor is the one of the reference copy; float32 uses a value that
# is representable (1e-300 is zero in float32).
FLOORS = {"float64": 1e-300, "float32": 1e-30}

# Row blocks of the compiled VJP kernels (Numba, Cython). Each block adds its
# source derivatives into its own buffer, and the buffers are added in block
# order. The partition depends only on the row count (not on the thread
# count), so the result is bitwise the same for every thread count. The cap
# limits the buffer memory (blocks x sources x 17 values) of large lattices.
VJP_BLOCK_ROWS = 32
VJP_MAX_BLOCKS = 128


def row_blocks(m: int) -> int:
    """Return the number of row blocks of a compiled VJP call.

    Parameters
    ----------
    m : int
        Number of rows (target points) of the call.

    Returns
    -------
    int
        ``ceil(m / VJP_BLOCK_ROWS)``, at least 1 and at most
        ``VJP_MAX_BLOCKS``. It does not depend on the thread count.
    """
    return max(1, min(-(-int(m) // VJP_BLOCK_ROWS), VJP_MAX_BLOCKS))


def row_block_edges(m: int) -> np.ndarray:
    """Return the row edges of the blocks of :func:`row_blocks`.

    Block ``t`` holds the rows ``(t * m) // nb`` to ``((t + 1) * m) // nb``,
    the same partition as the compiled kernels.

    Parameters
    ----------
    m : int
        Number of rows of the call.

    Returns
    -------
    numpy.ndarray
        Integer edges, shape (nb + 1,).
    """
    nb = row_blocks(m)
    return np.array([(t * int(m)) // nb for t in range(nb + 1)], dtype=np.int64)


# Target number of values per (rows, sources) temporary of one chunk.
_CHUNK_VALUES = 400_000


def floor_of(dtype) -> float:
    """Return the floor of the squared norms and denominators for *dtype*."""
    return FLOORS["float32" if np.dtype(dtype) == np.float32 else "float64"]


def _dot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u[..., 0] * v[..., 0] + u[..., 1] * v[..., 1] + u[..., 2] * v[..., 2]


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack((u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1],
                     u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2],
                     u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]), axis=-1)


def _seg_forward(P, A, B, rc2, floor):
    """Return the state of the finite segments A -> B at P (shapes (r, 1, 3), (1, n, 3), (r, n))."""
    r1 = P - A
    r2 = P - B
    r0 = B - A
    c = _cross(r1, r2)
    cr2 = _dot(c, c)
    r0sq = _dot(r0, r0)
    n1 = np.sqrt(_dot(r1, r1) + floor)
    n2 = np.sqrt(_dot(r2, r2) + floor)
    t1 = r1 / n1[..., None]
    t2 = r2 / n2[..., None]
    dot12 = _dot(r1, r2)
    n12 = n1 * n2
    out = dot12 > 0.0
    den_out = np.where(out, n12 * (n12 + dot12), 1.0)
    proj_out = cr2 * (n1 + n2) / den_out
    proj_in = _dot(r0, t1 - t2)
    proj = np.where(out, proj_out, proj_in)
    denom = cr2 + rc2 * r0sq
    ok = denom > floor
    safe = np.where(ok, denom, 1.0)
    k = np.where(ok, _INV_4PI * proj / safe, 0.0)
    v = c * k[..., None]
    return v, (r1, r2, r0, c, cr2, r0sq, n1, n2, t1, t2, dot12, n12, out, den_out, proj_out,
               rc2, ok, safe, k)


def _seg_backward(u, st):
    """Return (gP, gA, gB, grc2) of a segment for the adjoint *u* of its velocity."""
    (r1, r2, r0, c, cr2, r0sq, n1, n2, t1, t2, dot12, n12, out, den_out, proj_out,
     rc2, ok, safe, k) = st
    gc = k[..., None] * u
    gk = _dot(u, c)
    gproj = np.where(ok, gk * _INV_4PI / safe, 0.0)
    gden = np.where(ok, -gk * k / safe, 0.0)
    gcr2 = gden.copy()
    grc2 = gden * r0sq
    gr0sq = gden * rc2
    gpo = np.where(out, gproj, 0.0)
    gpi = np.where(out, 0.0, gproj)
    # Stable form: proj = cr2 (n1 + n2) / D, D = n12 (n12 + dot12).
    gcr2 = gcr2 + gpo * (n1 + n2) / den_out
    gsum = gpo * cr2 / den_out
    gD = -gpo * proj_out / den_out
    gn12 = gD * (2.0 * n12 + dot12)
    gdot12 = gD * n12
    gn1 = gsum + gn12 * n2
    gn2 = gsum + gn12 * n1
    gr1 = gdot12[..., None] * r2
    gr2 = gdot12[..., None] * r1
    # Direct form: proj = r0 . (r1 / n1 - r2 / n2).
    gr0 = gpi[..., None] * (t1 - t2)
    gt1 = gpi[..., None] * r0
    gr1 = gr1 + gt1 / n1[..., None]
    gr2 = gr2 - gt1 / n2[..., None]
    gn1 = gn1 - _dot(gt1, t1) / n1
    gn2 = gn2 + _dot(gt1, t2) / n2
    # Norms, cross product and |r0|^2.
    gr1 = gr1 + (gn1 / n1)[..., None] * r1
    gr2 = gr2 + (gn2 / n2)[..., None] * r2
    gc = gc + (2.0 * gcr2)[..., None] * c
    gr1 = gr1 + _cross(r2, gc)
    gr2 = gr2 + _cross(gc, r1)
    gr0 = gr0 + (2.0 * gr0sq)[..., None] * r0
    return gr1 + gr2, -gr1 - gr0, gr0 - gr2, grc2


def _ray_forward(P, A, d, rc2, floor):
    """Return the state of the semi-infinite lines from A along d at P."""
    r = P - A
    c = _cross(d, r)
    cr2 = _dot(c, c)
    rn = np.sqrt(_dot(r, r) + floor)
    dr = _dot(d, r)
    behind = dr < 0.0
    den_b = np.where(behind, rn * (rn - dr), 1.0)
    oc_b = cr2 / den_b
    oc = np.where(behind, oc_b, 1.0 + dr / rn)
    denom = cr2 + rc2
    ok = denom > floor
    safe = np.where(ok, denom, 1.0)
    k = np.where(ok, _INV_4PI * oc / safe, 0.0)
    v = c * k[..., None]
    return v, (r, d, c, cr2, rn, dr, behind, den_b, oc_b, ok, safe, k)


def _ray_backward(u, st):
    """Return (gP, gA, gd, grc2) of a ray for the adjoint *u* of its velocity."""
    r, d, c, cr2, rn, dr, behind, den_b, oc_b, ok, safe, k = st
    gc = k[..., None] * u
    gk = _dot(u, c)
    goc = np.where(ok, gk * _INV_4PI / safe, 0.0)
    gden = np.where(ok, -gk * k / safe, 0.0)
    gcr2 = gden.copy()
    grc2 = gden
    gb = np.where(behind, goc, 0.0)
    gf = np.where(behind, 0.0, goc)
    # Behind: oc = cr2 / E, E = rn (rn - dr).
    gcr2 = gcr2 + gb / den_b
    gE = -gb * oc_b / den_b
    grn = gE * (2.0 * rn - dr)
    gdr = -gE * rn
    # Ahead: oc = 1 + dr / rn.
    gdr = gdr + gf / rn
    grn = grn - gf * dr / (rn * rn)
    gr = (grn / rn)[..., None] * r + gdr[..., None] * d
    gd = gdr[..., None] * r
    gc = gc + (2.0 * gcr2)[..., None] * c
    gd = gd + _cross(r, gc)
    gr = gr + _cross(gc, d)
    return gr, -gr, gd, grc2


def _chunk(m: int, n: int) -> int:
    return max(1, min(m, _CHUNK_VALUES // max(1, n)))


def _pair_core(rc2_src, tg_rc2, src_group, tg_group, i0, i1, use_tg):
    """Return the pair core (rows, n) and the mask of the cross-group pairs (or None)."""
    rc2 = np.broadcast_to(rc2_src[None, :], (i1 - i0, rc2_src.shape[0]))
    if not use_tg:
        return rc2, None
    other = tg_group[i0:i1, None] != src_group[None, :]
    return rc2 + np.where(other, tg_rc2[i0:i1, None], 0.0), other


def _keep(own, src_panel, edge, src_edge_a, src_edge_b, i0, i1):
    """Return the keep masks of the bound segment and of the two legs (rows, n)."""
    keep_b = own[i0:i1, None] != src_panel[None, :]
    keep_la = edge[i0:i1, None] != src_edge_a[None, :]
    keep_lb = edge[i0:i1, None] != src_edge_b[None, :]
    return keep_b, keep_la, keep_lb


def _horseshoe_rows(P, a, b, a_te, b_te, d, rc2, keeps, floor):
    """Return the pair velocity (rows, n, 3) and the states of the five filaments."""
    Pr = P[:, None, :]
    A, B, AT, BT, D = a[None], b[None], a_te[None], b_te[None], d[None]
    keep_b, keep_la, keep_lb = keeps
    v_b, s_b = _seg_forward(Pr, A, B, rc2, floor)
    v_la, s_la = _seg_forward(Pr, AT, A, rc2, floor)
    v_lb, s_lb = _seg_forward(Pr, B, BT, rc2, floor)
    v_rb, s_rb = _ray_forward(Pr, BT, D, rc2, floor)
    v_ra, s_ra = _ray_forward(Pr, AT, D, rc2, floor)
    v_b = np.where(keep_b[..., None], v_b, 0.0)
    v_la = np.where(keep_la[..., None], v_la, 0.0)
    v_lb = np.where(keep_lb[..., None], v_lb, 0.0)
    v = (v_b + v_la + v_lb) + v_rb - v_ra
    return v, (s_b, s_la, s_lb, s_rb, s_ra)


def horseshoe_forward(P, X, a, b, a_te, b_te, d, rc2_src, tg_rc2, src_group, tg_group,
                      own, src_panel, edge, src_edge_a, src_edge_b, s, col, n_u, mode, use_tg):
    """Return the system rows (mode 0) or the induced velocity (mode 1) of the horseshoes.

    Parameters
    ----------
    P : numpy.ndarray
        Target points [m], shape (m, 3).
    X : numpy.ndarray
        Target normals, shape (m, 3) (mode 0; not used in mode 1).
    a, b, a_te, b_te : numpy.ndarray
        Source corners [m], shape (n, 3).
    d : numpy.ndarray
        Unit wake directions of the sources, shape (n, 3).
    rc2_src : numpy.ndarray
        Squared source core radii [m^2], shape (n,).
    tg_rc2 : numpy.ndarray
        Squared cross-group target core radii [m^2], shape (m,).
    src_group, tg_group : numpy.ndarray
        Core groups of the sources (n,) and of the targets (m,).
    own, edge : numpy.ndarray
        Per target the panel whose bound segment holds it, and the edge
        whose line holds it, or -1 (shape (m,)).
    src_panel, src_edge_a, src_edge_b : numpy.ndarray
        Topology ids of the sources (panel, edge of the a leg, edge of the
        b leg; -2 for images), shape (n,).
    s : numpy.ndarray
        Weight of each source (sign, or sign times circulation [m^2/s]).
    col : numpy.ndarray
        Unknown of each source, shape (n,).
    n_u : int
        Number of unknowns (columns of the rows of mode 0).
    mode : int
        0 for the system rows, 1 for the induced velocity.
    use_tg : bool
        Apply the cross-group target core.

    Returns
    -------
    numpy.ndarray
        Mode 0: rows (m, n_u) [1/m times the weight]. Mode 1: velocity
        (m, 3) [m/s for a circulation weight].
    """
    floor = floor_of(P.dtype)
    m, n = P.shape[0], a.shape[0]
    out = np.zeros((m, int(n_u)) if mode == 0 else (m, 3), dtype=P.dtype)
    step = _chunk(m, n)
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        rc2, _ = _pair_core(rc2_src, tg_rc2, src_group, tg_group, i0, i1, use_tg)
        keeps = _keep(own, src_panel, edge, src_edge_a, src_edge_b, i0, i1)
        v, _ = _horseshoe_rows(P[i0:i1], a, b, a_te, b_te, d, rc2, keeps, floor)
        if mode == 0:
            a_src = _dot(v, X[i0:i1, None, :]) * s[None, :]
            acc = np.zeros((i1 - i0, int(n_u)), dtype=P.dtype)
            # The sources of one unknown are added in source order.
            np.add.at(acc.T, col, a_src.T)
            out[i0:i1] = acc
        else:
            out[i0:i1] = (v * s[None, :, None]).sum(axis=1)
    return out


def horseshoe_vjp(P, X, a, b, a_te, b_te, d, rc2_src, tg_rc2, src_group, tg_group,
                  own, src_panel, edge, src_edge_a, src_edge_b, s, col, W, use_tg):
    """Return the vector-Jacobian product of the horseshoe products.

    The arguments are those of :func:`horseshoe_forward`, with *W* the
    adjoint weights of the system rows (shape (m, n_u)) or None for the
    induced velocity (then *X* holds the adjoint of the induced velocity).
    The adjoint of the pair velocity is ``u_ij = s[j] W[i, col[j]] X[i]``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(gP, gX, g_tg_rc2, ga, gb, ga_te, gb_te, gd, g_rc2_src, gs)``:
        the derivatives with respect to the target points (m, 3), the
        target vectors X (m, 3), the squared target cores (m,), the source
        corners and wake directions (n, 3), the squared source cores (n,)
        and the source weights (n,).
    """
    floor = floor_of(P.dtype)
    m, n = P.shape[0], a.shape[0]
    dt = P.dtype
    gP = np.zeros((m, 3), dtype=dt)
    gX = np.zeros((m, 3), dtype=dt)
    gtg = np.zeros(m, dtype=dt)
    ga = np.zeros((n, 3), dtype=dt)
    gb = np.zeros((n, 3), dtype=dt)
    gat = np.zeros((n, 3), dtype=dt)
    gbt = np.zeros((n, 3), dtype=dt)
    gd = np.zeros((n, 3), dtype=dt)
    grc = np.zeros(n, dtype=dt)
    gs = np.zeros(n, dtype=dt)
    step = _chunk(m, n)
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        rc2, other = _pair_core(rc2_src, tg_rc2, src_group, tg_group, i0, i1, use_tg)
        keeps = _keep(own, src_panel, edge, src_edge_a, src_edge_b, i0, i1)
        v, (s_b, s_la, s_lb, s_rb, s_ra) = _horseshoe_rows(P[i0:i1], a, b, a_te, b_te, d, rc2,
                                                          keeps, floor)
        wf = W[i0:i1][:, col] if W is not None else np.ones((i1 - i0, n), dtype=dt)
        cij = wf * s[None, :]
        Xr = X[i0:i1, None, :]
        u = cij[..., None] * Xr
        gX[i0:i1] = (cij[..., None] * v).sum(axis=1)
        gs += (wf * _dot(Xr, v)).sum(axis=0)
        keep_b, keep_la, keep_lb = keeps
        gP_b, gA_b, gB_b, gr_b = _seg_backward(u * keep_b[..., None], s_b)
        gP_la, gA_la, gB_la, gr_la = _seg_backward(u * keep_la[..., None], s_la)
        gP_lb, gA_lb, gB_lb, gr_lb = _seg_backward(u * keep_lb[..., None], s_lb)
        gP_rb, gA_rb, gd_rb, gr_rb = _ray_backward(u, s_rb)
        gP_ra, gA_ra, gd_ra, gr_ra = _ray_backward(-u, s_ra)
        gP[i0:i1] = (gP_b + gP_la + gP_lb + gP_rb + gP_ra).sum(axis=1)
        ga += (gA_b + gB_la).sum(axis=0)
        gb += (gB_b + gA_lb).sum(axis=0)
        gat += (gA_la + gA_ra).sum(axis=0)
        gbt += (gB_lb + gA_rb).sum(axis=0)
        gd += (gd_rb + gd_ra).sum(axis=0)
        grc2 = gr_b + gr_la + gr_lb + gr_rb + gr_ra
        grc += grc2.sum(axis=0)
        if other is not None:
            gtg[i0:i1] = np.where(other, grc2, 0.0).sum(axis=1)
    return gP, gX, gtg, ga, gb, gat, gbt, gd, grc, gs


def trefftz_forward(q, qn, p, gamma, d, rc2, tg_rc2, src_group, tg_group, use_tg):
    """Return the normal wash [m/s] of 2-D point vortices at Trefftz-plane points.

    ``w[i] = sum_j gamma[j] / (2 pi) ((d x r_ij) . qn[i]) / (|r_ij|^2 + c2_ij)``
    with ``r_ij = q[i] - p[j]`` and the core ``c2_ij = rc2[j] (+ tg_rc2[i]``
    for vortices of another core group).

    Parameters
    ----------
    q, qn : numpy.ndarray
        Points [m] and their unit normals, shape (m, 3).
    p : numpy.ndarray
        Vortex positions [m], shape (n, 3).
    gamma : numpy.ndarray
        Vortex circulations [m^2/s], shape (n,).
    d : numpy.ndarray
        Unit normal of the Trefftz plane, shape (3,).
    rc2, tg_rc2 : numpy.ndarray
        Squared vortex cores (n,) and cross-group point cores (m,) [m^2].
    src_group, tg_group : numpy.ndarray
        Core groups of the vortices (n,) and of the points (m,).
    use_tg : bool
        Apply the cross-group core.

    Returns
    -------
    numpy.ndarray
        Normal wash, shape (m,).
    """
    m, n = q.shape[0], p.shape[0]
    out = np.zeros(m, dtype=q.dtype)
    step = _chunk(m, n)
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        c2, _ = _pair_core(rc2, tg_rc2, src_group, tg_group, i0, i1, use_tg)
        r = q[i0:i1, None, :] - p[None, :, :]
        v = _cross(np.broadcast_to(d, r.shape), r)
        k = _INV_2PI * gamma[None, :] / (_dot(r, r) + c2)
        out[i0:i1] = (_dot(v, qn[i0:i1, None, :]) * k).sum(axis=1)
    return out


def trefftz_vjp(q, qn, p, gamma, d, rc2, tg_rc2, src_group, tg_group, use_tg, gw):
    """Return the vector-Jacobian product of :func:`trefftz_forward` for the adjoint *gw* (m,).

    Returns
    -------
    tuple of numpy.ndarray
        ``(gq, gqn, g_tg_rc2, gp, ggamma, g_rc2, gd)`` with the shapes of
        q, qn, tg_rc2, p, gamma, rc2 and d.
    """
    m, n = q.shape[0], p.shape[0]
    dt = q.dtype
    gq = np.zeros((m, 3), dtype=dt)
    gqn = np.zeros((m, 3), dtype=dt)
    gtg = np.zeros(m, dtype=dt)
    gp = np.zeros((n, 3), dtype=dt)
    gg = np.zeros(n, dtype=dt)
    grc = np.zeros(n, dtype=dt)
    gd = np.zeros(3, dtype=dt)
    step = _chunk(m, n)
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        c2, other = _pair_core(rc2, tg_rc2, src_group, tg_group, i0, i1, use_tg)
        r = q[i0:i1, None, :] - p[None, :, :]
        dd = np.broadcast_to(d, r.shape)
        v = _cross(dd, r)
        D = _dot(r, r) + c2
        k = _INV_2PI * gamma[None, :] / D
        qq = qn[i0:i1, None, :]
        e = _dot(v, qq)
        g = gw[i0:i1, None]
        ge = g * k
        gk = g * e
        gg += (gk * _INV_2PI / D).sum(axis=0)
        gD = -gk * k / D
        grc += gD.sum(axis=0)
        if other is not None:
            gtg[i0:i1] = np.where(other, gD, 0.0).sum(axis=1)
        gqn[i0:i1] = (ge[..., None] * v).sum(axis=1)
        gv = ge[..., None] * qq
        gr = (2.0 * gD)[..., None] * r + _cross(gv, dd)
        gd += _cross(r, gv).sum(axis=(0, 1))
        gq[i0:i1] = gr.sum(axis=1)
        gp -= gr.sum(axis=0)
    return gq, gqn, gtg, gp, gg, grc, gd
