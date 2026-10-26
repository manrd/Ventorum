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
wake rays from ``b_te`` and ``a_te`` along ``d`` (Katz and Plotkin 2001,
section 10.4). The core acts on the distance ``D`` from the target to the
finite filament, as in :mod:`ventorum.aero.vortex`. Over a segment the
denominator is ``|r1 x r2|^2 + rc^2 |r0|^2``, with the stable form of the
projection where ``r1 . r2 > 0``. In front of the start point of a ray it
is ``|d x r|^2 + rc^2``. Outside a segment (before ``A`` or after ``B``)
and behind the start point of a ray, the exact Biot-Savart law (in its
form without cancellation) is multiplied by the Scully factor
``D^2 / (D^2 + rc^2)``, with ``D`` the distance to the nearest end point
(Scully 1975; Vatistas, Kozel and Mih 1991, n = 1). ``D^2`` is C1 across
the end points. At exactly collinear outside points the velocity is zero,
but its derivative is the exact-law derivative times the factor. Pairs
whose target lies on a source segment by the mesh topology (the own panel
of a load point, the edge line of a leg mid-point) are zero, with zero
derivatives.

A source can have nodes on its legs (``a_k``, ``b_k``, see
:class:`ventorum.aero.vortex.HorseshoeSet`): its left leg is then
``a_te -> a_k -> a`` and its right leg ``b -> b_k -> b_te``. The forward
segments keep the edge ids of the legs; the aft segments have their own
ids (``src_edge_ka``, ``src_edge_kb``). On a source without a node,
``a_k`` is ``a_te`` and ``b_k`` is ``b_te``: the leg is computed as
without nodes.

A source can also be on a leg chain (``leg_next``, see
:class:`ventorum.aero.vortex.HorseshoeSet`): its own leg segments then end
at ``a_e``, ``b_e`` (the bound points of the next source) instead of the
trailing edge, and its legs continue on the legs of the next source. The
own segments of a source are on the legs of every source before it on its
chain, so in the VJP their adjoint is the sum of the pair adjoints of these
sources (a prefix sum along the chain, the adjoint of the suffix sum of the
forward product).

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
* M. P. Scully, "Computation of helicopter rotor wake geometry and its
  influence on rotor harmonic airloads", ASRL TR 178-1, Massachusetts
  Institute of Technology, 1975.
* G. H. Vatistas, V. Kozel and W. C. Mih, "A simpler model for
  concentrated vortices", Experiments in Fluids 11, 1991, pp. 73-76.
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
    d1 = _dot(r1, r1) + floor
    d2 = _dot(r2, r2) + floor
    n1 = np.sqrt(d1)
    n2 = np.sqrt(d2)
    t1 = r1 / n1[..., None]
    t2 = r2 / n2[..., None]
    dot12 = _dot(r1, r2)
    n12 = n1 * n2
    e = n12 + dot12
    # Over the segment: the stable projection where r1 . r2 > 0, else the
    # direct form.
    os_ = dot12 > 0.0
    den_os = np.where(os_, n12 * e, 1.0)
    proj_os = cr2 * (n1 + n2) / den_os
    proj_in = _dot(r0, t1 - t2)
    proj = np.where(os_, proj_os, proj_in)
    denom = cr2 + rc2 * r0sq
    ok = denom > floor
    safe = np.where(ok, denom, 1.0)
    k_in = np.where(ok, _INV_4PI * proj / safe, 0.0)
    # Outside the segment (before A or after B): exact law times
    # F = D2 / (D2 + rc2), D2 the squared distance to the nearest end point.
    before = _dot(r0, r1) < 0.0
    after = _dot(r0, r2) > 0.0
    outside = before | after
    D2 = np.where(before, d1, d2)
    ok_out = outside & (e > floor)
    den_out = np.where(ok_out, n12 * e, 1.0)
    q = 1.0 / (D2 + rc2)
    F = D2 * q
    k_out = np.where(ok_out, _INV_4PI * (n1 + n2) / den_out * F, 0.0)
    k = np.where(outside, k_out, k_in)
    v = c * k[..., None]
    return v, (r1, r2, r0, c, cr2, r0sq, n1, n2, t1, t2, dot12, n12, e, os_, den_os, proj_os,
               rc2, ok, safe, k_in, k, outside, before, ok_out, den_out, q, F, k_out)


def _seg_backward(u, st):
    """Return (gP, gA, gB, grc2) of a segment for the adjoint *u* of its velocity."""
    (r1, r2, r0, c, cr2, r0sq, n1, n2, t1, t2, dot12, n12, e, os_, den_os, proj_os,
     rc2, ok, safe, k_in, k, outside, before, ok_out, den_out, q, F, k_out) = st
    gc = k[..., None] * u
    gk = _dot(u, c)
    # Over the segment: k = proj / (4 pi (cr2 + rc2 r0sq)).
    gki = np.where(outside, 0.0, gk)
    gproj = np.where(ok, gki * _INV_4PI / safe, 0.0)
    gden = np.where(ok, -gki * k_in / safe, 0.0)
    gcr2 = gden.copy()
    grc2 = gden * r0sq
    gr0sq = gden * rc2
    gpo = np.where(os_, gproj, 0.0)
    gpi = np.where(os_, 0.0, gproj)
    # Stable form: proj = cr2 (n1 + n2) / G, G = n12 e.
    gcr2 = gcr2 + gpo * (n1 + n2) / den_os
    gsum = gpo * cr2 / den_os
    gG = -gpo * proj_os / den_os
    gn12 = gG * e
    ge = gG * n12
    # Outside: k = (n1 + n2) / (4 pi n12 e) F, F = D2 / (D2 + rc2).
    gko = np.where(ok_out, gk, 0.0)
    gsum = gsum + gko * _INV_4PI * F / den_out
    gG = -gko * k_out / den_out
    gn12 = gn12 + gG * e
    ge = ge + gG * n12
    gF = gko * _INV_4PI * (n1 + n2) / den_out
    # dF/dD2 = rc2 / (D2 + rc2)^2, dF/drc2 = -D2 / (D2 + rc2)^2.
    gD2 = gF * (rc2 * q) * q
    grc2 = grc2 - gF * F * q
    # e = n12 + dot12.
    gn12 = gn12 + ge
    gdot12 = ge
    gn1 = gsum + gn12 * n2
    gn2 = gsum + gn12 * n1
    # D2 = |r1|^2 before A, |r2|^2 after B.
    gr1 = gdot12[..., None] * r2 + (2.0 * np.where(before, gD2, 0.0))[..., None] * r1
    gr2 = gdot12[..., None] * r1 + (2.0 * np.where(before, 0.0, gD2))[..., None] * r2
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
    D2 = _dot(r, r) + floor
    rn = np.sqrt(D2)
    dr = _dot(d, r)
    # In front of the start point: k = (1 + cos(t)) / (4 pi (cr2 + rc2)).
    denom = cr2 + rc2
    ok = denom > floor
    safe = np.where(ok, denom, 1.0)
    k_in = np.where(ok, _INV_4PI * (1.0 + dr / rn) / safe, 0.0)
    # Behind it: exact law times F = D2 / (D2 + rc2).
    behind = dr < 0.0
    e = rn * (rn - dr)
    ok_out = behind & (e > floor)
    den_out = np.where(ok_out, e, 1.0)
    q = 1.0 / (D2 + rc2)
    F = D2 * q
    k_out = np.where(ok_out, _INV_4PI / den_out * F, 0.0)
    k = np.where(behind, k_out, k_in)
    v = c * k[..., None]
    return v, (r, d, c, cr2, rn, dr, rc2, ok, safe, k_in, k, behind, ok_out, den_out, q, F, k_out)


def _ray_backward(u, st):
    """Return (gP, gA, gd, grc2) of a ray for the adjoint *u* of its velocity."""
    r, d, c, cr2, rn, dr, rc2, ok, safe, k_in, k, behind, ok_out, den_out, q, F, k_out = st
    gc = k[..., None] * u
    gk = _dot(u, c)
    # In front: k = (1 + dr / rn) / (4 pi (cr2 + rc2)).
    gkf = np.where(behind, 0.0, gk)
    goc = np.where(ok, gkf * _INV_4PI / safe, 0.0)
    gden = np.where(ok, -gkf * k_in / safe, 0.0)
    gcr2 = gden.copy()
    grc2 = gden.copy()
    gdr = goc / rn
    grn = -goc * dr / (rn * rn)
    # Behind: k = F / (4 pi E), E = rn (rn - dr), D2 = |r|^2.
    gkb = np.where(ok_out, gk, 0.0)
    gE = -gkb * k_out / den_out
    grn = grn + gE * (2.0 * rn - dr)
    gdr = gdr - gE * rn
    gF = gkb * _INV_4PI / den_out
    gD2 = gF * (rc2 * q) * q
    grc2 = grc2 - gF * F * q
    gr = (grn / rn + 2.0 * gD2)[..., None] * r + gdr[..., None] * d
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


def _kinked(a_te, b_te, kink):
    """Return the indices of the sources with a leg node, shape (k,); None without nodes.

    The flags ``kink[4]`` give them when present; else a source has a node
    where its node differs from its trailing-edge point.
    """
    if kink is None:
        return None
    if len(kink) > 4 and kink[4] is not None:
        return np.flatnonzero(np.asarray(kink[4]) != 0)
    a_k, b_k = kink[0], kink[1]
    return np.flatnonzero(np.any(a_k != a_te, axis=1) | np.any(b_k != b_te, axis=1))


def _chain(kink):
    """Return ``(a_e, b_e, leg_next)`` of the leg chains in *kink*, or None."""
    if kink is None or len(kink) < 8 or kink[7] is None:
        return None
    nxt = np.asarray(kink[7]) != 0
    if not np.any(nxt):
        return None
    return kink[5], kink[6], nxt


def _horseshoe_rows(P, a, b, a_te, b_te, d, rc2, keeps, floor, kink=None, kidx=None, kkeeps=None):
    """Return the pair velocity (rows, n, 3) and the states of the filaments.

    With leg nodes (*kink* = ``(a_k, b_k, ...)``), the forward segments of
    the legs end at the nodes, and the aft segments of the sources *kidx*
    (masks *kkeeps*) are added; their states are the sixth item. With leg
    chains the aft segments start (a leg) and end (b leg) at ``a_e``,
    ``b_e``, and each source adds the own leg segments of the sources after
    it on its chain.
    """
    Pr = P[:, None, :]
    A, B, AT, BT, D = a[None], b[None], a_te[None], b_te[None], d[None]
    AK, BK = (AT, BT) if kink is None else (kink[0][None], kink[1][None])
    chain = _chain(kink)
    AE, BE = (AT, BT) if chain is None else (chain[0][None], chain[1][None])
    keep_b, keep_la, keep_lb = keeps
    v_b, s_b = _seg_forward(Pr, A, B, rc2, floor)
    v_la, s_la = _seg_forward(Pr, AK, A, rc2, floor)
    v_lb, s_lb = _seg_forward(Pr, B, BK, rc2, floor)
    v_rb, s_rb = _ray_forward(Pr, BT, D, rc2, floor)
    v_ra, s_ra = _ray_forward(Pr, AT, D, rc2, floor)
    v_b = np.where(keep_b[..., None], v_b, 0.0)
    v_la = np.where(keep_la[..., None], v_la, 0.0)
    v_lb = np.where(keep_lb[..., None], v_lb, 0.0)
    if kink is None or kidx.size == 0:
        v = (v_b + v_la + v_lb) + v_rb - v_ra
        return v, (s_b, s_la, s_lb, s_rb, s_ra, None)
    rck = rc2[:, kidx]
    v_ka, s_ka = _seg_forward(Pr, AE[:, kidx], AK[:, kidx], rck, floor)
    v_kb, s_kb = _seg_forward(Pr, BK[:, kidx], BE[:, kidx], rck, floor)
    v_ka = np.where(kkeeps[0][..., None], v_ka, 0.0)
    v_kb = np.where(kkeeps[1][..., None], v_kb, 0.0)
    v_la = v_la.copy()
    v_lb = v_lb.copy()
    v_la[:, kidx] += v_ka
    v_lb[:, kidx] += v_kb
    v = (v_b + v_la + v_lb) + v_rb - v_ra
    if chain is not None:
        from ventorum.aero.vortex import chain_suffix

        v = v + chain_suffix(v_la + v_lb, chain[2], axis=1)
    return v, (s_b, s_la, s_lb, s_rb, s_ra, (s_ka, s_kb))


def _kink_keeps(edge, kink, kidx, i0, i1):
    """Return the keep masks (rows, k) of the aft segments of the sources *kidx*."""
    if kink is None or kidx.size == 0:
        return None
    sea_k, seb_k = kink[2], kink[3]
    return (edge[i0:i1, None] != sea_k[None, kidx], edge[i0:i1, None] != seb_k[None, kidx])


def horseshoe_forward(P, X, a, b, a_te, b_te, d, rc2_src, tg_rc2, src_group, tg_group,
                      own, src_panel, edge, src_edge_a, src_edge_b, s, col, n_u, mode, use_tg, kink=None):
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
    kink : tuple or None
        Leg nodes ``(a_k, b_k, src_edge_ka, src_edge_kb[, flags[, a_e, b_e,
        leg_next]])``: the nodes [m], shape (n, 3), the edge ids of the aft
        segments, shape (n,), optionally the flags (n,) of the sources with
        a node, and the leg chains: the ends of the own leg segments [m],
        shape (n, 3), and the chain flags (n,). None for straight legs.

    Returns
    -------
    numpy.ndarray
        Mode 0: rows (m, n_u) [1/m times the weight]. Mode 1: velocity
        (m, 3) [m/s for a circulation weight].
    """
    floor = floor_of(P.dtype)
    m, n = P.shape[0], a.shape[0]
    kidx = _kinked(a_te, b_te, kink)
    out = np.zeros((m, int(n_u)) if mode == 0 else (m, 3), dtype=P.dtype)
    step = _chunk(m, n)
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        rc2, _ = _pair_core(rc2_src, tg_rc2, src_group, tg_group, i0, i1, use_tg)
        keeps = _keep(own, src_panel, edge, src_edge_a, src_edge_b, i0, i1)
        v, _ = _horseshoe_rows(P[i0:i1], a, b, a_te, b_te, d, rc2, keeps, floor,
                               kink, kidx, _kink_keeps(edge, kink, kidx, i0, i1))
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
                  own, src_panel, edge, src_edge_a, src_edge_b, s, col, W, use_tg, kink=None):
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
        and the source weights (n,). With *kink*, two more items: the
        derivatives with respect to the leg nodes ``a_k`` and ``b_k``
        (n, 3); they are zero on a source without a node, whose leg
        derivative is in ``ga_te`` and ``gb_te``. With leg chains, two
        more items: the derivatives with respect to the ends ``a_e`` and
        ``b_e`` of the own leg segments (n, 3).
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
    kidx = _kinked(a_te, b_te, kink)
    chain = _chain(kink)
    if kink is not None:
        gak = np.zeros((n, 3), dtype=dt)
        gbk = np.zeros((n, 3), dtype=dt)
    if chain is not None:
        from ventorum.aero.vortex import chain_prefix

        gae = np.zeros((n, 3), dtype=dt)
        gbe = np.zeros((n, 3), dtype=dt)
    step = _chunk(m, n)
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        rc2, other = _pair_core(rc2_src, tg_rc2, src_group, tg_group, i0, i1, use_tg)
        keeps = _keep(own, src_panel, edge, src_edge_a, src_edge_b, i0, i1)
        kkeeps = _kink_keeps(edge, kink, kidx, i0, i1)
        v, (s_b, s_la, s_lb, s_rb, s_ra, s_k) = _horseshoe_rows(P[i0:i1], a, b, a_te, b_te, d, rc2,
                                                               keeps, floor, kink, kidx, kkeeps)
        wf = W[i0:i1][:, col] if W is not None else np.ones((i1 - i0, n), dtype=dt)
        cij = wf * s[None, :]
        Xr = X[i0:i1, None, :]
        u = cij[..., None] * Xr
        gX[i0:i1] = (cij[..., None] * v).sum(axis=1)
        gs += (wf * _dot(Xr, v)).sum(axis=0)
        keep_b, keep_la, keep_lb = keeps
        gP_b, gA_b, gB_b, gr_b = _seg_backward(u * keep_b[..., None], s_b)
        # The own leg segments carry the adjoint of every source before them
        # on their chain.
        u_leg = u if chain is None else chain_prefix(u, chain[2], axis=1)
        gP_la, gA_la, gB_la, gr_la = _seg_backward(u_leg * keep_la[..., None], s_la)
        gP_lb, gA_lb, gB_lb, gr_lb = _seg_backward(u_leg * keep_lb[..., None], s_lb)
        gP_rb, gA_rb, gd_rb, gr_rb = _ray_backward(u, s_rb)
        gP_ra, gA_ra, gd_ra, gr_ra = _ray_backward(-u, s_ra)
        if s_k is None:
            gP[i0:i1] = (gP_b + gP_la + gP_lb + gP_rb + gP_ra).sum(axis=1)
        else:
            # Aft segments of the legs with a node (sources kidx).
            uk = u_leg[:, kidx]
            gP_ka, gA_ka, gB_ka, gr_ka = _seg_backward(uk * kkeeps[0][..., None], s_k[0])
            gP_kb, gA_kb, gB_kb, gr_kb = _seg_backward(uk * kkeeps[1][..., None], s_k[1])
            gP_la = gP_la.copy()
            gP_la[:, kidx] += gP_ka + gP_kb
            gr_la = gr_la.copy()
            gr_la[:, kidx] += gr_ka + gr_kb
            gP[i0:i1] = (gP_b + gP_la + gP_lb + gP_rb + gP_ra).sum(axis=1)
        ga += (gA_b + gB_la).sum(axis=0)
        gb += (gB_b + gA_lb).sum(axis=0)
        if kink is None:
            gat += (gA_la + gA_ra).sum(axis=0)
            gbt += (gB_lb + gA_rb).sum(axis=0)
        else:
            # The forward segments start (a leg) and end (b leg) at the nodes.
            gak += gA_la.sum(axis=0)
            gbk += gB_lb.sum(axis=0)
            gat += gA_ra.sum(axis=0)
            gbt += gA_rb.sum(axis=0)
            if s_k is not None:
                # The aft segments end at the trailing edge, or on a chain
                # at the ends of the own legs.
                (gat if chain is None else gae)[kidx] += gA_ka.sum(axis=0)
                gak[kidx] += gB_ka.sum(axis=0)
                gbk[kidx] += gA_kb.sum(axis=0)
                (gbt if chain is None else gbe)[kidx] += gB_kb.sum(axis=0)
        gd += (gd_rb + gd_ra).sum(axis=0)
        grc2 = gr_b + gr_la + gr_lb + gr_rb + gr_ra
        grc += grc2.sum(axis=0)
        if other is not None:
            gtg[i0:i1] = np.where(other, grc2, 0.0).sum(axis=1)
    if kink is None:
        return gP, gX, gtg, ga, gb, gat, gbt, gd, grc, gs
    # On a source without a node the forward segment is the whole leg.
    straight = np.ones(n, dtype=bool)
    straight[kidx] = False
    gat[straight] += gak[straight]
    gbt[straight] += gbk[straight]
    gak[straight] = 0.0
    gbk[straight] = 0.0
    if chain is not None:
        return gP, gX, gtg, ga, gb, gat, gbt, gd, grc, gs, gak, gbk, gae, gbe
    return gP, gX, gtg, ga, gb, gat, gbt, gd, grc, gs, gak, gbk


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
