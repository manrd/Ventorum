# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Post-processing for the 3D source-doublet panel method.

This module computes surface tangential perturbation velocities from the doublet
gradient, evaluates surface pressure coefficients, integrates near-field forces
and moments, and calculates far-field Trefftz-plane induced drag.

Surface gradient
----------------
The solver uses :func:`build_surface_gradient_operator`, a sparse matrix
built once per mesh, so the gradient of any doublet vector is one sparse
product. On a panel whose corner nodes all lie inside the surface, the
doublet values are first reconstructed at the corner nodes with a
least-squares linear fit in the node tangent plane, and the gradient
follows from the node values with the Green-Gauss theorem on the panel
polygon (Blazek 2015, chapter 5). The node values are exact for a field
that is linear in the tangent plane, also at irregular nodes (three
panels at a node, planform kinks, cosine end strips). On a panel with a
corner on a wake shedding edge or on a mesh border, a node value would
sit about half a panel away from the node, and a node value is blind to
a spanwise saw-tooth of the trailing-edge doublets; these panels use the
edge-neighbour least-squares fit of :func:`compute_surface_gradient`,
anchored at the panel value. No step differentiates across the doublet
jump of the wake. The previous operator (inverse-distance node values,
Shepard 1968) is kept for comparison in
``_build_surface_gradient_operator_shepard``.

Published sources
-----------------
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, section 9.4 and chapter 12.
* Drela, M., 2014, Flight Vehicle Aerodynamics, MIT Press, chapter 5.
* Stevens, B. L., Lewis, F. L., and Johnson, E. N., 2016, Aircraft Control
  and Simulation, 3rd ed., Wiley, chapter 2.
* Shepard, D., 1968, A two-dimensional interpolation function for
  irregularly-spaced data, Proceedings of the 23rd ACM National Conference,
  pp. 517-524 (inverse distance weights).
* Blazek, J., 2015, Computational Fluid Dynamics: Principles and
  Applications, 3rd ed., Butterworth-Heinemann, chapter 5 (least-squares
  and Green-Gauss gradients).

Conventions
-----------
* Surface velocity: V = V_onset_tangential - grad(mu).
* Pressure coefficient: Cp = 1 - |V|^2 / V_inf^2 without body rates. With
  body rates, Cp = (|V_k|^2 - |V|^2) / V_inf^2, where V_k is the onset
  velocity at the panel (free stream minus Omega x (x - x_ref)); see
  :func:`compute_surface_velocity_and_cp`.
* Pressure force on panel: dF = -Cp * q_inf * Area * n.
* Wind axes: Lift along L_hat = [-sin(alpha), 0, cos(alpha)], side force
  along Y_hat = L_hat x u_hat, drag along u_hat.
* Moments: Flight-dynamics body axes (AVL convention):
  Cl = -Mx / (q_inf * S_ref * b_ref)  (right wing down positive)
  Cm = +My / (q_inf * S_ref * c_ref)  (nose up positive)
  Cn = -Mz / (q_inf * S_ref * b_ref)  (nose right positive).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

from ventorum.aero.system import freestream_direction, lift_direction
from ventorum.core.datatypes import FlightCondition
from ventorum.panel3d.wake import PanelWake

__all__ = [
    "FOLD_ANGLE_DEG",
    "TREFFTZ_CORE_FRACTION",
    "build_edge_neighbourhood",
    "build_surface_gradient_operator",
    "compute_surface_gradient",
    "compute_surface_velocity_and_cp",
    "compute_trefftz_induced_drag",
    "integrate_forces_and_moments",
    "surface_gradient",
]

#: Largest angle between the normals of two neighbour panels for the
#: surface gradient of a mesh without panel kinds [deg]. An edge with a
#: larger fold is a sharp edge of the surface: the gradient does not use
#: values across it. A mesh of the wing builder marks its sharp edges with
#: the panel kinds instead (see :func:`build_surface_gradient_operator`).
#: This is a method choice, not a tuned value.
FOLD_ANGLE_DEG = 45.0

#: Core radius of the 2D trailing vortices in the Trefftz plane, as a
#: fraction of the mean length of the wake strip trailing edges [-]. The
#: core only removes the singularity when a vortex lies on an evaluation
#: point (for example at a wing-fin junction); the evaluation points are the
#: strip midpoints, half a strip away from the nearest vortex, so the core
#: changes the drag by about (2 * 0.01)^2 relative on a uniform strip layout.
TREFFTZ_CORE_FRACTION = 0.01


def build_edge_neighbourhood(
    panels: np.ndarray,
    wake_edges: np.ndarray | None = None,
) -> list[list[int]]:
    """Find neighbour panels that share an edge, excluding wake shedding edges.

    Parameters
    ----------
    panels : numpy.ndarray
        Corner node indices of panels with shape (Np, 4).
    wake_edges : numpy.ndarray or None, optional
        Wake shedding edges with shape (Nw, 4).

    Returns
    -------
    list of list of int
        List of neighbour panel indices for each panel.
    """
    wake_set = set()
    if wake_edges is not None and len(wake_edges) > 0:
        w_arr = np.asarray(wake_edges, dtype=int).reshape(-1, 4)
        for row in w_arr:
            na, nb = int(row[2]), int(row[3])
            wake_set.add((min(na, nb), max(na, nb)))

    edge_to_panels: dict[tuple[int, int], list[int]] = {}
    n_panels = panels.shape[0]

    for i in range(n_panels):
        row = panels[i]
        for k in range(4):
            a, b = int(row[k]), int(row[(k + 1) % 4])
            if a == b:
                continue
            edge = (min(a, b), max(a, b))
            if edge in wake_set:
                # Do not connect across the wake shedding boundary
                continue
            edge_to_panels.setdefault(edge, []).append(i)

    neighbour_sets = [set() for _ in range(n_panels)]
    for p_list in edge_to_panels.values():
        if len(p_list) == 2:
            p0, p1 = p_list[0], p_list[1]
            neighbour_sets[p0].add(p1)
            neighbour_sets[p1].add(p0)
        elif len(p_list) > 2:
            for p in p_list:
                for q in p_list:
                    if p != q:
                        neighbour_sets[p].add(q)

    return [sorted(s) for s in neighbour_sets]


def compute_surface_gradient(
    centroids: np.ndarray,
    normals: np.ndarray,
    mu: np.ndarray,
    neighbour_indices: list[list[int]],
) -> np.ndarray:
    """Compute the tangential surface gradient of doublet strength on each panel.

    A least-squares fit over neighbour panels that share an edge is solved in
    the tangent plane of the panel. This function does not stop at sharp
    edges: a neighbour across a fold of about 90 deg (tip cap, base panel,
    control side closure) gives very large gradients. The solver uses
    :func:`build_surface_gradient_operator`, which treats sharp edges and
    the pole fans of latitude-longitude meshes; this function stays
    available for comparisons.

    Parameters
    ----------
    centroids : numpy.ndarray
        Panel centroid coordinates with shape (Np, 3) in m.
    normals : numpy.ndarray
        Panel unit normal vectors with shape (Np, 3).
    mu : numpy.ndarray
        Doublet strengths on panels with shape (Np,) in m^2/s.
    neighbour_indices : list of list of int
        List of neighbour indices for each panel.

    Returns
    -------
    numpy.ndarray
        Surface gradient vectors with shape (Np, 3) in m/s.
    """
    n_panels = centroids.shape[0]
    grad_mu = np.zeros((n_panels, 3), dtype=np.float64)

    for i in range(n_panels):
        nbrs = neighbour_indices[i]
        if len(nbrs) == 0:
            continue

        c_i = centroids[i]
        n_i = normals[i]
        dr = centroids[nbrs] - c_i[None, :]

        # Project neighbour displacement vectors onto the panel tangent plane
        dr_normal = np.sum(dr * n_i[None, :], axis=-1, keepdims=True)
        dr_tangent = dr - dr_normal * n_i[None, :]
        dmu = mu[nbrs] - mu[i]

        # Solve least-squares problem: dr_tangent @ g = dmu
        g, _, _, _ = np.linalg.lstsq(dr_tangent, dmu, rcond=None)

        # Enforce exact tangency to panel normal
        g_proj = g - np.dot(g, n_i) * n_i
        grad_mu[i] = g_proj

    return grad_mu


def _find_root(parent: dict[int, int], p: int) -> int:
    """Return the root of *p* in a union-find forest (with path halving)."""
    while parent[p] != p:
        parent[p] = parent[parent[p]]
        p = parent[p]
    return p


def _unique_corners(row: np.ndarray) -> list[int]:
    """Return the distinct corner nodes of one panel in loop order."""
    out: list[int] = []
    for a in row:
        a = int(a)
        if not out or (a != out[-1] and a != out[0]):
            out.append(a)
    return out


def _distinct_corners(pan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the distinct corners of all panels (the rule of :func:`_unique_corners`).

    Parameters
    ----------
    pan : numpy.ndarray
        Corner node indices with shape (Np, 4).

    Returns
    -------
    corner : numpy.ndarray
        Distinct corner nodes in loop order with shape (Np, 4), padded
        with -1.
    count : numpy.ndarray
        Count of distinct corners of each panel with shape (Np,).
    """
    n_p, width = pan.shape
    corner = np.full((n_p, width), -1, dtype=pan.dtype)
    corner[:, 0] = pan[:, 0]
    count = np.ones(n_p, dtype=np.intp)
    last = pan[:, 0].copy()
    row = np.arange(n_p)
    for j in range(1, width):
        a = pan[:, j]
        take = (a != last) & (a != pan[:, 0])
        corner[row[take], count[take]] = a[take]
        last = np.where(take, a, last)
        count = count + take
    return corner, count


def _counts_to_ptr(counts: np.ndarray) -> np.ndarray:
    """Return the CSR row pointer of the row lengths *counts*, shape (n + 1,)."""
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.intp)


def _csr_expand(ptr: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one item per entry of the CSR rows *ids*.

    Parameters
    ----------
    ptr : numpy.ndarray
        CSR row pointer with shape (n + 1,).
    ids : numpy.ndarray
        Row indices with shape (m,). A row can occur more than once.

    Returns
    -------
    owner : numpy.ndarray
        Position in *ids* of each item.
    entry : numpy.ndarray
        Index of the CSR entry of each item.
    """
    start = ptr[ids]
    count = ptr[ids + 1] - start
    owner = np.repeat(np.arange(ids.size), count)
    first = np.cumsum(count) - count
    entry = np.arange(owner.size) + np.repeat(start - first, count)
    return owner, entry


def _pairs_in_runs(ptr: np.ndarray, run: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return all index pairs (a, b) with a < b of the entries in one CSR row.

    Parameters
    ----------
    ptr : numpy.ndarray
        CSR row pointer with shape (n + 1,).
    run : numpy.ndarray
        Row of each entry with shape (m,), in non-decreasing order.

    Returns
    -------
    a, b : numpy.ndarray
        Entry indices of each pair.
    """
    n = run.size
    after = ptr[run + 1] - 1 - np.arange(n)
    a = np.repeat(np.arange(n), after)
    first = np.cumsum(after) - after
    b = a + 1 + np.arange(a.size) - np.repeat(first, after)
    return a, b


def _fold_pairs(nrm: np.ndarray, pa: np.ndarray, pb: np.ndarray, cos_fold: float) -> np.ndarray:
    """Return True for each panel pair whose normals fold more than the limit.

    Parameters
    ----------
    nrm : numpy.ndarray
        Panel unit normals with shape (Np, 3) [-].
    pa, pb : numpy.ndarray
        Panel indices of the pairs with shape (m,).
    cos_fold : float
        Cosine of the fold angle limit [-].

    Returns
    -------
    numpy.ndarray
        True where the dot product of the normals is less than *cos_fold*.
    """
    a = nrm[pa]
    b = nrm[pb]
    dot = np.einsum("ij,ij->i", a, b)
    sharp = dot < cos_fold
    # A product sum in a different order can differ from numpy.dot by
    # round-off. For the pairs near the limit, numpy.dot decides, as in the
    # loop version, so that the cut edges do not change.
    scale = np.einsum("ij,ij->i", np.abs(a), np.abs(b))
    for j in np.nonzero(np.abs(dot - cos_fold) <= 1e-12 * scale)[0]:
        sharp[j] = float(np.dot(nrm[pa[j]], nrm[pb[j]])) < cos_fold
    return sharp


def _wake_node_groups(
    special: np.ndarray,
    inc_p: np.ndarray,
    inc_w: np.ndarray,
    groups: tuple[np.ndarray, np.ndarray, np.ndarray],
    opp_codes: np.ndarray,
    inc_edges: tuple[np.ndarray, np.ndarray],
    key_panels: tuple[np.ndarray, np.ndarray],
    cut: np.ndarray,
    n_p: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the groups at wake nodes without the panels across the wake.

    Parameters
    ----------
    special : numpy.ndarray
        Incidences (node v, panel i) at a wake node of a panel with
        panels on the other side of the wake, shape (S,).
    inc_p : numpy.ndarray
        Panel of each incidence with shape (Ni,).
    inc_w : numpy.ndarray
        Inverse-distance weight of each incidence (times its multiplicity)
        with shape (Ni,) in 1/m.
    groups : tuple of numpy.ndarray
        Group CSR pointer, group members (incidences) and group of each
        incidence.
    opp_codes : numpy.ndarray
        Sorted codes ``i * Np + q`` of the panels q on the other side of
        the wake of panel i.
    inc_edges : tuple of numpy.ndarray
        CSR pointer over the incidences and the keys of the edges of the
        panel at the node of each incidence.
    key_panels : tuple of numpy.ndarray
        CSR pointer over the keys and the panels of each key.
    cut : numpy.ndarray
        True for each cut key, shape (Nk,).
    n_p : int
        Count of panels (the base of the codes).

    Returns
    -------
    col : numpy.ndarray
        Member panels of the S lists.
    weight : numpy.ndarray
        Normalized inverse-distance weights of the members [-].
    ptr : numpy.ndarray
        CSR pointer of the S lists, shape (S + 1,).
    closed : numpy.ndarray
        True for each list whose group is closed, shape (S,).
    """
    g_ptr, g_member, inc_grp = groups
    ie_ptr, ie_key = inc_edges
    k_ptr, k_pan = key_panels
    n_s = special.size
    owner, entry = _csr_expand(g_ptr, inc_grp[special])
    member = g_member[entry]
    i_s = inc_p[special][owner]
    q = inc_p[member]
    keep = (q == i_s) | ~np.isin(i_s * n_p + q, opp_codes)
    s_k = owner[keep]
    m_k = member[keep]
    q_k = q[keep]
    w_k = inc_w[m_k]
    w_sum = np.bincount(s_k, weights=w_k, minlength=n_s)
    ptr = _counts_to_ptr(np.bincount(s_k, minlength=n_s))
    kept = np.unique(s_k * n_p + q_k)
    # Each edge at the node of each kept member is shared when it is not
    # cut and another kept member lies on it.
    t_own, t_entry = _csr_expand(ie_ptr, m_k)
    t_s = s_k[t_own]
    t_q = q_k[t_own]
    t_key = ie_key[t_entry]
    r_own, r_entry = _csr_expand(k_ptr, t_key)
    r = k_pan[r_entry]
    other = (r != t_q[r_own]) & np.isin(t_s[r_own] * n_p + r, kept)
    shared = ~cut[t_key] & (np.bincount(r_own[other], minlength=t_key.size) > 0)
    closed = np.bincount(t_s[~shared], minlength=n_s) == 0
    return q_k, w_k / w_sum[s_k], ptr, closed


def _green_gauss_corner_weights(
    pts: np.ndarray,
    corner: np.ndarray,
    count: np.ndarray,
    nrm: np.ndarray,
    area: np.ndarray,
) -> np.ndarray:
    """Return the Green-Gauss weights of the corner node values of panels.

    The edge (a, b) adds ``(p_b - p_a) x n / (2 A)`` to the weights of its
    two corners (Blazek 2015, chapter 5).

    Parameters
    ----------
    pts : numpy.ndarray
        Node coordinates with shape (Nn, 3) in m.
    corner : numpy.ndarray
        Distinct corner nodes with shape (Ng, 4), padded with -1.
    count : numpy.ndarray
        Count of distinct corners with shape (Ng,).
    nrm : numpy.ndarray
        Panel unit normals with shape (Ng, 3) [-].
    area : numpy.ndarray
        Panel areas with shape (Ng,) in m^2.

    Returns
    -------
    numpy.ndarray
        Weights with shape (Ng, 4, 3) in 1/m.
    """
    n_g, width = corner.shape
    scale = 0.5 / area
    edge_w = np.zeros((n_g, width, 3))
    for e in range(width):
        has = np.nonzero(e < count)[0]
        b = np.where(e + 1 < count[has], e + 1, 0)
        side = pts[corner[has, b]] - pts[corner[has, e]]
        edge_w[has, e] = np.cross(side, nrm[has]) * scale[has, None]
    weight = np.zeros((n_g, width, 3))
    for e in range(width):
        has = np.nonzero(e < count)[0]
        prev = count[has] - 1 if e == 0 else np.full(has.size, e - 1)
        weight[has, e] = edge_w[has, e] + edge_w[has, prev]
    return weight


#: Relative singular value below which a displacement direction of the
#: edge-neighbour least-squares fit is round-off [-].
LS_RCOND = 1e-5

#: Panel kind whose least-squares gradient keeps only the direction along its
#: row of neighbours (the flat tip caps of the wing builder).
ROW_GRADIENT_KIND = "tip"


def _stencil_pinv(dr_t: np.ndarray, row_only: np.ndarray | bool) -> np.ndarray:
    """Return the pseudo-inverse of in-plane stencil displacements.

    Parameters
    ----------
    dr_t : numpy.ndarray
        In-plane displacements from the panel centroid to its neighbour
        centroids with shape (..., k, 3) in m.
    row_only : numpy.ndarray or bool
        Flag with the batch shape of *dr_t* (or one flag). True keeps only
        the leading singular direction (the gradient along the row of
        neighbours). False keeps every direction with a singular value
        above ``LS_RCOND`` times the largest.

    Returns
    -------
    numpy.ndarray
        Pseudo-inverse with shape (..., 3, k) in 1/m. False flags give the
        result of ``numpy.linalg.pinv(dr_t, rcond=LS_RCOND)``.
    """
    u, s, vt = np.linalg.svd(dr_t, full_matrices=False)
    large = s > LS_RCOND * np.max(s, axis=-1, keepdims=True)
    flag = np.broadcast_to(np.asarray(row_only, dtype=bool), s.shape[:-1])
    large[flag, 1:] = False
    s = np.divide(1.0, s, where=large, out=s)
    s[~large] = 0.0
    return np.matmul(np.swapaxes(vt, -1, -2), s[..., None] * np.swapaxes(u, -1, -2))


#: Largest magnitude of a least-squares node weight [-]. A node with a larger
#: weight keeps the inverse-distance weights (a node far outside its panel
#: centroids). This is a method choice, not a tuned value.
NODE_WEIGHT_LIMIT = 2.0


def _linear_node_weights(dd: np.ndarray, nrm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the least-squares linear node weights of panel groups.

    The node value is the value at the node of the least-squares fit
    ``mu_j = a + b . d_j`` over the group, with ``d_j`` the displacement
    from the node to the centroid of panel ``j`` projected on the node
    tangent plane (least-squares reconstruction, Blazek 2015, chapter 5).
    The node normal is the normalized sum of the group normals. The
    weights are the first row of the pseudo-inverse of the matrix
    ``[1, d_j / L]``, with ``L`` the root-mean-square length of the
    projected displacements, so that the threshold LS_RCOND does not
    depend on the panel size. They sum to 1, and they give the exact node
    value of every field that is linear in the node tangent plane. A
    group is valid when the matrix has rank 3 and no weight is larger
    than NODE_WEIGHT_LIMIT in magnitude.

    Parameters
    ----------
    dd : numpy.ndarray
        Displacements from the node to the member centroids with shape
        (..., k, 3) in m.
    nrm : numpy.ndarray
        Unit normals of the members with shape (..., k, 3) [-].

    Returns
    -------
    weights : numpy.ndarray
        Weights with shape (..., k) [-].
    valid : numpy.ndarray
        True for each valid group, with the batch shape of *dd*.
    """
    n_v = np.sum(nrm, axis=-2)
    n_v = n_v / np.maximum(np.linalg.norm(n_v, axis=-1, keepdims=True), 1e-300)
    dt = dd - (dd @ n_v[..., :, None]) * n_v[..., None, :]
    size = np.sqrt(np.mean(np.sum(dt * dt, axis=-1), axis=-1))
    x = np.concatenate([np.ones(dt.shape[:-1] + (1,)), dt / np.maximum(size, 1e-300)[..., None, None]], axis=-1)
    u, sv, vt = np.linalg.svd(x, full_matrices=False)
    large = sv > LS_RCOND * sv[..., :1]
    inv = np.where(large, 1.0 / np.where(large, sv, 1.0), 0.0)
    weights = np.einsum("...j,...j,...kj->...k", vt[..., :, 0], inv, u)
    valid = (np.sum(large, axis=-1) >= 3) & (np.max(np.abs(weights), axis=-1) <= NODE_WEIGHT_LIMIT)
    return weights, valid


def _list_node_weights(
    col: np.ndarray,
    ptr: np.ndarray,
    node: np.ndarray,
    pts: np.ndarray,
    cen: np.ndarray,
    nrm: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    """Return the least-squares node weights of CSR member lists.

    The lists with the same length share one stacked call of
    :func:`_linear_node_weights`.

    Parameters
    ----------
    col : numpy.ndarray
        Member panels of all lists with shape (M,).
    ptr : numpy.ndarray
        CSR pointer of the lists with shape (L + 1,).
    node : numpy.ndarray
        Node of each list with shape (L,).
    pts : numpy.ndarray
        Node coordinates with shape (Nn, 3) in m.
    cen : numpy.ndarray
        Panel centroids with shape (Np, 3) in m.
    nrm : numpy.ndarray
        Panel unit normals with shape (Np, 3) [-].
    fallback : numpy.ndarray
        Inverse-distance weights of the members with shape (M,) [-]. A
        list that is not valid keeps them.

    Returns
    -------
    numpy.ndarray
        Weights of the members with shape (M,) [-].
    """
    out = np.array(fallback, dtype=float, copy=True)
    length = np.diff(ptr)
    for k in np.unique(length):
        if k == 0:
            continue
        lists = np.nonzero(length == k)[0]
        idx = ptr[lists][:, None] + np.arange(k)[None, :]
        members = col[idx]
        w, valid = _linear_node_weights(cen[members] - pts[node[lists]][:, None, :], nrm[members])
        out[idx[valid]] = w[valid]
    return out


def _least_squares_entries(
    ls: np.ndarray,
    nb_p: np.ndarray,
    nb_q: np.ndarray,
    cen: np.ndarray,
    nrm: np.ndarray,
    n_p: int,
    row_only: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Return the matrix entries of the edge-neighbour least-squares gradient.

    The panels with the same count of neighbours share one stacked
    pseudo-inverse call (the same matrices as one call per panel).

    Parameters
    ----------
    ls : numpy.ndarray
        Border panels with shape (Nl,).
    nb_p, nb_q : numpy.ndarray
        Neighbour pairs (panel, neighbour), sorted by panel, then by
        neighbour.
    cen : numpy.ndarray
        Panel centroids with shape (Np, 3) in m.
    nrm : numpy.ndarray
        Panel unit normals with shape (Np, 3) [-].
    n_p : int
        Count of panels.
    row_only : numpy.ndarray or None, optional
        Flag of each panel with shape (Np,): True keeps only the gradient
        along the row of neighbours (see :func:`_stencil_pinv`). None
        gives False for all panels.

    Returns
    -------
    rows, cols, vals : list of numpy.ndarray
        Row indices, column indices and values (in 1/m) of the entries.
    """
    if row_only is None:
        row_only = np.zeros(n_p, dtype=bool)
    nb_ptr = _counts_to_ptr(np.bincount(nb_p, minlength=n_p))
    n_nb = nb_ptr[ls + 1] - nb_ptr[ls]
    comp = np.arange(3)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for n in np.unique(n_nb):
        ps = ls[n_nb == n]
        if n == 0:
            # No neighbour: explicit zeros on the panel itself.
            rows.append((3 * ps[:, None] + comp[None, :]).ravel())
            cols.append(np.repeat(ps, 3))
            vals.append(np.zeros(3 * ps.size))
            continue
        nb = nb_q[nb_ptr[ps][:, None] + np.arange(n)[None, :]]
        dr = cen[nb] - cen[ps][:, None, :]
        n_ps = nrm[ps]
        dr_t = dr - (dr @ n_ps[:, :, None]) * n_ps[:, None, :]
        # Displacement directions below LS_RCOND of the largest are round-off
        # (a row of collinear centroids gives the gradient along the row).
        # A tip cap keeps only the direction along its row.
        pinv = _stencil_pinv(dr_t, row_only[ps])
        mat = np.concatenate([pinv, -np.sum(pinv, axis=2, keepdims=True)], axis=2)
        idx = np.concatenate([nb, ps[:, None]], axis=1)
        rows.append(np.broadcast_to(3 * ps[:, None, None] + comp[None, :, None], mat.shape).ravel())
        cols.append(np.broadcast_to(idx[:, None, :], mat.shape).ravel())
        vals.append(mat.ravel())
    return rows, cols, vals


def _gradient_matrix(rows: list, cols: list, vals: list, n_p: int) -> scipy.sparse.csr_matrix:
    """Return the (3 Np, Np) CSR matrix of the entries, with duplicates summed."""
    r = np.concatenate(rows).astype(int) if rows else np.zeros(0, dtype=int)
    c = np.concatenate(cols).astype(int) if cols else np.zeros(0, dtype=int)
    v = np.concatenate(vals).astype(float) if vals else np.zeros(0, dtype=float)
    op = scipy.sparse.csr_matrix((v, (r, c)), shape=(3 * n_p, n_p))
    op.sum_duplicates()
    return op


def build_surface_gradient_operator(
    nodes: np.ndarray,
    panels: np.ndarray,
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    wake_edges: np.ndarray | None = None,
    kinds: np.ndarray | None = None,
    fold_angle_deg: float = FOLD_ANGLE_DEG,
) -> scipy.sparse.csr_matrix:
    """Build the sparse operator of the tangential surface gradient.

    The gradient of panel ``i`` is ``(G @ mu)[3 * i: 3 * i + 3]``.

    1. Node groups. At each corner node ``v`` of panel ``i`` the group is
       the set of panels around ``v`` that panel ``i`` reaches through
       edges at ``v`` that are not cut. Wake shedding edges are cut, and
       so are sharp edges, where a doublet value on the other side does
       not belong to the tangent plane of the panel. With panel *kinds*
       (more than one kind in the mesh), a sharp edge is an edge between
       panels of different kinds: the wing builder gives the flat tip caps,
       the trailing-edge base panels and the control side closures their
       own kinds, and they meet the surface at about 90 deg. The folds
       between surface panels (for example at a coarse leading edge, where
       the first upper and lower panels can meet at more than 90 deg) stay
       connected, because the surface is smooth there. Without kinds, a
       sharp edge is an edge where the normals of two panels differ by
       more than *fold_angle_deg*. At a node of a
       wake shedding edge, the group of a panel also excludes the panels on
       the other side of the wake: the panel paired with it in a wake edge
       and the panels that share a wake edge with it. A group is closed
       when every edge at ``v`` of its panels is shared by two panels of
       the group (an interior node); it is open at a wake shedding edge or
       at a mesh border or at a sharp edge.
    2. Interior panels (all corner groups closed). The node value is the
       value at the node of the least-squares linear fit of the group
       values in the node tangent plane (see :func:`_linear_node_weights`;
       least-squares reconstruction, Blazek 2015, chapter 5). It is exact
       for every field that is linear in that plane, on any mesh: also at
       nodes with three panels, planform kinks and cosine end strips,
       where the weighted centroids do not average to the node. A group
       whose fit has rank below 3 (fewer than three panels or collinear
       centroids) or a weight larger than NODE_WEIGHT_LIMIT in magnitude
       keeps the inverse-distance weights (Shepard 1968, weight 1 / d with
       d the distance from the node to the panel centroid). The
       Green-Gauss theorem on the panel polygon then gives
       ``grad_i = (1 / A_i) * sum_e 0.5 * (mu_a + mu_b) * (p_b - p_a) x n_i``
       over the edges ``e = (a, b)`` (Blazek 2015, chapter 5). The cross
       product with the unit normal keeps the gradient in the panel plane.
       With exact node values the theorem is exact for a linear field on
       a planar polygon.
    3. Border panels (a corner group is open). The gradient is the
       edge-neighbour least-squares fit of :func:`compute_surface_gradient`
       in the panel plane, anchored at the panel value, over the panels
       that share an edge that is not cut. It is exact for
       a linear doublet field on a flat stencil, and the trailing-edge
       pressure stays sensitive to the strength of each wake strip, which
       the pressure Kutta iteration needs.

    The previous operator (inverse-distance node values for all groups) is
    kept in :func:`_build_surface_gradient_operator_shepard` for
    comparison.

    Parameters
    ----------
    nodes : numpy.ndarray
        Node coordinates with shape (Nn, 3) in m.
    panels : numpy.ndarray
        Corner node indices with shape (Np, 4). A triangle repeats a node.
    centroids : numpy.ndarray
        Panel centroids with shape (Np, 3) in m.
    normals : numpy.ndarray
        Panel unit normals with shape (Np, 3) [-].
    areas : numpy.ndarray
        Panel areas with shape (Np,) in m^2.
    wake_edges : numpy.ndarray or None, optional
        Wake shedding edges with shape (Nw, 4): upper panel, lower panel,
        node a, node b.
    kinds : numpy.ndarray or None, optional
        Panel kind codes with shape (Np,) (for example "surface", "tip",
        "base", "side"). With more than one kind, the edges between kinds
        are the sharp edges.
    fold_angle_deg : float, optional
        Largest angle between the normals of two panels across an edge
        that the gradient connects when the mesh has no kinds [deg].
        Default ``FOLD_ANGLE_DEG`` (45).

    Returns
    -------
    scipy.sparse.csr_matrix
        Operator with shape (3 Np, Np) in 1/m.
    """
    return _gradient_operator_arrays(
        nodes, panels, centroids, normals, areas, wake_edges, kinds, fold_angle_deg, shepard=False
    )


def _build_surface_gradient_operator_shepard(
    nodes: np.ndarray,
    panels: np.ndarray,
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    wake_edges: np.ndarray | None = None,
    kinds: np.ndarray | None = None,
    fold_angle_deg: float = FOLD_ANGLE_DEG,
) -> scipy.sparse.csr_matrix:
    """Build the previous surface gradient operator (reference for tests).

    This is the operator that the solver used before the least-squares
    node values. Interior panels interpolate the doublet values to their corner
    nodes with inverse-distance weights (Shepard 1968) and apply the
    Green-Gauss theorem on the panel polygon (Blazek 2015, chapter 5).
    Border panels use the edge-neighbour least-squares fit in the panel
    plane. The inverse-distance node value is not exact for a linear field
    where the weighted centroids do not average to the node (nodes with
    three panels, planform kinks, dihedral creases, cosine end strips), and
    there the gradient error does not decrease with refinement. The solver
    does not call this function. Tests use it to measure the change.

    Parameters
    ----------
    nodes : numpy.ndarray
        Node coordinates with shape (Nn, 3) in m.
    panels : numpy.ndarray
        Corner node indices with shape (Np, 4). A triangle repeats a node.
    centroids : numpy.ndarray
        Panel centroids with shape (Np, 3) in m.
    normals : numpy.ndarray
        Panel unit normals with shape (Np, 3) [-].
    areas : numpy.ndarray
        Panel areas with shape (Np,) in m^2.
    wake_edges : numpy.ndarray or None, optional
        Wake shedding edges with shape (Nw, 4): upper panel, lower panel,
        node a, node b.
    kinds : numpy.ndarray or None, optional
        Panel kind codes with shape (Np,).
    fold_angle_deg : float, optional
        Largest angle between the normals of two panels across an edge
        that the gradient connects when the mesh has no kinds [deg].

    Returns
    -------
    scipy.sparse.csr_matrix
        Operator with shape (3 Np, Np) in 1/m.
    """
    return _gradient_operator_arrays(
        nodes, panels, centroids, normals, areas, wake_edges, kinds, fold_angle_deg, shepard=True
    )


def _gradient_operator_arrays(
    nodes: np.ndarray,
    panels: np.ndarray,
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    wake_edges: np.ndarray | None,
    kinds: np.ndarray | None,
    fold_angle_deg: float,
    shepard: bool,
) -> scipy.sparse.csr_matrix:
    """Build the surface gradient operator with array operations.

    This is the array version of the loop build in
    :func:`_build_surface_gradient_operator_reference`: the same groups,
    the same sparsity pattern, and values equal to round-off. The
    arguments are those of :func:`build_surface_gradient_operator`.
    *shepard* True gives the previous operator (see
    :func:`_build_surface_gradient_operator_shepard`).

    Returns
    -------
    scipy.sparse.csr_matrix
        Operator with shape (3 Np, Np) in 1/m.
    """
    pts = np.asarray(nodes, dtype=float)
    pan = np.asarray(panels, dtype=int)
    cen = np.asarray(centroids, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    area = np.asarray(areas, dtype=float)
    n_p = pan.shape[0]
    w_rows = (
        np.empty((0, 4), dtype=int)
        if wake_edges is None
        else np.asarray(wake_edges, dtype=int).reshape(-1, 4)
    )
    if n_p == 0:
        return _gradient_matrix([], [], [], 0)

    # Distinct corners of each panel in loop order, and their count.
    corner, n_corner = _distinct_corners(pan)
    has_corner = np.arange(corner.shape[1])[None, :] < n_corner[:, None]
    c_pan, c_pos = np.nonzero(has_corner)
    c_node = corner[c_pan, c_pos]

    # Edge incidences (panel, low node, high node) and their undirected keys.
    nxt = np.where(c_pos + 1 < n_corner[c_pan], c_pos + 1, 0)
    c_next = corner[c_pan, nxt]
    is_edge = c_node != c_next
    e_pan = c_pan[is_edge]
    e_lo = np.minimum(c_node[is_edge], c_next[is_edge])
    e_hi = np.maximum(c_node[is_edge], c_next[is_edge])
    n_code = int(max(pan.max(), w_rows[:, 2:].max(initial=0))) + 1
    keys, e_key = np.unique(e_lo * n_code + e_hi, return_inverse=True)
    e_key = e_key.reshape(-1)
    n_key = keys.size

    # Panels of each key (CSR), and all pairs of two entries of one key.
    k_order = np.argsort(e_key, kind="stable")
    k_pan = e_pan[k_order]
    k_key = e_key[k_order]
    k_ptr = _counts_to_ptr(np.bincount(e_key, minlength=n_key))
    pair_a, pair_b = _pairs_in_runs(k_ptr, k_key)
    pa = k_pan[pair_a]
    pb = k_pan[pair_b]
    pair_key = k_key[pair_a]
    n_distinct = np.bincount(np.unique(e_key * n_p + e_pan) // n_p, minlength=n_key)

    # Cut keys: wake shedding edges, and sharp edges (kinds differ, or a
    # fold larger than fold_angle_deg without kinds).
    w_lo = np.minimum(w_rows[:, 2], w_rows[:, 3])
    w_hi = np.maximum(w_rows[:, 2], w_rows[:, 3])
    is_wake_key = np.isin(keys, w_lo * n_code + w_hi)
    kind_arr = None if kinds is None else np.asarray(kinds).reshape(-1)
    use_kinds = kind_arr is not None and kind_arr.shape[0] == n_p and np.unique(kind_arr).size > 1
    if use_kinds:
        sharp_pair = np.asarray(kind_arr[pa] != kind_arr[pb], dtype=bool)
    else:
        sharp_pair = _fold_pairs(nrm, pa, pb, float(np.cos(np.radians(fold_angle_deg))))
    cut = is_wake_key | (np.bincount(pair_key[sharp_pair], minlength=n_key) > 0)
    ok_key = ~cut & (n_distinct >= 2)

    # Corner incidences (node v, panel p) and their multiplicity.
    inc_codes, c_inc = np.unique(c_node * n_p + c_pan, return_inverse=True)
    c_inc = c_inc.reshape(-1)
    n_inc = inc_codes.size
    inc_mult = np.bincount(c_inc, minlength=n_inc).astype(float)
    inc_v = inc_codes // n_p
    inc_p = inc_codes % n_p
    e_inc_lo = np.searchsorted(inc_codes, e_lo * n_p + e_pan)
    e_inc_hi = np.searchsorted(inc_codes, e_hi * n_p + e_pan)

    # Node groups: connected components of the incidences. Two incidences
    # at node v connect through a key at v that is not cut (one extra graph
    # vertex for each end of each key).
    joined = ~cut[e_key]
    link_a = np.concatenate([e_inc_lo[joined], e_inc_hi[joined]])
    link_b = n_inc + np.concatenate([2 * e_key[joined], 2 * e_key[joined] + 1])
    n_vert = n_inc + 2 * n_key
    graph = scipy.sparse.coo_matrix((np.ones(link_a.size), (link_a, link_b)), shape=(n_vert, n_vert))
    _, label = scipy.sparse.csgraph.connected_components(graph, directed=False)
    _, inc_grp = np.unique(label[:n_inc], return_inverse=True)
    inc_grp = inc_grp.reshape(-1)
    n_grp = int(inc_grp.max()) + 1
    g_member = np.argsort(inc_grp, kind="stable")
    g_ptr = _counts_to_ptr(np.bincount(inc_grp, minlength=n_grp))

    # A group is closed when each edge at the node of each member is not
    # cut and has a second panel (that panel is then in the group).
    bad = ~ok_key[e_key]
    inc_bad = np.zeros(n_inc, dtype=bool)
    inc_bad[e_inc_lo[bad]] = True
    inc_bad[e_inc_hi[bad]] = True
    closed = (np.bincount(inc_grp[inc_bad], minlength=n_grp) == 0)[inc_grp]

    # Inverse-distance weights (Shepard 1968) of the group members.
    dist = np.linalg.norm(cen[inc_p] - pts[inc_v], axis=1)
    inc_w = inc_mult * (1.0 / np.maximum(dist, 1e-300))
    grp_sum = np.bincount(inc_grp, weights=inc_w, minlength=n_grp)

    # Member lists: one for each group, then one for each incidence at a
    # wake node whose group loses the panels on the other side of the wake.
    list_of_inc = inc_grp.copy()
    list_col = [inc_p[g_member]]
    list_w = [inc_w[g_member] / grp_sum[inc_grp[g_member]]]
    list_ptr = [g_ptr]
    on_wake = is_wake_key[pair_key] & (pa != pb)
    opp_i = np.concatenate([pa[on_wake], pb[on_wake], w_rows[:, 0], w_rows[:, 1]])
    opp_q = np.concatenate([pb[on_wake], pa[on_wake], w_rows[:, 1], w_rows[:, 0]])
    valid = (opp_i != opp_q) & (opp_i >= 0) & (opp_i < n_p) & (opp_q >= 0) & (opp_q < n_p)
    opp_codes = np.unique(opp_i[valid] * n_p + opp_q[valid])
    special = np.nonzero(np.isin(inc_v, w_rows[:, 2:]) & np.isin(inc_p, opp_codes // n_p))[0]
    if special.size > 0:
        ie_inc = np.concatenate([e_inc_lo, e_inc_hi])
        ie_key = np.concatenate([e_key, e_key])[np.argsort(ie_inc, kind="stable")]
        ie_ptr = _counts_to_ptr(np.bincount(ie_inc, minlength=n_inc))
        f_col, f_w, f_ptr, f_closed = _wake_node_groups(
            special,
            inc_p,
            inc_w,
            (g_ptr, g_member, inc_grp),
            opp_codes,
            (ie_ptr, ie_key),
            (k_ptr, k_pan),
            cut,
            n_p,
        )
        list_of_inc[special] = n_grp + np.arange(special.size)
        closed[special] = f_closed
        list_col.append(f_col)
        list_w.append(f_w)
        list_ptr.append(g_ptr[-1] + f_ptr[1:])
    all_col = np.concatenate(list_col)
    all_w = np.concatenate(list_w)
    all_ptr = np.concatenate(list_ptr)
    if not shepard:
        # Node of each member list: the node of its group, then the node of
        # each incidence at a wake node.
        list_node = inc_v[g_member[g_ptr[:-1]]]
        if special.size > 0:
            list_node = np.concatenate([list_node, inc_v[special]])
        all_w = _list_node_weights(all_col, all_ptr, list_node, pts, cen, nrm, all_w)

    # Panels with a row: at least three distinct corners and an area that
    # is not zero or negative.
    open_corners = np.bincount(c_pan[~closed[c_inc]], minlength=n_p)
    has_row = (n_corner >= 3) & ~(area <= 0.0)
    row_only = (
        kind_arr == ROW_GRADIENT_KIND
        if kind_arr is not None and kind_arr.shape[0] == n_p
        else np.zeros(n_p, dtype=bool)
    )
    nb_pair = ~cut[pair_key] & (pa != pb)
    nb_codes = np.unique(np.concatenate([pa[nb_pair] * n_p + pb[nb_pair], pb[nb_pair] * n_p + pa[nb_pair]]))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    # Interior panels: node values and Green-Gauss (Blazek 2015). A corner
    # weight component that is exactly zero gives no entry.
    gg = np.nonzero(has_row & (open_corners == 0))[0]
    if gg.size > 0:
        c_gg = corner[gg]
        corner_w = _green_gauss_corner_weights(pts, c_gg, n_corner[gg], nrm[gg], area[gg])
        g_slot, g_pos = np.nonzero(has_corner[gg])
        g_inc = np.searchsorted(inc_codes, c_gg[g_slot, g_pos] * n_p + gg[g_slot])
        owner, entry = _csr_expand(all_ptr, list_of_inc[g_inc])
        o_slot = g_slot[owner]
        o_pos = g_pos[owner]
        for comp in range(3):
            cw = corner_w[o_slot, o_pos, comp]
            use = cw != 0.0
            rows.append(3 * gg[o_slot[use]] + comp)
            cols.append(all_col[entry[use]])
            vals.append(cw[use] * all_w[entry[use]])

    # Border panels: edge-neighbour least squares, anchored at the panel value.
    ls = np.nonzero(has_row & (open_corners > 0))[0]
    if ls.size > 0:
        ls_rows, ls_cols, ls_vals = _least_squares_entries(
            ls, nb_codes // n_p, nb_codes % n_p, cen, nrm, n_p, row_only
        )
        rows.extend(ls_rows)
        cols.extend(ls_cols)
        vals.extend(ls_vals)
    return _gradient_matrix(rows, cols, vals, n_p)


def _build_surface_gradient_operator_reference(
    nodes: np.ndarray,
    panels: np.ndarray,
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    wake_edges: np.ndarray | None = None,
    kinds: np.ndarray | None = None,
    fold_angle_deg: float = FOLD_ANGLE_DEG,
    shepard: bool = False,
) -> scipy.sparse.csr_matrix:
    """Build the surface gradient operator with Python loops (reference).

    This is the first implementation of
    :func:`build_surface_gradient_operator`, with dictionaries, a
    union-find per node and one Python step per matrix entry. The solver
    does not call it. The parity tests compare the array version with it:
    the sparsity pattern must be the same and the values equal to
    round-off. *shepard* True gives the loop build of the previous
    operator (:func:`_build_surface_gradient_operator_shepard`).

    Parameters
    ----------
    nodes : numpy.ndarray
        Node coordinates with shape (Nn, 3) in m.
    panels : numpy.ndarray
        Corner node indices with shape (Np, 4). A triangle repeats a node.
    centroids : numpy.ndarray
        Panel centroids with shape (Np, 3) in m.
    normals : numpy.ndarray
        Panel unit normals with shape (Np, 3) [-].
    areas : numpy.ndarray
        Panel areas with shape (Np,) in m^2.
    wake_edges : numpy.ndarray or None, optional
        Wake shedding edges with shape (Nw, 4): upper panel, lower panel,
        node a, node b.
    kinds : numpy.ndarray or None, optional
        Panel kind codes with shape (Np,).
    fold_angle_deg : float, optional
        Largest angle between the normals of two panels across an edge
        that the gradient connects when the mesh has no kinds [deg].
    shepard : bool, optional
        True builds the previous operator (inverse-distance node values and
        Green-Gauss). Default False.

    Returns
    -------
    scipy.sparse.csr_matrix
        Operator with shape (3 Np, Np) in 1/m.
    """
    pts = np.asarray(nodes, dtype=float)
    pan = np.asarray(panels, dtype=int)
    cen = np.asarray(centroids, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    area = np.asarray(areas, dtype=float)
    n_p = pan.shape[0]

    w_rows = (
        np.empty((0, 4), dtype=int)
        if wake_edges is None
        else np.asarray(wake_edges, dtype=int).reshape(-1, 4)
    )
    wake_set: set[tuple[int, int]] = set()
    wake_nodes: set[int] = set()
    partner: dict[int, set[int]] = {}
    for up_i, lo_i, na, nb in w_rows:
        wake_set.add((min(int(na), int(nb)), max(int(na), int(nb))))
        wake_nodes.update((int(na), int(nb)))
        partner.setdefault(int(up_i), set()).add(int(lo_i))
        partner.setdefault(int(lo_i), set()).add(int(up_i))

    corners = [_unique_corners(pan[i]) for i in range(n_p)]
    edge_panels: dict[tuple[int, int], list[int]] = {}
    incident: dict[int, list[int]] = {}
    for i, vs in enumerate(corners):
        k = len(vs)
        for e in range(k):
            a, b = vs[e], vs[(e + 1) % k]
            if a == b:
                continue
            edge_panels.setdefault((min(a, b), max(a, b)), []).append(i)
        for v in vs:
            incident.setdefault(v, []).append(i)

    # Sharp edges: with panel kinds, an edge between panels of different
    # kinds; without them, an edge with a fold larger than fold_angle_deg.
    # The gradient treats them as wake edges (no values across them).
    kind_arr = None if kinds is None else np.asarray(kinds).reshape(-1)
    use_kinds = kind_arr is not None and kind_arr.shape[0] == n_p and np.unique(kind_arr).size > 1
    cos_fold = float(np.cos(np.radians(fold_angle_deg)))
    cut_set: set[tuple[int, int]] = set(wake_set)
    for key, ps in edge_panels.items():
        if key in cut_set or len(ps) < 2:
            continue
        for a_i in range(len(ps)):
            for b_i in range(a_i + 1, len(ps)):
                pa, pb = ps[a_i], ps[b_i]
                if use_kinds:
                    sharp = kind_arr[pa] != kind_arr[pb]
                else:
                    sharp = float(np.dot(nrm[pa], nrm[pb])) < cos_fold
                if sharp:
                    cut_set.add(key)

    # Panels on the other side of the wake for each trailing-edge panel.
    opposite: dict[int, set[int]] = {}
    for edge in wake_set:
        on_edge = edge_panels.get(edge, [])
        for i in on_edge:
            opposite.setdefault(i, set()).update(q for q in on_edge if q != i)
    for i, ps in partner.items():
        opposite.setdefault(i, set()).update(ps)

    # Groups of panels around each node (union-find over non-wake edges).
    group_of: dict[tuple[int, int], list[int]] = {}
    for v, ring in incident.items():
        parent = {p: p for p in ring}
        for p in ring:
            vs = corners[p]
            k = len(vs)
            for e in range(k):
                a, b = vs[e], vs[(e + 1) % k]
                if v not in (a, b) or a == b:
                    continue
                key = (min(a, b), max(a, b))
                if key in cut_set:
                    continue
                for q in edge_panels.get(key, []):
                    if q in parent:
                        ra, rb = _find_root(parent, p), _find_root(parent, q)
                        if ra != rb:
                            parent[rb] = ra
        members: dict[int, list[int]] = {}
        for p in ring:
            members.setdefault(_find_root(parent, p), []).append(p)
        for p in ring:
            group_of[(v, p)] = members[_find_root(parent, p)]

    # Edge-neighbour least-squares gradient of a border panel (no wake
    # edges): g_p = sum_j W_p[:, j] mu[idx_p[j]], with the minimum-norm
    # solution in the panel plane.
    edge_nbrs: list[set[int]] = [set() for _ in range(n_p)]
    for key, ps in edge_panels.items():
        if key in cut_set:
            continue
        for a in ps:
            for b in ps:
                if a != b:
                    edge_nbrs[a].add(b)
    ls_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def ls_weights(p: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the panel indices and the (3, k) weights of the gradient of p."""
        if p not in ls_cache:
            nb = np.asarray(sorted(edge_nbrs[p]), dtype=int)
            if nb.size == 0:
                ls_cache[p] = (np.array([p], dtype=int), np.zeros((3, 1)))
            else:
                dr = cen[nb] - cen[p]
                dr_t = dr - np.outer(dr @ nrm[p], nrm[p])
                # Displacement directions below LS_RCOND of the largest are
                # round-off (a row of collinear centroids gives the gradient
                # along the row). A tip cap keeps only the direction along its row.
                row = kind_arr is not None and kind_arr.shape[0] == n_p and kind_arr[p] == ROW_GRADIENT_KIND
                pinv = _stencil_pinv(dr_t, bool(row))
                idx = np.concatenate([nb, [p]])
                mat = np.hstack([pinv, -np.sum(pinv, axis=1, keepdims=True)])
                ls_cache[p] = (idx, mat)
        return ls_cache[p]

    def edges_at(p: int, v: int) -> list[tuple[int, int]]:
        """Return the edges of panel p at node v."""
        vs = corners[p]
        k = len(vs)
        out = []
        for e in range(k):
            a, b = vs[e], vs[(e + 1) % k]
            if a != b and v in (a, b):
                out.append((min(a, b), max(a, b)))
        return out

    node_cache: dict[tuple[int, tuple[int, ...]], tuple[np.ndarray, np.ndarray, bool]] = {}

    def node_weights(v: int, grp: list[int]) -> tuple[np.ndarray, np.ndarray, bool]:
        """Node weights of a group at node v, and True if the group closes."""
        key = (v, tuple(sorted(grp)))
        if key not in node_cache:
            g_idx = np.asarray(key[1], dtype=int)
            dist = np.linalg.norm(cen[g_idx] - pts[v], axis=1)
            w = 1.0 / np.maximum(dist, 1e-300)
            w = w / np.sum(w)
            if not shepard:
                # A panel with the node at two corners occurs two times in
                # the group; the fit uses each panel once and splits its
                # weight between the two entries.
                uniq, inverse, count = np.unique(g_idx, return_inverse=True, return_counts=True)
                w_ls, valid = _linear_node_weights(cen[uniq] - pts[v], nrm[uniq])
                if valid:
                    w = w_ls[inverse] / count[inverse]
            members = set(key[1])
            closed = True
            for p in key[1]:
                for edge in edges_at(p, v):
                    shared = edge not in cut_set and any(
                        q != p and q in members for q in edge_panels.get(edge, [])
                    )
                    if not shared:
                        closed = False
                        break
                if not closed:
                    break
            node_cache[key] = (g_idx, w, closed)
        return node_cache[key]

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n_p):
        vs = corners[i]
        k = len(vs)
        if k < 3 or area[i] <= 0.0:
            continue
        corner_nodes = []
        all_closed = True
        for v in vs:
            grp = group_of[(v, i)]
            if v in wake_nodes and i in opposite:
                excl = opposite[i]
                grp = [q for q in grp if q == i or q not in excl]
            g_idx, w, closed = node_weights(v, grp)
            corner_nodes.append((g_idx, w))
            all_closed = all_closed and closed
        if not all_closed:
            # Panel at a wake shedding edge or at a mesh border: edge-neighbour
            # least squares, anchored at the panel value.
            idx, mat = ls_weights(i)
            for comp in range(3):
                rows.extend([3 * i + comp] * idx.size)
                cols.extend(idx.tolist())
                vals.extend(mat[comp].tolist())
            continue
        n_i = nrm[i]
        corner_w = np.zeros((k, 3))
        for e in range(k):
            b = (e + 1) % k
            ev = np.cross(pts[vs[b]] - pts[vs[e]], n_i) * (0.5 / area[i])
            corner_w[e] += ev
            corner_w[b] += ev
        for kk in range(k):
            g_idx, w = corner_nodes[kk]
            for comp in range(3):
                cw = corner_w[kk, comp]
                if cw == 0.0:
                    continue
                rows.extend([3 * i + comp] * g_idx.size)
                cols.extend(g_idx.tolist())
                vals.extend((cw * w).tolist())
    op = scipy.sparse.csr_matrix(
        (np.asarray(vals, dtype=float), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
        shape=(3 * n_p, n_p),
    )
    op.sum_duplicates()
    return op


def surface_gradient(operator: scipy.sparse.spmatrix, mu: np.ndarray) -> np.ndarray:
    """Apply the surface gradient operator to one or more doublet vectors.

    Parameters
    ----------
    operator : scipy.sparse.spmatrix
        Operator of :func:`build_surface_gradient_operator` with shape
        (3 Np, Np) in 1/m.
    mu : numpy.ndarray
        Doublet strengths with shape (Np,) or (Np, K) in m^2/s.

    Returns
    -------
    numpy.ndarray
        Tangential gradients with shape (Np, 3) or (Np, 3, K) in m/s.
    """
    m = np.asarray(mu, dtype=float)
    out = operator @ m
    n_p = operator.shape[1]
    if m.ndim == 1:
        return np.asarray(out).reshape(n_p, 3)
    return np.asarray(out).reshape(n_p, 3, m.shape[1])


def compute_surface_velocity_and_cp(
    V_onset: np.ndarray,
    normals: np.ndarray,
    grad_mu: np.ndarray,
    V_inf: float,
    rates: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute local surface velocity and pressure coefficient on each panel.

    Parameters
    ----------
    V_onset : numpy.ndarray
        Onset velocity vector at panel centroids with shape (Np, 3) in m/s.
        With body rates it includes the rotation term
        ``-Omega x (x - x_ref)``.
    normals : numpy.ndarray
        Panel unit normal vectors with shape (Np, 3).
    grad_mu : numpy.ndarray
        Tangential gradient of doublet potential with shape (Np, 3) in m/s.
    V_inf : float
        Free-stream speed in m/s. At or below 1e-12 m/s, Cp is set to zero.
    rates : bool, default=False
        True when the flight condition has a body rate (p, q or r not
        zero). The pressure then includes the kinematic term of the
        rotating body (see Notes).

    Returns
    -------
    V_surf : numpy.ndarray
        Surface velocity vectors relative to the body with shape (Np, 3)
        in m/s.
    cp : numpy.ndarray
        Pressure coefficients on panels with shape (Np,).

    Notes
    -----
    Without body rates the flow is steady in the earth frame and the
    Bernoulli equation gives ``Cp = 1 - |V_surf|^2 / V_inf^2``.

    With body rates the flow is steady in the body frame (quasi-steady
    rotation), but not in the earth frame. The unsteady Bernoulli equation
    in the earth frame, ``p - p_inf = -rho (d(phi)/dt + |grad(phi)|^2 / 2)``,
    applies to the perturbation potential phi (the air is at rest far
    away). At a point fixed in the earth frame, ``d(phi)/dt = -v_b .
    grad(phi)``, with ``v_b`` the velocity of the body point there. With
    ``V_k = -v_b`` (the local onset velocity V_onset) and the velocity
    relative to the body ``V_surf = V_k + grad(phi)``, the result is::

        Cp = (|V_k|^2 - |V_surf|^2) / V_inf^2.

    Without rates ``|V_k| = V_inf`` and the two forms are the same; the
    first form is kept in that case, so results without rates do not
    change.
    """
    # Tangential component of onset flow: V_onset - (V_onset . n) * n
    v_norm = np.sum(V_onset * normals, axis=-1, keepdims=True)
    V_onset_t = V_onset - v_norm * normals

    # Surface velocity: V = V_onset_t - grad(mu)
    V_surf = V_onset_t - grad_mu

    speed_sq = np.sum(V_surf**2, axis=-1)
    if not V_inf > 1.0e-12:
        cp = np.zeros(V_onset.shape[0], dtype=float)
    elif rates:
        onset_sq = np.sum(np.asarray(V_onset) ** 2, axis=-1)
        cp = (onset_sq - speed_sq) / (V_inf**2)
    else:
        cp = 1.0 - speed_sq / (V_inf**2)

    return V_surf, cp


def integrate_forces_and_moments(
    cp: np.ndarray,
    areas: np.ndarray,
    normals: np.ndarray,
    centroids: np.ndarray,
    condition: FlightCondition,
    S_ref: float,
    b_ref: float,
    c_ref: float,
    ref_point: np.ndarray | None = None,
) -> dict[str, float]:
    """Integrate surface pressures into force and moment coefficients.

    Parameters
    ----------
    cp : numpy.ndarray
        Pressure coefficients on panels with shape (Np,).
    areas : numpy.ndarray
        Panel surface areas with shape (Np,) in m^2.
    normals : numpy.ndarray
        Panel outward unit normals with shape (Np, 3).
    centroids : numpy.ndarray
        Panel centroid coordinates with shape (Np, 3) in m.
    condition : FlightCondition
        Flight condition with alpha [rad], beta [rad], V_inf [m/s], rho [kg/m^3].
    S_ref : float
        Reference planform area in m^2.
    b_ref : float
        Reference span in m.
    c_ref : float
        Reference chord in m.
    ref_point : numpy.ndarray or None, optional
        Moment reference point with shape (3,) in m. Default is the origin.

    Returns
    -------
    dict
        Dictionary with CL, CD_pressure, CY, Cl, Cm, Cn.
    """
    rho = float(condition.rho)
    v_inf = float(condition.V_inf)
    q_inf = 0.5 * rho * v_inf**2

    # Panel force vector in geometry axes [N]: dF = -Cp * q_inf * Area * n
    panel_forces = -cp[:, None] * areas[:, None] * normals * q_inf
    total_force = np.sum(panel_forces, axis=0)

    # Reference directions
    alpha = float(condition.alpha)
    beta = float(condition.beta)
    u_hat = freestream_direction(alpha, beta)
    l_hat = lift_direction(alpha)
    y_hat = np.cross(l_hat, u_hat)

    # Aerodynamic force coefficients
    ref_force = q_inf * S_ref
    if ref_force > 1.0e-14:
        cl_val = float(np.dot(total_force, l_hat) / ref_force)
        cy_val = float(np.dot(total_force, y_hat) / ref_force)
        cd_val = float(np.dot(total_force, u_hat) / ref_force)
    else:
        cl_val = 0.0
        cy_val = 0.0
        cd_val = 0.0

    # Moments about reference point in geometry axes [N m]
    rp = np.zeros(3, dtype=float) if ref_point is None else np.asarray(ref_point, dtype=float)
    r_arm = centroids - rp[None, :]
    panel_moments = np.cross(r_arm, panel_forces)
    total_moment = np.sum(panel_moments, axis=0)

    # Convert to AVL flight-dynamics body axis moment coefficients
    if ref_force > 1.0e-14 and b_ref > 1.0e-14 and c_ref > 1.0e-14:
        c_roll = float(-total_moment[0] / (ref_force * b_ref))
        c_pitch = float(+total_moment[1] / (ref_force * c_ref))
        c_yaw = float(-total_moment[2] / (ref_force * b_ref))
    else:
        c_roll = 0.0
        c_pitch = 0.0
        c_yaw = 0.0

    return {
        "CL": cl_val,
        "CD_pressure": cd_val,
        "CY": cy_val,
        "Cl": c_roll,
        "Cm": c_pitch,
        "Cn": c_yaw,
    }


def _trefftz_plane_basis(wake_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the unit wake direction d and the in-plane axes e1, e2 (e1 x e2 = d).

    For d = x the axes are e1 = y and e2 = z, as in the body-axis wake.
    """
    d = np.asarray(wake_dir, dtype=float).reshape(3)
    d = d / np.linalg.norm(d)
    up = np.array([0.0, 0.0, 1.0])
    e2 = up - np.dot(up, d) * d
    if np.linalg.norm(e2) <= 1.0e-9:
        side = np.array([0.0, 1.0, 0.0])
        e2 = np.cross(d, side)
    e2 = e2 / np.linalg.norm(e2)
    e1 = np.cross(e2, d)
    e1 = e1 / np.linalg.norm(e1)
    return d, e1, e2


def _condition_arrays(conditions):
    """Return onset weights, speeds, densities and wind axes for K conditions.

    Speeds are in m/s, rates in rad/s, densities in kg/m^3 and axes are
    dimensionless. Weights have shape (K, 6); axes have shape (K, 3).
    """
    data = np.array([
        [c.alpha, c.beta, c.V_inf, c.rho, c.p, c.q, c.r] for c in conditions
    ], dtype=float)
    alpha, beta, speed, rho = data[:, :4].T
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    drag = np.column_stack([ca * cb, -sb, sa * cb])
    lift = np.column_stack([-sa, np.zeros_like(sa), ca])
    coeffs = np.column_stack([speed[:, None] * drag, data[:, 4:]])
    return coeffs, speed, rho, lift, drag


def _surface_velocity_batch(onset, normals, gradient, speed, rates=None):
    """Return velocity [m/s] (K, N, 3) and Cp [-] (K, N).

    Onset and gradient have shape (K, N, 3) in m/s. Normals have shape
    (N, 3) [-]. Speed has shape (K,) in m/s; Cp is zero for a speed at or
    below 1e-12 m/s. Rates has shape (K,) (bool), True for a condition with
    a body rate, or None for no rates. Use the formulas of
    :func:`compute_surface_velocity_and_cp` for each condition.
    """
    normal_speed = np.sum(onset * normals, axis=-1, keepdims=True)
    velocity = (onset - normal_speed * normals) - gradient
    valid = speed > 1e-12
    speed_sq = np.sum(velocity**2, axis=-1)
    cp = np.zeros(velocity.shape[:2])
    np.divide(speed_sq, speed[:, None] ** 2, out=cp, where=valid[:, None])
    cp = np.where(valid[:, None], 1.0 - cp, 0.0)
    if rates is not None and np.any(rates & valid):
        rows = np.flatnonzero(rates & valid)
        onset_sq = np.sum(onset[rows] ** 2, axis=-1)
        cp[rows] = (onset_sq - speed_sq[rows]) / speed[rows, None] ** 2
    return velocity, cp


def _force_coefficients_batch(force, moment, speed, rho, lift, drag, S_ref, b_ref, c_ref):
    """Return six coefficient arrays [-], one value per condition.

    Force and moment have shape (K, 3) in N and N m. Speed and density
    have shape (K,) in m/s and kg/m^3. Lift and drag are unit axes [-].
    S_ref is in m^2; b_ref and c_ref are in m. Follow the force and moment
    conventions of :func:`integrate_forces_and_moments`.
    """
    ref = 0.5 * rho * speed**2 * S_ref
    valid = ref > 1e-14
    out = {}
    for name, axis in (("CL", lift), ("CY", np.cross(lift, drag)), ("CD_pressure", drag)):
        out[name] = np.divide(np.sum(force * axis, axis=-1), ref,
                              out=np.zeros_like(ref), where=valid)
    valid_m = valid & (b_ref > 1e-14) & (c_ref > 1e-14)
    for j, name, length, sign in ((0, "Cl", b_ref, -1), (1, "Cm", c_ref, 1),
                                  (2, "Cn", b_ref, -1)):
        out[name] = np.divide(sign * moment[:, j], ref * length,
                              out=np.zeros_like(ref), where=valid_m)
    return out


def _integrate_batch(cp, geom, speed, rho, lift, drag, S_ref, b_ref, c_ref, ref_point):
    """Integrate Cp [-] (K, N) on geometry in m and m^2 into coefficients [-].

    Speed is in m/s and density in kg/m^3. Reference lengths and point
    are in m; reference area is in m^2. Wind axes are dimensionless.
    """
    q = 0.5 * rho * speed**2
    force = -cp[..., None] * geom.area[:, None] * geom.normal * q[:, None, None]
    moment = np.cross(geom.centroid - ref_point, force)
    return _force_coefficients_batch(
        np.sum(force, axis=1), np.sum(moment, axis=1), speed, rho, lift, drag,
        S_ref, b_ref, c_ref,
    )


def _trefftz_factors(wake, wake_dir=None, ground=None):
    """Return the shared Trefftz velocity factors in 1/m and strip lengths in m.

    The two velocity matrices have shape (Ns, Nv). The two in-plane
    normal components have shape (Ns,) [-]. Ground is a unit normal and
    an offset in m. Use the endpoint vortices and core of
    :func:`compute_trefftz_induced_drag` (Katz and Plotkin, chapter 12).
    """
    pl = np.array(wake.p_left, dtype=float, copy=True)
    pr = np.array(wake.p_right, dtype=float, copy=True)
    d = (np.array([1.0, 0.0, 0.0]) if wake_dir is None
         else np.asarray(wake_dir) / np.linalg.norm(wake_dir))
    if len(wake.panels):
        first = np.fromiter((idx[0] for idx in wake.strip_panel_indices), dtype=int)
        quad = np.asarray(wake.panels)[first]
        nw = np.cross(quad[:, 2] - quad[:, 0], quad[:, 3] - quad[:, 1])
        flip = np.sum((pr - pl) * np.cross(nw, d), axis=1) < 0.0
        pl[flip], pr[flip] = wake.p_right[flip], wake.p_left[flip]
    segment = pr - pl
    length = np.linalg.norm(segment, axis=-1)
    safe = np.maximum(length, 1e-12)
    vertices = np.vstack([pr, pl])
    if wake_dir is None and ground is None:
        q = (0.5 * (pl + pr))[:, 1:]
        pv = vertices[:, 1:]
        seg = segment[:, 1:]
    else:
        _, e1, e2 = _trefftz_plane_basis(d)
        basis = np.column_stack([e1, e2])
        q = (0.5 * (pl + pr)) @ basis
        seg = segment @ basis
        if ground is not None:
            normal, offset = ground
            height = vertices @ normal - offset
            vertices = np.vstack([vertices, vertices - 2.0 * height[:, None] * normal])
        pv = vertices @ basis
    normal = np.column_stack([-seg[:, 1], seg[:, 0]]) / safe[:, None]
    rq = q[:, None, :] - pv[None, :, :]
    rc_sq = (TREFFTZ_CORE_FRACTION * float(np.mean(safe))) ** 2
    denom = 2.0 * np.pi * (rq[..., 0] ** 2 + rq[..., 1] ** 2 + rc_sq)
    return -rq[..., 1] / denom, rq[..., 0] / denom, normal[:, 0], normal[:, 1], length


def _trefftz_batch(gamma, speed, rho, S_ref, factors, ground=False):
    """Return Trefftz drag coefficients [-] for circulation (K, Ns) in m^2/s.

    Speed and density have shape (K,) in m/s and kg/m^3. S_ref is in m^2.
    Factors are from :func:`_trefftz_factors`. Ground selects image vortices.
    """
    if gamma.shape[1] == 0 or S_ref <= 1e-12:
        return np.zeros(len(speed))
    gamma = np.ascontiguousarray(gamma)
    vy, vz, ny, nz, length = factors
    gv = np.concatenate([gamma, -gamma], axis=1)
    if ground:
        gv = np.concatenate([gv, -gv], axis=1)
    wash = (np.sum(gv[:, None, :] * vy[None], axis=-1) * ny
            + np.sum(gv[:, None, :] * vz[None], axis=-1) * nz)
    drag = -0.5 * rho * np.sum(gamma * wash * length, axis=1)
    ref = 0.5 * rho * speed**2 * S_ref
    return np.divide(drag, ref, out=np.zeros_like(speed), where=speed > 1e-12)


def compute_trefftz_induced_drag(
    wake: PanelWake,
    mu: np.ndarray,
    V_inf: float,
    rho: float,
    S_ref: float,
    *,
    wake_dir: np.ndarray | None = None,
    ground: tuple[np.ndarray, float] | None = None,
    wake_gamma: np.ndarray | None = None,
) -> float:
    """Compute far-field induced drag from trailing vortex sheet in Trefftz plane.

    Each wake strip carries the circulation Gamma_s = mu_upper - mu_lower
    (or *wake_gamma*). The trailing edges of the strips are projected on
    the Trefftz plane (normal to the wake direction); each strip end
    carries a 2D point vortex, +Gamma_s at the end with the larger
    projection on n_w x d (n_w the strip normal on the upper side, d the
    wake direction) and -Gamma_s at the other end. The normal wash at the strip midpoints gives
    D_i = -0.5 * rho * sum(Gamma_s * w_n,s * l_s). The vortices have a core
    of radius ``TREFFTZ_CORE_FRACTION`` times the mean strip length. In
    ground effect the image vortices (the ends reflected in the ground,
    with opposite circulation) add their normal wash (method of images).

    Parameters
    ----------
    wake : PanelWake
        Precomputed wake geometry.
    mu : numpy.ndarray
        Doublet strengths on body panels with shape (Np,) in m^2/s.
    V_inf : float
        Free-stream speed in m/s.
    rho : float
        Air density in kg/m^3.
    S_ref : float
        Reference area in m^2.
    wake_dir : numpy.ndarray or None, optional
        Wake direction with shape (3,) [-]. None is the body x axis.
    ground : tuple of (numpy.ndarray, float) or None, optional
        Ground plane (unit normal [-], offset [m]) with
        ``dot(x, normal) = offset``. None is free air.
    wake_gamma : numpy.ndarray or None, optional
        Strip circulations with shape (Ns,) in m^2/s. None uses
        ``mu[wake_up] - mu[wake_lo]``.

    Returns
    -------
    float
        Far-field induced drag coefficient CDi_trefftz.
    """
    n_strips = len(wake.wake_up)
    if n_strips == 0 or V_inf <= 1.0e-12 or S_ref <= 1.0e-12:
        return 0.0

    # Circulation of each wake strip: Gamma_s = mu_upper - mu_lower
    if wake_gamma is None:
        gamma_w = mu[wake.wake_up] - mu[wake.wake_lo]
    else:
        gamma_w = np.asarray(wake_gamma, dtype=float).reshape(n_strips)

    # Endpoints of strip segments in the Trefftz plane. The vortex of
    # circulation +Gamma about the wake direction d sits at the end with the
    # larger projection on n_w x d, where n_w is the normal of the wake strip
    # (the side of the upper trailing-edge panel). For a horizontal surface
    # with the upper side up this is the end with the larger y.
    pl = np.array(wake.p_left, dtype=float, copy=True)
    pr = np.array(wake.p_right, dtype=float, copy=True)
    d_hat = (
        np.array([1.0, 0.0, 0.0])
        if wake_dir is None
        else np.asarray(wake_dir, dtype=float).reshape(3) / np.linalg.norm(wake_dir)
    )
    if len(wake.panels) > 0:
        first = np.array([idx[0] for idx in wake.strip_panel_indices], dtype=int)
        quad = np.asarray(wake.panels, dtype=float)[first]
        n_w = np.cross(quad[:, 2] - quad[:, 0], quad[:, 3] - quad[:, 1])
        t_hat = np.cross(n_w, d_hat[None, :])
        flip = np.sum((pr - pl) * t_hat, axis=1) < 0.0
        if np.any(flip):
            pl[flip], pr[flip] = wake.p_right[flip], wake.p_left[flip]

    seg_vec = pr - pl
    seg_len = np.linalg.norm(seg_vec, axis=-1)
    safe_len = np.maximum(seg_len, 1.0e-12)

    if wake_dir is None and ground is None:
        # Body-axis wake in free air (the Trefftz plane is the y-z plane).
        # Segment midpoint evaluation points
        q_eval = 0.5 * (pl + pr)

        # In-plane normal vector: n = d_wake x seg_vec / seg_len
        # For body-axis wake d_wake = [1, 0, 0]:
        # [1, 0, 0] x [dx, dy, dz] = [0, -dz, dy]
        dy = seg_vec[:, 1]
        dz = seg_vec[:, 2]
        qn = np.column_stack([np.zeros_like(dy), -dz, dy]) / safe_len[:, None]

        # Trailing vortices at right endpoints (+gamma_w) and left endpoints (-gamma_w)
        pv = np.vstack([pr, pl])
        gv = np.concatenate([gamma_w, -gamma_w])

        # Core radius regularization to prevent zero-distance divisions
        rc = TREFFTZ_CORE_FRACTION * float(np.mean(safe_len))
        rc_sq = rc**2

        w_n = np.zeros(n_strips, dtype=np.float64)
        for s in range(n_strips):
            qs = q_eval[s]
            rq = qs[None, :] - pv
            r2 = rq[:, 1] ** 2 + rq[:, 2] ** 2
            denom = 2.0 * np.pi * (r2 + rc_sq)

            vy = -rq[:, 2] / denom
            vz = rq[:, 1] / denom

            v_y_ind = float(np.sum(gv * vy))
            v_z_ind = float(np.sum(gv * vz))
            w_n[s] = v_y_ind * qn[s, 1] + v_z_ind * qn[s, 2]

        # Induced drag: D_i = -0.5 * rho * sum(Gamma_s * w_n,s * l_s)
        d_ind = -0.5 * rho * float(np.sum(gamma_w * w_n * seg_len))
        q_inf = 0.5 * rho * V_inf**2
        return float(d_ind / (q_inf * S_ref))

    # General Trefftz plane normal to the wake direction, with ground images.
    _, e1, e2 = _trefftz_plane_basis(
        np.array([1.0, 0.0, 0.0]) if wake_dir is None else wake_dir
    )

    def _plane(points: np.ndarray) -> np.ndarray:
        return np.column_stack([points @ e1, points @ e2])

    seg2 = _plane(seg_vec)
    qn2 = np.column_stack([-seg2[:, 1], seg2[:, 0]]) / safe_len[:, None]
    q2 = _plane(0.5 * (pl + pr))
    pv3 = np.vstack([pr, pl])
    gv = np.concatenate([gamma_w, -gamma_w])
    if ground is not None:
        k_hat = np.asarray(ground[0], dtype=float).reshape(3)
        offset = float(ground[1])
        height = pv3 @ k_hat - offset
        pv3 = np.vstack([pv3, pv3 - 2.0 * height[:, None] * k_hat[None, :]])
        gv = np.concatenate([gv, -gv])
    pv2 = _plane(pv3)
    rc_sq = (TREFFTZ_CORE_FRACTION * float(np.mean(safe_len))) ** 2
    rq = q2[:, None, :] - pv2[None, :, :]
    denom = 2.0 * np.pi * (rq[..., 0] ** 2 + rq[..., 1] ** 2 + rc_sq)
    v1 = np.sum(-gv[None, :] * rq[..., 1] / denom, axis=1)
    v2 = np.sum(gv[None, :] * rq[..., 0] / denom, axis=1)
    w_n = v1 * qn2[:, 0] + v2 * qn2[:, 1]
    d_ind = -0.5 * rho * float(np.sum(gamma_w * w_n * seg_len))
    q_inf = 0.5 * rho * V_inf**2
    return float(d_ind / (q_inf * S_ref))
