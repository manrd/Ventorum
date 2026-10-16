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
doublet values are first interpolated to the corner nodes with inverse
distance weights (Shepard 1968), and the gradient follows from the node
values with the Green-Gauss theorem on the panel polygon (Blazek 2015,
chapter 5). The node values smooth the doublet error of the thin
triangle fans at the poles of latitude-longitude meshes, where the
edge-neighbour fit is weak. On a panel with a corner on a wake shedding
edge or on a mesh border, a node mean would sit about half a panel away
from the node, and a node mean is blind to a spanwise saw-tooth of the
trailing-edge doublets; these panels use the edge-neighbour least-squares
fit of :func:`compute_surface_gradient`, anchored at the panel value. No
step differentiates across the doublet jump of the wake.

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
  Applications, 3rd ed., Butterworth-Heinemann, chapter 5 (Green-Gauss
  gradient).

Conventions
-----------
* Surface velocity: V = V_onset_tangential - grad(mu).
* Pressure coefficient: Cp = 1 - |V|^2 / V_inf^2.
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
       inverse-distance mean of the group values (Shepard 1968, weight
       1 / d with d the distance from the node to the panel centroid).
       The Green-Gauss theorem on the panel polygon then gives
       ``grad_i = (1 / A_i) * sum_e 0.5 * (mu_a + mu_b) * (p_b - p_a) x n_i``
       over the edges ``e = (a, b)`` (Blazek 2015, chapter 5). The cross
       product with the unit normal keeps the gradient in the panel plane.
    3. Border panels (a corner group is open). The gradient is the
       edge-neighbour least-squares fit of :func:`compute_surface_gradient`
       in the panel plane, anchored at the panel value, over the panels
       that share an edge that is not cut. It is exact for
       a linear doublet field, and the trailing-edge pressure stays
       sensitive to the strength of each wake strip, which the pressure
       Kutta iteration needs.

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
                # Displacement directions below 1e-5 of the largest are round-off
                # (a row of collinear centroids gives the gradient along the row).
                pinv = np.linalg.pinv(dr_t, rcond=1e-5)
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
        """Inverse-distance weights of a group at node v, and True if the group closes."""
        key = (v, tuple(sorted(grp)))
        if key not in node_cache:
            g_idx = np.asarray(key[1], dtype=int)
            dist = np.linalg.norm(cen[g_idx] - pts[v], axis=1)
            w = 1.0 / np.maximum(dist, 1e-300)
            w = w / np.sum(w)
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
) -> tuple[np.ndarray, np.ndarray]:
    """Compute local surface velocity and pressure coefficient on each panel.

    Parameters
    ----------
    V_onset : numpy.ndarray
        Onset velocity vector at panel centroids with shape (Np, 3) in m/s.
    normals : numpy.ndarray
        Panel unit normal vectors with shape (Np, 3).
    grad_mu : numpy.ndarray
        Tangential gradient of doublet potential with shape (Np, 3) in m/s.
    V_inf : float
        Free-stream speed in m/s.

    Returns
    -------
    V_surf : numpy.ndarray
        Surface velocity vectors with shape (Np, 3) in m/s.
    cp : numpy.ndarray
        Pressure coefficients on panels with shape (Np,).
    """
    # Tangential component of onset flow: V_onset - (V_onset . n) * n
    v_norm = np.sum(V_onset * normals, axis=-1, keepdims=True)
    V_onset_t = V_onset - v_norm * normals

    # Surface velocity: V = V_onset_t - grad(mu)
    V_surf = V_onset_t - grad_mu

    speed_sq = np.sum(V_surf**2, axis=-1)
    if V_inf > 1.0e-12:
        cp = 1.0 - speed_sq / (V_inf**2)
    else:
        cp = np.zeros(V_onset.shape[0], dtype=float)

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
