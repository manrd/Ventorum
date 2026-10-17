# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Inviscid data of the viscous-inviscid coupling of Ventorum-2D.

For one airfoil and one angle of attack (free air or ground effect) this
module builds what the coupled Newton solver of :mod:`ventorum.panel2d.vii`
needs from the panel method of :mod:`ventorum.panel2d.solver`:

* the inviscid nodal vortex strengths gamma_inv;
* the wake line: it starts at the trailing edge (the midpoint of a blunt
  edge) along the bisector of the two trailing-edge panels and then follows
  the inviscid streamline (Heun predictor-corrector steps along the unit
  velocity), with geometrically growing steps; in ground effect the
  velocity field includes the mirror images;
* the source sensitivities: the change of gamma at the nodes and of the
  tangential velocity at the wake panel midpoints per unit source strength
  of every source panel (airfoil panels, the trailing-edge panel of a blunt
  edge, wake panels), with the factorization of the inviscid matrix reused
  for all source columns (Lighthill 1958 transpiration; Drela 1989).

References
----------
M. J. Lighthill, On displacement thickness, Journal of Fluid Mechanics 4,
1958, pp. 383-392.
M. Drela, XFOIL: An analysis and design system for low Reynolds number
airfoils, Lecture Notes in Engineering 54, Springer, 1989, pp. 1-12.
J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., 2001.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from ventorum.panel2d import solver as sv
from ventorum.panel2d.influence import (
    compute_panel_local_coordinates_numpy,
    linear_vortex_velocity_influence,
    source_panel_velocity_influence,
)

__all__ = ["CouplingSystem", "build_coupling_system", "wake_spacing", "WakeGroundError"]


class WakeGroundError(ValueError):
    """Raised when the traced wake line reaches the ground."""


@dataclass
class CouplingSystem:
    """Inviscid data of one coupled case.

    Attributes
    ----------
    alpha : float
        Angle of attack [deg].
    nodes : np.ndarray
        Node coordinates as solved (rotated and placed in ground effect),
        shape (N, 2) [m].
    nodes_ref : np.ndarray
        Node coordinates in the body frame, shape (N, 2) [m].
    layout : PanelLayout
        Panel index structure (one body).
    s : np.ndarray
        Arc length along the contour from node 0, shape (N,) [m].
    x_over_c : np.ndarray
        Body-frame chordwise position of the nodes over the reference chord
        [-], measured from the leading edge.
    gamma_inv : np.ndarray
        Inviscid nodal vortex strengths [-], shape (N,).
    gamma_src : np.ndarray
        d gamma / d sigma, shape (N, n_src): columns are the airfoil panels
        (in layout order, with the trailing-edge panel of a blunt edge) and
        then the wake panels.
    wake_nodes : np.ndarray
        Wake node coordinates, shape (Nw, 2) [m]; node 0 is the trailing
        edge.
    wake_len : np.ndarray
        Wake panel lengths, shape (Nw - 1,) [m].
    q_inv : np.ndarray
        Inviscid tangential velocity at the wake panel midpoints [-].
    q_src : np.ndarray
        d q / d sigma at the wake panel midpoints, shape (Nw - 1, n_src).
    panel_len : np.ndarray
        Airfoil panel lengths (layout order), shape (Np,) [m].
    te_gap : float
        Trailing-edge gap normal to the bisector [m] (0 for a sharp edge).
    chord_ref : float
        Reference chord [m].
    e_drag, e_lift : np.ndarray
        Drag and lift directions in the solved frame.
    ground : bool
        True in ground effect.
    x_ref : tuple of float
        Moment reference point in the body frame [m]: ``moment_ref`` of
        :func:`build_coupling_system`, by default the quarter-chord point
        on the chord line of the reference geometry (see
        :func:`ventorum.panel2d.solver.reference_point`).
    """

    alpha: float
    nodes: np.ndarray
    nodes_ref: np.ndarray
    layout: sv.PanelLayout
    s: np.ndarray
    x_over_c: np.ndarray
    gamma_inv: np.ndarray
    gamma_src: np.ndarray
    wake_nodes: np.ndarray
    wake_len: np.ndarray
    q_inv: np.ndarray
    q_src: np.ndarray
    panel_len: np.ndarray
    te_gap: float
    chord_ref: float
    e_drag: np.ndarray
    e_lift: np.ndarray
    ground: bool
    x_ref: tuple = (0.25, 0.0)

    @property
    def n_nodes(self) -> int:
        """Return the number of airfoil nodes."""
        return len(self.nodes)

    @property
    def n_wake(self) -> int:
        """Return the number of wake nodes."""
        return len(self.wake_nodes)


def wake_spacing(first: float, total: float, n: int) -> np.ndarray:
    """Return n wake panel lengths that grow geometrically from ``first``.

    The ratio r solves first (r^n - 1) / (r - 1) = total. When n equal
    panels of length ``first`` already exceed ``total``, the panels are
    equal (total / n).

    Parameters
    ----------
    first : float
        Length of the first panel [m].
    total : float
        Wake length [m].
    n : int
        Number of panels.

    Returns
    -------
    np.ndarray
        Panel lengths [m], shape (n,).
    """
    if first * n >= total:
        return np.full(n, total / n)

    def f(r):
        return first * (r**n - 1.0) / (r - 1.0) - total

    lo, hi = 1.0 + 1e-12, 2.0
    while f(hi) < 0.0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    r = 0.5 * (lo + hi)
    d = first * r ** np.arange(n)
    return d * (total / d.sum())


def _velocity_matrices(field, nodes, layout, wake_nodes, ground):
    """Return the global velocity at field points per unit gamma and sigma.

    Returns (Vg, Vs): Vg of shape (Nc, 2, N) per unit nodal gamma, Vs of
    shape (Nc, 2, n_src) per unit source strength of the airfoil and wake
    panels.
    """
    ps, pe = sv.panel_ends(nodes, layout)
    wps, wpe = wake_nodes[:-1], wake_nodes[1:]

    def panel_vel(p_s, p_e, vortex):
        xi, eta, L, _ = compute_panel_local_coordinates_numpy(field, p_s, p_e)
        d = p_e - p_s
        ln = np.hypot(d[:, 0], d[:, 1])
        tx = (d[:, 0] / ln)[None, :]
        ty = (d[:, 1] / ln)[None, :]
        if vortex:
            u1, v1, u2, v2 = linear_vortex_velocity_influence(xi, eta, L)
            g1 = np.stack([u1 * tx - v1 * ty, u1 * ty + v1 * tx], 1)
            g2 = np.stack([u2 * tx - v2 * ty, u2 * ty + v2 * tx], 1)
            return g1, g2
        u, v = source_panel_velocity_influence(xi, eta, L)
        return np.stack([u * tx - v * ty, u * ty + v * tx], 1)

    p1, p2 = sv._incidence(np, layout, None)
    g1, g2 = panel_vel(ps, pe, True)
    vs_body = panel_vel(ps, pe, False)
    vs_wake = panel_vel(wps, wpe, False)
    if ground:
        mps, mpe = sv._mirror(ps), sv._mirror(pe)
        h1, h2 = panel_vel(mps, mpe, True)
        g1 = g1 - h1
        g2 = g2 - h2
        vs_body = vs_body + panel_vel(mps, mpe, False)
        vs_wake = vs_wake + panel_vel(sv._mirror(wps), sv._mirror(wpe), False)
    vg = g1 @ p1 + g2 @ p2
    for b, p in enumerate(layout.te_panel):
        if p >= 0:
            s0, e0 = layout.node_ranges[b]
            col = layout.te_kv[b] * (g1[..., p] + g2[..., p]) + layout.te_ks[b] * vs_body[..., p]
            vg[..., s0] += col
            vg[..., e0 - 1] -= col
    return vg, np.concatenate([vs_body, vs_wake], -1)


def _vortex_velocity(points, nodes, layout, gamma, ground, v_inf):
    """Return the inviscid velocity at points (free stream plus the gamma sheets)."""
    vg, _ = _velocity_matrices(points, nodes, layout, np.array([[1.0, 0.0], [2.0, 0.0]]), ground)
    v = vg @ gamma
    return v + v_inf[None, :]


def _trace_wake(nodes, layout, gamma, ground, v_inf, lengths):
    """Return the wake nodes along the inviscid streamline from the trailing edge."""
    n = len(nodes)
    te = 0.5 * (nodes[0] + nodes[n - 1])
    tu = nodes[0] - nodes[1]
    tl = nodes[n - 1] - nodes[n - 2]
    bis = tu / np.hypot(*tu) + tl / np.hypot(*tl)
    bis = bis / np.hypot(*bis)
    pts = [te, te + lengths[0] * bis]
    for k in range(1, len(lengths)):
        p = pts[-1]
        d1 = _vortex_velocity(p[None], nodes, layout, gamma, ground, v_inf)[0]
        d1 = d1 / np.hypot(*d1)
        q = p + lengths[k] * d1
        d2 = _vortex_velocity(q[None], nodes, layout, gamma, ground, v_inf)[0]
        d2 = d2 / np.hypot(*d2)
        d = d1 + d2
        pts.append(p + lengths[k] * d / np.hypot(*d))
    w = np.array(pts)
    if ground and np.any(w[:, 1] <= 0.0):
        raise WakeGroundError("The traced wake line reaches the ground.")
    return w


def build_coupling_system(body, alpha_deg: float, *, height=None, height_reference="pivot", pivot=None,
                          chord_ref=None, wake_length: float = 1.0, n_wake: int = 30,
                          moment_ref=None) -> CouplingSystem:
    """Build the inviscid data of one coupled case.

    Parameters
    ----------
    body : np.ndarray or AirfoilShape
        Airfoil coordinates in Selig order, shape (N, 2) [m], or a shape
        (its reference geometry gives the defaults, see
        :func:`ventorum.panel2d.solver.resolve_reference`).
    alpha_deg : float
        Angle of attack [deg].
    height : float or None
        Height above the ground [m] (pivot or trailing-edge reference), or
        None for free air.
    height_reference : {"pivot", "trailing_edge"}
        Reference point of the height.
    pivot : tuple of float or None
        Rotation point [m]; default the quarter-chord point on the chord
        line of the reference geometry (see
        :func:`ventorum.panel2d.solver.reference_point`).
    chord_ref : float or None
        Reference chord [m] of the coefficients, of the Reynolds number,
        of x/c and of the wake length; default the reference chord of the
        body (see :func:`ventorum.panel2d.solver.section_reference`): for
        a flapped shape the chord of the undeflected section, else the
        chord of the body as given.
    wake_length : float
        Wake length over the reference chord [-].
    n_wake : int
        Number of wake panels.
    moment_ref : tuple of float or None
        Moment reference point (x, y) [m] in the body frame; default the
        quarter-chord point on the chord line of the reference geometry.

    Returns
    -------
    CouplingSystem
        The inviscid data.
    """
    chord_ref, x_ref, quarter = sv.resolve_reference(body, chord_ref, moment_ref)
    b = sv._as_numpy(body)
    layout = sv.panel_layout([b])
    xle = float(np.min(b[:, 0]))
    if pivot is None:
        pivot = quarter
    ground = height is not None
    a = np.radians(alpha_deg)
    if ground:
        nodes = sv.place_in_ground_effect([b], alpha_deg, height, height_reference, pivot)[0]
        v_inf = np.array([1.0, 0.0])
        rhs = -nodes[:, 1]
        e_drag = np.array([1.0, 0.0])
    else:
        nodes = b
        v_inf = np.array([np.cos(a), np.sin(a)])
        rhs = np.cos(a) * (-nodes[:, 1]) + np.sin(a) * nodes[:, 0]
        e_drag = v_inf.copy()
    e_lift = np.array([-e_drag[1], e_drag[0]])
    A = sv.system_matrix(nodes, layout, ground=ground)
    lu = scipy.linalg.lu_factor(A)
    n = layout.n_nodes
    gamma_inv = scipy.linalg.lu_solve(lu, sv.full_rhs(rhs, layout))[:n]
    ps, pe = sv.panel_ends(nodes, layout)
    panel_len = np.hypot(pe[:, 0] - ps[:, 0], pe[:, 1] - ps[:, 1])
    first = 0.5 * (panel_len[0] + panel_len[n - 2])
    lengths = wake_spacing(first, wake_length * chord_ref, n_wake)
    wake = _trace_wake(nodes, layout, gamma_inv, ground, v_inf, lengths)
    psi_src = np.concatenate([
        sv.body_source_stream_matrix(nodes, layout, ground=ground),
        sv.wake_source_stream_matrix(nodes, wake, ground=ground),
    ], -1)
    gamma_src = scipy.linalg.lu_solve(lu, sv.full_rhs(-psi_src, layout, columns=True))[:n]
    mids = 0.5 * (wake[:-1] + wake[1:])
    dw = wake[1:] - wake[:-1]
    wl = np.hypot(dw[:, 0], dw[:, 1])
    tw = dw / wl[:, None]
    vg, vs = _velocity_matrices(mids, nodes, layout, wake, ground)
    q_inv = np.einsum("kc,kcn,n->k", tw, vg, gamma_inv) + tw @ v_inf
    q_src = np.einsum("kc,kcn,nm->km", tw, vg, gamma_src) + np.einsum("kc,kcm->km", tw, vs)
    tu = nodes[0] - nodes[1]
    tl = nodes[n - 1] - nodes[n - 2]
    bis = tu / np.hypot(*tu) + tl / np.hypot(*tl)
    bis = bis / np.hypot(*bis)
    gap_vec = nodes[0] - nodes[n - 1]
    te_gap = 0.0 if layout.sharp[0] else float(abs(gap_vec[0] * bis[1] - gap_vec[1] * bis[0]))
    seg = np.hypot(np.diff(nodes[:, 0]), np.diff(nodes[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return CouplingSystem(
        alpha=float(alpha_deg),
        nodes=nodes,
        nodes_ref=b,
        layout=layout,
        s=s,
        x_over_c=(b[:, 0] - xle) / chord_ref,
        gamma_inv=gamma_inv,
        gamma_src=gamma_src,
        wake_nodes=wake,
        wake_len=wl,
        q_inv=q_inv,
        q_src=q_src,
        panel_len=panel_len,
        te_gap=te_gap,
        chord_ref=float(chord_ref),
        e_drag=e_drag,
        e_lift=e_lift,
        ground=ground,
        x_ref=x_ref,
    )
