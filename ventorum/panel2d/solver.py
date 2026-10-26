# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Ventorum-2D: two-dimensional inviscid panel solver with ground effect.

Linear-strength vortex panels with node values gamma_j and constant-strength
source panels (prescribed strengths sigma, default zero) on one or more
bodies. Boundary condition: constant stream function on each body, applied
at the nodes (Dirichlet), with the stream-function constant of each body as
an extra unknown. Kutta condition per body: gamma_first + gamma_last = 0.
At a sharp trailing edge the first and last nodes coincide and their two
stream-function equations are the same; the equation of the last node is
then replaced by an extrapolation condition: the second differences of
gamma at the two trailing-edge ends add to zero (Drela 1989, section 2;
the exact form is to be checked against the paper, see
``docs/theory/panel2d.md``). A blunt trailing edge is closed by a panel from
the last node to the first node.

In free air the matrix does not depend on the angle of attack: it is
factorized once and two right-hand sides (free stream along x and along y)
are superposed. In ground effect (flat ground at y = 0) each body has a
mirror image with opposite vortex strength and equal source strength; the
angle of attack rotates the geometry about a pivot and the free stream is
along +x.

References
----------
J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., Cambridge
University Press, 2001, chapter 11 (linear-strength vortex method and the
stream-function formulation) and section 6.6 (method of images).
M. Drela, XFOIL: An analysis and design system for low Reynolds number
airfoils, Lecture Notes in Engineering 54, Springer, 1989, pp. 1-12
(the paper only).
H. Glauert, Theoretical relationships for an aerofoil with hinged flap,
ARC R and M 1095, 1927 (the chord of the undeflected section is the
reference chord of a flapped section).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import torch

from ventorum.airfoil.geometry import AirfoilShape, SectionReference, leading_edge_index
from ventorum.panel2d.influence import (
    compute_panel_local_coordinates_numpy,
    compute_panel_local_coordinates_torch,
    image_source_stream_influence,
    linear_vortex_stream_influence,
    source_panel_stream_influence,
    wake_source_stream_influence,
)

__all__ = ["Panel2DResult", "solve_inviscid", "section_chord", "section_reference", "reference_point",
           "resolve_reference"]

#: Trailing-edge gap below which a trailing edge is sharp, over the body chord [-].
SHARP_TE_GAP = 1e-4


@dataclass
class Panel2DResult:
    """Aerodynamic results of the 2D panel method.

    Attributes
    ----------
    alpha : float, np.ndarray, or torch.Tensor
        Angle of attack [deg].
    Cl : float, np.ndarray, or torch.Tensor
        Lift coefficient [-] (in free air from the total circulation,
        in ground effect from surface pressure integration).
    Cl_circulation : float, np.ndarray, or torch.Tensor
        Lift coefficient from the total circulation (Kutta-Joukowski,
        2 Gamma / c) [-].
    Cl_pressure : float, np.ndarray, or torch.Tensor
        Lift coefficient from surface pressure integration [-] (wall panels
        only: the trailing-edge gap panel of a blunt edge is not a wall).
    Cd_pressure : float, np.ndarray, or torch.Tensor
        Pressure drag coefficient from surface pressure integration [-]
        (wall panels only).
    Cm : float, np.ndarray, or torch.Tensor
        Pitching moment coefficient about the moment reference point
        (``moment_ref``; by default the quarter-chord point on the chord
        line of the reference geometry, see :func:`reference_point`; nose
        up positive, in the body-0 frame before rotation) [-].
    Cp : np.ndarray, torch.Tensor, or list
        Pressure coefficients at the nodes [-].
    ue : np.ndarray, torch.Tensor, or list
        Surface tangential velocity over the free-stream speed at the nodes
        [-] (signed as gamma: positive when the flow turns clockwise).
    gamma : np.ndarray, torch.Tensor, or list
        Nodal vortex strengths over the free-stream speed [-].
    coordinates : np.ndarray, torch.Tensor, or list
        Node coordinates as solved [m] (rotated and placed in ground effect).
    circulation : float, np.ndarray, or torch.Tensor
        Circulation per body over the free-stream speed [m].
    """

    alpha: float | np.ndarray | torch.Tensor
    Cl: float | np.ndarray | torch.Tensor
    Cl_circulation: float | np.ndarray | torch.Tensor
    Cl_pressure: float | np.ndarray | torch.Tensor
    Cd_pressure: float | np.ndarray | torch.Tensor
    Cm: float | np.ndarray | torch.Tensor
    Cp: np.ndarray | torch.Tensor | list
    ue: np.ndarray | torch.Tensor | list
    gamma: np.ndarray | torch.Tensor | list
    coordinates: np.ndarray | torch.Tensor | list
    circulation: float | np.ndarray | torch.Tensor


# ---------------------------------------------------------------------------
# Geometry and assembly (shared by the inviscid and the viscous solvers)
# ---------------------------------------------------------------------------


@dataclass
class PanelLayout:
    """Index structure of the panels of one or more bodies.

    Attributes
    ----------
    node_ranges : list of tuple
        (first, last + 1) node index of each body.
    sharp : list of bool
        True for a sharp trailing edge (coincident end nodes).
    idx1, idx2 : np.ndarray
        Start and end node index of each panel (a blunt trailing edge adds
        a panel from the last node to the first node).
    panel_body : np.ndarray
        Body index of each panel.
    te_panel : list of int
        Index of the trailing-edge panel of each body, or -1 when sharp.
    te_kv, te_ks : list of float
        Coefficients of the trailing-edge gap panel of each body (zero when
        sharp): its constant vortex strength is te_kv (gamma_first -
        gamma_last) and its constant source strength te_ks (gamma_first -
        gamma_last), with te_kv = -(s . t)/2 and te_ks = (t . n_out)/2, s the
        unit vector along the gap panel (last node to first node), t the
        unit trailing-edge bisector and n_out the outer normal of the gap
        panel (Drela 1989; signs derived for this convention).
    """

    node_ranges: list
    sharp: list
    idx1: np.ndarray
    idx2: np.ndarray
    panel_body: np.ndarray
    te_panel: list
    te_kv: list
    te_ks: list

    @property
    def n_nodes(self) -> int:
        """Return the total number of nodes."""
        return self.node_ranges[-1][1]

    @property
    def n_bodies(self) -> int:
        """Return the number of bodies."""
        return len(self.node_ranges)

    @property
    def n_panels(self) -> int:
        """Return the total number of panels."""
        return len(self.idx1)


def panel_layout(bodies_np: list) -> PanelLayout:
    """Return the panel index structure of a list of (N, 2) bodies.

    Parameters
    ----------
    bodies_np : list of np.ndarray
        Body coordinates in Selig order [m].

    Returns
    -------
    PanelLayout
        The index structure.
    """
    ranges, sharp, i1, i2, pb, te, kv, ks = [], [], [], [], [], [], [], []
    start = 0
    n_pan = 0
    for b_id, b in enumerate(bodies_np):
        n = len(b)
        chord = float(np.max(b[:, 0]) - np.min(b[:, 0]))
        gap = float(np.hypot(*(b[0] - b[-1])))
        is_sharp = gap <= SHARP_TE_GAP * (chord if chord > 0.0 else 1.0)
        ranges.append((start, start + n))
        sharp.append(is_sharp)
        i1.append(np.arange(start, start + n - 1))
        i2.append(np.arange(start + 1, start + n))
        pb.append(np.full(n - 1, b_id))
        n_pan += n - 1
        if is_sharp:
            te.append(-1)
            kv.append(0.0)
            ks.append(0.0)
        else:
            i1.append(np.array([start + n - 1]))
            i2.append(np.array([start]))
            pb.append(np.array([b_id]))
            te.append(n_pan)
            n_pan += 1
            tu = b[0] - b[1]
            tl = b[-1] - b[-2]
            t = tu / np.hypot(*tu) + tl / np.hypot(*tl)
            t = t / np.hypot(*t)
            sv_ = (b[0] - b[-1]) / gap
            n_out = np.array([sv_[1], -sv_[0]])
            kv.append(-0.5 * float(sv_ @ t))
            ks.append(0.5 * float(t @ n_out))
        start += n
    return PanelLayout(ranges, sharp, np.concatenate(i1), np.concatenate(i2), np.concatenate(pb), te, kv, ks)


def section_chord(body) -> float:
    """Return the chord of a section as given: the distance from the leading edge to the trailing-edge midpoint.

    The trailing-edge midpoint is the midpoint of the first and the last
    point. The leading edge is the point farthest from it, the definition
    of the airfoil package (:func:`ventorum.airfoil.geometry.leading_edge_index`).
    For a section with the leading edge at (0, 0) and the trailing-edge
    midpoint at (1, 0) the chord is 1. A plain flap moves the trailing
    edge: the chord of a deflected section is shorter than the chord of
    the section before the deflection. The reference chord of the solvers
    comes from :func:`resolve_reference`: a flapped
    :class:`ventorum.airfoil.AirfoilShape` holds the chord of the section
    before the deflection.

    Parameters
    ----------
    body : array_like
        Section coordinates in Selig order, shape (N, 2) [m].

    Returns
    -------
    float
        Chord [m]; 1.0 for a section of zero size.
    """
    pts = np.asarray(body, dtype=np.float64)
    mid = 0.5 * (pts[0] + pts[-1])
    chord = float(np.linalg.norm(pts[leading_edge_index(pts)] - mid))
    return chord if chord > 0.0 else 1.0


def _as_numpy(body) -> np.ndarray:
    """Return the coordinates of a body (array, tensor or shape) as a float64 numpy array."""
    if isinstance(body, AirfoilShape):
        return body.coords
    if isinstance(body, torch.Tensor):
        return body.detach().cpu().numpy().astype(np.float64)
    return np.asarray(body, dtype=np.float64)


def section_reference(body) -> SectionReference:
    """Return the reference geometry of a section.

    Parameters
    ----------
    body : array_like or AirfoilShape
        Section coordinates in Selig order, shape (N, 2) [m], or a shape.

    Returns
    -------
    SectionReference
        The stored reference geometry of a shape (for example the section
        before a plain-flap deflection, see
        :meth:`ventorum.airfoil.AirfoilShape.with_flap`), else the one of
        the section as given: the leading edge (the point farthest from
        the trailing-edge midpoint), the trailing-edge midpoint and their
        distance (:func:`section_chord`) [m].
    """
    if isinstance(body, AirfoilShape):
        return body.reference_geometry()
    return SectionReference.from_coords(_as_numpy(body))


def reference_point(body) -> tuple[float, float]:
    """Return the quarter-chord point on the chord line of the reference geometry [m].

    The point lies 0.25 of the way from the leading edge to the
    trailing-edge midpoint of the reference geometry
    (:func:`section_reference`). For a section with the leading edge at
    (0, 0) and the trailing-edge midpoint at (1, 0) it is (0.25, 0); for a
    shape with a deflected plain flap it is the quarter-chord point of the
    section before the deflection. It is the default moment reference
    point and the default pivot of the ground-effect placement.

    Parameters
    ----------
    body : array_like or AirfoilShape
        Section coordinates, shape (N, 2) [m], or a shape.

    Returns
    -------
    tuple of float
        (x, y) of the point [m].
    """
    return section_reference(body).quarter_chord


def resolve_reference(body, chord_ref=None, moment_ref=None):
    """Return the reference chord, the moment reference point and the default pivot of a section.

    Explicit values win over everything. The defaults come from the
    reference geometry of the section (:func:`section_reference`): its
    reference chord, and the quarter-chord point on its chord line
    (:func:`reference_point`).

    Parameters
    ----------
    body : array_like or AirfoilShape
        Section coordinates, shape (N, 2) [m], or a shape.
    chord_ref : float or None
        Reference chord [m], positive and finite, or None for the default.
    moment_ref : tuple of float or None
        Moment reference point (x, y) [m] in the section frame, or None
        for the default.

    Returns
    -------
    chord_ref : float
        Reference chord [m].
    moment_ref : tuple of float
        Moment reference point (x, y) [m].
    pivot : tuple of float
        Default pivot of the ground-effect placement: the quarter-chord
        point on the chord line of the reference geometry [m] (it does not
        follow an explicit ``moment_ref``).

    Raises
    ------
    ValueError
        If ``chord_ref`` is not positive and finite, or ``moment_ref`` is
        not a point of two finite numbers.
    """
    ref = section_reference(body)
    pivot = ref.quarter_chord
    if chord_ref is None:
        chord = float(ref.chord_ref)
    else:
        chord = float(chord_ref)
        if not np.isfinite(chord) or chord <= 0.0:
            raise ValueError(f"chord_ref must be a positive length, got {chord_ref}.")
    if moment_ref is None:
        point = pivot
    else:
        point = check_point(moment_ref, "moment_ref")
    return chord, point, pivot


def _moment_alias(x_ref, moment_ref):
    """Return the moment reference point from ``moment_ref`` and its former name ``x_ref``."""
    if x_ref is None:
        return moment_ref
    if moment_ref is None:
        return x_ref
    if check_point(x_ref, "x_ref") != check_point(moment_ref, "moment_ref"):
        raise ValueError(f"x_ref {x_ref!r} and moment_ref {moment_ref!r} differ; x_ref is the former name of "
                         "moment_ref: give one of them.")
    return moment_ref


def check_point(value, name: str) -> tuple[float, float]:
    """Return a point (x, y) as two floats; raise ValueError when it is not two finite numbers.

    Parameters
    ----------
    value : array_like
        The point (x, y) [m].
    name : str
        Argument name for the error message.

    Returns
    -------
    tuple of float
        (x, y) [m].
    """
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a point (x, y) of two finite numbers, got {value!r}.") from None
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a point (x, y) of two finite numbers, got {value!r}.")
    return (float(arr[0]), float(arr[1]))


def _xp(a):
    """Return the array module of ``a``."""
    return torch if isinstance(a, torch.Tensor) else np


def _local(xp, field, ps, pe):
    """Return the local panel coordinates of field points."""
    if xp is np:
        return compute_panel_local_coordinates_numpy(field, ps, pe)
    return compute_panel_local_coordinates_torch(field, ps, pe)


def _incidence(xp, layout: PanelLayout, like):
    """Return the panel-to-node incidence matrices of the linear vortex panels.

    The trailing-edge gap panels carry no linear vortex (see
    :class:`PanelLayout`); their rows are zero.
    """
    npn, nn = layout.n_panels, layout.n_nodes
    p1 = np.zeros((npn, nn))
    p2 = np.zeros((npn, nn))
    p1[np.arange(npn), layout.idx1] = 1.0
    p2[np.arange(npn), layout.idx2] = 1.0
    for p in layout.te_panel:
        if p >= 0:
            p1[p] = 0.0
            p2[p] = 0.0
    if xp is np:
        return p1, p2
    return (torch.as_tensor(p1, dtype=like.dtype, device=like.device),
            torch.as_tensor(p2, dtype=like.dtype, device=like.device))


def panel_ends(nodes, layout: PanelLayout):
    """Return the start and end points of all panels.

    Parameters
    ----------
    nodes : array
        All node coordinates, shape (..., N, 2) [m].
    layout : PanelLayout
        Index structure.

    Returns
    -------
    tuple
        ``(p_start, p_end)``, each of shape (..., Np, 2) [m].
    """
    xp = _xp(nodes)
    if xp is np:
        return nodes[..., layout.idx1, :], nodes[..., layout.idx2, :]
    i1 = torch.as_tensor(layout.idx1, device=nodes.device)
    i2 = torch.as_tensor(layout.idx2, device=nodes.device)
    return nodes[..., i1, :], nodes[..., i2, :]


def _mirror(p):
    """Return the mirror image of points about y = 0."""
    q = p * 1.0
    q[..., 1] = -q[..., 1]
    return q


def vortex_stream_matrix(nodes, layout: PanelLayout, field=None, ground: bool = False):
    """Return the stream function at field points per unit nodal gamma.

    Linear vortex panels between the nodes, and the vortex and source
    sheets of the trailing-edge gap panels (strengths from the two
    trailing-edge nodes, see :class:`PanelLayout`).

    Parameters
    ----------
    nodes : array
        Node coordinates, shape (..., N, 2) [m].
    layout : PanelLayout
        Index structure.
    field : array, optional
        Field points, shape (..., Nc, 2) [m]. Default: the nodes.
    ground : bool
        Add the mirror images (opposite vortex strength) about y = 0.

    Returns
    -------
    array
        Matrix of shape (..., Nc, N) [m].
    """
    xp = _xp(nodes)
    if field is None:
        field = nodes
    ps, pe = panel_ends(nodes, layout)
    p1, p2 = _incidence(xp, layout, nodes)
    xi, eta, L, _ = _local(xp, field, ps, pe)
    eta = xp.where(abs(eta) < 1e-12 * L, 0.0 * eta + 0.0, eta)
    a1, a2 = linear_vortex_stream_influence(xi, eta, L)
    if ground:
        xi, eta, L, _ = _local(xp, field, _mirror(ps), _mirror(pe))
        b1, b2 = linear_vortex_stream_influence(xi, eta, L)
        a1 = a1 - b1
        a2 = a2 - b2
    out = a1 @ p1 + a2 @ p2
    if any(p >= 0 for p in layout.te_panel):
        src = body_source_stream_matrix(nodes, layout, field=field, ground=ground)
        out = out * 1.0
        for b, p in enumerate(layout.te_panel):
            if p < 0:
                continue
            s, e = layout.node_ranges[b]
            col = layout.te_kv[b] * (a1[..., :, p] + a2[..., :, p]) + layout.te_ks[b] * src[..., :, p]
            out[..., :, s] = out[..., :, s] + col
            out[..., :, e - 1] = out[..., :, e - 1] - col
    return out


def body_source_stream_matrix(nodes, layout: PanelLayout, field=None, ground: bool = False):
    """Return the stream function at field points per unit body-panel source strength.

    The real panels use the branch of
    :func:`ventorum.panel2d.influence.source_panel_stream_influence` (cut
    along the outward normal); the mirror images (equal strength) use the
    branch of :func:`ventorum.panel2d.influence.image_source_stream_influence`.

    Parameters
    ----------
    nodes : array
        Node coordinates, shape (..., N, 2) [m].
    layout : PanelLayout
        Index structure.
    field : array, optional
        Field points, shape (..., Nc, 2) [m]. Default: the nodes.
    ground : bool
        Add the mirror images about y = 0.

    Returns
    -------
    array
        Matrix of shape (..., Nc, Np) [m].
    """
    xp = _xp(nodes)
    if field is None:
        field = nodes
    ps, pe = panel_ends(nodes, layout)
    xi, eta, L, _ = _local(xp, field, ps, pe)
    out = source_panel_stream_influence(xi, eta, L)
    if ground:
        out = out + image_source_stream_influence(field, _mirror(ps), _mirror(pe))
    return out


def wake_source_stream_matrix(field, wake_nodes, ground: bool = False):
    """Return the stream function at field points per unit wake-panel source strength.

    Parameters
    ----------
    field : array
        Field points, shape (..., Nc, 2) [m].
    wake_nodes : array
        Wake node coordinates from the trailing edge downstream, shape
        (..., Nw, 2) [m]; panel k joins nodes k and k + 1.
    ground : bool
        Add the mirror images (equal strength) about y = 0.

    Returns
    -------
    array
        Matrix of shape (..., Nc, Nw - 1) [m].
    """
    xp = _xp(field)
    ps = wake_nodes[..., :-1, :]
    pe = wake_nodes[..., 1:, :]
    xi, eta, L, _ = _local(xp, field, ps, pe)
    out = wake_source_stream_influence(xi, eta, L)
    if ground:
        out = out + image_source_stream_influence(field, _mirror(ps), _mirror(pe))
    return out


def system_matrix(nodes, layout: PanelLayout, ground: bool = False):
    """Return the influence matrix of the stream-function formulation.

    Unknowns: the nodal gamma (N values) and the stream-function constant of
    each body. Rows: the stream-function condition at each node (the last
    node of a body with a sharp trailing edge holds the extrapolation
    condition instead), then the Kutta condition of each body.

    Parameters
    ----------
    nodes : array
        Node coordinates, shape (..., N, 2) [m].
    layout : PanelLayout
        Index structure.
    ground : bool
        Ground effect by mirror images.

    Returns
    -------
    array
        Matrix of shape (..., N + n_bodies, N + n_bodies).
    """
    xp = _xp(nodes)
    nn = layout.n_nodes
    nb = layout.n_bodies
    a_g = vortex_stream_matrix(nodes, layout, ground=ground)
    batch = tuple(a_g.shape[:-2])
    if xp is np:
        A = np.zeros(batch + (nn + nb, nn + nb))
    else:
        A = torch.zeros(batch + (nn + nb, nn + nb), dtype=nodes.dtype, device=nodes.device)
    A[..., :nn, :nn] = a_g
    for b, (s, e) in enumerate(layout.node_ranges):
        A[..., s:e, nn + b] = -1.0
        A[..., nn + b, s] = 1.0
        A[..., nn + b, e - 1] = 1.0
        if layout.sharp[b]:
            A[..., s, :] = 0.0
            A[..., s, s] = 1.0
            A[..., s, s + 1] = -2.0
            A[..., s, s + 2] = 1.0
            A[..., e - 1, :] = 0.0
            A[..., e - 1, e - 1] = 1.0
            A[..., e - 1, e - 2] = -2.0
            A[..., e - 1, e - 3] = 1.0
    return A


def condition_rows(layout: PanelLayout) -> np.ndarray:
    """Return a 0/1 mask of the node rows that hold a stream-function condition.

    Parameters
    ----------
    layout : PanelLayout
        Index structure.

    Returns
    -------
    np.ndarray
        Mask of shape (N,): 0 at the extrapolation rows of sharp trailing
        edges, 1 elsewhere.
    """
    m = np.ones(layout.n_nodes)
    for b, (s, e) in enumerate(layout.node_ranges):
        if layout.sharp[b]:
            m[s] = 0.0
            m[e - 1] = 0.0
    return m


def full_rhs(psi_rhs, layout: PanelLayout, columns: bool = False):
    """Return the full right-hand side from stream-function values at the nodes.

    Parameters
    ----------
    psi_rhs : array
        Right-hand side of the node rows, shape (..., N), or (..., N, k)
        with ``columns=True``.
    layout : PanelLayout
        Index structure.
    columns : bool
        True when ``psi_rhs`` holds k right-hand-side columns.

    Returns
    -------
    array
        Right-hand side with zero extrapolation and Kutta rows, shape
        (..., N + n_bodies) or (..., N + n_bodies, k).
    """
    xp = _xp(psi_rhs)
    mask = condition_rows(layout)
    if xp is torch:
        mask = torch.as_tensor(mask, dtype=psi_rhs.dtype, device=psi_rhs.device)
    if columns:
        z_shape = tuple(psi_rhs.shape[:-2]) + (layout.n_bodies, psi_rhs.shape[-1])
        body = psi_rhs * mask[:, None]
        axis = -2
    else:
        z_shape = tuple(psi_rhs.shape[:-1]) + (layout.n_bodies,)
        body = psi_rhs * mask
        axis = -1
    if xp is np:
        return np.concatenate([body, np.zeros(z_shape)], axis)
    return torch.cat([body, torch.zeros(z_shape, dtype=psi_rhs.dtype, device=psi_rhs.device)], axis)


def strip_check(nodes_np: np.ndarray, layout: PanelLayout) -> None:
    """Raise ValueError when a node lies in the outer strip of a source panel of another body.

    The source stream function of a body panel is single-valued only
    outside the strip {eta < 0, 0 < xi < L} on its outer side (see
    :func:`ventorum.panel2d.influence.source_panel_stream_influence`). For a
    single airfoil no node lies there; for several bodies (for example a
    slotted flap below a main element) one may, and then the prescribed
    sources are not supported.

    Parameters
    ----------
    nodes_np : np.ndarray
        Node coordinates, shape (N, 2) [m].
    layout : PanelLayout
        Index structure.
    """
    if layout.n_bodies < 2:
        return
    ps, pe = panel_ends(nodes_np, layout)
    xi, eta, L, _ = compute_panel_local_coordinates_numpy(nodes_np, ps, pe)
    node_body = np.zeros(layout.n_nodes, dtype=int)
    for b, (s, e) in enumerate(layout.node_ranges):
        node_body[s:e] = b
    other = node_body[:, None] != layout.panel_body[None, :]
    bad = other & (eta < -1e-12 * L) & (xi > 0.0) & (xi < L)
    if np.any(bad):
        raise ValueError(
            "A node of one body lies in the outer strip of a source panel of another body; "
            "prescribed source strengths are not supported for this multi-element layout."
        )


# ---------------------------------------------------------------------------
# Inviscid solver
# ---------------------------------------------------------------------------


def _body_list(bodies):
    """Return the bodies as a list."""
    if isinstance(bodies, (np.ndarray, torch.Tensor, AirfoilShape)):
        return [bodies]
    if isinstance(bodies, (list, tuple)):
        return list(bodies)
    raise TypeError("bodies must be an (N, 2) array, an AirfoilShape or a list of them.")


def place_in_ground_effect(bodies_np, alpha_deg, height, height_reference, pivot):
    """Rotate bodies by alpha about the pivot and place them at the height.

    Parameters
    ----------
    bodies_np : list of np.ndarray
        Body coordinates [m].
    alpha_deg : float
        Angle of attack [deg] (nose up rotates the geometry clockwise).
    height : float
        Height above the ground [m].
    height_reference : {"pivot", "trailing_edge"}
        Point whose height is ``height``: the pivot or the lowest
        trailing-edge point.
    pivot : tuple of float
        Rotation point (x, y) [m].

    Returns
    -------
    list of np.ndarray
        Rotated and placed bodies [m].

    Raises
    ------
    ValueError
        If the geometry is under or touches the ground, or the reference is
        unknown.
    """
    px, py = pivot
    a = np.radians(alpha_deg)
    ca, sa = np.cos(a), np.sin(a)
    rot = []
    for b in bodies_np:
        dx = b[:, 0] - px
        dy = b[:, 1] - py
        rot.append(np.column_stack([px + dx * ca + dy * sa, py - dx * sa + dy * ca]))
    if height_reference == "pivot":
        dy_shift = height - py
    elif height_reference == "trailing_edge":
        dy_shift = height - float(min([b[0, 1] for b in rot] + [b[-1, 1] for b in rot]))
    else:
        raise ValueError(f"Unknown height_reference '{height_reference}'.")
    out = [np.column_stack([b[:, 0], b[:, 1] + dy_shift]) for b in rot]
    for b in out:
        if np.min(b[:, 1]) <= 0.0:
            raise ValueError("Geometry under or touching the ground")
    return out


def _sigma_vector(sigma, n_panels, xp, like=None):
    """Return the source strengths as one vector, or None when absent or zero."""
    if sigma is None:
        return None
    if xp is np:
        v = np.concatenate([np.asarray(s, dtype=np.float64).ravel() for s in sigma]) \
            if isinstance(sigma, (list, tuple)) else np.asarray(sigma, dtype=np.float64).ravel()
        if v.size != n_panels:
            raise ValueError(f"sigma must hold {n_panels} panel values, got {v.size}.")
        return None if np.all(v == 0.0) else v
    if isinstance(sigma, (list, tuple)):
        v = torch.cat([torch.as_tensor(s, dtype=like.dtype, device=like.device).reshape(-1) for s in sigma])
    else:
        v = torch.as_tensor(sigma, dtype=like.dtype, device=like.device).reshape(-1)
    if v.numel() != n_panels:
        raise ValueError(f"sigma must hold {n_panels} panel values, got {v.numel()}.")
    return None if not bool(torch.any(v != 0.0)) else v


def _loads(xp, gamma, nodes, nodes_ref, layout, chord_ref, x_ref, alpha_rad, ground):
    """Return Cl, Cl_pressure, Cd_pressure, Cm and the circulation per body.

    ``gamma`` has shape (..., N). ``nodes`` are the solved coordinates,
    ``nodes_ref`` the body-frame coordinates before rotation (moment).
    """
    ps, pe = panel_ends(nodes, layout)
    i1 = layout.idx1 if xp is np else torch.as_tensor(layout.idx1, device=gamma.device)
    i2 = layout.idx2 if xp is np else torch.as_tensor(layout.idx2, device=gamma.device)
    g1 = gamma[..., i1]
    g2 = gamma[..., i2]
    dx = pe[..., 0] - ps[..., 0]
    dy = pe[..., 1] - ps[..., 1]
    lens = (dx * dx + dy * dy) ** 0.5
    dcirc = 0.5 * (g1 + g2) * lens
    for b, p in enumerate(layout.te_panel):
        if p >= 0:
            s_, e_ = layout.node_ranges[b]
            dcirc[..., p] = layout.te_kv[b] * (gamma[..., s_] - gamma[..., e_ - 1]) * lens[..., p]
    pb = np.zeros((layout.n_panels, layout.n_bodies))
    pb[np.arange(layout.n_panels), layout.panel_body] = 1.0
    if xp is torch:
        pb = torch.as_tensor(pb, dtype=gamma.dtype, device=gamma.device)
    circ = dcirc @ pb
    cl = 2.0 * circ.sum(-1) / chord_ref
    cp = 1.0 - gamma * gamma
    # Panel mean of Cp = 1 - gamma^2 for gamma linear on the panel (exact
    # integral of the pressure of the linear-vorticity solution).
    cpp = 1.0 - (g1 * g1 + g1 * g2 + g2 * g2) / 3.0
    # The trailing-edge gap panel is not a wall (it carries the gap sheets);
    # its pressure is not integrated.
    wall = np.ones(layout.n_panels)
    for p in layout.te_panel:
        if p >= 0:
            wall[p] = 0.0
    if xp is torch:
        wall = torch.as_tensor(wall, dtype=gamma.dtype, device=gamma.device)
    cpp = cpp * wall
    fx = -cpp * dy
    fy = cpp * dx
    sx = fx.sum(-1) / chord_ref
    sy = fy.sum(-1) / chord_ref
    if ground:
        cl_p, cd_p = sy, sx
    else:
        ca, sa = xp.cos(alpha_rad), xp.sin(alpha_rad)
        cl_p = sy * ca - sx * sa
        cd_p = sy * sa + sx * ca
    qs, qe = panel_ends(nodes_ref, layout)
    rdx = qe[..., 0] - qs[..., 0]
    rdy = qe[..., 1] - qs[..., 1]
    xm = 0.5 * (qs[..., 0] + qe[..., 0])
    ym = 0.5 * (qs[..., 1] + qe[..., 1])
    rfx = -cpp * rdy
    rfy = cpp * rdx
    cm = (-(xm - x_ref[0]) * rfy + (ym - x_ref[1]) * rfx).sum(-1) / chord_ref**2
    return cl, cl_p, cd_p, cm, circ, cp


def solve_inviscid(
    bodies,
    alpha_deg,
    *,
    height: float | None = None,
    height_reference: str = "pivot",
    pivot: tuple[float, float] | None = None,
    sigma=None,
    x_ref: tuple[float, float] | None = None,
    chord_ref: float | None = None,
    moment_ref: tuple[float, float] | None = None,
    backend: str = "numpy",
    device=None,
    dtype=None,
) -> Panel2DResult:
    """Solve 2D inviscid flow over one or more bodies, with ground effect.

    Parameters
    ----------
    bodies : np.ndarray, torch.Tensor, AirfoilShape, or list of them
        One or more bodies, each an (N, 2) array in Selig order [m] or an
        :class:`ventorum.airfoil.AirfoilShape`. The reference geometry of
        body 0 (:func:`section_reference`) gives the default reference
        chord, moment reference point and pivot; a shape with a deflected
        plain flap holds the geometry of the section before the
        deflection (:meth:`ventorum.airfoil.AirfoilShape.with_flap`).
    alpha_deg : float, list of float, np.ndarray, or torch.Tensor
        Angle of attack [deg].
    height : float, optional
        Height above flat ground at y = 0 [m]. None solves in free air.
    height_reference : {"pivot", "trailing_edge"}, optional
        Height reference point. Default "pivot".
    pivot : tuple of float, optional
        Pivot point (x, y) [m] for the geometry rotation. Default: the
        quarter-chord point on the chord line of the reference geometry of
        body 0 (see :func:`reference_point`). An explicit ``moment_ref``
        does not move it.
    sigma : array or list of arrays, optional
        Prescribed constant source strengths of all panels (over the
        free-stream speed) [-], in panel order (a blunt trailing edge adds
        one panel per body, last). Default None (zero). Sources change only
        the right-hand side.
    x_ref : tuple of float, optional
        Former name of ``moment_ref``, kept as an alias: give one of them
        (both only when they are equal).
    chord_ref : float, optional
        Reference chord [m] of the coefficients, positive. It wins over
        the reference geometry. Default: the reference chord of body 0
        (see :func:`section_reference`): the chord of the section before
        the deflection for a flapped shape, as in thin-airfoil theory
        (Glauert 1927); else the chord of body 0 as given (distance from
        the leading edge to the trailing-edge midpoint, see
        :func:`section_chord`). A flapped section given as an array has a
        shorter chord than the undeflected section: give ``chord_ref``,
        or give the flapped shape.
    moment_ref : tuple of float, optional
        Moment reference point (x, y) [m] in the body-0 frame before
        rotation. It wins over the reference geometry. Default: the
        quarter-chord point on the chord line of the reference geometry
        of body 0, 0.25 of the way from its leading edge to its
        trailing-edge midpoint (see :func:`reference_point`).
    backend : {"numpy", "torch"}, optional
        Computational backend. Default "numpy".
    device : str or torch.device, optional
        Torch device ("cpu" or "cuda").
    dtype : torch.dtype, optional
        Torch data type. Default float64.

    Returns
    -------
    Panel2DResult
        Aerodynamic results and node distributions.

    Raises
    ------
    ValueError
        If the backend is unknown, ``chord_ref`` is not positive and
        finite, ``moment_ref`` (or ``x_ref``) is not a point of two finite
        numbers, or ``x_ref`` and ``moment_ref`` differ.
    """
    if backend not in ("numpy", "torch"):
        raise ValueError(f"Unknown backend '{backend}'. Supported backends: 'numpy', 'torch'.")
    blist = _body_list(bodies)
    bodies_np = [_as_numpy(b) for b in blist]
    layout = panel_layout(bodies_np)
    moment_ref = _moment_alias(x_ref, moment_ref)
    chord_ref, x_ref, quarter = resolve_reference(blist[0], chord_ref, moment_ref)
    if pivot is None:
        pivot = quarter
    is_scalar = np.ndim(alpha_deg) == 0 if not isinstance(alpha_deg, torch.Tensor) else alpha_deg.ndim == 0
    alphas = np.atleast_1d(np.asarray(
        alpha_deg.detach().cpu().numpy() if isinstance(alpha_deg, torch.Tensor) else alpha_deg, dtype=np.float64))
    ground = height is not None
    nodes_ref = np.vstack(bodies_np)
    if ground:
        placed = [np.vstack(place_in_ground_effect(bodies_np, a, height, height_reference, pivot))
                  for a in alphas]
        nodes_np = np.stack(placed)
    else:
        nodes_np = nodes_ref[None]
    if sigma is not None and layout.n_bodies > 1:
        for k in range(nodes_np.shape[0]):
            strip_check(nodes_np[k], layout)

    if backend == "numpy":
        xp = np
        nodes = nodes_np
        ref = nodes_ref
        sig = _sigma_vector(sigma, layout.n_panels, np)
    else:
        xp = torch
        if device is None:
            device = "cpu"
        if dtype is None:
            dtype = torch.float64
        nodes = torch.as_tensor(nodes_np, dtype=dtype, device=device)
        ref = torch.as_tensor(nodes_ref, dtype=dtype, device=device)
        sig = _sigma_vector(sigma, layout.n_panels, torch, like=nodes)
    nn = layout.n_nodes
    if not ground:
        A = system_matrix(nodes[0], layout)
        rhs = xp.stack([-nodes[0, :, 1], nodes[0, :, 0]], -1)
        if sig is not None:
            src = body_source_stream_matrix(nodes[0], layout) @ sig
            rhs = xp.concatenate([rhs, -src[:, None]], -1) if xp is np else torch.cat([rhs, -src[:, None]], -1)
        rhs = full_rhs(rhs, layout, columns=True)
        if xp is np:
            lu = scipy.linalg.lu_factor(A)
            sol = scipy.linalg.lu_solve(lu, rhs)
        else:
            sol = torch.linalg.solve(A, rhs)
        a_rad = np.radians(alphas)
        if xp is np:
            ca, sa = np.cos(a_rad), np.sin(a_rad)
        else:
            a_t = torch.as_tensor(a_rad, dtype=nodes.dtype, device=nodes.device)
            ca, sa = torch.cos(a_t), torch.sin(a_t)
        sols = ca[:, None] * sol[None, :, 0] + sa[:, None] * sol[None, :, 1]
        if sig is not None:
            sols = sols + sol[None, :, 2]
        gamma = sols[:, :nn]
        alpha_r = ca * 0.0 + (a_rad if xp is np else a_t)
        cl, clp, cdp, cm, circ, cp = _loads(xp, gamma, nodes[0], ref, layout, chord_ref, x_ref, alpha_r, False)
        coords_out = [nodes_ref[s:e] for s, e in layout.node_ranges] if xp is np else \
            [ref[s:e] for s, e in layout.node_ranges]
        coords_out = coords_out[0] if layout.n_bodies == 1 else coords_out
        coords_per_alpha = None
    else:
        A = system_matrix(nodes, layout, ground=True)
        psi = -nodes[..., 1]
        if sig is not None:
            psi = psi - body_source_stream_matrix(nodes, layout, ground=True) @ sig
        rhs = full_rhs(psi, layout)
        if xp is np:
            sol = np.linalg.solve(A, rhs[..., None])[..., 0]
        else:
            sol = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
        gamma = sol[:, :nn]
        cl, clp, cdp, cm, circ, cp = _loads(xp, gamma, nodes, ref, layout, chord_ref, x_ref, None, True)
        coords_per_alpha = [[nodes[k, s:e] for s, e in layout.node_ranges] for k in range(len(alphas))]
        coords_per_alpha = [c[0] if layout.n_bodies == 1 else c for c in coords_per_alpha]
        coords_out = coords_per_alpha
    return _pack(xp, alphas, is_scalar, layout, cl, clp, cdp, cm, circ, cp, gamma,
                 coords_out if not ground else coords_per_alpha, ground)


def _pack(xp, alphas, is_scalar, layout, cl, clp, cdp, cm, circ, cp, gamma, coords, ground):
    """Return the Panel2DResult of a solve."""
    nb = layout.n_bodies

    def split(arr, k=None):
        a = arr if k is None else arr[k]
        if nb == 1:
            return a
        return [a[..., s:e] for s, e in layout.node_ranges]

    if xp is np:
        def num(v):
            return float(v)
    else:
        def num(v):
            return v
    if is_scalar:
        circ0 = circ[0] if nb > 1 else (float(circ[0, 0]) if xp is np else circ[0, 0])
        return Panel2DResult(
            alpha=float(alphas[0]) if xp is np else torch.as_tensor(alphas[0], dtype=gamma.dtype,
                                                                     device=gamma.device),
            Cl=num(clp[0]) if ground else num(cl[0]),
            Cl_circulation=num(cl[0]),
            Cl_pressure=num(clp[0]),
            Cd_pressure=num(cdp[0]),
            Cm=num(cm[0]),
            Cp=split(cp, 0),
            ue=split(gamma, 0),
            gamma=split(gamma, 0),
            coordinates=coords[0] if ground else coords,
            circulation=circ0,
        )
    alpha_out = alphas if xp is np else torch.as_tensor(alphas, dtype=gamma.dtype, device=gamma.device)
    return Panel2DResult(
        alpha=alpha_out,
        Cl=clp if ground else cl,
        Cl_circulation=cl,
        Cl_pressure=clp,
        Cd_pressure=cdp,
        Cm=cm,
        Cp=split(cp),
        ue=split(gamma),
        gamma=split(gamma),
        coordinates=coords,
        circulation=circ if nb > 1 else circ[:, 0],
    )
