# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Closed quadrilateral meshes of thick lifting surfaces for the 3D panel method.

The mesh wraps a thick section around each lifting surface. Each spanwise
station carries the airfoil shape (upper surface, lower surface, flat tip
cap; an open trailing edge is closed by default, see below). The panels are
plane quadrilaterals (or triangles) with outward corner order, and the
trailing-edge strips carry the wake shedding edges for the solver.

Finite trailing edge
--------------------
A section with an open (blunt) trailing edge is closed by default
(``trailing_edge="close"``). The upper and the lower ordinates move toward
the mid point of the trailing-edge gap g by (g / 2) x^4 (x the chordwise
station in unit chord), so the camber line does not move and the thickness
changes by at most the gap. For the NACA 4-digit and 5-digit thickness
law this is exactly the published closed trailing edge: the coefficient of
x^4 changes from -0.1015 to -0.1036 (Ladson, Brooks, Hill and Sproles 1996),
and the difference of the two laws is the half gap times x^4.

The reason: with constant-strength panels and the internal Dirichlet
condition, a solid base between the two trailing-edge corners has no
fine-mesh limit that satisfies the Kutta condition at both corners. With
the wake shed from one corner and the Morino condition, the circulation
flows around the base from the other corner when the trailing-edge panels
become shorter than the base height, and the lift falls with chordwise
refinement toward a limit well below the lift of the closed section. Other
wake attachments (the other corner, the base centre, both
corners), a base without its Dirichlet row and a base with outflow give
the same circulation, because the potential outside the base region does
not change; an explicit velocity or pressure Kutta condition on the
trailing-edge panels makes the lift rise without limit, because the
corner flows are singular. ``trailing_edge="base"`` keeps the former
blunt base (base panels, wake from the upper corner) for comparisons.

Section placement
-----------------
Each spanwise station uses the frame of the Ventorum vortex-lattice builder
(:mod:`ventorum.geometry.lattice`): the same leading-edge reference line,
the same spanwise stations, and the same twist rotation about the local
spanwise axis (the untwisted reference line in the y-z plane) through the
quarter-chord point. The section lies in the plane spanned by the twisted
chord direction and the section normal, which is the cross product of the
untwisted chord direction (x) and the local spanwise axis, twisted with the
chord. This normal is the normal of the lattice surface at that station
(``z`` for a flat wing, ``-y`` for a vertical fin with the span up). So the
section thickness follows the dihedral, and the leading-edge and
trailing-edge nodes are the lattice points: the thin-surface limit matches
the vortex lattice exactly. At the root of a symmetric surface the lattice
uses the y axis as the twist axis, so the root section lies in the plane
y = 0 with the normal z.

Root section of a symmetric surface
-----------------------------------
The root section in the plane y = 0 is the cut of the two half wings at
the plane of symmetry (a miter joint). With the dihedral G of the first
panel row, a thickness t along z is only t cos(G) normal to the local
surface. The option ``scale_root_thickness`` (default True) multiplies the
offsets of the root section from its camber line, along the section normal
in the plane y = 0, by 1 / cos(G). The thickness normal to the local
surface then equals the section thickness. The camber line, and so the
leading-edge and trailing-edge nodes, do not move. For a symmetric section
without root twist, the first panel row is then an exact prism: the volume
of the half wing equals the volume of the same wing with zero dihedral to
round-off. A cambered section keeps a small volume difference: the miter
plane cuts the cambered slab at an angle, so the difference is about
sin(G) times the first moment of the section area about the chord line
(the integral of thickness times camber). With root twist T the thickness
normal to the surface is t cos(G) / sqrt(1 - sin(T)^2 sin(G)^2) before
the scaling, so the scaled thickness has a second-order error in T. The
option has no effect on a surface with zero dihedral at the root, on a
one-sided surface (its root section is normal to its span axis, for
example a vertical fin) and on a surface whose first panel row is
vertical (the two halves coincide and there is no miter). The factor is
not capped. When the builder applies it with a first-row dihedral above
``ROOT_SCALE_WARN_DIHEDRAL_DEG`` (60 deg, a factor above 2), it issues a
UserWarning: the root section is strongly scaled and the result needs a
check.

Fold guard
----------
Two neighbour section planes that are not parallel cross on a line parallel
to x. At the root of a symmetric surface with dihedral G the root plane
(y = 0) and the next plane (normal to the span) cross at a distance of
about d / tan(G) from the reference line, with d the width of the root
strip. When the section reaches past that line (thickness t > d / tan(G)),
the nodes of the next station lie beyond the root plane and the strip
folds. With the cosine spanwise spacing this happens when the panel count
grows, because the root strip becomes narrow. The same can happen at a
dihedral break. With ``fold_guard=True`` (default)
the builder turns each station plane toward the plane of the station
before it, from the root outward, only as far as needed to keep every
node of each of the two sections at least ``FOLD_MARGIN`` times the
station spacing away from the plane of the other section. A guarded
station keeps its leading- and trailing-edge nodes (the lattice points);
only its thickness direction turns, and with ``scale_root_thickness`` its
offsets from the camber line grow by 1 / cos(B), with B the angle between
the guarded plane and the plane normal to the span, as at the root. A
station that does not fold keeps its plane bit for bit.

Published sources
-----------------
* Hess, J. L. and Smith, A. M. O., 1964, Calculation of non-lifting
  potential flow about arbitrary three-dimensional bodies, Journal of Ship
  Research, Vol. 8, No. 2, pp. 22-44 (quadrilateral source panels on a
  closed body, outward normal convention).
* Hess, J. L. and Smith, A. M. O., 1967, Calculation of potential flow about
  arbitrary bodies, Progress in Aeronautical Sciences, Vol. 8, pp. 1-138
  (panel layout of closed bodies and lifting surfaces).
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed., Cambridge
  University Press, chapters 10 to 12 (panel corner order, plane
  quadrilateral panels, signed volume of a closed mesh).
* Glauert, H., 1926, The Elements of Aerofoil and Airscrew Theory, Cambridge
  University Press, Chapter VII (thin-airfoil theory of a hinged flap,
  flap effectiveness factor).
* Glauert, H., 1927, Theoretical relationships for an aerofoil with hinged
  flap, Aeronautical Research Committee, R and M 1095 (plain flap as a
  hinged aft portion of the section).
* Abbott, I. H. and von Doenhoff, A. E., 1959, Theory of Wing Sections,
  Dover (airfoil section geometry used in the tests).
* Ladson, C. L., Brooks, C. W., Hill, A. S. and Sproles, D. W., 1996,
  Computer Program To Obtain Ordinates for NACA Airfoils, NASA TM-4741
  (closed trailing edge of the NACA 4-digit thickness law).
* Maskew, B., 1987, Program VSAERO Theory Document, NASA CR-4023,
  sections 2.2 and 3.1.4 (the Kutta condition is implied by the wake
  doublet; any panel edge may shed a wake and the user chooses a
  physically realistic wake model; part-span flap with prescribed vortex
  separations at the flap edge, section 4.4).
* O'Rourke, J., 1998, Computational Geometry in C, 2nd ed., Cambridge
  University Press, Chapter 1 (triangulation of a simple polygon by ear
  clipping, used for the control side closures).

Control surfaces
----------------
A plain control surface deflects the section shapes inside its span range
with ``ventorum.airfoil.flap.deflect_plain_flap`` at the control chord
fraction and deflection. The sign follows the VLM controls: positive moves
the trailing edge down on a horizontal surface. The chordwise resampling
runs on the deflected shape with a node at the hinge on both surfaces. Aft
of the hinge, the stations of each deflected surface are stretched from the
hinge to its rotated trailing edge, so no node falls beyond the trailing
edge. The trailing-edge nodes use the exact rotated trailing-edge point, so
a check of the rotation about the hinge line passes to round-off.

Side edges of a control (``control_edge``). With "transition" (default)
the deflection changes linearly over a spanwise transition centred on the
side edge of the control, of width ``control_transition`` times the local
chord. The builder adds a station at each end of the transition, and the
stations inside it carry the deflection times the ramp factor. The
surface stays continuous (a sealed edge, as the camber step of a vortex
lattice), so there is no side face and no gap. With "gap" (and always at
an edge between two deflected controls) the mesh carries two spanwise
stations at the same y (one deflected, one undeflected). The nodes that
the two sections share (forward of the hinge) are common. The gap between
the two sections, in the plane of the station, is closed with flat side
panels: the gap polygon is cut at the crossings of the two section
outlines into simple lobes, and each lobe is triangulated (ear clipping),
so each side panel is a plane convex triangle. A crossing node on a
section outline also splits the adjacent surface panel into triangles, so
the mesh stays watertight with no hanging node. The closure models no gap
flow.

Why the transition: near the trailing edge of a side gap the two section
outlines cross (a flap down puts the flap upper surface below the lower
surface of the fixed part). The region between them in the station plane
is a window of fluid that joins the flow under the fixed part to the flow
over the flap. The potential difference across the window is of the order
of the circulation, so the potential flow through the window is singular
at its sharp edges, and the circulation of the strips next to the edge
falls as the strips become narrower: CL falls and the induced drag rises
with spanwise refinement. The two trailing-edge nodes of the gap are also
not joined by a wake, so the two wake strips end in two full-strength
trailing vortices. A side-edge wake (Maskew 1987 prescribes vortex
separations at the flap edge) between the two trailing-edge nodes bounds
the induced drag but does not stop the loss of lift; a wake panel over the
window made the solution unstable. The sealed transition converges with
spanwise refinement, and its converged result changes very little with the
transition width (see ``tests/test_panel3d_controls.py``). Zero deflection
gives the same mesh as before, bit for bit.

Metadata for strip loads
------------------------
Each panel carries a spanwise strip index (``strip``), a chordwise index
(``chord_index``) and a surface side (``side``: +1 upper, -1 lower, 0 for
base, tip and side panels). Each node carries its spanwise station row
(``node_station``). The panel order of a mesh with controls differs from
the order of a plain mesh; use these fields, not the panel order.

Coordinate conventions
----------------------
Geometry axes: x aft [m], y right [m], z up [m]. Airfoil input arrays use the
Selig order: from the trailing edge along the upper surface to the leading
edge, then along the lower surface back to the trailing edge. Unit chord with
the leading edge at (0, 0) and the trailing-edge midpoint at (1, 0). A closed
trailing edge repeats the same point at the start and the end.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ventorum.geometry.lattice import _surface_eta, resolve_spacing, surface_reference_line

__all__ = [
    "CONTROL_TRANSITION",
    "FOLD_MARGIN",
    "ROOT_SCALE_WARN_DIHEDRAL_DEG",
    "PanelMesh",
    "check_mesh",
    "mesh_aircraft",
    "mesh_lifting_surface",
]

#: Trailing-edge gap over local chord below which the edge counts as sharp.
_TE_GAP_TOL = 1e-6
#: Tolerance [-] to merge a control boundary with an existing span station.
_ETA_MERGE_TOL = 1e-12
#: Tolerance [-] to assign a span midpoint to a control range.
_CTRL_TOL = 1e-9
#: Distance [-] (fraction of the local chord) below which two nodes of the
#: two stations of a control side edge are the same node.
_NODE_MERGE_REL = 1e-9
#: Parameter tolerance [-] of segment crossings in the side closure.
_PARAM_TOL = 1e-9
#: Half-mesh root edges: |y| below this fraction of the mesh size [-].
_Y_ROOT_REL = 1e-9
#: Outward: component volume above this fraction of its size cubed [-].
_VOLUME_REL = 1e-8
#: Default planarity limit of check_mesh: corner distance from the mean
#: plane over the panel diameter [-].
WARP_LIMIT = 0.05
#: Root-thickness scaling: smallest cosine [-] of the dihedral of the first
#: panel row. A row closer to the vertical has no miter (the two halves of
#: a symmetric surface coincide), so the root section is not scaled.
_COS_ROOT_MIN = 1e-9
#: Root-thickness scaling: dihedral of the first panel row [deg] above which
#: the builder issues a UserWarning. Above 60 deg the factor 1 / cos(G) is
#: above 2, so the root section is strongly scaled and the result needs a
#: check. The scaling is not capped (owner decision); this limit gives only
#: the warning.
ROOT_SCALE_WARN_DIHEDRAL_DEG = 60.0
#: Fold guard: smallest distance of a section node from the plane of the
#: neighbour section, as a fraction of the station spacing [-]. Any
#: positive value prevents a fold; 0.05 changes only the stations that
#: fold or nearly fold. This is a method choice, not a tuned value.
FOLD_MARGIN = 0.05
#: Default width of the deflection transition at the side edge of a
#: control surface, as a fraction of the local chord [-]. The converged
#: result changes very little with this width (see the module notes); a
#: wider transition is resolved by fewer strips, so the spanwise sequence
#: becomes monotonic at a coarser mesh. This is a method choice.
CONTROL_TRANSITION = 0.2
#: Accepted values of the trailing-edge and control-edge options.
_TRAILING_EDGE_OPTIONS = ("close", "base")
_CONTROL_EDGE_OPTIONS = ("transition", "gap")
#: Bisection steps of the fold guard [-].
_GUARD_STEPS = 60
_XHAT = np.array([1.0, 0.0, 0.0])


@dataclass(slots=True)
class PanelMesh:
    """Closed quadrilateral mesh of one or more thick lifting surfaces.

    Attributes
    ----------
    nodes : numpy.ndarray
        Node positions [m], shape (Nn, 3).
    panels : numpy.ndarray
        Panel corner node indices, shape (Np, 4). A triangle repeats its
        third node in the fourth entry.
    component : numpy.ndarray
        Component index of each panel, shape (Np,).
    kind : numpy.ndarray
        Panel kind codes, shape (Np,): "surface", "tip", "base" (blunt
        trailing edge) or "side" (flat side closure at the span edge of a
        deflected control surface).
    wake_edges : numpy.ndarray
        Wake shedding edges, shape (Nw, 4): upper trailing-edge panel index,
        lower trailing-edge panel index, node a, node b. Nodes a and b span
        one spanwise strip at the trailing edge (the upper edge when the
        trailing edge is blunt).
    half : bool
        True for a right-half mesh only, open at y = 0 [m].
    tip : str
        Tip closure of the mesh builder: "flat" or "none" (open tips, only
        for tests). :func:`check_mesh` accepts open tip edges only for "none".
    strip : numpy.ndarray or None
        Spanwise strip index of each panel, shape (Np,), -1 for tip and side
        panels. A full symmetric surface numbers its strips from the left
        tip to the right tip; a half mesh and a one-sided surface from the
        root to the tip. :func:`mesh_aircraft` offsets the indices so that
        they are unique over the mesh. None for meshes from other builders.
    chord_index : numpy.ndarray or None
        Chordwise index of each panel, shape (Np,): 0 at the leading edge
        to n_c - 1 at the trailing edge on each surface (n_c includes the
        hinge stations of controls), n_c for a base panel, -1 for tip and
        side panels.
    side : numpy.ndarray or None
        Surface side of each panel, shape (Np,): +1 upper, -1 lower, 0 for
        base, tip and side panels.
    node_station : numpy.ndarray or None
        Spanwise station row of each node, shape (Nn,). It increases along
        the span; the smallest and the largest value of a component mark
        its two ends (tips, or root and tip). Values are per component.
    """

    nodes: np.ndarray
    panels: np.ndarray
    component: np.ndarray
    kind: np.ndarray
    wake_edges: np.ndarray
    half: bool = False
    tip: str = "flat"
    strip: np.ndarray | None = None
    chord_index: np.ndarray | None = None
    side: np.ndarray | None = None
    node_station: np.ndarray | None = None


def _target_x(n_chord: int, chord_spacing: str) -> np.ndarray:
    """Chordwise stations in unit chord for one surface, from 0 to 1."""
    m = str(chord_spacing).lower().replace("_", "-")
    if m in ("cosine", "full-cosine"):
        k = np.arange(n_chord + 1, dtype=float)
        return 0.5 * (1.0 - np.cos(np.pi * k / float(n_chord)))
    if m == "uniform":
        return np.linspace(0.0, 1.0, n_chord + 1)
    raise ValueError(f"Unknown chord_spacing {chord_spacing!r}; use 'cosine' or 'uniform'.")


def _split_selig(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a Selig array into upper (ascending x) and lower (ascending x) parts."""
    a = np.asarray(coords, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("Each section_coords array must have shape (N, 2).")
    if a.shape[0] < 4:
        raise ValueError("Each section_coords array needs at least 4 points.")
    if not np.all(np.isfinite(a)):
        raise ValueError("section_coords holds non-finite values.")
    i_le = int(np.argmin(a[:, 0]))
    if i_le == 0 or i_le == a.shape[0] - 1:
        raise ValueError("section_coords has no leading-edge minimum inside the array.")
    return a[: i_le + 1][::-1], a[i_le:]


def _resample_section(coords: np.ndarray, x_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample one section to the target x stations; interpolate y on x per surface."""
    up, lo = _split_selig(coords)
    return np.interp(x_target, up[:, 0], up[:, 1]), np.interp(x_target, lo[:, 0], lo[:, 1])


def _rotate_vecs(v: np.ndarray, axis: np.ndarray, angle: np.ndarray | float) -> np.ndarray:
    """Rodrigues rotation of row vectors *v* about unit axes *axis*."""
    v = np.asarray(v, dtype=float)
    axis = np.asarray(axis, dtype=float)
    ang = np.broadcast_to(np.asarray(angle, dtype=float), (v.shape[0],))
    c = np.cos(ang)[:, None]
    s = np.sin(ang)[:, None]
    kdv = np.sum(axis * v, axis=1)[:, None]
    return v * c + np.cross(axis, v) * s + axis * kdv * (1.0 - c)


def _section_frames(axis: np.ndarray, twist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the chord direction and the section normal of each station.

    The chord direction is x twisted about the local spanwise axis. The
    section normal is x cross the spanwise axis (the normal of the lattice
    surface), twisted with the chord. A flat wing gives the normal z; a
    vertical fin with the span up gives -y.

    Parameters
    ----------
    axis : numpy.ndarray
        Unit spanwise axes in the y-z plane, shape (n, 3).
    twist : numpy.ndarray
        Twist with incidence [rad], shape (n,).

    Returns
    -------
    tuple of numpy.ndarray
        Chord directions and section normals (unit vectors), shape (n, 3).
    """
    n = axis.shape[0]
    cdir = _rotate_vecs(np.tile(_XHAT, (n, 1)), axis, twist)
    ndir = _rotate_vecs(np.cross(np.tile(_XHAT, (n, 1)), axis), axis, twist)
    return cdir, ndir


def _flip_panel(row) -> list[int]:
    """Return the panel with the opposite orientation, triangles kept as (a, b, c, c)."""
    a, b, c, d = (int(v) for v in row)
    if c == d:
        return [a, c, b, b]
    return [a, d, c, b]


def _interp_rows(fr: np.ndarray, mat: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Linear interpolation of each column of *mat* (sections by rows) at *eta*."""
    fr = np.asarray(fr, dtype=float)
    out = np.empty((len(eta), mat.shape[1]), dtype=float)
    for k in range(mat.shape[1]):
        out[:, k] = np.interp(eta, fr, mat[:, k])
    return out


def _as_coord_list(section_coords: np.ndarray | Sequence[np.ndarray], n_sec: int) -> list[np.ndarray]:
    """Return one coordinate array per section."""
    if isinstance(section_coords, np.ndarray):
        return [np.asarray(section_coords, dtype=float) for _ in range(n_sec)]
    items = list(section_coords)
    if len(items) != n_sec:
        raise ValueError(f"section_coords has {len(items)} arrays for {n_sec} sections.")
    return [np.asarray(c, dtype=float) for c in items]


def _active_controls(surf) -> list:
    """Return the controls of a surface with nonzero deflection [rad]."""
    out = []
    for c in list(getattr(surf, "controls", [])):
        if float(getattr(c, "deflection", 0.0)) != 0.0:
            out.append(c)
    return out


def _control_at_eta(eta: float, controls: list):
    """Return the control covering span fraction *eta* [-], else None."""
    for c in controls:
        if float(c.eta_start) - _CTRL_TOL <= float(eta) <= float(c.eta_end) + _CTRL_TOL:
            return c
    return None


def _union_xt(xt_base: np.ndarray, controls: list) -> np.ndarray:
    """Return chord stations with a node at each active hinge [-]."""
    vals = sorted([float(v) for v in list(xt_base)] + [float(c.hinge_x_c) for c in controls])
    merged: list[float] = []
    for v in vals:
        if merged and abs(v - merged[-1]) <= _ETA_MERGE_TOL:
            continue
        merged.append(v)
    return np.array(merged, dtype=float)


def _unique_sorted(vals: list[float]) -> np.ndarray:
    """Return sorted unique values merged within the merge tolerance."""
    out: list[float] = []
    for v in sorted(float(x) for x in vals):
        if out and abs(v - out[-1]) <= _ETA_MERGE_TOL:
            continue
        out.append(v)
    return np.array(out, dtype=float)


def _edge_lists(eta_base: np.ndarray, controls: list) -> tuple[np.ndarray, list]:
    """Return augmented edge stations and their control for one half.

    Interior control boundaries appear twice at the same y [m]: the gap
    between the pair is closed with flat side panels.
    """
    bounds: list[float] = []
    for c in controls:
        for b in (float(c.eta_start), float(c.eta_end)):
            if _ETA_MERGE_TOL < b < 1.0 - _ETA_MERGE_TOL:
                bounds.append(b)
    uniq = _unique_sorted(list(np.asarray(eta_base, dtype=float).ravel()) + bounds)
    if uniq.size == 1:
        return uniq, [_control_at_eta(float(uniq[0]), controls)]
    seg_ctrl = []
    for i in range(uniq.size - 1):
        mid = 0.5 * (float(uniq[i]) + float(uniq[i + 1]))
        seg_ctrl.append(_control_at_eta(mid, controls))
    etas: list[float] = []
    ctrls: list = []
    etas.append(float(uniq[0]))
    ctrls.append(seg_ctrl[0])
    for k in range(1, uniq.size - 1):
        left = seg_ctrl[k - 1]
        right = seg_ctrl[k]
        if left is right:
            etas.append(float(uniq[k]))
            ctrls.append(left)
        else:
            etas.append(float(uniq[k]))
            ctrls.append(left)
            etas.append(float(uniq[k]))
            ctrls.append(right)
    etas.append(float(uniq[-1]))
    ctrls.append(seg_ctrl[-1])
    return np.array(etas, dtype=float), ctrls


def _span_length_rate(surf, eta: float) -> float:
    """Return the length of the reference line per unit span parameter at *eta* [m]."""
    h = 1e-6
    a = max(0.0, float(eta) - h)
    b = min(1.0, float(eta) + h)
    _, y_ref, z_ref = surface_reference_line(surf, np.array([a, b]))
    return float(np.hypot(y_ref[1] - y_ref[0], z_ref[1] - z_ref[0])) / max(b - a, 1e-300)


def _transition_edge_lists(
    surf,
    eta_base: np.ndarray,
    controls: list,
    chord_of,
    width: float,
) -> tuple[np.ndarray, list, np.ndarray]:
    """Return the stations, their control and their deflection factor for sealed control edges.

    At a side edge between a control and an undeflected part of the span,
    the deflection changes linearly over a transition centred on the edge,
    of spanwise length *width* times the local chord. Each end of a
    transition becomes a station. The transition is not longer than the
    control, does not reach past the root or the tip, and does not reach
    into the transition of the next edge. At an edge between two deflected
    controls the two stations at the same span parameter of
    :func:`_edge_lists` stay (gap with side panels).

    Parameters
    ----------
    surf : LiftingSurface
        Surface definition.
    eta_base : numpy.ndarray
        Span parameters of the lattice stations [-].
    controls : list
        Controls with nonzero deflection.
    chord_of : callable
        Local chord [m] at a span parameter.
    width : float
        Transition length over the local chord [-].

    Returns
    -------
    tuple
        Span parameters of the stations [-] (increasing, duplicates at gap
        edges), the control of each station (or None) and the deflection
        factor of each station [-] (1 inside a control, 0 to 1 in a
        transition).
    """
    tol = _CTRL_TOL

    def neighbour(b: float, outside: float):
        e = b + outside * 10.0 * tol
        if e <= 0.0 or e >= 1.0:
            return None
        return _control_at_eta(e, controls)

    # Boundaries: (eta, control, side) with side -1 at the start, +1 at the end.
    bounds = []
    for c in controls:
        for b, sd in ((float(c.eta_start), -1.0), (float(c.eta_end), 1.0)):
            if _ETA_MERGE_TOL < b < 1.0 - _ETA_MERGE_TOL:
                bounds.append((b, c, sd))
    gap_bounds = []
    ramps = []
    for b, c, sd in bounds:
        other = neighbour(b, sd)
        if other is not None and other is not c:
            gap_bounds.append(b)
        else:
            ramps.append([b, c, sd, float(width) * float(chord_of(b)) / max(_span_length_rate(surf, b), 1e-300)])
    # Limits of each ramp: inside the control, before the root or the tip,
    # and before the transition or the gap of the next edge outside.
    edges = sorted({b for b, _, _ in bounds} | {0.0, 1.0})
    for r in ramps:
        b, c, sd, w = r
        span_c = float(c.eta_end) - float(c.eta_start)
        w = min(w, span_c)
        nxt = [e for e in edges if (e - b) * sd > _ETA_MERGE_TOL]
        room = abs((min(nxt) if sd > 0 else max(nxt)) - b) if nxt else 0.0
        end_room = (1.0 - b) if sd > 0 else b
        # Up to the root or the tip the whole room is free; another edge
        # shares the room with its own transition.
        w = min(w, 2.0 * room if abs(room - end_room) <= _ETA_MERGE_TOL else room)
        r[3] = max(w, 0.0)
    ramps = [r for r in ramps if r[3] > _ETA_MERGE_TOL]
    extra = []
    for b, _, _, w in ramps:
        extra.extend([b - 0.5 * w, b + 0.5 * w])
    ramp_at = {}
    for b, c, sd, w in ramps:
        ramp_at[(id(c), sd)] = (b, w)

    def factor(eta: float, c) -> float:
        f = 1.0
        for sd, b0 in ((-1.0, float(c.eta_start)), (1.0, float(c.eta_end))):
            key = (id(c), sd)
            if key in ramp_at:
                b, w = ramp_at[key]
                t = (eta - (b - 0.5 * w)) / w if sd < 0 else ((b + 0.5 * w) - eta) / w
                f = min(f, min(max(t, 0.0), 1.0))
            elif sd < 0 and eta < b0 - tol:
                f = 0.0
            elif sd > 0 and eta > b0 + tol:
                f = 0.0
        return f

    def station_entry(eta: float, prefer=None):
        best = (None, 0.0)
        for c in controls:
            if prefer is not None and c is not prefer:
                continue
            f = factor(eta, c)
            if f > best[1]:
                best = (c, f)
        return best

    uniq = _unique_sorted(list(np.asarray(eta_base, dtype=float).ravel()) + extra + gap_bounds)
    gaps = _unique_sorted(gap_bounds) if gap_bounds else np.empty(0)
    etas: list[float] = []
    ctrls: list = []
    facs: list[float] = []
    for e in uniq:
        e = float(e)
        if gaps.size and np.min(np.abs(gaps - e)) <= _ETA_MERGE_TOL:
            left = _control_at_eta(e - 10.0 * tol, controls)
            right = _control_at_eta(e + 10.0 * tol, controls)
            for c in (left, right):
                c_e, f_e = station_entry(e, prefer=c) if c is not None else (None, 0.0)
                etas.append(e)
                ctrls.append(c_e)
                facs.append(f_e)
            continue
        c_e, f_e = station_entry(e)
        etas.append(e)
        ctrls.append(c_e if f_e > 0.0 else None)
        facs.append(f_e if f_e > 0.0 else 0.0)
    return np.array(etas, dtype=float), ctrls, np.array(facs, dtype=float)


def _coords_from_rows(xt: np.ndarray, yu: np.ndarray, yl: np.ndarray) -> np.ndarray:
    """Return a Selig ordered polygon from upper and lower rows [chord units]."""
    xt = np.asarray(xt, dtype=float)
    yu = np.asarray(yu, dtype=float)
    yl = np.asarray(yl, dtype=float)
    upper = np.column_stack([xt[::-1], yu[::-1]])
    lower = np.column_stack([xt[1:], yl[1:]])
    return np.vstack([upper, lower])


def _right_delta_deg(ctrl, mirror_side: bool) -> float:
    """Return the deflection [deg] for one edge station of a control."""
    if ctrl is None:
        return 0.0
    delta = float(ctrl.deflection)
    if mirror_side and not bool(ctrl.symmetric):
        delta = -delta
    return float(np.degrees(delta))


def _single_valued(surf: np.ndarray) -> np.ndarray:
    """Return the surface points with increasing x [chord units].

    On the compressed side of a deflected flap the moved surface can start
    forward of the hinge corner without a crossing of the fixed surface
    (for example when the lower surface rises toward the trailing edge).
    The moved points that lie forward of a point already kept are dropped,
    so the surface is single valued in x for the chordwise resampling. The
    shape changes only inside the small notch at the hinge.

    Parameters
    ----------
    surf : numpy.ndarray
        Surface points from the leading to the trailing edge [chord units],
        shape (N, 2).

    Returns
    -------
    numpy.ndarray
        Points with strictly increasing x [chord units].
    """
    keep = [0]
    for i in range(1, surf.shape[0]):
        if surf[i, 0] > surf[keep[-1], 0]:
            keep.append(i)
    return surf[keep]


def _deflected_station(
    xt: np.ndarray,
    yu: np.ndarray,
    yl: np.ndarray,
    ctrl,
    delta_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the chordwise stations and ordinates of one station [chord units].

    Forward of the hinge the stations stay at *xt*. Aft of the hinge the
    stations of each deflected surface are stretched from the hinge to its
    rotated trailing edge, so no station falls beyond the trailing edge.
    The trailing-edge nodes use the exact rotated trailing-edge point.

    Parameters
    ----------
    xt : numpy.ndarray
        Chordwise stations [chord units], from 0 to 1, with the hinge station.
    yu, yl : numpy.ndarray
        Undeflected upper and lower ordinates at *xt* [chord units].
    ctrl : ControlSurface or None
        Control of the station.
    delta_deg : float
        Deflection [deg], positive trailing edge down.

    Returns
    -------
    tuple of numpy.ndarray
        (xu, zu, xl, zl): chordwise stations and ordinates of the upper and
        the lower surface [chord units].
    """
    xt = np.asarray(xt, dtype=float)
    if ctrl is None or delta_deg == 0.0:
        return xt.copy(), np.array(yu, dtype=float), xt.copy(), np.array(yl, dtype=float)
    from ventorum.airfoil.flap import deflect_plain_flap, flap_hinge_point

    xh = float(ctrl.hinge_x_c)
    base = _coords_from_rows(xt, yu, yl)
    hinge = flap_hinge_point(base, xh, 0.5)
    theta = -float(np.radians(delta_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    # The mesh resamples the section with a node at the hinge, so it keeps
    # the outline as constructed (no hinge treatment).
    moved = deflect_plain_flap(base, xh, delta_deg, hinge_treatment="none")
    up_d, lo_d = _split_selig(moved)
    out = []
    for surf_d, y_te in ((_single_valued(up_d), float(yu[-1])), (_single_valued(lo_d), float(yl[-1]))):
        d = np.array([float(xt[-1]), y_te]) - hinge
        q = hinge + np.array([cos_t * d[0] - sin_t * d[1], sin_t * d[0] + cos_t * d[1]])
        xs = xt.copy()
        aft = xt > xh
        xs[aft] = xh + (xt[aft] - xh) * ((float(q[0]) - xh) / (float(xt[-1]) - xh))
        zs = np.interp(xs, surf_d[:, 0], surf_d[:, 1])
        xs[-1] = float(q[0])
        zs[-1] = float(q[1])
        out.append((xs, zs))
    (xu, zu), (xl, zl) = out
    # The leading edge stays fixed. The Selig polygon of the rows has one
    # leading-edge point (the upper one), so keep both input ordinates there.
    zu[0] = float(yu[0])
    zl[0] = float(yl[0])
    return xu, zu, xl, zl


def _span_axes_for_edges(surf, eta_edges: np.ndarray) -> np.ndarray:
    """Return unit spanwise axes for edge stations with duplicate y [m]."""
    eta_edges = np.asarray(eta_edges, dtype=float)
    uniq = _unique_sorted(list(eta_edges.ravel()))
    x_ref, y_ref, z_ref = surface_reference_line(surf, uniq)
    if uniq.size >= 2:
        dy = np.gradient(y_ref, uniq)
        dz = np.gradient(z_ref, uniq)
    else:
        dy = np.zeros_like(uniq)
        dz = np.zeros_like(uniq)
    axis_u = np.column_stack([np.zeros_like(uniq), dy, dz])
    axis_u /= np.maximum(np.linalg.norm(axis_u, axis=1, keepdims=True), 1e-300)
    if bool(getattr(surf, "is_symmetric", False)):
        axis_u[uniq <= 1e-12] = np.array([0.0, 1.0, 0.0])
    out = np.empty((eta_edges.size, 3), dtype=float)
    for j, e in enumerate(eta_edges):
        k = int(np.argmin(np.abs(uniq - float(e))))
        out[j] = axis_u[k]
    return out


def _root_thickness_factor(surf, eta: np.ndarray) -> float:
    """Return the thickness factor [-] of the root section of a symmetric surface.

    The factor is 1 / cos(G), with G the dihedral of the first panel row
    (the reference line from the root station to the next station, in the
    y-z plane). It is 1 for a one-sided surface, for stations that do not
    start at the root, for zero dihedral and for a first panel row that is
    vertical or turns past the vertical.

    Parameters
    ----------
    surf : LiftingSurface
        Surface definition.
    eta : numpy.ndarray
        Span fractions of the stations [-], increasing, with duplicates at
        control side edges.

    Returns
    -------
    float
        Factor on the offsets of the root section from its camber line [-].
    """
    if not bool(getattr(surf, "is_symmetric", False)):
        return 1.0
    eta = np.asarray(eta, dtype=float)
    if eta.size < 2 or float(eta[0]) > 1e-12:
        return 1.0
    nxt = eta[eta > float(eta[0]) + _ETA_MERGE_TOL]
    if nxt.size == 0:
        return 1.0
    _, y_ref, z_ref = surface_reference_line(surf, np.array([float(eta[0]), float(nxt[0])]))
    dy = float(y_ref[1] - y_ref[0])
    dz = float(z_ref[1] - z_ref[0])
    length = float(np.hypot(dy, dz))
    if length <= 0.0:
        return 1.0
    cos_g = dy / length
    if cos_g <= _COS_ROOT_MIN or cos_g >= 1.0:
        return 1.0
    return 1.0 / cos_g


def _warn_strong_root_scaling(surf, factor: float) -> None:
    """Issue a UserWarning when the root-thickness factor comes from a steep first panel row.

    Parameters
    ----------
    surf : LiftingSurface
        Surface definition (for the name in the message).
    factor : float
        Applied root-thickness factor 1 / cos(G) [-], with G the dihedral
        of the first panel row.
    """
    if not (math.isfinite(factor) and factor > 1.0):
        return
    dihedral_deg = math.degrees(math.acos(min(1.0, 1.0 / factor)))
    if dihedral_deg <= ROOT_SCALE_WARN_DIHEDRAL_DEG:
        return
    name = str(getattr(surf, "name", ""))
    warnings.warn(
        f"Surface {name!r}: the root section is strongly scaled. The first panel row has a dihedral of "
        f"{dihedral_deg:.1f} deg (above {ROOT_SCALE_WARN_DIHEDRAL_DEG:g} deg), so the root-thickness factor "
        f"1/cos(dihedral) is {factor:.2f}. The result needs checking (for example against "
        "scale_root_thickness=False).",
        UserWarning,
        stacklevel=3,
    )


def _scale_about_camber(yu: np.ndarray, yl: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale the offsets of a section from its camber line [chord units].

    The camber line is the mean of the upper and the lower ordinate at
    each chordwise station, so it does not move.

    Parameters
    ----------
    yu, yl : numpy.ndarray
        Upper and lower ordinates at the same chordwise stations [chord units].
    factor : float
        Factor on the offsets from the camber line [-].

    Returns
    -------
    tuple of numpy.ndarray
        Scaled upper and lower ordinates [chord units].
    """
    yu = np.asarray(yu, dtype=float)
    yl = np.asarray(yl, dtype=float)
    camber = 0.5 * (yu + yl)
    half = 0.5 * (yu - yl)
    return camber + factor * half, camber - factor * half


def _close_trailing_edge(x: np.ndarray, yu: np.ndarray, yl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Close an open trailing edge with a quartic thickness change [chord units].

    The upper and the lower ordinates move toward the mid point of the
    trailing-edge gap g by (g / 2) x^4, so the camber line does not move.
    For the NACA 4-digit thickness law this gives the closed trailing edge
    of Ladson et al. (1996): the x^4 coefficient -0.1036 instead of -0.1015.
    A sharp trailing edge (gap at most the sharp-edge tolerance) is
    returned unchanged.

    Parameters
    ----------
    x : numpy.ndarray
        Chordwise stations [chord units], from 0 to 1.
    yu, yl : numpy.ndarray
        Upper and lower ordinates at *x* [chord units].

    Returns
    -------
    tuple of numpy.ndarray
        Closed upper and lower ordinates [chord units].
    """
    x = np.asarray(x, dtype=float)
    yu = np.asarray(yu, dtype=float)
    yl = np.asarray(yl, dtype=float)
    gap = float(yu[-1] - yl[-1])
    if abs(gap) <= _TE_GAP_TOL:
        return yu, yl
    ramp = 0.5 * gap * x**4
    return yu - ramp, yl + ramp


def _station_offsets(
    xu: np.ndarray,
    zu: np.ndarray,
    xl: np.ndarray,
    zl: np.ndarray,
    chord: float,
    twist: float,
    factor: float,
) -> tuple[float, float]:
    """Return the smallest and the largest section offset across the span plane [m].

    The offset of a section point is its distance from the quarter-chord
    point along the direction in the section plane that is normal to x.
    With the twist T about the span axis, a point (x, z) in chord units
    has the offset c (z cos T - (x - 0.25) sin T). The offsets from the
    camber line (mean of the upper and the lower ordinate at each index)
    are multiplied by *factor* first.

    Parameters
    ----------
    xu, zu, xl, zl : numpy.ndarray
        Chordwise stations and ordinates of the upper and the lower surface
        [chord units].
    chord : float
        Local chord [m].
    twist : float
        Twist with incidence [rad].
    factor : float
        Factor on the offsets from the camber line [-].

    Returns
    -------
    tuple of float
        Smallest and largest offset [m].
    """
    zc = 0.5 * (np.asarray(zu, dtype=float) + np.asarray(zl, dtype=float))
    h = 0.5 * (np.asarray(zu, dtype=float) - np.asarray(zl, dtype=float))
    s_t = float(np.sin(twist))
    c_t = float(np.cos(twist))
    s_u = chord * ((zc + factor * h) * c_t - (np.asarray(xu, dtype=float) - 0.25) * s_t)
    s_l = chord * ((zc - factor * h) * c_t - (np.asarray(xl, dtype=float) - 0.25) * s_t)
    both = np.concatenate([s_u, s_l])
    return float(np.min(both)), float(np.max(both))


def _guard_axes(
    axis: np.ndarray,
    qc: np.ndarray,
    offsets,
    margin: float = FOLD_MARGIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn the station planes so that no section crosses the plane of its neighbour.

    The stations are processed from the first to the last. The plane of
    station j (normal: its span axis a_j in the y-z plane, through its
    quarter-chord point q_j) must keep every node of station j - 1 on its
    inboard side and the plane of station j - 1 every node of station j on
    its outboard side, each by at least *margin* times the station spacing
    d. A node of station j at the offset s (along m_j = x cross a_j) has
    the distance (q_j - q_{j-1}) . a_{j-1} + s m_j . a_{j-1} from the plane
    of station j - 1. When the condition fails, the axis angle of station
    j moves toward the angle of station j - 1 by the largest fraction that
    passes (bisection). The first station keeps its plane (the miter at
    the root of a symmetric surface). Stations at the same span parameter
    (control side edges) share the plane of the station before them.

    Parameters
    ----------
    axis : numpy.ndarray
        Unit span axes in the y-z plane, shape (n, 3).
    qc : numpy.ndarray
        Quarter-chord points of the stations [m], shape (n, 3).
    offsets : callable
        ``offsets(j, tilt)`` returns the smallest and the largest section
        offset [m] of station j when its plane turns by *tilt* [rad] from
        the plane normal to the span (see :func:`_station_offsets`).
    margin : float
        Smallest node distance from the neighbour plane over the station
        spacing [-].

    Returns
    -------
    tuple of numpy.ndarray
        Guarded axes, shape (n, 3) (unchanged rows bit for bit), and the
        tilt of each station from its plane normal to the span [rad],
        shape (n,) (zero for an unchanged station).
    """
    axis = np.asarray(axis, dtype=float)
    n = axis.shape[0]
    phi = np.arctan2(axis[:, 2], axis[:, 1])
    phi_hat = phi.copy()
    tilt = np.zeros(n)
    out = axis.copy()
    yz = np.asarray(qc, dtype=float)[:, 1:3]
    size = float(np.max(np.ptp(yz, axis=0))) if n > 1 else 1.0
    tol = 1e-12 * max(size, 1e-300)

    def unit(angle: float) -> np.ndarray:
        return np.array([np.cos(angle), np.sin(angle)])

    def normal_dir(angle: float) -> np.ndarray:
        return np.array([-np.sin(angle), np.cos(angle)])

    for j in range(1, n):
        step = yz[j] - yz[j - 1]
        d = float(np.hypot(step[0], step[1]))
        if d <= tol:
            phi_hat[j] = phi_hat[j - 1]
            tilt[j] = tilt[j - 1]
            if tilt[j] != 0.0:
                out[j] = np.array([0.0, np.cos(phi_hat[j]), np.sin(phi_hat[j])])
            continue
        ph0 = float(phi_hat[j - 1])
        a0 = unit(ph0)
        m0 = normal_dir(ph0)
        lo0, hi0 = offsets(j - 1, float(tilt[j - 1]))

        def passes(ph1: float, j=j, step=step, d=d, ph0=ph0, a0=a0, m0=m0, lo0=lo0, hi0=hi0) -> bool:
            a1 = unit(ph1)
            m1 = normal_dir(ph1)
            lo1, hi1 = offsets(j, ph1 - float(phi[j]))
            near1 = min(float(step @ a0) + s * float(m1 @ a0) for s in (lo1, hi1))
            far0 = max(float(-step @ a1) + s * float(m0 @ a1) for s in (lo0, hi0))
            return near1 >= margin * d and far0 <= -margin * d

        if passes(float(phi[j])):
            continue
        lo_t, hi_t = 0.0, 1.0
        for _ in range(_GUARD_STEPS):
            mid = 0.5 * (lo_t + hi_t)
            if passes(ph0 + mid * (float(phi[j]) - ph0)):
                lo_t = mid
            else:
                hi_t = mid
        phi_hat[j] = ph0 + lo_t * (float(phi[j]) - ph0)
        tilt[j] = float(phi_hat[j] - phi[j])
        out[j] = np.array([0.0, np.cos(phi_hat[j]), np.sin(phi_hat[j])])
    return out, tilt


# ---------------------------------------------------------------------------
# Polygon helpers of the control side closures
# ---------------------------------------------------------------------------


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """Return the z component of the 2D cross product a x b."""
    return float(a[0] * b[1] - a[1] * b[0])


def _ear_clip(ids: list[int], xy: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a simple polygon by ear clipping; keep its orientation.

    Among the valid ears the one with the largest smallest angle is cut
    first. An ear is valid when it is convex in the polygon orientation and
    no other vertex is inside it or on its boundary (O'Rourke 1998,
    chapter 1).

    Parameters
    ----------
    ids : list of int
        Node indices of the polygon in order.
    xy : numpy.ndarray
        Plane coordinates of the polygon vertices [m], shape (n, 2).

    Returns
    -------
    list of tuple of int
        Triangles as node index triples in the polygon orientation.
    """
    n = len(ids)
    if n < 3:
        return []
    area2 = float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))
    sgn = 1.0 if area2 >= 0.0 else -1.0
    ext = float(np.max(np.ptp(xy, axis=0))) if n else 1.0
    tol = 1e-14 * max(ext, 1e-300) ** 2
    idx = list(range(n))
    tris: list[tuple[int, int, int]] = []
    while len(idx) > 3:
        m = len(idx)
        best = -1
        best_q = -1.0
        best_cr = -np.inf
        best_cr_pos = 0
        for pos in range(m):
            i0, i1, i2 = idx[pos - 1], idx[pos], idx[(pos + 1) % m]
            a, b, c = xy[i0], xy[i1], xy[i2]
            cr = sgn * _cross2(b - a, c - b)
            if cr > best_cr:
                best_cr = cr
                best_cr_pos = pos
            if cr <= tol:
                continue
            others = [idx[q] for q in range(m) if q not in ((pos - 1) % m, pos, (pos + 1) % m)]
            if others:
                p = xy[others]
                s0 = sgn * ((b[0] - a[0]) * (p[:, 1] - a[1]) - (b[1] - a[1]) * (p[:, 0] - a[0]))
                s1 = sgn * ((c[0] - b[0]) * (p[:, 1] - b[1]) - (c[1] - b[1]) * (p[:, 0] - b[0]))
                s2 = sgn * ((a[0] - c[0]) * (p[:, 1] - c[1]) - (a[1] - c[1]) * (p[:, 0] - c[0]))
                if np.any((s0 >= -tol) & (s1 >= -tol) & (s2 >= -tol)):
                    continue
            q = _min_angle(a, b, c)
            if q > best_q:
                best_q = q
                best = pos
        if best < 0:
            best = best_cr_pos
        m = len(idx)
        tris.append((ids[idx[best - 1]], ids[idx[best]], ids[idx[(best + 1) % m]]))
        del idx[best]
    tris.append((ids[idx[0]], ids[idx[1]], ids[idx[2]]))
    return tris


def _min_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the smallest interior angle [rad] of the triangle (a, b, c)."""
    out = np.pi
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        u = q - p
        v = r - p
        nu = float(np.hypot(u[0], u[1]))
        nv = float(np.hypot(v[0], v[1]))
        if nu <= 0.0 or nv <= 0.0:
            return 0.0
        cosang = float(np.clip((u[0] * v[0] + u[1] * v[1]) / (nu * nv), -1.0, 1.0))
        out = min(out, float(np.arccos(cosang)))
    return out


def _split_lobes(poly: list[int]) -> list[list[int]]:
    """Split a closed polygon with repeated nodes into simple lobes.

    Each repeated node closes a lobe. The lobes keep the traversal order of
    the polygon. Lobes with fewer than 3 nodes (folds) are dropped.
    """
    stack: list[int] = []
    where: dict[int, int] = {}
    lobes: list[list[int]] = []
    for v in list(poly) + [poly[0]]:
        if v in where:
            i = where[v]
            lobe = stack[i:]
            for u in lobe[1:]:
                where.pop(u, None)
            stack = stack[: i + 1]
            if len(lobe) >= 3:
                lobes.append(lobe)
        else:
            where[v] = len(stack)
            stack.append(v)
    return lobes


def _strip_triangles(p_chain: list[int], q_chain: list[int], pos: np.ndarray) -> list[list[int]]:
    """Triangulate the polygon p_chain + reversed(q_chain) between two node chains.

    The diagonal with the shorter length is chosen at each step. The
    triangles keep the orientation of the polygon.
    """
    tris: list[list[int]] = []
    i = 0
    m = 0
    n_p = len(p_chain) - 1
    n_q = len(q_chain) - 1
    while i < n_p or m < n_q:
        if i == n_p:
            tri = [p_chain[i], q_chain[m + 1], q_chain[m]]
            m += 1
        elif m == n_q:
            tri = [p_chain[i], p_chain[i + 1], q_chain[m]]
            i += 1
        else:
            d1 = float(np.linalg.norm(pos[p_chain[i + 1]] - pos[q_chain[m]]))
            d2 = float(np.linalg.norm(pos[p_chain[i]] - pos[q_chain[m + 1]]))
            if d1 <= d2:
                tri = [p_chain[i], p_chain[i + 1], q_chain[m]]
                i += 1
            else:
                tri = [p_chain[i], q_chain[m + 1], q_chain[m]]
                m += 1
        if len(set(tri)) == 3:
            tris.append([tri[0], tri[1], tri[2], tri[2]])
    return tris


def _polygon_panels(p_chain: list[int], q_chain: list[int], pos: np.ndarray) -> list[list[int]]:
    """Return the panels of the polygon p_chain + reversed(q_chain).

    Two plain chains (two nodes each) give one quadrilateral, or one
    triangle when two corners are the same node. Chains with extra nodes
    (crossing nodes of a side closure) give triangles.
    """
    if len(p_chain) == 2 and len(q_chain) == 2:
        ring = [p_chain[0], p_chain[1], q_chain[1], q_chain[0]]
        uniq: list[int] = []
        for v in ring:
            if not uniq or v != uniq[-1]:
                uniq.append(v)
        if len(uniq) > 1 and uniq[-1] == uniq[0]:
            uniq.pop()
        if len(set(uniq)) < 3:
            return []
        if len(uniq) == 4:
            return [ring]
        return [[uniq[0], uniq[1], uniq[2], uniq[2]]]
    return _strip_triangles(p_chain, q_chain, pos)


def _build_station_mesh(
    upper: np.ndarray,
    lower: np.ndarray,
    sharp: np.ndarray,
    eta: np.ndarray,
    chord: np.ndarray,
    origin: np.ndarray,
    cdir: np.ndarray,
    ndir: np.ndarray,
    cap_first: bool,
    cap_last: bool,
) -> dict:
    """Build nodes, panels, metadata and wake edges from station rows.

    Consecutive stations with the same span parameter form a control side
    edge. Their common nodes are shared, and the gap between the two
    section outlines is closed with flat side triangles.

    Parameters
    ----------
    upper, lower : numpy.ndarray
        Node positions of the upper and the lower surface [m], shape
        (ns, nc + 1, 3), from the leading to the trailing edge.
    sharp : numpy.ndarray
        Sharp trailing edge flag of each station, shape (ns,).
    eta : numpy.ndarray
        Span parameter of each station [-], shape (ns,), increasing.
    chord : numpy.ndarray
        Local chord of each station [m], shape (ns,).
    origin, cdir, ndir : numpy.ndarray
        Plane origin [m] and unit in-plane axes of each station, shape (ns, 3).
    cap_first, cap_last : bool
        Close the first (flipped) or the last station with a flat cap.

    Returns
    -------
    dict
        nodes, station_of, panels, kinds, strip, chord_index, side, wake and
        n_strips.
    """
    ns = upper.shape[0]
    nc = upper.shape[1] - 1
    dup_next = [abs(float(eta[j + 1]) - float(eta[j])) <= _ETA_MERGE_TOL for j in range(ns - 1)]
    nodes: list[np.ndarray] = []
    station_of: list[int] = []

    def add_node(p: np.ndarray, station: int) -> int:
        nodes.append(np.asarray(p, dtype=float).reshape(3))
        station_of.append(int(station))
        return len(nodes) - 1

    uid = np.full((ns, nc + 1), -1, dtype=int)
    lid = np.full((ns, nc + 1), -1, dtype=int)
    for j in range(ns):
        cand: list[int] = []
        if j > 0 and dup_next[j - 1]:
            cand = sorted({int(v) for v in uid[j - 1]} | {int(v) for v in lid[j - 1]})
        tol = _NODE_MERGE_REL * float(chord[j])

        def node_for(p: np.ndarray, cand=cand, tol=tol, j=j) -> int:
            for c in cand:
                if float(np.linalg.norm(nodes[c] - p)) <= tol:
                    return c
            return add_node(p, j)

        uid[j, 0] = lid[j, 0] = node_for(0.5 * (upper[j, 0] + lower[j, 0]))
        for k in range(1, nc):
            uid[j, k] = node_for(upper[j, k])
            lid[j, k] = node_for(lower[j, k])
        if sharp[j]:
            uid[j, nc] = lid[j, nc] = node_for(0.5 * (upper[j, nc] + lower[j, nc]))
        else:
            uid[j, nc] = node_for(upper[j, nc])
            lid[j, nc] = node_for(lower[j, nc])

    # Extra nodes on station edges: key (station, "u" | "l" | "b", k) ->
    # list of (parameter from the first to the second edge node, node id).
    extras: dict[tuple[int, str, int], list[tuple[float, int]]] = {}
    side_tris: list[list[int]] = []

    def loop_of(j: int) -> tuple[list[int], list[tuple[str, int, bool]]]:
        ring = [int(uid[j, k]) for k in range(nc + 1)]
        edges: list[tuple[str, int, bool]] = [("u", k, True) for k in range(nc)]
        if int(lid[j, nc]) != int(uid[j, nc]):
            ring.append(int(lid[j, nc]))
            edges.append(("b", nc, True))
        for k in range(nc - 1, -1, -1):
            ring.append(int(lid[j, k]))
            edges.append(("l", k, False))
        return ring, edges

    for a in range(ns - 1):
        if not dup_next[a]:
            continue
        b = a + 1
        ring_a, edges_a = loop_of(a)
        ring_b, edges_b = loop_of(b)
        if len(ring_a) != len(ring_b):
            raise RuntimeError("The two stations of a control side edge have different outlines.")
        diff = [i for i in range(len(ring_a)) if ring_a[i] != ring_b[i]]
        if not diff:
            continue
        i0 = diff[0] - 1
        i1 = diff[-1] + 1
        if i0 < 0 or i1 > len(ring_a) - 1:
            raise RuntimeError("The two sections of a control side edge share no leading-edge node.")
        chain_a = ring_a[i0: i1 + 1]
        chain_b = ring_b[i0: i1 + 1]
        seg_a = edges_a[i0: i1]
        seg_b = edges_b[i0: i1]
        pos3 = np.vstack(nodes)

        def to2d(ids: list[int], a=a, pos3=pos3) -> np.ndarray:
            rel = pos3[ids] - origin[a]
            return np.column_stack([rel @ cdir[a], rel @ ndir[a]])

        pa = to2d(chain_a)
        pb = to2d(chain_b)
        ins_a: list[dict[int, float]] = [dict() for _ in range(len(chain_a) - 1)]
        ins_b: list[dict[int, float]] = [dict() for _ in range(len(chain_b) - 1)]
        for i in range(len(chain_a) - 1):
            ea = {chain_a[i], chain_a[i + 1]}
            d = pa[i + 1] - pa[i]
            for m in range(len(chain_b) - 1):
                if ea & {chain_b[m], chain_b[m + 1]}:
                    continue
                e = pb[m + 1] - pb[m]
                den = _cross2(d, e)
                if abs(den) <= 1e-14 * float(np.linalg.norm(d)) * float(np.linalg.norm(e)):
                    continue
                w = pb[m] - pa[i]
                t = _cross2(w, e) / den
                u = _cross2(w, d) / den
                if t < -_PARAM_TOL or t > 1.0 + _PARAM_TOL or u < -_PARAM_TOL or u > 1.0 + _PARAM_TOL:
                    continue
                end_a = t <= _PARAM_TOL or t >= 1.0 - _PARAM_TOL
                end_b = u <= _PARAM_TOL or u >= 1.0 - _PARAM_TOL
                if not end_a and not end_b:
                    p3 = pos3[chain_a[i]] + t * (pos3[chain_a[i + 1]] - pos3[chain_a[i]])
                    x_id = add_node(p3, a)
                    pos3 = np.vstack([pos3, p3[None, :]])
                    ins_a[i][x_id] = t
                    ins_b[m][x_id] = u
                elif end_a and not end_b:
                    v_id = chain_a[i] if t <= 0.5 else chain_a[i + 1]
                    ins_b[m].setdefault(v_id, u)
                elif end_b and not end_a:
                    v_id = chain_b[m] if u <= 0.5 else chain_b[m + 1]
                    ins_a[i].setdefault(v_id, t)

        def expand(chain, segs, ins, station):
            out = [chain[0]]
            for i in range(len(chain) - 1):
                extra = sorted(ins[i].items(), key=lambda kv: kv[1])
                out.extend(v for v, _ in extra)
                out.append(chain[i + 1])
                if extra:
                    kind_e, k_e, forward = segs[i]
                    lst = extras.setdefault((station, kind_e, k_e), [])
                    for v, t in extra:
                        lst.append((t if forward else 1.0 - t, v))
            return out

        full_a = expand(chain_a, seg_a, ins_a, a)
        full_b = expand(chain_b, seg_b, ins_b, b)
        poly = full_a + full_b[::-1][1:-1]
        pos3 = np.vstack(nodes)
        for lobe in _split_lobes(poly):
            rel = pos3[lobe] - origin[a]
            xy = np.column_stack([rel @ cdir[a], rel @ ndir[a]])
            for t3 in _ear_clip(lobe, xy):
                side_tris.append([t3[0], t3[1], t3[2], t3[2]])

    pos3 = np.vstack(nodes)

    def edge_chain(j: int, kind_e: str, k: int, first: int, second: int) -> list[int]:
        extra = sorted(extras.get((j, kind_e, k), []), key=lambda tv: tv[0])
        return [first] + [v for _, v in extra] + [second]

    panels: list[list[int]] = []
    kinds: list[str] = []
    strip_of: list[int] = []
    chord_of: list[int] = []
    side_of: list[int] = []

    def add(rows: list[list[int]], kind: str, s: int, k: int, sd: int, flip: bool = False) -> list[int]:
        out = []
        for r in rows:
            panels.append(_flip_panel(r) if flip else [int(v) for v in r])
            kinds.append(kind)
            strip_of.append(s)
            chord_of.append(k)
            side_of.append(sd)
            out.append(len(panels) - 1)
        return out

    wake: list[list[int]] = []
    s = 0
    for j in range(ns - 1):
        if dup_next[j]:
            continue
        up_te: list[int] = []
        lo_te: list[int] = []
        for k in range(nc):
            p_u = edge_chain(j, "u", k, int(uid[j, k]), int(uid[j, k + 1]))
            q_u = edge_chain(j + 1, "u", k, int(uid[j + 1, k]), int(uid[j + 1, k + 1]))
            got_u = add(_polygon_panels(p_u, q_u, pos3), "surface", s, k, 1)
            p_l = edge_chain(j, "l", k, int(lid[j, k]), int(lid[j, k + 1]))
            q_l = edge_chain(j + 1, "l", k, int(lid[j + 1, k]), int(lid[j + 1, k + 1]))
            got_l = add(_polygon_panels(p_l, q_l, pos3), "surface", s, k, -1, flip=True)
            if k == nc - 1:
                up_te, lo_te = got_u, got_l
        if not (sharp[j] and sharp[j + 1]):
            p_b = edge_chain(j, "b", nc, int(uid[j, nc]), int(lid[j, nc]))
            q_b = edge_chain(j + 1, "b", nc, int(uid[j + 1, nc]), int(lid[j + 1, nc]))
            add(_polygon_panels(p_b, q_b, pos3), "base", s, nc, 0)
        ua, ub = int(uid[j, nc]), int(uid[j + 1, nc])
        la, lb = int(lid[j, nc]), int(lid[j + 1, nc])
        pu = [p for p in up_te if ua in panels[p] and ub in panels[p]]
        pl = [p for p in lo_te if la in panels[p] and lb in panels[p]]
        if pu and pl:
            wake.append([pu[0], pl[0], ua, ub])
        s += 1
    n_strips = s
    add(side_tris, "side", -1, -1, 0)

    def add_cap(station: int, flip: bool) -> None:
        u = uid[station]
        v = lid[station]
        if nc == 1:
            q = [(u[0], u[1], v[1], v[1])]
        else:
            q = [(u[0], u[1], v[1], v[1])]
            for k in range(1, nc - 1):
                q.append((u[k], u[k + 1], v[k + 1], v[k]))
            if sharp[station]:
                q.append((u[nc - 1], u[nc], v[nc - 1], v[nc - 1]))
            else:
                q.append((u[nc - 1], u[nc], v[nc], v[nc - 1]))
        rows = []
        for p1, p2, p3, p4 in q:
            ring = [int(p1), int(p2), int(p3), int(p4)]
            if len(set(ring)) < 3:
                continue
            rows.append(ring)
        add(rows, "tip", -1, -1, 0, flip=flip)

    if cap_last:
        add_cap(ns - 1, flip=False)
    if cap_first:
        add_cap(0, flip=True)
    return {
        "nodes": nodes,
        "station_of": station_of,
        "panels": panels,
        "kinds": kinds,
        "strip": strip_of,
        "chord_index": chord_of,
        "side": side_of,
        "wake": wake,
        "n_strips": n_strips,
    }


def _station_rows(qc, chord, cdir, ndir, xu, zu, xl, zl) -> tuple[np.ndarray, np.ndarray]:
    """Return the 3D upper and lower rows of stations [m]."""
    ns = qc.shape[0]
    nc1 = xu.shape[1]
    upper = np.empty((ns, nc1, 3))
    lower = np.empty((ns, nc1, 3))
    for j in range(ns):
        for k in range(nc1):
            upper[j, k] = qc[j] + chord[j] * ((xu[j, k] - 0.25) * cdir[j] + zu[j, k] * ndir[j])
            lower[j, k] = qc[j] + chord[j] * ((xl[j, k] - 0.25) * cdir[j] + zl[j, k] * ndir[j])
    return upper, lower


def _mesh_with_controls(
    surf,
    coord_sorted: list[np.ndarray],
    fr: np.ndarray,
    chords: np.ndarray,
    twists: np.ndarray,
    xt_new: np.ndarray,
    eta_base: np.ndarray,
    active: list,
    tip: str,
    half: bool,
    scale_root_thickness: bool = True,
    trailing_edge: str = "close",
    control_edge: str = "transition",
    control_transition: float = CONTROL_TRANSITION,
    fold_guard: bool = True,
) -> PanelMesh:
    """Build a watertight mesh with deflected plain control surfaces."""
    from ventorum.geometry.controls import validate_controls

    validate_controls(surf)
    nc = int(xt_new.size - 1)
    yu_mat = []
    yl_mat = []
    for c in coord_sorted:
        yu, yl = _resample_section(c, xt_new)
        if trailing_edge == "close":
            yu, yl = _close_trailing_edge(xt_new, yu, yl)
        yu_mat.append(yu)
        yl_mat.append(yl)
    yu_mat = np.vstack(yu_mat)
    yl_mat = np.vstack(yl_mat)
    symmetric = bool(getattr(surf, "is_symmetric", False))
    mirror = bool(getattr(surf, "mirror_y", False))
    eta_base = np.asarray(eta_base, dtype=float)
    if control_edge == "transition":
        edge_etas, edge_ctrls, edge_factor = _transition_edge_lists(
            surf, eta_base, active, lambda e: float(np.interp(e, fr, chords)), control_transition,
        )
    else:
        edge_etas, edge_ctrls = _edge_lists(eta_base, active)
        edge_factor = np.ones(edge_etas.size)
    ns = int(edge_etas.size)
    yu_base = _interp_rows(fr, yu_mat, edge_etas)
    yl_base = _interp_rows(fr, yl_mat, edge_etas)
    sharp_base = np.abs(yu_base[:, -1] - yl_base[:, -1]) <= _TE_GAP_TOL
    if scale_root_thickness:
        factor = _root_thickness_factor(surf, edge_etas)
        if factor != 1.0:
            _warn_strong_root_scaling(surf, factor)
            yu_base[0], yl_base[0] = _scale_about_camber(yu_base[0], yl_base[0], factor)
    has_asym = any(not bool(c.symmetric) for c in active)
    if symmetric and has_asym and half:
        raise ValueError(
            "A half mesh (half=True) needs symmetric control deflections; "
            "an antisymmetric control makes the geometry asymmetric about y = 0."
        )

    chord_e = np.interp(edge_etas, fr, chords)
    twist_e = np.interp(edge_etas, fr, twists) + float(surf.incidence)
    x_ref, y_ref, z_ref = surface_reference_line(surf, edge_etas)
    offset = np.asarray(surf.position, dtype=float).reshape(3)
    le_e = np.column_stack([x_ref, y_ref, z_ref]) + offset
    axis_e = _span_axes_for_edges(surf, edge_etas)
    qc_e = le_e + 0.25 * chord_e[:, None] * _XHAT

    def station_rows(j: int, mirror_side: bool):
        ctrl = edge_ctrls[j]
        delta = _right_delta_deg(ctrl, mirror_side) * float(edge_factor[j])
        return _deflected_station(xt_new, yu_base[j], yl_base[j], ctrl, delta)

    def section_rows(mirror_side: bool):
        xu = np.empty((ns, nc + 1))
        zu = np.empty((ns, nc + 1))
        xl = np.empty((ns, nc + 1))
        zl = np.empty((ns, nc + 1))
        for j in range(ns):
            xu[j], zu[j], xl[j], zl[j] = station_rows(j, mirror_side)
        return xu, zu, xl, zl

    sides = (False, True) if (symmetric and has_asym) else (bool(mirror and not symmetric),)
    if fold_guard:
        rows_by_side = [section_rows(m) for m in sides]

        def offsets(j: int, tilt: float):
            f = 1.0 / np.cos(tilt) if (scale_root_thickness and tilt != 0.0) else 1.0
            lo, hi = np.inf, -np.inf
            for xu_s, zu_s, xl_s, zl_s in rows_by_side:
                a, b = _station_offsets(xu_s[j], zu_s[j], xl_s[j], zl_s[j], float(chord_e[j]), float(twist_e[j]), f)
                lo, hi = min(lo, a), max(hi, b)
            return lo, hi

        axis_e, tilt_e = _guard_axes(axis_e, qc_e, offsets)
        if scale_root_thickness:
            for j in np.flatnonzero(tilt_e != 0.0):
                yu_base[j], yl_base[j] = _scale_about_camber(yu_base[j], yl_base[j], 1.0 / np.cos(tilt_e[j]))
    cdir_e, ndir_e = _section_frames(axis_e, twist_e)

    if symmetric and has_asym:
        # Full mesh from the left tip to the right tip: the left side is the
        # mirror image of the right side with the antisymmetric deflection.
        rows_r = section_rows(False)
        rows_l = section_rows(True)
        flip_y = np.array([1.0, -1.0, 1.0])
        merge_root = all(
            np.array_equal(rows_r[i][0], rows_l[i][0]) for i in range(4)
        )
        left = np.arange(ns)[::-1][:-1] if merge_root else np.arange(ns)[::-1]
        right = np.arange(ns)
        sel = np.concatenate([left, right])
        is_left = np.concatenate([np.ones(left.size, dtype=bool), np.zeros(right.size, dtype=bool)])
        n_full = sel.size
        signed = np.where(is_left, -edge_etas[sel], edge_etas[sel])
        qc = np.where(is_left[:, None], qc_e[sel] * flip_y, qc_e[sel])
        cdir = np.where(is_left[:, None], cdir_e[sel] * flip_y, cdir_e[sel])
        ndir = np.where(is_left[:, None], ndir_e[sel] * flip_y, ndir_e[sel])
        chord_f = chord_e[sel]
        xu = np.where(is_left[:, None], rows_l[0][sel], rows_r[0][sel])
        zu = np.where(is_left[:, None], rows_l[1][sel], rows_r[1][sel])
        xl = np.where(is_left[:, None], rows_l[2][sel], rows_r[2][sel])
        zl = np.where(is_left[:, None], rows_l[3][sel], rows_r[3][sel])
        upper, lower = _station_rows(qc, chord_f, cdir, ndir, xu, zu, xl, zl)
        built = _build_station_mesh(
            upper, lower, sharp_base[sel], signed, chord_f, qc, cdir, ndir,
            cap_first=(tip == "flat"), cap_last=(tip == "flat"),
        )
        return _finish_mesh(built, n_full, half=False, tip=tip, mirror_full=False, mirror_flip=False)

    mirror_side = bool(mirror and not symmetric)
    xu, zu, xl, zl = section_rows(mirror_side)
    upper, lower = _station_rows(qc_e, chord_e, cdir_e, ndir_e, xu, zu, xl, zl)
    built = _build_station_mesh(
        upper, lower, sharp_base, edge_etas, chord_e, qc_e, cdir_e, ndir_e,
        cap_first=(tip == "flat" and not symmetric), cap_last=(tip == "flat"),
    )
    return _finish_mesh(
        built, ns, half=bool(half and symmetric), tip=tip,
        mirror_full=bool(symmetric and not half), mirror_flip=bool(mirror and not symmetric),
    )


def _finish_mesh(built: dict, ns: int, half: bool, tip: str, mirror_full: bool, mirror_flip: bool) -> PanelMesh:
    """Mirror a one-sided build when needed and return the PanelMesh.

    Parameters
    ----------
    built : dict
        Output of :func:`_build_station_mesh`.
    ns : int
        Number of station rows of the build.
    half : bool
        Half mesh flag of the result.
    tip : str
        Tip closure mode.
    mirror_full : bool
        Add the mirror image about y = 0 (stations > 0 mirrored, root shared).
    mirror_flip : bool
        Replace the build by its mirror image about y = 0.

    Returns
    -------
    PanelMesh
        Mesh with metadata.
    """
    nodes = [np.asarray(p, dtype=float) for p in built["nodes"]]
    station = list(built["station_of"])
    panels = [list(r) for r in built["panels"]]
    kinds = list(built["kinds"])
    strip = list(built["strip"])
    chord_i = list(built["chord_index"])
    side = list(built["side"])
    wake = [list(w) for w in built["wake"]]
    n_s = int(built["n_strips"])
    if mirror_full:
        right_n = len(nodes)
        index_map = np.full(right_n, -1, dtype=int)
        node_station = []
        for i in range(right_n):
            node_station.append(ns - 1 + station[i])
        for i in range(right_n):
            if station[i] == 0:
                index_map[i] = i
            else:
                p = nodes[i].copy()
                p[1] = -p[1]
                nodes.append(p)
                node_station.append(ns - 1 - station[i])
                index_map[i] = len(nodes) - 1
        n_right = len(panels)
        for pi in range(n_right):
            panels.append(_flip_panel([index_map[v] for v in panels[pi]]))
            kinds.append(kinds[pi])
            chord_i.append(chord_i[pi])
            side.append(side[pi])
            strip.append(n_s - 1 - strip[pi] if strip[pi] >= 0 else -1)
        for pi in range(n_right):
            if strip[pi] >= 0:
                strip[pi] = n_s + strip[pi]
        for w in list(wake):
            wake.append([w[0] + n_right, w[1] + n_right, int(index_map[w[2]]), int(index_map[w[3]])])
    else:
        node_station = list(station)
        if mirror_flip:
            for p in nodes:
                p[1] = -p[1]
            panels = [_flip_panel(r) for r in panels]
    panel_arr = np.array(panels, dtype=int).reshape(-1, 4)
    return PanelMesh(
        nodes=np.vstack(nodes),
        panels=panel_arr,
        component=np.zeros(panel_arr.shape[0], dtype=int),
        kind=np.array(kinds, dtype=str),
        wake_edges=np.array(wake, dtype=int).reshape(-1, 4),
        half=bool(half),
        tip=str(tip),
        strip=np.array(strip, dtype=int),
        chord_index=np.array(chord_i, dtype=int),
        side=np.array(side, dtype=int),
        node_station=np.array(node_station, dtype=int),
    )


def mesh_lifting_surface(
    surface,
    section_coords,
    n_chord: int = 40,
    chord_spacing: str = "cosine",
    tip: str = "flat",
    half: bool = False,
    n_span=None,
    scale_root_thickness: bool = True,
    trailing_edge: str = "close",
    control_edge: str = "transition",
    control_transition: float = CONTROL_TRANSITION,
    fold_guard: bool = True,
) -> PanelMesh:
    """Build a closed quadrilateral mesh of one thick lifting surface.

    Parameters
    ----------
    surface : LiftingSurface
        Surface definition (planform, twist, sweep, dihedral [rad]).
    section_coords : numpy.ndarray or list of numpy.ndarray
        One (N, 2) unit-chord Selig array for all sections, or one array
        per section in the order of ``surface.sections``.
    n_chord : int
        Chordwise panels per surface (upper and lower).
    chord_spacing : str
        "cosine" (clustered at leading and trailing edge) or "uniform" in x.
    tip : str
        "flat" (flat tip cap) or "none" (open tip, only for tests).
    half : bool
        True keeps only the right half, open at y = 0 [m]. Without effect
        on a non-symmetric surface (it is already one side).
    n_span : int or None
        Spanwise panels per semi-span. None uses the surface value when set,
        else 20. The stations match the lattice builder for that count.
    scale_root_thickness : bool
        True (default) multiplies the offsets of the root section of a
        symmetric surface from its camber line (in the plane y = 0) by
        1 / cos(G), with G the dihedral of the first panel row [rad], so
        that the thickness normal to the local surface equals the section
        thickness. False keeps the thickness of the root section along its
        section normal (the mesh before this option, bit for bit). With
        ``fold_guard`` the same factor 1 / cos(B) applies to a station that
        the guard turns by B [rad]. The factor is not capped; above a
        first-row dihedral of ``ROOT_SCALE_WARN_DIHEDRAL_DEG`` (60 deg) the
        builder issues a UserWarning.
    trailing_edge : str
        "close" (default) closes an open trailing edge with the quartic
        thickness change of the module notes; "base" keeps the open edge
        with base panels and the wake at the upper corner (the former mesh;
        its results do not converge with chordwise refinement).
    control_edge : str
        "transition" (default) changes the deflection of a control linearly
        over a spanwise transition centred on each side edge that borders
        an undeflected part; "gap" keeps two stations at the same y with
        flat side panels (the former mesh). An edge between two deflected
        controls is always a gap.
    control_transition : float
        Length of the transition over the local chord [-] (default
        ``CONTROL_TRANSITION``). It is shortened where the control, the
        root, the tip or the next edge leave less room.
    fold_guard : bool
        True (default) turns a station plane toward the plane of its inboard
        neighbour when the two sections would cross (see the module notes);
        stations that do not fold do not change. False keeps every section
        plane normal to the span (the former mesh, bit for bit).

    Returns
    -------
    PanelMesh
        Closed mesh with outward panels, wake shedding edges and the strip
        metadata (``strip``, ``chord_index``, ``side``, ``node_station``).
        When the surface carries control surfaces with nonzero deflection
        [rad], the sections inside each control range use the deflected
        shape.

    Raises
    ------
    ValueError
        For bad inputs, and for ``half=True`` with an antisymmetric control
        deflection on a symmetric surface.

    Warns
    -----
    UserWarning
        When ``scale_root_thickness`` scales the root section and the
        dihedral of the first panel row is above
        ``ROOT_SCALE_WARN_DIHEDRAL_DEG`` (60 deg): the root section is
        strongly scaled and the result needs checking.

    Notes
    -----
    Each section lies in the plane of the local chord direction and the
    local section normal (the normal of the lattice surface at the station),
    so the section thickness follows the dihedral; the fold guard can turn
    the planes of a few stations next to the root or a dihedral break. The
    control deflection [rad] follows the VLM sign convention (positive
    trailing edge down on a horizontal surface). The chordwise stations
    gain a node at each active hinge. A transition edge adds a station at
    each end of the transition; a gap edge adds flat side triangles between
    two stations at the same y [m]. Zero deflection keeps the mesh bit for
    bit.

    At the root of a symmetric surface the section lies in the plane y = 0
    (the miter of the two halves). There a thickness t along the section
    normal is t cos(G) normal to the surface of the first panel row, with G
    its dihedral. ``scale_root_thickness`` corrects this (see the module
    notes). Only symmetric surfaces with a nonzero dihedral of the first
    panel row change; a one-sided surface, for example a vertical fin, has
    its root section normal to its span axis and does not change.
    """
    surf = surface
    n_chord = int(n_chord)
    if n_chord < 1:
        raise ValueError(f"n_chord must be at least 1, got {n_chord}.")
    if tip not in ("flat", "none"):
        raise ValueError(f"Unknown tip {tip!r}; use 'flat' or 'none'.")
    if trailing_edge not in _TRAILING_EDGE_OPTIONS:
        raise ValueError(f"Unknown trailing_edge {trailing_edge!r}; use 'close' or 'base'.")
    if control_edge not in _CONTROL_EDGE_OPTIONS:
        raise ValueError(f"Unknown control_edge {control_edge!r}; use 'transition' or 'gap'.")
    control_transition = float(control_transition)
    if not np.isfinite(control_transition) or control_transition <= 0.0:
        raise ValueError(f"control_transition must be positive, got {control_transition}.")
    secs = list(surf.sections)
    if len(secs) < 2:
        raise ValueError("A lifting surface needs at least two sections.")
    if n_span is None:
        n_span_eff = int(surf.n_panels) if surf.n_panels is not None else 20
    else:
        n_span_eff = int(n_span)
    if n_span_eff < 1:
        raise ValueError(f"n_span must be at least 1, got {n_span_eff}.")

    active = _active_controls(surf)
    if active:
        order_c = sorted(range(len(secs)), key=lambda i: float(secs[i].y_frac))
        secs_sorted_c = [secs[i] for i in order_c]
        coord_list_c = _as_coord_list(section_coords, len(secs))
        coord_sorted_c = [coord_list_c[i] for i in order_c]
        fr_c = np.array([float(s.y_frac) for s in secs_sorted_c])
        chords_c = np.array([float(s.chord) for s in secs_sorted_c])
        twists_c = np.array([float(s.twist) for s in secs_sorted_c])
        xt_base_c = _target_x(n_chord, chord_spacing)
        xt_new_c = _union_xt(xt_base_c, active)
        spacing_c = resolve_spacing(surf.spacing, surf, "vlm")
        eta_base_c, _ = _surface_eta(surf, n_span_eff, spacing_c)
        return _mesh_with_controls(
            surf, coord_sorted_c, fr_c, chords_c, twists_c, xt_new_c, eta_base_c, active, tip, half,
            scale_root_thickness=bool(scale_root_thickness),
            trailing_edge=trailing_edge,
            control_edge=control_edge,
            control_transition=control_transition,
            fold_guard=bool(fold_guard),
        )

    order = sorted(range(len(secs)), key=lambda i: float(secs[i].y_frac))
    secs_sorted = [secs[i] for i in order]
    coord_list = _as_coord_list(section_coords, len(secs))
    coord_sorted = [coord_list[i] for i in order]
    fr = np.array([float(s.y_frac) for s in secs_sorted])
    chords = np.array([float(s.chord) for s in secs_sorted])
    twists = np.array([float(s.twist) for s in secs_sorted])

    xt = _target_x(n_chord, chord_spacing)
    nc1 = n_chord + 1
    yu_rows = []
    yl_rows = []
    for c in coord_sorted:
        yu, yl = _resample_section(c, xt)
        if trailing_edge == "close":
            yu, yl = _close_trailing_edge(xt, yu, yl)
        yu_rows.append(yu)
        yl_rows.append(yl)
    yu_mat = np.vstack(yu_rows)
    yl_mat = np.vstack(yl_rows)

    spacing = resolve_spacing(surf.spacing, surf, "vlm")
    eta_e, _ = _surface_eta(surf, n_span_eff, spacing)
    eta_e = np.asarray(eta_e, dtype=float)
    ns = len(eta_e)
    chord_e = np.interp(eta_e, fr, chords)
    twist_e = np.interp(eta_e, fr, twists) + float(surf.incidence)
    x_ref, y_ref, z_ref = surface_reference_line(surf, eta_e)
    offset = np.asarray(surf.position, dtype=float).reshape(3)
    le_flat = np.column_stack([x_ref, y_ref, z_ref]) + offset
    if ns >= 2:
        dy = np.gradient(y_ref, eta_e)
        dz = np.gradient(z_ref, eta_e)
    else:  # pragma: no cover - one spanwise panel still has two edge stations
        dy = np.zeros_like(eta_e)
        dz = np.zeros_like(eta_e)
    axis = np.column_stack([np.zeros_like(eta_e), dy, dz])
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-300)
    if surf.is_symmetric:
        axis[eta_e <= 1e-12] = np.array([0.0, 1.0, 0.0])
    qc = le_flat + 0.25 * chord_e[:, None] * _XHAT
    yu_e = _interp_rows(fr, yu_mat, eta_e)
    yl_e = _interp_rows(fr, yl_mat, eta_e)
    sharp = np.abs(yu_e[:, -1] - yl_e[:, -1]) <= _TE_GAP_TOL
    if scale_root_thickness:
        factor = _root_thickness_factor(surf, eta_e)
        if factor != 1.0:
            _warn_strong_root_scaling(surf, factor)
            yu_e[0], yl_e[0] = _scale_about_camber(yu_e[0], yl_e[0], factor)
    if fold_guard:

        def offsets(j: int, tilt: float):
            f = 1.0 / np.cos(tilt) if (scale_root_thickness and tilt != 0.0) else 1.0
            return _station_offsets(xt, yu_e[j], xt, yl_e[j], float(chord_e[j]), float(twist_e[j]), f)

        axis, tilt_e = _guard_axes(axis, qc, offsets)
        if scale_root_thickness:
            for j in np.flatnonzero(tilt_e != 0.0):
                yu_e[j], yl_e[j] = _scale_about_camber(yu_e[j], yl_e[j], 1.0 / np.cos(tilt_e[j]))
    cdir, ndir = _section_frames(axis, twist_e)

    upper = np.empty((ns, nc1, 3))
    lower = np.empty((ns, nc1, 3))
    for j in range(ns):
        upper[j] = qc[j] + chord_e[j] * ((xt - 0.25)[:, None] * cdir[j] + yu_e[j][:, None] * ndir[j])
        lower[j] = qc[j] + chord_e[j] * ((xt - 0.25)[:, None] * cdir[j] + yl_e[j][:, None] * ndir[j])

    nodes: list[np.ndarray] = []
    station_of: list[int] = []

    def add_node(p: np.ndarray, station: int) -> int:
        nodes.append(np.asarray(p, dtype=float).reshape(3))
        station_of.append(int(station))
        return len(nodes) - 1

    uid = np.full((ns, nc1), -1, dtype=int)
    lid = np.full((ns, nc1), -1, dtype=int)
    for j in range(ns):
        uid[j, 0] = lid[j, 0] = add_node(0.5 * (upper[j, 0] + lower[j, 0]), j)
        for k in range(1, n_chord):
            uid[j, k] = add_node(upper[j, k], j)
            lid[j, k] = add_node(lower[j, k], j)
        if sharp[j]:
            uid[j, n_chord] = lid[j, n_chord] = add_node(0.5 * (upper[j, n_chord] + lower[j, n_chord]), j)
        else:
            uid[j, n_chord] = add_node(upper[j, n_chord], j)
            lid[j, n_chord] = add_node(lower[j, n_chord], j)

    panels: list[list[int]] = []
    kinds: list[str] = []
    strip_of: list[int] = []
    chord_of: list[int] = []
    side_of: list[int] = []
    up_pan = np.full((ns - 1, n_chord), -1, dtype=int)
    lo_pan = np.full((ns - 1, n_chord), -1, dtype=int)

    def add_panel(row: list[int], kind: str, s: int, k: int, sd: int) -> int:
        panels.append([int(v) for v in row])
        kinds.append(kind)
        strip_of.append(s)
        chord_of.append(k)
        side_of.append(sd)
        return len(panels) - 1

    for j in range(ns - 1):
        for k in range(n_chord):
            up_pan[j, k] = add_panel([uid[j, k], uid[j, k + 1], uid[j + 1, k + 1], uid[j + 1, k]], "surface", j, k, 1)
            lo_pan[j, k] = add_panel([lid[j, k], lid[j + 1, k], lid[j + 1, k + 1], lid[j, k + 1]], "surface", j, k, -1)
        if sharp[j] and sharp[j + 1]:
            pass
        else:
            add_panel([uid[j, n_chord], lid[j, n_chord], lid[j + 1, n_chord], uid[j + 1, n_chord]], "base", j, n_chord, 0)

    def add_cap(station: int, flip: bool) -> None:
        """Flat cap of the airfoil loop at one spanwise station."""
        u = uid[station]
        v = lid[station]
        if n_chord == 1:
            q: list[tuple[int, int, int, int]] = [(u[0], u[1], v[1], v[1])]
        else:
            q = [(u[0], u[1], v[1], v[1])]
            for k in range(1, n_chord - 1):
                q.append((u[k], u[k + 1], v[k + 1], v[k]))
            if sharp[station]:
                q.append((u[n_chord - 1], u[n_chord], v[n_chord - 1], v[n_chord - 1]))
            else:
                q.append((u[n_chord - 1], u[n_chord], v[n_chord], v[n_chord - 1]))
        for row in q:
            add_panel(_flip_panel(row) if flip else list(row), "tip", -1, -1, 0)

    symmetric = bool(surf.is_symmetric)
    if tip == "flat":
        add_cap(ns - 1, flip=False)
        if not symmetric:
            add_cap(0, flip=True)

    wake: list[list[int]] = []
    for j in range(ns - 1):
        wake.append([int(up_pan[j, n_chord - 1]), int(lo_pan[j, n_chord - 1]), int(uid[j, n_chord]), int(uid[j + 1, n_chord])])

    built = {
        "nodes": nodes,
        "station_of": station_of,
        "panels": panels,
        "kinds": kinds,
        "strip": strip_of,
        "chord_index": chord_of,
        "side": side_of,
        "wake": wake,
        "n_strips": ns - 1,
    }
    mirror = bool(getattr(surf, "mirror_y", False))
    return _finish_mesh(
        built, ns, half=bool(half and symmetric), tip=tip,
        mirror_full=bool(symmetric and not half), mirror_flip=bool(mirror and not symmetric),
    )


def mesh_aircraft(
    aircraft,
    section_coords_by_surface,
    scale_root_thickness: bool = True,
    **kw,
) -> PanelMesh:
    """Build one panel mesh for all surfaces of an aircraft.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft with the lifting surfaces.
    section_coords_by_surface : dict or list
        Dict from surface name (or index) to the section coordinates of
        :func:`mesh_lifting_surface`, or a list in surface order.
    scale_root_thickness : bool
        Root-section thickness option of :func:`mesh_lifting_surface` for
        every surface (default True).
    **kw
        Extra arguments of :func:`mesh_lifting_surface` (n_chord, n_span,
        chord_spacing, tip, half, trailing_edge, control_edge,
        control_transition, fold_guard).

    Returns
    -------
    PanelMesh
        Combined mesh with one component index per surface. Strip indices
        are offset so that they are unique over the mesh; node stations
        stay per component.
    """
    surfs = list(aircraft.surfaces)
    if isinstance(section_coords_by_surface, dict):
        coord_list = []
        for i, s in enumerate(surfs):
            if s.name in section_coords_by_surface:
                coord_list.append(section_coords_by_surface[s.name])
            elif i in section_coords_by_surface:
                coord_list.append(section_coords_by_surface[i])
            elif str(i) in section_coords_by_surface:
                coord_list.append(section_coords_by_surface[str(i)])
            else:
                raise KeyError(f"No section coordinates for surface {s.name!r}.")
    else:
        coord_list = list(section_coords_by_surface)
        if len(coord_list) != len(surfs):
            raise ValueError(f"Got {len(coord_list)} coordinate sets for {len(surfs)} surfaces.")
    node_parts = []
    panel_parts = []
    kind_parts = []
    comp_parts = []
    wake_parts = []
    strip_parts = []
    chord_parts = []
    side_parts = []
    station_parts = []
    n_nodes = 0
    n_panels = 0
    n_strips = 0
    half_all = True
    for i, (s, c) in enumerate(zip(surfs, coord_list)):
        sub = mesh_lifting_surface(s, c, scale_root_thickness=scale_root_thickness, **kw)
        node_parts.append(np.asarray(sub.nodes, dtype=float))
        panel_parts.append(np.asarray(sub.panels, dtype=int) + n_nodes)
        kind_parts.append(np.asarray(sub.kind))
        comp_parts.append(np.full(np.asarray(sub.panels).shape[0], i, dtype=int))
        w = np.asarray(sub.wake_edges, dtype=int).copy()
        if w.size:
            w[:, 0:2] += n_panels
            w[:, 2:4] += n_nodes
        wake_parts.append(w.reshape(-1, 4))
        st = np.asarray(sub.strip, dtype=int)
        strip_parts.append(np.where(st >= 0, st + n_strips, -1))
        chord_parts.append(np.asarray(sub.chord_index, dtype=int))
        side_parts.append(np.asarray(sub.side, dtype=int))
        station_parts.append(np.asarray(sub.node_station, dtype=int))
        n_strips += int(np.max(st)) + 1 if np.any(st >= 0) else 0
        n_nodes += int(np.asarray(sub.nodes).shape[0])
        n_panels += int(np.asarray(sub.panels).shape[0])
        half_all = half_all and bool(sub.half)
    return PanelMesh(
        nodes=np.vstack(node_parts),
        panels=np.vstack(panel_parts),
        component=np.concatenate(comp_parts),
        kind=np.concatenate(kind_parts),
        wake_edges=np.vstack(wake_parts).reshape(-1, 4),
        half=bool(half_all and len(surfs) > 0),
        tip=str(kw.get("tip", "flat")),
        strip=np.concatenate(strip_parts),
        chord_index=np.concatenate(chord_parts),
        side=np.concatenate(side_parts),
        node_station=np.concatenate(station_parts),
    )


def _panel_geometry(p: np.ndarray) -> tuple[float, np.ndarray, float, float, float, bool]:
    """Area [m^2], unit normal, aspect ratio, skew, warp and convexity of one panel."""
    p1, p2, p3, p4 = (np.asarray(q, dtype=float).reshape(3) for q in p)
    n = np.cross(p3 - p1, p4 - p2)
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-300:
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-300:
        n = np.cross(p3 - p1, p4 - p1)
        n_norm = float(np.linalg.norm(n))
    unit = n / n_norm if n_norm > 1e-300 else np.zeros(3)
    area = 0.5 * float(np.linalg.norm(np.cross(p2 - p1, p3 - p1)))
    area += 0.5 * float(np.linalg.norm(np.cross(p3 - p1, p4 - p1)))
    uniq = [p1, p2, p3, p4]
    loop = []
    for q in uniq:
        if not loop or float(np.linalg.norm(q - loop[-1])) > 0.0:
            loop.append(q)
    if len(loop) > 1 and float(np.linalg.norm(loop[-1] - loop[0])) == 0.0:
        loop.pop()
    edges = []
    for i in range(len(loop)):
        edges.append(float(np.linalg.norm(loop[(i + 1) % len(loop)] - loop[i])))
    edges = [e for e in edges if e > 0.0]
    aspect = float(max(edges) / min(edges)) if len(edges) >= 2 else 1.0
    skew = 0.0
    m = len(loop)
    for i in range(m):
        a = loop[i - 1] - loop[i]
        b = loop[(i + 1) % m] - loop[i]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na > 0.0 and nb > 0.0:
            skew = max(skew, abs(float(np.dot(a, b))) / (na * nb))
    pts = np.vstack(loop)
    diam = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            diam = max(diam, float(np.linalg.norm(pts[i] - pts[j])))
    if diam > 0.0 and float(np.linalg.norm(unit)) > 0.0:
        mean = np.mean(pts, axis=0)
        warp = float(np.max(np.abs((pts - mean) @ unit)) / diam)
    else:
        warp = 0.0
    # Convexity in the mean plane: every corner turns the same way as the normal.
    convex = m >= 3 and diam > 0.0 and float(np.linalg.norm(unit)) > 0.0
    if convex:
        tol = 1e-12 * diam * diam
        for i in range(m):
            turn = float(np.dot(np.cross(loop[i] - loop[i - 1], loop[(i + 1) % m] - loop[i]), unit))
            if turn < -tol:
                convex = False
                break
    return area, unit, aspect, float(skew), float(warp), bool(convex)


def _panel_geometry_batch(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Area [m^2], unit normal, aspect ratio [-], skew [-], warp [-] and convexity of each panel.

    Vectorized twin of :func:`_panel_geometry`: all panels at once, with the
    same special cases. The normal falls back through the three cross
    products when the diagonals are degenerate. The distinct-corner loop
    drops the consecutive repeats and one closing repeat. Edge lengths count
    only above zero; the aspect ratio is 1.0 with fewer than two edges. The
    diameter runs over the up to six corner pairs of the distinct loop. The
    warp is 0 for a zero diameter or a zero normal. The convexity test turns
    every corner of the distinct loop the same way as the normal, with the
    tolerance 1e-12 * diam**2.

    Parameters
    ----------
    p : numpy.ndarray
        Corner positions [m], shape (Np, 4, 3).

    Returns
    -------
    tuple of numpy.ndarray
        (area, unit, aspect, skew, warp, convex): areas [m^2] shape (Np,),
        unit normals shape (Np, 3), aspect ratios [-], skews [-] and warps
        [-] each shape (Np,), and the convex flags shape (Np,).
    """
    p1, p2, p3, p4 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    n0 = np.cross(p3 - p1, p4 - p2)
    n1 = np.cross(p2 - p1, p3 - p1)
    n2 = np.cross(p3 - p1, p4 - p1)
    nrm0 = np.linalg.norm(n0, axis=1)
    nrm1 = np.linalg.norm(n1, axis=1)
    nrm2 = np.linalg.norm(n2, axis=1)
    take1 = nrm0 <= 1e-300
    take2 = np.where(take1, nrm1, nrm0) <= 1e-300
    nvec = np.where(take2[:, None], n2, np.where(take1[:, None], n1, n0))
    nrm = np.where(take2, nrm2, np.where(take1, nrm1, nrm0))
    live = nrm > 1e-300
    unit = np.where(live[:, None], nvec / np.where(live, nrm, 1.0)[:, None], 0.0)
    area = 0.5 * (nrm1 + nrm2)

    n_p = p.shape[0]
    rows = np.arange(n_p)
    # The distinct-corner loop of each panel: drop the consecutive repeats,
    # then one closing repeat (the last kept corner equal to the first).
    keep = np.empty(p.shape[:2], dtype=bool)
    keep[:, 0] = True
    keep[:, 1:] = np.any(p[:, 1:] != p[:, :-1], axis=2)
    last = 3 - np.argmax(keep[:, ::-1], axis=1)
    closing = (np.sum(keep, axis=1) > 1) & np.all(p[rows, last] == p[:, 0], axis=1)
    keep[rows[closing], last[closing]] = False

    # Neighbour of each kept corner in the distinct loop (wrap at both ends).
    kk = np.arange(4)
    nxt = keep[:, :, None] & keep[:, None, :] & (kk[None, :] > kk[:, None])
    no_nxt = ~np.any(nxt, axis=2)
    nxt |= keep[:, None, :] & no_nxt[:, :, None]
    prv = keep[:, :, None] & keep[:, None, :] & (kk[None, :] < kk[:, None])
    no_prv = ~np.any(prv, axis=2)
    prv |= keep[:, None, :] & no_prv[:, :, None]
    nxt_p = np.take_along_axis(p, np.argmax(nxt, axis=2)[:, :, None], axis=1)
    prv_p = np.take_along_axis(p, (3 - np.argmax(prv[:, :, ::-1], axis=2))[:, :, None], axis=1)

    edges = np.linalg.norm(nxt_p - p, axis=2)
    live_edge = keep & (edges > 0.0)
    lengths = np.where(live_edge, edges, np.inf)
    e_min = np.min(lengths, axis=1)
    aspect = np.where(np.isfinite(e_min), np.max(np.where(live_edge, edges, 0.0), axis=1) / e_min, 1.0)

    va = prv_p - p
    vb = nxt_p - p
    na = np.linalg.norm(va, axis=2)
    nb = np.linalg.norm(vb, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        corner = np.where(keep & (na > 0.0) & (nb > 0.0), np.abs(np.sum(va * vb, axis=2)) / (na * nb), 0.0)
    skew = np.max(corner, axis=1)

    ii, jj = np.triu_indices(4, k=1)
    diam = np.max(np.where(keep[:, ii] & keep[:, jj], np.linalg.norm(p[:, ii] - p[:, jj], axis=2), 0.0), axis=1)
    count = np.sum(keep, axis=1)
    mean = np.sum(np.where(keep[:, :, None], p, 0.0), axis=1) / count[:, None]
    dev = np.abs((p - mean[:, None, :]) @ unit[:, :, None])[:, :, 0]
    dev = np.where(keep, dev, 0.0)
    live_unit = np.any(unit != 0.0, axis=1)
    warp = np.where((diam > 0.0) & live_unit, np.max(dev, axis=1) / np.where(diam > 0.0, diam, 1.0), 0.0)

    # Convexity in the mean plane: every corner turns the same way as the normal.
    turn = np.sum(np.cross(p - prv_p, nxt_p - p) * unit[:, None, :], axis=2)
    convex = (count >= 3) & (diam > 0.0) & live_unit & ~np.any(keep & (turn < -(1e-12 * diam * diam)[:, None]), axis=1)
    return area, unit, aspect, skew, warp, convex


def _station_folds(nodes: np.ndarray, panels: np.ndarray, station: np.ndarray, comp_of: np.ndarray) -> np.ndarray:
    """Return a fold flag for each panel from the planes of the spanwise stations.

    The nodes of one station row (``node_station``) of a component lie in
    one plane (the section plane). For two neighbour rows with different
    planes, every node of the outboard row must lie on the outboard side
    of the inboard plane, and every node of the inboard row on the inboard
    side of the outboard plane. A node on the wrong side means that the
    strip between the two rows folds (the surface passes through itself).
    Rows that are not plane (or have fewer than three nodes) are skipped,
    and so are two rows in the same plane (the two stations of a control
    gap).

    Parameters
    ----------
    nodes : numpy.ndarray
        Node positions [m], shape (Nn, 3).
    panels : numpy.ndarray
        Panel corner node indices, shape (Np, 4).
    station : numpy.ndarray
        Station row of each node, shape (Nn,), increasing along the span.
    comp_of : numpy.ndarray
        Component of each panel, shape (Np,).

    Returns
    -------
    numpy.ndarray
        True for each panel with a corner node on the wrong side, shape (Np,).
    """
    bad = np.zeros(nodes.shape[0], dtype=bool)
    for c in np.unique(comp_of):
        ids = np.unique(panels[comp_of == c].ravel())
        if ids.size < 4:
            continue
        st = station[ids]
        levels = np.unique(st)
        if levels.size < 2:
            continue
        size = float(np.max(np.ptp(nodes[ids], axis=0)))
        tol = 1e-9 * max(size, 1e-300)
        planes: dict = {}
        for lv in levels:
            pts = nodes[ids[st == lv]]
            if pts.shape[0] < 3:
                continue
            cen = np.mean(pts, axis=0)
            _, sv, vt = np.linalg.svd(pts - cen, full_matrices=False)
            if sv.size < 3 or sv[1] <= 1e-9 * max(float(sv[0]), 1e-300) or sv[2] > 1e-6 * float(sv[0]):
                continue
            planes[int(lv)] = (cen, vt[-1])
        ordered = sorted(planes)
        if len(ordered) < 2:
            continue
        # Orient each normal toward the next row that is not in the same
        # plane (else away from the previous one): two rows of a control gap
        # share a plane, and their centroids give no direction.
        for i, lv in enumerate(ordered):
            cen, nrm = planes[lv]
            ref = None
            for k in list(range(i + 1, len(ordered))) + list(range(i - 1, -1, -1)):
                step = planes[ordered[k]][0] - cen
                if abs(float(np.dot(step, nrm))) > tol:
                    ref = step if k > i else -step
                    break
            if ref is not None and float(np.dot(ref, nrm)) < 0.0:
                planes[lv] = (cen, -nrm)
        for a, b in zip(ordered[:-1], ordered[1:]):
            ca, na = planes[a]
            cb, nb = planes[b]
            if abs(float(np.dot(cb - ca, na))) <= tol and abs(abs(float(np.dot(na, nb))) - 1.0) <= 1e-9:
                continue
            ia = ids[st == a]
            ib = ids[st == b]
            bad[ib[(nodes[ib] - ca) @ na < -tol]] = True
            bad[ia[(nodes[ia] - cb) @ nb > tol]] = True
    return np.any(bad[panels], axis=1)


def _station_folds_batch(nodes: np.ndarray, panels: np.ndarray, station: np.ndarray, comp_of: np.ndarray) -> np.ndarray:
    """Fold flag for each panel from the station planes, with array operations.

    Vectorized twin of :func:`_station_folds`: the plane of each station row
    comes from one stacked SVD of the rows with equal point count, the
    orientation of the normals runs on the centroid matrix, and the node
    tests run on the two neighbour planes of each row. The rules match the
    loop. Rows with fewer than three points are skipped, and so are rows
    that fail the singular-value guards, two rows in one plane, and rows
    without a neighbour plane. A node on the wrong side of a neighbour
    plane means that the strip folds.

    Parameters
    ----------
    nodes : numpy.ndarray
        Node positions [m], shape (Nn, 3).
    panels : numpy.ndarray
        Panel corner node indices, shape (Np, 4).
    station : numpy.ndarray
        Station row of each node, shape (Nn,), increasing along the span.
    comp_of : numpy.ndarray
        Component of each panel, shape (Np,).

    Returns
    -------
    numpy.ndarray
        True for each panel with a corner node on the wrong side, shape (Np,).
    """
    bad = np.zeros(nodes.shape[0], dtype=bool)
    for c in np.unique(comp_of):
        ids = np.unique(panels[comp_of == c].ravel())
        if ids.size < 4:
            continue
        st = station[ids]
        levels = np.unique(st)
        if levels.size < 2:
            continue
        size = float(np.max(np.ptp(nodes[ids], axis=0)))
        tol = 1e-9 * max(size, 1e-300)
        # The points of every row, ordered by station and then by node id,
        # the same order as nodes[ids[st == level]] in the loop.
        order = np.argsort(st, kind="stable")
        flat = nodes[ids[order]]
        counts = np.bincount(np.searchsorted(levels, st[order]), minlength=levels.size)
        groups: list[tuple[np.ndarray, np.ndarray]] = []
        if int(counts.min()) == int(counts.max()):
            groups.append((np.arange(levels.size), flat.reshape(levels.size, int(counts[0]), 3)))
        else:
            # Rare: rows of one component with different point counts.
            rows = np.split(flat, np.cumsum(counts)[:-1])
            for g in np.unique(counts):
                sel = np.flatnonzero(counts == g)
                pts = rows[int(sel[0])][None, :, :] if sel.size == 1 else np.stack([rows[int(i)] for i in sel])
                groups.append((sel, pts))
        cen_all = np.zeros((levels.size, 3))
        nrm_all = np.zeros((levels.size, 3))
        ok_all = np.zeros(levels.size, dtype=bool)
        for sel, pts in groups:
            g = pts.shape[1]
            if g < 3:
                continue
            cen = pts.sum(axis=1) / float(g)
            _, sv, vt = np.linalg.svd(pts - cen[:, None, :], full_matrices=False)
            ok_all[sel] = (sv[:, 1] > 1e-9 * np.maximum(sv[:, 0], 1e-300)) & (sv[:, 2] <= 1e-6 * sv[:, 0])
            cen_all[sel] = cen
            nrm_all[sel] = vt[:, -1, :]
        plane = np.flatnonzero(ok_all)
        if plane.size < 2:
            continue
        cens = cen_all[plane]
        nrms = nrm_all[plane]
        m = plane.size
        rows_i = np.arange(m)
        # Orient each normal toward the next row that is not in the same
        # plane (else away from the previous one): two rows of a control gap
        # share a plane, and their centroids give no direction.
        diff = cens[None, :, :] - cens[:, None, :]
        turn = (diff * nrms[:, None, :]).sum(axis=2)
        big = np.abs(turn) > tol
        higher = big & (rows_i[None, :] > rows_i[:, None])
        lower = big & (rows_i[None, :] < rows_i[:, None])
        first_up = np.argmax(higher, axis=1)
        first_dn = m - 1 - np.argmax(lower[:, ::-1], axis=1)
        up_ok = higher[rows_i, first_up]
        dn_ok = lower[rows_i, first_dn]
        flip = np.where(up_ok, turn[rows_i, first_up] < 0.0,
                        np.where(dn_ok, turn[rows_i, first_dn] > 0.0, False))
        nrms = np.where(flip[:, None], -nrms, nrms)
        # The neighbour planes of a row: pair j joins plane j and plane j+1
        # and is skipped when the two rows share a plane.
        step = cens[1:] - cens[:-1]
        pair_d = (step * nrms[:-1]).sum(axis=1)
        pair_n = (nrms[:-1] * nrms[1:]).sum(axis=1)
        skip = (np.abs(pair_d) <= tol) & (np.abs(np.abs(pair_n) - 1.0) <= 1e-9)
        # Each node is tested against the row below and the row above it.
        lvl = np.searchsorted(levels, st)
        pos = np.searchsorted(plane, lvl)
        in_plane = np.zeros(ids.size, dtype=bool)
        has = pos < m
        in_plane[has] = plane[pos[has]] == lvl[has]
        at = nodes[ids]
        below = pos - 1
        above = pos + 1
        prev_on = in_plane & (below >= 0)
        next_on = in_plane & (above <= m - 1)
        ip = np.where(prev_on, below, 0)
        rel = at - cens[ip]
        d_prev = (rel * nrms[ip]).sum(axis=1)
        jn = np.where(next_on, above, 0)
        jp = np.where(next_on, pos, 0)
        rel2 = at - cens[jn]
        d_next = (rel2 * nrms[jn]).sum(axis=1)
        hit = (prev_on & ~skip[np.where(prev_on, below, 0)] & (d_prev < -tol)) | (
            next_on & ~skip[jp] & (d_next > tol))
        bad[ids[hit]] = True
    return np.any(bad[panels], axis=1)


def _check_mesh_reference(mesh: PanelMesh, warp_limit: float = WARP_LIMIT) -> dict:
    """Loop reference of :func:`check_mesh`, kept only for the parity tests.

    The per-panel geometry runs :func:`_panel_geometry` in a Python loop and
    the edge bookkeeping uses Python dicts. The solver never calls this
    function. The vectorized :func:`check_mesh` must return the same
    booleans and counts, and the float metrics equal to round-off.

    Parameters
    ----------
    mesh : PanelMesh
        Mesh to check.
    warp_limit : float
        Planarity limit [-]: largest allowed distance of a corner from the
        mean plane of its panel over the panel diameter.

    Returns
    -------
    dict
        watertight : every edge is shared by exactly two panels, except the
        root edges of a half mesh (both nodes on y = 0 within a relative
        tolerance) and the edges of the end stations when ``mesh.tip`` is
        "none" (needs ``mesh.node_station``).
        orientation_consistent : the two panels of each shared edge run
        along it in opposite directions.
        outward : orientation consistent and the signed volume of each
        component above a relative tolerance.
        volume [m^3] (sum over components) and component_volume [m^3] per
        component.
        min_area [m^2]; per-panel aspect_ratio, skew, warp and convex
        arrays with max_aspect_ratio, max_skew and max_warp.
        folded : per-panel flag of a folded strip (needs
        ``mesh.node_station``): a corner node lies on the wrong side of the
        section plane of the neighbour station row, so the surface passes
        through itself (see :func:`_station_folds`). A fold need not make a
        panel non-convex. n_folded : number of folded panels.
        n_nonconvex : number of panels that are not convex in their mean
        plane or are folded (both make a surface that is not valid; the
        trust score reads this count as folded panels).
        planar : max_warp below *warp_limit*.
        valid_panels : no panel non-convex or folded, all planar and with
        positive area.
        n_open_edges : number of edges used by one panel only.
    """
    nodes = np.asarray(mesh.nodes, dtype=float)
    panels = np.asarray(mesh.panels, dtype=int)
    n_p = panels.shape[0]
    areas = np.empty(n_p)
    aspects = np.empty(n_p)
    skews = np.empty(n_p)
    warps = np.empty(n_p)
    convex = np.empty(n_p, dtype=bool)
    comp_raw = np.asarray(mesh.component).reshape(-1)
    comp_of = comp_raw.astype(int) if comp_raw.shape[0] == n_p else np.zeros(n_p, dtype=int)
    comp_ids = sorted({int(c) for c in comp_of})
    comp_volume = {c: 0.0 for c in comp_ids}
    for i in range(n_p):
        p = nodes[panels[i]]
        areas[i], _, aspects[i], skews[i], warps[i], convex[i] = _panel_geometry(p)
        p1, p2, p3, p4 = (np.asarray(q, dtype=float).reshape(3) for q in p)
        comp_volume[int(comp_of[i])] += float(np.dot(p1, np.cross(p2, p3)) + np.dot(p1, np.cross(p3, p4))) / 6.0
    volume = float(sum(comp_volume.values()))

    counts: dict[tuple[int, int], int] = {}
    direction: dict[tuple[int, int], int] = {}
    owners: dict[tuple[int, int], int] = {}
    for i in range(n_p):
        row = panels[i]
        for a, b in ((row[0], row[1]), (row[1], row[2]), (row[2], row[3]), (row[3], row[0])):
            a, b = int(a), int(b)
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            counts[key] = counts.get(key, 0) + 1
            direction[key] = direction.get(key, 0) + (1 if a < b else -1)
            owners.setdefault(key, i)

    comp_nodes: dict[int, np.ndarray] = {}
    for c in comp_ids:
        comp_nodes[c] = np.unique(panels[comp_of == c].ravel())
    station = None if mesh.node_station is None else np.asarray(mesh.node_station, dtype=int)
    ends: dict[int, tuple[int, int]] = {}
    if station is not None and station.shape[0] == nodes.shape[0]:
        for c in comp_ids:
            st = station[comp_nodes[c]]
            ends[c] = (int(np.min(st)), int(np.max(st)))
    else:
        station = None
    size = float(np.max(np.ptp(nodes, axis=0))) if nodes.shape[0] else 1.0
    y_tol = _Y_ROOT_REL * max(size, 1e-300)
    tip_open = str(getattr(mesh, "tip", "flat")) == "none"

    watertight = True
    consistent = True
    n_open = 0
    for key, cnt in counts.items():
        if cnt == 2:
            if direction[key] != 0:
                consistent = False
            continue
        if cnt > 2:
            watertight = False
            continue
        n_open += 1
        a, b = key
        root_ok = bool(mesh.half) and abs(float(nodes[a, 1])) <= y_tol and abs(float(nodes[b, 1])) <= y_tol
        tip_ok = False
        if tip_open and station is not None:
            c = int(comp_of[owners[key]])
            lo, hi = ends[c]
            sa, sb = int(station[a]), int(station[b])
            tip_ok = (sa == sb) and (sa in (lo, hi))
        if not (root_ok or tip_ok):
            watertight = False

    outward = consistent
    for c in comp_ids:
        ext = nodes[comp_nodes[c]]
        size_c = float(np.max(np.ptp(ext, axis=0))) if ext.shape[0] else 0.0
        if not comp_volume[c] > _VOLUME_REL * size_c**3:
            outward = False
    max_warp = float(np.max(warps)) if n_p else 0.0
    min_area = float(np.min(areas)) if n_p else 0.0
    if station is not None and n_p:
        folded = _station_folds(nodes, panels, station, comp_of)
    else:
        folded = np.zeros(n_p, dtype=bool)
    n_folded = int(np.sum(folded))
    n_nonconvex = int(np.sum(~convex | folded))
    planar = bool(max_warp <= float(warp_limit))
    return {
        "watertight": bool(watertight),
        "orientation_consistent": bool(consistent),
        "outward": bool(outward),
        "volume": volume,
        "component_volume": {int(c): float(v) for c, v in comp_volume.items()},
        "min_area": min_area,
        "aspect_ratio": aspects,
        "skew": skews,
        "warp": warps,
        "convex": convex,
        "max_aspect_ratio": float(np.max(aspects)) if n_p else 1.0,
        "max_skew": float(np.max(skews)) if n_p else 0.0,
        "max_warp": max_warp,
        "folded": folded,
        "n_folded": n_folded,
        "n_nonconvex": n_nonconvex,
        "planar": planar,
        "valid_panels": bool(n_nonconvex == 0 and planar and min_area > 0.0),
        "n_open_edges": int(n_open),
    }


def check_mesh(mesh: PanelMesh, warp_limit: float = WARP_LIMIT) -> dict:
    """Check closure, orientation and quality of a panel mesh.

    The check runs with array operations: the per-panel geometry of
    :func:`_panel_geometry_batch` for all panels at once, the edge
    bookkeeping on the flat directed edge list with ``numpy.unique`` and
    ``numpy.bincount``, and the fold detection of
    :func:`_station_folds_batch`. The decisions match the loop reference
    :func:`_check_mesh_reference` (kept for the parity tests): every
    boolean and count is identical, and the float metrics are equal to
    round-off.

    Parameters
    ----------
    mesh : PanelMesh
        Mesh to check.
    warp_limit : float
        Planarity limit [-]: largest allowed distance of a corner from the
        mean plane of its panel over the panel diameter.

    Returns
    -------
    dict
        watertight : every edge is shared by exactly two panels, except the
        root edges of a half mesh (both nodes on y = 0 within a relative
        tolerance) and the edges of the end stations when ``mesh.tip`` is
        "none" (needs ``mesh.node_station``).
        orientation_consistent : the two panels of each shared edge run
        along it in opposite directions.
        outward : orientation consistent and the signed volume of each
        component above a relative tolerance.
        volume [m^3] (sum over components) and component_volume [m^3] per
        component.
        min_area [m^2]; per-panel aspect_ratio, skew, warp and convex
        arrays with max_aspect_ratio, max_skew and max_warp.
        folded : per-panel flag of a folded strip (needs
        ``mesh.node_station``): a corner node lies on the wrong side of the
        section plane of the neighbour station row, so the surface passes
        through itself (see :func:`_station_folds_batch`). A fold need not make a
        panel non-convex. n_folded : number of folded panels.
        n_nonconvex : number of panels that are not convex in their mean
        plane or are folded (both make a surface that is not valid; the
        trust score reads this count as folded panels).
        planar : max_warp below *warp_limit*.
        valid_panels : no panel non-convex or folded, all planar and with
        positive area.
        n_open_edges : number of edges used by one panel only.
    """
    nodes = np.asarray(mesh.nodes, dtype=float)
    panels = np.asarray(mesh.panels, dtype=int).reshape(-1, 4)
    n_p = panels.shape[0]
    corners = nodes[panels]
    areas, _, aspects, skews, warps, convex = _panel_geometry_batch(corners)
    comp_raw = np.asarray(mesh.component).reshape(-1)
    comp_of = comp_raw.astype(int) if comp_raw.shape[0] == n_p else np.zeros(n_p, dtype=int)
    comp_ids = sorted({int(c) for c in comp_of})
    comp_index = np.asarray(comp_ids, dtype=int)
    # Signed volume of each panel: the two tetrahedra of the corner fan.
    contrib = (
        np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2]), axis=1)
        + np.sum(corners[:, 0] * np.cross(corners[:, 2], corners[:, 3]), axis=1)
    ) / 6.0
    comp_volume = {
        int(c): float(v) for c, v in zip(comp_ids, np.bincount(np.searchsorted(comp_index, comp_of),
                                                               weights=contrib, minlength=len(comp_ids)))
    }
    volume = float(sum(comp_volume.values()))

    # The 4 Np directed edges; the undirected keys give the count, the
    # direction sum and the first owner of each edge.
    ea = panels.ravel()
    eb = panels[:, [1, 2, 3, 0]].ravel()
    used = ea != eb
    ea, eb = ea[used], eb[used]
    lo = np.minimum(ea, eb)
    hi = np.maximum(ea, eb)
    owner_of = np.repeat(np.arange(n_p, dtype=int), 4)[used]
    keys = lo.astype(np.int64) * max(int(nodes.shape[0]), 1) + hi.astype(np.int64)
    _, first, inv, counts = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)
    dir_sum = np.bincount(inv, weights=np.where(ea < eb, 1.0, -1.0))

    comp_nodes = {c: np.unique(panels[comp_of == c].ravel()) for c in comp_ids}
    station = None if mesh.node_station is None else np.asarray(mesh.node_station, dtype=int)
    ends: dict[int, tuple[int, int]] = {}
    if station is not None and station.shape[0] == nodes.shape[0]:
        for c in comp_ids:
            st = station[comp_nodes[c]]
            ends[c] = (int(np.min(st)), int(np.max(st)))
    else:
        station = None
    size = float(np.max(np.ptp(nodes, axis=0))) if nodes.shape[0] else 1.0
    y_tol = _Y_ROOT_REL * max(size, 1e-300)
    tip_open = str(getattr(mesh, "tip", "flat")) == "none"

    # The root rule of a half mesh and the open-tip rule work on the edge arrays.
    first_edge = first[counts == 1]
    a_open = lo[first_edge]
    b_open = hi[first_edge]
    root_ok = bool(mesh.half) & (np.abs(nodes[a_open, 1]) <= y_tol) & (np.abs(nodes[b_open, 1]) <= y_tol)
    tip_ok = np.zeros(first_edge.size, dtype=bool)
    if tip_open and station is not None:
        ends_lo = np.asarray([ends[c][0] for c in comp_ids])
        ends_hi = np.asarray([ends[c][1] for c in comp_ids])
        slot = np.searchsorted(comp_index, comp_of[owner_of[first_edge]])
        sa = station[a_open]
        sb = station[b_open]
        tip_ok = (sa == sb) & ((sa == ends_lo[slot]) | (sa == ends_hi[slot]))
    n_open = int(first_edge.size)
    watertight = not (bool(np.any(counts > 2)) or bool(np.any(~(root_ok | tip_ok))))
    consistent = not bool(np.any(dir_sum[counts == 2] != 0.0))

    outward = consistent
    for c in comp_ids:
        ext = nodes[comp_nodes[c]]
        size_c = float(np.max(np.ptp(ext, axis=0))) if ext.shape[0] else 0.0
        if not comp_volume[c] > _VOLUME_REL * size_c**3:
            outward = False
    max_warp = float(np.max(warps)) if n_p else 0.0
    min_area = float(np.min(areas)) if n_p else 0.0
    if station is not None and n_p:
        folded = _station_folds_batch(nodes, panels, station, comp_of)
    else:
        folded = np.zeros(n_p, dtype=bool)
    n_folded = int(np.sum(folded))
    n_nonconvex = int(np.sum(~convex | folded))
    planar = bool(max_warp <= float(warp_limit))
    return {
        "watertight": bool(watertight),
        "orientation_consistent": bool(consistent),
        "outward": bool(outward),
        "volume": volume,
        "component_volume": {int(c): float(v) for c, v in comp_volume.items()},
        "min_area": min_area,
        "aspect_ratio": aspects,
        "skew": skews,
        "warp": warps,
        "convex": convex,
        "max_aspect_ratio": float(np.max(aspects)) if n_p else 1.0,
        "max_skew": float(np.max(skews)) if n_p else 0.0,
        "max_warp": max_warp,
        "folded": folded,
        "n_folded": n_folded,
        "n_nonconvex": n_nonconvex,
        "planar": planar,
        "valid_panels": bool(n_nonconvex == 0 and planar and min_area > 0.0),
        "n_open_edges": int(n_open),
    }
