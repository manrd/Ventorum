# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Vortex-lattice geometry: from :class:`Aircraft` to horseshoe panels.

Geometry conventions
--------------------
* Axes: x aft (downstream), y to the right wing, z up.
* ``semi_span`` is the length of the surface measured in the y-z plane
  (along the dihedral line). With a dihedral angle G, a station at span
  fraction eta is at ``y = eta*b*cos(G)`` and ``z = eta*b*sin(G)``. So a
  surface with ``dihedral = 90 deg`` is a vertical fin of height ``b``.
* If any section sets ``z_le`` explicitly, ``semi_span`` is the projected
  span (y extent): ``y = eta*b`` and ``z`` comes from the sections. Sections
  without ``z_le`` then get ``z = eta*b*tan(G)``.
* The leading-edge sweep is measured in the plane of the surface:
  ``x_le = eta*b*tan(sweep_le)`` unless a section sets ``x_le``.
* Twist plus surface incidence rotates each section about its quarter-chord
  point (nose up is positive). The quarter-chord line does not move.

Lattice
-------
Each surface is cut into spanwise strips. Each strip has ``n_chord``
chordwise panels. Each panel carries one bent horseshoe (see
:mod:`ventorum.aero.vortex`): the bound vortex is on the 1/4-chord line of
the panel, the legs run along the strip edges to the trailing edge, and then
into the wake.

Two collocation rules are available:

* ``"vlm"``: the control point is at ``1/4 + CLAF/2`` of the panel length,
  with ``CLAF = a0 / (2 pi)``. For a flat plate (``CLAF = 1``) this is the
  classical 3/4-chord point. Moving the point gives the section lift slope
  ``a0`` in two-dimensional flow.
* ``"llt"``: one panel per strip and the control point on the bound vortex
  (Phillips & Snyder 2000).

In the spanwise direction the control point is at the mid parameter of the
spacing function (the "theta" midpoint for cosine spacing), not at the mean
of the strip edges. See :mod:`ventorum.geometry.discretization`.

Known section mean lines carry the vortex lattice and its local normals.
Sections without a shape keep the zero-lift line: the boundary-condition
normal is the geometric normal turned nose-up by ``-alpha_L0``.

Sources: J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., 2001
(camber-surface lattice and lumped vortices); C. Coulliette and A. Plotkin,
The Aeronautical Journal 100, 1996 (real camber line in ground effect);
S. E. Widnall and T. M. Barrows, J. Fluid Mech. 41, 1970 (limits of the
mean-plane boundary condition near the ground).
"""

from __future__ import annotations

import bisect
import warnings
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ventorum.core.datatypes import (
    Aircraft,
    AirfoilType,
    ControlSurface,
    LiftingSurface,
    LinearAirfoil,
    SolverSettings,
    TabulatedAirfoil,
    WingSection,
)
from ventorum.geometry.controls import (
    ControlJoinField,
    deflected_airfoil,
    flap_effectiveness,  # noqa: F401 (re-exported for lattice_cache builder helpers)
    flap_moment_derivative,  # noqa: F401 (re-exported for lattice_cache builder helpers)
    validate_controls,
)
from ventorum.geometry.discretization import compute_surface_n_panels, get_spacing
from ventorum.utils.vec import cross3

#: Limits of the control-point factor CLAF = a0 / (2 pi) of the vortex lattice
#: [-]. They keep the control point away from the bound vortex (lower limit)
#: and within about one panel length (upper limit). They are a code setting
#: with no published source; a warning tells the user when they act.
CLAF_MIN = 0.2
CLAF_MAX = 2.0

#: Window of a moved hinge node [-]. In chordwise control mode, the lattice
#: node nearest to the hinge moves onto the hinge only when the two panels
#: next to it then have between HINGE_PANEL_MIN_FACTOR and
#: HINGE_PANEL_MAX_FACTOR times their initial lengths. Otherwise the hinge
#: snaps to the nearest node and a warning gives the effective hinge. These
#: factors are a code setting with no published source or study.
HINGE_PANEL_MIN_FACTOR = 0.5
HINGE_PANEL_MAX_FACTOR = 1.25

# Core radius of a horseshoe as a fraction of the smaller panel dimension
# (strip width or chordwise panel length). The nearest control point is half
# a panel dimension away, so the velocity there changes by about 4e-4 of its
# value. The core caps the velocity when a point of one surface comes close
# to a vortex of another surface. (A core that scales with the strip width
# only is too large for short chordwise panels on wide strips: it weakens
# the self-influence and over-predicts lift.)
CORE_RADIUS_FRACTION = 0.01
# Core added when a vortex of one surface acts on a point of another surface
# (a fraction of the target strip width). A wing trailing vortex that passes
# through a tail then gives a result that does not depend on where it falls
# between two tail control points. Surfaces that share an edge (two halves of
# a V-tail, a split wing, a wing and its winglet) are one vortex sheet: the
# added core does not apply between them (see ``core_groups``).
CROSS_SURFACE_CORE_FRACTION = 0.5  # owner decision of 2026-10-03, from the core-size study with a wing and a tail


# ═══════════════════════════════════════════════════════════════════════════════
# Section geometry along one surface
# ═══════════════════════════════════════════════════════════════════════════════

def _sorted_sections(surf: LiftingSurface) -> list[WingSection]:
    return sorted(surf.sections, key=lambda s: s.y_frac)


def _rotate(v: np.ndarray, axis: np.ndarray, angle: np.ndarray | float) -> np.ndarray:
    """Rodrigues rotation of vectors *v* (n, 3) about unit *axis* (n, 3)."""
    angle = np.broadcast_to(np.asarray(angle, dtype=float), (v.shape[0],))
    c = np.cos(angle)[:, None]
    s = np.sin(angle)[:, None]
    k_dot_v = np.sum(axis * v, axis=1)[:, None]
    return v * c + cross3(axis, v) * s + axis * k_dot_v * (1.0 - c)


def _rotate_flap_points(points: np.ndarray, hinge: np.ndarray, axis: np.ndarray, delta: float) -> np.ndarray:
    """Rotate flap points [m] about the unit hinge axis by *delta* [rad].

    Use the rigid camber-surface rotation of Katz and Plotkin (2001).
    The caller selects points strictly aft of the effective hinge.

    Parameters
    ----------
    points : numpy.ndarray
        Points in body axes [m], shape (n, 3).
    hinge : numpy.ndarray
        Point on the hinge axis [m], shape (3,).
    axis : numpy.ndarray
        Unit hinge direction [-], shape (3,).
    delta : float
        Signed deflection [rad].

    Returns
    -------
    numpy.ndarray
        Rotated points in body axes [m], shape (n, 3).
    """
    return hinge + _rotate(points - hinge, axis[None, :], delta)


def surface_reference_line(surf: LiftingSurface, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leading-edge line of the flat (untwisted) right half, without the position offset.

    Returns
    -------
    x_le, y, z : arrays with the shape of *eta*.
    """
    secs = _sorted_sections(surf)
    fr = np.array([s.y_frac for s in secs])
    b = surf.semi_span
    eta = np.asarray(eta, dtype=float)

    if any(s.x_le is not None for s in secs):
        xv = np.array([s.x_le if s.x_le is not None else s.y_frac * b * np.tan(surf.sweep_le) for s in secs])
        x_le = np.interp(eta, fr, xv)
    else:
        x_le = eta * b * np.tan(surf.sweep_le)

    if any(s.z_le is not None for s in secs):
        # Projected-span convention: semi_span is the y extent.
        zv = np.array([s.z_le if s.z_le is not None else s.y_frac * b * np.tan(surf.dihedral) for s in secs])
        y = eta * b
        z = np.interp(eta, fr, zv)
    else:
        # True-length convention: semi_span is measured along the dihedral line.
        y = eta * b * np.cos(surf.dihedral)
        z = eta * b * np.sin(surf.dihedral)
    return x_le, y, z


def surface_edge_geometry(surf: LiftingSurface, eta: np.ndarray) -> dict[str, np.ndarray]:
    """Leading edge, trailing edge and properties at span fractions *eta* (right half).

    The position offset of the surface is applied. Twist and incidence rotate
    each section about its quarter-chord point.
    """
    secs = _sorted_sections(surf)
    fr = np.array([s.y_frac for s in secs])
    eta = np.asarray(eta, dtype=float)
    chord = np.interp(eta, fr, np.array([s.chord for s in secs]))
    twist = np.interp(eta, fr, np.array([s.twist for s in secs])) + surf.incidence

    x_le, y, z = surface_reference_line(surf, eta)

    # Local spanwise axis in the y-z plane (root to tip), used for the twist rotation.
    if len(eta) >= 2:
        dy = np.gradient(y, eta)
        dz = np.gradient(z, eta)
    else:
        _, y1, z1 = surface_reference_line(surf, np.array([0.0, 1.0]))
        dy = np.full_like(eta, y1[1] - y1[0])
        dz = np.full_like(eta, z1[1] - z1[0])
    span_axis = np.column_stack([np.zeros_like(eta), dy, dz])
    span_axis /= np.maximum(np.linalg.norm(span_axis, axis=1, keepdims=True), 1e-300)
    if surf.is_symmetric:
        # The root section is shared by both halves: twist it about the y-axis
        # (the bisector of the two span directions), so it stays in y = 0.
        span_axis[eta <= 1e-12] = np.array([0.0, 1.0, 0.0])

    x_hat = np.tile(np.array([1.0, 0.0, 0.0]), (len(eta), 1))
    chord_dir = _rotate(x_hat, span_axis, twist)

    offset = np.asarray(surf.position, dtype=float)
    le_flat = np.column_stack([x_le, y, z]) + offset
    qc = le_flat + 0.25 * chord[:, None] * x_hat
    le = qc - 0.25 * chord[:, None] * chord_dir
    te = qc + 0.75 * chord[:, None] * chord_dir
    return {"le": le, "te": te, "qc": qc, "chord": chord, "twist": twist, "chord_dir": chord_dir}


# ═══════════════════════════════════════════════════════════════════════════════
# Section aerodynamic properties along the span
# ═══════════════════════════════════════════════════════════════════════════════

class _AirfoilBlender:
    """Interpolate section airfoils along the span (with a cache)."""

    def __init__(self, sections: list[WingSection]):
        self.secs = sections
        first_af = sections[0].airfoil if sections else None
        if sections and all(s.airfoil is first_af for s in sections):
            self._single_airfoil = first_af
        else:
            self._single_airfoil = None
        self.fr = [float(s.y_frac) for s in sections]
        self._cache: dict[tuple[int, int, float], AirfoilType] = {}

    def at(self, eta: float) -> AirfoilType:
        if self._single_airfoil is not None:
            return self._single_airfoil
        fr = self.fr
        if eta <= fr[0]:
            return self.secs[0].airfoil
        if eta >= fr[-1]:
            return self.secs[-1].airfoil
        j = bisect.bisect_right(fr, eta) - 1
        j = min(max(j, 0), len(fr) - 2)
        af1, af2 = self.secs[j].airfoil, self.secs[j + 1].airfoil
        span = fr[j + 1] - fr[j]
        w = 0.0 if span <= 0 else float((eta - fr[j]) / span)
        if af1 is af2 or w <= 1e-12:
            return af1
        if w >= 1.0 - 1e-12:
            return af2
        if isinstance(af1, LinearAirfoil) and isinstance(af2, LinearAirfoil):
            return LinearAirfoil(
                name=f"{af1.name}|{af2.name}",
                a0=(1 - w) * af1.a0 + w * af2.a0,
                alpha_L0=(1 - w) * af1.alpha_L0 + w * af2.alpha_L0,
                Cd0=(1 - w) * af1.Cd0 + w * af2.Cd0,
                Cm0=(1 - w) * af1.Cm0 + w * af2.Cm0,
            )
        if isinstance(af1, TabulatedAirfoil) and isinstance(af2, TabulatedAirfoil):
            key = (id(af1), id(af2), round(w, 6))
            if key not in self._cache:
                from ventorum.aero.polars import blend_tabulated
                self._cache[key] = blend_tabulated(af1, af2, w)
            return self._cache[key]
        # Mixed airfoil types: use the nearer section.
        return af1 if w < 0.5 else af2


def airfoil_linear_properties(af: AirfoilType) -> tuple[float, float, float, float]:
    """Return ``(a0, alpha_L0, Cd0, Cm0)`` of an airfoil (linear fit for tabulated polars)."""
    if isinstance(af, LinearAirfoil):
        return float(af.a0), float(af.alpha_L0), float(af.Cd0), float(af.Cm0)
    a0 = float(af.a0)
    aL0 = float(af.alpha_L0)
    alpha, _, cd, cm = af.tables()  # sorted: np.interp needs ascending angles
    cd0 = float(np.interp(aL0, alpha, cd))
    cm0 = float(np.interp(aL0, alpha, cm)) if cm is not None else 0.0
    return a0, aL0, cd0, cm0


def section_cl(af: AirfoilType, alpha: np.ndarray) -> np.ndarray:
    """Return the section lift coefficient at *alpha* [rad] as an array."""
    return np.asarray(af.Cl(alpha), dtype=float)


def section_cd(af: AirfoilType, alpha: np.ndarray) -> np.ndarray:
    """Return the section drag coefficient at *alpha* [rad] as an array."""
    return np.asarray(af.Cd(alpha), dtype=float)


def section_cm(af: AirfoilType, alpha: np.ndarray) -> np.ndarray:
    """Return the section pitching-moment coefficient at *alpha* [rad] as an array.

    A tabulated airfoil without Cm data gives zero.
    """
    if isinstance(af, TabulatedAirfoil) and af.Cm_data is None:
        return np.zeros_like(np.asarray(alpha, dtype=float))
    return np.asarray(af.Cm(alpha), dtype=float)


def section_cl_slope(af: AirfoilType, alpha: np.ndarray) -> np.ndarray:
    """dCl/dalpha of the section at *alpha* [1/rad]."""
    if isinstance(af, LinearAirfoil):
        return np.full_like(np.asarray(alpha, dtype=float), af.a0)
    return np.asarray(af.Cl_slope(np.asarray(alpha, dtype=float)), dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# Lattice container
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SurfaceSlice:
    """Where one surface sits inside the aircraft lattice."""

    name: str
    index: int
    is_symmetric: bool
    strips: slice
    edge_le: np.ndarray      # (n_edges, 3) leading-edge nodes, left to right
    edge_te: np.ndarray      # (n_edges, 3) trailing-edge nodes
    edge_chord: np.ndarray   # (n_edges,)
    edge_twist: np.ndarray   # (n_edges,)
    # Unit spanwise direction of each strip from the geometry definition (the
    # untwisted reference line, without node displacements), projected on the
    # plane normal to the untwisted chord: (n_strips, 3). The leg-force rule
    # takes the kinks of the surface from it (ventorum.aero.loads.leg_force_rules).
    strip_span_def: np.ndarray | None = None
    # True for each edge that is joined to an edge of another surface: (n_edges,).
    edge_joined: np.ndarray | None = None
    # Chord-line leading-edge nodes before camber offset [m]: (n_edges, 3).
    edge_le_chord: np.ndarray | None = None
    # Chord-line trailing-edge nodes before camber offset [m]: (n_edges, 3).
    edge_te_chord: np.ndarray | None = None


@dataclass
class VortexLattice:
    """Panels and strips of an aircraft, in body axes.

    All lengths are in metres and all angles in radians.

    Attributes
    ----------
    a_kink, b_kink : numpy.ndarray or None
        First nodes of the chordwise legs after the bound points [m], shape
        (n_panels, 3), or None when no leg has a node. Without a chain
        (``leg_next`` False) the left leg of a panel runs ``a_te -> a_kink
        -> a`` and the right leg ``b -> b_kink -> b_te``. A panel ahead of
        the hinge of a chordwise flap has the hinge point of its strip edge
        here, so its legs follow the deflected mean surface. On all other
        panels ``a_kink`` is ``a_te`` and ``b_kink`` is ``b_te`` (the legs
        are straight).
    leg_next : numpy.ndarray or None
        Node chains of the chordwise legs, shape (n_panels,) bool, or None.
        On a strip with geometric camber the legs follow the mean surface:
        the legs of a panel run from its bound points through the bound
        points of the panels aft of it on the same strip edge (the nodes of
        the edge) to the trailing edge. ``leg_next`` is True on each panel
        of such a strip except the last; its legs end at the bound points of
        the next panel and continue on its legs (see
        :class:`ventorum.aero.vortex.HorseshoeSet`). ``a_kink`` is then the
        bound point of the next panel, or the hinge point of a chordwise
        flap where the hinge lies between the two bound points. Strips
        without a mean line keep their legs.
    chord : numpy.ndarray
        Strip chord [m].
    twist : numpy.ndarray
        Strip twist, with the incidence of the surface [rad].
        Keeps the undeformed (jig) twist; the deformed shape is in the nodes.
    width : numpy.ndarray
        Strip width [m].
    area : numpy.ndarray
        Strip area [m^2].

    Panels are ordered strip by strip; inside a strip from leading edge to
    trailing edge. Strips are ordered surface by surface; on a symmetric
    surface from the left tip to the right tip.
    """

    collocation: str
    n_chord: int
    # --- panels -----------------------------------------------------------
    a: np.ndarray
    b: np.ndarray
    a_te: np.ndarray
    b_te: np.ndarray
    cp: np.ndarray
    normal_bc: np.ndarray
    panel_strip: np.ndarray
    panel_mirror: np.ndarray
    panel_length: np.ndarray
    # --- strips -----------------------------------------------------------
    strip_surface: np.ndarray
    le_left: np.ndarray
    le_right: np.ndarray
    te_left: np.ndarray
    te_right: np.ndarray
    qc_mid: np.ndarray
    dl: np.ndarray
    chord: np.ndarray
    twist: np.ndarray
    width: np.ndarray
    area: np.ndarray
    chord_dir: np.ndarray
    normal: np.ndarray
    normal_bc_strip: np.ndarray
    a0: np.ndarray
    alpha_L0: np.ndarray
    Cd0: np.ndarray
    Cm0: np.ndarray
    airfoils: list[AirfoilType]
    strip_mirror: np.ndarray
    strip_is_right: np.ndarray
    eta: np.ndarray
    cp_frac: np.ndarray
    surfaces: list[SurfaceSlice] = field(default_factory=list)
    # Core group of each strip: surfaces that share an edge have the same group.
    strip_core_group: np.ndarray | None = None
    # Warnings about surface edges that are near each other (see ``near_miss_warnings``).
    join_warnings: list[str] = field(default_factory=list)
    # Cache of the parts of the vortex system that do not change with the
    # angle of attack (see ventorum.aero.vortex). None means no cache. A
    # sweep sets it to a dict; the solvers use it only out of ground effect.
    kernel_cache: dict | None = field(default=None, repr=False, compare=False)
    # Results that depend only on the lattice (airfoil groups, symmetry
    # geometry, Trefftz strip data). Shared by the shallow copies that the
    # lattice cache hands out, so they are computed once per geometry.
    geom_cache: dict = field(default_factory=dict, repr=False, compare=False)
    control_info: list[dict] = field(default_factory=list)
    # Warnings that build_lattice issues about the mesh (a hinge that snaps
    # to a node, a hinge that moves a node of a cambered strip). The lattice
    # cache issues them again when it returns a cached lattice, and the
    # solvers give them to the trust score as notes.
    mesh_warnings: list[str] = field(default_factory=list)
    geometric_camber: np.ndarray | None = None
    # Hinge nodes of the chordwise legs (see ``a_kink`` in the class notes).
    # None when no leg has a hinge node.
    a_kink: np.ndarray | None = None
    b_kink: np.ndarray | None = None
    # Node chains of the legs on strips with geometric camber (see
    # ``leg_next`` in the class notes). None when no strip has one.
    leg_next: np.ndarray | None = None
    # Boundary-condition normals before the turn of the chordwise flaps.
    # None when no chordwise control turns a panel.
    normal_bc_clean: np.ndarray | None = None
    camber_fallback: bool = False
    has_camber_offset: bool = False
    camber_frames: list[dict] | None = None

    @property
    def n_panels(self) -> int:
        """Return the number of panels."""
        return int(self.a.shape[0])

    @property
    def n_strips(self) -> int:
        """Return the number of spanwise strips."""
        return int(self.chord.shape[0])

    @property
    def rc(self) -> np.ndarray:
        """Core radius of every panel horseshoe [m]."""
        return CORE_RADIUS_FRACTION * np.minimum(self.width[self.panel_strip], self.panel_length)

    @property
    def force_points(self) -> np.ndarray:
        """Points on every bound vortex where the induced velocity is taken (same span fraction as the control point)."""
        f = self.cp_frac[self.panel_strip][:, None]
        return self.a + f * (self.b - self.a)

    @property
    def has_tabulated(self) -> bool:
        """Return True if one or more airfoils are tabulated."""
        return any(isinstance(af, TabulatedAirfoil) for af in self.airfoils)

    def all_points(self) -> np.ndarray:
        """Return edge nodes and non-flat mean-surface points for clearance checks [m]."""
        deflected = any(
            info["mode"] == "chordwise" and info["deflection_rad"] != 0.0 for info in self.control_info
        )
        pts = [s.edge_le for s in self.surfaces]
        if deflected:
            pts.extend([self.te_left, self.te_right])
            pts.extend(p[None, :] for pair in self.geom_cache.get("flap_hinges", {}).values() for p in pair)
        else:
            pts.extend(s.edge_te for s in self.surfaces)
        if self.has_camber_offset or deflected:
            pts.extend([self.a, self.b, self.cp, self.a_te, self.b_te])
        return np.vstack(pts)

    def can_fold_symmetry(self) -> bool:
        """Return True if every panel has a mirror panel (all surfaces symmetric)."""
        return bool(np.all(self.panel_mirror >= 0))


# Join rule for surfaces that share an edge (see ``core_groups``).
# The tolerance is the smaller local strip width, within two limits that are
# fractions of the shorter edge segment. A cross-surface core between two
# surfaces with a gap of a fraction g of the strip width lowers the lift of
# a split wing by about 6.5 %, 2.7 % and 0.6 % at g = 0.1, 0.4 and 1 (40
# strips; see the "Split surfaces" case of the verification report). A
# tolerance of 0.25 strip width, as first planned, is too small. The upper
# limit stops a coarse mesh from joining two real surfaces (a biplane).
JOIN_TOL_STRIP_FRACTION = 1.0
JOIN_TOL_CHORD_FRACTION = 1e-6      # lower limit of the tolerance, as a fraction of the chord
JOIN_TOL_CHORD_MAX_FRACTION = 0.1   # upper limit of the tolerance, as a fraction of the chord
JOIN_MIN_OVERLAP = 0.5              # least overlap, as a fraction of the shorter edge segment
NEAR_MISS_MAX_ANGLE_DEG = 10.0      # edges closer to parallel than this can be a near miss
NEAR_MISS_MAX_GAP_CHORDS = 2.0      # largest gap that is a near miss, in local chords


def _edge_strip_widths(surf: SurfaceSlice) -> np.ndarray:
    """Return the width of the narrowest strip at each edge of *surf* [m].

    The width of a strip is the distance between the quarter-chord points of
    its two edges.
    """
    qc = surf.edge_le + 0.25 * (surf.edge_te - surf.edge_le)
    seg = np.linalg.norm(np.diff(qc, axis=0), axis=1)
    out = np.empty(qc.shape[0])
    out[0], out[-1] = seg[0], seg[-1]
    if seg.size > 1:
        out[1:-1] = np.minimum(seg[:-1], seg[1:])
    return out


def _perpendicular_distance(p: np.ndarray, origin: np.ndarray, unit: np.ndarray) -> np.ndarray:
    """Distance of the points *p* (n, 3) from the line through *origin* with direction *unit* [m]."""
    d = p - origin
    return np.linalg.norm(d - np.outer(d @ unit, unit), axis=1)


def _interior_edge(surf: SurfaceSlice, k: int) -> bool:
    """Return True when edge *k* of *surf* is an interior strip edge.

    The two end edges are not interior. The root edge of a symmetric surface
    (its plane of symmetry) is not interior either: it is at the same place
    for every mesh.
    """
    n = int(np.asarray(surf.edge_le).shape[0])
    if k == 0 or k == n - 1:
        return False
    return not (getattr(surf, "is_symmetric", False) and n % 2 == 1 and k == n // 2)


def _relate_surfaces(si: SurfaceSlice, sj: SurfaceSlice) -> tuple[bool, tuple[float, float] | None, dict]:
    """Compare the edges of two surfaces.

    Return ``(joined, near_miss, info)``. ``joined`` is True when an edge
    segment (the line from the leading-edge node to the trailing-edge node
    of one section) of *si* lies on an edge segment of *sj* within the
    tolerance, over at least ``JOIN_MIN_OVERLAP`` of the shorter segment,
    and the chordwise grids of the two edges match: the leading-edge and
    trailing-edge nodes of one edge are, along the edge line, within the
    tolerance of the end nodes of the other edge (in the same or the opposite
    order). The tolerance is ``JOIN_TOL_STRIP_FRACTION`` times the smaller
    local strip width, kept between ``JOIN_TOL_CHORD_FRACTION`` times the
    longer segment and ``JOIN_TOL_CHORD_MAX_FRACTION`` times the shorter
    segment. ``near_miss`` is None, or ``(gap, local_chord)`` [m] of the
    closest pair of edges that are nearly parallel and overlap, but are not
    within the tolerance. ``info`` has two keys: ``"grid_mismatch"`` is True
    when two edges lie on each other within the tolerance but their
    chordwise grids do not match (they are not joined); ``"interior"`` lists
    ``"i"`` and/or ``"j"`` when the surfaces are joined only through an
    interior strip edge of *si* or *sj* (no pair of boundary edges is
    joined; see :func:`_interior_edge`); ``"edges"`` lists the joined pairs
    ``(edge of si, edge of sj)``.
    """
    info: dict = {"grid_mismatch": False, "interior": [], "edges": []}
    wi, wj = _edge_strip_widths(si), _edge_strip_widths(sj)
    a0_all, b0_all = np.asarray(si.edge_le, dtype=float), np.asarray(sj.edge_le, dtype=float)
    va, vb = np.asarray(si.edge_te, dtype=float) - a0_all, np.asarray(sj.edge_te, dtype=float) - b0_all
    la_all, lb_all = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)
    ia, ib = np.flatnonzero(la_all >= 1e-12), np.flatnonzero(lb_all >= 1e-12)
    if ia.size == 0 or ib.size == 0:
        return False, None, info
    b0, lb, wjb = b0_all[ib], lb_all[ib], wj[ib]
    b1 = b0 + vb[ib]
    ub = vb[ib] / lb[:, None]

    def _perp(p: np.ndarray, origin: np.ndarray, unit: np.ndarray) -> np.ndarray:
        # Distance of the points p (..., 3) from the lines through origin with direction unit [m].
        d = p - origin
        return np.linalg.norm(d - np.sum(d * unit, axis=-1)[..., None] * unit, axis=-1)

    # All pairs of edges at once, in blocks of rows of si to bound the memory.
    # The pairs are visited in the same order as a double loop (row-major), so
    # the first closest near miss is the same.
    best: tuple[float, float] | None = None
    joined = False
    interior: set[str] = set()
    boundary = False
    rows = max(1, 2_000_000 // max(ib.size, 1))
    for s0 in range(0, ia.size, rows):
        sel = ia[s0:s0 + rows]
        a0 = a0_all[sel][:, None, :]
        la = la_all[sel][:, None]
        ua = va[sel][:, None, :] / la[..., None]
        a1 = a0 + va[sel][:, None, :]
        gap = np.maximum(
            np.maximum(_perp(b0[None], a0, ua), _perp(b1[None], a0, ua)),
            np.maximum(_perp(a0, b0[None], ub[None]), _perp(a1, b0[None], ub[None])),
        )
        t0 = np.sum((b0[None] - a0) * ua, axis=-1)
        t1 = np.sum((b1[None] - a0) * ua, axis=-1)
        t_lo, t_hi = np.minimum(t0, t1), np.maximum(t0, t1)
        lmin, lmax = np.minimum(la, lb[None]), np.maximum(la, lb[None])
        overlap = np.maximum(0.0, np.minimum(la, t_hi) - np.maximum(0.0, t_lo)) / lmin
        valid = overlap >= JOIN_MIN_OVERLAP
        tol = np.clip(
            JOIN_TOL_STRIP_FRACTION * np.minimum(wi[sel][:, None], wjb[None]),
            JOIN_TOL_CHORD_FRACTION * lmax, JOIN_TOL_CHORD_MAX_FRACTION * lmin,
        )
        close = valid & (gap <= tol)
        grid = ((np.abs(t0) <= tol) & (np.abs(t1 - la) <= tol)) | ((np.abs(t1) <= tol) & (np.abs(t0 - la) <= tol))
        pairs = close & grid
        if np.any(pairs):
            joined = True
            for r, c in zip(*np.nonzero(pairs)):
                info["edges"].append((int(sel[r]), int(ib[c])))
                ei, ej = _interior_edge(si, int(sel[r])), _interior_edge(sj, int(ib[c]))
                if not (ei or ej):
                    boundary = True
                if ei:
                    interior.add("i")
                if ej:
                    interior.add("j")
        if np.any(close & ~grid):
            info["grid_mismatch"] = True
        if joined:
            continue
        cosang = np.minimum(1.0, np.abs(np.sum(ua * ub[None], axis=-1)))
        near = valid & ~close & (np.degrees(np.arccos(cosang)) < NEAR_MISS_MAX_ANGLE_DEG) & (
            gap <= NEAR_MISS_MAX_GAP_CHORDS * lmin)
        if np.any(near):
            k = int(np.argmin(np.where(near, gap, np.inf)))
            g, c = float(gap.flat[k]), float(lmin.flat[k])
            if best is None or g < best[0]:
                best = (g, c)
    if joined:
        info["grid_mismatch"] = False
        # The join tolerance can also accept the interior edge next to a
        # joined boundary edge; only a join with no boundary pair is reported.
        info["interior"] = [] if boundary else sorted(interior)
        return True, None, info
    return False, best, info


def check_overlaps(lattice: VortexLattice) -> None:
    """Refuse a lattice where two surfaces overlap.

    Two control points of different surfaces closer than 1.0e-3 times the
    smaller local chord of the two strips mean that the surfaces overlap
    (they share more than an edge). The strips of one surface never count
    as an overlap with each other. Two surfaces that only share an edge
    still pass: their control points are not at the same place.

    Parameters
    ----------
    lattice : VortexLattice
        Built lattice; control points ``cp`` are in body axes [m] and
        ``chord`` holds the strip chords [m].

    Raises
    ------
    ValueError
        If two control points of different surfaces overlap, naming both
        surfaces.
    """
    if len(lattice.surfaces) < 2:
        return  # the strips of one surface never count as an overlap with each other
    from scipy.spatial import cKDTree

    OVERLAP_FRACTION = 1.0e-3
    cp = np.asarray(lattice.cp, dtype=float)
    if cp.shape[0] < 2:
        return
    panels = np.asarray(lattice.panel_strip, dtype=int)
    panel_surface = np.asarray(lattice.strip_surface, dtype=int)[panels]
    panel_chord = np.asarray(lattice.chord, dtype=float)[panels]
    keep = np.flatnonzero(np.all(np.isfinite(cp), axis=1))
    if keep.size < 2:
        return
    pts = cp[keep]
    surf_of = panel_surface[keep]
    chord = panel_chord[keep]
    r_max = OVERLAP_FRACTION * float(np.max(chord))
    if not np.isfinite(r_max) or r_max <= 0.0:
        return
    tree = cKDTree(pts)
    for i, j in sorted(tree.query_pairs(r_max)):
        if surf_of[i] == surf_of[j]:
            continue
        dist = float(np.linalg.norm(pts[i] - pts[j]))
        limit = OVERLAP_FRACTION * min(float(chord[i]), float(chord[j]))
        if dist < limit:
            name_i = lattice.surfaces[int(surf_of[i])].name
            name_j = lattice.surfaces[int(surf_of[j])].name
            raise ValueError(
                f"Surfaces '{name_i}' and '{name_j}' overlap: control points "
                f"{int(keep[i])} and {int(keep[j])} are {dist:.3g} m apart "
                f"(closer than 1.0e-3 times the smaller local chord, {limit:.3g} m). "
                "Two surfaces at the same place are not valid: move them apart "
                "or join them into one surface."
            )


def core_groups(surfaces: list[SurfaceSlice]) -> np.ndarray:
    """Return the core group of each surface.

    Two surfaces are in the same group if they share an edge: an edge segment
    of one (the line from the leading-edge node to the trailing-edge node of
    one section, after mirroring) lies on an edge segment of the other within
    a tolerance, over at least half of the shorter segment, and the chordwise
    grids of the two edges match (see :func:`_relate_surfaces`). The
    tolerance is the smaller local strip width, with limits. The groups are
    the connected sets of this relation.
    """
    n = len(surfaces)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _relate_surfaces(surfaces[i], surfaces[j])[0]:
                parent[find(j)] = find(i)
    return np.array([find(i) for i in range(n)], dtype=int)


def mark_joined_edges(surfaces: list[SurfaceSlice]) -> None:
    """Set ``edge_joined`` of each surface: True for the edges joined to an edge of another surface.

    The join rule is the one of :func:`core_groups` (see :func:`_relate_surfaces`).
    """
    for sf in surfaces:
        sf.edge_joined = np.zeros(np.asarray(sf.edge_le).shape[0], dtype=bool)
    for i in range(len(surfaces)):
        for j in range(i + 1, len(surfaces)):
            joined, _, info = _relate_surfaces(surfaces[i], surfaces[j])
            if joined:
                for ei, ej in info["edges"]:
                    surfaces[i].edge_joined[ei] = True
                    surfaces[j].edge_joined[ej] = True


def near_miss_warnings(surfaces: list[SurfaceSlice]) -> list[str]:
    """Return the warnings about the edges of surfaces that are near each other.

    * Near miss: two edges are nearly parallel (angle below
      ``NEAR_MISS_MAX_ANGLE_DEG``), overlap, and are further apart than the
      join tolerance but not more than ``NEAR_MISS_MAX_GAP_CHORDS`` local
      chords.
    * Chordwise grids that do not match: two edges lie on each other within
      the join tolerance, but their end nodes are offset along the edge. The
      surfaces are not joined.
    * Interior edge: a joined edge is an interior strip edge of the other
      surface (for example a tail tip on the side of a fin). The join then
      depends on the mesh.

    Surfaces that are in one group through a third surface give no near-miss
    or grid warning.
    """
    groups = core_groups(surfaces)
    out = []
    for i in range(len(surfaces)):
        for j in range(i + 1, len(surfaces)):
            joined, near, info = _relate_surfaces(surfaces[i], surfaces[j])
            names = {"i": surfaces[i].name, "j": surfaces[j].name}
            if joined:
                for side in info["interior"]:
                    other = names["j" if side == "i" else "i"]
                    out.append(
                        f"An edge of surface '{other}' meets an interior strip edge of surface '{names[side]}'. "
                        "The surfaces are joined there, but the join depends on the mesh. Put a section of "
                        f"'{names[side]}' at the junction, or split it there."
                    )
                continue
            if groups[i] == groups[j]:
                continue
            if info["grid_mismatch"]:
                out.append(
                    f"Surfaces '{surfaces[i].name}' and '{surfaces[j].name}' have edges that lie on each other, "
                    "but their chordwise grids do not match (the leading-edge or trailing-edge nodes are offset "
                    "along the edge). They are not joined: the cross-surface core acts between them. To join "
                    "them, make the leading and trailing edges of the two edges match."
                )
            elif near is not None:
                out.append(
                    f"Surfaces '{surfaces[i].name}' and '{surfaces[j].name}' have edges {1e3 * near[0]:.1f} mm "
                    "apart that are nearly parallel. They are not joined: the cross-surface core acts "
                    "between them. If they are one surface, make the edges match."
                )
    return out


def resolve_spacing(spacing: str | None, surf: LiftingSurface, collocation: str) -> str:
    """Spacing rule for a surface. ``"auto"`` depends on the method.

    Lifting line: always tip clustering (``"half-cosine"``, which is cosine
    spacing over the whole span). Clustering at the root makes the lifting
    line worse wherever the quarter-chord line has a kink (sweep, or the
    forward sweep of a tapered wing with a straight leading edge).
    Vortex lattice: :func:`determine_optimal_spacing`.
    """
    s = (spacing or "auto").lower().replace("_", "-")
    if s != "auto":
        return s
    if collocation == "llt":
        return "half-cosine"
    from ventorum.geometry.discretization import determine_optimal_spacing
    return determine_optimal_spacing(surf)


def _section_breaks(surf: LiftingSurface) -> list[float]:
    """Span fractions of interior sections where the planform has a kink or the airfoil changes."""
    secs = _sorted_sections(surf)
    # A cranked wing has a few breaks; many sections describe a smooth curve
    # (for example an elliptic or circular planform), which needs no snapping.
    if len(secs) < 3 or len(secs) > 6:
        return []
    fr = np.array([s.y_frac for s in secs])
    out = []
    for k in range(1, len(secs) - 1):
        lo, mid, hi = secs[k - 1], secs[k], secs[k + 1]
        if not (mid.airfoil is lo.airfoil and mid.airfoil is hi.airfoil) and (
                repr(mid.airfoil) != repr(lo.airfoil) or repr(mid.airfoil) != repr(hi.airfoil)):
            out.append(float(fr[k]))
            continue
        d0, d1 = fr[k] - fr[k - 1], fr[k + 1] - fr[k]
        for attr in ("chord", "twist", "x_le", "z_le"):
            v = [getattr(lo, attr), getattr(mid, attr), getattr(hi, attr)]
            if any(x is None for x in v):
                continue
            s0, s1 = (v[1] - v[0]) / d0, (v[2] - v[1]) / d1
            if abs(s1 - s0) > 1e-9 * max(1.0, abs(s0), abs(s1)):
                out.append(float(fr[k]))
                break
    return out


def _surface_eta(surf: LiftingSurface, n_panels: int, spacing: str) -> tuple[np.ndarray, np.ndarray]:
    """Panel edges and control-point span fractions (mid parameter) of one half.

    A section where the planform has a kink or the airfoil changes is put on
    a strip edge (the nearest edge moves to it), so no strip averages across
    the break. This is done only for surfaces with at most 4 interior
    sections and when the breaks are at most a third of the panel count; a
    smooth planform given by many sections keeps the plain spacing.

    Control surface span limits (not 0 and not 1) are snapped after the
    section breaks, with the same nearest-edge and quarter-width rules,
    independently of the section-count rule.
    """
    edges, mids = get_spacing(spacing, n_panels, surf=surf)
    edges = np.asarray(edges, dtype=float).copy()
    mids = np.asarray(mids, dtype=float).copy()
    edges[0], edges[-1] = 0.0, 1.0
    breaks = _section_breaks(surf)
    moved: set[int] = set()
    if breaks and len(breaks) <= max(1, (len(edges) - 1) // 3):
        for eb in breaks:
            cand = [i for i in range(1, len(edges) - 1) if i not in moved]
            if not cand:
                break
            i = min(cand, key=lambda j: abs(edges[j] - eb))
            if abs(edges[i] - eb) < 1e-12:
                moved.add(i)
                continue
            # Keep both neighbouring strips at least a quarter of their old width.
            w_left, w_right = edges[i] - edges[i - 1], edges[i + 1] - edges[i]
            if eb - edges[i - 1] < 0.25 * w_left or edges[i + 1] - eb < 0.25 * w_right:
                continue
            edges[i] = eb
            mids[i - 1] = 0.5 * (edges[i - 1] + edges[i])
            mids[i] = 0.5 * (edges[i] + edges[i + 1])
            moved.add(i)

    controls = getattr(surf, "controls", [])
    if controls:
        ctrl_limits = []
        for c in controls:
            for lim in (c.eta_start, c.eta_end):
                if 1e-12 < lim < 1.0 - 1e-12:
                    ctrl_limits.append(float(lim))
        for clim in ctrl_limits:
            dist = np.abs(edges - clim)
            min_dist_idx = int(np.argmin(dist))
            if dist[min_dist_idx] < 1e-12:
                if 1 <= min_dist_idx < len(edges) - 1:
                    moved.add(min_dist_idx)
                continue
            cand = [i for i in range(1, len(edges) - 1) if i not in moved]
            if not cand:
                continue
            i = min(cand, key=lambda j: abs(edges[j] - clim))
            w_left, w_right = edges[i] - edges[i - 1], edges[i + 1] - edges[i]
            if clim - edges[i - 1] < 0.25 * w_left or edges[i + 1] - clim < 0.25 * w_right:
                continue
            edges[i] = clim
            mids[i - 1] = 0.5 * (edges[i - 1] + edges[i])
            mids[i] = 0.5 * (edges[i] + edges[i + 1])
            moved.add(i)

    return edges, mids


def _surface_mesh_eta(
    surf: LiftingSurface,
    settings: SolverSettings,
    collocation: str,
    ref_semi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Spanwise strip edges and control-point stations of the defining half of a surface.

    This function applies the panel-count rule (the surface ``n_panels``, the
    settings ``n_panels`` or the proportional panel count) and the spacing
    rule of the solver. :func:`build_lattice` and
    :func:`ventorum.geometry.deformation.undeformed_nodes` both use it, so
    they always give the same stations.

    Parameters
    ----------
    surf : LiftingSurface
        The surface.
    settings : SolverSettings
        Spanwise panel count and spacing.
    collocation : {"vlm", "llt"}
        Control-point rule. The ``"auto"`` spacing depends on it.
    ref_semi : float
        Reference semi-span for proportional panel counts [m]: the largest
        semi-span of the aircraft surfaces.

    Returns
    -------
    tuple of numpy.ndarray
        The edge stations ``eta_e`` (n_edges,) and the control-point
        stations ``eta_mid`` (n_edges - 1,), as fractions of the semi-span.
    """
    if getattr(settings, "proportional_panels", False) and surf.n_panels is None:
        n_sp = compute_surface_n_panels(
            surf, base_n_panels=settings.n_panels, reference_semi_span=ref_semi,
            min_panels=getattr(settings, "min_panels", 8),
        )
    else:
        n_sp = surf.n_panels if surf.n_panels is not None else settings.n_panels
    spacing = resolve_spacing(surf.spacing if surf.spacing is not None else settings.spacing, surf, collocation)
    return _surface_eta(surf, int(n_sp), spacing)


def _chordwise_fractions(n_chord: int, chord_spacing: str) -> np.ndarray:
    if chord_spacing in ("cosine", "full-cosine"):
        return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_chord + 1)))
    return np.linspace(0.0, 1.0, n_chord + 1)


#: Chordwise stations [-] that test whether a mean line has camber.
_CAMBER_PROBE = np.linspace(0.0, 1.0, 21)


def _hinge_node(xi: np.ndarray, hinge_x_c: float) -> tuple[int, bool]:
    """Return the chordwise node nearest to a hinge and if it can move there.

    Parameters
    ----------
    xi : np.ndarray
        Chordwise node fractions from 0 to 1 [-].
    hinge_x_c : float
        Hinge position as a fraction of the chord [-].

    Returns
    -------
    tuple of (int, bool)
        Index of the interior node nearest to the hinge, and True when the
        node can move onto the hinge inside the window of
        ``HINGE_PANEL_MIN_FACTOR`` and ``HINGE_PANEL_MAX_FACTOR``.
    """
    n_chord = len(xi) - 1
    k_hinge = 1 + int(np.argmin(np.abs(xi[1:n_chord] - hinge_x_c)))
    len_before_old = xi[k_hinge] - xi[k_hinge - 1]
    len_after_old = xi[k_hinge + 1] - xi[k_hinge]
    len_before_new = hinge_x_c - xi[k_hinge - 1]
    len_after_new = xi[k_hinge + 1] - hinge_x_c
    tol = 1e-12
    lo, hi = HINGE_PANEL_MIN_FACTOR, HINGE_PANEL_MAX_FACTOR
    fits = (lo * len_before_old - tol <= len_before_new <= hi * len_before_old + tol and
            lo * len_after_old - tol <= len_after_new <= hi * len_after_old + tol)
    return k_hinge, bool(fits)


def _hinge_chord_counts(hinge_x_c: float, n_chord: int, chord_spacing: str, n_max: int = 3) -> list[int]:
    """Return chordwise panel counts near *n_chord* that can hold a hinge node.

    Parameters
    ----------
    hinge_x_c : float
        Hinge position as a fraction of the chord [-].
    n_chord : int
        Present chordwise panel count.
    chord_spacing : str
        Chordwise panel distribution.
    n_max : int, optional
        Largest number of counts to return.

    Returns
    -------
    list of int
        Counts between 2 and ``max(4 * n_chord, 100)`` for which a lattice node can
        move onto the hinge. They are the counts nearest to *n_chord*, in
        increasing order.
    """
    out = []
    for n in sorted(range(2, max(4 * n_chord, 100) + 1), key=lambda m: (abs(m - n_chord), m)):
        if _hinge_node(_chordwise_fractions(n, chord_spacing), hinge_x_c)[1]:
            out.append(n)
            if len(out) >= n_max:
                break
    return sorted(out)


class _MeanLine:
    """Piecewise linear section mean line in chord units.

    Katz and Plotkin (2001) place the lumped vortices on this line.
    The union of the two surface grids gives the exact slope of
    ``AirfoilShape.camber`` between its knots.
    """

    def __init__(self, spec: Any):
        from ventorum.airfoil import airfoil

        shape = airfoil(spec).normalized()
        self.x = np.unique(np.concatenate([shape.upper()[:, 0], shape.lower()[:, 0], [0.0, 1.0]]))
        self.z = np.asarray(shape.camber(self.x))
        self.slope = np.diff(self.z) / np.diff(self.x)

    def at(self, xi: np.ndarray, *, derivative: bool = False) -> np.ndarray:
        """Return height [chord units] or slope [-] at chord fractions *xi*."""
        if derivative:
            k = np.clip(np.searchsorted(self.x, xi, side="right") - 1, 0, self.slope.size - 1)
            return np.where((xi < self.x[0]) | (xi > self.x[-1]), 0.0, self.slope[k])
        return np.interp(xi, self.x, self.z)

    def alpha_l0(self) -> float:
        """Return the thin-airfoil zero-lift angle [rad].

        Integrate each constant slope exactly with x = (1 - cos(theta))/2
        (Katz and Plotkin, 2001, thin-airfoil Fourier coefficients).
        """
        theta = np.arccos(1.0 - 2.0 * np.clip(self.x, 0.0, 1.0))
        return float(np.sum(self.slope * np.diff(theta - np.sin(theta))) / np.pi)


def _mean_line_at(
    sections: list[WingSection], lines: list[_MeanLine | None], eta: np.ndarray, xi: np.ndarray,
    *, derivative: bool = False,
) -> np.ndarray:
    """Interpolate mean-line heights [chord units] or slopes [-] at fixed *xi*."""
    eta, xi = np.broadcast_arrays(eta, xi)
    fr = np.array([s.y_frac for s in sections])
    values = np.array([np.zeros_like(xi) if line is None else line.at(xi, derivative=derivative) for line in lines])
    j = np.clip(np.searchsorted(fr, eta, side="right") - 1, 0, len(fr) - 2)
    w = np.clip((eta - fr[j]) / (fr[j + 1] - fr[j]), 0.0, 1.0)
    lo = np.take_along_axis(values, j[None], axis=0)[0]
    hi = np.take_along_axis(values, (j + 1)[None], axis=0)[0]
    return (1.0 - w) * lo + w * hi


def _camber_edge_offset(frames: list[dict], surface: int, edge: int, xi: np.ndarray) -> np.ndarray:
    """Return a section mean-line offset [m] in the displaced edge frame."""
    f = frames[surface]
    z = _mean_line_at(f["sections"], f["lines"], np.abs(f["eta"][edge]), np.asarray(xi))
    return f["chord"][edge] * z[..., None] * f["normal"][edge]


def _joined_camber_offset(frames: list[dict], surface: int, edge: int, xi: np.ndarray) -> np.ndarray:
    """Return the mean offset [m] of the distinct edges at a junction."""
    members = frames[surface]["joined"].get(edge, [(surface, edge)])
    return np.mean([_camber_edge_offset(frames, s, e, xi) for s, e in members], axis=0)


def _camber_edge_frames(
    surfaces: list[SurfaceSlice], mean_surfaces: list[tuple], aircraft: Aircraft,
) -> tuple[list[dict] | None, list[str]]:
    """Build chord frames and shared-edge groups for mean-line offsets [m].

    The existing surface join rule defines the groups. Each distinct edge
    contributes once to the mean, including an edge with zero offset.
    """
    if not any(all(line is not None for line in lines) for _, lines, _ in mean_surfaces):
        return None, []
    frames = []
    parent = {}
    for si, (sf, (secs, lines, eta_e)) in enumerate(zip(surfaces, mean_surfaces)):
        edge_c = np.linalg.norm(sf.edge_te - sf.edge_le, axis=1)
        edge_dir = (sf.edge_te - sf.edge_le) / np.maximum(edge_c[:, None], 1e-300)
        q = sf.edge_le + 0.25 * (sf.edge_te - sf.edge_le)
        span = np.gradient(q, axis=0)
        span[:, 0] = 0.0
        normal = cross3(edge_dir, span)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-300)
        if sf.is_symmetric:
            eta = np.concatenate([-eta_e[::-1], eta_e[1:]])
        elif getattr(aircraft.surfaces[sf.index], "mirror_y", False):
            eta = -eta_e[::-1]
        else:
            eta = eta_e
        frames.append({"sections": secs, "lines": lines, "eta": eta, "chord": edge_c,
                       "normal": normal, "le": sf.edge_le.copy(), "te": sf.edge_te.copy(), "joined": {}})
        for e in range(len(eta)):
            parent[si, e] = (si, e)

    def find(edge):
        while parent[edge] != edge:
            parent[edge] = parent[parent[edge]]
            edge = parent[edge]
        return edge

    for i, si in enumerate(surfaces):
        for j in range(i + 1, len(surfaces)):
            joined, _, info = _relate_surfaces(si, surfaces[j])
            if joined:
                for ei, ej in info["edges"]:
                    parent[find((j, ej))] = find((i, ei))
    groups = {}
    for edge in parent:
        groups.setdefault(find(edge), []).append(edge)
    warnings_out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        knots = [np.array([0.0, 1.0])]
        for s, e in members:
            frames[s]["joined"][e] = members
            knots.extend(line.x for line in frames[s]["lines"] if line is not None)
        xi = np.unique(np.concatenate(knots))
        offsets = [_camber_edge_offset(frames, s, e, xi) for s, e in members]
        gap = max(float(np.max(np.linalg.norm(a - b, axis=-1))) for a in offsets for b in offsets)
        chord = min(frames[s]["chord"][e] for s, e in members)
        # Report a difference above the lower geometric join tolerance.
        if gap > JOIN_TOL_CHORD_FRACTION * chord:
            names = ", ".join(repr(surfaces[s].name) for s in sorted({s for s, _ in members}))
            warnings_out.append(
                f"Shared-edge camber offsets of surfaces {names} differ by {gap:.6g} m. "
                "The joined edge uses their mean offset."
            )
    return frames, warnings_out


def build_lattice(
    aircraft: Aircraft,
    settings: SolverSettings | None = None,
    *,
    collocation: str = "vlm",
    n_chord: int = 1,
    chord_spacing: str = "uniform",
) -> VortexLattice:
    """Build the vortex lattice of an aircraft.

    Parameters
    ----------
    aircraft : Aircraft
    settings : SolverSettings
        Spanwise panel count and spacing. A surface's own ``n_panels`` and
        ``spacing`` take precedence.
    collocation : {"vlm", "llt"}
        Control-point rule (see the module docstring).
    n_chord : int
        Chordwise panels per strip (``"llt"`` always uses 1).
    chord_spacing : {"uniform", "cosine"}
        Chordwise panel distribution.
    """
    settings = settings or SolverSettings()
    if collocation not in ("vlm", "llt"):
        raise ValueError(f"Unknown collocation rule {collocation!r}; use 'vlm' or 'llt'.")
    if collocation == "llt":
        n_chord = 1
    n_chord = max(1, int(n_chord))
    xi = _chordwise_fractions(n_chord, chord_spacing)

    for surf in aircraft.surfaces:
        validate_controls(surf)

    is_chordwise = (collocation == "vlm" and n_chord >= 2)
    mode_str = "chordwise" if is_chordwise else "section"

    ref_semi = max((s.semi_span for s in aircraft.surfaces), default=1.0)

    strip_parts: dict[str, list[Any]] = {k: [] for k in (
        "surface", "le_l", "le_r", "te_l", "te_r", "chord", "twist", "a0", "aL0", "cd0", "cm0",
        "mirror", "is_right", "eta", "frac",
    )}
    strip_xi_all: list[np.ndarray] = []
    strip_ctrl_all: list[ControlSurface | None] = []
    strip_hinge_eff_all: list[float | None] = []
    # Chordwise mode: the effective hinge fraction of every strip with a
    # deflected geometry (control strips and join strips), and the
    # deflection terms of its two edge nodes (see ControlJoinField).
    strip_flap_hinge_all: list[float | None] = []
    strip_flap_nominal_all: list[float] = []
    strip_side_terms_all: list[tuple[list, list]] = []
    control_info: list[dict[str, Any]] = []
    # Warnings of the mesh construction (VortexLattice.mesh_warnings).
    mesh_warnings: list[str] = []
    mean_surfaces: list[tuple[list[WingSection], list[_MeanLine | None], np.ndarray]] = []
    camber_warnings: list[str] = []
    camber_fallback = False

    airfoils: list[AirfoilType] = []
    surfaces: list[SurfaceSlice] = []
    strip_offset = 0

    # One deflected airfoil for each (base airfoil, deflection, hinge). The
    # loads group the tabulated strips by airfoil object, so a new object for
    # each strip makes the nonlinear solve much slower. Each memo value also
    # holds the base airfoil, so its id stays unique while the memo lives.
    deflected_memo: dict[tuple[int, float, float], tuple[AirfoilType, AirfoilType]] = {}

    def _deflected(base: AirfoilType, delta: float, hinge_x_c: float) -> AirfoilType:
        key = (id(base), float(delta), float(hinge_x_c))
        hit = deflected_memo.get(key)
        if hit is None:
            hit = (base, deflected_airfoil(base, delta, hinge_x_c))
            deflected_memo[key] = hit
        return hit[1]

    for s_idx, surf in enumerate(aircraft.surfaces):
        eta_e, eta_mid = _surface_mesh_eta(surf, settings, collocation, ref_semi)
        secs = _sorted_sections(surf)
        supplied = [sec.shape is not None for sec in secs]
        # The ground-effect note concerns cambered sections that keep the
        # turned normal: a section without a mean line and with a nonzero
        # polar alpha_L0 or Cm0 (an uncambered section has no camber error).
        camber_fallback |= not all(supplied) and any(
            any(v != 0.0 for v in airfoil_linear_properties(sec.airfoil)[1::2]) for sec in secs
        )
        if any(supplied) and not all(supplied):
            camber_warnings.append(
                f"camber of surface {surf.name} modelled by a turned normal: not every section gives a mean line"
            )
        use_mean_line = collocation == "vlm" and n_chord > 1 and all(supplied)
        lines = [_MeanLine(sec.shape) if use_mean_line else None for sec in secs]
        for sec, line in zip(secs, lines):
            if line is not None and abs(line.alpha_l0() - sec.airfoil.alpha_L0) > np.radians(0.5):
                warnings.warn(
                    f"[{surf.name}] section at eta={sec.y_frac:g}: mean line and polar alpha_L0 "
                    "differ by more than 0.5 deg. The mean line sets the VLM boundary condition.",
                    RuntimeWarning, stacklevel=2,
                )
        mean_surfaces.append((secs, lines, eta_e))
        geo = surface_edge_geometry(surf, eta_e)
        d = getattr(surf, "node_displacements", None)
        if d is not None:
            if d.eta is not None and (
                d.eta.shape != eta_e.shape or float(np.max(np.abs(d.eta - eta_e))) > 1e-12
            ):
                raise ValueError(
                    f"[{surf.name}] node_displacements.eta does not match the spanwise stations of the "
                    f"mesh ({eta_e.size} edges). The displacements were made for another mesh or "
                    "solver. Use the same settings and solver for "
                    "ventorum.geometry.undeformed_nodes and for the analysis."
                )
            expected_shape = (len(eta_e), 3)
            if d.le.shape != expected_shape:
                raise ValueError(
                    f"[{surf.name}] node_displacements.le has shape {d.le.shape}, expected {expected_shape}. "
                    "Use ventorum.geometry.undeformed_nodes to get the node positions of the mesh."
                )
            if d.te.shape != expected_shape:
                raise ValueError(
                    f"[{surf.name}] node_displacements.te has shape {d.te.shape}, expected {expected_shape}. "
                    "Use ventorum.geometry.undeformed_nodes to get the node positions of the mesh."
                )
            if not np.all(np.isfinite(d.le)):
                raise ValueError(f"[{surf.name}] node_displacements.le contains non-finite values.")
            if not np.all(np.isfinite(d.te)):
                raise ValueError(f"[{surf.name}] node_displacements.te contains non-finite values.")
            max_le = float(np.max(np.linalg.norm(d.le, axis=1))) if d.le.size else 0.0
            max_te = float(np.max(np.linalg.norm(d.te, axis=1))) if d.te.size else 0.0
            if max_le > surf.semi_span * (1.0 + 1e-9):
                raise ValueError(
                    f"[{surf.name}] node_displacements.le has displacement {max_le:.4g} m larger than semi-span {surf.semi_span:.4g} m."
                )
            if max_te > surf.semi_span * (1.0 + 1e-9):
                raise ValueError(
                    f"[{surf.name}] node_displacements.te has displacement {max_te:.4g} m larger than semi-span {surf.semi_span:.4g} m."
                )
            le_e = geo["le"] + d.le
            te_e = geo["te"] + d.te
            chord_undef = geo["chord"]
            chord_def = np.linalg.norm(te_e - le_e, axis=1)
            if np.any(chord_def <= 1e-6 * chord_undef):
                k = int(np.flatnonzero(chord_def <= 1e-6 * chord_undef)[0])
                raise ValueError(
                    f"[{surf.name}] node_displacements: deformed chord at edge {k} ({chord_def[k]:.4g} m) "
                    f"is not larger than 1e-6 times undeformed chord ({chord_undef[k]:.4g} m)."
                )
            qc_undef = geo["le"] + 0.25 * (geo["te"] - geo["le"])
            dl_undef = qc_undef[1:] - qc_undef[:-1]
            width_undef = np.sqrt(dl_undef[:, 1] ** 2 + dl_undef[:, 2] ** 2)
            qc_def = le_e + 0.25 * (te_e - le_e)
            dl_def = qc_def[1:] - qc_def[:-1]
            width_def = np.sqrt(dl_def[:, 1] ** 2 + dl_def[:, 2] ** 2)
            if np.any(width_def <= 1e-6 * width_undef):
                k = int(np.flatnonzero(width_def <= 1e-6 * width_undef)[0])
                raise ValueError(
                    f"[{surf.name}] node_displacements: deformed strip width of strip {k} ({width_def[k]:.4g} m) "
                    f"is not larger than 1e-6 times undeformed width ({width_undef[k]:.4g} m)."
                )
            edge_chord_half = chord_def
        else:
            le_e, te_e = geo["le"], geo["te"]
            edge_chord_half = geo["chord"]
        # Fraction of the strip width (root edge to tip edge) where the control point is.
        frac_r = (eta_mid - eta_e[:-1]) / np.maximum(eta_e[1:] - eta_e[:-1], 1e-300)
        n_r = len(eta_mid)

        blender = _AirfoilBlender(_sorted_sections(surf))
        base_right_af = [blender.at(float(e)) for e in eta_mid]

        controls = getattr(surf, "controls", [])
        ctrl_for_k: list[ControlSurface | None] = [None] * n_r
        hinge_eff_for_k: list[float | None] = [None] * n_r
        xi_for_k: list[np.ndarray] = [xi] * n_r
        snap_of: dict[int, tuple[float | None, np.ndarray]] = {}

        for c in controls:
            k_ctrl = np.flatnonzero((eta_mid >= c.eta_start - 1e-9) & (eta_mid <= c.eta_end + 1e-9))
            if len(k_ctrl) == 0:
                raise ValueError(
                    f"Control {c.name!r} on surface {surf.name!r} has no strip; "
                    "use more panels or a wider span."
                )
            eta_start_eff = float(eta_e[k_ctrl[0]])
            eta_end_eff = float(eta_e[k_ctrl[-1] + 1])
            if is_chordwise:
                k_hinge, fits = _hinge_node(xi, c.hinge_x_c)
                if fits:
                    hinge_x_c_eff = float(c.hinge_x_c)
                    ctrl_xi = xi.copy()
                    ctrl_xi[k_hinge] = c.hinge_x_c
                    cambered = any(
                        line is not None and float(np.max(np.abs(line.at(_CAMBER_PROBE)))) > 0.0
                        for line in lines
                    )
                    if cambered and abs(float(c.hinge_x_c) - float(xi[k_hinge])) > 1e-12:
                        counts = _hinge_chord_counts(float(c.hinge_x_c), n_chord, chord_spacing)
                        mesh_warnings.append(
                            f"[{surf.name}] control {c.name!r}: the hinge at x/c = {float(c.hinge_x_c):.4g} "
                            f"moves a chordwise node of the control strips (n_chord = {n_chord}). On a "
                            "cambered mean line the legs of the control strips and of their clean "
                            "neighbours then do not coincide at the control side edges, and the leg "
                            "forces there do not converge with the spanwise mesh. Use a chordwise count "
                            "with a node at the hinge"
                            + (f", for example n_chord = {', '.join(str(n) for n in counts)}." if counts else ".")
                        )
                        warnings.warn(mesh_warnings[-1], RuntimeWarning, stacklevel=2)
                else:
                    hinge_x_c_eff = float(xi[k_hinge])
                    ctrl_xi = xi.copy()
                    counts = _hinge_chord_counts(float(c.hinge_x_c), n_chord, chord_spacing)
                    advice = (
                        f"Use a chordwise count that puts a node near the hinge, for example "
                        f"n_chord = {', '.join(str(n) for n in counts)} ({chord_spacing} spacing)."
                        if counts else "Use a chordwise count that puts a node near the hinge."
                    )
                    mesh_warnings.append(
                        f"[{surf.name}] control {c.name!r}: no lattice node can move onto the hinge at "
                        f"x/c = {float(c.hinge_x_c):.4g} with n_chord = {n_chord}. The hinge snaps to the "
                        f"nearest node at x/c = {hinge_x_c_eff:.4g} (lattice control_info 'hinge_x_c_eff'). "
                        + advice
                    )
                    warnings.warn(mesh_warnings[-1], RuntimeWarning, stacklevel=2)
            else:
                hinge_x_c_eff = None
                ctrl_xi = xi.copy()

            control_info.append({
                "surface": surf.name,
                "name": c.name,
                "mode": mode_str,
                "eta_start_eff": eta_start_eff,
                "eta_end_eff": eta_end_eff,
                "hinge_x_c": float(c.hinge_x_c),
                "hinge_x_c_eff": hinge_x_c_eff,
                "n_strips": len(k_ctrl),
                "deflection_rad": float(c.deflection),
                "symmetric": bool(c.symmetric),
            })

            snap_of[id(c)] = (hinge_x_c_eff, ctrl_xi)
            for k in k_ctrl:
                ctrl_for_k[k] = c
                hinge_eff_for_k[k] = hinge_x_c_eff
                xi_for_k[k] = ctrl_xi

        # Chordwise mode: the deflected geometry of the controls and of
        # their joins (sealed side edges, see ControlJoinField). A strip in
        # a join outside its control takes the hinge and the chordwise nodes
        # of that control.
        if surf.is_symmetric:
            kind = "symmetric"
        elif getattr(surf, "mirror_y", False):
            kind = "mirror"
        else:
            kind = "plain"
        side_sign = -1.0 if kind == "mirror" else 1.0
        join_field = None
        flap_hinge_for_k: list[float | None] = [None] * n_r
        flap_nominal_for_k: list[float] = [np.nan] * n_r
        if is_chordwise and controls:
            def chord_of(e: float, _surf=surf) -> float:
                return float(surface_edge_geometry(_surf, np.array([e]))["chord"][0])

            def rate_of(e: float, _surf=surf) -> float:
                # Length of the reference line per unit span parameter [m].
                h_ = 1e-6
                lo_, hi_ = max(0.0, e - h_), min(1.0, e + h_)
                _, y_r, z_r = surface_reference_line(_surf, np.array([lo_, hi_]))
                return float(np.hypot(y_r[1] - y_r[0], z_r[1] - z_r[0])) / max(hi_ - lo_, 1e-300)

            join_field = ControlJoinField(controls, kind, chord_of, rate_of)
            for k in range(n_r):
                hc = None
                for e in (eta_mid[k], eta_e[k], eta_e[k + 1]):
                    hc = join_field.at(side_sign * float(e))[1]
                    if hc is not None:
                        break
                if hc is None:
                    continue
                h_eff, c_xi = snap_of[id(hc)]
                flap_hinge_for_k[k] = h_eff
                flap_nominal_for_k[k] = float(hc.hinge_x_c)
                if ctrl_for_k[k] is None:
                    xi_for_k[k] = c_xi

        props_of: dict[int, tuple[float, float, float, float]] = {}
        if not is_chordwise and controls:
            right_af = []
            for k in range(n_r):
                c = ctrl_for_k[k]
                if c is not None:
                    right_af.append(_deflected(base_right_af[k], c.deflection, c.hinge_x_c))
                else:
                    right_af.append(base_right_af[k])
        else:
            right_af = list(base_right_af)

        for af in right_af:
            if id(af) not in props_of:
                props_of[id(af)] = airfoil_linear_properties(af)
        right_props = np.array([props_of[id(af)] for af in right_af]).reshape(-1, 4)

        has_asymmetric_deflection = any(
            not c.symmetric and c.deflection != 0.0 for c in controls
        )

        mirror = np.array([1.0, -1.0, 1.0])
        if surf.is_symmetric and getattr(surf, "mirror_y", False):
            raise ValueError(f"[{surf.name}] mirror_y=True needs is_symmetric=False.")
        if surf.is_symmetric:
            tol = 1e-9 * max(1.0, surf.semi_span)
            if abs(le_e[0, 1]) > tol or abs(te_e[0, 1]) > tol:
                raise ValueError(
                    f"[{surf.name}] is symmetric, but its root is at y = {le_e[0, 1]:.4g} m, not on the plane "
                    "y = 0. A symmetric surface is mirrored about y = 0 and must start there. For a surface "
                    "off the plane (for example twin fins), set is_symmetric=False and add surf.mirrored() "
                    "to the aircraft as the left copy."
                )
            mirror = np.array([1.0, -1.0, 1.0])
            le_left_e = (le_e * mirror)[::-1]
            te_left_e = (te_e * mirror)[::-1]
            edge_ref = np.vstack([(geo["qc"] * mirror)[::-1], geo["qc"][1:]])
            edge_le = np.vstack([le_left_e, le_e[1:]])
            edge_te = np.vstack([te_left_e, te_e[1:]])
            edge_chord = np.concatenate([edge_chord_half[::-1], edge_chord_half[1:]])
            edge_twist = np.concatenate([geo["twist"][::-1], geo["twist"][1:]])

            if not is_chordwise and controls:
                left_af = []
                for j in range(n_r):
                    k = n_r - 1 - j
                    c = ctrl_for_k[k]
                    if c is not None:
                        delta = c.deflection if c.symmetric else -c.deflection
                        left_af.append(_deflected(base_right_af[k], delta, c.hinge_x_c))
                    else:
                        left_af.append(base_right_af[k])
                for af in left_af:
                    if id(af) not in props_of:
                        props_of[id(af)] = airfoil_linear_properties(af)
                left_props = np.array([props_of[id(af)] for af in left_af]).reshape(-1, 4)
                afs = left_af + right_af
                props = np.vstack([left_props, right_props])
            else:
                afs = right_af[::-1] + right_af
                props = np.vstack([right_props[::-1], right_props])

            eta_s = np.concatenate([-eta_mid[::-1], eta_mid])
            if has_asymmetric_deflection:
                local_mirror = np.full(2 * n_r, -1)
            else:
                # Left strip k (k = 0 at the left tip) mirrors right strip n_r - 1 - k.
                local_mirror = np.concatenate([n_r + np.arange(n_r)[::-1], np.arange(n_r)[::-1]])
            is_right = np.concatenate([np.zeros(n_r, bool), np.ones(n_r, bool)])
            # Left strips run from the tip edge to the root edge.
            frac = np.concatenate([(1.0 - frac_r)[::-1], frac_r])

            xi_surf = [xi_for_k[n_r - 1 - j] for j in range(n_r)] + xi_for_k
            hinge_eff_surf = [hinge_eff_for_k[n_r - 1 - j] for j in range(n_r)] + hinge_eff_for_k
            ctrl_surf = [ctrl_for_k[n_r - 1 - j] for j in range(n_r)] + ctrl_for_k
            flap_hinge_surf = flap_hinge_for_k[::-1] + flap_hinge_for_k
            flap_nominal_surf = flap_nominal_for_k[::-1] + flap_nominal_for_k
            node_eta = np.concatenate([-eta_e[::-1], eta_e[1:]])
        elif getattr(surf, "mirror_y", False):
            # Mirror image (y -> -y) of the surface; strips run left to right
            # (tip to root), as on the left half of a symmetric surface.
            edge_le = (le_e * mirror)[::-1]
            edge_te = (te_e * mirror)[::-1]
            edge_ref = (geo["qc"] * mirror)[::-1]
            edge_chord, edge_twist = edge_chord_half[::-1], geo["twist"][::-1]
            if not is_chordwise and controls:
                afs = []
                for j in range(n_r):
                    k = n_r - 1 - j
                    c = ctrl_for_k[k]
                    if c is not None:
                        delta = c.deflection if c.symmetric else -c.deflection
                        afs.append(_deflected(base_right_af[k], delta, c.hinge_x_c))
                    else:
                        afs.append(base_right_af[k])
                for af in afs:
                    if id(af) not in props_of:
                        props_of[id(af)] = airfoil_linear_properties(af)
                props = np.array([props_of[id(af)] for af in afs]).reshape(-1, 4)
            else:
                afs = right_af[::-1]
                props = right_props[::-1]
            eta_s = -eta_mid[::-1]
            local_mirror = np.full(n_r, -1)
            is_right = np.zeros(n_r, bool)
            frac = (1.0 - frac_r)[::-1]

            xi_surf = [xi_for_k[n_r - 1 - j] for j in range(n_r)]
            hinge_eff_surf = [hinge_eff_for_k[n_r - 1 - j] for j in range(n_r)]
            ctrl_surf = [ctrl_for_k[n_r - 1 - j] for j in range(n_r)]
            flap_hinge_surf = flap_hinge_for_k[::-1]
            flap_nominal_surf = flap_nominal_for_k[::-1]
            node_eta = -eta_e[::-1]
        else:
            edge_le, edge_te = le_e, te_e
            edge_ref = geo["qc"]
            edge_chord, edge_twist = edge_chord_half, geo["twist"]
            afs = right_af
            props = right_props
            eta_s = eta_mid
            local_mirror = np.full(n_r, -1)
            is_right = np.ones(n_r, bool)
            frac = frac_r

            xi_surf = list(xi_for_k)
            hinge_eff_surf = list(hinge_eff_for_k)
            ctrl_surf = list(ctrl_for_k)
            flap_hinge_surf = list(flap_hinge_for_k)
            flap_nominal_surf = list(flap_nominal_for_k)
            node_eta = np.asarray(eta_e, dtype=float)

        n_s = edge_le.shape[0] - 1
        strip_parts["surface"].append(np.full(n_s, s_idx))
        strip_parts["le_l"].append(edge_le[:-1])
        strip_parts["le_r"].append(edge_le[1:])
        strip_parts["te_l"].append(edge_te[:-1])
        strip_parts["te_r"].append(edge_te[1:])
        strip_parts["chord"].append(0.5 * (edge_chord[:-1] + edge_chord[1:]))
        strip_parts["twist"].append(0.5 * (edge_twist[:-1] + edge_twist[1:]))
        strip_parts["a0"].append(props[:, 0])
        strip_parts["aL0"].append(props[:, 1])
        strip_parts["cd0"].append(props[:, 2])
        strip_parts["cm0"].append(props[:, 3])
        strip_parts["mirror"].append(np.where(local_mirror >= 0, local_mirror + strip_offset, -1))
        strip_parts["is_right"].append(is_right)
        strip_parts["eta"].append(eta_s)
        strip_parts["frac"].append(frac)
        strip_xi_all.extend(xi_surf)
        strip_hinge_eff_all.extend(hinge_eff_surf)
        strip_ctrl_all.extend(ctrl_surf)
        strip_flap_hinge_all.extend(flap_hinge_surf)
        strip_flap_nominal_all.extend(flap_nominal_surf)
        if join_field is not None:
            node_terms = [join_field.at(float(e))[0] for e in node_eta]
            strip_side_terms_all.extend((node_terms[j], node_terms[j + 1]) for j in range(len(node_eta) - 1))
        else:
            strip_side_terms_all.extend(([], []) for _ in range(len(node_eta) - 1))
        airfoils.extend(afs)
        # The untwisted reference line has the chord along x: its spanwise
        # direction in the y-z plane is the dihedral direction of the strip.
        span_def = np.diff(np.asarray(edge_ref, dtype=float), axis=0)
        span_def[:, 0] = 0.0
        span_def /= np.maximum(np.linalg.norm(span_def, axis=1, keepdims=True), 1e-300)
        surfaces.append(SurfaceSlice(
            name=surf.name, index=s_idx, is_symmetric=surf.is_symmetric,
            strips=slice(strip_offset, strip_offset + n_s),
            edge_le=edge_le, edge_te=edge_te, edge_chord=edge_chord, edge_twist=edge_twist,
            strip_span_def=span_def,
            edge_le_chord=edge_le.copy(),
            edge_te_chord=edge_te.copy(),
        ))
        strip_offset += n_s

    cat = {k: np.concatenate(v) if k not in ("le_l", "le_r", "te_l", "te_r") else np.vstack(v)
           for k, v in strip_parts.items()}
    le_l, le_r, te_l, te_r = cat["le_l"], cat["le_r"], cat["te_l"], cat["te_r"]
    chord = cat["chord"]

    qc_l = le_l + 0.25 * (te_l - le_l)
    qc_r = le_r + 0.25 * (te_r - le_r)
    dl = qc_r - qc_l
    qc_mid = 0.5 * (qc_l + qc_r)
    width = np.sqrt(dl[:, 1] ** 2 + dl[:, 2] ** 2)
    width = np.maximum(width, 1e-12)
    area = chord * width

    cdir = (te_l - le_l) / np.maximum(np.linalg.norm(te_l - le_l, axis=1, keepdims=True), 1e-300)
    cdir += (te_r - le_r) / np.maximum(np.linalg.norm(te_r - le_r, axis=1, keepdims=True), 1e-300)
    cdir /= np.maximum(np.linalg.norm(cdir, axis=1, keepdims=True), 1e-300)
    s_hat = dl / np.maximum(np.linalg.norm(dl, axis=1, keepdims=True), 1e-300)
    normal = cross3(cdir, s_hat)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-300)
    # Zero-lift line: turn the normal nose-up by -alpha_L0 about the span axis in
    # the y-z plane, the same axis as the twist. The section data are defined in
    # streamwise sections, so camber then acts like the same incidence also on
    # a swept surface (a turn about the swept strip axis would give
    # alpha_L0 * cos(sweep)).
    span_yz = dl.copy()
    span_yz[:, 0] = 0.0
    span_yz /= np.maximum(np.linalg.norm(span_yz, axis=1, keepdims=True), 1e-300)
    normal_bc_strip = _rotate(normal, span_yz, -cat["aL0"])

    # --- panels ---------------------------------------------------------------
    n_strips = chord.shape[0]
    claf_raw = cat["a0"] / (2.0 * np.pi)
    claf = np.clip(claf_raw, CLAF_MIN, CLAF_MAX)
    if collocation != "llt" and np.any(claf != claf_raw):
        warnings.warn(
            f"Section lift slope a0 outside {CLAF_MIN:g} to {CLAF_MAX:g} times 2 pi on some strips "
            f"(a0/(2 pi) from {float(np.min(claf_raw)):.3g} to {float(np.max(claf_raw)):.3g}): the vortex-lattice "
            "control point is limited to that range, so the 2D lift slope of the lattice is not a0 on those "
            "strips and the lattice and the lifting line do not agree there.",
            RuntimeWarning,
            stacklevel=2,
        )
    pa, pb, pcp, pstrip = [], [], [], []
    frac = cat["frac"][:, None]
    strip_xi_arr = np.array(strip_xi_all)  # shape (n_strips, n_chord + 1)
    for k in range(n_chord):
        x0 = strip_xi_arr[:, k, None]
        x1 = strip_xi_arr[:, k + 1, None]
        dx = x1 - x0
        f_bound = x0 + 0.25 * dx
        pa.append(le_l + f_bound * (te_l - le_l))
        pb.append(le_r + f_bound * (te_r - le_r))
        if collocation == "llt":
            pcp.append(pa[-1] + frac * (pb[-1] - pa[-1]))
        else:
            f_cp = x0 + 0.25 * dx + 0.5 * dx * claf[:, None]
            p_l = le_l + f_cp * (te_l - le_l)
            p_r = le_r + f_cp * (te_r - le_r)
            pcp.append(p_l + frac * (p_r - p_l))
        pstrip.append(np.arange(n_strips))
    # Order panels strip by strip, chordwise inside a strip.
    a = np.stack(pa, axis=1).reshape(-1, 3)
    b = np.stack(pb, axis=1).reshape(-1, 3)
    cp = np.stack(pcp, axis=1).reshape(-1, 3)
    panel_strip = np.repeat(np.arange(n_strips), n_chord)
    panel_k = np.tile(np.arange(n_chord), n_strips)
    d_xi = np.diff(strip_xi_arr, axis=1)  # shape (n_strips, n_chord)
    panel_length = (d_xi * chord[:, None]).reshape(-1)
    strip_mirror = cat["mirror"].astype(int)
    panel_mirror = np.where(
        strip_mirror[panel_strip] >= 0,
        strip_mirror[panel_strip] * n_chord + panel_k,
        -1,
    )

    normal_bc = normal_bc_strip[panel_strip].copy()
    geometric_camber = np.zeros(n_strips, dtype=bool)
    # Strips whose bound points have a non-zero mean-line offset: their legs
    # follow the curved mean surface (see _leg_nodes). A zero mean line
    # (a symmetric section) keeps the straight legs, bit for bit.
    curved = np.zeros(n_strips, dtype=bool)
    # Mean-surface node of the leg of each panel between its bound point and
    # the bound point of the next panel of its strip [m] (see _leg_nodes).
    mid_a = mid_b = None
    camber_frames, offset_warnings = _camber_edge_frames(surfaces, mean_surfaces, aircraft)
    # Keep the old arithmetic for sections without a mean line and for
    # symmetric mean lines. This preserves the uncambered results in bits.
    for si, (sf, (secs, lines, _eta_e)) in enumerate(zip(surfaces, mean_surfaces)):
        active = all(line is not None for line in lines)
        if camber_frames is None or (not active and not camber_frames[si]["joined"]):
            continue
        idx = np.arange(sf.strips.start, sf.strips.stop)
        known = np.full(len(idx), active)
        geometric_camber[idx] = known
        # The strip chord remains the reference geometry. Its normals and
        # chord lengths carry the mean-line offset after node displacements.
        frame = camber_frames[si]
        edge_c, edge_n, eta_full = frame["chord"], frame["normal"], frame["eta"]
        p_idx = (idx[:, None] * n_chord + np.arange(n_chord)).ravel()
        x0, dx = strip_xi_arr[idx, :-1], np.diff(strip_xi_arr[idx], axis=1)
        xb = x0 + 0.25 * dx
        xc = xb + 0.5 * dx * claf[idx, None]
        z_l = _mean_line_at(secs, lines, np.abs(eta_full[:-1, None]), xb)
        z_r = _mean_line_at(secs, lines, np.abs(eta_full[1:, None]), xb)
        z_c = _mean_line_at(secs, lines, np.abs(cat["eta"][idx, None]), xc)
        slope = _mean_line_at(secs, lines, np.abs(cat["eta"][idx, None]), xc, derivative=True)
        off_l = edge_c[:-1, None, None] * z_l[..., None] * edge_n[:-1, None, :]
        off_r = edge_c[1:, None, None] * z_r[..., None] * edge_n[1:, None, :]
        for e in frame["joined"]:
            if e < len(idx):
                off_l[e] = _joined_camber_offset(camber_frames, si, e, xb[e])
            if e > 0:
                off_r[e - 1] = _joined_camber_offset(camber_frames, si, e, xb[e - 1])
        curved[idx] = known & (np.any(off_l != 0.0, axis=(1, 2)) | np.any(off_r != 0.0, axis=(1, 2)))
        if n_chord > 1:
            # The node at the middle chord fraction between two bound points,
            # on the mean line of the strip edge.
            xm = 0.5 * (xb[:, :-1] + xb[:, 1:])
            z_ml = _mean_line_at(secs, lines, np.abs(eta_full[:-1, None]), xm)
            z_mr = _mean_line_at(secs, lines, np.abs(eta_full[1:, None]), xm)
            off_ml = edge_c[:-1, None, None] * z_ml[..., None] * edge_n[:-1, None, :]
            off_mr = edge_c[1:, None, None] * z_mr[..., None] * edge_n[1:, None, :]
            for e in frame["joined"]:
                if e < len(idx):
                    off_ml[e] = _joined_camber_offset(camber_frames, si, e, xm[e])
                if e > 0:
                    off_mr[e - 1] = _joined_camber_offset(camber_frames, si, e, xm[e - 1])
            if mid_a is None:
                mid_a = np.zeros_like(a)
                mid_b = np.zeros_like(b)
            m_idx = (idx[:, None] * n_chord + np.arange(n_chord - 1)).ravel()
            mid_a[m_idx] = (le_l[idx, None] + xm[..., None] * (te_l[idx] - le_l[idx])[:, None] + off_ml).reshape(-1, 3)
            mid_b[m_idx] = (le_r[idx, None] + xm[..., None] * (te_r[idx] - le_r[idx])[:, None] + off_mr).reshape(-1, 3)
        a[p_idx] += off_l.reshape(-1, 3)
        b[p_idx] += off_r.reshape(-1, 3)
        f_mid = cat["frac"][idx, None]
        edge_vec = frame["te"] - frame["le"]
        cp_chord = np.linalg.norm((1.0 - f_mid) * edge_vec[:-1] + f_mid * edge_vec[1:], axis=1)
        cp[p_idx] += (cp_chord[:, None, None] * z_c[..., None] * normal[idx, None, :]).reshape(-1, 3)
        tangent = cdir[idx, None, :] + slope[..., None] * normal[idx, None, :]
        nb = cross3(tangent.reshape(-1, 3), np.repeat(s_hat[idx], n_chord, axis=0))
        nb /= np.maximum(np.linalg.norm(nb, axis=1, keepdims=True), 1e-300)
        use = np.repeat(known, n_chord)
        normal_bc[p_idx[use]] = nb[use]
        normal_bc_strip[idx[known]] = normal[idx[known]]
        z_le = _mean_line_at(secs, lines, np.abs(eta_full), np.zeros_like(eta_full))
        z_te = _mean_line_at(secs, lines, np.abs(eta_full), np.ones_like(eta_full))
        off_le, off_te = edge_c[:, None] * z_le[:, None] * edge_n, edge_c[:, None] * z_te[:, None] * edge_n
        for e in frame["joined"]:
            off_le[e] = _joined_camber_offset(camber_frames, si, e, np.array(0.0))
            off_te[e] = _joined_camber_offset(camber_frames, si, e, np.array(1.0))
        sf.edge_le = sf.edge_le + off_le
        sf.edge_te = sf.edge_te + off_te
        le_l[idx], le_r[idx] = sf.edge_le[:-1], sf.edge_le[1:]
        te_l[idx], te_r[idx] = sf.edge_te[:-1], sf.edge_te[1:]
    normal_bc_clean = None
    flap_hinges = {}
    leg_kinks = {}
    flap_field = None
    # Strip edges (n_strips, 2) of the flap strips whose hinge point is not a
    # leg node (see _leg_nodes); None without chordwise flaps.
    hinge_off = None
    # Lattice before the flap turn: the leg-force rules come from it.
    undeflected = None
    if is_chordwise:
        normal_bc_clean = normal_bc.copy()
        undeflected = (a.copy(), b.copy(), te_l.copy(), te_r.copy(), cp.copy())
        mid_undeflected = None if mid_a is None else (mid_a.copy(), mid_b.copy())
        strip_surf = cat["surface"].astype(int)

        def hinge_point(s: int, side: int, h: float) -> np.ndarray:
            # Hinge point [m] at fraction h of the left (0) or right (1) edge
            # of strip s, on the mean line when the surface has one.
            # The trailing edges before the turn (the loop turns te_l, te_r).
            le, te = (le_l, undeflected[2]) if side == 0 else (le_r, undeflected[3])
            if camber_frames is None:
                return le[s] + h * (te[s] - le[s])
            si = int(strip_surf[s])
            frame = camber_frames[si]
            e = s - surfaces[si].strips.start + side
            return (frame["le"][e] + h * (frame["te"][e] - frame["le"][e])
                    + _joined_camber_offset(camber_frames, si, e, np.array(h)))

        def unit(v: np.ndarray) -> np.ndarray:
            n = float(np.linalg.norm(v))
            # Two hinge points at the same place only on a degenerate strip
            # (zero width: the solvers refuse it, build_lattice accepts it).
            return v / n if n > 1e-300 else np.array([0.0, 1.0, 0.0])

        angle = np.zeros((n_strips, 2))
        hinge_of = np.full(n_strips, np.nan)
        for s in range(n_strips):
            if strip_flap_hinge_all[s] is not None:
                hinge_of[s] = strip_flap_hinge_all[s]
        # Neighbours of each edge node of the flap strips: ``partner`` is
        # the (strip, side) of the other strip at the same node, ``far`` the
        # (strip, side) of the next node beyond it (for the hinge axis); -1
        # where there is none. Inside a surface they are the neighbour
        # strip. An end node of a surface takes them from a coincident end
        # node of another surface with a flap strip of the same hinge (a
        # surface joined to its mirror copy at the root): the two surfaces
        # then turn the node about the same line, as one symmetric surface.
        partner = np.full((n_strips, 2, 2), -1, dtype=np.int64)
        far = np.full((n_strips, 2, 2), -1, dtype=np.int64)
        ends = []
        for si, sl in enumerate(surfaces):
            if sl.strips.stop > sl.strips.start:
                ends += [(sl.strips.start, 0, si), (sl.strips.stop - 1, 1, si)]
        for s in range(n_strips):
            if np.isnan(hinge_of[s]):
                continue
            if s > 0 and strip_surf[s - 1] == strip_surf[s] and np.isfinite(hinge_of[s - 1]):
                partner[s, 0] = (s - 1, 1)
                far[s, 0] = (s - 1, 0)
            if s + 1 < n_strips and strip_surf[s + 1] == strip_surf[s] and np.isfinite(hinge_of[s + 1]):
                partner[s, 1] = (s + 1, 0)
                far[s, 1] = (s + 1, 1)
        def coincident(s: int, side: int, q: int, jq: int) -> bool:
            # True when edge jq of strip q is at the place of edge side of
            # strip s (leading and trailing edge before the flap turn).
            tol = 1e-9 * max(1.0, float(chord[s]))
            return bool(np.max(np.abs((le_l, le_r)[jq][q] - (le_l, le_r)[side][s])) <= tol
                        and np.max(np.abs(undeflected[2 + jq][q] - undeflected[2 + side][s])) <= tol)

        for s, side, si in ends:
            if np.isnan(hinge_of[s]) or partner[s, side, 0] >= 0:
                continue
            for q, jq, sq in ends:
                if sq == si or hinge_of[q] != hinge_of[s]:
                    continue
                if coincident(s, side, q, jq):
                    partner[s, side] = (q, jq)
                    far[s, side] = (q, 1 - jq)
                    break
        coef: dict[str, np.ndarray] = {}
        # Turn of the two edges of each flap strip: (axis, angle) left and right.
        edge_turn: dict[int, tuple] = {}
        for s in range(n_strips):
            h = strip_flap_hinge_all[s]
            if h is None:
                continue
            H_l = hinge_point(s, 0, h)
            H_r = hinge_point(s, 1, h)
            flap_hinges[s] = (H_l.copy(), H_r.copy())
            # The hinge axis of an edge node is the central difference of
            # the hinge points of its neighbour nodes, so the two strips at
            # a node turn it about the same line: the deflected surface has
            # no gap between strips (also at the root of a surface with
            # dihedral).
            fl, fr = far[s, 0], far[s, 1]
            ax_l = unit(H_r - (hinge_point(int(fl[0]), int(fl[1]), h) if fl[0] >= 0 else H_l))
            ax_r = unit((hinge_point(int(fr[0]), int(fr[1]), h) if fr[0] >= 0 else H_r) - H_l)
            for side, terms in enumerate(strip_side_terms_all[s]):
                for c, f in terms:
                    angle[s, side] += f * c.deflection
                    coef.setdefault(c.name, np.zeros((n_strips, 2)))[s, side] += f
            d_l, d_r = angle[s]
            edge_turn[s] = (ax_l, d_l, ax_r, d_r, h)
            f_cp = float(cat["frac"][s])
            H_f = H_l + f_cp * (H_r - H_l)
            ax_f = unit(ax_l + f_cp * (ax_r - ax_l))
            d_f = d_l + f_cp * (d_r - d_l)
            for k in range(n_chord):
                x0_k = strip_xi_arr[s, k]
                dx_k = strip_xi_arr[s, k + 1] - x0_k
                xb_k = x0_k + 0.25 * dx_k
                xc_k = xb_k + 0.5 * dx_k * claf[s]
                p_idx = s * n_chord + k
                for points, fraction, H, ax, dd in ((a, xb_k, H_l, ax_l, d_l), (b, xb_k, H_r, ax_r, d_r),
                                                    (cp, xc_k, H_f, ax_f, d_f)):
                    if fraction > h and dd != 0.0:
                        points[p_idx:p_idx + 1] = _rotate_flap_points(points[p_idx:p_idx + 1], H, ax, dd)
                if mid_a is not None and k < n_chord - 1:
                    # The mean-surface node aft of the hinge turns with the flap.
                    x1_k = strip_xi_arr[s, k + 1]
                    xm_k = 0.5 * (xb_k + x1_k + 0.25 * (strip_xi_arr[s, k + 2] - x1_k))
                    for points, H, ax, dd in ((mid_a, H_l, ax_l, d_l), (mid_b, H_r, ax_r, d_r)):
                        if xm_k > h and dd != 0.0:
                            points[p_idx:p_idx + 1] = _rotate_flap_points(points[p_idx:p_idx + 1], H, ax, dd)
                if x0_k >= h - 1e-12 and d_f != 0.0:
                    normal_bc[p_idx] = _rotate(normal_bc[p_idx:p_idx + 1], ax_f[None, :], d_f)[0]
                # The chordwise legs of the panels ahead of the hinge follow
                # the deflected mean surface: bound point, hinge point of the
                # strip edge, deflected trailing edge. The node is there also
                # at zero deflection (on the straight leg), so the loads and
                # their derivatives are continuous in the deflection.
                if xb_k <= h:
                    leg_kinks[p_idx] = (H_l, H_r)
            # The surface edge arrays keep the reference geometry used by
            # the core-group rule.
            if d_l != 0.0:
                te_l[s:s + 1] = _rotate_flap_points(te_l[s:s + 1], H_l, ax_l, d_l)
            if d_r != 0.0:
                te_r[s:s + 1] = _rotate_flap_points(te_r[s:s + 1], H_r, ax_r, d_r)
        if flap_hinges:
            # Per flap strip: the deflection at the span fraction of its
            # control point [rad] and the hinge of its control as defined.
            frac_s = np.asarray(cat["frac"], dtype=float)
            # An edge that a flap strip shares with a strip of a different
            # hinge has a different hinge point on the legs of the other
            # strip: a strip without a hinge (the outer edge of the join at
            # a side edge of a control) or a strip of a control with another
            # hinge fraction. The other strip can be the neighbour strip of
            # the same surface or the end strip of another surface whose end
            # edge is at the same place (a joined surface). On a leg chain
            # the legs of the two strips must pass through the same nodes,
            # so the hinge point is not a node of either strip at that edge:
            # the legs of both use the mean-surface node. On a symmetric
            # section the hinge point and the mean-surface node are on one
            # straight line and the legs coincide also with the hinge node.
            hinge_off = np.zeros((n_strips, 2), dtype=bool)
            for s in np.flatnonzero(np.isfinite(hinge_of)):
                for side, q in ((0, s - 1), (1, s + 1)):
                    if 0 <= q < n_strips and strip_surf[q] == strip_surf[s] and hinge_of[q] != hinge_of[s]:
                        hinge_off[s, side] = True
            for s, side, si in ends:
                if np.isnan(hinge_of[s]):
                    continue
                for q, jq, sq in ends:
                    if sq != si and hinge_of[q] != hinge_of[s] and coincident(s, side, q, jq):
                        hinge_off[s, side] = True
            flap_field = {"hinge": hinge_of, "angle": angle, "coef": coef, "partner": partner, "far": far,
                          "strip_angle": angle[:, 0] + frac_s * (angle[:, 1] - angle[:, 0]),
                          "hinge_x_c": np.asarray(strip_flap_nominal_all, dtype=float),
                          "hinge_off": hinge_off}

    a_kink = b_kink = None
    leg_next = None
    chain_strips = np.flatnonzero(curved) if n_chord > 1 else np.zeros(0, dtype=np.int64)
    if leg_kinks or chain_strips.size:
        a_kink, b_kink, leg_next = _leg_nodes(a, b, te_l[panel_strip], te_r[panel_strip], leg_kinks,
                                              chain_strips, n_chord, mid_a, mid_b, hinge_off)

    lattice = VortexLattice(
        collocation=collocation,
        n_chord=n_chord,
        a=a,
        b=b,
        a_te=te_l[panel_strip],
        b_te=te_r[panel_strip],
        cp=cp,
        normal_bc=normal_bc,
        panel_strip=panel_strip,
        panel_mirror=panel_mirror,
        panel_length=panel_length,
        strip_surface=cat["surface"].astype(int),
        le_left=le_l,
        le_right=le_r,
        te_left=te_l,
        te_right=te_r,
        qc_mid=qc_mid,
        dl=dl,
        chord=chord,
        twist=cat["twist"],
        width=width,
        area=area,
        chord_dir=cdir,
        normal=normal,
        normal_bc_strip=normal_bc_strip,
        a0=cat["a0"],
        alpha_L0=cat["aL0"],
        Cd0=cat["cd0"],
        Cm0=cat["cm0"],
        airfoils=airfoils,
        strip_mirror=strip_mirror,
        strip_is_right=cat["is_right"].astype(bool),
        eta=cat["eta"],
        cp_frac=cat["frac"],
        surfaces=surfaces,
        strip_core_group=core_groups(surfaces)[cat["surface"].astype(int)],
        join_warnings=near_miss_warnings(surfaces) + camber_warnings + offset_warnings,
        mesh_warnings=mesh_warnings,
        control_info=control_info,
        geometric_camber=geometric_camber,
        a_kink=a_kink,
        b_kink=b_kink,
        leg_next=leg_next,
        normal_bc_clean=(
            normal_bc_clean if normal_bc_clean is not None and np.any(normal_bc_clean != normal_bc) else None
        ),
        camber_fallback=camber_fallback,
        has_camber_offset=any(line is not None and np.any(line.z != 0.0)
                              for _, lines, _ in mean_surfaces for line in lines),
        camber_frames=camber_frames,
    )
    if camber_frames is not None:
        lattice.geom_cache["camber_frames"] = camber_frames
        lattice.geom_cache["mean_surfaces"] = mean_surfaces
    lattice.geom_cache["flap_hinges"] = flap_hinges
    if flap_field is not None:
        lattice.geom_cache["flap_field"] = flap_field
        # Bound points and control points before the flap turn (the linear
        # 2D section model of the viscous coupling uses them).
        lattice.geom_cache["flap_points_clean"] = (undeflected[0], undeflected[1], undeflected[4])
    mark_joined_edges(surfaces)
    check_overlaps(lattice)
    if collocation == "vlm":
        # The leg-force rules depend only on the mesh; their warnings join the
        # notes of every result of this lattice.
        from ventorum.aero.loads import leg_force_rules

        if flap_field is not None:
            # The rules of a lattice with chordwise controls come from the
            # lattice before the flap turn (the same topology). The legs of
            # two strips that turn about different lines separate a little
            # with the deflection; a rule from the deflected points would
            # change when the gap passes the join tolerance, and the loads
            # would jump.
            a0, b0, te_l0, te_r0, _ = undeflected
            mid0 = (None, None) if mid_undeflected is None else mid_undeflected
            a_k0, b_k0, next0 = _leg_nodes(a0, b0, te_l0[panel_strip], te_r0[panel_strip], leg_kinks,
                                           chain_strips, n_chord, *mid0, hinge_off)
            flat = replace(lattice, a=a0, b=b0, a_te=te_l0[panel_strip], b_te=te_r0[panel_strip],
                           a_kink=a_k0, b_kink=b_k0, leg_next=next0, te_left=te_l0, te_right=te_r0,
                           geom_cache={})
            normal, scale, kind = leg_force_rules(flat)
            # The rule normal of a leg on the flap (aft of the hinge) turns
            # with its edge node: only the velocity tangent to the deflected
            # surface acts there. The two strips at a node use the same turn.
            normal = normal.copy()
            n_p = panel_strip.size
            for s, (ax_l, d_l, ax_r, d_r, h) in edge_turn.items():
                xb_s = [strip_xi_arr[s, k] + 0.25 * (strip_xi_arr[s, k + 1] - strip_xi_arr[s, k])
                        for k in range(n_chord)]
                for k in range(n_chord):
                    p_idx = s * n_chord + k
                    chained = bool(curved[s]) and k < n_chord - 1
                    # Rows 0, 1 are the forward segments of the legs (bound
                    # point to the first node), rows 2, 3 the aft segments
                    # (first node to the end of the leg), see leg_segments
                    # and _leg_nodes. A segment turns when it lies aft of
                    # the hinge. Aft panel: the forward segments, and on a
                    # chain also the aft segments (mean-surface node to the
                    # next bound point; without a chain the aft segments
                    # have zero length). Panel ahead of the hinge: the aft
                    # segments only when the first node is the hinge point.
                    # On a chain that is only the last panel ahead of the
                    # hinge: the panels ahead of it have a mean-surface node
                    # ahead of the hinge, and their legs stay.
                    if xb_s[k] > h:
                        rows = ((0, ax_l, d_l), (1, ax_r, d_r)) + (((2, ax_l, d_l), (3, ax_r, d_r)) if chained else ())
                    elif chained and xb_s[k + 1] <= h:
                        rows = ()
                    else:
                        rows = ((2, ax_l, d_l), (3, ax_r, d_r))
                    for r, ax, dd in rows:
                        if dd != 0.0:
                            i = r * n_p + p_idx
                            normal[i] = _rotate(normal[i:i + 1], ax[None, :], dd)[0]
            lattice.geom_cache["leg_force_rules"] = (normal, scale, kind)
        else:
            leg_force_rules(lattice)
    return lattice


def _leg_nodes(
    a: np.ndarray,
    b: np.ndarray,
    a_te: np.ndarray,
    b_te: np.ndarray,
    hinge_nodes: dict,
    chain_strips: np.ndarray,
    n_chord: int,
    mid_a: np.ndarray | None = None,
    mid_b: np.ndarray | None = None,
    hinge_off: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return the first leg nodes and the chain flags of the panels (see ``VortexLattice.leg_next``).

    One mechanism gives the hinge nodes of the chordwise flaps and the nodes
    of the mean surface: the legs of a panel pass through the nodes of its
    strip edge, in chordwise order, to the trailing edge. On a strip with
    geometric camber the nodes are the bound points of the panels aft of it,
    one mean-surface node between each two bound points (at the middle chord
    fraction) and the hinge point of a flap; on a strip without a mean line
    only the hinge point. Where the hinge point lies between two bound
    points, it is the node between them (one node per segment pair). The
    two strips at an edge must have the same nodes there, so that their
    legs coincide. Thus on a leg chain the hinge point is not a node at an
    edge that the flap strip shares with a strip of a different hinge (no
    hinge or another hinge fraction, on the same surface or on a joined
    surface; *hinge_off*); the mean-surface node is the node there, as on
    the other strip.

    Parameters
    ----------
    a, b : numpy.ndarray
        Bound points of the panels [m], shape (n_panels, 3), after the flap
        turn.
    a_te, b_te : numpy.ndarray
        Trailing-edge points of the panels [m], shape (n_panels, 3).
    hinge_nodes : dict
        Panel index to the hinge points ``(H_l, H_r)`` [m] of its strip
        edges, for the panels whose bound points are at or ahead of the
        hinge.
    chain_strips : numpy.ndarray
        Strips with geometric camber.
    n_chord : int
        Chordwise panels per strip.
    mid_a, mid_b : numpy.ndarray or None
        Mean-surface nodes [m], shape (n_panels, 3), between the bound points
        of each panel and of the next panel of its strip, after the flap
        turn; None to use the bound point of the next panel as the node.
    hinge_off : numpy.ndarray or None
        Strip edges whose hinge point is not a node of a leg chain, shape
        (n_strips, 2) bool (left and right edge); None for none.

    Returns
    -------
    a_kink, b_kink : numpy.ndarray
        First node of each leg after the bound point [m], shape (n_panels, 3).
    leg_next : numpy.ndarray or None
        Chain flags, shape (n_panels,), or None without chains.
    """
    a_kink = a_te.copy()
    b_kink = b_te.copy()
    for p_idx, (H_l, H_r) in hinge_nodes.items():
        a_kink[p_idx] = H_l
        b_kink[p_idx] = H_r
    if chain_strips.size == 0:
        return a_kink, b_kink, None
    leg_next = np.zeros(a.shape[0], dtype=bool)
    for s in chain_strips:
        for k in range(n_chord - 1):
            p = int(s) * n_chord + k
            leg_next[p] = True
            # The hinge point is a node of this leg only if it lies between
            # the bound points of this panel and of the next one.
            hinge = p in hinge_nodes and p + 1 not in hinge_nodes
            off_a = off_b = not hinge
            if hinge and hinge_off is not None:
                off_a, off_b = bool(hinge_off[s, 0]), bool(hinge_off[s, 1])
            if off_a:
                a_kink[p] = a[p + 1] if mid_a is None else mid_a[p]
            if off_b:
                b_kink[p] = b[p + 1] if mid_b is None else mid_b[p]
    return a_kink, b_kink, leg_next


# ═══════════════════════════════════════════════════════════════════════════════
# Strip-level view (for plots and the older DiscretizedSurface API)
# ═══════════════════════════════════════════════════════════════════════════════

def lattice_to_discretized(
    lattice: VortexLattice,
    point_map: Any = None,
    vector_map: Any = None,
    half_mesh: bool = False,
) -> list:
    """One :class:`DiscretizedSurface` per surface, at strip level.

    Parameters
    ----------
    point_map, vector_map : callables or None
        Optional maps applied to points ``(n, 3) -> (n, 3)`` and to direction
        vectors (for example a change of frame for plots).
    half_mesh : bool
        Keep only the right half of symmetric surfaces.
    """
    from ventorum.core.datatypes import DiscretizedSurface

    pm = point_map or (lambda p: p)
    vm = vector_map or (lambda v: v)
    out = []
    for surf in lattice.surfaces:
        sl = surf.strips
        idx = np.arange(sl.start, sl.stop)
        edge_le, edge_te = surf.edge_le, surf.edge_te
        if half_mesh and surf.is_symmetric:
            keep = lattice.strip_is_right[idx]
            first = int(np.argmax(keep))
            idx = idx[keep]
            edge_le = edge_le[first:]
            edge_te = edge_te[first:]
        nodes_qc = edge_le + 0.25 * (edge_te - edge_le)
        frac = lattice.cp_frac[idx][:, None]
        cp_l = lattice.le_left[idx] + 0.75 * (lattice.te_left[idx] - lattice.le_left[idx])
        cp_r = lattice.le_right[idx] + 0.75 * (lattice.te_right[idx] - lattice.le_right[idx])
        qc_l = lattice.le_left[idx] + 0.25 * (lattice.te_left[idx] - lattice.le_left[idx])
        qc_r = lattice.le_right[idx] + 0.25 * (lattice.te_right[idx] - lattice.le_right[idx])
        afs = [lattice.airfoils[i] for i in idx]
        groups: dict[int, tuple[Any, list[int]]] = {}
        for k, af in enumerate(afs):
            groups.setdefault(id(af), (af, []))[1].append(k)
        airfoil_groups = [(af, np.asarray(ks, dtype=int)) for af, ks in groups.values()]
        has_profile = any(
            (not isinstance(af, LinearAirfoil)) or af.Cd0 > 0.0 for af in afs
        )
        out.append(DiscretizedSurface(
            nodes_qc=pm(nodes_qc),
            control_points=pm(cp_l + frac * (cp_r - cp_l)),
            normals=vm(lattice.normal[idx]),
            y_panels=lattice.cp[idx * lattice.n_chord][:, 1] if not point_map else pm(qc_l + frac * (qc_r - qc_l))[:, 1],
            dy_panels=lattice.width[idx].copy(),
            chords=lattice.chord[idx].copy(),
            twists=lattice.twist[idx].copy(),
            airfoils=afs,
            surface_name=surf.name,
            surface_index=surf.index,
            panel_centers_qc=pm(qc_l + frac * (qc_r - qc_l)),
            has_profile_drag=bool(has_profile),
            airfoil_groups=airfoil_groups,
            is_half_mesh=bool(half_mesh and surf.is_symmetric),
            nodes_le=pm(edge_le),
            nodes_te=pm(edge_te),
        ))
    return out
