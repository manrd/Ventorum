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

Camber is modelled by the zero-lift line: the boundary-condition normal is
the geometric normal turned nose-up by ``-alpha_L0``.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ventorum.core.datatypes import (
    Aircraft,
    AirfoilType,
    LiftingSurface,
    LinearAirfoil,
    SolverSettings,
    TabulatedAirfoil,
    WingSection,
)
from ventorum.geometry.discretization import compute_surface_n_panels, get_spacing
from ventorum.utils.vec import cross3

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
CROSS_SURFACE_CORE_FRACTION = 0.5  # owner decision D-03 (2026-10-03), from the T-0013 core-size study


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


@dataclass
class VortexLattice:
    """Panels and strips of an aircraft, in body axes.

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
    # Warnings about surface edges that are near but not joined (see ``near_miss_warnings``).
    join_warnings: list[str] = field(default_factory=list)
    # Cache of the parts of the vortex system that do not change with the
    # angle of attack (see ventorum.aero.vortex). None means no cache. A
    # sweep sets it to a dict; the solvers use it only out of ground effect.
    kernel_cache: dict | None = field(default=None, repr=False, compare=False)
    # Results that depend only on the lattice (airfoil groups, symmetry
    # geometry, Trefftz strip data). Shared by the shallow copies that the
    # lattice cache hands out, so they are computed once per geometry.
    geom_cache: dict = field(default_factory=dict, repr=False, compare=False)

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
        """Leading- and trailing-edge nodes of every strip (for clearance checks)."""
        pts = [s.edge_le for s in self.surfaces] + [s.edge_te for s in self.surfaces]
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


def _relate_surfaces(si: SurfaceSlice, sj: SurfaceSlice) -> tuple[bool, tuple[float, float] | None]:
    """Compare the edges of two surfaces.

    Return ``(joined, near_miss)``. ``joined`` is True when an edge segment
    (the line from the leading-edge node to the trailing-edge node of one
    section) of *si* lies on an edge segment of *sj* within the tolerance,
    over at least ``JOIN_MIN_OVERLAP`` of the shorter segment. The tolerance
    is ``JOIN_TOL_STRIP_FRACTION`` times the smaller local strip width, kept
    between ``JOIN_TOL_CHORD_FRACTION`` times the longer segment and
    ``JOIN_TOL_CHORD_MAX_FRACTION`` times the shorter segment.
    ``near_miss`` is None, or ``(gap, local_chord)`` [m] of the closest pair
    of edges that are nearly parallel and overlap, but are not joined.
    """
    wi, wj = _edge_strip_widths(si), _edge_strip_widths(sj)
    best: tuple[float, float] | None = None
    for e in range(si.edge_le.shape[0]):
        a0, a1 = si.edge_le[e], si.edge_te[e]
        la = float(np.linalg.norm(a1 - a0))
        if la < 1e-12:
            continue
        ua = (a1 - a0) / la
        for f in range(sj.edge_le.shape[0]):
            b0, b1 = sj.edge_le[f], sj.edge_te[f]
            lb = float(np.linalg.norm(b1 - b0))
            if lb < 1e-12:
                continue
            ub = (b1 - b0) / lb
            gap = max(
                float(np.max(_perpendicular_distance(np.vstack([b0, b1]), a0, ua))),
                float(np.max(_perpendicular_distance(np.vstack([a0, a1]), b0, ub))),
            )
            t = np.sort([(b0 - a0) @ ua, (b1 - a0) @ ua])
            overlap = max(0.0, min(la, float(t[1])) - max(0.0, float(t[0]))) / min(la, lb)
            if overlap < JOIN_MIN_OVERLAP:
                continue
            tol = float(np.clip(
                JOIN_TOL_STRIP_FRACTION * min(wi[e], wj[f]),
                JOIN_TOL_CHORD_FRACTION * max(la, lb), JOIN_TOL_CHORD_MAX_FRACTION * min(la, lb),
            ))
            if gap <= tol:
                return True, None
            angle = np.degrees(np.arccos(min(1.0, abs(float(ua @ ub)))))
            chord = min(la, lb)
            if angle < NEAR_MISS_MAX_ANGLE_DEG and gap <= NEAR_MISS_MAX_GAP_CHORDS * chord:
                if best is None or gap < best[0]:
                    best = (gap, chord)
    return False, best


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
    a tolerance, over at least half of the shorter segment (see
    :func:`_relate_surfaces`). The tolerance is the smaller local strip width,
    with limits. The groups are the connected sets of this relation.
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


def near_miss_warnings(surfaces: list[SurfaceSlice]) -> list[str]:
    """Return one warning for each pair of surfaces whose edges are near but not joined.

    Two edges are a near miss when they are nearly parallel (angle below
    ``NEAR_MISS_MAX_ANGLE_DEG``), overlap, and are further apart than the
    join tolerance but not more than ``NEAR_MISS_MAX_GAP_CHORDS`` local
    chords. Surfaces that are in one group through a third surface give no
    warning.
    """
    groups = core_groups(surfaces)
    out = []
    for i in range(len(surfaces)):
        for j in range(i + 1, len(surfaces)):
            if groups[i] == groups[j]:
                continue
            _, near = _relate_surfaces(surfaces[i], surfaces[j])
            if near is not None:
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
    """
    edges, mids = get_spacing(spacing, n_panels, surf=surf)
    edges = np.asarray(edges, dtype=float).copy()
    mids = np.asarray(mids, dtype=float).copy()
    edges[0], edges[-1] = 0.0, 1.0
    breaks = _section_breaks(surf)
    if breaks and len(breaks) <= max(1, (len(edges) - 1) // 3):
        moved: set[int] = set()
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
    return edges, mids


def _chordwise_fractions(n_chord: int, chord_spacing: str) -> np.ndarray:
    if chord_spacing in ("cosine", "full-cosine"):
        return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_chord + 1)))
    return np.linspace(0.0, 1.0, n_chord + 1)


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

    ref_semi = max((s.semi_span for s in aircraft.surfaces), default=1.0)

    strip_parts: dict[str, list[Any]] = {k: [] for k in (
        "surface", "le_l", "le_r", "te_l", "te_r", "chord", "twist", "a0", "aL0", "cd0", "cm0",
        "mirror", "is_right", "eta", "frac",
    )}
    airfoils: list[AirfoilType] = []
    surfaces: list[SurfaceSlice] = []
    strip_offset = 0

    for s_idx, surf in enumerate(aircraft.surfaces):
        if getattr(settings, "proportional_panels", False) and surf.n_panels is None:
            n_sp = compute_surface_n_panels(
                surf, base_n_panels=settings.n_panels, reference_semi_span=ref_semi,
                min_panels=getattr(settings, "min_panels", 8),
            )
        else:
            n_sp = surf.n_panels if surf.n_panels is not None else settings.n_panels
        spacing = resolve_spacing(surf.spacing if surf.spacing is not None else settings.spacing, surf, collocation)
        eta_e, eta_mid = _surface_eta(surf, int(n_sp), spacing)
        geo = surface_edge_geometry(surf, eta_e)
        # Fraction of the strip width (root edge to tip edge) where the control point is.
        frac_r = (eta_mid - eta_e[:-1]) / np.maximum(eta_e[1:] - eta_e[:-1], 1e-300)

        blender = _AirfoilBlender(_sorted_sections(surf))
        right_af = [blender.at(float(e)) for e in eta_mid]
        # The linear properties depend only on the airfoil object: compute them
        # once per distinct object (most surfaces share one airfoil).
        props_of: dict[int, tuple[float, float, float, float]] = {}
        for af in right_af:
            if id(af) not in props_of:
                props_of[id(af)] = airfoil_linear_properties(af)
        right_props = np.array([props_of[id(af)] for af in right_af]).reshape(-1, 4)
        n_r = len(eta_mid)

        le_e, te_e = geo["le"], geo["te"]
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
            edge_le = np.vstack([le_left_e, le_e[1:]])
            edge_te = np.vstack([te_left_e, te_e[1:]])
            edge_chord = np.concatenate([geo["chord"][::-1], geo["chord"][1:]])
            edge_twist = np.concatenate([geo["twist"][::-1], geo["twist"][1:]])
            afs = right_af[::-1] + right_af
            props = np.vstack([right_props[::-1], right_props])
            eta_s = np.concatenate([-eta_mid[::-1], eta_mid])
            # Left strip k (k = 0 at the left tip) mirrors right strip n_r - 1 - k.
            local_mirror = np.concatenate([n_r + np.arange(n_r)[::-1], np.arange(n_r)[::-1]])
            is_right = np.concatenate([np.zeros(n_r, bool), np.ones(n_r, bool)])
            # Left strips run from the tip edge to the root edge.
            frac = np.concatenate([(1.0 - frac_r)[::-1], frac_r])
        elif getattr(surf, "mirror_y", False):
            # Mirror image (y -> -y) of the surface; strips run left to right
            # (tip to root), as on the left half of a symmetric surface.
            edge_le = (le_e * mirror)[::-1]
            edge_te = (te_e * mirror)[::-1]
            edge_chord, edge_twist = geo["chord"][::-1], geo["twist"][::-1]
            afs = right_af[::-1]
            props = right_props[::-1]
            eta_s = -eta_mid[::-1]
            local_mirror = np.full(n_r, -1)
            is_right = np.zeros(n_r, bool)
            frac = (1.0 - frac_r)[::-1]
        else:
            edge_le, edge_te = le_e, te_e
            edge_chord, edge_twist = geo["chord"], geo["twist"]
            afs = right_af
            props = right_props
            eta_s = eta_mid
            local_mirror = np.full(n_r, -1)
            is_right = np.ones(n_r, bool)
            frac = frac_r

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
        airfoils.extend(afs)
        surfaces.append(SurfaceSlice(
            name=surf.name, index=s_idx, is_symmetric=surf.is_symmetric,
            strips=slice(strip_offset, strip_offset + n_s),
            edge_le=edge_le, edge_te=edge_te, edge_chord=edge_chord, edge_twist=edge_twist,
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
    claf = np.clip(cat["a0"] / (2.0 * np.pi), 0.2, 2.0)
    pa, pb, pcp, pstrip, _pmirror_k = [], [], [], [], []
    frac = cat["frac"][:, None]
    for k in range(n_chord):
        x0, x1 = xi[k], xi[k + 1]
        dx = x1 - x0
        f_bound = x0 + 0.25 * dx
        pa.append(le_l + f_bound * (te_l - le_l))
        pb.append(le_r + f_bound * (te_r - le_r))
        if collocation == "llt":
            pcp.append(pa[-1] + frac * (pb[-1] - pa[-1]))
        else:
            f_cp = (x0 + 0.25 * dx + 0.5 * dx * claf)[:, None]
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
    panel_length = np.diff(xi)[panel_k] * chord[panel_strip]
    strip_mirror = cat["mirror"].astype(int)
    panel_mirror = np.where(
        strip_mirror[panel_strip] >= 0,
        strip_mirror[panel_strip] * n_chord + panel_k,
        -1,
    )

    lattice = VortexLattice(
        collocation=collocation,
        n_chord=n_chord,
        a=a,
        b=b,
        a_te=te_l[panel_strip],
        b_te=te_r[panel_strip],
        cp=cp,
        normal_bc=normal_bc_strip[panel_strip],
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
        join_warnings=near_miss_warnings(surfaces),
    )
    check_overlaps(lattice)
    return lattice


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
