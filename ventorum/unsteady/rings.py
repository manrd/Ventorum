# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Vortex-ring lattice geometry, ground images and steady equivalence helpers.

This module maps a :class:`ventorum.geometry.lattice.VortexLattice` to a
quadrilateral vortex-ring lattice (:class:`RingLattice`), builds mirror
images for flat-ground effect (:func:`ground_image_rings`), and provides a
private steady verification helper that checks the ring model against the
existing horseshoe vortex-lattice solver.

Ring layout (Katz and Plotkin 2001, chapters 12 and 13):

* Each lattice panel ``p`` on strip ``s`` and chordwise index ``k`` carries
  one four-corner vortex ring ``corners[p, 0..3]`` [m].
* The leading segment ``0 -> 1`` lies on the panel quarter-chord bound
  segment ``(a[p], b[p])``.
* The side segments ``1 -> 2`` and ``3 -> 0`` follow the right and left strip
  edges.
* For interior panels (``k < n_chord - 1``), the trailing corners ``(3, 2)``
  lie on the quarter-chord bound points ``(a[p + 1], b[p + 1])`` of the next
  panel aft in the same strip.
* For the last panel of each strip (``k == n_chord - 1``), the bound segment
  lies at chord fraction ``1 - 0.75 * d_xi_last`` and the trailing edge lies
  at ``1.0``, so ``te - bound`` spans ``0.75`` of the last panel length along
  each strip edge. The default trailing-edge closure places corners ``(3, 2)``
  at ``0.25`` of the last panel length aft of the trailing edge along each
  strip edge: ``te + (0.25 / 0.75) * (te - bound)``. Adjacent strips and
  joined surfaces share their common trailing-edge corner with zero gap.
* On lattices with curved mean-line leg chains (``leg_next``) or chordwise
  flap hinge nodes (``a_kink``, ``b_kink``), each side edge carries the
  intermediate nodes of the steady chordwise leg as a bent polyline
  (``side_nodes`` and ``side_counts``), so the unsteady surface matches the
  steady vortex-lattice surface without cutting across camber or flap hinges.

Lifting-line collocation (``collocation="llt"``) is refused with
:class:`ventorum.core.errors.InputError`.

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, chapters 12 and 13 (vortex-ring lattice, steady
  closure and method of images).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ventorum.aero.system import (
    GroundPlane,
    freestream_direction,
    make_unknown_map,
)
from ventorum.aero.vortex_ring import (
    ring_normal_influence,
    steady_wake_normal_influence,
)
from ventorum.core.datatypes import FlightCondition
from ventorum.core.errors import InputError
from ventorum.geometry.lattice import (
    CROSS_SURFACE_CORE_FRACTION,
    VortexLattice,
)
from ventorum.solvers.core import _vlm_rhs

@dataclass(slots=True)
class RingLattice:
    """Vortex-ring representation of a :class:`VortexLattice`.

    All lengths are in metres and all direction vectors are dimensionless unit
    vectors in geometry axes (x aft, y right, z up).

    Attributes
    ----------
    corners : numpy.ndarray
        Ring corner coordinates [m], shape ``(n_panels, 4, 3)``. Corner order
        is ``0`` front-left, ``1`` front-right, ``2`` rear-right, ``3``
        rear-left.
    cp : numpy.ndarray
        Collocation points [m], shape ``(n_panels, 3)``.
    normal_bc : numpy.ndarray
        Unit boundary-condition normals [-], shape ``(n_panels, 3)``.
    panel_strip : numpy.ndarray
        Spanwise strip index of each panel, shape ``(n_panels,)`` int64.
    panel_chord_index : numpy.ndarray
        Chordwise index ``0 .. n_chord - 1`` of each panel inside its strip,
        shape ``(n_panels,)`` int64.
    te_mask : numpy.ndarray
        Boolean mask that is True for the trailing-edge ring of each strip,
        shape ``(n_panels,)``.
    te_left : numpy.ndarray
        Left trailing-edge closure corner of each strip [m], shape
        ``(n_strips, 3)`` (equals ``corners[te_mask, 3]``).
    te_right : numpy.ndarray
        Right trailing-edge closure corner of each strip [m], shape
        ``(n_strips, 3)`` (equals ``corners[te_mask, 2]``).
    rc : numpy.ndarray
        Core radius of each ring [m], shape ``(n_panels,)`` (same rule as
        :attr:`ventorum.geometry.lattice.VortexLattice.rc`).
    wake_dir : numpy.ndarray
        Unit wake direction of each strip for the steady closure [-], shape
        ``(n_strips, 3)``.
    n_chord : int
        Number of chordwise panels per strip.
    strip_surface : numpy.ndarray
        Surface index of each strip, shape ``(n_strips,)`` int64.
    strip_core_group : numpy.ndarray
        Core group index of each strip, shape ``(n_strips,)`` int64.
    strip_width : numpy.ndarray
        Width of each strip [m], shape ``(n_strips,)``.
    te_extension_fraction : float
        Trailing-edge closure distance aft of the trailing edge as a fraction
        of the last panel length [-].
    side_nodes : numpy.ndarray or None
        Padded intermediate nodes along the bent side edges [m], shape
        ``(n_panels, 2, K, 3)``, or None when all side edges are straight.
        Axis-1 index ``0`` holds the right side edge nodes in cyclic order
        ``1 -> 2``; axis-1 index ``1`` holds the left side edge nodes in
        cyclic order ``3 -> 0``.
    side_counts : numpy.ndarray or None
        Number of active intermediate nodes per ring side edge, shape
        ``(n_panels, 2)`` int64, or None when *side_nodes* is None.
    """

    corners: np.ndarray
    cp: np.ndarray
    normal_bc: np.ndarray
    panel_strip: np.ndarray
    panel_chord_index: np.ndarray
    te_mask: np.ndarray
    te_left: np.ndarray
    te_right: np.ndarray
    rc: np.ndarray
    wake_dir: np.ndarray
    n_chord: int
    strip_surface: np.ndarray
    strip_core_group: np.ndarray
    strip_width: np.ndarray
    te_extension_fraction: float = 0.25
    side_nodes: np.ndarray | None = None
    side_counts: np.ndarray | None = None

    @property
    def n_panels(self) -> int:
        """Return the number of vortex rings."""
        return int(self.corners.shape[0])

    @property
    def n_strips(self) -> int:
        """Return the number of spanwise strips."""
        return int(self.te_left.shape[0])

    @property
    def te_indices(self) -> np.ndarray:
        """Return the panel indices of the trailing-edge rings, shape ``(n_strips,)``."""
        return np.flatnonzero(self.te_mask)

    @property
    def has_side_nodes(self) -> bool:
        """Return True when at least one ring side edge carries intermediate nodes."""
        return (
            self.side_nodes is not None
            and self.side_counts is not None
            and bool(np.any(self.side_counts > 0))
        )


@dataclass(slots=True)
class RingGroundImage:
    """Mirror image of a :class:`RingLattice` in a :class:`GroundPlane`.

    The image rings and steady wake legs use the reflected geometry with
    circulation sign ``-1.0``, so the normal velocity vanishes on the ground
    plane (method of images, Katz and Plotkin 2001).

    Attributes
    ----------
    corners : numpy.ndarray
        Reflected ring corners [m], shape ``(n_panels, 4, 3)``.
    te_left : numpy.ndarray
        Reflected left trailing-edge closure corners [m], shape
        ``(n_strips, 3)``.
    te_right : numpy.ndarray
        Reflected right trailing-edge closure corners [m], shape
        ``(n_strips, 3)``.
    wake_dir : numpy.ndarray
        Reflected unit wake directions [-], shape ``(n_strips, 3)``.
    sign : float
        Circulation multiplier [-] for the image velocities (``-1.0``).
    side_nodes : numpy.ndarray or None
        Reflected padded intermediate side-edge nodes [m], shape
        ``(n_panels, 2, K, 3)``, or None when all side edges are straight.
    side_counts : numpy.ndarray or None
        Number of active intermediate nodes per ring side edge, shape
        ``(n_panels, 2)`` int64, or None when *side_nodes* is None.
    """

    corners: np.ndarray
    te_left: np.ndarray
    te_right: np.ndarray
    wake_dir: np.ndarray
    sign: float = -1.0
    side_nodes: np.ndarray | None = None
    side_counts: np.ndarray | None = None


def _normalize_wake_dir(wake_dir: np.ndarray | None, n_strips: int) -> np.ndarray:
    """Validate and normalise *wake_dir* to shape ``(n_strips, 3)``."""
    if wake_dir is None:
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float64), (n_strips, 1))
    wd = np.asarray(wake_dir, dtype=np.float64)
    if wd.ndim == 1:
        if wd.shape != (3,):
            raise InputError(f"wake_dir must have shape (3,) or ({n_strips}, 3), got {wd.shape}.")
        wd = np.tile(wd, (n_strips, 1))
    elif wd.ndim == 2:
        if wd.shape != (n_strips, 3):
            raise InputError(f"wake_dir must have shape (3,) or ({n_strips}, 3), got {wd.shape}.")
        wd = wd.copy()
    else:
        raise InputError(f"wake_dir must be 1-D or 2-D, got shape {wd.shape}.")
    if not np.all(np.isfinite(wd)):
        raise InputError("wake_dir must contain finite numbers.")
    norms = np.linalg.norm(wd, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise InputError("wake_dir must have non-zero length for every strip.")
    return wd / norms


def _point_distinct(p: np.ndarray, q: np.ndarray, tol: float) -> bool:
    """Return True when 3-D points *p* and *q* are farther apart than *tol* [m]."""
    dx = float(p[0] - q[0])
    dy = float(p[1] - q[1])
    dz = float(p[2] - q[2])
    return (dx * dx + dy * dy + dz * dz) > (tol * tol)


def _build_side_polylines(
    lattice: VortexLattice,
    corners: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build padded bent side-edge polylines from the steady leg nodes of *lattice*.

    Each side edge of ring ``p`` carries the intermediate nodes of the steady
    chordwise leg between its two end corners:

    * On a mean-line leg chain (``leg_next`` active on the strip), panel ``p``
      has intermediate node ``b_kink[p]`` on its right edge ``1 -> 2`` and
      ``a_kink[p]`` on its left edge ``3 -> 0``.
    * On a non-chain strip with chordwise flap hinge nodes (``a_kink``,
      ``b_kink``), only the ring whose side edge crosses the hinge carries the
      hinge node.
    * On a trailing-edge ring with ``te_extension_fraction > 0`` that also has
      an intermediate leg node ahead of the trailing edge, the physical
      trailing-edge node (``b_te[p]`` / ``a_te[p]``) is included so the side
      polyline passes through the physical trailing edge before extending aft.

    Returns ``(None, None)`` when no ring has any intermediate side node.
    """
    if lattice.a_kink is None and lattice.b_kink is None:
        return None, None

    n_strips = lattice.n_strips
    n_chord = int(lattice.n_chord)
    n_panels = lattice.n_panels
    a_te = np.asarray(lattice.a_te, dtype=np.float64)
    b_te = np.asarray(lattice.b_te, dtype=np.float64)
    a_kink = a_te if lattice.a_kink is None else np.asarray(lattice.a_kink, dtype=np.float64)
    b_kink = b_te if lattice.b_kink is None else np.asarray(lattice.b_kink, dtype=np.float64)
    leg_next = (
        np.zeros(n_panels, dtype=bool)
        if lattice.leg_next is None
        else np.asarray(lattice.leg_next, dtype=bool)
    )
    panel_len = np.asarray(lattice.panel_length, dtype=np.float64)

    right_lists: list[list[np.ndarray]] = [[] for _ in range(n_panels)]
    left_lists: list[list[np.ndarray]] = [[] for _ in range(n_panels)]
    max_nodes = 0

    for s in range(n_strips):
        p0 = s * n_chord
        is_chain = (n_chord > 1) and bool(np.any(leg_next[p0 : p0 + n_chord]))
        for k in range(n_chord):
            p = p0 + k
            is_te = k == (n_chord - 1)
            tol = 1e-13 * max(float(panel_len[p]), 1e-12)

            # Right side edge: c1 = b[p] -> ... -> c2.
            phys_r = b_te[p] if is_te else corners[p, 2]
            bk = b_kink[p]
            if is_chain:
                use_r = _point_distinct(bk, corners[p, 1], tol) and _point_distinct(bk, phys_r, tol)
            else:
                has_kink_r = _point_distinct(bk, b_te[p], tol)
                crosses_r = has_kink_r and (is_te or _point_distinct(bk, b_kink[p + 1], tol))
                use_r = crosses_r and _point_distinct(bk, corners[p, 1], tol) and _point_distinct(bk, phys_r, tol)
            if use_r:
                r_nodes = [bk]
                if is_te and _point_distinct(phys_r, corners[p, 2], tol):
                    r_nodes.append(phys_r)
                right_lists[p] = r_nodes
                if len(r_nodes) > max_nodes:
                    max_nodes = len(r_nodes)

            # Left side edge: c3 -> ... -> c0 = a[p] (cyclic order, aft to forward).
            phys_l = a_te[p] if is_te else corners[p, 3]
            ak = a_kink[p]
            if is_chain:
                use_l = _point_distinct(ak, corners[p, 0], tol) and _point_distinct(ak, phys_l, tol)
            else:
                has_kink_l = _point_distinct(ak, a_te[p], tol)
                crosses_l = has_kink_l and (is_te or _point_distinct(ak, a_kink[p + 1], tol))
                use_l = crosses_l and _point_distinct(ak, corners[p, 0], tol) and _point_distinct(ak, phys_l, tol)
            if use_l:
                l_nodes: list[np.ndarray] = []
                if is_te and _point_distinct(phys_l, corners[p, 3], tol):
                    l_nodes.append(phys_l)
                l_nodes.append(ak)
                left_lists[p] = l_nodes
                if len(l_nodes) > max_nodes:
                    max_nodes = len(l_nodes)

    if max_nodes == 0:
        return None, None

    side_nodes = np.empty((n_panels, 2, max_nodes, 3), dtype=np.float64)
    side_counts = np.zeros((n_panels, 2), dtype=np.int64)
    # Pad unused slots with the side end corner so every stored point is finite and physical.
    side_nodes[:, 0, :, :] = corners[:, 2, :][:, None, :]
    side_nodes[:, 1, :, :] = corners[:, 0, :][:, None, :]
    for p in range(n_panels):
        rn = right_lists[p]
        if rn:
            side_counts[p, 0] = len(rn)
            for idx_n, pt in enumerate(rn):
                side_nodes[p, 0, idx_n, :] = pt
        ln = left_lists[p]
        if ln:
            side_counts[p, 1] = len(ln)
            for idx_n, pt in enumerate(ln):
                side_nodes[p, 1, idx_n, :] = pt
    return side_nodes, side_counts


def _validate_lattice_for_rings(lattice: VortexLattice) -> None:
    """Validate that *lattice* can be mapped cleanly to vortex rings."""
    if lattice.collocation != "vlm":
        raise InputError(
            f"build_ring_lattice requires vortex-lattice collocation ('vlm'), got {lattice.collocation!r}."
        )
    n_strips = lattice.n_strips
    n_chord = int(lattice.n_chord)
    n_panels = lattice.n_panels
    if n_strips < 1 or n_chord < 1 or n_panels != n_strips * n_chord:
        raise InputError(
            f"Invalid lattice panel layout: n_panels={n_panels}, n_strips={n_strips}, n_chord={n_chord}."
        )
    expected_strip = np.repeat(np.arange(n_strips, dtype=np.int64), n_chord)
    if not np.array_equal(np.asarray(lattice.panel_strip, dtype=np.int64), expected_strip):
        raise InputError("build_ring_lattice requires panels ordered strip by strip from leading to trailing edge.")


def build_ring_lattice(
    lattice: VortexLattice,
    *,
    wake_dir: np.ndarray | None = None,
    te_extension_fraction: float = 0.25,
) -> RingLattice:
    """Build a :class:`RingLattice` from an existing :class:`VortexLattice`.

    Parameters
    ----------
    lattice : VortexLattice
        Built vortex-lattice mesh (``collocation="vlm"``).
    wake_dir : numpy.ndarray or None, optional
        Unit wake direction [-] used by the steady closure, shape ``(3,)`` or
        ``(n_strips, 3)``. None uses the body x-axis ``(1, 0, 0)``.
    te_extension_fraction : float, optional
        Distance of the trailing-edge ring closure aft of the trailing edge as
        a fraction of the trailing-edge panel length [-]. The default ``0.25``
        follows Katz and Plotkin (2001, chapter 13). Use ``0.0`` to close the
        last ring directly on the trailing edge.

    Returns
    -------
    RingLattice
        Vortex-ring lattice with one vortex ring per lattice panel, including
        bent side polylines for cambered mean lines and deflected chordwise
        flaps.

    Raises
    ------
    InputError
        If *lattice* uses lifting-line collocation or an invalid panel layout,
        or if *wake_dir* or *te_extension_fraction* is invalid.
    """
    _validate_lattice_for_rings(lattice)
    te_ext = float(te_extension_fraction)
    if not np.isfinite(te_ext) or te_ext < 0.0:
        raise InputError(f"te_extension_fraction must be finite and >= 0, got {te_extension_fraction!r}.")

    n_strips = lattice.n_strips
    n_chord = int(lattice.n_chord)
    n_panels = lattice.n_panels
    wd = _normalize_wake_dir(wake_dir, n_strips)

    panel_strip = np.repeat(np.arange(n_strips, dtype=np.int64), n_chord)
    panel_chord_index = np.tile(np.arange(n_chord, dtype=np.int64), n_strips)
    te_mask = panel_chord_index == (n_chord - 1)
    te_idx = np.flatnonzero(te_mask)

    a = np.asarray(lattice.a, dtype=np.float64)
    b = np.asarray(lattice.b, dtype=np.float64)
    corners = np.empty((n_panels, 4, 3), dtype=np.float64)
    corners[:, 0, :] = a
    corners[:, 1, :] = b

    if n_chord > 1:
        interior = np.flatnonzero(~te_mask)
        corners[interior, 2, :] = b[interior + 1]
        corners[interior, 3, :] = a[interior + 1]

    # Trailing-edge closure rule: the last panel bound vortex lies at
    # 1 - 0.75 * d_xi_last along each strip edge, so (te - bound) spans 0.75
    # of the last panel length. Extending by (te_ext / 0.75) * (te - bound)
    # places the trailing corners te_ext panel lengths aft of the TE along
    # each strip edge. Joined surface edges share their bound and TE nodes,
    # so their trailing-edge closures stay continuous across junctions.
    te_l_geom = np.asarray(lattice.te_left, dtype=np.float64)
    te_r_geom = np.asarray(lattice.te_right, dtype=np.float64)
    scale = te_ext / 0.75
    te_left = te_l_geom + scale * (te_l_geom - a[te_idx])
    te_right = te_r_geom + scale * (te_r_geom - b[te_idx])
    corners[te_idx, 2, :] = te_right
    corners[te_idx, 3, :] = te_left

    side_nodes, side_counts = _build_side_polylines(lattice, corners)

    strip_surface = np.asarray(lattice.strip_surface, dtype=np.int64).copy()
    raw_group = lattice.strip_core_group
    strip_core_group = (
        strip_surface.copy() if raw_group is None else np.asarray(raw_group, dtype=np.int64).copy()
    )

    return RingLattice(
        corners=corners,
        cp=np.asarray(lattice.cp, dtype=np.float64).copy(),
        normal_bc=np.asarray(lattice.normal_bc, dtype=np.float64).copy(),
        panel_strip=panel_strip,
        panel_chord_index=panel_chord_index,
        te_mask=te_mask,
        te_left=te_left,
        te_right=te_right,
        rc=np.asarray(lattice.rc, dtype=np.float64).copy(),
        wake_dir=wd,
        n_chord=n_chord,
        strip_surface=strip_surface,
        strip_core_group=strip_core_group,
        strip_width=np.asarray(lattice.width, dtype=np.float64).copy(),
        te_extension_fraction=te_ext,
        side_nodes=side_nodes,
        side_counts=side_counts,
    )


def ground_image_rings(
    rings: RingLattice,
    ground: GroundPlane,
    *,
    wake_dir: np.ndarray | None = None,
) -> RingGroundImage:
    """Return the mirror-image rings and steady wake legs for *ground*.

    Uses :meth:`ventorum.aero.system.GroundPlane.reflect_points` for all ring,
    side-polyline and trailing-edge corners and
    :meth:`ventorum.aero.system.GroundPlane.reflect_directions` for the wake
    directions. The image circulation sign is ``-1.0``, matching the steady
    image method in :mod:`ventorum.aero.system`.

    Parameters
    ----------
    rings : RingLattice
        Real vortex-ring lattice.
    ground : GroundPlane
        Flat ground plane in geometry axes.
    wake_dir : numpy.ndarray or None, optional
        Override unit wake direction [-], shape ``(3,)`` or ``(n_strips, 3)``.
        None uses ``rings.wake_dir``.

    Returns
    -------
    RingGroundImage
        Reflected ring corners, reflected side-edge nodes, reflected
        trailing-edge corners, reflected wake directions and circulation sign
        ``-1.0``.
    """
    wd = rings.wake_dir if wake_dir is None else _normalize_wake_dir(wake_dir, rings.n_strips)
    return RingGroundImage(
        corners=ground.reflect_points(rings.corners),
        te_left=ground.reflect_points(rings.te_left),
        te_right=ground.reflect_points(rings.te_right),
        wake_dir=ground.reflect_directions(wd),
        sign=-1.0,
        side_nodes=None if rings.side_nodes is None else ground.reflect_points(rings.side_nodes),
        side_counts=None if rings.side_counts is None else rings.side_counts.copy(),
    )


def _rings_to_horseshoe_gamma(gamma_rings: np.ndarray, n_chord: int) -> np.ndarray:
    """Convert ring circulations [m^2/s] to horseshoe circulations [m^2/s].

    On each strip, ``gamma_hs[0] = Gamma[0]`` and
    ``gamma_hs[k] = Gamma[k] - Gamma[k - 1]`` for ``k >= 1``.
    """
    g = np.asarray(gamma_rings, dtype=np.float64).reshape(-1, int(n_chord))
    out = np.empty_like(g)
    out[:, 0] = g[:, 0]
    if n_chord > 1:
        out[:, 1:] = g[:, 1:] - g[:, :-1]
    return out.ravel()


def _ring_pair_cores(
    rings: RingLattice,
    rc: np.ndarray | float | None,
    use_cross_surface_core: bool,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Return ``(rc_rings, rc_wake)`` [m], adding cross-surface target cores when active."""
    te_idx = rings.te_indices
    if rc is not None:
        arr = np.asarray(rc, dtype=np.float64)
        if arr.ndim == 0:
            val = float(arr)
            return val, val
        if arr.ndim == 1:
            return arr, arr[te_idx]
        if arr.ndim == 2:
            return arr, arr[:, te_idx]
        raise InputError(f"rc must be scalar, 1-D or 2-D, got shape {arr.shape}.")

    rc_src = rings.rc
    if not use_cross_surface_core:
        return rc_src, rc_src[te_idx]

    panel_group = rings.strip_core_group[rings.panel_strip]
    if np.all(panel_group == panel_group[0]):
        return rc_src, rc_src[te_idx]

    tg_rc2 = (CROSS_SURFACE_CORE_FRACTION * rings.strip_width[rings.panel_strip]) ** 2
    other = panel_group[:, None] != panel_group[None, :]
    rc_mat = np.sqrt(rc_src[None, :] ** 2 + np.where(other, tg_rc2[:, None], 0.0))
    return rc_mat, rc_mat[:, te_idx]


def _steady_ring_influence(
    rings: RingLattice,
    ground: GroundPlane | None = None,
    *,
    wake_dir: np.ndarray | None = None,
    rc: np.ndarray | float | None = None,
    use_cross_surface_core: bool = True,
) -> np.ndarray:
    """Assemble the steady normal-velocity influence matrix [1/m] of *rings*.

    Includes the closed bound rings, the steady semi-infinite wake closure on
    the trailing-edge rings, and their mirror images when *ground* is given.
    """
    wd = rings.wake_dir if wake_dir is None else _normalize_wake_dir(wake_dir, rings.n_strips)
    te_idx = rings.te_indices
    rc_ring, rc_wake = _ring_pair_cores(rings, rc, use_cross_surface_core)

    A = ring_normal_influence(
        rings.cp, rings.normal_bc, rings.corners, rc_ring,
        side_nodes=rings.side_nodes, side_counts=rings.side_counts,
    )
    A[:, te_idx] += steady_wake_normal_influence(
        rings.cp, rings.normal_bc, rings.te_left, rings.te_right, wd, rc_wake,
    )
    if ground is not None:
        img = ground_image_rings(rings, ground, wake_dir=wd)
        A += img.sign * ring_normal_influence(
            rings.cp, rings.normal_bc, img.corners, rc_ring,
            side_nodes=img.side_nodes, side_counts=img.side_counts,
        )
        A[:, te_idx] += img.sign * steady_wake_normal_influence(
            rings.cp, rings.normal_bc, img.te_left, img.te_right, img.wake_dir, rc_wake,
        )
    return A


def _solve_steady_rings(
    lattice: VortexLattice,
    condition: FlightCondition,
    *,
    ground: GroundPlane | None = None,
    wake_dir: np.ndarray | None = None,
    ref_point: np.ndarray | None = None,
    te_extension_fraction: float = 0.25,
    rc: np.ndarray | float | None = None,
    use_cross_surface_core: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the steady vortex-ring system for *lattice* and *condition*.

    This private helper supports steady parity verification against the
    existing horseshoe vortex-lattice solver. It assembles the ring system
    (with steady wake closure and optional ground images), solves it with the
    right-hand side of :func:`ventorum.solvers.core.assemble_vlm`, and
    converts ring circulations to horseshoe circulations.

    Parameters
    ----------
    lattice : VortexLattice
        Built vortex-lattice mesh (``collocation="vlm"``).
    condition : FlightCondition
        Flight state (speed [m/s], angles [rad], optional body rates [rad/s]).
    ground : GroundPlane or None, optional
        Ground plane in geometry axes, or None for free air.
    wake_dir : numpy.ndarray or None, optional
        Unit wake direction [-], shape ``(3,)`` or ``(n_strips, 3)``. None
        uses the free-stream direction of *condition*.
    ref_point : numpy.ndarray or None, optional
        Rotation centre [m] for body rates, shape ``(3,)``. None uses the
        origin.
    te_extension_fraction : float, optional
        Trailing-edge closure offset as a fraction of the trailing-edge panel
        length [-]. Default ``0.25``.
    rc : numpy.ndarray or float or None, optional
        Optional core radius override [m]. None uses ``lattice.rc`` (plus
        cross-surface core when *use_cross_surface_core* is True).
    use_cross_surface_core : bool, optional
        When True and *rc* is None, add the cross-surface target core between
        different core groups as in :mod:`ventorum.aero.system`.

    Returns
    -------
    gamma_rings : numpy.ndarray
        Solved vortex-ring circulations [m^2/s], shape ``(n_panels,)``.
    gamma_horseshoe : numpy.ndarray
        Equivalent horseshoe circulations [m^2/s], shape ``(n_panels,)``.
    """
    wd = (
        freestream_direction(condition.alpha, condition.beta)
        if wake_dir is None
        else np.asarray(wake_dir, dtype=np.float64)
    )
    rings = build_ring_lattice(lattice, wake_dir=wd, te_extension_fraction=te_extension_fraction)
    A = _steady_ring_influence(
        rings, ground=ground, wake_dir=wd, rc=rc, use_cross_surface_core=use_cross_surface_core,
    )
    umap = make_unknown_map(lattice, False)
    rhs = _vlm_rhs(lattice, umap, condition, ref_point)
    gamma_rings = np.linalg.solve(A, rhs)
    gamma_hs = _rings_to_horseshoe_gamma(gamma_rings, rings.n_chord)
    return gamma_rings, gamma_hs
