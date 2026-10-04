# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Free stream, ground plane and source sets for the lattice solvers.

Everything is in body axes (x aft, y right, z up). The aircraft is not
rotated. Angle of attack and sideslip set the free-stream direction. In
ground effect, the ground plane is tilted in body axes so that it stays
parallel to the free stream (level flight over flat ground), and the bank
angle turns the plane about the free-stream direction.

Image method: a source horseshoe is mirrored in the ground plane and its
circulation changes sign, so the velocity normal to the ground is zero on the
plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ventorum.aero.vortex import HorseshoeSet, Targets
from ventorum.core.datatypes import FlightCondition
from ventorum.core.errors import GroundStrikeError
from ventorum.geometry.lattice import CROSS_SURFACE_CORE_FRACTION, VortexLattice
from ventorum.utils.vec import cross3


def freestream_direction(alpha: float, beta: float = 0.0) -> np.ndarray:
    """Return the unit free-stream direction in body axes.

    Parameters
    ----------
    alpha : float
        Angle of attack [rad].
    beta : float, optional
        Sideslip angle [rad]. The default is 0.

    Returns
    -------
    numpy.ndarray
        Unit vector, shape (3,).

    Notes
    -----
    Positive sideslip ``beta`` is wind from the right (nose left of the
    velocity vector), the same convention as AVL.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    return np.array([ca * cb, -sb, sa * cb])


def wake_direction(
    condition: FlightCondition,
    ground: GroundPlane | None = None,
    alignment: str = "freestream",
) -> np.ndarray:
    """Direction of the semi-infinite wake legs (from the trailing edge).

    * ``"freestream"`` (default): along the free stream. In ground effect this
      keeps the wake parallel to the ground, which the image method needs.
    * ``"body"``: along the body x-axis, as in AVL. The wake geometry and the
      system matrix then do not depend on alpha and beta, and the circulation
      is a linear function of the free-stream components (for example
      proportional to sin(alpha) for a flat wing). It is not allowed in
      ground effect.
    """
    if alignment == "body":
        if ground is not None:
            import warnings
            warnings.warn("wake_alignment='body' is not valid in ground effect; the free-stream "
                          "direction is used.", RuntimeWarning, stacklevel=3)
        else:
            return np.array([1.0, 0.0, 0.0])
    elif alignment != "freestream":
        raise ValueError(f"wake_alignment={alignment!r} must be 'freestream' or 'body'.")
    return freestream_direction(condition.alpha, condition.beta)


def lift_direction(alpha: float) -> np.ndarray:
    """Return the unit lift direction for the angle of attack *alpha* [rad].

    The direction is normal to the free stream and is in the body x-z plane.
    """
    return np.array([-np.sin(alpha), 0.0, np.cos(alpha)])


def side_direction(alpha: float, beta: float) -> np.ndarray:
    """Return the unit side-force direction (positive towards the right wing).

    *alpha* and *beta* are the angle of attack and the sideslip angle [rad].
    """
    return cross3(lift_direction(alpha), freestream_direction(alpha, beta))


def ground_normal(alpha: float, beta: float = 0.0, phi: float = 0.0) -> np.ndarray:
    """Return the unit normal of the ground in body axes.

    The normal points up, to the aircraft. *alpha*, *beta* and *phi* are the
    angle of attack, the sideslip angle and the bank angle [rad].

    The ground is parallel to the free stream. With ``phi = 0`` the wings are
    level: the normal is the lift direction ``(-sin(alpha), 0, cos(alpha))``,
    which is normal to the free stream for every sideslip and has no y
    component. The bank angle turns the ground about the free-stream
    direction (a wind-axis bank angle); positive ``phi`` puts the right wing
    nearer to the ground.
    """
    v = freestream_direction(alpha, beta)
    k0 = lift_direction(alpha)
    k = k0 * np.cos(phi) + cross3(v, k0) * np.sin(phi)
    return k / np.linalg.norm(k)


@dataclass(slots=True)
class GroundPlane:
    """Flat ground plane in body axes.

    The points x on the plane satisfy ``dot(x, normal) = offset``.
    """

    normal: np.ndarray
    offset: float

    def height(self, pts: np.ndarray) -> np.ndarray:
        """Distance of points above the ground (negative below)."""
        return np.asarray(pts, dtype=float) @ self.normal - self.offset

    def reflect_points(self, pts: np.ndarray) -> np.ndarray:
        """Return the mirror images of the points *pts* [m] in the ground."""
        h = self.height(pts)
        return pts - 2.0 * h[..., None] * self.normal

    def reflect_directions(self, v: np.ndarray) -> np.ndarray:
        """Return the mirror images of the direction vectors *v* in the ground."""
        return v - 2.0 * (v @ self.normal)[..., None] * self.normal


def make_ground_plane(
    lattice: VortexLattice,
    h: float,
    alpha: float,
    beta: float = 0.0,
    phi: float = 0.0,
    ref_point: np.ndarray | None = None,
    height_ref: Literal["ref", "min"] = "ref",
) -> GroundPlane:
    """Place the ground plane for a given height convention.

    Parameters
    ----------
    h : float
        Height [m] (must be > 0).
    ref_point : (3,) or None
        Reference point for ``height_ref='ref'``. Default: the origin.
    height_ref : {'ref', 'min'}
        * ``'ref'``: height of *ref_point*.
        * ``'min'``: smallest height of any leading- or trailing-edge node.

        For a height at the root quarter chord or trailing edge, give
        ``ref_point=Aircraft.root_point('qc')`` (or ``'te'``) with
        ``height_ref='ref'``.
    """
    if h is None or h <= 0.0:
        raise ValueError(f"Ground height h={h} must be strictly positive.")
    k = ground_normal(alpha, beta, phi)
    if height_ref == "ref":
        p = np.zeros(3) if ref_point is None else np.asarray(ref_point, dtype=float)
        offset = float(p @ k - h)
    elif height_ref == "min":
        offset = float(np.min(lattice.all_points() @ k) - h)
    else:
        raise ValueError(f"Unknown height_ref={height_ref!r}; use 'ref' or 'min'.")
    return GroundPlane(normal=k, offset=offset)


def ground_plane_from_condition(
    lattice: VortexLattice,
    condition: FlightCondition,
    ref_point: np.ndarray | None = None,
) -> GroundPlane | None:
    """Ground plane of a :class:`FlightCondition`.

    ``condition.h`` is the height of the moment reference point *ref_point*
    (the origin of the geometry axes if None), the same convention as
    :func:`ventorum.ground_effect.analyze_ground_effect` with
    ``height_ref='ref'``.
    """
    if condition.h is None:
        return None
    return make_ground_plane(
        lattice, float(condition.h), condition.alpha, condition.beta,
        getattr(condition, "phi", 0.0), ref_point=ref_point, height_ref="ref",
    )


def is_symmetric_condition(condition: FlightCondition, ground: GroundPlane | None) -> bool:
    """Return True if the flow is symmetric about the body x-z plane."""
    if abs(condition.beta) > 1e-12:
        return False
    if ground is not None and abs(ground.normal[1]) > 1e-12:
        return False
    return True


@dataclass(slots=True)
class UnknownMap:
    """Which panels carry the unknowns, and how panels map to unknowns."""

    unknown_panels: np.ndarray   # panel index of each unknown
    panel_column: np.ndarray     # unknown index of each panel
    symmetric: bool

    @property
    def n(self) -> int:
        """Return the number of unknowns."""
        return int(self.unknown_panels.shape[0])


def make_unknown_map(lattice: VortexLattice, use_symmetry: bool) -> UnknownMap:
    """Map panels to unknowns, folding mirror panels when *use_symmetry*."""
    n = lattice.n_panels
    if use_symmetry and lattice.can_fold_symmetry():
        right = np.flatnonzero(lattice.strip_is_right[lattice.panel_strip])
        col = np.full(n, -1)
        col[right] = np.arange(right.size)
        left = np.flatnonzero(col < 0)
        col[left] = col[lattice.panel_mirror[left]]
        return UnknownMap(unknown_panels=right, panel_column=col, symmetric=True)
    return UnknownMap(unknown_panels=np.arange(n), panel_column=np.arange(n), symmetric=False)


def build_sources(
    lattice: VortexLattice,
    wake_dir: np.ndarray,
    umap: UnknownMap,
    ground: GroundPlane | None = None,
) -> HorseshoeSet:
    """All source horseshoes (real panels and ground images)."""
    n = lattice.n_panels
    d = np.tile(np.asarray(wake_dir, dtype=float), (n, 1))
    group = _core_group(lattice)[lattice.panel_strip]
    real = HorseshoeSet(
        a=lattice.a, b=lattice.b, a_te=lattice.a_te, b_te=lattice.b_te,
        wake_dir=d, rc=lattice.rc, sign=np.ones(n), column=umap.panel_column.copy(), group=group,
    )
    if ground is None:
        return real
    img = HorseshoeSet(
        a=ground.reflect_points(lattice.a),
        b=ground.reflect_points(lattice.b),
        a_te=ground.reflect_points(lattice.a_te),
        b_te=ground.reflect_points(lattice.b_te),
        wake_dir=ground.reflect_directions(d),
        rc=lattice.rc,
        sign=-np.ones(n),
        column=umap.panel_column.copy(),
        group=group,
    )
    return real.concatenate(img)


def _core_group(lattice: VortexLattice) -> np.ndarray:
    g = lattice.strip_core_group
    return lattice.strip_surface if g is None else g


def panel_targets(lattice: VortexLattice, panels: np.ndarray) -> Targets:
    """Cross-surface core data for evaluation points on the given panels."""
    strip = lattice.panel_strip[np.asarray(panels)]
    return Targets(group=_core_group(lattice)[strip],
                   rc=CROSS_SURFACE_CORE_FRACTION * lattice.width[strip])


def check_ground_clearance(lattice: VortexLattice, ground: GroundPlane | None) -> float | None:
    """Smallest height of any lattice node. Raises if the aircraft touches the ground."""
    if ground is None:
        return None
    h_min = float(np.min(ground.height(lattice.all_points())))
    if h_min <= 0.0:
        raise GroundStrikeError(
            f"Geometry touches or crosses the ground plane (minimum clearance {h_min:.4g} m). "
            "No result is computed for a ground strike."
        )
    return h_min
