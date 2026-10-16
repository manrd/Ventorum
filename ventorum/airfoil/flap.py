# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Rotate the aft part of airfoil coordinates about a plain-flap hinge.

A plain flap is the aft part of an airfoil section behind a hinge
station. This module deflects it as a rigid body. It rotates every point
aft of the hinge about the hinge point. The fixed part keeps its points.

The sign of the deflection selects the compressed side and the stretched
side. A trailing edge down deflection compresses the lower surface and
stretches the upper surface; a trailing edge up deflection does the
opposite. On the compressed side, the fixed and the moved surface are cut
at the crossing of their polylines, and the overlapping points are
removed. When the two polylines do not cross (on coarse input the moved
surface can pass the fixed corner), both parts are cut at the hinge line
(the line through the hinge point normal to the chord) and joined there.
On the stretched side, a circular arc about the hinge joins the fixed and
the moved surface. The result must be a simple polygon in Selig order
with a positive area; otherwise the function raises.

The leading edge that splits the surfaces is the point farthest from the
trailing-edge midpoint, the same definition as
:func:`ventorum.airfoil.geometry.leading_edge_index`.

Hinge height: ``y_hinge`` is a fraction of the local thickness at the
hinge station, from the lower surface (0) to the upper surface (1). The
default 0.5 is the mid-thickness point. The theory page
``docs/theory/airfoils.md`` states this definition for the whole
package.

Sign convention: ``deflection`` in degrees [deg] is positive when the
trailing edge moves down. A positive deflection rotates the flap
clockwise in the section plane, with the chord axis to the right and the
normal axis up. This matches the control-surface convention of Ventorum:
a positive deflection moves the trailing edge down on a horizontal
surface (see ``docs/theory/control_surfaces.md``).

The construction is pure geometry. It models no aerodynamics.

References
----------
* H. Glauert, The Elements of Aerofoil and Airscrew Theory, Cambridge
  University Press, 1926, chapter VII: thin-airfoil theory of a hinged
  flap.
* H. Glauert, Theoretical relationships for an aerofoil with hinged
  flap, Aeronautical Research Committee, R&M 1095, 1927: the plain flap
  as a hinged aft portion of the section.
* I. H. Abbott and A. E. von Doenhoff, Theory of Wing Sections, Dover,
  1959: airfoil section geometry and the NACA 4-digit profile equations
  used in the tests.
"""

from __future__ import annotations

import math
import operator

import numpy as np

from ventorum.airfoil.geometry import (
    AirfoilGeometryError,
    leading_edge_index,
    polygon_self_intersection,
)

__all__ = ["deflect_plain_flap", "flap_hinge_point"]

_X_HINGE_MIN = 0.05
_X_HINGE_MAX = 0.98
_MAX_DEFLECTION_DEG = 60.0
_DEDUPE_TOL = 1e-14
_CROSS_EPS = 1e-9


def _as_coords(coords: np.ndarray) -> np.ndarray:
    """Return the input as a finite (N, 2) float array."""
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must be an array of shape (N, 2).")
    if arr.shape[0] < 4:
        raise ValueError("coords must hold at least 4 points.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("coords must hold finite values only.")
    return arr


def _check_station(x_hinge: float) -> float:
    """Return the hinge station [chord units] after range checks."""
    try:
        xh = float(x_hinge)
    except (TypeError, ValueError):
        raise ValueError("x_hinge must be a scalar in (0.05, 0.98).") from None
    if not math.isfinite(xh) or not (_X_HINGE_MIN < xh < _X_HINGE_MAX):
        raise ValueError("x_hinge must lie in (0.05, 0.98) chord units.")
    return xh


def _check_deflection(deflection: float) -> float:
    """Return the deflection [deg] after range checks."""
    try:
        delta = float(deflection)
    except (TypeError, ValueError):
        raise ValueError("deflection must be a scalar in [-60, 60] deg.") from None
    if not math.isfinite(delta) or abs(delta) > _MAX_DEFLECTION_DEG:
        raise ValueError("deflection must satisfy |deflection| <= 60 deg.")
    return delta


def _check_fraction(y_hinge: float) -> float:
    """Return the hinge height fraction [-] after range checks."""
    try:
        yh = float(y_hinge)
    except (TypeError, ValueError):
        raise ValueError("y_hinge must be a scalar in [0, 1].") from None
    if not math.isfinite(yh) or not (0.0 <= yh <= 1.0):
        raise ValueError("y_hinge must lie in [0, 1].")
    return yh


def _check_arc_count(n_arc: int | None) -> int | None:
    """Return the arc point count [-] after range checks."""
    if n_arc is None:
        return None
    try:
        n = operator.index(n_arc)
    except TypeError:
        raise ValueError("n_arc must be an integer of at least 2.") from None
    if n < 2:
        raise ValueError("n_arc must be an integer of at least 2.")
    return n


def _surfaces(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split Selig-ordered points at the leading edge.

    The leading edge is the point farthest from the trailing-edge
    midpoint (:func:`ventorum.airfoil.geometry.leading_edge_index`).
    Both returned surfaces run from the leading edge to the trailing
    edge [chord units].
    """
    x = arr[:, 0]
    y = arr[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if area <= 0.0:
        raise ValueError(
            "coords must run counterclockwise in Selig order "
            "(trailing edge, upper surface, leading edge, lower surface)."
        )
    le = leading_edge_index(arr)
    if le == 0 or le == arr.shape[0] - 1:
        raise ValueError("coords must run from the trailing edge to the leading edge and back.")
    return arr[: le + 1][::-1].copy(), arr[le:].copy()


def _aft_start(xs: np.ndarray, xh: float) -> int:
    """Return the first index of the monotonic aft run of one surface.

    The surface runs from the leading edge to the trailing edge. Points
    near the leading edge can have a short run of decreasing x (a
    cambered nose, or a deflected section whose farthest point is not
    the minimum x). The function accepts this run only forward of the
    hinge station. From the returned index to the trailing edge, x does
    not decrease.
    """
    drops = np.nonzero(np.diff(xs) < 0.0)[0]
    start = int(drops[-1]) + 1 if drops.size else 0
    if np.any(xs[: start + 1] >= xh):
        raise ValueError(
            "Each surface must have non-decreasing x from the leading-edge "
            "region to the trailing edge, at and aft of the hinge station."
        )
    if float(xs[-1]) <= xh:
        raise ValueError("The trailing edge must lie aft of the hinge station.")
    return start


def flap_hinge_point(coords: np.ndarray, x_hinge: float, y_hinge: float = 0.5) -> np.ndarray:
    """Return the flap hinge point [chord units].

    Parameters
    ----------
    coords : numpy.ndarray
        Airfoil coordinates [chord units], shape (N, 2), in Selig order.
    x_hinge : float
        Hinge chord station [chord units], in (0.05, 0.98).
    y_hinge : float
        Hinge height as a fraction [-] of the local thickness at
        ``x_hinge``, from the lower surface (0) to the upper surface
        (1). Default 0.5 (mid-thickness).

    Returns
    -------
    numpy.ndarray
        Hinge point [chord units], shape (2,).
    """
    arr = _as_coords(coords)
    xh = _check_station(x_hinge)
    yh = _check_fraction(y_hinge)
    upper, lower = _surfaces(arr)
    return _hinge_from_surfaces(upper, lower, xh, yh)


def _hinge_from_surfaces(upper: np.ndarray, lower: np.ndarray, xh: float, yh: float) -> np.ndarray:
    """Return the hinge point [chord units] from the two surfaces."""
    ku = _aft_start(upper[:, 0], xh)
    kl = _aft_start(lower[:, 0], xh)
    y_up = float(np.interp(xh, upper[ku:, 0], upper[ku:, 1]))
    y_lo = float(np.interp(xh, lower[kl:, 0], lower[kl:, 1]))
    return np.array([xh, y_lo + yh * (y_up - y_lo)])


def _rotate_points(pts: np.ndarray, hinge: np.ndarray, cos_t: float, sin_t: float) -> np.ndarray:
    """Rotate points about the hinge by the angle of (cos_t, sin_t)."""
    d = pts - hinge
    return hinge + np.column_stack([cos_t * d[:, 0] - sin_t * d[:, 1],
                                   sin_t * d[:, 0] + cos_t * d[:, 1]])


def _crossing(fixed: np.ndarray, moved: np.ndarray) -> tuple[int, int, np.ndarray] | None:
    """Return the cut point of the fixed and the moved surface.

    The search tests every pair of segments and returns the crossing
    nearest to the hinge on the fixed side (largest fixed index, then
    smallest moved index). None means the polylines do not cross.
    """
    p = fixed[:-1][:, None, :]
    d = (fixed[1:] - fixed[:-1])[:, None, :]
    q = moved[:-1][None, :, :]
    e = (moved[1:] - moved[:-1])[None, :, :]
    denom = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]
    qx = q[..., 0] - p[..., 0]
    qy = q[..., 1] - p[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qx * e[..., 1] - qy * e[..., 0]) / denom
        u = (qx * d[..., 1] - qy * d[..., 0]) / denom
    hit = (
        (denom != 0.0)
        & (t >= -_CROSS_EPS) & (t <= 1.0 + _CROSS_EPS)
        & (u >= -_CROSS_EPS) & (u <= 1.0 + _CROSS_EPS)
    )
    if not bool(np.any(hit)):
        return None
    rows, cols = np.nonzero(hit)
    i = int(np.max(rows))
    j = int(np.min(cols[rows == i]))
    point = fixed[i] + min(1.0, max(0.0, float(t[i, j]))) * (fixed[i + 1] - fixed[i])
    return i, j, point


def _hinge_line_cut(moved: np.ndarray, xh: float) -> tuple[int, np.ndarray]:
    """Return where the moved surface first reaches the hinge line x = xh.

    The result (j, point) means that the point lies on the segment from
    ``moved[j]`` to ``moved[j + 1]``, so the kept moved points are
    ``moved[j + 1:]``. When the first moved point already lies at or aft
    of the hinge line, the result is (-1, moved[0]).
    """
    xs = moved[:, 0]
    if float(xs[0]) >= xh:
        return -1, moved[0].copy()
    for j in range(moved.shape[0] - 1):
        x0 = float(xs[j])
        x1 = float(xs[j + 1])
        if x0 < xh <= x1:
            frac = (xh - x0) / (x1 - x0)
            return j, moved[j] + frac * (moved[j + 1] - moved[j])
    raise AirfoilGeometryError(
        "The deflected surface does not reach the hinge line; reduce the deflection."
    )


def _hinge_arc(start: np.ndarray, end: np.ndarray, hinge: np.ndarray, n: int) -> np.ndarray:
    """Return n points on the circular arc about the hinge from start to end."""
    radius = float(np.linalg.norm(start - hinge))
    if radius == 0.0:
        return np.vstack([start, end])
    a0 = math.atan2(start[1] - hinge[1], start[0] - hinge[0])
    a1 = math.atan2(end[1] - hinge[1], end[0] - hinge[0])
    sweep = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
    angles = a0 + np.linspace(0.0, sweep, n)
    return hinge + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _without_close_dups(pts: np.ndarray) -> np.ndarray:
    """Remove consecutive points closer than the dedupe tolerance."""
    keep = [0]
    for i in range(1, pts.shape[0]):
        if float(np.linalg.norm(pts[i] - pts[keep[-1]])) > _DEDUPE_TOL:
            keep.append(i)
    return pts[keep]


def _default_arc_count(fixed: np.ndarray, moved: np.ndarray, hinge: np.ndarray) -> int:
    """Return an arc count [-] with panel lengths near the local panel length."""
    arc_len = float(np.linalg.norm(fixed[-1] - hinge)) * abs(
        math.atan2(moved[0][1] - hinge[1], moved[0][0] - hinge[0])
        - math.atan2(fixed[-1][1] - hinge[1], fixed[-1][0] - hinge[0]))
    first = float(np.linalg.norm(moved[1] - moved[0]))
    if fixed.shape[0] >= 2:
        last = float(np.linalg.norm(fixed[-1] - fixed[-2]))
        local = 0.5 * (last + first)
    else:
        local = first
    if local <= 0.0:
        return 2
    return max(2, round(arc_len / local) + 1)


def _deflect_surface(surf: np.ndarray, xh: float, hinge: np.ndarray,
                     cos_t: float, sin_t: float, n_arc: int | None,
                     compressed: bool) -> np.ndarray:
    """Deflect one surface given from the leading to the trailing edge.

    ``compressed`` is True for the side that the rotation compresses
    (the lower surface for trailing edge down). That side is cut; the
    other side gets the filler arc.
    """
    xs = surf[:, 0]
    k0 = _aft_start(xs, xh)
    k = k0 + int(np.searchsorted(xs[k0:], xh, side="right"))
    y_corner = float(np.interp(xh, xs[k0:], surf[k0:, 1]))
    if surf[k - 1, 0] == xh:
        fixed = surf[:k].copy()
        corner = surf[k - 1].copy()
    else:
        corner = np.array([xh, y_corner])
        fixed = np.vstack([surf[:k], corner])
    moved = _rotate_points(np.vstack([corner, surf[k:]]), hinge, cos_t, sin_t)
    if compressed:
        hit = _crossing(fixed, moved)
        if hit is not None:
            i, j, point = hit
            return _without_close_dups(np.vstack([fixed[: i + 1], point, moved[j + 1:]]))
        j, point = _hinge_line_cut(moved, xh)
        return _without_close_dups(np.vstack([fixed, point, moved[j + 1:]]))
    n = n_arc if n_arc is not None else _default_arc_count(fixed, moved, hinge)
    arc = _hinge_arc(fixed[-1], moved[0], hinge, n)
    return _without_close_dups(np.vstack([fixed, arc[1:-1], moved]))


def deflect_plain_flap(coords: np.ndarray, x_hinge: float, deflection: float,
                       y_hinge: float = 0.5, n_arc: int | None = None) -> np.ndarray:
    """Deflect a plain flap by rotating the points aft of the hinge.

    Parameters
    ----------
    coords : numpy.ndarray
        Airfoil coordinates [chord units], shape (N, 2), in Selig order.
    x_hinge : float
        Hinge chord station [chord units], in (0.05, 0.98).
    deflection : float
        Flap deflection [deg], positive when the trailing edge moves
        down. Values satisfy |deflection| <= 60.
    y_hinge : float
        Hinge height as a fraction [-] of the local thickness at
        ``x_hinge``, from the lower surface (0) to the upper surface
        (1). Default 0.5 (mid-thickness).
    n_arc : int or None
        Point count [-] of the filler arc on the stretched side,
        including both end points. None selects a count with arc panel
        lengths near the local panel length. Minimum 2.

    Returns
    -------
    numpy.ndarray
        Deflected coordinates [chord units] in Selig order, a simple
        polygon with a positive area. The trailing-edge points are the
        rotated input points.

    Raises
    ------
    ValueError
        If an input is out of range, the points are not in Selig order,
        or a surface has decreasing x at or aft of the hinge station.
    AirfoilGeometryError
        If the deflected section is not a simple polygon with a positive
        area (a subclass of ValueError).
    """
    arr = _as_coords(coords)
    xh = _check_station(x_hinge)
    delta = _check_deflection(deflection)
    yh = _check_fraction(y_hinge)
    n = _check_arc_count(n_arc)
    if delta == 0.0:
        return arr.copy()
    upper, lower = _surfaces(arr)
    hinge = _hinge_from_surfaces(upper, lower, xh, yh)
    theta = -math.radians(delta)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    upper_new = _deflect_surface(upper, xh, hinge, cos_t, sin_t, n, compressed=delta < 0.0)
    lower_new = _deflect_surface(lower, xh, hinge, cos_t, sin_t, n, compressed=delta > 0.0)
    out = np.vstack([upper_new[::-1], lower_new[1:]])
    _check_output(out, xh, delta, yh)
    return out


def _check_output(out: np.ndarray, xh: float, delta: float, yh: float) -> None:
    """Raise AirfoilGeometryError when the output is not a simple CCW polygon."""
    x = out[:, 0]
    y = out[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    pair = polygon_self_intersection(out)
    if area <= 0.0 or pair is not None:
        where = "" if pair is None else f" (segments {pair[0]} and {pair[1]} cross)"
        raise AirfoilGeometryError(
            f"The plain flap at x_hinge={xh:g}, deflection={delta:g} deg, "
            f"y_hinge={yh:g} gives a section that is not a simple polygon with "
            f"a positive area{where}. Use finer input points or a smaller deflection."
        )
