# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Rotate the aft part of airfoil coordinates about a plain-flap hinge.

A plain flap is the aft part of an airfoil section behind a hinge
station. This module deflects it as a rigid body. It rotates every point
aft of the hinge about the hinge point. The fixed part keeps its points.
The side that the rotation compresses is cut at the crossing of the fixed
and the moved surface, and the overlapping points are removed. The side
that the rotation stretches gets a circular arc about the hinge that
joins the fixed and the moved surface. The result is a simple polygon in
Selig order.

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

__all__ = ["deflect_plain_flap", "flap_hinge_point"]

_X_HINGE_MIN = 0.05
_X_HINGE_MAX = 0.98
_MAX_DEFLECTION_DEG = 60.0
_FIXED_MARGIN = 0.05
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


def _leading_edge(arr: np.ndarray) -> int:
    """Return the index of the leading edge (minimum x [chord units])."""
    le = int(np.argmin(arr[:, 0]))
    if le == 0 or le == arr.shape[0] - 1:
        raise ValueError("coords must run from the trailing edge to the leading edge and back.")
    return le


def _check_monotonic(xs: np.ndarray) -> None:
    """Raise when the surface abscissa is not non-decreasing."""
    if not bool(np.all(np.diff(xs) >= 0.0)):
        raise ValueError("Each surface must have non-decreasing x from leading to trailing edge.")


def flap_hinge_point(coords: np.ndarray, x_hinge: float, y_hinge: float = 0.5) -> np.ndarray:
    """Return the flap hinge point [chord units].

    Parameters
    ----------
    coords : numpy.ndarray
        Airfoil coordinates [chord units], shape (N, 2), in Selig order.
    x_hinge : float
        Hinge chord station [chord units], in (0.05, 0.98).
    y_hinge : float
        Hinge height as a fraction [-] of the local thickness, from the
        lower surface (0) to the upper surface (1). Default 0.5.

    Returns
    -------
    numpy.ndarray
        Hinge point [chord units], shape (2,).
    """
    arr = _as_coords(coords)
    xh = _check_station(x_hinge)
    yh = _check_fraction(y_hinge)
    le = _leading_edge(arr)
    upper = arr[: le + 1][::-1]
    lower = arr[le:]
    _check_monotonic(upper[:, 0])
    _check_monotonic(lower[:, 0])
    y_up = float(np.interp(xh, upper[:, 0], upper[:, 1]))
    y_lo = float(np.interp(xh, lower[:, 0], lower[:, 1]))
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
    best = None
    for i in range(fixed.shape[0] - 1):
        p1 = fixed[i]
        d = fixed[i + 1] - p1
        for j in range(moved.shape[0] - 1):
            p3 = moved[j]
            e = moved[j + 1] - p3
            if (max(p1[0], fixed[i + 1, 0]) < min(p3[0], moved[j + 1, 0])
                    or max(p3[0], moved[j + 1, 0]) < min(p1[0], fixed[i + 1, 0])
                    or max(p1[1], fixed[i + 1, 1]) < min(p3[1], moved[j + 1, 1])
                    or max(p3[1], moved[j + 1, 1]) < min(p1[1], fixed[i + 1, 1])):
                continue
            denom = d[0] * e[1] - d[1] * e[0]
            if denom == 0.0:
                continue
            qx = p3[0] - p1[0]
            qy = p3[1] - p1[1]
            t = (qx * e[1] - qy * e[0]) / denom
            u = (qx * d[1] - qy * d[0]) / denom
            if t < -_CROSS_EPS or t > 1.0 + _CROSS_EPS or u < -_CROSS_EPS or u > 1.0 + _CROSS_EPS:
                continue
            point = p1 + min(1.0, max(0.0, t)) * d
            if best is None or i > best[0] or (i == best[0] and j < best[1]):
                best = (i, j, point)
    return best


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
                     cos_t: float, sin_t: float, n_arc: int | None) -> np.ndarray:
    """Deflect one surface given from the leading to the trailing edge."""
    xs = surf[:, 0]
    _check_monotonic(xs)
    k = int(np.searchsorted(xs, xh, side="right"))
    y_corner = float(np.interp(xh, xs, surf[:, 1]))
    if k > 0 and surf[k - 1, 0] == xh:
        fixed = surf[:k].copy()
        corner = surf[k - 1].copy()
    else:
        corner = np.array([xh, y_corner])
        fixed = np.vstack([surf[:k], corner])
    moved = _rotate_points(np.vstack([corner, surf[k:]]), hinge, cos_t, sin_t)
    hit = _crossing(fixed, moved)
    if hit is not None:
        i, j, point = hit
        return _without_close_dups(np.vstack([fixed[: i + 1], point, moved[j + 1:]]))
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
        Hinge height as a fraction [-] of the local thickness, from the
        lower surface (0) to the upper surface (1). Default 0.5.
    n_arc : int or None
        Point count [-] of the filler arc on the stretched side,
        including both end points. None selects a count with arc panel
        lengths near the local panel length. Minimum 2.

    Returns
    -------
    numpy.ndarray
        Deflected coordinates [chord units] in Selig order, a simple
        polygon. The trailing-edge points are the rotated input points.
    """
    arr = _as_coords(coords)
    xh = _check_station(x_hinge)
    delta = _check_deflection(deflection)
    yh = _check_fraction(y_hinge)
    n = _check_arc_count(n_arc)
    if delta == 0.0:
        return arr.copy()
    hinge = flap_hinge_point(arr, xh, yh)
    theta = -math.radians(delta)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    le = _leading_edge(arr)
    upper = arr[: le + 1][::-1].copy()
    lower = arr[le:].copy()
    upper_new = _deflect_surface(upper, xh, hinge, cos_t, sin_t, n)
    lower_new = _deflect_surface(lower, xh, hinge, cos_t, sin_t, n)
    return np.vstack([upper_new[::-1], lower_new[1:]])
