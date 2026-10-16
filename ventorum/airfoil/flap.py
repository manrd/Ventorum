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

Hinge treatments: the option ``hinge_treatment`` selects what happens to
the nodes near the hinge after the construction above.

* "none" (default): the outline stays as constructed. At small
  deflections the hinge-line joint on the compressed side can be a very
  short panel between two corners of about 90 deg, and the inserted
  hinge-station points can make short panels next to the input points.
* "merge": nodes in the hinge zone (|x - x_hinge| < 0.05 chord units,
  never the trailing-edge or the leading-edge nodes) are removed by two
  rules. Rule 1: no panel with a removable end is shorter than 0.5 times
  the mean length of its two neighbour panels; of the two ends of a
  short panel, the node whose removal leaves the smaller largest turning
  angle at its neighbours goes, so the removal also smooths the corner
  (at equal angles, the node nearest to the line through its own
  neighbours). Rule 2: a node with a
  turning angle above 5 deg is removed when this lowers the largest
  turning angle of the node and its two neighbours and the new panel is
  at most 2 times the mean length of its neighbour panels.
  The kept nodes do not move, so the outline changes only by the chords
  that replace the removed nodes. The panel count goes down by the
  number of removed nodes.
* "repanel": rule 1 of "merge" removes the short panels (the two hinge
  nodes are kept), then a parametric not-a-knot cubic spline
  (:class:`ventorum.airfoil.paneling.AirfoilSpline`) goes through the
  remaining points, and the nodes are distributed again along it with
  the weight function of :func:`ventorum.airfoil.paneling.repanel` in one
  pass. The trailing-edge points, the leading edge of the spline and the
  hinge node of each surface (the crossing point on a cut side, else the
  point of the surface at the hinge station) stay nodes. The panel count
  is the input count, or ``n_panels``. The spline rounds the corners at
  the hinge, so the surface near the hinge moves; the theory page
  ``docs/theory/airfoil_flaps.md`` gives the tested bound of the
  deviation from the deflected outline.

All treatments keep the trailing-edge points as the exact rotated input
points and check that the result is a simple polygon with a positive
area. "none" and "merge" keep the fixed part (x < x_hinge - 0.05)
unchanged.

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
* C. de Boor, A Practical Guide to Splines, Springer, 1978, chapter IV:
  the not-a-knot cubic spline of the "repanel" treatment.
"""

from __future__ import annotations

import math
import operator

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

from ventorum.airfoil.geometry import (
    AirfoilGeometryError,
    leading_edge_index,
    polygon_self_intersection,
)

__all__ = ["HINGE_TREATMENTS", "deflect_plain_flap", "flap_hinge_point"]

#: Names of the hinge treatments of :func:`deflect_plain_flap` [-].
HINGE_TREATMENTS = ("none", "repanel", "merge")

_X_HINGE_MIN = 0.05
_X_HINGE_MAX = 0.98
_MAX_DEFLECTION_DEG = 60.0
_DEDUPE_TOL = 1e-14
_CROSS_EPS = 1e-9
#: Half width of the hinge zone, and the margin of the fixed part [chord units].
_FIXED_MARGIN = 0.05
#: Smallest panel count of the "repanel" treatment [-] (one panel per part).
_MIN_PANELS = 4
#: Weight function values of the "repanel" treatment [-] (the defaults of
#: :func:`ventorum.airfoil.paneling.repanel`).
_LE_REFINEMENT = 4.0
_TE_REFINEMENT = 3.0
_CURVATURE_WEIGHT = 2.0
#: "merge": smallest panel length over the mean length of its neighbours [-].
_MERGE_MIN_FRACTION = 0.5
#: "merge": turning angle above which a corner node is a candidate [deg].
_MERGE_MAX_TURN_DEG = 5.0
#: "merge": largest new panel length over the mean length of its neighbours [-].
_MERGE_MAX_STRETCH = 2.0


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
    """Rotate points about the hinge by the angle of (cos_t, sin_t).

    A zero angle returns a copy, so the points do not change by round-off.
    """
    if cos_t == 1.0 and sin_t == 0.0:
        return pts.copy()
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
                     compressed: bool) -> tuple[np.ndarray, np.ndarray]:
    """Deflect one surface given from the leading to the trailing edge.

    ``compressed`` is True for the side that the rotation compresses
    (the lower surface for trailing edge down). That side is cut; the
    other side gets the filler arc.

    Returns the deflected surface [chord units] and its hinge node
    [chord units]: the crossing point on a cut side, else the point of
    the fixed surface at the hinge station.
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
            return _without_close_dups(np.vstack([fixed[: i + 1], point, moved[j + 1:]])), point
        j, point = _hinge_line_cut(moved, xh)
        return _without_close_dups(np.vstack([fixed, point, moved[j + 1:]])), fixed[-1]
    n = n_arc if n_arc is not None else _default_arc_count(fixed, moved, hinge)
    arc = _hinge_arc(fixed[-1], moved[0], hinge, n)
    return _without_close_dups(np.vstack([fixed, arc[1:-1], moved])), fixed[-1]


def _check_treatment(hinge_treatment: str) -> str:
    """Return the hinge treatment name [-] after a check."""
    if not isinstance(hinge_treatment, str) or hinge_treatment not in HINGE_TREATMENTS:
        raise ValueError(
            f"hinge_treatment must be one of {', '.join(repr(t) for t in HINGE_TREATMENTS)}; "
            f"got {hinge_treatment!r}."
        )
    return hinge_treatment


def _check_panel_count(n_panels: int | None, treatment: str) -> int | None:
    """Return the panel count [-] of the repanel treatment after checks."""
    if n_panels is None:
        return None
    if treatment != "repanel":
        raise ValueError("n_panels applies only to hinge_treatment='repanel'.")
    try:
        n = operator.index(n_panels)
    except TypeError:
        raise ValueError(f"n_panels must be an integer of at least {_MIN_PANELS}.") from None
    if n < _MIN_PANELS:
        raise ValueError(f"n_panels must be an integer of at least {_MIN_PANELS}.")
    return n


def deflect_plain_flap(coords: np.ndarray, x_hinge: float, deflection: float,
                       y_hinge: float = 0.5, n_arc: int | None = None,
                       hinge_treatment: str = "none",
                       n_panels: int | None = None) -> np.ndarray:
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
    hinge_treatment : {"none", "repanel", "merge"}
        Treatment of the nodes near the hinge after the deflection [-].
        "none" (default) keeps the deflected outline as constructed.
        "repanel" fits a spline through the deflected outline and
        redistributes all nodes on it, with a node kept at the hinge
        node of each surface, at the leading edge and at the trailing
        edge. "merge" removes nodes near the hinge until no panel there
        is short and no isolated sharp corner remains; the kept nodes
        do not move. The module docstring gives the rules and limits.
    n_panels : int or None
        Panel count [-] of the "repanel" treatment, at least 4. None
        keeps the panel count of the input (N - 1). Only valid with
        ``hinge_treatment="repanel"``.

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

    Notes
    -----
    A zero deflection returns a copy of the input with "none". With
    "repanel" and "merge", a zero deflection applies the treatment to
    the input with a node added at the hinge station of each surface, so
    the result changes continuously with the deflection.
    """
    arr = _as_coords(coords)
    xh = _check_station(x_hinge)
    delta = _check_deflection(deflection)
    yh = _check_fraction(y_hinge)
    n = _check_arc_count(n_arc)
    treatment = _check_treatment(hinge_treatment)
    n_rep = _check_panel_count(n_panels, treatment)
    if delta == 0.0 and treatment == "none":
        return arr.copy()
    upper, lower = _surfaces(arr)
    hinge = _hinge_from_surfaces(upper, lower, xh, yh)
    theta = -math.radians(delta)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    upper_new, hinge_up = _deflect_surface(upper, xh, hinge, cos_t, sin_t, n, compressed=delta < 0.0)
    lower_new, hinge_lo = _deflect_surface(lower, xh, hinge, cos_t, sin_t, n, compressed=delta > 0.0)
    out = np.vstack([upper_new[::-1], lower_new[1:]])
    _check_output(out, xh, delta, yh)
    if treatment == "none":
        return out
    i_le = upper_new.shape[0] - 1
    k_up = int(np.argmin(np.linalg.norm(out[: i_le + 1] - hinge_up, axis=1)))
    k_lo = i_le + int(np.argmin(np.linalg.norm(out[i_le:] - hinge_lo, axis=1)))
    if treatment == "repanel":
        count = arr.shape[0] - 1 if n_rep is None else n_rep
        kept = _merge_hinge_nodes(out, xh, protect=(k_up, k_lo), corners=False)
        result = _repanel_deflected(out[kept], (kept.index(k_up), kept.index(k_lo)), xh, count)
    else:
        result = out[_merge_hinge_nodes(out, xh)]
    _check_output(result, xh, delta, yh)
    return result


def _split_at_breaks(grid: np.ndarray, weight: np.ndarray, breaks: np.ndarray,
                     n_panels: int) -> tuple[np.ndarray, np.ndarray]:
    """Equidistribute the weight between fixed arc-length breakpoints.

    Each part between two breakpoints gets a panel count proportional to
    its weight integral (largest remainder rounding, at least one panel
    per part). Each part splits its integral into equal pieces. Returns
    the node positions [chord units] and the node index of each
    breakpoint [-].
    """
    step = np.diff(grid)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * step)])
    inverse = PchipInterpolator(cum, grid)
    w_b = np.interp(breaks, grid, cum)
    parts = np.diff(w_b)
    raw = n_panels * parts / float(np.sum(parts))
    counts = np.maximum(1, np.round(raw)).astype(int)
    while int(np.sum(counts)) < n_panels:
        counts[int(np.argmax(raw - counts))] += 1
    while int(np.sum(counts)) > n_panels:
        spare = np.where(counts > 1, raw - counts, np.inf)
        counts[int(np.argmin(spare))] -= 1
    pieces = []
    for p in range(parts.size):
        s = np.asarray(inverse(np.linspace(w_b[p], w_b[p + 1], counts[p] + 1)), dtype=float)
        s[0] = breaks[p]
        s[-1] = breaks[p + 1]
        pieces.append(s if p == 0 else s[1:])
    return np.concatenate(pieces), np.concatenate([[0], np.cumsum(counts)])


def _repanel_deflected(out: np.ndarray, keep: tuple[int, int], xh: float,
                       n_panels: int) -> np.ndarray:
    """Repanel a deflected section on the spline through its points.

    The parametric not-a-knot cubic spline of
    :class:`ventorum.airfoil.paneling.AirfoilSpline` (de Boor, 1978,
    chapter IV) passes through every point of ``out`` [chord units] (the
    deflected outline after rule 1 of :func:`_merge_hinge_nodes`). The
    new nodes equidistribute the weight
    function of :func:`ventorum.airfoil.paneling.repanel` (default
    refinement values) along the arc length of this spline, in one pass.
    The breakpoints of the equidistribution are the trailing-edge ends,
    the two hinge nodes ``keep`` (indices into ``out``) and the spline
    leading edge, so these points stay nodes. The hinge and the
    trailing-edge nodes are the exact input points.

    The curvature term of the weight uses the peak curvature of the
    fixed part (x < xh - 0.05) as reference and is clipped at 1, so the
    corners at the hinge get the node density of the leading edge and do
    not change the density elsewhere. The weight is smoothed until the
    adjacent panel length ratio is at most 1.5, as in ``repanel``.
    The repeated passes of ``repanel`` are not used: each pass fits a new
    spline through the last nodes, and at the hinge corners this moves
    the surface away from the deflected outline.
    """
    from ventorum.airfoil.paneling import (
        _FINE_GRID,
        _MAX_RATIO,
        _MAX_SIGMA,
        _MAX_SMOOTHING,
        AirfoilSpline,
        _adjacent_ratio,
        _weight,
    )

    spline = AirfoilSpline(out)
    s_pts = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(out, axis=0), axis=1))])
    length = spline.length
    s_le = spline.s_leading_edge
    k_up, k_lo = keep
    breaks = np.array([0.0, s_pts[k_up], s_le, s_pts[k_lo], length])
    if not np.all(np.diff(breaks) > 0.0):
        raise AirfoilGeometryError(
            "The hinge nodes and the leading edge are not in Selig order on the deflected section."
        )
    grid = np.linspace(0.0, length, _FINE_GRID)
    kappa = np.asarray(spline.curvature(grid), dtype=float)
    fixed = spline(grid)[:, 0] < xh - _FIXED_MARGIN
    ref = float(np.max(kappa[fixed])) if np.any(fixed) else float(np.max(kappa))
    curvature_norm = np.minimum(kappa / ref, 1.0) if ref > 0.0 else np.zeros_like(kappa)
    weight = _weight(grid, curvature_norm, s_le, length, _LE_REFINEMENT, _TE_REFINEMENT,
                     _CURVATURE_WEIGHT)
    node_s, at = _split_at_breaks(grid, weight, breaks, n_panels)
    sigma = 2.0
    for _ in range(_MAX_SMOOTHING):
        if _adjacent_ratio(spline(node_s)) <= _MAX_RATIO:
            break
        mean = float(np.mean(weight))
        if float(np.ptp(weight)) <= 1e-12 * mean:
            break
        if sigma > _MAX_SIGMA:
            weight = np.full_like(weight, mean)
        else:
            smooth = gaussian_filter1d(weight, sigma)
            weight = 0.5 * (smooth + float(np.mean(smooth)))
            sigma *= 1.5
        node_s, at = _split_at_breaks(grid, weight, breaks, n_panels)
    nodes = spline(node_s)
    nodes[at[0]] = out[0]
    nodes[at[1]] = out[k_up]
    nodes[at[3]] = out[k_lo]
    nodes[at[4]] = out[-1]
    return nodes


def _turn(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Signed turning angle [rad] at p1 of the path p0, p1, p2 (left turn positive)."""
    a = p1 - p0
    b = p2 - p1
    return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


def _chord_distance(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Distance [chord units] from p1 to the line through p0 and p2."""
    d = p2 - p0
    norm = math.hypot(d[0], d[1])
    if norm == 0.0:
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    return abs(d[0] * (p1[1] - p0[1]) - d[1] * (p1[0] - p0[0])) / norm


def _turn_after_removal(pts: np.ndarray, p: int) -> float:
    """Largest turning angle [rad] at the neighbours of node p after its removal."""
    m = pts.shape[0]
    worst = 0.0
    if p - 2 >= 0:
        worst = max(worst, abs(_turn(pts[p - 2], pts[p - 1], pts[p + 1])))
    if p + 2 <= m - 1:
        worst = max(worst, abs(_turn(pts[p - 1], pts[p + 1], pts[p + 2])))
    return worst


def _neighbour_mean(lengths: np.ndarray, i: int) -> float:
    """Mean length [chord units] of the panels next to panel i."""
    near = [lengths[j] for j in (i - 1, i + 1) if 0 <= j < lengths.size]
    return float(np.mean(near))


def _merge_hinge_nodes(out: np.ndarray, xh: float, protect: tuple[int, ...] = (),
                       corners: bool = True) -> list[int]:
    """Return the indices of the nodes that stay after the merge rules.

    The kept nodes do not move.

    The removable nodes lie in the hinge zone |x - xh| < 0.05 [chord
    units]; the trailing-edge nodes and the leading-edge node are never
    removed. Two rules apply in turn until neither removes a node:

    1. Short panels. A panel with a removable end is short when its
       length is less than ``_MERGE_MIN_FRACTION`` (0.5) times the mean
       length of its two neighbour panels. The shortest panel (by this
       ratio) loses the end node whose removal leaves the smaller
       largest turning angle at its neighbours, so the removal also
       smooths the corner; at equal angles, the end node nearest to the
       line through its own two neighbours goes.
    2. Corners. A removable node whose turning angle is larger than
       ``_MERGE_MAX_TURN_DEG`` (5 deg) is removed when this lowers the
       largest turning angle of the node and its two neighbours and the
       new panel is at most ``_MERGE_MAX_STRETCH`` (2) times the mean
       length of its neighbour panels. The node with the largest turning
       angle is tried first; one node goes per pass, then rule 1 runs
       again.

    A removal that would make the polygon touch or cross itself is not
    done. The nodes in ``protect`` (indices into ``out``) are never
    removed. ``corners=False`` applies rule 1 only.
    """
    n = out.shape[0]
    le = leading_edge_index(out)
    removable = (np.abs(out[:, 0] - xh) < _FIXED_MARGIN)
    removable[[0, n - 1, le, *protect]] = False
    idx = list(range(n))
    min_fraction = _MERGE_MIN_FRACTION
    max_turn = math.radians(_MERGE_MAX_TURN_DEG)

    def simple_without(pos: int) -> bool:
        trial = out[idx[:pos] + idx[pos + 1:]]
        return polygon_self_intersection(trial) is None

    while True:
        removed = False
        # Rule 1: short panels.
        pts = out[idx]
        lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        short = []
        for i in range(lengths.size):
            if not (removable[idx[i]] or removable[idx[i + 1]]):
                continue
            ratio = lengths[i] / _neighbour_mean(lengths, i)
            if ratio < min_fraction:
                short.append((ratio, i))
        for _, i in sorted(short):
            ends = [p for p in (i, i + 1) if removable[idx[p]]]
            ends.sort(key=lambda p: (round(_turn_after_removal(pts, p), 12),
                                     _chord_distance(pts[p - 1], pts[p], pts[p + 1])))
            for p in ends:
                if simple_without(p):
                    del idx[p]
                    removed = True
                    break
            if removed:
                break
        if removed:
            continue
        if not corners:
            break
        # Rule 2: isolated sharp corners.
        pts = out[idx]
        m = len(idx)
        turns = np.zeros(m)
        for p in range(1, m - 1):
            turns[p] = _turn(pts[p - 1], pts[p], pts[p + 1])
        order = sorted((p for p in range(1, m - 1)
                        if removable[idx[p]] and abs(turns[p]) > max_turn),
                       key=lambda p: -abs(turns[p]))
        for p in order:
            before = max(abs(turns[p - 1]), abs(turns[p]), abs(turns[p + 1]))
            after = 0.0
            if p - 1 >= 1:
                after = max(after, abs(_turn(pts[p - 2], pts[p - 1], pts[p + 1])))
            if p + 1 <= m - 2:
                after = max(after, abs(_turn(pts[p - 1], pts[p + 1], pts[p + 2])))
            if after >= before:
                continue
            new_len = float(np.linalg.norm(pts[p + 1] - pts[p - 1]))
            near = [float(np.linalg.norm(pts[q + 1] - pts[q])) for q in (p - 2, p + 1) if 0 <= q < m - 1]
            if new_len > _MERGE_MAX_STRETCH * float(np.mean(near)):
                continue
            if simple_without(p):
                del idx[p]
                removed = True
                break
        if not removed:
            break
    return idx


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
