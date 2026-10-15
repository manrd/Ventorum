# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Parametric airfoil spline and panel node distributions.

This module fits a parametric cubic spline through airfoil coordinates and
redistributes the nodes for panel methods. The nodes cluster at the leading
edge (high curvature) and stay refined at the trailing edge. The node density
follows the equidistribution of a weight function along arc length: the nodes
split the cumulative weight integral into equal parts. This equidistribution
principle is a standard method of numerical grid generation.

Sources: de Boor, C., 1978, A Practical Guide to Splines, Springer. The cubic
spline uses not-a-knot end conditions (Chapter IV). The curvature of a
parametric plane curve uses the standard differential geometry formula given
in the same text. The cosine spacing uses the standard Chebyshev node
distribution.

Coordinate convention: arrays of shape (N, 2) in Selig order, from the
trailing edge along the upper surface to the leading edge and back along the
lower surface (counterclockwise, positive signed area). Coordinates use m.
The common unit-chord case places the leading edge at (0, 0) m and the
trailing-edge midpoint at (1, 0) m.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq

#: Samples of the weight function along arc length.
_FINE_GRID = 4096
#: Samples of the tangency function in the leading-edge search.
_LE_SEARCH_GRID = 2049
#: End margin excluded from the leading-edge search (fraction of length).
_LE_END_MARGIN = 0.005
#: Closed trailing edge tolerance (fraction of chord).
_CLOSED_RTOL = 1e-9
#: Gaussian width at the leading edge (fraction of total arc length).
_LE_SIGMA = 0.05
#: Gaussian width at each trailing-edge end (fraction of total arc length).
_TE_SIGMA = 0.03
#: Maximum allowed adjacent panel length ratio.
_MAX_RATIO = 1.5
#: Maximum weight smoothing iterations in one equidistribution pass.
_MAX_SMOOTHING = 31
#: Maximum fixed-point passes in `repanel`.
_MAX_PASSES = 12
#: Pass-to-pass stop tolerance: maximum node shift in m.
_FIXED_TOL = 1e-9


def _as_array(coords: np.ndarray) -> np.ndarray:
    """Copy of the input as a finite float array of shape (N, 2)."""
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if arr.shape[0] < 4:
        raise ValueError("coords needs at least four points")
    if not np.all(np.isfinite(arr)):
        raise ValueError("coords must hold finite values")
    return arr.copy()


def _signed_area(coords: np.ndarray) -> float:
    """Signed polygon area in m^2 (positive for counterclockwise order)."""
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _require_selig(coords: np.ndarray) -> None:
    """Raise ValueError unless coords run counterclockwise in Selig order."""
    if _signed_area(coords) <= 0.0:
        raise ValueError(
            "coords must run counterclockwise in Selig order "
            "(trailing edge to leading edge along the upper surface first)"
        )


def _adjacent_ratio(nodes: np.ndarray) -> float:
    """Largest length ratio of two adjacent panels (dimensionless)."""
    lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    if np.any(lengths <= 0.0):
        return float("inf")
    both = np.maximum(lengths[:-1] / lengths[1:], lengths[1:] / lengths[:-1])
    return float(np.max(both))


class AirfoilSpline:
    """Parametric cubic spline x(s), y(s) of airfoil coordinates.

    The parameter ``s`` is the cumulative arc length in m of the input
    polyline. Each component uses a not-a-knot cubic spline (de Boor, 1978,
    Chapter IV).

    Parameters
    ----------
    coords : array_like, shape (N, 2)
        Airfoil coordinates in m in Selig order (counterclockwise).

    Attributes
    ----------
    length : float
        Total arc length in m.
    s_leading_edge : float
        Arc length position in m of the leading edge.
    """

    def __init__(self, coords: np.ndarray) -> None:
        """Fit the spline and locate the leading edge."""
        pts = _as_array(coords)
        _require_selig(pts)
        self._pts = pts
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self._s = np.concatenate([[0.0], np.cumsum(seg)])
        self._spline_x = CubicSpline(self._s, pts[:, 0])
        self._spline_y = CubicSpline(self._s, pts[:, 1])
        self._te_mid = 0.5 * (pts[0] + pts[-1])
        chord = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        gap = float(np.linalg.norm(pts[0] - pts[-1]))
        self._closed = bool(gap <= _CLOSED_RTOL * max(chord, 1e-300))
        self._s_le = self._find_leading_edge()

    @property
    def length(self) -> float:
        """Total arc length in m."""
        return float(self._s[-1])

    @property
    def s_leading_edge(self) -> float:
        """Arc length position in m of the leading edge."""
        return float(self._s_le)

    @property
    def closed(self) -> bool:
        """True when the input trailing edge is closed (dimensionless)."""
        return self._closed

    def _tangency(self, s: float) -> float:
        """Tangent dot the line from the curve point to the TE midpoint."""
        px = float(self._spline_x(s))
        py = float(self._spline_y(s))
        tx = float(self._spline_x(s, 1))
        ty = float(self._spline_y(s, 1))
        return tx * (px - self._te_mid[0]) + ty * (py - self._te_mid[1])

    def _find_leading_edge(self) -> float:
        """Arc length where the tangent is normal to the TE direction.

        The condition is zero distance gradient to the trailing-edge
        midpoint. The search starts from the input point farthest from the
        trailing-edge midpoint and solves the scalar root with Brent's
        method in the sign bracket nearest to that start.
        """
        length = float(self._s[-1])
        dist = np.linalg.norm(self._pts - self._te_mid, axis=1)
        start = float(self._s[int(np.argmax(dist))])
        grid = np.linspace(0.0, length, _LE_SEARCH_GRID)
        px = self._spline_x(grid)
        py = self._spline_y(grid)
        tx = self._spline_x(grid, 1)
        ty = self._spline_y(grid, 1)
        values = tx * (px - self._te_mid[0]) + ty * (py - self._te_mid[1])
        margin = _LE_END_MARGIN * length
        best: tuple[float, float] | None = None
        best_mid = 0.0
        for i in range(grid.size - 1):
            a = float(grid[i])
            b = float(grid[i + 1])
            if b < margin or a > length - margin:
                continue
            if values[i] == 0.0:
                return a
            if values[i] * values[i + 1] < 0.0:
                mid = 0.5 * (a + b)
                if best is None or abs(mid - start) < abs(best_mid - start):
                    best = (a, b)
                    best_mid = mid
        if best is None:
            return start
        return float(brentq(self._tangency, best[0], best[1], xtol=1e-14, maxiter=100))

    def __call__(self, s: np.ndarray | float) -> np.ndarray:
        """Evaluate the spline at arc length positions.

        Parameters
        ----------
        s : float or array_like
            Arc length positions in m.

        Returns
        -------
        numpy.ndarray
            Points in m, shape (M, 2), or shape (2,) for scalar input.
        """
        arr = np.asarray(s, dtype=float)
        flat = arr.ravel()
        out = np.column_stack([self._spline_x(flat), self._spline_y(flat)])
        if arr.ndim == 0:
            return out[0]
        return out

    def derivative(self, s: np.ndarray | float, order: int = 1) -> np.ndarray:
        """Evaluate spline derivatives at arc length positions.

        Parameters
        ----------
        s : float or array_like
            Arc length positions in m.
        order : int
            Derivative order (0, 1, 2 or 3, dimensionless).

        Returns
        -------
        numpy.ndarray
            Derivatives in m**(1 - order), shape (M, 2),
            or shape (2,) for scalar input.
        """
        if order not in (0, 1, 2, 3):
            raise ValueError("order must be 0, 1, 2 or 3")
        arr = np.asarray(s, dtype=float)
        flat = arr.ravel()
        out = np.column_stack(
            [self._spline_x(flat, order), self._spline_y(flat, order)]
        )
        if arr.ndim == 0:
            return out[0]
        return out

    def curvature(self, s: np.ndarray | float) -> np.ndarray:
        """Evaluate the curvature magnitude at arc length positions.

        The curvature uses kappa = |x' y'' - y' x''| / (x'^2 + y'^2)^1.5.

        Parameters
        ----------
        s : float or array_like
            Arc length positions in m.

        Returns
        -------
        numpy.ndarray
            Curvature magnitude in 1/m, shape (M,), or scalar for scalar input.
        """
        arr = np.asarray(s, dtype=float)
        flat = arr.ravel()
        xp = self._spline_x(flat, 1)
        yp = self._spline_y(flat, 1)
        xpp = self._spline_x(flat, 2)
        ypp = self._spline_y(flat, 2)
        denom = (xp * xp + yp * yp) ** 1.5
        out = np.abs(xp * ypp - yp * xpp) / np.maximum(denom, 1e-300)
        if arr.ndim == 0:
            return out[0]
        return out


def _weight(
    grid: np.ndarray,
    curvature_norm: np.ndarray,
    s_le: float,
    length: float,
    le_refinement: float,
    te_refinement: float,
    curvature_weight: float,
) -> np.ndarray:
    """Monitor weight along arc length (dimensionless, positive)."""
    sig_le = _LE_SIGMA * length
    sig_te = _TE_SIGMA * length
    gauss_le = np.exp(-(((grid - s_le) / sig_le) ** 2))
    gauss_te = np.exp(-((grid / sig_te) ** 2)) + np.exp(
        -(((length - grid) / sig_te) ** 2)
    )
    return (
        1.0
        + curvature_weight * curvature_norm
        + le_refinement * gauss_le
        + te_refinement * gauss_te
    )


def _split_equidistribute(
    grid: np.ndarray,
    weight: np.ndarray,
    s_le: float,
    length: float,
    n_panels: int,
) -> np.ndarray:
    """Node positions from the inverse cumulative weight integral.

    The leading edge is always a node: the upper part [0, s_le] and the lower
    part [s_le, length] get node counts proportional to their weight
    integrals, and each part splits its integral into equal pieces. Arc
    length positions come out in m.
    """
    step = np.diff(grid)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * step)])
    total = float(cum[-1])
    inverse = PchipInterpolator(cum, grid)
    w_le = float(np.interp(s_le, grid, cum))
    n_up = int(round(n_panels * w_le / total))
    n_up = min(max(n_up, 1), n_panels - 1)
    n_lo = n_panels - n_up
    targets_up = np.linspace(0.0, w_le, n_up + 1)
    s_up = np.asarray(inverse(targets_up), dtype=float)
    s_up[0] = 0.0
    s_up[-1] = s_le
    targets_lo = np.linspace(w_le, total, n_lo + 1)
    s_lo = np.asarray(inverse(targets_lo), dtype=float)
    s_lo[0] = s_le
    s_lo[-1] = length
    return np.concatenate([s_up, s_lo[1:]])


def _equidistribute_pass(
    spline: AirfoilSpline,
    n_panels: int,
    le_refinement: float,
    te_refinement: float,
    curvature_weight: float,
) -> np.ndarray:
    """Node set of one fit-and-equidistribute pass in m.

    Build the weight on a fine grid, split its cumulative integral at the
    leading edge, invert each part with a monotone cubic, and smooth the
    weight until the adjacent panel length ratio stays at most 1.5.
    """
    length = spline.length
    s_le = spline.s_leading_edge
    grid = np.linspace(0.0, length, _FINE_GRID)
    kappa = np.asarray(spline.curvature(grid), dtype=float)
    peak = float(np.max(kappa))
    curvature_norm = kappa / peak if peak > 0.0 else np.zeros_like(kappa)
    weight = _weight(
        grid, curvature_norm, s_le, length, le_refinement, te_refinement, curvature_weight
    )
    node_s = _split_equidistribute(grid, weight, s_le, length, n_panels)
    sigma = 2.0
    for _ in range(_MAX_SMOOTHING):
        nodes = spline(node_s)
        if _adjacent_ratio(nodes) <= _MAX_RATIO:
            break
        smooth = gaussian_filter1d(weight, sigma)
        weight = 0.5 * (smooth + float(np.mean(smooth)))
        sigma *= 1.5
        node_s = _split_equidistribute(grid, weight, s_le, length, n_panels)
    nodes = spline(node_s)
    if spline.closed:
        nodes[-1] = nodes[0]
    return nodes


def repanel(
    coords: np.ndarray,
    n_panels: int = 160,
    le_refinement: float = 4.0,
    te_refinement: float = 3.0,
    curvature_weight: float = 2.0,
) -> np.ndarray:
    """Redistribute airfoil nodes by equidistribution of a weight function.

    The node density is proportional to w(s) = 1 + curvature_weight * local
    normalized curvature, plus Gaussian refinement terms at the leading edge
    and at both trailing-edge ends. The nodes split the cumulative weight
    integral into equal parts, with the leading edge always a node. The
    weight is smoothed until the adjacent panel length ratio stays at most
    1.5 everywhere.

    The fit-and-equidistribute pass repeats through the new nodes until the
    set stops moving (fixed point of the equidistribution), so repaneling an
    already repaneled set returns the same nodes.

    Default values (le_refinement 4.0, te_refinement 3.0, curvature_weight
    2.0, Gaussian widths 5 percent and 3 percent of arc length) cluster the
    nodes at the leading edge so the smallest panels sit there, keep the
    adjacent panel length ratio below 1.5, and resolve the spline ends.

    A closed trailing edge stays closed; an open trailing edge keeps its two
    end points exactly.

    Parameters
    ----------
    coords : array_like, shape (N, 2)
        Airfoil coordinates in m in Selig order (counterclockwise).
    n_panels : int
        Number of panels (dimensionless). The output holds n_panels + 1 nodes.
    le_refinement : float
        Amplitude of the Gaussian refinement at the leading edge
        (dimensionless, nonnegative).
    te_refinement : float
        Amplitude of the Gaussian refinement at each trailing-edge end
        (dimensionless, nonnegative).
    curvature_weight : float
        Weight of the normalized curvature term (dimensionless, nonnegative).

    Returns
    -------
    numpy.ndarray
        Paneled coordinates in m, shape (n_panels + 1, 2), in Selig order.
    """
    if n_panels < 2:
        raise ValueError("n_panels must be at least 2")
    if le_refinement < 0.0 or te_refinement < 0.0 or curvature_weight < 0.0:
        raise ValueError("refinement weights must be nonnegative")
    nodes = _equidistribute_pass(
        AirfoilSpline(coords), n_panels, le_refinement, te_refinement, curvature_weight
    )
    for _ in range(_MAX_PASSES):
        update = _equidistribute_pass(
            AirfoilSpline(nodes),
            n_panels,
            le_refinement,
            te_refinement,
            curvature_weight,
        )
        shift = float(np.max(np.linalg.norm(update - nodes, axis=1)))
        nodes = update
        if shift <= _FIXED_TOL:
            break
    return nodes


def repanel_cosine(coords: np.ndarray, n_per_side: int) -> np.ndarray:
    """Resample each surface with cosine spacing in arc length.

    Each side (upper and lower) gets ``n_per_side`` panels with Chebyshev
    cosine nodes measured in arc length from the leading edge, dense at the
    leading edge and at the trailing edge. The leading edge is always a node.

    A closed trailing edge stays closed; an open trailing edge keeps its two
    end points exactly.

    Parameters
    ----------
    coords : array_like, shape (N, 2)
        Airfoil coordinates in m in Selig order (counterclockwise).
    n_per_side : int
        Panels per surface (dimensionless, at least 1).

    Returns
    -------
    numpy.ndarray
        Paneled coordinates in m, shape (2 * n_per_side + 1, 2), in Selig order.
    """
    if n_per_side < 1:
        raise ValueError("n_per_side must be at least 1")
    spline = AirfoilSpline(coords)
    length = spline.length
    s_le = spline.s_leading_edge
    beta = np.linspace(0.0, np.pi, n_per_side + 1)
    frac = 0.5 * (1.0 - np.cos(beta))
    s_up = frac * s_le
    s_lo = s_le + frac * (length - s_le)
    node_s = np.concatenate([s_up, s_lo[1:]])
    nodes = spline(node_s)
    if spline.closed:
        nodes[-1] = nodes[0]
    return nodes
