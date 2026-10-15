# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Airfoil section shape: storage, checks and section properties.

Coordinate convention
---------------------
A shape is a numpy array of shape (N, 2), float64, in Selig order: from the
trailing edge along the upper surface to the leading edge, then along the
lower surface back to the trailing edge (counterclockwise, positive signed
area). Unit chord has the leading edge at (0, 0) and the trailing-edge
midpoint at (1, 0). A closed trailing edge repeats the same point at the
start and the end.

Methods
-------
The section properties use linear interpolation between the stored points.
The signed area uses the shoelace formula. The self-intersection test uses
the orientation test of two segments. The thickness distribution of a
symmetric NACA 4-digit section follows the thickness equation of Abbott
and von Doenhoff.

References
----------
- M. S. Selig, J. J. Guglielmo, A. P. Broeren and P. Giguere, *Summary of
  Low-Speed Airfoil Data*, Vol. 1, SoarTech Publications, 1995 (Selig
  coordinate file format: name line, then x y pairs from the trailing edge).
- D. Lednicer, "A CFD evaluation of wind tunnel test sections", AIAA
  paper 2010-0458, 2010 (Lednicer coordinate file format: name line, point
  counts of the two surfaces, upper surface then lower surface).
- I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, Dover,
  1959, chapter 6 (NACA 4-digit thickness equation).
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge
  University Press, 2001, chapter 11 (airfoil geometry and thickness
  and camber definitions).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ventorum.core.errors import VentorumError


class AirfoilGeometryError(VentorumError):
    """Bad airfoil geometry or airfoil file (a ValueError for compatibility)."""


@dataclass
class AirfoilShape:
    """One airfoil section as an ordered polygon.

    Parameters
    ----------
    name : str
        Descriptive label of the section.
    coords : numpy.ndarray
        Point coordinates [chord units], shape (N, 2), in Selig order
        (counterclockwise from the trailing edge).
    """

    name: str = "airfoil"
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    def __post_init__(self) -> None:
        """Convert the coordinates to a float64 copy and check the shape."""
        arr = np.array(self.coords, dtype=np.float64, copy=True)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise AirfoilGeometryError(
                f"Airfoil coordinates must have shape (N, 2); got {arr.shape}."
            )
        if arr.shape[0] == 0:
            raise AirfoilGeometryError("Airfoil needs at least one point; got none.")
        if not np.all(np.isfinite(arr)):
            raise AirfoilGeometryError("Airfoil coordinates must be finite numbers.")
        self.coords = arr
        self.name = str(self.name)

    @property
    def n_points(self) -> int:
        """Number of stored points."""
        return int(self.coords.shape[0])

    @property
    def te_gap(self) -> float:
        """Distance between the first and the last point [chord units]."""
        return float(np.linalg.norm(self.coords[0] - self.coords[-1]))

    @property
    def _te_midpoint(self) -> np.ndarray:
        """Midpoint of the trailing edge [chord units]."""
        return 0.5 * (self.coords[0] + self.coords[-1])

    @property
    def is_closed(self) -> bool:
        """True when the first point equals the last point to 1e-12 chord."""
        chord = self.chord
        tol = 1e-12 * max(chord, 1e-300)
        return bool(self.te_gap <= tol)

    @property
    def leading_edge_index(self) -> int:
        """Index of the point farthest from the trailing-edge midpoint."""
        mid = self._te_midpoint
        dist = np.linalg.norm(self.coords - mid, axis=1)
        return int(np.argmax(dist))

    @property
    def chord(self) -> float:
        """Distance from the leading edge to the trailing-edge midpoint [chord units]."""
        mid = self._te_midpoint
        return float(np.linalg.norm(self.coords[self.leading_edge_index] - mid))

    def normalized(self) -> AirfoilShape:
        """Return a new shape with the leading edge at (0, 0) and the trailing edge at (1, 0).

        The stored points are translated, rotated and scaled. The name is kept.

        Returns
        -------
        AirfoilShape
            Normalized copy of this shape (coordinates [chord units]).
        """
        coords = self.coords
        mid = self._te_midpoint
        le = coords[self.leading_edge_index].copy()
        vec = mid - le
        length = float(np.linalg.norm(vec))
        if length <= 0.0:
            raise AirfoilGeometryError("Cannot normalize a shape with zero chord.")
        angle = float(np.arctan2(vec[1], vec[0]))
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        out = (coords - le) @ rot.T / length
        return AirfoilShape(name=self.name, coords=out)

    def _split_surfaces(self) -> tuple[np.ndarray, np.ndarray]:
        """Split the polygon into upper and lower surfaces from the leading edge.

        Each branch runs from the leading edge to one trailing-edge point.
        The branch with the larger mean y is the upper surface.
        """
        k = self.leading_edge_index
        branch_a = self.coords[: k + 1][::-1].copy()
        branch_b = self.coords[k:].copy()
        mean_a = float(np.mean(branch_a[:, 1])) if branch_a.shape[0] else -np.inf
        mean_b = float(np.mean(branch_b[:, 1])) if branch_b.shape[0] else np.inf
        if mean_a >= mean_b:
            return branch_a, branch_b
        return branch_b, branch_a

    def upper(self) -> np.ndarray:
        """Return the upper surface from the leading edge to the trailing edge.

        Returns
        -------
        numpy.ndarray
            Points [chord units], shape (M, 2), x increasing, with the
            leading-edge point first.
        """
        upper, _ = self._split_surfaces()
        return upper

    def lower(self) -> np.ndarray:
        """Return the lower surface from the leading edge to the trailing edge.

        Returns
        -------
        numpy.ndarray
            Points [chord units], shape (M, 2), x increasing, with the
            leading-edge point first.
        """
        _, lower = self._split_surfaces()
        return lower

    @staticmethod
    def _surface_y(surf: np.ndarray, xq: np.ndarray) -> np.ndarray:
        """Linear interpolation of one surface at the stations *xq* [chord units]."""
        xs = np.asarray(surf[:, 0], dtype=float)
        ys = np.asarray(surf[:, 1], dtype=float)
        order = np.argsort(xs, kind="mergesort")
        xs = xs[order]
        ys = ys[order]
        uniq_x: list[float] = []
        uniq_y: list[float] = []
        for x, y in zip(xs.tolist(), ys.tolist()):
            if uniq_x and abs(x - uniq_x[-1]) <= 1e-14:
                uniq_y[-1] = 0.5 * (uniq_y[-1] + y)
            else:
                uniq_x.append(x)
                uniq_y.append(y)
        xa = np.array(uniq_x, dtype=float)
        ya = np.array(uniq_y, dtype=float)
        if xa.size == 1:
            return np.full(np.shape(xq), float(ya[0]), dtype=float)
        return np.interp(np.asarray(xq, dtype=float), xa, ya).astype(float)

    def thickness(self, x: float | np.ndarray) -> float | np.ndarray:
        """Return the thickness at the chord stations *x* [chord units].

        Parameters
        ----------
        x : float or numpy.ndarray
            Chord stations [-], in [0, 1].

        Returns
        -------
        float or numpy.ndarray
            Upper y minus lower y [chord units], by linear interpolation
            on each surface.
        """
        scalar = np.ndim(x) == 0
        xq = np.atleast_1d(np.asarray(x, dtype=float))
        yu = self._surface_y(self.upper(), xq)
        yl = self._surface_y(self.lower(), xq)
        out = yu - yl
        return float(out[0]) if scalar else out

    def camber(self, x: float | np.ndarray) -> float | np.ndarray:
        """Return the camber at the chord stations *x* [chord units].

        Parameters
        ----------
        x : float or numpy.ndarray
            Chord stations [-], in [0, 1].

        Returns
        -------
        float or numpy.ndarray
            Mean of the upper and lower y [chord units], by linear
            interpolation on each surface.
        """
        scalar = np.ndim(x) == 0
        xq = np.atleast_1d(np.asarray(x, dtype=float))
        yu = self._surface_y(self.upper(), xq)
        yl = self._surface_y(self.lower(), xq)
        out = 0.5 * (yu + yl)
        return float(out[0]) if scalar else out

    def max_thickness(self) -> tuple[float, float]:
        """Return the largest thickness and its chord station.

        Returns
        -------
        value : float
            Largest thickness [chord units].
        x : float
            Chord station of the largest thickness [-], in [0, 1].
        """
        xfine = np.linspace(0.0, 1.0, 2001)
        t = np.asarray(self.thickness(xfine), dtype=float)
        k = int(np.argmax(t))
        return float(t[k]), float(xfine[k])

    def max_camber(self) -> tuple[float, float]:
        """Return the largest camber and its chord station.

        Returns
        -------
        value : float
            Largest camber [chord units].
        x : float
            Chord station of the largest camber [-], in [0, 1].
        """
        xfine = np.linspace(0.0, 1.0, 2001)
        c = np.asarray(self.camber(xfine), dtype=float)
        k = int(np.argmax(c))
        return float(c[k]), float(xfine[k])

    def area(self) -> float:
        """Return the signed polygon area [chord^2] (positive counterclockwise)."""
        x = self.coords[:, 0]
        y = self.coords[:, 1]
        return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def problems(self) -> list[str]:
        """List the geometry defects of this shape (empty when the shape is good).

        Checks: fewer than 10 points; repeated consecutive points; clockwise
        order (negative area); self-intersection of the polygon; a surface
        whose x is not monotonic (a note with the word monotonic, not an error).
        """
        out: list[str] = []
        n = self.n_points
        if n < 10:
            out.append(f"only {n} points: fewer than 10 points")
        diffs = np.linalg.norm(np.diff(self.coords, axis=0), axis=1)
        for i, d in enumerate(diffs.tolist()):
            if d <= 1e-12:
                out.append(f"repeated consecutive points at index {i} and {i + 1}")
        if self.area() < 0.0:
            out.append(f"clockwise order (signed area {self.area():.6g} is negative)")
        pair = self._self_intersection()
        if pair is not None:
            i, j = pair
            out.append(f"self-intersection between segments {i}-{i + 1} and {j}-{j + 1}")
        upper, lower = self._split_surfaces()
        if upper.shape[0] >= 2 and np.any(np.diff(upper[:, 0]) < -1e-12):
            out.append("note: non-monotonic x on the upper surface")
        if lower.shape[0] >= 2 and np.any(np.diff(lower[:, 0]) < -1e-12):
            out.append("note: non-monotonic x on the lower surface")
        return out

    def validate(self) -> None:
        """Raise AirfoilGeometryError when this shape has a hard defect.

        The non-monotonic x notes of :meth:`problems` are not errors.
        """
        hard = [m for m in self.problems() if "monotonic" not in m.lower()]
        if hard:
            raise AirfoilGeometryError("Bad airfoil shape: " + "; ".join(hard))

    def _self_intersection(self) -> tuple[int, int] | None:
        """Return the first pair of crossing segments, or None."""
        pts = self.coords
        nseg = pts.shape[0] - 1
        if nseg < 4:
            return None
        closed = self.is_closed
        for i in range(nseg):
            p1, p2 = pts[i], pts[i + 1]
            if float(np.linalg.norm(p2 - p1)) <= 1e-14:
                continue
            for j in range(i + 2, nseg):
                if i == 0 and j == nseg - 1 and closed:
                    continue
                p3, p4 = pts[j], pts[j + 1]
                if float(np.linalg.norm(p4 - p3)) <= 1e-14:
                    continue
                if _segments_cross(p1, p2, p3, p4):
                    return (i, j)
        return None


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Signed area of the triangle (a, b, c) [chord^2]."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Return True when *b* is collinear with segment a-c and lies in its box."""
    return bool(
        min(a[0], c[0]) - 1e-14 <= b[0] <= max(a[0], c[0]) + 1e-14
        and min(a[1], c[1]) - 1e-14 <= b[1] <= max(a[1], c[1]) + 1e-14
    )


def _segments_cross(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """Return True when the closed segments p1-p2 and p3-p4 meet away from shared ends."""
    d1 = _orientation(p3, p4, p1)
    d2 = _orientation(p3, p4, p2)
    d3 = _orientation(p1, p2, p3)
    d4 = _orientation(p1, p2, p4)
    if ((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0)) and (
        (d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0)
    ):
        return True
    if abs(d1) <= 1e-14 and _on_segment(p3, p1, p4):
        return True
    if abs(d2) <= 1e-14 and _on_segment(p3, p2, p4):
        return True
    if abs(d3) <= 1e-14 and _on_segment(p1, p3, p2):
        return True
    if abs(d4) <= 1e-14 and _on_segment(p1, p4, p2):
        return True
    return False
