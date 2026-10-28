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

Reference geometry
------------------
A shape can hold a reference geometry (:class:`SectionReference`): the
leading edge and the trailing-edge midpoint of the chord line, and the
reference chord of the aerodynamic coefficients. A plain-flap deflection
(:meth:`AirfoilShape.with_flap`) stores the reference geometry of the
section before the deflection, the reference of thin-airfoil theory
(Glauert, R&M 1095, 1927). A shape without a stored reference geometry
derives it from the points as given (:meth:`SectionReference.from_coords`).

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
- I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, Dover,
  1959, chapter 6 (NACA 4-digit thickness equation).
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge
  University Press, 2001, chapter 11 (airfoil geometry and thickness
  and camber definitions).
- H. Glauert, "Theoretical relationships for an aerofoil with hinged
  flap", ARC R&M 1095, 1927 (the chord of the undeflected section is the
  reference chord of a flapped section).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ventorum.core.errors import VentorumError


class AirfoilGeometryError(VentorumError):
    """Bad airfoil geometry or airfoil file (a ValueError for compatibility)."""


def _point(value: Any, name: str) -> tuple[float, float]:
    """Return *value* as a point (x, y) of two finite floats [chord units]."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise AirfoilGeometryError(f"{name} must be a point (x, y) of two finite numbers; got {value!r}.")
    return (float(arr[0]), float(arr[1]))


@dataclass(frozen=True)
class SectionReference:
    """Reference geometry of a section: the chord line and the reference chord.

    The chord line runs from the leading edge to the trailing-edge
    midpoint. The default moment reference point of the 2D solvers is the
    quarter-chord point on this line (:attr:`quarter_chord`). For a section
    with a deflected plain flap the reference geometry is the one of the
    section before the deflection, as in thin-airfoil theory (Glauert,
    R&M 1095, 1927).

    Parameters
    ----------
    leading_edge : tuple of float
        Leading-edge point (x, y) [chord units].
    trailing_edge : tuple of float
        Trailing-edge midpoint (x, y) [chord units].
    chord_ref : float or None
        Reference chord [chord units], positive. None (default) sets the
        distance from the leading edge to the trailing-edge midpoint.

    Raises
    ------
    AirfoilGeometryError
        If a point is not two finite numbers, the reference chord is not
        positive and finite, or the two points coincide and no reference
        chord is given.
    """

    leading_edge: tuple[float, float]
    trailing_edge: tuple[float, float]
    chord_ref: float | None = None

    def __post_init__(self) -> None:
        """Convert the points and the chord to floats and check them."""
        le = _point(self.leading_edge, "leading_edge")
        te = _point(self.trailing_edge, "trailing_edge")
        if self.chord_ref is None:
            chord = float(np.hypot(te[0] - le[0], te[1] - le[1]))
            if chord <= 0.0:
                raise AirfoilGeometryError(
                    "The leading edge and the trailing edge of a reference geometry coincide; "
                    "give chord_ref."
                )
        else:
            chord = float(self.chord_ref)
            if not np.isfinite(chord) or chord <= 0.0:
                raise AirfoilGeometryError(f"chord_ref must be a positive length; got {self.chord_ref!r}.")
        object.__setattr__(self, "leading_edge", le)
        object.__setattr__(self, "trailing_edge", te)
        object.__setattr__(self, "chord_ref", chord)

    @classmethod
    def from_coords(cls, coords: np.ndarray) -> SectionReference:
        """Derive the reference geometry of a section from its points as given.

        The leading edge is the point farthest from the trailing-edge
        midpoint (:func:`leading_edge_index`), the trailing edge is the
        midpoint of the first and the last point, and the reference chord
        is the distance between them (1.0 for a section of zero size).

        Parameters
        ----------
        coords : numpy.ndarray
            Section points [chord units], shape (N, 2), in Selig order.

        Returns
        -------
        SectionReference
            The derived reference geometry.
        """
        pts = np.asarray(coords, dtype=np.float64)
        le = pts[leading_edge_index(pts)]
        mid = 0.5 * (pts[0] + pts[-1])
        chord = float(np.linalg.norm(le - mid))
        return cls(leading_edge=(float(le[0]), float(le[1])), trailing_edge=(float(mid[0]), float(mid[1])),
                   chord_ref=chord if chord > 0.0 else 1.0)

    @property
    def quarter_chord(self) -> tuple[float, float]:
        """Quarter-chord point on the chord line [chord units].

        The point lies 0.25 of the way from the leading edge to the
        trailing-edge midpoint.
        """
        le, te = self.leading_edge, self.trailing_edge
        return (le[0] + 0.25 * (te[0] - le[0]), le[1] + 0.25 * (te[1] - le[1]))

    def mapped(self, origin: np.ndarray, rotation: np.ndarray, scale: float) -> SectionReference:
        """Return the reference geometry after the map p -> rotation (p - origin) / scale.

        Parameters
        ----------
        origin : numpy.ndarray
            Point subtracted first [chord units], shape (2,).
        rotation : numpy.ndarray
            Rotation matrix [-], shape (2, 2).
        scale : float
            Length divisor [chord units], positive.

        Returns
        -------
        SectionReference
            The mapped points and the reference chord divided by ``scale``.
        """
        pts = (np.array([self.leading_edge, self.trailing_edge]) - origin) @ np.asarray(rotation).T / scale
        return SectionReference(leading_edge=pts[0], trailing_edge=pts[1], chord_ref=self.chord_ref / scale)


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
    reference : SectionReference or None
        Stored reference geometry (chord line and reference chord) [chord
        units]. :meth:`with_flap` sets it to the geometry of the section
        before the deflection; :meth:`repanel`, :meth:`repanel_cosine`,
        :meth:`normalized` and the factory copy keep it. None (default):
        the geometry is derived from the points as given (see
        :meth:`reference_geometry`). The file writers do not store it.
    """

    name: str = "airfoil"
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    reference: SectionReference | None = None

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
        if self.reference is not None and not isinstance(self.reference, SectionReference):
            raise AirfoilGeometryError(
                f"reference must be a SectionReference or None; got {type(self.reference).__name__}."
            )
        self.coords = arr
        self.name = str(self.name)

    def reference_geometry(self) -> SectionReference:
        """Return the reference geometry of this shape.

        Returns
        -------
        SectionReference
            The stored reference geometry, or the one derived from the
            points as given (:meth:`SectionReference.from_coords`) when
            none is stored [chord units].
        """
        if self.reference is not None:
            return self.reference
        return SectionReference.from_coords(self.coords)

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
        """Index of the point farthest from the trailing-edge midpoint.

        This is the package definition of the leading edge on a point
        polygon (see :func:`leading_edge_index`).
        """
        return leading_edge_index(self.coords)

    @property
    def chord(self) -> float:
        """Distance from the leading edge to the trailing-edge midpoint [chord units]."""
        mid = self._te_midpoint
        return float(np.linalg.norm(self.coords[self.leading_edge_index] - mid))

    def normalized(self) -> AirfoilShape:
        """Return a new shape with the leading edge at (0, 0) and the trailing edge at (1, 0).

        The stored points are translated, rotated and scaled. The name is
        kept. A stored reference geometry gets the same map (it stays
        attached to the section).

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
        ref = None if self.reference is None else self.reference.mapped(le, rot, length)
        return AirfoilShape(name=self.name, coords=out, reference=ref)

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

    @classmethod
    def naca(cls, designation: str | int, **kwargs: Any) -> AirfoilShape:
        """Build a NACA 4-digit or 5-digit section as a shape.

        This method calls :func:`ventorum.airfoil.naca.naca` and stores
        the points. The thickness follows Abbott and von Doenhoff (1959,
        section 6.4). The 5-digit mean lines follow Jacobs, Ward and
        Pinkerton (NACA Report 460, 1933).

        Parameters
        ----------
        designation : str or int
            Designation such as "2412", "NACA 2412" or "23012" [-].
        **kwargs : Any
            Extra arguments of :func:`ventorum.airfoil.naca.naca`:
            ``n_points`` [-], ``spacing`` [-], ``closed_te`` [-].
            ``name`` [str] sets the shape name.

        Returns
        -------
        AirfoilShape
            Shape with coordinates [chord units] in Selig order.
        """
        from ventorum.airfoil.naca import naca as _naca_array

        name = kwargs.pop("name", None)
        if name is None:
            text = str(designation).strip()
            name = text if text.upper().startswith("NACA") else f"NACA {text}"
        coords = _naca_array(designation, **kwargs)
        return cls(name=str(name), coords=coords)

    @classmethod
    def from_file(cls, path: str | Path) -> AirfoilShape:
        """Read a Selig, Lednicer or plain file as a shape.

        This method calls :func:`ventorum.airfoil.io.read_airfoil`.
        Selig and co-workers (*Summary of Low-Speed Airfoil Data*, 1995)
        define the Selig format.

        Parameters
        ----------
        path : str or pathlib.Path
            Path of the file.

        Returns
        -------
        AirfoilShape
            Shape with coordinates [chord units] in Selig order.
        """
        from ventorum.airfoil.io import read_airfoil as _read_airfoil

        shape = _read_airfoil(path)
        return cls(name=shape.name, coords=shape.coords)

    def repanel(self, n_panels: int = 160, **kwargs: Any) -> AirfoilShape:
        """Redistribute the nodes by equidistribution of a weight function.

        This method calls :func:`ventorum.airfoil.paneling.repanel`.
        The spline uses not-a-knot end conditions (de Boor, 1978,
        Chapter IV).

        Parameters
        ----------
        n_panels : int
            Number of panels [-]. Default 160.
        **kwargs : Any
            Extra arguments of the array function: ``le_refinement`` [-],
            ``te_refinement`` [-], ``curvature_weight`` [-].

        Returns
        -------
        AirfoilShape
            Repaneled copy (coordinates [chord units]) in Selig order. It
            keeps the stored reference geometry.
        """
        from ventorum.airfoil.paneling import repanel as _repanel_array

        coords = _repanel_array(self.coords, n_panels=n_panels, **kwargs)
        return AirfoilShape(name=self.name, coords=coords, reference=self.reference)

    def repanel_cosine(self, n_per_side: int) -> AirfoilShape:
        """Resample each surface with cosine spacing in arc length.

        This method calls
        :func:`ventorum.airfoil.paneling.repanel_cosine`. Each surface
        gets Chebyshev cosine nodes, dense at the leading edge and at
        the trailing edge.

        Parameters
        ----------
        n_per_side : int
            Panels per surface [-].

        Returns
        -------
        AirfoilShape
            Resampled copy (coordinates [chord units]) in Selig order. It
            keeps the stored reference geometry.
        """
        from ventorum.airfoil.paneling import repanel_cosine as _cosine_array

        coords = _cosine_array(self.coords, n_per_side)
        return AirfoilShape(name=self.name, coords=coords, reference=self.reference)

    def with_flap(
        self,
        x_hinge: float,
        deflection: float,
        y_hinge: float = 0.5,
        n_arc: int | None = None,
        hinge_treatment: str = "repanel",
        n_panels: int | None = None,
    ) -> AirfoilShape:
        """Deflect a plain flap about a hinge point.

        This method calls
        :func:`ventorum.airfoil.flap.deflect_plain_flap`. The flap is a
        hinged aft portion of the section (Glauert, 1926, chapter VII;
        Glauert, R&M 1095, 1927).

        Parameters
        ----------
        x_hinge : float
            Hinge chord station [chord units], in (0.05, 0.98).
        deflection : float
            Flap deflection [deg], positive when the trailing edge moves
            down. Values satisfy |deflection| <= 60.
        y_hinge : float
            Hinge height as a fraction [-] of the local thickness at
            ``x_hinge``, from the lower surface (0) to the upper surface
            (1). Default 0.5 (mid-thickness). This is the one hinge-height
            definition of the package (``docs/theory/airfoils.md``).
        n_arc : int or None
            Point count [-] of the filler arc on the stretched side.
            None selects the default of the array function.
        hinge_treatment : {"none", "repanel", "merge"}
            Treatment of the nodes near the hinge [-]. "repanel"
            (default) distributes the nodes again on a spline through the
            outline; "none" keeps the deflected outline; "merge" removes short panels
            and isolated sharp corners near the hinge by node removal (see
            :func:`ventorum.airfoil.flap.deflect_plain_flap`).
        n_panels : int or None
            Panel count [-] of the "repanel" treatment. None keeps the
            panel count of this shape.

        Returns
        -------
        AirfoilShape
            Deflected copy (coordinates [chord units]) in Selig order, a
            simple polygon. Its reference geometry
            (:attr:`reference`) is the one of the section before the
            deflection: the stored reference geometry of this shape, or
            the one derived from its points (leading edge, trailing-edge
            midpoint and their distance as the reference chord). The
            2D solvers then use the chord of the undeflected section as
            the reference chord and its quarter-chord point as the moment
            reference point, as thin-airfoil theory does (Glauert, R&M
            1095, 1927).

        Raises
        ------
        ValueError
            If an input is out of range, or the deflected section is not
            a simple polygon (:class:`AirfoilGeometryError`).
        """
        from ventorum.airfoil.flap import deflect_plain_flap as _deflect_array

        coords = _deflect_array(self.coords, x_hinge, deflection, y_hinge=y_hinge, n_arc=n_arc,
                                hinge_treatment=hinge_treatment, n_panels=n_panels)
        return AirfoilShape(name=self.name, coords=coords, reference=self.reference_geometry())

    def to_file(self, path: str | Path, fmt: str = "selig", precision: int = 6) -> None:
        """Write this shape in Selig or Lednicer format.

        This method calls :func:`ventorum.airfoil.io.write_selig` or
        :func:`ventorum.airfoil.io.write_lednicer`. Selig and
        co-workers (*Summary of Low-Speed Airfoil Data*, 1995) define
        the Selig format. The formats hold the points only: a stored
        reference geometry is not written.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        fmt : str
            Format name [-]: "selig" or "lednicer". Default "selig".
        precision : int
            Digits after the decimal point [-]. Default 6.
        """
        from ventorum.airfoil.io import write_lednicer as _write_lednicer
        from ventorum.airfoil.io import write_selig as _write_selig

        key = str(fmt).strip().lower()
        if key == "selig":
            _write_selig(self, path, precision=precision)
            return
        if key == "lednicer":
            _write_lednicer(self, path, precision=precision)
            return
        raise ValueError(f"Unknown airfoil format {fmt!r}: use 'selig' or 'lednicer'.")

    def _self_intersection(self) -> tuple[int, int] | None:
        """Return the first pair of crossing segments, or None."""
        return polygon_self_intersection(self.coords)


def leading_edge_index(coords: np.ndarray) -> int:
    """Return the index of the leading-edge point of a section polygon.

    The leading edge is the point farthest from the trailing-edge
    midpoint, the midpoint of the first and the last point. This is the
    one definition of the leading edge in the airfoil package. On a
    point polygon it selects a stored point (:class:`AirfoilShape`, the
    surface split of the flap functions). On a spline it is the point
    where the distance to the trailing-edge midpoint has its maximum,
    so the tangent is normal to the line to that midpoint
    (:attr:`ventorum.airfoil.paneling.AirfoilSpline.s_leading_edge`).

    Parameters
    ----------
    coords : numpy.ndarray
        Section points [chord units], shape (N, 2), in Selig order.

    Returns
    -------
    int
        Index of the leading-edge point [-].
    """
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        raise AirfoilGeometryError(
            f"Airfoil coordinates must have shape (N, 2) with N > 0; got {pts.shape}."
        )
    mid = 0.5 * (pts[0] + pts[-1])
    return int(np.argmax(np.linalg.norm(pts - mid, axis=1)))


#: Row block size of the vectorized segment-pair test [-].
_PAIR_BLOCK = 256


def polygon_self_intersection(coords: np.ndarray, tol: float = 1e-14) -> tuple[int, int] | None:
    """Return the first pair of crossing or touching polygon edges, or None.

    The polygon closes from the last point to the first point. When the
    first and the last point are equal (a closed trailing edge), the
    repeated point is dropped. Edges that share an end point (neighbours
    in the ring) are not tested. An edge of zero length is skipped. Two
    edges fail when they cross, or when an end point of one edge lies on
    the other edge (orientation test with the tolerance ``tol``).

    Parameters
    ----------
    coords : numpy.ndarray
        Polygon points [chord units], shape (N, 2).
    tol : float
        Tolerance [chord^2] of the orientation test. Default 1e-14.

    Returns
    -------
    tuple of int or None
        Indices (i, j), i < j, of the first failing edge pair in row
        order, where edge k runs from point k to point k + 1 of the
        ring, or None for a simple polygon.
    """
    pts = np.asarray(coords, dtype=float)
    if pts.shape[0] >= 2 and float(np.linalg.norm(pts[0] - pts[-1])) <= 1e-12:
        pts = pts[:-1]
    m = pts.shape[0]
    if m < 4:
        return None
    a = pts
    b = np.roll(pts, -1, axis=0)
    good = np.linalg.norm(b - a, axis=1) > 1e-14
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    cols = np.arange(m)
    for start in range(0, m, _PAIR_BLOCK):
        rows = np.arange(start, min(start + _PAIR_BLOCK, m))
        ii = np.repeat(rows, m)
        jj = np.tile(cols, rows.size)
        keep = (jj >= ii + 2) & ~((ii == 0) & (jj == m - 1)) & good[ii] & good[jj]
        ii = ii[keep]
        jj = jj[keep]
        boxes = np.all(lo[ii] <= hi[jj] + tol, axis=1) & np.all(lo[jj] <= hi[ii] + tol, axis=1)
        ii = ii[boxes]
        jj = jj[boxes]
        if ii.size == 0:
            continue
        p1, p2, p3, p4 = a[ii], b[ii], a[jj], b[jj]
        d1 = _orient(p3, p4, p1)
        d2 = _orient(p3, p4, p2)
        d3 = _orient(p1, p2, p3)
        d4 = _orient(p1, p2, p4)
        proper = (((d1 > 0.0) & (d2 < 0.0)) | ((d1 < 0.0) & (d2 > 0.0))) & (
            ((d3 > 0.0) & (d4 < 0.0)) | ((d3 < 0.0) & (d4 > 0.0))
        )
        touch = (
            ((np.abs(d1) <= tol) & _within(p3, p1, p4, tol))
            | ((np.abs(d2) <= tol) & _within(p3, p2, p4, tol))
            | ((np.abs(d3) <= tol) & _within(p1, p3, p2, tol))
            | ((np.abs(d4) <= tol) & _within(p1, p4, p2, tol))
        )
        bad = np.nonzero(proper | touch)[0]
        if bad.size:
            k = int(bad[0])
            return int(ii[k]), int(jj[k])
    return None


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Signed doubled area of the triangles (a, b, c) [chord^2], row by row."""
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def _within(a: np.ndarray, p: np.ndarray, c: np.ndarray, tol: float) -> np.ndarray:
    """Return True where the point p lies in the box of the segment a-c, row by row."""
    return (
        (np.minimum(a[:, 0], c[:, 0]) - tol <= p[:, 0])
        & (p[:, 0] <= np.maximum(a[:, 0], c[:, 0]) + tol)
        & (np.minimum(a[:, 1], c[:, 1]) - tol <= p[:, 1])
        & (p[:, 1] <= np.maximum(a[:, 1], c[:, 1]) + tol)
    )
