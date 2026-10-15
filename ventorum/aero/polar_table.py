# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Section polar tables over Reynolds number and flap deflection.

The viscous coupling needs section data at the local Reynolds number of
each strip and at each flap deflection. This module builds that data
offline: it runs one polar per (Reynolds number, flap deflection) pair
through any section polar source (XFOIL now, the in-house 2D solver
later), stores the results on a grid over (Re, flap deflection, alpha),
and interpolates between the grid nodes with honest gaps.

Interpolation is bilinear in log10(Re) and in flap deflection at fixed
angle of attack. Section data change slowly with log10(Re); the
Reynolds sweeps plotted against a logarithmic Reynolds axis in Abbott
and Von Doenhoff (1959, Theory of Wing Sections, Dover) show this
behaviour. Points that the source did not converge stay NaN, and
queries outside the table range raise an error instead of extrapolating.

A source may offer a ``run_xfoil_batch`` hook with the calling
convention of :func:`ventorum.aero.xfoil_runner.run_xfoil_batch`
(a list of case dicts with keys ``section``, ``Re``, ``alphas_deg``
and ``flap``, plus ``workers``; one tabulated polar or exception object
per case, in input order). The builder detects the hook by feature
check and never by type test. Without the hook the builder calls
``source.polar`` once per pair, in parallel threads unless
``workers=1``.

Height axis: the in-house 2D solver adds an optional height axis
``h_over_c`` [-] (height of the section above the ground in chords)
to these tables. When set, stored arrays have four dimensions
(n_Re, n_flap, n_h, n_alpha). The height axis uses linear
interpolation in 1/h because the aerodynamic ground effect from
image singularities scales roughly inversely with the ground height
(Drela, 1989; Wieselsberger, 1921). Free-air tables keep their
three-dimensional shape (n_Re, n_flap, n_alpha) for backward
compatibility.

Sources: Drela, M., 1989, XFOIL: An analysis and design system for low
Reynolds number airfoils; Abbott, I. H. and Von Doenhoff, A. E., 1959,
Theory of Wing Sections, Dover; Wieselsberger, C., 1921, Wing resistance
near the ground, NACA TM-77; bilinear and trilinear interpolation on a
rectangular grid follows the standard multidimensional interpolation
described in Press, W. H. and others, Numerical Recipes, Cambridge
University Press.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import json
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ventorum.core.datatypes import TabulatedAirfoil

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ventorum.aero.polar_source import SectionPolarSource


def _as_sorted_axis(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Return *values* as a float array, strictly ascending and finite."""
    axis = np.atleast_1d(np.asarray(values, dtype=float))
    if axis.size == 0:
        raise ValueError(f"{name} must hold at least one value.")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must hold finite values only.")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly ascending, got {axis}.")
    return axis


@dataclass(slots=True)
class PolarTable:
    """Section polars tabulated over Reynolds number, flap deflection and height.

    Parameters
    ----------
    Re : np.ndarray
        Reynolds numbers [-], strictly ascending, shape (n_Re,).
    flap_deg : np.ndarray
        Flap deflections [deg], positive down, strictly ascending,
        shape (n_flap,). A single zero entry means the clean airfoil.
    alpha_deg : np.ndarray
        Angles of attack [deg], unique values, shape (n_alpha,).
        The order is kept as built.
    Cl, Cd, Cm : np.ndarray
        Lift, drag and pitching-moment coefficients [-], each of shape
        (n_Re, n_flap, n_alpha) for free air or
        (n_Re, n_flap, n_h, n_alpha) with a height axis. NaN marks a
        point where the source did not converge.
    source_info : dict
        Provenance: source name, section, options and version.
    last_dropped : int
        Number of alpha points dropped by the last :meth:`at` call.
    h_over_c : np.ndarray or None
        Heights above ground in chords [-], strictly ascending,
        shape (n_h,). None indicates free air.

    Notes
    -----
    The height axis uses linear interpolation in 1/h because the
    aerodynamic ground effect from image singularities scales roughly
    inversely with the ground height (Drela, 1989; Wieselsberger, 1921).
    """

    Re: np.ndarray
    flap_deg: np.ndarray
    alpha_deg: np.ndarray
    Cl: np.ndarray
    Cd: np.ndarray
    Cm: np.ndarray
    source_info: dict[str, Any] = field(default_factory=dict)
    last_dropped: int = field(default=0, compare=False)
    h_over_c: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "Re", _as_sorted_axis(self.Re, "Re"))
        if np.any(self.Re <= 0.0):
            raise ValueError("Re must hold positive values only.")
        object.__setattr__(
            self, "flap_deg", _as_sorted_axis(self.flap_deg, "flap_deg")
        )
        alpha = np.atleast_1d(np.asarray(self.alpha_deg, dtype=float))
        if alpha.size == 0:
            raise ValueError("alpha_deg must hold at least one value.")
        if not np.all(np.isfinite(alpha)):
            raise ValueError("alpha_deg must hold finite values only.")
        if np.unique(alpha).size != alpha.size:
            raise ValueError("alpha_deg must hold unique values.")
        object.__setattr__(self, "alpha_deg", alpha)
        if self.h_over_c is not None:
            object.__setattr__(
                self, "h_over_c", _as_sorted_axis(self.h_over_c, "h_over_c")
            )
            if np.any(self.h_over_c <= 0.0):
                raise ValueError("h_over_c must hold positive values only.")
            shape = (self.Re.size, self.flap_deg.size, self.h_over_c.size, alpha.size)
        else:
            shape = (self.Re.size, self.flap_deg.size, alpha.size)
        for name in ("Cl", "Cd", "Cm"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != shape:
                raise ValueError(
                    f"{name} has shape {arr.shape}, expected {shape}."
                )
            object.__setattr__(self, name, arr)
        object.__setattr__(self, "source_info", dict(self.source_info))

    def coverage(self) -> np.ndarray:
        """Return the converged fraction of alpha points.

        Returns
        -------
        np.ndarray
            Fraction of converged (non-NaN lift) alpha points [-],
            shape (n_Re, n_flap) for free air or (n_Re, n_flap, n_h)
            with a height axis.
        """
        return np.mean(np.isfinite(self.Cl), axis=-1)

    def at(
        self, Re: float, flap_deg: float, h_over_c: float | None = None
    ) -> TabulatedAirfoil:
        """Interpolate the table at one operating point.

        The interpolation is bilinear in log10(Re) and in flap
        deflection at each angle of attack for free air, and trilinear
        with linear interpolation in 1/h when a height axis is present.
        Ground influence scales roughly with 1/h. The interpolation in
        height is therefore linear in 1/h. An alpha point is dropped
        when any table value used for it is NaN. The dropped count is
        stored in :attr:`last_dropped` and in the polar name.

        Parameters
        ----------
        Re : float
            Reynolds number [-]. Must lie inside the table range.
        flap_deg : float
            Flap deflection [deg], positive down. Must lie inside the
            table range.
        h_over_c : float or None
            Height above ground in chords [-]. Required if the table
            holds a height axis. Must be None for a free-air table.

        Returns
        -------
        TabulatedAirfoil
            The interpolated polar. Angles are [rad]. Its ``Re`` field
            holds the query Reynolds number [-].

        Raises
        ------
        ValueError
            If the query lies outside the table range, or if ``h_over_c``
            is missing for a table with a height axis, or given for a
            free-air table. No extrapolation is done.
        """
        query_re = float(Re)
        query_flap = float(flap_deg)
        if not np.isfinite(query_re) or not np.isfinite(query_flap):
            raise ValueError("The query (Re, flap_deg) must be finite.")
        i_re, weight_re = _bracket(self.Re, query_re, "Re", use_log=True)
        j_flap, weight_flap = _bracket(self.flap_deg, query_flap, "flap_deg")

        if self.h_over_c is None:
            if h_over_c is not None:
                raise ValueError(
                    "This polar table has no height axis (free air only)."
                )
            rows = (
                (0, 0, i_re[0], j_flap[0]),
                (0, 1, i_re[0], j_flap[1]),
                (1, 0, i_re[1], j_flap[0]),
                (1, 1, i_re[1], j_flap[1]),
            )
            used_rows = {
                (di, dj)
                for di, dj, _, _ in rows
                if (di == 0 or weight_re > 0.0)
                and (dj == 0 or weight_flap > 0.0)
                and (di == 1 or weight_re < 1.0)
                and (dj == 1 or weight_flap < 1.0)
            }
            ok = np.ones(self.alpha_deg.size, dtype=bool)
            for di, dj, ii, jj in rows:
                if (di, dj) not in used_rows:
                    continue
                ok &= (
                    np.isfinite(self.Cl[ii, jj, :])
                    & np.isfinite(self.Cd[ii, jj, :])
                    & np.isfinite(self.Cm[ii, jj, :])
                )
            cl = _bilinear(self.Cl, i_re, j_flap, weight_re, weight_flap)
            cd = _bilinear(self.Cd, i_re, j_flap, weight_re, weight_flap)
            cm = _bilinear(self.Cm, i_re, j_flap, weight_re, weight_flap)
            name_suffix = ""
        else:
            if h_over_c is None:
                raise ValueError(
                    "h_over_c is required for a polar table with a height axis."
                )
            query_h = float(h_over_c)
            if not np.isfinite(query_h):
                raise ValueError("The query h_over_c must be finite.")
            k_h, weight_h = _bracket(
                self.h_over_c, query_h, "h_over_c", use_inv=True
            )
            rows_3d = tuple(
                (di, dj, dk, i_re[di], j_flap[dj], k_h[dk])
                for di in (0, 1)
                for dj in (0, 1)
                for dk in (0, 1)
            )
            used_3d = {
                (di, dj, dk)
                for di, dj, dk, _, _, _ in rows_3d
                if (di == 0 or weight_re > 0.0)
                and (dj == 0 or weight_flap > 0.0)
                and (dk == 0 or weight_h > 0.0)
                and (di == 1 or weight_re < 1.0)
                and (dj == 1 or weight_flap < 1.0)
                and (dk == 1 or weight_h < 1.0)
            }
            ok = np.ones(self.alpha_deg.size, dtype=bool)
            for di, dj, dk, ii, jj, kk in rows_3d:
                if (di, dj, dk) not in used_3d:
                    continue
                ok &= (
                    np.isfinite(self.Cl[ii, jj, kk, :])
                    & np.isfinite(self.Cd[ii, jj, kk, :])
                    & np.isfinite(self.Cm[ii, jj, kk, :])
                )
            cl = _trilinear(
                self.Cl, i_re, j_flap, k_h, weight_re, weight_flap, weight_h
            )
            cd = _trilinear(
                self.Cd, i_re, j_flap, k_h, weight_re, weight_flap, weight_h
            )
            cm = _trilinear(
                self.Cm, i_re, j_flap, k_h, weight_re, weight_flap, weight_h
            )
            name_suffix = f", h={query_h:.6g}"

        keep = np.flatnonzero(ok)
        dropped = int(self.alpha_deg.size - keep.size)
        object.__setattr__(self, "last_dropped", dropped)
        name = (
            f"table(Re={query_re:.6g}, flap={query_flap:.6g} deg{name_suffix}, "
            f"{dropped} of {self.alpha_deg.size} alpha dropped)"
        )
        return TabulatedAirfoil(
            name=name,
            alpha=np.radians(self.alpha_deg[keep]),
            Cl_data=np.asarray(cl[keep], dtype=float),
            Cd_data=np.asarray(cd[keep], dtype=float),
            Cm_data=np.asarray(cm[keep], dtype=float),
            Re=query_re,
        )

    def to_npz(self, path: str | Path) -> Path:
        """Store the table in a numpy ``.npz`` file (no pickle).

        Parameters
        ----------
        path : str or Path
            Output file path.

        Returns
        -------
        Path
            The file written.

        Raises
        ------
        ValueError
            If ``source_info`` is not JSON serialisable.
        """
        try:
            info_text = json.dumps(self.source_info)
        except TypeError as exc:
            raise ValueError(
                "source_info must be JSON serialisable."
            ) from exc
        out = Path(path)
        save_dict: dict[str, Any] = {
            "Re": np.asarray(self.Re, dtype=float),
            "flap_deg": np.asarray(self.flap_deg, dtype=float),
            "alpha_deg": np.asarray(self.alpha_deg, dtype=float),
            "Cl": np.asarray(self.Cl, dtype=float),
            "Cd": np.asarray(self.Cd, dtype=float),
            "Cm": np.asarray(self.Cm, dtype=float),
            "source_info": np.asarray(info_text),
        }
        if self.h_over_c is not None:
            save_dict["h_over_c"] = np.asarray(self.h_over_c, dtype=float)
        np.savez(out, **save_dict)
        return out

    @classmethod
    def from_npz(cls, path: str | Path) -> PolarTable:
        """Read a table stored by :meth:`to_npz`.

        Parameters
        ----------
        path : str or Path
            Input file path.

        Returns
        -------
        PolarTable
            The stored table.

        Raises
        ------
        ValueError
            If a required array is missing or the stored data is not
            a valid table.
        """
        try:
            stored = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read polar table {path}: {exc}.") from exc
        with stored:
            keys = set(stored.files)
            missing = {
                "Re",
                "flap_deg",
                "alpha_deg",
                "Cl",
                "Cd",
                "Cm",
                "source_info",
            } - keys
            if missing:
                raise ValueError(
                    f"Polar table {path} misses {sorted(missing)}."
                )
            try:
                info = json.loads(str(stored["source_info"]))
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Polar table {path} has unreadable source_info."
                ) from exc
            h_over_c = (
                np.asarray(stored["h_over_c"], dtype=float)
                if "h_over_c" in keys
                else None
            )
            return cls(
                Re=np.asarray(stored["Re"], dtype=float),
                flap_deg=np.asarray(stored["flap_deg"], dtype=float),
                alpha_deg=np.asarray(stored["alpha_deg"], dtype=float),
                Cl=np.asarray(stored["Cl"], dtype=float),
                Cd=np.asarray(stored["Cd"], dtype=float),
                Cm=np.asarray(stored["Cm"], dtype=float),
                source_info=info,
                h_over_c=h_over_c,
            )


def _bracket(
    axis: np.ndarray,
    query: float,
    name: str,
    *,
    use_log: bool = False,
    use_inv: bool = False,
) -> tuple[tuple[int, int], float]:
    """Locate the cell of *axis* that holds *query* and its weight.

    Parameters
    ----------
    axis : np.ndarray
        Strictly ascending axis values.
    query : float
        Query value in axis units.
    name : str
        Axis name used in the error message.
    use_log : bool
        Interpolate in log10 of the axis (Reynolds number [-]).
    use_inv : bool
        Interpolate linearly in 1 / axis (ground height ratio [-]).

    Returns
    -------
    (int, int)
        Lower and upper node indices (equal on a single-point axis).
    float
        Weight of the upper node in [0, 1].

    Raises
    ------
    ValueError
        If *query* lies outside the axis range.
    """
    lo, hi = float(axis[0]), float(axis[-1])
    slack = 1e-12 * max(1.0, abs(lo), abs(hi))
    if query < lo - slack or query > hi + slack:
        raise ValueError(
            f"{name}={query:g} lies outside the table range [{lo:g}, {hi:g}]."
        )
    query = min(max(query, lo), hi)
    if axis.size == 1:
        return (0, 0), 0.0
    if use_inv:
        upper = int(np.searchsorted(axis, query, side="right")) - 1
        upper = min(max(upper, 0), axis.size - 2)
        lower = upper
        upper = lower + 1
        inv_low = 1.0 / float(axis[lower])
        inv_high = 1.0 / float(axis[upper])
        inv_query = 1.0 / float(query)
        span = inv_high - inv_low
        weight = 0.0 if span == 0.0 else (inv_query - inv_low) / span
        return (lower, upper), min(max(weight, 0.0), 1.0)
    if use_log:
        scaled = np.log10(axis)
        value = float(np.log10(query))
    else:
        scaled = axis
        value = query
    upper = int(np.searchsorted(scaled, value, side="right")) - 1
    upper = min(max(upper, 0), axis.size - 2)
    lower = upper
    upper = lower + 1
    span = float(scaled[upper] - scaled[lower])
    weight = 0.0 if span == 0.0 else (value - float(scaled[lower])) / span
    return (lower, upper), min(max(weight, 0.0), 1.0)


def _bilinear(
    values: np.ndarray,
    i_re: tuple[int, int],
    j_flap: tuple[int, int],
    weight_re: float,
    weight_flap: float,
) -> np.ndarray:
    """Blend one table corner block to the query point.

    Parameters
    ----------
    values : np.ndarray
        Table array of shape (n_Re, n_flap, n_alpha) [-].
    i_re : (int, int)
        Lower and upper Reynolds indices.
    j_flap : (int, int)
        Lower and upper flap indices.
    weight_re, weight_flap : float
        Upper-node weights in [0, 1].

    Returns
    -------
    np.ndarray
        Blended values [-], shape (n_alpha,). A zero weight skips its
        corner, so NaN in an unused corner never spreads.
    """

    def blend(low: np.ndarray, high: np.ndarray, weight: float) -> np.ndarray:
        if weight <= 0.0:
            return low
        if weight >= 1.0:
            return high
        return (1.0 - weight) * low + weight * high

    low_re = blend(values[i_re[0], j_flap[0], :], values[i_re[0], j_flap[1], :], weight_flap)
    high_re = blend(values[i_re[1], j_flap[0], :], values[i_re[1], j_flap[1], :], weight_flap)
    return blend(low_re, high_re, weight_re)


def _trilinear(
    values: np.ndarray,
    i_re: tuple[int, int],
    j_flap: tuple[int, int],
    k_h: tuple[int, int],
    weight_re: float,
    weight_flap: float,
    weight_h: float,
) -> np.ndarray:
    """Blend one table block across Reynolds, flap, and height.

    Parameters
    ----------
    values : np.ndarray
        Table array of shape (n_Re, n_flap, n_h, n_alpha) [-].
    i_re : (int, int)
        Lower and upper Reynolds indices.
    j_flap : (int, int)
        Lower and upper flap indices.
    k_h : (int, int)
        Lower and upper height indices.
    weight_re, weight_flap, weight_h : float
        Upper-node weights in [0, 1].

    Returns
    -------
    np.ndarray
        Blended values [-], shape (n_alpha,). A zero weight skips its
        corner, so NaN in an unused corner never spreads.
    """

    def blend(low: np.ndarray, high: np.ndarray, weight: float) -> np.ndarray:
        if weight <= 0.0:
            return low
        if weight >= 1.0:
            return high
        return (1.0 - weight) * low + weight * high

    c00 = blend(
        values[i_re[0], j_flap[0], k_h[0], :],
        values[i_re[0], j_flap[0], k_h[1], :],
        weight_h,
    )
    c01 = blend(
        values[i_re[0], j_flap[1], k_h[0], :],
        values[i_re[0], j_flap[1], k_h[1], :],
        weight_h,
    )
    low_re = blend(c00, c01, weight_flap)

    c10 = blend(
        values[i_re[1], j_flap[0], k_h[0], :],
        values[i_re[1], j_flap[0], k_h[1], :],
        weight_h,
    )
    c11 = blend(
        values[i_re[1], j_flap[1], k_h[0], :],
        values[i_re[1], j_flap[1], k_h[1], :],
        weight_h,
    )
    high_re = blend(c10, c11, weight_flap)

    return blend(low_re, high_re, weight_re)


def _flap_argument(
    deflection_deg: float, x_hinge: float | None, y_hinge: float
) -> tuple[float, float, float] | None:
    """Build the source flap tuple for one deflection.

    Parameters
    ----------
    deflection_deg : float
        Flap deflection [deg], positive down.
    x_hinge : float or None
        Hinge chordwise position, fraction of chord [-].
    y_hinge : float
        Hinge vertical position, fraction of chord [-].

    Returns
    -------
    tuple or None
        ``(x_hinge, y_hinge, deflection_deg)`` or None for a zero
        deflection, which runs the clean airfoil.

    Raises
    ------
    ValueError
        If the deflection is not zero but no hinge is given.
    """
    if deflection_deg == 0.0:
        return None
    if x_hinge is None:
        raise ValueError(
            "x_hinge is required for a nonzero flap deflection."
        )
    if not np.isfinite(x_hinge) or not np.isfinite(y_hinge):
        raise ValueError("x_hinge and y_hinge must be finite.")
    return (float(x_hinge), float(y_hinge), float(deflection_deg))


def _source_failed_as_gap(error: BaseException) -> bool:
    """Report whether *error* means the source did not converge.

    Parameters
    ----------
    error : BaseException
        Error raised by the source (or returned by its batch hook).

    Returns
    -------
    bool
        True for a convergence RuntimeError, which becomes a NaN gap.
        Any other error propagates to the caller.
    """
    return isinstance(error, RuntimeError) and "converg" in str(error).lower()


def _place_on_grid(
    polar: TabulatedAirfoil | None,
    alpha_rad: np.ndarray,
    tolerance_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map one source polar onto the requested alpha grid.

    Only computed points are stored: a requested angle takes the value
    of the nearest tabulated source point within *tolerance_rad*,
    otherwise NaN. Source points (converged or NaN) are never
    interpolated across a gap.

    Parameters
    ----------
    polar : TabulatedAirfoil or None
        The source polar, or None for a fully failed pair.
    alpha_rad : np.ndarray
        Requested angles [rad].
    tolerance_rad : float
        Largest accepted distance to a tabulated point [rad].

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
        Cl, Cd, Cm rows [-] on the requested grid (NaN for gaps).
    """
    cl = np.full(alpha_rad.shape, np.nan)
    cd = np.full(alpha_rad.shape, np.nan)
    cm = np.full(alpha_rad.shape, np.nan)
    if polar is None:
        return cl, cd, cm
    tab_alpha, tab_cl, tab_cd, tab_cm = polar.tables()
    if tab_cm is None:
        tab_cm = np.zeros_like(tab_cl)
    gap = np.abs(tab_alpha.reshape(-1, 1) - alpha_rad.reshape(1, -1))
    nearest = np.argmin(gap, axis=0)
    distance = gap[nearest, np.arange(alpha_rad.size)]
    hit = distance <= tolerance_rad
    cl[hit] = np.asarray(tab_cl, dtype=float)[nearest[hit]]
    cd[hit] = np.asarray(tab_cd, dtype=float)[nearest[hit]]
    cm[hit] = np.asarray(tab_cm, dtype=float)[nearest[hit]]
    return cl, cd, cm


def _source_supports_ground(source: Any) -> bool:
    """Report whether *source* supports ground effect (height keyword)."""
    if getattr(source, "supports_ground", False):
        return True
    polar_fn = getattr(source, "polar", None)
    if not callable(polar_fn):
        return False
    try:
        sig = inspect.signature(polar_fn)
        return "height" in sig.parameters
    except (TypeError, ValueError):
        return False


def build_polar_table(
    source: SectionPolarSource,
    section: str | np.ndarray,
    Re_values: Sequence[float] | np.ndarray,
    alpha_deg: Sequence[float] | np.ndarray,
    flap_deg: Sequence[float] | np.ndarray = (0.0,),
    x_hinge: float | None = None,
    y_hinge: float = 0.5,
    h_over_c: Sequence[float] | np.ndarray | None = None,
    workers: int | None = None,
) -> PolarTable:
    """Build a polar table over Reynolds number, flap deflection and height.

    One source call runs per grid point: (Re, flap) for free air or
    (Re, flap, height) when a height axis is requested. A source that
    offers a ``run_xfoil_batch`` hook runs its cases through that hook
    (found by feature check, never by type test). Other sources run
    through ``source.polar``, in parallel threads unless ``workers=1``.
    A case that fails to converge becomes a NaN slice; any other error
    stops the build.

    Parameters
    ----------
    source : SectionPolarSource
        Section polar provider (XFOIL now, the in-house 2D solver
        later).
    section : str or np.ndarray
        A NACA name or points (N, 2) in Selig order (unit chord, the
        leading edge at (0, 0), the trailing-edge midpoint at (1, 0)).
    Re_values : sequence of float
        Reynolds numbers [-], at least one positive value. Sorted
        ascending in the table.
    alpha_deg : sequence of float
        Angles of attack [deg], at least one unique value.
    flap_deg : sequence of float
        Flap deflections [deg], positive down. The default (0.0,)
        builds the clean airfoil.
    x_hinge : float or None
        Hinge chordwise position, fraction of chord [-]. Required for
        a nonzero deflection.
    y_hinge : float
        Hinge vertical position, fraction of chord [-].
    h_over_c : sequence of float or None
        Heights above ground in chords [-], strictly positive. None
        builds a free-air table.
    workers : int or None
        Thread pool size for sources without a batch hook, or the
        worker count forwarded to the hook. None uses the pool
        default. 1 runs sequentially.

    Returns
    -------
    PolarTable
        The built table, with NaN where the source did not converge.

    Raises
    ------
    ValueError
        If ``h_over_c`` is requested but ``source`` does not support
        ground effect.
    """
    res = np.atleast_1d(np.asarray(Re_values, dtype=float))
    if res.size == 0:
        raise ValueError("Re_values must hold at least one value.")
    if not np.all(np.isfinite(res)):
        raise ValueError("Re_values must hold finite values only.")
    if np.any(res <= 0.0):
        raise ValueError("Re_values must hold positive values only.")
    if np.unique(res).size != res.size:
        raise ValueError("Re_values must hold unique values.")
    requested_flap = np.atleast_1d(np.asarray(flap_deg, dtype=float))
    if requested_flap.size == 0:
        raise ValueError("flap_deg must hold at least one value.")
    if not np.all(np.isfinite(requested_flap)):
        raise ValueError("flap_deg must hold finite values only.")
    if np.unique(requested_flap).size != requested_flap.size:
        raise ValueError("flap_deg must hold unique values.")
    requested_alpha = np.atleast_1d(np.asarray(alpha_deg, dtype=float))
    if requested_alpha.size == 0:
        raise ValueError("alpha_deg must hold at least one value.")
    if not np.all(np.isfinite(requested_alpha)):
        raise ValueError("alpha_deg must hold finite values only.")
    if np.unique(requested_alpha).size != requested_alpha.size:
        raise ValueError("alpha_deg must hold unique values.")
    if workers is not None:
        workers = operator.index(workers)
        if workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}.")

    if h_over_c is not None:
        if not _source_supports_ground(source):
            raise ValueError(
                f"Source {type(source).__name__} does not support ground effect."
            )
        h_sorted: np.ndarray | None = _as_sorted_axis(h_over_c, "h_over_c")
        if np.any(h_sorted <= 0.0):
            raise ValueError("h_over_c must hold positive values only.")
    else:
        h_sorted = None

    order_re = np.argsort(res, kind="stable")
    res = res[order_re]
    flap_sorted = np.sort(requested_flap)
    alpha_rad = np.radians(requested_alpha)
    if requested_alpha.size > 1:
        step = float(np.diff(np.sort(np.unique(requested_alpha))).min())
        tolerance_rad = float(np.radians(0.25 * step))
    else:
        tolerance_rad = 1e-6

    if h_sorted is None:
        grid_items = [
            (i_re, j_flap, None, float(re), float(flap), None)
            for i_re, re in enumerate(res)
            for j_flap, flap in enumerate(flap_sorted)
        ]
    else:
        grid_items = [
            (i_re, j_flap, k_h, float(re), float(flap), float(h))
            for i_re, re in enumerate(res)
            for j_flap, flap in enumerate(flap_sorted)
            for k_h, h in enumerate(h_sorted)
        ]
    flaps = [
        _flap_argument(flap, x_hinge, float(y_hinge))
        for _, _, _, _, flap, _ in grid_items
    ]
    polars = _run_cases(
        source, section, requested_alpha, grid_items, flaps, workers
    )

    if h_sorted is None:
        n_re, n_flap, n_alpha = res.size, flap_sorted.size, requested_alpha.size
        cl = np.full((n_re, n_flap, n_alpha), np.nan)
        cd = np.full((n_re, n_flap, n_alpha), np.nan)
        cm = np.full((n_re, n_flap, n_alpha), np.nan)
        for (i_re, j_flap, _, _, _, _), polar in zip(grid_items, polars):
            row_cl, row_cd, row_cm = _place_on_grid(polar, alpha_rad, tolerance_rad)
            cl[i_re, j_flap, :] = row_cl
            cd[i_re, j_flap, :] = row_cd
            cm[i_re, j_flap, :] = row_cm
    else:
        n_re, n_flap, n_h, n_alpha = (
            res.size,
            flap_sorted.size,
            h_sorted.size,
            requested_alpha.size,
        )
        cl = np.full((n_re, n_flap, n_h, n_alpha), np.nan)
        cd = np.full((n_re, n_flap, n_h, n_alpha), np.nan)
        cm = np.full((n_re, n_flap, n_h, n_alpha), np.nan)
        for (i_re, j_flap, k_h, _, _, _), polar in zip(grid_items, polars):
            row_cl, row_cd, row_cm = _place_on_grid(polar, alpha_rad, tolerance_rad)
            cl[i_re, j_flap, k_h, :] = row_cl
            cd[i_re, j_flap, k_h, :] = row_cd
            cm[i_re, j_flap, k_h, :] = row_cm

    version = getattr(source, "_version", None) or "unknown"
    opts: dict[str, Any] = {"x_hinge": x_hinge, "y_hinge": float(y_hinge)}
    if h_sorted is not None:
        opts["h_over_c"] = [float(v) for v in h_sorted]
    info: dict[str, Any] = {
        "source": type(source).__name__,
        "section": section if isinstance(section, str) else "coordinates",
        "options": opts,
        "version": version,
    }
    return PolarTable(
        Re=res,
        flap_deg=flap_sorted,
        alpha_deg=requested_alpha,
        Cl=cl,
        Cd=cd,
        Cm=cm,
        source_info=info,
        h_over_c=h_sorted,
    )


def _run_cases(
    source: SectionPolarSource,
    section: str | np.ndarray,
    requested_alpha: np.ndarray,
    grid_items: list[tuple[int, int, int | None, float, float, float | None]],
    flaps: list[tuple[float, float, float] | None],
    workers: int | None,
) -> list[TabulatedAirfoil | None]:
    """Run one source polar per grid point, in input order.

    Parameters
    ----------
    source : SectionPolarSource
        Section polar provider.
    section : str or np.ndarray
        A NACA name or points (N, 2) in Selig order.
    requested_alpha : np.ndarray
        Requested angles of attack [deg].
    grid_items : list
        Tuples of (i_re, j_flap, k_h, re, flap, height).
    flaps : list
        One source flap tuple (or None) per case.
    workers : int or None
        Thread pool size, or the count forwarded to the batch hook.

    Returns
    -------
    list
        One TabulatedAirfoil per case, or None for a case that did not
        converge. Any other error propagates.
    """
    hook = getattr(source, "run_xfoil_batch", None)
    if callable(hook):
        cases = [
            {
                "section": section,
                "Re": re,
                "alphas_deg": [float(a) for a in requested_alpha],
                "flap": flap,
                **({} if h is None else {"height": h}),
            }
            for (_, _, _, re, _, h), flap in zip(grid_items, flaps)
        ]
        results = hook(cases, workers)
        return [_checked_item(item) for item in results]

    def run_one(args: tuple[float, tuple[float, float, float] | None, float | None]):
        re, flap, h = args
        try:
            if h is None:
                return source.polar(
                    section, re, requested_alpha, flap=flap
                )
            return source.polar(
                section, re, requested_alpha, flap=flap, height=h
            )
        except RuntimeError as exc:
            if _source_failed_as_gap(exc):
                return None
            raise

    jobs = [(re, flap, h) for (_, _, _, re, _, h), flap in zip(grid_items, flaps)]
    if workers == 1:
        return [run_one(job) for job in jobs]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as pool:
        return list(pool.map(run_one, jobs))


def _checked_item(item: object) -> TabulatedAirfoil | None:
    """Map one batch-hook result to a polar or a NaN gap.

    Parameters
    ----------
    item : object
        One entry returned by the ``run_xfoil_batch`` hook: a
        TabulatedAirfoil or an exception object.

    Returns
    -------
    TabulatedAirfoil or None
        The polar, or None for a pair that did not converge.

    Raises
    ------
    BaseException
        The stored error, when it is not a convergence failure.
    """
    if isinstance(item, BaseException):
        if _source_failed_as_gap(item):
            return None
        raise item
    return item  # type: ignore[return-value]
