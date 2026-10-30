# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Define all dataclasses of Ventorum.

The dataclasses hold the airfoil models, the geometry, the flight
conditions, the solver settings and the results.

Design philosophy
-----------------
Every piece of data that flows between modules is a dataclass with
``slots=True`` (mutable) so that solvers can populate result fields
incrementally. Some supporting dataclasses in other modules are frozen.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from dataclasses import dataclass, field
from typing import Literal, Any

import numpy as np

from ventorum.core.constants import RHO_SL


# ═══════════════════════════════════════════════════════════════════════════════
# Airfoil section models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class LinearAirfoil:
    """Linear (thin-airfoil) aerodynamic model for a 2D section.

    Parameters
    ----------
    name : str
        Descriptive label (e.g. ``"NACA0012"``).
    a0 : float
        Lift-curve slope [1/rad].  Default is 2π (thin-airfoil theory).
    alpha_L0 : float
        Zero-lift angle of attack [rad].  Positive for cambered airfoils with
        negative zero-lift AoA convention (i.e. ``alpha_L0 < 0`` lifts at
        ``alpha = 0``).
    Cd0 : float
        Zero-lift profile drag coefficient.
    Cm0 : float
        Zero-lift pitching moment coefficient.
    """

    name: str = "Thin Airfoil"
    a0: float = 2.0 * np.pi
    alpha_L0: float = 0.0
    Cd0: float = 0.0
    Cm0: float = 0.0

    def clone(self) -> LinearAirfoil:
        """Fast explicit clone avoiding reflection."""
        return LinearAirfoil(
            name=self.name,
            a0=self.a0,
            alpha_L0=self.alpha_L0,
            Cd0=self.Cd0,
            Cm0=self.Cm0,
        )

    def __copy__(self) -> LinearAirfoil:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> LinearAirfoil:
        return self.clone()

    # -- convenience ------------------------------------------------------------
    def Cl(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """Section lift coefficient at angle of attack *alpha* [rad]."""
        return self.a0 * (alpha - self.alpha_L0)

    def Cd(self, alpha: float | np.ndarray) -> float | np.ndarray:  # noqa: ARG002
        """Section drag coefficient (constant for linear model)."""
        if isinstance(alpha, (int, float)):
            return float(self.Cd0)
        return np.full(np.shape(alpha), self.Cd0, dtype=float)

    def Cm(self, alpha: float | np.ndarray) -> float | np.ndarray:  # noqa: ARG002
        """Section pitching moment coefficient (constant for linear model)."""
        if isinstance(alpha, (int, float)):
            return float(self.Cm0)
        return np.full(np.shape(alpha), self.Cm0, dtype=float)


@dataclass(slots=True)
class TabulatedAirfoil:
    """Nonlinear aerodynamic model from polar-data tables.

    The tables are stored as the user gives them. Sorting by alpha, the
    merge of repeated alpha values (the present rule: mean values, with a
    warning) and the interpolation are done together, with one permutation
    for all arrays, when the interpolation is built. A scalar *Cm_data* is a
    constant pitching moment. If you assign new arrays after construction,
    the interpolation is made again. If you change an array in place, assign
    it again (for example ``af.Cl_data = af.Cl_data``) to clear the cached
    interpolation.

    Parameters
    ----------
    name : str
        Descriptive label (e.g. ``"NACA2412_Re1e6"``).
    alpha : np.ndarray
        Angle-of-attack array [rad].
    Cl_data : np.ndarray
        Lift-coefficient array (same length as *alpha*).
    Cd_data : np.ndarray
        Drag-coefficient array.
    Cm_data : np.ndarray | None
        Pitching-moment array (optional).
    Re : float | None
        Reynolds number this polar was generated / measured at.
    Cl_circulation_data : np.ndarray | None
        Circulation lift coefficient ``2 Gamma / (V c)`` [-] at *alpha*
        (optional, same length as *alpha*). A section in ground effect has
        two lift definitions: ``Cl_data`` holds the force (pressure) lift,
        this array the lift of the total circulation. They differ near a
        wall because the Kutta-Joukowski relation on the total circulation
        does not give the force there (C. Coulliette and A. Plotkin,
        "Aerofoil ground effect revisited", The Aeronautical Journal 100,
        1996). None (default) means that the polar gives no circulation lift. In free
        air the two definitions are equal for a single section.
    gap_mask : np.ndarray | None
        Gap mask of the table (optional): one bool per interval between
        neighbouring angles of the sorted table (see :meth:`tables`),
        shape ``(n_alpha - 1,)``. True marks an interval that bridges a
        failed (dropped) angle of the source: there the values come from
        the interpolation across the gap, not from a solution. A lookup
        strictly inside a bridged interval gives a RuntimeWarning, once
        per polar. None (default) or no True value means no known gap
        (for example a polar file written before the mask existed).
    """

    name: str = "Tabulated Airfoil"
    alpha: np.ndarray = field(default_factory=lambda: np.array([]))
    Cl_data: np.ndarray = field(default_factory=lambda: np.array([]))
    Cd_data: np.ndarray = field(default_factory=lambda: np.array([]))
    Cm_data: np.ndarray | None = None
    Re: float | None = None
    Cl_circulation_data: np.ndarray | None = None
    gap_mask: np.ndarray | None = None
    _gap_warned: bool = field(default=False, init=False, repr=False)
    _cached_a0: float | None = field(default=None, init=False, repr=False)
    _cached_alpha_L0: float | None = field(default=None, init=False, repr=False)
    _interp: dict | None = field(default=None, init=False, repr=False)
    _prepared_arrays: tuple | None = field(default=None, init=False, repr=False)
    _prepared_circulation: np.ndarray | None = field(default=None, init=False, repr=False)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name == "gap_mask":
            object.__setattr__(self, "_gap_warned", False)
        if name in _TABLE_FIELDS:
            # New data: the interpolation and the linear fit are made again on next use.
            object.__setattr__(self, "_interp", None)
            object.__setattr__(self, "_cached_a0", None)
            object.__setattr__(self, "_cached_alpha_L0", None)
            object.__setattr__(self, "_prepared_arrays", None)
            object.__setattr__(self, "_prepared_circulation", None)

    def __post_init__(self) -> None:
        # Initialisation does not prepare arrays; they are prepared on first use.
        pass

    def clone(self) -> TabulatedAirfoil:
        """Fast explicit clone avoiding reflection."""
        inst = TabulatedAirfoil(
            name=self.name,
            alpha=self.alpha.copy(),
            Cl_data=self.Cl_data.copy(),
            Cd_data=self.Cd_data.copy(),
            Cm_data=self.Cm_data.copy() if self.Cm_data is not None else None,
            Re=self.Re,
            Cl_circulation_data=(
                self.Cl_circulation_data.copy() if self.Cl_circulation_data is not None else None
            ),
            gap_mask=None if self.gap_mask is None else np.array(self.gap_mask, dtype=bool),
        )
        inst._cached_a0 = self._cached_a0
        inst._cached_alpha_L0 = self._cached_alpha_L0
        inst._prepared_arrays = self._prepared_arrays
        inst._prepared_circulation = self._prepared_circulation
        return inst

    def __copy__(self) -> TabulatedAirfoil:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> TabulatedAirfoil:
        return self.clone()

    def _prepare_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Sort tables by alpha and merge repeated alphas. Returns (alpha, Cl, Cd, Cm).

        This is called once per data change. The result is cached in
        ``_prepared_arrays``.
        """
        if self._prepared_arrays is not None:
            return self._prepared_arrays

        a = np.atleast_1d(np.asarray(self.alpha, dtype=float))
        n = a.size
        tables = {
            "Cl_data": self.Cl_data,
            "Cd_data": self.Cd_data,
            "Cm_data": self.Cm_data,
            "Cl_circulation_data": self.Cl_circulation_data,
        }
        arrs: dict[str, np.ndarray | None] = {}
        for key, val in tables.items():
            if val is None:
                arrs[key] = None
                continue
            arr = np.asarray(val, dtype=float)
            if arr.ndim == 0 and key == "Cm_data":
                arr = np.full(n, float(arr))
            arr = np.atleast_1d(arr)
            if arr.size != n:
                raise ValueError(
                    f"TabulatedAirfoil {self.name!r}: {key} has {arr.size} values, alpha has {n}."
                )
            arrs[key] = arr

        if n > 1:
            order = np.argsort(a, kind="stable")
            a = a[order]
            arrs = {k: (None if v is None else v[order]) for k, v in arrs.items()}
            uniq, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
            if uniq.size < n:
                warnings.warn(
                    f"TabulatedAirfoil {self.name!r}: {n - uniq.size} repeated alpha value(s); "
                    "the rows with the same alpha are replaced by their mean.",
                    RuntimeWarning, stacklevel=4,
                )
                a = uniq
                arrs = {
                    k: (None if v is None else np.bincount(inverse, weights=v) / counts)
                    for k, v in arrs.items()
                }

        cl = arrs["Cl_data"]
        cd = arrs["Cd_data"]
        cm = arrs["Cm_data"]
        self._prepared_circulation = arrs["Cl_circulation_data"]
        self._prepared_arrays = (a, cl, cd, cm)
        return self._prepared_arrays

    def tables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Return the tables sorted by alpha, with repeated angles merged.

        The attributes keep the arrays as the user gave them (any order).
        Code that needs ascending angles (for example ``np.interp``) must
        use these arrays.

        Returns
        -------
        alpha : numpy.ndarray
            Angles of attack [rad], ascending, each angle once.
        Cl, Cd : numpy.ndarray
            Lift and drag coefficients at *alpha*.
        Cm : numpy.ndarray or None
            Pitching-moment coefficients at *alpha* (None without moment data).
        """
        return self._prepare_arrays()

    # -- interpolation ----------------------------------------------------------
    # Shape-preserving cubic (PCHIP) interpolation: continuous first
    # derivative (needed by Newton's method near the maximum lift) and no
    # overshoot between the points. Outside the table the end values are kept.
    def _pchip(self) -> dict:
        a, cl, cd, cm = self._prepare_arrays()
        if self._interp is None:
            from scipy.interpolate import PchipInterpolator

            f = {
                "Cl": PchipInterpolator(a, np.asarray(cl, dtype=float), extrapolate=False),
                "Cd": PchipInterpolator(a, np.asarray(cd, dtype=float), extrapolate=False),
            }
            if cm is not None:
                f["Cm"] = PchipInterpolator(a, np.asarray(cm, dtype=float), extrapolate=False)
            f["dCl"] = f["Cl"].derivative()
            if self._prepared_circulation is not None:
                f["Clc"] = PchipInterpolator(a, np.asarray(self._prepared_circulation, dtype=float), extrapolate=False)
                f["dClc"] = f["Clc"].derivative()
            self._interp = f
        return self._interp

    # -- gaps -----------------------------------------------------------------------
    @property
    def has_gaps(self) -> bool:
        """True when the gap mask marks at least one bridged interval."""
        return self.gap_mask is not None and bool(np.any(self.gap_mask))

    def in_gap(self, alpha: float | np.ndarray) -> bool | np.ndarray:
        """Return True where *alpha* [rad] is strictly inside a bridged interval of the gap mask.

        Raises
        ------
        ValueError
            If the gap mask does not have one value per interval of the
            sorted table.
        """
        a = np.asarray(alpha, dtype=float)
        out = np.zeros(a.shape, dtype=bool)
        if self.has_gaps:
            table = self._prepare_arrays()[0]
            mask = np.atleast_1d(np.asarray(self.gap_mask, dtype=bool))
            if mask.size != table.size - 1:
                raise ValueError(
                    f"TabulatedAirfoil {self.name!r}: gap_mask has {mask.size} values, the sorted table has "
                    f"{table.size - 1} intervals."
                )
            idx = np.searchsorted(table, a, side="right") - 1
            valid = (idx >= 0) & (idx < mask.size)
            safe = np.clip(idx, 0, max(mask.size - 1, 0))
            out = valid & (a > table[safe]) & mask[safe]
        return out if np.ndim(alpha) else bool(out)

    def _check_gap(self, alpha) -> None:
        """Warn once per polar when a lookup is inside a bridged interval."""
        if self._gap_warned or not self.has_gaps:
            return
        if np.any(self.in_gap(alpha)):
            object.__setattr__(self, "_gap_warned", True)
            warnings.warn(
                f"TabulatedAirfoil {self.name!r}: a lookup is inside an angle interval that the polar bridges "
                "across a failed angle of its source; the value there is interpolated, not a solution.",
                RuntimeWarning, stacklevel=4,
            )

    def _eval(self, key: str, alpha, data) -> np.ndarray:
        a = np.asarray(alpha, dtype=float)
        self._check_gap(a)
        f = self._pchip()
        prep = self._prepare_arrays()
        lo, hi = float(prep[0][0]), float(prep[0][-1])
        out = f[key](np.clip(a, lo, hi))
        return out if np.ndim(alpha) else float(out)

    def Cl(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """Return the lift coefficient at *alpha* [rad]; end values outside the table."""
        return self._eval("Cl", alpha, self.Cl_data)

    def Cd(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """Return the drag coefficient at *alpha* [rad]; end values outside the table."""
        return self._eval("Cd", alpha, self.Cd_data)

    def Cm(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """Return the pitching-moment coefficient at *alpha* [rad]; zero if there is no Cm data."""
        if self.Cm_data is not None:
            return self._eval("Cm", alpha, self.Cm_data)
        if np.ndim(alpha) == 0:
            return 0.0
        return np.zeros(np.shape(alpha), dtype=float)

    def Cl_slope(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """dCl/dalpha [1/rad]; zero outside the table."""
        a = np.asarray(alpha, dtype=float)
        self._check_gap(a)
        f = self._pchip()
        prep = self._prepare_arrays()
        lo, hi = float(prep[0][0]), float(prep[0][-1])
        out = np.where((a >= lo) & (a <= hi), f["dCl"](np.clip(a, lo, hi)), 0.0)
        return out if np.ndim(alpha) else float(out)

    @property
    def has_circulation_lift(self) -> bool:
        """True when the polar holds the circulation lift (``Cl_circulation_data``)."""
        return self.Cl_circulation_data is not None

    def Cl_circulation(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """Return the circulation lift coefficient at *alpha* [rad]; end values outside the table.

        Raises
        ------
        ValueError
            If the polar has no circulation lift data.
        """
        if self.Cl_circulation_data is None:
            raise ValueError(f"TabulatedAirfoil {self.name!r} has no circulation lift data.")
        return self._eval("Clc", alpha, self.Cl_circulation_data)

    def Cl_circulation_slope(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """Return d(Cl_circulation)/dalpha [1/rad]; zero outside the table.

        Raises
        ------
        ValueError
            If the polar has no circulation lift data.
        """
        if self.Cl_circulation_data is None:
            raise ValueError(f"TabulatedAirfoil {self.name!r} has no circulation lift data.")
        a = np.asarray(alpha, dtype=float)
        self._check_gap(a)
        f = self._pchip()
        prep = self._prepare_arrays()
        lo, hi = float(prep[0][0]), float(prep[0][-1])
        out = np.where((a >= lo) & (a <= hi), f["dClc"](np.clip(a, lo, hi)), 0.0)
        return out if np.ndim(alpha) else float(out)

    def _compute_linear_fit(self) -> None:
        """Lift slope and zero-lift angle from the points near the zero-lift angle.

        The zero lift angle is first found where Cl changes sign nearest to
        alpha = 0; the line is then fitted to the points from 4 deg below to
        6 deg above it (at least 3 points), so a cambered or low-Reynolds
        polar is fitted in its own linear range.
        """
        a, cl, _, _ = self._prepare_arrays()
        if len(a) < 2:
            self._cached_a0 = 2.0 * np.pi
            self._cached_alpha_L0 = 0.0
            return
        sign_change = np.flatnonzero(np.sign(cl[:-1]) * np.sign(cl[1:]) <= 0)
        if sign_change.size:
            k = sign_change[np.argmin(np.abs(a[sign_change]))]
            da = a[k + 1] - a[k]
            aL0_guess = a[k] - cl[k] * da / (cl[k + 1] - cl[k]) if cl[k + 1] != cl[k] else a[k]
        else:
            aL0_guess = 0.0
        mask = (a >= aL0_guess - np.radians(4.0)) & (a <= aL0_guess + np.radians(6.0))
        if mask.sum() < 3:
            order = np.argsort(np.abs(a - aL0_guess))[:3]
            mask = np.zeros_like(a, dtype=bool)
            mask[order] = True
        coeffs = np.polyfit(a[mask], cl[mask], 1)
        self._cached_a0 = float(coeffs[0])
        self._cached_alpha_L0 = float(-coeffs[1] / coeffs[0]) if abs(coeffs[0]) > 1e-12 else 0.0

    @property
    def a0(self) -> float:
        """Approximate lift-curve slope from the linear region of the polar."""
        if self._cached_a0 is None:
            self._compute_linear_fit()
        return self._cached_a0

    @property
    def alpha_L0(self) -> float:
        """Approximate zero-lift angle of attack [rad]."""
        if self._cached_alpha_L0 is None:
            self._compute_linear_fit()
        return self._cached_alpha_L0


# Fields of TabulatedAirfoil whose change clears the cached interpolation.
_TABLE_FIELDS = frozenset({"alpha", "Cl_data", "Cd_data", "Cm_data", "Cl_circulation_data"})


# ═══════════════════════════════════════════════════════════════════════════════
# Interpolation of many tabulated airfoils together (private)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_pchip_batch(airfoils: Any) -> None:
    """Build the cached interpolation of many tabulated airfoils together.

    The airfoils that have the same sorted angles and the same tables
    share one PCHIP construction per coefficient: their tables are the
    columns of one array. The construction is column by column, so each
    airfoil gets the same coefficients as from its own construction
    (:meth:`TabulatedAirfoil._pchip`); only the call overhead is shared.
    An airfoil that already has its interpolation does not change.

    Parameters
    ----------
    airfoils : sequence
        Airfoil models. Only the instances of :class:`TabulatedAirfoil`
        (not of a subclass) without a cached interpolation are used.
    """
    from scipy.interpolate import PchipInterpolator, PPoly

    groups: dict[tuple, list[TabulatedAirfoil]] = {}
    seen: set[int] = set()
    for af in airfoils:
        if type(af) is not TabulatedAirfoil or af._interp is not None or id(af) in seen:
            continue
        seen.add(id(af))
        a, _, _, cm = af._prepare_arrays()
        if a.size < 2:
            continue  # The construction of the airfoil itself gives the error.
        key = (a.size, a.tobytes(), cm is not None, af._prepared_circulation is not None)
        groups.setdefault(key, []).append(af)
    for (_, _, has_cm, has_circ), members in groups.items():
        if len(members) < 2:
            continue
        a = members[0]._prepare_arrays()[0]
        columns = {
            "Cl": [m._prepare_arrays()[1] for m in members],
            "Cd": [m._prepare_arrays()[2] for m in members],
        }
        if has_cm:
            columns["Cm"] = [m._prepare_arrays()[3] for m in members]
        if has_circ:
            columns["Clc"] = [m._prepared_circulation for m in members]
        built = {
            name: PchipInterpolator(a, np.stack([np.asarray(c, dtype=float) for c in cols], axis=1),
                                    extrapolate=False)
            for name, cols in columns.items()
        }
        built["dCl"] = built["Cl"].derivative()
        if has_circ:
            built["dClc"] = built["Clc"].derivative()
        for j, af in enumerate(members):
            f = {
                name: PPoly.construct_fast(np.ascontiguousarray(pp.c[:, :, j]), pp.x, extrapolate=False)
                for name, pp in built.items()
            }
            object.__setattr__(af, "_interp", f)


def _piecewise_values(x: np.ndarray, c: np.ndarray, columns: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return the values of stacked piecewise polynomials, one point per polynomial.

    Parameters
    ----------
    x : numpy.ndarray
        Breakpoints [rad], ascending, shape ``(n,)``, shared by all polynomials.
    c : numpy.ndarray
        Coefficients, highest power first, shape ``(k, n - 1, m)``: column
        ``j`` holds polynomial ``j`` (the layout of ``scipy.interpolate.PPoly``).
    columns : numpy.ndarray
        Polynomial (column) of each point, shape ``(p,)``.
    alpha : numpy.ndarray
        Points [rad] inside ``[x[0], x[-1]]``, shape ``(p,)``.

    Returns
    -------
    numpy.ndarray
        Values at the points, shape ``(p,)``. An interval includes its left
        breakpoint; the last interval also includes ``x[-1]``. The sum
        ``sum_m c[k-1-m] s^m`` runs from the constant term up, with
        ``s = alpha - x[i]`` and the powers made by repeated products.
    """
    i = np.clip(np.searchsorted(x, alpha, side="right") - 1, 0, x.size - 2)
    s = alpha - x[i]
    k = c.shape[0]
    out = c[k - 1, i, columns] * 1.0
    z = np.ones_like(s)
    for p in range(1, k):
        z = z * s
        out = out + c[k - 1 - p, i, columns] * z
    return out


class _LiftLookup:
    """Lift coefficient and lift slope of many airfoils, each at its own angle (private).

    The lookup gives the same values as :meth:`TabulatedAirfoil.Cl` and
    :meth:`TabulatedAirfoil.Cl_slope` (or the circulation lift and its
    slope) of each airfoil, with one vectorised evaluation for each group
    of airfoils with the same angles. It reads the cached interpolation of
    the airfoils, so it must be made again when the data of an airfoil
    change. The entries that are not an instance of
    :class:`TabulatedAirfoil` (not of a subclass) are not handled
    (``handled`` is False): the caller reads them itself.

    Parameters
    ----------
    airfoils : sequence
        Airfoil model of each entry.
    circulation : sequence of bool
        True for an entry that reads the circulation lift.
    """

    def __init__(self, airfoils: Any, circulation: Any) -> None:
        airfoils = list(airfoils)
        circ = np.asarray(circulation, dtype=bool)
        _build_pchip_batch([af for af, c in zip(airfoils, circ) if type(af) is TabulatedAirfoil])
        self.n = len(airfoils)
        self.handled = np.zeros(self.n, dtype=bool)
        groups: dict[tuple, dict] = {}
        for idx, (af, c) in enumerate(zip(airfoils, circ)):
            if type(af) is not TabulatedAirfoil:
                continue
            if c and af.Cl_circulation_data is None:
                raise ValueError(f"TabulatedAirfoil {af.name!r} has no circulation lift data.")
            f = af._pchip()
            pp, dpp = (f["Clc"], f["dClc"]) if c else (f["Cl"], f["dCl"])
            x = np.asarray(pp.x, dtype=float)
            key = (x.size, x.tobytes(), pp.c.shape[0], dpp.c.shape[0])
            g = groups.setdefault(key, {"x": x, "entries": [], "cols": [], "pp": {}, "gap": []})
            col = g["pp"].setdefault((id(af), bool(c)), (len(g["pp"]), pp, dpp))[0]
            g["entries"].append(idx)
            g["cols"].append(col)
            if af.has_gaps:
                g["gap"].append((len(g["entries"]) - 1, af))
            self.handled[idx] = True
        self.groups = []
        for g in groups.values():
            polys = sorted(g["pp"].values(), key=lambda t: t[0])
            self.groups.append((
                np.asarray(g["entries"], dtype=np.intp),
                g["x"],
                np.stack([np.asarray(t[1].c, dtype=float) for t in polys], axis=2),
                np.stack([np.asarray(t[2].c, dtype=float) for t in polys], axis=2),
                np.asarray(g["cols"], dtype=np.intp),
                g["gap"],
            ))

    def __call__(self, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the lift coefficient [-] and the lift slope [1/rad] at the angles *alpha* [rad].

        The values of the entries that are not handled are NaN.
        """
        alpha = np.asarray(alpha, dtype=float)
        cl = np.full(self.n, np.nan)
        slope = np.full(self.n, np.nan)
        for entries, x, c, dc, cols, gap in self.groups:
            a = alpha[entries]
            for pos, af in gap:
                af._check_gap(a[pos])
            lo, hi = x[0], x[-1]
            ac = np.clip(a, lo, hi)
            cl[entries] = _piecewise_values(x, c, cols, ac)
            slope[entries] = np.where((a >= lo) & (a <= hi), _piecewise_values(x, dc, cols, ac), 0.0)
        return cl, slope

# Type alias for convenience
AirfoilType = LinearAirfoil | TabulatedAirfoil


# ═══════════════════════════════════════════════════════════════════════════════
# Control surfaces
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class ControlSurface:
    """A hinged control surface (flap, aileron, elevator, rudder) of a lifting surface.

    Parameters
    ----------
    name : str
        Identifier of the control surface.
    eta_start : float
        Span fraction of the semi-span where the control starts [-], in [0, 1].
    eta_end : float
        Span fraction where the control ends [-], in [0, 1].
    hinge_x_c : float
        Hinge position, fraction of the local chord from the leading edge [-],
        in (0, 1).
    deflection : float
        Deflection angle [rad], positive right-hand rotation about the strip hinge axis.
    symmetric : bool
        If True, symmetric deflection on left copy; if False, antisymmetric.
    """

    name: str = "flap"
    eta_start: float = 0.0
    eta_end: float = 1.0
    hinge_x_c: float = 0.75
    deflection: float = 0.0
    symmetric: bool = True

    def clone(self) -> ControlSurface:
        """Fast explicit clone avoiding reflection."""
        return ControlSurface(
            name=self.name,
            eta_start=self.eta_start,
            eta_end=self.eta_end,
            hinge_x_c=self.hinge_x_c,
            deflection=self.deflection,
            symmetric=self.symmetric,
        )

    def __copy__(self) -> ControlSurface:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> ControlSurface:
        return self.clone()

    @property
    def deflection_deg(self) -> float:
        """Deflection angle in degrees [deg]."""
        return float(np.degrees(self.deflection))


# ═══════════════════════════════════════════════════════════════════════════════
# Wing geometry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class NodeDisplacements:
    """Displacements [m] of the leading-edge and trailing-edge lattice nodes of the defining half of a surface.

    Parameters
    ----------
    le : np.ndarray
        Leading-edge node displacements [m], shape (n_edges, 3).
    te : np.ndarray
        Trailing-edge node displacements [m], shape (n_edges, 3).
    eta : np.ndarray or None
        Optional spanwise stations of the edges that the displacements were
        made for, as fractions of the semi-span [-], shape (n_edges,), as
        :func:`ventorum.geometry.undeformed_nodes` gives them. When it is
        set, :func:`ventorum.geometry.lattice.build_lattice` compares it with the
        edge stations of the mesh (absolute tolerance 1e-12) and raises
        ``ValueError`` if they are different. None (the default) skips this
        check.
    """

    le: np.ndarray
    te: np.ndarray
    eta: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "le", np.asarray(self.le, dtype=float))
        object.__setattr__(self, "te", np.asarray(self.te, dtype=float))
        if self.eta is not None:
            object.__setattr__(self, "eta", np.asarray(self.eta, dtype=float))

    def clone(self) -> NodeDisplacements:
        """Fast explicit clone."""
        return NodeDisplacements(
            le=np.asarray(self.le, dtype=float).copy(),
            te=np.asarray(self.te, dtype=float).copy(),
            eta=None if self.eta is None else np.asarray(self.eta, dtype=float).copy(),
        )

    def __copy__(self) -> NodeDisplacements:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> NodeDisplacements:
        return self.clone()


@dataclass(slots=True)
class WingSection:
    """Properties at a single spanwise station of a lifting surface.

    The user defines a small number of *sections* (at least root + tip);
    the solver interpolates between them when discretising into panels.

    Parameters
    ----------
    y_frac : float
        Spanwise position as a fraction of the semi-span, in ``[0, 1]``
        (0 = root, 1 = tip).
    chord : float
        Local chord length [m].
    twist : float
        Geometric twist angle [rad].  Positive → leading edge up (wash-in).
    x_le : float | None
        Leading-edge *x*-offset from the root LE [m].  If *None*, computed
        from the surface-level ``sweep_le`` parameter during discretisation.
    z_le : float | None
        Leading-edge *z*-offset from the root LE [m].  If *None*, computed
        from the surface-level ``dihedral`` parameter during discretisation.
    airfoil : AirfoilType
        Aerodynamic model for this section.
    shape : Any, optional
        Airfoil shape of the section. Accepts any input that
        ventorum.airfoil.airfoil() supports (NACA string, integer,
        coordinate array, or AirfoilShape). The 3D panel method meshes it
        (None: NACA 0012). The vortex lattice (two or more chordwise panels)
        puts its lattice on the mean line of the shape when every section of
        the surface gives one; the polar alpha_L0 and Cm0 then do not enter
        the boundary condition. A mean line alpha_L0 that differs from the
        polar alpha_L0 by more than 0.5 deg gives a trust warning. The
        lifting-line solvers (linear, nonlinear and Fourier) do not use the
        shape: they use the section polar (``airfoil``) only, and a section
        with a shape gives a note in the trust warnings. The vortex lattice
        with one chordwise panel does not use the shape either: it turns the
        normal by the polar alpha_L0 and uses the polar Cm0, with a note in
        the trust warnings. Default is None.
    """

    y_frac: float = 0.0
    chord: float = 1.0
    twist: float = 0.0
    x_le: float | None = None
    z_le: float | None = None
    airfoil: AirfoilType = field(default_factory=LinearAirfoil)
    shape: Any = None

    def clone(self) -> WingSection:
        """Fast explicit clone."""
        af = self.airfoil.clone() if hasattr(self.airfoil, "clone") else self.airfoil
        shape_clone = self.shape
        if hasattr(self.shape, "clone"):
            shape_clone = self.shape.clone()
        elif hasattr(self.shape, "copy"):
            shape_clone = self.shape.copy()
        return WingSection(
            y_frac=self.y_frac,
            chord=self.chord,
            twist=self.twist,
            x_le=self.x_le,
            z_le=self.z_le,
            airfoil=af,
            shape=shape_clone,
        )

    def __copy__(self) -> WingSection:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> WingSection:
        return self.clone()


@dataclass(slots=True)
class LiftingSurface:
    """A complete lifting surface (wing, horizontal tail, canard, etc.).

    Parameters
    ----------
    name : str
        Human-readable label.
    semi_span : float
        Semi-span length [m], measured in the y-z plane along the dihedral
        line. A station at span fraction ``eta`` is at
        ``y = eta*semi_span*cos(dihedral)``, ``z = eta*semi_span*sin(dihedral)``.
        If any section sets ``z_le``, ``semi_span`` is the projected span
        (y extent) instead and ``z`` comes from the sections.
    sections : list[WingSection]
        Ordered list of cross-section definitions from root (``y_frac=0``)
        to tip (``y_frac=1``).  At least two sections are required.
    is_symmetric : bool
        If *True*, the surface is mirrored about the aircraft's xz-plane.
    position : np.ndarray
        ``[x, y, z]`` offset of the surface root LE from the origin of the
        geometry axes [m].
    incidence : float
        Surface incidence angle relative to the aircraft reference [rad].
        It is added to the section twist (rotation about the quarter chord).
    sweep_le : float
        Leading-edge sweep angle [rad], measured in the plane of the surface.
        Used to compute ``x_le`` at each section when the section's own
        ``x_le`` is *None*.
    dihedral : float
        Dihedral angle [rad]. ``pi/2`` with ``is_symmetric=False`` gives a
        vertical fin of height ``semi_span``.
    mirror_y : bool
        If *True*, the surface is the mirror image (y -> -y) of the geometry
        that the other fields define, including the position, camber and
        twist. Use :meth:`mirrored` to make the left copy of a surface that is
        not on the plane y = 0 (for example twin fins); ``is_symmetric`` must
        then be *False*.
    controls : list[ControlSurface]
        Hinged control surfaces defined on this surface.
    node_displacements : NodeDisplacements | None
        Displacements [m] of the leading- and trailing-edge nodes of the defining half.
    """

    name: str = "Wing"
    semi_span: float = 5.0
    sections: list[WingSection] = field(default_factory=lambda: [
        WingSection(y_frac=0.0, chord=2.0),
        WingSection(y_frac=1.0, chord=2.0),
    ])
    is_symmetric: bool = True
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    incidence: float = 0.0
    sweep_le: float = 0.0
    dihedral: float = 0.0
    n_panels: int | None = None
    spacing: str | None = None
    mirror_y: bool = False
    controls: list[ControlSurface] = field(default_factory=list)
    node_displacements: NodeDisplacements | None = None

    def clone(self) -> LiftingSurface:
        """Fast explicit clone."""
        return LiftingSurface(
            name=self.name,
            semi_span=self.semi_span,
            sections=[sec.clone() for sec in self.sections],
            is_symmetric=self.is_symmetric,
            position=self.position.copy(),
            incidence=self.incidence,
            sweep_le=self.sweep_le,
            dihedral=self.dihedral,
            n_panels=self.n_panels,
            spacing=self.spacing,
            mirror_y=self.mirror_y,
            controls=[c.clone() for c in self.controls],
            node_displacements=self.node_displacements.clone() if self.node_displacements is not None else None,
        )

    def mirrored(self, name: str | None = None) -> LiftingSurface:
        """Mirror image of this surface across the plane y = 0.

        For a surface that is not on the plane y = 0, define the right copy
        with ``is_symmetric=False`` and add ``surf.mirrored()`` to the
        aircraft as the left copy.
        """
        if self.is_symmetric:
            raise ValueError(f"[{self.name}] mirrored() needs is_symmetric=False; a symmetric surface already has both halves.")
        out = self.clone()
        out.mirror_y = not self.mirror_y
        out.name = name if name is not None else f"{self.name} (mirror)"
        return out

    def __copy__(self) -> LiftingSurface:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> LiftingSurface:
        return self.clone()


@dataclass(slots=True)
class Aircraft:
    """Collection of lifting surfaces.

    Parameters
    ----------
    name : str
        Aircraft identifier.
    surfaces : list[LiftingSurface]
        All lifting surfaces to include in the analysis.
    S_ref : float | None
        Reference wing area [m²].  If *None*, computed from the main surface
        (see :meth:`main_surface_index`).
    b_ref : float | None
        Reference span [m].  If *None*, computed from the main surface.
    c_ref : float | None
        Reference chord (mean aerodynamic chord) [m].  If *None*, computed
        from the main surface.
    ref_point : (3,) array or None
        Moment reference point [m] in geometry axes (for example the centre
        of gravity). If *None*, the origin. In ground effect, the height
        ``h`` is the height of this point (``height_ref='ref'``).
    """

    name: str = "Aircraft"
    surfaces: list[LiftingSurface] = field(default_factory=lambda: [
        LiftingSurface(),
    ])
    S_ref: float | None = None
    b_ref: float | None = None
    c_ref: float | None = None
    ref_point: np.ndarray | None = None
    # Names of the reference values that were filled automatically. A value
    # that the user assigns is removed from this set (see __setattr__).
    _auto_ref: set = field(default_factory=set, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name in _REF_FIELDS:
            # An assignment by the user: keep this value from now on.
            try:
                self._auto_ref.discard(name)
            except AttributeError:
                pass  # during __init__, before _auto_ref exists

    def clone(self) -> Aircraft:
        """Fast explicit clone."""
        out = Aircraft(
            name=self.name,
            surfaces=[surf.clone() for surf in self.surfaces],
            S_ref=self.S_ref,
            b_ref=self.b_ref,
            c_ref=self.c_ref,
            ref_point=None if self.ref_point is None else np.asarray(self.ref_point, dtype=float).copy(),
        )
        out._auto_ref = set(self._auto_ref)
        return out

    def __copy__(self) -> Aircraft:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> Aircraft:
        return self.clone()

    def control_names(self) -> list[str]:
        """Return the sorted, unique control surface names across all surfaces."""
        names = {c.name for surf in self.surfaces for c in getattr(surf, "controls", [])}
        return sorted(names)

    def set_deflection(self, name: str, deflection: float) -> int:
        """Set the deflection [rad] of every control with this name on all surfaces.

        Parameters
        ----------
        name : str
            Control surface name.
        deflection : float
            Deflection angle [rad].

        Returns
        -------
        int
            Number of control surfaces updated.

        Raises
        ------
        ValueError
            If no control surface with *name* exists on any surface.
        """
        known = self.control_names()
        if name not in known:
            raise ValueError(
                f"Unknown control surface {name!r}; known control names: {known}."
            )
        count = 0
        for surf in self.surfaces:
            for c in getattr(surf, "controls", []):
                if c.name == name:
                    c.deflection = float(deflection)
                    count += 1
        return count

    def moment_reference(self) -> np.ndarray:
        """Moment reference point [m] in geometry axes (the origin if ``ref_point`` is None)."""
        return np.zeros(3) if self.ref_point is None else np.asarray(self.ref_point, dtype=float).reshape(3)

    def main_surface_index(self) -> int:
        """Index of the main surface: the surface with the largest projected planform area.

        The area is the planform area of the flat (untwisted) surface
        projected on the x-y plane, after mirroring: a symmetric surface
        counts both halves, and a half surface counts together with its
        mirror copies (see :meth:`LiftingSurface.mirrored`). A tie (relative
        difference below 1e-9) goes to the first surface in the list, so the
        result does not depend on round-off. If no surface has a projected
        area (only vertical fins), the first surface is used.

        Reason: the main surface gives the reference values (*S_ref*,
        *b_ref*, *c_ref*). The order of the surfaces or a dihedral near
        45 deg (a V-tail listed first) must not make a small surface the
        reference.
        """
        from ventorum.geometry.lattice import surface_reference_line

        def projected_area(surf: LiftingSurface) -> float:
            secs = sorted(surf.sections, key=lambda s: s.y_frac)
            eta = np.array([s.y_frac for s in secs], dtype=float)
            chords = np.array([s.chord for s in secs], dtype=float)
            _, y, _ = surface_reference_line(surf, eta)
            area = float(np.sum(0.5 * (chords[:-1] + chords[1:]) * np.abs(np.diff(y))))
            return 2.0 * area if surf.is_symmetric else area

        own = [projected_area(s) for s in self.surfaces]
        total = [own[i] + sum(own[j] for j, t in enumerate(self.surfaces) if j != i and _is_mirror_pair(s, t))
                 for i, s in enumerate(self.surfaces)]
        best = 0
        for i in range(1, len(total)):
            if total[i] > total[best] * (1.0 + 1e-9) and total[i] > 0.0:
                best = i
        return best

    def reference_surfaces(self) -> list[LiftingSurface]:
        """Return the main surface and its mirror copies.

        See :meth:`main_surface_index` and :meth:`LiftingSurface.mirrored`.
        """
        k = self.main_surface_index()
        s0 = self.surfaces[k]
        out = [s0]
        for i, s in enumerate(self.surfaces):
            if i != k and _is_mirror_pair(s0, s):
                out.append(s)
        return out

    def root_point(self, which: str = "qc") -> np.ndarray:
        """Root quarter-chord (``"qc"``) or trailing-edge (``"te"``) point of the main surface [m].

        The point is computed on the undeformed geometry.
        The root is the section at ``y_frac = 0``. On a mirror copy
        (``mirror_y=True``) it is at the mirrored y.
        """
        from ventorum.geometry.lattice import surface_edge_geometry

        if which not in ("qc", "te"):
            raise ValueError(f"Unknown root point {which!r}; use 'qc' or 'te'.")
        surf = self.surfaces[self.main_surface_index()]
        geo = surface_edge_geometry(surf, np.array([0.0]))
        le, te = geo["le"][0], geo["te"][0]
        p = le + 0.25 * (te - le) if which == "qc" else te
        if surf.mirror_y:
            p = p * np.array([1.0, -1.0, 1.0])
        return p

    def compute_reference_values(self, auto: dict[str, float] | None = None) -> None:
        """Fill *S_ref*, *b_ref*, *c_ref* that the user did not set.

        The values come from the main surface (see :meth:`main_surface_index`)
        and its mirror copies: projected span, projected planform area and
        mean aerodynamic chord ``(1/S) * integral(c^2 dy)``. A surface with no
        projected span (a vertical fin) uses the length along the surface.
        Values that were filled automatically are computed again at each
        call, so they follow later changes of the geometry. A value that the
        user assigns (also after an automatic value) is kept. To go back to
        the automatic value, assign None.

        *auto* holds automatic values computed before for the same surfaces
        (``b_ref``, ``S_ref``, ``c_ref``); the solvers pass it from their
        geometry cache. The same assignment rule applies.
        """
        if auto is None:
            auto = self._auto_reference_values()
        for key, val in auto.items():
            if getattr(self, key) is None or key in self._auto_ref:
                object.__setattr__(self, key, val)
                self._auto_ref.add(key)

    def _auto_reference_values(self) -> dict[str, float]:
        """Return the automatic ``b_ref``, ``S_ref`` and ``c_ref`` of the surfaces (see compute_reference_values)."""
        from ventorum.geometry.lattice import surface_reference_line

        span = area = int_c2 = 0.0
        for surf in self.reference_surfaces():
            secs = sorted(surf.sections, key=lambda s: s.y_frac)
            eta = np.array([s.y_frac for s in secs], dtype=float)
            chords = np.array([s.chord for s in secs], dtype=float)
            _, y, _ = surface_reference_line(surf, eta)
            dy = np.abs(np.diff(y))
            if float(np.sum(dy)) <= 1e-9 * max(surf.semi_span, 1e-12):
                dy = np.diff(eta) * surf.semi_span
            factor = 2.0 if surf.is_symmetric else 1.0
            span += factor * float(np.sum(dy))
            area += factor * float(np.sum(0.5 * (chords[:-1] + chords[1:]) * dy))
            int_c2 += factor * float(np.sum(dy / 3.0 * (chords[:-1] ** 2 + chords[:-1] * chords[1:] + chords[1:] ** 2)))
        return {
            "b_ref": span,
            "S_ref": area,
            "c_ref": (int_c2 / area) if area > 0 else 1.0,
        }


# Reference values of Aircraft that can be filled automatically.
_REF_FIELDS = frozenset({"S_ref", "b_ref", "c_ref"})


def _same_optional(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.isclose(a, b))


def _same_airfoil(a: AirfoilType, b: AirfoilType) -> bool:
    """Return True if the two airfoil models give the same section data."""
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, LinearAirfoil):
        return bool(np.allclose([a.a0, a.alpha_L0, a.Cd0, a.Cm0], [b.a0, b.alpha_L0, b.Cd0, b.Cm0]))
    pairs = [(a.alpha, b.alpha), (a.Cl_data, b.Cl_data), (a.Cd_data, b.Cd_data)]
    if (a.Cm_data is None) != (b.Cm_data is None):
        return False
    if a.Cm_data is not None:
        pairs.append((a.Cm_data, b.Cm_data))
    return all(np.shape(x) == np.shape(y) and np.allclose(x, y) for x, y in pairs)


def _is_mirror_pair(a: LiftingSurface, b: LiftingSurface) -> bool:
    """Return True if *b* is the mirror copy of *a*.

    A mirror copy has the same geometry and the opposite mirror flag.
    """
    if a.is_symmetric or b.is_symmetric or a.mirror_y == b.mirror_y:
        return False
    if len(a.sections) != len(b.sections):
        return False
    same = (
        np.isclose(a.semi_span, b.semi_span)
        and np.allclose(np.asarray(a.position, dtype=float), np.asarray(b.position, dtype=float))
        and np.isclose(a.dihedral, b.dihedral)
        and np.isclose(a.sweep_le, b.sweep_le)
        and np.isclose(a.incidence, b.incidence)
    )
    if not same:
        return False
    secs_a = sorted(a.sections, key=lambda s: s.y_frac)
    secs_b = sorted(b.sections, key=lambda s: s.y_frac)
    return all(
        np.isclose(sa.y_frac, sb.y_frac) and np.isclose(sa.chord, sb.chord) and np.isclose(sa.twist, sb.twist)
        and _same_optional(sa.x_le, sb.x_le) and _same_optional(sa.z_le, sb.z_le)
        and _same_airfoil(sa.airfoil, sb.airfoil)
        for sa, sb in zip(secs_a, secs_b)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Flight conditions & solver settings
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class FlightCondition:
    """Free-stream conditions for the analysis.

    Parameters
    ----------
    V_inf : float
        Free-stream velocity [m/s].
    alpha : float
        Angle of attack [rad].
    beta : float
        Sideslip angle [rad]. All the lattice solvers (``"vlm"``,
        ``"linear"``, ``"nonlinear"``) accept it; the ``"fourier"`` solver
        refuses a sideslip that is not zero.
    rho : float
        Air density [kg/m³].
    h : float or None
        Height [m] of ``Aircraft.ref_point`` (default: the origin of the
        geometry axes) above the ground plane. *None* means free air.
    phi : float
        Bank (roll) angle [rad].
    p, q, r : float
        Body angular rates [rad/s] about the moment reference point, in the
        flight-dynamics body axes (x forward, y right, z down): ``p > 0``
        right wing down, ``q > 0`` nose up, ``r > 0`` nose right (the signs
        of Cl, Cm and Cn). Default 0. The lattice solvers (``"vlm"``,
        ``"linear"``, ``"nonlinear"``) solve the quasi-steady rotating flow
        (see :mod:`ventorum.aero.system`); the ``"fourier"`` solver refuses
        a rate that is not zero.
    mach : float or None
        Mach number [-] for the Prandtl-Glauert compressibility correction
        in the vortex-lattice solver. *None* (default) or 0 means no
        correction. The scope of Ventorum is incompressible flow (Mach below
        0.3). Between 0.3 and 0.7 the correction is an estimate with the
        trust rating UNRELIABLE. The validity check and the solvers refuse
        ``mach >= 0.7`` with :class:`~ventorum.core.errors.ValidityError`.
        The constructor accepts 0 <= mach < 1. A stated Mach number that
        differs from ``V_inf`` divided by the ISA speed of sound at the
        density ``rho`` by more than 0.05 gives a warning and a trust note.
    """

    V_inf: float = 50.0
    alpha: float = np.radians(5.0)
    beta: float = 0.0
    rho: float = RHO_SL
    h: float | None = None  # Height of Aircraft.ref_point (default: the origin) above the ground plane [m]
    phi: float = 0.0  # Bank / roll angle [rad] (default 0.0)
    p: float = 0.0  # Roll rate [rad/s], body axes, > 0 right wing down
    q: float = 0.0  # Pitch rate [rad/s], body axes, > 0 nose up
    r: float = 0.0  # Yaw rate [rad/s], body axes, > 0 nose right
    mach: float | None = None  # Mach number [-] (0 <= mach < 1; solvers refuse mach >= 0.7)

    def __post_init__(self) -> None:
        if self.mach is not None:
            if not (0.0 <= self.mach < 1.0):
                raise ValueError(
                    f"mach must satisfy 0 <= mach < 1 (subsonic), got {self.mach}."
                )

    @property
    def phi_deg(self) -> float:
        """Roll / bank angle in degrees."""
        return float(np.degrees(self.phi))

    @property
    def beta_deg(self) -> float:
        """Sideslip angle in degrees."""
        return float(np.degrees(self.beta))

    @property
    def alpha_deg(self) -> float:
        """Angle of attack in degrees."""
        return float(np.degrees(self.alpha))

    def clone(self) -> FlightCondition:
        """Fast explicit clone."""
        return FlightCondition(
            V_inf=self.V_inf,
            alpha=self.alpha,
            beta=self.beta,
            rho=self.rho,
            h=self.h,
            phi=self.phi,
            p=self.p,
            q=self.q,
            r=self.r,
            mach=self.mach,
        )

    def __copy__(self) -> FlightCondition:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> FlightCondition:
        return self.clone()


@dataclass(slots=True)
class SolverSettings:
    """Configuration knobs for the LLT solvers.

    Parameters
    ----------
    solver_type : str
        ``"auto"`` (default): always ``"vlm"``, also when a section has a
        tabulated polar (select ``"nonlinear"`` explicitly to use the polar
        past its linear range).
        ``"vlm"``: vortex-lattice method (lifting surface).
        ``"linear"``: linear lifting line (Phillips & Snyder).
        ``"nonlinear"``: nonlinear lifting line with section polars.
        ``"fourier"``: Lanchester–Prandtl lifting-line theory, Glauert's Fourier series (unswept planar wing).
        Old names (``"horseshoe"``, ``"lattice"``, ``"llt"``, ``"linear_llt"``)
        still work for now with a ``FutureWarning`` that names the new name.
    n_panels : int
        Number of spanwise panels **per semi-span**.
    n_chord : int or None
        Chordwise panels per strip for the vortex-lattice method. ``None``
        selects it automatically: 4 out of ground effect, more in ground
        effect so that a chordwise panel is not longer than the gap height.
    chord_spacing : str
        ``"uniform"`` (default) or ``"cosine"`` chordwise distribution.
    wake_alignment : str
        ``"freestream"`` (default): the wake leaves the trailing edge along the
        free stream. ``"body"``: along the body x-axis, as in AVL; the wake and
        the system matrix then do not change with alpha and beta (not allowed
        in ground effect).
    spacing : str
        ``"auto"`` (default), ``"cosine"`` (clustered at root and tip),
        ``"half-cosine"`` (tip-clustered; cosine over the whole span),
        ``"root"`` or ``"uniform"``. ``"auto"`` gives ``"half-cosine"`` for
        the lifting-line solvers, and for the VLM ``"half-cosine"`` on planar
        wings and ``"cosine"`` with sweep or dihedral (taken from the actual
        leading-edge line, so explicit ``x_le``/``z_le`` count too).
    max_iterations : int
        Maximum Newton iterations of the nonlinear lifting line.
    tolerance : float
        Convergence threshold of the nonlinear lifting line: the largest
        error in section lift coefficient, ``max_i |R_i| / (V^2 dA_i)``.
    proportional_panels : bool
        If True, scales panel counts proportionally across surfaces based on span.
    min_panels : int
        Minimum panels per semi-span when proportional scaling is enabled.
    panel_kutta : str
        Kutta condition of the 3D panel solver (``"panel3d"``):
        ``"morino"`` (default, linear) or ``"pressure"`` (iterative, zero
        trailing-edge pressure jump). The other solvers ignore it.
    panel_backend : str
        Backend of the 3D panel solver: ``"auto"`` (default: the device
        and precision selection of :mod:`ventorum.gpu`, the Numba kernels
        on the CPU, or the numpy kernels when Numba is not installed),
        ``"numpy"``, ``"numba"`` or ``"gpu"``. The other
        solvers ignore it.
    panel_wake_length : float or None
        Wake length of the 3D panel solver [m]. None (default) gives 50
        times the largest streamwise chord. The other solvers ignore it.
    panel_chord_spacing : str
        Chordwise panel distribution of the 3D panel solver: ``"cosine"``
        (default) or ``"uniform"``. The other solvers use
        ``chord_spacing``.
    panel_estimate_mesh_error : bool
        If True, the 3D panel solver runs a mesh convergence study for each
        condition and adds the error estimate of the given mesh to the
        trust notes. Default False (the study costs much more than one
        solve). The other solvers ignore it.
    panel_scale_root_thickness : bool
        If True (default), the 3D panel mesh multiplies the thickness of
        the root section of a symmetric surface (in the plane y = 0) by
        1 / cos(G), with G the dihedral of the first panel row [rad], so
        that the thickness normal to the local surface equals the section
        thickness. False keeps the thickness along the root section normal.
        The other solvers ignore it.
    panel_cache : bool
        If True (default), the 3D panel solver keeps the mesh, the solver
        and its factorized systems of the last solves
        (:mod:`ventorum.panel3d.cache`), so that a repeated solve of the same
        geometry with the same mesh settings skips the setup and, for a
        known configuration, the factorization. The results are the same
        bits as without the cache. False builds everything again for each
        solve. ``ventorum.panel3d.clear_cache()`` frees the cache. The other
        solvers ignore it.
    panel_ground_solver : str
        Ground-effect linear solve of the 3D panel solver: ``"auto"``
        (default), ``"direct"`` or ``"krylov"``. Auto reuses an anchor
        LU with GMRES for CPU ground configurations with Morino Kutta.
        Free air, GPU backends and pressure Kutta use the direct solve.
    """

    solver_type: Literal["auto", "vlm", "linear", "nonlinear", "fourier"] | str = "auto"
    n_panels: int = 80
    spacing: Literal["auto", "cosine", "half-cosine", "root", "uniform"] | str = "auto"
    max_iterations: int = 200
    tolerance: float = 1.0e-6
    proportional_panels: bool = False
    min_panels: int = 8
    use_symmetry: bool = True  # Toggle symmetry plane on Y=0 (default True)
    n_chord: int | None = None
    chord_spacing: str = "uniform"
    wake_alignment: str = "freestream"
    panel_kutta: str = "morino"
    panel_backend: str = "auto"
    panel_wake_length: float | None = None
    panel_chord_spacing: str = "cosine"
    panel_estimate_mesh_error: bool = False
    panel_scale_root_thickness: bool = True
    panel_cache: bool = True
    panel_ground_solver: str = "auto"

    def clone(self) -> SolverSettings:
        """Fast explicit clone.

        A subclass keeps its type and its own fields (the copy uses
        :func:`dataclasses.replace`).
        """
        if type(self) is not SolverSettings:
            return dataclasses.replace(self)
        return SolverSettings(
            solver_type=self.solver_type,
            n_panels=self.n_panels,
            n_chord=self.n_chord,
            chord_spacing=self.chord_spacing,
            wake_alignment=self.wake_alignment,
            spacing=self.spacing,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            proportional_panels=self.proportional_panels,
            min_panels=self.min_panels,
            use_symmetry=self.use_symmetry,
            panel_kutta=self.panel_kutta,
            panel_backend=self.panel_backend,
            panel_wake_length=self.panel_wake_length,
            panel_chord_spacing=self.panel_chord_spacing,
            panel_estimate_mesh_error=self.panel_estimate_mesh_error,
            panel_scale_root_thickness=self.panel_scale_root_thickness,
            panel_cache=self.panel_cache,
            panel_ground_solver=self.panel_ground_solver,
        )

    def __copy__(self) -> SolverSettings:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> SolverSettings:
        return self.clone()


# ═══════════════════════════════════════════════════════════════════════════════
# Discretised geometry (internal, produced by the geometry processor)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class DiscretizedSurface:
    """Panel-level representation of a single lifting surface.

    Created by :func:`ventorum.geometry.processing.discretize_surface`.

    Convention: arrays indexed ``[i]`` where ``i = 0 … N-1`` for *N* panels.
    """

    # Bound-vortex node positions at the 1/4-chord line  [N+1, 3]
    nodes_qc: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    # Control-point positions at the 3/4-chord              [N, 3]
    control_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    # Outward surface normals at the control points          [N, 3]
    normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    # Panel-centre spanwise coordinate                       [N]
    y_panels: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Panel widths                                           [N]
    dy_panels: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Interpolated chord at each panel centre                [N]
    chords: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Interpolated twist at each panel centre                [N]
    twists: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Airfoil objects at each panel (nearest-section rule)
    airfoils: list[AirfoilType] = field(default_factory=list)
    # Metadata
    surface_name: str = ""
    surface_index: int = 0
    # Precomputed cached properties for force integration & polars
    panel_centers_qc: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    has_profile_drag: bool = False
    airfoil_groups: list[tuple[Any, np.ndarray]] = field(default_factory=list)
    is_half_mesh: bool = False  # True if this surface only contains the right semi-span (y >= 0)
    # Leading- and trailing-edge positions at the panel edges [N+1, 3]
    nodes_le: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    nodes_te: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))


# ═══════════════════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SpanwiseResult:
    """Spanwise distributions for a **single** lifting surface.

    Attributes
    ----------
    y : numpy.ndarray
        Spanwise position of each strip [m].
    gamma : numpy.ndarray
        Circulation of each strip [m^2/s].
    Cl, Cd_i, Cd_profile : numpy.ndarray
        Section lift, induced-drag and profile-drag coefficients [-].
    alpha_eff, alpha_i : numpy.ndarray
        Effective and induced angle of attack of each strip [rad].
    local_lift : numpy.ndarray
        Lift per unit span [N/m].
    chord : numpy.ndarray or None
        Strip chord [m].
    Cm_section : numpy.ndarray or None
        Section pitching moment about the quarter chord [-].
    """

    y: np.ndarray = field(default_factory=lambda: np.empty(0))
    gamma: np.ndarray = field(default_factory=lambda: np.empty(0))
    Cl: np.ndarray = field(default_factory=lambda: np.empty(0))
    Cd_i: np.ndarray = field(default_factory=lambda: np.empty(0))
    Cd_profile: np.ndarray | None = None
    alpha_eff: np.ndarray = field(default_factory=lambda: np.empty(0))
    alpha_i: np.ndarray = field(default_factory=lambda: np.empty(0))
    local_lift: np.ndarray = field(default_factory=lambda: np.empty(0))
    surface_name: str = ""
    # Strip chord [m] and section pitching moment about the quarter chord.
    chord: np.ndarray | None = None
    Cm_section: np.ndarray | None = None

    def clone(self) -> SpanwiseResult:
        """Fast explicit clone."""
        return SpanwiseResult(
            y=self.y.copy(),
            gamma=self.gamma.copy(),
            Cl=self.Cl.copy(),
            Cd_i=self.Cd_i.copy(),
            Cd_profile=self.Cd_profile.copy() if self.Cd_profile is not None else None,
            alpha_eff=self.alpha_eff.copy(),
            alpha_i=self.alpha_i.copy(),
            local_lift=self.local_lift.copy(),
            surface_name=self.surface_name,
            chord=self.chord.copy() if self.chord is not None else None,
            Cm_section=self.Cm_section.copy() if self.Cm_section is not None else None,
        )

    def __copy__(self) -> SpanwiseResult:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> SpanwiseResult:
        return self.clone()


@dataclass(slots=True)
class TrustScore:
    """Quantitative validity score and physical uncertainty assessment.

    Parameters
    ----------
    score : float
        Composite trust score in [0.0, 1.0]. 1.0 indicates perfect adherence
        to the assumptions of Lanchester–Prandtl lifting-line theory; values
        < 0.4 indicate that the flow physics severely violate them.
    rating : Literal["HIGH", "MODERATE", "LOW", "UNRELIABLE"]
        Qualitative categorization of analysis validity.
    uncertainty_CL : float
        Estimated absolute uncertainty margin (+/- Delta CL).
    uncertainty_CDi : float
        Estimated absolute uncertainty margin (+/- Delta CDi).
    uncertainty_LD : float
        Estimated absolute uncertainty margin (+/- Delta L/D).
    factors : dict[str, float]
        Individual penalty factor contributions (0.0 = no penalty).
    warnings : list[str]
        Specific physics alerts explaining what aerodynamic limits were approached.
    recommendations : list[str]
        Engineering advice on alternative higher-order methods (e.g. VLM, CFD).
    """

    score: float = 1.0
    rating: Literal["HIGH", "MODERATE", "LOW", "UNRELIABLE"] = "HIGH"
    uncertainty_CL: float = 0.02
    uncertainty_CDi: float = 0.001
    uncertainty_LD: float = 0.5
    factors: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def clone(self) -> TrustScore:
        """Fast explicit clone."""
        return TrustScore(
            score=self.score,
            rating=self.rating,
            uncertainty_CL=self.uncertainty_CL,
            uncertainty_CDi=self.uncertainty_CDi,
            uncertainty_LD=self.uncertainty_LD,
            factors=dict(self.factors),
            warnings=list(self.warnings),
            recommendations=list(self.recommendations),
        )

    def __copy__(self) -> TrustScore:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> TrustScore:
        return self.clone()

    def to_dict(self) -> dict[str, Any]:
        """Convert the trust assessment to a JSON-serializable dictionary."""
        return {
            "score": round(float(self.score), 4),
            "rating": self.rating,
            "uncertainty_CL": round(float(self.uncertainty_CL), 4),
            "uncertainty_CDi": round(float(self.uncertainty_CDi), 6),
            "uncertainty_LD": round(float(self.uncertainty_LD), 2),
            "factors": {k: round(float(v), 4) for k, v in self.factors.items()},
            "warnings": list(self.warnings),
            "recommendations": list(self.recommendations),
        }

    def summary_str(self) -> str:
        """Concise one-line summary string for user interfaces and agent reasoning."""
        warn_str = f" | Warnings: {len(self.warnings)}" if self.warnings else " | No warnings"
        return (
            f"Trust: {self.score:.2f} ({self.rating}) | "
            f"heuristic band (not calibrated) CL +/- {self.uncertainty_CL:.3f}, CDi +/- {self.uncertainty_CDi:.5f}"
            f"{warn_str}"
        )


@dataclass(slots=True)
class IntegratedResult:
    """Wing-level (or aircraft-level) integrated coefficients.

    Sign conventions (the same as AVL):

    * ``CL`` is normal to the free stream in the body x-z plane, ``CD`` is
      along the free stream, ``CY`` is positive to the right wing.
    * ``CDi`` is the induced drag from the Trefftz plane. Use it as the
      reference value of the induced drag.
    * ``CDi_nearfield`` is the induced drag from the forces on the surface
      (a diagnostic). It is not reliable on swept wings; do not use it in
      place of ``CDi``.
    * ``CY_farfield`` is the side force of the trailing vortex sheet in the
      Trefftz plane, in wind axes (positive to the right). ``Cn_trefftz`` is
      the yawing moment of the Trefftz-plane induced-drag distribution about
      the moment reference point, in stability axes (``Cn > 0`` nose right).
      Both are diagnostics that show where the near field is weak; the
      near-field ``CY`` and ``Cn`` stay the result values. They are None
      for the Fourier solver.
    * ``Cl``, ``Cm``, ``Cn`` are body-axis moments about the reference point
      in the standard aircraft convention: ``Cl > 0`` right wing down,
      ``Cm > 0`` nose up, ``Cn > 0`` nose right. Static stability needs
      ``Cm_alpha < 0``, ``Cl_beta < 0`` and ``Cn_beta > 0``.
    """

    CL: float = 0.0
    CDi: float = 0.0
    CDp: float | None = None
    CD_total: float | None = None
    e: float = 0.0             # span efficiency factor CL^2 / (pi AR CDi)
    CL_alpha: float | None = None
    AR: float = 0.0
    Cl: float = 0.0            # Rolling moment coefficient (positive right wing down)
    Cm: float = 0.0            # Pitching moment coefficient (positive nose up)
    Cn: float = 0.0            # Yawing moment coefficient (positive nose right)
    trust: TrustScore | None = None
    CY: float = 0.0            # Side force coefficient (positive to the right)
    CDi_nearfield: float | None = None  # Induced drag from surface forces (diagnostic)
    CY_farfield: float | None = None    # Side force from the Trefftz plane, wind axes (diagnostic)
    Cn_trefftz: float | None = None     # Yaw moment of the Trefftz induced drag, stability axes (diagnostic)

    def clone(self) -> IntegratedResult:
        """Fast explicit clone."""
        return IntegratedResult(
            CL=self.CL,
            CDi=self.CDi,
            CDp=self.CDp,
            CD_total=self.CD_total,
            e=self.e,
            CL_alpha=self.CL_alpha,
            AR=self.AR,
            Cl=self.Cl,
            Cm=self.Cm,
            Cn=self.Cn,
            trust=self.trust.clone() if self.trust is not None else None,
            CY=self.CY,
            CDi_nearfield=self.CDi_nearfield,
            CY_farfield=self.CY_farfield,
            Cn_trefftz=self.Cn_trefftz,
        )

    def __copy__(self) -> IntegratedResult:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> IntegratedResult:
        return self.clone()

    def to_dict(self) -> dict[str, Any]:
        """Return the integrated coefficients as a dictionary of plain Python values.

        Returns
        -------
        dict
            The keys ``CL``, ``CDi``, ``CDp``, ``CD_total``, ``e``,
            ``CL_alpha`` [1/rad], ``AR``, ``Cl``, ``Cm``, ``Cn``, ``CY``,
            ``CDi_nearfield``, ``CY_farfield`` and ``Cn_trefftz``
            (dimensionless coefficients as float, or None when the solver
            does not give the value), and ``trust`` (the dictionary of the
            trust score, or None).
        """
        return {
            "CL": float(self.CL),
            "CDi": float(self.CDi),
            "CDp": None if self.CDp is None else float(self.CDp),
            "CD_total": None if self.CD_total is None else float(self.CD_total),
            "e": float(self.e),
            "CL_alpha": None if self.CL_alpha is None else float(self.CL_alpha),
            "AR": float(self.AR),
            "Cl": float(self.Cl),
            "Cm": float(self.Cm),
            "Cn": float(self.Cn),
            "CY": float(self.CY),
            "CDi_nearfield": None if self.CDi_nearfield is None else float(self.CDi_nearfield),
            "CY_farfield": None if self.CY_farfield is None else float(self.CY_farfield),
            "Cn_trefftz": None if self.Cn_trefftz is None else float(self.Cn_trefftz),
            "trust": self.trust.to_dict() if self.trust is not None else None,
        }


@dataclass(slots=True)
class SolverResult:
    """Top-level container returned by every solver.

    Carries both per-surface spanwise data and aircraft-level totals, plus
    solver metadata (iteration count, residual history, etc.).

    The induced drag ``totals.CDi`` comes from the Trefftz plane and is the
    reference value. ``totals.CDi_nearfield`` (from the surface forces) is a
    diagnostic only; it is not reliable on swept wings.

    Attributes
    ----------
    b_ref : float or None
        Reference span of the solve [m]. :meth:`moments` needs it for the
        stability and wind axes.
    c_ref : float or None
        Reference chord of the solve [m]. :meth:`moments` needs it for the
        stability and wind axes.
    """

    # Per-surface spanwise distributions (one entry per LiftingSurface)
    spanwise: list[SpanwiseResult] = field(default_factory=list)
    # Aircraft-level integrated coefficients
    totals: IntegratedResult = field(default_factory=IntegratedResult)
    # Solver metadata
    solver_type: str = ""
    converged: bool = True
    iterations: int = 0
    residual_history: list[float] = field(default_factory=list)
    # Fourier coefficients (only populated by the Fourier solver)
    fourier_coefficients: np.ndarray | None = None
    # Operating flight condition (optional)
    condition: FlightCondition | None = None
    execution_time: float = 0.0
    symmetry_used: bool = False  # Records if Y=0 symmetry plane was utilized
    # Solver internals kept for post-processing (lattice, loads, ground plane).
    details: dict[str, Any] = field(default_factory=dict)
    # Reference span and chord [m] of the solve.
    b_ref: float | None = None
    c_ref: float | None = None

    def clone(self) -> SolverResult:
        """Fast explicit clone."""
        return SolverResult(
            spanwise=[sw.clone() for sw in self.spanwise],
            totals=self.totals.clone(),
            solver_type=self.solver_type,
            converged=self.converged,
            iterations=self.iterations,
            residual_history=list(self.residual_history),
            fourier_coefficients=self.fourier_coefficients.copy() if self.fourier_coefficients is not None else None,
            condition=self.condition.clone() if self.condition is not None else None,
            execution_time=self.execution_time,
            symmetry_used=self.symmetry_used,
            details=dict(self.details),
            b_ref=self.b_ref,
            c_ref=self.c_ref,
        )

    def __copy__(self) -> SolverResult:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> SolverResult:
        return self.clone()

    def moments(self, axes: str = "body") -> dict[str, float] | dict[str, dict[str, float]]:
        """Moment coefficients in the selected axis system.

        Parameters
        ----------
        axes : str, optional
            ``"body"`` (default), ``"stability"`` or ``"wind"`` for one
            set ``{"Cl", "Cm", "Cn"}``; ``"all"`` for a dict with the
            three sets. The stored values are body-axis moments; the
            rotation uses the alpha and beta [rad] of ``condition``
            (0 when there is no condition). CL, CD and CY are relative
            to the free stream in every set, so they are not changed.

        Returns
        -------
        dict
            One set, or the three sets for ``axes="all"``.

        Raises
        ------
        ValueError
            If *axes* is not a known value, or if *b_ref* or *c_ref*
            is None when expressing moments in non-body axes.
        """
        if axes not in ("body", "stability", "wind", "all"):
            raise ValueError(f"'axes' must be one of ['body', 'stability', 'wind', 'all'], got {axes!r}.")
        tot = self.totals
        if axes == "body":
            return {"Cl": float(tot.Cl), "Cm": float(tot.Cm), "Cn": float(tot.Cn)}
        if self.b_ref is None:
            raise ValueError(f"'b_ref' is required to express moments in {axes!r} axes.")
        if self.c_ref is None:
            raise ValueError(f"'c_ref' is required to express moments in {axes!r} axes.")
        from ventorum.core.axes import transform_moments

        alpha = float(self.condition.alpha) if self.condition is not None else 0.0
        beta = float(self.condition.beta) if self.condition is not None else 0.0
        if axes == "all":
            return {
                name: transform_moments(tot.Cl, tot.Cm, tot.Cn, alpha, beta, name, b_ref=self.b_ref, c_ref=self.c_ref)
                for name in ("body", "stability", "wind")
            }
        return transform_moments(tot.Cl, tot.Cm, tot.Cn, alpha, beta, axes, b_ref=self.b_ref, c_ref=self.c_ref)

    def to_dict(self, detail_level: str = "standard") -> dict[str, Any]:
        """Convert solver result to a dictionary.

        Parameters
        ----------
        detail_level : str, optional
            ``"summary"``: the solver data, the condition and the totals.
            ``"standard"`` (default): also the spanwise arrays. ``"full"``:
            also the dispatch record (``details["dispatch"]``).

        Returns
        -------
        dict
            Dictionary of plain Python values: ``solver_type``,
            ``converged``, ``iterations``, ``execution_time`` [s],
            ``totals`` (see :meth:`IntegratedResult.to_dict`), ``condition``
            (``V_inf`` [m/s], ``alpha`` [rad], ``beta`` [rad]) when the
            result has one, ``spanwise`` (``y`` [m], ``Cl``, ``Cd_i`` per
            surface) and ``dispatch``, as *detail_level* sets.

        Raises
        ------
        ValueError
            If *detail_level* is not ``"summary"``, ``"standard"`` or
            ``"full"``.
        """
        if detail_level not in ("summary", "standard", "full"):
            raise ValueError(
                f"detail_level={detail_level!r}: use 'summary', 'standard' or 'full'."
            )
        out: dict[str, Any] = {
            "solver_type": self.solver_type,
            "converged": bool(self.converged),
            "iterations": int(self.iterations),
            "execution_time": float(self.execution_time),
            "totals": self.totals.to_dict(),
        }
        if self.condition is not None:
            out["condition"] = {
                "V_inf": float(self.condition.V_inf),
                "alpha": float(self.condition.alpha),
                "beta": float(self.condition.beta),
            }
        if detail_level in ("standard", "full"):
            out["spanwise"] = [
                {
                    "surface_name": sw.surface_name,
                    "y": sw.y.tolist() if isinstance(sw.y, np.ndarray) else list(sw.y),
                    "Cl": sw.Cl.tolist() if isinstance(sw.Cl, np.ndarray) else list(sw.Cl),
                    "Cd_i": sw.Cd_i.tolist() if isinstance(sw.Cd_i, np.ndarray) else list(sw.Cd_i),
                }
                for sw in self.spanwise
            ]
        if detail_level == "full":
            if "dispatch" in self.details:
                out["dispatch"] = dict(self.details["dispatch"])
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# JSON serialisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _airfoil_to_dict(af: AirfoilType) -> dict:
    """Serialise an airfoil to a plain dict."""
    if isinstance(af, LinearAirfoil):
        return {
            "type": "linear",
            "name": af.name,
            "a0": af.a0,
            "alpha_L0": af.alpha_L0,
            "Cd0": af.Cd0,
            "Cm0": af.Cm0,
        }
    return {
        "type": "tabulated",
        "name": af.name,
        "alpha": af.alpha.tolist(),
        "Cl": af.Cl_data.tolist(),
        "Cd": af.Cd_data.tolist(),
        "Cm": af.Cm_data.tolist() if af.Cm_data is not None else None,
        "Re": af.Re,
    }


def _airfoil_from_dict(d: dict) -> AirfoilType:
    """Deserialise an airfoil from a plain dict."""
    if d["type"] == "linear":
        return LinearAirfoil(
            name=d.get("name", ""),
            a0=d.get("a0", 2.0 * np.pi),
            alpha_L0=d.get("alpha_L0", 0.0),
            Cd0=d.get("Cd0", 0.0),
            Cm0=d.get("Cm0", 0.0),
        )
    return TabulatedAirfoil(
        name=d.get("name", ""),
        alpha=np.asarray(d["alpha"]),
        Cl_data=np.asarray(d["Cl"]),
        Cd_data=np.asarray(d["Cd"]),
        Cm_data=np.asarray(d["Cm"]) if d.get("Cm") is not None else None,
        Re=d.get("Re"),
    )


def _shape_to_json(shape: Any) -> Any:
    """Serialise a section shape specification to a JSON-compatible object."""
    if shape is None:
        return None
    if isinstance(shape, (str, int, float, bool)):
        return shape
    if hasattr(shape, "coords"):
        return {"name": getattr(shape, "name", "airfoil"), "coords": np.asarray(shape.coords, dtype=float).tolist()}
    if isinstance(shape, np.ndarray):
        return shape.tolist()
    if hasattr(shape, "__fspath__"):
        return str(shape)
    return shape


def _shape_from_json(data: Any) -> Any:
    """Deserialise a section shape specification from JSON data."""
    if data is None:
        return None
    if isinstance(data, dict) and "coords" in data:
        from ventorum.airfoil.geometry import AirfoilShape

        return AirfoilShape(name=data.get("name", "airfoil"), coords=np.asarray(data["coords"], dtype=float))
    if isinstance(data, list):
        return np.asarray(data, dtype=float)
    return data


def aircraft_to_json(ac: Aircraft) -> str:
    """Serialise an :class:`Aircraft` to a JSON string."""
    surfaces = []
    for surf in ac.surfaces:
        sections = []
        for sec in surf.sections:
            sec_dict = {
                "y_frac": sec.y_frac,
                "chord": sec.chord,
                "twist": sec.twist,
                "x_le": sec.x_le,
                "z_le": sec.z_le,
                "airfoil": _airfoil_to_dict(sec.airfoil),
            }
            if getattr(sec, "shape", None) is not None:
                sec_dict["shape"] = _shape_to_json(sec.shape)
            sections.append(sec_dict)
        controls = [
            {
                "name": c.name,
                "eta_start": c.eta_start,
                "eta_end": c.eta_end,
                "hinge_x_c": c.hinge_x_c,
                "deflection": c.deflection,
                "symmetric": c.symmetric,
            }
            for c in getattr(surf, "controls", [])
        ]
        nd = getattr(surf, "node_displacements", None)
        node_disp_json = (
            {
                "le": nd.le.tolist(),
                "te": nd.te.tolist(),
                "eta": None if nd.eta is None else nd.eta.tolist(),
            }
            if nd is not None
            else None
        )
        surfaces.append({
            "name": surf.name,
            "semi_span": surf.semi_span,
            "sections": sections,
            "is_symmetric": surf.is_symmetric,
            "position": surf.position.tolist(),
            "incidence": surf.incidence,
            "sweep_le": surf.sweep_le,
            "dihedral": surf.dihedral,
            "n_panels": surf.n_panels,
            "spacing": surf.spacing,
            "mirror_y": surf.mirror_y,
            "controls": controls,
            "node_displacements": node_disp_json,
        })
    data = {
        "name": ac.name,
        "surfaces": surfaces,
        # Values filled automatically are not saved, so they follow the geometry after loading.
        "S_ref": None if "S_ref" in ac._auto_ref else ac.S_ref,
        "b_ref": None if "b_ref" in ac._auto_ref else ac.b_ref,
        "c_ref": None if "c_ref" in ac._auto_ref else ac.c_ref,
        "ref_point": None if ac.ref_point is None else np.asarray(ac.ref_point, dtype=float).tolist(),
    }
    return json.dumps(data, indent=2)


def aircraft_from_json(text: str) -> Aircraft:
    """Deserialise an :class:`Aircraft` from a JSON string."""
    data = json.loads(text)
    surfaces = []
    for sd in data["surfaces"]:
        sections = []
        for sec_d in sd["sections"]:
            sections.append(WingSection(
                y_frac=sec_d["y_frac"],
                chord=sec_d["chord"],
                twist=sec_d.get("twist", 0.0),
                x_le=sec_d.get("x_le"),
                z_le=sec_d.get("z_le"),
                airfoil=_airfoil_from_dict(sec_d["airfoil"]),
                shape=_shape_from_json(sec_d.get("shape")),
            ))
        controls = [
            ControlSurface(
                name=cd["name"],
                eta_start=cd["eta_start"],
                eta_end=cd["eta_end"],
                hinge_x_c=cd["hinge_x_c"],
                deflection=cd["deflection"],
                symmetric=cd.get("symmetric", True),
            )
            for cd in sd.get("controls", [])
        ]
        nd_data = sd.get("node_displacements")
        if nd_data is not None:
            eta_data = nd_data.get("eta")
            node_disp = NodeDisplacements(
                le=np.asarray(nd_data["le"], dtype=float),
                te=np.asarray(nd_data["te"], dtype=float),
                eta=None if eta_data is None else np.asarray(eta_data, dtype=float),
            )
        else:
            node_disp = None
        surfaces.append(LiftingSurface(
            name=sd.get("name", "Surface"),
            semi_span=sd["semi_span"],
            sections=sections,
            is_symmetric=sd.get("is_symmetric", True),
            position=np.asarray(sd.get("position", [0, 0, 0])),
            incidence=sd.get("incidence", 0.0),
            sweep_le=sd.get("sweep_le", 0.0),
            dihedral=sd.get("dihedral", 0.0),
            n_panels=sd.get("n_panels", None),
            spacing=sd.get("spacing", None),
            mirror_y=sd.get("mirror_y", False),
            controls=controls,
            node_displacements=node_disp,
        ))
    return Aircraft(
        name=data.get("name", "Aircraft"),
        surfaces=surfaces,
        S_ref=data.get("S_ref"),
        b_ref=data.get("b_ref"),
        c_ref=data.get("c_ref"),
        ref_point=None if data.get("ref_point") is None else np.asarray(data["ref_point"], dtype=float),
    )
