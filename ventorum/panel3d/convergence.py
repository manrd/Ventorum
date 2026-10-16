# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Mesh convergence and Richardson error estimates for Ventorum-3DP.

The module refines the panel mesh in both directions at once: the
spanwise count and the chordwise count are both multiplied by the same
ratio from the base mesh (the mesh of the user), over three or four
levels. For each requested quantity the module stores the values per
level, the observed order of convergence, the Richardson-extrapolated
value, the error bar of the finest level, and the error estimate of the
base mesh. A quantity whose observed order falls outside [0.5, 4.0] is
flagged as outside the asymptotic range.

The classification, the observed order, the Richardson value and the
error bar of the finest level come from
:func:`ventorum.core.error_bars.coefficient_error_bar` (Roache 1998;
Celik et al. 2008), the layer-1 numerical error bar of Ventorum; this
module does not copy its rules. The linear refinement measure is the
spanwise panel count of the main surface: both directions are refined by
the same ratio. Per-level scaling of surfaces with their own panel count
reuses the level helper of :mod:`ventorum.geometry.mesh_convergence`.

Error estimate of the base mesh. With a monotonic convergence the
estimate is ``Fs * |f_ext - f_0|``, the safety factor of the error bar
times the distance from the base-mesh value to the Richardson value. In
the other states with an error bar (round-off, oscillatory, order not
found) the estimate is ``|f_fine - f_0|`` plus the half-width of the
finest level. A divergent sequence gives no estimate.

Cost. The study solves ``levels`` meshes. Panel counts grow by about
``ratio**2`` per level because both directions are refined, and the dense
LU factorization grows with the cube of the count. The cost factor is
``sum((Nk / N0)**3)``, about ``1 + 1.5**6 + 1.5**12`` (about 142) for
three levels at ratio 1.5. The solver option that attaches this estimate
to a result stays off by default for this reason.

Published sources
-----------------
* P. J. Roache, Verification and Validation in Computational Science
  and Engineering, Hermosa Publishers, 1998 (Grid Convergence Index
  and the safety factors).
* I. B. Celik, U. Ghia, P. J. Roache, C. J. Freitas, H. Coleman and
  P. E. Raad, "Procedure for estimation and reporting of uncertainty
  due to discretization in CFD applications", Journal of Fluids
  Engineering, Vol. 130, No. 7, 078001, 2008 (fixed-point iteration of
  the observed order for non-constant refinement ratios).
* J. L. Hess and A. M. O. Smith, "Calculation of potential flow about
  arbitrary bodies", Progress in Aeronautical Sciences, Vol. 8, 1967,
  pp. 1-138 (quadrilateral source panels on a closed body).
* J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., Cambridge
  University Press, 2001, chapter 12 (constant source-doublet panels;
  the surface velocity and Cp error falls with the panel size).
* J. D. Anderson, Fundamentals of Aerodynamics, 5th ed., McGraw-Hill,
  2011, section 4.5 (incompressible flow past a sphere: Cp max of 1 at
  the stagnation point, zero lift and zero drag).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverResult, SolverSettings
from ventorum.core.error_bars import ORDER_LOWER, ORDER_UPPER, coefficient_error_bar
from ventorum.geometry.mesh_convergence import _level_aircraft

__all__ = [
    "ConvergenceReport",
    "QuantityConvergence",
    "level_counts",
    "order_and_extrapolation",
    "panel_mesh_convergence",
]

#: Smallest spanwise or chordwise panel count of a level.
MIN_LEVEL_COUNT = 2

#: Quantities that a convergence study can track.
KNOWN_QUANTITIES = ("CL", "CDi", "CD_total", "CY", "Cl", "Cm", "Cn")


def _quantity_value(result: SolverResult, quantity: str) -> float:
    """Return one integrated coefficient of a solver result."""
    totals = result.totals
    value = {
        "CL": totals.CL,
        "CDi": totals.CDi,
        "CD_total": totals.CD_total,
        "CY": totals.CY,
        "Cl": totals.Cl,
        "Cm": totals.Cm,
        "Cn": totals.Cn,
    }[quantity]
    if value is None:
        raise ValueError(f"Quantity {quantity!r} is not available in this result.")
    return float(value)


@dataclass(slots=True)
class QuantityConvergence:
    """Convergence of one quantity over the mesh levels.

    Attributes
    ----------
    quantity : str
        Quantity name (for example ``"CL"``).
    values : list of float
        Quantity values per level [-], coarse level (the base mesh) first.
    total_panels : list of int
        Solved panel count per level [-], coarse level first.
    r21 : float
        Refinement ratio of the two finest levels [-].
    r32 : float
        Refinement ratio of the two coarsest levels of the triplet [-].
    observed_order : float or None
        Observed order of convergence [-], or None when not found.
    f_extrapolated : float or None
        Richardson-extrapolated value [-] (monotonic convergence only).
    abs_error : float or None
        Half-width of the error bar of the finest level [-], or None.
    relative_error : float or None
        abs_error divided by the reference value [-] (the Richardson value
        when there is one, else the finest value).
    in_asymptotic_range : bool
        True for a monotonic convergence with the order in [0.5, 4.0].
    state : str
        ``"monotonic"``, ``"roundoff"``, ``"oscillatory"``,
        ``"divergent"`` or ``"not_converged_order"``.
    note : str
        Human-readable note about this quantity.
    base_value : float or None
        Value on the base mesh (the mesh of the user) [-].
    base_abs_error : float or None
        Error estimate of the base-mesh value [-] (see the module notes).
    base_relative_error : float or None
        base_abs_error divided by the reference value [-].
    """

    quantity: str = "CL"
    values: list[float] = field(default_factory=list)
    total_panels: list[int] = field(default_factory=list)
    r21: float = 1.5
    r32: float = 1.5
    observed_order: float | None = None
    f_extrapolated: float | None = None
    abs_error: float | None = None
    relative_error: float | None = None
    in_asymptotic_range: bool = False
    state: str = "divergent"
    note: str = ""
    base_value: float | None = None
    base_abs_error: float | None = None
    base_relative_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the convergence of this quantity as plain Python values."""
        def num(value: float | None) -> float | None:
            if value is None:
                return None
            number = float(value)
            return number if math.isfinite(number) else None

        return {
            "quantity": str(self.quantity),
            "values": [num(v) for v in self.values],
            "total_panels": [int(n) for n in self.total_panels],
            "r21": float(self.r21),
            "r32": float(self.r32),
            "observed_order": num(self.observed_order),
            "f_extrapolated": num(self.f_extrapolated),
            "abs_error": num(self.abs_error),
            "relative_error": num(self.relative_error),
            "in_asymptotic_range": bool(self.in_asymptotic_range),
            "state": str(self.state),
            "note": str(self.note),
            "base_value": num(self.base_value),
            "base_abs_error": num(self.base_abs_error),
            "base_relative_error": num(self.base_relative_error),
        }


@dataclass
class ConvergenceReport:
    """Mesh convergence report of a 3D panel study.

    There is no existing convergence report type for panels, so this is a
    new dataclass. It carries one :class:`QuantityConvergence` per tracked
    quantity, the mesh levels, and the cost of the study.

    Attributes
    ----------
    quantities : dict
        Convergence per quantity name.
    n_span_levels : list of int
        Spanwise panels per semi-span [-], coarse level first.
    n_chord_levels : list of int
        Chordwise panels per surface [-], coarse level first.
    total_panels : list of int
        Solved panel count per level [-], coarse level first.
    levels : int
        Number of mesh levels (3 or 4).
    ratio : float
        Refinement ratio applied in both directions [-].
    cost_factor : float
        Study cost relative to one base solve [-], as dense LU work
        ``sum((Nk / N0)**3)``.
    notes : list of str
        Human-readable notes, one per quantity plus the study header.
    results : list of SolverResult
        Solver results of each level, coarse level first (not in
        ``to_dict``).
    """

    quantities: dict[str, QuantityConvergence] = field(default_factory=dict)
    n_span_levels: list[int] = field(default_factory=list)
    n_chord_levels: list[int] = field(default_factory=list)
    total_panels: list[int] = field(default_factory=list)
    levels: int = 3
    ratio: float = 1.5
    cost_factor: float = 1.0
    notes: list[str] = field(default_factory=list)
    results: list[SolverResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a dict with JSON types only."""
        return {
            "quantities": {k: v.to_dict() for k, v in self.quantities.items()},
            "n_span_levels": [int(n) for n in self.n_span_levels],
            "n_chord_levels": [int(n) for n in self.n_chord_levels],
            "total_panels": [int(n) for n in self.total_panels],
            "levels": int(self.levels),
            "ratio": float(self.ratio),
            "cost_factor": float(self.cost_factor),
            "notes": [str(n) for n in self.notes],
        }

    def summary(self) -> str:
        """Return a short text table with values, order and error."""
        lines = [
            f"Panel mesh convergence: {self.levels} levels, ratio {self.ratio:.3f}, "
            f"cost factor {self.cost_factor:.1f}",
            f"{'quantity':<8} {'values (coarse to fine)':<42} {'order':<7} {'extrapolated':<13} "
            f"{'base err':<9} state",
        ]
        for name, conv in self.quantities.items():
            vals = ", ".join(f"{v:.5f}" for v in conv.values)
            order = "n/a" if conv.observed_order is None else f"{conv.observed_order:.2f}"
            ext = "n/a" if conv.f_extrapolated is None else f"{conv.f_extrapolated:.5f}"
            err = "n/a" if conv.base_relative_error is None else f"{conv.base_relative_error:.2%}"
            lines.append(f"{name:<8} {vals:<42} {order:<7} {ext:<13} {err:<9} {conv.state}")
        return "\n".join(lines)


def level_counts(n_base: int, ratio: float, levels: int) -> list[int]:
    """Return the panel counts of each refinement level.

    Level ``k`` holds ``floor(n_base * ratio**k)`` panels, at least one
    more than the previous level and at least two panels.

    Parameters
    ----------
    n_base : int
        Panel count of the base (coarsest) mesh [-].
    ratio : float
        Refinement ratio applied at each level [-].
    levels : int
        Number of levels [-].

    Returns
    -------
    list of int
        Strictly increasing panel counts [-], coarse level first.
    """
    counts: list[int] = []
    for k in range(int(levels)):
        n = max(MIN_LEVEL_COUNT, int(math.floor(float(n_base) * float(ratio) ** k + 1e-9)))
        if counts:
            n = max(n, counts[-1] + 1)
        counts.append(int(n))
    return counts


def order_and_extrapolation(
    values: Sequence[float],
    levels: Sequence[int],
    quantity: str = "f",
    base_value: float | None = None,
) -> QuantityConvergence:
    """Compute the observed order, the Richardson estimate and the error bars of 3 values.

    This is a pure function of the three level values (coarse level
    first). It calls :func:`ventorum.core.error_bars.coefficient_error_bar`
    for the classification, the order, the Richardson value and the bar of
    the finest level, and adds the error estimate of the base mesh (see the
    module notes).

    Parameters
    ----------
    values : sequence of float
        Quantity values [-] of the coarse, medium and fine levels.
    levels : sequence of int
        Linear panel counts [-] of the coarse, medium and fine levels
        (strictly increasing); the refinement ratios are their quotients.
    quantity : str
        Quantity name for the report.
    base_value : float or None, optional
        Value on the base mesh [-]. None takes the coarse value of the
        triplet.

    Returns
    -------
    QuantityConvergence
        Observed order, Richardson value, the bar of the finest level, the
        error estimate of the base mesh and the asymptotic-range flag.
    """
    vals = [float(v) for v in values]
    counts = [int(n) for n in levels]
    if len(vals) != 3 or len(counts) != 3:
        raise ValueError(
            f"order_and_extrapolation needs 3 level values and 3 levels, got {len(vals)} and {len(counts)}."
        )
    if not (counts[2] > counts[1] > counts[0] >= 1):
        raise ValueError(f"levels={counts} must be strictly increasing positive panel counts, coarse first.")
    bar = coefficient_error_bar(str(quantity), vals[::-1], counts[::-1])
    layer = bar.layers["numerical"]
    f_fine = vals[2]
    f_base = vals[0] if base_value is None else float(base_value)
    half = layer.half_width
    order = layer.observed_order
    f_ext = layer.f_extrapolated
    ref = abs(f_ext) if f_ext is not None else abs(f_fine)
    rel = None if half is None else half / max(ref, 1e-300)

    base_err = None
    if layer.state == "monotonic" and f_ext is not None and layer.safety_factor is not None:
        base_err = float(layer.safety_factor) * abs(f_ext - f_base)
    elif half is not None:
        base_err = abs(f_fine - f_base) + float(half)
    base_rel = None if base_err is None else base_err / max(ref, 1e-300)

    in_range = bool(
        layer.state == "monotonic" and order is not None and ORDER_LOWER <= order <= ORDER_UPPER
    )
    if order is not None:
        note = f"{quantity}: observed order {order:.3f}"
        note += (
            " is in the asymptotic range."
            if in_range
            else f" is outside [{ORDER_LOWER}, {ORDER_UPPER}] (not in the asymptotic range)."
        )
    else:
        note = f"{quantity}: convergence state {layer.state}; no observed order."
    if base_err is not None:
        note += (
            f" Error estimate of the given mesh: {f_base:.6g} +/- {base_err:.3g}"
            + (f" ({base_rel:.2%})." if base_rel is not None else ".")
        )
    else:
        note += " No error estimate of the given mesh."
    for extra in bar.notes:
        note += f" {extra[0].upper()}{extra[1:]}."
    return QuantityConvergence(
        quantity=str(quantity),
        values=vals,
        r21=counts[2] / counts[1],
        r32=counts[1] / counts[0],
        observed_order=order,
        f_extrapolated=f_ext,
        abs_error=half,
        relative_error=rel,
        in_asymptotic_range=in_range,
        state=str(layer.state),
        note=note,
        base_value=f_base,
        base_abs_error=base_err,
        base_relative_error=base_rel,
    )


def panel_mesh_convergence(
    aircraft: Aircraft | LiftingSurface,
    condition: FlightCondition | None,
    settings: SolverSettings,
    levels: int = 3,
    ratio: float = 1.5,
    quantities: Sequence[str] = ("CL", "CDi", "Cm"),
) -> ConvergenceReport:
    """Run a mesh convergence study of the 3D panel method.

    The study refines the spanwise and chordwise counts by *ratio* from
    the base mesh over *levels* levels and solves each level with the
    panel method. The observed order, the Richardson value and the bar of
    the finest level come from the three finest levels; the error
    estimate of the base mesh (level 0, the mesh of the user) follows the
    module notes.

    Parameters
    ----------
    aircraft : Aircraft or LiftingSurface
        Case geometry. A single surface is wrapped into an aircraft.
    condition : FlightCondition or None
        Flight condition, the same for all levels. None uses the default
        condition (alpha 5 deg, V_inf 50 m/s, free air).
    settings : SolverSettings
        Solver settings of the base (coarsest) mesh. The spanwise count
        is ``settings.n_panels`` [-] per semi-span; the chordwise count
        is ``settings.n_chord`` [-] per surface, or 20 when None (the
        default of the panel solver). The solver type is set to the
        panel method on every level, and the levels do not run a study of
        their own.
    levels : int
        Number of mesh levels: 3 or 4.
    ratio : float
        Refinement ratio applied in both directions [-]. Must exceed 1.
    quantities : sequence of str
        Tracked quantities (any of ``"CL"``, ``"CDi"``, ``"CD_total"``,
        ``"CY"``, ``"Cl"``, ``"Cm"``, ``"Cn"``).

    Returns
    -------
    ConvergenceReport
        Values per level, observed order, Richardson value, the error bar
        of the finest level and the error estimate of the base mesh per
        quantity, with the asymptotic-range flag.
    """
    from ventorum import analyze
    from ventorum.solvers.panel3d import _DEFAULT_N_CHORD

    if int(levels) not in (3, 4):
        raise ValueError(f"levels must be 3 or 4, got {levels}.")
    if not float(ratio) > 1.0:
        raise ValueError(f"ratio must exceed 1, got {ratio}.")
    names = tuple(quantities)
    if not names:
        raise ValueError("quantities must hold at least one quantity name.")
    for name in names:
        if name not in KNOWN_QUANTITIES:
            raise ValueError(f"Unknown quantity {name!r}; use one of {list(KNOWN_QUANTITIES)}.")

    ac = (
        Aircraft(name="SingleWing", surfaces=[aircraft])
        if isinstance(aircraft, LiftingSurface)
        else aircraft
    )
    cond = condition.clone() if condition is not None else FlightCondition()

    n_span0 = int(settings.n_panels)
    n_chord0 = int(settings.n_chord) if settings.n_chord is not None else int(_DEFAULT_N_CHORD)
    spans = level_counts(n_span0, float(ratio), int(levels))
    chords = level_counts(n_chord0, float(ratio), int(levels))

    base = settings.clone()
    base.solver_type = "panel3d"
    base.panel_estimate_mesh_error = False
    n_base = max(int(settings.n_panels), 1)

    results: list[SolverResult] = []
    for n_span, n_chord in zip(spans, chords):
        level_settings = base.clone()
        level_settings.n_panels = n_span
        level_settings.n_chord = n_chord
        level_aircraft = _level_aircraft(ac, n_span, n_base)
        results.append(analyze(level_aircraft, condition=cond, settings=level_settings))

    totals = [int(np.size(res.details.get("cp", []))) for res in results]
    if any(n <= 0 for n in totals):
        totals = [2 * ns * nc for ns, nc in zip(spans, chords)]

    conv_quantities: dict[str, QuantityConvergence] = {}
    for name in names:
        series = [_quantity_value(res, name) for res in results]
        conv = order_and_extrapolation(series[-3:], spans[-3:], quantity=name, base_value=series[0])
        conv.values = [float(v) for v in series]
        conv.total_panels = [int(n) for n in totals]
        conv_quantities[name] = conv

    n0 = max(totals[0], 1)
    cost = float(sum((n / n0) ** 3 for n in totals))
    header = (
        f"Panel mesh convergence: {int(levels)} levels, ratio {float(ratio):.3f}, "
        f"span panels {spans}, chord panels {chords}, cost factor {cost:.1f} "
        f"relative to one base solve (dense LU work)."
    )
    notes = [header] + [conv.note for conv in conv_quantities.values()]
    return ConvergenceReport(
        quantities=conv_quantities,
        n_span_levels=[int(n) for n in spans],
        n_chord_levels=[int(n) for n in chords],
        total_panels=[int(n) for n in totals],
        levels=int(levels),
        ratio=float(ratio),
        cost_factor=cost,
        notes=notes,
        results=results,
    )
