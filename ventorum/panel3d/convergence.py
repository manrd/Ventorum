# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Mesh convergence and Richardson error estimates for Ventorum-3DP.

The module refines the panel mesh in both directions at once: the
spanwise count and the chordwise count are both multiplied by the same
ratio from the base mesh, over three or four levels. For each requested
quantity the module stores the values per level, the observed order of
convergence, the Richardson-extrapolated value and a relative error
estimate. A quantity whose observed order falls outside [0.5, 4.0] is
flagged as outside the asymptotic range.

The observed order and the Richardson value reuse
:func:`ventorum.core.error_bars.observed_order` and
:func:`ventorum.core.error_bars.richardson_extrapolate` (Celik et al.
2008). The safety factors 1.25 inside the asymptotic range and 3.0
outside it, the order limits 0.5 and 4.0, and the roundoff,
oscillatory, divergent and not-converged-order rules follow the same
conventions as the layer-1 numerical error bars of Ventorum. Per-level
scaling of surfaces with their own panel count reuses the level helper
of :mod:`ventorum.geometry.mesh_convergence`.

Cost. The study solves ``levels`` meshes. Panel counts grow by about
``ratio**2`` per level because both directions are refined, and dense
assembly and solve work grows with the square of the count. The cost
factor ``sum((Nk / N0)**2)`` is about 30 for three levels at ratio 1.5.
The solver option that attaches this estimate to a result stays off by
default for this reason.

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
  arbitrary bodies", Journal of Ship Research, Vol. 8, 1964
  (quadrilateral source-doublet panels on a closed body).
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
from ventorum.geometry.mesh_convergence import _level_aircraft
from ventorum.panel3d.mesh import PanelMesh

__all__ = [
    "ConvergenceReport",
    "QuantityConvergence",
    "build_sphere_mesh",
    "level_counts",
    "order_and_extrapolation",
    "panel_mesh_convergence",
]

#: Lower limit of the trusted observed-order range (project decision).
ORDER_LOWER = 0.5

#: Upper limit of the trusted observed-order range (project decision).
ORDER_UPPER = 4.0

#: Safety factor inside the asymptotic range (Roache 1998).
SAFETY_IN_RANGE = 1.25

#: Safety factor outside the asymptotic range (project decision).
SAFETY_OUT_OF_RANGE = 3.0

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
        Quantity values per level [-], coarse level first.
    total_panels : list of int
        Solved panel count per level [-], coarse level first.
    r21 : float
        Refinement ratio of the two finest levels [-].
    r32 : float
        Refinement ratio of the two coarsest levels [-].
    observed_order : float or None
        Observed order of convergence [-], or None when not found.
    f_extrapolated : float or None
        Richardson-extrapolated value [-] (information only).
    abs_error : float or None
        Absolute error bar of the finest level [-], or None without one.
    relative_error : float or None
        Absolute error divided by the reference value [-].
    in_asymptotic_range : bool
        False when the order is outside [0.5, 4.0] or was not found.
    state : str
        ``"monotonic"``, ``"roundoff"``, ``"oscillatory"``,
        ``"divergent"`` or ``"not_converged_order"``.
    note : str
        Human-readable note about this quantity.
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
        Study cost relative to one base solve [-], as dense work
        ``sum((Nk / N0)**2)``.
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
            f"{'quantity':<8} {'values (coarse to fine)':<42} {'order':<7} {'extrapolated':<13} {'rel err':<9} state",
        ]
        for name, conv in self.quantities.items():
            vals = ", ".join(f"{v:.5f}" for v in conv.values)
            order = "n/a" if conv.observed_order is None else f"{conv.observed_order:.2f}"
            ext = "n/a" if conv.f_extrapolated is None else f"{conv.f_extrapolated:.5f}"
            err = "n/a" if conv.relative_error is None else f"{conv.relative_error:.2%}"
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
    r21: float,
    r32: float,
    quantity: str = "f",
) -> QuantityConvergence:
    """Compute the observed order and the Richardson estimate of 3 values.

    This is a pure function of the three level values (coarse level
    first). It calls :func:`ventorum.core.error_bars.observed_order` and
    :func:`ventorum.core.error_bars.richardson_extrapolate`; it does not
    reimplement them.

    Parameters
    ----------
    values : sequence of float
        Quantity values [-] of the coarse, medium and fine levels.
    r21 : float
        Refinement ratio of the medium to the fine level [-].
    r32 : float
        Refinement ratio of the coarse to the medium level [-].
    quantity : str
        Quantity name for the report.

    Returns
    -------
    QuantityConvergence
        Observed order, Richardson value, absolute and relative error of
        the fine level, and the asymptotic-range flag.
    """
    from ventorum.core.error_bars import observed_order, richardson_extrapolate

    vals = [float(v) for v in values]
    if len(vals) != 3:
        raise ValueError(f"order_and_extrapolation needs 3 level values, got {len(vals)}.")
    f3, f2, f1 = vals[0], vals[1], vals[2]
    r21 = float(r21)
    r32 = float(r32)
    if not (r21 > 1.0 and r32 > 1.0):
        raise ValueError(f"Refinement ratios must exceed 1, got r21={r21} and r32={r32}.")

    def report(
        state: str,
        order: float | None,
        f_ext: float | None,
        abs_err: float | None,
        in_range: bool,
        note: str,
    ) -> QuantityConvergence:
        rel = None
        if abs_err is not None:
            ref = abs(f_ext) if f_ext is not None else abs(f1)
            rel = abs_err / max(ref, 1e-300)
        return QuantityConvergence(
            quantity=str(quantity),
            values=vals,
            r21=r21,
            r32=r32,
            observed_order=order,
            f_extrapolated=f_ext,
            abs_error=abs_err,
            relative_error=rel,
            in_asymptotic_range=bool(in_range),
            state=state,
            note=note,
        )

    if not all(math.isfinite(v) for v in vals):
        return report("divergent", None, None, None, False, f"{quantity}: a level value is not finite.")

    eps21 = f2 - f1
    eps32 = f3 - f2
    scale = max(1.0, abs(f1), abs(f2), abs(f3))
    spread = max(abs(eps21), abs(eps32))
    if spread <= 1e-12 * scale:
        return report(
            "roundoff", None, f1, 3.0 * spread, False,
            f"{quantity}: level changes are at round-off level.",
        )

    big_r = math.inf if eps32 == 0.0 else eps21 / eps32
    if big_r < 0.0:
        return report(
            "oscillatory", None, None, None, False,
            f"{quantity}: values oscillate with refinement, so there is no estimate.",
        )
    if big_r >= 1.0:
        return report(
            "divergent", None, None, None, False,
            f"{quantity}: the change grows with refinement, so there is no estimate.",
        )

    order = observed_order(f1, f2, f3, r21, r32)
    if order is None or not math.isfinite(order):
        if eps21 == 0.0:
            half = 3.0 * max(abs(eps21), abs(eps32))
        else:
            half = 3.0 * abs(eps21)
        return report(
            "not_converged_order", None, None, half, False,
            f"{quantity}: the observed-order iteration did not converge; the bar uses a conservative rule.",
        )

    if ORDER_LOWER <= order <= ORDER_UPPER:
        safety = SAFETY_IN_RANGE
        p_used = order
        in_range = True
    else:
        safety = SAFETY_OUT_OF_RANGE
        p_used = min(max(order, ORDER_LOWER), ORDER_UPPER)
        in_range = False
    f_ext = richardson_extrapolate(f1, f2, r21, p_used)
    abs_err = safety * abs(f1 - f_ext)
    note = f"{quantity}: observed order {order:.3f}"
    if not in_range:
        note += f" is outside [{ORDER_LOWER}, {ORDER_UPPER}] (not in the asymptotic range)."
    else:
        note += " is in the asymptotic range."
    return report("monotonic", order, f_ext, abs_err, in_range, note)


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
    panel method. The observed order and the Richardson value of each
    quantity come from the three finest levels.

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
        panel method on every level.
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
        Values per level, observed order, Richardson value and relative
        error per quantity, with the asymptotic-range flag.
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

    triplet = totals[-3:]
    r21 = math.sqrt(triplet[2] / triplet[1])
    r32 = math.sqrt(triplet[1] / triplet[0])

    conv_quantities: dict[str, QuantityConvergence] = {}
    for name in names:
        series = [_quantity_value(res, name) for res in results]
        conv = order_and_extrapolation(series[-3:], r21, r32, quantity=name)
        conv.values = [float(v) for v in series]
        conv.total_panels = [int(n) for n in totals]
        conv_quantities[name] = conv

    n0 = max(totals[0], 1)
    cost = float(sum((n / n0) ** 2 for n in totals))
    header = (
        f"Panel mesh convergence: {int(levels)} levels, ratio {float(ratio):.3f}, "
        f"span panels {spans}, chord panels {chords}, cost factor {cost:.1f} "
        f"relative to one base solve."
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


def build_sphere_mesh(radius_m: float, n_lat: int, n_lon: int) -> PanelMesh:
    """Build a closed latitude-longitude quadrilateral mesh of a sphere.

    The mesh verifies the panel method against the analytic
    incompressible flow past a sphere (Anderson 2011, section 4.5): Cp
    max of 1 at the stagnation point, zero lift and zero drag. Panels
    point outward and share single nodes at the poles, so the mesh is
    watertight.

    Parameters
    ----------
    radius_m : float
        Sphere radius [m]. Must be positive and finite.
    n_lat : int
        Latitude bands from pole to pole [-]. Must be at least 2.
    n_lon : int
        Panels around each latitude ring [-]. Must be at least 3.

    Returns
    -------
    PanelMesh
        Closed sphere mesh with outward panels (no wake edges).
    """
    radius = float(radius_m)
    n_lat = int(n_lat)
    n_lon = int(n_lon)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"radius_m must be positive and finite, got {radius_m}.")
    if n_lat < 2:
        raise ValueError(f"n_lat must be at least 2, got {n_lat}.")
    if n_lon < 3:
        raise ValueError(f"n_lon must be at least 3, got {n_lon}.")

    nodes: list[list[float]] = [[0.0, 0.0, radius]]
    index: dict[tuple[int, int], int] = {}
    for i in range(1, n_lat):
        theta = math.pi * i / n_lat
        for j in range(n_lon):
            phi = 2.0 * math.pi * j / n_lon
            index[(i, j)] = len(nodes)
            nodes.append([
                radius * math.sin(theta) * math.cos(phi),
                radius * math.sin(theta) * math.sin(phi),
                radius * math.cos(theta),
            ])
    north = 0
    south = len(nodes)
    nodes.append([0.0, 0.0, -radius])

    def ring(i: int, j: int) -> int:
        if i == 0:
            return north
        if i == n_lat:
            return south
        return index[(i, j % n_lon)]

    panels: list[list[int]] = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = ring(i, j)
            b = ring(i, j + 1)
            c = ring(i + 1, j + 1)
            d = ring(i + 1, j)
            if i == 0:
                panels.append([b, c, d, d])
            elif i == n_lat - 1:
                panels.append([a, b, c, c])
            else:
                panels.append([d, c, b, a])
    n_panels = len(panels)
    return PanelMesh(
        nodes=np.asarray(nodes, dtype=float),
        panels=np.asarray(panels, dtype=int),
        component=np.zeros(n_panels, dtype=int),
        kind=np.array(["surface"] * n_panels),
        wake_edges=np.zeros((0, 4), dtype=int),
        half=False,
    )
