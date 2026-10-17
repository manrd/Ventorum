# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Mesh convergence and Richardson error estimates for Ventorum-3DP.

The module refines the panel mesh in each direction separately. From the
base mesh (the mesh of the user, ``n_span`` spanwise and ``n_chord``
chordwise panels) a chordwise refinement multiplies the chordwise count by
a ratio over three or four levels at the base spanwise count, and a
spanwise refinement multiplies the spanwise count by the same ratio at the
base chordwise count. The base mesh is the first level of both sequences.
For each requested quantity and each direction the module stores the
values per level, the observed order of convergence, the
Richardson-extrapolated value, the error bar of the finest level, and the
error estimate of the base mesh. A direction whose observed order falls
outside [0.5, 4.0] is flagged as outside the asymptotic range.

Reason for the separate directions. On a thick wing the chordwise error
and the spanwise error of the lift have opposite signs (too little lift
from the chordwise discretization, too much from the spanwise one). When
both counts grow together, the two errors partly cancel and the sequence
is often not monotonic, so a refinement in both directions at once gives
no reliable order or Richardson value. Each direction alone has one sign
of error, so its sequence is monotonic once that direction is in its
asymptotic range.

Separable error model. The error of a quantity is modelled as the sum of
a spanwise and a chordwise term,
``f(ns, nc) = f_inf + a ns**(-p) + b nc**(-q)``. The Richardson value of
the chordwise sequence is then ``R_c = f(ns_0, inf)`` and that of the
spanwise sequence ``R_s = f(inf, nc_0)``, so the value of the converged
mesh is ``f_inf = R_c + R_s - f_0``, with ``f_0`` the base-mesh value. The
error estimate of the base mesh is the sum of the two directional
estimates, ``e = e_c + e_s``. Because the base error is
``f_0 - f_inf = (f_0 - R_c) + (f_0 - R_s)`` and each directional estimate
is at least the distance to its Richardson value, under the model the
sum bounds the base error also when the two errors have opposite signs.

Directional estimate. The classification, the observed order, the
Richardson value and the error bar of the finest level of each direction
come from :func:`ventorum.core.error_bars.coefficient_error_bar` (Roache
1998; Celik et al. 2008), the layer-1 numerical error bar of Ventorum;
this module does not copy its rules. The three finest levels of a
direction go to the error bar. The error estimate of the base mesh in one
direction is ``Fs * |f_ext - f_0|`` for a monotonic convergence, with the
safety factor of the error bar; in the other states with an error bar
(round-off, oscillatory, order not found) it is ``|f_fine - f_0|`` plus the
half-width of the finest level. A divergent direction gives no estimate,
and then the combined estimate is not given either. Per-level scaling of
surfaces with their own spanwise count reuses the level helper of
:mod:`ventorum.geometry.mesh_convergence`.

Cost. The study solves the base mesh and ``levels - 1`` more meshes in
each direction (five meshes for three levels). A chordwise or a spanwise
level multiplies the panel count by about the ratio, and the dense LU
factorization grows with the cube of the count. The cost factor is
``sum((Nk / N0)**3)`` over the solved meshes, about
``1 + 2 (r**3 + r**6)`` for three levels at ratio r: about 30 at ratio
1.5. The solver option that attaches this estimate to a result stays off
by default for this reason.

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
* L. F. Richardson and J. A. Gaunt, "The deferred approach to the
  limit", Philosophical Transactions of the Royal Society of London,
  Series A, Vol. 226, 1927, pp. 299-361 (extrapolation of a sequence of
  discretizations).
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
    "MeshErrorEstimate",
    "QuantityConvergence",
    "directional_estimate",
    "level_counts",
    "order_and_extrapolation",
    "panel_mesh_convergence",
]

#: Smallest spanwise or chordwise panel count of a level.
MIN_LEVEL_COUNT = 2

#: Quantities that a convergence study can track.
KNOWN_QUANTITIES = ("CL", "CDi", "CD_total", "CY", "Cl", "Cm", "Cn")

#: Relative tolerance [-] of the common base value of the two directions.
_BASE_REL_TOL = 1e-9


def _num(value: float | None) -> float | None:
    """Return a finite float, or None."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None

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
    """Convergence of one quantity over the mesh levels of one refinement direction.

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
        return {
            "quantity": str(self.quantity),
            "values": [_num(v) for v in self.values],
            "total_panels": [int(n) for n in self.total_panels],
            "r21": float(self.r21),
            "r32": float(self.r32),
            "observed_order": _num(self.observed_order),
            "f_extrapolated": _num(self.f_extrapolated),
            "abs_error": _num(self.abs_error),
            "relative_error": _num(self.relative_error),
            "in_asymptotic_range": bool(self.in_asymptotic_range),
            "state": str(self.state),
            "note": str(self.note),
            "base_value": _num(self.base_value),
            "base_abs_error": _num(self.base_abs_error),
            "base_relative_error": _num(self.base_relative_error),
        }


@dataclass(slots=True)
class MeshErrorEstimate:
    """Error estimate of the base mesh from a chordwise and a spanwise refinement.

    The two directions combine with the separable error model of the module
    notes: the Richardson value is ``R_c + R_s - f_0`` and the error
    estimate of the base mesh is the sum of the two directional estimates.

    Attributes
    ----------
    quantity : str
        Quantity name (for example ``"CL"``).
    base_value : float or None
        Value on the base mesh (the mesh of the user) [-].
    f_extrapolated : float or None
        Richardson value of the separable model ``R_c + R_s - f_0`` [-],
        or None when a direction has no Richardson value.
    base_abs_error : float or None
        Error estimate of the base-mesh value [-]: the chordwise plus the
        spanwise estimate, or None when a direction gives none.
    base_relative_error : float or None
        base_abs_error divided by the reference value [-] (the Richardson
        value when there is one, else the base value).
    in_asymptotic_range : bool
        True when both directions converge monotonically with an order in
        [0.5, 4.0].
    state : str
        ``"monotonic"`` (both directions monotonic), ``"estimate"`` (both
        directions give an estimate, at least one is not monotonic) or
        ``"no_estimate"`` (a direction gives no estimate).
    note : str
        Human-readable note about this quantity.
    chordwise : QuantityConvergence or None
        Convergence of the chordwise refinement (base spanwise count).
    spanwise : QuantityConvergence or None
        Convergence of the spanwise refinement (base chordwise count).
    """

    quantity: str = "CL"
    base_value: float | None = None
    f_extrapolated: float | None = None
    base_abs_error: float | None = None
    base_relative_error: float | None = None
    in_asymptotic_range: bool = False
    state: str = "no_estimate"
    note: str = ""
    chordwise: QuantityConvergence | None = None
    spanwise: QuantityConvergence | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the estimate of this quantity as plain Python values."""
        return {
            "quantity": str(self.quantity),
            "base_value": _num(self.base_value),
            "f_extrapolated": _num(self.f_extrapolated),
            "base_abs_error": _num(self.base_abs_error),
            "base_relative_error": _num(self.base_relative_error),
            "in_asymptotic_range": bool(self.in_asymptotic_range),
            "state": str(self.state),
            "note": str(self.note),
            "chordwise": None if self.chordwise is None else self.chordwise.to_dict(),
            "spanwise": None if self.spanwise is None else self.spanwise.to_dict(),
        }


@dataclass
class ConvergenceReport:
    """Mesh convergence report of a 3D panel study with directional refinement.

    It carries one :class:`MeshErrorEstimate` per tracked quantity (with the
    chordwise and the spanwise convergence), the mesh levels of the two
    directions, the solved meshes and the cost of the study.

    Attributes
    ----------
    quantities : dict
        Error estimate per quantity name.
    n_span_levels : list of int
        Spanwise panels per semi-span of the spanwise refinement [-], base
        mesh first (at the base chordwise count).
    n_chord_levels : list of int
        Chordwise panels per surface of the chordwise refinement [-], base
        mesh first (at the base spanwise count).
    meshes : list of tuple of int
        ``(n_span, n_chord)`` of each solved mesh: the base mesh, then the
        finer chordwise levels, then the finer spanwise levels.
    total_panels : list of int
        Solved panel count of each mesh of *meshes* [-].
    levels : int
        Number of mesh levels per direction (3 or 4), the base mesh
        included.
    ratio : float
        Refinement ratio of each direction [-].
    cost_factor : float
        Study cost relative to one base solve [-], as dense LU work
        ``sum((Nk / N0)**3)`` over the solved meshes.
    notes : list of str
        Human-readable notes, one per quantity plus the study header.
    results : list of SolverResult
        Solver results of each mesh of *meshes* (not in ``to_dict``).
    """

    quantities: dict[str, MeshErrorEstimate] = field(default_factory=dict)
    n_span_levels: list[int] = field(default_factory=list)
    n_chord_levels: list[int] = field(default_factory=list)
    meshes: list[tuple[int, int]] = field(default_factory=list)
    total_panels: list[int] = field(default_factory=list)
    levels: int = 3
    ratio: float = 1.5
    cost_factor: float = 1.0
    notes: list[str] = field(default_factory=list)
    results: list[SolverResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a dict with JSON types only."""
        return {
            "method": "directional",
            "quantities": {k: v.to_dict() for k, v in self.quantities.items()},
            "n_span_levels": [int(n) for n in self.n_span_levels],
            "n_chord_levels": [int(n) for n in self.n_chord_levels],
            "meshes": [[int(a), int(b)] for a, b in self.meshes],
            "total_panels": [int(n) for n in self.total_panels],
            "levels": int(self.levels),
            "ratio": float(self.ratio),
            "cost_factor": float(self.cost_factor),
            "notes": [str(n) for n in self.notes],
        }

    def summary(self) -> str:
        """Return a short text table with the base value, the orders and the error estimate."""
        lines = [
            f"Panel mesh convergence (chordwise and spanwise refinement): {self.levels} levels per "
            f"direction, ratio {self.ratio:.3f}, cost factor {self.cost_factor:.1f}",
            f"{'quantity':<8} {'base value':<12} {'chord order':<12} {'span order':<11} {'extrapolated':<13} "
            f"{'base err':<9} state",
        ]

        def order(conv: QuantityConvergence | None) -> str:
            if conv is None or conv.observed_order is None:
                return "n/a"
            return f"{conv.observed_order:.2f}"

        for name, est in self.quantities.items():
            base = "n/a" if est.base_value is None else f"{est.base_value:.5f}"
            ext = "n/a" if est.f_extrapolated is None else f"{est.f_extrapolated:.5f}"
            err = "n/a" if est.base_relative_error is None else f"{est.base_relative_error:.2%}"
            lines.append(
                f"{name:<8} {base:<12} {order(est.chordwise):<12} {order(est.spanwise):<11} {ext:<13} "
                f"{err:<9} {est.state}"
            )
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


def _direction_convergence(
    values: Sequence[float],
    levels: Sequence[int],
    quantity: str,
    direction: str,
) -> QuantityConvergence:
    """Return the convergence of one direction: the three finest levels and the base-mesh estimate."""
    vals = [float(v) for v in values]
    counts = [int(n) for n in levels]
    if len(vals) < 3 or len(vals) != len(counts):
        raise ValueError(
            f"The {direction} refinement needs at least 3 values and one count per value, "
            f"got {len(vals)} values and {len(counts)} counts."
        )
    conv = order_and_extrapolation(vals[-3:], counts[-3:], quantity=quantity, base_value=vals[0])
    conv.values = vals
    conv.note = f"{direction.capitalize()} refinement, {conv.note}"
    return conv


def directional_estimate(
    chord_values: Sequence[float],
    chord_levels: Sequence[int],
    span_values: Sequence[float],
    span_levels: Sequence[int],
    quantity: str = "f",
) -> MeshErrorEstimate:
    """Combine a chordwise and a spanwise refinement into the error estimate of the base mesh.

    This is a pure function of the two refinement sequences. Both start
    at the base mesh, so their first values are the same. Each direction
    goes to :func:`order_and_extrapolation` (its three finest levels, and
    the base-mesh estimate of that direction). The two directions combine
    with the separable error model of the module notes: the Richardson
    value is ``R_c + R_s - f_0`` and the error estimate of the base mesh is
    the sum of the two directional estimates.

    Parameters
    ----------
    chord_values : sequence of float
        Quantity values [-] of the chordwise refinement, base mesh first
        (three or more levels).
    chord_levels : sequence of int
        Chordwise panel counts [-] of these levels, strictly increasing.
    span_values : sequence of float
        Quantity values [-] of the spanwise refinement, base mesh first
        (three or more levels).
    span_levels : sequence of int
        Spanwise panel counts [-] of these levels, strictly increasing.
    quantity : str
        Quantity name for the report.

    Returns
    -------
    MeshErrorEstimate
        The combined Richardson value and error estimate of the base mesh,
        with the convergence of each direction.

    Raises
    ------
    ValueError
        When a sequence has fewer than three levels, when the counts are
        not strictly increasing, or when the two sequences do not start at
        the same base value.
    """
    c_vals = [float(v) for v in chord_values]
    s_vals = [float(v) for v in span_values]
    if c_vals and s_vals and not math.isclose(c_vals[0], s_vals[0], rel_tol=_BASE_REL_TOL, abs_tol=1e-15):
        raise ValueError(
            f"The chordwise and the spanwise refinement must start at the same base mesh: base values "
            f"{c_vals[0]!r} and {s_vals[0]!r} differ."
        )
    chord = _direction_convergence(c_vals, chord_levels, quantity, "chordwise")
    span = _direction_convergence(s_vals, span_levels, quantity, "spanwise")
    f0 = c_vals[0]

    err = None
    if chord.base_abs_error is not None and span.base_abs_error is not None:
        err = float(chord.base_abs_error) + float(span.base_abs_error)
    f_ext = None
    if chord.f_extrapolated is not None and span.f_extrapolated is not None:
        f_ext = float(chord.f_extrapolated) + float(span.f_extrapolated) - f0
    ref = abs(f_ext) if f_ext is not None else abs(f0)
    rel = None if err is None else err / max(ref, 1e-300)
    if err is None:
        state = "no_estimate"
    elif chord.state == "monotonic" and span.state == "monotonic":
        state = "monotonic"
    else:
        state = "estimate"
    in_range = bool(chord.in_asymptotic_range and span.in_asymptotic_range)

    def describe(conv: QuantityConvergence, direction: str) -> str:
        text = f"{direction} refinement {conv.state}"
        if conv.observed_order is not None:
            text += f" (observed order {conv.observed_order:.3f})"
        return text

    note = f"{quantity}: {describe(chord, 'chordwise')}, {describe(span, 'spanwise')}."
    if not in_range:
        note += f" A direction is not in the asymptotic range (monotonic, order in [{ORDER_LOWER}, {ORDER_UPPER}])."
    if err is not None:
        note += (
            f" Error estimate of the given mesh (sum of the chordwise and the spanwise estimates): "
            f"{f0:.6g} +/- {err:.3g} ({rel:.2%})."
        )
    else:
        missing = "chordwise" if chord.base_abs_error is None else "spanwise"
        note += f" No error estimate of the given mesh: the {missing} refinement gives none."
    if f_ext is not None:
        note += f" Richardson value of the separable model: {f_ext:.6g}."
    return MeshErrorEstimate(
        quantity=str(quantity),
        base_value=f0,
        f_extrapolated=f_ext,
        base_abs_error=err,
        base_relative_error=rel,
        in_asymptotic_range=in_range,
        state=state,
        note=note,
        chordwise=chord,
        spanwise=span,
    )


def panel_mesh_convergence(
    aircraft: Aircraft | LiftingSurface,
    condition: FlightCondition | None,
    settings: SolverSettings,
    levels: int = 3,
    ratio: float = 1.5,
    quantities: Sequence[str] = ("CL", "CDi", "Cm"),
) -> ConvergenceReport:
    """Run a mesh convergence study of the 3D panel method with separate directions.

    The study solves the base mesh, a chordwise refinement (the chordwise
    count multiplied by *ratio* per level, at the base spanwise count) and
    a spanwise refinement (the spanwise count multiplied by *ratio* per
    level, at the base chordwise count), *levels* levels each with the base
    mesh as the first level. :func:`directional_estimate` gives, per
    quantity, the observed order, the Richardson value and the bar of the
    finest level of each direction, and the error estimate of the base
    mesh (the sum of the two directional estimates, see the module notes).

    Parameters
    ----------
    aircraft : Aircraft or LiftingSurface
        Case geometry. A single surface is wrapped into an aircraft.
    condition : FlightCondition or None
        Flight condition, the same for all levels. None uses the default
        condition (alpha 5 deg, V_inf 50 m/s, free air).
    settings : SolverSettings
        Solver settings of the base mesh, the mesh of the user. The
        spanwise count is ``settings.n_panels`` [-] per semi-span; the
        chordwise count is ``settings.n_chord`` [-] per surface, or the
        default of the panel solver when None. The solver type is set to
        the panel method on every level, and the levels do not run a study
        of their own.
    levels : int
        Number of mesh levels per direction, the base mesh included: 3 or
        4. The study solves ``1 + 2 (levels - 1)`` meshes.
    ratio : float
        Refinement ratio of each direction [-]. Must exceed 1.
    quantities : sequence of str
        Tracked quantities (any of ``"CL"``, ``"CDi"``, ``"CD_total"``,
        ``"CY"``, ``"Cl"``, ``"Cm"``, ``"Cn"``).

    Returns
    -------
    ConvergenceReport
        Per quantity the error estimate of the base mesh with the
        convergence of each direction, the levels, the solved meshes and
        the cost factor.
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
    meshes = [(spans[0], chords[0])]
    meshes += [(spans[0], nc) for nc in chords[1:]]
    meshes += [(ns, chords[0]) for ns in spans[1:]]

    base = settings.clone()
    base.solver_type = "panel3d"
    base.panel_estimate_mesh_error = False
    n_base = max(int(settings.n_panels), 1)

    results: list[SolverResult] = []
    for n_span, n_chord in meshes:
        level_settings = base.clone()
        level_settings.n_panels = n_span
        level_settings.n_chord = n_chord
        level_aircraft = _level_aircraft(ac, n_span, n_base)
        results.append(analyze(level_aircraft, condition=cond, settings=level_settings))

    totals = [int(np.size(res.details.get("cp", []))) for res in results]
    if any(n <= 0 for n in totals):
        totals = [2 * ns * nc for ns, nc in meshes]
    n_lv = int(levels)
    chord_idx = list(range(0, n_lv))
    span_idx = [0] + list(range(n_lv, 2 * n_lv - 1))

    estimates: dict[str, MeshErrorEstimate] = {}
    for name in names:
        series = [_quantity_value(res, name) for res in results]
        est = directional_estimate(
            [series[i] for i in chord_idx], chords, [series[i] for i in span_idx], spans, quantity=name,
        )
        est.chordwise.total_panels = [int(totals[i]) for i in chord_idx]
        est.spanwise.total_panels = [int(totals[i]) for i in span_idx]
        estimates[name] = est

    n0 = max(totals[0], 1)
    cost = float(sum((n / n0) ** 3 for n in totals))
    header = (
        f"Panel mesh convergence (chordwise and spanwise refinement, {n_lv} levels each, ratio "
        f"{float(ratio):.3f}): chord panels {chords} at {spans[0]} span panels, span panels {spans} at "
        f"{chords[0]} chord panels, cost factor {cost:.1f} relative to one base solve (dense LU work)."
    )
    notes = [header] + [est.note for est in estimates.values()]
    return ConvergenceReport(
        quantities=estimates,
        n_span_levels=[int(n) for n in spans],
        n_chord_levels=[int(n) for n in chords],
        meshes=[(int(a), int(b)) for a, b in meshes],
        total_panels=[int(n) for n in totals],
        levels=n_lv,
        ratio=float(ratio),
        cost_factor=cost,
        notes=notes,
        results=results,
    )
