# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Compute numerical error bars (layer 1) from three spanwise mesh levels.

The bar covers the spanwise discretisation error of one force or moment
coefficient with the Grid Convergence Index (GCI) of Roache (1998). The
observed order of convergence comes from the fixed-point iteration of
Celik et al. (2008) for non-constant refinement ratios. The convergence
states (monotonic, oscillatory, divergent) follow Oberkampf and Roy
(2010). The oscillatory rule, the order limits 0.5 and 4.0, and the
"roundoff" and "not_converged_order" rules are project decisions
(conservative choices), not results of these sources.

Index 1 is always the FINE mesh: ``f1`` and ``N1`` are the fine-level
value and panel count, ``f2`` and ``N2`` the medium level, ``f3`` and
``N3`` the coarse level. All half-widths are absolute, never relative,
so a coefficient near zero needs no division.

The bar covers the spanwise discretisation error only. The chordwise
error is not in the bar. The bar is about the mesh, not about the model.

References
----------
P. J. Roache, Verification and Validation in Computational Science and
Engineering, Hermosa Publishers, 1998.
I. B. Celik, U. Ghia, P. J. Roache, C. J. Freitas, H. Coleman and
P. E. Raad, "Procedure for estimation and reporting of uncertainty due
to discretization in CFD applications", Journal of Fluids Engineering
130(7), 078001, 2008.
W. L. Oberkampf and C. J. Roy, Verification and Validation in
Scientific Computing, Cambridge University Press, 2010.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    SolverResult,
    SolverSettings,
)

#: Method string stored in every layer contribution of this module.
METHOD = "GCI, Roache (1998); observed order, Celik et al. (2008)"

#: Allowed values of ``ErrorBar.status``. This task sets only
#: ``"numerical_only"``; later layers set the other states.
STATUS_VALUES = (
    "statistically_calibrated",
    "model_form_estimated",
    "numerical_only",
    "outside_envelope",
)

#: Coefficients with an error bar, in output order.
COEFFICIENT_NAMES = ("CL", "CD", "CY", "Cl", "Cm", "Cn")

#: Note present in the notes of every result of this module.
LAYER_NOTE = (
    "Layer 1 only: spanwise discretisation error. "
    "It is not an error against experiment."
)

#: Note of a bar without an interval (divergent convergence).
DIVERGENT_NOTE = (
    "no numerical error bar: the coefficient diverges with mesh refinement"
)

#: Note of a bar whose observed-order iteration did not converge.
NOT_CONVERGED_ORDER_NOTE = (
    "the observed order of convergence was not found; "
    "the bar uses a conservative rule"
)

#: Smallest fine-level panel count. Below it the coarse level is too coarse.
MIN_FINE_PANELS = 16

#: Trusted range of the observed order. Outside it the safety factor is 3.0
#: and the order is clipped to this range (project decision).
ORDER_LOWER = 0.5
ORDER_UPPER = 4.0


@dataclass(slots=True)
class LayerContribution:
    """Uncertainty contribution of one layer to one coefficient.

    Attributes
    ----------
    layer : str
        Layer name, here always ``"numerical"``.
    half_width : float or None
        Half-width of the interval in coefficient units, or None when the
        layer gives no interval.
    method : str
        Method that produced the contribution.
    state : str
        Convergence state: ``"roundoff"``, ``"monotonic"``,
        ``"oscillatory"``, ``"divergent"`` or ``"not_converged_order"``.
    observed_order : float or None
        Observed order of convergence, when found.
    safety_factor : float or None
        Safety factor used in the half-width, when one applies.
    f_extrapolated : float or None
        Richardson-extrapolated value (information only; the interval is
        centred on the fine-level value, not on this value).
    levels : list[int]
        Spanwise panel counts [N1, N2, N3], index 1 is the fine mesh.
    values : list[float]
        Coefficient values [f1, f2, f3], index 1 is the fine mesh.
    """

    layer: str = "numerical"
    half_width: float | None = None
    method: str = METHOD
    state: str = "monotonic"
    observed_order: float | None = None
    safety_factor: float | None = None
    f_extrapolated: float | None = None
    levels: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the contribution as a dict with JSON types only."""
        half = self.half_width
        order = self.observed_order
        safety = self.safety_factor
        f_ext = self.f_extrapolated
        return {
            "layer": str(self.layer),
            "half_width": None if half is None else float(half),
            "method": str(self.method),
            "state": str(self.state),
            "observed_order": None if order is None else float(order),
            "safety_factor": None if safety is None else float(safety),
            "f_extrapolated": None if f_ext is None else float(f_ext),
            "levels": [int(n) for n in self.levels],
            "values": [float(v) for v in self.values],
        }


@dataclass(slots=True)
class ErrorBar:
    """Error bar of one coefficient, with one entry per layer.

    Attributes
    ----------
    name : str
        Coefficient name (``"CL"``, ``"CD"``, ``"CY"``, ``"Cl"``,
        ``"Cm"`` or ``"Cn"``).
    value : float
        Coefficient value on the fine mesh (f1).
    interval_low : float or None
        Lower bound of the interval, or None without an interval.
    interval_high : float or None
        Upper bound of the interval, or None without an interval.
    status : str
        One of ``STATUS_VALUES``. This task sets only ``"numerical_only"``.
    coverage : float or None
        Stated probability of coverage. Always None in this task (only a
        calibrated bar has a coverage).
    layers : dict[str, LayerContribution]
        Contribution of each layer, keyed by layer name.
    notes : list[str]
        Human-readable notes about this bar.
    """

    name: str = "CL"
    value: float = 0.0
    interval_low: float | None = None
    interval_high: float | None = None
    status: str = "numerical_only"
    coverage: float | None = None
    layers: dict[str, LayerContribution] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Check that the status is one of the allowed values."""
        if self.status not in STATUS_VALUES:
            raise ValueError(
                f"Unknown error-bar status {self.status!r}. "
                f"Use one of {list(STATUS_VALUES)}."
            )

    @property
    def state(self) -> str | None:
        """Convergence state of the numerical layer, or None without it."""
        layer = self.layers.get("numerical")
        return layer.state if layer is not None else None

    @property
    def half_width(self) -> float | None:
        """Half-width of the numerical layer, or None without an interval."""
        layer = self.layers.get("numerical")
        return layer.half_width if layer is not None else None

    def to_dict(self) -> dict[str, Any]:
        """Return the bar as a dict with JSON types only."""
        low = self.interval_low
        high = self.interval_high
        cover = self.coverage
        return {
            "name": str(self.name),
            "value": float(self.value),
            "interval_low": None if low is None else float(low),
            "interval_high": None if high is None else float(high),
            "status": str(self.status),
            "coverage": None if cover is None else float(cover),
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "notes": [str(n) for n in self.notes],
        }


@dataclass(slots=True)
class ErrorBarResult:
    """Numerical error bars of the six force and moment coefficients.

    Attributes
    ----------
    bars : dict[str, ErrorBar]
        Bars keyed by coefficient (``"CL"``, ``"CD"``, ``"CY"``,
        ``"Cl"``, ``"Cm"``, ``"Cn"``).
    axes : str
        Moment axis system of the Cl, Cm and Cn bars (``"body"``,
        ``"stability"`` or ``"wind"``).
    drag_basis : str
        Drag coefficient used for the CD bar.
    solver : str
        Solver that produced the three mesh levels.
    levels : list[int]
        Spanwise panel counts [N1, N2, N3], index 1 is the fine mesh.
    results : list[SolverResult]
        Solver results of the fine, medium and coarse levels. They are
        not part of ``to_dict``.
    notes : list[str]
        Human-readable notes about this result.
    """

    bars: dict[str, ErrorBar] = field(default_factory=dict)
    axes: str = "body"
    drag_basis: str = ""
    solver: str = ""
    levels: list[int] = field(default_factory=list)
    results: list[SolverResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a dict with JSON types only."""
        return {
            "bars": {k: v.to_dict() for k, v in self.bars.items()},
            "axes": str(self.axes),
            "drag_basis": str(self.drag_basis),
            "solver": str(self.solver),
            "levels": [int(n) for n in self.levels],
            "notes": [str(n) for n in self.notes],
        }

    def summary(self) -> str:
        """Return a short Markdown table with name, value, low, high, state."""
        lines = [
            "| name | value | low | high | state |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name in COEFFICIENT_NAMES:
            bar = self.bars.get(name)
            if bar is None:
                continue
            low = "n/a" if bar.interval_low is None else f"{bar.interval_low:.6f}"
            high = "n/a" if bar.interval_high is None else f"{bar.interval_high:.6f}"
            lines.append(
                f"| {bar.name} | {bar.value:.6f} | {low} | {high} | {bar.state} |"
            )
        return "\n".join(lines)


def mesh_levels(n_fine: int) -> list[int]:
    """Return the three spanwise panel counts [N1, N2, N3].

    Parameters
    ----------
    n_fine : int
        Fine-level (user) panel count per semi-span (N1).

    Returns
    -------
    list[int]
        ``[N1, N2, N3]`` with ``N2 = round(N1 / sqrt(2))`` and
        ``N3 = round(N2 / sqrt(2))``. Index 1 is the fine mesh.

    Raises
    ------
    ValueError
        If ``n_fine`` is below 16 (the coarse level would be too coarse).
    """
    n1 = int(n_fine)
    if n1 < MIN_FINE_PANELS:
        raise ValueError(
            f"n_fine={n1} is too coarse for a three-level error bar; "
            f"use at least {MIN_FINE_PANELS} spanwise panels."
        )
    root2 = math.sqrt(2.0)
    n2 = int(round(n1 / root2))
    n3 = int(round(n2 / root2))
    return [n1, n2, n3]


def refinement_ratios(levels: list[int]) -> tuple[float, float]:
    """Return the refinement ratios (r21, r32) from real panel counts.

    Parameters
    ----------
    levels : list[int]
        Spanwise panel counts [N1, N2, N3], index 1 is the fine mesh.

    Returns
    -------
    tuple[float, float]
        ``(N1 / N2, N2 / N3)``, with the representative cell size
        ``h_i = 1 / N_i`` (spanwise only).
    """
    n1, n2, n3 = (int(levels[0]), int(levels[1]), int(levels[2]))
    return (n1 / n2, n2 / n3)


def observed_order(
    f1: float,
    f2: float,
    f3: float,
    r21: float,
    r32: float,
) -> float | None:
    """Return the observed order of convergence, or None when not found.

    Use the fixed-point iteration of Celik et al. (2008) for non-constant
    refinement ratios, with index 1 = fine::

        eps21 = f2 - f1,  eps32 = f3 - f2,  s = sign(eps32 / eps21)
        p = | ln|eps32 / eps21| + q(p) | / ln(r21)
        q(p) = ln((r21^p - s) / (r32^p - s))

    The iteration starts from ``p = ln|eps32 / eps21| / ln(r21)`` (q = 0),
    runs at most 100 times and stops when the change of p is below 1e-10.

    Parameters
    ----------
    f1 : float
        Fine-level value.
    f2 : float
        Medium-level value.
    f3 : float
        Coarse-level value.
    r21 : float
        Refinement ratio of fine to medium (N1 / N2).
    r32 : float
        Refinement ratio of medium to coarse (N2 / N3).

    Returns
    -------
    float or None
        The observed order, or None when the iteration does not converge
        or gives a non-finite value.
    """
    eps21 = float(f2) - float(f1)
    eps32 = float(f3) - float(f2)
    r21 = float(r21)
    r32 = float(r32)
    if not (math.isfinite(eps21) and math.isfinite(eps32)):
        return None
    if not (math.isfinite(r21) and math.isfinite(r32)):
        return None
    if r21 <= 1.0 or r32 <= 1.0:
        return None
    if eps21 == 0.0 or eps32 == 0.0:
        return None
    ratio = eps32 / eps21
    if not math.isfinite(ratio) or ratio == 0.0:
        return None
    s = 1.0 if ratio > 0.0 else -1.0
    log_r21 = math.log(r21)
    if not math.isfinite(log_r21) or log_r21 <= 0.0:
        return None
    log_abs = math.log(abs(ratio))
    if not math.isfinite(log_abs):
        return None
    p = log_abs / log_r21
    if not math.isfinite(p):
        return None
    for _ in range(100):
        try:
            a1 = r21**p - s
            a2 = r32**p - s
        except OverflowError:
            return None
        if not (math.isfinite(a1) and math.isfinite(a2)):
            return None
        if a1 <= 0.0 or a2 <= 0.0:
            return None
        q = math.log(a1 / a2)
        if not math.isfinite(q):
            return None
        p_new = abs(log_abs + q) / log_r21
        if not math.isfinite(p_new):
            return None
        if abs(p_new - p) < 1e-10:
            return float(p_new)
        p = p_new
    return None


def richardson_extrapolate(f1: float, f2: float, r21: float, p_used: float) -> float:
    """Return the Richardson-extrapolated value (information only).

    Parameters
    ----------
    f1 : float
        Fine-level value.
    f2 : float
        Medium-level value.
    r21 : float
        Refinement ratio of fine to medium (N1 / N2).
    p_used : float
        Order used in the extrapolation.

    Returns
    -------
    float
        ``(r21^p_used * f1 - f2) / (r21^p_used - 1)``.
    """
    denom = float(r21) ** float(p_used) - 1.0
    return (float(r21) ** float(p_used) * float(f1) - float(f2)) / denom


def coefficient_error_bar(
    name: str,
    values: list[float],
    levels: list[int],
) -> ErrorBar:
    """Build the layer-1 error bar of one coefficient from three levels.

    Classify the convergence with ``R = eps21 / eps32`` where
    ``eps21 = f2 - f1`` and ``eps32 = f3 - f2`` (index 1 is the fine
    mesh):

    * ``"roundoff"``: the changes are at round-off level. The half-width
      is ``3 * max(|eps21|, |eps32|)``.
    * ``"monotonic"`` (``0 <= R < 1``): the observed order comes from
      :func:`observed_order`. With ``0.5 <= p <= 4.0`` the safety factor
      is 1.25, else it is 3.0 with the order clipped to [0.5, 4.0]. The
      half-width is ``Fs * |eps21| / (r21^p_used - 1)``.
    * ``"oscillatory"`` (``R < 0``): the half-width is
      ``3.0 * 0.5 * (max - min)`` of the three values (project decision).
    * ``"divergent"`` (``R >= 1``): the change grows with refinement, so
      there is no interval.
    * ``"not_converged_order"``: monotonic values whose observed-order
      iteration fails. The half-width is ``3.0 * |eps21|`` (a simple
      conservative project rule).

    A bar with an interval is centred on the fine-level value f1 (not on
    the extrapolated value).

    Parameters
    ----------
    name : str
        Coefficient name.
    values : list[float]
        Coefficient values [f1, f2, f3], index 1 is the fine mesh.
    levels : list[int]
        Spanwise panel counts [N1, N2, N3], index 1 is the fine mesh.

    Returns
    -------
    ErrorBar
        Bar with status ``"numerical_only"`` and coverage None.
    """
    f1, f2, f3 = (float(values[0]), float(values[1]), float(values[2]))
    n1, n2, n3 = (int(levels[0]), int(levels[1]), int(levels[2]))
    r21, r32 = refinement_ratios([n1, n2, n3])
    notes: list[str] = []

    finite = math.isfinite(f1) and math.isfinite(f2) and math.isfinite(f3)
    fine_ratios = math.isfinite(r21) and math.isfinite(r32) and r21 > 1.0 and r32 > 1.0
    if not finite or not fine_ratios:
        layer = LayerContribution(
            half_width=None,
            state="divergent",
            levels=[n1, n2, n3],
            values=[f1, f2, f3],
        )
        notes.append(DIVERGENT_NOTE)
        if not finite:
            notes.append("a mesh-level value is not finite")
        return ErrorBar(
            name=name,
            value=f1,
            status="numerical_only",
            layers={"numerical": layer},
            notes=notes,
        )

    eps21 = f2 - f1
    eps32 = f3 - f2
    scale = max(1.0, abs(f1), abs(f2), abs(f3))
    spread = max(abs(eps21), abs(eps32))

    if spread <= 1e-12 * scale:
        half = 3.0 * spread
        layer = LayerContribution(
            half_width=half,
            state="roundoff",
            levels=[n1, n2, n3],
            values=[f1, f2, f3],
        )
        return ErrorBar(
            name=name,
            value=f1,
            interval_low=f1 - half,
            interval_high=f1 + half,
            status="numerical_only",
            layers={"numerical": layer},
            notes=notes,
        )

    if eps32 == 0.0:
        if eps21 > 0.0:
            big_r = math.inf
        elif eps21 < 0.0:
            big_r = -math.inf
        else:
            big_r = math.nan
    else:
        big_r = eps21 / eps32

    if math.isnan(big_r):
        layer = LayerContribution(
            half_width=0.0,
            state="roundoff",
            levels=[n1, n2, n3],
            values=[f1, f2, f3],
        )
        return ErrorBar(
            name=name,
            value=f1,
            interval_low=f1,
            interval_high=f1,
            status="numerical_only",
            layers={"numerical": layer},
            notes=notes,
        )

    if big_r < 0.0:
        half = 1.5 * (max(f1, f2, f3) - min(f1, f2, f3))
        layer = LayerContribution(
            half_width=half,
            state="oscillatory",
            levels=[n1, n2, n3],
            values=[f1, f2, f3],
        )
        return ErrorBar(
            name=name,
            value=f1,
            interval_low=f1 - half,
            interval_high=f1 + half,
            status="numerical_only",
            layers={"numerical": layer},
            notes=notes,
        )

    if big_r >= 1.0:
        layer = LayerContribution(
            half_width=None,
            state="divergent",
            levels=[n1, n2, n3],
            values=[f1, f2, f3],
        )
        notes.append(DIVERGENT_NOTE)
        return ErrorBar(
            name=name,
            value=f1,
            status="numerical_only",
            layers={"numerical": layer},
            notes=notes,
        )

    order = observed_order(f1, f2, f3, r21, r32)
    if order is None or not math.isfinite(order):
        half = 3.0 * abs(eps21)
        layer = LayerContribution(
            half_width=half,
            state="not_converged_order",
            levels=[n1, n2, n3],
            values=[f1, f2, f3],
        )
        notes.append(NOT_CONVERGED_ORDER_NOTE)
        return ErrorBar(
            name=name,
            value=f1,
            interval_low=f1 - half,
            interval_high=f1 + half,
            status="numerical_only",
            layers={"numerical": layer},
            notes=notes,
        )

    if ORDER_LOWER <= order <= ORDER_UPPER:
        safety = 1.25
        p_used = order
    else:
        safety = 3.0
        p_used = min(max(order, ORDER_LOWER), ORDER_UPPER)
    half = safety * abs(eps21) / (r21**p_used - 1.0)
    f_ext = richardson_extrapolate(f1, f2, r21, p_used)
    layer = LayerContribution(
        half_width=half,
        state="monotonic",
        observed_order=order,
        safety_factor=safety,
        f_extrapolated=f_ext,
        levels=[n1, n2, n3],
        values=[f1, f2, f3],
    )
    return ErrorBar(
        name=name,
        value=f1,
        interval_low=f1 - half,
        interval_high=f1 + half,
        status="numerical_only",
        layers={"numerical": layer},
        notes=notes,
    )


def _surface_clip_note(
    base_settings: SolverSettings,
    results: list[SolverResult],
) -> str | None:
    """Note when a surface was not refined with the others, or None.

    With proportional panels, ``min_panels`` can hold a small surface at
    the same panel count on two mesh levels. The check reads the strip
    count of each surface from the lattice of each level. Levels without
    a lattice (Fourier solver) give no note.
    """
    try:
        lattices = [res.details.get("lattice") for res in results]
        if any(lat is None for lat in lattices):
            return None
        n_surf = len(lattices[0].surfaces)
        if any(len(lat.surfaces) != n_surf for lat in lattices):
            return None
        names = [sw.surface_name for sw in results[0].spanwise]
        for index in range(n_surf):
            counts = [len(lat.surfaces[index].strips) for lat in lattices]
            if not (counts[0] > counts[1] > counts[2]):
                label = names[index] if index < len(names) else f"surface {index}"
                floor = int(base_settings.min_panels)
                return (
                    f"proportional panels with min_panels={floor}: the {label!r} "
                    f"surface has the same panel count at two mesh levels, "
                    "so it was not refined with the others"
                )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None
    return None


def numerical_error_bars(
    geometry: Aircraft | LiftingSurface,
    condition: FlightCondition | None = None,
    settings: SolverSettings | None = None,
    *,
    axes: str = "body",
    solver: str | None = None,
    use_symmetry: bool | None = None,
) -> ErrorBarResult:
    """Compute layer-1 numerical error bars from three spanwise meshes.

    Solve the case on three spanwise mesh levels and build one GCI error
    bar (see :func:`coefficient_error_bar`) for each of the six force
    and moment coefficients (CL, CD, CY, Cl, Cm, Cn).

    The user mesh is the FINE level: ``N1 = settings.n_panels`` (with
    ``settings=None`` this is the ``SolverSettings`` default, 80). The
    coarser levels are ``N2 = round(N1 / sqrt(2))`` and
    ``N3 = round(N2 / sqrt(2))``. Only the spanwise panel count changes;
    the chordwise count, the spacing rule and every other setting stay
    as the user gave them. An automatic chordwise count is resolved one
    time on the fine level and then used for all three levels, because
    in ground effect it can depend on the mesh. A surface with its own
    ``n_panels`` is scaled in the same way. The bar covers the spanwise
    discretisation error only.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Case geometry. A single surface is wrapped into an aircraft.
    condition : FlightCondition or None
        Flight condition, the same for the three levels. When None, the
        default condition is used (alpha 5 deg, V_inf 50 m/s, free air).
    settings : SolverSettings or None
        Solver settings. When None, the defaults are used (the VLM with
        80 spanwise panels per semi-span).
    axes : str
        Moment axis system of the Cl, Cm and Cn bars: ``"body"``,
        ``"stability"`` or ``"wind"``. ``"all"`` is refused.
    solver : str or None
        Solver name (``"vlm"``, ``"linear"``, ``"nonlinear"`` or
        ``"fourier"``). When None, the solver of *settings* is used.
    use_symmetry : bool or None
        Use the y = 0 symmetry plane when the geometry and flow allow
        it. When None, the value of *settings* is used.

    Returns
    -------
    ErrorBarResult
        Bars with status ``"numerical_only"`` and coverage None, the
        three solver results (fine, medium, coarse) and the notes.

    Raises
    ------
    ValueError
        If *axes* is not a known axis system, or if the fine-level panel
        count is below 16 (the coarse level would be too coarse).
    """
    from ventorum import analyze
    from ventorum.geometry.mesh_convergence import _level_aircraft
    from ventorum.solvers.factory import resolve_solver_type

    if axes not in ("body", "stability", "wind"):
        raise ValueError(
            f"'axes' must be one of ['body', 'stability', 'wind'], got {axes!r}."
        )
    aircraft = (
        Aircraft(name="SingleWing", surfaces=[geometry])
        if isinstance(geometry, LiftingSurface)
        else geometry
    )
    cond = condition.clone() if condition is not None else FlightCondition()
    base = settings.clone() if settings is not None else SolverSettings()
    if solver is not None:
        base.solver_type = solver
    if use_symmetry is not None:
        base.use_symmetry = bool(use_symmetry)

    n1 = int(base.n_panels)
    if n1 < MIN_FINE_PANELS:
        raise ValueError(
            f"n_panels={n1} is too coarse for a three-level error bar; "
            f"use at least {MIN_FINE_PANELS} spanwise panels."
        )
    levels = mesh_levels(n1)
    canonical = resolve_solver_type(base.solver_type)

    frozen_chord: int | None = None
    if canonical == "vlm" and base.n_chord is None:
        from ventorum.solvers.horseshoe import HorseshoeSolver

        fine_aircraft = _level_aircraft(aircraft, levels[0], n1)
        probe = base.clone()
        probe.n_panels = levels[0]
        frozen_chord = int(
            HorseshoeSolver().resolve_n_chord(
                fine_aircraft, probe, cond, None, fine_aircraft.moment_reference()
            )
        )

    level_results: list[SolverResult] = []
    for n_level in levels:
        level_aircraft = _level_aircraft(aircraft, n_level, n1)
        level_settings = base.clone()
        level_settings.n_panels = n_level
        if frozen_chord is not None:
            level_settings.n_chord = frozen_chord
        level_results.append(analyze(level_aircraft, condition=cond, settings=level_settings))

    fine = level_results[0]
    use_total = fine.totals.CDp is not None
    if use_total:
        drag_basis = "CD_total (induced + profile)"

        def drag_of(res: SolverResult) -> float:
            total = res.totals.CD_total
            return float(total) if total is not None else float(res.totals.CDi)
    else:
        drag_basis = "CDi only: the airfoils have no profile drag"

        def drag_of(res: SolverResult) -> float:
            return float(res.totals.CDi)

    series: dict[str, list[float]] = {name: [] for name in COEFFICIENT_NAMES}
    for res in level_results:
        moments = res.moments(axes)
        series["CL"].append(float(res.totals.CL))
        series["CD"].append(drag_of(res))
        series["CY"].append(float(res.totals.CY))
        series["Cl"].append(float(moments["Cl"]))
        series["Cm"].append(float(moments["Cm"]))
        series["Cn"].append(float(moments["Cn"]))

    bars = {
        name: coefficient_error_bar(name, series[name], levels)
        for name in COEFFICIENT_NAMES
    }
    notes = [LAYER_NOTE]
    clip = _surface_clip_note(base, level_results) if base.proportional_panels else None
    if clip is not None:
        notes.append(clip)
    return ErrorBarResult(
        bars=bars,
        axes=axes,
        drag_basis=drag_basis,
        solver=fine.solver_type,
        levels=levels,
        results=level_results,
        notes=notes,
    )
