# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Static stability derivatives and lift-coefficient targets for reference cases.

The validation runner uses these functions to compute the Ventorum values of
the derivative quantities of a reference case (for example ``Cl_beta``), to
find the angle of attack where Ventorum gives a lift coefficient, and to fit
the slope of a quantity against the lift coefficient.

Method:

* Derivatives are central differences: alpha +/- 0.5 deg, beta +/- 1 deg,
  as the agent tool ``ventorum_stability_derivatives``. All cases of one
  point use one mesh (in ground effect the automatic chordwise count of the
  case with the smallest gap is used for all cases).
* Moment derivatives are first taken in body axes. They are then expressed
  in the requested axes of the reference condition (alpha_0, beta_0), held
  fixed while alpha and beta change, with the reference span and chord
  (:func:`ventorum.core.axes.transform_moments`). CL, CD and CY are relative
  to the free stream in every axis system.
* The angle of attack for a lift coefficient comes from an alpha sweep (one
  mesh) and linear interpolation inside the bracket of the sweep. The
  interpolation is repeated inside the bracket (regula falsi) until the
  lift coefficient agrees with the target. A target outside the CL range
  of the sweep is "not reached": it is never extrapolated.
* The slope against CL is the least-squares straight line through the
  quantity at equally spaced lift coefficients.

References
----------
B. Etkin and L. D. Reid, Dynamics of Flight: Stability and Control, 3rd ed.,
Wiley, 1996 (static stability derivatives and axis systems).

B. L. Stevens, F. L. Lewis and E. N. Johnson, Aircraft Control and
Simulation, 3rd ed., Wiley, 2016, chapter 2 (axis transformations).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

#: Coefficients that a derivative quantity can differentiate.
DERIVATIVE_COEFFICIENTS = ("CL", "CD", "CY", "Cl", "Cm", "Cn")
#: Variables of a derivative quantity.
DERIVATIVE_VARIABLES = ("alpha", "beta")
#: Units of the stored values of a derivative quantity.
DERIVATIVE_UNITS = ("1/deg", "1/rad")
#: Axis systems of moment derivatives.
DERIVATIVE_AXES = ("body", "stability", "wind")
#: Default axis system of moment derivatives (as NACA stability-tunnel data).
DEFAULT_DERIVATIVE_AXES = "stability"
#: Central-difference steps [deg].
STEP_DEG = {"alpha": 0.5, "beta": 1.0}
#: Number of equally spaced lift coefficients of a slope fit.
SLOPE_POINTS = 5
#: Default angles of attack [deg] of the sweep that finds a lift coefficient.
CL_SWEEP_ALPHA_DEG = tuple(float(a) for a in np.arange(-10.0, 20.0 + 1e-9, 2.0))
#: Lift-coefficient tolerance of the search for a target CL.
CL_TOLERANCE = 1e-10
#: Largest number of interpolation steps inside the bracket.
CL_MAX_STEPS = 30

_MOMENTS = ("Cl", "Cm", "Cn")


def split_derivative(quantity: str) -> tuple[str, str] | None:
    """Split a derivative quantity ``<C>_<v>`` into coefficient and variable.

    Parameters
    ----------
    quantity : str
        Quantity name, for example ``"Cl_beta"``.

    Returns
    -------
    tuple of str or None
        ``(coefficient, variable)`` for a name with one underscore, else
        None. The parts are not checked against the allowed lists.
    """
    if not isinstance(quantity, str) or quantity.count("_") != 1:
        return None
    coefficient, variable = quantity.split("_")
    return coefficient, variable


def is_derivative(quantity: str) -> bool:
    """Return True if *quantity* is an allowed derivative quantity ``<C>_<v>``.

    Parameters
    ----------
    quantity : str
        Quantity name.

    Returns
    -------
    bool
        True when the coefficient is in :data:`DERIVATIVE_COEFFICIENTS` and
        the variable is in :data:`DERIVATIVE_VARIABLES`.
    """
    parts = split_derivative(quantity)
    return parts is not None and parts[0] in DERIVATIVE_COEFFICIENTS and parts[1] in DERIVATIVE_VARIABLES


def per_rad_to_unit(value_per_rad: float, unit: str) -> float:
    """Convert a derivative per radian to the unit *unit*.

    Parameters
    ----------
    value_per_rad : float
        Derivative [1/rad].
    unit : str
        ``"1/rad"`` or ``"1/deg"``.

    Returns
    -------
    float
        Derivative in *unit*.

    Raises
    ------
    ValueError
        If *unit* is not a known unit.
    """
    if unit == "1/rad":
        return float(value_per_rad)
    if unit == "1/deg":
        return float(value_per_rad) * math.pi / 180.0
    raise ValueError(f"'unit' must be one of {list(DERIVATIVE_UNITS)}, got {unit!r}.")


def coefficient(totals: Any, name: str) -> tuple[float, str]:
    """Read one coefficient from integrated totals (CD falls back to CDi).

    Parameters
    ----------
    totals : object
        Integrated totals of a solve (``SolverResult.totals``).
    name : str
        Coefficient name (``CL``, ``CD``, ``CDi``, ``CY``, ``Cl``, ``Cm``,
        ``Cn``). Moments are in body axes.

    Returns
    -------
    tuple
        ``(value, note)``. The note is ``"CDi (no profile drag)"`` when CD
        falls back to CDi, else an empty string.
    """
    if name == "CD":
        if totals.CD_total is not None:
            return float(totals.CD_total), ""
        return float(totals.CDi), "CDi (no profile drag)"
    return float(getattr(totals, name)), ""


def one_mesh_settings(aircraft: Any, settings: Any, conditions: Sequence[Any]) -> Any:
    """Return one set of solver settings for all *conditions* (one mesh).

    In ground effect the automatic chordwise count of the vortex-lattice
    solver depends on the attitude. The cases of one finite difference or
    of one sweep must use the same mesh, so the largest automatic count of
    the cases is used for all of them. Otherwise (count given, not a
    lattice solver, or no ground) *settings* is returned unchanged.

    Parameters
    ----------
    aircraft : Aircraft
        The aircraft (its reference point is the moment reference).
    settings : SolverSettings
        Solver settings.
    conditions : sequence of FlightCondition
        Conditions of the cases.

    Returns
    -------
    SolverSettings
        *settings*, or a copy with a fixed ``n_chord``.
    """
    from ventorum.solvers.factory import resolve_solver_type

    if settings.n_chord is not None or resolve_solver_type(settings.solver_type) != "vlm":
        return settings
    if all(fc.h is None for fc in conditions):
        return settings
    from ventorum.solvers.horseshoe import VortexLatticeSolver

    solver = VortexLatticeSolver()
    ref_point = aircraft.moment_reference()
    count = max(solver.resolve_n_chord(aircraft, settings, fc, None, ref_point) for fc in conditions)
    out = settings.clone()
    out.n_chord = int(count)
    return out


def central_difference(
    aircraft: Any,
    quantity: str,
    alpha_deg: float,
    beta_deg: float,
    make_condition: Callable[[float, float], Any],
    settings: Any,
    axes: str = DEFAULT_DERIVATIVE_AXES,
) -> float:
    """Compute one static derivative by central differences.

    Parameters
    ----------
    aircraft : Aircraft
        The aircraft. Its reference point is the moment reference point.
    quantity : str
        Derivative quantity ``<C>_<v>``, for example ``"Cl_beta"``.
    alpha_deg : float
        Angle of attack of the point [deg].
    beta_deg : float
        Sideslip angle of the point [deg], positive for wind from the right.
    make_condition : callable
        ``make_condition(alpha_deg, beta_deg)`` returns the
        ``FlightCondition`` of one case (speed, density and height).
    settings : SolverSettings
        Solver settings. All cases use one mesh (:func:`one_mesh_settings`).
    axes : str, optional
        Axes of moment derivatives: ``"body"``, ``"stability"`` (default)
        or ``"wind"``, of the reference condition held fixed.

    Returns
    -------
    float
        Derivative [1/rad].

    Raises
    ------
    ValueError
        If *quantity* is not an allowed derivative quantity or *axes* is not
        a known axis system.
    """
    import ventorum as vt
    from ventorum.core.axes import transform_moments

    if not is_derivative(quantity):
        raise ValueError(f"'quantity' must be a derivative quantity <C>_<v>, got {quantity!r}.")
    if axes not in DERIVATIVE_AXES:
        raise ValueError(f"'axes' must be one of {list(DERIVATIVE_AXES)}, got {axes!r}.")
    name, variable = split_derivative(quantity)
    step = STEP_DEG[variable]
    if variable == "alpha":
        cases = [(alpha_deg + step, beta_deg), (alpha_deg - step, beta_deg)]
    else:
        cases = [(alpha_deg, beta_deg + step), (alpha_deg, beta_deg - step)]
    conditions = [make_condition(a, b) for a, b in cases]
    case_settings = one_mesh_settings(aircraft, settings, conditions)
    plus, minus = (vt.analyze(aircraft.clone(), condition=fc, settings=case_settings) for fc in conditions)
    width = math.radians(2.0 * step)
    if name not in _MOMENTS:
        return (coefficient(plus.totals, name)[0] - coefficient(minus.totals, name)[0]) / width
    d_body = [(getattr(plus.totals, m) - getattr(minus.totals, m)) / width for m in _MOMENTS]
    if axes == "body":
        return float(d_body[_MOMENTS.index(name)])
    rotated = transform_moments(
        *d_body,
        math.radians(alpha_deg),
        math.radians(beta_deg),
        axes,
        b_ref=plus.b_ref,
        c_ref=plus.c_ref,
    )
    return float(rotated[name])


def alpha_for_cl(
    aircraft: Any,
    cl_target: float,
    beta_deg: float,
    make_condition: Callable[[float, float], Any],
    settings: Any,
    alpha_grid_deg: Sequence[float] = CL_SWEEP_ALPHA_DEG,
) -> tuple[float | None, tuple[float, float], Any]:
    """Find the angle of attack where Ventorum gives a lift coefficient.

    An alpha sweep (one mesh for all points) gives CL(alpha). The first
    bracket of the sweep that holds the target is refined by linear
    interpolation (regula falsi) until CL agrees with the target within
    :data:`CL_TOLERANCE`. A target outside the CL range of the sweep is
    not reached and is never extrapolated.

    Parameters
    ----------
    aircraft : Aircraft
        The aircraft.
    cl_target : float
        Target lift coefficient [-].
    beta_deg : float
        Sideslip angle [deg].
    make_condition : callable
        ``make_condition(alpha_deg, beta_deg)`` returns the
        ``FlightCondition`` of one case.
    settings : SolverSettings
        Solver settings.
    alpha_grid_deg : sequence of float, optional
        Angles of attack of the sweep [deg], in increasing order.

    Returns
    -------
    tuple
        ``(alpha_deg, (cl_min, cl_max), sweep_settings)``: the angle of
        attack [deg] (None when the target is not reached), the CL range of
        the solved sweep points, and the one-mesh settings of the sweep.
    """
    import ventorum as vt

    grid = [float(a) for a in alpha_grid_deg]
    sweep_settings = one_mesh_settings(aircraft, settings, [make_condition(a, beta_deg) for a in grid])

    def lift(alpha: float) -> float:
        cond = make_condition(alpha, beta_deg)
        return float(vt.analyze(aircraft.clone(), condition=cond, settings=sweep_settings).totals.CL)

    points = []
    for alpha in grid:
        try:
            points.append((alpha, lift(alpha)))
        except Exception:  # noqa: BLE001 - a failed sweep point (for example a ground strike) is skipped
            continue
    if not points:
        return None, (math.nan, math.nan), sweep_settings
    cls = [cl for _, cl in points]
    cl_range = (min(cls), max(cls))
    target = float(cl_target)
    for alpha, cl in points:
        if cl == target:
            return alpha, cl_range, sweep_settings
    brackets = [
        (lower, upper) for lower, upper in zip(points[:-1], points[1:])
        if (lower[1] - target) * (upper[1] - target) < 0.0
    ]
    if not brackets:
        return None, cl_range, sweep_settings
    (a_lo, c_lo), (a_hi, c_hi) = brackets[0]
    alpha = a_lo
    for _ in range(CL_MAX_STEPS):
        alpha = a_lo + (target - c_lo) * (a_hi - a_lo) / (c_hi - c_lo)
        cl = lift(alpha)
        if abs(cl - target) <= CL_TOLERANCE:
            break
        if (c_lo - target) * (cl - target) < 0.0:
            a_hi, c_hi = alpha, cl
        else:
            a_lo, c_lo = alpha, cl
    return alpha, cl_range, sweep_settings


def slope_fit(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Fit a straight line by least squares.

    Parameters
    ----------
    x : sequence of float
        Abscissas (for example lift coefficients [-]).
    y : sequence of float
        Ordinates in the unit of the quantity (same length, at least 2 values).

    Returns
    -------
    tuple of float
        ``(slope, largest_residual)``: the slope dy/dx and the largest
        absolute residual of the fitted line, in the unit of *y*.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    residual = float(np.max(np.abs(ys - (slope * xs + intercept))))
    return float(slope), residual
