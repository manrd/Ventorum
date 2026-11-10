# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Trim solver for longitudinal and lateral aircraft trim.

This module provides functions to calculate the angle of attack and control
deflections required to achieve a target lift coefficient with zero aerodynamic
moments.

Longitudinal trim solves for angle of attack and pitch control deflection to satisfy
CL = CL_target and Cm = 0.

Lateral trim solves for angle of attack, pitch control, roll control, and yaw control
deflections to satisfy CL = CL_target, Cm = 0, Cl = 0, and Cn = 0 at the specified
sideslip angle.

The solver uses Newton's method with a finite-difference Jacobian computed via central
differences with a step size of 1.0e-4 rad.

Notes
-----
The solves run on the device that the settings and the global device select
(the low-order methods do not need float64, so trim does not force the CPU). The result
states the device and the precision.
In float32 the round-off of the coefficients is about 1e-7 (measured on the GPU
pipelines against float64), so the convergence tolerances get the floor
``FLOAT32_TOLERANCE_FLOOR``; with the 1e-4 rad step the round-off error of the
Jacobian entries is then about 1e-3 per rad (about 1e-4 relative to a lift slope
of 5 per rad), which is enough for Newton's method.

When condition.h is set (ground effect), the height of Aircraft.ref_point above the
ground plane remains constant while the angle of attack changes. With the automatic
chordwise count of the vortex lattice, all solves of one pass of a trim in ground
effect use one count, so that the residual is continuous in the unknowns. The first
pass uses the automatic count of the start point. When the automatic count at the
trimmed state differs, a new pass starts from the trimmed state with that count (at
most ``GROUND_N_CHORD_MAX_PASSES`` passes), so that the result equals an analysis at
the trimmed state. When a later pass does not converge, the result is the trimmed state
of the pass before it, with a note. ``max_iterations`` of :func:`trim` is the limit of
the Newton steps in total over all passes.

References
----------
B. Etkin and L. D. Reid, Dynamics of Flight: Stability and Control, 3rd ed., Wiley, 1996.

J. E. Dennis and R. B. Schnabel, Numerical Methods for Unconstrained Optimization and
Nonlinear Equations, SIAM, 1996.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from ventorum.core.datatypes import Aircraft, FlightCondition, SolverResult, SolverSettings
from ventorum.core.errors import GroundStrikeError, ValidityError

# The documented failures of a solve during the iteration. Each one gives the
# status "not_converged". All other exceptions propagate to the caller.
_SOLVE_FAILURES = (GroundStrikeError, ValidityError)

# Singular-Jacobian limits: relative column norm and condition number.
_COLUMN_NORM_LIMIT = 1.0e-8
_CONDITION_LIMIT = 1.0e10

#: Smallest convergence tolerance [-] of CL and of the moments when a solve is in
#: float32: about ten times the float32 round-off of the coefficients.
FLOAT32_TOLERANCE_FLOOR = 1.0e-6


def _json_float(value: float) -> float | None:
    """Return ``float(value)``, or None when the value is not finite (strict JSON)."""
    v = float(value)
    return v if math.isfinite(v) else None


@dataclass(slots=True)
class TrimResult:
    """Result of an aircraft trim calculation.

    When a solve fails during the iteration (ground strike, lifting line too near
    the ground, or an inner solve that does not converge), the result gives the
    last point with a good solve and a note on the failed point. When no solve
    succeeded, the coefficients are NaN and ``result`` is None.

    Parameters
    ----------
    status : str
        Trim status: "trimmed", "not_converged", "alpha_limit", or "control_limit".
    converged : bool
        True if the solver reached all convergence tolerances.
    alpha_deg : float
        Angle of attack at the reported point [deg].
    deflections_deg : dict[str, float]
        Control surface deflections at the reported point [deg].
    CL : float
        Lift coefficient at the reported point [-]. NaN when no solve succeeded.
    CD : float
        Drag coefficient at the reported point [-]. NaN when no solve succeeded.
    drag_basis : str
        Basis for CD: "CD_total" if profile drag exists, else "CDi"; "none" when
        no solve succeeded.
    CY : float
        Side force coefficient at the reported point [-]. NaN when no solve succeeded.
    Cl : float
        Rolling moment coefficient in body axes at the reported point [-]. NaN when
        no solve succeeded.
    Cm : float
        Pitching moment coefficient in body axes at the reported point [-]. NaN when
        no solve succeeded.
    Cn : float
        Yawing moment coefficient in body axes at the reported point [-]. NaN when
        no solve succeeded.
    iterations : int
        Number of Newton steps done, the total over all passes (a trim in
        ground effect with the automatic chordwise count can have more than
        one pass, see :func:`trim`).
    residual_history : list[float]
        Largest absolute residual [-] at each evaluated point, from the start point,
        for all passes in their order. Each pass has its Newton steps plus one
        entries, so the list has ``iterations + len(pass_residual_histories)``
        entries for every status (``iterations + 1`` with one pass). The entry is
        NaN for a point where the solve failed.
    jacobian : list[list[float]]
        Jacobian matrix at the reported point [1/rad]. Rows correspond to residuals,
        columns correspond to unknowns. All entries are NaN when the Jacobian was
        not computed at the reported point.
    result : SolverResult or None
        Complete aerodynamic solution at the reported point. None when no solve
        succeeded.
    aircraft : Aircraft
        Clone of the input aircraft with the deflections of the reported point.
    notes : list[str]
        Informational notes and convergence diagnostics.
    device : str
        Device of the solves: "cpu", "gpu", "mixed", or "none" when no solve
        succeeded.
    precision : str
        Precision of the solves: "float64", "float32", "mixed", or "none".
    pass_residual_histories : list[list[float]]
        The residual history [-] of each pass, in order. ``residual_history``
        is these lists joined. A trim with one pass has one list, equal to
        ``residual_history``.
    """

    status: str
    converged: bool
    alpha_deg: float
    deflections_deg: dict[str, float]
    CL: float
    CD: float
    drag_basis: str
    CY: float
    Cl: float
    Cm: float
    Cn: float
    iterations: int
    residual_history: list[float] = field(default_factory=list)
    jacobian: list[list[float]] = field(default_factory=list)
    result: SolverResult | None = None
    aircraft: Aircraft = field(default_factory=Aircraft)
    notes: list[str] = field(default_factory=list)
    device: str = "none"
    precision: str = "none"
    pass_residual_histories: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary without result and aircraft objects.

        A value that is not finite (NaN) is given as None, so that the dictionary
        is strict JSON.
        """
        return {
            "status": str(self.status),
            "converged": bool(self.converged),
            "alpha_deg": _json_float(self.alpha_deg),
            "deflections_deg": {k: _json_float(v) for k, v in self.deflections_deg.items()},
            "CL": _json_float(self.CL),
            "CD": _json_float(self.CD),
            "drag_basis": str(self.drag_basis),
            "CY": _json_float(self.CY),
            "Cl": _json_float(self.Cl),
            "Cm": _json_float(self.Cm),
            "Cn": _json_float(self.Cn),
            "iterations": int(self.iterations),
            "residual_history": [_json_float(r) for r in self.residual_history],
            "jacobian": [[_json_float(elem) for elem in row] for row in self.jacobian],
            "notes": list(self.notes),
            "device": str(self.device),
            "precision": str(self.precision),
            "pass_residual_histories": [[_json_float(r) for r in h] for h in self.pass_residual_histories],
        }


#: Largest number of trim passes in ground effect with the automatic chordwise
#: count (see :func:`trim`).
GROUND_N_CHORD_MAX_PASSES = 3


def _automatic_ground_n_chord(
    aircraft: Aircraft,
    condition: FlightCondition,
    settings: SolverSettings | None,
    alpha_rad: float,
) -> int | None:
    """Return the automatic chordwise count of a trim point in ground effect.

    In ground effect the automatic chordwise count of the vortex lattice
    (:meth:`ventorum.solvers.lattice_base.LatticeSolver.resolve_n_chord`)
    is an integer that depends on the attitude: the angle of attack and the
    deflections. This function gives the count of :func:`ventorum.analyze`
    at the angle of attack *alpha_rad* with the current deflections of
    *aircraft*.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft with the deflections of the point.
    condition : FlightCondition
        Flight condition of the trim.
    settings : SolverSettings or None
        Settings of the trim. None is the default of :func:`ventorum.analyze`.
    alpha_rad : float
        Angle of attack of the point [rad].

    Returns
    -------
    int or None
        The automatic count, or None when the count is not automatic in
        ground effect: no ground (``condition.h`` None), a count in the
        settings, or a lifting-line solver.
    """
    from ventorum.solvers.factory import make_solver, resolve_solver_type
    from ventorum.utils.validation import validate_aircraft

    if condition.h is None:
        return None
    # The default settings of ventorum.analyze.
    st = SolverSettings(solver_type="auto", n_panels=80) if settings is None else settings
    if st.n_chord is not None:
        return None
    solver = make_solver(resolve_solver_type(st.solver_type))
    if solver.collocation != "vlm":
        return None
    # The same check as the solve; an invalid aircraft raises here.
    validate_aircraft(aircraft)
    fc = condition.clone()
    fc.alpha = float(alpha_rad)
    # A strike gives the floor count; the solve then reports the strike.
    return int(solver.resolve_n_chord(aircraft, st, fc, None, aircraft.moment_reference()))


def _with_n_chord(settings: SolverSettings | None, n_chord: int) -> SolverSettings:
    """Return a copy of the settings (None: the defaults of ``analyze``) with ``n_chord`` set."""
    st = SolverSettings(solver_type="auto", n_panels=80) if settings is None else settings.clone()
    st.n_chord = int(n_chord)
    return st


def _get_control_deflection(ac: Aircraft, name: str) -> float:
    """Return the current deflection [rad] of a named control surface."""
    for surf in ac.surfaces:
        for c in getattr(surf, "controls", []):
            if c.name == name:
                return float(c.deflection)
    return 0.0


def trim(
    aircraft: Aircraft,
    condition: FlightCondition,
    settings: SolverSettings | None = None,
    *,
    CL_target: float,
    pitch_control: str,
    roll_control: str | None = None,
    yaw_control: str | None = None,
    alpha_bounds_deg: tuple[float, float] = (-10.0, 20.0),
    max_iterations: int = 20,
    tol_CL: float = 1.0e-6,
    tol_moment: float = 1.0e-7,
) -> TrimResult:
    """Calculate angle of attack and control deflections for trimmed flight.

    Finds the angle of attack and control deflections that achieve the target
    lift coefficient with zero pitching moment (longitudinal trim), and optionally
    zero rolling and yawing moments at the given sideslip angle (lateral trim).

    Parameters
    ----------
    aircraft : Aircraft
        Input aircraft geometry. The input aircraft is never modified.
    condition : FlightCondition
        Flight condition providing free-stream velocity, sideslip angle, and ground
        height. When condition.h is set, height remains constant during trim.
    settings : SolverSettings or None
        Solver settings. If None, default settings are used.
    CL_target : float
        Target lift coefficient [-].
    pitch_control : str
        Name of the control surface used for pitching moment trim.
    roll_control : str or None
        Name of the control surface used for rolling moment trim. Must be provided
        together with yaw_control for lateral trim.
    yaw_control : str or None
        Name of the control surface used for yawing moment trim. Must be provided
        together with roll_control for lateral trim.
    alpha_bounds_deg : tuple[float, float]
        Lower and upper bounds for angle of attack [deg]. Default is (-10.0, 20.0).
        A start point outside the bounds (alpha, or a deflection outside
        [-30, 30] deg) moves to the nearest point in the bounds, and a note
        of the result says so.
    max_iterations : int
        Maximum number of Newton iterations, the total over all passes (see
        the notes). Default is 20.
    tol_CL : float
        Convergence tolerance on lift coefficient residual ``abs(CL - CL_target)`` [-].
        Default is 1.0e-6.
    tol_moment : float
        Convergence tolerance on moment residuals (``abs(Cm)``, ``abs(Cl)``, ``abs(Cn)``) [-].
        Default is 1.0e-7.

    Returns
    -------
    TrimResult
        Trim solution containing status, angles, deflections, aerodynamic totals,
        iteration history, Jacobian matrix, and trimmed aircraft clone.

    Raises
    ------
    ValueError
        Only for invalid input, before the iteration starts: unknown control name
        (the message lists the known names), non-finite CL_target, bounds in the
        wrong order, max_iterations < 1, one lateral control without the other,
        lateral controls that are not distinct, tolerances that are not positive
        and finite, invalid solver settings (for example an unknown solver_type),
        the Fourier solver (it does not accept a deflected control), or an
        invalid flight condition.

    Notes
    -----
    The start point is the angle of attack ``condition.alpha`` and the current
    deflections of the controls of the given aircraft.

    A positive deflection is a right-hand rotation of the flap about its hinge
    axis. On a horizontal surface, a positive deflection moves the trailing edge
    down. On a vertical fin with dihedral 90 deg, a positive deflection moves the
    trailing edge to +y. A trailing-edge-up elevator has a negative deflection.

    A target that cannot be reached does not raise an exception: it gives a status.
    The documented failures of a solve during the iteration (``GroundStrikeError``,
    ``ValidityError`` for a lifting line too near the ground, and an inner solve
    with ``converged=False``) give the status "not_converged" and a note. All
    other exceptions of the solver propagate to the caller.

    The solves run on the device that the settings and the global device select;
    the result states the device and the precision. When a solve is in float32,
    the tolerances are at least ``FLOAT32_TOLERANCE_FLOOR`` (1e-6), and a note
    says so. The function does not change the global device.

    When ``condition.h`` is set (ground effect), the height of ``Aircraft.ref_point``
    above the ground plane stays constant while the angle of attack changes.
    When the vortex lattice finds its chordwise count automatically
    (``settings.n_chord`` None), the count depends on the attitude near the
    ground. Then all solves of one pass of the trim (the Newton steps and the
    finite differences of the Jacobian) use one count: the first pass the
    automatic count of the start point. After a pass converges, trim finds
    the automatic count at the trimmed state. When it differs, a new pass
    starts from the trimmed state with that count. This repeats until the
    count does not change, at most ``GROUND_N_CHORD_MAX_PASSES`` (3) passes.
    The final result then equals :func:`ventorum.analyze` at the trimmed
    state with the automatic count. When the count still changes after the
    last pass, the result is the last pass and a note says so. When a later
    pass does not converge, the result is the trimmed state of the pass
    before it (status "trimmed"), and a note says that its count is not the
    automatic count of the trimmed state and gives that count.
    *max_iterations* is the limit of the Newton steps in total over all
    passes: a later pass can do only the steps that the passes before it did
    not use. A later pass that has no step left evaluates its start point
    only; when that point is not trimmed with the new count, the pass does
    not converge and the result is the trimmed state of the pass before it.
    ``iterations`` is the total over all passes, ``residual_history`` joins the histories of the passes, and
    ``pass_residual_histories`` gives the history of each pass; a note gives
    the number of passes and their counts. A count in ``settings.n_chord``
    is used as given.

    References
    ----------
    B. Etkin and L. D. Reid, Dynamics of Flight: Stability and Control, 3rd ed.,
    Wiley, 1996.

    J. E. Dennis and R. B. Schnabel, Numerical Methods for Unconstrained
    Optimization and Nonlinear Equations, SIAM, 1996.
    """
    from ventorum import analyze
    from ventorum.solvers.factory import resolve_solver_type
    from ventorum.utils.validation import validate_flight_condition, validate_solver_settings

    # Validate inputs
    if not math.isfinite(CL_target):
        raise ValueError(f"CL_target must be a finite number; got {CL_target}.")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be at least 1; got {max_iterations}.")
    if len(alpha_bounds_deg) != 2 or alpha_bounds_deg[0] >= alpha_bounds_deg[1]:
        raise ValueError(
            f"alpha_bounds_deg must be a tuple (min, max) with min < max; got {alpha_bounds_deg}."
        )
    if (roll_control is None) != (yaw_control is None):
        raise ValueError("Both roll_control and yaw_control must be provided together for lateral trim, or neither.")

    known_controls = aircraft.control_names()
    if pitch_control not in known_controls:
        raise ValueError(
            f"Unknown pitch control {pitch_control!r}; known control names: {known_controls}."
        )
    if roll_control is not None and roll_control not in known_controls:
        raise ValueError(
            f"Unknown roll control {roll_control!r}; known control names: {known_controls}."
        )
    if yaw_control is not None and yaw_control not in known_controls:
        raise ValueError(
            f"Unknown yaw control {yaw_control!r}; known control names: {known_controls}."
        )

    is_lateral = roll_control is not None and yaw_control is not None
    if is_lateral:
        ctrl_set = {pitch_control, roll_control, yaw_control}
        if len(ctrl_set) < 3:
            raise ValueError("pitch_control, roll_control, and yaw_control must be distinct control surfaces.")

    if tol_CL <= 0.0 or not math.isfinite(tol_CL):
        raise ValueError(f"tol_CL must be positive and finite; got {tol_CL}.")
    if tol_moment <= 0.0 or not math.isfinite(tol_moment):
        raise ValueError(f"tol_moment must be positive and finite; got {tol_moment}.")

    # Validate the condition and the settings before the iteration. An invalid
    # input then raises ValueError and does not become a "not_converged" status.
    validate_flight_condition(condition)
    if settings is not None:
        validate_solver_settings(settings)
        if resolve_solver_type(settings.solver_type) == "fourier":
            raise ValueError(
                "The Fourier solver does not accept a deflected control, so it cannot trim. "
                "Use solver_type 'vlm', 'linear' or 'nonlinear'."
            )

    ac_clone = aircraft.clone()
    unknown_names: list[str] = ["alpha", pitch_control]
    if is_lateral:
        assert roll_control is not None and yaw_control is not None
        unknown_names.extend([roll_control, yaw_control])

    n_vars = len(unknown_names)

    # Deflection bounds are [-30, 30] deg per control surface convention
    ctrl_min_rad = math.radians(-30.0)
    ctrl_max_rad = math.radians(30.0)
    alpha_min_rad = math.radians(alpha_bounds_deg[0])
    alpha_max_rad = math.radians(alpha_bounds_deg[1])

    lower_bounds = np.array(
        [alpha_min_rad, ctrl_min_rad] + ([ctrl_min_rad, ctrl_min_rad] if is_lateral else []),
        dtype=float,
    )
    upper_bounds = np.array(
        [alpha_max_rad, ctrl_max_rad] + ([ctrl_max_rad, ctrl_max_rad] if is_lateral else []),
        dtype=float,
    )

    # Initial state vector
    x_init: list[float] = [float(condition.alpha), _get_control_deflection(ac_clone, pitch_control)]
    if is_lateral:
        assert roll_control is not None and yaw_control is not None
        x_init.extend([
            _get_control_deflection(ac_clone, roll_control),
            _get_control_deflection(ac_clone, yaw_control),
        ])

    x_start = np.array(x_init, dtype=float)
    x = np.clip(x_start, lower_bounds, upper_bounds)
    nan_jacobian = np.full((n_vars, n_vars), np.nan, dtype=float)

    def _describe(x_vec: np.ndarray) -> str:
        """Return the values of the unknowns of a point [deg] as text."""
        return ", ".join(
            f"{name} = {math.degrees(float(val)):.4f} deg" for name, val in zip(unknown_names, x_vec)
        )

    start_notes: list[str] = []
    if np.any(x != x_start):
        start_notes.append(
            f"The start point ({_describe(x_start)}) is outside the bounds; the iteration starts "
            f"from the nearest point in the bounds ({_describe(x)})."
        )

    def _set_deflections(x_vec: np.ndarray) -> None:
        """Set the deflections of the unknowns of a point on the aircraft clone."""
        ac_clone.set_deflection(pitch_control, float(x_vec[1]))
        if is_lateral:
            assert roll_control is not None and yaw_control is not None
            ac_clone.set_deflection(roll_control, float(x_vec[2]))
            ac_clone.set_deflection(yaw_control, float(x_vec[3]))

    # In ground effect with the automatic chordwise count, each pass of the
    # trim uses one count, so that the residual is continuous in the unknowns.
    _set_deflections(x)
    n_chord_auto = _automatic_ground_n_chord(ac_clone, condition, settings, float(x[0]))
    # The settings of the solves of the current pass.
    run_settings = settings if n_chord_auto is None else _with_n_chord(settings, n_chord_auto)

    devices: set[str] = set()
    precisions: set[str] = set()

    def _evaluate(x_vec: np.ndarray) -> tuple[SolverResult | None, np.ndarray | None, str | None]:
        fc = condition.clone()
        fc.alpha = float(x_vec[0])
        _set_deflections(x_vec)

        # Only the documented solver failures become a status. All other
        # exceptions (programming errors, KeyboardInterrupt) propagate.
        try:
            sol = analyze(ac_clone, fc, settings=run_settings)
        except _SOLVE_FAILURES as exc:
            return None, None, f"The solve failed at ({_describe(x_vec)}): {exc}"

        if not sol.converged:
            return None, None, f"The inner solve did not converge at ({_describe(x_vec)})."
        devices.add(str(sol.details.get("device", "cpu")))
        precisions.add(str(sol.details.get("precision", "float64")))

        tot = sol.totals
        if not is_lateral:
            res_vec = np.array([tot.CL - CL_target, tot.Cm], dtype=float)
        else:
            res_vec = np.array([tot.CL - CL_target, tot.Cm, tot.Cl, tot.Cn], dtype=float)

        if not np.all(np.isfinite(res_vec)):
            return None, None, f"The solve gave non-finite residual values at ({_describe(x_vec)})."
        return sol, res_vec, None

    def _compute_jacobian(x_vec: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        h_step = 1.0e-4
        J_mat = np.zeros((n_vars, n_vars), dtype=float)
        for col in range(n_vars):
            x_plus = x_vec.copy()
            x_minus = x_vec.copy()
            if x_vec[col] + h_step > upper_bounds[col]:
                x_plus[col] = upper_bounds[col]
                x_minus[col] = upper_bounds[col] - 2.0 * h_step
            elif x_vec[col] - h_step < lower_bounds[col]:
                x_plus[col] = lower_bounds[col] + 2.0 * h_step
                x_minus[col] = lower_bounds[col]
            else:
                x_plus[col] = x_vec[col] + h_step
                x_minus[col] = x_vec[col] - h_step

            _, R_plus, err_plus = _evaluate(x_plus)
            if err_plus is not None or R_plus is None:
                return None, f"Jacobian calculation failed for +step on {unknown_names[col]}: {err_plus}"

            _, R_minus, err_minus = _evaluate(x_minus)
            if err_minus is not None or R_minus is None:
                return None, f"Jacobian calculation failed for -step on {unknown_names[col]}: {err_minus}"

            J_mat[:, col] = (R_plus - R_minus) / (2.0 * h_step)

        if not np.all(np.isfinite(J_mat)):
            return None, "Jacobian contains non-finite entries."
        return J_mat, None

    def _singular_reason(J_mat: np.ndarray) -> str | None:
        """Return the reason why the Jacobian is singular, or None."""
        J_norm = float(np.linalg.norm(J_mat))
        col_norms = np.linalg.norm(J_mat, axis=0)
        no_effect = [
            (unknown_names[j], float(col_norms[j]))
            for j in range(n_vars)
            if col_norms[j] <= _COLUMN_NORM_LIMIT * J_norm
        ]
        if no_effect:
            text = ", ".join(f"{name!r} (column norm {cn:.2e})" for name, cn in no_effect)
            return (
                f"Singular Jacobian: the unknown {text} has no effect on the residuals "
                f"(Jacobian norm {J_norm:.2e})."
            )
        cond_J = float(np.linalg.cond(J_mat))
        if not math.isfinite(cond_J) or cond_J > _CONDITION_LIMIT:
            return (
                f"Singular Jacobian: condition number is {cond_J:.2e}, "
                f"above the limit {_CONDITION_LIMIT:.0e}."
            )
        return None

    def _build_result(
        status_str: str,
        converged_flag: bool,
        x_final: np.ndarray,
        last_sol: SolverResult | None,
        n_iters: int,
        history: list[float],
        jac_matrix: np.ndarray,
        notes_list: list[str],
    ) -> TrimResult:
        _set_deflections(x_final)
        final_x[:] = [float(v) for v in x_final]

        # Collect deflections from all surfaces
        defs_deg: dict[str, float] = {}
        for s in ac_clone.surfaces:
            for c in getattr(s, "controls", []):
                defs_deg[c.name] = float(np.degrees(c.deflection))

        if last_sol is None:
            # No solve succeeded: report NaN, never false zeros.
            nan = float("nan")
            CL = CD = CY = Cl = Cm = Cn = nan
            basis = "none"
        else:
            tot = last_sol.totals
            if tot.CD_total is not None and tot.CDp is not None:
                CD = float(tot.CD_total)
                basis = "CD_total"
            else:
                CD = float(tot.CDi)
                basis = "CDi"
            CL, CY = float(tot.CL), float(tot.CY)
            Cl, Cm, Cn = float(tot.Cl), float(tot.Cm), float(tot.Cn)

        return TrimResult(
            status=status_str,
            converged=converged_flag,
            alpha_deg=float(np.degrees(x_final[0])),
            deflections_deg=defs_deg,
            CL=CL,
            CD=CD,
            drag_basis=basis,
            CY=CY,
            Cl=Cl,
            Cm=Cm,
            Cn=Cn,
            iterations=int(n_iters),
            residual_history=list(history),
            jacobian=[[float(val) for val in row] for row in jac_matrix],
            result=last_sol,
            aircraft=ac_clone,
            notes=list(notes_list),
            device=_set_text(devices),
            precision=_set_text(precisions),
            pass_residual_histories=[list(history)],
        )

    def _set_text(values: set[str]) -> str:
        """Return the one value of *values*, "mixed" for more, or "none" for no value."""
        return "none" if not values else (next(iter(values)) if len(values) == 1 else "mixed")

    def _join_passes(res: TrimResult, passes: list[TrimResult]) -> TrimResult:
        """Give *res* the iteration count and the residual histories of all *passes*."""
        histories = [list(p.residual_history) for p in passes]
        res.iterations = int(sum(p.iterations for p in passes))
        res.residual_history = [r for h in histories for r in h]
        res.pass_residual_histories = histories
        res.device = _set_text(devices)
        res.precision = _set_text(precisions)
        return res

    def _newton(x0: np.ndarray, budget: int) -> TrimResult:
        """Run the Newton iteration from *x0* with the settings ``run_settings``.

        *budget* is the number of Newton steps that this pass can do: the
        part of *max_iterations* that the passes before it did not use.
        """
        x = x0.copy()
        notes: list[str] = list(start_notes)
        residual_history: list[float] = []
        outward_counts = np.zeros(n_vars, dtype=int)
        # The last point with a good solve, its solution and its Jacobian.
        good_x: np.ndarray | None = None
        good_sol: SolverResult | None = None
        good_J = nan_jacobian

        # iter_idx is the number of Newton steps done before this point. Every
        # stop returns iterations = iter_idx, thus
        # len(residual_history) == iterations + 1.
        for iter_idx in range(budget + 1):
            sol, R, err = _evaluate(x)
            if err is not None or sol is None or R is None:
                residual_history.append(float("nan"))
                notes.append(err if err is not None else "Evaluation failed.")
                if good_x is None:
                    notes.append("No solve succeeded: the coefficients are NaN.")
                    return _build_result(
                        "not_converged", False, x, None, iter_idx, residual_history, nan_jacobian, notes
                    )
                notes.append(f"The result gives the last point with a good solve ({_describe(good_x)}).")
                return _build_result(
                    "not_converged", False, good_x, good_sol, iter_idx, residual_history, good_J, notes
                )

            good_x, good_sol, good_J = x.copy(), sol, nan_jacobian
            max_res = float(np.max(np.abs(R)))
            residual_history.append(max_res)

            # Check convergence. In float32 the tolerances have a floor at
            # the round-off of the coefficients.
            tl_CL, tl_m = tol_CL, tol_moment
            if "float32" in precisions:
                tl_CL = max(tol_CL, FLOAT32_TOLERANCE_FLOOR)
                tl_m = max(tol_moment, FLOAT32_TOLERANCE_FLOOR)
                if (tl_CL, tl_m) != (tol_CL, tol_moment) and not any("float32" in n for n in notes):
                    notes.append(f"The solves are in float32: the tolerances are tol_CL = {tl_CL:.1e} and "
                                 f"tol_moment = {tl_m:.1e} (floor {FLOAT32_TOLERANCE_FLOOR:.0e}, about five "
                                 "times the float32 round-off of the coefficients).")
            if abs(R[0]) <= tl_CL and all(abs(r) <= tl_m for r in R[1:]):
                J_final, err_final = _compute_jacobian(x)
                if J_final is not None:
                    good_J = J_final
                else:
                    notes.append(f"The Jacobian at the trim point is not available: {err_final}")
                return _build_result(
                    "trimmed", True, x, sol, iter_idx, residual_history, good_J, notes
                )

            if iter_idx == budget:
                notes.append(f"Maximum iterations ({max_iterations}, the total over all passes) reached "
                             "without convergence.")
                J_final, err_final = _compute_jacobian(x)
                if J_final is not None:
                    good_J = J_final
                else:
                    notes.append(f"The Jacobian at the last point is not available: {err_final}")
                return _build_result(
                    "not_converged", False, x, sol, iter_idx, residual_history, good_J, notes
                )

            # Compute Jacobian
            J, err_J = _compute_jacobian(x)
            if err_J is not None or J is None:
                notes.append(err_J if err_J is not None else "Jacobian computation failed.")
                return _build_result(
                    "not_converged", False, x, sol, iter_idx, residual_history, nan_jacobian, notes
                )
            good_J = J

            # Check singularity: an unknown with no effect, or a large condition number
            reason = _singular_reason(J)
            if reason is not None:
                notes.append(reason)
                return _build_result(
                    "not_converged", False, x, sol, iter_idx, residual_history, good_J, notes
                )
            try:
                dx = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError as exc:
                notes.append(f"Singular Jacobian: {exc}.")
                return _build_result(
                    "not_converged", False, x, sol, iter_idx, residual_history, good_J, notes
                )

            if not np.all(np.isfinite(dx)):
                notes.append("Newton step contains non-finite entries.")
                return _build_result(
                    "not_converged", False, x, sol, iter_idx, residual_history, good_J, notes
                )

            # Bound limit stop check
            limit_stop = False
            stop_status = ""
            for j in range(n_vars):
                at_lower = (x[j] - lower_bounds[j]) <= 1.0e-10
                points_outward_lower = at_lower and (dx[j] < -1.0e-10)

                at_upper = (upper_bounds[j] - x[j]) <= 1.0e-10
                points_outward_upper = at_upper and (dx[j] > 1.0e-10)

                if points_outward_lower or points_outward_upper:
                    outward_counts[j] += 1
                else:
                    outward_counts[j] = 0

                if outward_counts[j] >= 2 and not limit_stop:
                    limit_stop = True
                    if j == 0:
                        stop_status = "alpha_limit"
                        bnd_deg = alpha_bounds_deg[0] if points_outward_lower else alpha_bounds_deg[1]
                        notes.append(f"Angle of attack reached bound of {bnd_deg:.1f} deg.")
                    else:
                        stop_status = "control_limit"
                        ctrl_name = unknown_names[j]
                        bnd_deg = -30.0 if points_outward_lower else 30.0
                        notes.append(
                            f"Control surface {ctrl_name!r} reached deflection bound of {bnd_deg:.1f} deg."
                        )

            if limit_stop:
                return _build_result(
                    stop_status, False, x, sol, iter_idx, residual_history, good_J, notes
                )

            # Step limit: scale down if largest change exceeds 5 deg
            max_step = math.radians(5.0)
            largest_change = float(np.max(np.abs(dx)))
            if largest_change > max_step:
                dx = dx * (max_step / largest_change)

            # Cut at bounds
            x = np.clip(x + dx, lower_bounds, upper_bounds)

        # Not reached: the loop returns when iter_idx == budget.
        raise AssertionError("The trim loop ended without a result.")

    # The unknowns [rad] of the reported point of the last pass.
    final_x: list[float] = [float(v) for v in x]

    if n_chord_auto is None:
        return _newton(x, max_iterations)

    # Ground effect with the automatic chordwise count: a pass uses one count.
    # When the automatic count at the trimmed state differs, a new pass starts
    # from the trimmed state with that count. The final result then equals an
    # analysis at the trimmed state with the automatic count.
    pass_counts = [n_chord_auto]
    passes: list[TrimResult] = []
    # The trimmed pass before the current one and its unknowns [rad].
    trimmed_res: TrimResult | None = None
    trimmed_x: np.ndarray | None = None
    history_text = ("The iterations are the total over all passes; residual_history joins the residual "
                    "histories of the passes, and pass_residual_histories gives each pass.")
    while True:
        # max_iterations is a total limit: a pass gets the steps that are left.
        res = _newton(x, max_iterations - sum(p.iterations for p in passes))
        passes.append(res)
        if res.status != "trimmed":
            if trimmed_res is None:
                res.notes.append(
                    f"Ground effect: the solves of pass {len(pass_counts)} of the trim use n_chord = "
                    f"{pass_counts[-1]}, the automatic chordwise count of the start point of the pass. "
                    "The trim did not converge, so the count was not checked at the end point."
                )
                return _join_passes(res, passes)
            # A later pass failed: return the trimmed pass before it. The
            # aircraft clone is shared, so its deflections go back to that state.
            assert trimmed_x is not None
            _set_deflections(trimmed_x)
            final_x[:] = [float(v) for v in trimmed_x]
            k = len(pass_counts)
            for note in res.notes:
                if note not in start_notes:
                    trimmed_res.notes.append(f"Pass {k}: {note}")
            trimmed_res.notes.append(
                f"Ground effect: pass {k} of the trim, with n_chord = {pass_counts[-1]} (the automatic "
                f"chordwise count at the trimmed state of pass {k - 1}), did not converge (status "
                f"{res.status!r}). The result is the trimmed state of pass {k - 1}, with n_chord = "
                f"{pass_counts[-2]}. This is not the automatic chordwise count of the trimmed state, which is "
                f"{pass_counts[-1]}, so the result differs from an analysis there with the automatic count. "
                f"Passes: {k}, counts {', '.join(str(n) for n in pass_counts)}. {history_text} Set "
                "settings.n_chord to select one count."
            )
            return _join_passes(trimmed_res, passes)
        x = np.array(final_x, dtype=float)
        _set_deflections(x)
        n_next = _automatic_ground_n_chord(ac_clone, condition, settings, float(x[0]))
        assert n_next is not None
        counts_text = ", ".join(str(n) for n in pass_counts)
        if n_next == pass_counts[-1]:
            res.notes.append(
                f"Ground effect: all solves of the last pass of the trim use n_chord = {n_next}, the "
                "automatic chordwise count at the trimmed state, so the result equals an analysis there "
                f"(passes: {len(pass_counts)}, counts {counts_text}). {history_text} Set settings.n_chord "
                "to select a different count."
            )
            return _join_passes(res, passes)
        if len(pass_counts) >= GROUND_N_CHORD_MAX_PASSES:
            res.notes.append(
                f"Ground effect: the automatic chordwise count changed after each of {len(pass_counts)} "
                f"passes of the trim (counts {counts_text}); at the trimmed state of the last pass it is "
                f"{n_next}. The result is the last pass, with n_chord = {pass_counts[-1]}, so it differs "
                f"from an analysis at the trimmed state with the automatic count. {history_text} Set "
                "settings.n_chord to select one count."
            )
            return _join_passes(res, passes)
        trimmed_res, trimmed_x = res, x.copy()
        pass_counts.append(n_next)
        run_settings = _with_n_chord(settings, n_next)
