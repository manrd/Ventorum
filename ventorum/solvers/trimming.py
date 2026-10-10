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
Every solve in this module runs on the CPU in float64 precision. Finite differences
are not suitable for float32 precision. The trim function temporarily sets the global
device to "cpu" and restores the previous device setting upon completion. For this
reason, trim must not run concurrently with other calls that alter the global device.

When condition.h is set (ground effect), the height of Aircraft.ref_point above the
ground plane remains constant while the angle of attack changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

import ventorum.gpu
from ventorum.core.datatypes import Aircraft, FlightCondition, SolverResult, SolverSettings


@dataclass(slots=True)
class TrimResult:
    """Result of an aircraft trim calculation.

    Parameters
    ----------
    status : str
        Trim status: "trimmed", "not_converged", "alpha_limit", or "control_limit".
    converged : bool
        True if the solver reached all convergence tolerances.
    alpha_deg : float
        Trimmed angle of attack [deg].
    deflections_deg : dict[str, float]
        Trimmed control surface deflections [deg].
    CL : float
        Lift coefficient at the final point [-].
    CD : float
        Drag coefficient at the final point [-].
    drag_basis : str
        Basis for CD: "CD_total" if profile drag exists, else "CDi".
    CY : float
        Side force coefficient at the final point [-].
    Cl : float
        Rolling moment coefficient in body axes at the final point [-].
    Cm : float
        Pitching moment coefficient in body axes at the final point [-].
    Cn : float
        Yawing moment coefficient in body axes at the final point [-].
    iterations : int
        Number of Newton iterations performed.
    residual_history : list[float]
        Largest absolute residual at each iteration.
    jacobian : list[list[float]]
        Jacobian matrix at the final point [1/rad]. Rows correspond to residuals,
        columns correspond to unknowns.
    result : SolverResult
        Complete aerodynamic solution at the final point.
    aircraft : Aircraft
        Clone of the input aircraft with trimmed control deflections applied.
    notes : list[str]
        Informational notes and convergence diagnostics.
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
    result: SolverResult = field(default_factory=SolverResult)
    aircraft: Aircraft = field(default_factory=Aircraft)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary without result and aircraft objects."""
        return {
            "status": str(self.status),
            "converged": bool(self.converged),
            "alpha_deg": float(self.alpha_deg),
            "deflections_deg": {k: float(v) for k, v in self.deflections_deg.items()},
            "CL": float(self.CL),
            "CD": float(self.CD),
            "drag_basis": str(self.drag_basis),
            "CY": float(self.CY),
            "Cl": float(self.Cl),
            "Cm": float(self.Cm),
            "Cn": float(self.Cn),
            "iterations": int(self.iterations),
            "residual_history": [float(r) for r in self.residual_history],
            "jacobian": [[float(elem) for elem in row] for row in self.jacobian],
            "notes": list(self.notes),
        }


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
    max_iterations : int
        Maximum number of Newton iterations. Default is 20.
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
        If inputs are invalid (unknown control name, non-finite CL_target, inverted
        bounds, max_iterations < 1, or mismatched lateral controls).
    """
    from ventorum import analyze

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

    x = np.clip(np.array(x_init, dtype=float), lower_bounds, upper_bounds)

    old_device = ventorum.gpu.get_device()
    try:
        ventorum.gpu.set_device("cpu")

        def _evaluate(x_vec: np.ndarray) -> tuple[SolverResult | None, np.ndarray | None, str | None]:
            fc = condition.clone()
            fc.alpha = float(x_vec[0])
            ac_clone.set_deflection(pitch_control, float(x_vec[1]))
            if is_lateral:
                assert roll_control is not None and yaw_control is not None
                ac_clone.set_deflection(roll_control, float(x_vec[2]))
                ac_clone.set_deflection(yaw_control, float(x_vec[3]))

            try:
                sol = analyze(ac_clone, fc, settings=settings)
            except Exception as exc:  # noqa: BLE001
                return None, None, f"Solver evaluation failed: {exc}"

            tot = sol.totals
            if not is_lateral:
                res_vec = np.array([tot.CL - CL_target, tot.Cm], dtype=float)
            else:
                res_vec = np.array([tot.CL - CL_target, tot.Cm, tot.Cl, tot.Cn], dtype=float)

            if not np.all(np.isfinite(res_vec)):
                return None, None, "Solver produced non-finite residual values."
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
            if last_sol is None:
                last_sol = SolverResult()

            last_sol.details["device"] = "cpu"
            last_sol.details["precision"] = "float64"

            ac_clone.set_deflection(pitch_control, float(x_final[1]))
            if is_lateral:
                assert roll_control is not None and yaw_control is not None
                ac_clone.set_deflection(roll_control, float(x_final[2]))
                ac_clone.set_deflection(yaw_control, float(x_final[3]))

            # Collect deflections from all surfaces
            defs_deg: dict[str, float] = {}
            for s in ac_clone.surfaces:
                for c in getattr(s, "controls", []):
                    defs_deg[c.name] = float(np.degrees(c.deflection))

            tot = last_sol.totals
            if tot.CD_total is not None and tot.CDp is not None:
                cd_val = float(tot.CD_total)
                basis = "CD_total"
            else:
                cd_val = float(tot.CDi)
                basis = "CDi"

            return TrimResult(
                status=status_str,
                converged=converged_flag,
                alpha_deg=float(np.degrees(x_final[0])),
                deflections_deg=defs_deg,
                CL=float(tot.CL),
                CD=cd_val,
                drag_basis=basis,
                CY=float(tot.CY),
                Cl=float(tot.Cl),
                Cm=float(tot.Cm),
                Cn=float(tot.Cn),
                iterations=int(n_iters),
                residual_history=list(history),
                jacobian=[[float(val) for val in row] for row in jac_matrix],
                result=last_sol,
                aircraft=ac_clone,
                notes=list(notes_list),
            )

        notes: list[str] = []
        residual_history: list[float] = []
        outward_counts = np.zeros(n_vars, dtype=int)
        last_J = np.zeros((n_vars, n_vars), dtype=float)
        res_current: SolverResult | None = None

        for iter_idx in range(max_iterations + 1):
            res_current, R, err = _evaluate(x)
            if err is not None or res_current is None or R is None:
                notes.append(err if err is not None else "Evaluation failed.")
                return _build_result(
                    "not_converged", False, x, res_current, iter_idx, residual_history, last_J, notes
                )

            max_res = float(np.max(np.abs(R)))
            residual_history.append(max_res)

            # Check convergence
            if abs(R[0]) <= tol_CL and all(abs(r) <= tol_moment for r in R[1:]):
                J_final, _ = _compute_jacobian(x)
                if J_final is not None:
                    last_J = J_final
                return _build_result(
                    "trimmed", True, x, res_current, iter_idx, residual_history, last_J, notes
                )

            if iter_idx == max_iterations:
                notes.append(f"Maximum iterations ({max_iterations}) reached without convergence.")
                J_final, _ = _compute_jacobian(x)
                if J_final is not None:
                    last_J = J_final
                return _build_result(
                    "not_converged", False, x, res_current, iter_idx, residual_history, last_J, notes
                )

            # Compute Jacobian
            J, err_J = _compute_jacobian(x)
            if err_J is not None or J is None:
                notes.append(err_J if err_J is not None else "Jacobian computation failed.")
                return _build_result(
                    "not_converged", False, x, res_current, iter_idx, residual_history, last_J, notes
                )
            last_J = J

            # Check singularity
            try:
                cond_J = np.linalg.cond(J)
                if cond_J > 1.0e15:
                    notes.append(f"Singular Jacobian: condition number is {cond_J:.2e}.")
                    return _build_result(
                        "not_converged", False, x, res_current, iter_idx, residual_history, last_J, notes
                    )
                dx = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError as exc:
                notes.append(f"Singular Jacobian: {exc}.")
                return _build_result(
                    "not_converged", False, x, res_current, iter_idx, residual_history, last_J, notes
                )

            if not np.all(np.isfinite(dx)):
                notes.append("Newton step contains non-finite entries.")
                return _build_result(
                    "not_converged", False, x, res_current, iter_idx, residual_history, last_J, notes
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
                    stop_status, False, x, res_current, iter_idx + 1, residual_history, last_J, notes
                )

            # Step limit: scale down if largest change exceeds 5 deg
            max_step = math.radians(5.0)
            largest_change = float(np.max(np.abs(dx)))
            if largest_change > max_step:
                dx = dx * (max_step / largest_change)

            # Cut at bounds
            x = np.clip(x + dx, lower_bounds, upper_bounds)

        # Fallback if loop exits
        return _build_result(
            "not_converged", False, x, res_current, max_iterations, residual_history, last_J, notes
        )

    finally:
        ventorum.gpu.set_device(old_device)
