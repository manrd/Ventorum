# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Static stability derivatives: force and moment derivatives with alpha and beta.

:func:`static_derivatives` solves the base condition and two conditions for
each of the two flow angles alpha and beta with the angle changed by plus
and minus a step. Central differences give the derivatives of CL, CD, CY,
Cl, Cm and Cn. All five cases are solved as one batch on one lattice, so the
differences contain no mesh change.

References
----------
* B. Etkin and L. D. Reid, Dynamics of Flight: Stability and Control,
  3rd ed., Wiley, 1996: static stability derivatives and neutral point.
* J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., Cambridge
  University Press, 2001.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverResult, SolverSettings

__all__ = ["StaticDerivativesResult", "static_derivatives"]

_VARIABLES = ("alpha", "beta")
_COEFFS = ("CL", "CD", "CY", "Cl", "Cm", "Cn")


def _json_float(value: float | None) -> float | None:
    """Return ``float(value)``, or None when the value is None or not finite."""
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


@dataclass
class StaticDerivativesResult:
    """Result of :func:`static_derivatives`.

    Attributes
    ----------
    derivatives : dict[str, float]
        Derivative of each coefficient [1/rad], keyed
        ``"<coefficient>_<variable>"``, for example ``"CL_alpha"``
        and ``"Cm_alpha"``.
    matrix : numpy.ndarray
        The derivatives as a (6, 2) array: rows CL, CD, CY, Cl, Cm, Cn;
        columns alpha, beta.
    axes : str
        Axes of the moments: ``"body"``, ``"stability"`` or ``"wind"`` of
        the base condition, held fixed.
    steps : dict[str, float]
        Steps of the differences: ``alpha_deg`` and ``beta_deg`` [deg].
    drag_basis : str
        ``"CD_total"`` if the sections have profile drag, else ``"CDi"``.
    b_ref : float
        Reference span [m].
    c_ref : float
        Reference chord [m].
    ref_point : numpy.ndarray
        Moment reference point [m], shape (3,).
    base : SolverResult
        Result of the base condition.
    device : str
        Device of the solves (``"cpu"``, ``"gpu"`` or ``"mixed"``).
    precision : str
        Precision of the solves (``"float64"``, ``"float32"`` or ``"mixed"``).
    n_solves : int
        Number of solves done.
    notes : list[str]
        Notes on the method and its limits.
    static_margin : float or None
        Static margin fraction ``-Cm_alpha / CL_alpha`` [-], using the
        body-axis pitching moment derivative.
    neutral_point_x : float or None
        Neutral point x position [m], geometry axes (x aft).
    """

    derivatives: dict[str, float]
    matrix: np.ndarray
    axes: str
    steps: dict[str, float]
    drag_basis: str
    b_ref: float
    c_ref: float
    ref_point: np.ndarray
    base: SolverResult
    device: str
    precision: str
    n_solves: int
    notes: list[str] = field(default_factory=list)
    static_margin: float | None = None
    neutral_point_x: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary without the result object."""
        matrix_list = []
        for row in self.matrix:
            matrix_list.append([_json_float(v) for v in row])
        return {
            "derivatives": {k: _json_float(v) for k, v in self.derivatives.items()},
            "matrix": matrix_list,
            "axes": self.axes,
            "steps": {k: _json_float(v) for k, v in self.steps.items()},
            "drag_basis": self.drag_basis,
            "b_ref": _json_float(self.b_ref),
            "c_ref": _json_float(self.c_ref),
            "ref_point": [_json_float(v) for v in self.ref_point],
            "device": self.device,
            "precision": self.precision,
            "n_solves": int(self.n_solves),
            "notes": list(self.notes),
            "static_margin": _json_float(self.static_margin),
            "neutral_point_x": _json_float(self.neutral_point_x),
        }


def _coefficients(res: SolverResult, use_cd_total: bool) -> np.ndarray:
    """Return ``[CL, CD, CY, Cl, Cm, Cn]`` (body-axis moments) of a result."""
    t = res.totals
    cd = t.CD_total if (use_cd_total and t.CD_total is not None) else t.CDi
    return np.array([t.CL, cd, t.CY, t.Cl, t.Cm, t.Cn], dtype=float)


def _check_step(name: str, value: float) -> float:
    """Check a finite positive difference step [deg]."""
    v = float(value)
    if not (math.isfinite(v) and 0.0 < v <= 10.0):
        raise ValueError(f"'{name}' must satisfy 0 < {name} <= 10, got {value!r}.")
    return v


def static_derivatives(
    geometry: Aircraft | LiftingSurface,
    condition: FlightCondition,
    settings: SolverSettings | None = None,
    *,
    alpha_step_deg: float = 0.5,
    beta_step_deg: float = 1.0,
    axes: str = "body",
) -> StaticDerivativesResult:
    """Compute static stability derivatives with alpha and beta.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Case geometry. A single surface is wrapped into an aircraft.
    condition : FlightCondition
        Base flight condition.
    settings : SolverSettings or None
        Solver settings (default: :class:`SolverSettings`).
    alpha_step_deg : float
        Difference step for the angle of attack [deg]. Default 0.5.
    beta_step_deg : float
        Difference step for the sideslip angle [deg]. Default 1.0.
    axes : str
        Moment axes: ``"body"`` (default), ``"stability"`` or ``"wind"``.

    Returns
    -------
    StaticDerivativesResult
        The 12 derivatives, the base result, the device and the notes.

    Raises
    ------
    ValueError
        For an invalid step, an unknown axis system, or the Fourier solver.
    """
    from ventorum.core.axes import moment_axes_matrix
    from ventorum.geometry import lattice_cache
    from ventorum.solvers.factory import make_solver, resolve_solver_type
    from ventorum.utils.deprecation import canonical_settings
    from ventorum.utils.validation import validate_aircraft, validate_flight_condition, validate_solver_settings

    ac = Aircraft(name="SingleWing", surfaces=[geometry]) if isinstance(geometry, LiftingSurface) else geometry
    ac = ac.clone()
    settings = SolverSettings() if settings is None else settings
    validate_flight_condition(condition)
    settings = canonical_settings(settings)
    validate_solver_settings(settings)
    canonical = resolve_solver_type(settings.solver_type)
    if canonical == "fourier":
        raise ValueError("The Fourier solver has no sideslip model. Use solver_type 'vlm', 'linear' or 'nonlinear'.")

    d_alpha_deg = _check_step("alpha_step_deg", alpha_step_deg)
    d_beta_deg = _check_step("beta_step_deg", beta_step_deg)
    d_alpha = math.radians(d_alpha_deg)
    d_beta = math.radians(d_beta_deg)

    if axes not in ("body", "stability", "wind"):
        raise ValueError(f"'axes' must be one of ['body', 'stability', 'wind'], got {axes!r}.")

    validate_aircraft(ac)
    solver = make_solver(canonical)
    fp = lattice_cache.surfaces_fingerprint(ac)
    geo = lattice_cache.geometry_info(fp, ac)
    ac.compute_reference_values(auto=geo["auto_ref"])
    rp = np.asarray(ac.moment_reference(), dtype=float)
    b_ref, c_ref = float(ac.b_ref), float(ac.c_ref)

    conds = [
        condition,
        dataclasses.replace(condition, alpha=float(condition.alpha) + d_alpha),
        dataclasses.replace(condition, alpha=float(condition.alpha) - d_alpha),
        dataclasses.replace(condition, beta=float(condition.beta) + d_beta),
        dataclasses.replace(condition, beta=float(condition.beta) - d_beta),
    ]

    if settings.n_chord is None and solver.collocation == "vlm":
        settings = settings.clone()
        settings.n_chord = max(solver.resolve_n_chord(ac, settings, c, None, rp, fingerprint=fp) for c in conds)

    lattice = solver.build(ac, settings, condition, None, rp, fingerprint=fp)
    results = solver.solve_batch(
        lattice, conds, settings, ac.S_ref, b_ref, c_ref,
        ref_point=rp, main_surface=geo["main"], continuation=False,
    )
    for r, c in zip(results, conds):
        r.condition = c
    base = results[0]
    use_cd_total = base.totals.CD_total is not None
    coeffs = [_coefficients(r, use_cd_total) for r in results]

    body = np.empty((6, 2))
    body[:, 0] = (coeffs[1] - coeffs[2]) / (2.0 * d_alpha)
    body[:, 1] = (coeffs[3] - coeffs[4]) / (2.0 * d_beta)

    if axes == "body":
        matrix = body
    else:
        rot = moment_axes_matrix(
            float(condition.alpha), float(condition.beta),
            to=axes, from_axes="body", b_ref=b_ref, c_ref=c_ref,
        )
        matrix = body.copy()
        matrix[3:] = rot @ body[3:]

    derivs = {f"{c}_{v}": float(matrix[i, j]) for i, c in enumerate(_COEFFS) for j, v in enumerate(_VARIABLES)}

    cl_alpha = float(body[0, 0])
    cm_alpha_body = float(body[4, 0])
    if abs(cl_alpha) > 1e-9:
        x_np = float(rp[0] - c_ref * cm_alpha_body / cl_alpha)
        sm = float((x_np - rp[0]) / c_ref)
    else:
        x_np = None
        sm = None

    devices = {str(r.details.get("device", "cpu")) for r in results}
    precisions = {str(r.details.get("precision", "float64")) for r in results}
    notes = [
        f"Static derivatives per radian [1/rad], steps alpha {d_alpha_deg:g} deg, beta {d_beta_deg:g} deg.",
        f"Moment derivatives in the '{axes}' axes of the base condition, held fixed. CL, CD and CY are "
        "relative to the free stream.",
        "Static margin and neutral point use the body-axis Cm_alpha.",
    ]
    return StaticDerivativesResult(
        derivatives=derivs,
        matrix=matrix,
        axes=axes,
        steps={"alpha_deg": d_alpha_deg, "beta_deg": d_beta_deg},
        drag_basis="CD_total" if use_cd_total else "CDi",
        b_ref=b_ref,
        c_ref=c_ref,
        ref_point=rp,
        base=base,
        device=devices.pop() if len(devices) == 1 else "mixed",
        precision=precisions.pop() if len(precisions) == 1 else "mixed",
        n_solves=len(results),
        notes=notes,
        static_margin=sm,
        neutral_point_x=x_np,
    )
