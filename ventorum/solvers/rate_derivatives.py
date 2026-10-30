# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Rate derivatives: force and moment derivatives per unit body angular rate.

:func:`rate_derivatives` solves the base condition and, for each of the
three nondimensional body rates ``p_hat = p b / (2 V)``,
``q_hat = q c / (2 V)`` and ``r_hat = r b / (2 V)``, two conditions with the
rate changed by plus and minus a step. Central differences give the
derivatives of CL, CD, CY, Cl, Cm and Cn. All the cases are solved as one
batch on one lattice, so the differences contain no mesh change.

The solvers model the rotation as quasi-steady: the onset velocity at a
point ``x`` is ``V_inf d - Omega x (x - x_ref)`` (rotation about the moment
reference point, as AVL), and the wake stays straight along the free
stream (see :mod:`ventorum.aero.system`). In ground effect the derivatives
are at constant height of the reference point and constant attitude.

References
----------
* B. Etkin and L. D. Reid, Dynamics of Flight: Stability and Control,
  3rd ed., Wiley, 1996: the rate derivatives and their nondimensional
  rates.
* J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., Cambridge
  University Press, 2001: the kinematic velocity of a body-fixed frame in
  the boundary condition of the vortex lattice.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverResult, SolverSettings
from ventorum.solvers.lattice_base import LARGE_ROTATION_FRACTION

__all__ = ["RateDerivativesResult", "rate_derivatives", "rate_axes_transform"]

_RATES = ("p", "q", "r")
_COEFFS = ("CL", "CD", "CY", "Cl", "Cm", "Cn")


def _json_float(value: float) -> float | None:
    """Return ``float(value)``, or None when the value is not finite (strict JSON)."""
    v = float(value)
    return v if math.isfinite(v) else None


@dataclass
class RateDerivativesResult:
    """Result of :func:`rate_derivatives`.

    Attributes
    ----------
    derivatives : dict[str, float]
        Derivative of each coefficient per unit nondimensional rate, keyed
        ``"<coefficient>_<rate>"``, for example ``"Cl_p"`` = dCl/d(p b / 2V),
        ``"Cm_q"`` = dCm/d(q c / 2V), ``"Cn_r"`` = dCn/d(r b / 2V). Six
        coefficients (CL, CD, CY, Cl, Cm, Cn) times three rates [-].
    matrix : numpy.ndarray
        The same values as a (6, 3) array: rows CL, CD, CY, Cl, Cm, Cn;
        columns p_hat, q_hat, r_hat.
    axes : str
        Axes of the moments and of the rates: ``"body"``, ``"stability"``
        or ``"wind"`` of the base condition, held fixed.
    step : float
        Step of the nondimensional rates in the differences [-].
    drag_basis : str
        ``"CD_total"`` if the sections have profile drag, else ``"CDi"``.
    b_ref : float
        Reference span of the nondimensional rates p b/(2V), r b/(2V) [m].
    c_ref : float
        Reference chord of the nondimensional rate q c/(2V) [m].
    ref_point : numpy.ndarray
        Moment reference point, the centre of the rotation [m], shape (3,).
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
    """

    derivatives: dict[str, float]
    matrix: np.ndarray
    axes: str
    step: float
    drag_basis: str
    b_ref: float
    c_ref: float
    ref_point: np.ndarray
    base: SolverResult
    device: str
    precision: str
    n_solves: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary without the result object."""
        return {
            "derivatives": {k: _json_float(v) for k, v in self.derivatives.items()},
            "axes": self.axes,
            "step": float(self.step),
            "drag_basis": self.drag_basis,
            "b_ref_m": float(self.b_ref),
            "c_ref_m": float(self.c_ref),
            "ref_point_m": [float(x) for x in self.ref_point],
            "device": self.device,
            "precision": self.precision,
            "n_solves": int(self.n_solves),
            "notes": list(self.notes),
        }


def rate_axes_transform(
    matrix_body: np.ndarray,
    rotation: np.ndarray,
    b_ref: float,
    c_ref: float,
) -> np.ndarray:
    """Return the rate derivatives in other axes from the body-axis derivatives.

    The rates and the moments turn together with the orthogonal matrix
    *rotation* (body to the target axes, as in :mod:`ventorum.core.axes`).
    The nondimensional rates and moments use different reference lengths
    (b for roll and yaw, c for pitch), so the rotation acts on the
    dimensional values: with ``L = diag(b, c, b)``, the moment block is
    ``J' = L^-1 R L J L R^T L^-1`` and the force block is
    ``F' = F L R^T L^-1``. For the stability axes (a rotation about y) the
    lengths do not mix and ``J' = R J R^T``.

    Parameters
    ----------
    matrix_body : numpy.ndarray
        Body-axis derivatives [-] (per unit nondimensional rate), shape (6, 3): rows CL, CD, CY, Cl, Cm, Cn;
        columns p_hat, q_hat, r_hat.
    rotation : numpy.ndarray
        Rotation matrix from body to the target axes, shape (3, 3).
    b_ref : float
        Reference span [m].
    c_ref : float
        Reference chord [m].

    Returns
    -------
    numpy.ndarray
        Derivatives in the target axes, shape (6, 3).
    """
    m = np.asarray(matrix_body, dtype=float)
    R = np.asarray(rotation, dtype=float)
    L = np.diag([float(b_ref), float(c_ref), float(b_ref)])
    Li = np.diag([1.0 / float(b_ref), 1.0 / float(c_ref), 1.0 / float(b_ref)])
    rates = L @ R.T @ Li            # d(rate_hat_body)/d(rate_hat_axes)
    out = np.empty_like(m)
    out[:3] = m[:3] @ rates
    out[3:] = Li @ R @ L @ m[3:] @ rates
    return out


def _coefficients(res: SolverResult, use_cd_total: bool) -> np.ndarray:
    """Return ``[CL, CD, CY, Cl, Cm, Cn]`` (body-axis moments) of a result."""
    t = res.totals
    cd = t.CD_total if (use_cd_total and t.CD_total is not None) else t.CDi
    return np.array([t.CL, cd, t.CY, t.Cl, t.Cm, t.Cn], dtype=float)


def rate_derivatives(
    geometry: Aircraft | LiftingSurface,
    condition: FlightCondition,
    settings: SolverSettings | None = None,
    *,
    step: float = 0.01,
    axes: str = "body",
) -> RateDerivativesResult:
    """Compute the derivatives of the force and moment coefficients per unit body rate.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Geometry. It is not changed. The rotation is about its moment
        reference point (``Aircraft.ref_point``, default the origin).
    condition : FlightCondition
        Base flight condition. Its body rates ``p``, ``q``, ``r`` [rad/s]
        (default 0) are the point of the differences. With ``condition.h``
        (ground effect) the reference point stays at this height.
    settings : SolverSettings or None
        Solver settings (default: :class:`SolverSettings`). The solvers
        ``"vlm"``, ``"linear"`` and ``"nonlinear"`` are accepted; the
        Fourier solver has no body rates and is refused.
    step : float
        Step of the nondimensional rates p b/(2V), q c/(2V), r b/(2V) [-],
        in (0, 0.1]. Default 0.01 (a wing-tip helix angle of about 0.6 deg).
    axes : str
        Axes of the moments and of the rates: ``"body"`` (default),
        ``"stability"`` or ``"wind"``, of the base condition and held fixed.
        AVL gives its rate derivatives in the stability axes.

    Returns
    -------
    RateDerivativesResult
        The 18 derivatives, the base result, the device and the notes.

    Raises
    ------
    ValueError
        For a step outside (0, 0.1], an unknown axis system, the Fourier
        solver, or the 3D panel solver (give its body rates in the
        flight condition of ``analyze`` instead).

    Notes
    -----
    Central differences ``(C(w + h) - C(w - h)) / (2 h)`` in each
    nondimensional body rate ``w``. The circulation of the vortex lattice
    and of the linear lifting line is linear in the rates and the forces are
    quadratic, so the central difference is exact to round-off for them,
    except for the terms with the local speed (the lifting-line section
    equation, the profile drag and the section-moment term of one chordwise
    panel, also in the vortex lattice with ``n_chord=1``), which are second
    order in the step.

    The derivatives in other axes come from the body-axis derivatives by
    :func:`rate_axes_transform` (the rates and the moments turn together).

    The cases run on the selected device (:mod:`ventorum.gpu`); the
    result reports ``device`` (``"gpu"`` or ``"cpu"``) and ``precision``.
    """
    from ventorum.aero.system import rotation_velocity
    from ventorum.core.axes import rotation_body_to_stability, rotation_body_to_wind
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
        raise ValueError("The Fourier solver has no body rates. Use solver_type 'vlm', 'linear' or 'nonlinear'.")
    if canonical not in ("vlm", "linear", "nonlinear"):
        raise ValueError(
            f"rate_derivatives supports solver_type 'vlm', 'linear' or 'nonlinear', got {settings.solver_type!r}. "
            "For the 3D panel method, give the body rates p, q, r in the FlightCondition of analyze and take "
            "the differences."
        )
    step = float(step)
    if not (math.isfinite(step) and 0.0 < step <= 0.1):
        raise ValueError(f"step must satisfy 0 < step <= 0.1, got {step}.")
    if axes not in ("body", "stability", "wind"):
        raise ValueError(f"'axes' must be one of ['body', 'stability', 'wind'], got {axes!r}.")

    # The same preparation as LatticeSolver.solve, then one batch on one lattice.
    validate_aircraft(ac)
    solver = make_solver(canonical)
    fp = lattice_cache.surfaces_fingerprint(ac)
    geo = lattice_cache.geometry_info(fp, ac)
    ac.compute_reference_values(auto=geo["auto_ref"])
    rp = np.asarray(ac.moment_reference(), dtype=float)
    b_ref, c_ref = float(ac.b_ref), float(ac.c_ref)
    V = float(condition.V_inf)
    lattice = solver.build(ac, settings, condition, None, rp, fingerprint=fp)
    lattice.kernel_cache = {}   # the rate cases share the wake direction: one influence matrix

    to_rate = {"p": 2.0 * V / b_ref, "q": 2.0 * V / c_ref, "r": 2.0 * V / b_ref}
    conds = [condition]
    for name in _RATES:
        base_rate = float(getattr(condition, name))
        for sgn in (1.0, -1.0):
            conds.append(dataclasses.replace(condition, **{name: base_rate + sgn * step * to_rate[name]}))
    results = solver.solve_batch(lattice, conds, settings, ac.S_ref, b_ref, c_ref, ref_point=rp,
                                 main_surface=geo["main"], continuation=False)
    for r, c in zip(results, conds):
        r.condition = c
    base = results[0]
    use_cd_total = base.totals.CD_total is not None
    coeffs = [_coefficients(r, use_cd_total) for r in results]
    body = np.empty((6, 3))
    for j in range(3):
        body[:, j] = (coeffs[1 + 2 * j] - coeffs[2 + 2 * j]) / (2.0 * step)

    rot = {"body": np.eye(3),
           "stability": rotation_body_to_stability(float(condition.alpha)),
           "wind": rotation_body_to_wind(float(condition.alpha), float(condition.beta))}[axes]
    matrix = body if axes == "body" else rate_axes_transform(body, rot, b_ref, c_ref)
    derivs = {f"{c}_{w}": float(matrix[i, j]) for i, c in enumerate(_COEFFS) for j, w in enumerate(_RATES)}

    devices = {str(r.details.get("device", "cpu")) for r in results}
    precisions = {str(r.details.get("precision", "float64")) for r in results}
    notes = [
        f"Derivatives per unit nondimensional rate p b/(2V), q c/(2V), r b/(2V) (b_ref {b_ref:.6g} m, c_ref "
        f"{c_ref:.6g} m), central differences with step {step:g}.",
        f"Moments and rates in the '{axes}' axes of the base condition, held fixed; positive p, q, r give "
        "right wing down, nose up, nose right. CL, CD and CY are relative to the free stream.",
        "Quasi-steady rotation about the moment reference point: the onset velocity includes "
        "-Omega x (x - x_ref) and the wake stays straight along the free stream (as AVL).",
    ]
    if canonical in ("linear", "nonlinear"):
        notes.append("Lifting line: the section onset is taken at the three-quarter chord (thin-airfoil rule), "
                     "so a pitch rate changes the section lift as the equal change of the angle of attack.")
    if condition.h is not None:
        notes.append("Ground effect: the derivatives are at constant height of the reference point and constant "
                     "attitude. The ground images are mirror images of the vortices; the rotation enters only the "
                     "boundary condition on the aircraft.")
    pts = lattice.all_points()
    w_max = 0.0
    for c in conds:
        w = rotation_velocity(c, pts, rp)
        if w is not None:
            w_max = max(w_max, float(np.max(np.linalg.norm(w, axis=1))) / V)
    if w_max > LARGE_ROTATION_FRACTION:
        notes.append(f"The rotation speed reaches {w_max:.3f} V_inf on the geometry (above "
                     f"{LARGE_ROTATION_FRACTION:g}): the straight wake of the quasi-steady model is then a poor "
                     "approximation.")
    return RateDerivativesResult(
        derivatives=derivs, matrix=matrix, axes=axes, step=step,
        drag_basis="CD_total" if use_cd_total else "CDi", b_ref=b_ref, c_ref=c_ref, ref_point=rp,
        base=base,
        device=devices.pop() if len(devices) == 1 else "mixed",
        precision=precisions.pop() if len(precisions) == 1 else "mixed",
        n_solves=len(results), notes=notes,
    )
