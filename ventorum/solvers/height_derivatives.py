# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Height derivatives: the static derivatives of one flight state in ground effect.

:func:`height_derivatives` solves the base condition (``FlightCondition.h``
set) and, for each of the four variables alpha, beta, bank angle phi and
height h/c, two conditions with the variable changed by plus and minus a
step. Central differences give the derivatives of CL, CD, CY, Cl, Cm and Cn.
All the cases are solved as one batch on one lattice, so the differences
contain no mesh change.

Every derivative is about the moment reference point (the pivot), with the
height of that point held fixed: a change of alpha or phi turns the ground
about the pivot, and a change of h moves the ground along its normal. Near
the ground the lift depends on the attitude and on the height, and each has
its own aerodynamic centre, measured aft of the pivot [m]::

    x_alpha = -c * Cm_alpha / CL_alpha,    x_h = -c * Cm_h / CL_h

Longitudinal static stability at constant speed needs the height centre
ahead of the pitch centre (Irodov criterion)::

    irodov_margin = (x_alpha - x_h) / c > 0.

The criterion applies when the pivot is the centre of gravity.

References
----------
* R. D. Irodov, "Criteria of longitudinal stability of ekranoplan",
  Uchenye Zapiski TsAGI 1(4), 1970, pp. 63-72 (in Russian); English machine
  translation, Foreign Technology Division, 1974, DTIC AD-A002918.
* K. V. Rozhdestvensky, "Wing-in-ground effect vehicles", Progress in
  Aerospace Sciences 42, 2006, pp. 211-283: the pitch and height
  aerodynamic centres about the centre of gravity at a constant height of
  that point.
* B. Etkin and L. D. Reid, Dynamics of Flight: Stability and Control,
  3rd ed., Wiley, 1996: the static derivatives and their axes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from ventorum.core.constants import MIN_CENTRE_SLOPE
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverResult, SolverSettings
from ventorum.core.errors import GroundStrikeError

__all__ = ["HeightDerivativesResult", "height_derivatives"]

_VARIABLES = ("alpha", "beta", "phi", "h")
_COEFFS = ("CL", "CD", "CY", "Cl", "Cm", "Cn")
# A slope below this value [per rad or per unit h/c] gives no aerodynamic centre.
_MIN_SLOPE = MIN_CENTRE_SLOPE


def _json_float(value: float | None) -> float | None:
    """Return ``float(value)``, or None when the value is None or not finite (strict JSON)."""
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


@dataclass
class HeightDerivativesResult:
    """Result of :func:`height_derivatives`.

    Attributes
    ----------
    derivatives : dict[str, float]
        Derivative of each coefficient, keyed ``"<coefficient>_<variable>"``,
        for example ``"CL_alpha"`` = dCL/dalpha [1/rad], ``"Cl_phi"`` =
        dCl/dphi [1/rad], ``"Cm_h"`` = dCm/d(h/c_ref) [-]. Six coefficients
        (CL, CD, CY, Cl, Cm, Cn) times four variables (alpha, beta, phi, h).
    matrix : numpy.ndarray
        The same values as a (6, 4) array: rows CL, CD, CY, Cl, Cm, Cn;
        columns alpha [1/rad], beta [1/rad], phi [1/rad], h/c_ref [-].
    irodov : dict[str, Any]
        ``x_alpha_m`` and ``x_h_m``: pitch and height aerodynamic centres
        [m] aft of the pivot (None when the lift slope is near zero);
        ``margin``: (x_alpha - x_h)/c_ref [-], > 0 for static stability;
        ``verdict``: ``"stable"``, ``"unstable"`` or ``"not available"``.
    axes : str
        Axes of the moments: ``"body"``, ``"stability"`` or ``"wind"`` of
        the base condition, held fixed.
    steps : dict[str, float]
        Steps of the differences: ``alpha_deg``, ``beta_deg``, ``phi_deg``
        [deg] and ``h_m`` [m].
    drag_basis : str
        ``"CD_total"`` if the sections have profile drag, else ``"CDi"``.
    h_m : float
        Height of the pivot above the ground in the base state [m].
    h_min_m : float
        Smallest clearance of the geometry in the base state [m].
    b_ref : float
        Reference span [m].
    c_ref : float
        Reference chord [m]; the height variable is h/c_ref.
    ref_point : numpy.ndarray
        Moment reference point, the pivot [m], shape (3,).
    n_chord : int
        Number of chordwise panels of all the cases.
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
    irodov: dict[str, Any]
    axes: str
    steps: dict[str, float]
    drag_basis: str
    h_m: float
    h_min_m: float
    b_ref: float
    c_ref: float
    ref_point: np.ndarray
    n_chord: int
    base: SolverResult
    device: str
    precision: str
    n_solves: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary without the result object."""
        return {
            "derivatives": {k: _json_float(v) for k, v in self.derivatives.items()},
            "irodov": {k: (_json_float(v) if k != "verdict" else v) for k, v in self.irodov.items()},
            "axes": self.axes,
            "steps": {k: float(v) for k, v in self.steps.items()},
            "drag_basis": self.drag_basis,
            "h_m": float(self.h_m),
            "h_min_m": float(self.h_min_m),
            "b_ref_m": float(self.b_ref),
            "c_ref_m": float(self.c_ref),
            "ref_point_m": [float(x) for x in self.ref_point],
            "n_chord": int(self.n_chord),
            "device": self.device,
            "precision": self.precision,
            "n_solves": int(self.n_solves),
            "notes": list(self.notes),
        }


def _coefficients(res: SolverResult, use_cd_total: bool) -> np.ndarray:
    """Return ``[CL, CD, CY, Cl, Cm, Cn]`` (body-axis moments) of a result.

    The base case sets *use_cd_total* once for all the cases. A case has
    no ``CD_total`` only when its profile drag is exactly zero, so then
    ``CD_total = CDi + 0`` and the value ``CDi`` keeps the same drag basis.
    """
    t = res.totals
    cd = t.CD_total if (use_cd_total and t.CD_total is not None) else t.CDi
    return np.array([t.CL, cd, t.CY, t.Cl, t.Cm, t.Cn], dtype=float)


def _check_step(name: str, value: float, upper: float) -> float:
    v = float(value)
    if not (math.isfinite(v) and 0.0 < v <= upper):
        raise ValueError(f"{name} must satisfy 0 < {name} <= {upper:g}, got {value}.")
    return v


def _irodov(matrix_body: np.ndarray, c_ref: float) -> dict[str, Any]:
    """Return the pitch and height aerodynamic centres [m] and the Irodov margin."""
    CL_a, CL_h = matrix_body[0, 0], matrix_body[0, 3]
    Cm_a, Cm_h = matrix_body[4, 0], matrix_body[4, 3]
    x_alpha = -c_ref * Cm_a / CL_a if abs(CL_a) > _MIN_SLOPE else None
    x_h = -c_ref * Cm_h / CL_h if abs(CL_h) > _MIN_SLOPE else None
    if x_alpha is None or x_h is None:
        return {"x_alpha_m": x_alpha, "x_h_m": x_h, "margin": None, "verdict": "not available"}
    margin = (x_alpha - x_h) / c_ref
    verdict = "stable" if margin > 0.0 else "unstable"
    return {"x_alpha_m": float(x_alpha), "x_h_m": float(x_h), "margin": float(margin), "verdict": verdict}


def height_derivatives(
    geometry: Aircraft | LiftingSurface,
    condition: FlightCondition,
    settings: SolverSettings | None = None,
    *,
    h_step: float = 0.05,
    alpha_step_deg: float = 0.5,
    beta_step_deg: float = 1.0,
    phi_step_deg: float = 0.5,
    axes: str = "body",
) -> HeightDerivativesResult:
    """Compute the static derivatives of one flight state in ground effect.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Geometry. It is not changed. The pivot is its moment reference point
        (``Aircraft.ref_point``, default the origin); give the centre of
        gravity for the Irodov criterion.
    condition : FlightCondition
        Base flight condition. ``condition.h`` [m] (the height of the pivot
        above the ground) must be set. Its alpha, beta and phi [rad] are the
        point of the differences; its body rates are kept in every case.
    settings : SolverSettings or None
        Solver settings (default: :class:`SolverSettings`). The solvers
        ``"vlm"``, ``"linear"`` and ``"nonlinear"`` are accepted; the
        Fourier solver has no ground effect and is refused. With
        ``n_chord=None`` all the cases use the largest automatic chordwise
        count of the nine cases (normally that of the case with the smallest
        gap to the ground).
    h_step : float
        Height step as a fraction of the smallest clearance of the base
        state [-], in (0, 0.5]. Default 0.05.
    alpha_step_deg : float
        Step of the angle of attack [deg], in (0, 5]. Default 0.5.
    beta_step_deg : float
        Step of the sideslip angle [deg], in (0, 5]. Default 1.0.
    phi_step_deg : float
        Step of the bank angle [deg], in (0, 5]. Default 0.5.
    axes : str
        Axes of the moments: ``"body"`` (default), ``"stability"`` or
        ``"wind"`` of the base condition, held fixed.

    Returns
    -------
    HeightDerivativesResult
        The 24 derivatives, the Irodov centres and margin, the base result,
        the device and the notes.

    Raises
    ------
    ValueError
        If ``condition.h`` is None, for a step outside its range, an
        unknown axis system, or the Fourier solver.
    GroundStrikeError
        If the base case or a stepped case touches the ground.
    ValidityError
        If a lifting-line solver is used below its ground-effect limit in
        one of the cases.

    Notes
    -----
    Central differences ``(C(v + d) - C(v - d)) / (2 d)`` in each variable
    ``v``, with the other three held fixed. The pivot keeps its height in
    every case: alpha and phi turn the ground about the pivot (a change of
    attitude), and the height step moves the ground along its normal (a
    change of height at constant attitude).

    The moment rows in other axes turn the dimensional moments with the
    axes of the base condition: ``C' = L^-1 R L C`` with
    ``L = diag(b, c, b)``, because Cl and Cn use b_ref and Cm uses c_ref.
    CL, CD and CY are relative to the free stream in every set.

    The solves run on the selected device (see :mod:`ventorum.gpu`).
    """
    from ventorum.aero.system import ground_plane_from_condition
    from ventorum.core.axes import rotation_body_to_stability, rotation_body_to_wind
    from ventorum.geometry import lattice_cache
    from ventorum.solvers.factory import make_solver, resolve_solver_type
    from ventorum.utils.deprecation import canonical_settings
    from ventorum.utils.validation import validate_aircraft, validate_flight_condition, validate_solver_settings

    ac = Aircraft(name="SingleWing", surfaces=[geometry]) if isinstance(geometry, LiftingSurface) else geometry
    ac = ac.clone()
    settings = SolverSettings() if settings is None else settings
    validate_flight_condition(condition)
    if condition.h is None:
        raise ValueError("height_derivatives needs ground effect: set condition.h, the height [m] of the "
                         "moment reference point above the ground.")
    settings = canonical_settings(settings)
    validate_solver_settings(settings)
    canonical = resolve_solver_type(settings.solver_type)
    if canonical == "fourier":
        raise ValueError("The Fourier solver has no ground effect. Use solver_type 'vlm', 'linear' or "
                         "'nonlinear'.")
    h_frac = _check_step("h_step", h_step, 0.5)
    d_alpha = math.radians(_check_step("alpha_step_deg", alpha_step_deg, 5.0))
    d_beta = math.radians(_check_step("beta_step_deg", beta_step_deg, 5.0))
    d_phi = math.radians(_check_step("phi_step_deg", phi_step_deg, 5.0))
    if axes not in ("body", "stability", "wind"):
        raise ValueError(f"'axes' must be one of ['body', 'stability', 'wind'], got {axes!r}.")

    validate_aircraft(ac)
    solver = make_solver(canonical)
    fp = lattice_cache.surfaces_fingerprint(ac)
    geo = lattice_cache.geometry_info(fp, ac)
    ac.compute_reference_values(auto=geo["auto_ref"])
    rp = np.asarray(ac.moment_reference(), dtype=float)
    b_ref, c_ref = float(ac.b_ref), float(ac.c_ref)
    h0 = float(condition.h)

    # Smallest clearance of the base state, on the lattice with one chordwise
    # panel (the nodes of the leading and trailing edges).
    chord_spacing = getattr(settings, "chord_spacing", "uniform")
    from ventorum.geometry.lattice import build_lattice

    key = lattice_cache.lattice_key(ac, settings, "vlm", 1, chord_spacing, fingerprint=fp)
    probe = lattice_cache.get_or_build(key, lambda: build_lattice(
        ac, settings, collocation="vlm", n_chord=1, chord_spacing=chord_spacing))
    h_min = float(np.min(ground_plane_from_condition(probe, condition, rp).height(probe.all_points())))
    if h_min <= 0.0:
        raise GroundStrikeError(f"The base state touches the ground (minimum clearance {h_min:.4g} m at "
                                f"h={h0:g} m). No result is computed for a ground strike.")
    d_h = h_frac * h_min

    conds = [condition]
    labels = ["base"]
    for name, d, unit in (("alpha", d_alpha, "rad"), ("beta", d_beta, "rad"), ("phi", d_phi, "rad"),
                          ("h", d_h, "m")):
        base_val = float(getattr(condition, name))
        for sgn in (1.0, -1.0):
            conds.append(dataclasses.replace(condition, **{name: base_val + sgn * d}))
            labels.append(f"{name} {'+' if sgn > 0 else '-'}{d:.4g} {unit}")
    for c, lab in zip(conds[1:], labels[1:]):
        hm = float(np.min(ground_plane_from_condition(probe, c, rp).height(probe.all_points())))
        if hm <= 0.0:
            raise GroundStrikeError(
                f"The case '{lab}' of the differences touches the ground (minimum clearance {hm:.4g} m). "
                "Use smaller steps (alpha_step_deg, phi_step_deg, h_step) or a larger height.")

    # One chordwise count for all the cases (the count of the smallest gap),
    # so that the differences contain no mesh change.
    if settings.n_chord is None and solver.collocation == "vlm":
        settings = settings.clone()
        settings.n_chord = max(solver.resolve_n_chord(ac, settings, c, None, rp, fingerprint=fp) for c in conds)
    lattice = solver.build(ac, settings, condition, None, rp, fingerprint=fp)
    results = solver.solve_batch(lattice, conds, settings, ac.S_ref, b_ref, c_ref, ref_point=rp,
                                 main_surface=geo["main"], continuation=False)
    for r, c in zip(results, conds):
        r.condition = c
    base = results[0]
    use_cd_total = base.totals.CD_total is not None
    coeffs = [_coefficients(r, use_cd_total) for r in results]
    denominators = (2.0 * d_alpha, 2.0 * d_beta, 2.0 * d_phi, 2.0 * d_h / c_ref)
    body = np.empty((6, 4))
    for j, den in enumerate(denominators):
        body[:, j] = (coeffs[1 + 2 * j] - coeffs[2 + 2 * j]) / den

    if axes == "body":
        matrix = body
    else:
        R = (rotation_body_to_stability(float(condition.alpha)) if axes == "stability"
             else rotation_body_to_wind(float(condition.alpha), float(condition.beta)))
        L = np.diag([b_ref, c_ref, b_ref])
        Li = np.diag([1.0 / b_ref, 1.0 / c_ref, 1.0 / b_ref])
        matrix = body.copy()
        matrix[3:] = Li @ R @ L @ body[3:]
    derivs = {f"{c}_{v}": float(matrix[i, j]) for i, c in enumerate(_COEFFS) for j, v in enumerate(_VARIABLES)}
    irodov = _irodov(body, c_ref)

    devices = {str(r.details.get("device", "cpu")) for r in results}
    precisions = {str(r.details.get("precision", "float64")) for r in results}
    notes = [
        f"Central differences about the state alpha={math.degrees(condition.alpha):g} deg, "
        f"beta={math.degrees(condition.beta):g} deg, phi={math.degrees(condition.phi):g} deg, h={h0:g} m "
        f"(h/c_ref={h0 / c_ref:.4g}, smallest clearance {h_min:.4g} m). Steps: alpha "
        f"{math.degrees(d_alpha):g} deg, beta {math.degrees(d_beta):g} deg, phi {math.degrees(d_phi):g} deg, "
        f"h {d_h:.4g} m ({h_frac:g} of the smallest clearance).",
        "Every derivative is about the moment reference point (the pivot) with the height of that point held "
        "fixed: alpha and phi turn the ground about the pivot, h moves the ground along its normal. The h "
        "column is per unit h/c_ref.",
        f"Moments in the '{axes}' axes of the base condition, held fixed (C' = L^-1 R L C, L = diag(b, c, b)); "
        "CL, CD and CY are relative to the free stream. Cl > 0 right wing down, Cm > 0 nose up, Cn > 0 nose "
        "right; positive phi puts the right wing nearer to the ground.",
        "Irodov criterion: margin = (x_alpha - x_h)/c_ref > 0 for longitudinal static stability at constant "
        "speed (x_alpha, x_h aft of the pivot). It applies only when the pivot is the centre of gravity.",
        "Quasi-steady, level flight: the ground stays parallel to the free stream. Heave-rate (flight-path) "
        "effects are not modelled.",
    ]
    return HeightDerivativesResult(
        derivatives=derivs, matrix=matrix, irodov=irodov, axes=axes,
        steps={"alpha_deg": math.degrees(d_alpha), "beta_deg": math.degrees(d_beta),
               "phi_deg": math.degrees(d_phi), "h_m": d_h},
        drag_basis="CD_total" if use_cd_total else "CDi", h_m=h0, h_min_m=h_min, b_ref=b_ref, c_ref=c_ref,
        ref_point=rp, n_chord=int(lattice.n_chord), base=base,
        device=devices.pop() if len(devices) == 1 else "mixed",
        precision=precisions.pop() if len(precisions) == 1 else "mixed",
        n_solves=len(results), notes=notes,
    )
