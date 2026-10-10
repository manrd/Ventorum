# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Control derivatives per unit deflection and hinge-moment derivatives.

:func:`control_derivatives` solves the base case and, for each control name,
two cases with the deflection changed by plus and minus a step. Central
differences give the derivatives of the force and moment coefficients per
unit deflection [1/rad]. The hinge moment of each copy of each control
(:func:`ventorum.aero.hinge.hinge_moments`) gives, from the same solves, the
derivatives of the hinge-moment coefficient with deflection, and two more
solves at alpha +/- 0.5 deg give its derivative with the angle of attack.

All controls with one name move together, as with
:meth:`ventorum.core.datatypes.Aircraft.set_deflection`. The derivative is
per unit change of the ``deflection`` field: the left half of an
antisymmetric control moves by the opposite amount.

References
----------
B. Etkin and L. D. Reid, Dynamics of Flight: Stability and Control, 3rd ed.,
Wiley, 1996: control derivatives and hinge-moment coefficients.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from ventorum.aero.hinge import HingeMoment, hinge_moments
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverResult, SolverSettings

__all__ = ["ControlDerivatives", "HingeMomentDerivatives", "ControlDerivativesResult", "control_derivatives"]

# Largest deflection of a control [rad] (the limit of validate_controls).
_MAX_DEFLECTION = math.radians(30.0)
# Step of the angle of attack for the hinge-moment derivative [deg] (as the stability derivatives).
_ALPHA_STEP_DEG = 0.5


def _json_float(value: float) -> float | None:
    """Return ``float(value)``, or None when the value is not finite (strict JSON)."""
    v = float(value)
    return v if math.isfinite(v) else None


@dataclass
class ControlDerivatives:
    """Derivatives of the coefficients per unit deflection of one control name [1/rad].

    Attributes
    ----------
    control : str
        Control name.
    deflection : float
        Deflection of the base case [rad] (of the first control with this name).
    step : float
        Deflection step of the differences [rad].
    scheme : str
        ``"central"``, or ``"backward"``/``"forward"`` (second-order one-sided
        differences) when the central step would pass the 30 deg limit.
    CL, CD, CY : float
        Force-coefficient derivatives [1/rad], relative to the free stream.
    Cl, Cm, Cn : float
        Moment-coefficient derivatives [1/rad] in the axes of the result.
    """

    control: str
    deflection: float
    step: float
    scheme: str
    CL: float
    CD: float
    CY: float
    Cl: float
    Cm: float
    Cn: float

    def to_dict(self) -> dict[str, Any]:
        """Return the values as a dictionary of plain Python types (non-finite values as None)."""
        out: dict[str, Any] = {"control": self.control, "scheme": self.scheme,
                               "deflection_rad": _json_float(self.deflection), "step_rad": _json_float(self.step)}
        for k in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
            out[f"{k}_delta_per_rad"] = _json_float(getattr(self, k))
        return out


@dataclass
class HingeMomentDerivatives:
    """Hinge moment of one copy of a control at the base case, and its derivatives.

    Attributes
    ----------
    moment : HingeMoment
        Hinge moment of the base case.
    Ch_alpha : float
        Derivative of ``Ch`` with the angle of attack [1/rad] (NaN if not
        computed).
    Ch_delta : dict[str, float]
        Derivative of ``Ch`` with the deflection of each control name
        [1/rad]. The item of the own control name is the hinge-moment
        derivative of the control; the other items are the interference of
        the other controls.
    """

    moment: HingeMoment
    Ch_alpha: float
    Ch_delta: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return the values as a dictionary of plain Python types (non-finite values as None)."""
        out = self.moment.to_dict()
        out["Ch_alpha_per_rad"] = _json_float(self.Ch_alpha)
        out["Ch_delta_per_rad"] = {k: _json_float(v) for k, v in self.Ch_delta.items()}
        return out


@dataclass
class ControlDerivativesResult:
    """Result of :func:`control_derivatives`.

    Attributes
    ----------
    derivatives : dict[str, ControlDerivatives]
        Coefficient derivatives per control name.
    hinge : list[HingeMomentDerivatives]
        Hinge moment and its derivatives for each copy of each control.
    axes : str
        Axes of the moment derivatives: ``"body"``, ``"stability"`` or
        ``"wind"`` of the base condition, held fixed.
    drag_basis : str
        ``"CD_total"`` if the sections have profile drag, else ``"CDi"``.
    base : SolverResult
        Result of the base case.
    device : str
        Device of the solves (``"cpu"``, ``"gpu"`` or ``"mixed"``).
    precision : str
        Precision of the solves (``"float64"``, ``"float32"`` or ``"mixed"``).
    n_solves : int
        Number of solves done.
    notes : list[str]
        Notes on the method and its limits.
    """

    derivatives: dict[str, ControlDerivatives]
    hinge: list[HingeMomentDerivatives]
    axes: str
    drag_basis: str
    base: SolverResult
    device: str
    precision: str
    n_solves: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary without the result object."""
        return {
            "derivatives": {k: v.to_dict() for k, v in self.derivatives.items()},
            "hinge": [h.to_dict() for h in self.hinge],
            "axes": self.axes,
            "drag_basis": self.drag_basis,
            "device": self.device,
            "precision": self.precision,
            "n_solves": int(self.n_solves),
            "notes": list(self.notes),
        }


def _deflections(ac: Aircraft, name: str) -> list[float]:
    """Return the deflections [rad] of all controls with this name."""
    return [float(c.deflection) for s in ac.surfaces for c in getattr(s, "controls", []) if c.name == name]


def _shifted(ac: Aircraft, name: str, d: float) -> Aircraft:
    """Return a clone of *ac* with the deflection of every control *name* changed by *d* [rad]."""
    out = ac.clone()
    for s in out.surfaces:
        for c in getattr(s, "controls", []):
            if c.name == name:
                c.deflection = float(c.deflection) + float(d)
    return out


def _coefficients(res: SolverResult, use_cd_total: bool) -> np.ndarray:
    """Return ``[CL, CD, CY, Cl, Cm, Cn]`` (body-axis moments) of a result."""
    t = res.totals
    cd = t.CD_total if (use_cd_total and t.CD_total is not None) else t.CDi
    return np.array([t.CL, cd, t.CY, t.Cl, t.Cm, t.Cn], dtype=float)


def _hinge_key(m: HingeMoment) -> tuple[str, str, str]:
    """Return the identity of a copy of a control: surface, control name and side."""
    return (m.surface, m.control, m.side)


def control_derivatives(
    geometry: Aircraft | LiftingSurface,
    condition: FlightCondition,
    settings: SolverSettings | None = None,
    *,
    controls: list[str] | None = None,
    step_deg: float = 1.0,
    axes: str = "body",
    hinge_alpha: bool = True,
) -> ControlDerivativesResult:
    """Compute control derivatives per unit deflection and hinge-moment derivatives.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Geometry with control surfaces. It is not changed.
    condition : FlightCondition
        Base flight condition. With ``condition.h`` (ground effect), the
        moment reference point stays at this height in every case.
    settings : SolverSettings or None
        Solver settings (default: :class:`SolverSettings`). The Fourier
        solver has no control surfaces and is refused.
    controls : list of str or None
        Control names to differentiate. None means all names of the
        aircraft.
    step_deg : float
        Deflection step of the differences [deg], in (0, 10]. Default 1.
    axes : str
        Axes of the moment derivatives: ``"body"`` (default),
        ``"stability"`` or ``"wind"``, of the base condition and held fixed.
    hinge_alpha : bool
        Also compute the hinge-moment derivative with the angle of attack
        (two more solves at alpha +/- 0.5 deg). Default True.

    Returns
    -------
    ControlDerivativesResult
        Coefficient derivatives per control name, and the hinge moment of
        each copy of each control with its derivatives.

    Raises
    ------
    ValueError
        For an aircraft without controls, an unknown control name, a step
        outside (0, 10] deg, an unknown axis system, or the Fourier solver.

    Notes
    -----
    Central differences ``(C(delta + h) - C(delta - h)) / (2 h)``. When
    ``|delta| + h`` passes the 30 deg limit, the second-order one-sided
    difference ``(3 C(delta) - 4 C(delta - h) + C(delta - 2 h)) / (2 h)`` (or its
    forward form) replaces it. The cases with a changed deflection use the
    same mesh: the hinge snap and the control strips do not depend on the
    deflection.

    The solves run on the device that the settings and the global device
    select. The result states the device and the precision. With a 1 deg
    step, the float32 round-off of the GPU paths is about 1e-5 of the
    derivative.

    The vortex lattice (``n_chord >= 2``) turns the normals of the flap
    panels, so its derivatives include the change of the chordwise load.
    The lifting lines use the thin-airfoil deflected section (Glauert 1926).
    """
    from ventorum import analyze
    from ventorum.core.axes import rotation_body_to_stability, rotation_body_to_wind
    from ventorum.solvers.factory import resolve_solver_type
    from ventorum.utils.validation import validate_flight_condition, validate_solver_settings

    ac = Aircraft(name="SingleWing", surfaces=[geometry]) if isinstance(geometry, LiftingSurface) else geometry
    ac = ac.clone()
    settings = SolverSettings() if settings is None else settings
    validate_flight_condition(condition)
    validate_solver_settings(settings)
    if resolve_solver_type(settings.solver_type) == "fourier":
        raise ValueError("The Fourier solver does not accept a deflected control. "
                         "Use solver_type 'vlm', 'linear' or 'nonlinear'.")
    known = ac.control_names()
    if not known:
        raise ValueError("The aircraft has no control surface.")
    names = list(known) if controls is None else list(controls)
    for n in names:
        if n not in known:
            raise ValueError(f"Unknown control surface {n!r}; known control names: {known}.")
    if len(set(names)) != len(names):
        raise ValueError(f"Control names must be unique, got {names}.")
    step_deg = float(step_deg)
    if not (math.isfinite(step_deg) and 0.0 < step_deg <= 10.0):
        raise ValueError(f"step_deg must satisfy 0 < step_deg <= 10, got {step_deg}.")
    if axes not in ("body", "stability", "wind"):
        raise ValueError(f"'axes' must be one of ['body', 'stability', 'wind'], got {axes!r}.")
    h = math.radians(step_deg)

    results: list[SolverResult] = []

    def solve(aircraft: Aircraft, cond: FlightCondition) -> SolverResult:
        r = analyze(aircraft, cond, settings)
        results.append(r)
        return r

    base = solve(ac, condition)
    use_cd_total = base.totals.CD_total is not None
    rot = {"body": np.eye(3),
           "stability": rotation_body_to_stability(float(condition.alpha)),
           "wind": rotation_body_to_wind(float(condition.alpha), float(condition.beta))}[axes]

    def coeffs(r: SolverResult) -> np.ndarray:
        c = _coefficients(r, use_cd_total)
        c[3:] = rot @ c[3:]
        return c

    def hinge_ch(r: SolverResult) -> dict[tuple[str, str, str], float]:
        # Two controls with one name on one surface have the same key: index the copies by position.
        ms = hinge_moments(r)
        if [_hinge_key(m) for m in ms] != base_ids:
            raise RuntimeError("The control copies changed between the solves of the differences.")
        return {i: m.Ch for i, m in enumerate(ms)}

    base_hinge = hinge_moments(base)
    base_ids = [_hinge_key(m) for m in base_hinge]
    keys = list(range(len(base_hinge)))
    c0 = coeffs(base)
    ch0 = {i: m.Ch for i, m in enumerate(base_hinge)}

    derivs: dict[str, ControlDerivatives] = {}
    ch_delta: dict[int, dict[str, float]] = {k: {} for k in keys}
    for name in names:
        defl = _deflections(ac, name)
        top = max(defl) + h
        bottom = min(defl) - h
        if top <= _MAX_DEFLECTION + 1e-12 and bottom >= -_MAX_DEFLECTION - 1e-12:
            scheme = "central"
            r_p = solve(_shifted(ac, name, h), condition)
            r_m = solve(_shifted(ac, name, -h), condition)
            dc = (coeffs(r_p) - coeffs(r_m)) / (2.0 * h)
            chp, chm = hinge_ch(r_p), hinge_ch(r_m)
            dch = {k: (chp[k] - chm[k]) / (2.0 * h) for k in keys}
        else:
            sgn = -1.0 if top > _MAX_DEFLECTION + 1e-12 else 1.0
            if min(defl) + sgn * 2.0 * h < -_MAX_DEFLECTION - 1e-12 or max(defl) + sgn * 2.0 * h > _MAX_DEFLECTION + 1e-12:
                raise ValueError(
                    f"Control {name!r}: the deflections {[round(math.degrees(x), 3) for x in defl]} deg leave no "
                    f"room for a step of {step_deg:g} deg inside the 30 deg limit. Use a smaller step_deg."
                )
            scheme = "forward" if sgn > 0 else "backward"
            r_1 = solve(_shifted(ac, name, sgn * h), condition)
            r_2 = solve(_shifted(ac, name, sgn * 2.0 * h), condition)
            dc = sgn * (-3.0 * c0 + 4.0 * coeffs(r_1) - coeffs(r_2)) / (2.0 * h)
            ch1, ch2 = hinge_ch(r_1), hinge_ch(r_2)
            dch = {k: sgn * (-3.0 * ch0[k] + 4.0 * ch1[k] - ch2[k]) / (2.0 * h) for k in keys}
        derivs[name] = ControlDerivatives(
            control=name, deflection=defl[0], step=h, scheme=scheme,
            CL=float(dc[0]), CD=float(dc[1]), CY=float(dc[2]), Cl=float(dc[3]), Cm=float(dc[4]), Cn=float(dc[5]),
        )
        for k in keys:
            ch_delta[k][name] = float(dch[k])

    ch_alpha = {k: float("nan") for k in keys}
    if hinge_alpha:
        da = math.radians(_ALPHA_STEP_DEG)
        r_ap = solve(ac, _with_alpha(condition, float(condition.alpha) + da))
        r_am = solve(ac, _with_alpha(condition, float(condition.alpha) - da))
        chp, chm = hinge_ch(r_ap), hinge_ch(r_am)
        ch_alpha = {k: (chp[k] - chm[k]) / (2.0 * da) for k in keys}

    hinge = [HingeMomentDerivatives(moment=m, Ch_alpha=ch_alpha[i], Ch_delta=ch_delta[i])
             for i, m in enumerate(base_hinge)]

    devices = {str(r.details.get("device", "cpu")) for r in results}
    precisions = {str(r.details.get("precision", "float64")) for r in results}
    modes = {m.mode for m in base_hinge}
    notes = [
        f"Derivatives per unit change of the 'deflection' field [1/rad], step {step_deg:g} deg. All controls with "
        "one name move together; the left half of an antisymmetric control moves by the opposite amount.",
        f"Moment derivatives in the '{axes}' axes of the base condition, held fixed. CL, CD and CY are relative "
        "to the free stream.",
        "Hinge moment positive when it tends to increase the local deflection (trailing edge down on a "
        "horizontal surface). Ch = H / (q S_f c_f), S_f the flap area aft of the hinge, c_f the "
        "root-mean-square flap chord.",
    ]
    if "chordwise" in modes:
        notes.append("Chordwise hinge moments come from the bound-vortex forces of the flap panels. They converge "
                     "slowly with n_chord (the load is logarithmic at the hinge): use 16 or more chordwise panels "
                     "and see the convergence table in the theory manual (control surfaces).")
    if "section" in modes:
        notes.append("Section hinge moments use thin-airfoil theory of a plain flap on a flat plate: no camber "
                     "term, no gap, attached flow only.")
    return ControlDerivativesResult(
        derivatives=derivs, hinge=hinge, axes=axes, drag_basis="CD_total" if use_cd_total else "CDi",
        base=base,
        device=devices.pop() if len(devices) == 1 else "mixed",
        precision=precisions.pop() if len(precisions) == 1 else "mixed",
        n_solves=len(results), notes=notes,
    )


def _with_alpha(condition: FlightCondition, alpha: float) -> FlightCondition:
    """Return a copy of *condition* with another angle of attack [rad]."""
    return dataclasses.replace(condition, alpha=float(alpha))
