# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Public time-domain unsteady response and indicial-response interfaces.

:func:`time_response` marches the prescribed-wake time-domain vortex-ring
lattice solver (:mod:`ventorum.unsteady.timedomain`) for six-axis rigid-body
motion, moving control surfaces and arbitrary convected gust or wind-shear
fields, in free air or in flat-ground effect.

:func:`indicial_response` computes unit-step indicial aerodynamic responses
as functions of nondimensional distance travelled in semichords
``s = 2 V_inf t / c_ref`` [-]:

* ``kind="alpha"`` (Wagner): uniform onset step over the whole wing at
  ``t = 0``, ``V_inf cos(beta) lift_direction(alpha) amplitude`` (the
  first-order change of the free stream for an incidence step at fixed
  geometry).
* ``kind="gust"`` (Kussner): sharp-edged vertical gust front starting at the
  leading edge at ``t = 0``.
* ``kind="control"``: step deflection of a named control surface at ``t = 0``.

Force coefficients ``(CL, CD, CY)`` are projected onto the mean wind axes and
moment coefficients ``(Cl, Cm, Cn)`` are reported in the body axes of the mean
state about the instantaneous reference point, using the nominal ``V_inf``
[m/s] and ``rho`` [kg/m^3] of the mean flight condition.

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, chapter 13 (time-domain unsteady vortex-ring
  lattice, impulsive start, wake shedding and unsteady Bernoulli loads).
* H. Wagner, "Uber die Entstehung des dynamischen Auftriebes von
  Tragflugeln", ZAMM 5(1), 1925, pp. 17-35 (indicial lift after an incidence
  step).
* H. G. Kussner, "Zusammenfassender Bericht uber den instationaren Auftrieb
  von Flugeln", Luftfahrtforschung 13(12), 1936, pp. 410-424 (sharp-edged
  gust penetration).
* R. T. Jones, "The Unsteady Lift of a Wing of Finite Aspect Ratio", NACA
  Report 681, 1940 (two-exponential approximation to the Wagner function).
* W. R. Sears and B. O. Sparks, "On the Reaction of an Elastic Wing to
  Vertical Gusts", Journal of the Aeronautical Sciences 9(2), 1941, pp. 64-67
  (two-exponential approximation to the Kussner function; also in
  R. L. Bisplinghoff, H. Ashley and R. L. Halfman, "Aeroelasticity",
  Addison-Wesley, 1955, chapter 5).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ventorum.aero.system import (
    freestream_direction,
    ground_plane_from_condition,
    lift_direction,
    side_direction,
)
from ventorum.aero.vortex import kernel_cache_scope
from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    SolverResult,
    SolverSettings,
    TrustScore,
)
from ventorum.core.errors import InputError
from ventorum.unsteady.derivatives import (
    MIN_TRUST_WAKE_CHORDS,
    _build_unsteady_trust,
    _check_incompressible_mach,
    _check_no_steady_rates,
    _json_float,
)
from ventorum.unsteady.gusts import sharp_edged
from ventorum.unsteady.timedomain import _solve_time_rings

__all__ = [
    "IndicialResult",
    "TimeResponseResult",
    "indicial_response",
    "time_response",
]

_COEFFS = ("CL", "CD", "CY", "Cl", "Cm", "Cn")
_STORE_OPTIONS = ("loads", "strengths")
_INDICIAL_KINDS = ("alpha", "gust", "control")


def _pack_1d(arr: np.ndarray) -> list[float | None]:
    """Convert a 1-D numpy array to a strict JSON list of floats."""
    return [_json_float(float(v)) for v in arr]


def _pack_nd(arr: np.ndarray) -> Any:
    """Convert an N-D numpy array to nested strict JSON lists of floats."""
    if arr.ndim == 1:
        return _pack_1d(arr)
    return [_pack_nd(sub) for sub in arr]


def _pack_trust(trust: TrustScore | None) -> dict[str, Any] | None:
    """Return a strict JSON-compatible dictionary for *trust*, or None."""
    if trust is None:
        return None
    return {
        "score": _json_float(trust.score),
        "rating": str(trust.rating),
        "warnings": list(trust.warnings),
        "recommendations": list(trust.recommendations),
    }


def _validate_positive_float(value: Any, name: str) -> float:
    """Validate *value* as a finite positive float (not bool)."""
    if isinstance(value, bool):
        raise InputError(f"{name} must be a positive finite number, got {value!r}.")
    try:
        val = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be a positive finite number, got {value!r}.") from exc
    if not math.isfinite(val) or val <= 0.0:
        raise InputError(f"{name} must be finite and > 0, got {value!r}.")
    return val


def _validate_n_steps(value: Any) -> int:
    """Validate *value* as an integer >= 1 (not bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise InputError(f"n_steps must be an integer >= 1, got {value!r}.")
    n_val = int(value)
    if n_val < 1:
        raise InputError(f"n_steps must be >= 1, got {n_val}.")
    return n_val


def _prepare_setup(
    geometry: Aircraft | LiftingSurface,
    condition: FlightCondition,
    settings: SolverSettings | None,
    *,
    caller: str,
) -> tuple[Aircraft, SolverSettings, Any, dict[str, Any], np.ndarray, Any, Any]:
    """Validate inputs, build the VLM lattice and ground plane, and solve the steady mean state."""
    from ventorum.geometry import lattice_cache
    from ventorum.solvers.factory import make_solver, resolve_solver_type
    from ventorum.utils.deprecation import canonical_settings
    from ventorum.utils.validation import (
        validate_aircraft,
        validate_flight_condition,
        validate_solver_settings,
    )

    ac = Aircraft(name="SingleWing", surfaces=[geometry]) if isinstance(geometry, LiftingSurface) else geometry
    ac = ac.clone()
    sett = SolverSettings() if settings is None else settings.clone()
    sett = canonical_settings(sett)
    validate_solver_settings(sett)
    canonical = resolve_solver_type(sett.solver_type)
    if canonical != "vlm":
        raise InputError(
            f"{caller} supports only solver_type='vlm' (or 'auto'), got {sett.solver_type!r}."
        )

    _check_incompressible_mach(condition)
    validate_flight_condition(condition)
    _check_no_steady_rates(condition, caller)
    validate_aircraft(ac)

    solver = make_solver("vlm")
    fp = lattice_cache.surfaces_fingerprint(ac)
    geo = lattice_cache.geometry_info(fp, ac)
    ac.compute_reference_values(auto=geo["auto_ref"])
    rp = np.asarray(ac.moment_reference(), dtype=np.float64)

    lattice = solver.build(ac, sett, condition, None, rp, fingerprint=fp)
    ground = ground_plane_from_condition(lattice, condition, rp)
    return ac, sett, solver, geo, rp, lattice, ground


def _solve_steady_base(
    solver: Any,
    lattice: Any,
    condition: FlightCondition,
    sett: SolverSettings,
    ac: Aircraft,
    rp: np.ndarray,
    main_surface: Any,
) -> tuple[SolverResult, SolverSettings]:
    """Run the steady VLM base solve on *lattice* to preserve steady trust metadata."""
    fixed_sett = sett.clone()
    fixed_sett.n_chord = int(lattice.n_chord)
    with kernel_cache_scope(lattice):
        base = solver.solve_batch(
            lattice,
            [condition],
            fixed_sett,
            float(ac.S_ref),
            float(ac.b_ref),
            float(ac.c_ref),
            ref_point=rp,
            main_surface=main_surface,
            continuation=False,
        )[0]
    base.condition = condition
    return base, fixed_sett


def _default_dt_for_lattice(lattice: Any, V_inf: float) -> float:
    """Return the default convective time step ``dx_TE_mean / V_inf`` [s] for *lattice*."""
    n_c = int(lattice.n_chord)
    n_s = int(lattice.n_strips)
    te_idx = np.arange(n_s, dtype=np.int64) * n_c + (n_c - 1)
    dx_te = np.asarray(lattice.panel_length[te_idx], dtype=np.float64)
    w = np.asarray(lattice.width, dtype=np.float64)
    dx_te_mean = float(np.sum(w * dx_te) / np.sum(w))
    return dx_te_mean / float(V_inf)


def _device_execution_notes(
    device: str,
    precision: str,
    solver_notes: Sequence[str] = (),
) -> list[str]:
    """Return execution-device notes for the time-domain vortex-ring march."""
    if device == "gpu":
        return [f"Time-domain vortex-ring marching ran on the GPU in {precision}."]
    if any("GPU unavailable" in note or "GPU out of memory" in note for note in solver_notes):
        return []
    return [f"Time-domain vortex-ring marching ran on the CPU in {precision}."]


@dataclass
class TimeResponseResult:
    """Result of :func:`time_response`.

    Force coefficients ``(CL, CD, CY)`` are along the mean wind axes (lift,
    free stream, side). Moment coefficients ``(Cl, Cm, Cn)`` are in body axes
    of the mean state (``Cl > 0`` right wing down, ``Cm > 0`` nose up,
    ``Cn > 0`` nose right) about the instantaneous reference point. All
    coefficients are normalised by the nominal ``V_inf`` [m/s] and ``rho``
    [kg/m^3] of the mean flight condition.

    Attributes
    ----------
    time : numpy.ndarray
        Time instants ``t[n] = n * dt`` [s], shape ``(n_steps + 1,)``.
    s : numpy.ndarray
        Nondimensional distance travelled in semichords
        ``2 * V_inf * t / c_ref`` [-], shape ``(n_steps + 1,)``.
    x, y, z : numpy.ndarray
        Prescribed rigid-body translations of ``ref_point`` along mean-flight
        axes ``(e_x, e_y, e_z)`` [m], shape ``(n_steps + 1,)``.
    roll, pitch, yaw : numpy.ndarray
        Prescribed rigid-body 3-2-1 Euler rotations about ``ref_point`` [rad]
        (AVL signs), shape ``(n_steps + 1,)``.
    x_dot, y_dot, z_dot : numpy.ndarray
        Time derivatives of ``(x, y, z)`` [m/s], shape ``(n_steps + 1,)``.
    roll_dot, pitch_dot, yaw_dot : numpy.ndarray
        Time derivatives of ``(roll, pitch, yaw)`` [rad/s], shape
        ``(n_steps + 1,)``.
    controls : dict[str, numpy.ndarray]
        Prescribed control-surface deflection histories [rad], each of shape
        ``(n_steps + 1,)``.
    controls_dot : dict[str, numpy.ndarray]
        Time derivatives of control-surface deflections [rad/s], each of shape
        ``(n_steps + 1,)``.
    gust_velocity : numpy.ndarray
        Gust velocity at the instantaneous reference point in geometry axes
        [m/s], shape ``(n_steps + 1, 3)``.
    CL, CD, CY : numpy.ndarray
        Force coefficients along mean wind axes [-], shape ``(n_steps + 1,)``.
    Cl, Cm, Cn : numpy.ndarray
        Moment coefficients in body axes of the mean state [-], shape
        ``(n_steps + 1,)``.
    strip_cl : numpy.ndarray
        Per-strip section lift coefficient per step [-], shape
        ``(n_steps + 1, n_strips)``.
    gamma_bound : numpy.ndarray or None
        Bound vortex-ring circulations at the last step [m^2/s], shape
        ``(n_panels,)`` when ``store="strengths"``, else None.
    gamma_wake : numpy.ndarray or None
        Wake vortex-ring circulations at the last step [m^2/s], shape
        ``(n_steps, n_strips)`` when ``store="strengths"``, else None.
    wake_corners : numpy.ndarray or None
        Wake vortex-ring corner coordinates at the last step [m], shape
        ``(n_steps, n_strips, 4, 3)`` when ``store="strengths"``, else None.
    store : str
        Storage mode used: ``"loads"`` or ``"strengths"``.
    dt : float
        Time step [s].
    dt_default : float
        Default convective time step ``dx_te_mean / V_inf`` [s].
    dx_w : float
        Wake row convective length ``V_inf * dt`` [m].
    dx_te_mean : float
        Strip-width-weighted mean trailing-edge panel length [m].
    n_steps : int
        Number of time steps marched [-].
    n_panels : int
        Number of bound vortex rings [-].
    n_strips : int
        Number of spanwise strips [-].
    n_chord : int
        Number of chordwise panels per strip [-].
    in_ground_effect : bool
        True when solved over a flat ground plane.
    h_m : float or None
        Height of ``ref_point`` above the ground [m], or None in free air.
    S_ref : float
        Reference area [m^2].
    b_ref : float
        Reference span [m].
    c_ref : float
        Reference chord [m].
    ref_point : numpy.ndarray
        Mean moment reference point [m], shape ``(3,)``.
    V_inf : float
        Nominal free-stream speed of the mean flight condition [m/s].
    rho : float
        Nominal air density of the mean flight condition [kg/m^3].
    base : SolverResult
        Steady vortex-lattice result of the mean state.
    trust : TrustScore or None
        Trust score of the mean state with unsteady warnings added.
    trust_notes : list[str]
        Unsteady trust warnings and notes triggered for this run.
    notes : list[str]
        Solver, device, gust and trust notes.
    device : str
        Execution device of the time march (``"cpu"`` or ``"gpu"``).
    precision : str
        Floating-point precision of the time march (``"float64"`` or
        ``"float32"``).
    """

    time: np.ndarray
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    roll: np.ndarray
    pitch: np.ndarray
    yaw: np.ndarray
    x_dot: np.ndarray
    y_dot: np.ndarray
    z_dot: np.ndarray
    roll_dot: np.ndarray
    pitch_dot: np.ndarray
    yaw_dot: np.ndarray
    controls: dict[str, np.ndarray]
    controls_dot: dict[str, np.ndarray]
    gust_velocity: np.ndarray
    CL: np.ndarray
    CD: np.ndarray
    CY: np.ndarray
    Cl: np.ndarray
    Cm: np.ndarray
    Cn: np.ndarray
    strip_cl: np.ndarray
    gamma_bound: np.ndarray | None
    gamma_wake: np.ndarray | None
    wake_corners: np.ndarray | None
    store: str
    dt: float
    dt_default: float
    dx_w: float
    dx_te_mean: float
    n_steps: int
    n_panels: int
    n_strips: int
    n_chord: int
    in_ground_effect: bool
    h_m: float | None
    S_ref: float
    b_ref: float
    c_ref: float
    ref_point: np.ndarray
    V_inf: float
    rho: float
    base: SolverResult
    trust: TrustScore | None
    trust_notes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    device: str = "cpu"
    precision: str = "float64"

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible dictionary of the result."""
        out: dict[str, Any] = {
            "time": _pack_1d(self.time),
            "s": _pack_1d(self.s),
            "CL": _pack_1d(self.CL),
            "CD": _pack_1d(self.CD),
            "CY": _pack_1d(self.CY),
            "Cl": _pack_1d(self.Cl),
            "Cm": _pack_1d(self.Cm),
            "Cn": _pack_1d(self.Cn),
            "strip_cl": _pack_nd(self.strip_cl),
            "motion": {
                "x_m": _pack_1d(self.x),
                "y_m": _pack_1d(self.y),
                "z_m": _pack_1d(self.z),
                "roll_rad": _pack_1d(self.roll),
                "pitch_rad": _pack_1d(self.pitch),
                "yaw_rad": _pack_1d(self.yaw),
                "x_dot_m_s": _pack_1d(self.x_dot),
                "y_dot_m_s": _pack_1d(self.y_dot),
                "z_dot_m_s": _pack_1d(self.z_dot),
                "roll_dot_rad_s": _pack_1d(self.roll_dot),
                "pitch_dot_rad_s": _pack_1d(self.pitch_dot),
                "yaw_dot_rad_s": _pack_1d(self.yaw_dot),
            },
            "controls": {name: _pack_1d(arr) for name, arr in self.controls.items()},
            "controls_dot": {name: _pack_1d(arr) for name, arr in self.controls_dot.items()},
            "gust_velocity": _pack_nd(self.gust_velocity),
            "store": self.store,
            "n_steps": int(self.n_steps),
            "dt_s": _json_float(self.dt),
            "dt_default_s": _json_float(self.dt_default),
            "dx_w_m": _json_float(self.dx_w),
            "dx_te_mean_m": _json_float(self.dx_te_mean),
            "in_ground_effect": bool(self.in_ground_effect),
            "h_m": None if self.h_m is None else _json_float(self.h_m),
            "S_ref_m2": _json_float(self.S_ref),
            "b_ref_m": _json_float(self.b_ref),
            "c_ref_m": _json_float(self.c_ref),
            "ref_point_m": _pack_1d(self.ref_point),
            "V_inf_m_s": _json_float(self.V_inf),
            "rho_kg_m3": _json_float(self.rho),
            "mesh": {
                "n_panels": int(self.n_panels),
                "n_strips": int(self.n_strips),
                "n_chord": int(self.n_chord),
            },
            "device": str(self.device),
            "precision": str(self.precision),
            "trust": _pack_trust(self.trust),
            "trust_notes": list(self.trust_notes),
            "notes": list(self.notes),
        }
        if self.store == "strengths" and self.gamma_bound is not None:
            out["gamma_bound"] = _pack_1d(self.gamma_bound)
            out["gamma_wake"] = _pack_nd(self.gamma_wake) if self.gamma_wake is not None else None
            out["wake_corners"] = _pack_nd(self.wake_corners) if self.wake_corners is not None else None
        return out


@dataclass
class IndicialResult:
    """Result of :func:`indicial_response`.

    All six aerodynamic coefficient histories and ``strip_cl`` are divided by
    the step ``amplitude`` and reported per unit step [1/rad] (per radian of
    incidence step for ``kind="alpha"``, per unit ``w_g / V_inf`` for
    ``kind="gust"``, and per radian of control deflection for
    ``kind="control"``).

    Attributes
    ----------
    kind : str
        Indicial response kind: ``"alpha"`` (Wagner), ``"gust"`` (Kussner), or
        ``"control"`` (control-surface step).
    s : numpy.ndarray
        Nondimensional distance travelled in semichords
        ``2 * V_inf * t / c_ref`` [-], shape ``(n_steps + 1,)``.
    time : numpy.ndarray
        Physical time instants ``t`` [s], shape ``(n_steps + 1,)``.
    amplitude : float
        Step amplitude used [rad] (or [-] for ``w_g / V_inf``).
    control : str or None
        Named control surface when ``kind="control"``, else None.
    CL, CD, CY : numpy.ndarray
        Force-coefficient indicial responses per unit step [1/rad], shape
        ``(n_steps + 1,)``.
    Cl, Cm, Cn : numpy.ndarray
        Moment-coefficient indicial responses per unit step [1/rad] in mean
        body axes, shape ``(n_steps + 1,)``.
    strip_cl : numpy.ndarray
        Per-strip section lift indicial response per unit step [1/rad], shape
        ``(n_steps + 1, n_strips)``.
    steady : dict[str, float] or None
        Steady final derivatives ``{CL, CD, CY, Cl, Cm, Cn}`` [1/rad] from the
        steady VLM solver on the same lattice for ``kind="alpha"`` and
        ``kind="control"``; None for ``kind="gust"``. For ``kind="alpha"``
        they are the long-time limit of the step: the same onset at fixed
        geometry (and fixed ground), forces on the mean wind axes.
    dt : float
        Time step [s].
    n_steps : int
        Number of time steps marched [-].
    n_panels : int
        Number of bound vortex rings [-].
    n_strips : int
        Number of spanwise strips [-].
    n_chord : int
        Number of chordwise panels per strip [-].
    in_ground_effect : bool
        True when solved over a flat ground plane.
    h_m : float or None
        Height of ``ref_point`` above the ground [m], or None in free air.
    S_ref : float
        Reference area [m^2].
    b_ref : float
        Reference span [m].
    c_ref : float
        Reference chord [m].
    ref_point : numpy.ndarray
        Moment reference point [m], shape ``(3,)``.
    base : SolverResult
        Steady vortex-lattice result of the mean state.
    trust : TrustScore or None
        Trust score of the mean state with unsteady warnings added.
    trust_notes : list[str]
        Unsteady trust warnings and notes triggered for this run.
    notes : list[str]
        Solver, device, gust and trust notes.
    device : str
        Execution device (``"cpu"`` or ``"gpu"``).
    precision : str
        Floating-point precision (``"float64"`` or ``"float32"``).
    """

    kind: str
    s: np.ndarray
    time: np.ndarray
    amplitude: float
    control: str | None
    CL: np.ndarray
    CD: np.ndarray
    CY: np.ndarray
    Cl: np.ndarray
    Cm: np.ndarray
    Cn: np.ndarray
    strip_cl: np.ndarray
    steady: dict[str, float] | None
    dt: float
    n_steps: int
    n_panels: int
    n_strips: int
    n_chord: int
    in_ground_effect: bool
    h_m: float | None
    S_ref: float
    b_ref: float
    c_ref: float
    ref_point: np.ndarray
    base: SolverResult
    trust: TrustScore | None
    trust_notes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    device: str = "cpu"
    precision: str = "float64"

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible dictionary of the result."""
        return {
            "kind": self.kind,
            "s": _pack_1d(self.s),
            "time": _pack_1d(self.time),
            "amplitude": _json_float(self.amplitude),
            "control": self.control,
            "CL": _pack_1d(self.CL),
            "CD": _pack_1d(self.CD),
            "CY": _pack_1d(self.CY),
            "Cl": _pack_1d(self.Cl),
            "Cm": _pack_1d(self.Cm),
            "Cn": _pack_1d(self.Cn),
            "strip_cl": _pack_nd(self.strip_cl),
            "steady": None if self.steady is None else {k: _json_float(v) for k, v in self.steady.items()},
            "dt_s": _json_float(self.dt),
            "n_steps": int(self.n_steps),
            "in_ground_effect": bool(self.in_ground_effect),
            "h_m": None if self.h_m is None else _json_float(self.h_m),
            "S_ref_m2": _json_float(self.S_ref),
            "b_ref_m": _json_float(self.b_ref),
            "c_ref_m": _json_float(self.c_ref),
            "ref_point_m": _pack_1d(self.ref_point),
            "mesh": {
                "n_panels": int(self.n_panels),
                "n_strips": int(self.n_strips),
                "n_chord": int(self.n_chord),
            },
            "device": str(self.device),
            "precision": str(self.precision),
            "trust": _pack_trust(self.trust),
            "trust_notes": list(self.trust_notes),
            "notes": list(self.notes),
        }


def time_response(
    geometry: Aircraft | LiftingSurface,
    condition: FlightCondition,
    settings: SolverSettings | None = None,
    *,
    t_end: float | None = None,
    n_steps: int | None = None,
    dt: float | None = None,
    motion: Mapping[str, Sequence[float] | np.ndarray | Callable[[float], float]] | None = None,
    controls: Mapping[str, Sequence[float] | np.ndarray | Callable[[float], float]] | None = None,
    gust: Callable[[np.ndarray, float], np.ndarray] | None = None,
    store: str = "loads",
) -> TimeResponseResult:
    """Compute the time-domain aerodynamic response using prescribed-wake vortex rings.

    Marches a lifting surface or aircraft in incompressible flow (Mach <= 0.3)
    in free air or over a flat ground plane (``condition.h`` set), with
    optional six-axis prescribed motion, moving control surfaces and convected
    gust or wind-shear fields. Coefficients ``(CL, CD, CY)`` (mean wind axes)
    and ``(Cl, Cm, Cn)`` (mean body axes) use the nominal ``V_inf`` [m/s] and
    ``rho`` [kg/m^3] of *condition*.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Geometry to analyse. It is not modified. Rotations and moments are
        about ``Aircraft.ref_point`` [m] (default the origin).
    condition : FlightCondition
        Mean flight state: ``V_inf`` [m/s], ``rho`` [kg/m^3], ``alpha`` [rad],
        ``beta`` [rad], ``phi`` [rad], and optional ground height ``h`` [m].
        Must satisfy Mach <= 0.3. The body rates ``p``, ``q``, ``r`` must be 0.
    settings : SolverSettings or None, optional
        Solver settings (default :class:`SolverSettings`). Only the
        vortex-lattice solver (``"vlm"`` or ``"auto"``) is supported.
    t_end : float or None, optional
        End time [s] (must be ``> 0``). Give exactly one of *t_end* or
        *n_steps*.
    n_steps : int or None, optional
        Number of time steps to march (integer ``>= 1``). Give exactly one of
        *t_end* or *n_steps*.
    dt : float or None, optional
        Time step [s] (must be ``> 0``). None uses the solver default
        ``dx_TE / V_inf`` [s].
    motion : mapping of str to array or callable, or None, optional
        Prescribed six-axis rigid-body displacements of ``ref_point``:
        ``'x'``, ``'y'``, ``'z'`` [m] along mean-flight axes and ``'roll'``,
        ``'pitch'``, ``'yaw'`` [rad] (3-2-1 sequence, AVL signs). Each entry
        is a 1-D array of length ``n_steps + 1`` or a callable of time ``t``
        [s].
    controls : mapping of str to array or callable, or None, optional
        Prescribed control-surface deflection histories [rad], keyed by
        ``ControlSurface.name``. Each entry is a 1-D array of length
        ``n_steps + 1`` or a callable of time ``t`` [s].
    gust : callable or None, optional
        Gust velocity field ``gust(points, t) -> velocities`` where *points*
        has shape ``(n, 3)`` [m] in geometry axes and the return value has
        shape ``(n, 3)`` [m/s] (see :mod:`ventorum.unsteady.gusts`). On the
        GPU, a gust callable is evaluated on the host at every step and its
        velocities are copied to the device, so it adds a host cost per step.
    store : {'loads', 'strengths'}, optional
        ``"loads"`` (default) stores time histories and ``strip_cl`` only.
        ``"strengths"`` also stores final ``gamma_bound`` [m^2/s],
        ``gamma_wake`` [m^2/s] and ``wake_corners`` [m].

    Returns
    -------
    TimeResponseResult
        Time histories of motion, controls, gust velocity at ``ref_point``,
        aerodynamic coefficients, trust metadata and diagnostic notes.

    Raises
    ------
    InputError
        If both or neither of *t_end* and *n_steps* are given; if *dt*,
        *store*, *motion*, *controls* or *gust* are invalid; if
        ``settings.solver_type`` is not VLM; if Mach > 0.3; or if the wing or
        deflected flap reaches or crosses the ground plane at any step.
    GroundStrikeError
        If the mean geometry touches or crosses the ground plane when
        ``condition.h`` is set.
    """
    if (t_end is None) == (n_steps is None):
        raise InputError("Give exactly one of t_end [s] or n_steps (not both and not neither).")
    if store not in _STORE_OPTIONS:
        raise InputError(f"store must be one of {_STORE_OPTIONS}, got {store!r}.")
    dt_val = None if dt is None else _validate_positive_float(dt, "dt")
    t_end_val = None if t_end is None else _validate_positive_float(t_end, "t_end")
    n_steps_val = None if n_steps is None else _validate_n_steps(n_steps)

    ac, sett, solver, geo, rp, lattice, ground = _prepare_setup(
        geometry, condition, settings, caller="time_response",
    )

    dt_default = _default_dt_for_lattice(lattice, float(condition.V_inf))
    dt_eff = dt_default if dt_val is None else dt_val
    if n_steps_val is None:
        assert t_end_val is not None
        n_steps_val = max(1, int(math.ceil(t_end_val / dt_eff - 1.0e-12)))

    resp = _solve_time_rings(
        lattice,
        condition,
        S_ref=float(ac.S_ref),
        b_ref=float(ac.b_ref),
        c_ref=float(ac.c_ref),
        ref_point=rp,
        motion=motion,
        controls=controls,
        gust=gust,
        n_steps=n_steps_val,
        dt=dt_val,
        ground=ground,
        wake_alignment=sett.wake_alignment,
    )

    base, _ = _solve_steady_base(solver, lattice, condition, sett, ac, rp, geo["main"])

    te_idx = np.arange(resp.n_strips, dtype=np.int64) * resp.n_chord + (resp.n_chord - 1)
    dx_te = np.asarray(lattice.panel_length[te_idx], dtype=np.float64)
    dx_te_ratio = dx_te / resp.dx_w
    trust, trust_notes = _build_unsteady_trust(
        base.totals.trust,
        n_chord=resp.n_chord,
        dx_te_ratio=dx_te_ratio,
        k_values=np.array([0.0], dtype=np.float64),
        wake_chords=MIN_TRUST_WAKE_CHORDS,
    )

    notes: list[str] = [
        "Force coefficients (CL, CD, CY) are in mean wind axes and moment coefficients (Cl, Cm, Cn) "
        "are in body axes of the mean state, normalised by the nominal V_inf and rho of condition.",
    ]
    notes.extend(_device_execution_notes(resp.device, resp.precision, resp.notes))
    notes.extend(trust_notes)
    for note in resp.notes:
        if note not in notes:
            notes.append(note)
        if trust is not None and note not in trust.warnings:
            trust.warnings.append(note)

    s_arr = (2.0 * float(condition.V_inf) / float(ac.c_ref)) * resp.time
    keep_str = store == "strengths"

    return TimeResponseResult(
        time=resp.time,
        s=s_arr,
        x=resp.x,
        y=resp.y,
        z=resp.z,
        roll=resp.roll,
        pitch=resp.pitch,
        yaw=resp.yaw,
        x_dot=resp.x_dot,
        y_dot=resp.y_dot,
        z_dot=resp.z_dot,
        roll_dot=resp.roll_dot,
        pitch_dot=resp.pitch_dot,
        yaw_dot=resp.yaw_dot,
        controls=resp.controls,
        controls_dot=resp.controls_dot,
        gust_velocity=resp.gust_velocity,
        CL=resp.CL,
        CD=resp.CD,
        CY=resp.CY,
        Cl=resp.Cl,
        Cm=resp.Cm,
        Cn=resp.Cn,
        strip_cl=resp.strip_cl,
        gamma_bound=resp.gamma_bound if keep_str else None,
        gamma_wake=resp.gamma_wake if keep_str else None,
        wake_corners=resp.wake_corners if keep_str else None,
        store=store,
        dt=float(resp.dt),
        dt_default=float(resp.dt_default),
        dx_w=float(resp.dx_w),
        dx_te_mean=float(resp.dx_te_mean),
        n_steps=int(resp.n_steps),
        n_panels=int(resp.n_panels),
        n_strips=int(resp.n_strips),
        n_chord=int(resp.n_chord),
        in_ground_effect=ground is not None,
        h_m=None if condition.h is None else float(condition.h),
        S_ref=float(ac.S_ref),
        b_ref=float(ac.b_ref),
        c_ref=float(ac.c_ref),
        ref_point=rp.copy(),
        V_inf=float(condition.V_inf),
        rho=float(condition.rho),
        base=base,
        trust=trust,
        trust_notes=trust_notes,
        notes=notes,
        device=str(resp.device),
        precision=str(resp.precision),
    )


def _indicial_alpha_steady(
    solver: Any,
    lattice: Any,
    condition: FlightCondition,
    settings: SolverSettings,
    ac: Aircraft,
    rp: np.ndarray,
    main_surface: int,
    ground: Any,
) -> dict[str, float]:
    """Return the steady limit of the indicial alpha step per radian.

    The step tilts the onset at fixed geometry, and the time march projects
    the forces on the fixed mean wind axes. The steady solves at
    ``alpha +/- d_a`` with the same lattice have the same onset. In ground
    effect they keep the mean ground plane, so the wing does not rotate
    against the ground. Their forces are projected on the mean wind axes
    (so ``CD`` includes ``-CL0`` and ``CL`` includes ``+CD0``). The moments are
    in body axes and need no projection.
    """
    d_a = 1.0e-4
    alpha0 = float(condition.alpha)
    beta0 = float(condition.beta)
    axes0 = (
        freestream_direction(alpha0, beta0),
        side_direction(alpha0, beta0),
        lift_direction(alpha0),
    )
    conds = [
        FlightCondition(
            V_inf=condition.V_inf, rho=condition.rho, alpha=alpha0 + sgn * d_a,
            beta=condition.beta, phi=condition.phi, h=condition.h,
        )
        for sgn in (1.0, -1.0)
    ]
    S_ref, b_ref, c_ref = float(ac.S_ref), float(ac.b_ref), float(ac.c_ref)
    with kernel_cache_scope(lattice):
        if ground is None:
            results = solver.solve_batch(
                lattice, conds, settings, S_ref, b_ref, c_ref,
                ref_point=rp, main_surface=main_surface, continuation=False,
            )
        else:
            results = [
                solver.solve_lattice(
                    lattice, c, settings, S_ref, b_ref, c_ref,
                    ground=ground, ref_point=rp, main_surface=main_surface,
                )
                for c in conds
            ]

    def _fixed_axes(res: Any, cond: FlightCondition) -> np.ndarray:
        t = res.totals
        a, b = float(cond.alpha), float(cond.beta)
        force = (
            float(t.CDi_nearfield) * freestream_direction(a, b)
            + float(t.CY) * side_direction(a, b)
            + float(t.CL) * lift_direction(a)
        )
        return np.array([
            force @ axes0[2], force @ axes0[0], force @ axes0[1],
            float(t.Cl), float(t.Cm), float(t.Cn),
        ])

    diff = (_fixed_axes(results[0], conds[0]) - _fixed_axes(results[1], conds[1])) / (2.0 * d_a)
    return {key: float(val) for key, val in zip(("CL", "CD", "CY", "Cl", "Cm", "Cn"), diff, strict=True)}


def indicial_response(
    geometry: Aircraft | LiftingSurface,
    condition: FlightCondition,
    settings: SolverSettings | None = None,
    *,
    kind: str,
    s_end: float = 20.0,
    amplitude: float = 1.0e-4,
    control: str | None = None,
) -> IndicialResult:
    """Compute unit-step indicial aerodynamic responses as functions of semichords ``s``.

    Marches a small step perturbation from ``t = 0`` to
    ``t_end = s_end * c_ref / (2 * V_inf)`` [s] at the default convective time
    step ``dt = dx_TE / V_inf`` [s] and divides the response (minus the
    unperturbed mean-state transient when the mean state is loaded) by
    *amplitude*.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Geometry to analyse. It is not modified.
    condition : FlightCondition
        Mean flight state: ``V_inf`` [m/s], ``rho`` [kg/m^3], ``alpha`` [rad],
        ``beta`` [rad], ``phi`` [rad], and optional ground height ``h`` [m].
        Must satisfy Mach <= 0.3. The body rates ``p``, ``q``, ``r`` must be 0.
    settings : SolverSettings or None, optional
        Solver settings (default :class:`SolverSettings`). Only the
        vortex-lattice solver (``"vlm"`` or ``"auto"``) is supported.
    kind : {'alpha', 'gust', 'control'}
        Indicial response kind:

        * ``"alpha"`` (Wagner): uniform onset step
          ``V_inf cos(beta) lift_direction(alpha) amplitude`` [m/s] over the
          whole wing at ``t = 0`` (an incidence step at fixed geometry).
        * ``"gust"`` (Kussner): sharp-edged vertical gust front starting at
          the foremost leading edge at ``t = 0`` with ``w_g = V_inf * amplitude``
          [m/s].
        * ``"control"``: step deflection of *control* by ``amplitude`` [rad]
          at ``t = 0``.
    s_end : float, optional
        Final nondimensional distance travelled ``2 * V_inf * t / c_ref`` [-]
        (must be ``> 0``). Default ``20.0`` (10 reference chords).
    amplitude : float, optional
        Step size [rad] (or dimensionless ``w_g / V_inf`` [-] for
        ``kind="gust"``), must be ``> 0``. Default ``1e-4``.
    control : str or None, optional
        Control surface name. Required when ``kind="control"`` and refused for
        other kinds.

    Returns
    -------
    IndicialResult
        Semichord grid ``s`` [-], six unit-step coefficient histories [1/rad],
        ``strip_cl`` [1/rad], steady final derivatives [1/rad] (for
        ``"alpha"`` and ``"control"``), trust metadata and notes.

    Raises
    ------
    InputError
        If *kind*, *s_end*, *amplitude* or *control* are invalid; if
        ``settings.solver_type`` is not VLM; or if Mach > 0.3.
    """
    if not isinstance(kind, str) or kind not in _INDICIAL_KINDS:
        raise InputError(f"kind must be one of {_INDICIAL_KINDS}, got {kind!r}.")
    s_end_val = _validate_positive_float(s_end, "s_end")
    amp_val = _validate_positive_float(amplitude, "amplitude")
    if kind == "control":
        if not isinstance(control, str) or not control.strip():
            raise InputError("kind='control' requires a non-empty 'control' name.")
    elif control is not None:
        raise InputError(f"control={control!r} is only accepted when kind='control'.")

    ac, sett, solver, geo, rp, lattice, ground = _prepare_setup(
        geometry, condition, settings, caller="indicial_response",
    )

    V_inf = float(condition.V_inf)
    c_ref = float(ac.c_ref)
    t_end = s_end_val * c_ref / (2.0 * V_inf)
    dt_default = _default_dt_for_lattice(lattice, V_inf)
    n_steps = max(1, int(math.ceil(t_end / dt_default - 1.0e-12)))

    base, fixed_sett = _solve_steady_base(solver, lattice, condition, sett, ac, rp, geo["main"])

    # Determine whether the mean state carries nonzero steady lift/circulation.
    has_mean_load = bool(
        abs(float(base.totals.CL)) > 1.0e-13
        or abs(float(base.totals.Cm)) > 1.0e-13
        or abs(float(base.totals.CY)) > 1.0e-13
        or abs(float(base.totals.Cl)) > 1.0e-13
        or abs(float(base.totals.Cn)) > 1.0e-13
        or any(
            float(np.max(np.abs(sp.gamma))) > 1.0e-13
            for sp in base.spanwise
            if getattr(sp, "gamma", None) is not None and len(sp.gamma) > 0
        )
    )

    d_0 = freestream_direction(float(condition.alpha), float(condition.beta))

    gust_arg: Callable[[np.ndarray, float], np.ndarray] | None = None
    ctrl_arg: dict[str, np.ndarray] | None = None
    steady_map: dict[str, float] | None = None
    extra_notes: list[str] = []

    if kind == "alpha":
        # The step is the first-order change of the free stream for d_alpha =
        # amplitude at fixed geometry: d(V_inf d(alpha, beta))/d_alpha =
        # V_inf cos(beta) lift_direction(alpha).
        beta0 = float(condition.beta)
        v_vec = (V_inf * amp_val * math.cos(beta0)) * lift_direction(float(condition.alpha))

        def _uniform_vertical(pts: np.ndarray, t: float) -> np.ndarray:
            _ = t
            return np.broadcast_to(v_vec[None, :], pts.shape).copy()

        gust_arg = _uniform_vertical
        steady_map = _indicial_alpha_steady(
            solver, lattice, condition, fixed_sett, ac, rp, geo["main"], ground,
        )
        if ground is not None:
            extra_notes.append(
                "Indicial kind='alpha' tilts the onset at fixed wing attitude and fixed ground; "
                "the steady values are the same input solved with the steady lattice."
            )
    elif kind == "gust":
        w_g = V_inf * amp_val
        le_pts = np.vstack([sf.edge_le for sf in lattice.surfaces])
        x_front = float(np.min(le_pts @ d_0))
        gust_arg = sharp_edged(w_g, "vertical", x_front=x_front)
        steady_map = None
    else:
        assert control is not None
        known = list(ac.control_names())
        if control not in known:
            raise InputError(f"Unknown control surface {control!r}; available controls: {known}.")
        # The step is an increment over the built deflection of each surface
        # with this control name (the surfaces may have different deflections).
        ctrl_arg = {control: np.full(n_steps + 1, amp_val, dtype=np.float64)}
        from ventorum.solvers.control_derivatives import control_derivatives

        cd_res = control_derivatives(ac, condition, fixed_sett, controls=[control], axes="body")
        cd_ctrl = cd_res.derivatives[control]
        steady_map = {
            "CL": float(cd_ctrl.CL),
            "CD": float(cd_ctrl.CD),
            "CY": float(cd_ctrl.CY),
            "Cl": float(cd_ctrl.Cl),
            "Cm": float(cd_ctrl.Cm),
            "Cn": float(cd_ctrl.Cn),
        }

    resp_pert = _solve_time_rings(
        lattice,
        condition,
        S_ref=float(ac.S_ref),
        b_ref=float(ac.b_ref),
        c_ref=float(ac.c_ref),
        ref_point=rp,
        controls=ctrl_arg,
        gust=gust_arg,
        n_steps=n_steps,
        ground=ground,
        wake_alignment=sett.wake_alignment,
        control_increments=True,
    )

    solver_notes = list(resp_pert.notes)
    if has_mean_load:
        resp_base = _solve_time_rings(
            lattice,
            condition,
            S_ref=float(ac.S_ref),
            b_ref=float(ac.b_ref),
            c_ref=float(ac.c_ref),
            ref_point=rp,
            n_steps=n_steps,
            ground=ground,
            wake_alignment=sett.wake_alignment,
        )
        for note in resp_base.notes:
            if note not in solver_notes:
                solver_notes.append(note)
        CL_ind = (resp_pert.CL - resp_base.CL) / amp_val
        CD_ind = (resp_pert.CD - resp_base.CD) / amp_val
        CY_ind = (resp_pert.CY - resp_base.CY) / amp_val
        Cl_ind = (resp_pert.Cl - resp_base.Cl) / amp_val
        Cm_ind = (resp_pert.Cm - resp_base.Cm) / amp_val
        Cn_ind = (resp_pert.Cn - resp_base.Cn) / amp_val
        strip_ind = (resp_pert.strip_cl - resp_base.strip_cl) / amp_val
    else:
        CL_ind = resp_pert.CL / amp_val
        CD_ind = resp_pert.CD / amp_val
        CY_ind = resp_pert.CY / amp_val
        Cl_ind = resp_pert.Cl / amp_val
        Cm_ind = resp_pert.Cm / amp_val
        Cn_ind = resp_pert.Cn / amp_val
        strip_ind = resp_pert.strip_cl / amp_val

    te_idx = np.arange(resp_pert.n_strips, dtype=np.int64) * resp_pert.n_chord + (resp_pert.n_chord - 1)
    dx_te = np.asarray(lattice.panel_length[te_idx], dtype=np.float64)
    dx_te_ratio = dx_te / resp_pert.dx_w
    trust, trust_notes = _build_unsteady_trust(
        base.totals.trust,
        n_chord=resp_pert.n_chord,
        dx_te_ratio=dx_te_ratio,
        k_values=np.array([0.0], dtype=np.float64),
        wake_chords=MIN_TRUST_WAKE_CHORDS,
    )

    notes: list[str] = [
        f"Indicial response kind='{kind}' per unit step (amplitude={amp_val:g}), "
        "normalised by the nominal V_inf and rho of condition.",
    ]
    notes.extend(_device_execution_notes(resp_pert.device, resp_pert.precision, solver_notes))
    notes.extend(extra_notes)
    notes.extend(trust_notes)
    for note in solver_notes:
        if note not in notes:
            notes.append(note)
        if trust is not None and note not in trust.warnings:
            trust.warnings.append(note)

    s_arr = (2.0 * V_inf / c_ref) * resp_pert.time

    return IndicialResult(
        kind=kind,
        s=s_arr,
        time=resp_pert.time,
        amplitude=amp_val,
        control=control,
        CL=CL_ind,
        CD=CD_ind,
        CY=CY_ind,
        Cl=Cl_ind,
        Cm=Cm_ind,
        Cn=Cn_ind,
        strip_cl=strip_ind,
        steady=steady_map,
        dt=float(resp_pert.dt),
        n_steps=int(resp_pert.n_steps),
        n_panels=int(resp_pert.n_panels),
        n_strips=int(resp_pert.n_strips),
        n_chord=int(resp_pert.n_chord),
        in_ground_effect=ground is not None,
        h_m=None if condition.h is None else float(condition.h),
        S_ref=float(ac.S_ref),
        b_ref=float(ac.b_ref),
        c_ref=float(ac.c_ref),
        ref_point=rp.copy(),
        base=base,
        trust=trust,
        trust_notes=trust_notes,
        notes=notes,
        device=str(resp_pert.device),
        precision=str(resp_pert.precision),
    )
