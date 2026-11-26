# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Public free-body flight interface of the time-domain vortex-ring solver.

:func:`free_flight` couples the prescribed-wake time-domain vortex-ring
lattice solver (:mod:`ventorum.unsteady.timedomain`) to the rigid-body
equations of motion (:mod:`ventorum.unsteady.freeflight`). The aircraft moves
freely in the degrees of freedom that the user selects, under the
aerodynamic loads, gravity, a point thrust force and moment, and control
deflections that can depend on time or on the state. The run is in free air
or in flat-ground effect, in incompressible flow (Mach <= 0.3).

Force coefficients ``(CL, CD, CY)`` are along the mean wind axes and moment
coefficients ``(Cl, Cm, Cn)`` are in the body axes of the mean state about the
instantaneous reference point, normalised by the nominal ``V_inf`` [m/s] and
``rho`` [kg/m^3] of the mean flight condition. The dimensional aerodynamic
loads at the centre of gravity are in the inertial mean-flight frame.

References
----------
* B. Etkin and L. D. Reid, "Dynamics of Flight: Stability and Control", 3rd
  ed., Wiley, 1996, chapter 4 (rigid-body equations of motion).
* B. L. Stevens and F. L. Lewis, "Aircraft Control and Simulation", 2nd ed.,
  Wiley, 2003, chapter 1 (rigid-body equations and Euler-angle kinematics).
* T. R. Kane and D. A. Levinson, "Dynamics: Theory and Applications",
  McGraw-Hill, 1985 (partial velocities and the generalised equations of
  motion).
* U. Kuttler and W. A. Wall, "Fixed-point fluid-structure interaction solvers
  with dynamic relaxation", Computational Mechanics 43, 2008, pp. 61-72
  (Aitken relaxation of the implicit coupling).
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, chapter 13 (time-domain vortex-ring lattice).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
    _json_float,
)
from ventorum.unsteady.freeflight import FlightState, _solve_free_flight
from ventorum.unsteady.response import (
    _STORE_OPTIONS,
    _default_dt_for_lattice,
    _device_execution_notes,
    _pack_1d,
    _pack_nd,
    _pack_trust,
    _prepare_setup,
    _solve_steady_base,
    _validate_n_steps,
    _validate_positive_float,
)
from ventorum.unsteady.timedomain import _MOTION_AXES

__all__ = [
    "FlightState",
    "FreeFlightResult",
    "free_flight",
]


@dataclass
class FreeFlightResult:
    """Result of :func:`free_flight`.

    The inertial frame is the mean-flight frame: it translates at the
    constant mean flight velocity, ``e_x`` is aft along the mean flight path,
    ``e_z`` is up (along the lift direction in free air, along the ground
    normal in ground effect) and ``e_y`` is right. Gravity acts along
    ``-e_z``. The pose is the displacement of ``ref_point`` and the 3-2-1
    Euler rotation about it (AVL signs: roll > 0 right wing down, pitch > 0
    nose up, yaw > 0 nose right).

    Attributes
    ----------
    time : numpy.ndarray
        Time instants [s], shape ``(n_steps + 1,)``.
    s : numpy.ndarray
        Distance travelled in semichords ``2 * V_inf * t / c_ref`` [-], shape
        ``(n_steps + 1,)``.
    x, y, z : numpy.ndarray
        Displacements of ``ref_point`` in the inertial frame [m], shape
        ``(n_steps + 1,)``.
    roll, pitch, yaw : numpy.ndarray
        3-2-1 Euler angles about ``ref_point`` [rad], shape ``(n_steps + 1,)``.
    x_dot, y_dot, z_dot : numpy.ndarray
        Rates of ``(x, y, z)`` [m/s], shape ``(n_steps + 1,)``.
    roll_dot, pitch_dot, yaw_dot : numpy.ndarray
        Euler-angle rates [rad/s], shape ``(n_steps + 1,)``.
    controls, controls_dot : dict[str, numpy.ndarray]
        Control deflection histories [rad] and their rates [rad/s], each of
        shape ``(n_steps + 1,)``.
    thrust_force : numpy.ndarray
        Thrust force in body axes [N], shape ``(n_steps + 1, 3)``.
    thrust_moment : numpy.ndarray
        Thrust moment about the centre of gravity in body axes, including
        ``(thrust_point - cg) x thrust_force`` [N m], shape
        ``(n_steps + 1, 3)``.
    gust_velocity : numpy.ndarray
        Gust velocity at the instantaneous reference point in geometry axes
        [m/s], shape ``(n_steps + 1, 3)``.
    CL, CD, CY : numpy.ndarray
        Force coefficients along the mean wind axes [-], shape
        ``(n_steps + 1,)``.
    Cl, Cm, Cn : numpy.ndarray
        Moment coefficients in the body axes of the mean state about
        ``ref_point`` [-], shape ``(n_steps + 1,)``.
    strip_cl : numpy.ndarray
        Section lift coefficient per strip [-], shape
        ``(n_steps + 1, n_strips)``.
    aero_force_cg, aero_moment_cg : numpy.ndarray
        Aerodynamic force [N] and moment about the centre of gravity [N m] in
        the inertial frame, shape ``(n_steps + 1, 3)``.
    kinetic_energy, potential_energy, energy : numpy.ndarray
        Kinetic energy, potential energy ``m g z_cg`` and their sum [J], shape
        ``(n_steps + 1,)``.
    aero_work, thrust_work, work : numpy.ndarray
        Cumulative work of the aerodynamic loads, of the thrust and of both on
        the free degrees of freedom [J], shape ``(n_steps + 1,)``.
    angular_momentum : numpy.ndarray
        Angular momentum about the centre of gravity in the inertial frame
        [kg m^2/s], shape ``(n_steps + 1, 3)``.
    initial_force_imbalance, initial_moment_imbalance : numpy.ndarray
        Net force [N] and net moment about the centre of gravity [N m] at
        ``t = 0`` in the inertial frame, shape ``(3,)``, with the
        impulsive-start load (no wake at ``t = 0``).
    initial_force_imbalance_body, initial_moment_imbalance_body : numpy.ndarray
        The same net force [N] and moment [N m] in body axes, shape ``(3,)``.
        A constant thrust equal to minus these values cancels the load at
        ``t = 0``. The start is impulsive, so this does not trim the steady
        state.
    steady_force_imbalance, steady_moment_imbalance : numpy.ndarray
        Net force ``F_aero + F_grav`` [N] and net moment about the centre of
        gravity [N m] of the steady vortex-lattice load of the mean state
        (the load of ``base``, with the induced drag ``CDi``), without
        thrust, in the inertial frame, shape ``(3,)``.
    steady_force_imbalance_body, steady_moment_imbalance_body : numpy.ndarray
        The same steady net force [N] and moment [N m] in body axes at the
        mean pose, shape ``(3,)``. A constant thrust equal to minus these
        values trims the mean state: after the starting transient the loads
        approach the steady load and the body stays near the mean state.
    subiterations : numpy.ndarray
        Coupling sub-iterations per step [-], shape ``(n_steps + 1,)``.
    residuals : numpy.ndarray
        Final relative state residual per step [-], shape ``(n_steps + 1,)``.
    dofs : tuple[str, ...]
        Free degrees of freedom.
    ground_contact : bool
        True when the run stopped because a wing or flap node reached the
        ground.
    ground_contact_step : int or None
        Step of the ground contact, or None.
    gamma_bound, gamma_wake, wake_corners : numpy.ndarray or None
        Bound ring circulations [m^2/s], wake ring circulations [m^2/s] and
        wake ring corners [m] at the last valid step when
        ``store="strengths"``, else None.
    store : str
        Storage mode: ``"loads"`` or ``"strengths"``.
    dt, dt_default, dx_w, dx_te_mean : float
        Time step [s], default convective time step [s], wake row length [m]
        and mean trailing-edge panel length [m].
    n_steps, n_panels, n_strips, n_chord : int
        Completed steps, bound rings, spanwise strips and chordwise panels.
    in_ground_effect : bool
        True when solved over a flat ground plane.
    h_m : float or None
        Height of ``ref_point`` above the ground at ``t = 0`` [m], or None.
    S_ref : float
        Reference area [m^2].
    b_ref : float
        Reference span [m].
    c_ref : float
        Reference chord [m].
    ref_point, cg : numpy.ndarray
        Reference point [m] and centre of gravity [m] in geometry axes, shape
        ``(3,)``.
    mass : float
        Mass [kg].
    inertia : numpy.ndarray
        Inertia tensor about the centre of gravity in geometry axes [kg m^2],
        shape ``(3, 3)``.
    gravity : float
        Gravitational acceleration [m/s^2].
    V_inf : float
        Nominal free-stream speed [m/s].
    rho : float
        Nominal air density [kg/m^3].
    base : SolverResult
        Steady vortex-lattice result of the mean state.
    trust : TrustScore or None
        Trust score of the mean state with the unsteady warnings added.
    trust_notes : list[str]
        Unsteady trust warnings of this run.
    notes : list[str]
        Solver, device, coupling, gust, ground-contact and trust notes.
    device, precision : str
        Execution device (``"cpu"`` or ``"gpu"``) and precision
        (``"float64"`` or ``"float32"``).
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
    thrust_force: np.ndarray
    thrust_moment: np.ndarray
    gust_velocity: np.ndarray
    CL: np.ndarray
    CD: np.ndarray
    CY: np.ndarray
    Cl: np.ndarray
    Cm: np.ndarray
    Cn: np.ndarray
    strip_cl: np.ndarray
    aero_force_cg: np.ndarray
    aero_moment_cg: np.ndarray
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    energy: np.ndarray
    aero_work: np.ndarray
    thrust_work: np.ndarray
    work: np.ndarray
    angular_momentum: np.ndarray
    initial_force_imbalance: np.ndarray
    initial_moment_imbalance: np.ndarray
    initial_force_imbalance_body: np.ndarray
    initial_moment_imbalance_body: np.ndarray
    steady_force_imbalance: np.ndarray
    steady_moment_imbalance: np.ndarray
    steady_force_imbalance_body: np.ndarray
    steady_moment_imbalance_body: np.ndarray
    subiterations: np.ndarray
    residuals: np.ndarray
    dofs: tuple[str, ...]
    ground_contact: bool
    ground_contact_step: int | None
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
    cg: np.ndarray
    mass: float
    inertia: np.ndarray
    gravity: float
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
            "thrust_force_N": _pack_nd(self.thrust_force),
            "thrust_moment_Nm": _pack_nd(self.thrust_moment),
            "gust_velocity": _pack_nd(self.gust_velocity),
            "aero_force_cg_N": _pack_nd(self.aero_force_cg),
            "aero_moment_cg_Nm": _pack_nd(self.aero_moment_cg),
            "energy": {
                "kinetic_J": _pack_1d(self.kinetic_energy),
                "potential_J": _pack_1d(self.potential_energy),
                "total_J": _pack_1d(self.energy),
                "aero_work_J": _pack_1d(self.aero_work),
                "thrust_work_J": _pack_1d(self.thrust_work),
                "work_J": _pack_1d(self.work),
            },
            "angular_momentum_kg_m2_s": _pack_nd(self.angular_momentum),
            "initial_imbalance": {
                "force_N": _pack_1d(self.initial_force_imbalance),
                "moment_Nm": _pack_1d(self.initial_moment_imbalance),
                "force_body_N": _pack_1d(self.initial_force_imbalance_body),
                "moment_body_Nm": _pack_1d(self.initial_moment_imbalance_body),
            },
            "steady_imbalance": {
                "force_N": _pack_1d(self.steady_force_imbalance),
                "moment_Nm": _pack_1d(self.steady_moment_imbalance),
                "force_body_N": _pack_1d(self.steady_force_imbalance_body),
                "moment_body_Nm": _pack_1d(self.steady_moment_imbalance_body),
            },
            "coupling": {
                "subiterations": [int(v) for v in np.asarray(self.subiterations).tolist()],
                "residuals": _pack_1d(self.residuals),
            },
            "dofs": list(self.dofs),
            "ground_contact": bool(self.ground_contact),
            "ground_contact_step": None if self.ground_contact_step is None else int(self.ground_contact_step),
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
            "cg_m": _pack_1d(self.cg),
            "mass_kg": _json_float(self.mass),
            "inertia_kg_m2": _pack_nd(self.inertia),
            "gravity_m_s2": _json_float(self.gravity),
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


def free_flight(
    geometry: Aircraft | LiftingSurface,
    condition: FlightCondition,
    settings: SolverSettings | None = None,
    *,
    mass: float,
    inertia: np.ndarray | Sequence[Sequence[float]],
    cg: np.ndarray | Sequence[float] | None = None,
    dofs: Sequence[str] = _MOTION_AXES,
    t_end: float | None = None,
    n_steps: int | None = None,
    dt: float | None = None,
    initial_state: Mapping[str, float] | None = None,
    motion: Mapping[str, Sequence[float] | np.ndarray | Callable[[float], float]] | None = None,
    controls: Mapping[str, Sequence[float] | np.ndarray | Callable[..., float]] | None = None,
    thrust: Callable[..., tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]] | None = None,
    thrust_point: np.ndarray | Sequence[float] | None = None,
    gust: Callable[[np.ndarray, float], np.ndarray] | None = None,
    gravity: float = 9.80665,
    store: str = "loads",
    coupling_tol: float = 1.0e-10,
    max_subiterations: int = 20,
) -> FreeFlightResult:
    """Compute the free-body flight response with the time-domain vortex-ring solver.

    The aircraft moves freely in the degrees of freedom *dofs* under the
    aerodynamic loads of the prescribed-wake vortex-ring lattice, gravity, a
    point thrust force and moment, and the control deflections. The other
    degrees of freedom follow *motion* (default zero). Each step couples the
    loads and the rigid-body equations implicitly with sub-iterations. The
    run is in free air, or over a flat ground plane when ``condition.h`` is
    set; when a wing or flap node reaches the ground, the run stops and the
    result holds the valid steps.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Geometry to analyse. It is not modified. The pose, the rotations and
        the coefficients use ``Aircraft.ref_point`` [m] (default the origin).
    condition : FlightCondition
        Mean flight state: ``V_inf`` [m/s], ``rho`` [kg/m^3], ``alpha``
        [rad], ``beta`` [rad], and optional ground height ``h`` [m]. The
        inertial frame translates at this mean velocity. Must satisfy
        Mach <= 0.3. The body rates ``p``, ``q``, ``r`` must be 0; give
        initial rates through *initial_state*.
    settings : SolverSettings or None, optional
        Solver settings (default :class:`SolverSettings`). Only the
        vortex-lattice solver (``"vlm"`` or ``"auto"``) is supported.
    mass : float
        Mass [kg] (finite, ``> 0``).
    inertia : array_like
        Inertia tensor about the centre of gravity in geometry axes
        [kg m^2], shape ``(3, 3)``, symmetric and positive definite.
    cg : array_like or None, optional
        Centre of gravity [m] in geometry axes, shape ``(3,)``. None uses
        ``Aircraft.ref_point``.
    dofs : sequence of str, optional
        Free degrees of freedom, a subset of ``('x', 'y', 'z', 'roll',
        'pitch', 'yaw')``. Default: all six.
    t_end : float or None, optional
        End time [s] (``> 0``). Give exactly one of *t_end* or *n_steps*.
    n_steps : int or None, optional
        Number of time steps (``>= 1``). Give exactly one of *t_end* or
        *n_steps*.
    dt : float or None, optional
        Time step [s] (``> 0``). None uses the solver default
        ``dx_TE / V_inf`` [s].
    initial_state : mapping of str to float or None, optional
        Initial displacements (keys ``'x'``, ..., ``'yaw'``) [m, rad] and
        rates (keys ``'x_dot'``, ..., ``'yaw_dot'``) [m/s, rad/s] of the
        free degrees of freedom. Missing entries are zero.
    motion : mapping of str to array or callable, or None, optional
        Prescribed displacements [m, rad] of the degrees of freedom that are
        not in *dofs*, each a 1-D array of length ``n_steps + 1`` or a
        callable of time ``t`` [s].
    controls : mapping of str to array or callable, or None, optional
        Control deflections [rad] keyed by ``ControlSurface.name``: a 1-D
        array of length ``n_steps + 1``, a callable ``fn(t)`` or a callable
        ``fn(t, state)`` that gets a :class:`FlightState`.
    thrust : callable or None, optional
        ``thrust(t) -> (force, moment)`` or ``thrust(t, state) -> (force,
        moment)``, with the force [N] and the moment [N m] in body axes
        (geometry axes that turn with the body), each of shape ``(3,)``.
    thrust_point : array_like or None, optional
        Point of the thrust force [m] in geometry axes, shape ``(3,)``. None
        uses *cg*. The moment ``(thrust_point - cg) x force`` is added.
    gust : callable or None, optional
        Gust field ``gust(points, t) -> velocities`` [m/s] (see
        :mod:`ventorum.unsteady.gusts`). On the GPU, a gust callable is
        evaluated on the host at every step and its velocities are copied to
        the device, so it adds a host cost per step.
    gravity : float, optional
        Gravitational acceleration [m/s^2] along ``-e_z``. Default
        ``9.80665``.
    store : {'loads', 'strengths'}, optional
        ``"loads"`` (default) keeps the histories only. ``"strengths"`` also
        keeps the final ring strengths and wake corners.
    coupling_tol : float, optional
        Relative state convergence tolerance of the implicit coupling
        sub-iterations [-]. Default ``1e-10``.
    max_subiterations : int, optional
        Maximum coupling sub-iterations per step (``>= 1``). Default ``20``.
        A step that does not reach *coupling_tol* gets a note, and the run
        continues. Light bodies (large apparent-mass load compared with the
        inertia) can need more sub-iterations.

    Returns
    -------
    FreeFlightResult
        Histories of the pose, rates, controls, thrust, aerodynamic loads,
        energy, work and angular momentum, the initial imbalance, the coupling
        statistics, trust metadata and notes.

    Raises
    ------
    InputError
        If both or neither of *t_end* and *n_steps* are given; if a mass,
        inertia, point, DOF, state, motion, control, thrust, gust or *store*
        input is invalid; if ``settings.solver_type`` is not VLM; if
        Mach > 0.3; or if the pitch angle reaches the 3-2-1 gimbal-lock limit
        near 90 deg.
    GroundStrikeError
        If the mean geometry touches or crosses the ground plane when
        ``condition.h`` is set.

    Notes
    -----
    The march starts impulsively: there is no wake at ``t = 0``, and the
    starting wake forms over the first chord lengths of travel (Katz and
    Plotkin 2001, section 13.11). The initial imbalance is the imbalance of
    this impulsive-start load. A constant thrust equal to minus
    ``initial_force_imbalance_body`` and ``initial_moment_imbalance_body``
    (from a first run with ``n_steps=1``) cancels the load at ``t = 0`` only.
    It does not trim the steady state: during the starting transient the
    loads change and the body moves. To trim the mean state, use minus
    ``steady_force_imbalance_body`` and ``steady_moment_imbalance_body``
    (the steady load of the mean state plus the weight). The body then moves
    a little during the starting transient and returns towards the mean
    state when the motion is stable.
    """
    if (t_end is None) == (n_steps is None):
        raise InputError("Give exactly one of t_end [s] or n_steps (not both and not neither).")
    if store not in _STORE_OPTIONS:
        raise InputError(f"store must be one of {_STORE_OPTIONS}, got {store!r}.")
    dt_val = None if dt is None else _validate_positive_float(dt, "dt")
    t_end_val = None if t_end is None else _validate_positive_float(t_end, "t_end")
    n_steps_val = None if n_steps is None else _validate_n_steps(n_steps)

    ac, sett, solver, geo, rp, lattice, ground = _prepare_setup(
        geometry, condition, settings, caller="free_flight",
    )

    dt_default = _default_dt_for_lattice(lattice, float(condition.V_inf))
    dt_eff = dt_default if dt_val is None else dt_val
    if n_steps_val is None:
        assert t_end_val is not None
        n_steps_val = max(1, int(math.ceil(t_end_val / dt_eff - 1.0e-12)))

    base, _ = _solve_steady_base(solver, lattice, condition, sett, ac, rp, geo["main"])
    tot = base.totals
    steady_loads = {
        "CL": float(tot.CL),
        "CD": float(tot.CDi),
        "CY": float(tot.CY),
        "Cl": float(tot.Cl),
        "Cm": float(tot.Cm),
        "Cn": float(tot.Cn),
    }

    resp = _solve_free_flight(
        lattice,
        condition,
        S_ref=float(ac.S_ref),
        b_ref=float(ac.b_ref),
        c_ref=float(ac.c_ref),
        ref_point=rp,
        mass=mass,
        inertia=inertia,
        cg=cg,
        gravity=gravity,
        dofs=dofs,
        initial_state=initial_state,
        motion=motion,
        thrust=thrust,
        thrust_point=thrust_point,
        controls=controls,
        gust=gust,
        n_steps=n_steps_val,
        dt=dt_val,
        ground=ground,
        wake_alignment=sett.wake_alignment,
        steady_loads=steady_loads,
        coupling_tol=coupling_tol,
        max_subiterations=max_subiterations,
    )

    te_idx = np.arange(resp.n_strips, dtype=np.int64) * resp.n_chord + (resp.n_chord - 1)
    dx_te = np.asarray(lattice.panel_length[te_idx], dtype=np.float64)
    trust, trust_notes = _build_unsteady_trust(
        base.totals.trust,
        n_chord=resp.n_chord,
        dx_te_ratio=dx_te / resp.dx_w,
        k_values=np.array([0.0], dtype=np.float64),
        wake_chords=MIN_TRUST_WAKE_CHORDS,
    )

    notes: list[str] = [
        "Force coefficients (CL, CD, CY) are in mean wind axes and moment coefficients (Cl, Cm, Cn) "
        "are in body axes of the mean state, normalised by the nominal V_inf and rho of condition.",
        "The inertial frame translates at the mean flight velocity of condition; gravity acts along -z of "
        "this frame. Dimensional aerodynamic loads at the centre of gravity are in this frame.",
    ]
    notes.extend(_device_execution_notes(resp.device, resp.precision, resp.notes))
    notes.extend(trust_notes)
    for note in resp.notes:
        if note not in notes:
            notes.append(note)
        if trust is not None and note not in trust.warnings:
            trust.warnings.append(note)

    keep_str = store == "strengths"
    return FreeFlightResult(
        time=resp.time,
        s=(2.0 * float(condition.V_inf) / float(ac.c_ref)) * resp.time,
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
        thrust_force=resp.thrust_force,
        thrust_moment=resp.thrust_moment,
        gust_velocity=resp.gust_velocity,
        CL=resp.CL,
        CD=resp.CD,
        CY=resp.CY,
        Cl=resp.Cl,
        Cm=resp.Cm,
        Cn=resp.Cn,
        strip_cl=resp.strip_cl,
        aero_force_cg=resp.aero_force_cg,
        aero_moment_cg=resp.aero_moment_cg,
        kinetic_energy=resp.kinetic_energy,
        potential_energy=resp.potential_energy,
        energy=resp.energy,
        aero_work=resp.aero_work,
        thrust_work=resp.thrust_work,
        work=resp.work,
        angular_momentum=resp.angular_momentum,
        initial_force_imbalance=resp.initial_force_imbalance,
        initial_moment_imbalance=resp.initial_moment_imbalance,
        initial_force_imbalance_body=resp.initial_force_imbalance_body,
        initial_moment_imbalance_body=resp.initial_moment_imbalance_body,
        steady_force_imbalance=resp.steady_force_imbalance,
        steady_moment_imbalance=resp.steady_moment_imbalance,
        steady_force_imbalance_body=resp.steady_force_imbalance_body,
        steady_moment_imbalance_body=resp.steady_moment_imbalance_body,
        subiterations=resp.subiterations,
        residuals=resp.residuals,
        dofs=tuple(resp.dofs),
        ground_contact=bool(resp.ground_contact),
        ground_contact_step=resp.ground_contact_step,
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
        cg=np.asarray(resp.cg, dtype=np.float64).copy(),
        mass=float(resp.mass),
        inertia=np.asarray(resp.inertia, dtype=np.float64).copy(),
        gravity=float(resp.gravity),
        V_inf=float(condition.V_inf),
        rho=float(condition.rho),
        base=base,
        trust=trust,
        trust_notes=trust_notes,
        notes=notes,
        device=str(resp.device),
        precision=str(resp.precision),
    )
