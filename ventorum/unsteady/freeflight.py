# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Private free-body flight coupling for the time-domain vortex-ring solver.

Couples the single-step time-domain vortex-ring marcher
(:class:`ventorum.unsteady.timedomain._TimeDomainStepper`) to the six-degree-of-freedom
Newton-Euler rigid-body equations of motion about the aircraft centre of
gravity, with selectable free degrees of freedom, gravity, a body-axes point
thrust force and moment, and time- or state-dependent control-surface
deflections.

Formulation
-----------
1. **Frames.** The inertial frame is the mean-flight frame ``(e_x, e_y, e_z)``
   of the time-domain ring solver (translating at the constant mean flight
   velocity ``V_0 = V_inf * d_0`` [m/s]; ``e_x = d_0`` aft along the mean
   flight path, ``e_z`` up along ``lift_direction`` in free air or along
   ``ground.normal`` in flat-ground effect, and ``e_y = e_z x e_x`` right).
   Gravity acts along ``-e_z`` with acceleration ``g`` [m/s^2] (default
   ``9.80665`` m/s^2). The six-axis body pose ``(x, y, z, roll, pitch, yaw)``
   consists of translations ``(x, y, z)`` [m] of ``ref_point`` along
   ``(e_x, e_y, e_z)`` and intrinsic 3-2-1 Euler rotations
   ``(roll, pitch, yaw)`` [rad] about ``ref_point`` with AVL signs.
2. **Rigid-body equations.** Translational motion follows Newton's second law
   in the inertial frame ``m * a_I = F_aero,I + F_grav,I + F_thrust,I``.
   When the centre of gravity is not at ``ref_point``, the equations are
   written in the generalised coordinates of the pose (Kane's method): each
   free DOF gives one row of the Newton and Euler equations projected on its
   partial velocities, and the prescribed DOFs enter with their accelerations.
   Rotational motion about the centre of gravity follows Euler's equation in
   body-fixed geometry axes ``I_b * d(omega_b)/dt + omega_b x (I_b * omega_b)
   = M_aero,cg,b + M_thrust,cg,b`` (Etkin and Reid 1996, chapter 4; Stevens and
   Lewis 2003, chapter 1), where ``omega_b = S(roll, pitch) * [roll_dot,
   pitch_dot, yaw_dot]^T``. The 3-2-1 Euler-angle kinematics have a gimbal-lock
   singularity at ``|pitch| = 90 deg``; pitch angles at or above 89 deg are
   refused with :class:`ventorum.core.errors.InputError`.
3. **Implicit coupling and Aitken relaxation.** At each step ``n -> n + 1``,
   free degrees of freedom are predicted by second-order Taylor extrapolation,
   aerodynamic loads at ``n + 1`` are evaluated from the restored stepper
   state, and the rigid-body equations are integrated from ``n`` to ``n + 1``
   with the second-order trapezoidal rule. Sub-iterations repeat with Aitken
   dynamic relaxation (Irons and Tuck 1969; Kuttler and Wall 2008) until the
   relative state change falls below ``coupling_tol`` (default ``1e-10``).

References
----------
* B. Etkin and L. D. Reid, "Dynamics of Flight: Stability and Control", 3rd
  ed., Wiley, 1996, chapter 4.
* B. L. Stevens and F. L. Lewis, "Aircraft Control and Simulation", 2nd ed.,
  Wiley, 2003, chapter 1.
* T. R. Kane and D. A. Levinson, "Dynamics: Theory and Applications",
  McGraw-Hill, 1985 (partial velocities and the generalised equations of
  motion).
* B. M. Irons and R. C. Tuck, "A version of the Aitken accelerator for
  computer iteration", International Journal for Numerical Methods in
  Engineering 1, 1969, pp. 275-277.
* U. Kuttler and W. A. Wall, "Fixed-point fluid-structure interaction solvers
  with dynamic relaxation", Computational Mechanics 43, 2008, pp. 61-72.
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, chapter 13.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from ventorum.aero.system import (
    GroundPlane,
    freestream_direction,
    lift_direction,
    side_direction,
)
from ventorum.core.datatypes import FlightCondition
from ventorum.core.errors import InputError
from ventorum.geometry.lattice import VortexLattice
from ventorum.unsteady.timedomain import (
    _MOTION_AXES,
    _rotation_matrix_321,
    _sample_motion,
    _StepLoads,
    _TimeDomainStepper,
)

# Gimbal-lock pitch limit [rad] for the 3-2-1 Euler sequence (89 deg, 1 deg below singularity).
_GIMBAL_LOCK_PITCH_RAD = math.radians(89.0)
# Bounds on the Aitken dynamic relaxation factor [-].
_AITKEN_OMEGA_MIN = 0.05
_AITKEN_OMEGA_MAX = 1.25


@dataclass(frozen=True, slots=True)
class FlightState:
    """Read-only state record passed to state-dependent thrust and control callables.

    :func:`ventorum.free_flight` gives this record to a ``thrust(t, state)``
    callable and to each control callable ``fn(t, state)``. The values are
    the state at the time ``t`` of the call. Inside the implicit coupling
    sub-iterations, the record holds the current iterate of the step.

    Attributes
    ----------
    t : float
        Current time [s].
    x, y, z : float
        Displacements of ``ref_point`` along ``(e_x, e_y, e_z)`` [m].
    roll, pitch, yaw : float
        3-2-1 Euler angles about ``ref_point`` [rad] (AVL signs).
    x_dot, y_dot, z_dot : float
        Translational velocities along ``(e_x, e_y, e_z)`` [m/s].
    roll_dot, pitch_dot, yaw_dot : float
        3-2-1 Euler-angle rates [rad/s].
    controls : Mapping[str, float]
        Read-only mapping of current control-surface deflections [rad].
    """

    t: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    x_dot: float
    y_dot: float
    z_dot: float
    roll_dot: float
    pitch_dot: float
    yaw_dot: float
    controls: Mapping[str, float]

    @property
    def time(self) -> float:
        """Current time [s] (alias of :attr:`t`)."""
        return self.t

    @property
    def theta(self) -> float:
        """Pitch angle [rad] (alias of :attr:`pitch`)."""
        return self.pitch

    @property
    def theta_dot(self) -> float:
        """Pitch rate [rad/s] (alias of :attr:`pitch_dot`)."""
        return self.pitch_dot


# Private alias of the public state record (kept for the private tests).
_FreeFlightState = FlightState


@dataclass(slots=True)
class _FreeFlightResponse:
    """Private result record of the free-body time-domain vortex-ring solver.

    Attributes
    ----------
    time : numpy.ndarray
        Time instants [s], shape ``(n_steps + 1,)``.
    x, y, z : numpy.ndarray
        Displacements of ``ref_point`` in the inertial mean-flight frame [m],
        shape ``(n_steps + 1,)``.
    roll, pitch, yaw : numpy.ndarray
        3-2-1 Euler angles about ``ref_point`` [rad] (AVL signs), shape
        ``(n_steps + 1,)``.
    theta : numpy.ndarray
        Pitch angle [rad] (matches ``pitch``), shape ``(n_steps + 1,)``.
    x_dot, y_dot, z_dot : numpy.ndarray
        Translational rates in the inertial mean-flight frame [m/s], shape
        ``(n_steps + 1,)``.
    roll_dot, pitch_dot, yaw_dot : numpy.ndarray
        3-2-1 Euler-angle rates [rad/s], shape ``(n_steps + 1,)``.
    theta_dot : numpy.ndarray
        Pitch Euler-angle rate [rad/s] (matches ``pitch_dot``), shape
        ``(n_steps + 1,)``.
    controls : dict[str, numpy.ndarray]
        Control-surface deflection histories [rad], each of shape
        ``(n_steps + 1,)``.
    controls_dot : dict[str, numpy.ndarray]
        Control-surface deflection rate histories [rad/s], each of shape
        ``(n_steps + 1,)``.
    thrust_force : numpy.ndarray
        Thrust force history in body axes [N], shape ``(n_steps + 1, 3)``.
    thrust_moment : numpy.ndarray
        Total thrust moment history about the centre of gravity in body axes
        (including ``(thrust_point - cg) x thrust_force``) [N m], shape
        ``(n_steps + 1, 3)``.
    CL, CD, CY : numpy.ndarray
        Aerodynamic force coefficients along mean wind axes [-], shape
        ``(n_steps + 1,)``.
    Cl, Cm, Cn : numpy.ndarray
        Aerodynamic moment coefficients in mean body axes about ``ref_point``
        [-], shape ``(n_steps + 1,)``.
    strip_cl : numpy.ndarray
        Per-strip section lift coefficient [-], shape
        ``(n_steps + 1, n_strips)``.
    aero_force_cg : numpy.ndarray
        Dimensional aerodynamic force at the centre of gravity in the inertial
        mean-flight frame [N], shape ``(n_steps + 1, 3)``.
    aero_moment_cg : numpy.ndarray
        Dimensional aerodynamic moment about the centre of gravity in the
        inertial mean-flight frame [N m], shape ``(n_steps + 1, 3)``.
    kinetic_energy : numpy.ndarray
        Total kinetic energy [J]: the translational energy of the centre of
        gravity plus the rotational energy about it, shape ``(n_steps + 1,)``.
    potential_energy : numpy.ndarray
        Gravitational potential energy ``m * g * z_cg`` [J], with ``z_cg``
        the height of the centre of gravity in the inertial frame, shape
        ``(n_steps + 1,)``.
    energy : numpy.ndarray
        Total mechanical energy ``kinetic_energy + potential_energy`` [J],
        shape ``(n_steps + 1,)``.
    aero_work : numpy.ndarray
        Cumulative work of the aerodynamic force and moment on the free DOFs
        [J], shape ``(n_steps + 1,)``.
    thrust_work : numpy.ndarray
        Cumulative work of the thrust force and moment on the free DOFs [J],
        shape ``(n_steps + 1,)``.
    work : numpy.ndarray
        Cumulative work of the aerodynamic and thrust loads
        ``aero_work + thrust_work`` [J], shape ``(n_steps + 1,)``.
    angular_momentum : numpy.ndarray
        Angular momentum vector about the centre of gravity in the inertial
        frame [kg m^2/s], shape ``(n_steps + 1, 3)``.
    initial_force_imbalance : numpy.ndarray
        Net force ``F_aero + F_grav + F_thrust`` at ``t = 0`` in the inertial
        frame [N], shape ``(3,)``.
    initial_moment_imbalance : numpy.ndarray
        Net moment ``M_aero,cg + M_thrust,cg`` about the centre of gravity at
        ``t = 0`` in the inertial frame [N m], shape ``(3,)``.
    initial_force_imbalance_body : numpy.ndarray
        Net force ``F_aero + F_grav + F_thrust`` at ``t = 0`` in body geometry
        axes [N], shape ``(3,)``.
    initial_moment_imbalance_body : numpy.ndarray
        Net moment ``M_aero,cg + M_thrust,cg`` about the centre of gravity at
        ``t = 0`` in body geometry axes [N m], shape ``(3,)``.
    subiterations : numpy.ndarray
        Number of implicit coupling sub-iterations per step [-], shape
        ``(n_steps + 1,)``.
    residuals : numpy.ndarray
        Final relative state residual per step [-], shape ``(n_steps + 1,)``.
    dofs : tuple[str, ...]
        Free degrees of freedom.
    ground_contact : bool
        True when the run stopped early because a wing or flap node reached the
        ground plane in free flight.
    ground_contact_step : int or None
        Step index at which ground contact occurred, or None.
    gamma_bound : numpy.ndarray
        Bound vortex-ring circulations at the last valid step [m^2/s], shape
        ``(n_panels,)``.
    gamma_wake : numpy.ndarray
        Wake vortex-ring circulations at the last valid step [m^2/s], shape
        ``(n_steps, n_strips)``.
    wake_corners : numpy.ndarray
        Wake vortex-ring corners at the last valid step [m], shape
        ``(n_steps, n_strips, 4, 3)``.
    gust_velocity : numpy.ndarray
        Gust velocity at the instantaneous reference point in geometry axes
        [m/s], shape ``(n_steps + 1, 3)``.
    dt, dt_default, dx_w, dx_te_mean : float
        Time step [s], default convective time step [s], wake row length [m]
        and mean trailing-edge panel length [m].
    n_steps, n_panels, n_strips, n_chord : int
        Number of completed steps, bound panels, spanwise strips and chordwise
        panels.
    te_correction : bool
        True when the trailing-edge boundary load correction is included.
    c_ref, b_ref, S_ref : float
        Reference chord [m], reference span [m] and reference area [m^2].
    ref_point, cg : numpy.ndarray
        Reference point [m] and centre of gravity [m] in geometry axes, shape
        ``(3,)``.
    mass, gravity : float
        Aircraft mass [kg] and gravitational acceleration [m/s^2].
    inertia : numpy.ndarray
        Inertia tensor about the centre of gravity in geometry axes [kg m^2],
        shape ``(3, 3)``.
    steady_force_imbalance, steady_moment_imbalance : numpy.ndarray or None
        Net force ``F_aero + F_grav`` [N] and net moment about the centre of
        gravity [N m] of the steady load of the mean state, without thrust,
        in the inertial frame, shape ``(3,)``. None when *steady_loads* is
        not given.
    steady_force_imbalance_body, steady_moment_imbalance_body : numpy.ndarray or None
        The same steady net force [N] and moment [N m] in body axes at the
        mean pose, shape ``(3,)``. A constant thrust equal to minus these
        values trims the mean state. None when *steady_loads* is not given.
    notes : list[str]
        Diagnostic notes (time-step ratio, gust notes, device fallback notes,
        non-converged steps, and ground contact).
    device : str
        Execution device: ``"cpu"`` or ``"gpu"``.
    precision : str
        Kernel arithmetic precision: ``"float64"`` or ``"float32"``.
    """

    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    roll: np.ndarray
    pitch: np.ndarray
    yaw: np.ndarray
    theta: np.ndarray
    x_dot: np.ndarray
    y_dot: np.ndarray
    z_dot: np.ndarray
    roll_dot: np.ndarray
    pitch_dot: np.ndarray
    yaw_dot: np.ndarray
    theta_dot: np.ndarray
    controls: dict[str, np.ndarray]
    controls_dot: dict[str, np.ndarray]
    thrust_force: np.ndarray
    thrust_moment: np.ndarray
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
    subiterations: np.ndarray
    residuals: np.ndarray
    dofs: tuple[str, ...]
    ground_contact: bool
    ground_contact_step: int | None
    gamma_bound: np.ndarray
    gamma_wake: np.ndarray
    wake_corners: np.ndarray
    gust_velocity: np.ndarray
    dt: float
    dt_default: float
    dx_w: float
    dx_te_mean: float
    n_steps: int
    n_panels: int
    n_strips: int
    n_chord: int
    te_correction: bool
    c_ref: float
    b_ref: float
    S_ref: float
    ref_point: np.ndarray
    cg: np.ndarray
    mass: float
    gravity: float
    inertia: np.ndarray
    notes: list[str] = field(default_factory=list)
    steady_force_imbalance: np.ndarray | None = None
    steady_moment_imbalance: np.ndarray | None = None
    steady_force_imbalance_body: np.ndarray | None = None
    steady_moment_imbalance_body: np.ndarray | None = None
    device: str = "cpu"
    precision: str = "float64"


_FreeFlightResult = _FreeFlightResponse


def _euler_rate_matrix_body(roll: float, pitch: float) -> np.ndarray:
    """Return matrix ``S(roll, pitch)`` [-] mapping ``[roll_dot, pitch_dot, yaw_dot]^T`` to ``omega_b`` [rad/s].

    Body axes are geometry axes rotated with the body (x aft, y right, z up)
    and Euler angles follow the 3-2-1 sequence with AVL signs (Etkin and Reid
    1996, chapter 4), so ``R(roll, pitch, yaw) @ omega_b`` equals
    :func:`ventorum.unsteady.timedomain._angular_velocity_321`.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array(
        [
            [-1.0, 0.0, sp],
            [0.0, cr, cp * sr],
            [0.0, sr, -cp * cr],
        ],
        dtype=np.float64,
    )


def _euler_rate_matrix_body_dot(
    roll: float,
    pitch: float,
    roll_dot: float,
    pitch_dot: float,
) -> np.ndarray:
    """Return time derivative ``dS/dt`` [1/s] of :func:`_euler_rate_matrix_body`."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array(
        [
            [0.0, 0.0, cp * pitch_dot],
            [0.0, -sr * roll_dot, -sp * sr * pitch_dot + cp * cr * roll_dot],
            [0.0, cr * roll_dot, sp * cr * pitch_dot + cp * sr * roll_dot],
        ],
        dtype=np.float64,
    )


def _skew(v: np.ndarray) -> np.ndarray:
    """Return the cross-product matrix ``[v]x`` with ``[v]x @ w = v x w`` [-], shape ``(3, 3)``."""
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def _check_gimbal_lock(pitch: float, context: str = "") -> None:
    """Raise :class:`InputError` if *pitch* [rad] is non-finite or near the 90 deg gimbal lock."""
    if not math.isfinite(pitch):
        raise InputError(f"Non-finite pitch angle {pitch!r} rad{context}.")
    if abs(pitch) >= _GIMBAL_LOCK_PITCH_RAD:
        deg = math.degrees(pitch)
        raise InputError(
            f"Pitch angle {deg:.4g} deg{context} reaches or exceeds the 3-2-1 Euler-angle "
            "gimbal lock limit near |pitch| = 90 deg (limit 89 deg)."
        )


def _callable_takes_state(fn: Callable[..., Any], name: str) -> bool:
    """Return True if *fn* accepts ``(t, state)``, False if it accepts ``(t,)``."""
    if not callable(fn):
        raise InputError(f"{name} must be a callable, got {type(fn).__name__}.")
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise InputError(f"Cannot inspect signature of {name}: {exc}") from exc

    pos_params = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    n_required = sum(1 for p in pos_params if p.default is inspect.Parameter.empty)
    n_max = math.inf if has_varargs else len(pos_params)

    if n_required > 2 or n_max < 1:
        raise InputError(
            f"{name} callable must accept (t) or (t, state) (1 or 2 positional arguments), "
            f"got signature {sig}."
        )
    if n_required == 2:
        return True
    if len(pos_params) >= 2 and pos_params[1].name in ("state", "st", "_state"):
        return True
    return False


def _validate_dofs(dofs: Sequence[str]) -> tuple[str, ...]:
    """Validate *dofs* as a sequence of unique axis names from :data:`_MOTION_AXES`."""
    if isinstance(dofs, (str, bytes, bool)) or not isinstance(dofs, Sequence):
        raise InputError(f"dofs must be a sequence of DOF names from {_MOTION_AXES}, got {dofs!r}.")
    out: list[str] = []
    seen: set[str] = set()
    for item in dofs:
        if not isinstance(item, str) or item not in _MOTION_AXES:
            raise InputError(f"Unknown DOF {item!r} in dofs; allowed DOFs are {_MOTION_AXES}.")
        if item in seen:
            raise InputError(f"Duplicate DOF {item!r} in dofs; all DOFs must be unique.")
        seen.add(item)
        out.append(item)
    return tuple(out)


def _validate_mass_and_gravity(mass: Any, gravity: Any) -> tuple[float, float]:
    """Validate *mass* [kg] (> 0) and *gravity* [m/s^2] (>= 0)."""
    if isinstance(mass, bool):
        raise InputError(f"mass must be a positive finite number [kg], got {mass!r}.")
    try:
        m_val = float(mass)
    except (TypeError, ValueError) as exc:
        raise InputError(f"mass must be a positive finite number [kg], got {mass!r}.") from exc
    if not math.isfinite(m_val) or m_val <= 0.0:
        raise InputError(f"mass must be finite and > 0 [kg], got {mass!r}.")

    if isinstance(gravity, bool):
        raise InputError(f"gravity must be a non-negative finite number [m/s^2], got {gravity!r}.")
    try:
        g_val = float(gravity)
    except (TypeError, ValueError) as exc:
        raise InputError(f"gravity must be a non-negative finite number [m/s^2], got {gravity!r}.") from exc
    if not math.isfinite(g_val) or g_val < 0.0:
        raise InputError(f"gravity must be finite and >= 0 [m/s^2], got {gravity!r}.")
    return m_val, g_val


def _validate_inertia(inertia: Any) -> np.ndarray:
    """Validate *inertia* [kg m^2] as a 3x3 finite, symmetric, positive-definite matrix."""
    try:
        arr = np.asarray(inertia, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError(f"inertia must be a 3x3 array [kg m^2], got {inertia!r}.") from exc
    if arr.shape != (3, 3):
        raise InputError(f"inertia must have shape (3, 3), got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise InputError("inertia must contain only finite numbers.")
    scale = max(float(np.max(np.abs(arr))), 1.0)
    if not np.allclose(arr, arr.T, atol=1.0e-12 * scale, rtol=1.0e-12):
        raise InputError("inertia tensor must be symmetric (I == I.T).")
    sym = 0.5 * (arr + arr.T)
    eigvals = np.linalg.eigvalsh(sym)
    if float(np.min(eigvals)) <= 0.0:
        raise InputError(
            f"inertia tensor must be positive definite; minimum eigenvalue is {float(np.min(eigvals)):.4g}."
        )
    return sym.copy()


def _validate_point3(pt: Any, name: str, default: np.ndarray) -> np.ndarray:
    """Validate *pt* [m] as a finite 3-vector, returning *default* when *pt* is None."""
    if pt is None:
        return default.copy()
    try:
        arr = np.asarray(pt, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be a finite 3-vector [m], got {pt!r}.") from exc
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise InputError(f"{name} must be a finite 3-vector [m] of shape (3,), got {pt!r}.")
    return arr.copy()


def _second_order_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    """Return second-order finite-difference derivative of 1-D *arr* on uniform step *dt* [s]."""
    n_pts = int(arr.size)
    dots = np.empty(n_pts, dtype=np.float64)
    if n_pts == 2:
        slope = (arr[1] - arr[0]) / dt
        dots[0] = slope
        dots[1] = slope
    else:
        dots[0] = (-3.0 * arr[0] + 4.0 * arr[1] - arr[2]) / (2.0 * dt)
        dots[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
        dots[-1] = (3.0 * arr[-1] - 4.0 * arr[-2] + arr[-3]) / (2.0 * dt)
    return dots


def _eval_thrust(
    thrust_fn: Callable[..., Any] | None,
    takes_state: bool,
    t: float,
    state: FlightState,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate *thrust_fn* at *t* [s] and *state*, returning ``(force_b, moment_b)`` [N, N m]."""
    if thrust_fn is None:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    try:
        raw = thrust_fn(float(t), state) if takes_state else thrust_fn(float(t))
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"thrust callable failed at t = {float(t):.6g} s: {exc}") from exc
    if not isinstance(raw, (tuple, list)) or len(raw) != 2:
        raise InputError("thrust callable must return a 2-tuple (force, moment).")
    try:
        f_arr = np.asarray(raw[0], dtype=np.float64)
        m_arr = np.asarray(raw[1], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError("thrust force and moment must be numeric 3-vectors.") from exc
    if f_arr.shape != (3,) or m_arr.shape != (3,):
        raise InputError(
            f"thrust force and moment must each have shape (3,), got {f_arr.shape} and {m_arr.shape}."
        )
    if not np.all(np.isfinite(f_arr)) or not np.all(np.isfinite(m_arr)):
        raise InputError(f"thrust callable returned non-finite values at t = {float(t):.6g} s.")
    return f_arr.copy(), m_arr.copy()


def _make_state(
    t: float,
    q: np.ndarray,
    q_dot: np.ndarray,
    ctrl_map: Mapping[str, float],
) -> FlightState:
    """Build a read-only :class:`FlightState` from 6-vectors *q*, *q_dot* and *ctrl_map*."""
    return FlightState(
        t=float(t),
        x=float(q[0]),
        y=float(q[1]),
        z=float(q[2]),
        roll=float(q[3]),
        pitch=float(q[4]),
        yaw=float(q[5]),
        x_dot=float(q_dot[0]),
        y_dot=float(q_dot[1]),
        z_dot=float(q_dot[2]),
        roll_dot=float(q_dot[3]),
        pitch_dot=float(q_dot[4]),
        yaw_dot=float(q_dot[5]),
        controls=MappingProxyType(dict(ctrl_map)),
    )


def _steady_imbalance(
    steady_loads: Mapping[str, float] | None,
    *,
    qS: float,
    b_ref: float,
    c_ref: float,
    d_0: np.ndarray,
    S_0: np.ndarray,
    L_0: np.ndarray,
    E_mat: np.ndarray,
    F_grav_g: np.ndarray,
    r_cg_rel_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the steady net force [N] and moment about the cg [N m] of the mean pose, or None.

    The load is the steady aerodynamic load plus the weight, without thrust,
    at the mean pose (body axes equal geometry axes). The function returns
    ``(F_inertial, M_inertial, F_body, M_body)``.
    """
    if steady_loads is None:
        return None
    if not isinstance(steady_loads, Mapping):
        raise InputError(f"steady_loads must be a mapping of coefficients, got {type(steady_loads).__name__}.")
    vals: dict[str, float] = {}
    for key in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
        if key not in steady_loads:
            raise InputError(f"steady_loads needs the key {key!r}.")
        val = float(steady_loads[key])
        if not math.isfinite(val):
            raise InputError(f"steady_loads[{key!r}] must be finite, got {steady_loads[key]!r}.")
        vals[key] = val
    F_aero_g = qS * (vals["CD"] * d_0 + vals["CY"] * S_0 + vals["CL"] * L_0)
    M_ref_g = np.array([-qS * b_ref * vals["Cl"], qS * c_ref * vals["Cm"], -qS * b_ref * vals["Cn"]], dtype=np.float64)
    M_cg_g = M_ref_g - np.cross(r_cg_rel_b, F_aero_g)
    F_net_g = F_aero_g + F_grav_g
    return (E_mat.T @ F_net_g, E_mat.T @ M_cg_g, F_net_g.copy(), M_cg_g.copy())


def _solve_free_flight(
    lattice: VortexLattice,
    condition: FlightCondition,
    *,
    S_ref: float,
    b_ref: float,
    c_ref: float,
    ref_point: np.ndarray | None = None,
    mass: float,
    inertia: np.ndarray | Sequence[Sequence[float]],
    cg: np.ndarray | Sequence[float] | None = None,
    gravity: float = 9.80665,
    dofs: Sequence[str] = _MOTION_AXES,
    initial_state: Mapping[str, float] | None = None,
    motion: Mapping[str, Sequence[float] | np.ndarray | Callable[[float], float]] | None = None,
    thrust: Callable[..., tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]] | None = None,
    thrust_point: np.ndarray | Sequence[float] | None = None,
    controls: Mapping[str, Sequence[float] | np.ndarray | Callable[..., float]] | None = None,
    gust: Callable[[np.ndarray, float], np.ndarray] | None = None,
    n_steps: int,
    dt: float | None = None,
    ground: GroundPlane | None = None,
    wake_dir: np.ndarray | None = None,
    te_extension_fraction: float = 0.25,
    rc: np.ndarray | float | None = None,
    use_cross_surface_core: bool = True,
    te_correction: bool = True,
    wake_alignment: str = "freestream",
    backend: str | None = None,
    coupling_tol: float = 1.0e-10,
    max_subiterations: int = 20,
    aerodynamics_off: bool = False,
    steady_loads: Mapping[str, float] | None = None,
) -> _FreeFlightResponse:
    """Couple the time-domain vortex-ring solver to the six-DOF rigid-body equations of motion.

    Parameters
    ----------
    lattice : VortexLattice
        Built vortex-lattice mesh (``collocation="vlm"``).
    condition : FlightCondition
        Mean flight condition (``V_inf`` [m/s], ``rho`` [kg/m^3], ``alpha``,
        ``beta`` [rad]).
    S_ref, b_ref, c_ref : float
        Reference area [m^2], reference span [m] and reference chord [m].
    ref_point : numpy.ndarray or None, optional
        Moment reference and rotation pivot point [m] in geometry axes, shape
        ``(3,)``. None uses the origin.
    mass : float
        Aircraft mass [kg] (must be finite and ``> 0``).
    inertia : numpy.ndarray or sequence of sequence of float
        Inertia tensor about the centre of gravity in geometry axes [kg m^2],
        shape ``(3, 3)``, symmetric and positive definite.
    cg : numpy.ndarray, sequence of float or None, optional
        Centre of gravity [m] in geometry axes, shape ``(3,)``. None uses
        *ref_point*.
    gravity : float, optional
        Gravitational acceleration [m/s^2] acting along ``-e_z`` of the
        inertial mean-flight frame. Default ``9.80665``.
    dofs : sequence of str, optional
        Subset of ``('x', 'y', 'z', 'roll', 'pitch', 'yaw')`` that move freely
        under the rigid-body equations of motion. Frozen DOFs follow *motion*
        (default zero).
    initial_state : mapping of str to float or None, optional
        Initial displacements [m, rad] (keys ``'<dof>'``) and initial rates
        [m/s, rad/s] (keys ``'<dof>_dot'``) of the free DOFs in *dofs*.
        Missing entries default to zero.
    motion : mapping of str to array or callable, or None, optional
        Prescribed displacements [m, rad] for the frozen DOFs (not in *dofs*).
    thrust : callable or None, optional
        Callable ``thrust(t) -> (force, moment)`` or
        ``thrust(t, state) -> (force, moment)`` returning body-axes force [N]
        and moment [N m], each of shape ``(3,)``.
    thrust_point : numpy.ndarray, sequence of float or None, optional
        Application point of the thrust force [m] in geometry axes, shape
        ``(3,)``. None uses *cg*. The moment ``(thrust_point - cg) x force`` is
        added to the thrust moment about the centre of gravity.
    controls : mapping of str to array or callable, or None, optional
        Control-surface deflections [rad], keyed by ``ControlSurface.name``.
        Each entry may be a 1-D array of length ``n_steps + 1``, a callable
        ``fn(t)``, or a callable ``fn(t, state)``.
    gust : callable or None, optional
        Convected gust velocity field ``gust(points, t) -> velocities`` [m/s].
        On the GPU, a gust callable is evaluated on the host at every step and
        its velocities are copied to the device, so it adds a host cost per
        step.
    n_steps : int
        Number of time steps to march (``>= 1``).
    dt : float or None, optional
        Time step [s]. None uses ``dx_TE / V_inf`` [s].
    ground : GroundPlane or None, optional
        Flat ground plane in mean-flight axes, or None for free air.
    wake_dir : numpy.ndarray or None, optional
        Unit wake convection direction [-].
    te_extension_fraction : float, optional
        Trailing-edge ring closure fraction [-]. Default ``0.25``.
    rc : numpy.ndarray or float or None, optional
        Vortex-core radius override [m].
    use_cross_surface_core : bool, optional
        Include cross-surface target cores when *rc* is None. Default True.
    te_correction : bool, optional
        Include trailing-edge boundary load correction. Default True.
    wake_alignment : {'freestream', 'body'}, optional
        Wake convection alignment rule. Default ``'freestream'``.
    backend : {'auto', 'numba', 'numpy'} or None, optional
        Ring-kernel backend override.
    coupling_tol : float, optional
        Relative state convergence tolerance for implicit sub-iterations [-].
        Default ``1e-10``.
    max_subiterations : int, optional
        Maximum implicit sub-iterations per step (``>= 1``). Default ``20``.
    aerodynamics_off : bool, optional
        Private flag setting aerodynamic loads to zero for pure rigid-body
        verification checks. Default False.
    steady_loads : mapping of str to float or None, optional
        Steady coefficients of the mean state, keys ``'CL'``, ``'CD'``,
        ``'CY'`` (mean wind axes) and ``'Cl'``, ``'Cm'``, ``'Cn'`` (body axes
        about *ref_point*) [-]. When given, the result holds the steady
        imbalance of the mean pose (aerodynamic load plus weight, no thrust).
        Default None.

    Returns
    -------
    _FreeFlightResponse
        Time histories of pose, rates, controls, thrust, aerodynamic loads,
        mechanical energy and work, angular momentum, initial trim imbalance,
        sub-iteration counts and residuals, and diagnostic notes.

    Raises
    ------
    InputError
        If any input argument has an invalid type, shape, sign or value, or if
        ``|pitch|`` reaches the 3-2-1 gimbal-lock limit near 90 deg.
    """
    # 1. Validate rigid-body, DOF and coupling inputs.
    dofs_tuple = _validate_dofs(dofs)
    dof_set = set(dofs_tuple)
    mass_val, gravity_val = _validate_mass_and_gravity(mass, gravity)
    I_b = _validate_inertia(inertia)

    rp = np.zeros(3, dtype=np.float64) if ref_point is None else _validate_point3(ref_point, "ref_point", np.zeros(3))
    cg_vec = _validate_point3(cg, "cg", rp)
    thrust_pt_vec = _validate_point3(thrust_point, "thrust_point", cg_vec)
    r_cg_rel_b = cg_vec - rp
    r_thrust_cg_b = thrust_pt_vec - cg_vec

    if isinstance(coupling_tol, bool):
        raise InputError(f"coupling_tol must be a positive finite number, got {coupling_tol!r}.")
    try:
        tol_val = float(coupling_tol)
    except (TypeError, ValueError) as exc:
        raise InputError(f"coupling_tol must be a positive finite number, got {coupling_tol!r}.") from exc
    if not math.isfinite(tol_val) or tol_val <= 0.0:
        raise InputError(f"coupling_tol must be finite and > 0, got {coupling_tol!r}.")

    if isinstance(max_subiterations, bool) or not isinstance(max_subiterations, (int, np.integer)):
        raise InputError(f"max_subiterations must be an integer >= 1, got {max_subiterations!r}.")
    max_sub_int = int(max_subiterations)
    if max_sub_int < 1:
        raise InputError(f"max_subiterations must be >= 1, got {max_sub_int}.")

    thrust_takes_state = False
    if thrust is not None:
        thrust_takes_state = _callable_takes_state(thrust, "thrust")

    # Validate initial_state against free DOFs.
    q0_free = np.zeros(6, dtype=np.float64)
    qd0_free = np.zeros(6, dtype=np.float64)
    if initial_state is not None:
        if not isinstance(initial_state, Mapping):
            raise InputError(f"initial_state must be a mapping, got {type(initial_state).__name__}.")
        axis_to_idx = {ax: idx for idx, ax in enumerate(_MOTION_AXES)}
        rate_to_idx = {f"{ax}_dot": idx for idx, ax in enumerate(_MOTION_AXES)}
        for key, raw_val in initial_state.items():
            if not isinstance(key, str) or (key not in axis_to_idx and key not in rate_to_idx):
                raise InputError(
                    f"Unknown initial_state key {key!r}; allowed keys are <dof> and <dof>_dot for dofs in {_MOTION_AXES}."
                )
            base_ax = key[:-4] if key.endswith("_dot") else key
            if base_ax not in dof_set:
                raise InputError(
                    f"initial_state key {key!r} refers to frozen DOF {base_ax!r} (not in dofs={dofs_tuple}); "
                    "prescribe frozen DOFs through motion."
                )
            if isinstance(raw_val, bool):
                raise InputError(f"initial_state[{key!r}] must be a finite number, got {raw_val!r}.")
            try:
                f_val = float(raw_val)
            except (TypeError, ValueError) as exc:
                raise InputError(f"initial_state[{key!r}] must be a finite number, got {raw_val!r}.") from exc
            if not math.isfinite(f_val):
                raise InputError(f"initial_state[{key!r}] must be finite, got {raw_val!r}.")
            if key in axis_to_idx:
                q0_free[axis_to_idx[key]] = f_val
            else:
                qd0_free[rate_to_idx[key]] = f_val

    # Validate prescribed motion mapping for frozen DOFs.
    if motion is not None:
        if not isinstance(motion, Mapping):
            raise InputError(f"motion must be a mapping from axis name to history, got {type(motion).__name__}.")
        for key in motion:
            if key not in _MOTION_AXES:
                raise InputError(f"Unknown motion axis {key!r}; allowed axes are {_MOTION_AXES}.")
            if key in dof_set:
                raise InputError(
                    f"Motion axis {key!r} is in free dofs={dofs_tuple} and cannot also be prescribed in motion."
                )

    # Validate controls mapping and classify callables of (t) vs (t, state).
    ctrl_takes_state: dict[str, bool] = {}
    has_state_controls = False
    if controls is not None:
        if not isinstance(controls, Mapping):
            raise InputError(
                f"controls must be a mapping from control name to deflection history, got {type(controls).__name__}."
            )
        for cname, spec in controls.items():
            if callable(spec):
                takes_st = _callable_takes_state(spec, f"controls[{cname!r}]")
                ctrl_takes_state[cname] = takes_st
                if takes_st:
                    has_state_controls = True
            else:
                ctrl_takes_state[cname] = False

    # 2. Construct the aerodynamic stepper.
    all_frozen = len(dofs_tuple) == 0
    if all_frozen and not has_state_controls:
        stepper = _TimeDomainStepper.from_prescribed(
            lattice,
            condition,
            S_ref=S_ref,
            b_ref=b_ref,
            c_ref=c_ref,
            ref_point=rp,
            motion=motion,
            controls=controls,  # type: ignore[arg-type]
            gust=gust,
            n_steps=n_steps,
            dt=dt,
            ground=ground,
            wake_dir=wake_dir,
            te_extension_fraction=te_extension_fraction,
            rc=rc,
            use_cross_surface_core=use_cross_surface_core,
            te_correction=te_correction,
            wake_alignment=wake_alignment,
            backend=backend,
        )
    else:
        ctrl_names_list = [] if controls is None else list(controls.keys())
        stepper = _TimeDomainStepper(
            lattice,
            condition,
            S_ref=S_ref,
            b_ref=b_ref,
            c_ref=c_ref,
            ref_point=rp,
            n_steps=n_steps,
            dt=dt,
            ground=ground,
            wake_dir=wake_dir,
            te_extension_fraction=te_extension_fraction,
            rc=rc,
            use_cross_surface_core=use_cross_surface_core,
            te_correction=te_correction,
            wake_alignment=wake_alignment,
            backend=backend,
            gust=gust,
            is_stationary=False,
            build_Dynamic_controls=ctrl_names_list if ctrl_names_list else None,
        )

    try:
        return _run_free_flight_march(
            stepper,
            condition=condition,
            rp=rp,
            cg_vec=cg_vec,
            r_cg_rel_b=r_cg_rel_b,
            r_thrust_cg_b=r_thrust_cg_b,
            mass_val=mass_val,
            gravity_val=gravity_val,
            I_b=I_b,
            dofs_tuple=dofs_tuple,
            dof_set=dof_set,
            all_frozen=all_frozen,
            q0_free=q0_free,
            qd0_free=qd0_free,
            motion=motion,
            thrust=thrust,
            thrust_takes_state=thrust_takes_state,
            controls=controls,
            ctrl_takes_state=ctrl_takes_state,
            has_state_controls=has_state_controls,
            ground=ground,
            tol_val=tol_val,
            max_sub_int=max_sub_int,
            aerodynamics_off=aerodynamics_off,
            steady_loads=steady_loads,
        )
    finally:
        stepper.release_gpu()


def _run_free_flight_march(
    stepper: _TimeDomainStepper,
    *,
    condition: FlightCondition,
    rp: np.ndarray,
    cg_vec: np.ndarray,
    r_cg_rel_b: np.ndarray,
    r_thrust_cg_b: np.ndarray,
    mass_val: float,
    gravity_val: float,
    I_b: np.ndarray,
    dofs_tuple: tuple[str, ...],
    dof_set: set[str],
    all_frozen: bool,
    q0_free: np.ndarray,
    qd0_free: np.ndarray,
    motion: Mapping[str, Sequence[float] | np.ndarray | Callable[[float], float]] | None,
    thrust: Callable[..., tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]] | None,
    thrust_takes_state: bool,
    controls: Mapping[str, Sequence[float] | np.ndarray | Callable[..., float]] | None,
    ctrl_takes_state: dict[str, bool],
    has_state_controls: bool,
    ground: GroundPlane | None,
    tol_val: float,
    max_sub_int: int,
    aerodynamics_off: bool,
    steady_loads: Mapping[str, float] | None,
) -> _FreeFlightResponse:
    """Run the coupled free-flight time march on an initialised *stepper*."""
    n_steps_int = stepper.n_steps
    n_s = stepper.n_strips
    dt_val = stepper.dt
    time = stepper.time
    n_pts = n_steps_int + 1

    # 3. Sample prescribed motion for frozen DOFs and time-only controls.
    q_presc = np.zeros((n_pts, 6), dtype=np.float64)
    qd_presc = np.zeros((n_pts, 6), dtype=np.float64)
    qdd_presc = np.zeros((n_pts, 6), dtype=np.float64)
    if all_frozen and stepper.m_axes is not None:
        for idx, ax in enumerate(_MOTION_AXES):
            q_presc[:, idx] = stepper.m_axes[ax][0]
            qd_presc[:, idx] = stepper.m_axes[ax][1]
            qdd_presc[:, idx] = _second_order_diff(qd_presc[:, idx], dt_val)
    elif motion is not None:
        for idx, ax in enumerate(_MOTION_AXES):
            if ax in motion:
                arr, dot = _sample_motion(motion[ax], time, dt_val, ax, stepper.notes)
                q_presc[:, idx] = arr
                qd_presc[:, idx] = dot
                qdd_presc[:, idx] = _second_order_diff(dot, dt_val)

    ctrl_hist: dict[str, np.ndarray] = {}
    ctrl_dot_hist: dict[str, np.ndarray] = {}
    if all_frozen and not has_state_controls:
        for cname, arr in stepper.ctrl_arrs.items():
            ctrl_hist[cname] = arr.copy()
            ctrl_dot_hist[cname] = stepper.ctrl_dots[cname].copy()
    elif controls is not None:
        for cname, spec in controls.items():
            if not ctrl_takes_state[cname]:
                arr, dot = _sample_motion(spec, time, dt_val, f"controls[{cname!r}]", stepper.notes)
                ctrl_hist[cname] = arr
                ctrl_dot_hist[cname] = dot
            else:
                ctrl_hist[cname] = np.zeros(n_pts, dtype=np.float64)
                ctrl_dot_hist[cname] = np.zeros(n_pts, dtype=np.float64)

    # Basis matrix E = [e_x, e_y, e_z] mapping inertial components to mean geometry axes.
    d_0 = freestream_direction(float(condition.alpha), float(condition.beta))
    L_0 = lift_direction(float(condition.alpha))
    S_0 = side_direction(float(condition.alpha), float(condition.beta))
    E_mat = np.column_stack([stepper.e_x, stepper.e_y, stepper.e_z])
    F_grav_I = np.array([0.0, 0.0, -mass_val * gravity_val], dtype=np.float64)
    F_grav_g = E_mat @ F_grav_I
    qS = stepper.qS
    b_ref_f = stepper.b_ref
    c_ref_f = stepper.c_ref

    free_mask = np.array([ax in dof_set for ax in _MOTION_AXES], dtype=bool)
    free_idx = np.flatnonzero(free_mask)

    # 4. Preallocate output time series.
    q_hist = np.zeros((n_pts, 6), dtype=np.float64)
    qd_hist = np.zeros((n_pts, 6), dtype=np.float64)
    qdd_hist = np.zeros((n_pts, 6), dtype=np.float64)
    CL = np.zeros(n_pts, dtype=np.float64)
    CD = np.zeros(n_pts, dtype=np.float64)
    CY = np.zeros(n_pts, dtype=np.float64)
    Cl = np.zeros(n_pts, dtype=np.float64)
    Cm = np.zeros(n_pts, dtype=np.float64)
    Cn = np.zeros(n_pts, dtype=np.float64)
    strip_cl = np.zeros((n_pts, n_s), dtype=np.float64)
    gust_ref = np.zeros((n_pts, 3), dtype=np.float64)
    thrust_force_hist = np.zeros((n_pts, 3), dtype=np.float64)
    thrust_moment_hist = np.zeros((n_pts, 3), dtype=np.float64)
    aero_force_cg_hist = np.zeros((n_pts, 3), dtype=np.float64)
    aero_moment_cg_hist = np.zeros((n_pts, 3), dtype=np.float64)
    kinetic_energy = np.zeros(n_pts, dtype=np.float64)
    potential_energy = np.zeros(n_pts, dtype=np.float64)
    aero_power = np.zeros(n_pts, dtype=np.float64)
    thrust_power = np.zeros(n_pts, dtype=np.float64)
    aero_work = np.zeros(n_pts, dtype=np.float64)
    thrust_work = np.zeros(n_pts, dtype=np.float64)
    angular_momentum = np.zeros((n_pts, 3), dtype=np.float64)
    subiterations = np.zeros(n_pts, dtype=np.int64)
    residuals = np.zeros(n_pts, dtype=np.float64)

    def _eval_step_controls(
        step_idx: int,
        q_vec: np.ndarray,
        qd_vec: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float], np.ndarray | None, np.ndarray | None]:
        """Evaluate control deflections [rad] and rates [rad/s] at *step_idx* for state *(q_vec, qd_vec)*."""
        if not ctrl_hist:
            return {}, {}, None, None
        t_now = float(time[step_idx])
        c_vals: dict[str, float] = {}
        c_rates: dict[str, float] = {}
        for cname in ctrl_hist:
            if not ctrl_takes_state.get(cname, False):
                c_vals[cname] = float(ctrl_hist[cname][step_idx])
                c_rates[cname] = float(ctrl_dot_hist[cname][step_idx])
            else:
                c_vals[cname] = float(
                    ctrl_hist[cname][step_idx - 1]
                    if step_idx > 0
                    else stepper.delta0_by_name.get(cname, 0.0)
                )

        if has_state_controls and controls is not None:
            st_for_ctrl = _make_state(t_now, q_vec, qd_vec, c_vals)
            for cname, spec in controls.items():
                if not ctrl_takes_state[cname]:
                    continue
                try:
                    val = float(spec(t_now, st_for_ctrl))  # type: ignore[misc]
                except InputError:
                    raise
                except Exception as exc:
                    raise InputError(
                        f"controls[{cname!r}] callable failed at t = {t_now:.6g} s: {exc}"
                    ) from exc
                if not math.isfinite(val):
                    raise InputError(
                        f"controls[{cname!r}] callable returned non-finite value {val!r} at t = {t_now:.6g} s."
                    )
                c_vals[cname] = val
                if step_idx == 0:
                    h_fd = 1.0e-7 * dt_val
                    v1 = float(spec(t_now + h_fd, st_for_ctrl))  # type: ignore[misc]
                    v2 = float(spec(t_now + 2.0 * h_fd, st_for_ctrl))  # type: ignore[misc]
                    rate = (-3.0 * val + 4.0 * v1 - v2) / (2.0 * h_fd)
                elif step_idx == 1:
                    rate = (val - float(ctrl_hist[cname][0])) / dt_val
                else:
                    rate = (
                        3.0 * val - 4.0 * float(ctrl_hist[cname][step_idx - 1]) + float(ctrl_hist[cname][step_idx - 2])
                    ) / (2.0 * dt_val)
                if not math.isfinite(rate):
                    raise InputError(
                        f"controls[{cname!r}] callable produced non-finite rate at t = {t_now:.6g} s."
                    )
                c_rates[cname] = rate

        if stepper.ctrl_rig is None:
            return c_vals, c_rates, None, None
        if all_frozen and not has_state_controls:
            return c_vals, c_rates, stepper.ctrl_rig.dd_edges[step_idx], stepper.ctrl_rig.rate_edges[step_idx]
        dd_s, rate_s = stepper.ctrl_rig.edges_for_values(c_vals, c_rates)
        stepper.ctrl_rig.dd_edges[step_idx] = dd_s
        stepper.ctrl_rig.rate_edges[step_idx] = rate_s
        return c_vals, c_rates, dd_s, rate_s

    def _dynamics_and_diagnostics(
        step_idx: int,
        q_vec: np.ndarray,
        qd_vec: np.ndarray,
        loads: _StepLoads,
        c_vals: Mapping[str, float],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Compute rigid-body accelerations, forces, moments, energy, power and angular momentum."""
        roll_n, pitch_n, yaw_n = float(q_vec[3]), float(q_vec[4]), float(q_vec[5])
        _check_gimbal_lock(pitch_n, context=f" at step {step_idx} (t = {float(time[step_idx]):.6g} s)")
        R = _rotation_matrix_321(roll_n, pitch_n, yaw_n)

        # Dimensional aerodynamic force and moment from solver coefficients.
        if aerodynamics_off:
            F_aero_g = np.zeros(3, dtype=np.float64)
            M_aero_ref_g = np.zeros(3, dtype=np.float64)
        else:
            F_aero_g = qS * (loads.CD * d_0 + loads.CY * S_0 + loads.CL * L_0)
            M_aero_ref_g = np.array(
                [-qS * b_ref_f * loads.Cl, qS * c_ref_f * loads.Cm, -qS * b_ref_f * loads.Cn],
                dtype=np.float64,
            )
        r_cg_g = R @ r_cg_rel_b
        M_aero_cg_g = M_aero_ref_g - np.cross(r_cg_g, F_aero_g)
        F_aero_I = E_mat.T @ F_aero_g
        M_aero_cg_I = E_mat.T @ M_aero_cg_g
        M_aero_cg_b = R.T @ M_aero_cg_g

        # Body-axes thrust force and moment about the centre of gravity.
        st_now = _make_state(float(time[step_idx]), q_vec, qd_vec, c_vals)
        F_th_b, M_th_raw_b = _eval_thrust(thrust, thrust_takes_state, float(time[step_idx]), st_now)
        M_th_cg_b = M_th_raw_b + np.cross(r_thrust_cg_b, F_th_b)
        F_th_g = R @ F_th_b
        F_th_I = E_mat.T @ F_th_g

        # Net force in inertial frame and net moment in body axes.
        F_net_I = F_aero_I + F_grav_I + F_th_I
        M_net_cg_b = M_aero_cg_b + M_th_cg_b
        F_net_b = R.T @ (F_aero_g + F_grav_g) + F_th_b
        M_net_cg_I = E_mat.T @ (R @ M_net_cg_b)

        # Accelerations of the six pose coordinates from the equations of motion
        # in the generalised coordinates u = [p_dot, eta_dot] (Kane's method;
        # Etkin and Reid 1996, chapter 4). p is the translation of ref_point and
        # the centre of gravity sits at r_b = cg - ref_point in body axes, so
        # v_cg = p_dot + J eta_dot and omega_b = S eta_dot. Each free DOF gives
        # one row of G^T [m a_cg - F; I omega_b_dot + omega_b x I omega_b - M] = 0.
        # The frozen DOFs follow their prescribed accelerations; their
        # constraint forces are not computed. With r_b = 0 these rows reduce
        # to m p_ddot = F and the Euler equation about the centre of gravity.
        eta_dot = qd_vec[3:6]
        S_mat = _euler_rate_matrix_body(roll_n, pitch_n)
        S_dot = _euler_rate_matrix_body_dot(roll_n, pitch_n, float(eta_dot[0]), float(eta_dot[1]))
        omega_b = S_mat @ eta_dot
        RE = E_mat.T @ R
        J_mat = -RE @ _skew(r_cg_rel_b) @ S_mat
        G_mat = np.zeros((6, 6), dtype=np.float64)
        G_mat[:3, :3] = np.eye(3)
        G_mat[:3, 3:] = J_mat
        G_mat[3:, 3:] = S_mat
        D_mat = np.zeros((6, 6), dtype=np.float64)
        D_mat[:3, :3] = mass_val * np.eye(3)
        D_mat[3:, 3:] = I_b
        Sd_eta = S_dot @ eta_dot
        a_cg_vel = RE @ (np.cross(Sd_eta, r_cg_rel_b) + np.cross(omega_b, np.cross(omega_b, r_cg_rel_b)))
        f_gen = np.concatenate(
            [
                F_net_I - mass_val * a_cg_vel,
                M_net_cg_b - np.cross(omega_b, I_b @ omega_b) - I_b @ Sd_eta,
            ]
        )
        M_gen = G_mat.T @ D_mat @ G_mat
        rhs_gen = G_mat.T @ f_gen
        qdd = qdd_presc[step_idx].copy()
        if free_idx.size:
            presc_idx = np.flatnonzero(~free_mask)
            rhs_free = rhs_gen[free_idx] - M_gen[np.ix_(free_idx, presc_idx)] @ qdd[presc_idx]
            qdd[free_idx] = np.linalg.solve(M_gen[np.ix_(free_idx, free_idx)], rhs_free)

        # Energy of the centre of gravity, generalised power of the free DOFs
        # and the angular momentum about the centre of gravity in the inertial frame.
        v_cg_I = qd_vec[:3] + J_mat @ eta_dot
        T_kin = 0.5 * mass_val * float(v_cg_I @ v_cg_I) + 0.5 * float(omega_b @ (I_b @ omega_b))
        z_cg = float(q_vec[2]) + float((RE @ r_cg_rel_b)[2])
        V_pot = mass_val * gravity_val * z_cg

        Q_aero = G_mat.T @ np.concatenate([F_aero_I, M_aero_cg_b])
        Q_thrust = G_mat.T @ np.concatenate([F_th_I, M_th_cg_b])
        P_aero = float(np.sum(Q_aero[free_idx] * qd_vec[free_idx]))
        P_thrust = float(np.sum(Q_thrust[free_idx] * qd_vec[free_idx]))

        H_inertial = RE @ (I_b @ omega_b)
        return (
            qdd,
            F_th_b,
            M_th_cg_b,
            F_aero_I,
            M_aero_cg_I,
            T_kin,
            V_pot,
            P_aero,
            P_thrust,
            H_inertial,
            F_net_I,
            M_net_cg_I,
            F_net_b,
            M_net_cg_b,
        )

    def _zero_loads() -> _StepLoads:
        return _StepLoads(
            CL=0.0,
            CD=0.0,
            CY=0.0,
            Cl=0.0,
            Cm=0.0,
            Cn=0.0,
            strip_cl=np.zeros(n_s, dtype=np.float64),
            gust_ref=np.zeros(3, dtype=np.float64),
            F_tot=np.zeros(3, dtype=np.float64),
            M_tot=np.zeros(3, dtype=np.float64),
        )

    # 5. Evaluate initial step n = 0 (t = 0).
    q0 = q_presc[0].copy()
    qd0 = qd_presc[0].copy()
    q0[free_mask] = q0_free[free_mask]
    qd0[free_mask] = qd0_free[free_mask]
    _check_gimbal_lock(float(q0[4]), context=" at t = 0 s")

    c_vals0, c_rates0, dd_step0, rate_step0 = _eval_step_controls(0, q0, qd0)
    for cname, val in c_vals0.items():
        ctrl_hist[cname][0] = val
        ctrl_dot_hist[cname][0] = c_rates0[cname]

    if ground is not None and not all_frozen:
        gc0 = stepper.min_ground_clearance(
            q0[0], q0[1], q0[2], q0[3], q0[4], q0[5], dd_step=dd_step0,
        )
        assert gc0 is not None
        h_min0, _ = gc0
        if h_min0 <= 0.0:
            kind_str = "Wing" if stepper.ctrl_rig is None else "Wing or flap"
            raise InputError(
                f"{kind_str} node reaches or crosses the ground plane at step 0 "
                f"(t = 0 s, minimum height {h_min0:.4g} m)."
            )

    if aerodynamics_off:
        loads0 = _zero_loads()
        stepper.step = 1
    elif all_frozen and not has_state_controls:
        loads0 = stepper.advance_prescribed(0)
    else:
        loads0 = stepper.advance(
            x=q0[0],
            y=q0[1],
            z=q0[2],
            roll=q0[3],
            pitch=q0[4],
            yaw=q0[5],
            x_dot=qd0[0],
            y_dot=qd0[1],
            z_dot=qd0[2],
            roll_dot=qd0[3],
            pitch_dot=qd0[4],
            yaw_dot=qd0[5],
            dd_step=dd_step0,
            rate_step=rate_step0,
        )

    (
        qdd0,
        F_th0,
        M_th0,
        F_aero0,
        M_aero0,
        T0,
        V0,
        Pa0,
        Pt0,
        H0,
        init_F_imb_I,
        init_M_imb_I,
        init_F_imb_b,
        init_M_imb_b,
    ) = _dynamics_and_diagnostics(0, q0, qd0, loads0, c_vals0)
    steady_imb = _steady_imbalance(
        steady_loads,
        qS=qS,
        b_ref=b_ref_f,
        c_ref=c_ref_f,
        d_0=d_0,
        S_0=S_0,
        L_0=L_0,
        E_mat=E_mat,
        F_grav_g=F_grav_g,
        r_cg_rel_b=r_cg_rel_b,
    )

    q_hist[0] = q0
    qd_hist[0] = qd0
    qdd_hist[0] = qdd0
    CL[0], CD[0], CY[0] = loads0.CL, loads0.CD, loads0.CY
    Cl[0], Cm[0], Cn[0] = loads0.Cl, loads0.Cm, loads0.Cn
    strip_cl[0] = loads0.strip_cl
    gust_ref[0] = loads0.gust_ref
    thrust_force_hist[0] = F_th0
    thrust_moment_hist[0] = M_th0
    aero_force_cg_hist[0] = F_aero0
    aero_moment_cg_hist[0] = M_aero0
    kinetic_energy[0] = T0
    potential_energy[0] = V0
    aero_power[0] = Pa0
    thrust_power[0] = Pt0
    angular_momentum[0] = H0
    subiterations[0] = 0
    residuals[0] = 0.0

    ground_contact = False
    ground_contact_step: int | None = None
    last_valid_step = n_steps_int
    omega_aitken_prev = np.ones(free_idx.size, dtype=np.float64)

    # 6. Time-marching loop over steps 1 .. n_steps.
    for step in range(1, n_steps_int + 1):
        if all_frozen:
            q_s = q_presc[step].copy()
            qd_s = qd_presc[step].copy()
            c_vals_s, c_rates_s, dd_s, rate_s = _eval_step_controls(step, q_s, qd_s)
            for cname, val in c_vals_s.items():
                ctrl_hist[cname][step] = val
                ctrl_dot_hist[cname][step] = c_rates_s[cname]

            if ground is not None and has_state_controls:
                gc_s = stepper.min_ground_clearance(
                    q_s[0], q_s[1], q_s[2], q_s[3], q_s[4], q_s[5], dd_step=dd_s,
                )
                assert gc_s is not None
                if gc_s[0] <= 0.0:
                    raise InputError(
                        f"Wing or flap node reaches or crosses the ground plane at step {step} "
                        f"(t = {float(time[step]):.6g} s, minimum height {gc_s[0]:.4g} m)."
                    )

            if aerodynamics_off:
                loads_s = _zero_loads()
                stepper.step = step + 1
            elif not has_state_controls:
                loads_s = stepper.advance_prescribed(step)
            else:
                loads_s = stepper.advance(
                    x=q_s[0],
                    y=q_s[1],
                    z=q_s[2],
                    roll=q_s[3],
                    pitch=q_s[4],
                    yaw=q_s[5],
                    x_dot=qd_s[0],
                    y_dot=qd_s[1],
                    z_dot=qd_s[2],
                    roll_dot=qd_s[3],
                    pitch_dot=qd_s[4],
                    yaw_dot=qd_s[5],
                    dd_step=dd_s,
                    rate_step=rate_s,
                )

            (
                qdd_s,
                F_th_s,
                M_th_s,
                F_aero_s,
                M_aero_s,
                T_s,
                V_s,
                Pa_s,
                Pt_s,
                H_s,
                _,
                _,
                _,
                _,
            ) = _dynamics_and_diagnostics(step, q_s, qd_s, loads_s, c_vals_s)

            q_hist[step] = q_s
            qd_hist[step] = qd_s
            qdd_hist[step] = qdd_s
            CL[step], CD[step], CY[step] = loads_s.CL, loads_s.CD, loads_s.CY
            Cl[step], Cm[step], Cn[step] = loads_s.Cl, loads_s.Cm, loads_s.Cn
            strip_cl[step] = loads_s.strip_cl
            gust_ref[step] = loads_s.gust_ref
            thrust_force_hist[step] = F_th_s
            thrust_moment_hist[step] = M_th_s
            aero_force_cg_hist[step] = F_aero_s
            aero_moment_cg_hist[step] = M_aero_s
            kinetic_energy[step] = T_s
            potential_energy[step] = V_s
            aero_power[step] = Pa_s
            thrust_power[step] = Pt_s
            aero_work[step] = aero_work[step - 1] + 0.5 * dt_val * (aero_power[step - 1] + Pa_s)
            thrust_work[step] = thrust_work[step - 1] + 0.5 * dt_val * (thrust_power[step - 1] + Pt_s)
            angular_momentum[step] = H_s
            subiterations[step] = 1
            residuals[step] = 0.0
            continue

        # Free-body step n -> n + 1 with second-order predictor, trapezoidal rule and Aitken relaxation.
        q_n = q_hist[step - 1]
        qd_n = qd_hist[step - 1]
        qdd_n = qdd_hist[step - 1]

        q_k = q_presc[step].copy()
        qd_k = qd_presc[step].copy()
        if step >= 2:
            qdd_nm1 = qdd_hist[step - 2]
            qd_k[free_mask] = qd_n[free_mask] + dt_val * (
                1.5 * qdd_n[free_mask] - 0.5 * qdd_nm1[free_mask]
            )
        else:
            qd_k[free_mask] = qd_n[free_mask] + dt_val * qdd_n[free_mask]
        q_k[free_mask] = q_n[free_mask] + 0.5 * dt_val * (qd_n[free_mask] + qd_k[free_mask])

        saved_stepper = stepper.save_state()
        omega_vec = omega_aitken_prev.copy()
        prev_r_qd: np.ndarray | None = None
        converged = False
        sub_count = 0
        res_norm = 0.0
        contact_triggered = False

        for sub in range(max_sub_int):
            sub_count = sub + 1
            c_vals_k, c_rates_k, dd_k, rate_k = _eval_step_controls(step, q_k, qd_k)

            if ground is not None:
                pitch_clamped = float(np.clip(q_k[4], -_GIMBAL_LOCK_PITCH_RAD, _GIMBAL_LOCK_PITCH_RAD))
                gc_k = stepper.min_ground_clearance(
                    q_k[0], q_k[1], q_k[2], q_k[3], pitch_clamped, q_k[5], dd_step=dd_k,
                )
                assert gc_k is not None
                h_min_k, pt_min_k = gc_k
                if h_min_k <= 0.0:
                    stepper.restore_state(saved_stepper)
                    stepper.notes.append(
                        f"Ground contact at step {step} (t = {float(time[step]):.6g} s): "
                        f"lowest node ({pt_min_k[0]:.4g}, {pt_min_k[1]:.4g}, {pt_min_k[2]:.4g}) m "
                        f"reached height {h_min_k:.4g} m above the ground plane; run stopped at step {step - 1}."
                    )
                    ground_contact = True
                    ground_contact_step = step
                    last_valid_step = step - 1
                    contact_triggered = True
                    break

            _check_gimbal_lock(
                float(q_k[4]),
                context=f" at step {step} (t = {float(time[step]):.6g} s)",
            )

            if aerodynamics_off:
                loads_k = _zero_loads()
                stepper.step = step + 1
            else:
                stepper.restore_state(saved_stepper)
                loads_k = stepper.advance(
                    x=q_k[0],
                    y=q_k[1],
                    z=q_k[2],
                    roll=q_k[3],
                    pitch=q_k[4],
                    yaw=q_k[5],
                    x_dot=qd_k[0],
                    y_dot=qd_k[1],
                    z_dot=qd_k[2],
                    roll_dot=qd_k[3],
                    pitch_dot=qd_k[4],
                    yaw_dot=qd_k[5],
                    dd_step=dd_k,
                    rate_step=rate_k,
                )

            (
                qdd_k,
                F_th_k,
                M_th_k,
                F_aero_k,
                M_aero_k,
                T_k,
                V_k,
                Pa_k,
                Pt_k,
                H_k,
                _,
                _,
                _,
                _,
            ) = _dynamics_and_diagnostics(step, q_k, qd_k, loads_k, c_vals_k)

            # Trapezoidal integration of free DOFs from step - 1 to step.
            qd_tilde = qd_k.copy()
            q_tilde = q_k.copy()
            qd_tilde[free_mask] = qd_n[free_mask] + 0.5 * dt_val * (qdd_n[free_mask] + qdd_k[free_mask])
            q_tilde[free_mask] = q_n[free_mask] + 0.5 * dt_val * (qd_n[free_mask] + qd_tilde[free_mask])

            # Relative residual across free DOFs (since q_tilde - q_k = 0.5 * dt * (qd_tilde - qd_k),
            # the relative change of both rate and displacement is measured by r_qd / scale_qd).
            r_qd = qd_tilde[free_idx] - qd_k[free_idx]
            n_f = free_idx.size
            scales_qd = np.empty(n_f, dtype=np.float64)
            trans_vel_scale = max(
                float(np.max(np.abs(qd_tilde[:3]))),
                float(np.max(np.abs(qd_n[:3]))),
                float(np.max(np.abs(q_tilde[:3]))) / dt_val,
            )
            rot_rate_scale = max(
                float(np.max(np.abs(qd_tilde[3:6]))),
                float(np.max(np.abs(qd_n[3:6]))),
                float(np.max(np.abs(q_tilde[3:6]))) / dt_val,
            )
            global_floor = max(1.0e-12 * max(trans_vel_scale, rot_rate_scale), 1.0e-30)
            for pos, dof_i in enumerate(free_idx):
                grp_s = trans_vel_scale if dof_i < 3 else rot_rate_scale
                scales_qd[pos] = max(
                    abs(float(qd_tilde[dof_i])),
                    abs(float(qd_n[dof_i])),
                    abs(float(q_tilde[dof_i])) / dt_val,
                    grp_s,
                    global_floor,
                )

            r_scaled = r_qd / scales_qd
            res_norm = float(np.max(np.abs(r_scaled)))
            if res_norm <= tol_val:
                q_k = q_tilde
                qd_k = qd_tilde
                converged = True
                break

            # Per-DOF Aitken dynamic relaxation (Irons and Tuck 1969; Kuttler and Wall 2008).
            if prev_r_qd is not None:
                dr_qd = r_qd - prev_r_qd
                for pos in range(n_f):
                    dr_i = float(dr_qd[pos])
                    if abs(dr_i) > 1.0e-150 * scales_qd[pos]:
                        w_new = -float(omega_vec[pos]) * float(prev_r_qd[pos]) / dr_i
                        omega_vec[pos] = float(np.clip(w_new, _AITKEN_OMEGA_MIN, _AITKEN_OMEGA_MAX))
            prev_r_qd = r_qd.copy()

            qd_k[free_idx] = qd_k[free_idx] + omega_vec * r_qd
            q_k[free_idx] = q_n[free_idx] + 0.5 * dt_val * (qd_n[free_idx] + qd_k[free_idx])

        if contact_triggered:
            break

        omega_aitken_prev = omega_vec.copy()

        # Final ground-clearance check on the accepted step state.
        if ground is not None:
            c_vals_k, c_rates_k, dd_k, rate_k = _eval_step_controls(step, q_k, qd_k)
            gc_fin = stepper.min_ground_clearance(
                q_k[0], q_k[1], q_k[2], q_k[3], q_k[4], q_k[5], dd_step=dd_k,
            )
            assert gc_fin is not None
            h_min_fin, pt_min_fin = gc_fin
            if h_min_fin <= 0.0:
                stepper.restore_state(saved_stepper)
                stepper.notes.append(
                    f"Ground contact at step {step} (t = {float(time[step]):.6g} s): "
                    f"lowest node ({pt_min_fin[0]:.4g}, {pt_min_fin[1]:.4g}, {pt_min_fin[2]:.4g}) m "
                    f"reached height {h_min_fin:.4g} m above the ground plane; run stopped at step {step - 1}."
                )
                ground_contact = True
                ground_contact_step = step
                last_valid_step = step - 1
                break

        if not converged:
            stepper.notes.append(
                f"Coupling sub-iterations did not reach coupling_tol = {tol_val:.3e} at step {step} "
                f"(t = {float(time[step]):.6g} s) after {max_sub_int} iterations "
                f"(final residual {res_norm:.3e}); continuing."
            )

        for cname, val in c_vals_k.items():
            ctrl_hist[cname][step] = val
            ctrl_dot_hist[cname][step] = c_rates_k[cname]

        (
            qdd_fin,
            F_th_fin,
            M_th_fin,
            F_aero_fin,
            M_aero_fin,
            T_fin,
            V_fin,
            Pa_fin,
            Pt_fin,
            H_fin,
            _,
            _,
            _,
            _,
        ) = _dynamics_and_diagnostics(step, q_k, qd_k, loads_k, c_vals_k)

        q_hist[step] = q_k
        qd_hist[step] = qd_k
        qdd_hist[step] = qdd_fin
        CL[step], CD[step], CY[step] = loads_k.CL, loads_k.CD, loads_k.CY
        Cl[step], Cm[step], Cn[step] = loads_k.Cl, loads_k.Cm, loads_k.Cn
        strip_cl[step] = loads_k.strip_cl
        gust_ref[step] = loads_k.gust_ref
        thrust_force_hist[step] = F_th_fin
        thrust_moment_hist[step] = M_th_fin
        aero_force_cg_hist[step] = F_aero_fin
        aero_moment_cg_hist[step] = M_aero_fin
        kinetic_energy[step] = T_fin
        potential_energy[step] = V_fin
        aero_power[step] = Pa_fin
        thrust_power[step] = Pt_fin
        aero_work[step] = aero_work[step - 1] + 0.5 * dt_val * (aero_power[step - 1] + Pa_fin)
        thrust_work[step] = thrust_work[step - 1] + 0.5 * dt_val * (thrust_power[step - 1] + Pt_fin)
        angular_momentum[step] = H_fin
        subiterations[step] = sub_count
        residuals[step] = res_norm

    end_idx = last_valid_step + 1
    ctrl_out = {cname: arr[:end_idx].copy() for cname, arr in ctrl_hist.items()}
    ctrl_dot_out = {cname: arr[:end_idx].copy() for cname, arr in ctrl_dot_hist.items()}
    kin_out = kinetic_energy[:end_idx].copy()
    pot_out = potential_energy[:end_idx].copy()
    aw_out = aero_work[:end_idx].copy()
    tw_out = thrust_work[:end_idx].copy()

    return _FreeFlightResponse(
        time=time[:end_idx].copy(),
        x=q_hist[:end_idx, 0].copy(),
        y=q_hist[:end_idx, 1].copy(),
        z=q_hist[:end_idx, 2].copy(),
        roll=q_hist[:end_idx, 3].copy(),
        pitch=q_hist[:end_idx, 4].copy(),
        yaw=q_hist[:end_idx, 5].copy(),
        theta=q_hist[:end_idx, 4].copy(),
        x_dot=qd_hist[:end_idx, 0].copy(),
        y_dot=qd_hist[:end_idx, 1].copy(),
        z_dot=qd_hist[:end_idx, 2].copy(),
        roll_dot=qd_hist[:end_idx, 3].copy(),
        pitch_dot=qd_hist[:end_idx, 4].copy(),
        yaw_dot=qd_hist[:end_idx, 5].copy(),
        theta_dot=qd_hist[:end_idx, 4].copy(),
        controls=ctrl_out,
        controls_dot=ctrl_dot_out,
        thrust_force=thrust_force_hist[:end_idx].copy(),
        thrust_moment=thrust_moment_hist[:end_idx].copy(),
        CL=CL[:end_idx].copy(),
        CD=CD[:end_idx].copy(),
        CY=CY[:end_idx].copy(),
        Cl=Cl[:end_idx].copy(),
        Cm=Cm[:end_idx].copy(),
        Cn=Cn[:end_idx].copy(),
        strip_cl=strip_cl[:end_idx].copy(),
        aero_force_cg=aero_force_cg_hist[:end_idx].copy(),
        aero_moment_cg=aero_moment_cg_hist[:end_idx].copy(),
        kinetic_energy=kin_out,
        potential_energy=pot_out,
        energy=kin_out + pot_out,
        aero_work=aw_out,
        thrust_work=tw_out,
        work=aw_out + tw_out,
        angular_momentum=angular_momentum[:end_idx].copy(),
        initial_force_imbalance=init_F_imb_I.copy(),
        initial_moment_imbalance=init_M_imb_I.copy(),
        initial_force_imbalance_body=init_F_imb_b.copy(),
        initial_moment_imbalance_body=init_M_imb_b.copy(),
        subiterations=subiterations[:end_idx].copy(),
        residuals=residuals[:end_idx].copy(),
        dofs=dofs_tuple,
        ground_contact=ground_contact,
        ground_contact_step=ground_contact_step,
        gamma_bound=stepper.gamma_bound(),
        gamma_wake=stepper.gamma_wake(last_valid_step),
        wake_corners=stepper.wake_corners(last_valid_step),
        gust_velocity=gust_ref[:end_idx].copy(),
        dt=dt_val,
        dt_default=stepper.dt_default,
        dx_w=stepper.dx_w,
        dx_te_mean=stepper.dx_te_mean,
        n_steps=last_valid_step,
        n_panels=stepper.n_panels,
        n_strips=n_s,
        n_chord=stepper.n_chord,
        te_correction=stepper.te_correction,
        c_ref=c_ref_f,
        b_ref=b_ref_f,
        S_ref=stepper.S_ref,
        ref_point=rp.copy(),
        cg=cg_vec.copy(),
        mass=mass_val,
        gravity=gravity_val,
        inertia=I_b.copy(),
        notes=list(stepper.notes),
        steady_force_imbalance=None if steady_imb is None else steady_imb[0],
        steady_moment_imbalance=None if steady_imb is None else steady_imb[1],
        steady_force_imbalance_body=None if steady_imb is None else steady_imb[2],
        steady_moment_imbalance_body=None if steady_imb is None else steady_imb[3],
        device=stepper.device,
        precision=stepper.precision,
    )
