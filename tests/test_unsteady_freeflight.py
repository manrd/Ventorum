# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the private free-body flight coupling of the time-domain vortex-ring solver.

Covers stepper state save and restore, round-off parity with
``_solve_time_rings`` when all degrees of freedom are frozen (in free air and
at ``h / c = 0.5`` with a moving flap and a convected gust), rigid-body free
fall and constant-thrust acceleration with aerodynamics off, torque-free 3-D
rotation conservation of kinetic energy and inertial angular momentum at second
order in ``dt`` plus intermediate-axis instability, the large-mass and
large-inertia frozen limit, trimmed short-period frequency and damping ratio
against ``dynamic_derivatives`` in free air and in ground effect, second-order
time-step convergence of the coupled short-period response, state-feedback
pitch damping through ``controls(t, state)``, time- and state-dependent thrust
with an offset application point, free-flight ground-contact termination, and
all input validation checks.

References
----------
* B. Etkin and L. D. Reid, "Dynamics of Flight: Stability and Control", 3rd
  ed., Wiley, 1996, chapters 4 and 6.
* B. L. Stevens and F. L. Lewis, "Aircraft Control and Simulation", 2nd ed.,
  Wiley, 2003, chapter 1.
* U. Kuttler and W. A. Wall, "Fixed-point fluid-structure interaction solvers
  with dynamic relaxation", Computational Mechanics 43, 2008, pp. 61-72.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import optimize

import ventorum as vt
from ventorum.aero.system import make_ground_plane
from ventorum.core.errors import InputError
from ventorum.geometry.lattice import build_lattice
from ventorum.unsteady.freeflight import (
    _FreeFlightResponse,
    _FreeFlightState,
    _solve_free_flight,
)
from ventorum.unsteady.gusts import one_minus_cosine
from ventorum.unsteady.timedomain import _solve_time_rings, _TimeDomainStepper


def _flapped_rect_wing(
    *,
    semi_span: float = 3.0,
    chord: float = 1.0,
    n_panels: int = 4,
    flap_name: str = "flap",
    hinge_x_c: float = 0.75,
    with_flap: bool = True,
) -> vt.Aircraft:
    """Build a flat rectangular wing with an optional full-span chordwise control surface."""
    controls = (
        [
            vt.ControlSurface(
                name=flap_name,
                eta_start=0.0,
                eta_end=1.0,
                hinge_x_c=hinge_x_c,
                deflection=0.0,
            )
        ]
        if with_flap
        else []
    )
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=semi_span,
        n_panels=n_panels,
        controls=controls,
        sections=[
            vt.WingSection(y_frac=0.0, chord=chord),
            vt.WingSection(y_frac=1.0, chord=chord),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25 * chord, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _wing_tail_aircraft(
    *,
    n_panels_wing: int = 4,
    n_panels_tail: int = 4,
    with_elevator: bool = True,
) -> vt.Aircraft:
    """Build a statically stable wing-tail aircraft with optional tail elevator."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        n_panels=n_panels_wing,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    tail_controls = (
        [
            vt.ControlSurface(
                name="elevator",
                eta_start=0.0,
                eta_end=1.0,
                hinge_x_c=0.5,
                deflection=0.0,
            )
        ]
        if with_elevator
        else []
    )
    tail = vt.LiftingSurface(
        name="tail",
        semi_span=1.0,
        n_panels=n_panels_tail,
        position=np.array([1.8, 0.0, 0.0]),
        controls=tail_controls,
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.5),
            vt.WingSection(y_frac=1.0, chord=0.5),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing, tail], ref_point=np.array([-0.15, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _fit_damped_sinusoid(
    t: np.ndarray,
    y: np.ndarray,
    omega_guess: float,
    sigma_guess: float,
) -> tuple[float, float, float]:
    """Fit ``A * exp(sigma * t) * cos(omega_d * t + phi) + c0`` to *y* by nonlinear least squares.

    Parameters
    ----------
    t : numpy.ndarray
        Time grid [s], shape ``(m,)``.
    y : numpy.ndarray
        Signal samples [-], shape ``(m,)``.
    omega_guess : float
        Initial estimate of damped angular frequency [rad/s].
    sigma_guess : float
        Initial estimate of decay rate ``sigma < 0`` [1/s].

    Returns
    -------
    omega_d : float
        Fitted damped frequency [rad/s].
    omega_n : float
        Fitted natural frequency [rad/s].
    zeta : float
        Fitted damping ratio ``-sigma / omega_n`` [-].
    """
    t0 = float(t[0])
    tau = t - t0
    amp0 = max(float(np.max(np.abs(y))), 1.0e-12)

    def _residual(p: np.ndarray) -> np.ndarray:
        sigma, omega_d, a_cos, a_sin, offset = p
        env = np.exp(sigma * tau)
        model = env * (a_cos * np.cos(omega_d * tau) + a_sin * np.sin(omega_d * tau)) + offset
        return (model - y) / amp0

    p0 = np.array([sigma_guess, omega_guess, float(y[0]), 0.0, 0.0], dtype=np.float64)
    res = optimize.least_squares(_residual, p0, ftol=1.0e-13, xtol=1.0e-13, gtol=1.0e-13)
    sigma_fit = float(res.x[0])
    omega_d_fit = abs(float(res.x[1]))
    omega_n_fit = math.hypot(sigma_fit, omega_d_fit)
    zeta_fit = -sigma_fit / omega_n_fit
    return omega_d_fit, omega_n_fit, zeta_fit


def _short_period_linear_eigenvalue(
    ac: vt.Aircraft,
    cond: vt.FlightCondition,
    st: vt.SolverSettings,
    *,
    mass: float,
    I_yy: float,
    k_eval: float = 0.0,
    pitch_damper_gain: float = 0.0,
    control_name: str | None = None,
    use_height_terms: bool = False,
) -> tuple[complex, float, float, float]:
    """Build the 2-DOF (heave z, pitch theta) linear model from ``dynamic_derivatives`` at *k_eval*.

    Returns ``(lambda_sp, omega_d, omega_n, zeta)`` where ``lambda_sp`` [1/s] is
    the short-period eigenvalue with positive imaginary part, ``omega_d``
    [rad/s] is the damped frequency, ``omega_n`` [rad/s] is the natural
    frequency, and ``zeta`` [-] is the damping ratio.
    """
    ctrls = (control_name,) if (control_name is not None and pitch_damper_gain != 0.0) else ()
    dd = vt.dynamic_derivatives(ac, cond, st, k=(float(k_eval),), wake_chords=8.0, controls=ctrls)

    V_inf = float(cond.V_inf)
    rho = float(cond.rho)
    S_ref = float(ac.S_ref)
    c_ref = float(ac.c_ref)
    q_inf = 0.5 * rho * V_inf * V_inf
    qS = q_inf * S_ref
    qSc = qS * c_ref
    t_scale = c_ref / (2.0 * V_inf)

    cl_a = float(dd.derivatives["CL_alpha"][0])
    cl_q = float(dd.derivatives["CL_q"][0])
    cl_ad = float(dd.derivatives["CL_alpha_dot"][0])
    cm_a = float(dd.derivatives["Cm_alpha"][0])
    cm_q = float(dd.derivatives["Cm_q"][0])
    cm_ad = float(dd.derivatives["Cm_alpha_dot"][0])

    if use_height_terms:
        cl_h = float(dd.derivatives["CL_h"][0])
        cm_h = float(dd.derivatives["Cm_h"][0])
        cl_hd = float(dd.derivatives["CL_h_dot"][0])
        cm_hd = float(dd.derivatives["Cm_h_dot"][0])
    else:
        cl_h = 0.0
        cm_h = 0.0
        cl_hd = -cl_a
        cm_hd = -cm_a

    if control_name is not None and pitch_damper_gain != 0.0:
        cl_d = float(dd.derivatives[f"CL_delta_{control_name}"][0])
        cl_dd = float(dd.derivatives[f"CL_delta_dot_{control_name}"][0])
        cm_d = float(dd.derivatives[f"Cm_delta_{control_name}"][0])
        cm_dd = float(dd.derivatives[f"Cm_delta_dot_{control_name}"][0])
    else:
        cl_d = 0.0
        cl_dd = 0.0
        cm_d = 0.0
        cm_dd = 0.0

    K_g = float(pitch_damper_gain)

    M2 = np.array(
        [
            [mass + qS * (t_scale / V_inf) * cl_ad, qS * t_scale * K_g * cl_dd],
            [qSc * (t_scale / V_inf) * cm_ad, I_yy + qSc * t_scale * K_g * cm_dd],
        ],
        dtype=np.float64,
    )
    K2 = np.array(
        [
            [qS * cl_h / c_ref, qS * cl_a],
            [qSc * cm_h / c_ref, qSc * cm_a],
        ],
        dtype=np.float64,
    )
    D2 = np.array(
        [
            [qS * cl_hd / V_inf, qS * (t_scale * (cl_q + cl_ad) - K_g * cl_d)],
            [qSc * cm_hd / V_inf, qSc * (t_scale * (cm_q + cm_ad) - K_g * cm_d)],
        ],
        dtype=np.float64,
    )

    M2_inv = np.linalg.inv(M2)
    A = np.zeros((4, 4), dtype=np.float64)
    A[0, 2] = 1.0
    A[1, 3] = 1.0
    A[2:, :2] = M2_inv @ K2
    A[2:, 2:] = M2_inv @ D2

    eigvals = np.linalg.eigvals(A)
    pos_im = [ev for ev in eigvals if ev.imag > 1.0e-6]
    assert len(pos_im) >= 1, f"Expected oscillatory short-period eigenvalue, got {eigvals}"
    lam = max(pos_im, key=lambda z: abs(z.imag))
    omega_d = float(abs(lam.imag))
    omega_n = float(abs(lam))
    zeta = float(-lam.real / omega_n)
    return complex(lam), omega_d, omega_n, zeta


# ---------------------------------------------------------------------------
# 1. Stepper save/restore and all-DOFs-frozen parity with _solve_time_rings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h_ground", [None, 0.5])
def test_stepper_save_restore_and_all_dofs_frozen_equals_solve_time_rings(h_ground):
    """All DOFs frozen equals _solve_time_rings to round-off (free air and h/c = 0.5, flap + gust)."""
    ac = _flapped_rect_wing(semi_span=3.0, chord=1.0, n_panels=4)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0), h=h_ground)
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )

    n_steps = 8
    omega = 2.0 * 0.25 * cond.V_inf / ac.c_ref
    motion = {
        "x": lambda t: 0.015 * math.sin(omega * t),
        "y": lambda t: -0.01 * math.sin(0.8 * omega * t + 0.1),
        "z": lambda t: 0.025 * math.sin(omega * t + 0.2),
        "roll": lambda t: math.radians(1.2) * math.sin(0.9 * omega * t),
        "pitch": lambda t: math.radians(1.8) * math.sin(omega * t + 0.3),
        "yaw": lambda t: math.radians(0.9) * math.sin(0.7 * omega * t - 0.1),
    }
    controls = {"flap": lambda t: math.radians(5.0) * math.sin(omega * t + 0.4)}
    gust = one_minus_cosine(amplitude=1.2, length=4.0, direction="vertical", x_front=0.0)

    res_ref = _solve_time_rings(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        motion=motion,
        controls=controls,
        gust=gust,
        n_steps=n_steps,
        ground=ground,
    )
    res_ff = _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=25.0,
        inertia=np.diag([12.0, 18.0, 28.0]),
        dofs=(),
        motion=motion,
        controls=controls,
        gust=gust,
        n_steps=n_steps,
        ground=ground,
    )

    assert isinstance(res_ff, _FreeFlightResponse)
    for attr in (
        "time",
        "CL",
        "CD",
        "CY",
        "Cl",
        "Cm",
        "Cn",
        "strip_cl",
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "x_dot",
        "y_dot",
        "z_dot",
        "roll_dot",
        "pitch_dot",
        "yaw_dot",
        "gamma_bound",
        "gamma_wake",
        "wake_corners",
        "gust_velocity",
    ):
        a = getattr(res_ff, attr)
        b = getattr(res_ref, attr)
        scale = max(float(np.max(np.abs(b))), 1.0)
        assert np.max(np.abs(a - b)) <= 1.0e-13 * scale, f"{attr} mismatch at h={h_ground}"
    assert np.max(np.abs(res_ff.controls["flap"] - res_ref.controls["flap"])) <= 1.0e-14
    assert np.max(np.abs(res_ff.controls_dot["flap"] - res_ref.controls_dot["flap"])) <= 1.0e-14

    # Direct check of _TimeDomainStepper save_state and restore_state with polluted intermediate trials.
    stepper = _TimeDomainStepper.from_prescribed(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        motion=motion,
        controls=controls,
        gust=gust,
        n_steps=n_steps,
        ground=ground,
    )
    cl_seq = np.empty(n_steps + 1, dtype=np.float64)
    cm_seq = np.empty(n_steps + 1, dtype=np.float64)
    for step in range(n_steps + 1):
        snap = stepper.save_state()
        # Run two bogus trial steps with perturbed pose, rates and flap deflection, then restore.
        for trial_scale in (0.5, -1.3):
            stepper.restore_state(snap)
            dd_bogus = None if stepper.ctrl_rig is None else trial_scale * stepper.ctrl_rig.dd_edges[step]
            rate_bogus = None if stepper.ctrl_rig is None else trial_scale * stepper.ctrl_rig.rate_edges[step]
            stepper.advance(
                x=float(res_ref.x[step]) + 0.04 * trial_scale,
                y=float(res_ref.y[step]) - 0.03 * trial_scale,
                z=float(res_ref.z[step]) + 0.02 * trial_scale,
                roll=float(res_ref.roll[step]) + 0.02 * trial_scale,
                pitch=float(res_ref.pitch[step]) - 0.03 * trial_scale,
                yaw=float(res_ref.yaw[step]) + 0.01 * trial_scale,
                x_dot=float(res_ref.x_dot[step]) + 0.5 * trial_scale,
                y_dot=float(res_ref.y_dot[step]) - 0.4 * trial_scale,
                z_dot=float(res_ref.z_dot[step]) + 0.6 * trial_scale,
                roll_dot=float(res_ref.roll_dot[step]) + 0.2 * trial_scale,
                pitch_dot=float(res_ref.pitch_dot[step]) - 0.3 * trial_scale,
                yaw_dot=float(res_ref.yaw_dot[step]) + 0.1 * trial_scale,
                dd_step=dd_bogus,
                rate_step=rate_bogus,
            )
        stepper.restore_state(snap)
        loads = stepper.advance_prescribed(step)
        cl_seq[step] = loads.CL
        cm_seq[step] = loads.Cm

    assert np.max(np.abs(cl_seq - res_ref.CL)) <= 1.0e-14
    assert np.max(np.abs(cm_seq - res_ref.Cm)) <= 1.0e-14
    assert np.max(np.abs(stepper.gamma_bound() - res_ref.gamma_bound)) <= 1.0e-13


# ---------------------------------------------------------------------------
# 2. Aerodynamics off: free fall in z and constant body-axes thrust in x
# ---------------------------------------------------------------------------


def test_aerodynamics_off_free_fall_and_constant_x_thrust():
    """With aerodynamics off, free fall equals z0 - g t^2 / 2 and constant x thrust gives exact parabola."""
    ac = _flapped_rect_wing(n_panels=2, with_flap=False)
    lat = build_lattice(ac, n_chord=2, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)
    mass = 15.0
    inertia = np.diag([4.0, 6.0, 9.0])
    g = 9.80665
    z0 = 25.0

    res_fall = _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=mass,
        inertia=inertia,
        gravity=g,
        dofs=("x", "y", "z", "roll", "pitch", "yaw"),
        initial_state={"z": z0},
        n_steps=20,
        dt=0.05,
        aerodynamics_off=True,
    )

    t = res_fall.time
    z_exact = z0 - 0.5 * g * t * t
    zd_exact = -g * t
    assert np.max(np.abs(res_fall.z - z_exact)) <= 1.0e-13 * max(abs(z0), 1.0)
    assert np.max(np.abs(res_fall.z_dot - zd_exact)) <= 1.0e-13 * (g * float(t[-1]))
    for zero_field in ("x", "y", "roll", "pitch", "yaw", "x_dot", "y_dot", "roll_dot", "pitch_dot", "yaw_dot"):
        assert np.max(np.abs(getattr(res_fall, zero_field))) <= 1.0e-15
    assert np.max(np.abs(res_fall.CL)) == 0.0
    assert np.max(np.abs(res_fall.energy - res_fall.energy[0])) <= 1.0e-12 * abs(res_fall.energy[0])

    # Constant body-axes thrust in x without rotation gives exact constant acceleration a_x = F_x / m.
    F_x = 90.0
    x0 = -1.25
    v0 = 3.5
    a_x = F_x / mass
    res_thrust = _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=mass,
        inertia=inertia,
        gravity=0.0,
        dofs=("x", "y", "z", "roll", "pitch", "yaw"),
        initial_state={"x": x0, "x_dot": v0},
        thrust=lambda t: (np.array([F_x, 0.0, 0.0]), np.zeros(3)),
        n_steps=20,
        dt=0.04,
        aerodynamics_off=True,
    )
    t_th = res_thrust.time
    x_exact = x0 + v0 * t_th + 0.5 * a_x * t_th * t_th
    xd_exact = v0 + a_x * t_th
    assert np.max(np.abs(res_thrust.x - x_exact)) <= 1.0e-13 * max(float(np.max(np.abs(x_exact))), 1.0)
    assert np.max(np.abs(res_thrust.x_dot - xd_exact)) <= 1.0e-13 * float(np.max(np.abs(xd_exact)))
    # Work of thrust equals the gain in kinetic energy to round-off.
    dE = res_thrust.energy - res_thrust.energy[0]
    assert np.max(np.abs(dE - res_thrust.thrust_work)) <= 1.0e-12 * max(float(np.max(np.abs(dE))), 1.0)


# ---------------------------------------------------------------------------
# 3. Aerodynamics off: torque-free 3-D rotation conservation & intermediate axis
# ---------------------------------------------------------------------------


def test_aerodynamics_off_torque_free_rotation_conservation_and_instability():
    """Torque-free 3-D rotation conserves kinetic energy and inertial angular momentum at O(dt^2)."""
    ac = _flapped_rect_wing(n_panels=2, with_flap=False)
    lat = build_lattice(ac, n_chord=2, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)
    mass = 10.0
    # Three distinct principal moments of inertia I_xx < I_yy < I_zz [kg m^2].
    inertia = np.diag([2.0, 5.0, 9.0])

    init_rot = {"roll_dot": 0.7, "pitch_dot": 0.5, "yaw_dot": -0.4}
    t_end = 0.6
    dts = (0.02, 0.01, 0.005)
    err_E = []
    err_H = []

    for dt_val in dts:
        n_steps = int(round(t_end / dt_val))
        res = _solve_free_flight(
            lat,
            cond,
            S_ref=ac.S_ref,
            b_ref=ac.b_ref,
            c_ref=ac.c_ref,
            ref_point=ac.ref_point,
            mass=mass,
            inertia=inertia,
            gravity=0.0,
            dofs=("roll", "pitch", "yaw"),
            initial_state=init_rot,
            n_steps=n_steps,
            dt=dt_val,
            aerodynamics_off=True,
            coupling_tol=1.0e-12,
        )
        E0 = float(res.kinetic_energy[0])
        H0 = res.angular_momentum[0]
        H0_norm = float(np.linalg.norm(H0))
        err_E.append(float(np.max(np.abs(res.kinetic_energy - E0))) / E0)
        err_H.append(float(np.max(np.linalg.norm(res.angular_momentum - H0[None, :], axis=1))) / H0_norm)

    order_E_1 = math.log2(err_E[0] / err_E[1])
    order_E_2 = math.log2(err_E[1] / err_E[2])
    order_H_1 = math.log2(err_H[0] / err_H[1])
    order_H_2 = math.log2(err_H[1] / err_H[2])

    assert 1.85 <= order_E_1 <= 2.15, f"Kinetic energy order (0.02->0.01)={order_E_1:.3f}, errs={err_E}"
    assert 1.85 <= order_E_2 <= 2.15, f"Kinetic energy order (0.01->0.005)={order_E_2:.3f}, errs={err_E}"
    assert 1.85 <= order_H_1 <= 2.15, f"Angular momentum order (0.02->0.01)={order_H_1:.3f}, errs={err_H}"
    assert 1.85 <= order_H_2 <= 2.15, f"Angular momentum order (0.01->0.005)={order_H_2:.3f}, errs={err_H}"

    # Qualitative check of intermediate-axis instability: spin about the x-axis (roll_dot = 3.0 rad/s)
    # when I_xx is the minor principal axis ([1, 5, 9]), major axis ([9, 5, 1]), or intermediate axis ([5, 9, 1]).
    n_inst = 80
    dt_inst = 0.02
    init_spin_x = {"roll_dot": 3.0, "pitch_dot": 0.02, "yaw_dot": 0.02}
    res_minor = _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=mass,
        inertia=np.diag([1.0, 5.0, 9.0]),
        gravity=0.0,
        dofs=("roll", "pitch", "yaw"),
        initial_state=init_spin_x,
        n_steps=n_inst,
        dt=dt_inst,
        aerodynamics_off=True,
    )
    res_major = _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=mass,
        inertia=np.diag([9.0, 5.0, 1.0]),
        gravity=0.0,
        dofs=("roll", "pitch", "yaw"),
        initial_state=init_spin_x,
        n_steps=n_inst,
        dt=dt_inst,
        aerodynamics_off=True,
    )
    res_inter = _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=mass,
        inertia=np.diag([5.0, 9.0, 1.0]),
        gravity=0.0,
        dofs=("roll", "pitch", "yaw"),
        initial_state=init_spin_x,
        n_steps=n_inst,
        dt=dt_inst,
        aerodynamics_off=True,
    )

    trans_minor = max(float(np.max(np.abs(res_minor.pitch_dot))), float(np.max(np.abs(res_minor.yaw_dot))))
    trans_major = max(float(np.max(np.abs(res_major.pitch_dot))), float(np.max(np.abs(res_major.yaw_dot))))
    trans_inter = max(float(np.max(np.abs(res_inter.pitch_dot))), float(np.max(np.abs(res_inter.yaw_dot))))
    assert trans_minor < 0.08
    assert trans_major < 0.08
    assert trans_inter > 2.0, f"Intermediate-axis rotation did not show instability: max transverse rate {trans_inter}"


def test_aerodynamics_off_offset_cg_follows_exact_parabola():
    """With the centre of gravity away from ref_point, the centre of gravity follows the free-fall parabola.

    All six DOFs are free, the aerodynamics are off and gravity acts. The pose
    is the motion of ref_point, so the rotation moves ref_point about the
    centre of gravity. The centre of gravity must follow
    ``p_cg0 + v_cg0 t - g t^2 / 2 e_z`` exactly; the integration error falls at
    second order with dt. The energy ``T + m g z_cg`` and the angular momentum
    about the centre of gravity are conserved to errors of second order.
    """
    from ventorum.unsteady.freeflight import _euler_rate_matrix_body
    from ventorum.unsteady.timedomain import _rotation_matrix_321

    ac = _flapped_rect_wing(n_panels=2, with_flap=False)
    lat = build_lattice(ac, n_chord=2, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)
    mass = 10.0
    inertia = np.array([[2.0, 0.0, 0.1], [0.0, 5.0, 0.0], [0.1, 0.0, 9.0]])
    rp = np.asarray(ac.ref_point, dtype=float)
    r_b = np.array([0.30, 0.10, -0.05])
    cg = rp + r_b
    g = 9.80665
    init = {"z_dot": 1.0, "x_dot": -0.5, "roll_dot": 0.7, "pitch_dot": 0.5, "yaw_dot": -0.4}
    t_end = 0.6
    errs_p, errs_E, errs_H = [], [], []
    for dt_val in (0.02, 0.01, 0.005):
        res = _solve_free_flight(
            lat,
            cond,
            S_ref=ac.S_ref,
            b_ref=ac.b_ref,
            c_ref=ac.c_ref,
            ref_point=rp,
            mass=mass,
            inertia=inertia,
            cg=cg,
            gravity=g,
            initial_state=init,
            n_steps=int(round(t_end / dt_val)),
            dt=dt_val,
            aerodynamics_off=True,
            coupling_tol=1.0e-13,
        )
        # At alpha = beta = 0 in free air the mean-flight frame equals geometry axes.
        p_cg = np.empty((res.time.size, 3))
        for i in range(res.time.size):
            R = _rotation_matrix_321(res.roll[i], res.pitch[i], res.yaw[i])
            p_cg[i] = np.array([res.x[i], res.y[i], res.z[i]]) + R @ r_b
        R0 = _rotation_matrix_321(res.roll[0], res.pitch[0], res.yaw[0])
        eta0 = np.array([res.roll_dot[0], res.pitch_dot[0], res.yaw_dot[0]])
        omega0 = _euler_rate_matrix_body(res.roll[0], res.pitch[0]) @ eta0
        v_cg0 = np.array([res.x_dot[0], res.y_dot[0], res.z_dot[0]]) + R0 @ np.cross(omega0, r_b)
        t = res.time[:, None]
        p_exact = p_cg[0] + v_cg0 * t + np.array([0.0, 0.0, -0.5 * g]) * t**2
        errs_p.append(float(np.max(np.linalg.norm(p_cg - p_exact, axis=1))))
        E = res.energy
        errs_E.append(float(np.max(np.abs(E - E[0]))) / float(res.kinetic_energy[0]))
        H = res.angular_momentum
        errs_H.append(float(np.max(np.linalg.norm(H - H[0], axis=1))) / float(np.linalg.norm(H[0])))
        assert not res.notes or all("did not reach" not in n for n in res.notes)

    for name, errs in (("cg path", errs_p), ("energy", errs_E), ("angular momentum", errs_H)):
        orders = [math.log2(errs[0] / errs[1]), math.log2(errs[1] / errs[2])]
        for order in orders:
            assert 1.85 <= order <= 2.15, f"{name}: orders {orders}, errors {errs}"
    # At dt = 0.005 s the cg path error is a small fraction of the 1.8 m fall.
    assert errs_p[2] < 1.0e-3, errs_p


# ---------------------------------------------------------------------------
# 4. Very large mass and inertia: motion and load bounds against frozen run
# ---------------------------------------------------------------------------


def test_very_large_mass_and_inertia_obeys_derived_bounds():
    """With very large mass and inertia, free motion stays below F_max T^2 / (2 m) and loads match frozen."""
    ac = _flapped_rect_wing(semi_span=3.0, chord=1.0, n_panels=4)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(4.0))

    n_steps = 8
    controls = {"flap": lambda t: math.radians(4.0) * math.sin(10.0 * t)}
    res_frozen = _solve_time_rings(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        controls=controls,
        n_steps=n_steps,
    )

    mass_huge = 1.0e12
    I_huge = 1.0e12
    res_huge = _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=mass_huge,
        inertia=I_huge * np.eye(3),
        gravity=0.0,
        dofs=("x", "y", "z", "roll", "pitch", "yaw"),
        controls=controls,
        n_steps=n_steps,
    )

    T_dur = float(res_frozen.time[-1])
    F_max = float(np.max(np.abs(res_huge.aero_force_cg)))
    M_max = float(np.max(np.abs(res_huge.aero_moment_cg)))
    assert F_max > 10.0 and M_max > 1.0

    # Analytical kinematic upper bounds from |a| <= F_max / m and |alpha| <= M_max / I:
    pos_bound = 0.5 * (F_max / mass_huge) * T_dur * T_dur * 1.05
    vel_bound = (F_max / mass_huge) * T_dur * 1.05
    ang_bound = 0.5 * (M_max / I_huge) * T_dur * T_dur * 1.05
    rate_bound = (M_max / I_huge) * T_dur * 1.05

    for pos_attr in ("x", "y", "z"):
        assert float(np.max(np.abs(getattr(res_huge, pos_attr)))) <= pos_bound
    for vel_attr in ("x_dot", "y_dot", "z_dot"):
        assert float(np.max(np.abs(getattr(res_huge, vel_attr)))) <= vel_bound
    for ang_attr in ("roll", "pitch", "yaw"):
        assert float(np.max(np.abs(getattr(res_huge, ang_attr)))) <= ang_bound
    for rate_attr in ("roll_dot", "pitch_dot", "yaw_dot"):
        assert float(np.max(np.abs(getattr(res_huge, rate_attr)))) <= rate_bound

    # Derived aerodynamic perturbation bound:
    # Effective incidence change is bounded by d_alpha_max = ang_bound + (vel_bound + rate_bound * b_ref) / V_inf,
    # multiplied by a conservative sensitivity factor 20 (covering lift/moment slopes and dGamma/dt / dt).
    d_alpha_max = ang_bound + (vel_bound + rate_bound * ac.b_ref) / cond.V_inf
    coeff_bound = 20.0 * d_alpha_max
    for coeff in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
        diff = float(np.max(np.abs(getattr(res_huge, coeff) - getattr(res_frozen, coeff))))
        assert diff <= coeff_bound, f"{coeff} diff={diff:.3e} > bound={coeff_bound:.3e}"


# ---------------------------------------------------------------------------
# 5 & 6. Trimmed short period in free air and in ground effect (h/c = 0.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h_ground", [None, pytest.param(0.5, marks=pytest.mark.slow)])
def test_trimmed_short_period_matches_dynamic_derivatives(h_ground):
    """Trimmed short-period frequency and damping agree with dynamic_derivatives in free air and ground effect.

    Configuration: wing-tail aircraft with CG at x = -0.15 m (ahead of the
    neutral point), mass = 400.0 kg, I_yy = 120.0 kg m^2, V_inf = 25.0 m/s,
    mean state alpha = 0, free DOFs ('z', 'pitch') with 'x' frozen. At
    t = 0, a constant thrust force and moment equal to minus the t = 0
    aerodynamic load plus weight trim the body exactly, so
    initial_force_imbalance and initial_moment_imbalance vanish to round-off.
    The exact trim at a lifting state (alpha = 2 deg) is also checked. The
    damped sinusoid is fitted to pitch for t >= 0.12 s over 110 convective
    steps (about 1.5 periods).

    Tolerance derivation:
    1. Equivalent constant-coefficient harmonic derivatives in
       dynamic_derivatives(k) assume purely sinusoidal motion exp(i omega_d t),
       whereas the short-period mode decays as exp((sigma + i omega_d) t) with
       damping ratio zeta ~ 0.36 to 0.39; the decaying wake memory shifts the
       effective unsteady derivatives by O(zeta * k) ~ 1.5 % in frequency and
       ~0.018 in damping ratio.
    2. Time-step discretisation (dt = dx_TE / V_inf at n_chord = 4) and
       second-order backward difference for dGamma/dt contribute ~0.5 % in
       frequency and ~0.005 in damping ratio.
    Combined tolerance: 3.0 % relative on damped frequency omega_d and 0.025
    absolute on damping ratio zeta.
    """
    ac = _wing_tail_aircraft(n_panels_wing=4, n_panels_tail=4, with_elevator=True)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, spacing="half-cosine", chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=0.0, h=h_ground)
    lat = build_lattice(ac, st, collocation="vlm", n_chord=4, chord_spacing="uniform")
    ground = (
        make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )

    mass = 400.0
    I_yy = 120.0
    inertia = np.diag([80.0, I_yy, 180.0])
    g = 9.80665

    # 1. Linear short-period eigenvalue from dynamic_derivatives, iterating k once.
    use_h = h_ground is not None
    _, omega_d_0, _, _ = _short_period_linear_eigenvalue(
        ac, cond, st, mass=mass, I_yy=I_yy, k_eval=0.0, use_height_terms=use_h,
    )
    k_sp = omega_d_0 * ac.c_ref / (2.0 * cond.V_inf)
    lam_lin, omega_d_lin, omega_n_lin, zeta_lin = _short_period_linear_eigenvalue(
        ac, cond, st, mass=mass, I_yy=I_yy, k_eval=k_sp, use_height_terms=use_h,
    )

    # 2. Verify initial trim balance at a lifting condition (alpha = 2 deg) as well as alpha = 0.
    cond_lift = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=np.radians(2.0), h=h_ground)
    ground_lift = (
        make_ground_plane(lat, h_ground, cond_lift.alpha, cond_lift.beta, ref_point=ac.ref_point, height_ref="ref")
        if h_ground is not None
        else None
    )
    res_t0_lift = _solve_free_flight(
        lat,
        cond_lift,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=mass,
        inertia=inertia,
        gravity=g,
        dofs=(),
        n_steps=1,
        ground=ground_lift,
    )
    F_trim_lift = -res_t0_lift.initial_force_imbalance_body
    M_trim_lift = -res_t0_lift.initial_moment_imbalance_body
    res_trimmed_lift = _solve_free_flight(
        lat,
        cond_lift,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=mass,
        inertia=inertia,
        gravity=g,
        dofs=("z", "pitch"),
        thrust=lambda t: (F_trim_lift, M_trim_lift),
        n_steps=1,
        ground=ground_lift,
    )
    assert np.max(np.abs(res_trimmed_lift.initial_force_imbalance)) <= 1.0e-11
    assert np.max(np.abs(res_trimmed_lift.initial_moment_imbalance)) <= 1.0e-11

    # 3. Perturbed short-period run about the mean state alpha = 0, trimmed exactly at t = 0 with a
    #    constant thrust force and moment equal to minus the t = 0 aerodynamic load plus weight.
    res_t0 = _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=mass,
        inertia=inertia,
        gravity=g,
        dofs=(),
        n_steps=1,
        ground=ground,
    )
    F_trim_b = -res_t0.initial_force_imbalance_body
    M_trim_b = -res_t0.initial_moment_imbalance_body
    thrust_fn = lambda t: (F_trim_b, M_trim_b)  # noqa: E731
    q0 = 1.0e-3
    n_steps = 110
    res_pert = _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=mass,
        inertia=inertia,
        gravity=g,
        dofs=("z", "pitch"),
        initial_state={"pitch_dot": q0},
        thrust=thrust_fn,
        n_steps=n_steps,
        ground=ground,
    )
    # The trim is exact on the free path when no perturbation is applied. The
    # perturbed run starts with a pitch rate, so its t = 0 load differs from the
    # trim load by the pitch-rate load, and its imbalance is not zero.
    res_trim_free = _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=mass,
        inertia=inertia,
        gravity=g,
        dofs=("z", "pitch"),
        thrust=thrust_fn,
        n_steps=1,
        ground=ground,
    )
    assert np.max(np.abs(res_trim_free.initial_force_imbalance)) <= 1.0e-9
    assert np.max(np.abs(res_trim_free.initial_moment_imbalance)) <= 1.0e-9

    mask = res_pert.time >= 0.12
    omega_d_fit, omega_n_fit, zeta_fit = _fit_damped_sinusoid(
        res_pert.time[mask], res_pert.pitch[mask], omega_d_lin, lam_lin.real,
    )

    rel_err_omega = abs(omega_d_fit - omega_d_lin) / omega_d_lin
    abs_err_zeta = abs(zeta_fit - zeta_lin)
    assert rel_err_omega <= 0.03, (
        f"h={h_ground}: omega_d_fit={omega_d_fit:.4f}, omega_d_lin={omega_d_lin:.4f}, rel_err={rel_err_omega:.4%}"
    )
    assert abs_err_zeta <= 0.025, (
        f"h={h_ground}: zeta_fit={zeta_fit:.4f}, zeta_lin={zeta_lin:.4f}, abs_err={abs_err_zeta:.4f}"
    )
    _ = omega_n_lin, omega_n_fit


# ---------------------------------------------------------------------------
# 7. Short-period time-step convergence (second order in dt) & sub-iterations
# ---------------------------------------------------------------------------


def test_short_period_dt_convergence_second_order_and_subiteration_stats():
    """Coupled response converges at second order in dt when the aerodynamic load is small.

    The density is rho = 0.01 kg/m^3, so that the rigid-body terms (a
    state-feedback spring and damper in the thrust) dominate and the
    aerodynamic load is a small coupled perturbation. This isolates the
    order of the predictor, trapezoidal rule and Aitken sub-iterations. At
    sea-level density the prescribed-wake marcher itself limits the order
    (see the slow companion test).
    """
    ac = _wing_tail_aircraft(n_panels_wing=3, n_panels_tail=2, with_elevator=False)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=25.0, rho=0.01, alpha=0.0)

    mass = 600.0
    I_yy = 180.0
    inertia = np.diag([100.0, I_yy, 260.0])
    g = 9.80665

    def _trim_spring_thrust(t_val: float, state: _FreeFlightState) -> tuple[np.ndarray, np.ndarray]:
        _ = t_val
        return (
            np.array([0.0, 0.0, mass * g - 36.0 * mass * state.z - 2.0 * mass * state.z_dot]),
            np.array([0.0, -49.0 * I_yy * state.pitch - 3.0 * I_yy * state.pitch_dot, 0.0]),
        )

    # Base step dt_0 = 0.02 s over t_end = 0.32 s: refine dt = 0.02, 0.01, 0.005, 0.0025 s.
    t_end = 0.32
    dts = (0.02, 0.01, 0.005, 0.0025)
    runs = []
    for dt_val in dts:
        n_steps = int(round(t_end / dt_val))
        res = _solve_free_flight(
            lat,
            cond,
            S_ref=ac.S_ref,
            b_ref=ac.b_ref,
            c_ref=ac.c_ref,
            ref_point=ac.ref_point,
            mass=mass,
            inertia=inertia,
            gravity=g,
            dofs=("z", "pitch"),
            initial_state={"pitch_dot": 0.02, "z_dot": 0.1},
            thrust=_trim_spring_thrust,
            n_steps=n_steps,
            dt=dt_val,
            coupling_tol=1.0e-11,
        )
        runs.append(res)
        # Every step converges below coupling_tol within max_subiterations.
        assert int(np.max(res.subiterations[1:])) < 20
        assert float(np.max(res.residuals[1:])) <= 1.0e-11

    ref = runs[-1]
    scale_pitch = float(np.max(np.abs(ref.pitch)))
    scale_z = float(np.max(np.abs(ref.z)))
    err_pitch = []
    err_z = []
    for idx, stride in ((0, 8), (1, 4), (2, 2)):
        r_c = runs[idx]
        assert np.allclose(r_c.time, ref.time[::stride], atol=1.0e-14)
        err_pitch.append(float(np.max(np.abs(r_c.pitch - ref.pitch[::stride]))) / scale_pitch)
        err_z.append(float(np.max(np.abs(r_c.z - ref.z[::stride]))) / scale_z)

    order_pitch = math.log2(err_pitch[0] / err_pitch[1])
    order_z = math.log2(err_z[0] / err_z[1])
    assert 1.75 <= order_pitch <= 2.35, f"Pitch dt order={order_pitch:.3f}, errs={err_pitch}"
    assert 1.75 <= order_z <= 2.35, f"Heave dt order={order_z:.3f}, errs={err_z}"

    mean_sub = float(np.mean(runs[1].subiterations[1:]))
    max_sub = int(np.max(runs[1].subiterations[1:]))
    assert 2.0 <= mean_sub <= 8.0
    assert 2 <= max_sub <= 12


@pytest.mark.slow
def test_short_period_dt_convergence_at_sea_level_density_records_order():
    """At rho = 1.225 kg/m^3 the coupled short period converges as dt halves (order recorded).

    Observed order of the pitch history relative to the finest run is about
    1.1 to 1.2 (not 2). The prescribed-motion marcher ``_solve_time_rings``
    alone shows an order below 1 in CL and Cm at the same mesh, so the
    aerodynamic step, not the rigid-body integrator, limits the order. The
    test asserts convergence (errors fall with dt, order >= 0.9) and that
    every step converges in the sub-iterations.
    """
    ac = _wing_tail_aircraft(n_panels_wing=3, n_panels_tail=2, with_elevator=False)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=0.0)
    mass = 400.0
    I_yy = 120.0
    inertia = np.diag([80.0, I_yy, 180.0])
    g = 9.80665
    thrust_fn = lambda t: (np.array([0.0, 0.0, mass * g]), np.zeros(3))  # noqa: E731

    t_end = 0.32
    dts = (0.02, 0.01, 0.005, 0.0025)
    runs = []
    for dt_val in dts:
        res = _solve_free_flight(
            lat,
            cond,
            S_ref=ac.S_ref,
            b_ref=ac.b_ref,
            c_ref=ac.c_ref,
            ref_point=ac.ref_point,
            mass=mass,
            inertia=inertia,
            gravity=g,
            dofs=("z", "pitch"),
            initial_state={"pitch_dot": 1.0e-2},
            thrust=thrust_fn,
            n_steps=int(round(t_end / dt_val)),
            dt=dt_val,
            coupling_tol=1.0e-11,
        )
        assert int(np.max(res.subiterations[1:])) < 20
        assert float(np.max(res.residuals[1:])) <= 1.0e-11
        runs.append(res)

    ref = runs[-1]
    scale_pitch = float(np.max(np.abs(ref.pitch)))
    err_pitch = [
        float(np.max(np.abs(runs[idx].pitch - ref.pitch[::stride]))) / scale_pitch
        for idx, stride in ((0, 8), (1, 4), (2, 2))
    ]
    assert err_pitch[0] > err_pitch[1] > err_pitch[2]
    order_pitch = math.log2(err_pitch[0] / err_pitch[1])
    assert order_pitch >= 0.9, f"Pitch dt order={order_pitch:.3f}, errs={err_pitch}"


# ---------------------------------------------------------------------------
# 8. State-feedback pitch damper delta = -K * q through controls(t, state)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_state_feedback_pitch_damper_matches_linear_model():
    """Pitch damper delta = -K * q through controls(t, state) increases zeta matching dynamic_derivatives."""
    ac = _wing_tail_aircraft(n_panels_wing=4, n_panels_tail=4, with_elevator=True)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, spacing="half-cosine", chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=0.0)
    lat = build_lattice(ac, st, collocation="vlm", n_chord=4, chord_spacing="uniform")

    mass = 400.0
    I_yy = 120.0
    inertia = np.diag([80.0, I_yy, 180.0])
    g = 9.80665
    # With AVL trailing-edge-down sign convention, Cm_delta_elevator < 0 (nose-down moment for delta > 0),
    # so delta = -K_damper * pitch_dot with K_damper = -0.10 s deflects trailing-edge down when pitching up.
    K_damper = -0.10

    # Open-loop and closed-loop linear predictions (iterating k once).
    _, wd0_open, _, _ = _short_period_linear_eigenvalue(ac, cond, st, mass=mass, I_yy=I_yy, k_eval=0.0)
    k_open = wd0_open * ac.c_ref / (2.0 * cond.V_inf)
    lam_open, wd_open_lin, _, zeta_open_lin = _short_period_linear_eigenvalue(
        ac, cond, st, mass=mass, I_yy=I_yy, k_eval=k_open,
    )

    _, wd0_cl, _, _ = _short_period_linear_eigenvalue(
        ac, cond, st, mass=mass, I_yy=I_yy, k_eval=0.0, pitch_damper_gain=K_damper, control_name="elevator",
    )
    k_cl = wd0_cl * ac.c_ref / (2.0 * cond.V_inf)
    lam_cl, wd_cl_lin, _, zeta_cl_lin = _short_period_linear_eigenvalue(
        ac, cond, st, mass=mass, I_yy=I_yy, k_eval=k_cl, pitch_damper_gain=K_damper, control_name="elevator",
    )
    assert zeta_cl_lin > zeta_open_lin + 0.10

    thrust_fn = lambda t: (np.array([0.0, 0.0, mass * g]), np.zeros(3))  # noqa: E731
    q0 = 1.0e-3
    n_steps = 110

    res_open = _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=mass,
        inertia=inertia,
        gravity=g,
        dofs=("z", "pitch"),
        initial_state={"pitch_dot": q0},
        thrust=thrust_fn,
        n_steps=n_steps,
    )
    res_closed = _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=mass,
        inertia=inertia,
        gravity=g,
        dofs=("z", "pitch"),
        initial_state={"pitch_dot": q0},
        thrust=thrust_fn,
        controls={"elevator": lambda t, state: -K_damper * state.pitch_dot},
        n_steps=n_steps,
    )

    assert np.allclose(res_closed.controls["elevator"], -K_damper * res_closed.pitch_dot, atol=1.0e-12)
    mask = res_open.time >= 0.12
    wd_open_fit, _, zeta_open_fit = _fit_damped_sinusoid(
        res_open.time[mask], res_open.pitch[mask], wd_open_lin, lam_open.real,
    )
    wd_cl_fit, _, zeta_cl_fit = _fit_damped_sinusoid(
        res_closed.time[mask], res_closed.pitch[mask], wd_cl_lin, lam_cl.real,
    )

    assert zeta_cl_fit > zeta_open_fit + 0.10
    assert abs(wd_cl_fit - wd_cl_lin) / wd_cl_lin <= 0.03
    assert abs(zeta_cl_fit - zeta_cl_lin) <= 0.025
    d_zeta_fit = zeta_cl_fit - zeta_open_fit
    d_zeta_lin = zeta_cl_lin - zeta_open_lin
    assert abs(d_zeta_fit - d_zeta_lin) <= 0.025


# ---------------------------------------------------------------------------
# 9. Thrust as callable of t and (t, state), and thrust_point offset moment
# ---------------------------------------------------------------------------


def test_thrust_callable_signatures_and_offset_thrust_point():
    """Thrust works as thrust(t) and thrust(t, state), and thrust_point off CG produces r x F moment."""
    ac = _flapped_rect_wing(n_panels=2, with_flap=False)
    lat = build_lattice(ac, n_chord=2, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)
    mass = 10.0
    I_yy = 5.0
    inertia = np.diag([3.0, I_yy, 6.0])
    cg = np.array([0.30, 0.0, 0.0])

    # 1. thrust(t) with thrust_point offset: F = [0, 0, -20 N] at x = 0.80 m (0.50 m aft of CG at 0.30 m).
    # In geometry axes (x aft, y right, z up), r = thrust_point - cg = [0.50, 0, 0],
    # so M_cg = r x F = [0.50, 0, 0] x [0, 0, -20] = [0, +10.0, 0] N m (nose-up pitching moment).
    thrust_pt = np.array([0.80, 0.0, 0.0])
    F_z = -20.0
    expected_My = 0.50 * (-F_z)  # +10.0 N m
    # Only pitch is free, so the frozen translations hold ref_point fixed and
    # the body pivots about ref_point (x = 0.25 m), with the centre of gravity
    # 0.05 m aft of it. The pitch equation is then taken about the pivot: the
    # thrust moment about ref_point over the pitch inertia about ref_point
    # (parallel-axis term m r^2). The acceleration is constant, so the
    # trapezoidal rule is exact.
    r_pivot_cg = cg - np.asarray(ac.ref_point, dtype=float)
    r_pivot_thrust = thrust_pt - np.asarray(ac.ref_point, dtype=float)
    M_pivot = float(np.cross(r_pivot_thrust, np.array([0.0, 0.0, F_z]))[1])  # +11.0 N m
    I_pivot = I_yy + mass * float(r_pivot_cg[0] ** 2 + r_pivot_cg[2] ** 2)
    alpha_y = M_pivot / I_pivot  # about +2.189 rad/s^2

    res_pt = _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=mass,
        inertia=inertia,
        cg=cg,
        gravity=0.0,
        dofs=("pitch",),
        thrust=lambda t: (np.array([0.0, 0.0, F_z]), np.zeros(3)),
        thrust_point=thrust_pt,
        n_steps=10,
        dt=0.02,
        aerodynamics_off=True,
    )
    t = res_pt.time
    assert np.allclose(res_pt.thrust_moment[:, 1], expected_My, atol=1.0e-14)
    assert np.allclose(res_pt.pitch, 0.5 * alpha_y * t * t, atol=1.0e-13)
    assert np.allclose(res_pt.pitch_dot, alpha_y * t, atol=1.0e-13)

    # 2. thrust(t, state) state-dependent damping force F_x = -c_d * state.x_dot:
    # m * x_ddot = -c_d * x_dot -> x_dot(t) = v0 * exp(-(c_d / m) * t).
    c_damp = 5.0
    v0 = 2.0

    def _state_thrust(t_val: float, state: _FreeFlightState) -> tuple[np.ndarray, np.ndarray]:
        assert isinstance(state, _FreeFlightState)
        assert math.isclose(state.t, t_val, rel_tol=0.0, abs_tol=1.0e-15)
        return np.array([-c_damp * state.x_dot, 0.0, 0.0]), np.zeros(3)

    res_st = _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=mass,
        inertia=inertia,
        gravity=0.0,
        dofs=("x",),
        initial_state={"x_dot": v0},
        thrust=_state_thrust,
        n_steps=20,
        dt=0.02,
        aerodynamics_off=True,
    )
    xd_exact = v0 * np.exp(-(c_damp / mass) * res_st.time)
    assert np.max(np.abs(res_st.x_dot - xd_exact)) <= 5.0e-5


# ---------------------------------------------------------------------------
# 10. Ground contact in free flight stops run with note and keeps valid steps
# ---------------------------------------------------------------------------


def test_ground_contact_in_free_flight_stops_with_note():
    """A free descent at h/c = 0.5 stops when a node reaches the ground and returns the valid steps."""
    ac = _flapped_rect_wing(semi_span=3.0, chord=1.0, n_panels=4)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    h_ground = 0.5
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0, h=h_ground)
    ground = make_ground_plane(lat, h_ground, cond.alpha, cond.beta, ref_point=ac.ref_point, height_ref="ref")

    # With dt_default = 0.25 / 20 = 0.0125 s and downward velocity z_dot = -10.0 m/s,
    # the descending wing at h = 0.5 m reaches the ground plane inside 10 steps.
    res = _solve_free_flight(
        lat,
        cond,
        S_ref=ac.S_ref,
        b_ref=ac.b_ref,
        c_ref=ac.c_ref,
        ref_point=ac.ref_point,
        mass=250.0,
        inertia=np.diag([50.0, 75.0, 100.0]),
        gravity=9.80665,
        dofs=("z", "pitch"),
        initial_state={"z_dot": -10.0},
        n_steps=15,
        ground=ground,
    )

    assert res.ground_contact is True
    assert res.ground_contact_step is not None
    assert 1 <= res.ground_contact_step < 15
    assert res.n_steps == res.ground_contact_step - 1
    assert res.time.shape == (res.n_steps + 1,)
    assert res.z.shape == (res.n_steps + 1,)
    assert res.CL.shape == (res.n_steps + 1,)
    assert res.gamma_wake.shape[0] == res.n_steps
    assert res.wake_corners.shape[0] == res.n_steps
    # Last returned step is strictly above the ground.
    assert float(res.z[-1]) > -h_ground
    assert any(
        "ground" in note.lower() and f"step {res.ground_contact_step}" in note.lower()
        for note in res.notes
    ), f"Expected ground-contact note naming step {res.ground_contact_step}, got {res.notes}"


# ---------------------------------------------------------------------------
# 11. Every input check
# ---------------------------------------------------------------------------


def test_all_free_flight_input_checks():
    """Invalid DOFs, mass, inertia, shapes, thrust and gimbal lock raise InputError."""
    ac = _flapped_rect_wing(n_panels=2, with_flap=False)
    lat = build_lattice(ac, n_chord=2, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)
    base_kw = {
        "S_ref": ac.S_ref,
        "b_ref": ac.b_ref,
        "c_ref": ac.c_ref,
        "ref_point": ac.ref_point,
        "mass": 10.0,
        "inertia": np.diag([2.0, 4.0, 6.0]),
        "n_steps": 2,
    }

    # Unknown DOF, string passed as dofs, duplicate DOF, free DOF also in motion, frozen/unknown in initial_state.
    with pytest.raises(InputError, match="dofs|DOF"):
        _solve_free_flight(lat, cond, dofs=("heave",), **base_kw)
    with pytest.raises(InputError, match="dofs|DOF"):
        _solve_free_flight(lat, cond, dofs="z", **base_kw)  # type: ignore[arg-type]
    with pytest.raises(InputError, match="duplicate|unique"):
        _solve_free_flight(lat, cond, dofs=("z", "z"), **base_kw)
    with pytest.raises(InputError, match="motion"):
        _solve_free_flight(lat, cond, dofs=("z",), motion={"z": np.zeros(3)}, **base_kw)
    with pytest.raises(InputError, match="initial_state"):
        _solve_free_flight(lat, cond, dofs=("z",), initial_state={"x": 1.0}, **base_kw)
    with pytest.raises(InputError, match="initial_state"):
        _solve_free_flight(lat, cond, dofs=("z",), initial_state={"unknown": 1.0}, **base_kw)
    with pytest.raises(InputError, match="initial_state"):
        _solve_free_flight(lat, cond, dofs=("z",), initial_state={"z": float("nan")}, **base_kw)

    # Non-positive, non-finite or bool mass and gravity.
    for bad_mass in (0.0, -5.0, float("nan"), float("inf"), True):
        kw = {**base_kw, "mass": bad_mass}
        with pytest.raises(InputError, match="mass"):
            _solve_free_flight(lat, cond, dofs=("z",), **kw)
    for bad_g in (-1.0, float("nan"), float("inf"), True):
        with pytest.raises(InputError, match="gravity"):
            _solve_free_flight(lat, cond, dofs=("z",), gravity=bad_g, **base_kw)

    # Inertia tensor checks: wrong shape, non-finite, non-symmetric, non-positive-definite.
    for bad_I in (
        np.ones((3, 2)),
        np.array([1.0, 2.0, 3.0]),
        np.diag([1.0, float("nan"), 3.0]),
        np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),  # non-symmetric
        np.diag([2.0, 0.0, 3.0]),  # semi-definite (zero eigenvalue)
        np.diag([2.0, -1.0, 3.0]),  # negative eigenvalue
    ):
        kw = {**base_kw, "inertia": bad_I}
        with pytest.raises(InputError, match="inertia"):
            _solve_free_flight(lat, cond, dofs=("pitch",), **kw)

    # Wrong shapes or non-finite cg and thrust_point.
    with pytest.raises(InputError, match="cg"):
        _solve_free_flight(lat, cond, dofs=("z",), cg=np.array([0.0, 0.0]), **base_kw)
    with pytest.raises(InputError, match="cg"):
        _solve_free_flight(lat, cond, dofs=("z",), cg=np.array([0.0, float("nan"), 0.0]), **base_kw)
    with pytest.raises(InputError, match="thrust_point"):
        _solve_free_flight(lat, cond, dofs=("z",), thrust_point=np.array([0.0, 0.0]), **base_kw)
    with pytest.raises(InputError, match="thrust_point"):
        _solve_free_flight(lat, cond, dofs=("z",), thrust_point=np.array([0.0, float("inf"), 0.0]), **base_kw)

    # Invalid thrust callable, wrong thrust shapes, non-finite thrust force or moment.
    with pytest.raises(InputError, match="thrust"):
        _solve_free_flight(lat, cond, dofs=("z",), thrust="not_callable", **base_kw)  # type: ignore[arg-type]
    with pytest.raises(InputError, match="thrust"):
        _solve_free_flight(lat, cond, dofs=("z",), thrust=lambda: (np.zeros(3), np.zeros(3)), **base_kw)
    with pytest.raises(InputError, match="thrust"):
        _solve_free_flight(lat, cond, dofs=("z",), thrust=lambda t: (np.zeros(2), np.zeros(3)), **base_kw)
    with pytest.raises(InputError, match="thrust"):
        _solve_free_flight(
            lat, cond, dofs=("z",), thrust=lambda t: (np.array([float("nan"), 0.0, 0.0]), np.zeros(3)), **base_kw,
        )
    with pytest.raises(InputError, match="thrust"):
        _solve_free_flight(
            lat, cond, dofs=("z",), thrust=lambda t: (np.zeros(3), np.array([0.0, float("inf"), 0.0])), **base_kw,
        )

    # Gimbal lock near |pitch| = 90 deg (both initial and during dynamic pitch-up).
    with pytest.raises(InputError, match="gimbal lock"):
        _solve_free_flight(
            lat, cond, dofs=("pitch",), initial_state={"pitch": math.radians(89.5)}, **base_kw,
        )
    with pytest.raises(InputError, match="gimbal lock"):
        _solve_free_flight(
            lat,
            cond,
            dofs=("pitch",),
            initial_state={"pitch": math.radians(80.0), "pitch_dot": 20.0},
            aerodynamics_off=True,
            **base_kw,
        )
