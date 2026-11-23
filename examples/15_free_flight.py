# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Example 15: Free-body flight with the time-domain vortex-ring solver.

Runs three free-flight cases on a small wing-tail aircraft (mass 80 kg,
I_yy = 24 kg m^2, V_inf = 25 m/s, alpha = 4 deg), with the heave (z) and
pitch degrees of freedom free. In each case a constant thrust force and
moment equal to minus the steady imbalance of the mean state trims the mean
state. The march starts impulsively (no wake at t = 0), so the body moves a
little while the starting wake forms.

1. A smooth elevator doublet (one sine period) after the starting
   transient, in free air.
2. The same doublet with a pitch damper on the elevator,
   ``delta = doublet(t) + K * pitch_dot``, given as a callable of
   ``(t, state)``. A positive deflection moves the trailing edge down, so the
   damper gives a nose-down moment when the nose pitches up.
3. A descent towards flat ground from h / c = 0.6 with an initial sink rate
   of 0.3 m/s (the mean state is trimmed in ground effect at h / c = 0.6).

Prints summary tables and saves a three-panel figure to
``examples/output/15_free_flight.png``.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ventorum as vt

MASS = 80.0  # kg
INERTIA = np.diag([16.0, 24.0, 36.0])  # kg m^2, about the centre of gravity
V_INF = 25.0  # m/s
ALPHA = math.radians(4.0)  # rad
PITCH_DAMPER_GAIN = 0.10  # s (rad of elevator per rad/s of pitch rate)
DOUBLET_START = 0.20  # s
DOUBLET_PERIOD = 0.08  # s
DOUBLET_AMPLITUDE = math.radians(2.0)  # rad
N_STEPS = 100


def _build_aircraft() -> vt.Aircraft:
    """Build an AR 6 wing with a tail and a full-span elevator; the CG is ahead of the neutral point."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    elevator = vt.ControlSurface(name="elevator", eta_start=0.0, eta_end=1.0, hinge_x_c=0.5, deflection=0.0)
    tail = vt.LiftingSurface(
        name="tail",
        semi_span=1.0,
        position=np.array([1.8, 0.0, 0.0]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.5)],
        controls=[elevator],
    )
    ac = vt.Aircraft(name="wing_tail", surfaces=[wing, tail], ref_point=np.array([-0.15, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _steady_trim_thrust(ac, cond, settings):
    """Return a constant thrust callable that trims the steady load of the mean state."""
    probe = vt.free_flight(ac, cond, settings, mass=MASS, inertia=INERTIA, dofs=(), n_steps=1)
    force_b = -probe.steady_force_imbalance_body
    moment_b = -probe.steady_moment_imbalance_body
    print(f"Trim thrust in body axes: force {np.round(force_b, 3)} N, moment {np.round(moment_b, 3)} N m")
    return lambda t: (force_b, moment_b)


def _doublet(t: float) -> float:
    """Return the smooth elevator doublet [rad] at time *t* [s].

    The doublet is one sine period. A smooth input has a finite rate; a
    step in a callable has an infinite rate at the jump.
    """
    if DOUBLET_START <= t < DOUBLET_START + DOUBLET_PERIOD:
        return DOUBLET_AMPLITUDE * math.sin(2.0 * math.pi * (t - DOUBLET_START) / DOUBLET_PERIOD)
    return 0.0


def _print_summary(title: str, res: vt.FreeFlightResult) -> None:
    """Print the key free-flight values."""
    print(f"=== {title} ===")
    print(f"Steps: {res.n_steps}, dt = {res.dt:.5f} s, t_end = {res.time[-1]:.4f} s, free DOFs {res.dofs}")
    print(
        f"pitch: min = {math.degrees(np.min(res.pitch)):+.4f} deg, max = {math.degrees(np.max(res.pitch)):+.4f} deg, "
        f"final = {math.degrees(res.pitch[-1]):+.4f} deg"
    )
    print(f"z: final = {res.z[-1]:+.5f} m, z_dot final = {res.z_dot[-1]:+.5f} m/s")
    print(f"CL: min = {np.min(res.CL):+.5f}, max = {np.max(res.CL):+.5f}")
    print(
        f"Coupling sub-iterations: mean {np.mean(res.subiterations[1:]):.2f}, max {int(np.max(res.subiterations))}; "
        f"ground contact: {res.ground_contact}"
    )
    print()


def main() -> None:
    """Run the three free-flight cases and save plots to examples/output/."""
    ac = _build_aircraft()
    settings = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=8, chord_spacing="uniform")

    cond = vt.FlightCondition(V_inf=V_INF, rho=1.225, alpha=ALPHA)
    thrust = _steady_trim_thrust(ac, cond, settings)
    res_free = vt.free_flight(
        ac, cond, settings, mass=MASS, inertia=INERTIA, dofs=("z", "pitch"), n_steps=N_STEPS,
        thrust=thrust, controls={"elevator": _doublet},
    )
    _print_summary("1. Elevator doublet, free air", res_free)

    res_damped = vt.free_flight(
        ac, cond, settings, mass=MASS, inertia=INERTIA, dofs=("z", "pitch"), n_steps=N_STEPS,
        thrust=thrust,
        controls={"elevator": lambda t, state: _doublet(t) + PITCH_DAMPER_GAIN * state.pitch_dot},
    )
    _print_summary("2. Same doublet with a pitch damper", res_damped)

    cond_ge = vt.FlightCondition(V_inf=V_INF, rho=1.225, alpha=ALPHA, h=0.6)
    thrust_ge = _steady_trim_thrust(ac, cond_ge, settings)
    res_ge = vt.free_flight(
        ac, cond_ge, settings, mass=MASS, inertia=INERTIA, dofs=("z", "pitch"), n_steps=N_STEPS,
        initial_state={"z_dot": -0.3}, thrust=thrust_ge,
    )
    _print_summary("3. Descent towards the ground from h/c = 0.6", res_ge)

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.0), sharex=True)
    axes[0].plot(res_free.time, np.degrees(res_free.pitch), label="no damper")
    axes[0].plot(res_damped.time, np.degrees(res_damped.pitch), label="pitch damper")
    axes[0].set_ylabel("pitch [deg]")
    axes[0].legend()
    axes[1].plot(res_damped.time, np.degrees(res_damped.controls["elevator"]))
    axes[1].set_ylabel("elevator [deg]")
    axes[2].plot(res_ge.time, res_ge.z_dot)
    axes[2].set_ylabel("z_dot near ground [m/s]")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "15_free_flight.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
