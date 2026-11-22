# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Example 14: Time-domain unsteady response, gusts and ground-effect wind shear.

Runs three prescribed-wake vortex-ring cases on an AR 6 rectangular wing:
1. Pitch oscillation combined with a moving trailing-edge flap in free air.
2. Penetration of a 1-cosine vertical gust in free air.
3. Logarithmic atmospheric head-wind shear near flat ground (h / c = 0.6).

Prints summary tables and saves a three-panel figure to
``examples/output/14_time_response.png``.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ventorum as vt


def _build_wing() -> vt.Aircraft:
    """Build an AR 6 rectangular wing with a full-span 25 % chord trailing-edge flap."""
    flap = vt.ControlSurface(
        name="flap",
        eta_start=0.0,
        eta_end=1.0,
        hinge_x_c=0.75,
        deflection=0.0,
    )
    wing = vt.LiftingSurface(
        name="rectangular_wing",
        semi_span=3.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
        controls=[flap],
    )
    ac = vt.Aircraft(name="AR6_rect_flap", surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _print_summary(title: str, res: vt.TimeResponseResult) -> None:
    """Print key response extrema, final values and solver notes."""
    print(f"=== {title} ===")
    print(
        f"Steps: {res.n_steps}, dt = {res.dt:.5f} s, t_end = {res.time[-1]:.4f} s, "
        f"s_end = {res.s[-1]:.2f} semichords"
    )
    print(
        f"CL: min = {np.min(res.CL):+.5f}, max = {np.max(res.CL):+.5f}, final = {res.CL[-1]:+.5f}"
    )
    print(
        f"Cm: min = {np.min(res.Cm):+.5f}, max = {np.max(res.Cm):+.5f}, final = {res.Cm[-1]:+.5f}"
    )
    print(
        f"CD: min = {np.min(res.CD):+.5f}, max = {np.max(res.CD):+.5f}, final = {res.CD[-1]:+.5f}"
    )
    if res.trust is not None:
        print(f"Steady trust: {res.trust.rating} (score = {res.trust.score:.3f})")
    print("Notes:")
    for note in res.notes:
        print(f"  - {note}")
    print()


def main() -> None:
    """Run the three time-domain response cases and save plots to examples/output/."""
    ac = _build_wing()
    settings = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=8, chord_spacing="uniform")

    # 1. Pitch oscillation with a moving trailing-edge flap in free air.
    cond_free = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=math.radians(3.0))
    omega = 2.0 * 0.25 * cond_free.V_inf / ac.c_ref  # reduced frequency k = 0.25
    res_pitch_flap = vt.time_response(
        ac,
        cond_free,
        settings,
        n_steps=48,
        motion={"pitch": lambda t: math.radians(2.0) * math.sin(omega * t)},
        controls={"flap": lambda t: math.radians(5.0) * math.sin(omega * t + 0.25 * math.pi)},
    )
    _print_summary("Case 1: Pitch oscillation + moving flap (free air, k = 0.25)", res_pitch_flap)

    # 2. 1-cosine vertical gust in free air (peak 2 m/s over 6 m length).
    gust_cos = vt.one_minus_cosine(amplitude=2.0, length=6.0, direction="vertical", x_front=0.0)
    res_gust = vt.time_response(
        ac,
        cond_free,
        settings,
        n_steps=64,
        gust=gust_cos,
    )
    _print_summary("Case 2: 1-cosine vertical gust (free air, w0 = 2 m/s, L = 6 m)", res_gust)

    # 3. Logarithmic head-wind shear near flat ground (h / c = 0.6).
    cond_ge = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=math.radians(3.0), h=0.6)
    shear_prof = vt.logarithmic_shear(u_ref=8.0, z_ref=10.0, z0=0.05)
    gust_shear = vt.wind_shear(shear_prof, direction="head")
    res_shear = vt.time_response(
        ac,
        cond_ge,
        settings,
        n_steps=48,
        gust=gust_shear,
    )
    _print_summary("Case 3: Logarithmic head-wind shear in ground effect (h/c = 0.6)", res_shear)

    # Save three-panel summary figure.
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "14_time_response.png"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel 1: pitch + flap
    ax = axes[0]
    ax.plot(res_pitch_flap.s, res_pitch_flap.CL, label="CL")
    ax2 = ax.twinx()
    ax2.plot(res_pitch_flap.s, res_pitch_flap.Cm, color="tab:orange", linestyle="--", label="Cm")
    ax.set_title("Pitch + moving flap (free air)")
    ax.set_xlabel("Semichords s = 2 V t / c [-]")
    ax.set_ylabel("CL [-]")
    ax2.set_ylabel("Cm [-]", color="tab:orange")
    ax.grid(True, alpha=0.3)

    # Panel 2: 1-cos gust
    ax = axes[1]
    ax.plot(res_gust.s, res_gust.CL, label="CL")
    ax2 = ax.twinx()
    ax2.plot(res_gust.s, res_gust.gust_velocity[:, 2], color="tab:green", linestyle=":", label="w_g(ref)")
    ax.set_title("1-cos vertical gust (w0 = 2 m/s, L = 6 m)")
    ax.set_xlabel("Semichords s = 2 V t / c [-]")
    ax.set_ylabel("CL [-]")
    ax2.set_ylabel("w_g at ref_point [m/s]", color="tab:green")
    ax.grid(True, alpha=0.3)

    # Panel 3: ground-effect head-wind shear
    ax = axes[2]
    ax.plot(res_shear.s, res_shear.CL, label="CL (shear, h/c=0.6)")
    ax.axhline(res_shear.base.totals.CL, color="tab:gray", linestyle="--", label="Steady CL (no shear)")
    ax.set_title("Head-wind log shear (h/c = 0.6)")
    ax.set_xlabel("Semichords s = 2 V t / c [-]")
    ax.set_ylabel("CL [-]")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
