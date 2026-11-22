# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Example 13: Frequency-dependent dynamic derivatives in free air and ground effect.

Computes longitudinal dynamic derivatives for a rectangular wing of aspect
ratio 6 in free air and at ground height h / c = 0.5 for reduced frequencies
k = 0, 0.05 and 0.2, and prints the resulting derivative tables.
"""

from __future__ import annotations

import numpy as np

import ventorum as vt


def _print_case(title: str, res: vt.DynamicDerivativesResult) -> None:
    """Print mean state, mesh metadata and frequency-dependent derivatives."""
    print(f"=== {title} ===")
    print(
        f"Mean state: CL0 = {res.CL0:+.5f}, Cm0 = {res.Cm0:+.5f} "
        f"(axes={res.axes}, mesh={res.n_strips} strips x {res.n_chord} chordwise = {res.n_panels} panels, "
        f"wake={res.n_wake_rows} rows)"
    )
    if res.trust is not None:
        print(f"Steady trust: {res.trust.rating} (score={res.trust.score:.3f})")
    header = (
        f"{'k [-]':>7s}  {'CL_a':>8s}  {'Cm_a':>8s}  {'CL_q':>8s}  {'Cm_q':>8s}  "
        f"{'CL_adot':>9s}  {'Cm_adot':>9s}  {'CL_h':>8s}  {'Cm_h':>8s}  {'CL_hdot':>9s}  {'CL_hdot_g':>9s}"
    )
    print(header)
    print("-" * len(header))
    d = res.derivatives
    for i, k_val in enumerate(res.k):
        print(
            f"{k_val:7.3f}  {d['CL_alpha'][i]:8.4f}  {d['Cm_alpha'][i]:8.4f}  "
            f"{d['CL_q'][i]:8.4f}  {d['Cm_q'][i]:8.4f}  "
            f"{d['CL_alpha_dot'][i]:9.4f}  {d['Cm_alpha_dot'][i]:9.4f}  "
            f"{d['CL_h'][i]:8.4f}  {d['Cm_h'][i]:8.4f}  "
            f"{d['CL_h_dot'][i]:9.4f}  {d['CL_h_dot_ground'][i]:9.4f}"
        )
    if res.trust_notes:
        print("Trust notes:")
        for note in res.trust_notes:
            print(f"  - {note}")
    print()


def main() -> None:
    """Run free-air and ground-effect dynamic derivatives for a rectangular AR 6 wing."""
    wing = vt.LiftingSurface(
        name="rectangular_wing",
        semi_span=3.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    aircraft = vt.Aircraft(
        name="AR6_rect",
        surfaces=[wing],
        ref_point=np.array([0.25, 0.0, 0.0]),
    )
    settings = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=8)
    k_values = (0.0, 0.05, 0.2)

    cond_free = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0))
    res_free = vt.dynamic_derivatives(aircraft, cond_free, settings, k=k_values)
    _print_case("Free air (AR 6 rectangular wing, alpha = 4 deg)", res_free)

    cond_ge = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0), h=0.5)
    res_ge = vt.dynamic_derivatives(aircraft, cond_ge, settings, k=k_values)
    _print_case("Ground effect (AR 6 rectangular wing, alpha = 4 deg, h/c = 0.5)", res_ge)


if __name__ == "__main__":
    main()
