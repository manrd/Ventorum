# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Example 11: Aircraft trim solver.

Demonstrates longitudinal and lateral trim for an aircraft configuration:
1. Wing with ailerons, horizontal tail with an elevator, and vertical fin with a rudder.
2. Longitudinal trim to target CL = 0.5 with zero pitching moment (Cm = 0).
3. Lateral trim with sideslip (beta = 3 deg) to target CL = 0.5 with zero rolling,
   pitching, and yawing moments (Cl = 0, Cm = 0, Cn = 0).
"""

from __future__ import annotations

import numpy as np

import ventorum as vt


def main() -> None:
    # --- 1. Geometry definition -----------------------------------------------
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=6.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.8),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
        controls=[
            vt.ControlSurface(
                name="aileron",
                eta_start=0.6,
                eta_end=1.0,
                hinge_x_c=0.75,
                deflection=0.0,
                symmetric=False,
            )
        ],
        n_panels=20,
    )

    h_tail = vt.LiftingSurface(
        name="h_tail",
        semi_span=2.0,
        position=np.array([4.6, 0.0, 0.3]),
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.9),
            vt.WingSection(y_frac=1.0, chord=0.5),
        ],
        controls=[
            vt.ControlSurface(
                name="elevator",
                eta_start=0.0,
                eta_end=1.0,
                hinge_x_c=0.70,
                deflection=0.0,
                symmetric=True,
            )
        ],
        n_panels=10,
    )

    v_fin = vt.LiftingSurface(
        name="v_fin",
        semi_span=1.8,
        is_symmetric=False,
        dihedral=np.pi / 2.0,
        position=np.array([4.6, 0.0, 0.0]),
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.9),
            vt.WingSection(y_frac=1.0, chord=0.5),
        ],
        controls=[
            vt.ControlSurface(
                name="rudder",
                eta_start=0.0,
                eta_end=1.0,
                hinge_x_c=0.70,
                deflection=0.0,
                symmetric=True,
            )
        ],
        n_panels=8,
    )

    aircraft = vt.Aircraft(
        name="TrimDemoAircraft",
        surfaces=[wing, h_tail, v_fin],
        ref_point=np.array([0.45, 0.0, 0.0]),
    )

    settings = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=4)

    # --- 2. Longitudinal trim -------------------------------------------------
    print("=" * 60)
    print("Ventorum Example 11: Aircraft Trim Solver")
    print("=" * 60)

    cond_long = vt.FlightCondition(V_inf=35.0, alpha=np.radians(2.0), beta=0.0)
    print("\n--- 1. Longitudinal Trim (CL_target = 0.50, beta = 0 deg) ---")

    res_long = vt.trim(
        aircraft,
        cond_long,
        settings,
        CL_target=0.50,
        pitch_control="elevator",
    )

    print(f"Status:       {res_long.status} (converged={res_long.converged})")
    print(f"Iterations:   {res_long.iterations}")
    print(f"Alpha:        {res_long.alpha_deg:.3f} deg")
    print(f"Elevator:     {res_long.deflections_deg['elevator']:.3f} deg")
    print(f"CL:           {res_long.CL:.6f}")
    print(f"CDi:          {res_long.CD:.6f} ({res_long.drag_basis})")
    print(f"Cm:           {res_long.Cm:.2e}")

    # --- 3. Lateral trim at sideslip -----------------------------------------
    beta_deg = 3.0
    cond_lat = vt.FlightCondition(V_inf=35.0, alpha=np.radians(2.0), beta=np.radians(beta_deg))
    print(f"\n--- 2. Lateral Trim (CL_target = 0.50, beta = {beta_deg:.1f} deg) ---")

    res_lat = vt.trim(
        aircraft,
        cond_lat,
        settings,
        CL_target=0.50,
        pitch_control="elevator",
        roll_control="aileron",
        yaw_control="rudder",
    )

    print(f"Status:       {res_lat.status} (converged={res_lat.converged})")
    print(f"Iterations:   {res_lat.iterations}")
    print(f"Alpha:        {res_lat.alpha_deg:.3f} deg")
    print(f"Elevator:     {res_lat.deflections_deg['elevator']:.3f} deg")
    print(f"Aileron:      {res_lat.deflections_deg['aileron']:.3f} deg")
    print(f"Rudder:       {res_lat.deflections_deg['rudder']:.3f} deg")
    print(f"CL:           {res_lat.CL:.6f}")
    print(f"CDi:          {res_lat.CD:.6f} ({res_lat.drag_basis})")
    print(f"Cl:           {res_lat.Cl:.2e}")
    print(f"Cm:           {res_lat.Cm:.2e}")
    print(f"Cn:           {res_lat.Cn:.2e}")


if __name__ == "__main__":
    main()
