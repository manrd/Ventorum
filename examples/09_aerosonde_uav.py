# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Example 9: Aerosonde UAV Aerodynamic Model & Comprehensive Visualization Suite
=============================================================================
This example models the Aerosonde UAV - a classic benchmark platform in fixed-wing
unmanned aerial vehicle flight mechanics (Beard & McLain, 2012, *Small Unmanned
Aircraft: Theory and Practice*; BYU MAVSim benchmark).

Key Platform Specifications:
-----------------------------
- Wingspan (b):            2.8956 m (~9.5 ft)
- Reference Wing Area (S): 0.550 m² (~5.92 ft²)
- Mean Chord (c_bar):      0.194 m (~7.64 in)
- Aspect Ratio (AR):       ~15.24 (High-efficiency, long-endurance glider/recon)
- Main Wing Taper Ratio:   0.583 (Root chord = 0.24 m, Tip chord = 0.14 m)
- Wing Dihedral:           2.0° (lateral stability)
- Wing LE Sweep:           1.5° (straight quarter-chord)
- Geometric Washout:       -2.0° aerodynamic twist at the tip (stall mitigation)
- Main Wing Airfoil:       NACA 4412 (cambered high-endurance profile)
- Inverted V-Tail:         b_semi = 0.42 m (projected b_h ≈ 0.66 m), anhedral = -38.0°
                           Mounted at x = 1.05 m, apex z = +0.12 m with -1.5° incidence
- Tail Airfoil:            NACA 0012 (symmetric ruddervators)

This script performs:
1. Complete geometry and aerodynamic model setup of the Aerosonde UAV.
2. JSON configuration export for persistence and interoperability.
3. High-fidelity nonlinear lifting-line solution at cruise conditions.
4. Angle-of-attack sweep (-4° to +16°) capturing stall and drag rise.
5. Ground-effect height sweep (h/c from 0.1 to 2.0).
6. Execution and export of ALL 11 visualization plots available in Ventorum.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import ventorum as vt
from ventorum.core.config import save_aircraft_to_json, load_aircraft_from_json
from ventorum.ground_effect.sweep import sweep_height
from ventorum.ground_effect.plotting import plot_height_sweep


def build_aerosonde_uav() -> vt.Aircraft:
    """Build the Aerosonde UAV aerodynamic geometry."""
    # -------------------------------------------------------------------------
    # 1. Section Aerodynamics (NACA 4412 Main Wing + NACA 0012 Tail)
    # -------------------------------------------------------------------------
    # Synthetic realistic polar for NACA 4412 with linear region and stall onset
    alpha_deg = np.arange(-8.0, 22.0, 0.5)
    alpha_rad = np.radians(alpha_deg)

    # Linear lift slope a0 ≈ 6.28 /rad with zero-lift alpha_L0 ≈ -4.0 deg
    cl_lin = 2.0 * np.pi * (alpha_rad - np.radians(-4.0))
    stall_deg = 14.0
    cl_nonlinear = np.where(
        alpha_deg <= stall_deg,
        cl_lin,
        1.55 - 0.70 * (1.0 - np.exp(-(alpha_deg - stall_deg) / 3.0)),
    )
    cl_data = np.clip(cl_nonlinear, -0.60, 1.58)

    # Drag polar: profile drag bucket + lift-induced profile drag + stall rise
    cd_data = (
        0.0080
        + 0.0040 * ((alpha_deg + 2.0) / 10.0) ** 2
        + 0.0350 * np.maximum(0.0, alpha_deg - 12.0) ** 2 / 50.0
    )

    af_naca4412 = vt.TabulatedAirfoil(
        name="NACA 4412",
        alpha=alpha_rad,
        Cl_data=cl_data,
        Cd_data=cd_data,
        Re=300000,
    )

    af_naca0012 = vt.LinearAirfoil(
        name="NACA 0012",
        a0=2.0 * np.pi,
        alpha_L0=0.0,
        Cd0=0.0065,
    )

    # -------------------------------------------------------------------------
    # 2. Main Wing Surface Definition
    # -------------------------------------------------------------------------
    # Semi-span = 1.4478 m -> Full span = 2.8956 m
    # Root chord = 0.24 m, Tip chord = 0.14 m -> Area = 0.5502 m²
    main_wing = vt.LiftingSurface(
        name="Main Wing",
        semi_span=1.4478,
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.24, twist=0.0, airfoil=af_naca4412),
            vt.WingSection(
                y_frac=1.0,
                chord=0.14,
                twist=np.radians(-2.0),  # -2.0° washout for stall progression
                airfoil=af_naca4412,
            ),
        ],
        sweep_le=np.radians(1.5),   # 1.5° leading-edge sweep
        dihedral=np.radians(2.0),   # 2.0° dihedral for lateral stability
        is_symmetric=True,
        position=np.array([0.0, 0.0, 0.0]),
        n_panels=40,
        spacing="half-cosine",      # Geometry-adaptive: tip clustered, coarse at planar root (dGamma/dy = 0)
    )

    # -------------------------------------------------------------------------
    # 3. Inverted V-Tail (Empennage)
    # -------------------------------------------------------------------------
    # The signature Aerosonde inverted V-tail (ruddervators):
    # Twin booms support an anhedral inverted V-tail (dihedral = -38.0°).
    # Semi-span = 0.42 m -> Projected horizontal span ≈ 0.66 m
    # Root chord = 0.14 m, Tip chord = 0.10 m
    # Position: Apex mounted at x = 1.05 m aft, z = +0.12 m above wing plane
    # The negative dihedral causes the stabilizer panels to slope downward
    # towards the twin booms, matching the iconic Aerosonde inverted "V" shape.
    inverted_v_tail = vt.LiftingSurface(
        name="Inverted V-Tail",
        semi_span=0.42,
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.14, airfoil=af_naca0012),
            vt.WingSection(y_frac=1.0, chord=0.10, airfoil=af_naca0012),
        ],
        sweep_le=np.radians(8.0),             # Leading edge sweep on V-tail
        dihedral=np.radians(-38.0),           # Negative dihedral (anhedral) = Inverted V
        is_symmetric=True,
        position=np.array([1.05, 0.0, 0.12]), # Apex mounted at top
        incidence=np.radians(-1.5),           # Trim incidence
        n_panels=16,
        spacing="cosine",                     # Refined at non-planar apex junction kink and tip
    )

    # -------------------------------------------------------------------------
    # 4. Assemble Aircraft Model
    # -------------------------------------------------------------------------
    aircraft = vt.Aircraft(
        name="Aerosonde UAV",
        surfaces=[main_wing, inverted_v_tail],
    )
    aircraft.compute_reference_values()
    return aircraft


def main():
    parser = argparse.ArgumentParser(
        description="Aerosonde UAV Aerodynamic Model & Visualization Suite"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display all generated Matplotlib figures interactively",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "output" / "aerosonde"),
        help="Directory where visualization figures and artifacts will be saved",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("  Ventorum: Aerosonde UAV Aerodynamic Model & Visualization Suite")
    print("=" * 75)

    # 1. Build and serialize aircraft model
    aircraft = build_aerosonde_uav()
    json_path = out_dir / "aerosonde_uav.json"
    save_aircraft_to_json(aircraft, json_path)
    print(f"\n[+] Aircraft Model: {aircraft.name}")
    print(f"    - Reference Area (S_ref):     {aircraft.S_ref:.4f} m^2  (target: 0.55 m^2)")
    print(f"    - Wingspan (b_ref):           {aircraft.b_ref:.4f} m   (target: 2.90 m)")
    print(f"    - Mean Aero Chord (c_ref):    {aircraft.c_ref:.4f} m   (target: 0.19 m)")
    print(f"    - Wing Aspect Ratio (AR):     {aircraft.b_ref**2 / aircraft.S_ref:.2f}")
    print(f"    - Number of Lifting Surfaces: {len(aircraft.surfaces)}")
    print(f"    - Serialized configuration:   {json_path}")

    # Verify reload roundtrip
    _ = load_aircraft_from_json(json_path)
    print("    - JSON reload test:           PASSED")

    # -------------------------------------------------------------------------
    # 2. Cruise Point Analysis
    # -------------------------------------------------------------------------
    cruise_alpha_deg = 3.0
    cruise_V_inf = 25.0  # m/s (~48.6 knots, typical Aerosonde cruise)

    settings = vt.SolverSettings(
        solver_type="vlm",
    )
    flight_cond = vt.FlightCondition(
        V_inf=cruise_V_inf,
        alpha=np.radians(cruise_alpha_deg),
    )

    print(f"\n[+] Running Cruise Analysis (alpha = {cruise_alpha_deg:.1f} deg, V_inf = {cruise_V_inf:.1f} m/s)...")
    print(f"    - Main Wing Mesh: {aircraft.surfaces[0].n_panels} panels/semi ({aircraft.surfaces[0].spacing})")
    print(f"    - V-Tail Mesh:    {aircraft.surfaces[1].n_panels} panels/semi ({aircraft.surfaces[1].spacing})")
    result = vt.analyze(aircraft, condition=flight_cond, settings=settings)

    t = result.totals
    print(f"    - Total CL:                   {t.CL:.4f}")
    print(f"    - Total CDi (induced):        {t.CDi:.6f}")
    if t.CD_total is not None:
        print(f"    - Total CD (profile+induced): {t.CD_total:.6f}")
        print(f"    - Total L/D:                  {t.CL / t.CD_total:.2f}")
    print(f"    - Span Efficiency (e):        {t.e:.4f}")

    # -------------------------------------------------------------------------
    # 3. Angle-of-Attack Sweep
    # -------------------------------------------------------------------------
    sweep_alphas_deg = np.linspace(-4.0, 14.0, 19)
    print(f"\n[+] Executing Alpha Sweep ({sweep_alphas_deg[0]:.1f} deg to {sweep_alphas_deg[-1]:.1f} deg, {len(sweep_alphas_deg)} points)...")
    sweep_settings = vt.SolverSettings(
        solver_type="vlm",
    )
    sweep_results = vt.analyze_sweep(
        aircraft,
        alpha_deg_range=sweep_alphas_deg,
        V_inf=cruise_V_inf,
        solver="vlm",
        backend="thread",
        settings=sweep_settings,
    )
    sweep_alphas_rad = np.radians(sweep_alphas_deg)
    print("    - Alpha sweep completed successfully.")

    # -------------------------------------------------------------------------
    # 4. Ground Effect Height Sweep
    # -------------------------------------------------------------------------
    heights = np.linspace(0.10, 2.0, 12)
    print(f"\n[+] Executing Ground-Effect Height Sweep (h = {heights[0]:.2f}m to {heights[-1]:.2f}m)...")
    ge_sweep = sweep_height(
        aircraft,
        heights=heights,
        alpha_deg=cruise_alpha_deg,
        V_inf=cruise_V_inf,
        n_panels=25,
    )
    print("    - Ground effect sweep completed successfully.")

    # -------------------------------------------------------------------------
    # 5. Execute and Save ALL Available Visualizations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  Generating ALL Available Ventorum Visualizations (21 Visualizations)")
    print("=" * 75)

    figures: dict[str, plt.Figure] = {}

    # 1. 3D Geometry Wireframe, Control Points, and Normals
    print("  [1/21] 3D Geometry Wireframe with Panels and Normals...")
    fig1 = vt.plot_geometry(
        aircraft,
        settings=settings,
        show_normals=True,
        show_control_points=True,
        elev=20.0,
        azim=-125.0,
    )
    fig1.axes[0].set_title(
        f"Aerosonde UAV  -  3D Geometry Wireframe with Inverted V-Tail\n(b = {aircraft.b_ref:.2f}m, S = {aircraft.S_ref:.2f}m²)",
        fontsize=12,
        fontweight="bold",
    )
    figures["01_geometry_3d.png"] = fig1

    # 2. 3D Surface Colored Panels  -  Section Lift Coefficient (Cl)
    print("  [2/21] 3D Surface Panels  -  Lift Coefficient Cl...")
    figures["02a_surface_cl_panels.png"] = vt.plot_surface_results(
        aircraft, result, metric="Cl", settings=settings, cmap="viridis", elev=22.0, azim=-125.0
    )

    # 3. 3D Surface Colored Panels  -  Circulation (Gamma)
    print("  [3/21] 3D Surface Panels  -  Circulation Strength Gamma...")
    figures["02b_surface_gamma_panels.png"] = vt.plot_surface_results(
        aircraft, result, metric="gamma", settings=settings, cmap="plasma", elev=22.0, azim=-125.0
    )

    # 4. 3D Surface Colored Panels  -  Effective Angle of Attack (alpha_eff)
    print("  [4/21] 3D Surface Panels  -  Effective Angle alpha_eff...")
    figures["02c_surface_alpha_eff_panels.png"] = vt.plot_surface_results(
        aircraft, result, metric="alpha_eff", settings=settings, cmap="inferno", elev=22.0, azim=-125.0
    )

    # 5. 3D Surface Colored Panels  -  Induced Downwash Angle (alpha_i)
    print("  [5/21] 3D Surface Panels  -  Induced Downwash Angle alpha_i...")
    figures["02d_surface_downwash_panels.png"] = vt.plot_surface_results(
        aircraft, result, metric="alpha_i", settings=settings, cmap="coolwarm", elev=22.0, azim=-125.0
    )

    # 6. 3D Surface Colored Panels  -  Sectional Induced Drag (Cd_i)
    print("  [6/21] 3D Surface Panels  -  Sectional Induced Drag Cd_i...")
    figures["02e_surface_cdi_panels.png"] = vt.plot_surface_results(
        aircraft, result, metric="Cd_i", settings=settings, cmap="magma", elev=22.0, azim=-125.0
    )

    # 7. 3D Vortex Wake System (Bound Vortices + Trailing Filaments)
    print("  [7/21] 3D Vortex Wake System (Quarter-chord Bound + Trailing Filaments)...")
    figures["03_vortex_wake_3d.png"] = vt.plot_vortex_wake(
        aircraft, result, elev=22.0, azim=-125.0
    )

    # 8. 2D Downstream Trefftz-Plane Cross-Flow & Streamlines
    print("  [8/21] Trefftz-Plane Downwash Contours & Cross-Flow Streamlines...")
    figures["04_trefftz_plane.png"] = vt.plot_trefftz_plane(
        aircraft, result, grid_res=(35, 25), streamlines=True
    )

    # 9. 2D Planform Blueprint Layout with MAC & Dimensions
    print("  [9/21] 2D Top-Down Planform Blueprint Layout with Mean Aerodynamic Chord...")
    figures["05_planform_blueprint_2d.png"] = vt.plot_planform_2d(
        aircraft, show_mac=True, show_quarter_chord=True, show_dimensions=True
    )

    # 10. Dimensional Span Loading L'(y) vs Elliptic & Root Bending Moment
    print("  [10/21] Dimensional Span Loading L'(y) [N/m] & Root Bending Moment...")
    figures["06_span_loading_and_bending.png"] = vt.plot_span_loading(
        aircraft, result, show_elliptic=True
    )

    # 11. Spanwise Circulation Distribution Gamma(y)/Gamma_max
    print("  [11/21] Spanwise Circulation Distribution Gamma(y)/Gamma_max...")
    figures["07_lift_distribution.png"] = vt.plot_lift_distribution(result, show_elliptic=True)

    # 12. Comprehensive 2x2 Spanwise Distributions Panel
    print("  [12/21] Comprehensive 2x2 Spanwise Distributions Panel...")
    figures["08_all_distributions_panel.png"] = vt.plot_all_distributions(result)

    # 13. Multi-Surface Aerodynamic Force Breakdown (Wing vs V-Tail)
    print("  [13/21] Multi-Surface Force and Drag Share Breakdown...")
    figures["09_component_breakdown.png"] = vt.plot_component_breakdown(result)

    # 14. Lifting-line Fourier circulation harmonic spectrum (A_n, delta, e)
    print("  [14/21] Lifting-line Fourier circulation harmonic spectrum...")
    figures["10_fourier_spectrum.png"] = vt.plot_fourier_spectrum(result, n_harmonics=15)

    # 15. 2D Airfoil Characteristics (NACA 4412 Main Wing Profile)
    print("  [15/21] 2D Airfoil Characteristics (NACA 4412)...")
    figures["11_airfoil_polar_naca4412.png"] = vt.plot_airfoil_polar(
        aircraft.surfaces[0].sections[0].airfoil,
        alpha_deg_range=np.linspace(-6.0, 18.0, 60),
    )

    # 16. Total Aircraft Lift Curve (CL vs alpha)
    print("  [16/21] Aircraft Lift Curve (CL vs alpha)...")
    figures["12_lift_curve_cl_alpha.png"] = vt.plot_cl_vs_alpha(sweep_results, alpha_range=sweep_alphas_rad)

    # 17. Aircraft Drag Polar (CL vs CD)
    print("  [17/21] Aircraft Drag Polar (CL vs CD)...")
    figures["13_drag_polar.png"] = vt.plot_drag_polar(sweep_results)

    # 18. Alpha Sweep 1x2 Summary (Lift Curve + Drag Polar)
    print("  [18/21] Alpha Sweep 1x2 Summary...")
    figures["14_alpha_sweep_summary.png"] = vt.plot_sweep_summary(sweep_results, alpha_range=sweep_alphas_rad)

    # 19. Aerodynamic Efficiency (L/D) & UAV Endurance Factor (CL^1.5 / CD)
    print("  [19/21] Aerodynamic Efficiency (L/D) & UAV Endurance Parameter...")
    figures["15_efficiency_and_endurance.png"] = vt.plot_efficiency_curves(
        sweep_results, alpha_range=sweep_alphas_rad
    )

    # 20. Longitudinal Pitching Moment Stability (Cm vs alpha & Cm vs CL)
    print("  [20/21] Longitudinal Pitching Moment Stability & Static Margin...")
    figures["16_pitching_moment_stability.png"] = vt.plot_pitching_moment(
        sweep_results, alpha_range=sweep_alphas_rad
    )

    # 21. Ground Effect Height Sweep (CL, CDi, L/D, e vs h/c)
    print("  [21/21] Ground-Effect Aerodynamic Augmentation vs Height (h/c)...")
    figures["17_ground_effect_sweep.png"] = plot_height_sweep(ge_sweep, alpha_deg=cruise_alpha_deg)

    # Executive Masterpiece: 8-Panel Publication-Grade Aircraft Dashboard
    print("  [+] Generating Executive 8-Panel Aerodynamic Performance Dashboard...")
    fig_dashboard = vt.plot_aircraft_dashboard(
        aircraft=aircraft,
        result=result,
        sweep_results=sweep_results,
        alpha_sweep_deg=sweep_alphas_deg,
        settings=settings,
        figsize=(22, 11),
    )
    figures["18_executive_aircraft_dashboard.png"] = fig_dashboard

    # Solver Convergence
    fig_conv = vt.plot_convergence(result)
    figures["19_solver_convergence.png"] = fig_conv

    # Save all figures to disk
    print(f"\n[+] Saving {len(figures)} visualization artifacts to disk...")
    for filename, fig in figures.items():
        save_path = out_dir / filename
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"    - Saved: {save_path}")

    # Also save the primary multi-panel summary figure directly to examples/
    main_summary_path = out_dir / "aerosonde_uav_summary.png"
    figures["14_alpha_sweep_summary.png"].savefig(main_summary_path, dpi=200, bbox_inches="tight")
    print(f"    - Saved primary summary polar to: {main_summary_path}")

    # Also save executive dashboard directly to examples/
    dashboard_summary_path = out_dir / "aerosonde_uav_dashboard.png"
    fig_dashboard.savefig(dashboard_summary_path, dpi=200, bbox_inches="tight")
    print(f"    - Saved executive dashboard to: {dashboard_summary_path}")

    print("\n" + "=" * 75)
    print(f"  ALL {len(figures)} VISUALIZATIONS SUCCESSFULLY GENERATED & SAVED")
    print(f"  Artifacts directory: {out_dir.resolve()}")
    print("=" * 75)

    if args.show:
        print("\nDisplaying interactive Matplotlib windows. Close all windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()
