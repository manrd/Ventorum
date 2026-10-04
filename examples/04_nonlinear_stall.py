# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Example 4: Nonlinear Polars - Stall Prediction
===============================================
Uses synthetic NACA 2412 polar data to demonstrate the iterative nonlinear
solver and stall behaviour.
"""

import numpy as np
import matplotlib.pyplot as plt
import ventorum as vt

# --- Create a synthetic NACA 2412 polar (approximate) -------------------------
alpha_deg = np.arange(-10, 22, 0.5)
alpha_rad = np.radians(alpha_deg)

# Approximate Cl curve: linear up to ~12°, then stall
Cl_linear = 0.1 + 2 * np.pi * (alpha_rad - np.radians(-2.0))  # a0≈2π, αL0≈-2°
Cl_stall = 1.5 * np.exp(-((alpha_deg - 12) / 4) ** 2)  # Gaussian decay
Cl = np.where(alpha_deg < 12, Cl_linear, 1.5 - 0.8 * (1 - np.exp(-(alpha_deg - 12) / 5)))
Cl = np.clip(Cl, -1.0, 1.55)

# Simple Cd curve: drag bucket + increase near stall
Cd = 0.008 + 0.005 * (alpha_deg / 10) ** 2 + 0.02 * np.maximum(0, alpha_deg - 10) ** 2 / 100

airfoil_2412 = vt.TabulatedAirfoil(
    name="NACA2412 (synthetic)",
    alpha=alpha_rad,
    Cl_data=Cl,
    Cd_data=Cd,
)

# --- Define wing with this airfoil --------------------------------------------
wing = vt.LiftingSurface(
    name="Wing with NACA2412",
    semi_span=5.0,
    sections=[
        vt.WingSection(y_frac=0.0, chord=2.0, airfoil=airfoil_2412),
        vt.WingSection(y_frac=1.0, chord=1.0, airfoil=airfoil_2412),
    ],
)

# --- Run nonlinear analysis ---------------------------------------------------
print("=" * 60)
print("Ventorum Example 4: Nonlinear Polars / Stall Prediction")
print("=" * 60)

# "nonlinear": lifting line with the section polar (Newton iteration). The
# default "auto" would also choose it, because the sections have tabulated polars.
settings = vt.SolverSettings(
    solver_type="nonlinear",
    n_panels=40,
    max_iterations=100,
)

for alpha_deg_val in [2.0, 5.0, 8.0, 10.0, 13.0, 16.0]:
    cond = vt.FlightCondition(V_inf=50.0, alpha=np.radians(alpha_deg_val))
    aircraft = vt.Aircraft(surfaces=[wing])
    result = vt.analyze(aircraft, condition=cond, settings=settings)
    t = result.totals
    conv_str = "[CONV]" if result.converged else "[NO-CONV]"
    print(
        f"  alpha = {alpha_deg_val:5.1f} deg  ->  CL = {t.CL:+.4f}  "
        f"CDi = {t.CDi:.6f}  e = {t.e:.4f}  "
        f"iters = {result.iterations:3d}  {conv_str}"
    )

# --- Plot at α = 10° ----------------------------------------------------------
cond = vt.FlightCondition(V_inf=50.0, alpha=np.radians(10.0))
result = vt.analyze(vt.Aircraft(surfaces=[wing]), condition=cond, settings=settings)
fig1 = vt.plot_all_distributions(result)
fig2 = vt.plot_convergence(result)

# --- Alpha sweep ---------------------------------------------------------------
alpha_range = np.radians(np.arange(-2, 18, 1.0))
sweep_results = vt.alpha_sweep(
    vt.Aircraft(surfaces=[wing]),
    vt.FlightCondition(V_inf=50.0),
    settings,
    alpha_range,
)
fig3 = vt.plot_sweep_summary(sweep_results, alpha_range)

plt.show()
