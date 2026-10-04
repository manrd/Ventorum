# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Example 2: Tapered Wing
=======================
A simple tapered wing (taper ratio 0.5) with geometric twist (washout).
Demonstrates the effect of taper and twist on lift distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
import ventorum as vt

# --- Define a tapered wing with washout --------------------------------------
wing = vt.LiftingSurface(
    name="Tapered Wing (λ=0.5)",
    semi_span=6.0,
    sections=[
        vt.WingSection(y_frac=0.0, chord=2.0, twist=np.radians(2.0)),   # root
        vt.WingSection(y_frac=1.0, chord=1.0, twist=np.radians(-2.0)),  # tip (washout)
    ],
)

# --- Analyse ------------------------------------------------------------------
print("=" * 60)
print("Ventorum Example 2: Tapered Wing with Washout")
print("=" * 60)

for alpha_deg in [0.0, 3.0, 6.0, 9.0]:
    result = vt.analyze(wing, alpha_deg=alpha_deg, n_panels=60)
    t = result.totals
    print(f"  alpha = {alpha_deg:5.1f} deg  ->  CL = {t.CL:+.4f}  CDi = {t.CDi:.6f}  e = {t.e:.4f}")

# --- Distributions at α = 6° -------------------------------------------------
result = vt.analyze(wing, alpha_deg=6.0, n_panels=80)
fig = vt.plot_all_distributions(result)

# --- Alpha sweep & drag polar -------------------------------------------------
alpha_range = np.radians(np.arange(-2, 13, 1.0))
cond = vt.FlightCondition(V_inf=50.0)
settings = vt.SolverSettings(n_panels=60)
sweep_results = vt.alpha_sweep(
    vt.Aircraft(surfaces=[wing]),
    cond, settings,
    alpha_range=alpha_range,
)
fig2 = vt.plot_sweep_summary(sweep_results, alpha_range)

plt.show()
