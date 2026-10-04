# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Example 3: Swept Wing with Dihedral
====================================
A 30° swept wing with 5° dihedral - geometry that *requires* the
VLM solver (the Fourier solver will refuse to run).
"""

import numpy as np
import matplotlib.pyplot as plt
import ventorum as vt

# --- Define a swept wing with dihedral ----------------------------------------
wing = vt.LiftingSurface(
    name="Swept Wing (Sweep=30 deg, Dihedral=5 deg)",
    semi_span=6.0,
    sections=[
        vt.WingSection(y_frac=0.0, chord=3.0),
        vt.WingSection(y_frac=1.0, chord=1.5),
    ],
    sweep_le=np.radians(30.0),
    dihedral=np.radians(5.0),
)

aircraft = vt.Aircraft(name="Swept Config", surfaces=[wing])

# --- Analyse ------------------------------------------------------------------
print("=" * 60)
print("Ventorum Example 3: Swept Wing with Dihedral")
print("=" * 60)

result = vt.analyze(aircraft, alpha_deg=5.0, solver="vlm", n_panels=60)
t = result.totals
print(f"  CL    = {t.CL:.4f}")
print(f"  CDi   = {t.CDi:.6f}")
print(f"  e     = {t.e:.4f}")
print(f"  AR    = {t.AR:.2f}")

# --- 3D geometry plot ---------------------------------------------------------
settings = vt.SolverSettings(solver_type="vlm", n_panels=40)
fig_geom = vt.plot_geometry(aircraft, settings, show_normals=True)

# --- Distributions ------------------------------------------------------------
fig_dist = vt.plot_all_distributions(result)

# --- Alpha sweep ---------------------------------------------------------------
alpha_range = np.radians(np.arange(-2, 13, 1.0))
cond = vt.FlightCondition(V_inf=60.0)
sweep_results = vt.alpha_sweep(aircraft, cond, settings, alpha_range)
fig_sweep = vt.plot_sweep_summary(sweep_results, alpha_range)

plt.show()
