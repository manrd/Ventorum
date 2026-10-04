# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Example 5: Wing + Horizontal Tail - Multi-Surface Interaction
=============================================================
Demonstrates a conventional wing-tail configuration where the tail
experiences downwash from the main wing's trailing vortex system.
"""

import numpy as np
import matplotlib.pyplot as plt
import ventorum as vt

# --- Main wing ----------------------------------------------------------------
main_wing = vt.LiftingSurface(
    name="Main Wing",
    semi_span=6.0,
    sections=[
        vt.WingSection(y_frac=0.0, chord=2.5),
        vt.WingSection(y_frac=1.0, chord=1.5),
    ],
    position=np.array([0.0, 0.0, 0.0]),
)

# --- Horizontal tail (behind and above the wing) -----------------------------
h_tail = vt.LiftingSurface(
    name="H-Tail",
    semi_span=2.5,
    sections=[
        vt.WingSection(y_frac=0.0, chord=1.2),
        vt.WingSection(y_frac=1.0, chord=0.6),
    ],
    position=np.array([5.0, 0.0, 0.5]),   # 5m behind, 0.5m above
    incidence=np.radians(-2.0),            # slight nose-down for trim
)

aircraft = vt.Aircraft(
    name="Wing-Tail Configuration",
    surfaces=[main_wing, h_tail],
)

# --- Analyse ------------------------------------------------------------------
print("=" * 60)
print("Ventorum Example 5: Wing + Horizontal Tail")
print("=" * 60)

settings = vt.SolverSettings(solver_type="vlm", n_panels=40)
result = vt.analyze(aircraft, alpha_deg=5.0, settings=settings)

for sw in result.spanwise:
    y_range = np.max(np.abs(sw.y))
    Cl_avg = np.mean(sw.Cl) if len(sw.Cl) > 0 else 0
    print(f"\n  [{sw.surface_name}]")
    print(f"    Avg Cl   = {Cl_avg:.4f}")
    print(f"    Span     = {2 * y_range:.2f} m")

t = result.totals
print("\n  [TOTAL]")
print(f"    CL   = {t.CL:.4f}")
print(f"    CDi  = {t.CDi:.6f}")
print(f"    e    = {t.e:.4f}")
print(f"    AR   = {t.AR:.2f}")

# --- Geometry plot ------------------------------------------------------------
fig_geom = vt.plot_geometry(aircraft, settings)

# --- Distribution plots -------------------------------------------------------
fig_dist = vt.plot_all_distributions(result)

# --- Compare wing-alone vs wing-tail -----------------------------------------
print("\n--- Comparison: Wing alone vs Wing+Tail ---")
result_alone = vt.analyze(
    vt.Aircraft(surfaces=[main_wing]),
    alpha_deg=5.0,
    settings=settings,
)
print(f"  Wing alone:  CL = {result_alone.totals.CL:.4f}  CDi = {result_alone.totals.CDi:.6f}")
print(f"  Wing+Tail:   CL = {result.totals.CL:.4f}  CDi = {result.totals.CDi:.6f}")

plt.show()
