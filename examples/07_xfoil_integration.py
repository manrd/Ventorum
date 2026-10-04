# Author: Manuel Alejandro Rodriguez Diaz, PhD
import numpy as np
import matplotlib.pyplot as plt
import ventorum as vt

print("=" * 60)
print("Ventorum Example 7: XFOIL Integration")
print("=" * 60)

# Generate a polar on-the-fly using XFOIL
print("Calling XFOIL to generate NACA 4412 polar at Re = 1,000,000...")
try:
    polar_4412 = vt.run_xfoil(
        airfoil="naca 4412",
        Re=1e6,
        alpha_start=-5.0,
        alpha_end=15.0,
        alpha_step=1.0,
        max_iter=150
    )
    print("Successfully generated polar!")
except FileNotFoundError:
    print("\nXFOIL is not installed or not in the system PATH; this example is skipped.")
    print("Install XFOIL and make sure the 'xfoil' command runs to use it.")
    raise SystemExit(0) from None
except RuntimeError as e:
    print(f"\nError running XFOIL:\n{e}")
    raise SystemExit(1) from None

# Define a wing using the generated polar
sections = [
    vt.WingSection(y_frac=0.0, chord=1.5, airfoil=polar_4412),
    vt.WingSection(y_frac=1.0, chord=0.75, airfoil=polar_4412)
]

wing = vt.LiftingSurface(
    name="XFOIL Wing",
    semi_span=4.0,
    sections=sections
)

# Analyze the wing
print("\nAnalyzing wing at alpha = 8.0 deg using Nonlinear Solver...")
result = vt.analyze(wing, alpha_deg=8.0)

print("\nResults:")
print(f"  CL = {result.totals.CL:.4f}")
print(f"  CDi = {result.totals.CDi:.5f}")
print(f"  CDp = {result.totals.CDp:.5f}")
print(f"  CD_total = {result.totals.CD_total:.5f}")

# Plot the drag polar for the 3D wing
print("\nRunning alpha sweep to generate 3D drag polar...")
sweep_results = vt.analyze_sweep(
    wing,
    alpha_deg_range=np.arange(-2, 12, 2.0),
    solver="nonlinear"
)

fig = vt.plot_sweep_summary(sweep_results, np.radians(np.arange(-2, 12, 2.0)))
plt.show()
