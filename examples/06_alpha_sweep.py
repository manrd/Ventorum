# Author: Manuel Alejandro Rodriguez Diaz, PhD
import numpy as np
import matplotlib.pyplot as plt
import ventorum as vt

print("=" * 60)
print("Ventorum Example 6: Alpha Sweep and Drag Polar")
print("=" * 60)

# 1. Define a standard tapered wing
sections = [
    vt.WingSection(y_frac=0.0, chord=2.0),
    vt.WingSection(y_frac=1.0, chord=1.0, twist=np.radians(-2.0))
]
wing = vt.LiftingSurface(
    name="Tapered Wing",
    semi_span=5.0,
    sections=sections
)

# 2. Define a range of angles of attack
alphas_deg = np.arange(-4, 12, 1.0)
print(f"Sweeping from {alphas_deg[0]} deg to {alphas_deg[-1]} deg...")

# 3. Run the sweep using the convenience wrapper
results = vt.analyze_sweep(
    wing,
    alpha_deg_range=alphas_deg,
    V_inf=50.0,
    solver="vlm",
    n_panels=40
)

# 4. Plot the results using the built-in summary plot
print("Generating sweep summary plot...")
fig = vt.plot_sweep_summary(results, np.radians(alphas_deg))

print("Displaying plots. Close the window to exit.")
plt.show()
