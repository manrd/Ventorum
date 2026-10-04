# First analysis

This example analyses a tapered wing with wash-out at 5 deg angle of attack.

```python
import ventorum as vt

# 1. Define the sections (root and tip). Twist is in radians.
sections = [
    vt.WingSection(y_frac=0.0, chord=2.0),
    vt.WingSection(y_frac=1.0, chord=1.0, twist=-0.05),
]

# 2. Build the lifting surface (a symmetric wing of 10 m span).
wing = vt.LiftingSurface(name="Main Wing", semi_span=5.0, sections=sections)

# 3. Run the analysis. The default solver is the vortex-lattice method.
result = vt.analyze(wing, alpha_deg=5.0, V_inf=50.0)

# 4. Read the totals and the spanwise results.
print(f"CL  = {result.totals.CL:.4f}")
print(f"CDi = {result.totals.CDi:.5f}")
print(f"e   = {result.totals.e:.3f}")
print(result.totals.trust.summary_str())

# 5. Plot the spanwise distributions.
import matplotlib.pyplot as plt
fig = vt.plot_all_distributions(result)
plt.show()
```

`vt.analyze` accepts a `LiftingSurface` or an `Aircraft`. For full control, give a `FlightCondition` and a `SolverSettings` object instead of the short arguments.
