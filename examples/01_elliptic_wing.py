# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Example 1: Elliptic Wing - Analytical Validation
=================================================
An elliptic planform should produce e = 1.0 (perfect span efficiency).
This is the canonical benchmark for any LLT implementation.
"""

import numpy as np
import ventorum as vt

# --- Define an elliptic planform using many sections --------------------------
N_SEC = 21
y_fracs = np.linspace(0, 1, N_SEC)
c_root = 2.0
semi_span = 5.0

sections = []
for yf in y_fracs:
    # Elliptic chord: c(y) = c_root * sqrt(1 - (y/b_semi)^2)
    c = c_root * np.sqrt(1.0 - yf ** 2)
    c = max(c, 0.01)  # avoid zero chord at tip
    # Keep the 1/4 chord line straight
    x_le = 0.25 * c_root - 0.25 * c
    sections.append(vt.WingSection(y_frac=yf, chord=c, x_le=x_le))

wing = vt.LiftingSurface(
    name="Elliptic Wing",
    semi_span=semi_span,
    sections=sections,
)

# --- Run both solvers for comparison ------------------------------------------
print("=" * 60)
print("Ventorum Example 1: Elliptic Wing Validation")
print("=" * 60)

for solver_name in ["fourier", "vlm"]:
    result = vt.analyze(wing, alpha_deg=5.0, solver=solver_name, n_panels=60)
    t = result.totals
    print(f"\n[{solver_name.upper()} Solver]")
    print(f"  CL      = {t.CL:.6f}")
    print(f"  CDi     = {t.CDi:.8f}")
    print(f"  e       = {t.e:.6f}  (should be ~= 1.0)")
    print(f"  AR      = {t.AR:.2f}")
    print(f"  CL/CDi  = {t.CL / t.CDi:.1f}")

    if solver_name == "fourier" and result.fourier_coefficients is not None:
        A = result.fourier_coefficients
        print(f"  A1      = {A[0]:.6f}")
        print(f"  A3/A1   = {A[2] / A[0]:.6f}  (should be ~= 0)")

# --- Plot distributions from VLM solver ---------------------------------
result = vt.analyze(wing, alpha_deg=5.0, solver="vlm", n_panels=60)
fig = vt.plot_all_distributions(result)
# plt.show()

print("\n[OK] Validation complete.")
