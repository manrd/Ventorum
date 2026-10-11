# Get control derivatives and hinge moments

`ventorum.control_derivatives` gives the change of the force and moment coefficients per unit deflection of each control, and the hinge moment of each copy of each control with its derivatives. `ventorum.hinge_moments` gives the hinge moments of one solved case, with no extra solve.

## Control derivatives

```python
import numpy as np
import ventorum as vt

wing = vt.LiftingSurface(
    name="wing",
    semi_span=5.0,
    sections=[vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.8)],
    controls=[
        vt.ControlSurface(name="flap", eta_start=0.0, eta_end=0.5, hinge_x_c=0.75),
        vt.ControlSurface(name="aileron", eta_start=0.6, eta_end=1.0, hinge_x_c=0.75, symmetric=False),
    ],
)
condition = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
settings = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=16)

res = vt.control_derivatives(wing, condition, settings)

for name, d in res.derivatives.items():
    print(f"{name}: CL_delta={d.CL:.4f}/rad  Cl_delta={d.Cl:.4f}/rad  Cm_delta={d.Cm:.4f}/rad")
for h in res.hinge:
    m = h.moment
    print(f"{m.control} ({m.side}): Ch={m.Ch:.4f}  Ch_alpha={h.Ch_alpha:.4f}/rad  "
          f"Ch_delta={h.Ch_delta[m.control]:.4f}/rad")
print(res.device, res.precision, res.n_solves)
```

- The derivatives are per radian of the `deflection` field. All controls with one name move together. The left half of an antisymmetric control (an aileron) moves by the opposite amount, so its `Ch_delta` with respect to its own name has the opposite sign of the right half.
- `axes="stability"` or `axes="wind"` gives the moment derivatives in the axes of the base condition. `CL`, `CD` and `CY` are always relative to the free stream.
- Each `ControlDerivatives` also gives the far-field diagnostics `CY_farfield` and `Cn_trefftz` (per rad; `Cn_trefftz` stays in stability axes). The near-field values stay the reported derivatives.
- `controls=["flap"]` limits the derivatives to some names. `hinge_alpha=False` skips the two solves for `Ch_alpha`.
- The method uses central differences with a 1 deg step (`step_deg`). Near the 30 deg limit it uses a one-sided difference; `scheme` tells which.
- `res.to_dict()` gives a JSON-compatible dictionary.

## Hinge moments of one case

```python
res_case = vt.analyze(wing, condition, settings)
for m in vt.hinge_moments(res_case):
    print(m.control, m.side, m.mode, f"H={m.H:.3f} N m", f"Ch={m.Ch:.4f}")
```

A hinge moment is positive when it tends to increase the local deflection of that copy (trailing edge down on a horizontal surface). The coefficient uses the flap area aft of the hinge and the root-mean-square flap chord.

## Which model gives the hinge moment

- Vortex lattice with `n_chord >= 2`: the panel loads aft of the hinge. The value converges slowly with `n_chord`; use 16 or more chordwise panels. The theory manual gives a convergence table.
- Lifting lines, and the vortex lattice with `n_chord = 1`: thin-airfoil theory of a plain flap on each strip (no gap, no nose balance, attached flow only).
- The Fourier solver does not accept control surfaces.

The [theory chapter](../theory/control_surfaces.md) gives the formulas and the verification.
