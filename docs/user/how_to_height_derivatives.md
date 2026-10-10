# Get height derivatives and the Irodov criterion

`ventorum.height_derivatives` gives the static derivatives of one flight state in ground effect: the derivatives of CL, CD, CY, Cl, Cm and Cn with respect to the angle of attack, the sideslip angle, the bank angle and the height. It also gives the pitch and height aerodynamic centres and the Irodov margin of that state.

## One flight state

```python
import numpy as np
import ventorum as vt

wing = vt.LiftingSurface(
    name="wing",
    semi_span=1.5,
    sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
)
aircraft = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.3, 0.0, 0.0]))   # the centre of gravity
condition = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=0.3, phi=np.radians(2.0))
hd = vt.height_derivatives(aircraft, condition, vt.SolverSettings(solver_type="vlm", n_panels=16))
d = hd.derivatives
print(f"CL_alpha={d['CL_alpha']:.3f}  CL_h={d['CL_h']:.4f}  Cm_h={d['Cm_h']:.4f}  Cl_phi={d['Cl_phi']:.4f}")
print(hd.irodov)
print(hd.device, hd.n_solves, hd.n_chord)
```

- `condition.h` [m] is the height of the moment reference point (the pivot) above the ground. It must be set: the function works only in ground effect.
- Every derivative is about the pivot, with the height of the pivot held fixed. A change of alpha or of the bank angle turns the ground about the pivot. A change of height moves the ground at constant attitude.
- The keys are `"<coefficient>_<variable>"` for CL, CD, CY, Cl, Cm, Cn and `alpha`, `beta`, `phi` [1/rad] and `h` [per unit h/c_ref]. `hd.matrix` has the same values as a (6, 4) array.
- Positive `phi` puts the right wing nearer to the ground. A bank-restoring rolling moment has `Cl_phi < 0`.
- `axes="stability"` or `axes="wind"` gives the moments in those axes of the base state. CL, CD and CY are relative to the free stream.
- `hd.to_dict()` gives a JSON-compatible dictionary.

## Irodov criterion

`hd.irodov` holds `x_alpha_m` and `x_h_m` (the pitch and height aerodynamic centres [m] aft of the pivot), `margin` = (x_alpha - x_h)/c_ref and `verdict`. Longitudinal static stability in ground effect at constant speed needs `margin > 0`: the height centre must be ahead of the pitch centre (R. D. Irodov 1970). The criterion applies only when the pivot is the centre of gravity. Set `Aircraft.ref_point` to the centre of gravity.

## Steps and mesh

- Central differences with the steps `alpha_step_deg=0.5`, `beta_step_deg=1.0`, `phi_step_deg=0.5` and `h_step=0.05` (a fraction of the smallest clearance of the base state). `hd.steps` gives the steps used.
- All nine cases are solved as one batch on one mesh. With `n_chord=None`, all cases use the chordwise count of the case with the smallest gap.
- A case that touches the ground stops the function with `GroundStrikeError`; the message names the case. Use smaller steps or a larger height.
- The solves run on the selected device; `hd.device` and `hd.precision` say which.

## Which solver to use

- Vortex lattice (default): valid down to small gaps. The trust score warns below h_min/c = 0.3.
- Lifting lines: refused below h_min/c = 1 (see the ground-effect notes). The Fourier solver has no ground effect.

For a grid of heights, angles and bank angles, use `GroundEffectSweep` (see [Ground effect](how_to_ground_effect.md)). The [theory chapter](../theory/ground_effect.md) gives the method and its limits.
