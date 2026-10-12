# Get rate derivatives (p, q, r)

`FlightCondition` takes the body angular rates `p`, `q` and `r` [rad/s]. `ventorum.rate_derivatives` gives the derivatives of the force and moment coefficients per unit nondimensional rate: p b/(2V), q c/(2V) and r b/(2V).

## Rates in one solve

```python
import numpy as np
import ventorum as vt

wing = vt.LiftingSurface(
    name="wing",
    semi_span=5.0,
    sections=[vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.8)],
)
condition = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), p=0.2)   # roll rate 0.2 rad/s
res = vt.analyze(wing, condition, vt.SolverSettings(solver_type="vlm", n_panels=20))
print(f"Cl = {res.totals.Cl:.5f}")
```

- The rates are in the flight-dynamics body axes: `p > 0` right wing down, `q > 0` nose up, `r > 0` nose right (the signs of Cl, Cm and Cn).
- The rotation is about the moment reference point (`Aircraft.ref_point`, the origin by default). Set it to the centre of gravity.
- The model is quasi-steady: the wake stays straight along the free stream. A result gets a note when the rotation speed on the geometry is above 0.1 V.

## Rate derivatives

```python
aircraft = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.3, 0.0, 0.0]))
rd = vt.rate_derivatives(aircraft, condition.clone(), vt.SolverSettings(solver_type="vlm", n_panels=20),
                         axes="stability")
d = rd.derivatives
print(f"Cl_p={d['Cl_p']:.4f}  Cn_p={d['Cn_p']:.4f}  Cm_q={d['Cm_q']:.4f}  Cl_r={d['Cl_r']:.4f}  Cn_r={d['Cn_r']:.4f}")
print(rd.device, rd.n_solves)
```

- The keys are `"<coefficient>_<rate>"` for CL, CD, CY, Cl, Cm, Cn and p, q, r. `rd.matrix` has the same values as a (6, 3) array.
- `axes="stability"` gives the rates and the moments in the stability axes, as AVL does. `axes="body"` (default) and `axes="wind"` are also accepted. CL, CD and CY are relative to the free stream.
- The function uses central differences with the step `step=0.01` in each nondimensional rate. All cases are solved as one batch on one mesh.
- With `condition.h` (ground effect), the derivatives are at constant height of the reference point.
- The solvers `"vlm"`, `"linear"` and `"nonlinear"` accept rates. The Fourier solver does not. The cases with rates run on the selected device (`ventorum.gpu.set_device`): the GPU pipelines of the three solvers apply the rotation velocity at each point, in free air and in ground effect. With the device `"auto"`, a small batch can run on the CPU because it is faster there. `rd.device` and `rd.precision` say where and how the cases ran.
- `rd.to_dict()` gives a JSON-compatible dictionary.

## Which model to use

- Vortex lattice: the rotation velocity enters the boundary condition at each control point, so the chordwise load of a pitch rate is modelled. Use 8 or more chordwise panels for `Cm_q` of a tailless wing.
- Lifting lines: the section onset is taken at the three-quarter chord, and the section moment gets the thin-airfoil term of the rotation. They are not valid for swept wings (see the lifting-line notes).

The [theory chapter](../theory/rotation_rates.md) gives the equations, the assumptions and the verification.
