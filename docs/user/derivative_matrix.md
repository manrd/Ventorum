# Aerodynamic derivative matrix

`ventorum.derivative_matrix` gives the full aerodynamic derivative matrix of one flight state in one call.

The matrix contains the sensitivity of the six force and moment coefficients (CL, CD, CY, Cl, Cm, Cn) to static variables (alpha, beta, and in ground effect phi and h/c), control surface deflections, and body rotation rates (p_hat, q_hat, r_hat).

## Example

```python
import numpy as np
import ventorum as vt

wing = vt.LiftingSurface(
    name="wing",
    semi_span=5.0,
    sections=[vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.8)],
    controls=[
        vt.ControlSurface(name="flap", eta_start=0.1, eta_end=0.5, hinge_x_c=0.75, deflection=np.radians(2.0)),
        vt.ControlSurface(name="aileron", eta_start=0.6, eta_end=0.95, hinge_x_c=0.75, deflection=0.0, symmetric=False),
    ],
)
aircraft = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.3, 0.0, 0.0]))
condition = vt.FlightCondition(V_inf=35.0, alpha=np.radians(3.0), beta=np.radians(1.0))
settings = vt.SolverSettings(solver_type="linear", n_panels=16)

res = vt.derivative_matrix(aircraft, condition, settings, axes="body")

print(f"Matrix shape: {res.matrix.shape}")
print(f"Columns: {res.columns}")
print(f"Rows: {res.rows}")
print(f"Static margin: {res.diagnostics['static_margin']:.4f}")
print(f"Neutral point x: {res.diagnostics['neutral_point_x']:.4f} m")
print(res.as_table())
```

## Matrix layout

The matrix has six fixed rows:
- CL: lift coefficient [-].
- CD: drag coefficient [-].
- CY: side-force coefficient [-].
- Cl: rolling moment coefficient [-].
- Cm: pitching moment coefficient [-].
- Cn: yawing moment coefficient [-].

The columns appear in group order:
1. Static: `alpha` and `beta` [1/rad]. In ground effect, `phi` [1/rad] and `h_over_c` [-] follow.
2. Control: `delta:<name>` [1/rad] for each control surface.
3. Rate: `p_hat`, `q_hat`, `r_hat` [-] for the nondimensional body rotation rates ($p b / 2V$, $q c / 2V$, $r b / 2V$).

## Axes

The `axes` argument selects the axis system:
- `"body"` (default): fixed to the geometry.
- `"stability"`: turned by alpha about the y axis.
- `"wind"`: turned by beta about the z axis.
- `"all"`: returns a dictionary with matrices in body, stability, and wind axes.

Force rows (CL, CD, CY) are relative to the free stream in every set. In the static, height and control columns, the moment rows turn with the reference span $b_\text{ref}$ and chord $c_\text{ref}$. In the rate columns, the rates turn together with the moments, as in `rate_derivatives`. Thus each column equals the result of its source function in the same axes.

## Ground effect

When `condition.h` is set, `derivative_matrix` adds the height group by default. The static columns come from height derivatives at fixed pivot height. Requesting the height group in free air raises a `ValueError`.

## Diagnostics and export

`DerivativeMatrixResult` contains diagnostics:
- `static_margin`: static margin fraction of body-axis pitch stability.
- `neutral_point_x`: neutral point x position in geometry coordinates [m].
- `irodov_margin`: height-pitch stability margin in ground effect.
- `controls`: one entry per control name, `diagnostics["controls"][name]`, with the Trefftz-plane lateral derivatives `CY_farfield` and `Cn_trefftz` [1/rad] and the trust warnings when the near/far lateral rule applies.

Export methods:
- `res.to_dict()`: returns a JSON-safe dictionary with lists for arrays and None for non-finite values.
- `res.as_table()`: returns a formatted Markdown table with axis labels.
