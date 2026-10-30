# Conventions

| Item | Convention |
| --- | --- |
| Geometry axes | x aft, y to the right, z up |
| Angles in `FlightCondition` | Radians. The short arguments of `analyze` (`alpha_deg`) are in degrees. |
| `semi_span` | Length along the dihedral line. If a section gives `z_le`, it is the projected y extent. |
| Reference values | `S_ref`, `b_ref`, `c_ref` are projected on the x-y plane; `c_ref` is the mean aerodynamic chord. Values you do not set come from the main surface (the surface with the largest projected planform area) and its mirror copy. |
| Moments | As in AVL: `Cl > 0` right wing down, `Cm > 0` nose up, `Cn > 0` nose right. |
| Moment reference point | `Aircraft.ref_point` (for example the centre of gravity), the origin if not set. |
| Sideslip | `beta > 0` is wind from the right. Positive dihedral gives `Cl_beta < 0`. |
| Mach number | Dimensionless, stored in `FlightCondition.mach`. Scope: $M < 0.3$. From 0.3 to below 0.7 an estimate rated UNRELIABLE; refused at 0.7 and above. Supported by VLM (`solver_type="vlm"`). |
| Wake | `wake_alignment="freestream"` (default) sends the wake along the free stream; `"body"` sends it along the x axis, as in AVL. |
| Units | SI: m, m/s, kg/m^3, N. |

## Moment axes

A moment vector `M = (Cl, Cm, Cn)` can be expressed in three right-handed
systems. All three share the AVL moment signs above. The force
coefficients `CL`, `CD` and `CY` keep their present definitions
(relative to the free stream) in every set; only the moments change.

| System | Definition |
| --- | --- |
| `body` | Fixed to the aircraft. The triple (Cl, Cm, Cn) is the moment vector in the flight-dynamics body axes (x forward, y right, z down); the geometry axes are x aft, y right, z up. This is the default of every solver result and every agent tool. |
| `stability` | The body axes turned by the angle of attack `alpha` about the y axis, so that x lies along the projection of the free stream on the plane of symmetry. |
| `wind` | The stability axes turned by the sideslip angle `beta` about the z axis, so that x lies along the free stream. |

The physical rotation acts on the dimensional moment vector (L_x, M_y, N_z).
Because Cl and Cn use b_ref while Cm uses c_ref, the coefficient vector
C = (Cl, Cm, Cn) transforms as:

    C' = L^-1 R L C,   with L = diag(b_ref, c_ref, b_ref)

With `ca = cos(alpha)`, `sa = sin(alpha)`, `cb = cos(beta)`,
`sb = sin(beta)` (alpha and beta in rad), the rotation matrices are:

- Body to stability (rotation by `alpha` about y):

```text
R_bs = [[ca, 0, sa],
        [ 0, 1,  0],
        [-sa, 0, ca]]
```

Because `R_bs` mixes only Cl and Cn which both use `b_ref`, `L^-1 R_bs L = R_bs`.
The transform equals the plain rotation for stability axes.

- Stability to wind (rotation by `beta` about the z axis of the flight-dynamics axes, z down):

```text
R_sw = [[cb, sb, 0],
        [-sb, cb, 0],
        [ 0,  0, 1]]
```

- Body to wind: `R_bw = R_sw R_bs`. The inverse maps use the transpose.

When `beta` is not 0 and `b_ref` is not equal to `c_ref`, Cl and Cm mix, and
`L^-1 R_bw L` scales each component by the reference lengths.

Two checks: at `alpha = 90` deg and `beta = 0`, `(Cl, Cm, Cn) = (1, 2, 3)`
gives `(3, 2, -1)` in stability and wind axes. At `alpha = 0` and
`beta = 90` deg, `(Cl, Cm, Cn) = (1, 2, 3)` gives `(2, -1, 3)` when
`b_ref = c_ref = 1`, and `(1, -2, 3)` when `b_ref = 2` and `c_ref = 1`. The function
`ventorum.core.axes.transform_moments` applies these maps, and
`SolverResult.moments(axes=...)` returns the stored body-axis moments in
the selected system (`axes="all"` returns the three sets).

## Mach number and compressibility

`FlightCondition` holds an optional `mach` field (`float | None = None`). When `mach` is `None` or `0.0`, the flow is incompressible. Scope: incompressible flow, Mach number below 0.3. With `FlightCondition.mach` from 0.3 to below 0.7, the vortex lattice applies the Prandtl-Glauert correction in the form of Goethert; the result is an estimate with the trust rating UNRELIABLE. A Mach number of 0.7 or more is refused with `ValidityError`. The constructor of `FlightCondition` accepts $0 \le M < 1$; the validity check of `analyze` and the solvers refuse $M \ge 0.7$.

When `mach > 0`, `analyze` compares it with `V_inf / a`, where `a` is the ISA speed of sound at the flight-condition density `rho` (340.29 m/s at sea level), the same value as in the envelope check of the speed. If the two differ by more than 0.05, `analyze` gives a `RuntimeWarning` and the trust score gets a note. The correction always uses the stated Mach number; the trust score gives the compressibility penalty of the larger of the two Mach numbers.

Only the vortex-lattice solver (`solver_type="vlm"`) supports compressible flow. It applies the Prandtl-Glauert compressibility correction in the form given by Goethert (1946), in free air. The CPU path and the GPU pipeline both apply it; the GPU pipeline solves each group of conditions with the same Mach number and free-stream direction as one batch.

The lifting-line solvers (linear, nonlinear, and Fourier) and the 3D panel method refuse `mach > 0` with `ValidityError`. Ground-effect analyses also refuse `mach > 0` with `ValidityError`. Adjoint gradients support `mach > 0` for the vortex lattice in free air, with the input `"mach"` (per unit Mach number); in ground effect and for the lifting lines they refuse `mach > 0` with `GradientScopeError`.

