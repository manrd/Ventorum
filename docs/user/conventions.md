# Conventions

| Item | Convention |
| --- | --- |
| Geometry axes | x aft, y to the right, z up |
| Angles in `FlightCondition` | Radians. The short arguments of `analyze` (`alpha_deg`) are in degrees. |
| `semi_span` | Length along the dihedral line. If a section gives `z_le`, it is the projected y extent. |
| Reference values | `S_ref`, `b_ref`, `c_ref` are projected on the x-y plane; `c_ref` is the mean aerodynamic chord. Values you do not set come from the main surface (the first surface that is not vertical) and its mirror copy. |
| Moments | As in AVL: `Cl > 0` right wing down, `Cm > 0` nose up, `Cn > 0` nose right. |
| Moment reference point | `Aircraft.ref_point` (for example the centre of gravity), the origin if not set. |
| Sideslip | `beta > 0` is wind from the right. Positive dihedral gives `Cl_beta < 0`. |
| Wake | `wake_alignment="freestream"` (default) sends the wake along the free stream; `"body"` sends it along the x axis, as in AVL. |
| Units | SI: m, m/s, kg/m^3, N. |

## Moment axes

A moment vector `M = (Cl, Cm, Cn)` can be expressed in three right-handed
systems. All three share the AVL moment signs above. The force
coefficients `CL`, `CD` and `CY` keep their present definitions
(relative to the free stream) in every set; only the moments change.

| System | Definition |
| --- | --- |
| `body` | Fixed to the aircraft (x aft, y right, z up). This is the default of every solver result and every agent tool. |
| `stability` | The body axes turned by the angle of attack `alpha` about the y axis, so that x lies along the projection of the free stream on the plane of symmetry. |
| `wind` | The stability axes turned by the sideslip angle `beta` about the z axis, so that x lies along the free stream. |

With `ca = cos(alpha)`, `sa = sin(alpha)`, `cb = cos(beta)`,
`sb = sin(beta)` (alpha and beta in rad), a moment vector transforms as
`M_out = R M_in`:

- Body to stability (rotation by `alpha` about y):

```text
R_bs = [[ca, 0, sa],
        [ 0, 1,  0],
        [-sa, 0, ca]]
```

- Stability to wind (rotation by `-beta` about z):

```text
R_sw = [[cb, sb, 0],
        [-sb, cb, 0],
        [ 0,  0, 1]]
```

- Body to wind: `R_bw = R_sw R_bs`. The inverse maps use the transpose.

Two checks: at `alpha = 90` deg and `beta = 0`, `(Cl, Cm, Cn) = (1, 2, 3)`
gives `(3, 2, -1)` in stability and wind axes. At `alpha = 0` and
`beta = 90` deg it gives `(2, -1, 3)` in wind axes. The function
`ventorum.core.axes.transform_moments` applies these maps, and
`SolverResult.moments(axes=...)` returns the stored body-axis moments in
the selected system (`axes="all"` returns the three sets).

