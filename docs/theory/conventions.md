# Conventions

This chapter gives the axes, the angles, the reference values and the notation that all other chapters use.

## Axes and angles

The geometry axes have x aft (downstream), y to the right wing and z up. All lengths are in m and all angles in rad, unless the text gives deg. The aircraft does not rotate: the angle of attack $\alpha$ [rad] and the sideslip angle $\beta$ [rad] set the direction of the free stream in the geometry axes,

```{math}
\hat{\mathbf u} = (\cos\alpha\cos\beta,\; -\sin\beta,\; \sin\alpha\cos\beta).
```

Positive $\beta$ is wind from the right (nose left of the velocity vector). The free-stream velocity is $\mathbf V_\infty = V_\infty \hat{\mathbf u}$, with $V_\infty$ [m/s].

The lift direction is normal to the free stream, in the x-z plane of the geometry axes:

```{math}
\hat{\mathbf L} = (-\sin\alpha,\; 0,\; \cos\alpha), \qquad
\hat{\mathbf Y} = \hat{\mathbf L} \times \hat{\mathbf u}.
```

$\hat{\mathbf Y}$ is the direction of the side force, positive to the right.

## Moment signs and moment axes

The moments use the signs of the standard flight-dynamics body axes (x forward, y right, z down): the rolling moment $C_l > 0$ puts the right wing down, the pitching moment $C_m > 0$ is nose up and the yawing moment $C_n > 0$ is nose right. The geometry axes and the flight-dynamics body axes differ by a rotation of 180 deg about y, so for a moment vector $\mathbf M$ [N m] in the geometry axes

```{math}
C_l = -\frac{M_x}{q_\infty S_\text{ref} b_\text{ref}}, \qquad
C_m = \frac{M_y}{q_\infty S_\text{ref} c_\text{ref}}, \qquad
C_n = -\frac{M_z}{q_\infty S_\text{ref} b_\text{ref}},
```

with the dynamic pressure $q_\infty = \tfrac12 \rho V_\infty^2$ [Pa] and the air density $\rho$ [kg/m^3].

The moment coefficients can be given in three axis systems (Stevens, Lewis and Johnson 2016, chapter 2): body (the default), stability (body axes turned by $\alpha$ about y) and wind (stability axes turned by $\beta$ about z). For the vector $(C_l, C_m, C_n)$:

```{math}
R_{bs} = \begin{pmatrix} \cos\alpha & 0 & \sin\alpha \\ 0 & 1 & 0 \\ -\sin\alpha & 0 & \cos\alpha \end{pmatrix}, \qquad
R_{sw} = \begin{pmatrix} \cos\beta & \sin\beta & 0 \\ -\sin\beta & \cos\beta & 0 \\ 0 & 0 & 1 \end{pmatrix}, \qquad
R_{bw} = R_{sw} R_{bs}.
```

The force coefficients $C_L$, $C_D$ and $C_Y$ are always relative to the free stream ($\hat{\mathbf L}$, $\hat{\mathbf u}$ and $\hat{\mathbf Y}$).

## Reference values

If the user does not give them, the reference values come from the main surface and its mirror copies: the projected span $b_\text{ref}$ [m], the projected planform area $S_\text{ref}$ [m^2] and the mean aerodynamic chord (a surface with no projected span, for example a fin, uses the length along the surface)

```{math}
c_\text{ref} = \frac{1}{S_\text{ref}} \int c^2 \, dy .
```

The chord varies linearly between two sections, so on each interval of length $\Delta y$ the integral is exact: $\int c^2 dy = \tfrac{\Delta y}{3}(c_1^2 + c_1 c_2 + c_2^2)$. The aspect ratio is $AR = b_\text{ref}^2 / S_\text{ref}$. The moments are about the moment reference point (`Aircraft.ref_point`, the origin if it is not given).

## Section geometry

A section has a chord $c$ [m], a twist [rad] and an airfoil. Twist and surface incidence turn the section about its quarter-chord point, nose up positive, about the local spanwise axis in the y-z plane. The quarter-chord line therefore does not move with the twist. Between two sections, the chord, the twist and the leading-edge position are linear in the span fraction $\eta$.

## Notation

| Symbol | Meaning | Unit |
| --- | --- | --- |
| $\Gamma$ | circulation of a horseshoe vortex or a strip | m^2/s |
| $a_0$ | section lift slope | 1/rad |
| $\alpha_{L0}$ | section zero-lift angle | rad |
| $C_{d0}$, $C_{m0}$ | section profile drag and pitching moment (about the quarter chord) of a linear airfoil | - |
| $\alpha_\text{eff}$, $\alpha_i$ | section effective and induced angle of attack | rad |
| $h$ | height above the ground | m |
| $e$ | span efficiency $C_L^2 / (\pi\,AR\,C_{Di})$ | - |

## Implementation

- `ventorum.aero.system.freestream_direction`, `lift_direction`, `side_direction`.
- `ventorum.core.axes`: the rotations $R_{bs}$, $R_{sw}$, $R_{bw}$ and `transform_moments`.
- `ventorum.core.datatypes.Aircraft.compute_reference_values`.
- `ventorum.geometry.lattice.surface_edge_geometry`: the section geometry and the twist rotation.

## Verification

[V9 (sign conventions)](../verification_report.md) checks the signs of the rolling and yawing moments.

## References

- B. L. Stevens, F. L. Lewis and E. N. Johnson, *Aircraft Control and Simulation*, 3rd ed., Wiley, 2016, chapter 2.
