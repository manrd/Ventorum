# Control surfaces

This chapter describes the aerodynamic models of hinged control surfaces in Ventorum.

## Purpose

Control surfaces change the lift, pitching moment, rolling moment and yawing moment of lifting surfaces. A user can define control surfaces such as flaps, ailerons, elevators and rudders on any lifting surface. The vortex lattice method, the linear lifting line and the nonlinear lifting line calculate the effect of control deflection.

## Definition and signs

A control surface is a hinged aft portion of a lifting surface. The hinge position $E_h = \text{hinge\_x\_c}$ is the fraction of local chord from the leading edge.

A positive deflection is a right-hand rotation of the flap about its hinge axis. The axis points from the first to the second edge of each strip in the lattice order. On the defining half of a surface, this order runs from root to tip. On a horizontal surface, a positive deflection moves the trailing edge down on both halves. On a vertical fin with dihedral 90 deg, a positive deflection moves the trailing edge to $+y$. This sign matches a positive section twist.

On the left half of a symmetric surface and on a mirror copy, deflection depends on the `symmetric` flag:
- When `symmetric=True` (flaps, elevators), the flap deflection mirrors the right half. The rotation angle is $+\delta$ about the hinge axis on both halves.
- When `symmetric=False` (ailerons), the flap deflection is antisymmetric. The rotation angle is $+\delta$ on the right half and $-\delta$ on the left half.

When a symmetric surface has an antisymmetric control (`symmetric=False`) with non-zero deflection, Ventorum does not fold symmetry. The full aircraft lattice is solved.

## Vortex lattice: rotated boundary condition

The vortex lattice method uses the chordwise mode when `n_chord >= 2`.

The mesh moves the interior chordwise panel edge nearest to the hinge onto the hinge coordinate `hinge_x_c`. This move occurs only when both neighbouring panels keep a length between 0.5 and 1.25 times their length before the move. If this length rule fails, the edge does not move. The effective hinge coordinate `hinge_x_c_eff` is then the nearest existing interior edge.

A panel is on the flap when its leading chordwise edge fraction is at or behind `hinge_x_c_eff`. For each flap panel, the boundary-condition normal vector turns by the signed deflection angle about the strip hinge axis. The panel node positions, the wake geometry, the strip normal vector and the chord direction vector do not change. The airfoil data do not change in this mode.

## Lifting line: thin-airfoil flap theory

The lifting line solvers and the single-panel vortex lattice (`n_chord = 1`) use the section mode.

The strip airfoil is replaced by a deflected airfoil derived from the thin-airfoil theory of a hinged flap. With chord coordinate $x/c = (1 - \cos\theta) / 2$, the hinge at $x/c = E_h$ corresponds to angle:

$$\theta_h = \arccos(1 - 2 E_h)$$

Flap deflection $\delta$ produces changes in lift and pitching moment:

$$C_l = 2 \pi (\alpha + \tau \delta)$$

$$C_{m,c/4} = C_{m0} + \frac{dC_m}{d\delta} \delta$$

The theoretical flap effectiveness factor $\tau$ and moment derivative are:

$$\tau = 1 - \frac{\theta_h - \sin\theta_h}{\pi}$$

$$\frac{dC_m}{d\delta} = -\frac{1}{2} \sin\theta_h (1 - \cos\theta_h)$$

Representative values:

| hinge_x_c | $\theta_h$ [rad] | $\tau$ | $dC_m/d\delta$ [1/rad] |
| --- | --- | --- | --- |
| 0.60 | 1.772154 | 0.747785 | -0.587878 |
| 0.70 | 1.982313 | 0.660746 | -0.641561 |
| 0.75 | 2.094395 | 0.608998 | -0.649519 |
| 0.80 | 2.214297 | 0.549815 | -0.640000 |
| 0.90 | 2.498092 | 0.395819 | -0.540000 |

The deflected airfoil applies the same rule to the section polar of the strip. The flap shifts the polar by $\tau \delta$ in angle of attack and adds $\frac{dC_m}{d\delta} \delta$ to the pitching moment:

$$C_{l,\text{new}}(\alpha) = C_l(\alpha + \tau \delta)$$

$$C_{d,\text{new}}(\alpha) = C_d(\alpha + \tau \delta)$$

$$C_{m,\text{new}}(\alpha) = C_m(\alpha + \tau \delta) + \frac{dC_m}{d\delta} \delta$$

For a `TabulatedAirfoil`, the new table has the angles $\alpha_{\text{new}} = \alpha - \tau \delta$, the same lift and drag rows, and the pitching-moment rows plus $\frac{dC_m}{d\delta} \delta$. Thus the pitching moment follows the shifted lift curve, also in stall: at an angle of the base table, the deflected section has the same lift coefficient, and its pitching moment changes by $\frac{dC_m}{d\delta} \delta$ only. The angle range of the table also moves by $-\tau \delta$. A table without pitching-moment data gets $C_{m,\text{new}} = \frac{dC_m}{d\delta} \delta$ at all angles.

For a `LinearAirfoil`, the pitching moment and the profile drag do not change with the angle of attack. The same rule then gives:
- $\alpha_{L0,\text{new}} = \alpha_{L0} - \tau \delta$
- $C_{m0,\text{new}} = C_{m0} + \frac{dC_m}{d\delta} \delta$
- the lift slope $a_0$ and $C_{d0}$ do not change.

## Hinge moments

The hinge moment $H$ of a copy of a control is the aerodynamic moment about its hinge line. It is positive when it tends to increase the local deflection of the copy: a right-hand moment about the hinge axis of the strips (trailing edge down on a horizontal surface, on both halves). The coefficient is

$$C_h = \frac{H}{q\, S_f\, \bar c_f}, \qquad S_f = \sum_i c_{f,i}\, w_i, \qquad \bar c_f = \frac{\sum_i c_{f,i}^2\, w_i}{S_f}$$

with $q$ the dynamic pressure, $c_{f,i} = (1 - E_h)\, c_i$ the flap chord aft of the hinge and $w_i$ the width of strip $i$. With this root-mean-square flap chord, $C_h$ is the $c_f^2 w$-weighted mean of the strip values, and a two-dimensional section gives the section coefficient. The reference area and chord of the control aft of the hinge follow Etkin and Reid (1996). `ventorum.hinge_moments(result)` gives the hinge moment of each copy of each control of a solved case: the right and left halves of a symmetric surface are separate items.

### Thin-airfoil section hinge moment

Thin-airfoil theory (Glauert 1926) gives the load of a flat plate at the angle $\alpha$ with a flap deflected by $\delta$. With $x/c = (1 - \cos\theta)/2$, the Fourier coefficients of the load are

$$A_0 = \alpha + \delta\left(1 - \frac{\theta_h}{\pi}\right), \qquad A_n = \frac{2\,\delta\,\sin n\theta_h}{n\,\pi}$$

The section hinge moment is the integral of the load times its arm aft of the hinge, $C_h = -\frac{2}{E^2}\int_{x_h}^{1} \frac{\gamma}{V}\,(x - x_h)\,dx$ with $E = 1 - E_h$. The series of the flap terms sums to a logarithmic load at the hinge, $\frac{\delta}{\pi}\ln\left|\sin\frac{\theta + \theta_h}{2} \big/ \sin\frac{\theta - \theta_h}{2}\right|$. The integrals give $C_h = C_{h\alpha}\,\alpha + C_{h\delta}\,\delta$ with

$$C_{h\alpha} = -\frac{(\pi - \theta_h)\left(\cos\theta_h - \frac{1}{2}\right) + \sin\theta_h\left(1 - \frac{\cos\theta_h}{2}\right)}{E^2}$$

$$C_{h\delta} = \left(1 - \frac{\theta_h}{\pi}\right) C_{h\alpha} - \frac{\sin\theta_h\left[(\pi - \theta_h)\cos\theta_h + \sin\theta_h\right]}{2\pi E^2}$$

The second integral uses an integration by parts, with the derivative $-\sin\theta_h / (\cos\theta_h - \cos\theta)$ of the logarithm. At $E_h \to 0$ both derivatives go to $-\pi/2$ (the moment of a flat plate about its leading edge). At $E_h \to 1$, $C_{h\alpha} \to 0$ and $C_{h\delta} \to -8/(3\pi)$. The function `hinge_moment_derivatives(hinge_x_c)` gives the two values:

| hinge_x_c | $C_{h\alpha}$ [1/rad] | $C_{h\delta}$ [1/rad] |
| --- | --- | --- |
| 0.60 | -0.744804 | -1.012657 |
| 0.70 | -0.627407 | -0.965403 |
| 0.75 | -0.565347 | -0.943608 |
| 0.80 | -0.499382 | -0.922877 |
| 0.90 | -0.344856 | -0.884229 |

### Section mode

The lifting lines and the single-panel vortex lattice use the section hinge moment on each strip of the control:

$$C_{h,i} = C_{h\alpha}\,\alpha_{fp,i} + C_{h\delta}\,\delta_i, \qquad \alpha_{fp,i} = \alpha_{\text{eff},i} - \alpha_{L0,i} - \tau\,\delta_i$$

Here $\alpha_{L0,i}$ is the zero-lift angle of the deflected strip section and $\delta_i$ the local deflection. In the linear range, $\alpha_{fp,i} = C_{l,i}/a_0 - \tau\,\delta_i$: the flat-plate angle that gives the strip lift together with the flap. The camber of the base section adds no hinge moment in this model.

### Chordwise mode

The vortex lattice with `n_chord >= 2` uses the panel loads. The force on the bound vortex of each flap panel is the Kutta-Joukowski force $\rho\,\Gamma\,(\mathbf V_\infty + \mathbf v) \times \mathbf l$ (Katz and Plotkin 2001), with the induced velocity $\mathbf v$ at the force point from all horseshoes, the ground images and the wake of the case. $H$ is the sum of the moments of these forces about the hinge line of each strip. The forces on the chordwise legs are not included: the two legs of one horseshoe carry opposite vorticity over the same chordwise interval, so their moments about the hinge cancel, except for the difference of the induced velocity at the two legs.

The thin-airfoil load is logarithmic at the hinge, so the lattice hinge moment converges slowly with the number of chordwise panels. On a rectangular wing of aspect ratio 40 (hinge 0.75, uniform chordwise panels, $\delta = 5$ deg, $\alpha = 3$ deg, 20 spanwise panels per half), the lattice value approaches the thin-airfoil value at the lift of each strip:

| n_chord | $C_h$ (lattice) | $C_h$ (thin airfoil at the strip lift) | difference |
| --- | --- | --- | --- |
| 4 | -0.11488 | -0.10466 | +9.77 % |
| 8 | -0.11263 | -0.10599 | +6.26 % |
| 16 | -0.11016 | -0.10672 | +3.22 % |
| 32 | -0.10847 | -0.10710 | +1.28 % |
| 64 | -0.10750 | -0.10729 | +0.20 % |

## Control derivatives

`ventorum.control_derivatives(geometry, condition, settings)` gives, for each control name, the derivatives of $C_L$, $C_D$, $C_Y$, $C_l$, $C_m$ and $C_n$ per unit deflection [1/rad]. For each copy of each control, it gives the hinge moment with its derivatives $\partial C_h/\partial\delta$ (with respect to every control name) and $\partial C_h/\partial\alpha$.

The derivatives are central finite differences of solves at $\delta \pm h$ (default $h = 1$ deg). All controls with one name move together, and the derivative is per unit change of the `deflection` field: the left half of an antisymmetric control moves by $-h$. When $|\delta| + h$ passes the 30 deg limit, the second-order one-sided difference $\left(3C(\delta) - 4C(\delta \mp h) + C(\delta \mp 2h)\right)/(\pm 2h)$ replaces the central difference. $\partial C_h/\partial\alpha$ uses $\alpha \pm 0.5$ deg; in ground effect the moment reference point stays at its height. The moment derivatives are in the body axes (default), or in the stability or wind axes of the base condition, held fixed. The solves with a changed deflection use the same mesh: the hinge snap and the control strips do not depend on the deflection.

The same differences give the derivatives of the far-field diagnostics `CY_farfield` (wind axes) and `Cn_trefftz` (stability axes of the base condition, for every choice of axes); see [Forces and moments](forces_and_moments.md). `to_dict` gives them as `CY_farfield_delta_per_rad` and `Cn_trefftz_delta_per_rad`. The near-field derivatives stay the reported values.

The solves run on the device that the settings select, and the result gives the device and the precision. With a 1 deg step, the float32 round-off of the GPU paths is about $10^{-5}$ of a derivative.

## Assumptions and limits

The control surface model uses these assumptions:
- Small deflection angles: the formulation assumes $|\delta| \le 30^\circ$.
- No gap between flap and main wing surface.
- No flap-edge vortex model other than the vortex lattice itself.
- No viscous separation effects: profile drag shifts with the effective angle of attack only.
- In chordwise mode, the trailing edge nodes do not translate. Ground-clearance checks do not detect deflected flap positions.
- A snapped panel can be up to 1.25 times its normal length. The ground-effect rule that limits chordwise panel length to the gap height does not see this extension.
- At control side edges, the bound vortices of neighbouring strips can end at different chordwise locations.
- The side force of an antisymmetric deflection (aileron) on a flat wing is not modelled: the rotated-normal model has no flap side edges, and the leg forces use the tangential velocity only (see [Forces and moments](forces_and_moments.md)). On a thin wing this side force is small and proportional to the angle of attack times the deflection; DATCOM neglects it for plain wings. The far-field `CY_farfield` does not show it either: the Trefftz trace uses the trailing edges of the lattice, which a deflection does not move.
- The classical Fourier solver does not support deflected control surfaces.
- The section hinge moment is the thin-airfoil value of a plain flap: no gap, no nose balance, no camber term, attached flow only. Past stall it has no physical meaning.
- The chordwise hinge moment converges slowly with the number of chordwise panels (table above). Use 16 or more chordwise panels for hinge moments.

## Verification

The implementation is verified in `tests/test_control_surfaces.py`:
- `test_flap_effectiveness_matches_thin_airfoil_table`
- `test_zero_deflection_full_span_gives_the_same_bits`
- `test_section_mode_equals_shifted_airfoil`
- `test_deflected_airfoil_follows_the_table_rule`
- `test_vlm_flap_ratio_approaches_tau`
- `test_flap_signs`
- `test_aileron_signs_and_no_symmetry_fold`
- `test_rudder_signs`
- `test_twin_rudders_with_mirror_copy`
- `test_symmetric_wing_equals_half_plus_mirror_copy`
- `test_span_limits_snap_to_strip_edges`
- `test_hinge_snaps_on_control_strips_only`
- `test_cache_key_follows_every_control_field`
- `test_set_deflection_links_controls_by_name`
- `test_invalid_controls_are_refused`
- `test_fourier_refuses_deflection`
- `test_json_round_trip_keeps_controls`
- `test_control_in_ground_effect_runs`
- `test_gpu_control_surfaces_equal_cpu`

The hinge moments and the control derivatives are verified in `tests/test_control_derivatives.py`:
- `test_thin_airfoil_hinge_derivatives_equal_quadrature` (the closed forms against a quadrature of the load)
- `test_thin_airfoil_hinge_derivatives_limits`
- `test_vlm_hinge_moment_converges_to_thin_airfoil`
- `test_section_mode_hinge_moment_equals_thin_airfoil_formula`
- `test_hinge_moment_signs`
- `test_hinge_moments_symmetric_wing_equal_half_plus_mirror`
- `test_lifting_line_flap_derivative_is_tau_times_lift_slope`
- `test_vlm_full_span_flap_effectiveness_ratio`
- `test_aileron_derivatives`
- `test_one_sided_difference_near_the_limit`
- `test_moment_derivatives_in_stability_axes`
- `test_control_derivatives_in_ground_effect`
- `test_gpu_hinge_moments_equal_cpu`

## References

- H. Glauert, *The Elements of Aerofoil and Airscrew Theory*, Cambridge University Press, 1926.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001.
- B. Etkin and L. D. Reid, *Dynamics of Flight: Stability and Control*, 3rd ed., Wiley, 1996.
