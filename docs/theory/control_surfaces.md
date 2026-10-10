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

## Assumptions and limits

The control surface model uses these assumptions:
- Small deflection angles: the formulation assumes $|\delta| \le 30^\circ$.
- No gap between flap and main wing surface.
- No flap-edge vortex model other than the vortex lattice itself.
- No viscous separation effects: profile drag shifts with the effective angle of attack only.
- In chordwise mode, the trailing edge nodes do not translate. Ground-clearance checks do not detect deflected flap positions.
- A snapped panel can be up to 1.25 times its normal length. The ground-effect rule that limits chordwise panel length to the gap height does not see this extension.
- At control side edges, the bound vortices of neighbouring strips can end at different chordwise locations.
- The classical Fourier solver does not support deflected control surfaces.

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
- `test_span_limits_snap_to_strip_edges`
- `test_hinge_snaps_on_control_strips_only`
- `test_cache_key_follows_every_control_field`
- `test_set_deflection_links_controls_by_name`
- `test_invalid_controls_are_refused`
- `test_fourier_refuses_deflection`
- `test_json_round_trip_keeps_controls`
- `test_control_in_ground_effect_runs`
- `test_gpu_control_surfaces_equal_cpu`

## References

- H. Glauert, *The Elements of Aerofoil and Airscrew Theory*, Cambridge University Press, 1926.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001.
