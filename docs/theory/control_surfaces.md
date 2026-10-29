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

## Vortex lattice: geometric deflection

The vortex lattice method uses the chordwise mode when `n_chord >= 2`.

The mesh moves the interior chordwise panel edge nearest to the hinge onto the hinge coordinate `hinge_x_c`. This move occurs only when both neighbouring panels keep a length between 0.5 and 1.25 times their length before the move. If this length rule fails, the edge does not move. The move applies to the strips of that control and of its joins (see below). The effective hinge coordinate `hinge_x_c_eff` is then the nearest existing interior edge.

A panel is on the flap when its leading chordwise edge fraction is at or behind `hinge_x_c_eff`. The hinge point of a strip edge is the point at this fraction on the section mean surface; a section without a mean line uses its chord line. The points of a strip edge aft of the hinge (bound endpoints and trailing-edge point) turn about the hinge axis of that edge node: the line through its hinge point along the central difference of the hinge points of the two neighbour nodes (one-sided at the ends of a surface). The two strips at a node therefore turn it about the same line by the same angle, and the deflected surface has no gap between strips, also at the root of a surface with dihedral. A control point turns about the axis, the hinge point and the angle interpolated between the two edges at its span fraction, and the boundary-condition normal of a flap panel turns with it. Points at or ahead of the hinge stay fixed; no second normal rotation is applied. This is a camber-surface vortex lattice (Katz and Plotkin 2001).

The chordwise legs of a panel ahead of the hinge run from its bound point to the hinge point of its strip edge, then to the deflected trailing edge (one extra straight segment per leg, `VortexLattice.a_kink` and `b_kink`), so they follow the deflected mean surface. The leg forces act on both segments. The node is there on every strip of a chordwise control, also at zero deflection, where it lies on the straight leg: the loads and their derivatives are continuous in the deflection. Strips without a chordwise control have straight legs and keep their results bit for bit; on a strip of an undeflected chordwise control only the leg-force integration over the two segments changes the results, at the round-off of that integration. On a strip with geometric camber the legs run through the bound points of the panels aft of them (the leg chains of [Vortex kernels](vortex_kernels.md)), and the hinge point is the node between the two bound points on both sides of the hinge, in place of the mean-surface node there: one node mechanism gives both. The two strips at an edge must have the same nodes there, so that their legs coincide. Where a flap strip meets a strip with a different hinge, the hinge point is therefore not a node of either strip at that edge: the legs of both use the mean-surface node (the flap field entry `hinge_off`). This applies at the outer edge of a join (the other strip has no hinge), at the edge between two controls with different hinge fractions, and at the end of a surface joined to another surface whose end strip has no hinge or another hinge. On a cambered section the hinge point and the mean-surface node are not on one line, and two legs with different nodes on one edge are apart (a few percent of the chord on a cambered section). The velocity of one leg at the other leg then grows as the vortex core of narrower strips gets smaller, and so does the leg force: CL did not converge with the spanwise panels, also at zero deflection. At the outer edge of a join the deflection is zero, so the hinge point is not a kink of the surface there. At the other two edges the deflection is not zero in general; see the limits below.

Sealed side edges (joins). At a side edge of a control the deflection changes linearly over a join of spanwise length `CONTROL_JOIN` (0.2) times the local chord, centred on the edge (`ventorum.geometry.controls.ControlJoinField`). This is the transition of the 3D panel method (`CONTROL_TRANSITION`), with the same rules: the join is not longer than the control, does not reach past the root or the tip, and shares the room to the next control edge with the join of that edge. A join also lies between two controls with different deflections, and at the root of a symmetric surface under an antisymmetric control. The deflection of a node is a function of its span position only, so the join is the same geometry on every mesh. The strips in a join outside the control take its hinge fraction and its chordwise nodes. Without the join, the separated edge points of a deflected strip and its undeflected neighbour act as an open slit: the circulation next to the side edge falls as the strips become narrower, and a part-span flap does not converge with the spanwise mesh (`test_part_span_flap_spanwise_refinement`).

The wake starts at the deflected trailing edge and follows the selected wake direction. The same geometric model applies in free air and in ground effect. The ground images, clearance check and Trefftz plane use the deflected points. The strip reference chord, area, normal and airfoil data keep their reference values. Hinge moments of cambered sections use the mean-line hinge even at zero deflection; these replace the former chord-line hinge moments. Rigid lattice transforms carry the hinge endpoints and the leg-force rules with the other geometry.

The leg-force rules of a lattice with a chordwise control come from the lattice before the flap turn, which has the same topology, and are kept for every deflection (see [Forces and moments](forces_and_moments.md)). The legs of two strips that turn about slightly different lines separate a little with the deflection; a rule from the deflected points would change when the gap passes the join tolerance, and the loads would jump (`test_leg_rules_from_undeflected_lattice`).

Chordwise panel count. The flap effectiveness of the lattice (the ratio of the flap lift slope to the angle-of-attack lift slope) converges at first order with `n_chord`: on a rectangular wing of aspect ratio 40 with a full-span flap (hinge at 0.75 c), it is about 40/`n_chord` % below Glauert's $	au$. When `n_chord` is automatic (`SolverSettings.n_chord = None`) and a control has a nonzero deflection, the automatic count is at least `CONTROL_MIN_AUTO_N_CHORD` = 16 (`ventorum.solvers.lattice_base`), for an error of about 2.5 %. The ground rule (a chordwise panel not longer than the gap) and the maximum automatic count (`MAX_AUTO_N_CHORD`) stay. Without a deflected control the automatic count does not change. A count that the user sets is used as given: from 2 to 15 chordwise panels with a deflected control, the trust score gets the note "flap effectiveness about 40/n_chord % low (first-order chordwise convergence)" with the number filled in; the note does not change the score. The hinge-node rule above applies to every count: with 16 panels, a hinge that is not on a node still snaps and gives its warning. `tests/test_vlm_auto_n_chord_controls.py` checks the automatic count, the effectiveness against $	au$ and the note.

The adjoint control derivative includes the motion of these points, the hinge nodes, the wake origins and the hinge axes through the node chain; the coefficient of a node angle per unit deflection holds the join factor. The kernels of the gradients carry the extra leg segment. Mean-line camber offsets carry through the adjoint control derivatives. All controls with the same name move together. The width of a join comes from the defined geometry (section chords and reference line); the gradients keep it fixed.

## Lifting line: thin-airfoil flap theory

The lifting line solvers and the single-panel vortex lattice (`n_chord = 1`) use the section mode.

The strip airfoil is replaced by a deflected airfoil derived from the thin-airfoil theory of a hinged flap (Glauert 1926, chapter VII; Glauert 1927). With chord coordinate $x/c = (1 - \cos\theta) / 2$, the hinge at $x/c = E_h$ corresponds to angle:

$$\theta_h = \arccos(1 - 2 E_h)$$

Flap deflection $\delta$ produces changes in lift and pitching moment:

$$C_l = 2 \pi (\alpha + \tau \delta)$$

$$C_{m,c/4} = C_{m0} + \frac{dC_m}{d\delta} \delta$$

The theoretical flap effectiveness factor $\tau$ and moment derivative are:

$$\tau = 1 - \frac{\theta_h - \sin\theta_h}{\pi}$$

$$\frac{dC_m}{d\delta} = -\frac{1}{2} \sin\theta_h (1 - \cos\theta_h)$$

The functions `ventorum.geometry.controls.flap_effectiveness(hinge_x_c)` and `flap_moment_derivative(hinge_x_c)` give the two values. `test_flap_effectiveness_matches_thin_airfoil_table` checks them at five hinge positions.

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

with $q$ the dynamic pressure, $c_{f,i} = (1 - E_h)\, c_i$ the flap chord aft of the hinge and $w_i$ the width of strip $i$. With this mean chord of the flap ($\sum c_f^2 w / \sum c_f w$, the chord weighted by the flap area), $C_h$ is the $c_f^2 w$-weighted mean of the strip values, and a two-dimensional section gives the section coefficient ($H$ per unit span divided by $q\,c_f^2$, as in Glauert 1927). The reference area and chord of the control aft of the hinge follow Etkin and Reid (1996). `ventorum.hinge_moments(result)` gives the hinge moment of each copy of each control of a solved case: the right and left halves of a symmetric surface are separate items.

### Thin-airfoil section hinge moment

Thin-airfoil theory (Glauert 1926, chapter VII; for the flap, Glauert 1927) gives the load of a flat plate at the angle $\alpha$ with a flap deflected by $\delta$. With $x/c = (1 - \cos\theta)/2$, the Fourier coefficients of the load are

$$A_0 = \alpha + \delta\left(1 - \frac{\theta_h}{\pi}\right), \qquad A_n = \frac{2\,\delta\,\sin n\theta_h}{n\,\pi}$$

The section hinge moment is the integral of the load times its arm aft of the hinge, $C_h = -\frac{2}{E^2}\int_{x_h}^{1} \frac{\gamma}{V}\,(x - x_h)\,dx$ with $E = 1 - E_h$. The series of the flap terms sums to a logarithmic load at the hinge, $\frac{\delta}{\pi}\ln\left|\sin\frac{\theta + \theta_h}{2} \big/ \sin\frac{\theta - \theta_h}{2}\right|$. The integrals give $C_h = C_{h\alpha}\,\alpha + C_{h\delta}\,\delta$ with

$$C_{h\alpha} = -\frac{(\pi - \theta_h)\left(\cos\theta_h - \frac{1}{2}\right) + \sin\theta_h\left(1 - \frac{\cos\theta_h}{2}\right)}{E^2}$$

$$C_{h\delta} = \left(1 - \frac{\theta_h}{\pi}\right) C_{h\alpha} - \frac{\sin\theta_h\left[(\pi - \theta_h)\cos\theta_h + \sin\theta_h\right]}{2\pi E^2}$$

The second integral uses an integration by parts, with the derivative $-\sin\theta_h / (\cos\theta_h - \cos\theta)$ of the logarithm. At $E_h \to 0$ both derivatives go to $-\pi/2$ (the moment of a flat plate about its leading edge). At $E_h \to 1$, $C_{h\alpha} \to 0$ and $C_{h\delta} \to -8/(3\pi)$. The expression of $C_{h\alpha}$ is the coefficient $b_1$ of Glauert (1927, equation 14), converted from his coefficients (based on $\rho V^2$) to coefficients based on $q = \tfrac12\rho V^2$. The function `hinge_moment_derivatives(hinge_x_c)` gives the two values. `test_thin_airfoil_hinge_derivatives_equal_quadrature` checks them against a quadrature of the load.

### Section mode

The lifting lines and the single-panel vortex lattice use the section hinge moment on each strip of the control:

$$C_{h,i} = C_{h\alpha}\,\alpha_{fp,i} + C_{h\delta}\,\delta_i, \qquad \alpha_{fp,i} = \alpha_{\text{eff},i} - \alpha_{L0,i} - \tau\,\delta_i$$

Here $\alpha_{L0,i}$ is the zero-lift angle of the deflected strip section and $\delta_i$ the local deflection. In the linear range, $\alpha_{fp,i} = C_{l,i}/a_0 - \tau\,\delta_i$: the flat-plate angle that gives the strip lift together with the flap. The camber of the base section enters only through the strip lift, as the equivalent flat-plate angle $-\alpha_{L0}$ of the base section. The hinge-moment term of the camber line itself is not modelled.

### Chordwise mode

The vortex lattice with `n_chord >= 2` uses the panel loads. The force on the bound vortex of each flap panel is the Kutta-Joukowski force $\rho\,\Gamma\,(\mathbf V_\infty + \mathbf v_\Omega + \mathbf v) \times \mathbf l$ (Katz and Plotkin 2001), with the onset $\mathbf v_\Omega$ of the body rates about the moment reference point (see [Body rates](rotation_rates.md)) and the induced velocity $\mathbf v$ at the force point from all horseshoes, the ground images and the wake of the case. $H$ is the sum of the moments of these forces about the hinge line of each strip. This is the same axis on the mean surface that rotates the flap; it is not interpolated from the deflected trailing edge. The force points and bound segments belong to the deflected geometry. The forces on the chordwise legs are not included: the two legs of one horseshoe carry opposite vorticity over the same chordwise interval, so their moments about the hinge cancel, except for the difference of the induced velocity at the two legs.

The thin-airfoil load is logarithmic at the hinge, so the lattice hinge moment converges slowly with the number of chordwise panels. On a rectangular wing of aspect ratio 40 (hinge 0.75, uniform chordwise panels, $\delta = 5$ deg, $\alpha = 3$ deg, 20 spanwise panels per half), the lattice value approaches the thin-airfoil value at the lift of each strip as the chordwise count increases. `test_vlm_hinge_moment_converges_to_thin_airfoil` in `tests/test_control_derivatives.py` computes this case and checks that the difference decreases.

## Control derivatives

`ventorum.control_derivatives(geometry, condition, settings)` gives, for each control name, the derivatives of $C_L$, $C_D$, $C_Y$, $C_l$, $C_m$ and $C_n$ per unit deflection [1/rad]. For each copy of each control, it gives the hinge moment with its derivatives $\partial C_h/\partial\delta$ (with respect to every differentiated control name; by default all names) and $\partial C_h/\partial\alpha$.

The derivatives are central finite differences of solves at $\delta \pm h$ (default $h = 1$ deg). All controls with one name move together, and the derivative is per unit change of the `deflection` field: the left half of an antisymmetric control moves by $-h$. When $|\delta| + h$ passes the 30 deg limit, the second-order one-sided difference $\left(3C(\delta) - 4C(\delta \mp h) + C(\delta \mp 2h)\right)/(\pm 2h)$ replaces the central difference. $\partial C_h/\partial\alpha$ uses $\alpha \pm 0.5$ deg; in ground effect the moment reference point stays at its height. The moment derivatives are in the body axes (default), or in the stability or wind axes of the base condition, held fixed. The solves with a changed deflection use the same mesh: the hinge snap and the control strips do not depend on the deflection.

The same differences give the derivatives of the far-field diagnostics `CY_farfield` (wind axes) and `Cn_trefftz` (stability axes of the base condition, for every choice of axes); see [Forces and moments](forces_and_moments.md). `to_dict` gives them as `CY_farfield_delta_per_rad` and `Cn_trefftz_delta_per_rad`. The near-field derivatives stay the reported values.

The solves run on the device and in the precision that `ventorum.gpu.set_device` and `ventorum.gpu.set_precision` select, and the result gives the device and the precision. With a 1 deg step, the relative round-off of the float32 GPU paths is about the float32 round-off of the coefficients divided by the step; no test checks this float32 error yet (`test_gpu_hinge_moments_equal_cpu` uses float64).

## Assumptions and limits

The control surface model uses these assumptions:
- Small deflection angles: the formulation assumes $|\delta| \le 30^\circ$.
- No gap between flap and main wing surface.
- No flap-edge vortex model other than the vortex lattice itself.
- No viscous separation effects: profile drag shifts with the effective angle of attack only.
- A snapped panel can be up to 1.25 times its normal length. The ground-effect rule that limits chordwise panel length to the gap height does not see this extension.
- At the outer end of a join, the bound vortices of neighbouring strips can end at different chordwise locations (the join strips have the hinge node of the control, their neighbours do not). On a cambered mean line their legs then do not coincide at the join edge, and the leg forces there do not converge with the spanwise mesh. The lattice warns in that case; use a chordwise count with a node at the hinge. This warning and the warning of a hinge that snaps to a node are stored on the lattice (`VortexLattice.mesh_warnings`): each solve issues them, also with a lattice from the lattice cache, and the trust score lists them (they do not change its value).
- Two adjacent controls with different hinge fractions: the join between them blends the deflection, but each strip keeps the hinge of its own control, so the hinge line jumps at the strip edge nearest to the middle of the join. Each strip turns its edge points about its own hinge point. Where the deflection of that edge is not zero, the two strips have different edge points there: an open slit of about 2e-2 chord at 10 deg (hinges at 0.75 and 0.625 chord). The flap increment of CL then does not converge with the spanwise panels. It converges when the edge deflection is zero (opposite deflections, or a span gap between the controls).
- A control at the end of a surface that is joined to another surface: the join does not continue across the joint, so the end strip has the full deflection and the strip of the other surface has its own (zero for a clean surface). The two strips then have different edge points at the joint, an open slit, and the flap increment does not converge with the spanwise panels. Two joined surfaces with the same hinge and deflection at the joint turn the joint node about one axis and have no slit.
- The join is a sealed transition: it models no side gap and no side-edge vortex of a real flap.
- The aileron side force follows the existing leg rules with the tangential velocity relative to the reference strip normal. This is not a resolved flap-edge flow model. On the rectangular flat-wing test at positive incidence, the near-field side force is positive for right flap down. It is finite and changes sign when the aileron deflection changes sign. The Trefftz plane uses the moved trailing edges. The join at the inboard end of the aileron makes the step of the trailing edge continuous, and the trace then gives a positive far-field side force too (`test_aileron_far_field_side_force_sign`, thin-wing theory). DATCOM (Finck 1978, section 6.2.3) neglects the plain-wing control side force.
- The classical Fourier solver does not support deflected control surfaces.
- The section hinge moment is the thin-airfoil value of a plain flap: no gap, no nose balance, no camber term, attached flow only. Past stall it has no physical meaning.
- The flap effectiveness converges at first order with the number of chordwise panels (see *Vortex lattice: geometric deflection*). The automatic count uses 16 or more chordwise panels with a deflected control; a smaller count that the user sets gets a trust note.
- The chordwise hinge moment converges slowly with the number of chordwise panels (see *Chordwise mode* above). Use 16 or more chordwise panels for hinge moments.

## Verification

`tests/test_geometric_flap.py` checks the moved points and normals, the hinge nodes of the legs, the joins and the root join, wake origins, flap clearance, hinge axis, control gradients, the kernel parity with leg nodes of every backend (numpy reference, Numba, Cython, torch, GPU in float64 and float32), the small-deflection flap slope at incidence against the normal-turn model, the spanwise convergence of full-span and part-span flaps, the leg-force rules and the aileron side-force sign reversal. Its slow comparison with the thin-wing panel method keeps explicit lift and moment limits in free air and ground effect. The rigid full-chord rotation is checked at the geometry-operation level: the public control validator excludes a hinge at the leading edge, and the snap rule selects interior edges only.

The thin-wing panel comparison in ground effect still exceeds its lift bound.

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

- H. Glauert, *The Elements of Aerofoil and Airscrew Theory*, Cambridge University Press, 1926. Chapter VII (thin aerofoil theory).
- H. Glauert, "Theoretical relationships for an aerofoil with hinged flap", Aeronautical Research Committee, R&M 1095, 1927.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001.
- B. Etkin and L. D. Reid, *Dynamics of Flight: Stability and Control*, 3rd ed., Wiley, 1996.
- R. D. Finck, *USAF Stability and Control DATCOM*, McDonnell Douglas Corporation for the Air Force Flight Dynamics Laboratory, Wright-Patterson Air Force Base, revised 1978 (AFWAL-TR-83-3048), section 6.2.3.
