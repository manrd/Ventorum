# Body rates and rate derivatives

## Model

The aircraft turns with the body angular rates p, q and r about the moment reference point. The solvers model this rotation as quasi-steady: the circulation is that of a steady flow with the instantaneous onset velocity at each point of the aircraft, and the wake stays straight along the free stream. The rate derivatives are the derivatives of the force and moment coefficients per unit nondimensional rate, from central differences of these solves. The vortex lattice and the numerical lifting lines (linear and nonlinear) accept the rates, in free air and in ground effect. The classical (Fourier) lifting line does not.

## Equations

### Rates and onset velocity

The rates are given in the flight-dynamics body axes (x forward, y right, z down): p > 0 right wing down, q > 0 nose up, r > 0 nose right, the signs of Cl, Cm and Cn (see [Conventions](conventions.md)). In the geometry axes (x aft, y right, z up) the angular velocity is

$$
\boldsymbol\Omega = (-p,\; q,\; -r).
$$

A point x fixed to the aircraft moves with the velocity Omega x (x - x_ref) relative to the reference point x_ref. The air at rest then has, relative to that point, the onset velocity

$$
\mathbf U(\mathbf x) = V_\infty\,\hat{\mathbf d} - \boldsymbol\Omega \times (\mathbf x - \mathbf x_{ref}),
$$

with d the free-stream direction. This is the kinematic velocity of a body-fixed frame (Katz and Plotkin 2001). The rotation centre is the moment reference point of the solve, as in AVL.

The nondimensional rates are

$$
\hat p = \frac{p\,b}{2V_\infty},\qquad \hat q = \frac{q\,c}{2V_\infty},\qquad \hat r = \frac{r\,b}{2V_\infty},
$$

with b and c the reference span and chord (Etkin and Reid 1996).

### Vortex lattice

The flow-tangency condition at each control point uses the onset velocity at that point:

$$
\left(\mathbf U(\mathbf x_{cp,i}) + \sum_j \mathbf v_{ij}\,\Gamma_j\right)\cdot \mathbf n_i = 0 .
$$

The Kutta-Joukowski force on each bound vortex and on each on-surface leg uses the onset velocity at its force point: F = rho Gamma (U + v) x l. The circulation is linear in the rates and the forces are quadratic.

### Numerical lifting lines

The section equation of each strip uses the onset at the three-quarter-chord point of the strip, at the span fraction of the control point. In thin-airfoil theory the lift of a normal velocity that changes linearly along the chord equals the lift of its value at the three-quarter chord (the collocation point of the lumped-vortex element, Katz and Plotkin 2001). A pitch rate q about the quarter chord gives the normal velocity q (x - c/4), so the section lift is that of the angle q c / (2 V): the lift of the equal change of the angle of attack.

The linear lifting line of Phillips and Snyder (2000) becomes, with the local onset U_i and its unit vector e_i,

$$
\frac{2\,|\mathbf e_i \times \mathbf{dl}_i|}{a_{0,i}\,dA_i}\,\Gamma_i - \sum_j (\mathbf v_{ij}\cdot \mathbf n_i)\,\Gamma_j
= \mathbf U_i\cdot \mathbf n_i - |\mathbf U_i|\,\alpha_{L0,i}.
$$

With no rates, U_i = V d and the equation is the original one. The nonlinear lifting line uses the local velocity W_i = U_i + sum_j v_ij Gamma_j in the section angle and the residual 2 |W_i x dl_i| Gamma_i - |U_i|^2 dA_i Cl_i(alpha_i).

### Section moment with one chordwise panel

A lifting line, or a vortex lattice with one chordwise panel, has no chordwise load. In thin-airfoil theory, a normal velocity that grows along the chord at the rate k [1/s] acts as a parabolic camber. Glauert's coefficients of the equivalent camber line are A_1 = k c / (2 U) and A_2 = 0, so the moment about the quarter chord is

$$
C_{m,c/4} = \frac{\pi}{4}(A_2 - A_1) = -\frac{\pi}{8}\,\frac{k\,c}{U}.
$$

For a pitch rate, k = q, and the section term is -pi/4 per unit q c / (2 V). The code adds this term to the section moment of each strip, with k = -n . (Omega x e_c) (e_c the chord direction, n the section normal).

This term is the two-dimensional value at every aspect ratio. A vortex lattice with two or more chordwise panels has no section term: its chordwise load gives the three-dimensional value, which is smaller in magnitude by an amount that falls with the aspect ratio, and which it approaches slowly from below in magnitude as the chordwise panel count grows. Thus C_m_q has a step between one and two chordwise panels; `test_vlm_pitch_damping_steps_from_section_term_to_chordwise_load` gives the measured values. Use at least 8 chordwise panels for C_m_q. This is a recommendation from that measurement (a straight wing of aspect ratio 8): 8 is the smallest chordwise panel count of the test that is within 2 % of the value extrapolated from 4, 8 and 16 panels; 4 panels are not.

### Profile drag and section moment

With rates, the profile drag of a strip acts along the local onset at its quarter chord, with the local dynamic pressure, and the section moment uses the same onset. The section lift coefficient that the results report stays 2 Gamma / (V c).

### Ground effect

The ground images stay the mirror images of the vortices, with the opposite circulation. In the earth frame the air and the ground are at rest, so the perturbation velocity of the vortex system must have no normal component on the ground; the rotation of the aircraft enters only the boundary condition on the aircraft. The derivatives are at constant height of the reference point and constant attitude.

### Rate derivatives and axes

`rate_derivatives` uses central differences of step h in each nondimensional body rate. The force coefficients are relative to the free stream in every axis system. In other axes the rates and the moments turn together with the rotation R of [Conventions](conventions.md). The nondimensional values use different reference lengths (b for roll and yaw, c for pitch), so the rotation acts on the dimensional values. With L = diag(b, c, b), the moment block J and the force block F of the derivative matrix become

$$
J' = L^{-1} R\, L\, J\, L\, R^{T} L^{-1},\qquad F' = F\, L\, R^{T} L^{-1}.
$$

For the stability axes (a rotation about y) the reference lengths do not mix, and J' = R J R^T.

### Closed forms for the verification (elliptic wing)

For an elliptic wing of aspect ratio A and section lift slope a_0, Glauert's series of the classical lifting line gives, in stability axes, with mu = a_0 / (pi A) (derivation: the roll rate adds the antisymmetric angle p y / V, and the yaw rate adds the antisymmetric change of the local speed; each forces the sin 2 theta term of the series):

$$
C_{l_p} = -\frac{a_0\,\pi A}{8(\pi A + 2a_0)},\qquad
C_{l_r} = \frac{C_L}{8}\left(1 + \frac{1+\mu}{1+2\mu}\right),
$$

$$
C_{n_p} = -\frac{C_L}{8}\,\frac{1-\mu}{1+2\mu},\qquad
C_{n_r} = -\frac{3\,C_L^2\,(1+\mu)}{8\,\pi A\,(1+2\mu)}.
$$

For large A the first three tend to the strip-theory values -a_0/8, C_L/4 and -C_L/8 of the elliptic wing (Etkin and Reid 1996 give the strip-theory method). C_n_r here has no profile-drag part.

The linear lifting line differs from these closed forms by a term of order alpha^2 (|C_l_p| smaller, |C_n_p| larger). The cause is the linearisation of Phillips and Snyder (2000): the magnitude of the Kutta-Joukowski force omits the induced velocity, and the section angle uses its small-angle form. The nonlinear lifting line with linear sections keeps both and agrees with the closed forms more closely; `test_linear_lifting_line_roll_derivatives_differ_from_closed_forms_by_order_alpha_squared` gives the measured values.

## Assumptions

- Quasi-steady flow: no wake memory of the motion, no added-mass terms.
- Straight wake along the free stream. A solve gets a trust note when the rotation speed on the geometry is above 0.1 V (a local helix angle of about 6 deg).
- Small rates: the derivatives are about the base condition.
- The assumptions of the solver (linear sections for the VLM, attached flow).

## Envelope

- The Fourier solver refuses a rate that is not zero (`ValidityError`).
- The GPU pipelines support body rates for the vortex lattice and the numerical lifting lines (linear and nonlinear), in free air and in ground effect. Solves run on the selected device, and the result states the device. One fallback remains: a nonlinear batch with continuation that mixes symmetric and non-symmetric cases (for example a pitch rate in some cases and a roll rate in others) runs on the CPU, because the continuation needs the cases in order. `rate_derivatives` solves without continuation and runs on the GPU.
- In ground effect the rate derivatives do not include the coupling with the height change (see [Ground effect](ground_effect.md)).
- **Mesh sensitivity of the lateral values.** With body rates, the near-field lateral values (CY, Cl, Cn) of the lattice solvers are sensitive to the spanwise mesh. In a mesh study of a swept wing with a roll rate, Cn changed by about 0.8 % at each doubling of the spanwise panel count, and the cosine and the uniform spacing gave values 5 to 7 % apart. Each lattice result with a body rate gets a trust note (no penalty). Compare two spanwise panel counts or spacings before you use the lateral rate derivatives.

## Implementation

- `ventorum.aero.system.rotation_velocity`, `rotation_vector`, `has_rates`: the onset of the rotation.
- `ventorum.solvers.core`: `assemble_vlm` (right-hand side), `llt_onset`, the linear and nonlinear lifting-line systems.
- `ventorum.gpu.engine`: `vlm_solve` (rotation wash in the right-hand side), `llt_terms` and `loads`; `ventorum.gpu.kernels`: `llt_rhs_rates`, `loads_strip_rates`. A batch without rates keeps the kernels without rates (the host flag `has_rates` of `batch_data` selects them).
- `ventorum.aero.loads.compute_loads_batch`: forces, profile drag and section moment with the local onset; the section-moment term of one chordwise panel.
- `ventorum.aero.hinge`: the chordwise hinge moment with the onset at the flap force points.
- `ventorum.solvers.rate_derivatives`: `rate_derivatives`, `rate_axes_transform`.

## Verification

`tests/test_rate_derivatives.py` checks the closed forms above with the linear lifting line, the three-quarter-chord rule (a pitch rate gives the lift of the equal incidence change), the section moment -pi/4, the equivalence of a roll rate and an antisymmetric linear twist at zero lift in free air and in ground effect, the symmetry of the derivatives, the independence of the VLM derivatives from the step, and the axes transform against solves with rates given in the target axes. `tests/test_gpu_rates.py` checks the parity of the GPU pipelines against the CPU path for VLM, linear and nonlinear lifting line in free air and in ground effect.

## References

- B. Etkin and L. D. Reid, *Dynamics of Flight: Stability and Control*, 3rd ed., Wiley, 1996.
- H. Glauert, *The Elements of Aerofoil and Airscrew Theory*, Cambridge University Press, 1926.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001.
- W. F. Phillips and D. O. Snyder, "Modern adaptation of Prandtl's classic lifting-line theory", *Journal of Aircraft* 37(4), 2000, pp. 662-670.
