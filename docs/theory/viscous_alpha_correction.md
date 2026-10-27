---
orphan: true
---

# Viscous alpha-correction coupling

This page describes the first viscous variant (V1) of the lattice solvers: an angle-of-attack correction (decambering) that couples the vortex-lattice method or the linear lifting line with 2D section polars. The code is `ventorum.viscous.alpha_correction.solve_viscous`.

## Overview

The inviscid solvers give linear lift curves and no stall. The coupling gives each spanwise strip $j$ one correction angle $\delta_j$ [rad]. The correction changes only the right-hand side of the strip (a uniform change of its angle of attack); the system matrix does not change. The coupling finds the corrections for which the lift of every strip in the lattice equals the lift of its viscous section polar at the effective angle of the strip.

## Published sources

1. J. B. Tseng and C. E. Lan, NASA TP-2828, 1988: alpha-reduction coupling of a vortex lattice with section data.
2. C. P. van Dam, J. C. Vander Kam and P. C. Paris, "Design-Oriented High-Lift Methodology for General Aviation and Civil Transport Aircraft", *Journal of Aircraft*, 38(6), 2001, pp. 1076-1084.
3. S. Gallay and E. Laurendeau, "Nonlinear Generalized Lifting-Line Coupling Algorithms for Pre/Poststall Flows", *AIAA Journal*, 53(7), 2015, pp. 1784-1792.
4. R. Mukherjee and A. Gopalarathnam, "Poststall Prediction of Multiple-Lifting-Surface Configurations Using a Decambering Approach", *Journal of Aircraft*, 43(3), 2006, pp. 660-668 (decambering: the correction is the departure of the section data from the linear model).
5. H. Glauert, "Theoretical Relationships for an Aerofoil with Hinged Flap", ARC R&M 1095, 1927 (thin-airfoil flap shift); H. Glauert, *The Elements of Aerofoil and Airscrew Theory*, Cambridge University Press, 1926 (thin-airfoil theory, flap effectiveness).
6. J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001 (discrete vortex method, method of images).
7. ISO 2533:1975, Standard Atmosphere (dynamic viscosity of air at sea level).

## Inviscid response of the strips

The circulation $\boldsymbol{\Gamma}$ of the lattice satisfies $A \boldsymbol{\Gamma} = \mathbf{b}$. A correction $\delta_k$ of strip $k$ turns the onset direction of the boundary conditions of that strip from $\mathbf{d}(\alpha)$ to $\mathbf{d}(\alpha - \delta_k)$. To first order the right-hand side of the strip changes by $\delta_k \, \partial \mathbf{b} / \partial \delta_k$, with $\partial \mathbf{b}/\partial \delta_k = V_\infty \, \mathbf{n} \cdot \partial \mathbf{d}/\partial \alpha$ (vortex lattice, boundary normals $\mathbf{n}$) or $-V_\infty \, \mathbf{n} \cdot \partial \mathbf{d}/\partial \alpha$ (linear lifting line, whose right-hand side has the opposite sign). The strip lift coefficient $C_{l,\text{inv},j} = 2 \Gamma_j / (V_\infty c_j)$, with $\Gamma_j$ the total circulation of the strip, is therefore linear in the corrections:

$$\mathbf{C}_{l,\text{inv}}(\boldsymbol{\delta}) = \mathbf{C}_{l,\text{inv}}(\mathbf{0}) - D \boldsymbol{\delta}.$$

Column $k$ of $D$ comes from one back-substitution with the LU factors of $A$ for the right-hand side $\partial \mathbf{b}/\partial \delta_k$. A positive correction lowers the lift of the strip.

## Section relation of a lattice strip and the effective angle

Each strip of the lattice behaves like a 2D section of the lattice model, $C_l = a_j (\alpha_e - \alpha_{0,j})$:

- **Vortex-lattice method.** The chordwise bound vortices, the control points and the boundary normals of the strip form a 2D discrete-vortex section (thin-airfoil theory with small angles; Katz and Plotkin 2001). The normals give the panel angles $\theta_k$, which hold the camber (the zero-lift turn of the normals) and the flap (the turn of the panels behind the hinge). Two right-hand sides with one factorization, a unit angle and the panel angles, give the circulations $\Gamma_\alpha$ and $\Gamma_\theta$. Then $a_j = 2 \sum \Gamma_\alpha / c_j$ and $\alpha_{0,j} = -\sum \Gamma_\theta / \sum \Gamma_\alpha$. For a clean thin strip this gives $a_j = 2\pi$ and $\alpha_{0,j} = 0$ to round-off.
- **Linear lifting line.** The section relation is the strip airfoil of the lattice: its $a_0$ and $\alpha_{L0}$. On a strip with a deflected control, $\alpha_{L0}$ already has the Glauert flap shift.

The effective angle of strip $j$, measured from the chord of the clean section, is

$$\alpha_{e,j} = \alpha_{0,j} + \frac{C_{l,\text{inv},j}}{a_j},$$

and the viscous polar is read at the viscous section angle

$$\alpha_{v,j} = \alpha_{e,j} + \delta_j.$$

**Flaps are counted once.** The lattice gives the inviscid flap effect in $C_{l,\text{inv}}$, and $\alpha_{0,j}$ (from the lattice itself) takes it out of the effective angle. The polar of a strip with a deflected control is the flapped polar (a table at the flap deflection, a source with the flap of the control, or a single polar with the Glauert shift). With polars that follow thin-airfoil theory (slope $2\pi$, Glauert flap shift), the lifting-line result equals the inviscid result.

**Ground effect.** When the polar of a strip holds the ground height (a table with a height axis, or a source with ground support), the 2D section of a vortex-lattice strip also has its image vortices in the ground, so the 2D ground effect is not counted twice. A free-air polar is read with the free-air section relation. The height of a strip is the height of its quarter-chord point above the ground divided by its chord. The lifting line has no chordwise extent: its section relation has no 2D ground effect, and the polar gives it.

### Reference of a flapped vortex-lattice strip

The inviscid flap lift of the vortex lattice converges with the chordwise panel count as $1/n_\text{chord}$: on a coarse chordwise mesh the lattice flap has a smaller zero-lift shift than thin-airfoil theory. A flapped polar carries the full flap effect. If the polar were read as it is, the coupling would replace the flap effect of the lattice by the full flap effect of the polar: with thin-airfoil polars the coupled result would equal the mesh-converged inviscid value and be higher than the inviscid lattice at the same panel count by the chordwise discretization error of the lattice flap, which halves when $n_\text{chord}$ doubles.

**Owner decision (for now): the reference of the coupled vortex-lattice result is the inviscid vortex lattice at the same panel count.** The coupling therefore reads the polar relative to the thin-airfoil model of the same section. This follows the decambering idea of Mukherjee and Gopalarathnam (2006): the linear (thin-airfoil) part of the section lift comes from the potential-flow model, and only the departure of the section data from that linear model is a correction. The thin-airfoil model of strip $j$ is

$$C_{l,\text{TA},j}(\alpha) = a_j \left(\alpha - \alpha_{0,\text{TA},j}\right), \qquad \alpha_{0,\text{TA},j} = \alpha_{0,\text{clean},j} - \tau(x_h)\,\phi_j\,\kappa_j,$$

with:

- $a_j$ the lift slope of the lattice section, and $\alpha_{0,\text{clean},j}$ the zero-lift angle of the clean lattice section (the panel normals before the flap turn). Without a mean line, all panels have the strip normal, which holds the camber tilt. A uniform panel angle is an angle of attack, so $\alpha_{0,\text{clean},j}$ is then exact at any panel count, also with images. On a lattice on the section mean line, the panel normals follow the camber, and $\alpha_{0,\text{clean},j}$ is the zero-lift angle of the discrete section with these normals, from the same factorization.
- $\tau(x_h) = 1 - (\theta_h - \sin\theta_h)/\pi$, $\cos\theta_h = 1 - 2 x_h$, the flap effectiveness of thin-airfoil theory (Glauert 1926, 1927) at the hinge $x_h$ [-] of the control, and $\phi_j$ [rad] the flap deflection of the polar of the strip.
- $\kappa_j$ the ground factor of the flap shift: 1 in free air and for a free-air polar. When the polar holds the ground height, $\kappa_j$ is the ratio of the flap shifts of a converged 2D discrete-vortex section of the strip with and without its images in the ground (256 lumped vortices, `GROUND_FLAP_REFERENCE_PANELS`, with a panel edge at the hinge). The ratio removes most of the discretization error of this reference section.

**Thin-airfoil kappa and thick sections.** The ground factor $\kappa_j$ is a thin-airfoil value. For thick sections ($t/c > 0.04$) below $h/c = 0.30$, $\kappa_j$ can differ from thick-section values by more than 2 % (about 0.1 to 0.2 deg of flap shift at 10 to 15 deg deflection), and above 5 % below $h/c \approx 0.12$. When the coupling runs in ground effect with a flap deflection on a strip whose polar comes from a thick section ($t/c > 0.04$) and the smallest strip height $h/c$ is below 0.30, the coupling raises one `RuntimeWarning` per run and records a trust warning. In addition, `_section_flap_shift` checks the ground clearance of the strip trailing edge and refuses cases with the trailing edge at or below the ground ($h \le 0$) with a `ValueError`.

The departure of the polar from its thin-airfoil value, $C_{l,\text{visc},j}(\alpha) - C_{l,\text{TA},j}(\alpha)$, is added to the lift of the lattice section $a_j(\alpha - \alpha_{0,j})$. At the viscous section angle this gives the target lift

$$C_{l,\text{target},j} = C_{l,\text{visc},j}(\alpha_{v,j}) - a_j\left(\alpha_{0,j} - \alpha_{0,\text{TA},j}\right).$$

The constant $a_j(\alpha_{0,j} - \alpha_{0,\text{TA},j})$ is the flap lift that the lattice section lacks against thin-airfoil theory. Properties:

- A polar that equals thin-airfoil theory gives $\boldsymbol{\delta} = 0$ at any chordwise panel count, so the coupled result equals the inviscid lattice at the same panel count (`tests/test_alpha_correction.py` checks full-span and partial-span flaps at $n_\text{chord}$ = 4, 8 and 16 and $\alpha$ = 0 and 4 deg).
- A polar with another flap effect, viscous losses or stall gives the correction of its departure from thin-airfoil theory. This correction does not depend on the chordwise panel count (a test with half of the Glauert flap shift gets $\delta_j = \tau\phi/2$ on every flapped strip at $n_\text{chord}$ = 4 and 16).
- The coarse lattice keeps its flap deficit also at the maximum lift: the strip lift is lower than the polar lift at $\alpha_{v,j}$ by the constant above. The polar statistics of the trust score compare this strip lift with the maximum lift of the polar.
- A strip without a deflected control has $\alpha_{0,\text{TA},j} = \alpha_{0,j}$ and the offset is exactly zero: the results of wings without flaps do not change. The lifting line has no chordwise panels, and its strip airfoil already has the Glauert shift, so $\alpha_{0,\text{TA},j} = \alpha_{0,j}$ there too.
- The hinge and the deflection of the thin-airfoil model are those of the polar, not those of the lattice. A lattice that moves its hinge to a chordwise station, or a swept hinge that turns the section normals by another angle, therefore still gives zero correction with a thin-airfoil polar.
- The mesh-converged comparison is still available: the tests print the coupled CL at $n_\text{chord} = 4$ against the Richardson extrapolation of the inviscid CL at $n_\text{chord}$ = 8 and 16, without an assertion.

`details["alpha0_thin_airfoil"]` holds $\alpha_{0,\text{TA},j}$ [rad] per strip, next to `alpha0_section` ($\alpha_{0,j}$) and `lift_slope_section` ($a_j$).

## Residual and Newton iteration

The residual of strip $j$ is

$$r_j(\boldsymbol{\delta}) = C_{l,\text{inv},j}(\boldsymbol{\delta}) - C_{l,\text{visc},j}(\alpha_{v,j}(\boldsymbol{\delta}), Re_j) + a_j\left(\alpha_{0,j} - \alpha_{0,\text{TA},j}\right).$$

The last term is the constant flap offset of a vortex-lattice strip with a deflected control (zero elsewhere), so it does not change the Jacobian.

With $\partial \alpha_{v,j} / \partial \delta_k = -D_{jk}/a_j + \delta_{jk}$ and the polar slope $m_j = \mathrm{d}C_{l,\text{visc},j}/\mathrm{d}\alpha$ at $\alpha_{v,j}$, $M = \operatorname{diag}(m_j)$, the Jacobian is

$$J = \frac{\partial \mathbf{r}}{\partial \boldsymbol{\delta}} = -\left(I - M \operatorname{diag}(a)^{-1}\right) D - M.$$

The Newton step $\omega J^{-1}(-\mathbf{r})$ has the under-relaxation factor $\omega \in (0, 1]$ (input `relaxation`, default 1) and a backtracking line search on $\frac{1}{2}\|\mathbf{r}\|_2^2$. When no Newton step lowers the residual, a damped fixed-point step $\boldsymbol{\delta} + \omega' (D^{-1}\mathbf{r})$ is tried. The iteration stops when $\max_j |r_j|$ is below `tol` (default $10^{-6}$) or after `max_iter` iterations (default 50).

The Reynolds number of strip $j$ is $Re_j = \rho V_\infty c_j / \mu$, with the density $\rho$ [kg/m^3] and the speed $V_\infty$ [m/s] of the flight condition and the dynamic viscosity $\mu$ = `MU_SL`, the single ISA sea-level constant of the package (`ventorum.aero.parasite`, ISO 2533). The flight condition has no viscosity field.

## Factorizations and sweeps

Each condition assembles the system matrix of its lattice. When the matrix is the same as in the previous condition, its LU factors are used again; otherwise it is factorized. With the default free-stream wake the wake direction, and so the matrix, depends on the angle of attack: a sweep factorizes once per condition. With `wake_alignment="body"` out of ground effect the matrix does not change, and a sweep factorizes once. The result reports the real count (`details["factorization_count"]`, the number of factorizations of the whole call, and `details["factorized"]` per condition). Within one condition, the response matrix $D$, the iteration and the branch search use the same factors.

A sweep runs from the angle nearest to zero upward and from there downward. Each condition starts from the corrections of its converged neighbour.

## Post-stall branches and honesty

A strip with $m_j \le 0$ at the solution has the flag `"post-stall, non-unique"`. Then the solver also runs from a stalled start (every strip $5$ deg above the angle of the Cl maximum of its polar) and from an attached start (every strip $5$ deg below it). Every other converged solution whose CL differs by more than $10^{-3}$ (`BRANCH_CL_DIFFERENCE`) is a branch: `details["branches"]` holds all of them (the first is the main result), `details["non_unique"]` is true, and the trust score gets the penalty `POST_STALL_BRANCH_PENALTY`.

Only converged solutions form branches. When the main iteration does not converge, the result has `converged=False`, a warning and a note, no branch search and no branch verdict.

## Profile drag

The profile drag comes from the polar Cd of each strip at its viscous section angle $\alpha_{v,j}$ and Reynolds number, integrated over the span:

$$C_{D_p} = \frac{1}{S_\text{ref}} \sum_j C_{d,j}(\alpha_{v,j}, Re_j) \, c_j \, \Delta y_j.$$

It is reported as `totals.CDp`, apart from the induced drag `totals.CDi` (Trefftz plane); `totals.CD_total` is their sum.

## Section polars

- `PolarTable`: the polar at the strip Reynolds number and flap deflection [deg] (and the strip height h/c [-] with a height axis). The flap hinge of the table must be the hinge of the control.
- A section polar source (XFOIL or Ventorum-2D): the strip section comes from `WingSection.shape` or from a NACA designation in the name of the section airfoil. A strip without a section shape is refused: there is no default section. The flap is `(hinge_x_c, 0.5, deflection_deg)` of the control (hinge height at half the local thickness). A source with ground support also gets the strip height. Between two sections with different shapes, the two polars are blended linearly.
- A single `TabulatedAirfoil` or `LinearAirfoil`: used on every strip, with the Glauert flap shift on a deflected strip.

## Hardware

The coupling runs on the CPU. A GPU path is planned for a later version.

## Thick-surface panel coupling

`solve_viscous(..., model="panel3d")` matches the pressure-integrated lift of
each reporting strip. It uses the existing body panel partition, including
tip and control closure panels. Its area-weighted strip lift is the lift
reported in `totals.CL`. The vortex-lattice and lifting-line paths keep
their own section models.

The panel coupling uses these published methods:

- M. J. Lighthill, "On Displacement Thickness", *Journal of Fluid
  Mechanics*, 4(4), 1958, pp. 383-392: the normal transpiration equivalent
  of a displacement of the flow boundary.
- L. Morino and C. C. Kuo, "Subsonic Potential Aerodynamics for Complex
  Configurations: A General Theory", *AIAA Journal*, 12(2), 1974,
  pp. 191-197: the interior Dirichlet formulation.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., 2001,
  chapters 11 and 12: source and doublet panels, surface pressure, and
  the Kutta condition. The method of images supplies the ground boundary.
- S. A. Kinnas and C. Y. Hsin, "Boundary Element Method for the Analysis
  of the Unsteady Flow Around Extreme Propeller Geometries", *AIAA
  Journal*, 30(3), 1992: iterative pressure Kutta closure.

### Source response and reuse of the factorization

The interior system is $A\mu=-B\sigma$, with
$\sigma_i=-\mathbf n_i\cdot\mathbf V_i$. A positive strip correction
$\delta_k$ [rad] reduces the onset angle. To first order,

$$\Delta\mathbf V_i=-V_\infty\mathbf u_\alpha\delta_k,
\qquad S_{ik}=V_\infty\mathbf n_i\cdot\mathbf u_\alpha,
\quad i\in\mathcal P_k,$$

where $\mathbf u_\alpha=\cos\beta[-\sin\alpha,0,\cos\alpha]^T$.
The change is applied to both the source boundary condition and the
tangential onset used for pressure. The ground images have the same
source strengths and reflected doublet geometry as the inviscid solver.

During each bounded source-matrix chunk, the assembly also accumulates

$$R_{x,:,k}=\sum_{i\in\mathcal P_k}B_{:,i}\sigma_{\mathrm{unit},i,0},
\qquad
R_{z,:,k}=\sum_{i\in\mathcal P_k}B_{:,i}\sigma_{\mathrm{unit},i,2}.$$

No source matrix is retained and no influence pair is evaluated again
for the strip response. Before the Morino factors are released, the same
LU factors solve $U_x=-A^{-1}R_x$ and $U_z=-A^{-1}R_z$. Then

$$G(\alpha,\beta)=V_\infty\cos\beta
\left(\sin\alpha\,U_x-\cos\alpha\,U_z\right),
\qquad \mu(\delta)=\mu_0+G\delta.$$

For a fixed free-air configuration, $\sum_kG_{:,k}=-\partial\mu/\partial\alpha$.
Every angle of a sweep uses these unit responses without another body
triangular solve. Response arrays count toward the factor pool budget
and are released with their configuration. The shared body source sums
also count toward that budget. As for the inviscid factor pool, a running
configuration may exceed the budget until its condition group finishes.
Only one condition's velocity-response columns are formed at a time.

### The thick section reference

The reporting section plane cuts the actual upper and lower mesh panels.
The resulting polygon is expressed in the clean chord frame. A 2D Morino
solve on this polygon supplies the pressure lift at zero and one degree,
which define $a_{j,2D}$ [1/rad] and $\alpha_{0,j,2D}$ [rad]. This is a
constant-source/doublet Dirichlet solve with the logarithmic 2D Green
function and a semi-infinite wake. It does not use the slope of the
viscous polar, a prescribed thin-airfoil slope, or a finite-span substitute.
Two unit onsets share one section factorization. Section factorizations
are separate from the 3D body factorization count.

Surface gradients use adjacent panel values and inverse-distance node
interpolation. The upper and lower trailing-edge connections are cut.
The same two unit responses give the inviscid section moment as a
quadratic in the sine and cosine of the section angle. If a polar has a
ground-height axis, the section reference includes the corresponding
source and doublet images. A free-air polar uses the free-air reference.

### Pressure residual and Jacobian

The coupled strip equations are

$$r_j=C_{l,p,j}(\delta)-C_{l,\mathrm{visc},j}(\alpha_{v,j})+o_j,
\qquad
\alpha_{v,j}=\alpha_{0,j,2D}+\frac{C_{l,p,j}}{a_{j,2D}}+\delta_j.$$

The flap offset $o_j$ uses the clean thick-section zero-lift angle and
the polar's Glauert flap shift, with the same ground factor as the
lattice coupling. It is zero without a deflected control.

Let $H_{jk}=\partial C_{l,p,j}/\partial x_k$, where $x$ holds the strip
angles and, for pressure Kutta, the wake shifts. The surface gradient
columns of the body response are computed once per condition. The
velocity is affine in $x$, so its exact pressure derivative is

$$\frac{\partial C_{p,i}}{\partial x_k}
=-\frac{2}{V_\infty^2}\mathbf V_i\cdot
\frac{\partial\mathbf V_i}{\partial x_k}.$$

Strip pressure integration gives $H$. If $m_j=dC_{l,\mathrm{visc},j}/d\alpha$,
then the lift rows of the Jacobian are

$$J_{jk}=\left(1-\frac{m_j}{a_{j,2D}}\right)H_{jk}
-m_j I_{jk},$$

where $I$ selects the strip-angle columns. The Newton solve uses a
backtracking line search and a damped step on the inviscid response when
the Newton step fails. At a sharp stall turnover, the attached and
stalled seeds can restart a failed primary iteration within its remaining
iteration budget. Only converged solutions form reported branches.
`recovery_runs` and `branch_search_runs` report the two kinds of attempts.

With `panel_kutta="pressure"`, the body doublets are

$$\mu=\mu_0+G\delta+Z\Delta\mu_w,\qquad Z=-A^{-1}D^w.$$

The Newton residual also contains $C_{p,\mathrm{up}}-C_{p,\mathrm{lo}}$
at each wake strip. Its Jacobian is the difference of the corresponding
pressure derivatives. Thus lift matching and pressure Kutta closure are
solved together. The wake response uses the same body LU and is cached
once per configuration. Corrected wake strengths supply the Trefftz drag.

### Profile drag, moment, and supported execution

Profile drag is $\sum_j S_j C_{d,j}(\alpha_{v,j})/S_\mathrm{ref}$.
The lumped section moment correction is

$$\Delta C_{m,j}=C_{m,\mathrm{visc},j}(\alpha_{v,j})
-C_{m,2D,\mathrm{inv},j}(\alpha_{e,j}),\qquad
\alpha_{e,j}=\alpha_{0,j,2D}+C_{l,p,j}/a_{j,2D}.$$

The strip moment receives this correction about its local span axis.
The area-and-chord integral corrects the total body moments. The pressure
distribution itself keeps its one-variable correction. A polar with the
section lift line gives zero angle correction; matching the inviscid
moment and zero profile drag also reproduces all inviscid loads.

NumPy and Numba CPU backends are supported. A GPU request raises an error.
In ground effect, `auto` selects one direct LU per configuration and
records this choice in the dispatch report. An explicit `krylov` request
raises an error because that mode does not supply the required direct
strip responses. Different ground attitudes can have different wakes and
therefore different factorizations. Free-air sweeps share one body LU.
The viscous call owns its solver for the sweep; it does not share an
inviscid solver from the global cache.

A direct `PanelSolver` call with viscous strips on a half mesh rejects
asymmetric flow. Use a full mesh for that flow. Only symmetric half-mesh
responses or full-mesh responses are stored. The pressure-response cache
applies the same restriction, including when a response is already cached.

The optional `panel_estimate_mesh_error=True` setting raises `ValueError`
with this coupling because its refinement driver uses inviscid panel
solves. Run the mesh study inviscid. Keep this setting at its default value,
`False`, for the coupled solve.
