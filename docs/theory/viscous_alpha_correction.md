---
orphan: true
---

# Viscous alpha-correction coupling

This page describes the first viscous variant (V1) of the lattice solvers: an angle-of-attack correction (decambering) that couples the vortex-lattice method or the linear lifting line with 2D section polars. The code is `ventorum.viscous.alpha_correction.solve_viscous`.

## Overview

The inviscid solvers give linear lift curves and no stall. The coupling gives each spanwise strip $j$ one correction angle $\delta_j$ [rad]. The correction changes only the right-hand side of the strip (a uniform change of its section angle of attack); the system matrix does not change. The coupling finds the corrections for which the lift of every strip in the lattice equals the lift of its viscous section polar at the effective angle of the strip.

## Published sources

1. J. B. Tseng and C. E. Lan, NASA TP-2828, 1988: alpha-reduction coupling of a vortex lattice with section data.
2. C. P. van Dam, J. C. Vander Kam and P. C. Paris, "Design-Oriented High-Lift Methodology for General Aviation and Civil Transport Aircraft", *Journal of Aircraft*, 38(6), 2001, pp. 1076-1084.
3. S. Gallay and E. Laurendeau, "Nonlinear Generalized Lifting-Line Coupling Algorithms for Pre/Poststall Flows", *AIAA Journal*, 53(7), 2015, pp. 1784-1792.
4. R. Mukherjee and A. Gopalarathnam, "Poststall Prediction of Multiple-Lifting-Surface Configurations Using a Decambering Approach", *Journal of Aircraft*, 43(3), 2006, pp. 660-668 (decambering: the correction is the departure of the section data from the linear model).
5. H. Glauert, "Theoretical Relationships for an Aerofoil with Hinged Flap", ARC R&M 1095, 1927 (thin-airfoil flap shift); H. Glauert, *The Elements of Aerofoil and Airscrew Theory*, Cambridge University Press, 1926 (thin-airfoil theory, flap effectiveness).
6. J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001 (discrete vortex method, method of images).
7. ISO 2533:1975, Standard Atmosphere (dynamic viscosity of air at sea level).
8. C. Coulliette and A. Plotkin, ["Aerofoil ground effect revisited"](https://www.cambridge.org/core/journals/aeronautical-journal/article/abs/aerofoil-ground-effectrevisited/561E5C4DD13ED87435EF7B438859EE20), *The Aeronautical Journal* 100, 1996 (the surface pressure lift of a section in ground effect, eq. 30, and its difference from the circulation lift).
9. R. P. Brent, *Algorithms for Minimization without Derivatives*, Prentice-Hall, 1973, chapter 4 (bracketed root finder; the inversion of the 2D ground pitch curve of the thick-surface coupling).

## Section plane of a strip

The section plane of strip $j$ holds its chord direction $\mathbf{c}_j$ and its normal $\mathbf{n}_j$. Its span axis is the unit vector $\mathbf{e}_j = \mathbf{n}_j \times \mathbf{c}_j$, normal to the section plane: $+y$ for a planar strip, $(0, \cos\Gamma, \sin\Gamma)$ for a strip with the dihedral $\Gamma$, $+z$ for a vertical fin. The 2D section relation of a strip (below) is a 2D problem in this plane, and its section angle of attack is the angle of the in-plane onset from the chord towards the normal.

The unit free-stream direction $\mathbf{d}(\alpha, \beta)$ has the in-plane part $\mathbf{d}_{p,j} = \mathbf{d} - (\mathbf{d} \cdot \mathbf{e}_j)\,\mathbf{e}_j$. A change of the section angle turns $\mathbf{d}_{p,j}$ about $\mathbf{e}_j$; the spanwise part does not change. Per radian the onset changes by

$$\mathbf{u}_j = \mathbf{d}_{p,j} \times \mathbf{e}_j .$$

The norm of $\mathbf{u}_j$ is that of $\mathbf{d}_{p,j}$, and $\mathbf{u}_j$ lies in the section plane, normal to the in-plane onset: $\mathbf{l}_j = \mathbf{u}_j / |\mathbf{u}_j|$ is the section lift direction. For a planar strip, $\mathbf{e}_j = +y$ and $\mathbf{u}_j = \partial \mathbf{d}/\partial \alpha = \cos\beta\,(-\sin\alpha, 0, \cos\alpha)$: the correction is a change of the angle of attack, as before. For a strip with dihedral, $\mathbf{n}_j \cdot \partial \mathbf{d}/\partial \alpha$ is only $\cos\Gamma$ times $\mathbf{n}_j \cdot \mathbf{u}_j$, and on a vertical fin it is zero. A correction about the global $y$ axis would therefore change the section angle by $\delta_j \cos\Gamma$ instead of $\delta_j$, and a fin would keep its inviscid side force whatever its polar. With $\mathbf{u}_j$, a rigid rotation of a configuration together with its onset does not change the corrections or the loads (`tests/test_alpha_correction_section_frame.py`).

## Inviscid response of the strips

The circulation $\boldsymbol{\Gamma}$ of the lattice satisfies $A \boldsymbol{\Gamma} = \mathbf{b}$. A correction $\delta_k$ of strip $k$ turns the in-plane onset of the boundary conditions of that strip by $-\delta_k$ about its span axis: the onset direction changes by $-\delta_k \mathbf{u}_k$. To first order the right-hand side of the strip changes by $\delta_k \, \partial \mathbf{b} / \partial \delta_k$, with $\partial \mathbf{b}/\partial \delta_k = V_\infty \, \mathbf{n} \cdot \mathbf{u}_k$ (vortex lattice, boundary normals $\mathbf{n}$) or $-V_\infty \, \mathbf{n} \cdot \mathbf{u}_k$ (linear lifting line, whose right-hand side has the opposite sign). The strip lift coefficient $C_{l,\text{inv},j} = 2 \Gamma_j / (V_\infty c_j)$, with $\Gamma_j$ the total circulation of the strip, is therefore linear in the corrections:

$$\mathbf{C}_{l,\text{inv}}(\boldsymbol{\delta}) = \mathbf{C}_{l,\text{inv}}(\mathbf{0}) - D \boldsymbol{\delta}.$$

Column $k$ of $D$ comes from one back-substitution with the LU factors of $A$ for the right-hand side $\partial \mathbf{b}/\partial \delta_k$. A positive correction lowers the lift of the strip.

## Section relation of a lattice strip and the effective angle

Each strip of the lattice behaves like a 2D section of the lattice model in its section plane, $C_l = a_j (\alpha_e - \alpha_{0,j})$:

- **Vortex-lattice method.** The chordwise bound vortices, the control points and the boundary normals of the strip form a 2D discrete-vortex section (thin-airfoil theory with small angles; Katz and Plotkin 2001). The normals give the panel angles $\theta_k$, which hold the camber (the zero-lift turn of the normals) and the flap (the turn of the panels behind the hinge). Two right-hand sides with one factorization, a unit angle and the panel angles, give the circulations $\Gamma_\alpha$ and $\Gamma_\theta$. Then $a_j = 2 \sum \Gamma_\alpha / c_j$ and $\alpha_{0,j} = -\sum \Gamma_\theta / \sum \Gamma_\alpha$. For a clean thin strip this gives $a_j = 2\pi$ and $\alpha_{0,j} = 0$ to round-off.
- **Linear lifting line.** The section relation is the strip airfoil of the lattice: its $a_0$ and $\alpha_{L0}$. On a strip with a deflected control, $\alpha_{L0}$ already has the Glauert flap shift.

The effective angle of strip $j$, measured from the chord of the clean section, is

$$\alpha_{e,j} = \alpha_{0,j} + \frac{C_{l,\text{inv},j}}{a_j},$$

and the viscous polar is read at the viscous section angle

$$\alpha_{v,j} = \alpha_{e,j} + \delta_j.$$

**Flaps are counted once.** The lattice gives the inviscid flap effect in $C_{l,\text{inv}}$, and $\alpha_{0,j}$ (from the lattice itself) takes it out of the effective angle. The polar of a strip with a deflected control is the flapped polar (a table at the flap deflection, a source with the flap of the control, or a single polar with the Glauert shift). With polars that follow thin-airfoil theory (slope $2\pi$, Glauert flap shift), the lifting-line result equals the inviscid result.

**Ground effect.** When the polar of a strip holds the ground height (a table with a height axis, or a source with ground support), the 2D section of a vortex-lattice strip also has its image vortices in the ground, so the 2D ground effect is not counted twice. The images are in the ground line of the section plane: the plane cuts the ground in a line, and a point at the height $h$ above the ground has the distance $h/\cos\Gamma_g$ from that line in the section plane, where $\Gamma_g$ is the angle of the span axis $\mathbf{e}_j$ from the ground plane (the dihedral of the strip relative to the ground). Its image is at $2h/\cos\Gamma_g$, on the other side of the line. The thick-surface coupling uses the same ground line (see "The thick section reference" below). The 3D mirror image would be at only $2h\cos\Gamma_g$ in the section plane, the image of a strip at the height $h\cos^2\Gamma_g$. When the span axis lies in the ground plane, the two images are the same. A section plane that holds the ground normal has no ground line, and the coupling refuses it with a `ValueError`. A free-air polar is read with the free-air section relation. Every coupling (vortex lattice, lifting line and thick surface) reads the ground polar of a strip at the same section-plane distance: $h/(c_j\cos\Gamma_g)$, with $h$ the height of the quarter-chord point above the ground (normal to the ground). The section relation and the polar then see the same ground distance. A planar strip ($\cos\Gamma_g = 1$) reads at $h/c_j$ with no arithmetic change. `details["strip_h_over_c"]` keeps the vertical height $h/c_j$. A check on a NACA 2412 wing (AR 8, dihedral 20 deg, $lpha$ = 4 deg, $h/c$ = 0.5 and 0.25, Ventorum-2D ground polars) moved the vortex-lattice CL towards the CL of the thick-surface coupling at two meshes of each model; the change was smaller than the mesh error of the thick-surface result. The lifting line has no chordwise extent: its section relation has no 2D ground effect, and the polar gives it.

**Steep strips.** A 2D ground polar describes a section above a parallel ground. A strip whose span axis is steeper than 60 deg to the ground plane (`STEEP_STRIP_LIMIT_DEG`; a fin, a strongly bent tip) is not such a section: near 90 deg its section plane holds the ground normal and has no ground line. Such a strip therefore uses free-air section data in every coupling (vortex lattice, lifting line and thick surface): a polar source with ground support runs without the height, its section relation has no images, and the result gets a trust note with the number of these strips. A free-air polar for the surface in the polar dict is read as usual. A polar table with a height axis has no free-air data, so a steep strip that only has a ground table is refused with a `ValueError` that names the surface and its angle and asks for a free-air polar for it. The limit of 60 deg is a code setting, not a value from a source. Strips at 60 deg or less keep the ground relation above.

The ground plane of a condition is parallel to the free stream, so in the section frame it is tilted by the attitude of the strip. A polar with a height axis is a pitch curve: the section pitches about its quarter chord at a fixed height, with the onset parallel to the ground. The vortex-lattice section is linear in the angle (small angles) and keeps the image positions of the attitude. Its line therefore has the value of the section at its own attitude, but its slope is the slope at a fixed ground line, not the slope of the pitch curve. An identity polar from a converged flat-plate pitch curve thus leaves a small correction that is proportional to the downwash of the strip; it converges to a small value when the mesh is refined. The thick-surface coupling inverts the pitch curve itself (see "The thick section reference" below).

### Lift definitions in ground effect

A 2D section near a wall has two lift coefficients. The circulation lift is $C_{l,\Gamma} = 2\Gamma/(V_\infty c)$, from the total circulation $\Gamma$ of the section. The pressure lift $C_{l,p}$ is the integral of the surface pressure: it is the force on the section. In free air the Kutta-Joukowski relation makes the two equal for a single section. Near a wall it does not give the force from the total circulation, and the two differ (Coulliette and Plotkin 1996, eq. 30). Ventorum-2D reports $C_{l,p}$ as `Cl` in ground effect and keeps $C_{l,\Gamma}$ as `Cl_circulation` (see [](panel2d.md)).

**The 2D lift coefficient of a polar.** The lift of a section polar has one of two definitions:

- A free-air polar gives the circulation lift as `Cl`. In free air the circulation lift and the pressure lift are equal, so this `Cl` is also the pressure lift.
- A ground polar (a polar with the ground height) gives the pressure lift as `Cl` and, when its source has it, the circulation lift as `Cl_circulation`.
- The vortex-lattice and lifting-line couplings use the circulation lift in ground effect: they read `Cl_circulation` when the ground polar has it, and `Cl` with a trust note when it does not. In free air they read `Cl`. The thick-surface coupling (`model="panel3d"`) reads `Cl`, the pressure lift, also in ground effect (see the list below).

The coupling uses these definitions:

- **Residual: circulation with circulation.** The strip lift $C_{l,\text{inv},j} = 2\Gamma_j/(V_\infty c_j)$ of the lattice is a circulation lift. The section relation $(a_j, \alpha_{0,j})$ of a vortex-lattice strip with images, and the ground factor $\kappa_j$ of the flap shift, also come from sums of circulations. On a strip whose polar holds the ground height, the residual therefore reads the circulation lift of the polar, $C_{l,\Gamma,\text{visc}}$, and its slope in the Jacobian. A ground polar of Ventorum-2D has it: `Ventorum2DPolarSource` stores it in `TabulatedAirfoil.Cl_circulation_data`, `build_polar_table` stores it in `PolarTable.Cl_circulation`, and `PolarTable.at` interpolates it as `Cl`. The branch search uses the angle of the largest circulation lift. `details["strip_circulation_lift"]` marks these strips.
- **The ground factor is consistent.** $\kappa_j$ is the ratio of two zero-lift shifts of a 2D discrete-vortex section, and each shift is a ratio of circulation sums. It is thus a circulation quantity, and it agrees with a polar that gives the circulation lift. No change of $\kappa_j$ is necessary for this definition.
- **Reported loads: the force.** The loads of the result are the Kutta-Joukowski forces on the bound vortices with the local velocity, the velocity of the ground images included ([](forces_and_moments.md)). This is the force on the wing, the 3D equivalent of the pressure lift. The polar `Cl` (the pressure lift in ground effect) stays in the polar for reports.
- **Polar without circulation lift.** A ground polar without circulation lift data (for example a user table with a height axis) is read with its `Cl`, as before. The result then gets a trust note and one `RuntimeWarning` per call: when that `Cl` is a pressure lift, the residual compares two lift definitions, and the coupled lift is too low at small $h/c$.
- **Free air.** A free-air polar is read with `Cl`. A free-air polar of Ventorum-2D has no circulation lift data, because there the two definitions are equal. Free-air results do not change.
- **Thick-surface panel coupling.** `model="panel3d"` matches the pressure lift of each strip with the polar `Cl`: pressure with pressure. It does not read the circulation lift.

**Assumption: the pitch of the polar section.** A ground-effect polar is a set of 2D solutions at one height $h/c$ of the reference point. Ventorum-2D turns the section about its quarter-chord point on the chord line and keeps the height of that point. The coupling reads the polar at the section-plane height of the strip quarter-chord point and at the viscous section angle $\alpha_{v,j}$. So it assumes that the strip section has the attitude of the polar section: pitched about $c/4$ to $\alpha_{v,j}$. The lattice strip has its geometric attitude, and $\alpha_{v,j}$ differs from it by the induced angle and the correction $\delta_j$. Near the ground the height of the trailing edge, and so the 2D ground effect, therefore differ between the strip and the polar section. The vortex-lattice and lifting-line couplings have no correction for this difference; the thick-surface coupling reads its effective angle on the pitch curve of the strip section (see "The thick section reference" below).

**Statement of the assumption and its measured size.** A ground polar pitches the section about its quarter chord at the section angle $\alpha_{v,j}$, with the onset parallel to the ground. It does not keep the strip at its attitude. The vortex-lattice and lifting-line couplings accept this difference. One case gives its size: a planar AR 8 flat-plate wing at $\alpha$ = 4 deg (the attitude of its strips). For its most loaded strip, the 2D flat-plate lift at the effective angle (about 3.9 deg) with the ground line of the strip attitude differs from the lift of the pitched section by $-0.089$ % at $h/c = 0.5$ and by $-0.041$ % at $h/c = 0.25$ (`test_size_of_the_polar_attitude_assumption` in `tests/test_alpha_correction_ground_reference.py`, which pins these values). The difference grows with the difference between $\alpha_{v,j}$ and the attitude (large corrections $\delta_j$, for example near stall); these cases are not measured.

### Reference of a flapped vortex-lattice strip

The inviscid flap lift of the vortex lattice converges with the chordwise panel count as $1/n_\text{chord}$: on a coarse chordwise mesh the lattice flap has a smaller zero-lift shift than thin-airfoil theory. A flapped polar carries the full flap effect. If the polar were read as it is, the coupling would replace the flap effect of the lattice by the full flap effect of the polar: with thin-airfoil polars the coupled result would equal the mesh-converged inviscid value and be higher than the inviscid lattice at the same panel count by the chordwise discretization error of the lattice flap, which halves when $n_\text{chord}$ doubles.

**Owner decision (for now): the reference of the coupled vortex-lattice result is the inviscid vortex lattice at the same panel count.** The coupling therefore reads the polar relative to the thin-airfoil model of the same section. This follows the decambering idea of Mukherjee and Gopalarathnam (2006): the linear (thin-airfoil) part of the section lift comes from the potential-flow model, and only the departure of the section data from that linear model is a correction. The thin-airfoil model of strip $j$ is

$$C_{l,\text{TA},j}(\alpha) = a_j \left(\alpha - \alpha_{0,\text{TA},j}\right), \qquad \alpha_{0,\text{TA},j} = \alpha_{0,\text{clean},j} - \tau(x_h)\,\phi_j\,\kappa_j,$$

with:

- $a_j$ the lift slope of the lattice section, and $\alpha_{0,\text{clean},j}$ the zero-lift angle of the clean lattice section (the panel normals before the flap turn). Without a mean line, all panels have the strip normal, which holds the camber tilt. A uniform panel angle is an angle of attack, so $\alpha_{0,\text{clean},j}$ is then exact at any panel count, also with images. On a lattice on the section mean line, the panel normals follow the camber, and $\alpha_{0,\text{clean},j}$ is the zero-lift angle of the discrete section with these normals, from the same factorization.
- $\tau(x_h) = 1 - (\theta_h - \sin\theta_h)/\pi$, $\cos\theta_h = 1 - 2 x_h$, the flap effectiveness of thin-airfoil theory (Glauert 1926, 1927) at the hinge $x_h$ [-] of the control, and $\phi_j$ [rad] the flap deflection of the polar of the strip.
- $\kappa_j$ the ground factor of the flap shift: 1 in free air and for a free-air polar. When the polar holds the ground height, $\kappa_j$ is the ratio of the flap shifts of a converged 2D discrete-vortex section of the strip with and without its images in the ground (256 lumped vortices, `GROUND_FLAP_REFERENCE_PANELS`, with a panel edge at the hinge). The reference section is on the undeflected chord line of the strip (the chord direction and the chord before the flap turn); the flap is only in its panel angles, as in the lattice section. So $\kappa_j$ does not depend on the flap deflection, and it is the same ground factor for every deflection of the polar. The ratio removes most of the discretization error of this reference section.

**Thin-airfoil kappa and thick sections.** The ground factor $\kappa_j$ is a thin-airfoil value. For thick sections ($t/c > 0.04$) below $h/c = 0.30$, $\kappa_j$ can differ from thick-section values by more than 2 % (about 0.1 to 0.2 deg of flap shift at 10 to 15 deg deflection), and above 5 % below $h/c \approx 0.12$. When the coupling runs in ground effect with a flap deflection on a strip whose polar comes from a thick section ($t/c > 0.04$) and the smallest strip height $h/c$ is below 0.30, the coupling raises one `RuntimeWarning` per run and records a trust warning. In addition, `_section_flap_shift` checks the ground clearance of the trailing edge of the reference section and of the deflected trailing edge of the strip, and refuses cases with one of them at or below the ground ($h \le 0$) with a `ValueError`.

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

The last term is the constant flap offset of a vortex-lattice strip with a deflected control (zero elsewhere), so it does not change the Jacobian. On a strip whose polar holds the ground height and has circulation lift data, $C_{l,\text{visc},j}$ is the circulation lift of the polar (see "Lift definitions in ground effect").

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

**Policy when the coupling does not converge.** The coupling does not raise an error and does not fall back to the inviscid result. It returns the last iterate: the corrections $\boldsymbol{\delta}$ after the last accepted Newton or damped fixed-point step. Each accepted step lowers $\frac{1}{2}\|\mathbf{r}\|_2^2$, so the last iterate has the smallest residual norm of the iteration. The iteration stops after `max_iter` iterations, or earlier when no step lowers the residual. The result then has:

- `converged=False`, and the residual history in `details["viscous_residual_history"]`;
- one `RuntimeWarning` and a note in `details["notes"]` that give the iteration count, the largest strip residual and the tolerance;
- the trust penalty of a solver that did not converge (`evaluate_aerodynamic_trust` with `converged=False`), with its warning in the trust report;
- no branch search and no branch verdict.

The thick-surface coupling first tries the stalled and the attached starts within the remaining iterations when a strip of the last iterate is post-stall; if one of them converges, it is the result. In a sweep, a condition that did not converge does not give the start of the next condition. This is the policy of all V1 couplings (vortex lattice, lifting line and thick surface): use such a result only with its warning.

## Profile drag

The profile drag comes from the polar Cd of each strip at its viscous section angle $\alpha_{v,j}$ and Reynolds number, integrated over the span:

$$C_{D_p} = \frac{1}{S_\text{ref}} \sum_j C_{d,j}(\alpha_{v,j}, Re_j) \, c_j \, \Delta y_j.$$

It is reported as `totals.CDp`, apart from the induced drag `totals.CDi` (Trefftz plane); `totals.CD_total` is their sum.

## Section polars

- `PolarTable`: the polar at the strip Reynolds number and flap deflection [deg] (and the strip height h/c [-] with a height axis). The flap hinge of the table must be the hinge of the control.
- A section polar source (XFOIL or Ventorum-2D): the strip section comes from `WingSection.shape` or from a NACA designation in the name of the section airfoil. A strip without a section shape is refused: there is no default section. The flap is `(hinge_x_c, 0.5, deflection_deg)` of the control (hinge height at half the local thickness). A source with ground support also gets the strip height. Between two sections with different shapes, the two polars are blended linearly.
- A single `TabulatedAirfoil` or `LinearAirfoil`: used on every strip, with the Glauert flap shift on a deflected strip.
- A dict: maps a strip index, a surface name, a section airfoil name or `"default"` to one of the polars above. A key that matches none of them (for example a misspelt surface name) raises a `ValueError` that lists the valid keys. A strip without an entry uses the airfoil of its wing section; the result then has a trust note (and the lattice paths a `RuntimeWarning`).

**Gaps of a section polar.** A section polar source drops the angles that do not converge, and the polar interpolates across them (the gap mask of the polar, see [](ventorum_2d.md)). When a polar of the condition has a bridged interval, the result gets a trust note with the number of strips that read their polar inside a bridged interval at the solution, the number of polars with gaps and the number of dropped angles (`last_dropped` of the sources and tables). The polar also gives one `RuntimeWarning` when the coupling reads it inside a gap. A strip with two bracketing sections counts when one of its two polars has a gap at its angle. The note has no penalty in the trust score. The polars that the coupling makes from other polars keep the gaps: the interpolation of a `PolarTable` at one operating point (`PolarTable.at`) marks each interval that bridges a dropped (NaN) point, a flap deflection (`deflected_airfoil`) keeps the mask of its base polar, and a blend of two polars (`blend_tabulated`) marks an interval when a polar with a weight above 0 bridges it.

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
$\delta_k$ [rad] reduces the section angle of strip $k$. To first order,

$$\Delta\mathbf V_i=-V_\infty\mathbf u_k\delta_k,
\qquad S_{ik}=V_\infty\mathbf n_i\cdot\mathbf u_k,
\quad i\in\mathcal P_k,$$

where $\mathbf u_k$ is the onset change per unit section angle of the
reporting section of strip $k$ (see "Section plane of a strip"). For a
planar strip $\mathbf u_k=\cos\beta[-\sin\alpha,0,\cos\alpha]^T$.
The change is applied to both the source boundary condition and the
tangential onset used for pressure. On a half mesh the mirrored panels
get the mirrored change. The ground images have the same source
strengths and reflected doublet geometry as the inviscid solver.

During each bounded source-matrix chunk, the assembly also accumulates

$$R_{x,:,k}=\sum_{i\in\mathcal P_k}B_{:,i}\sigma_{\mathrm{unit},i,0},
\qquad
R_{z,:,k}=\sum_{i\in\mathcal P_k}B_{:,i}\sigma_{\mathrm{unit},i,2},$$

and, when a span axis is not $+y$ (dihedral, fins), also
$R_{y,:,k}=\sum_{i\in\mathcal P_k}B_{:,i}\sigma_{\mathrm{unit},i,1}$.

No source matrix is retained and no influence pair is evaluated again
for the strip response. Before the Morino factors are released, the same
LU factors solve $U_x=-A^{-1}R_x$, $U_z=-A^{-1}R_z$ (and
$U_y=-A^{-1}R_y$). Then column $k$ of the doublet response is

$$G_{:,k}=-V_\infty\left(u_{k,x}U_{x,:,k}+u_{k,y}U_{y,:,k}
+u_{k,z}U_{z,:,k}\right),
\qquad \mu(\delta)=\mu_0+G\delta.$$

For planar strips this is
$G=V_\infty\cos\beta\left(\sin\alpha\,U_x-\cos\alpha\,U_z\right)$.

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

The line and the lift curve of the cut polygon are properties of the
mesh. Their slope differs from the slope of the converged section by a
discretization error that falls about in proportion to the panel size
(first order in the mesh). The 3D strip lift has the same error. If the
polar were read directly, an identity polar (the polar equals the
converged inviscid 2D section) would not give zero correction on a
finite mesh: it would replace the mesh lift of the section by the
converged lift.

**Owner decision: the reference of the coupled 3D panel result is the
inviscid 3D panel result at the same mesh**, as for the vortex-lattice
coupling (see "Reference of a flapped vortex-lattice strip"). The
coupling therefore reads the departure of the polar from the converged
inviscid 2D section and adds it to the same-mesh 2D section (a defect
correction; the decambering idea of Mukherjee and Gopalarathnam 2006).
The target lift of strip $j$ is

$$c_{l,\mathrm{target},j}(\alpha_v)=C_{l,\mathrm{visc},j}(\alpha_v)
-C_{l,\mathrm{conv},j}(\alpha_v)+C_{l,\mathrm{mesh},j}(\alpha_v).$$

$C_{l,\mathrm{mesh},j}$ is the 2D Morino lift curve of the cut polygon
(the pitch curve with images when the polar holds the ground height).
$C_{l,\mathrm{conv},j}$ is the converged inviscid 2D lift of the same
cut section: the section plane cuts a mesh of the same surfaces (same
spanwise mesh, controls and root treatment) with
`SECTION_REFERENCE_PANELS` (128) cosine-spaced chordwise panels per
side, and Ventorum-2D (`ventorum.panel2d.solver.solve_inviscid`,
linear-strength vortex panels, second order in the panel size) solves
this polygon with the unit chord and the pivot and moment reference at
the quarter chord. Its `Cl` is the lift of a Ventorum-2D polar: the
circulation lift in free air, the pressure lift in ground effect, where
the section pitches about the quarter chord at the height of the strip
section (the same pitch curve as the same-mesh reference). The solves
are at knots every `SECTION_REFERENCE_STEP` (0.5 deg) when they are first
needed, joined by a cubic Hermite interpolation with central-difference
knot slopes, so the curve and its slope are continuous. Outside the knots
at which the section clears the ground by 0.01 chord (and at most
60 deg), the curve is the tangent continuation. 128 panels per side give
$C_l$ within $10^{-4}$ of 256 panels per side, and the interpolation is
within $10^{-5}$ of a direct solve (NACA 0012, 10.25 deg, free air and
$h/c$ = 0.5; `tests/test_panel3d_v1_ground_reference.py` checks both).

A polar that equals the converged inviscid 2D section then gives zero
correction on any mesh, up to the difference between that polar and the
reference of the coupling. The tests check a CL correction below 0.1 %
at 72, 264 and 1056 panels in ground effect and at 8 and 40 chordwise
panels in free air, with the Ventorum-2D curves of a 100-point NACA 0012
in a 0.5 deg table as the polar. A polar that is the same-mesh section
itself now gets the mesh-error correction: its lift is that of the
coarse cut section, lower than the converged lift, so the coupled CL is
lower than the inviscid 3D panel CL by about the section mesh error.
The details `cl_polar`, `cl_section_converged` and `cl_section_mesh`
give the three parts of the target lift at $\alpha_{v,j}$, and
`cl_viscous` gives the target lift.

Surface gradients use adjacent panel values and least-squares node
values. The upper and lower trailing-edge connections are cut.
The same two unit responses give the inviscid section moment as a
quadratic in the sine and cosine of the section angle. If a polar has a
ground-height axis, the section reference includes the corresponding
source and doublet images. A free-air polar uses the free-air reference.

**Attitude of the ground reference.** The ground line of the section
plane is tilted by the attitude $\theta_j$ of the strip: the onset is
parallel to the ground at the angle $\theta_j$ from the chord. The
section line must not come from the onsets 0 and 1 deg with this tilted
ground line: those onsets cross the ground, and the line does not give
the section at its own attitude (its value there was too high, the more
so near the ground and at high attitude). The reference is the pitch
curve $C_l(\alpha; h)$ of a polar with a height axis: the section
pitches about its quarter chord at the fixed quarter-chord height $h$,
with its own ground line and an onset parallel to it. Three 2D solves
at $\theta_j-\varepsilon$, $\theta_j$ and $\theta_j+\varepsilon$
($\varepsilon$ = `GROUND_SECTION_STEP`, 0.5 deg) give the tangent at
the attitude,

$$a_{j,2D}=\frac{C_l(\theta_j+\varepsilon)-C_l(\theta_j-\varepsilon)}{2\varepsilon},
\qquad
\alpha_{0,j,2D}=\theta_j-\frac{C_l(\theta_j)}{a_{j,2D}},$$

and the moment quadratic $A+B\cos2\alpha+C\sin2\alpha$ through the
three pitch-curve moments. The tangent is exact at the attitude. Away
from it, its error grows with the square of $\alpha_{e,j}-\theta_j$
and with the curvature of the pitch curve, which is large near the
ground. Strips with a large downwash (near the tips) have their
effective angle far from the attitude. With the tangent, an identity
polar therefore left a local correction there that did not fall with
the mesh.

**Inversion of the section curve.** The effective angle of a strip does
not come from the tangent or the line. It is the angle $\alpha_{e,j}$
at which the same-mesh lift curve of the cut section has the strip lift,

$$C_{l,\mathrm{mesh},j}(\alpha_{e,j}) = c_{l,j}.$$

In ground effect this is the pitch curve $C_l(\alpha; h_j)$, with one
2D Morino solve with images per attitude (the same solve that gives the
three points of the tangent). In free air one Morino factorization gives
the lift at any onset angle and its exact derivative. A bracketed root
finder (Brent 1973) solves this equation to the round-off of the curve
(absolute tolerance $10^{-15}$ rad). The inversion of the
tangent, $\alpha_{0,j,2D}+c_{l,j}/a_{j,2D}$, is the start value; the
bracket grows from it in steps that double. The Newton Jacobian uses
$\partial\alpha_{e,j}/\partial c_{l,j}=1/C_l'(\alpha_{e,j})$, with a
central difference of $10^{-5}$ rad on the pitch curve
(`GROUND_CURVE_STEP`) and the exact derivative in free air. The search stays at the attitudes between $-90$
and $+90$ deg at which the cut section clears the ground. A lift outside
the range of these attitudes gets the tangent continuation from the
last attitude of the range and a trust note. The reported section line
(`lift_slope_section`, `alpha0_section`) and the moment quadratic stay
those of the tangent; the inviscid section moment is read at the
inverted angle.

With the tangent, the converged identity polar left a local
correction at the tip strips that did not fall with the mesh; with the
inversion and the converged reference, the correction is zero up to the
difference between the polar and the reference
(`tests/test_panel3d_v1_ground_reference.py`).

### Pressure residual and Jacobian

The strip lift is the pressure force of the strip panels along the
section lift direction $\mathbf l_j$ (see "Section plane of a strip"),

$$C_{l,p,j}=-\frac{1}{S_j}\sum_{i\in\mathcal P_j}C_{p,i}A_i\,
\mathbf n_i\cdot\mathbf l_j,$$

with the strip area $S_j$ [m^2] and the panel areas $A_i$ [m^2]. This
is the lift of the 2D section reference: the 2D Morino lift is the
force in the section plane normal to the in-plane onset. For a planar
strip, $\mathbf l_j=[-\sin\alpha,0,\cos\alpha]^T$ is the global lift
axis. On a strip with dihedral the global lift axis would give only
about $\cos\Gamma$ of the section lift, and on a fin no lift at all.
The spanwise `Cl` of the panel result keeps the global lift axis;
`details["cl_inviscid"]` holds $C_{l,p,j}$.

The coupled strip equations are

$$r_j=C_{l,p,j}(\delta)-c_{l,\mathrm{target},j}(\alpha_{v,j}),
\qquad
\alpha_{v,j}=C_{l,\mathrm{mesh},j}^{-1}\left(C_{l,p,j}\right)+\delta_j.$$

There is no thin-airfoil flap offset in the thick-surface coupling. The
rule is that an identity polar gives zero correction. The converged and
the same-mesh references of a strip with a deflected control are both
the flapped cut section (the reference mesh has the same controls), so
the departure of a flapped polar from the converged flapped section
already holds the flap. The offset of the vortex-lattice coupling
(see "Reference of a flapped vortex-lattice strip") would count the flap
a second time; no path of the thick-surface coupling has an unflapped
reference. A polar equal to the converged flapped section gives a CL
correction below 0.1 %
(`tests/test_panel3d_viscous_coupling.py::test_converged_flapped_section_polar_gives_zero_correction`).
The details `flap_offset` (zero) and `alpha0_thin_airfoil` (equal to
`alpha0_section`) stay in the result.

Let $H_{jk}=\partial C_{l,p,j}/\partial x_k$, where $x$ holds the strip
angles and, for pressure Kutta, the wake shifts. The surface gradient
columns of the body response are computed once per condition. The
velocity is affine in $x$, so its exact pressure derivative is

$$\frac{\partial C_{p,i}}{\partial x_k}
=-\frac{2}{V_\infty^2}\mathbf V_i\cdot
\frac{\partial\mathbf V_i}{\partial x_k}.$$

Strip pressure integration gives $H$. If
$t_j=dc_{l,\mathrm{target},j}/d\alpha$ at $\alpha_{v,j}$ and
$s_j=C_{l,\mathrm{mesh},j}'(\alpha_{e,j})$, then the lift rows of the
Jacobian are

$$J_{jk}=\left(1-\frac{t_j}{s_j}\right)H_{jk}
-t_j I_{jk},$$

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
