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
4. R. Mukherjee and A. Gopalarathnam, "Poststall Prediction of Multiple-Lifting-Surface Configurations Using a Decambering Approach", *Journal of Aircraft*, 43(3), 2006, pp. 660-668.
5. H. Glauert, "Theoretical Relationships for an Aerofoil with Hinged Flap", ARC R&M 1095, 1927 (thin-airfoil flap shift).
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

**Flaps are counted once.** The lattice gives the inviscid flap effect in $C_{l,\text{inv}}$, and $\alpha_{0,j}$ (from the lattice itself) takes it out of the effective angle. The polar of a strip with a deflected control is the flapped polar (a table at the flap deflection, a source with the flap of the control, or a single polar with the Glauert shift). With polars that follow thin-airfoil theory (slope $2\pi$, Glauert flap shift), the lifting-line result equals the inviscid result. The inviscid vortex-lattice flap lift converges with the chordwise panel count as $1/n_	ext{chord}$ and is too low on a coarse chordwise mesh; the coupling takes the flap lift from the polar, so its result equals the mesh-converged inviscid value at every chordwise panel count. Against the inviscid result at the same chordwise panel count, the coupled result is higher by the chordwise discretization error of the lattice flap, which halves when the panel count doubles (tests in `tests/test_alpha_correction.py`).

**Ground effect.** When the polar of a strip holds the ground height (a table with a height axis, or a source with ground support), the 2D section of a vortex-lattice strip also has its image vortices in the ground, so the 2D ground effect is not counted twice. A free-air polar is read with the free-air section relation. The height of a strip is the height of its quarter-chord point above the ground divided by its chord. The lifting line has no chordwise extent: its section relation has no 2D ground effect, and the polar gives it.

## Residual and Newton iteration

The residual of strip $j$ is

$$r_j(\boldsymbol{\delta}) = C_{l,\text{inv},j}(\boldsymbol{\delta}) - C_{l,\text{visc},j}(\alpha_{v,j}(\boldsymbol{\delta}), Re_j).$$

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
