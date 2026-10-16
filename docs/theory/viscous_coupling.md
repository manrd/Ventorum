---
orphan: true
---

# Viscous coupling: angle-of-attack correction and parasite drag

## Model

Ventorum incorporates viscous aerodynamic effects into three-dimensional potential-flow solutions through two complementary methods:
1. Viscous alpha-correction coupling (decambering): couples 3D potential-flow solvers (vortex-lattice method and linear lifting-line theory) with non-linear 2D section polars. The method modifies the boundary conditions of each spanwise strip using an effective angle-of-attack correction while keeping the 3D potential influence matrix unchanged.
2. Parasite-drag component build-up: computes the zero-lift friction and form drag of non-lifting bodies (fuselages, nacelles, hulls, and struts) and provides independent profile-drag cross-checks for lifting surfaces using classical empirical form factors.

Both methods operate without altering the underlying linear matrix factorizations, preserving high computational speed and enabling honest physical trust estimation.

## Viscous alpha-correction coupling

### Physical model and coupling overview

Linear potential-flow methods predict linear lift curves and neglect boundary layer thickening, separation, and stall. The viscous alpha-correction method couples the 3D inviscid response with tabulated or analytical 2D section polars (Tseng and Lan, 1988; van Dam, Vander Kam and Paris, 2001; Gallay and Laurendeau, 2015; Mukherjee and Gopalarathnam, 2006).

The lifting surface is divided into $N_s$ spanwise strips. On each strip $j$, viscous decambering and boundary layer displacement reduce the effective circulation. The method introduces a correction angle $\delta_j$ [rad] on each strip, which modifies the boundary condition right-hand side of the inviscid system while preserving the aerodynamic influence matrix.

### Inviscid strip response

The 3D potential solver governs circulation vector $\boldsymbol{\Gamma}$ through the linear system:

```{math}
A \boldsymbol{\Gamma} = \mathbf{b}(\alpha, \boldsymbol{\delta}),
```

where $A$ is the aerodynamic influence coefficient matrix and $\mathbf{b}$ is the flow-tangency right-hand side. Because the potential system is linear, the strip inviscid lift coefficient vector $\mathbf{C}_{l,\text{inv}}$ responds linearly to strip angle adjustments:

```{math}
\mathbf{C}_{l,\text{inv}}(\boldsymbol{\delta}) = \mathbf{C}_{l,\text{inv}}(\mathbf{0}) - D \boldsymbol{\delta},
```

where $D \in \mathbb{R}^{N_s \times N_s}$ is the strip influence response matrix. Column $k$ of matrix $D$ represents the reduction in strip lift coefficients when strip $k$ is given a unit decambering angle. Matrix $D$ is computed directly from the LU factorization of $A$ through $N_s$ triangular back-substitutions:

```{math}
D_{:, k} = \Delta \mathbf{C}_{l,\text{inv}}(\mathbf{e}_k).
```

Because matrix $A$ depends only on geometry, matrix $D$ is computed once per configuration.

### Viscous section matching

For strip $j$, the effective angle of attack seen by the airfoil section is:

```{math}
\alpha_{\text{eff}, j} = \alpha_0 + \frac{C_{l,\text{inv}, j}(\boldsymbol{\delta})}{a_{0, j}},
```

where $\alpha_0$ is the geometric angle of attack and $a_{0, j}$ is the linear 2D section lift slope. Applying decambering correction $\delta_j$, the modified effective angle for section polar lookup is:

```{math}
\alpha_{v, j} = \alpha_{\text{eff}, j} - \delta_j.
```

The 2D section polar supplies the viscous lift coefficient $C_{l,\text{visc}, j}(\alpha_{v, j}, Re_j)$ and profile drag coefficient $C_{d,\text{visc}, j}(\alpha_{v, j}, Re_j)$ at the local strip Reynolds number:

```{math}
Re_j = \frac{V_\infty c_j}{\nu}.
```

The residual equation on strip $j$ enforces equality between the inviscid strip lift and the viscous polar lift:

```{math}
r_j(\boldsymbol{\delta}) = C_{l,\text{inv}, j}(\boldsymbol{\delta}) - C_{l,\text{visc}, j}(\alpha_{v, j}, Re_j) = 0.
```

### Exact Newton-Raphson Jacobian

The non-linear system $\mathbf{r}(\boldsymbol{\delta}) = \mathbf{0}$ is solved by Newton-Raphson iteration. Differentiating the residual with respect to the decambering vector $\boldsymbol{\delta}$:

```{math}
\frac{\partial \mathbf{C}_{l,\text{inv}}}{\partial \boldsymbol{\delta}} = -D,
```

```{math}
\frac{\partial \alpha_{v, j}}{\partial \delta_k} = \frac{1}{a_{0, j}} \frac{\partial C_{l,\text{inv}, j}}{\partial \delta_k} - \delta_{jk} = -\frac{1}{a_{0, j}} D_{jk} - \delta_{jk}.
```

Let $m_j = \left.\frac{d C_{l,\text{visc}, j}}{d \alpha}\right|_{\alpha_{v, j}}$ be the local section lift curve slope, and $M = \operatorname{diag}(m_1, \dots, m_{N_s})$. Differentiating the viscous term gives:

```{math}
\frac{\partial \mathbf{C}_{l,\text{visc}}}{\partial \boldsymbol{\delta}} = M \left(-\operatorname{diag}(a_0)^{-1} D - I\right).
```

The exact Jacobian matrix $J = \frac{\partial \mathbf{r}}{\partial \boldsymbol{\delta}}$ is:

```{math}
J = -D - M \left(-\operatorname{diag}(a_0)^{-1} D - I\right) = -\left(I - M \operatorname{diag}(a_0)^{-1}\right) D + M.
```

This exact analytical Jacobian enables rapid quadratic convergence in pre-stall regimes.

### Numerical safeguards and continuation

1. Under-relaxation and line search: A backtracking line search monitors the residual $L_2$ norm $\|\mathbf{r}\|_2$. If a full Newton step increases the residual, step size halving restores monotonic decrease.
2. Factorization reuse: Matrix $A$ is factorized once. Each iteration requires only the solve of the small $N_s \times N_s$ Jacobian system.
3. Angle continuation: For angle-of-attack sweeps, the converged decambering vector $\boldsymbol{\delta}$ of the preceding angle initializes the next angle.

### Post-stall branches and trust reporting

In post-stall conditions, the local section lift slope becomes non-positive ($m_j \le 0$). The governing system can develop multiple solution branches and hysteresis:
- Strip stall detection: Any strip exhibiting $m_j \le 0$ at the converged state is flagged as post-stall.
- Secondary branch search: When stall is detected, the solver restarts from an alternate stalled initial guess. If the secondary solve converges to a distinct solution ($\Delta C_L > 10^{-3}$), both branches are reported.
- Trust penalty: Post-stall solutions receive an aerodynamic trust penalty of 0.35 (`POST_STALL_BRANCH_PENALTY`) to inform the user of increased physical uncertainty.
- Convergence status: When the iteration reaches `max_iter` without satisfying tolerance, the result is marked `converged=False` with diagnostic warnings.

### Drag summation

The coupled solver reports two distinct drag components:
1. Induced drag ($C_{D,\text{ind}}$): Computed from the converged 3D wake circulation and downwash distributions in the Trefftz plane.
2. Profile drag ($C_{D,\text{prof}}$): Integrated from 2D section polars at converged effective angles $\alpha_{v, j}$ and Reynolds numbers $Re_j$:

```{math}
C_{D,\text{prof}} = \frac{1}{S_{\text{ref}}} \sum_{j=1}^{N_s} C_{d,\text{visc}, j}(\alpha_{v, j}, Re_j) \, c_j \Delta y_j.
```

Total aircraft drag is the sum:

```{math}
C_D = C_{D,\text{ind}} + C_{D,\text{prof}}.
```

## Parasite-drag build-up

### Component build-up method

The parasite drag of components not covered by lifting-surface models (fuselages, nacelles, hulls, and struts) is estimated using the component build-up method of Raymer (2018, section 12.5). Each component contributes its flat-plate skin friction multiplied by a form factor and an interference factor:

```{math}
C_{D0, i} = C_{f, i} \, FF_i \, Q_i \, \frac{S_{\text{wet}, i}}{S_{\text{ref}}}, \qquad
C_{D0} = \sum_i C_{D0, i},
```

where $C_{f, i}$ is the skin-friction coefficient, $FF_i$ is the form factor, $Q_i$ is the interference factor, $S_{\text{wet}, i}$ is the wetted area [m^2], and $S_{\text{ref}}$ is the reference wing area [m^2].

### Skin friction relations

The Reynolds number is $Re = \rho V_\infty L / \mu$, with reference length $L$ and sea-level dynamic viscosity $\mu = 1.7894 \times 10^{-5}$ Pa s:
- Laminar flow uses the Blasius (1908) flat-plate solution:

```{math}
C_{f, \text{lam}} = \frac{1.328}{\sqrt{Re}}.
```

- Fully turbulent flow uses the Prandtl-Schlichting relation (Schlichting, 2017):

```{math}
C_{f, \text{turb}} = \frac{0.455}{(\log_{10} Re)^{2.58}}.
```

- Mixed laminar-turbulent flow with transition at fraction $x_{\text{tr}} \in [0, 1]$ uses the Schlichting composite rule ($Re_{\text{tr}} = x_{\text{tr}} Re$):

```{math}
C_f = C_{f, \text{turb}}(Re) - x_{\text{tr}} \left[C_{f, \text{turb}}(Re_{\text{tr}}) - C_{f, \text{lam}}(Re_{\text{tr}})\right].
```

- Surface roughness cutoff uses the Raymer (2018) cutoff Reynolds number based on equivalent sand grain roughness $k$ [m]:

```{math}
Re_{\text{co}} = 38.21 \left(\frac{L}{k}\right)^{1.053}, \qquad
C_f = C_f(\min(Re, Re_{\text{co}})).
```

When $Re > Re_{\text{co}}$, the boundary layer is treated as fully turbulent at $Re_{\text{co}}$.

### Form factors

1. Lifting surfaces:
The default wing form factor follows Raymer (2018, section 12.5) for incompressible flow (Mach below 0.3). With thickness ratio $t/c$, chordwise position of maximum thickness $x_t$, and sweep of the maximum-thickness line $\Lambda$:

```{math}
FF = \left[1 + \frac{0.6}{x_t}\left(\frac{t}{c}\right) + 100 \left(\frac{t}{c}\right)^4\right] (\cos\Lambda)^{0.28}.
```

An alternative wing form factor follows Torenbeek (1982, appendix F), evaluated on the normal thickness ratio $(t/c)_n = (t/c) / \cos\Lambda$:

```{math}
FF = 1 + 2.7 (t/c)_n + 100 (t/c)_n^4.
```

2. Fuselages and streamlined bodies:
The default body form factor follows Raymer (2018) with fineness ratio $f = l/d$:

```{math}
FF = 1 + \frac{60}{f^3} + \frac{f}{400}.
```

An alternative streamlined-body form factor follows Hoerner (1965, chapter 6):

```{math}
FF = 1 + 1.5 \left(\frac{d}{l}\right)^{3/2} + 7 \left(\frac{d}{l}\right)^3.
```

### Wetted area and interference factors

Lifting surface wetted area is approximated from planform area $S$ (Raymer, 2018):

```{math}
S_{\text{wet}} \approx 2 S \left(1 + 0.2 \frac{t}{c}\right).
```

Interference factor $Q$ accounts for junction interference drag:
- $Q = 1.0$ for clean, well-faired wing-body junctions and isolated bodies.
- $Q = 1.05$ to $1.15$ for standard low/mid-wing configurations.
- $Q = 1.20$ to $1.30$ for unfilleted intersections.

### Uncertainty and trust

Each component carries a relative uncertainty (default 0.10). The total absolute uncertainty is the root sum square of independent component uncertainties:

```{math}
u_{\text{total}} = \sqrt{\sum_i u_i^2}, \qquad u_i = 0.10 \, C_{D0, i}.
```

This estimate provides conceptual design drag levels and cross-checks profile-drag polar predictions.

## Assumptions

- Decambering acts as a local modification of the potential boundary condition without altering the 3D potential influence matrix.
- Section polars represent quasi-steady, two-dimensional section aerodynamics at local strip Reynolds numbers.
- Slender non-lifting bodies without large-scale separated base flows.
- Subsonic incompressible flow (Mach below 0.3).

## Envelope

- Pre-stall and post-stall regimes up to moderate angles of attack; deep post-stall hysteresis is flagged with trust penalties.
- Streamlined bodies with fineness ratios $f \ge 3$.
- Clean and conventional aircraft geometries.

## Implementation

- Alpha-correction coupling: `ventorum.viscous.alpha_correction.solve_alpha_correction`, `strip_influence_matrix`.
- Parasite-drag build-up: `ventorum.aero.parasite.compute_parasite_drag`, `component_skin_friction`, `form_factor_wing_raymer`, `form_factor_body_raymer`.

## Verification

The methods in this chapter are verified by tests in the test suite:
- Decambering coupling and exact Jacobian: `tests/test_viscous_alpha_correction.py`.
- Parasite drag relations and form factors: `tests/test_parasite_drag.py`.

## References

- H. Blasius, "Grenzschichten in Fluessigkeiten mit kleiner Reibung", *Zeitschrift fuer Mathematik und Physik*, Vol. 56, 1908, pp. 1-37.
- R. Gallay and E. Laurendeau, "Nonlinear Generalized Lifting-Line Coupling Algorithms for Pre/Poststall Flows", *AIAA Journal*, Vol. 53, No. 7, 2015, pp. 1904-1915.
- S. F. Hoerner, *Fluid-Dynamic Drag: Practical Information on Aerodynamic Drag and Hydrodynamic Resistance*, Published by the Author, Midland Park, NJ, 1965.
- R. Mukherjee and A. Gopalarathnam, "Poststall Aerodynamic Prediction for Wings Using Decambering Approach", *Journal of Aircraft*, Vol. 43, No. 3, 2006, pp. 660-668.
- D. P. Raymer, *Aircraft Design: A Conceptual Approach*, 6th ed., AIAA Education Series, Reston, VA, 2018.
- H. Schlichting and K. Gersten, *Boundary-Layer Theory*, 9th ed., Springer-Verlag, Berlin, 2017.
- E. Torenbeek, *Synthesis of Subsonic Airplane Design*, Delft University Press, Kluwer Academic Publishers, Dordrecht, 1982.
- J. B. Tseng and C. E. Lan, "Calculation of Aerodynamic Characteristics of Wings with Flaps in Subsonic Viscous Flow", NASA TP-2828, 1988.
- C. P. van Dam, J. C. Vander Kam and P. C. Paris, "Design-Oriented High-Lift Methodology for General Aviation and Civil Transport Aircraft", *Journal of Aircraft*, Vol. 38, No. 6, 2001, pp. 1105-1115.
