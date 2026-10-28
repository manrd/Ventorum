---
orphan: true
---

# Viscous coupling: angle-of-attack correction and parasite drag

## Model

Ventorum incorporates viscous aerodynamic effects into three-dimensional potential-flow solutions through two complementary methods:
1. Viscous alpha-correction coupling (decambering): couples 3D potential-flow solvers (vortex-lattice method, linear lifting-line theory and, with `model="panel3d"`, the 3D panel method) with non-linear 2D section polars. The method modifies the boundary conditions of each spanwise strip using an effective angle-of-attack correction while keeping the 3D potential influence matrix unchanged.
2. Parasite-drag component build-up: computes the zero-lift friction and form drag of non-lifting bodies (fuselages, nacelles, hulls, and struts) and provides independent profile-drag cross-checks for lifting surfaces using classical empirical form factors.

Both methods operate without altering the underlying linear matrix factorizations, preserving high computational speed and enabling honest physical trust estimation.

## Viscous alpha-correction coupling

### Physical model and coupling overview

Linear potential-flow methods predict linear lift curves and neglect boundary-layer thickening, separation and stall. The viscous alpha-correction method (V1) couples the vortex-lattice method, the linear lifting line or the 3D panel method (`model="panel3d"`, which matches the pressure-integrated lift of each strip; see the detailed page, section *Thick-surface panel coupling*) with tabulated or analytical 2D section polars (Tseng and Lan, 1988; van Dam, Vander Kam and Paris, 2001; Gallay and Laurendeau, 2015; Mukherjee and Gopalarathnam, 2006). The detailed page is [Viscous alpha-correction coupling](viscous_alpha_correction.md); this section gives the model.

Each spanwise strip $j$ gets one correction angle $\delta_j$ [rad]. The correction changes only the right-hand side of the strip (a uniform change of its angle of attack); the influence matrix does not change.

### Inviscid strip response

The circulation $\boldsymbol{\Gamma}$ satisfies

```{math}
A \boldsymbol{\Gamma} = \mathbf{b}(\alpha, \boldsymbol{\delta}).
```

A correction $\delta_k$ lowers the section angle of strip $k$: the onset of its boundary conditions turns by $-\delta_k$ about the span axis of the strip, in its section plane. For a planar strip this turns the onset direction from $\mathbf{d}(\alpha)$ to $\mathbf{d}(\alpha - \delta_k)$. The strip lift coefficient $C_{l,\text{inv},j} = 2\Gamma_j / (V_\infty c_j)$ is linear in the corrections:

```{math}
\mathbf{C}_{l,\text{inv}}(\boldsymbol{\delta}) = \mathbf{C}_{l,\text{inv}}(\mathbf{0}) - D \boldsymbol{\delta},
```

where column $k$ of $D \in \mathbb{R}^{N_s \times N_s}$ comes from one back-substitution with the LU factors of $A$. A positive correction lowers the lift of the strip.

### Effective angle and viscous section matching

Each strip behaves like a 2D section of the lattice model, $C_l = a_j(\alpha_e - \alpha_{0,j})$. For the vortex-lattice method, $a_j$ [1/rad] and $\alpha_{0,j}$ [rad] come from the lattice itself: the chordwise vortices, control points and boundary normals of the strip (camber and flap panels) form a 2D discrete-vortex section, and two right-hand sides with one factorization give its slope and zero-lift angle (thin-airfoil theory; Katz and Plotkin, 2001). For the lifting line they are the $a_0$ and $\alpha_{L0}$ of the strip airfoil (with the Glauert flap shift on a deflected strip). The effective angle and the viscous section angle are

```{math}
\alpha_{e, j} = \alpha_{0,j} + \frac{C_{l,\text{inv}, j}}{a_j}, \qquad
\alpha_{v, j} = \alpha_{e, j} + \delta_j .
```

The polar of the strip (the flapped polar on a strip with a deflected control) supplies $C_{l,\text{visc}, j}(\alpha_{v, j}, Re_j)$ and $C_{d,\text{visc}, j}(\alpha_{v, j}, Re_j)$ at the strip Reynolds number

```{math}
Re_j = \frac{\rho V_\infty c_j}{\mu},
```

with the density $\rho$ and the speed $V_\infty$ of the flight condition and the ISA sea-level dynamic viscosity $\mu$ of the code (`MU_SL` in `ventorum.aero.parasite`, ISO 2533). Because $\alpha_{0,j}$ comes from the lattice, the inviscid flap effect is counted once: polars that follow thin-airfoil theory give the inviscid result. When the polar holds the ground height, the 2D section of a vortex-lattice strip also has its image vortices in the ground.

The residual of strip $j$ is

```{math}
r_j(\boldsymbol{\delta}) = C_{l,\text{inv}, j}(\boldsymbol{\delta}) - C_{l,\text{visc}, j}(\alpha_{v, j}, Re_j) + a_j\left(\alpha_{0,j} - \alpha_{0,\text{TA},j}\right).
```

The last term is the constant flap offset of a vortex-lattice strip with a deflected control, with $\alpha_{0,\text{TA},j}$ the zero-lift angle of the thin-airfoil model of the strip. It is zero on every other strip and does not change the Jacobian (see [Viscous alpha-correction coupling](viscous_alpha_correction.md)).

### Newton-Raphson Jacobian

With $\partial \alpha_{v,j}/\partial \delta_k = -D_{jk}/a_j + \delta_{jk}$, the local polar slope $m_j = \mathrm{d}C_{l,\text{visc}, j}/\mathrm{d}\alpha$ at $\alpha_{v, j}$ and $M = \operatorname{diag}(m_1, \dots, m_{N_s})$, the Jacobian is

```{math}
J = \frac{\partial \mathbf{r}}{\partial \boldsymbol{\delta}} = -\left(I - M \operatorname{diag}(a)^{-1}\right) D - M.
```

### Numerical safeguards and continuation

1. Line search and under-relaxation: the Newton step has an under-relaxation factor (input `relaxation`, default 1) and a backtracking line search on $\frac{1}{2}\|\mathbf{r}\|_2^2$; a damped fixed-point step follows when no Newton step lowers the residual.
2. Factorization reuse: within one condition, $D$, the iteration and the branch search use one factorization. Between the conditions of a sweep the factors are used again when the matrix does not change (for example `wake_alignment="body"` out of ground effect). With the default free-stream wake the matrix depends on the angle of attack, so a sweep factorizes once per condition. The result reports the real count.
3. Angle continuation: a sweep runs from the angle nearest to zero upward and from there downward; each condition starts from the corrections of its converged neighbour.

### Post-stall branches and trust reporting

In post-stall conditions the local section slope is not positive ($m_j \le 0$), and the coupled equations can have more than one solution:
- Strip stall detection: a strip with $m_j \le 0$ at a converged solution gets the flag "post-stall, non-unique".
- Branch search: the solver also runs from a stalled start and from an attached start. Every other converged solution whose CL differs by more than $10^{-3}$ is reported as a branch.
- Trust penalty: a result with more than one branch gets the trust penalty `POST_STALL_BRANCH_PENALTY`.
- Convergence status: when the iteration reaches `max_iter` without the tolerance, the result has `converged=False`, a warning and no branch verdict.

### Drag summation

The coupled solver reports two drag components:
1. Induced drag ($C_{D_i}$): from the converged circulation in the Trefftz plane.
2. Profile drag ($C_{D_p}$): from the 2D section polars at the solved angles $\alpha_{v, j}$ and Reynolds numbers $Re_j$:

```{math}
C_{D_p} = \frac{1}{S_{\text{ref}}} \sum_{j=1}^{N_s} C_{d,\text{visc}, j}(\alpha_{v, j}, Re_j) \, c_j \Delta y_j.
```

The total drag is the sum:

```{math}
C_D = C_{D_i} + C_{D_p}.
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

The Reynolds number is $Re = \rho V_\infty L / \mu$, with reference length $L$ and the ISA sea-level dynamic viscosity of the code, $\mu = 1.789 \times 10^{-5}$ Pa s (`MU_SL`, ISO 2533):
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
Re_{\text{co}} = 38.21 \left(\frac{L}{k}\right)^{1.053}.
```

Raymer evaluates the turbulent skin friction at $\min(Re, Re_{\text{co}})$. The cutoff acts on the turbulent part only; the laminar part keeps its value. For mixed flow the module adds the roughness increment of the turbulent law, weighted by the turbulent fraction of the length (an approximation of this module):

```{math}
C_f = C_{f,\text{smooth}}(Re, x_{\text{tr}}) + (1 - x_{\text{tr}}) \left[C_{f,\text{turb}}(\min(Re, Re_{\text{co}})) - C_{f,\text{turb}}(Re)\right].
```

The increment is zero below the cutoff, so $C_f$ is continuous across the cutoff. A fully laminar plate keeps $C_{f,\text{lam}}(Re)$ (see [Parasite drag](parasite_drag.md)).

### Form factors

1. Lifting surfaces:
The default wing form factor follows Raymer (2018, section 12.5) for incompressible flow (Mach below 0.3). With thickness ratio $t/c$, chordwise position of maximum thickness $x_t$, and sweep of the maximum-thickness line $\Lambda$:

```{math}
FF = \left[1 + \frac{0.6}{x_t}\left(\frac{t}{c}\right) + 100 \left(\frac{t}{c}\right)^4\right] (\cos\Lambda)^{0.28}.
```

An alternative wing form factor follows Torenbeek (1982, appendix F), evaluated on the streamwise thickness ratio $t/c$:

```{math}
FF = 1 + 2.7 \frac{t}{c} + 100 \left(\frac{t}{c}\right)^4.
```

This form has no sweep term.

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

Lifting surface wetted area comes from the exposed planform area $S_{\text{exposed}}$:

```{math}
S_{\text{wet}} = 2 S_{\text{exposed}} \left(1 + 0.2 \frac{t}{c}\right).
```

The factor 2 counts the upper and the lower surface. The term $0.2\,t/c$ is an approximation of the thickness factor between the planform and the wetted area of a section (see [Parasite drag](parasite_drag.md)).

The interference factor $Q$ accounts for junction interference drag. It is an input of each component (default 1, no interference), never a hidden constant; typical values for a wing-body junction are 1.0 to 1.3 (Raymer, 2018, section 12.5).

### Uncertainty and trust

Each component carries a relative uncertainty $\epsilon_i$ (an input, default 0.10). The total absolute uncertainty is the root sum square of independent component uncertainties:

```{math}
u_{\text{total}} = \sqrt{\sum_i u_i^2}, \qquad u_i = \epsilon_i \, C_{D0, i}.
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

- Alpha-correction coupling: `ventorum.viscous.alpha_correction.solve_viscous`, `strip_reynolds_number`.
- Parasite-drag build-up: `ventorum.aero.parasite.parasite_drag`, `DragComponent`, `flat_plate_cf`, `form_factor_lifting`, `form_factor_body`, `wetted_area_lifting_surface`.
- The coupling runs on the CPU; a GPU path is planned for a later version.

## Verification

The methods in this chapter are verified by tests in the test suite:
- Alpha-correction coupling (inviscid limit, lifting-line slope, flaps, branches, factorization count, profile drag): `tests/test_alpha_correction.py`.
- Parasite drag relations and form factors: `tests/test_parasite_drag.py`.

## References

- H. Blasius, "Grenzschichten in Fluessigkeiten mit kleiner Reibung", *Zeitschrift fuer Mathematik und Physik*, Vol. 56, 1908, pp. 1-37.
- S. Gallay and E. Laurendeau, "Nonlinear Generalized Lifting-Line Coupling Algorithms for Pre/Poststall Flows", *AIAA Journal*, Vol. 53, No. 7, 2015, pp. 1784-1792.
- H. Glauert, "Theoretical Relationships for an Aerofoil with Hinged Flap", ARC R&M 1095, 1927.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001.
- ISO 2533:1975, Standard Atmosphere.
- S. F. Hoerner, *Fluid-Dynamic Drag: Practical Information on Aerodynamic Drag and Hydrodynamic Resistance*, Published by the Author, Midland Park, NJ, 1965.
- R. Mukherjee and A. Gopalarathnam, "Poststall Prediction of Multiple-Lifting-Surface Configurations Using a Decambering Approach", *Journal of Aircraft*, Vol. 43, No. 3, 2006, pp. 660-668.
- D. P. Raymer, *Aircraft Design: A Conceptual Approach*, 6th ed., AIAA Education Series, Reston, VA, 2018.
- H. Schlichting and K. Gersten, *Boundary-Layer Theory*, 9th ed., Springer-Verlag, Berlin, 2017.
- E. Torenbeek, *Synthesis of Subsonic Airplane Design*, Delft University Press, Kluwer Academic Publishers, Dordrecht, 1982.
- J. B. Tseng and C. E. Lan, NASA TP-2828, 1988 (alpha-reduction coupling of a vortex lattice with section data).
- C. P. van Dam, J. C. Vander Kam and P. C. Paris, "Design-Oriented High-Lift Methodology for General Aviation and Civil Transport Aircraft", *Journal of Aircraft*, Vol. 38, No. 6, 2001, pp. 1076-1084.
