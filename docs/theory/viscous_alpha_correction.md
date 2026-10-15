---
orphan: true
---

# Viscous Alpha-Correction Coupling

This document describes the viscous coupling method based on angle-of-attack correction (decambering) for 3D lifting surfaces.

## Overview

Inviscid solvers such as the Vortex Lattice Method (VLM) and the linear lifting-line theory (LLT) assume potential flow. They predict linear lift curves and neglect boundary layer separation and stall.

The viscous alpha-correction method couples 3D inviscid solvers with non-linear 2D section polars. The method modifies the local angle of attack on each spanwise strip using a correction angle $\delta_j$. The correction accounts for the loss of circulation due to boundary layer decambering and separation.

The formulation maintains the linear inviscid influence matrix throughout the non-linear iteration. The aerodynamic influence coefficient (AIC) matrix is factorized only once per geometry.

## Published References

The method implemented in Ventorum builds on published literature:

1. Tseng, J. B., and Lan, C. E. (1988), "Calculation of Aerodynamic Characteristics of Wings with Flaps in Subsonic Viscous Flow", NASA TP-2828.
2. van Dam, C. P., Vander Kam, J. C., and Paris, P. (2001), "Design-oriented high-lift methodology for general aviation and civil transport aircraft", *Journal of Aircraft*, 38(6), pp. 1111-1118.
3. Gallay, R., and Laurendeau, E. (2015), "Nonlinear generalized lifting-line coupling algorithms for pre/poststall flows", *AIAA Journal*, 53(7), pp. 1903-1915.
4. Mukherjee, R., and Gopalarathnam, A. (2006), "Poststall and Hysteresis Simulation for Wings Using a Decambering Approach", *Journal of Aircraft*, 43(3), pp. 642-651.

## Mathematical Formulation

Let the wing be divided into $N_s$ spanwise strips.

### Inviscid Response

In an inviscid panel solver or lifting-line model, the circulation vector $\boldsymbol{\Gamma}$ satisfies a linear system:

$$A \boldsymbol{\Gamma} = \mathbf{b}(\alpha, \boldsymbol{\delta})$$

Here $A$ is the inviscid influence matrix, and $\mathbf{b}$ is the boundary condition right-hand side.
A uniform angle reduction $\delta_k$ applied to strip $k$ alters the right-hand side.

Because the system is linear, the strip inviscid lift coefficient vector $\mathbf{C}_{l,\text{inv}}$ responds linearly to strip angle adjustments:

$$\mathbf{C}_{l,\text{inv}}(\boldsymbol{\delta}) = \mathbf{C}_{l,\text{inv}}(\mathbf{0}) - D \boldsymbol{\delta}$$

Here $D \in \mathbb{R}^{N_s \times N_s}$ is the strip influence response matrix.
Column $k$ of matrix $D$ represents the reduction in strip lift coefficients when strip $k$ is given a unit decambering angle.
Matrix $D$ is computed directly from the LU factorization of $A$ by solving with unit disturbance vectors on each strip.

### Viscous Section Matching

For each strip $j$, the effective angle of attack seen by the airfoil section is:

$$\alpha_{\text{eff},j} = \alpha_0 + \frac{C_{l,\text{inv},j}(\boldsymbol{\delta})}{a_{0,j}}$$

where $\alpha_0$ is the geometric angle of attack and $a_{0,j}$ is the 2D section lift slope in the linear regime.
With a decambering correction $\delta_j$, the modified effective angle for polar lookup is:

$$\alpha_{v,j} = \alpha_{\text{eff},j} - \delta_j$$

The viscous section polar provides the viscous lift coefficient $C_{l,\text{visc},j}(\alpha_{v,j}, \text{Re}_j)$ and profile drag coefficient $C_{d,\text{visc},j}(\alpha_{v,j}, \text{Re}_j)$ at the local Reynolds number $\text{Re}_j$.

The residual equation for each strip $j$ enforces equality between inviscid strip lift and viscous polar lift:

$$r_j(\boldsymbol{\delta}) = C_{l,\text{inv},j}(\boldsymbol{\delta}) - C_{l,\text{visc},j}(\alpha_{v,j}, \text{Re}_j) = 0$$

### Exact Newton-Raphson Jacobian

The vector residual is $\mathbf{r}(\boldsymbol{\delta}) = \mathbf{C}_{l,\text{inv}}(\boldsymbol{\delta}) - \mathbf{C}_{l,\text{visc}}(\boldsymbol{\alpha}_v)$.

Differentiating with respect to $\boldsymbol{\delta}$:

$$\frac{\partial \mathbf{C}_{l,\text{inv}}}{\partial \boldsymbol{\delta}} = -D$$

$$\frac{\partial \alpha_{v,j}}{\partial \delta_k} = \frac{1}{a_{0,j}} \frac{\partial C_{l,\text{inv},j}}{\partial \delta_k} - \delta_{jk} = -\frac{1}{a_{0,j}} D_{jk} - \delta_{jk}$$

Let $m_j = \frac{d C_{l,\text{visc},j}}{d \alpha}$ evaluated at $\alpha_{v,j}$, and $M = \operatorname{diag}(m_1, \dots, m_{N_s})$.
Then:

$$\frac{\partial \mathbf{C}_{l,\text{visc}}}{\partial \boldsymbol{\delta}} = M \left( -\operatorname{diag}(a_0)^{-1} D - I \right)$$

The exact Jacobian $J = \frac{\partial \mathbf{r}}{\partial \boldsymbol{\delta}}$ is:

$$J = -D - M \left( -\operatorname{diag}(a_0)^{-1} D - I \right) = -\left( I - M \operatorname{diag}(a_0)^{-1} \right) D + M$$

In Ventorum, this exact Jacobian enables rapid quadratic convergence in pre-stall conditions.

### Algorithmic Safeguards

1. **Under-Relaxation and Backtracking**: A line search monitors the $L_2$ residual norm $\|\mathbf{r}\|$. If a full Newton step increases the residual, step size halving restores monotonicity.
2. **LU Factorization Reuse**: Matrix $A$ is factorized once per wing geometry. Computing $D$ requires $N_s$ triangular solves. Subsequent iterations require only solves of the small $N_s \times N_s$ Jacobian system.
3. **Continuation Sweeps**: When solving sweeps over angles of attack, the converged $\boldsymbol{\delta}$ of the previous step initializes the next step.

## Post-Stall Branches and Honesty

In the post-stall regime, the local lift curve slope $m_j \le 0$. The governing system can exhibit multiple equilibrium states (hysteresis and non-unique solutions).

Ventorum applies the following checks:
- **Detection**: Any strip with $m_j \le 0$ at the converged solution is flagged as `post_stall_non_unique`.
- **Secondary Branch Search**: When post-stall conditions are detected, the solver restarts from an alternate stalled initial guess. If the secondary solution converges to a different branch ($\Delta C_L > 10^{-3}$), both branches are recorded.
- **Trust Score Penalty**: Post-stall solutions receive an aerodynamic trust penalty `POST_STALL_BRANCH_PENALTY` (0.35) to inform the user that post-stall low-order predictions have high uncertainty.
- **Non-Convergence Reporting**: When the Newton iteration reaches `max_iter` without satisfying tolerance, the result is marked `converged=False` with a user warning. Unconverged results are never silently presented as valid.

## Drag Calculation

The solver reports two distinct drag components:
1. **Induced Drag ($C_{D,\text{ind}}$)**: Computed from the converged wake circulation and downwash distributions in the 3D solver.
2. **Profile Drag ($C_{D,\text{prof}}$)**: Evaluated from the 2D section polars at the converged local effective angles $\alpha_{v,j}$ and Reynolds numbers $\text{Re}_j$:

$$C_{D,\text{prof}} = \frac{1}{S_{\text{ref}}} \sum_{j=1}^{N_s} C_{d,\text{visc},j}(\alpha_{v,j}, \text{Re}_j) \cdot c_j \Delta y_j$$

The total wing drag is $C_D = C_{D,\text{ind}} + C_{D,\text{prof}}$.

## Hardware Execution

The alpha-correction coupling algorithm runs on the CPU. The GPU pipeline for batch viscous coupling will be implemented in a subsequent phase.
