---
orphan: true
---

# Ventorum-3DP Potential Flow Panel Solver

This document describes the theory, boundary conditions, wake model, unit-onset superposition, and post-processing of the three-dimensional source-doublet panel solver in Ventorum-3DP.

## Overview

Ventorum-3DP solves the steady, incompressible, inviscid potential flow around arbitrary three-dimensional lifting configurations and non-lifting bodies. The method discretizes the surface into planar quadrilateral and triangular panels. Each body panel carries a constant source distribution of strength $\sigma$ and a constant doublet distribution of strength $\mu$.

The total velocity potential $\Phi$ in the fluid is the sum of the onset potential $\Phi_{\infty}$ and the perturbation potential $\phi$:

$$\Phi(\mathbf{x}) = \Phi_{\infty}(\mathbf{x}) + \phi(\mathbf{x})$$

The perturbation potential satisfies Laplace's equation in the fluid domain:

$$\nabla^2 \phi = 0$$

## Formulation and Boundary Conditions

Ventorum-3DP uses the internal Dirichlet boundary condition introduced by Morino and Kuo (1974). An fictitious flow field is defined inside the closed body. The internal perturbation potential is set to zero throughout the interior:

$$\phi_{\text{int}}(\mathbf{x}) = 0$$

By Green's third identity, the perturbation potential at an interior point $\mathbf{x}_{\text{int}}$ approaches the internal boundary:

$$\phi_{\text{int}}(\mathbf{x}_i) = \sum_{j=1}^{N} \mu_j C_{ij} + \sum_{k=1}^{N_w} \mu_{w, k} C_{ik}^w + \sum_{j=1}^{N} \sigma_j B_{ij} = 0$$

where:
* $C_{ij}$ is the doublet influence coefficient of body panel $j$ on collocation point $i$.
* $C_{ik}^w$ is the doublet influence coefficient of wake panel $k$ on collocation point $i$.
* $B_{ij}$ is the source influence coefficient of body panel $j$ on collocation point $i$.
* $N$ is the total number of body panels.
* $N_w$ is the total number of wake panels.

### Collocation Points and Self-Influence

The collocation point of each panel is placed at its area centroid $\mathbf{c}_i$. The evaluation from the interior side of the panel gives the self-influence term for a unit doublet:

$$C_{ii} = +\frac{1}{2}$$

Source panels exhibit continuous potential across the surface, giving a smooth self-influence integral:

$$B_{ii} = \int_{S_i} -\frac{1}{4\pi \|\mathbf{c}_i - \mathbf{x}'\|} \, dS'$$

### Known Source Strengths

The external Neumann boundary condition requires zero normal velocity through the impermeable surface:

$$\nabla \Phi \cdot \hat{\mathbf{n}} = 0 \implies \frac{\partial \phi}{\partial n} = -\mathbf{V}_{\text{onset}} \cdot \hat{\mathbf{n}}$$

Because $\phi_{\text{int}} = 0$, the jump in normal velocity across the boundary equals the source strength:

$$\sigma_j = \left(\frac{\partial \phi}{\partial n}\right)_{\text{ext}} - \left(\frac{\partial \phi}{\partial n}\right)_{\text{int}} = -\mathbf{V}_{\text{onset}}(\mathbf{c}_j) \cdot \hat{\mathbf{n}}_j$$

The source strengths are known directly from the onset flow and surface geometry.

## Wake Model and Kutta Condition

### Body-Axis Wake

Each lifting surface sheds a flat vortex sheet from its trailing edge. The wake is discretized into strips of quadrilateral panels extending downstream along the body $+x$ axis. The default wake length is 50 times the maximum chord of the surface. Each strip contains $n_{\text{wake}}$ panels ($n_{\text{wake}} = 1$ by default).

The wake position is fixed in body axes. It does not rotate with angle of attack $\alpha$ or sideslip angle $\beta$. This formulation keeps the aerodynamic influence matrix independent of the flight condition.

### Morino Kutta Condition

The Kutta condition enforces finite velocity at the trailing edge. In the Morino potential formulation, this condition sets the doublet strength of each wake strip $\mu_w$ equal to the difference between the upper and lower surface doublet strengths at the trailing edge:

$$\mu_w = \mu_{\text{upper}} - \mu_{\text{lower}}$$

All panels along the same streamwise wake strip carry this identical doublet strength $\mu_w$.

Substituting the Kutta condition into the boundary integral equation incorporates the wake influence directly into the body doublet influence matrix:

$$A_{ij} = C_{ij} + \sum_{s \in \mathcal{W}_j} \pm C_{i, s}^w$$

where $+C_{i, s}^w$ adds to the upper trailing-edge panel column and $-C_{i, s}^w$ adds to the lower trailing-edge panel column. The final linear system is:

$$\mathbf{A} \boldsymbol{\mu} = -\mathbf{B} \boldsymbol{\sigma}$$

## Six Unit-Onset Superposition

The aerodynamic influence matrix $\mathbf{A}$ and source influence matrix $\mathbf{B}$ depend only on the surface and wake geometry. They do not depend on the flight condition.

Ventorum-3DP factors matrix $\mathbf{A}$ once by LU decomposition:

$$\mathbf{P} \mathbf{A} = \mathbf{L} \mathbf{U}$$

The solver computes six unit onset solutions:
1. Unit forward speed $u = 1$: $\mathbf{V}^{(u)} = [1, 0, 0]^T$
2. Unit lateral speed $v = 1$: $\mathbf{V}^{(v)} = [0, 1, 0]^T$
3. Unit vertical speed $w = 1$: $\mathbf{V}^{(w)} = [0, 0, 1]^T$
4. Unit roll rate $p = 1$: $\mathbf{V}^{(p)} = [0, -(z - z_{\text{ref}}), y - y_{\text{ref}}]^T$
5. Unit pitch rate $q = 1$: $\mathbf{V}^{(q)} = [-(z - z_{\text{ref}}), 0, x - x_{\text{ref}}]^T$
6. Unit yaw rate $r = 1$: $\mathbf{V}^{(r)} = [-(y - y_{\text{ref}}), x - x_{\text{ref}}, 0]^T$

The unit right-hand sides $\mathbf{b}^{(k)} = -\mathbf{B} \boldsymbol{\sigma}^{(k)}$ are solved simultaneously using the stored LU factors:

$$\boldsymbol{\mu}^{(k)} = \mathbf{A}^{-1} \mathbf{b}^{(k)}, \quad k \in \{u, v, w, p, q, r\}$$

For any arbitrary flight condition with speed $V_{\infty}$, angles $(\alpha, \beta)$, and rotation rates $(p, q, r)$, the translation velocity components in geometry axes are:

$$u = V_{\infty} \cos\alpha \cos\beta, \quad v = -V_{\infty} \sin\beta, \quad w = V_{\infty} \sin\alpha \cos\beta$$

The doublet and source strengths for the condition are obtained by linear superposition:

$$\boldsymbol{\mu} = u \boldsymbol{\mu}^{(u)} + v \boldsymbol{\mu}^{(v)} + w \boldsymbol{\mu}^{(w)} + p \boldsymbol{\mu}^{(p)} + q \boldsymbol{\mu}^{(q)} + r \boldsymbol{\mu}^{(r)}$$

$$\boldsymbol{\sigma} = u \boldsymbol{\sigma}^{(u)} + v \boldsymbol{\sigma}^{(v)} + w \boldsymbol{\sigma}^{(w)} + p \boldsymbol{\sigma}^{(p)} + q \boldsymbol{\sigma}^{(q)} + r \boldsymbol{\sigma}^{(r)}$$

## Surface Velocity and Pressure Coefficient

### Tangential Perturbation Velocity

On the outer surface of a closed body with zero internal perturbation potential, the tangential perturbation velocity equals the negative surface gradient of the doublet strength:

$$\mathbf{v}_{\text{pert}, t} = -\nabla_s \mu$$

The surface gradient $\nabla_s \mu$ on panel $i$ is calculated from a weighted least-squares fit over all neighbour panels that share an edge with panel $i$. The edge search explicitly excludes the wake shedding line to prevent differentiation across the wake jump discontinuity.

For each neighbour $j$, the displacement vector $\Delta \mathbf{r}_{ij} = \mathbf{c}_j - \mathbf{c}_i$ is projected onto the tangent plane of panel $i$:

$$\Delta \mathbf{r}_{ij, t} = \Delta \mathbf{r}_{ij} - (\Delta \mathbf{r}_{ij} \cdot \hat{\mathbf{n}}_i) \hat{\mathbf{n}}_i$$

The least-squares system solves:

$$\Delta \mathbf{r}_{ij, t} \cdot \mathbf{g}_i \approx \mu_j - \mu_i$$

The solved gradient $\mathbf{g}_i$ is then projected onto the panel plane to enforce exact tangency:

$$\nabla_s \mu_i = \mathbf{g}_i - (\mathbf{g}_i \cdot \hat{\mathbf{n}}_i) \hat{\mathbf{n}}_i$$

The total surface velocity on panel $i$ is:

$$\mathbf{V}_{\text{surf}, i} = \mathbf{V}_{\text{onset}, t}(\mathbf{c}_i) - \nabla_s \mu_i$$

where $\mathbf{V}_{\text{onset}, t} = \mathbf{V}_{\text{onset}} - (\mathbf{V}_{\text{onset}} \cdot \hat{\mathbf{n}}_i) \hat{\mathbf{n}}_i$.

### Pressure Coefficient

The local pressure coefficient $C_p$ is determined from Bernoulli's equation:

$$C_p = 1 - \frac{\|\mathbf{V}_{\text{surf}}\|^2}{V_{\infty}^2}$$

## Force and Moment Integration

### Near-Field Surface Integration

The total aerodynamic force vector in geometry axes is obtained by integrating surface pressure over all panels:

$$\mathbf{F} = \sum_{i=1}^{N} -C_{p, i} \, q_{\infty} \, A_i \, \hat{\mathbf{n}}_i$$

where $q_{\infty} = \frac{1}{2} \rho_{\infty} V_{\infty}^2$.

The aerodynamic coefficients in stability/wind axes are:

$$C_L = \frac{\mathbf{F} \cdot \hat{\mathbf{l}}_{\text{wind}}}{q_{\infty} S_{\text{ref}}}, \quad C_Y = \frac{\mathbf{F} \cdot \hat{\mathbf{y}}_{\text{wind}}}{q_{\infty} S_{\text{ref}}}, \quad C_{D, \text{pressure}} = \frac{\mathbf{F} \cdot \hat{\mathbf{u}}_{\text{wind}}}{q_{\infty} S_{\text{ref}}}$$

where $\hat{\mathbf{u}}_{\text{wind}} = [\cos\alpha \cos\beta, -\sin\beta, \sin\alpha \cos\beta]^T$, $\hat{\mathbf{l}}_{\text{wind}} = [-\sin\alpha, 0, \cos\alpha]^T$, and $\hat{\mathbf{y}}_{\text{wind}} = \hat{\mathbf{l}}_{\text{wind}} \times \hat{\mathbf{u}}_{\text{wind}}$.

### Moments

The total aerodynamic moment vector about the reference point $\mathbf{r}_{\text{ref}}$ in geometry axes is:

$$\mathbf{M} = \sum_{i=1}^{N} (\mathbf{c}_i - \mathbf{r}_{\text{ref}}) \times \left(-C_{p, i} \, q_{\infty} \, A_i \, \hat{\mathbf{n}}_i\right)$$

Moments follow the standard flight-dynamics body axis convention (AVL style):

$$C_l = -\frac{M_x}{q_{\infty} S_{\text{ref}} b_{\text{ref}}} \quad (\text{right wing down positive})$$

$$C_m = +\frac{M_y}{q_{\infty} S_{\text{ref}} c_{\text{ref}}} \quad (\text{nose up positive})$$

$$C_n = -\frac{M_z}{q_{\infty} S_{\text{ref}} b_{\text{ref}}} \quad (\text{nose right positive})$$

### Far-Field Trefftz Plane Induced Drag

Near-field pressure drag integration on coarse meshes can suffer from numerical discretization noise. Ventorum-3DP also computes the induced drag in the downstream Trefftz plane.

Each wake strip $s$ carries circulation $\Gamma_s = \mu_{w, s}$. The wake shedding edges form a line of trailing vortex filaments in the Trefftz plane. The downwash $w_{n, s}$ induced at the midpoint of each strip segment is evaluated using the 2D Biot-Savart law.

The far-field induced drag is:

$$D_i = -\frac{1}{2} \rho_{\infty} \sum_{s=1}^{N_{\text{strips}}} \Gamma_s \, w_{n, s} \, \Delta y_s$$

$$C_{Di, \text{trefftz}} = \frac{D_i}{q_{\infty} S_{\text{ref}}}$$

## Published Sources

1. Morino, L. and Kuo, C. C., 1974, "Subsonic potential aerodynamics for complex configurations: A general theory", AIAA Journal, Vol. 12, No. 2, pp. 191-197.
2. Maskew, B., 1987, "Program VSAERO Theory Document: A Computer Program for Calculating the Non-Linear Aerodynamic Characteristics of Arbitrary Configurations", NASA Contractor Report 4023.
3. Katz, J. and Plotkin, A., 2001, "Low-Speed Aerodynamics", 2nd ed., Cambridge University Press, Section 9.4 and Chapter 12.
4. Drela, M., 2014, "Flight Vehicle Aerodynamics", MIT Press, Chapter 5.
5. Stevens, B. L., Lewis, F. L., and Johnson, E. N., 2016, "Aircraft Control and Simulation", 3rd ed., Wiley, Chapter 2.
