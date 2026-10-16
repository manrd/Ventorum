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

Ventorum-3DP uses the internal Dirichlet boundary condition introduced by Morino and Kuo (1974). A fictitious flow field is defined inside the closed body. The internal perturbation potential is set to zero throughout the interior:

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

### Wake Geometry

Each lifting surface sheds a flat vortex sheet from its trailing edge. The wake is discretized into strips of quadrilateral panels, one strip per trailing-edge panel pair. Each strip contains $n_{\text{wake}}$ panels ($n_{\text{wake}} = 1$ by default). The default wake length is 50 times the largest streamwise chord of the mesh: the largest x extent of a section, where a section is the set of nodes of one component in the station plane of a trailing-edge node (`ventorum.panel3d.wake.largest_streamwise_chord`). The option `wake_length` sets another length in m.

In free air the wake follows the body $+x$ axis. It does not rotate with angle of attack $\alpha$ or sideslip angle $\beta$, so the aerodynamic influence matrix does not depend on the flight condition and one factorization serves a whole sweep.

In ground effect the wake follows the free stream, parallel to the ground (owner decision). The matrix depends on the attitude in ground effect anyway (see below), so this costs no extra factorization, and no wake panel goes below the ground plane: every wake panel keeps the height of its trailing edge. With an explicit ground plane that is not parallel to the free stream, the wake follows the projection of the free stream on the ground plane. The solver checks that the body and the wake stay above the ground. Far from the ground the image solve therefore tends to a free-air solve with a free-stream wake, not to the body-axis free-air solve; the difference is the effect of the wake direction.

### Morino Kutta Condition

The Kutta condition enforces finite velocity at the trailing edge. In the Morino potential formulation, this condition sets the doublet strength of each wake strip $\mu_w$ equal to the difference between the upper and lower surface doublet strengths at the trailing edge:

$$\mu_w = \mu_{\text{upper}} - \mu_{\text{lower}}$$

All panels along the same streamwise wake strip carry this identical doublet strength $\mu_w$.

Substituting the Kutta condition into the boundary integral equation incorporates the wake influence directly into the body doublet influence matrix:

$$A_{ij} = C_{ij} + \sum_{s \in \mathcal{W}_j} \pm C_{i, s}^w$$

where $+C_{i, s}^w$ adds to the upper trailing-edge panel column and $-C_{i, s}^w$ adds to the lower trailing-edge panel column. The final linear system is:

$$\mathbf{A} \boldsymbol{\mu} = -\mathbf{B} \boldsymbol{\sigma}$$

### Pressure Kutta Condition

The Morino condition leaves a pressure jump at a thick trailing edge. The pressure Kutta condition removes it by adjusting the wake strip strengths. The option `kutta="pressure"` in `PanelSolver` selects it. The default `kutta="morino"` keeps the linear result.

The unknowns are the wake strip strength corrections $\delta_s$. The residual at each trailing-edge panel pair is the pressure jump:

$$R_s = C_{p, \text{upper}(s)} - C_{p, \text{lower}(s)}$$

The iteration starts from the Morino solution ($\boldsymbol{\delta} = 0$). Each Newton step solves:

$$\mathbf{J} \, \Delta\boldsymbol{\delta} = -\mathbf{R}$$

Each wake strength enters the solved linear system on the right-hand side only. The sensitivity of the body doublets to correction $t$ reuses the factored body matrix with one triangular solve per wake strip:

$$\mathbf{z}_t = \mathbf{A}^{-1} (-\mathbf{D}_{:, t})$$

where $\mathbf{D}_{:, t}$ sums the wake doublet influences of strip $t$ (and of its images). The surface pressure depends on the square of the surface speed, so the Jacobian entry uses the current surface velocity $\mathbf{V}_i$ and the surface gradient $\mathbf{g}_{t, i}$ of the sensitivity:

$$J_{s, t} = \frac{2 \left( \mathbf{V}_{\text{upper}(s)} \cdot \mathbf{g}_{t, \text{upper}(s)} - \mathbf{V}_{\text{lower}(s)} \cdot \mathbf{g}_{t, \text{lower}(s)} \right)}{V_{\infty}^2}$$

The surface gradient is a sparse linear operator (see below), so all columns $\mathbf{g}_t$ come from one sparse product. The solver takes at most 20 Newton steps with backtracking (at most five halvings of the step) and stops when $\max |R_s| < 10^{-6}$. The result reports the step count and the final residual. Non-convergence gives a `RuntimeWarning` and a trust note in the result, not an exception. Broyden rank updates are allowed after the first exact Jacobian; the present code recomputes the exact Jacobian at each step.

With symmetry folding the residual uses the strips of the full wake:
* Symmetric condition on a half mesh: one correction per right strip, which also applies to its mirror strip; the sensitivities come from the symmetric system.
* Asymmetric condition (sideslip, roll rate, yaw rate) on a half mesh: one correction per strip of the full wake. A correction on one side is half a symmetric and half an antisymmetric correction, so its sensitivity is $\frac{1}{2}(\mathbf{z}^{\text{sym}}_t \pm \mathbf{z}^{\text{asym}}_t)$ on the two halves. The symmetric and the antisymmetric factorizations are reused; no new factorization is needed.

The Trefftz drag of a pressure Kutta result uses the corrected wake strengths $\mu_w = \mu_{\text{upper}} - \mu_{\text{lower}} + \delta_s$.

### Finite (blunt) Trailing Edge

The mesh builder closes an open trailing edge by default (`trailing_edge="close"` of `mesh_lifting_surface`). With the trailing-edge gap $g$ (in chord units) of a section, the upper and the lower ordinates move toward the mid point of the gap:

$$z_u' = z_u - \tfrac{1}{2} g \, x^4, \qquad z_l' = z_l + \tfrac{1}{2} g \, x^4$$

The camber line does not move, the thickness changes by at most $g$, and for the NACA 4-digit and 5-digit thickness law the result is the published closed trailing edge (the $x^4$ coefficient $-0.1036$ instead of $-0.1015$; Ladson, Brooks, Hill and Sproles, 1996).

The reason is the Kutta condition. Maskew (1987, sections 2.2 and 3.1.4) states that the wake doublet implies the Kutta condition at a trailing edge, that any panel edge may shed a wake, and that the user must choose a physically realistic wake model. With constant-strength panels and the internal Dirichlet condition, a solid base between the two trailing-edge corners has no fine-mesh limit that satisfies the Kutta condition at both corners:

* With the wake shed from the upper corner and the Morino condition, the circulation flows around the base from the lower corner once the trailing-edge panels become shorter than the base height. The Kutta point is then in effect at the upper corner, and the lift falls with chordwise refinement toward a limit well below the lift of the closed section (`tests/test_panel3d_refinement.py`).
* The wake from the lower corner, from the base centre or from both corners, a base without its Dirichlet equation (doublet tied to the trailing-edge doublets), and a base with an outflow source give the same circulation: the potential outside the region behind the base does not change when the wake moves inside that region, and the outflow changes only the thickness part of the solution.
* An explicit Kutta condition on the trailing-edge panels (equal chordwise velocity, Maskew 1987 section 2.2) or the pressure Kutta condition make the lift rise without limit: the flow around both corners into the base is singular.

The two-dimensional gap model of Drela (1989), used by Ventorum-2D, avoids this with a gap panel that is not a wall. It has no counterpart in the constant-strength Dirichlet method that converged in these tests, so the 3D panel mesh closes the gap. The option `trailing_edge="base"` keeps the former blunt base (base panels, wake from the upper corner) for comparisons. The base drag of a separated blunt trailing edge is a viscous effect that the potential solution does not give in either case.

## Six Unit-Onset Superposition

The aerodynamic influence matrix $\mathbf{A}$ and source influence matrix $\mathbf{B}$ depend only on the surface and wake geometry. They do not depend on the flight condition.

Ventorum-3DP factors matrix $\mathbf{A}$ by LU decomposition:

$$\mathbf{P} \mathbf{A} = \mathbf{L} \mathbf{U}$$

The solver computes six unit onset solutions:
1. Unit forward speed $u = 1$: $\mathbf{V}^{(u)} = [1, 0, 0]^T$
2. Unit lateral speed $v = 1$: $\mathbf{V}^{(v)} = [0, 1, 0]^T$
3. Unit vertical speed $w = 1$: $\mathbf{V}^{(w)} = [0, 0, 1]^T$
4. Unit roll rate $p = 1$: $\mathbf{V}^{(p)} = [0, -(z - z_{\text{ref}}), y - y_{\text{ref}}]^T$
5. Unit pitch rate $q = 1$: $\mathbf{V}^{(q)} = [-(z - z_{\text{ref}}), 0, x - x_{\text{ref}}]^T$
6. Unit yaw rate $r = 1$: $\mathbf{V}^{(r)} = [-(y - y_{\text{ref}}), x - x_{\text{ref}}, 0]^T$

The unit right-hand sides $\mathbf{b}^{(k)} = -\mathbf{B} \boldsymbol{\sigma}^{(k)}$ are solved with the stored LU factors:

$$\boldsymbol{\mu}^{(k)} = \mathbf{A}^{-1} \mathbf{b}^{(k)}, \quad k \in \{u, v, w, p, q, r\}$$

For any arbitrary flight condition with speed $V_{\infty}$, angles $(\alpha, \beta)$, and rotation rates $(p, q, r)$, the translation velocity components in geometry axes are:

$$u = V_{\infty} \cos\alpha \cos\beta, \quad v = -V_{\infty} \sin\beta, \quad w = V_{\infty} \sin\alpha \cos\beta$$

The doublet and source strengths for the condition are obtained by linear superposition:

$$\boldsymbol{\mu} = u \boldsymbol{\mu}^{(u)} + v \boldsymbol{\mu}^{(v)} + w \boldsymbol{\mu}^{(w)} + p \boldsymbol{\mu}^{(p)} + q \boldsymbol{\mu}^{(q)} + r \boldsymbol{\mu}^{(r)}$$

$$\boldsymbol{\sigma} = u \boldsymbol{\sigma}^{(u)} + v \boldsymbol{\sigma}^{(v)} + w \boldsymbol{\sigma}^{(w)} + p \boldsymbol{\sigma}^{(p)} + q \boldsymbol{\sigma}^{(q)} + r \boldsymbol{\sigma}^{(r)}$$

The doublets (and so the circulation of each wake strip and the Kutta-Joukowski lift of the wake) are linear in the onset components. The pressure-integrated coefficients are not: $C_p$ is quadratic in the surface velocity.

### Configurations and factorization count

A configuration is the wake and the image system of a group of conditions: all free-air conditions form one configuration; in ground effect each (attitude, height) group is one configuration. The construction of `PanelSolver` builds the geometry only. A system is assembled and factorized when a condition first needs it, and each factorization is counted in one place (`PanelSolver.factorization_count`). A free-air sweep on a full mesh needs one factorization. On a half mesh the antisymmetric system is factorized only when a condition has sideslip, roll rate or yaw rate.

## Surface Velocity and Pressure Coefficient

### Tangential Perturbation Velocity

On the outer surface of a closed body with zero internal perturbation potential, the tangential perturbation velocity equals the negative surface gradient of the doublet strength:

$$\mathbf{v}_{\text{pert}, t} = -\nabla_s \mu$$

The surface gradient is a sparse linear operator $\mathbf{G}$ built once per mesh (`build_surface_gradient_operator`), so $\nabla_s \mu = \mathbf{G} \boldsymbol{\mu}$ for any doublet vector.

1. Node groups. At each corner node $v$ of panel $i$, the group is the set of panels around $v$ that panel $i$ reaches through edges at $v$ that are not cut. Wake shedding edges are cut, so the gradient never differentiates across the doublet jump of the wake. Sharp edges are cut too: in a mesh with panel kinds (surface, tip, base, side), an edge between panels of different kinds; in a mesh with one kind, an edge where the panel normals differ by more than 45 deg (a method choice). The flat tip caps, the trailing-edge base panels and the control side closures meet the surface at about 90 deg, and a doublet value across such an edge does not belong to the tangent plane of the panel. The folds between surface panels stay connected, because a coarse leading edge can fold by more than 90 deg on a smooth surface. At a node of a wake edge, the group of a trailing-edge panel also excludes the panels on the other side of the wake.
2. Interior panels (every corner group closes around its node). The node value is the inverse-distance mean of the group values (Shepard, 1968; weight $1/d$ with $d$ the distance from the node to the panel centroid), and the Green-Gauss theorem on the panel polygon (Blazek, 2015, chapter 5) gives

$$\nabla_s \mu_i = \frac{1}{A_i} \sum_{e = (a, b)} \frac{\mu_a + \mu_b}{2} \, (\mathbf{p}_b - \mathbf{p}_a) \times \hat{\mathbf{n}}_i.$$

The node values smooth the doublet error of the thin triangle fans at the poles of latitude-longitude meshes, where an edge-neighbour fit is weak.
3. Border panels (a corner group is open: a wake edge, a sharp edge or a mesh border). A node mean would sit about half a panel away from the node, and a node mean is blind to a spanwise saw-tooth of the trailing-edge doublets, which the pressure Kutta iteration must see. These panels use the least-squares fit over the panels that share an edge that is not cut, in the tangent plane and anchored at the panel value:

$$\Delta \mathbf{r}_{ij, t} \cdot \nabla_s \mu_i \approx \mu_j - \mu_i, \qquad \Delta \mathbf{r}_{ij, t} = \Delta \mathbf{r}_{ij} - (\Delta \mathbf{r}_{ij} \cdot \hat{\mathbf{n}}_i) \hat{\mathbf{n}}_i.$$

The minimum-norm solution keeps the gradient in the panel plane; displacement directions below $10^{-5}$ of the largest are treated as round-off (a row of collinear centroids, as on a tip cap, gives the gradient along the row only).

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

Near-field pressure drag integration on coarse meshes can suffer from numerical discretization noise. Ventorum-3DP also computes the induced drag in the downstream Trefftz plane, the plane normal to the wake direction (the body $x$ axis in free air, the free stream in ground effect).

Each wake strip $s$ carries circulation $\Gamma_s = \mu_{w, s}$. The trailing edge of each strip is projected on the Trefftz plane; its two ends carry 2D point vortices, $+\Gamma_s$ at the end with the larger projection on $\hat{\mathbf{n}}_w \times \hat{\mathbf{d}}$ ($\hat{\mathbf{n}}_w$ the strip normal on the upper side, $\hat{\mathbf{d}}$ the wake direction) and $-\Gamma_s$ at the other end. For a horizontal surface this is the end with the larger $y$. The normal wash $w_{n, s}$ at the midpoint of each strip segment follows from the 2D Biot-Savart law. In ground effect the image vortices (the ends reflected in the ground, with opposite circulation) add their normal wash (method of images), so the induced drag falls near the ground.

The vortices have a core of radius $r_c = 0.01 \, \bar{l}$, where $\bar{l}$ is the mean length of the strip trailing edges (`TREFFTZ_CORE_FRACTION`); the squared distance in the Biot-Savart kernel is $r^2 + r_c^2$. The core only removes the singularity when a vortex lies on an evaluation point.

The far-field induced drag is:

$$D_i = -\frac{1}{2} \rho_{\infty} \sum_{s=1}^{N_{\text{strips}}} \Gamma_s \, w_{n, s} \, l_s$$

$$C_{Di, \text{trefftz}} = \frac{D_i}{q_{\infty} S_{\text{ref}}}$$

The product $w_{n, s} \, l_s$ uses the in-plane normal $\hat{\mathbf{d}} \times (\mathbf{p}_r - \mathbf{p}_l) / l_s$, so it equals the normal wash times the length of the segment projected on the Trefftz plane, also for a swept trailing edge.

## Symmetry Folding about y = 0

When a configuration is symmetric about the x-z plane, the problem can be solved on a right-half mesh (`PanelMesh.half=True`). This halves the number of unknowns $N$. The assembly computes the self and mirror influences of the half mesh, $4 (N/2)^2 = N^2$ entries for $\mathbf{B}$ and $\mathbf{C}$ instead of $2 N^2$: half the assembly work of the full mesh. One factorization of an $N/2$ system costs one eighth of a full factorization; a symmetric condition needs only the symmetric system (one eighth), and a condition with sideslip or rates needs both systems (one quarter).

The solver uses two systems on the half mesh:
1. Symmetric system: Unit onsets $u$, $w$, and $q$ produce symmetric normal velocity on the surface. Doublets and sources on the left-half mirror panels have equal strengths:

$$\mu_{\text{mirror}} = +\mu, \quad \sigma_{\text{mirror}} = +\sigma$$

The symmetric aerodynamic influence matrix and source matrix add the influence of the mirror panels:

$$\mathbf{A}^{\text{sym}} = \mathbf{A}_{\text{self}} + \mathbf{A}_{\text{mirror}}, \quad \mathbf{B}^{\text{sym}} = \mathbf{B}_{\text{self}} + \mathbf{B}_{\text{mirror}}$$

2. Antisymmetric system: Unit onsets $v$, $p$, and $r$ produce antisymmetric normal velocity on the surface. Doublets and sources on the left-half mirror panels have opposite strengths:

$$\mu_{\text{mirror}} = -\mu, \quad \sigma_{\text{mirror}} = -\sigma$$

The antisymmetric matrices subtract the mirror panel influences:

$$\mathbf{A}^{\text{asym}} = \mathbf{A}_{\text{self}} - \mathbf{A}_{\text{mirror}}, \quad \mathbf{B}^{\text{asym}} = \mathbf{B}_{\text{self}} - \mathbf{B}_{\text{mirror}}$$

Any flight condition with sideslip or asymmetric rates is evaluated by linear superposition of the symmetric and antisymmetric unit solutions. The post-processing (surface gradient, forces, Trefftz drag) runs on the reconstructed full mesh; the right-half and mirror panels have the same order, and the builder metadata of the mesh (strip index, chordwise index, side, node station) is kept.

## Ground Effect by Method of Images

Near flat ground, Ventorum-3DP models the ground plane using the classical method of images (Wieselsberger 1921; Katz and Plotkin 2001).

### Ground Plane and Image Geometry

The aircraft remains in its body geometry axes while the ground plane orientation moves with attitude $(\alpha, \beta, \phi)$. The unit normal pointing from the ground to the aircraft is $\hat{\mathbf{k}} = \text{ground\_normal}(\alpha, \beta, \phi)$. The ground plane satisfies $\mathbf{x} \cdot \hat{\mathbf{k}} = o$, where $o = \mathbf{r}_{\text{ref}} \cdot \hat{\mathbf{k}} - h$. An explicit ground plane (normal and offset) can replace `condition.h`, as for the vortex lattice.

An image body is placed below the ground plane by reflecting every panel corner across the ground:

$$\mathbf{x}' = \mathbf{x} - 2 (\mathbf{x} \cdot \hat{\mathbf{k}} - o) \hat{\mathbf{k}}$$

The loop order of the reflected panel vertices is reversed to preserve an outward-pointing unit normal on the image body: $\hat{\mathbf{n}}' = \mathcal{R} \hat{\mathbf{n}}$.

### Boundary Condition and Mirror Signs

To make the ground plane a stream surface ($\mathbf{V} \cdot \hat{\mathbf{k}} = 0$ on the ground), the image body carries:
* Equal source strength: $\sigma' = +\sigma$
* Equal doublet strength: $\mu' = +\mu$

Because the outward normal of the image panel is the mirror image of the real panel normal ($\hat{\mathbf{n}}' = \mathcal{R} \hat{\mathbf{n}}$), equal source and doublet strengths cause the normal velocities from the real and image panels to cancel exactly at the ground plane.

The wake image follows the wake: each wake panel is reflected across the ground plane with doublet strength $\mu_w' = \mu_w$. The wake follows the free stream, parallel to the ground, so the wake and its image do not cross.

### Attitude Dependence and Condition Grouping

In ground effect, the relative orientation and distance between the aircraft and the ground depend on the attitude $(\alpha, \beta, \phi)$ and height $h$, and the wake follows the free stream. Consequently, the influence matrix depends on attitude and height. Linear superposition of the six unit onsets remains valid for a fixed attitude and height.

The method `solve_conditions` groups flight conditions by configuration (ground normal, ground offset and wake direction), factorizes the system once per group, and evaluates all conditions within that group.

Any geometry or wake touching or crossing below the ground plane raises a `GroundStrikeError` (a `ValueError`).

## Combined Symmetry and Ground Effect

When solving a symmetric aircraft in ground effect on a half mesh (`PanelMesh.half=True`), four-fold images are used:
1. Real right-half body above ground: $\mathbf{A}_{\text{self}}, \mathbf{B}_{\text{self}}$
2. Y-mirror body (left wing) above ground: $\mathbf{A}_m, \mathbf{B}_m$
3. Ground-mirror body (right wing) below ground: $\mathbf{A}_g, \mathbf{B}_g$
4. YG-mirror body (left wing) below ground: $\mathbf{A}_{yg}, \mathbf{B}_{yg}$

For symmetric onsets ($u, w, q$):

$$\mathbf{A}^{\text{sym}} = \mathbf{A}_{\text{self}} + \mathbf{A}_m + \mathbf{A}_g + \mathbf{A}_{yg}, \quad \mathbf{B}^{\text{sym}} = \mathbf{B}_{\text{self}} + \mathbf{B}_m + \mathbf{B}_g + \mathbf{B}_{yg}$$

For asymmetric onsets ($v, p, r$):

$$\mathbf{A}^{\text{asym}} = \mathbf{A}_{\text{self}} - \mathbf{A}_m + \mathbf{A}_g - \mathbf{A}_{yg}, \quad \mathbf{B}^{\text{asym}} = \mathbf{B}_{\text{self}} - \mathbf{B}_m + \mathbf{B}_g - \mathbf{B}_{yg}$$

The four-fold images need a configuration that is symmetric about $y = 0$: the ground normal and the wake direction must have no $y$ component. With bank ($\phi \neq 0$) the ground normal has a $y$ component, and with sideslip the free-stream wake has one. In these cases the half-mesh solver solves the full mesh of the configuration instead (a full-mesh solver built once, when needed; its factorizations count in the factorization count) and returns the distributions of the right half. The assembly of a symmetric configuration in ground effect is again half the work of the full mesh, and its factorization one eighth.

## Root Section of a Symmetric Surface with Dihedral

The mesh builder (`ventorum.panel3d.mesh`) puts each section in the plane of the local chord direction and the local normal of the lattice surface, so the section thickness follows the dihedral. At the root of a symmetric surface the vortex lattice twists the section about the $y$ axis, so the root section lies in the plane $y = 0$ with the normal $z$: it is the cut of the two half wings at the plane of symmetry (a miter joint). The local surface of the first panel row has the normal $(0, -\sin\Gamma, \cos\Gamma)$, with $\Gamma$ the dihedral of that row. A thickness $t$ along $z$ is therefore only $t\cos\Gamma$ normal to the surface, and the half wing loses volume in its first panel row.

The option `scale_root_thickness` of `mesh_lifting_surface` and `mesh_aircraft` (setting `panel_scale_root_thickness`, default True, an owner decision) corrects this. It writes the root section as camber line plus offsets, $z_{u,l} = z_c \pm h$, and multiplies the offsets by $1/\cos\Gamma$:

$$z_{u,l}' = z_c \pm \frac{h}{\cos\Gamma},$$

with $\Gamma = \operatorname{atan2}(\Delta z, \Delta y)$ of the reference line from the root station to the next station. The thickness normal to the local surface then equals the section thickness. The camber line does not move, so the leading-edge and trailing-edge nodes stay on the lattice points.

Properties of the scaled root section:

* Symmetric section, no root twist: a root node at the offset $h$ is at the normal offset $h$ and at the span coordinate $h\tan\Gamma$ of the first panel row. The ruled panels of the row are parallel to the span axis, and the row is a prism cut by the plane $y = 0$. The volume of the half wing equals the volume of the same wing with zero dihedral to round-off (`tests/test_panel3d_mesh.py`). Without the option, a 30 deg half wing has less volume; the loss is in the first panel row, so it depends on the spanwise panel count.
* Cambered section: the miter plane cuts the cambered slab at an angle. The volume difference to the zero-dihedral wing is $-\left(\sin\Gamma + \tfrac{1}{2}\tan\Gamma\,(1 - \cos\Gamma)\right) Q$, with $Q = \int t\,z_c\,\mathrm{d}x$ the first moment of the section area about the chord line (in chord units, times the chord cubed for a volume). This is the geometry of a miter of a cambered surface, not a mesh error: an exact miter, which also scales the camber line, gives $-\tan\Gamma\,Q$. A test checks the formula on a NACA 2412 half wing.
* Root twist $\theta$: before the scaling, the thickness normal to the surface is $t\cos\Gamma/\sqrt{1 - \sin^2\theta\,\sin^2\Gamma}$. The factor $1/\cos\Gamma$ leaves an error of second order in $\theta$.
* No effect: a surface with zero dihedral at the root (the mesh stays bit for bit the same), a one-sided surface (its root section is normal to its span axis, for example a vertical fin with $\Gamma = 90$ deg) and a symmetric surface whose first panel row is vertical (the two halves coincide and there is no miter).
* A deflected control surface at the root deflects the scaled section: the hinge point at half the local thickness does not move.

## Solver Family Interface

Ventorum-3DP connects to the solver architecture of Ventorum through the class `PanelMethodSolver` in `ventorum.solvers.panel3d`. Users and automated tools can select it through the factory with `solver="panel3d"`.

### Inputs

The solver takes three standard input objects:
* `Aircraft`: One or more lifting surfaces. Each surface definition contains wing sections. The optional field `WingSection.shape` specifies the airfoil geometry (NACA designation, point array, or `AirfoilShape`). When `shape` is None, the solver defaults to NACA 0012 and reports a warning in the result notes. The functions `aircraft_to_json` and `aircraft_from_json` serialize and restore this field.
* `FlightCondition`: Free-stream velocity $V_{\infty}$, angles $\alpha$ and $\beta$, density $\rho$, optional ground height $h$, bank angle $\phi$, and angular rates $p, q, r$. The solver also accepts a sequence of flight conditions to execute batch evaluations and sweeps. The keyword `ground` (a `GroundPlane`) replaces `condition.h` for all conditions.
* `SolverSettings`: `n_panels` (spanwise panels per semi-span, as the lattice), `n_chord` (chordwise panels per surface, 20 when None), and the panel fields `panel_kutta` ("morino" or "pressure"), `panel_backend` ("auto", "numpy", "numba" or "gpu"), `panel_wake_length` (m; None gives 50 largest chords), `panel_chord_spacing` ("cosine", the default, or "uniform"), `panel_estimate_mesh_error` (False by default) and `panel_scale_root_thickness` (True by default; see the root section above). The other solvers ignore the panel fields.

Symmetry plane folding and ground images are used automatically when the geometry and the flight conditions allow them (no sideslip, roll or yaw rate, bank, or deflected antisymmetric control).

Backend "auto" follows the device and precision selection of `ventorum.gpu`, as the lattice solvers do: the device setting "cpu" keeps the solve on the CPU, a CPU worker thread stays on the CPU, and otherwise `ventorum.gpu.use_gpu` decides with the work of the dense systems ($N^2$ influence evaluations per configuration, twice in ground effect). The panel method has no cost model of its own yet, so the decision uses the thresholds of the vortex lattice. On the GPU the precision is `ventorum.gpu.get_precision()`; on the CPU the numpy reference kernels are used.

### Results

The solve returns a standard `SolverResult`:
* `totals`: Integrated aerodynamic coefficients ($C_L, C_D, C_Y, C_l, C_m, C_n$). The reported induced drag $C_{Di}$ comes from the far-field Trefftz plane. The surface force integration provides near-field pressure drag as $C_{Di,\text{nearfield}}$.
* `spanwise`: Spanwise distributions on each lifting surface. Sectional coefficients ($C_l, C_{d,\text{pressure}}, C_m$) are computed by strip integration of panel pressure forces at the lattice strip stations. Each panel belongs to the lattice strip whose station planes (normal to the span in the y-z plane) enclose its centroid; a mesh with controls (extra stations, side closures) maps every panel once. When the mesh carries the strip index of the mesh builder and its strips are the lattice strips, the map uses it. Strip lift sums to the total aircraft $C_L$.
* `details["cp"]`: The surface pressure coefficients of the solved mesh panels.
* `details["panel"]`: The panel options used (Kutta option, chordwise spacing and count, wake length in m, half mesh, root-thickness scaling, and the mesh choice `trailing_edge`).
* `totals.trust`: The physical validity trust score. For the 3D panel method, the score evaluates the mesh (`ventorum.core.trust.panel3d_mesh_checks`) and the ground clearance. The factor `mesh_resolution` gives the discretization error from the difference of the near-field pressure drag and the far-field Trefftz induced drag (and from the chordwise panel count when the caller gives it): a warning for a probable error above 2 % in CL or 4 % in CDi, a strong warning above 5 % in CL or 10 % in CDi. The factor `mesh_adequacy` gives the panel shape and validity from `check_mesh`: a percentile of the panel aspect ratio and skew, the warp, the closure, and folded (non-convex) panels. A mesh-error study of the method calibrated the thresholds; the [trust score chapter](trust_score.md) (section *3D panel method*) gives the method and the limits. With `panel_estimate_mesh_error`, the notes add the mesh error estimate of the given mesh (see the mesh convergence section of the Ventorum-3DP chapter).
* `details["dispatch"]`: The execution dispatch record: the selected backend, device, arithmetic precision, the reason of the selection, and the counted number of factorizations.

## Published Sources

1. Morino, L. and Kuo, C. C., 1974, "Subsonic potential aerodynamics for complex configurations: A general theory", AIAA Journal, Vol. 12, No. 2, pp. 191-197.
2. Maskew, B., 1987, "Program VSAERO Theory Document: A Computer Program for Calculating the Non-Linear Aerodynamic Characteristics of Arbitrary Configurations", NASA Contractor Report 4023.
3. Katz, J. and Plotkin, A., 2001, "Low-Speed Aerodynamics", 2nd ed., Cambridge University Press, Section 9.4 and Chapter 12.
4. Wieselsberger, C., 1921, "Ueber den Fluegelwiderstand in der Naehe des Bodens", Zeitschrift fuer Flugtechnik und Motorluftschiffahrt, Vol. 12, No. 10; English translation: "Wing resistance near the ground", NACA TM 77, 1922.
5. Drela, M., 2014, "Flight Vehicle Aerodynamics", MIT Press, Chapter 5.
6. Stevens, B. L., Lewis, F. L., and Johnson, E. N., 2016, "Aircraft Control and Simulation", 3rd ed., Wiley, Chapter 2.
7. Kinnas, S. A. and Hsin, C. Y., 1992, "Boundary element method for the analysis of the unsteady flow around extreme propeller geometries", AIAA Journal, Vol. 30, No. 3 (iterative pressure Kutta).
8. Shepard, D., 1968, "A two-dimensional interpolation function for irregularly-spaced data", Proceedings of the 23rd ACM National Conference, pp. 517-524 (inverse distance weights).
9. Blazek, J., 2015, "Computational Fluid Dynamics: Principles and Applications", 3rd ed., Butterworth-Heinemann, Chapter 5 (Green-Gauss gradient).
10. Ladson, C. L., Brooks, C. W., Hill, A. S. and Sproles, D. W., 1996, "Computer Program To Obtain Ordinates for NACA Airfoils", NASA TM-4741 (closed trailing edge of the NACA 4-digit thickness law).
11. Drela, M., 1989, "XFOIL: An analysis and design system for low Reynolds number airfoils", Lecture Notes in Engineering 54, Springer, pp. 1-12 (trailing-edge gap panel of the two-dimensional method).
