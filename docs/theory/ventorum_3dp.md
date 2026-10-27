---
orphan: true
---

# Ventorum-3DP: Three-dimensional potential flow panel method

## Model

Ventorum-3DP solves steady, incompressible, inviscid potential flow around arbitrary three-dimensional lifting configurations and non-lifting bodies. The solver uses a low-order boundary element formulation with flat quadrilateral and triangular panels carrying constant distributions of sources and doublets (Morino and Kuo, 1974; Hess and Smith, 1967; Katz and Plotkin, 2001). The formulation enforces an internal Dirichlet boundary condition of zero perturbation potential inside the closed body. The linear aerodynamic influence matrix is factorized once per configuration (once for all free-air conditions) by LU decomposition and solved for six fundamental unit onset flow states (three translation velocities and three angular rotation rates). Flight conditions at arbitrary angles of attack, sideslip, and rotation rates follow by linear superposition. The method supports Morino and iterative non-linear pressure Kutta conditions, planar symmetry folding, ground effect via the method of images, near-field pressure integration, far-field Trefftz-plane induced drag calculation, grid convergence analysis, and multi-backend execution on CPU (NumPy, Numba) and CUDA GPU (NVIDIA Warp, PyTorch).

## Influence kernels

### Panel geometry and coordinate transformations

A panel is defined by four ordered vertex coordinates $\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3, \mathbf{p}_4$ [m] in three-dimensional Cartesian space. Triangular panels repeat the third vertex at the fourth entry ($\mathbf{p}_4 = \mathbf{p}_3$).

Diagonal vectors define the raw normal:

```{math}
\mathbf{d}_{13} = \mathbf{p}_3 - \mathbf{p}_1, \qquad \mathbf{d}_{24} = \mathbf{p}_4 - \mathbf{p}_2.
```

The outward unit normal vector points into the fluid:

```{math}
\mathbf{n}_{\text{raw}} = \mathbf{d}_{13} \times \mathbf{d}_{24}, \qquad \hat{\mathbf{n}} = \frac{\mathbf{n}_{\text{raw}}}{\|\mathbf{n}_{\text{raw}}\|}.
```

The projected panel area $A$ [m^2] is:

```{math}
A = \frac{1}{2} \|\mathbf{n}_{\text{raw}}\|.
```

Non-planar quadrilaterals are projected onto the mean plane passing through the vertex average $\mathbf{p}_{\text{avg}} = \frac{1}{4} \sum_{k=1}^4 \mathbf{p}_k$:

```{math}
\mathbf{p}'_k = \mathbf{p}_k - \left[(\mathbf{p}_k - \mathbf{p}_{\text{avg}}) \cdot \hat{\mathbf{n}}\right] \hat{\mathbf{n}}.
```

The panel centroid $\mathbf{p}_0$ is the mean of the corner points (for a quadrilateral that is not a parallelogram, this is not the area centroid):
- For quadrilaterals: $\mathbf{p}_0 = \frac{1}{4} \sum_{k=1}^4 \mathbf{p}'_k$.
- For triangles: $\mathbf{p}_0 = \frac{1}{3} (\mathbf{p}'_1 + \mathbf{p}'_2 + \mathbf{p}'_3)$.

An orthonormal local reference frame $(\hat{\mathbf{l}}, \hat{\mathbf{m}}, \hat{\mathbf{n}})$ is defined on the panel:

```{math}
\hat{\mathbf{l}} = \frac{\mathbf{p}'_2 - \mathbf{p}'_1}{\|\mathbf{p}'_2 - \mathbf{p}'_1\|}, \qquad \hat{\mathbf{m}} = \hat{\mathbf{n}} \times \hat{\mathbf{l}}.
```

For any target field point $\mathbf{x} \in \mathbb{R}^3$, the local coordinates relative to the panel centroid are:

```{math}
x = (\mathbf{x} - \mathbf{p}_0) \cdot \hat{\mathbf{l}}, \qquad y = (\mathbf{x} - \mathbf{p}_0) \cdot \hat{\mathbf{m}}, \qquad z = (\mathbf{x} - \mathbf{p}_0) \cdot \hat{\mathbf{n}}.
```

Corner coordinates in the local plane are denoted $(x_k, y_k, 0)$ for $k = 1, \dots, 4$. The distance from the target point to corner $k$ is:

```{math}
r_k = \sqrt{(x - x_k)^2 + (y - y_k)^2 + z^2}.
```

For edge $k$ connecting corner $(x_k, y_k)$ to $(x_{k+1}, y_{k+1})$ with length $d_k = \sqrt{(x_{k+1} - x_k)^2 + (y_{k+1} - y_k)^2}$, the in-plane perpendicular distance is:

```{math}
r_{\perp, k} = \frac{(x - x_k)(y_{k+1} - y_k) - (y - y_k)(x_{k+1} - x_k)}{d_k}.
```

The edge logarithmic integral is:

```{math}
L_k = \ln\left(\frac{r_k + r_{k+1} + d_k}{r_k + r_{k+1} - d_k}\right).
```

When the target point lies on the edge segment ($r_k + r_{k+1} - d_k \le 0$), $L_k$ is set to 0.

### Exact singularity formulas

1. Solid angle:
The panel is split into two planar triangles: $(0, 1, 2)$ and $(0, 2, 3)$. For each triangle with vertices $\mathbf{a}, \mathbf{b}, \mathbf{c}$ relative to the target point, the solid angle follows from the van Oosterom and Strackee (1983) formula:

```{math}
\tan\left(\frac{\Omega_\Delta}{2}\right) = \frac{\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})}{\|\mathbf{a}\| \|\mathbf{b}\| \|\mathbf{c}\| + (\mathbf{a}\cdot\mathbf{b})\|\mathbf{c}\| + (\mathbf{a}\cdot\mathbf{c})\|\mathbf{b}\| + (\mathbf{b}\cdot\mathbf{c})\|\mathbf{a}\|}.
```

The total panel solid angle is $\Omega = \Omega_{\Delta, 1} + \Omega_{\Delta, 2}$.

2. Constant-strength doublet potential and velocity:
The doublet velocity potential per unit strength is:

```{math}
\Phi_D = \frac{\Omega}{4 \pi}.
```

Approaching the panel plane from the interior side ($z \to 0^-$) at the centroid gives the self-influence value $C_{ii} = +1/2$.
By Stokes theorem, the velocity induced by a constant-strength doublet panel equals the velocity of an equivalent vortex ring of unit circulation along the panel perimeter:

```{math}
\mathbf{V}_D = -\sum_{k=1}^4 \mathbf{v}_{\text{segment}}(\mathbf{p}'_k \to \mathbf{p}'_{k+1}),
```

where each straight vortex segment velocity is evaluated using the Biot-Savart formula.

3. Constant-strength source potential and velocity:
The source velocity potential per unit strength is:

```{math}
\Phi_S = \frac{1}{4 \pi} \left[ \sum_{k=1}^4 r_{\perp, k} L_k - z \Omega \right].
```

In local panel coordinates, the source-induced velocity components are:

```{math}
V_{S, x} = +\frac{1}{4 \pi} \sum_{k=1}^4 \frac{y_{k+1} - y_k}{d_k} L_k, \qquad
V_{S, y} = -\frac{1}{4 \pi} \sum_{k=1}^4 \frac{x_{k+1} - x_k}{d_k} L_k, \qquad
V_{S, z} = -\frac{1}{4 \pi} \Omega = -\Phi_D,
```

with $L_k = \ln\left((r_k + r_{k+1} + d_k)/(r_k + r_{k+1} - d_k)\right)$, where $r_k$ and $r_{k+1}$ are the distances from the target to the two ends of edge $k$ and $d_k$ is the edge length. Each velocity is $\mathbf{V} = +\nabla \Phi$, so the far-field velocities below are the gradients of the far-field potentials.

Transforming to global coordinates yields:

```{math}
\mathbf{V}_S = V_{S, x} \hat{\mathbf{l}} + V_{S, y} \hat{\mathbf{m}} + V_{S, z} \hat{\mathbf{n}}.
```

### Far-field point expansions

When the target distance $r = \|\mathbf{x} - \mathbf{p}_0\|$ exceeds `far_ratio` times the panel diameter $D = \max_{i < j} \|\mathbf{p}_i - \mathbf{p}_j\|$, point-singularity expansions are used:

```{math}
\Phi_S^{\text{far}} = -\frac{A}{4 \pi r}, \qquad \mathbf{V}_S^{\text{far}} = +\frac{A \mathbf{r}}{4 \pi r^3},
```

```{math}
\Phi_D^{\text{far}} = -\frac{A (\mathbf{r} \cdot \hat{\mathbf{n}})}{4 \pi r^3}, \qquad \mathbf{V}_D^{\text{far}} = -\frac{A}{4 \pi} \left[\frac{\hat{\mathbf{n}}}{r^3} - \frac{3 (\mathbf{r} \cdot \hat{\mathbf{n}}) \mathbf{r}}{r^5}\right].
```

Setting `far_ratio = np.inf` disables approximations and computes exact integrals everywhere.

## Surface meshes and geometry

### PanelMesh representation

The class `PanelMesh` represents the surface of one or more thick lifting surfaces. The mesh is composed of:
- `nodes`: Array of 3D node coordinates [m] of shape $(N_n, 3)$.
- `panels`: Array of four corner node indices of shape $(N_p, 4)$.
- `component`: Integer component index for each panel.
- `kind`: Array marking panels as `"surface"`, `"tip"`, or `"base"`.
- `wake_edges`: Array of wake shedding edge descriptors of shape $(N_w, 4)$ storing upper panel index, lower panel index, and the two trailing-edge node indices.
- `half`: Boolean flag indicating whether the mesh represents a right-half geometry open at $y = 0$.

### Wing meshing and trailing-edge treatment

Each lifting surface is discretized into spanwise strips. At each span station, the wing section carries its airfoil coordinates. The upper and lower surfaces are covered with quadrilateral panels. An open trailing edge (gap above the tolerance $10^{-6} c$) is closed by default: the upper and the lower ordinates move toward the mid point of the gap $g$ by $\tfrac{1}{2} g x^4$, which for the NACA 4-digit thickness law is the published closed trailing edge (Ladson et al., 1996). With a solid base between the two trailing-edge corners the Morino Kutta condition puts the Kutta point at the wake corner on fine chordwise meshes, and the lift does not converge to the lift of the closed section (see the [panel solver chapter](panel3d.md), section *Finite (blunt) Trailing Edge*). The option `trailing_edge="base"` keeps flat base panels and the wake at the upper corner. Flat tip caps close the wing tips to guarantee a watertight enclosed volume.

The section planes are normal to the span, except the root section of a symmetric surface (the plane $y = 0$). Where a narrow strip next to the root or a dihedral break would let two neighbour sections cross, the fold guard turns the plane of the outboard station toward the plane of its inboard neighbour until every node keeps a margin from the other plane (see the panel solver chapter, section *Fold guard*).

### Deflected plain control surfaces

Plain control surfaces deflect the section geometry inside the specified span range using `ventorum.airfoil.flap.deflect_plain_flap`. Positive deflection rotates the trailing edge downward. Chordwise node distributions insert a node at the hinge line, preserving the geometric corner. Trailing-edge points rotate about the hinge line to floating-point roundoff.

At a lateral spanwise boundary of the control surface that borders an undeflected part, the deflection changes linearly over a transition of 0.2 local chords centred on the boundary (default `control_edge="transition"`), so the surface stays continuous. With `control_edge="gap"`, and at a boundary between two deflected controls, the mesh inserts two spanwise stations at the same span station $y$, one deflected and one undeflected, and flat side panels close the gap between the two curves. The gap closure leaves a window of fluid between the two outlines near the trailing edge, and the flapped wing does not converge with spanwise refinement; the [controls page](panel3d_controls.md) gives the reasons. Neither closure models gap leakage flow.

### Mesh validation

The function `check_mesh` inspects the closed mesh for geometric quality:
- Panel aspect ratio: $AR = \max(\text{edge}) / \min(\text{edge})$.
- Panel skewness: angle between panel diagonal vectors.
- Panel non-planarity (warp): perpendicular distance of vertices from the mean panel plane.
- Enclosed signed volume: computed by Gauss divergence theorem to verify consistent outward normal orientation.
- Folded strips: a node of a station row on the wrong side of the section plane of its neighbour row (needs the station rows of the mesh builder). Folded panels count as non-convex panels.

## Dirichlet formulation and system assembly

### Boundary integral equation

Ventorum-3DP sets the fictitious internal perturbation potential to zero throughout the interior of each closed body ($\phi_{\text{int}} = 0$). By Green's third identity, the perturbation potential at collocation point $\mathbf{c}_i$ approaches the internal boundary:

```{math}
\phi_{\text{int}}(\mathbf{c}_i) = \sum_{j=1}^{N} \mu_j C_{ij} + \sum_{k=1}^{N_w} \mu_{w, k} C_{ik}^w + \sum_{j=1}^{N} \sigma_j B_{ij} = 0,
```

where $C_{ij}$ is the doublet influence coefficient of body panel $j$, $C_{ik}^w$ is the doublet influence of wake panel $k$, and $B_{ij}$ is the source influence coefficient of body panel $j$.

The external flow tangency condition requires:

```{math}
\frac{\partial \phi}{\partial n} = -\mathbf{V}_{\text{onset}} \cdot \hat{\mathbf{n}}.
```

Because $\phi_{\text{int}} = 0$, the jump in normal velocity across the boundary equals the source strength:

```{math}
\sigma_j = -\mathbf{V}_{\text{onset}}(\mathbf{c}_j) \cdot \hat{\mathbf{n}}_j.
```

The source strengths $\sigma_j$ are known directly from the surface geometry and onset flow.

### Wake model and Kutta options

Lifting surfaces shed flat vortex sheets from their trailing edges, one strip of panels per trailing-edge panel pair. The default wake length is 50 times the largest streamwise chord of the mesh (the largest x extent of a section). In free air the wake panels extend downstream along the body $+x$ axis, so the influence matrix does not change with the flight condition and one factorization serves a whole sweep. In ground effect the wake follows the free stream, parallel to the ground: the matrix depends on the attitude in ground effect anyway, and no wake panel goes below the ground plane.

1. Morino Kutta condition (`kutta="morino"`):
The doublet strength of each wake strip $\mu_w$ equals the difference between upper and lower surface doublet strengths at the trailing edge:

```{math}
\mu_w = \mu_{\text{upper}} - \mu_{\text{lower}}.
```

The wake influences fold directly into the body doublet matrix:

```{math}
A_{ij} = C_{ij} + \sum_{s \in \mathcal{W}_j} \pm C_{i, s}^w.
```

This yields the linear system:

```{math}
\mathbf{A} \boldsymbol{\mu} = -\mathbf{B} \boldsymbol{\sigma}.
```

2. Iterative pressure Kutta condition (`kutta="pressure"`):
At thick trailing edges, the Morino condition can leave a residual pressure jump. The pressure Kutta option computes wake strip correction doublets $\delta_s$ to eliminate the trailing-edge pressure jump:

```{math}
R_s = C_{p, \text{upper}(s)} - C_{p, \text{lower}(s)} = 0.
```

Starting from the Morino solution ($\boldsymbol{\delta} = 0$), a Newton-Raphson iteration updates $\boldsymbol{\delta}$:

```{math}
\mathbf{J} \, \Delta\boldsymbol{\delta} = -\mathbf{R}.
```

The sensitivity of body doublets to wake corrections reuses the factored body matrix $\mathbf{A}$:

```{math}
\mathbf{z}_t = \mathbf{A}^{-1} (-\mathbf{D}_{:, t}),
```

where $\mathbf{D}_{:, t}$ is the wake doublet influence of strip $t$. The exact Jacobian matrix is:

```{math}
J_{s, t} = \frac{2 \left(\mathbf{V}_{\text{upper}(s)} \cdot \mathbf{g}_{t, \text{upper}(s)} - \mathbf{V}_{\text{lower}(s)} \cdot \mathbf{g}_{t, \text{lower}(s)}\right)}{V_\infty^2},
```

where $\mathbf{g}_{t}$ is the surface gradient of sensitivity $\mathbf{z}_t$ (one sparse product for all columns). The iteration uses line-search backtracking and converges when $\max |R_s| < 10^{-6}$ (at most 20 steps). On a half mesh, a symmetric condition has one correction per right strip (it also applies to the mirror strip); an asymmetric condition (sideslip, roll or yaw rate) has one correction per strip of the full wake, with the sensitivities $\frac{1}{2}(\mathbf{z}^{\text{sym}}_t \pm \mathbf{z}^{\text{asym}}_t)$ from the symmetric and the antisymmetric factorizations. No new factorization is needed.

## Six unit-onset superposition

Influence matrices $\mathbf{A}$ and $\mathbf{B}$ depend only on the geometry of the body, the wake and the images. Matrix $\mathbf{A}$ of each configuration is factored by LU decomposition when a condition first needs it, and the solver counts each factorization (`PanelSolver.factorization_count`):

```{math}
\mathbf{P} \mathbf{A} = \mathbf{L} \mathbf{U}.
```

The solver computes six unit-onset solution vectors:
1. Unit surge ($u = 1$): $\mathbf{V}^{(u)} = [1, 0, 0]^T$.
2. Unit sway ($v = 1$): $\mathbf{V}^{(v)} = [0, 1, 0]^T$.
3. Unit heave ($w = 1$): $\mathbf{V}^{(w)} = [0, 0, 1]^T$.
4. Unit roll rate ($p = 1$): $\mathbf{V}^{(p)} = [0, -(z - z_{\text{ref}}), y - y_{\text{ref}}]^T$.
5. Unit pitch rate ($q = 1$): $\mathbf{V}^{(q)} = [-(z - z_{\text{ref}}), 0, x - x_{\text{ref}}]^T$.
6. Unit yaw rate ($r = 1$): $\mathbf{V}^{(r)} = [-(y - y_{\text{ref}}), x - x_{\text{ref}}, 0]^T$.

The unit right-hand sides $\mathbf{b}^{(k)} = -\mathbf{B} \boldsymbol{\sigma}^{(k)}$ are solved simultaneously:

```{math}
\boldsymbol{\mu}^{(k)} = \mathbf{A}^{-1} \mathbf{b}^{(k)}, \qquad k \in \{u, v, w, p, q, r\}.
```

For any flight condition $(V_\infty, \alpha, \beta, p, q, r)$, translation velocity components in geometry axes are:

```{math}
u = V_\infty \cos\alpha \cos\beta, \qquad v = -V_\infty \sin\beta, \qquad w = V_\infty \sin\alpha \cos\beta.
```

The surface doublets and sources follow by linear superposition:

```{math}
\boldsymbol{\mu} = u \boldsymbol{\mu}^{(u)} + v \boldsymbol{\mu}^{(v)} + w \boldsymbol{\mu}^{(w)} + p \boldsymbol{\mu}^{(p)} + q \boldsymbol{\mu}^{(q)} + r \boldsymbol{\mu}^{(r)},
```

```{math}
\boldsymbol{\sigma} = u \boldsymbol{\sigma}^{(u)} + v \boldsymbol{\sigma}^{(v)} + w \boldsymbol{\sigma}^{(w)} + p \boldsymbol{\sigma}^{(p)} + q \boldsymbol{\sigma}^{(q)} + r \boldsymbol{\sigma}^{(r)}.
```

## Symmetry folding and ground effect

### Symmetry folding on half meshes

Configurations with geometric mirror symmetry about the x-z plane ($y = 0$) can be solved on a right-half mesh (`PanelMesh.half=True`). This halves the unknown count $N$. The assembly computes the self and mirror influences, $4 (N/2)^2 = N^2$ entries of $\mathbf{B}$ and $\mathbf{C}$ instead of $2 N^2$ (half the work). One factorization of a half-mesh system costs one eighth of a full factorization; a symmetric condition needs only the symmetric system, a condition with sideslip or rates needs both (one quarter of the full work).

Two systems are solved on the half mesh:
1. Symmetric system ($u, w, q$): Mirror panels carry equal strengths ($\mu' = +\mu, \sigma' = +\sigma$):

```{math}
\mathbf{A}^{\text{sym}} = \mathbf{A}_{\text{self}} + \mathbf{A}_{\text{mirror}}, \qquad \mathbf{B}^{\text{sym}} = \mathbf{B}_{\text{self}} + \mathbf{B}_{\text{mirror}}.
```

2. Antisymmetric system ($v, p, r$): Mirror panels carry opposite strengths ($\mu' = -\mu, \sigma' = -\sigma$):

```{math}
\mathbf{A}^{\text{asym}} = \mathbf{A}_{\text{self}} - \mathbf{A}_{\text{mirror}}, \qquad \mathbf{B}^{\text{asym}} = \mathbf{B}_{\text{self}} - \mathbf{B}_{\text{mirror}}.
```

Superposition of the two systems yields exact solutions for arbitrary asymmetric flight conditions.

### Ground effect by method of images

Near a flat boundary, the aircraft keeps its body geometry axes while the ground plane orientation moves with attitude $(\alpha, \beta, \phi)$. The unit normal pointing from the ground to the aircraft is:

```{math}
\hat{\mathbf{k}} = \hat{\mathbf{l}}_{\text{wind}} \cos\phi + (\hat{\mathbf{u}}_{\text{wind}} \times \hat{\mathbf{l}}_{\text{wind}}) \sin\phi.
```

The ground plane equation is $\mathbf{x} \cdot \hat{\mathbf{k}} = o$, with $o = \mathbf{r}_{\text{ref}} \cdot \hat{\mathbf{k}} - h$. Panel vertices reflect across the plane:

```{math}
\mathbf{x}' = \mathbf{x} - 2 (\mathbf{x} \cdot \hat{\mathbf{k}} - o) \hat{\mathbf{k}}.
```

The image panel vertex order is reversed to keep an outward-pointing unit normal. To cancel normal velocity at the ground, the image body carries:

```{math}
\mu' = +\mu, \qquad \sigma' = +\sigma.
```

The image wake is the reflection of the wake. The wake follows the free stream in ground effect, so it stays parallel to the ground and keeps the height of the trailing edge. The influence matrix depends on the attitude and the height; the solver groups the conditions by configuration (ground normal, offset and wake direction) and factorizes once per group. An explicit ground plane can replace `condition.h`. Geometry or wake on or below the ground raises `GroundStrikeError`.

The solver builds the wake of one attitude once and shares it between heights while that attitude is cached.

When solving a symmetric aircraft in ground effect on a half mesh, four-fold images are used: self, Y-mirror, Ground-mirror, and YG-mirror. They need a configuration that is symmetric about $y = 0$: with bank the ground normal has a $y$ component, and with sideslip the free-stream wake has one. The half-mesh solver then solves the full mesh of that configuration instead and returns the right half.

## Post-processing

### Tangential velocity and pressure coefficient

On the outer surface of a closed body with zero internal perturbation potential, the tangential perturbation velocity equals the negative surface gradient of the doublet strength:

```{math}
\mathbf{v}_{\text{pert}, t} = -\nabla_s \mu.
```

The surface gradient is a sparse linear operator built once per mesh. Around each corner node, the group of a panel is the set of panels that it reaches through edges that are not cut; wake shedding edges are cut, and so are sharp edges (in a mesh with panel kinds, the edges between the surface, tip, base and side panels, which meet at about 90 deg; in a mesh with one kind, folds above 45 deg, a method choice). The folds between surface panels stay connected, because a coarse leading edge can fold by more than 90 deg on a smooth surface. On an interior panel (all corner groups closed), the doublet values are interpolated to the corner nodes with inverse distance weights (Shepard, 1968), and the Green-Gauss theorem on the panel polygon (Blazek, 2015) gives

```{math}
\nabla_s \mu_i = \frac{1}{A_i} \sum_{e = (a, b)} \frac{\mu_a + \mu_b}{2} \, (\mathbf{p}_b - \mathbf{p}_a) \times \hat{\mathbf{n}}_i.
```

The node values smooth the doublet error of the thin triangle fans at the poles of latitude-longitude meshes. A panel with a corner on a wake edge, a sharp edge or a mesh border uses a least-squares fit over the panels that share an uncut edge, in the tangent plane and anchored at the panel value:

```{math}
\Delta \mathbf{r}_{ij, t} = \Delta \mathbf{r}_{ij} - (\Delta \mathbf{r}_{ij} \cdot \hat{\mathbf{n}}_i) \hat{\mathbf{n}}_i, \qquad \Delta \mathbf{r}_{ij, t} \cdot \nabla_s \mu_i \approx \mu_j - \mu_i.
```

The minimum-norm solution stays in the panel plane; displacement directions below $10^{-5}$ of the largest are treated as round-off. The anchored fit keeps the trailing-edge pressure sensitive to each wake strip, which the pressure Kutta iteration needs.

The total surface velocity is:

```{math}
\mathbf{V}_{\text{surf}, i} = \mathbf{V}_{\text{onset}, t}(\mathbf{c}_i) - \nabla_s \mu_i.
```

Without body rates, Bernoulli's equation gives the pressure coefficient:

```{math}
C_p = 1 - \frac{\|\mathbf{V}_{\text{surf}}\|^2}{V_\infty^2}.
```

#### Pressure with body rates

With body rates $p, q, r$ the body turns with the angular velocity $\boldsymbol{\Omega}$ (in geometry axes $\boldsymbol{\Omega} = (-p, q, -r)$). The point $\mathbf{x}$ of the body moves with the velocity $\mathbf{v}_b(\mathbf{x})$ in the earth frame. The local onset velocity of the solver is $\mathbf{V}_k(\mathbf{x}) = -\mathbf{v}_b(\mathbf{x}) = V_\infty \hat{\mathbf{d}} - \boldsymbol{\Omega} \times (\mathbf{x} - \mathbf{x}_{\text{ref}})$. The quasi-steady model takes the flow as steady in the body frame. The flow is then not steady in the earth frame, and the pressure follows from the unsteady Bernoulli equation for the perturbation potential $\phi$ in the earth frame, where the air is at rest far from the body:

```{math}
p - p_\infty = -\rho \left( \frac{\partial \phi}{\partial t} + \frac{1}{2} \|\nabla \phi\|^2 \right).
```

The potential is steady in the body frame. Thus at a point fixed in the earth frame $\partial \phi / \partial t = -\mathbf{v}_b \cdot \nabla \phi$. The velocity relative to the body is $\mathbf{V}_{\text{surf}} = \nabla \phi - \mathbf{v}_b = \nabla \phi + \mathbf{V}_k$, so $\|\mathbf{V}_{\text{surf}}\|^2 = \|\nabla \phi\|^2 - 2\, \mathbf{v}_b \cdot \nabla \phi + \|\mathbf{V}_k\|^2$. Substitution gives:

```{math}
C_p = \frac{p - p_\infty}{\frac{1}{2} \rho V_\infty^2} = \frac{\|\mathbf{V}_k(\mathbf{x})\|^2 - \|\mathbf{V}_{\text{surf}}\|^2}{V_\infty^2}.
```

Without rates $\|\mathbf{V}_k\| = V_\infty$ and the two forms are the same. The solver keeps the first form in that case, so results without rates do not change. The kinematic term $\|\mathbf{V}_k\|^2 / V_\infty^2 - 1$ does not depend on the doublets, so the pressure Kutta Jacobian does not change; the pressure Kutta sets the jump of the full $C_p$ above to zero. The Morino batch path, the single-condition path, the pressure Kutta and the viscous strip coupling use this $C_p$, on the CPU and on the GPU. In ground effect the same quasi-steady formula applies; the model does not include the change of height and attitude in time.

Check (`tests/test_panel3d_rotation_pressure.py`): a sphere of radius $a$ flies a circle with the pitch rate $q$ (or the yaw rate $r$) about its centre. In the earth frame the flow is the flow of a translating sphere whose velocity turns, so the only force is the added-mass force $-m_a\, d\mathbf{U}/dt$. Here $m_a = \frac{2}{3} \pi \rho a^3$ (half the displaced mass) and the centripetal acceleration is $\|d\mathbf{U}/dt\| = \Omega V_\infty$. With $S_{\text{ref}} = \pi a^2$ the coefficient of the force towards the centre of the turn is $-\frac{4}{3} a \Omega / V_\infty$. A 16 x 32 mesh gives this value to 2.1 %. The former formula $1 - \|\mathbf{V}_{\text{surf}}\|^2 / V_\infty^2$ gave about three times this value ($-4 a \Omega / V_\infty$).

### Forces and moments

Integrating surface pressure over all panels gives the aerodynamic force vector:

```{math}
\mathbf{F} = \sum_{i=1}^{N} -C_{p, i} \, q_\infty \, A_i \, \hat{\mathbf{n}}_i, \qquad q_\infty = \frac{1}{2} \rho_\infty V_\infty^2.
```

Force coefficients in stability/wind axes are:

```{math}
C_L = \frac{\mathbf{F} \cdot \hat{\mathbf{l}}_{\text{wind}}}{q_\infty S_{\text{ref}}}, \qquad
C_Y = \frac{\mathbf{F} \cdot \hat{\mathbf{y}}_{\text{wind}}}{q_\infty S_{\text{ref}}}, \qquad
C_{D, \text{pressure}} = \frac{\mathbf{F} \cdot \hat{\mathbf{u}}_{\text{wind}}}{q_\infty S_{\text{ref}}}.
```

The aerodynamic moment vector about reference point $\mathbf{r}_{\text{ref}}$ is:

```{math}
\mathbf{M} = \sum_{i=1}^{N} (\mathbf{c}_i - \mathbf{r}_{\text{ref}}) \times \left(-C_{p, i} \, q_\infty \, A_i \, \hat{\mathbf{n}}_i\right).
```

Moments follow AVL flight-dynamics conventions:

```{math}
C_l = -\frac{M_x}{q_\infty S_{\text{ref}} b_{\text{ref}}}, \qquad
C_m = +\frac{M_y}{q_\infty S_{\text{ref}} c_{\text{ref}}}, \qquad
C_n = -\frac{M_z}{q_\infty S_{\text{ref}} b_{\text{ref}}}.
```

### Trefftz-plane induced drag

Far-field induced drag is computed in the Trefftz plane, normal to the wake direction (the body $x$ axis in free air, the free stream in ground effect), from the trailing vortex sheet circulation $\Gamma_s = \mu_{w, s}$. Each strip trailing edge is projected on the plane; its ends carry 2D point vortices $\pm \Gamma_s$, with $+\Gamma_s$ at the end with the larger projection on $\hat{\mathbf{n}}_w \times \hat{\mathbf{d}}$ (strip normal on the upper side, wake direction). In ground effect the image vortices (reflected ends, opposite circulation) add their normal wash. The normal wash $w_{n, s}$ at the midpoint of each segment of length $l_s$ follows from the 2D Biot-Savart law with the squared distance $r^2 + r_c^2$, where the core radius is $r_c = 0.01 \bar{l}$ and $\bar{l}$ the mean segment length:

```{math}
D_i = -\frac{1}{2} \rho_\infty \sum_{s=1}^{N_{\text{strips}}} \Gamma_s \, w_{n, s} \, l_s, \qquad
C_{Di, \text{trefftz}} = \frac{D_i}{q_\infty S_{\text{ref}}}.
```

The in-plane normal is $\hat{\mathbf{d}} \times (\mathbf{p}_r - \mathbf{p}_l) / l_s$, so the product $w_{n, s} \, l_s$ uses the length of the segment projected on the Trefftz plane, also for sweep and dihedral.

## Computational backends

Ventorum-3DP supports three computational backends:
- `"numpy"` (default): Reference vectorised kernel evaluations and SciPy LAPACK LU solves on CPU.
- `"numba"`: JIT-compiled CPU kernels executing identical formulas with multithreaded loops and SciPy LU factorizations.
- `"gpu"`: GPU pipeline using NVIDIA Warp kernels for matrix assembly (one thread per panel-target pair) and PyTorch `torch.linalg.lu_factor` / `lu_solve` on the device. Unit onset solves are batched in a single call. Matrix assembly runs in float64; the matrices are kept, factorized and solved in the selected precision (float32 or float64). The gpu backend does no factorization on the CPU and keeps no copy of the matrices on the host. Before each configuration the solver estimates the device memory of its body, mirror and image matrices, systems and factors, and refuses a solve that does not fit (no silent fallback).

The solver family layer selects the backend with `panel_backend="auto"` as the lattice solvers do (`ventorum.gpu`: device setting, CPU worker threads, and `use_gpu` with the work of the dense systems and the thresholds of the vortex lattice); the precision on the GPU is the selected precision. On the CPU, "auto" runs the compiled Numba kernels (they agree with the numpy reference kernels to round-off), or the numpy reference kernels when Numba is not installed; the dispatch reason names the CPU kernels.

## Mesh convergence and Richardson extrapolation

The module `ventorum.panel3d.convergence` estimates the mesh error of the base mesh (the mesh of the user, $n_s$ spanwise and $n_c$ chordwise panels) with a refinement in each direction separately:
- Chordwise refinement: the chordwise count is multiplied by a constant ratio $r$ (default 1.5) over three or four levels at the base spanwise count. Spanwise refinement: the spanwise count is multiplied by $r$ at the base chordwise count. The base mesh is the first level of both sequences.
- Reason. On a thick wing the chordwise error and the spanwise error of the lift have opposite signs (too little lift from the chordwise discretization, too much from the spanwise one). When both counts grow together, the two errors partly cancel, and the sequence is often not monotonic: a refinement in both directions at once gives no reliable order or Richardson value. Each direction alone has one sign of error, so its sequence is monotonic once that direction is in its asymptotic range.
- For each direction, the three finest levels go to the layer-1 numerical error bar of Ventorum (`ventorum.core.error_bars.coefficient_error_bar`, Roache 1998; Celik et al. 2008). It classifies the convergence (monotonic, oscillatory, divergent, round-off, order not found), computes the observed order $p$ by the fixed-point iteration of Celik et al. for non-constant ratios, the Richardson value (Richardson and Gaunt 1927)

```{math}
f_{\text{extrap}} = \frac{r_{21}^{p} f_1 - f_2}{r_{21}^{p} - 1},
```

and the bar of the finest level with the safety factor $F_s$ (1.25 inside the asymptotic range $p \in [0.5, 4.0]$; 3.0 outside it).
- The error estimate of the base mesh in one direction is $e = F_s \, |f_{\text{extrap}} - f_0|$ for a monotonic convergence, and $|f_1 - f_0|$ plus the bar of the finest level in the other states with a bar. A divergent direction gives no estimate.
- Separable error model. The error is modelled as the sum of a spanwise and a chordwise term,

```{math}
f(n_s, n_c) = f_\infty + a\, n_s^{-p_s} + b\, n_c^{-p_c}.
```

The chordwise Richardson value is then $R_c = f(n_{s,0}, \infty)$ and the spanwise one $R_s = f(\infty, n_{c,0})$, so the converged value is $f_\infty = R_c + R_s - f_0$. The error estimate of the base mesh is the sum of the two directional estimates, $e = e_c + e_s$. The base error is $f_0 - f_\infty = (f_0 - R_c) + (f_0 - R_s)$, and each directional estimate is at least the distance to its Richardson value, so under the model the sum bounds the base error also when the two errors have opposite signs. This estimate goes to the trust notes when `panel_estimate_mesh_error` is set (`MeshErrorEstimate`, with the convergence of each direction).
- Cost. The study solves the base mesh and $L - 1$ more meshes in each direction ($L$ levels per direction; five meshes for three levels). A directional level multiplies the panel count by about $r$, and the dense LU work grows with the cube of the panel count. The study therefore costs $\sum_k (N_k / N_0)^3$ base solves over the solved meshes, about $1 + 2\,(r^3 + r^6)$ for three levels: about 30 at $r = 1.5$ (four levels add about $2 r^9$, about 107 in total).
- Checks (`tests/test_panel3d_convergence.py`): a constructed separable function with errors of opposite sign (the limit and the orders are recovered exactly); a sphere refined in latitude and longitude against the exact potential flow (Cp max and the axial force of the upstream half: order near 2 in both directions, the Richardson value more than ten times closer to the exact value than the base value, and the estimate covers the exact error); a rectangular wing, whose estimate covers one more level in each direction.

## Solver interface

`PanelMethodSolver` connects Ventorum-3DP to the standard solver API (`solver="panel3d"`):
- Inputs: `Aircraft`, `FlightCondition` (or sequences for sweeps), `SolverSettings`, and an optional `GroundPlane` (keyword `ground`) that replaces `condition.h`. The panel fields of `SolverSettings` are `panel_kutta`, `panel_backend`, `panel_wake_length`, `panel_chord_spacing` (cosine by default), `panel_estimate_mesh_error` and `panel_scale_root_thickness` (True by default; the root section of a symmetric surface with dihedral keeps its thickness normal to the local surface, see the panel solver theory page); `n_chord` is the chordwise panel count per surface (40 when None).
- Outputs: `SolverResult` containing `totals` (`IntegratedResult`), `spanwise` (`SpanwiseResult` strip loads, each panel assigned to the lattice strip whose station planes enclose its centroid, or by the strip index of the mesh builder), surface `cp` distribution, panel options (`details["panel"]`), trust score, and execution `dispatch` records with the counted factorizations.

### Krylov solve in ground effect

The default `"auto"` uses Krylov for CPU ground configurations with Morino Kutta and direct LU elsewhere.
The Krylov option uses
right-preconditioned GMRES for CPU float64 configurations with Morino
Kutta. The first configuration at each ground offset is a direct anchor.
Later configurations use that anchor LU as the right preconditioner and
its corresponding unit-onset solution as the initial guess. Each system
matrix is still assembled. The solver checks the final residual of the
original system against its relative tolerance. A restart length and an
inner iteration cap bound the solve. If any onset fails, the whole
configuration uses direct LU and becomes the new anchor for that offset.
The solver keeps only the current anchor factors at each offset. It drops
finished work matrices and keeps unit solutions for cached configurations.
Free air, GPU backends and pressure Kutta use direct LU.

The unit onsets of each system run in lockstep. Each onset keeps its own
Arnoldi basis, Hessenberg matrix and Givens rotations. Modified
Gram-Schmidt forms each basis independently. The active columns share
each matrix product and each solve with the anchor LU factors. An onset
that reaches the inner limit leaves the active group. The remaining
onsets continue in their own spaces. The configuration and the dispatch
record count these matrix passes, including the initial, restart and
final true-residual products. Direct solves have no lockstep passes.

The method follows Y. Saad and M. H. Schultz,
["GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric
Linear Systems"](https://doi.org/10.1137/0907058), SIAM Journal on Scientific and Statistical Computing,
and Y. Saad, [*Iterative Methods for Sparse Linear Systems*](https://www-users.cse.umn.edu/~saad/books.html), second edition,
SIAM, chapters on Krylov methods and preconditioned iterations. Specific
page and equation numbers for right preconditioning: to be checked.

## Assumptions

- Steady, incompressible, irrotational potential flow.
- Thin vortex wake without roll-up, shed along the body axis in free air and along the free stream in ground effect.
- Rigid aircraft geometry and rigid ground plane.
- Attached potential flow without flow separation or stall.

## Envelope

- Pre-stall flight conditions at Mach below 0.3.
- Closed, watertight surface meshes without self-intersections.
- In ground effect, ground clearance of the body and the wake above zero (ground strikes raise `GroundStrikeError`, a `ValueError`).
- No wake sheet touches or crosses another closed surface: the wake is rigid and flat, so a surface directly behind another surface at the same height is refused with `ValidityError` (see the panel solver chapter, section *Wake Through Another Surface*).

## Implementation

- Mesh generation and quality: `ventorum.panel3d.mesh.PanelMesh`, `mesh_aircraft`, `check_mesh`.
- Influence kernels: `ventorum.panel3d.kernels`, `ventorum.panel3d.kernels_numba`, `ventorum.panel3d.kernels_warp`.
- Wake geometry and shedding: `ventorum.panel3d.wake.PanelWake`, `build_wake`, `check_wake_crossing`.
- Kutta conditions: `ventorum.panel3d.kutta.solve_pressure_kutta`.
- Image systems and symmetry: `ventorum.panel3d.images`.
- Core solver: `ventorum.panel3d.solver.PanelSolver`, `PanelResult`.
- Post-processing: `ventorum.panel3d.post`.
- GPU acceleration: `ventorum.panel3d.gpu`.
- Convergence studies: `ventorum.panel3d.convergence.panel_mesh_convergence`.
- Solver adapter: `ventorum.solvers.panel3d.PanelMethodSolver`.

## Verification

The 3D panel method implementation is verified by tests in the test suite:
- Kernel parity and asymptotic limits: `tests/test_panel3d_kernels.py`, `tests/test_panel3d_backends.py`.
- Mesh construction and control surface deflections: `tests/test_panel3d_mesh.py`, `tests/test_panel3d_controls.py`.
- Potential flow solution, unit onsets, surface gradient and Trefftz drag (sphere in two orientations, ellipsoid on all panels, stored free-air run, factorization count): `tests/test_panel3d_solver.py`.
- Wake sheets through another surface (refused tandem, unchanged offset tandem, wing and tail, biplane): `tests/test_panel3d_wake_crossing.py`.
- Symmetry plane and ground effect images (explicit mirror with the free-stream wake, Trefftz drag with images, bank and sideslip on a half mesh): `tests/test_panel3d_images.py`.
- Iterative pressure Kutta condition (stored Morino run, asymmetric half mesh against the full mesh): `tests/test_panel3d_kutta.py`.
- Mesh convergence and Richardson extrapolation: `tests/test_panel3d_convergence.py`.
- Refinement regressions (open trailing edge with chordwise refinement, dihedral root strip with spanwise refinement): `tests/test_panel3d_refinement.py`; flapped wing with spanwise refinement: `tests/test_panel3d_controls.py`.
- Solver family integration, settings, ground plane, strip map and trust scores: `tests/test_panel3d_factory.py`.
- GPU pipeline parity (when CUDA is available): `tests/test_panel3d_gpu.py`.

## References

- J. D. Anderson, *Fundamentals of Aerodynamics*, 5th ed., McGraw-Hill, New York, 2011.
- I. B. Celik, U. Ghia, P. J. Roache, C. J. Freitas, H. Coleman and P. E. Raad, "Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications", *Journal of Fluids Engineering*, Vol. 130, No. 7, 078001, 2008.
- M. Drela, *Flight Vehicle Aerodynamics*, MIT Press, Cambridge, MA, 2014.
- J. Blazek, *Computational Fluid Dynamics: Principles and Applications*, 3rd ed., Butterworth-Heinemann, Oxford, 2015, chapter 5.
- J. L. Hess and A. M. O. Smith, "Calculation of Potential Flow About Arbitrary Bodies", *Progress in Aeronautical Sciences*, Vol. 8, Pergamon Press, 1967, pp. 1-138.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, Cambridge, 2001.
- S. A. Kinnas and C. Y. Hsin, "Boundary Element Method for the Analysis of the Unsteady Flow Around Extreme Propeller Geometries", *AIAA Journal*, Vol. 30, No. 3, 1992, pp. 688-696.
- C. L. Ladson, C. W. Brooks, A. S. Hill and D. W. Sproles, "Computer Program To Obtain Ordinates for NACA Airfoils", NASA TM-4741, 1996.
- B. Maskew, "Program VSAERO Theory Document: A Computer Program for Calculating the Non-Linear Aerodynamic Characteristics of Arbitrary Configurations", NASA Contractor Report 4023, 1987.
- L. Morino and C. C. Kuo, "Subsonic Potential Aerodynamics for Complex Configurations: A General Theory", *AIAA Journal*, Vol. 12, No. 2, 1974, pp. 191-197.
- J. N. Newman, "Distributions of Sources and Normal Dipoles Over a Quadrilateral Panel", *Journal of Engineering Mathematics*, Vol. 20, 1986, pp. 113-126.
- L. F. Richardson and J. A. Gaunt, "The Deferred Approach to the Limit", *Philosophical Transactions of the Royal Society of London, Series A*, Vol. 226, 1927, pp. 299-361.
- P. J. Roache, *Verification and Validation in Computational Science and Engineering*, Hermosa Publishers, Albuquerque, NM, 1998.
- D. Shepard, "A Two-Dimensional Interpolation Function for Irregularly-Spaced Data", *Proceedings of the 23rd ACM National Conference*, 1968, pp. 517-524.
- B. L. Stevens, F. L. Lewis and E. N. Johnson, *Aircraft Control and Simulation: Dynamics, Controls Design, and Autonomous Systems*, 3rd ed., John Wiley & Sons, Hoboken, NJ, 2016.
- A. van Oosterom and J. Strackee, "The Solid Angle of a Plane Triangle", *IEEE Transactions on Biomedical Engineering*, Vol. BME-30, No. 2, 1983, pp. 125-126.
- C. Wieselsberger, "Wing Resistance Near the Ground", NACA TM-77, 1922; translated from *Zeitschrift fuer Flugtechnik und Motorluftschiffahrt*, Vol. 12, No. 10, 1921.
