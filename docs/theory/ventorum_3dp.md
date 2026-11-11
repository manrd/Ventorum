# Ventorum-3DP: Three-dimensional potential flow panel method

## Model

Ventorum-3DP solves steady, incompressible, inviscid potential flow around arbitrary three-dimensional lifting configurations and non-lifting bodies. The solver uses a low-order boundary element formulation with flat quadrilateral and triangular panels carrying constant distributions of sources and doublets (Morino and Kuo, 1974; Hess and Smith, 1967; Katz and Plotkin, 2001). The formulation enforces an internal Dirichlet boundary condition of zero perturbation potential inside the closed body. The linear aerodynamic influence matrix is factorized once per configuration (once for all free-air conditions) by LU decomposition and solved for six fundamental unit onset flow states (three translation velocities and three angular rotation rates). In ground effect on the CPU with the Morino Kutta condition, the default ground solve (`panel_ground_solver="auto"`) factorizes only one anchor configuration per ground offset and solves the other configurations with GMRES, preconditioned by the anchor LU (see *Krylov solve in ground effect*). Flight conditions at arbitrary angles of attack, sideslip, and rotation rates follow by linear superposition. The method supports Morino and iterative non-linear pressure Kutta conditions, planar symmetry folding, ground effect via the method of images, near-field pressure integration, far-field Trefftz-plane induced drag calculation, grid convergence analysis, and multi-backend execution on CPU (NumPy, Numba) and CUDA GPU (NVIDIA Warp, PyTorch).

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

The panel centroid $\mathbf{p}_0$ (the collocation point and the origin of the local frame) is the mean of the corner points (for a quadrilateral that is not a parallelogram, this is not the area centroid; the far-field expansions use the area centroid, see below):
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
\mathbf{V}_D = \nabla \Phi_D = \sum_{k=1}^4 \mathbf{v}_{\text{segment}}(\mathbf{p}'_k \to \mathbf{p}'_{k+1}),
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

### Far-field multipole expansions

A pair of a target and a panel uses the far-field formulas when the distance $r = \|\mathbf{x} - \mathbf{p}_c\|$ from the area centroid $\mathbf{p}_c$ of the panel exceeds `far_ratio` times the panel diameter $D = \max_{i < j} \|\mathbf{p}_i - \mathbf{p}_j\|$ (default `far_ratio = 5`). The far-field formulas are the Taylor (Cartesian multipole) expansion of $1/\|\mathbf{x} - \mathbf{y}\|$ about $\mathbf{p}_c$, integrated over the panel, up to the second moments of area (the derivation of the Cartesian multipole expansion is in Jackson 1999, section 4.1).

With $\mathbf{y} = \mathbf{p}_c + \mathbf{s}$ on the panel and $\mathbf{r} = \mathbf{x} - \mathbf{p}_c$:

```{math}
\frac{1}{\|\mathbf{r} - \mathbf{s}\|} = \frac{1}{r} + \frac{\mathbf{r} \cdot \mathbf{s}}{r^3} + \frac{3 (\mathbf{r} \cdot \mathbf{s})^2 - r^2 s^2}{2 r^5} + O\!\left(\frac{s^3}{r^4}\right).
```

The integral of $\mathbf{s}$ over the panel is zero about the area centroid, so the first-moment term drops out. The panel moments are the area $A$ [m^2] and the second-moment tensor $\mathbf{Q} = \int \mathbf{s} \mathbf{s}^{\mathsf{T}} \, dA$ [m^4]. $\mathbf{Q}$ lies in the panel plane ($\mathbf{Q} \hat{\mathbf{n}} = 0$). With the scalars $a = \mathbf{r}^{\mathsf{T}} \mathbf{Q} \mathbf{r}$ [m^6] and $q = \operatorname{tr} \mathbf{Q}$ [m^4], the integrated kernel is:

```{math}
S(\mathbf{r}) = \int \frac{dA}{\|\mathbf{r} - \mathbf{s}\|} = \frac{A}{r} + \frac{3 a - r^2 q}{2 r^5} + O\!\left(\frac{A D^3}{r^4}\right).
```

The source and doublet potentials and their gradients follow from $S$: $\Phi_S = -S / (4 \pi)$ and $\Phi_D = (\hat{\mathbf{n}} \cdot \nabla S) / (4 \pi)$. With $z = \mathbf{r} \cdot \hat{\mathbf{n}}$ and $f = (3 q / 2 - 15 a / (2 r^2)) / r^5$:

```{math}
\Phi_S^{\text{far}} = -\frac{1}{4 \pi} \left[\frac{A}{r} + \frac{3 a - r^2 q}{2 r^5}\right], \qquad
\mathbf{V}_S^{\text{far}} = \frac{1}{4 \pi} \left[\left(\frac{A}{r^3} - f\right) \mathbf{r} - \frac{3 \mathbf{Q} \mathbf{r}}{r^5}\right],
```

```{math}
\Phi_D^{\text{far}} = \frac{z}{4 \pi} \left[f - \frac{A}{r^3}\right], \qquad
\mathbf{V}_D^{\text{far}} = \frac{1}{4 \pi} \left[\left(f - \frac{A}{r^3}\right) \hat{\mathbf{n}} + z \left(\frac{105 a / 2 - 15 r^2 q / 2}{r^9} + \frac{3 A}{r^5}\right) \mathbf{r} - \frac{15 z}{r^7} \mathbf{Q} \mathbf{r}\right].
```

The first terms ($A$ only) are the point source and the point doublet. The area centroid and the moments come from the projected corners in the local frame with Green's theorem in the plane (sums over the four edges, a derivation): with $c_k = x_k y_{k+1} - x_{k+1} y_k$, $A = \frac{1}{2} \sum c_k$, $x_c = \frac{1}{6 A} \sum (x_k + x_{k+1}) c_k$, $\int x^2 dA = \frac{1}{12} \sum (x_k^2 + x_k x_{k+1} + x_{k+1}^2) c_k$, $\int x y \, dA = \frac{1}{24} \sum (x_k y_{k+1} + 2 x_k y_k + 2 x_{k+1} y_{k+1} + x_{k+1} y_k) c_k$, and the same for $y$. The parallel-axis rule moves the moments to the area centroid. A triangle (the fourth corner equal to the third) has an edge of zero length that adds zero.

Truncation error of one pair, relative to the size of the leading term:

| Expansion | General quadrilateral | Parallelogram |
| --- | --- | --- |
| Point singularity about the corner mean $\mathbf{p}_0$ | $O(D / r)$ | $O((D / r)^2)$ |
| Point singularity about the area centroid $\mathbf{p}_c$ | $O((D / r)^2)$ | $O((D / r)^2)$ |
| Area centroid with the second moments (used) | $O((D / r)^3)$ | $O((D / r)^4)$ |

For a parallelogram the third moments are zero (point symmetry about $\mathbf{p}_c$), so the error is one order smaller. The expansion centre is used only for the far-field formulas. The collocation point and the local frame of the exact formulas stay at the corner mean $\mathbf{p}_0$. So the results with `far_ratio = np.inf` do not depend on the far-field formulas.

Why the second moments are necessary: the switch distance is a fixed multiple of the panel diameter. On a refined mesh, more of the surface is in the far set of each target, and the far-set panels at a few diameters from the target carry a relative error that does not decrease with the panel size. The panels of a rectangular wing are parallelograms, so for them the area centroid alone changes nothing. With the point singularities alone, the lift of an aspect-ratio-6 rectangular wing moved away from the exact-integration value as the spanwise panel count grew, in free air and more in ground effect. With the second moments, `tests/test_panel3d_far_field.py` holds this difference below a small bound from 8 to 64 spanwise panels per semi-span.

Setting `far_ratio = np.inf` disables the far-field formulas and computes exact integrals everywhere.

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

Lifting surfaces shed flat vortex sheets from their trailing edges, one strip of panels per trailing-edge panel pair. The default wake length is 50 times the largest streamwise chord of the mesh (the largest x extent of a section). In free air the wake panels extend downstream along the body $+x$ axis, so the influence matrix does not change with the flight condition and one factorization serves a whole sweep. The free-air wake does not turn with the free stream, so the Trefftz plane is normal to the body $x$ axis. The far-field drag is then the induced force along the body $x$ axis, not along the free stream. Its relative difference to the drag of a wake along the free stream is of order $\sin^2 \alpha$ (an estimate, not a measured value: about 1 % of $C_{D_i}$ at $\alpha = 6$ deg and 7 % at 15 deg). The solver has no option for a free-stream wake in free air. In ground effect the wake follows the free stream, parallel to the ground: the matrix depends on the attitude in ground effect anyway, and no wake panel goes below the ground plane.

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

The image wake is the reflection of the wake. The wake follows the free stream in ground effect, so it stays parallel to the ground and keeps the height of the trailing edge. The influence matrix depends on the attitude and the height; the solver groups the conditions by configuration (ground normal, offset and wake direction) and factorizes once per group with the direct solve. With the default Krylov ground solve on the CPU, only the anchor group of each ground offset is factorized, and GMRES solves the other groups (see *Krylov solve in ground effect*). An explicit ground plane can replace `condition.h`. Geometry or wake on or below the ground raises `GroundStrikeError`.

The solver builds the wake of one attitude once and shares it between heights while that attitude is cached.

When solving a symmetric aircraft in ground effect on a half mesh, four-fold images are used: self, Y-mirror, Ground-mirror, and YG-mirror. They need a configuration that is symmetric about $y = 0$: with bank the ground normal has a $y$ component, and with sideslip the free-stream wake has one. The half-mesh solver then solves the full mesh of that configuration instead and returns the right half.

## Post-processing

### Tangential velocity and pressure coefficient

On the outer surface of a closed body with zero internal perturbation potential, the tangential perturbation velocity equals the negative surface gradient of the doublet strength:

```{math}
\mathbf{v}_{\text{pert}, t} = -\nabla_s \mu.
```

The surface gradient is a sparse linear operator built once per mesh. Around each corner node, the group of a panel is the set of panels that it reaches through edges that are not cut; wake shedding edges are cut, and so are sharp edges (in a mesh with panel kinds, the edges between the surface, tip, base and side panels, which meet at about 90 deg; in a mesh with one kind, folds above 45 deg, a method choice). The folds between surface panels stay connected, because a coarse leading edge can fold by more than 90 deg on a smooth surface. On an interior panel (all corner groups closed), the doublet values are reconstructed at the corner nodes with a least-squares linear fit in the node tangent plane (Blazek, 2015), which is exact for a field that is linear in that plane on any mesh (also at nodes with three panels, planform kinks and cosine end strips). A degenerate group (fit of rank below 3, or a weight larger than 2 in magnitude) keeps inverse distance weights (Shepard, 1968). The Green-Gauss theorem on the panel polygon (Blazek, 2015) then gives

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
\mathbf{M} = \sum_{i=1}^{N} (\mathbf{p}_{c, i} - \mathbf{r}_{\text{ref}}) \times \left(-C_{p, i} \, q_\infty \, A_i \, \hat{\mathbf{n}}_i\right).
```

The arm goes to the area centroid $\mathbf{p}_{c, i}$ of each panel, where the force of a constant $C_p$ acts. The corner mean (the collocation point) is a different point on a skewed or tapered panel. The viscous coupling uses the same arm.

Moments follow AVL flight-dynamics conventions:

```{math}
C_l = -\frac{M_x}{q_\infty S_{\text{ref}} b_{\text{ref}}}, \qquad
C_m = +\frac{M_y}{q_\infty S_{\text{ref}} c_{\text{ref}}}, \qquad
C_n = -\frac{M_z}{q_\infty S_{\text{ref}} b_{\text{ref}}}.
```

Spanwise strips. The panels of each spanwise strip $j$ (width $w_j$ in the $y$-$z$ plane, chord $c_j$ at the strip mid span fraction, area $S_j$ from the mean of the edge chords) give the strip lift $L_j = \mathbf{F}_j \cdot \hat{\mathbf{l}}_{\text{wind}}$ [N]. The strip outputs are:

```{math}
C_{l, j} = \frac{L_j}{q_j S_j}, \qquad
q_j = \tfrac{1}{2} \rho_\infty \lVert \mathbf{U}_{qc, j} \rVert^2, \qquad
L'_j = \frac{L_j}{w_j}, \qquad
\Gamma_j = \frac{L'_j}{\rho_\infty V_\infty}, \qquad
C_{l, j}^{\infty} = \frac{2 \Gamma_j}{V_\infty c_j} = \frac{L'_j}{q_\infty c_j}.
```

$\mathbf{U}_{qc, j}$ is the onset velocity (the free stream and the velocity of the body rates about the moment reference point, without the induced velocity) at the strip quarter-chord point, the leading edge of the strip at its mid span fraction plus $c_j / 4$ in $x$. This is the rule of the lattice strip $C_l$ (see the [rotation rates page](rotation_rates.md)). Without body rates $q_j = q_\infty$. $C_{l, j}$ is `Cl`, and the effective angle $\alpha_{L0} + C_{l, j} / a_0$ uses it. $C_{l, j}^{\infty}$ is `Cl_freestream`, with the same rule as the lattice strips: `Cl_freestream * q_inf * c` is $\rho_\infty V_\infty \Gamma_j$. It stays on $q_\infty$ also with body rates. The strip `Cm_section` is the strip pressure moment $M_{y, j}$ about the strip quarter-chord point over $q_j S_j c_j$, on the same local dynamic pressure as $C_{l, j}$ (the lattice section moment also uses the local onset). The strip `Cd_i` stays on $q_\infty$ also with body rates: it is a part of the total induced drag, which uses $q_\infty$. With body rates $C_{l, j} = C_{l, j}^{\infty} \, (V_\infty / \lVert \mathbf{U}_{qc, j} \rVert)^2 \, w_j c_j / S_j$. A yaw rate changes $\lVert \mathbf{U}_{qc} \rVert$ to first order in $r y / V_\infty$, so the two coefficients then differ by a relative amount of about $2 \lvert r y \rvert / V_\infty$ at the span station $y$. Without body rates $C_{l, j}$ and $C_{l, j}^{\infty}$ differ only by $S_j / (w_j c_j)$: they are equal when the chord at the strip mid span fraction is the mean of the edge chords (constant chord, or a linear taper with uniform spacing). With a taper and a cosine spacing the strip mid span fraction is not the middle of the strip, and the two differ slightly (most at the tip strip): on a taper 0.4 wing with cosine spacing and 16 strips per semi-span the difference is 0.36 % at the tip strip. The difference is kept on purpose: $C_{l, j}$ is the pressure lift of the real strip area $S_j$, and $C_{l, j}^{\infty}$ keeps the identity `Cl_freestream * q_inf * c` $= \rho_\infty V_\infty \Gamma_j$ of the lattice strips.

### Trefftz-plane induced drag

Far-field induced drag is computed in the Trefftz plane, normal to the wake direction (the body $x$ axis in free air, the free stream in ground effect), from the trailing vortex sheet circulation $\Gamma_s = \mu_{w, s}$. Each strip trailing edge is projected on the plane; its ends carry 2D point vortices $\pm \Gamma_s$, with $+\Gamma_s$ at the end with the larger projection on $\hat{\mathbf{n}}_w \times \hat{\mathbf{d}}$ (strip normal on the upper side, wake direction). In ground effect the image vortices (reflected ends, opposite circulation) add their normal wash. The normal wash $w_{n, s}$ at the midpoint of each segment of length $l_s$ follows from the 2D Biot-Savart law with the squared distance $r^2 + r_c^2$, where the core radius is $r_c = 0.01 \bar{l}$ and $\bar{l}$ the mean segment length:

```{math}
D_i = -\frac{1}{2} \rho_\infty \sum_{s=1}^{N_{\text{strips}}} \Gamma_s \, w_{n, s} \, l_s, \qquad
C_{Di, \text{trefftz}} = \frac{D_i}{q_\infty S_{\text{ref}}}.
```

The in-plane normal is $\hat{\mathbf{d}} \times (\mathbf{p}_r - \mathbf{p}_l) / l_s$, so the product $w_{n, s} \, l_s$ uses the length of the segment projected on the Trefftz plane, also for sweep and dihedral.

#### Yawing moment of the induced drag

The Trefftz plane also gives `Cn_trefftz`, the yawing moment of the induced-drag distribution, in stability axes. It is a far-field diagnostic next to the near-field `Cn`, with the rule of the vortex lattice (see [Forces and moments](forces_and_moments.md), section *Far-field lateral diagnostics*). The induced drag of each strip, $D'_s = -\tfrac{1}{2} \rho_\infty \Gamma_s w_{n, s} l_s$ [N] (the terms of the sum above), acts along the free stream $\hat{\mathbf{u}}$ at the midpoint $\mathbf{q}_s$ of the strip segment. The arm $\mathbf{a}_s$ is the vector $\mathbf{r}_s$ from the moment reference point to $\mathbf{q}_s$, projected on the Trefftz plane: $\mathbf{a}_s = \mathbf{r}_s - (\mathbf{r}_s \cdot \hat{\mathbf{d}}) \hat{\mathbf{d}}$. With the lift direction $\hat{\mathbf{L}}$ (the stability $z$ axis, up in the geometry axes):

```{math}
C_{n, T} = -\frac{\sum_s \left( \mathbf{a}_s \times D'_s \hat{\mathbf{u}} \right) \cdot \hat{\mathbf{L}}}{q_\infty S_{\text{ref}} b_{\text{ref}}}
= \frac{\sum_s D'_s \, y_s}{q_\infty S_{\text{ref}} b_{\text{ref}}},
```

where $y_s$ is the lateral arm along the wind-axis side direction. More drag on the right wing gives $C_{n, T} > 0$ (nose right), the AVL sign of `Cn`. The far-field drag and its distribution are those of the Trefftz-plane analysis (E. Trefftz, 1921; Katz and Plotkin, 2001; Drela, 2014). In ground effect the normal wash includes the image vortices; the image strips carry no moment. A symmetric flow on a half mesh (no sideslip, roll rate or yaw rate) gives 0 exactly, as in the lattice. The single-condition, batch and pressure Kutta paths use the same equations; the GPU batch path computes the strip terms $\Gamma_s w_{n, s} l_s$ and the strip circulations $\Gamma_s$ on the device in float64 and the moment on the host. The viscous strip coupling (`viscous_panel3d`) computes `Cn_trefftz` and `CY_farfield` from its coupled wake strengths with the same equations; its post-stall branches carry their own values.

The Trefftz plane also gives `CY_farfield`, the side force of the trailing vortex sheet, in wind axes (positive to the right), with the rule of the vortex lattice. The strip trace $\mathrm d\mathbf s_s$ goes from the $-\Gamma_s$ end to the $+\Gamma_s$ end (along $\hat{\mathbf{n}}_w \times \hat{\mathbf{d}}$) and is projected on the Trefftz plane. Its Kutta-Joukowski force is

$$
\mathbf F_s = -\rho_\infty V_\infty \Gamma_s \, (\hat{\mathbf{u}} \times \mathrm d\mathbf s_s), \qquad
C_{Y, \text{ff}} = \frac{\sum_s \mathbf F_s \cdot \hat{\mathbf{y}}_w}{q_\infty S_\text{ref}}, \qquad \hat{\mathbf{y}}_w = \hat{\mathbf{L}} \times \hat{\mathbf{u}}
$$

(Trefftz, 1921; Katz and Plotkin, 2001; Drela, 2014). The minus sign: with the doublet sign of the solver, a strip that lifts towards its upper side $\hat{\mathbf{n}}_w$ has $\Gamma_s = \mu_u - \mu_l < 0$, so $-\Gamma_s$ is the circulation of the Kutta-Joukowski law ($\hat{\mathbf{u}} \times \mathrm d\mathbf s_s$ points along $\hat{\mathbf{n}}_w$). The induced drag and `Cn_trefftz` are quadratic in $\Gamma_s$, so this sign does not enter them. The same sum along $\hat{\mathbf{L}}$ gives the lift of the wing (within 2 %; a check in the tests). A planar wing has no trace component normal to its plane, so `CY_farfield` is 0 for a planar wing in sideslip and with rates; the side force comes from dihedral and from fins. The ground images do not enter (the force on the real sheet), as in the lattice. A symmetric flow on a half mesh gives 0 exactly. All paths that give `Cn_trefftz` also give `CY_farfield`.

Properties of the free-air value. The free-air wake follows the body $x$ axis, so the Trefftz plane is the $y$-$z$ plane. On a planar wing in sideslip the side onset lies in the plane of the wing, the circulation stays symmetric and `Cn_trefftz` is 0. A vortex lattice with the wake along the free stream gives a value that is not zero there, because its wake turns with the sideslip; with the wake along the body $x$ axis (`wake_alignment="body"`) it also gives 0. With dihedral, and with roll and yaw rates, the circulation is antisymmetric and the value agrees with the lattice (checks below). `Cn_trefftz` does not use the surface gradient, so it does not have the coarse-mesh leading-edge error of the near-field `Cn` (see *Envelope*).

Checks (`tests/test_panel3d_trefftz_yaw.py`): the value equals an independent sum $\sum_s D'_s y_s / (q_\infty S b)$ of the wake strips on half and full meshes; a roll rate $p > 0$ gives $C_{n, T} > 0$ and a yaw rate $r > 0$ gives $C_{n, T} < 0$, and the opposite rate gives the opposite value; a symmetric flow gives 0; on a NACA 0006 wing of aspect ratio 8 (16 x 16 mesh) the yaw-rate, roll-rate and dihedral-sideslip values agree with the lattice `Cn_trefftz` within 10 % (Morino and pressure Kutta; the tolerance and the measured values are in the test); the GPU batch and single-condition paths agree with numpy (float64 to $10^{-10}$, float32 to its rounding floor). `tests/test_panel3d_rotation_pressure.py` compares the far-field yaw damping $C_{n_r}$ (central difference of `Cn_trefftz` in the stability yaw rate) with the lattice.

Checks of `CY_farfield` (`tests/test_panel3d_trefftz_side_force.py`): the value equals an independent Kutta-Joukowski sum over the wake strips on half and full meshes, and the same sum along the lift direction gives the pressure lift within 2 %; on a wing with 10 deg dihedral, sideslip $\beta > 0$ and a roll rate $p > 0$ give $C_{Y, \text{ff}} < 0$ and a yaw rate $r > 0$ gives $C_{Y, \text{ff}} > 0$ (the lattice has the same signs), and the opposite onset gives the opposite value; a planar wing and a symmetric flow give 0; on the NACA 0006 wing of aspect ratio 8 with 10 deg dihedral (16 x 16 mesh) the sideslip, yaw-rate and roll-rate values agree with the lattice `CY_farfield` (wake along the body $x$ axis) within 10 % (Morino and pressure Kutta; the measured values are in the test); the viscous coupling gives the inviscid values for the identity polar; the GPU batch and single-condition paths agree with numpy (float64 to $10^{-10}$, float32 to its rounding floor).

Float32 floor. On the GPU in float32, the doublets of a full-mesh solve are not symmetric to rounding. A case that is 0 in exact arithmetic then gives about $10^{-8}$ in `Cn_trefftz` and $10^{-7}$ in `CY_farfield`. Example: the planar NACA 0006 wing of aspect ratio 8 at $\beta = \pm 5$ deg (full mesh; `Cn_trefftz` is 0 in exact arithmetic, see above) gives $8.1 \times 10^{-8}$ for both signs of $\beta$ on the 24 x 24 mesh in float32, and about $2 \times 10^{-15}$ in float64 on the CPU and the GPU. The automatic backend can select the float32 GPU for this mesh size. These values are far below the floors of the near/far trust rule.

Trust. The 3D panel trust score applies the near-field and far-field lateral rule of the lattice (`ventorum.core.trust.near_far_lateral_warnings`, see [Forces and moments](forces_and_moments.md)): `CY` against `CY_farfield` in wind axes, and the near-field `Cn` in stability axes against `Cn_trefftz`, with the same floors. The viscous coupling compares the side force only, as the lattice coupling. On the NACA 0006 wing of aspect ratio 8 (16 x 16 mesh, alpha 4 deg) the rule does not fire for the yaw rates $\hat r = 0.01$ and $0.05$, for sideslip up to 5 deg on the planar wing, and for sideslip 5 deg and a yaw rate on the wing with 10 deg dihedral. On the planar wing in sideslip the body-axis Trefftz plane gives `Cn_trefftz` = 0 (see above), and the near-field `Cn` grows with $\beta$: at 5 deg the difference is 0.95 times the Cn floor, and at 8 deg the Cn warning fires, as for the lattice with either wake alignment. At 10 deg the near-field side force of the thick wing (0.00102) is above the `CY` floor, so the `CY` warning fires instead; the thin lattice stays below that floor. With a roll rate the near-field and far-field `Cn` have opposite signs in both solvers, and the Cn warning fires for both. The numbers come from a study script (not in the repository) that runs `vt.analyze` on these cases; the tests check the cases that must not fire.

## Computational backends

Ventorum-3DP supports three computational backends:
- `"numpy"` (default): Reference vectorised kernel evaluations and SciPy LAPACK LU solves on CPU.
- `"numba"`: JIT-compiled CPU kernels executing identical formulas with multithreaded loops and SciPy LU factorizations.
- `"gpu"`: GPU pipeline using NVIDIA Warp kernels for matrix assembly (one thread per panel-target pair) and PyTorch `torch.linalg.lu_factor` / `lu_solve` on the device. Unit onset solves are batched in a single call. Matrix assembly runs in float64; the matrices are kept, factorized and solved in the selected precision (float32 or float64). The gpu backend does no factorization on the CPU and keeps no copy of the matrices on the host. Before each configuration the solver estimates the device memory of its body, mirror and image matrices, systems and factors, and refuses a solve that does not fit (no silent fallback).

Float64 residual of the float32 solve. On the float32 path the system matrix is rounded to float32 after the float64 assembly, so the LU solve gives the solution of a nearby system. After the unit solve of each system, the solver measures the solution against the float64 system: it runs the float64 doublet pair kernels of the assembly again (body and mirror, and the ground image pair), one block of target rows at a time, multiplies each block by the solution in float64 and releases it, then adds the float64 wake strips through the trailing-edge fold. So it accumulates $A_{64} x$ without storing $A_{64}$. The right-hand sides $b = -R$ are the float64 source sums of the assembly. The dispatch record gives $\lVert b_k - A_{64} x_k \rVert_2 / \lVert b_k \rVert_2$ for each unit onset $k$ (`float64_relative_residuals`). The relative residual bounds the normwise backward error of the solution (Higham, 2002, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, chapter 7). The cost is about one more pass of the doublet kernel per system (no source kernel and no factorization) and one float64 block of rows (at most 64 MiB) of device memory, which the memory estimate already holds for the float32 assembly. The float64 GPU path and the CPU backends do not compute it. When a value is above 1e-4 (`FLOAT32_RESIDUAL_WARNING`, the same limit in free air and in ground effect) or is not finite, the trust of the result gets a warning, with no penalty: the float32 solution is then the exact solution of a system that differs from the float64 system by more than that relative amount, and the loads can have a relative error of that order times the condition number. The measured values of normal cases are 2.6e-7 to 1.65e-5 (the largest in ground effect), so the limit is about six times the largest measured value.

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
- Inputs: `Aircraft`, `FlightCondition` (or sequences for sweeps), `SolverSettings`, and an optional `GroundPlane` (keyword `ground`) that replaces `condition.h`. The panel fields of `SolverSettings` are `panel_kutta`, `panel_backend`, `panel_ground_solver` (`"auto"` by default), `panel_wake_length`, `panel_chord_spacing` (cosine by default), `panel_estimate_mesh_error`, `panel_scale_root_thickness` (True by default; the root section of a symmetric surface with dihedral keeps its thickness normal to the local surface, see the panel solver theory page) and `panel_cache` (True by default; keeps the mesh and the factorized systems of the last solves); `n_chord` is the chordwise panel count per surface (40 when None).
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
- No wake sheet touches or crosses another closed surface: the wake is rigid and flat, so a surface directly behind another surface at the same height is refused with `ValidityError` (see the panel solver chapter, section *Wake Through Another Surface*). The solver keeps this refusal. A wake that ends at a body and continues from it (the wake model of B. Maskew, "Program VSAERO theory document", NASA CR-4023, 1987) is not modelled.
- **Chordwise bias.** The low-order scheme (constant-strength source and doublet panels) at a trailing edge with a finite angle gives a lift that is low by about $(55 \text{ to } 85)/n_\text{chord}$ percent, for $t/c$ from 6 to 24 %. At the default `n_chord` of 40 the measured bias is -1.5 to -2.1 %. The 2D panel scheme of the same kind has the same error, so it is a property of the method, not a defect. Each 3D panel result gets a trust note with the bias range for its `n_chord` (no penalty). The Richardson extrapolation in `n_chord` estimates the error of a mesh: see [Run the 3D panel method](../user/how_to_panel3d.md), section *Estimate the chordwise error*.
- **Roll-rate side force.** The flat tip cap has one panel across the thickness and a chordwise-only surface gradient, so a planar wing gets $C_{Y_p} = 0$ by construction of the cap model, not by the physics (see [Body rates](rotation_rates.md), section *Envelope*).
- **Coarse-mesh leading-edge speed.** The least-squares node values of the surface gradient (see *Tangential velocity and pressure coefficient*) are exact for a field that is linear in the node tangent plane. At the leading edge of a coarse cosine chord the doublet strength is strongly curved across the nose, and the node value on the leading-edge fold is too low for it. The first upper panel row (chord index 0) is then too slow. On the AR 8 rectangular NACA 0012 wing of the yaw-rate test (16 spanwise panels per semi-span, alpha 4 deg), the median speed error of that row against a 48-panel chord (interpolated at the same $x$ on the same station) is -9.6 % with 16 chordwise panels (-9.2 % at $y = 0.2$ m), -6.3 % with 24 and -1.1 % with 32. The error falls with chord refinement; span refinement alone does not remove it. Rows 1 and 2 are good. The error acts on the leading-edge suction, so it moves the near-field values of the axial pressure force on coarse wings, most of all the near-field yaw damping. The stability $C_{n_r}$ is -0.002537 on the 16 x 16 mesh against -0.001702 for the vortex lattice (a known limit, marked `xfail` in `tests/test_panel3d_rotation_pressure.py`). With chord refinement it moves towards about -0.0019 to -0.0020: -0.002277 on 24 x 24, -0.002164 on 32 x 32 and -0.002234 on 16 x 32. The far-field value from `Cn_trefftz` does not use the surface gradient: -0.001777 on 16 x 16, -0.001815 on 24 x 24 and -0.001836 on 32 x 32. Use the far-field $C_{n_r}$ on coarse meshes, or refine the chord. Source of the numbers: the row speeds come from `test_coarse_leading_edge_row_speed_converges_with_the_chord` in `tests/test_panel3d_rotation_pressure.py`; the derivatives come from central differences of `vt.analyze` in the stability yaw rate (the method of the yaw-rate tests in that file) on the listed meshes.
- **Dihedral breaks.** At a dihedral break inside a surface (a gull wing), the peak of the strip lift at the break grows as the mesh is refined, so the strip loads next to the break are not converged. A break above `ventorum.core.trust.DIHEDRAL_BREAK_NOTE_DEG` (5 deg) gives a trust note (no penalty). Use the integrated loads.

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
- Far-field multipole expansions (order of the per-pair error, lift with spanwise refinement against exact integration, backend parity): `tests/test_panel3d_far_field.py`.
- Mesh construction and control surface deflections: `tests/test_panel3d_mesh.py`, `tests/test_panel3d_controls.py`.
- Potential flow solution, unit onsets, surface gradient and Trefftz drag (sphere in two orientations, ellipsoid on all panels, stored free-air run, factorization count): `tests/test_panel3d_solver.py`.
- Wake sheets through another surface (refused tandem, unchanged offset tandem, wing and tail, biplane): `tests/test_panel3d_wake_crossing.py`.
- Symmetry plane and ground effect images (explicit mirror with the free-stream wake, Trefftz drag with images, bank and sideslip on a half mesh): `tests/test_panel3d_images.py`.
- Iterative pressure Kutta condition (stored Morino run, asymmetric half mesh against the full mesh): `tests/test_panel3d_kutta.py`.
- Mesh convergence and Richardson extrapolation: `tests/test_panel3d_convergence.py`.
- Refinement regressions (open trailing edge with chordwise refinement, dihedral root strip with spanwise refinement): `tests/test_panel3d_refinement.py`; flapped wing with spanwise refinement: `tests/test_panel3d_controls.py`.
- Solver family integration, settings, ground plane, strip map and trust scores: `tests/test_panel3d_factory.py`.
- GPU pipeline parity (when CUDA is available): `tests/test_panel3d_gpu.py`.
- Yawing moment of the Trefftz-plane induced drag (`Cn_trefftz`: independent strip sum, sign, lattice parity, all paths, GPU): `tests/test_panel3d_trefftz_yaw.py`.
- Side force of the Trefftz plane (`CY_farfield`: independent Kutta-Joukowski sum, lift sign check, sign, lattice parity, all paths, viscous coupling, GPU) and the near/far lateral trust rule: `tests/test_panel3d_trefftz_side_force.py`.

## References

- J. D. Anderson, *Fundamentals of Aerodynamics*, 5th ed., McGraw-Hill, New York, 2011.
- I. B. Celik, U. Ghia, P. J. Roache, C. J. Freitas, H. Coleman and P. E. Raad, "Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications", *Journal of Fluids Engineering*, Vol. 130, No. 7, 078001, 2008.
- M. Drela, *Flight Vehicle Aerodynamics*, MIT Press, Cambridge, MA, 2014.
- J. Blazek, *Computational Fluid Dynamics: Principles and Applications*, 3rd ed., Butterworth-Heinemann, Oxford, 2015, chapter 5.
- J. L. Hess and A. M. O. Smith, "Calculation of Potential Flow About Arbitrary Bodies", *Progress in Aeronautical Sciences*, Vol. 8, Pergamon Press, 1967, pp. 1-138.
- J. D. Jackson, *Classical Electrodynamics*, 3rd ed., John Wiley & Sons, New York, 1999, section 4.1.
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
- E. Trefftz, "Prandtlsche Tragflaechen- und Propeller-Theorie", *Zeitschrift fuer angewandte Mathematik und Mechanik*, Vol. 1, 1921.
- C. Wieselsberger, "Wing Resistance Near the Ground", NACA TM-77, 1922; translated from *Zeitschrift fuer Flugtechnik und Motorluftschiffahrt*, Vol. 12, No. 10, 1921.
