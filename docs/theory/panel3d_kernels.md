# Ventorum-3DP Influence Kernels

This document gives the mathematical formulation, sign conventions, and asymptotic rules for the constant-strength source and doublet quadrilateral panel kernels of Ventorum-3DP.

## Overview

Ventorum-3DP implements a low-order potential flow panel method with an internal Dirichlet boundary condition. The flow field is represented by singularity distributions of constant source strength $\sigma$ and constant doublet strength $\mu$ distributed over flat polygonal panels.

Each panel is specified by four ordered vertices $\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3, \mathbf{p}_4$ with coordinates in meters [m]. Triangular panels are represented as quadrilaterals with the fourth vertex equal to the third vertex ($\mathbf{p}_4 = \mathbf{p}_3$).

## Panel Geometry and Coordinate Systems

### Unit Normal Vector

The panel normal is determined from the diagonal vectors:

$$\mathbf{d}_{13} = \mathbf{p}_3 - \mathbf{p}_1, \quad \mathbf{d}_{24} = \mathbf{p}_4 - \mathbf{p}_2$$

The outward normal vector points into the fluid (out of the body) by the right-hand rule:

$$\mathbf{n}_{\text{raw}} = \mathbf{d}_{13} \times \mathbf{d}_{24}$$

$$\hat{\mathbf{n}} = \frac{\mathbf{n}_{\text{raw}}}{\|\mathbf{n}_{\text{raw}}\|}$$

The projected panel area $A$ in square meters [m^2] satisfies:

$$A = \frac{1}{2} \|\mathbf{n}_{\text{raw}}\|$$

### Planar Projection

Non-planar quadrilaterals are projected onto the mean plane through the corner average $\mathbf{p}_{\text{avg}} = \frac{1}{4} \sum_{k=1}^4 \mathbf{p}_k$ with normal $\hat{\mathbf{n}}$:

$$\mathbf{p}'_k = \mathbf{p}_k - [(\mathbf{p}_k - \mathbf{p}_{\text{avg}}) \cdot \hat{\mathbf{n}}] \hat{\mathbf{n}}$$

The panel reference point (collocation point) $\mathbf{p}_0$ is the corner average of the projected corners:
* For quadrilaterals: $\mathbf{p}_0 = \frac{1}{4} \sum_{k=1}^4 \mathbf{p}'_k$. For a general quadrilateral this point is not the area centroid (for example, the area centroid of a trapezoid is nearer to its longer side).
* For triangles ($\mathbf{p}_4 = \mathbf{p}_3$): $\mathbf{p}_0 = \frac{1}{3} (\mathbf{p}'_1 + \mathbf{p}'_2 + \mathbf{p}'_3)$, which is also the area centroid.

A corner counts as repeated (triangle) and an edge counts as degenerate when its length is at most $10^{-12}$ times the panel diameter.

### Local Coordinate Frame

An orthonormal local reference frame $(\hat{\mathbf{l}}, \hat{\mathbf{m}}, \hat{\mathbf{n}})$ is established on each panel:

$$\hat{\mathbf{l}} = \frac{\mathbf{p}'_2 - \mathbf{p}'_1}{\|\mathbf{p}'_2 - \mathbf{p}'_1\|}, \quad \hat{\mathbf{m}} = \hat{\mathbf{n}} \times \hat{\mathbf{l}}$$

For any target point $\mathbf{x} \in \mathbb{R}^3$, the local coordinates relative to the centroid are:

$$x = (\mathbf{x} - \mathbf{p}_0) \cdot \hat{\mathbf{l}}, \quad y = (\mathbf{x} - \mathbf{p}_0) \cdot \hat{\mathbf{m}}, \quad z = (\mathbf{x} - \mathbf{p}_0) \cdot \hat{\mathbf{n}}$$

Corner positions in the local frame are denoted by $(x_k, y_k, 0)$ for $k = 1, \dots, 4$.

The distance from the target point to corner $k$ is:

$$r_k = \sqrt{(x - x_k)^2 + (y - y_k)^2 + z^2}$$

For each edge $k$ from $(x_k, y_k)$ to $(x_{k+1}, y_{k+1})$ with length $d_k = \sqrt{(x_{k+1} - x_k)^2 + (y_{k+1} - y_k)^2}$, the in-plane perpendicular distance is:

$$r_{\perp, k} = \frac{(x - x_k)(y_{k+1} - y_k) - (y - y_k)(x_{k+1} - x_k)}{d_k}$$

The edge logarithmic integral is:

$$L_k = \ln\left( \frac{r_k + r_{k+1} + d_k}{r_k + r_{k+1} - d_k} \right)$$

The denominator has cancellation near the edge. The kernels use the identities

$$r_k + r_{k+1} - d_k = \frac{2 (r_k r_{k+1} + \mathbf{a} \cdot \mathbf{b})}{r_k + r_{k+1} + d_k}, \qquad r_k r_{k+1} + \mathbf{a} \cdot \mathbf{b} = \frac{\|\mathbf{a} \times \mathbf{b}\|^2}{r_k r_{k+1} - \mathbf{a} \cdot \mathbf{b}} \ \text{for} \ \mathbf{a} \cdot \mathbf{b} < 0,$$

where $\mathbf{a}$ and $\mathbf{b}$ are the vectors from the target to the two edge ends and $\|\mathbf{a} \times \mathbf{b}\|^2 = d_k^2 (r_{\perp, k}^2 + z^2)$. When the target lies on the edge segment or on an edge end, $L_k$ is infinite; the kernels set $L_k = 0$ there (the finite part). The source potential stays continuous, because $r_{\perp, k} L_k \to 0$.

## Sign Conventions

1. Source potential: A unit point source produces potential $\Phi_S = -\frac{1}{4 \pi r}$ in the far field.
2. Doublet potential: A unit doublet oriented along $+\hat{\mathbf{n}}$ produces far-field potential $\Phi_D = -\frac{A (\mathbf{r} \cdot \hat{\mathbf{n}})}{4 \pi r^3}$.
3. Discontinuity across the panel:
   * Approaching the panel from the fluid side ($z \to 0^+$), $\Phi_D \to -1/2$.
   * Approaching the panel from the body side ($z \to 0^-$), $\Phi_D \to +1/2$.
   * Total potential jump: $\Phi_D^+ - \Phi_D^- = -1$.
4. On-plane band: a target counts as a point of the panel plane when

   $$|z| \le \tau = 32 \, \epsilon \, (\|\mathbf{p}_0\| + D),$$

   with $\epsilon$ the machine epsilon of the input dtype (float64 or float32) and $D$ the panel diameter. The band is relative to the panel size and to the coordinate magnitude. So the round-off of collocation points far from the origin, or of float32 inputs, does not move a self term to the other side of the panel.
5. Body-side limit in the band: the doublet potential is the limit from the body (interior) side. It equals $W / (4 \pi)$, where $W$ is the angle that the panel subtends in its plane at the target:

   $$W = \sum_k \operatorname{atan2}\left(\mathbf{a}_k \times \mathbf{a}_{k+1}, \ \mathbf{a}_k \cdot \mathbf{a}_{k+1}\right), \qquad \mathbf{a}_k = (x_k - x, \ y_k - y).$$

   An edge that holds the target (distance from its line at most $\tau$ and between its ends, or an end within $\tau$) adds nothing. This gives $+1/2$ inside the panel (the self-influence at the own centroid, consistent with the internal Dirichlet condition), $0$ outside, $+1/4$ on an edge and $\theta / (4 \pi)$ at a corner with the interior angle $\theta$. These values are also the limits from the body side along the normal.
6. Velocity definition: the kernels return the gradient of the potential,

   $$\mathbf{V} = +\nabla \Phi.$$

   This is the convention of the solver and its post-processing (the perturbation velocity is the gradient of the perturbation potential). A unit source pushes the fluid away from the panel, and the normal velocity increases by $1$ from the body side to the fluid side.

## Exact Influence Kernels

### Solid Angle

The panel is decomposed into two triangles: $(0, 1, 2)$ and $(0, 2, 3)$. For each triangle with vertices $\mathbf{a}, \mathbf{b}, \mathbf{c}$ relative to the target point, the solid angle is computed with the van Oosterom and Strackee (1983) formula:

$$\tan\left(\frac{\Omega_\Delta}{2}\right) = \frac{\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})}{\|\mathbf{a}\| \|\mathbf{b}\| \|\mathbf{c}\| + (\mathbf{a}\cdot\mathbf{b})\|\mathbf{c}\| + (\mathbf{a}\cdot\mathbf{c})\|\mathbf{b}\| + (\mathbf{b}\cdot\mathbf{c})\|\mathbf{a}\|}$$

$$\Omega = \Omega_{\Delta, 1} + \Omega_{\Delta, 2}$$

### Doublet Potential

The doublet potential per unit strength is:

$$\Phi_D = \frac{\Omega}{4 \pi}$$

### Doublet Velocity

By Stokes theorem, the velocity induced by a constant-strength doublet panel equals the velocity of an equivalent vortex ring of circulation $\Gamma = 1$ that runs along the panel perimeter in the corner order (Katz and Plotkin 2001, section 10.4):

$$\mathbf{V}_D = \nabla \Phi_D = \mathbf{v}_{\text{ring}} = \sum_{k=1}^4 \mathbf{v}_{\text{segment}}(\mathbf{p}'_k \to \mathbf{p}'_{k+1})$$

Each straight vortex segment velocity comes from the Biot-Savart law. With $\mathbf{r}_1 = \mathbf{x} - \mathbf{p}'_k$, $\mathbf{r}_2 = \mathbf{x} - \mathbf{p}'_{k+1}$ and $\mathbf{r}_0 = \mathbf{r}_1 - \mathbf{r}_2$, the usual form is

$$\mathbf{v}_{\text{segment}} = \frac{1}{4 \pi} \frac{\mathbf{r}_1 \times \mathbf{r}_2}{\|\mathbf{r}_1 \times \mathbf{r}_2\|^2} \left[ \mathbf{r}_0 \cdot \left( \frac{\mathbf{r}_1}{r_1} - \frac{\mathbf{r}_2}{r_2} \right) \right].$$

The expansions $\mathbf{r}_0 \cdot (\mathbf{r}_1 / r_1 - \mathbf{r}_2 / r_2) = (r_1 + r_2)(r_1 r_2 - \mathbf{r}_1 \cdot \mathbf{r}_2) / (r_1 r_2)$ and $\|\mathbf{r}_1 \times \mathbf{r}_2\|^2 = (r_1 r_2)^2 - (\mathbf{r}_1 \cdot \mathbf{r}_2)^2$ give the identical form that the kernels use:

$$\mathbf{v}_{\text{segment}} = \frac{1}{4 \pi} \frac{(r_1 + r_2) \, (\mathbf{r}_1 \times \mathbf{r}_2)}{r_1 r_2 \, (r_1 r_2 + \mathbf{r}_1 \cdot \mathbf{r}_2)}.$$

This form has no cancellation on the extension of the segment line, where the velocity is zero. For $\mathbf{r}_1 \cdot \mathbf{r}_2 < 0$ the kernels evaluate $r_1 r_2 + \mathbf{r}_1 \cdot \mathbf{r}_2 = \|\mathbf{r}_1 \times \mathbf{r}_2\|^2 / (r_1 r_2 - \mathbf{r}_1 \cdot \mathbf{r}_2)$. A target on the segment (within $\tau$ of its line, between its ends) or within $\tau$ of an end is on the vortex line. The kernels then drop that segment (zero contribution) and keep the finite rest.

### Source Potential

The source potential per unit strength is:

$$\Phi_S = \frac{1}{4 \pi} \left[ \sum_{k=1}^4 r_{\perp, k} L_k - z \Omega \right]$$

In the on-plane band, $\Omega$ is the body-side limit $W$.

### Source Velocity

In local panel coordinates ($\mathbf{V}_S = +\nabla \Phi_S$):

$$V_{S, x} = +\frac{1}{4 \pi} \sum_{k=1}^4 \frac{y_{k+1} - y_k}{d_k} L_k$$

$$V_{S, y} = -\frac{1}{4 \pi} \sum_{k=1}^4 \frac{x_{k+1} - x_k}{d_k} L_k$$

$$V_{S, z} = -\frac{1}{4 \pi} \Omega = -\Phi_D$$

On the fluid side next to the panel ($\Omega \to -2 \pi$) the normal component tends to $+1/2$; on the body side ($\Omega \to +2 \pi$) it tends to $-1/2$. In the on-plane band the kernels use the body-side value. On an edge the in-plane components have a logarithmic singularity; the kernels drop the singular edge term there.

In global Cartesian coordinates:

$$\mathbf{V}_S = V_{S, x} \hat{\mathbf{l}} + V_{S, y} \hat{\mathbf{m}} + V_{S, z} \hat{\mathbf{n}}$$

## Far-Field Approximations

When the distance $r = \|\mathbf{x} - \mathbf{p}_c\|$ from the area centroid $\mathbf{p}_c$ of the panel exceeds `far_ratio` times the panel diameter $D = \max_{i < j} \|\mathbf{p}_i - \mathbf{p}_j\|$, a multipole expansion about $\mathbf{p}_c$ replaces the exact integrals. It keeps the area $A$ and the second moments of area $\mathbf{Q} = \int \mathbf{s} \mathbf{s}^{\mathsf{T}} \, dA$ of the panel (Cartesian multipole expansion, Jackson 1999, section 4.1). With $a = \mathbf{r}^{\mathsf{T}} \mathbf{Q} \mathbf{r}$, $q = \operatorname{tr} \mathbf{Q}$ and $z = \mathbf{r} \cdot \hat{\mathbf{n}}$:

$$\Phi_S^{\text{far}} = -\frac{1}{4 \pi} \left[ \frac{A}{r} + \frac{3 a - r^2 q}{2 r^5} \right], \quad \Phi_D^{\text{far}} = \frac{z}{4 \pi} \left[ \frac{3 q / 2 - 15 a / (2 r^2)}{r^5} - \frac{A}{r^3} \right]$$

The velocities are the gradients of these potentials. The first terms ($A$ only) are the point source $-A / (4 \pi r)$ and the point doublet $-A z / (4 \pi r^3)$. The full formulas, the moments of the panel and the error table are in the [Ventorum-3DP chapter](ventorum_3dp.md) (section *Far-field multipole expansions*). The collocation point stays at the corner mean.

Setting `far_ratio = np.inf` disables the approximation and uses exact integration everywhere.

### Accuracy of the expansion

The relative error of one pair is $O((D / r)^3)$ for a general quadrilateral and $O((D / r)^4)$ for a parallelogram. The point singularities alone have the relative error $O((D / r)^2)$, and $O(D / r)$ when they are taken about the corner mean of a quadrilateral that is not a parallelogram. `tests/test_panel3d_far_field.py` checks these orders against the exact kernels. The tests in `tests/test_panel3d_kernels.py` check the accuracy at $r = 10 D$ and $r = 25 D$ for a square, a skewed quadrilateral, an aspect-ratio-20 rectangle and a triangle.

## References

* J. D. Jackson, "Classical Electrodynamics", 3rd ed., John Wiley & Sons, 1999, section 4.1.
* J. L. Hess and A. M. O. Smith, "Calculation of non-lifting potential flow about arbitrary three-dimensional bodies", Journal of Ship Research 8 (1964) 22-44.
* J. L. Hess and A. M. O. Smith, "Calculation of potential flow about arbitrary bodies", Progress in Aeronautical Sciences 8 (1967) 1-138, Pergamon Press.
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge University Press, 2001.
* J. N. Newman, "Distributions of sources and normal dipoles over a quadrilateral panel", Journal of Engineering Mathematics 20 (1986) 113-126.
* A. van Oosterom and J. Strackee, "The solid angle of a plane triangle", IEEE Transactions on Biomedical Engineering 30 (1983) 125-126.
