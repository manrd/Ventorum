---
orphan: true
---

# 2D Inviscid Panel Solver with Ground Effect

This document describes the mathematical formulation of the two-dimensional inviscid panel solver in Ventorum (`ventorum.panel2d`).

## Formulation Overview

The solver uses a Dirichlet boundary condition on the stream function $\psi$.
The fluid is incompressible, inviscid, and irrotational.
The velocity vector $(u, v)$ relates to the stream function by:
$$u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$$

The total stream function $\psi$ is the superposition of the free stream, vortex panels, and source panels:
$$\psi(x, y) = \psi_\infty(x, y) + \psi_\gamma(x, y) + \psi_\sigma(x, y)$$

In free air, with angle of attack $\alpha$, the free-stream stream function is:
$$\psi_\infty(x, y) = V_\infty (y \cos\alpha - x \sin\alpha)$$

The Dirichlet boundary condition enforces that the surface of each lifting body forms a streamline:
$$\psi(x_i, y_i) = \Psi_b$$
where $(x_i, y_i)$ are collocation points at the surface nodes of body $b$, and $\Psi_b$ is an unknown constant stream function value for that body.

## Panel Singularities and Stream Function Influences

The body contour is discretized into $M$ flat panels connected by $N$ nodes.
Node coordinates are ordered in Selig convention (from the upper trailing edge, forward around the leading edge, to the lower trailing edge).

### Linear-Strength Vortex Panels

Vorticity varies linearly over each panel $j$ from $\gamma_1$ at local coordinate $s = 0$ to $\gamma_2$ at $s = L$:
$$\gamma(s) = \gamma_1 \left(1 - \frac{s}{L}\right) + \gamma_2 \left(\frac{s}{L}\right)$$

A point vortex of strength $d\Gamma = \gamma(s) ds$ induces a stream function at field point $(\xi, \eta)$:
$$d\psi = \frac{1}{4\pi} \ln(r^2) d\Gamma = \frac{1}{4\pi} \ln\left((\xi - s)^2 + \eta^2\right) \gamma(s) ds$$

Let $t = \xi - s$, so $dt = -ds$.
When $s = 0$, $t = \xi$.
When $s = L$, $t = \xi - L$.
The required indefinite integrals are:
$$F_0(t, \eta) = \int \ln(t^2 + \eta^2) dt = t \ln(t^2 + \eta^2) - 2t + 2\eta \arctan\left(\frac{t}{\eta}\right)$$
$$F_1(t, \eta) = \int t \ln(t^2 + \eta^2) dt = \frac{1}{2}(t^2 + \eta^2) \left(\ln(t^2 + \eta^2) - 1\right)$$

Over the panel length from $s = 0$ to $s = L$:
$$I_0 = \int_0^L \ln\left((\xi - s)^2 + \eta^2\right) ds = F_0(\xi, \eta) - F_0(\xi - L, \eta)$$
$$I_1 = \int_0^L s \ln\left((\xi - s)^2 + \eta^2\right) ds = \xi I_0 - \left[F_1(\xi, \eta) - F_1(\xi - L, \eta)\right]$$

The stream function influence of panel $j$ with unit nodal strengths is:
$$\psi = \psi_1 \gamma_1 + \psi_2 \gamma_2$$
where:
$$\psi_1 = \frac{1}{4\pi} \left[ \left(1 - \frac{\xi}{L}\right) I_0 + \frac{1}{L} \left(F_1(\xi, \eta) - F_1(\xi - L, \eta)\right) \right]$$
$$\psi_2 = \frac{1}{4\pi} \left[ \frac{\xi}{L} I_0 - \frac{1}{L} \left(F_1(\xi, \eta) - F_1(\xi - L, \eta)\right) \right]$$

### Constant-Strength Source Panels

A source panel has uniform strength $\sigma$.
A point source of strength $dQ = \sigma ds$ induces a stream function:
$$d\psi = \frac{\sigma}{2\pi} \theta(s) ds$$
where $\theta(s)$ is the angle from the source point $(s, 0)$ to the field point $(\xi, \eta)$ in local coordinates:
$$\theta(s) = \arctan\left(\frac{\eta}{\xi - s}\right)$$

Integrating by parts with $t = \xi - s$:
$$\int \arctan\left(\frac{\eta}{t}\right) dt = t \arctan\left(\frac{\eta}{t}\right) + \frac{1}{2}\eta \ln(t^2 + \eta^2)$$

Transforming to global coordinates with panel orientation angle $\theta_{\text{panel}}$ gives the closed-form stream function:
$$\psi_\sigma = \frac{\sigma}{2\pi} \left[ \xi \arctan\left(\frac{\eta}{\xi}\right) + \frac{1}{2}\eta \ln(\xi^2 + \eta^2) - (\xi - L) \arctan\left(\frac{\eta}{\xi - L}\right) - \frac{1}{2}\eta \ln\left((\xi - L)^2 + \eta^2\right) + \theta_{\text{panel}} L \right]$$

## Boundary Conditions and System of Equations

### Surface Dirichlet Condition

On each body surface, the stream function satisfies:
$$\sum_{j=1}^{N_{\text{panels}}} \left( \psi_{1, ij} \gamma_{1, j} + \psi_{2, ij} \gamma_{2, j} \right) - \Psi_b = -\psi_\infty(x_i, y_i) - \sum_{j=1}^{N_{\text{panels}}} \psi_{\sigma, ij} \sigma_j$$

For $N_b$ nodes on body $b$, this provides $N_b$ equations.
The value $\Psi_b$ is an unknown scalar for each body.

### Kutta Condition

Flow must leave the trailing edge smoothly.

1. **Sharp Trailing Edge (gap $\le 10^{-4} c$)**:
The upper and lower trailing edge vortex strengths satisfy:
$$\gamma_{\text{first}} + \gamma_{\text{last}} = 0$$
This condition closes the circulation.

2. **Blunt Trailing Edge (gap $> 10^{-4} c$)**:
A flat base panel connects the lower trailing edge node ($N-1$) to the upper trailing edge node (0).
The vortex strength on the base panel varies linearly from $\gamma_{N-1}$ to $\gamma_0$.
The Kutta condition enforces zero net jump in pressure across the base panel.

## Two-Right-Hand-Side Superposition in Free Air

In free air, the geometric influence matrix does not depend on the angle of attack $\alpha$.
The solver factorizes the matrix once with LU decomposition.
It solves two fundamental right-hand sides:
1. Unit free stream along $x$: $\psi_{\infty, x} = y$
2. Unit free stream along $y$: $\psi_{\infty, y} = -x$

For any list of angles of attack, the solution vector is obtained by linear superposition:
$$\mathbf{x}(\alpha) = \cos\alpha \, \mathbf{x}_x + \sin\alpha \, \mathbf{x}_y + \mathbf{x}_\sigma$$
Prescribed source strengths $\sigma$ enter as an additional right-hand side term without re-factorizing the matrix.

## Ground Effect via Method of Images

Ground effect simulates flow over an infinite flat wall at $y = 0$.
The method of images places an inverted mirror geometry at $(x, -y)$ below the ground plane.

### Sign Rules for Image Panels

1. **Vortex image panel**:
A vortex of strength $\gamma$ at $(x, y)$ induces vertical velocity $u_y \sim \gamma (x - x_0) / r^2$.
To cancel the normal velocity at the ground plane ($y = 0$), the image panel at $(x, -y)$ must have opposite vorticity:
$$\gamma_{\text{image}} = -\gamma_{\text{real}}$$
Because vorticity changes sign, its stream function influence subtracts:
$$A_{\text{vortex}}^{\text{GE}} = A_{\text{vortex}}^{\text{real}} - A_{\text{vortex}}^{\text{image}}$$

2. **Source image panel**:
A source of strength $\sigma$ at $(x, y)$ induces normal velocity away from the wall.
To cancel the normal velocity across the ground plane, the image panel must have equal source strength:
$$\sigma_{\text{image}} = +\sigma_{\text{real}}$$
Its stream function influence adds:
$$A_{\text{source}}^{\text{GE}} = A_{\text{source}}^{\text{real}} + A_{\text{source}}^{\text{image}}$$

In ground effect, the angle of attack rotates the physical body about a pivot point (default quarter chord) and positions it at height $h$.
The free stream remains parallel to the ground along $+x$.
Matrices for ground effect are assembled and solved in batches.

## Forces and Moments

1. **Circulation Lift (Kutta-Joukowski)**:
The circulation of each body is computed by integrating vorticity around its perimeter:
$$\Gamma = \oint \gamma(s) ds = \sum_{j=1}^{M} \frac{1}{2}(\gamma_{1, j} + \gamma_{2, j}) L_j$$
The lift coefficient is:
$$C_l = \frac{2\Gamma}{V_\infty c_{\text{ref}}}$$

2. **Surface Pressure Coefficient**:
From the Bernoulli equation, the surface tangential velocity is $u_e = \gamma$ (non-dimensionalized by $V_\infty$).
The pressure coefficient at each node is:
$$C_p = 1 - \left(\frac{u_e}{V_\infty}\right)^2 = 1 - \gamma^2$$

3. **Pressure Integration**:
Integrating $C_p$ along panel normal vectors gives normal force $C_n$, axial force $C_a$, and pitching moment $C_m$ about reference point $(x_{\text{ref}}, y_{\text{ref}})$.

## References

1. J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001, chapter 11.
2. M. Drela, "XFOIL: An Analysis and Design System for Low Reynolds Number Airfoils", *Lecture Notes in Engineering*, Vol. 54, Springer, 1989, pp. 1-12.
3. L. M. Milne-Thomson, *Theoretical Aerodynamics*, 4th ed., Dover Publications, 1958.
