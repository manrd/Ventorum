---
orphan: true
---

# 2D Viscous-Inviscid Panel Solver

This document describes the mathematical formulation of the two-dimensional coupled viscous-inviscid panel solver in Ventorum (`ventorum.panel2d.viscous`).

## Physical Model and Coupling Overview

Real fluid flows at moderate and high Reynolds numbers develop thin shear layers near solid walls.
The flow field divides into two regions:
1. An outer potential flow region where viscosity is negligible.
2. An inner boundary layer and wake region where viscous diffusion is significant.

The outer flow governs the pressure field and edge velocity.
The inner boundary layer retards the fluid, creating a displacement thickness $\delta^*$.
This displacement deflects the outer streamlines away from the body surface.

The solver models this displacement effect through wall transpiration (Lighthill, 1958).
Instead of moving the physical mesh, equivalent normal blowing velocities are applied at the original solid surface.

## Wall Transpiration and Mass Defect

The mass flow defect in the boundary layer across stations is:
$$m(s) = u_e(s) \delta^*(s)$$
where:
- $u_e$ is the outer edge velocity ratio (dimensionless).
- $\delta^*$ is the boundary layer displacement thickness [m].
- $s$ is the surface arc length coordinate [m].

By continuity, the equivalent normal transpiration velocity at the wall is:
$$v_n(s) = \frac{d}{ds} \left[ u_e(s) \delta^*(s) \right]$$

In the panel solver, each panel $j$ carries a constant-strength source of strength:
$$\sigma_j = \frac{m_{j+1} - m_j}{\Delta s_j}$$
where $\Delta s_j$ is the panel length [m].

## Mass-Defect Influence Matrix

The inviscid flow satisfies the stream function Dirichlet boundary condition:
$$A \gamma = \text{rhs}_{inv} - \Psi_s \sigma$$
where:
- $A$ is the inviscid linear vortex influence matrix of shape $(N+1, N+1)$.
- $\gamma$ is the nodal vortex sheet strength vector of length $N$.
- $\Psi_s$ is the source panel stream function influence matrix of shape $(N, M)$.
- $\sigma$ is the panel source strength vector of length $M$.

The linear system decomposes by superposition:
$$\gamma = \gamma_{inv} + D \sigma$$
where the mass-defect influence matrix $D$ is:
$$D = -A^{-1} \Psi_s$$

Because $A$ depends only on the body geometry, its LU factorization is computed once.
Subsequent boundary layer iterations solve for edge velocity updates using back-substitution without re-factorizing $A$:
$$u_e = u_{e, inv} + D \sigma$$

## Integral Boundary Layer System

The boundary layer is modeled by two integral equations for momentum and kinetic energy:
1. Von Karman momentum integral equation:
$$\frac{d\theta}{ds} + (2 + H) \frac{\theta}{u_e} \frac{du_e}{ds} = \frac{C_f}{2}$$
2. Kinetic energy shape parameter equation:
$$\theta \frac{dH^*}{ds} + H^* (1 - H) \frac{\theta}{u_e} \frac{du_e}{ds} = 2 C_D - H^* \frac{C_f}{2}$$
where:
- $\theta$ is the momentum thickness [m].
- $H = \delta^* / \theta$ is the kinematic shape factor (dimensionless).
- $H^* = \theta^* / \theta$ is the kinetic energy shape factor (dimensionless).
- $C_f$ is the skin-friction coefficient (dimensionless).
- $C_D$ is the dissipation coefficient (dimensionless).

The equations are closed using the laminar and turbulent closure relations of Drela and Giles (1987).

## Stagnation Point and Surface Marching

The stagnation point is located by finding the zero-crossing of the edge velocity:
$$u_e(s_{stag}) = 0$$
The node closest to the stagnation point serves as the origin for both the upper and lower surface boundary layer marches.

At the stagnation point, self-similar Hiemenz flow supplies the initial condition:
$$H_0 = 2.216$$
$$\theta_0 = \sqrt{\frac{F_0}{(2 + H_0) a_{stag} Re}}$$
where:
- $a_{stag} = |du_e / ds|_{stag}$ is the velocity gradient at the stagnation point [m^-1].
- $F_0 = (Re_\theta C_f / 2)_0$ is the stagnation skin-friction factor (dimensionless).
- $Re$ is the Reynolds number based on chord (dimensionless).

## Transition Prediction

Boundary layer transition is evaluated by two mechanisms:
1. Forced transition at user-specified stations $x_{tr} / c$.
2. Free transition via the $e^N$ envelope amplification method.

The amplification exponent $N$ integrates the spatial amplification rate:
$$\frac{dN}{ds} = \frac{dN}{dRe_\theta} \frac{dRe_\theta}{ds}$$
Transition occurs when $N$ reaches the critical threshold $N_{crit}$ (typically 9.0 for quiet wind tunnels), or when laminar separation occurs ($H \ge 3.6$).

Upon transition, the shape factor resets to an attached turbulent value ($H \approx 1.45$), establishing a continuous momentum thickness across the transition front.

## Profile Drag Calculation

Profile drag is calculated by the Squire-Young formula evaluated at the downstream boundary layer state:
$$C_d = 2 \theta_{TE} \left( \frac{u_{e, TE}}{V_\infty} \right)^{\frac{H_{TE} + 5}{2}}$$
where:
- $\theta_{TE} = \theta_{upper} + \theta_{lower} + \theta_{base}$ is the combined momentum thickness at the trailing edge [m].
- $H_{TE} = (\delta^*_{upper} + \delta^*_{lower} + h_{te}) / \theta_{TE}$ is the trailing edge shape factor (dimensionless).
- $u_{e, TE}$ is the average edge velocity near the trailing edge (dimensionless).

As a consistency check, drag is also computed by nearfield pressure and skin-friction surface integration:
$$C_{d, near} = C_{d, pressure} + C_{d, friction}$$
$$C_{d, pressure} = \frac{1}{c} \oint C_p n_x ds + C_{D, base} \frac{h_{te}}{c}$$
$$C_{d, friction} = \frac{1}{c} \oint C_f t_x ds$$
where $n_x$ and $t_x$ are the surface normal and tangential components along the drag axis.

## Streamline Wake Tracing and Wake Source Panels

The wake trajectory is traced from the trailing-edge bisector along the local potential streamline (Drela, 1989).
The trajectory integration uses a second-order Runge-Kutta (Heun predictor-corrector) scheme.
Wake panel lengths expand geometrically away from the trailing edge with ratio $r = 1.10$.

For symmetric airfoils at zero angle of attack, the wake follows the chord line exactly ($y = 0$).
At positive angles of attack, the wake deflects downwash-wise and aligns asymptotically with the free-stream direction.

The wake carries source panels to model its displacement defect:
$$m_w(s) = |u_{e, w}(s)| \delta^*_w(s)$$
$$\sigma_{w, j} = \frac{m_w(s_{j+1}) - m_w(s_j)}{\Delta s_{w, j}}$$
The wake edge velocity relaxes toward the free stream:
$$u_{e, w}(s) = 1 + (u_{e, TE} - 1) e^{-3 s / c}$$
The stream function influence of the wake source panels is integrated continuously without branch cuts across the airfoil surface.

## Blunt Trailing Edge Treatment

For airfoils with a finite trailing-edge thickness $h_{te}$, the base creates a separated dead-air region (Hoerner, 1965).
The wake begins with the geometric gap added to the boundary layer displacement:
$$\delta^*_{w, 0} = \delta^*_{upper} + \delta^*_{lower} + h_{te}$$

The momentum thickness includes a base momentum defect increment:
$$\theta_{w, 0} = \theta_{upper} + \theta_{lower} + \frac{1}{2} C_{D, base} h_{te}$$
where the base pressure drag coefficient is parameterized by the trailing-edge edge velocity:
$$C_{D, base} = \max(u_{e, TE}^2 - 1.0, 0.15)$$

This base momentum deficit enters the Squire-Young drag integration and produces a drag increment that scales with $h_{te} / c$.

## Flapped Sections and Separation Handling

Plain trailing-edge flap deflections induce concentrated adverse pressure gradients near the hinge line on the suction surface.
To prevent numerical breakdown from the Goldstein boundary-layer singularity near separation, the march applies a quasi-inverse deceleration limiter (Veldman, 2009):
$$\left| \frac{du_e}{ds} \right| \le 0.3 \bar{u}_e \frac{\Delta s}{\theta}$$

Separation is detected when either:
1. Kinematic shape factor $H$ reaches the laminar separation threshold ($H \ge 3.8$) or turbulent separation threshold ($H \ge 2.6$).
2. Skin-friction coefficient falls to zero ($C_f \le 0$).

When separation occurs, the solver flags the separated surface, records the failure diagnosis reason string in `ViscousResult.reason`, and returns cleanly without raising an unhandled exception.

## Ground Effect Coupling

In ground effect at height $h$ above a flat boundary, the potential flow solver mirrors the body across $y = 0$.
The boundary layer on the image airfoil mirrors the real airfoil:
$$\delta^*_{image}(x, -y) = \delta^*_{real}(x, y)$$
The mass-defect influence matrix accounts for both the direct panels and the image source panels:
$$D_{GE} = -A_{GE}^{-1} (\Psi_{s, real} + \Psi_{s, image})$$
At large heights ($h/c \ge 500$), the mirror influence decays as $O(1/h^2)$ and approaches the free-air asymptotic limit.

## References

- M. J. Lighthill, 'On displacement thickness', Journal of Fluid Mechanics, Vol. 4, Issue 4, 1958, pp. 383-392.
- M. Drela and M. B. Giles, 'Viscous-inviscid analysis of transonic and low Reynolds number airfoils', AIAA Journal, Vol. 25, No. 10, 1987, pp. 1347-1355.
- M. Drela, 'XFOIL: An Analysis and Design System for Low Reynolds Number Airfoils', Lecture Notes in Engineering, Vol. 54, Springer, 1989, pp. 1-12.
- H. B. Squire and A. D. Young, 'The calculation of the profile drag of aerofoils', Aeronautical Research Council Reports and Memoranda, R&M 1838, 1938.
- S. F. Hoerner, 'Fluid-Dynamic Drag', Published by the Author, 1965, chapter 3 (base drag of blunt trailing edges).
- A. E. P. Veldman, 'Quasi-simultaneous interaction methods for viscous-inviscid flow', Computers & Fluids, Vol. 38, Issue 3, 2009, pp. 496-508.
