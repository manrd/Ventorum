---
orphan: true
---

# 2D viscous-inviscid solver (Ventorum-2D)

## Model

`solve_viscous` (`ventorum/panel2d/viscous.py`) couples the panel method of [](panel2d.md) with the integral boundary layer of [](ibl2d_march.md) on the upper surface, the lower surface and the wake. The boundary layer acts on the outer flow by wall and wake transpiration (Lighthill 1958): a source sheet of strength $\sigma = dm/d\xi$, $m = u_e\delta^*$ the mass defect. The whole system is solved by one global Newton iteration with the edge velocity eliminated (fully simultaneous coupling; Drela and Giles 1987; Drela 1989, the papers only). The engine is `ventorum/panel2d/vii.py`; the inviscid data are built in `ventorum/panel2d/coupling.py`.

## Inviscid data of a case

For each angle (and height) the inviscid matrix is factorized once. The solver forms:

- the inviscid nodal $\gamma_{inv}$;
- the wake line: it starts at the trailing edge (the midpoint of a blunt edge) along the bisector of the two trailing-edge panels, then follows the inviscid streamline by Heun predictor-corrector steps along the unit velocity, with panel lengths that grow geometrically from the mean length of the two trailing-edge panels to the wake length (default one chord, 30 panels); in ground effect the velocity includes the mirror images, and a wake line that reaches the ground stops the case with a reason;
- the source sensitivities: $\partial\gamma/\partial\sigma$ at the nodes for every airfoil and wake source panel (one solve with many right-hand sides, the factorization reused), and the tangential velocity at the wake panel midpoints per unit source strength (vortex and source velocity influences, with images in ground effect).

## Stations and edge velocity

The stagnation point lies on the panel where the total $\gamma$ changes sign, by linear interpolation. The upper stations run from it over the upper nodes to the trailing edge, the lower stations likewise, and the wake stations from the trailing edge downstream. A node closer to the stagnation point than 0.1 of its next interval is not a station (its mass defect is zero; constant `STAGNATION_DROP`): next to the stagnation point $u_e$ is nearly zero and the logarithmic differences of the next interval would be ill-conditioned.

Station edge velocities: $u_e = \pm\gamma$ at the airfoil nodes (sign of the side), $\gamma$ of the trailing-edge node at the first wake station (both trailing-edge speeds are equal by the Kutta condition), the mean of the two adjacent wake-panel midpoint velocities at the other wake stations and a linear extrapolation at the last one. Source strengths: on an airfoil panel $\sigma L$ = ($m$ where the layer leaves the panel) - ($m$ where it enters), on the stagnation panel both ends are exits, on a wake panel $\sigma L = m_{down} - m_{up}$. The gap panel of a blunt edge carries no viscous source: the gap thickness is the displacement of its inviscid gap source. Together:

```{math}
u_e = u_{e,inv} + D\,m, \qquad D = P\,S\,B,
```

with $P$ the station-velocity map, $S$ the source sensitivities and $B$ the source matrix of the current structure.

## Variables and equations

Per station: $\theta$, the mass defect $m$ and $c$ ($N$ laminar, $\sqrt{C_\tau}$ turbulent and wake). At the first station of each side the second variable is $\delta^*$ instead of $m$ (there $u_e$ is small and $\delta^* = m/u_e$ would be ill-conditioned); $u_e$ then solves $(I - D\,\mathrm{diag}(\delta^*_S))\,u_e = u_{e,inv} + D\,m_N$. Equations per station:

- first station of each side: the stagnation-point similarity equations of [](ibl2d_march.md) with the velocity gradient $a = (u_{e,upper} + u_{e,lower})/\Delta s$ of the stagnation panel (when a node is dropped, $a = u_e/\xi$ of the first station), and $N = 0$;
- every other airfoil station: the three equations of the interval that ends at it (laminar, transition, post-transition or turbulent);
- first wake station: $\theta_w = \theta_u + \theta_l$, $\delta^*_w = \delta^*_u + \delta^*_l + h_{TE}$ ($h_{TE}$ the trailing-edge gap normal to the bisector; Drela 1989), and $C_{\tau,w}$ the $\theta$-weighted mean of the two sides (a side that is still laminar at the trailing edge enters with $0.7\,C_{\tau,EQ}$); the weighted mean is the author's choice and is to be checked against the source;
- other wake stations: the wake interval equations (no skin friction, doubled dissipation).

The Jacobian is exact: forward-mode derivatives of the station equations with respect to $(\theta, \delta^*, c, u_e)$, then the chain rule through $\delta^* = m/u_e$ and $u_e(m)$. The linear system is dense ($3 \times$ the number of stations) and is solved directly.

## Structure updates

Between Newton iterations: the stagnation point is found again from the total $\gamma$ (it follows $\alpha$ and the viscous solution); when it moves to another panel the stations are reassigned ($\theta$, $\delta^*$ and $c$ carried over node by node) and the node-drop decision is made again. On the same panel a node can be dropped (the stagnation point came too close to it) but not restored, so that the decision cannot oscillate. The transition interval of each side is the first interval where $N$ reaches $n_{crit}$ (or the forced position); when the point leaves its interval downstream the new interval is estimated with the growth rate of the last laminar station. With the continuous transition equations of [](ibl2d_march.md) the solution changes continuously when the transition point passes a station. Near convergence (residual below 1e-3), a transition interval that alternates between two positions is held at the downstream one; the point then lies at the station between them.

## Newton step control

These rules change the path, not the converged solution:

- step limit: $\theta$, $m$ (or $\delta^*$) and $\sqrt{C_\tau}$ fall by at most 50 % and rise by at most 200 %, $H - 1$ falls by at most 50 %, $u_e$ stays positive;
- backtracking line search on the 2-norm of the residual (Armijo coefficient 1e-4);
- when the accepted step factor is below 1e-2 (near-singular Jacobian, for example at laminar separation), a Levenberg-Marquardt step with adaptive damping is taken instead (Levenberg 1944; Marquardt 1963);
- a cold start tries three first estimates in turn until one converges: (1) direct marches on the inviscid edge velocity ($H$ held past a separation point), with the coupling term scaled by $\omega = 1/64$; $\omega$ is doubled when the residual is below 0.1, or when the residual has not halved in 4 iterations (at small $\omega$ the problem is nearly the direct problem, which has no solution past a separation point); a negative edge velocity at the first station moves the stagnation point first; (2) three sweeps of quasi-simultaneous marches, in which each station follows the local interaction law $u_e = u_{e,inv} + D\,m$ with the mass defects of the other stations held (Veldman 2009), with $\omega = 1$; the local law removes the singularity of the direct march at separation, which helps flapped sections; (3) the direct marches with $\omega = 1$;
- alpha sweeps start each angle from the converged state of the previous angle; when that fails the angle step is halved up to three times; a cold start is made at the inviscid zero-lift angle and continued to the angle in steps of at most 2 deg.

Converged means: $\omega = 1$, the largest absolute residual of all equations below `tol` (default 1e-6) and no structure change in the last two evaluations. The residuals are relative (logarithmic differences and ratios), except the amplification equation (absolute in $N$). A case that does not converge returns `converged=False`, the reached residual and a reason, with a warning. There are no result overrides.

## Loads

- $C_l$: Kutta-Joukowski of the total circulation (including the gap-panel vortex); $C_{l,pressure}$ and $C_m$ from the surface pressure (wall panels), $C_m$ about the quarter chord, pressure only.
- $C_d$: Squire-Young at the wake end, $C_d = 2\,\theta_{end}\,u_{e,end}^{(H_{end} + 5)/2}/c$ (Squire and Young 1938).
- $C_{d,friction}$: $\int C_f u_e^2\,(\hat t_{flow}\cdot\hat e_D)\,ds/c$ over both sides, from the stagnation point.
- $C_{d,pressure}$: surface pressure integral over the wall panels (exact for $\gamma$ linear on each panel).

$C_{d,friction} + C_{d,pressure}$ is an independent check of $C_d$; nothing replaces one by the other. At small lift they agree; as the lift grows the surface sum falls below the Squire-Young drag, also on a finer panel mesh. The difference grows with the lift; it is the difference between the thin-layer surface pressure and the momentum deficit of the wake model, and no correction is applied.

## Ground effect

The image method of the inviscid solver; the image body carries the mirror image of the real boundary layer and wake (equal source strengths) and there is no boundary layer on the ground. There is no height limit: a large height gives the free-air answer by computation ($h/c$ = 50 is within 1e-4 of free air on $C_l$ for NACA 0012 at 4 deg).

## Multi-element and flaps

One body only: more bodies raise `ValueError`. A plain flap is part of the single-element geometry.

## Batched solver

`solve_viscous_batch` (`ventorum/panel2d/viscous_batch.py`) runs the same algorithm with torch on the CPU or on CUDA. Cases with the same geometry, Reynolds number, height and transition settings form sequences; pass $p$ solves the $p$-th angle of every sequence; the cases of a pass with the same node and wake counts form one batched Newton iteration (residuals, Jacobians, $D$, edge velocities and `torch.linalg.solve` on the device; converged cases leave the active set). Failed first attempts and the first angle of each sequence follow the sequential fallback path, one case at a time. In float64 the batch equals the sequential solver; in float32 the residual stops near the float32 round-off level, and a case that does not reach the tolerance is reported with its residual.

## Comparison with XFOIL 6.99 (black box)

For NACA 0012 at Re 1e6 (160 panels, $n_{crit} = 9$) the drag is above XFOIL 6.99 at small angles and the lift is a few percent lower. Causes found: (1) with the edge velocity of XFOIL as input, the laminar march with the published closures (which match exact Falkner-Skan profiles) gives $H$ 1 to 4 % higher in the adverse gradient, so the $e^N$ transition comes about 0.05 chord earlier; (2) the shear-stress start value $0.7\,C_{\tau,EQ}$ (to be checked against the paper) makes the turbulent layer grow faster just after transition. With transition forced at 5 % chord the drag agrees within 2 %. The tests print the comparison table; the numbers are not stored in the repository.

## Verification

`tests/test_panel2d_viscous.py`, `tests/test_panel2d_viscous_wake.py`, `tests/test_panel2d_viscous_batch.py`.

## References

- M. J. Lighthill, "On displacement thickness", Journal of Fluid Mechanics 4, 1958, pp. 383-392.
- M. Drela and M. B. Giles, "Viscous-inviscid analysis of transonic and low Reynolds number airfoils", AIAA Journal 25(10), 1987, pp. 1347-1355.
- M. Drela, "XFOIL: An analysis and design system for low Reynolds number airfoils", Lecture Notes in Engineering 54, Springer, 1989, pp. 1-12.
- H. B. Squire and A. D. Young, "The calculation of the profile drag of aerofoils", ARC R and M 1838, 1938.
- K. Levenberg, "A method for the solution of certain non-linear problems in least squares", Quarterly of Applied Mathematics 2, 1944, pp. 164-168.
- D. W. Marquardt, "An algorithm for least-squares estimation of nonlinear parameters", SIAM Journal on Applied Mathematics 11, 1963, pp. 431-441.
- A. E. P. Veldman, "Quasi-simultaneous interaction methods for viscous-inviscid flow", Computers and Fluids 38(3), 2009, pp. 496-508.
