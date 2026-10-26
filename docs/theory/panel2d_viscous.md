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
- when the accepted step factor is below 1e-2 (near-singular Jacobian, for example at laminar separation), a Levenberg-Marquardt step with adaptive damping is taken instead (Levenberg 1944; Marquardt 1963): the damping $\mu$ of $(J^TJ + \mu\,\mathrm{diag}(J^TJ))\,\Delta x = -J^TR$ starts at its last value and grows by a factor 10 until the 2-norm of the residual falls, with at most 12 values of $\mu$ (one linear solve each); after a fallback step of the same Newton solve found no decrease, with at most `lm_trials` values (default 2, see the early stops below);
- a singular Newton matrix (its linear solve fails) gives no Newton step: the Levenberg-Marquardt step is tried instead, as after a failed line search;
- a cold start tries three first estimates in turn until one converges: (1) direct marches on the inviscid edge velocity ($H$ held past a separation point), with the coupling term scaled by $\omega = 1/64$; $\omega$ is doubled when the residual is below 0.1, or when the residual has not halved in 4 iterations (at small $\omega$ the problem is nearly the direct problem, which has no solution past a separation point); a negative edge velocity at the first station moves the stagnation point first; (2) three sweeps of quasi-simultaneous marches, in which each station follows the local interaction law $u_e = u_{e,inv} + D\,m$ with the mass defects of the other stations held (Veldman 2009), with $\omega = 1$; the local law removes the singularity of the direct march at separation, which helps flapped sections; (3) the direct marches with $\omega = 1$;
- alpha sweeps start each angle from the converged state of the previous angle; when that fails the angle step is halved up to three times; a cold start is made at the inviscid zero-lift angle and continued to the angle in steps of at most 2 deg.

Converged means: $\omega = 1$, the largest absolute residual of all equations below `tol` (default 1e-6) and no structure change in the last two evaluations. The residuals are relative (logarithmic differences and ratios), except the amplification equation (absolute in $N$). A case that does not converge returns `converged=False`, the reached residual, the work counters and a reason, with a warning. Its loads ($C_l$, $C_{l,circulation}$, $C_{l,pressure}$, $C_d$, $C_{d,friction}$, $C_{d,pressure}$, $C_m$), transition positions and node and wake distributions are NaN: the last iterate is not a solution (the batched solver does the same). A singular linear system never raises an exception: the angle fails with a reason that names it. There are no result overrides.

## Early stops (fail fast)

Without early stops, an angle that cannot converge runs every attempt of the chain above to `max_iter`: the warm start, the halved angle steps, the cold start at the zero-lift angle with its three first estimates, the continuation and the cold start at the angle. That is dozens of full Newton solves, many times the work of a converged angle. The rules below are method choices of Ventorum-2D. They change the work of a failed angle. They do not change a converged solution: a converged angle is the end of a chain of converged Newton solves, and the rules only stop failed solves earlier, skip attempts that did not rescue an angle in the test sample, or reuse a result that is the same for every angle.

Rules of one Newton solve (`vii.newton_solve`):

- stagnation window: the solve stops when its smallest residual has not fallen by the factor `stall_factor` (default 0.5) in the last `stall_window` iterations (default 12). The window starts again when $\omega$ changes. At $\omega = 1$ it also starts again when the structure changes (stagnation panel or transition interval), because the residuals of two structures belong to two different discrete problems. Below $\omega = 1$ a structure change does not restart the window: there the transition intervals are not frozen and can alternate without end;
- stagnation (fixed point): at $\omega = 1$ the solve stops when the line search and the Levenberg-Marquardt fallback both fail twice in a row on the same state and structure. The next iterations would repeat the last one exactly, so this stop does not change the final iterate;
- floor: both stagnation stops apply only while the smallest residual is above 1000 times `tol` (constant `STALL_FLOOR`). Near the tolerance the residual can stall at the round-off level of the arithmetic (float32 on a GPU) for some iterations and still reach the tolerance; the failed solves of the test sample stall far above this level;
- divergence: at $\omega = 1$ the solve stops when the residual is more than `divergence_factor` (default 1e4) times its smallest earlier value at $\omega = 1$; a non-finite residual stops the solve as before;
- Levenberg-Marquardt trials: after a fallback step of a Newton solve found no decrease of the residual while the residual was above the floor of the stagnation stops, the later fallback steps of that solve try at most `lm_trials` values of the damping (default 2; one linear solve each) instead of 12. In the test sample no Newton solve that converged in float64 had a failed fallback step, so the rule changes only failed solves. Near the tolerance a float32 solve can have failed fallback steps at the round-off level and still converge; below the floor a failed step therefore does not set the cap;
- singular: at $\omega = 1$, when the Newton matrix is singular and the Levenberg-Marquardt fallback fails, the solve stops at once with the status "singular" (the next iterations would repeat the last one exactly); below $\omega = 1$ this is a failed step. This stop does not depend on `stall_window`. A singular linear system at another place of the Newton solve (edge velocities, structure update) stops the unfinished cases of the solve with the status "singular", and one in the start values of an angle ends the chain of attempts of that angle; a first estimate of a cold start whose marches meet a singular system is skipped.

Rules of the chain of attempts of one angle (`viscous.py`):

- work budget: the sum of the Newton iterations of all attempts of one angle is at most `work_budget` (default 250); each Newton solve gets the iterations that are left as its limit, and the chain ends when none are left;
- cold start: when the attempt from the quasi-simultaneous marches stops by stagnation, divergence or a non-finite residual with a residual above `COLD_SKIP_RESIDUAL` (1e-2), the third first estimate (direct marches with $\omega = 1$) is not tried. A failure of the first estimate does not end the cold start, because the quasi-simultaneous marches often rescue it;
- halved angle steps: when the solve at an intermediate angle stops by stagnation, that step is not halved again and the halving at that level ends. The second half-step (from the intermediate angle to the angle) is always halved again;
- zero-lift cold start: the cold start at the zero-lift angle is the same for every angle of a sweep; it is solved once per sweep and reused (also when it fails). A cold start that the work budget cut is not reused.

Each early stop is named in the reason of a failed angle ("stagnation", "divergence", "singular", "work budget", "cold start ended", "halving of the angle step ended"). `stall_window=0` switches the stagnation stops and the end of the cold start off, `divergence_factor=0` the divergence stop and `work_budget=0` the budget. With all three off and `lm_trials=12` (the former fixed number of trials), the Newton iterations are those of the solver before the early stops (the reuse of the zero-lift cold start changes only the work); there, a case that met a singular matrix raised an exception.

Work counters: the result gives per angle `newton_iterations` (sum of the Newton iterations of all its Newton solves, failed attempts included), `linear_solves` (solves of the Newton or Levenberg-Marquardt linear system) and `newton_solves` (number of attempts). They measure the cost independently of the machine load.

How the defaults were chosen: a sample of NACA 0012, 2412 and 4415 sections, 120 and 160 panels, plain flaps of 0 to 20 deg at x/c 0.75, Reynolds numbers 1e6 and 3e6 and sweeps of alpha from -2 to 8 deg was solved with and without the rules. The window is the smallest one, plus a margin of two iterations, with which every converged Newton solve of the sample still converges; shorter windows stopped slow but sure solves (long plateaus of the residual during the coupling continuation, or a slow decrease at $\omega = 1$). The budget is above the work of the costliest converged angle of the sample, with a margin. The quasi-simultaneous first estimate rescued many failed direct first estimates; the third first estimate rescued none. The halving after a failed second half-step rescued angles down to the last halving; the halving after a stagnation at an intermediate angle rescued none. The cap of the Levenberg-Marquardt trials was chosen on the same sample, from the deterministic work counters (not the run time): with the cap, every angle that converged with the former 12 trials still converges with the same $C_l$, $C_d$ and $C_m$ (to 1e-10), and the failed angles need fewer linear solves. A cap on every fallback step did not meet this rule: a fallback step of a converging solve can need many damping values before the residual falls.

## Loads

- Reference geometry: the chord line from the leading edge to the trailing-edge midpoint, and the reference chord, as in the inviscid solver ([](panel2d.md)). The section can be an array or an `AirfoilShape`; a flapped shape (`AirfoilShape.with_flap`) holds the reference geometry of the section before the deflection, as in thin-airfoil theory (Glauert 1927). For an array, the reference geometry is the one of the section as given: the leading edge is the point farthest from the trailing-edge midpoint (the definition of `ventorum.airfoil`). A plain flap moves the trailing edge, so a deflected section given as an array is shorter and its chord line is tilted.
- Reference chord $c$ (argument `chord_ref`): the explicit value, else the reference chord of the reference geometry. The polar source and the agent tool pass the flapped shape, so a flapped section uses the chord of the undeflected section. All coefficients, the Reynolds number, the $x/c$ of transition and the wake length use $c$.
- Moment reference point (argument `moment_ref`, a point $(x, y)$ in the body frame): the explicit value, else the quarter-chord point on the chord line of the reference geometry, $x_{LE} + (x_{TE} - x_{LE})/4$. For a section with the leading edge at the origin and the trailing-edge midpoint at $(1, 0)$ it is $(0.25, 0)$, the former default; before, the point was $(x_{min} + c/4,\ 0)$ for every section. The same quarter-chord point is the default pivot in ground effect; an explicit `moment_ref` does not move the pivot. The moment point changes only $C_m$, not the solution.
- $C_l$: the circulation lift in free air, unchanged; the surface pressure lift ($C_{l,pressure}$) in ground effect (Coulliette and Plotkin 1996, eq. 30; see [](panel2d.md)). $C_{l,circulation}$ keeps $2\Gamma/c$ from the total circulation, including the gap-panel vortex, in both cases. $C_{l,pressure}$ and $C_m$ come from the surface pressure on the wall panels. $C_m$ is about the moment reference point, pressure only. The moment transfer between two points is the one of [](panel2d.md) with $C_{l,pressure}$ and $C_{d,pressure}$.
- $C_d$: Squire-Young at the wake end, $C_d = 2\,\theta_{end}\,u_{e,end}^{(H_{end} + 5)/2}/c$ (Squire and Young 1938).
- $C_{d,friction}$: $\int C_f u_e^2\,(\hat t_{flow}\cdot\hat e_D)\,ds/c$ over both sides, from the stagnation point.
- $C_{d,pressure}$: surface pressure integral over the wall panels (exact for $\gamma$ linear on each panel).

$C_{d,friction} + C_{d,pressure}$ is an independent check of $C_d$; nothing replaces one by the other. At small lift they agree; as the lift grows the surface sum falls below the Squire-Young drag, also on a finer panel mesh. The difference grows with the lift; it is the difference between the thin-layer surface pressure and the momentum deficit of the wake model, and no correction is applied.

## Ground effect

The image method of the inviscid solver; the image body carries the mirror image of the real boundary layer and wake (equal source strengths) and there is no boundary layer on the ground. There is no height limit: a large height gives the free-air answer by computation ($h/c$ = 50 is within 1e-4 of free air on $C_{l,circulation}$ for NACA 0012 at 4 deg).

## Multi-element and flaps

One body only: more bodies raise `ValueError`. A plain flap is part of the single-element geometry.

## Plain flap continuation in deflection

`solve_viscous_flapped` (`ventorum/panel2d/viscous.py`) solves a section with a deflected plain flap by continuation in the deflection angle. A direct cold solve often fails at large deflections because of strong suction peaks and local boundary-layer separation at the hinge corner. The continuation starts at 0 deg deflection and advances to the target deflection in steps of at most `step_deg` (default 2.5 deg).

Each step warm-starts from the converged boundary-layer state of the previous deflection. When a step does not converge, the step size is halved (down to 0.25 deg). If a step fails at 0.25 deg, the solver stops and returns `converged=False` with the reached deflection in its details, and all loads set to NaN.

Limits: continuation converges for plain flaps up to at least 15 deg on NACA 2412 (and reaches 17.5 deg without fairing). At larger deflections without a hinge fairing, the sharp corner causes strong adverse pressure gradients and numerical separation that stop convergence.

## Batched solver

`solve_viscous_batch` (`ventorum/panel2d/viscous_batch.py`) runs the same algorithm with torch on the CPU or on CUDA. Cases with the same geometry, Reynolds number, height and transition settings form sequences; pass $p$ solves the $p$-th angle of every sequence; the cases of a pass with the same node and wake counts form one batched Newton iteration (residuals, Jacobians, $D$, edge velocities and `torch.linalg.solve` on the device; converged cases leave the active set). Failed first attempts and the first angle of each sequence follow the sequential fallback path, one case at a time. In float64 the batch equals the sequential solver; in float32 the residual stops near the float32 round-off level, and a case that does not reach the tolerance is reported with its residual.

## Comparison with XFOIL 6.99 (black box)

At the lower Reynolds numbers the drag is above that of XFOIL 6.99 at small angles. Causes found: (1) with the edge velocity of XFOIL as input, the laminar march with the published closures (which match exact Falkner-Skan profiles) gives a higher $H$ in the adverse gradient, so the $e^N$ transition comes earlier; (2) the shear-stress start value $0.7\,C_{\tau,EQ}$ (to be checked against the paper) makes the turbulent layer grow faster just after transition. With the transition forced at the same position the drag agrees. The tests print the comparison table; the numbers are not stored in the repository.

## Verification

`tests/test_panel2d_viscous.py`, `tests/test_panel2d_viscous_wake.py`, `tests/test_panel2d_viscous_batch.py`, `tests/test_panel2d_flap_continuation.py`.

## References

- M. J. Lighthill, "On displacement thickness", Journal of Fluid Mechanics 4, 1958, pp. 383-392.
- M. Drela and M. B. Giles, "Viscous-inviscid analysis of transonic and low Reynolds number airfoils", AIAA Journal 25(10), 1987, pp. 1347-1355.
- M. Drela, "XFOIL: An analysis and design system for low Reynolds number airfoils", Lecture Notes in Engineering 54, Springer, 1989, pp. 1-12.
- H. B. Squire and A. D. Young, "The calculation of the profile drag of aerofoils", ARC R and M 1838, 1938.
- K. Levenberg, "A method for the solution of certain non-linear problems in least squares", Quarterly of Applied Mathematics 2, 1944, pp. 164-168.
- D. W. Marquardt, "An algorithm for least-squares estimation of nonlinear parameters", SIAM Journal on Applied Mathematics 11, 1963, pp. 431-441.
- H. Glauert, "Theoretical relationships for an aerofoil with hinged flap", ARC R and M 1095, 1927.
- A. E. P. Veldman, "Quasi-simultaneous interaction methods for viscous-inviscid flow", Computers and Fluids 38(3), 2009, pp. 496-508.
