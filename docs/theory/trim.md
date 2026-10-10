# Trim solver

This chapter presents the mathematical formulation and numerical method of the aircraft trim solver in Ventorum.

## Purpose

The trim solver finds the angle of attack and control surface deflections that establish aerodynamic equilibrium. Longitudinal trim balances lift against a target value while driving pitching moment to zero. Lateral trim balances lift and drives rolling, pitching, and yawing moments to zero at a specified sideslip angle.

## Unknowns and equations

The equilibrium equations depend on the requested trim mode. Moments are evaluated in body axes about the moment reference point of the aircraft.

### Longitudinal trim

Longitudinal trim solves two nonlinear equations for two unknowns:

$$x = \begin{bmatrix} \alpha \\ \delta_{\text{pitch}} \end{bmatrix}$$

$$R(x) = \begin{bmatrix} C_L(x) - C_{L,\text{target}} \\ C_m(x) \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

Here $\alpha$ is the angle of attack and $\delta_{\text{pitch}}$ is the pitch control deflection.

### Lateral trim

When both roll and yaw control surfaces are specified, lateral trim solves four equations for four unknowns at a fixed sideslip angle $\beta$:

$$x = \begin{bmatrix} \alpha \\ \delta_{\text{pitch}} \\ \delta_{\text{roll}} \\ \delta_{\text{yaw}} \end{bmatrix}$$

$$R(x) = \begin{bmatrix} C_L(x) - C_{L,\text{target}} \\ C_m(x) \\ C_l(x) \\ C_n(x) \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}$$

Here $\delta_{\text{roll}}$ is the roll control deflection and $\delta_{\text{yaw}}$ is the yaw control deflection. Sideslip angle $\beta$ remains constant at the value given in the flight condition.

## Numerical method

The nonlinear system $R(x) = 0$ is solved with Newton's method:

$$J(x_k) \Delta x_k = -R(x_k)$$

### Finite-difference Jacobian

The Jacobian matrix $J = \partial R / \partial x$ is formed by central differences for each unknown:

$$\frac{\partial R}{\partial x_j} \approx \frac{R(x + h e_j) - R(x - h e_j)}{2 h}$$

The step size is $h = 1.0 \times 10^{-4}$ rad for every variable. Each iteration reuses the central solution and evaluates $2 n$ perturbed solutions, where $n$ is the number of unknowns. When a variable lies within $h$ of a boundary, the stencil shifts inward to keep all evaluations within valid limits.

### Step limit and bounds

To maintain robust convergence, the Newton step is constrained:
- If the maximum absolute component $\max_j |\Delta x_j|$ exceeds $5^\circ$ ($0.087266$ rad), the entire step vector $\Delta x$ is scaled down so that the maximum component equals $5^\circ$.
- Each unknown is constrained by physical bounds. The angle of attack is restricted to `alpha_bounds_deg` (default $-10^\circ$ to $+20^\circ$). Control surface deflections are restricted to $[-30^\circ, 30^\circ]$.
- When a candidate point violates a bound, the variable is clamped at that bound.

### Stop rules

The iteration stops when one of the following criteria is met:
1. **Convergence**: $|C_L - C_{L,\text{target}}| \le \text{tol\_CL}$ (default $10^{-6}$) and every moment residual is less than or equal to $\text{tol\_moment}$ (default $10^{-7}$). The status is `"trimmed"`.
2. **Bound limit**: An unknown sits at a boundary and the computed Newton step points outward in two consecutive iterations. The solver terminates with status `"alpha_limit"` or `"control_limit"`.
3. **Iteration limit**: The iteration count reaches `max_iterations` without satisfying the convergence tolerances. The status is `"not_converged"`.
4. **Numerical failure**: The Jacobian is singular or an evaluation produces non-finite values. The solver terminates with status `"not_converged"`. The Jacobian is singular when the column of one unknown has a norm of at most $10^{-8}$ times the norm of the Jacobian (the unknown has no effect on the residuals), or when the condition number is above $10^{10}$.
5. **Solve failure**: A solve fails with a ground strike, the lifting line is too near the ground, or an inner nonlinear solve does not converge. The status is `"not_converged"`. The result gives the last point with a good solve and a note on the failed point. When no solve succeeded, the coefficients are NaN and `result` is None. Invalid input raises `ValueError` before the iteration starts. All other exceptions propagate.

## Ground effect convention

When `condition.h` is defined, Ventorum models flight in ground effect over a flat boundary plane. In accordance with the ground-effect solver conventions of Ventorum, the vertical distance from `Aircraft.ref_point` to the ground plane remains constant while the angle of attack changes.

## Precision and device execution

All trim calculations execute on the CPU using 64-bit floating-point arithmetic (`float64`). For the central difference with step $h$, the truncation error is $O(h^2)$ and the round-off error is about $\varepsilon / h$, with $\varepsilon$ the machine epsilon of the arithmetic (about $1.2 \times 10^{-7}$ in `float32`, $2.2 \times 10^{-16}$ in `float64`). With $h = 1.0 \times 10^{-4}$ rad, $\varepsilon / h$ is about $10^{-3}$ in `float32` and about $2 \times 10^{-12}$ in `float64`.

The solver sets the global execution device to `"cpu"` for the duration of the trim solve and restores the previous device setting in a `finally` block upon completion.

## Assumptions and limits

The trim solver operates under the following limits:
- Static aerodynamic trim only. Dynamic derivatives and inertial forces are not included.
- No propulsion model. Thrust forces and propulsor gyroscopic moments are not modeled.
- No aircraft weight model. The target lift coefficient is specified directly by the user.
- Rigid aircraft geometry. Aeroelastic deformations are not considered.

## Verification

The trim solver is verified by the unit test suite in `tests/test_trim.py`:
- Longitudinal trim convergence verified against independent aerodynamic analysis.
- Parity with linear predictions based on numerical stability derivatives.
- Elevator trim slope consistency with longitudinal static stability ($C_{m\alpha} < 0$).
- Lateral trim convergence to zero moments ($C_l = C_m = C_n = 0$) under non-zero sideslip.
- Boundary limit detection when target lift exceeds achievable performance.
- Ground effect lift increase demonstrating reduced trim angle of attack.
- Multi-solver compatibility across vortex lattice and lifting-line solvers.
- Input geometry immutability ensuring that input objects are never modified in place.
- CPU device and precision restoration.
- Failure paths: a ground strike during the iteration and at the start point, a control with no effect (singular Jacobian), an inner solve that does not converge, invalid settings, and exceptions that must propagate.

## References

- B. Etkin and L. D. Reid, *Dynamics of Flight: Stability and Control*, 3rd ed., Wiley, 1996.
- J. E. Dennis and R. B. Schnabel, *Numerical Methods for Unconstrained Optimization and Nonlinear Equations*, SIAM, 1996.
