# Numerical lifting line

## Model

The numerical lifting line (`solver_type="linear"` and `"nonlinear"`) is the method of W. F. Phillips and D. O. Snyder (2000). Each surface is cut into spanwise strips, and each strip carries one bent horseshoe vortex with its bound vortex on the quarter-chord line (see [Vortex filaments](vortex_kernels.md)). At one point on each bound vortex, the three-dimensional Kutta-Joukowski force is set equal to the lift of the section at its local angle of attack. The local velocity includes the velocity that all horseshoes induce. The method applies the Lanchester–Prandtl lifting-line theory to arbitrary planforms, dihedral, twist and several surfaces. The linear solver uses a linear section lift curve. The nonlinear solver takes the section lift from the section polar and solves the equations with Newton's method.

## Equations

Strip $i$ has the bound-vortex vector $\mathbf{dl}_i$ [m], the area $dA_i$ [m^2], the unit normal $\hat{\mathbf n}_i$ and the unit chord direction $\hat{\mathbf a}_i$. Let $\mathbf v_{ij}$ [1/m] be the velocity at the control point of strip $i$ from the horseshoe of strip $j$ with unit circulation. The local velocity [m/s] is

```{math}
\mathbf W_i = V_\infty \hat{\mathbf u} + \sum_j \mathbf v_{ij}\, \Gamma_j ,
```

and the local angle of attack is $\alpha_i = \operatorname{atan2}(\mathbf W_i\cdot\hat{\mathbf n}_i,\ \mathbf W_i\cdot\hat{\mathbf a}_i)$. The Kutta-Joukowski force on the bound vortex, $\rho\,\Gamma_i\,|\mathbf W_i \times \mathbf{dl}_i|$, must equal the section lift $\tfrac12 \rho V_\infty^2\, dA_i\, C_l(\alpha_i)$. The residual of strip $i$ is (Phillips and Snyder 2000, in dimensional form)

```{math}
:label: eq-llt-nonlinear
R_i = 2\,|\mathbf W_i \times \mathbf{dl}_i|\,\Gamma_i - V_\infty^2\, dA_i\, C_l(\alpha_i) = 0 .
```

### Linear system

For a linear section, $C_l = a_0(\alpha - \alpha_{L0})$. Phillips and Snyder linearise equation {eq}`eq-llt-nonlinear` in two steps: the magnitude of the force uses the free stream only, $|\mathbf W_i \times \mathbf{dl}_i| \approx V_\infty |\hat{\mathbf u} \times \mathbf{dl}_i|$, and the angle uses small angles, $\alpha_i \approx \mathbf W_i \cdot \hat{\mathbf n}_i / V_\infty$. Divide by $V_\infty a_{0,i} dA_i$:

```{math}
:label: eq-llt-linear
\frac{2\,|\hat{\mathbf u} \times \mathbf{dl}_i|}{a_{0,i}\, dA_i}\,\Gamma_i - \sum_j (\mathbf v_{ij}\cdot\hat{\mathbf n}_i)\,\Gamma_j
= V_\infty \left( \hat{\mathbf u}\cdot\hat{\mathbf n}_i - \alpha_{L0,i} \right).
```

The linear solver solves equation {eq}`eq-llt-linear` with one dense LU factorisation.

### Newton's method

The nonlinear solver starts from the linear solution (or from a given start value, for example the previous angle of a sweep) and solves equation {eq}`eq-llt-nonlinear` with Newton's method. The Jacobian is exact:

```{math}
\frac{\partial R_i}{\partial \Gamma_j} = 2\,\delta_{ij}\,|\mathbf W_i \times \mathbf{dl}_i|
+ 2\,\Gamma_i\, \mathbf v_{ij}\cdot(\mathbf{dl}_i \times \hat{\mathbf w}_i)
- V_\infty^2\, dA_i\, \frac{dC_l}{d\alpha}\bigg|_{\alpha_i}
\frac{(\mathbf W_i\cdot\hat{\mathbf a}_i)(\mathbf v_{ij}\cdot\hat{\mathbf n}_i) - (\mathbf W_i\cdot\hat{\mathbf n}_i)(\mathbf v_{ij}\cdot\hat{\mathbf a}_i)}{(\mathbf W_i\cdot\hat{\mathbf a}_i)^2 + (\mathbf W_i\cdot\hat{\mathbf n}_i)^2},
```

with $\hat{\mathbf w}_i = (\mathbf W_i \times \mathbf{dl}_i)/|\mathbf W_i \times \mathbf{dl}_i|$. The second term comes from $\partial|\mathbf W_i \times \mathbf{dl}_i|/\partial\Gamma_j = \hat{\mathbf w}_i\cdot(\mathbf v_{ij}\times\mathbf{dl}_i)$ and the scalar triple product; the third term is the derivative of the atan2 function.

The step is globalised as follows:

1. A backtracking line search on the merit function $f = \tfrac12 \sum_i (R_i / (V_\infty^2 dA_i))^2$ halves the step until $f$ falls by a sufficient amount (a sufficient-decrease test of Armijo type; J. Nocedal and S. J. Wright 2006, chapter 3).
2. If no Newton step is accepted (past the maximum lift the section slope is negative and the Newton direction can fail), a damped fixed-point step towards $\Gamma_i = V_\infty^2 dA_i C_l(\alpha_i) / (2|\mathbf W_i \times \mathbf{dl}_i|)$ is tried, with the largest step fraction (from one half down) that lowers $f$.
3. If the solve does not converge, the solver starts again from scaled linear solutions and keeps the result with the smallest residual.

The solve converges when $\max_i |R_i| / (V_\infty^2 dA_i)$, the largest error in the section lift coefficient, is below `SolverSettings.tolerance`. A solve that does not converge gives a warning, `converged=False`, and a trust penalty.

### Control point on the bound vortex

The control point is on the bound vortex (the quarter chord), at the mid parameter of the spanwise spacing (see [Vortex-lattice method](vortex_lattice.md)). It is not at the three-quarter chord. The section polar already contains the two-dimensional effect of the bound vortex of its own section. A three-quarter-chord point would add that effect a second time. On its own bound vortex, the self-induced velocity of the straight bound segment is zero, so only the trailing legs and the other strips contribute.

## Section polars

A linear airfoil has the constants $a_0$, $\alpha_{L0}$, $C_{d0}$ and $C_{m0}$. A tabulated airfoil gives $C_l$, $C_d$ and (optionally) $C_m$ as functions of the angle of attack:

- The tables are interpolated with a shape-preserving piecewise cubic Hermite interpolant (`scipy.interpolate.PchipInterpolator`; F. N. Fritsch and R. E. Carlson 1980; F. N. Fritsch and J. Butland 1984), so that $dC_l/d\alpha$ is continuous for Newton's method.
- Outside the table the end values are used ($dC_l/d\alpha = 0$). The trust score marks such strips (see [Trust score](trust_score.md)).
- For the linear solvers (the linear lifting line, the vortex lattice and the classical lifting line), $a_0$ and $\alpha_{L0}$ come from a least-squares line through the table points near the zero-lift angle (`TabulatedAirfoil._compute_linear_fit`).
- A polar can come from a file (`ventorum.aero.polars.load_xfoil_polar`, `load_csv_polar`) or from a run of XFOIL as a separate program (`ventorum.aero.xfoil_runner.run_xfoil`; M. Drela 1989). Ventorum takes the polar as given; its accuracy is that of its source.
- Between two sections with different airfoils, linear data blend linearly along the span; two tabulated polars blend linearly on all the table angles of both polars inside the angle range that both cover, with a warning when the ranges differ (`ventorum.aero.polars.blend_tabulated`).

## Assumptions

- One horseshoe per strip: no chordwise extent. Forces act on the bound vortex only.
- The section behaves as in two-dimensional flow at the local angle of attack and the local velocity (strip theory).
- Rigid, straight wake (see [Vortex filaments](vortex_kernels.md)); inviscid, incompressible flow.

## Envelope

- **Sweep.** With sweep, or with a kink of the quarter-chord line, the numerical lifting line is not grid convergent: the lift falls as panels are added ([V4](../verification_report.md), [V5](../verification_report.md)). A tapered wing with a straight leading edge has a swept quarter-chord line. A note is given above the mean sweep `ventorum.solvers.lattice_base.LLT_SWEEP_WARNING_DEG`, and the trust score lowers the rating. Use the vortex lattice for swept wings.
- **Ground effect.** One horseshoe per strip cannot follow the chordwise variation of the image flow. The lifting line is refused below `LLT_GE_MIN_H_OVER_C` (the smallest clearance divided by the reference chord) and gives a note below `LLT_GE_WARN_H_OVER_C` ([V7](../verification_report.md)).
- **Stall.** Past the maximum lift the solution of equation {eq}`eq-llt-nonlinear` need not be unique. The nonlinear solver reports `converged=False` when Newton's method does not reach the tolerance. Results past the stall of a polar get a trust penalty.
- **Low aspect ratio.** Chordwise effects are not modelled; the trust score lowers the rating at low aspect ratio.

## Implementation

- `ventorum.solvers.linear.LinearLLTSolver` and `ventorum.solvers.nonlinear.NonlinearSolver` (both on `ventorum.solvers.lattice_base.LatticeSolver`).
- `ventorum.solvers.core`: `assemble_llt_linear`, `solve_llt_linear`, `solve_llt_linear_batch` (all angles of a sweep in one kernel call and one call of the dense solver), `solve_llt_nonlinear`.
- The lattice is built with `collocation="llt"` (`ventorum.geometry.lattice.build_lattice`): one chordwise panel per strip and the control point on the bound vortex.
- The GPU pipelines (`ventorum.gpu`) solve the same equations in float32 or float64.

## Verification

- [V1](../verification_report.md): elliptic wing, both solvers.
- [V2](../verification_report.md): rectangular and tapered wings against Glauert's solution.
- [V4](../verification_report.md), [V5](../verification_report.md): the sweep limit.
- [V7](../verification_report.md): lift increment in ground effect against the vortex lattice.

## References

- W. F. Phillips and D. O. Snyder, "Modern adaptation of Prandtl's classic lifting-line theory", *Journal of Aircraft* 37(4), 2000, pp. 662-670.
- W. F. Phillips, *Mechanics of Flight*, Wiley, 2004, section 1.9.
- F. N. Fritsch and R. E. Carlson, "Monotone piecewise cubic interpolation", *SIAM Journal on Numerical Analysis* 17(2), 1980, pp. 238-246.
- F. N. Fritsch and J. Butland, "A method for constructing local monotone piecewise cubic interpolants", *SIAM Journal on Scientific and Statistical Computing* 5(2), 1984, pp. 300-304.
- M. Drela, "XFOIL: an analysis and design system for low Reynolds number airfoils", in T. J. Mueller (ed.), *Low Reynolds Number Aerodynamics*, Lecture Notes in Engineering 54, Springer, 1989, pp. 1-12.
- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006, chapter 3.
