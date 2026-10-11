# Adjoint gradients

This chapter gives the discrete adjoint method of `ventorum.gradients`: exact gradients
of the solver outputs with respect to the lattice nodes, the flow variables, the linear
section properties and the reference values, for the vortex-lattice method and the
linear numerical lifting line.

## Model in one paragraph

The residual of the linear solve is `R(G, x) = A(x) G - b(x)`, with the circulation `G`
and the inputs `x`. The circulation comes from the normal production solve. One LU
factor of `A` gives `G` and every adjoint vector `lambda_j` from `A^T lambda_j =
df_j/dG`. The total derivative of the output `f_j` is the gradient of the Lagrangian
`f_j(G*, x) - lambda_j . (A(x) G* - b(x))` at the fixed converged `G*`. No automatic
differentiation runs through the linear solve.

## Equations

With the partial derivatives at the fixed `G*`:

```text
df_j/dx = partial f_j/partial x - lambda_j^T (partial A/partial x G* - partial b/partial x)
A^T lambda_j = df_j/dG
```

The partial derivatives come from a differentiable reference copy, in PyTorch, of the
lattice built from the edge nodes, the residual (system rows) and the outputs. Every
part with O(n^2) work (system rows, near-field load points, Trefftz points) runs in row
chunks; the chunk size comes from a memory budget (`VENTORUM_GRAD_CHUNK`, default the
budget of `vortex_torch`).

## Assumptions

* The mesh topology is fixed: spanwise stations, chordwise count and chordwise
  fractions do not change with the inputs.
* The flow is linear in the circulation: vortex lattice and linear lifting line with
  linear airfoils, zero control deflections and zero body rates.
* The folded symmetric system applies only when the case is symmetric and every
  requested input and output is symmetric (no beta, phi, node, section or reference
  change, no lateral output). Else the full system is used.

## Envelope

Outside phase 1 the functions raise `ValueError`: the nonlinear lifting line, tabulated
airfoils, nonzero control deflections, nonzero body rates and the Fourier solver
(see T-0071). In ground effect `h` is the height of the moment reference point, and the
ground plane and its images follow the inputs; a ground strike raises as in `analyze`.

The forward check runs at every call: the reference copy also gives the output values,
and above 1e-10 relative difference in float64 (1e-4 in float32) the call raises
`RuntimeError` with the names of the outputs.

## Implementation

* `ventorum/gradients/reference.py`: the differentiable copy (lattice from edge nodes,
  system rows, Kutta-Joukowski and leg forces with the tangential velocity, Trefftz
  drag, profile drag and section moment of linear airfoils, ground images, cross-surface
  core, axes transforms).
* `ventorum/gradients/adjoint.py`: `gradients`, `gradient_vjp`, `GradientResult`.
* `ventorum/gradients/check.py`: `check_gradients` with the Richardson reference
  (central differences with steps h and h/2, error estimate |FD(h) - FD(h/2)| / 3).

Precision: float32 on the selected device by default, also for the gradient partials and
the adjoint solve; float64 stays available through the existing precision setting. The
shipped rule uses sequential source accumulation with the coordinates in body axes.
Pairwise accumulation of the source terms, coordinates relative to a local origin of
the surface and Kahan summation of the reductions were compared on the test wing and
do not improve the float32 node gradients there (see the task report); they stay
selectable through `VENTORUM_GRAD_ACCUM` and `VENTORUM_GRAD_ORIGIN` for study. Float32
results carry a note that names the 1e-3 gate, and `check_gradients` reports pass or
fail per component.

## Verification

* `tests/test_gradients.py`: forward parity to 1e-12 (float64), adjoint against
  Richardson references (1e-6 float64, 1e-3 float32), translation and rotation
  identities, stability-derivative agreement, the symmetric fold rule, the strip-force
  VJP, refusals and the GPU device.
* The values come from `validation/run_verification.py` (unchanged by this feature).

## References

* J. R. R. A. Martins and J. T. Hwang, "Review and unification of methods for computing
  derivatives of multidisciplinary computational models", AIAA Journal 51(11), 2013,
  pp. 2582-2599.
* J. R. R. A. Martins and A. Ning, "Engineering Design Optimization", Cambridge
  University Press, 2021, ch. 6.
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge University
  Press, 2001.
* W. F. Phillips and D. O. Snyder, "Modern adaptation of Prandtl's classic
  lifting-line theory", Journal of Aircraft 37(4), 2000, pp. 662-670.
* M. Drela, "Flight Vehicle Aerodynamics", MIT Press, 2014.
