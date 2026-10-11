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
budget of `vortex_torch`, in values). The budget counts all values that one chunk holds
in its forward and in its backward pass. The forward pass keeps only the inputs of each
chunk; the backward pass evaluates each chunk again, takes its vector-Jacobian product
and frees it before the next chunk. So the memory of the gradient stays bounded by one
chunk, also for one backward pass per output.

### Target points on a vortex segment

Some target points lie on a vortex segment by the construction of the mesh: the load
point of a panel on its own bound segment, the collocation point of a lifting-line strip
on its own bound segment, and the leg mid-point of a panel on the legs of all panels that
use the same edge line. These points stay on their segment for every node motion, so the
induced velocity of such a pair is identically zero. The cored kernel gives zero there,
but its partial derivatives are terms of order 1/rc^2 that cancel only in exact
arithmetic. The reference copy sets these pairs to zero by the mesh topology (index
masks of the own panel and of the edge line), not by a distance threshold.

The segment kernel uses the projection `r0 . (t1 - t2)` where the target is over the
segment (`r1 . r2 <= 0`), and the equal value `|r1 x r2|^2 (n1 + n2) / (n1 n2 (n1 n2 +
r1 . r2))` elsewhere (Katz and Plotkin 2001, the two forms of the Biot-Savart law of a
straight segment). The second form has no cancellation when `t1` and `t2` are almost
equal (far points, points near the extension of the segment). The semi-infinite kernel
uses `1 + cos(t) = |d x r|^2 / (|r| (|r| - d . r))` behind its start point for the same
reason. Both forms give the same values as the production kernel to round-off.

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

The forward check runs at every call: the differentiated reference copy also gives the
output values, and above 1e-10 relative difference in float64 (1e-4 in float32) the call
raises `RuntimeError` with the names of the outputs. CDi, CDp and CD are compared
relative to their production values. CL and CY are components of one force vector, and
Cl, Cm and Cn of one moment vector: they are compared relative to the largest production
magnitude among the components of the same vector, because the round-off of a component
is relative to its vector (CY, Cl and Cn of a symmetric state are zero by symmetry). The
requested vector outputs (`strip_force`, `section_cl`) are compared relative to the
largest production magnitude of the array.

## Implementation

* `ventorum/gradients/reference.py`: the differentiable copy (lattice from edge nodes,
  system rows, Kutta-Joukowski and leg forces with the tangential velocity, Trefftz
  drag, profile drag and section moment of linear airfoils, ground images, cross-surface
  core, axes transforms).
* `ventorum/gradients/adjoint.py`: `gradients`, `gradient_vjp`, `GradientResult`.
* `ventorum/gradients/check.py`: `check_gradients` with the Richardson reference
  (central differences with steps h and h/2, error estimate |FD(h) - FD(h/2)| / 3).

Device and precision: the gradients run on the device of the production solve (the
device selection of `ventorum.gpu`). The gradient partials and the adjoint solve use the
GPU precision setting: float32 by default, float64 when the setting selects it.
`precision_primal` is the precision of the production solve. The source terms are added
in sequence, with the coordinates in body axes: pairwise addition of the source terms,
coordinates relative to a local origin and compensated (Kahan) addition give the same
float32 accuracy within the round-off, and sequential addition is the fastest. Float32
results carry a note that names the 1e-3 gate.

`check_gradients` compares a component relatively when the error bound of its reference
(the larger of the Richardson estimate and the round-off of the differences) is below
1/10 of the tolerance. The node step is 1e-3 of the smallest core radius of the lattice:
a larger step crosses the core-radius length scale, where the differences do not
converge. The other inputs use a ladder of three steps (1, 0.3 and 0.1 times the base
step, 1e-4 of the base value with an absolute floor for a zero base value); each
component uses the step with the smallest error bound, and the spread to the Richardson
value of the neighbour step joins the bound. Two cases have no relative scale of their
own and use the largest resolved reference of the outputs for the same input (else the
output magnitude per unit input): a component below rtol times that scale, and a zero
derivative (a reference that is zero within its error bound, for example dCL/drho). The
check fails when a component fails and when an input group has no compared component.

## Verification

* `tests/test_gradients.py`: forward parity to 1e-12 (float64), also with the fold in
  the production solve and in the adjoint, adjoint against Richardson references (1e-6
  float64, 1e-3 float32, node directions included), translation and rotation
  identities, agreement with `stability_derivatives`, the symmetric fold rule, the
  strip-force VJP, refusals, the on-segment pairs in float32, the chunk invariance, the
  strict forward check and the GPU device.
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
