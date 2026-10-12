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
lattice built from the edge nodes, the residual (system rows) and the outputs. The parts
with O(n^2) work (system rows, near-field load points, Trefftz points) run in one of two
paths, selected by the keyword `method` (see "Vector-Jacobian products of the kernel"
below):

* The compiled kernels (`method="kernels"`, and the default `"auto"` unless the
  backend selection below gives the reference copy): one forward kernel and one
  hand-derived vector-Jacobian-product (VJP) kernel per part, in Numba, Cython or numpy
  on the CPU and NVIDIA Warp on the GPU. The Numba, Cython and Warp kernels store no
  tensor of point-source pairs; the numpy backend holds the pairs in row chunks of about
  4e5 values.
* The PyTorch reference copy (`method="reference"`): row chunks with automatic
  differentiation. The chunk size comes from a memory budget (`VENTORUM_GRAD_CHUNK`,
  default the budget of `vortex_torch`, in values). The budget counts all values that
  one chunk holds in its forward and in its backward pass. The forward pass keeps only
  the inputs of each chunk; the backward pass evaluates each chunk again, takes its
  vector-Jacobian product and frees it before the next chunk. So the memory of the
  gradient stays bounded by one chunk, also for one backward pass per output.

The O(n) parts (lattice from the nodes, forces, coefficients) stay in PyTorch on both
paths. Both paths give the same gradients to round-off.

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

### Vector-Jacobian products of the kernel

For given adjoint weights on the induced velocity, the VJP gives the derivatives of the
kernel with respect to the target points, the source corners, the wake directions, the
core radii and the source weights (sign times circulation). The derivation applies the
reverse mode (Griewank and Walther 2008, chapter 3) to each elementary operation of the
kernel formula, in reverse order. Below, `u` is the adjoint (the weight) of the velocity
`v` of one filament at one target, and `g_x` is the derivative of `u . v` with respect to
`x`.

Finite segment `A -> B` at the point `P` (Katz and Plotkin 2001, section 10.4, with the
core as in `vortex.py`):

```text
r1 = P - A,  r2 = P - B,  r0 = B - A,  c = r1 x r2
n1 = sqrt(|r1|^2 + f),  n2 = sqrt(|r2|^2 + f),  t1 = r1 / n1,  t2 = r2 / n2
proj = |c|^2 (n1 + n2) / D,  D = n1 n2 (n1 n2 + r1 . r2)     if r1 . r2 > 0
proj = r0 . (t1 - t2)                                         else
den = |c|^2 + rc2 |r0|^2,  k = proj / (4 pi den),  v = k c
```

The floor `f` is 1e-300 in float64 arithmetic (all CPU kernels, which compute in
float64) and 1e-30 in the float32 Warp kernels and in the numpy reference with float32
arrays. The PyTorch reference copy adds 1e-300 in both precisions. A pair with `den` at or below
the floor has `v = 0` and zero derivatives. The reverse sweep:

```text
g_c = k u,  g_k = u . c,  g_proj = g_k / (4 pi den),  g_den = -g_k k / den
g_cr2 = g_den,  g_rc2 = g_den |r0|^2,  g_r0sq = g_den rc2
stable form:  g_cr2 += g_proj (n1 + n2) / D,  g_n1 = g_n2 = g_proj |c|^2 / D
              g_D = -g_proj proj / D,  g_n12 = g_D (2 n1 n2 + r1 . r2),  g_dot = g_D n1 n2
              g_n1 += g_n12 n2,  g_n2 += g_n12 n1,  g_r1 = g_dot r2,  g_r2 = g_dot r1
direct form:  g_r0 = g_proj (t1 - t2),  g_t1 = g_proj r0
              g_r1 = g_t1 / n1,  g_r2 = -g_t1 / n2
              g_n1 = -(g_t1 . t1) / n1,  g_n2 = (g_t1 . t2) / n2
norms:        g_r1 += (g_n1 / n1) r1,  g_r2 += (g_n2 / n2) r2
cross:        g_c += 2 g_cr2 c,  g_r1 += r2 x g_c,  g_r2 += g_c x r1,  g_r0 += 2 g_r0sq r0
end points:   g_P = g_r1 + g_r2,  g_A = -g_r1 - g_r0,  g_B = g_r0 - g_r2
```

Semi-infinite line from `A` along the unit vector `d`:

```text
r = P - A,  c = d x r,  rn = sqrt(|r|^2 + f),  dr = d . r
oc = |c|^2 / E,  E = rn (rn - dr)    if dr < 0
oc = 1 + dr / rn                     else
den = |c|^2 + rc2,  k = oc / (4 pi den),  v = k c
```

```text
g_c = k u,  g_k = u . c,  g_oc = g_k / (4 pi den),  g_den = -g_k k / den
g_cr2 = g_den,  g_rc2 = g_den
behind:  g_cr2 += g_oc / E,  g_E = -g_oc oc / E,  g_rn = g_E (2 rn - dr),  g_dr = -g_E rn
ahead:   g_dr = g_oc / rn,  g_rn = -g_oc dr / rn^2
g_r = (g_rn / rn) r + g_dr d,  g_d = g_dr r
g_c += 2 g_cr2 c,  g_d += r x g_c,  g_r += g_c x d
g_P = g_r,  g_A = -g_r
```

The horseshoe is the bound segment `a -> b`, the legs `a_te -> a` and `b -> b_te`, plus
the ray from `b_te` minus the ray from `a_te` (adjoint `-u`). The derivatives of the
corners add the end-point derivatives of the filaments that share them. The pair core is
`rc2 = rc_src^2 + rc_tg^2` for a target and a source of different core groups, else
`rc_src^2`; `g_rc2` goes to the source core and, for the pairs of different groups, to
the target core. The pairs that the mesh topology puts on a segment (see above) have zero
velocity and zero derivatives on both paths.

The products use the kernel with the source weights `s_j` (sign, or sign times
circulation) and the unknown `col_j` of each source:

```text
system rows:      A[i, col_j] += s_j N_i . v_ij,  adjoint W[i, n]:  u_ij = s_j W[i, col_j] N_i
induced velocity: V_i = sum_j s_j v_ij,           adjoint X_i:      u_ij = s_j X_i
g_N_i = sum_j s_j W[i, col_j] v_ij
g_s_j = sum_i W[i, col_j] N_i . v_ij  (system rows),  g_s_j = sum_i X_i . v_ij  (V)
```

The Trefftz normal wash `w_i = sum_j gamma_j / (2 pi) ((d x r_ij) . n_i) / (|r_ij|^2 +
c2_ij)` with `r_ij = q_i - p_j` has the VJP below, for the adjoint `g_w_i`, with
`e = (d x r) . n_i`, `D = |r|^2 + c2` and `k = gamma_j / (2 pi D)`:

```text
g_e = g_w k,  g_k = g_w e,  g_gamma_j += g_k / (2 pi D),  g_D = -g_k k / D,  g_c2 = g_D
g_n_i += g_e (d x r),  g_v = g_e n_i,  g_d += r x g_v,  g_r = 2 g_D r + g_v x d
g_q_i += g_r,  g_p_j -= g_r
```

The compiled kernels compute each pair once in the forward kernel. The CPU VJP kernels
compute each pair once more (forward and reverse sweep together); the Warp VJP computes
each pair twice (one row launch and one column launch). With Numba and Cython the rows
run in parallel in blocks;
each block adds its source derivatives into its own buffer, and the buffers are added in
block order. The block partition depends only on the row count (blocks of 32 rows, at
most 128 blocks), not on the thread count, so the result is bitwise the same for every
thread count and thread timing. The numpy backend runs the rows in sequence, in chunks.
Each pair is computed in float64 on the CPU (scalar CPU
code has the same speed in float32 and float64), also for a float32 request: the O(n^2)
partials of the CPU kernels are then float64 values cast to float32, while the O(n)
parts and the adjoint solve use float32. A float32 `GradientResult` of a CPU kernel
path says this in its `notes`. On the
GPU one launch adds the target derivatives per block of sources and a second launch adds
the source derivatives per block of targets; the block sums are added in float64, and no
atomic additions are used. The float32 GPU kernels compute each pair in float32. On the
kernel path the system rows of the lifting line take the normal component per source
(the reference copy contracts the velocity tensor of each unknown with the normals); the
results are equal to round-off.

## Assumptions

* The mesh topology is fixed: spanwise stations, chordwise count and chordwise
  fractions do not change with the inputs.
* The planform input (`"planform"`, see `ventorum/sensitivity/planform.py`) refines
  this rule: the panel count, the spacing choice and the snap assignment of each
  spanwise station stay fixed, but the stations move with the parameters. A station
  snapped to a section break tracks the span fraction of its section; the control
  stations next to a moved snap are the means of their edges, as in the mesh
  builder. The chain rule runs through a differentiable port of the geometry
  builder: edge geometry, station rule, strip interpolation of the section
  properties, control-point fractions and automatic reference values. The port
  reproduces the numpy builder to 1e-14.
* An edge on its break (a station that coincides with a section break at the base
  point) switches the control stations from the spacing rule to the edge means on
  any practical span-fraction step. The gradient there is one-sided, and
  `check_gradients` skips it with a reason. A parameter change that adds, removes
  or moves a snap to another edge (or flips the automatic spacing) is skipped the
  same way: the fixed-topology gradient has no finite-difference reference there.
  Wings whose interior sections are all kinked keep every break and verify fully.
* The planform parameters per surface are the semi-span [m], the leading-edge
  sweep [rad], the dihedral [rad], the incidence [rad] and the position [m], and
  per section the chord [m], the twist [rad], `x_le` [m] (None when unset), `z_le`
  [m] (None when unset), the span fraction (dimensionless) and the airfoil
  incidence `alpha_L0` [rad]. The root and tip span fractions are pinned (0 and
  1). The position y of a symmetric surface must stay 0. A uniform position shift
  of a surface equals the summed node gradients of that surface, and all position
  gradients plus the reference-point gradient sum to zero (the translation rule).
* The flow is linear in the circulation: vortex lattice and linear lifting line with
  linear airfoils, zero control deflections and zero body rates.
* The folded symmetric system applies only when the case is symmetric and every
  requested input and output is symmetric (no beta, phi, section or reference change,
  no lateral output). Node inputs are symmetric when every surface is symmetric: the
  inputs are the nodes of the defining half, and the other half is their mirror image.
  Else the full system is used. The y-change of the root node of a symmetric surface is
  the only node change that the fold does not cover. Its exact derivative is zero: the
  geometry with the root at y = +e is the mirror image of the geometry with the root at
  y = -e, so a symmetric output is an even function of e. The fold sets this derivative
  to zero; the full system gives zero to round-off.

## Envelope

The functions raise `GradientScopeError` (a `ValueError`) for the configurations that
they do not cover yet: the nonlinear lifting line, tabulated airfoils, nonzero control
deflections, nonzero body rates and the Fourier solver. A later version adds them. In ground effect `h` is the height of the moment reference point, and the
ground plane and its images follow the inputs; a ground strike raises as in `analyze`.

The forward check runs at every call: the differentiated reference copy also gives the
output values. Above 1e-10 relative difference (when the gradients and the production
solve are both float64; else above 1e-4) the call raises `RuntimeError` with the names
of the outputs. CDi, CDp and CD are compared relative to their production values; a
production value of exactly zero (for example CDp with Cd0 = 0) uses the scale of the
force vector. CL and CY are components of one force vector, and
Cl, Cm and Cn of one moment vector: they are compared relative to the largest production
magnitude among the components of the same vector, because the round-off of a component
is relative to its vector (CY, Cl and Cn of a symmetric state are zero by symmetry). The
requested vector outputs (`strip_force`, `section_cl`) are compared relative to the
largest production magnitude of the array.

## Implementation

* `ventorum/sensitivity/reference.py`: the differentiable copy (lattice from edge nodes,
  system rows, Kutta-Joukowski and leg forces with the tangential velocity, Trefftz
  drag, profile drag and section moment of linear airfoils, ground images, cross-surface
  core, axes transforms).
* `ventorum/sensitivity/adjoint.py`: `gradients`, `gradient_vjp`, `GradientResult`.
  `wrt="planform"` is refused together with `wrt="nodes"` (the same freedom); it
  always uses the full system.
* `ventorum/sensitivity/planform.py`: the differentiable port (edge geometry, station
  rule with frozen snap assignment, strip properties, fractions, automatic
  references) and the full-strip assembly maps.
* `ventorum/sensitivity/kernels.py`: the autograd functions of the kernel path, the
  backend adapters and the backend selection (`method`, the machine profile entry
  `grad_vjp`, `set_vjp_backend`).
* `ventorum/aero/vortex_vjp.py` (numpy reference of the VJP), `vortex_vjp_numba.py`,
  `vortex_vjp_cython.pyx` and `ventorum/gpu/vjp_kernels.py` (Warp, float32 and
  float64): the forward and VJP kernels.
* `ventorum/sensitivity/check.py`: `check_gradients` with the Richardson reference
  (central differences with steps h and h/2, error estimate |FD(h) - FD(h/2)| / 3).

Path of the partials: `method="auto"` (the default) takes the backend that
`set_vjp_backend` forces, else the `grad_vjp` entry of the machine profile for the
device and the size class, else the default backend of the device: Numba (then Cython,
then numpy) on the CPU, Warp on the GPU, and the PyTorch reference copy on a GPU without
Warp. `method="kernels"` takes the forced or the default compiled backend and raises
`ValueError` when the device has no compiled backend. `method="reference"` takes the
PyTorch reference copy. `GradientResult.partials` names the path and the backend (for example
`"kernels:numba"` or `"reference:torch"`); `GradientResult.method` stays `"adjoint"`.

Device and precision: the gradients run on the device of the production solve (the
device selection of `ventorum.gpu`). The gradient partials and the adjoint solve use the
GPU precision setting: float32 by default (or the precision of the machine profile),
float64 when the setting selects it.
`precision_primal` is the precision of the production solve. The source terms are added
in sequence, with the coordinates in body axes: pairwise addition of the source terms,
coordinates relative to a local origin and compensated (Kahan) addition give the same
float32 accuracy within the round-off, and sequential addition is the fastest. Float32
results carry a note that names the 1e-3 gate.

`check_gradients` compares a component relatively when the error bound of its reference
(the larger of the Richardson estimate and the round-off of the differences) is at most
rtol/10 times the magnitude of the reference. The node step is 1e-3 of the smallest core radius of the lattice:
a larger step crosses the core-radius length scale, where the differences do not
converge. The node directions have one step and no spread term, so their round-off
bound is also at least 6 sigma / h, where sigma is the noise level of the production
solve per output: the standard deviation of the residuals of a quadratic fit through 10
solves at a spacing of 1e-3 of the node step along the same direction. A node motion
changes the rounding of all lattice quantities, and this noise can be much larger than
the machine epsilon times the output (about 80 times on a wing and tail in ground effect
with one chordwise panel). The Richardson value has a noise of about 1.9 sigma / h.
The other inputs use a ladder of three steps (1, 0.3 and 0.1 times the base
step; the base step is 1e-4 rad for the angles, 1e-4 of the value for `V_inf`, `rho`,
`h`, `S_ref`, `b_ref` and `c_ref`, 1e-4 `c_ref` for `ref_point`, and max(1e-4 |x|, floor)
for the section properties, with a fixed floor per property); each component uses the
step with the smallest relative error bound, and the spread to the Richardson
value of the neighbour step joins the bound. Two cases have no relative scale of their
own and use the largest resolved reference of the outputs for the same input (else the
output magnitude per unit input): a component below rtol times that scale, and a zero
derivative (a reference that is zero within 10 times its error bound at every step,
for example dCL/drho). A zero derivative counts as compared only when 10 times its
error bound is below rtol times that scale; else it is skipped, with a reason. The
check fails when a component fails and when an input group has no compared component.

## Verification

* `tests/test_gradients.py`: forward parity to 1e-12 (float64), also with the fold in
  the production solve and in the adjoint, adjoint against Richardson references (1e-6
  float64, 1e-3 float32, node directions included), translation and rotation
  identities, agreement with `stability_derivatives`, the symmetric fold rule, the
  strip-force VJP, refusals, the on-segment pairs in float32, the chunk invariance, the
  strict forward check and the GPU device. The kernel path: gradients equal to the
  reference path to 1e-12 (float64) for all parity configurations and CPU backends, the
  float32 gate (1e-3) for each CPU backend, the Warp gradients on the GPU, the `method`
  keyword and the fold of the node inputs (equal to the full system to 1e-12).
* `tests/test_gradient_planform.py`: port parity to 1e-14 (sections, sweep, dihedral,
  incidence, position, symmetric and mirror surfaces, section and control breaks,
  explicit `x_le`/`z_le`), Richardson checks of each planform parameter (1e-6
  float64) on the parity configurations and on wings with breaks, the translation
  identity of the position gradients, the nodes-planform refusal, the agreement of
  the airfoil incidence with the sections view, the kernel-reference agreement and
  the planform how-to example. Components whose mesh stations jump within the
  difference step are skipped with a reason.
* `tests/test_vjp_kernels.py`: each VJP backend (numpy, Numba, Cython, Warp) against
  the PyTorch reference VJP for the products and all input gradients (1e-12 float64,
  1e-5 float32), the thread paths, the backend selection and the machine profile.
* The values come from `validation/run_verification.py` (unchanged by this feature).

## References

* J. R. R. A. Martins and J. T. Hwang, "Review and unification of methods for computing
  derivatives of multidisciplinary computational models", AIAA Journal 51(11), 2013,
  pp. 2582-2599.
* J. R. R. A. Martins and A. Ning, "Engineering Design Optimization", Cambridge
  University Press, 2021, ch. 6.
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge University
  Press, 2001.
* A. Griewank and A. Walther, "Evaluating Derivatives: Principles and Techniques of
  Algorithmic Differentiation", 2nd ed., SIAM, 2008.
* W. F. Phillips and D. O. Snyder, "Modern adaptation of Prandtl's classic
  lifting-line theory", Journal of Aircraft 37(4), 2000, pp. 662-670.
* M. Drela, "Flight Vehicle Aerodynamics", MIT Press, 2014.
