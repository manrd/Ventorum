# Changelog

All notable changes to Ventorum are in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before version 1.0.0,
a minor version can change a public interface.

## [Unreleased]

### Added

- Body rates on the GPU pipelines: cases with p, q, r run on the selected device for the vortex
  lattice and both lifting lines, in free air and in ground effect (per-point rotation onset at
  the control points, the three-quarter-chord points and the force points). GPU float64 equals
  the CPU to round-off; rate cases run 3 to 49 times faster than on the CPU on the test machine.
  Rate-free cases keep their speed. `rate_derivatives` and `derivative_matrix` report the
  device. A nonlinear batch with continuation that mixes symmetric and non-symmetric cases still
  runs on the CPU (stated in the theory chapter).
- Gradients with respect to the planform parameters: `gradients`, `gradient_vjp` and
  `check_gradients` accept the input `"planform"` (per surface `semi_span`, `sweep_le`,
  `dihedral`, `incidence`, `position`; per section `chord`, `twist`, `x_le`, `z_le`, `y_frac`,
  `alpha_L0`; lengths in m, angles in rad). The chain rule runs through a PyTorch copy of the
  geometry builder (equal to the production nodes to round-off); the mesh topology stays
  fixed. At a section break that a mesh edge meets, the `y_frac` derivative follows the branch
  that any nonzero step takes, and a note names the section. Mirror pairs with automatic
  reference values get a note: the gradient is per copy.
- Adjoint gradients of control deflections (input `"controls"`, per control name, 1/rad),
  body rates (inputs `"p"`, `"q"`, `"r"`, rad/s), the nonlinear lifting line (Newton Jacobian
  at the converged state; a solve that did not converge is refused; a note past the maximum
  lift) and tabulated airfoils (derivative of the production interpolant; one-sided at a table
  point, with a note). The symmetric fold applies only when every requested input is
  symmetric. A residual check (componentwise backward error) now guards the forward check.
  `check_gradients` covers the new inputs and uses one-sided steps at the 30 deg deflection
  limit. Lifting-line node gradients carry a note: the vortex core on collinear bound
  segments makes them inexact.
- Agent benchmark harness (`validation/agent_benchmark/`, standard library only): a public task
  set of 49 tasks in 9 levels, an MCP client loop for OpenAI-compatible and Anthropic endpoints,
  strict scoring with a provenance check of every reported number, error kinds, repeats, resume
  and a cost cap, and a second track that drives existing clients. References are computed at
  run time; the task file holds no reference values. Keys come only from environment variables
  and are removed from every output.

### Changed

- `input_error_bars` uses the adjoint for the nonlinear lifting line and tabulated airfoils;
  central differences stay only for the Fourier solver. Section inputs (gradients and
  uncertainties) are refused on tabulated airfoils for every solver.

### Fixed

- Adjoint gradients with 24 or more cosine chordwise panels no longer fail the forward check:
  the chordwise panel fractions are now recovered without round-off growth.

## [0.4.0] - 2026-10-11

### Added

- Compiled VJP kernels of the adjoint gradients: hand-derived vector-Jacobian products of the
  horseshoe kernel in numpy, Numba, Cython (OpenMP) and NVIDIA Warp (float32 and float64),
  chosen by the tuner (new profile entry `grad_vjp`; old profiles still load). `gradients`,
  `gradient_vjp` and `check_gradients` take `method="auto"`, `"kernels"` or `"reference"`
  (the PyTorch reference path stays); `GradientResult.partials` names the path and backend.
  The gradients cost about 10 to 17 production solves (before: 90 to 400). Node inputs of
  symmetric requests use the folded system. CPU results are bitwise the same for every thread
  count. On the CPU the kernels evaluate the pair terms in float64 also for float32 requests
  (stated in the notes).
- Derivative matrix: `derivative_matrix` gives the static (alpha, beta; in ground effect also
  phi and h/c), control (per control name) and rate (p, q, r nondimensional) derivatives of the
  six coefficients of one flight state in one result, with the units, the method of each
  column, the axes (`body`, `stability`, `wind` or `all`) and diagnostics (far-field
  derivatives per control with the near/far warning, static margin and neutral point, Irodov
  margin in ground effect). Every column equals the existing function. New
  `static_derivatives` for the free-air static columns. MCP tool
  `ventorum_derivative_matrix` (twelve tools now). The agent tool `ventorum_error_bars` takes
  optional `input_uncertainties` and then returns the combined bars of layers 1 and 2. The
  totals of the agent tools give `CY_farfield` and `Cn_trefftz`.
- Far-field lateral diagnostics: every vortex-lattice and lifting-line result gives
  `totals.CY_farfield` (side force of the trailing vortex sheet in the Trefftz plane, wind
  axes) and `totals.Cn_trefftz` (yawing moment of the Trefftz-plane induced drag, stability
  axes), also on the GPU. They are diagnostics; the near-field `CY` and `Cn` stay the result
  values. The trust score warns (no penalty) when the near and far values differ by more than
  5 % and more than a floor (CY 0.001, Cn 0.0002); the yawing moments are compared only when
  the side force is below the CY floor. The control derivatives give the far-field
  derivatives `CY_farfield_delta_per_rad` and `Cn_trefftz_delta_per_rad`.
- Reference-case format: derivative quantities `<C>_alpha` and `<C>_beta` (C in CL, CD, CY,
  Cl, Cm, Cn) with `derivative_unit` (1/deg or 1/rad) and `axes` (default stability axes);
  results against a lift-coefficient list `CL` instead of `alpha_deg` (a CL that the alpha
  sweep does not reach is reported, never extrapolated); slope entries (`slope_of = "CL"`,
  `CL_range`) with a least-squares fit. The validation runner computes the Ventorum values
  with central differences (alpha 0.5 deg, beta 1 deg, one mesh), as the agent tool
  `ventorum_stability_derivatives`. Existing case files do not change.
- Error bars, layer 2 (input uncertainty): `input_error_bars` propagates the standard
  uncertainties of the flow inputs (alpha, beta, phi [rad], V_inf [m/s], rho [kg/m^3], h [m])
  and of the linear section properties per surface (a0, alpha_L0, Cd0, Cm0, fully correlated
  along the span) to the six coefficients with the first-order law of the GUM (JCGM 100:2008),
  with optional correlations of the flow inputs. The half-width is k u_c (default k = 2). The
  gradients come from the adjoint method; outside its scope the flow inputs use central
  differences of the production solve and section inputs are refused. Each bar gives the share
  of each input. `combine_error_bars` combines layers 1 and 2 by root sum of squares. New
  error-bar statuses `"input_only"` and `"numerical_and_input"`.
- `GradientScopeError` (a subclass of `ValueError`): the gradients raise it for the
  configurations that they do not cover yet (nonlinear lifting line, tabulated airfoils,
  deflected controls, body rates, Fourier solver).
- Adjoint gradients (phase 1): `gradients` gives exact gradients of the scalar outputs
  (CL, CY, CDi, CDp, CD, Cl, Cm and Cn, in body, stability or wind axes) with respect to
  the lattice nodes, the flow variables, the linear section properties and the reference
  values, for the vortex lattice and the linear lifting line, from one solve and one
  adjoint computation per output (discrete adjoint, no finite differences, fixed mesh
  topology). `gradient_vjp` covers the vector outputs (`strip_force`, `section_cl`) with
  a seed per output. `check_gradients` checks the gradients against Richardson-extrapolated
  central differences. The reference copy reports `forward_mismatch` and raises above
  1e-10 in float64 (1e-4 in float32). Gradients run on the device of the production solve,
  in float32 by default (float64 through the precision setting); the folded symmetric
  system is used only when the case and every requested input and output are symmetric.
  The row chunks of the O(n^2) work bound the memory also in the backward pass.
  `GradientResult` holds the output. The functions are in the subpackage `ventorum.sensitivity`
  and are exported from `ventorum`.
- Height derivatives: `height_derivatives` gives the static derivatives of one flight state in
  ground effect (`FlightCondition.h` set): CL, CD, CY, Cl, Cm and Cn with respect to alpha, beta,
  the bank angle and h/c_ref, about the moment reference point with its height held fixed (central
  differences, one batch on one mesh, on the selected device), and the pitch and height
  aerodynamic centres with the Irodov margin of that state. Moments in body, stability or wind axes
  (with the reference lengths b and c).
- Body rates and rate derivatives: `FlightCondition` has the body angular rates `p`, `q`, `r`
  [rad/s] (flight-dynamics body axes, rotation about the moment reference point). The vortex lattice
  and the linear and nonlinear lifting lines solve the quasi-steady rotating flow, in free air and in
  ground effect (the lifting lines take the section onset at the three-quarter chord; one chordwise
  panel gets the thin-airfoil section moment of the rotation). `rate_derivatives` gives the
  derivatives of CL, CD, CY, Cl, Cm and Cn per unit p b/2V, q c/2V and r b/2V (central differences,
  one batch), in body, stability or wind axes (`rate_axes_transform`). The Fourier solver refuses
  rates; cases with rates run on the CPU path.
- Control derivatives and hinge moments: `control_derivatives` gives, for each control name, the
  derivatives of CL, CD, CY, Cl, Cm and Cn per unit deflection (central differences, one-sided near
  the 30 deg limit; moments in body, stability or wind axes), and for each copy of each control the
  hinge-moment coefficient with its derivatives in deflection and angle of attack. `hinge_moments`
  gives the hinge moments of one solved case: from the bound-vortex forces of the flap panels
  (vortex lattice with two or more chordwise panels) or from thin-airfoil section theory (lifting
  lines). `hinge_moment_derivatives` gives the thin-airfoil Ch_alpha and Ch_delta of a plain flap
  (Glauert 1926).
- Numerical error bars (layer 1 of the error bars): `numerical_error_bars` solves a case on three
  spanwise mesh levels and gives, for each coefficient, an interval from the grid convergence
  index (Roache 1998) with the observed order of Celik et al. (2008). Each bar states its
  convergence state (`monotonic`, `oscillatory`, `not_converged_order`, `divergent` or
  `roundoff`) and its notes. `ErrorBar` and `ErrorBarResult` hold the output.
- `validation/gpu_gate.py`: one command for the GPU checks of a phase gate (all GPU tests and all
  examples with `VENTORUM_DEVICE=gpu`, optionally again in float64). It writes a short Markdown log
  with the versions, the GPU model and one row per step, and no machine identifiers. Exit code 0
  (pass), 1 (a step failed or timed out) or 2 (no GPU).
- `validation/perf_history.py`: a speed history of fixed workloads per commit (record mode, which
  times the code of its own tree) and a side-by-side timing of two source trees (compare mode,
  one fresh process per run, alternating order). The output holds no machine identifier.
- Control surfaces: `ControlSurface` on a `LiftingSurface` (spanwise limits, hinge line as a
  fraction of the chord, deflection, symmetric or antisymmetric on the mirror copy, limit 30 deg),
  and `Aircraft.set_deflection`. The vortex lattice with two or more chordwise panels turns the
  normals of the panels aft of the hinge. The lifting lines, and the vortex lattice with one
  chordwise panel, use a section polar deflected by thin-airfoil theory (Glauert 1926). An
  antisymmetric deflection turns off the symmetry fold. The Fourier solver refuses a deflected
  control.
- Deformed geometry: `NodeDisplacements` on a `LiftingSurface` moves the leading-edge and
  trailing-edge nodes of the lattice (for example from a structural solver). `undeformed_nodes`
  gives the nodes of the mesh, and `displacements_from_section_motion` builds displacements from
  section heave and twist. The reference values stay those of the undeformed shape. An optional
  station check refuses displacements built for another mesh. The Fourier solver refuses node
  displacements.
- Trim solver `trim`: a Newton method with a central finite-difference Jacobian finds the angle
  of attack and the control deflections for a target CL and zero Cm (and, on request, zero Cl
  and Cn), in free air and in ground effect at a constant height. It runs on the CPU in float64.
  A target that cannot be reached gives a status and notes; invalid input raises ValueError.
  `examples/11_trim.py` shows its use.
- `validation/avl`: a black-box comparison harness with AVL. It writes AVL input files from a
  Ventorum aircraft, runs AVL with a command stream, reads its output (the exit status of AVL is
  not used), checks the angles and the mesh counts of each run, and compares forces, moments and
  derivatives with Ventorum in free air and in ground effect (geometry rotation). It also
  compares the run times. Nothing from AVL is in the code: only its input and output formats.
- Agent tools for the P1 features: the surface object takes `controls` (flaps, ailerons,
  elevators, rudders; deflections in degrees) and `node_displacements` (metres), so every geometry
  tool accepts them. New tools: `ventorum_trim` (target CL with zero moments; a target that is not
  reached gives `trimmed: false` and a status, not an error), `ventorum_error_bars` (layer-1
  numerical error bars) and `ventorum_undeformed_nodes` (the mesh nodes to build displacements
  from). The agent interface now has 11 tools.

### Changed

- **Breaking (agent tools and MCP):** the geometry argument has no `anyOf` now, because some MCP
  clients drop the keys of an `anyOf` object argument (a multi-surface `wing` arrived empty). Each
  tool that takes a geometry takes exactly one of `wing` (one surface) or `aircraft` (several
  surfaces, with `surfaces` and the reference values). A call with both or neither is refused with
  `invalid_input`. A `wing` with `surfaces` or a reference value is refused with a hint to
  `aircraft`. Each `ventorum_batch_evaluate` candidate is an object with exactly one of the keys
  `wing` or `aircraft`. The Python tool functions take `aircraft=None` after `wing`.
- Agent tools: a refusal of an unknown key says where the key belongs when the same tool accepts
  it at another place (for example `'solver' belongs in 'settings'`), and new hints cover common
  invented keys (`x_start`, `position`, `cg`, `mach`, `tail` and others).
- Breaking: `transform_moments` takes the keyword-only arguments `b_ref` and `c_ref` [m] (required).
  The new function `moment_axes_matrix` returns the 3 x 3 matrix of a change of moment axes.
  `SolverResult` has the fields `b_ref` and `c_ref` [m]; every solver sets them, and
  `SolverResult.moments` needs them for the stability and wind axes.
- `trim` runs its solves on the device that the settings and the global device select, and no
  longer sets the global device to the CPU. `TrimResult` has `device` and `precision`. In float32
  the convergence tolerances are at least `FLOAT32_TOLERANCE_FLOOR` (1e-6, about five times the
  float32 round-off of the coefficients), with a note. The agent tool `ventorum_trim` follows.
- The default angle of attack is 5 deg in every interface. The agent tool `ventorum_ground_effect`
  changes from 4 deg to 5 deg. The `Ventorum` instance changes from 0 deg to 5 deg. The
  ground-effect Python API (`analyze_ground_effect`, `prepare_ground_case`, `sweep_height`,
  `sweep_roll`, `GroundEffectSweep.run_sweep` and `GroundEffectCondition`) changes from 4 deg
  to 5 deg. The agent flight condition, `vt.analyze` and `FlightCondition` already used 5 deg.
- The agent tools refuse an explicit JSON `null` for every key, at the top level and in nested
  objects (`invalid_input` that names the key). Omit the key to get its default. In a Python call,
  `None` for an optional argument still means "use the default".
- Lower limits on the input: semi-span and section chord at least 1e-6 m, planform area at least
  1e-12 m^2, reference area `S_ref` at least 1e-12 m^2, reference span `b_ref` and reference chord
  `c_ref` at least 1e-6 m. A smaller value gave an internal division by zero or overflow in the
  loads; it is now an input error.
- The free-stream speed must be at least 0.1 m/s (`validate_flight_condition` and the agent
  schemas). A smaller speed gave an internal division by zero in the loads.
- The main surface (the source of the reference values) is the surface with the largest projected
  planform area. The texts now say this; the code did not change.
- Documentation: the README names the tuned kernel backends, the API index lists the hardware
  and GPU packages, and the agent map lists all packages. The recovery audit refers to its
  measured speed tables and gives no typed speed ratios.
- Continuous integration uses the first versions of the checkout and setup-python actions that
  run on Node.js 24.
- Ground-effect sweeps on the CPU solve their cases as batches (one dense solve and one loads
  pass per chunk), grouped by unknown map, with the probe lattice from the lattice cache. The
  induced drag in the Trefftz plane uses one half of a symmetric case. The results are the same
  to round-off (the induced drag changes by about 1e-14 relative). The sweep does not change the
  device setting.
- Breaking: the worker-count argument `n_workers` is now `n_jobs` everywhere (`GroundEffectSweep`,
  `sweep_height`, `sweep_roll`, `sweep_alpha`, `Ventorum`, `Ventorum.analyze_sweep`,
  `run_parallel_instances`, `VentorumCaseManager.create_case` and the agent tool
  `ventorum_batch_evaluate`). The old name is removed: a Python call with `n_workers=` raises
  `TypeError`, and the agent tool refuses the key with a hint to `n_jobs`. `Ventorum(n_jobs="auto")`
  now uses the automatic parallel plan (before, `"auto"` meant 1).
- The canonical solver names are `"auto"`, `"vlm"`, `"linear"`, `"nonlinear"` and `"fourier"`. The
  old names `"horseshoe"`, `"lattice"`, `"llt"` and `"linear_llt"` still work, with one
  `FutureWarning` at the public call that names the new name.
- Each solver has one canonical class name that matches its solver string: `VortexLatticeSolver`
  (`"vlm"`), `LinearLLTSolver` (`"linear"`), `NonlinearLLTSolver` (`"nonlinear"`) and
  `FourierSolver` (`"fourier"`). The old names `HorseshoeSolver`, `LinearSolver` and
  `NonlinearSolver` still work from the same modules (the same class object) until version 1.0,
  with one `FutureWarning` that names the new name. They are not in `__all__` any more. The
  non-convergence warning of the nonlinear lifting line names `NonlinearLLTSolver`.
- The vortex-lattice force on the chordwise legs that lie on the surface uses only the velocity
  tangent to the surface. The normal velocity at the leg points is a leak of the discrete lattice;
  its force was an in-plane force that an impermeable surface cannot carry. Results change for
  cases with an antisymmetric control deflection at angle of attack (no aileron side force on a
  flat wing; the aileron yawing moment now agrees with the Trefftz-plane value) and, at second
  order, for non-planar wings in sideslip. The rolling moment due to sideslip of flat wings does
  not change. Verification case V9 (fin side force) and V10 (separate surfaces at a dihedral
  break) change at the fifth digit and by up to 0.3 percentage points.

### Fixed

- Small defects from the review of P1: `input_error_bars` refuses repeated correlation pairs
  and a covariance that is not positive semi-definite; `combine_error_bars` accepts a layer-1
  result without a CD bar (with a note) and no longer raises when a half-width is None; the
  numerical error bars note the refinement ratios of surfaces with their own `n_panels`; the
  Irodov centre-slope threshold is one constant (1e-9) in the height derivatives and the
  sweeps; the trim result notes a start point clipped into the bounds; the Mach warning of the
  input checks uses the ISA speed of sound at the flight density; a zero-valued component of
  `check_gradients` counts as compared only when its bound is small enough.
- The float32 reference copy of the gradients gave NaN for a target point on a segment end
  point (a distance floor of 1e-300 is 0 in float32); the floor now follows the precision.
- `check_gradients` assumed a round-off level of the solve that is too small for node motion;
  it now measures the noise of each node direction and skips components at the noise level.
  The node gradients of the linear lifting line are verified with a polynomial fit.
- `import ventorum` loaded PyTorch through the gradient package; the gradient functions now
  load on first use.
- A lattice-only value of None (the polar limits of linear airfoils) was computed again for
  every case of a sweep; it is now cached.
- Moment coefficients in wind axes: the change of axes now acts on the dimensional moments,
  C' = L^-1 R L C with L = diag(b_ref, c_ref, b_ref), because Cl and Cn use the reference span and
  Cm uses the reference chord. Before, the rotation acted on the coefficients, so the wind-axis Cl
  and Cm were wrong with sideslip when b_ref is not equal to c_ref (for example the wind-axis Cl of
  a wing of aspect ratio 10 at 5 deg sideslip by about 75 %). This changes `SolverResult.moments`,
  the wind set of the agent tools, the wind-axis derivatives of `control_derivatives` and of the
  agent tool `ventorum_stability_derivatives`. Body and stability axes, the rate and height
  derivatives (already correct) and the verification report do not change.
- The agent tools accept a 1-D numpy array where they accept a list of numbers.

## [0.3.0] - 2026-10-04

This version closes the first development phase (P0): the verified core, its kernel backends,
the GPU pipelines, the theory manual and the first validation runs.

### Added

- Cython kernel backend for the verified vortex kernels (OpenMP threads on Linux and Windows).
- PyTorch kernel backend for the verified vortex kernels, on the CPU and on a CUDA GPU.
- The tuner (`ventorum-tune`) measures each backend for each kernel function and case size, and
  writes the fastest one to the machine profile. `"auto"` uses the profile at run time.
- GPU pipelines of the vortex lattice and of the linear and nonlinear lifting line
  (`ventorum.gpu`): batches and single solves, free air and ground effect, float32 (default) and
  float64, with fused NVIDIA Warp kernels. The optional extra `gpu` installs Warp. The tuner
  stores a cost model of the GPU, and `device="auto"` uses it.
- Selectable axes of the force and moment coefficients (`body`, `stability`, `wind` or `all`),
  also for the stability derivatives. Every result states its axes.
- Every result states the device (CPU or GPU) and the precision (float32 or float64) of its solve.
- Reference-case format (one Markdown file with one `toml` block), its parser, and the
  validation runner `validation/run_validation.py`. A case can give a ground height for each
  point. Theory cases have no grade and do not count in the validation statistics.
- `transform_lattice`: rotate and move a vortex lattice about a point.
- Benchmark suite of Ventorum (`validation/benchmark_suite.py`).
- MCP tools for the machine capabilities and for a tuning run.
- A work budget for each agent tool call: a call that is too large is refused before the solve.
- Theory manual: one chapter for each model, with its equations, assumptions, envelope of
  validity and references with credit to the authors.
- Recovery audit of the original code (`docs/design/recovery_audit.md`) and the performance
  architecture (`docs/design/performance_architecture.md`).
- Known speed limits in `docs/user/performance_limits.md`.
- This changelog.

### Changed

- The project and the package are now Ventorum (`ventorum`).
- Faster single solves and sweeps of all lattice solvers on the CPU: lattice and geometry
  caches, one kernel pass in the lifting line, lifting-line sweeps as one batch, vortex-lattice
  sweeps in batches. Results are unchanged to round-off.
- Faster Fourier (classical) lifting line and Numba vortex kernel, with no change of results.
- The cross-surface vortex core in the Trefftz plane uses half the strip width. Verification
  case V10 changed with it.
- Two surfaces that meet along an edge are joined also when their edges match only to a
  tolerance. Edges that are near but not joined give a warning.
- The main surface is chosen by a rule that does not depend on the order of the surfaces.
- In ground effect, all cases of one derivative or one sweep use the same chordwise mesh.
- The trust score is lower, with a warning, when a solve uses section data outside the polar
  table, passes the stall of the polar, or leaves the flow envelope (Mach number above 0.3, low
  aspect ratio at a high angle of attack).
- The agent tools report the trust of each ground-effect row, solve the angles that the agent
  asks for, and give no mesh recommendation without convergence. The machine-capabilities tool
  sends no machine identifier.
- Every dependency has an upper version bound.

### Fixed

- Inputs that cannot give a valid result (non-finite or extreme values, overlapping surfaces)
  are refused before the solve with a `ValueError` that names the field.
- A `TabulatedAirfoil` keeps each Cl, Cd and Cm value with its alpha value, whatever the order
  of the assignments.
- The classical lifting line refuses geometry outside its model (sweep, dihedral, vertical
  offset) and names the vortex lattice as the solver to use.
- The classical lifting line computes the pitching moment as r x F, also when the wing and the
  moment reference point are at different heights.
- A `Ventorum` instance always returns the results of its last run.
- A mesh-convergence study says "converged" only when a mesh meets the tolerance, and reports
  the panel count that it really solved.
- The Irodov height-pitch margin of one flight state has one value, whatever height reference
  the user picks.
- A solve does not change the BLAS thread count of the process and does not make the user's
  arrays read-only.
