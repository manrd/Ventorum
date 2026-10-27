# Changelog

All notable changes to Ventorum are in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before version 1.0.0,
a minor version can change a public interface.

## [Unreleased]

### Added

- Ventorum-3DP: trust notes for far-field kernel and outer-strip ground effects, and dihedral root scaling warning. A finite `far_ratio` in ground effect adds a trust note on far-field kernel lift bias (about 1 to 4 %); in free air, the note appears when spanwise panel count exceeds 80. In ground effect with a closed tip, a trust note reports that outermost strip loads are not resolved. When station root thickness scaling exceeds $1/\cos(75^\circ) \approx 3.86$, a `RuntimeWarning` is raised and a trust warning is recorded.

- Viscous alpha-correction coupling (V1): trust warning and `RuntimeWarning` when a thick section polar ($t/c > 0.04$) runs with a flap deflection in ground effect below $h/c = 0.30$. The ground factor $\kappa$ of the flap increment is a thin-airfoil value; for thick sections below $h/c = 0.30$ it can differ by more than 2 % (about 0.1 to 0.2 deg of flap shift). `_section_flap_shift` guards against trailing-edge ground strike, raising `ValueError` when the strip trailing edge is at or below the ground ($h \le 0$).

- Ventorum-2D: `solve_viscous_flapped` solves a section with a deflected plain flap by
  continuation in the deflection (from 0 deg, warm-started steps, halved on a failure), where a
  direct cold solve often fails. Each step has its own work budget, with a total cap; the result
  reports the reached deflection and the step history. `solve_viscous` does not change.

- Viscous alpha-correction coupling for the 3D panel method: `solve_viscous(..., model="panel3d")`
  matches the surface-pressure lift of each spanwise strip to its section polar by rotating the onset on
  the strip's body panels (one factorization for a whole sweep), closes the trailing-edge pressure jump
  with the pressure Kutta condition, takes the section lift lines from a 2D Morino solve of each strip
  section, and adds the viscous section moment and the profile drag from the polar. A half mesh with
  asymmetric flow, a GPU backend and the mesh-error estimate are refused.
- Ventorum-2D in ground effect: `Cl` is the surface pressure lift (inviscid and viscous results), and
  the new field `Cl_circulation` keeps 2 Gamma / c (Coulliette and Plotkin 1996). Free-air results do
  not change. Near the ground the circulation lift was 12 % (h/c 0.2) to 33 % (h/c 0.1) too high.

- Adjoint gradients (`gradients`, `gradient_vjp`, `check_gradients`) of vortex-lattice surfaces whose
  lattice lies on a cambered section mean line: the reference copy builds the same mean-line points
  and normals as the solver (also with geometric flaps, joined surfaces and span-varying sections),
  so the earlier refusal for these surfaces is removed. The input `sections` stays refused for a
  shaped vortex-lattice surface (the mean line, not alpha_L0, sets its boundary condition).

- The lattice solvers add a trust warning on Trefftz induced drag CDi (and drag-derived outputs) when the smallest trailing-edge height above the ground is below 0.15 of the local chord (constant `_CDI_TE_CLEARANCE`). The warning states the clearance and notes the dependence of Trefftz CDi on the height of the far wake.

- Ventorum-3DP, a first version of the three-dimensional panel method (`ventorum.panel3d`,
  solver name `"panel3d"`): low-order constant-strength source and doublet panels with the
  internal Dirichlet condition (Morino and Kuo 1974; Maskew, NASA CR-4023; Katz and Plotkin
  2001). Exact quadrilateral influence kernels with far-field point formulas in numpy, Numba
  and NVIDIA Warp (equal to round-off in float64); closed thick-surface meshes from the
  `LiftingSurface` definition and section shapes (new optional `WingSection.shape`), with
  dihedral, blunt trailing edges, tip closures and deflected plain control surfaces; a
  prescribed wake along the body axis in free air (one factorization serves a whole sweep by
  unit-onset superposition) and along the free stream in ground effect; symmetry folding and
  ground effect by images; the Morino Kutta condition and, as an option, the pressure Kutta
  condition; a GPU solve path (Warp assembly, torch factorization); a mesh-convergence error
  estimate. New `SolverSettings` fields `panel_kutta`, `panel_backend`, `panel_wake_length`,
  `panel_chord_spacing`, `panel_estimate_mesh_error`, `panel_scale_root_thickness` (no effect
  on the other solvers). The root section of a symmetric surface with dihedral (in the plane
  y = 0) has its thickness scaled by 1 / cos(dihedral), so that the thickness normal to the
  local surface equals the section thickness (option `scale_root_thickness` of the mesh
  builders, default on).
- Ventorum-2D, the own airfoil code: `ventorum.airfoil` (shape type, Selig and Lednicer
  readers and writers, NACA 4-digit and 5-digit generators, spline repaneling, plain-flap
  deflection), `ventorum.panel2d` (linear-vorticity panel method with source panels, ground
  effect by images, and a viscous-inviscid solver with a global Newton coupling through the
  mass-defect matrix, also batched with torch) and `ventorum.ibl2d` (two-equation integral
  boundary layer with the closures of Drela and Giles 1987, the lag equation and the e^N
  envelope transition model). Some constants are marked in the theory pages to be checked
  against the paper.
- Section polars: `SectionPolarSource` with an XFOIL source and a Ventorum-2D source (the only
  one with ground effect), a polar cache, and polar tables over Reynolds number, flap
  deflection and ground height (`ventorum.aero.polar_source`, `polar_source_v2d`,
  `polar_cache`, `polar_table`).
- First viscous coupling of the vortex lattice and the linear lifting line with 2D polars by
  angle-of-attack correction (`ventorum.viscous.solve_viscous`), with post-stall branch flags
  and a trust entry. On a vortex-lattice strip with a deflected control the polar is read
  relative to its thin-airfoil model (Glauert flap shift), so that polars that follow
  thin-airfoil theory give the inviscid vortex lattice at the same panel count.
- Parasite-drag build-up (`ventorum.aero.parasite`): flat-plate skin friction with transition
  and roughness, form factors, components with an uncertainty field.
- Agent tools `ventorum_airfoil_geometry` and `ventorum_airfoil_polar` with a work budget;
  the server resolves XFOIL only from its own configuration.
- Exact references: Karman-Trefftz airfoils, sphere and ellipsoid surface velocities.
- 2D plots (`ventorum.visualization.airfoil2d`) and two examples.

### Changed

- Ventorum-2D lift in ground effect: `Panel2DResult.Cl` is the surface pressure lift in ground
  effect (`Cl_pressure`, Coulliette and Plotkin 1996, eq. 30). Near a wall the Kutta-Joukowski
  relation does not hold; the circulation lift is 12 % (h/c = 0.2) to 33 % (h/c = 0.1) too high
  for a thin section. A new field `Panel2DResult.Cl_circulation` keeps the circulation lift
  (2 Gamma / c). In free air, `Cl` stays the circulation lift bit for bit. The 2D polar source
  (`Ventorum2DPolarSource`) reports the pressure lift for polars with a ground height.

- Geometric chordwise controls in the vortex lattice (`n_chord >= 2`). The mean surface aft of
  the hinge turns about the hinge axis of each strip edge node, with the moved points in the
  wake, the ground images, the clearance and ground-strike checks and the Trefftz plane. The
  chordwise legs of the panels ahead of the hinge have a node at the hinge point (one extra
  straight segment per leg, in every kernel backend, the GPU pipelines and the gradient
  kernels), so they follow the deflected surface; with straight legs the leg forces made
  the flap slope at incidence too low. A join of 0.2 local chords seals each control side edge (the
  transition of the 3D panel method), so part-span flaps converge with the spanwise mesh. The
  leg-force rules of a lattice with a chordwise control come from the undeflected lattice. The
  results of chordwise controls change, also at zero deflection at the level of the leg-force
  integration; lattices without chordwise controls do not change.
- Vortex lattice on the real mean line of cambered sections. When every section of a surface
  gives a shape (`WingSection.shape`), the lattice points lie on the section mean surface and the
  boundary-condition normals are the normals of that surface (Katz and Plotkin 2001; Coulliette
  and Plotkin 1996 in ground effect), in free air and in ground effect. The section polar
  alpha_L0 and Cm0 then do not enter the boundary condition or the loads (the geometry gives
  them; a warning when the shape and the polar differ by more than 0.5 deg). Surfaces without
  shapes, surfaces with some sections without a shape (with a result note), one chordwise panel
  and the lifting lines keep the turned-normal model; in ground effect a cambered section
  without a shape gets a trust note. Uncambered wings give the same results in bits. Adjoint
  gradients of surfaces with a cambered mean line are refused for now (the gradient chains do
  not yet carry the mean-line offset); symmetric shapes with an uncambered polar keep them.

- The 3D panel solver uses 40 chordwise panels per surface by default (was 20). A mesh-error study of the method found 40 the smallest count for which no test wing gets the mesh warning of the trust score; it costs about 8 times the factorization work and 4 times the matrix memory of 20.
- Ventorum-3DP: the panel solver gives its chordwise panel count (`n_chord`) and its largest
  strip chord to the trust score. The calibrated chordwise rule of the mesh checks and the
  ground rule for chordwise panels longer than the gap now act for the panel method. A result
  with fewer chordwise panels than the warning count, also with the default count, gets the
  chordwise warning (rating MODERATE when no other rule acts).
- Ventorum-3DP mesh: when the root-thickness scaling applies with a first-row dihedral above
  60 deg (factor 1 / cos(dihedral) above 2), the mesh builder issues a `UserWarning` that the
  root section is strongly scaled and that the result needs checking. The factor is not
  capped.
- Ventorum-3DP mesh: the panel mesh builders and the solver factory raise `ValueError`
  when a surface has non-zero node displacements. The message states that the 3D panel method
  does not yet support them, and that the vortex lattice does. Zero or absent displacements
  change nothing.
- Ventorum-3DP mesh-error estimate (`panel_mesh_convergence`, `panel_estimate_mesh_error`):
  the chordwise and the spanwise panel counts are refined separately (three levels each, the
  base mesh first), each with its own Richardson extrapolation, and the error estimate of the
  base mesh is the sum of the two directional estimates (separable error model). The former
  refinement of both counts at once did not converge on thick wings, because the chordwise
  and the spanwise lift errors have opposite signs. The study now solves five meshes (cost
  about 30 base solves at ratio 1.5, before about 142). New `directional_estimate` and
  `MeshErrorEstimate`; `ConvergenceReport.quantities` holds a `MeshErrorEstimate` per
  quantity with the convergence of each direction, and the report has the solved `meshes`.
- Plain flaps: the hinge treatment "repanel" (a spline through the deflected outline with nodes kept at the hinge, the leading edge and the trailing edge) is the default of `deflect_plain_flap` and `AirfoilShape.with_flap`; "none" (the outline as constructed) and "merge" (node removal at the hinge) stay options. A zero deflection returns the section unchanged for every treatment. The 3D panel mesh keeps the outline as constructed. The 2D solver revision is raised, so cached flap polars are computed again.
- `run_xfoil` runs XFOIL headless on every platform: graphics off first (the graphics flag is
  checked), its own temporary folder, no console window on Windows. New options for
  coordinates, transition, panel count, flap (hinge height as a fraction of the thickness) and
  a two-direction sweep; the default sweep stays linear.
- The air viscosity is one constant, `ventorum.core.constants.MU_SL`.
- Ventorum-2D viscous solver (also the batched solver): an angle that does not converge has NaN
  loads (`Cl`, `Cl_pressure`, `Cd`, `Cd_friction`, `Cd_pressure`, `Cm`), NaN transition
  positions and NaN node and wake distributions; its flag, reason, residual and work counters
  stay. Before, it carried the values of its last iterate. The polar source and the agent tool
  already leave such angles out (null in JSON); the Cp plots skip curves without a finite value.
- Ventorum-2D reference chord: by default the distance from the leading edge to the
  trailing-edge midpoint of the section as given (before, the x extent). New argument
  `chord_ref` of `solve_viscous` and `ViscousCase` (it was already in `solve_inviscid`): for a
  flapped section the caller gives the chord of the undeflected section, as in thin-airfoil
  theory and XFOIL; the polar source and the agent tool do. The moment is taken about the
  quarter-chord point of the reference chord (before, the fixed point (0.25, 0)); for a unit
  section with the leading edge at the origin both defaults are the former ones. The results of
  flapped sections change, and those of cambered sections change a little (their x extent is
  not their chord). New functions `ventorum.panel2d.solver.section_chord` and `reference_point`.
- Ventorum-2D reference chord and moment reference point are inputs everywhere: `chord_ref`
  and the new `moment_ref` (a point (x, y) in the section frame) of `solve_inviscid`,
  `solve_viscous`, `ViscousCase` and `build_coupling_system`, options of the Ventorum-2D polar
  source (part of its cache key), and optional inputs of the agent tool
  `ventorum_airfoil_polar` (bounds in the schema; the output states the values used, and
  `ventorum_airfoil_geometry` states the defaults). Explicit values win over everything. The
  former name `x_ref` of `solve_inviscid` stays as an alias of `moment_ref`. When no value is
  given, the defaults come from the reference geometry of the section: the solvers accept an
  `AirfoilShape`, and `AirfoilShape.with_flap` stores the reference geometry of the section
  before the deflection (new `SectionReference`: leading edge, trailing-edge midpoint and
  reference chord; `repanel`, `repanel_cosine`, `normalized` and the factory copy keep it). The
  default moment point and the default pivot are the quarter-chord point on the chord line of
  the reference geometry, 0.25 of the way from the leading edge to the trailing-edge midpoint;
  before, the point was (x_min + c/4, 0). The polar source, the agent tool and the example
  `airfoil_naca_flap.py` give the flapped shape to the solver, so a flapped section uses the
  undeflected chord and its quarter-chord point. Sections with the leading edge at (0, 0) and
  the trailing-edge midpoint at (1, 0) keep their results bit for bit; for other sections the
  moment and, in ground effect, the pivot move to the chord line. A flapped section given as a
  point array uses the chord line of the deflected outline: give the shape or the values.
  `reference_point(body)` has no chord argument now; new `section_reference` and
  `resolve_reference`.
- Ventorum-2D Levenberg-Marquardt fallback: after a fallback step of a Newton solve found no
  decrease of the residual (above the floor of the stagnation stops), the later fallback steps
  of that solve try at most `lm_trials`
  damping values (one linear solve each; default 2, before 12). New option of
  `solve_viscous`, `ViscousCase` and the polar source. Converged angles do not change; failed
  angles need fewer linear solves.
- The 2D solver revision is raised, so cached Ventorum-2D polars are computed again.

### Fixed

- Ventorum-3DP with body rates p, q, r: the surface pressure now includes the kinematic term of the rotating body, Cp = (|V_k|^2 - |V|^2) / V_inf^2, with V_k the local onset velocity. The flow is steady in the body frame, so the unsteady Bernoulli equation of the earth frame applies. The former Cp = 1 - |V|^2 / V_inf^2 gave three times the added-mass force of a sphere on a circular path; on an AR 8 NACA 0012 wing at 4 deg (16 x 16 panels) CL_q changes from 4.33 to 4.66, Cm_q from -0.616 to -0.671, and CY_r (stability axes) from -0.0437 to -0.0031. The fix applies to the CPU and GPU paths, the pressure Kutta and the viscous strip coupling. Results without rates do not change.
- Ventorum-3DP: the flat tip cap of a cambered section gave cap pressures of order 1e3 to 1e4, and with dihedral the cap force made the lift too high (NACA 2412 at 30 deg dihedral: CL 0.556 instead of about 0.43). A tip cap panel now keeps the surface gradient along its row of neighbours only. Symmetric sections do not change.
- Ventorum-2D retries a residual-window stop once for the cold quasi-simultaneous first estimate at the
  requested angle (not at the zero-lift start of an angle continuation). The stage limit and angle
  budget stay unchanged. This lets slow cold starts converge in ground effect (NACA 4412 at 4 deg failed
  before at five heights from h/c 0.7 to 1.3). The solver revision for result caches is now 2026-10-27.1.
- Vortex lattice on a cambered section mean line: the horseshoe legs now follow the mean surface.
  The legs of a panel run from its bound points through the bound points of the panels aft of it on
  the same strip edge to the trailing edge, with one node on the mean surface between each two
  bound points (new lattice and source field `leg_next`; the hinge node of a chordwise flap takes
  the place of the mean-surface node between the two bound points around the hinge). The straight legs
  left the curved mean surface and, on narrow strips, passed close to control points of other
  strips, so the solve was ill-conditioned: an AR 8 NACA 6409 wing gave CL 7.74 at 0 deg with the
  default settings (now 0.502) and a NACA 4412 wing 0.078 (now 0.334). All kernel backends (numpy,
  Numba, Cython, torch, NVIDIA Warp), the leg forces on the CPU and the GPU, the VJP kernels of the
  gradients and the gradient reference copy carry the chains. The leg forces of a chain segment use
  the circulation sum of the panels ahead of it. Strips without a mean line and strips with a zero
  mean line keep their results bit for bit. The trust-score warning on oscillating strip lift stays
  as a diagnostic; the default settings no longer give it. A rebuilt Cython extension is needed for
  the Cython kernels; an older build falls back to Numba for sets with chains.
- Ventorum-3DP: the mesh-error estimate (`panel_estimate_mesh_error`) and `panel_mesh_convergence`
  dropped an explicit ground plane and the `ref_point` of the call, so they estimated free air or the
  moment about the aircraft reference. Both inputs now reach every level of the study.
- Ventorum-3DP: with a spanwise `spacing` in the solver settings and none on the surface, the
  panel mesh used the default spacing while the strip loads used the settings spacing, so panels
  were assigned to the wrong strips (root strip Cl about 2.4 to 20 instead of about 0.39 on an AR 8
  NACA 0012 wing at 4 deg; the totals moved by less than 0.1 %). The mesh now uses the same rule:
  the surface spacing when set, else the settings spacing. Default settings do not change.
- Viscous alpha-correction coupling (VLM): on a lattice on the section mean line, the
  thin-airfoil model of a strip with a deflected chordwise flap lost the camber of the section
  (its clean zero-lift angle was that of the uncambered strip normal). The lift offset of the
  flapped strips was then wrong by about 0.2 in Cl: an AR 8 NACA 2412 wing with a 10 deg flap at
  4 deg gave CL 1.17, now 1.00. The clean angle is now the zero-lift angle of the discrete
  section with the panel normals before the flap turn (new lattice field `normal_bc_clean`).
  Strips without a mean line do not change.
- Ventorum-2D: a singular Newton matrix raised `numpy.linalg.LinAlgError` out of
  `solve_viscous` (for example NACA 0012 with a 20 deg flap and the stop rules off). It is now a
  failed step: the Levenberg-Marquardt fallback is tried, and when that also fails at the full
  coupling the solve stops with the status "singular". The angle then returns
  `converged=False` with a reason.
- Ventorum-2D: the structure update of the Newton solve divided zero by zero at a node next to
  the stagnation point that is not a station (RuntimeWarning in normal runs). The results do
  not change.

### Known limits

- Ventorum-2D: some drag and transition cases do not yet meet their limits (marked as expected
  failures with the reason); plain flaps at small deflections leave a short step panel at the
  hinge that the viscous solver cannot handle yet.

## [0.5.0] - 2026-10-14

### Added

- Compressibility correction for the vortex lattice: the optional `FlightCondition.mach`
  (0 <= M < 1, default None = no correction, results unchanged) applies the Prandtl-Glauert
  transformation in the form of Goethert (NACA TM 1105). The aspect-ratio-8 lift slope follows
  the Helmbold-Diederich formula within 0.5 % up to Mach 0.6. First version: VLM only, free air
  only, on the CPU and the GPU (equal to round-off in float64); adjoint gradients with Mach > 0
  and a new gradient input `"mach"` (per unit Mach); the lifting lines and ground effect refuse
  Mach > 0. The trust envelope stays at Mach 0.3 until
  validation; a warning is given above Mach 0.7.
- Cross-tool speed harness (`validation/speed/`): times each tool and Ventorum at equal
  accuracy (mesh converged to 0.5 % in CL and CDi) on the same cases, black box only. Adapters
  for AVL, VSPAERO, AeroSandbox, MachUpX and Ventorum, each at the settings of its public
  documentation; Flow5 is skipped (no documented batch mode).
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
  and are removed from every output. A task stops with `loop:repeat_call` when the model sends
  a successful call again, and a tool call that the endpoint cannot parse counts as a format
  failure (no retry, not excluded).
- Dispatch record: every lattice-solver result has `details["dispatch"]` with the device, the
  precision, the kernel backend per kernel, the threads, the workers and the source of the
  parallel plan that ran, the tuning profile in use, and one sentence with the reason of the
  device choice (for example the GPU cost model, the work threshold, a CPU worker thread, or a
  nonlinear case solved again on the CPU). GPU results give None for threads, workers and plan
  source. `SolverResult.to_dict` and `IntegratedResult.to_dict` give the record with
  `detail_level="full"`, and so do the agent tools (not `batch_evaluate`). The default agent
  output does not change. The record adds about 1 % to a small GPU sweep (more garbage
  collection) and no measurable time on the CPU.
- OpenMDAO components (optional extra `mdo`, `openmdao>=3.30,<4`): `ventorum.mdo.VentorumComponent`
  wraps one analysis with analytic partials from the adjoint (`gradients`), and
  `VentorumMatrixFreeComponent` adds `strip_force` [N] and `section_cl` with products from
  `gradient_vjp`. Inputs are the flow values, the node displacements and the control
  deflections, with OpenMDAO units; device and precision follow the global settings.
  `import ventorum` does not need OpenMDAO. Example 12 minimises CD at fixed CL with twist.

### Changed

- The near-field loads of the vortex lattice evaluate each shared trailing-edge wake ray once
  (the rays of one strip that start at the same point merge, with their circulations summed).
  A 33-case VLM sweep is about 7 % faster, a wing-tail sweep about 6 %. Results equal to
  round-off; one round-off line of the verification report moves (7.0e-16 to 6.2e-16).
- The numpy reference vortex kernel is 13 % to 24 % faster (fewer temporary arrays; the
  outside law of the finite-filament core only where it is needed). Results equal to
  round-off; the PyTorch kernel is unchanged.
- Machine profile schema 3 (by addition; a schema-2 profile still loads): `ventorum-tune` also
  writes the GPU limits `min_work` and `min_cases` per solver family (from the measured
  crossover), the kernel threads of one case inside a multi-case batch per size class
  (`batch_threads`), and the chunk sizes of the CPU ground batch and the gradient partials
  (`chunk_mib`). `VENTORUM_GRAD_CHUNK` still wins over the profile. `--quick` takes about 30 s
  on the test machine. No result changes; see the performance limits (PL-6).
- Vortex core and leg forces (result change, approved by the owner): the core is applied on the
  distance to the finite filament in every kernel family (numpy, Numba, Cython, PyTorch, Warp,
  and the gradient kernels). The leg force of a strip edge follows the kind of edge: legs on a
  shared edge of one surface carry no net force; at a kink (from the geometry definition:
  dihedral change, mirrored-root dihedral, angle between joined surfaces) the leg velocity is
  projected on the kink line; free tip edges use the full velocity. Legs of joined edges are
  matched with the join tolerance. This removes a side force that grew with the mesh at
  junctions (a fin on a tail tip: CY_beta from -0.46 to -2.46 between 10 and 80 panels before,
  within 0.2 % now) and at winglets. `stability_derivatives` reports the trust and warnings of
  all its cases and compares CY_beta with the Trefftz-plane value. Joined edges whose chordwise
  grids do not match are not joined, with a warning. Verification case V10 changes (reason in
  the commit). The Numba and Cython kernels are faster; the numpy and PyTorch CPU kernels are
  slower (see the performance limits, PL-5).
- `input_error_bars` uses the adjoint for the nonlinear lifting line and tabulated airfoils;
  central differences stay only for the Fourier solver. Section inputs (gradients and
  uncertainties) are refused on tabulated airfoils for every solver.

### Fixed

- The tuned single-case thread count could make single VLM solves 13 to 20 % slower than all
  cores: the tuner kept the fastest count of a short timing with no margin, and a tie rule
  favours fewer threads. The tuner now times each count through the profile reader with at least
  5 repeats, keeps a value only when it beats all cores by more than 5 %, and the self-check of a
  full run also times two fixed single solves and drops the single entries when either is more
  than 2 % slower.
- The tuned `batch_threads` setting could halve the CPU sweep throughput: the tuner timed an
  8-case linear lifting-line batch and applied the result to every solver family. The tuner now
  times a 33-case batch of each family (vlm, linear, nonlinear) on the sweep path, keeps a value
  only when it beats all cores by more than 5 %, and a full run ends with a self-check that drops
  the values if a fixed sweep is more than 2 % slower. The profile stores
  `batch_threads[family][size_class]`; the old flat form still loads and is ignored (all cores).
  Run `ventorum-tune` again to replace an old entry.

- The lift-slope clamp of the gradient reference (section lift slope between 0.2 and 2 times
  2 pi) has the same derivative convention for all torch versions: 1 on the closed interval,
  0 outside it. Before, torch 2.14 gave 0 at the bounds.
- Adjoint gradients with 24 or more cosine chordwise panels no longer fail the forward check:
  the chordwise panel fractions are now recovered without round-off growth.
- The mesh-convergence study (`ventorum_mesh_convergence`) gave a recommended mesh that could
  change from run to run: the measured solve time decided between levels of equal panel count.
  The selection and the efficiency score now use a deterministic cost (panel counts; the score
  uses the panels solved squared). Measured times stay in the output as information.
- The gradient of a tabulated airfoil at the ends of its table was 0 with newer PyTorch
  versions (the derivative of `torch.clamp` at a bound is not documented and changed). The
  reference model now gives the end slope with every PyTorch version.

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
