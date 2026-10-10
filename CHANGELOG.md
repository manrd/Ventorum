# Changelog

All notable changes to Ventorum are in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before version 1.0.0,
a minor version can change a public interface.

## [Unreleased]

### Changed

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

### Fixed

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
