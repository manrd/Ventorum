# Changelog

All notable changes to Ventorum are in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before version 1.0.0,
a minor version can change a public interface.

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
