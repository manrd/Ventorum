# Recovery audit of the original code

This audit gives a recovery decision for every component of `ventorum/legacy` (task T-0015). The owner's rule applies: unverified code is code to verify, not code to delete. **Nothing is deleted.** A "leave out" decision needs the owner's approval, and even then the code stays in `legacy/` until the owner says otherwise. The owner approved all decisions of this audit on 2026-10-02.

Decisions:

- **restore**: put the legacy component back as it is.
- **port**: bring its method onto the verified core, with a parity test against the core reference.
- **reimplement**: write the capability again on the verified core, from the published method; the legacy code is a guide only.
- **leave out**: the verified core already does the same work (it is superseded), or the component has a physics defect. A reason is given.
- **done**: already ported by an accepted task.

The speed numbers come from `validation/legacy_speed.py`. They are in the section "Measured speed" at the end, with the machine and the date.

## Summary

The present code is **slower** than the original code in most single solves of the same model. The first claim ("the original code is 25 to 800 times faster than the first rewrite") was measured against the first rewrite, and the core has changed since then. The measurement on this machine (80 spanwise panels, see the tables below) gives:

- Fourier lifting line: the core is about 150 times slower for one solve and about 40 times slower for a sweep. This is the largest gap, and it is in a solver with the same model.
- Linear lifting line: the core is about 3 times slower.
- VLM with one chordwise panel against the legacy horseshoe solver: the core is about 5 times slower (the legacy solver has known physics defects, so this compares different results).
- Influence-matrix kernel on the same straight horseshoes: the legacy Numba kernel is about 6 times faster than the core Numba kernel (N = 640 and 1600).
- Batched alpha sweep on the GPU (legacy, PyTorch on CUDA) against the core CPU sweep: the legacy sweep is about 9 times faster for 33 angles. One GPU solve is slower than one CPU solve.
- The legacy GPU nonlinear solver is 10 times slower than the core CPU nonlinear solver for one solve and equal for a sweep of 33 angles (and it has the half-lift defect).
- Dense linear solve: LAPACK `dgesv` through SciPy is 12 times faster than `numpy.linalg.solve` at N = 160 and slower above N = 1600 on this machine.

The audit makes these gaps into cards: T-0025 (Fourier speed), T-0026 (kernel speed), T-0027 (batched sweeps on the GPU), T-0028 (batched GPU nonlinear lifting line, study first), T-0029 (solve overhead of the lattice solvers), T-0030 (benchmark suite) and T-0031 (geometry transform utilities). The GPU kernels themselves are in T-0016 (open).

The audit also found that several legacy modules **no longer ran**: they imported `ventorum.analyze` (now the core entry point) and passed legacy data classes to it. This was a defect of the move into `legacy/`, not of the original code. It is fixed (see "Legacy modules that did not run").

## Decisions per module

### Solvers (`legacy/solvers`)

| Module | What it does | Verification state | Present equivalent | Speed (legacy / core, see tables) | Decision | Reason | Card |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `base.py` | Abstract solver interface | Used by every legacy solver | `solvers/base.py`, `solvers/lattice_base.py` | none | leave out | Superseded | none |
| `fourier.py` | Glauert Fourier-series lifting line, multi right-hand-side sweep | Test: CL within 1 % of the core (`test_legacy_linear_and_fourier_agree_with_the_verified_core[fourier]`); sweep not tested; ignores `incidence` | `solvers/fourier.py` (adds Cm, incidence, exact profile drag, refusals) | Legacy about 150 times faster (one solve), 40 times (sweep) | port (speed) | Same model; the core spends its time outside the Fourier system | T-0025 |
| `horseshoe.py` | One horseshoe per strip, 3/4-chord point, symmetry, ground images, precomputed sweep cache | Tests: path parity to 1e-10; strict xfail: ignores camber, over-predicts CDi | `solvers/horseshoe.py` (full VLM, `n_chord` panels) | Legacy about 5 times faster than the core VLM with one chordwise panel | leave out (model); port (speed ideas) | Physics defects; the core VLM is the verified model. The speed gap goes to T-0029 | T-0029 |
| `linear.py` | Phillips and Snyder numerical lifting line, symmetry, precomputed sweep cache | Test: CL within 1 % of the core; sweep not tested | `solvers/linear.py` (same equations) | Legacy about 3 times faster | port (speed) | Same model, slower core | T-0029 |
| `nonlinear.py` | Relaxation iteration on tabulated polars, compiled loop (Numba, Cython) | No test; known defect (docstring only): about half the lift | `solvers/nonlinear.py` (Newton method, quarter-chord formulation) | Legacy about 5 times faster, with the wrong result | leave out | Physics defect; the core uses a different, verified iteration | none |
| `gpu_horseshoe.py` | Horseshoe model in PyTorch; batched alpha sweep (chunks of 2000 cases) | Test only on a CUDA machine (without CUDA the test compares the CPU solver with itself); repeats the horseshoe defects | none | Legacy GPU sweep about 9 times faster than the core CPU sweep; one GPU solve slower | port (batching) | Batched sweeps are the main GPU benefit for LLT and VLM; the model is ported onto the verified VLM, not the defective horseshoe | T-0016 (kernels), T-0027 (batched sweeps) |
| `gpu_nonlinear.py` | Batched relaxation on the GPU, per-case relaxation factor and convergence mask | No test; half-lift defect; uses `alpha + twist` (ignores dihedral) | none | Legacy 10 times slower for one solve, equal for a sweep | reimplement (study first) | The batching idea is useful for large sweeps; the iteration must be the verified Newton method | T-0028 |

### Aerodynamic kernels (`legacy/aero`)

| Module | What it does | Verification state | Present equivalent | Speed | Decision | Reason | Card |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `acceleration.py` | Global backend switch (numpy, Numba, Cython, hybrid) | Test: path parity; defects: the context restores the resolved mode, global state is not thread-safe | `aero/vortex.py` backend per kernel and size class, from the machine profile (T-0002) | none | leave out | Superseded by the tuned choice per kernel | none |
| `biot_savart.py` | Scalar Biot-Savart functions | No test | `aero/vortex.py` `segment_velocity`, `ray_velocity` (vectorised) | none | leave out | Superseded | none |
| `cython_accel.py` | Loader and wrappers of the legacy Cython kernels | Test: path parity | `aero/vortex.py` `_cy_call` | see `cython_kernels.pyx` | done | Ported in T-0001 | T-0001 |
| `cython_kernels.pyx` | Single-thread C loops: velocity matrix, AIC, relaxation loop | Test: path parity to 1e-10 | `aero/vortex_cython.pyx` (OpenMP) | Core Cython faster (it is multi-threaded) | done | Ported in T-0001; the relaxation loop is not needed (Newton method) | T-0001 |
| `numba_kernels.py` | Numba velocity matrix and AIC, relaxation loop | Test: path parity (matrix only) | `aero/vortex_numba.py` | Legacy kernel about 6 times faster on the same straight horseshoes | port (speed) | Find why the legacy kernel is faster (fewer segments, loop order) and bring it to the core, with a parity test | T-0026 |
| `gpu_influence.py` | PyTorch AIC, batch dimension, symmetry and ground images on the device, `TensorPolarTable`, device geometry cache | Test only on CUDA (see `gpu_horseshoe.py`); `TensorPolarTable` not tested (division by zero for repeated angles); no chunking (memory) | none | Legacy torch on CUDA 2 to 3 times faster than the core Numba kernel for one matrix | port | GPU use is a core principle | T-0016, T-0027, T-0028 |
| `influence.py` | NumPy AIC with symmetry and ground images; geometry caches for sweeps | Indirect tests; camber defect in the right-hand side; straight legs tilted with alpha | `aero/vortex.py`, `aero/system.py` (unknown folding, ground plane, `kernel_cache`) | Legacy numpy about as fast as core numpy | leave out | Superseded; defects | none |
| `forces.py` | Strip loads, integration, trust score | Indirect tests; probable site of the CDi defect (near-field drag at the 3/4-chord point) | `aero/loads.py` (Trefftz-plane CDi, symmetric fold) | part of the solve times | leave out | Superseded; defect | none |
| `polars.py` | Polar files, blend | No test; Re parser and sort defects | `aero/polars.py` (defects fixed) | none | leave out | Superseded | none |
| `xfoil_runner.py` | Runs XFOIL | No test; no time-out | `aero/xfoil_runner.py` (time-out, tested) | none | leave out | Superseded | none |

### Core types (`legacy/core`)

| Module | What it does | Verification state | Present equivalent | Speed | Decision | Reason | Card |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `config.py` | Aircraft JSON input and output | No test; `dict_to_aircraft` does not read the output of `aircraft_to_json` | `core/config.py` | none | leave out | Superseded; defect | none |
| `constants.py` | ISA values, vortex core radius, defaults | Indirect | `core/constants.py` | none | leave out | Superseded | none |
| `datatypes.py` | Data classes | Indirect | `core/datatypes.py` | none | leave out | Superseded. Its `control_deflections` field is an idea for the control-surface work of the next phase | none |
| `symmetry.py` | Half-mesh solve and mirror reconstruction, with a reason string | Indirect only | `aero/system.py` `make_unknown_map`, `aero/loads.py` `_symmetric_fold` | none | leave out | Superseded: the core folds the unknowns and the force evaluation. The card "symmetry folding of the force evaluation" was removed (owner, 2026-10-02); folding of mirrored pairs stays in T-0029 | none |
| `trust.py` | Heuristic trust score | Indirect | `core/trust.py` | none | leave out | Superseded | none |

### Geometry (`legacy/geometry`)

| Module | What it does | Verification state | Present equivalent | Speed | Decision | Reason | Card |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `discretization.py` | Spanwise spacing rules, panel counts | Indirect; control point at the mean of the edges (error falls only as 1/n) | `geometry/discretization.py` (Multhopp points) | none | leave out | Superseded; defect | none |
| `processing.py` | One-row horseshoe mesh, Fourier stations | Indirect; nearest-section airfoil, control-point z ignores twist | `geometry/lattice.py`, `geometry/processing.py` | part of the solve times | leave out | Superseded | none |
| `transform.py` | Rotation matrices, rigid rotation of a mesh about a point, ground clearance, roll strike limit | No test; rotation order in the docstring is wrong; strike search checks one side only | `ground_effect/solver.py` (ground plane tilted in body axes, both sides) | none | reimplement (small) | The ground-effect parts are superseded. A public helper to rotate and move surfaces (pitch, roll, yaw about a point) is useful for deformed geometry and design studies | T-0031 |
| `mesh_convergence.py` | Mesh study with GCI | No test; does not run (calls the core `analyze` with `backend`) | `geometry/mesh_convergence.py` (tested) | none | leave out | Superseded | none |

### Hardware (`legacy/hardware`)

| Module | What it does | Verification state | Present equivalent | Speed | Decision | Reason | Card |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `config.py` | Machine profile file tied to a fingerprint | No test; saves the raw machine GUID and host name; `clear_machine_config()` deletes the file at every search path | `hardware/profile.py` (hash only) | none | leave out | Superseded; privacy defect | none |
| `detector.py` | Hardware scan and fingerprint | No test | `hardware/detector.py` (ported, strips identifiers) | none | done | Ported before T-0001 | none |
| `tuner.py` | CPU against GPU crossover, GPU sweep crossover, worker scaling, thread against process pool, multi-instance split | No test; does not finish (the worker-scaling step calls the core with legacy types); GPU crossover defaults to the GPU when the GPU never wins | `hardware/tuner.py` (backend per kernel and size class) | none | port (GPU measurements) | The GPU crossover and batch size belong in the tuner once the GPU path exists | T-0018 (study), T-0027 |

### Utilities and entry points

| Module | What it does | Verification state | Present equivalent | Speed | Decision | Reason | Card |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `utils/benchmark.py` | Timing suites: panel scaling, sweeps, ground effect, threads, multi-instance non-interference, GPU batches, symmetry, mesh study, backend comparison | No test; most suites do not run (core entry points with legacy types). It runs **no other aerodynamic tool**: no executables, no files of other tools, no imports, no copied code (checked line by line) | none (only `validation/bench_kernels.py`) | none | reimplement | A benchmark suite of Ventorum itself, with the multi-instance non-interference check, is part of the benchmark programme. The cross-tool comparison is a separate, black-box harness | T-0030 |
| `utils/linalg.py` | LAPACK `dgesv` solve, GPU dispatch heuristic, worker count | Indirect | `utils/parallel.py` (workers); no `dgesv` wrapper | see "Dense linear solve" | leave out (dgesv); part of T-0029 | Decide with the measured numbers in T-0029; the GPU heuristic is replaced by the tuner | T-0029 |
| `utils/progress.py` | Progress bars | Indirect | `utils/progress.py` | none | leave out | Superseded (`BenchmarkTracker` not needed) | none |
| `utils/validation.py` | Input checks | Indirect; the Fourier check misses sweep given by `x_le` (the core copy has the same defect) | `utils/validation.py` | none | leave out | Superseded. The shared defect is reported to T-0005 (Fourier guard) | T-0005 |
| `drag_polar.py` | Alpha sweep with hardware routing (GPU, process pool), plots | No test; drops `phi` in the workers | `visualization/drag_polar.py`, `utils/parallel.py` | none | leave out (routing goes to T-0027) | Superseded except the GPU routing | T-0027 |
| `instance.py` | Solver instances and case manager | No test; `analyze()` fails (calls the core with `backend`) | `instance.py` (tested) | none | leave out | Superseded; defect | none |
| `__init__.py` files | Re-exports, `legacy.analyze`, `legacy.analyze_sweep`, `ensure_machine_tuned` | `legacy.analyze` used by all legacy tests | `ventorum/__init__.py` | none | keep as they are | Needed to run the legacy code for comparison | none |

## Legacy modules that did not run

These modules imported `analyze` or `analyze_sweep` from `ventorum` (the verified core since the move) and passed legacy data classes to it: `utils/benchmark.py` (most suites), `geometry/mesh_convergence.py`, `hardware/tuner.py` (so the legacy `tune_machine()` could not finish) and `instance.py` (`Ventorum.analyze`). By the owner's decision (2026-10-02) the 12 imports now load the legacy entry points (`ventorum.legacy.analyze`, `analyze_sweep`). `tests/test_legacy_entry_points.py` runs each module on a small case. The "does not run" notes in the tables describe the state before this fix.

## Gaps in the legacy tests

- `test_legacy_gpu_solver_matches_cpu_solver` runs the PyTorch code only on a CUDA machine. Without CUDA, `legacy.analyze(solver="gpu_horseshoe")` runs the CPU solver, and the test compares it with itself. A direct `GPUHorseshoeSolver(device="cpu")` would test the PyTorch path on every machine.
- The nonlinear half-lift defect has no expected-failure test.
- `TensorPolarTable` divides by zero when two table angles are equal.

## Measured speed

The tables below are the output of `python validation/legacy_speed.py` on the owner's machine (Windows, 2026-10-02).

Generated by `validation/legacy_speed.py` (Python 3.12.10, numpy 2.5.2, ventorum 0.2.0, 12 CPU cores, PyTorch present).

### Influence-matrix kernels (straight horseshoes, N vortices at N points)

Median time in ms. Largest difference of any path from the legacy numpy matrix: 1.7e-15 (relative to the largest entry).

| N | path | time [ms] |
|---|---|---|
| 160 | legacy numpy | 7.44 |
| 160 | legacy numba | 0.22 |
| 160 | legacy cython (1 thread) | 1.52 |
| 160 | legacy torch (cuda) | 2.36 |
| 160 | core numpy | 14.71 |
| 160 | core numba | 1.02 |
| 160 | core cython | 1.00 |
| 160 | **best legacy / best core** | 0.22 |
| 640 | legacy numpy | 270.59 |
| 640 | legacy numba | 2.50 |
| 640 | legacy cython (1 thread) | 24.98 |
| 640 | legacy torch (cuda) | 6.32 |
| 640 | core numpy | 299.00 |
| 640 | core numba | 14.93 |
| 640 | core cython | 14.92 |
| 640 | **best legacy / best core** | 0.17 |
| 1600 | legacy numpy | 1878.23 |
| 1600 | legacy numba | 17.64 |
| 1600 | legacy cython (1 thread) | 155.11 |
| 1600 | legacy torch (cuda) | 32.36 |
| 1600 | core numpy | 1953.40 |
| 1600 | core numba | 96.27 |
| 1600 | core cython | 94.40 |
| 1600 | **best legacy / best core** | 0.19 |

### Solves of a rectangular wing (AR 8, 80 spanwise panels, cosine spacing)

One solve at alpha = 5 deg, and an alpha sweep of 33 angles from -4 to 12 deg with the `solve_sweep` method of each solver. Median time in ms. A ratio above 1 means that the core is faster.

| case | CL legacy | CL core | 1 solve legacy | 1 solve core | legacy / core | sweep legacy | sweep core | legacy / core | note |
|---|---|---|---|---|---|---|---|---|---|
| linear lifting line | 0.4232 | 0.4213 | 1.59 | 5.28 | 0.30 | 35.00 | 88.27 | 0.40 | same model |
| Fourier lifting line | 0.4222 | 0.4222 | 0.74 | 111.73 | 0.01 | 3.73 | 144.89 | 0.03 | same model |
| horseshoe (legacy) / VLM, 1 chordwise panel (core) | 0.4117 | 0.3973 | 1.38 | 6.67 | 0.21 | 33.17 | 116.02 | 0.29 | legacy has known defects |
| horseshoe (legacy) / VLM, 4 chordwise panels (core) | 0.4117 | 0.3995 | 1.25 | 39.48 | 0.03 | 33.94 | 678.22 | 0.05 | legacy has known defects |
| nonlinear lifting line (tabulated polar) | 0.2325 | 0.4220 | 2.28 | 11.27 | 0.20 | 72.05 | 291.94 | 0.25 | legacy gives about half the lift |
| GPU horseshoe on cuda (legacy) / VLM, 1 chordwise panel (core) | 0.4117 | 0.3973 | 16.01 | 6.45 | 2.48 | 11.75 | 109.25 | 0.11 | legacy has known defects |
| GPU nonlinear on cuda (legacy) / nonlinear lifting line (core) | 0.2325 | 0.4220 | 119.76 | 11.38 | 10.53 | 301.24 | 290.87 | 1.04 | legacy gives about half the lift |

### Dense linear solve

`legacy.utils.linalg.fast_linear_solve` (SciPy LAPACK dgesv) against `numpy.linalg.solve`, one right-hand side, median time in ms.

| N | dgesv (legacy) | numpy.linalg.solve | numpy / dgesv |
|---|---|---|---|
| 160 | 0.27 | 3.34 | 12.49 |
| 640 | 8.86 | 16.21 | 1.83 |
| 1600 | 122.71 | 81.72 | 0.67 |
| 3200 | 833.64 | 465.92 | 0.56 |

