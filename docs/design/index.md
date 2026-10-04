# Software description

This document describes the architecture of Ventorum: the packages, the data flow, the kernel backends, the tuner, the parallelism and the legacy package.

## Packages

| Package | Responsibility |
| --- | --- |
| `ventorum.core` | Data types (geometry, flight condition, settings, results), constants, errors, trust score, JSON input and output |
| `ventorum.geometry` | Lattice of strips and panels from the surfaces, spanwise spacing, mesh convergence studies |
| `ventorum.aero` | Vortex kernels (horseshoe induced velocity in four backends), system assembly and ground images, loads, section polars, XFOIL |
| `ventorum.solvers` | Solver classes (vortex lattice, linear and nonlinear lifting line, Fourier) and their selection by name |
| `ventorum.hardware` | Hardware detection, machine fingerprint, machine profile, tuner (`ventorum-tune`) |
| `ventorum.utils` | Parallel plan and thread policy (`parallel.py`), validation, JSON helpers |
| `ventorum.ground_effect` | Ground-effect analysis, sweeps, stability derivatives and plots |
| `ventorum.agent` | Tool interface for AI agents: schemas, strict parsing, MCP server, command line |
| `ventorum.visualization` | Plots of geometry, distributions, polars, wake and Trefftz plane |
| `ventorum.reference` | Independent analytic solutions used by the verification |
| `ventorum.instance` | Stateful case objects (`Ventorum`) and parallel runs of several instances |
| `ventorum.legacy` | The original solvers and acceleration paths (Numba, Cython, PyTorch GPU kernels, hardware autotuner), kept to study and port their optimisations. Some of its solvers have known physics defects; see its module docstring and `tests/test_legacy.py`. |

## Data flow of one analysis

1. The user gives an `Aircraft` (or a `LiftingSurface`), a `FlightCondition` and `SolverSettings`.
2. `solvers.factory` selects the solver class from `solver_type`.
3. `geometry.lattice.build_lattice` builds the strips and panels in body axes.
4. `aero.system` places the wake and the ground images, and assembles the influence matrix with `aero.vortex`.
5. The solve runs inside `utils.parallel.solve_threads`, which sets the kernel thread count, limits BLAS to one thread, and stores the panel count for the backend resolver.
6. Each kernel call resolves its backend with `aero.vortex.kernel_backend_for` (one backend per kernel function and size class; see below).
7. The solver solves for the circulation (one linear solve, or Newton iteration for the nonlinear lifting line).
8. `aero.loads` computes the forces and moments, the Trefftz-plane induced drag and the section data.
9. `core.trust` evaluates the trust score, and the solver returns a `SolverResult`.

## Kernel backends and the resolver

The four kernel functions have fixed keys: `"tensor"` (`horseshoe_velocity_tensor`), `"influence"` (`influence_matrix`), `"induced"` (`induced_velocity`) and `"trefftz"` (`trefftz_normalwash`).

Five backends exist: `"numpy"` (the reference kernels in `aero/vortex.py`), `"numba"` (compiled kernels in `aero/vortex_numba.py`), `"cython"` (compiled C kernels in `aero/vortex_cython.pyx`), `"torch"` (PyTorch tensors on CPU or CUDA in `aero/vortex_torch.py`, float64 only) and `"auto"` (select per call, not a kernel).

`kernel_backend_for(kernel, n_panels)` resolves the backend of one call. The precedence is fixed:

1. Forced backend: `set_kernel_backend(x)` with `x` other than `"auto"`, or `VENTORUM_KERNEL=x` at start. It applies to all kernels. The profile does not apply. A forced backend that is not available falls back as `get_kernel_backend` does.
2. Profile: with `"auto"` and a matching machine profile, the entry `settings["kernels"][size_class][kernel]` wins. A missing entry, or one that names a backend that is not available now, falls back to the platform default. It raises no error and writes no warning on each call.
3. Platform default: with `"auto"` and no profile, macOS uses Numba if installed, else Cython if compiled, else numpy. All other systems use Cython if compiled, else Numba if installed, else numpy. The default never selects the PyTorch backend.

The size class comes from `solve_threads(n_panels)` when the call runs inside a solve, else from the source count of the call. One solve can use two backends at once (hybrid mode, for example Cython for the influence matrix and Numba for the tensor). All backends agree with the numpy reference to round-off (see `tests/test_kernels.py`).

## The tuner and the profile schema 2

Run the tuner once after installation, and again after a hardware change:

```bash
ventorum-tune            # full run; or: python -m ventorum.hardware tune
ventorum-tune --quick    # skips the large cases
ventorum-tune show       # prints the stored profile and says if it matches this machine
ventorum-tune path       # prints the profile file path
```

The tuner (`hardware/tuner.py`) measures in this order. First, for each size class (small, medium, large unless `quick=True`), it times each available backend for each of the four kernels directly, with all cores. It skips numpy at medium and large when a compiled backend exists. Within 3 % of the fastest, Numba wins, then Cython, then PyTorch, then numpy. Next, with the selected backends active, it measures the best thread count for one case alone, and the best split of the cores for batches. It also records the GPUs that it finds and the torch device of the run.

The profile (schema 2, `SCHEMA_VERSION = 2`) stores the choices in `settings["kernels"][size_class][kernel]`, the raw times in `measurements["kernels"][size_class][kernel][backend]`, plus `settings["cython_threads"]` (`"openmp"`, `"python"` or `None`), `settings["torch_device"]`, `settings["single"]`, `settings["batch"]` and `settings["gpu"]`. A profile with an older schema is ignored, so users run `ventorum-tune` again. The tuner never runs before an ordinary analysis. Ventorum does not depend on it: without a profile, or when the profile belongs to another machine (fingerprint check), it uses built-in defaults that work on any machine. Tuning changes the speed; the device choice can change results at the float32 round-off level (about 1e-7 to 1e-6 relative).

## Parallelism

Ventorum uses all cores with no manual tuning, at two levels:

| Level | What runs in parallel | Who decides |
| --- | --- | --- |
| Inside one case | The compiled vortex kernels run on several threads | Ventorum: the tuned thread count of the machine, else all cores |
| Across cases | Independent cases (sweep angles, batch cases, ground-effect grids, instances) run side by side | Ventorum: the tuned plan, else a default by case size |

`solve_threads(n_panels)` is the context of one solve: it sets the kernel thread count of the calling thread, limits BLAS to one thread so that the two thread pools do not compete, and stores the panel count for the backend resolver. `plan_parallel` selects workers and threads per worker; the product never exceeds the core count. `case_executor` runs the cases, and each worker keeps its share of the threads. A case that runs inside a worker never starts a second pool. See [Parallel execution and tuning](../user/parallel).

## The Cython thread paths

The Cython kernels take a `num_threads` argument and loop over the evaluation points with `prange`. With OpenMP (Linux and Windows builds), the kernel splits its rows itself. Without OpenMP (the default macOS build), Ventorum splits the evaluation points into row blocks and runs one kernel call per block on Python threads; the kernels release the GIL, so the blocks run in parallel. Each row is computed alone, so both paths give the same result. See [Known performance limits](../user/performance_limits).

## The legacy package

`ventorum.legacy` keeps the original solvers and acceleration paths (Numba, Cython, PyTorch GPU kernels, hardware autotuner). It is kept to study and port its optimisations to the verified core. Some of its solvers have known physics defects; see its module docstring and `tests/test_legacy.py`. Do not use its results until they are corrected. Do not delete anything in it.

## Known limits

Ventorum must use the full speed of each machine. Cases where it does not yet do so are listed in [Known performance limits](../user/performance_limits). Each entry gives the cause, what Ventorum does now, what it costs, and what must happen to remove the limit.

## Design decisions

The design decisions and their reasons are kept in the owner's conceptual design document and are copied here when they become stable.

## Recovery audit

The recovery audit gives a decision for every component of the original code (`ventorum/legacy`).

```{toctree}
:maxdepth: 1

recovery_audit
performance_architecture
```
