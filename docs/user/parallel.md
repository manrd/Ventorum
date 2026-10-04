# Parallel execution and tuning

Ventorum uses all the cores of the machine with no manual tuning, at two levels:

| Level | What runs in parallel | Who decides |
| --- | --- | --- |
| Inside one case | The compiled (Numba) vortex kernels run on several threads | Ventorum: the tuned thread count of the machine, else all cores |
| Across cases | Independent cases (sweep angles, batch cases, ground-effect grids, instances) run side by side | Ventorum: the tuned plan, else a default by case size |

The product of the cases in parallel and the threads per case never exceeds the number of cores. A case that runs inside a worker never starts a second pool. Inside a solve, the BLAS library runs on one thread, so that it does not compete with the kernel threads.

## Tuning

Run the tuner once after installation, and again after a hardware change:

```bash
ventorum-tune            # about half a minute; or: python -m ventorum.hardware tune
ventorum-tune --quick    # skips the large cases
ventorum-tune show       # prints the stored profile and says if it matches this machine
```

The tuner measures, for small, medium and large cases, the kernel backend for each kernel function (tensor, influence, induced, trefftz), the best thread count for one case, and the best split of the cores for batches. It stores a profile in the user configuration folder (`ventorum-tune path` prints it). The tuner never runs before an ordinary analysis. Ventorum does not depend on it: without a profile, or when the profile belongs to another machine, it uses built-in defaults that work on any machine. The tuning changes only the speed, never the results. The profile also stores `torch_device`, the torch device of the tuning run.

## Kernel backend choice

The four kernel functions (velocity tensor, influence matrix, induced velocity, Trefftz normal wash) do not have to use the same backend. The tuner times each kernel function for each available backend (Numba, Cython, PyTorch, numpy) at each case size, and writes the fastest one to the profile. When backends are within 3 %, Numba wins (it needs no C compiler on reinstall), then Cython, then PyTorch, then numpy. At run time, `"auto"` uses the backend of the profile for the kernel function and the case size of the call. One solve can use two backends: this is the "hybrid" mode.

The backend of a call is resolved in this order:

1. `set_kernel_backend(x)` with `x` other than `"auto"`, or `VENTORUM_KERNEL=x` at start: `x` for all kernels. The profile does not apply.
2. `"auto"` with a matching profile: the profile entry for the kernel and the size class. If the entry is missing, or names a backend that is not available now, the default below is used.
3. `"auto"` without a profile: the default of the platform.

The default of the platform is Numba where it is installed on macOS (the Cython build has no OpenMP there, see [known performance limits](performance_limits)), and Cython where the extension is compiled on all other systems. Without a compiled backend, it is numpy. The default never selects the PyTorch backend: the tuner selects it where it wins, or you force it with `VENTORUM_KERNEL=torch`.

| Environment variable | Effect |
| --- | --- |
| `VENTORUM_CONFIG_DIR` | Folder of the profile |
| `VENTORUM_DISABLE_AUTOTUNE=1` | Ignore the profile; use the defaults |
| `VENTORUM_KERNEL=numpy` | Force the numpy reference kernels for all kernel functions (overrides the profile) |
| `VENTORUM_KERNEL=numba` | Force the compiled Numba kernels for all kernel functions (overrides the profile) |
| `VENTORUM_KERNEL=cython` | Force the compiled Cython kernels for all kernel functions (OpenMP threads; Python threads on macOS, see [known performance limits](performance_limits)) |
| `VENTORUM_KERNEL=torch` | Force the PyTorch kernels for all kernel functions (float64 only; device from `VENTORUM_TORCH_DEVICE`) |
| `VENTORUM_TORCH_DEVICE` | Torch device: `cpu`, `cuda` or `cuda:N`; default is `cuda` when a CUDA GPU is available, else `cpu` (Apple `mps` is not used: it has no float64) |
| `VENTORUM_TORCH_CHUNK` | Target number of float64 values in the largest temporary of one torch chunk (default `2.0e7`); the results do not depend on it |

The full table of all environment variables is in [Environment variables](environment).

## Settings for experts

| Function | Setting | Meaning |
| --- | --- | --- |
| `analyze_sweep`, `alpha_sweep` | `n_jobs` | Vortex lattice: `"auto"` (default): the tuned plan or the default; an integer: that many cases in parallel; `-1`: one case per core. Lifting lines: not used (one batch for all angles) |
| `GroundEffectSweep.run_sweep` | `n_workers` | The same, for the cases of the grid |
| `run_parallel_instances` | `max_concurrent_instances` | Instances in parallel; each runs its own cases inside its worker |

When you give the number of cases in parallel, the kernel threads of each case are the cores divided by that number.

:::{note}
The PyTorch backend runs the verified kernels on the CPU or on a CUDA GPU (select it in the profile or force it with `VENTORUM_KERNEL=torch`). The tuner records the GPUs that it finds and the torch device of its run. The linear solve stays on the CPU.
:::
