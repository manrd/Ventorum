# Force a kernel backend

The four kernel functions (`tensor`, `influence`, `induced`, `trefftz`) can each use another backend. Normally you leave the choice on `"auto"` and let the tuned profile decide per kernel and case size (see [Tune the machine](how_to_tune)). Force one backend only for a test or a comparison.

## Force it in Python

```python
from ventorum.aero import vortex

vortex.set_kernel_backend("cython")   # "auto", "numba", "numpy", "cython" or "torch"
print(vortex.get_kernel_backend())
print(vortex.kernel_backend_for("influence"))  # resolved backend of one kernel
vortex.set_kernel_backend("auto")    # restore the automatic choice
```

`set_kernel_backend` with a value other than `"auto"` applies to all kernels and overrides the profile. It raises `ValueError` when the backend is not available (for example `"cython"` without the compiled extension). With `"auto"`, `get_kernel_backend` returns the platform default, not the profile; `kernel_backend_for` returns the resolved backend of one kernel.

## Force it from the environment

Set `VENTORUM_KERNEL` before the process starts. It accepts the same values (`auto`, `numba`, `numpy`, `cython`, `torch`) and selects the start value. A forced backend that is not available falls back to the default.

## Precedence

The resolver uses this order:

1. Forced backend (`set_kernel_backend` or `VENTORUM_KERNEL`): it wins for all kernels.
2. Profile: with `"auto"` and a matching profile, the entry for the kernel and the size class wins. A missing entry, or one that names an unavailable backend, falls back silently.
3. Platform default: with `"auto"` and no profile, macOS prefers Numba, then Cython, then numpy. Other systems prefer Cython, then Numba, then numpy. The default never selects PyTorch.

The size class comes from the panel count of the solve. One solve can mix backends (hybrid mode, for example Cython for one kernel and Numba for another).

## The PyTorch backend

`VENTORUM_KERNEL=torch` forces the PyTorch kernels (float64 only) for all kernel functions. `VENTORUM_TORCH_DEVICE` selects the device (`cpu`, `cuda` or `cuda:N`; default is `cuda` when a CUDA GPU is available, else `cpu`). Apple `mps` is not used because it has no float64 support. `VENTORUM_TORCH_CHUNK` sets the target number of float64 values in the largest temporary of one chunk; the results do not depend on it. On the CPU each call uses the kernel thread count and restores the previous count after it.

The tuner measures the torch backend where torch is installed and stores the device of the run in the profile. The platform default never selects it; select it in the profile or force it.

## The Cython thread paths

With OpenMP (Linux and Windows builds) the Cython kernels split their rows themselves. Without OpenMP (the default macOS build) Ventorum splits the rows into blocks and runs one kernel call per block on Python threads. Both paths give the same result. See [Known performance limits](performance_limits).

## All variables in one place

The full table of environment variables is in [Environment variables](environment).
