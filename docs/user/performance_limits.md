# Known performance limits

Ventorum must use the full speed of each machine. This page lists the known
cases where it does not yet do so. Each entry gives the cause, what Ventorum
does now, what it costs, and what must happen to remove the limit. Remove
an entry only when the limit is gone and a test proves it.

## PL-1: OpenMP clashes with PyTorch on macOS (Cython, probably Numba)

**Cause.** PyTorch, a dependency of Ventorum, ships its own copy of the
OpenMP runtime (`torch/lib/libomp.dylib`). The Cython kernels built with the
Homebrew libomp load a second copy. Two copies in one process stop it:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

The error occurs in both import orders. Linux (GCC libgomp) and Windows
(MSVC vcomp) do not have this problem. The test
`tests/test_kernels.py::test_cython_kernels_run_in_a_process_with_pytorch`
checks it on every operating system in CI.

The Numba kernels probably have the same problem: their OpenMP threading
layer can load a second libomp. This is not confirmed yet; the test
`test_numba_kernels_run_in_a_process_with_pytorch` records it.

macOS problems are handled after the general development (see
`AGENTS.md`). Until then CI does not test macOS. The hand-started workflow
`.github/workflows/macos-debug.yml` runs the macOS checks on request.

**What Ventorum does now.** On macOS, `setup.py` builds the Cython kernels
without OpenMP. Ventorum splits the evaluation points into row blocks and
runs one kernel call per block on Python threads (the kernels release the
GIL). Each row is computed alone, so the results are bit for bit the same
as with OpenMP (`test_cython_python_threads_equal_openmp_threads`). Any
other build without OpenMP uses the same path.

**Cost.** Small cases lose the most: the thread pool adds a fixed cost to
each kernel call. Large cases lose less. To measure it on a machine with an
OpenMP build, run:

```bash
python validation/bench_kernels.py --cython-threads
```

The cost on macOS itself is not measured yet. The measurements and the
work to remove the limit are in
[issue 1](https://github.com/manrd/Ventorum/issues/1).

**How to remove the limit.** Options to study:

1. Use one OpenMP runtime for the whole process: link the extension
   against the libomp that PyTorch loads, if this can be done without a
   dependency on the internal file layout of PyTorch.
2. A lower fixed cost for small cases: fewer, larger blocks, or one
   kernel call that covers several right-hand sides.
3. Let the tuner (`ventorum-tune`) measure the Python-thread path against
   Numba on each Mac; it selects the faster backend per case size.

`VENTORUM_MACOS_OPENMP=1` at build time keeps the libomp build for
experiments in a process that does not load PyTorch. Do not use it for
normal work.
