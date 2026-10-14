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

**The VJP kernels of the adjoint gradients.** The Cython VJP extension
(`ventorum.aero.vortex_vjp_cython`) has the same build rule and the same
fallback: without OpenMP its rows run in blocks on Python threads, and the
source derivatives of the blocks are added in block order
(`test_cython_python_threads_equal_openmp` and
`test_cython_python_threads_bitwise_equal_openmp` in `tests/test_vjp_kernels.py`).
The Numba VJP kernels use `prange`, so they probably have the same
OpenMP problem as the Numba solve kernels. Neither is tested on macOS yet.
On the CPU the gradients use Numba by default; when Numba fails on a Mac, select the
Cython or numpy VJP backend (`ventorum.sensitivity.kernels.set_vjp_backend`).

## PL-2: Nonlinear sweeps past the maximum lift gain little on the GPU

**Cause.** A nonlinear lifting-line sweep with continuation starts each
case from the converged circulation of the case before it. The GPU solves
all cases together, and it repeats the solves until no case changes its
root (at most three passes, see `ventorum/gpu/pipeline.py`). Past the
maximum lift, roots can change from pass to pass, and some cases do not
converge in the Newton iterations that the GPU gives them. The cases from
the first such case to the end of the sweep are then solved on the CPU, in
order, with the restarts of the CPU solver.

**What Ventorum does now.** The results are those of the CPU sweep. The
cases before the first unsettled case come from the GPU.

**Cost.** The cases past the maximum lift are the expensive cases of such
a sweep (many Newton iterations and restarts), so the sweep takes about
the time of the CPU sweep. A sweep that ends before the maximum lift is
not affected.

**How to remove the limit.** Options to study:

1. Run the restarts of one case on the GPU as one batch of start values.
2. Solve the unsettled cases on the GPU one after the other. The gain is
   small for medium lattices: one Newton iteration of one case costs about
   the same on the GPU as on the CPU there.

## PL-3: Host work limits the GPU throughput of small cases

**Cause.** A GPU solve has a fixed cost on the host: the Python code of the
pipeline, the launch of each kernel and the transfer of the results. Each
case of a batch also gets its result objects (loads, spanwise results,
trust score) from Python code on the host, and the garbage collector of
Python scans the many objects that PyTorch, NVIDIA Warp and SciPy create
when they are imported.

**What Ventorum does now.** The `"auto"` device keeps small solves on the
CPU (`ventorum.gpu.use_gpu`, see [Run solves on the GPU](how_to_gpu)).
The per-case inputs of a batch go to the GPU in one transfer, and one
kernel each makes the per-case data.

**Cost.** A single solve of a small lattice is faster on the CPU. For a
large batch of small lattices, the host work, not the GPU, sets the number
of cases per second. To see the times on a machine, run `ventorum-tune`:
step 4 prints the CPU and GPU times per solver family, lattice and batch
size.

**How to remove the limit.** Options to study:

1. CUDA graphs for the fixed sequence of kernels of a batch.
2. Result objects that keep the arrays of the batch and make the objects
   of a case only when they are read (a change of the interface: owner
   decision).

## PL-4: Batched LU of small systems on the GPU

**Cause.** The direct solves of a GPU batch use `torch.linalg` (LU with
partial pivoting). For a batch of matrices, PyTorch selects its library
(cuSOLVER or MAGMA) by itself. On some GPUs, cuSOLVER is faster for small
systems and MAGMA for large ones. Only a setting for the whole process
selects the library, so Ventorum does not change it.

**What Ventorum does now.** It uses the choice of PyTorch. The nonlinear
lifting line factorises the Jacobian of each case at each Newton iteration
(`engine.newton_solve`). The linear solves of larger systems use the
inverse of a reference matrix and corrections (`engine.batched_solve`),
not one factorisation per case.

**Cost.** The Newton iterations of small and medium lattices take longer
than they could.

**How to remove the limit.** A batched LU kernel for small systems in the
GPU pipelines, or a choice of the library per call if PyTorch offers one.

## PL-5: The numpy and PyTorch CPU kernels pay for the finite-filament core

**Cause.** The vortex core is applied on the distance to the finite
filament (the closest point of the segment), not on the distance to its
infinite line. Each segment evaluation needs the end-point projections and
one more branch. The compiled kernels (Numba, Cython) do this in the inner
loop at almost no cost. The numpy reference kernel and the PyTorch kernel
on the CPU work on whole arrays, so they compute and store the extra
arrays for every segment and target.

**What Ventorum does now.** The tuner selects the backend per kernel and
case size. On a 12-core machine it selects Numba or Cython for the
influence matrix at all the measured sizes, so the default path does not
use the slower kernels.

**Cost.** Measured on a 12-core Xeon E5 v3 (influence matrix of a
rectangular wing, no kernel cache, median of 5): the numpy kernel takes 21 %
to 32 % longer and the PyTorch CPU kernel 26 % to 34 % longer than with the
core on the infinite line. The Numba and Cython kernels are 14 % to 21 %
faster than before (fewer operations after the restructuring). The fixed
timing workloads of `validation/perf_history.py` did not become slower.

**How to remove the limit.** Fuse the projection and the core factor into
one pass of the numpy and PyTorch kernels (fewer temporary arrays), and
measure again with the tuner.

## PL-6: The untuned GPU thresholds keep small solves on the CPU

**Cause.** Without a tuned cost model (no machine profile, or a solve whose
panel count is not known), the `"auto"` device uses one fixed work threshold
and one fixed case threshold per solver family (`DEFAULT_MIN_WORK`,
`DEFAULT_MIN_CASES` in `ventorum/gpu/__init__.py`). One value per family
cannot fit every GPU: the thresholds stay conservative, so small solves and
small batches run on the CPU even on GPUs that would win them.

**What Ventorum does now.** `ventorum-tune` measures the crossover per family
on the machine from its GPU step, without extra timing: the smallest work
where the GPU median beats the CPU median at 1 case (`min_work`), and the
smallest winning case count from the 1 and 33 case timings at the smallest
measured lattice (`min_cases`), stored in the profile (schema 3). The cost
model decides whenever the panel count is known; the thresholds only decide
when it is not known.

**Cost.** A solve without a known panel count can run on the CPU while the
GPU would be faster. To see the crossovers on a machine, run
`ventorum-tune`: its step 6 prints them.

**How to remove the limit.** Options to study:

1. Resolve the panel count before every device decision, so the cost model
   always decides (a change of the call structure: owner decision).
2. Thresholds per lattice size class instead of one per family.
