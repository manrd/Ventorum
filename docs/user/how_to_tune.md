# Tune the machine

Run the tuner once after installation, and again after a hardware change:

```bash
ventorum-tune            # full run; or: python -m ventorum.hardware tune
ventorum-tune --quick    # skips the large cases
ventorum-tune show       # prints the stored profile and says if it matches this machine
ventorum-tune path       # prints the profile file path
```

## What the tuner measures

The tuner (`ventorum.hardware.tuner`) measures in this order:

1. For each size class (small, medium, large unless `--quick`), the fastest kernel backend for each kernel function (`tensor`, `influence`, `induced`, `trefftz`). Within 3 % of the fastest, Numba wins, then Cython, then PyTorch, then numpy.
2. With those backends active, the best thread count for one case that runs alone. An entry is written only when its median time beats all cores by more than 5 % (median of at least 5 repeats); else single solves use all cores.
3. With those backends active, the best split of the cores for batches (cases in parallel against threads per case).
3b. With those backends active, for each solver family (`vlm`, `linear`, `nonlinear`) and size class, the kernel threads of one case inside a 33-case batch on the sweep path (`solve_batch` with the CPU path). A cell is written (`batch_threads[family][size_class]`) only when its median time beats all cores by more than 5 % (median of 3 repeats); else the sweeps use all cores. A profile with the old flat form (per size class, without the family) still loads; the reader ignores that form and uses all cores.
4. With a CUDA GPU and the GPU pipelines (see [Run solves on the GPU](how_to_gpu)), the cost model of the `"auto"` device: for the vortex lattice and the linear and nonlinear lifting lines, at a few panel counts, the times of sweeps of a few batch sizes on the CPU (with the settings of steps 1 to 3) and on the GPU.
5. A full run ends with a self-check. It times the fixed single solves `single_vlm_n20_c4` and `single_vlm_n80_c4` with and without the single entries, and the fixed sweeps `sweep_vlm_n40_c4_33` and `sweep_linear_n80_33` with and without the batch entries. If either single workload is slower with the entries by more than 2 %, it drops all single entries and prints why; if either sweep workload is slower by more than 2 %, it drops all batch entries.

It also records the GPUs that it finds and the torch device of the run.

## The profile

The tuner writes a JSON profile (schema 3; a schema-2 profile still loads) to the user configuration folder. The folder is `VENTORUM_CONFIG_DIR` when set, else the platform folder (see [Environment variables](environment)). The profile stores the kernel choice per size class and kernel, the raw times, the thread plan, the single-case threads (only entries that beat all cores by more than 5 %), the threads of one case inside a batch per solver family and size class (`batch_threads[family][size_class]`, only cells that beat all cores by more than 5 %), the GPU limits, the chunk sizes, the Cython thread mode (`openmp`, `python` or none) and the torch device. A profile with the old flat `batch_threads` form (per size class, without the family) still loads; the reader ignores that form and uses all cores.

A profile with an older schema is ignored; run `ventorum-tune` again. A profile of another machine is ignored (fingerprint check). Ventorum never runs the tuner by itself and never depends on the profile: without a matching profile it uses built-in defaults that work on any machine.

## Effect on results

Tuning changes the speed; the device choice can change results at the float32 round-off level (about 1e-7 to 1e-6 relative). All backends agree with the numpy reference to round-off. Check the active profile with `ventorum-tune show`. Disable it for one run with `VENTORUM_DISABLE_AUTOTUNE=1`.

## Background

For the backend precedence and the platform defaults, see [Force a kernel backend](how_to_backends). For the two levels of parallelism, see [Parallel execution and tuning](parallel.md). For the architecture, see the [software description](../design/index).
