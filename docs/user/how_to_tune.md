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
2. With those backends active, the best thread count for one case that runs alone.
3. With those backends active, the best split of the cores for batches (cases in parallel against threads per case).
4. With a CUDA GPU and the GPU pipelines (see [Run solves on the GPU](how_to_gpu)), the cost model of the `"auto"` device: for the vortex lattice and the linear and nonlinear lifting lines, at a few panel counts, the times of sweeps of a few batch sizes on the CPU (with the settings of steps 1 to 3) and on the GPU.

It also records the GPUs that it finds and the torch device of the run.

## The profile

The tuner writes a JSON profile (schema 2) to the user configuration folder. The folder is `VENTORUM_CONFIG_DIR` when set, else the platform folder (see [Environment variables](environment)). The profile stores the kernel choice per size class and kernel, the raw times, the thread plan, the Cython thread mode (`openmp`, `python` or none) and the torch device.

A profile with an older schema is ignored; run `ventorum-tune` again. A profile of another machine is ignored (fingerprint check). Ventorum never runs the tuner by itself and never depends on the profile: without a matching profile it uses built-in defaults that work on any machine.

## Effect on results

Tuning changes only the speed, never the results. All backends agree with the numpy reference to round-off. Check the active profile with `ventorum-tune show`. Disable it for one run with `VENTORUM_DISABLE_AUTOTUNE=1`.

## Background

For the backend precedence and the platform defaults, see [Force a kernel backend](how_to_backends). For the two levels of parallelism, see [Parallel execution and tuning](parallel.md). For the architecture, see the [software description](../design/index).
