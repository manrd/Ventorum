# Run a sweep in parallel

Independent cases run side by side with no manual tuning. Ventorum selects the workers and the threads per worker from the tuned profile, else from a default by case size. The product never exceeds the core count. A case inside a worker never starts a second pool.

## Angle-of-attack sweep

```python
results = vt.analyze_sweep(
    wing,
    alpha_deg_range=alphas_deg,   # angles in degrees
    V_inf=50.0,
    solver="horseshoe",
    n_panels=40,
    n_jobs="auto",                # "auto" (default), an integer, or -1 (one case per core)
    backend="auto",               # "auto" (threads), "thread" or "serial"
)
```

`n_jobs="auto"` uses the tuned batch plan or the default. Give an integer to fix the number of cases in parallel; the kernel threads of each case are then the cores divided by that number. The nonlinear solver always runs in sequence.

The lower-level call is `vt.alpha_sweep(aircraft, condition, settings, alpha_range)` with `alpha_range` in radians. It takes the same `n_jobs` and `backend` arguments.

## Ground-effect grid

`GroundEffectSweep` runs grids of heights, angles of attack and bank angles with one chordwise mesh for all cases. Its `n_workers` argument works like `n_jobs` above. See [Run a ground-effect study](how_to_ground_effect).

## Independent instances

`run_parallel_instances` runs several independent `Ventorum` stateful cases at once (see `examples/08_multi_instance_parallel.py`):

```python
from ventorum.instance import run_parallel_instances

done = run_parallel_instances(
    instances,
    max_concurrent_instances=None,  # default: at most one per core
    instance_backend="thread",      # "thread" (default) or "serial"; no process pool exists
)
```

Each instance runs its own cases inside its worker with the worker share of the cores. A failing instance records its error and the others still complete, unless `raise_on_error=True`.

## Background

For the two levels and the thread policy (`solve_threads`, BLAS on one thread), see [Parallel execution and tuning](parallel.md). To measure the machine first, see [Tune the machine](how_to_tune).
