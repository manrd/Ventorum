# User manual

This manual tells you how to install Ventorum, build a geometry, run an analysis and read the results. For the equations behind each solver, see the [theory manual](../theory/index). For the architecture, see the [software description](../design/index).

## Tutorials

Start here. These pages walk through small complete examples.

```{toctree}
:maxdepth: 1

installation
quickstart
tutorials
```

## How-to guides

Task-oriented guides. Each guide solves one practical task.

```{toctree}
:maxdepth: 1

how_to_tune
how_to_backends
how_to_gpu
how_to_sweeps
how_to_ground_effect
```

## Reference

Exact facts about the interface.

```{toctree}
:maxdepth: 1

geometry
conventions
solvers
ground_effect
results
parallel
environment
performance_limits
examples
```

## Explanation

Background that explains why Ventorum works this way.

- [Theory manual](../theory/index): equations, assumptions, limits and references of each model.
- [Software description](../design/index): architecture, kernel backends, tuner, parallelism and legacy package.
- [Conventions](conventions): axes, units, angles and moment systems.
- [Results and trust](results.md): what a `SolverResult` holds and how far to trust it.
