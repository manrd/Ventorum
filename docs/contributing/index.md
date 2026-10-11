# Contributing

```{toctree}
:maxdepth: 1

documentation
agent_tasks
```

## GPU checks at a phase gate

CI has no GPU, and no self-hosted runner is used while the repository is
public. So run the GPU checks by hand at each phase gate on a machine
with an NVIDIA GPU:

```bash
python validation/gpu_gate.py <log.md> [--float64] [--example-timeout SECONDS] [--allow-no-gpu]
```

The script runs all GPU tests (`pytest -m gpu`) and every
`examples/[0-9]*.py` with `VENTORUM_DEVICE=gpu`, then writes a short
Markdown log. `--float64` runs the examples a second time with
`VENTORUM_GPU_PRECISION=float64`. `--example-timeout` sets the timeout
[s] of each example (default 600). Exit codes: 0 when every step passes,
1 when a step fails or times out, 2 when there is no GPU. Keep the log
with the gate record. Do not commit the log when it holds results that
are not public.
