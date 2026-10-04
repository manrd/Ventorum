# Run solves on the GPU

Ventorum can solve the vortex lattice and the linear and nonlinear lifting lines on an NVIDIA CUDA GPU. The GPU pipelines solve a batch of flight conditions on one lattice together (the angles of a sweep, the cases of `solve_batch`), and also single solves of large lattices. The equations, the checks and the result objects are the same as on the CPU; see [Solvers](solvers.md).

## Requirements

* An NVIDIA GPU with a CUDA driver.
* PyTorch with CUDA (a dependency of Ventorum; install the CUDA build of PyTorch).
* NVIDIA Warp: `pip install "ventorum[gpu]"` (or `pip install warp-lang`).

Check the machine:

```python
from ventorum import gpu

print(gpu.available(), gpu.unavailable_reason())
print(gpu.info())
```

Without a GPU, or without these packages, every solve runs on the CPU and nothing changes.

## Select the device

```python
from ventorum import gpu

gpu.set_device("auto")   # default: the GPU for large solves, the CPU for small ones
gpu.set_device("gpu")    # the GPU for every solve that the GPU pipelines support
gpu.set_device("cpu")    # never the GPU
```

The environment variable `VENTORUM_DEVICE` (`auto`, `cpu`, `gpu`) sets the start value. With `"auto"`, the choice depends on the solver family (`"vlm"`, `"linear"` or `"nonlinear"`), the panel count and the number of cases:

* After `ventorum-tune` on a machine with a GPU, the profile holds a cost model of the machine: the times of batches of a few sizes, on the CPU and on the GPU, at a few panel counts. `"auto"` estimates the two times of the solve from the model and selects the faster device.
* Without a cost model, a solve runs on the GPU when its work is at least `gpu.min_work(family)` horseshoe evaluations (the number of control points times the number of panels times the number of cases; twice that in ground effect), or when the batch has at least `gpu.min_cases(family)` cases.

Small solves stay on the CPU: the time to start the GPU work is larger than the gain. A large batch of small cases goes to the GPU: each case has a fixed cost on the CPU. `gpu.info()` shows the active values and if a cost model is present.

## Select the precision

```python
gpu.set_precision("float64")   # the CPU results to round-off
gpu.set_precision("float32")   # faster on most GPUs
gpu.set_precision("auto")      # default: float32
```

The environment variable `VENTORUM_GPU_PRECISION` sets the start value.

* `float64` gives the results of the CPU solvers to round-off (the order of the sums differs).
* `float32` computes the vortex kernels in single precision. The kernels keep the digits that matter in float64 (point differences, the distance of a wake leg to a point, a point on the line of a bound vortex). The linear systems are solved with float64 residuals until the error of the solve is far below the float32 error of the matrix; the Newton steps of the nonlinear lifting line use the float32 factorisation of the Jacobian (its entries have float32 errors of the same size). The relative error of the coefficients is of the order of 1e-7 to 1e-6, small against the errors of the methods. Consumer GPUs have much less float64 than float32 throughput, so `float32` is much faster there.

Each result states its device and precision: `result.details["device"]` is `"gpu"` and `result.details["precision"]` is `"float32"` or `"float64"` for a GPU solve (the keys are absent for a CPU solve).

## What runs on the GPU

| Solver | Batch (sweeps) | Single solve |
| --- | --- | --- |
| Vortex lattice | yes | yes |
| Linear lifting line | yes | yes |
| Nonlinear lifting line | yes | yes |
| Fourier lifting line | no (CPU) | no (CPU) |

All of them in free air and in ground effect, symmetric or with sideslip or bank, with linear or tabulated airfoils. A batch with different kinds of cases (symmetric and not, in and out of ground effect) is split into groups, and each group is one GPU batch; a nonlinear sweep with continuation and mixed cases runs on the CPU.

These pipelines use the GPU batches:

* `solve_sweep` and `solve_batch` of the lattice solvers, and `alpha_sweep` (also `Ventorum.analyze_sweep`);
* the ground-effect sweeps (`GroundEffectSweep.run_sweep`, `sweep_height`, `sweep_alpha`, `sweep_roll`): all cases of the grid in one batch;
* the polar tool of the agent interface;
* a single solve (`solve`, `vt.analyze`) of a large lattice.

A solve inside a pool of CPU workers (cases in parallel) stays on the CPU with the `"auto"` device: the pool was chosen for the CPU.

The nonlinear lifting line on the GPU solves all cases together with the Newton method of the CPU solver. A sweep out of ground effect starts each case from the solution of the case before it (continuation); the GPU repeats the solves until no case changes its root, and the cases that do not settle or do not converge are solved on the CPU in order, with the restarts of the CPU solver. The results are therefore those of the CPU sweep.

## Memory

Large batches are split into chunks of cases, so the GPU memory stays bounded. The results do not depend on the chunks.

## Speed

* A GPU that was idle runs at low clocks for the first part of the next work (on the test machine, about 0.3 s of GPU work). A short solve after an idle time can then take some times longer than the same solve in a series. The tuner measures a GPU in use.
* A nonlinear sweep past the maximum lift gains little on the GPU: the cases that do not settle are solved on the CPU, in order (see above).

The known speed limits of the GPU pipelines (nonlinear sweeps past the maximum lift, the host work of small cases, the batched LU of small systems) are entries PL-2 to PL-4 of [Known performance limits](performance_limits).
