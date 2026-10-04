# Installation

Ventorum needs Python 3.11 or later. From the root of the repository:

```bash
pip install -e .
```

| Extra | Command | Adds |
| --- | --- | --- |
| Tables | `pip install -e .[tables]` | `pandas`, for table export |
| Development | `pip install -e .[dev]` | `pytest`, `pytest-cov`, `ruff` |
| Documentation | `pip install -e .[docs]` | `sphinx`, `pydata-sphinx-theme`, `myst-parser` |
| GPU | `pip install -e .[gpu]` | `warp-lang` (NVIDIA Warp, Apache-2.0), for the GPU pipelines (`ventorum.gpu`) |

The required dependencies are `numpy`, `scipy`, `matplotlib`, `numba` (compiled CPU kernels) and `torch` (GPU kernels and the GPU pipelines). The GPU pipelines also need the `gpu` extra, an NVIDIA GPU with a CUDA driver and the CUDA build of PyTorch (see [Run solves on the GPU](how_to_gpu)). On a machine with no GPU you can install the smaller CPU build of PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The installation also compiles the Cython kernels when a C compiler is present. Without a compiler the installation continues, and Ventorum uses the Numba and numpy kernels. XFOIL is optional: Ventorum can call it to make section polars (`ventorum.run_xfoil`) when it is installed.

## After installation

Run the tuner once. It measures the machine and stores the best thread settings for it (about half a minute):

```bash
ventorum-tune
```

Ventorum also works without this step, with defaults that suit any machine. Run it again after a hardware change. See [Parallel execution and tuning](parallel.md).
