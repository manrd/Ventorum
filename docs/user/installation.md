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
| OpenMDAO | `pip install -e .[mdo]` | `openmdao` (Apache-2.0), for the OpenMDAO components (`ventorum.mdo`; see [How to use Ventorum in OpenMDAO](how_to_openmdao)) |
| Study | `pip install -e .[study]` | `jax[cpu]`; no module of the package imports it |

The required dependencies are `numpy`, `scipy`, `matplotlib`, `numba` (compiled CPU kernels), `threadpoolctl` (thread limits of the BLAS and OpenMP pools in the parallel plan) and `torch` (GPU kernels and the GPU pipelines). The GPU pipelines also need the `gpu` extra, an NVIDIA GPU with a CUDA driver and the CUDA build of PyTorch (see [Run solves on the GPU](how_to_gpu)). On a machine with no GPU you can install the smaller CPU build of PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The installation also compiles the Cython kernels when a C compiler is present. Without a compiler the installation continues, and Ventorum uses the Numba and numpy kernels. XFOIL is optional: Ventorum can call it to make section polars (`ventorum.run_xfoil`, `ventorum.aero.polar_source.XfoilPolarSource`) when it is installed.

## XFOIL

Ventorum finds XFOIL in this order: the `executable` argument of a library call, the environment variable `VENTORUM_XFOIL`, then the program `xfoil` on the PATH (`xfoil.exe` on Windows). The agent tools never take an executable from their input; they use only `VENTORUM_XFOIL` or the PATH.

Ventorum runs XFOIL headless on every platform, also on a server without a display:

- Each command stream starts with `PLOP`, `G` and a blank line. This switches the XFOIL graphics off before any other command.
- Each XFOIL process runs in its own new temporary folder. An `xfoil.def` file in your working folder therefore cannot change the start state.
- `G` is a toggle. Ventorum does not assume the start state: it reads the graphics-enable flag that XFOIL prints in the plotting-options menu. If the last printed flag is not `F`, the call stops with an error. If the XFOIL build prints no flag, Ventorum continues and gives one warning.
- On Windows the process starts without a console window. On other platforms Ventorum changes no environment variable (for example `DISPLAY`).

## After installation

Run the tuner once. It measures the machine and stores the best thread settings for it (about half a minute):

```bash
ventorum-tune
```

Ventorum also works without this step, with defaults that suit any machine. Run it again after a hardware change. See [Parallel execution and tuning](parallel.md).
