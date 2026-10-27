# Environment variables

This page is the reference of every environment variable that the code reads. One table holds all of them. The command below prints the code lines that this table documents:

```bash
grep -rn "os.environ" ventorum | grep -v legacy
```

(plus two build-time variables of `setup.py`, marked as such).

| Variable | Read at | Effect |
| --- | --- | --- |
| `VENTORUM_KERNEL` | Start of `ventorum.aero.vortex` | Start value of the kernel backend: `auto` (default), `numba`, `numpy`, `cython` or `torch`. Same as `set_kernel_backend`. Unknown values fall back to `auto`. |
| `VENTORUM_TORCH_DEVICE` | Each torch call (`ventorum.aero.vortex_torch`) | Torch device: `cpu`, `cuda` or `cuda:N`. Default is `cuda` when a CUDA GPU is available, else `cpu`. Apple `mps` is not used (no float64). |
| `VENTORUM_TORCH_CHUNK` | Each torch call (`ventorum.aero.vortex_torch`) | Target number of float64 values in the largest temporary of one torch chunk (default `2.0e7`). The results do not depend on it. |
| `VENTORUM_DEVICE` | Start of `ventorum.gpu` | Where the lattice solvers run: `auto` (default: the GPU for large solves when it is available), `cpu` or `gpu`. Same as `ventorum.gpu.set_device`. Unknown values fall back to `auto`. See [Run solves on the GPU](how_to_gpu). |
| `VENTORUM_GPU_PRECISION` | Start of `ventorum.gpu` | Arithmetic of the GPU pipelines: `auto` (default, float32), `float32` or `float64`. Same as `ventorum.gpu.set_precision`. |
| `VENTORUM_CONFIG_DIR` | Each profile lookup (`ventorum.hardware.profile`) | Folder of the machine profile. Without it: `%LOCALAPPDATA%\Ventorum` on Windows, `~/Library/Application Support/Ventorum` on macOS, `$XDG_CONFIG_HOME/ventorum` or `~/.config/ventorum` elsewhere. Print it with `ventorum-tune path`. |
| `VENTORUM_DISABLE_AUTOTUNE` | Each profile lookup (`ventorum.hardware.profile`) | Set to `1` (also `true`, `yes`, `on`) to ignore the profile and use the built-in defaults. |
| `VENTORUM_AGENT_AUDIT_LOG` | Each agent tool call (`ventorum.agent.dispatcher`) | File path of the agent audit log. One JSON line per call is appended. Without it nothing is written. |
| `VENTORUM_CACHE` | Each new `PolarCache` without a folder (`ventorum.aero.polar_cache`) | Base folder of the disk cache of section polars; the cache uses its subfolder `polars`. Without it: `~/.cache/ventorum/polars`. The XFOIL and Ventorum-2D polar sources use this cache when they get no cache of their own. |
| `VENTORUM_XFOIL` | Each XFOIL start (`ventorum.aero.xfoil_runner.resolve_xfoil_executable`) | Name or path of the XFOIL executable. An explicit executable argument wins. Without it: the `xfoil` program on the `PATH`. |
| `VENTORUM_GRAD_CHUNK` | Each chunked kernel product of the gradient reference (`ventorum.sensitivity.reference.chunk_budget`) | Chunk budget of the adjoint gradient kernels, in values (one value takes 8 bytes). A positive number wins over the tuned value of the machine profile and the built-in default; other values are ignored. The results do not depend on it. The tuner sets it for its own measurements and restores it after. |
| `VENTORUM_DEV_UNSAFE_FLOAT32_PANEL_KERNELS` | Each entry of `ventorum.panel3d.kernels.unsafe_float32_kernels` | Developer path only, not a user setting. The value `1`, together with the context manager `unsafe_float32_kernels`, lets the 3D panel kernels run with float32 inputs or arithmetic, with a warning. Float32 kernels give wrong influences near long panels; without this variable the kernels refuse float32. |
| `LOCALAPPDATA` | Each profile lookup on Windows (`ventorum.hardware.profile`) | Base of the default profile folder on Windows. This is a platform variable, not an Ventorum setting. |
| `XDG_CONFIG_HOME` | Each profile lookup outside Windows and macOS (`ventorum.hardware.profile`) | Base of the default profile folder on Linux and other systems. This is a platform variable, not an Ventorum setting. |
| `VENTORUM_LIBOMP_PREFIX` | Build time (`setup.py`, macOS only) | Folder of the libomp installation used to build the Cython kernels with OpenMP. Falls back to `brew --prefix libomp`, then to the default Homebrew paths. |
| `VENTORUM_MACOS_OPENMP` | Build time (`setup.py`, macOS only) | Set to `1` to keep the libomp build of the Cython kernels for experiments in a process without PyTorch. The default macOS build has no OpenMP and runs the Cython kernels on Python threads (see [Known performance limits](performance_limits)). |

## Use

Set a run-time variable before the process starts (for example `VENTORUM_KERNEL=torch`). The forced kernel backend can also be set inside Python with `set_kernel_backend` (see [Force a kernel backend](how_to_backends)). Set a build-time variable before the build (for example before `pip install -e .`).
