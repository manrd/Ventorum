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
| `VENTORUM_CONFIG_DIR` | Each profile lookup (`ventorum.hardware.profile`) | Folder of the machine profile. Without it: `%LOCALAPPDATA%\Ventorum` on Windows, `~/Library/Application Support/Ventorum` on macOS, `$XDG_CONFIG_HOME/ventorum` or `~/.config/ventorum` elsewhere. Print it with `ventorum-tune path`. |
| `VENTORUM_DISABLE_AUTOTUNE` | Each profile lookup (`ventorum.hardware.profile`) | Set to `1` (also `true`, `yes`, `on`) to ignore the profile and use the built-in defaults. |
| `VENTORUM_AGENT_AUDIT_LOG` | Each agent tool call (`ventorum.agent.dispatcher`) | File path of the agent audit log. One JSON line per call is appended. Without it nothing is written. |
| `LOCALAPPDATA` | Each profile lookup on Windows (`ventorum.hardware.profile`) | Base of the default profile folder on Windows. This is a platform variable, not an Ventorum setting. |
| `XDG_CONFIG_HOME` | Each profile lookup outside Windows and macOS (`ventorum.hardware.profile`) | Base of the default profile folder on Linux and other systems. This is a platform variable, not an Ventorum setting. |
| `VENTORUM_LIBOMP_PREFIX` | Build time (`setup.py`, macOS only) | Folder of the libomp installation used to build the Cython kernels with OpenMP. Falls back to `brew --prefix libomp`, then to the default Homebrew paths. |
| `VENTORUM_MACOS_OPENMP` | Build time (`setup.py`, macOS only) | Set to `1` to keep the libomp build of the Cython kernels for experiments in a process without PyTorch. The default macOS build has no OpenMP and runs the Cython kernels on Python threads (see [Known performance limits](performance_limits)). |

## Use

Set a run-time variable before the process starts (for example `VENTORUM_KERNEL=torch`). The forced kernel backend can also be set inside Python with `set_kernel_backend` (see [Force a kernel backend](how_to_backends)). Set a build-time variable before the build (for example before `pip install -e .`).
