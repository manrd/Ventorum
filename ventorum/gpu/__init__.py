# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""GPU pipelines of the lattice solvers (optional).

The GPU pipelines solve one flight condition, or a batch of flight
conditions on the same lattice (the angles of a sweep), on an NVIDIA CUDA
GPU. All cases of a batch go through each step together:

1. fused vortex kernels (NVIDIA Warp) give the system matrices or the
   velocity tensors of all cases in one launch (:mod:`ventorum.gpu.kernels`);
2. batched dense solves (PyTorch, cuSOLVER) give the circulation;
3. the induced velocities for the loads and the Trefftz-plane normal wash
   come from the same kernels;
4. the post-processing of the loads and the result objects are the same
   code as on the CPU.

The method and the results are those of the CPU solvers (see
:mod:`ventorum.solvers`). Only the arithmetic changes (see *precision*).

Selection
---------
* :func:`set_device` (or the environment variable ``VENTORUM_DEVICE``):
  ``"auto"`` (default) runs a solve on the GPU when the GPU is available and
  the work of the solve is large enough to gain from it; ``"cpu"`` never uses
  the GPU; ``"gpu"`` uses the GPU for every solve that the GPU pipelines
  support.
* :func:`set_precision` (or ``VENTORUM_GPU_PRECISION``): ``"float64"`` gives
  the results of the CPU solvers to round-off; ``"float32"`` is faster on
  most GPUs, and its error is about 1e-7 to 1e-6 relative (the owner
  accepts float32 for the GPU paths). ``"auto"`` is ``"float32"``.

The GPU pipelines need the packages ``torch`` (with CUDA) and ``warp-lang``.
Without them, or without a CUDA GPU, all solves run on the CPU.
"""

from __future__ import annotations

import importlib.util
import os
import threading

DEVICES = ("auto", "cpu", "gpu")
PRECISIONS = ("auto", "float32", "float64")

# Default precision of "auto" on the GPU: float32, which the owner accepts
# for the GPU paths.
AUTO_PRECISION = "float32"

# Built-in rule of the "auto" device, per solver family (the solver names),
# for a machine without a tuned cost model: a solve runs on the GPU when its
# work (horseshoe evaluations of the system, all cases of the batch, see
# ventorum.gpu.pipeline.work_estimate) is at least DEFAULT_MIN_WORK, or when
# the batch has at least DEFAULT_MIN_CASES cases (each case has a fixed cost
# on the CPU, so a large batch gains on the GPU also for a small lattice). A
# vortex-lattice case costs much more on the CPU per unit of work than a
# linear lifting-line case (chordwise panels, three load point sets), and a
# nonlinear case repeats its solves, so their values are lower. The tuner
# replaces the rule with a cost model of the machine (key "gpu", "model").
DEFAULT_MIN_WORK = {"vlm": 5.0e4, "linear": 3.0e5, "nonlinear": 3.0e5}
DEFAULT_MIN_CASES = {"vlm": 24, "linear": 20, "nonlinear": 12}
FAMILIES = ("vlm", "linear", "nonlinear")


def _env(name: str, allowed: tuple[str, ...], default: str) -> str:
    value = os.environ.get(name, default).strip().lower()
    return value if value in allowed else default


_device = _env("VENTORUM_DEVICE", DEVICES, "auto")
_precision = _env("VENTORUM_GPU_PRECISION", PRECISIONS, "auto")
_lock = threading.Lock()
_status: tuple[bool, str | None] | None = None


def _check() -> tuple[bool, str | None]:
    """Return (available, reason) for the GPU pipelines; the test runs once per process."""
    global _status
    with _lock:
        if _status is not None:
            return _status
        reason = None
        if importlib.util.find_spec("torch") is None:
            reason = "PyTorch is not installed."
        elif importlib.util.find_spec("warp") is None:
            reason = "NVIDIA Warp is not installed (pip install warp-lang)."
        else:
            try:
                import torch

                if not torch.cuda.is_available():
                    reason = "PyTorch finds no CUDA GPU."
            except Exception as exc:  # noqa: BLE001 - a broken installation
                reason = f"PyTorch cannot be imported: {exc}"
        if reason is None:
            try:
                import warp as wp

                wp.config.log_level = getattr(wp, "LOG_WARNING", wp.config.log_level)
                wp.init()
                if not wp.is_cuda_available():
                    reason = "NVIDIA Warp finds no CUDA device."
            except Exception as exc:  # noqa: BLE001 - a broken installation
                reason = f"NVIDIA Warp cannot be initialised: {exc}"
        _status = (reason is None, reason)
        return _status


def available() -> bool:
    """Return True if the GPU pipelines can run on this machine."""
    return _check()[0]


def unavailable_reason() -> str | None:
    """Return why the GPU pipelines cannot run, or None if they can."""
    return _check()[1]


def set_device(name: str) -> None:
    """Select where the lattice solvers run: ``"auto"``, ``"cpu"`` or ``"gpu"``.

    Raises
    ------
    ValueError
        If *name* is not known, or if it is ``"gpu"`` and the GPU pipelines
        cannot run (the message gives the reason).
    """
    global _device
    key = str(name).strip().lower()
    if key not in DEVICES:
        raise ValueError(f"Unknown device {name!r}; use one of {DEVICES}.")
    if key == "gpu" and not available():
        raise ValueError(f"The GPU pipelines cannot run: {unavailable_reason()}")
    _device = key


def get_device() -> str:
    """Return the device setting: ``"auto"``, ``"cpu"`` or ``"gpu"``."""
    return _device


def set_precision(name: str) -> None:
    """Select the arithmetic of the GPU pipelines: ``"auto"``, ``"float32"`` or ``"float64"``."""
    global _precision
    key = str(name).strip().lower()
    if key not in PRECISIONS:
        raise ValueError(f"Unknown precision {name!r}; use one of {PRECISIONS}.")
    _precision = key


def get_precision() -> str:
    """Return the precision of the GPU pipelines: ``"float32"`` or ``"float64"`` (``"auto"`` resolved)."""
    if _precision != "auto":
        return _precision
    from ventorum.hardware.profile import tuned_setting

    tuned = tuned_setting("gpu", "precision", default=None)
    return tuned if tuned in ("float32", "float64") else AUTO_PRECISION


def min_work(family: str = "vlm") -> float:
    """Return the smallest work (horseshoe evaluations) for which ``"auto"`` uses the GPU.

    *family* is the solver: ``"vlm"``, ``"linear"`` or ``"nonlinear"``.
    """
    from ventorum.hardware.profile import tuned_setting

    if family not in FAMILIES:
        raise ValueError(f"Unknown solver family {family!r}; use one of {FAMILIES}.")
    tuned = tuned_setting("gpu", "min_work", default=None)
    value = tuned.get(family) if isinstance(tuned, dict) else None
    try:
        return float(value) if value is not None else DEFAULT_MIN_WORK[family]
    except (TypeError, ValueError):
        return DEFAULT_MIN_WORK[family]


def min_cases(family: str = "vlm") -> int:
    """Return the smallest batch (cases) for which ``"auto"`` uses the GPU without a cost model."""
    from ventorum.hardware.profile import tuned_setting

    if family not in FAMILIES:
        raise ValueError(f"Unknown solver family {family!r}; use one of {FAMILIES}.")
    tuned = tuned_setting("gpu", "min_cases", default=None)
    value = tuned.get(family) if isinstance(tuned, dict) else None
    try:
        return int(value) if value is not None else DEFAULT_MIN_CASES[family]
    except (TypeError, ValueError):
        return DEFAULT_MIN_CASES[family]


def _interp_loglog(xs, ys, x: float) -> float:
    """Return the power-law interpolation of the table (xs, ys) at x (end slopes outside the table)."""
    import math

    pts = sorted(zip(xs, ys))
    if len(pts) == 1:
        return float(pts[0][1])
    lx = math.log(max(float(x), 1.0))
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        if lx <= math.log(x1) or i == len(pts) - 2:
            a0, a1 = math.log(x0), math.log(x1)
            b0, b1 = math.log(max(y0, 1e-12)), math.log(max(y1, 1e-12))
            return math.exp(b0 + (b1 - b0) * (lx - a0) / (a1 - a0))
    return float(pts[-1][1])


def _valid_times(table, n_rows: int) -> bool:
    """Return True if *table* is a valid time table of a cost model (see :func:`cost_model`)."""
    if not isinstance(table, dict):
        return False
    ks, ts = table.get("k"), table.get("t")
    if not isinstance(ks, list) or len(ks) < 2 or not isinstance(ts, list) or len(ts) != n_rows:
        return False
    try:
        if int(ks[0]) != 1 or any(float(b) <= float(a) for a, b in zip(ks, ks[1:])):
            return False
        return all(isinstance(row, list) and len(row) == len(ks) and all(float(t) > 0.0 for t in row)
                   for row in ts)
    except (TypeError, ValueError):
        return False


def cost_model(family: str):
    """Return the tuned cost model of *family* for the GPU precision, or None.

    The tuner writes the model to the machine profile (key ``"gpu"``,
    ``"model"``, family, precision): ``"n"`` is a list of panel counts, and
    ``"cpu"`` and ``"gpu"`` each hold ``"k"`` (batch sizes, cases, the first
    one 1) and ``"t"`` (``t[i][j]``: the time [s] of a batch of ``k[j]``
    cases on the lattice of ``n[i]`` panels). An entry that does not have
    this form is ignored.
    """
    from ventorum.hardware.profile import tuned_setting

    model = tuned_setting("gpu", "model", family, get_precision(), default=None)
    if not isinstance(model, dict) or not isinstance(model.get("n"), list) or not model["n"]:
        return None
    try:
        if any(float(n) < 1.0 for n in model["n"]):
            return None
    except (TypeError, ValueError):
        return None
    if not all(_valid_times(model.get(dev), len(model["n"])) for dev in ("cpu", "gpu")):
        return None
    return model


def estimate_time(model: dict, device: str, n_panels: int, n_cases: int) -> float:
    """Return the estimated time [s] of a batch of *n_cases* on a lattice of *n_panels* panels on *device*.

    A power law in the panel count between the panel counts of the model,
    and linear in the cases between its batch sizes. Outside the model, the
    end slopes continue.
    """
    table = model[device]
    ks = [float(k) for k in table["k"]]
    ts = [_interp_loglog(model["n"], [float(row[j]) for row in table["t"]], n_panels) for j in range(len(ks))]
    K = max(float(n_cases), 1.0)
    j = 0
    while j < len(ks) - 2 and K > ks[j + 1]:
        j += 1
    slope = max((ts[j + 1] - ts[j]) / (ks[j + 1] - ks[j]), 0.0)
    return ts[j] + slope * (K - ks[j])


def use_gpu(work: float, family: str = "vlm", n_cases: int = 1, n_panels: int | None = None) -> bool:
    """Return True if a solve of the solver *family* runs on the GPU.

    *work* is the work of the solve (see
    :func:`ventorum.gpu.pipeline.work_estimate`), *n_cases* its number of
    cases and *n_panels* the panel count of its lattice. With the ``"auto"``
    device, a tuned cost model of the machine (:func:`cost_model`) compares
    the estimated times; without one, the built-in rule
    (``DEFAULT_MIN_WORK``, ``DEFAULT_MIN_CASES``) decides.
    """
    if _device == "cpu":
        return False
    if _device == "gpu":
        return available()
    if not available():
        return False
    model = cost_model(family) if n_panels is not None else None
    if model is not None:
        return estimate_time(model, "gpu", n_panels, n_cases) < estimate_time(model, "cpu", n_panels, n_cases)
    return work >= min_work(family) or n_cases >= min_cases(family)


def info() -> dict:
    """Return the GPU settings and the GPU of this machine (for reports and agent tools)."""
    ok, reason = _check()
    out = {"device": _device, "precision": get_precision(), "available": ok, "reason": reason,
           "min_work": {f: min_work(f) for f in FAMILIES}, "min_cases": {f: min_cases(f) for f in FAMILIES},
           "cost_model": {f: cost_model(f) is not None for f in FAMILIES}}
    if ok:
        import torch

        props = torch.cuda.get_device_properties(0)
        out.update(name=props.name, memory_gb=round(props.total_memory / 1024 ** 3, 2),
                   compute_capability=f"{props.major}.{props.minor}")
    return out
