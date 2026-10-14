# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Shared types and helpers of the cross-tool speed harness.

Each external tool runs as a black box: the harness writes input
files or a case script, starts the tool in a subprocess, and reads
back plain numbers. A missing tool gives a skipped result with the
name of the environment variable that would point to it. A failed
run gives a failed result with the reason, never silent zeros.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Change limit [-] of the equal-accuracy rule (0.5 %).
CONVERGENCE_TOL = 0.005

#: Timing repeats of a full run (medians of 5).
REPEATS_FULL = 5

#: Timing repeats of a quick run.
REPEATS_QUICK = 1

#: Angles of a full sweep (33 points).
FULL_N_ALPHA = 33

#: Name of the JSON file that a Python tool script writes.
RESULT_FILENAME = "result.json"


@dataclass
class SweepResult:
    """Outcome of one tool sweep over a list of angles.

    Attributes
    ----------
    tool : str
        Tool name (for example ``"avl"``).
    case_id : str
        Case identifier (for example ``"rect"``).
    mesh_label : str
        Human readable mesh description.
    mesh : int or None
        Mesh key (spanwise count or tool mesh index); None when skipped.
    angles_deg : list of float
        Sweep angles [deg].
    CL : list
        Lift coefficients [-]; None entries mark failed points.
    CDi : list
        Induced drag coefficients [-]; None entries mark failed points.
    ok : bool
        True only when every point returned values.
    skipped : bool
        True when the tool was not run (missing program or no batch mode).
    reason : str or None
        Skip or failure reason (None when ok).
    elapsed_s : float
        Wall time [s] of the sweep.
    """

    tool: str
    case_id: str = ""
    mesh_label: str = ""
    mesh: int | None = None
    angles_deg: list[float] = field(default_factory=list)
    CL: list[float | None] = field(default_factory=list)
    CDi: list[float | None] = field(default_factory=list)
    ok: bool = False
    skipped: bool = False
    reason: str | None = None
    elapsed_s: float = 0.0


def skipped_result(tool: str, case_id: str, reason: str) -> SweepResult:
    """Return a skipped sweep result with *reason*."""
    return SweepResult(tool=tool, case_id=case_id, ok=False, skipped=True, reason=reason)


def failed_result(tool: str, case_id: str, mesh_label: str,
                  angles_deg: list[float], reason: str) -> SweepResult:
    """Return a failed sweep result; every point reads None."""
    n = len(angles_deg)
    return SweepResult(
        tool=tool, case_id=case_id, mesh_label=mesh_label,
        angles_deg=list(angles_deg), CL=[None] * n, CDi=[None] * n,
        ok=False, skipped=False, reason=reason,
    )


def median_seconds(samples: list[float]) -> float:
    """Return the median of wall-time samples [s]."""
    return float(statistics.median(samples))


def max_relative_change(coarse: list[float | None],
                        fine: list[float | None]) -> float | None:
    """Return the largest change between two sweeps [-].

    The change at each angle is ``|fine - coarse|`` divided by the
    peak absolute value of the fine sweep. The peak scale keeps a
    zero crossing (for example CL near zero lift) from blowing up
    the metric. Points with None on either side are ignored. None
    is returned when no point pair remains.
    """
    pairs = [(c, f) for c, f in zip(coarse, fine)
             if c is not None and f is not None]
    if not pairs:
        return None
    peak = max(abs(f) for _, f in pairs)
    if peak <= 1e-12:
        return 0.0 if all(abs(f - c) <= 1e-12 for c, f in pairs) else None
    return max(abs(f - c) / peak for c, f in pairs)


def convergence_errors(coarse_cl: list[float | None], coarse_cdi: list[float | None],
                       fine_cl: list[float | None],
                       fine_cdi: list[float | None]) -> tuple[float | None, float | None]:
    """Return the CL and CDi refinement changes [-]."""
    return (max_relative_change(coarse_cl, fine_cl),
            max_relative_change(coarse_cdi, fine_cdi))


def has_converged(coarse_cl: list[float | None], coarse_cdi: list[float | None],
                  fine_cl: list[float | None], fine_cdi: list[float | None],
                  tol: float = CONVERGENCE_TOL) -> tuple[bool, float | None, float | None]:
    """Check the equal-accuracy rule on two refined sweeps.

    Returns (converged, cl_change, cdi_change). Both changes must be
    below *tol* [-]. A None change (no comparable points) never passes.
    """
    cl_err, cdi_err = convergence_errors(coarse_cl, coarse_cdi, fine_cl, fine_cdi)
    ok = (cl_err is not None and cdi_err is not None
          and cl_err < tol and cdi_err < tol)
    return ok, cl_err, cdi_err


def find_tool(explicit: str | None, env_names: tuple[str, ...] | list[str]) -> Path | None:
    """Locate an external tool program.

    *explicit* (a command-line path) takes precedence, then each name
    in *env_names* is read from the environment. The first path that
    is an existing file wins. None is returned when neither gives
    a file.
    """
    candidates: list[str | None] = [explicit]
    candidates += [os.environ.get(name) for name in env_names]
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if path.is_file():
                return path
    return None


def refuse_inside_repository(out: Path, root: Path) -> Path:
    """Resolve *out* and refuse it when it lies inside the repository."""
    resolved = out.resolve()
    root_resolved = root.resolve()
    if resolved == root_resolved or root_resolved in resolved.parents:
        raise ValueError(
            f"Refused: --out {out} is inside the repository ({root_resolved}). "
            "Results stay private; give a folder OUTSIDE the repository."
        )
    return resolved


def scrub_home(message: object) -> str:
    """Return *message* with the home folder path replaced by ~."""
    text = str(message)
    home = str(Path.home())
    for form in {home, home.replace("\\", "/")}:
        if form:
            text = text.replace(form, "~")
    return text


def run_json_script(python_exe: str | Path, script: str | Path, workdir: str | Path,
                    timeout: float) -> dict:
    """Run a Python tool script in a subprocess and read its result.

    The script runs with *workdir* as its working folder and writes
    :data:`RESULT_FILENAME` there with ``{"CL": [...], "CDi": [...]}``.
    Running each Python tool in its own process keeps its import time
    and its threads apart from the Ventorum process.

    Parameters
    ----------
    python_exe : str or Path
        Python interpreter that runs the script (for example the
        interpreter of a separate environment).
    script : str or Path
        Path of the case script.
    workdir : str or Path
        Working folder of the run (created as needed).
    timeout : float
        Wall-time limit [s] of the script.

    Returns
    -------
    dict
        Parsed JSON mapping.

    Raises
    ------
    RuntimeError
        If the script fails, times out, or writes no result file.
    """
    folder = Path(workdir)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / RESULT_FILENAME
    if target.exists():
        target.unlink()
    try:
        completed = subprocess.run(
            [str(python_exe), str(script)],
            capture_output=True, text=True, timeout=float(timeout), cwd=folder,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Tool script timed out after {timeout:g} s.") from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot start tool script: {scrub_home(exc)}.") from exc
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "")[-500:]
        raise RuntimeError(f"Tool script failed (code {completed.returncode}): {tail.strip()}")
    if not target.is_file():
        raise RuntimeError("Tool script wrote no result.json; the run is failed, not zero.")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read result.json: {scrub_home(exc)}.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("result.json holds no mapping.")
    return payload


def timed_call(func, *args, **kwargs) -> tuple[object, float]:
    """Run *func* once and return (result, wall time [s])."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - t0


def python_exe() -> str:
    """Return the path of the current Python interpreter."""
    return sys.executable
