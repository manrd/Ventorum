# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""MachUpX adapter of the speed harness (black box).

MachUpX is a Python package. It always runs in a subprocess (one case
script per case set) so that its import time and its threads do not
mix with the Ventorum process. The interpreter of that subprocess
comes from ``--machupx-python`` or the ``VENTORUM_MACHUPX_PYTHON``
variable. Only the documented public API is used; no source file of
the package is opened. The case script is
:mod:`validation.speed.run_machupx_case`; the case description it
reads is built by :mod:`validation.speed.case_spec`.

Settings, each from the public MachUpX documentation (readthedocs
``Input Files`` and ``User Interface`` pages plus the published
``examples/Traditional`` example):

* Scene from a dict input with SI units, solved through the
  documented ``Scene`` class (``solve_forces``); one scene per angle.
* Linear solver for the speed comparison (stated in the record; the
  documentation recommends the nonlinear solver for accuracy).
* Default flat-plate airfoil (thin airfoil theory, no profile drag),
  so the drag coefficient is the induced drag, read as ``CDi``.
* Cosine clustering of the grid (documented default); mesh
  refinement by the documented grid count ``N`` per semispan.
* Secondary surfaces connect to the main wing root; the body x axis
  points forward (tail offset behind the wing is negative, as in the
  published example).

A missing interpreter gives a skipped result. The timed region covers
only the solve loop; the import time is measured separately, reported
in ``result.json``, and stated in the harness README.
"""

from __future__ import annotations

import json
from pathlib import Path

from validation.speed import case_spec
from validation.speed.base import SweepResult, failed_result, median_seconds, skipped_result
from validation.speed.base import find_tool as _find_base
from validation.speed.base import run_json_script, scrub_home

#: Tool name in the reports.
TOOL_NAME = "machupx"

#: Environment variable that holds the subprocess interpreter.
ENV_VARS = ("VENTORUM_MACHUPX_PYTHON",)

#: No ground effect runs (free air only).
SUPPORTS_GROUND_EFFECT = False

#: Grid counts per semispan of the convergence study.
MESHES = [20, 40, 80]

#: Case script (runs in the subprocess, next to this file).
HELPER = Path(__file__).resolve().parent / "run_machupx_case.py"


def find_tool(explicit: str | None = None) -> Path | None:
    """Locate the subprocess interpreter (explicit path first, then the variable)."""
    return _find_base(explicit, ENV_VARS)


def is_available(explicit: str | None = None) -> bool:
    """Return True when the subprocess interpreter file exists."""
    return find_tool(explicit) is not None


def describe_mesh(mesh: int) -> str:
    """Return a human readable mesh label."""
    return f"N {int(mesh)} cosine"


def run_sweep(python_exe: str | Path | None, aircraft, mesh: int,
              angles_deg: list[float], workdir: str | Path | None,
              timeout: float = 300.0) -> SweepResult:
    """Run one MachUpX sweep in a subprocess and time the solve loop.

    Parameters
    ----------
    python_exe : str, Path or None
        Interpreter that runs the case script in a subprocess.
    aircraft : Aircraft
        Case geometry.
    mesh : int
        Grid count ``N`` per semispan of every lifting surface.
    angles_deg : list of float
        Sweep angles [deg].
    workdir : str, Path or None
        Folder for the run files (``case.json`` in, ``result.json``
        and the scene files out; all stay there as the settings
        record). None uses a temporary folder that is discarded.
    timeout : float
        Wall-time limit [s] of the case script.

    Returns
    -------
    SweepResult
        CL [-] and CDi [-] per angle with the solve-loop wall time [s]
        (import time excluded and reported in ``result.json``).
    """
    import tempfile

    case_id = getattr(aircraft, "name", "")
    angles = [float(a) for a in angles_deg]
    exe = Path(python_exe) if python_exe is not None else find_tool(None)
    if exe is None or not Path(exe).is_file():
        return skipped_result(
            TOOL_NAME, case_id,
            "MachUpX interpreter not found: set VENTORUM_MACHUPX_PYTHON "
            "or pass --machupx-python PATH.",
        )
    try:
        spec = case_spec.aircraft_spec(aircraft, mesh, angles)
    except ValueError as exc:
        return failed_result(TOOL_NAME, case_id, describe_mesh(mesh), angles, scrub_home(exc))
    with tempfile.TemporaryDirectory(prefix="ventorum_speed_mx_") as tmp:
        root = Path(workdir) if workdir is not None else Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        (root / "case.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        try:
            payload = run_json_script(exe, HELPER, root, float(timeout))
        except RuntimeError as exc:
            return failed_result(TOOL_NAME, case_id, describe_mesh(mesh), angles, scrub_home(exc))
    try:
        cl, cdi, version, elapsed = case_spec.check_result_payload(payload, angles)
    except (ValueError, TypeError) as exc:
        return failed_result(TOOL_NAME, case_id, describe_mesh(mesh), angles, scrub_home(exc))
    return SweepResult(
        tool=TOOL_NAME, case_id=case_id,
        mesh_label=f"{describe_mesh(mesh)} linear, MachUpX {version}",
        mesh=int(mesh), angles_deg=angles, CL=cl, CDi=cdi,
        ok=True, skipped=False, reason=None, elapsed_s=elapsed,
    )


def time_sweeps(python_exe: str | Path | None, aircraft, mesh: int,
                angles_deg: list[float], repeats: int, timeout: float = 300.0,
                workdir: str | Path | None = None) -> tuple[float, list[float], int]:
    """Time *repeats* MachUpX sweeps. Return (median [s], samples [s], failed points)."""
    import tempfile

    samples: list[float] = []
    failed = 0
    with tempfile.TemporaryDirectory(prefix="ventorum_speed_mxtime_") as tmp:
        root = Path(workdir) if workdir is not None else Path(tmp)
        for rep in range(int(repeats)):
            res = run_sweep(python_exe, aircraft, mesh, list(angles_deg),
                            root / f"rep{rep:02d}", timeout)
            samples.append(res.elapsed_s)
            failed += sum(1 for v in res.CL if v is None)
            if not res.ok:
                failed += len(res.angles_deg) - sum(1 for v in res.CL if v is None)
    return median_seconds(samples), samples, failed
