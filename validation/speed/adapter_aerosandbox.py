# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""AeroSandbox adapter of the speed harness (black box, pending API check).

AeroSandbox is a Python package. It always runs in a subprocess
(one case script per case set) so that its import time and its
threads do not mix with the Ventorum process. The interpreter of
that subprocess comes from ``--aerosandbox-python`` or the
``VENTORUM_AEROSANDBOX_PYTHON`` variable. Only the documented
public API is used; no source file of the package is opened.

A missing interpreter gives a skipped result. A present
interpreter still gives a skipped result until the owner confirms
the documented API calls of the solve script. The helper
:func:`run_json_script` of :mod:`validation.speed.base` runs the
script and reads its numbers.
"""

from __future__ import annotations

from pathlib import Path

from validation.speed.base import SweepResult, skipped_result
from validation.speed.base import find_tool as _find_base

#: Tool name in the reports.
TOOL_NAME = "aerosandbox"

#: Environment variable that holds the subprocess interpreter.
ENV_VARS = ("VENTORUM_AEROSANDBOX_PYTHON",)

#: Ground effect runs start only after the API check.
SUPPORTS_GROUND_EFFECT = False

#: Planned meshes of the convergence study.
MESHES = [20, 40, 80]


def find_tool(explicit: str | None = None) -> Path | None:
    """Locate the subprocess interpreter (explicit path first, then the variable)."""
    return _find_base(explicit, ENV_VARS)


def is_available(explicit: str | None = None) -> bool:
    """Return True when the subprocess interpreter file exists."""
    return find_tool(explicit) is not None


def describe_mesh(mesh: int) -> str:
    """Return a human readable mesh label."""
    return f"mesh {mesh}"


def run_sweep(python_exe: str | Path | None, aircraft, mesh: int,
              angles_deg: list[float], workdir: str | Path | None,
              timeout: float = 300.0) -> SweepResult:
    """Report a skipped AeroSandbox sweep (API check is pending).

    Parameters
    ----------
    python_exe : str, Path or None
        Interpreter that would run the case script in a subprocess.
    aircraft : Aircraft
        Case geometry (unused until the API is confirmed).
    mesh : int
        Mesh key (kept in the record).
    angles_deg : list of float
        Sweep angles [deg] (kept in the record).
    workdir : str, Path or None
        Unused (kept so all adapters share one call shape).
    timeout : float
        Unused (kept so all adapters share one call shape).

    Returns
    -------
    SweepResult
        Always skipped, with the reason in the record.
    """
    case_id = getattr(aircraft, "name", "")
    exe = Path(python_exe) if python_exe is not None else find_tool(None)
    if exe is None or not Path(exe).is_file():
        return skipped_result(
            TOOL_NAME, case_id,
            "AeroSandbox interpreter not found: set VENTORUM_AEROSANDBOX_PYTHON "
            "or pass --aerosandbox-python PATH.",
        )
    out = skipped_result(
        TOOL_NAME, case_id,
        "AeroSandbox interpreter is present but its solve script is pending "
        "a documented public API check; no timed run until the owner confirms it.",
    )
    out.mesh = int(mesh)
    out.mesh_label = describe_mesh(mesh)
    out.angles_deg = [float(a) for a in angles_deg]
    return out


def time_sweeps(python_exe: str | Path | None, aircraft, mesh: int,
                angles_deg: list[float], repeats: int, timeout: float = 300.0,
                workdir: str | Path | None = None) -> tuple[float, list[float], int]:
    """Return empty timing (the tool is skipped until the API is confirmed)."""
    return 0.0, [], 0
