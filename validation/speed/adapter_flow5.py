# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Flow5 adapter of the speed harness (black box, no batch mode).

Flow5 is used only when its user documentation describes a batch
(no window) mode. The checked documentation describes no such
mode, so the adapter always reports skipped: "no batch mode
documented" when the program is present, and "not found" when it
is missing. The window is never driven. No Flow5 source code
is read.
"""

from __future__ import annotations

from pathlib import Path

from validation.speed.base import SweepResult, skipped_result
from validation.speed.base import find_tool as _find_base

#: Tool name in the reports.
TOOL_NAME = "flow5"

#: Environment variable that holds the Flow5 program path.
ENV_VARS = ("VENTORUM_FLOW5",)

#: Ground effect runs start only after a batch mode is documented.
SUPPORTS_GROUND_EFFECT = False

#: Planned meshes of the convergence study (unused until batch mode exists).
MESHES = [20, 40]


def find_tool(explicit: str | None = None) -> Path | None:
    """Locate the Flow5 program (explicit path first, then the variable)."""
    return _find_base(explicit, ENV_VARS)


def is_available(explicit: str | None = None) -> bool:
    """Return True when a Flow5 program file exists."""
    return find_tool(explicit) is not None


def describe_mesh(mesh: int) -> str:
    """Return a human readable mesh label."""
    return f"mesh {mesh}"


def run_sweep(flow5_exe: str | Path | None, aircraft, mesh: int,
              angles_deg: list[float], workdir: str | Path | None,
              timeout: float = 120.0) -> SweepResult:
    """Report a skipped Flow5 sweep (no batch mode is documented).

    Parameters
    ----------
    flow5_exe : str, Path or None
        Path of ``flow5.exe``.
    aircraft : Aircraft
        Case geometry (unused).
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
    exe = Path(flow5_exe) if flow5_exe is not None else find_tool(None)
    if exe is None or not Path(exe).is_file():
        return skipped_result(TOOL_NAME, case_id,
                              "Flow5 not found: set VENTORUM_FLOW5 or pass --flow5 PATH.")
    out = skipped_result(
        TOOL_NAME, case_id,
        "Flow5 has no batch mode documented in its user documentation; "
        "the harness skips it and never drives its window.",
    )
    out.mesh = int(mesh)
    out.mesh_label = describe_mesh(mesh)
    out.angles_deg = [float(a) for a in angles_deg]
    return out


def time_sweeps(flow5_exe: str | Path | None, aircraft, mesh: int,
                angles_deg: list[float], repeats: int, timeout: float = 120.0,
                workdir: str | Path | None = None) -> tuple[float, list[float], int]:
    """Return empty timing (the tool is skipped without a batch mode)."""
    return 0.0, [], 0
