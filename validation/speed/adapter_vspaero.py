# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""VSPAERO adapter of the speed harness (black box, pending mapping).

VSPAERO is the vortex-lattice program of OpenVSP. The harness reads
its path from ``--vspaero`` or the ``VENTORUM_VSPAERO`` variable and
the OpenVSP Python interpreter from ``VENTORUM_VSP_PYTHON`` (that
interpreter is separate, so it is called as a subprocess). A missing
program gives a skipped result.

A present program still gives a skipped result until the owner
confirms the documented input mapping (geometry format, mesh
settings and output files). This keeps the harness honest: it
never times a run whose settings are not the documented best
settings of the tool. No VSPAERO source code is read.
"""

from __future__ import annotations

from pathlib import Path

from validation.speed.base import SweepResult, skipped_result
from validation.speed.base import find_tool as _find_base

#: Tool name in the reports.
TOOL_NAME = "vspaero"

#: Environment variables that hold the program paths.
ENV_VARS = ("VENTORUM_VSPAERO", "VENTORUM_VSP_PYTHON")

#: Ground effect runs start only after the mapping check.
SUPPORTS_GROUND_EFFECT = False

#: Planned spanwise meshes of the convergence study.
MESHES = [20, 40, 80]


def find_tool(explicit: str | None = None) -> Path | None:
    """Locate the VSPAERO program (explicit path first, then the variables)."""
    return _find_base(explicit, ENV_VARS)


def is_available(explicit: str | None = None) -> bool:
    """Return True when a VSPAERO program file exists."""
    return find_tool(explicit) is not None


def describe_mesh(mesh: int) -> str:
    """Return a human readable mesh label."""
    return f"mesh {mesh}"


def run_sweep(vspaero_exe: str | Path | None, aircraft, mesh: int,
              angles_deg: list[float], workdir: str | Path | None,
              timeout: float = 120.0) -> SweepResult:
    """Report a skipped VSPAERO sweep (mapping check is pending).

    Parameters
    ----------
    vspaero_exe : str, Path or None
        Path of ``vspaero.exe``.
    aircraft : Aircraft
        Case geometry (unused until the mapping is confirmed).
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
    exe = Path(vspaero_exe) if vspaero_exe is not None else find_tool(None)
    if exe is None or not Path(exe).is_file():
        return skipped_result(TOOL_NAME, case_id,
                              "VSPAERO not found: set VENTORUM_VSPAERO or pass --vspaero PATH.")
    out = skipped_result(
        TOOL_NAME, case_id,
        "VSPAERO is present but its input mapping is pending a user documentation "
        "check; no timed run until the owner confirms the documented best settings.",
    )
    out.mesh = int(mesh)
    out.mesh_label = describe_mesh(mesh)
    out.angles_deg = [float(a) for a in angles_deg]
    return out


def time_sweeps(vspaero_exe: str | Path | None, aircraft, mesh: int,
                angles_deg: list[float], repeats: int, timeout: float = 120.0,
                workdir: str | Path | None = None) -> tuple[float, list[float], int]:
    """Return empty timing (the tool is skipped until the mapping is confirmed)."""
    return 0.0, [], 0
