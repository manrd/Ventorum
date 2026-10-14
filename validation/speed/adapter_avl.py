# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""AVL adapter of the speed harness (black box).

The adapter reuses the AVL comparison harness of this repository
(``validation.avl``): it writes AVL input files, runs ``avl.exe``
once per angle, and reads CL from ``CLtot`` and CDi from ``CDff``.
No AVL source code is read. The program path comes from the
``--avl`` option or the ``VENTORUM_AVL_EXE`` variable, as in the
AVL harness. A missing program gives a skipped result with a note.
"""

from __future__ import annotations

import time
from pathlib import Path

from ventorum.core.datatypes import SolverSettings

from validation.speed.base import SweepResult, median_seconds, scrub_home, skipped_result
from validation.speed.base import find_tool as _find_base
from validation.speed.cases import N_CHORD

#: Tool name in the reports.
TOOL_NAME = "avl"

#: Environment variables that hold the AVL program path.
ENV_VARS = ("VENTORUM_AVL_EXE",)

#: AVL supports ground effect through the geometry-rotation method.
SUPPORTS_GROUND_EFFECT = True

#: Spanwise meshes of the convergence study.
MESHES = [20, 40]


def find_tool(explicit: str | None = None) -> Path | None:
    """Locate ``avl.exe`` (explicit path first, then ``VENTORUM_AVL_EXE``)."""
    return _find_base(explicit, ENV_VARS)


def is_available(explicit: str | None = None) -> bool:
    """Return True when the AVL program file exists."""
    return find_tool(explicit) is not None


def describe_mesh(mesh: int) -> str:
    """Return a human readable mesh label."""
    return f"n_panels {mesh}, n_chord {N_CHORD}"


def _base_settings(mesh: int) -> SolverSettings:
    """Return the mesh settings of one AVL sweep."""
    return SolverSettings(solver_type="vlm", n_panels=int(mesh), n_chord=N_CHORD)


def run_sweep(avl_exe: str | Path | None, aircraft, mesh: int,
              angles_deg: list[float], workdir: str | Path | None,
              timeout: float = 60.0) -> SweepResult:
    """Run one AVL sweep (one process per angle) and time it.

    Parameters
    ----------
    avl_exe : str, Path or None
        Path of ``avl.exe``. None or a wrong path gives a skipped
        or failed result, never an exception.
    aircraft : Aircraft
        Case geometry.
    mesh : int
        Spanwise panels per semi-span.
    angles_deg : list of float
        Sweep angles [deg].
    workdir : str, Path or None
        Folder for the run files (one subfolder per angle). None
        uses a temporary folder that is discarded.
    timeout : float
        Wall-time limit [s] of one AVL run.

    Returns
    -------
    SweepResult
        CL [-] from ``CLtot`` and CDi [-] from ``CDff`` per angle,
        with the wall time [s] of the whole sweep.
    """
    from validation.avl.avl_files import build_case
    from validation.avl.avl_run import run_case

    angles = [float(a) for a in angles_deg]
    case_id = getattr(aircraft, "name", "")
    exe = Path(avl_exe) if avl_exe is not None else find_tool(None)
    if exe is None or not exe.is_file():
        return skipped_result(TOOL_NAME, case_id,
                              "AVL not found: set VENTORUM_AVL_EXE or pass --avl PATH.")
    import tempfile

    base_settings = _base_settings(mesh)
    cl: list[float | None] = []
    cdi: list[float | None] = []
    first_error: str | None = None
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ventorum_speed_avl_") as tmp:
        root = Path(workdir) if workdir is not None else Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        for alpha in angles:
            sub = root / f"a{alpha:+07.2f}".replace(".", "p")
            try:
                case = build_case(aircraft, base_settings, alpha_deg=alpha)
            except ValueError as exc:
                first_error = first_error or scrub_home(exc)
                cl.append(None)
                cdi.append(None)
                continue
            result = run_case(exe, case, sub, timeout=float(timeout))
            if result.ok and result.forces is not None:
                cl.append(float(result.forces["CLtot"]))
                cdi.append(float(result.forces["CDff"]))
            else:
                first_error = first_error or scrub_home(result.error)
                cl.append(None)
                cdi.append(None)
    elapsed = time.perf_counter() - t0
    if any(v is None for v in cl) or any(v is None for v in cdi):
        out = SweepResult(
            tool=TOOL_NAME, case_id=case_id, mesh_label=describe_mesh(mesh),
            mesh=int(mesh), angles_deg=angles, CL=cl, CDi=cdi,
            ok=False, skipped=False,
            reason=f"AVL sweep has failed points: {first_error}",
            elapsed_s=elapsed,
        )
        return out
    return SweepResult(
        tool=TOOL_NAME, case_id=case_id, mesh_label=describe_mesh(mesh),
        mesh=int(mesh), angles_deg=angles, CL=cl, CDi=cdi,
        ok=True, skipped=False, reason=None, elapsed_s=elapsed,
    )


def time_sweeps(avl_exe: str | Path | None, aircraft, mesh: int,
                angles_deg: list[float], repeats: int, timeout: float = 60.0,
                workdir: str | Path | None = None) -> tuple[float, list[float], int]:
    """Time *repeats* AVL sweeps. Return (median [s], samples [s], failed points)."""
    import tempfile

    samples: list[float] = []
    failed = 0
    with tempfile.TemporaryDirectory(prefix="ventorum_speed_avltime_") as tmp:
        root = Path(workdir) if workdir is not None else Path(tmp)
        for rep in range(int(repeats)):
            res = run_sweep(avl_exe, aircraft, mesh, list(angles_deg),
                            root / f"rep{rep:02d}", timeout)
            samples.append(res.elapsed_s)
            failed += sum(1 for v in res.CL if v is None)
    return median_seconds(samples), samples, failed
