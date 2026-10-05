# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Run AVL as a black-box program, one case per folder.

Each run writes the ``.avl`` geometry file and the command stream into
its own folder, starts one AVL process with the command stream on its
standard input, and reads back the ``FT`` and ``ST`` output files. The
exit code of AVL is NOT a verdict: a run is good only when its output
files exist and parse completely. A failed run is reported as failed
(never a silent zero, never a crash).
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from validation.avl.avl_files import AvlCase, command_stream
from validation.avl.avl_output import parse_banner, parse_ft, parse_st

#: Name of the geometry file inside a run folder.
AVL_FILENAME = "case.avl"
#: Name of the command stream file inside a run folder.
COMMANDS_FILENAME = "commands.txt"
#: Name of the total-forces file inside a run folder.
FT_FILENAME = "ft.txt"
#: Name of the stability-derivatives file inside a run folder.
ST_FILENAME = "st.txt"
#: Name of the captured standard-output file inside a run folder.
STDOUT_FILENAME = "stdout.txt"

#: Default wall-time limit [s] of one AVL run.
DEFAULT_TIMEOUT_S = 60.0


@dataclass
class AvlRunResult:
    """Outcome of one AVL run.

    Attributes
    ----------
    ok : bool
        True only when the output files exist and parse completely.
    workdir : Path
        Run folder (holds the input and output files).
    returncode : int or None
        Process exit code (None when the process never ran).
    elapsed_s : float
        Wall time [s] of the process (0 when it never ran).
    banner : str or None
        AVL version text from the program banner.
    forces : dict or None
        Parsed ``FT`` file (None when missing or invalid).
    stabderivs : dict or None
        Parsed ``ST`` file (None when missing or invalid).
    error : str or None
        Short failure reason (None when ok).
    """

    ok: bool
    workdir: Path = field(default_factory=lambda: Path("."))
    returncode: int | None = None
    elapsed_s: float = 0.0
    banner: str | None = None
    forces: dict[str, float] | None = None
    stabderivs: dict[str, float] | None = None
    error: str | None = None


def find_avl(explicit: str | None = None) -> Path | None:
    """Locate the AVL program.

    Parameters
    ----------
    explicit : str or None
        Path from the ``--avl`` option (takes precedence).

    Returns
    -------
    Path or None
        Path from ``--avl`` or from the ``VENTORUM_AVL_EXE`` variable,
        or None when neither gives a file.
    """
    for candidate in (explicit, os.environ.get("VENTORUM_AVL_EXE")):
        if candidate:
            path = Path(candidate)
            if path.is_file():
                return path
    return None


def _read_text(path: Path, label: str) -> str:
    """Read *path* or raise ``ValueError`` (a failed run, not a crash)."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"{label} file {path.name} is missing: {exc.strerror or exc}.") from exc


def run_case(
    avl_exe: str | Path | None,
    case: AvlCase,
    workdir: str | Path,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> AvlRunResult:
    """Run one AVL case in *workdir* and parse its outputs.

    Parameters
    ----------
    avl_exe : str, Path or None
        Path of ``avl.exe`` (or ``avl``). A wrong path gives a failed
        result, not an exception.
    case : AvlCase
        Geometry text and run angles from :func:`build_case`.
    workdir : str or Path
        Folder for the input and output files (created as needed; one
        case per folder so parallel runs never share files).
    timeout : float
        Wall-time limit [s] of the AVL process.

    Returns
    -------
    AvlRunResult
        Good only when ``ft.txt`` and ``st.txt`` exist and parse
        completely. The ``error`` field names the reason otherwise.
    """
    folder = Path(workdir)
    started = time.perf_counter()
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return AvlRunResult(ok=False, workdir=folder, error=f"Cannot make folder: {exc}.")
    (folder / AVL_FILENAME).write_text(case.avl_text, encoding="utf-8")
    commands = command_stream(case.run_alpha_deg, case.run_beta_deg, FT_FILENAME, ST_FILENAME)
    (folder / COMMANDS_FILENAME).write_text(commands, encoding="utf-8")

    if avl_exe is None or not Path(avl_exe).is_file():
        return AvlRunResult(ok=False, workdir=folder, error=f"AVL program not found: {avl_exe}.")

    try:
        completed = subprocess.run(
            [str(avl_exe), AVL_FILENAME],
            input=commands,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=folder,
        )
    except subprocess.TimeoutExpired:
        return AvlRunResult(
            ok=False, workdir=folder, elapsed_s=float(timeout),
            error=f"AVL run timed out after {timeout:g} s.",
        )
    except OSError as exc:
        return AvlRunResult(ok=False, workdir=folder, error=f"Cannot start AVL: {exc}.")
    elapsed = time.perf_counter() - started
    stdout = completed.stdout or ""
    (folder / STDOUT_FILENAME).write_text(stdout, encoding="utf-8")

    try:
        banner: str | None = parse_banner(stdout)
    except ValueError:
        banner = None
    try:
        forces = parse_ft(_read_text(folder / FT_FILENAME, "Total-forces"))
        stabderivs = parse_st(_read_text(folder / ST_FILENAME, "Stability-derivatives"))
    except ValueError as exc:
        return AvlRunResult(
            ok=False, workdir=folder, returncode=completed.returncode,
            elapsed_s=elapsed, banner=banner, error=str(exc),
        )
    if banner is None:
        return AvlRunResult(
            ok=False, workdir=folder, returncode=completed.returncode,
            elapsed_s=elapsed, forces=forces, stabderivs=stabderivs,
            error="AVL banner not found in the program output.",
        )
    return AvlRunResult(
        ok=True, workdir=folder, returncode=completed.returncode,
        elapsed_s=elapsed, banner=banner, forces=forces, stabderivs=stabderivs,
    )
