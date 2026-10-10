# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Run AVL as a black-box program, one case per folder.

Each run writes the ``.avl`` geometry file and the command stream into
its own folder, starts one AVL process with the command stream on its
standard input, and reads back the ``FT`` and ``ST`` output files. The
exit code of AVL is NOT a verdict: a run is good only when its output
files exist, parse completely and agree with the requested angles and
mesh counts. Old output files are deleted before each run. A failed
run is reported as failed (never a silent zero, never a crash).
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from validation.avl.avl_files import AvlCase, command_stream, session_command_stream
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

#: Largest accepted difference [deg] between the requested angles and
#: the ``Alpha`` and ``Beta`` that AVL prints in its output files.
ANGLE_TOL_DEG = 1e-4


@dataclass
class AvlRunResult:
    """Outcome of one AVL run.

    Attributes
    ----------
    ok : bool
        True only when the output files exist, parse completely and
        agree with the requested case.
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


def check_parsed(parsed: dict[str, float], case: AvlCase, label: str = "FT") -> None:
    """Check a parsed AVL output against the case that was requested.

    The ``Alpha`` and ``Beta`` [deg] in the file must equal the run
    angles of *case* (absolute tolerance :data:`ANGLE_TOL_DEG`). This
    catches an output file of an earlier run and a command stream that
    got out of step. The surface, strip and vortex counts of the file
    header must equal the counts that the writer expects (when *case*
    gives them). This catches a mesh that AVL changed without a
    message, for example at its fixed array limits.

    Parameters
    ----------
    parsed : dict
        Result of :func:`parse_ft` or :func:`parse_st`.
    case : AvlCase
        The requested case.
    label : str
        Name of the file in the error message.

    Raises
    ------
    ValueError
        If an angle or a count does not agree. The message names the
        value, the requested value and the value in the file.
    """
    for key, wanted in (("Alpha", case.run_alpha_deg), ("Beta", case.run_beta_deg)):
        got = float(parsed[key])
        if abs(got - float(wanted)) > ANGLE_TOL_DEG:
            raise ValueError(
                f"{label} file gives {key} = {got:g} deg, but the run asked for "
                f"{float(wanted):g} deg (stale output or a command stream out of step)."
            )
    for key, wanted in (("n_surfaces", case.n_surfaces), ("n_strips", case.n_strips),
                        ("n_vortices", case.n_vortices)):
        if wanted is None:
            continue
        got = int(round(float(parsed[key])))
        if got != int(wanted):
            raise ValueError(
                f"{label} file gives {key} = {got}, but the geometry file asks for "
                f"{int(wanted)} (AVL changed the mesh, for example at an array limit)."
            )


def _clear_outputs(folder: Path, names: list[str]) -> None:
    """Delete old output files in *folder* before a run.

    AVL asks "File exists. Append/Overwrite/Cancel" when an output file
    exists. That prompt takes the next line of the command stream, so
    the stream gets out of step and the old file stays. A run must
    start without these files.
    """
    for name in names:
        path = folder / name
        if path.exists():
            path.unlink()


def _parse_outputs(folder: Path, case: AvlCase, ft_name: str,
                   st_name: str) -> tuple[dict[str, float], dict[str, float]]:
    """Parse and check the ``FT`` and ``ST`` files of one run case."""
    forces = parse_ft(_read_text(folder / ft_name, "Total-forces"))
    check_parsed(forces, case, "FT")
    stabderivs = parse_st(_read_text(folder / st_name, "Stability-derivatives"))
    check_parsed(stabderivs, case, "ST")
    return forces, stabderivs


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
        case per folder so parallel runs never share files). Old output
        files in it are deleted before the run.
    timeout : float
        Wall-time limit [s] of the AVL process.

    Returns
    -------
    AvlRunResult
        Good only when ``ft.txt`` and ``st.txt`` exist, parse
        completely and agree with the requested angles and mesh counts
        (see :func:`check_parsed`). The ``error`` field names the
        reason otherwise.
    """
    folder = Path(workdir)
    started = time.perf_counter()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        _clear_outputs(folder, [FT_FILENAME, ST_FILENAME, STDOUT_FILENAME])
    except OSError as exc:
        return AvlRunResult(ok=False, workdir=folder, error=f"Cannot prepare folder: {exc}.")
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
        forces, stabderivs = _parse_outputs(folder, case, FT_FILENAME, ST_FILENAME)
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


def run_session(
    avl_exe: str | Path | None,
    cases: list[AvlCase],
    workdir: str | Path,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> list[AvlRunResult]:
    """Run several AVL cases in one AVL process (one session).

    The first geometry file goes on the command line. A case whose
    geometry text is not the geometry that is loaded is read with the
    ``LOAD`` command (a ground-effect sweep has one rotated geometry
    per angle). A free-air sweep keeps one geometry, so AVL runs all
    its angles on one loaded configuration. Each case writes its own
    ``FT`` and ``ST`` files.

    Parameters
    ----------
    avl_exe : str, Path or None
        Path of ``avl.exe``. A wrong path gives failed results.
    cases : list of AvlCase
        Run cases in run order (at least one).
    workdir : str or Path
        Folder for all the input and output files of the session. Old
        output files in it are deleted before the run.
    timeout : float
        Wall-time limit [s] of each case; the session limit is this
        value times the case count.

    Returns
    -------
    list of AvlRunResult
        One result per case, with the same checks as :func:`run_case`.
        ``elapsed_s`` is the wall time [s] of the whole session.

    Raises
    ------
    ValueError
        If *cases* is empty.
    """
    if not cases:
        raise ValueError("run_session needs at least one case.")
    folder = Path(workdir)
    started = time.perf_counter()
    n = len(cases)
    avl_names = [f"case{k:03d}.avl" for k in range(n)]
    ft_names = [f"ft{k:03d}.txt" for k in range(n)]
    st_names = [f"st{k:03d}.txt" for k in range(n)]

    def failed(reason: str, elapsed: float = 0.0) -> list[AvlRunResult]:
        return [AvlRunResult(ok=False, workdir=folder, elapsed_s=elapsed, error=reason)
                for _ in cases]

    try:
        folder.mkdir(parents=True, exist_ok=True)
        _clear_outputs(folder, ft_names + st_names + [STDOUT_FILENAME])
    except OSError as exc:
        return failed(f"Cannot prepare folder: {exc}.")
    entries: list[tuple[str | None, float, float, str, str]] = []
    loaded: str | None = None
    for k, case in enumerate(cases):
        load_name: str | None = None
        if k == 0 or case.avl_text != loaded:
            (folder / avl_names[k]).write_text(case.avl_text, encoding="utf-8")
            load_name = None if k == 0 else avl_names[k]
            loaded = case.avl_text
        entries.append((load_name, case.run_alpha_deg, case.run_beta_deg,
                        ft_names[k], st_names[k]))
    commands = session_command_stream(entries)
    (folder / COMMANDS_FILENAME).write_text(commands, encoding="utf-8")

    if avl_exe is None or not Path(avl_exe).is_file():
        return failed(f"AVL program not found: {avl_exe}.")
    session_timeout = float(timeout) * n
    try:
        completed = subprocess.run(
            [str(avl_exe), avl_names[0]],
            input=commands,
            capture_output=True,
            text=True,
            timeout=session_timeout,
            cwd=folder,
        )
    except subprocess.TimeoutExpired:
        return failed(f"AVL session timed out after {session_timeout:g} s.", session_timeout)
    except OSError as exc:
        return failed(f"Cannot start AVL: {exc}.")
    stdout = completed.stdout or ""
    (folder / STDOUT_FILENAME).write_text(stdout, encoding="utf-8")
    try:
        banner: str | None = parse_banner(stdout)
    except ValueError:
        banner = None
    results: list[AvlRunResult] = []
    for k, case in enumerate(cases):
        try:
            forces, stabderivs = _parse_outputs(folder, case, ft_names[k], st_names[k])
        except ValueError as exc:
            results.append(AvlRunResult(ok=False, workdir=folder,
                                        returncode=completed.returncode,
                                        banner=banner, error=str(exc)))
            continue
        if banner is None:
            results.append(AvlRunResult(
                ok=False, workdir=folder, returncode=completed.returncode,
                forces=forces, stabderivs=stabderivs,
                error="AVL banner not found in the program output.",
            ))
            continue
        results.append(AvlRunResult(
            ok=True, workdir=folder, returncode=completed.returncode,
            banner=banner, forces=forces, stabderivs=stabderivs,
        ))
    elapsed = time.perf_counter() - started
    for result in results:
        result.elapsed_s = elapsed
    return results
