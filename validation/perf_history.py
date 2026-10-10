"""Measure the speed of fixed Ventorum workloads over time.

Usage
-----
    python validation/perf_history.py --history PATH [--note TEXT] [--tiny]
    python validation/perf_history.py --report PATH
    python validation/perf_history.py --compare TREE_A TREE_B [--rounds N] [--workloads a,b]

Record mode appends one JSON object per line to the history file of
``--history``. The owner keeps that file outside the repository. The script
refuses a history path inside the repository, and a record holds no machine
name, no user name and no path. Record mode times the ``ventorum`` package of
the tree that holds this script: the measurement runs in a child process with
``PYTHONPATH`` set to that tree, and the child checks its import. A record
made with ``--tiny`` has the mark ``[tiny]`` at the end of its note.

Report mode prints one Markdown row per record, one column per workload, and
the percent change against the previous record of the same kind (tiny or
full).

Compare mode times two source trees side by side: one subprocess per run,
``PYTHONPATH`` set to the tree, A then B in even rounds and B then A in odd
rounds. Each subprocess checks that it imports ``tree/ventorum/__init__.py``.
This is the only proof of a speed change (see
``docs/design/performance_architecture.md``, "How to check a change"). A
history row is a trend.

Workloads
---------
The names in :data:`WORKLOAD_NAMES` are fixed. Never change the definition of
a workload: the history would then compare different work. Add a new workload
under a new name.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import ventorum as vt
from ventorum import gpu
from ventorum.utils.parallel import cpu_cores

#: Root of the source tree that holds this script.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Fixed workload names, in the order of the record and of the report.
WORKLOAD_NAMES = [
    "single_vlm_n20_c4",
    "single_vlm_n80_c4",
    "single_linear_n80",
    "single_nonlinear_n80",
    "sweep_vlm_n40_c4_33",
    "sweep_linear_n80_33",
    "wingtail_vlm_n40_c4_33",
    "ground_vlm_n40_6x9",
]

#: Sweep workloads. They run again on the GPU as rows with the suffix "_gpu".
SWEEP_WORKLOADS = [
    "sweep_vlm_n40_c4_33",
    "sweep_linear_n80_33",
    "wingtail_vlm_n40_c4_33",
    "ground_vlm_n40_6x9",
]

#: Timed calls per workload in a normal run (one warm-up call comes first).
TIMED_CALLS = 7
#: Warm-up time [s] of the GPU before the GPU rows.
GPU_WARMUP_S = 0.5
#: Timeout [s] of one subprocess of the compare mode.
CHILD_TIMEOUT_S = 600.0
#: Timeout [s] of the child process of the record mode (all workloads).
RECORD_TIMEOUT_S = 3600.0
#: Mark at the end of the note of a record made with ``--tiny``.
TINY_MARK = "[tiny]"
#: Line that must stay under the report table.
TREND_LINE = "A history row is a trend. It is not a proof of a speed change: use --compare."
#: Prefix of the JSON line that a subprocess of the compare mode prints.
JSON_MARKER = "__PERF_HISTORY_JSON__"

#: Fields of one record, in the order of the record.
RECORD_FIELDS = (
    "date",
    "ventorum_version",
    "git_commit",
    "git_dirty",
    "python",
    "numpy",
    "cpu_cores",
    "gpu",
    "workloads",
    "note",
)

# Geometry of the workloads [m].
_CHORD = 1.25  # rectangular wing chord: semi-span 5 m gives aspect ratio 8
_TAIL_SEMI_SPAN = 1.5
_TAIL_CHORD = 0.6
_TAIL_X = 4.0  # tail 4 m behind the reference point
_TAIL_Z = 0.3  # tail 0.3 m above the reference point

# Flight conditions of the workloads [deg].
_ALPHA_SINGLE_DEG = 4.0
_ALPHA_MIN_DEG = -4.0
_ALPHA_MAX_DEG = 12.0
_SWEEP_CASES = 33
_GROUND_HEIGHTS = 6
_GROUND_ALPHAS = 9
_H_OVER_C_MIN = 0.25
_H_OVER_C_MAX = 2.0

# Test mode (--tiny): small meshes, few angles, one timed call.
_TINY_PANELS = 8
_TINY_CHORD_PANELS = 2
_TINY_SWEEP_CASES = 5
_TINY_GROUND_HEIGHTS = 3
_TINY_GROUND_ALPHAS = 3


@dataclass(frozen=True)
class _Spec:
    """Definition of one workload: its solver and its mesh."""

    kind: str  # "single", "sweep", "wingtail" or "ground"
    solver: str  # "vlm", "linear" or "nonlinear"
    panels: int  # spanwise panels per semi-span
    chord: int | None  # chordwise panels (VLM only, None = automatic)


#: The fixed workload table. A change here changes the meaning of the history.
_SPECS: dict[str, _Spec] = {
    "single_vlm_n20_c4": _Spec("single", "vlm", 20, 4),
    "single_vlm_n80_c4": _Spec("single", "vlm", 80, 4),
    "single_linear_n80": _Spec("single", "linear", 80, None),
    "single_nonlinear_n80": _Spec("single", "nonlinear", 80, None),
    "sweep_vlm_n40_c4_33": _Spec("sweep", "vlm", 40, 4),
    "sweep_linear_n80_33": _Spec("sweep", "linear", 80, None),
    "wingtail_vlm_n40_c4_33": _Spec("wingtail", "vlm", 40, 4),
    "ground_vlm_n40_6x9": _Spec("ground", "vlm", 40, None),
}


@dataclass(frozen=True)
class Workload:
    """One workload: its call, its unit and its cases per call.

    Parameters
    ----------
    name : str
        Fixed name from :data:`WORKLOAD_NAMES`.
    call : callable
        Zero-argument call that runs the workload one time.
    unit : str
        ``"ms"`` for a latency workload, ``"cases/s"`` for a sweep workload.
    cases : int
        Flight conditions solved in one call (1 for a single solve).
    """

    name: str
    call: Callable[[], Any]
    unit: str
    cases: int


def _rect_wing() -> vt.LiftingSurface:
    """Return the rectangular wing of the workloads: aspect ratio 8.

    Returns
    -------
    LiftingSurface
        Semi-span 5 m, chord 1.25 m, linear airfoil at both sections.
    """
    return vt.LiftingSurface(
        name="PerfWing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=_CHORD, airfoil=vt.LinearAirfoil()),
            vt.WingSection(y_frac=1.0, chord=_CHORD, airfoil=vt.LinearAirfoil()),
        ],
    )


def _wing_tail() -> vt.Aircraft:
    """Return the wing and the horizontal tail of the wingtail workload.

    Returns
    -------
    Aircraft
        The wing at the origin, the tail 4 m behind and 0.3 m up, tail
        semi-span 1.5 m, tail chord 0.6 m.
    """
    tail = vt.LiftingSurface(
        name="PerfTail",
        semi_span=_TAIL_SEMI_SPAN,
        position=np.array([_TAIL_X, 0.0, _TAIL_Z]),
        sections=[
            vt.WingSection(y_frac=0.0, chord=_TAIL_CHORD, airfoil=vt.LinearAirfoil()),
            vt.WingSection(y_frac=1.0, chord=_TAIL_CHORD, airfoil=vt.LinearAirfoil()),
        ],
    )
    return vt.Aircraft(name="PerfWingTail", surfaces=[_rect_wing(), tail])


def _settings(solver: str, panels: int, n_chord: int | None) -> vt.SolverSettings:
    """Return fresh solver settings for one call of a workload.

    Parameters
    ----------
    solver : str
        Solver name: ``"vlm"``, ``"linear"`` or ``"nonlinear"``.
    panels : int
        Spanwise panels per semi-span.
    n_chord : int or None
        Chordwise panels, or None for the automatic rule.

    Returns
    -------
    SolverSettings
    """
    return vt.SolverSettings(solver_type=solver, n_panels=panels, n_chord=n_chord)


def build_workload(name: str, tiny: bool) -> Workload:
    """Return the fixed workload *name* with its call, unit and case count.

    Parameters
    ----------
    name : str
        One of :data:`WORKLOAD_NAMES`.
    tiny : bool
        Test mode: small meshes, few angles, small ground grid.

    Returns
    -------
    Workload

    Raises
    ------
    ValueError
        If *name* is not a fixed workload name.
    """
    if name not in _SPECS:
        raise ValueError(f"unknown workload {name!r}; the fixed names are {WORKLOAD_NAMES}")
    spec = _SPECS[name]
    panels = _TINY_PANELS if tiny else spec.panels
    n_chord = None if spec.chord is None else (_TINY_CHORD_PANELS if tiny else spec.chord)

    if spec.kind == "ground":
        heights = np.linspace(
            _H_OVER_C_MIN, _H_OVER_C_MAX, _TINY_GROUND_HEIGHTS if tiny else _GROUND_HEIGHTS
        ) * _CHORD
        alphas = np.linspace(
            _ALPHA_MIN_DEG, _ALPHA_MAX_DEG, _TINY_GROUND_ALPHAS if tiny else _GROUND_ALPHAS
        )
        sweep = vt.GroundEffectSweep(_rect_wing(), solver=spec.solver, n_panels=panels)
        cases = int(heights.size * alphas.size)
        return Workload(name, lambda: sweep.run_sweep(heights, alphas), "cases/s", cases)

    geometry: vt.Aircraft | vt.LiftingSurface
    geometry = _wing_tail() if spec.kind == "wingtail" else _rect_wing()

    if spec.kind == "single":
        return Workload(
            name,
            lambda: vt.analyze(
                geometry, alpha_deg=_ALPHA_SINGLE_DEG, settings=_settings(spec.solver, panels, n_chord)
            ),
            "ms",
            1,
        )

    n_cases = _TINY_SWEEP_CASES if tiny else _SWEEP_CASES
    alphas = np.linspace(_ALPHA_MIN_DEG, _ALPHA_MAX_DEG, n_cases)
    return Workload(
        name,
        lambda: vt.analyze_sweep(geometry, alphas, settings=_settings(spec.solver, panels, n_chord)),
        "cases/s",
        n_cases,
    )


def measure(workload: Workload, tiny: bool) -> dict[str, Any]:
    """Time *workload*: one warm-up call, then the timed calls.

    Parameters
    ----------
    workload : Workload
        The workload to time.
    tiny : bool
        Test mode with one timed call. A normal run has 7 timed calls.

    Returns
    -------
    dict
        ``unit``, ``median``, ``min`` and ``max`` in the unit of the
        workload: ms per call for a latency workload, cases per second for a
        sweep workload. The timing uses :func:`time.perf_counter`.
    """
    workload.call()  # warm-up: caches and compiled kernels
    gc.collect()  # collect the garbage of the warm-up before the timed calls
    calls = 1 if tiny else TIMED_CALLS
    values = np.empty(calls, dtype=float)
    for index in range(calls):
        start = time.perf_counter()
        workload.call()
        elapsed = time.perf_counter() - start
        values[index] = 1000.0 * elapsed if workload.unit == "ms" else workload.cases / elapsed
    return {
        "unit": workload.unit,
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def run_workload(name: str, tiny: bool, device: str) -> dict[str, Any]:
    """Run the workload *name* on *device* and return its timing record.

    Parameters
    ----------
    name : str
        One of :data:`WORKLOAD_NAMES`.
    tiny : bool
        Test mode: small meshes and one timed call.
    device : str
        ``"cpu"`` or ``"gpu"``.

    Returns
    -------
    dict
        ``unit``, ``median``, ``min`` and ``max``.
    """
    gpu.set_device(device)
    return measure(build_workload(name, tiny), tiny)


def run_workloads(tiny: bool) -> dict[str, dict[str, Any]]:
    """Run every workload on the CPU, and the sweeps on the GPU when one exists.

    Parameters
    ----------
    tiny : bool
        Test mode: small meshes and one timed call.

    Returns
    -------
    dict
        One entry per row: the workload names, and the sweep names with the
        suffix ``"_gpu"`` when the GPU pipelines are available.
    """
    previous = gpu.get_device()
    rows: dict[str, dict[str, Any]] = {}
    try:
        gpu.set_device("cpu")
        for name in WORKLOAD_NAMES:
            rows[name] = run_workload(name, tiny, "cpu")
        if gpu.available():
            warm_gpu()
            for name in SWEEP_WORKLOADS:
                rows[name + "_gpu"] = run_workload(name, tiny, "gpu")
    finally:
        gpu.set_device(previous)
    return rows


def warm_gpu(seconds: float = GPU_WARMUP_S) -> None:
    """Keep the GPU busy for *seconds* [s] before the GPU rows.

    The first work on an idle GPU runs slower than the next work.
    """
    import torch

    matrix = torch.randn(512, 512, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        matrix @ matrix
    torch.cuda.synchronize()


def git_state(tree: Path) -> tuple[str, bool]:
    """Return the commit hash and the dirty flag of the source tree *tree*.

    Parameters
    ----------
    tree : Path
        Folder of a git work tree.

    Returns
    -------
    tuple of (str, bool)
        ``("unknown", False)`` when git does not run or *tree* is not a git
        work tree.
    """
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=tree, capture_output=True, text=True, timeout=60
        )
        if rev.returncode != 0:
            return "unknown", False
        status = subprocess.run(
            ["git", "--no-optional-locks", "status", "--porcelain"],
            cwd=tree,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown", False
    commit = rev.stdout.strip() or "unknown"
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else False
    return commit, dirty


def tiny_note(note: str) -> str:
    """Return *note* with the mark :data:`TINY_MARK` at its end.

    Parameters
    ----------
    note : str
        Text of the ``--note`` option.

    Returns
    -------
    str
        The note of a record made with ``--tiny``.
    """
    return f"{note} {TINY_MARK}" if note else TINY_MARK


def is_tiny_record(record: dict[str, Any]) -> bool:
    """Return True when *record* was made with ``--tiny``.

    Parameters
    ----------
    record : dict
        One record of the history file.

    Returns
    -------
    bool
        True when the note ends with :data:`TINY_MARK`.
    """
    return str(record.get("note") or "").rstrip().endswith(TINY_MARK)


def make_record(
    note: str,
    rows: dict[str, dict[str, Any]],
    tree: Path | None = None,
    tiny: bool = False,
) -> dict[str, Any]:
    """Build the record of one run.

    Parameters
    ----------
    note : str
        Text of the ``--note`` option.
    rows : dict
        Timing rows from :func:`run_workloads`.
    tree : Path or None
        Source tree of the git commit. Defaults to the tree of this script.
    tiny : bool
        True when the rows used the test meshes. The note then gets the
        mark :data:`TINY_MARK`.

    Returns
    -------
    dict
        The fields of the record. The record holds no machine name, no user
        name and no path.
    """
    commit, dirty = git_state(REPO_ROOT if tree is None else tree)
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "ventorum_version": vt.__version__,
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_cores": int(cpu_cores()),
        "gpu": gpu.info().get("name"),
        "workloads": rows,
        "note": tiny_note(note) if tiny else note,
    }


def run_record(note: str, tiny: bool, tree: Path | None = None) -> dict[str, Any]:
    """Run all workloads in this process and return the record.

    Parameters
    ----------
    note : str
        Text of the ``--note`` option.
    tiny : bool
        Test mode: small meshes and one timed call.
    tree : Path or None
        Source tree of the git commit. Defaults to the tree of this script.

    Returns
    -------
    dict
        The record of :func:`make_record`.
    """
    return make_record(note, run_workloads(tiny), tree, tiny)


def record_in_child(note: str, tiny: bool) -> dict[str, Any]:
    """Time the ``ventorum`` package of :data:`REPO_ROOT` in a child process.

    The parent process can import ``ventorum`` from another tree (for
    example an editable install). The child has ``PYTHONPATH`` set to
    :data:`REPO_ROOT` and checks that it imports ``ventorum`` from that tree,
    so the times and the git commit of the record come from the same tree.

    Parameters
    ----------
    note : str
        Text of the ``--note`` option.
    tiny : bool
        Test mode: small meshes and one timed call.

    Returns
    -------
    dict
        The record of :func:`make_record`.

    Raises
    ------
    SystemExit
        If the child fails or does not import ``ventorum`` from the tree.
    """
    options = ["--record-child", f"--note={note}", "--expect-tree", str(REPO_ROOT)]
    if tiny:
        options.append("--tiny")
    return _child(options, REPO_ROOT, timeout=RECORD_TIMEOUT_S)


def _is_inside(path: Path, root: Path) -> bool:
    """Return True when the resolved *path* is *root* or lies inside *root*."""
    candidate = os.path.normcase(str(path.resolve()))
    base = os.path.normcase(str(root.resolve()))
    return candidate == base or candidate.startswith(base + os.sep)


def check_history_path(path: Path, root: Path | None = None) -> Path:
    """Return the resolved history path, or stop when it is inside the repository.

    Parameters
    ----------
    path : Path
        Path given to ``--history``.
    root : Path or None
        Repository root. Defaults to the tree of this script.

    Returns
    -------
    Path
        The resolved path, which is outside the repository.

    Raises
    ------
    SystemExit
        If the path is inside the repository.
    """
    base = REPO_ROOT if root is None else root
    resolved = Path(path).expanduser().resolve()
    if _is_inside(resolved, base):
        raise SystemExit(
            f"error: the history file {resolved} is inside the repository {base.resolve()}. "
            "A history file must live outside the repository; give another --history path."
        )
    return resolved


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append *record* to *path* as one JSON line.

    Parameters
    ----------
    path : Path
        History file outside the repository (see :func:`check_history_path`).
    record : dict
        The record to append. It must hold the fields of
        :data:`RECORD_FIELDS` and no other field.

    Raises
    ------
    SystemExit
        If the record does not match the record fields.
    """
    if set(record) != set(RECORD_FIELDS):
        raise SystemExit(
            f"error: the record fields are {list(RECORD_FIELDS)} and the record has {list(record)}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def read_history(path: Path) -> list[dict[str, Any]]:
    """Read every record of the history file *path*.

    Parameters
    ----------
    path : Path
        History file, one JSON object per line. Empty lines are skipped.

    Returns
    -------
    list of dict
        The records in file order.

    Raises
    ------
    SystemExit
        If the file cannot be read or a line is not JSON.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"error: cannot read the history file {path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"error: the history file {path}, line {number}, is not JSON: {exc}") from exc
    return records


@dataclass
class ReportCell:
    """One cell of the report table.

    Parameters
    ----------
    value : float or None
        Median of the workload in that record, or None when the record does
        not hold the workload.
    change_pct : float or None
        Change [%] of the value against the previous record, or None when the
        previous record has no value.
    """

    value: float | None
    change_pct: float | None


@dataclass
class ReportRow:
    """One record as one table row.

    Parameters
    ----------
    date : str
        ISO date and time of the record.
    version : str
        Ventorum version of the record.
    commit : str
        Short git commit of the record.
    cells : list of ReportCell
        One cell per column of :class:`Report`.
    tiny : bool
        True when the record was made with ``--tiny``.
    """

    date: str
    version: str
    commit: str
    cells: list[ReportCell]
    tiny: bool = False


@dataclass
class Report:
    """The whole report table.

    Parameters
    ----------
    columns : list of str
        Workload names, in first-seen order.
    units : list of str or None
        Unit per column (``"ms"`` or ``"cases/s"``).
    rows : list of ReportRow
        One row per record, in file order.
    """

    columns: list[str]
    units: list[str | None]
    rows: list[ReportRow]


def build_report(records: Sequence[dict[str, Any]]) -> Report:
    """Build the report table of the records.

    Parameters
    ----------
    records : sequence of dict
        Records in file order (the oldest first).

    Returns
    -------
    Report
        The percent change of a cell is against the previous record of the
        same kind (tiny or full) that has a value for that workload. The
        report never compares a tiny record with a full record.
    """
    columns: list[str] = []
    units: dict[str, str | None] = {}
    for record in records:
        for name, entry in (record.get("workloads") or {}).items():
            if name not in units:
                columns.append(name)
                units[name] = None
            if units[name] is None and isinstance(entry, dict):
                unit = entry.get("unit")
                units[name] = str(unit) if unit is not None else None

    rows: list[ReportRow] = []
    previous: dict[tuple[bool, str], float] = {}
    for record in records:
        tiny = is_tiny_record(record)
        workloads = record.get("workloads") or {}
        cells: list[ReportCell] = []
        for name in columns:
            entry = workloads.get(name)
            value: float | None = None
            if isinstance(entry, dict):
                raw = entry.get("median")
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    value = float(raw)
            change: float | None = None
            key = (tiny, name)
            if value is not None and previous.get(key) not in (None, 0.0):
                change = 100.0 * (value - previous[key]) / previous[key]
            cells.append(ReportCell(value=value, change_pct=change))
            if value is not None:
                previous[key] = value
        commit = str(record.get("git_commit") or "unknown")
        rows.append(
            ReportRow(
                date=str(record.get("date") or ""),
                version=str(record.get("ventorum_version") or ""),
                commit=commit[:7],
                cells=cells,
                tiny=tiny,
            )
        )
    return Report(columns=columns, units=[units[name] for name in columns], rows=rows)


def _cell_text(cell: ReportCell) -> str:
    """Return the text of one report cell."""
    if cell.value is None:
        return "-"
    text = f"{cell.value:.6g}"
    if cell.change_pct is not None:
        text += f" ({cell.change_pct:+.1f}%)"
    return text


def render_report(report: Report) -> str:
    """Return the report table as Markdown text.

    Parameters
    ----------
    report : Report
        Table from :func:`build_report`.

    Returns
    -------
    str
        The Markdown table, an empty line and :data:`TREND_LINE`. The commit
        cell of a tiny record has the mark :data:`TINY_MARK`.
    """
    header = ["date", "version", "commit"]
    header += [f"{name} ({unit})" if unit else name for name, unit in zip(report.columns, report.units)]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in report.rows:
        commit = f"{row.commit} {TINY_MARK}" if row.tiny else row.commit
        cells = [row.date, row.version, commit]
        cells += [_cell_text(cell) for cell in row.cells]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if any(row.tiny for row in report.rows):
        lines.append(
            f"A row with {TINY_MARK} used the test meshes. The change in percent compares "
            "only records of the same kind."
        )
    lines.append(TREND_LINE)
    return "\n".join(lines) + "\n"


@dataclass
class TreeInfo:
    """What the compare mode learned about one source tree.

    Parameters
    ----------
    tree : Path
        Folder that holds the ``ventorum`` package under test.
    version : str
        Version of the imported package.
    cython : bool
        True when the compiled Cython kernel is importable. The kernel
        backend changes the time, so the table header reports it.
    file : str
        Imported ``ventorum/__init__.py``, relative to *tree*.
    """

    tree: Path
    version: str
    cython: bool
    file: str = ""


@dataclass
class CompareRow:
    """One workload of the side-by-side table.

    Parameters
    ----------
    workload : str
        Fixed workload name.
    unit : str
        ``"ms"`` or ``"cases/s"``.
    median_a : float
        Median over the rounds of the medians of tree A.
    median_b : float
        Median over the rounds of the medians of tree B.
    ratio : float
        ``median_b / median_a``.
    spread : float
        Max minus min of the ratio over the rounds.
    """

    workload: str
    unit: str
    median_a: float
    median_b: float
    ratio: float
    spread: float


@dataclass
class CompareResult:
    """Result of the compare mode.

    Parameters
    ----------
    rounds : int
        Number of alternating rounds.
    tiny : bool
        True when the runs used the test meshes.
    tree_a, tree_b : TreeInfo
        The two source trees.
    rows : list of CompareRow
        One row per workload.
    """

    rounds: int
    tiny: bool
    tree_a: TreeInfo
    tree_b: TreeInfo
    rows: list[CompareRow]


def _have_cython() -> bool:
    """Return True when this process has the compiled Cython kernel."""
    from ventorum.aero import vortex

    return bool(vortex._HAVE_CYTHON)


def _import_checked(tree: Path) -> Any:
    """Import ``ventorum`` and stop when the import does not come from *tree*.

    Parameters
    ----------
    tree : Path
        Source tree that must hold the imported package.

    Returns
    -------
    module
        The imported ``ventorum`` package.

    Raises
    ------
    SystemExit
        If the package cannot be imported, if it is a namespace package
        (no ``__init__.py``), or if its ``__init__.py`` is not
        ``tree/ventorum/__init__.py``. A file elsewhere under *tree* (for
        example in a nested worktree) is also refused.
    """
    try:
        import ventorum as imported
    except ImportError as exc:
        raise SystemExit(f"error: ventorum cannot be imported for the tree {tree}: {exc}") from exc
    location = getattr(imported, "__file__", None)
    if location is None:
        raise SystemExit(
            f"error: ventorum is imported as a namespace package (no __init__.py) and not from the "
            f"tree {tree}. The subprocess does not test this tree; check PYTHONPATH."
        )
    found = Path(location).resolve()
    expected = (Path(tree) / "ventorum" / "__init__.py").resolve()
    if found != expected:
        raise SystemExit(
            f"error: ventorum is imported from {found} and not from the tree {tree}. "
            "The subprocess does not test this tree; check PYTHONPATH."
        )
    return imported


def _child(options: Sequence[str], tree: Path, timeout: float = CHILD_TIMEOUT_S) -> dict[str, Any]:
    """Run this script as a child process on *tree* and return its JSON line.

    Parameters
    ----------
    options : sequence of str
        Child options, for example ``["--probe", "--expect-tree", tree]``.
    tree : Path
        Source tree. ``PYTHONPATH`` is set to this tree alone.
    timeout : float
        Time limit [s] of the child.

    Returns
    -------
    dict
        The JSON payload of the child.

    Raises
    ------
    SystemExit
        If the child fails, times out or prints no payload.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    command = [sys.executable, str(Path(__file__).resolve()), *options]
    try:
        proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"error: the run in the tree {tree} did not finish in {timeout:g} s."
        ) from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise SystemExit(f"error: the run in the tree {tree} failed:\n{detail}")
    for line in proc.stdout.splitlines():
        if line.startswith(JSON_MARKER):
            try:
                return json.loads(line[len(JSON_MARKER):].strip())
            except json.JSONDecodeError as exc:
                raise SystemExit(f"error: the run in the tree {tree} printed a bad result line: {exc}") from exc
    raise SystemExit(f"error: the run in the tree {tree} printed no result line:\n{proc.stdout.strip()}")


def _probe(tree: Path) -> TreeInfo:
    """Return the version and the kernel backends that *tree* gives.

    Parameters
    ----------
    tree : Path
        Source tree under test.

    Returns
    -------
    TreeInfo
    """
    payload = _child(["--probe", "--expect-tree", str(tree)], tree)
    raw = Path(str(payload.get("file") or "?"))
    try:
        relative = raw.resolve().relative_to(Path(tree).resolve()).as_posix()
    except (OSError, ValueError):
        relative = raw.name
    return TreeInfo(
        tree=tree,
        version=str(payload.get("version") or "?"),
        cython=bool(payload.get("cython")),
        file=relative,
    )


def run_compare(
    tree_a: Path,
    tree_b: Path,
    rounds: int = 3,
    workloads: Sequence[str] | None = None,
    tiny: bool = False,
) -> CompareResult:
    """Time the workloads on two source trees, alternating A and B.

    Even rounds (0, 2, ...) run A then B; odd rounds run B then A, so that
    neither tree always runs first. Every run is a child process with ``PYTHONPATH`` set to its tree, so the
    subprocess tests the tree under test and not the installed package. Each
    child checks its own import (see :func:`_import_checked`).

    Parameters
    ----------
    tree_a, tree_b : Path
        Folders that each hold a ``ventorum`` package (two worktrees, for
        example).
    rounds : int
        Number of rounds (default 3, at least 1).
    workloads : sequence of str or None
        Workload names to run. All fixed names when None.
    tiny : bool
        Test mode: small meshes and one timed call.

    Returns
    -------
    CompareResult

    Raises
    ------
    SystemExit
        If a tree is not a folder, a name is unknown, or a run fails.
    """
    if rounds < 1:
        raise SystemExit("error: --rounds must be at least 1.")
    names = list(WORKLOAD_NAMES) if workloads is None else [str(name) for name in workloads]
    if not names:
        raise SystemExit("error: --workloads selects no workload.")
    unknown = [name for name in names if name not in WORKLOAD_NAMES]
    if unknown:
        raise SystemExit(
            f"error: unknown workload name(s): {', '.join(unknown)}. The fixed names are: "
            f"{', '.join(WORKLOAD_NAMES)}."
        )
    left = Path(tree_a).expanduser().resolve()
    right = Path(tree_b).expanduser().resolve()
    for tree in (left, right):
        if not tree.is_dir():
            raise SystemExit(f"error: {tree} is not a folder.")

    info_a = _probe(left)
    info_b = _probe(right)
    rows: list[CompareRow] = []
    for name in names:
        medians_a: list[float] = []
        medians_b: list[float] = []
        ratios: list[float] = []
        unit = ""
        options = ["--run", name]
        if tiny:
            options.append("--tiny")
        for index in range(rounds):
            order = ((left, "a"), (right, "b")) if index % 2 == 0 else ((right, "b"), (left, "a"))
            runs: dict[str, dict[str, Any]] = {}
            for tree, side in order:
                runs[side] = _child([*options, "--expect-tree", str(tree)], tree)
            run_a, run_b = runs["a"], runs["b"]
            if not unit:
                unit = str(run_a["unit"])
            if str(run_b["unit"]) != unit:
                raise SystemExit(
                    f"error: workload {name} gives the unit {run_b['unit']} in tree B and {unit} in tree A."
                )
            median_a = float(run_a["median"])
            median_b = float(run_b["median"])
            medians_a.append(median_a)
            medians_b.append(median_b)
            ratios.append(median_b / median_a)
        median_a = float(np.median(medians_a))
        median_b = float(np.median(medians_b))
        rows.append(
            CompareRow(
                workload=name,
                unit=unit,
                median_a=median_a,
                median_b=median_b,
                ratio=median_b / median_a,
                spread=float(max(ratios) - min(ratios)),
            )
        )
    return CompareResult(rounds=rounds, tiny=tiny, tree_a=info_a, tree_b=info_b, rows=rows)


def _cython_text(cython: bool) -> str:
    """Return the header text of one tree about the Cython kernel."""
    if cython:
        return "compiled Cython kernel"
    return "no compiled Cython kernel (the kernel backend changes the time)"


def _tree_line(label: str, info: TreeInfo) -> str:
    """Return the header line of one tree: folder name, imported file, version, kernel."""
    imported = info.file or "?"
    return (
        f"Tree {label}: {Path(info.tree).name} (imports {imported}, ventorum {info.version}, "
        f"{_cython_text(info.cython)})"
    )


def render_compare(result: CompareResult) -> str:
    """Return the side-by-side table as Markdown text.

    Parameters
    ----------
    result : CompareResult
        Result of :func:`run_compare`.

    Returns
    -------
    str
        The header with both trees and the table. The header gives the folder
        name of each tree (not the full path), the imported file relative to
        the tree, and names a tree without a compiled Cython kernel.
    """
    lines = [
        f"Side-by-side timing: {result.rounds} round(s), A then B in even rounds and B then A in "
        "odd rounds, one subprocess per run.",
        _tree_line("A", result.tree_a),
        _tree_line("B", result.tree_b),
        "The spread column is max minus min of the ratio B/A over the rounds.",
        "A ratio below 1 means that B is faster for a workload in ms; "
        "a ratio above 1 means that B is faster for a workload in cases/s.",
    ]
    if result.tiny:
        lines.append("Test meshes and one timed call (--tiny): the table does not show the full workloads.")
    lines.append("")
    header = ["workload", "unit", "median A", "median B", "ratio B/A", "spread of rounds"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in result.rows:
        cells = [
            row.workload,
            row.unit,
            f"{row.median_a:.6g}",
            f"{row.median_b:.6g}",
            f"{row.ratio:.4f}",
            f"{row.spread:.4f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the script from the command line.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments without the program name.

    Returns
    -------
    int
        Exit code 0.

    Raises
    ------
    SystemExit
        On a usage error, on a refused history path or on a failed run. The
        message of the exit is the error text.
    """
    parser = argparse.ArgumentParser(
        prog="perf_history.py",
        description="Append a speed record, print the history, or time two source trees side by side.",
    )
    parser.add_argument("--history", type=Path, help="history file outside the repository; appends one record")
    parser.add_argument("--note", default="", help="text stored in the record as the note")
    parser.add_argument("--report", type=Path, metavar="PATH", help="print the history table of this file")
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("TREE_A", "TREE_B"),
        help="time these two source trees side by side, alternating A and B",
    )
    parser.add_argument("--rounds", type=int, default=3, help="rounds of the compare mode (default 3)")
    parser.add_argument("--workloads", default="", help="comma-separated workload names of the compare mode")
    parser.add_argument("--tiny", action="store_true", help="test mode: small meshes and one timed call")
    parser.add_argument("--run", metavar="NAME", help="internal: run one workload and print one JSON line")
    parser.add_argument("--probe", action="store_true", help="internal: print version and kernel backends")
    parser.add_argument(
        "--record-child", action="store_true", help="internal: run all workloads and print the record"
    )
    parser.add_argument("--expect-tree", type=Path, help="internal: check that ventorum comes from this tree")
    args = parser.parse_args(argv)

    modes = sum(value is not None for value in (args.history, args.report, args.compare))
    modes += (args.run is not None) + int(args.probe) + int(args.record_child)
    if modes > 1:
        parser.error("give one mode only: --history, --report, --compare, --run, --probe or --record-child.")

    if args.run is not None or args.probe or args.record_child:
        if args.expect_tree is None:
            parser.error("--run, --probe and --record-child need --expect-tree TREE.")
        tree = Path(args.expect_tree).expanduser().resolve()
        imported = _import_checked(tree)
        if args.probe:
            payload = {"version": imported.__version__, "cython": _have_cython(), "file": imported.__file__}
            print(f"{JSON_MARKER} " + json.dumps(payload))
            return 0
        if args.record_child:
            print(f"{JSON_MARKER} " + json.dumps(run_record(args.note, args.tiny, tree)))
            return 0
        if args.run not in WORKLOAD_NAMES:
            parser.error(f"unknown workload {args.run!r}; the fixed names are: {', '.join(WORKLOAD_NAMES)}.")
        payload = run_workload(args.run, args.tiny, "cpu")
        payload["workload"] = args.run
        print(f"{JSON_MARKER} " + json.dumps(payload))
        return 0

    if args.report is not None:
        records = read_history(args.report)
        sys.stdout.write(render_report(build_report(records)))
        return 0

    if args.compare is not None:
        names = [part.strip() for part in str(args.workloads).split(",") if part.strip()]
        result = run_compare(
            args.compare[0], args.compare[1], rounds=args.rounds, workloads=names or None, tiny=args.tiny
        )
        sys.stdout.write(render_compare(result))
        return 0

    if args.history is None:
        parser.error("--history PATH is required to write a record.")
    history = check_history_path(args.history)
    record = record_in_child(args.note, args.tiny)
    append_record(history, record)
    print(f"appended a record with {len(record['workloads'])} workloads to {history}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
