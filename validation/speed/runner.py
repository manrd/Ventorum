# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Time each low-order tool and Ventorum at equal accuracy.

The runner uses the same cases for every tool (see
:mod:`validation.speed.cases`), finds the mesh where the CL and
CDi sweeps of each tool change by less than 0.5 % on refinement,
then times whole sweeps at that mesh. Each tool runs as a black
box: an external program or a separate interpreter, never as a
Ventorum dependency. A missing tool is skipped with a note.

Usage
-----
python validation/speed/runner.py --out DIR [--quick]
    [--tools avl,ventorum] [--timeout S] [--repeats N]
    [--avl PATH] [--vspaero PATH] [--flow5 PATH]
    [--aerosandbox-python PATH] [--machupx-python PATH]

``--out`` is required and must point OUTSIDE the repository
(results stay private). ``--quick`` runs 1 case, 3 angles and
1 timing repeat. The full run uses 4 cases, 33 angles per sweep
and medians of 5 repeats.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import ventorum as vt
from ventorum.utils.parallel import cpu_cores

from validation.speed import adapter_aerosandbox, adapter_avl, adapter_flow5
from validation.speed import adapter_machupx, adapter_ventorum, adapter_vspaero
from validation.speed import cases
from validation.speed.base import (
    REPEATS_FULL,
    REPEATS_QUICK,
    SweepResult,
    has_converged,
    median_seconds,
    refuse_inside_repository,
    scrub_home,
)

ROOT = Path(__file__).resolve().parents[2]

#: Adapter registry by tool name.
ADAPTERS: dict[str, object] = {
    "avl": adapter_avl,
    "vspaero": adapter_vspaero,
    "flow5": adapter_flow5,
    "aerosandbox": adapter_aerosandbox,
    "machupx": adapter_machupx,
    "ventorum": adapter_ventorum,
}

#: Default timeout [s] of one external sweep.
DEFAULT_TIMEOUT_S = 120.0

#: Tools that take the tool path as first sweep argument.
_EXE_FIRST = ("avl", "vspaero", "flow5", "aerosandbox", "machupx")


def _tool_name(adapter: object) -> str:
    """Return the tool name of an adapter module or stub."""
    name = getattr(adapter, "TOOL_NAME", None)
    return str(name) if name else "unknown"


def _call_run_sweep(adapter: object, explicit, aircraft, mesh: int,
                    angles: list[float], workdir: Path | None,
                    timeout: float) -> SweepResult:
    """Call the run_sweep of *adapter* with its own argument shape."""
    if _tool_name(adapter) in _EXE_FIRST:
        return adapter.run_sweep(explicit, aircraft, mesh, angles, workdir, timeout)  # type: ignore[attr-defined]
    try:
        return adapter.run_sweep(aircraft, mesh, angles, workdir, timeout)  # type: ignore[attr-defined]
    except TypeError:
        return adapter.run_sweep(aircraft, mesh, angles)  # type: ignore[attr-defined]


def _call_time_sweeps(adapter: object, explicit, aircraft, mesh: int,
                      angles: list[float], repeats: int, timeout: float,
                      workdir: Path | None) -> tuple[float, list[float]]:
    """Call the time_sweeps of *adapter*. Return (median [s], samples [s])."""
    name = _tool_name(adapter)
    if name in _EXE_FIRST:
        median, samples, _failed = adapter.time_sweeps(  # type: ignore[attr-defined]
            explicit, aircraft, mesh, angles, repeats, timeout, workdir)
        return float(median), list(samples)
    try:
        median, samples = adapter.time_sweeps(aircraft, mesh, angles, repeats)  # type: ignore[attr-defined]
    except TypeError:
        median, samples = adapter.time_sweeps(  # type: ignore[attr-defined]
            aircraft, mesh, angles, repeats, timeout, workdir)
        median = float(median) if not isinstance(median, (list, tuple)) else median_seconds(list(median))
        if isinstance(samples, (int, float)):
            samples = [float(samples)]
        else:
            samples = list(samples)
    return float(median), list(samples)


def converge_tool(adapter: object, aircraft, angles_deg: list[float],
                  workdir: Path | None, timeout: float,
                  explicit=None) -> dict:
    """Find the converged mesh of one tool on one case.

    Each mesh in ``adapter.MESHES`` runs once. The first mesh whose
    CL and CDi sweeps both change by less than 0.5 % against the
    previous mesh wins. Without convergence the finest mesh is kept
    with a note. A skipped or failed sweep gives a record with
    ``skipped`` or ``ok`` set accordingly and no timing.
    """
    name = _tool_name(adapter)
    angles = [float(a) for a in angles_deg]
    meshes: list[int] = list(adapter.MESHES)  # type: ignore[attr-defined]
    good: list[SweepResult] = []
    for mesh in meshes:
        current = _call_run_sweep(adapter, explicit, aircraft, mesh, angles,
                                  workdir, timeout)
        if current.skipped:
            return {
                "tool": name, "skipped": True, "ok": False,
                "reason": current.reason, "mesh": None, "mesh_label": None,
                "converged": None, "cl_change": None, "cdi_change": None,
                "note": None, "median_s": None, "samples_s": [],
                "repeats": 0, "angles_deg": angles, "CL": [], "CDi": [],
            }
        if not current.ok:
            return {
                "tool": name, "skipped": False, "ok": False,
                "reason": current.reason, "mesh": int(mesh),
                "mesh_label": current.mesh_label or adapter.describe_mesh(mesh),  # type: ignore[attr-defined]
                "converged": None, "cl_change": None, "cdi_change": None,
                "note": None, "median_s": None, "samples_s": [],
                "repeats": 0, "angles_deg": angles, "CL": list(current.CL),
                "CDi": list(current.CDi),
            }
        if good:
            ok, cl_err, cdi_err = has_converged(good[-1].CL, good[-1].CDi,
                                                current.CL, current.CDi)
            if ok:
                return {
                    "tool": name, "skipped": False, "ok": True, "reason": None,
                    "mesh": int(mesh), "mesh_label": current.mesh_label,
                    "converged": True, "cl_change": cl_err, "cdi_change": cdi_err,
                    "note": None, "median_s": None, "samples_s": [],
                    "repeats": 0, "angles_deg": angles,
                    "CL": list(current.CL), "CDi": list(current.CDi),
                }
        good.append(current)
    current = good[-1]
    if len(good) >= 2:
        _ok, last_cl, last_cdi = has_converged(good[-2].CL, good[-2].CDi,
                                               current.CL, current.CDi)
    else:
        last_cl, last_cdi = None, None
    return {
        "tool": name, "skipped": False, "ok": True, "reason": None,
        "mesh": int(current.mesh or meshes[-1]), "mesh_label": current.mesh_label,
        "converged": False, "cl_change": last_cl, "cdi_change": last_cdi,
        "note": ("Mesh is not converged at the finest level; "
                 "the finest mesh is kept and timed."),
        "median_s": None, "samples_s": [], "repeats": 0,
        "angles_deg": angles, "CL": list(current.CL), "CDi": list(current.CDi),
    }


def _git_commit() -> str:
    """Return the short git commit of the repository, or 'unknown'."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _fmt(value: float | None, digits: int = 4) -> str:
    """Format a number for the tables ('n/a' for no value)."""
    if value is None:
        return "n/a"
    if abs(value) < 0.5 * 10.0 ** -digits:
        value = 0.0
    return f"{value:.{digits}f}"


def write_markdown(meta: dict, case_records: list[dict]) -> str:
    """Render the Markdown report from *meta* and *case_records*."""
    parts = [
        "# Cross-tool speed at equal accuracy (black box)",
        "",
        f"Date: {meta['date']}",
        f"Ventorum version: {meta['ventorum_version']}, git commit {meta['git_commit']}",
        f"CPU cores: {meta['cpu_cores']}",
        f"Angles [deg]: {', '.join(f'{a:g}' for a in meta['angles_deg'])}",
        f"Timing repeats: {meta['repeats']} (each cell shows the median)",
        "",
        "Fairness rules: the same machine runs every tool; each tool "
        "uses its best documented settings; Ventorum runs with its "
        "defaults (no manual tuning). Python tools run in a subprocess "
        "(one script per case set) so their import time and threads do "
        "not mix with Ventorum. Timings need a quiet machine; treat "
        "small differences as noise.",
        "",
        "Equal accuracy: each tool runs at the mesh where its CL and "
        "CDi sweeps both change by less than 0.5 % on refinement "
        "(change at each angle divided by the peak absolute value of "
        "the fine sweep; the largest over the sweep must pass). The "
        "tables give the mesh and the two changes. A tool without a "
        "converged mesh keeps its finest mesh with a note. Missing "
        "tools are skipped with the variable that would point to them. "
        "Results stay private until the owner decides.",
        "",
    ]
    for case in case_records:
        parts += [f"## {case['label']} ({case['id']})", "",
                  "| tool | mesh | converged | CL_change | CDi_change | median_s | samples_s |",
                  "|---|---|---|---|---|---|---|"]
        notes: list[str] = []
        for name in meta["tools"]:
            entry = case["tools"].get(name, {})
            if entry.get("skipped"):
                parts.append(f"| {name} | skipped | n/a | n/a | n/a | n/a | n/a |")
                if entry.get("reason"):
                    notes.append(f"{name}: {entry['reason']}")
                continue
            if not entry.get("ok", False):
                parts.append(f"| {name} | {entry.get('mesh_label', 'n/a')} "
                             "| failed | n/a | n/a | n/a | n/a |")
                if entry.get("reason"):
                    notes.append(f"{name}: {entry['reason']}")
                continue
            samples_s = ", ".join(f"{s:.3f}" for s in entry.get("samples_s", []))
            parts.append(
                f"| {name} | {entry.get('mesh_label', 'n/a')} "
                f"| {str(entry.get('converged')).lower()} "
                f"| {_fmt(entry.get('cl_change'))} | {_fmt(entry.get('cdi_change'))} "
                f"| {_fmt(entry.get('median_s'), 3)} | {samples_s} |"
            )
            if entry.get("note"):
                notes.append(f"{name}: {entry['note']}")
        if notes:
            parts += ["", "Notes:", ""] + [f"* {note}" for note in notes]
        parts += [""]
    return "\n".join(parts) + "\n"


def run_all(out: Path, quick: bool, tools: list[str], timeout: float,
            repeats: int, explicit: dict[str, str | None]) -> tuple[dict, list[dict]]:
    """Run every case and tool. Return (meta, case_records)."""
    angles = cases.sweep_angles(quick)
    specs = cases.case_list(quick)
    meta = {
        "date": datetime.date.today().isoformat(),
        "ventorum_version": vt.__version__,
        "git_commit": _git_commit(),
        "cpu_cores": cpu_cores(),
        "angles_deg": angles,
        "tools": list(tools),
        "repeats": int(repeats),
        "quick": bool(quick),
    }
    case_records: list[dict] = []
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        aircraft = spec["build"]()
        aircraft.compute_reference_values()
        tool_entries: dict[str, dict] = {}
        for name in tools:
            adapter = ADAPTERS[name]
            folder = runs / f"{spec['id']}_{name}"
            folder.mkdir(parents=True, exist_ok=True)
            try:
                record = converge_tool(adapter, aircraft, angles, folder,
                                       timeout, explicit.get(name))
            except Exception as exc:  # noqa: BLE001 - a failed tool is a failed entry
                record = {
                    "tool": name, "skipped": False, "ok": False,
                    "reason": scrub_home(exc), "mesh": None, "mesh_label": None,
                    "converged": None, "cl_change": None, "cdi_change": None,
                    "note": None, "median_s": None, "samples_s": [],
                    "repeats": 0, "angles_deg": angles, "CL": [], "CDi": [],
                }
            if record.get("ok") and not record.get("skipped"):
                try:
                    median, samples = _call_time_sweeps(
                        adapter, explicit.get(name), aircraft,
                        int(record["mesh"]), angles, int(repeats), timeout, folder)
                    record["median_s"] = median
                    record["samples_s"] = samples
                    record["repeats"] = int(repeats)
                except Exception as exc:  # noqa: BLE001 - timing failure keeps convergence
                    record["median_s"] = None
                    record["samples_s"] = []
                    record["note"] = "; ".join(filter(None, [
                        record.get("note"), f"Timing failed: {scrub_home(exc)}"]))
            tool_entries[name] = record
        case_records.append({"id": spec["id"], "label": spec["label"], "tools": tool_entries})
    return meta, case_records


def main(argv: list[str] | None = None) -> int:
    """Entry point of the speed harness."""
    parser = argparse.ArgumentParser(
        description="Time each low-order tool and Ventorum at equal accuracy.")
    parser.add_argument("--out", required=True,
                        help="Output folder OUTSIDE the repository.")
    parser.add_argument("--quick", action="store_true",
                        help="1 case, 3 angles, 1 timing repeat.")
    parser.add_argument("--tools", default=None,
                        help="Comma-separated tool names (default: all).")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        help="Timeout [s] of one external sweep.")
    parser.add_argument("--repeats", type=int, default=None,
                        help="Timing repeats (default: 1 when quick, else 5).")
    parser.add_argument("--avl", default=None, help="Path of avl.exe (or VENTORUM_AVL_EXE).")
    parser.add_argument("--vspaero", default=None,
                        help="Path of vspaero (or VENTORUM_VSPAERO).")
    parser.add_argument("--flow5", default=None, help="Path of flow5 (or VENTORUM_FLOW5).")
    parser.add_argument("--aerosandbox-python", default=None,
                        help="Interpreter of the AeroSandbox environment.")
    parser.add_argument("--machupx-python", default=None,
                        help="Interpreter of the MachUpX environment.")
    args = parser.parse_args(argv)

    if args.tools:
        tools = [t.strip().lower() for t in args.tools.split(",") if t.strip()]
    else:
        tools = list(ADAPTERS)
    unknown = [t for t in tools if t not in ADAPTERS]
    if unknown:
        print(f"Unknown tools: {', '.join(unknown)}. Known: {', '.join(ADAPTERS)}.",
              file=sys.stderr)
        return 2
    try:
        out = refuse_inside_repository(Path(args.out), ROOT)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    repeats = args.repeats
    if repeats is None:
        repeats = REPEATS_QUICK if args.quick else REPEATS_FULL
    if repeats < 1:
        print("Repeats must be at least 1.", file=sys.stderr)
        return 2
    explicit = {
        "avl": args.avl,
        "vspaero": args.vspaero,
        "flow5": args.flow5,
        "aerosandbox": args.aerosandbox_python,
        "machupx": args.machupx_python,
        "ventorum": None,
    }

    out.mkdir(parents=True, exist_ok=True)
    meta, case_records = run_all(out, args.quick, tools, float(args.timeout),
                                 int(repeats), explicit)
    (out / "speed_comparison.md").write_text(write_markdown(meta, case_records),
                                             encoding="utf-8")
    with open(out / "speed_comparison.json", "w", encoding="utf-8") as handle:
        json.dump({"meta": meta, "cases": case_records}, handle, indent=1, default=str)
    print(f"Wrote {out / 'speed_comparison.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
