# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""VSPAERO adapter of the speed harness (black box).

VSPAERO is the vortex-lattice program of OpenVSP. The harness reads
the ``vspaero`` program path from ``--vspaero`` or ``VENTORUM_VSPAERO``
and the OpenVSP Python interpreter from ``VENTORUM_VSP_PYTHON`` (that
interpreter is separate, so it runs as a subprocess). The ``vsp``
program is the sibling of ``vspaero`` in the same folder. No VSPAERO
or OpenVSP source code is read.

Run shape: the helper :mod:`validation.speed.vsp_make_geom` builds the
``.vsp3`` geometry in a ``py -3.13`` subprocess; one generated
``.vspscript`` analysis then runs with ``vsp -script`` in the same
folder, and the adapter reads CL from ``CLtot`` and CDi from ``CDi``
of the ``.polar`` file. The ``.vspaero`` settings echo and the
``.polar`` header are checked (wrong reference values or angles fail
the run, never silent zeros).

Settings, each from the public OpenVSP documentation (published
``Swept_Wing_API`` tutorial with ``TR1208.vspscript``,
``TestAnalysisVSPAERO.vspscript`` and ``Wing.vspscript`` examples,
plus the API documentation):

* Vortex-lattice mode: thin geometry set only (``GeomSet`` none,
  ``ThinGeomSet`` all) in ``VSPAEROComputeGeometry`` and
  ``VSPAEROSweep`` (degenerate geometry, as the documentation
  describes).
* Free air at Mach 0 (the documented default; the program reports
  Mach 0.001, still incompressible).
* The alpha list of the case in one run (uniform
  ``AlphaStart``/``AlphaEnd``/``AlphaNpts`` range); a non-uniform
  angle list runs one single-point sweep per angle at the exact
  requested angles.
* Symmetry off (0): the documented X-Z symmetry option was tested and
  is not used (see the harness README).
* Thread option ``NCPU`` set to all CPU cores.
* Default wake iterations (not set; the ``.vspaero`` echo records 5).
* Geometry from a ``.vsp3`` file made by the OpenVSP Python API in a
  ``py -3.13`` subprocess. Mesh refinement by the documented
  tessellation settings of the wing: ``SectTess_U`` per section
  (uniform spanwise density over the surfaces), chordwise
  tessellation at its documented default.
* Reference values ``Sref``/``bref``/``cref`` from the case
  (``RefFlag``/``WingID`` are omitted: with them the program
  overrides the given values).

A missing program gives a skipped result with the variable name.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from validation.speed import case_spec
from validation.speed.base import SweepResult, failed_result, median_seconds, skipped_result
from validation.speed.base import find_tool as _find_base
from validation.speed.base import scrub_home
from validation.speed.cases import V_INF as V_INF_M_S

#: Tool name in the reports.
TOOL_NAME = "vspaero"

#: Environment variables that hold the program paths.
ENV_VARS = ("VENTORUM_VSPAERO", "VENTORUM_VSP_PYTHON")

#: No ground effect runs (free air only).
SUPPORTS_GROUND_EFFECT = False

#: Spanwise tessellation of the main surface in the convergence study.
MESHES = [20, 40, 80]

#: Geometry helper (runs under the OpenVSP Python interpreter).
GEOM_HELPER = Path(__file__).resolve().parent / "vsp_make_geom.py"


def find_tool(explicit: str | None = None) -> Path | None:
    """Locate the VSPAERO program (explicit path first, then the variables)."""
    return _find_base(explicit, (ENV_VARS[0],))


def is_available(explicit: str | None = None) -> bool:
    """Return True when a VSPAERO program file exists."""
    return find_tool(explicit) is not None


def _find_vsp(vspaero_exe: Path) -> Path | None:
    """Return the ``vsp`` sibling of the VSPAERO program, or None."""
    name = "vsp.exe" if sys.platform == "win32" else "vsp"
    candidate = Path(vspaero_exe).parent / name
    return candidate if candidate.is_file() else None


def _find_vsp_python() -> Path | None:
    """Return the OpenVSP Python interpreter file, or None."""
    return _find_base(None, (ENV_VARS[1],))


def describe_mesh(mesh: int) -> str:
    """Return a human readable mesh label."""
    return f"SectTess_U {int(mesh)}, Tess_W default, sym 0"


def _run_capture(cmd: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run *cmd* in *cwd* and return the completed process (text output)."""
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=float(timeout), cwd=str(cwd),
    )


def analysis_script_text(vsp3_path: str, stem_angles: list[float],
                         s_ref: float, b_ref: float, c_ref: float,
                         ncpu: int) -> str:
    """Return the ``.vspscript`` text of one VSPAERO sweep.

    A single uniform range runs the whole angle list at once; a
    one-angle list runs a single point.
    """
    first, last = stem_angles[0], stem_angles[-1]
    lines = [
        "void main()",
        "{",
        f'    ReadVSPFile("{vsp3_path}");',
        "    Update();",
        '    string comp = "VSPAEROComputeGeometry";',
        "    SetAnalysisInputDefaults(comp);",
        "    array<int> noSet = {-1};",
        "    array<int> allSet = {1};",
        '    SetIntAnalysisInput(comp, "GeomSet", noSet);',
        '    SetIntAnalysisInput(comp, "ThinGeomSet", allSet);',
        "    ExecAnalysis(comp);",
        '    string sweep = "VSPAEROSweep";',
        "    SetAnalysisInputDefaults(sweep);",
        '    SetIntAnalysisInput(sweep, "GeomSet", noSet);',
        '    SetIntAnalysisInput(sweep, "ThinGeomSet", allSet);',
        f"    array<double> aStart = {{{first!r}}};",
        f"    array<double> aEnd = {{{last!r}}};",
        f"    array<int> aNpts = {{{len(stem_angles)}}};",
        '    SetDoubleAnalysisInput(sweep, "AlphaStart", aStart);',
        '    SetDoubleAnalysisInput(sweep, "AlphaEnd", aEnd);',
        '    SetIntAnalysisInput(sweep, "AlphaNpts", aNpts);',
        "    array<double> mStart = {0.0};",
        "    array<double> mEnd = {0.0};",
        "    array<int> mNpts = {1};",
        '    SetDoubleAnalysisInput(sweep, "MachStart", mStart);',
        '    SetDoubleAnalysisInput(sweep, "MachEnd", mEnd);',
        '    SetIntAnalysisInput(sweep, "MachNpts", mNpts);',
        f"    array<double> vinf = {{{V_INF_M_S!r}}};",
        '    SetDoubleAnalysisInput(sweep, "Vinf", vinf);',
        f"    array<double> sref = {{{s_ref!r}}};",
        f"    array<double> bref = {{{b_ref!r}}};",
        f"    array<double> cref = {{{c_ref!r}}};",
        '    SetDoubleAnalysisInput(sweep, "Sref", sref);',
        '    SetDoubleAnalysisInput(sweep, "bref", bref);',
        '    SetDoubleAnalysisInput(sweep, "cref", cref);',
        "    array<int> sym = {0};",
        '    SetIntAnalysisInput(sweep, "Symmetry", sym);',
        f"    array<int> ncpu = {{{int(ncpu)}}};",
        '    SetIntAnalysisInput(sweep, "NCPU", ncpu);',
        "    ExecAnalysis(sweep);",
        '    Print("harness sweep done");',
        "    while (GetNumTotalErrors() > 0)",
        "    {",
        "        ErrorObj err = PopLastError();",
        "        Print(err.GetErrorString());",
        "    }",
        "    VSPExit(0);",
        "}",
        "",
    ]
    return "\n".join(lines)


def parse_polar(text: str) -> dict[float, tuple[float, float]]:
    """Return {angle [deg]: (CL, CDi)} of a ``.polar`` file.

    Columns are found by header name (``AoA``, ``CLtot``, ``CDi``),
    so column order does not matter. Raise ValueError when a column
    or a data value is missing.
    """
    rows = [line.split() for line in text.splitlines() if line.split()]
    header_at = next(
        (i for i, row in enumerate(rows) if "AoA" in row and "CLtot" in row), None
    )
    if header_at is None:
        raise ValueError("The .polar file holds no AoA/CLtot header.")
    header = rows[header_at]
    try:
        ia = header.index("AoA")
        icl = header.index("CLtot")
        icdi = header.index("CDi")
    except ValueError as exc:
        raise ValueError("The .polar header misses AoA, CLtot or CDi.") from exc
    out: dict[float, tuple[float, float]] = {}
    for row in rows[header_at + 1:]:
        try:
            out[float(row[ia])] = (float(row[icl]), float(row[icdi]))
        except (IndexError, ValueError) as exc:
            raise ValueError("The .polar file holds a bad data row.") from exc
    if not out:
        raise ValueError("The .polar file holds no data rows.")
    return out


def parse_vspaero_echo(text: str) -> dict[str, object]:
    """Return the settings of a ``.vspaero`` echo file.

    Keys: ``Sref``, ``Mach``, ``AoA`` (list [deg]), ``Symmetry``,
    ``WakeIters``, ``Vinf``. Raise ValueError on a bad file.
    """
    out: dict[str, object] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key == "Sref":
            out["Sref"] = float(value)
        elif key == "Mach":
            out["Mach"] = float(value)
        elif key == "AoA":
            out["AoA"] = [float(v) for v in value.split(",") if v.strip()]
        elif key == "Symmetry":
            out["Symmetry"] = int(float(value))
        elif key == "WakeIters":
            out["WakeIters"] = int(float(value))
        elif key == "Vinf":
            out["Vinf"] = float(value)
    if "Sref" not in out or "AoA" not in out:
        raise ValueError("The .vspaero echo holds no Sref or AoA.")
    return out


def _jobs(angles: list[float]) -> list[list[float]]:
    """Split *angles* into sweep runs (one range, or one point each)."""
    if case_spec.is_uniform_angles(angles):
        return [list(angles)]
    return [[a] for a in angles]


def run_sweep(vspaero_exe: str | Path | None, aircraft, mesh: int,
              angles_deg: list[float], workdir: str | Path | None,
              timeout: float = 120.0) -> SweepResult:
    """Run one VSPAERO sweep (geometry plus analysis) and time it.

    Parameters
    ----------
    vspaero_exe : str, Path or None
        Path of ``vspaero``. None or a wrong path gives a skipped
        or failed result, never an exception.
    aircraft : Aircraft
        Case geometry.
    mesh : int
        Spanwise tessellation (``SectTess_U``) of the largest surface;
        smaller surfaces scale by semi-span.
    angles_deg : list of float
        Sweep angles [deg].
    workdir : str, Path or None
        Folder for the run files (spec, ``.vsp3``, scripts, outputs;
        all stay there as the settings record). None uses a temporary
        folder that is discarded.
    timeout : float
        Wall-time limit [s] of each subprocess call.

    Returns
    -------
    SweepResult
        CL [-] from ``CLtot`` and CDi [-] from ``CDi`` per angle,
        with the wall time [s] of the whole sweep.
    """
    import tempfile

    case_id = getattr(aircraft, "name", "")
    angles = [float(a) for a in angles_deg]
    if not angles:
        return failed_result(TOOL_NAME, case_id, describe_mesh(mesh), angles, "No angles given.")
    exe = Path(vspaero_exe) if vspaero_exe is not None else find_tool(None)
    if exe is None or not Path(exe).is_file():
        return skipped_result(TOOL_NAME, case_id,
                              "VSPAERO not found: set VENTORUM_VSPAERO or pass --vspaero PATH.")
    vsp = _find_vsp(exe)
    if vsp is None:
        return skipped_result(TOOL_NAME, case_id,
                              "The vsp program is missing next to VENTORUM_VSPAERO; "
                              "both come from the OpenVSP install.")
    vspy = _find_vsp_python()
    if vspy is None or not vspy.is_file():
        return skipped_result(TOOL_NAME, case_id,
                              "OpenVSP Python not found: set VENTORUM_VSP_PYTHON "
                              "to the interpreter with the OpenVSP API.")
    try:
        spec = case_spec.aircraft_spec(aircraft, mesh, angles, V_INF_M_S)
    except ValueError as exc:
        return failed_result(TOOL_NAME, case_id, describe_mesh(mesh), angles, scrub_home(exc))
    tess = case_spec.scaled_counts(spec, mesh)
    for surf in spec["surfaces"]:
        surf["tess_u"] = tess[surf["name"]]
    ncpu = os.cpu_count() or 1
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ventorum_speed_vsp_") as tmp:
        root = Path(workdir) if workdir is not None else Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        stem = f"{spec['case_id']}_m{int(mesh)}"
        (root / "case.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        vsp3 = root / f"{stem}.vsp3"
        info_path = root / "geom_info.json"
        try:
            done = _run_capture(
                [str(vspy), str(GEOM_HELPER), "--spec", str(root / "case.json"),
                 "--out", str(vsp3), "--info", str(info_path)],
                root, timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return failed_result(TOOL_NAME, case_id, describe_mesh(mesh), angles, scrub_home(exc))
        if done.returncode != 0 or not vsp3.is_file():
            tail = ((done.stderr or "") + (done.stdout or ""))[-500:]
            return failed_result(
                TOOL_NAME, case_id, describe_mesh(mesh), angles,
                f"Geometry build failed (code {done.returncode}): {scrub_home(tail.strip())}")
        try:
            version = str(json.loads(info_path.read_text(encoding="utf-8"))["tool_version"])
        except (OSError, ValueError, KeyError) as exc:
            return failed_result(TOOL_NAME, case_id, describe_mesh(mesh), angles, scrub_home(exc))
        vsp3_posix = vsp3.as_posix()
        found: dict[float, tuple[float, float]] = {}
        jobs = _jobs(angles)
        for job_index, job in enumerate(jobs):
            script = root / ("sweep.vspscript" if len(jobs) == 1
                             else f"sweep_a{job_index:02d}.vspscript")
            script.write_text(
                analysis_script_text(vsp3_posix, job, spec["s_ref_m2"],
                                     spec["b_ref_m"], spec["c_ref_m"], ncpu),
                encoding="utf-8",
            )
            try:
                done = _run_capture([str(vsp), "-script", str(script)], root, timeout)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return failed_result(
                    TOOL_NAME, case_id, describe_mesh(mesh), angles, scrub_home(exc))
            if done.returncode != 0:
                tail = ((done.stderr or "") + (done.stdout or ""))[-500:]
                return failed_result(
                    TOOL_NAME, case_id, describe_mesh(mesh), angles,
                    f"VSPAERO run failed (code {done.returncode}): {scrub_home(tail.strip())}")
            polar = root / f"{stem}.polar"
            echo = root / f"{stem}.vspaero"
            try:
                table = parse_polar(polar.read_text(encoding="utf-8"))
                settings = parse_vspaero_echo(echo.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                return failed_result(
                    TOOL_NAME, case_id, describe_mesh(mesh), angles, scrub_home(exc))
            if abs(float(settings["Sref"]) - spec["s_ref_m2"]) > 1e-9 * spec["s_ref_m2"]:
                return failed_result(
                    TOOL_NAME, case_id, describe_mesh(mesh), angles,
                    f"VSPAERO ran with Sref {settings['Sref']}, "
                    f"not the case {spec['s_ref_m2']}.")
            for angle in job:
                hit = next((v for k, v in table.items() if abs(k - angle) <= 1e-6), None)
                if hit is None:
                    return failed_result(
                        TOOL_NAME, case_id, describe_mesh(mesh), angles,
                        f"The .polar file misses angle {angle:g} deg.")
                found[angle] = hit
    elapsed = time.perf_counter() - t0
    cl = [found[a][0] for a in angles]
    cdi = [found[a][1] for a in angles]
    return SweepResult(
        tool=TOOL_NAME, case_id=case_id,
        mesh_label=f"{describe_mesh(mesh)}, {version}, NCPU {ncpu}",
        mesh=int(mesh), angles_deg=angles, CL=cl, CDi=cdi,
        ok=True, skipped=False, reason=None, elapsed_s=elapsed,
    )


def time_sweeps(vspaero_exe: str | Path | None, aircraft, mesh: int,
                angles_deg: list[float], repeats: int, timeout: float = 120.0,
                workdir: str | Path | None = None) -> tuple[float, list[float], int]:
    """Time *repeats* VSPAERO sweeps. Return (median [s], samples [s], failed points)."""
    import tempfile

    samples: list[float] = []
    failed = 0
    with tempfile.TemporaryDirectory(prefix="ventorum_speed_vsptime_") as tmp:
        root = Path(workdir) if workdir is not None else Path(tmp)
        for rep in range(int(repeats)):
            res = run_sweep(vspaero_exe, aircraft, mesh, list(angles_deg),
                            root / f"rep{rep:02d}", timeout)
            samples.append(res.elapsed_s)
            failed += sum(1 for v in res.CL if v is None)
            if not res.ok and not res.skipped:
                failed += len(res.angles_deg) - sum(1 for v in res.CL if v is None)
    return median_seconds(samples), samples, failed
