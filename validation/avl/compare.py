# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Compare Ventorum against AVL as a black box.

The program runs the same cases in Ventorum and in AVL (the
vortex-lattice program of M. Drela and H. Youngren), and writes tables
of the differences (forces, moments, static stability derivatives)
and of the run times. AVL is used as a black box only: the harness
writes AVL input files, runs ``avl.exe``, and reads its output files.
No AVL source code is read or copied; the harness is written from the
AVL user documentation (see ``validation/avl/README.md``).

Usage
-----
python validation/avl/compare.py --avl PATH --out DIR [--quick]
    [--workers N] [--timeout S]

``--out`` is required and must point OUTSIDE the repository (results
are private). ``--quick`` runs 2 cases, 3 angles and 1 timing repeat.
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import ventorum as vt
from ventorum.core.datatypes import Aircraft, FlightCondition, SolverSettings
from ventorum.utils.parallel import cpu_cores
from validation.avl.avl_files import build_case
from validation.avl.avl_run import DEFAULT_TIMEOUT_S, find_avl, run_case

ROOT = Path(__file__).resolve().parents[2]

#: Full angle sweep [deg].
FULL_ANGLES = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
#: Quick angle sweep [deg] (spans the range, includes a negative angle).
QUICK_ANGLES = [-4.0, 4.0, 10.0]
#: Base angle [deg] of the static derivatives.
ALPHA0_DEG = 4.0
#: Offsets [deg] of the central differences (alpha +-0.5, beta +-1).
DALPHA_DEG = 0.5
DBETA_DEG = 1.0
#: Spanwise meshes.
FULL_MESHES = [20, 40]
QUICK_MESHES = [20]
#: Chordwise panels of every case.
N_CHORD = 4
#: Free-stream velocity [m/s] of every case.
V_INF = 50.0

#: Force and moment variables of the sweep tables.
FORCE_VARS = ("CL", "CDi", "CY", "Cl", "Cm", "Cn")
#: Static derivatives compared (stability axes, per rad).
DERIV_VARS = ("CL_alpha", "Cm_alpha", "CY_beta", "Cl_beta", "Cn_beta")
#: Rate derivatives parsed from AVL (Ventorum has none yet).
RATE_VARS = (
    "CLp", "CLq", "CLr", "CYp", "CYq", "CYr", "CDp", "CDq", "CDr",
    "Clp", "Clq", "Clr", "Cmp", "Cmq", "Cmr", "Cnp", "Cnq", "Cnr",
)


# --------------------------------------------------------------------------
# Case geometries (synthetic only, no reference data)
# --------------------------------------------------------------------------

def build_rect() -> Aircraft:
    """Rectangular wing, aspect ratio 8, chord 1 m."""
    surf = vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
    )
    return Aircraft(name="rect_AR8", surfaces=[surf])


def build_swept() -> Aircraft:
    """Swept (30 deg) tapered (0.5) wing, aspect ratio 6."""
    surf = vt.LiftingSurface(
        name="Wing",
        semi_span=3.0,
        sweep_le=np.radians(30.0),
        sections=[vt.WingSection(0.0, 4.0 / 3.0), vt.WingSection(1.0, 2.0 / 3.0)],
    )
    return Aircraft(name="swept_tapered", surfaces=[surf])


def build_dihedral() -> Aircraft:
    """Rectangular wing, aspect ratio 8, with 5 deg dihedral."""
    surf = vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        dihedral=np.radians(5.0),
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
    )
    return Aircraft(name="dihedral", surfaces=[surf])


def build_wing_tail() -> Aircraft:
    """Rectangular wing (AR 8) with a horizontal tail."""
    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
    )
    tail = vt.LiftingSurface(
        name="HTail",
        semi_span=1.6,
        position=np.array([4.5, 0.0, 0.3]),
        sections=[vt.WingSection(0.0, 0.8), vt.WingSection(1.0, 0.6)],
    )
    return Aircraft(name="wing_tail", surfaces=[wing, tail])


def build_wing_tail_fin() -> Aircraft:
    """Rectangular wing (AR 8) with a horizontal tail and a vertical fin."""
    ac = build_wing_tail()
    ac.name = "wing_tail_fin"
    fin = vt.LiftingSurface(
        name="Fin",
        semi_span=1.2,
        dihedral=np.radians(90.0),
        is_symmetric=False,
        position=np.array([4.5, 0.0, 0.3]),
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 0.6)],
    )
    ac.surfaces.append(fin)
    return ac


def case_list(quick: bool) -> list[dict]:
    """Return the case geometries of this run."""
    cases = [
        {"id": "rect", "label": "Rectangular wing AR 8", "build": build_rect, "h": None},
        {"id": "swept", "label": "Swept tapered wing AR 6", "build": build_swept, "h": None},
        {"id": "dihedral", "label": "Wing with 5 deg dihedral", "build": build_dihedral, "h": None},
        {"id": "wingtail", "label": "Wing and horizontal tail", "build": build_wing_tail, "h": None},
        {"id": "wingtailfin", "label": "Wing, tail and vertical fin",
         "build": build_wing_tail_fin, "h": None},
        {"id": "ge_h10", "label": "Rectangular wing in ground effect h/c = 1.0",
         "build": build_rect, "h": 1.0},
        {"id": "ge_h05", "label": "Rectangular wing in ground effect h/c = 0.5",
         "build": build_rect, "h": 0.5},
    ]
    if quick:
        return [cases[0], cases[5]]
    return cases


# --------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------

def ventorum_settings(mesh: int, wake: str) -> SolverSettings:
    """Solver settings of one Ventorum column."""
    return SolverSettings(solver_type="vlm", n_panels=mesh, n_chord=N_CHORD,
                          wake_alignment=wake)


def solve_ventorum(ac: Aircraft, mesh: int, wake: str, alpha_deg: float,
                   beta_deg: float = 0.0, h: float | None = None):
    """Run one Ventorum analysis."""
    condition = FlightCondition(
        V_inf=V_INF, alpha=float(np.radians(alpha_deg)),
        beta=float(np.radians(beta_deg)), h=h,
    )
    return vt.analyze(ac, condition=condition, settings=ventorum_settings(mesh, wake))


def _triple_body(result) -> dict[str, float]:
    """Body-axis force triple (CL, CY, CDi, Cl, Cm, Cn)."""
    t = result.totals
    return {"CL": t.CL, "CDi": t.CDi, "CY": t.CY,
            "Cl": t.Cl, "Cm": t.Cm, "Cn": t.Cn}


def _triple_stability(result) -> dict[str, float]:
    """Stability-axis force triple."""
    t = result.totals
    m = result.moments("stability")
    return {"CL": t.CL, "CDi": t.CDi, "CY": t.CY,
            "Cl": m["Cl"], "Cm": m["Cm"], "Cn": m["Cn"]}


def _triple_avl(parsed: dict[str, float], stability: bool) -> dict[str, float]:
    """Force triple from a parsed AVL FT file."""
    if stability:
        cl, cm, cn = parsed["Cl_prim"], parsed["Cmtot"], parsed["Cn_prim"]
    else:
        cl, cm, cn = parsed["Cltot"], parsed["Cmtot"], parsed["Cntot"]
    return {"CL": parsed["CLtot"], "CDi": parsed["CDff"], "CY": parsed["CYff"],
            "Cl": cl, "Cm": cm, "Cn": cn}


def solve_avl(avl_exe: Path, alpha_deg: float, beta_deg: float, base_settings: SolverSettings,
              ac: Aircraft, h: float | None, folder: Path, timeout: float):
    """Write, run and parse one AVL case. Returns (AvlCase, AvlRunResult)."""
    case = build_case(ac, base_settings, alpha_deg=alpha_deg, beta_deg=beta_deg, ground_h=h)
    return case, run_case(avl_exe, case, folder, timeout=timeout)


def ventorum_derivatives(ac: Aircraft, mesh: int, wake: str, h: float | None) -> dict[str, float]:
    """Static derivatives by central differences, moments in stability axes."""
    da = float(np.radians(DALPHA_DEG))
    db = float(np.radians(DBETA_DEG))
    a0 = float(np.radians(ALPHA0_DEG))

    def stab(alpha: float, beta: float) -> tuple[float, dict[str, float]]:
        condition = FlightCondition(V_inf=V_INF, alpha=alpha, beta=beta, h=h)
        result = vt.analyze(ac, condition=condition, settings=ventorum_settings(mesh, wake))
        return result.totals.CL, result.moments("stability")

    def side(alpha: float, beta: float) -> float:
        condition = FlightCondition(V_inf=V_INF, alpha=alpha, beta=beta, h=h)
        return vt.analyze(ac, condition=condition,
                          settings=ventorum_settings(mesh, wake)).totals.CY

    cl_hi, m_hi = stab(a0 + da, 0.0)
    cl_lo, m_lo = stab(a0 - da, 0.0)
    _, mb_hi = stab(a0, db)
    _, mb_lo = stab(a0, -db)
    cy_hi = side(a0, db)
    cy_lo = side(a0, -db)
    return {
        "CL_alpha": (cl_hi - cl_lo) / (2.0 * da),
        "Cm_alpha": (m_hi["Cm"] - m_lo["Cm"]) / (2.0 * da),
        "CY_beta": (cy_hi - cy_lo) / (2.0 * db),
        "Cl_beta": (mb_hi["Cl"] - mb_lo["Cl"]) / (2.0 * db),
        "Cn_beta": (mb_hi["Cn"] - mb_lo["Cn"]) / (2.0 * db),
    }


def ventorum_xnp(ac: Aircraft, derivs: dict[str, float]) -> float | None:
    """Neutral point x [m] from the static derivatives."""
    if abs(derivs["CL_alpha"]) < 1e-12:
        return None
    ref = ac.moment_reference()
    cref = ac.c_ref if ac.c_ref is not None else 1.0
    return float(ref[0] - cref * derivs["Cm_alpha"] / derivs["CL_alpha"])


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def _median_seconds(samples: list[float]) -> float:
    """Median of wall-time samples [s]."""
    return float(statistics.median(samples))


def _avl_sweep_once(avl_exe: Path, ac: Aircraft, mesh: int, h: float | None,
                    angles: list[float], workers: int, timeout: float, folder: Path) -> float:
    """Run one AVL sweep and return its wall time [s]."""
    base = SolverSettings(solver_type="vlm", n_panels=mesh, n_chord=N_CHORD)
    t0 = time.perf_counter()

    def one(alpha_deg: float) -> None:
        sub = folder / f"a{alpha_deg:+07.2f}".replace(".", "p")
        case = build_case(ac, base, alpha_deg=alpha_deg, ground_h=h)
        run_case(avl_exe, case, sub, timeout=timeout)

    folder.mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        for alpha in angles:
            one(alpha)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, angles))
    return time.perf_counter() - t0


def _ventorum_sweep_once(ac: Aircraft, mesh: int, h: float | None,
                         angles: list[float]) -> float:
    """Run one Ventorum sweep and return its wall time [s]."""
    t0 = time.perf_counter()
    settings = ventorum_settings(mesh, "freestream")
    if h is None:
        vt.analyze_sweep(ac, np.array(angles, dtype=float), settings=settings)
    else:
        for alpha in angles:
            condition = FlightCondition(V_inf=V_INF, alpha=float(np.radians(alpha)), h=h)
            vt.analyze(ac, condition=condition, settings=settings)
    return time.perf_counter() - t0


def time_device_sweep(ac: Aircraft, mesh: int, h: float | None,
                      angles: list[float], device: str, repeats: int) -> tuple[float, list[float]]:
    """Time Ventorum sweeps on *device*. Returns (median, samples)."""
    from ventorum import gpu

    samples = []
    for _ in range(repeats):
        old = gpu.get_device()
        gpu.set_device(device)
        try:
            samples.append(_ventorum_sweep_once(ac, mesh, h, angles))
        finally:
            gpu.set_device(old)
    return _median_seconds(samples), samples


# --------------------------------------------------------------------------
# Output refusal, header, tables
# --------------------------------------------------------------------------

def refuse_inside_repository(out: Path) -> Path:
    """Resolve *out* and refuse it when it lies inside the repository."""
    resolved = out.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError(
            f"Refused: --out {out} is inside the repository ({ROOT}). "
            "Results are private; give a folder OUTSIDE the repository."
        )
    return resolved


def _git_commit() -> str:
    """Short git commit of the repository, or 'unknown'."""
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


def _fmt(value: float | None, digits: int = 6) -> str:
    """Format a number for the markdown tables ('n/a' or 'failed')."""
    if value is None:
        return "n/a"
    if abs(value) < 0.5 * 10.0 ** -digits:
        value = 0.0
    return f"{value:.{digits}f}"


def _rel(value_v: float | None, value_a: float | None) -> float | None:
    """Relative difference (V - A) / |A|, or None when not defined."""
    if value_v is None or value_a is None or abs(value_a) <= 1e-12:
        return None
    return (value_v - value_a) / abs(value_a)


def _run_dir(runs: Path, set_id: str, mesh: int, alpha_deg: float, beta_deg: float) -> Path:
    """Folder of one stored AVL run (relative names only, no user paths)."""
    return runs / f"{set_id}_np{mesh}_a{alpha_deg:+07.2f}_b{beta_deg:+06.2f}".replace(".", "p")


def run_set(avl_exe: Path, spec: dict, mesh: int, angles: list[float],
            runs: Path, timeout: float) -> dict:
    """Run one case set in Ventorum and in AVL. Returns the JSON record."""
    ac = spec["build"]()
    ac.compute_reference_values()
    h = spec["h"]
    stability = h is not None
    set_id = spec["id"]
    base = SolverSettings(solver_type="vlm", n_panels=mesh, n_chord=N_CHORD)

    rows: list[dict] = []
    avl_alpha0_st: dict | None = None
    banner: str | None = None
    for alpha in angles:
        resolves: dict = {"alpha_deg": alpha}
        vent: dict = {}
        for wake in ("body", "freestream"):
            if h is not None and wake == "body":
                vent[wake] = None
                continue
            try:
                result = solve_ventorum(ac, mesh, wake, alpha, 0.0, h)
                vent[wake] = _triple_stability(result) if stability else _triple_body(result)
            except Exception as exc:  # noqa: BLE001 - a failed solve is a failed row
                vent[wake] = None
                resolves[f"ventorum_{wake}_error"] = str(exc)
        folder = _run_dir(runs, set_id, mesh, alpha, 0.0)
        try:
            case, avl_result = solve_avl(avl_exe, alpha, 0.0, base, ac, h, folder, timeout)
        except Exception as exc:  # noqa: BLE001 - a failed run is a failed row
            avl_result = None
            resolves["avl_error"] = str(exc)
        if avl_result is not None and avl_result.banner and banner is None:
            banner = avl_result.banner
        if avl_result is not None and avl_result.ok and avl_result.stabderivs is not None:
            triple = _triple_avl(avl_result.stabderivs, stability)
        else:
            triple = None
            if avl_result is not None and not avl_result.ok:
                resolves["avl_error"] = avl_result.error
        if alpha == ALPHA0_DEG and avl_result is not None and avl_result.ok:
            avl_alpha0_st = avl_result.stabderivs
        row = {"alpha_deg": alpha, "ventorum": vent, "avl": triple, "notes": []}
        row.update(resolves)
        rows.append(row)

    # Derivative extra runs (alpha +-0.5 deg, beta +-1 deg at ALPHA0_DEG).
    extras = [(ALPHA0_DEG + DALPHA_DEG, 0.0), (ALPHA0_DEG - DALPHA_DEG, 0.0),
              (ALPHA0_DEG, DBETA_DEG), (ALPHA0_DEG, -DBETA_DEG)]
    for alpha_e, beta_e in extras:
        folder = _run_dir(runs, set_id, mesh, alpha_e, beta_e)
        solve_avl(avl_exe, alpha_e, beta_e, base, ac, h, folder, timeout)

    derivs: dict = {}
    try:
        v_deriv_body = None if stability else ventorum_derivatives(ac, mesh, "body", h)
    except Exception as exc:  # noqa: BLE001
        v_deriv_body = None
        derivs["ventorum_body_error"] = str(exc)
    try:
        v_deriv_free = ventorum_derivatives(ac, mesh, "freestream", h)
    except Exception as exc:  # noqa: BLE001
        v_deriv_free = None
        derivs["ventorum_freestream_error"] = str(exc)
    derivs["ventorum_body"] = v_deriv_body
    derivs["ventorum_freestream"] = v_deriv_free
    if avl_alpha0_st is not None:
        derivs["avl"] = {
            "CL_alpha": avl_alpha0_st["CLa"], "Cm_alpha": avl_alpha0_st["Cma"],
            "CY_beta": avl_alpha0_st["CYb"], "Cl_beta": avl_alpha0_st["Clb"],
            "Cn_beta": avl_alpha0_st["Cnb"],
        }
        derivs["avl_rates"] = {name: avl_alpha0_st[name] for name in RATE_VARS}
        derivs["avl_xnp"] = avl_alpha0_st["Xnp"]
    else:
        derivs["avl"] = None
        derivs["avl_rates"] = None
        derivs["avl_xnp"] = None
        derivs["avl_error"] = "No good AVL run at alpha = 4 deg."
    xnp: dict = {"ventorum_body": None, "ventorum_freestream": None, "avl": derivs["avl_xnp"]}
    if v_deriv_free is not None:
        xnp["ventorum_freestream"] = ventorum_xnp(ac, v_deriv_free)
    if v_deriv_body is not None:
        xnp["ventorum_body"] = ventorum_xnp(ac, v_deriv_body)
    derivs["xnp"] = xnp

    return {
        "id": set_id, "label": spec["label"], "mesh": mesh, "h": h,
        "stability_moments": stability, "rows": rows, "derivatives": derivs,
        "banner": banner,
    }


def time_set(avl_exe: Path, spec: dict, mesh: int, angles: list[float],
             workers: int, timeout: float, repeats: int) -> dict:
    """Time one case set. Returns the JSON timing record."""
    ac = spec["build"]()
    h = spec["h"]
    avl_native, avl_enhanced, v_cpu, v_auto = [], [], [], []
    for rep in range(repeats):
        with tempfile.TemporaryDirectory(prefix="ventorum_avl_time_") as tmp:
            tmp_path = Path(tmp)
            avl_native.append(_avl_sweep_once(
                avl_exe, ac, mesh, h, angles, 1, timeout, tmp_path / f"n{rep}"))
            avl_enhanced.append(_avl_sweep_once(
                avl_exe, ac, mesh, h, angles, workers, timeout, tmp_path / f"e{rep}"))
        _, samples = time_device_sweep(ac, mesh, h, angles, "cpu", 1)
        v_cpu.extend(samples)
        _, samples = time_device_sweep(ac, mesh, h, angles, "auto", 1)
        v_auto.extend(samples)
    return {
        "avl_native_s": avl_native, "avl_native_median_s": _median_seconds(avl_native),
        "avl_enhanced_s": avl_enhanced, "avl_enhanced_median_s": _median_seconds(avl_enhanced),
        "ventorum_cpu_s": v_cpu, "ventorum_cpu_median_s": _median_seconds(v_cpu),
        "ventorum_auto_s": v_auto, "ventorum_auto_median_s": _median_seconds(v_auto),
        "workers": workers, "repeats": repeats,
    }


def write_markdown(meta: dict, sets: list[dict]) -> str:
    """Render the markdown report."""
    parts = [
        "# AVL comparison (black box)",
        "",
        f"Date: {meta['date']}",
        f"Ventorum version: {meta['ventorum_version']}, git commit {meta['git_commit']}",
        f"AVL: {meta['avl_banner']}",
        f"CPU cores: {meta['cpu_cores']}",
        "",
        "Method: one AVL SURFACE per Ventorum surface (YDUPLICATE about "
        "y = 0 for symmetric surfaces), one AVL SECTION per Ventorum "
        "section, AVL alpha and beta equal Ventorum alpha and beta in "
        "free air. In ground effect the geometry turns nose up by alpha "
        "about the reference point, AVL runs at alpha = 0 with a solid "
        "wall (iZsym = 1) at Zsym = Zref - h, and AVL body moments are "
        "Ventorum stability moments. Ventorum runs each point twice: "
        "with the body-axis wake (the same wake model as AVL) and with "
        "the default free-stream wake. In ground effect the body-wake "
        "column reads n/a. See validation/avl/README.md for the method.",
        "",
        "Timing: each value is the wall time of the whole sweep "
        "(writing the input files, the runs and reading the outputs). "
        f"Enhanced AVL used up to {meta['workers']} processes at the same "
        "time (ventorum.utils.parallel.cpu_cores(); this count can "
        "include hyper-threads). Ventorum ran analyze_sweep with its "
        "defaults (no manual tuning). Timings need a quiet machine; "
        "treat small differences as noise.",
        "",
    ]
    for record in sets:
        mesh = record["mesh"]
        parts += [f"## {record['label']} (n_panels {mesh}, n_chord {N_CHORD})", ""]
        if record["h"] is not None:
            parts += [f"Ground effect at h = {record['h']} m (geometry-rotation method).", ""]
        parts += ["| alpha_deg | variable | Ventorum_body | Ventorum_free | AVL "
                  "| diff_body | diff_free | rel_body_% | rel_free_% |",
                  "|" + "|".join(["---"] * 9) + "|"]
        for row in record["rows"]:
            body = row["ventorum"].get("body")
            free = row["ventorum"].get("freestream")
            avl = row["avl"]
            for var in FORCE_VARS:
                vv_b = body[var] if body else None
                vv_f = free[var] if free else None
                va_v = avl[var] if avl else None
                db = (vv_b - va_v) if (vv_b is not None and va_v is not None) else None
                df = (vv_f - va_v) if (vv_f is not None and va_v is not None) else None
                rb = _rel(vv_b, va_v)
                rf = _rel(vv_f, va_v)
                rb_s = f"{100.0 * rb:.2f}" if rb is not None else "n/a"
                rf_s = f"{100.0 * rf:.2f}" if rf is not None else "n/a"
                avl_s = _fmt(va_v) if avl is not None else "failed"
                parts.append(
                    f"| {row['alpha_deg']:.2f} | {var} | {_fmt(vv_b)} | {_fmt(vv_f)} "
                    f"| {avl_s} | {_fmt(db)} | {_fmt(df)} | {rb_s} | {rf_s} |"
                )
        parts += ["", "### Static derivatives (stability axes, per rad)", "",
                  "| derivative | Ventorum_body | Ventorum_free | AVL | diff_body | diff_free |",
                  "|---|---|---|---|---|---|"]
        derivs = record["derivatives"]
        for var in DERIV_VARS:
            vb = derivs["ventorum_body"]
            vf = derivs["ventorum_freestream"]
            va = derivs["avl"]
            vv_b = vb[var] if vb else None
            vv_f = vf[var] if vf else None
            va_v = va[var] if va else None
            db = (vv_b - va_v) if (vv_b is not None and va_v is not None) else None
            df = (vv_f - va_v) if (vv_f is not None and va_v is not None) else None
            avl_s = _fmt(va_v, 4) if va is not None else "failed"
            parts.append(
                f"| {var} | {_fmt(vv_b, 4)} | {_fmt(vv_f, 4)} | {avl_s} "
                f"| {_fmt(db, 4)} | {_fmt(df, 4)} |"
            )
        xn = derivs["xnp"]
        avl_x = _fmt(xn["avl"], 4) if xn["avl"] is not None else "failed"
        parts += [f"| xnp_m | {_fmt(xn['ventorum_body'], 4)} | "
                  f"{_fmt(xn['ventorum_freestream'], 4)} | {avl_x} | | |",
                  "",
                  "AVL rate derivatives (Ventorum: not available yet):",
                  ""]
        if derivs["avl_rates"] is not None:
            for name in RATE_VARS:
                parts.append(f"| {name} | {_fmt(derivs['avl_rates'][name], 4)} |")
        else:
            parts.append("No good AVL run at alpha = 4 deg.")
        parts += ["", "### Timing (median of repeats, wall seconds)", "",
                  "| method | median_s | samples_s |",
                  "|---|---|---|"]
        timing = record["timing"]
        for label, med, samples in (
            ("AVL native (one process after the other)", timing["avl_native_median_s"],
             timing["avl_native_s"]),
            (f"AVL enhanced ({timing['workers']} processes)", timing["avl_enhanced_median_s"],
             timing["avl_enhanced_s"]),
            ("Ventorum CPU", timing["ventorum_cpu_median_s"], timing["ventorum_cpu_s"]),
            ("Ventorum auto device", timing["ventorum_auto_median_s"],
             timing["ventorum_auto_s"]),
        ):
            samples_s = ", ".join(f"{s:.2f}" for s in samples)
            parts.append(f"| {label} | {med:.2f} | {samples_s} |")
        parts += [""]
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Entry point of the comparison program."""
    parser = argparse.ArgumentParser(description="Compare Ventorum against AVL (black box).")
    parser.add_argument("--avl", default=None, help="Path of avl.exe (or VENTORUM_AVL_EXE).")
    parser.add_argument("--out", required=True, help="Output folder OUTSIDE the repository.")
    parser.add_argument("--quick", action="store_true",
                        help="2 cases, 3 angles, 1 timing repeat.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel AVL processes (default: cpu_cores()).")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        help="Timeout [s] of one AVL run.")
    args = parser.parse_args(argv)

    avl_exe = find_avl(args.avl)
    if avl_exe is None:
        print("AVL program not found. Give --avl PATH or set VENTORUM_AVL_EXE.",
              file=sys.stderr)
        return 2
    try:
        out = refuse_inside_repository(Path(args.out))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    workers = args.workers or cpu_cores()
    angles = QUICK_ANGLES if args.quick else FULL_ANGLES
    meshes = QUICK_MESHES if args.quick else FULL_MESHES
    repeats = 1 if args.quick else 3

    out.mkdir(parents=True, exist_ok=True)
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    sets: list[dict] = []
    banners: list[str] = []
    for spec in case_list(args.quick):
        for mesh in meshes:
            record = run_set(avl_exe, spec, mesh, angles, runs, args.timeout)
            record["timing"] = time_set(avl_exe, spec, mesh, angles, workers,
                                        args.timeout, repeats)
            sets.append(record)
            if record["banner"] and record["banner"] not in banners:
                banners.append(record["banner"])
    meta = {
        "date": datetime.date.today().isoformat(),
        "ventorum_version": vt.__version__,
        "git_commit": _git_commit(),
        "avl_banner": "; ".join(banners) if banners else "unknown (all AVL runs failed)",
        "cpu_cores": cpu_cores(),
        "workers": workers,
        "angles_deg": angles,
        "meshes": meshes,
        "repeats": repeats,
    }
    (out / "avl_comparison.md").write_text(write_markdown(meta, sets), encoding="utf-8")
    with open(out / "avl_comparison.json", "w", encoding="utf-8") as handle:
        json.dump({"meta": meta, "sets": sets}, handle, indent=1, default=str)
    print(f"Wrote {out / 'avl_comparison.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
