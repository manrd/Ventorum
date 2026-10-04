# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Aerodynamic tools for AI agents.

Each tool takes strict inputs (see :mod:`ventorum.agent.schemas`) and
returns a JSON-safe dict with ``"status": "success"`` or ``"status":
"error"``. A tool does not raise for bad input or solver refusals; it
returns an error payload.
"""

from __future__ import annotations

import functools
import threading
from typing import Any
from collections.abc import Callable

import numpy as np

import ventorum as vt
from ventorum.agent.response import (
    computation_device_precision,
    condition_payload,
    device_summary_text,
    drag_and_ld,
    error_info,
    error_payload,
    fmt,
    geometry_payload,
    lift_to_drag,
    metrics_payload,
    moments_all_sets,
    moments_in_axes,
    result_device_precision,
    rnd,
    sectional_payload,
    settings_payload,
    spanwise_payload,
    trust_payload,
)
from ventorum.agent.schemas import (
    MAX_CALL_WORK,
    MAX_TOTAL_PANELS,
    MIN_ALPHA_STEP_DEG,
    InputError,
    boolean,
    build_aircraft_from_spec,
    integer,
    make_flight_condition,
    number,
    number_list,
    parse_axes,
    parse_condition,
    parse_detail_level,
    parse_settings,
    string,
    vector3,
)
from ventorum.core.constants import RHO_SL
from ventorum.core.datatypes import Aircraft, FlightCondition, SolverSettings
from ventorum.core.errors import GroundStrikeError, ValidityError
from ventorum.ground_effect.solver import MAX_BANK_DEG
from ventorum.geometry import lattice_cache
from ventorum.geometry.lattice import build_lattice
from ventorum.hardware.profile import size_class
from ventorum.solvers.factory import resolve_solver_type
from ventorum.solvers.lattice_base import DEFAULT_N_CHORD, MAX_AUTO_N_CHORD
from ventorum.utils.jsonsafe import json_safe
from ventorum.utils.parallel import estimate_panels, run_cases

MAX_POLAR_POINTS = 61

# Only one tuning run at a time. A second call while one runs is refused
# with "a tuning run is in progress" (see tune_machine).
_TUNE_LOCK = threading.Lock()


def _tool(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Catch every exception of a tool and return an error payload; make the output JSON-safe."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            out = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - a tool must not raise
            return error_payload(exc)
        return json_safe(out)

    return wrapper


def _fmt_point(p: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.3g}" for v in p) + "]"


def _check_method(ac: Aircraft, solver: str | None, wake_alignment: str, in_ground: bool) -> None:
    """Refuse a method that is not valid in ground effect, before any solve.

    The Fourier solver has no ground effect: :class:`ValidityError`
    (``invalid_method``). The body-axis wake is not allowed in ground effect:
    :class:`InputError` (``invalid_input``), as the schema says.
    """
    if not in_ground:
        return
    if resolve_solver_type(solver) == "fourier":
        raise ValidityError("The Fourier solver has no ground effect. Use solver='vlm'.")
    if wake_alignment == "body":
        raise InputError("settings.wake_alignment 'body' is not allowed in ground effect. Use 'freestream'.")


def _uses_lattice_chord(ac: Aircraft, sett: SolverSettings) -> bool:
    return resolve_solver_type(sett.solver_type) == "vlm"


def _check_mesh(ac: Aircraft, sett: SolverSettings) -> int:
    """Refuse a mesh larger than MAX_TOTAL_PANELS. Return the number of spanwise strips.

    For an automatic n_chord the check uses the smallest automatic value;
    :func:`_case_settings` reduces a larger automatic value.
    """
    chord_spacing = getattr(sett, "chord_spacing", "uniform")
    key = lattice_cache.lattice_key(ac, sett, "vlm", 1, chord_spacing)
    probe = lattice_cache.get_or_build(key, lambda: build_lattice(
        ac, sett, collocation="vlm", n_chord=1, chord_spacing=chord_spacing,
    ))
    strips = probe.n_strips
    if _uses_lattice_chord(ac, sett):
        n_chord = int(sett.n_chord) if sett.n_chord is not None else DEFAULT_N_CHORD
    else:
        n_chord = 1
    total = strips * n_chord
    if total > MAX_TOTAL_PANELS:
        auto = " (smallest automatic value)" if sett.n_chord is None and n_chord > 1 else ""
        raise InputError(
            f"The mesh is too large: {strips} spanwise strips x {n_chord} chordwise panels{auto} = {total} "
            f"panels. The limit is {MAX_TOTAL_PANELS} panels. Use a smaller n_panels or n_chord.")
    return strips


def _case_settings(ac: Aircraft, sett: SolverSettings, strips: int, condition: FlightCondition,
                   ground: Any = None, ref_point: np.ndarray | None = None) -> SolverSettings:
    """Return the solver settings for one case.

    If n_chord is automatic and its value makes the mesh larger than
    MAX_TOTAL_PANELS, return a copy with n_chord reduced to the limit.
    Otherwise return *sett*.
    """
    if sett.n_chord is not None or not _uses_lattice_chord(ac, sett):
        return sett
    cap = MAX_TOTAL_PANELS // strips
    if cap >= MAX_AUTO_N_CHORD:
        return sett
    from ventorum.solvers.horseshoe import HorseshoeSolver

    rp = ac.moment_reference() if ref_point is None else ref_point
    auto = HorseshoeSolver().resolve_n_chord(ac, sett, condition, ground, rp)
    if auto <= cap:
        return sett
    out = sett.clone()
    out.n_chord = int(cap)
    return out


def _common_settings(ac: Aircraft, sett: SolverSettings, strips: int,
                     conditions: list[FlightCondition]) -> tuple[SolverSettings, bool]:
    """Return one set of solver settings for all cases of a difference.

    In ground effect the automatic n_chord depends on the attitude. The
    cases of one finite difference must use the same mesh, so the count of
    the case with the smallest gap to the ground (the largest count) is
    used for all of them, reduced to the MAX_TOTAL_PANELS limit if needed.
    Returns the settings and True if the count was reduced to the limit.
    Otherwise (n_chord given, no lattice chord, no ground) *sett* is
    returned unchanged.
    """
    if sett.n_chord is not None or not _uses_lattice_chord(ac, sett):
        return sett, False
    if all(fc.h is None for fc in conditions):
        # Out of ground effect the automatic count does not change with the attitude.
        out = _case_settings(ac, sett, strips, conditions[0])
        return out, out is not sett
    from ventorum.solvers.horseshoe import HorseshoeSolver

    rp = ac.moment_reference()
    solver = HorseshoeSolver()
    auto = max(solver.resolve_n_chord(ac, sett, fc, None, rp) for fc in conditions)
    cap = MAX_TOTAL_PANELS // strips
    out = sett.clone()
    out.n_chord = int(min(auto, cap))
    return out, cap < auto


def _mesh_note(capped: bool, sett: SolverSettings | None = None) -> dict[str, Any]:
    if not capped:
        return {}
    used = f" to n_chord={sett.n_chord}" if sett is not None and sett.n_chord is not None else ""
    return {"mesh_note": f"The automatic n_chord was reduced{used} to keep the mesh at or below "
                         f"{MAX_TOTAL_PANELS} panels."}


def _used_n_chord(res: Any) -> int | None:
    """Return the chordwise panel count that the solve of *res* used, or None."""
    details = getattr(res, "details", None)
    lat = details.get("lattice") if isinstance(details, dict) else None
    if lat is not None and hasattr(lat, "n_chord"):
        return int(lat.n_chord)
    return None


def _apply_axes(
    Cl: float,
    Cm: float,
    Cn: float,
    alpha_deg: float,
    beta_deg: float,
    axes: str,
) -> tuple[float | None, float | None, float | None]:
    """Return the rounded (Cl, Cm, Cn) triple in the requested axes."""
    if axes in ("body", "all"):
        return rnd(Cl), rnd(Cm), rnd(Cn)
    c, m, n = moments_in_axes(Cl, Cm, Cn, alpha_deg, beta_deg, axes)
    return rnd(c), rnd(m), rnd(n)


def _planned_panels(
    ac: Aircraft,
    sett: SolverSettings,
    condition: FlightCondition | None = None,
    ground: Any = None,
    ref_point: np.ndarray | None = None,
) -> int:
    """Return the panel count of the planned lattice, without solving.

    The count uses the real spanwise strips (including the ``n_panels``
    of each surface) times the chordwise count that the solve will use.
    An automatic ``n_chord`` is resolved without solving; the mesh-limit
    cap of :func:`_case_settings` applies.
    """
    chord_spacing = getattr(sett, "chord_spacing", "uniform")
    key = lattice_cache.lattice_key(ac, sett, "vlm", 1, chord_spacing)
    probe = lattice_cache.get_or_build(key, lambda: build_lattice(
        ac, sett, collocation="vlm", n_chord=1, chord_spacing=chord_spacing,
    ))
    strips = probe.n_strips
    if not _uses_lattice_chord(ac, sett):
        return int(strips)
    if sett.n_chord is not None:
        return int(strips * int(sett.n_chord))
    if condition is None and ground is None:
        return int(strips * DEFAULT_N_CHORD)
    from ventorum.solvers.horseshoe import HorseshoeSolver

    rp = ac.moment_reference() if ref_point is None else ref_point
    try:
        n = int(HorseshoeSolver().resolve_n_chord(ac, sett, condition, ground, rp))
    except Exception:  # noqa: BLE001 - only an estimate
        n = int(DEFAULT_N_CHORD)
    cap = MAX_TOTAL_PANELS // max(int(strips), 1)
    if cap < MAX_AUTO_N_CHORD and n > cap:
        n = max(int(cap), 1)
    return int(strips * n)


def _check_call_work(estimate: int, hint: str) -> None:
    """Refuse a call whose estimated work is above MAX_CALL_WORK.

    The work of one solve of N panels is N^2 units. The estimate is
    the sum over all solves of the call. The error type is
    ``invalid_input``. The message gives the estimate, the budget and
    the input that reduces the work most.
    """
    if int(estimate) > int(MAX_CALL_WORK):
        raise InputError(
            f"The estimated work is {int(estimate)} units (sum of N^2 over all planned solves), "
            f"above the budget {int(MAX_CALL_WORK)} units. "
            f"Reduce the work with {hint}.")


def _workers(val: Any) -> int | str:
    if val == "auto":
        return "auto"
    return integer(val, "n_workers", minimum=1, maximum=64)


# ═══════════════════════════════════════════════════════════════════════════════
# Wing analysis
# ═══════════════════════════════════════════════════════════════════════════════

@_tool
def wing_analysis(
    wing: Any,
    flight_condition: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    detail_level: str = "standard",
    axes: str = "body",
) -> dict[str, Any]:
    """Aerodynamic coefficients at one flight condition.

    Parameters
    ----------
    wing : dict
        Surface spec or aircraft spec (see the tool schema).
    flight_condition : dict or None
        ``V_inf_m_s``, ``alpha_deg``, ``beta_deg``, ``rho_kg_m3`` or
        ``altitude_m``, ``h_m`` (ground effect).
    settings : dict or None
        ``solver``, ``n_panels``, ``n_chord``, ``wake_alignment``.
    detail_level : 'summary', 'standard' or 'full'
    axes : 'body', 'stability', 'wind' or 'all'
        Axis system of the moment coefficients Cl, Cm and Cn.
        ``"body"`` (default) is fixed to the aircraft. ``"all"`` returns
        the three sets in ``metrics["moments"]``. CL, CD and CY are
        relative to the free stream in every set.
    """
    ac = build_aircraft_from_spec(wing)
    c = parse_condition(flight_condition)
    sett = parse_settings(settings)
    dl = parse_detail_level(detail_level)
    ax = parse_axes(axes)
    _check_method(ac, sett.solver_type, sett.wake_alignment, c["h_m"] is not None)
    strips = _check_mesh(ac, sett)

    fc = make_flight_condition(c)
    _check_call_work(
        _planned_panels(ac, sett, fc) ** 2,
        "fewer panels (smaller n_panels or n_chord)")
    case_sett = _case_settings(ac, sett, strips, fc)
    res = vt.analyze(ac, fc, case_sett)
    m = metrics_payload(res)
    m["Cl"], m["Cm"], m["Cn"] = _apply_axes(
        res.totals.Cl, res.totals.Cm, res.totals.Cn, c["alpha_deg"], c["beta_deg"], ax)
    if ax == "all":
        m["moments"] = moments_all_sets(
            res.totals.Cl, res.totals.Cm, res.totals.Cn, c["alpha_deg"], c["beta_deg"])
    trust = trust_payload(res.totals.trust)
    rp = ac.moment_reference()
    where = "free air" if c["h_m"] is None else f"h={c['h_m']:g} m above the ground"
    device, precision = result_device_precision(res)
    summary = (
        f"[Ventorum RESULT] {ac.name}: CL={fmt(m['CL'], '.4f')}, CDi={fmt(m['CDi'], '.5f')}, "
        f"CD={fmt(m['CD'], '.5f')} ({m['drag_basis']}), L/D={fmt(m['L_over_D'], '.2f')}, "
        f"e={fmt(m['e'], '.3f')}, Cm={fmt(m['Cm'], '.4f')} about {_fmt_point(rp)} m. "
        f"Condition: alpha={c['alpha_deg']:g} deg, beta={c['beta_deg']:g} deg, V={c['V_inf_m_s']:g} m/s, "
        f"{where}. Solver: {res.solver_type}, converged={bool(res.converged)}. "
        f"{device_summary_text(device, precision)}"
    )
    if res.totals.trust is not None:
        t = res.totals.trust
        summary += f" Trust {t.score:.2f} ({t.rating}), {len(t.warnings)} warning(s)."

    out: dict[str, Any] = {
        "status": "success",
        "executive_summary": summary,
        "metrics": m,
        "trust": trust,
        "condition_used": condition_payload(c),
        "settings_used": {**settings_payload(sett, res), **_mesh_note(case_sett is not sett, case_sett)},
        "axes": ax,
    }
    if dl in ("standard", "full"):
        out["geometry"] = geometry_payload(ac)
        out["sectional_diagnostics"] = sectional_payload(res)
    if dl == "full":
        out["spanwise_distributions"] = spanwise_payload(res)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Polar sweep
# ═══════════════════════════════════════════════════════════════════════════════

def _gpu_takes(n_cases: int, n_panels: int) -> bool:
    """Return True if the GPU pipelines take a batch of *n_cases* on a lattice of *n_panels* panels (free air)."""
    from ventorum import gpu

    return gpu.get_device() != "cpu" and gpu.use_gpu(float(n_cases) * float(n_panels) ** 2, "vlm", n_cases, n_panels)


def _batched_polar(ac: Aircraft, sett: SolverSettings, conditions: list[FlightCondition],
                   alphas: list[float]) -> list[tuple]:
    """Solve the angles of a lattice polar out of ground effect as one batch.

    Each angle gets the same result, to the last bit, as a single solve
    (``vt.analyze``). Returns the items ``(alpha, result, None, settings)``
    of the polar table.
    """
    from ventorum.solvers.factory import make_solver
    from ventorum.utils.validation import validate_aircraft, validate_flight_condition, validate_solver_settings

    # The same steps as vt.analyze and LatticeSolver.solve, once for all angles.
    for fc in conditions:
        validate_flight_condition(fc)
    validate_solver_settings(sett)
    solver = make_solver(resolve_solver_type(sett.solver_type))
    aircraft = ac.clone()
    validate_aircraft(aircraft)
    fp = lattice_cache.surfaces_fingerprint(aircraft)
    geo = lattice_cache.geometry_info(fp, aircraft)
    aircraft.compute_reference_values(auto=geo["auto_ref"])
    rp = aircraft.moment_reference()
    lattice = solver.build(aircraft, sett, conditions[0], None, rp, fingerprint=fp)
    results = solver.solve_batch(lattice, conditions, sett, aircraft.S_ref, aircraft.b_ref, aircraft.c_ref,
                                 ref_point=rp, main_surface=geo["main"], continuation=False)
    out = []
    for a, fc, res in zip(alphas, conditions, results):
        res.condition = fc
        out.append((a, res, None, sett))
    return out



@_tool
def polar_sweep(
    wing: Any,
    alpha_start_deg: float,
    alpha_end_deg: float,
    alpha_step_deg: float,
    flight_condition: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    detail_level: str = "standard",
    axes: str = "body",
) -> dict[str, Any]:
    """Sweep of the angle of attack with the lift slope and the maximum L/D.

    ``flight_condition`` takes the same keys as :func:`wing_analysis`
    without ``alpha_deg``. In ground effect all angles use one mesh: the
    mesh of the first valid angle. The ``axes`` input selects the axis
    system of Cm (``"body"`` by default, ``"all"`` adds the three moment
    sets to each row).
    """
    ac = build_aircraft_from_spec(wing)
    c = parse_condition(flight_condition,
                        allow=("V_inf_m_s", "beta_deg", "rho_kg_m3", "altitude_m", "h_m"))
    sett = parse_settings(settings)
    dl = parse_detail_level(detail_level)
    ax = parse_axes(axes)
    a0 = number(alpha_start_deg, "alpha_start_deg", minimum=-30, maximum=30)
    a1 = number(alpha_end_deg, "alpha_end_deg", minimum=-30, maximum=30)
    da = number(alpha_step_deg, "alpha_step_deg", minimum=MIN_ALPHA_STEP_DEG, maximum=30)
    if a1 < a0:
        raise InputError(f"alpha_end_deg ({a1}) must be >= alpha_start_deg ({a0}).")
    n = int(np.floor((a1 - a0) / da + 1e-9)) + 1
    if n > MAX_POLAR_POINTS:
        raise InputError(f"The sweep has {n} angles; the limit is {MAX_POLAR_POINTS}. Use a larger step.")
    alphas = [round(a0 + k * da, 10) for k in range(n)]
    _check_method(ac, sett.solver_type, sett.wake_alignment, c["h_m"] is not None)
    strips = _check_mesh(ac, sett)
    if c["h_m"] is not None:
        estimate = sum(
            _planned_panels(ac, sett, make_flight_condition(c, alpha_deg=a)) ** 2 for a in alphas)
    else:
        estimate = len(alphas) * (_planned_panels(ac, sett) ** 2)
    _check_call_work(
        estimate,
        "fewer angles (smaller range or larger alpha_step_deg) or fewer panels "
        "(smaller n_panels or n_chord)")
    capped: list[SolverSettings] = []
    fixed: SolverSettings | None = None

    def run(a: float, s: SolverSettings | None = None):
        fc = make_flight_condition(c, alpha_deg=a)
        try:
            fresh = s is None
            s = _case_settings(ac, sett, strips, fc) if s is None else s
            if fresh and s is not sett:
                capped.append(s)
            return a, vt.analyze(ac.clone(), fc, s), None, s
        except (GroundStrikeError, ValidityError) as exc:
            return a, None, exc, s

    in_ground = c["h_m"] is not None
    if in_ground:
        # One mesh for all angles: solve in order until the first valid
        # angle, pin its chordwise count, and reuse it for the rest.
        first_done: list[tuple[float, Any, Any, Any]] = []
        remaining = list(alphas)
        while remaining and fixed is None:
            a = remaining.pop(0)
            item = run(a)
            first_done.append(item)
            if item[1] is not None:
                s_first = item[3]
                used = _used_n_chord(item[1])
                if s_first is sett:
                    fixed = sett.clone()
                    if fixed.n_chord is None and used is not None:
                        fixed.n_chord = int(used)
                else:
                    fixed = s_first.clone()
                    if used is not None:
                        fixed.n_chord = int(used)
        rest = run_cases(lambda a: run(a, fixed), remaining, estimate_panels(ac, fixed or sett),
                         "auto") if fixed is not None else []
        outputs = first_done + list(rest)
    elif (resolve_solver_type(sett.solver_type) in ("linear", "nonlinear")
          or (resolve_solver_type(sett.solver_type) == "vlm"
              and (size_class(estimate_panels(ac, sett)) == "small"
                   or _gpu_takes(len(alphas), _planned_panels(ac, sett)))
              and _case_settings(ac, sett, strips, make_flight_condition(c, alpha_deg=alphas[0])) is sett)):
        # Lifting lines, small vortex lattices and batches that the GPU takes:
        # one batch. A larger vortex lattice on the CPU gains more from cases
        # in parallel.
        outputs = _batched_polar(ac, sett, [make_flight_condition(c, alpha_deg=a) for a in alphas], alphas)
    else:
        outputs = run_cases(run, alphas, estimate_panels(ac, sett), "auto")

    table: list[dict[str, Any]] = []
    ok: list[tuple[float, Any]] = []
    for a, res, err, _s in outputs:
        if res is None:
            info = error_info(err)
            table.append({"alpha_deg": a, "status": "ground_strike" if info["type"] == "ground_strike"
                          else "refused", "message": info["message"]})
            continue
        m = metrics_payload(res)
        _, cm_ax, _ = _apply_axes(res.totals.Cl, res.totals.Cm, res.totals.Cn, a, c["beta_deg"],
                                  ax)
        device, precision = result_device_precision(res)
        row: dict[str, Any] = {
            "alpha_deg": a, "CL": m["CL"], "CDi": m["CDi"], "CD": m["CD"],
            "L_over_D": m["L_over_D"], "Cm": cm_ax, "e": m["e"], "converged": m["converged"],
            "trust_score": rnd(res.totals.trust.score, 3) if res.totals.trust else None,
            "status": "ok", "device": device, "precision": precision}
        if in_ground:
            row["n_chord"] = _used_n_chord(res)
        if ax == "all":
            row["moments"] = moments_all_sets(
                res.totals.Cl, res.totals.Cm, res.totals.Cn, a, c["beta_deg"])
        table.append(row)
        ok.append((a, res))

    if not ok:
        kinds = {row["status"] for row in table}
        etype = "ground_strike" if "ground_strike" in kinds else "invalid_method"
        return error_payload(etype, "No angle of attack in the sweep gave a result.", polar_table=table)

    a_ok = np.array([a for a, _ in ok])
    cl_ok = np.array([r.totals.CL for _, r in ok])
    summary: dict[str, Any] = {"n_points": n, "n_ok": len(ok), "n_failed": n - len(ok)}
    if len(ok) >= 2:
        slope, icpt = np.polyfit(np.radians(a_ok), cl_ok, 1)
        resid = cl_ok - (slope * np.radians(a_ok) + icpt)
        summary.update(
            CL_alpha_per_rad=rnd(slope, 4),
            CL_alpha_per_deg=rnd(slope * np.pi / 180.0, 5),
            alpha_zero_lift_deg=rnd(np.degrees(-icpt / slope), 3) if abs(slope) > 1e-9 else None,
            fit_max_abs_residual_CL=rnd(np.max(np.abs(resid)), 5),
            fit_note="Linear least-squares fit of CL over all valid angles. A large residual means CL is "
                     "not linear in this range.",
        )
    else:
        summary["CL_alpha_per_rad"] = None
        summary["fit_note"] = "Only one valid angle; no lift slope."

    profile = all(r.totals.CD_total is not None and r.totals.CDp is not None for _, r in ok)
    best_ld, best_a = None, None
    for a, r in ok:
        cd, _, ld = drag_and_ld(r.totals)
        if ld is not None and r.totals.CL > 0 and (best_ld is None or ld > best_ld):
            best_ld, best_a = ld, a
    summary["max_L_over_D"] = rnd(best_ld, 3)
    summary["alpha_at_max_L_over_D_deg"] = best_a
    summary["drag_basis"] = ("CD_total (induced + profile)" if profile
                             else "CDi only: the airfoils have no profile drag, cd0 = 0")
    if not profile:
        summary["L_over_D_note"] = ("With induced drag only, L/D = 1/(k*CL) increases without limit as CL "
                                    "goes to zero, so the maximum is at the smallest positive CL of the "
                                    "sweep. Give the airfoil a cd0 for a useful maximum L/D.")
    if best_ld is not None and best_a is not None and best_a in (alphas[0], alphas[-1]):
        summary["max_L_over_D_at_sweep_end"] = True

    slope_txt = (f"CL_alpha={fmt(summary['CL_alpha_per_rad'], '.3f')}/rad"
                 if summary.get("CL_alpha_per_rad") is not None else "CL_alpha not available")
    ld_txt = (f"max L/D={best_ld:.2f} at alpha={best_a:g} deg ({summary['drag_basis']})" if best_ld is not None
              else "no positive-lift point for L/D")
    exec_summary = (f"[Ventorum RESULT] Polar of {ac.name} from {a0:g} to {alphas[-1]:g} deg "
                    f"({len(ok)} of {n} angles valid): {slope_txt}, {ld_txt}.")
    if len(ok) < n:
        exec_summary += f" {n - len(ok)} angle(s) failed; see polar_table."
    device_all, precision_all = computation_device_precision([r for _, r in ok])
    exec_summary += f" {device_summary_text(device_all, precision_all)}"

    note_sett = capped[0] if capped else None
    settings_used = {**settings_payload(sett, ok[0][1]), **_mesh_note(bool(capped), note_sett)}
    settings_used["device"] = device_all
    settings_used["precision"] = precision_all
    out: dict[str, Any] = {
        "status": "success",
        "executive_summary": exec_summary,
        "polar_summary": summary,
        "condition_used": {k: v for k, v in condition_payload(c).items() if k != "alpha_deg"},
        "settings_used": settings_used,
        "axes": ax,
    }
    if dl == "summary":
        failed = [row for row in table if row["status"] != "ok"]
        if failed:
            out["failed_points"] = failed
    else:
        out["polar_table"] = table
        out["geometry"] = geometry_payload(ac)
    if dl == "full":
        out["trust_per_point"] = [{"alpha_deg": a, **trust_payload(r.totals.trust)} for a, r in ok]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Ground effect
# ═══════════════════════════════════════════════════════════════════════════════

def _k_factor(cl: float, cdi: float) -> float | None:
    return cdi / (cl * cl) if abs(cl) > 1e-6 else None


def _irodov(ac, ok, alpha, phi, beta, V, rho, rp, href, fixed, c_ref, dl, sweep_cls, result_cls):
    """Irodov height-pitch margins at the valid heights. Return (payload, summary text).

    The centre-alpha cases of the height rows are reused when beta = 0 (the
    sweep is at beta = 0). Only alpha - 1 and alpha + 1 deg are new cases.
    """
    hs = [h for h, _ in ok]
    side = [alpha - 1.0, alpha + 1.0]
    alphas = [alpha - 1.0, alpha, alpha + 1.0]
    reuse = beta == 0.0
    sweep = sweep_cls(ac.clone(), settings=fixed).run_sweep(
        hs, side if reuse else alphas, [phi], V_inf=V, rho=rho, ref_point=rp,
        height_ref=href, compute_strike_limit=False)
    shape = (len(hs), 3, 1)
    CL, Cm, HR = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    if reuse:
        CL[:, [0, 2], 0], Cm[:, [0, 2], 0] = sweep.CL[:, :, 0], sweep.Cm[:, :, 0]
        HR[:, [0, 2], 0] = sweep.h_ref_grid[:, :, 0]
        CL[:, 1, 0] = [r.CL for _, r in ok]
        Cm[:, 1, 0] = [r.Cm for _, r in ok]
        HR[:, 1, 0] = [r.h_ref for _, r in ok]
    else:
        CL, Cm, HR = sweep.CL, sweep.Cm, sweep.h_ref_grid
    grid = result_cls(heights=np.asarray(hs, dtype=float), alphas_deg=np.asarray(alphas, dtype=float),
                      phis_deg=np.asarray([phi], dtype=float), grid_shape=shape, CL=CL, Cm=Cm,
                      c_ref=c_ref, ref_point=np.asarray(rp, dtype=float), height_ref=href, h_ref_grid=HR)
    der = grid.compute_stability_derivatives()
    n = len(hs)
    irows = []
    for i, h in enumerate(hs):
        xa = float(der["x_alpha"][i, 1]) if "x_alpha" in der else float("nan")
        xh = float(der["x_h"][i, 1]) if "x_h" in der else float("nan")
        mg = float(der["irodov_margin"][i, 1]) if "irodov_margin" in der else float("nan")
        verdict = ("stable (margin > 0)" if mg > 0 else "unstable (margin <= 0)") if np.isfinite(mg) \
            else "not available (a neighbour case failed)"
        irows.append({"h_m": h, "h_over_c": rnd(h / c_ref, 5),
                      "x_alpha_m": rnd(xa, 5), "x_h_m": rnd(xh, 5),
                      "x_alpha_over_c": rnd(xa / c_ref, 5), "x_h_over_c": rnd(xh / c_ref, 5),
                      "irodov_margin": rnd(mg, 5), "verdict": verdict,
                      "height_difference": "central" if 0 < i < n - 1 else "one-sided"})
    notes = [
        "x_alpha and x_h are the pitch and height aerodynamic centres in metres aft of the moment "
        f"reference point (the pivot, {[rnd(v, 5) for v in rp]} m). irodov_margin = (x_alpha - x_h)/c_ref; "
        "height-pitch static stability needs a margin > 0 (x_h ahead of x_alpha).",
        f"Pitch derivatives: alpha {alpha - 1:g}, {alpha:g}, {alpha + 1:g} deg at constant height of the "
        f"moment reference point. Height derivatives: constant alpha. The margin of one flight state does "
        f"not depend on height_ref. {der.get('note', '')}",
        f"Bank angle {phi:g} deg for all cases; same chordwise mesh (n_chord={fixed.n_chord}).",
    ]
    if not reuse:
        notes.append(f"The Irodov sweep is at beta = 0 deg (the rows above use beta = {beta:g} deg).")
    if sweep.errors:
        notes.append(f"{len(sweep.errors)} case(s) of the Irodov sweep failed (strike or refusal); "
                     "the margins next to them are null.")
    # Headline: the lowest height with a central height difference and a valid margin.
    head = next((r for r in irows[1:-1] if r["irodov_margin"] is not None), None)
    if head is None and n == 2 and irows[0]["irodov_margin"] is not None:
        head = irows[0]
        notes.append("Only 2 valid heights: the height derivative is a one-sided (first-order) difference. "
                     "Give 3 or more heights for a central difference.")
    payload: dict[str, Any] = {"rows": irows, "c_ref_m": rnd(c_ref, 5), "notes": notes,
                               "pivot_point_m": [rnd(v, 5) for v in rp],
                               "headline_h_m": head["h_m"] if head else None}
    if dl == "full":
        payload["derivative_grids"] = json_safe({k: v for k, v in der.items() if k not in ("note",)})
    txt = ""
    if head is not None:
        kind = "central" if head["height_difference"] == "central" else "one-sided"
        txt = (f" Irodov margin at h={head['h_m']:g} m ({kind} height difference): "
               f"{head['irodov_margin']:+.3f} ({head['verdict']}).")
    return payload, txt


@_tool
def ground_effect(
    wing: Any,
    heights_m: list[float],
    alpha_deg: float = 4.0,
    phi_deg: float = 0.0,
    beta_deg: float = 0.0,
    height_ref: str = "ref",
    V_inf_m_s: float = 50.0,
    rho_kg_m3: float = RHO_SL,
    ref_point_m: list[float] | None = None,
    settings: dict[str, Any] | None = None,
    detail_level: str = "standard",
    axes: str = "body",
) -> dict[str, Any]:
    """Ground effect at several heights compared with free air.

    All cases (every height and the free-air reference) use the same
    chordwise panel count: the count that the lowest valid height needs,
    or ``settings.n_chord`` if given. The moments Cl, Cm and Cn are in
    body axes by default (the same axes as :func:`wing_analysis`); the
    ``axes`` input selects ``"stability"``, ``"wind"`` or ``"all"``
    (the three sets are then in each row under ``"moments"``). CL, CD
    and CY are relative to the free stream in every set.
    """
    from ventorum.ground_effect import GroundEffectSweep, analyze_ground_effect
    from ventorum.ground_effect.solver import place_ground
    from ventorum.ground_effect.sweep import GroundEffectSweepResult

    ac = build_aircraft_from_spec(wing)
    heights = number_list(heights_m, "heights_m", min_len=1, max_len=20, exclusive_min=0)
    if len(set(heights)) != len(heights):
        raise InputError("heights_m has duplicate values.")
    heights = sorted(heights)
    alpha = number(alpha_deg, "alpha_deg", minimum=-30, maximum=30)
    phi = number(phi_deg, "phi_deg", minimum=-60, maximum=60)
    beta = number(beta_deg, "beta_deg", minimum=-30, maximum=30)
    href = string(height_ref, "height_ref", ("ref", "min", "qc", "te"))
    V = number(V_inf_m_s, "V_inf_m_s", exclusive_min=0, maximum=340)
    rho = number(rho_kg_m3, "rho_kg_m3", exclusive_min=0, maximum=2)
    rp = vector3(ref_point_m, "ref_point_m") if ref_point_m is not None else ac.moment_reference()
    ac.ref_point = np.array(rp, dtype=float)
    sett = parse_settings(settings)
    dl = parse_detail_level(detail_level)
    ax = parse_axes(axes)
    c_ref = float(ac.c_ref)
    _check_method(ac, sett.solver_type, sett.wake_alignment, True)
    strips = _check_mesh(ac, sett)
    a_rad, b_rad, p_rad = (float(np.radians(v)) for v in (alpha, beta, phi))

    # Work budget before any solve. All heights share the mesh fixed at
    # the lowest height (the largest automatic n_chord), plus free air
    # and the Irodov height-pitch sweep (2 extra solves per height at
    # beta = 0, else 3 per height).
    _fc_free = FlightCondition(V_inf=V, alpha=a_rad, beta=b_rad, rho=rho)
    _n_fixed = 0
    for _h in heights:
        _gp, _ = place_ground(ac, sett, _h, a_rad, b_rad, p_rad, rp, href)
        _n_fixed = max(_n_fixed, _planned_panels(ac, sett, _fc_free, ground=_gp, ref_point=rp))
    _n_extra = (2 * len(heights) if beta == 0.0 else 3 * len(heights)) if len(heights) >= 2 else 0
    _check_call_work(
        (len(heights) + 1 + _n_extra) * (_n_fixed ** 2),
        "fewer heights or fewer panels (smaller n_panels or n_chord)")

    def first_settings(h: float):
        # Reduce an automatic n_chord for this height if the mesh is too large.
        gp, _ = place_ground(ac, sett, h, a_rad, b_rad, p_rad, rp, href)
        fc = FlightCondition(V_inf=V, alpha=a_rad, beta=b_rad, rho=rho)
        return _case_settings(ac, sett, strips, fc, ground=gp, ref_point=rp)

    def run(h: float, s):
        try:
            return h, analyze_ground_effect(
                ac.clone(), h, alpha_deg=alpha, phi_deg=phi, beta_deg=beta, V_inf=V, rho=rho,
                ref_point=rp, height_ref=href, settings=s, compute_strike_limit=True), None
        except (GroundStrikeError, ValidityError) as exc:
            return h, None, exc

    # 1. Lowest heights first, until one is valid: it fixes the chordwise mesh.
    outputs: list[tuple[float, Any, Any]] = []
    fixed = None
    remaining = list(heights)
    capped = False
    while remaining:
        h = remaining.pop(0)
        s0 = first_settings(h)
        item = run(h, s0)
        outputs.append(item)
        if item[1] is not None:
            capped = s0 is not sett
            fixed = s0.clone()
            if fixed.n_chord is None:
                fixed.n_chord = int(item[1].n_chord)
            break
    # 2. The other heights with the same mesh.
    if fixed is not None and remaining:
        outputs.extend(run_cases(lambda hh: run(hh, fixed), remaining, estimate_panels(ac, fixed), "auto"))

    # 3. Free air with the same mesh.
    free = None
    if fixed is not None:
        free = vt.analyze(ac.clone(), FlightCondition(V_inf=V, alpha=float(np.radians(alpha)),
                                                       beta=float(np.radians(beta)), rho=rho), fixed)
    cl_f = float(free.totals.CL) if free is not None else None
    k_f = _k_factor(cl_f, float(free.totals.CDi)) if free is not None else None

    rows: list[dict[str, Any]] = []
    ok: list[tuple[float, Any]] = []
    for h, res, err in sorted(outputs, key=lambda t: t[0]):
        row: dict[str, Any] = {"h_m": h, "h_over_c": rnd(h / c_ref, 5)}
        if res is None:
            info = error_info(err)
            row.update(status="ground_strike" if info["type"] == "ground_strike" else "refused",
                       message=info["message"])
            rows.append(row)
            continue
        k = _k_factor(res.CL, res.CDi)
        lim = res.phi_strike_limit
        lim_found = lim is not None and lim < MAX_BANK_DEG
        cl_ax, cm_ax, cn_ax = _apply_axes(
            res.Cl_body, res.Cm_body, res.Cn_body, alpha, beta, ax)
        device, precision = result_device_precision(res)
        row.update(
            status="ok",
            h_min_m=rnd(res.h_min, 5),
            h_min_over_c=rnd(res.h_min_over_c, 5),
            CL=rnd(res.CL), CDi=rnd(res.CDi, 7), CD=rnd(res.CD, 7),
            Cm=cm_ax, Cl=cl_ax, Cn=cn_ax,
            CL_ratio=rnd(res.CL / cl_f, 5) if cl_f is not None and abs(cl_f) > 1e-9 else None,
            induced_drag_factor_ratio=rnd(k / k_f, 5) if k is not None and k_f else None,
            L_over_D=rnd(lift_to_drag(res.CL, res.CD), 3),
            e=rnd(res.e, 4),
            phi_strike_limit_deg=rnd(lim, 3) if lim_found else None,
            bank_strike_limit_found=bool(lim_found),
            n_chord=int(res.n_chord),
            device=device,
            precision=precision,
        )
        if ax == "all":
            row["moments"] = moments_all_sets(
                res.Cl_body, res.Cm_body, res.Cn_body, alpha, beta)
        rows.append(row)
        ok.append((h, res))

    n_strike = sum(r["status"] == "ground_strike" for r in rows)
    n_refused = sum(r["status"] == "refused" for r in rows)
    if not ok:
        etype = "ground_strike" if n_strike else "invalid_method"
        msg = (f"No height gave a result: {n_strike} ground strike(s), {n_refused} refusal(s). "
               "See 'rows' for each height.")
        return error_payload(etype, msg, rows=rows)

    h_low, r_low = ok[0]
    low_row = next(r for r in rows if r["h_m"] == h_low)
    summary = {
        "n_heights": len(heights), "n_ok": len(ok), "n_ground_strike": n_strike, "n_refused": n_refused,
        "lowest_valid_height_m": h_low,
        "CL_ratio_at_lowest": low_row["CL_ratio"],
        "induced_drag_factor_ratio_at_lowest": low_row["induced_drag_factor_ratio"],
        "phi_strike_limit_deg_at_lowest": low_row["phi_strike_limit_deg"],
        "bank_strike_limit_found_at_lowest": low_row["bank_strike_limit_found"],
        "bank_strike_note": (f"phi_strike_limit_deg is the bank angle of the first ground contact. The search "
                             f"stops at {MAX_BANK_DEG:g} deg: if no point touches up to that angle, "
                             "phi_strike_limit_deg is null and bank_strike_limit_found is false."),
        "n_chord_all_cases": int(fixed.n_chord),
        "solver": r_low.solver_type,
        "ratio_definitions": "CL_ratio = CL / CL_free_air. induced_drag_factor_ratio = (CDi/CL^2) / "
                             "(CDi/CL^2)_free_air; below 1 means less induced drag for the same lift.",
    }
    _, free_cm, _ = _apply_axes(
        free.totals.Cl, free.totals.Cm, free.totals.Cn, alpha, beta, ax)
    free_air = {"CL": rnd(cl_f), "CDi": rnd(free.totals.CDi, 7), "Cm": free_cm,
                "L_over_D": rnd(drag_and_ld(free.totals)[2], 3), "e": rnd(free.totals.e, 4),
                "note": "Free air at the same alpha and beta, bank angle 0, same mesh."}
    if ax == "all":
        free_air["moments"] = moments_all_sets(
            free.totals.Cl, free.totals.Cm, free.totals.Cn, alpha, beta)

    out: dict[str, Any] = {
        "status": "success",
        "summary": summary,
        "free_air": free_air,
        "rows": rows,
        "axes": ax,
        "condition_used": {"alpha_deg": alpha, "phi_deg": phi, "beta_deg": beta, "V_inf_m_s": V,
                           "rho_kg_m3": rho, "height_ref": href, "ref_point_m": [rnd(v, 5) for v in rp],
                           "heights_meaning": f"heights_m are metres above the ground of the '{href}' point."},
        "settings_used": {**settings_payload(fixed), "n_chord": int(fixed.n_chord), **_mesh_note(capped, fixed)},
    }
    device_all, precision_all = computation_device_precision([r for _, r in ok] + ([free] if free is not None else []))
    out["settings_used"]["device"] = device_all
    out["settings_used"]["precision"] = precision_all
    free_device, free_precision = result_device_precision(free)
    out["free_air"]["device"] = free_device
    out["free_air"]["precision"] = free_precision

    # Irodov height-pitch criterion.
    irodov_txt = ""
    if len(ok) >= 2:
        out["irodov"], irodov_txt = _irodov(ac, ok, alpha, phi, beta, V, rho, rp, href, fixed, c_ref, dl,
                                            GroundEffectSweep, GroundEffectSweepResult)
    else:
        out["irodov"] = {"rows": [], "notes": ["The Irodov criterion needs at least 2 valid heights."]}

    fail_txt = ""
    if n_strike or n_refused:
        bad = [f"{r['h_m']:g} m ({r['status']})" for r in rows if r["status"] != "ok"]
        fail_txt = f" Failed heights: {', '.join(bad)}."
    cr = summary["CL_ratio_at_lowest"]
    kr = summary["induced_drag_factor_ratio_at_lowest"]
    out["executive_summary"] = (
        f"[Ventorum RESULT] Ground effect of {ac.name} at alpha={alpha:g} deg, phi={phi:g} deg: at "
        f"h={h_low:g} m (h/c={h_low / c_ref:.3f}, '{href}' point) CL is "
        f"{fmt(cr, '.3f')} x free air and the induced-drag factor is "
        f"{fmt(kr, '.3f')} x free air. {len(ok)} of {len(heights)} heights valid; "
        f"same chordwise mesh (n_chord={fixed.n_chord}) for all cases.{fail_txt}{irodov_txt} "
        f"{device_summary_text(device_all, precision_all)}"
    )

    if dl == "summary":
        out["rows"] = [{k: r.get(k) for k in ("h_m", "status", "CL_ratio", "induced_drag_factor_ratio",
                                              "phi_strike_limit_deg", "bank_strike_limit_found", "message",
                                              "device", "precision")
                        if k in r} for r in rows]
        out["irodov"].pop("derivative_grids", None)
    else:
        out["geometry"] = geometry_payload(ac)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Stability derivatives
# ═══════════════════════════════════════════════════════════════════════════════

@_tool
def stability_derivatives(
    wing: Any,
    flight_condition: dict[str, Any] | None = None,
    x_cg_m: float | None = None,
    settings: dict[str, Any] | None = None,
    detail_level: str = "standard",
    axes: str = "body",
) -> dict[str, Any]:
    """Compute the static stability derivatives about the centre of gravity.

    Central differences: alpha +/- 0.5 deg, beta +/- 1 deg. The moment
    reference point is ``[x_cg_m, 0, z_ref]`` (``z_ref`` from the aircraft
    reference point). With ``flight_condition.h_m`` the ground is ``h_m``
    below that point for every case, so the derivatives are at constant
    height. The ``axes`` input selects the axis system of the point
    moments and of the moment derivatives (``"body"`` by default,
    ``"all"`` adds the three sets); the derivatives are taken in the
    axes of the reference condition, held fixed while alpha and beta
    change. CL_alpha and CY_beta are relative to the free stream in
    every set. The static margin and the neutral point use the
    body-axis Cm_alpha and keep their values in every set.
    """
    ac = build_aircraft_from_spec(wing)
    c = parse_condition(flight_condition)
    sett = parse_settings(settings)
    dl = parse_detail_level(detail_level)
    ax = parse_axes(axes)
    base = ac.moment_reference()
    x_cg = number(x_cg_m, "x_cg_m") if x_cg_m is not None else float(base[0])
    ac.ref_point = np.array([x_cg, 0.0, float(base[2])])
    ac.compute_reference_values()
    c_ref = float(ac.c_ref)
    _check_method(ac, sett.solver_type, sett.wake_alignment, c["h_m"] is not None)
    strips = _check_mesh(ac, sett)
    _check_call_work(
        5 * (_planned_panels(ac, sett, make_flight_condition(c)) ** 2),
        "fewer panels (smaller n_panels or n_chord)")
    d_a, d_b = 0.5, 1.0
    a, b = c["alpha_deg"], c["beta_deg"]
    cases = {"base": (a, b), "a+": (a + d_a, b), "a-": (a - d_a, b), "b+": (a, b + d_b), "b-": (a, b - d_b)}
    # All cases use one mesh, so that the differences contain no mesh change.
    case_sett, capped = _common_settings(
        ac, sett, strips, [make_flight_condition(c, alpha_deg=aa, beta_deg=bb) for aa, bb in cases.values()])

    def run(item):
        key, (aa, bb) = item
        fc = make_flight_condition(c, alpha_deg=aa, beta_deg=bb)
        try:
            return key, vt.analyze(ac.clone(), fc, case_sett)
        except GroundStrikeError as exc:
            raise GroundStrikeError(f"Case alpha={aa:g} deg, beta={bb:g} deg: {exc}") from exc

    res = dict(run_cases(run, list(cases.items()), estimate_panels(ac, sett), "auto"))

    t = {k: r.totals for k, r in res.items()}
    da, db = np.radians(2 * d_a), np.radians(2 * d_b)
    CL_a = (t["a+"].CL - t["a-"].CL) / da
    CY_b = (t["b+"].CY - t["b-"].CY) / db
    d_m_da_body = np.array([(t["a+"].Cl - t["a-"].Cl) / da,
                            (t["a+"].Cm - t["a-"].Cm) / da,
                            (t["a+"].Cn - t["a-"].Cn) / da], dtype=float)
    d_m_db_body = np.array([(t["b+"].Cl - t["b-"].Cl) / db,
                            (t["b+"].Cm - t["b-"].Cm) / db,
                            (t["b+"].Cn - t["b-"].Cn) / db], dtype=float)
    Cm_a_body = float(d_m_da_body[1])

    if abs(CL_a) > 1e-9:
        x_np = x_cg - c_ref * Cm_a_body / CL_a
        sm = (x_np - x_cg) / c_ref
    else:
        x_np = sm = None

    from ventorum.core.axes import rotation_body_to_stability, rotation_body_to_wind

    a0, b0 = float(np.radians(a)), float(np.radians(b))
    rotations = {"body": np.eye(3),
                 "stability": rotation_body_to_stability(a0),
                 "wind": rotation_body_to_wind(a0, b0)}

    def _derivs_for(name: str) -> dict[str, Any]:
        """Return the rounded derivatives in the axis system *name*."""
        rot = rotations[name]
        d_m_da = rot @ d_m_da_body
        d_m_db = rot @ d_m_db_body
        cm_a, cl_b, cn_b = float(d_m_da[1]), float(d_m_db[0]), float(d_m_db[2])
        return {
            "CL_alpha_per_rad": rnd(CL_a, 5), "CL_alpha_per_deg": rnd(CL_a * np.pi / 180, 6),
            "Cm_alpha_per_rad": rnd(cm_a, 5), "Cm_alpha_per_deg": rnd(cm_a * np.pi / 180, 6),
            "Cl_beta_per_rad": rnd(cl_b, 6), "Cn_beta_per_rad": rnd(cn_b, 6),
            "CY_beta_per_rad": rnd(CY_b, 6),
            "x_cg_m": rnd(x_cg, 5),
            "neutral_point_x_m": rnd(x_np, 5),
            "static_margin_fraction": rnd(sm, 5),
            "c_ref_m": rnd(c_ref, 5),
        }

    by_axes = {name: _derivs_for(name) for name in ("body", "stability", "wind")}
    derivs = by_axes["body"] if ax == "all" else by_axes[ax]
    rot_sel = rotations["body"] if ax == "all" else rotations[ax]
    d_m_da_sel = rot_sel @ d_m_da_body
    d_m_db_sel = rot_sel @ d_m_db_body
    Cm_a_sel = float(d_m_da_sel[1])
    Cl_b_sel = float(d_m_db_sel[0])
    Cn_b_sel = float(d_m_db_sel[2])

    if sm is None:
        pitch = "not available (CL_alpha is near zero)"
    elif sm > 0:
        pitch = (f"statically stable: static margin {sm * 100:+.1f} % of c_ref; the neutral point "
                 f"({x_np:.4f} m) is aft of the CG ({x_cg:.4f} m)")
    elif sm < 0:
        pitch = (f"statically UNSTABLE: static margin {sm * 100:+.1f} % of c_ref; the neutral point "
                 f"({x_np:.4f} m) is ahead of the CG ({x_cg:.4f} m)")
    else:
        pitch = "neutral: static margin 0"
    roll = (f"stable (Cl_beta = {Cl_b_sel:.5f}/rad < 0)" if Cl_b_sel < 0
            else f"not stable (Cl_beta = {Cl_b_sel:.5f}/rad >= 0)")
    yaw = (f"stable (Cn_beta = {Cn_b_sel:.5f}/rad > 0)" if Cn_b_sel > 0
           else f"not stable (Cn_beta = {Cn_b_sel:.5f}/rad <= 0)")

    notes = [
        "Moments about the CG point [x_cg_m, 0, z_ref]. x is aft. Cl > 0 right wing down, Cm > 0 nose up, "
        "Cn > 0 nose right.",
        "static_margin_fraction = (x_np - x_cg)/c_ref = -Cm_alpha/CL_alpha. Positive = stable.",
        "A wing without a fin has a small Cn_beta; its sign alone says little about directional stability.",
        f"Point moments and moment derivatives use the '{ax}' axes: the axes of the reference condition "
        f"(alpha_0={a:g} deg, beta_0={b:g} deg), held fixed while alpha and beta change, so each moment "
        "derivative vector is R(alpha_0, beta_0) times the body-axis vector. CL_alpha and CY_beta are "
        "relative to the free stream in every set. The static margin and the neutral point use the "
        "body-axis Cm_alpha and keep their values in every set.",
    ]
    if c["h_m"] is not None:
        notes.append(f"In ground effect: the CG stays {c['h_m']:g} m above the ground in every case (alpha "
                     "and beta change at constant height). The Irodov height-pitch criterion needs the "
                     "derivatives with respect to height: use the ventorum_ground_effect tool.")

    derivs = by_axes["body"] if ax == "all" else by_axes[ax]
    sm_txt = f"{sm * 100:+.1f} %" if sm is not None else "n/a"
    xnp_txt = f"{x_np:.4f} m" if x_np is not None else "n/a"
    h_txt = "" if c["h_m"] is None else f", h={c['h_m']:g} m"
    device_all, precision_all = computation_device_precision(list(res.values()))
    exec_summary = (
        f"[Ventorum RESULT] Stability of {ac.name} at alpha={a:g} deg, beta={b:g} deg{h_txt}: "
        f"CL_alpha={CL_a:.3f}/rad, Cm_alpha={Cm_a_sel:.4f}/rad about x_cg={x_cg:.4f} m, neutral point "
        f"{xnp_txt}, static margin {sm_txt} of c_ref. Pitch: {pitch.split(':')[0]}. "
        f"Roll (Cl_beta): {roll.split(' (')[0]}. Yaw (Cn_beta): {yaw.split(' (')[0]}. "
        f"{device_summary_text(device_all, precision_all)}"
    )
    _, base_cm, _ = _apply_axes(t["base"].Cl, t["base"].Cm, t["base"].Cn, a, b, ax)
    base_point: dict[str, Any] = {
        "CL": rnd(t["base"].CL), "Cm_about_cg": base_cm, "CDi": rnd(t["base"].CDi, 7)}
    if ax == "all":
        base_point["moments"] = moments_all_sets(
            t["base"].Cl, t["base"].Cm, t["base"].Cn, a, b)
    settings_used = {**settings_payload(sett, res["base"]), **_mesh_note(capped, case_sett)}
    settings_used["device"] = device_all
    settings_used["precision"] = precision_all
    out: dict[str, Any] = {
        "status": "success",
        "executive_summary": exec_summary,
        "stability_derivatives": derivs,
        "stability_assessment": {"pitch": pitch, "roll_Cl_beta": roll, "yaw_Cn_beta": yaw},
        "base_point": base_point,
        "notes": notes,
        "condition_used": condition_payload(c),
        "settings_used": settings_used,
        "trust": trust_payload(t["base"].trust),
        "axes": ax,
    }
    if ax == "all":
        out["derivatives_by_axes"] = by_axes
    if dl == "summary":
        out.pop("trust")
    else:
        out["geometry"] = geometry_payload(ac)
    if dl == "full":
        out["perturbed_cases"] = {}
        for k, v in t.items():
            aa, bb = cases[k]
            cl_p, cm_p, cn_p = _apply_axes(v.Cl, v.Cm, v.Cn, aa, bb, ax)
            device_k, precision_k = result_device_precision(res[k])
            case: dict[str, Any] = {"alpha_deg": aa, "beta_deg": bb, "CL": rnd(v.CL),
                                    "Cm": cm_p, "Cl": cl_p, "Cn": cn_p, "CY": rnd(v.CY),
                                    "device": device_k, "precision": precision_k}
            if ax == "all":
                case["moments"] = moments_all_sets(v.Cl, v.Cm, v.Cn, aa, bb)
            out["perturbed_cases"][k] = case
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Batch evaluation
# ═══════════════════════════════════════════════════════════════════════════════

_OBJECTIVES = ("max_L_over_D", "min_CDi", "max_CL")


@_tool
def batch_evaluate(
    candidates: list[Any],
    flight_condition: dict[str, Any] | None = None,
    objective: str = "max_L_over_D",
    settings: dict[str, Any] | None = None,
    n_workers: int | str = "auto",
    axes: str = "body",
) -> dict[str, Any]:
    """Analyse several candidates at one condition and rank the valid ones.

    A candidate that fails (bad input or solver refusal) is listed in
    ``failed`` with its error. It is never dropped silently. Each ranking
    reports the ``n_chord`` that the solve used, with a ``mesh_note``
    when the automatic value was reduced to the mesh limit. The ``axes``
    input selects the axis system of Cm (``"body"`` by default,
    ``"all"`` adds the three moment sets to each ranking).
    """
    if not isinstance(candidates, list) or not (1 <= len(candidates) <= 50):
        raise InputError("'candidates' must be a list of 1 to 50 wing specs.")
    c = parse_condition(flight_condition)
    sett = parse_settings(settings)
    obj = string(objective, "objective", _OBJECTIVES)
    workers = _workers(n_workers)
    ax = parse_axes(axes)
    in_ground = c["h_m"] is not None
    if in_ground and sett.wake_alignment == "body":
        raise InputError("settings.wake_alignment 'body' is not allowed in ground effect. Use 'freestream'.")
    fc = make_flight_condition(c)

    failed: list[dict[str, Any]] = []
    built: list[tuple[int, str, Any, int]] = []
    for i, cand in enumerate(candidates):
        name = cand.get("name") if isinstance(cand, dict) and isinstance(cand.get("name"), str) \
            else f"candidate_{i + 1}"
        try:
            ac_i = build_aircraft_from_spec(cand, f"candidates[{i}]")
            _check_method(ac_i, sett.solver_type, sett.wake_alignment, in_ground)
            built.append((i, name, ac_i, _check_mesh(ac_i, sett)))
        except Exception as exc:  # noqa: BLE001
            failed.append({"index": i, "name": name, "error": error_info(exc)})

    def run(item):
        i, name, ac, strips = item
        try:
            s = _case_settings(ac, sett, strips, fc)
            return i, name, ac, vt.analyze(ac, fc, s), None, s
        except Exception as exc:  # noqa: BLE001
            return i, name, ac, None, exc, None

    if built:
        _check_call_work(
            sum(_planned_panels(ac_i, sett, fc) ** 2 for _, _, ac_i, _ in built),
            "fewer candidates or fewer panels (smaller n_panels or n_chord)")
    outputs = []
    if built:
        outputs = run_cases(run, built, max((estimate_panels(b[2], sett) or 0) for b in built) or None, workers)

    ranked: list[dict[str, Any]] = []
    ok_results: list[Any] = []
    for i, name, ac, res, err, s_used in outputs:
        if res is None:
            failed.append({"index": i, "name": name, "error": error_info(err)})
            continue
        tot = res.totals
        cd, basis, ld = drag_and_ld(tot)
        S, b = float(ac.S_ref), float(ac.b_ref)
        _, cm_ax, _ = _apply_axes(tot.Cl, tot.Cm, tot.Cn, c["alpha_deg"], c["beta_deg"], ax)
        device, precision = result_device_precision(res)
        row: dict[str, Any] = {
            "index": i, "name": name,
            "CL": rnd(tot.CL), "CDi": rnd(tot.CDi, 7), "CD": rnd(cd, 7), "drag_basis": basis,
            "L_over_D": rnd(ld, 3), "e": rnd(tot.e, 4), "Cm": cm_ax,
            "S_ref_m2": rnd(S, 5), "b_ref_m": rnd(b, 5), "aspect_ratio": rnd(b * b / S, 4) if S > 0 else None,
            "converged": bool(res.converged),
            "trust_score": rnd(tot.trust.score, 3) if tot.trust else None,
            "trust_rating": tot.trust.rating if tot.trust else None,
            "warnings": list(tot.trust.warnings) if tot.trust else [],
            "device": device, "precision": precision,
        }
        ok_results.append(res)
        used_chord = _used_n_chord(res)
        if used_chord is None and s_used is not None and s_used.n_chord is not None:
            used_chord = int(s_used.n_chord)
        if used_chord is not None:
            row["n_chord"] = int(used_chord)
        if s_used is not None and s_used is not sett:
            row.update(_mesh_note(True, s_used))
        if ax == "all":
            row["moments"] = moments_all_sets(
                tot.Cl, tot.Cm, tot.Cn, c["alpha_deg"], c["beta_deg"])
        ranked.append(row)
    failed.sort(key=lambda f: f["index"])

    if not ranked:
        return error_payload(failed[0]["error"]["type"],
                             f"All {len(candidates)} candidates failed. See 'failed' for each error.",
                             failed=failed)

    def key(row):
        v = row["L_over_D"] if obj == "max_L_over_D" else row["CDi"] if obj == "min_CDi" else row["CL"]
        if v is None:
            return float("inf")
        return v if obj == "min_CDi" else -v

    ranked.sort(key=key)
    for r, row in enumerate(ranked, 1):
        row["rank"] = r

    notes = []
    if len({row["drag_basis"] for row in ranked}) > 1:
        notes.append("Some candidates have profile drag and some do not; their L/D values are not comparable.")
    if obj == "max_L_over_D" and all(row["drag_basis"].startswith("CDi only") for row in ranked):
        notes.append("L/D uses induced drag only (no airfoil cd0). The ranking favours low CL and high span.")
    best = ranked[0]
    device_all, precision_all = computation_device_precision(ok_results)
    summary = (f"[Ventorum RESULT] Batch of {len(candidates)} candidates ({len(ranked)} valid, "
               f"{len(failed)} failed), objective {obj}: best is '{best['name']}' with L/D={best['L_over_D']}, "
               f"CL={best['CL']}, CDi={best['CDi']}. "
               f"{device_summary_text(device_all, precision_all)}")
    if failed:
        summary += " Failed: " + ", ".join(f"'{f['name']}' ({f['error']['type']})" for f in failed) + "."
    settings_used = settings_payload(sett)
    settings_used["device"] = device_all
    settings_used["precision"] = precision_all
    return {
        "status": "success",
        "executive_summary": summary,
        "objective": obj,
        "n_candidates": len(candidates),
        "n_succeeded": len(ranked),
        "n_failed": len(failed),
        "rankings": ranked,
        "failed": failed,
        "notes": notes,
        "condition_used": condition_payload(c),
        "settings_used": settings_used,
        "axes": ax,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Mesh convergence
# ═══════════════════════════════════════════════════════════════════════════════

@_tool
def mesh_convergence(
    wing: Any,
    flight_condition: dict[str, Any] | None = None,
    tolerance_pct: float = 0.5,
    target_metric: str = "both",
    panel_counts: list[int] | None = None,
    spacing_schemes: list[str] | None = None,
    alpha_sweep_deg: list[float] | None = None,
    ref_n_panels: int = 160,
    solver: str | None = None,
    detail_level: str = "standard",
) -> dict[str, Any]:
    """Spanwise mesh convergence study (wraps ``run_mesh_convergence_study``)."""
    from ventorum.geometry.mesh_convergence import run_mesh_convergence_study

    ac = build_aircraft_from_spec(wing)
    c = parse_condition(flight_condition)
    dl = parse_detail_level(detail_level)
    tol = number(tolerance_pct, "tolerance_pct", exclusive_min=0, maximum=50)
    metric = string(target_metric, "target_metric", ("both", "CL", "CDi", "circulation"))
    counts = None
    if panel_counts is not None:
        if not isinstance(panel_counts, list) or not (2 <= len(panel_counts) <= 12):
            raise InputError("'panel_counts' must be a list of 2 to 12 integers.")
        counts = [integer(v, f"panel_counts[{i}]", minimum=4, maximum=400) for i, v in enumerate(panel_counts)]
    schemes: tuple[str, ...] = ("auto", "half-cosine", "cosine", "uniform")
    if spacing_schemes is not None:
        if not isinstance(spacing_schemes, list) or not (1 <= len(spacing_schemes) <= 5):
            raise InputError("'spacing_schemes' must be a list of 1 to 5 strings.")
        schemes = tuple(string(v, f"spacing_schemes[{i}]", ("auto", "half-cosine", "cosine", "uniform", "root"))
                        for i, v in enumerate(spacing_schemes))
    alphas = (number_list(alpha_sweep_deg, "alpha_sweep_deg", min_len=1, max_len=15, minimum=-30, maximum=30)
              if alpha_sweep_deg is not None else None)
    ref_n = integer(ref_n_panels, "ref_n_panels", minimum=20, maximum=400)
    finest = max(counts) if counts else 80
    if ref_n <= finest:
        raise InputError(f"ref_n_panels ({ref_n}) must be larger than the finest tested mesh ({finest}).")
    solver_type = (string(solver, "solver", ("auto", "vlm", "linear", "nonlinear", "fourier"))
                   if solver is not None else None)
    _check_method(ac, solver_type, "freestream", c["h_m"] is not None)
    # Every level uses the scaled aircraft (a surface with its own
    # n_panels is scaled with the level). Check each planned mesh
    # against the per-mesh limit, then the total work against the budget.
    from ventorum.geometry.mesh_convergence import _level_aircraft

    _n_base = 80
    if counts is None:
        _levels = [n for n in (10, 15, 20, 30, 40, 60, 80) if n < ref_n] or [10, 20, 40]
    else:
        _levels = [int(n) for n in counts if int(n) < ref_n] or [int(n) for n in counts]
    _levels = list(dict.fromkeys(_levels)) + [int(ref_n)]
    _level_panels: list[int] = []
    for _n_level in _levels:
        _lvl_ac = _level_aircraft(ac, int(_n_level), _n_base)
        _lvl_sett = SolverSettings(solver_type=solver_type or "auto", n_panels=int(_n_level),
                                   proportional_panels=True)
        _cond_alphas = [c["alpha_deg"], *(alphas or [])]
        _n_level_panels = 0
        for _a in _cond_alphas:
            _fc_a = make_flight_condition(c, alpha_deg=_a)
            _strips_a = build_lattice(_lvl_ac, _lvl_sett, collocation="vlm", n_chord=1).n_strips
            if _lvl_sett.n_chord is not None:
                _n_a = int(_strips_a * int(_lvl_sett.n_chord))
            elif resolve_solver_type(_lvl_sett.solver_type) != "vlm":
                _n_a = int(_strips_a)
            else:
                from ventorum.solvers.horseshoe import HorseshoeSolver

                try:
                    _nc = int(HorseshoeSolver().resolve_n_chord(
                        _lvl_ac, _lvl_sett, _fc_a, None, _lvl_ac.moment_reference()))
                except Exception:  # noqa: BLE001 - only an estimate
                    _nc = int(DEFAULT_N_CHORD)
                _n_a = int(_strips_a * _nc)
            _n_level_panels = max(_n_level_panels, _n_a)
        if _n_level_panels > MAX_TOTAL_PANELS:
            raise InputError(
                f"The mesh at level N={_n_level} is too large: {_n_level_panels} panels. "
                f"The limit is {MAX_TOTAL_PANELS} panels. Use a smaller ref_n_panels, "
                "smaller panel_counts, or fewer surface n_panels.")
        _level_panels.append(_n_level_panels)
    if alphas:
        _match = any(abs(float(a) - float(c["alpha_deg"])) < 1e-9 for a in alphas)
        _solves_per_level = len(alphas) if _match else len(alphas) + 1
    else:
        _solves_per_level = 1
    _estimate = _solves_per_level * sum(
        n * n for n in _level_panels[:-1] for _ in schemes) + _solves_per_level * (_level_panels[-1] ** 2)
    _check_call_work(
        _estimate,
        "fewer panel_counts levels, fewer spacing_schemes, fewer alpha_sweep_deg angles, "
        "a smaller ref_n_panels, or fewer panels")

    study = run_mesh_convergence_study(
        case=ac,
        condition=make_flight_condition(c),
        tolerance_pct=tol,
        target_metric=metric,
        spacing_schemes=schemes,
        panel_counts=counts,
        ref_n_panels=ref_n,
        alpha_sweep_deg=alphas,
        evaluate_sweep=True if alphas else None,
        solver_type=solver_type,
    )
    rec, mn = study.recommended_mesh, study.minimal_mesh
    do_sweep = len(study.alpha_tested_deg) > 1
    from ventorum.geometry.mesh_convergence import smallest_reached_error_pct

    smallest_err = smallest_reached_error_pct(study.points, metric, do_sweep)
    study_results = [p.result for p in study.points if p.result is not None]
    if study.reference_result is not None:
        study_results.append(study.reference_result)
    device_all, precision_all = computation_device_precision(study_results) if study_results else ("cpu", "float64")
    if study.converged:
        executive_summary = (
            f"[Ventorum RESULT] Mesh convergence of {ac.name} (tolerance {tol:g} % on {metric}, reference "
            f"N={ref_n}): smallest mesh within tolerance N={mn.panels_solved} panels solved "
            f"(spanwise setting {mn.n_panels} {mn.spacing}); recommended N="
            f"{rec.panels_solved} panels solved (spanwise setting {rec.n_panels} {rec.spacing}) "
            f"(CL error {rec.error_cl_pct:.3f} %, CDi error "
            f"{rec.error_cdi_pct:.3f} %). Errors are relative to the reference mesh, not to experiment. "
            f"{device_summary_text(device_all, precision_all)}"
        )
    else:
        executive_summary = (
            f"[Ventorum RESULT] Mesh convergence of {ac.name}: not converged within tolerance "
            f"{tol:g} % on {metric} (reference N={ref_n}); smallest error reached {smallest_err:.3f} %. "
            f"Finest evaluated mesh N={rec.panels_solved} panels solved "
            f"(spanwise setting {rec.n_panels} {rec.spacing}) "
            f"(CL error {rec.error_cl_pct:.3f} %, CDi error "
            f"{rec.error_cdi_pct:.3f} %). Errors are relative to the reference mesh, not to experiment. "
            f"{device_summary_text(device_all, precision_all)}"
        )
    out: dict[str, Any] = {
        "status": "success",
        "executive_summary": executive_summary,
        "case_name": study.case_name,
        "converged": bool(study.converged),
        "tolerance_pct": tol,
        "target_metric": metric,
        "minimal_mesh": mn.to_dict(),
        "recommended_mesh": rec.to_dict(),
        "recommended_settings": {
            "n_panels": study.recommended_settings.n_panels,
            "spacing": study.recommended_settings.spacing,
        },
        "generalization": study.generalization.to_dict(),
        "condition_used": condition_payload(c),
        "device": device_all,
        "precision": precision_all,
    }
    if dl in ("standard", "full"):
        out["summary_markdown"] = study.summary(as_markdown=True)
    if dl == "full":
        out["all_evaluated_points"] = [p.to_dict() for p in study.points]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Machine capabilities and tuning
# ═══════════════════════════════════════════════════════════════════════════════

def _profile_status(hw_fingerprint: str) -> tuple[dict[str, Any], str]:
    """Return the tuning profile status and one sentence of advice.

    The status is ``"disabled"`` (autotuning switched off), ``"none"`` (no
    profile file), ``"old schema"``, ``"other machine"`` or ``"valid"`` (with
    the creation date and whether the run was quick). *hw_fingerprint* is the
    fingerprint of this machine.
    """
    from ventorum.hardware import profile as _prof

    if _prof.autotune_disabled():
        return ({"status": "disabled"},
                "Autotuning is disabled by VENTORUM_DISABLE_AUTOTUNE, so no tuning profile is used.")
    data = _prof.read_profile()
    if data is None:
        return ({"status": "none"}, "Run ventorum_tune_machine to measure this machine.")
    if data.get("schema") != _prof.SCHEMA_VERSION:
        return ({"status": "old schema"}, "Run ventorum_tune_machine to measure this machine.")
    if data.get("fingerprint") != hw_fingerprint:
        return ({"status": "other machine"}, "Run ventorum_tune_machine to measure this machine.")
    return ({"status": "valid", "created_utc": data.get("created_utc"),
             "quick": bool(data.get("quick", False))},
            "This machine has a valid tuning profile. Run ventorum_tune_machine again after a "
            "hardware change.")


@_tool
def machine_capabilities() -> dict[str, Any]:
    """Hardware capabilities, kernel backends and tuning profile status.

    Takes no input. The hardware data never contains a machine identifier
    or a fingerprint hash. The profile status is ``"none"``,
    ``"valid"``, ``"other machine"``, ``"old schema"`` or ``"disabled"``.
    The GPU pipeline device and precision come from ``ventorum.gpu.info()``.

    Returns
    -------
    dict
        Payload with ``"status"``, ``"executive_summary"``, ``"hardware"``,
        ``"kernel_backends"``, ``"cython_threads"``, ``"torch_device"``,
        ``"gpu"``, ``"gpu_pipeline"``, ``"profile"`` and ``"advice"``.
    """
    from ventorum.hardware import detector as _det
    from ventorum.hardware import tuner as _tuner

    hw = _det.scan_hardware()
    backends = _tuner._available_backends()
    cython_threads = _tuner._cython_threads()
    torch_device = _tuner._torch_device()
    gpu = {
        "available": bool(hw.gpu_available),
        "name": hw.gpu_name,
        "count": int(hw.gpu_count),
        "vram_gb": hw.gpu_vram_gb,
        "cuda_version": hw.cuda_version,
        "compute_capability": (list(hw.cuda_compute_capability)
                               if hw.cuda_compute_capability else None),
    }
    from ventorum import gpu as _gpu

    gpu_pipeline = _gpu.info()
    profile, advice = _profile_status(hw.fingerprint)
    gpu_txt = hw.gpu_name if hw.gpu_available else "no GPU"
    summary = (
        f"[Ventorum RESULT] Machine: {hw.system} {hw.machine}, {hw.logical_cores} threads "
        f"({hw.physical_cores} cores), {hw.total_ram_gb:g} GB RAM, GPU: {gpu_txt}. "
        f"Kernel backends: {', '.join(backends)}. Cython threads: {cython_threads}. "
        f"Torch device: {torch_device}. GPU pipeline: device {gpu_pipeline['device']}, "
        f"precision {gpu_pipeline['precision']}. Profile: {profile['status']}. {advice}"
    )
    return {
        "status": "success",
        "executive_summary": summary,
        "hardware": hw.public_dict(),
        "kernel_backends": backends,
        "cython_threads": cython_threads,
        "torch_device": torch_device,
        "gpu": gpu,
        "gpu_pipeline": gpu_pipeline,
        "profile": profile,
        "advice": advice,
    }


@_tool
def tune_machine(quick: bool = True, save: bool = True) -> dict[str, Any]:
    """Measure this machine and return (and by default save) its tuning profile.

    A second call while one runs is refused with ``"invalid_input"``. This
    tool is outside the work budget: it takes about half a minute with
    *quick* true and up to two minutes with *quick* false. Tuning changes
    the speed; the device choice can change results at the float32
    round-off level (about 1e-7 to 1e-6 relative).

    Parameters
    ----------
    quick : bool
        Skip the large cases (about 3 times faster). Default True.
    save : bool
        Write the profile to the user configuration folder. Default True.

    Returns
    -------
    dict
        Payload with ``"status"``, ``"executive_summary"``, the kernel
        choices per size class (``"kernels"``), the thread and batch settings
        (``"single"``, ``"batch"``) and the tuning time
        (``"tuning_seconds"``).
    """
    q = boolean(quick, "quick")
    s = boolean(save, "save")
    if not _TUNE_LOCK.acquire(blocking=False):
        raise InputError("a tuning run is in progress")
    try:
        from ventorum.hardware.tuner import tune_machine as run_tuner

        profile = run_tuner(quick=q, save=s, verbose=False)
    finally:
        _TUNE_LOCK.release()
    settings = profile.get("settings", {})
    kernels = settings.get("kernels", {})
    seconds = profile.get("tuning_seconds")
    out: dict[str, Any] = {
        "status": "success",
        "executive_summary": (
            f"[Ventorum RESULT] Tuned this machine in {fmt(seconds, '.1f')} s "
            f"(quick={q}, saved={s}): kernel backends per size class for "
            f"{', '.join(sorted(kernels))}. Tuning changes the speed; the device choice can "
            "change results at the float32 round-off level (about 1e-7 to 1e-6 relative)."
        ),
        "kernels": kernels,
        "single": settings.get("single", {}),
        "batch": settings.get("batch", {}),
        "tuning_seconds": seconds,
        "quick": bool(profile.get("quick", q)),
    }
    if s:
        from ventorum.hardware.profile import profile_path

        out["profile_path"] = str(profile_path())
    return out
