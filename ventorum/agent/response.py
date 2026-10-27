# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Shared helpers that build the tool payloads.

Every payload has ``"status": "success"`` or ``"status": "error"``. An
error payload has ``"error": {"type": ..., "message": ...}`` with the type
``invalid_input``, ``ground_strike``, ``invalid_method`` or ``internal``.
All payloads go through :func:`ventorum.utils.jsonsafe.json_safe`, so
NaN and infinity become null.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from ventorum.core.datatypes import Aircraft, IntegratedResult, SolverResult, TrustScore
from ventorum.core.errors import GroundStrikeError, ValidityError
from ventorum.utils.jsonsafe import json_safe

ERROR_TYPES = ("invalid_input", "ground_strike", "invalid_method", "internal")

TRUST_NOTE = ("The bands are heuristic estimates from the validity checks. They are NOT calibrated error "
              "bounds. Use the score, the warnings and the recommendations as guidance only.")


def error_type_of(exc: BaseException) -> str:
    """Map an exception to one of the error types."""
    if isinstance(exc, np.linalg.LinAlgError):
        # A singular system is refused as invalid input, not an internal failure.
        return "invalid_input"
    if isinstance(exc, GroundStrikeError):
        return "ground_strike"
    if isinstance(exc, ValidityError):
        return "invalid_method"
    if isinstance(exc, ValueError):  # includes InputError and VentorumError
        return "invalid_input"
    return "internal"


def error_info(exc: BaseException) -> dict[str, str]:
    """Return the ``error`` object for the exception *exc*."""
    kind = error_type_of(exc)
    msg = str(exc) or type(exc).__name__
    if isinstance(exc, np.linalg.LinAlgError):
        msg = (f"{msg} (the linear system is singular; this usually happens with overlapping "
               "or degenerate geometry. Check the mesh for duplicate or zero-area panels.)")
    elif kind == "internal":
        msg = f"{type(exc).__name__}: {msg}"
    return {"type": kind, "message": msg}


def error_payload(exc_or_type: BaseException | str, message: str | None = None, **extra: Any) -> dict[str, Any]:
    """Return a complete error payload.

    Parameters
    ----------
    exc_or_type : BaseException or str
        An exception, or the name of one of the error types.
    message : str or None, optional
        Error message. It is used only when *exc_or_type* is a type name.
    **extra : Any
        Additional keys for the payload.

    Returns
    -------
    dict
        JSON-safe payload with ``"status": "error"``.

    Raises
    ------
    ValueError
        If *exc_or_type* is a string that is not a known error type.
    """
    if isinstance(exc_or_type, BaseException):
        err = error_info(exc_or_type)
    else:
        if exc_or_type not in ERROR_TYPES:
            raise ValueError(f"Unknown error type {exc_or_type!r}.")
        err = {"type": exc_or_type, "message": message or ""}
    return json_safe({"status": "error", "error": err, **extra})


def rnd(x: Any, n: int = 6) -> float | None:
    """Round a number; None for None, NaN or infinity."""
    if x is None:
        return None
    v = float(x)
    return round(v, n) if math.isfinite(v) else None


def fmt(x: Any, spec: str, none: str = "n/a") -> str:
    """Format a number with *spec*. Give *none* for None, NaN or infinity."""
    v = rnd(x, 12)
    return none if v is None else format(float(x), spec)


def lift_to_drag(cl: Any, cd: Any) -> float | None:
    """L/D, or None if the drag is zero (L/D is not defined)."""
    if cl is None or cd is None:
        return None
    cd = float(cd)
    return float(cl) / cd if cd > 1e-12 else None


def trust_payload(trust: TrustScore | None) -> dict[str, Any]:
    """Trust assessment with the uncertainty bands labelled as heuristic."""
    if trust is None:
        return {"available": False, "note": "The solver gave no trust assessment."}
    return {
        "score": rnd(trust.score, 4),
        "rating": trust.rating,
        "warnings": list(trust.warnings),
        "recommendations": list(trust.recommendations),
        "factors": {k: rnd(v, 4) for k, v in trust.factors.items()},
        "heuristic_bands_not_calibrated": {
            "CL": rnd(trust.uncertainty_CL, 4),
            "CDi": rnd(trust.uncertainty_CDi, 6),
            "L_over_D": rnd(trust.uncertainty_LD, 2),
        },
        "note": TRUST_NOTE,
    }


#: Largest number of warning labels in one table row of a compact payload.
ROW_WARNINGS_CAP = 3
#: Largest number of warnings in the summary of a compact payload.
SUMMARY_WARNINGS_CAP = 6

_HEAD_RE = re.compile(r"^(.*?)(?::\s|\s\(|\.\s|\.$|$)")


def warning_label(text: str) -> str:
    """Return the short label of a trust warning.

    The label is the text before the first colon, opening parenthesis or
    end of sentence, for example ``"Severe stall risk"`` for
    ``"Severe stall risk: peak section Cl=1.72 (...)"``. Warnings of the
    same kind at different conditions have the same label.

    Parameters
    ----------
    text : str
        Full warning text.

    Returns
    -------
    str
        The label, or the full text if no label is found.
    """
    m = _HEAD_RE.match(text.strip())
    head = m.group(1).strip() if m else ""
    return head or text.strip()


def row_warnings(warnings: list[str], cap: int = ROW_WARNINGS_CAP) -> tuple[list[str], int]:
    """Return the deduplicated warning labels of one row and the count of the labels not shown.

    Parameters
    ----------
    warnings : list of str
        Full warning texts of the row.
    cap : int
        Largest number of labels to return.

    Returns
    -------
    tuple of (list of str, int)
        At most *cap* labels in their first order, and the number of the
        other labels.
    """
    labels = list(dict.fromkeys(warning_label(w) for w in warnings))
    return labels[:cap], max(0, len(labels) - cap)


def merge_warnings(items: list[tuple[Any, float | None, list[str]]], at_key: str,
                   cap: int = SUMMARY_WARNINGS_CAP) -> tuple[list[dict[str, Any]], int]:
    """Return the union of the warnings of several results, grouped by label.

    Parameters
    ----------
    items : list of (Any, float or None, list of str)
        One item per result: its position (for example the angle of
        attack [deg]), its trust score and its full warning texts.
    at_key : str
        Name of the key that lists the positions of each warning, for
        example ``"alpha_deg"``.
    cap : int
        Largest number of warnings to return.

    Returns
    -------
    tuple of (list of dict, int)
        One dict per label: ``"warning"``, the full text at the position
        with the lowest trust score, and *at_key*, the positions that have
        the warning. The labels of the lowest score come first. The int is
        the number of the labels not shown.
    """
    groups: dict[str, dict[str, Any]] = {}
    for pos, score, warns in items:
        s = float("inf") if score is None else float(score)
        for w in warns:
            lab = warning_label(w)
            g = groups.get(lab)
            if g is None:
                groups[lab] = {"text": w, "score": s, "at": [pos]}
                continue
            if pos not in g["at"]:
                g["at"].append(pos)
            if s < g["score"]:
                g["text"], g["score"] = w, s
    order = sorted(groups.values(), key=lambda g: g["score"])
    out = [{"warning": g["text"], at_key: g["at"]} for g in order[:cap]]
    return out, max(0, len(order) - cap)


def compact_trust(trust: TrustScore | dict[str, Any] | None,
                  cap: int = SUMMARY_WARNINGS_CAP) -> dict[str, Any]:
    """Return the trust score, the rating and the deduplicated warnings, without the bands.

    Parameters
    ----------
    trust : TrustScore, dict or None
        Trust assessment of the result, or a payload of
        :func:`trust_payload` (for example with the warnings of several
        cases).
    cap : int
        Largest number of warnings to return.

    Returns
    -------
    dict
        ``score``, ``rating``, ``warnings`` (full texts, deduplicated, at
        most *cap*) and ``warnings_not_shown``; or ``available`` false when
        there is no trust assessment.
    """
    if isinstance(trust, dict):
        if "rating" not in trust:
            return dict(trust)
        score, rating, warns = trust.get("score"), trust["rating"], trust.get("warnings", [])
    elif trust is None:
        return {"available": False, "note": "The solver gave no trust assessment."}
    else:
        score, rating, warns = trust.score, trust.rating, trust.warnings
    warns = list(dict.fromkeys(warns))
    return {"score": rnd(score, 4), "rating": rating, "warnings": warns[:cap],
            "warnings_not_shown": max(0, len(warns) - cap)}


def drag_and_ld(tot: IntegratedResult) -> tuple[float, str, float | None]:
    """Return (CD, drag basis, L/D). CD includes profile drag when the airfoils give it."""
    if tot.CD_total is not None and tot.CDp is not None:
        cd = float(tot.CD_total)
        basis = "CD_total (induced + profile)"
    else:
        cd = float(tot.CDi)
        basis = "CDi only: the airfoils have no profile drag, cd0 = 0"
    return cd, basis, lift_to_drag(tot.CL, cd)


def metrics_payload(res: SolverResult) -> dict[str, Any]:
    """Integrated coefficients of one solution."""
    tot = res.totals
    cd, basis, ld = drag_and_ld(tot)
    out: dict[str, Any] = {
        "CL": rnd(tot.CL),
        "CDi": rnd(tot.CDi, 7),
        "CDp": rnd(tot.CDp, 7),
        "CD": rnd(cd, 7),
        "drag_basis": basis,
        "L_over_D": rnd(ld, 3),
        "e": rnd(tot.e, 4),
        "AR": rnd(tot.AR, 4),
        "Cm": rnd(tot.Cm),
        "Cl": rnd(tot.Cl),
        "Cn": rnd(tot.Cn),
        "CY": rnd(tot.CY),
        "solver": res.solver_type,
        "converged": bool(res.converged),
    }
    # Far-field lateral diagnostics (Trefftz plane) after the other keys, in
    # the order that wing_analysis already used, so that its output does not change.
    if tot.CY_farfield is not None:
        out["CY_farfield"] = rnd(tot.CY_farfield)
    if tot.Cn_trefftz is not None:
        out["Cn_trefftz"] = rnd(tot.Cn_trefftz)
    return out


def geometry_payload(ac: Aircraft) -> dict[str, Any]:
    """Return the reference values and the surfaces of the aircraft *ac*."""
    ac.compute_reference_values()
    S, b = float(ac.S_ref), float(ac.b_ref)
    return {
        "S_ref_m2": rnd(S, 5),
        "b_ref_m": rnd(b, 5),
        "c_ref_m": rnd(ac.c_ref, 5),
        "aspect_ratio": rnd(b * b / S, 4) if S > 0 else None,
        "ref_point_m": [rnd(v, 5) for v in ac.moment_reference()],
        "surfaces": [
            {"name": s.name, "symmetric": bool(s.is_symmetric), "mirror_copy": bool(s.mirror_y),
             "semi_span_m": rnd(s.semi_span, 5), "n_sections": len(s.sections)}
            for s in ac.surfaces
        ],
    }


def sectional_payload(res: SolverResult) -> dict[str, Any]:
    """Peak sectional values over all surfaces."""
    best: dict[str, Any] = {"peak_sectional_Cl": None, "surface": None, "y_m": None,
                            "max_alpha_eff_deg": None}
    peak = -1.0
    max_ae = None
    for sw in res.spanwise:
        if len(sw.Cl):
            i = int(np.argmax(np.abs(sw.Cl)))
            if abs(float(sw.Cl[i])) > peak:
                peak = abs(float(sw.Cl[i]))
                best.update(peak_sectional_Cl=rnd(sw.Cl[i], 4), surface=sw.surface_name, y_m=rnd(sw.y[i], 4))
        if len(sw.alpha_eff):
            v = float(np.degrees(np.max(sw.alpha_eff)))
            max_ae = v if max_ae is None else max(max_ae, v)
    best["max_alpha_eff_deg"] = rnd(max_ae, 3)
    best["note"] = ("Sectional values from the solver strips. The linear airfoil model has no stall; "
                    "compare peak_sectional_Cl with the real airfoil Cl_max.")
    return best


def dispatch_payload(res: Any | None) -> dict[str, Any] | None:
    """Return a copy of the dispatch record of a result, for the full agent output.

    Parameters
    ----------
    res : SolverResult, object with a ``solver_result`` attribute, or None
        The result. An object without ``details`` (for example a
        ``GroundEffectResult``) is read through its ``solver_result``.

    Returns
    -------
    dict or None
        A copy of ``details["dispatch"]`` (see the user manual, results
        page), or None when the result has no dispatch record (for example
        the Fourier solver) or *res* is None. The agent tools give it only
        at ``detail_level="full"``, under the key ``"dispatch"``.

    Notes
    -----
    A CPU record has ``device`` ``"cpu"``, ``precision`` ``"float64"``,
    ``backends`` with the four vortex kernels (``"tensor"``,
    ``"influence"``, ``"induced"``, ``"trefftz"``), and integer
    ``threads`` and ``workers`` with their ``plan_source``. A GPU record has
    ``device`` ``"gpu"``, the GPU precision, ``backends`` with the kernel
    groups that ran on the GPU for the solver family (``"warp"``) and
    ``"dense_solve": "torch"``, and None for ``threads``, ``workers`` and
    ``plan_source`` (the CPU thread plan does not apply).
    """
    target = res
    if target is not None and not hasattr(target, "details") and hasattr(target, "solver_result"):
        target = target.solver_result
    details = getattr(target, "details", None)
    if isinstance(details, dict) and isinstance(details.get("dispatch"), dict):
        rec = dict(details["dispatch"])
        if isinstance(rec.get("backends"), dict):
            rec["backends"] = dict(rec["backends"])
        return rec
    return None


def spanwise_payload(res: SolverResult) -> list[dict[str, Any]]:
    """Spanwise arrays per surface."""
    out = []
    for sw in res.spanwise:
        out.append({
            "surface_name": sw.surface_name,
            "y_m": [rnd(v, 5) for v in sw.y],
            "chord_m": [rnd(v, 5) for v in sw.chord] if sw.chord is not None else None,
            "gamma_m2_s": [rnd(v, 6) for v in sw.gamma],
            "Cl": [rnd(v, 5) for v in sw.Cl],
            "Cd_i": [rnd(v, 7) for v in sw.Cd_i],
            "alpha_eff_deg": [rnd(np.degrees(v), 4) for v in sw.alpha_eff],
        })
    return out


def condition_payload(c: dict[str, Any]) -> dict[str, Any]:
    """Return the flight condition that the tool used."""
    return {
        "V_inf_m_s": rnd(c["V_inf_m_s"], 4),
        "alpha_deg": rnd(c.get("alpha_deg"), 4),
        "beta_deg": rnd(c["beta_deg"], 4),
        "rho_kg_m3": rnd(c["rho_kg_m3"], 5),
        "density_source": c["density_source"],
        "h_m": rnd(c["h_m"], 5),
        "ground": "free air" if c["h_m"] is None else
                  f"flat ground {c['h_m']:g} m below the moment reference point",
    }


def settings_payload(s: Any, res: SolverResult | None = None) -> dict[str, Any]:
    """Return the solver settings that the tool used.

    The payload states the device (``"cpu"`` or ``"gpu"``) and the
    precision (``"float32"`` or ``"float64"``) of the result. A CPU
    solve has no device keys in its details, so it reports ``"cpu"``
    and ``"float64"``.
    """
    out = {
        "solver_requested": s.solver_type,
        "n_panels": int(s.n_panels),
        "n_chord": s.n_chord if s.n_chord is not None else "auto",
        "wake_alignment": s.wake_alignment,
    }
    device, precision = result_device_precision(res)
    out["device"] = device
    out["precision"] = precision
    if res is not None:
        target = res
        if not hasattr(target, "details") and hasattr(target, "solver_result"):
            target = target.solver_result
        if hasattr(target, "solver_type"):
            out["solver_used"] = target.solver_type
        details = target.details if hasattr(target, "details") else None
        lat = details.get("lattice") if isinstance(details, dict) else None
        if lat is not None and hasattr(lat, "n_chord"):
            out["n_chord_used"] = int(lat.n_chord)
    return out


def result_device_precision(res: Any | None) -> tuple[str, str]:
    """Return the (device, precision) of a solver result.

    A GPU solve stores ``"gpu"`` and ``"float32"`` or ``"float64"`` in
    its details. A CPU solve stores no keys, so it reports ``"cpu"``
    and ``"float64"``. A ground-effect result reports the device of its
    solver result.
    """
    try:
        target = res
        if target is not None and not hasattr(target, "details") and hasattr(target, "solver_result"):
            target = target.solver_result
        details = target.details if target is not None and hasattr(target, "details") else None
        if isinstance(details, dict):
            device = details.get("device", "cpu")
            precision = details.get("precision", "float64")
            if device not in ("cpu", "gpu"):
                device = "cpu"
            if precision not in ("float32", "float64"):
                precision = "float64"
            return str(device), str(precision)
    except Exception:  # noqa: BLE001 - a bad details object reports the CPU default
        pass
    return "cpu", "float64"


def computation_device_precision(results: Any) -> tuple[str, str]:
    """Return the common (device, precision) of solver results, or mixed.

    *results* is one result or a list of results. When all results share
    the same device and precision, return them. When they differ, return
    ``"mixed"`` for the value that differs.
    """
    items = list(results) if isinstance(results, (list, tuple)) else [results]
    devices = {result_device_precision(r)[0] for r in items}
    precisions = {result_device_precision(r)[1] for r in items}
    device = next(iter(devices)) if len(devices) == 1 else "mixed"
    precision = next(iter(precisions)) if len(precisions) == 1 else "mixed"
    return device, precision


def device_summary_text(device: str, precision: str) -> str:
    """Return the sentence that states the device and precision of a result."""
    if device == "mixed" or precision == "mixed":
        return f"Computed on mixed devices in mixed precision (device {device}, precision {precision})."
    return f"Computed on {device} in {precision}."


def moments_in_axes(
    Cl: float,
    Cm: float,
    Cn: float,
    alpha_deg: float,
    beta_deg: float,
    axes: str,
    *,
    b_ref: float,
    c_ref: float,
) -> tuple[float, float, float]:
    """Return the body-axis triple (Cl, Cm, Cn) expressed in *axes*.

    *alpha_deg* and *beta_deg* are the angle of attack and the sideslip
    angle [deg]. *axes* is ``"body"``, ``"stability"`` or ``"wind"``.
    *b_ref* and *c_ref* [m] are the reference span and chord.
    """
    from ventorum.core.axes import transform_moments

    m = transform_moments(Cl, Cm, Cn, float(np.radians(alpha_deg)), float(np.radians(beta_deg)),
                          axes, b_ref=b_ref, c_ref=c_ref)
    return float(m["Cl"]), float(m["Cm"]), float(m["Cn"])


def moments_all_sets(
    Cl: float,
    Cm: float,
    Cn: float,
    alpha_deg: float,
    beta_deg: float,
    *,
    b_ref: float,
    c_ref: float,
) -> dict[str, dict[str, float | None]]:
    """Return the moment triple in the body, stability and wind axes.

    *b_ref* and *c_ref* [m] are the reference span and chord.
    """
    from ventorum.core.axes import transform_moments

    alpha, beta = float(np.radians(alpha_deg)), float(np.radians(beta_deg))
    return {
        name: {k: rnd(v) for k, v in transform_moments(Cl, Cm, Cn, alpha, beta, name, b_ref=b_ref, c_ref=c_ref).items()}
        for name in ("body", "stability", "wind")
    }
