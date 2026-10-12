# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Strict scoring of one agent run of one task.

Final answer: the last ``<final_answer>{...}</final_answer>`` block of the
last message, parsed as strict JSON (NaN refused). If strict parsing fails,
one repair is tried (code fences, ``//`` comments, trailing commas, Python
literals); a repaired block gets the code ``format_repaired``.

A task run succeeds when one accepted outcome of the task matches:

1. the status is in the accepted set of the outcome;
2. each answer value is within its tolerance of the reference;
3. the required caveat codes are present;
4. provenance: for an outcome with values, at least one call to an accepted
   tool returned ``success``, and each reported number is within its
   tolerance of a number in a tool result (or of the task formula applied
   to a tool result). A right number with no source fails
   (``unsupported_value``);
5. the argument checks of the task pass (for example the span in metres).

The geometry and the flight condition of the model calls are compared with
the canonical ones after both are built with
:func:`ventorum.agent.build_aircraft_from_spec` and
:func:`ventorum.agent.build_flight_condition_from_spec`, so equivalent inputs
(``span_m`` or ``semi_span_m``, ``chord_m`` or two equal ``sections``) are
equal.

Error kinds (one run can have several): ``invalid_input:<sub>``,
``invalid_method``, ``ground_strike``, ``protocol``, ``wrong_tool``,
``no_tool``, ``wrong_units``, ``sign_error``, ``geometry_drift``,
``hallucinated_value``, ``unsupported_value``, ``missed_refusal``,
``false_refusal``, ``format_error``, ``format_repaired``, ``limit:<kind>``,
``context_overflow``, ``endpoint_error``. Three diagnostic codes are added:
``wrong_value`` (a wrong value that is in a tool result), ``missing_value``
(an answer key with no value) and ``tool_internal`` (a tool reply with the
error type ``internal``).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np

from validation.agent_benchmark.references import evaluate_reference
from validation.agent_benchmark.tasks import STATUSES, Tolerance, TaskVariant

FINAL_BLOCK = re.compile(r"<final_answer>(.*?)</final_answer>", re.S | re.I)
FINAL_OPEN = re.compile(r"<final_answer>(.*)$", re.S | re.I)
#: One number, then an optional unit text (letters, the degree sign, ``/``, ``%``, ``^``, digits).
ONE_NUMBER = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-z\u00b0\u00b5/%^0-9_ .*()\-]*)$")
#: Unit exponents (``m^2``, ``rad^-1``) and digits after a letter (``m2``). Other digits are a second number.
UNIT_DIGITS = re.compile(r"\^[-+]?\d+|(?<=[A-Za-z])\d+")
#: Strings that mean "no value" (compared after strip and lower case).
PLACEHOLDERS = frozenset({"", "n/a", "na", "unknown", "none", "null", "-"})
#: Minus signs that are not ASCII (Unicode minus, figure dash, small and full-width hyphen-minus).
MINUS_SIGNS = {"\u2212": "-", "\u2012": "-", "\ufe63": "-", "\uff0d": "-"}
DEG = 180.0 / math.pi
UNIT_FACTORS = (DEG, 1.0 / DEG, 0.3048, 1.0 / 0.3048, 0.514444, 1.0 / 0.514444, 1000.0, 0.001, 100.0, 0.01)
UNIT_RATIO_TOL = 0.005
GEOMETRY_RTOL = 2e-3
EXCLUDED_ARG_PATHS = ("wing", "aircraft", "flight_condition", "settings", "candidates", "detail_level", "axes")


# ---------------------------------------------------------------------------
# Final answer
# ---------------------------------------------------------------------------

def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _repair(text: str) -> str:
    s = text.strip()
    s = re.sub(r"^```[A-Za-z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = re.sub(r"(?m)(^|[\s,{\[])//[^\n]*$", r"\1", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    return s


def extract_final_answer(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Extract the final-answer object from the last message.

    Parameters
    ----------
    text : str
        Text of the last model message.

    Returns
    -------
    tuple
        ``(answer, codes)``. ``answer`` is ``{"status", "values",
        "caveats"}`` or None; ``codes`` holds ``format_error`` or
        ``format_repaired`` if they apply.
    """
    blocks = FINAL_BLOCK.findall(text or "")
    repaired_needed = False
    if blocks:
        body = blocks[-1]
    else:
        m = FINAL_OPEN.search(text or "")
        if not m:
            return None, ["format_error"]
        body = m.group(1)
        repaired_needed = True
    obj = None
    if not repaired_needed:
        try:
            obj = json.loads(body.strip(), parse_constant=_no_constant)
        except ValueError:
            obj = None
    codes: list[str] = []
    if obj is None:
        try:
            obj = json.loads(_repair(body), parse_constant=_no_constant)
            codes.append("format_repaired")
        except ValueError:
            return None, ["format_error"]
    if not isinstance(obj, dict) or obj.get("status") not in STATUSES:
        return None, ["format_error"]
    values = obj.get("values") if isinstance(obj.get("values"), dict) else {}
    raw_caveats = obj.get("caveats", [])
    if isinstance(raw_caveats, str):
        # One code as a string instead of a list: accept it as a one-element list, and flag the repair.
        raw_caveats = [raw_caveats.strip()] if raw_caveats.strip() else []
        if "format_repaired" not in codes:
            codes.append("format_repaired")
    caveats = [c for c in raw_caveats if isinstance(c, str)] if isinstance(raw_caveats, list) else []
    return {"status": obj["status"], "values": values, "caveats": caveats}, codes


def _clean_number_text(v: str) -> str:
    for k, rep in MINUS_SIGNS.items():
        v = v.replace(k, rep)
    return v


def is_number_range(v: Any) -> bool:
    """Return True if *v* is a string that starts with a number and holds a second number (``"3 to 5"``)."""
    if not isinstance(v, str):
        return False
    m = ONE_NUMBER.match(_clean_number_text(v))
    return bool(m) and bool(re.search(r"\d", UNIT_DIGITS.sub("", m.group(2))))


def normalise_value(v: Any) -> tuple[Any, bool]:
    """Return ``(value, as_string)``.

    A placeholder string (``"n/a"``, ``""``, ``"unknown"``, ``"none"``,
    ``"null"``, ``"na"``, ``"-"``, any case) gives None. A string with one
    number and an optional unit (``"4.8 /rad"``, ``"5.5\u00b0"``, also with a
    Unicode minus) or a boolean word is converted. A string with two numbers
    (a range such as ``"3 to 5"``) is not converted.
    """
    if isinstance(v, str):
        s = v.strip().lower()
        if s in PLACEHOLDERS:
            return None, False
        if s in ("true", "yes"):
            return True, True
        if s in ("false", "no"):
            return False, True
        if not is_number_range(v):
            m = ONE_NUMBER.match(_clean_number_text(v))
            if m:
                return float(m.group(1)), True
    if isinstance(v, int) and not isinstance(v, bool):
        return float(v), False
    return v, False


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def value_matches(value: Any, ref: Any, tol: Tolerance) -> bool:
    """Return True if *value* agrees with the reference *ref* within *tol*."""
    if value is None:
        return False
    if tol.kind == "set":
        if not isinstance(value, list) or not isinstance(ref, list):
            return False
        return {str(x).strip().lower() for x in value} == {str(x).strip().lower() for x in ref}
    if isinstance(ref, bool):
        return isinstance(value, bool) and value == ref
    if tol.kind == "exact":
        if _is_num(ref):
            return _is_num(value) and abs(float(value) - float(ref)) <= 1e-9 * max(1.0, abs(float(ref)))
        return value == ref
    if not (_is_num(value) and _is_num(ref)):
        return False
    return abs(float(value) - float(ref)) <= tol.limit(float(ref))


def unit_or_sign(value: float, ref: float, tol: Tolerance) -> str | None:
    """Return ``"wrong_units"`` or ``"sign_error"`` if *value* is *ref* with a unit factor or the wrong sign."""
    if not (_is_num(value) and _is_num(ref)) or ref == 0:
        return None
    lim = max(tol.limit(ref) if tol.kind == "numeric" else 0.0, UNIT_RATIO_TOL * abs(ref))
    if abs(value + ref) <= lim:
        return "sign_error"
    for f in UNIT_FACTORS:
        if abs(value - f * ref) <= max(UNIT_RATIO_TOL * abs(f * ref), tol.limit(f * ref) if tol.kind == "numeric" else 0):
            return "wrong_units"
    return None


def numbers_in(obj: Any) -> list[float]:
    """Return all finite numbers (not booleans) in a JSON object."""
    out: list[float] = []
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)
        elif _is_num(o):
            out.append(float(o))
    return out


# ---------------------------------------------------------------------------
# Geometry and condition comparison
# ---------------------------------------------------------------------------

_ETA = np.linspace(0.0, 1.0, 41)


def _surface_print(surf: Any) -> dict[str, Any]:
    from ventorum.geometry.lattice import surface_reference_line

    secs = sorted(surf.sections, key=lambda s: s.y_frac)
    fr = np.array([s.y_frac for s in secs])
    x_le, y, z = surface_reference_line(surf, _ETA)
    pos = np.asarray(surf.position, dtype=float)

    def af(attr: str) -> np.ndarray:
        return np.interp(_ETA, fr, np.array([float(getattr(s.airfoil, attr, 0.0)) for s in secs]))

    return {
        "x_le": x_le + pos[0], "y": y + pos[1], "z": z + pos[2],
        "chord": np.interp(_ETA, fr, np.array([s.chord for s in secs])),
        "twist": np.interp(_ETA, fr, np.array([s.twist for s in secs])) + surf.incidence,
        "a0": af("a0"), "alpha_L0": af("alpha_L0"), "Cd0": af("Cd0"), "Cm0": af("Cm0"),
        "flags": (bool(surf.is_symmetric), bool(surf.mirror_y)),
        "controls": sorted((round(c.eta_start, 4), round(c.eta_end, 4), round(c.hinge_x_c, 4), bool(c.symmetric))
                           for c in surf.controls),
    }


def geometry_print(ac: Any) -> dict[str, Any]:
    """Return a mesh-free fingerprint of a built aircraft (lengths in m, angles in rad)."""
    surfs = [_surface_print(s) for s in ac.surfaces]
    surfs.sort(key=lambda d: (round(float(d["x_le"][0]), 3), round(float(d["z"][-1]), 3), round(float(d["y"][-1]), 3)))
    rp = np.asarray(ac.moment_reference(), dtype=float).reshape(3)
    return {"surfaces": surfs, "refs": np.array([ac.S_ref, ac.b_ref, ac.c_ref], dtype=float), "ref_point": rp}


def classify_arrays(model: np.ndarray, canon: np.ndarray) -> str | None:
    """Compare two arrays. Return None if equal, else ``sign_error``, ``wrong_units`` or ``geometry_drift``."""
    a = np.atleast_1d(np.asarray(model, dtype=float))
    b = np.atleast_1d(np.asarray(canon, dtype=float))
    if a.shape != b.shape:
        return "geometry_drift"
    scale = max(float(np.max(np.abs(b))) if b.size else 0.0, 1e-9)
    if np.all(np.abs(a - b) <= GEOMETRY_RTOL * scale + 1e-9):
        return None
    nz = np.abs(b) > 1e-9 * scale
    if np.any(nz) and np.all(np.abs(a[~nz]) <= GEOMETRY_RTOL * scale + 1e-9):
        if np.all(np.abs(a[nz] + b[nz]) <= GEOMETRY_RTOL * np.abs(b[nz]) + 1e-9):
            return "sign_error"
        ratio = a[nz] / b[nz]
        for f in UNIT_FACTORS:
            if np.all(np.abs(ratio - f) <= UNIT_RATIO_TOL * f):
                return "wrong_units"
    return "geometry_drift"


def compare_geometry(model: dict[str, Any], canon: dict[str, Any]) -> set[str]:
    """Compare two fingerprints of :func:`geometry_print`. Return the error codes (empty if equal)."""
    codes: set[str] = set()
    if len(model["surfaces"]) != len(canon["surfaces"]):
        return {"geometry_drift"}
    for ms, cs in zip(model["surfaces"], canon["surfaces"]):
        if ms["flags"] != cs["flags"] or ms["controls"] != cs["controls"]:
            codes.add("geometry_drift")
        for key in ("x_le", "y", "z", "chord", "twist", "a0", "alpha_L0", "Cd0", "Cm0"):
            c = classify_arrays(ms[key], cs[key])
            if c:
                codes.add(c)
    for key in ("refs", "ref_point"):
        c = classify_arrays(model[key], canon[key])
        if c:
            codes.add(c)
    return codes


def _condition_of(tool: str, args: dict[str, Any]) -> dict[str, float]:
    from ventorum.agent.schemas import AGENT_TOOL_DEFINITIONS, parse_condition

    props = {t["name"]: t["parameters"]["properties"] for t in AGENT_TOOL_DEFINITIONS}.get(tool, {})
    out: dict[str, float] = {}
    if "flight_condition" in props:
        allowed = props["flight_condition"].get("properties")
        allow = tuple(allowed) if isinstance(allowed, dict) else None
        c = parse_condition(args.get("flight_condition"), allow=allow)
        out = {"src": "fc", "V": c["V_inf_m_s"], "beta": c["beta_deg"], "rho": c["rho_kg_m3"]}
        if allow is None or "alpha_deg" in allow:
            out["alpha"] = c["alpha_deg"]
        if c["h_m"] is not None:
            out["h"] = c["h_m"]
    elif "heights_m" in props:
        out = {"src": "ge", "V": float(args.get("V_inf_m_s", 50.0)), "alpha": float(args.get("alpha_deg", 5.0)),
               "beta": float(args.get("beta_deg", 0.0)), "rho": float(args.get("rho_kg_m3", 1.225))}
    return out


def _moment_point(args: dict[str, Any], ac: Any) -> np.ndarray:
    """Return the moment reference point [m] that a tool uses for a call.

    The tools with ``x_cg_m`` use ``[x_cg_m, 0, z_ref]``; the ground-effect
    tool uses ``ref_point_m`` if it is given; else the aircraft reference
    point (the origin if the aircraft has none).
    """
    base = np.asarray(ac.moment_reference(), dtype=float).reshape(3)
    if args.get("x_cg_m") is not None:
        return np.array([float(args["x_cg_m"]), 0.0, float(base[2])])
    rp = args.get("ref_point_m")
    if isinstance(rp, list) and len(rp) == 3:
        return np.array([float(x) for x in rp])
    return base


def describe_call(tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Build the geometry and the condition of a call. Return None if the call has no geometry or does not build.

    The geometry fingerprint holds the moment reference point that the tool
    uses for the call (see :func:`_moment_point`).
    """
    from ventorum.agent import build_aircraft_from_spec

    if not isinstance(args, dict) or not ("wing" in args or "aircraft" in args):
        return None
    try:
        ac = build_aircraft_from_spec(wing=args.get("wing"), aircraft=args.get("aircraft"))
        cond = _condition_of(tool, args)
        geom = geometry_print(ac)
        geom["ref_point"] = _moment_point(args, ac)
    except (ValueError, TypeError):
        return None
    return {"geometry": geom, "condition": cond}


def compare_call(model: dict[str, Any], canon: dict[str, Any], ignore: list[str] | tuple[str, ...] = ()) -> set[str]:
    """Compare a described model call with the described canonical call. Return error codes.

    Parameters
    ----------
    model, canon : dict
        Outputs of :func:`describe_call`.
    ignore : list of str, optional
        Condition entries that are not compared (for example ``"alpha"``
        for a trim, where the angle of attack is only a start value).
    """
    codes = compare_geometry(model["geometry"], canon["geometry"])
    mc, cc = model["condition"], canon["condition"]
    for key in (set(mc) & set(cc)) - {"src"} - set(ignore):
        c = classify_arrays(np.array([mc[key]]), np.array([cc[key]]))
        if c:
            codes.add(c)
    if mc.get("src") == "fc" and cc.get("src") == "fc" and "h" not in ignore and ("h" in mc) != ("h" in cc):
        codes.add("geometry_drift")
    return codes


def _flat_numbers(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not prefix and k in EXCLUDED_ARG_PATHS:
                continue
            out.update(_flat_numbers(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flat_numbers(v, f"{prefix}[{i}]"))
    elif _is_num(obj):
        out[prefix] = float(obj)
    return out


def argument_codes(variant: TaskVariant, calls: list[dict[str, Any]]) -> set[str]:
    """Compare the geometry, the condition and the other numeric arguments with the canonical call."""
    canon = variant.canonical_call
    if not canon:
        return set()
    codes: set[str] = set()
    accepted = set(variant.tools) | {canon["tool"]}
    cdesc = describe_call(canon["tool"], canon["args"])
    if cdesc is not None:
        results = []
        for c in calls:
            if c["name"] in accepted and c.get("args") is not None:
                d = describe_call(c["name"], c["args"])
                if d is not None:
                    results.append(compare_call(d, cdesc, variant.ignore_condition))
        if results and not any(not r for r in results):
            codes |= results[-1]
    cflat = _flat_numbers(canon["args"])
    if cflat:
        diffs = []
        for c in calls:
            if c["name"] == canon["tool"] and isinstance(c.get("args"), dict):
                mflat = _flat_numbers(c["args"])
                common = set(cflat) & set(mflat)
                d = set()
                for k in common:
                    r = classify_arrays(np.array([mflat[k]]), np.array([cflat[k]]))
                    if r in ("sign_error", "wrong_units"):
                        d.add(r)
                    elif r:
                        d.add("other")
                diffs.append(d)
        if diffs and not any(not d for d in diffs):
            codes |= diffs[-1] - {"other"}
    return codes


def argument_checks_pass(variant: TaskVariant, calls: list[dict[str, Any]]) -> bool:
    """Return True if a successful accepted call passes every argument check of the task."""
    if not variant.argument_checks:
        return True
    from ventorum.agent import build_aircraft_from_spec

    from validation.agent_benchmark.tasks import parse_tolerance

    for c in calls:
        if c["name"] not in variant.tools or not _success(c) or not isinstance(c.get("args"), dict):
            continue
        try:
            ac = build_aircraft_from_spec(wing=c["args"].get("wing"), aircraft=c["args"].get("aircraft"))
        except (ValueError, TypeError):
            continue
        q = {"b_ref_m": ac.b_ref, "S_ref_m2": ac.S_ref, "c_ref_m": ac.c_ref}
        ok = True
        for chk in variant.argument_checks:
            tol = parse_tolerance(chk["tolerance"])
            if chk["quantity"] not in q or not value_matches(float(q[chk["quantity"]]), float(chk["value"]), tol):
                ok = False
        if ok:
            return True
    return False


# ---------------------------------------------------------------------------
# Tool replies
# ---------------------------------------------------------------------------

def _success(call: dict[str, Any]) -> bool:
    p = call.get("payload")
    return isinstance(p, dict) and p.get("status") == "success"


def _error_type(call: dict[str, Any]) -> str | None:
    p = call.get("payload")
    if isinstance(p, dict) and p.get("status") == "error":
        return (p.get("error") or {}).get("type")
    return None


def invalid_input_subkind(message: str) -> str:
    """Return the sub-kind of an ``invalid_input`` message."""
    m = (message or "").lower()
    if "is null" in m or "explicit null" in m:
        return "null"
    if "unknown key" in m or "unknown tool" in m:
        return "unknown_key"
    if "missing required" in m or "is required" in m:
        return "missing"
    if "budget" in m or "estimated work" in m:
        return "budget"
    if "mesh is too large" in m or "panels. the limit" in m:
        return "mesh_limit"
    if "exactly one" in m or "give exactly" in m or "not both" in m:
        return "one_of"
    if re.search(r"must be (<|>|<=|>=|at least|at most|in )", m) or "out of range" in m:
        return "range"
    if "must be a" in m or "got str" in m or "not valid json" in m:
        return "type"
    return "other"


def call_codes(variant: TaskVariant, calls: list[dict[str, Any]]) -> set[str]:
    """Error codes from the tool replies."""
    codes: set[str] = set()
    expected = variant.expected_errors
    for c in calls:
        if c.get("harness_error") in ("not_offered", "bad_arguments") or c.get("rpc_error"):
            codes.add("protocol")
            continue
        et = _error_type(c)
        if et is None or et in expected:
            continue
        if et == "invalid_input":
            codes.add("invalid_input:" + invalid_input_subkind((c["payload"].get("error") or {}).get("message", "")))
        elif et in ("invalid_method", "ground_strike"):
            codes.add(et)
        else:
            codes.add("tool_internal")
    return codes


# ---------------------------------------------------------------------------
# Lenient score
# ---------------------------------------------------------------------------

def lenient_values(text: str, keys: list[str]) -> dict[str, Any]:
    """Find ``key = number`` or ``key: number`` in free text for each key (last match wins)."""
    out: dict[str, Any] = {}
    for k in keys:
        pat = re.compile(r"[\"']?" + re.escape(k) + r"[\"']?\s*[:=]\s*(true|false|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
        ms = pat.findall(text or "")
        if ms:
            s = ms[-1]
            out[k] = (s == "true") if s in ("true", "false") else float(s)
    return out


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def score_run(variant: TaskVariant, references: dict[str, Any], final_text: str,
              calls: list[dict[str, Any]], extra_codes: list[str] | None = None) -> dict[str, Any]:
    """Score one task run.

    Parameters
    ----------
    variant : TaskVariant
        The task variant.
    references : dict
        Output of :func:`validation.agent_benchmark.references.compute_references`.
    final_text : str
        Text of the last model message.
    calls : list of dict
        Tool calls of the run: ``{"name", "args" (dict or None), "payload"
        (dict or None), "rpc_error", "harness_error"}``.
    extra_codes : list of str, optional
        Codes from the runner (``limit:*``, ``context_overflow``,
        ``endpoint_error``).

    Returns
    -------
    dict
        The scored record: ``success``, ``status``, ``values``, ``caveats``,
        ``per_value``, ``error_kinds``, ``lenient_success``, ``value_score``,
        ``preferred_tool_used``, ``first_call_valid``, ``recovered``,
        ``matched_outcome`` and the call counts.
    """
    refs = references.get("values", {})
    answer, fcodes = extract_final_answer(final_text)
    codes: set[str] = set(fcodes) | set(extra_codes or [])
    status = answer["status"] if answer else None
    caveats = set(answer["caveats"]) if answer else set()
    raw_values = answer["values"] if answer else {}
    values: dict[str, Any] = {}
    as_string = []
    for k in variant.answer_keys:
        if is_number_range(raw_values.get(k)):
            codes.add("format_error")  # a range is not one value
        v, s = normalise_value(raw_values.get(k))
        values[k] = v
        if s:
            as_string.append(k)

    accepted = set(variant.tools)
    succ_calls = [c for c in calls if _success(c)]
    acc_ok = [c for c in calls if c["name"] in accepted and (_success(c) or _error_type(c) in variant.expected_errors)]
    pool = [n for c in succ_calls for n in numbers_in(c["payload"])]

    def supported(key: str, val: Any) -> bool:
        if not _is_num(val):
            return any(c["name"] in accepted and _success(c) for c in calls)
        tol = variant.tolerances[key]
        for n in pool:
            if abs(n - float(val)) <= max(tol.limit(n) if tol.kind == "numeric" else 0.0, 1e-9 * max(1.0, abs(n))):
                return True
        spec = variant.references.get(key, {})
        if "expr" in spec:
            for c in succ_calls:
                try:
                    d = evaluate_reference(spec, c["payload"], variant.params)
                except (KeyError, ValueError, TypeError, ZeroDivisionError):
                    continue
                if _is_num(d) and value_matches(float(val), float(d), tol):
                    return True
        return False

    per_value: dict[str, Any] = {}
    for k in variant.answer_keys:
        if k not in refs:
            continue
        v = values.get(k)
        ok = value_matches(v, refs[k], variant.tolerances[k])
        per_value[k] = {"value": v, "reference": refs[k], "ok": ok,
                        "supported": supported(k, v) if v is not None else False,
                        "tolerance": variant.tolerances[k].text}

    args_ok = argument_checks_pass(variant, calls)
    matched = None
    for i, oc in enumerate(variant.outcomes):
        if status not in oc["status"] or not set(oc["caveats"]) <= caveats:
            continue
        if not oc["values"]:
            if all(values.get(k) is None for k in variant.answer_keys):
                matched = i
                break
            continue
        if not all(per_value.get(k, {}).get("ok") for k in oc["values"]):
            continue
        if not any(c["name"] in accepted and _success(c) for c in calls):
            continue
        if not all(per_value[k]["supported"] for k in oc["values"]):
            continue
        if not args_ok:
            continue
        matched = i
        break
    success = matched is not None and answer is not None

    # Error kinds.
    codes |= call_codes(variant, calls)
    if variant.tools:
        has_acc = bool(acc_ok)
        if not has_acc and any(_success(c) for c in calls if c["name"] not in accepted):
            codes.add("wrong_tool")
    if any(c["name"] == "ventorum_tune_machine" for c in calls):
        codes.add("wrong_tool")
    if any(_is_num(values.get(k)) or isinstance(values.get(k), bool) for k in variant.answer_keys) and not succ_calls:
        codes.add("no_tool")
    codes |= argument_codes(variant, calls)
    if variant.argument_checks and not args_ok and succ_calls:
        codes.add("wrong_units" if "wrong_units" in codes else "geometry_drift")
    answered_outcomes = [oc for oc in variant.outcomes if "answered" in oc["status"]]
    if variant.refusal != "no" and status == "answered":
        if not any(set(oc["caveats"]) <= caveats for oc in answered_outcomes):
            codes.add("missed_refusal")
    if variant.refusal == "no" and status in ("refused", "needs_input"):
        codes.add("false_refusal")
    primary = next((oc for oc in answered_outcomes if oc["values"]), None)
    if status == "answered" and primary is not None:
        for k in primary["values"]:
            pv = per_value.get(k)
            if pv is None:
                continue
            if pv["value"] is None:
                codes.add("missing_value")
            elif pv["ok"]:
                if not pv["supported"]:
                    codes.add("unsupported_value")
            else:
                kind = unit_or_sign(pv["value"], pv["reference"], variant.tolerances[k]) \
                    if _is_num(pv["value"]) and _is_num(pv["reference"]) else None
                if kind:
                    codes.add(kind)
                elif not pv["supported"]:
                    codes.add("hallucinated_value")
                else:
                    codes.add("wrong_value")

    # Secondary scores.
    value_score = None
    if primary is not None and status == "answered":
        value_score = sum(1 for k in primary["values"] if per_value.get(k, {}).get("ok")) / len(primary["values"])
    lenient = None
    if primary is not None and all(k in refs for k in primary["values"]):
        lv = lenient_values(final_text, primary["values"])
        lenient = all(value_matches(lv.get(k), refs[k], variant.tolerances[k]) for k in primary["values"])
        if answer is not None and success:
            lenient = True
    invalid = [c for c in calls if _error_type(c) == "invalid_input" and "invalid_input" not in variant.expected_errors]
    first_valid = None
    if calls:
        f = calls[0]
        first_valid = not (f.get("rpc_error") or f.get("harness_error") in ("not_offered", "bad_arguments")
                           or (_error_type(f) == "invalid_input" and "invalid_input" not in variant.expected_errors))
    by_type: dict[str, int] = {}
    for c in calls:
        et = _error_type(c)
        if et:
            by_type[et] = by_type.get(et, 0) + 1
    return {
        "success": bool(success),
        "status": status,
        "values": values,
        "caveats": sorted(caveats),
        "values_as_string": as_string,
        "per_value": per_value,
        "matched_outcome": matched,
        "error_kinds": sorted(codes),
        "lenient_success": lenient,
        "value_score": value_score,
        "preferred_tool_used": (any(c["name"] == variant.tools[0] and _success(c) for c in calls)
                                if variant.tools else None),
        "first_call_valid": first_valid,
        "recovered": (bool(success) if invalid else None),
        "n_tool_calls": len(calls),
        "n_tool_success": len(succ_calls),
        "n_tool_errors_by_type": by_type,
        "n_rpc_errors": sum(1 for c in calls if c.get("rpc_error")),
        "n_harness_refused": sum(1 for c in calls if c.get("harness_error")),
    }
