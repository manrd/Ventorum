# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Reference values of the agent benchmark, computed at run time.

The reference of each answer key comes from a golden call: the canonical
tool arguments of the task through :func:`ventorum.agent.call_tool` (the
same code path and defaults that the model sees), or a formula of values of
that payload (for example the lift force from CL, the density, the speed
and the reference area).

:func:`selftest` compares six golden calls with the library API
(``vt.analyze``, ``vt.static_derivatives``, ``vt.trim``,
``vt.derivative_matrix``, ``analyze_ground_effect`` and
``vt.numerical_error_bars``) to find drift in the agent layer.

No reference value is stored in the repository.
"""

from __future__ import annotations

import fnmatch
import math
import warnings
from typing import Any
from collections.abc import Callable

from validation.agent_benchmark.tasks import TaskSet, TaskVariant, get_path, safe_eval

CallFn = Callable[[str, dict[str, Any]], dict[str, Any]]


def in_process_call(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run a tool in this process with :func:`ventorum.agent.call_tool`."""
    from ventorum.agent import call_tool

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call_tool(tool, args)


def mcp_call(client: Any, timeout_s: float = 600.0) -> CallFn:
    """Return a call function that runs the tools through an MCP client."""

    def call(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        out = client.call_tool(tool, args, timeout_s=timeout_s)
        if out["payload"] is None:
            return {"status": "error", "error": {"type": "protocol", "message": out["text"]}}
        return out["payload"]

    return call


def _plain(v: Any) -> Any:
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return v


def evaluate_reference(spec: dict[str, Any], payload: dict[str, Any], params: dict[str, Any]) -> Any:
    """Return the value of one reference spec in one payload.

    Parameters
    ----------
    spec : dict
        ``{"path": ...}`` or ``{"expr": ..., "vars": {name: path}}``.
    payload : dict
        A tool payload.
    params : dict
        Task parameters (also names in the expression).

    Raises
    ------
    KeyError
        A path is not in the payload.
    """
    if "path" in spec:
        return _plain(get_path(payload, spec["path"]))
    names = dict(params)
    for name, path in spec.get("vars", {}).items():
        names[name] = _plain(get_path(payload, path))
    return _plain(safe_eval(spec["expr"], names))


def compute_references(variant: TaskVariant, call: CallFn | None = None) -> dict[str, Any]:
    """Run the golden calls of a task variant and extract the reference values.

    Parameters
    ----------
    variant : TaskVariant
        The task variant.
    call : callable, optional
        ``call(tool, args) -> payload``. The default runs the tool in this
        process.

    Returns
    -------
    dict
        ``{"task_id", "seed", "calls", "values", "valid", "problems"}``.
        ``calls`` gives per golden call the status, the error type and the
        execution time; ``values`` gives the reference of each answer key
        that has one; ``valid`` is False if a golden call does not give the
        expected result.
    """
    call = call or in_process_call
    problems: list[str] = []
    calls: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name, c in variant.calls.items():
        payload = call(c["tool"], c["args"])
        payloads[name] = payload
        status = payload.get("status")
        etype = (payload.get("error") or {}).get("type")
        got = "success" if status == "success" else etype
        info = {"tool": c["tool"], "status": status, "error_type": etype,
                "execution_time_ms": payload.get("execution_time_ms")}
        if got != c["expect"]:
            problems.append(f"call {name}: expected {c['expect']}, got {got}"
                            + (f" ({(payload.get('error') or {}).get('message')})" if etype else ""))
        if c.get("expect_trust"):
            rating = (payload.get("trust") or {}).get("rating")
            info["trust_rating"] = rating
            if rating != c["expect_trust"]:
                problems.append(f"call {name}: expected trust {c['expect_trust']}, got {rating}")
        for path, pattern in (c.get("expect_paths") or {}).items():
            try:
                val = get_path(payload, path)
            except KeyError:
                val = None
            if not (isinstance(val, str) and fnmatch.fnmatch(val, pattern)):
                problems.append(f"call {name}: {path} = {val!r} does not match {pattern!r}")
        settings = payload.get("settings_used") or {}
        info["device"] = settings.get("device", payload.get("device"))
        info["precision"] = settings.get("precision", payload.get("precision"))
        calls[name] = info
    values: dict[str, Any] = {}
    for key, spec in variant.references.items():
        cname = spec.get("call", variant.canonical)
        payload = payloads.get(cname)
        if payload is None or payload.get("status") != "success":
            continue
        try:
            val = evaluate_reference(spec, payload, variant.params)
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            problems.append(f"reference {key}: {exc}")
            continue
        if isinstance(val, float) and not math.isfinite(val):
            problems.append(f"reference {key} is not finite")
        values[key] = val
    needed = {k for oc in variant.outcomes for k in oc["values"]}
    for k in sorted(needed - set(values)):
        problems.append(f"no reference value for {k}")
    return {"task_id": variant.id, "seed": variant.seed, "calls": calls, "values": values,
            "valid": not problems, "problems": problems}


def compute_all(task_set: TaskSet, task_ids: list[str], seed: int = 0,
                call: CallFn | None = None) -> dict[str, dict[str, Any]]:
    """Compute the references of several tasks. Return them by task id."""
    return {tid: compute_references(task_set.variant(tid, seed), call) for tid in task_ids}


def regression_check(current: dict[str, dict[str, Any]], previous: dict[str, dict[str, Any]],
                     rel: float = 1e-4) -> list[str]:
    """Compare two reference sets of the same Ventorum commit.

    Parameters
    ----------
    current, previous : dict
        Reference sets by task id (from :func:`compute_all`).
    rel : float, optional
        Relative limit. A difference above it is a change of a result.

    Returns
    -------
    list of str
        The changes (empty if none).
    """
    out: list[str] = []
    for tid, cur in current.items():
        prev = previous.get(tid)
        if prev is None or prev.get("seed") != cur.get("seed"):
            continue
        for key, a in cur.get("values", {}).items():
            if key not in prev.get("values", {}):
                continue
            b = prev["values"][key]
            if isinstance(a, float) and isinstance(b, float):
                if abs(a - b) > rel * max(abs(a), abs(b), 1e-12):
                    out.append(f"{tid} {key}: {b!r} -> {a!r}")
            elif isinstance(a, list) and isinstance(b, list):
                if sorted(map(str, a)) != sorted(map(str, b)):
                    out.append(f"{tid} {key}: {b!r} -> {a!r}")
            elif a != b:
                out.append(f"{tid} {key}: {b!r} -> {a!r}")
    return out


# ---------------------------------------------------------------------------
# Self-test against the library API
# ---------------------------------------------------------------------------

def selftest(task_set: TaskSet | None = None, rel: float = 0.005) -> list[dict[str, Any]]:
    """Compare six golden calls of the public set with the library API.

    Parameters
    ----------
    task_set : TaskSet, optional
        The task set (default ``tasks_v1.json``).
    rel : float, optional
        Relative limit of the agreement (default 0.5 %).

    Returns
    -------
    list of dict
        One record per check: ``{"name", "agent", "library", "rel_diff", "ok"}``.
    """
    import numpy as np

    import ventorum as vt
    from ventorum.agent import build_aircraft_from_spec, build_flight_condition_from_spec
    from ventorum.agent.schemas import parse_settings
    from ventorum.ground_effect import analyze_ground_effect

    ts = task_set or TaskSet()
    checks: list[dict[str, Any]] = []

    def add(name: str, agent: float, lib: float) -> None:
        d = abs(agent - lib) / max(abs(lib), 1e-12)
        checks.append({"name": name, "agent": agent, "library": lib, "rel_diff": d, "ok": bool(d <= rel)})

    def golden(tid: str) -> tuple[TaskVariant, dict[str, Any]]:
        v = ts.variant(tid, 0)
        c = v.canonical_call
        return v, in_process_call(c["tool"], c["args"])

    def geom(args: dict[str, Any]):
        return build_aircraft_from_spec(wing=args.get("wing"), aircraft=args.get("aircraft"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 1. Single wing, vt.analyze.
        v, p = golden("AB-01")
        a = v.canonical_call["args"]
        res = vt.analyze(geom(a), build_flight_condition_from_spec(a["flight_condition"]))
        add("AB-01 CL (vt.analyze)", p["metrics"]["CL"], float(res.totals.CL))
        # 2. Static derivatives of an aircraft.
        v, p = golden("AB-19")
        a = v.canonical_call["args"]
        sd = vt.static_derivatives(geom(a), build_flight_condition_from_spec(a["flight_condition"]))
        add("AB-19 Cm_alpha (vt.static_derivatives)", p["stability_derivatives"]["Cm_alpha_per_rad"],
            float(sd.derivatives["Cm_alpha"]))
        add("AB-19 x_np (vt.static_derivatives)", p["stability_derivatives"]["neutral_point_x_m"],
            float(sd.neutral_point_x))
        # 3. Trim.
        v, p = golden("AB-24")
        a = v.canonical_call["args"]
        tr = vt.trim(geom(a), build_flight_condition_from_spec(a["flight_condition"]),
                     CL_target=a["CL_target"], pitch_control=a["pitch_control"])
        add("AB-24 alpha (vt.trim)", p["alpha_deg"], float(tr.alpha_deg))
        add("AB-24 elevator (vt.trim)", p["deflections_deg"]["elevator"], float(tr.deflections_deg["elevator"]))
        # 4. Derivative matrix (pitch-rate column).
        v, p = golden("AB-31")
        a = v.canonical_call["args"]
        ac = geom(a)
        ac.ref_point = np.array([a["x_cg_m"], 0.0, 0.0])
        dm = vt.derivative_matrix(ac, build_flight_condition_from_spec(a["flight_condition"]))
        rows, cols = list(dm.rows), list(dm.columns)
        add("AB-31 CL_q (vt.derivative_matrix)", p["matrix"]["CL"]["q_hat"],
            float(dm.matrix[rows.index("CL")][cols.index("q_hat")]))
        add("AB-31 Cm_q (vt.derivative_matrix)", p["matrix"]["Cm"]["q_hat"],
            float(dm.matrix[rows.index("Cm")][cols.index("q_hat")]))
        # 5. Ground effect.
        v, p = golden("AB-14")
        a = v.canonical_call["args"]
        h0 = a["heights_m"][0]
        ge = analyze_ground_effect(geom(a), h0, a["alpha_deg"], V_inf=a["V_inf_m_s"])
        row = next(r for r in p["rows"] if abs(r["h_m"] - h0) < 1e-12)
        add("AB-14 CL at lowest height (analyze_ground_effect)", row["CL"], float(ge.CL))
        # 6. Numerical error bars.
        v, p = golden("AB-28")
        a = v.canonical_call["args"]
        eb = vt.numerical_error_bars(geom(a), build_flight_condition_from_spec(a["flight_condition"]),
                                     parse_settings(a["settings"]))
        add("AB-28 CL value (vt.numerical_error_bars)", p["error_bars"]["bars"]["CL"]["value"],
            float(eb.bars["CL"].value))
    return checks
