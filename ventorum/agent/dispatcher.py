# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Dispatcher for agent tool calls.

:func:`call_tool` takes a tool name and its arguments (a dict or a JSON
string), checks the argument keys against the tool schema, runs the tool
and returns a JSON-safe payload. It never raises.

Audit log: only if the environment variable ``VENTORUM_AGENT_AUDIT_LOG`` is
set to a file path, one JSON line per call is appended to that file.
Otherwise nothing is written.
"""

from __future__ import annotations

import datetime
import json
import os
import time
from typing import Any
from collections.abc import Callable

from ventorum.agent.response import error_payload
from ventorum.agent.schemas import AGENT_TOOL_DEFINITIONS, TOOL_NAMES, check_keys
from ventorum.agent.tools import (
    batch_evaluate,
    ground_effect,
    machine_capabilities,
    mesh_convergence,
    polar_sweep,
    stability_derivatives,
    tune_machine,
    wing_analysis,
)
from ventorum.utils.jsonsafe import json_safe

AUDIT_ENV_VAR = "VENTORUM_AGENT_AUDIT_LOG"
PREFIX = "ventorum_"

TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "ventorum_wing_analysis": wing_analysis,
    "ventorum_polar_sweep": polar_sweep,
    "ventorum_ground_effect": ground_effect,
    "ventorum_stability_derivatives": stability_derivatives,
    "ventorum_batch_evaluate": batch_evaluate,
    "ventorum_mesh_convergence": mesh_convergence,
    "ventorum_machine_capabilities": machine_capabilities,
    "ventorum_tune_machine": tune_machine,
}
_PARAMS = {t["name"]: t["parameters"] for t in AGENT_TOOL_DEFINITIONS}
assert set(TOOL_FUNCTIONS) == set(TOOL_NAMES)


def resolve_tool_name(name: Any) -> str | None:
    """Full tool name for *name* (with or without the ``ventorum_`` prefix), or None."""
    if not isinstance(name, str):
        return None
    n = name.strip()
    if n in TOOL_FUNCTIONS:
        return n
    if PREFIX + n in TOOL_FUNCTIONS:
        return PREFIX + n
    return None


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _write_audit(entry: dict[str, Any]) -> None:
    path = os.environ.get(AUDIT_ENV_VAR)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(entry), allow_nan=False, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass  # The audit log must not break a tool call.


def _run(tool_name: Any, arguments: Any) -> tuple[str | None, dict[str, Any]]:
    name = resolve_tool_name(tool_name)
    if name is None:
        return None, error_payload(
            "invalid_input", f"Unknown tool {tool_name!r}. Available tools: {', '.join(TOOL_NAMES)}.")

    if arguments is None:
        args: Any = {}
    elif isinstance(arguments, str):
        try:
            args = json.loads(arguments, parse_constant=_no_constant) if arguments.strip() else {}
        except ValueError as err:  # includes json.JSONDecodeError
            return name, error_payload("invalid_input", f"The arguments are not valid JSON: {err}.")
    else:
        args = arguments
    if not isinstance(args, dict):
        return name, error_payload("invalid_input",
                                   f"The arguments must be a JSON object, got {type(args).__name__}.")

    params = _PARAMS[name]
    try:
        args = check_keys(args, params["properties"], f"{name} arguments")
    except ValueError as exc:
        return name, error_payload(exc)
    missing = [k for k in params.get("required", []) if k not in args]
    if missing:
        return name, error_payload("invalid_input", f"{name}: missing required argument(s): {', '.join(missing)}.")

    return name, TOOL_FUNCTIONS[name](**args)


def call_tool(tool_name: str, arguments: dict[str, Any] | str | None = None) -> dict[str, Any]:
    """Run an agent tool and return its payload. This function never raises.

    Parameters
    ----------
    tool_name : str
        Tool name, for example ``"ventorum_wing_analysis"`` (the prefix
        ``ventorum_`` is optional).
    arguments : dict, JSON string or None
        Tool arguments. They must agree with the tool schema; unknown keys
        are refused.
    """
    t0 = time.perf_counter()
    name: str | None = None
    try:
        name, result = _run(tool_name, arguments)
        result = json_safe(result)
        json.dumps(result, allow_nan=False)  # Proof that the payload is strict JSON.
    except Exception as exc:  # noqa: BLE001 - the dispatcher must not raise
        result = error_payload("internal", f"{type(exc).__name__}: {exc}")
    if name is not None:
        result["tool"] = name
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    result["execution_time_ms"] = elapsed_ms
    _write_audit({
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "tool": name if name is not None else str(tool_name),
        "status": result.get("status"),
        "error_type": (result.get("error") or {}).get("type"),
        "duration_ms": elapsed_ms,
        "arguments": arguments if isinstance(arguments, (dict, str)) else None,
    })
    return result
