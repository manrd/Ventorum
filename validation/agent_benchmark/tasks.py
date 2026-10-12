# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Task set of the agent benchmark: load, check, make variants, build prompts.

A task file (for example ``tasks_v1.json``) holds task templates. A template
gives the prompt text, the answer keys, the accepted tools, the golden calls
(the canonical tool arguments), the paths of the reference values in the
tool payloads, the tolerances and the accepted outcomes. The file holds no
reference values: :mod:`.references` runs the golden calls at run time.

Template syntax:

* A string ``"=expr"`` in a call argument is an expression of the task
  parameters (``+ - * / **``, comparisons, ``degrees``, ``radians``,
  ``sqrt``, ``int``, ``float``, ``abs``, ``round``, ``min``, ``max``).
* A string ``"@G1"`` is the geometry ``G1`` of the ``geometries`` table.
* In the prompt, ``{name}`` is a parameter value or the words of a geometry;
  ``{name|cap}`` gives the same text with a capital first letter.

Seed 0 gives the public set (the ``value`` of each parameter). Another seed
draws each parameter that has a ``range`` from a uniform distribution and
rounds it to ``decimals`` (held-out variants).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import operator
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_TASK_FILE = PACKAGE_DIR / "tasks_v1.json"
TOOL_PREFIX = "ventorum_"

STATUSES = ("answered", "refused", "needs_input")
CAVEAT_CODES = (
    "out_of_envelope", "method_refused", "ground_strike", "missing_input", "ambiguous_units",
    "invalid_geometry", "out_of_scope", "no_stall_model", "no_profile_drag", "not_trimmed",
    "mesh_limit", "low_trust",
)
REFUSAL_FLAGS = ("no", "yes", "or_caveat")
EXPECT_VALUES = ("success", "invalid_input", "invalid_method", "ground_strike")
TASK_ID = re.compile(r"^AB-\d{2}$")


class TaskFileError(ValueError):
    """The task file is not valid."""


# ---------------------------------------------------------------------------
# Safe expressions
# ---------------------------------------------------------------------------

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow, ast.Mod: operator.mod}
_CMPOPS = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
           ast.Eq: operator.eq, ast.NotEq: operator.ne}
_FUNCS = {"degrees": math.degrees, "radians": math.radians, "sqrt": math.sqrt, "int": int, "float": float,
          "abs": abs, "round": round, "min": min, "max": max}


def safe_eval(expr: str, names: dict[str, Any]) -> Any:
    """Evaluate an arithmetic expression with the given names.

    Parameters
    ----------
    expr : str
        Expression text, for example ``"0.5*rho*v**2*s*cl"``.
    names : dict
        Values of the names in the expression.

    Returns
    -------
    float, int or bool
        The value of the expression.

    Raises
    ------
    ValueError
        If the expression has a construct that is not allowed, or a name
        that is not known.
    """
    tree = ast.parse(expr, mode="eval")

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            if node.id in ("True", "False"):
                return node.id == "True"
            raise ValueError(f"unknown name {node.id!r} in expression {expr!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -ev(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +ev(node.operand)
            if isinstance(node.op, ast.Not):
                return not ev(node.operand)
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            for op, comp in zip(node.ops, node.comparators):
                if type(op) not in _CMPOPS:
                    break
                right = ev(comp)
                if not _CMPOPS[type(op)](left, right):
                    return False
                left = right
            else:
                return True
        if isinstance(node, ast.BoolOp):
            vals = [ev(v) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
                and not node.keywords):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"construct not allowed in expression {expr!r}: {ast.dump(node)[:60]}")

    return ev(tree)


# ---------------------------------------------------------------------------
# JSON paths and tolerances
# ---------------------------------------------------------------------------

_PATH_TOKEN = re.compile(r"\.?([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]|\['([^']+)'\]|\[([A-Za-z_][A-Za-z0-9_]*)=([^\]]+)\]")


def parse_path(path: str) -> list[tuple[str, Any]]:
    """Split a payload path such as ``matrix.CL['delta:flap_per_rad']`` into steps.

    Parameters
    ----------
    path : str
        The path. Steps are ``.key``, ``[index]``, ``['key']`` and
        ``[field=value]`` (the first list item whose field equals the
        number or string *value*).

    Returns
    -------
    list of tuple
        The steps as ``("key", name)``, ``("index", i)`` or
        ``("select", (field, value))``.
    """
    steps: list[tuple[str, Any]] = []
    pos = 0
    while pos < len(path):
        m = _PATH_TOKEN.match(path, pos)
        if not m or m.end() == pos:
            raise ValueError(f"bad path {path!r} at position {pos}")
        if m.group(1) is not None:
            steps.append(("key", m.group(1)))
        elif m.group(2) is not None:
            steps.append(("index", int(m.group(2))))
        elif m.group(3) is not None:
            steps.append(("key", m.group(3)))
        else:
            raw = m.group(5).strip()
            try:
                val: Any = float(raw)
            except ValueError:
                val = raw.strip("'\"")
            steps.append(("select", (m.group(4), val)))
        pos = m.end()
    return steps


def get_path(obj: Any, path: str) -> Any:
    """Return the value at *path* in *obj*.

    Raises
    ------
    KeyError
        If a step of the path does not exist.
    """
    cur = obj
    for kind, arg in parse_path(path):
        if kind == "key":
            if not isinstance(cur, dict) or arg not in cur:
                raise KeyError(f"{path}: no key {arg!r}")
            cur = cur[arg]
        elif kind == "index":
            if not isinstance(cur, list) or arg >= len(cur):
                raise KeyError(f"{path}: no index {arg}")
            cur = cur[arg]
        else:
            fld, val = arg
            if not isinstance(cur, list):
                raise KeyError(f"{path}: not a list at [{fld}=...]")
            for item in cur:
                if isinstance(item, dict) and fld in item:
                    iv = item[fld]
                    if isinstance(val, float) and isinstance(iv, (int, float)) and not isinstance(iv, bool):
                        if abs(float(iv) - val) <= 1e-9 * max(1.0, abs(val)):
                            cur = item
                            break
                    elif iv == val:
                        cur = item
                        break
            else:
                raise KeyError(f"{path}: no item with {fld} = {val}")
    return cur


@dataclass(frozen=True)
class Tolerance:
    """A tolerance of an answer value.

    Attributes
    ----------
    kind : str
        ``"numeric"``, ``"exact"`` or ``"set"``.
    rel : float
        Relative limit (fraction of the reference), numeric kind only.
    abs : float
        Absolute limit, in the unit of the value, numeric kind only.
    text : str
        The tolerance as written in the task file.
    """

    kind: str
    rel: float = 0.0
    abs: float = 0.0
    text: str = ""

    def limit(self, ref: float) -> float:
        """Return the allowed absolute difference for the reference *ref*."""
        return max(self.rel * abs(ref), self.abs)


_TOL_PART = re.compile(r"^(r)([0-9.eE+-]+)%$|^(a)([0-9.eE+-]+)$")


def parse_tolerance(text: str) -> Tolerance:
    """Parse a tolerance such as ``r1%``, ``a0.03``, ``r2%|a5e-5``, ``exact`` or ``set``.

    ``r1%`` means abs(x - ref) <= 0.01 abs(ref); ``a0.03`` is an absolute
    limit; ``r2%|a5e-5`` is the larger of the two.
    """
    t = text.strip()
    if t in ("exact", "set"):
        return Tolerance(kind=t, text=t)
    rel = 0.0
    ab = 0.0
    for part in t.split("|"):
        m = _TOL_PART.match(part.strip())
        if not m:
            raise ValueError(f"bad tolerance {text!r}")
        if m.group(1):
            rel = float(m.group(2)) / 100.0
        else:
            ab = float(m.group(4))
    return Tolerance(kind="numeric", rel=rel, abs=ab, text=t)


# ---------------------------------------------------------------------------
# Task set
# ---------------------------------------------------------------------------

def full_tool_name(short: str) -> str:
    """Return the full MCP tool name of a short name (``wing_analysis``)."""
    return short if short.startswith(TOOL_PREFIX) else TOOL_PREFIX + short


def format_number(x: Any) -> str:
    """Format a parameter value for a prompt (no trailing zeros)."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x.is_integer() and abs(x) < 1e15:
            return str(int(x))
        return format(x, ".10g")
    return str(x)


@dataclass
class TaskVariant:
    """One task with its parameters resolved (the input of a benchmark run).

    Attributes
    ----------
    id : str
        Task id (``AB-NN``).
    seed : int
        Variant seed; 0 is the public set.
    level, category, refusal : int, str, str
        Task level (1 to 9), category text and refusal flag.
    prompt : str
        Full user prompt, with the ``Answer keys:`` line.
    answer_keys : list of str
        Keys of the final answer.
    tools : list of str
        Accepted tools (full names), the preferred tool first.
    focused_tools : list of str
        Task tools of the focused condition (full names).
    calls : dict
        Golden calls by name: ``{"tool", "args", "expect", ...}``.
    canonical : str or None
        Name of the canonical call, or None if no call is needed.
    references : dict
        Reference specs by answer key.
    tolerances : dict
        Parsed tolerances by answer key.
    outcomes : list of dict
        Accepted outcomes: ``{"status": [...], "caveats": [...], "values": [...]}``.
    argument_checks : list of dict
        Checks of the built geometry: ``{"quantity", "value", "tolerance"}``.
    params : dict
        Parameter values of this variant.
    """

    id: str
    seed: int
    level: int
    category: str
    refusal: str
    prompt: str
    answer_keys: list[str]
    tools: list[str]
    focused_tools: list[str]
    calls: dict[str, dict[str, Any]]
    canonical: str | None
    references: dict[str, dict[str, Any]]
    tolerances: dict[str, Tolerance]
    outcomes: list[dict[str, Any]]
    argument_checks: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_call(self) -> dict[str, Any] | None:
        """The canonical golden call, or None."""
        return self.calls.get(self.canonical) if self.canonical else None

    @property
    def expected_errors(self) -> set[str]:
        """Error types that the golden calls of this task expect."""
        return {c["expect"] for c in self.calls.values() if c.get("expect") != "success"}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict of the variant."""
        d = dict(self.__dict__)
        d["tolerances"] = {k: v.text for k, v in self.tolerances.items()}
        return d


class TaskSet:
    """A loaded and checked task file.

    Parameters
    ----------
    path : str or Path, optional
        Task file. The default is ``tasks_v1.json`` next to this module.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_TASK_FILE
        raw = self.path.read_bytes()
        self.file_hash = hashlib.sha256(raw).hexdigest()
        self.data = json.loads(raw.decode("utf-8"))
        problems = validate_task_data(self.data)
        if problems:
            raise TaskFileError(f"{self.path.name}: " + "; ".join(problems[:20]))
        self.version = self.data["task_set"]
        self.system_prompt: str = self.data["system_prompt"]
        self.system_prompt_hash = hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()
        self.canary: str = self.data["canary"]
        self.tasks: list[dict[str, Any]] = self.data["tasks"]
        self._by_id = {t["id"]: t for t in self.tasks}

    @property
    def ids(self) -> list[str]:
        """Task ids in file order."""
        return [t["id"] for t in self.tasks]

    @property
    def track_b_subset(self) -> list[str]:
        """Task ids of the client track (track B)."""
        return list(self.data.get("track_b_subset", []))

    def task(self, task_id: str) -> dict[str, Any]:
        """Return the raw template of *task_id*."""
        if task_id not in self._by_id:
            raise KeyError(f"unknown task id {task_id!r}")
        return self._by_id[task_id]

    def select(self, spec: str | list[str] | None) -> list[str]:
        """Return task ids from a selection.

        Parameters
        ----------
        spec : str, list of str or None
            None or ``"all"`` gives all tasks; ``"track_b"`` gives the
            client-track subset; otherwise a comma-separated list of ids or
            ranges (``AB-01,AB-05:AB-09``), or ``level:N``.
        """
        if spec is None or spec == "all":
            return self.ids
        if spec == "track_b":
            return self.track_b_subset
        items = spec if isinstance(spec, list) else [s.strip() for s in spec.split(",") if s.strip()]
        out: list[str] = []
        for item in items:
            if item.startswith("level:"):
                lev = int(item.split(":", 1)[1])
                out.extend(t["id"] for t in self.tasks if t["level"] == lev)
            elif ":" in item:
                a, b = item.split(":", 1)
                ids = self.ids
                out.extend(ids[ids.index(a):ids.index(b) + 1])
            else:
                self.task(item)
                out.append(item)
        seen: set[str] = set()
        return [i for i in out if not (i in seen or seen.add(i))]

    # -- parameters ----------------------------------------------------------
    def _param_specs(self, task: dict[str, Any]) -> dict[str, dict[str, Any]]:
        specs = dict(self.data.get("parameters", {}))
        specs.update(task.get("parameters", {}))
        return specs

    def _used_names(self, task: dict[str, Any]) -> set[str]:
        names: set[str] = set()
        geoms = self.data.get("geometries", {})

        def scan(obj: Any) -> None:
            if isinstance(obj, dict):
                for v in obj.values():
                    scan(v)
            elif isinstance(obj, list):
                for v in obj:
                    scan(v)
            elif isinstance(obj, str):
                if obj.startswith("@") and obj[1:] in geoms:
                    scan(geoms[obj[1:]]["spec"])
                elif obj.startswith("="):
                    names.update(n.id for n in ast.walk(ast.parse(obj[1:], mode="eval")) if isinstance(n, ast.Name))

        for m in re.finditer(r"\{([A-Za-z_][A-Za-z0-9_]*)(?:\|cap)?\}", task["prompt"]):
            n = m.group(1)
            if n in geoms:
                scan(geoms[n]["spec"])
                for m2 in re.finditer(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", geoms[n]["words"]):
                    names.add(m2.group(1))
            else:
                names.add(n)
        scan(task.get("calls", {}))
        scan(task.get("argument_checks", []))
        return names - set(_FUNCS)

    def parameters(self, task_id: str, seed: int = 0) -> dict[str, Any]:
        """Return the parameter values of a task variant.

        Parameters
        ----------
        task_id : str
            Task id.
        seed : int, optional
            Variant seed. 0 (default) gives the public values.
        """
        task = self.task(task_id)
        specs = self._param_specs(task)
        out: dict[str, Any] = {}
        for name in sorted(self._used_names(task)):
            if name not in specs:
                raise TaskFileError(f"{task_id}: parameter {name!r} has no definition")
            spec = specs[name]
            val = spec["value"]
            if seed != 0 and "range" in spec:
                lo, hi = spec["range"]
                rng = random.Random(f"{self.version}:{seed}:{task_id}:{name}")
                val = rng.uniform(float(lo), float(hi))
                val = round(val, int(spec.get("decimals", 3)))
                val = float(val)
            out[name] = val
        return out

    # -- resolution ------------------------------------------------------------
    def _resolve(self, obj: Any, params: dict[str, Any]) -> Any:
        geoms = self.data.get("geometries", {})
        if isinstance(obj, dict):
            return {k: self._resolve(v, params) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve(v, params) for v in obj]
        if isinstance(obj, str):
            if obj.startswith("@") and obj[1:] in geoms:
                return self._resolve(copy.deepcopy(geoms[obj[1:]]["spec"]), params)
            if obj.startswith("="):
                val = safe_eval(obj[1:], params)
                if isinstance(val, float) and not isinstance(val, bool):
                    val = float(round(val, 12))
                return val
        return obj

    def render_prompt(self, task: dict[str, Any], params: dict[str, Any]) -> str:
        """Return the prompt text of a task (without the answer-key line)."""
        geoms = self.data.get("geometries", {})

        def fill(text: str) -> str:
            def rep(m: re.Match) -> str:
                name, cap = m.group(1), m.group(2)
                if name in geoms:
                    s = fill(geoms[name]["words"])
                elif name in params:
                    s = format_number(params[name])
                else:
                    raise TaskFileError(f"{task['id']}: unknown placeholder {{{name}}}")
                return s[:1].upper() + s[1:] if cap else s

            return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)(\|cap)?\}", rep, text)

        return fill(task["prompt"])

    def variant(self, task_id: str, seed: int = 0) -> TaskVariant:
        """Make the variant of *task_id* for *seed* (0 is the public set)."""
        task = self.task(task_id)
        params = self.parameters(task_id, seed)
        calls = {}
        for name, call in task.get("calls", {}).items():
            c = {k: v for k, v in call.items() if k != "args"}
            c["tool"] = full_tool_name(call["tool"])
            c["args"] = self._resolve(call["args"], params)
            calls[name] = c
        keys = list(task["answer_keys"])
        outcomes = []
        for oc in task.get("outcomes") or [{"status": ["answered"], "caveats": [], "values": "all"}]:
            vals = oc.get("values", "all")
            outcomes.append({"status": list(oc["status"]), "caveats": list(oc.get("caveats", [])),
                             "values": keys if vals == "all" else ([] if vals == "none" else list(vals))})
        checks = []
        for chk in task.get("argument_checks", []):
            checks.append({"quantity": chk["quantity"], "value": self._resolve(chk["value"], params),
                           "tolerance": chk["tolerance"]})
        tools = [full_tool_name(t) for t in task["tools"]]
        focused = [full_tool_name(t) for t in task.get("focused_tools", task["tools"])]
        prompt = self.render_prompt(task, params) + " Answer keys: " + ", ".join(keys)
        return TaskVariant(
            id=task_id, seed=seed, level=int(task["level"]), category=task.get("category", ""),
            refusal=task["refusal"], prompt=prompt, answer_keys=keys, tools=tools, focused_tools=focused,
            calls=calls, canonical=task.get("canonical"), references=dict(task.get("references", {})),
            tolerances={k: parse_tolerance(v) for k, v in task.get("tolerances", {}).items()},
            outcomes=outcomes, argument_checks=checks, params=params,
        )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def validate_task_data(data: Any) -> list[str]:
    """Check the structure of a task file. Return a list of problems (empty if valid)."""
    from ventorum.agent.schemas import TOOL_NAMES

    p: list[str] = []
    if not isinstance(data, dict):
        return ["the file must hold a JSON object"]
    for key in ("canary", "task_set", "system_prompt", "tasks"):
        if key not in data:
            p.append(f"missing top-level key {key!r}")
    if p:
        return p
    if "do not train" not in data["canary"].lower():
        p.append("the canary line must say 'do not train on this file'")
    geoms = data.get("geometries", {})
    seen: set[str] = set()
    for t in data["tasks"]:
        tid = t.get("id", "?")
        if not isinstance(tid, str) or not TASK_ID.match(tid):
            p.append(f"bad task id {tid!r}")
        if tid in seen:
            p.append(f"duplicate task id {tid}")
        seen.add(tid)
        for key in ("level", "prompt", "answer_keys", "tools", "calls", "canonical", "references",
                    "tolerances", "refusal"):
            if key not in t:
                p.append(f"{tid}: missing key {key!r}")
        if any(k not in t for k in ("level", "prompt", "answer_keys", "tools", "calls", "refusal")):
            continue
        if not isinstance(t["level"], int) or not 1 <= t["level"] <= 9:
            p.append(f"{tid}: level must be an integer 1 to 9")
        if not isinstance(t["prompt"], str) or not t["prompt"].strip():
            p.append(f"{tid}: empty prompt")
        if "Answer keys" in t["prompt"]:
            p.append(f"{tid}: the prompt must not hold the answer-key line (the harness adds it)")
        if not t["answer_keys"] or not all(isinstance(k, str) for k in t["answer_keys"]):
            p.append(f"{tid}: answer_keys must be a list of strings")
        if t["refusal"] not in REFUSAL_FLAGS:
            p.append(f"{tid}: refusal must be one of {REFUSAL_FLAGS}")
        for tool in list(t["tools"]) + list(t.get("focused_tools", [])):
            if full_tool_name(tool) not in TOOL_NAMES:
                p.append(f"{tid}: unknown tool {tool!r}")
        for name, call in t["calls"].items():
            if full_tool_name(call.get("tool", "")) not in TOOL_NAMES:
                p.append(f"{tid}: call {name}: unknown tool {call.get('tool')!r}")
            if call.get("expect") not in EXPECT_VALUES:
                p.append(f"{tid}: call {name}: expect must be one of {EXPECT_VALUES}")
            for s in _strings(call.get("args")):
                if s.startswith("@") and s[1:] not in geoms:
                    p.append(f"{tid}: unknown geometry {s}")
        canon = t.get("canonical")
        if canon is not None and canon not in t["calls"]:
            p.append(f"{tid}: canonical call {canon!r} is not in calls")
        if canon is None and t["tools"]:
            p.append(f"{tid}: a task with accepted tools needs a canonical call")
        outcomes = t.get("outcomes") or [{"status": ["answered"], "caveats": [], "values": "all"}]
        need: set[str] = set()
        for oc in outcomes:
            if not oc.get("status") or any(s not in STATUSES for s in oc["status"]):
                p.append(f"{tid}: bad outcome status {oc.get('status')}")
            for c in oc.get("caveats", []):
                if c not in CAVEAT_CODES:
                    p.append(f"{tid}: unknown caveat code {c!r}")
            vals = oc.get("values", "all")
            need.update(t["answer_keys"] if vals == "all" else ([] if vals == "none" else vals))
        for k in need:
            ref = t.get("references", {}).get(k)
            if not ref or not ("path" in ref or "expr" in ref):
                p.append(f"{tid}: answer key {k} has no reference")
            elif ref.get("call", canon) not in t["calls"]:
                p.append(f"{tid}: reference of {k} names an unknown call")
            if k not in t.get("tolerances", {}):
                p.append(f"{tid}: answer key {k} has no tolerance")
        for tol in t.get("tolerances", {}).values():
            try:
                parse_tolerance(tol)
            except ValueError as exc:
                p.append(f"{tid}: {exc}")
        for ref in t.get("references", {}).values():
            for path in [ref.get("path")] + list(ref.get("vars", {}).values()):
                if path:
                    try:
                        parse_path(path)
                    except ValueError as exc:
                        p.append(f"{tid}: {exc}")
    return p


def _strings(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return [obj] if isinstance(obj, str) else []


def check_call_arguments(tool: str, args: dict[str, Any], *, deep: bool = True) -> list[str]:
    """Check the arguments of a golden call with the dispatcher rules.

    Parameters
    ----------
    tool : str
        Full tool name.
    args : dict
        Tool arguments.
    deep : bool, optional
        If True (default), also build the geometry and the flight
        condition with the strict parsers of the agent layer.

    Returns
    -------
    list of str
        Problems found (empty if the arguments are valid).
    """
    from ventorum.agent.schemas import (
        AGENT_TOOL_DEFINITIONS,
        build_aircraft_from_spec,
        check_keys,
        parse_condition,
        parse_settings,
    )

    params = {t["name"]: t["parameters"] for t in AGENT_TOOL_DEFINITIONS}.get(tool)
    if params is None:
        return [f"unknown tool {tool}"]
    out: list[str] = []
    try:
        check_keys(args, params["properties"], f"{tool} arguments")
    except ValueError as exc:
        out.append(str(exc))
    missing = [k for k in params.get("required", []) if k not in args]
    if missing:
        out.append(f"missing required argument(s): {', '.join(missing)}")
    if deep and not out:
        try:
            if "wing" in args or "aircraft" in args:
                build_aircraft_from_spec(wing=args.get("wing"), aircraft=args.get("aircraft"))
            if "flight_condition" in args:
                allow = None if "alpha_deg" in params["properties"].get("flight_condition", {}).get(
                    "properties", {"alpha_deg": 1}) else ("V_inf_m_s", "beta_deg", "rho_kg_m3", "altitude_m", "h_m")
                parse_condition(args["flight_condition"], allow=allow)
            if "settings" in args:
                parse_settings(args["settings"])
            for cand in args.get("candidates", []):
                build_aircraft_from_spec(wing=cand.get("wing"), aircraft=cand.get("aircraft"))
        except ValueError as exc:
            out.append(str(exc))
    return out
