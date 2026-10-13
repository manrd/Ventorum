# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Command line of the agent benchmark (track A: own MCP client loop).

    python -m validation.agent_benchmark.runner run --config CONFIG --model LABEL
        --condition focused --tasks AB-01,AB-19 --repeats 3 --out DIR
    python -m validation.agent_benchmark.runner report RUN_DIR
    python -m validation.agent_benchmark.runner references --out DIR [--tasks ...] [--seed N]
    python -m validation.agent_benchmark.runner selftest

``run`` starts a new Ventorum MCP server for each task run, sends one hidden
warm-up call, gives the task prompt and the tool list to the model, runs the
tool calls of the model through ``tools/call`` until the model gives a final
answer or a limit is reached, and scores the run. It writes one folder per
run (``manifest.json``, ``references.json``, ``events.jsonl``,
``results.jsonl``, ``audit/``) under ``--out``, which must be outside the
repository. A second ``run`` with the same ``--run-id`` skips the finished
task runs (resume). ``report`` writes ``summary.md`` and ``summary.csv``
from ``results.jsonl``.

The private config (a JSON file outside the repository) gives the models
(adapter, endpoint, model string, name of the key variable, context length,
prices) and optional limits. No key, host name or user name is written to
any output file.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import getpass
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable

from validation.agent_benchmark import HARNESS_VERSION, LOG_SCHEMA_VERSION
from validation.agent_benchmark.adapters import (
    BaseAdapter,
    ConfigError,
    EndpointError,
    ToolCallParseError,
    make_adapter,
    parse_arguments,
)
from validation.agent_benchmark.mcp_client import (
    McpServerExit,
    McpTimeout,
    other_mcp_servers,
    sandbox_config_dir,
    start_session,
)
from validation.agent_benchmark.references import compute_all, regression_check, selftest
from validation.agent_benchmark.scoring import score_run
from validation.agent_benchmark.tasks import TaskSet, TaskVariant, check_call_arguments

REPO_ROOT = Path(__file__).resolve().parents[2]
BLACKLIST = ("nemotron3ultra",)
CONDITIONS = ("full", "focused")
DISTRACTORS = ("ventorum_machine_capabilities", "ventorum_undeformed_nodes", "ventorum_mesh_convergence",
               "ventorum_error_bars")
#: Hidden warm-up call: a small wing that no task uses.
WARMUP_CALL = ("ventorum_wing_analysis", {"wing": {"span_m": 7.0, "chord_m": 0.7},
                                          "flight_condition": {"V_inf_m_s": 25.0, "alpha_deg": 3.0}})
LIMIT_NOTICE = ("The tool-call limit of this task is reached. Do not call a tool again. "
                "Give your final answer now, with the <final_answer> block.")
CONTEXT_FILL = 0.98


class RunRefused(RuntimeError):
    """The run cannot start (bad config, blacklisted model, output folder inside the repository)."""


class RunStopped(RuntimeError):
    """The run stopped before its end (reference change)."""


# ---------------------------------------------------------------------------
# Limits, config, checks
# ---------------------------------------------------------------------------

@dataclass
class Limits:
    """Limits of one run. Times in s, cost in USD.

    The defaults of ``task_time_s`` and ``request_time_s`` depend on the
    model kind: 1800 s and 900 s for a local model, 600 s and 180 s for an
    API model.
    """

    max_turns: int = 12
    max_tool_calls: int = 10
    task_time_s: float | None = None
    tool_time_s: float = 300.0
    request_time_s: float | None = None
    max_output_tokens: int = 4096
    max_task_tokens: int | None = None
    cost_cap_usd: float = 20.0

    def resolved(self, local: bool) -> Limits:
        """Return a copy with the defaults that depend on the model kind filled in."""
        d = asdict(self)
        if d["task_time_s"] is None:
            d["task_time_s"] = 1800.0 if local else 600.0
        if d["request_time_s"] is None:
            d["request_time_s"] = 900.0 if local else 180.0
        return Limits(**d)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the private config file (JSON)."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or not isinstance(cfg.get("models"), dict):
        raise ConfigError("the config must be a JSON object with a 'models' object")
    return cfg


def check_model_allowed(label: str, model_cfg: dict[str, Any]) -> None:
    """Refuse a model on the owner blacklist (the label and the model string are checked)."""
    for name in (label, str(model_cfg.get("model", ""))):
        norm = re.sub(r"[^a-z0-9]", "", name.lower())
        for bad in BLACKLIST:
            if bad in norm:
                raise RunRefused(f"the model {name!r} is on the blacklist; the harness does not run it")


def check_out_dir(out: str | Path) -> Path:
    """Return the resolved output folder. Refuse a folder inside the repository or inside any git checkout.

    The check is by identity: each existing folder on the path (the folder
    itself or its nearest existing ancestor, and all ancestors above it) is
    compared with the repository root by :func:`os.path.samefile`, so other
    spellings of the same folder (a long-path prefix, a UNC share, a short
    name or a junction) are refused too. A folder whose ancestor holds a
    ``.git`` entry (a folder, or the file of a worktree) is also refused.
    """
    p = Path(out).expanduser()
    try:
        p = p.resolve()
    except OSError:
        p = p.absolute()
    root = REPO_ROOT.resolve()
    if p == root or root in p.parents:
        raise RunRefused(f"the output folder must be outside the repository: {p}")
    for anc in (p, *p.parents):
        try:
            if not anc.exists():
                continue
            same = os.path.samefile(anc, root)
            git = (anc / ".git").exists()
        except OSError:
            continue
        if same:
            raise RunRefused(f"the output folder must be outside the repository: {p}")
        if git:
            raise RunRefused(f"the output folder must not be inside a repository checkout ({anc.name}): {p}")
    return p


# ---------------------------------------------------------------------------
# Redaction and logs
# ---------------------------------------------------------------------------

KEY_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-+/=]{8,}"),
    re.compile(r"(?=[A-Za-z0-9+/_\-]*[A-Z])(?=[A-Za-z0-9+/_\-]*[a-z])(?=[A-Za-z0-9+/_\-]*\d)[A-Za-z0-9+/_\-]{40,}={0,2}"),
)
SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.IGNORECASE)
MIN_SECRET_LEN = 8


def key_env_names(config: Any) -> set[str]:
    """Return the names of all key variables in a config (every ``api_key_env`` value, at any depth)."""
    out: set[str] = set()
    stack = [config]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "api_key_env" and isinstance(v, str) and v:
                    out.add(v)
                else:
                    stack.append(v)
        elif isinstance(o, list):
            stack.extend(o)
    return out


class Redactor:
    """Remove keys, user names, host names and home paths from log records.

    Parameters
    ----------
    secret_env_names : iterable of str, optional
        Names of more environment variables that hold keys (for example the
        ``api_key_env`` names of the config). Their values are removed,
        whatever the variable name is. The values of variables whose names
        look like secrets are always removed.
    """

    def __init__(self, secret_env_names: Any = ()) -> None:
        lits: list[tuple[str, str]] = []
        names = set(secret_env_names or ())
        for k, v in os.environ.items():
            if (SECRET_ENV.search(k) or k in names) and v and len(v) >= MIN_SECRET_LEN:
                lits.append((v, "<redacted-key>"))
        home = str(Path.home())
        if len(home) > 3:
            lits.append((home, "~"))
            lits.append((home.replace("\\", "/"), "~"))
        for name in (self._user(), self._host()):
            if name and len(name) >= 3:
                lits.append((name, "<redacted>"))
        lits.sort(key=lambda t: -len(t[0]))
        self.literals = lits

    @staticmethod
    def _user() -> str:
        try:
            return getpass.getuser()
        except (OSError, KeyError, ImportError):
            return ""

    @staticmethod
    def _host() -> str:
        try:
            return socket.gethostname()
        except OSError:
            return ""

    def text(self, s: str) -> str:
        """Redact one string."""
        for lit, rep in self.literals:
            if lit in s:
                s = s.replace(lit, rep)
        for pat in KEY_PATTERNS:
            s = pat.sub("<redacted-key>", s)
        return s

    def __call__(self, obj: Any) -> Any:
        """Redact all strings (also dict keys) in a JSON object."""
        if isinstance(obj, str):
            return self.text(obj)
        if isinstance(obj, dict):
            return {self.text(k) if isinstance(k, str) else k: self(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self(v) for v in obj]
        return obj


class JsonlLog:
    """Append redacted JSON records to a JSON Lines file (thread-safe)."""

    def __init__(self, path: Path, redact: Redactor):
        self.path = path
        self.redact = redact
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        """Append one record."""
        line = json.dumps(self.redact(record), allow_nan=False, default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def write_json(path: Path, obj: Any, redact: Redactor) -> None:
    """Write a redacted JSON file."""
    path.write_text(json.dumps(redact(obj), indent=2, allow_nan=False, default=str), encoding="utf-8")


def redact_file(path: Path, redact: Redactor) -> None:
    """Redact a JSON Lines file in place (for example a server audit log).

    A line that is not JSON is redacted as text.
    """
    if not path.is_file():
        return
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.dumps(redact(json.loads(line)), allow_nan=False, default=str))
        except ValueError:
            out.append(redact.text(line))
    path.write_text("".join(x + "\n" for x in out), encoding="utf-8")


def latest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last record of each task and repeat (a run that is done again replaces the earlier record)."""
    last: dict[tuple[Any, Any], dict[str, Any]] = {}
    for r in records:
        last[(r.get("task_id"), r.get("repeat"))] = r
    return list(last.values())


def finished_pairs(records: list[dict[str, Any]]) -> set[tuple[Any, Any]]:
    """Return the task and repeat pairs that a resume skips. A pair whose last record has ``endpoint_error`` runs again."""
    return {(r["task_id"], r["repeat"]) for r in latest_records(records)
            if "endpoint_error" not in r.get("error_kinds", [])}


def partial_costs(events_path: Path) -> float:
    """Return the cost [USD] of the task runs that stopped at the cost cap (``run_stopped`` events)."""
    return sum(float(e.get("cost_usd", 0.0) or 0.0) for e in read_jsonl(events_path)
               if e.get("type") == "run_stopped")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file (missing file: empty list; a broken last line is skipped)."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------------------------------------------------------------------------
# Manifest facts
# ---------------------------------------------------------------------------

def _version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> dict[str, Any]:
    """Return the git commit of the repository and a flag for local changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=20).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "ventorum"], cwd=REPO_ROOT,
                                    capture_output=True, text=True, timeout=20).stdout.strip())
        return {"commit": head or None, "ventorum_changed": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "ventorum_changed": None}


def environment_facts() -> dict[str, Any]:
    """Versions and OS facts for the manifest (no host name, no user name)."""
    import ventorum

    return {
        "ventorum_version": getattr(ventorum, "__version__", None),
        "git": git_commit(),
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "torch": _version("torch"),
        "os": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
    }


def make_run_id(label: str, condition: str, task_ids: list[str], seed: int) -> str:
    """Return a run id: UTC time plus a short hash."""
    now = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    h = hashlib.sha256(f"{label}|{condition}|{','.join(task_ids)}|{seed}|{time.time_ns()}".encode()).hexdigest()[:8]
    return f"{now}-{h}"


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def request_cost(usage: dict[str, Any], prices: dict[str, float] | None, adapter_name: str) -> float:
    """Return the cost of one request [USD] from the usage and the price table (USD per million tokens)."""
    if not prices:
        return 0.0
    inp = float(usage.get("input_tokens", 0))
    out = float(usage.get("output_tokens", 0))
    cr = float(usage.get("cache_read_tokens", 0))
    cw = float(usage.get("cache_write_tokens", 0))
    if adapter_name == "openai_chat":
        inp = max(inp - cr, 0.0)  # prompt_tokens include the cached tokens
    return (inp * prices.get("input", 0.0) + out * prices.get("output", 0.0)
            + cr * prices.get("cache_read", prices.get("input", 0.0))
            + cw * prices.get("cache_write", prices.get("input", 0.0))) / 1e6


# ---------------------------------------------------------------------------
# One task run
# ---------------------------------------------------------------------------

def tools_for(condition: str, variant: TaskVariant, all_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the tool list that the model gets.

    ``full``: all tools as served. ``focused``: the task tools plus two
    fixed distractor tools that the task does not need.
    """
    if condition == "full":
        return list(all_tools)
    names = list(variant.focused_tools)
    extra = [d for d in DISTRACTORS if d not in names][:2]
    keep = set(names) | set(extra)
    return [t for t in all_tools if t["name"] in keep]


class Session:
    """Wrap a client so that the runner can close it; a shared client in tests is not closed."""

    def __init__(self, client: Any, owned: bool):
        self.client = client
        self.owned = owned

    def close(self, kill: bool = False) -> None:
        """Close (or kill) the server if this session owns it."""
        if not self.owned:
            return
        if kill:
            self.client.kill()
        self.client.close()


SessionFactory = Callable[[Path, Path], Session]


def call_key(args: dict[str, Any]) -> str:
    """Return the normalised JSON text of tool-call arguments (sorted keys, no spaces).

    Two calls of the same tool with equal keys are identical calls.
    """
    return json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)


def default_session_factory(audit_log: Path, config_dir: Path, timeout_s: float = 300.0,
                            drop_env: Any = ()) -> Session:
    """Start a new isolated server for one task run. The variables in *drop_env* are not given to the server."""
    return Session(start_session(audit_log, config_dir, timeout_s=timeout_s, drop_env=drop_env), owned=True)


@dataclass
class RunContext:
    """State shared by the task runs of one benchmark run."""

    run_id: str
    run_dir: Path
    task_set: TaskSet
    model_label: str
    model_cfg: dict[str, Any]
    adapter: BaseAdapter
    condition: str
    limits: Limits
    references: dict[str, dict[str, Any]]
    events: JsonlLog
    results: JsonlLog
    redact: Redactor
    session_factory: SessionFactory
    warmup: bool
    temperature: float
    model_seed: int | None
    cost_so_far: float = 0.0
    stop_reason: str | None = None
    warmup_facts: dict[str, Any] | None = None
    lock: Any = field(default_factory=threading.Lock)


def run_task(ctx: RunContext, variant: TaskVariant, repeat: int) -> dict[str, Any] | None:
    """Run and score one task run. Return the result record, or None if the run stopped at the cost cap."""
    base = {"schema_version": LOG_SCHEMA_VERSION, "run_id": ctx.run_id, "task_id": variant.id, "repeat": repeat}

    def event(kind: str, turn: int, **data: Any) -> None:
        ctx.events.write({**base, "type": kind, "turn": turn, "time_utc": _now(), **data})

    audit = ctx.run_dir / "audit" / f"{variant.id}_r{repeat}.jsonl"
    session = ctx.session_factory(audit, ctx.run_dir / "sandbox_config")
    client = session.client
    codes: list[str] = []
    calls: list[dict[str, Any]] = []
    succeeded: set[tuple[str, str]] = set()  # successful calls: (tool, normalised JSON of the arguments)
    usage_tot = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}
    estimated = False
    cost = 0.0
    model_time = 0.0
    tool_exec_s = 0.0
    turns = 0
    final_text = ""
    stopped_cost = False
    killed = False
    try:
        offered = tools_for(ctx.condition, variant, client.tools)
        offered_names = {t["name"] for t in offered}
        if ctx.warmup:
            w0 = time.perf_counter()
            wout = client.call_tool(WARMUP_CALL[0], WARMUP_CALL[1], timeout_s=ctx.limits.tool_time_s)
            p = wout.get("payload") or {}
            facts = {"status": p.get("status"), "device": (p.get("settings_used") or {}).get("device"),
                     "precision": (p.get("settings_used") or {}).get("precision"),
                     "duration_s": round(time.perf_counter() - w0, 3)}
            event("warmup", 0, **facts)
            with ctx.lock:
                if ctx.warmup_facts is None:
                    ctx.warmup_facts = facts
        messages: list[dict[str, Any]] = [{"role": "system", "content": ctx.task_set.system_prompt},
                                          {"role": "user", "content": variant.prompt}]
        event("task_start", 0, prompt=variant.prompt, n_tools=len(offered), tools=sorted(offered_names),
              tools_bytes=len(json.dumps(offered)))
        t0 = time.monotonic()
        lim = ctx.limits
        notice_sent = False
        ctx_len = ctx.model_cfg.get("context_length")
        while True:
            if turns >= lim.max_turns:
                codes.append("limit:turns")
                break
            remaining = lim.task_time_s - (time.monotonic() - t0)
            if remaining <= 0:
                codes.append("limit:time")
                break
            try:
                reply = ctx.adapter.complete(messages, offered, temperature=ctx.temperature, seed=ctx.model_seed,
                                             max_tokens=lim.max_output_tokens,
                                             timeout_s=max(1.0, min(lim.request_time_s, remaining)))
            except ToolCallParseError as exc:
                # The endpoint could not parse the tool call of the model: a model format failure (no retry).
                codes += ["format_error", "format_error:tool_call_parse"]
                event("tool_call_parse_error", turns + 1, message=str(exc))
                break
            except EndpointError as exc:
                codes.append("endpoint_error")
                event("endpoint_error", turns + 1, message=str(exc))
                break
            turns += 1
            model_time += reply.duration_s
            for k in usage_tot:
                usage_tot[k] += int(reply.usage.get(k, 0) or 0)
            estimated = estimated or bool(reply.usage.get("estimated"))
            c = request_cost(reply.usage, ctx.model_cfg.get("prices_usd_per_mtok"), ctx.adapter.name)
            cost += c
            event("model_reply", turns, text=reply.text,
                  tool_calls=[{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in reply.tool_calls],
                  usage=reply.usage, finish_reason=reply.finish_reason, duration_s=round(reply.duration_s, 4),
                  retries=reply.retries, provider_model=reply.provider_model, cost_usd=c)
            with ctx.lock:
                over_cap = ctx.cost_so_far + cost >= lim.cost_cap_usd
            if over_cap:
                stopped_cost = True
                break
            if ctx_len and int(reply.usage.get("input_tokens", 0) or 0) >= CONTEXT_FILL * float(ctx_len):
                codes.append("context_overflow")
                final_text = reply.text
                break
            if lim.max_task_tokens and usage_tot["input_tokens"] + usage_tot["output_tokens"] > lim.max_task_tokens:
                codes.append("limit:tokens")
                final_text = reply.text
                break
            if not reply.tool_calls:
                final_text = reply.text
                if reply.truncated:
                    codes.append("limit:tokens")
                break
            if time.monotonic() - t0 > lim.task_time_s:
                codes.append("limit:time")
                final_text = reply.text
                break
            messages.append({"role": "assistant", "content": reply.text,
                             "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                                            for tc in reply.tool_calls]})
            if notice_sent:
                final_text = reply.text
                break
            over = False
            stop = False
            for tc in reply.tool_calls:
                rec: dict[str, Any] = {"name": tc.name, "arguments": tc.arguments, "args": None, "payload": None,
                                       "rpc_error": None, "harness_error": None}
                if len([x for x in calls if x["harness_error"] != "limit"]) >= lim.max_tool_calls:
                    rec["harness_error"] = "limit"
                    text = "Error: the tool-call limit of this task is reached. The call was not run."
                    over = True
                elif tc.name not in offered_names:
                    rec["harness_error"] = "not_offered"
                    text = f"Error: unknown tool {tc.name!r}. Use one of: {', '.join(sorted(offered_names))}."
                else:
                    args, problem = parse_arguments(tc.arguments)
                    if args is None:
                        rec["harness_error"] = "bad_arguments"
                        text = f"Error: {problem}."
                    elif (tc.name, call_key(args)) in succeeded:
                        # An identical repeat of a successful call: do not run it, stop the task.
                        rec["args"] = args
                        rec["harness_error"] = "repeat_call"
                        codes.append("loop:repeat_call")
                        event("tool_call", turns, name=tc.name, arguments=args, harness_error="repeat_call")
                        calls.append(rec)
                        stop = True
                        break
                    else:
                        rec["args"] = args
                        remaining = lim.task_time_s - (time.monotonic() - t0)
                        ts0 = time.perf_counter()
                        try:
                            out = client.call_tool(tc.name, args, timeout_s=max(1.0, min(lim.tool_time_s, remaining)))
                        except McpTimeout:
                            codes.append("limit:tool_time" if remaining > lim.tool_time_s else "limit:time")
                            client.kill()
                            killed = True
                            stop = True
                            event("tool_call", turns, name=tc.name, arguments=args, status="timeout",
                                  duration_s=round(time.perf_counter() - ts0, 4))
                            calls.append(rec)
                            break
                        except McpServerExit as exc:
                            codes.append("tool_internal")
                            stop = True
                            event("tool_call", turns, name=tc.name, arguments=args, status="server_exit",
                                  message=str(exc))
                            calls.append(rec)
                            break
                        rec["payload"] = out["payload"]
                        rec["rpc_error"] = out["rpc_error"]
                        text = out["text"]
                        p = out["payload"] or {}
                        if p.get("status") == "success" and not out["is_error"] and not out["rpc_error"]:
                            succeeded.add((tc.name, call_key(args)))
                        ex_ms = p.get("execution_time_ms")
                        if isinstance(ex_ms, (int, float)):
                            tool_exec_s += float(ex_ms) / 1000.0
                        event("tool_call", turns, name=tc.name, arguments=args, is_error=out["is_error"],
                              status=p.get("status"), error_type=(p.get("error") or {}).get("type"),
                              rpc_error=out["rpc_error"], bytes=out["bytes"], execution_time_ms=ex_ms,
                              duration_s=round(time.perf_counter() - ts0, 4), payload=out["payload"])
                if rec["harness_error"]:
                    event("tool_call", turns, name=tc.name, arguments=tc.arguments, harness_error=rec["harness_error"])
                calls.append(rec)
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": text})
            if stop:
                break
            if over:
                if "limit:tool_calls" not in codes:
                    codes.append("limit:tool_calls")
                messages.append({"role": "user", "content": LIMIT_NOTICE})
                notice_sent = True
        wall = time.monotonic() - t0
    finally:
        session.close(kill=killed)
        redact_file(audit, ctx.redact)
    if stopped_cost:
        with ctx.lock:
            ctx.stop_reason = "cost cap"
            ctx.cost_so_far += cost
        event("run_stopped", turns, reason="cost cap", cost_usd=cost)
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = score_run(variant, ctx.references[variant.id], final_text, calls, codes)
    excluded = any(c in score["error_kinds"] for c in ("endpoint_error", "context_overflow"))
    record = {
        **base, "seed": variant.seed, "level": variant.level, "category": variant.category,
        "condition": ctx.condition, "model": ctx.model_label, "excluded": excluded, **score,
        "final_text": final_text, "turns": turns,
        "tokens": {**usage_tot, "estimated": estimated},
        "times_s": {"wall": round(wall, 4), "model": round(model_time, 4), "tool": round(tool_exec_s, 4),
                    "overhead": round(max(wall - model_time - tool_exec_s, 0.0), 4)},
        "cost_usd": cost, "time_utc": _now(),
    }
    with ctx.lock:
        ctx.cost_so_far += cost
    ctx.results.write(record)
    return record


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# One benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(config: dict[str, Any], model_label: str, condition: str, task_ids: list[str] | None,
                  repeats: int, out_dir: str | Path, *, run_id: str | None = None, seed: int = 0,
                  task_set: TaskSet | None = None, adapter: BaseAdapter | None = None,
                  session_factory: SessionFactory | None = None, warmup: bool = True,
                  limits: Limits | None = None, reference_call: Callable | None = None,
                  check_other_servers: bool = True, baseline: str | Path | None = None,
                  jobs: int = 1) -> Path:
    """Run the benchmark for one model and one condition.

    Parameters
    ----------
    config : dict
        The private config (``models``, optional ``limits``).
    model_label : str
        Key of the model in ``config["models"]``.
    condition : str
        ``"full"`` or ``"focused"``.
    task_ids : list of str or None
        Tasks to run (None: all).
    repeats : int
        Repeats per task (k).
    out_dir : str or Path
        Output root folder, outside the repository.
    run_id : str, optional
        Run id. Give the id of an earlier run to resume it.
    seed : int, optional
        Variant seed (0: the public set).
    task_set : TaskSet, optional
        Task set (default ``tasks_v1.json``).
    adapter : BaseAdapter, optional
        Model adapter (default: made from the config).
    session_factory : callable, optional
        ``(audit_log, config_dir) -> Session``; the default starts a new
        isolated server per task run.
    warmup : bool, optional
        Send the hidden warm-up call before the task clock starts.
    limits : Limits, optional
        Limits (default: the config ``limits`` over the built-in defaults).
    reference_call : callable, optional
        ``(tool, args) -> payload`` for the golden calls (default: in process).
    check_other_servers : bool, optional
        Warn when other Ventorum MCP servers run on this machine.
    baseline : str or Path, optional
        A stored reference set to compare with (regression check at 1e-4).
    jobs : int, optional
        Task runs in parallel (API models only; the times are then not
        comparable).

    Returns
    -------
    Path
        The run folder.
    """
    models = config.get("models", {})
    if model_label not in models:
        raise RunRefused(f"the model {model_label!r} is not in the config")
    model_cfg = dict(models[model_label])
    check_model_allowed(model_label, model_cfg)
    if condition not in CONDITIONS:
        raise RunRefused(f"the condition must be one of {CONDITIONS}")
    local = bool(model_cfg.get("local", False))
    if jobs > 1 and local:
        raise RunRefused("--jobs above 1 is only for API models (no concurrent local benchmarks)")
    out_root = check_out_dir(out_dir)
    ts = task_set or TaskSet()
    ids = ts.select(task_ids)
    if limits is None:
        limits = Limits(**config.get("limits", {}))
    limits = limits.resolved(local)
    cap = limits.cost_cap_usd
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not cap > 0:
        raise RunRefused(f"cost_cap_usd must be above 0 USD (got {cap!r}); give a large cap for a free model")
    adapter = adapter or make_adapter(model_cfg)
    key_names = key_env_names(config)
    redact = Redactor(key_names)

    run_id = run_id or make_run_id(model_label, condition, ids, seed)
    if not re.match(r"^[A-Za-z0-9_.\-]+$", run_id):
        raise RunRefused("the run id may only hold letters, digits, '_', '-' and '.'")
    run_dir = out_root / run_id
    manifest_path = run_dir / "manifest.json"
    resume = manifest_path.is_file()
    if resume:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        same = (old.get("model", {}).get("label") == model_label and old.get("condition") == condition
                and old.get("seed") == seed and old.get("task_set", {}).get("sha256") == ts.file_hash)
        if not same:
            raise RunRefused("the run id belongs to a run with another model, condition, seed or task file")
    (run_dir / "audit").mkdir(parents=True, exist_ok=True)

    if check_other_servers:
        n_other = other_mcp_servers()
        if n_other:
            print(f"Warning: {n_other} other Ventorum MCP server process(es) run on this machine. They can hold "
                  "GPU memory and lock compiled files. Times can be affected.", file=sys.stderr)
    else:
        n_other = None

    facts = environment_facts()
    refs = prepare_references(ts, ids, seed, run_dir, out_root, redact, facts, reference_call, baseline)

    if not resume:
        manifest = {
            "schema_version": LOG_SCHEMA_VERSION, "harness_version": HARNESS_VERSION, "run_id": run_id,
            "created_utc": _now(),
            "task_set": {"version": ts.version, "file": ts.path.name, "sha256": ts.file_hash},
            "system_prompt_sha256": ts.system_prompt_hash, "seed": seed, "tasks": ids, "repeats": repeats,
            "condition": condition,
            "model": {"label": model_label, **adapter.describe(), "local": local,
                      "context_length": model_cfg.get("context_length")},
            "temperature": float(model_cfg.get("temperature", 0.0)), "model_seed": model_cfg.get("seed", 0),
            "limits": asdict(limits), "jobs": jobs, "times_comparable": jobs == 1,
            "other_mcp_servers": n_other, "environment": facts, "status": "running",
        }
        write_json(manifest_path, manifest, redact)
    sandbox_config_dir(run_dir / "sandbox_config")

    ctx = RunContext(
        run_id=run_id, run_dir=run_dir, task_set=ts, model_label=model_label, model_cfg=model_cfg,
        adapter=adapter, condition=condition, limits=limits, references=refs,
        events=JsonlLog(run_dir / "events.jsonl", redact), results=JsonlLog(run_dir / "results.jsonl", redact),
        redact=redact,
        session_factory=session_factory or (lambda a, c: default_session_factory(a, c, limits.tool_time_s,
                                                                                  drop_env=key_names)),
        warmup=warmup, temperature=float(model_cfg.get("temperature", 0.0)),
        model_seed=model_cfg.get("seed", 0), lock=threading.Lock(),
    )
    done_records = read_jsonl(run_dir / "results.jsonl")
    done = finished_pairs(done_records)
    ctx.cost_so_far = (sum(float(r.get("cost_usd", 0.0)) for r in done_records)
                       + partial_costs(run_dir / "events.jsonl"))
    todo = [(rep, tid) for rep in range(1, repeats + 1) for tid in ids if (tid, rep) not in done]
    variants = {tid: ts.variant(tid, seed) for tid in ids}

    def one(item: tuple[int, str]) -> None:
        rep, tid = item
        with ctx.lock:
            if not ctx.stop_reason and ctx.cost_so_far >= limits.cost_cap_usd:
                ctx.stop_reason = "cost cap"
        if ctx.stop_reason:
            return
        run_task(ctx, variants[tid], rep)

    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(one, todo))
    else:
        for item in todo:
            one(item)
            if ctx.stop_reason:
                break

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = f"stopped ({ctx.stop_reason})" if ctx.stop_reason else "finished"
    manifest["updated_utc"] = _now()
    manifest["cost_usd"] = ctx.cost_so_far
    if ctx.warmup_facts and not manifest.get("warmup"):
        manifest["warmup"] = ctx.warmup_facts
    write_json(manifest_path, manifest, redact)
    return run_dir


def prepare_references(ts: TaskSet, ids: list[str], seed: int, run_dir: Path, out_root: Path, redact: Redactor,
                       facts: dict[str, Any], reference_call: Callable | None = None,
                       baseline: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Compute or load the references of a run and check them (track A and track B).

    The golden calls of the tasks that ``references.json`` of the run folder
    does not hold yet are run. The run stops if a golden call does not give
    the expected result, or if a value differs (at 1e-4 relative) from the
    references of an earlier run of the same Ventorum commit (or from
    *baseline*).

    Raises
    ------
    RunStopped
        A golden call is not valid, or a reference value changed.
    """
    ref_path = run_dir / "references.json"
    if ref_path.is_file():
        refs = json.loads(ref_path.read_text(encoding="utf-8"))
        missing = [t for t in ids if t not in refs]
        if missing:
            refs.update(compute_all(ts, missing, seed, reference_call))
            write_json(ref_path, refs, redact)
    else:
        refs = compute_all(ts, ids, seed, reference_call)
        write_json(ref_path, refs, redact)
    bad = {t: refs[t]["problems"] for t in ids if not refs[t]["valid"]}
    if bad:
        raise RunStopped(f"golden calls do not give the expected results: {bad}")
    prev = _previous_references(out_root, run_dir, facts, ts, seed, baseline)
    if prev is not None:
        changes = regression_check({t: refs[t] for t in ids}, prev)
        if changes:
            raise RunStopped("reference values changed for the same Ventorum commit (a verification matter): "
                             + "; ".join(changes[:10]))
    return refs


def _previous_references(out_root: Path, run_dir: Path, facts: dict[str, Any], ts: TaskSet, seed: int,
                         baseline: str | Path | None) -> dict[str, Any] | None:
    if baseline is not None:
        return json.loads(Path(baseline).read_text(encoding="utf-8"))
    commit = facts["git"].get("commit")
    if not commit or facts["git"].get("ventorum_changed"):
        return None
    best = None
    for m in sorted(out_root.glob("*/manifest.json")):
        if m.parent == run_dir:
            continue
        try:
            man = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        g = man.get("environment", {}).get("git", {})
        if (g.get("commit") == commit and not g.get("ventorum_changed") and man.get("seed") == seed
                and man.get("task_set", {}).get("sha256") == ts.file_hash and (m.parent / "references.json").is_file()):
            best = m.parent / "references.json"
    return json.loads(best.read_text(encoding="utf-8")) if best else None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def bootstrap_interval(per_task: dict[str, list[bool]], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95 % interval of the success rate by a two-level bootstrap (tasks, then repeats)."""
    tasks = [v for v in per_task.values() if v]
    if not tasks:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        tot = 0.0
        for _ in range(len(tasks)):
            reps = tasks[rng.randrange(len(tasks))]
            tot += sum(reps[rng.randrange(len(reps))] for _ in range(len(reps))) / len(reps)
        stats.append(tot / len(tasks))
    stats.sort()
    return stats[int(0.025 * (n_boot - 1))], stats[int(0.975 * (n_boot - 1))]


def _rate(vals: list[Any]) -> float | None:
    v = [bool(x) for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def _fmt(x: Any, spec: str = ".3f") -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return format(x, spec)


def summarise(records: list[dict[str, Any]], extra_cost_usd: float = 0.0) -> dict[str, Any]:
    """Compute the summary statistics of the result records of one run.

    Only the last record of each task and repeat counts (a run that is done
    again replaces the earlier record). The cost is the sum over all
    records, plus *extra_cost_usd* (the task runs that stopped at the cost
    cap) [USD].
    """
    all_records = records
    records = latest_records(records)
    inc = [r for r in records if not r.get("excluded")]
    per_task: dict[str, list[bool]] = {}
    for r in inc:
        per_task.setdefault(r["task_id"], []).append(bool(r["success"]))
    k = max((len(v) for v in per_task.values()), default=0)
    full_k = [v for v in per_task.values() if len(v) == k and k > 0]
    levels: dict[int, list[bool]] = {}
    for r in inc:
        levels.setdefault(int(r["level"]), []).append(bool(r["success"]))
    kinds: dict[str, int] = {}
    for r in records:
        for c in r.get("error_kinds", []):
            kinds[c] = kinds.get(c, 0) + 1
    lo, hi = bootstrap_interval(per_task)

    def mean(xs: list[float]) -> float | None:
        return statistics.fmean(xs) if xs else None

    return {
        "n_records": len(records), "n_excluded": len(records) - len(inc), "n_tasks": len(per_task),
        "repeats_max": k,
        "strict_success_rate": _rate([r["success"] for r in inc]), "interval_95": [lo, hi],
        "lenient_success_rate": _rate([r.get("lenient_success") for r in inc]),
        "value_score_mean": mean([r["value_score"] for r in inc if r.get("value_score") is not None]),
        "preferred_tool_rate": _rate([r.get("preferred_tool_used") for r in inc]),
        "first_call_valid_rate": _rate([r.get("first_call_valid") for r in inc]),
        "recovery_rate": _rate([r.get("recovered") for r in inc]),
        "pass_k": (sum(all(v) for v in full_k) / len(full_k)) if full_k else None,
        "tasks_changing": sum(1 for v in per_task.values() if len(set(v)) > 1),
        "per_level": {lev: _rate(v) for lev, v in sorted(levels.items())},
        "tool_calls_mean": mean([r["n_tool_calls"] for r in inc]),
        "input_tokens_mean": mean([r["tokens"]["input_tokens"] for r in inc]),
        "output_tokens_mean": mean([r["tokens"]["output_tokens"] for r in inc]),
        "tokens_estimated": any(r["tokens"].get("estimated") for r in inc),
        "wall_s_mean": mean([r["times_s"]["wall"] for r in inc]),
        "model_s_mean": mean([r["times_s"]["model"] for r in inc]),
        "tool_s_mean": mean([r["times_s"]["tool"] for r in inc]),
        "cost_usd_total": sum(float(r.get("cost_usd", 0.0)) for r in all_records) + float(extra_cost_usd),
        "error_kinds": dict(sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def report(run_dir: str | Path) -> dict[str, Any]:
    """Write ``summary.md`` and ``summary.csv`` of a run from ``results.jsonl``. Return the summary."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    all_records = read_jsonl(run_dir / "results.jsonl")
    s = summarise(all_records, partial_costs(run_dir / "events.jsonl"))
    records = latest_records(all_records)
    redact = Redactor()
    m = manifest.get("model", {})
    lines = [
        f"# Agent benchmark summary: run {manifest['run_id']}",
        "",
        "This file is generated by `python -m validation.agent_benchmark.runner report`. Do not edit it.",
        "",
        f"- Model: {m.get('label')} ({m.get('adapter')}, {m.get('model')})",
        f"- Condition: {manifest.get('condition')}; task set {manifest['task_set']['version']}; "
        f"variant seed {manifest.get('seed')}; repeats {manifest.get('repeats')}",
        f"- Ventorum {manifest['environment'].get('ventorum_version')}, commit "
        f"{manifest['environment']['git'].get('commit')}; harness {manifest.get('harness_version')}",
        f"- Times comparable: {manifest.get('times_comparable')}; status: {manifest.get('status')}",
        "",
        "## Main metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| task runs (excluded) | {s['n_records']} ({s['n_excluded']}) |",
        f"| strict success rate | {_fmt(s['strict_success_rate'])} |",
        f"| 95 % interval (bootstrap over tasks) | {_fmt(s['interval_95'][0])} to {_fmt(s['interval_95'][1])} |",
        f"| pass^{s['repeats_max']} | {_fmt(s['pass_k'])} |",
        f"| tasks whose result changes between repeats | {s['tasks_changing']} |",
        f"| lenient success rate (not for ranking) | {_fmt(s['lenient_success_rate'])} |",
        f"| value score | {_fmt(s['value_score_mean'])} |",
        f"| preferred-tool rate | {_fmt(s['preferred_tool_rate'])} |",
        f"| first-call validity | {_fmt(s['first_call_valid_rate'])} |",
        f"| recovery rate | {_fmt(s['recovery_rate'])} |",
        f"| tool calls per task | {_fmt(s['tool_calls_mean'], '.2f')} |",
        f"| input / output tokens per task{' (estimated)' if s['tokens_estimated'] else ''} | "
        f"{_fmt(s['input_tokens_mean'], '.0f')} / {_fmt(s['output_tokens_mean'], '.0f')} |",
        f"| wall / model / tool time per task [s] | {_fmt(s['wall_s_mean'], '.2f')} / "
        f"{_fmt(s['model_s_mean'], '.2f')} / {_fmt(s['tool_s_mean'], '.2f')} |",
        f"| estimated cost [USD] | {_fmt(s['cost_usd_total'], '.4f')} |",
        "",
        "## Success rate per level",
        "",
        "| level | rate |",
        "| --- | --- |",
    ]
    lines += [f"| {lev} | {_fmt(v)} |" for lev, v in s["per_level"].items()]
    lines += ["", "## Error kinds", "", "| code | task runs |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in s["error_kinds"].items()]
    lines += ["", "## Tasks", "", "| task | level | successes | error kinds |", "| --- | --- | --- | --- |"]
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_task.setdefault(r["task_id"], []).append(r)
    rows = []
    for tid in sorted(by_task):
        rs = by_task[tid]
        kinds = sorted({c for r in rs for c in r.get("error_kinds", [])})
        nsucc = sum(1 for r in rs if r["success"] and not r.get("excluded"))
        lines.append(f"| {tid} | {rs[0]['level']} | {nsucc}/{len(rs)} | {', '.join(kinds)} |")
        rows.append({"task_id": tid, "level": rs[0]["level"], "runs": len(rs), "successes": nsucc,
                     "excluded": sum(1 for r in rs if r.get("excluded")),
                     "mean_tool_calls": statistics.fmean(r["n_tool_calls"] for r in rs),
                     "mean_input_tokens": statistics.fmean(r["tokens"]["input_tokens"] for r in rs),
                     "mean_output_tokens": statistics.fmean(r["tokens"]["output_tokens"] for r in rs),
                     "mean_wall_s": statistics.fmean(r["times_s"]["wall"] for r in rs),
                     "cost_usd": sum(float(r.get("cost_usd", 0.0)) for r in rs),
                     "error_kinds": ";".join(kinds)})
    (run_dir / "summary.md").write_text(redact.text("\n".join(lines) + "\n"), encoding="utf-8")
    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["task_id", "level", "runs", "successes", "excluded", "mean_tool_calls", "mean_input_tokens",
                  "mean_output_tokens", "mean_wall_s", "cost_usd", "error_kinds"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return s


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_selftest(task_file: str | Path | None = None, rel: float = 0.005) -> bool:
    """Check the task file and the golden calls against the library API. Print the result."""
    ts = TaskSet(task_file)
    ok = True
    print(f"Task set {ts.version}: {len(ts.tasks)} tasks, file sha256 {ts.file_hash[:12]}")
    for tid in ts.ids:
        v = ts.variant(tid, 0)
        c = v.canonical_call
        if c is None:
            continue
        probs = check_call_arguments(c["tool"], c["args"], deep=c["expect"] == "success")
        if probs:
            ok = False
            print(f"FAIL {tid}: canonical arguments: {probs}")
    checks = selftest(ts, rel=rel)
    for chk in checks:
        flag = "ok  " if chk["ok"] else "FAIL"
        print(f"{flag} {chk['name']}: relative difference {chk['rel_diff']:.2e} (limit {rel:.1e})")
        ok = ok and chk["ok"]
    print("selftest passed" if ok else "selftest FAILED")
    return ok


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run the command line. Return the exit status."""
    ap = argparse.ArgumentParser(prog="python -m validation.agent_benchmark.runner",
                                 description="Agent benchmark of the Ventorum MCP interface.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Run the benchmark for one model and one condition.")
    r.add_argument("--config", required=True, help="Private config file (JSON, outside the repository).")
    r.add_argument("--model", required=True, help="Model label in the config.")
    r.add_argument("--condition", choices=CONDITIONS, default="focused")
    r.add_argument("--tasks", default="all", help="all, track_b, level:N, or ids such as AB-01,AB-05:AB-09.")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--out", required=True, help="Output root folder, outside the repository.")
    r.add_argument("--run-id", help="Run id; give an earlier id to resume that run.")
    r.add_argument("--seed", type=int, default=0, help="Variant seed (0: public set).")
    r.add_argument("--task-file", help="Task file (default tasks_v1.json).")
    r.add_argument("--baseline", help="Stored reference set for the regression check.")
    r.add_argument("--no-warmup", action="store_true", help="Do not send the hidden warm-up call.")
    r.add_argument("--jobs", type=int, default=1, help="Parallel task runs (API models only).")
    for name in ("max_turns", "max_tool_calls", "max_output_tokens", "max_task_tokens"):
        r.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("task_time_s", "tool_time_s", "request_time_s", "cost_cap_usd"):
        r.add_argument("--" + name.replace("_", "-"), type=float)
    rp = sub.add_parser("report", help="Write summary.md and summary.csv of a run.")
    rp.add_argument("run_dir")
    rf = sub.add_parser("references", help="Compute the reference values and write them to a folder.")
    rf.add_argument("--out", required=True)
    rf.add_argument("--tasks", default="all")
    rf.add_argument("--seed", type=int, default=0)
    rf.add_argument("--task-file")
    st = sub.add_parser("selftest", help="Check the task file and the golden calls against the library API.")
    st.add_argument("--task-file")
    ns = ap.parse_args(argv)

    try:
        if ns.cmd == "selftest":
            return 0 if run_selftest(ns.task_file) else 1
        if ns.cmd == "report":
            s = report(ns.run_dir)
            print(f"Wrote {Path(ns.run_dir) / 'summary.md'}: strict success rate {_fmt(s['strict_success_rate'])} "
                  f"over {s['n_records'] - s['n_excluded']} task runs.")
            return 0
        if ns.cmd == "references":
            out = check_out_dir(ns.out)
            out.mkdir(parents=True, exist_ok=True)
            ts = TaskSet(ns.task_file)
            refs = compute_all(ts, ts.select(ns.tasks), ns.seed)
            path = out / f"references_{ts.version}_seed{ns.seed}.json"
            write_json(path, {"environment": environment_facts(), "references": refs}, Redactor())
            n_bad = sum(1 for v in refs.values() if not v["valid"])
            print(f"Wrote {path}: {len(refs)} tasks, {n_bad} with problems.")
            return 0 if n_bad == 0 else 1
        cfg = load_config(ns.config)
        lim_over = {k: getattr(ns, k) for k in ("max_turns", "max_tool_calls", "max_output_tokens", "max_task_tokens",
                                                "task_time_s", "tool_time_s", "request_time_s", "cost_cap_usd")
                    if getattr(ns, k) is not None}
        limits = Limits(**{**cfg.get("limits", {}), **lim_over})
        run_dir = run_benchmark(cfg, ns.model, ns.condition, None if ns.tasks == "all" else ns.tasks, ns.repeats,
                                ns.out, run_id=ns.run_id, seed=ns.seed,
                                task_set=TaskSet(ns.task_file) if ns.task_file else None,
                                warmup=not ns.no_warmup, limits=limits, baseline=ns.baseline, jobs=ns.jobs)
        s = report(run_dir)
        print(f"Run folder: {run_dir}")
        print(f"Strict success rate {_fmt(s['strict_success_rate'])} over {s['n_records'] - s['n_excluded']} task runs.")
        return 0
    except (RunRefused, RunStopped, ConfigError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
