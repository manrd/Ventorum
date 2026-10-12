# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Track B of the agent benchmark: existing MCP clients driven by scripts.

The client (opencode or Claude Code) runs headless in an empty temporary
folder, with only the Ventorum MCP server and no shell, file or web tools.
The harness scores each run from the final text of the client and from the
server audit log (``VENTORUM_AGENT_AUDIT_LOG``): it replays the logged tool
arguments through :func:`ventorum.agent.call_tool` to get the payloads for
the provenance check. Results of track B are reported apart from track A,
because each client adds its own system prompt and counts tokens in its own
way.

    python -m validation.agent_benchmark.client_track --config CONFIG --client opencode
        --model MODEL --out DIR [--tasks track_b] [--repeats 1]

The client command lines are templates. The defaults below are the known
forms; check them against the installed client version, and override them
in the private config (``clients.<name>.command``). The placeholders are
``{exe}``, ``{model}``, ``{prompt}``, ``{mcp_config}`` and ``{workdir}``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any
from collections.abc import Callable

from validation.agent_benchmark import HARNESS_VERSION, LOG_SCHEMA_VERSION
from validation.agent_benchmark.mcp_client import make_server_env, sandbox_config_dir
from validation.agent_benchmark.references import compute_all, in_process_call
from validation.agent_benchmark.runner import (
    JsonlLog,
    Redactor,
    RunRefused,
    _now,
    check_model_allowed,
    check_out_dir,
    environment_facts,
    load_config,
    make_run_id,
    read_jsonl,
    report,
    write_json,
)
from validation.agent_benchmark.scoring import score_run
from validation.agent_benchmark.tasks import TaskSet, TaskVariant

#: Built-in tools of the clients that must be off (no shell, files or web).
OPENCODE_TOOLS_OFF = ("bash", "edit", "write", "read", "patch", "glob", "grep", "list", "webfetch", "websearch",
                      "todowrite", "todoread", "task")
CLAUDE_TOOLS_OFF = ("Bash", "Read", "Write", "Edit", "MultiEdit", "Glob", "Grep", "LS", "WebFetch", "WebSearch",
                    "Task", "NotebookEdit", "TodoWrite")

DEFAULT_CLIENTS: dict[str, dict[str, Any]] = {
    "opencode": {
        "exe": "opencode-cli",
        "command": ["{exe}", "run", "--standalone", "-m", "{model}", "{prompt}"],
        "output": "text",
    },
    "claude_code": {
        "exe": "claude",
        "command": ["{exe}", "-p", "{prompt}", "--output-format", "json", "--model", "{model}",
                    "--mcp-config", "{mcp_config}", "--strict-mcp-config",
                    "--allowedTools", "mcp__ventorum", "--disallowedTools", ",".join(CLAUDE_TOOLS_OFF)],
        "output": "json",
    },
}


def server_entry(audit_log: Path, config_dir: Path) -> tuple[list[str], dict[str, str]]:
    """Return the command and the environment of the Ventorum MCP server for a client config."""
    env = make_server_env(audit_log, config_dir)
    keep = {k: env[k] for k in ("PYTHONPATH", "MPLBACKEND", "PYTHONIOENCODING", "VENTORUM_AGENT_AUDIT_LOG",
                                "VENTORUM_CONFIG_DIR") if k in env}
    return [sys.executable, "-m", "ventorum.agent", "--mcp"], keep


def write_client_config(client: str, workdir: Path, audit_log: Path, config_dir: Path) -> Path:
    """Write the MCP config of a client into *workdir*. Return its path.

    opencode reads ``opencode.json`` in the project folder: only the
    Ventorum server, and the built-in tools off. Claude Code gets an MCP
    config file for ``--mcp-config``.
    """
    cmd, env = server_entry(audit_log, config_dir)
    if client == "opencode":
        cfg = {"mcp": {"ventorum": {"type": "local", "command": cmd, "enabled": True, "environment": env}},
               "tools": {name: False for name in OPENCODE_TOOLS_OFF}}
        path = workdir / "opencode.json"
    elif client == "claude_code":
        cfg = {"mcpServers": {"ventorum": {"command": cmd[0], "args": cmd[1:], "env": env}}}
        path = workdir / "mcp_config.json"
    else:
        raise RunRefused(f"unknown client {client!r}")
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def build_command(client_cfg: dict[str, Any], model: str, prompt: str, mcp_config: Path, workdir: Path) -> list[str]:
    """Fill in the command template of a client."""
    subs = {"{exe}": str(client_cfg["exe"]), "{model}": model, "{prompt}": prompt,
            "{mcp_config}": str(mcp_config), "{workdir}": str(workdir)}
    out = []
    for part in client_cfg["command"]:
        for k, v in subs.items():
            part = part.replace(k, v)
        out.append(part)
    return out


def parse_client_output(stdout: str, kind: str) -> dict[str, Any]:
    """Return ``{"text", "usage", "cost_usd"}`` from the client output (JSON or plain text)."""
    if kind == "json":
        try:
            obj = json.loads(stdout.strip().splitlines()[-1] if stdout.strip() else "{}")
        except (json.JSONDecodeError, IndexError):
            obj = None
        if isinstance(obj, dict):
            u = obj.get("usage") or {}
            return {"text": str(obj.get("result", "")),
                    "usage": {"input_tokens": int(u.get("input_tokens", 0) or 0),
                              "output_tokens": int(u.get("output_tokens", 0) or 0),
                              "cache_read_tokens": int(u.get("cache_read_input_tokens", 0) or 0),
                              "cache_write_tokens": int(u.get("cache_creation_input_tokens", 0) or 0),
                              "estimated": not bool(u)},
                    "cost_usd": float(obj.get("total_cost_usd", 0.0) or 0.0)}
    return {"text": stdout, "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                                      "cache_write_tokens": 0, "estimated": True}, "cost_usd": 0.0}


def calls_from_audit(audit_log: Path, call: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None
                     ) -> list[dict[str, Any]]:
    """Rebuild the tool calls of a run from the server audit log.

    The audit log has the tool name, the status and the arguments of each
    call, not the payload. The arguments are run again (default: in this
    process) to get the payloads for the provenance check.
    """
    call = call or in_process_call
    out = []
    for e in read_jsonl(audit_log):
        args = e.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        rec: dict[str, Any] = {"name": e.get("tool"), "arguments": e.get("arguments"), "args": None, "payload": None,
                               "rpc_error": None, "harness_error": None}
        if isinstance(args, dict):
            rec["args"] = args
            rec["payload"] = call(str(e.get("tool")), args)
        else:
            rec["harness_error"] = "bad_arguments"
        out.append(rec)
    return out


def score_client_run(variant: TaskVariant, references: dict[str, Any], final_text: str, audit_log: Path,
                     call: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
                     extra_codes: list[str] | None = None) -> dict[str, Any]:
    """Score one client run from its final text and its audit log."""
    calls = calls_from_audit(audit_log, call)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return score_run(variant, references, final_text, calls, extra_codes)


def run_client_track(config: dict[str, Any], client: str, model: str, task_ids: list[str] | str | None,
                     repeats: int, out_dir: str | Path, *, run_id: str | None = None, seed: int = 0,
                     task_set: TaskSet | None = None, timeout_s: float = 1800.0) -> Path:
    """Run a client on a task subset and score each run. Return the run folder."""
    check_model_allowed(model, {"model": model})
    out_root = check_out_dir(out_dir)
    ccfg = {**DEFAULT_CLIENTS.get(client, {}), **(config.get("clients", {}).get(client, {}))}
    if not ccfg.get("command"):
        raise RunRefused(f"no command for the client {client!r}")
    ts = task_set or TaskSet()
    ids = ts.select(task_ids if task_ids is not None else "track_b")
    run_id = run_id or make_run_id(f"{client}:{model}", "client", ids, seed)
    run_dir = out_root / run_id
    (run_dir / "audit").mkdir(parents=True, exist_ok=True)
    redact = Redactor()
    ref_path = run_dir / "references.json"
    if ref_path.is_file():
        refs = json.loads(ref_path.read_text(encoding="utf-8"))
    else:
        refs = compute_all(ts, ids, seed)
        write_json(ref_path, refs, redact)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        write_json(manifest_path, {
            "schema_version": LOG_SCHEMA_VERSION, "harness_version": HARNESS_VERSION, "run_id": run_id,
            "track": "B", "created_utc": _now(), "client": client, "client_command": ccfg["command"],
            "task_set": {"version": ts.version, "file": ts.path.name, "sha256": ts.file_hash},
            "system_prompt_sha256": ts.system_prompt_hash, "seed": seed, "tasks": ids, "repeats": repeats,
            "condition": "client", "model": {"label": f"{client}:{model}", "adapter": client, "model": model},
            "environment": environment_facts(), "status": "running",
            "times_comparable": True,
        }, redact)
    cfg_dir = sandbox_config_dir(run_dir / "sandbox_config")
    results = JsonlLog(run_dir / "results.jsonl", redact)
    events = JsonlLog(run_dir / "events.jsonl", redact)
    done = {(r["task_id"], r["repeat"]) for r in read_jsonl(run_dir / "results.jsonl")}
    for rep in range(1, repeats + 1):
        for tid in ids:
            if (tid, rep) in done:
                continue
            v = ts.variant(tid, seed)
            audit = run_dir / "audit" / f"{tid}_r{rep}.jsonl"
            work = Path(tempfile.mkdtemp(prefix="vt_client_"))
            try:
                mcfg = write_client_config(client, work, audit, cfg_dir)
                prompt = ts.system_prompt + "\n\nTask: " + v.prompt
                cmd = build_command(ccfg, model, prompt, mcfg, work)
                codes: list[str] = []
                t0 = time.monotonic()
                try:
                    proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True, encoding="utf-8",
                                          errors="replace", timeout=timeout_s)
                    stdout, rc = proc.stdout, proc.returncode
                except subprocess.TimeoutExpired as exc:
                    stdout = exc.stdout if isinstance(exc.stdout, str) else ""
                    rc = None
                    codes.append("limit:time")
                except OSError as exc:
                    raise RunRefused(f"cannot start the client: {exc}") from exc
                wall = time.monotonic() - t0
            finally:
                shutil.rmtree(work, ignore_errors=True)
            out = parse_client_output(stdout, ccfg.get("output", "text"))
            if rc not in (0, None):
                codes.append("endpoint_error")
            score = score_client_run(v, refs[tid], out["text"], audit, extra_codes=codes)
            events.write({"schema_version": LOG_SCHEMA_VERSION, "run_id": run_id, "task_id": tid, "repeat": rep,
                          "type": "client_run", "time_utc": _now(), "return_code": rc, "final_text": out["text"]})
            results.write({
                "schema_version": LOG_SCHEMA_VERSION, "run_id": run_id, "task_id": tid, "repeat": rep,
                "seed": seed, "level": v.level, "category": v.category, "condition": "client",
                "model": f"{client}:{model}", "excluded": "endpoint_error" in score["error_kinds"], **score,
                "final_text": out["text"], "turns": None, "tokens": out["usage"],
                "times_s": {"wall": round(wall, 3), "model": 0.0, "tool": 0.0, "overhead": 0.0},
                "cost_usd": out["cost_usd"], "time_utc": _now(),
            })
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    man["status"] = "finished"
    man["updated_utc"] = _now()
    write_json(manifest_path, man, redact)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    """Run the command line of track B. Return the exit status."""
    ap = argparse.ArgumentParser(prog="python -m validation.agent_benchmark.client_track",
                                 description="Track B: score existing MCP clients on a task subset.")
    ap.add_argument("--config", required=True, help="Private config file (JSON, outside the repository).")
    ap.add_argument("--client", required=True, choices=sorted(DEFAULT_CLIENTS))
    ap.add_argument("--model", required=True, help="Model string of the client.")
    ap.add_argument("--out", required=True, help="Output root folder, outside the repository.")
    ap.add_argument("--tasks", default="track_b")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--run-id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ns = ap.parse_args(argv)
    try:
        run_dir = run_client_track(load_config(ns.config), ns.client, ns.model, ns.tasks, ns.repeats, ns.out,
                                   run_id=ns.run_id, seed=ns.seed, timeout_s=ns.timeout_s)
    except RunRefused as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    s = report(run_dir)
    print(f"Run folder: {run_dir}; strict success rate {s['strict_success_rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
