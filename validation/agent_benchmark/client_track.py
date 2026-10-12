# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Track B of the agent benchmark: existing MCP clients driven by scripts.

The client (opencode or Claude Code) runs headless in an empty temporary
folder, with only the Ventorum MCP server and no shell, file or web tools.
The prompt goes to the client through its standard input (and the file
``prompt.txt`` in the work folder), never on the command line. The client
runs in a new process group; at a timeout the harness stops the client and
all its child processes (for example the MCP server).

The harness scores each run from the final text of the client and from the
server audit log (``VENTORUM_AGENT_AUDIT_LOG``): it replays the logged tool
arguments through a separate Ventorum MCP server with a sandbox config
folder, to get the payloads for the provenance check. A logged call of
``ventorum_tune_machine`` is never replayed: its status comes from the audit
line, with no numbers. Results of track B are reported apart from track A,
because each client adds its own system prompt and counts tokens in its own
way.

    python -m validation.agent_benchmark.client_track --config CONFIG --client opencode
        --model MODEL --out DIR [--tasks track_b] [--repeats 1]

The client command lines are templates. The defaults below are the known
forms; check them against the installed client version, and override them
in the private config (``clients.<name>.command``). The placeholders are
``{exe}``, ``{model}``, ``{prompt_file}``, ``{mcp_config}`` and ``{workdir}``.
A template with ``{prompt}`` is refused: the prompt must not be on the
command line.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any
from collections.abc import Callable

from validation.agent_benchmark import HARNESS_VERSION, LOG_SCHEMA_VERSION
from validation.agent_benchmark.mcp_client import make_server_env, sandbox_config_dir, start_session
from validation.agent_benchmark.references import mcp_call
from validation.agent_benchmark.runner import (
    JsonlLog,
    Redactor,
    RunRefused,
    _now,
    check_model_allowed,
    check_out_dir,
    environment_facts,
    finished_pairs,
    key_env_names,
    load_config,
    make_run_id,
    prepare_references,
    read_jsonl,
    redact_file,
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

#: Tools that are never replayed from the audit log (they change the machine state).
NO_REPLAY = ("ventorum_tune_machine",)

DEFAULT_CLIENTS: dict[str, dict[str, Any]] = {
    "opencode": {
        "exe": "opencode-cli",
        "command": ["{exe}", "run", "--standalone", "-m", "{model}"],
        "output": "text",
    },
    "claude_code": {
        "exe": "claude",
        "command": ["{exe}", "-p", "--output-format", "json", "--model", "{model}",
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


def build_command(client_cfg: dict[str, Any], model: str, prompt_file: Path, mcp_config: Path,
                  workdir: Path) -> list[str]:
    """Fill in the command template of a client.

    Raises
    ------
    RunRefused
        The template has the placeholder ``{prompt}`` (the prompt goes
        through the standard input or ``{prompt_file}``).
    """
    if any("{prompt}" in str(part) for part in client_cfg["command"]):
        raise RunRefused("the client command must not hold {prompt}: the prompt goes through the standard input "
                         "or {prompt_file}")
    subs = {"{exe}": str(client_cfg["exe"]), "{model}": model, "{prompt_file}": str(prompt_file),
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


def _status_payload(entry: dict[str, Any]) -> dict[str, Any]:
    """Return a payload with the status of an audit line only (no numbers)."""
    p: dict[str, Any] = {"status": entry.get("status"), "replayed": False}
    if entry.get("error_type"):
        p["error"] = {"type": entry.get("error_type")}
    return p


def calls_from_audit(audit_log: Path, call: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
                     config_dir: Path | None = None, drop_env: Any = ()) -> list[dict[str, Any]]:
    """Rebuild the tool calls of a run from the server audit log.

    The audit log has the tool name, the status and the arguments of each
    call, not the payload. The arguments are run again to get the payloads
    for the provenance check. The tools of :data:`NO_REPLAY` are not run
    again: their payload holds only the status of the audit line.

    Parameters
    ----------
    audit_log : Path
        The audit log of the client run.
    call : callable, optional
        ``call(tool, args) -> payload``. The default starts a Ventorum MCP
        server with a sandbox copy of the profile folder (in *config_dir*,
        or in a temporary folder), and closes it at the end.
    config_dir : Path, optional
        Sandbox profile folder of the default server.
    drop_env : iterable of str, optional
        Environment variables that the default server must not get.
    """
    session = None
    tmp: Path | None = None
    out = []
    try:
        for e in read_jsonl(audit_log):
            tool = str(e.get("tool"))
            args = e.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = None
            rec: dict[str, Any] = {"name": e.get("tool"), "arguments": e.get("arguments"), "args": None,
                                   "payload": None, "rpc_error": None, "harness_error": None}
            if tool in NO_REPLAY:
                rec["args"] = args if isinstance(args, dict) else None
                rec["payload"] = _status_payload(e)
            elif isinstance(args, dict):
                if call is None:
                    if config_dir is None:
                        tmp = Path(tempfile.mkdtemp(prefix="vt_replay_cfg_"))
                        config_dir = tmp
                    session = start_session(config_dir=sandbox_config_dir(config_dir), drop_env=drop_env)
                    call = mcp_call(session)
                rec["args"] = args
                rec["payload"] = call(tool, args)
            else:
                rec["harness_error"] = "bad_arguments"
            out.append(rec)
    finally:
        if session is not None:
            session.close()
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    return out


def kill_tree(proc: subprocess.Popen) -> None:
    """Stop a process and all its child processes (Windows: ``taskkill /T /F``; POSIX: the process group)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if proc.poll() is None:
        proc.kill()


def run_client_process(cmd: list[str], cwd: Path, prompt: str, timeout_s: float,
                       env: dict[str, str] | None = None) -> tuple[str, int | None, bool]:
    """Run a client with the prompt on its standard input.

    The client starts in a new process group. At the timeout, the client and
    all its child processes are stopped.

    Parameters
    ----------
    cmd : list of str
        Command line (with no prompt in it).
    cwd : Path
        Working folder.
    prompt : str
        Prompt text, written to the standard input.
    timeout_s : float
        Time limit [s].
    env : dict, optional
        Environment (default: the environment of this process).

    Returns
    -------
    tuple
        ``(stdout, return_code, timed_out)``; ``return_code`` is None after
        a timeout.

    Raises
    ------
    OSError
        The client cannot start.
    """
    kw: dict[str, Any] = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", env=env, **kw)
    try:
        stdout, _ = proc.communicate(prompt, timeout=timeout_s)
        return stdout or "", proc.returncode, False
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        try:
            stdout, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            stdout = ""
        return stdout or "", None, True


def score_client_run(variant: TaskVariant, references: dict[str, Any], final_text: str, audit_log: Path,
                     call: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
                     extra_codes: list[str] | None = None) -> dict[str, Any]:
    """Score one client run from its final text and its audit log (see :func:`calls_from_audit`)."""
    calls = calls_from_audit(audit_log, call)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return score_run(variant, references, final_text, calls, extra_codes)


def run_client_track(config: dict[str, Any], client: str, model: str, task_ids: list[str] | str | None,
                     repeats: int, out_dir: str | Path, *, run_id: str | None = None, seed: int = 0,
                     task_set: TaskSet | None = None, timeout_s: float = 1800.0,
                     reference_call: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
                     baseline: str | Path | None = None) -> Path:
    """Run a client on a task subset and score each run. Return the run folder.

    The references are checked as in track A (see
    :func:`validation.agent_benchmark.runner.prepare_references`): the run
    stops before a client starts if a golden call is not valid or a
    reference value changed.
    """
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
    key_names = key_env_names(config)
    redact = Redactor(key_names)
    facts = environment_facts()
    refs = prepare_references(ts, ids, seed, run_dir, out_root, redact, facts, reference_call, baseline)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        write_json(manifest_path, {
            "schema_version": LOG_SCHEMA_VERSION, "harness_version": HARNESS_VERSION, "run_id": run_id,
            "track": "B", "created_utc": _now(), "client": client, "client_command": ccfg["command"],
            "task_set": {"version": ts.version, "file": ts.path.name, "sha256": ts.file_hash},
            "system_prompt_sha256": ts.system_prompt_hash, "seed": seed, "tasks": ids, "repeats": repeats,
            "condition": "client", "model": {"label": f"{client}:{model}", "adapter": client, "model": model},
            "environment": facts, "status": "running",
            "times_comparable": True,
        }, redact)
    cfg_dir = sandbox_config_dir(run_dir / "sandbox_config")
    results = JsonlLog(run_dir / "results.jsonl", redact)
    events = JsonlLog(run_dir / "events.jsonl", redact)
    done = finished_pairs(read_jsonl(run_dir / "results.jsonl"))
    replay: dict[str, Any] = {}

    def replay_call(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        # One replay server for the whole run, with its own sandbox profile folder.
        if "call" not in replay:
            replay["session"] = start_session(config_dir=sandbox_config_dir(run_dir / "replay_config"),
                                              drop_env=key_names)
            replay["call"] = mcp_call(replay["session"])
        return replay["call"](tool, args)

    try:
        _run_client_tasks(ts, ids, repeats, seed, done, client, model, ccfg, cfg_dir, refs, run_id, run_dir,
                          timeout_s, replay_call, results, events, redact)
    finally:
        if "session" in replay:
            replay["session"].close()
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    man["status"] = "finished"
    man["updated_utc"] = _now()
    write_json(manifest_path, man, redact)
    return run_dir


def _run_client_tasks(ts: TaskSet, ids: list[str], repeats: int, seed: int, done: set, client: str, model: str,
                      ccfg: dict[str, Any], cfg_dir: Path, refs: dict[str, Any], run_id: str, run_dir: Path,
                      timeout_s: float, replay_call: Callable, results: JsonlLog, events: JsonlLog,
                      redact: Redactor) -> None:
    """Run and score the client runs that are not done yet."""
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
                prompt_file = work / "prompt.txt"
                prompt_file.write_text(prompt, encoding="utf-8")
                cmd = build_command(ccfg, model, prompt_file, mcfg, work)
                codes: list[str] = []
                t0 = time.monotonic()
                try:
                    stdout, rc, timed_out = run_client_process(cmd, work, prompt, timeout_s)
                except OSError as exc:
                    raise RunRefused(f"cannot start the client: {exc}") from exc
                if timed_out:
                    codes.append("limit:time")
                wall = time.monotonic() - t0
            finally:
                shutil.rmtree(work, ignore_errors=True)
            out = parse_client_output(stdout, ccfg.get("output", "text"))
            if rc not in (0, None):
                codes.append("endpoint_error")
            score = score_client_run(v, refs[tid], out["text"], audit, call=replay_call, extra_codes=codes)
            redact_file(audit, redact)
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
