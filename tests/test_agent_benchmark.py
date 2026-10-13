# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the agent benchmark harness (validation/agent_benchmark).

No network and no LLM: a scripted fake model replays fixed tool calls and
final answers through the real Ventorum MCP server. One server process is
shared by the tests of this module (it is warmed up once), so that each
test stays short.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import pytest

from validation.agent_benchmark.adapters import ScriptedAdapter
from validation.agent_benchmark.mcp_client import sandbox_config_dir, start_session
from validation.agent_benchmark.references import compute_references, mcp_call
from validation.agent_benchmark.runner import (
    WARMUP_CALL,
    Limits,
    RunRefused,
    Session,
    check_out_dir,
    read_jsonl,
    report,
    run_benchmark,
)
from validation.agent_benchmark.scoring import compare_call, describe_call, extract_final_answer
from validation.agent_benchmark.tasks import TaskSet, check_call_arguments

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def task_set():
    return TaskSet()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    d = tmp_path_factory.mktemp("agent_benchmark_server")
    client = start_session(audit_log=d / "audit.jsonl", config_dir=sandbox_config_dir(d / "config"))
    client.call_tool(*WARMUP_CALL, timeout_s=300)  # compile and warm up once for the module
    yield client
    client.kill()
    client.close()


@pytest.fixture(scope="module")
def refs(task_set, server):
    cache: dict[str, dict] = {}

    def get(tid: str) -> dict:
        if tid not in cache:
            cache[tid] = compute_references(task_set.variant(tid, 0), mcp_call(server))
        return cache[tid]

    return get


def final(status: str, values: dict, caveats: list | None = None) -> str:
    body = json.dumps({"status": status, "values": values, "caveats": caveats or []})
    return f"Here is the result.\n<final_answer>\n{body}\n</final_answer>"


def last_payload(messages: list[dict]) -> dict:
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    return json.loads(tool_msgs[-1]["content"])


CONFIG = {"models": {"fake": {"adapter": "scripted", "model": "fake-model", "local": True}}}


def run_fake(server, task_set, tmp_path, task_ids, steps, *, limits=None, repeats=1, run_id="test-run",
             delay_s=0.0, repeat_last=False, config=None):
    cfg = config or CONFIG
    label = next(iter(cfg["models"]))
    adapter = ScriptedAdapter({**cfg["models"][label], "delay_s": delay_s, "repeat_last": repeat_last}, steps)
    run_dir = run_benchmark(
        cfg, label, "focused", task_ids, repeats, tmp_path, run_id=run_id, task_set=task_set,
        adapter=adapter, session_factory=lambda audit, cdir: Session(server, owned=False), warmup=False,
        limits=limits, reference_call=mcp_call(server), check_other_servers=False,
    )
    return run_dir, read_jsonl(run_dir / "results.jsonl"), adapter


def canonical_args(task_set, tid):
    return task_set.variant(tid, 0).canonical_call["args"]


def call_step(name, args):
    return {"text": "", "tool_calls": [{"name": name, "arguments": args}]}


# ---------------------------------------------------------------------------
# Client, task file, references
# ---------------------------------------------------------------------------

def test_mcp_client_lists_12_tools(server):
    """The client starts the server and gets 12 tools and protocol 2025-06-18."""
    assert server.protocol_version == "2025-06-18"
    names = [t["name"] for t in server.tools]
    assert len(names) == 12 and len(set(names)) == 12
    assert "ventorum_wing_analysis" in names and "ventorum_tune_machine" in names
    assert all("inputSchema" in t for t in server.tools)


def test_task_file_valid(task_set):
    """49 tasks with unique ids and all fields; every canonical call has valid arguments."""
    raw = json.loads((ROOT / "validation" / "agent_benchmark" / "tasks_v1.json").read_text(encoding="utf-8"))
    assert "do not train on this file" in raw["canary"].lower()
    tasks = raw["tasks"]
    assert len(tasks) == 49
    ids = [t["id"] for t in tasks]
    assert len(set(ids)) == 49 and ids == [f"AB-{i:02d}" for i in range(1, 50)]
    for t in tasks:
        for key in ("level", "prompt", "answer_keys", "tools", "calls", "canonical", "tolerances", "refusal"):
            assert key in t, (t["id"], key)
        v = task_set.variant(t["id"], 0)
        assert v.prompt.endswith("Answer keys: " + ", ".join(v.answer_keys))
        assert "{" not in v.prompt and "}" not in v.prompt, v.id
        call = v.canonical_call
        if call is None:
            assert not t["tools"] and t["refusal"] == "yes", v.id
            continue
        problems = check_call_arguments(call["tool"], call["args"], deep=call["expect"] == "success")
        assert not problems, (v.id, problems)
        for k in {k for oc in v.outcomes for k in oc["values"]}:
            assert k in v.tolerances and k in v.references, (v.id, k)
    # A held-out variant changes the parameters but keeps the structure.
    a, b = task_set.variant("AB-02", 0), task_set.variant("AB-02", 5)
    assert a.params != b.params and a.answer_keys == b.answer_keys


def test_references_reproduce(task_set, refs):
    """The golden calls of AB-01, AB-19 and AB-24 give finite values; refusal tasks give the expected error."""
    for tid in ("AB-01", "AB-19", "AB-24"):
        r = refs(tid)
        assert r["valid"], (tid, r["problems"])
        assert r["values"] and all(isinstance(x, float) and math.isfinite(x) for x in r["values"].values()), tid
    for tid, etype in (("AB-39", "invalid_input"), ("AB-41", "invalid_method"), ("AB-42", "ground_strike"),
                       ("AB-49", "invalid_input")):
        r = refs(tid)
        assert r["valid"], (tid, r["problems"])
        assert r["calls"]["main"]["error_type"] == etype


# ---------------------------------------------------------------------------
# Scoring through the real server
# ---------------------------------------------------------------------------

def test_score_correct_answer(server, task_set, refs, tmp_path):
    vals = refs("AB-01")["values"]
    steps = [call_step("ventorum_wing_analysis", canonical_args(task_set, "AB-01")),
             {"text": final("answered", {"CL": vals["CL"], "CDi": vals["CDi"]})}]
    _, res, _ = run_fake(server, task_set, tmp_path, ["AB-01"], steps)
    assert res[0]["success"] is True
    assert res[0]["error_kinds"] == []
    assert res[0]["preferred_tool_used"] and res[0]["first_call_valid"]


def test_score_wrong_value(server, task_set, refs, tmp_path):
    vals = refs("AB-01")["values"]
    steps = [call_step("ventorum_wing_analysis", canonical_args(task_set, "AB-01")),
             {"text": final("answered", {"CL": vals["CDi"], "CDi": vals["CDi"]})}]
    _, res, _ = run_fake(server, task_set, tmp_path, ["AB-01"], steps)
    assert res[0]["success"] is False
    assert "wrong_value" in res[0]["error_kinds"]
    assert res[0]["per_value"]["CDi"]["ok"] and not res[0]["per_value"]["CL"]["ok"]


def test_score_missed_refusal(server, task_set, tmp_path):
    steps = [call_step("ventorum_wing_analysis", canonical_args(task_set, "AB-42")),
             {"text": final("answered", {"CL": 0.61})}]
    _, res, _ = run_fake(server, task_set, tmp_path, ["AB-42"], steps)
    assert res[0]["success"] is False
    assert "missed_refusal" in res[0]["error_kinds"]
    assert "ground_strike" not in res[0]["error_kinds"]  # the task expects this error type


def test_score_correct_refusal(server, task_set, tmp_path):
    steps = [call_step("ventorum_wing_analysis", canonical_args(task_set, "AB-42")),
             {"text": final("refused", {}, ["ground_strike"])}]
    _, res, _ = run_fake(server, task_set, tmp_path, ["AB-42"], steps)
    assert res[0]["success"] is True and res[0]["error_kinds"] == []


def test_score_false_refusal(server, task_set, tmp_path):
    steps = [call_step("ventorum_wing_analysis", canonical_args(task_set, "AB-01")),
             {"text": final("refused", {}, ["out_of_envelope"])}]
    _, res, _ = run_fake(server, task_set, tmp_path, ["AB-01"], steps)
    assert res[0]["success"] is False
    assert "false_refusal" in res[0]["error_kinds"]


def test_score_hallucinated_value(server, task_set, refs, tmp_path):
    vals = refs("AB-01")["values"]
    steps = [call_step("ventorum_wing_analysis", canonical_args(task_set, "AB-01")),
             {"text": final("answered", {"CL": 0.7777, "CDi": vals["CDi"]})}]
    _, res, _ = run_fake(server, task_set, tmp_path, ["AB-01"], steps)
    assert res[0]["success"] is False
    assert "hallucinated_value" in res[0]["error_kinds"]


def test_score_unsupported_value(server, task_set, refs, tmp_path):
    """A right number that no tool result holds is a lucky guess: it fails."""
    vals = refs("AB-01")["values"]
    args = json.loads(json.dumps(canonical_args(task_set, "AB-01")))
    args["flight_condition"]["alpha_deg"] += 2.0
    steps = [call_step("ventorum_wing_analysis", args),
             {"text": final("answered", {"CL": vals["CL"], "CDi": vals["CDi"]})}]
    _, res, _ = run_fake(server, task_set, tmp_path, ["AB-01"], steps)
    assert res[0]["success"] is False
    assert "unsupported_value" in res[0]["error_kinds"]
    assert "geometry_drift" in res[0]["error_kinds"]


def test_score_wrong_units(server, task_set, tmp_path):
    """An angle in radians given as degrees is a unit error."""
    v = task_set.variant("AB-06", 0)
    args = json.loads(json.dumps(v.canonical_call["args"]))
    args["flight_condition"]["alpha_deg"] = v.params["alpha_rad"]
    steps = [call_step("ventorum_wing_analysis", args),
             lambda msgs, tools: {"text": final("answered", {"CL": last_payload(msgs)["metrics"]["CL"]})}]
    _, res, _ = run_fake(server, task_set, tmp_path, ["AB-06"], steps)
    assert res[0]["success"] is False
    assert "wrong_units" in res[0]["error_kinds"]


def test_score_sign_error(server, task_set, tmp_path):
    """Washout given as positive twist is a sign error."""
    args = json.loads(json.dumps(canonical_args(task_set, "AB-04")))
    args["wing"]["tip_twist_deg"] = -args["wing"]["tip_twist_deg"]

    def answer(msgs, tools):
        m = last_payload(msgs)["metrics"]
        return {"text": final("answered", {"CL": m["CL"], "CD": m["CD"], "L_over_D": m["L_over_D"]})}

    _, res, _ = run_fake(server, task_set, tmp_path, ["AB-04"], [call_step("ventorum_wing_analysis", args), answer])
    assert res[0]["success"] is False
    assert "sign_error" in res[0]["error_kinds"]
    assert "geometry_drift" not in res[0]["error_kinds"]


def test_score_format_repair(server, task_set, refs, tmp_path):
    vals = refs("AB-01")["values"]
    body = f'{{"status": "answered", "values": {{"CL": {vals["CL"]!r}, "CDi": {vals["CDi"]!r},}}, "caveats": [],}}'
    text = "<final_answer>\n```json\n" + body + "\n```\n</final_answer>"
    steps = [call_step("ventorum_wing_analysis", canonical_args(task_set, "AB-01")), {"text": text}]
    _, res, _ = run_fake(server, task_set, tmp_path, ["AB-01"], steps)
    assert res[0]["success"] is True
    assert "format_repaired" in res[0]["error_kinds"]
    ans, codes = extract_final_answer("no block here")
    assert ans is None and codes == ["format_error"]


def test_equivalent_geometry_accepted(server, task_set, refs, tmp_path):
    """semi_span_m instead of span_m, and two equal sections instead of chord_m, are not geometry_drift."""
    v = task_set.variant("AB-01", 0)
    canon = v.canonical_call["args"]
    span, chord = canon["wing"]["span_m"], canon["wing"]["chord_m"]
    equiv = {"wing": {"semi_span_m": span / 2.0, "sections": [{"y_frac": 0.0, "chord_m": chord},
                                                               {"y_frac": 1.0, "chord_m": chord}]},
             "flight_condition": dict(canon["flight_condition"])}
    tool = v.canonical_call["tool"]
    assert compare_call(describe_call(tool, equiv), describe_call(tool, canon)) == set()
    shifted = json.loads(json.dumps(equiv))
    shifted["wing"]["sections"][1]["chord_m"] = 0.5 * chord
    assert "geometry_drift" in compare_call(describe_call(tool, shifted), describe_call(tool, canon))
    vals = refs("AB-01")["values"]
    steps = [call_step(tool, equiv), {"text": final("answered", {"CL": vals["CL"], "CDi": vals["CDi"]})}]
    _, res, _ = run_fake(server, task_set, tmp_path, ["AB-01"], steps)
    assert res[0]["success"] is True
    assert "geometry_drift" not in res[0]["error_kinds"]


# ---------------------------------------------------------------------------
# Limits, keys, resume, blacklist
# ---------------------------------------------------------------------------

BAD_CALL = call_step("ventorum_wing_analysis", {"wing": {"span_m": 10.0, "chord_m": -1.0}})


def test_limits(server, task_set, tmp_path):
    """Turn, tool-call and time limits stop a looping fake model with the right limit code."""
    _, res, ad = run_fake(server, task_set, tmp_path / "turns", ["AB-01"], [BAD_CALL], repeat_last=True,
                          limits=Limits(max_turns=3, max_tool_calls=100))
    assert "limit:turns" in res[0]["error_kinds"] and res[0]["turns"] == 3 and ad.n_requests == 3
    assert "invalid_input:range" in res[0]["error_kinds"]

    _, res, ad = run_fake(server, task_set, tmp_path / "calls", ["AB-01"], [BAD_CALL], repeat_last=True,
                          limits=Limits(max_turns=50, max_tool_calls=2))
    assert "limit:tool_calls" in res[0]["error_kinds"]
    assert res[0]["n_harness_refused"] >= 1 and res[0]["n_tool_calls"] - res[0]["n_harness_refused"] == 2
    assert ad.n_requests == 4  # two runs, one refused call, then the notice and one more reply

    _, res, _ = run_fake(server, task_set, tmp_path / "time", ["AB-01"], [BAD_CALL], repeat_last=True,
                         delay_s=0.2, limits=Limits(max_turns=1000, max_tool_calls=1000, task_time_s=0.5))
    assert "limit:time" in res[0]["error_kinds"]
    assert res[0]["success"] is False


def test_no_key_in_logs(server, task_set, tmp_path, monkeypatch):
    """A fake key in the environment does not appear in any output file."""
    key = "sk-test" + "Z9" * 20
    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setenv("VENTORUM_BENCH_FAKE_TOKEN", "tok" + "Q7" * 12)
    args = json.loads(json.dumps(canonical_args(task_set, "AB-01")))
    args["wing"]["name"] = "wing " + key
    steps = [call_step("ventorum_wing_analysis", args),
             {"text": f"I used the key {key} and Bearer {key}. " + final("refused", {}, [])}]
    run_dir, res, _ = run_fake(server, task_set, tmp_path, ["AB-01"], steps)
    report(run_dir)
    files = [p for p in run_dir.rglob("*") if p.is_file()]
    assert any(p.name == "events.jsonl" for p in files) and any(p.name == "summary.md" for p in files)
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        assert key not in text, p.name
        assert "Q7Q7Q7Q7" not in text, p.name


def test_resume(server, task_set, tmp_path):
    """A second run of the same run_id skips the finished records."""
    def refuse(msgs, tools):
        return {"text": final("refused", {}, [])}

    run_dir, res, ad = run_fake(server, task_set, tmp_path, ["AB-01", "AB-06"], [refuse], repeat_last=True,
                                run_id="resume-run")
    assert len(res) == 2 and ad.n_requests == 2
    _, res2, ad2 = run_fake(server, task_set, tmp_path, ["AB-01", "AB-06"], [refuse], repeat_last=True,
                            run_id="resume-run")
    assert ad2.n_requests == 0 and len(res2) == 2
    _, res3, ad3 = run_fake(server, task_set, tmp_path, ["AB-01", "AB-06"], [refuse], repeat_last=True,
                            run_id="resume-run", repeats=2)
    assert ad3.n_requests == 2 and len(res3) == 4
    assert sorted((r["task_id"], r["repeat"]) for r in res3) == [("AB-01", 1), ("AB-01", 2), ("AB-06", 1),
                                                                 ("AB-06", 2)]
    summary = report(run_dir)
    assert summary["n_records"] == 4 and summary["strict_success_rate"] == 0.0
    assert (run_dir / "summary.csv").is_file()


def test_blacklisted_model_refused(server, task_set, tmp_path):
    for cfg in ({"models": {"nemotron-3-ultra": {"adapter": "scripted", "model": "x"}}},
                {"models": {"free": {"adapter": "scripted", "model": "nvidia/Nemotron-3-Ultra-free"}}}):
        with pytest.raises(RunRefused, match="blacklist"):
            run_fake(server, task_set, tmp_path, ["AB-01"], [], config=cfg)
    assert not any(tmp_path.iterdir())


def test_output_folder_inside_repository_refused():
    with pytest.raises(RunRefused, match="outside the repository"):
        check_out_dir(ROOT / "agent_benchmark_runs")


def test_client_track_scores_from_audit_log(server, task_set, refs, tmp_path):
    """Track B scores a run from the final text and the server audit log (no client is started)."""
    from validation.agent_benchmark.client_track import build_command, score_client_run, write_client_config

    v = task_set.variant("AB-01", 0)
    vals = refs("AB-01")["values"]
    audit = tmp_path / "audit.jsonl"
    entry = {"tool": "ventorum_wing_analysis", "status": "success", "error_type": None,
             "arguments": v.canonical_call["args"]}
    audit.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    score = score_client_run(v, refs("AB-01"), final("answered", {"CL": vals["CL"], "CDi": vals["CDi"]}),
                             audit, call=mcp_call(server))
    assert score["success"] is True and score["n_tool_calls"] == 1
    path = write_client_config("claude_code", tmp_path, audit, tmp_path / "cfg")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert list(cfg["mcpServers"]) == ["ventorum"]
    prompt_file = tmp_path / "prompt.txt"
    cmd = build_command({"exe": "client", "command": ["{exe}", "-p", "--file", "{prompt_file}", "--mcp-config",
                                                      "{mcp_config}"]}, "m", prompt_file, path, tmp_path)
    assert cmd == ["client", "-p", "--file", str(prompt_file), "--mcp-config", str(path)]


# ---------------------------------------------------------------------------
# Regression tests of the review findings
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """Return True if the process *pid* runs (no signal is sent)."""
    if os.name == "nt":
        import ctypes

        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            k32.GetExitCodeProcess(h, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _score(task_set, tid, final_text, calls, references=None):
    from validation.agent_benchmark.scoring import score_run

    v = task_set.variant(tid, 0)
    return score_run(v, references if references is not None else compute_references(v, lambda t, a: {}),
                     final_text, calls)


def test_client_track_never_replays_the_tuner(tmp_path, monkeypatch):
    """Track B takes the tuner status from the audit line and replays the other calls in a sandbox session."""
    from validation.agent_benchmark import client_track
    from validation.agent_benchmark.scoring import numbers_in

    audit = tmp_path / "audit.jsonl"
    lines = [{"tool": "ventorum_tune_machine", "status": "success", "error_type": None, "arguments": {}},
             {"tool": "ventorum_wing_analysis", "status": "success", "error_type": None,
              "arguments": {"wing": {"span_m": 10.0, "chord_m": 1.0}}}]
    audit.write_text("".join(json.dumps(e) + "\n" for e in lines), encoding="utf-8")
    seen = []

    def call(name, args):
        seen.append(name)
        return {"status": "success", "metrics": {"CL": 0.3}}

    calls = client_track.calls_from_audit(audit, call)
    assert seen == ["ventorum_wing_analysis"]
    assert calls[0]["name"] == "ventorum_tune_machine" and calls[0]["payload"]["status"] == "success"
    assert numbers_in(calls[0]["payload"]) == []

    # The default replay never runs in this process: it uses a server with a sandbox config folder.
    def no_in_process(*a, **k):
        raise AssertionError("in-process replay")

    started = {}

    class FakeClient:
        def call_tool(self, name, args, timeout_s=None):
            seen.append("session:" + name)
            return {"payload": {"status": "success"}, "text": "{}", "is_error": False, "rpc_error": None,
                    "bytes": 2}

        def kill(self):
            pass

        def close(self):
            started["closed"] = True

    def fake_start(audit_log=None, config_dir=None, timeout_s=300.0, **kw):
        started["config_dir"] = config_dir
        return FakeClient()

    monkeypatch.setattr(client_track, "in_process_call", no_in_process, raising=False)
    monkeypatch.setattr(client_track, "start_session", fake_start, raising=False)
    seen.clear()
    client_track.calls_from_audit(audit)
    assert seen == ["session:ventorum_wing_analysis"]
    assert started["config_dir"] is not None and started.get("closed")


PLACEHOLDERS = ("n/a", "", "unknown", "None", "null", "NA", "-", "N/A", "UNKNOWN")


def test_refusal_with_placeholder_values(task_set):
    """AB-43: a correct refusal with placeholder values succeeds and gets no no_tool code."""
    for ph in PLACEHOLDERS:
        s = _score(task_set, "AB-43", final("needs_input", {"CL": ph, "CDi": ph}, ["missing_input"]), [])
        assert s["success"] is True, ph
        assert "no_tool" not in s["error_kinds"], ph
        assert s["values"] == {"CL": None, "CDi": None}, ph
    s = _score(task_set, "AB-43", final("needs_input", {"CL": 0.3, "CDi": "n/a"}, ["missing_input"]), [])
    assert s["success"] is False and "no_tool" in s["error_kinds"]
    s = _score(task_set, "AB-43", final("needs_input", {"CL": True}, ["missing_input"]), [])
    assert "no_tool" in s["error_kinds"]


def test_trim_start_alpha_is_not_geometry_drift(server, task_set, refs):
    """AB-24: alpha in the trim flight condition is only a start value."""
    v = task_set.variant("AB-24", 0)
    assert v.ignore_condition == ["alpha"]
    call = mcp_call(server)
    for fc in ({"V_inf_m_s": v.params["V"]}, {"V_inf_m_s": v.params["V"], "alpha_deg": 0.0},
               {"V_inf_m_s": v.params["V"], "alpha_deg": 5.0}):
        args = json.loads(json.dumps(v.canonical_call["args"]))
        args["flight_condition"] = fc
        p = call("ventorum_trim", args)
        assert p["status"] == "success", fc
        text = final("answered", {"alpha_deg": p["alpha_deg"], "elevator_deg": p["deflections_deg"]["elevator"]})
        s = _score(task_set, "AB-24", text,
                   [{"name": "ventorum_trim", "args": args, "payload": p, "rpc_error": None, "harness_error": None}],
                   refs("AB-24"))
        assert "geometry_drift" not in s["error_kinds"], (fc, s["error_kinds"])
        assert s["success"] is True, (fc, s["error_kinds"])
    # A speed change is still a drift.
    args = json.loads(json.dumps(v.canonical_call["args"]))
    args["flight_condition"]["V_inf_m_s"] = v.params["V"] + 10.0
    canon = describe_call("ventorum_trim", v.canonical_call["args"])
    assert "geometry_drift" in compare_call(describe_call("ventorum_trim", args), canon, ignore=v.ignore_condition)


def test_ignore_condition_validated(tmp_path):
    """The loader refuses an unknown name in ignore_condition."""
    from validation.agent_benchmark.tasks import TaskFileError

    raw = json.loads((ROOT / "validation" / "agent_benchmark" / "tasks_v1.json").read_text(encoding="utf-8"))
    for bad in (["alpha", "mach"], "alpha"):
        data = json.loads(json.dumps(raw))
        data["tasks"][23]["ignore_condition"] = bad
        path = tmp_path / "tasks_bad.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(TaskFileError, match="ignore_condition"):
            TaskSet(path)


def test_redactor_gets_config_key_variables(server, task_set, tmp_path, monkeypatch):
    """A key in a variable of any name that the config names is redacted and kept from the server."""
    from validation.agent_benchmark.mcp_client import make_server_env
    from validation.agent_benchmark.runner import Redactor, key_env_names

    key = "AIza" + "b7Kq2" * 7
    assert len(key) == 39
    monkeypatch.setenv("LLM_GATEWAY_CRED", key)
    cfg = {"models": {"fake": {"adapter": "scripted", "model": "fake-model", "local": True,
                               "api_key_env": "LLM_GATEWAY_CRED"}}}
    names = key_env_names(cfg)
    assert names == {"LLM_GATEWAY_CRED"}
    assert key not in Redactor(names).text("HTTP 400: API key not valid: " + key)
    assert "LLM_GATEWAY_CRED" not in make_server_env(drop=names)
    steps = [{"text": f"The endpoint said: key {key} is not valid. " + final("refused", {}, [])}]
    run_dir, _, _ = run_fake(server, task_set, tmp_path, ["AB-01"], steps, config=cfg)
    report(run_dir)
    for p in run_dir.rglob("*"):
        if p.is_file():
            assert key not in p.read_text(encoding="utf-8", errors="replace"), p.name


def test_cost_cap_stop_and_resume(server, task_set, tmp_path):
    """The cost cap stops the run; the partial cost is recorded and counts on resume."""
    cfg = {"models": {"api": {"adapter": "scripted", "model": "m", "local": False,
                              "prices_usd_per_mtok": {"input": 1000.0, "output": 1000.0}}}}
    refuse = {"text": final("refused", {}, [])}
    run_dir, res, ad = run_fake(server, task_set, tmp_path, ["AB-01", "AB-02"], [refuse], repeat_last=True,
                                run_id="cap", config=cfg, limits=Limits(cost_cap_usd=0.01))
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert man["status"] == "stopped (cost cap)" and ad.n_requests == 1 and res == []
    assert man["cost_usd"] > 0.01
    _, res2, ad2 = run_fake(server, task_set, tmp_path, ["AB-01", "AB-02"], [refuse], repeat_last=True,
                            run_id="cap", config=cfg, limits=Limits(cost_cap_usd=0.01))
    man2 = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert ad2.n_requests == 0 and res2 == []
    assert man2["status"] == "stopped (cost cap)" and man2["cost_usd"] == pytest.approx(man["cost_usd"])
    assert report(run_dir)["cost_usd_total"] == pytest.approx(man["cost_usd"])


def test_cost_cap_must_be_positive(server, task_set, tmp_path):
    for cap in (0.0, -1.0):
        with pytest.raises(RunRefused, match="cost_cap_usd"):
            run_fake(server, task_set, tmp_path, ["AB-01"], [], limits=Limits(cost_cap_usd=cap))


def test_endpoint_error_retried_on_resume(server, task_set, tmp_path):
    """A record with endpoint_error is not done: a resume runs the task again and the report keeps the last record."""
    from validation.agent_benchmark.adapters import EndpointError

    def down(msgs, tools):
        raise EndpointError("HTTP 503")

    run_dir, res, _ = run_fake(server, task_set, tmp_path, ["AB-01"], [down], run_id="retry")
    assert len(res) == 1 and "endpoint_error" in res[0]["error_kinds"]
    _, res2, ad2 = run_fake(server, task_set, tmp_path, ["AB-01"], [{"text": final("refused", {}, [])}],
                            run_id="retry")
    assert ad2.n_requests == 1 and len(res2) == 2
    s = report(run_dir)
    assert s["n_records"] == 1 and s["n_excluded"] == 0


def test_output_folder_repository_identity(tmp_path):
    """The repository is refused by identity (prefix forms), and so is any folder under a git checkout."""
    if os.name == "nt":
        with pytest.raises(RunRefused, match="outside the repository"):
            check_out_dir("\\\\?\\" + str(ROOT / "agent_benchmark_runs"))
    for kind in ("dir", "file"):
        fake = tmp_path / f"checkout_{kind}"
        fake.mkdir()
        if kind == "dir":
            (fake / ".git").mkdir()
        else:
            (fake / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
        with pytest.raises(RunRefused, match="repository"):
            check_out_dir(fake / "runs" / "out")
    assert check_out_dir(tmp_path / "plain" / "out") == (tmp_path / "plain" / "out").resolve()


def test_output_folder_unc_form():
    """The repository is refused through an administrative UNC share, where the share is available."""
    if os.name != "nt":
        pytest.skip("UNC paths exist only on Windows")
    drive, rest = os.path.splitdrive(str(ROOT))
    unc = "\\\\localhost\\" + drive.rstrip(":") + "$" + rest
    if not os.path.exists(unc):
        pytest.skip("the administrative share is not available without administrator rights")
    with pytest.raises(RunRefused, match="outside the repository"):
        check_out_dir(unc + "\\agent_benchmark_runs")


def test_mcp_client_reads_large_reply_fast():
    """A 2 MB reply is read in well under the bound (buffered pipes)."""
    from validation.agent_benchmark.mcp_client import McpClient, make_server_env

    srv = ("import sys,json\n"
           "for line in sys.stdin:\n"
           "    m=json.loads(line)\n"
           "    if 'id' in m: print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':{'pad':'x'*2000000}}),"
           " flush=True)\n")
    c = McpClient(command=[sys.executable, "-c", srv], env=make_server_env()).start()
    try:
        c.request("warm", {}, timeout_s=60)
        t0 = time.perf_counter()
        result, n = c.request("big", {}, timeout_s=60)
        dt = time.perf_counter() - t0
    finally:
        c.close()
    assert n > 2_000_000 and len(result["pad"]) == 2_000_000
    assert dt < 2.0, dt


def test_client_process_tree_killed_on_timeout(tmp_path):
    """A client that times out is killed with all its children; the prompt goes through stdin."""
    from validation.agent_benchmark.client_track import RunRefused as CtRefused
    from validation.agent_benchmark.client_track import build_command, run_client_process

    pid_file = tmp_path / "child.pid"
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],\n"
        "                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n", encoding="utf-8")
    t0 = time.monotonic()
    stdout, rc, timed_out = run_client_process([sys.executable, str(parent)], tmp_path, "hello", 4.0)
    assert timed_out and rc is None and time.monotonic() - t0 < 30
    child = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 10
    while _pid_alive(child) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _pid_alive(child), "the child of the client is still running"

    echo = tmp_path / "echo.py"
    echo.write_text("import sys\nsys.stdout.write('got:' + sys.stdin.read())\n", encoding="utf-8")
    stdout, rc, timed_out = run_client_process([sys.executable, str(echo)], tmp_path, "the task prompt", 60.0)
    assert stdout == "got:the task prompt" and rc == 0 and not timed_out
    with pytest.raises(CtRefused, match="prompt"):
        build_command({"exe": "c", "command": ["{exe}", "{prompt}"]}, "m", tmp_path / "p.txt", tmp_path, tmp_path)


def test_client_track_checks_references(tmp_path):
    """Track B stops when a golden call does not give the expected result, before a client starts."""
    from validation.agent_benchmark.client_track import run_client_track
    from validation.agent_benchmark.runner import RunStopped

    cfg = {"models": {}, "clients": {"claude_code": {"exe": "no-such-client-program"}}}
    with pytest.raises(RunStopped, match="golden calls"):
        run_client_track(cfg, "claude_code", "m", ["AB-01"], 1, tmp_path, run_id="refcheck",
                         reference_call=lambda t, a: {"status": "error", "error": {"type": "internal"}})


def test_geometry_compares_reference_point(task_set):
    """AB-35 without ref_point_m is geometry_drift; an equal point through x_cg_m is not."""
    v = task_set.variant("AB-35", 0)
    canon = describe_call(v.canonical_call["tool"], v.canonical_call["args"])
    args = json.loads(json.dumps(v.canonical_call["args"]))
    del args["aircraft"]["ref_point_m"]
    assert "geometry_drift" in compare_call(describe_call("ventorum_wing_analysis", args), canon)
    args["aircraft"]["ref_point_m"] = [1.5, 0.0, 0.0]
    assert "geometry_drift" in compare_call(describe_call("ventorum_wing_analysis", args), canon)
    v = task_set.variant("AB-19", 0)
    canon = describe_call(v.canonical_call["tool"], v.canonical_call["args"])
    args = json.loads(json.dumps(v.canonical_call["args"]))
    del args["aircraft"]["ref_point_m"]
    assert compare_call(describe_call(v.canonical_call["tool"], args), canon) == set()


def test_answer_format_edge_cases(task_set):
    """Caveats as one string, ranges, the degree sign and the Unicode minus."""
    from validation.agent_benchmark.scoring import normalise_value

    text = '<final_answer>{"status": "needs_input", "values": {}, "caveats": "missing_input"}</final_answer>'
    ans, codes = extract_final_answer(text)
    assert ans["caveats"] == ["missing_input"] and "format_repaired" in codes
    s = _score(task_set, "AB-43", text, [])
    assert s["success"] is True and "format_repaired" in s["error_kinds"]
    for rng in ("3 to 5", "3 - 5", "3 or 5", "between 3 and 5"):
        val, _ = normalise_value(rng)
        assert not isinstance(val, float), rng
    s = _score(task_set, "AB-43", final("answered", {"CL": "0.3 to 0.4", "CDi": 0.004}), [])
    assert "format_error" in s["error_kinds"]
    assert normalise_value("5.49\u00b0") == (5.49, True)
    assert normalise_value("\u22121.73 deg") == (-1.73, True)
    assert normalise_value("4.8 /rad") == (4.8, True)
    assert normalise_value("1.2 kg/m^3") == (1.2, True)


def test_jobs_two(server, task_set, tmp_path):
    """--jobs 2 runs every task once and marks the times as not comparable."""
    cfg = {"models": {"api": {"adapter": "scripted", "model": "m", "local": False}}}
    label = "api"
    adapter = ScriptedAdapter({**cfg["models"][label], "repeat_last": True}, [{"text": final("refused", {}, [])}])
    run_dir = run_benchmark(cfg, label, "focused", ["AB-01", "AB-02", "AB-06"], 1, tmp_path, run_id="jobs",
                            task_set=task_set, adapter=adapter,
                            session_factory=lambda audit, cdir: Session(server, owned=False), warmup=False,
                            reference_call=mcp_call(server), check_other_servers=False, jobs=2)
    recs = read_jsonl(run_dir / "results.jsonl")
    assert sorted(r["task_id"] for r in recs) == ["AB-01", "AB-02", "AB-06"]
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert man["jobs"] == 2 and man["times_comparable"] is False and man["status"] == "finished"


@pytest.mark.slow
def test_default_session_warmup_and_audit_redacted(task_set, tmp_path, monkeypatch):
    """The default session starts a server per task, sends the hidden warm-up, and redacts the run's audit file."""
    key = "AIza" + "c8Lm3" * 7
    monkeypatch.setenv("LLM_GATEWAY_CRED", key)
    cfg = {"models": {"fake": {"adapter": "scripted", "model": "fake-model", "local": True,
                               "api_key_env": "LLM_GATEWAY_CRED"}}}
    args = json.loads(json.dumps(canonical_args(task_set, "AB-01")))
    args["wing"]["name"] = "wing " + key
    first_request = {}

    def first(msgs, tools):
        first_request["roles"] = [m["role"] for m in msgs]
        return call_step("ventorum_wing_analysis", args)

    def answer(msgs, tools):
        m = last_payload(msgs)["metrics"]
        return {"text": final("answered", {"CL": m["CL"], "CDi": m["CDi"]})}

    adapter = ScriptedAdapter(cfg["models"]["fake"], [first, answer])
    run_dir = run_benchmark(cfg, "fake", "focused", ["AB-01"], 1, tmp_path, run_id="default-session",
                            task_set=task_set, adapter=adapter, check_other_servers=False)
    assert first_request["roles"] == ["system", "user"]
    events = read_jsonl(run_dir / "events.jsonl")
    warm = [e for e in events if e["type"] == "warmup"]
    assert len(warm) == 1 and warm[0]["status"] == "success"
    res = read_jsonl(run_dir / "results.jsonl")
    assert res[0]["success"] is True, res[0]["error_kinds"]
    audit = run_dir / "audit" / "AB-01_r1.jsonl"
    text = audit.read_text(encoding="utf-8")
    assert text.strip() and "ventorum_wing_analysis" in text
    assert key not in text and "<redacted-key>" in text


# ---------------------------------------------------------------------------
# Harness rules: repeated successful call, tool-call parse failure
# ---------------------------------------------------------------------------

def test_repeat_successful_call_stops_the_task(server, task_set, tmp_path):
    """An identical repeat of a successful call is not run: the task stops with loop:repeat_call."""
    args = canonical_args(task_set, "AB-01")
    # The same call again, with the keys in another order and as a JSON string (equal after normalisation).
    repeat = json.dumps(dict(reversed(list(args.items()))))
    steps = [call_step("ventorum_wing_analysis", args), call_step("ventorum_wing_analysis", repeat),
             {"text": final("refused", {}, [])}]
    run_dir, res, ad = run_fake(server, task_set, tmp_path, ["AB-01"], steps)
    assert ad.n_requests == 2
    events = [e for e in read_jsonl(run_dir / "events.jsonl") if e["type"] == "tool_call"]
    ran = [e for e in events if not e.get("harness_error")]
    assert len(ran) == 1 and ran[0]["status"] == "success"
    assert [e["harness_error"] for e in events if e.get("harness_error")] == ["repeat_call"]
    rec = res[0]
    assert "loop:repeat_call" in rec["error_kinds"]
    assert rec["excluded"] is False and rec["success"] is False
    assert report(run_dir)["n_excluded"] == 0


def test_repeat_of_failed_call_is_run(server, task_set, tmp_path):
    """A repeat of a call whose result was an error is run again (no loop:repeat_call)."""
    _, res, ad = run_fake(server, task_set, tmp_path, ["AB-01"], [BAD_CALL, BAD_CALL, {"text": final("refused", {})}])
    assert ad.n_requests == 3
    assert "loop:repeat_call" not in res[0]["error_kinds"]
    assert res[0]["n_tool_calls"] == 2 and res[0]["n_harness_refused"] == 0


class _StubEndpoint:
    """A local HTTP stub of a chat-completions endpoint that always returns one error status."""

    def __init__(self, status: int, body: str):
        import http.server
        import threading

        stub = self
        self.status = status
        self.body = body.encode("utf-8")
        self.n_requests = 0

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - name from the standard library
                n = int(self.headers.get("Content-Length", 0) or 0)
                self.rfile.read(n)
                stub.n_requests += 1
                self.send_response(stub.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(stub.body)))
                self.end_headers()
                self.wfile.write(stub.body)

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


PARSE_500 = json.dumps({"error": {"message": "error parsing tool call: raw='{\"wing\": {', err=unexpected end of "
                                             "JSON input"}})
SYNTAX_500 = json.dumps({"error": "Value looks like object, but can't find closing '}' symbol: Syntax Error"})


@pytest.fixture
def no_proxy(monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


@pytest.mark.parametrize("body", [PARSE_500, SYNTAX_500])
def test_adapter_tool_call_parse_error_not_retried(no_proxy, body):
    """An endpoint error that shows a tool-call parse failure raises ToolCallParseError after one request."""
    from validation.agent_benchmark.adapters import ToolCallParseError, make_adapter

    stub = _StubEndpoint(500, body)
    try:
        ad = make_adapter({"adapter": "openai_chat", "base_url": stub.url, "model": "m", "retries": 2})
        with pytest.raises(ToolCallParseError):
            ad.complete([{"role": "user", "content": "x"}], [], timeout_s=10.0)
    finally:
        stub.close()
    assert stub.n_requests == 1


def _stub_run(server, task_set, tmp_path, stub, run_id):
    from validation.agent_benchmark.adapters import make_adapter

    cfg = {"models": {"stub": {"adapter": "openai_chat", "base_url": stub.url, "model": "m", "local": True,
                               "retries": 1}}}
    adapter = make_adapter(cfg["models"]["stub"])
    run_dir = run_benchmark(cfg, "stub", "focused", ["AB-01"], 1, tmp_path, run_id=run_id, task_set=task_set,
                            adapter=adapter, session_factory=lambda audit, cdir: Session(server, owned=False),
                            warmup=False, reference_call=mcp_call(server), check_other_servers=False)
    return run_dir, read_jsonl(run_dir / "results.jsonl")


def test_endpoint_tool_call_parse_failure_is_format_error(server, task_set, tmp_path, no_proxy):
    """HTTP 500 with a tool-call parse message: no retry, format_error (tool_call_parse), not excluded."""
    stub = _StubEndpoint(500, PARSE_500)
    try:
        run_dir, res = _stub_run(server, task_set, tmp_path, stub, "parse500")
    finally:
        stub.close()
    assert stub.n_requests == 1
    rec = res[0]
    assert "format_error" in rec["error_kinds"] and "format_error:tool_call_parse" in rec["error_kinds"]
    assert "endpoint_error" not in rec["error_kinds"]
    assert rec["excluded"] is False and rec["success"] is False
    ev = [e for e in read_jsonl(run_dir / "events.jsonl") if e["type"] == "tool_call_parse_error"]
    assert len(ev) == 1 and "parsing tool call" in ev[0]["message"]
    # A resume does not run the task again: the record is a model failure, not an endpoint failure.
    stub2 = _StubEndpoint(500, PARSE_500)
    try:
        _, res2 = _stub_run(server, task_set, tmp_path, stub2, "parse500")
    finally:
        stub2.close()
    assert stub2.n_requests == 0 and len(res2) == 1


def test_endpoint_503_still_retried_and_excluded(server, task_set, tmp_path, no_proxy):
    """HTTP 503 (no parse message) keeps the retry and the exclusion rules."""
    stub = _StubEndpoint(503, json.dumps({"error": "service unavailable"}))
    try:
        _, res = _stub_run(server, task_set, tmp_path, stub, "e503")
    finally:
        stub.close()
    assert stub.n_requests == 2  # one request and one retry
    rec = res[0]
    assert "endpoint_error" in rec["error_kinds"] and "format_error:tool_call_parse" not in rec["error_kinds"]
    assert rec["excluded"] is True
