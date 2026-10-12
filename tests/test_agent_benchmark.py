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
    cmd = build_command({"exe": "client", "command": ["{exe}", "-p", "{prompt}", "--mcp-config", "{mcp_config}"]},
                        "m", "hello", path, tmp_path)
    assert cmd == ["client", "-p", "hello", "--mcp-config", str(path)]


# ---------------------------------------------------------------------------
# Regression tests of the review findings
# ---------------------------------------------------------------------------

def _score(task_set, tid, final_text, calls, references=None):
    from validation.agent_benchmark.scoring import score_run

    v = task_set.variant(tid, 0)
    return score_run(v, references if references is not None else compute_references(v, lambda t, a: {}),
                     final_text, calls)


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
