# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests of the agent machine-capability and tuning tools (T-0017).
"""

from __future__ import annotations

import json

import pytest

from ventorum.agent import call_tool, handle_message
from ventorum.agent import tools as agent_tools
from ventorum.hardware import profile as prof


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTORUM_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("VENTORUM_DISABLE_AUTOTUNE", raising=False)
    prof.reset_profile_cache()
    yield tmp_path
    prof.reset_profile_cache()


def strict_json(payload):
    """The payload must be strict JSON (no NaN, no infinity)."""
    return json.loads(json.dumps(payload, allow_nan=False))


def assert_error(payload, kind, text=None):
    assert payload["status"] == "error", payload
    assert payload["error"]["type"] == kind, payload
    if text is not None:
        assert text in payload["error"]["message"], payload["error"]["message"]
    strict_json(payload)


def test_capabilities_without_profile(config_dir):
    p = call_tool("ventorum_machine_capabilities", {})
    strict_json(p)
    assert p["status"] == "success"
    assert p["tool"] == "ventorum_machine_capabilities"
    assert p["profile"]["status"] == "none"
    assert isinstance(p["advice"], str) and p["advice"]
    assert p["executive_summary"].startswith("[Ventorum RESULT]")
    assert "machine_guid" not in p["hardware"] and "node_name" not in p["hardware"]
    assert "numpy" in p["kernel_backends"]
    assert p["cython_threads"] in ("openmp", "python", None)


def test_capabilities_never_contain_the_machine_id(config_dir):
    from ventorum.hardware.detector import scan_hardware

    p = call_tool("ventorum_machine_capabilities", {})
    strict_json(p)
    assert p["status"] == "success"
    blob = json.dumps(p)
    assert "machine_guid" not in blob and "node_name" not in blob
    hw = scan_hardware(detect_gpu=False)
    assert hw.machine_guid and hw.machine_guid not in blob


def test_tune_tool_writes_a_profile_and_capabilities_see_it(config_dir, monkeypatch):
    from ventorum.hardware import tuner

    monkeypatch.setattr(tuner, "_CASES", {"small": (6, 2), "medium": (8, 2), "large": (10, 2)})
    monkeypatch.setattr(tuner, "_REPEATS", {"small": 1, "medium": 1, "large": 1})
    p = call_tool("ventorum_tune_machine", {"quick": True, "save": True})
    strict_json(p)
    assert p["status"] == "success", p
    assert (config_dir / prof.PROFILE_FILENAME).exists()
    assert set(p["kernels"]) == {"small", "medium"}
    assert p["tuning_seconds"] is not None and p["tuning_seconds"] >= 0
    c = call_tool("ventorum_machine_capabilities", {})
    assert c["status"] == "success"
    assert c["profile"]["status"] == "valid"
    assert c["profile"]["quick"] is True
    assert c["profile"]["created_utc"]


def test_second_tuning_call_is_refused(config_dir):
    agent_tools._TUNE_LOCK.acquire()
    try:
        p = call_tool("ventorum_tune_machine", {"quick": True, "save": False})
    finally:
        agent_tools._TUNE_LOCK.release()
    assert_error(p, "invalid_input", "a tuning run is in progress")


def test_tune_tool_refuses_non_boolean_inputs(config_dir):
    assert_error(call_tool("ventorum_tune_machine", {"quick": "yes"}), "invalid_input", "quick")
    assert_error(call_tool("ventorum_tune_machine", {"save": 1}), "invalid_input", "save")
    assert_error(call_tool("ventorum_tune_machine", {"speed": True}), "invalid_input", "speed")


def test_both_tools_are_listed_by_the_mcp_server():
    line = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    reply = json.loads(handle_message(line))
    names = [t["name"] for t in reply["result"]["tools"]]
    assert "ventorum_machine_capabilities" in names
    assert "ventorum_tune_machine" in names
    by_name = {t["name"]: t for t in reply["result"]["tools"]}
    assert by_name["ventorum_machine_capabilities"]["inputSchema"]["additionalProperties"] is False
    tune = by_name["ventorum_tune_machine"]
    assert tune["inputSchema"]["additionalProperties"] is False
    assert set(tune["inputSchema"]["properties"]) == {"quick", "save"}
    strict_json(reply)
