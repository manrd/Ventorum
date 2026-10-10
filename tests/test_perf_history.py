# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the speed history script ``validation/perf_history.py``.

The tests cover the record fields, the refusal of a history file inside the
repository, the percent change of the report, the import check of the compare
mode and the fixed workload names.
"""

from __future__ import annotations

import getpass
import json
import math
import platform
import subprocess
import sys
import types
from pathlib import Path

import pytest

from validation import perf_history as ph

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The fields that every record must hold (task T-0059, decision 4).
REQUIRED_FIELDS = {
    "date",
    "ventorum_version",
    "git_commit",
    "git_dirty",
    "python",
    "numpy",
    "cpu_cores",
    "gpu",
    "workloads",
    "note",
}


def _strings(obj) -> list[str]:
    """Return every string of a JSON object, keys and values included."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        out: list[str] = []
        for key, value in obj.items():
            out.extend(_strings(key))
            out.extend(_strings(value))
        return out
    if isinstance(obj, (list, tuple)):
        return [text for item in obj for text in _strings(item)]
    return []


def _forbidden_tokens() -> list[str]:
    """Return the machine and user tokens that a record must not contain."""
    tokens: list[str] = []
    node = platform.node()
    if node:
        tokens.append(node)
    user = getpass.getuser()
    if len(user) >= 4:
        tokens.append(user)
    home = str(Path.home())
    if home:
        tokens.append(home)
    return tokens


def _entry(unit: str, median: float) -> dict:
    """Return one workload entry of a hand-written record."""
    return {"unit": unit, "median": median, "min": 0.9 * median, "max": 1.1 * median}


def _record(date: str, workloads: dict) -> dict:
    """Return one hand-written record with every field of decision 4."""
    return {
        "date": date,
        "ventorum_version": "0.3.0",
        "git_commit": "0123456789abcdef",
        "git_dirty": False,
        "python": "3.12.0",
        "numpy": "2.0.0",
        "cpu_cores": 8,
        "gpu": None,
        "workloads": workloads,
        "note": "hand written",
    }


@pytest.mark.slow
def test_record_has_all_fields_and_no_identifiers(tmp_path):
    """One tiny run writes one JSON line with the fields of decision 4 and no identifier."""
    from ventorum import gpu

    history = tmp_path / "perf_history.jsonl"
    assert ph.main(["--history", str(history), "--note", "tiny test record", "--tiny"]) == 0

    lines = history.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert set(record.keys()) == REQUIRED_FIELDS
    assert isinstance(record["git_dirty"], bool)
    assert isinstance(record["cpu_cores"], int) and record["cpu_cores"] >= 1
    assert record["gpu"] is None or isinstance(record["gpu"], str)
    assert record["note"] == "tiny test record [tiny]"  # a tiny record is marked
    assert record["ventorum_version"] and record["python"] and record["numpy"]

    for name in ph.WORKLOAD_NAMES:
        assert name in record["workloads"]
    if gpu.available():
        for name in ph.SWEEP_WORKLOADS:
            assert name + "_gpu" in record["workloads"]
    for name, entry in record["workloads"].items():
        assert set(entry.keys()) == {"unit", "median", "min", "max"}
        assert entry["unit"] in ("ms", "cases/s")
        assert math.isfinite(entry["median"]) and entry["median"] > 0.0
        assert 0.0 < entry["min"] <= entry["median"] <= entry["max"], name

    for token in _forbidden_tokens():
        for text in _strings(record):
            assert token not in text, f"{token!r} found in the record"


def test_history_inside_repository_is_refused(tmp_path):
    """A history path inside the repository is refused and no file is written."""
    inside = [
        REPO_ROOT / "perf_history.jsonl",
        REPO_ROOT / "validation" / "perf_history.jsonl",
    ]
    for path in inside:
        with pytest.raises(SystemExit) as exc:
            ph.main(["--history", str(path), "--tiny"])
        assert "inside the repository" in str(exc.value)
        assert not path.exists()

    # A path that does not exist yet outside the repository passes the check.
    outside = ph.check_history_path(tmp_path / "perf_history.jsonl")
    assert outside == (tmp_path / "perf_history.jsonl").resolve()


def test_report_shows_change_between_records(tmp_path):
    """The report shows the percent change of each workload against the previous record."""
    history = tmp_path / "history.jsonl"
    first = _record(
        "2026-10-01T00:00:00",
        {
            "single_vlm_n20_c4": _entry("ms", 100.0),
            "sweep_vlm_n40_c4_33": _entry("cases/s", 200.0),
        },
    )
    second = _record(
        "2026-10-02T00:00:00",
        {
            "single_vlm_n20_c4": _entry("ms", 90.0),
            "sweep_vlm_n40_c4_33": _entry("cases/s", 250.0),
        },
    )
    history.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")

    report = ph.build_report(ph.read_history(history))
    assert report.columns == ["single_vlm_n20_c4", "sweep_vlm_n40_c4_33"]
    assert report.units == ["ms", "cases/s"]
    assert len(report.rows) == 2
    assert report.rows[0].commit == "0123456"  # short commit

    # Numbers are compared as numbers.
    assert report.rows[0].cells[0].value == pytest.approx(100.0)
    assert report.rows[0].cells[0].change_pct is None  # first record: no previous value
    assert report.rows[1].cells[0].value == pytest.approx(90.0)
    assert report.rows[1].cells[0].change_pct == pytest.approx(-10.0)
    assert report.rows[1].cells[1].change_pct == pytest.approx(25.0)

    text = ph.render_report(report)
    assert "(-10.0%)" in text
    assert "(+25.0%)" in text
    assert ph.TREND_LINE in text
    assert "A history row is a trend." in text


@pytest.mark.slow
def test_compare_checks_the_imported_tree(tmp_path):
    """Compare the repository with itself, and stop on a tree without ventorum."""
    result = ph.run_compare(
        REPO_ROOT, REPO_ROOT, rounds=1, workloads=["single_vlm_n20_c4"], tiny=True
    )
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.workload == "single_vlm_n20_c4"
    assert row.unit == "ms"
    assert math.isfinite(row.ratio) and row.ratio > 0.0
    assert math.isfinite(row.spread) and row.spread >= 0.0
    assert row.median_a > 0.0 and row.median_b > 0.0

    text = ph.render_compare(result)
    assert "single_vlm_n20_c4" in text
    assert "ratio B/A" in text

    fake = tmp_path / "tree_without_ventorum"
    fake.mkdir()
    with pytest.raises(SystemExit) as exc:
        ph.run_compare(REPO_ROOT, fake, rounds=1, workloads=["single_vlm_n20_c4"], tiny=True)
    assert "ventorum" in str(exc.value).lower()


def test_workload_names_are_fixed():
    """The workload names are exactly the names of the task card."""
    assert list(ph.WORKLOAD_NAMES) == [
        "single_vlm_n20_c4",
        "single_vlm_n80_c4",
        "single_linear_n80",
        "single_nonlinear_n80",
        "sweep_vlm_n40_c4_33",
        "sweep_linear_n80_33",
        "wingtail_vlm_n40_c4_33",
        "ground_vlm_n40_6x9",
    ]
    assert set(ph.SWEEP_WORKLOADS) <= set(ph.WORKLOAD_NAMES)


def test_record_times_the_ventorum_of_its_own_tree(tmp_path, monkeypatch):
    """Record mode stops when the timed ventorum is not the tree of the git commit.

    Regression test: record mode timed the first ``ventorum`` on the import
    path (often an editable install of another tree) and recorded the git
    commit of the tree that holds the script.
    """
    fake = tmp_path / "fake_tree"
    fake.mkdir()
    monkeypatch.setattr(ph, "REPO_ROOT", fake)
    history = tmp_path / "history.jsonl"
    with pytest.raises(SystemExit) as exc:
        ph.main(["--history", str(history), "--tiny"])
    message = str(exc.value)
    assert "ventorum" in message
    assert "not from the tree" in message or "cannot be imported" in message
    assert str(fake) in message
    assert not history.exists()


def test_tiny_record_is_marked_and_not_compared_with_full(tmp_path):
    """A tiny record has the mark in its note; the report compares records of the same kind only."""
    record = ph.make_record("smoke", {}, tree=tmp_path, tiny=True)
    assert record["note"] == "smoke [tiny]"
    assert ph.is_tiny_record(record)
    assert ph.make_record("", {}, tree=tmp_path, tiny=True)["note"] == "[tiny]"
    full = ph.make_record("smoke", {}, tree=tmp_path, tiny=False)
    assert full["note"] == "smoke"
    assert not ph.is_tiny_record(full)

    def with_note(date, median, note):
        rec = _record(date, {"single_vlm_n20_c4": _entry("ms", median)})
        rec["note"] = note
        return rec

    records = [
        with_note("2026-10-01T00:00:00", 100.0, "full one"),
        with_note("2026-10-02T00:00:00", 10.0, "tiny one [tiny]"),
        with_note("2026-10-03T00:00:00", 90.0, "full two"),
        with_note("2026-10-04T00:00:00", 12.0, "tiny two [tiny]"),
    ]
    report = ph.build_report(records)
    changes = [row.cells[0].change_pct for row in report.rows]
    assert changes[0] is None
    assert changes[1] is None  # first tiny record: no tiny record before it
    assert changes[2] == pytest.approx(-10.0)  # against the first full record
    assert changes[3] == pytest.approx(20.0)  # against the first tiny record
    assert [row.tiny for row in report.rows] == [False, True, False, True]

    text = ph.render_report(report)
    assert "0123456 [tiny]" in text
    assert "same kind" in text


def _fake_ventorum(monkeypatch, location):
    """Put a fake ``ventorum`` module with ``__file__`` = *location* in ``sys.modules``."""
    module = types.ModuleType("ventorum")
    module.__file__ = None if location is None else str(location)
    monkeypatch.setitem(sys.modules, "ventorum", module)
    return module


def test_import_check_requires_the_init_file_of_the_tree(tmp_path, monkeypatch):
    """The import check accepts only TREE/ventorum/__init__.py, not any file under TREE."""
    outer = tmp_path / "outer"
    nested = outer / "worktrees" / "nested"
    init = nested / "ventorum" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text("", encoding="utf-8")

    module = _fake_ventorum(monkeypatch, init)
    # A package of a nested worktree is under the outer tree, but it is not its package.
    with pytest.raises(SystemExit) as exc:
        ph._import_checked(outer)
    assert "not from the tree" in str(exc.value)
    assert ph._import_checked(nested) is module

    # A namespace package has no __file__.
    _fake_ventorum(monkeypatch, None)
    with pytest.raises(SystemExit) as exc:
        ph._import_checked(nested)
    assert "namespace package" in str(exc.value)


def test_compare_header_names_the_tree_folder_and_the_imported_file(tmp_path):
    """The header gives the folder name and the imported file relative to the tree, not the full path."""
    info = ph._probe(REPO_ROOT)
    assert info.file == "ventorum/__init__.py"

    tree_a = tmp_path / "tree_a"
    tree_b = tmp_path / "tree_b"
    result = ph.CompareResult(
        rounds=1,
        tiny=True,
        tree_a=ph.TreeInfo(tree=tree_a, version="0.3.0", cython=False, file="ventorum/__init__.py"),
        tree_b=ph.TreeInfo(tree=tree_b, version="0.3.1", cython=True, file="ventorum/__init__.py"),
        rows=[],
    )
    text = ph.render_compare(result)
    assert "Tree A: tree_a (imports ventorum/__init__.py, ventorum 0.3.0" in text
    assert "Tree B: tree_b (imports ventorum/__init__.py, ventorum 0.3.1" in text
    assert str(tmp_path) not in text


def test_compare_alternates_the_order_of_the_trees(tmp_path, monkeypatch):
    """Even rounds run A then B; odd rounds run B then A. The ratio stays B/A."""
    tree_a = tmp_path / "tree_a"
    tree_b = tmp_path / "tree_b"
    tree_a.mkdir()
    tree_b.mkdir()
    order: list[str] = []

    def fake_child(options, tree, timeout=None):
        tree = Path(tree)
        if "--probe" in options:
            return {"version": "0.3.0", "cython": True, "file": str(tree / "ventorum" / "__init__.py")}
        label = "A" if tree == tree_a.resolve() else "B"
        order.append(label)
        return {"unit": "ms", "median": 10.0 if label == "A" else 20.0, "min": 1.0, "max": 30.0}

    monkeypatch.setattr(ph, "_child", fake_child)
    result = ph.run_compare(tree_a, tree_b, rounds=4, workloads=["single_vlm_n20_c4"], tiny=True)
    assert order == ["A", "B", "B", "A", "A", "B", "B", "A"]
    row = result.rows[0]
    assert row.median_a == pytest.approx(10.0)
    assert row.median_b == pytest.approx(20.0)
    assert row.ratio == pytest.approx(2.0)
    assert row.spread == pytest.approx(0.0)


def test_measure_collects_garbage_after_the_warm_up(monkeypatch):
    """The garbage of the warm-up call is collected before the timed calls."""
    events: list[str] = []
    monkeypatch.setattr(ph.gc, "collect", lambda *args: events.append("gc") or 0)
    workload = ph.Workload("probe", lambda: events.append("call"), "ms", 1)
    ph.measure(workload, tiny=True)
    assert events == ["call", "gc", "call"]
    events.clear()
    ph.measure(workload, tiny=False)
    assert events == ["call", "gc"] + ["call"] * ph.TIMED_CALLS


def test_git_status_takes_no_optional_locks(tmp_path, monkeypatch):
    """The dirty check does not take the optional git index lock."""
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="0123456789abcdef\n", stderr="")

    monkeypatch.setattr(ph.subprocess, "run", fake_run)
    assert ph.git_state(tmp_path) == ("0123456789abcdef", True)
    assert ["git", "--no-optional-locks", "status", "--porcelain"] in commands


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
