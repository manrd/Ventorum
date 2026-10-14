# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the cross-tool speed harness.

All tests run without the real tools. Fake tools are small Python
scripts that write a JSON result file. Missing tools give skipped
results with a note, never a crash.
"""

import json
import sys
from pathlib import Path

import pytest

from validation.speed import base, cases, runner
from validation.speed import adapter_aerosandbox as asbox
from validation.speed import adapter_avl as avl_adapter
from validation.speed import adapter_flow5 as flow5
from validation.speed import adapter_machupx as machup
from validation.speed import adapter_ventorum as ventorum_adapter
from validation.speed import adapter_vspaero as vspaero


def _write_fake_tool(path: Path, cl: float = 0.5, cdi: float = 0.01) -> Path:
    """Write a fake tool script that writes result.json with fixed values."""
    script = path / "fake_tool.py"
    script.write_text(
        "import json\n"
        "from pathlib import Path\n"
        f"Path('result.json').write_text(json.dumps({{'CL': [{cl}], 'CDi': [{cdi}]}}))\n",
        encoding="utf-8",
    )
    return script


def test_median_of_samples():
    """Median of odd and even sample lists."""
    assert base.median_seconds([1.0, 3.0, 2.0]) == pytest.approx(2.0)
    assert base.median_seconds([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)


def test_max_relative_change_uses_peak_scale():
    """A zero crossing does not blow up the change metric."""
    coarse = [0.0, 0.5, 1.0]
    fine = [0.001, 0.505, 1.005]
    change = base.max_relative_change(coarse, fine)
    assert change == pytest.approx(0.005 / 1.005, rel=0.01)


def test_max_relative_change_needs_values():
    """Empty or all-None sweeps give None."""
    assert base.max_relative_change([], []) is None
    assert base.max_relative_change([None], [None]) is None


def test_has_converged_threshold():
    """Changes below 0.5 % pass, larger changes fail."""
    assert base.has_converged([1.0], [0.1], [1.004], [0.1004])[0] is True
    ok, cl_err, cdi_err = base.has_converged([1.0], [0.1], [1.006], [0.1])
    assert ok is False
    assert cl_err == pytest.approx(0.006 / 1.006, rel=0.01)
    assert cdi_err == pytest.approx(0.0)


def test_find_tool_missing_gives_none(monkeypatch):
    """No file at the explicit path or the variable gives None."""
    monkeypatch.delenv("VENTORUM_SPEED_TEST_EXE", raising=False)
    assert base.find_tool(None, ("VENTORUM_SPEED_TEST_EXE",)) is None
    assert base.find_tool("/no/such/file.exe", ("VENTORUM_SPEED_TEST_EXE",)) is None


def test_find_tool_uses_env_file(tmp_path, monkeypatch):
    """An existing file in the variable is found."""
    exe = tmp_path / "tool.exe"
    exe.write_text("x", encoding="utf-8")
    monkeypatch.setenv("VENTORUM_SPEED_TEST_EXE", str(exe))
    assert base.find_tool(None, ("VENTORUM_SPEED_TEST_EXE",)) == exe


def test_refuse_inside_repository(tmp_path):
    """An output folder inside the repository is refused."""
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="inside the repository"):
        base.refuse_inside_repository(root / "tmp_speed_refuse", root)
    outside = tmp_path / "speed_out"
    assert base.refuse_inside_repository(outside, root) == outside.resolve()


def test_scrub_home_removes_user_path():
    """The home folder path is replaced in error text."""
    home = str(Path.home())
    assert home not in base.scrub_home(f"bad file {home}/tool.exe")


def test_cases_full_and_quick():
    """Full run has 4 cases and 33 angles; quick has 1 case and 3 angles."""
    assert len(cases.case_list(quick=False)) == 4
    assert len(cases.case_list(quick=True)) == 1
    assert len(cases.sweep_angles(quick=False)) == 33
    assert len(cases.sweep_angles(quick=True)) == 3
    ids = [c["id"] for c in cases.case_list(quick=False)]
    assert ids == ["rect", "tapered", "swept", "wingtail"]


def test_case_builders_make_aircraft():
    """Each builder returns an aircraft with reference values."""
    for spec in cases.case_list(quick=False):
        ac = spec["build"]()
        ac.compute_reference_values()
        assert ac.S_ref > 0.0


def test_run_json_script_with_fake_tool(tmp_path):
    """A fake tool script runs in a subprocess and its JSON is read."""
    script = _write_fake_tool(tmp_path)
    out = base.run_json_script(sys.executable, script, tmp_path / "work", 60.0)
    assert out == {"CL": [0.5], "CDi": [0.01]}


def test_run_json_script_missing_result_fails(tmp_path):
    """A script that writes no result file gives an error, not zeros."""
    script = tmp_path / "silent.py"
    script.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="result.json"):
        base.run_json_script(sys.executable, script, tmp_path / "work", 60.0)


def test_avl_adapter_skips_when_missing(monkeypatch):
    """Without the AVL program the adapter reports skipped."""
    monkeypatch.delenv("VENTORUM_AVL_EXE", raising=False)
    assert avl_adapter.find_tool(None) is None
    assert avl_adapter.is_available(None) is False
    ac = cases.build_rect()
    res = avl_adapter.run_sweep(None, ac, 20, [0.0, 4.0], None, 10.0)
    assert res.skipped and not res.ok
    assert "VENTORUM_AVL_EXE" in (res.reason or "")


def test_vspaero_adapter_skips_when_missing(monkeypatch):
    """Without VSPAERO the adapter reports skipped with the variable name."""
    monkeypatch.delenv("VENTORUM_VSPAERO", raising=False)
    monkeypatch.delenv("VENTORUM_VSP_PYTHON", raising=False)
    assert vspaero.is_available(None) is False
    res = vspaero.run_sweep(None, cases.build_rect(), 20, [4.0], None, 10.0)
    assert res.skipped and not res.ok
    assert "VENTORUM_VSPAERO" in (res.reason or "")


def test_vspaero_adapter_present_stays_honest(tmp_path, monkeypatch):
    """A present VSPAERO without a checked mapping stays skipped, not timed."""
    exe = tmp_path / "vspaero.exe"
    exe.write_text("x", encoding="utf-8")
    monkeypatch.setenv("VENTORUM_VSPAERO", str(exe))
    assert vspaero.is_available(str(exe)) is True
    res = vspaero.run_sweep(str(exe), cases.build_rect(), 20, [4.0], None, 10.0)
    assert res.skipped and not res.ok
    assert "mapping" in (res.reason or "").lower()


def test_flow5_reports_no_batch_mode(tmp_path, monkeypatch):
    """Flow5 always skips: no batch mode is documented in the harness."""
    monkeypatch.delenv("VENTORUM_FLOW5", raising=False)
    res = flow5.run_sweep(None, cases.build_rect(), 20, [4.0], None, 10.0)
    assert res.skipped
    assert "VENTORUM_FLOW5" in (res.reason or "")
    exe = tmp_path / "flow5.exe"
    exe.write_text("x", encoding="utf-8")
    res2 = flow5.run_sweep(str(exe), cases.build_rect(), 20, [4.0], None, 10.0)
    assert res2.skipped
    assert "batch" in (res2.reason or "").lower()


def test_python_tool_adapters_skip_when_missing(monkeypatch):
    """AeroSandbox and MachUpX skip with the interpreter variable name."""
    monkeypatch.delenv("VENTORUM_AEROSANDBOX_PYTHON", raising=False)
    monkeypatch.delenv("VENTORUM_MACHUPX_PYTHON", raising=False)
    assert asbox.is_available(None) is False
    assert machup.is_available(None) is False
    r1 = asbox.run_sweep(None, cases.build_rect(), 20, [4.0], None, 10.0)
    r2 = machup.run_sweep(None, cases.build_rect(), 20, [4.0], None, 10.0)
    assert r1.skipped and r2.skipped
    assert "VENTORUM_AEROSANDBOX_PYTHON" in (r1.reason or "")
    assert "VENTORUM_MACHUPX_PYTHON" in (r2.reason or "")


def test_ventorum_adapter_runs_tiny_sweep():
    """Ventorum solves a tiny sweep and returns CL and CDi per angle."""
    ac = cases.build_rect()
    res = ventorum_adapter.run_sweep(ac, 8, [-2.0, 2.0], None, 120.0)
    assert res.ok and not res.skipped
    assert len(res.CL) == 2 and len(res.CDi) == 2
    assert res.CL[1] > res.CL[0]
    assert all(v is not None and v >= 0.0 for v in res.CDi)
    assert res.elapsed_s >= 0.0


def test_ventorum_time_sweeps_warms_up():
    """One untimed warm-up sweep runs before the timed repeats."""
    calls = []

    def fake_run(ac, mesh, angles, workdir, timeout):
        calls.append(1)
        return base.SweepResult(
            tool="ventorum", case_id="c", mesh_label="m", mesh=mesh,
            angles_deg=list(angles), CL=[0.1] * len(angles),
            CDi=[0.01] * len(angles), ok=True, skipped=False,
            reason=None, elapsed_s=0.01 * len(calls),
        )

    import validation.speed.adapter_ventorum as vad

    real = vad.run_sweep
    vad.run_sweep = fake_run
    try:
        median, samples = vad.time_sweeps(cases.build_rect(), 8, [0.0], 2)
    finally:
        vad.run_sweep = real
    assert len(calls) == 3
    assert samples == [0.02, 0.03]
    assert median == pytest.approx(0.025)


class _StubAdapter:
    """In-memory adapter with scripted sweeps for runner tests."""

    TOOL_NAME = "stub"
    SUPPORTS_GROUND_EFFECT = False
    MESHES = [10, 20]

    def __init__(self, values: dict[int, tuple[list[float], list[float]]]):
        self.values = values
        self.calls: list[int] = []

    def is_available(self, explicit=None):
        return True

    def describe_mesh(self, mesh):
        return f"mesh {mesh}"

    def run_sweep(self, aircraft, mesh, angles_deg, workdir, timeout):
        self.calls.append(mesh)
        cl, cdi = self.values[mesh]
        return base.SweepResult(
            tool="stub", case_id="c", mesh_label=f"mesh {mesh}", mesh=mesh,
            angles_deg=list(angles_deg), CL=list(cl), CDi=list(cdi),
            ok=True, skipped=False, reason=None, elapsed_s=0.05,
        )

    def time_sweeps(self, aircraft, mesh, angles_deg, repeats, timeout=None,
                    workdir=None):
        return 0.05, [0.05] * repeats


def test_runner_picks_converged_mesh(tmp_path):
    """The runner selects the first mesh where both changes are below 0.5 %."""
    stub = _StubAdapter({10: ([1.0], [0.1]), 20: ([1.001], [0.1001])})
    record = runner.converge_tool(stub, cases.build_rect(), [4.0], tmp_path, 10.0)
    assert record["mesh"] == 20
    assert record["converged"] is True
    assert stub.calls == [10, 20]


def test_runner_keeps_finest_when_not_converged(tmp_path):
    """Without convergence the runner keeps the finest mesh with a note."""
    stub = _StubAdapter({10: ([1.0], [0.1]), 20: ([1.5], [0.2])})
    record = runner.converge_tool(stub, cases.build_rect(), [4.0], tmp_path, 10.0)
    assert record["mesh"] == 20
    assert record["converged"] is False
    assert "not converged" in (record["note"] or "").lower()


def test_runner_quick_writes_md_and_json(tmp_path, monkeypatch):
    """A quick run with one stub tool writes Markdown and JSON outside the repo."""
    monkeypatch.setitem(runner.ADAPTERS, "stub",
                        _StubAdapter({10: ([0.4], [0.01]), 20: ([0.401], [0.01001])}))
    out = tmp_path / "speed_out"
    code = runner.main(["--out", str(out), "--quick", "--tools", "stub"])
    assert code == 0
    md = out / "speed_comparison.md"
    js = out / "speed_comparison.json"
    assert md.is_file() and js.is_file()
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["meta"]["repeats"] == 1
    assert payload["cases"][0]["tools"]["stub"]["mesh"] == 20
    text = md.read_text(encoding="utf-8")
    assert "Equal accuracy" in text
    assert "median" in text.lower()


def test_runner_skips_missing_tool(tmp_path, monkeypatch):
    """A missing tool gives a skipped entry with the reason, not a failure."""
    monkeypatch.delenv("VENTORUM_AVL_EXE", raising=False)
    out = tmp_path / "speed_out"
    code = runner.main(["--out", str(out), "--quick", "--tools", "avl"])
    assert code == 0
    payload = json.loads((out / "speed_comparison.json").read_text(encoding="utf-8"))
    entry = payload["cases"][0]["tools"]["avl"]
    assert entry["skipped"] is True
    assert "VENTORUM_AVL_EXE" in (entry["reason"] or "")


def test_runner_refuses_inside_repository():
    """An output folder inside the repository stops with code 2."""
    root = Path(__file__).resolve().parents[1]
    assert runner.main(["--out", str(root / "tmp_speed_refuse"), "--quick"]) == 2


def test_runner_unknown_tool_is_rejected(tmp_path):
    """An unknown tool name stops with code 2 and writes nothing."""
    out = tmp_path / "speed_out"
    assert runner.main(["--out", str(out), "--tools", "no_such_tool"]) == 2
    assert not out.exists()


def test_speed_files_hold_no_private_ids_or_em_dash():
    """The new harness files hold no planning IDs and no em dash."""
    import re

    private = re.compile(r"(?<!TN )\b(T-\d{4}|D-\d{2}|P[12]-\d{2})\b")
    folder = Path(__file__).resolve().parents[1] / "validation" / "speed"
    files = list(folder.glob("*.py")) + [folder / "README.md"]
    assert files, "speed harness files are missing"
    hits = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if "T-NNNN" in text or "P1-NN" in text:
            continue
        for n, line in enumerate(text.splitlines(), start=1):
            if "T-NNNN" in line or "P1-NN" in line:
                continue
            if private.search(line):
                hits.append(f"{path.name}:{n}")
        if "\u2014" in text:
            hits.append(f"{path.name}: em dash")
    assert not hits, f"Forbidden text in speed harness: {hits}"
    test_text = Path(__file__).read_text(encoding="utf-8")
    assert "\u2014" not in test_text
