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

from validation.speed import base, cases, case_spec, runner
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


def test_vspaero_adapter_present_runs_with_stubs(tmp_path, monkeypatch):
    """With stubbed subprocess calls a present VSPAERO runs, not skips."""
    exe = tmp_path / "vspaero.exe"
    exe.write_text("x", encoding="utf-8")
    (tmp_path / "vsp.exe").write_text("x", encoding="utf-8")
    (tmp_path / "vsp").write_text("x", encoding="utf-8")
    monkeypatch.setenv("VENTORUM_VSP_PYTHON", str(tmp_path / "vsp_python.exe"))
    Path(tmp_path / "vsp_python.exe").write_text("x", encoding="utf-8")
    polar = (
        "banner one\nbanner two\n"
        "      CDi             AoA             CLtot\n"
        "  0.004493481238  -4.000000000000  -0.330047000806\n"
        "  0.002242413557   3.000000000000   0.234493731993\n"
    )
    echo = (
        "Sref = 8\nCref = 1\nBref = 8\nMach = 0\n"
        "AoA = -4, 3\nSymmetry = 0\nWakeIters = 5\nVinf = 50\n"
    )

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, cwd, timeout):
        cwd = Path(cwd)
        if "-script" in cmd:
            for vsp3 in cwd.glob("*.vsp3"):
                (cwd / f"{vsp3.stem}.polar").write_text(polar, encoding="utf-8")
                (cwd / f"{vsp3.stem}.vspaero").write_text(echo, encoding="utf-8")
        else:
            out = cmd[cmd.index("--out") + 1]
            info = cmd[cmd.index("--info") + 1]
            Path(out).write_text("vsp3", encoding="utf-8")
            Path(info).write_text('{"tool_version": "OpenVSP 9.9.9-fake"}', encoding="utf-8")
        return _Done()

    monkeypatch.setattr(vspaero, "_run_capture", fake_run)
    work = tmp_path / "work"
    res = vspaero.run_sweep(str(exe), cases.build_rect(), 20, [-4.0, 3.0], work, 10.0)
    assert res.ok and not res.skipped
    assert res.CL == [pytest.approx(-0.330047000806), pytest.approx(0.234493731993)]
    assert res.CDi == [pytest.approx(0.004493481238), pytest.approx(0.002242413557)]
    assert "9.9.9-fake" in res.mesh_label
    assert (work / "case.json").is_file()


def test_vspaero_adapter_missing_pieces_skip(tmp_path, monkeypatch):
    """A missing vsp sibling or Python interpreter skips with its name."""
    exe = tmp_path / "vspaero.exe"
    exe.write_text("x", encoding="utf-8")
    monkeypatch.delenv("VENTORUM_VSP_PYTHON", raising=False)
    res = vspaero.run_sweep(str(exe), cases.build_rect(), 20, [4.0], None, 10.0)
    assert res.skipped and "vsp" in (res.reason or "").lower()
    (tmp_path / "vsp.exe").write_text("x", encoding="utf-8")
    (tmp_path / "vsp").write_text("x", encoding="utf-8")
    res2 = vspaero.run_sweep(str(exe), cases.build_rect(), 20, [4.0], None, 10.0)
    assert res2.skipped and "VENTORUM_VSP_PYTHON" in (res2.reason or "")


def test_vspaero_verdict_rejects_wrong_sref(tmp_path, monkeypatch):
    """A settings echo with the wrong Sref fails the run, never skips."""
    exe = tmp_path / "vspaero.exe"
    exe.write_text("x", encoding="utf-8")
    (tmp_path / "vsp.exe").write_text("x", encoding="utf-8")
    (tmp_path / "vsp").write_text("x", encoding="utf-8")
    monkeypatch.setenv("VENTORUM_VSP_PYTHON", str(tmp_path / "vsp_python.exe"))
    Path(tmp_path / "vsp_python.exe").write_text("x", encoding="utf-8")

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, cwd, timeout):
        cwd = Path(cwd)
        if "-script" in cmd:
            for vsp3 in cwd.glob("*.vsp3"):
                (cwd / f"{vsp3.stem}.polar").write_text(
                    "h\n      AoA             CLtot             CDi\n"
                    "  4.000000000000   0.300000000000   0.003000000000\n",
                    encoding="utf-8",
                )
                (cwd / f"{vsp3.stem}.vspaero").write_text(
                    "Sref = 100\nMach = 0\nAoA = 4\nSymmetry = 0\n",
                    encoding="utf-8",
                )
        else:
            out = cmd[cmd.index("--out") + 1]
            info = cmd[cmd.index("--info") + 1]
            Path(out).write_text("vsp3", encoding="utf-8")
            Path(info).write_text('{"tool_version": "OpenVSP 9.9.9-fake"}', encoding="utf-8")
        return _Done()

    monkeypatch.setattr(vspaero, "_run_capture", fake_run)
    res = vspaero.run_sweep(str(exe), cases.build_rect(), 20, [4.0], tmp_path / "w", 10.0)
    assert not res.ok and not res.skipped
    assert "Sref" in (res.reason or "")


def test_vspaero_analysis_script_uses_documented_inputs():
    """The sweep script sets the documented inputs, X-Z symmetry and the wake iterations (owner choice)."""
    text = vspaero.analysis_script_text("C:/work/m.vsp3", [-4.0, 3.0, 10.0], 8.0, 8.0, 1.0, 12)
    for marker in ("VSPAEROComputeGeometry", "ThinGeomSet", "VSPAEROSweep",
                   "AlphaStart", "AlphaNpts", "MachStart", "Sref", "NCPU", "Symmetry"):
        assert marker in text
    assert f"array<int> wakeit = {{{vspaero.WAKE_ITERS}}};" in text and "WakeNumIter" in text
    assert "array<int> sym = {1};" in text
    assert "RefFlag" not in text and "WingID" not in text
    assert text.startswith("void main()")
    assert text.rstrip().endswith("}")


def test_vspaero_jobs_split_angles():
    """Uniform angles run once; other lists run one point per angle."""
    assert vspaero._jobs([-4.0, 3.0, 10.0]) == [[-4.0, 3.0, 10.0]]
    assert vspaero._jobs([-4.0, 4.0, 10.0]) == [[-4.0], [4.0], [10.0]]
    assert vspaero._jobs([5.0]) == [[5.0]]


def test_vspaero_polar_parser_reads_names():
    """Column order does not matter; bad files raise ValueError."""
    text = (
        "banner\n"
        "      CDi             AoA             CLtot\n"
        "  0.004493481238  -4.000000000000  -0.330047000806\n"
        "  0.002242413557   3.000000000000   0.234493731993\n"
    )
    table = vspaero.parse_polar(text)
    assert table[-4.0] == (pytest.approx(-0.330047000806), pytest.approx(0.004493481238))
    assert table[3.0] == (pytest.approx(0.234493731993), pytest.approx(0.002242413557))
    with pytest.raises(ValueError, match="header"):
        vspaero.parse_polar("      AoA             CLtot\n  4.0 0.3\n")
    with pytest.raises(ValueError, match="no data"):
        vspaero.parse_polar("      AoA             CLtot             CDi\n")


def test_vspaero_echo_parser_reads_settings():
    """The echo parser reads Sref, Mach, angles, symmetry and wake count."""
    settings = vspaero.parse_vspaero_echo(
        "Sref = 8\nMach = 0\nAoA = -4, 3, 10\nSymmetry = 0\nWakeIters = 5\nVinf = 50\n"
    )
    assert settings["Sref"] == pytest.approx(8.0)
    assert settings["AoA"] == [-4.0, 3.0, 10.0]
    assert settings["Symmetry"] == 0
    assert settings["WakeIters"] == 5
    with pytest.raises(ValueError, match="Sref"):
        vspaero.parse_vspaero_echo("Mach = 0\n")


def test_case_spec_rect_maps_geometry():
    """The rect wing spec holds reference values and resolved stations."""
    spec = case_spec.aircraft_spec(cases.build_rect(), 20, [-4.0, 4.0])
    assert spec["s_ref_m2"] == pytest.approx(8.0)
    assert spec["b_ref_m"] == pytest.approx(8.0)
    assert spec["angles_deg"] == [-4.0, 4.0]
    (surf,) = spec["surfaces"]
    assert surf["semi_span_m"] == pytest.approx(4.0)
    assert [s["chord_m"] for s in surf["stations"]] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert surf["sweep_resolved_deg"] == pytest.approx(0.0)
    assert case_spec.scaled_counts(spec, 20) == {surf["name"]: 20}
    assert case_spec.is_uniform_angles([-4.0, 3.0, 10.0]) is True
    assert case_spec.is_uniform_angles([-4.0, 4.0, 10.0]) is False


def test_case_spec_rejects_many_sections():
    """A three-section surface fails with a clear reason, never silently."""
    import ventorum as vt

    surf = vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(0.5, 0.8), vt.WingSection(1.0, 0.6)],
    )
    ac = vt.Aircraft(name="three", surfaces=[surf])
    with pytest.raises(ValueError, match="two-section"):
        case_spec.aircraft_spec(ac, 20, [0.0])


def test_case_spec_tail_scales_by_span():
    """The tail tessellation scales with semi-span at equal density."""
    spec = case_spec.aircraft_spec(cases.build_wing_tail(), 20, [0.0])
    counts = case_spec.scaled_counts(spec, 20)
    assert counts["Wing"] == 20
    assert counts["HTail"] == 8


def _canned_payload(cl=(0.3, 0.4), cdi=(0.003, 0.004)):
    """Return a fake case-script payload with version and solve time."""
    return {"CL": list(cl), "CDi": list(cdi),
            "tool_version": "9.9.9-fake", "elapsed_solve_s": 0.5, "import_s": 1.0}


def test_asb_runs_with_fake_tool(tmp_path, monkeypatch):
    """A stubbed case script still drives the full adapter path."""
    seen = {}

    def fake_run(exe, script, workdir, timeout):
        seen["script"] = Path(script).name
        assert (Path(workdir) / "case.json").is_file()
        return _canned_payload()

    monkeypatch.setattr(asbox, "run_json_script", fake_run)
    work = tmp_path / "work"
    fake_exe = tmp_path / "py.exe"
    fake_exe.write_text("x", encoding="utf-8")
    res = asbox.run_sweep(str(fake_exe), cases.build_rect(), 20,
                          [-4.0, 4.0], work, 10.0)
    assert seen["script"] == "run_aerosandbox_case.py"
    assert res.ok and not res.skipped
    assert res.CL == [pytest.approx(0.3), pytest.approx(0.4)]
    assert res.CDi == [pytest.approx(0.003), pytest.approx(0.004)]
    assert "9.9.9-fake" in res.mesh_label
    assert res.elapsed_s == pytest.approx(0.5)


def test_mx_runs_with_fake_tool(tmp_path, monkeypatch):
    """A stubbed case script still drives the full adapter path."""
    seen = {}

    def fake_run(exe, script, workdir, timeout):
        seen["script"] = Path(script).name
        assert (Path(workdir) / "case.json").is_file()
        return _canned_payload((0.5,), (0.01,))

    monkeypatch.setattr(machup, "run_json_script", fake_run)
    fake_exe = tmp_path / "py.exe"
    fake_exe.write_text("x", encoding="utf-8")
    res = machup.run_sweep(str(fake_exe), cases.build_rect(), 20,
                           [4.0], tmp_path / "work", 10.0)
    assert seen["script"] == "run_machupx_case.py"
    assert res.ok and not res.skipped
    assert res.CL == [pytest.approx(0.5)]
    with pytest.raises(ValueError, match="points"):
        case_spec.check_result_payload(_canned_payload((0.5,), (0.01,)), [4.0, 5.0])


def test_helper_scripts_hold_documented_calls():
    """Each helper script compiles and names its documented API calls."""
    folder = Path(__file__).resolve().parents[1] / "validation" / "speed"
    checks = {
        "vsp_make_geom.py": ("SectTess_U", "Sym_Planar_Flag", "Sweep_Location", "GetVSPVersion"),
        "run_aerosandbox_case.py": ("VortexLatticeMethod", "spanwise_resolution",
                                    "chordwise_resolution", "OperatingPoint", "result.json"),
        "run_machupx_case.py": ("Scene", "solve_forces", "cosine_cluster", "result.json"),
    }
    for name, markers in checks.items():
        text = (folder / name).read_text(encoding="utf-8")
        compile(text, name, "exec")
        for marker in markers:
            assert marker in text, f"{name} misses {marker}"
    assert '"linear"' in (folder / "run_machupx_case.py").read_text(encoding="utf-8")


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


def test_vspaero_half_model_when_symmetric(tmp_path, monkeypatch):
    """With X-Z symmetry the geometry builder makes right halves only (no planar symmetry)."""
    import sys
    import types

    from validation.speed import vsp_make_geom

    calls = []
    fake = types.SimpleNamespace(
        ClearVSPModel=lambda: None, AddGeom=lambda *a: "W", SetGeomName=lambda *a: None,
        SetParmVal=lambda gid, name, group, val: calls.append((name, group, val)),
        Update=lambda: None, WriteVSPFile=lambda *a: None, SET_ALL=0)
    monkeypatch.setitem(sys.modules, "openvsp", fake)
    surf = {"name": "W", "symmetric": True, "mirror_y": False, "semi_span_m": 4.0, "sweep_resolved_deg": 0.0,
            "dihedral_deg": 0.0, "tess_u": 20, "position_m": [0.0, 0.0, 0.0],
            "stations": [{"chord_m": 1.0, "twist_deg": 0.0}, {"chord_m": 1.0, "twist_deg": 0.0}]}
    for half, expected in ((True, 0.0), (False, 2.0)):
        calls.clear()
        vsp_make_geom.build_case({"surfaces": [dict(surf)], "half_model": half}, str(tmp_path / "m.vsp3"))
        assert ("Sym_Planar_Flag", "Sym", expected) in calls
