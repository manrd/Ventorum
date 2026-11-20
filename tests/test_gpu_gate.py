# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests for validation/gpu_gate.py.

The script runs the GPU checks of the phase gate and writes a short
Markdown log. The tests never run the real GPU tests or examples: the
dry-run tests use ``--dry-run``, and the log tests run ``main()``
in-process with a fake step runner.
"""

import getpass
import importlib.util
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import ventorum.gpu as gpu_mod

SCRIPT = Path(__file__).resolve().parents[1] / "validation" / "gpu_gate.py"
REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((REPO_ROOT / "examples").glob("[0-9]*.py"))
EXAMPLE_NAMES = [f"examples/{path.name}" for path in EXAMPLES]


def _load():
    """Import validation/gpu_gate.py from its file path."""
    spec = importlib.util.spec_from_file_location("gpu_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_gate"] = module
    spec.loader.exec_module(module)
    return module


def _dry_run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", *args],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )


@pytest.fixture(autouse=True)
def _no_forced_path(monkeypatch):
    """Ignore a forcing variable of the shell; each test sets the GPU state it needs."""
    monkeypatch.delenv("VENTORUM_GPU_GATE_FORCE_NO_GPU", raising=False)


@pytest.fixture
def fake_gpu(monkeypatch):
    """Make the GPU look available, so the runner tests also pass on CI (no GPU)."""
    monkeypatch.setattr(gpu_mod, "available", lambda: True)
    monkeypatch.setattr(gpu_mod, "unavailable_reason", lambda: None)
    monkeypatch.setattr(gpu_mod, "info", lambda: {"name": "Test GPU"})


def _passing(step):
    """Fake step runner: every step passes."""
    mod = _load()
    return mod.StepResult(name=step.name, status="pass", seconds=0.1, last_line="ok")


def test_dry_run_lists_steps_in_order():
    proc = _dry_run()
    assert proc.returncode == 0
    assert EXAMPLES, "no example scripts found"
    text = proc.stdout
    pytest_pos = text.find("pytest")
    assert pytest_pos >= 0
    assert "-m gpu" in text[pytest_pos:]
    assert "VENTORUM_DEVICE=gpu" in text
    positions = []
    for name in EXAMPLE_NAMES:
        pos = text.find(name)
        assert pos >= 0, f"dry run does not list {name}"
        positions.append(pos)
    assert positions == sorted(positions), "examples are not in sorted order"
    assert pytest_pos < positions[0], "the pytest step is not first"


def test_dry_run_float64():
    proc = _dry_run("--float64")
    assert proc.returncode == 0
    text = proc.stdout
    assert "VENTORUM_GPU_PRECISION=float64" in text
    for name in EXAMPLE_NAMES:
        assert text.count(name) == 2, f"{name} has no second float64 pass"


def test_no_gpu_exit_code(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(gpu_mod, "available", lambda: False)
    monkeypatch.setattr(gpu_mod, "unavailable_reason", lambda: "Test GPU is missing.")
    log = tmp_path / "gpu_gate.md"
    assert mod.main([str(log)]) == 2
    text = log.read_text(encoding="utf-8")
    assert "Test GPU is missing." in text
    log_allow = tmp_path / "gpu_gate_allow.md"
    assert mod.main([str(log_allow), "--allow-no-gpu"], runner=_passing) == 0
    assert "NOT GPU evidence" in log_allow.read_text(encoding="utf-8")


def test_log_has_no_machine_identifiers(tmp_path, fake_gpu):
    mod = _load()
    log = tmp_path / "gpu_gate.md"
    assert mod.main([str(log)], runner=_passing) == 0
    text = log.read_text(encoding="utf-8")
    assert "- GPU: Test GPU" in text
    assert socket.gethostname() not in text
    user = getpass.getuser()
    # A very short user name occurs in normal words, so only a longer
    # name gives a strict check; the home-path check below still guards it.
    if len(user) > 2:
        assert user not in text
    assert str(Path.home()) not in text


def test_failed_step_gives_exit_1(tmp_path, fake_gpu):
    mod = _load()

    def _fail_first_example(step):
        if "01_elliptic_wing" in step.name:
            return mod.StepResult(
                name=step.name, status="fail", seconds=1.0, last_line="AssertionError"
            )
        return mod.StepResult(name=step.name, status="pass", seconds=0.1, last_line="ok")

    log = tmp_path / "gpu_gate.md"
    assert mod.main([str(log)], runner=_fail_first_example) == 1
    text = log.read_text(encoding="utf-8")
    assert text.strip().endswith("GPU gate: FAIL (1 failures)")


def test_timeout_output_is_text(monkeypatch):
    """A timeout gives the last output line as text (the captured output is bytes)."""
    mod = _load()

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1.0, output=b"first\nlast line\n", stderr=b"")

    monkeypatch.setattr(mod.subprocess, "run", _timeout)
    result = mod.run_step(mod.Step(name="slow", argv=["x"], env={}, timeout=1.0))
    assert result.status == "timeout"
    assert result.last_line == "last line"


def test_steps_import_this_tree(monkeypatch):
    """Each step imports Ventorum from the tree of the script (the logged commit)."""
    mod = _load()
    seen = {}

    def _run(argv, cwd, env, **kwargs):
        seen["path"] = env["PYTHONPATH"].split(mod.os.pathsep)[0]
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _run)
    mod.run_step(mod.Step(name="s", argv=["x"], env={}, timeout=None))
    assert Path(seen["path"]) == REPO_ROOT


def test_dry_run_shows_pytest_timeout_and_skip_report():
    proc = _dry_run()
    assert proc.returncode == 0
    pytest_line = next(line for line in proc.stdout.splitlines() if "pytest" in line)
    assert "-rs" in pytest_line
    assert "(timeout 3600 s)" in pytest_line
    proc = _dry_run("--pytest-timeout", "120")
    assert proc.returncode == 0
    pytest_line = next(line for line in proc.stdout.splitlines() if "pytest" in line)
    assert "(timeout 120 s)" in pytest_line


def test_pytest_timeout_must_be_positive():
    proc = _dry_run("--pytest-timeout", "0")
    assert proc.returncode == 2
    assert "--pytest-timeout must be positive" in proc.stderr


PYTEST_SKIP_OUTPUT = (
    "ss.s                                                         [100%]\n"
    "=========================== short test summary info ===========================\n"
    "SKIPPED [2] tests/test_gpu_pipeline.py:30: could not import 'warp': No module named 'warp'\n"
    "SKIPPED [1] tests/test_gpu_rates.py:12: no CUDA device\n"
    "1 passed, 3 skipped, 900 deselected in 4.20s\n"
)


def test_pytest_step_with_skips_is_not_pass(monkeypatch):
    """A pytest step that exits 0 with skipped tests gives the result "skip"."""
    mod = _load()

    def _run(argv, cwd, env, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=PYTEST_SKIP_OUTPUT, stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _run)
    step = mod.build_plan(False)[0]
    result = mod.run_step(step)
    assert result.status == "skip"
    assert result.skipped == 3
    assert len(result.skip_reasons) == 2
    assert "could not import 'warp'" in result.skip_reasons[0]
    # An example step does not count skips.
    example = mod.Step(name="examples/x.py", argv=["x"], env={}, timeout=1.0)
    assert mod.run_step(example).status == "pass"


def test_pytest_step_without_skips_passes(monkeypatch):
    mod = _load()

    def _run(argv, cwd, env, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="....\n4 passed, 900 deselected in 9.1s\n", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _run)
    result = mod.run_step(mod.build_plan(False)[0])
    assert result.status == "pass"
    assert result.skipped == 0


def _skipping_pytest(step):
    """Fake step runner: the pytest step has skipped tests, the examples pass."""
    mod = sys.modules["gpu_gate"]
    if step.counts_skips:
        return mod.StepResult(
            name=step.name, status="skip", seconds=1.0, last_line="1 passed, 3 skipped",
            skipped=3, skip_reasons=["SKIPPED [3] tests/test_gpu_pipeline.py:30: no warp"],
        )
    return mod.StepResult(name=step.name, status="pass", seconds=0.1, last_line="ok")


def test_skipped_tests_make_the_gate_incomplete(tmp_path, fake_gpu):
    mod = _load()
    log = tmp_path / "gpu_gate.md"
    assert mod.main([str(log)], runner=_skipping_pytest) == mod.EXIT_SKIPPED == 3
    text = log.read_text(encoding="utf-8")
    assert "GPU gate: PASS" not in text
    assert text.strip().endswith("GPU gate: INCOMPLETE (1 steps with skipped tests)")
    assert "| pytest -m gpu | skip |" in text
    assert "Skipped tests in step 'pytest -m gpu': 3" in text
    assert "- SKIPPED [3] tests/test_gpu_pipeline.py:30: no warp" in text


def test_failure_wins_over_skips(tmp_path, fake_gpu):
    mod = _load()

    def _runner(step):
        if "01_elliptic_wing" in step.name:
            return mod.StepResult(name=step.name, status="timeout", seconds=600.0, last_line="-")
        return _skipping_pytest(step)

    log = tmp_path / "gpu_gate.md"
    assert mod.main([str(log)], runner=_runner) == 1
    assert log.read_text(encoding="utf-8").strip().endswith("GPU gate: FAIL (1 failures)")


def test_forced_no_gpu_path_with_skips_is_not_evidence(tmp_path, monkeypatch):
    """The forced no-GPU path runs the steps; the skipped GPU tests make it incomplete."""
    mod = _load()
    monkeypatch.setenv("VENTORUM_GPU_GATE_FORCE_NO_GPU", "1")
    log = tmp_path / "gpu_gate.md"
    assert mod.main([str(log)], runner=_skipping_pytest) == 2
    log_allow = tmp_path / "gpu_gate_allow.md"
    assert mod.main([str(log_allow), "--allow-no-gpu"], runner=_skipping_pytest) == 3
    text = log_allow.read_text(encoding="utf-8")
    assert "NOT GPU evidence" in text
    assert "GPU gate: INCOMPLETE" in text
