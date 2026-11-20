# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests for the exit code of validation/run_verification.py.

The script must write the report in all cases and exit with a non-zero
code when a check of the report fails. The tests replace the report
builder with a fake one, so they do not run the verification cases.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "validation" / "run_verification.py"

PASSING = (
    "# Ventorum verification report\n"
    "\n"
    "| case | error | check |\n"
    "|---|---|---|\n"
    "| AR 8 | 0.002 % | pass |\n"
    "\n"
    "A FAIL in a sentence is not a verdict.\n"
)


@pytest.fixture(scope="module")
def script():
    """Import validation/run_verification.py from its file path."""
    spec = importlib.util.spec_from_file_location("run_verification", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_verification"] = module
    spec.loader.exec_module(module)
    return module


def _fake_main(text):
    def _main(out_path=None):
        out = Path(out_path)
        out.write_text(text, encoding="utf-8")
        return out
    return _main


def test_failed_rows_finds_only_verdict_cells(script):
    failing = PASSING + "| AR 4 | 3.1 % | FAIL |\n| order | FAIL | pass |\n"
    assert script.failed_rows(PASSING) == []
    assert script.failed_rows(failing) == ["| AR 4 | 3.1 % | FAIL |", "| order | FAIL | pass |"]


def test_exit_code_zero_when_all_checks_pass(script, tmp_path, monkeypatch, capsys):
    out = tmp_path / "report.md"
    monkeypatch.setattr(script, "main", _fake_main(PASSING))
    assert script.cli([str(out)]) == 0
    assert out.read_text(encoding="utf-8") == PASSING
    assert str(out) in capsys.readouterr().out


def test_exit_code_one_when_a_check_fails(script, tmp_path, monkeypatch, capsys):
    out = tmp_path / "report.md"
    failing = PASSING + "| AR 4 | 3.1 % | FAIL |\n"
    monkeypatch.setattr(script, "main", _fake_main(failing))
    assert script.cli([str(out)]) == 1
    # The report is written also when a check fails.
    assert out.read_text(encoding="utf-8") == failing
    err = capsys.readouterr().err
    assert "1 failed checks" in err
    assert "| AR 4 | 3.1 % | FAIL |" in err


def test_committed_report_has_no_failed_check(script):
    text = (SCRIPT.parents[1] / "docs" / "verification_report.md").read_text(encoding="utf-8")
    assert script.failed_rows(text) == []
