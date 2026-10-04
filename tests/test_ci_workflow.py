# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""The GitHub workflow files are valid YAML (review of T-0010).

A one-line mapping (``permissions: contents: read``) is not valid YAML; it
stopped the whole CI workflow. PyYAML is not a dependency of Ventorum, so the
test runs only where it is installed.

The main workflow must also run every gate: ruff, the test suite, the
verification report on both CPU backends with the report comparison, the
Sphinx documentation with warnings as errors, and the examples.
"""

from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
def test_workflow_is_valid_yaml(path):
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and "jobs" in data
    if "permissions" in data:
        assert isinstance(data["permissions"], dict)


def _steps():
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load((WORKFLOWS / "tests.yml").read_text(encoding="utf-8"))
    steps = data["jobs"]["test"]["steps"]
    runs = [str(s.get("run", "")) for s in steps]
    envs = [dict(s.get("env") or {}) for s in steps]
    return runs, envs


def test_workflow_runs_ruff():
    runs, _ = _steps()
    assert any("ruff check" in r for r in runs)


def test_workflow_runs_pytest():
    runs, _ = _steps()
    assert any("pytest" in r for r in runs)


def test_workflow_compares_default_verification_report():
    runs, _ = _steps()
    assert any("run_verification.py" in r for r in runs)
    assert any("compare_reports.py" in r for r in runs)


def test_workflow_compares_numba_verification_report():
    runs, envs = _steps()
    numba_runs = [r for r, e in zip(runs, envs)
                  if "run_verification.py" in r and e.get("VENTORUM_KERNEL") == "numba"]
    assert len(numba_runs) == 1
    assert sum("compare_reports.py" in r for r in runs) >= 2


def test_workflow_builds_docs_with_warnings_as_errors():
    runs, _ = _steps()
    assert any("sphinx-build" in r and "-W" in r for r in runs)


def test_workflow_runs_the_examples():
    runs, _ = _steps()
    assert any("examples/" in r for r in runs)
