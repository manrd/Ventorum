# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""The GitHub workflow files are valid YAML (review of T-0010).

A one-line mapping (``permissions: contents: read``) is not valid YAML; it
stopped the whole CI workflow. PyYAML is not a dependency of Ventorum, so the
test runs only where it is installed.
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
