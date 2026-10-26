# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the 3D panel method user guide.

This module verifies that the user guide page exists, contains no forbidden
tokens, is registered in the user documentation toctree, and that every code
example runs and produces expected results.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from ventorum import gpu
from ventorum.core.datatypes import SolverResult
from ventorum.panel3d import cache

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs" / "user"
GUIDE_PATH = DOCS_DIR / "how_to_panel3d.md"
INDEX_PATH = DOCS_DIR / "index.md"


@pytest.fixture(autouse=True)
def _clear_panel_cache():
    """Ensure the panel cache is cleared before and after each test."""
    cache.clear_cache()
    yield
    cache.clear_cache()


def test_panel3d_guide_file_exists() -> None:
    """Verify that the user guide file exists and is non-empty."""
    assert GUIDE_PATH.is_file(), f"Missing user guide at {GUIDE_PATH}"
    content = GUIDE_PATH.read_text(encoding="utf-8")
    assert len(content.strip()) > 0, "Guide file is empty"


def test_panel3d_guide_no_em_dash_or_private_ids() -> None:
    """Verify that the guide contains no em dashes or private planning IDs."""
    text = GUIDE_PATH.read_text(encoding="utf-8")
    assert "\u2014" not in text, "The user guide contains an em dash character"

    private_pattern = re.compile(r"(?<!TN )\b(T-\d{4}|D-\d{2}|P[1-3]-\d{2})\b")
    hits = private_pattern.findall(text)
    assert not hits, f"Private planning IDs found in user guide: {hits}"


def test_panel3d_guide_in_toctree() -> None:
    """Verify that how_to_panel3d is listed in the user index toctree."""
    text = INDEX_PATH.read_text(encoding="utf-8")
    assert "how_to_panel3d" in text, "how_to_panel3d is not in docs/user/index.md"


def _extract_python_blocks() -> list[str]:
    """Extract all python code blocks from the user guide."""
    text = GUIDE_PATH.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    return [b.strip() for b in blocks if b.strip()]


def _run_guide_blocks() -> dict:
    """Run all code blocks in sequence and return the execution namespace."""
    blocks = _extract_python_blocks()
    assert len(blocks) >= 5, f"Expected at least 5 code blocks, got {len(blocks)}"

    namespace: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, block in enumerate(blocks):
            code = compile(block, f"how_to_panel3d.md[block_{i}]", "exec")
            exec(code, namespace)  # noqa: S102
    return namespace


def test_panel3d_guide_code_blocks_run() -> None:
    """Execute every code block in the user guide in sequence."""
    namespace = _run_guide_blocks()

    # Verify free-air solve
    res_free = namespace.get("res_free")
    assert isinstance(res_free, SolverResult)
    assert res_free.converged
    assert res_free.solver_type == "panel3d"
    assert res_free.totals.CL > 0.0
    assert res_free.totals.CDi >= 0.0
    assert "cp" in res_free.details
    assert isinstance(res_free.details["cp"], np.ndarray)
    assert len(res_free.details["cp"]) > 0

    # Verify ground-effect solve
    res_ground = namespace.get("res_ground")
    assert isinstance(res_ground, SolverResult)
    assert res_ground.converged
    assert res_ground.details["h_min"] is not None

    # Verify custom settings solve
    res_custom = namespace.get("res_custom")
    assert isinstance(res_custom, SolverResult)
    assert res_custom.converged
    assert res_custom.details["panel"]["kutta"] == "pressure"

    # Verify CPU backend selection
    res_cpu = namespace.get("res_cpu")
    assert isinstance(res_cpu, SolverResult)
    assert res_cpu.details["dispatch"]["backend"] == "numpy"
    assert res_cpu.details["dispatch"]["device"] == "cpu"

    # Verify flapped wing solve
    res_flap = namespace.get("res_flap")
    assert isinstance(res_flap, SolverResult)
    assert res_flap.converged
    assert res_flap.totals.CL > 0.0


def test_panel3d_guide_runs_when_gpu_reported_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that the code blocks run when ``gpu.available()`` reports no GPU (the GPU branch is skipped)."""
    monkeypatch.setattr(gpu, "available", lambda: False)
    namespace = _run_guide_blocks()

    res_free = namespace.get("res_free")
    assert isinstance(res_free, SolverResult)
    assert res_free.converged
    assert not gpu.available()


def test_panel3d_guide_numbers_match_the_code() -> None:
    """The numbers that the guide states are the values in the code."""
    from ventorum.core.datatypes import SolverSettings
    from ventorum.panel3d import mesh, solver, wake
    from ventorum.solvers import panel3d

    text = GUIDE_PATH.read_text(encoding="utf-8")
    assert f"{solver._FACTOR_BUDGET_HOST_FRACTION * 100:.0f} % of physical system memory" in text
    assert f"{solver._FACTOR_BUDGET_DEVICE_FRACTION * 100:.0f} % of free GPU memory" in text
    assert f"at most {solver._MAX_CONFIGURATIONS} configurations" in text
    assert f"CONTROL_TRANSITION = {mesh.CONTROL_TRANSITION}" in text
    assert f"default: {wake.DEFAULT_WAKE_CHORDS:.0f} times the largest streamwise chord" in text
    assert f"default: {panel3d._DEFAULT_N_CHORD} when None" in text
    assert f"default: {SolverSettings().n_panels})" in text
