# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""One canonical name per concept: n_jobs and solver names (T-0057)."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

import ventorum as vt
from ventorum.agent import call_tool
from ventorum.ground_effect import GroundEffectSweep, sweep_alpha, sweep_height, sweep_roll

_ALIASES = {
    "horseshoe": "vlm",
    "lattice": "vlm",
    "llt": "linear",
    "linear_llt": "linear",
}
_CANONICAL = ("auto", "vlm", "linear", "nonlinear", "fourier")


def _wing() -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.5),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )


def _straight_wing() -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )


def test_n_workers_is_removed():
    """n_workers= raises TypeError; n_jobs=2 matches the 2-worker result."""
    wing = _wing()
    alphas = np.array([0.0, 2.0, 4.0])

    with pytest.raises(TypeError):
        GroundEffectSweep(wing, n_workers=2)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        vt.Ventorum("W", geometry=wing, n_workers=2)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        sweep_height(wing, [2.0], n_workers=2)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        sweep_roll(wing, [0.0], n_workers=2)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        sweep_alpha(wing, [2.0], n_workers=2)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        vt.VentorumCaseManager().create_case("C", wing, n_workers=2)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        vt.Ventorum("W", geometry=wing).analyze_sweep([0.0], n_workers=2)  # type: ignore[call-arg]
    from ventorum.agent.tools import batch_evaluate

    with pytest.raises(TypeError):
        batch_evaluate.__wrapped__([{"span_m": 10.0, "chord_m": 1.0}], n_workers=2)  # type: ignore[call-arg]

    s1 = GroundEffectSweep(wing, n_jobs=1, n_panels=10).run_sweep(
        [2.0, 5.0], [2.0], [0.0], compute_strike_limit=False)
    s2 = GroundEffectSweep(wing, n_jobs=2, n_panels=10).run_sweep(
        [2.0, 5.0], [2.0], [0.0], compute_strike_limit=False)
    np.testing.assert_allclose(np.asarray(s1.CL, dtype=float), np.asarray(s2.CL, dtype=float),
                               rtol=0.0, atol=0.0)

    i1 = vt.Ventorum("A", geometry=wing, alpha_sweep_deg=alphas, n_jobs=1, n_panels=10)
    i2 = vt.Ventorum("B", geometry=wing, alpha_sweep_deg=alphas, n_jobs=2, n_panels=10)
    r1 = i1.run(progress=False)
    r2 = i2.run(progress=False)
    assert [r.totals.CL for r in r1] == pytest.approx([r.totals.CL for r in r2])


def test_mcp_batch_evaluate_n_jobs():
    """The MCP tool takes n_jobs and refuses n_workers with a hint to n_jobs."""
    cand = {"span_m": 10.0, "chord_m": 1.0}
    ok = call_tool("ventorum_batch_evaluate", {
        "candidates": [cand],
        "settings": {"n_panels": 8},
        "n_jobs": 1,
    })
    assert ok["status"] == "success", ok

    bad = call_tool("ventorum_batch_evaluate", {
        "candidates": [cand],
        "settings": {"n_panels": 8},
        "n_workers": 1,
    })
    assert bad["status"] == "error", bad
    assert bad["error"]["type"] == "invalid_input"
    assert "n_jobs" in bad["error"]["message"]


def test_instance_summary_keys():
    """Summaries carry n_jobs and no n_workers."""
    wing = _wing()
    inst = vt.Ventorum("S", geometry=wing, n_jobs=2, n_panels=10)
    inst.analyze(alpha_deg=2.0)
    summary = inst.get_summary()
    assert summary["n_jobs"] == 2
    assert "n_workers" not in summary

    mgr = vt.VentorumCaseManager()
    mgr.create_case("C", wing, alpha_sweep_deg=np.array([0.0, 2.0]), n_jobs=1, n_panels=10)
    for record in mgr.summary():
        assert "n_jobs" in record
        assert "n_workers" not in record


def test_solver_aliases_warn():
    """Old solver names warn and give the canonical solver; canonical names stay silent."""
    wing = _wing()
    for old, canonical in _ALIASES.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = vt.analyze(wing, alpha_deg=3.0, solver=old, n_panels=12)
        future = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(future) == 1
        assert old in str(future[0].message)
        assert canonical in str(future[0].message)
        assert "test_api_names" in future[0].filename
        assert res.solver_type == canonical

    for name in _CANONICAL:
        check_wing = _straight_wing() if name == "fourier" else wing
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vt.analyze(check_wing, alpha_deg=3.0, solver=name, n_panels=12)
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []


def test_instance_auto_means_automatic(monkeypatch):
    """Ventorum with n_jobs='auto' uses the automatic plan."""
    import ventorum.utils.parallel as parallel

    seen: list[object] = []
    real = parallel.plan_parallel

    def spy(n_tasks, n_panels=None, n_jobs="auto"):
        seen.append(n_jobs)
        return real(n_tasks, n_panels, n_jobs)

    monkeypatch.setattr(parallel, "plan_parallel", spy)
    wing = _wing()
    assert vt.Ventorum("D", geometry=wing, n_panels=10).n_jobs == 1
    inst = vt.Ventorum("A", geometry=wing, alpha_sweep_deg=np.array([0.0, 2.0, 4.0]),
                       n_jobs="auto", n_panels=10)
    assert inst.n_jobs == "auto"
    inst.run(progress=False)
    assert seen and all(job == "auto" for job in seen)


def test_docs_and_examples_use_canonical_names():
    """Docs, examples and README use n_jobs and canonical solver names."""
    root = Path(__file__).resolve().parents[1]
    targets = (
        list((root / "docs" / "user").glob("*.md"))
        + list((root / "docs" / "agent").glob("*.md"))
        + list((root / "examples").glob("*.py"))
        + [root / "README.md"]
    )
    assert targets
    solver_pattern = re.compile(r"(solver|solution_type|solver_type)\s*=\s*[\"']([^\"']+)[\"']")
    for path in targets:
        text = path.read_text(encoding="utf-8")
        if "## Old names" in text:
            text = text.split("## Old names")[0]
        assert "n_workers=" not in text, path
        for match in solver_pattern.finditer(text):
            assert match.group(2) not in ("horseshoe", "lattice", "llt", "linear_llt"), (path, match.group(0))


def test_old_solver_name_warns_once_per_entry_point_and_keeps_filters(monkeypatch):
    """Each public entry point warns one time and does not change the warning filters.

    The warning filters are global to the process. A change of them in a solve
    is not safe when solves run in threads (parallel sweeps).
    """
    wing = _wing()
    calls = {
        "analyze": lambda: vt.analyze(wing, alpha_deg=2.0, solver="horseshoe", n_panels=12),
        "analyze_sweep": lambda: vt.analyze_sweep(wing, [0.0, 2.0, 4.0], solver="horseshoe", n_panels=12),
        "Ventorum": lambda: vt.Ventorum("W", geometry=wing, solver="llt", n_panels=12).analyze(),
        "GroundEffectSweep": lambda: GroundEffectSweep(wing, solver="horseshoe", n_panels=12).run_sweep(
            [2.0], [2.0], compute_strike_limit=False),
    }
    for name, call in calls.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            changes = []
            monkeypatch.setattr(warnings, "simplefilter", lambda *a, _c=changes, **k: _c.append("simplefilter"))
            monkeypatch.setattr(warnings, "filterwarnings", lambda *a, _c=changes, **k: _c.append("filterwarnings"))
            try:
                call()
            finally:
                monkeypatch.undo()
        assert changes == [], (name, changes)
        future = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(future) == 1, (name, [str(w.message) for w in future])
        assert "test_api_names" in future[0].filename, name
