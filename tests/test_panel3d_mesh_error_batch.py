# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Regression tests for shared directional mesh-error studies."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import scipy.linalg

import ventorum as vt
import ventorum.solvers.panel3d as factory
from ventorum.panel3d.convergence import ConvergenceReport, panel_mesh_convergence

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture
def aircraft():
    """Return a rectangular wing with span 8 m and chord 1 m."""
    return vt.Aircraft(
        name="Wing",
        surfaces=[vt.LiftingSurface(
            name="Wing", semi_span=4.0,
            sections=[
                vt.WingSection(y_frac=0.0, chord=1.0, shape="0012"),
                vt.WingSection(y_frac=1.0, chord=1.0, shape="0012"),
            ],
            is_symmetric=True,
        )],
    )


@pytest.fixture
def conditions():
    """Return three distinct conditions with angles in rad and speeds in m/s."""
    return [
        vt.FlightCondition(alpha=np.radians(a), V_inf=v)
        for a, v in ((2.0, 22.0), (4.0, 30.0), (6.0, 40.0))
    ]


def _settings(backend="numpy"):
    return vt.SolverSettings(
        solver_type="panel3d", n_panels=4, n_chord=4, panel_backend=backend,
    )


@pytest.mark.parametrize("backend", ["numpy", "numba", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("levels", [3, 4])
def test_batch_reports_equal_single_studies(aircraft, conditions, backend, levels, monkeypatch):
    """Each level uses one solver call and preserves every report value."""
    if backend == "gpu":
        torch = pytest.importorskip("torch")
        pytest.importorskip("warp")
        if not torch.cuda.is_available():
            pytest.skip("A CUDA GPU is not available.")
    settings = _settings(backend)
    expected = [panel_mesh_convergence(aircraft, c, settings, levels=levels) for c in conditions]
    original = factory.PanelMethodSolver.solve
    calls = []

    def counted(self, ac, conds, st, **kwargs):
        calls.append(list(conds))
        return original(self, ac, conds, st, **kwargs)

    monkeypatch.setattr(factory.PanelMethodSolver, "solve", counted)
    actual = panel_mesh_convergence(aircraft, conditions, settings, levels=levels)
    assert isinstance(actual, list)
    assert len(actual) == len(conditions)
    assert len(calls) == 1 + 2 * (levels - 1)
    assert all(len(call) == len(conditions) for call in calls)
    for report, reference in zip(actual, expected):
        assert report.to_dict() == reference.to_dict()
        for result, single in zip(report.results, reference.results):
            assert np.array_equal(result.details["cp"], single.details["cp"])


@pytest.mark.parametrize("single", [False, True])
def test_base_results_skip_base_solve(aircraft, conditions, single, monkeypatch):
    """Reusing base results removes one construction and keeps the whole report."""
    settings = _settings()
    conds = conditions[:1] if single else conditions
    argument = conds[0] if single else conds
    expected = panel_mesh_convergence(aircraft, argument, settings)
    base = factory.PanelMethodSolver().solve(aircraft, conds, settings)
    original = factory.PanelSolver
    constructions = []

    def counted(mesh, **kwargs):
        constructions.append(mesh)
        return original(mesh, **kwargs)

    monkeypatch.setattr(factory, "PanelSolver", counted)
    actual = panel_mesh_convergence(aircraft, argument, settings, base_results=tuple(base))
    assert len(constructions) == 4
    reports = [actual] if single else actual
    references = [expected] if single else expected
    for i, (report, reference) in enumerate(zip(reports, references)):
        assert report.to_dict() == reference.to_dict()
        assert report.results[0] is base[i]


def test_factory_shares_study_and_reuses_base(aircraft, conditions, monkeypatch):
    """The factory builds one base and four refined solvers for three conditions."""
    settings = _settings()
    expected = [panel_mesh_convergence(aircraft, c, settings).to_dict() for c in conditions]
    baseline = factory.PanelMethodSolver().solve(aircraft, conditions, settings)
    settings.panel_estimate_mesh_error = True
    original_solver = factory.PanelSolver
    original_study = factory._estimate_panel_mesh_error
    original_lu = scipy.linalg.lu_factor
    constructions = []
    studies = []
    factorizations = []

    def counted_solver(mesh, **kwargs):
        constructions.append(mesh)
        return original_solver(mesh, **kwargs)

    def counted_study(ac, conds, st, *, base_results):
        assert len(base_results) == len(conditions)
        assert all("mesh_convergence" not in r.details for r in base_results)
        studies.append(base_results)
        return original_study(ac, conds, st, base_results=base_results)

    def counted_lu(*args, **kwargs):
        factorizations.append(1)
        return original_lu(*args, **kwargs)

    monkeypatch.setattr(factory, "PanelSolver", counted_solver)
    monkeypatch.setattr(factory, "_estimate_panel_mesh_error", counted_study)
    monkeypatch.setattr(scipy.linalg, "lu_factor", counted_lu)
    results = factory.PanelMethodSolver().solve(aircraft, conditions, settings)
    assert len(studies) == 1
    assert len(constructions) == len(factorizations) == 5
    assert len({id(mesh) for mesh in constructions}) == 5
    for result, old, report in zip(results, baseline, expected):
        assert result.details["mesh_convergence"] == report
        assert np.array_equal(result.details["cp"], old.details["cp"])
        assert result.details["notes"] == old.details["notes"] + report["notes"]
        trust = dataclasses.asdict(result.totals.trust)
        old_trust = dataclasses.asdict(old.totals.trust)
        old_trust["warnings"].extend(n for n in report["notes"] if n not in old_trust["warnings"])
        assert trust == old_trust


@pytest.mark.parametrize("default", [False, True])
def test_single_and_default_return_one_report(aircraft, conditions, default):
    """A scalar condition or None keeps the scalar return type and values."""
    cond = None if default else conditions[0]
    report = panel_mesh_convergence(aircraft.surfaces[0], cond, _settings())
    reference = panel_mesh_convergence(
        aircraft.surfaces[0], [vt.FlightCondition() if default else cond], _settings(),
    )
    assert isinstance(report, ConvergenceReport)
    assert report.to_dict() == reference[0].to_dict()


def test_base_results_require_one_result_per_condition(aircraft, conditions):
    """A partial base level cannot silently omit a condition."""
    with pytest.raises(ValueError, match="base_results"):
        panel_mesh_convergence(aircraft, conditions, _settings(), base_results=[])


def test_empty_sequence_returns_no_reports(aircraft):
    """An empty condition sequence has no reports or solves."""
    assert panel_mesh_convergence(aircraft, [], _settings(), base_results=[]) == []


def test_mixed_symmetry_reports_equal_single_studies(aircraft):
    """Symmetric and sideslip conditions in one batch keep the reports of single studies.

    The factory solves a half mesh only when every condition of a call is
    symmetric, so the study solves each symmetry group in its own call, and
    it does not reuse base results that came from one mixed (full-mesh) call.
    """
    settings = _settings()
    conds = [
        vt.FlightCondition(alpha=np.radians(4.0), V_inf=30.0),
        vt.FlightCondition(alpha=np.radians(4.0), beta=np.radians(5.0), V_inf=30.0),
    ]
    expected = [panel_mesh_convergence(aircraft, c, settings).to_dict() for c in conds]
    actual = panel_mesh_convergence(aircraft, conds, settings)
    assert [r.to_dict() for r in actual] == expected
    mixed_base = factory.PanelMethodSolver().solve(aircraft, conds, settings)
    reused = panel_mesh_convergence(aircraft, conds, settings, base_results=mixed_base)
    assert [r.to_dict() for r in reused] == expected
    assert reused[0].results[0] is not mixed_base[0]


def test_factory_override_gives_the_study_of_the_same_reference(aircraft, conditions):
    """With an explicit moment reference point the study uses the same point.

    Every level of the study solves with the moment reference point of the
    call (owner decision 2026-10-26: the estimate belongs to the case that
    the user solves), so it equals a study that is given the same point.
    """
    rp = np.array([0.25, 0.0, 0.0])
    settings = _settings()
    expected = [panel_mesh_convergence(aircraft, c, settings, ref_point=rp).to_dict() for c in conditions]
    settings.panel_estimate_mesh_error = True
    results = factory.PanelMethodSolver().solve(aircraft, conditions, settings, ref_point=rp)
    assert [r.details["mesh_convergence"] for r in results] == expected
