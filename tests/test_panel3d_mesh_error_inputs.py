"""Check the ground plane and moment reference of panel mesh-error studies."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero.system import GroundPlane
from ventorum.panel3d.convergence import directional_estimate, level_counts, panel_mesh_convergence
from ventorum.solvers.panel3d import PanelMethodSolver

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture
def aircraft():
    return vt.Aircraft(
        name="Wing",
        surfaces=[vt.LiftingSurface(
            name="Wing", semi_span=4.0, is_symmetric=True,
            sections=[
                vt.WingSection(y_frac=0.0, chord=1.0, shape="2412"),
                vt.WingSection(y_frac=1.0, chord=1.0, shape="2412"),
            ],
        )],
    )


@pytest.fixture
def settings():
    return vt.SolverSettings(
        solver_type="panel3d", n_panels=4, n_chord=4, panel_backend="numpy",
    )


def _assert_same_bits(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same_bits(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            _assert_same_bits(value, reference)
    elif isinstance(expected, float):
        assert np.array_equal(np.asarray(actual).view(np.uint64), np.asarray(expected).view(np.uint64))
    else:
        assert actual == expected


def _check_levels(report, aircraft, condition, settings, *, condition_index=0, **inputs):
    solver = PanelMethodSolver()
    for (ns, nc), result in zip(report.meshes, report.results):
        reference = solver.solve(
            aircraft, condition, dataclasses.replace(settings, n_panels=ns, n_chord=nc), **inputs,
        )
        if isinstance(reference, list):
            reference = reference[condition_index]
        for name, estimate in report.quantities.items():
            assert getattr(result.totals, name) == getattr(reference.totals, name)
            if result is report.results[0]:
                assert estimate.base_value == getattr(reference.totals, name)
        assert result.symmetry_used == reference.symmetry_used
        assert np.array_equal(result.details["cp"], reference.details["cp"])


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("tilted", [False, True])
def test_estimate_uses_explicit_ground(aircraft, settings, batched, tilted):
    normal = np.array([0.0, 0.02 if tilted else 0.0, 1.0])
    ground = GroundPlane(normal=normal / np.linalg.norm(normal), offset=-0.3)
    conditions = [vt.FlightCondition(alpha=np.radians(4.0), V_inf=30.0)]
    if batched:
        conditions.append(dataclasses.replace(conditions[0], alpha=np.radians(6.0)))
    argument = conditions if batched else conditions[0]
    main = PanelMethodSolver().solve(
        aircraft, argument, dataclasses.replace(settings, panel_estimate_mesh_error=True), ground=ground,
    )
    reports = panel_mesh_convergence(aircraft, argument, settings, ground=ground)
    for index, (result, report, condition) in enumerate(zip(
        main if batched else [main], reports if batched else [reports], conditions,
    )):
        estimate = result.details["mesh_convergence"]["quantities"]
        for name in ("CL", "CDi", "Cm"):
            assert estimate[name]["base_value"] == getattr(result.totals, name)
        free = PanelMethodSolver().solve(aircraft, condition, settings)
        assert not np.isclose(estimate["CL"]["base_value"], free.totals.CL, rtol=1e-3, atol=1e-14)
        _check_levels(report, aircraft, argument, settings, condition_index=index, ground=ground)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("height", [None, 0.4])
def test_estimate_uses_ref_point(aircraft, settings, batched, height):
    ref_point = np.array([0.75, 0.0, 0.0])
    conditions = [vt.FlightCondition(alpha=np.radians(4.0), V_inf=30.0, h=height)]
    argument = conditions if batched else conditions[0]
    main = PanelMethodSolver().solve(
        aircraft, argument, dataclasses.replace(settings, panel_estimate_mesh_error=True), ref_point=ref_point,
    )
    report = panel_mesh_convergence(aircraft, argument, settings, ref_point=ref_point)
    main = main[0] if batched else main
    report = report[0] if batched else report
    estimate = main.details["mesh_convergence"]["quantities"]["Cm"]
    assert estimate["base_value"] == main.totals.Cm
    default = PanelMethodSolver().solve(aircraft, conditions[0], settings)
    assert not np.isclose(estimate["base_value"], default.totals.Cm, rtol=1e-3, atol=1e-14)
    _check_levels(report, aircraft, conditions[0], settings, ref_point=ref_point)


def test_estimate_uses_ground_and_ref_point(aircraft, settings):
    condition = vt.FlightCondition(alpha=np.radians(4.0), V_inf=30.0, h=0.6)
    inputs = {
        "ground": GroundPlane(normal=np.array([0.0, 0.0, 1.0]), offset=-0.3),
        "ref_point": np.array([0.75, 0.0, 0.0]),
    }
    main = PanelMethodSolver().solve(
        aircraft, condition, dataclasses.replace(settings, panel_estimate_mesh_error=True), **inputs,
    )
    base = PanelMethodSolver().solve(aircraft, condition, settings, **inputs)
    expected = panel_mesh_convergence(aircraft, condition, settings, **inputs)
    reused = panel_mesh_convergence(aircraft, condition, settings, base_results=[base], **inputs)
    assert reused.results[0] is base
    _assert_same_bits(main.details["mesh_convergence"], expected.to_dict())
    _assert_same_bits(reused.to_dict(), expected.to_dict())
    _check_levels(expected, aircraft, condition, settings, **inputs)


def test_explicit_ground_sets_symmetry_groups(aircraft, settings):
    conditions = [
        vt.FlightCondition(alpha=np.radians(4.0), V_inf=30.0),
        vt.FlightCondition(alpha=np.radians(4.0), beta=np.radians(3.0), V_inf=30.0),
    ]
    normal = np.array([0.0, 0.02, 1.0])
    ground = GroundPlane(normal=normal / np.linalg.norm(normal), offset=-0.3)
    base = PanelMethodSolver().solve(aircraft, conditions, settings, ground=ground)
    reports = panel_mesh_convergence(aircraft, conditions, settings, ground=ground, base_results=base)
    for index, report in enumerate(reports):
        assert report.results[0] is base[index]
        assert all(not result.symmetry_used for result in report.results)
        _check_levels(report, aircraft, conditions, settings, condition_index=index, ground=ground)


@pytest.mark.parametrize("height", [None, 0.4])
def test_defaults_unchanged(aircraft, settings, height):
    condition = vt.FlightCondition(alpha=np.radians(4.0), V_inf=30.0, h=height)
    spans = level_counts(settings.n_panels, 1.5, 3)
    chords = level_counts(settings.n_chord, 1.5, 3)
    meshes = [(spans[0], nc) for nc in chords] + [(ns, chords[0]) for ns in spans[1:]]
    # Compute the old level path with analyze, without explicit inputs.
    results = [
        vt.analyze(aircraft, condition=condition, settings=dataclasses.replace(settings, n_panels=ns, n_chord=nc))
        for ns, nc in meshes
    ]
    expected = {}
    for name in ("CL", "CDi", "Cm"):
        values = [getattr(result.totals, name) for result in results]
        estimate = directional_estimate(values[:3], chords, [values[i] for i in (0, 3, 4)], spans, quantity=name)
        panels = [len(result.details["cp"]) for result in results]
        estimate.chordwise.total_panels = panels[:3]
        estimate.spanwise.total_panels = [panels[i] for i in (0, 3, 4)]
        expected[name] = estimate.to_dict()
    report = panel_mesh_convergence(aircraft, condition, settings)
    attached = PanelMethodSolver().solve(
        aircraft, condition, dataclasses.replace(settings, panel_estimate_mesh_error=True),
    ).details["mesh_convergence"]
    _assert_same_bits(report.to_dict()["quantities"], expected)
    _assert_same_bits(attached["quantities"], expected)
