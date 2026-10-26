"""Acceptance checks for pressure-based viscous panel coupling."""

from dataclasses import replace

import numpy as np
import pytest

from ventorum.airfoil import airfoil
from ventorum.core.datatypes import (
    ControlSurface, FlightCondition, LiftingSurface, LinearAirfoil, SolverSettings,
    TabulatedAirfoil, WingSection,
)
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.solver import PanelSolver
from ventorum.solvers.factory import make_solver
from ventorum.solvers.panel3d import _surface_strip_maps
from ventorum.viscous.alpha_correction import solve_viscous


def _wing(ar=8.0, shape="0012"):
    return LiftingSurface(
        name="Wing", semi_span=ar / 2, spacing="uniform",
        sections=[WingSection(y_frac=0, chord=1, shape=shape),
                  WingSection(y_frac=1, chord=1, shape=shape)],
    )


def _settings(**kwargs):
    return replace(SolverSettings(n_panels=3, n_chord=10, spacing="uniform",
                                  panel_backend="numpy", panel_cache=False), **kwargs)


def _condition(alpha=4.0, **kwargs):
    return FlightCondition(V_inf=30.0, alpha=np.radians(alpha), **kwargs)


def _solve(polar=None, *, wing=None, settings=None, conditions=None, **kwargs):
    return solve_viscous(
        _wing() if wing is None else wing,
        _condition() if conditions is None else conditions,
        _settings() if settings is None else settings,
        LinearAirfoil(a0=5.0) if polar is None else polar,
        model="panel3d", tol=1e-10, **kwargs,
    )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("half", [True, False])
def test_partition_identity(backend, half):
    wing = _wing()
    coords = airfoil("0012").coords
    mesh = mesh_lifting_surface(wing, [coords, coords], n_chord=4, n_span=2, half=half)
    right, left = _surface_strip_maps(mesh, wing, 2, np.linspace(0, 1, 3), 0, half)
    maps = right if half else left + right
    solver = PanelSolver(mesh, S_ref=8, b_ref=8, c_ref=1, backend=backend)
    solver._set_viscous_strips(maps)
    condition = _condition()
    solver.solve_conditions([condition])
    cfg = solver._configs[("free",)]
    g = solver._viscous_response(cfg, condition)
    derivative = condition.V_inf * (
        -np.sin(condition.alpha) * cfg.unit_mu[:, 0]
        + np.cos(condition.alpha) * cfg.unit_mu[:, 2]
    )
    assert np.max(np.abs(g.sum(axis=1) + derivative)) < 1e-10 * np.max(np.abs(derivative))
    assert solver.factorization_count == 1
    assert all(factors is None for factors in cfg.systems.values())


@pytest.mark.parametrize("kutta", ["morino", "pressure"])
@pytest.mark.parametrize("warm", [False, True])
def test_half_mesh_viscous_asymmetric_flow_refused(kutta, warm):
    wing = _wing()
    coords = airfoil("0012").coords
    mesh = mesh_lifting_surface(wing, [coords, coords], n_chord=4, n_span=2, half=True)
    maps, _ = _surface_strip_maps(mesh, wing, 2, np.linspace(0, 1, 3), 0, True)
    solver = PanelSolver(mesh, S_ref=8, b_ref=8, c_ref=1, kutta=kutta, backend="numpy")
    solver._set_viscous_strips(maps)
    if warm:
        solver.solve_conditions([_condition()])
        cfg = solver._configs[("free",)]
        units = cfg.viscous_units.copy()
    count = solver.factorization_count
    with pytest.raises(ValueError, match="asymmetric.*half mesh.*full mesh"):
        solver.solve_conditions([_condition(beta=np.radians(5))])
    assert solver.factorization_count == count
    if warm:
        np.testing.assert_array_equal(cfg.viscous_units, units)


@pytest.mark.parametrize("warm", [False, True])
def test_half_mesh_viscous_pressure_cache_refuses_asymmetric_flow(warm):
    wing = _wing()
    coords = airfoil("0012").coords
    mesh = mesh_lifting_surface(wing, [coords, coords], n_chord=4, n_span=2, half=True)
    maps, _ = _surface_strip_maps(mesh, wing, 2, np.linspace(0, 1, 3), 0, True)
    solver = PanelSolver(mesh, S_ref=8, b_ref=8, c_ref=1, kutta="pressure", backend="numpy")
    solver._set_viscous_strips(maps)
    condition = _condition()
    solver.solve_conditions([condition])
    cfg = solver._configs[("free",)]
    if warm:
        solver._viscous_pressure_response(cfg, solver._condition_coeffs(condition))
    cached = cfg.viscous_wake
    count = solver._viscous_solve_count
    with pytest.raises(ValueError, match="asymmetric.*half mesh.*full mesh"):
        solver._viscous_pressure_response(
            cfg, solver._condition_coeffs(_condition(beta=np.radians(5))),
        )
    assert cfg.viscous_wake is cached
    assert solver._viscous_solve_count == count


def test_viscous_mesh_error_estimate_refused():
    message = ("panel_estimate_mesh_error is not supported with the viscous panel coupling; "
               "run the mesh study inviscid")
    with pytest.raises(ValueError, match=message):
        _solve(settings=_settings(n_panels=4, panel_estimate_mesh_error=True))


@pytest.mark.parametrize("half", [True, False])
def test_partition_matches_finite_difference(half):
    wing = _wing()
    coords = airfoil("0012").coords
    mesh = mesh_lifting_surface(wing, [coords, coords], n_chord=4, n_span=2, half=half)
    right, left = _surface_strip_maps(mesh, wing, 2, np.linspace(0, 1, 3), 0, half)
    solver = PanelSolver(mesh, S_ref=8, b_ref=8, c_ref=1, backend="numpy")
    solver._set_viscous_strips(right if half else left + right)
    condition = _condition(beta=0 if half else np.radians(5))
    step = 1e-3
    conditions = [replace(condition, alpha=condition.alpha + k * step) for k in (-2, -1, 1, 2)]
    results = solver.solve_conditions(conditions)
    derivative = (results[0].mu - 8 * results[1].mu + 8 * results[2].mu - results[3].mu) / (12 * step)
    response = solver._viscous_response(solver._configs[("free",)], condition).sum(axis=1)
    assert np.max(np.abs(response + derivative)) < 1e-10 * max(1, np.max(np.abs(derivative)))
    assert solver.factorization_count == 1


def test_identity_polar_gives_zero_correction():
    first = _solve()
    a = first.details["lift_slope_section"]
    a0 = first.details["alpha0_section"]
    polars = {i: LinearAirfoil(a0=slope, alpha_L0=zero) for i, (slope, zero) in enumerate(zip(a, a0))}
    result = _solve(polars)
    inv = make_solver("panel3d").solve(_wing(), _condition(), _settings())
    assert result.converged
    assert np.max(np.abs(result.details["delta"])) < 1e-9
    np.testing.assert_allclose(result.details["cp"], inv.details["cp"], rtol=1e-12, atol=1e-14)
    for key in ("CL", "CDi", "CY", "Cl", "Cn"):
        assert getattr(result.totals, key) == pytest.approx(getattr(inv.totals, key), abs=1e-12)


def test_cl_matches_polar():
    polar = LinearAirfoil(a0=4.5)
    result = _solve(polar)
    assert result.converged
    cl = np.concatenate([sw.Cl for sw in result.spanwise])
    np.testing.assert_allclose(cl, polar.Cl(result.details["alpha_eff"]), atol=1e-10, rtol=0)
    area = result.details["strip_area"]
    assert result.totals.CL == pytest.approx(np.dot(cl, area) / 8, abs=1e-12)


@pytest.mark.slow
def test_matches_vlm_v1_high_aspect_ratio():
    from ventorum.panel2d.viscous import solve_viscous as solve_2d

    angles = np.arange(-2.0, 8.1, 1.0)
    section = solve_2d(airfoil("0012", n_points=161), angles, Re=2e6, max_iter=100, tol=1e-5)
    assert np.all(section.converged)
    polar = TabulatedAirfoil(alpha=np.radians(angles), Cl_data=np.asarray(section.Cl),
                            Cd_data=np.asarray(section.Cd), Cm_data=np.asarray(section.Cm), Re=2e6)
    wing = _wing(20)
    settings = _settings(n_panels=12, n_chord=32)
    cond = _condition()
    panel = _solve(polar, wing=wing, settings=settings, conditions=cond)
    vlm = solve_viscous(wing, cond, settings, polar, model="vlm", tol=1e-10)
    assert panel.converged and vlm.converged
    print(f"real polar comparison: CL_panel={panel.totals.CL:.16g}, CL_vlm={vlm.totals.CL:.16g}")
    assert abs(panel.totals.CL / vlm.totals.CL - 1) < 0.03


def test_pressure_kutta_closure():
    result = _solve(settings=_settings(panel_kutta="pressure"))
    assert result.converged
    assert np.max(np.abs(result.details["trailing_edge_pressure_jump"])) < 1e-8
    np.testing.assert_allclose(result.spanwise[0].Cl,
                               5 * result.details["alpha_eff"], atol=1e-10, rtol=0)


def test_sweep_reuses_one_factorization(monkeypatch):
    count = 0
    original = PanelSolver._factorize

    def factor(self, matrix):
        nonlocal count
        count += 1
        return original(self, matrix)

    monkeypatch.setattr(PanelSolver, "_factorize", factor)
    conditions = [_condition(a) for a in (-2, 0, 2, 4, 6)]
    sweep = _solve(conditions=conditions)
    assert count == 1
    assert all(r.details["factorization_count"] == 1 for r in sweep)
    for cond, result in zip(conditions, sweep):
        single = _solve(conditions=cond)
        assert single.totals.CL == pytest.approx(result.totals.CL, abs=1e-10)
        np.testing.assert_allclose(single.details["delta"], result.details["delta"], atol=1e-9)


def test_negative_sweep_starts_from_nearest_positive(monkeypatch):
    from ventorum.viscous import alpha_correction as module

    evaluate = module._PanelCoupling.evaluate
    residual = module._panel_residual
    initial = {}
    current = None

    def condition(coupling, solver, cfg, cond):
        nonlocal current
        current = cond.alpha
        return evaluate(coupling, solver, cfg, cond)

    def record(x, *args):
        initial.setdefault(current, x.copy())
        return residual(x, *args)

    monkeypatch.setattr(module._PanelCoupling, "evaluate", condition)
    monkeypatch.setattr(module, "_panel_residual", record)
    conds = [_condition(a) for a in (0, 6, -2)]
    results = _solve(conditions=conds)
    np.testing.assert_array_equal(initial[conds[-1].alpha], results[0].details["delta"][3:])


def test_stall_branch():
    angles = np.radians(np.arange(-20.0, 41.0))
    cl = np.where(angles <= np.radians(12), 6.3 * angles,
                  np.where(angles >= np.radians(13), 0.5,
                           6.3 * np.radians(12) + (0.5 - 6.3 * np.radians(12))
                           * (np.degrees(angles) - 12)))
    polar = TabulatedAirfoil(alpha=angles, Cl_data=cl, Cd_data=np.full_like(cl, 0.01))
    result = _solve(polar, conditions=_condition(16), settings=_settings(n_panels=2))
    assert result.converged
    assert result.details["post_stall"]
    assert result.details["branch_search_runs"] == 2
    for branch in result.details.get("branches", [result]):
        assert branch.converged
        assert branch.totals.e == pytest.approx(
            branch.totals.CL**2 / (np.pi * branch.totals.AR * branch.totals.CDi))
        np.testing.assert_allclose(branch.spanwise[0].Cl,
                                   polar.Cl(branch.details["alpha_eff"]), atol=1e-10)
    assert all(flag == "post-stall, non-unique" for flag in result.details["strip_flags"])


def test_profile_drag_and_moment():
    polar = LinearAirfoil(a0=5, Cd0=0.023, Cm0=-0.08)
    result = _solve(polar, wing=_wing(shape="2412"))
    area = result.details["strip_area"]
    correction = polar.Cm(result.details["alpha_eff"]) - result.details["cm_section_inviscid"]
    np.testing.assert_allclose(result.spanwise[0].Cm_section,
                               result.details["cm_section_pressure"] + correction, atol=1e-12)
    assert result.totals.CDp == pytest.approx(np.dot(area, polar.Cd(result.details["alpha_eff"])) / 8)
    assert result.totals.Cm == pytest.approx(result.details["panel_result"].Cm + np.dot(area, correction) / 8)


def test_gpu_request_refused():
    with pytest.raises(ValueError, match="CPU.*GPU|GPU.*CPU"):
        _solve(settings=_settings(panel_backend="gpu"))


@pytest.mark.parametrize("kutta", ["morino", "pressure"])
def test_ground_supported(kutta):
    result = _solve(settings=_settings(panel_kutta=kutta, panel_ground_solver="direct"),
                    conditions=_condition(h=2.0))
    assert result.converged
    assert result.details["factorization_count"] == 1
    np.testing.assert_allclose(result.spanwise[0].Cl, 5 * result.details["alpha_eff"], atol=1e-10)


def test_krylov_refused():
    with pytest.raises(ValueError, match="one.*LU|one.*factorization"):
        _solve(settings=_settings(panel_ground_solver="krylov"), conditions=_condition(h=2.0))


def test_response_memory_eviction():
    wing = _wing()
    coords = airfoil("0012").coords
    mesh = mesh_lifting_surface(wing, [coords, coords], n_chord=4, n_span=2, half=True)
    maps, _ = _surface_strip_maps(mesh, wing, 2, np.linspace(0, 1, 3), 0, True)
    solver = PanelSolver(mesh, S_ref=8, b_ref=8, c_ref=1)
    solver._set_viscous_strips(maps)
    solver.solve_conditions([_condition()])
    cfg = solver._configs[("free",)]
    assert solver._factor_bytes(cfg)[0] >= cfg.viscous_units.nbytes
    solver._factor_budget_host = 0
    solver._trim_factors()
    assert cfg.viscous_units is None
    assert not cfg.systems


@pytest.mark.parametrize("kutta", ["morino", "pressure"])
def test_analytic_residual_jacobian(monkeypatch, kutta):
    from ventorum.viscous import alpha_correction as module

    original = module._panel_residual
    checked = False

    def check(x, *args):
        nonlocal checked
        state = original(x, *args)
        if not checked:
            trial = x + np.linspace(-0.01, 0.02, len(x))
            jac = original(trial, *args)["jac"]
            fd = np.empty_like(jac)
            for k in range(len(x)):
                step = np.zeros_like(x)
                step[k] = 1e-6
                fd[:, k] = (original(trial + step, *args)["r"] - original(trial - step, *args)["r"]) / 2e-6
            np.testing.assert_allclose(jac, fd, rtol=2e-7, atol=2e-8)
            checked = True
        return state

    monkeypatch.setattr(module, "_panel_residual", check)
    result = _solve(settings=_settings(panel_kutta=kutta))
    assert checked and result.converged


def test_pressure_sweep_reuses_wake_response():
    settings = _settings(panel_kutta="pressure")
    single = _solve(settings=settings)
    sweep = _solve(settings=settings, conditions=[_condition(a) for a in (0, 2, 4, 6, 8)])
    assert all(r.converged for r in sweep)
    assert sweep[-1].details["factorization_count"] == 1
    assert sweep[-1].details["solve_count"] == single.details["solve_count"]
    assert sweep[-1].details["solve_rhs_count"] == single.details["solve_rhs_count"]


def test_ground_auto_sweep():
    conditions = [_condition(a, h=2.0) for a in (0, 2, 4)]
    results = _solve(conditions=conditions)
    for cond, res in zip(conditions, results):
        assert res.converged
        assert res.details["dispatch"]["ground_solver"] == "direct"
        assert "one direct LU" in res.details["dispatch"]["ground_solver_reason"]
        assert res.totals.CL == pytest.approx(_solve(conditions=cond).totals.CL, abs=1e-10)
    assert results[-1].details["factorization_count"] == len(conditions)


def test_full_mesh_backend_parity():
    cond = _condition(beta=np.radians(3))
    numpy = _solve(conditions=cond, settings=_settings(use_symmetry=False))
    numba = _solve(conditions=cond, settings=_settings(use_symmetry=False, panel_backend="numba"))
    assert numpy.converged and numba.converged
    np.testing.assert_allclose(numpy.details["cp"], numba.details["cp"], atol=1e-12, rtol=1e-12)


def test_flap_offset_and_moment():
    wing = _wing(shape="2412")
    wing.controls = [ControlSurface("flap", 0, 1, 0.75, np.radians(5))]
    result = _solve(wing=wing)
    assert result.converged
    np.testing.assert_allclose(result.details["cl_inviscid"],
                               result.details["cl_viscous"] - result.details["flap_offset"], atol=1e-10)
    assert np.all(np.isfinite(result.spanwise[0].Cm_section))


def test_unconverged_has_no_branch_verdict():
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = _solve(max_iter=0)
    assert not result.converged
    assert "branches" not in result.details
    assert not result.details["non_unique"]


def test_ground_polar_section_reference_uses_images():
    class Source:
        supports_ground = True

        def __init__(self):
            self.heights = []

        def polar(self, **kwargs):
            self.heights.append(kwargs.get("height"))
            angles = np.radians(kwargs["alphas_deg"])
            return TabulatedAirfoil(alpha=angles, Cl_data=5 * angles, Cd_data=np.zeros_like(angles))

    source = Source()
    free = _solve(source)
    ground = _solve(source, conditions=_condition(h=0.6))
    assert ground.converged
    assert any(h is not None for h in source.heights)
    assert np.max(np.abs(ground.details["lift_slope_section"] - free.details["lift_slope_section"])) > 1e-3
    np.testing.assert_allclose(ground.spanwise[0].Cl, 5 * ground.details["alpha_eff"], atol=1e-10)


def test_identity_moment_preserves_all_loads():
    from ventorum.panel3d.mesh import _close_trailing_edge, _resample_section, _target_x
    from ventorum.solvers.panel3d import _morino_section_line

    x = _target_x(10, "cosine")
    yu, yl = _resample_section(airfoil("2412").coords, x)
    yu, yl = _close_trailing_edge(x, yu, yl)
    polygon = np.vstack([np.column_stack([x[::-1], yu[::-1]]), np.column_stack([x[1:], yl[1:]])])
    slope, zero, cm = _morino_section_line(polygon)

    class IdentityPolar(LinearAirfoil):
        def Cm(self, angle):
            co, si = np.cos(angle), np.sin(angle)
            return cm[0] + cm[1] * co**2 + cm[2] * co * si + cm[3] * si**2

    wing = _wing(shape="2412")
    result = _solve(IdentityPolar(a0=slope, alpha_L0=zero), wing=wing)
    inv = make_solver("panel3d").solve(wing, _condition(), _settings())
    assert result.converged
    assert np.max(np.abs(result.details["delta"])) < 1e-9
    for key in ("CL", "CDi", "Cm", "CY", "Cl", "Cn"):
        assert getattr(result.totals, key) == pytest.approx(getattr(inv.totals, key), abs=1e-12)


def test_partition_assembly_evaluates_each_pair_once(monkeypatch):
    coords = airfoil("0012").coords
    wing = _wing()
    mesh = mesh_lifting_surface(wing, [coords, coords], n_chord=4, n_span=2, half=True)
    maps, _ = _surface_strip_maps(mesh, wing, 2, np.linspace(0, 1, 3), 0, True)
    solver = PanelSolver(mesh, S_ref=8, b_ref=8, c_ref=1)
    solver._set_viscous_strips(maps)
    pairs = 0
    original = solver._matrices

    def count(points, panels):
        nonlocal pairs
        pairs += len(points) * len(panels)
        return original(points, panels)

    monkeypatch.setattr(solver, "_matrices", count)
    solver.solve_conditions([_condition(a) for a in range(10)])
    assert pairs == 2 * len(solver.centroids)**2
    assert solver.factorization_count == 1
