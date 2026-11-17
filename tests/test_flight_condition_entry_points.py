"""Every solve entry point refuses a flight condition that cannot be solved.

Regression test: only ``ventorum.analyze`` checked the flight condition.
``V_inf=-50`` gave CL 0.32 with the trust rating HIGH on the other entry
points, and ``V_inf=0`` raised a ZeroDivisionError without a message. The
solvers now check each case at a shared lower entry point, with the error
messages of ``ventorum.analyze`` and without repeating its warnings.
"""

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.solvers.fourier import FourierSolver
from ventorum.visualization.drag_polar import alpha_sweep

BAD_SPEEDS = [-50.0, 0.0]
ALPHAS = np.radians([2.0, 4.0])
MATCH = r"V_inf=.* must be >= "


def _wing() -> vt.LiftingSurface:
    """Return a rectangular wing (semi-span 4 m, chord 1 m)."""
    return vt.LiftingSurface(name="W", semi_span=4.0, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])


def _settings(solver: str = "vlm") -> vt.SolverSettings:
    return vt.SolverSettings(solver_type=solver, n_panels=12)


def _built(solver):
    """Return the aircraft and the lattice of *solver* for the wing."""
    ac = vt.Aircraft(surfaces=[_wing()])
    ac.compute_reference_values()
    lattice = solver.build(ac, _settings("vlm" if solver.collocation == "vlm" else "linear"),
                           vt.FlightCondition(alpha=0.05))
    return ac, lattice


@pytest.mark.parametrize("V", BAD_SPEEDS)
@pytest.mark.parametrize("kind", ["vlm", "linear", "nonlinear"])
def test_lattice_solve_refuses_bad_speed(kind, V):
    solver = {"vlm": vt.VortexLatticeSolver, "linear": vt.LinearLLTSolver,
              "nonlinear": vt.NonlinearLLTSolver}[kind]()
    with pytest.raises(ValueError, match=MATCH):
        solver.solve(_wing(), vt.FlightCondition(V_inf=V, alpha=0.07), _settings(kind))
    with pytest.raises(ValueError, match=MATCH):
        solver.solve_sweep(_wing(), vt.FlightCondition(V_inf=V), _settings(kind), ALPHAS)


@pytest.mark.parametrize("V", BAD_SPEEDS)
def test_solve_lattice_and_solve_batch_refuse_bad_speed(V):
    solver = vt.VortexLatticeSolver()
    ac, lattice = _built(solver)
    cond = vt.FlightCondition(V_inf=V, alpha=0.07)
    with pytest.raises(ValueError, match=MATCH):
        solver.solve_lattice(lattice, cond, _settings(), ac.S_ref, ac.b_ref, ac.c_ref)
    good = vt.FlightCondition(V_inf=30.0, alpha=0.07)
    with pytest.raises(ValueError, match=MATCH):
        solver.solve_batch(lattice, [good, cond], _settings(), ac.S_ref, ac.b_ref, ac.c_ref)


@pytest.mark.parametrize("V", BAD_SPEEDS)
@pytest.mark.parametrize("backend", ["serial", "thread"])
def test_alpha_sweep_refuses_bad_speed(backend, V):
    ac = vt.Aircraft(surfaces=[_wing()])
    with pytest.raises(ValueError, match=MATCH):
        alpha_sweep(ac, vt.FlightCondition(V_inf=V), _settings(), ALPHAS, n_jobs=2, backend=backend)


@pytest.mark.parametrize("V", BAD_SPEEDS)
def test_fourier_solver_refuses_bad_speed(V):
    solver = FourierSolver()
    with pytest.raises(ValueError, match=MATCH):
        solver.solve(_wing(), vt.FlightCondition(V_inf=V, alpha=0.07), _settings("fourier"))
    with pytest.raises(ValueError, match=MATCH):
        solver.solve_sweep(_wing(), vt.FlightCondition(V_inf=V), _settings("fourier"), ALPHAS)


@pytest.mark.parametrize("V", BAD_SPEEDS)
def test_package_and_instance_sweeps_refuse_bad_speed(V):
    with pytest.raises(ValueError, match=MATCH):
        vt.analyze_sweep(_wing(), np.array([2.0, 4.0]), V_inf=V, n_panels=12)
    inst = vt.Ventorum("A", geometry=_wing(), alpha_sweep_deg=np.array([2.0, 4.0]), n_jobs=1, n_panels=12)
    with pytest.raises(ValueError, match=MATCH):
        inst.analyze_sweep(V_inf=V)


def test_sweep_angle_out_of_range_is_refused_by_the_solver():
    """An angle above pi rad in a direct solver sweep gets the error of analyze."""
    with pytest.raises(ValueError, match="alpha"):
        vt.VortexLatticeSolver().solve_sweep(_wing(), vt.FlightCondition(), _settings(), np.array([0.0, 4.0]))


def _count(fn, text: str) -> int:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
    return sum(text in str(w.message) for w in rec)


@pytest.mark.parametrize("entry", ["analyze", "analyze_sweep", "solve", "fourier_analyze"])
def test_envelope_warning_is_given_once(entry):
    """The warning of the speed check (Mach above 0.3) is not repeated by the lower check."""
    V = 120.0  # Mach about 0.35 at sea level.
    calls = {
        "analyze": lambda: vt.analyze(_wing(), alpha_deg=4.0, V_inf=V, n_panels=12),
        "analyze_sweep": lambda: vt.analyze_sweep(_wing(), np.array([2.0, 4.0, 6.0]), V_inf=V, n_panels=12),
        "solve": lambda: vt.VortexLatticeSolver().solve(_wing(), vt.FlightCondition(V_inf=V, alpha=0.07),
                                                        _settings()),
        "fourier_analyze": lambda: vt.analyze(_wing(), alpha_deg=4.0, V_inf=V, n_panels=12, solver="fourier"),
    }
    assert _count(calls[entry], "above the incompressible limit") == 1


def test_valid_condition_results_do_not_change():
    """A valid condition gives the same result through solve, solve_sweep and alpha_sweep."""
    cond = vt.FlightCondition(V_inf=30.0, alpha=float(ALPHAS[1]))
    solver = vt.VortexLatticeSolver()
    single = solver.solve(_wing(), cond, _settings())
    sweep = solver.solve_sweep(_wing(), cond, _settings(), ALPHAS)
    poly = alpha_sweep(vt.Aircraft(surfaces=[_wing()]), cond, _settings(), ALPHAS, backend="serial")
    assert sweep[1].totals.CL == pytest.approx(single.totals.CL, rel=1e-12)
    assert poly[1].totals.CL == pytest.approx(single.totals.CL, rel=1e-12)
