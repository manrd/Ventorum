# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the opt-in condition estimate of the direct lattice solves.

``SolverSettings.report_condition`` makes the vortex lattice and the linear
lifting line solve with an LU factorisation and fill
``SolveInfo.condition_number`` with the LAPACK 1-norm estimate. The default
(False) keeps the solve path of numpy and its bits.

The estimate of Hager and Higham is a lower bound of the exact 1-norm
condition number, so the tests require ``exact / 3 <= estimate <= exact``
(with a round-off margin on the upper bound).
"""

from __future__ import annotations

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero.system import ground_plane_from_condition, wake_direction
from ventorum.solvers import core
from ventorum.solvers.factory import make_solver
from ventorum.solvers.horseshoe import VortexLatticeSolver
from ventorum.solvers.lattice_base import CONDITION_GPU_NOTE, assemble_system_matrix

# Round-off margin of the upper bound (the estimate is a lower bound).
_UPPER = 1.0 + 1.0e-8


def _wing(n_panels: int = 6) -> vt.Aircraft:
    """Return a small tapered wing."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.5)],
        n_panels=n_panels,
    )
    return vt.Aircraft(name="Wing", surfaces=[wing])


def _in_band(estimate: float | None, exact: float) -> bool:
    """Return True when the estimate is in [exact / 3, exact]."""
    return estimate is not None and np.isfinite(estimate) and exact / 3.0 <= estimate <= _UPPER * exact


@pytest.mark.parametrize("solver, h, beta_deg", [
    ("vlm", None, 0.0),
    ("vlm", 0.4, 0.0),
    ("vlm", None, 1.0),
    ("linear", None, 0.0),
    ("linear", 2.0, 0.0),
])
def test_condition_estimate_within_factor_three_of_exact(solver, h, beta_deg):
    """Verify that the estimate of a single solve is within a factor of 3 of the exact 1-norm condition number."""
    ac = _wing()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(beta_deg), h=h)
    st = vt.SolverSettings(solver_type=solver, n_panels=6, report_condition=True)

    res = vt.analyze(ac, cond, st)

    exact = float(np.linalg.cond(assemble_system_matrix(ac, cond, st), 1))
    assert _in_band(res.details["condition_number"], exact)


@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_condition_estimate_of_a_sweep(solver):
    """Verify that each case of a sweep (one batch solve) gets the estimate of its own matrix."""
    ac = _wing()
    st = vt.SolverSettings(solver_type=solver, n_panels=6, report_condition=True)
    alphas = np.radians([0.0, 4.0, 8.0])
    base = vt.FlightCondition(V_inf=30.0, alpha=0.0, h=0.5 if solver == "vlm" else None)

    results = make_solver(solver).solve_sweep(ac, base, st, alphas)

    for a, res in zip(alphas, results):
        cond = vt.FlightCondition(V_inf=30.0, alpha=float(a), h=base.h)
        st_k = st.clone()
        if solver == "vlm":
            st_k.n_chord = res.details["lattice"].n_chord
        exact = float(np.linalg.cond(assemble_system_matrix(ac, cond, st_k), 1))
        assert _in_band(res.details["condition_number"], exact)


@pytest.mark.parametrize("ground", [False, True])
def test_vlm_batch_estimate_per_case_and_shared(ground):
    """Verify the estimates of solve_vlm_batch: one per matrix, shared by the cases of one matrix."""
    ac = _wing()
    ac.compute_reference_values()
    h = 0.5 if ground else None
    st = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=4)
    lat = VortexLatticeSolver().build(ac, st, vt.FlightCondition(V_inf=30.0, h=h))
    conds = [vt.FlightCondition(V_inf=30.0, alpha=np.radians(a), h=h, q=q)
             for a, q in [(2.0, 0.0), (2.0, 0.5), (6.0, 0.0)]]
    rp = ac.moment_reference()
    grounds = [ground_plane_from_condition(lat, c, rp) if ground else None for c in conds]
    wds = np.array([wake_direction(c, g, "freestream") for c, g in zip(conds, grounds)])

    plain = core.solve_vlm_batch(lat, conds, grounds, True, wds, rp)
    sols = core.solve_vlm_batch(lat, conds, grounds, True, wds, rp, report_condition=True)

    estimates = [info.condition_number for _, info in sols]
    for k, c in enumerate(conds):
        A = core.assemble_vlm(lat, c, grounds[k], True, wds[k], rp)[0]
        assert _in_band(estimates[k], float(np.linalg.cond(A, 1)))
        np.testing.assert_allclose(sols[k][0], plain[k][0], rtol=1.0e-12, atol=1.0e-14)
        assert plain[k][1].condition_number is None
    # The first two cases have one matrix (same alpha, only the pitch rate differs).
    assert estimates[0] == estimates[1]


@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_default_leaves_the_solve_bit_identical(solver):
    """Verify that the default (report_condition False) keeps the numpy solve and its bits."""
    ac = _wing()
    ac.compute_reference_values()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type=solver, n_panels=6)
    assert st.report_condition is False

    res = vt.analyze(ac, cond, st)

    lat = res.details["lattice"]
    rp = ac.moment_reference()
    wd = wake_direction(cond, None, "freestream")
    if solver == "vlm":
        A, rhs, umap = core.assemble_vlm(lat, cond, None, True, wd, rp)
        g, info = core.solve_vlm(lat, cond, None, True, wd, rp)
    else:
        A, rhs, umap = core.assemble_llt_linear(lat, cond, None, True, wd, rp)
        g, info = core.solve_llt_linear(lat, cond, None, True, wd, rp)
    expected = np.linalg.solve(A, rhs)[umap.panel_column]
    assert np.array_equal(res.details["gamma"], expected)
    assert np.array_equal(g, expected)
    assert info.condition_number is None
    assert "condition_number" not in res.details

    on = vt.analyze(ac, cond, vt.SolverSettings(solver_type=solver, n_panels=6, report_condition=True))
    np.testing.assert_allclose(on.details["gamma"], expected, rtol=1.0e-12, atol=1.0e-14)
    assert on.totals.CL == pytest.approx(res.totals.CL, rel=1.0e-12)


def test_setting_is_cloned():
    """Verify that clone keeps report_condition."""
    st = vt.SolverSettings(report_condition=True)
    assert st.clone().report_condition is True


def test_estimate_of_a_two_by_two_matrix():
    """Verify the solve and the estimate on a 2 x 2 matrix, where the estimate is exact."""
    A = np.array([[4.0, 1.0], [2.0, 3.0]])
    x, c = core.lu_solve_with_condition(A, np.array([1.0, 2.0]))
    np.testing.assert_allclose(A @ x, [1.0, 2.0])
    assert c == pytest.approx(np.linalg.cond(A, 1), rel=1.0e-12)


@pytest.mark.gpu
def test_gpu_path_gives_no_estimate_and_a_note():
    """Verify that a GPU solve leaves the estimate None and adds a note (the estimate is CPU-only)."""
    from ventorum import gpu

    if not gpu.available():
        pytest.skip("No CUDA GPU.")
    ac = _wing(n_panels=40)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=40, report_condition=True)
    old = gpu.get_device()
    try:
        gpu.set_device("gpu")
        res = vt.analyze(ac, cond, st)
    finally:
        gpu.set_device(old)
    assert res.details.get("device") == "gpu"
    assert res.details["condition_number"] is None
    assert CONDITION_GPU_NOTE in res.details["notes"]


@pytest.mark.parametrize("kappa, warned", [
    (None, False),
    (1.0e9, False),
    (1.0e10, False),
    (1.1e10, True),
    (1.0e13, True),
    (float("inf"), True),
])
def test_trust_warning_above_the_condition_limit(kappa, warned):
    """Verify the trust warning of a large condition estimate: above 1e10 or not finite, no penalty."""
    from ventorum.core.trust import CONDITION_NUMBER_WARNING, evaluate_aerodynamic_trust

    assert CONDITION_NUMBER_WARNING == 1.0e10
    base = evaluate_aerodynamic_trust(AR=8.0, CL=0.4, CDi=0.007, solver_type="vlm", n_chord=4)
    t = evaluate_aerodynamic_trust(AR=8.0, CL=0.4, CDi=0.007, solver_type="vlm", n_chord=4,
                                   condition_number=kappa)
    found = [w for w in t.warnings if "condition number" in w]
    assert bool(found) == warned
    # A warning only: the score and the penalties do not change.
    assert t.score == base.score and t.factors == base.factors
    if warned and np.isfinite(kappa):
        assert f"{kappa:.2e}" in found[0]
        assert f"about {np.log10(kappa):.0f} of the 16 significant float64 digits" in found[0]
        assert f"about {16 - np.log10(kappa):.0f} remain" in found[0]


@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_large_condition_estimate_reaches_the_trust_of_the_result(solver, monkeypatch):
    """Verify that the estimate of the solve reaches the trust warnings of the result.

    The real estimate of a small wing is far below the limit, so it gives no
    warning. A solve whose estimate is above the limit (here a scaled
    estimate) gives the warning.
    """
    ac = _wing()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type=solver, n_panels=6, report_condition=True)

    normal = vt.analyze(ac, cond, st)
    assert normal.details["condition_number"] < 1.0e10
    assert not any("condition number" in w for w in normal.totals.trust.warnings)

    real = core.lu_solve_with_condition

    def scaled(A, rhs):
        x, c = real(A, rhs)
        return x, c * 1.0e12

    monkeypatch.setattr(core, "lu_solve_with_condition", scaled)
    bad = vt.analyze(ac, cond, st)
    assert bad.details["condition_number"] > 1.0e10
    assert any("Ill-conditioned system matrix" in w for w in bad.totals.trust.warnings)
    # The default (no estimate) gives no check.
    plain = vt.analyze(ac, cond, vt.SolverSettings(solver_type=solver, n_panels=6))
    assert not any("condition number" in w for w in plain.totals.trust.warnings)


@pytest.mark.parametrize("solver, on_cpu", [("vlm", True), ("linear", True), ("nonlinear", False)])
def test_auto_device_keeps_the_condition_estimate_on_the_cpu(solver, on_cpu, monkeypatch):
    """The automatic device keeps a solve with report_condition on the CPU, before the cost model.

    Before the fix the automatic device could send the solve to the GPU,
    where the estimate is None, although the settings asked for it.
    """
    from ventorum import gpu
    from ventorum.gpu import pipeline

    monkeypatch.setattr(gpu, "get_device", lambda: "auto")
    monkeypatch.setattr(gpu, "use_gpu", lambda *a, **k: (False, "cost model"))
    s = make_solver(solver)
    ac = _wing()
    st = vt.SolverSettings(solver_type=solver, report_condition=True)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    lattice = s.build(ac, st, cond)
    out, reason = pipeline.solve_batch(s, lattice, [cond], st, 1.0, 1.0, 1.0, np.zeros(3), 0, False, 0.0,
                                       return_reason=True)
    assert out is None
    assert (reason == pipeline.REASON_REPORT_CONDITION) is on_cpu
    st_off = vt.SolverSettings(solver_type=solver, report_condition=False)
    _, reason_off = pipeline.solve_batch(s, lattice, [cond], st_off, 1.0, 1.0, 1.0, np.zeros(3), 0, False, 0.0,
                                         return_reason=True)
    assert reason_off == "cost model"


@pytest.mark.gpu
def test_auto_device_gives_the_estimate_on_a_gpu_machine():
    """With the automatic device and a GPU, a solve with report_condition runs on the CPU and gives the value."""
    from ventorum import gpu

    if not gpu.available():
        pytest.skip("No CUDA GPU.")
    ac = _wing(n_panels=80)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    old = gpu.get_device()
    try:
        gpu.set_device("auto")
        res = vt.analyze(ac, cond, vt.SolverSettings(solver_type="vlm", n_panels=80, report_condition=True))
    finally:
        gpu.set_device(old)
    assert res.details["dispatch"]["device"] == "cpu"
    assert "report_condition" in res.details["dispatch"]["reason"]
    assert res.details["condition_number"] is not None and np.isfinite(res.details["condition_number"])


def test_auto_device_keeps_a_ground_sweep_with_the_estimate_on_the_cpu(monkeypatch):
    """The GPU batch of a ground-effect sweep goes to the CPU when the settings ask for the estimate."""
    from ventorum import gpu
    from ventorum.ground_effect import sweep
    from ventorum.gpu.pipeline import REASON_REPORT_CONDITION

    def no_gpu(*args, **kwargs):
        raise AssertionError("the cost model must not run")

    monkeypatch.setattr(gpu, "get_device", lambda: "auto")
    monkeypatch.setattr(gpu, "use_gpu", no_gpu)
    st = vt.SolverSettings(solver_type="vlm", report_condition=True)
    reason: list[str] = []
    out = sweep._gpu_batch(_wing(), st, [(0, 0, 0)], np.array([0.5]), np.array([4.0]), np.array([0.0]),
                           30.0, 1.225, np.zeros(3), "ref", reason_out=reason)
    assert out is None
    assert reason == [REASON_REPORT_CONDITION]
