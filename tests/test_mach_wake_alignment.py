"""A body-axis wake with the compressibility correction.

Regression test: with ``wake_alignment="body"`` and ``mach > 0`` the solve
used a wake along the stretched free stream, while ``details["wake_dir"]``
and the Trefftz plane used the body x-axis. The lift then jumped between
``mach=None`` and a very small Mach number (rectangular wing of aspect
ratio 8 at alpha 10 deg: CL 0.791643 at mach None, 0.796063 at mach 1e-4).
The solve now maps the body-axis wake into the stretched axes of the
Goethert rule (B. H. Goethert, NACA TM 1105, 1946), on the CPU, on the GPU
and in the differentiable gradient reference.
"""

import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.aero.loads import stretched_wake_direction
from ventorum.aero.system import freestream_direction
from ventorum.core.errors import ValidityError
from ventorum.solvers.core import assemble_vlm
from ventorum.solvers.lattice_base import assemble_system_matrix

ALPHA = np.radians(10.0)
FIELDS = ("CL", "CDi", "Cm")


def _wing() -> vt.Aircraft:
    """Return a rectangular wing of aspect ratio 8 (semi-span 4 m, chord 1 m)."""
    return vt.Aircraft(surfaces=[vt.LiftingSurface(name="W", semi_span=4.0,
                                                   sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])])


def _settings(wake: str) -> vt.SolverSettings:
    return vt.SolverSettings(solver_type="vlm", n_panels=16, wake_alignment=wake)


def _solve(wake: str, mach, alpha=ALPHA, beta=0.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vt.VortexLatticeSolver().solve(
            _wing(), vt.FlightCondition(alpha=alpha, beta=beta, V_inf=50.0, mach=mach), _settings(wake))


@pytest.mark.parametrize("alpha_deg, beta_deg", [(10.0, 0.0), (6.0, 5.0)])
def test_body_wake_mach_is_continuous_at_zero(alpha_deg, beta_deg):
    """The body-axis wake at mach 1e-6 gives the mach None result to about 1e-6 relative."""
    a, b = np.radians(alpha_deg), np.radians(beta_deg)
    ref = _solve("body", None, a, b)
    low = _solve("body", 1e-6, a, b)
    for f in FIELDS:
        assert getattr(low.totals, f) == pytest.approx(getattr(ref.totals, f), rel=1e-6), f
    # The sideslip case also has lateral moments: they stay continuous.
    assert low.totals.Cl == pytest.approx(ref.totals.Cl, rel=1e-6, abs=1e-9)
    assert low.totals.Cn == pytest.approx(ref.totals.Cn, rel=1e-6, abs=1e-9)
    assert np.array_equal(low.details["wake_dir"], [1.0, 0.0, 0.0])


def test_body_wake_with_mach_differs_from_the_freestream_wake():
    """At mach 0.3 the body-axis wake keeps the sign and the size of its difference at mach None."""
    body_0, free_0 = _solve("body", None), _solve("freestream", None)
    body_m, free_m = _solve("body", 0.3), _solve("freestream", 0.3)
    d0 = body_0.totals.CL - free_0.totals.CL
    dm = body_m.totals.CL - free_m.totals.CL
    assert d0 < 0.0 and dm < 0.0
    # The Prandtl-Glauert factor 1 / beta (1.048 at M 0.3) acts on the difference too.
    assert dm / d0 == pytest.approx(1.0 / math.sqrt(1.0 - 0.09), rel=0.05)


def test_stretched_wake_direction():
    """The free stream maps to exactly (1, 0, 0); the body x-axis to its stretched image."""
    mach = 0.5
    pg = math.sqrt(1.0 - mach * mach)
    cond = vt.FlightCondition(alpha=ALPHA, mach=mach)
    assert np.array_equal(stretched_wake_direction(cond, None), [1.0, 0.0, 0.0])
    assert np.array_equal(stretched_wake_direction(cond, freestream_direction(ALPHA, 0.0)), [1.0, 0.0, 0.0])
    # Without sideslip the body x-axis is (cos(alpha), 0, -sin(alpha)) in
    # free-stream axes; the stretch divides the first component by beta.
    w = stretched_wake_direction(cond, np.array([1.0, 0.0, 0.0]))
    expected = np.array([math.cos(ALPHA) / pg, 0.0, -math.sin(ALPHA)])
    assert np.allclose(w, expected / np.linalg.norm(expected), rtol=0.0, atol=1e-15)
    # The angle to the stretched free stream: tan(theta) = -beta tan(alpha).
    assert math.atan2(w[2], w[0]) == pytest.approx(math.atan(-pg * math.tan(ALPHA)), abs=1e-14)


def test_body_wake_sweep_and_batch_with_mach_equal_single_solves():
    """A sweep, a batch and the system matrix with a body-axis wake and mach run and agree."""
    solver = vt.VortexLatticeSolver()
    alphas = np.radians([0.0, 4.0, 8.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sweep = solver.solve_sweep(_wing(), vt.FlightCondition(V_inf=50.0, mach=0.2), _settings("body"), alphas)
    for a, res in zip(alphas, sweep):
        single = _solve("body", 0.2, alpha=a)
        for f in FIELDS:
            assert getattr(res.totals, f) == pytest.approx(getattr(single.totals, f), rel=1e-12, abs=1e-15)
    A_body = assemble_system_matrix(_wing(), vt.FlightCondition(alpha=ALPHA, mach=0.2), _settings("body"))
    A_free = assemble_system_matrix(_wing(), vt.FlightCondition(alpha=ALPHA, mach=0.2), _settings("freestream"))
    assert A_body.shape == A_free.shape
    assert not np.allclose(A_body, A_free)


def test_freestream_wake_matrix_with_mach_is_unchanged():
    """The free-stream wake (None or the free-stream direction) gives the same matrix to the last bit."""
    ac = _wing()
    ac.compute_reference_values()
    lattice = vt.VortexLatticeSolver().build(ac, _settings("freestream"), vt.FlightCondition(alpha=ALPHA))
    cond = vt.FlightCondition(alpha=ALPHA, mach=0.2)
    A_none, rhs_none, _ = assemble_vlm(lattice, cond, None, True, None)
    A_free, rhs_free, _ = assemble_vlm(lattice, cond, None, True, freestream_direction(ALPHA, 0.0))
    assert np.array_equal(A_none, A_free)
    assert np.array_equal(rhs_none, rhs_free)
    A_body, _, _ = assemble_vlm(lattice, cond, None, True, np.array([1.0, 0.0, 0.0]))
    assert not np.allclose(A_body, A_free)


def test_mach_in_ground_effect_is_still_refused():
    """The body-axis wake does not open the compressibility correction in ground effect."""
    solver = vt.VortexLatticeSolver()
    with warnings.catch_warnings(), pytest.raises(ValidityError, match="ground effect"):
        warnings.simplefilter("ignore")
        solver.solve(_wing(), vt.FlightCondition(alpha=np.radians(4.0), V_inf=50.0, mach=0.2, h=1.0),
                     _settings("body"))


def test_body_wake_without_mach_is_unchanged():
    """The body-axis wake at mach None keeps its result (CL 0.791643 in the defect report)."""
    res = _solve("body", None)
    assert res.totals.CL == pytest.approx(0.791643, abs=5e-7)
    assert np.allclose(res.details["wake_dir"], [1.0, 0.0, 0.0])


def test_freestream_wake_mach_is_continuous_at_zero():
    """The free-stream wake at mach 1e-6 gives the mach None result to about 1e-6 relative."""
    ref = _solve("freestream", None)
    low = _solve("freestream", 1e-6)
    assert low.totals.CL == pytest.approx(ref.totals.CL, rel=1e-6)
    assert low.totals.CDi == pytest.approx(ref.totals.CDi, rel=1e-6)
    assert low.totals.CDi_nearfield == pytest.approx(ref.totals.CDi_nearfield, rel=1e-6)


def test_body_wake_goethert_stretched_geometry():
    """Body-axis wake at M 0.5 agrees with the hand-stretched wing at M 0 times 1/beta.

    The wing is stretched along the body x-axis, so the body-axis wake of
    the stretched wing is its own stretched image. As in
    ``tests/test_goethert_geometry.py``, the solver stretches along the free
    stream, so the two sides differ by terms of order M^2 alpha^2; the
    tolerance is the one of that file (5e-3 at alpha up to 4 deg).
    """
    mach = 0.5
    pg = math.sqrt(1.0 - mach * mach)

    def wing(stretch):
        return vt.Aircraft(name="W", surfaces=[vt.LiftingSurface(
            name="W", semi_span=3.0, sections=[vt.WingSection(0.0, 1.25 * stretch), vt.WingSection(1.0, 0.75 * stretch)],
            sweep_le=math.atan(math.tan(math.radians(30.0)) * stretch))], ref_point=np.array([0.4 * stretch, 0.0, 0.0]))

    st = vt.SolverSettings(solver_type="vlm", n_panels=16, n_chord=4, wake_alignment="body")
    alpha = math.radians(2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_m = vt.analyze(wing(1.0), condition=vt.FlightCondition(V_inf=50.0, alpha=alpha, mach=mach), settings=st)
        res_0 = vt.analyze(wing(1.0 / pg), condition=vt.FlightCondition(V_inf=50.0, alpha=alpha), settings=st)
    for f in FIELDS:
        rule = float(getattr(res_0.totals, f)) / pg
        assert abs(rule - float(getattr(res_m.totals, f))) / abs(rule) < 5e-3, f


@pytest.mark.slow
def test_body_wake_mach_gradients_pass_the_check():
    """check_gradients passes with a body-axis wake at mach 0.4 (float64, rtol 1e-6)."""
    wing = vt.LiftingSurface(name="W", semi_span=4.0, sections=[
        vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.8)], sweep_le=np.radians(10.0))
    ac = vt.Aircraft(surfaces=[wing])
    cond = vt.FlightCondition(V_inf=50.0, alpha=np.radians(6.0), beta=np.radians(3.0), mach=0.4)
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=2, wake_alignment="body")
    old = gpu._precision
    try:
        gpu.set_precision("float64")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = vt.check_gradients(ac, cond, st, outputs=("CL", "CY", "CD", "Cl", "Cm", "Cn"),
                                     wrt=("alpha", "beta", "mach", "nodes"), n_directions=2, rtol=1e-6, seed=0)
    finally:
        gpu.set_precision(old)
    for group in ("alpha", "beta", "mach"):
        assert out["groups"][group]["compared"] > 0, out["groups"]
    failed = [k for k, v in out["details"].items() if v.get("compared") and not v.get("passed")]
    assert not failed, failed
    assert out["pass"], out["reasons"]


def test_body_wake_mach_gradient_reference_matches_production():
    """The gradient reference with a body-axis wake and mach reproduces the production solve."""
    wing = vt.LiftingSurface(name="W", semi_span=4.0, sections=[
        vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.8)])
    ac = vt.Aircraft(surfaces=[wing])
    cond = vt.FlightCondition(V_inf=50.0, alpha=np.radians(8.0), mach=0.4)
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=2, wake_alignment="body")
    old = gpu._precision
    try:
        gpu.set_precision("float64")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = vt.gradients(ac, cond, st, outputs=("CL", "CDi", "Cm"), wrt=("alpha", "mach"), method="reference")
    finally:
        gpu.set_precision(old)
    assert res.forward_mismatch <= 1e-12


@pytest.mark.gpu
def test_gpu_body_wake_with_mach_equals_cpu():
    """A GPU sweep with a body-axis wake and mach equals the CPU sweep (float64)."""
    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    old_dev, old_prec = gpu.get_device(), gpu._precision
    alphas = np.radians(np.linspace(-4.0, 12.0, 17))
    out = {}
    try:
        gpu.set_precision("float64")
        for dev in ("cpu", "gpu"):
            gpu.set_device(dev)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[dev] = vt.VortexLatticeSolver().solve_sweep(
                    _wing(), vt.FlightCondition(V_inf=50.0, beta=np.radians(3.0), mach=0.4), _settings("body"), alphas)
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)
    assert out["gpu"][0].details["dispatch"]["device"] != "cpu"
    for r_cpu, r_gpu in zip(out["cpu"], out["gpu"]):
        for f in ("CL", "CDi", "Cm", "Cl", "Cn", "CY"):
            a, b = float(getattr(r_cpu.totals, f)), float(getattr(r_gpu.totals, f))
            assert abs(a - b) <= 1e-10 * max(abs(a), 1e-6), (f, a, b)
