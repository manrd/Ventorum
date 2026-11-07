# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Section lift coefficient of the lattice strips with body rates.

With body rates the onset velocity changes along the span, so the section
lift coefficient has two possible normalisations:

* ``SpanwiseResult.Cl``: the local dynamic pressure,
  ``Cl = 2 Gamma / (|U_qc| c)``, with ``U_qc`` the local onset (free stream
  plus body-rate velocity, no induced velocity) at the quarter-chord
  mid-point of the strip. The effective angle ``alpha_eff = alpha_L0 + Cl /
  a0`` of the vortex lattice and of the linear lifting line uses this
  value, and so do the section polar ``Cd(alpha_eff)`` and
  ``Cm(alpha_eff)``.
* ``SpanwiseResult.Cl_freestream``: the free-stream dynamic pressure,
  ``2 Gamma / (V_inf c)``.

Without rates the two are equal.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.aero.system import freestream_direction, rotation_velocity
from ventorum.solvers.factory import make_solver


@pytest.fixture(autouse=True)
def _cpu():
    """Run on the CPU; restore the device and the precision after the test."""
    old_dev, old_prec = gpu.get_device(), gpu._precision
    gpu.set_device("cpu")
    yield
    gpu.set_device(old_dev)
    gpu.set_precision(old_prec)


def _polar() -> vt.TabulatedAirfoil:
    """Cambered polar with a drag and a moment that change with the angle (made up)."""
    a = np.radians(np.arange(-10.0, 21.0, 1.0))
    return vt.TabulatedAirfoil(name="RateTab", alpha=a, Cl_data=0.95 * 2.0 * np.pi * (a + np.radians(2.0)),
                               Cd_data=0.008 + 0.02 * a ** 2, Cm_data=-0.05 + 0.03 * a)


def _tapered(airfoil=None) -> vt.Aircraft:
    """Tapered wing, span 10 m, root chord 1.5 m, tip chord 0.75 m."""
    af = airfoil if airfoil is not None else _polar()
    wing = vt.LiftingSurface(
        name="Wing", semi_span=5.0, sweep_le=np.radians(5.0),
        sections=[vt.WingSection(y_frac=0.0, chord=1.5, airfoil=af),
                  vt.WingSection(y_frac=1.0, chord=0.75, airfoil=af)])
    ac = vt.Aircraft(surfaces=[wing])
    ac.compute_reference_values()
    return ac


_V = 30.0
#: Nondimensional rates p b / (2 V) = 0.1 and r b / (2 V) = 0.1 (b = 10 m).
_RATES = {"roll": {"p": 0.1 * 2.0 * _V / 10.0}, "yaw": {"r": 0.1 * 2.0 * _V / 10.0}}


def _solve(solver: str, ac: vt.Aircraft, cond: vt.FlightCondition):
    n_chord = 4 if solver == "vlm" else 1
    st = vt.SolverSettings(solver_type=solver, n_panels=16, n_chord=n_chord, tolerance=1e-12)
    s = make_solver(solver)
    lattice = s.build(ac, st, cond, None, ac.ref_point)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = s.solve_batch(lattice, [cond], st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=ac.ref_point,
                            _cpu_only=True)[0]
    return lattice, res


def _strip(lattice, res, name: str) -> np.ndarray:
    """Return the spanwise array *name* in lattice strip order."""
    out = np.empty(lattice.n_strips)
    for surf, sw in zip(lattice.surfaces, res.spanwise):
        out[surf.strips] = getattr(sw, name)
    return out


@pytest.mark.parametrize("solver", ["vlm", "linear", "nonlinear"])
@pytest.mark.parametrize("rate", list(_RATES))
def test_rated_strip_cl_uses_the_local_onset(solver, rate):
    """Cl = 2 Gamma / (|U_qc| c) and Cl_freestream = 2 Gamma / (V_inf c) with a body rate."""
    ac = _tapered()
    cond = vt.FlightCondition(V_inf=_V, alpha=np.radians(4.0), **_RATES[rate])
    lattice, res = _solve(solver, ac, cond)
    gamma = _strip(lattice, res, "gamma")
    u_qc = (cond.V_inf * freestream_direction(cond.alpha, cond.beta)[None, :]
            + rotation_velocity(cond, lattice.qc_mid, ac.ref_point))
    local = 2.0 * gamma / (np.linalg.norm(u_qc, axis=1) * lattice.chord)
    free = 2.0 * gamma / (cond.V_inf * lattice.chord)
    np.testing.assert_allclose(_strip(lattice, res, "Cl"), local, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(_strip(lattice, res, "Cl_freestream"), free, rtol=1e-12, atol=1e-14)
    # The two normalisations differ by more than round-off on this case.
    assert np.max(np.abs(local - free)) > 1e-4


@pytest.mark.parametrize("solver", ["vlm", "linear"])
@pytest.mark.parametrize("rate", list(_RATES))
def test_alpha_eff_uses_the_local_strip_cl(solver, rate):
    """alpha_eff = alpha_L0 + Cl / a0 with the local Cl (vortex lattice and linear lifting line)."""
    ac = _tapered()
    cond = vt.FlightCondition(V_inf=_V, alpha=np.radians(4.0), **_RATES[rate])
    lattice, res = _solve(solver, ac, cond)
    cl = _strip(lattice, res, "Cl")
    expected = lattice.alpha_L0 + cl / np.maximum(lattice.a0, 1e-9)
    np.testing.assert_allclose(_strip(lattice, res, "alpha_eff"), expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("solver", ["vlm", "linear", "nonlinear"])
def test_without_rates_the_two_normalisations_are_equal(solver):
    """Without body rates Cl and Cl_freestream are the same array, 2 Gamma / (V_inf c)."""
    ac = _tapered()
    cond = vt.FlightCondition(V_inf=_V, alpha=np.radians(4.0))
    lattice, res = _solve(solver, ac, cond)
    cl = _strip(lattice, res, "Cl")
    np.testing.assert_array_equal(cl, _strip(lattice, res, "Cl_freestream"))
    np.testing.assert_allclose(cl, 2.0 * _strip(lattice, res, "gamma") / (_V * lattice.chord), rtol=1e-14)


def test_fourier_fills_cl_freestream():
    """The Fourier solver (free stream only) gives Cl_freestream equal to Cl."""
    af = vt.LinearAirfoil(a0=2.0 * np.pi, Cd0=0.008)
    wing = vt.LiftingSurface(name="Wing", semi_span=5.0,
                             sections=[vt.WingSection(y_frac=0.0, chord=1.0, x_le=-0.25, airfoil=af),
                                       vt.WingSection(y_frac=1.0, chord=1.0, x_le=-0.25, airfoil=af)])
    res = vt.analyze(vt.Aircraft(surfaces=[wing]), vt.FlightCondition(V_inf=_V, alpha=np.radians(4.0)),
                     vt.SolverSettings(solver_type="fourier", n_panels=16))
    for sw in res.spanwise:
        np.testing.assert_array_equal(sw.Cl_freestream, sw.Cl)


def test_cl_freestream_is_serialised_and_cloned():
    """to_dict, the agent spanwise payload and clone carry Cl_freestream."""
    from ventorum.agent.response import spanwise_payload

    cond = vt.FlightCondition(V_inf=_V, alpha=np.radians(4.0), **_RATES["yaw"])
    _, res = _solve("vlm", _tapered(), cond)
    d = res.to_dict(detail_level="standard")
    for sw, item in zip(res.spanwise, d["spanwise"]):
        assert item["Cl_freestream"] == pytest.approx(sw.Cl_freestream.tolist(), rel=1e-15)
    for sw, item in zip(res.spanwise, spanwise_payload(res)):
        assert item["Cl_freestream"] == pytest.approx(sw.Cl_freestream.tolist(), abs=1e-5)
    sw = res.spanwise[0]
    c = sw.clone()
    np.testing.assert_array_equal(c.Cl_freestream, sw.Cl_freestream)
    assert c.Cl_freestream is not sw.Cl_freestream


@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_profile_drag_uses_the_local_alpha_eff(solver):
    """The section polar is read at the alpha_eff of the local Cl (yaw rate, tabulated polar)."""
    from ventorum.geometry.lattice import section_cd

    ac = _tapered()
    cond = vt.FlightCondition(V_inf=_V, alpha=np.radians(4.0), **_RATES["yaw"])
    lattice, res = _solve(solver, ac, cond)
    cl = _strip(lattice, res, "Cl")
    ae = lattice.alpha_L0 + cl / np.maximum(lattice.a0, 1e-9)
    np.testing.assert_allclose(_strip(lattice, res, "Cd_profile"), section_cd(_polar(), ae), rtol=1e-12)


# ---------------------------------------------------------------------------
# GPU parity
# ---------------------------------------------------------------------------

def _need_gpu():
    if not gpu.available():
        pytest.skip(f"GPU pipeline unavailable: {gpu.unavailable_reason()}")


@pytest.mark.gpu
@pytest.mark.parametrize("precision,tol", [("float64", 1e-10), ("float32", 1e-4)])
@pytest.mark.parametrize("solver", ["vlm", "linear"])
@pytest.mark.parametrize("rate", list(_RATES))
def test_gpu_strip_cl_matches_cpu(solver, rate, precision, tol):
    """GPU and CPU give the same Cl, Cl_freestream, alpha_eff, Cd_profile and totals with a body rate."""
    _need_gpu()
    ac = _tapered()
    cond = vt.FlightCondition(V_inf=_V, alpha=np.radians(4.0), **_RATES[rate])
    st = vt.SolverSettings(solver_type=solver, n_panels=16, n_chord=4 if solver == "vlm" else 1)
    s = make_solver(solver)
    lattice = s.build(ac, st, cond, None, ac.ref_point)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpu = s.solve_batch(lattice, [cond], st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=ac.ref_point,
                            _cpu_only=True)[0]
        gpu.set_device("gpu")
        gpu.set_precision(precision)
        res = s.solve_batch(lattice, [cond], st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=ac.ref_point)[0]
    assert res.details["device"] == "gpu"
    for name in ("Cl", "Cl_freestream", "alpha_eff", "Cd_profile", "Cm_section"):
        a, b = _strip(lattice, cpu, name), _strip(lattice, res, name)
        scale = max(float(np.max(np.abs(a))), 1e-6)
        assert np.max(np.abs(a - b)) <= tol * scale, (name, np.max(np.abs(a - b)), scale)
    cl_scale = max(abs(cpu.totals.CL), 1e-3)
    for name in ("CL", "CD_total", "Cl", "Cm", "Cn"):
        a, b = getattr(cpu.totals, name), getattr(res.totals, name)
        assert abs(a - b) <= tol * max(abs(a), cl_scale), (name, a, b)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

@pytest.fixture
def float64_cpu():
    """Float64 gradients on the CPU."""
    gpu.set_precision("float64")
    yield


@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_gradients_with_rates_and_a_tabulated_polar(solver, float64_cpu):
    """check_gradients passes for CD, Cm, Cn with body rates (the outputs that depend on Cl_strip).

    The forward check of the gradients also compares the reference copy
    with the production totals and with the production section Cl to
    1e-10: the three copies of the strip Cl (CPU loads, GPU loads,
    gradient reference) must use the same normalisation.
    """
    ac = _tapered()
    cond = vt.FlightCondition(V_inf=_V, alpha=np.radians(4.0), p=0.4, r=-0.5)
    st = vt.SolverSettings(solver_type=solver, n_panels=10, n_chord=2 if solver == "vlm" else 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CD", "Cm", "Cn"), wrt=("alpha", "p", "r"),
                                 rtol=1e-6, seed=0)
        n = sum(sw.Cl.size for sw in vt.analyze(ac, cond, st).spanwise)
        # The forward check raises RuntimeError when the reference section
        # Cl differs from the production section Cl.
        vjp = vt.gradient_vjp(ac, cond, st, seeds={"section_cl": np.ones(n)}, wrt=("alpha", "r"))
    bad = {k: v for k, v in out["details"].items() if v["compared"] and not v["passed"]}
    assert out["pass"], (out["reasons"], bad)
    assert np.isfinite(float(vjp["r"]))

