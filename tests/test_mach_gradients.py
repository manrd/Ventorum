# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Adjoint gradients with the Prandtl-Glauert compressibility correction."""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu


@contextlib.contextmanager
def _float64():
    """Set float64 gradient precision; restore the stored setting after the block."""
    old = gpu._precision
    try:
        gpu.set_precision("float64")
        yield
    finally:
        gpu.set_precision(old)


def _make_wing_tail():
    """Return a wing and tail aircraft with a flap control on the wing."""
    wing = vt.LiftingSurface(
        name="MainWing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.5),
            vt.WingSection(y_frac=1.0, chord=0.9),
        ],
        sweep_le=np.radians(15.0),
        dihedral=np.radians(2.0),
    )
    wing.controls = [
        vt.ControlSurface(
            name="flap",
            eta_start=0.2,
            eta_end=0.6,
            hinge_x_c=0.75,
            deflection=np.radians(5.0),
        )
    ]
    tail = vt.LiftingSurface(
        name="Tail",
        semi_span=1.8,
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.8),
            vt.WingSection(y_frac=1.0, chord=0.5),
        ],
        position=np.array([4.5, 0.0, 0.3]),
    )
    return vt.Aircraft(name="WingTail", surfaces=[wing, tail])


def _settings():
    """Return the solver settings of the Mach gradient checks."""
    return vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=2)


def _condition(mach: float):
    """Return the flight condition of the Mach gradient checks at *mach* [-].

    The case has sideslip, a control deflection and body rates, so all
    inputs of the scope move the result.
    """
    return vt.FlightCondition(
        V_inf=50.0,
        alpha=np.radians(4.0),
        beta=np.radians(2.0),
        mach=float(mach),
        p=0.05,
        q=0.03,
        r=0.04,
    )


@pytest.mark.slow
def test_mach_none_and_zero_gradients_equal() -> None:
    """Gradients at mach None and mach 0 agree bit for bit."""
    ac = _make_wing_tail()
    st = _settings()
    c_none = vt.FlightCondition(
        V_inf=50.0, alpha=np.radians(4.0), beta=np.radians(2.0), mach=None)
    c_zero = vt.FlightCondition(
        V_inf=50.0, alpha=np.radians(4.0), beta=np.radians(2.0), mach=0.0)
    with _float64():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = vt.gradients(ac, c_none, st, outputs=("CL", "CD", "Cm"),
                              wrt=("alpha", "beta", "nodes"))
            r2 = vt.gradients(ac, c_zero, st, outputs=("CL", "CD", "Cm"),
                              wrt=("alpha", "beta", "nodes"))
    for out in ("CL", "CD", "Cm"):
        assert r1.jacobian[out]["alpha"] == r2.jacobian[out]["alpha"]
        assert r1.jacobian[out]["beta"] == r2.jacobian[out]["beta"]
        for name in r1.jacobian[out]["nodes"]:
            for tag in ("le", "te"):
                assert np.array_equal(
                    r1.jacobian[out]["nodes"][name][tag],
                    r2.jacobian[out]["nodes"][name][tag])


def test_mach_zero_derivative_is_zero() -> None:
    """The mach derivative at mach 0 is 0 (the slope of beta at M = 0 is 0)."""
    ac = _make_wing_tail()
    st = _settings()
    cond = vt.FlightCondition(
        V_inf=50.0, alpha=np.radians(4.0), beta=np.radians(2.0), mach=0.0)
    with _float64():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = vt.gradients(ac, cond, st, outputs=("CL", "CD", "Cm"),
                               wrt=("mach",))
    for out in ("CL", "CD", "Cm"):
        assert float(res.jacobian[out]["mach"]) == 0.0


def test_mach_kernels_match_reference() -> None:
    """The kernel path and the reference path agree at mach 0.3 to 1e-12."""
    ac = _make_wing_tail()
    st = _settings()
    cond = _condition(0.3)
    outputs = ("CL", "CD", "Cm")
    wrt = ("alpha", "beta", "mach", "controls")
    with _float64():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_ref = vt.gradients(ac, cond, st, outputs=outputs, wrt=wrt,
                                 method="reference")
            r_ker = vt.gradients(ac, cond, st, outputs=outputs, wrt=wrt,
                                 method="kernels")
    assert r_ref.forward_mismatch <= 1e-12
    assert r_ker.forward_mismatch <= 1e-12
    for out in outputs:
        for key in ("alpha", "beta", "mach"):
            assert abs(float(r_ref.jacobian[out][key])
                       - float(r_ker.jacobian[out][key])) <= 1e-12 * max(
                           1.0, abs(float(r_ref.jacobian[out][key])))
        for name in r_ref.jacobian[out]["controls"]:
            assert abs(float(r_ref.jacobian[out]["controls"][name])
                       - float(r_ker.jacobian[out]["controls"][name])) <= 1e-12


@pytest.mark.slow
@pytest.mark.parametrize("mach", [0.3, 0.6])
def test_check_mach_free_air(mach: float) -> None:
    """check_gradients passes at mach 0.3 and 0.6 for all free-air inputs.

    Wing and tail with sideslip, a flap control and body rates, float64,
    rtol 1e-6. The planform input has its own test (it cannot combine with
    nodes).
    """
    ac = _make_wing_tail()
    st = _settings()
    cond = _condition(mach)
    outputs = ("CL", "CY", "CD", "Cl", "Cm", "Cn")
    wrt = ("alpha", "beta", "V_inf", "rho", "phi", "p", "q", "r",
           "nodes", "sections", "reference", "controls", "mach")
    with _float64():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = vt.check_gradients(ac, cond, st, outputs=outputs, wrt=wrt,
                                     n_directions=2, rtol=1e-6, seed=0)
    assert out["compared"] > 0
    for group, counts in out["groups"].items():
        assert counts["compared"] > 0, (mach, group, out["groups"])
    failed = [k for k, v in out["details"].items()
              if v.get("compared") and not v.get("passed")]
    assert not failed, (mach, failed)
    assert out["pass"], (mach, out["reasons"])


@pytest.mark.slow
@pytest.mark.parametrize("mach", [0.3, 0.6])
def test_check_mach_planform(mach: float) -> None:
    """check_gradients passes at mach 0.3 and 0.6 for the planform input.

    The planform input is refused with a deflected chordwise control (the
    width of the control joins is not in the gradient chain), so this check
    sets the flap to zero deflection.
    """
    ac = _make_wing_tail()
    ac.set_deflection("flap", 0.0)
    st = _settings()
    cond = _condition(mach)
    with _float64():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = vt.check_gradients(ac, cond, st, outputs=("CL", "CD", "Cm"),
                                     wrt=("planform", "mach"), rtol=1e-6)
    assert out["groups"]["planform"]["compared"] > 0
    assert out["groups"]["mach"]["compared"] > 0
    failed = [k for k, v in out["details"].items()
              if v.get("compared") and not v.get("passed")]
    assert not failed, (mach, failed)
    assert out["pass"], (mach, out["reasons"])


def test_mach_gradients_gpu_equal_cpu():
    """Mach gradients on the GPU device equal the CPU gradients to round-off (float64)."""
    import warnings

    import numpy as np

    import ventorum as vt
    from ventorum import gpu

    if not gpu.available():
        import pytest

        pytest.skip("no GPU")
    wing = vt.LiftingSurface(name="W", semi_span=4.0, sections=[
        vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.8)])
    ac = vt.Aircraft(surfaces=[wing])
    cond = vt.FlightCondition(V_inf=40.0, alpha=np.radians(4.0), beta=np.radians(2.0), mach=0.5)
    st = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=3)
    old_dev, old_prec = gpu.get_device(), gpu.get_precision()
    out = {}
    try:
        gpu.set_precision("float64")
        for dev in ("cpu", "gpu"):
            gpu.set_device(dev)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[dev] = vt.gradients(ac, cond, st, outputs=("CL", "CDi", "Cm"), wrt=("alpha", "beta", "mach"))
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)
    assert out["gpu"].device != "cpu"
    for o in ("CL", "CDi", "Cm"):
        for w in ("alpha", "beta", "mach"):
            a = float(np.asarray(out["cpu"].jacobian[o][w]))
            b = float(np.asarray(out["gpu"].jacobian[o][w]))
            assert abs(a - b) <= 1e-10 * max(abs(a), 1e-12), (o, w, a, b)
