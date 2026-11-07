# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the float64 boundary-condition check of the float32 adjoint gradients.

With float32 gradients (or a float32 production solve), the boundary-condition
check in the system of the reference copy is coarse (limit 3e-2): the float32
data of that system limit it. The float64 check assembles the system once in
float64, only for the check, and computes the backward error of the
production circulation in it. Its limit comes from the round-off floor of the
production state.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.sensitivity import adjoint


@contextlib.contextmanager
def _gpu_settings(device: str | None = None, precision: str | None = None):
    """Set the device and the precision; restore the raw stored settings after the block."""
    old_dev, old_prec = gpu.get_device(), gpu._precision
    try:
        if device is not None:
            gpu.set_device(device)
        if precision is not None:
            gpu.set_precision(precision)
        yield
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)


def _section() -> vt.LinearAirfoil:
    return vt.LinearAirfoil(name="Sec", a0=2.0 * np.pi, alpha_L0=np.radians(-2.0), Cd0=0.008, Cm0=-0.05)


def _aircraft() -> vt.Aircraft:
    """A tapered wing with dihedral and a flap."""
    w = vt.LiftingSurface(
        name="Wing", semi_span=5.0, dihedral=np.radians(3.0),
        sections=[vt.WingSection(y_frac=0.0, chord=2.0, airfoil=_section()),
                  vt.WingSection(y_frac=1.0, chord=1.0, airfoil=_section())],
        controls=[vt.ControlSurface(name="flap", eta_start=0.2, eta_end=0.55, hinge_x_c=0.75,
                                    deflection=np.radians(5.0), symmetric=True)])
    return vt.Aircraft(name="Wing", surfaces=[w])


_COND = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(2.0))
_GROUND = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(2.0), h=1.0)

_SETTINGS = {
    "vlm": vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=4),
    "linear": vt.SolverSettings(solver_type="linear", n_panels=10),
    "nonlinear": vt.SolverSettings(solver_type="nonlinear", n_panels=10, tolerance=1e-12),
}


def _prep(solver: str, cond: vt.FlightCondition) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return adjoint._prepare_case(_aircraft(), cond, _SETTINGS[solver], "body", ("alpha",), ("CL",))


def _coarse(prep: dict, g_unknown: np.ndarray) -> float:
    """Return the backward error of the coarse check (system of the reference copy) for a state."""
    import torch

    model = prep["model"]
    t = adjoint.make_input_tensors(model, prep["base"], (), prep["device"], prep["precision"], prep)
    g = torch.as_tensor(g_unknown, dtype=model.b["dtype"], device=model.b["device"])
    with torch.no_grad():
        _, A, rhs = model.forward(t, g)
    return adjoint._residual_check(prep, A, rhs, g)


def _perturbed_state(prep: dict) -> np.ndarray:
    """Return the production state with the error of a 0.01 deg angle in one boundary condition.

    The error in row 3 of the float64 system is ``0.01 deg * max|rhs|``; the
    state that solves the system with this error differs from the production
    state by ``A^-1`` times it.
    """
    A, rhs, g = adjoint.float64_check_system(prep)
    e = np.zeros(A.shape[0])
    e[3] = np.radians(0.01) * float(np.max(np.abs(rhs.cpu().numpy())))
    dg = np.linalg.solve(A.cpu().numpy(), e)
    return np.asarray(prep["g_unknown"], dtype=float) + dg


@pytest.mark.parametrize("solver, ground", [("vlm", False), ("vlm", True), ("linear", False), ("nonlinear", False)])
def test_float64_check_passes_float32_gradients_and_reports_both(solver, ground):
    """Verify that float32 gradients pass the float64 check and report both backward errors."""
    cond = _GROUND if ground else _COND
    with _gpu_settings(device="cpu", precision="float32"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = vt.gradients(_aircraft(), cond, _SETTINGS[solver], outputs=("CL",), wrt=("alpha",))
        prep = _prep(solver, cond)
        omega = adjoint._residual_check_float64(prep)
    assert res.precision_gradient == "float32"
    assert omega <= adjoint._float64_check_limit(prep)
    assert any("Boundary-condition checks" in n and "float64 assembly" in n for n in res.notes)


@pytest.mark.parametrize("solver", ["vlm", "linear", "nonlinear"])
def test_float64_check_refuses_a_perturbed_state_that_the_coarse_check_accepts(solver):
    """Verify that a state with a 0.01 deg boundary-condition error fails the float64 check.

    The coarse float32 check accepts the same state: its float32 data limit it.
    """
    with _gpu_settings(device="cpu", precision="float32"):
        prep = _prep(solver, _COND)
        bad = _perturbed_state(prep)
        assert _coarse(prep, bad) <= adjoint._RESIDUAL_TOL_FLOAT32
        prep["g_unknown"] = bad
        with pytest.raises(RuntimeError, match="float64 system"):
            adjoint._residual_check_float64(prep)


def test_float64_check_refuses_a_wrong_right_hand_side():
    """Verify that the float64 check refuses a 0.01 deg error in one right-hand side entry."""
    with _gpu_settings(device="cpu", precision="float32"):
        prep = _prep("vlm", _COND)
        A, rhs, g = adjoint.float64_check_system(prep)
        assert adjoint._residual_check_float64(prep, (A, rhs, g)) <= adjoint._float64_check_limit(prep)
        bad = rhs.clone()
        bad[3] += float(np.radians(0.01)) * float(bad.abs().max())
        with pytest.raises(RuntimeError, match="float64 system"):
            adjoint._residual_check_float64(prep, (A, bad, g))


def test_float64_gradients_keep_one_check():
    """Verify that float64 gradients of a float64 solve run only the float64 check of the reference copy."""
    with _gpu_settings(device="cpu", precision="float64"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = vt.gradients(_aircraft(), _COND, _SETTINGS["vlm"], outputs=("CL",), wrt=("alpha",))
    assert any("Boundary-condition check:" in n for n in res.notes)
    assert not any("float64 assembly" in n for n in res.notes)


@pytest.mark.gpu
@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_float64_check_on_the_gpu_with_a_float32_solve(solver):
    """Verify the float64 check of a float32 GPU solve: the correct state passes, a perturbed state fails."""
    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    with _gpu_settings(device="gpu", precision="float32"):
        prep = _prep(solver, _COND)
        assert prep["precision_primal"] == "float32"
        assert prep["device"] == "cuda"
        omega = adjoint._residual_check_float64(prep)
        assert omega <= adjoint._float64_check_limit(prep)
        prep["g_unknown"] = _perturbed_state(prep)
        with pytest.raises(RuntimeError, match="float64 system"):
            adjoint._residual_check_float64(prep)
