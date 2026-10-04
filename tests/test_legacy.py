# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the original solvers and acceleration paths (``ventorum.legacy``).

The legacy package is kept to study and port its optimisations. These tests
check that it imports and runs, that its acceleration paths agree with each
other, and how its results compare with the verified core. The known
physics defects are written as strict expected failures: when a defect is
corrected, its test passes, pytest reports it, and the marker is removed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import ventorum as vt
import ventorum.legacy as legacy
from ventorum.legacy.aero.acceleration import acceleration_context
from ventorum.legacy.aero.cython_accel import cython_kernels

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _rect(lib, alpha_L0=0.0):
    af = lib.LinearAirfoil(alpha_L0=alpha_L0)
    return lib.LiftingSurface(
        semi_span=5.0,
        sections=[lib.WingSection(y_frac=0.0, chord=1.25, airfoil=af), lib.WingSection(y_frac=1.0, chord=1.25, airfoil=af)],
    )


def _legacy(solver, alpha_L0=0.0, **kw):
    return legacy.analyze(_rect(legacy, alpha_L0), alpha_deg=5.0, V_inf=25.0, solver=solver, n_panels=80, **kw).totals


def _core(solver, alpha_L0=0.0):
    return vt.analyze(_rect(vt, alpha_L0), alpha_deg=5.0, V_inf=25.0, solver=solver, n_panels=80).totals


@pytest.mark.parametrize("solver", ["linear", "fourier"])
def test_legacy_linear_and_fourier_agree_with_the_verified_core(solver):
    """The original linear lifting line and Fourier solver agree with theory within about 1 %."""
    old, new = _legacy(solver), _core(solver)
    assert old.CL == pytest.approx(new.CL, rel=0.01)
    assert old.CDi == pytest.approx(new.CDi, rel=0.015)


@pytest.mark.parametrize("mode", ["numpy", "numba"] + (["cython", "hybrid"] if cython_kernels is not None else []))
def test_legacy_acceleration_paths_give_the_same_result(mode):
    """All acceleration paths of the original code solve the same equations."""
    with acceleration_context("numpy"):
        ref = _legacy("horseshoe")
    with acceleration_context(mode):
        t = _legacy("horseshoe")
    assert t.CL == pytest.approx(ref.CL, rel=1e-10)
    assert t.CDi == pytest.approx(ref.CDi, rel=1e-10)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_legacy_gpu_solver_matches_cpu_solver():
    """The batched PyTorch solver gives the CPU result (on a GPU when one is present, else on the CPU)."""
    cpu = _legacy("horseshoe")
    gpu = _legacy("gpu_horseshoe")
    assert gpu.CL == pytest.approx(cpu.CL, rel=1e-9)


@pytest.mark.xfail(strict=True, reason="Known defect: the original horseshoe solver ignores camber (alpha_L0)")
def test_legacy_horseshoe_uses_camber():
    assert _legacy("horseshoe", alpha_L0=np.radians(-3.0)).CL > _legacy("horseshoe").CL + 0.1


@pytest.mark.xfail(strict=True, reason="Known defect: the original horseshoe solver over-predicts induced drag")
def test_legacy_horseshoe_induced_drag():
    assert _legacy("horseshoe").CDi == pytest.approx(_core("linear").CDi, rel=0.05)
