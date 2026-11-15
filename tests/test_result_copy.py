# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of copy.copy, copy.deepcopy and to_dict of a SolverResult.

A copy shares the solver objects in ``details`` (the cached lattice, the
loads and the ground plane) and copies all other data. After a GPU solve the
cached lattice holds device data that cannot be copied, so a copy that
copied the lattice failed.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu


def _aircraft(semi_span: float) -> vt.Aircraft:
    af = vt.LinearAirfoil()
    wing = vt.LiftingSurface(
        name="wing", semi_span=semi_span,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0, airfoil=af),
                  vt.WingSection(y_frac=1.0, chord=0.8, airfoil=af)],
    )
    return vt.Aircraft(name="copy test", surfaces=[wing])


def _solve(device: str, semi_span: float, h: float | None = None) -> vt.SolverResult:
    old = gpu.get_device()
    try:
        gpu.set_device(device)
        cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=h)
        st = vt.SolverSettings(solver_type="vlm", n_panels=40, n_chord=8)
        return vt.analyze(_aircraft(semi_span), cond, st)
    finally:
        gpu.set_device(old)


def _check_copies(res: vt.SolverResult) -> None:
    for copier in (copy.copy, copy.deepcopy, vt.SolverResult.clone):
        out = copier(res)
        # The solver objects stay shared.
        for key in ("lattice", "loads", "ground"):
            assert out.details[key] is res.details[key]
        # All other data is copied.
        assert out.details["notes"] is not res.details["notes"]
        assert out.details["gamma"] is not res.details["gamma"]
        ref_cl = res.spanwise[0].Cl.copy()
        ref_gamma = np.array(res.details["gamma"], copy=True)
        ref_warnings = list(res.totals.trust.warnings)
        out.spanwise[0].Cl[:] = 99.0
        out.details["gamma"][...] = 99.0
        out.details["notes"].append("added on the copy")
        out.totals.trust.warnings.append("added on the copy")
        out.totals.CL = -1.0
        np.testing.assert_array_equal(res.spanwise[0].Cl, ref_cl)
        np.testing.assert_array_equal(res.details["gamma"], ref_gamma)
        assert "added on the copy" not in res.details["notes"]
        assert res.totals.trust.warnings == ref_warnings
        assert res.totals.CL != -1.0


@pytest.mark.parametrize("h", [None, 1.5])
def test_copy_of_a_cpu_result_shares_the_lattice(h):
    res = _solve("cpu", semi_span=4.1, h=h)
    assert res.details.get("device", "cpu") == "cpu"
    if h is not None:
        assert res.details["ground"] is not None
    _check_copies(res)


@pytest.mark.gpu
def test_copy_of_a_gpu_result_shares_the_lattice():
    if not gpu.available():
        pytest.skip(gpu.unavailable_reason())
    res = _solve("gpu", semi_span=4.3)
    assert res.details["device"] == "gpu"
    _check_copies(res)


def test_copy_of_a_result_that_refers_to_itself():
    """A result whose details hold the result itself (the list of branches) can be copied."""
    res = vt.SolverResult(details={"notes": ["a"]})
    res.details["branches"] = [res, vt.SolverResult()]
    for out in (copy.copy(res), copy.deepcopy(res), res.clone()):
        assert out.details["branches"][0] is out
        assert out.details["branches"][1] is not res.details["branches"][1]

