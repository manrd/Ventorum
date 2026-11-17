# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of copy.copy and copy.deepcopy of 3D panel results (CPU and GPU).

A panel result keeps ``details["mesh"]`` (the panel mesh, which the solver
cache can share between results) and ``details["panel_result"]`` (the
panel solution arrays). Both hold numpy arrays of the size of the mesh
only: after a GPU solve the device arrays stay in the cached solver, not
in the result. A copy therefore deep-copies them; it must work, keep the
values, and share no mutable data with the original.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import ventorum as vt
from ventorum.solvers.factory import make_solver


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _wing() -> vt.Aircraft:
    surf = vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0, shape="0012"),
            vt.WingSection(y_frac=1.0, chord=1.0, shape="0012"),
        ],
        is_symmetric=True,
    )
    ac = vt.Aircraft(name="RectWing", surfaces=[surf])
    ac.compute_reference_values()
    return ac


def _solve(backend: str, h: float | None) -> vt.SolverResult:
    settings = vt.SolverSettings(solver_type="panel3d", n_panels=8, n_chord=8, panel_backend=backend)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=h)
    return make_solver("panel3d").solve(_wing(), cond, settings)


def _device_objects(obj, seen: set[int] | None = None, depth: int = 0) -> list[str]:
    """Return the type names of PyTorch or Warp objects found in *obj*."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 6 or isinstance(obj, (str, bytes, int, float, np.ndarray)):
        return []
    seen.add(id(obj))
    if type(obj).__module__.startswith(("torch", "warp")):
        return [type(obj).__name__]
    found: list[str] = []
    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, (list, tuple)):
        children = list(obj)
    else:
        names = list(getattr(obj, "__dict__", {})) + list(getattr(type(obj), "__slots__", ()))
        children = [getattr(obj, n) for n in names if hasattr(obj, n)]
    for child in children:
        found += _device_objects(child, seen, depth + 1)
    return found


def _check_copies(res: vt.SolverResult) -> None:
    assert _device_objects(res.details) == []
    for make_copy in (copy.copy, copy.deepcopy):
        dup = make_copy(res)
        assert dup.totals.CL == res.totals.CL
        assert dup.totals.CDi == res.totals.CDi
        assert dup.details["mesh"] is not res.details["mesh"]
        assert dup.details["panel_result"] is not res.details["panel_result"]
        assert np.array_equal(dup.details["mesh"].nodes, res.details["mesh"].nodes)
        assert np.array_equal(dup.details["panel_result"].cp, res.details["panel_result"].cp)
        # The copy shares no mutable data with the original.
        saved = res.details["panel_result"].cp.copy()
        dup.details["panel_result"].cp[...] = 0.0
        dup.details["notes"].append("copy only")
        assert np.array_equal(res.details["panel_result"].cp, saved)
        assert "copy only" not in res.details["notes"]


@pytest.mark.parametrize("h", [None, 0.5])
def test_copy_of_a_cpu_panel_result(h):
    """copy.copy and copy.deepcopy of a CPU panel result work and are independent."""
    _check_copies(_solve("numpy", h))


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("h", [None, 0.5])
def test_copy_of_a_gpu_panel_result(h):
    """A GPU panel result holds no device object in details; its copies work and are independent."""
    res = _solve("gpu", h)
    assert res.details["dispatch"].get("device") == "gpu"
    _check_copies(res)
