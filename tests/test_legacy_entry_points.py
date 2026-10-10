# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""The legacy entry points run on the legacy solvers (recovery audit).

After the move into ``ventorum.legacy``, some legacy modules imported the
entry points of the verified core (``ventorum.analyze``) and passed legacy
data classes to them, so they failed. These smoke tests run each of them on
a small case.
"""

from __future__ import annotations

import numpy as np

import ventorum.legacy as legacy
from ventorum.legacy.geometry.mesh_convergence import run_mesh_convergence_study
from ventorum.legacy.hardware import scan_hardware
from ventorum.legacy.hardware.tuner import benchmark_cpu_worker_scaling
from ventorum.legacy.instance import Ventorum
from ventorum.legacy.utils import benchmark as bm


def _wing():
    return legacy.LiftingSurface(semi_span=5.0, sections=[legacy.WingSection(0.0, 1.0), legacy.WingSection(1.0, 1.0)])


def test_legacy_performance_benchmark_runs():
    out = bm.run_performance_benchmark(panel_counts=(10,), solvers=("horseshoe",), n_repeats=1,
                                       verbose=False, show_progress=False)
    assert out


def test_legacy_sweep_benchmark_runs():
    out = bm.run_sweep_benchmark(alpha_range=(-2.0, 6.0, 3), solvers=("horseshoe",), n_panels=10, n_repeats=1,
                                 verbose=False, show_progress=False)
    assert out


def test_legacy_instance_single_point_run():
    inst = Ventorum("rect", geometry=_wing(), n_panels=12, alpha_deg=4.0)
    res = inst.analyze()
    ref = legacy.analyze(_wing(), alpha_deg=4.0, n_panels=12).totals
    assert np.isclose(res.totals.CL, ref.CL, rtol=1e-12)


def test_legacy_mesh_convergence_study_runs():
    res = run_mesh_convergence_study(_wing(), spacing_schemes=("cosine",), panel_counts=[8, 16],
                                     ref_n_panels=32, evaluate_sweep=False)
    assert res is not None


def test_legacy_tuner_worker_scaling_runs():
    out = benchmark_cpu_worker_scaling(scan_hardware(), quick=True, show_progress=False)
    assert isinstance(out, dict)
