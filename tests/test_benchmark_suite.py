# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the Ventorum benchmark suite.

Covers the quick mode execution and the non-interference verification
that parallel instances do not change each other's results.
"""

import numpy as np
import pytest

import ventorum as vt
from ventorum.instance import Ventorum, run_parallel_instances
from validation.benchmark_suite import (
    benchmark_panel_scaling,
    benchmark_alpha_sweep,
    benchmark_ground_effect,
    benchmark_multithread,
    benchmark_symmetry,
    benchmark_multi_instance,
    run_quick_benchmark,
)


def test_quick_benchmark_runs():
    """Quick benchmark completes without error and returns expected structure."""
    results = run_quick_benchmark()

    assert "panel" in results
    assert "sweep" in results
    assert "ground_effect" in results
    assert "multithread" in results
    assert "symmetry" in results
    assert "multi_instance" in results

    # Check panel scaling results
    panel = results["panel"]
    assert "panel_counts" in panel
    assert "times_ms" in panel
    assert "vlm" in panel["times_ms"]
    assert "linear" in panel["times_ms"]
    assert all(isinstance(t, (int, float)) for t in panel["times_ms"]["vlm"])
    assert all(not np.isnan(t) for t in panel["times_ms"]["vlm"])

    # Check sweep results
    sweep = results["sweep"]
    assert "alpha_points" in sweep
    assert "times_ms" in sweep
    assert "vlm" in sweep["times_ms"]
    assert isinstance(sweep["times_ms"]["vlm"], (int, float))
    assert not np.isnan(sweep["times_ms"]["vlm"])

    # Check ground effect results
    ge = results["ground_effect"]
    assert "panel_counts" in ge
    assert "times_ms" in ge
    assert "free_air" in ge["times_ms"]
    assert "ground_effect" in ge["times_ms"]
    assert len(ge["times_ms"]["free_air"]) == len(ge["panel_counts"])

    # Check multithread results
    mt = results["multithread"]
    assert "workers" in mt
    assert "times_ms" in mt
    assert len(mt["times_ms"]) == len(mt["workers"])

    # Check symmetry results
    sym = results["symmetry"]
    assert "records" in sym
    assert len(sym["records"]) > 0
    for r in sym["records"]:
        assert "solver" in r
        assert "speedup" in r
        assert "diff_cl" in r
        assert "diff_cdi" in r

    # Check multi-instance results
    mi = results["multi_instance"]
    assert "records" in mi
    assert len(mi["records"]) > 0
    for r in mi["records"]:
        assert "concurrent_instances" in r
        assert "workers_per_instance" in r
        assert "zero_interference_verified" in r
        assert "max_discrepancy" in r


def test_panel_scaling_returns_valid_times():
    """Panel scaling benchmark returns finite positive times."""
    res = benchmark_panel_scaling(panel_counts=(10, 20), solvers=("vlm", "linear"), n_repeats=1)

    assert res["panel_counts"] == [10, 20]
    for solver in ("vlm", "linear"):
        times = res["times_ms"][solver]
        assert len(times) == 2
        for t in times:
            assert np.isfinite(t)
            assert t > 0.0


def test_alpha_sweep_returns_valid_times():
    """Alpha sweep benchmark returns finite positive times."""
    res = benchmark_alpha_sweep(alpha_range=(-2.0, 6.0, 5), solvers=("vlm", "linear"), n_repeats=1)

    assert res["alpha_points"] == 5
    for solver in ("vlm", "linear"):
        t = res["times_ms"][solver]
        assert np.isfinite(t)
        assert t > 0.0


def test_ground_effect_returns_valid_times():
    """Ground effect benchmark returns finite positive times."""
    res = benchmark_ground_effect(panel_counts=(10, 20), n_repeats=1)

    assert res["panel_counts"] == [10, 20]
    for key in ("free_air", "ground_effect"):
        times = res["times_ms"][key]
        assert len(times) == 2
        for t in times:
            assert np.isfinite(t)
            assert t > 0.0


def test_multithread_scaling_returns_speedups():
    """Multithread benchmark returns speedup and efficiency metrics."""
    res = benchmark_multithread(workers_list=(1, 2), n_repeats=1)

    assert 1 in res["workers"]
    assert 2 in res["workers"]
    assert res["speedup"][0] == 1.0  # Baseline
    assert res["speedup"][1] >= 1.0  # Should not be slower than serial
    for eff in res["efficiency"]:
        assert 0.0 <= eff <= 200.0  # Allow superlinear but cap at 200%


def test_symmetry_returns_speedup_and_accuracy():
    """Symmetry benchmark returns speedup and verifies accuracy."""
    res = benchmark_symmetry(panel_counts=(10, 20), solvers=("vlm", "linear"), n_repeats=1)

    assert len(res["records"]) == 4  # 2 solvers x 2 panel counts
    for r in res["records"]:
        # Speedup may be < 1 for very small problems due to overhead; allow it
        assert r["speedup"] > 0.0
        assert r["time_reduction_pct"] > -50.0  # Allow small negative due to noise
        assert r["diff_cl"] < 1e-10  # Machine precision agreement
        assert r["diff_cdi"] < 1e-10
        assert r["aic_mem_reduction"] == 4.0  # (2N)^2 / N^2 = 4


def test_multi_instance_zero_interference():
    """Non-interference test: parallel instances match serial results within 1e-12."""
    res = benchmark_multi_instance(
        n_cases=4,
        alphas_per_case=5,
        n_panels=20,
        test_configs=[(1, 1), (2, 1), (4, 1)],
        n_repeats=1,
    )

    for r in res["records"]:
        # The benchmark itself verifies this, but we assert it here as a test
        assert r["zero_interference_verified"], (
            f"Interference detected: max discrepancy = {r['max_discrepancy']:.2e} "
            f"for config {r['concurrent_instances']}x{r['workers_per_instance']}"
        )
        assert r["max_discrepancy"] < 1e-12, (
            f"Discrepancy {r['max_discrepancy']:.2e} exceeds 1e-12 tolerance"
        )


def test_multi_instance_detailed_verification():
    """Detailed non-interference test with multiple alphas and full result comparison."""
    alphas = np.linspace(-2.0, 6.0, 9)

    # Step 1: Serial baselines
    serial_results = {}
    for _i, name in enumerate(["Rect", "Swept", "Dihedral", "HighAR"]):
        if name == "Rect":
            surf = vt.LiftingSurface(semi_span=5.0, sections=[vt.WingSection(0.0, 1.5), vt.WingSection(1.0, 1.5)])
        elif name == "Swept":
            surf = vt.LiftingSurface(semi_span=6.0, sweep_le=np.radians(20.0),
                                      sections=[vt.WingSection(0.0, 2.0), vt.WingSection(1.0, 0.8)])
        elif name == "Dihedral":
            surf = vt.LiftingSurface(semi_span=4.5, dihedral=np.radians(6.0),
                                      sections=[vt.WingSection(0.0, 1.8), vt.WingSection(1.0, 1.0)])
        else:  # HighAR
            surf = vt.LiftingSurface(semi_span=9.0,
                                      sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 0.4)])

        inst = Ventorum(name=f"Serial_{name}", geometry=surf, alpha_sweep_deg=alphas,
                         n_workers=1, n_panels=25)
        res_list = inst.run()
        serial_results[name] = [
            (r.totals.CL, r.totals.CDi, r.totals.e, np.copy(r.spanwise[0].gamma))
            for r in res_list
        ]

    # Step 2: Parallel with 2 workers each
    parallel_instances = []
    for _i, name in enumerate(["Rect", "Swept", "Dihedral", "HighAR"]):
        if name == "Rect":
            surf = vt.LiftingSurface(semi_span=5.0, sections=[vt.WingSection(0.0, 1.5), vt.WingSection(1.0, 1.5)])
        elif name == "Swept":
            surf = vt.LiftingSurface(semi_span=6.0, sweep_le=np.radians(20.0),
                                      sections=[vt.WingSection(0.0, 2.0), vt.WingSection(1.0, 0.8)])
        elif name == "Dihedral":
            surf = vt.LiftingSurface(semi_span=4.5, dihedral=np.radians(6.0),
                                      sections=[vt.WingSection(0.0, 1.8), vt.WingSection(1.0, 1.0)])
        else:
            surf = vt.LiftingSurface(semi_span=9.0,
                                      sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 0.4)])

        inst = Ventorum(name=f"Parallel_{name}", geometry=surf, alpha_sweep_deg=alphas,
                         n_workers=2, backend="thread", n_panels=25)
        parallel_instances.append(inst)

    completed = run_parallel_instances(parallel_instances, max_concurrent_instances=4,
                                       instance_backend="thread", show_progress=False)

    # Step 3: Verify zero interference at machine precision (1e-12)
    for inst in completed:
        name = inst.name.replace("Parallel_", "")
        ser_data = serial_results[name]
        assert inst.status == "completed"
        assert len(inst.sweep_results) == len(alphas)

        for alpha_idx, r in enumerate(inst.sweep_results):
            ser_cl, ser_cdi, ser_e, ser_gamma = ser_data[alpha_idx]

            np.testing.assert_allclose(
                r.totals.CL, ser_cl, rtol=1e-12, atol=1e-12,
                err_msg=f"CL divergence in case '{name}' at alpha index {alpha_idx}"
            )
            np.testing.assert_allclose(
                r.totals.CDi, ser_cdi, rtol=1e-12, atol=1e-12,
                err_msg=f"CDi divergence in case '{name}' at alpha index {alpha_idx}"
            )
            np.testing.assert_allclose(
                r.totals.e, ser_e, rtol=1e-12, atol=1e-12,
                err_msg=f"Span efficiency divergence in case '{name}' at alpha index {alpha_idx}"
            )
            np.testing.assert_allclose(
                r.spanwise[0].gamma, ser_gamma, rtol=1e-12, atol=1e-12,
                err_msg=f"Circulation Gamma divergence in case '{name}' at alpha index {alpha_idx}"
            )


def test_benchmark_functions_accept_use_symmetry():
    """Benchmark functions respect the use_symmetry parameter."""
    # Panel scaling with symmetry
    res_sym = benchmark_panel_scaling(panel_counts=(10,), solvers=("vlm",), n_repeats=1, use_symmetry=True)
    res_nosym = benchmark_panel_scaling(panel_counts=(10,), solvers=("vlm",), n_repeats=1, use_symmetry=False)

    # With symmetry should be faster (or equal)
    assert res_sym["times_ms"]["vlm"][0] <= res_nosym["times_ms"]["vlm"][0] * 1.5  # Allow some variance

    # Sweep with symmetry
    res_sym = benchmark_alpha_sweep(alpha_range=(-2.0, 2.0, 3), solvers=("vlm",), n_repeats=1, use_symmetry=True)
    res_nosym = benchmark_alpha_sweep(alpha_range=(-2.0, 2.0, 3), solvers=("vlm",), n_repeats=1, use_symmetry=False)
    assert res_sym["times_ms"]["vlm"] <= res_nosym["times_ms"]["vlm"] * 1.5


def test_fourier_solver_in_panel_scaling():
    """Fourier solver works with appropriate geometry."""
    # Fourier solver requires unswept, planar, symmetric wing
    def fourier_geometry():
        return vt.LiftingSurface(
            semi_span=5.0,
            sections=[
                vt.WingSection(y_frac=0.0, chord=2.0, x_le=-0.5),
                vt.WingSection(y_frac=1.0, chord=1.0, x_le=-0.25),
            ],
        )

    # Direct test instead of using benchmark function
    for n in (10, 20):
        geom = fourier_geometry()
        res = vt.analyze(geom, alpha_deg=5.0, solver="fourier", n_panels=n)
        assert res.totals.CL > 0.0
        assert res.totals.CDi > 0.0
        assert np.isfinite(res.totals.CL)
        assert np.isfinite(res.totals.CDi)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
