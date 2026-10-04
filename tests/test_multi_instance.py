# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Unit and integration tests for Ventorum multi-instance parallel execution
and non-interference verification.
"""

import numpy as np
import pytest

import ventorum as vt
from ventorum.instance import Ventorum, run_parallel_instances, VentorumCaseManager


@pytest.fixture
def sample_wings():
    """Build a dictionary of distinct aerodynamic cases."""
    # Case 1: Classical Straight Rectangular Wing
    wing_rect = vt.LiftingSurface(
        name="RectangularWing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.5),
            vt.WingSection(y_frac=1.0, chord=1.5),
        ],
    )

    # Case 2: Swept Tapered Wing with Washout
    wing_swept = vt.LiftingSurface(
        name="SweptWing",
        semi_span=6.0,
        sweep_le=np.radians(20.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0, twist=0.0),
            vt.WingSection(y_frac=1.0, chord=0.8, twist=np.radians(-2.0)),
        ],
    )

    # Case 3: Wing with Dihedral
    wing_dihedral = vt.LiftingSurface(
        name="DihedralWing",
        semi_span=4.5,
        dihedral=np.radians(6.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.8),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )

    # Case 4: High Aspect Ratio Sailplane Wing
    wing_high_ar = vt.LiftingSurface(
        name="SailplaneWing",
        semi_span=9.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=0.7, chord=0.8),
            vt.WingSection(y_frac=1.0, chord=0.4, twist=np.radians(-1.5)),
        ],
    )

    return {
        "rect": wing_rect,
        "swept": wing_swept,
        "dihedral": wing_dihedral,
        "sailplane": wing_high_ar,
    }


def test_single_instance_solve(sample_wings):
    """Test that a single Ventorum instance solves a single operating point correctly."""
    inst = Ventorum(
        name="SingleTest",
        geometry=sample_wings["rect"],
        alpha_deg=5.0,
        V_inf=45.0,
        n_panels=20,
    )
    res = inst.run()
    assert inst.status == "completed"
    assert inst.result is not None
    assert res.totals.CL > 0.3
    assert res.totals.CDi > 0.0
    assert inst.execution_time > 0.0

    summary = inst.get_summary()
    assert summary["name"] == "SingleTest"
    assert summary["status"] == "completed"
    assert "single_point" in summary
    assert "CL" in summary["single_point"]


def test_single_instance_sweep_with_workers(sample_wings):
    """Test that an Ventorum instance can run an internal alpha sweep using multiple workers."""
    alphas = np.linspace(-2.0, 8.0, 11)
    inst = Ventorum(
        name="SweepTest",
        geometry=sample_wings["swept"],
        alpha_sweep_deg=alphas,
        n_workers=2,
        backend="thread",
        n_panels=20,
    )
    results = inst.run()
    assert inst.status == "completed"
    assert inst.sweep_results is not None
    assert len(results) == len(alphas)

    # Polar extraction
    alphas_out, cls, cdis = inst.get_polar()
    assert len(cls) == len(alphas)
    assert np.all(np.diff(cls) > 0.0)  # Linear regime: CL monotonically increases with alpha
    assert np.all(cdis >= 0.0)


def test_multi_instance_parallel_zero_interference(sample_wings):
    """Verify that running multiple independent Ventorum instances simultaneously,
    each with multiple workers, yields mathematically identical results to running
    them sequentially in complete isolation (zero crosstalk / interference).
    """
    alphas = np.linspace(-2.0, 6.0, 9)

    # Step 1: Run each case sequentially in complete serial isolation (baseline)
    serial_results = {}
    for case_key, wing in sample_wings.items():
        inst_serial = Ventorum(
            name=f"Serial_{case_key}",
            geometry=wing,
            alpha_sweep_deg=alphas,
            n_workers=1,
            n_panels=25,
        )
        res_list = inst_serial.run()
        serial_results[case_key] = [
            (r.totals.CL, r.totals.CDi, r.totals.e, np.copy(r.spanwise[0].gamma))
            for r in res_list
        ]

    # Step 2: Run all cases concurrently in parallel, where EACH instance uses 2 workers
    parallel_instances = [
        Ventorum(
            name=f"Parallel_{case_key}",
            geometry=sample_wings[case_key],
            alpha_sweep_deg=alphas,
            n_workers=2,  # Multi-worker within each instance
            backend="thread",
            n_panels=25,
            metadata={"origin_key": case_key},
        )
        for case_key in sample_wings
    ]

    completed_instances = run_parallel_instances(
        parallel_instances,
        max_concurrent_instances=4,
        instance_backend="thread",
        show_progress=False,
    )

    # Step 3: Numerical verification of absolute zero interference
    for inst in completed_instances:
        case_key = inst.metadata["origin_key"]
        ser_data = serial_results[case_key]
        assert inst.status == "completed"
        assert len(inst.sweep_results) == len(alphas)

        for alpha_idx, r in enumerate(inst.sweep_results):
            ser_cl, ser_cdi, ser_e, ser_gamma = ser_data[alpha_idx]

            # Assert numerical identity down to machine precision
            np.testing.assert_allclose(
                r.totals.CL, ser_cl, rtol=1e-12, atol=1e-12,
                err_msg=f"CL divergence in case '{case_key}' at alpha index {alpha_idx}!",
            )
            np.testing.assert_allclose(
                r.totals.CDi, ser_cdi, rtol=1e-12, atol=1e-12,
                err_msg=f"CDi divergence in case '{case_key}' at alpha index {alpha_idx}!",
            )
            np.testing.assert_allclose(
                r.totals.e, ser_e, rtol=1e-12, atol=1e-12,
                err_msg=f"Span efficiency divergence in case '{case_key}' at alpha index {alpha_idx}!",
            )
            np.testing.assert_allclose(
                r.spanwise[0].gamma, ser_gamma, rtol=1e-12, atol=1e-12,
                err_msg=f"Circulation Gamma divergence in case '{case_key}' at alpha index {alpha_idx}!",
            )


def test_case_manager_workflow(sample_wings):
    """Test the high-level VentorumCaseManager orchestrator."""
    mgr = VentorumCaseManager("AeroStudy")
    alphas = np.array([0.0, 5.0, 10.0])

    mgr.create_case("CaseA", sample_wings["rect"], alpha_sweep_deg=alphas, n_workers=2, n_panels=20)
    mgr.create_case("CaseB", sample_wings["swept"], alpha_sweep_deg=alphas, n_workers=2, n_panels=20)

    completed = mgr.run_all(max_concurrent_instances=2, show_progress=False)
    assert len(completed) == 2
    assert all(c.status == "completed" for c in completed)

    summaries = mgr.summary()
    assert len(summaries) == 2
    assert summaries[0]["name"] == "CaseA"
    assert summaries[1]["name"] == "CaseB"
    assert "sweep" in summaries[0]


def test_copy_isolation(sample_wings):
    """Verify that copying an Ventorum instance produces a deep, isolated copy."""
    inst1 = Ventorum("Original", geometry=sample_wings["rect"], alpha_deg=3.0, n_panels=20)
    inst2 = inst1.copy(deep=True)

    inst2.name = "Modified"
    inst2.condition.alpha = np.radians(8.0)

    assert inst1.name == "Original"
    assert np.isclose(inst1.condition.alpha, np.radians(3.0))
    assert np.isclose(inst2.condition.alpha, np.radians(8.0))


def test_error_isolation(sample_wings):
    """Verify that a failing instance does not abort other concurrent instances."""
    inst_good1 = Ventorum("Good1", geometry=sample_wings["rect"], alpha_deg=5.0, n_panels=20)
    inst_bad = Ventorum("Bad", geometry=None, alpha_deg=5.0, n_panels=20)  # Missing geometry
    inst_good2 = Ventorum("Good2", geometry=sample_wings["swept"], alpha_deg=5.0, n_panels=20)

    run_parallel_instances([inst_good1, inst_bad, inst_good2], max_concurrent_instances=3, show_progress=False)

    assert inst_good1.status == "completed"
    assert inst_good1.result is not None
    assert inst_bad.status == "failed"
    assert inst_bad.error is not None
    assert inst_good2.status == "completed"
    assert inst_good2.result is not None


def test_case_manager_analytics_and_optimization(sample_wings):
    """Test VentorumCaseManager analytics methods: table, polars, optimization, and dataframe."""
    mgr = VentorumCaseManager("AeroOptimizationStudy")
    alphas = np.array([0.0, 4.0, 8.0])

    # Case 1: High aspect ratio wing (sailplane)
    mgr.create_case("Sailplane", sample_wings["sailplane"], alpha_sweep_deg=alphas, n_workers=1, n_panels=20)
    # Case 2: Rectangular wing
    mgr.create_case("Rectangular", sample_wings["rect"], alpha_sweep_deg=alphas, n_workers=1, n_panels=20)

    mgr.run_all(show_progress=False)

    # 1. Summary table
    table_str = mgr.summary_table()
    assert "Sailplane" in table_str
    assert "Rectangular" in table_str
    assert "completed" in table_str

    # 2. Polars retrieval
    polars = mgr.get_all_polars()
    assert "Sailplane" in polars
    assert "Rectangular" in polars
    a_tap, cl_tap, cdi_tap = polars["Sailplane"]
    assert len(a_tap) == 3

    # 3. Optimal case selection
    best_inst, best_cl, summary = mgr.find_optimal(metric="CL", objective="max")
    assert best_inst.name in ("Sailplane", "Rectangular")
    assert best_cl > 0.0

    # 4. to_dataframe or structured records
    df = mgr.to_dataframe()
    assert len(df) == 2

    # 5. Instance get_L_over_D
    ld = best_inst.get_L_over_D()
    assert isinstance(ld, np.ndarray)
    assert len(ld) == 3


def test_given_settings_are_kept(sample_wings):
    """Settings given to the instance are not overwritten by the defaults (also in copy())."""
    s = vt.SolverSettings(n_panels=24, use_symmetry=False)
    inst = Ventorum("Keep", geometry=sample_wings["rect"], settings=s)
    assert inst.settings.n_panels == 24 and inst.settings.use_symmetry is False
    cp = inst.copy()
    assert cp.settings.n_panels == 24 and cp.settings.use_symmetry is False
    # Explicit arguments still change the given settings.
    inst2 = Ventorum("Override", geometry=sample_wings["rect"], settings=s, n_panels=16, use_symmetry=True)
    assert inst2.settings.n_panels == 16 and inst2.settings.use_symmetry is True
    assert s.n_panels == 24  # the caller's object does not change
    # Without settings the defaults apply.
    inst3 = Ventorum("Default", geometry=sample_wings["rect"])
    assert inst3.settings.n_panels == 80 and inst3.settings.use_symmetry is True
    mgr = VentorumCaseManager()
    assert mgr.create_case("C", sample_wings["rect"], settings=s).settings.n_panels == 24


def test_default_solver_matches_analyze(sample_wings):
    """The instance default solver is 'auto', the same as ventorum.analyze."""
    inst = Ventorum("Auto", geometry=sample_wings["rect"], n_panels=12, alpha_deg=4.0)
    assert inst.settings.solver_type == "auto"
    res = inst.analyze()
    ref = vt.analyze(sample_wings["rect"], alpha_deg=4.0, n_panels=12)
    assert res.totals.CL == pytest.approx(ref.totals.CL, rel=1e-12)


def test_polar_uses_swept_alphas(sample_wings):
    """get_polar and get_summary use the angles of the last sweep, not the configured ones."""
    inst = Ventorum("Polar", geometry=sample_wings["rect"], n_panels=12, alpha_sweep_deg=[0, 2, 4, 6])
    inst.analyze_sweep([10.0])
    a, cl, cdi = inst.get_polar()
    assert a.tolist() == [10.0] and len(cl) == 1 and len(cdi) == 1
    sw = inst.get_summary()["sweep"]
    assert sw["n_alphas"] == 1 and sw["alpha_deg_min"] == 10.0 and sw["alpha_deg_max"] == 10.0

    inst2 = Ventorum("NoConfig", geometry=sample_wings["rect"], n_panels=12)
    inst2.analyze_sweep([0.0, 2.0, 4.0])
    a2, cl2, _ = inst2.get_polar()
    assert a2.tolist() == [0.0, 2.0, 4.0] and len(cl2) == 3
    assert "sweep" in inst2.get_summary()


def test_process_backend_is_refused(sample_wings):
    inst = Ventorum("Proc", geometry=sample_wings["rect"], n_panels=12, alpha_sweep_deg=[0, 2],
                     backend="process", n_workers=2)
    with pytest.raises(ValueError, match="thread"):
        inst.run()
