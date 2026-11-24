# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the GPU time-domain vortex-ring march.

Covers GPU against CPU parity in float64 (1e-11 relative) and float32 (the
float32 floor of :mod:`tests.test_gpu_ring_kernels`) for
:func:`ventorum.time_response`, :func:`ventorum.indicial_response` and
:func:`ventorum.free_flight`, in free air and at ``h / c = 0.5``, with
six-axis motion, moving control surfaces, convected 1-cosine gusts and
state-feedback free flight; automatic device selection and device-memory
release; GPU out-of-memory fallback; the CPU fallback note when CUDA is hidden
in a new process; and CPU device notes and result fields without a GPU.
"""

from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.geometry.lattice import build_lattice
from ventorum.unsteady.gusts import one_minus_cosine
from ventorum.unsteady.harmonic import NOTE_GPU_OUT_OF_MEMORY
from ventorum.unsteady.timedomain import _TimeDomainStepper, _select_time_device

# Same tolerance floor as tests/test_gpu_ring_kernels.py and tests/test_gpu_pipeline.py.
TOL = {"float64": 1e-11, "float32": 2e-5}
ROOT = Path(__file__).resolve().parents[1]


def _need_gpu() -> None:
    """Skip the test when the CUDA GPU pipelines cannot run."""
    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")


@pytest.fixture
def gpu_device():
    """Run the test with GPU settings restored afterwards."""
    _need_gpu()
    old_dev, old_prec = gpu.get_device(), gpu._precision
    yield
    gpu.release()
    gpu.set_device(old_dev)
    gpu.set_precision(old_prec)


def _flapped_wing(
    *,
    semi_span: float = 3.0,
    chord: float = 1.0,
    n_panels: int = 4,
    cambered: bool = False,
) -> vt.Aircraft:
    """Build a rectangular wing with a full-span chordwise flap."""
    flap = vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=0.0)
    shape = "4412" if cambered else None
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=semi_span,
        n_panels=n_panels,
        controls=[flap],
        sections=[
            vt.WingSection(y_frac=0.0, chord=chord, shape=shape),
            vt.WingSection(y_frac=1.0, chord=chord, shape=shape),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25 * chord, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _wing_tail(*, ref_x: float = -0.15) -> vt.Aircraft:
    """Build a stable wing-tail aircraft with a tail elevator."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        n_panels=4,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    tail = vt.LiftingSurface(
        name="tail",
        semi_span=1.0,
        n_panels=4,
        position=np.array([1.8, 0.0, 0.0]),
        controls=[vt.ControlSurface(name="elevator", eta_start=0.0, eta_end=1.0, hinge_x_c=0.5, deflection=0.0)],
        sections=[vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.5)],
    )
    ac = vt.Aircraft(surfaces=[wing, tail], ref_point=np.array([ref_x, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _assert_rel_close(got: np.ndarray, ref: np.ndarray, tol: float, label: str) -> None:
    """Assert max absolute difference is within ``tol * max(max|ref|, 1.0)``."""
    scale = max(float(np.max(np.abs(ref))), 1.0)
    err = float(np.max(np.abs(got - ref)))
    assert err <= tol * scale, f"{label}: rel error {err / scale:.3e} exceeds {tol:.3e}"


# ---------------------------------------------------------------------------
# 1. CPU tests without the gpu marker: device fields, notes and selection
# ---------------------------------------------------------------------------


def test_cpu_device_fields_notes_and_fallback(monkeypatch):
    """CPU runs record device, precision and CPU notes; unavailable GPU adds a fallback note."""
    ac = _flapped_wing(n_panels=4)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=math.radians(3.0))

    wt = _wing_tail()
    mass = 400.0
    inertia = np.diag([80.0, 120.0, 180.0])

    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        r_tr = vt.time_response(ac, cond, st, n_steps=4)
        r_ind = vt.indicial_response(ac, cond, st, kind="alpha", s_end=2.0)
        r_ff = vt.free_flight(wt, cond, st, mass=mass, inertia=inertia, dofs=("z", "pitch"), n_steps=4)

        for res in (r_tr, r_ind, r_ff):
            assert res.device == "cpu"
            assert res.precision == "float64"
            d = res.to_dict()
            assert d["device"] == "cpu"
            assert d["precision"] == "float64"
            assert any("CPU" in n and "float64" in n for n in res.notes)
            assert not any("GPU unavailable" in n for n in res.notes)

        monkeypatch.setattr(gpu, "get_device", lambda: "gpu")
        monkeypatch.setattr(gpu, "available", lambda: False)
        monkeypatch.setattr(gpu, "unavailable_reason", lambda: "PyTorch finds no CUDA GPU.")

        fb_tr = vt.time_response(ac, cond, st, n_steps=4)
        fb_ind = vt.indicial_response(ac, cond, st, kind="alpha", s_end=2.0)
        fb_ff = vt.free_flight(wt, cond, st, mass=mass, inertia=inertia, dofs=("z", "pitch"), n_steps=4)

        for fb, ref in ((fb_tr, r_tr), (fb_ind, r_ind), (fb_ff, r_ff)):
            assert fb.device == "cpu"
            assert fb.precision == "float64"
            assert any("GPU unavailable" in n and "CPU" in n for n in fb.notes)
            assert fb.trust is not None
            assert any("GPU unavailable" in w and "CPU" in w for w in fb.trust.warnings)
            np.testing.assert_array_equal(fb.CL, ref.CL)
            np.testing.assert_array_equal(fb.Cm, ref.Cm)
    finally:
        gpu.set_device(old_dev)


def test_select_time_device_auto_work_rule(monkeypatch):
    """Auto device selection passes march work n_steps * n_bound * n_wake to gpu.use_gpu."""
    seen: list[tuple[float, str, int, int | None]] = []

    def fake_use_gpu(work: float, family: str = "vlm", n_cases: int = 1, n_panels=None, return_reason: bool = False):
        seen.append((float(work), str(family), int(n_cases), n_panels))
        decision = work >= 5.0e4
        return (decision, "fake") if return_reason else decision

    monkeypatch.setattr(gpu, "get_device", lambda: "auto")
    monkeypatch.setattr(gpu, "available", lambda: True)
    monkeypatch.setattr(gpu, "get_precision", lambda: "float32")
    monkeypatch.setattr(gpu, "use_gpu", fake_use_gpu)

    # Small march: n_steps=4, n_bound=16, n_strips=4 -> n_wake=16, work=1024 < 5e4 -> CPU.
    dev_s, prec_s, notes_s = _select_time_device(4, 16, 4, has_ground=False)
    assert (dev_s, prec_s, notes_s) == ("cpu", "float64", [])
    assert seen[-1] == (4.0 * 16.0 * 16.0, "vlm", 1, None)

    # Large march: n_steps=20, n_bound=64, n_strips=16 -> n_wake=320, work=409600 >= 5e4 -> GPU.
    dev_l, prec_l, notes_l = _select_time_device(20, 64, 16, has_ground=False)
    assert (dev_l, prec_l, notes_l) == ("gpu", "float32", [])
    assert seen[-1] == (20.0 * 64.0 * 320.0, "vlm", 1, None)

    # Ground effect doubles the work estimate.
    dev_g, prec_g, notes_g = _select_time_device(10, 32, 8, has_ground=True)
    assert (dev_g, prec_g, notes_g) == ("gpu", "float32", [])
    assert seen[-1] == (2.0 * 10.0 * 32.0 * 80.0, "vlm", 1, None)


# ---------------------------------------------------------------------------
# 2. Subprocess fallback test with CUDA hidden (CUDA_VISIBLE_DEVICES=-1)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cuda_hidden_cpu_fallback_note_subprocess():
    """With CUDA_VISIBLE_DEVICES=-1 and VENTORUM_DEVICE=gpu, time_response and free_flight fall back to CPU."""
    script = """
import json, math
import numpy as np
import ventorum as vt
from ventorum import gpu

wing = vt.LiftingSurface(
    name="wing",
    semi_span=3.0,
    n_panels=4,
    sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
)
tail = vt.LiftingSurface(
    name="tail",
    semi_span=1.0,
    n_panels=4,
    position=np.array([1.8, 0.0, 0.0]),
    sections=[vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.5)],
)
ac = vt.Aircraft(surfaces=[wing, tail], ref_point=np.array([-0.15, 0.0, 0.0]))
cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=math.radians(3.0))
st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")

r_tr = vt.time_response(ac, cond, st, n_steps=4)
r_ind = vt.indicial_response(ac, cond, st, kind="alpha", s_end=2.0)
r_ff = vt.free_flight(ac, cond, st, mass=400.0, inertia=np.diag([80.0, 120.0, 180.0]), dofs=("z", "pitch"), n_steps=4)

print(json.dumps({
    "gpu_available": gpu.available(),
    "gpu_setting": gpu.get_device(),
    "tr": {
        "device": r_tr.device,
        "precision": r_tr.precision,
        "notes": r_tr.notes,
        "trust_warnings": [] if r_tr.trust is None else list(r_tr.trust.warnings),
        "CL_last": float(r_tr.CL[-1]),
    },
    "ind": {
        "device": r_ind.device,
        "precision": r_ind.precision,
        "notes": r_ind.notes,
        "trust_warnings": [] if r_ind.trust is None else list(r_ind.trust.warnings),
        "CL_last": float(r_ind.CL[-1]),
    },
    "ff": {
        "device": r_ff.device,
        "precision": r_ff.precision,
        "notes": r_ff.notes,
        "trust_warnings": [] if r_ff.trust is None else list(r_ff.trust.warnings),
        "z_last": float(r_ff.z[-1]),
    },
}))
"""
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(ROOT)
    base_env["MPLBACKEND"] = "Agg"
    base_env["NUMBA_NUM_THREADS"] = "4"
    base_env["OMP_NUM_THREADS"] = "4"
    base_env["MKL_NUM_THREADS"] = "4"
    base_env["OPENBLAS_NUM_THREADS"] = "4"
    base_env["CUDA_VISIBLE_DEVICES"] = "-1"

    proc_gpu = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        env=dict(base_env, VENTORUM_DEVICE="gpu"),
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    out_gpu = json.loads(proc_gpu.stdout.strip().splitlines()[-1])
    assert out_gpu["gpu_available"] is False
    assert out_gpu["gpu_setting"] == "gpu"

    proc_cpu = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        env=dict(base_env, VENTORUM_DEVICE="cpu"),
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    out_cpu = json.loads(proc_cpu.stdout.strip().splitlines()[-1])

    for key in ("tr", "ind", "ff"):
        assert out_gpu[key]["device"] == "cpu"
        assert out_gpu[key]["precision"] == "float64"
        assert any("GPU unavailable" in n and "CPU" in n for n in out_gpu[key]["notes"])
        assert any("GPU unavailable" in w and "CPU" in w for w in out_gpu[key]["trust_warnings"])
        assert out_cpu[key]["device"] == "cpu"
        assert not any("GPU unavailable" in n for n in out_cpu[key]["notes"])

    assert out_gpu["tr"]["CL_last"] == pytest.approx(out_cpu["tr"]["CL_last"], rel=1e-14, abs=1e-14)
    assert out_gpu["ind"]["CL_last"] == pytest.approx(out_cpu["ind"]["CL_last"], rel=1e-14, abs=1e-14)
    assert out_gpu["ff"]["z_last"] == pytest.approx(out_cpu["ff"]["z_last"], rel=1e-14, abs=1e-14)


# ---------------------------------------------------------------------------
# 3. GPU against CPU parity: time_response, indicial_response, free_flight
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("h_ground", [None, 0.5])
def test_gpu_time_response_motion_flap_gust_equals_cpu(gpu_device, precision, h_ground):
    """GPU time_response with 6-axis motion, moving flap and 1-cos gust matches CPU."""
    from ventorum.gpu import ring_kernels as RK

    ac = _flapped_wing(n_panels=4, cambered=False)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=math.radians(4.0), h=h_ground)

    n_steps = 8
    omega = 2.0 * 0.25 * cond.V_inf / ac.c_ref
    motion = {
        "x": lambda t: 0.02 * math.sin(omega * t),
        "y": lambda t: 0.015 * math.sin(0.8 * omega * t),
        "z": lambda t: 0.03 * math.sin(omega * t),
        "roll": lambda t: math.radians(1.5) * math.sin(0.9 * omega * t),
        "pitch": lambda t: math.radians(2.0) * math.sin(omega * t + 0.2),
        "yaw": lambda t: math.radians(1.0) * math.sin(0.7 * omega * t),
    }
    controls = {"flap": lambda t: math.radians(6.0) * math.sin(omega * t + 0.3)}
    gust = one_minus_cosine(amplitude=1.5, length=4.0, direction="vertical", x_front=0.0)

    gpu.set_device("cpu")
    ref = vt.time_response(
        ac, cond, st, n_steps=n_steps, motion=motion, controls=controls, gust=gust, store="strengths",
    )
    assert ref.device == "cpu"
    assert ref.precision == "float64"

    gpu.set_device("gpu")
    gpu.set_precision(precision)
    got = vt.time_response(
        ac, cond, st, n_steps=n_steps, motion=motion, controls=controls, gust=gust, store="strengths",
    )
    assert got.device == "gpu"
    assert got.precision == precision
    assert any("GPU" in n and precision in n for n in got.notes)
    d = got.to_dict()
    assert d["device"] == "gpu"
    assert d["precision"] == precision

    # Device memory is released at the end of the run.
    assert RK.device_bytes() == 0

    tol = TOL[precision]
    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
        _assert_rel_close(getattr(got, name), getattr(ref, name), tol, f"{name} (h={h_ground}, {precision})")
    _assert_rel_close(got.strip_cl, ref.strip_cl, 5.0 * tol if precision == "float32" else tol, "strip_cl")
    assert got.gamma_bound is not None and ref.gamma_bound is not None
    assert got.gamma_wake is not None and ref.gamma_wake is not None
    assert got.wake_corners is not None and ref.wake_corners is not None
    _assert_rel_close(got.gamma_bound, ref.gamma_bound, tol, "gamma_bound")
    _assert_rel_close(got.gamma_wake, ref.gamma_wake, tol, "gamma_wake")
    _assert_rel_close(got.wake_corners, ref.wake_corners, tol, "wake_corners")


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("h_ground", [None, 0.5])
def test_gpu_time_response_cambered_no_gust_equals_cpu(gpu_device, precision, h_ground):
    """GPU time_response without gust (device wake buffers) and bent camber polylines matches CPU."""
    ac = _flapped_wing(n_panels=4, cambered=True)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=math.radians(3.0), h=h_ground)

    n_steps = 8
    omega = 2.0 * 0.2 * cond.V_inf / ac.c_ref
    motion = {"z": lambda t: 0.02 * math.sin(omega * t), "pitch": lambda t: math.radians(1.5) * math.sin(omega * t)}

    gpu.set_device("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = vt.time_response(ac, cond, st, n_steps=n_steps, motion=motion, store="strengths")

    gpu.set_device("gpu")
    gpu.set_precision(precision)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        got = vt.time_response(ac, cond, st, n_steps=n_steps, motion=motion, store="strengths")

    assert got.device == "gpu"
    assert got.precision == precision
    tol = TOL[precision]
    for name in ("CL", "CD", "Cm", "gamma_bound", "gamma_wake", "wake_corners"):
        _assert_rel_close(getattr(got, name), getattr(ref, name), tol, name)
    _assert_rel_close(got.strip_cl, ref.strip_cl, 5.0 * tol if precision == "float32" else tol, "strip_cl")


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_gpu_time_response_stationary_ground_caches_equals_cpu(gpu_device, precision):
    """GPU stationary wake and ground-image caches at h/c = 0.5 match CPU."""
    ac = _flapped_wing(n_panels=4, cambered=False)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=math.radians(4.0), h=0.5)

    gpu.set_device("cpu")
    ref = vt.time_response(ac, cond, st, n_steps=8, store="strengths")

    gpu.set_device("gpu")
    gpu.set_precision(precision)
    got = vt.time_response(ac, cond, st, n_steps=8, store="strengths")

    assert got.device == "gpu"
    assert got.precision == precision
    tol = TOL[precision]
    for name in ("CL", "CD", "Cm", "gamma_bound", "gamma_wake", "wake_corners"):
        _assert_rel_close(getattr(got, name), getattr(ref, name), tol, f"stat_ge {name}")
    _assert_rel_close(got.strip_cl, ref.strip_cl, 5.0 * tol if precision == "float32" else tol, "stat_ge strip_cl")


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("h_ground", [None, 0.5])
def test_gpu_indicial_alpha_equals_cpu(gpu_device, precision, h_ground):
    """GPU indicial_response(kind='alpha') matches CPU in float64 and float32."""
    ac = _flapped_wing(n_panels=4, cambered=False)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0, h=h_ground)

    gpu.set_device("cpu")
    ref = vt.indicial_response(ac, cond, st, kind="alpha", s_end=6.0)
    assert ref.device == "cpu"

    gpu.set_device("gpu")
    gpu.set_precision(precision)
    got = vt.indicial_response(ac, cond, st, kind="alpha", s_end=6.0)
    assert got.device == "gpu"
    assert got.precision == precision
    assert any("GPU" in n and precision in n for n in got.notes)

    tol = TOL[precision]
    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
        _assert_rel_close(getattr(got, name), getattr(ref, name), tol, f"indicial {name}")
    _assert_rel_close(got.strip_cl, ref.strip_cl, 5.0 * tol if precision == "float32" else tol, "indicial strip_cl")


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("h_ground", [None, 0.5])
def test_gpu_free_flight_equals_cpu(gpu_device, precision, h_ground):
    """GPU free_flight with z and pitch free, state-feedback elevator and thrust matches CPU."""
    from ventorum.gpu import ring_kernels as RK

    ac = _wing_tail()
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=math.radians(2.0), h=h_ground)

    def elevator(t, state):
        return -0.05 * state.pitch_dot

    def thrust(t, state):
        return np.array([-40.0, 0.0, 0.0]), np.array([0.0, 5.0 * state.pitch, 0.0])

    kw = {
        "mass": 400.0,
        "inertia": np.diag([80.0, 120.0, 180.0]),
        "cg": np.array([-0.1, 0.0, 0.0]),
        "dofs": ("z", "pitch"),
        "initial_state": {"pitch_dot": 2.0e-3, "z_dot": -0.01},
        "controls": {"elevator": elevator},
        "thrust": thrust,
        "thrust_point": np.array([0.3, 0.0, 0.05]),
        "n_steps": 10,
        "store": "strengths",
    }

    gpu.set_device("cpu")
    ref = vt.free_flight(ac, cond, st, **kw)
    assert ref.device == "cpu"
    assert ref.precision == "float64"

    gpu.set_device("gpu")
    gpu.set_precision(precision)
    got = vt.free_flight(ac, cond, st, **kw)
    assert got.device == "gpu"
    assert got.precision == precision
    assert any("GPU" in n and precision in n for n in got.notes)
    d = got.to_dict()
    assert d["device"] == "gpu"
    assert d["precision"] == precision
    assert RK.device_bytes() == 0

    tol = TOL[precision]
    for name in (
        "z", "pitch", "z_dot", "pitch_dot",
        "CL", "CD", "Cm",
        "gamma_bound", "gamma_wake", "wake_corners",
    ):
        _assert_rel_close(getattr(got, name), getattr(ref, name), tol, f"free_flight {name}")
    _assert_rel_close(got.strip_cl, ref.strip_cl, 5.0 * tol if precision == "float32" else tol, "free_flight strip_cl")
    _assert_rel_close(got.controls["elevator"], ref.controls["elevator"], tol, "free_flight elevator")

    if precision == "float64":
        np.testing.assert_array_equal(got.subiterations, ref.subiterations)


@pytest.mark.gpu
def test_gpu_free_flight_with_gust_equals_cpu(gpu_device):
    """GPU free_flight with a 1-cos gust exercises device wake-node save and restore."""
    ac = _wing_tail()
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=math.radians(2.0), h=0.5)
    kw = {
        "mass": 400.0,
        "inertia": np.diag([80.0, 120.0, 180.0]),
        "dofs": ("z", "pitch"),
        "gust": one_minus_cosine(amplitude=0.8, length=4.0, direction="vertical", x_front=0.0),
        "n_steps": 6,
        "store": "strengths",
    }

    gpu.set_device("cpu")
    ref = vt.free_flight(ac, cond, st, **kw)

    gpu.set_device("gpu")
    gpu.set_precision("float64")
    got = vt.free_flight(ac, cond, st, **kw)

    assert got.device == "gpu"
    assert got.precision == "float64"
    tol = TOL["float64"]
    for name in ("z", "pitch", "CL", "Cm", "gamma_bound", "gamma_wake", "wake_corners"):
        _assert_rel_close(getattr(got, name), getattr(ref, name), tol, f"free_flight_gust {name}")
    np.testing.assert_array_equal(got.subiterations, ref.subiterations)


@pytest.mark.gpu
def test_auto_device_switches_by_march_work(gpu_device):
    """Auto device keeps a tiny march on CPU and runs a large march on GPU."""
    ac = _flapped_wing(n_panels=4)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=math.radians(3.0))

    gpu.set_device("auto")
    gpu.set_precision("float32")

    # Small march: 4 steps, 32 bound panels, 8 strips -> work = 4 * 32 * 32 = 4096 < 5e4 -> CPU.
    r_small = vt.time_response(ac, cond, st, n_steps=4)
    assert r_small.device == "cpu"
    assert r_small.precision == "float64"

    # Larger march: 20 steps, 32 bound panels, 8 strips -> work = 20 * 32 * 160 = 102400 >= 5e4 -> GPU.
    r_large = vt.time_response(ac, cond, st, n_steps=20)
    assert r_large.device == "gpu"
    assert r_large.precision == "float32"


# ---------------------------------------------------------------------------
# 4. Device memory management and OOM fallback
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_gpu_stepper_memory_registered_and_released(gpu_device):
    """_TimeDomainStepper registers its CUDA buffers and frees them via release_gpu and gpu.release."""
    import torch

    from ventorum.gpu import ring_kernels as RK

    gpu.set_device("gpu")
    gpu.set_precision("float64")
    gpu.release()
    gc.collect()
    assert RK.device_bytes() == 0

    ac = _flapped_wing(n_panels=6)
    lat = build_lattice(ac, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=25.0, rho=1.225, alpha=math.radians(3.0))

    stepper = _TimeDomainStepper.from_prescribed(
        lat, cond, S_ref=ac.S_ref, b_ref=ac.b_ref, c_ref=ac.c_ref, ref_point=ac.ref_point, n_steps=6,
    )
    assert stepper.device == "gpu"
    stepper.advance_prescribed(0)
    stepper.advance_prescribed(1)

    held = RK.device_bytes()
    assert held > 0
    allocated = torch.cuda.memory_allocated()
    freed = gpu.release()
    gc.collect()
    assert freed >= held
    assert RK.device_bytes() == 0
    assert torch.cuda.memory_allocated() <= allocated - held


@pytest.mark.gpu
def test_gpu_time_march_oom_fallback_adds_note(gpu_device, monkeypatch):
    """Two GPU out-of-memory errors fall back to CPU with one warning and the fallback note."""
    import torch

    from ventorum.gpu import ring_kernels as RK

    ac = _flapped_wing(n_panels=4)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4, chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=math.radians(3.0))

    gpu.set_device("cpu")
    ref = vt.time_response(ac, cond, st, n_steps=4)

    oom_cls = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError

    def raise_oom(*args, **kwargs):
        raise oom_cls("CUDA out of memory. Tried to allocate 20.00 GiB (test)")

    gpu.set_device("gpu")
    monkeypatch.setattr(RK, "ring_normal_influence_gpu", raise_oom)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = vt.time_response(ac, cond, st, n_steps=4)

    assert any("out of memory" in str(w.message).lower() for w in caught)
    assert got.device == "cpu"
    assert got.precision == "float64"
    assert NOTE_GPU_OUT_OF_MEMORY in got.notes
    assert got.trust is not None and NOTE_GPU_OUT_OF_MEMORY in got.trust.warnings
    np.testing.assert_array_equal(got.CL, ref.CL)
    np.testing.assert_array_equal(got.Cm, ref.Cm)
