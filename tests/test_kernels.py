# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Parity tests of the vortex kernels and of the symmetric fold in the loads.

The numpy kernels are the reference. The compiled (Numba) kernels and the
symmetric fold must give the same results to round-off.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero import loads as loads_mod
from ventorum.aero import vortex as V
from ventorum.aero.system import UnknownMap, build_sources, make_ground_plane, panel_targets
from ventorum.geometry.lattice import build_lattice


def _wing_and_tail():
    wing = vt.LiftingSurface(name="wing", semi_span=4.0, n_panels=8, dihedral=np.radians(5.0),
                              sections=[vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.6)])
    tail = vt.LiftingSurface(name="tail", semi_span=1.2, n_panels=4, position=np.array([3.5, 0.0, 0.2]),
                              sections=[vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.4)])
    return vt.Aircraft(surfaces=[wing, tail])


@pytest.fixture
def numpy_backend():
    old = V._kernel_backend  # the setting ("auto" stays "auto"), not the resolved backend
    yield
    V.set_kernel_backend(old)


@pytest.fixture(autouse=True)
def _restore_kernel_backend():
    """Restore the backend setting after every test of this file.

    Some torch tests left the backend on "torch", and the tests of other
    files that ran after them failed.
    """
    old = V._kernel_backend
    yield
    V._kernel_backend = old


@pytest.mark.parametrize("with_ground", [False, True])
def test_numba_kernels_equal_numpy_kernels(numpy_backend, with_ground):
    ac = _wing_and_tail()
    lat = build_lattice(ac, vt.SolverSettings(n_panels=8, n_chord=3), collocation="vlm")
    n = lat.n_panels
    ground = make_ground_plane(lat, 0.8, np.radians(4.0), 0.0, np.radians(2.0), height_ref="min") if with_ground else None
    src = build_sources(lat, np.array([1.0, 0.0, 0.05]), UnknownMap(np.arange(n), np.arange(n), False), ground)
    tg = panel_targets(lat, np.arange(n))
    gamma = np.linspace(0.5, 2.0, n)
    out = {}
    for backend in ("numpy", "numba"):
        V.set_kernel_backend(backend)
        out[backend] = (
            V.horseshoe_velocity_tensor(lat.cp, src, tg),
            V.influence_matrix(lat.cp, lat.normal_bc, src, n, tg),
            V.induced_velocity(lat.force_points, src, gamma, tg),
        )
    for ref, new in zip(out["numpy"], out["numba"]):
        assert np.max(np.abs(new - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_symmetric_fold_of_the_loads_equals_the_full_evaluation(monkeypatch):
    ac = _wing_and_tail()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    st = vt.SolverSettings(n_panels=8, n_chord=3)
    folded = vt.analyze(ac, condition=cond, settings=st).totals
    monkeypatch.setattr(loads_mod, "_symmetric_fold", lambda *a, **k: None)
    full = vt.analyze(ac, condition=cond, settings=st).totals
    for name in ("CL", "CDi", "CDi_nearfield", "Cm"):
        assert getattr(folded, name) == pytest.approx(getattr(full, name), rel=1e-12)
    for name in ("CY", "Cl", "Cn"):
        assert abs(getattr(folded, name) - getattr(full, name)) < 1e-14


def test_kernel_backend_switch_is_checked():
    with pytest.raises(ValueError):
        V.set_kernel_backend("cuda-please")


def test_numba_trefftz_kernel_equals_numpy(numpy_backend):
    rng = np.random.default_rng(3)
    q = rng.normal(size=(40, 3))
    qn = rng.normal(size=(40, 3))
    p = rng.normal(size=(60, 3))
    g = rng.normal(size=60)
    d = np.array([1.0, 0.0, 0.0])
    rc = rng.uniform(0.0, 0.1, size=60)
    V.set_kernel_backend("numpy")
    ref = V.trefftz_normalwash(q, qn, p, g, d, rc)
    V.set_kernel_backend("numba")
    new = V.trefftz_normalwash(q, qn, p, g, d, rc)
    assert np.max(np.abs(new - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_sweep_cache_gives_the_same_results_as_single_solves():
    """A sweep keeps the fixed part of the vortex system between angles; the results must not change."""
    ac = _wing_and_tail()
    st = vt.SolverSettings(n_panels=8, n_chord=3)
    alphas = np.array([-2.0, 3.0, 8.0])
    sweep = vt.analyze_sweep(ac, alphas, settings=st, n_jobs=1)
    for a, r in zip(alphas, sweep):
        single = vt.analyze(ac, alpha_deg=float(a), settings=st).totals
        for name in ("CL", "CDi", "Cm"):
            assert getattr(r.totals, name) == pytest.approx(getattr(single, name), rel=1e-11, abs=1e-14)


@pytest.mark.parametrize("with_ground", [False, True])
def test_cython_kernels_equal_numpy_kernels(numpy_backend, with_ground):
    pytest.importorskip("ventorum.aero.vortex_cython")
    ac = _wing_and_tail()
    lat = build_lattice(ac, vt.SolverSettings(n_panels=8, n_chord=3), collocation="vlm")
    n = lat.n_panels
    ground = make_ground_plane(lat, 0.8, np.radians(4.0), 0.0, np.radians(2.0), height_ref="min") if with_ground else None
    src = build_sources(lat, np.array([1.0, 0.0, 0.05]), UnknownMap(np.arange(n), np.arange(n), False), ground)
    tg = panel_targets(lat, np.arange(n))
    gamma = np.linspace(0.5, 2.0, n)
    out = {}
    for backend in ("numpy", "cython"):
        V.set_kernel_backend(backend)
        out[backend] = (
            V.horseshoe_velocity_tensor(lat.cp, src, tg),
            V.influence_matrix(lat.cp, lat.normal_bc, src, n, tg),
            V.induced_velocity(lat.force_points, src, gamma, tg),
        )
    for ref, new in zip(out["numpy"], out["cython"]):
        assert np.max(np.abs(new - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_cython_kernel_parts_add_up():
    """Part 1 (fixed legs) plus part 2 (wake legs) equals part 0 (all legs)."""
    pytest.importorskip("ventorum.aero.vortex_cython")
    from ventorum.aero import vortex_cython as cy
    from ventorum.aero import vortex_numba as nb

    ac = _wing_and_tail()
    lat = build_lattice(ac, vt.SolverSettings(n_panels=8, n_chord=3), collocation="vlm")
    n = lat.n_panels
    src = build_sources(lat, np.array([1.0, 0.0, 0.05]), UnknownMap(np.arange(n), np.arange(n), False), None)
    tg = panel_targets(lat, np.arange(n))
    gamma = np.linspace(0.5, 2.0, n)
    a, b, a_te, b_te, d, rc2, grp, sign = nb.pack_sources(src)
    tg_group, tg_rc2, use_tg = nb.pack_targets(n, src, tg)
    args = (np.ascontiguousarray(lat.cp, dtype=float), np.ascontiguousarray(lat.normal_bc, dtype=float),
            a, b, a_te, b_te, d, rc2, grp, sign,
            np.ascontiguousarray(src.column, dtype=np.int64), n, tg_group, tg_rc2, use_tg)
    A0 = cy.normal_influence_kernel(*args, 0, 1)
    A1 = cy.normal_influence_kernel(*args, 1, 1)
    A2 = cy.normal_influence_kernel(*args, 2, 1)
    assert np.max(np.abs(A1 + A2 - A0)) <= 1e-13 * np.max(np.abs(A0))
    g_src = gamma[src.column] * src.sign
    gc = np.ascontiguousarray(g_src, dtype=float)
    iargs = (np.ascontiguousarray(lat.force_points, dtype=float), a, b, a_te, b_te, d, rc2, grp, gc,
             tg_group, tg_rc2, use_tg)
    v0 = cy.induced_velocity_kernel(*iargs, 0, 1)
    v1 = cy.induced_velocity_kernel(*iargs, 1, 1)
    v2 = cy.induced_velocity_kernel(*iargs, 2, 1)
    assert np.max(np.abs(v1 + v2 - v0)) <= 1e-13 * np.max(np.abs(v0))


def test_cython_trefftz_kernel_equals_numpy(numpy_backend):
    pytest.importorskip("ventorum.aero.vortex_cython")
    rng = np.random.default_rng(3)
    q = rng.normal(size=(40, 3))
    qn = rng.normal(size=(40, 3))
    p = rng.normal(size=(60, 3))
    g = rng.normal(size=60)
    d = np.array([1.0, 0.0, 0.0])
    rc = rng.uniform(0.0, 0.1, size=60)
    V.set_kernel_backend("numpy")
    ref = V.trefftz_normalwash(q, qn, p, g, d, rc)
    V.set_kernel_backend("cython")
    new = V.trefftz_normalwash(q, qn, p, g, d, rc)
    assert np.max(np.abs(new - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_cython_threads_give_the_same_result():
    """One thread and four threads give the same velocity tensor."""
    pytest.importorskip("ventorum.aero.vortex_cython")
    from ventorum.aero import vortex_cython as cy
    from ventorum.aero import vortex_numba as nb

    ac = _wing_and_tail()
    lat = build_lattice(ac, vt.SolverSettings(n_panels=8, n_chord=3), collocation="vlm")
    n = lat.n_panels
    src = build_sources(lat, np.array([1.0, 0.0, 0.05]), UnknownMap(np.arange(n), np.arange(n), False), None)
    tg = panel_targets(lat, np.arange(n))
    a, b, a_te, b_te, d, rc2, grp, sign = nb.pack_sources(src)
    tg_group, tg_rc2, use_tg = nb.pack_targets(n, src, tg)
    Pc = np.ascontiguousarray(lat.cp, dtype=float)
    t1 = cy.velocity_tensor_kernel(Pc, a, b, a_te, b_te, d, rc2, grp, sign, tg_group, tg_rc2, use_tg, 0, 1)
    t4 = cy.velocity_tensor_kernel(Pc, a, b, a_te, b_te, d, rc2, grp, sign, tg_group, tg_rc2, use_tg, 0, 4)
    assert np.max(np.abs(t1 - t4)) <= 1e-13 * np.max(np.abs(t1))


def test_cython_full_threads_give_the_same_result():
    """The full core count gives the same result as one thread.

    Regression test for a race: the per-pair results were written through
    pointers to shared variables, so with many threads the sum was wrong.
    """
    pytest.importorskip("ventorum.aero.vortex_cython")
    from ventorum.aero import vortex_cython as cy
    from ventorum.aero import vortex_numba as nb
    from ventorum.utils.parallel import cpu_cores

    ac = _wing_and_tail()
    lat = build_lattice(ac, vt.SolverSettings(n_panels=8, n_chord=3), collocation="vlm")
    n = lat.n_panels
    src = build_sources(lat, np.array([1.0, 0.0, 0.05]), UnknownMap(np.arange(n), np.arange(n), False), None)
    tg = panel_targets(lat, np.arange(n))
    a, b, a_te, b_te, d, rc2, grp, sign = nb.pack_sources(src)
    tg_group, tg_rc2, use_tg = nb.pack_targets(n, src, tg)
    Pc = np.ascontiguousarray(lat.cp, dtype=float)
    t1 = cy.velocity_tensor_kernel(Pc, a, b, a_te, b_te, d, rc2, grp, sign, tg_group, tg_rc2, use_tg, 0, 1)
    t_full = cy.velocity_tensor_kernel(Pc, a, b, a_te, b_te, d, rc2, grp, sign, tg_group, tg_rc2, use_tg, 0, cpu_cores())
    assert np.max(np.abs(t1 - t_full)) <= 1e-13 * np.max(np.abs(t1))


@pytest.mark.parametrize("with_ground", [False, True])
def test_analysis_with_cython_backend_equals_numba_backend(with_ground):
    pytest.importorskip("ventorum.aero.vortex_cython")
    ac = _wing_and_tail()
    st = vt.SolverSettings(n_panels=8, n_chord=3)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    if with_ground:
        cond.h = 0.8
    old = V._kernel_backend  # the setting ("auto" stays "auto"), not the resolved backend
    try:
        V.set_kernel_backend("numba")
        numba_totals = vt.analyze(ac, condition=cond, settings=st).totals
        V.set_kernel_backend("cython")
        cython_totals = vt.analyze(ac, condition=cond, settings=st).totals
    finally:
        V.set_kernel_backend(old)
    for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn"):
        ref = getattr(numba_totals, name)
        new = getattr(cython_totals, name)
        assert abs(new - ref) <= 1e-11 * abs(ref) or abs(new - ref) <= 1e-14


def test_cython_backend_switch():
    pytest.importorskip("ventorum.aero.vortex_cython")
    old = V._kernel_backend  # the setting ("auto" stays "auto"), not the resolved backend
    try:
        V.set_kernel_backend("cython")
        assert V.get_kernel_backend() == "cython"
    finally:
        V.set_kernel_backend(old)


def test_cython_openmp_flag():
    """The extension reports its OpenMP build; CI requires OpenMP on Linux and Windows."""
    pytest.importorskip("ventorum.aero.vortex_cython")
    from ventorum.aero import vortex_cython as cy

    flag = cy.openmp_enabled()
    assert isinstance(flag, bool)
    if os.environ.get("VENTORUM_REQUIRE_OPENMP", "") == "1":
        assert flag, "The Cython kernels were compiled without OpenMP: they run in one thread."


_TORCH_AND_CYTHON = """
import concurrent.futures as cf
import numpy as np
{first}
{second}
from ventorum.aero import vortex_cython as cy

rng = np.random.default_rng(0)
m, n = 64, 48
P = rng.normal(size=(m, 3))
a, b, a_te, b_te = (rng.normal(size=(n, 3)) for _ in range(4))
d = np.tile([1.0, 0.0, 0.0], (n, 1))
rc2 = np.full(n, 1e-4)
grp = np.zeros(n, dtype=np.int64)
g = rng.normal(size=n)
tg_group = np.zeros(m, dtype=np.int64)
tg_rc2 = np.zeros(m)


def run(_):
    return cy.induced_velocity_kernel(P, a, b, a_te, b_te, d, rc2, grp, g, tg_group, tg_rc2, False, 0, 4)


with cf.ThreadPoolExecutor(2) as ex:
    out = list(ex.map(run, range(4)))
assert all(np.array_equal(out[0], o) for o in out)
x = torch.ones(256, 256)
(x @ x).sum().item()
print("ok")
"""


@pytest.mark.parametrize("torch_first", [True, False])
def test_cython_kernels_run_in_a_process_with_pytorch(torch_first):
    """Regression test: the OpenMP runtime of the Cython kernels must coexist with PyTorch.

    On macOS, PyTorch has its own copy of libomp. Two copies in one process
    can stop the process. The check runs in a subprocess so that a crash
    gives a test failure with its error message, not a stopped test run.
    """
    pytest.importorskip("ventorum.aero.vortex_cython")
    pytest.importorskip("torch")
    import subprocess
    import sys

    imports = ["import torch", "from ventorum.aero import vortex_cython"]
    first, second = imports if torch_first else imports[::-1]
    code = _TORCH_AND_CYTHON.format(first=first, second=second)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and "ok" in proc.stdout, (
        f"exit code {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


@pytest.mark.parametrize("with_ground", [False, True])
def test_cython_python_threads_equal_openmp_threads(monkeypatch, with_ground):
    """The row blocks on Python threads (the build without OpenMP) give the same bits as OpenMP."""
    pytest.importorskip("ventorum.aero.vortex_cython")
    from ventorum.utils import parallel as par

    ac = _wing_and_tail()
    lat = build_lattice(ac, vt.SolverSettings(n_panels=8, n_chord=3), collocation="vlm")
    n = lat.n_panels
    ground = make_ground_plane(lat, 0.8, np.radians(4.0), 0.0, np.radians(2.0), height_ref="min") if with_ground else None
    src = build_sources(lat, np.array([1.0, 0.0, 0.05]), UnknownMap(np.arange(n), np.arange(n), False), ground)
    tg = panel_targets(lat, np.arange(n))
    gamma = np.linspace(0.5, 2.0, n)
    q = lat.force_points
    q_normal = np.tile([0.0, 0.0, 1.0], (q.shape[0], 1))

    def run():
        with par.forced_single_threads(4), par.solve_threads(n):
            return (V.induced_velocity(lat.force_points, src, gamma, tg),
                    V.influence_matrix(lat.cp, lat.normal_bc, src, n, tg),
                    V.horseshoe_velocity_tensor(lat.cp, src, tg),
                    V.trefftz_normalwash(q, q_normal, q[::-1].copy(), gamma[: q.shape[0]], np.array([1.0, 0.0, 0.0]), 0.01))

    old = V._kernel_backend  # the setting ("auto" stays "auto"), not the resolved backend
    try:
        V.set_kernel_backend("cython")
        monkeypatch.setattr(V, "_CY_OPENMP", True)
        ref = run()
        monkeypatch.setattr(V, "_CY_OPENMP", False)
        monkeypatch.setattr(V, "_CY_MIN_ROWS", 1)  # many small blocks, also for this small lattice
        new = run()
    finally:
        V.set_kernel_backend(old)
    for r, x in zip(ref, new):
        assert r.shape == x.shape
        assert np.array_equal(r, x)


_TORCH_AND_NUMBA = """
import concurrent.futures as cf
import numpy as np
{first}
{second}
import numba
import ventorum as vt
from ventorum.aero import vortex as V

V.set_kernel_backend("numba")
wing = vt.LiftingSurface(name="W", semi_span=5.0, sections=[
    vt.WingSection(y_frac=0.0, chord=1.6), vt.WingSection(y_frac=1.0, chord=0.9)])
ac = vt.Aircraft(name="W", surfaces=[wing])
st = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=4)
ref = vt.analyze(ac, alpha_deg=4.0, settings=st).totals.CL
res = vt.analyze_sweep(ac, np.array([4.0, 4.0, 4.0, 4.0]), settings=st, n_jobs=2)
x = torch.ones(256, 256)
(x @ x).sum().item()
print("ok", numba.threading_layer())
"""


@pytest.mark.parametrize("torch_first", [True, False])
def test_numba_kernels_run_in_a_process_with_pytorch(torch_first):
    """Regression test: the threading layer of Numba must coexist with PyTorch.

    On macOS the OpenMP layer of Numba and PyTorch can load two copies of
    libomp. The check runs in a subprocess, with a parallel sweep, so that a
    crash gives a test failure with its error message.
    """
    pytest.importorskip("torch")
    import subprocess
    import sys

    imports = ["import torch", "import ventorum"]
    first, second = imports if torch_first else imports[::-1]
    code = _TORCH_AND_NUMBA.format(first=first, second=second)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and "ok" in proc.stdout, (
        f"exit code {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


def _torch_cpu(monkeypatch):
    """Skip when PyTorch is missing; else force the CPU device for the test."""
    pytest.importorskip("torch")
    from ventorum.aero import vortex_torch as vtorch

    if not vtorch.HAVE_TORCH:
        pytest.skip("PyTorch is not installed.")
    monkeypatch.setenv("VENTORUM_TORCH_DEVICE", "cpu")


def _need_cuda():
    """Skip when PyTorch or a CUDA GPU is missing."""
    torch = pytest.importorskip("torch")
    from ventorum.aero import vortex_torch as vtorch

    if not vtorch.HAVE_TORCH:
        pytest.skip("PyTorch is not installed.")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is not available.")
    return torch


def _torch_case(with_ground):
    """Build the lattice, sources, targets and strengths of the parity tests."""
    ac = _wing_and_tail()
    lat = build_lattice(ac, vt.SolverSettings(n_panels=8, n_chord=3), collocation="vlm")
    n = lat.n_panels
    ground = make_ground_plane(lat, 0.8, np.radians(4.0), 0.0, np.radians(2.0), height_ref="min") if with_ground else None
    src = build_sources(lat, np.array([1.0, 0.0, 0.05]), UnknownMap(np.arange(n), np.arange(n), False), ground)
    tg = panel_targets(lat, np.arange(n))
    gamma = np.linspace(0.5, 2.0, n)
    return lat, src, tg, gamma, n


def _torch_outputs(backend):
    """Run the three horseshoe kernels of one backend on the small lattice."""
    lat, src, tg, gamma, n = _torch_case(False)
    V.set_kernel_backend(backend)
    return (V.horseshoe_velocity_tensor(lat.cp, src, tg),
            V.influence_matrix(lat.cp, lat.normal_bc, src, n, tg),
            V.induced_velocity(lat.force_points, src, gamma, tg))


@pytest.mark.parametrize("with_ground", [False, True])
def test_torch_kernels_equal_numpy_kernels(numpy_backend, monkeypatch, with_ground):
    _torch_cpu(monkeypatch)
    lat, src, tg, gamma, n = _torch_case(with_ground)
    out = {}
    for backend in ("numpy", "torch"):
        V.set_kernel_backend(backend)
        out[backend] = (
            V.horseshoe_velocity_tensor(lat.cp, src, tg),
            V.influence_matrix(lat.cp, lat.normal_bc, src, n, tg),
            V.induced_velocity(lat.force_points, src, gamma, tg),
        )
    for ref, new in zip(out["numpy"], out["torch"]):
        assert np.max(np.abs(new - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_torch_kernel_parts_add_up(monkeypatch):
    """Part 1 (fixed legs) plus part 2 (wake legs) equals part 0 (all legs)."""
    _torch_cpu(monkeypatch)
    from ventorum.aero import vortex_numba as nb
    from ventorum.aero import vortex_torch as vtorch

    lat, src, tg, gamma, n = _torch_case(False)
    a, b, a_te, b_te, d, rc2, grp, sign = nb.pack_sources(src)
    tg_group, tg_rc2, use_tg = nb.pack_targets(n, src, tg)
    args = (np.ascontiguousarray(lat.cp, dtype=float), np.ascontiguousarray(lat.normal_bc, dtype=float),
            a, b, a_te, b_te, d, rc2, grp, sign,
            np.ascontiguousarray(src.column, dtype=np.int64), n, tg_group, tg_rc2, use_tg)
    A0 = vtorch.normal_influence_kernel(*args, 0)
    A1 = vtorch.normal_influence_kernel(*args, 1)
    A2 = vtorch.normal_influence_kernel(*args, 2)
    assert np.max(np.abs(A1 + A2 - A0)) <= 1e-13 * np.max(np.abs(A0))
    g_src = gamma[src.column] * src.sign
    gc = np.ascontiguousarray(g_src, dtype=float)
    iargs = (np.ascontiguousarray(lat.force_points, dtype=float), a, b, a_te, b_te, d, rc2, grp, gc,
             tg_group, tg_rc2, use_tg)
    v0 = vtorch.induced_velocity_kernel(*iargs, 0)
    v1 = vtorch.induced_velocity_kernel(*iargs, 1)
    v2 = vtorch.induced_velocity_kernel(*iargs, 2)
    assert np.max(np.abs(v1 + v2 - v0)) <= 1e-13 * np.max(np.abs(v0))


def test_torch_trefftz_kernel_equals_numpy(numpy_backend, monkeypatch):
    _torch_cpu(monkeypatch)
    rng = np.random.default_rng(3)
    q = rng.normal(size=(40, 3))
    qn = rng.normal(size=(40, 3))
    p = rng.normal(size=(60, 3))
    g = rng.normal(size=60)
    d = np.array([1.0, 0.0, 0.0])
    rc = rng.uniform(0.0, 0.1, size=60)
    V.set_kernel_backend("numpy")
    ref = V.trefftz_normalwash(q, qn, p, g, d, rc)
    V.set_kernel_backend("torch")
    new = V.trefftz_normalwash(q, qn, p, g, d, rc)
    assert np.max(np.abs(new - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_torch_chunk_size_does_not_change_the_result(monkeypatch):
    """A chunk of 1000 values gives the same bits as the default chunk."""
    _torch_cpu(monkeypatch)
    lat, src, tg, gamma, n = _torch_case(False)
    q = lat.force_points
    q_normal = np.tile([0.0, 0.0, 1.0], (q.shape[0], 1))
    V.set_kernel_backend("torch")

    def run():
        return (V.horseshoe_velocity_tensor(lat.cp, src, tg),
                V.influence_matrix(lat.cp, lat.normal_bc, src, n, tg),
                V.induced_velocity(lat.force_points, src, gamma, tg),
                V.trefftz_normalwash(q, q_normal, q[::-1].copy(), gamma[: q.shape[0]],
                                     np.array([1.0, 0.0, 0.0]), 0.01))

    old = V._kernel_backend  # the setting ("auto" stays "auto"), not the resolved backend
    try:
        monkeypatch.setenv("VENTORUM_TORCH_CHUNK", "1000")
        small = run()
        monkeypatch.delenv("VENTORUM_TORCH_CHUNK")
        big = run()
    finally:
        V.set_kernel_backend(old)
    for a, b in zip(small, big):
        assert a.shape == b.shape
        assert np.array_equal(a, b)


def test_analysis_with_torch_backend_equals_numba_backend(monkeypatch):
    _torch_cpu(monkeypatch)
    ac = _wing_and_tail()
    st = vt.SolverSettings(n_panels=8, n_chord=3)
    old = V._kernel_backend  # the setting ("auto" stays "auto"), not the resolved backend
    try:
        for with_ground in (False, True):
            cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
            if with_ground:
                cond.h = 0.8
            V.set_kernel_backend("numba")
            numba_totals = vt.analyze(ac, condition=cond, settings=st).totals
            V.set_kernel_backend("torch")
            torch_totals = vt.analyze(ac, condition=cond, settings=st).totals
            for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn"):
                ref = getattr(numba_totals, name)
                new = getattr(torch_totals, name)
                assert abs(new - ref) <= 1e-11 * abs(ref) or abs(new - ref) <= 1e-14
    finally:
        V.set_kernel_backend(old)


def test_torch_backend_switch(monkeypatch):
    pytest.importorskip("torch")
    from ventorum.aero import vortex_torch as vtorch

    if not vtorch.HAVE_TORCH:
        pytest.skip("PyTorch is not installed.")
    old = V._kernel_backend  # the setting ("auto" stays "auto"), not the resolved backend
    try:
        V.set_kernel_backend("torch")
        assert V.get_kernel_backend() == "torch"
    finally:
        V.set_kernel_backend(old)
    # Without torch, selecting the backend raises ValueError.
    monkeypatch.setattr(V, "_HAVE_TORCH", False)
    with pytest.raises(ValueError):
        V.set_kernel_backend("torch")


@pytest.mark.gpu
def test_cuda_kernels_equal_numpy_kernels(numpy_backend, monkeypatch):
    _need_cuda()
    monkeypatch.setenv("VENTORUM_TORCH_DEVICE", "cuda")
    for with_ground in (False, True):
        lat, src, tg, gamma, n = _torch_case(with_ground)
        out = {}
        for backend in ("numpy", "torch"):
            V.set_kernel_backend(backend)
            out[backend] = (
                V.horseshoe_velocity_tensor(lat.cp, src, tg),
                V.influence_matrix(lat.cp, lat.normal_bc, src, n, tg),
                V.induced_velocity(lat.force_points, src, gamma, tg),
            )
        for ref, new in zip(out["numpy"], out["torch"]):
            assert np.max(np.abs(new - ref)) <= 1e-11 * np.max(np.abs(ref))


@pytest.mark.gpu
def test_cuda_analysis_equals_cpu_analysis(monkeypatch):
    torch = _need_cuda()
    ac = _wing_and_tail()
    st = vt.SolverSettings(n_panels=8, n_chord=3)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    old = V._kernel_backend  # the setting ("auto" stays "auto"), not the resolved backend
    try:
        V.set_kernel_backend("torch")
        monkeypatch.setenv("VENTORUM_TORCH_DEVICE", "cpu")
        cpu_totals = vt.analyze(ac, condition=cond, settings=st).totals
        monkeypatch.setenv("VENTORUM_TORCH_DEVICE", "cuda")
        torch.cuda.synchronize()
        gpu_totals = vt.analyze(ac, condition=cond, settings=st).totals
    finally:
        V.set_kernel_backend(old)
    for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn"):
        ref = getattr(cpu_totals, name)
        new = getattr(gpu_totals, name)
        assert abs(new - ref) <= 1e-11 * abs(ref) or abs(new - ref) <= 1e-14


@pytest.mark.gpu
def test_cuda_tuner_measures_torch(monkeypatch):
    _need_cuda()
    monkeypatch.setenv("VENTORUM_TORCH_DEVICE", "cuda")
    from ventorum.hardware import tuner
    from ventorum.utils import parallel as par

    assert "torch" in tuner._available_backends()
    monkeypatch.setattr(tuner, "_CASES", {"small": (6, 2)})
    monkeypatch.setattr(tuner, "_REPEATS", {"small": 1})
    ac, st = tuner._wing(*tuner._CASES["small"])
    choices, times = tuner._measure_kernels("small", ac, st, par.cpu_cores())
    for kernel in V.KERNELS:
        assert "torch" in times[kernel]
        assert np.isfinite(times[kernel]["torch"])
        assert choices[kernel] in tuner._available_backends()


def test_import_does_not_load_torch():
    """Importing Ventorum must not import PyTorch (about 3 s, and an OpenMP clash on macOS).

    The torch backend loads PyTorch on its first use only.
    """
    import subprocess
    import sys

    code = "import sys, ventorum, ventorum.aero.vortex; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def _straight_horseshoes(n=48, with_images=False, with_groups=False):
    """Straight horseshoes with zero-length chordwise segments (a_te is a, b_te is b).

    This is the input of ``validation/legacy_speed.py``: the bent-horseshoe
    kernels must give the same matrix as the legacy straight-horseshoe
    kernels, and the zero-length legs must add nothing.
    """
    y = np.linspace(-5.0, 5.0, n + 1)
    a = np.column_stack([np.full(n, 0.25), y[:-1], np.zeros(n)])
    b = np.column_stack([np.full(n, 0.25), y[1:], np.zeros(n)])
    td = np.array([np.cos(np.radians(5.0)), 0.0, np.sin(np.radians(5.0))])
    P = np.column_stack([np.full(n, 0.75), 0.5 * (y[:-1] + y[1:]), np.full(n, 0.05)])
    nrm = np.tile([0.0, 0.0, 1.0], (n, 1))
    group = (np.arange(n) % 2).astype(np.int64) if with_groups else None
    hs = V.HorseshoeSet(a=a, b=b, a_te=a.copy(), b_te=b.copy(), wake_dir=np.tile(td, (n, 1)),
                        rc=np.full(n, 1e-4), sign=np.ones(n),
                        column=np.arange(n, dtype=np.int64), group=group)
    if with_images:
        # Mirror images in the plane z = -h, with sign -1 and the same
        # columns (two sources per unknown, as with a ground plane).
        h = 0.8
        img = V.HorseshoeSet(a=a * [1.0, 1.0, -1.0] + [0.0, 0.0, -2.0 * h],
                            b=b * [1.0, 1.0, -1.0] + [0.0, 0.0, -2.0 * h],
                            a_te=a * [1.0, 1.0, -1.0] + [0.0, 0.0, -2.0 * h],
                            b_te=b * [1.0, 1.0, -1.0] + [0.0, 0.0, -2.0 * h],
                            wake_dir=np.tile(td * [1.0, 1.0, -1.0], (n, 1)),
                            rc=np.full(n, 1e-4), sign=-np.ones(n),
                            column=np.arange(n, dtype=np.int64), group=group)
        hs = hs.concatenate(img)
    if with_groups:
        tg = V.Targets(group=(np.arange(n) % 2).astype(np.int64), rc=np.full(n, 0.02))
    else:
        tg = None
    gamma = np.linspace(0.5, 2.0, n)
    return P, nrm, hs, tg, gamma, n


@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize("with_groups", [False, True])
@pytest.mark.parametrize("backend", ["numba", "cython"])
def test_straight_horseshoes_equal_numpy(numpy_backend, backend, with_images, with_groups):
    """Compiled kernels agree with numpy on straight horseshoes (zero-length chordwise legs)."""
    if backend == "cython":
        pytest.importorskip("ventorum.aero.vortex_cython")
    P, nrm, hs, tg, gamma, n = _straight_horseshoes(with_images=with_images, with_groups=with_groups)
    out = {}
    for name in ("numpy", backend):
        V.set_kernel_backend(name)
        out[name] = (
            V.horseshoe_velocity_tensor(P, hs, tg),
            V.influence_matrix(P, nrm, hs, n, tg),
            V.induced_velocity(P, hs, gamma, tg),
        )
    for ref, new in zip(out["numpy"], out[backend]):
        assert np.max(np.abs(new - ref)) <= 1e-12 * np.max(np.abs(ref))


@pytest.mark.parametrize("backend", ["numba", "cython"])
def test_kernel_parts_add_up_on_straight_horseshoes(numpy_backend, backend):
    """Part 1 (fixed legs) plus part 2 (wake legs) equals part 0 (all legs), also with images."""
    if backend == "cython":
        cy = pytest.importorskip("ventorum.aero.vortex_cython")
    else:
        from ventorum.aero import vortex_numba as cy
    from ventorum.aero import vortex_numba as nb

    P, nrm, hs, tg, gamma, n = _straight_horseshoes(with_images=True, with_groups=True)
    a, b, a_te, b_te, d, rc2, grp, sign = nb.pack_sources(hs)
    tg_group, tg_rc2, use_tg = nb.pack_targets(P.shape[0], hs, tg)
    Pc = np.ascontiguousarray(P, dtype=float)
    Nc = np.ascontiguousarray(nrm, dtype=float)
    col = np.ascontiguousarray(hs.column, dtype=np.int64)
    extra = (1,) if backend == "cython" else ()
    args = (Pc, Nc, a, b, a_te, b_te, d, rc2, grp, sign, col, n, tg_group, tg_rc2, use_tg)
    A0 = cy.normal_influence_kernel(*args, 0, *extra)
    A1 = cy.normal_influence_kernel(*args, 1, *extra)
    A2 = cy.normal_influence_kernel(*args, 2, *extra)
    assert np.max(np.abs(A1 + A2 - A0)) <= 1e-13 * np.max(np.abs(A0))
    g_src = np.ascontiguousarray(gamma[hs.column] * hs.sign, dtype=float)
    iargs = (Pc, a, b, a_te, b_te, d, rc2, grp, g_src, tg_group, tg_rc2, use_tg)
    v0 = cy.induced_velocity_kernel(*iargs, 0, *extra)
    v1 = cy.induced_velocity_kernel(*iargs, 1, *extra)
    v2 = cy.induced_velocity_kernel(*iargs, 2, *extra)
    assert np.max(np.abs(v1 + v2 - v0)) <= 1e-13 * np.max(np.abs(v0))


# ── Test for ProgressBar.set_current (public function kept, with a test) ──────────

def test_progressbar_set_current():
    """Test that ProgressBar.set_current works (a public function that is kept, with a test)."""
    import io
    from ventorum.utils.progress import ProgressBar

    stream = io.StringIO()
    pb = ProgressBar(total=10, title="Test", unit="pts", stream=stream)
    pb.set_current(5, status="halfway")
    output = stream.getvalue()
    assert "Test" in output
    assert "5/10" in output
    assert "halfway" in output
    pb.finish("done")
    output = stream.getvalue()
    assert "COMPLETED" in output


def _trefftz_core_inputs():
    """Random Trefftz-plane inputs with two core groups and close point pairs."""
    rng = np.random.default_rng(13)
    q = rng.normal(size=(40, 3))
    qn = rng.normal(size=(40, 3))
    p = np.vstack([rng.normal(size=(50, 3)), q[:10] + 1e-3 * rng.normal(size=(10, 3))])
    g = rng.normal(size=60)
    d = np.array([1.0, 0.0, 0.0])
    rc = rng.uniform(0.0, 0.1, size=60)
    group = rng.integers(0, 2, size=60)
    targets = V.Targets(group=rng.integers(0, 2, size=40), rc=rng.uniform(0.05, 0.3, size=40))
    return q, qn, p, g, d, rc, group, targets


def test_trefftz_core_rule_numpy():
    """The numpy reference adds the target core only between different core groups."""
    q, qn, p, g, d, rc, group, targets = _trefftz_core_inputs()
    V.set_kernel_backend("numpy")
    got = V.trefftz_normalwash(q, qn, p, g, d, rc, group=group, targets=targets)
    want = np.zeros(q.shape[0])
    for i in range(q.shape[0]):
        for j in range(p.shape[0]):
            c2 = rc[j] ** 2 + (targets.rc[i] ** 2 if targets.group[i] != group[j] else 0.0)
            r = q[i] - p[j]
            want[i] += np.dot(np.cross(d, r), qn[i]) * g[j] / (2.0 * np.pi * (r @ r + c2))
    assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))
    plain = V.trefftz_normalwash(q, qn, p, g, d, rc)
    assert np.max(np.abs(got - plain)) > 1e-6 * np.max(np.abs(plain))


@pytest.mark.parametrize("backend", ["numba", "cython", "torch"])
def test_trefftz_core_parity(backend, numpy_backend, monkeypatch):
    """Each backend gives the numpy result with the cross-surface core."""
    if backend == "cython":
        pytest.importorskip("ventorum.aero.vortex_cython")
    if backend == "torch":
        _torch_cpu(monkeypatch)
    q, qn, p, g, d, rc, group, targets = _trefftz_core_inputs()
    V.set_kernel_backend("numpy")
    ref = V.trefftz_normalwash(q, qn, p, g, d, rc, group=group, targets=targets)
    V.set_kernel_backend(backend)
    new = V.trefftz_normalwash(q, qn, p, g, d, rc, group=group, targets=targets)
    assert np.max(np.abs(new - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_trefftz_core_parity_cuda(numpy_backend, monkeypatch):
    """The torch kernel on a CUDA GPU gives the numpy result with the cross-surface core."""
    _need_cuda()
    monkeypatch.setenv("VENTORUM_TORCH_DEVICE", "cuda")
    q, qn, p, g, d, rc, group, targets = _trefftz_core_inputs()
    V.set_kernel_backend("numpy")
    ref = V.trefftz_normalwash(q, qn, p, g, d, rc, group=group, targets=targets)
    V.set_kernel_backend("torch")
    new = V.trefftz_normalwash(q, qn, p, g, d, rc, group=group, targets=targets)
    assert np.max(np.abs(new - ref)) <= 1e-12 * np.max(np.abs(ref))


def _unknowns_case(with_ground):
    """Wing and tail lattice, symmetric unknown map, optional ground images."""
    from ventorum.aero.system import make_unknown_map

    ac = _wing_and_tail()
    lat = build_lattice(ac, vt.SolverSettings(n_panels=8), collocation="llt")
    umap = make_unknown_map(lat, True)
    ground = make_ground_plane(lat, 0.6, np.radians(3.0)) if with_ground else None
    hs = build_sources(lat, np.array([1.0, 0.0, 0.05]), umap, ground)
    P = lat.cp[umap.unknown_panels]
    return P, hs, umap.n, panel_targets(lat, umap.unknown_panels)


@pytest.mark.parametrize("backend", ["numba", "cython"])
@pytest.mark.parametrize("with_ground", [False, True])
def test_velocity_unknowns_kernel_parity(backend, with_ground, numpy_backend):
    """The fused kernel equals the tensor-plus-fold path of the same backend bit for bit, and numpy to 1e-12."""
    if backend == "cython":
        pytest.importorskip("ventorum.aero.vortex_cython")
    P, hs, n_unk, tg = _unknowns_case(with_ground)
    V.set_kernel_backend(backend)
    fused = V.velocity_tensor_unknowns(P, hs, n_unk, tg)
    folded = V._fold_columns(V.horseshoe_velocity_tensor(P, hs, tg), hs.column, n_unk)
    assert np.array_equal(fused, folded)
    V.set_kernel_backend("numpy")
    ref = V.velocity_tensor_unknowns(P, hs, n_unk, tg)
    assert np.max(np.abs(fused - ref)) <= 1e-12 * np.max(np.abs(ref))
