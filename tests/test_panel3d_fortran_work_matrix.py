"""Check exact CPU work matrices and their solver use without timing."""

from dataclasses import fields
from unittest.mock import Mock

import numba
import numpy as np
import pytest

from ventorum.airfoil import airfoil
from ventorum.core.datatypes import FlightCondition, LiftingSurface, WingSection
from ventorum.panel3d import kernels_numba as kernels
from ventorum.panel3d import solver as solver_module
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.solver import PanelSolver


def _bits(actual, expected):
    """Compare values, NaN positions and non-NaN float64 bit patterns."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.array_equal(actual, expected, equal_nan=True)
    if actual.dtype == np.float64:
        valid = ~np.isnan(expected)
        assert np.array_equal(actual[valid].view(np.uint64), expected[valid].view(np.uint64))


def _layout(values, order):
    if order == "strided":
        backing = np.empty((2 * values.shape[0], 2 * values.shape[1]))
        view = backing[::-2, ::2]
        view[:] = values
        return view
    return np.array(values, order=order, copy=True)


@pytest.mark.parametrize("shape", [(0, 0), (0, 7), (7, 0), (1, 7), (7, 1),
                                   (63, 65), (64, 64), (130, 129)])
@pytest.mark.parametrize("orders", [("C", "C"), ("F", "F"), ("C", "F"),
                                    ("F", "C"), ("strided", "strided")])
@pytest.mark.parametrize("provided", [False, True])
def test_kernel_layout_and_bits(shape, orders, provided):
    """Tile boundaries, empty axes and input strides preserve every element."""
    rng = np.random.default_rng(1729)
    a = _layout(rng.normal(size=shape), orders[0])
    b = _layout(rng.normal(size=shape), orders[1])
    a_before, b_before = a.copy(), b.copy()
    for function, args, reference in (
        (kernels.add_to_fortran_numba, (a, b), np.add(a, b, order="F")),
        (kernels.copy_to_fortran_numba, (a,), np.array(a, order="F", copy=True)),
    ):
        out = np.full(shape, np.nan, order="F") if provided else None
        actual = function(*args, out=out)
        assert actual.dtype == np.float64
        assert actual.flags.f_contiguous == reference.flags.f_contiguous
        assert actual.shape == reference.shape
        if provided:
            assert actual is out
        else:
            assert not np.shares_memory(actual, a)
            assert not np.shares_memory(actual, b)
        _bits(actual, reference)
    _bits(a, a_before)
    _bits(b, b_before)


@pytest.mark.parametrize("order", ["C", "F", "strided"])
def test_ieee_values(order):
    """Signed zeros, subnormals, infinities and NaNs keep NumPy semantics."""
    tiny = np.nextafter(0.0, 1.0)
    values = np.array([0.0, -0.0, tiny, -tiny, np.inf, -np.inf, np.nan, 1.0, -1.0])
    a = _layout(np.broadcast_to(values[:, None], (len(values), len(values))), order)
    b = _layout(np.broadcast_to(values[None, :], a.shape), order)
    with np.errstate(invalid="ignore"):
        reference = np.add(a, b, order="F")
    for out in (None, np.empty(a.shape, order="F")):
        _bits(kernels.add_to_fortran_numba(a, b, out=out), reference)
        _bits(kernels.copy_to_fortran_numba(a, out=out), np.array(a, order="F", copy=True))


@pytest.mark.parametrize("bad", [np.ones((3, 4), dtype=np.float32),
                                  np.ones((3, 4), dtype=np.int64),
                                  np.ones(12), np.ones((2, 3, 4)), [[1.0]]])
def test_refuse_invalid_inputs(bad):
    good = np.ones((3, 4))
    with pytest.raises(ValueError, match="float64|2D"):
        kernels.copy_to_fortran_numba(bad)
    with pytest.raises(ValueError, match="float64|2D"):
        kernels.add_to_fortran_numba(bad, good)
    with pytest.raises(ValueError, match="float64|2D"):
        kernels.add_to_fortran_numba(good, bad)


def test_refuse_mismatched_shapes():
    with pytest.raises(ValueError, match="shape"):
        kernels.add_to_fortran_numba(np.ones((3, 4)), np.ones((3, 1)))


@pytest.mark.parametrize("out", [np.empty((4, 3), order="F"), np.empty((3, 4)),
                                  np.empty((3, 4), dtype=np.float32, order="F"),
                                  np.empty((3, 4), dtype=int, order="F"),
                                  np.empty(12), np.empty((1, 3, 4)),
                                  np.empty((6, 8), order="F")[::2, ::2]])
def test_refuse_invalid_outputs(out):
    a = np.ones((3, 4))
    for function, args in ((kernels.add_to_fortran_numba, (a, a)),
                           (kernels.copy_to_fortran_numba, (a,))):
        with pytest.raises(ValueError, match="out"):
            function(*args, out=out)


def test_refuse_readonly_output():
    a = np.ones((3, 4))
    out = np.empty(a.shape, order="F")
    out.setflags(write=False)
    for function, args in ((kernels.add_to_fortran_numba, (a, a)),
                           (kernels.copy_to_fortran_numba, (a,))):
        with pytest.raises(ValueError, match="out"):
            function(*args, out=out)


def test_thread_count_keeps_bits():
    """One and two Numba threads produce the same bits."""
    a = np.random.default_rng(42).normal(size=(130, 129))
    b = a[::-1, ::-1]
    previous = numba.get_num_threads()
    try:
        results = []
        for threads in (1, 2):
            numba.set_num_threads(threads)
            results.append((kernels.add_to_fortran_numba(a, b), kernels.copy_to_fortran_numba(a)))
        for actual, expected in zip(*results, strict=True):
            _bits(actual, expected)
    finally:
        numba.set_num_threads(previous)


def _solver(**kwargs):
    wing = LiftingSurface(
        name="Wing", semi_span=2.0, is_symmetric=True,
        sections=[WingSection(y_frac=y, chord=1.0, shape="0012") for y in (0.0, 1.0)],
    )
    mesh = mesh_lifting_surface(
        wing, airfoil("0012", closed_te=True).coords, n_chord=6, n_span=5, half=True,
    )
    return PanelSolver(mesh, backend=kwargs.pop("backend", "numba"), **kwargs)


@pytest.mark.parametrize("ground_solver", ["direct", "krylov"])
def test_solver_bits_and_dispatch(ground_solver, monkeypatch):
    """Free air, ground and a Krylov sweep keep results, factors and unit solutions."""
    add = Mock(wraps=kernels.add_to_fortran_numba)
    copy = Mock(wraps=kernels.copy_to_fortran_numba)
    monkeypatch.setattr(kernels, "add_to_fortran_numba", add)
    monkeypatch.setattr(kernels, "copy_to_fortran_numba", copy)
    conditions = [FlightCondition(V_inf=25.0, alpha=0.04, p=0.01),
                  *[FlightCondition(V_inf=25.0, alpha=a, h=0.7) for a in (0.02, 0.04, 0.06)]]
    solver = _solver(ground_solver=ground_solver)
    actual = solver.solve_conditions(conditions)
    assert add.call_count > 0 and copy.call_count > 0
    if ground_solver == "krylov":
        assert any(cfg.ground_solver == "krylov" for cfg in solver._configs.values())
        assert not any(cfg.fallbacks for cfg in solver._configs.values())
    print(ground_solver, "kernel calls:", "add", add.call_count, "copy", copy.call_count)

    reference = _solver(ground_solver=ground_solver)
    monkeypatch.setattr(reference, "_add", lambda a, b: np.add(a, b, order="F"))
    monkeypatch.setattr(reference, "_copy", lambda a: np.array(a, order="F", copy=True))
    for result, expected in zip(actual, reference.solve_conditions(conditions), strict=True):
        for f in fields(result):
            if f.name == "condition":
                assert result.condition == expected.condition
            elif isinstance(getattr(result, f.name), (np.ndarray, float)):
                _bits(getattr(result, f.name), getattr(expected, f.name))
            else:
                assert getattr(result, f.name) == getattr(expected, f.name)
    for key, cfg in solver._configs.items():
        other = reference._configs[key]
        _bits(cfg.unit_mu, other.unit_mu)
        _bits(cfg.unit_grad, other.unit_grad)
        assert cfg.iterations == other.iterations
        assert cfg.relative_residuals == other.relative_residuals
        for parity, factors in cfg.systems.items():
            if factors is None:
                assert other.systems[parity] is None
            else:
                for a, b in zip(factors, other.systems[parity], strict=True):
                    _bits(a, b)

    add.reset_mock()
    copy.reset_mock()
    _solver(backend="numpy", ground_solver=ground_solver).solve_conditions(conditions)
    assert add.call_count == copy.call_count == 0


@pytest.mark.parametrize("backend", ["numba", "numpy"])
def test_c_order_work_matrix_numpy_call_counts(backend, monkeypatch):
    """Count NumPy Fortran conversions on the C-order solver work path."""
    a = np.arange(63 * 65, dtype=np.float64).reshape(63, 65)
    b = a + 0.5
    solver = _solver(backend=backend)
    # Compile before instrumentation, so compiler internals are excluded.
    kernels.add_to_fortran_numba(a, b)
    kernels.copy_to_fortran_numba(a)
    proxy = Mock(wraps=np)
    proxy.float64 = np.float64
    proxy.ndarray = np.ndarray
    monkeypatch.setattr(solver_module, "np", proxy)
    monkeypatch.setattr(kernels, "np", proxy)
    actual_add, actual_copy = solver._add(a, b), solver._copy(a)
    expected = 0 if backend == "numba" else 1
    assert proxy.add.call_count == proxy.array.call_count == expected
    assert actual_add.flags.f_contiguous and actual_copy.flags.f_contiguous
    _bits(actual_add, np.add(a, b, order="F"))
    _bits(actual_copy, a)
    print(backend, "C-order NumPy calls:", "add", proxy.add.call_count,
          "array", proxy.array.call_count)
