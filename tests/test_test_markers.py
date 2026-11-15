# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Check that each test that needs a CUDA GPU carries the ``gpu`` marker.

The GPU gate (``validation/gpu_gate.py``) runs ``pytest -m gpu``. A GPU test
without the marker is deselected there, and on a machine without a GPU it
skips, so no run tests it. This file reads the test files (static scan of
the source, no import) and finds the tests that skip, or return early, when
there is no CUDA GPU (the GPU pipelines need CUDA and NVIDIA Warp). Each of
them must have the marker on the test, on its class, on the module
(``pytestmark``) or on a ``pytest.param`` of its parametrization (also
through a module-level list of parameters).

A test skips on a missing GPU when its body (or a module helper that it
calls) has an ``if`` statement whose condition has one of the checks below
after ``not``, and whose body calls ``pytest.skip`` or returns, or when it
has a ``pytest.mark.skipif`` decorator with such a condition:

* ``gpu.available()`` (the GPU pipelines of :mod:`ventorum.gpu`);
* ``<x>.cuda.is_available()`` (PyTorch with CUDA);
* ``<x>.is_cuda_available()`` (NVIDIA Warp with CUDA);
* ``<x>.available(<backend>, "cuda")`` (a kernel backend on CUDA).

A test that needs only the Warp package and runs on the Warp CPU device
(``pytest.importorskip("warp")`` without a CUDA check) is not a GPU test.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def _is_cuda_check(call: ast.AST) -> bool:
    """Return True if *call* asks whether CUDA (or the GPU pipelines) can run."""
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
        return False
    attr, owner = call.func.attr, call.func.value
    if attr == "is_cuda_available":
        return True
    if attr == "is_available":
        return isinstance(owner, ast.Attribute) and owner.attr == "cuda"
    if attr == "available":
        if isinstance(owner, ast.Name) and owner.id == "gpu":
            return True
        return any(isinstance(arg, ast.Constant) and arg.value == "cuda" for arg in call.args)
    return False


def _is_gpu_absent_check(node: ast.AST) -> bool:
    """Return True if *node* has ``not`` before a CUDA check (see :func:`_is_cuda_check`)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.UnaryOp) and isinstance(sub.op, ast.Not):
            operand = sub.operand
            if any(_is_cuda_check(inner) for inner in ast.walk(operand)):
                return True
    return False


def _is_skip_call(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "skip" and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pytest")


def _skips_directly(func: ast.AST) -> bool:
    """Return True if the body of *func* skips or returns when the GPU is missing."""
    for sub in ast.walk(func):
        if isinstance(sub, ast.If) and _is_gpu_absent_check(sub.test):
            for stmt in sub.body:
                for inner in ast.walk(stmt):
                    if _is_skip_call(inner) or isinstance(inner, ast.Return):
                        return True
    return False


def _called_names(func: ast.AST) -> set[str]:
    return {sub.func.id for sub in ast.walk(func)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)}


def _has_gpu_mark(decorators: list[ast.expr], marked_names: frozenset[str] = frozenset()) -> bool:
    """Return True if a decorator has ``mark.gpu`` or names a module list that has it."""
    for dec in decorators:
        if "mark.gpu" in ast.unparse(dec):
            return True
        if any(isinstance(sub, ast.Name) and sub.id in marked_names for sub in ast.walk(dec)):
            return True
    return False


def _skipif_gpu(decorators: list[ast.expr]) -> bool:
    for dec in decorators:
        if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "skipif"
                and dec.args and _is_gpu_absent_check(dec.args[0])):
            return True
    return False


def _module_marked(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "pytestmark" for t in node.targets):
            if "mark.gpu" in ast.unparse(node.value):
                return True
    return False


def unmarked_gpu_tests(path: Path) -> list[str]:
    """Return ``"file:line name"`` of each test in *path* that skips without a GPU and has no ``gpu`` marker."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if _module_marked(tree):
        return []
    marked_names = frozenset(
        target.id for node in tree.body if isinstance(node, ast.Assign) and "mark.gpu" in ast.unparse(node.value)
        for target in node.targets if isinstance(target, ast.Name))
    helpers = {node.name for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and not node.name.startswith("test") and _skips_directly(node)}
    out: list[str] = []

    def visit(nodes: list[ast.stmt], class_marked: bool) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                visit(node.body, class_marked or _has_gpu_mark(node.decorator_list, marked_names))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                needs = (_skips_directly(node) or bool(_called_names(node) & helpers)
                         or _skipif_gpu(node.decorator_list))
                if needs and not (class_marked or _has_gpu_mark(node.decorator_list, marked_names)):
                    out.append(f"{path.name}:{node.lineno} {node.name}")

    visit(tree.body, False)
    return out


def test_scan_finds_an_unmarked_gpu_test(tmp_path):
    """The scan finds the skip forms and accepts the marker forms."""
    src = '''
import pytest
from ventorum import gpu


def _need_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")


def test_helper():
    _need_cuda()


def test_early_return():
    if not gpu.available():
        return


def test_warp_cuda():
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")


def test_backend_on_cuda(K):
    if not K.available("warp", "cuda"):
        pytest.skip("no Warp CUDA backend")


def test_warp_cpu_device():
    pytest.importorskip("warp")


def test_cpu_backend(K):
    if not K.available("numba", "cpu"):
        pytest.skip("no Numba")


_DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("device", _DEVICES)
def test_param_list(device):
    if device == "gpu" and not gpu.available():
        pytest.skip("no GPU")


@pytest.mark.skipif(not gpu.available(), reason="no GPU")
def test_skipif():
    pass


@pytest.mark.gpu
def test_marked():
    _need_cuda()


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_param(device):
    if device == "gpu" and not gpu.available():
        pytest.skip("no GPU")


def test_cpu_path():
    if gpu.available():
        pytest.skip("checks the path without a GPU")
'''
    path = tmp_path / "test_sample.py"
    path.write_text(src, encoding="utf-8")
    names = [entry.split(" ", 1)[1] for entry in unmarked_gpu_tests(path)]
    assert names == ["test_helper", "test_early_return", "test_warp_cuda", "test_backend_on_cuda", "test_skipif"]


def test_every_gpu_test_has_the_gpu_marker():
    """Each test that skips without CUDA or Warp has the ``gpu`` marker."""
    missing = [entry for path in sorted(TESTS.glob("test_*.py")) for entry in unmarked_gpu_tests(path)]
    assert not missing, ("These tests skip without a GPU but have no gpu marker, so the GPU gate "
                         "does not run them: " + ", ".join(missing))
