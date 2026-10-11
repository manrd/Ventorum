# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""One canonical class name per solver."""

from __future__ import annotations

import importlib
import re
import warnings
from pathlib import Path

import ventorum as vt
from ventorum.solvers.factory import make_solver

OLD_TO_NEW = {
    "HorseshoeSolver": "VortexLatticeSolver",
    "LinearSolver": "LinearLLTSolver",
    "NonlinearSolver": "NonlinearLLTSolver",
}

OLD_MODULES = {
    "HorseshoeSolver": ["ventorum", "ventorum.solvers", "ventorum.solvers.horseshoe"],
    "LinearSolver": ["ventorum", "ventorum.solvers", "ventorum.solvers.linear"],
    "NonlinearSolver": ["ventorum", "ventorum.solvers", "ventorum.solvers.nonlinear"],
}


def test_canonical_classes():
    """Canonical classes carry the canonical solver names."""
    assert vt.VortexLatticeSolver().name == "vlm"
    assert vt.LinearLLTSolver().name == "linear"
    assert vt.NonlinearLLTSolver().name == "nonlinear"
    assert vt.FourierSolver().name == "fourier"
    assert isinstance(make_solver("vlm"), vt.VortexLatticeSolver)
    assert isinstance(make_solver("linear"), vt.LinearLLTSolver)
    assert isinstance(make_solver("nonlinear"), vt.NonlinearLLTSolver)
    assert isinstance(make_solver("fourier"), vt.FourierSolver)


def test_old_names_warn_once_and_are_the_same_class():
    """Each old name warns once, names the new class, and is the same object."""
    for old, new in OLD_TO_NEW.items():
        canonical = getattr(vt, new)
        for module_name in OLD_MODULES[old]:
            module = importlib.import_module(module_name)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                obj = getattr(module, old)
            future = [w for w in caught if issubclass(w.category, FutureWarning)]
            assert len(future) == 1
            assert new in str(future[0].message)
            assert obj is canonical


def test_canonical_names_do_not_warn():
    """Canonical names and solver strings give no FutureWarning."""
    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.5),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert vt.VortexLatticeSolver().name == "vlm"
        assert vt.LinearLLTSolver().name == "linear"
        assert vt.NonlinearLLTSolver().name == "nonlinear"
        assert vt.FourierSolver().name == "fourier"
        for solver in ("vlm", "linear", "nonlinear"):
            vt.analyze(wing, alpha_deg=3.0, solver=solver, n_panels=12)
    assert [w for w in caught if issubclass(w.category, FutureWarning)] == []


def test_warning_points_at_user_code():
    """The deprecation warning points at the user code."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old_name = "HorseshoeSolver"
        getattr(vt, old_name)
    future = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert len(future) == 1
    assert Path(future[0].filename).name == Path(__file__).name


def test_no_old_names_in_package():
    """No file outside the alias code and the Old names table uses an old name."""
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"(?<!L\.)\b(HorseshoeSolver|LinearSolver|NonlinearSolver)\b")
    alias_modules = {
        "ventorum/utils/deprecation.py",
        "ventorum/solvers/horseshoe.py",
        "ventorum/solvers/linear.py",
        "ventorum/solvers/nonlinear.py",
        "ventorum/solvers/__init__.py",
        "ventorum/__init__.py",
    }
    bad: list[str] = []
    for base in ("ventorum", "examples", "validation"):
        for path in sorted((root / base).rglob("*.py")):
            rel = path.as_posix().split(str(root.as_posix()) + "/")[-1]
            if rel.startswith("ventorum/legacy/"):
                continue
            text = path.read_text(encoding="utf-8")
            if rel in alias_modules:
                text = _without_getattr(text)
                if rel == "ventorum/utils/deprecation.py":
                    continue
            if pattern.search(text):
                bad.append(rel)
    docs_base = root / "docs" / "user"
    for path in sorted(docs_base.rglob("*.md")):
        rel = path.as_posix().split(str(root.as_posix()) + "/")[-1]
        for line in path.read_text(encoding="utf-8").splitlines():
            if pattern.search(line) and "VortexLatticeSolver" not in line and "LinearLLTSolver" not in line \
                    and "NonlinearLLTSolver" not in line:
                bad.append(f"{rel}: {line.strip()}")
    assert bad == []


def _without_getattr(text: str) -> str:
    """Return *text* without the module __getattr__ block."""
    lines = text.splitlines()
    out: list[str] = []
    skip = False
    for line in lines:
        if line.startswith("def __getattr__"):
            skip = True
            continue
        if skip and (line.startswith("def ") or line.startswith("class ")):
            skip = False
        if not skip:
            out.append(line)
    return "\n".join(out)
