# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Regression tests: the numpy paths of the panel solvers do not import torch.

The 2D panel modules imported torch at the module level, and the 3D panel
solver imported torch in its LU solve before it checked the factor type.
Thus a CPU solve with SciPy factors failed when torch was not installed.
These tests run each numpy path in a new process that refuses torch.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.slow

# A None entry in sys.modules makes "import torch" fail and makes
# importlib.util.find_spec("torch") return None, as when torch is not installed.
_BLOCK_TORCH = 'import sys\nsys.modules["torch"] = None\n'


def _run_without_torch(body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a new Python process in which torch cannot be imported."""
    code = _BLOCK_TORCH + textwrap.dedent(body) + '\nassert sys.modules.get("torch") is None\nprint("OK")\n'
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)


def _check(proc: subprocess.CompletedProcess) -> None:
    """Assert that the process ended without error and printed OK."""
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.strip().endswith("OK"), proc.stdout[-3000:]


def test_panel2d_inviscid_numpy_without_torch():
    """Verify that the numpy 2D panel solve works when torch is not available."""
    _check(_run_without_torch(
        """
        import math
        from ventorum.airfoil import airfoil
        from ventorum.panel2d.solver import solve_inviscid

        res = solve_inviscid(airfoil("0012").coords, [0.0, 4.0])
        assert all(math.isfinite(float(c)) for c in res.Cl)
        res_ge = solve_inviscid(airfoil("0012").coords, 4.0, height=0.5)
        assert math.isfinite(float(res_ge.Cl))
        """
    ))


def test_panel3d_scipy_lu_without_torch():
    """Verify that a numpy 3D panel solve with SciPy factors works when torch is not available."""
    _check(_run_without_torch(
        """
        import math
        from ventorum import FlightCondition, LiftingSurface, WingSection
        from ventorum.airfoil import airfoil
        from ventorum.panel3d.mesh import mesh_lifting_surface
        from ventorum.panel3d.solver import PanelSolver

        wing = LiftingSurface(
            name="TestWing",
            semi_span=3.0,
            sections=[
                WingSection(y_frac=0.0, chord=1.0, shape="0012"),
                WingSection(y_frac=1.0, chord=0.8, shape="0012"),
            ],
        )
        mesh = mesh_lifting_surface(wing, airfoil("0012").coords, n_chord=8, n_span=6, half=True)
        res = PanelSolver(mesh, backend="numpy", cpu_lu="scipy").solve(FlightCondition(V_inf=35.0, alpha=math.radians(4.0)))
        assert math.isfinite(float(res.CL)) and float(res.CL) > 0.0
        """
    ))
