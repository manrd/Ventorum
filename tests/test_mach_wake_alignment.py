"""A body-axis wake with the compressibility correction is refused.

Regression test: with ``wake_alignment="body"`` and ``mach > 0`` the solve
used a wake along the free stream, while ``details["wake_dir"]`` and the
Trefftz plane used the body x-axis. The lift then jumped between
``mach=None`` and a very small Mach number. The combination is now refused
with ``ValidityError``.
"""

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.aero.loads import require_freestream_wake
from ventorum.aero.system import freestream_direction
from ventorum.core.errors import ValidityError
from ventorum.solvers.core import assemble_vlm
from ventorum.solvers.lattice_base import assemble_system_matrix

ALPHA = np.radians(10.0)


def _wing() -> vt.Aircraft:
    """Return a rectangular wing of aspect ratio 8 (semi-span 4 m, chord 1 m)."""
    return vt.Aircraft(surfaces=[vt.LiftingSurface(name="W", semi_span=4.0,
                                                   sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])])


def _settings(wake: str) -> vt.SolverSettings:
    return vt.SolverSettings(solver_type="vlm", n_panels=16, wake_alignment=wake)


def _solve(wake: str, mach):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vt.VortexLatticeSolver().solve(_wing(), vt.FlightCondition(alpha=ALPHA, V_inf=50.0, mach=mach),
                                              _settings(wake))


@pytest.mark.parametrize("mach", [1e-6, 1e-4, 0.3])
def test_body_wake_with_mach_is_refused(mach):
    with pytest.raises(ValidityError, match="wake_alignment='body'"):
        _solve("body", mach)


def test_body_wake_with_mach_is_refused_in_a_sweep_and_a_batch():
    solver = vt.VortexLatticeSolver()
    cond = vt.FlightCondition(V_inf=50.0, mach=0.2)
    with pytest.raises(ValidityError, match="wake_alignment='body'"):
        solver.solve_sweep(_wing(), cond, _settings("body"), np.radians([0.0, 4.0, 8.0]))
    with pytest.raises(ValidityError, match="wake_alignment='body'"):
        assemble_system_matrix(_wing(), vt.FlightCondition(alpha=ALPHA, mach=0.2), _settings("body"))


def test_low_level_mach_branches_refuse_a_body_wake():
    """assemble_vlm and the near-field loads refuse a wake that is not the free stream."""
    ac = _wing()
    ac.compute_reference_values()
    lattice = vt.VortexLatticeSolver().build(ac, _settings("freestream"), vt.FlightCondition(alpha=ALPHA))
    cond = vt.FlightCondition(alpha=ALPHA, mach=0.2)
    with pytest.raises(ValidityError, match="free stream"):
        assemble_vlm(lattice, cond, None, True, np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValidityError, match="free stream"):
        require_freestream_wake(cond, np.array([1.0, 0.0, 0.0]))
    # The free-stream direction, None and mach None are accepted.
    require_freestream_wake(cond, freestream_direction(ALPHA, 0.0))
    require_freestream_wake(cond, None)
    require_freestream_wake(vt.FlightCondition(alpha=ALPHA), np.array([1.0, 0.0, 0.0]))


def test_body_wake_without_mach_is_unchanged():
    """The body-axis wake at mach None keeps its result (CL 0.791643 in the defect report)."""
    res = _solve("body", None)
    assert res.totals.CL == pytest.approx(0.791643, abs=5e-7)
    assert np.allclose(res.details["wake_dir"], [1.0, 0.0, 0.0])


def test_freestream_wake_mach_is_continuous_at_zero():
    """The free-stream wake at mach 1e-6 gives the mach None result to about 1e-6 relative."""
    ref = _solve("freestream", None)
    low = _solve("freestream", 1e-6)
    assert low.totals.CL == pytest.approx(ref.totals.CL, rel=1e-6)
    assert low.totals.CDi == pytest.approx(ref.totals.CDi, rel=1e-6)
    assert low.totals.CDi_nearfield == pytest.approx(ref.totals.CDi_nearfield, rel=1e-6)


@pytest.mark.gpu
def test_gpu_sweep_refuses_a_body_wake_with_mach():
    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    old_dev, old_prec = gpu.get_device(), gpu._precision
    try:
        gpu.set_device("gpu")
        gpu.set_precision("float64")
        solver = vt.VortexLatticeSolver()
        alphas = np.radians(np.linspace(-4.0, 12.0, 17))
        with pytest.raises(ValidityError, match="wake_alignment='body'"):
            solver.solve_sweep(_wing(), vt.FlightCondition(V_inf=50.0, mach=0.2), _settings("body"), alphas)
        # The free-stream wake with Mach still runs on the GPU.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = solver.solve_sweep(_wing(), vt.FlightCondition(V_inf=50.0, mach=0.2), _settings("freestream"),
                                     alphas)
        assert len(res) == alphas.size
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)
