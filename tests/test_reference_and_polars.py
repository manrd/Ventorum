# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Regression tests for the reference values of Aircraft (user values, main
surface, mirror copies), the tabulated airfoil tables, the polar loaders and
the trust inputs.
"""

from __future__ import annotations

import json
import subprocess
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.datatypes import (
    SolverSettings,
    TabulatedAirfoil,
    _is_mirror_pair,
    aircraft_from_json,
    aircraft_to_json,
)
from ventorum.solvers.lattice_base import main_surface_strip_count


def _wing(semi_span: float = 3.0) -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="Wing", semi_span=semi_span,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.6)],
    )


def _fin() -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="Fin", semi_span=1.0,
        sections=[vt.WingSection(y_frac=0.0, chord=0.8), vt.WingSection(y_frac=1.0, chord=0.5)],
        is_symmetric=False, dihedral=np.radians(90.0), position=np.array([3.0, 0.0, 0.0]),
    )


def _off_plane_half() -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="Right", semi_span=3.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.6)],
        is_symmetric=False, position=np.array([0.0, 0.4, 0.0]),
    )


# ── Reference values set by the user ─────────────────────────────────────────

def test_user_reference_value_after_auto_is_kept():
    ac = vt.Aircraft(surfaces=[_wing()])
    ac.compute_reference_values()
    assert np.isclose(ac.S_ref, 4.8)
    ac.S_ref = 99.0
    ac.compute_reference_values()
    assert ac.S_ref == 99.0
    # The solver also calls compute_reference_values.
    vt.analyze(ac, vt.FlightCondition(), SolverSettings(n_panels=8))
    assert ac.S_ref == 99.0


def test_auto_reference_values_follow_geometry_and_user_values_stay():
    ac = vt.Aircraft(surfaces=[_wing(3.0)])
    ac.compute_reference_values()
    ac.c_ref = 0.5
    ac.surfaces[0].semi_span = 4.0
    ac.compute_reference_values()
    assert np.isclose(ac.b_ref, 8.0)
    assert np.isclose(ac.S_ref, 6.4)
    assert ac.c_ref == 0.5
    # None goes back to the automatic value.
    ac.c_ref = None
    ac.compute_reference_values()
    assert np.isclose(ac.c_ref, (2.0 * 4.0 / 3.0 * (1.0 + 0.6 + 0.36)) / 6.4)


def test_json_saves_only_user_reference_values():
    ac = vt.Aircraft(surfaces=[_wing()])
    ac.compute_reference_values()
    ac.S_ref = 99.0
    data = json.loads(aircraft_to_json(ac))
    assert data["S_ref"] == 99.0
    assert data["b_ref"] is None and data["c_ref"] is None
    back = aircraft_from_json(aircraft_to_json(ac))
    back.compute_reference_values()
    assert back.S_ref == 99.0 and np.isclose(back.b_ref, 6.0)


def test_clone_keeps_user_and_auto_flags():
    ac = vt.Aircraft(surfaces=[_wing()])
    ac.compute_reference_values()
    ac.b_ref = 7.0
    cl = ac.clone()
    cl.surfaces[0].semi_span = 4.0
    cl.compute_reference_values()
    assert cl.b_ref == 7.0
    assert np.isclose(cl.S_ref, 6.4)


# ── Main surface ─────────────────────────────────────────────────────────────

def test_fin_listed_first_does_not_set_reference_values():
    ac = vt.Aircraft(surfaces=[_fin(), _wing()])
    ac.compute_reference_values()
    ref = vt.Aircraft(surfaces=[_wing()])
    ref.compute_reference_values()
    assert ac.main_surface_index() == 1
    assert np.isclose(ac.S_ref, ref.S_ref)
    assert np.isclose(ac.b_ref, ref.b_ref)
    assert np.isclose(ac.c_ref, ref.c_ref)


def test_only_vertical_surfaces_fall_back_to_first():
    ac = vt.Aircraft(surfaces=[_fin()])
    assert ac.main_surface_index() == 0
    ac.compute_reference_values()
    assert np.isclose(ac.b_ref, 1.0)


def test_mirror_copy_first_reference_values_and_trust_panels():
    right = _off_plane_half()
    ac = vt.Aircraft(surfaces=[right.mirrored(), right])
    ac.compute_reference_values()
    assert len(ac.reference_surfaces()) == 2
    assert np.isclose(ac.S_ref, 4.8)
    assert np.isclose(ac.b_ref, 6.0)

    res = vt.analyze(ac, vt.FlightCondition(), SolverSettings(n_panels=10))
    assert main_surface_strip_count(res.details["lattice"], 0) == 10
    assert not any("Coarse spanwise mesh" in w for w in res.totals.trust.warnings)


def test_root_point_of_mirror_copy_is_the_root():
    right = _off_plane_half()
    ac = vt.Aircraft(surfaces=[right.mirrored(), right])
    qc = ac.root_point("qc")
    te = ac.root_point("te")
    np.testing.assert_allclose(qc, [0.25, -0.4, 0.0], atol=1e-12)
    np.testing.assert_allclose(te, [1.0, -0.4, 0.0], atol=1e-12)
    ac2 = vt.Aircraft(surfaces=[_fin(), _wing()])
    np.testing.assert_allclose(ac2.root_point("qc"), [0.25, 0.0, 0.0], atol=1e-12)


# ── Mirror pair ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("change", ["x_le", "z_le", "airfoil"])
def test_mirror_pair_compares_section_offsets_and_airfoil(change):
    right = _off_plane_half()
    left = right.mirrored()
    assert _is_mirror_pair(right, left)
    sec = left.sections[1]
    if change == "x_le":
        sec.x_le = 0.3
    elif change == "z_le":
        sec.z_le = 0.2
    else:
        sec.airfoil = vt.LinearAirfoil(alpha_L0=np.radians(-2.0))
    assert not _is_mirror_pair(right, left)


# ── Tabulated airfoil ────────────────────────────────────────────────────────

def _sorted_polar() -> TabulatedAirfoil:
    a = np.radians(np.arange(-4.0, 11.0, 2.0))
    return TabulatedAirfoil(name="ok", alpha=a, Cl_data=2 * np.pi * a + 0.2, Cd_data=0.01 + a ** 2,
                            Cm_data=-0.05 + 0.0 * a)


def test_tabulated_unsorted_alpha_is_sorted():
    ref = _sorted_polar()
    order = np.random.default_rng(1).permutation(ref.alpha.size)
    af = TabulatedAirfoil(name="shuffled", alpha=ref.alpha[order], Cl_data=ref.Cl_data[order],
                          Cd_data=ref.Cd_data[order], Cm_data=ref.Cm_data[order])
    # Arrays are stored as given; sorting happens on first use.
    # The interpolation should give the same results as the sorted reference.
    x = np.radians([-3.0, 1.0, 7.5])
    np.testing.assert_allclose(af.Cl(x), ref.Cl(x))
    np.testing.assert_allclose(af.Cd(x), ref.Cd(x))
    assert np.isclose(af.a0, ref.a0) and np.isclose(af.alpha_L0, ref.alpha_L0)

def test_tabulated_repeated_alpha_is_merged_with_warning():
    a = np.radians([0.0, 2.0, 2.0, 4.0])
    af = TabulatedAirfoil(alpha=a, Cl_data=np.array([0.0, 0.2, 0.3, 0.5]),
                          Cd_data=np.array([0.01, 0.01, 0.02, 0.02]))
    # Warning is emitted on first use (when interpolation is built).
    with pytest.warns(RuntimeWarning, match="repeated alpha"):
        cl_val = af.Cl(np.radians(2.0))
    assert np.isclose(cl_val, 0.25)
    assert np.isclose(af.Cd(np.radians(2.0)), 0.015)

def test_tabulated_cache_is_cleared_when_data_are_assigned():
    af = _sorted_polar()
    x = np.radians(3.0)
    cl0 = af.Cl(x)
    a00 = af.a0
    af.Cl_data = 2.0 * af.Cl_data
    assert np.isclose(af.Cl(x), 2.0 * cl0)
    assert np.isclose(af.a0, 2.0 * a00)
    # New alpha and data of a different length, assigned one after the other.
    af.alpha = np.radians([4.0, 0.0, 8.0])
    af.Cl_data = np.array([0.4, 0.0, 0.8])
    af.Cd_data = np.array([0.02, 0.01, 0.03])
    af.Cm_data = None
    # Arrays are stored as given; interpolation sorts them on use.
    assert np.isclose(af.Cl(np.radians(4.0)), 0.4)
    assert np.isclose(af.Cl(np.radians(0.0)), 0.0)
    assert np.isclose(af.Cl(np.radians(8.0)), 0.8)

def test_tabulated_cm_scalar():
    af = _sorted_polar()
    af.Cm_data = None
    out = af.Cm(0.05)
    assert isinstance(out, float) and out == 0.0
    assert af.Cm(np.array([0.0, 0.1])).shape == (2,)
    a = np.radians([0.0, 5.0, 10.0])
    af2 = TabulatedAirfoil(alpha=a, Cl_data=a * 6.0, Cd_data=0.01 + 0 * a, Cm_data=-0.07)
    assert np.isclose(af2.Cm(0.03), -0.07)


def test_tabulated_length_mismatch_raises():
    af = TabulatedAirfoil(alpha=np.radians([0.0, 5.0]), Cl_data=np.array([0.0]), Cd_data=np.array([0.01, 0.02]))
    # Length mismatch is reported when the interpolation is built (first use).
    with pytest.raises(ValueError, match="values, alpha has"):
        af.Cl(0.0)

# ── Polar loaders and XFOIL ──────────────────────────────────────────────────

_XFOIL_POLAR = """\
 XFOIL         Version 6.99

 Calculated polar for: NACA 0012

 Mach =   0.000     Re =     1.000 e 6     Ncrit =   9.000

   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr
  ------ -------- --------- --------- -------- -------- --------
   0.000   0.0000   0.00540   0.00100   0.0000   0.9000   0.9000
   2.000   0.2200   0.00560   0.00110  -0.0010   0.8000   0.9500
   4.000   0.4400   0.00600   0.00130  -0.0020   0.7000   1.0000
  -2.000  -0.2200   0.00560   0.00110   0.0010   0.9500   0.8000
  -4.000  -0.4400   0.00600   0.00130   0.0020   1.0000   0.7000
"""


def test_xfoil_polar_loader_sorts(tmp_path):
    from ventorum.aero.polars import load_xfoil_polar

    p = tmp_path / "polar.txt"
    p.write_text(_XFOIL_POLAR)
    af = load_xfoil_polar(p)
    np.testing.assert_allclose(np.degrees(af.alpha), [-4.0, -2.0, 0.0, 2.0, 4.0])
    np.testing.assert_allclose(af.Cl_data, [-0.44, -0.22, 0.0, 0.22, 0.44])
    assert np.isclose(af.Re, 1.0e6)
    assert np.isclose(af.Cl(np.radians(1.0)), 0.11, atol=1e-3)


def test_xfoil_run_has_timeout(monkeypatch):
    from ventorum.aero import xfoil_runner

    seen = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(cmd="xfoil", timeout=kwargs.get("timeout"))

    monkeypatch.setattr(xfoil_runner.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="did not finish"):
        xfoil_runner.run_xfoil("naca 0012", Re=1e6, timeout=5.0)
    assert seen["timeout"] == 5.0


# ── Docstrings ───────────────────────────────────────────────────────────────

def test_docstrings_state_auto_solver_and_ground_height():
    doc = vt.SolverSettings.__doc__
    assert "always ``\"vlm\"``" in doc
    assert "nonlinear\"`` if any section" not in doc
    fc_doc = vt.FlightCondition.__doc__
    assert "only affects horseshoe" not in fc_doc
    assert "Aircraft.ref_point" in fc_doc
    assert "not reliable on swept wings" in vt.SolverResult.__doc__


def test_no_warning_for_sorted_polar():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _sorted_polar().Cl(0.01)


# ── Test for load_csv_polar (public function kept, with a test) ────────────────────

def test_load_csv_polar(tmp_path):
    """Test that load_csv_polar loads a CSV polar correctly (a public function that is kept, with a test)."""
    from ventorum.aero.polars import load_csv_polar

    csv_content = """alpha,Cl,Cd,Cm
-4.0,-0.44,0.006,0.002
-2.0,-0.22,0.0055,0.001
0.0,0.0,0.005,0.0
2.0,0.22,0.0055,-0.001
4.0,0.44,0.006,-0.002
"""
    p = tmp_path / "polar.csv"
    p.write_text(csv_content)
    af = load_csv_polar(p)
    # Alpha should be sorted and in radians
    np.testing.assert_allclose(np.degrees(af.alpha), [-4.0, -2.0, 0.0, 2.0, 4.0])
    np.testing.assert_allclose(af.Cl_data, [-0.44, -0.22, 0.0, 0.22, 0.44])
    np.testing.assert_allclose(af.Cd_data, [0.006, 0.0055, 0.005, 0.0055, 0.006])
    np.testing.assert_allclose(af.Cm_data, [0.002, 0.001, 0.0, -0.001, -0.002])
