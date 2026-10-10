# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the AVL comparison harness (task T-0055).

Tests 1-7 run without AVL. Tests 8-10 need the real ``avl.exe`` and
skip with "AVL not found: set VENTORUM_AVL_EXE" when it is absent.
"""

import math
from pathlib import Path

import numpy as np
import pytest

import ventorum as vt
from validation.avl.avl_files import build_case
from validation.avl.avl_output import parse_ft, parse_st
from validation.avl.avl_run import AvlCase, find_avl, run_case
from validation.avl.compare import (
    ROOT,
    build_dihedral,
    build_rect,
    main,
    refuse_inside_repository,
)
from ventorum.core.datatypes import Aircraft, LinearAirfoil, SolverSettings, TabulatedAirfoil

DATA = Path(__file__).resolve().parent / "data" / "avl"


def _sections(text: str) -> list[tuple[float, float, float, float, float]]:
    """Parse the SECTION data lines as (x, y, z, chord, ainc)."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if line.strip() == "SECTION":
            out.append(tuple(float(v) for v in lines[i + 1].split()))
    return out


def _clafs(text: str) -> list[float]:
    """Parse the CLAF values in order."""
    lines = text.splitlines()
    return [float(lines[i + 1]) for i, line in enumerate(lines) if line.strip() == "CLAF"]


def _avl_exe():
    """Return the AVL program or skip."""
    exe = find_avl()
    if exe is None:
        pytest.skip("AVL not found: set VENTORUM_AVL_EXE")
    return exe


# --------------------------------------------------------------------------
# Without AVL
# --------------------------------------------------------------------------

def test_writer_symmetric_wing_file():
    """Rectangular symmetric wing: references, duplication, sections, CLAF, incidence."""
    ac = build_rect()
    settings = SolverSettings(solver_type="vlm", n_panels=20, n_chord=4)
    case = build_case(ac, settings, alpha_deg=4.0)
    text = case.avl_text
    assert "YDUPLICATE" in text
    assert "0.0" in text.split("YDUPLICATE")[1].split("SECTION")[0]
    header = text.splitlines()
    assert [float(v) for v in header[3].split()] == [8.0, 1.0, 8.0]
    secs = _sections(text)
    assert len(secs) == 2
    assert secs[0][:3] == (0.0, 0.0, 0.0)
    assert secs[1][:3] == (0.0, 4.0, 0.0)
    assert secs[0][3] == 1.0 and secs[1][3] == 1.0
    assert secs[0][4] == 0.0 and secs[1][4] == 0.0
    assert _clafs(text) == [1.0, 1.0]
    assert case.run_alpha_deg == 4.0
    assert case.run_beta_deg == 0.0
    assert case.notes == ()


def test_writer_alpha_L0_goes_into_incidence():
    """alpha_L0 = -2 deg and twist 1 deg give an incidence of 3 deg."""
    airfoil = LinearAirfoil(a0=2.0 * np.pi, alpha_L0=np.radians(-2.0))
    surf = vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[
            vt.WingSection(0.0, 1.0, twist=np.radians(1.0), airfoil=airfoil),
            vt.WingSection(1.0, 1.0, twist=np.radians(1.0), airfoil=airfoil),
        ],
    )
    ac = Aircraft(name="twisted", surfaces=[surf])
    case = build_case(ac, SolverSettings(n_panels=10, n_chord=4), alpha_deg=0.0)
    secs = _sections(case.avl_text)
    assert len(secs) == 2
    assert abs(secs[0][4] - 3.0) < 1e-9
    assert abs(secs[1][4] - 3.0) < 1e-9


def test_writer_refuses_tabulated_and_cm0():
    """Tabulated airfoils and nonzero Cm0 are refused with names."""
    tabulated = TabulatedAirfoil(
        name="polar",
        alpha=np.radians(np.array([-2.0, 0.0, 2.0])),
        Cl_data=np.array([-0.2, 0.0, 0.2]),
        Cd_data=np.array([0.01, 0.01, 0.01]),
    )
    surf = vt.LiftingSurface(
        name="WingTab",
        semi_span=4.0,
        sections=[
            vt.WingSection(0.0, 1.0, airfoil=tabulated),
            vt.WingSection(1.0, 1.0, airfoil=tabulated),
        ],
    )
    with pytest.raises(ValueError, match="WingTab"):
        build_case(Aircraft(name="a", surfaces=[surf]), SolverSettings())
    cambered = LinearAirfoil(Cm0=0.05)
    surf2 = vt.LiftingSurface(
        name="WingCm",
        semi_span=4.0,
        sections=[
            vt.WingSection(0.0, 1.0, airfoil=cambered),
            vt.WingSection(1.0, 1.0, airfoil=cambered),
        ],
    )
    with pytest.raises(ValueError, match="WingCm"):
        build_case(Aircraft(name="a", surfaces=[surf2]), SolverSettings())
    with pytest.raises(ValueError, match="Cm0"):
        build_case(Aircraft(name="a", surfaces=[surf2]), SolverSettings())
    case = build_case(Aircraft(name="a", surfaces=[surf2]), SolverSettings(),
                      allow_cm0=True)
    assert any("Cm0" in note for note in case.notes)


def test_writer_ground_effect_rotation():
    """Rotated leading edges equal rotation about ref; Zsym = Zref - h."""
    ac = build_rect()
    ac.ref_point = np.array([2.0, 0.0, 0.5])
    alpha_deg, h = 5.0, 1.0
    case = build_case(ac, SolverSettings(n_panels=10, n_chord=4),
                      alpha_deg=alpha_deg, ground_h=h)
    assert case.ground
    assert case.run_alpha_deg == 0.0
    assert case.zsym == pytest.approx(0.5 - h)
    header = case.avl_text.splitlines()
    assert header[2].split()[:2] == ["0", "1"]
    assert float(header[2].split()[2]) == pytest.approx(0.5 - h)
    secs = _sections(case.avl_text)
    angle = np.radians(alpha_deg)
    ca, sa = math.cos(angle), math.sin(angle)
    ref = np.array([2.0, 0.0, 0.5])
    for (x, y, z, _chord, ainc), (lx, ly, lz) in zip(
        secs, [(0.0, 0.0, 0.0), (0.0, 4.0, 0.0)]
    ):
        dx, dz = lx - ref[0], lz - ref[2]
        assert abs(x - (ref[0] + dx * ca + dz * sa)) < 1e-12
        assert abs(y - ly) < 1e-12
        assert abs(z - (ref[2] - dx * sa + dz * ca)) < 1e-12
        assert abs(ainc - alpha_deg) < 1e-9


def test_parsers_on_stored_output():
    """Parse the stored AVL files; check numbers read from the text."""
    ft = parse_ft((DATA / "rect_ft.txt").read_text(encoding="utf-8"))
    st = parse_st((DATA / "rect_st.txt").read_text(encoding="utf-8"))
    assert ft["Alpha"] == pytest.approx(4.0)
    assert ft["CLtot"] == pytest.approx(0.31949)
    assert ft["CDff"] == pytest.approx(0.0041856)
    assert ft["e"] == pytest.approx(0.9721)
    assert ft["Cmtot"] == pytest.approx(-0.07738)
    assert ft["Sref"] == pytest.approx(8.0)
    assert st["CLa"] == pytest.approx(4.560632)
    assert st["Cma"] == pytest.approx(-1.101173)
    assert st["CLq"] == pytest.approx(6.942764)
    assert st["Cmq"] == pytest.approx(-2.391322)
    assert st["Clp"] == pytest.approx(-0.513857)
    assert st["Xnp"] == pytest.approx(0.241452)


def test_parser_refuses_truncated_output(tmp_path):
    """A cut file gives an error, never zeros."""
    full = (DATA / "rect_st.txt").read_text(encoding="utf-8")
    cut = tmp_path / "cut.txt"
    cut.write_text(full[: len(full) // 2], encoding="utf-8")
    with pytest.raises(ValueError):
        parse_st(cut.read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        parse_ft("")
    with pytest.raises(ValueError):
        parse_ft(full[:200])


def test_out_dir_inside_repository_is_refused(tmp_path):
    """An output folder inside the repository is refused before writing."""
    with pytest.raises(ValueError, match="inside the repository"):
        refuse_inside_repository(ROOT / "tmp_avl_refuse_check")
    with pytest.raises(ValueError, match="inside the repository"):
        refuse_inside_repository(Path("validation") / "tmp_avl_refuse_check")
    dummy = tmp_path / "avl.exe"
    dummy.write_text("not a program", encoding="utf-8")
    target = ROOT / "tmp_avl_refuse_check"
    assert main(["--avl", str(dummy), "--out", str(target)]) == 2
    assert not target.exists()


# --------------------------------------------------------------------------
# With AVL
# --------------------------------------------------------------------------

def test_avl_runs_rectangular_wing(tmp_path):
    """One run: CL at alpha 4 deg within 2 % of Ventorum VLM, body wake."""
    exe = _avl_exe()
    ac = build_rect()
    mesh = 20
    settings = SolverSettings(solver_type="vlm", n_panels=mesh, n_chord=4)
    case = build_case(ac, settings, alpha_deg=4.0)
    result = run_case(exe, case, tmp_path / "run")
    assert result.ok, f"AVL run failed: {result.error}"
    cl_avl = result.forces["CLtot"]
    totals = vt.analyze(
        ac,
        condition=vt.FlightCondition(alpha=np.radians(4.0)),
        settings=SolverSettings(solver_type="vlm", n_panels=mesh, n_chord=4,
                                wake_alignment="body"),
    ).totals
    assert abs(cl_avl - totals.CL) / abs(totals.CL) < 0.02


def test_sideslip_sign_agrees(tmp_path):
    """Wing with dihedral at beta = +2 deg: same Cl sign in AVL and Ventorum."""
    exe = _avl_exe()
    ac = build_dihedral()
    case = build_case(ac, SolverSettings(n_panels=20, n_chord=4),
                      alpha_deg=4.0, beta_deg=2.0)
    result = run_case(exe, case, tmp_path / "run")
    assert result.ok, f"AVL run failed: {result.error}"
    cl_avl = result.forces["Cltot"]
    totals = vt.analyze(
        ac,
        condition=vt.FlightCondition(alpha=np.radians(4.0), beta=np.radians(2.0)),
        settings=SolverSettings(solver_type="vlm", n_panels=20, n_chord=4,
                                wake_alignment="body"),
    ).totals
    assert cl_avl != 0.0 and totals.Cl != 0.0
    assert math.copysign(1.0, cl_avl) == math.copysign(1.0, totals.Cl)


def test_failed_run_is_reported(tmp_path):
    """A wrong path or a broken input gives a failed row, not zeros."""
    ac = build_rect()
    case = build_case(ac, SolverSettings(n_panels=20, n_chord=4), alpha_deg=4.0)
    bad = run_case(tmp_path / "no_such_avl.exe", case, tmp_path / "bad_path")
    assert not bad.ok
    assert bad.forces is None and bad.stabderivs is None
    assert bad.error is not None
    exe = _avl_exe()
    broken = run_case(exe, AvlCase(avl_text="this is not a valid geometry file\n"),
                      tmp_path / "broken", timeout=30.0)
    assert not broken.ok
    assert broken.forces is None and broken.stabderivs is None
    assert broken.error is not None
