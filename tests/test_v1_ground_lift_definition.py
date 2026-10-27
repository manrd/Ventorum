# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Lift definitions of the V1 alpha-correction coupling in ground effect.

The strip lift of the lattice is a circulation lift ``2 Gamma / (V c)``. In
ground effect a 2D section has a pressure lift and a circulation lift, and
they differ near a wall (Coulliette and Plotkin 1996). The coupling must
compare circulation with circulation: a ground polar that holds the
circulation lift is read with it. These tests check that rule, the data
path of the circulation lift (polar, table, cache, source) and that free
air does not change.
"""

from __future__ import annotations

import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from ventorum.aero.polar_cache import PolarCache
from ventorum.aero.polar_source_v2d import Ventorum2DPolarSource
from ventorum.aero.polar_table import PolarTable, build_polar_table
from ventorum.aero.polars import blend_tabulated
from ventorum.core.datatypes import (
    ControlSurface,
    FlightCondition,
    LiftingSurface,
    SolverSettings,
    TabulatedAirfoil,
    WingSection,
)
from ventorum.panel2d.solver import solve_inviscid
from ventorum.solvers.factory import make_solver
from ventorum.viscous.alpha_correction import _GROUND_LIFT_NOTE, solve_viscous

#: Relative and absolute tolerances of the bit-identity checks [-].
REL_SAME = 1e-12
ABS_SAME = 1e-14


def _wing(controls=None) -> LiftingSurface:
    """Return a rectangular wing of aspect ratio 8 (chord 1 m, semi-span 4 m)."""
    return LiftingSurface(
        name="w",
        semi_span=4.0,
        sections=[WingSection(y_frac=0.0, chord=1.0), WingSection(y_frac=1.0, chord=1.0)],
        sweep_le=0.0,
        dihedral=0.0,
        controls=controls or [],
    )


def _polar() -> TabulatedAirfoil:
    """Return a synthetic polar with a soft stall above 8 deg."""
    a = np.radians(np.linspace(-10.0, 25.0, 71))
    cl = 5.9 * (a + 0.02) - 40.0 * np.maximum(a - np.radians(8.0), 0.0) ** 2
    return TabulatedAirfoil(
        name="p", alpha=a, Cl_data=cl, Cd_data=0.008 + 0.01 * a**2, Cm_data=-0.05 + 0.0 * a, Re=1e6
    )


def _table(heights=None, with_circulation=False) -> PolarTable:
    """Return a synthetic table over (Re, flap 0, [h/c,] alpha); Cl_circulation equals Cl on request."""
    al = np.linspace(-10.0, 20.0, 31)
    a = np.radians(al)
    base = 5.9 * (a + 0.02) - 40.0 * np.maximum(a - np.radians(8.0), 0.0) ** 2
    if heights is None:
        cl = np.stack([base, 1.02 * base])[:, None, :]
    else:
        f = 1.0 + 0.1 / np.asarray(heights)
        cl = np.stack([np.stack([base * fi for fi in f]), np.stack([1.02 * base * fi for fi in f])])[:, None]
    return PolarTable(
        Re=np.array([1e5, 1e8]),
        flap_deg=np.array([0.0]),
        alpha_deg=al,
        Cl=cl,
        Cd=np.full(cl.shape, 0.01),
        Cm=np.full(cl.shape, -0.04),
        h_over_c=None if heights is None else np.asarray(heights),
        source_info={"section": "synthetic"},
        Cl_circulation=cl.copy() if with_circulation else None,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Key regression: an ideal inviscid ground polar gives the inviscid lattice
# ═══════════════════════════════════════════════════════════════════════════════

def _thin_joukowski(n_panels: int = 200, xc: float = -0.003) -> np.ndarray:
    """Return a thin symmetric Joukowski section (t/c about 0.4 %), unit chord, Selig order."""
    b = 1.0
    zc = complex(xc, 0.0)
    radius = abs(b - zc)
    th = np.angle(b - zc) + np.linspace(0.0, 2.0 * np.pi, n_panels + 1)
    zeta = zc + radius * np.exp(1j * th)
    z = zeta + b * b / zeta
    z[0] = z[-1] = 2.0 * b
    xy = np.column_stack([z.real, z.imag])
    xmin = xy[:, 0].min()
    return (xy - [xmin, 0.0]) / (2.0 * b - xmin)


#: Table heights [-] around the two strip heights of the test (0.25 and 0.5).
_IDEAL_HEIGHTS = np.array([0.2, 0.25, 0.3, 0.45, 0.5, 0.55])


@pytest.fixture(scope="module")
def ideal_ground_tables():
    """Inviscid Ventorum-2D tables of a thin section: pressure Cl, with and without Cl_circulation."""
    section = _thin_joukowski()
    al = np.arange(-2.0, 10.0 + 1e-9, 1.0)
    cl_p = np.zeros((2, 1, _IDEAL_HEIGHTS.size, al.size))
    cl_c = np.zeros_like(cl_p)
    for k, h in enumerate(_IDEAL_HEIGHTS):
        for j, a in enumerate(al):
            res = solve_inviscid(section, float(a), height=float(h))
            cl_p[:, 0, k, j] = res.Cl
            cl_c[:, 0, k, j] = res.Cl_circulation
    zero = np.zeros_like(cl_p)
    common = dict(Re=np.array([1e4, 1e9]), flap_deg=np.array([0.0]), alpha_deg=al, Cd=zero, Cm=zero,
                  h_over_c=_IDEAL_HEIGHTS, source_info={"section": "thin Joukowski, inviscid"})
    return (PolarTable(Cl=cl_p, Cl_circulation=cl_c, **common), PolarTable(Cl=cl_p, **common))


#: Strip h/c [-] and the reference height [m] that gives it at alpha 4 deg.
_IDEAL_CASES = [(0.25, 0.267), (0.5, 0.517)]

#: Largest relative CL difference [-] of the coupled result from the inviscid
#: lattice with the ideal circulation polar. The rest is the thickness lift of
#: the 0.4 % section and the 2D pitch of the polar about c/4. Before the
#: change (pressure lift in the residual) the difference was several percent.
_IDEAL_TOL = 5e-3


@pytest.mark.parametrize("h_over_c, h", _IDEAL_CASES)
def test_ideal_ground_polar_gives_the_inviscid_lattice(ideal_ground_tables, h_over_c, h):
    """Regression: the residual compared the lattice circulation lift with the 2D pressure lift."""
    with_circ, _ = ideal_ground_tables
    st = SolverSettings(n_panels=12)
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(4.0), h=h)
    inviscid = make_solver("vlm").solve(_wing(), cond, st).totals.CL
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no lift-definition note with circulation data
        res = solve_viscous(_wing(), cond, st, with_circ, model="vlm")
    assert res.converged
    assert np.nanmin(res.details["strip_h_over_c"]) == pytest.approx(h_over_c, abs=2e-3)
    assert np.all(res.details["strip_circulation_lift"])
    assert _GROUND_LIFT_NOTE not in res.details["notes"]
    assert res.totals.CL == pytest.approx(inviscid, rel=_IDEAL_TOL)


@pytest.mark.parametrize("h_over_c, h", _IDEAL_CASES)
def test_pressure_only_ground_polar_keeps_the_old_rule_and_gets_a_note(ideal_ground_tables, h_over_c, h):
    """A ground table without Cl_circulation is read with its Cl, as before, and the result has a note."""
    _, pressure_only = ideal_ground_tables
    st = SolverSettings(n_panels=12)
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(4.0), h=h)
    inviscid = make_solver("vlm").solve(_wing(), cond, st).totals.CL
    with pytest.warns(RuntimeWarning, match="no circulation lift"):
        res = solve_viscous(_wing(), cond, st, pressure_only, model="vlm")
    assert not np.any(res.details["strip_circulation_lift"])
    assert _GROUND_LIFT_NOTE in res.details["notes"]
    assert _GROUND_LIFT_NOTE in res.totals.trust.warnings
    # The pressure lift is smaller than the circulation lift near the ground.
    assert res.totals.CL < (1.0 - 0.02) * inviscid


# ═══════════════════════════════════════════════════════════════════════════════
# Free air and polars without circulation data: same bits as before
# ═══════════════════════════════════════════════════════════════════════════════

#: Values from the code before the change (same machine-independent inputs).
_BEFORE = {
    "vlm_free_tab": dict(CL=0.5435441783124823, CDi=0.012147506851064837, CD_total=0.020203446828669952,
                         Cm=-0.18173593236450547, delta_sum=-0.3664367899874704),
    "llt_free_tab": dict(CL=0.5716715834197666, CDi=0.013986473816739278, CD_total=0.022048756873390855,
                         Cm=-0.19271024015380467, delta_sum=-0.35784389083069434),
    "vlm_free_table": dict(CL=0.47092298023484497, CDi=0.009110489619031542, CD_total=0.019110489619031543,
                           Cm=-0.15425799436047063, delta_sum=-0.3956453482234822),
    "vlm_free_flap": dict(CL=0.5068129961262599, CDi=0.012096230002177653, CD_total=0.020155506731226944,
                          Cm=-0.2685390020257866, delta_sum=-0.37021462987759),
    "vlm_ground_freepolar": dict(CL=0.48437998186320635, CDi=0.0040376687036339735,
                                 CD_total=0.012084837411244677, Cm=-0.17726797265903152,
                                 delta_sum=-0.3759016353990824),
    "vlm_ground_table_nocirc": dict(CL=0.5030628232398021, CDi=0.004337852631268478,
                                    CD_total=0.014337852631268477, Cm=-0.1722060563707576,
                                    delta_sum=-0.44038872235847437),
}


def _case(name: str):
    free = FlightCondition(V_inf=30.0, alpha=np.radians(6.0))
    ground = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=0.5)
    flap = [ControlSurface("flap", 0.3, 0.7, 0.75, np.radians(10.0))]
    return {
        "vlm_free_tab": (_wing(), free, _polar(), "vlm"),
        "llt_free_tab": (_wing(), free, _polar(), "llt"),
        "vlm_free_table": (_wing(), FlightCondition(V_inf=30.0, alpha=np.radians(5.0)), _table(), "vlm"),
        "vlm_free_flap": (_wing(flap), FlightCondition(V_inf=30.0, alpha=np.radians(3.0)), _polar(), "vlm"),
        "vlm_ground_freepolar": (_wing(), ground, _polar(), "vlm"),
        "vlm_ground_table_nocirc": (_wing(), ground, _table([0.2, 0.4, 0.8, 2.0]), "vlm"),
    }[name]


def _summary(res) -> dict:
    t = res.totals
    return dict(CL=t.CL, CDi=t.CDi, CD_total=t.CD_total, Cm=t.Cm,
                delta_sum=float(np.sum(res.details["delta"])))


@pytest.mark.parametrize("name", sorted(_BEFORE))
def test_results_without_circulation_data_do_not_change(name):
    """Free air, a free-air polar in ground effect and a ground table without Cl_circulation keep their bits."""
    wing, cond, polars, model = _case(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = solve_viscous(wing, cond, SolverSettings(n_panels=12, n_chord=4), polars, model=model)
    assert res.converged
    got = _summary(res)
    for key, value in _BEFORE[name].items():
        assert got[key] == pytest.approx(value, rel=REL_SAME, abs=ABS_SAME), key
    assert not np.any(res.details["strip_circulation_lift"])
    assert (_GROUND_LIFT_NOTE in res.details["notes"]) == (name == "vlm_ground_table_nocirc")


def test_circulation_equal_to_cl_gives_the_same_result_without_the_note():
    """Only the lift array of the residual changes: Cl_circulation equal to Cl gives the old bits."""
    wing, cond, _, _ = _case("vlm_ground_table_nocirc")
    table = _table([0.2, 0.4, 0.8, 2.0], with_circulation=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = solve_viscous(wing, cond, SolverSettings(n_panels=12, n_chord=4), table, model="vlm")
    assert np.all(res.details["strip_circulation_lift"])
    assert _GROUND_LIFT_NOTE not in res.details["notes"]
    got = _summary(res)
    for key, value in _BEFORE["vlm_ground_table_nocirc"].items():
        assert got[key] == pytest.approx(value, rel=REL_SAME, abs=ABS_SAME), key


def test_free_air_table_with_circulation_is_read_with_cl():
    """In free air the residual reads Cl, also when the polar has circulation data."""
    wing, cond, _, _ = _case("vlm_free_table")
    table = _table(with_circulation=True)
    table.Cl_circulation[...] = 0.5 * table.Cl  # never read in free air
    res = solve_viscous(wing, cond, SolverSettings(n_panels=12, n_chord=4), table, model="vlm")
    assert not np.any(res.details["strip_circulation_lift"])
    assert res.totals.CL == pytest.approx(_BEFORE["vlm_free_table"]["CL"], rel=REL_SAME, abs=ABS_SAME)


# ═══════════════════════════════════════════════════════════════════════════════
# Data path of the circulation lift
# ═══════════════════════════════════════════════════════════════════════════════

def test_tabulated_airfoil_interpolates_the_circulation_lift():
    a = np.radians(np.array([4.0, -2.0, 0.0, 2.0, 6.0]))  # unsorted on purpose
    af = TabulatedAirfoil(name="t", alpha=a, Cl_data=6.0 * a, Cd_data=0.01 + 0 * a,
                          Cl_circulation_data=6.5 * a + 0.01)
    assert af.has_circulation_lift
    q = np.radians(1.3)
    assert af.Cl_circulation(q) == pytest.approx(6.5 * q + 0.01, rel=1e-12)
    assert af.Cl_circulation_slope(q) == pytest.approx(6.5, rel=1e-12)
    assert af.Cl(q) == pytest.approx(6.0 * q, rel=1e-12)
    # End values outside the table, zero slope there.
    assert af.Cl_circulation(np.radians(20.0)) == pytest.approx(6.5 * np.radians(6.0) + 0.01)
    assert af.Cl_circulation_slope(np.radians(20.0)) == 0.0
    # A new array clears the cached interpolation.
    af.Cl_circulation_data = 7.0 * a
    assert af.Cl_circulation(q) == pytest.approx(7.0 * q, rel=1e-12)
    clone = af.clone()
    np.testing.assert_array_equal(clone.Cl_circulation_data, af.Cl_circulation_data)
    assert clone.Cl_circulation(q) == af.Cl_circulation(q)
    plain = TabulatedAirfoil(name="n", alpha=a, Cl_data=6.0 * a, Cd_data=0.01 + 0 * a)
    assert not plain.has_circulation_lift and plain.clone().Cl_circulation_data is None
    with pytest.raises(ValueError, match="no circulation lift"):
        plain.Cl_circulation(q)


def test_blend_keeps_the_circulation_lift_only_when_both_polars_have_it():
    a = np.radians(np.linspace(-4.0, 8.0, 13))
    af1 = TabulatedAirfoil(name="1", alpha=a, Cl_data=6 * a, Cd_data=0.01 + 0 * a, Cl_circulation_data=7 * a)
    af2 = TabulatedAirfoil(name="2", alpha=a, Cl_data=5 * a, Cd_data=0.01 + 0 * a, Cl_circulation_data=5 * a)
    mix = blend_tabulated(af1, af2, 0.25)
    np.testing.assert_allclose(mix.Cl_circulation_data, 0.75 * 7 * a + 0.25 * 5 * a, rtol=1e-12, atol=1e-15)
    af2.Cl_circulation_data = None
    assert blend_tabulated(af1, af2, 0.25).Cl_circulation_data is None


def test_table_interpolates_cl_circulation_like_cl(tmp_path):
    """Cl_circulation follows the interpolation of Cl (log10 Re, flap, 1/h) and its NaN gaps."""
    hs = np.array([0.2, 0.5, 1.0])
    al = np.array([-2.0, 0.0, 2.0, 4.0])
    rng = np.random.default_rng(3)
    cl = rng.normal(size=(2, 2, hs.size, al.size))
    circ = rng.normal(size=cl.shape)
    circ[0, 0, 0, 3] = np.nan
    zero = np.zeros_like(cl)
    kw = dict(Re=[1e5, 1e6], flap_deg=[0.0, 10.0], alpha_deg=al, Cd=zero, Cm=zero, h_over_c=hs,
              source_info={"options": {"x_hinge": 0.75}})
    table = PolarTable(Cl=cl, Cl_circulation=circ, **kw)
    same = PolarTable(Cl=circ, **kw)  # the circulation array read as Cl
    polar = table.at(Re=3e5, flap_deg=4.0, h_over_c=0.3)
    ref = same.at(Re=3e5, flap_deg=4.0, h_over_c=0.3)
    np.testing.assert_array_equal(polar.alpha, ref.alpha)
    np.testing.assert_array_equal(polar.Cl_circulation_data, ref.Cl_data)
    assert polar.alpha.size == 3 and table.last_dropped == 1  # the NaN of Cl_circulation is a gap
    # Free-air table: bilinear interpolation.
    free = PolarTable(Cl=cl[:, :, 0], Cl_circulation=circ[:, :, 1], **{**kw, "Cd": zero[:, :, 0],
                      "Cm": zero[:, :, 0], "h_over_c": None})
    free_ref = PolarTable(Cl=circ[:, :, 1], **{**kw, "Cd": zero[:, :, 0], "Cm": zero[:, :, 0], "h_over_c": None})
    np.testing.assert_array_equal(free.at(Re=2e5, flap_deg=7.0).Cl_circulation_data,
                                  free_ref.at(Re=2e5, flap_deg=7.0).Cl_data)
    # A table without the array gives polars without circulation data.
    assert PolarTable(Cl=cl, **kw).at(Re=3e5, flap_deg=4.0, h_over_c=0.3).Cl_circulation_data is None
    with pytest.raises(ValueError, match="Cl_circulation has shape"):
        PolarTable(Cl=cl, Cl_circulation=circ[..., :2], **kw)
    # Storage: the array is stored when present; an old file without it still loads.
    back = PolarTable.from_npz(table.to_npz(tmp_path / "with.npz"))
    np.testing.assert_array_equal(back.Cl_circulation, table.Cl_circulation)
    old = PolarTable.from_npz(PolarTable(Cl=cl, **kw).to_npz(tmp_path / "old.npz"))
    assert old.Cl_circulation is None
    with np.load(tmp_path / "old.npz") as stored:
        assert "Cl_circulation" not in stored.files


def _fake_result(alpha_deg, height):
    alphas = np.atleast_1d(np.asarray(alpha_deg, dtype=float))
    cl = 2.0 * np.pi * np.radians(alphas)
    shift = 0.0 if height is None else 0.05 / float(height)
    return SimpleNamespace(alpha=alphas, Cl=cl - shift, Cl_circulation=cl + shift, Cd=np.full(alphas.shape, 0.01),
                           Cm=np.zeros(alphas.shape), converged=np.ones(alphas.shape, dtype=bool))


@pytest.fixture
def fake_2d(monkeypatch):
    import ventorum.aero.polar_source_v2d as module

    calls: list[dict] = []

    def fake_solve(bodies, alpha_deg, Re, **kwargs):
        calls.append(kwargs)
        return _fake_result(alpha_deg, kwargs.get("height"))

    monkeypatch.setattr(module, "solve_viscous", fake_solve)
    return calls


def test_ventorum2d_source_stores_the_circulation_lift_in_ground_effect(tmp_path, fake_2d):
    source = Ventorum2DPolarSource(cache=PolarCache(tmp_path))
    free = source.polar("naca 0012", 1e6, [0.0, 4.0])
    assert free.Cl_circulation_data is None
    ground = source.polar("naca 0012", 1e6, [0.0, 4.0], height=0.25)
    ref = _fake_result([0.0, 4.0], 0.25)
    np.testing.assert_array_equal(ground.Cl_data, ref.Cl)
    np.testing.assert_array_equal(ground.Cl_circulation_data, ref.Cl_circulation)
    assert len(fake_2d) == 2
    # The cache gives the circulation lift back.
    again = source.polar("naca 0012", 1e6, [0.0, 4.0], height=0.25)
    assert len(fake_2d) == 2
    np.testing.assert_array_equal(again.Cl_circulation_data, ref.Cl_circulation)
    # A table of ground polars carries the circulation lift.
    table = build_polar_table(source, "naca 0012", [1e6], [0.0, 4.0], h_over_c=[0.25, 0.5], workers=1)
    np.testing.assert_array_equal(table.Cl_circulation[0, 0, 0], ref.Cl_circulation)
    assert build_polar_table(source, "naca 0012", [1e6], [0.0, 4.0], workers=1).Cl_circulation is None


def test_old_ground_cache_entry_without_circulation_is_solved_again(tmp_path, fake_2d):
    """Regression guard: a ground entry of the older format has no circulation lift; it is a miss."""
    cache = PolarCache(tmp_path)
    source = Ventorum2DPolarSource(cache=cache)
    source.polar("naca 0012", 1e6, [0.0, 4.0], height=0.4)
    source.polar("naca 0012", 1e6, [0.0, 4.0])
    assert len(fake_2d) == 2
    for path in tmp_path.glob("*.json"):  # strip the new key: the older file format
        data = json.loads(path.read_text())
        data.pop("Cl_circulation", None)
        path.write_text(json.dumps(data))
    ground = source.polar("naca 0012", 1e6, [0.0, 4.0], height=0.4)
    assert len(fake_2d) == 3 and ground.Cl_circulation_data is not None
    source.polar("naca 0012", 1e6, [0.0, 4.0])  # a free-air entry stays a hit
    assert len(fake_2d) == 3
    source.polar("naca 0012", 1e6, [0.0, 4.0], height=0.4)  # the new entry is a hit
    assert len(fake_2d) == 3
