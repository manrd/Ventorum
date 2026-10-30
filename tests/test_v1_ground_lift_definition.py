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
    Aircraft,
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


# ═══════════════════════════════════════════════════════════════════════════════
# Non-planar strips in ground effect: dihedral and a vertical fin
# ═══════════════════════════════════════════════════════════════════════════════

#: Table heights [-] of the analytic ground table. A strip at 59.5 deg to the
#: ground reads its polar at the section-plane height, about twice its
#: vertical height: the table goes to 5.
_ANALYTIC_HEIGHTS = np.array([0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0])


def _analytic_circulation_lift(alpha, h_over_c):
    """Return the circulation lift [-] of the analytic table at *alpha* [rad] and *h_over_c* [-]."""
    return (6.0 + 0.3 / h_over_c) * (alpha + 0.03)


def _analytic_pressure_lift(alpha, h_over_c):
    """Return the pressure lift [-] of the analytic table (smaller than the circulation lift)."""
    return (5.6 - 0.2 / h_over_c) * (alpha + 0.03)


def _analytic_ground_table() -> PolarTable:
    """Return a ground table that the table interpolation reproduces exactly.

    Both lift arrays have the form ``(a + b / h) (alpha + c)``. The table
    interpolates linearly in 1/h and the polar is a PCHIP in alpha, so
    these arrays come back without interpolation error. They do not
    depend on Re. The circulation lift differs from the pressure lift, so
    the test can tell which one the residual reads.
    """
    al = np.linspace(-10.0, 20.0, 31)
    a = np.radians(al)[None, :]
    h = _ANALYTIC_HEIGHTS[:, None]
    one = np.ones((2, 1, 1, 1))
    cl_c = one * _analytic_circulation_lift(a, h)[None, None]
    cl_p = one * _analytic_pressure_lift(a, h)[None, None]
    return PolarTable(
        Re=np.array([1e4, 1e9]), flap_deg=np.array([0.0]), alpha_deg=al, Cl=cl_p, Cl_circulation=cl_c,
        Cd=np.full(cl_p.shape, 0.01), Cm=np.full(cl_p.shape, -0.04), h_over_c=_ANALYTIC_HEIGHTS,
        source_info={"section": "analytic"},
    )


def _strip_circulation_lift(res, strips, V_inf):
    """Return the lattice strip lift ``2 Gamma / (V c)`` [-] from the solved panel circulations."""
    lat = res.details["lattice"]
    gamma = np.asarray(res.details["gamma"])
    return np.array([2.0 * np.sum(gamma[lat.panel_strip == s]) / (V_inf * lat.chord[s]) for s in strips])


def _assert_no_lift_or_convergence_warning(caught) -> None:
    """Fail on the lift-definition note or a convergence warning; other trust notes are allowed."""
    texts = [str(w.message) for w in caught]
    assert _GROUND_LIFT_NOTE not in texts
    assert not any("converge" in t.lower() for t in texts), texts


def _section_plane_cosine(res, strips) -> np.ndarray:
    """Return cos(Gamma_g) [-] of each strip: the in-plane part of the ground normal."""
    lat = res.details["lattice"]
    axes = np.cross(lat.normal[strips], lat.chord_dir[strips])
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    n = res.details["ground"].normal / np.linalg.norm(res.details["ground"].normal)
    return np.sqrt(1.0 - (axes @ n) ** 2)


def _dihedral_wing(dihedral_deg: float) -> LiftingSurface:
    """Return a symmetric rectangular wing of aspect ratio 8 with dihedral."""
    return LiftingSurface(
        name="w",
        semi_span=4.0,
        sections=[WingSection(y_frac=0.0, chord=1.0), WingSection(y_frac=1.0, chord=1.0)],
        dihedral=np.radians(dihedral_deg),
    )


@pytest.mark.parametrize("model, h", [("vlm", 0.4), ("llt", 1.5)])
def test_dihedral_wing_in_ground_effect_reads_the_circulation_lift(model, h):
    """A wing with dihedral 6 deg in ground effect: each strip residual reads Cl_circulation.

    The strip lift of the lattice comes from the solved circulations, and
    the circulation lift of the polar from the analytic function of the
    table. At convergence they agree on each strip at its own h/c. The
    pressure lift of the table does not agree.
    """
    V = 30.0
    cond = FlightCondition(V_inf=V, alpha=np.radians(4.0), h=h)
    st = SolverSettings(n_panels=16, n_chord=4 if model == "vlm" else 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        res = solve_viscous(_dihedral_wing(6.0), cond, st, _analytic_ground_table(), model=model, tol=1e-11)
    _assert_no_lift_or_convergence_warning(caught)
    assert res.converged
    assert _GROUND_LIFT_NOTE not in res.details["notes"]
    lat = res.details["lattice"]
    strips = np.flatnonzero(lat.strip_is_right)
    assert np.all(res.details["strip_circulation_lift"][strips])
    # Dihedral: the strip height grows to the tip, over more than one table
    # interval. The body pitches by alpha about the y axis, so a rise dz in
    # the body axes is a rise dz cos(alpha) above the ground (chord 1 m).
    hc = res.details["strip_h_over_c"][strips]
    y = lat.qc_mid[strips, 1]
    rise = (y.max() - y.min()) * np.tan(np.radians(6.0)) * np.cos(cond.alpha)
    assert hc[np.argmax(y)] - hc[np.argmin(y)] == pytest.approx(rise, rel=1e-6)
    alpha_v = res.details["alpha_eff"][strips]
    # The polar is read at the section-plane height h / (c cos(Gamma_g)).
    h_read = hc / _section_plane_cosine(res, strips)
    cl_lattice = _strip_circulation_lift(res, strips, V)
    np.testing.assert_allclose(cl_lattice, _analytic_circulation_lift(alpha_v, h_read), rtol=0, atol=1e-8)
    # The pressure lift is far from the lattice lift: the test can see a wrong definition.
    assert np.min(np.abs(cl_lattice - _analytic_pressure_lift(alpha_v, h_read))) > 1e-3


def _wing_and_fin() -> Aircraft:
    """Return a symmetric wing with a vertical fin (dihedral 90 deg) at x = 3 m."""
    fin = LiftingSurface(
        name="fin",
        semi_span=1.0,
        sections=[WingSection(y_frac=0.0, chord=0.8), WingSection(y_frac=1.0, chord=0.8)],
        is_symmetric=False,
        dihedral=np.radians(90.0),
        position=np.array([3.0, 0.0, 0.0]),
    )
    return Aircraft(surfaces=[_dihedral_wing(0.0), fin])


def _linear_free_polar() -> TabulatedAirfoil:
    """Return a free-air polar that is linear in alpha (no interpolation error)."""
    a = np.radians(np.linspace(-10.0, 20.0, 31))
    return TabulatedAirfoil(name="fin polar", alpha=a, Cl_data=5.8 * (a - 0.01), Cd_data=0.01 + 0.0 * a,
                            Cm_data=0.0 * a)


@pytest.mark.parametrize("model, h", [("vlm", 0.4), ("llt", 1.5)])
def test_fin_with_a_free_air_polar_in_ground_effect(model, h):
    """Wing and vertical fin in ground effect: a fin with a free-air polar does not read the ground table.

    The wing strips read the circulation lift of the ground table. The fin
    strips read the Cl of their free-air polar, and on the lattice their
    section relation has no ground images (the free-air section slope).
    """
    V = 30.0
    cond = FlightCondition(V_inf=V, alpha=np.radians(4.0), beta=np.radians(3.0), h=h)
    st = SolverSettings(n_panels=12, n_chord=4 if model == "vlm" else 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        res = solve_viscous(_wing_and_fin(), cond, st, {"w": _analytic_ground_table(), "fin": _linear_free_polar()},
                            model=model, tol=1e-11)
    _assert_no_lift_or_convergence_warning(caught)
    assert res.converged
    assert _GROUND_LIFT_NOTE not in res.details["notes"]
    lat = res.details["lattice"]
    fin = np.flatnonzero(lat.strip_surface == 1)
    wing = np.flatnonzero((lat.strip_surface == 0) & lat.strip_is_right)
    assert fin.size > 0 and wing.size > 0
    used = res.details["strip_circulation_lift"]
    assert np.all(used[wing]) and not np.any(used[fin])
    alpha_v = res.details["alpha_eff"]
    hc = res.details["strip_h_over_c"]
    np.testing.assert_allclose(_strip_circulation_lift(res, wing, V),
                               _analytic_circulation_lift(alpha_v[wing], hc[wing]), rtol=0, atol=1e-8)
    np.testing.assert_allclose(_strip_circulation_lift(res, fin, V), 5.8 * (alpha_v[fin] - 0.01),
                               rtol=0, atol=1e-8)
    if model == "vlm":
        # Free-air section of the fin strips: no images, the 2 pi of thin-airfoil theory.
        np.testing.assert_allclose(res.details["lift_slope_section"][fin], 2.0 * np.pi, rtol=1e-6)


@pytest.mark.parametrize("model, h", [("vlm", 0.4), ("llt", 1.5)])
def test_fin_strips_with_only_a_ground_table_are_refused(model, h):
    """Regression: a fin strip that read a ground table had a degenerate section slope.

    The images of the section came from the 3D mirror in the ground, and
    the slope was -5.35 to 4.75 1/rad at 90 deg and 160 1/rad at 89 deg.
    A strip steeper than 60 deg to the ground now uses free-air section
    data. A ground table has none, so the coupling refuses the case and
    asks for a free-air polar for the surface.
    """
    # The fin is 90 deg to the body; at alpha 4 deg it is 86 deg to the ground.
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=h)
    st = SolverSettings(n_panels=12, n_chord=4 if model == "vlm" else 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match=r"'fin'.*86\.0 deg.*free-air polar for this surface"):
            solve_viscous(_wing_and_fin(), cond, st, _analytic_ground_table(), model=model)


# ═══════════════════════════════════════════════════════════════════════════════
# Image distance in the section plane of a strip with dihedral
# ═══════════════════════════════════════════════════════════════════════════════

def _flat_wing(dihedral_deg: float) -> LiftingSurface:
    """Return a flat rectangular one-sided wing (chord 1 m, semi-span 4 m) with dihedral."""
    return LiftingSurface(
        name="w", semi_span=4.0, is_symmetric=False,
        sections=[WingSection(y_frac=0.0, chord=1.0), WingSection(y_frac=1.0, chord=1.0)],
        dihedral=np.radians(dihedral_deg),
    )


def _lattice(dihedral_deg: float):
    cond = FlightCondition(V_inf=30.0, alpha=0.0)
    return make_solver("vlm").build(Aircraft(surfaces=[_flat_wing(dihedral_deg)]), SolverSettings(n_panels=8, n_chord=6), cond,
                                    None, np.zeros(3))


@pytest.mark.parametrize("dihedral_deg", [20.0, 45.0])
def test_dihedral_section_images_are_at_the_section_plane_distance(dihedral_deg):
    """The 2D section of a strip with dihedral sees the ground at h / cos(Gamma) in its own plane.

    The section relation of a strip with dihedral Gamma at the height h
    equals that of a planar strip at the height h / cos(Gamma), the
    distance from the quarter-chord point to the ground line of the
    section plane (the 3D panel coupling uses the same line). The old 3D
    mirror image gave 2 h cos(Gamma) in the section plane, the section of a
    planar strip at h cos(Gamma).
    """
    from ventorum.aero.system import GroundPlane
    from ventorum.viscous.alpha_correction import _section_images, _vlm_section_lines

    gamma = np.radians(dihedral_deg)
    tilted, planar = _lattice(dihedral_deg), _lattice(0.0)
    s = tilted.n_strips // 2
    ground_t = GroundPlane(np.array([0.0, 0.0, 1.0]), float(tilted.qc_mid[s, 2]) - 0.3)
    h = float(ground_t.height(tilted.qc_mid[s][None, :])[0])
    assert h == pytest.approx(0.3, abs=1e-12)
    ground_p = GroundPlane(np.array([0.0, 0.0, 1.0]), float(planar.qc_mid[s, 2]) - h / np.cos(gamma))
    one = np.array([s])
    got = _vlm_section_lines(tilted, one, ground_t, np.array([True]))
    ref = _vlm_section_lines(planar, one, ground_p, np.array([True]))
    np.testing.assert_allclose(np.ravel(got), np.ravel(ref), rtol=1e-10, atol=1e-12)
    # The ground changes the section: the check can see a wrong distance.
    free = _vlm_section_lines(planar, one, None, np.array([False]))
    assert abs(float(ref[0][0]) - float(free[0][0])) > 1e-2
    # The image of a point is 2 h / cos(Gamma) from it in the section plane.
    q = tilted.qc_mid[s][None, :]
    d = _section_images(ground_t, q, tilted.chord_dir[s], tilted.normal[s]) - q
    in_plane = np.hypot(d @ tilted.chord_dir[s], d @ tilted.normal[s])
    assert float(in_plane[0]) == pytest.approx(2.0 * h / np.cos(gamma), rel=1e-12)


def test_planar_strip_images_are_the_3d_mirror_images():
    """A strip whose span axis lies in the ground plane keeps the 3D mirror images, bit for bit."""
    from ventorum.aero.system import GroundPlane
    from ventorum.viscous.alpha_correction import _section_images

    lat = _lattice(0.0)
    ground = GroundPlane(np.array([np.sin(0.07), 0.0, np.cos(0.07)]), -0.4)
    for s in range(lat.n_strips):
        pts = lat.cp[lat.panel_strip == s]
        np.testing.assert_array_equal(_section_images(ground, pts, lat.chord_dir[s], lat.normal[s]),
                                      ground.reflect_points(pts))


def test_section_plane_normal_to_the_ground_line_is_refused():
    """A section plane that holds the ground normal has no ground line: a clear error, no NaN."""
    from ventorum.aero.system import GroundPlane
    from ventorum.viscous.alpha_correction import _section_images

    ground = GroundPlane(np.array([0.0, 0.0, 1.0]), -1.0)
    pts = np.array([[0.25, 0.0, 0.5]])
    with pytest.raises(ValueError, match="no normal in this 2D section plane"):
        _section_images(ground, pts, np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]))


# ═══════════════════════════════════════════════════════════════════════════════
# Steep strips (more than 60 deg to the ground) use free-air section data
# ═══════════════════════════════════════════════════════════════════════════════

def _steep_wing(dihedral_deg: float) -> LiftingSurface:
    """Return a symmetric NACA 0012 wing (chord 1 m, semi-span 2 m) with dihedral."""
    return LiftingSurface(
        name="w", semi_span=2.0,
        sections=[WingSection(y_frac=0.0, chord=1.0, shape="0012"), WingSection(y_frac=1.0, chord=1.0, shape="0012")],
        dihedral=np.radians(dihedral_deg),
    )


def _steep_case(dihedral_deg, polars, h=0.4):
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(2.0), h=h)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        res = solve_viscous(_steep_wing(dihedral_deg), cond, SolverSettings(n_panels=8, n_chord=4), polars,
                            model="vlm", tol=1e-10)
    return res, [str(w.message) for w in caught]


def _free_air_slope(dihedral_deg):
    from ventorum.viscous.alpha_correction import _vlm_section_lines

    lat = make_solver("vlm").build(Aircraft(surfaces=[_steep_wing(dihedral_deg)]),
                                   SolverSettings(n_panels=8, n_chord=4),
                                   FlightCondition(V_inf=30.0, alpha=np.radians(2.0)), None, np.zeros(3))
    strips = np.arange(lat.n_strips)
    return _vlm_section_lines(lat, strips, None, np.zeros(lat.n_strips, dtype=bool))[0]


def test_steep_strip_with_a_ground_source_reads_free_air(tmp_path, fake_2d):
    """A 60.5 deg strip: the ground source runs without the height, no images, a trust note."""
    from ventorum.viscous.alpha_correction import STEEP_STRIP_LIMIT_DEG

    assert STEEP_STRIP_LIMIT_DEG == 60.0
    res, texts = _steep_case(60.5, Ventorum2DPolarSource(cache=PolarCache(tmp_path)))
    assert res.converged
    assert fake_2d and all(call.get("height") is None for call in fake_2d)
    assert not np.any(res.details["strip_circulation_lift"])
    np.testing.assert_allclose(res.details["lift_slope_section"], _free_air_slope(60.5), rtol=1e-12)
    n_ind = int(np.sum(res.details["lattice"].strip_is_right))
    note = (f"{n_ind} of {n_ind} strips are steeper than 60 deg to the ground plane; "
            "they use free-air section data")
    assert any(t.startswith(note) for t in res.details["notes"])
    assert any(t.startswith(note) for t in res.totals.trust.warnings)
    assert any(t.startswith(note) for t in texts)


def test_steep_strip_with_only_a_ground_table_is_refused():
    with pytest.raises(ValueError, match=r"'w'.*60\.4 deg.*free-air polar for this surface"):
        _steep_case(60.5, _analytic_ground_table())


def test_steep_strip_with_a_free_air_polar_in_the_dict_runs():
    res, _ = _steep_case(60.5, {"w": _linear_free_polar()})
    assert res.converged
    assert not any("steeper than" in t for t in res.details["notes"])


@pytest.mark.parametrize("kind", ["source", "table"])
def test_strip_below_the_limit_keeps_the_ground_relation(tmp_path, fake_2d, kind):
    """A 59.5 deg strip reads the polar in ground effect, with images in its section relation."""
    polars = Ventorum2DPolarSource(cache=PolarCache(tmp_path)) if kind == "source" else _analytic_ground_table()
    res, _ = _steep_case(59.5, polars)
    assert res.converged
    if kind == "source":
        assert fake_2d and all(call.get("height") is not None for call in fake_2d)
    assert np.all(res.details["strip_circulation_lift"][res.details["lattice"].strip_is_right])
    assert not any("steeper than" in t for t in res.details["notes"])
    assert np.min(np.abs(res.details["lift_slope_section"] - _free_air_slope(59.5))) > 1e-3


# ═══════════════════════════════════════════════════════════════════════════════
# Polar read height of a strip with dihedral
# ═══════════════════════════════════════════════════════════════════════════════

class _HeightSource:
    """Ground source that records the heights it gets and gives a linear circulation polar."""

    supports_ground = True

    def __init__(self):
        self.heights = []

    def polar(self, section, Re, alphas_deg, *, mach=0.0, flap=None, height=None, **options):
        self.heights.append(height)
        alpha = np.radians(np.asarray(alphas_deg, dtype=float))
        cl = 2.0 * np.pi * alpha
        return TabulatedAirfoil(name="lin", alpha=alpha, Cl_data=cl, Cl_circulation_data=cl,
                                Cd_data=np.full(alpha.shape, 0.01), Cm_data=np.zeros(alpha.shape), Re=float(Re))


@pytest.mark.parametrize("model", ["vlm", "llt", "panel3d"])
def test_dihedral_strip_reads_the_polar_at_the_section_plane_height(model):
    """A strip with dihedral reads its ground polar at h / (c cos(Gamma_g)) in every V1 path.

    The 2D section relation of the strip has its images at the section-plane
    distance h / cos(Gamma_g). The polar is now read at the same distance;
    before, it was read at the vertical height h / c.
    """
    source = _HeightSource()
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=1.5 if model == "llt" else 0.5)
    wing = LiftingSurface(
        name="w", semi_span=4.0, dihedral=np.radians(20.0),
        sections=[WingSection(y_frac=0.0, chord=1.0, shape="0012"), WingSection(y_frac=1.0, chord=1.0, shape="0012")],
    )
    if model == "panel3d":
        st = SolverSettings(solver_type="panel3d", n_panels=4, n_chord=8, spacing="uniform",
                            panel_backend="numpy", panel_cache=False, panel_ground_solver="direct")
    else:
        st = SolverSettings(n_panels=8, n_chord=4 if model == "vlm" else 1, spacing="uniform")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = solve_viscous(wing, cond, st, source, model=model)
    assert res.converged
    # The strips of the polar read: the lattice of the condition (one chordwise panel is enough).
    from ventorum.aero.system import ground_plane_from_condition

    lat = make_solver("vlm").build(Aircraft(surfaces=[wing]), SolverSettings(n_panels=st.n_panels, n_chord=1, spacing="uniform"),
                                   cond, None, np.zeros(3))
    ground = ground_plane_from_condition(lat, cond, np.zeros(3))
    strips = np.flatnonzero(lat.strip_is_right)
    hc = ground.height(lat.qc_mid[strips]) / lat.chord[strips]
    probe = SimpleNamespace(details={"lattice": lat, "ground": ground})
    cosine = _section_plane_cosine(probe, strips)
    assert np.all(cosine < 0.95)
    expected = np.round(hc / cosine, 6)
    got = np.array(sorted(set(source.heights)))
    np.testing.assert_allclose(got, np.sort(np.unique(expected)), rtol=0, atol=2e-6)
    # The vertical heights are not read.
    assert np.min(np.abs(got - np.round(hc.min(), 6))) > 1e-2


def test_planar_strip_read_height_is_the_vertical_height_bit_for_bit():
    """A planar strip reads its polar at h / c with no arithmetic change, also with a tilted ground."""
    from ventorum.aero.system import GroundPlane
    from ventorum.viscous.alpha_correction import _polar_read_heights

    lat = _lattice(0.0)
    strips = np.arange(lat.n_strips)
    ground = GroundPlane(np.array([np.sin(0.07), 0.0, np.cos(0.07)]), -0.4)
    hc = ground.height(lat.qc_mid) / lat.chord
    assert np.array_equal(_polar_read_heights(lat, strips, ground, hc), hc)
