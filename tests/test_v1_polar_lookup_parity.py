# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Parity of the vectorised polar lookup of the V1 alpha-correction coupling.

The Newton iteration of the coupling reads the lift coefficient and the
lift slope of each strip polar at each step. The vectorised lookup
(``_LiftLookup``) and the shared PCHIP construction of many polars
(``_build_pchip_batch``) must give the same values as the per-strip reads
of :class:`TabulatedAirfoil`, to round-off (relative 1e-12). These tests
compare the two paths on random angles and heights, and compare complete
V1 solves with the per-strip path.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ventorum.viscous.alpha_correction as ac
from ventorum.aero.polar_table import PolarTable
from ventorum.core.datatypes import (
    FlightCondition,
    LiftingSurface,
    LinearAirfoil,
    SolverSettings,
    TabulatedAirfoil,
    WingSection,
    _build_pchip_batch,
    _LiftLookup,
)
from ventorum.geometry.lattice import section_cl, section_cl_slope

#: Relative and absolute tolerances of the parity checks [-].
REL_SAME = 1e-12
ABS_SAME = 1e-14

ALPHA_DEG = np.linspace(-20.0, 45.0, 131)
HEIGHTS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0])


def _stall_curves(alpha_deg: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Cl, Cd and Cm [-] of a synthetic polar with a stall near 10 deg."""
    a = np.radians(alpha_deg)
    a_stall = np.radians(10.0)
    cl = np.where(a <= a_stall, 2.0 * np.pi * a, 2.0 * np.pi * a_stall - 2.0 * (a - a_stall)) * factor
    cd = 0.01 + 0.005 * a**2 + 0.4 * np.maximum(a - a_stall, 0.0)
    cm = -0.05 * np.sin(2.0 * a) * factor
    return cl, cd, cm


def _table(circulation: bool) -> PolarTable:
    """Return a synthetic polar table with a height axis (h/c 0.05 to 2)."""
    curves = [_stall_curves(ALPHA_DEG, 1.0 + 0.05 / h) for h in HEIGHTS]
    cl = np.array([[[c[0] for c in curves]]] * 2)
    cd = np.array([[[c[1] for c in curves]]] * 2)
    cm = np.array([[[c[2] for c in curves]]] * 2)
    circ = cl * 1.07 + 0.01 if circulation else None
    return PolarTable(Re=np.array([1e5, 1e8]), flap_deg=np.array([0.0]), alpha_deg=ALPHA_DEG,
                      Cl=cl, Cd=cd, Cm=cm, Cl_circulation=circ, h_over_c=HEIGHTS,
                      source_info={"source": "synthetic stall"})


def _free_polar() -> TabulatedAirfoil:
    """Return a free-air tabulated polar (-20 to 45 deg)."""
    cl, cd, cm = _stall_curves(ALPHA_DEG, 1.0)
    return TabulatedAirfoil(name="free", alpha=np.radians(ALPHA_DEG), Cl_data=cl, Cd_data=cd, Cm_data=cm, Re=1e6)


def _random_angles(rng: np.random.Generator, n: int, table_alpha: np.ndarray) -> np.ndarray:
    """Return angles [rad]: random values inside and outside the table, and table points."""
    a = rng.uniform(np.radians(-30.0), np.radians(55.0), n)
    pick = rng.random(n) < 0.2
    a[pick] = rng.choice(table_alpha, int(pick.sum()))
    a[0], a[1] = table_alpha[0], table_alpha[-1]
    return a


def _scalar_reads(polars, circulation, alpha):
    """Return Cl [-] and dCl/dalpha [1/rad] read strip by strip (the per-strip path)."""
    cl = np.array([
        float(p.Cl_circulation(a)) if c else float(section_cl(p, a))
        for p, a, c in zip(polars, alpha, circulation)
    ])
    slope = np.array([
        float(p.Cl_circulation_slope(a)) if c else float(section_cl_slope(p, a))
        for p, a, c in zip(polars, alpha, circulation)
    ])
    return cl, slope


@pytest.mark.parametrize("circulation", [False, True])
def test_lookup_matches_per_strip_reads_on_random_angles_and_heights(circulation):
    """The lookup gives the per-strip values on random angles and random heights."""
    rng = np.random.default_rng(7 + int(circulation))
    table = _table(circulation)
    for _ in range(5):
        heights = rng.uniform(0.05, 2.0, 37)
        polars = [table.at(Re=1e6, flap_deg=0.0, h_over_c=float(h)) for h in heights]
        # The reference airfoils make their own interpolation (a clone has no cache).
        reference = [p.clone() for p in polars]
        circ = np.full(len(polars), circulation)
        lookup = _LiftLookup(polars, circ)
        assert lookup.handled.all()
        for _ in range(20):
            alpha = _random_angles(rng, len(polars), polars[0].tables()[0])
            cl, slope = lookup(alpha)
            cl_ref, slope_ref = _scalar_reads(reference, circ, alpha)
            np.testing.assert_allclose(cl, cl_ref, rtol=REL_SAME, atol=ABS_SAME)
            np.testing.assert_allclose(slope, slope_ref, rtol=REL_SAME, atol=ABS_SAME)


def test_shared_construction_matches_own_construction():
    """Cl, Cd, Cm, the circulation lift and the slopes agree after the shared construction."""
    rng = np.random.default_rng(11)
    table = _table(True)
    polars = [table.at(Re=1e6, flap_deg=0.0, h_over_c=float(h)) for h in rng.uniform(0.05, 2.0, 12)]
    reference = [p.clone() for p in polars]
    _build_pchip_batch(polars)
    assert all(p._interp is not None for p in polars)
    alpha = _random_angles(rng, 400, polars[0].tables()[0])
    for p, r in zip(polars, reference):
        for name in ("Cl", "Cd", "Cm", "Cl_slope", "Cl_circulation", "Cl_circulation_slope"):
            np.testing.assert_allclose(getattr(p, name)(alpha), getattr(r, name)(alpha),
                                       rtol=REL_SAME, atol=ABS_SAME, err_msg=name)


def test_lookup_with_mixed_entries():
    """Shared polars, different angle grids and a linear airfoil give the per-strip values."""
    rng = np.random.default_rng(3)
    free = _free_polar()
    short_deg = np.linspace(-10.0, 20.0, 41)
    cl, cd, cm = _stall_curves(short_deg, 1.1)
    short = TabulatedAirfoil(name="short", alpha=np.radians(short_deg), Cl_data=cl, Cd_data=cd, Cm_data=cm)
    linear = LinearAirfoil()
    polars = [free, short, free, linear, short, free]
    circ = np.zeros(len(polars), dtype=bool)
    lookup = _LiftLookup(polars, circ)
    np.testing.assert_array_equal(lookup.handled, [True, True, True, False, True, True])
    for _ in range(50):
        alpha = rng.uniform(np.radians(-30.0), np.radians(55.0), len(polars))
        got_cl, got_slope = lookup(alpha)
        ref_cl, ref_slope = _scalar_reads(polars, circ, alpha)
        h = lookup.handled
        np.testing.assert_allclose(got_cl[h], ref_cl[h], rtol=REL_SAME, atol=ABS_SAME)
        np.testing.assert_allclose(got_slope[h], ref_slope[h], rtol=REL_SAME, atol=ABS_SAME)
        assert np.all(np.isnan(got_cl[~h]))


def test_lookup_keeps_the_gap_warning():
    """A lookup inside a bridged interval warns once per polar, as the per-strip read does."""
    polar = _free_polar()
    mask = np.zeros(polar.tables()[0].size - 1, dtype=bool)
    mask[60] = True
    polar.gap_mask = mask
    a_gap = 0.5 * (polar.tables()[0][60] + polar.tables()[0][61])
    lookup = _LiftLookup([polar, polar], [False, False])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lookup(np.array([0.0, a_gap]))
        lookup(np.array([a_gap, a_gap]))
    assert sum("bridges" in str(w.message) for w in caught) == 1


class _PerStripLookup:
    """The per-strip polar reads of the Newton residual (the path before the vectorised lookup)."""

    def __init__(self, airfoils, circulation) -> None:
        self.polars = list(airfoils)
        self.circ = np.asarray(circulation, dtype=bool)
        self.handled = np.ones(len(self.polars), dtype=bool)

    def __call__(self, alpha):
        return _scalar_reads(self.polars, self.circ, alpha)


def _wing() -> LiftingSurface:
    """Return a rectangular wing of aspect ratio 8 (chord 1 m, semi-span 4 m)."""
    return LiftingSurface(name="w", semi_span=4.0,
                          sections=[WingSection(y_frac=0.0, chord=1.0), WingSection(y_frac=1.0, chord=1.0)])


def _assert_same_result(new, old):
    """Check the totals and the strip values of two solves to round-off."""
    for name in ("CL", "CDi", "Cm", "CD_total"):
        np.testing.assert_allclose(getattr(new.totals, name), getattr(old.totals, name),
                                   rtol=REL_SAME, atol=ABS_SAME, err_msg=name)
    assert new.converged == old.converged
    assert len(new.spanwise) == len(old.spanwise)
    for s_new, s_old in zip(new.spanwise, old.spanwise):
        for name in ("Cl", "Cd_i", "Cd_profile", "alpha_eff", "Cm_section", "gamma"):
            np.testing.assert_allclose(getattr(s_new, name), getattr(s_old, name),
                                       rtol=REL_SAME, atol=ABS_SAME, err_msg=name)


@pytest.mark.parametrize(
    ("model", "settings", "h_over_c", "circulation"),
    [
        ("vlm", SolverSettings(n_panels=16, n_chord=4), None, False),
        ("vlm", SolverSettings(n_panels=16, n_chord=4), 0.3, False),
        ("vlm", SolverSettings(n_panels=16, n_chord=4), 0.3, True),
        ("llt", SolverSettings(n_panels=32), None, False),
        ("llt", SolverSettings(n_panels=32), 1.5, False),
    ],
)
def test_solve_matches_per_strip_path(monkeypatch, model, settings, h_over_c, circulation):
    """A V1 sweep through the stall gives the same results with the vectorised lookup."""
    polar = _free_polar() if h_over_c is None else _table(circulation)
    conditions = [
        FlightCondition(V_inf=40.0, alpha=np.radians(a), h=h_over_c)
        for a in (4.0, 11.0, 14.0)
    ]

    def solve():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ac.solve_viscous(_wing(), conditions, settings, polar, model=model)

    new = solve()
    monkeypatch.setattr(ac, "_LiftLookup", _PerStripLookup)
    old = solve()
    assert len(new) == len(old)
    for r_new, r_old in zip(new, old):
        _assert_same_result(r_new, r_old)
