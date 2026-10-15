# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for section polar tables over Reynolds number and flap deflection.

Unit tests use a fake analytic source with data linear in log10(Re) and in
flap deflection, plus planted NaN points. The integration test uses the real
XFOIL binary and skips when no executable is found.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from ventorum.aero.polar_table import PolarTable, build_polar_table
from ventorum.core.datatypes import TabulatedAirfoil


_XFOIL_EXE = shutil.which(os.environ.get("VENTORUM_XFOIL", "xfoil"))
needs_xfoil = pytest.mark.skipif(
    _XFOIL_EXE is None, reason="XFOIL executable not found"
)


class _AnalyticSource:
    """Fake polar source with data linear in log10(Re) and flap deflection.

    At fixed angle of attack every coefficient is an affine function of
    log10(Re) and of the flap deflection, so bilinear table interpolation
    reproduces it exactly. ``drop`` maps (Re, flap_deg) to a list of alpha
    values [deg] returned as NaN. ``raise_for`` maps (Re, flap_deg) to an
    exception raised by :meth:`polar`. With ``with_hook`` the source also
    offers a ``run_xfoil_batch`` hook with the same behaviour.
    """

    def __init__(self, drop=None, raise_for=None, no_cm=False, with_hook=False):
        self.drop = dict(drop or {})
        self.raise_for = dict(raise_for or {})
        self.no_cm = no_cm
        self.with_hook = with_hook
        self.calls: list[dict] = []
        self.hook_workers: list = []

    @staticmethod
    def _coeffs(Re, flap_deg, alpha_deg):
        logre = float(np.log10(float(Re)))
        alpha = np.radians(np.asarray(alpha_deg, dtype=float))
        a0 = float(np.radians(-2.0))
        cl = (
            2.0 * np.pi * (alpha - a0)
            + 0.12 * (logre - 6.0)
            + 0.012 * float(flap_deg)
        )
        cd = (
            0.0085
            + 0.6 * alpha**2
            - 0.0012 * (logre - 6.0)
            + 0.00025 * float(flap_deg)
        )
        cm = (
            np.zeros_like(alpha)
            - 0.045
            + 0.0008 * (logre - 6.0)
            - 0.0035 * float(flap_deg)
        )
        return cl, cd, cm

    def _make_polar(self, Re, flap_deg, alphas_deg):
        key = (float(Re), float(flap_deg))
        if key in self.raise_for:
            raise self.raise_for[key]
        alphas = np.asarray(list(alphas_deg), dtype=float)
        cl, cd, cm = self._make_coeffs(Re, flap_deg, alphas)
        for bad in self.drop.get(key, []):
            hit = np.flatnonzero(np.isclose(alphas, float(bad)))
            cl[hit] = np.nan
            cd[hit] = np.nan
            cm[hit] = np.nan
        return TabulatedAirfoil(
            name=f"fake Re={Re:g} flap={flap_deg:g}",
            alpha=np.radians(alphas),
            Cl_data=np.asarray(cl, dtype=float),
            Cd_data=np.asarray(cd, dtype=float),
            Cm_data=None if self.no_cm else np.asarray(cm, dtype=float),
            Re=float(Re),
        )

    _make_coeffs = _coeffs

    def polar(self, section, Re, alphas_deg, *, mach=0.0, flap=None, **options):
        flap_deg = 0.0 if flap is None else float(flap[2])
        self.calls.append(
            {"Re": float(Re), "flap_deg": flap_deg, "flap": flap}
        )
        return self._make_polar(float(Re), flap_deg, alphas_deg)

    def run_xfoil_batch(self, cases, workers=None):
        """Batch hook with the run_xfoil_batch calling convention."""
        self.hook_workers.append(workers)
        out = []
        for case in cases:
            flap = case.get("flap")
            flap_deg = 0.0 if flap is None else float(flap[2])
            self.calls.append(
                {"Re": float(case["Re"]), "flap_deg": flap_deg, "flap": flap}
            )
            key = (float(case["Re"]), flap_deg)
            if key in self.raise_for:
                out.append(self.raise_for[key])
                continue
            out.append(
                self._make_polar(case["Re"], flap_deg, case["alphas_deg"])
            )
        return out


def _std_table(source=None, **kwargs):
    params = {
        "source": _AnalyticSource() if source is None else source,
        "section": "naca 2412",
        "Re_values": [5e5, 1e6],
        "alpha_deg": list(range(-2, 9)),
        "flap_deg": [0.0, 10.0],
        "x_hinge": 0.75,
    }
    params.update(kwargs)
    return build_polar_table(**params)


def test_build_sorts_axes_and_shapes_match():
    table = _std_table(
        Re_values=[1e6, 5e5], flap_deg=[10.0, 0.0], workers=1
    )
    assert list(table.Re) == [5e5, 1e6]
    assert list(table.flap_deg) == [0.0, 10.0]
    assert list(table.alpha_deg) == list(range(-2, 9))
    assert table.Cl.shape == (2, 2, 11)
    assert table.Cd.shape == (2, 2, 11)
    assert table.Cm.shape == (2, 2, 11)
    assert np.all(np.isfinite(table.Cl))
    # Data stays attached to its (Re, flap) pair after sorting.
    cl_low, _, _ = _AnalyticSource._coeffs(5e5, 0.0, [0.0])
    assert table.Cl[0, 0, 2] == pytest.approx(float(cl_low[0]))


def test_interpolation_matches_log_linear_data():
    table = _std_table(workers=1)
    re_mid = float(np.sqrt(5e5 * 1e6))
    polar = table.at(re_mid, 5.0)
    assert table.last_dropped == 0
    assert len(polar.alpha) == 11
    for deg in (-2.0, 0.0, 3.0, 8.0):
        cl_e, cd_e, cm_e = _AnalyticSource._coeffs(re_mid, 5.0, [deg])
        assert polar.Cl(np.radians(deg)) == pytest.approx(float(cl_e[0]), rel=1e-12)
        assert polar.Cd(np.radians(deg)) == pytest.approx(float(cd_e[0]), rel=1e-12)
        assert polar.Cm(np.radians(deg)) == pytest.approx(float(cm_e[0]), rel=1e-12)
    # Queries on grid nodes return the node values.
    node = table.at(1e6, 10.0)
    cl_e, _, _ = _AnalyticSource._coeffs(1e6, 10.0, [3.0])
    assert node.Cl(np.radians(3.0)) == pytest.approx(float(cl_e[0]), rel=1e-12)
    assert node.Re == pytest.approx(1e6)


def test_nan_neighbours_drop_alpha_and_count_is_recorded():
    source = _AnalyticSource(drop={(5e5, 0.0): [8.0]})
    table = _std_table(source=source, workers=1)
    assert float(np.sum(np.isnan(table.Cl))) == 1.0
    # The stencil of this query holds the NaN corner, so alpha 8 is dropped.
    polar = table.at(float(np.sqrt(5e5 * 1e6)), 5.0)
    assert table.last_dropped == 1
    assert len(polar.alpha) == 10
    assert 8.0 not in np.degrees(np.asarray(polar.alpha))
    assert "1 of 11" in polar.name
    # The clean corner keeps the full grid when queried on its node.
    clean = table.at(1e6, 10.0)
    assert table.last_dropped == 0
    assert len(clean.alpha) == 11
    assert "0 of 11" in clean.name
    # The NaN node itself loses that alpha point.
    dirty = table.at(5e5, 0.0)
    assert table.last_dropped == 1
    assert len(dirty.alpha) == 10


def test_out_of_range_queries_raise():
    table = _std_table(workers=1)
    with pytest.raises(ValueError):
        table.at(4e5, 0.0)
    with pytest.raises(ValueError):
        table.at(2e6, 0.0)
    with pytest.raises(ValueError):
        table.at(5e5, -1.0)
    with pytest.raises(ValueError):
        table.at(5e5, 11.0)
    with pytest.raises(ValueError):
        table.at(float("nan"), 0.0)
    # Boundary queries are inside the range.
    assert len(table.at(5e5, 0.0).alpha) == 11
    assert len(table.at(1e6, 10.0).alpha) == 11


def test_npz_round_trip_is_exact(tmp_path):
    source = _AnalyticSource(drop={(1e6, 10.0): [0.0]})
    table = _std_table(source=source, workers=1)
    path = tmp_path / "polars.npz"
    assert table.to_npz(path) == path
    back = PolarTable.from_npz(path)
    np.testing.assert_array_equal(back.Re, table.Re)
    np.testing.assert_array_equal(back.flap_deg, table.flap_deg)
    np.testing.assert_array_equal(back.alpha_deg, table.alpha_deg)
    np.testing.assert_array_equal(back.Cl, table.Cl)
    np.testing.assert_array_equal(back.Cd, table.Cd)
    np.testing.assert_array_equal(back.Cm, table.Cm)
    assert back.source_info == table.source_info
    first = table.at(7.5e5, 3.0)
    second = back.at(7.5e5, 3.0)
    np.testing.assert_array_equal(second.alpha, first.alpha)
    np.testing.assert_array_equal(second.Cl_data, first.Cl_data)


def test_batch_hook_runs_when_source_offers_it():
    hooked = _AnalyticSource(with_hook=True)
    plain = _AnalyticSource()
    assert hooked.with_hook
    from_hook = _std_table(source=hooked, workers=3)
    assert hooked.hook_workers == [3]
    # Six (Re, flap) pairs go through the hook in one call batch.
    assert len(hooked.calls) == 2 * 2
    direct = _std_table(source=plain, workers=1)
    np.testing.assert_array_equal(from_hook.Cl, direct.Cl)
    np.testing.assert_array_equal(from_hook.Cd, direct.Cd)


def test_hook_exception_for_one_pair_becomes_nan():
    err = RuntimeError("XFOIL did not converge for this case")
    hooked = _AnalyticSource(
        with_hook=True, raise_for={(1e6, 10.0): err}
    )
    table = _std_table(source=hooked, workers=1)
    assert np.all(np.isnan(table.Cl[1, 1, :]))
    assert np.all(np.isfinite(table.Cl[0, 0, :]))
    cov = table.coverage()
    assert cov[1, 1] == pytest.approx(0.0)
    gone = table.at(1e6, 10.0)
    assert table.last_dropped == 11
    assert len(gone.alpha) == 0


def test_workers_thread_path_matches_sequential():
    threaded = _std_table(source=_AnalyticSource(), workers=2)
    sequential = _std_table(source=_AnalyticSource(), workers=1)
    np.testing.assert_array_equal(threaded.Cl, sequential.Cl)
    np.testing.assert_array_equal(threaded.Cm, sequential.Cm)
    with pytest.raises(ValueError):
        _std_table(source=_AnalyticSource(), workers=0)


def test_failed_pair_becomes_nan_slice():
    err = RuntimeError("XFOIL failed to converge for any point")
    source = _AnalyticSource(raise_for={(5e5, 10.0): err})
    table = _std_table(source=source, workers=1)
    assert np.all(np.isnan(table.Cl[0, 1, :]))
    assert np.all(np.isfinite(table.Cl[1, 1, :]))
    gone = table.at(5e5, 10.0)
    assert table.last_dropped == 11
    assert len(gone.alpha) == 0


def test_unexpected_error_propagates():
    source = _AnalyticSource(raise_for={(5e5, 0.0): ValueError("bad input")})
    with pytest.raises(ValueError, match="bad input"):
        _std_table(source=source, workers=1)


def test_flap_needs_hinge():
    source = _AnalyticSource()
    with pytest.raises(ValueError, match="x_hinge"):
        _std_table(source=source, x_hinge=None)
    clean = _std_table(source=_AnalyticSource(), flap_deg=[0.0], x_hinge=None)
    assert clean.flap_deg.shape == (1,)
    # Zero deflection runs the clean airfoil without a flap tuple.
    recorded = _AnalyticSource()
    _std_table(source=recorded, flap_deg=[0.0], x_hinge=None, workers=1)
    assert all(call["flap"] is None for call in recorded.calls)
    # Nonzero deflection forwards the hinge position.
    recorded = _AnalyticSource()
    _std_table(
        source=recorded, flap_deg=[10.0], x_hinge=0.75, y_hinge=0.5, workers=1
    )
    assert recorded.calls[0]["flap"] == (0.75, 0.5, 10.0)


def test_coverage_reports_converged_fraction():
    source = _AnalyticSource(drop={(5e5, 0.0): [-2.0, -1.0], (1e6, 10.0): [8.0]})
    table = _std_table(source=source, workers=1)
    cov = table.coverage()
    assert cov.shape == (2, 2)
    assert cov[0, 0] == pytest.approx(9.0 / 11.0)
    assert cov[1, 1] == pytest.approx(10.0 / 11.0)
    assert cov[0, 1] == pytest.approx(1.0)
    assert cov[1, 0] == pytest.approx(1.0)


def test_single_point_table_without_moment():
    source = _AnalyticSource(no_cm=True)
    table = _std_table(
        source=source,
        Re_values=[1e6],
        alpha_deg=[-2.0, 0.0, 2.0],
        flap_deg=[0.0],
        x_hinge=None,
        workers=1,
    )
    assert table.Cl.shape == (1, 1, 3)
    polar = table.at(1e6, 0.0)
    assert len(polar.alpha) == 3
    assert np.all(np.asarray(polar.Cm_data) == 0.0)
    with pytest.raises(ValueError):
        table.at(2e6, 0.0)
    with pytest.raises(ValueError):
        table.at(1e6, 5.0)


def test_source_info_records_source_options_and_version():
    table = _std_table(workers=1)
    info = table.source_info
    assert info["source"] == "_AnalyticSource"
    assert info["section"] == "naca 2412"
    assert info["options"] == {"x_hinge": 0.75, "y_hinge": 0.5}
    assert "version" in info


def test_from_npz_rejects_bad_input(tmp_path):
    table = _std_table(workers=1)
    path = tmp_path / "polars.npz"
    table.to_npz(path)
    bad = tmp_path / "bad.npz"
    data = dict(np.load(path, allow_pickle=False))
    del data["Cm"]
    np.savez(bad, **data)
    with pytest.raises(ValueError, match="Cm"):
        PolarTable.from_npz(bad)
    wrong = tmp_path / "wrong.npz"
    data = dict(np.load(path, allow_pickle=False))
    data["Cl"] = data["Cl"][:, :, :-1]
    np.savez(wrong, **data)
    with pytest.raises(ValueError):
        PolarTable.from_npz(wrong)


def test_table_rejects_bad_axes():
    good = {
        "Re": np.array([5e5, 1e6]),
        "flap_deg": np.array([0.0, 10.0]),
        "alpha_deg": np.array([-2.0, 0.0, 2.0]),
    }
    zeros = np.zeros((2, 2, 3))
    with pytest.raises(ValueError, match="ascending"):
        PolarTable(
            Re=np.array([1e6, 5e5]),
            flap_deg=good["flap_deg"],
            alpha_deg=good["alpha_deg"],
            Cl=zeros,
            Cd=zeros,
            Cm=zeros,
        )
    with pytest.raises(ValueError, match="positive"):
        PolarTable(
            Re=np.array([-5e5, 1e6]),
            flap_deg=good["flap_deg"],
            alpha_deg=good["alpha_deg"],
            Cl=zeros,
            Cd=zeros,
            Cm=zeros,
        )
    with pytest.raises(ValueError):
        PolarTable(
            Re=good["Re"],
            flap_deg=good["flap_deg"],
            alpha_deg=np.array([-2.0, -2.0, 2.0]),
            Cl=zeros,
            Cd=zeros,
            Cm=zeros,
        )
    with pytest.raises(ValueError, match="shape"):
        PolarTable(
            Re=good["Re"],
            flap_deg=good["flap_deg"],
            alpha_deg=good["alpha_deg"],
            Cl=np.zeros((2, 2, 2)),
            Cd=zeros,
            Cm=zeros,
        )


def test_build_rejects_bad_inputs():
    with pytest.raises(ValueError):
        _std_table(source=_AnalyticSource(), Re_values=[], workers=1)
    with pytest.raises(ValueError):
        _std_table(source=_AnalyticSource(), Re_values=[-1e5], workers=1)
    with pytest.raises(ValueError):
        _std_table(source=_AnalyticSource(), alpha_deg=[], workers=1)


@needs_xfoil
def test_xfoil_table_covers_range_and_shows_flap_and_reynolds_effects():
    from ventorum.aero.polar_source import XfoilPolarSource

    source = XfoilPolarSource(cache=False)
    table = build_polar_table(
        source,
        "naca 2412",
        [5e5, 1e6],
        list(range(-2, 9)),
        flap_deg=[0.0, 10.0],
        x_hinge=0.75,
    )
    cov = table.coverage()
    assert cov.shape == (2, 2)
    assert float(cov.min()) > 0.8
    cl_clean = table.at(1e6, 0.0).Cl(np.radians(0.0))
    cl_flap = table.at(1e6, 10.0).Cl(np.radians(0.0))
    assert float(cl_flap) > float(cl_clean)
    cd_low = table.at(5e5, 0.0).Cd(np.radians(2.0))
    cd_high = table.at(1e6, 0.0).Cd(np.radians(2.0))
    assert float(cd_low) > float(cd_high)
