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
    exception raised by :meth:`polar`.
    """

    def __init__(self, drop=None, raise_for=None, no_cm=False):
        self.drop = dict(drop or {})
        self.raise_for = dict(raise_for or {})
        self.no_cm = no_cm
        self.calls: list[dict] = []

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


def test_calls_run_in_parallel_threads():
    """The builder calls source.polar from several threads at once.

    The batch hook path was dead (no source had the hook and its case
    format did not match). The builder now has one path: a thread pool
    that calls ``source.polar``. Two calls must meet at a barrier, which
    is only possible when they run at the same time.
    """
    import threading

    class _BarrierSource(_AnalyticSource):
        def __init__(self):
            super().__init__()
            self.barrier = threading.Barrier(2, timeout=10.0)
            self.threads: set[int] = set()

        def polar(self, section, Re, alphas_deg, *, mach=0.0, flap=None, **options):
            self.threads.add(threading.get_ident())
            self.barrier.wait()
            return super().polar(section, Re, alphas_deg, mach=mach, flap=flap, **options)

    source = _BarrierSource()
    table = _std_table(source=source, Re_values=[1e6], flap_deg=[0.0, 10.0], workers=2)
    assert len(source.calls) == 2
    assert len(source.threads) == 2
    assert table.Cl.shape == (1, 2, 11)


def test_source_without_hook_still_runs_when_a_batch_attribute_exists():
    """An attribute named like the old hook is ignored: polar() runs."""

    class _OldHookSource(_AnalyticSource):
        def run_xfoil_batch(self, cases, workers=None):
            raise AssertionError("the builder must not call a batch hook")

    table = _std_table(source=_OldHookSource(), workers=2)
    direct = _std_table(source=_AnalyticSource(), workers=1)
    np.testing.assert_array_equal(table.Cl, direct.Cl)


def test_exception_for_one_pair_becomes_nan_in_threads():
    err = RuntimeError("XFOIL did not converge for this case")
    source = _AnalyticSource(raise_for={(1e6, 10.0): err})
    table = _std_table(source=source, workers=3)
    assert np.all(np.isnan(table.Cl[1, 1, :]))
    assert np.all(np.isfinite(table.Cl[0, 0, :]))
    cov = table.coverage()
    assert cov[1, 1] == pytest.approx(0.0)
    gone = table.at(1e6, 10.0)
    assert table.last_dropped == 11
    assert len(gone.alpha) == 0


def test_default_workers_is_the_cpu_count_minus_one(monkeypatch):
    import concurrent.futures

    from ventorum.aero import polar_table as module

    seen: list = []
    real_pool = concurrent.futures.ThreadPoolExecutor

    def recording_pool(max_workers=None, **kwargs):
        seen.append(max_workers)
        return real_pool(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(module.os, "cpu_count", lambda: 5)
    monkeypatch.setattr(module.concurrent.futures, "ThreadPoolExecutor", recording_pool)
    _std_table(source=_AnalyticSource())
    assert seen == [4]
    monkeypatch.setattr(module.os, "cpu_count", lambda: 1)
    calls_before = len(seen)
    _std_table(source=_AnalyticSource())
    # One worker: the calls run one after the other, without a pool.
    assert len(seen) == calls_before


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


def test_source_info_uses_the_public_describe_method():
    """Regression: the options of the source were missing and the version came from a private attribute."""

    class _DescribedSource(_AnalyticSource):
        _version = "private value that must not be read"

        def describe(self):
            return {
                "source": "DescribedSource",
                "options": {"ncrit": 7.0, "xtr_top": 0.4, "xtr_bottom": 1.0,
                            "n_panels": 200, "sweep": "from_zero"},
                "version": "fake 1.0",
            }

    table = _std_table(source=_DescribedSource(), workers=1)
    info = table.source_info
    assert info["source"] == "DescribedSource"
    assert info["version"] == "fake 1.0"
    assert info["options"] == {
        "ncrit": 7.0, "xtr_top": 0.4, "xtr_bottom": 1.0, "n_panels": 200,
        "sweep": "from_zero", "x_hinge": 0.75, "y_hinge": 0.5,
    }
    plain = _std_table(source=_AnalyticSource(), workers=1).source_info
    assert plain["version"] == "unknown"


def test_source_info_of_the_xfoil_source_holds_its_options(monkeypatch):
    from ventorum.aero import polar_source

    monkeypatch.setattr(
        polar_source, "xfoil_version", lambda executable=None: "XFOIL Version 6.99"
    )

    class _FakeXfoil(polar_source.XfoilPolarSource):
        def polar(self, section, Re, alphas_deg, *, mach=0.0, flap=None, **options):
            flap_deg = 0.0 if flap is None else float(flap[2])
            return _AnalyticSource()._make_polar(float(Re), flap_deg, alphas_deg)

    source = _FakeXfoil(cache=False, ncrit=7.0, xtr_top=0.4, n_panels=200)
    info = _std_table(source=source, workers=1).source_info
    assert info["source"] == "_FakeXfoil"  # the class name of the source
    assert info["version"] == "XFOIL Version 6.99"
    opts = info["options"]
    assert opts["ncrit"] == 7.0 and opts["xtr_top"] == 0.4
    assert opts["n_panels"] == 200 and opts["sweep"] == "from_zero"
    assert opts["x_hinge"] == 0.75 and opts["y_hinge"] == 0.5


def test_flap_hinge_height_is_a_thickness_fraction():
    recorded = _AnalyticSource()
    _std_table(source=recorded, flap_deg=[0.0, 10.0], x_hinge=0.7, y_hinge=0.25, workers=1)
    flaps = {c["flap"] for c in recorded.calls}
    assert flaps == {None, (0.7, 0.25, 10.0)}
    with pytest.raises(ValueError, match="thickness"):
        _std_table(source=_AnalyticSource(), x_hinge=0.75, y_hinge=1.5, workers=1)


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


class _AnalyticGroundSource(_AnalyticSource):
    """Analytic polar source supporting height with linear 1/h effect."""

    @staticmethod
    def _coeffs(Re, flap_deg, alpha_deg, height=None):
        cl, cd, cm = _AnalyticSource._coeffs(Re, flap_deg, alpha_deg)
        if height is not None:
            inv_h = 1.0 / float(height)
            cl = cl + 0.05 * inv_h
            cd = cd + 0.005 * inv_h
            cm = cm - 0.01 * inv_h
        return cl, cd, cm

    def _make_polar(self, Re, flap_deg, alphas_deg, height=None):
        key = (float(Re), float(flap_deg))
        if key in self.raise_for:
            raise self.raise_for[key]
        alphas = np.asarray(list(alphas_deg), dtype=float)
        cl, cd, cm = self._coeffs(Re, flap_deg, alphas, height=height)
        for bad in self.drop.get(key, []):
            hit = np.flatnonzero(np.isclose(alphas, float(bad)))
            cl[hit] = np.nan
            cd[hit] = np.nan
            cm[hit] = np.nan
        h_str = f" h={height:g}" if height is not None else ""
        return TabulatedAirfoil(
            name=f"fake Re={Re:g} flap={flap_deg:g}{h_str}",
            alpha=np.radians(alphas),
            Cl_data=np.asarray(cl, dtype=float),
            Cd_data=np.asarray(cd, dtype=float),
            Cm_data=None if self.no_cm else np.asarray(cm, dtype=float),
            Re=float(Re),
        )

    def polar(
        self,
        section,
        Re,
        alphas_deg,
        *,
        mach=0.0,
        flap=None,
        height=None,
        **options,
    ):
        flap_deg = 0.0 if flap is None else float(flap[2])
        self.calls.append(
            {
                "Re": float(Re),
                "flap_deg": flap_deg,
                "flap": flap,
                "height": height,
            }
        )
        return self._make_polar(
            float(Re), flap_deg, alphas_deg, height=height
        )


def test_ground_table_shapes_coverage_and_interpolation():
    source = _AnalyticGroundSource()
    alphas = list(range(-2, 3))
    table = build_polar_table(
        source,
        "naca 2412",
        [5e5, 1e6],
        alphas,
        flap_deg=[0.0, 10.0],
        x_hinge=0.75,
        h_over_c=[0.2, 0.5, 1e3],
        workers=1,
    )
    assert table.h_over_c is not None
    assert np.allclose(table.h_over_c, [0.2, 0.5, 1e3])
    assert table.Cl.shape == (2, 2, 3, 5)
    assert table.Cd.shape == (2, 2, 3, 5)
    assert table.Cm.shape == (2, 2, 3, 5)

    cov = table.coverage()
    assert cov.shape == (2, 2, 3)
    assert np.all(cov == 1.0)

    for re in (5e5, 1e6):
        for flap in (0.0, 10.0):
            for h in (0.2, 0.5, 1e3):
                polar = table.at(re, flap, h_over_c=h)
                for deg in (-2.0, 0.0, 2.0):
                    cl_e, cd_e, cm_e = _AnalyticGroundSource._coeffs(
                        re, flap, [deg], height=h
                    )
                    assert polar.Cl(np.radians(deg)) == pytest.approx(
                        float(cl_e[0]), rel=1e-12
                    )
                    assert polar.Cd(np.radians(deg)) == pytest.approx(
                        float(cd_e[0]), rel=1e-12
                    )
                    assert polar.Cm(np.radians(deg)) == pytest.approx(
                        float(cm_e[0]), rel=1e-12
                    )

    h_mid = 0.3
    polar_interp = table.at(5e5, 0.0, h_over_c=h_mid)
    cl_mid, cd_mid, cm_mid = _AnalyticGroundSource._coeffs(
        5e5, 0.0, [0.0], height=h_mid
    )
    assert polar_interp.Cl(0.0) == pytest.approx(float(cl_mid[0]), rel=1e-12)
    assert polar_interp.Cd(0.0) == pytest.approx(float(cd_mid[0]), rel=1e-12)
    assert polar_interp.Cm(0.0) == pytest.approx(float(cm_mid[0]), rel=1e-12)

    free_source = _AnalyticSource()
    free_table = build_polar_table(
        free_source,
        "naca 2412",
        [5e5, 1e6],
        alphas,
        flap_deg=[0.0, 10.0],
        x_hinge=0.75,
        workers=1,
    )
    polar_g_inf = table.at(1e6, 0.0, h_over_c=1e3)
    polar_free = free_table.at(1e6, 0.0)
    for deg in alphas:
        cl_g = polar_g_inf.Cl(np.radians(deg))
        cl_f = polar_free.Cl(np.radians(deg))
        assert abs(cl_g - cl_f) < 1e-3


def test_build_polar_table_rejects_source_without_ground_support():
    source = _AnalyticSource()
    with pytest.raises(ValueError, match="does not support ground effect"):
        build_polar_table(
            source,
            "naca 2412",
            [5e5, 1e6],
            [-2, 0, 2],
            h_over_c=[0.2, 0.5],
            workers=1,
        )


def test_polar_table_at_validates_h_over_c():
    source_free = _AnalyticSource()
    table_free = _std_table(source=source_free, workers=1)
    with pytest.raises(ValueError, match="no height axis"):
        table_free.at(1e6, 0.0, h_over_c=0.5)

    source_g = _AnalyticGroundSource()
    table_g = build_polar_table(
        source_g,
        "naca 2412",
        [5e5, 1e6],
        [-2, 0, 2],
        flap_deg=[0.0, 10.0],
        x_hinge=0.75,
        h_over_c=[0.2, 0.5],
        workers=1,
    )
    with pytest.raises(ValueError, match="h_over_c is required"):
        table_g.at(1e6, 0.0)

    with pytest.raises(ValueError, match="outside the table range"):
        table_g.at(1e6, 0.0, h_over_c=0.1)
    with pytest.raises(ValueError, match="outside the table range"):
        table_g.at(1e6, 0.0, h_over_c=0.8)


def test_polar_table_npz_roundtrip_and_backward_compatibility(tmp_path):
    table_free = _std_table(workers=1)
    path_free = tmp_path / "free.npz"
    table_free.to_npz(path_free)
    loaded_free = PolarTable.from_npz(path_free)
    assert loaded_free.h_over_c is None
    assert loaded_free.Cl.ndim == 3
    assert loaded_free.Cd.ndim == 3
    assert loaded_free.Cm.ndim == 3
    assert loaded_free.at(1e6, 0.0).Cl(0.0) == pytest.approx(
        table_free.at(1e6, 0.0).Cl(0.0)
    )

    source_g = _AnalyticGroundSource()
    table_g = build_polar_table(
        source_g,
        "naca 2412",
        [5e5, 1e6],
        [-2, 0, 2],
        flap_deg=[0.0, 10.0],
        x_hinge=0.75,
        h_over_c=[0.2, 0.5, 1.0],
        workers=1,
    )
    path_g = tmp_path / "ground.npz"
    table_g.to_npz(path_g)
    loaded_g = PolarTable.from_npz(path_g)
    assert loaded_g.h_over_c is not None
    assert np.allclose(loaded_g.h_over_c, [0.2, 0.5, 1.0])
    assert loaded_g.Cl.shape == (2, 2, 3, 3)
    assert np.allclose(loaded_g.Cl, table_g.Cl)
    assert np.allclose(loaded_g.Cd, table_g.Cd)
    assert np.allclose(loaded_g.Cm, table_g.Cm)
    assert loaded_g.at(5e5, 0.0, h_over_c=0.3).Cl(0.0) == pytest.approx(
        table_g.at(5e5, 0.0, h_over_c=0.3).Cl(0.0)
    )


def test_table_rejects_bad_ground_axes():
    good = {
        "Re": np.array([5e5, 1e6]),
        "flap_deg": np.array([0.0, 10.0]),
        "alpha_deg": np.array([-2.0, 0.0, 2.0]),
    }
    zeros_4d = np.zeros((2, 2, 2, 3))
    with pytest.raises(ValueError, match="ascending"):
        PolarTable(
            Re=good["Re"],
            flap_deg=good["flap_deg"],
            h_over_c=np.array([0.5, 0.2]),
            alpha_deg=good["alpha_deg"],
            Cl=zeros_4d,
            Cd=zeros_4d,
            Cm=zeros_4d,
        )
    with pytest.raises(ValueError, match="positive"):
        PolarTable(
            Re=good["Re"],
            flap_deg=good["flap_deg"],
            h_over_c=np.array([0.0, 0.5]),
            alpha_deg=good["alpha_deg"],
            Cl=zeros_4d,
            Cd=zeros_4d,
            Cm=zeros_4d,
        )
    with pytest.raises(ValueError, match="shape"):
        PolarTable(
            Re=good["Re"],
            flap_deg=good["flap_deg"],
            h_over_c=np.array([0.2, 0.5]),
            alpha_deg=good["alpha_deg"],
            Cl=np.zeros((2, 2, 3)),
            Cd=np.zeros((2, 2, 3)),
            Cm=np.zeros((2, 2, 3)),
        )

