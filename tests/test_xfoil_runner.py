# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the XFOIL runner upgrade, the polar cache and the polar source.

Unit tests replace ``subprocess.run`` with a fake that records the command
stream and writes a canned polar file, so they need no XFOIL binary.
Integration tests use the real XFOIL and skip when no executable is found.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from ventorum.aero import xfoil_runner
from ventorum.core.datatypes import TabulatedAirfoil


_FAKE_POLAR_HEAD = """\
 XFOIL         Version 6.99

 Calculated polar for: FAKE

 Mach =   0.000     Re =     1.000 e 6     Ncrit =   9.000

   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr
  ------ -------- --------- --------- -------- -------- --------
"""


# Plotting-options output recorded from XFOIL 6.99 (black box): the menu
# shows the graphics flag when it opens (T) and after the G toggle (F).
_FAKE_PLOP_OUTPUT = """\
 XFOIL   c>   ...............................................

  G raphics-enable flag:        T
  S ize of plot object          10.00"

      Option, Value   (or <Return>)    c>   ...............................................

  G raphics-enable flag:        F
  S ize of plot object          10.00"

      Option, Value   (or <Return>)    c>
 XFOIL   c>
"""


def _fake_rows_text(rows: list[tuple[float, float, float, float]]) -> str:
    lines = [_FAKE_POLAR_HEAD]
    for alpha, cl, cd, cm in rows:
        lines.append(
            f"  {alpha:7.3f}   {cl:7.4f}   {cd:7.5f}   0.00100  {cm:7.4f}   0.9000   0.9000"
        )
    return "\n".join(lines) + "\n"


def _polar_path_from_stream(stream: str) -> str | None:
    lines = stream.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "pacc" and i + 1 < len(lines):
            candidate = lines[i + 1].strip()
            if candidate:
                return candidate
    return None


def _make_fake_run(rows=None, record=None):
    """Build a fake ``subprocess.run``.

    ``rows`` is a fixed row list or a function of the command stream that
    returns the row list. The fake writes the canned polar to the file named
    in the ``pacc`` block of the stream.
    """

    def fake_run(argv, **kwargs):
        stream = kwargs.get("input", "")
        if record is not None:
            record.append(
                {"argv": argv, "input": stream, "timeout": kwargs.get("timeout")}
            )
        data = rows(stream) if callable(rows) else rows
        path = _polar_path_from_stream(stream)
        if path is not None and data is not None:
            with open(path, "w") as handle:
                handle.write(_fake_rows_text(data))
        return subprocess.CompletedProcess(
            argv, 0, stdout=_FAKE_PLOP_OUTPUT + "fake xfoil", stderr=""
        )

    return fake_run


def _aseq_rows(stream: str) -> list[tuple[float, float, float, float]]:
    """Canned rows that follow the ``aseq`` line of the stream (inclusive ends)."""
    out: list[tuple[float, float, float, float]] = []
    for line in stream.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0] == "aseq":
            start, end, step = float(parts[1]), float(parts[2]), float(parts[3])
            alpha = start
            while (step > 0 and alpha <= end + 1e-9) or (
                step < 0 and alpha >= end - 1e-9
            ):
                out.append((alpha, 0.1 * alpha + 0.5, 0.01, -0.05))
                alpha += step
    return out


def _stream_lines(record: list[dict], index: int = 0) -> list[str]:
    return record[index]["input"].splitlines()


def _assert_subsequence(lines: list[str], wanted: list[str]) -> None:
    """Assert that ``wanted`` appears in ``lines`` in order (not always adjacent)."""
    pos = -1
    for item in wanted:
        try:
            pos = lines.index(item, pos + 1)
        except ValueError:
            raise AssertionError(f"{item!r} not found in order in {lines!r}") from None


# ── Command stream ───────────────────────────────────────────────────────────


def test_command_stream_has_ncrit_xtr_and_panel_count(monkeypatch):
    record: list[dict] = []
    rows = [(0.0, 0.0, 0.005, 0.0), (2.0, 0.22, 0.0056, -0.001)]
    monkeypatch.setattr(
        xfoil_runner.subprocess, "run", _make_fake_run(rows, record)
    )
    polar = xfoil_runner.run_xfoil(
        "naca 0012",
        Re=1e6,
        alpha_start=0.0,
        alpha_end=2.0,
        alpha_step=2.0,
        ncrit=7.5,
        xtr_top=0.3,
        xtr_bottom=0.6,
        n_panels=200,
        sweep="linear",
        timeout=60.0,
    )
    assert len(record) == 1
    lines = _stream_lines(record)
    _assert_subsequence(lines, ["ppar", "n 200", "", ""])
    _assert_subsequence(lines, ["vpar", "n 7.5", "xtr 0.3 0.6", ""])
    assert lines.index("ppar") < lines.index("pane")
    assert lines.index("vpar") > lines.index("oper")
    assert polar.name == "NACA 0012"
    assert np.isclose(polar.Re, 1e6)


def test_command_stream_has_coordinates_and_flap(monkeypatch, tmp_path):
    record: list[dict] = []
    rows = [(0.0, 0.2, 0.005, -0.05), (2.0, 0.4, 0.006, -0.05)]
    monkeypatch.setattr(
        xfoil_runner.subprocess, "run", _make_fake_run(rows, record)
    )
    coords = np.column_stack(
        [np.linspace(1.0, 0.0, 10), np.linspace(0.0, 0.0, 10)]
    )
    xfoil_runner.run_xfoil(
        "custom",
        Re=5e5,
        alpha_start=0.0,
        alpha_end=2.0,
        alpha_step=2.0,
        coordinates=coords,
        flap=(0.75, 0.0, 10.0),
        sweep="linear",
        timeout=60.0,
    )
    assert len(record) == 1
    lines = _stream_lines(record)
    load_lines = [line for line in lines if line.startswith("load ")]
    assert len(load_lines) == 1
    assert load_lines[0].endswith(".dat")
    # XFOIL asks for the hinge y location; 999 makes it ask for y/t.
    flap_at = lines.index("flap")
    assert lines[flap_at:flap_at + 5] == ["flap", "0.75", "999", "0.0", "10.0"]
    assert lines.index("gdes") < lines.index("pane")


def test_flap_hinge_height_is_a_thickness_fraction(monkeypatch):
    """Regression: y_hinge was sent as an absolute y (hinge 0.5 chord above the airfoil)."""
    record: list[dict] = []
    rows = [(0.0, 0.2, 0.005, -0.05)]
    monkeypatch.setattr(
        xfoil_runner.subprocess, "run", _make_fake_run(rows, record)
    )
    xfoil_runner.run_xfoil(
        "naca 2412", Re=1e6, alpha_start=0.0, alpha_end=0.0,
        flap=(0.7, 0.5, -5.0), timeout=60.0,
    )
    lines = _stream_lines(record)
    flap_at = lines.index("flap")
    # Prompts: hinge x, hinge y (999 = give y/t), hinge y/t, deflection [deg].
    assert lines[flap_at + 1:flap_at + 5] == ["0.7", "999", "0.5", "-5.0"]
    # Then the GDES menu is left before the paneling.
    assert lines[flap_at + 5] == ""
    assert lines.index("pane") > flap_at + 5


def test_check_flap_accepts_only_the_common_tuple():
    from ventorum.aero.xfoil_runner import check_flap

    assert check_flap(None) is None
    assert check_flap((0.75, 0.5, 10)) == (0.75, 0.5, 10.0)
    assert check_flap(np.array([0.8, 0.0, -3.0])) == (0.8, 0.0, -3.0)
    with pytest.raises(ValueError, match="3 numbers"):
        check_flap((0.75, 10.0))
    with pytest.raises(ValueError, match="thickness"):
        check_flap((0.75, 1.5, 10.0))
    with pytest.raises(ValueError, match="x_hinge"):
        check_flap((1.0, 0.5, 10.0))
    with pytest.raises(ValueError, match="90"):
        check_flap((0.75, 0.5, 95.0))
    with pytest.raises(ValueError, match="finite"):
        check_flap((0.75, 0.5, float("nan")))
    with pytest.raises(ValueError, match="string"):
        check_flap("0.75 0.5 10")


def test_run_xfoil_refuses_bad_flap_and_multiline_airfoil(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        xfoil_runner.subprocess, "run", lambda *a, **k: calls.append(a)
    )
    with pytest.raises(ValueError, match="3 numbers"):
        xfoil_runner.run_xfoil("naca 0012", Re=1e6, flap=(0.75, 10.0))
    with pytest.raises(ValueError, match="control characters"):
        xfoil_runner.run_xfoil("naca 0012\nquit", Re=1e6)
    assert calls == []


def test_run_xfoil_default_sweep_is_linear(monkeypatch):
    """Owner decision: run_xfoil keeps the old one-sweep behaviour by default."""
    record: list[dict] = []
    monkeypatch.setattr(
        xfoil_runner.subprocess, "run", _make_fake_run(_aseq_rows, record)
    )
    xfoil_runner.run_xfoil(
        "naca 0012", Re=1e6, alpha_start=-2.0, alpha_end=4.0, alpha_step=2.0,
        timeout=60.0,
    )
    assert len(record) == 1
    assert "aseq -2.0 4.0 2.0" in record[0]["input"]


def test_selig_writer_writes_name_and_points(tmp_path):
    from ventorum.aero.xfoil_runner import _write_selig_file

    coords = np.array([[1.0, 0.0], [0.5, 0.05], [0.0, 0.0], [0.5, -0.05]])
    path = tmp_path / "foil.dat"
    _write_selig_file(path, "WriterTest", coords)
    text = path.read_text().splitlines()
    assert text[0] == "WriterTest"
    assert len(text) == 5
    back = np.array([[float(v) for v in line.split()] for line in text[1:]])
    np.testing.assert_allclose(back, coords, atol=1e-9)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        _write_selig_file(tmp_path / "bad.dat", "Bad", np.zeros(4))
    with pytest.raises(ValueError, match="at least 2 points"):
        _write_selig_file(tmp_path / "short.dat", "Short", np.zeros((1, 2)))


# ── Executable resolution ────────────────────────────────────────────────────


def test_executable_resolution_order(monkeypatch):
    from ventorum.aero.xfoil_runner import resolve_xfoil_executable

    assert resolve_xfoil_executable("my-xfoil") == "my-xfoil"
    monkeypatch.setenv("VENTORUM_XFOIL", "/env/xfoil")
    assert resolve_xfoil_executable() == "/env/xfoil"
    assert resolve_xfoil_executable("my-xfoil") == "my-xfoil"
    monkeypatch.delenv("VENTORUM_XFOIL")
    # Then the program that shutil.which finds on PATH, else the bare name.
    monkeypatch.setattr(
        xfoil_runner.shutil, "which",
        lambda name: "found-by-which" if name == "xfoil" else None,
    )
    assert resolve_xfoil_executable() == "found-by-which"
    monkeypatch.setattr(xfoil_runner.shutil, "which", lambda name: None)
    assert resolve_xfoil_executable() == "xfoil"


def test_run_uses_resolved_executable(monkeypatch):
    record: list[dict] = []
    rows = [(0.0, 0.0, 0.005, 0.0)]
    monkeypatch.setattr(
        xfoil_runner.subprocess, "run", _make_fake_run(rows, record)
    )
    monkeypatch.setenv("VENTORUM_XFOIL", "/env/xfoil")
    xfoil_runner.run_xfoil(
        "naca 0012", Re=1e6, alpha_start=0.0, alpha_end=0.0, sweep="linear"
    )
    assert record[0]["argv"][0] == "/env/xfoil"


def test_xfoil_version_reads_banner_and_handles_missing(monkeypatch):
    from ventorum.aero.xfoil_runner import xfoil_version

    def fake_run(argv, **kwargs):
        # Graphics off first, then quit.
        assert kwargs.get("input") == "PLOP\nG\n\nquit\n"
        return subprocess.CompletedProcess(
            argv, 0, stdout="\n XFOIL Version 6.99\n more\n" + _FAKE_PLOP_OUTPUT,
            stderr="",
        )

    monkeypatch.setattr(xfoil_runner.subprocess, "run", fake_run)
    xfoil_version.cache_clear()
    assert xfoil_version("fake-exe-banner") == "XFOIL Version 6.99"

    def fake_plain(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="no banner", stderr="")

    monkeypatch.setattr(xfoil_runner.subprocess, "run", fake_plain)
    xfoil_version.cache_clear()
    assert xfoil_version("fake-exe-plain") == "unknown"

    def fake_missing(argv, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(xfoil_runner.subprocess, "run", fake_missing)
    xfoil_version.cache_clear()
    assert xfoil_version("fake-exe-missing") == "unknown"


# ── Two-direction sweep ──────────────────────────────────────────────────────


def test_from_zero_sweep_merges_and_sorts(monkeypatch):
    import warnings

    record: list[dict] = []
    monkeypatch.setattr(
        xfoil_runner.subprocess, "run", _make_fake_run(_aseq_rows, record)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        polar = xfoil_runner.run_xfoil(
            "naca 0012",
            Re=1e6,
            alpha_start=-2.0,
            alpha_end=4.0,
            alpha_step=2.0,
            sweep="from_zero",
            timeout=60.0,
        )
    assert len(record) == 2
    assert all(entry["timeout"] == 60.0 for entry in record)
    got_deg = np.degrees(np.asarray(polar.alpha))
    np.testing.assert_allclose(got_deg, [-2.0, 0.0, 2.0, 4.0])
    np.testing.assert_allclose(
        np.asarray(polar.Cl_data), 0.1 * np.array([-2.0, 0.0, 2.0, 4.0]) + 0.5
    )
    assert np.isclose(polar.Re, 1e6)


def test_sweep_keyword_rejects_unknown_value():
    with pytest.raises(ValueError, match="sweep"):
        xfoil_runner.run_xfoil("naca 0012", Re=1e6, sweep="sideways")
    with pytest.raises(ValueError, match="alpha_step"):
        xfoil_runner.run_xfoil("naca 0012", Re=1e6, alpha_step=0.0)


def test_linear_reverse_sweep_uses_negative_step(monkeypatch):
    record: list[dict] = []
    monkeypatch.setattr(
        xfoil_runner.subprocess, "run", _make_fake_run(_aseq_rows, record)
    )
    polar = xfoil_runner.run_xfoil(
        "naca 0012",
        Re=1e6,
        alpha_start=4.0,
        alpha_end=0.0,
        alpha_step=1.0,
        sweep="linear",
        timeout=60.0,
    )
    assert len(record) == 1
    assert "aseq 4.0 0.0 -1.0" in record[0]["input"]
    np.testing.assert_allclose(
        np.degrees(np.asarray(polar.alpha)), [0.0, 1.0, 2.0, 3.0, 4.0]
    )


def test_from_zero_with_zero_start_runs_one_leg(monkeypatch):
    record: list[dict] = []
    monkeypatch.setattr(
        xfoil_runner.subprocess, "run", _make_fake_run(_aseq_rows, record)
    )
    polar = xfoil_runner.run_xfoil(
        "naca 0012",
        Re=1e6,
        alpha_start=0.0,
        alpha_end=4.0,
        alpha_step=2.0,
        sweep="from_zero",
        timeout=60.0,
    )
    assert len(record) == 1
    np.testing.assert_allclose(
        np.degrees(np.asarray(polar.alpha)), [0.0, 2.0, 4.0]
    )


def test_from_zero_single_point_runs_alpha_zero(monkeypatch):
    record: list[dict] = []
    monkeypatch.setattr(
        xfoil_runner.subprocess, "run", _make_fake_run(_aseq_rows, record)
    )
    polar = xfoil_runner.run_xfoil(
        "naca 0012", Re=1e6, alpha_start=0.0, alpha_end=0.0, sweep="from_zero"
    )
    assert len(record) == 1
    assert len(polar.alpha) == 1
    assert np.degrees(float(polar.alpha[0])) == pytest.approx(0.0)


def test_failed_runs_raise_with_xfoil_output(monkeypatch):
    def fake_empty(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="fake fail", stderr="")

    monkeypatch.setattr(xfoil_runner.subprocess, "run", fake_empty)
    with pytest.raises(RuntimeError, match="did not generate a polar file"):
        xfoil_runner.run_xfoil("naca 0012", Re=1e6, sweep="linear")

    def fake_header_only(argv, **kwargs):
        path = _polar_path_from_stream(kwargs.get("input", ""))
        with open(path, "w") as handle:
            handle.write(_FAKE_POLAR_HEAD)
        return subprocess.CompletedProcess(argv, 0, stdout="fake fail", stderr="")

    monkeypatch.setattr(xfoil_runner.subprocess, "run", fake_header_only)
    with pytest.raises(RuntimeError, match="did not generate a polar file"):
        xfoil_runner.run_xfoil("naca 0012", Re=1e6, sweep="linear")


def test_filtered_empty_merge_raises(monkeypatch):
    def fake_zero_only(stream):
        return [(0.0, 0.5, 0.01, -0.05)]

    monkeypatch.setattr(
        xfoil_runner.subprocess, "run", _make_fake_run(fake_zero_only)
    )
    with pytest.raises(RuntimeError, match="empty polar file"):
        xfoil_runner.run_xfoil(
            "naca 0012",
            Re=1e6,
            alpha_start=2.0,
            alpha_end=4.0,
            alpha_step=1.0,
            sweep="linear",
        )


def test_subprocess_errors_are_reported(monkeypatch):
    def fake_called_process(argv, **kwargs):
        raise subprocess.CalledProcessError(3, argv, output="out", stderr="err")

    monkeypatch.setattr(xfoil_runner.subprocess, "run", fake_called_process)
    with pytest.raises(RuntimeError, match="exit code 3"):
        xfoil_runner.run_xfoil("naca 0012", Re=1e6, sweep="linear")

    def fake_missing(argv, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(xfoil_runner.subprocess, "run", fake_missing)
    with pytest.raises(FileNotFoundError, match="not found"):
        xfoil_runner.run_xfoil("naca 0012", Re=1e6, sweep="linear")


def test_merge_averages_shared_points_and_keeps_missing_moment():
    from ventorum.aero.xfoil_runner import _merge_runs

    def polar_at(alpha_deg, cl, with_cm=True):
        return TabulatedAirfoil(
            name="leg",
            alpha=np.radians(np.asarray(alpha_deg, dtype=float)),
            Cl_data=np.asarray(cl, dtype=float),
            Cd_data=np.full(len(alpha_deg), 0.01),
            Cm_data=None
            if not with_cm
            else np.full(len(alpha_deg), -0.05),
            Re=1e6,
        )

    merged = _merge_runs(
        [
            polar_at([0.0, 2.0], [0.50, 0.70]),
            polar_at([0.0, -2.0], [0.54, 0.30], with_cm=False),
        ],
        "Merged",
        -2.0,
        2.0,
    )
    np.testing.assert_allclose(np.degrees(np.asarray(merged.alpha)), [-2.0, 0.0, 2.0])
    np.testing.assert_allclose(np.asarray(merged.Cl_data), [0.30, 0.52, 0.70])
    assert merged.Cm_data is None
    assert merged.name == "Merged"


# ── Batch ────────────────────────────────────────────────────────────────────


def test_batch_returns_in_order_and_keeps_exceptions(monkeypatch):
    import time

    def fake_run_xfoil(**kwargs):
        if kwargs.get("Re") == -1.0:
            raise RuntimeError("boom")
        time.sleep(0.05 if kwargs.get("airfoil") == "slow" else 0.0)
        return kwargs.get("airfoil")

    monkeypatch.setattr(xfoil_runner, "run_xfoil", fake_run_xfoil)
    cases = [
        {"airfoil": "slow", "Re": 1e6},
        {"airfoil": "bad", "Re": -1.0},
        {"airfoil": "fast", "Re": 2e6},
    ]
    out = xfoil_runner.run_xfoil_batch(cases, workers=2)
    assert out[0] == "slow"
    assert isinstance(out[1], RuntimeError)
    assert out[2] == "fast"


def test_batch_rejects_bad_worker_count():
    with pytest.raises(ValueError, match="workers"):
        xfoil_runner.run_xfoil_batch([{"airfoil": "naca 0012", "Re": 1e6}], workers=0)


def test_batch_default_workers_runs_all_cases(monkeypatch):
    monkeypatch.setattr(
        xfoil_runner, "run_xfoil", lambda **kwargs: kwargs.get("airfoil")
    )
    out = xfoil_runner.run_xfoil_batch(
        [{"airfoil": "a", "Re": 1e6}, {"airfoil": "b", "Re": 1e6}], workers=None
    )
    assert out == ["a", "b"]


# ── Cache ────────────────────────────────────────────────────────────────────


def _flat_polar() -> TabulatedAirfoil:
    return TabulatedAirfoil(
        name="cached",
        alpha=np.radians(np.array([0.0, 2.0, 4.0])),
        Cl_data=np.array([0.2, 0.4, 0.6]),
        Cd_data=np.array([0.01, 0.01, 0.012]),
        Cm_data=np.array([-0.05, -0.05, -0.06]),
        Re=1e6,
    )


def test_polar_cache_roundtrip_and_clear(tmp_path):
    from ventorum.aero.polar_cache import PolarCache

    cache = PolarCache(directory=tmp_path / "polars")
    payload = {"airfoil": "naca 0012", "Re": 1e6}
    assert cache.get(payload) is None
    path = cache.put(payload, _flat_polar())
    assert path.exists()
    back = cache.get(payload)
    assert back is not None
    assert back.name == "cached"
    np.testing.assert_allclose(back.alpha, _flat_polar().alpha)
    np.testing.assert_allclose(back.Cl_data, _flat_polar().Cl_data)
    np.testing.assert_allclose(back.Cd_data, _flat_polar().Cd_data)
    np.testing.assert_allclose(back.Cm_data, _flat_polar().Cm_data)
    assert back.Re == 1e6
    assert cache.clear() == 1
    assert cache.clear() == 0
    assert cache.get(payload) is None


def test_polar_cache_roundtrip_without_moment(tmp_path):
    from ventorum.aero.polar_cache import PolarCache

    cache = PolarCache(directory=tmp_path / "polars")
    polar = _flat_polar()
    polar.Cm_data = None
    payload = {"airfoil": "naca 0012", "Re": 1e6}
    cache.put(payload, polar)
    back = cache.get(payload)
    assert back is not None
    assert back.Cm_data is None
    assert np.isclose(back.Cm(0.0), 0.0)


def test_polar_cache_ignores_corrupt_file(tmp_path):
    from ventorum.aero.polar_cache import PolarCache

    cache = PolarCache(directory=tmp_path / "polars")
    payload = {"airfoil": "naca 0012", "Re": 1e6}
    cache.put(payload, _flat_polar())
    cache.path_for(payload).write_text("not json {")
    assert cache.get(payload) is None
    cache.path_for(payload).write_text("{}")
    assert cache.get(payload) is None


def test_polar_cache_default_directory_uses_env(monkeypatch, tmp_path):
    from ventorum.aero.polar_cache import PolarCache

    monkeypatch.setenv("VENTORUM_CACHE", str(tmp_path / "v cache"))
    cache = PolarCache()
    assert str(cache.directory).endswith("polars")
    cache.put({"a": 1}, _flat_polar())
    assert cache.get({"a": 1}) is not None


# ── Polar source ─────────────────────────────────────────────────────────────


def _source_with_fake_run(monkeypatch, tmp_path, seen):
    from ventorum.aero import polar_source

    def fake_run_xfoil(airfoil, Re, alpha_start, alpha_end, alpha_step, **kwargs):
        seen.append(
            {
                "airfoil": airfoil,
                "Re": Re,
                "alpha_start": alpha_start,
                "alpha_end": alpha_end,
                "alpha_step": alpha_step,
                "kwargs": kwargs,
            }
        )
        return _flat_polar()

    monkeypatch.setattr(polar_source, "run_xfoil", fake_run_xfoil)
    monkeypatch.setattr(
        polar_source, "xfoil_version", lambda executable=None: "XFOIL Version 6.99"
    )
    return polar_source


def test_polar_source_caches_and_misses_on_changed_input(monkeypatch, tmp_path):
    from ventorum.aero.polar_cache import PolarCache

    seen: list[dict] = []
    polar_source = _source_with_fake_run(monkeypatch, tmp_path, seen)
    src = polar_source.XfoilPolarSource(
        cache=PolarCache(directory=tmp_path / "polars")
    )
    first = src.polar("naca 0012", 1e6, [-2.0, 0.0, 2.0, 4.0])
    second = src.polar("naca 0012", 1e6, [-2.0, 0.0, 2.0, 4.0])
    assert len(seen) == 1
    np.testing.assert_allclose(first.Cl_data, second.Cl_data)
    assert seen[0]["alpha_start"] == pytest.approx(-2.0)
    assert seen[0]["alpha_end"] == pytest.approx(4.0)
    assert seen[0]["alpha_step"] == pytest.approx(2.0)
    src.polar("naca 0012", 2e6, [-2.0, 0.0, 2.0, 4.0])
    assert len(seen) == 2
    src.polar("NACA 0012", 1e6, [-2.0, 0.0, 2.0, 4.0])
    assert len(seen) == 2


_BASE_CALL = {
    "section": "naca 0012",
    "Re": 1e6,
    "alphas_deg": [-2.0, 0.0, 2.0, 4.0],
    "kwargs": {},
}

# One changed input per case. Each must miss the cache.
_CHANGED_INPUTS = {
    "section_name": {"section": "naca 2412"},
    "section_coordinates": {
        "section": np.column_stack([np.linspace(1.0, 0.0, 6), np.zeros(6)])
    },
    "Re": {"Re": 2e6},
    "alphas": {"alphas_deg": [-2.0, 0.0, 2.0, 6.0]},
    "alpha_step": {"alphas_deg": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]},
    "mach": {"kwargs": {"mach": 0.1}},
    "flap": {"kwargs": {"flap": (0.75, 0.5, 10.0)}},
    "flap_hinge_height": {"kwargs": {"flap": (0.75, 0.2, 10.0)}},
    "ncrit": {"kwargs": {"ncrit": 7.0}},
    "xtr_top": {"kwargs": {"xtr_top": 0.3}},
    "xtr_bottom": {"kwargs": {"xtr_bottom": 0.3}},
    "n_panels": {"kwargs": {"n_panels": 200}},
    "sweep": {"kwargs": {"sweep": "linear"}},
    "max_iter": {"kwargs": {"max_iter": 40}},
    "timeout": {"kwargs": {"timeout": 30.0}},
}


@pytest.mark.parametrize("case", sorted(_CHANGED_INPUTS))
def test_polar_source_cache_misses_when_any_input_changes(monkeypatch, tmp_path, case):
    """Regression: the key missed max_iter and timeout; every input must count."""
    from ventorum.aero.polar_cache import PolarCache

    seen: list[dict] = []
    polar_source = _source_with_fake_run(monkeypatch, tmp_path, seen)
    src = polar_source.XfoilPolarSource(
        cache=PolarCache(directory=tmp_path / "polars")
    )

    def call(spec):
        src.polar(spec["section"], spec["Re"], spec["alphas_deg"], **spec["kwargs"])

    call(_BASE_CALL)
    call(_BASE_CALL)
    assert len(seen) == 1
    changed = {**_BASE_CALL, **_CHANGED_INPUTS[case]}
    call(changed)
    assert len(seen) == 2, case
    call(changed)
    assert len(seen) == 2, case


def test_polar_source_cache_key_uses_the_version_of_each_executable(
    monkeypatch, tmp_path
):
    """Regression: the version was read once per source and reused for another executable."""
    from ventorum.aero import polar_source
    from ventorum.aero.polar_cache import PolarCache

    runs: list[str | None] = []
    versions = {"xfoil-a": "XFOIL Version 6.99", "xfoil-b": "XFOIL Version 6.97"}
    version_calls: list[str] = []

    def fake_version(executable=None):
        version_calls.append(executable)
        return versions.get(executable, "unknown")

    def fake_run_xfoil(airfoil, Re, alpha_start, alpha_end, alpha_step, **kwargs):
        runs.append(kwargs.get("executable"))
        return _flat_polar()

    monkeypatch.setattr(polar_source, "run_xfoil", fake_run_xfoil)
    monkeypatch.setattr(polar_source, "xfoil_version", fake_version)
    cache = PolarCache(directory=tmp_path / "polars")
    src = polar_source.XfoilPolarSource(cache=cache)
    src.polar("naca 0012", 1e6, [0.0, 2.0], executable="xfoil-a")
    src.polar("naca 0012", 1e6, [0.0, 2.0], executable="xfoil-a")
    assert runs == ["xfoil-a"]
    # Another executable with another version must not read the first polar.
    src.polar("naca 0012", 1e6, [0.0, 2.0], executable="xfoil-b")
    assert runs == ["xfoil-a", "xfoil-b"]
    assert version_calls[-1] == "xfoil-b"
    # The constructor default executable follows the same rule.
    other = polar_source.XfoilPolarSource(cache=cache, executable="xfoil-b")
    other.polar("naca 0012", 1e6, [0.0, 2.0])
    assert runs == ["xfoil-a", "xfoil-b"]
    # An executable without a known version never uses the cache.
    src.polar("naca 0012", 1e6, [0.0, 2.0], executable="xfoil-c")
    src.polar("naca 0012", 1e6, [0.0, 2.0], executable="xfoil-c")
    assert runs == ["xfoil-a", "xfoil-b", "xfoil-c", "xfoil-c"]


def test_polar_source_defaults_and_describe(monkeypatch, tmp_path):
    seen: list[dict] = []
    polar_source = _source_with_fake_run(monkeypatch, tmp_path, seen)
    src = polar_source.XfoilPolarSource(cache=False, ncrit=7.0, n_panels=200)
    src.polar("naca 0012", 1e6, [0.0, 2.0], flap=(0.75, 0.5, 10.0))
    kwargs = seen[0]["kwargs"]
    # The source sweeps from zero by default; run_xfoil itself is linear.
    assert kwargs["sweep"] == "from_zero"
    assert kwargs["ncrit"] == 7.0 and kwargs["n_panels"] == 200
    assert kwargs["flap"] == (0.75, 0.5, 10.0)
    info = src.describe()
    assert info["source"] == "XfoilPolarSource"
    assert info["version"] == "XFOIL Version 6.99"
    opts = info["options"]
    assert opts["ncrit"] == 7.0 and opts["n_panels"] == 200
    assert opts["sweep"] == "from_zero"
    assert {"xtr_top", "xtr_bottom", "max_iter", "timeout"} <= set(opts)
    assert "executable" not in opts
    with pytest.raises(ValueError, match="Unknown XFOIL option"):
        polar_source.XfoilPolarSource(cache=False, bogus=1)
    with pytest.raises(ValueError, match="Unknown XFOIL option"):
        src.polar("naca 0012", 1e6, [0.0], bogus=1)
    with pytest.raises(ValueError, match="3 numbers"):
        src.polar("naca 0012", 1e6, [0.0], flap=(0.75, 10.0))


def test_polar_source_without_cache_runs_every_time(monkeypatch, tmp_path):
    seen: list[dict] = []
    polar_source = _source_with_fake_run(monkeypatch, tmp_path, seen)
    src = polar_source.XfoilPolarSource(cache=False)
    src.polar("naca 0012", 1e6, [0.0, 2.0])
    src.polar("naca 0012", 1e6, [0.0, 2.0])
    assert len(seen) == 2


def test_polar_source_passes_coordinates_and_rejects_conflicts(
    monkeypatch, tmp_path
):
    seen: list[dict] = []
    polar_source = _source_with_fake_run(monkeypatch, tmp_path, seen)
    src = polar_source.XfoilPolarSource(cache=False)
    coords = np.column_stack(
        [np.linspace(1.0, 0.0, 8), np.zeros(8)]
    )
    src.polar(coords, 1e6, [3.0])
    assert seen[0]["kwargs"]["coordinates"] is not None
    np.testing.assert_allclose(
        np.asarray(seen[0]["kwargs"]["coordinates"]), coords
    )
    assert seen[0]["alpha_start"] == pytest.approx(3.0)
    assert seen[0]["alpha_end"] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="coordinates"):
        src.polar("naca 0012", 1e6, [0.0, 2.0], coordinates=coords)


def test_polar_source_follows_protocol():
    from ventorum.aero.polar_source import SectionPolarSource, XfoilPolarSource

    assert isinstance(XfoilPolarSource(cache=False), SectionPolarSource)
    with pytest.raises(TypeError, match="cache"):
        XfoilPolarSource(cache="folder")


def test_polar_source_default_cache_is_used(monkeypatch, tmp_path):
    from ventorum.aero import polar_source

    class FakeCache(polar_source.PolarCache):
        def __init__(self, directory=None):
            super().__init__(directory=tmp_path / "default")

    monkeypatch.setattr(polar_source, "PolarCache", FakeCache)
    monkeypatch.setattr(
        polar_source, "run_xfoil", lambda *args, **kwargs: _flat_polar()
    )
    monkeypatch.setattr(
        polar_source, "xfoil_version", lambda executable=None: "XFOIL Version 6.99"
    )
    src = polar_source.XfoilPolarSource()
    assert isinstance(src._cache, FakeCache)
    first = src.polar("naca 0012", 1e6, [0.0, 2.0])
    second = src.polar("naca 0012", 1e6, [0.0, 2.0])
    np.testing.assert_allclose(first.Cl_data, second.Cl_data)


def test_polar_source_rejects_bad_inputs(monkeypatch, tmp_path):
    seen: list[dict] = []
    polar_source = _source_with_fake_run(monkeypatch, tmp_path, seen)
    src = polar_source.XfoilPolarSource(cache=False)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        src.polar(np.zeros(4), 1e6, [0.0, 2.0])
    with pytest.raises(ValueError, match="at least one angle"):
        src.polar("naca 0012", 1e6, [])
    assert seen == []


# ── Integration with the real XFOIL ──────────────────────────────────────────

_XFOIL_EXE = shutil.which(
    os.environ.get("VENTORUM_XFOIL", "xfoil")
)
needs_xfoil = pytest.mark.skipif(
    _XFOIL_EXE is None, reason="XFOIL executable not found"
)


def _naca0012_coordinates(n: int = 120) -> np.ndarray:
    """NACA 0012 points in Selig order with a closed trailing edge.

    Formula: the 4-digit thickness equation of Abbott and Von Doenhoff
    (Theory of Wing Sections, 1959, section on NACA 4-digit airfoils).
    """
    beta = np.linspace(0.0, 1.0, n)
    x = 0.5 * (1.0 - np.cos(np.pi * beta))
    yt = (
        5.0
        * 0.12
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
    )
    upper = np.column_stack([x[::-1], yt[::-1]])
    lower = np.column_stack([x[1:], -yt[1:]])
    closed = np.vstack([upper, lower, upper[:1]])
    return closed


@needs_xfoil
def test_naca0012_polar_shape():
    polar = xfoil_runner.run_xfoil(
        "naca 0012",
        Re=1e6,
        alpha_start=-2.0,
        alpha_end=4.0,
        alpha_step=1.0,
        sweep="from_zero",
        timeout=300.0,
    )
    assert abs(float(polar.Cl(0.0))) < 2e-3
    cls = np.asarray(polar.Cl(np.radians(np.arange(-2.0, 5.0, 1.0))))
    assert np.all(np.diff(cls) > 0.0)


@needs_xfoil
def test_flap_increases_lift():
    common = dict(Re=1e6, alpha_start=0.0, alpha_end=4.0, alpha_step=1.0,
                  n_panels=220, timeout=300.0)
    base = xfoil_runner.run_xfoil("naca 2412", **common)
    flapped = xfoil_runner.run_xfoil(
        "naca 2412", flap=(0.75, 0.0, 10.0), **common
    )
    assert float(flapped.Cl(0.0)) > float(base.Cl(0.0))


@needs_xfoil
def test_coordinates_match_builtin_naca():
    common = dict(Re=1e6, alpha_start=0.0, alpha_end=4.0, alpha_step=2.0,
                  timeout=300.0)
    builtin = xfoil_runner.run_xfoil("naca 0012", **common)
    from_coords = xfoil_runner.run_xfoil(
        "naca 0012 from coordinates",
        coordinates=_naca0012_coordinates(),
        **common,
    )
    assert (
        abs(
            float(from_coords.Cl(np.radians(4.0)))
            - float(builtin.Cl(np.radians(4.0)))
        )
        < 0.02
    )


def test_cache_payload_serialises_to_json(tmp_path):
    from ventorum.aero.polar_cache import PolarCache

    cache = PolarCache(directory=tmp_path / "polars")
    payload = {
        "coordinates": [[round(float(v), 9) for v in row] for row in np.zeros((3, 2))],
        "Re": 1e6,
        "flap": [0.75, 0.0, 10.0],
    }
    json.dumps(payload, sort_keys=True)
    cache.put(payload, _flat_polar())
    assert cache.get(payload) is not None


# ── Headless XFOIL ───────────────────────────────────────────────────────────


def _recording_run(record: list[dict], rows=None):
    """Fake subprocess.run that records every keyword and checks the folder."""

    inner = _make_fake_run(rows if rows is not None else _aseq_rows)

    def fake_run(argv, **kwargs):
        cwd = kwargs.get("cwd")
        record.append(
            {
                "input": kwargs.get("input", ""),
                "kwargs": dict(kwargs),
                "cwd_exists": cwd is not None and os.path.isdir(cwd),
                "cwd_files": sorted(os.listdir(cwd)) if cwd and os.path.isdir(cwd) else None,
            }
        )
        return inner(argv, **kwargs)

    return fake_run


def _check_headless(entry: dict) -> None:
    assert entry["input"].startswith("PLOP\nG\n\n"), entry["input"][:40]
    # Nothing comes before the header: the first command is PLOP.
    assert entry["input"].splitlines()[:3] == ["PLOP", "G", ""]
    kwargs = entry["kwargs"]
    if os.name == "nt":
        assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW
    else:
        assert "creationflags" not in kwargs
    # Own new temporary folder, not the folder of the user, empty at start.
    assert entry["cwd_exists"]
    assert os.path.abspath(kwargs["cwd"]) != os.path.abspath(os.getcwd())
    assert entry["cwd_files"] == []
    assert "env" not in kwargs


def test_every_run_path_is_headless(monkeypatch):
    """Regression: XFOIL opened its graphics window on every call."""
    record: list[dict] = []
    monkeypatch.setattr(xfoil_runner.subprocess, "run", _recording_run(record))
    # One linear sweep.
    xfoil_runner.run_xfoil("naca 0012", Re=1e6, alpha_start=0.0, alpha_end=2.0,
                           alpha_step=1.0, timeout=60.0)
    assert len(record) == 1
    # Both starts of the two-direction sweep.
    xfoil_runner.run_xfoil("naca 0012", Re=1e6, alpha_start=-2.0, alpha_end=2.0,
                           alpha_step=1.0, sweep="from_zero", timeout=60.0)
    assert len(record) == 3
    # The batch path.
    out = xfoil_runner.run_xfoil_batch(
        [{"airfoil": "naca 0012", "Re": 1e6, "alpha_start": 0.0, "alpha_end": 1.0},
         {"airfoil": "naca 2412", "Re": 1e6, "alpha_start": 0.0, "alpha_end": 1.0}],
        workers=2,
    )
    assert all(isinstance(item, TabulatedAirfoil) for item in out)
    assert len(record) == 5
    # The version probe.
    xfoil_runner.xfoil_version.cache_clear()
    xfoil_runner.xfoil_version("fake-exe-headless")
    xfoil_runner.xfoil_version.cache_clear()
    assert len(record) == 6
    for entry in record:
        _check_headless(entry)
    # Each start gets its own folder.
    folders = [entry["kwargs"]["cwd"] for entry in record]
    assert len(set(folders)) == len(folders)


def test_polar_source_runs_are_headless(monkeypatch, tmp_path):
    from ventorum.aero.polar_source import XfoilPolarSource

    record: list[dict] = []
    monkeypatch.setattr(xfoil_runner.subprocess, "run", _recording_run(record))
    XfoilPolarSource(cache=False).polar("naca 0012", 1e6, [-1.0, 0.0, 1.0])
    assert len(record) == 2
    for entry in record:
        _check_headless(entry)


def test_graphics_flag_parser_uses_the_last_flag_line():
    from ventorum.aero.xfoil_runner import check_graphics_disabled

    # T when the menu opens, F after the toggle: graphics off.
    assert check_graphics_disabled(_FAKE_PLOP_OUTPUT) is True
    # F then T: the toggle switched the graphics on.
    flipped = _FAKE_PLOP_OUTPUT.replace("flag:        T", "flag:        X")
    flipped = flipped.replace("flag:        F", "flag:        T")
    flipped = flipped.replace("flag:        X", "flag:        F")
    with pytest.raises(RuntimeError, match="graphics could not be disabled"):
        check_graphics_disabled(flipped)
    # No flag line (a build without the plotting menu): no decision.
    assert check_graphics_disabled(" XFOIL   c>\n no menu here\n") is False


def test_run_raises_when_graphics_stay_enabled(monkeypatch):
    flipped = _FAKE_PLOP_OUTPUT.replace("flag:        F", "flag:        T")

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=flipped, stderr="")

    monkeypatch.setattr(xfoil_runner.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="graphics could not be disabled"):
        xfoil_runner.run_xfoil("naca 0012", Re=1e6, alpha_start=0.0, alpha_end=0.0)


def test_missing_graphics_flag_warns_once(monkeypatch):
    rows = [(0.0, 0.0, 0.005, 0.0)]
    inner = _make_fake_run(rows)

    def fake_run(argv, **kwargs):
        inner(argv, **kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout="no plotting menu", stderr="")

    monkeypatch.setattr(xfoil_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(xfoil_runner, "_GRAPHICS_FLAG_WARNED", False)
    with pytest.warns(RuntimeWarning, match="graphics-enable flag"):
        xfoil_runner.run_xfoil("naca 0012", Re=1e6, alpha_start=0.0, alpha_end=0.0)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        polar = xfoil_runner.run_xfoil(
            "naca 0012", Re=1e6, alpha_start=0.0, alpha_end=0.0
        )
    assert len(polar.alpha) == 1


def test_ppar_and_vpar_blank_line_rules(monkeypatch):
    """PPAR needs two blank lines to leave after a change; VPAR needs exactly one."""
    record: list[dict] = []
    rows = [(0.0, 0.0, 0.005, 0.0)]
    monkeypatch.setattr(
        xfoil_runner.subprocess, "run", _make_fake_run(rows, record)
    )
    xfoil_runner.run_xfoil(
        "naca 0012", Re=1e6, alpha_start=0.0, alpha_end=0.0, n_panels=200,
        ncrit=7.5, xtr_top=0.3, xtr_bottom=0.6, max_iter=80, timeout=60.0,
    )
    lines = _stream_lines(record)
    k = lines.index("ppar")
    assert lines[k:k + 5] == ["ppar", "n 200", "", "", "pane"]
    v = lines.index("vpar")
    # One blank line returns to OPER; the next OPER command follows at once.
    assert lines[v:v + 5] == ["vpar", "n 7.5", "xtr 0.3 0.6", "", "iter 80"]
    assert lines[v + 4:v + 7] == ["iter 80", "mach 0.0", "visc 1000000.0"]
