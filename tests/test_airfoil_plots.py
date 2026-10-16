# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the 2D airfoil plots: shape, Cp, ground effect, and examples.

Every plot test checks the content of the axes: the number of lines and one
data value. The data checks use a relative tolerance of 1e-12: the plotted
values come from the same result arrays, so the measured error today is 0,
and the tolerance floor is 1e-12 relative. The example tests run the example
scripts as subprocesses with a temporary output folder and check the PNG
files; each takes several seconds (a new process and the 2D solves), so
they carry the slow marker.
"""

import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from ventorum.airfoil import AirfoilShape
from ventorum.panel2d.solver import solve_inviscid
from ventorum.visualization.airfoil2d import (
    plot_airfoil,
    plot_cp,
    plot_ground_effect_cp,
)

DATA_RTOL = 1e-12
DATA_ATOL = 1e-12

ROOT = Path(__file__).resolve().parents[1]


def _coords(designation="0012", n_points=81):
    """Unit-chord coordinates of a NACA section in Selig order."""
    return AirfoilShape.naca(designation, n_points=n_points).coords


def test_plot_airfoil_returns_axes_with_shape_line():
    coords = _coords()
    ax = plot_airfoil(coords)
    try:
        assert isinstance(ax, Axes)
        assert len(ax.lines) == 1
        np.testing.assert_allclose(
            ax.lines[0].get_xdata(), coords[:, 0], rtol=DATA_RTOL, atol=DATA_ATOL
        )
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), coords[:, 1], rtol=DATA_RTOL, atol=DATA_ATOL
        )
        assert "chord" in ax.get_xlabel()
        assert ax.get_aspect() == 1.0
    finally:
        plt.close(ax.figure)


def test_plot_airfoil_shape_nodes_and_ground():
    shape = AirfoilShape.naca("2412", n_points=81)
    ax = plot_airfoil(shape, show_nodes=True, ground_height=0.0)
    try:
        assert isinstance(ax, Axes)
        # Outline, node markers, ground line.
        assert len(ax.lines) == 3
        ground = ax.lines[2]
        np.testing.assert_allclose(
            ground.get_ydata(), np.full_like(ground.get_ydata(), 0.0),
            rtol=DATA_RTOL, atol=DATA_ATOL,
        )
        assert shape.name in ax.get_title()
    finally:
        plt.close(ax.figure)


def test_plot_cp_returns_axes_with_result_data():
    coords = _coords()
    res = solve_inviscid(coords, 5.0, backend="numpy")
    ax = plot_cp(res)
    try:
        assert isinstance(ax, Axes)
        assert len(ax.lines) == 1
        np.testing.assert_allclose(
            ax.lines[0].get_xdata(), coords[:, 0], rtol=DATA_RTOL, atol=DATA_ATOL
        )
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), np.asarray(res.Cp, dtype=float),
            rtol=DATA_RTOL, atol=DATA_ATOL,
        )
        assert ax.yaxis_inverted()
        assert "Cp" in ax.get_ylabel()
    finally:
        plt.close(ax.figure)


def test_plot_cp_sweep_selects_angle_index():
    coords = _coords()
    res = solve_inviscid(coords, [0.0, 5.0], backend="numpy")
    ax = plot_cp(res, index=1)
    try:
        assert len(ax.lines) == 1
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), np.asarray(res.Cp[1], dtype=float),
            rtol=DATA_RTOL, atol=DATA_ATOL,
        )
        assert "5.00" in ax.get_title()
    finally:
        plt.close(ax.figure)


def test_plot_cp_compare_result_draws_markers():
    coords = _coords()
    res_main = solve_inviscid(coords, 5.0, backend="numpy")
    res_ref = solve_inviscid(coords, 0.0, backend="numpy")
    ax = plot_cp(res_main, compare=res_ref)
    try:
        assert len(ax.lines) == 2
        assert ax.lines[1].get_linestyle() == "None"
        np.testing.assert_allclose(
            ax.lines[1].get_ydata(), np.asarray(res_ref.Cp, dtype=float),
            rtol=DATA_RTOL, atol=DATA_ATOL,
        )
    finally:
        plt.close(ax.figure)


def test_plot_cp_compare_arrays_draws_points():
    coords = _coords()
    res = solve_inviscid(coords, 5.0, backend="numpy")
    x_ref = np.linspace(0.0, 1.0, 11)
    cp_ref = np.full(11, -1.0)
    ax = plot_cp(res, compare=(x_ref, cp_ref))
    try:
        assert len(ax.lines) == 2
        np.testing.assert_allclose(
            ax.lines[1].get_xdata(), x_ref, rtol=DATA_RTOL, atol=DATA_ATOL
        )
        np.testing.assert_allclose(
            ax.lines[1].get_ydata(), cp_ref, rtol=DATA_RTOL, atol=DATA_ATOL
        )
    finally:
        plt.close(ax.figure)


def test_plot_ground_effect_cp_returns_one_line_per_height():
    coords = _coords("4412")
    results = {
        1.0: solve_inviscid(coords, 4.0, height=1.0, backend="numpy"),
        0.3: solve_inviscid(coords, 4.0, height=0.3, backend="numpy"),
    }
    ax = plot_ground_effect_cp(results)
    try:
        assert isinstance(ax, Axes)
        assert len(ax.lines) == 2
        assert ax.yaxis_inverted()
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("1" in lab for lab in labels)
        assert any("0.3" in lab for lab in labels)
    finally:
        plt.close(ax.figure)


def _run_example(name, outdir):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "examples" / name), str(outdir)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, (
        f"{name} failed with code {proc.returncode}.\n"
        f"stdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}"
    )
    return proc


@pytest.mark.slow
def test_example_flap_writes_png_files(tmp_path):
    _run_example("airfoil_naca_flap.py", tmp_path)
    for name in ("naca2412_flap_geometry.png", "naca2412_flap_cp.png"):
        path = tmp_path / name
        assert path.is_file(), f"missing output file {name}"
        assert path.stat().st_size > 0, f"empty output file {name}"


@pytest.mark.slow
def test_example_ground_effect_writes_png_files(tmp_path):
    _run_example("airfoil_ground_effect.py", tmp_path)
    for name in ("naca4412_ground_effect_cp.png", "naca4412_cl_vs_height.png"):
        path = tmp_path / name
        assert path.is_file(), f"missing output file {name}"
        assert path.stat().st_size > 0, f"empty output file {name}"


def test_ground_effect_cp_uses_the_body_frame():
    """Ground-effect curves use the body-frame x, not the turned nodes.

    The solver turns the section by the angle of attack in ground effect.
    For a normalized section (leading edge at (0, 0), unit chord) the
    plotted x must equal the input x/c. Before the fix, the plot used the
    turned solver nodes under the label x/c.
    """
    shape = AirfoilShape.naca("4412", n_points=81).normalized()
    coords = shape.coords
    alpha = 8.0
    results = {
        1.0: solve_inviscid(coords, alpha, height=1.0, backend="numpy"),
        0.3: solve_inviscid(coords, alpha, height=0.3, backend="numpy"),
    }
    solved_x = np.asarray(results[0.3].coordinates, dtype=float)[:, 0]
    assert np.max(np.abs(solved_x - coords[:, 0])) > 1e-3
    ax = plot_ground_effect_cp(results)
    try:
        assert len(ax.lines) == 2
        for line in ax.lines:
            np.testing.assert_allclose(
                line.get_xdata(), coords[:, 0], rtol=DATA_RTOL, atol=DATA_ATOL
            )
        assert "body frame" in ax.get_xlabel()
    finally:
        plt.close(ax.figure)


def test_ground_effect_cp_sweep_uses_each_angle():
    shape = AirfoilShape.naca("0012", n_points=61).normalized()
    coords = shape.coords
    res = solve_inviscid(coords, [2.0, 10.0], height=0.5, backend="numpy")
    ax = plot_ground_effect_cp({0.5: res}, index=1)
    try:
        np.testing.assert_allclose(
            ax.lines[0].get_xdata(), coords[:, 0], rtol=DATA_RTOL, atol=DATA_ATOL
        )
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), np.asarray(res.Cp[1], dtype=float),
            rtol=DATA_RTOL, atol=DATA_ATOL,
        )
    finally:
        plt.close(ax.figure)


def test_plot_cp_labels_the_node_frame():
    coords = _coords()
    res = solve_inviscid(coords, 5.0, backend="numpy")
    ax = plot_cp(res)
    try:
        assert "frame" in ax.get_xlabel()
        assert "chord units" in ax.get_xlabel()
    finally:
        plt.close(ax.figure)


def test_plot_cp_compare_with_fewer_angles_raises():
    """A compare result without the selected angle raises IndexError.

    Before the fix, the plot fell back to the first angle of the compare
    result without a signal.
    """
    coords = _coords()
    res = solve_inviscid(coords, [0.0, 5.0], backend="numpy")
    single = solve_inviscid(coords, 0.0, backend="numpy")
    with pytest.raises(IndexError, match="out of range"):
        plot_cp(res, index=1, compare=single)
    plt.close("all")
    ax = plot_cp(res, index=0, compare=single)
    try:
        assert len(ax.lines) == 2
    finally:
        plt.close(ax.figure)
