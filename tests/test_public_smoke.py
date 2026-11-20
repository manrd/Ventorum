# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Smoke tests of public functions that had no direct test.

The plot functions run with the Agg backend. Each test checks that the
function returns a figure with the expected axes, labels and plotted data.
The JSON test writes an aircraft to a file, reads it back and solves both.
The MCP test sends one valid and some invalid JSON-RPC requests to
``handle_request``.
"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import ventorum as vt  # noqa: E402
from ventorum.agent import handle_request  # noqa: E402
from ventorum.core.config import load_aircraft_from_json, save_aircraft_to_json  # noqa: E402
from ventorum.visualization import (  # noqa: E402
    plot_all_distributions,
    plot_alpha_distribution,
    plot_cl_distribution,
    plot_convergence,
    plot_drag_polar,
    plot_geometry,
    plot_induced_drag_distribution,
    plot_lift_distribution,
    plot_sweep_summary,
)

ALPHAS_DEG = (0.0, 4.0, 8.0)


def _aircraft():
    af = vt.LinearAirfoil(Cd0=0.01)
    wing = vt.LiftingSurface(name="wing", semi_span=3.0, sections=[
        vt.WingSection(y_frac=0.0, chord=1.0, airfoil=af),
        vt.WingSection(y_frac=1.0, chord=0.6, airfoil=af, twist=np.radians(-2.0))])
    return vt.Aircraft(surfaces=[wing])


def _solve(ac, alpha_deg, solver="vlm"):
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(alpha_deg))
    st = vt.SolverSettings(solver_type=solver, n_panels=12)
    return vt.analyze(ac, condition=cond, settings=st)


@pytest.fixture(scope="module")
def results():
    ac = _aircraft()
    return [_solve(ac, a) for a in ALPHAS_DEG]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("plot, ylabel", [
    (plot_cl_distribution, "Cl"),
    (plot_alpha_distribution, "Angle [°]"),
    (plot_induced_drag_distribution, "Cd_i"),
])
def test_distribution_plots(results, plot, ylabel):
    res = results[1]
    fig = plot(res)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlabel() == "y / (b/2)"
    assert ax.get_ylabel() == ylabel
    assert len(ax.get_lines()) >= len(res.spanwise)


def test_lift_distribution_plots_the_normalised_circulation(results):
    res = results[1]
    fig = plot_lift_distribution(res)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    # One line per surface, and the elliptic reference line.
    assert fig.axes[0].get_ylabel() == "Γ / Γ_max"
    lines = fig.axes[0].get_lines()
    assert len(lines) == len(res.spanwise) + 1
    gamma = np.asarray(lines[0].get_ydata())
    assert np.max(np.abs(gamma)) == pytest.approx(1.0, rel=1e-12)


def test_plot_given_axes_is_used(results):
    fig, ax = plt.subplots()
    out = plot_cl_distribution(results[1], ax=ax)
    assert out is fig
    assert len(fig.axes) == 1


def test_all_distributions_has_four_axes(results):
    fig = plot_all_distributions(results[1])
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4
    assert [ax.get_ylabel() for ax in fig.axes] == ["Γ / Γ_max", "Cl", "Angle [°]", "Cd_i"]


def test_drag_polar_plots_the_results(results):
    fig = plot_drag_polar(results)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    line = fig.axes[0].get_lines()[0]
    np.testing.assert_allclose(line.get_ydata(), [r.totals.CL for r in results], rtol=1e-12)
    np.testing.assert_allclose(line.get_xdata(), [r.totals.CDi for r in results], rtol=1e-12)


def test_sweep_summary_has_two_axes(results):
    fig = plot_sweep_summary(results, np.radians(ALPHAS_DEG))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    cl_line = fig.axes[0].get_lines()[0]
    np.testing.assert_allclose(cl_line.get_xdata(), ALPHAS_DEG, rtol=1e-12)
    np.testing.assert_allclose(cl_line.get_ydata(), [r.totals.CL for r in results], rtol=1e-12)


def test_geometry_plot_is_3d():
    fig = plot_geometry(_aircraft(), vt.SolverSettings(n_panels=6))
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1
    assert fig.axes[0].name == "3d"


def test_convergence_plot_without_iterations(results):
    """A linear solve has no residual history: the plot says so."""
    fig = plot_convergence(results[0])
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert any("No iteration data" in text.get_text() for text in fig.axes[0].texts)


def test_convergence_plot_with_iterations():
    res = _solve(_aircraft(), 4.0, solver="nonlinear")
    assert res.residual_history, "the nonlinear solver gave no residual history"
    fig = plot_convergence(res)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    line = ax.get_lines()[0]
    np.testing.assert_allclose(line.get_ydata(), res.residual_history, rtol=1e-12)


def test_aircraft_json_round_trip(tmp_path):
    ac = _aircraft()
    path = tmp_path / "aircraft.json"
    save_aircraft_to_json(ac, path)
    json.loads(path.read_text(encoding="utf-8"))  # the file is valid JSON
    ac2 = load_aircraft_from_json(path)
    assert [s.name for s in ac2.surfaces] == [s.name for s in ac.surfaces]
    r1 = _solve(ac, 4.0).totals
    r2 = _solve(ac2, 4.0).totals
    # The same geometry gives the same solve, to round-off.
    for name in ("CL", "CDi", "Cm"):
        assert getattr(r2, name) == pytest.approx(getattr(r1, name), rel=1e-12, abs=1e-15)
    assert r2.CDp == pytest.approx(r1.CDp, rel=1e-12)


def test_handle_request_valid():
    reply = handle_request({"jsonrpc": "2.0", "id": 7, "method": "ping"})
    assert reply == {"jsonrpc": "2.0", "id": 7, "result": {}}
    reply = handle_request({"jsonrpc": "2.0", "id": "t", "method": "tools/list"})
    assert reply["id"] == "t"
    assert "error" not in reply
    assert len(reply["result"]["tools"]) > 0


@pytest.mark.parametrize("msg, code", [
    ([1, 2], -32600),                                            # not an object
    ({"jsonrpc": "2.0", "id": None, "method": "ping"}, -32600),  # null id
    ({"jsonrpc": "1.0", "id": 1, "method": "ping"}, -32600),     # wrong version
    ({"jsonrpc": "2.0", "id": 1, "method": "no/such"}, -32601),  # unknown method
    ({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": [1]}, -32602),
    ({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "no_such_tool"}}, -32602),
])
def test_handle_request_invalid(msg, code):
    reply = handle_request(msg)
    assert "result" not in reply
    assert reply["error"]["code"] == code
    assert reply["error"]["message"]


def test_handle_request_notification_gets_no_reply():
    assert handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
