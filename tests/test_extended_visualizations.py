# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests for comprehensive physical, geometric, and aerodynamic visualizations.

Every plot test checks the content of the figure: the number of lines or
collections, the axis labels and one data value. The data checks use a
relative tolerance of 1e-12: the plotted values come from the same result
arrays, so the measured error today is 0, and the tolerance floor is 1e-12
relative.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import ventorum as vt

DATA_RTOL = 1e-12
DATA_ATOL = 1e-12


@pytest.fixture
def aircraft_and_results():
    wing = vt.LiftingSurface(
        name="Main Wing",
        semi_span=2.5,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=0.5, twist=np.radians(-1.5)),
        ],
        sweep_le=np.radians(4.0),
        dihedral=np.radians(2.0),
    )
    tail = vt.LiftingSurface(
        name="V-Tail",
        semi_span=0.7,
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.35),
            vt.WingSection(y_frac=1.0, chord=0.2),
        ],
        dihedral=np.radians(-35.0),
        position=np.array([2.0, 0.0, 0.15]),
    )
    ac = vt.Aircraft(name="Test Drone", surfaces=[wing, tail])
    settings = vt.SolverSettings(n_panels=20, spacing="uniform")
    res = vt.analyze(ac, alpha_deg=3.5, V_inf=30.0, settings=settings)

    # Sweep results
    alphas_deg = np.array([0.0, 2.0, 4.0, 6.0])
    sweep = vt.analyze_sweep(ac, alphas_deg, V_inf=30.0, n_panels=15, spacing="uniform")

    return ac, res, sweep, alphas_deg, settings


def _span_loading_value(result) -> float:
    """L'(y) = rho * V_inf * gamma at the first station of the first surface."""
    sw0 = result.spanwise[0]
    return result.condition.rho * result.condition.V_inf * sw0.gamma[0]


def _dashboard_table(fig, result) -> None:
    """The dashboard summary table exists and reports the lift coefficient."""
    table_axes = [ax for ax in fig.axes if ax.tables]
    assert len(table_axes) == 1
    table = table_axes[0].tables[0]
    # 16 parameters plus the header row, 2 columns.
    assert len(table.get_celld()) == 34
    # One data value: the CL cell of the table equals the result value.
    assert table.get_celld()[(8, 1)].get_text().get_text() == f"{result.totals.CL:.4f}"


def _dashboard_span_panel(fig, result) -> None:
    """The span loading panel of the dashboard plots rho * V_inf * gamma."""
    panels = [ax for ax in fig.axes if ax.get_ylabel() == r"$L'(y)$ [N/m]"]
    assert len(panels) == 1
    assert panels[0].get_xlabel() == "Spanwise Y [m]"
    np.testing.assert_allclose(
        panels[0].lines[0].get_ydata()[0], _span_loading_value(result),
        rtol=DATA_RTOL, atol=DATA_ATOL,
    )


def test_wake_and_trefftz_visualizations(aircraft_and_results):
    ac, res, _, _, _ = aircraft_and_results

    # 1. 3D Vortex Wake
    fig_wake = vt.plot_vortex_wake(ac, res)
    assert isinstance(fig_wake, Figure)
    ax_wake = fig_wake.axes[0]
    lat = res.details["lattice"]
    # One line per bound vortex and one line per shed trailing edge.
    n_lines = len(lat.a) + sum(len(lat.te_left[s.strips]) + 1 for s in lat.surfaces)
    assert len(ax_wake.lines) == n_lines
    assert ax_wake.get_xlabel() == "x [m] (aft)"
    assert ax_wake.get_ylabel() == "y [m] (right)"
    assert ax_wake.get_zlabel() == "z [m] (up)"
    assert fig_wake.axes[1].get_ylabel() == r"Shed vorticity $|\Delta\Gamma|$ [m$^2$/s]"
    # One data value: the first line is the first bound vortex of the lattice.
    np.testing.assert_allclose(ax_wake.lines[0].get_xdata(), [lat.a[0, 0], lat.b[0, 0]],
                               rtol=DATA_RTOL, atol=DATA_ATOL)
    np.testing.assert_allclose(ax_wake.lines[0].get_ydata(), [lat.a[0, 1], lat.b[0, 1]],
                               rtol=DATA_RTOL, atol=DATA_ATOL)
    plt.close(fig_wake)

    # 2. Trefftz Plane
    fig_trefftz = vt.plot_trefftz_plane(
        ac, res, grid_res=(15, 12), streamlines=True
    )
    assert isinstance(fig_trefftz, Figure)
    ax_trefftz = fig_trefftz.axes[0]
    # One contour collection and one streamline collection.
    assert len(ax_trefftz.collections) == 2
    # One line for the trailing edge of each surface.
    assert len(ax_trefftz.lines) == len(ac.surfaces)
    assert ax_trefftz.get_xlabel() == "y [m]"
    assert ax_trefftz.get_ylabel() == "z [m]"
    assert fig_trefftz.axes[1].get_ylabel() == r"Downwash angle $-w/V$ [deg]"
    # One data value: the 31 contour levels are symmetric about zero.
    levels = ax_trefftz.collections[0].levels
    assert len(levels) == 31
    np.testing.assert_allclose(levels[0], -levels[-1], rtol=DATA_RTOL, atol=DATA_ATOL)
    plt.close(fig_trefftz)


def test_planform_and_span_loading_visualizations(aircraft_and_results):
    ac, res, _, _, _ = aircraft_and_results

    # 1. 2D Planform Blueprint
    fig_plan = vt.plot_planform_2d(ac, show_mac=True, show_quarter_chord=True, show_dimensions=True)
    assert isinstance(fig_plan, Figure)
    ax_plan = fig_plan.axes[0]
    # Two quarter-chord lines (one per surface) and three MAC lines.
    assert len(ax_plan.lines) == 5
    # One planform polygon per surface.
    assert len(ax_plan.patches) == 2
    assert ax_plan.get_xlabel() == "Spanwise Coordinate Y [m]"
    assert ax_plan.get_ylabel() == "Chordwise Coordinate X [m]"
    assert ax_plan.lines[0].get_label() == "Main Wing (1/4c)"
    # One data value: the MAC line has the length of the mean aerodynamic chord.
    np.testing.assert_allclose(np.ptp(ax_plan.lines[2].get_ydata()), ac.c_ref,
                               rtol=DATA_RTOL, atol=DATA_ATOL)
    plt.close(fig_plan)

    # 2. Span Loading L'(y)
    fig_span = vt.plot_span_loading(ac, res, show_elliptic=True)
    assert isinstance(fig_span, Figure)
    ax_span = fig_span.axes[0]
    # One line per surface plus the elliptic reference line.
    assert len(ax_span.lines) == 3
    assert ax_span.get_xlabel() == "Spanwise Station Y [m]"
    assert ax_span.get_ylabel() == "Sectional Lift Loading $L'(y)$ [N/m]"
    # One data value: the first point of the first curve.
    np.testing.assert_allclose(ax_span.lines[0].get_xdata()[0], res.spanwise[0].y[0],
                               rtol=DATA_RTOL, atol=DATA_ATOL)
    np.testing.assert_allclose(ax_span.lines[0].get_ydata()[0], _span_loading_value(res),
                               rtol=DATA_RTOL, atol=DATA_ATOL)
    plt.close(fig_span)


def test_performance_stability_and_polar_visualizations(aircraft_and_results):
    _, _, sweep, alphas_deg, _ = aircraft_and_results

    # 1. Efficiency Curves
    fig_eff = vt.plot_efficiency_curves(sweep, alpha_range=np.radians(alphas_deg))
    assert isinstance(fig_eff, Figure)
    ax_ld, ax_end = fig_eff.axes
    assert len(ax_ld.lines) == 1
    assert len(ax_end.lines) == 1
    assert ax_ld.get_xlabel() == r"Angle of Attack $\alpha$ [°]"
    assert ax_ld.get_ylabel() == "Glide Ratio $L/D$"
    assert ax_end.get_ylabel() == r"Endurance Factor $C_L^{1.5} / C_D$"
    # One data value: L/D at the first angle of attack of the sweep.
    t0 = sweep[0].totals
    cd0 = t0.CD_total if t0.CD_total is not None else t0.CDi
    np.testing.assert_allclose(ax_ld.lines[0].get_ydata()[0], t0.CL / max(float(cd0), 1e-6),
                               rtol=DATA_RTOL, atol=DATA_ATOL)
    plt.close(fig_eff)

    # 2. Pitching Moment
    fig_cm = vt.plot_pitching_moment(sweep, alpha_range=np.radians(alphas_deg))
    assert isinstance(fig_cm, Figure)
    ax_cm_alpha, ax_cm_cl = fig_cm.axes
    # The curve and the zero reference line on each axes.
    assert len(ax_cm_alpha.lines) == 2
    assert len(ax_cm_cl.lines) == 2
    assert ax_cm_alpha.get_ylabel() == r"Pitching Moment Coefficient $C_m$"
    assert ax_cm_cl.get_xlabel() == r"Total Lift Coefficient $C_L$"
    # One data value: the Cm curve against the angle of attack.
    np.testing.assert_allclose(ax_cm_alpha.lines[0].get_ydata(), [r.totals.Cm for r in sweep],
                               rtol=DATA_RTOL, atol=DATA_ATOL)
    plt.close(fig_cm)

    # 3. Airfoil Polar
    airfoil = vt.LinearAirfoil(name="NACA 0012", a0=2.0 * np.pi, alpha_L0=0.0, Cd0=0.008)
    alpha_deg = np.linspace(-4, 12, 20)
    fig_polar = vt.plot_airfoil_polar(airfoil, alpha_deg_range=alpha_deg)
    assert isinstance(fig_polar, Figure)
    # Four axes: Cl-alpha (curve plus two reference lines), Cd-alpha,
    # Cl-Cd and Cm-alpha (curve plus the zero line).
    assert [len(ax.lines) for ax in fig_polar.axes] == [3, 1, 1, 2]
    ax_cl = fig_polar.axes[0]
    assert ax_cl.get_xlabel() == r"$\alpha$ [°]"
    assert ax_cl.get_ylabel() == "$C_l$"
    # One data value: the section lift curve of the airfoil.
    np.testing.assert_allclose(ax_cl.lines[0].get_ydata(), airfoil.Cl(np.radians(alpha_deg)),
                               rtol=DATA_RTOL, atol=DATA_ATOL)
    plt.close(fig_polar)


def test_fourier_and_breakdown_visualizations(aircraft_and_results):
    _, res, _, _, _ = aircraft_and_results

    # 1. Fourier Spectrum
    fig_four = vt.plot_fourier_spectrum(res, n_harmonics=10, surface_index=0)
    assert isinstance(fig_four, Figure)
    ax_four = fig_four.axes[0]
    # One bar per harmonic.
    assert len(ax_four.patches) == 10
    assert ax_four.get_xlabel() == "Harmonic Number $n$"
    assert ax_four.get_ylabel() == r"Relative Amplitude $|A_n / A_1|$ [%]"
    # One data value: the first bar is A1 / A1 = 100 %.
    np.testing.assert_allclose(ax_four.patches[0].get_height(), 100.0,
                               rtol=DATA_RTOL, atol=DATA_ATOL)
    plt.close(fig_four)

    # 2. Component Breakdown
    fig_brk = vt.plot_component_breakdown(res)
    assert isinstance(fig_brk, Figure)
    ax_force, ax_share = fig_brk.axes
    # Lift and drag bars for the two surfaces on each axes.
    assert len(ax_force.patches) == 4
    assert len(ax_share.patches) == 4
    assert ax_force.get_ylabel() == "Aerodynamic Force [N]"
    assert ax_share.get_ylabel() == "Share of Total Force [%]"
    # One data value: the lift bar of each surface, integrated from local lift.
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    lifts = []
    for sw in res.spanwise:
        order = np.argsort(sw.y)
        lifts.append(float(_trap(sw.local_lift[order], sw.y[order])))
    np.testing.assert_allclose([p.get_height() for p in ax_force.patches[:2]], lifts,
                               rtol=DATA_RTOL, atol=DATA_ATOL)
    plt.close(fig_brk)


def test_executive_dashboard(aircraft_and_results):
    ac, res, sweep, alphas_deg, settings = aircraft_and_results

    # Dashboard with full sweep
    fig_dash = vt.plot_aircraft_dashboard(
        aircraft=ac,
        result=res,
        sweep_results=sweep,
        alpha_sweep_deg=alphas_deg,
        settings=settings,
    )
    assert isinstance(fig_dash, Figure)
    # Eight panels, one colour bar and one twin axis for the endurance curve.
    assert len(fig_dash.axes) == 10
    _dashboard_table(fig_dash, res)
    _dashboard_span_panel(fig_dash, res)
    plt.close(fig_dash)

    # Dashboard with single operating point
    fig_dash_single = vt.plot_aircraft_dashboard(
        aircraft=ac,
        result=res,
        settings=settings,
    )
    assert isinstance(fig_dash_single, Figure)
    # Eight panels and one colour bar; the curve panels show single bars.
    assert len(fig_dash_single.axes) == 9
    _dashboard_table(fig_dash_single, res)
    _dashboard_span_panel(fig_dash_single, res)
    plt.close(fig_dash_single)


@pytest.fixture
def sweep_2deg():
    """Return a wing and a sweep from 0 to 14 deg in steps of 2 deg [deg]."""
    wing = vt.LiftingSurface(
        name="Wing", semi_span=2.5,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.5)],
    )
    tail = vt.LiftingSurface(
        name="Tail", semi_span=0.7,
        sections=[vt.WingSection(y_frac=0.0, chord=0.35), vt.WingSection(y_frac=1.0, chord=0.2)],
        position=np.array([2.0, 0.0, 0.15]),
    )
    ac = vt.Aircraft(name="Sweep", surfaces=[wing, tail])
    alphas_deg = np.arange(0.0, 15.0, 2.0)
    sweep = vt.analyze_sweep(ac, alphas_deg, V_inf=30.0, n_panels=15, spacing="uniform")
    return ac, sweep, alphas_deg


def _central_slope(alphas_deg, values):
    """Return the slope [1/rad] of the central fit of the plots (five points around the middle)."""
    mid = len(values) // 2
    lo, hi = max(0, mid - 2), min(len(values), mid + 3)
    return np.polyfit(np.radians(alphas_deg[lo:hi]), np.asarray(values)[lo:hi], 1)[0]


def test_sweep_plots_take_the_angles_from_the_results(sweep_2deg):
    """Without alpha_range the angle axis and the slopes come from the flight conditions of the results.

    Regression: the axis was the result index, so a 2 deg step doubled
    CL_alpha and Cm_alpha and moved the trim angle and the angle of
    (L/D)max.
    """
    ac, sweep, alphas_deg = sweep_2deg
    cls = [r.totals.CL for r in sweep]

    fig = vt.plot_cl_vs_alpha(sweep)
    ax = fig.axes[0]
    np.testing.assert_allclose(ax.lines[0].get_xdata(), alphas_deg, rtol=0.0, atol=1e-12)
    slope = _central_slope(alphas_deg, cls)
    assert f"CL_α ≈ {slope:.3f} /rad" in ax.texts[0].get_text()
    plt.close(fig)

    fig = vt.plot_efficiency_curves(sweep)
    np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), alphas_deg, rtol=0.0, atol=1e-12)
    plt.close(fig)

    fig = vt.plot_pitching_moment(sweep)
    ax_cm = fig.axes[0]
    np.testing.assert_allclose(ax_cm.lines[0].get_xdata(), alphas_deg, rtol=0.0, atol=1e-12)
    cm_alpha = _central_slope(alphas_deg, [r.totals.Cm for r in sweep])
    assert f" = {cm_alpha:.3f}$ /rad" in ax_cm.texts[0].get_text()
    plt.close(fig)

    fig = vt.plot_aircraft_dashboard(ac, sweep[3], sweep_results=sweep)
    ax6 = next(a for a in fig.axes if a.get_title().startswith("6."))
    ax7 = next(a for a in fig.axes if a.get_title().startswith("7."))
    np.testing.assert_allclose(ax6.lines[0].get_xdata(), alphas_deg, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(ax7.lines[0].get_xdata(), alphas_deg, rtol=0.0, atol=1e-12)
    cm_fit = np.polyfit(np.radians(alphas_deg), [r.totals.Cm for r in sweep], 1)[0]
    assert f"{cm_fit:.3f} /rad" in ax7.texts[0].get_text()
    plt.close(fig)

    # A given alpha_range [rad] still wins.
    fig = vt.plot_cl_vs_alpha(sweep, alpha_range=np.radians(alphas_deg + 1.0))
    np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), alphas_deg + 1.0, rtol=0.0, atol=1e-12)
    plt.close(fig)


def test_sweep_plots_refuse_a_result_without_an_angle(sweep_2deg):
    """A result without a flight condition and no alpha_range raises a clear error."""
    _, sweep, _ = sweep_2deg
    bad = [r.clone() for r in sweep]
    bad[2].condition = None
    for plot in (vt.plot_cl_vs_alpha, vt.plot_efficiency_curves, vt.plot_pitching_moment):
        with pytest.raises(ValueError, match="angle of attack"):
            plot(bad)
