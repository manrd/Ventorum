# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests for 3D colored surface panel visualizations (plot_surface_results).

Every test checks the content of the figure: the number of collections, the
axis labels, the colour bar text and one data value. The data checks use a
relative tolerance of 1e-12: the plotted values come from the same result
arrays, so the measured error today is 0, and the tolerance floor is 1e-12
relative.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

import ventorum as vt
from ventorum.geometry.processing import discretize_surface

DATA_RTOL = 1e-12
DATA_ATOL = 1e-12

METRICS = ["Cl", "gamma", "alpha_eff", "alpha_i", "Cd_i", "lift"]

METRIC_LABELS = {
    "Cl": "Section Lift Coefficient $C_l$",
    "gamma": r"Circulation $\Gamma$ [m$^2$/s]",
    "alpha_eff": r"Effective Angle of Attack $\alpha_{eff}$ [°]",
    "alpha_i": r"Induced Downwash Angle $\alpha_i$ [°]",
    "Cd_i": "Section Induced Drag $C_{di}$",
    "lift": "Section Lift $L'$ [N/m]",
}

METRIC_EXTRACTORS = {
    "Cl": lambda sw: sw.Cl,
    "gamma": lambda sw: sw.gamma,
    "alpha_eff": lambda sw: np.degrees(sw.alpha_eff),
    "alpha_i": lambda sw: np.degrees(sw.alpha_i),
    "Cd_i": lambda sw: sw.Cd_i,
    "lift": lambda sw: sw.local_lift,
}


def _expected_facecolors(aircraft, result, metric, settings, cmap_name):
    """Colours that the plot must produce for the metric values.

    The values, the limits and the colour map follow the same rules as the
    plot function.
    """
    all_vals = []
    for idx, surf in enumerate(aircraft.surfaces):
        ds = discretize_surface(surf, settings.n_panels, settings.spacing, idx)
        vals = np.asarray(METRIC_EXTRACTORS[metric](result.spanwise[idx]))
        n = min(len(ds.chords), len(vals))
        all_vals.append(vals[:n])
    all_vals = np.concatenate(all_vals)
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 0.1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return plt.get_cmap(cmap_name)(norm(all_vals))[:, :3]


@pytest.fixture
def sample_aircraft():
    wing = vt.LiftingSurface(
        name="Main Wing",
        semi_span=3.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.2),
            vt.WingSection(y_frac=1.0, chord=0.6, twist=np.radians(-2.0)),
        ],
        sweep_le=np.radians(5.0),
        dihedral=np.radians(3.0),
    )
    tail = vt.LiftingSurface(
        name="V-Tail",
        semi_span=0.8,
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.4),
            vt.WingSection(y_frac=1.0, chord=0.2),
        ],
        dihedral=np.radians(-30.0),
        position=np.array([2.5, 0.0, 0.2]),
    )
    return vt.Aircraft(name="Test Aircraft", surfaces=[wing, tail])


def test_plot_surface_results_all_metrics(sample_aircraft):
    settings = vt.SolverSettings(n_panels=20, spacing="uniform")
    result = vt.analyze(sample_aircraft, alpha_deg=4.0, V_inf=30.0, settings=settings)

    for m in METRICS:
        fig = vt.plot_surface_results(
            sample_aircraft,
            result,
            metric=m,
            settings=settings,
            cmap="plasma",
        )
        assert isinstance(fig, Figure)
        # One 3-D axes and one colour bar axes.
        assert len(fig.axes) == 2
        ax = fig.axes[0]
        # One coloured panel collection for the whole aircraft.
        assert len(ax.collections) == 1
        assert ax.get_xlabel() == "X [m]"
        assert ax.get_ylabel() == "Y [m]"
        assert ax.get_zlabel() == "Z [m]"
        assert fig.axes[1].get_ylabel() == METRIC_LABELS[m]
        # One data value: the colour of every panel encodes the metric there.
        # The 3-D draw stores the face colours in depth order, so the row
        # order of the collection is not the input order. Compare sorted rows.
        actual = np.asarray(ax.collections[0].get_facecolors())[:, :3]
        desired = _expected_facecolors(sample_aircraft, result, m, settings,
                                       "plasma")
        assert actual.shape == desired.shape
        order_actual = np.lexsort((actual[:, 2], actual[:, 1], actual[:, 0]))
        order_desired = np.lexsort((desired[:, 2], desired[:, 1], desired[:, 0]))
        np.testing.assert_allclose(
            actual[order_actual], desired[order_desired],
            rtol=DATA_RTOL, atol=DATA_ATOL,
        )
        plt.close(fig)


def test_plot_surface_results_alias(sample_aircraft):
    result = vt.analyze(sample_aircraft, alpha_deg=2.0, V_inf=25.0)
    fig = vt.plot_panel_results(sample_aircraft, result, metric="Cl")
    assert isinstance(fig, Figure)
    # One 3-D axes and one colour bar axes, one coloured panel collection.
    assert len(fig.axes) == 2
    ax = fig.axes[0]
    assert len(ax.collections) == 1
    assert ax.get_xlabel() == "X [m]"
    assert fig.axes[1].get_ylabel() == METRIC_LABELS["Cl"]
    # One data value: the title reports the lift coefficient of the result.
    assert f"{result.totals.CL:.4f}" in ax.get_title()
    plt.close(fig)


def test_plot_surface_results_invalid_metric(sample_aircraft):
    result = vt.analyze(sample_aircraft, alpha_deg=2.0, V_inf=25.0)
    with pytest.raises(ValueError, match="Unknown metric"):
        vt.plot_surface_results(sample_aircraft, result, metric="invalid_metric_xyz")


def _RecordingPolys(base):
    """Return a subclass of Poly3DCollection that keeps the polygons it gets (in ``polys``)."""

    class Recording(base):
        polys = []

        def __init__(self, verts, *args, **kwargs):
            Recording.polys.append(np.asarray(verts, dtype=float))
            super().__init__(verts, *args, **kwargs)

    return Recording


def _strip_y(polys):
    """Return the y of the quarter-chord mid point of each outline (LE left, LE right, TE right, TE left) [m]."""
    qc_left = polys[:, 0] + 0.25 * (polys[:, 3] - polys[:, 0])
    qc_right = polys[:, 1] + 0.25 * (polys[:, 2] - polys[:, 1])
    return 0.5 * (qc_left[:, 1] + qc_right[:, 1])


@pytest.fixture
def wing_and_tail():
    """Return a wing (10 m span) and a tail, solved with the default settings."""
    wing = vt.LiftingSurface(
        name="Wing", semi_span=5.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.6)],
    )
    tail = vt.LiftingSurface(
        name="Tail", semi_span=1.5,
        sections=[vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.3)],
        position=np.array([4.0, 0.0, 0.2]),
    )
    ac = vt.Aircraft(name="Wing and tail", surfaces=[wing, tail])
    return ac, vt.analyze(ac, alpha_deg=4.0, V_inf=30.0)


def test_surface_plot_strips_are_at_the_solved_positions(wing_and_tail, monkeypatch):
    """Each coloured strip is at the y of its solved strip (default settings, two surfaces).

    Regression: with settings None the plot built cosine strips, but the
    default solve uses the "auto" spacing, so the colours were up to
    1.25 m from their strips on a 10 m wing.
    """
    import mpl_toolkits.mplot3d.art3d as art3d

    ac, result = wing_and_tail
    rec = _RecordingPolys(art3d.Poly3DCollection)
    monkeypatch.setattr(art3d, "Poly3DCollection", rec)
    fig = vt.plot_surface_results(ac, result, metric="Cl")
    assert len(rec.polys) == 1
    y_solved = np.concatenate([sw.y for sw in result.spanwise])
    np.testing.assert_allclose(_strip_y(rec.polys[0]), y_solved, rtol=0.0, atol=1e-12)
    plt.close(fig)


def test_dashboard_strips_are_at_the_solved_positions(wing_and_tail, monkeypatch):
    """The 3-D panel of the dashboard puts each Cl at the y of its solved strip."""
    from ventorum.visualization import dashboard

    ac, result = wing_and_tail
    rec = _RecordingPolys(dashboard.Poly3DCollection)
    monkeypatch.setattr(dashboard, "Poly3DCollection", rec)
    fig = vt.plot_aircraft_dashboard(ac, result)
    assert len(rec.polys) == 1
    y_solved = np.concatenate([sw.y for sw in result.spanwise])
    np.testing.assert_allclose(_strip_y(rec.polys[0]), y_solved, rtol=0.0, atol=1e-12)
    plt.close(fig)


def test_surface_plot_refuses_a_strip_count_mismatch(wing_and_tail):
    """A result whose strip count is not the count of its layout raises; the plot does not truncate."""
    ac, result = wing_and_tail
    bad = result.clone()
    sw = bad.spanwise[1]
    for name in ("y", "gamma", "Cl", "Cd_i", "alpha_eff", "alpha_i", "local_lift"):
        setattr(sw, name, getattr(sw, name)[:-2])
    with pytest.raises(ValueError, match="Tail.*strips"):
        vt.plot_surface_results(ac, bad, metric="Cl")
    with pytest.raises(ValueError, match="Tail.*strips"):
        vt.plot_aircraft_dashboard(ac, bad)


def test_fourier_result_plots_in_surface_plot_and_dashboard(monkeypatch):
    """A Fourier result plots in plot_surface_results and plot_aircraft_dashboard at its own stations."""
    import mpl_toolkits.mplot3d.art3d as art3d
    from ventorum.visualization import dashboard

    wing = vt.LiftingSurface(
        name="Fourier Wing",
        semi_span=4.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    ac = vt.Aircraft(name="Fourier Aircraft", surfaces=[wing])
    settings = vt.SolverSettings(solver_type="fourier", n_panels=12)
    result = vt.analyze(ac, alpha_deg=4.0, V_inf=30.0, settings=settings)
    y_solved = np.asarray(result.spanwise[0].y, dtype=float)
    expected_edges = np.concatenate([[-wing.semi_span], 0.5 * (y_solved[:-1] + y_solved[1:]), [wing.semi_span]])

    rec_surf = _RecordingPolys(art3d.Poly3DCollection)
    monkeypatch.setattr(art3d, "Poly3DCollection", rec_surf)
    fig_surf = vt.plot_surface_results(ac, result, metric="Cl", settings=settings)
    assert isinstance(fig_surf, Figure)
    assert len(rec_surf.polys) == 1
    polys_surf = rec_surf.polys[0]
    assert len(polys_surf) == len(y_solved)
    y_lo = np.minimum(polys_surf[:, 0, 1], polys_surf[:, 1, 1])
    y_hi = np.maximum(polys_surf[:, 0, 1], polys_surf[:, 1, 1])
    assert np.all(y_lo <= y_solved) and np.all(y_solved <= y_hi)
    np.testing.assert_allclose(polys_surf[:, 0, 1], expected_edges[:-1], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(polys_surf[:, 1, 1], expected_edges[1:], rtol=0.0, atol=1e-12)
    plt.close(fig_surf)

    rec_dash = _RecordingPolys(dashboard.Poly3DCollection)
    monkeypatch.setattr(dashboard, "Poly3DCollection", rec_dash)
    fig_dash = vt.plot_aircraft_dashboard(ac, result, settings=settings)
    assert isinstance(fig_dash, Figure)
    assert len(rec_dash.polys) == 1
    polys_dash = rec_dash.polys[0]
    assert len(polys_dash) == len(y_solved)
    y_lo_d = np.minimum(polys_dash[:, 0, 1], polys_dash[:, 1, 1])
    y_hi_d = np.maximum(polys_dash[:, 0, 1], polys_dash[:, 1, 1])
    assert np.all(y_lo_d <= y_solved) and np.all(y_solved <= y_hi_d)
    np.testing.assert_allclose(polys_dash[:, 0, 1], expected_edges[:-1], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(polys_dash[:, 1, 1], expected_edges[1:], rtol=0.0, atol=1e-12)
    plt.close(fig_dash)


def test_dashboard_panel2_uses_solved_strips_and_solver_default(wing_and_tail):
    """Panel 2 of plot_aircraft_dashboard uses the solved lattice strips (or solver default when absent)."""
    ac, result = wing_and_tail
    fig = vt.plot_aircraft_dashboard(ac, result)
    ax2 = next(ax for ax in fig.axes if ax.get_title().startswith("2."))

    lat = result.details["lattice"]
    # Quarter-chord lines are the dashed lines plotted per surface before the MAC lines.
    qc_lines = [line for line in ax2.lines if line.get_linestyle() == "--"]
    assert len(qc_lines) == len(ac.surfaces)
    for sl, line in zip(lat.surfaces, qc_lines):
        s_idx = np.arange(sl.strips.start, sl.strips.stop)
        le_pts = np.vstack([lat.le_left[s_idx, :2], lat.le_right[s_idx[-1:], :2]])
        te_pts = np.vstack([lat.te_left[s_idx, :2], lat.te_right[s_idx[-1:], :2]])
        qc_pts = le_pts + 0.25 * (te_pts - le_pts)
        np.testing.assert_allclose(line.get_xdata(), qc_pts[:, 1], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(line.get_ydata(), qc_pts[:, 0], rtol=0.0, atol=1e-12)
    plt.close(fig)

    # Without a lattice on the result, settings=None uses the solver default (SolverSettings().n_panels), not 35.
    no_lat = result.clone()
    no_lat.details = dict(no_lat.details or {})
    no_lat.details.pop("lattice", None)
    fig2 = vt.plot_aircraft_dashboard(ac, no_lat, settings=None)
    ax2_def = next(ax for ax in fig2.axes if ax.get_title().startswith("2."))
    qc_lines2 = [line for line in ax2_def.lines if line.get_linestyle() == "--"]
    default_n = vt.SolverSettings().n_panels
    assert len(qc_lines2[0].get_xdata()) == 2 * default_n + 1
    plt.close(fig2)


