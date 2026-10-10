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
