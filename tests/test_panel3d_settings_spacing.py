# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for panel method settings spanwise spacing and strip loads."""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.geometry.lattice import _surface_eta, resolve_spacing
from ventorum.solvers.panel3d import PanelMethodSolver


def _ar8_surface(spacing: str | None = None) -> LiftingSurface:
    """Build a rectangular wing surface with aspect ratio 8."""
    return LiftingSurface(
        name="Wing",
        semi_span=4.0,
        is_symmetric=True,
        spacing=spacing,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, shape="0012"),
            WingSection(y_frac=1.0, chord=1.0, shape="0012"),
        ],
    )


def test_strip_loads_follow_settings_spacing() -> None:
    """The strip Cl distribution matches between surface spacing and settings spacing.

    Compare an aircraft with surface spacing set to None and settings spacing set
    to cosine against an aircraft with surface spacing set to cosine directly.
    """
    ac_settings = Aircraft(surfaces=[_ar8_surface(spacing=None)])
    ac_surface = Aircraft(surfaces=[_ar8_surface(spacing="cosine")])
    cond = FlightCondition(V_inf=50.0, alpha=np.radians(4.0))
    settings = SolverSettings(spacing="cosine", n_panels=10, n_chord=10)

    solver = PanelMethodSolver()
    res_settings = solver.solve(ac_settings, cond, settings)
    res_surface = solver.solve(ac_surface, cond, settings)

    cl_settings = res_settings.spanwise[0].Cl
    cl_surface = res_surface.spanwise[0].Cl

    # The Cl distribution must match to round-off
    scale = float(np.max(np.abs(cl_surface)))
    diff = np.max(np.abs(cl_settings - cl_surface))
    assert diff <= max(1e-12 * scale, 1e-14)

    # The local lift per unit span must match to round-off
    lift_settings = res_settings.spanwise[0].local_lift
    lift_surface = res_surface.spanwise[0].local_lift
    lift_scale = float(np.max(np.abs(lift_surface)))
    lift_diff = np.max(np.abs(lift_settings - lift_surface))
    assert lift_diff <= max(1e-12 * lift_scale, 1e-14)

    # Totals match to round-off
    assert abs(res_settings.totals.CL - res_surface.totals.CL) <= 1e-12 * abs(res_surface.totals.CL)

    # Root strip Cl is physical (around 0.38, not unphysically high)
    mid = len(cl_settings) // 2
    root_cl = float(cl_settings[mid])
    assert 0.35 <= root_cl <= 0.42


@pytest.mark.parametrize(
    ("surf_spacing", "settings_spacing"),
    [
        ("uniform", None),
        (None, "uniform"),
        ("cosine", None),
        (None, "cosine"),
    ],
)
def test_strip_edges_match_panel_edges(surf_spacing: str | None, settings_spacing: str | None) -> None:
    """Every strip edge equals a panel edge of the mesh for uniform and cosine spacing."""
    surf = _ar8_surface(spacing=surf_spacing)
    ac = Aircraft(surfaces=[surf])
    cond = FlightCondition(V_inf=50.0, alpha=np.radians(4.0))
    settings = SolverSettings(spacing=settings_spacing, n_panels=10, n_chord=6)

    res = PanelMethodSolver().solve(ac, cond, settings)
    mesh = res.details["mesh"]

    eff_spacing = resolve_spacing(
        surf.spacing if surf.spacing is not None else settings.spacing,
        surf,
        "vlm",
    )
    eta_e, _ = _surface_eta(surf, settings.n_panels, eff_spacing)
    y_strip_edges = np.asarray(eta_e, dtype=float) * float(surf.semi_span)

    # Unique y coordinates of the nodes on the positive semi-span
    nodes = np.asarray(mesh.nodes, dtype=float)
    y_nodes = np.unique(np.round(nodes[:, 1], decimals=8))
    y_nodes_pos = y_nodes[y_nodes >= -1e-8]

    assert len(y_nodes_pos) == len(y_strip_edges)
    assert np.allclose(y_nodes_pos, y_strip_edges, atol=1e-8)


@pytest.mark.parametrize("spacing", ["uniform", "cosine"])
def test_sum_of_strip_lift_is_total(spacing: str) -> None:
    """The integrated strip lift equals the total CL within 1e-10 relative."""
    surf = _ar8_surface(spacing=None)
    ac = Aircraft(surfaces=[surf])
    cond = FlightCondition(V_inf=50.0, alpha=np.radians(4.0))
    settings = SolverSettings(spacing=spacing, n_panels=10, n_chord=8)

    res = PanelMethodSolver().solve(ac, cond, settings)
    sw = res.spanwise[0]

    eff_spacing = resolve_spacing(
        surf.spacing if surf.spacing is not None else settings.spacing,
        surf,
        "vlm",
    )
    eta_e, _ = _surface_eta(surf, settings.n_panels, eff_spacing)
    y_e = np.asarray(eta_e, dtype=float) * float(surf.semi_span)
    dy = np.diff(y_e)
    widths = np.concatenate([dy[::-1], dy])

    q_inf = 0.5 * float(cond.rho) * float(cond.V_inf) ** 2
    s_ref = float(ac.S_ref)

    strip_lift_sum = float(np.sum(sw.local_lift * widths))
    cl_from_strips = strip_lift_sum / (q_inf * s_ref)

    rel_diff = abs(cl_from_strips - res.totals.CL) / abs(res.totals.CL)
    assert rel_diff < 1e-10
