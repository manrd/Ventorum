# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests for the surface gradient on the flat tip caps of 3DP wings.

The centroids of the cap panels of a cambered section follow the camber
line, so the edge-neighbour least-squares stencil of a cap panel is almost
one-dimensional. Before the fix the fit kept the weak direction across the
chord, and round-off in the doublet values gave cap pressures of order 1e3
to 1e4. With dihedral the cap force then changed the lift. A cap panel now
keeps only the gradient along its row of neighbours, as a symmetric section
already did.
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.airfoil import airfoil
from ventorum.core.datatypes import FlightCondition, LiftingSurface, WingSection
from ventorum.panel3d.kernels import _precompute_geometry
from ventorum.panel3d.mesh import check_mesh, mesh_lifting_surface
from ventorum.panel3d.post import (
    _build_surface_gradient_operator_reference,
    build_surface_gradient_operator,
)
from ventorum.panel3d.images import reconstruct_full_mesh
from ventorum.panel3d.solver import PanelSolver, _standardize_panels

ALPHA = np.radians(3.0)

# NACA 0012 results before the fix (commit 87fcc02), from this test setup:
# (CL, CD_pressure, CDi_trefftz, Cm, Cl) and the Cp of the right cap panels.
# The fix does not change a symmetric section: its cap stencils are already
# collinear to round-off.
_BASE_0012 = {
    0.0: (
        (0.24786632158307664, -0.00035238512346799416, 0.00247650183660262, -0.059638129543185765,
         8.347250399390949e-18),
        [-0.07719358198836423, -0.10268742060309055, -0.10672202439934608, -0.10024900924028035,
         -0.09104682984989343, -0.08094397731202752, -0.06912989740226383, -0.054342587623420036,
         -0.03508918272312811, -0.011091897716775101, 0.010790684044762355, 0.02171451144835035],
    ),
    10.0: (
        (0.24653794356113368, -0.0003913093053301684, 0.0024940655832007483, -0.06394217750223295,
         -1.2746973464940256e-17),
        [-0.04111752046700512, -0.06817859937315363, -0.08202336820266365, -0.08542353508113831,
         -0.08116551697993879, -0.07217983468392553, -0.05958514579324836, -0.04369358293184544,
         -0.024884842951579778, -0.004801532779372231, 0.00986858065261198, 0.011249845989052654],
    ),
}


def _wing(dihedral_deg: float) -> LiftingSurface:
    """Return a rectangular wing, semi-span 4 m, chord 1 m."""
    return LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, twist=0.0),
            WingSection(y_frac=1.0, chord=1.0, twist=0.0),
        ],
        is_symmetric=True,
        dihedral=np.radians(dihedral_deg),
    )


def _solve(section: str, dihedral_deg: float, tip: str, n_chord: int = 16, n_span: int = 10):
    """Return the half mesh and the result at alpha 3 deg in free air."""
    mesh = mesh_lifting_surface(
        _wing(dihedral_deg), airfoil(section).coords, n_chord=n_chord, n_span=n_span, half=True, tip=tip
    )
    result = PanelSolver(mesh, far_ratio=np.inf).solve(FlightCondition(V_inf=1.0, alpha=ALPHA))
    return mesh, result


@pytest.mark.parametrize("section", ["2412", "4412"])
@pytest.mark.parametrize("dihedral_deg", [0.0, 10.0, 30.0])
def test_cambered_cap_pressure_is_bounded(section, dihedral_deg):
    """The cap pressures of a cambered section are of the order of the wing pressures."""
    mesh, res = _solve(section, dihedral_deg, "flat")
    assert check_mesh(mesh)["watertight"]
    assert check_mesh(mesh)["outward"]
    assert check_mesh(mesh)["valid_panels"]
    kind = np.asarray(mesh.kind)
    cap_cp = np.abs(res.cp[kind == "tip"])
    wing_cp = np.abs(res.cp[kind == "surface"])
    assert np.all(np.isfinite(cap_cp))
    assert cap_cp.max() < 3.0
    assert cap_cp.max() < 3.0 * wing_cp.max()


@pytest.mark.parametrize("section", ["2412", "4412"])
@pytest.mark.parametrize("dihedral_deg", [10.0, 30.0])
def test_cambered_cap_lift_follows_open_tip(section, dihedral_deg):
    """With dihedral the lift of the capped wing stays within 2 % of the open-tip wing."""
    _, flat = _solve(section, dihedral_deg, "flat")
    _, open_tip = _solve(section, dihedral_deg, "none")
    assert flat.CL == pytest.approx(open_tip.CL, rel=0.02)


@pytest.mark.slow
@pytest.mark.parametrize("section", ["2412", "4412"])
@pytest.mark.parametrize("dihedral_deg", [10.0, 30.0])
def test_cambered_cap_lift_follows_open_tip_default_mesh(section, dihedral_deg):
    """On the default mesh (40 chordwise, 20 spanwise panels) the old cap lift was 5 % to 28 % high."""
    _, flat = _solve(section, dihedral_deg, "flat", n_chord=40, n_span=20)
    _, open_tip = _solve(section, dihedral_deg, "none", n_chord=40, n_span=20)
    assert flat.CL == pytest.approx(open_tip.CL, rel=0.02)


@pytest.mark.parametrize("dihedral_deg", [0.0, 10.0])
def test_symmetric_section_does_not_change(dihedral_deg):
    """NACA 0012 gives the results from before the fix (round-off tolerance for other BLAS builds)."""
    mesh, res = _solve("0012", dihedral_deg, "flat", n_chord=12, n_span=8)
    coeffs, cap_cp = _BASE_0012[dihedral_deg]
    got = (res.CL, res.CD_pressure, res.CDi_trefftz, res.Cm, res.Cl)
    np.testing.assert_allclose(got, coeffs, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(res.cp[np.asarray(mesh.kind) == "tip"], cap_cp, rtol=1e-12, atol=1e-14)


def test_cap_gradient_array_build_matches_reference():
    """The array build and the loop reference give the same cap rows on a cambered wing."""
    mesh = reconstruct_full_mesh(
        mesh_lifting_surface(_wing(10.0), airfoil("4412").coords, n_chord=12, n_span=6, half=True)
    )
    panels = _standardize_panels(mesh.panels, mesh.nodes)
    geom = _precompute_geometry(mesh.nodes[panels])
    args = (mesh.nodes, panels, geom.centroid, geom.normal, geom.area, mesh.wake_edges)
    op = build_surface_gradient_operator(*args, kinds=mesh.kind)
    ref = _build_surface_gradient_operator_reference(*args, kinds=mesh.kind)
    np.testing.assert_allclose(op.toarray(), ref.toarray(), rtol=1e-12, atol=1e-12)


def test_cap_gradient_is_along_the_row():
    """A cap panel of a cambered section has a gradient in one direction only."""
    mesh = reconstruct_full_mesh(
        mesh_lifting_surface(_wing(0.0), airfoil("2412").coords, n_chord=12, n_span=6, half=True)
    )
    panels = _standardize_panels(mesh.panels, mesh.nodes)
    geom = _precompute_geometry(mesh.nodes[panels])
    op = build_surface_gradient_operator(
        mesh.nodes, panels, geom.centroid, geom.normal, geom.area, mesh.wake_edges, kinds=mesh.kind
    ).toarray()
    kind = np.asarray(mesh.kind)
    for p in np.flatnonzero(kind == "tip"):
        block = op[3 * p:3 * p + 3]
        sv = np.linalg.svd(block, compute_uv=False)
        assert sv[1] <= 1e-10 * sv[0]
    # A surface panel next to the cap keeps a two-dimensional gradient.
    surf = np.flatnonzero(kind == "surface")
    ranks = [np.linalg.matrix_rank(op[3 * p:3 * p + 3]) for p in surf]
    assert min(ranks) == 2
