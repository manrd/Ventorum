# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for shared-hinge topology verification in gradient checks."""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.core.datatypes import (
    Aircraft,
    ControlSurface,
    FlightCondition,
    LiftingSurface,
    LinearAirfoil,
    SolverSettings,
    WingSection,
)
from ventorum.geometry.lattice import build_lattice
import ventorum.gpu as gpu
from ventorum.sensitivity import check_gradients, gradients, shared_hinge_topology
from ventorum.sensitivity.errors import GradientScopeError


@pytest.fixture(autouse=True)
def _float64_precision():
    old_device, old = gpu.get_device(), gpu._precision
    gpu.set_device("cpu")
    gpu.set_precision("float64")
    yield
    gpu.set_precision(old)
    gpu.set_device(old_device)


def _joined_flap_aircraft() -> Aircraft:
    """Build an aircraft with a symmetric inner wing and joined outer surfaces with flaps."""
    inner = LiftingSurface(
        name="inner",
        semi_span=2.0,
        is_symmetric=True,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, airfoil=LinearAirfoil()),
            WingSection(y_frac=1.0, chord=1.0, airfoil=LinearAirfoil()),
        ],
        controls=[ControlSurface("f", 0.0, 1.0, 0.3, 0.1)],
    )
    outer = LiftingSurface(
        name="outer",
        semi_span=2.0,
        is_symmetric=False,
        position=np.array([0.0, 2.0, 0.0]),
        sections=[
            WingSection(y_frac=0.0, chord=1.0, airfoil=LinearAirfoil()),
            WingSection(y_frac=1.0, chord=1.0, airfoil=LinearAirfoil()),
        ],
        controls=[ControlSurface("f", 0.0, 1.0, 0.3, 0.1)],
    )
    return Aircraft(surfaces=[inner, outer, outer.mirrored()])


def test_shared_hinge_topology_matches_lattice():
    """Topology helper returns the same result as the lattice for joined surfaces."""
    ac = _joined_flap_aircraft()
    st = SolverSettings(n_panels=4, n_chord=4)
    lat = build_lattice(ac, st, collocation="vlm", n_chord=4)
    topo = shared_hinge_topology(lat)

    lat_partner = lat.geom_cache["flap_field"]["partner"]
    assert topo == lat_partner
    assert np.array_equal(topo.partner, lat_partner)
    assert ("inner", "outer") in topo.shared_surfaces
    assert ("inner", "outer (mirror)") in topo.shared_surfaces

    single_wing = LiftingSurface(
        name="wing",
        semi_span=2.0,
        is_symmetric=True,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, airfoil=LinearAirfoil()),
            WingSection(y_frac=1.0, chord=1.0, airfoil=LinearAirfoil()),
        ],
        controls=[ControlSurface("f", 0.0, 1.0, 0.3, 0.1)],
    )
    ac_single = Aircraft(surfaces=[single_wing])
    lat_single = build_lattice(ac_single, st, collocation="vlm", n_chord=4)
    topo_single = shared_hinge_topology(lat_single)
    assert len(topo_single.shared_surfaces) == 0


def test_check_gradients_raises_when_node_step_separates_hinge():
    """check_gradients raises GradientScopeError when a perturbation separates a shared hinge."""
    ac = _joined_flap_aircraft()
    cond = FlightCondition(V_inf=30.0, alpha=0.05)
    st = SolverSettings(n_panels=4, n_chord=4)

    with pytest.raises(GradientScopeError) as exc_info:
        check_gradients(
            ac,
            cond,
            st,
            wrt=("nodes",),
            node_step=1e-4,
            outputs=("CL", "CDi", "Cm"),
        )

    msg = str(exc_info.value)
    assert "inner" in msg
    assert "outer" in msg
    assert "nodes" in msg
    assert "fixed" in msg.lower()
    assert "hinge topology" in msg.lower()
    assert "smaller" in msg.lower()


def test_check_gradients_passes_when_topology_preserved():
    """check_gradients passes on the joined-flap aircraft when topology is preserved."""
    ac = _joined_flap_aircraft()
    cond = FlightCondition(V_inf=30.0, alpha=0.05)
    st = SolverSettings(n_panels=4, n_chord=4)

    report = check_gradients(
        ac,
        cond,
        st,
        wrt=("alpha", "controls"),
        outputs=("CL", "CDi", "Cm"),
    )
    assert report["pass"], report.get("reasons")


def test_existing_fixtures_gradients_bit_identical():
    """Existing fixtures produce bit-identical adjoint gradients."""
    wing = LiftingSurface(
        name="wing",
        semi_span=4.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, shape="4412", airfoil=LinearAirfoil()),
            WingSection(y_frac=1.0, chord=0.6, shape="2412", airfoil=LinearAirfoil()),
        ],
        controls=[ControlSurface("f", 0.2, 0.8, 0.25, 0.05)],
    )
    ac = Aircraft(surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
    cond = FlightCondition(V_inf=30.0, alpha=0.05)
    st = SolverSettings(n_panels=4, n_chord=4)

    res1 = gradients(ac, cond, st, outputs=("CL", "CDi", "Cm"), wrt=("alpha", "controls"))
    res2 = gradients(ac, cond, st, outputs=("CL", "CDi", "Cm"), wrt=("alpha", "controls"))

    assert np.array_equal(res1.jacobian["CL"]["alpha"], res2.jacobian["CL"]["alpha"])
    assert np.array_equal(res1.jacobian["CL"]["controls"]["f"], res2.jacobian["CL"]["controls"]["f"])
