# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of deformed geometry: node displacements of the lattice (T-0053)."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.core.errors import GroundStrikeError
from ventorum.geometry.deformation import (
    displacements_from_section_motion,
    undeformed_nodes,
)


@pytest.fixture
def gpu_device():
    """Run the test with the GPU device selected; restore the settings after it."""
    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    old_dev, old_prec = gpu.get_device(), gpu._precision
    yield
    gpu.set_device(old_dev)
    gpu.set_precision(old_prec)


def _sample_wing(is_symmetric: bool = True) -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        is_symmetric=is_symmetric,
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )


# 1. test_zero_displacements_match_undeformed
@pytest.mark.parametrize("solver", ["vlm", "linear", "nonlinear"])
def test_zero_displacements_match_undeformed(solver: str):
    polar = vt.TabulatedAirfoil(
        alpha=np.radians(np.arange(-4.0, 14.0, 2.0)),
        Cl_data=2.0 * np.pi * np.radians(np.arange(-4.0, 14.0, 2.0)),
        Cd_data=0.01 + 0.02 * (np.radians(np.arange(-4.0, 14.0, 2.0))) ** 2,
    )
    af = polar if solver == "nonlinear" else vt.LinearAirfoil()
    wing_undef = vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.5, airfoil=af),
            vt.WingSection(y_frac=1.0, chord=1.0, airfoil=af),
        ],
    )
    ac_undef = vt.Aircraft(surfaces=[wing_undef])
    st = vt.SolverSettings(solver_type=solver, n_panels=16)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))

    res_undef = vt.analyze(ac_undef, cond, st)

    nodes = undeformed_nodes(ac_undef, st, solver=solver)["Wing"]
    d = vt.NodeDisplacements(
        le=np.zeros_like(nodes["le"]),
        te=np.zeros_like(nodes["te"]),
    )
    wing_def = wing_undef.clone()
    wing_def.node_displacements = d
    ac_def = vt.Aircraft(surfaces=[wing_def])

    res_def = vt.analyze(ac_def, cond, st)

    tot_u = res_undef.totals
    tot_d = res_def.totals

    assert np.isclose(tot_u.CL, tot_d.CL, rtol=1e-12, atol=1e-14)
    assert np.isclose(tot_u.CDi, tot_d.CDi, rtol=1e-12, atol=1e-14)
    assert np.isclose(tot_u.Cm, tot_d.Cm, rtol=1e-12, atol=1e-14)
    assert np.isclose(tot_u.CY, tot_d.CY, atol=1e-14)
    assert np.isclose(tot_u.Cl, tot_d.Cl, atol=1e-14)
    assert np.isclose(tot_u.Cn, tot_d.Cn, atol=1e-14)


# 2. test_none_keeps_the_same_bits
def test_none_keeps_the_same_bits():
    wing1 = _sample_wing()
    wing1.node_displacements = None
    ac1 = vt.Aircraft(surfaces=[wing1])

    wing2 = _sample_wing()
    ac2 = vt.Aircraft(surfaces=[wing2])

    st = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=2)
    lat1 = vt.build_lattice(ac1, st, collocation="vlm", n_chord=2)
    lat2 = vt.build_lattice(ac2, st, collocation="vlm", n_chord=2)

    assert np.array_equal(lat1.a, lat2.a)
    assert np.array_equal(lat1.b, lat2.b)
    assert np.array_equal(lat1.cp, lat2.cp)
    assert np.array_equal(lat1.normal_bc, lat2.normal_bc)
    assert np.array_equal(lat1.chord, lat2.chord)
    assert np.array_equal(lat1.width, lat2.width)

    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0))
    res1 = vt.analyze(ac1, cond, st)
    res2 = vt.analyze(ac2, cond, st)

    assert res1.totals.CL == res2.totals.CL
    assert res1.totals.CDi == res2.totals.CDi
    assert res1.totals.Cm == res2.totals.Cm


# 3. test_twist_by_displacements_equals_section_twist
@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_twist_by_displacements_equals_section_twist(solver: str):
    semi_span = 5.0
    chord = 2.0
    st = vt.SolverSettings(solver_type=solver, n_panels=24)
    cond = vt.FlightCondition(V_inf=40.0, alpha=np.radians(3.0))

    wing_disp = vt.LiftingSurface(
        name="Wing",
        semi_span=semi_span,
        sections=[
            vt.WingSection(y_frac=0.0, chord=chord, twist=0.0),
            vt.WingSection(y_frac=1.0, chord=chord, twist=0.0),
        ],
    )
    ac_disp = vt.Aircraft(surfaces=[wing_disp])
    nodes = undeformed_nodes(ac_disp, st, solver=solver)["Wing"]
    eta = nodes["eta"]
    twist_arr = eta * np.radians(-3.0)
    disp = displacements_from_section_motion(
        nodes,
        heave=np.zeros_like(eta),
        twist=twist_arr,
        pivot_x_c=0.25,
    )
    wing_disp.node_displacements = disp
    res_disp = vt.analyze(ac_disp, cond, st)

    wing_geom = vt.LiftingSurface(
        name="Wing",
        semi_span=semi_span,
        sections=[
            vt.WingSection(y_frac=0.0, chord=chord, twist=0.0),
            vt.WingSection(y_frac=1.0, chord=chord, twist=np.radians(-3.0)),
        ],
    )
    ac_geom = vt.Aircraft(surfaces=[wing_geom])
    res_geom = vt.analyze(ac_geom, cond, st)

    assert np.isclose(res_disp.totals.CL, res_geom.totals.CL, rtol=1e-10)
    assert np.isclose(res_disp.totals.CDi, res_geom.totals.CDi, rtol=1e-10)
    assert np.isclose(res_disp.totals.Cm, res_geom.totals.Cm, rtol=1e-10)


# 4. test_rigid_pitch_equals_angle_of_attack
def test_rigid_pitch_equals_angle_of_attack():
    pitch_angle = np.radians(2.0)
    alpha0 = np.radians(3.0)
    ref_pt = np.array([0.5, 0.0, 0.1])
    st = vt.SolverSettings(solver_type="vlm", n_panels=24, n_chord=2, wake_alignment="freestream")

    wing = _sample_wing()
    ac_base = vt.Aircraft(surfaces=[wing], ref_point=ref_pt)
    nodes = undeformed_nodes(ac_base, st, solver="vlm")["Wing"]

    cos_p, sin_p = np.cos(pitch_angle), np.sin(pitch_angle)
    r_y = np.array([
        [cos_p, 0.0, sin_p],
        [0.0, 1.0, 0.0],
        [-sin_p, 0.0, cos_p],
    ])

    le_disp = (nodes["le"] - ref_pt) @ r_y.T + ref_pt - nodes["le"]
    te_disp = (nodes["te"] - ref_pt) @ r_y.T + ref_pt - nodes["te"]

    wing_pitched = wing.clone()
    wing_pitched.node_displacements = vt.NodeDisplacements(le=le_disp, te=te_disp)
    ac_pitched = vt.Aircraft(surfaces=[wing_pitched], ref_point=ref_pt)

    res_pitched = vt.analyze(ac_pitched, vt.FlightCondition(V_inf=30.0, alpha=alpha0), st)
    res_undef = vt.analyze(ac_base, vt.FlightCondition(V_inf=30.0, alpha=alpha0 + pitch_angle), st)

    assert np.isclose(res_pitched.totals.CL, res_undef.totals.CL, rtol=1e-9)
    assert np.isclose(res_pitched.totals.CDi, res_undef.totals.CDi, rtol=1e-9)

    m_pitched = res_pitched.moments("wind")
    m_undef = res_undef.moments("wind")
    for k in ("Cl", "Cm", "Cn"):
        assert np.isclose(m_pitched[k], m_undef[k], atol=1e-9)


# 5. test_rigid_translation_with_reference_point
def test_rigid_translation_with_reference_point():
    dr = np.array([0.3, 0.0, 0.2])
    base_ref = np.array([0.2, 0.0, -0.1])
    st = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=2)
    cond = vt.FlightCondition(V_inf=35.0, alpha=np.radians(4.0))

    wing_undef = _sample_wing()
    ac_undef = vt.Aircraft(surfaces=[wing_undef], ref_point=base_ref)
    res_undef = vt.analyze(ac_undef, cond, st)

    nodes = undeformed_nodes(ac_undef, st, solver="vlm")["Wing"]
    n_edges = len(nodes["eta"])
    d = vt.NodeDisplacements(
        le=np.tile(dr, (n_edges, 1)),
        te=np.tile(dr, (n_edges, 1)),
    )
    wing_trans = wing_undef.clone()
    wing_trans.node_displacements = d
    ac_trans = vt.Aircraft(surfaces=[wing_trans], ref_point=base_ref + dr)
    res_trans = vt.analyze(ac_trans, cond, st)

    tot_u = res_undef.totals
    tot_t = res_trans.totals
    assert np.isclose(tot_u.CL, tot_t.CL, rtol=1e-10)
    assert np.isclose(tot_u.CDi, tot_t.CDi, rtol=1e-10)
    assert np.isclose(tot_u.Cm, tot_t.Cm, rtol=1e-10)
    assert np.isclose(tot_u.CY, tot_t.CY, atol=1e-14)
    assert np.isclose(tot_u.Cl, tot_t.Cl, atol=1e-14)
    assert np.isclose(tot_u.Cn, tot_t.Cn, atol=1e-14)


# 6. test_upward_bending_adds_dihedral_effect
def test_upward_bending_adds_dihedral_effect():
    semi_span = 5.0
    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=semi_span,
        is_symmetric=True,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.5),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    st = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=1)
    ac_flat = vt.Aircraft(surfaces=[wing])

    d_beta = np.radians(1.0)
    cond_p = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=d_beta)
    cond_m = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=-d_beta)

    cl_p_flat = vt.analyze(ac_flat, cond_p, st).totals.Cl
    cl_m_flat = vt.analyze(ac_flat, cond_m, st).totals.Cl
    cl_beta_flat = (cl_p_flat - cl_m_flat) / (2.0 * d_beta)

    nodes = undeformed_nodes(ac_flat, st, solver="vlm")["Wing"]
    eta = nodes["eta"]
    heave = 0.05 * semi_span * (eta ** 2)
    disp = displacements_from_section_motion(
        nodes,
        heave=heave,
        twist=np.zeros_like(eta),
        pivot_x_c=0.25,
    )
    wing_bent = wing.clone()
    wing_bent.node_displacements = disp
    ac_bent = vt.Aircraft(surfaces=[wing_bent])

    cl_p_bent = vt.analyze(ac_bent, cond_p, st).totals.Cl
    cl_m_bent = vt.analyze(ac_bent, cond_m, st).totals.Cl
    cl_beta_bent = (cl_p_bent - cl_m_bent) / (2.0 * d_beta)

    assert cl_beta_bent < cl_beta_flat


# 7. test_mirror_copy_gets_mirrored_displacements
def test_mirror_copy_gets_mirrored_displacements():
    semi_span = 4.0
    st = vt.SolverSettings(solver_type="vlm", n_panels=16, n_chord=2)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))

    wing_sym = vt.LiftingSurface(
        name="WingSym",
        semi_span=semi_span,
        is_symmetric=True,
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    ac_sym = vt.Aircraft(surfaces=[wing_sym])
    nodes = undeformed_nodes(ac_sym, st, solver="vlm")["WingSym"]
    eta = nodes["eta"]
    disp = displacements_from_section_motion(
        nodes,
        heave=0.03 * semi_span * (eta ** 2),
        twist=eta * np.radians(1.5),
        pivot_x_c=0.25,
    )
    wing_sym.node_displacements = disp

    lat_sym = vt.build_lattice(ac_sym, st, collocation="vlm", n_chord=2)
    assert lat_sym.can_fold_symmetry() is True

    wing_r = vt.LiftingSurface(
        name="WingR",
        semi_span=semi_span,
        is_symmetric=False,
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
        node_displacements=disp.clone(),
    )
    wing_l = wing_r.mirrored("WingL")
    ac_pair = vt.Aircraft(surfaces=[wing_r, wing_l])

    res_sym = vt.analyze(ac_sym, cond, st)
    res_pair = vt.analyze(ac_pair, cond, st)

    assert np.isclose(res_sym.totals.CL, res_pair.totals.CL, rtol=1e-10)
    assert np.isclose(res_sym.totals.Cl, res_pair.totals.Cl, atol=1e-14)
    assert np.isclose(res_sym.totals.Cn, res_pair.totals.Cn, atol=1e-14)


# 8. test_wrong_shape_is_refused, test_collapsed_strip_is_refused, test_too_large_displacement_is_refused
def test_wrong_shape_is_refused():
    wing = _sample_wing()
    ac = vt.Aircraft(surfaces=[wing])
    st = vt.SolverSettings(solver_type="vlm", n_panels=10)

    # Wrong number of edges
    d_wrong_len = vt.NodeDisplacements(
        le=np.zeros((5, 3)),
        te=np.zeros((5, 3)),
    )
    wing.node_displacements = d_wrong_len
    with pytest.raises(ValueError) as exc:
        vt.build_lattice(ac, st)
    msg = str(exc.value)
    assert wing.name in msg
    assert "node_displacements" in msg
    assert "ventorum.geometry.undeformed_nodes" in msg

    # Non-finite values
    nodes = undeformed_nodes(ac, st, solver="vlm")["Wing"]
    d_nan = vt.NodeDisplacements(
        le=np.full_like(nodes["le"], np.nan),
        te=np.zeros_like(nodes["te"]),
    )
    wing.node_displacements = d_nan
    with pytest.raises(ValueError) as exc_nan:
        vt.build_lattice(ac, st)
    assert wing.name in str(exc_nan.value)
    assert "node_displacements" in str(exc_nan.value)


def test_collapsed_strip_is_refused():
    wing = _sample_wing()
    ac = vt.Aircraft(surfaces=[wing])
    st = vt.SolverSettings(solver_type="vlm", n_panels=10)
    nodes = undeformed_nodes(ac, st, solver="vlm")["Wing"]

    # Collapsed chord: move trailing edge onto leading edge
    d_collapsed_chord = vt.NodeDisplacements(
        le=np.zeros_like(nodes["le"]),
        te=nodes["le"] - nodes["te"],
    )
    wing.node_displacements = d_collapsed_chord
    with pytest.raises(ValueError) as exc_chord:
        vt.build_lattice(ac, st)
    msg_chord = str(exc_chord.value)
    assert wing.name in msg_chord
    assert "node_displacements" in msg_chord

    # Collapsed strip width: move edge 1 onto edge 0
    d_le = np.zeros_like(nodes["le"])
    d_te = np.zeros_like(nodes["te"])
    d_le[1] = nodes["le"][0] - nodes["le"][1]
    d_te[1] = nodes["te"][0] - nodes["te"][1]
    wing.node_displacements = vt.NodeDisplacements(le=d_le, te=d_te)
    with pytest.raises(ValueError) as exc_width:
        vt.build_lattice(ac, st)
    msg_width = str(exc_width.value)
    assert wing.name in msg_width
    assert "node_displacements" in msg_width


def test_too_large_displacement_is_refused():
    wing = _sample_wing()
    ac = vt.Aircraft(surfaces=[wing])
    st = vt.SolverSettings(solver_type="vlm", n_panels=10)
    nodes = undeformed_nodes(ac, st, solver="vlm")["Wing"]

    d_too_large = vt.NodeDisplacements(
        le=np.full_like(nodes["le"], wing.semi_span * 1.5),
        te=np.zeros_like(nodes["te"]),
    )
    wing.node_displacements = d_too_large
    with pytest.raises(ValueError) as exc:
        vt.build_lattice(ac, st)
    msg = str(exc.value)
    assert wing.name in msg
    assert "node_displacements" in msg


# 9. test_undeformed_nodes_match_the_lattice
def test_undeformed_nodes_match_the_lattice():
    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=6.0,
        controls=[
            vt.ControlSurface(name="flap", eta_start=0.4, eta_end=0.8, hinge_x_c=0.75),
        ],
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    tail = vt.LiftingSurface(
        name="Tail",
        semi_span=2.0,
        position=np.array([4.0, 0.0, 0.5]),
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.8),
            vt.WingSection(y_frac=1.0, chord=0.5),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing, tail])
    st = vt.SolverSettings(proportional_panels=True, n_panels=40)

    for solver, colloc in [("vlm", "vlm"), ("linear", "llt")]:
        nodes = undeformed_nodes(ac, st, solver=solver)
        lat = vt.build_lattice(ac, st, collocation=colloc)
        surf_slice = [s for s in lat.surfaces if s.name == "Wing"][0]
        n_edges_half = len(nodes["Wing"]["eta"])
        # The right half edge_le coordinates in the lattice slice
        edge_le_lat_right = surf_slice.edge_le[n_edges_half - 1:]
        assert np.max(np.abs(nodes["Wing"]["le"] - edge_le_lat_right)) <= 1e-14
        # Check against _surface_eta directly
        from ventorum.geometry.discretization import compute_surface_n_panels
        from ventorum.geometry.lattice import _surface_eta, resolve_spacing
        n_sp = compute_surface_n_panels(wing, base_n_panels=st.n_panels, reference_semi_span=6.0, min_panels=8)
        spacing = resolve_spacing(st.spacing, wing, colloc)
        eta_expected, _ = _surface_eta(wing, n_sp, spacing)
        assert np.max(np.abs(nodes["Wing"]["eta"] - eta_expected)) <= 1e-14


def _swept_wing() -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sweep_le=np.radians(30.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )


def test_undeformed_nodes_follow_the_solver_spacing():
    """The "auto" spacing depends on the solver; the nodes must follow the real solver path."""
    from ventorum.solvers.factory import make_solver, resolve_solver_type

    ac = vt.Aircraft(surfaces=[_swept_wing()])
    st = vt.SolverSettings(n_panels=16)
    assert st.spacing == "auto"
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))

    eta_vlm = undeformed_nodes(ac, st, solver="vlm")["Wing"]["eta"]
    eta_llt = undeformed_nodes(ac, st, solver="linear")["Wing"]["eta"]
    assert eta_vlm.shape == eta_llt.shape
    assert np.max(np.abs(eta_vlm - eta_llt)) > 0.1

    for solver in ("vlm", "linear", "nonlinear", "auto", "llt", "horseshoe"):
        st_s = vt.SolverSettings(n_panels=16, solver_type=solver)
        nodes = undeformed_nodes(ac, st_s, solver=solver)["Wing"]
        lat = make_solver(resolve_solver_type(solver)).build(ac, st_s, cond)
        sl = lat.surfaces[0]
        n_half = len(nodes["eta"])
        assert sl.edge_le.shape[0] == 2 * n_half - 1
        assert np.max(np.abs(nodes["le"] - sl.edge_le[n_half - 1:])) <= 1e-14
        assert np.max(np.abs(nodes["te"] - sl.edge_te[n_half - 1:])) <= 1e-14


@pytest.mark.filterwarnings("ignore:Lifting line with")
def test_station_check_refuses_the_mesh_of_another_solver():
    """Displacements made for the VLM mesh of a swept wing are refused by the lifting line."""
    wing = _swept_wing()
    ac = vt.Aircraft(surfaces=[wing])
    st_vlm = vt.SolverSettings(n_panels=16, solver_type="vlm")
    st_llt = vt.SolverSettings(n_panels=16, solver_type="linear")
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))

    nodes = undeformed_nodes(ac, st_vlm, solver="vlm")["Wing"]
    eta = nodes["eta"]
    disp = displacements_from_section_motion(nodes, heave=0.05 * eta ** 2, twist=np.zeros_like(eta))
    assert disp.eta is not None and np.array_equal(disp.eta, eta)
    wing.node_displacements = disp

    # The same solver passes.
    res = vt.analyze(ac, cond, st_vlm)
    assert np.isfinite(res.totals.CL)

    # Another solver has other stations (with the same count): refused.
    with pytest.raises(ValueError, match=r"node_displacements\.eta") as exc:
        vt.analyze(ac, cond, st_llt)
    msg = str(exc.value)
    assert "[Wing]" in msg
    assert "Use the same settings and solver" in msg
    assert "undeformed_nodes" in msg

    # Stations with another edge count are also refused.
    wing.node_displacements = vt.NodeDisplacements(le=disp.le, te=disp.te, eta=eta[:-1])
    with pytest.raises(ValueError, match=r"node_displacements\.eta"):
        vt.build_lattice(ac, st_vlm)

    # eta None keeps the old behaviour: no station check, and the same bits.
    wing.node_displacements = vt.NodeDisplacements(le=disp.le, te=disp.te)
    assert wing.node_displacements.eta is None
    assert np.isfinite(vt.analyze(ac, cond, st_llt).totals.CL)
    res_none = vt.analyze(ac, cond, st_vlm)
    assert res_none.totals.CL == res.totals.CL
    assert res_none.totals.CDi == res.totals.CDi
    assert res_none.totals.Cm == res.totals.Cm


def test_undeformed_nodes_refuses_fourier():
    ac = vt.Aircraft(surfaces=[_sample_wing()])
    with pytest.raises(ValueError, match="Fourier"):
        undeformed_nodes(ac, solver="fourier")
    with pytest.raises(ValueError, match="Unknown solver"):
        undeformed_nodes(ac, solver="panel")


# 10. test_cache_key_follows_displacements
def test_cache_key_follows_displacements():
    wing = _sample_wing()
    ac = vt.Aircraft(surfaces=[wing])
    st = vt.SolverSettings(solver_type="vlm", n_panels=16)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))

    nodes = undeformed_nodes(ac, st, solver="vlm")["Wing"]
    d_orig = vt.NodeDisplacements(
        le=np.zeros_like(nodes["le"]),
        te=np.zeros_like(nodes["te"]),
    )
    wing.node_displacements = d_orig
    res1 = vt.analyze(ac, cond, st)

    # Change one value
    d_mod = vt.NodeDisplacements(
        le=nodes["le"] * 0.0,
        te=nodes["te"] * 0.0,
    )
    d_mod.te[-1, 2] = 0.05
    wing.node_displacements = d_mod
    res2 = vt.analyze(ac, cond, st)
    assert res1.totals.CL != res2.totals.CL

    # Restore old values
    d_restored = vt.NodeDisplacements(
        le=np.zeros_like(nodes["le"]),
        te=np.zeros_like(nodes["te"]),
    )
    wing.node_displacements = d_restored
    res3 = vt.analyze(ac, cond, st)
    assert res1.totals.CL == res3.totals.CL
    assert res1.totals.CDi == res3.totals.CDi


# 11. test_fourier_refuses_displacements
def test_fourier_refuses_displacements():
    wing = _sample_wing()
    wing.node_displacements = vt.NodeDisplacements(
        le=np.zeros((10, 3)),
        te=np.zeros((10, 3)),
    )
    ac = vt.Aircraft(surfaces=[wing])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="fourier")

    with pytest.raises(ValueError) as exc:
        vt.FourierSolver().solve(ac, cond, st)
    msg = str(exc.value)
    assert "vlm" in msg
    assert "linear" in msg or "nonlinear" in msg


# 12. test_json_round_trip_keeps_displacements
def test_json_round_trip_keeps_displacements():
    wing = _sample_wing()
    nodes = undeformed_nodes(vt.Aircraft(surfaces=[wing]), solver="vlm")["Wing"]
    disp = displacements_from_section_motion(
        nodes,
        heave=0.02 * (nodes["eta"] ** 2),
        twist=nodes["eta"] * np.radians(2.0),
        pivot_x_c=0.25,
    )
    wing.node_displacements = disp
    ac = vt.Aircraft(surfaces=[wing])

    json_str = vt.aircraft_to_json(ac)
    data = json.loads(json_str)
    assert data["surfaces"][0]["node_displacements"] is not None

    ac_loaded = vt.aircraft_from_json(json_str)
    wing_loaded = ac_loaded.surfaces[0]
    assert wing_loaded.node_displacements is not None
    # The JSON text keeps every float64 value exactly (shortest round-trip repr).
    assert np.array_equal(wing_loaded.node_displacements.le, disp.le)
    assert np.array_equal(wing_loaded.node_displacements.te, disp.te)
    assert np.array_equal(wing_loaded.node_displacements.eta, disp.eta)

    # Round trip without the stations (eta None).
    wing_no_eta = _sample_wing()
    wing_no_eta.node_displacements = vt.NodeDisplacements(le=disp.le, te=disp.te)
    json_no_eta = vt.aircraft_to_json(vt.Aircraft(surfaces=[wing_no_eta]))
    nd_no_eta = vt.aircraft_from_json(json_no_eta).surfaces[0].node_displacements
    assert nd_no_eta.eta is None
    assert np.array_equal(nd_no_eta.le, disp.le)
    assert np.array_equal(nd_no_eta.te, disp.te)

    # Test round-trip with None
    wing_none = _sample_wing()
    wing_none.node_displacements = None
    json_none = vt.aircraft_to_json(vt.Aircraft(surfaces=[wing_none]))
    ac_none_loaded = vt.aircraft_from_json(json_none)
    assert ac_none_loaded.surfaces[0].node_displacements is None


# 13. test_displacements_with_control_surface
def test_displacements_with_control_surface():
    st = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=2)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))

    flap_0 = vt.ControlSurface(name="flap", eta_start=0.3, eta_end=0.7, hinge_x_c=0.75, deflection=0.0)
    wing_0 = vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        controls=[flap_0],
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    nodes = undeformed_nodes(vt.Aircraft(surfaces=[wing_0]), st, solver="vlm")["Wing"]
    disp = displacements_from_section_motion(
        nodes,
        heave=0.04 * (nodes["eta"] ** 2),
        twist=np.zeros_like(nodes["eta"]),
        pivot_x_c=0.25,
    )
    wing_0.node_displacements = disp
    res_0 = vt.analyze(vt.Aircraft(surfaces=[wing_0]), cond, st)

    flap_def = vt.ControlSurface(name="flap", eta_start=0.3, eta_end=0.7, hinge_x_c=0.75, deflection=np.radians(10.0))
    wing_def = wing_0.clone()
    wing_def.controls = [flap_def]
    wing_def.node_displacements = disp.clone()
    res_def = vt.analyze(vt.Aircraft(surfaces=[wing_def]), cond, st)

    assert res_def.totals.CL > res_0.totals.CL


# 14. test_ground_clearance_uses_deformed_nodes
def test_ground_clearance_uses_deformed_nodes():
    wing = _sample_wing()
    st = vt.SolverSettings(solver_type="vlm", n_panels=16)
    # Undistorted wing at h=1.0 m clears the ground
    cond = vt.FlightCondition(V_inf=30.0, alpha=0.0, h=1.0)
    ac_undef = vt.Aircraft(surfaces=[wing])
    res_undef = vt.analyze(ac_undef, cond, st)
    assert res_undef.converged

    # Deform tip downward by 1.5 m so that it penetrates ground (clearance < 0)
    nodes = undeformed_nodes(ac_undef, st, solver="vlm")["Wing"]
    d_le = np.zeros_like(nodes["le"])
    d_te = np.zeros_like(nodes["te"])
    d_le[-1, 2] = -1.5
    d_te[-1, 2] = -1.5
    wing_down = wing.clone()
    wing_down.node_displacements = vt.NodeDisplacements(le=d_le, te=d_te)
    ac_down = vt.Aircraft(surfaces=[wing_down])

    with pytest.raises(GroundStrikeError):
        vt.analyze(ac_down, cond, st)


# 15. test_gpu_deformed_lattice_equals_cpu
@pytest.mark.gpu
@pytest.mark.parametrize("solver_type", ["vlm", "linear"])
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_gpu_deformed_lattice_equals_cpu(gpu_device, solver_type: str, precision: str):
    semi_span = 4.0
    st = vt.SolverSettings(solver_type=solver_type, n_panels=16, n_chord=2 if solver_type == "vlm" else 1)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))

    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=semi_span,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.8),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    nodes = undeformed_nodes(vt.Aircraft(surfaces=[wing]), st, solver=solver_type)["Wing"]
    disp = displacements_from_section_motion(
        nodes,
        heave=0.03 * semi_span * (nodes["eta"] ** 2),
        twist=nodes["eta"] * np.radians(2.0),
        pivot_x_c=0.25,
    )
    wing.node_displacements = disp
    ac = vt.Aircraft(surfaces=[wing])

    gpu.set_device("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_cpu = vt.analyze(ac.clone(), cond, st)

        gpu.set_device("gpu")
        gpu.set_precision(precision)
        res_gpu = vt.analyze(ac.clone(), cond, st)

    assert res_gpu.details.get("device") == "gpu"
    cl_max = max(abs(float(res_cpu.totals.CL)), 1e-3)
    tol = 1e-11 if precision == "float64" else 2e-5
    assert abs(res_cpu.totals.CL - res_gpu.totals.CL) <= tol * cl_max
    assert abs(res_cpu.totals.CDi - res_gpu.totals.CDi) <= tol * cl_max
    assert abs(res_cpu.totals.Cm - res_gpu.totals.Cm) <= tol * cl_max
