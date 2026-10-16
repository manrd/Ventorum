# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for deflected plain control surfaces in 3D panel meshes."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from ventorum.airfoil import airfoil
from ventorum.core.datatypes import Aircraft, ControlSurface, FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.panel3d.kernels import _precompute_geometry, influence_matrices
from ventorum.panel3d.mesh import check_mesh, mesh_lifting_surface
from ventorum.panel3d.solver import PanelSolver
from ventorum.solvers.horseshoe import VortexLatticeSolver


def _naca0012(n: int = 401) -> np.ndarray:
    """Selig ordered unit-chord coordinates of a NACA 0012 section."""
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * 0.12 * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )
    upper = np.column_stack([x, yt])[::-1]
    lower = np.column_stack([x, -yt])[1:]
    return np.vstack([upper, lower])


def _rect_wing() -> LiftingSurface:
    """Symmetric rectangular wing with unit chord and semi-span 4 m."""
    return LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, twist=0.0),
            WingSection(y_frac=1.0, chord=1.0, twist=0.0),
        ],
        is_symmetric=True,
    )


def _te_nodes(mesh) -> np.ndarray:
    """Return the unique trailing-edge node positions of a mesh [m]."""
    we = np.asarray(mesh.wake_edges)
    ids = np.unique(we[:, 2:4].ravel())
    return np.asarray(mesh.nodes)[ids]


def _check_flap_case(delta_deg: float) -> None:
    """Check watertightness and trailing-edge rotation for one deflection."""
    coords = _naca0012()
    ctrl = ControlSurface(
        name="flap",
        eta_start=0.4,
        eta_end=0.8,
        hinge_x_c=0.75,
        deflection=np.radians(delta_deg),
        symmetric=True,
    )
    surf = _rect_wing()
    surf.controls = [ctrl]
    mesh = mesh_lifting_surface(surf, coords, n_chord=12, n_span=12)
    info = check_mesh(mesh)
    assert info["watertight"]
    assert info["outward"]
    assert info["valid_panels"], f"non-convex {info['n_nonconvex']}, max warp {info['max_warp']:.2e}"
    side = np.asarray(mesh.kind) == "side"
    assert int(np.sum(side)) > 0
    side_panels = np.asarray(mesh.panels)[side]
    assert np.all(side_panels[:, 2] == side_panels[:, 3]), "side panels are triangles"
    y_side = np.asarray(mesh.nodes)[side_panels[:, :3], 1]
    assert np.all(np.ptp(y_side, axis=1) < 1e-12), "each side triangle lies in one plane y = const"
    assert set(np.round(np.abs(y_side[:, 0]), 9).tolist()) == {1.6, 3.2}

    delta = np.radians(delta_deg)
    te = _te_nodes(mesh)
    x_hinge = 0.75
    cos_d = float(np.cos(delta))
    sin_d = float(np.sin(delta))
    tol_eta = 1e-9
    for p in te:
        y = float(p[1])
        eta = abs(y) / 4.0
        inside = (0.4 - tol_eta <= eta <= 0.8 + tol_eta)
        on_edge = abs(eta - 0.4) < 1e-9 or abs(eta - 0.8) < 1e-9
        x_base, z_base = 1.0, 0.0
        dx = x_base - x_hinge
        dz = z_base - 0.0
        x_exp = x_hinge + dx * cos_d + dz * sin_d
        z_exp = 0.0 - dx * sin_d + dz * cos_d
        if inside and not on_edge:
            assert abs(float(p[0]) - x_exp) < 1e-12, f"TE x mismatch at y={y}"
            assert abs(float(p[2]) - z_exp) < 1e-12, f"TE z mismatch at y={y}"
        elif not inside:
            assert abs(float(p[0]) - x_base) < 1e-12, f"TE x changed outside at y={y}"
            assert abs(float(p[2]) - z_base) < 1e-12, f"TE z changed outside at y={y}"
        else:
            ok_moved = abs(float(p[0]) - x_exp) < 1e-12 and abs(float(p[2]) - z_exp) < 1e-12
            ok_same = (
                abs(float(p[0]) - x_base) < 1e-12 and abs(float(p[2]) - z_base) < 1e-12
            )
            assert ok_moved or ok_same, f"Boundary TE must be moved or same at y={y}"
    ys = np.array(sorted({round(float(p[1]), 9) for p in te}))
    for edge_eta in (0.4, 0.8):
        for sign in (1.0, -1.0):
            y_edge = sign * edge_eta * 4.0
            matches = ys[np.abs(ys - y_edge) < 1e-9]
            assert len(matches) == 1
            count = int(np.sum(np.abs(te[:, 1] - y_edge) < 1e-9))
            assert count == 2, f"Expected side pair at y={y_edge}, got {count}"


def test_zero_deflection_identical_mesh() -> None:
    """Zero deflection gives the same mesh as before, bit for bit."""
    from ventorum.geometry.lattice import _surface_eta, resolve_spacing

    coords = _naca0012()
    probe = _rect_wing()
    spacing = resolve_spacing(probe.spacing, probe, "vlm")
    eta_base, _ = _surface_eta(probe, 12, spacing)
    inner = sorted(float(e) for e in eta_base if 1e-12 < float(e) < 1.0 - 1e-12)
    eta_start = float(inner[2])
    eta_end = float(inner[5])
    plain = mesh_lifting_surface(_rect_wing(), coords, n_chord=12, n_span=12)
    surf = _rect_wing()
    surf.controls = [
        ControlSurface(
            name="flap",
            eta_start=eta_start,
            eta_end=eta_end,
            hinge_x_c=0.75,
            deflection=0.0,
            symmetric=True,
        )
    ]
    moved = mesh_lifting_surface(surf, coords, n_chord=12, n_span=12)
    np.testing.assert_array_equal(np.asarray(moved.nodes), np.asarray(plain.nodes))
    np.testing.assert_array_equal(np.asarray(moved.panels), np.asarray(plain.panels))
    np.testing.assert_array_equal(np.asarray(moved.kind), np.asarray(plain.kind))
    np.testing.assert_array_equal(np.asarray(moved.component), np.asarray(plain.component))
    np.testing.assert_array_equal(np.asarray(moved.wake_edges), np.asarray(plain.wake_edges))
    assert bool(moved.half) == bool(plain.half)
    for field in ("strip", "chord_index", "side", "node_station"):
        np.testing.assert_array_equal(np.asarray(getattr(moved, field)), np.asarray(getattr(plain, field)))


def test_flap_10deg_te_rotation() -> None:
    """Check the 10 deg flap over 40 to 80 percent semi-span."""
    _check_flap_case(10.0)


def test_flap_minus_20deg_te_rotation() -> None:
    """Check the minus 20 deg flap over 40 to 80 percent semi-span."""
    _check_flap_case(-20.0)


def test_panel_solver_flap_cl_increase() -> None:
    """Check that a positive flap deflection increases panel CL."""
    coords = _naca0012(201)
    surf0 = _rect_wing()
    mesh0 = mesh_lifting_surface(surf0, coords, n_chord=12, n_span=10)
    solver0 = PanelSolver(mesh0, far_ratio=np.inf)
    cond = FlightCondition(V_inf=50.0, alpha=np.radians(4.0))
    res0 = solver0.solve(cond)
    surf = _rect_wing()
    surf.controls = [
        ControlSurface(
            name="flap",
            eta_start=0.4,
            eta_end=0.8,
            hinge_x_c=0.75,
            deflection=np.radians(10.0),
            symmetric=True,
        )
    ]
    mesh = mesh_lifting_surface(surf, coords, n_chord=12, n_span=10)
    res = PanelSolver(mesh, far_ratio=np.inf).solve(cond)
    print(f"Panel flap: CL={res.CL:.6f} (base {res0.CL:.6f})")
    assert res.CL > res0.CL, "Positive flap must increase panel CL"


def test_panel_solver_aileron_sign_matches_vlm() -> None:
    """Check that an antisymmetric aileron gives the VLM rolling moment sign."""
    coords = _naca0012(201)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    wing_vlm = LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
        is_symmetric=True,
        controls=[
            ControlSurface(
                name="aileron",
                eta_start=0.4,
                eta_end=0.8,
                hinge_x_c=0.75,
                deflection=np.radians(5.0),
                symmetric=False,
            )
        ],
    )
    ac_vlm = Aircraft(name="A", surfaces=[wing_vlm], ref_point=np.zeros(3))
    settings = SolverSettings(solver_type="vlm", n_panels=10, n_chord=4)
    res_vlm = VortexLatticeSolver().solve(ac_vlm, cond, settings)
    cl_vlm = float(res_vlm.totals.Cl)
    surf = _rect_wing()
    surf.controls = [
        ControlSurface(
            name="aileron",
            eta_start=0.4,
            eta_end=0.8,
            hinge_x_c=0.75,
            deflection=np.radians(5.0),
            symmetric=False,
        )
    ]
    mesh = mesh_lifting_surface(surf, coords, n_chord=12, n_span=10)
    info = check_mesh(mesh)
    assert info["watertight"]
    assert info["outward"]
    res = PanelSolver(mesh, far_ratio=np.inf).solve(cond)
    print(f"Aileron Cl: panel={res.Cl:.6f}, vlm={cl_vlm:.6f}")
    assert cl_vlm < 0.0, f"VLM aileron must give Cl < 0, got {cl_vlm}"
    assert res.Cl < 0.0, f"Panel aileron must give Cl < 0, got {res.Cl}"
    assert np.sign(res.Cl) == np.sign(cl_vlm)


def _closed_naca0012(n: int = 401) -> np.ndarray:
    """NACA 0012 with a closed (sharp) trailing edge, Selig order."""
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * 0.12 * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )
    upper = np.column_stack([x, yt])[::-1]
    lower = np.column_stack([x, -yt])[1:]
    return np.vstack([upper, lower])


_COND = FlightCondition(V_inf=50.0, alpha=np.radians(4.0))
_S_REF = 8.0
_B_REF = 8.0


def _plain_wing(controls=None) -> LiftingSurface:
    """Symmetric rectangular wing, unit chord, semi-span 4 m, with controls."""
    surf = _rect_wing()
    surf.controls = list(controls or [])
    return surf


def _flap(delta_deg: float) -> list:
    """Symmetric plain flap over 40 to 80 percent semi-span, hinge at 75 percent chord."""
    return [ControlSurface(name="flap", eta_start=0.4, eta_end=0.8, hinge_x_c=0.75,
                           deflection=np.radians(delta_deg), symmetric=True)]


def _aileron(delta_deg: float) -> list:
    """Antisymmetric aileron over 40 to 80 percent semi-span, hinge at 75 percent chord."""
    return [ControlSurface(name="aileron", eta_start=0.4, eta_end=0.8, hinge_x_c=0.75,
                           deflection=np.radians(delta_deg), symmetric=False)]


def _panel_solve(controls, coords, n_chord: int, n_span: int = 10):
    """Mesh, solve (far_ratio default 5) and return (result, solver, mesh)."""
    mesh = mesh_lifting_surface(_plain_wing(controls), coords, n_chord=n_chord, n_span=n_span)
    solver = PanelSolver(mesh, S_ref=_S_REF, b_ref=_B_REF, c_ref=1.0)
    return solver.solve(_COND), solver, mesh


def _circulation_coefficients(mesh, result) -> tuple[float, float]:
    """Return CL and Cl from the wake strip strengths (Kutta-Joukowski).

    With the internal Dirichlet condition the perturbation potential on the
    surface is -mu, so the circulation of a strip is
    Gamma = phi_upper - phi_lower = -(mu_upper - mu_lower) at the trailing
    edge. The lift per strip is rho V Gamma times the strip width [m]; the
    rolling moment uses the AVL sign (Cl > 0 right wing down).
    """
    we = np.asarray(mesh.wake_edges)
    mu = np.asarray(result.mu)
    gamma = -(mu[we[:, 0]] - mu[we[:, 1]])
    ya = np.asarray(mesh.nodes)[we[:, 2], 1]
    yb = np.asarray(mesh.nodes)[we[:, 3], 1]
    v_inf = float(_COND.V_inf)
    lift = v_inf * gamma * np.abs(yb - ya)
    q = 0.5 * v_inf * v_inf
    cl = float(np.sum(lift) / (q * _S_REF))
    roll = float(-np.sum(0.5 * (ya + yb) * lift) / (q * _S_REF * _B_REF))
    return cl, roll


def _vlm_totals(controls, n_chord: int, n_span: int = 10):
    """VLM totals of the same planform and controls."""
    ac = Aircraft(name="A", surfaces=[_plain_wing(controls)], ref_point=np.zeros(3))
    settings = SolverSettings(solver_type="vlm", n_panels=n_span, n_chord=n_chord)
    return VortexLatticeSolver().solve(ac, _COND, settings).totals


def test_side_closure_self_influence_and_planes() -> None:
    """Every panel of a flap and an aileron mesh has the self term +1/2.

    An invalid side closure (bow-tie or non-convex quadrilateral) puts the
    collocation point outside its panel, so the self term of the doublet
    potential drops from +1/2 to 0. The meshes use a library section with
    a blunt trailing edge (base panels kept), and the default meshes (the
    trailing edge closed).
    """
    coords = airfoil("0012").coords
    options = ({"trailing_edge": "base"}, {})
    for controls in (_flap(10.0), _flap(-20.0), _aileron(5.0)):
        for kw in options:
            mesh = mesh_lifting_surface(_plain_wing(controls), coords, n_chord=12, n_span=10, **kw)
            info = check_mesh(mesh)
            assert info["watertight"] and info["outward"] and info["valid_panels"]
            verts = np.asarray(mesh.nodes)[np.asarray(mesh.panels)]
            colloc = _precompute_geometry(verts).centroid
            _, c_mat = influence_matrices(colloc, verts, far_ratio=np.inf)
            diag = np.diag(c_mat)
            assert np.allclose(diag, 0.5, rtol=0.0, atol=1e-12), f"self terms {diag.min():.3f} to {diag.max():.3f}"
            assert np.allclose(np.sum(c_mat, axis=1), 1.0, rtol=0.0, atol=2e-2)


@pytest.mark.parametrize(
    ("section", "n_chord"),
    [
        ("0012", 12),
        pytest.param("2412", 12, marks=pytest.mark.slow),
        pytest.param("0012", 20, marks=pytest.mark.slow),
        pytest.param("2412", 20, marks=pytest.mark.slow),
    ],
)
def test_flap_and_aileron_magnitudes_match_vlm(section: str, n_chord: int) -> None:
    """Flap delta CL and aileron Cl agree with the VLM to 25 percent.

    The panel values come from the circulation of the wake strips (the
    potential solution on the mesh); the VLM uses the same planform,
    controls and n_chord. The condition number of the panel matrix stays
    below 1e4 (the invalid side closures gave 8.6e6). The pressure
    integration is checked in the two tests below.
    """
    coords = airfoil(section).coords
    res0, solver0, mesh0 = _panel_solve([], coords, n_chord)
    res1, solver1, mesh1 = _panel_solve(_flap(10.0), coords, n_chord)
    res2, solver2, mesh2 = _panel_solve(_aileron(5.0), coords, n_chord)
    for solver in (solver0, solver1, solver2):
        assert np.linalg.cond(solver.A) < 1e4
    cl0, _ = _circulation_coefficients(mesh0, res0)
    cl1, _ = _circulation_coefficients(mesh1, res1)
    _, roll = _circulation_coefficients(mesh2, res2)
    vlm0 = _vlm_totals([], n_chord)
    vlm1 = _vlm_totals(_flap(10.0), n_chord)
    vlm2 = _vlm_totals(_aileron(5.0), n_chord)
    d_panel = cl1 - cl0
    d_vlm = float(vlm1.CL - vlm0.CL)
    print(f"{section} n_chord {n_chord}: flap dCL panel {d_panel:.4f} vlm {d_vlm:.4f}; "
          f"aileron Cl panel {roll:.5f} vlm {float(vlm2.Cl):.5f}")
    assert abs(d_panel / d_vlm - 1.0) < 0.25
    assert abs(roll / float(vlm2.Cl) - 1.0) < 0.25


def test_flap_and_aileron_pressure_closed_trailing_edge() -> None:
    """Pressure-integrated flap delta CL and aileron Cl agree with the VLM to 25 percent.

    The section has a closed (sharp) trailing edge, so the mesh has no base
    panels.
    """
    coords = _closed_naca0012()
    res0, _, _ = _panel_solve([], coords, 12)
    res1, _, _ = _panel_solve(_flap(10.0), coords, 12)
    res2, _, _ = _panel_solve(_aileron(5.0), coords, 12)
    d_vlm = float(_vlm_totals(_flap(10.0), 12).CL - _vlm_totals([], 12).CL)
    roll_vlm = float(_vlm_totals(_aileron(5.0), 12).Cl)
    print(f"closed 0012: flap dCL panel {res1.CL - res0.CL:.4f} vlm {d_vlm:.4f}; "
          f"aileron Cl panel {res2.Cl:.5f} vlm {roll_vlm:.5f}")
    assert abs((res1.CL - res0.CL) / d_vlm - 1.0) < 0.25
    assert abs(res2.Cl / roll_vlm - 1.0) < 0.25


@pytest.mark.parametrize("section", ["0012", "2412"])
def test_flap_pressure_delta_cl_blunt_library_sections(section: str) -> None:
    """Pressure-integrated flap delta CL of library sections (blunt trailing edge) versus VLM."""
    coords = airfoil(section).coords
    res0, _, _ = _panel_solve([], coords, 12)
    res1, _, _ = _panel_solve(_flap(10.0), coords, 12)
    d_vlm = float(_vlm_totals(_flap(10.0), 12).CL - _vlm_totals([], 12).CL)
    assert abs((res1.CL - res0.CL) / d_vlm - 1.0) < 0.25


def test_aileron_full_mesh_is_mirror_image_outside_the_controls() -> None:
    """The full mesh of an antisymmetric control mirrors the right side correctly.

    The left half of the full mesh (built from the left tip to the right
    tip) uses the mirror image of the right-side section frames. With
    dihedral and twist, the nodes outside the control span are the mirror
    image of each other.
    """
    surf = LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.2, twist=np.radians(3.0)),
            WingSection(y_frac=1.0, chord=0.8, twist=np.radians(-2.0)),
        ],
        is_symmetric=True,
        dihedral=np.radians(8.0),
        sweep_le=np.radians(10.0),
        controls=[ControlSurface(name="aileron", eta_start=0.5, eta_end=0.9, hinge_x_c=0.7,
                                 deflection=np.radians(8.0), symmetric=False)],
    )
    mesh = mesh_lifting_surface(surf, airfoil("2412").coords, n_chord=10, n_span=10)
    info = check_mesh(mesh)
    assert info["watertight"] and info["outward"] and info["valid_panels"]
    nodes = np.asarray(mesh.nodes)
    inboard = nodes[np.abs(nodes[:, 1]) < 0.45 * 4.0 * np.cos(np.radians(8.0))]
    mirrored = inboard * np.array([1.0, -1.0, 1.0])
    dist, _ = cKDTree(inboard).query(mirrored)
    assert float(np.max(dist)) < 1e-12


def test_half_mesh_rejects_antisymmetric_controls() -> None:
    """A half mesh cannot carry an antisymmetric control deflection."""
    with pytest.raises(ValueError):
        mesh_lifting_surface(_plain_wing(_aileron(5.0)), _naca0012(), n_chord=8, n_span=6, half=True)


def test_control_mesh_strip_metadata() -> None:
    """Strip metadata of a flap mesh: one strip per spanwise panel, triangles keep their strip."""
    mesh = mesh_lifting_surface(_plain_wing(_flap(10.0)), airfoil("0012").coords, n_chord=12, n_span=10)
    strip = np.asarray(mesh.strip)
    side = np.asarray(mesh.side)
    kind = np.asarray(mesh.kind)
    chord = np.asarray(mesh.chord_index)
    n_strips = int(strip.max()) + 1
    assert n_strips == len(mesh.wake_edges)
    assert np.all(strip[kind == "side"] == -1) and np.all(side[kind == "side"] == 0)
    n_c = int(chord[side == 1].max()) + 1
    nodes = np.asarray(mesh.nodes)
    cen = nodes[np.asarray(mesh.panels)].mean(axis=1)
    y_mid = [float(np.mean(cen[strip == s, 1])) for s in range(n_strips)]
    assert np.all(np.diff(y_mid) > 0.0)
    for s in range(n_strips):
        for sd in (1, -1):
            assert set(chord[(strip == s) & (side == sd)].tolist()) == set(range(n_c))
    for w in np.asarray(mesh.wake_edges):
        assert strip[w[0]] == strip[w[1]] and side[w[0]] == 1 and side[w[1]] == -1


@pytest.mark.parametrize("section", ["0012", "2412", "4415"])
def test_control_meshes_valid_for_mixed_controls(section: str) -> None:
    """Adjacent and root controls with different hinges give valid closed meshes.

    The cases: a flap at the root, a flap next to an antisymmetric aileron
    with another hinge on a tapered, twisted, swept wing with dihedral, a
    V-tail elevator, and a rudder on a vertical fin and on its mirror copy.
    The 4415 library section has different upper and lower ordinates at
    x = 0 after the resampling; both sections of a side edge must still
    share the leading-edge node. Every self term is +1/2. Each case runs
    with the open trailing edge and base panels and with the default mesh
    (closed trailing edge).
    """

    def ctrl(a, b, hinge, deg, sym=True):
        return ControlSurface(name="c", eta_start=a, eta_end=b, hinge_x_c=hinge,
                              deflection=np.radians(deg), symmetric=sym)

    def wing(controls, symmetric=True, dihedral=0.0, sweep=0.0, twist=(0.0, 0.0), chord=(1.0, 1.0)):
        return LiftingSurface(
            name="S",
            semi_span=3.0,
            sections=[
                WingSection(y_frac=0.0, chord=chord[0], twist=np.radians(twist[0])),
                WingSection(y_frac=1.0, chord=chord[1], twist=np.radians(twist[1])),
            ],
            is_symmetric=symmetric,
            dihedral=np.radians(dihedral),
            sweep_le=np.radians(sweep),
            controls=controls,
        )

    fin = wing([ctrl(0.0, 0.8, 0.7, 20.0, False)], symmetric=False, dihedral=90.0, chord=(1.2, 0.8))
    cases = [
        wing([ctrl(0.0, 0.4, 0.7, 20.0)]),
        wing([ctrl(0.1, 0.5, 0.7, 15.0), ctrl(0.5, 0.9, 0.75, -10.0, False)],
             dihedral=6.0, sweep=15.0, twist=(3.0, -2.0), chord=(1.4, 0.7)),
        wing([ctrl(0.1, 0.9, 0.7, -12.0)], dihedral=35.0, twist=(1.0, 1.0)),
        fin,
        fin.mirrored("FinL"),
    ]
    coords = airfoil(section).coords
    for surf in cases:
        for kw in ({"trailing_edge": "base"}, {}):
            mesh = mesh_lifting_surface(surf, coords, n_chord=10, n_span=10, **kw)
            info = check_mesh(mesh)
            assert info["watertight"] and info["outward"] and info["valid_panels"], (surf.controls, kw)
            verts = np.asarray(mesh.nodes)[np.asarray(mesh.panels)]
            _, c_mat = influence_matrices(_precompute_geometry(verts).centroid, verts, far_ratio=np.inf)
            assert np.allclose(np.diag(c_mat), 0.5, rtol=0.0, atol=1e-12)
