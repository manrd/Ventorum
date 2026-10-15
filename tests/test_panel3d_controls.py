# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for deflected plain control surfaces in 3D panel meshes."""

from __future__ import annotations

import numpy as np

from ventorum.core.datatypes import Aircraft, ControlSurface, FlightCondition, LiftingSurface, SolverSettings, WingSection
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
