# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the 3D source-doublet panel solver (Ventorum-3DP).

Published sources
-----------------
* Morino, L. and Kuo, C. C., 1974, Subsonic potential aerodynamics for
  complex configurations: A general theory, AIAA Journal, Vol. 12, No. 2,
  pp. 191-197.
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, Chapters 9 and 12.
* Maskew, B., 1987, Program VSAERO Theory Document, NASA CR-4023.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from ventorum.aero.system import freestream_direction
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.panel3d.mesh import PanelMesh, mesh_lifting_surface
from ventorum.panel3d.post import build_surface_gradient_operator, surface_gradient
from ventorum.panel3d.solver import PanelResult, PanelSolver
from ventorum.reference.analytic3d import (
    ellipsoid_quad_mesh,
    ellipsoid_surface_velocity,
    sphere_cp,
    sphere_quad_mesh,
)
from ventorum.solvers.horseshoe import VortexLatticeSolver


def _naca0012_coordinates(n: int = 401) -> np.ndarray:
    """Selig ordered unit-chord coordinates of a NACA 0012 section."""
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * 0.12 * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )
    upper = np.column_stack([x, yt])[::-1]
    lower = np.column_stack([x, -yt])[1:]
    return np.vstack([upper, lower])


def _naca2412_coordinates(n: int = 101) -> np.ndarray:
    """Selig ordered unit-chord coordinates of a NACA 2412 section (closed trailing edge)."""
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * 0.12 * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )
    m_c, p_c = 0.02, 0.4
    yc = np.where(
        x < p_c,
        m_c / p_c**2 * (2.0 * p_c * x - x**2),
        m_c / (1.0 - p_c) ** 2 * ((1.0 - 2.0 * p_c) + 2.0 * p_c * x - x**2),
    )
    upper = np.column_stack([x, yc + yt])[::-1]
    lower = np.column_stack([x, yc - yt])[1:]
    return np.vstack([upper, lower])


def _naca0012_test_coordinates(n: int = 61) -> np.ndarray:
    """NACA 0012 coordinates of the stored-run tests (guarded square root)."""
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * 0.12 * (
        0.2969 * np.sqrt(np.maximum(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )
    upper = np.column_stack([x, yt])[::-1]
    lower = np.column_stack([x, -yt])[1:]
    return np.vstack([upper, lower])


def _rectangular_wing_surface(semi_span: float = 4.0, chord: float = 1.0) -> LiftingSurface:
    """Symmetric rectangular wing surface definition."""
    return LiftingSurface(
        name="Wing",
        semi_span=semi_span,
        sections=[
            WingSection(y_frac=0.0, chord=chord),
            WingSection(y_frac=1.0, chord=chord),
        ],
        is_symmetric=True,
    )


def _closed_mesh(nodes: np.ndarray, panels: np.ndarray) -> PanelMesh:
    """Panel mesh of a closed body without wake."""
    return PanelMesh(
        nodes=nodes,
        panels=panels,
        component=np.zeros(len(panels), dtype=int),
        kind=np.full(len(panels), "surface"),
        wake_edges=np.empty((0, 4), dtype=int),
    )


def test_sphere_cp_both_orientations_and_dalembert():
    """Verify the sphere Cp error with the flow across and along the pole axis.

    The latitude-longitude mesh has its poles on z. With the flow along x
    (across the pole axis) the poles sit at the points of highest speed,
    on fans of thin triangles; with the flow along z the poles sit at the
    stagnation points. The card limit (maximum Cp error below 0.03 at 32 x
    64, decreasing with refinement) must hold in both orientations, and the
    net force must stay below 1e-3 (d'Alembert). One solver serves both
    flow directions (unit-onset superposition).
    """
    flows = {
        "across the pole axis": FlightCondition(V_inf=1.0, alpha=0.0),
        "along the pole axis": FlightCondition(V_inf=1.0, alpha=np.pi / 2.0),
    }
    errors: dict[str, list[float]] = {name: [] for name in flows}
    for n_lat, n_lon in [(16, 32), (32, 64)]:
        nodes, panels = sphere_quad_mesh(1.0, n_lat, n_lon)
        solver = PanelSolver(_closed_mesh(nodes, panels), far_ratio=5.0)
        results = solver.solve_conditions(list(flows.values()))
        for (name, cond), res in zip(flows.items(), results):
            direction = freestream_direction(float(cond.alpha), 0.0)
            cp_exact = sphere_cp(solver.centroids, 1.0, direction)
            errors[name].append(float(np.max(np.abs(res.cp - cp_exact))))

            q_inf = 0.5 * float(cond.rho) * cond.V_inf**2
            f_vec = -np.sum(res.cp[:, None] * solver.areas[:, None] * solver.normals, axis=0) * q_inf
            net_force_norm = float(np.linalg.norm(f_vec))
            assert net_force_norm < 1.0e-3, (
                f"Sphere {n_lat}x{n_lon} ({name}) force exceeds 1e-3: {net_force_norm}"
            )

    for name, err in errors.items():
        print(f"sphere Cp max error, flow {name}: 16x32 {err[0]:.4f}, 32x64 {err[1]:.4f}")
        assert err[1] < err[0], f"Error did not decrease ({name}): {err[0]} to {err[1]}"
        assert err[1] < 0.03, f"Error at 32x64 not below 0.03 ({name}): {err[1]}"


def test_ellipsoid_oblique_surface_velocity():
    """Verify the surface speed on an ellipsoid at oblique onset on all panels.

    The card asks for 2 % away from the poles. The node-based surface
    gradient keeps the error below 2 % on every panel, including the
    polar triangle fans of the mesh (poles on z) and the panels next to
    the stagnation points, so the test uses no mask.
    """
    a, b, c = 1.0, 0.5, 0.25
    nodes, panels = ellipsoid_quad_mesh(a, b, c, 32, 64, spacing="cosine")
    solver = PanelSolver(_closed_mesh(nodes, panels), far_ratio=5.0)

    # Oblique onset flow
    V_stream = np.array([10.0, 3.0, -2.0])
    v_mag = float(np.linalg.norm(V_stream))
    alpha = float(np.arctan2(V_stream[2], V_stream[0]))
    beta = float(-np.arcsin(V_stream[1] / v_mag))
    cond = FlightCondition(V_inf=v_mag, alpha=alpha, beta=beta)
    res = solver.solve(cond)

    speed_num = np.linalg.norm(res.velocity, axis=-1)
    v_exact = ellipsoid_surface_velocity(solver.centroids, a, b, c, V_stream)
    speed_exact = np.linalg.norm(v_exact, axis=-1)

    rel_err = np.abs(speed_num - speed_exact) / speed_exact
    max_rel_err = float(np.max(rel_err))
    print(f"ellipsoid max relative speed error on all panels: {max_rel_err:.4f}")
    assert max_rel_err < 0.02, f"Ellipsoid relative speed error exceeds 2 %: {max_rel_err:.4f}"


def test_surface_gradient_operator_exact_and_cut_at_wake():
    """Verify the node-based gradient on a flat grid with a cut line.

    On a uniform flat grid the inverse-distance node values of a linear
    doublet field are exact at interior nodes, and the Green-Gauss rule is
    exact on a flat panel, so the gradient is exact on panels away from the
    border. A constant doublet jump across a wake shedding line must not
    enter the gradient of the panels next to the line.
    """
    nx, ny = 12, 10
    xs = np.linspace(0.0, 1.2, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

    def nid(i: int, j: int) -> int:
        return i * (ny + 1) + j

    panels = []
    for i in range(nx):
        for j in range(ny):
            panels.append([nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)])
    panels = np.asarray(panels, dtype=int)
    verts = nodes[panels]
    centroids = verts.mean(axis=1)
    normals = np.tile([0.0, 0.0, 1.0], (len(panels), 1))
    areas = np.full(len(panels), (xs[1] - xs[0]) * (ys[1] - ys[0]))

    def pid(i: int, j: int) -> int:
        return i * ny + j

    # Cut line x = xs[6] for 2 <= j < 8: panels of column 5 (left) and 6 (right).
    cut = [[pid(5, j), pid(6, j), nid(6, j), nid(6, j + 1)] for j in range(2, 8)]
    op = build_surface_gradient_operator(nodes, panels, centroids, normals, areas, np.asarray(cut))
    op_plain = build_surface_gradient_operator(nodes, panels, centroids, normals, areas, None)

    g_true = np.array([0.7, -1.3, 0.0])
    mu_lin = centroids @ g_true
    interior = [
        pid(i, j) for i in range(1, nx - 1) for j in range(1, ny - 1)
    ]
    grad_plain = surface_gradient(op_plain, mu_lin)
    assert np.max(np.abs(grad_plain[interior] - g_true)) < 1e-12

    # Doublet jump of 5 across the cut on the right side.
    mu_jump = mu_lin + np.where(centroids[:, 0] > xs[6], 5.0, 0.0)
    grad_cut = surface_gradient(op, mu_jump)
    near_cut = [pid(i, j) for i in (5, 6) for j in range(3, 7)]
    assert np.max(np.abs(grad_cut[near_cut] - g_true)) < 1e-12
    # Without the cut the jump enters the gradient (the test is not vacuous).
    grad_no_cut = surface_gradient(op_plain, mu_jump)
    assert np.max(np.abs(grad_no_cut[near_cut] - g_true)) > 1.0


def test_superposition_exact_match():
    """Verify superposition of unit solutions equals direct solve to 1e-10."""
    surf = _rectangular_wing_surface(semi_span=4.0, chord=1.0)
    coords = _naca0012_coordinates(101)
    mesh = mesh_lifting_surface(surf, coords, n_chord=10, n_span=10)
    solver = PanelSolver(mesh, far_ratio=np.inf)

    cond = FlightCondition(
        V_inf=35.0,
        alpha=np.radians(3.5),
        beta=np.radians(-2.0),
        p=0.1,
        q=-0.05,
        r=0.08,
    )
    res_super = solver.solve(cond)

    # Direct solve with condition onset flow
    v_dir = solver._condition_velocity(cond, solver.centroids)
    sigma_direct = -np.sum(solver.normals * v_dir, axis=-1)
    rhs_direct = -solver.B @ sigma_direct
    mu_direct = np.linalg.solve(solver.A, rhs_direct)

    max_mu_diff = float(np.max(np.abs(res_super.mu - mu_direct)))
    assert max_mu_diff < 1.0e-10, f"Superposition mu difference exceeds 1e-10: {max_mu_diff}"


def test_rectangular_wing_ar8_naca0012():
    """Verify rectangular wing lift, symmetry, and Trefftz induced drag."""
    surf = _rectangular_wing_surface(semi_span=4.0, chord=1.0)
    coords = _naca0012_coordinates(201)
    mesh = mesh_lifting_surface(surf, coords, n_chord=20, n_span=20)
    solver = PanelSolver(mesh, far_ratio=np.inf)

    # 1. Zero lift at zero angle of attack
    cond0 = FlightCondition(V_inf=50.0, alpha=0.0)
    res0 = solver.solve(cond0)
    assert abs(res0.CL) < 1.0e-10, f"CL(0) is not below 1e-10: {res0.CL}"

    # 2. Lift at 4 degrees compared against VLM of the same planform
    cond4 = FlightCondition(V_inf=50.0, alpha=np.radians(4.0))
    res4 = solver.solve(cond4)

    ac = Aircraft(surfaces=[surf])
    st = SolverSettings(n_panels=20, n_chord=4)
    vlm_res = VortexLatticeSolver().solve(ac, cond4, st)
    cl_vlm = float(vlm_res.totals.CL)
    cdi_vlm = float(vlm_res.totals.CDi)

    print(f"Panel CL(4): {res4.CL:.6f}, VLM CL(4): {cl_vlm:.6f}")
    # Thickness raises lift slope slightly (0 to 10 % above VLM)
    assert res4.CL > cl_vlm, f"Panel CL ({res4.CL}) not above VLM CL ({cl_vlm})"
    assert res4.CL < 1.10 * cl_vlm, f"Panel CL ({res4.CL}) exceeds 10 % over VLM CL ({cl_vlm})"

    # 3. Equal lift on left and right halves
    q_inf = 0.5 * cond4.rho * cond4.V_inf**2
    dF = -res4.cp[:, None] * solver.areas[:, None] * solver.normals * q_inf
    l_dir = np.array([-np.sin(cond4.alpha), 0.0, np.cos(cond4.alpha)])
    lift_per_panel = dF @ l_dir
    right_mask = solver.centroids[:, 1] > 0
    left_mask = solver.centroids[:, 1] < 0
    l_right = float(np.sum(lift_per_panel[right_mask]))
    l_left = float(np.sum(lift_per_panel[left_mask]))
    half_sref = 0.5 * solver.S_ref
    cl_right = l_right / (q_inf * half_sref)
    cl_left = l_left / (q_inf * half_sref)
    assert abs(cl_right - cl_left) < 1.0e-10, f"Half wing lift mismatch: {abs(cl_right - cl_left)}"

    # 4. Trefftz plane induced drag positive and within 15 % of VLM CDi
    assert res4.CDi_trefftz > 0.0, f"Trefftz drag is not positive: {res4.CDi_trefftz}"
    assert abs(res4.CDi_trefftz - cdi_vlm) / cdi_vlm < 0.15, (
        f"CDi_trefftz ({res4.CDi_trefftz}) not within 15 % of VLM CDi ({cdi_vlm})"
    )


def test_unit_onset_linearity_of_doublets_and_wake_lift():
    """Verify that mu and the wake-circulation lift are linear in the onset.

    The card asked for the pressure-integrated CL linear in alpha to
    1e-10. That is not possible: Cp is quadratic in the surface velocity,
    so the pressure lift has terms in the square of the onset. With the
    body-axis wake the linear part of the solution is exact: the doublets
    are a linear function of the onset components (u, w) = V (cos(alpha),
    sin(alpha)), and so is the Kutta-Joukowski lift of the wake
    circulation (Trefftz plane), CL_KJ = 2 sum(Gamma_s dy_s) / (V S).
    The test uses a cambered section, so that both onset components carry
    circulation, and checks that the solution at alpha 6 deg is the linear
    combination of the solutions at 2 and 4 deg with the weights of the
    onset vectors, to round-off.
    """
    surf = _rectangular_wing_surface(semi_span=4.0, chord=1.0)
    mesh = mesh_lifting_surface(surf, _naca2412_coordinates(101), n_chord=10, n_span=10)
    solver = PanelSolver(mesh, far_ratio=np.inf)
    v_inf = 50.0
    angles = np.radians([2.0, 4.0, 6.0])
    results = solver.solve_conditions([FlightCondition(V_inf=v_inf, alpha=float(a)) for a in angles])

    onset = np.array([[np.cos(a), np.sin(a)] for a in angles]) * v_inf
    weights = np.linalg.solve(onset[:2].T, onset[2])

    mu = [r.mu for r in results]
    mu_pred = weights[0] * mu[0] + weights[1] * mu[1]
    scale = float(np.max(np.abs(mu[2])))
    assert np.max(np.abs(mu[2] - mu_pred)) < 1.0e-12 * scale

    wake = solver.wake
    dy = wake.p_right[:, 1] - wake.p_left[:, 1]

    def cl_kj(res: PanelResult) -> float:
        gamma = res.mu[wake.wake_up] - res.mu[wake.wake_lo]
        return float(2.0 * np.sum(gamma * dy) / (v_inf * solver.S_ref))

    cl = [cl_kj(r) for r in results]
    cl_pred = weights[0] * cl[0] + weights[1] * cl[1]
    assert abs(cl[2] - cl_pred) < 1.0e-12 * abs(cl[2])
    # The circulation of the cambered wing has a part from u (not vacuous).
    assert abs(cl[0] / np.sin(angles[0]) - cl[1] / np.sin(angles[1])) > 1.0e-3 * abs(cl[1])


def test_sideslip_sign_change():
    """Verify lateral coefficients CY, Cl, Cn change sign with beta."""
    surf = _rectangular_wing_surface(semi_span=4.0, chord=1.0)
    coords = _naca0012_coordinates(101)
    mesh = mesh_lifting_surface(surf, coords, n_chord=10, n_span=10)
    solver = PanelSolver(mesh, far_ratio=np.inf)

    cond_pos = FlightCondition(V_inf=50.0, alpha=np.radians(2.0), beta=np.radians(5.0))
    cond_neg = FlightCondition(V_inf=50.0, alpha=np.radians(2.0), beta=np.radians(-5.0))

    res_pos = solver.solve(cond_pos)
    res_neg = solver.solve(cond_neg)

    assert abs(res_pos.CY + res_neg.CY) < 1.0e-10, f"CY sign change failed: {res_pos.CY + res_neg.CY}"
    assert abs(res_pos.Cl + res_neg.Cl) < 1.0e-10, f"Cl sign change failed: {res_pos.Cl + res_neg.Cl}"
    assert abs(res_pos.Cn + res_neg.Cn) < 1.0e-10, f"Cn sign change failed: {res_pos.Cn + res_neg.Cn}"
    assert abs(res_pos.Cm - res_neg.Cm) < 1.0e-10, f"Cm symmetric check failed: {res_pos.Cm - res_neg.Cm}"


def test_solve_conditions_batch():
    """Verify solve_conditions returns consistent list of PanelResult."""
    surf = _rectangular_wing_surface(semi_span=3.0, chord=1.0)
    coords = _naca0012_coordinates(101)
    mesh = mesh_lifting_surface(surf, coords, n_chord=8, n_span=8)
    solver = PanelSolver(mesh, far_ratio=np.inf)

    conditions = [
        FlightCondition(V_inf=40.0, alpha=np.radians(1.0)),
        FlightCondition(V_inf=40.0, alpha=np.radians(3.0)),
        FlightCondition(V_inf=40.0, alpha=np.radians(5.0)),
    ]
    results = solver.solve_conditions(conditions)
    assert len(results) == 3
    for i, res in enumerate(results):
        assert isinstance(res, PanelResult)
        res_single = solver.solve(conditions[i])
        assert np.isclose(res.CL, res_single.CL, atol=1.0e-12)
        assert np.isclose(res.CD_pressure, res_single.CD_pressure, atol=1.0e-12)
        assert np.isclose(res.CDi_trefftz, res_single.CDi_trefftz, atol=1.0e-12)


# Values of a run made before the ground-effect wake change (free air, body
# axis wake). They do not depend on the surface gradient method.
_STORED_FREE_AIR = {
    False: {
        "n": 272,
        "mu": {0: 2.450267924624242, 7: 3.315504490990303, 37: 3.7879630323326454, 271: -0.6988622766055512},
        "mu_sum": -12.476474280438861,
        "mu_abs_sum": 783.0904197027365,
        "CDi": 0.004233828611102339,
        "sigma_abs_sum": 2285.451549319791,
        "wake_sum": 1664.0,
    },
    True: {
        "n": 136,
        "mu": {0: 2.43323004857819, 7: 3.3380171846878697, 37: 3.9010646242554325, 135: -0.5872020721156217},
        "mu_sum": 0.1197936848349741,
        "mu_abs_sum": 406.78969327039033,
        "CDi": 0.0042244555972296045,
        "sigma_abs_sum": 1178.531864830276,
        "wake_sum": 913.2253631008709,
    },
}


@pytest.mark.parametrize("half", [False, True])
def test_free_air_matches_stored_run(half: bool):
    """Verify that the free-air solution equals a run stored before the wake change.

    The ground-effect wake now follows the free stream; the free-air wake
    stays on the body x axis. The doublets, the sources, the wake geometry
    and the Trefftz drag must not change. On the machine that stored the
    values they are equal bit for bit; the test allows 1e-12 relative,
    because other LAPACK and BLAS builds can round the last bits
    differently. The half mesh case has sideslip and rates, so it uses
    both the symmetric and the antisymmetric systems.
    """
    stored = _STORED_FREE_AIR[half]
    surf = _rectangular_wing_surface(semi_span=4.0, chord=1.0)
    mesh = mesh_lifting_surface(surf, _naca0012_test_coordinates(61), n_chord=8, n_span=8, half=half)
    solver = PanelSolver(mesh, far_ratio=np.inf)
    cond = FlightCondition(
        V_inf=50.0,
        alpha=np.radians(4.0),
        beta=np.radians(5.0) if half else 0.0,
        p=0.1 if half else 0.0,
        r=0.05 if half else 0.0,
    )
    res = solver.solve(cond)
    assert len(res.mu) == stored["n"]
    for idx, value in stored["mu"].items():
        assert res.mu[idx] == pytest.approx(value, rel=1e-12, abs=0.0)
    tol_sum = 1e-12 * stored["mu_abs_sum"]
    assert abs(float(np.sum(res.mu)) - stored["mu_sum"]) <= tol_sum
    assert float(np.sum(np.abs(res.mu))) == pytest.approx(stored["mu_abs_sum"], rel=1e-12)
    assert res.CDi_trefftz == pytest.approx(stored["CDi"], rel=1e-12)
    assert float(np.sum(np.abs(res.sigma))) == pytest.approx(stored["sigma_abs_sum"], rel=1e-12)
    assert float(np.sum(solver.wake.panels)) == pytest.approx(stored["wake_sum"], rel=1e-12)


def test_default_wake_length_is_fifty_largest_chords():
    """Verify the default wake length: 50 times the largest streamwise chord.

    A swept wing has a mesh x extent much larger than its chord, so the
    test tells the two definitions apart.
    """
    surf = LiftingSurface(
        name="Swept",
        semi_span=3.0,
        sections=[WingSection(y_frac=0.0, chord=1.0), WingSection(y_frac=1.0, chord=0.6)],
        sweep_le=np.radians(30.0),
        is_symmetric=True,
    )
    mesh = mesh_lifting_surface(surf, _naca0012_coordinates(101), n_chord=6, n_span=6, half=True)
    x_extent = float(np.ptp(mesh.nodes[:, 0]))
    solver = PanelSolver(mesh, far_ratio=np.inf)
    assert x_extent > 2.0
    assert solver.wake_length == pytest.approx(50.0 * 1.0, rel=1e-12)
    lengths = np.linalg.norm(solver.wake.panels[:, 1] - solver.wake.panels[:, 0], axis=1)
    assert np.allclose(lengths, 50.0, rtol=1e-12)
    explicit = PanelSolver(mesh, far_ratio=np.inf, wake_length=7.5)
    assert explicit.wake_length == 7.5


def test_factorizations_are_lazy_and_counted(monkeypatch):
    """Verify that the solver factorizes only when needed and counts it.

    Construction builds no system. A free-air sweep on a full mesh needs
    one factorization. A symmetric sweep on a half mesh needs only the
    symmetric system; sideslip adds the antisymmetric one. The count of
    the solver equals the count of a wrapper around the LU routine.
    """
    calls = {"n": 0}
    original = scipy.linalg.lu_factor

    def counting(a, *args, **kwargs):
        calls["n"] += 1
        return original(a, *args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lu_factor", counting)
    surf = _rectangular_wing_surface(semi_span=3.0, chord=1.0)
    coords = _naca0012_coordinates(61)
    sweep = [FlightCondition(V_inf=30.0, alpha=np.radians(a)) for a in (-2.0, 0.0, 2.0, 4.0, 6.0)]

    full = PanelSolver(mesh_lifting_surface(surf, coords, n_chord=6, n_span=6), far_ratio=np.inf)
    assert calls["n"] == 0
    assert full.factorization_count == 0
    full.solve_conditions(sweep)
    full.solve(FlightCondition(V_inf=30.0, alpha=0.1, beta=0.05, p=0.2, r=0.1))
    assert calls["n"] == 1
    assert full.factorization_count == 1

    calls["n"] = 0
    half = PanelSolver(
        mesh_lifting_surface(surf, coords, n_chord=6, n_span=6, half=True), far_ratio=np.inf
    )
    assert calls["n"] == 0
    half.solve_conditions(sweep)
    assert calls["n"] == 1
    assert half.factorization_count == 1
    half.solve(FlightCondition(V_inf=30.0, alpha=0.05, beta=np.radians(5.0)))
    half.solve(FlightCondition(V_inf=30.0, alpha=0.05, r=0.3))
    assert calls["n"] == 2
    assert half.factorization_count == 2
