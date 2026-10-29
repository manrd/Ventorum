# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Surface pressure of the 3D panel method with body rates p, q, r.

With body rates the flow is steady in the body frame but not in the earth
frame. The unsteady Bernoulli equation in the earth frame gives
``Cp = (|V_k|^2 - |V|^2) / V_inf^2``, with ``V_k`` the local onset velocity
(free stream minus Omega x (x - x_ref)) and ``V`` the surface velocity
relative to the body. Without rates ``|V_k| = V_inf`` and the formula is the
usual ``Cp = 1 - |V|^2 / V_inf^2``.

Checks:

* a sphere of radius a that flies a circle (pitch rate q or yaw rate r about
  its centre): the only force is the added-mass force
  ``F = -m_a dU/dt`` with ``m_a = (2/3) pi rho a^3`` and the centripetal
  acceleration ``|dU/dt| = Omega V``. With ``S_ref = pi a^2``, the
  coefficient of the force towards the centre of the turn is
  ``-(4/3) a Omega / V``;
* the Cp of each solver path (Morino batch, single condition, pressure
  Kutta, gpu) satisfies the formula above;
* results without rates are unchanged;
* the yaw-rate derivatives CY_r and Cn_r of a rectangular wing agree with
  the vortex lattice (stability axes);
* the viscous strip coupling uses the same Cp.

The derivation of the formula and of the sphere force is in the theory
manual (docs/theory/ventorum_3dp.md, pressure with body rates).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.axes import rotation_body_to_stability
from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    LinearAirfoil,
    SolverSettings,
    WingSection,
)
from ventorum.panel3d.mesh import PanelMesh, mesh_lifting_surface
from ventorum.panel3d.post import _surface_velocity_batch, compute_surface_velocity_and_cp
from ventorum.panel3d.solver import PanelSolver
from ventorum.reference.analytic3d import sphere_quad_mesh
from ventorum.solvers.rate_derivatives import rate_axes_transform


@pytest.fixture(autouse=True)
def _cpu():
    """Select the CPU, and restore the previous device and precision after the test."""
    from ventorum import gpu

    old = gpu.get_device()
    old_prec = gpu._precision
    gpu.set_device("cpu")
    yield
    gpu.set_device(old)
    gpu.set_precision(old_prec)


def _sphere_solver(n_lat: int = 16, radius: float = 1.0, **kw) -> PanelSolver:
    """Panel solver of a sphere centred at the origin, S_ref = pi a^2 [m^2]."""
    nodes, panels = sphere_quad_mesh(radius, n_lat, 2 * n_lat)
    mesh = PanelMesh(
        nodes=nodes,
        panels=panels,
        component=np.zeros(len(panels), dtype=int),
        kind=np.full(len(panels), "surface"),
        wake_edges=np.empty((0, 4), dtype=int),
    )
    return PanelSolver(mesh, S_ref=np.pi * radius**2, b_ref=2.0 * radius,
                       c_ref=2.0 * radius, **kw)


def _naca0012_coordinates(n: int = 41) -> np.ndarray:
    """Selig ordered unit-chord coordinates of a NACA 0012 section."""
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


def _wing_solver(kutta: str = "morino", half: bool = True, **kw) -> PanelSolver:
    """Small rectangular NACA 0012 wing (span 6 m, chord 1 m), quarter-chord reference."""
    surface = LiftingSurface(
        name="Wing", semi_span=3.0,
        sections=[WingSection(y_frac=0.0, chord=1.0), WingSection(y_frac=1.0, chord=1.0)],
        is_symmetric=True,
    )
    mesh = mesh_lifting_surface(surface, _naca0012_coordinates(), n_chord=6, n_span=6, half=half)
    return PanelSolver(mesh, S_ref=6.0, b_ref=6.0, c_ref=1.0,
                       ref_point=np.array([0.25, 0.0, 0.0]), kutta=kutta, **kw)


# ---------------------------------------------------------------------------
# 1. Sphere on a circular path: added-mass force
# ---------------------------------------------------------------------------

#: Mesh tolerance of the sphere force [-], relative. The force comes from
#: the kinematic pressure term and the surface velocity; the 16 x 32 mesh
#: (512 panels) gives -0.06530 for -0.06667 (2.1 percent) and the 24 x 48
#: mesh gives 0.9 percent, so the error falls with the mesh.
SPHERE_TOLERANCE = 0.03


@pytest.mark.parametrize("rate", ["q", "r"])
def test_sphere_on_a_circle_has_the_added_mass_force(rate):
    """Pitch or yaw rate: the force coefficient towards the turn centre is -(4/3) a Omega / V."""
    radius, speed, omega = 1.0, 1.0, 0.05
    solver = _sphere_solver(16, radius)
    res = solver.solve(FlightCondition(V_inf=speed, alpha=0.0, **{rate: omega}))
    exact = -4.0 / 3.0 * radius * omega / speed
    # A pitch rate turns the path up (+z, lift axis); a yaw rate turns it
    # to the right (+y, side axis at alpha = beta = 0).
    value = res.CL if rate == "q" else res.CY
    assert value == pytest.approx(exact, rel=SPHERE_TOLERANCE)
    # The other force components are zero by symmetry.
    other = res.CY if rate == "q" else res.CL
    assert abs(other) < 1e-3 * abs(exact)
    assert abs(res.CD_pressure) < 1e-3 * abs(exact)


def test_sphere_force_is_linear_in_the_rate():
    """The added-mass force is linear in the rate: twice the rate gives twice the force."""
    solver = _sphere_solver(12)
    one = solver.solve(FlightCondition(V_inf=1.0, q=0.05)).CL
    two = solver.solve(FlightCondition(V_inf=1.0, q=0.10)).CL
    assert two == pytest.approx(2.0 * one, rel=1e-9)


# ---------------------------------------------------------------------------
# 2. The Cp of each solver path satisfies the rotating-body formula
# ---------------------------------------------------------------------------

def _expected_cp(solver: PanelSolver, res, condition: FlightCondition) -> np.ndarray:
    """Return (|V_k|^2 - |V|^2) / V_inf^2 [-] on the solved panels."""
    onset = solver._condition_velocity(condition, solver.centroids)
    return (np.sum(onset**2, axis=1) - np.sum(res.velocity**2, axis=1)) / condition.V_inf**2


_RATE_CONDITIONS = [
    FlightCondition(V_inf=20.0, alpha=np.radians(3.0), q=0.4),
    FlightCondition(V_inf=20.0, alpha=np.radians(3.0), p=0.5, r=0.3),
]


@pytest.mark.parametrize("kutta", ["morino", "pressure"])
@pytest.mark.parametrize("condition", _RATE_CONDITIONS, ids=["q", "p_r"])
def test_cp_of_the_solver_paths_includes_the_rotation(kutta, condition):
    """Morino batch, single-condition Morino and pressure Kutta: Cp = (|V_k|^2 - |V|^2) / V_inf^2."""
    solver = _wing_solver(kutta)
    res = solver.solve(condition)
    np.testing.assert_allclose(res.cp, _expected_cp(solver, res, condition), rtol=0, atol=1e-12)
    if kutta == "morino":
        key, _, _ = solver._config_key(condition, None)
        single = solver._evaluate_result(solver._configs[key], condition)
        np.testing.assert_allclose(single.cp, _expected_cp(solver, single, condition),
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(single.cp, res.cp, rtol=0, atol=1e-10)


def test_pressure_kutta_closes_on_the_rotating_pressure():
    """With a pitch rate the pressure Kutta sets the jump of the rotating-body Cp to zero."""
    solver = _wing_solver("pressure")
    condition = _RATE_CONDITIONS[0]
    res = solver.solve(condition)
    assert res.kutta_converged
    assert res.kutta_residual < 1e-6
    np.testing.assert_allclose(res.cp, _expected_cp(solver, res, condition), rtol=0, atol=1e-12)


def test_batch_cp_mixes_rate_and_rate_free_conditions():
    """One batch with and without rates: each row uses its own formula."""
    rng = np.random.default_rng(3)
    onset = rng.normal(size=(2, 5, 3))
    normals = rng.normal(size=(5, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    gradient = rng.normal(size=(2, 5, 3))
    speed = np.array([2.0, 3.0])
    rates = np.array([False, True])
    velocity, cp = _surface_velocity_batch(onset, normals, gradient, speed, rates)
    for k in range(2):
        v_k, cp_k = compute_surface_velocity_and_cp(onset[k], normals, gradient[k], speed[k],
                                                    bool(rates[k]))
        np.testing.assert_array_equal(velocity[k], v_k)
        np.testing.assert_allclose(cp[k], cp_k, rtol=1e-14, atol=1e-14)
    expected = (np.sum(onset[1] ** 2, axis=1) - np.sum(velocity[1] ** 2, axis=1)) / 9.0
    np.testing.assert_allclose(cp[1], expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(cp[0], 1.0 - np.sum(velocity[0] ** 2, axis=1) / 4.0,
                               rtol=1e-14, atol=1e-14)


class _ConvergedReferencePolar(LinearAirfoil):
    """Polar that is the converged inviscid 2D reference curve of one strip (exact, no table)."""

    def __init__(self, curve):
        super().__init__(a0=2 * np.pi)
        self.curve = curve

    def Cl(self, angle):
        return np.vectorize(lambda a: self.curve.evaluate(a)[0], otypes=[float])(
            np.asarray(angle, dtype=float))


@pytest.mark.parametrize("kutta", ["morino", "pressure"])
def test_viscous_coupling_uses_the_rotating_pressure(kutta, monkeypatch):
    """Identity polar with a pitch rate: the viscous coupling gives the inviscid Cp and loads."""
    from ventorum.solvers.factory import make_solver
    from ventorum.viscous import alpha_correction as module
    from ventorum.viscous.alpha_correction import solve_viscous

    # The 3D panel coupling reads a polar relative to the converged inviscid
    # 2D section of each strip (defect correction: polar - converged section
    # + same-mesh section). The identity polar is therefore the converged
    # section curve. The same-mesh section line that this test used before
    # now gets the mesh-error correction, so it is not an identity polar.
    captured = {}
    original = module._PanelCoupling.references

    def record(self, ground, uses_height, curves):
        refs = original(self, ground, uses_height, curves)
        captured["refs"] = list(self.full(np.array(refs, dtype=object)))
        return refs

    monkeypatch.setattr(module._PanelCoupling, "references", record)

    wing = LiftingSurface(
        name="Wing", semi_span=4.0, spacing="uniform",
        sections=[WingSection(y_frac=0, chord=1, shape="0012"),
                  WingSection(y_frac=1, chord=1, shape="0012")],
    )
    settings = SolverSettings(n_panels=3, n_chord=10, spacing="uniform", panel_backend="numpy",
                              panel_cache=False, panel_kutta=kutta)
    condition = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), q=1.5)
    solve_viscous(wing, condition, settings, LinearAirfoil(a0=5.0), model="panel3d", tol=1e-10)
    polars = {i: _ConvergedReferencePolar(curve) for i, curve in enumerate(captured["refs"])}
    result = solve_viscous(wing, condition, settings, polars, model="panel3d", tol=1e-10)
    inviscid = make_solver("panel3d").solve(wing, condition, settings)
    assert result.converged
    np.testing.assert_allclose(result.details["cp"], inviscid.details["cp"], rtol=1e-8, atol=1e-9)
    for key in ("CL", "CY", "Cl", "Cn"):
        assert getattr(result.totals, key) == pytest.approx(getattr(inviscid.totals, key), abs=1e-8)


# ---------------------------------------------------------------------------
# 3. Results without rates are unchanged
# ---------------------------------------------------------------------------

def _baseline_wing() -> Aircraft:
    secs = [WingSection(0.0, 1.0, x_le=-0.25, shape="0012"),
            WingSection(1.0, 0.6, x_le=0.0, shape="0012")]
    return Aircraft(surfaces=[LiftingSurface(semi_span=3.0, sections=secs)])


_KEYS = ("CL", "CDi_nearfield", "CY", "Cl", "Cm", "Cn")

#: Values without rates from the code before the rotating-body pressure
#: (commit 42f56a8, "Merge the control-point limit warning and a docstring
#: fix"): (CL, CDi_nearfield, CY, Cl, Cm, Cn), the sum of the panel Cp
#: values and the Cp of panel 3. Numpy backend, 6 x 6 panels. Updated for
#: the second-moment far-field expansion about the panel centroid (the old
#: expansion had an error of 1e-4 to 3e-4 relative on this mesh).
_BASELINE = {
    "morino_sym": ("morino", dict(alpha=4.0), (
        0.3327788051124851, -0.0032292424761471723, 1.5436973789198368e-18,
        1.4265904322469885e-18, -0.019742375535250867, -1.683531869243679e-19),
        -11.504605920145679, -0.08153225923135698),
    "morino_beta": ("morino", dict(alpha=4.0, beta=3.0), (
        0.3318673196341243, -0.003217419828229417, -0.00014471841447228097,
        -0.0011212561304619812, -0.01968830013581702, 6.641381625027584e-05),
        -22.92112840700115, -0.08138541221111817),
    "pressure_beta": ("pressure", dict(alpha=4.0, beta=3.0), (
        0.31517233171464176, -0.003622569633286369, -0.0001704076842781566,
        -0.000561965751579083, -0.014654001302152483, 8.118348757454537e-05),
        -22.707546992378646, -0.09000901968162722),
    "morino_ground": ("morino", dict(alpha=4.0, h=0.8), (
        0.3717514464133529, -0.0038660126844525583, 1.8567884144437119e-19,
        -3.503538267430104e-18, -0.023931779682072805, 1.0329354446645879e-18),
        -10.700572955622386, -0.022733656983591644),
}


@pytest.mark.parametrize("case", list(_BASELINE))
def test_results_without_rates_are_unchanged(case):
    """Without rates the coefficients and Cp agree with the stored values to round-off."""
    kutta, angles, totals, cp_sum, cp3 = _BASELINE[case]
    condition = FlightCondition(
        V_inf=30.0, alpha=np.radians(angles["alpha"]), beta=np.radians(angles.get("beta", 0.0)),
        h=angles.get("h"),
    )
    settings = SolverSettings(solver_type="panel3d", n_panels=6, n_chord=6, panel_kutta=kutta,
                              panel_backend="numpy", panel_cache=False)
    res = vt.analyze(_baseline_wing(), condition=condition, settings=settings)
    got = tuple(float(getattr(res.totals, k)) for k in _KEYS)
    cp = np.asarray(res.details["cp"])
    # Bit-identical on the machine that stored the values; other numpy and BLAS
    # builds (CI) change the dense solve in the last digits (about 3e-12 seen).
    np.testing.assert_allclose(got, totals, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose([float(np.sum(cp)), float(cp[3])], [cp_sum, cp3],
                               rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Yaw-rate derivatives of a wing agree with the vortex lattice
# ---------------------------------------------------------------------------

#: Agreement of the 3DP and VLM yaw-rate derivatives [-], relative. The
#: 3DP wing is thick (NACA 0012) and the lattice is thin, and Cn_r is a
#: small difference of near-field pressure drag terms, so only a loose
#: agreement is expected. On the 16 x 16 mesh: CY_r -0.00307 against
#: -0.00289 and Cn_r -0.00156 against -0.00170.
YAW_TOLERANCE = 0.25


def _rate_matrix(aircraft, condition, settings, step=0.01):
    """Body-axis rate derivatives [-] (6 coefficients x p, q, r), central differences."""
    b_ref, c_ref, speed = 8.0, 1.0, condition.V_inf
    to_rate = {"p": 2 * speed / b_ref, "r": 2 * speed / b_ref}
    matrix = np.zeros((6, 3))
    for j, name in ((0, "p"), (2, "r")):
        values = []
        for sign in (1.0, -1.0):
            cond = dataclasses.replace(condition, **{name: sign * step * to_rate[name]})
            t = vt.analyze(aircraft, condition=cond, settings=settings).totals
            values.append(np.array([t.CL, t.CDi, t.CY, t.Cl, t.Cm, t.Cn]))
        matrix[:, j] = (values[0] - values[1]) / (2.0 * step)
    return rate_axes_transform(matrix, rotation_body_to_stability(condition.alpha), b_ref, c_ref)


@pytest.mark.slow
def test_yaw_rate_derivatives_agree_with_the_vortex_lattice():
    """AR 8 NACA 0012 wing, alpha 4 deg, stability axes: CY_r and Cn_r of 3DP and VLM agree.

    Both are negative: at positive lift the outer (advancing) wing of the
    turn has more lift and more induced drag, which yaws the nose out of
    the turn (yaw damping, Cn_r < 0).
    """
    secs = [WingSection(0.0, 1.0, x_le=-0.25, shape="0012"),
            WingSection(1.0, 1.0, x_le=-0.25, shape="0012")]
    aircraft = Aircraft(surfaces=[LiftingSurface(semi_span=4.0, sections=secs)])
    condition = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    panel = _rate_matrix(aircraft, condition,
                         SolverSettings(solver_type="panel3d", n_panels=16, n_chord=16))
    lattice = _rate_matrix(aircraft, condition,
                           SolverSettings(solver_type="vlm", n_panels=40, n_chord=8))
    for row, name in ((2, "CY_r"), (5, "Cn_r")):
        p3, vlm = panel[row, 2], lattice[row, 2]
        assert vlm < 0.0 and p3 < 0.0, f"{name}: 3DP {p3:.5f}, VLM {vlm:.5f}"
        assert p3 == pytest.approx(vlm, rel=YAW_TOLERANCE), f"{name}: 3DP {p3:.5f}, VLM {vlm:.5f}"


# ---------------------------------------------------------------------------
# 5. GPU paths
# ---------------------------------------------------------------------------

def _need_cuda():
    """Skip the test when a CUDA GPU, PyTorch or Warp is missing."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is not available.")
    return torch


@pytest.mark.gpu
@pytest.mark.parametrize("condition", _RATE_CONDITIONS + [
    FlightCondition(V_inf=20.0, alpha=np.radians(3.0))], ids=["q", "p_r", "none"])
def test_gpu_float64_rate_pressure_matches_numpy(condition):
    """Gpu float64 (batch and single-condition paths) matches numpy to 1e-10 with rates."""
    torch = _need_cuda()
    ref = _wing_solver(backend="numpy")
    gpu = _wing_solver(backend="gpu", dtype=torch.float64)
    res_ref, res_gpu = ref.solve(condition), gpu.solve(condition)
    np.testing.assert_allclose(res_gpu.cp, res_ref.cp, rtol=1e-10, atol=1e-10)
    for name in ("CL", "CY", "Cl", "Cm", "Cn"):
        assert getattr(res_gpu, name) == pytest.approx(getattr(res_ref, name), rel=1e-10, abs=1e-12)
    key, _, _ = gpu._config_key(condition, None)
    single = gpu._evaluate_result(gpu._configs[key], condition)
    np.testing.assert_allclose(single.cp, res_ref.cp, rtol=1e-10, atol=1e-10)


@pytest.mark.gpu
def test_gpu_sphere_on_a_circle():
    """Gpu float64: the sphere on a circle has the added-mass force."""
    torch = _need_cuda()
    res = _sphere_solver(16, backend="gpu", dtype=torch.float64).solve(
        FlightCondition(V_inf=1.0, q=0.05))
    assert res.CL == pytest.approx(-4.0 / 3.0 * 0.05, rel=SPHERE_TOLERANCE)
