# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the far-field multipole expansions of the 3D panel kernels.

Beyond ``far_ratio`` panel diameters from the area centroid of a panel, the
kernels use a multipole expansion about the area centroid that keeps the
area and the second moments of area of the panel. Its relative error per
pair is O((D / r)^3), and O((D / r)^4) for a parallelogram. The point
singularities about the corner mean, used before, had the relative error
O(D / r) for a quadrilateral that is not a parallelogram and O((D / r)^2)
for a parallelogram. With them the lift of a wing moved away from the
exact-integration value as the spanwise panel count grew.

The tests check:

* the area centroid and the second moments of area against closed forms;
* the order of the per-pair error against the exact kernels;
* the lift of a rectangular wing with the default far_ratio against exact
  integration, from 8 to 64 spanwise panels, in free air and in ground
  effect;
* the parity of the numpy, Numba and Warp (GPU) kernels;
* the exact kernels (far_ratio = inf) against stored values of the
  previous implementation (the far-field change must not move them).

Published sources
-----------------
* J. D. Jackson, "Classical Electrodynamics", 3rd ed., John Wiley & Sons,
  1999, section 4.1 (Cartesian multipole expansion).
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, section 10.4 and chapter 12.
"""

from __future__ import annotations

import numpy as np
import pytest

import ventorum as vt
from ventorum.airfoil import airfoil
from ventorum.panel3d import kernels as K
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.solver import PanelSolver

_INV_4PI = 1.0 / (4.0 * np.pi)

#: Test panels [m]: two general quadrilaterals, a parallelogram and a triangle.
_TRAPEZOID = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.7, 0.5, 0.0], [0.3, 0.5, 0.0]])
_SKEW_QUAD = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.02], [0.8, 0.7, 0.0], [-0.1, 0.4, -0.02]])
_PARALLELOGRAM = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.3, 0.5, 0.0], [0.3, 0.5, 0.0]])
_TRIANGLE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.7, 0.0], [0.2, 0.7, 0.0]])

_FUNCS = (K.source_potential, K.doublet_potential, K.source_velocity, K.doublet_velocity)


def _directions(n: int = 300) -> np.ndarray:
    """Return fixed random unit vectors with shape (n, 3)."""
    rng = np.random.default_rng(11)
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1)[:, None]


def _pair_errors(panel: np.ndarray, ratio: float, approx=None) -> np.ndarray:
    """Return the largest relative far-field errors of one panel at r = ratio * D.

    The errors of the source potential, doublet potential, source velocity
    and doublet velocity are scaled by A / (4 pi r^k) with k = 1, 2, 2, 3
    (the size of the leading term). *approx* None uses the kernels with
    every pair in the far set; else it is a function (points, panel) that
    returns the four approximations.
    """
    pans = panel[None]
    geom = K._precompute_geometry(pans)
    diam = float(geom.diameter[0])
    area = float(geom.area[0])
    r = ratio * diam
    pts = geom.far_centre[0] + r * _directions()
    exact = [f(pts, pans, far_ratio=np.inf)[:, 0] for f in _FUNCS]
    if approx is None:
        far = [f(pts, pans, far_ratio=1e-9)[:, 0] for f in _FUNCS]
    else:
        far = approx(pts, panel)
    scale = [area * _INV_4PI / r, area * _INV_4PI / r**2, area * _INV_4PI / r**2, area * _INV_4PI / r**3]
    return np.array([np.max(np.abs(a - e)) / s for a, e, s in zip(far, exact, scale, strict=True)])


def _point_about_corner_mean(pts: np.ndarray, panel: np.ndarray) -> list:
    """Point source and point doublet about the corner mean (the former far field)."""
    geom = K._precompute_geometry(panel[None])
    R = pts - geom.centroid[0]
    r = np.linalg.norm(R, axis=1)
    n = geom.normal[0]
    area = float(geom.area[0])
    z = R @ n
    return [
        -_INV_4PI * area / r,
        -_INV_4PI * area * z / r**3,
        _INV_4PI * area * R / r[:, None] ** 3,
        -_INV_4PI * area * (n[None] / r[:, None] ** 3 - 3.0 * z[:, None] * R / r[:, None] ** 5),
    ]


def _orders(panel: np.ndarray, approx=None) -> np.ndarray:
    """Return the observed orders of the four errors between r = 8, 16 and 32 D."""
    e8, e16, e32 = (_pair_errors(panel, k, approx) for k in (8.0, 16.0, 32.0))
    return np.minimum(np.log2(e8 / e16), np.log2(e16 / e32))


# ---------------------------------------------------------------------------
# Panel moments
# ---------------------------------------------------------------------------


def test_area_moments_of_rectangle_and_triangle():
    """The area centroid and the second moments agree with the closed forms."""
    a, b = 0.8, 0.3
    rect = np.array([[0.0, 0.0, 0.0], [a, 0.0, 0.0], [a, b, 0.0], [0.0, b, 0.0]])
    g = K._precompute_geometry(rect[None])
    np.testing.assert_allclose(g.far_centre[0], [a / 2, b / 2, 0.0], atol=1e-15)
    np.testing.assert_allclose([g.q_xx[0], g.q_xy[0], g.q_yy[0]], [a**3 * b / 12, 0.0, a * b**3 / 12],
                               rtol=1e-13, atol=1e-17)

    # Reversed corner order (normal of opposite sign): same moments.
    g_rev = K._precompute_geometry(rect[::-1][None])
    np.testing.assert_allclose([g_rev.q_xx[0], g_rev.q_yy[0]], [a**3 * b / 12, a * b**3 / 12], rtol=1e-13)

    # Triangle: the area centroid is the corner mean; Q = A / 12 * sum of the
    # corner vectors from the centroid (outer products).
    g_tri = K._precompute_geometry(_TRIANGLE[None])
    np.testing.assert_allclose(g_tri.far_centre[0], g_tri.centroid[0], atol=1e-15)
    corners = _TRIANGLE[:3, :2] - _TRIANGLE[:3, :2].mean(axis=0)
    q_ref = float(g_tri.area[0]) / 12.0 * corners.T @ corners
    np.testing.assert_allclose([g_tri.q_xx[0], g_tri.q_xy[0], g_tri.q_yy[0]],
                               [q_ref[0, 0], q_ref[0, 1], q_ref[1, 1]], rtol=1e-12)


def test_area_centroid_of_trapezoid():
    """The area centroid of a trapezoid is not the corner mean."""
    g = K._precompute_geometry(_TRAPEZOID[None])
    # Parallel sides 1.0 (y = 0) and 0.4 (y = 0.5): y_c = h (a + 2 b) / (3 (a + b)).
    y_c = 0.5 * (1.0 + 2.0 * 0.4) / (3.0 * 1.4)
    np.testing.assert_allclose(g.far_centre[0], [0.5, y_c, 0.0], atol=1e-15)
    assert abs(g.centroid[0, 1] - y_c) > 0.01


# ---------------------------------------------------------------------------
# Per-pair error order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("panel", [_TRAPEZOID, _SKEW_QUAD, _TRIANGLE], ids=["trapezoid", "skew", "triangle"])
def test_far_field_error_is_third_order_for_general_panels(panel):
    """The per-pair error falls as (D / r)^3 (better than the required (D / r)^2)."""
    orders = _orders(panel)
    assert np.all(orders > 2.8), orders
    # At the default switch distance the error is small.
    assert np.all(_pair_errors(panel, 5.0) < 6e-3)


def test_far_field_error_is_fourth_order_for_parallelogram():
    """A parallelogram has zero third moments: the error falls as (D / r)^4."""
    orders = _orders(_PARALLELOGRAM)
    assert np.all(orders > 3.8), orders


def test_point_singularities_about_corner_mean_are_first_order():
    """Document the defect: the former formulas are first order for a trapezoid."""
    orders = _orders(_TRAPEZOID, _point_about_corner_mean)
    assert np.all(orders < 1.3), orders
    assert np.all(_orders(_TRAPEZOID) > 2.8)


# ---------------------------------------------------------------------------
# Lift of a wing against exact integration
# ---------------------------------------------------------------------------


def _wing_cl(n_span: int, h: float | None, far_ratio: float, backend: str = "numba") -> float:
    """Return CL of an AR 6 rectangular NACA 0012 wing (half mesh, n_chord 24, 5 deg)."""
    wing = vt.LiftingSurface(
        name="w",
        semi_span=3.0,
        sections=[vt.WingSection(y_frac=e, chord=1.0, shape="0012") for e in (0.0, 1.0)],
    )
    mesh = mesh_lifting_surface(wing, airfoil("0012").coords, n_chord=24, n_span=n_span, half=True)
    solver = PanelSolver(mesh, S_ref=6.0, b_ref=6.0, c_ref=1.0, ref_point=np.array([0.25, 0.0, 0.0]),
                         backend=backend, far_ratio=far_ratio)
    return solver.solve(vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0), h=h)).CL


#: Bound of |CL(far_ratio=5) / CL(far_ratio=inf) - 1| [-]. The former point
#: singularities gave up to 3.8e-3 (free air) and 4.7e-3 (h = 0.5 m) at 64
#: spanwise panels; the expansion with the second moments stays below 2e-5.
_CL_DRIFT_BOUND = 1e-4


@pytest.mark.parametrize("h", [None, 0.5], ids=["free_air", "ground_h0.5"])
def test_wing_lift_far_field_matches_exact_coarse(h):
    """With the default far_ratio, CL agrees with exact integration (16 spanwise panels)."""
    cl_far = _wing_cl(16, h, 5.0)
    cl_exact = _wing_cl(16, h, np.inf)
    assert abs(cl_far / cl_exact - 1.0) < _CL_DRIFT_BOUND


@pytest.mark.slow
@pytest.mark.parametrize("h", [None, 0.5], ids=["free_air", "ground_h0.5"])
def test_wing_lift_far_field_does_not_drift_with_spanwise_refinement(h):
    """CL with the default far_ratio stays near exact integration from 8 to 64 spanwise panels."""
    for n_span in (8, 16, 32, 64):
        cl_far = _wing_cl(n_span, h, 5.0)
        cl_exact = _wing_cl(n_span, h, np.inf)
        assert abs(cl_far / cl_exact - 1.0) < _CL_DRIFT_BOUND, (n_span, cl_far, cl_exact)


# ---------------------------------------------------------------------------
# Backend parity
# ---------------------------------------------------------------------------


def _random_panels():
    """Return random quadrilaterals and triangles, their mirror, targets and source strengths."""
    rng = np.random.default_rng(3)
    m = 60
    base = rng.uniform(-2.0, 2.0, size=(m, 1, 3))
    quad = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.25, 0.2, 0.01], [0.05, 0.15, -0.01]])
    pans = base + quad[None] * rng.uniform(0.5, 1.5, size=(m, 1, 1)) + rng.normal(scale=0.02, size=(m, 4, 3))
    pans[::7, 3] = pans[::7, 2]
    pts = rng.uniform(-4.0, 4.0, size=(200, 3))
    mirror = pans * np.array([1.0, -1.0, 1.0])
    sigma = rng.normal(size=(m, 6))
    return pans, mirror, pts, sigma


def _rel(a, b) -> float:
    a = np.asarray(a)
    return float(np.max(np.abs(a - np.asarray(b))) / np.max(np.abs(a)))


@pytest.mark.parametrize("far_ratio", [1.0, 5.0])
def test_numba_far_field_parity(far_ratio):
    """The Numba kernels agree with the numpy reference to round-off with many far pairs."""
    kn = pytest.importorskip("ventorum.panel3d.kernels_numba")
    if not kn.HAVE_NUMBA:
        pytest.skip("Numba is not installed.")
    pans, mirror, pts, sigma = _random_panels()
    n_far = int(K._far_mask(pts, K._precompute_geometry(pans), far_ratio).sum())
    assert n_far > 1000
    pairs = [
        (K.source_velocity(pts, pans, far_ratio=far_ratio), kn.source_velocity_numba(pts, pans, far_ratio=far_ratio)),
        (K.doublet_velocity(pts, pans, far_ratio=far_ratio),
         kn.doublet_velocity_numba(pts, pans, far_ratio=far_ratio)),
    ]
    b_ref, c_ref = K.influence_matrices(pts, pans, far_ratio=far_ratio)
    b_nb, c_nb = kn.influence_matrices_numba(pts, pans, far_ratio=far_ratio)
    pairs += [(b_ref, b_nb), (c_ref, c_nb)]
    c_pair, r_pair = K.pair_system(pts, pans, mirror, -1.0, sigma, far_ratio=far_ratio)
    c_pair_nb, r_pair_nb = kn.pair_system_numba(pts, pans, mirror, -1.0, sigma, far_ratio=far_ratio)
    pairs += [(c_pair, c_pair_nb), (r_pair, r_pair_nb)]
    pairs.append((K.doublet_pair_matrix(pts, pans, mirror, 1.0, far_ratio=far_ratio),
                  kn.doublet_pair_matrix_numba(pts, pans, mirror, 1.0, far_ratio=far_ratio)))
    for ref, got in pairs:
        assert _rel(ref, got) < 1e-12


def _need_cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    return torch


@pytest.mark.parametrize("far_ratio", [1.0, 5.0])
def test_warp_far_field_parity(far_ratio):
    """The Warp (GPU) kernels agree with the numpy reference to round-off with many far pairs."""
    _need_cuda()
    from ventorum.panel3d import kernels_warp as kw

    pans, mirror, pts, sigma = _random_panels()
    b_ref, c_ref = K.influence_matrices(pts, pans, far_ratio=far_ratio)
    b_w, c_w = kw.influence_matrices_warp(pts, pans, far_ratio=far_ratio)
    c_pair, r_pair = K.pair_system(pts, pans, mirror, -1.0, sigma, far_ratio=far_ratio)
    c_pair_w = kw.doublet_pair_matrix_warp(pts, pans, mirror, -1.0, far_ratio=far_ratio)
    r_pair_w = kw.source_pair_rhs_warp(pts, pans, mirror, -1.0, sigma, far_ratio=far_ratio)
    for ref, got in [(b_ref, b_w), (c_ref, c_w), (c_pair, c_pair_w), (r_pair, r_pair_w)]:
        assert _rel(ref, got.cpu().numpy()) < 1e-12


def test_gpu_solver_far_field_parity():
    """The gpu solver backend (float64) gives the numpy CL with the default far_ratio."""
    torch = _need_cuda()
    cl_ref = _wing_cl(8, 0.5, 5.0, backend="numpy")
    wing = vt.LiftingSurface(
        name="w",
        semi_span=3.0,
        sections=[vt.WingSection(y_frac=e, chord=1.0, shape="0012") for e in (0.0, 1.0)],
    )
    mesh = mesh_lifting_surface(wing, airfoil("0012").coords, n_chord=24, n_span=8, half=True)
    solver = PanelSolver(mesh, S_ref=6.0, b_ref=6.0, c_ref=1.0, ref_point=np.array([0.25, 0.0, 0.0]),
                         backend="gpu", far_ratio=5.0, dtype=torch.float64)
    cl_gpu = solver.solve(vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0), h=0.5)).CL
    assert abs(cl_gpu / cl_ref - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Exact kernels unchanged
# ---------------------------------------------------------------------------

#: CL, Cm and CDi (Trefftz) of a tapered half wing with far_ratio = inf from
#: the implementation before the far-field change (numpy backend). The
#: far-field change must not move them.
_EXACT_STORED = {
    None: (0.4013241955564673, -0.06540097500865187, 0.007305302148882255),
    0.5: (0.49023179325581084, -0.08465713558295335, 0.004361269712777281),
}


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("h", [None, 0.5], ids=["free_air", "ground_h0.5"])
def test_exact_kernel_results_unchanged(backend, h):
    """With far_ratio = inf the results equal the stored values of the former kernels."""
    wing = vt.LiftingSurface(
        name="w",
        semi_span=3.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.2, shape="0012"),
            vt.WingSection(y_frac=1.0, chord=0.6, x_le=0.4, shape="0012"),
        ],
    )
    mesh = mesh_lifting_surface(wing, airfoil("0012").coords, n_chord=8, n_span=6, half=True)
    solver = PanelSolver(mesh, S_ref=5.4, b_ref=6.0, c_ref=0.9, ref_point=np.array([0.25, 0.0, 0.0]),
                         backend=backend, far_ratio=np.inf)
    res = solver.solve(vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0), h=h))
    got = (res.CL, res.Cm, res.CDi_trefftz)
    # numpy: identical arithmetic (other BLAS builds: round-off); numba: round-off.
    rtol = 1e-11 if backend == "numpy" else 1e-9
    np.testing.assert_allclose(got, _EXACT_STORED[h], rtol=rtol)


def test_infinite_far_ratio_has_no_far_pair():
    """far_ratio = inf gives no far pair, so the exact integrals hold everywhere."""
    pans, _, pts, _ = _random_panels()
    geom = K._precompute_geometry(pans)
    assert not K._far_mask(pts, geom, np.inf).any()
    np.testing.assert_array_equal(K.doublet_potential(pts, pans, far_ratio=np.inf),
                                  K.doublet_potential(pts, pans, far_ratio=1e300))
