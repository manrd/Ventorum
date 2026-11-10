# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the float64 residual of the float32 GPU solve of the 3D panel method.

The float32 GPU path rounds the float64 system matrix to float32 before
the LU factorization. After the solve, the solver runs the float64
doublet pair kernels again, one block of rows at a time, and gives
``||b - A64 x|| / ||b||`` per unit onset in the dispatch record. These
tests check:

* the blocked float64 product against the full float64 matrix;
* the residual against a host float64 reference ``b - A x`` of the numpy
  backend (full mesh, free air);
* a small residual for a normal case (free air, half mesh with both
  parities, ground effect) and a large one for a perturbed solution;
* an empty record on the float64 GPU path and on the CPU;
* the new field of the factory dispatch record.

Published sources
-----------------
* Higham, N. J., 2002, Accuracy and Stability of Numerical Algorithms,
  2nd ed., SIAM, chapter 7 (normwise backward error of a computed
  solution from its residual) and chapter 9 (LU factorization).
* Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed.,
  Cambridge University Press, chapter 12.
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.core.datatypes import FlightCondition, LiftingSurface, WingSection
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.solver import PanelSolver

#: Upper limit of the float64 residual of a normal float32 solve [-]. The
#: float32 unit round-off is 6e-8. The rounding of A to float32 and the
#: float32 LU give residuals of 2e-7 to 8e-6 on these small wing meshes
#: (measured on a GTX 1660 SUPER; largest for the w and q onsets, whose
#: solutions carry the circulation). 5e-5 leaves a margin and is still far
#: below the residuals of a perturbed solution.
_NORMAL_LIMIT = 5.0e-5


def _need_cuda():
    """Skip the test when a CUDA GPU, PyTorch or Warp is missing."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is not available.")
    return torch


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


def _wing_mesh(half: bool, n_chord: int = 8, n_span: int = 8):
    """Rectangular NACA 0012 wing of semi-span 3 m and chord 1 m."""
    surf = LiftingSurface(
        name="Wing",
        semi_span=3.0,
        sections=[WingSection(y_frac=0.0, chord=1.0), WingSection(y_frac=1.0, chord=1.0)],
        is_symmetric=True,
    )
    return mesh_lifting_surface(surf, _naca0012_coordinates(), n_chord=n_chord, n_span=n_span, half=half)


def _only_config(solver: PanelSolver):
    """Return the single configuration of a solver."""
    assert len(solver._configs) == 1
    return next(iter(solver._configs.values()))


@pytest.mark.gpu
def test_blocked_float64_product_matches_the_full_matrix() -> None:
    """The blocked float64 product equals the full float64 pair matrix times x to round-off."""
    torch = _need_cuda()
    from ventorum.panel3d.kernels_warp import doublet_pair_matrix_warp, doublet_pair_matvec_warp
    from ventorum.panel3d.solver import mirror_panels_y

    mesh = _wing_mesh(half=True, n_chord=6, n_span=5)
    solver = PanelSolver(mesh, backend="numpy")
    pts, panels = solver.centroids, solver.body_verts
    mirror = mirror_panels_y(panels)
    rng = np.random.default_rng(3)
    x = torch.as_tensor(rng.standard_normal((len(pts), 4)), dtype=torch.float64, device="cuda")
    for sign in (1.0, -1.0):
        full = doublet_pair_matrix_warp(pts, panels, mirror, sign, dtype="float64", device="cuda")
        ref = (full @ x).cpu().numpy()
        # 7 rows per block: many blocks and a short last block.
        out = doublet_pair_matvec_warp(pts, panels, mirror, sign, x, device="cuda", block_rows=7)
        np.testing.assert_allclose(out.cpu().numpy(), ref, rtol=1e-13, atol=1e-13 * np.max(np.abs(ref)))
    # One-dimensional x gives a one-dimensional product.
    one = doublet_pair_matvec_warp(pts, panels, None, 1.0, x[:, 0], device="cuda")
    assert one.shape == (len(pts),)


@pytest.mark.gpu
def test_float64_residual_matches_a_host_reference() -> None:
    """The device residual equals ``||b - A x|| / ||b||`` with the numpy float64 matrix.

    Full mesh in free air: ``A`` (with the wake fold) and ``B`` come from
    the numpy backend, ``x`` is the float32 GPU unit solution.
    """
    torch = _need_cuda()
    mesh = _wing_mesh(half=False)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    gpu_solver = PanelSolver(mesh, backend="gpu", dtype="float32")
    gpu_solver.solve(cond)
    cfg = _only_config(gpu_solver)
    res = np.array([cfg.float64_residuals[k] for k in range(6)])

    host = PanelSolver(mesh, backend="numpy")
    mat_a, mat_b = host.A, host.B
    b = -(mat_b @ host._unit_sigma)
    x = cfg.unit_mu_t.to(torch.float64).cpu().numpy()
    ref = np.linalg.norm(b - mat_a @ x, axis=0) / np.linalg.norm(b, axis=0)
    print("float64 residuals (full mesh):", res)
    np.testing.assert_allclose(res, ref, rtol=1e-6)
    assert np.all(res < _NORMAL_LIMIT)
    # A float32 solution is not exact: the residual is above the float64 floor.
    assert np.all(res > 1.0e-12)


@pytest.mark.gpu
@pytest.mark.parametrize("ground", [False, True], ids=["free", "ground"])
def test_float64_residual_is_small_and_grows_for_a_perturbed_solution(ground: bool) -> None:
    """Half mesh, both parities: small residual; a perturbed solution gives a large one."""
    torch = _need_cuda()
    mesh = _wing_mesh(half=True)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), p=0.2, h=0.6 if ground else None)
    solver = PanelSolver(mesh, backend="gpu", dtype="float32")
    solver.solve(cond)
    cfg = _only_config(solver)
    assert set(cfg.float64_residuals) == set(range(6))
    res = np.array([cfg.float64_residuals[k] for k in range(6)])
    print(f"float64 residuals (half mesh, ground={ground}):", res)
    assert np.all(np.isfinite(res)) and np.all(res < _NORMAL_LIMIT)

    image = image_m = None
    if ground:
        from ventorum.panel3d.solver import reflect_panels_plane

        normal, offset = cfg.ground
        image = reflect_panels_plane(solver.body_verts, normal, offset)
        image_m = reflect_panels_plane(solver._body_verts_m, normal, offset)
    solver._config_strips(cfg)
    rng = np.random.default_rng(11)
    for parity in ("sym", "asym"):
        sign, cols = solver._parity_sign_cols(parity)
        _, rhs = solver._body_system(parity)
        if ground:
            _, rhs_img = solver._pair_system(image, image_m, sign, cols)
            rhs = rhs + rhs_img
        x = cfg.unit_mu_t[:, list(cols)].to(torch.float64)
        again = solver._float64_unit_residuals(cfg, parity, rhs, x, image, image_m)
        np.testing.assert_allclose(again, res[list(cols)], rtol=1e-9)
        # Relative perturbation of 1e-2 on each doublet value.
        noise = torch.as_tensor(rng.standard_normal(tuple(x.shape)), dtype=torch.float64, device=x.device)
        bad = solver._float64_unit_residuals(cfg, parity, rhs, x * (1.0 + 1.0e-2 * noise), image, image_m)
        print(f"perturbed residuals ({parity}):", bad)
        assert np.all(bad > 1.0e-3)
        assert np.all(bad > 100.0 * res[list(cols)])


@pytest.mark.gpu
def test_float64_residual_is_not_computed_for_float64_or_cpu() -> None:
    """The float64 GPU path and the CPU backends leave the record empty."""
    _need_cuda()
    mesh = _wing_mesh(half=True, n_chord=6, n_span=5)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    for kwargs in ({"backend": "gpu", "dtype": "float64"}, {"backend": "numpy"}):
        solver = PanelSolver(mesh, **kwargs)
        solver.solve(cond)
        assert _only_config(solver).float64_residuals == {}


@pytest.mark.gpu
def test_factory_dispatch_record_holds_the_float64_residuals() -> None:
    """The factory gives the residuals per unit onset in the dispatch record."""
    _need_cuda()
    from ventorum import gpu
    from ventorum.core.datatypes import Aircraft, SolverSettings
    from ventorum.solvers.factory import make_solver

    surf = LiftingSurface(
        name="Wing",
        semi_span=3.0,
        sections=[WingSection(y_frac=0.0, chord=1.0, shape="0012"), WingSection(y_frac=1.0, chord=1.0, shape="0012")],
        is_symmetric=True,
    )
    ac = Aircraft(name="W", surfaces=[surf])
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    settings = SolverSettings(n_panels=6, n_chord=8, panel_cache=False)
    old_device, old_precision = gpu.get_device(), gpu._precision
    try:
        gpu.set_device("gpu")
        gpu.set_precision("float32")
        disp32 = make_solver("panel3d").solve(ac, cond, settings).details["dispatch"]
        gpu.set_precision("float64")
        disp64 = make_solver("panel3d").solve(ac, cond, settings).details["dispatch"]
    finally:
        gpu.set_device(old_device)
        gpu.set_precision(old_precision)
    assert disp32["dtype"] == "float32"
    res = disp32["float64_relative_residuals"]
    # Symmetric flight on a half mesh: the symmetric unit onsets u, w, q.
    assert set(res) == {0, 2, 4}
    assert all(0.0 < v < _NORMAL_LIMIT for v in res.values())
    assert disp64["float64_relative_residuals"] == {}
    cpu = make_solver("panel3d").solve(ac, cond, SolverSettings(n_panels=6, n_chord=8, panel_backend="numpy"))
    assert cpu.details["dispatch"]["float64_relative_residuals"] == {}
