# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for 2D panel lift in ground effect and in free air.

Checks the lift coefficient Cl against surface pressure integration and
against the lumped-vortex reference with ground images.
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.airfoil.naca import naca4
from ventorum.panel2d.solver import solve_inviscid
from ventorum.reference.cambered_plate import cambered_plate
from ventorum.reference.karman_trefftz import karman_trefftz_airfoil


def test_free_air_bits_unchanged():
    """In free air Cl keeps the circulation lift and equals Cl_circulation."""
    coords = karman_trefftz_airfoil((-0.07, 0.04), 10.0, n_points=81)
    alpha_scalar = 4.0
    r_scalar = solve_inviscid(coords, alpha_scalar, chord_ref=1.0)
    cl_circ_scalar = 2.0 * float(r_scalar.circulation)
    assert r_scalar.Cl == cl_circ_scalar
    assert r_scalar.Cl_circulation == r_scalar.Cl

    alphas = np.array([-2.0, 0.0, 4.0, 8.0])
    r_vec = solve_inviscid(coords, alphas, chord_ref=1.0)
    cl_circ_vec = 2.0 * np.asarray(r_vec.circulation)
    assert np.array_equal(r_vec.Cl, cl_circ_vec)
    assert np.array_equal(r_vec.Cl_circulation, r_vec.Cl)


def test_ground_effect_cl_equals_cl_pressure():
    """In ground effect Cl equals Cl_pressure for scalar and vector angles."""
    coords = karman_trefftz_airfoil((-0.07, 0.04), 10.0, n_points=81)
    height = 0.3

    r_scalar = solve_inviscid(coords, 4.0, height=height)
    assert r_scalar.Cl == r_scalar.Cl_pressure
    assert r_scalar.Cl != r_scalar.Cl_circulation

    alphas = np.array([0.0, 4.0, 8.0])
    r_vec = solve_inviscid(coords, alphas, height=height)
    assert np.array_equal(r_vec.Cl, r_vec.Cl_pressure)
    assert not np.array_equal(r_vec.Cl, r_vec.Cl_circulation)


def test_ground_effect_torch_backend():
    """Torch backend reports Cl as Cl_pressure in ground effect."""
    torch = pytest.importorskip("torch")
    coords = karman_trefftz_airfoil((-0.07, 0.04), 10.0, n_points=81)
    alphas = np.array([0.0, 4.0])
    r_torch = solve_inviscid(coords, alphas, height=0.4, backend="torch", dtype=torch.float64)
    assert torch.equal(r_torch.Cl, r_torch.Cl_pressure)
    r_np = solve_inviscid(coords, alphas, height=0.4, backend="numpy")
    assert np.allclose(r_torch.Cl.numpy(), r_np.Cl, atol=1e-10, rtol=0.0)
    assert np.allclose(r_torch.Cl_circulation.numpy(), r_np.Cl_circulation, atol=1e-10, rtol=0.0)


def test_cl_circulation_keeps_circulation_lift():
    """Cl_circulation keeps 2 Gamma / c in free air and in ground effect."""
    coords = naca4("0012", n_points=101, closed_te=True)
    chord = 1.0

    for h in [None, 0.5, 0.2]:
        r = solve_inviscid(coords, 3.0, height=h)
        expected_circ_cl = 2.0 * float(r.circulation) / chord
        assert abs(r.Cl_circulation - expected_circ_cl) < 1e-14


def test_thin_flat_plate_matches_discrete_vortex_reference():
    """A thin flat plate at h/c 0.2 matches the 2D discrete-vortex reference within 1 %."""
    alpha_deg = 4.0
    alpha_rad = np.radians(alpha_deg)
    h_c = 0.2

    zero_line = lambda x: 0.0 * np.asarray(x)  # noqa: E731
    ref = cambered_plate(zero_line, zero_line, alpha=alpha_rad, h=h_c, n_panels=512)
    ref_cl = ref["CL"]

    plate_coords = naca4("0001", n_points=241, closed_te=True)
    res = solve_inviscid(plate_coords, alpha_deg, height=h_c)

    rel_diff = abs(res.Cl - ref_cl) / ref_cl
    assert rel_diff < 0.01

    circ_diff = abs(res.Cl_circulation - ref_cl) / ref_cl
    assert circ_diff > 0.08
