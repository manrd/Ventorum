# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests for TabulatedAirfoil assignment order and blend_tabulated alpha range.

Covers the issues found in the 4th review:
- Assignment order should keep the correct pairs.
- Length mismatch is reported when the interpolation is built.
- Repeated alpha values are merged once after an assignment.
- blend_tabulated warns about the kept alpha range when ranges differ.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero.polars import blend_tabulated


def test_assignment_in_any_order_keeps_the_pairs():
    """The reproduction from the task card gives the correct value."""
    af = vt.TabulatedAirfoil(
        alpha=np.radians([0, 5, 10]),
        Cl_data=np.array([0.0, 0.5, 1.0]),
        Cd_data=np.array([0.01, 0.01, 0.02]),
    )
    af.alpha = np.radians([10, 0, 5])
    af.Cl_data = np.array([1.2, 0.1, 0.6])
    af.Cd_data = np.array([0.02, 0.01, 0.01])
    # alpha=5 deg should give Cl=0.6 (the third value assigned with alpha=5)
    assert np.isclose(af.Cl(np.radians(5.0)), 0.6)

    # Also test assigning Cl before alpha
    af2 = vt.TabulatedAirfoil(
        alpha=np.radians([0, 5, 10]),
        Cl_data=np.array([0.0, 0.5, 1.0]),
        Cd_data=np.array([0.01, 0.01, 0.02]),
    )
    af2.Cl_data = np.array([1.2, 0.1, 0.6])
    af2.alpha = np.radians([10, 0, 5])
    af2.Cd_data = np.array([0.02, 0.01, 0.01])
    assert np.isclose(af2.Cl(np.radians(5.0)), 0.6)


def test_length_mismatch_is_reported_when_used():
    """Alpha with 3 values and Cl with 4 values raises ValueError at first Cl() call."""
    af = vt.TabulatedAirfoil(
        alpha=np.radians([0, 5, 10]),
        Cl_data=np.array([0.0, 0.5, 1.0]),
        Cd_data=np.array([0.01, 0.01, 0.02]),
    )
    af.alpha = np.radians([0, 5, 10])  # 3 values
    af.Cl_data = np.array([0.1, 0.2, 0.3, 0.4])  # 4 values
    # Assignment should not raise yet
    with pytest.raises(ValueError, match="has 4 values, alpha has 3"):
        af.Cl(np.radians(5.0))


def test_repeated_alpha_values_are_merged_once():
    """The present merge rule still holds after an assignment."""
    af = vt.TabulatedAirfoil(
        alpha=np.radians([0, 5, 10]),
        Cl_data=np.array([0.0, 0.5, 1.0]),
        Cd_data=np.array([0.01, 0.01, 0.02]),
    )
    # Assign new data with a repeated alpha
    af.alpha = np.radians([0, 5, 5, 10])
    af.Cl_data = np.array([0.0, 0.4, 0.6, 1.0])
    af.Cd_data = np.array([0.01, 0.01, 0.02, 0.02])
    # Should merge the two alpha=5 entries (mean of 0.4 and 0.6 = 0.5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cl_at_5 = af.Cl(np.radians(5.0))
        assert len(w) == 1
        assert "repeated alpha" in str(w[0].message).lower()
    assert np.isclose(cl_at_5, 0.5)


def test_blend_warns_about_the_kept_range():
    """Two polars with ranges -10..20 deg and -5..15 deg warn with kept range."""
    af1 = vt.TabulatedAirfoil(
        name="af1",
        alpha=np.radians(np.linspace(-10, 20, 31)),
        Cl_data=2 * np.pi * np.radians(np.linspace(-10, 20, 31)),
        Cd_data=0.01 * np.ones(31),
    )
    af2 = vt.TabulatedAirfoil(
        name="af2",
        alpha=np.radians(np.linspace(-5, 15, 21)),
        Cl_data=2 * np.pi * np.radians(np.linspace(-5, 15, 21)),
        Cd_data=0.01 * np.ones(21),
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        blended = blend_tabulated(af1, af2, 0.5)
        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
        msg = str(w[0].message)
        # Message should contain the kept range in degrees: -5 and 15
        assert "-5" in msg
        assert "15" in msg
    # Blended range should be the intersection
    assert np.isclose(np.degrees(blended.alpha.min()), -5.0)
    assert np.isclose(np.degrees(blended.alpha.max()), 15.0)


def test_blend_of_equal_ranges_does_not_warn():
    """Two polars with the same alpha range produce no warning."""
    af1 = vt.TabulatedAirfoil(
        name="af1",
        alpha=np.radians(np.linspace(-10, 20, 31)),
        Cl_data=2 * np.pi * np.radians(np.linspace(-10, 20, 31)),
        Cd_data=0.01 * np.ones(31),
    )
    af2 = vt.TabulatedAirfoil(
        name="af2",
        alpha=np.radians(np.linspace(-10, 20, 31)),
        Cl_data=2 * np.pi * np.radians(np.linspace(-10, 20, 31)),
        Cd_data=0.01 * np.ones(31),
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = blend_tabulated(af1, af2, 0.5)
        # Filter for RuntimeWarning about alpha range
        range_warnings = [x for x in w if issubclass(x.category, RuntimeWarning) and "range" in str(x.message).lower()]
        assert len(range_warnings) == 0


def test_strip_section_data_do_not_depend_on_the_table_order():
    """The lattice reads Cd0 and Cm0 from the sorted tables.

    The arrays are stored as the user gives them, so code that reads them
    directly with np.interp (which needs ascending angles) gave wrong values
    for a polar given in another order.
    """
    from ventorum.geometry.lattice import build_lattice

    a = np.radians(np.arange(-10.0, 16.0, 1.0))
    cl, cd, cm = 0.1 + 2.0 * np.pi * a, 0.01 + 0.2 * a**2, -0.05 + 0.01 * a
    order = np.random.default_rng(0).permutation(a.size)

    def strips(o):
        af = vt.TabulatedAirfoil(alpha=a[o], Cl_data=cl[o], Cd_data=cd[o], Cm_data=cm[o])
        wing = vt.LiftingSurface(semi_span=4.0, sections=[vt.WingSection(0.0, 1.0, airfoil=af),
                                                            vt.WingSection(1.0, 1.0, airfoil=af)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return build_lattice(vt.Aircraft(surfaces=[wing]), vt.SolverSettings(n_panels=8), collocation="vlm")

    ref, new = strips(np.arange(a.size)), strips(order)
    np.testing.assert_allclose(new.Cd0, ref.Cd0, rtol=1e-14)
    np.testing.assert_allclose(new.Cm0, ref.Cm0, rtol=1e-14)
