# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the section that the viscous alpha correction gives to a polar source.

A pre-flapped section shape stores the reference geometry of the section
before the deflection (its chord line and reference chord). The section
of the polar source must keep that reference: the shape is mapped to unit
chord with its own map, which maps the reference too. A plain shape must
give the same points and memo key as before, bit for bit.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ventorum.aero.polar_source_v2d import Ventorum2DPolarSource
from ventorum.airfoil import AirfoilShape, SectionReference, airfoil
from ventorum.viscous.alpha_correction import _source_section, _takes_shape


def _old_points(spec) -> tuple[np.ndarray, str]:
    """The points and the key of the earlier rule (rebuilt from bare coordinates)."""
    coords = np.asarray(airfoil(airfoil(spec).coords).to_unit_chord().coords, dtype=float)
    return coords, hashlib.sha256(np.round(coords, 9).tobytes()).hexdigest()


def _scaled(shape: AirfoilShape, factor: float, shift: tuple[float, float]) -> AirfoilShape:
    """Return the shape scaled by *factor* and moved by *shift*, with its reference."""
    off = np.asarray(shift, dtype=float)
    ref = shape.reference
    new_ref = SectionReference(
        leading_edge=tuple(np.asarray(ref.leading_edge) * factor + off),
        trailing_edge=tuple(np.asarray(ref.trailing_edge) * factor + off),
        chord_ref=float(ref.chord_ref) * factor,
    )
    return AirfoilShape(name=shape.name, coords=shape.coords * factor + off, reference=new_ref)


def test_plain_shape_is_bit_identical():
    """A shape without a stored reference gives the earlier points and key, bit for bit."""
    for spec in (airfoil("4412"), airfoil("0012").coords * 2.0 + 0.3):
        assert getattr(airfoil(spec), "reference", None) is None
        section, key = _source_section(spec)
        old_section, old_key = _old_points(spec)
        assert isinstance(section, np.ndarray)
        assert np.array_equal(section, old_section)
        assert key == old_key


def test_pre_flapped_shape_keeps_its_reference():
    """A pre-flapped shape goes to the source with the reference of the unflapped section."""
    flapped = airfoil("2412").with_flap(x_hinge=0.7, deflection=15.0)
    assert flapped.reference is not None
    section, key = _source_section(flapped)
    assert isinstance(section, AirfoilShape)
    expected = flapped.to_unit_chord()
    assert np.array_equal(section.coords, expected.coords)
    assert section.reference == expected.reference
    # The reference is the unflapped chord line at unit chord, leading edge at the origin.
    np.testing.assert_allclose(section.reference.leading_edge, (0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(section.reference.chord_ref, 1.0, rtol=1e-12)
    # The earlier rule derived another chord line from the flapped points.
    old_section, old_key = _old_points(flapped)
    derived = SectionReference.from_coords(old_section)
    assert abs(derived.trailing_edge[1] - section.reference.trailing_edge[1]) > 1e-3
    assert key != old_key


def test_pre_flapped_shape_key_does_not_depend_on_size_and_position():
    """A scaled and moved copy of a pre-flapped shape gives the same section and key."""
    flapped = airfoil("2412").with_flap(x_hinge=0.7, deflection=15.0)
    base, key = _source_section(flapped)
    moved, key_moved = _source_section(_scaled(flapped, 2.5, (0.4, -0.1)))
    np.testing.assert_allclose(moved.coords, base.coords, atol=1e-12)
    assert key_moved == key


def test_points_only_sources_get_the_earlier_points():
    """With keep_reference=False (a source that takes points only) the earlier points are given."""
    flapped = airfoil("2412").with_flap(x_hinge=0.7, deflection=15.0)
    section, key = _source_section(flapped, keep_reference=False)
    old_section, old_key = _old_points(flapped)
    assert np.array_equal(section, old_section)
    assert key == old_key


def test_only_the_ventorum_2d_source_takes_shapes():
    """The Ventorum-2D source takes an AirfoilShape; other sources get points."""
    assert _takes_shape(Ventorum2DPolarSource(cache=None))
    assert not _takes_shape(object())
