"""Main surface rule (T-0014): the surface with the largest projected planform area.

The area is projected on the x-y plane, after mirroring (a symmetric surface
counts both halves, and a half surface counts with its mirror copy). A tie
(relative difference below 1e-9) goes to the first surface in the list.
"""

import numpy as np

import ventorum as vt


def _surf(name, semi_span, chord, dihedral_deg=0.0, x=0.0, **kw):
    return vt.LiftingSurface(
        name=name, semi_span=semi_span, dihedral=np.radians(dihedral_deg), position=np.array([x, 0.0, 0.0]),
        sections=[vt.WingSection(0.0, chord), vt.WingSection(1.0, chord)], **kw,
    )


def test_v_tail_first_is_not_main():
    """A V-tail listed first does not give the reference values (4th review, item 10)."""
    vtail = _surf("vtail", 1.0, 0.5, dihedral_deg=40.0, x=4.0)
    wing = _surf("wing", 5.0, 1.0)
    ac = vt.Aircraft(surfaces=[vtail, wing])
    assert ac.main_surface_index() == 1
    ac.compute_reference_values()
    assert np.isclose(ac.S_ref, 10.0)


def test_main_surface_at_45_deg_tie():
    """At exactly 45 deg the projected y and z extents are equal; round-off must not decide.

    Two V-tail halves at 45 deg with the same projected area: the first wins,
    in either order.
    """
    a = _surf("a", 1.0, 0.5, dihedral_deg=45.0, x=4.0, is_symmetric=False)
    b = _surf("b", np.sqrt(2.0) / 2.0, 0.5, x=4.0, is_symmetric=False)   # flat, same projected span
    assert vt.Aircraft(surfaces=[a, b]).main_surface_index() == 0
    assert vt.Aircraft(surfaces=[b, a]).main_surface_index() == 0
    # One V-tail alone at 45 deg is the main surface.
    assert vt.Aircraft(surfaces=[_surf("v", 1.0, 0.5, dihedral_deg=45.0)]).main_surface_index() == 0


def test_main_surface_is_largest_projected_area():
    canard = _surf("canard", 1.5, 0.4, x=-2.0)
    wing = _surf("wing", 5.0, 1.0)
    fin = vt.LiftingSurface(name="fin", semi_span=2.0, dihedral=np.pi / 2, is_symmetric=False,
                             sections=[vt.WingSection(0.0, 2.0), vt.WingSection(1.0, 2.0)])
    assert vt.Aircraft(surfaces=[canard, fin, wing]).main_surface_index() == 2
    # A tandem with a larger rear wing: the rear wing is the main surface.
    front = _surf("front", 3.0, 0.8)
    rear = _surf("rear", 3.5, 0.8, x=3.0)
    assert vt.Aircraft(surfaces=[front, rear]).main_surface_index() == 1
    # A fin alone (no projected area) is the main surface: the first one.
    assert vt.Aircraft(surfaces=[fin]).main_surface_index() == 0


def test_half_wing_counts_with_its_mirror_copy():
    """A wing made of two halves is larger than a tail with more area than one half."""
    half = _surf("half", 5.0, 1.0, is_symmetric=False)
    tail = _surf("tail", 3.0, 1.0, x=5.0)          # 6 m^2, larger than one half wing (5 m^2)
    ac = vt.Aircraft(surfaces=[tail, half, half.mirrored()])
    assert ac.main_surface_index() == 1
