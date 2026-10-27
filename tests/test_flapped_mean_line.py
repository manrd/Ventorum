# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Part-span chordwise flaps on a cambered vortex lattice (mean-line leg chains).

On a strip with a mean line the legs run through the bound points of the
panels aft of them, with a node on the mean surface between two bound
points. On a flap strip the hinge point takes the place of the node
around the hinge. At the outer edge of the join at a side edge of a flap,
a flap strip shares its edge with a strip without a hinge, whose legs use
the mean-surface node. On a cambered section the two nodes are not on
one line, so the two legs on that edge did not coincide (a gap of about
4e-4 chord on a NACA 2412 with a hinge at 0.75 chord). The velocity of
one leg at the other leg grows as the vortex core of the narrower strips
gets smaller, and the leg force with it: CL of a NACA 2412 wing with a
part-span flap went from 0.51 at 16 spanwise panels to 1.15 at 80 (2.71
for a NACA 6409), while the circulation stayed smooth. The hinge point
is now not a node at such an edge.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.aero import vortex
from ventorum.geometry import lattice as mesh_mod
from ventorum.geometry import lattice_cache

_HINGE = 0.75


def _flap() -> vt.ControlSurface:
    return vt.ControlSurface("flap", 0.1, 0.4, _HINGE, np.radians(10.0))


def _aileron() -> vt.ControlSurface:
    return vt.ControlSurface("aileron", 0.6, 0.95, _HINGE, np.radians(10.0), symmetric=False)


def _wing(shape, controls=()) -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="w", semi_span=4.0,
        sections=[vt.WingSection(y_frac=e, chord=1.0, shape=shape) for e in (0.0, 1.0)],
        controls=list(controls),
    )


def _totals(shape, n_panels, controls=()):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return vt.VortexLatticeSolver().solve(
            vt.Aircraft(surfaces=[_wing(shape, controls)]),
            vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0)),
            vt.SolverSettings(n_panels=n_panels),
        ).totals


def _cl(shape, n_panels, controls=()) -> float:
    return float(_totals(shape, n_panels, controls).CL)


def _lattice(shape, n_panels=24, n_chord=4):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return mesh_mod.build_lattice(vt.Aircraft(surfaces=[_wing(shape, [_flap()])]),
                                      vt.SolverSettings(n_panels=n_panels), n_chord=n_chord)


@pytest.fixture(autouse=True)
def _restore_backends():
    old_backend = vortex._kernel_backend
    old_device, old_precision = gpu.get_device(), gpu._precision
    gpu.set_device("cpu")
    lattice_cache.clear()
    yield
    lattice_cache.clear()
    vortex.set_kernel_backend(old_backend)
    gpu.set_device(old_device)
    gpu.set_precision(old_precision)


# --- lattice -------------------------------------------------------------------

@pytest.mark.parametrize("shape", ["2412", "6409"])
def test_legs_coincide_on_every_interior_edge(shape):
    """The two strips at an interior edge have the same leg nodes there, also at the side edges of a flap."""
    lat = _lattice(shape)
    n_c = lat.n_chord
    a_e, b_e = vortex.chain_ends(lat.a, lat.b, lat.a_te, lat.b_te, lat.leg_next)
    for s in range(lat.n_strips - 1):
        if lat.strip_surface[s] != lat.strip_surface[s + 1]:
            continue
        right = s * n_c + np.arange(n_c)
        left = right + n_c
        for p_r, p_l in ((lat.b, lat.a), (lat.b_kink, lat.a_kink), (b_e, a_e)):
            np.testing.assert_allclose(p_r[right], p_l[left], rtol=0, atol=1e-12)
    # The flap has a join at both side edges on both wing halves: four edges
    # where a flap strip meets a strip without a hinge.
    assert int(lat.geom_cache["flap_field"]["hinge_off"].sum()) == 4


def test_rule_normals_turn_only_aft_of_the_hinge():
    """The leg-force rule normal turns with the flap only on the leg segments aft of the hinge."""
    from ventorum.aero.loads import leg_segments

    lat = _lattice("2412")
    n_c, n_p = lat.n_chord, lat.n_panels
    normal = lat.geom_cache["leg_force_rules"][0]
    start, end = leg_segments(lat)
    ff = lat.geom_cache["flap_field"]
    hinge_pts = lat.geom_cache["flap_hinges"]
    seen_turned = seen_kept = 0
    for s in np.flatnonzero(np.isfinite(ff["hinge"])):
        h_l, h_r = hinge_pts[s]
        chord_dir = lat.chord_dir[s]
        for k in range(n_c):
            p = s * n_c + k
            for row, h in ((0, h_l), (1, h_r), (2, h_l), (3, h_r)):
                i = row * n_p + p
                if np.linalg.norm(end[i] - start[i]) == 0.0:
                    continue
                mid = 0.5 * (start[i] + end[i])
                aft = float((mid - h) @ chord_dir) > 0.0
                d = ff["angle"][s, row % 2]
                if aft and d != 0.0:
                    assert abs(float(normal[i] @ lat.normal[s]) - np.cos(d)) < 1e-12
                    seen_turned += 1
                else:
                    np.testing.assert_allclose(normal[i], lat.normal[s], rtol=0, atol=1e-15)
                    seen_kept += 1
    assert seen_turned > 0 and seen_kept > 0


# --- solutions -----------------------------------------------------------------

#: Largest difference [-] between the flap increment of CL on a cambered
#: section and on a symmetric section on the same mesh. Thin-airfoil theory
#: adds the camber and the flap linearly; the lattice on the mean surface
#: differs by the turn of a curved surface about a hinge on the mean line.
#: The measured difference grows linearly with the camber: 3e-4 at 1 %
#: camber, 1.6e-3 at 6 % (NACA 6409, 80 spanwise panels). The limit is
#: about 2 % of the increment (0.15). The defect gave 0.07 to 1.9.
_INCREMENT_TOL = 3e-3
#: Largest CL difference [-] between 48 and 80 spanwise panels. A symmetric
#: section changes by 1.2e-3 (the flap side edges move with the strip
#: edges); a cambered section must not change more than that by much.
_CONVERGED_TOL = 3e-3
#: Largest CL range [-] over 24, 48 and 80 spanwise panels. A symmetric
#: section has a range of 0.010: at 24 panels the strip edges give a
#: shorter effective flap span.
_RANGE_TOL = 0.015


@pytest.mark.parametrize("shape", ["2412", "6409"])
def test_flap_increment_matches_symmetric_section(shape):
    """The flap increment of CL on a cambered wing is the increment on a symmetric wing (thin-section limit)."""
    inc = _cl(shape, 24, [_flap()]) - _cl(shape, 24)
    inc_sym = _cl("0012", 24, [_flap()]) - _cl("0012", 24)
    assert abs(inc - inc_sym) < _INCREMENT_TOL, (inc, inc_sym)


@pytest.mark.parametrize("shape", ["2412", "6409"])
def test_flapped_camber_converges_spanwise(shape):
    """CL of a cambered wing with a part-span flap converges with the spanwise panels (formerly 0.55 to 1.15)."""
    cl = {}
    for ns in (24, 48, 80):
        cl[ns] = _cl(shape, ns, [_flap()])
        inc_sym = _cl("0012", ns, [_flap()]) - _cl("0012", ns)
        assert abs(cl[ns] - _cl(shape, ns) - inc_sym) < _INCREMENT_TOL, (ns, cl[ns])
    assert abs(cl[48] - cl[80]) < _CONVERGED_TOL, cl
    assert max(cl.values()) - min(cl.values()) < _RANGE_TOL, cl


def test_flap_and_aileron_on_6409():
    """A flap and an antisymmetric aileron on a NACA 6409 wing give the CL of the flap alone (formerly 3.48)."""
    flap_only = _totals("6409", 80, [_flap()])
    both = _totals("6409", 80, [_flap(), _aileron()])
    clean = _cl("6409", 80)
    inc_sym = _cl("0012", 80, [_flap()]) - _cl("0012", 80)
    assert abs(float(flap_only.CL) - clean - inc_sym) < _INCREMENT_TOL
    # The antisymmetric aileron changes CL only by the second-order effect
    # of the deflected geometry (5e-4 on a symmetric section at 24 panels).
    assert abs(float(both.CL) - float(flap_only.CL)) < 3e-3
    # Positive aileron on the right wing (trailing edge down) rolls the
    # aircraft left: Cl < 0 (AVL axes).
    assert float(both.Cl) < -0.01


# Results of the code before the change (CL, CDi, Cm, CY, Cl, Cn) at 24
# spanwise panels and alpha 2 deg. A clean cambered wing and a symmetric
# wing with a flap do not have an edge where the fix applies, so the
# results stay the same. rtol 1e-10 and an absolute floor of 1e-12 accept
# other BLAS builds (CI).
_STORED = {
    "clean_2412": ("2412", (), (0.33641999107023657, 0.00467789703698421, -0.128160158481824,
                                -1.5105075164968115e-19, -3.498113159053798e-18, -3.006263983027834e-19)),
    "clean_6409": ("6409", (), (0.661392731373163, 0.018218398097771118, -0.30090120005641846,
                                4.028020043991498e-19, -2.4884037888955287e-18, -1.346082479544815e-18)),
    "flap_0012": ("0012", ("flap",), (0.3021835647263291, 0.0047381579972962055, -0.10544650849510419,
                                      -8.732621579747192e-20, -1.3333474064076837e-18, 2.858246416027368e-20)),
    "flap_aileron_0012": ("0012", ("flap", "aileron"),
                          (0.3026449283185084, 0.008478667932783063, -0.10563274302283415,
                           0.000800473196709901, -0.04464131186976753, -0.0007321139529711561)),
}


@pytest.mark.parametrize("case", sorted(_STORED))
def test_unaffected_cases_keep_their_results(case):
    """Clean cambered wings and flapped symmetric wings give the results of the code before the change."""
    shape, names, expected = _STORED[case]
    controls = [{"flap": _flap, "aileron": _aileron}[n]() for n in names]
    t = _totals(shape, 24, controls)
    got = [float(getattr(t, k)) for k in ("CL", "CDi", "Cm", "CY", "Cl", "Cn")]
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)
