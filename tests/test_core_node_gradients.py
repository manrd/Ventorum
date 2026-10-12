# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Node gradients of the linear lifting line with the vortex core on the distance to the finite filament.

The core acts on the distance from the point to the finite filament
(``ventorum.aero.vortex``). The earlier core acted on the distance to the
extended line of the filament. It changed the velocity at points near the
extended line of a bound segment, outside the segment, and the lifting-line
node gradients had errors of order one against the exact law (with sign
changes at the root and at a dihedral break), which small finite-difference
steps did not show.

The checks:

* the adjoint node gradients of the linear lifting line agree within 2 % with
  central differences of an exact-law reference kernel (the source core is
  zero for every pair whose point is outside the filament), on a plain
  tapered wing, a swept wing with dihedral, a wing with a dihedral break, and
  a wing with a tail;
* central differences of the production solve along a node direction do not
  depend on the step from 1e-8 m to 1e-3 m, and agree with the adjoint.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.aero import vortex as V
from ventorum.geometry.deformation import undeformed_nodes
from ventorum.sensitivity.check import _apply_node_step, _gather_node_grad, _production_scalars

OUTS = ("CL", "CDi", "Cm", "Cl", "Cn")
COND = vt.FlightCondition(V_inf=30.0, alpha=math.radians(4.0), beta=math.radians(3.0))
SETTINGS = vt.SolverSettings(solver_type="linear", n_panels=12)


def _section():
    return vt.LinearAirfoil(name="Sec", a0=2.0 * np.pi, alpha_L0=math.radians(-2.0), Cd0=0.008, Cm0=-0.05)


def _plain(name="Wing"):
    return vt.LiftingSurface(name=name, semi_span=5.0, sections=[
        vt.WingSection(y_frac=0.0, chord=2.0, airfoil=_section()),
        vt.WingSection(y_frac=1.0, chord=1.0, airfoil=_section())])


def _swept():
    return vt.LiftingSurface(name="Wing", semi_span=5.0, sweep_le=math.radians(15.0), dihedral=math.radians(5.0),
                             sections=[vt.WingSection(y_frac=0.0, chord=2.0, airfoil=_section()),
                                       vt.WingSection(y_frac=1.0, chord=1.0, airfoil=_section())])


def _break():
    """Tapered wing with dihedral 0 deg inboard and 10 deg outboard of y = 2.5 m."""
    secs = []
    for f in (0.0, 0.5, 1.0):
        z = 0.0 if f <= 0.5 else (f - 0.5) * 5.0 * math.tan(math.radians(10.0))
        secs.append(vt.WingSection(y_frac=f, chord=2.0 - f, z_le=z, airfoil=_section()))
    return vt.LiftingSurface(name="Wing", semi_span=5.0, sections=secs)


def _wing_tail():
    tail = vt.LiftingSurface(name="Tail", semi_span=2.0, position=np.array([4.0, 0.0, 0.5]), sections=[
        vt.WingSection(y_frac=0.0, chord=0.8, airfoil=_section()),
        vt.WingSection(y_frac=1.0, chord=0.5, airfoil=_section())])
    return vt.Aircraft(name="WingTail", surfaces=[_plain(), tail])


CASES = {
    "plain": lambda: vt.Aircraft(surfaces=[_plain()]),
    "swept": lambda: vt.Aircraft(surfaces=[_swept()]),
    "break": lambda: vt.Aircraft(surfaces=[_break()]),
    "wing_tail": _wing_tail,
}


def _exact_law_block(P, hs, tg=None):
    """Horseshoe block of the exact-law reference: no source core for points outside a filament.

    Over a filament the pair keeps the production core; the target core of a
    point of another core group is kept everywhere.
    """
    rc2_src = np.broadcast_to(np.asarray(hs.rc, dtype=float)[None, :] ** 2, (P.shape[0], hs.n))
    if tg is None or hs.group is None:
        rc2_tg = np.zeros((P.shape[0], hs.n))
    else:
        rc2_tg = np.where(tg.group[:, None] != hs.group[None, :], np.asarray(tg.rc, dtype=float)[:, None] ** 2, 0.0)

    def seg(A, B):
        r1 = P[:, None, :] - A[None]
        r0 = (B - A)[None]
        s1 = np.sum(r1 * r0, axis=-1)
        s2 = np.sum((r1 - r0) * r0, axis=-1)
        out = (s1 < 0.0) | (s2 > 0.0)
        return V.segment_velocity(P, A, B, np.sqrt(np.where(out, rc2_tg, rc2_src + rc2_tg)))

    def ray(A, d):
        dr = np.sum((P[:, None, :] - A[None]) * d[None], axis=-1)
        return V.ray_velocity(P, A, d, np.sqrt(np.where(dr < 0.0, rc2_tg, rc2_src + rc2_tg)))

    v = seg(hs.a, hs.b) + seg(hs.a_te, hs.a) + seg(hs.b, hs.b_te)
    return v + ray(hs.b_te, hs.wake_dir) - ray(hs.a_te, hs.wake_dir)


class _Nodes:
    """Production scalars as a function of a flat node displacement."""

    def __init__(self, ac):
        self.ac = ac
        self.base = undeformed_nodes(ac.clone(), SETTINGS, solver="linear")
        self.flat = [(s.name, tag, i, c) for s in ac.surfaces for tag in ("le", "te")
                     for i in range(int(self.base[s.name]["eta"].shape[0])) for c in range(3)]
        sym = {s.name for s in ac.surfaces if s.is_symmetric}
        self.free = np.array([j for j, (name, tag, i, c) in enumerate(self.flat)
                              if not (name in sym and i == 0 and c == 1)], dtype=int)
        self.disp0 = {s.name: None for s in ac.surfaces}

    def scalars(self, u=None, h=0.0):
        ac = self.ac.clone()
        if u is not None and h != 0.0:
            _apply_node_step(ac, self.base, self.disp0, self.flat, u, h)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc = _production_scalars(ac, COND, SETTINGS, "body")
        return np.array([sc[o] for o in OUTS])

    def fd(self, u, h):
        return (self.scalars(u, h) - self.scalars(u, -h)) / (2.0 * h)

    def adjoint(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = vt.gradients(self.ac.clone(), COND, SETTINGS.clone(), outputs=OUTS, wrt=("nodes",), axes="body")
        return np.array([_gather_node_grad(res.jacobian[o], self.flat) for o in OUTS])


@pytest.fixture
def cpu_float64():
    """CPU device, float64 gradients; restore the settings after the test."""
    old_dev, old_prec, old_kernel = gpu.get_device(), gpu._precision, V._kernel_backend
    gpu.set_device("cpu")
    gpu.set_precision("float64")
    yield
    gpu.set_device(old_dev)
    gpu.set_precision(old_prec)
    V._kernel_backend = old_kernel


@pytest.mark.slow
@pytest.mark.parametrize("case", list(CASES))
def test_linear_node_gradients_agree_with_the_exact_law(case, cpu_float64, monkeypatch):
    """Adjoint node gradients within 2 % of central differences (1e-6 m) of the exact-law reference kernel."""
    prob = _Nodes(CASES[case]())
    G = prob.adjoint()
    V.set_kernel_backend("numpy")
    monkeypatch.setattr(V, "_horseshoe_block", _exact_law_block)
    G_ref = np.zeros_like(G)
    for j in prob.free:
        u = np.zeros(len(prob.flat))
        u[j] = 1.0
        G_ref[:, j] = prob.fd(u, 1e-6)
    # The root y-components of a symmetric surface are not free (the mirror plane).
    G, G_ref = G[:, prob.free], G_ref[:, prob.free]
    for k, name in enumerate(OUTS):
        scale = np.max(np.abs(G_ref[k]))
        assert scale > 0.0
        err = np.max(np.abs(G[k] - G_ref[k])) / scale
        assert err < 0.02, (case, name, err)


@pytest.mark.slow
@pytest.mark.parametrize("case", ["plain", "break"])
def test_linear_node_finite_differences_do_not_depend_on_the_step(case, cpu_float64):
    """Central differences along a random node direction agree from 1e-8 m to 1e-3 m, and with the adjoint."""
    prob = _Nodes(CASES[case]())
    rng = np.random.default_rng(3)
    u = np.zeros(len(prob.flat))
    u[prob.free] = rng.standard_normal(prob.free.size)
    u /= np.linalg.norm(u)
    g = prob.adjoint() @ u
    scale = np.max(np.abs(g))
    for h in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        d = prob.fd(u, h)
        assert np.max(np.abs(d - g)) <= 2e-3 * scale, (case, h, d, g)
