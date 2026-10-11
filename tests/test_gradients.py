# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the adjoint gradients of the vortex lattice and the linear lifting line."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu


@pytest.fixture
def float64_precision():
    """Run the test with float64 gradient precision; restore it after it."""
    old = gpu.get_precision()
    gpu.set_precision("float64")
    yield
    gpu.set_precision(old)


@pytest.fixture
def float32_precision():
    """Run the test with float32 gradient precision; restore it after it."""
    old = gpu.get_precision()
    gpu.set_precision("float32")
    yield
    gpu.set_precision(old)


@pytest.fixture
def cpu_device():
    """Run the test on the CPU; restore the device setting after it."""
    old = gpu.get_device()
    gpu.set_device("cpu")
    yield
    gpu.set_device(old)


@pytest.fixture
def gpu_device():
    """Run the test with the GPU device selected; restore the settings after it."""
    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    old_dev, old_prec = gpu.get_device(), gpu.get_precision()
    yield
    gpu.set_device(old_dev)
    gpu.set_precision(old_prec)


def _linear_section() -> vt.LinearAirfoil:
    return vt.LinearAirfoil(
        name="GradSec", a0=2.0 * np.pi, alpha_L0=np.radians(-2.0), Cd0=0.008, Cm0=-0.05)


def _swept_wing() -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sweep_le=np.radians(15.0),
        dihedral=np.radians(5.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0, airfoil=_linear_section()),
            vt.WingSection(y_frac=1.0, chord=1.0, airfoil=_linear_section()),
        ],
    )


def _plain_wing() -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0, airfoil=_linear_section()),
            vt.WingSection(y_frac=1.0, chord=1.0, airfoil=_linear_section()),
        ],
    )


def _wing_tail() -> vt.Aircraft:
    wing = _plain_wing()
    tail = vt.LiftingSurface(
        name="Tail",
        semi_span=2.0,
        position=np.array([4.0, 0.0, 0.5]),
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.8, airfoil=_linear_section()),
            vt.WingSection(y_frac=1.0, chord=0.5, airfoil=_linear_section()),
        ],
    )
    return vt.Aircraft(name="WingTail", surfaces=[wing, tail])


def _with_displacements(ac: vt.Aircraft, st: vt.SolverSettings, solver: str) -> vt.Aircraft:
    """Return a clone of *ac* with smooth heave and twist displacements on every surface."""
    from ventorum.geometry.deformation import displacements_from_section_motion, undeformed_nodes

    out = ac.clone()
    nodes = undeformed_nodes(out, st, solver=solver)
    for surf in out.surfaces:
        bn = nodes[surf.name]
        eta = np.asarray(bn["eta"], dtype=float)
        surf.node_displacements = displacements_from_section_motion(
            bn, heave=0.02 * surf.semi_span * eta ** 2,
            twist=eta * np.radians(1.0), pivot_x_c=0.25)
    return out


def _parity_cases() -> list[tuple]:
    """Configurations of the forward-parity and Richardson tests.

    Every case has linear airfoils with nonzero Cd0, Cm0 and alpha_L0. All
    cases have beta = 3 deg, except the symmetric case ``vlm-swept-sym``
    (beta = 0, fold on), where the production solve and the gradient both
    use the folded system (see :func:`_case_io`). Returns (id, aircraft,
    condition, settings) rows.
    """
    cases = []
    beta = np.radians(3.0)
    # Symmetric state with the fold in the production solve and in the adjoint.
    ac = vt.Aircraft(surfaces=[_swept_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=4, use_symmetry=True)
    cases.append(("vlm-swept-sym", ac, cond, st))
    # Symmetric swept and dihedral wing, VLM with 4 chordwise panels, fold on and off.
    for fold in (True, False):
        ac = vt.Aircraft(surfaces=[_swept_wing()])
        cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta)
        st = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=4, use_symmetry=fold)
        cases.append((f"vlm-swept-{'fold' if fold else 'full'}", ac, cond, st))
    # Linear lifting line.
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta)
    cases.append(("linear", ac, cond, vt.SolverSettings(solver_type="linear", n_panels=12)))
    # Wing and tail (cross-surface core).
    ac = _wing_tail()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta)
    cases.append(("vlm-wing-tail", ac, cond, vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=2)))
    # Ground effect at h/c = 0.3 (VLM only).
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    ac.compute_reference_values()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta, h=0.3 * float(ac.c_ref))
    cases.append(("vlm-ground", ac, cond, vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=4)))
    # Node displacements present (plain wing: the heave helper needs a flat wing).
    st0 = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=2)
    ac = _with_displacements(vt.Aircraft(surfaces=[_plain_wing()]), st0, "vlm")
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta)
    cases.append(("vlm-displaced", ac, cond, st0))
    return cases


def _full_wrt(cond: vt.FlightCondition) -> tuple:
    wrt = ["nodes", "alpha", "beta", "V_inf", "rho", "phi", "sections", "reference"]
    if cond.h is not None:
        wrt.append("h")
    return tuple(wrt)


_SCALAR_OUTPUTS = ("CL", "CY", "CDi", "CDp", "CD", "Cl", "Cm", "Cn")
# Symmetric inputs and outputs: with these the adjoint uses the fold.
_SYM_WRT = ("alpha", "V_inf", "rho")
_SYM_OUTPUTS = ("CL", "CDi", "CDp", "CD", "Cm")


def _case_io(case_id: str, cond: vt.FlightCondition, check: bool = False) -> tuple[tuple, tuple]:
    """Return (wrt, outputs) of a parity case.

    The symmetric case requests only symmetric inputs and outputs, so that
    the adjoint uses the fold. *check* gives the outputs of the Richardson
    check (CL, CD, Cm).
    """
    if case_id == "vlm-swept-sym":
        return _SYM_WRT, (("CL", "CD", "Cm") if check else _SYM_OUTPUTS)
    return _full_wrt(cond), (("CL", "CD", "Cm") if check else _SCALAR_OUTPUTS)


def _failed(out: dict) -> dict:
    """Return the failed components and the reasons of a check_gradients result."""
    bad = {k: v for k, v in out["details"].items() if v["compared"] and not v["passed"]}
    return {"reasons": out["reasons"], "failed": bad}


@pytest.mark.parametrize("case_id,ac,cond,st", _parity_cases(), ids=[c[0] for c in _parity_cases()])
def test_forward_parity(case_id: str, ac: vt.Aircraft, cond: vt.FlightCondition,
                        st: vt.SolverSettings, float64_precision) -> None:
    """The differentiated reference copy equals the production totals to 1e-12 relative (float64)."""
    wrt, outputs = _case_io(case_id, cond)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=outputs, wrt=wrt, axes="body")
        prod = vt.analyze(ac.clone(), cond, st.clone())
    assert res.forward_mismatch <= 1e-12, (case_id, res.forward_mismatch)
    assert res.precision_primal == str(prod.details.get("precision", "float64"))
    if case_id == "vlm-swept-sym":
        # Both the production solve and the adjoint use the fold.
        assert prod.symmetry_used is True
        assert res.symmetric is True


@pytest.mark.parametrize("case_id,ac,cond,st", _parity_cases(), ids=[c[0] for c in _parity_cases()])
def test_adjoint_vs_richardson(case_id: str, ac: vt.Aircraft, cond: vt.FlightCondition,
                               st: vt.SolverSettings, float64_precision) -> None:
    """check_gradients passes (1e-6) for all scalar inputs and 3 node directions.

    Every input group has compared components (the node step follows the
    core radius, so the node directions are compared too).
    """
    wrt, outputs = _case_io(case_id, cond, check=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=outputs, wrt=wrt,
                                 n_directions=3, rtol=1e-6, seed=0)
    assert all(c["compared"] > 0 for c in out["groups"].values()), (case_id, out["groups"])
    if "nodes" in wrt:
        assert out["groups"]["nodes"]["compared"] > 0, case_id
    assert out["pass"], (case_id, _failed(out))


def test_translation_identity(float64_precision) -> None:
    """The node gradients plus the ref_point gradient sum to zero (1e-9 of the largest).

    Only the x and z components: a y translation moves the root of a
    symmetric surface off the plane y = 0, so it is not in the input space.
    """
    ac = vt.Aircraft(surfaces=[_plain_wing()], ref_point=np.array([0.5, 0.0, 0.1]))
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(2.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CL", "CDi", "Cm"),
                           wrt=("nodes", "reference"), axes="body")
    for o in ("CL", "CDi", "Cm"):
        j = res.jacobian[o]
        total = np.zeros(3)
        biggest = 0.0
        for arrs in j["nodes"].values():
            for tag in ("le", "te"):
                total += np.asarray(arrs[tag], dtype=float).sum(axis=0)
                biggest = max(biggest, float(np.max(np.abs(arrs[tag]))))
        total += np.asarray(j["reference"]["ref_point"], dtype=float)
        biggest = max(biggest, float(np.max(np.abs(j["reference"]["ref_point"]))))
        assert float(abs(total[0])) <= 1e-9 * biggest, (o, total)
        assert float(abs(total[2])) <= 1e-9 * biggest, (o, total)
    # A non-symmetric surface admits the y translation too: all three components.
    wing = _plain_wing()
    wing.is_symmetric = False
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.5, 0.3, 0.1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CL", "CDi", "Cm"),
                           wrt=("nodes", "reference"), axes="body")
    for o in ("CL", "CDi", "Cm"):
        j = res.jacobian[o]
        total = np.zeros(3)
        biggest = 0.0
        for arrs in j["nodes"].values():
            for tag in ("le", "te"):
                total += np.asarray(arrs[tag], dtype=float).sum(axis=0)
                biggest = max(biggest, float(np.max(np.abs(arrs[tag]))))
        total += np.asarray(j["reference"]["ref_point"], dtype=float)
        biggest = max(biggest, float(np.max(np.abs(j["reference"]["ref_point"]))))
        assert float(np.max(np.abs(total))) <= 1e-9 * biggest, (o, total)


def test_rotation_identity(float64_precision) -> None:
    """Node gradients contracted with a pitch rotation equal the alpha derivatives (1e-9).

    The wing is tapered and its airfoil has alpha_L0 = 0. With alpha_L0 not
    zero the identity does not hold in the production model itself (about
    1e-8): the zero-lift normal turns by -alpha_L0 about the span axis
    projected on the y-z plane (``build_lattice``), and that axis does not
    turn with a pitch of the nodes, so a pitched geometry is not the same
    model as a change of alpha. Moments are compared in wind axes.
    """
    chord = 1.5
    af = vt.LinearAirfoil(name="Sym", a0=2.0 * np.pi, alpha_L0=0.0, Cd0=0.0, Cm0=0.0)
    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0, airfoil=af),
            vt.WingSection(y_frac=1.0, chord=1.0, airfoil=af),
        ],
    )
    ref_pt = np.array([0.25 * chord, 0.0, 0.1])
    ac = vt.Aircraft(surfaces=[wing], ref_point=ref_pt)
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CL", "CDi", "Cl", "Cm", "Cn"),
                           wrt=("nodes", "alpha"), axes="wind")
    from ventorum.geometry.deformation import undeformed_nodes

    nodes = undeformed_nodes(ac, st, solver="vlm")
    for o in ("CL", "CDi", "Cl", "Cm", "Cn"):
        j = res.jacobian[o]
        rot = 0.0
        for surf in ac.surfaces:
            bn = nodes[surf.name]
            rel_le = np.asarray(bn["le"], dtype=float) - ref_pt
            rel_te = np.asarray(bn["te"], dtype=float) - ref_pt
            # Pitch nose up by d: u = (dz, 0, -dx) (see test_node_displacements).
            u_le = np.column_stack([rel_le[:, 2], np.zeros_like(rel_le[:, 0]), -rel_le[:, 0]])
            u_te = np.column_stack([rel_te[:, 2], np.zeros_like(rel_te[:, 0]), -rel_te[:, 0]])
            arrs = j["nodes"][surf.name]
            rot += float(np.sum(np.asarray(arrs["le"]) * u_le)
                         + np.sum(np.asarray(arrs["te"]) * u_te))
        alpha_der = float(j["alpha"])
        if o in ("Cl", "Cn"):
            # Symmetric zeros: absolute comparison.
            assert abs(rot - alpha_der) <= 1e-12, (o, rot, alpha_der)
        else:
            assert abs(rot - alpha_der) <= 1e-9 * max(abs(alpha_der), 1e-300), (o, rot, alpha_der)


def test_alpha_matches_stability_derivatives(float64_precision) -> None:
    """CL_alpha and Cm_alpha equal vt.stability_derivatives within its difference error.

    stability_derivatives uses central differences with alpha +/- 0.5 deg and
    rounds the per-rad values to 5 decimals. Its difference error is
    estimated from the same differences with 0.5 deg and 0.25 deg:
    (4/3) |FD(0.5 deg) - FD(0.25 deg)| (the leading h^2 term), with a
    margin of 25 % for the higher-order terms, plus the rounding.
    """
    import dataclasses

    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=16, n_chord=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CL", "Cm"), wrt=("alpha",), axes="body")
        sd = vt.stability_derivatives(
            aircraft=ac, flight_condition={"V_inf_m_s": 30.0, "alpha_deg": 4.0},
            settings={"solver": "vlm", "n_panels": 16, "n_chord": 2})["stability_derivatives"]
    for o, key in (("CL", "CL_alpha_per_rad"), ("Cm", "Cm_alpha_per_rad")):
        fds = []
        for step in (np.radians(0.5), np.radians(0.25)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v_p = vt.analyze(ac.clone(), dataclasses.replace(cond, alpha=cond.alpha + step), st.clone())
                v_m = vt.analyze(ac.clone(), dataclasses.replace(cond, alpha=cond.alpha - step), st.clone())
            fds.append((float(getattr(v_p.totals, o)) - float(getattr(v_m.totals, o))) / (2.0 * step))
        diff_err = 1.25 * (4.0 / 3.0) * abs(fds[0] - fds[1]) + 0.5e-5
        grad = float(res.jacobian[o]["alpha"])
        assert abs(grad - float(sd[key])) <= diff_err, (o, grad, sd[key], diff_err)


def test_symmetric_fold_rule(float64_precision) -> None:
    """A beta input or a lateral output selects the full system; results equal the full system."""
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sym = vt.gradients(ac, cond, st, outputs=("CL", "CD", "Cm"), wrt=("alpha",), axes="body")
        full_beta = vt.gradients(ac, cond, st, outputs=("CL", "CD", "Cm"), wrt=("alpha", "beta"),
                                 axes="body")
        full_lat = vt.gradients(ac, cond, st, outputs=("CL", "CY"), wrt=("alpha",), axes="body")
    assert sym.symmetric is True
    assert full_beta.symmetric is False
    assert full_lat.symmetric is False
    # The full system is the differentiated model; its zero lateral outputs
    # pass the forward check (relative to the force and moment vectors).
    assert full_beta.forward_mismatch <= 1e-12
    assert full_lat.forward_mismatch <= 1e-12
    for o in ("CL", "CD", "Cm"):
        assert sym.values[o] == full_beta.values[o] == full_lat.values.get(o, sym.values[o])
        assert sym.jacobian[o]["alpha"] == pytest.approx(full_beta.jacobian[o]["alpha"], rel=1e-12)
    assert full_lat.jacobian["CL"]["alpha"] == pytest.approx(sym.jacobian["CL"]["alpha"], rel=1e-12)


def test_vjp_strip_force(float64_precision) -> None:
    """gradient_vjp with a random seed on strip_force equals the Richardson directional derivative."""
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_s = vt.analyze(ac.clone(), cond, st.clone()).details["loads"].strip_force.shape[0]
    rng = np.random.default_rng(3)
    seed = rng.standard_normal((n_s, 3))

    def weighted(alpha: float) -> float:
        import dataclasses

        c = dataclasses.replace(cond, alpha=alpha)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = vt.analyze(ac.clone(), c, st.clone())
        return float(np.sum(seed * res.details["loads"].strip_force))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = vt.gradient_vjp(ac, cond, st, seeds={"strip_force": seed}, wrt=("alpha",), axes="body")
    grad = float(g["alpha"])
    fds = []
    for h in (1e-4, 5e-5):
        fds.append((weighted(cond.alpha + h) - weighted(cond.alpha - h)) / (2.0 * h))
    est = abs(fds[0] - fds[1]) / 3.0
    ref = (4.0 * fds[1] - fds[0]) / 3.0
    assert est <= 0.1 * 1e-6 * max(abs(ref), 1e-300), (fds, est)
    assert abs(grad - ref) <= 1e-6 * max(abs(ref), 1e-300), (grad, ref)


def test_refusals(float64_precision) -> None:
    """Refused configurations and bad names raise ValueError (GradientScopeError for configurations not covered yet)."""
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Unknown names.
        with pytest.raises(ValueError, match="Unknown output"):
            vt.gradients(ac, cond, st, outputs=("CLL",), wrt=("alpha",))
        with pytest.raises(ValueError, match="Unknown input"):
            vt.gradients(ac, cond, st, outputs=("CL",), wrt=("gamma",))
        with pytest.raises(ValueError, match="gradient_vjp"):
            vt.gradients(ac, cond, st, outputs=("strip_force",), wrt=("alpha",))
        # Empty output or input lists name the valid names.
        with pytest.raises(ValueError, match="No output requested.*CL"):
            vt.gradients(ac, cond, st, outputs=(), wrt=("alpha",))
        with pytest.raises(ValueError, match="No input requested.*nodes"):
            vt.gradients(ac, cond, st, outputs=("CL",), wrt=())
        with pytest.raises(ValueError, match="No input requested"):
            vt.gradient_vjp(ac, cond, st, seeds={"CL": 1.0}, wrt=())
        with pytest.raises(ValueError, match="No output requested"):
            vt.check_gradients(ac, cond, st, outputs=(), wrt=("alpha",))
        # Nonlinear solver.
        with pytest.raises(vt.GradientScopeError):
            vt.gradients(ac, cond, vt.SolverSettings(solver_type="nonlinear", n_panels=8),
                         outputs=("CL",), wrt=("alpha",))
        # Fourier solver.
        with pytest.raises(vt.GradientScopeError):
            vt.gradients(ac, cond, vt.SolverSettings(solver_type="fourier", n_panels=8),
                         outputs=("CL",), wrt=("alpha",))
        # Tabulated airfoil.
        af = vt.TabulatedAirfoil(
            alpha=np.radians(np.arange(-4.0, 14.0, 2.0)),
            Cl_data=2.0 * np.pi * np.radians(np.arange(-4.0, 14.0, 2.0)),
            Cd_data=0.01 + np.zeros(9))
        wing = vt.LiftingSurface(name="Wing", semi_span=5.0, sections=[
            vt.WingSection(y_frac=0.0, chord=2.0, airfoil=af),
            vt.WingSection(y_frac=1.0, chord=1.0, airfoil=af)])
        with pytest.raises(vt.GradientScopeError):
            vt.gradients(vt.Aircraft(surfaces=[wing]), cond, st, outputs=("CL",), wrt=("alpha",))
        # Control deflection not zero.
        flap = vt.ControlSurface(name="flap", eta_start=0.3, eta_end=0.7, hinge_x_c=0.75,
                                 deflection=np.radians(5.0))
        wing_c = _plain_wing()
        wing_c.controls = [flap]
        with pytest.raises(vt.GradientScopeError):
            vt.gradients(vt.Aircraft(surfaces=[wing_c]), cond, st, outputs=("CL",), wrt=("alpha",))
        # Body rates not zero.
        import dataclasses

        with pytest.raises(vt.GradientScopeError):
            vt.gradients(ac, dataclasses.replace(cond, q=0.01), st, outputs=("CL",), wrt=("alpha",))
        # Non-finite input.
        with pytest.raises(ValueError, match="finite"):
            vt.gradients(ac, dataclasses.replace(cond, alpha=math.nan), st,
                         outputs=("CL",), wrt=("alpha",))


@pytest.mark.parametrize("case_id,ac,cond,st", _parity_cases(), ids=[c[0] for c in _parity_cases()])
def test_float32_gate(case_id: str, ac: vt.Aircraft, cond: vt.FlightCondition,
                      st: vt.SolverSettings, float32_precision, cpu_device) -> None:
    """Float32 gradients (CPU) pass check_gradients with rtol 1e-3, node directions included."""
    wrt, outputs = _case_io(case_id, cond, check=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=outputs, wrt=wrt,
                                 n_directions=3, rtol=1e-3, seed=0)
    assert all(c["compared"] > 0 for c in out["groups"].values()), (case_id, out["groups"])
    assert out["pass"], (case_id, _failed(out))


@pytest.mark.gpu
def test_gpu_device(gpu_device) -> None:
    """GPU float32 meets the 1e-3 gate; GPU float64 equals CPU float64 to 1e-10.

    The gradients run on the device of the production solve. On the GPU in
    float32 the production solve is float32 too, so its finite differences
    cannot resolve the node directions (their step is below the core
    radius). The reference of the float32 gate is therefore the CPU float64
    adjoint gradient, which check_gradients verifies against Richardson
    differences at 1e-6 in the same test.
    """
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    outs = ("CL", "CDi", "Cm")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpu.set_device("gpu")
        gpu.set_precision("float32")
        res32 = vt.gradients(ac, cond, st, outputs=outs, wrt=("alpha", "nodes"), axes="body")
        gpu.set_precision("float64")
        res_gpu = vt.gradients(ac, cond, st, outputs=outs, wrt=("alpha", "nodes"), axes="body")
        gpu.set_device("cpu")
        res_cpu = vt.gradients(ac, cond, st, outputs=outs, wrt=("alpha", "nodes"), axes="body")
        check = vt.check_gradients(ac, cond, st, outputs=outs, wrt=("alpha", "nodes"),
                                   n_directions=3, rtol=1e-6, seed=0)
    assert check["pass"], _failed(check)
    assert res32.device == "cuda" and res_gpu.device == "cuda" and res_cpu.device == "cpu"
    assert res32.precision_gradient == "float32" and res32.precision_primal == "float32"
    assert res_gpu.precision_primal == "float64"
    rng = np.random.default_rng(0)

    def flat(res: vt.GradientResult, o: str) -> np.ndarray:
        arrs = res.jacobian[o]["nodes"]["Wing"]
        return np.concatenate([np.asarray(arrs["le"]).ravel(), np.asarray(arrs["te"]).ravel()])

    for o in outs:
        g64 = flat(res_cpu, o)
        # Float32 on the GPU: alpha and 3 random node directions within 1e-3.
        a64, a32 = float(res_cpu.jacobian[o]["alpha"]), float(res32.jacobian[o]["alpha"])
        assert abs(a32 - a64) <= 1e-3 * abs(a64), (o, a32, a64)
        for _ in range(3):
            u = rng.standard_normal(g64.shape[0])
            d64, d32 = float(g64 @ u), float(flat(res32, o) @ u)
            assert abs(d32 - d64) <= 1e-3 * abs(d64), (o, d32, d64)
        # Float64 on the GPU equals float64 on the CPU.
        denom = max(abs(res_cpu.values[o]), 1e-300)
        assert abs(res_gpu.values[o] - res_cpu.values[o]) / denom <= 1e-10, o
        scale = float(np.max(np.abs(g64)))
        assert float(np.max(np.abs(flat(res_gpu, o) - g64))) / scale <= 1e-10, o


def _node_vector(res: vt.GradientResult, o: str) -> np.ndarray:
    """Return all node gradients of output *o* as one flat array."""
    return np.concatenate([np.concatenate([np.asarray(v["le"]).ravel(), np.asarray(v["te"]).ravel()])
                           for v in res.jacobian[o]["nodes"].values()])


def test_float32_node_gradients_on_segment_pairs(cpu_device) -> None:
    """Float32 node gradients agree with float64 (regression of the on-segment pairs).

    Load points on their own bound segment and leg mid-points on the legs of
    the same edge line gave O(1/rc^2) partial derivatives that cancel only
    in float64 (float32 node errors up to 2e-2 on this lifting-line case).
    The pairs are now zero by the mesh topology, and the kernel uses the
    projection without cancellation off the segment.
    """
    case = {c[0]: c for c in _parity_cases()}
    out = {}
    for cid in ("linear", "vlm-swept-full"):
        _, ac, cond, st = case[cid]
        for prec in ("float64", "float32"):
            old = gpu.get_precision()
            gpu.set_precision(prec)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out[(cid, prec)] = vt.gradients(ac, cond, st, outputs=("CL", "CD", "Cm"),
                                                    wrt=("nodes", "alpha"))
            finally:
                gpu.set_precision(old)
        assert out[(cid, "float32")].forward_mismatch <= 1e-5, cid
        for o in ("CL", "CD", "Cm"):
            g64 = _node_vector(out[(cid, "float64")], o)
            g32 = _node_vector(out[(cid, "float32")], o)
            assert float(np.max(np.abs(g32 - g64))) <= 1e-4 * float(np.max(np.abs(g64))), (cid, o)


def test_check_reports_group_without_comparison(float64_precision) -> None:
    """check_gradients fails, with a reason, when an input group has no compared component."""
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # No finite difference resolves 1e-14: every alpha component is skipped.
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "Cm"), wrt=("alpha",), rtol=1e-14)
    assert out["groups"]["alpha"]["compared"] == 0
    assert out["pass"] is False
    assert any("'alpha'" in r for r in out["reasons"]), out["reasons"]


def test_check_zero_valued_sections(float64_precision) -> None:
    """Section properties with a zero base value use an absolute step (no false failure)."""
    af = vt.LinearAirfoil(name="Zero", a0=2.0 * np.pi, alpha_L0=0.0, Cd0=0.0, Cm0=0.0)
    wing = vt.LiftingSurface(name="Wing", semi_span=5.0, sections=[
        vt.WingSection(y_frac=0.0, chord=2.0, airfoil=af),
        vt.WingSection(y_frac=1.0, chord=1.0, airfoil=af)])
    ac = vt.Aircraft(surfaces=[wing])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "CD"), wrt=("sections",), rtol=1e-6)
    assert out["pass"], _failed(out)
    for prop in ("alpha_L0", "Cd0", "Cm0"):
        info = out["details"][f"CL:sections.Wing[0].{prop}"]
        assert info["step"] >= 1e-7, (prop, info)
    assert out["details"]["CL:sections.Wing[0].alpha_L0"]["mode"] == "relative"
    assert out["details"]["CD:sections.Wing[0].Cd0"]["mode"] == "relative"


def test_check_has_no_absolute_floor(float64_precision, monkeypatch) -> None:
    """A small derivative with a relative error above rtol fails (no absolute floor)."""
    from ventorum.sensitivity import adjoint

    real = adjoint.gradients

    def perturbed(*args, **kwargs):  # type: ignore[no-untyped-def]
        res = real(*args, **kwargs)
        res.jacobian["CD"]["beta"] *= 1.0 + 1e-5
        return res

    monkeypatch.setattr(adjoint, "gradients", perturbed)
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
    st = vt.SolverSettings(solver_type="linear", n_panels=12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "CD"), wrt=("beta",), rtol=1e-6)
    info = out["details"]["CD:beta"]
    # The absolute error is far below 1e-9, but the relative error is 1e-5.
    assert abs(info["grad"] - info["ref"]) < 1e-9
    assert info["mode"] == "relative" and info["passed"] is False
    assert out["pass"] is False


def test_forward_check_is_strict(float64_precision, monkeypatch) -> None:
    """The forward check has no absolute band and also compares the vector outputs."""
    from ventorum.sensitivity import adjoint

    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    real_values = adjoint._production_values
    real_vectors = adjoint._production_vectors

    def shifted_values(prep):  # type: ignore[no-untyped-def]
        out = real_values(prep)
        out["CDi"] *= 1.0 + 1e-9
        return out

    def shifted_vectors(prep, names):  # type: ignore[no-untyped-def]
        return {k: v * (1.0 + 1e-9) for k, v in real_vectors(prep, names).items()}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        monkeypatch.setattr(adjoint, "_production_values", shifted_values)
        # The CDi difference is about 1e-11 absolute (below the old 1e-9 band).
        with pytest.raises(RuntimeError, match="CDi"):
            vt.gradients(ac, cond, st, outputs=("CL",), wrt=("alpha",))
        monkeypatch.setattr(adjoint, "_production_values", real_values)
        monkeypatch.setattr(adjoint, "_production_vectors", shifted_vectors)
        n_s = vt.analyze(ac.clone(), cond, st.clone()).details["loads"].strip_force.shape[0]
        with pytest.raises(RuntimeError, match="strip_force"):
            vt.gradient_vjp(ac, cond, st, seeds={"strip_force": np.ones((n_s, 3))}, wrt=("alpha",))
        with pytest.raises(RuntimeError, match="section_cl"):
            vt.gradient_vjp(ac, cond, st, seeds={"section_cl": np.ones(n_s)}, wrt=("alpha",))


def test_output_without_graph(float64_precision) -> None:
    """An output that does not depend on the requested inputs has a zero gradient (CDp, alpha)."""
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CDp", "CL"), wrt=("alpha",))
    assert res.jacobian["CDp"]["alpha"] == 0.0
    assert res.jacobian["CL"]["alpha"] > 0.0


def test_chunk_budget_invariance(float64_precision, monkeypatch) -> None:
    """Small chunks (many row chunks in the forward and backward pass) give the same gradients."""
    case = {c[0]: c for c in _parity_cases()}
    _, ac, cond, st = case["vlm-ground"]
    wrt = ("nodes", "alpha", "h", "sections")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = vt.gradients(ac, cond, st, outputs=("CL", "CD", "Cm"), wrt=wrt)
        monkeypatch.setenv("VENTORUM_GRAD_CHUNK", "5e5")
        small = vt.gradients(ac, cond, st, outputs=("CL", "CD", "Cm"), wrt=wrt)
    for o in ("CL", "CD", "Cm"):
        a, b = _node_vector(ref, o), _node_vector(small, o)
        assert float(np.max(np.abs(a - b))) <= 1e-12 * float(np.max(np.abs(a))), o
        for key in ("alpha", "h"):
            assert small.jacobian[o][key] == pytest.approx(ref.jacobian[o][key], rel=1e-12, abs=1e-15)


def test_device_follows_production(float64_precision, cpu_device) -> None:
    """The gradients run on the device of the production solve; precision_primal is its precision."""
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CL",), wrt=("alpha",))
    assert res.device == "cpu"
    assert res.precision_primal == "float64"
    assert res.precision_gradient == "float64"


def test_user_manual_example_runs(float64_precision) -> None:
    """Prove that the gradient how-to example runs as written."""
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs" / "user" / "how_to_gradients.md").read_text(
        encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert len(blocks) >= 1, "the guide must have at least one Python example"
    namespace: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(blocks[0], "docs/user/how_to_gradients.md", "exec"), namespace)  # noqa: S102
    res = namespace["result"]
    assert math.isfinite(res.values["CL"]) and res.values["CL"] > 0.0
