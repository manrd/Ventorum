# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the adjoint gradients of the vortex lattice and the linear lifting line."""

from __future__ import annotations

import contextlib
import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu


@contextlib.contextmanager
def _gpu_settings(device: str | None = None, precision: str | None = None):
    """Set the device and the precision; restore the stored settings after the block.

    The stored precision setting can be ``"auto"``. ``gpu.get_precision()``
    resolves ``"auto"``, so the block keeps the raw setting to restore it.
    """
    old_dev, old_prec = gpu.get_device(), gpu._precision
    try:
        if device is not None:
            gpu.set_device(device)
        if precision is not None:
            gpu.set_precision(precision)
        yield
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)


@pytest.fixture
def float64_precision():
    """Run the test with float64 gradient precision; restore it after it."""
    with _gpu_settings(precision="float64"):
        yield


@pytest.fixture
def float32_precision():
    """Run the test with float32 gradient precision; restore it after it."""
    with _gpu_settings(precision="float32"):
        yield


@pytest.fixture
def cpu_device():
    """Run the test on the CPU; restore the device setting after it."""
    with _gpu_settings(device="cpu"):
        yield


@pytest.fixture
def gpu_device():
    """Run the test with the GPU device selected; restore the settings after it."""
    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    with _gpu_settings():
        yield


def test_gpu_settings_restore_the_auto_precision() -> None:
    """The precision fixtures restore the raw setting ``"auto"``, not the resolved value."""
    with _gpu_settings(precision="auto"):
        with _gpu_settings(precision="float32"):
            assert gpu.get_precision() == "float32"
        assert gpu._precision == "auto"
        with _gpu_settings(device="cpu", precision="float64"):
            assert gpu.get_device() == "cpu"
        assert gpu._precision == "auto"


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


def _float32_gate_failures(out: dict, rtol: float) -> dict:
    """Return the components of a check_gradients result that fail the float32 gate.

    Gate rule: a compared component passes when check_gradients passes it
    (relative to its own value), or when its absolute error is at most
    ``rtol`` times the largest reference magnitude of its column: all
    components ``"<output>:<input>"`` with the same input. A component that
    is tiny against the other outputs of its input (for example CD:phi of a
    wing in ground effect, about 1e-3 of CL:phi) carries the float32
    round-off of the large terms of that column, so its relative error is
    not a measure of the gradient accuracy. ``rtol`` itself is not changed.
    """
    col_scale: dict[str, float] = {}
    for key, v in out["details"].items():
        if v.get("compared") and v.get("ref") is not None:
            col = key.split(":", 1)[1]
            col_scale[col] = max(col_scale.get(col, 0.0), abs(float(v["ref"])))
    bad = {}
    for key, v in out["details"].items():
        if not v.get("compared") or v.get("passed"):
            continue
        col = key.split(":", 1)[1]
        if abs(float(v["grad"]) - float(v["ref"])) > rtol * col_scale[col]:
            bad[key] = v
    return bad


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

    Every input group has compared components. The vortex-lattice node step
    follows the core radius and the lifting-line node step follows the strip
    width, so the node directions are compared too.
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


def test_check_linear_lifting_line_nodes(float64_precision) -> None:
    """check_gradients compares and passes (1e-6) the node directions of the linear lifting line.

    Regression: the node step was 1e-3 of the smallest core radius (here
    about 4e-7 m). At that step the noise of the solve hides the
    derivative, so the check skipped every node direction and failed with
    "no compared component". The lifting-line node step now follows the
    strip width.
    """
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="linear", n_panels=12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "CD", "Cm"), wrt=("nodes",),
                                 n_directions=3, rtol=1e-6, seed=0)
    assert out["groups"]["nodes"]["compared"] > 0, out["groups"]
    assert out["pass"], _failed(out)


def test_linear_node_gradients_polynomial_fit(float64_precision) -> None:
    """The linear lifting-line node gradients agree with a polynomial fit (1e-6).

    The Richardson reference of check_gradients cannot resolve these node
    directions above the noise of the solve. A least-squares polynomial of
    degree 8 through 61 production solves over +/- 1e-5 m along each of 3
    random node directions gives the directional derivative at 0. The gate
    is 1e-6 relative to the largest directional derivative of each output.
    """
    from ventorum.geometry.deformation import undeformed_nodes
    from ventorum.sensitivity.check import _apply_node_step, _gather_node_grad, _production_scalars

    case = next(c for c in _parity_cases() if c[0] == "linear")
    _, ac, cond, st = case
    outs = ("CL", "CD", "Cm")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=outs, wrt=("nodes",), axes="body")
    base_nodes = undeformed_nodes(ac.clone(), st, solver="linear")
    disp0 = {s.name: (s.node_displacements.clone() if s.node_displacements is not None else None)
             for s in ac.surfaces}
    flat = [(s.name, tag, i, c) for s in ac.surfaces for tag in ("le", "te")
            for i in range(int(base_nodes[s.name]["eta"].shape[0])) for c in range(3)]
    frozen = [j for j, (name, tag, i, c) in enumerate(flat)
              if i == 0 and c == 1 and next(s for s in ac.surfaces if s.name == name).is_symmetric]
    rng = np.random.default_rng(0)
    t = np.linspace(-1e-5, 1e-5, 61)
    adj = {o: [] for o in outs}
    fit = {o: [] for o in outs}
    for _ in range(3):
        u = rng.standard_normal(len(flat))
        u[frozen] = 0.0
        u /= np.linalg.norm(u)
        vals = {o: [] for o in outs}
        for v in t:
            ac3 = ac.clone()
            _apply_node_step(ac3, base_nodes, disp0, flat, u, float(v))
            sc = _production_scalars(ac3, cond, st, "body")
            for o in outs:
                vals[o].append(sc[o])
        for o in outs:
            coef = np.polynomial.polynomial.polyfit(t / 1e-5, np.asarray(vals[o]), 8)
            fit[o].append(coef[1] / 1e-5)
            adj[o].append(float(_gather_node_grad(res.jacobian[o], flat) @ u))
    for o in outs:
        scale = max(abs(x) for x in adj[o])
        err = max(abs(a - f) for a, f in zip(adj[o], fit[o])) / scale
        assert err <= 1e-6, (o, err, adj[o], fit[o])


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
    """Refused configurations and bad names raise ValueError (GradientScopeError for the Fourier solver)."""
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
        # Fourier solver (closed form: no adjoint). The nonlinear lifting line,
        # tabulated airfoils, control deflections and body rates are covered
        # (tests/test_gradients_phase2.py).
        with pytest.raises(vt.GradientScopeError):
            vt.gradients(ac, cond, vt.SolverSettings(solver_type="fourier", n_panels=8),
                         outputs=("CL",), wrt=("alpha",))
        import dataclasses

        # Non-finite input.
        with pytest.raises(ValueError, match="finite"):
            vt.gradients(ac, dataclasses.replace(cond, alpha=math.nan), st,
                         outputs=("CL",), wrt=("alpha",))


@pytest.mark.parametrize("case_id,ac,cond,st", _parity_cases(), ids=[c[0] for c in _parity_cases()])
def test_float32_gate(case_id: str, ac: vt.Aircraft, cond: vt.FlightCondition,
                      st: vt.SolverSettings, float32_precision, cpu_device) -> None:
    """Float32 gradients (CPU) pass check_gradients with rtol 1e-3, node directions included.

    The gate rule is the one of :func:`_float32_gate_failures`: relative to
    the component, or absolute against rtol times the largest magnitude of
    its column (the same input over all outputs).
    """
    wrt, outputs = _case_io(case_id, cond, check=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=outputs, wrt=wrt,
                                 n_directions=3, rtol=1e-3, seed=0)
    assert all(c["compared"] > 0 for c in out["groups"].values()), (case_id, out["groups"])
    assert not _float32_gate_failures(out, 1e-3), (case_id, _failed(out))


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
            old = gpu._precision
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


def test_check_zero_mode_needs_a_small_noise_bound(float64_precision, monkeypatch) -> None:
    """A zero derivative counts as compared only when its noise bound is below rtol S."""
    from ventorum.sensitivity import check as C

    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "CD"), wrt=("rho",), rtol=1e-6)
    info = out["details"]["CL:rho"]
    assert info["mode"] == "zero" and info["compared"] and info["passed"], info
    real = C._ladder

    def noisy(*args, **kwargs):  # type: ignore[no-untyped-def]
        # The error bound of every reference becomes 1e6 times larger.
        fd = real(*args, **kwargs)
        return {o: [(h, ref, est * 1e6, rnd * 1e6) for h, ref, est, rnd in c] for o, c in fd.items()}

    monkeypatch.setattr(C, "_ladder", noisy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "CD"), wrt=("rho",), rtol=1e-6)
    info = out["details"]["CL:rho"]
    assert info["mode"] == "skipped" and not info["compared"], info
    assert "noise bound" in info["reason"]
    assert out["groups"]["rho"]["compared"] == 0
    assert out["pass"] is False


def test_forward_check_limit_needs_both_precisions_float64() -> None:
    """The 1e-10 forward-check limit needs a float64 primal and a float64 gradient."""
    torch = pytest.importorskip("torch")
    from ventorum.sensitivity.adjoint import _forward_check

    prod = {"CL": 0.5, "CY": 0.0, "Cl": 0.0, "Cm": -0.1, "Cn": 0.0}
    outputs = {k: torch.tensor(v * (1.0 + 1e-6), dtype=torch.float64) for k, v in prod.items()}
    for primal in ("float32", "float64"):
        for grad_prec in ("float32", "float64"):
            prep = {"precision": grad_prec, "precision_primal": primal,
                    "result": vt.SolverResult(details={"loads": None})}
            if primal == grad_prec == "float64":
                with pytest.raises(RuntimeError, match="above 1e-10"):
                    _forward_check(prep, outputs, prod, ())
            else:
                assert _forward_check(prep, outputs, prod, ()) == pytest.approx(1e-6, rel=1e-6)


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
        res.jacobian["CD"]["beta"] *= 1.0 + 1e-4
        return res

    monkeypatch.setattr(adjoint, "gradients", perturbed)
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
    st = vt.SolverSettings(solver_type="linear", n_panels=12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "CD"), wrt=("beta",), rtol=1e-5)
    info = out["details"]["CD:beta"]
    # The absolute error is below 1e-9, but the relative error is 1e-4, ten
    # times rtol. rtol is 1e-5 (not 1e-6) so that the round-off bound of the
    # central difference (about 2e-13 here) is well below rtol/10 |ref| on
    # every platform; with 1e-6 the component was skipped on Linux.
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


# --- Compiled VJP kernels of the partial derivatives ------------------------------------------


@pytest.fixture
def restore_vjp_backend():
    """Restore the forced VJP backend after the test."""
    from ventorum.sensitivity import kernels as K

    old = K.get_forced_backend()
    yield K
    K.set_vjp_backend(old)


def _all_gradients(res_jac: dict) -> dict[str, np.ndarray]:
    """Return every input group of one output jacobian as a flat array (nodes per surface and tag)."""
    out: dict[str, np.ndarray] = {}
    for key, val in res_jac.items():
        if key == "nodes":
            for s, arrs in val.items():
                for tag in ("le", "te"):
                    out[f"nodes.{s}.{tag}"] = np.asarray(arrs[tag], dtype=float).ravel()
        elif isinstance(val, dict):
            out[key] = np.concatenate([np.atleast_1d(np.asarray(v, dtype=float)).ravel() for v in val.values()])
        else:
            out[key] = np.atleast_1d(np.asarray(val, dtype=float))
    return out


def _assert_same_gradients(ref: vt.GradientResult, new: vt.GradientResult, outputs: tuple, tol: float,
                           tag: str) -> None:
    """Each input group of each output agrees to *tol* relative to its largest component.

    The scalar inputs (alpha, beta, V_inf, rho, h, phi) form one group.
    """
    for o in outputs:
        a, b = _all_gradients(ref.jacobian[o]), _all_gradients(new.jacobian[o])
        assert a.keys() == b.keys(), (tag, o)
        scal = [k for k in a if not k.startswith(("nodes.", "sections", "reference"))]
        groups = {k: [k] for k in a if k not in scal}
        if scal:
            groups["scalars"] = scal
        for g, keys in groups.items():
            x = np.concatenate([a[k] for k in keys])
            y = np.concatenate([b[k] for k in keys])
            scale = float(np.max(np.abs(x)))
            err = float(np.max(np.abs(x - y)))
            assert err <= tol * scale or (scale == 0.0 and err == 0.0), (tag, o, g, err, scale)


@pytest.mark.parametrize("case_id,ac,cond,st", _parity_cases(), ids=[c[0] for c in _parity_cases()])
def test_gradients_kernels_equal_reference(case_id: str, ac: vt.Aircraft, cond: vt.FlightCondition,
                                           st: vt.SolverSettings, float64_precision,
                                           restore_vjp_backend) -> None:
    """The compiled VJP kernels give the gradients of the PyTorch reference copy to 1e-12 (float64).

    All CPU backends, all parity configurations, all inputs and outputs.
    """
    K = restore_vjp_backend
    wrt, outputs = _case_io(case_id, cond)
    backends = [b for b in K.CPU_BACKENDS if K.available(b, "cpu")]
    assert "numba" in backends and "numpy" in backends
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = vt.gradients(ac, cond, st, outputs=outputs, wrt=wrt, method="reference")
        assert ref.partials == "reference:torch"
        for backend in backends:
            K.set_vjp_backend(backend)
            new = vt.gradients(ac, cond, st, outputs=outputs, wrt=wrt, method="kernels")
            assert new.partials == f"kernels:{backend}"
            assert new.method == "adjoint"
            assert new.symmetric == ref.symmetric
            assert new.forward_mismatch <= 1e-12, (case_id, backend, new.forward_mismatch)
            for o in outputs:
                assert new.values[o] == ref.values[o]
            _assert_same_gradients(ref, new, outputs, 1e-12, f"{case_id}:{backend}")


def test_method_keyword(float64_precision, restore_vjp_backend) -> None:
    """method selects the path of the partials; GradientResult.partials names it; the results agree."""
    K = restore_vjp_backend
    K.set_vjp_backend(None)
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(2.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=2)
    outs = ("CL", "CD", "Cm")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = vt.gradients(ac, cond, st, outputs=outs, wrt=("alpha", "nodes"), method="reference")
        ker = vt.gradients(ac, cond, st, outputs=outs, wrt=("alpha", "nodes"), method="kernels")
        auto = vt.gradients(ac, cond, st, outputs=outs, wrt=("alpha", "nodes"))
        vjp_ref = vt.gradient_vjp(ac, cond, st, seeds={"CL": 1.0, "Cm": 2.0}, wrt=("alpha",),
                                  method="reference")
        vjp_ker = vt.gradient_vjp(ac, cond, st, seeds={"CL": 1.0, "Cm": 2.0}, wrt=("alpha",),
                                  method="kernels")
        check = vt.check_gradients(ac, cond, st, outputs=("CL",), wrt=("alpha",), rtol=1e-6,
                                   method="reference")
    assert ref.partials == "reference:torch"
    assert ker.partials == "kernels:" + K.default_backend("cpu")
    path, backend = K.resolve("auto", "cpu", 32)  # 32 panels: the small size class
    assert auto.partials == f"{path}:{backend}"
    assert ref.to_dict()["partials"] == "reference:torch"
    assert check["partials"] == "reference:torch" and check["pass"]
    _assert_same_gradients(ref, ker, outs, 1e-12, "kernels")
    _assert_same_gradients(ref, auto, outs, 1e-12, "auto")
    assert vjp_ker["alpha"] == pytest.approx(vjp_ref["alpha"], rel=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="method"):
            vt.gradients(ac, cond, st, outputs=("CL",), wrt=("alpha",), method="fast")
        with pytest.raises(ValueError, match="method"):
            vt.gradient_vjp(ac, cond, st, seeds={"CL": 1.0}, wrt=("alpha",), method="")


@pytest.mark.parametrize("method", ["kernels", "reference"])
def test_node_fold_equals_full(method: str, float64_precision) -> None:
    """Node inputs of a symmetric request use the fold; the gradients equal the full system (1e-12).

    The full system is selected by one lateral output (CY). The root y
    gradients of the fold are exactly zero (a symmetric output is an even
    function of the root y); the full system gives zero to round-off.
    """
    cases = [
        (vt.Aircraft(surfaces=[_swept_wing()]), vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=4), None),
        (vt.Aircraft(surfaces=[_swept_wing()]), vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=4), 0.6),
        (vt.Aircraft(surfaces=[_plain_wing()]), vt.SolverSettings(solver_type="linear", n_panels=12), None),
        (_wing_tail(), vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=2), None),
    ]
    outs = ("CL", "CDi", "CD", "Cm")
    for ac, st, h in cases:
        cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=h)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold = vt.gradients(ac, cond, st, outputs=outs, wrt=("alpha", "nodes"), method=method)
            full = vt.gradients(ac, cond, st, outputs=outs + ("CY",), wrt=("alpha", "nodes"), method=method)
        assert fold.symmetric is True and full.symmetric is False
        for o in outs:
            for s, arrs in fold.jacobian[o]["nodes"].items():
                for tag in ("le", "te"):
                    a = np.asarray(arrs[tag], dtype=float)
                    b = np.asarray(full.jacobian[o]["nodes"][s][tag], dtype=float)
                    assert a[0, 1] == 0.0
                    assert float(np.max(np.abs(a - b))) <= 1e-12 * float(np.max(np.abs(b))), (o, s, tag)
            assert fold.jacobian[o]["alpha"] == pytest.approx(full.jacobian[o]["alpha"], rel=1e-12)
    # A surface that is not symmetric keeps the full system for node inputs.
    wing = _plain_wing()
    wing.is_symmetric = False
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(vt.Aircraft(surfaces=[wing]), cond, st, outputs=("CL",), wrt=("nodes",), method=method)
    assert res.symmetric is False


@pytest.mark.parametrize("backend", ["numba", "cython", "numpy"])
@pytest.mark.parametrize("case_id,ac,cond,st", _parity_cases(), ids=[c[0] for c in _parity_cases()])
def test_float32_gate_kernels(case_id: str, ac: vt.Aircraft, cond: vt.FlightCondition, st: vt.SolverSettings,
                              backend: str, float32_precision, cpu_device, restore_vjp_backend) -> None:
    """Float32 gradients with the compiled kernels pass the float32 gate with rtol 1e-3.

    The gate rule is the one of :func:`_float32_gate_failures` (relative, or
    absolute against rtol times the largest magnitude of the column).
    """
    K = restore_vjp_backend
    if not K.available(backend, "cpu"):
        pytest.skip(f"The {backend} VJP backend is not available.")
    K.set_vjp_backend(backend)
    wrt, outputs = _case_io(case_id, cond, check=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=outputs, wrt=wrt, n_directions=3, rtol=1e-3, seed=0,
                                 method="kernels")
    assert out["partials"] == f"kernels:{backend}"
    assert all(c["compared"] > 0 for c in out["groups"].values()), (case_id, out["groups"])
    assert not _float32_gate_failures(out, 1e-3), (case_id, _failed(out))


def _assert_same_gradients_per_output(ref: vt.GradientResult, new: vt.GradientResult, outputs: tuple,
                                      tol: float, tag: str) -> None:
    """Each output agrees to *tol* relative to the largest component of its gradient over all inputs.

    This is the normalisation of the gradient gate: the scale of an output
    is the largest magnitude of its *ref* gradient over all inputs (nodes,
    scalars, sections, reference). An output whose *ref* gradient is zero to
    round-off (largest component below 1e-10 of the largest gradient of the
    case, for example dCDp/dalpha of linear airfoils) uses the largest
    gradient of the case as its scale.
    """
    scales = {}
    for o in outputs:
        a = _all_gradients(ref.jacobian[o])
        scales[o] = max(float(np.max(np.abs(v))) for v in a.values())
    case_scale = max(scales.values())
    for o in outputs:
        a, b = _all_gradients(ref.jacobian[o]), _all_gradients(new.jacobian[o])
        assert a.keys() == b.keys(), (tag, o)
        scale = scales[o] if scales[o] > 1e-10 * case_scale else case_scale
        err = max(float(np.max(np.abs(a[k] - b[k]))) for k in a)
        assert err <= tol * scale or (scale == 0.0 and err == 0.0), (tag, o, err, scale)


def _difficult_surface(name: str, span: float, c0: float, c1: float, pos=(0.0, 0.0, 0.0), dih: float = 0.0,
                       sweep: float = 0.0, spacing: str | None = None) -> vt.LiftingSurface:
    """Return a two-section surface of the difficult GPU cases (span and chords in m, angles in rad)."""
    return vt.LiftingSurface(
        name=name, semi_span=span, position=np.array(pos, dtype=float), dihedral=dih, sweep_le=sweep,
        spacing=spacing,
        sections=[vt.WingSection(y_frac=0.0, chord=c0, airfoil=_linear_section()),
                  vt.WingSection(y_frac=1.0, chord=c1, airfoil=_linear_section())])


def _difficult_cases() -> list[tuple]:
    """Two difficult configurations of the GPU kernel tests: (id, aircraft, condition, settings).

    * ``vlm-wing-tail-ground-phi``: wing (sweep and dihedral) and tail in
      ground effect with sideslip and bank (CDi with respect to beta is
      sensitive).
    * ``vlm-tiny-cosine``: a 5 cm swept wing with cosine spacing (Cn with
      respect to the nodes is sensitive).
    """
    cases = []
    ac = vt.Aircraft(name="WT", surfaces=[
        _difficult_surface("Wing", 5.0, 2.0, 1.0, dih=np.radians(4.0), sweep=np.radians(10.0)),
        _difficult_surface("Tail", 2.0, 0.8, 0.5, pos=(4.0, 0.0, 0.5))])
    ac.compute_reference_values()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(2.0), h=0.4 * float(ac.c_ref),
                              phi=np.radians(3.0))
    cases.append(("vlm-wing-tail-ground-phi", ac, cond,
                  vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=1)))
    ac = vt.Aircraft(surfaces=[_difficult_surface("Wing", 0.05, 0.02, 0.004, sweep=np.radians(30.0),
                                                  spacing="cosine")])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(2.0))
    cases.append(("vlm-tiny-cosine", ac, cond, vt.SolverSettings(solver_type="vlm", n_panels=24, n_chord=2)))
    return cases


def _gpu_case_io(case_id: str, cond: vt.FlightCondition) -> tuple[tuple, tuple]:
    """Return (wrt, outputs) of a parity case, or all inputs and scalar outputs for a difficult case."""
    if case_id in ("vlm-wing-tail-ground-phi", "vlm-tiny-cosine"):
        return _full_wrt(cond), _SCALAR_OUTPUTS
    return _case_io(case_id, cond)


@pytest.mark.gpu
def test_gpu_kernels_equal_reference(gpu_device, restore_vjp_backend) -> None:
    """On the GPU the Warp kernels give the gradients of the reference copy (float64 1e-12, float32 1e-4).

    Each output is compared relative to the largest component of its gradient
    over all inputs (the normalisation of the gradient gate). The VJP products
    agree to 1e-5 in float32 (tests/test_vjp_kernels.py). The complete
    gradient of a small output (Cn in a small sideslip) is the difference of
    larger terms, so both float32 paths lose digits there; the float32
    tolerance of the complete gradients is 1e-4, the float32 tolerance of the
    forward check. The cases are the parity configurations and the wing and
    tail in ground effect with bank. The tiny cosine wing differs by about
    1.03e-4 in float32 here; it is in the float32 gate test only.
    """
    K = restore_vjp_backend
    if not K.available("warp", "cuda"):
        pytest.skip("The Warp VJP kernels need warp-lang.")
    K.set_vjp_backend(None)
    gpu.set_device("gpu")
    for case_id, ac, cond, st in _parity_cases() + _difficult_cases()[:1]:
        wrt, outputs = _gpu_case_io(case_id, cond)
        for prec, tol in (("float64", 1e-12), ("float32", 1e-4)):
            gpu.set_precision(prec)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref = vt.gradients(ac, cond, st, outputs=outputs, wrt=wrt, method="reference")
                new = vt.gradients(ac, cond, st, outputs=outputs, wrt=wrt)
            assert new.device == "cuda" and new.partials == "kernels:warp", (case_id, new.partials)
            _assert_same_gradients_per_output(ref, new, outputs, tol, f"{case_id}:{prec}")


@pytest.mark.gpu
def test_gpu_float32_gate_against_float64(gpu_device, restore_vjp_backend) -> None:
    """The Warp float32 gradients pass the 1e-3 gate against the Warp float64 gradients.

    Finite differences of a float32 production solve cannot resolve the node
    steps. The float32 gate on the GPU therefore compares the float32 kernel
    gradients with the float64 kernel gradients (these equal the reference
    copy to 1e-12, see test_gpu_kernels_equal_reference). Each output is
    compared relative to the largest component of its gradient over all
    inputs. The cases are the parity configurations and the two difficult
    cases.
    """
    K = restore_vjp_backend
    if not K.available("warp", "cuda"):
        pytest.skip("The Warp VJP kernels need warp-lang.")
    K.set_vjp_backend(None)
    gpu.set_device("gpu")
    for case_id, ac, cond, st in _parity_cases() + _difficult_cases():
        wrt, outputs = _gpu_case_io(case_id, cond)
        res = {}
        for prec in ("float64", "float32"):
            gpu.set_precision(prec)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res[prec] = vt.gradients(ac, cond, st, outputs=outputs, wrt=wrt, method="kernels")
            assert res[prec].partials == "kernels:warp", (case_id, res[prec].partials)
            assert res[prec].precision_gradient == prec, case_id
        _assert_same_gradients_per_output(res["float64"], res["float32"], outputs, 1e-3, f"{case_id}:gate")


@pytest.mark.parametrize("backend", ["numba", "cython", "numpy"])
def test_cpu_kernels_float32_note(backend: str, cpu_device, restore_vjp_backend) -> None:
    """A float32 request on a CPU kernel path says that the O(n^2) partials are float64 values.

    The note is absent on the reference path and for float64 requests.
    """
    from ventorum.sensitivity.adjoint import CPU_KERNELS_FLOAT64_NOTE

    K = restore_vjp_backend
    if not K.available(backend, "cpu"):
        pytest.skip(f"The {backend} VJP backend is not available.")
    K.set_vjp_backend(backend)
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=1)
    old = gpu._precision
    found = {}
    try:
        for prec in ("float32", "float64"):
            gpu.set_precision(prec)
            for method in ("kernels", "reference"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = vt.gradients(ac, cond, st, outputs=("CL",), wrt=("alpha", "nodes"), method=method)
                found[(prec, method)] = (res.precision_gradient, res.partials, CPU_KERNELS_FLOAT64_NOTE in res.notes)
    finally:
        gpu.set_precision(old)
    assert found[("float32", "kernels")] == ("float32", f"kernels:{backend}", True)
    assert found[("float32", "reference")] == ("float32", "reference:torch", False)
    assert found[("float64", "kernels")] == ("float64", f"kernels:{backend}", False)
    assert found[("float64", "reference")] == ("float64", "reference:torch", False)


def _wing_tail_fin_flap() -> vt.Aircraft:
    """Return a wing (flap at zero deflection), a tail and a vertical fin on the plane y = 0.

    The flap edges put leg mid-points of the wing exactly on end points of
    other wing segments.
    """
    flap = vt.ControlSurface(name="flap", eta_start=0.2, eta_end=0.6, hinge_x_c=0.7, deflection=0.0)
    wing = _difficult_surface("Wing", 5.0, 2.0, 1.0, dih=np.radians(4.0), sweep=np.radians(10.0))
    wing.controls = [flap]
    tail = _difficult_surface("Tail", 2.0, 0.8, 0.5, pos=(4.0, 0.0, 0.5))
    fin = _difficult_surface("Fin", 1.2, 0.9, 0.5, pos=(4.2, 0.0, 0.5), dih=np.pi / 2)
    fin.is_symmetric = False
    return vt.Aircraft(name="WTF", surfaces=[wing, tail, fin])


def test_kernels_target_on_end_point_float32() -> None:
    """The reference kernels stay finite in float32 at a target on a segment end point.

    Regression: the floor 1e-300 of the squared distances rounds to zero in
    float32, and the unit vectors became 0/0 (the forward check then raised
    with "worst inf").
    """
    import torch

    from ventorum.sensitivity.reference import ray_vel, seg_vel

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.4, 0.1]])
    A = np.array([[0.0, 0.0, 0.0], [0.5, -0.2, 0.0]])
    B = np.array([[1.0, 0.0, 0.0], [0.5, 0.8, 0.0]])
    d = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    rc2 = torch.full((3, 2), 1e-4, dtype=torch.float64)
    for fn, args in ((seg_vel, (A, B)), (ray_vel, (A, d))):
        v64 = fn(torch.tensor(pts), *(torch.tensor(a) for a in args), rc2).numpy()
        v32 = fn(torch.tensor(pts, dtype=torch.float32), *(torch.tensor(a, dtype=torch.float32) for a in args),
                 rc2.float()).numpy()
        assert np.all(np.isfinite(v32)), fn.__name__
        assert np.allclose(v32, v64, rtol=1e-5, atol=1e-6 * float(np.max(np.abs(v64)))), fn.__name__


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_float32_reference_with_target_on_end_point(device: str, restore_vjp_backend) -> None:
    """Float32 reference gradients of a wing with a flap, a tail and a fin pass the forward check.

    The float32 gradients of the reference path agree with the float64
    gradients to 1e-3 (per output, relative to the largest component).
    """
    if device == "gpu" and not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    ac = _wing_tail_fin_flap()
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(5.0), beta=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=4)
    outs = ("CL", "CY", "CD", "Cl", "Cm", "Cn")
    old_dev, old_prec = gpu.get_device(), gpu._precision
    try:
        gpu.set_device(device)
        res = {}
        for prec in ("float64", "float32"):
            gpu.set_precision(prec)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res[prec] = vt.gradients(ac, cond, st, outputs=outs, wrt=("nodes", "alpha", "beta"),
                                         method="reference")
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)
    assert res["float32"].precision_gradient == "float32"
    assert res["float32"].forward_mismatch <= 1e-4
    _assert_same_gradients_per_output(res["float64"], res["float32"], outs, 1e-3, f"wtf-flap0:{device}")


def test_check_noise_bound_skips_noisy_reference() -> None:
    """A finite-difference reference at the noise level of the solve is skipped, not compared.

    Regression: the round-off bound of the node directions was eps max(abs(f))
    / h, but a node motion changes the rounding of the whole lattice. The
    measured noise of the production solve was about 80 eps max(abs(f)) on the
    wing and tail in ground effect with one chordwise panel, and the check
    reported a false failure of 2e-6.
    """
    from ventorum.sensitivity import check as C

    rng = np.random.default_rng(3)
    sigma = 1e-14
    slope = 0.03

    def get(x: float) -> dict[str, float]:
        return {"CL": 0.6 + slope * x + 5.0 * x * x + sigma * float(rng.standard_normal())}

    noise = C._noise_level(get, 0.0, 1e-10, ("CL",))
    assert 0.3 * sigma <= noise["CL"] <= 3.0 * sigma, noise
    eps = float(np.finfo(np.float64).eps)
    h = 2.5e-7
    _, _, rnd_eps = C._richardson(get, 0.0, h, ("CL",), eps)["CL"]
    _, _, rnd = C._richardson(get, 0.0, h, ("CL",), eps, noise)["CL"]
    # The eps model claims a resolved reference; the measured noise does not.
    assert rnd_eps <= 0.1 * 1e-6 * slope
    assert rnd > 0.1 * 1e-6 * slope
    assert rnd >= C._NOISE_FACTOR * noise["CL"] / h


@pytest.mark.slow
def test_check_ground_effect_one_chordwise_panel(float64_precision, cpu_device) -> None:
    """check_gradients passes (1e-6) on the wing and tail in ground effect with one chordwise panel.

    Regression of a false failure (CL along a node direction, 2.1e-6): the
    adjoint gradient agrees with a polynomial fit of 61 production solves
    to 1e-8, but the finite-difference reference had the noise of the
    production solve, which the eps model of the round-off did not include.
    """
    case_id, ac, cond, st = _difficult_cases()[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "CD", "Cm", "Cn"), wrt=_full_wrt(cond),
                                 n_directions=3, rtol=1e-6, seed=0, method="reference")
    assert out["pass"], _failed(out)
    assert out["groups"]["nodes"]["compared"] > 0


@pytest.mark.parametrize("t", [-0.5, 1.75])
def test_reference_segment_derivative_at_collinear_point(t: float) -> None:
    """Central differences of seg_vel at an exactly collinear point outside the segment do not depend on the step.

    The target lies on the extended line of the segment, before A (t < 0)
    or after B (t > 1). The core uses the distance to the nearest end
    point, so the velocity is zero there but its derivative across the line
    is the exact-law derivative times the core factor. Central differences
    with steps from 1e-8 m to 1e-3 m agree with each other and with the
    autograd derivative.
    """
    torch = pytest.importorskip("torch")
    from ventorum.sensitivity.reference import seg_vel

    A = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    B = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float64)
    rc2 = torch.tensor([[0.25]], dtype=torch.float64)
    # Exactly collinear in binary floating point; the step is along z, where
    # the coordinate of the target is 0, so the step is exact.
    P0 = torch.tensor([[t, 0.5 * t, 0.0]], dtype=torch.float64)
    assert float(torch.abs(seg_vel(P0, A, B, rc2)).max()) == 0.0
    P = P0.clone().requires_grad_(True)
    v = seg_vel(P, A, B, rc2)[0, 0]
    jac = torch.stack([torch.autograd.grad(v[k], P, retain_graph=True)[0][0, 2] for k in range(3)])
    assert float(torch.abs(jac).max()) > 0.0
    for h in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        dz = torch.tensor([[0.0, 0.0, h]], dtype=torch.float64)
        fd = (seg_vel(P0 + dz, A, B, rc2) - seg_vel(P0 - dz, A, B, rc2))[0, 0] / (2.0 * h)
        err = float(torch.abs(fd - jac).max()) / float(torch.abs(jac).max())
        assert err <= 1e-5, (h, err)
