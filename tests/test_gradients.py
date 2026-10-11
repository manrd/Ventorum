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

    Every case has linear airfoils with nonzero Cd0, Cm0 and alpha_L0 and
    beta = 3 deg. Returns (id, aircraft, condition, settings) rows.
    """
    cases = []
    beta = np.radians(3.0)
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


@pytest.mark.parametrize("case_id,ac,cond,st", _parity_cases(), ids=[c[0] for c in _parity_cases()])
def test_forward_parity(case_id: str, ac: vt.Aircraft, cond: vt.FlightCondition,
                        st: vt.SolverSettings, float64_precision) -> None:
    """The reference copy equals the production totals to 1e-12 relative (float64)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=_SCALAR_OUTPUTS, wrt=_full_wrt(cond), axes="body")
    assert res.forward_mismatch <= 1e-12, (case_id, res.forward_mismatch)


@pytest.mark.slow
@pytest.mark.parametrize("case_id,ac,cond,st", _parity_cases(), ids=[c[0] for c in _parity_cases()])
def test_adjoint_vs_richardson(case_id: str, ac: vt.Aircraft, cond: vt.FlightCondition,
                               st: vt.SolverSettings, float64_precision) -> None:
    """check_gradients passes (1e-6) for all scalar inputs and 3 node directions."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "CD", "Cm"), wrt=_full_wrt(cond),
                                 n_directions=3, rtol=1e-6, seed=0)
    assert out["compared"] > 0, case_id
    assert out["pass"], (case_id, {k: v for k, v in out["details"].items() if v.get("pass") is False})


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

    The wing is untapered: on a tapered wing the quarter-chord line is swept,
    and the wake rotation of an alpha change couples with the sweep, so the
    two true derivatives differ at 1.2e-8 (each matches its own finite
    difference to 4e-10). Moments are compared in wind axes.
    """
    chord = 1.5
    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=chord),
            vt.WingSection(y_frac=1.0, chord=chord),
        ],
    )
    ref_pt = np.array([0.25 * chord, 0.0, 0.0])
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
    """CL_alpha and Cm_alpha equal central differences within their difference error."""
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=16, n_chord=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CL", "Cm"), wrt=("alpha",), axes="body")
    for o, step_deg in (("CL", 0.5), ("Cm", 0.5)):
        fds = []
        for step in (np.radians(step_deg), np.radians(step_deg / 2.0)):
            import dataclasses

            c_p = dataclasses.replace(cond, alpha=cond.alpha + step)
            c_m = dataclasses.replace(cond, alpha=cond.alpha - step)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v_p = vt.analyze(ac.clone(), c_p, st.clone()).totals
                v_m = vt.analyze(ac.clone(), c_m, st.clone()).totals
            fds.append((float(getattr(v_p, o)) - float(getattr(v_m, o))) / (2.0 * step))
        est = abs(fds[0] - fds[1]) / 3.0
        grad = float(res.jacobian[o]["alpha"])
        assert abs(grad - fds[1]) <= 10.0 * est + 1e-9 * max(abs(grad), 1e-300), (o, grad, fds, est)


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
    """Refused configurations and bad names raise ValueError (phase-2 card named)."""
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
        # Nonlinear solver.
        with pytest.raises(ValueError, match="T-0071"):
            vt.gradients(ac, cond, vt.SolverSettings(solver_type="nonlinear", n_panels=8),
                         outputs=("CL",), wrt=("alpha",))
        # Fourier solver.
        with pytest.raises(ValueError, match="T-0071"):
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
        with pytest.raises(ValueError, match="T-0071"):
            vt.gradients(vt.Aircraft(surfaces=[wing]), cond, st, outputs=("CL",), wrt=("alpha",))
        # Control deflection not zero.
        flap = vt.ControlSurface(name="flap", eta_start=0.3, eta_end=0.7, hinge_x_c=0.75,
                                 deflection=np.radians(5.0))
        wing_c = _plain_wing()
        wing_c.controls = [flap]
        with pytest.raises(ValueError, match="T-0071"):
            vt.gradients(vt.Aircraft(surfaces=[wing_c]), cond, st, outputs=("CL",), wrt=("alpha",))
        # Body rates not zero.
        import dataclasses

        with pytest.raises(ValueError, match="T-0071"):
            vt.gradients(ac, dataclasses.replace(cond, q=0.01), st, outputs=("CL",), wrt=("alpha",))
        # Non-finite input.
        with pytest.raises(ValueError, match="finite"):
            vt.gradients(ac, dataclasses.replace(cond, alpha=math.nan), st,
                         outputs=("CL",), wrt=("alpha",))


@pytest.mark.slow
@pytest.mark.parametrize("case_id,ac,cond,st", _parity_cases(), ids=[c[0] for c in _parity_cases()])
def test_float32_gate(case_id: str, ac: vt.Aircraft, cond: vt.FlightCondition,
                      st: vt.SolverSettings, float32_precision) -> None:
    """Float32 gradients (CPU) pass check_gradients with rtol 1e-3."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "CD", "Cm"), wrt=_full_wrt(cond),
                                 n_directions=3, rtol=1e-3, seed=0)
    assert out["compared"] > 0, case_id
    assert out["pass"], (case_id, {k: v for k, v in out["details"].items() if v.get("pass") is False})


@pytest.mark.gpu
def test_gpu_device(gpu_device) -> None:
    """GPU float32 passes the 1e-3 gate; GPU float64 equals CPU float64 to 1e-10."""
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    gpu.set_device("gpu")
    gpu.set_precision("float32")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "CDi", "Cm"),
                                 wrt=("alpha", "nodes"), n_directions=1, rtol=1e-3, seed=0)
        assert out["pass"], {k: v for k, v in out["details"].items() if v.get("pass") is False}
        gpu.set_precision("float64")
        res_gpu = vt.gradients(ac, cond, st, outputs=("CL", "CDi", "Cm"),
                               wrt=("alpha", "nodes"), axes="body")
    assert res_gpu.device == "cuda"
    gpu.set_device("cpu")
    gpu.set_precision("float64")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_cpu = vt.gradients(ac, cond, st, outputs=("CL", "CDi", "Cm"),
                               wrt=("alpha", "nodes"), axes="body")
    for o in ("CL", "CDi", "Cm"):
        denom = max(abs(res_cpu.values[o]), 1e-300)
        assert abs(res_gpu.values[o] - res_cpu.values[o]) / denom <= 1e-10, o
        for surf, arrs in res_cpu.jacobian[o]["nodes"].items():
            for tag in ("le", "te"):
                a = np.asarray(arrs[tag], dtype=float)
                b = np.asarray(res_gpu.jacobian[o]["nodes"][surf][tag], dtype=float)
                scale = max(float(np.max(np.abs(a))), 1e-300)
                assert np.max(np.abs(a - b)) / scale <= 1e-10, (o, surf, tag)


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
