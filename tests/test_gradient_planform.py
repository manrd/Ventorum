# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the adjoint gradients with respect to the planform parameters."""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu


@contextlib.contextmanager
def _gpu_settings(device: str | None = None, precision: str | None = None):
    """Set the device and the precision; restore the stored settings after the block."""
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
def cpu_device():
    """Run the test on the CPU; restore the device setting after it."""
    with _gpu_settings(device="cpu"):
        yield


def _linear_section() -> vt.LinearAirfoil:
    return vt.LinearAirfoil(
        name="PlanSec", a0=2.0 * np.pi, alpha_L0=np.radians(-2.0), Cd0=0.008, Cm0=-0.05)


def _plain_wing() -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0, airfoil=_linear_section()),
            vt.WingSection(y_frac=1.0, chord=1.0, airfoil=_linear_section()),
        ],
    )


def _swept_wing() -> vt.LiftingSurface:
    return vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sweep_le=np.radians(15.0),
        dihedral=np.radians(5.0),
        incidence=np.radians(1.0),
        position=np.array([0.5, 0.0, 0.2]),
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0, twist=np.radians(1.0), airfoil=_linear_section()),
            vt.WingSection(y_frac=1.0, chord=1.0, twist=np.radians(-1.0), airfoil=_linear_section()),
        ],
    )


def _kinked_wing() -> vt.LiftingSurface:
    """Four sections, every interior section kinked (stable section breaks).

    Every interior section kinks the chord and the twist and carries a
    distinct airfoil, so small parameter changes keep all breaks and add
    none: finite differences keep the mesh topology.
    """
    af0 = vt.LinearAirfoil(name="KinkA", a0=2.0 * np.pi, alpha_L0=np.radians(-2.0),
                           Cd0=0.008, Cm0=-0.05)
    af1 = vt.LinearAirfoil(name="KinkB", a0=2.0 * np.pi, alpha_L0=np.radians(-1.0),
                           Cd0=0.009, Cm0=-0.04)
    af2 = vt.LinearAirfoil(name="KinkC", a0=2.0 * np.pi, alpha_L0=np.radians(-3.0),
                           Cd0=0.008, Cm0=-0.06)
    af3 = vt.LinearAirfoil(name="KinkD", a0=2.0 * np.pi, alpha_L0=np.radians(-2.0),
                           Cd0=0.010, Cm0=-0.05)
    return vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sweep_le=np.radians(10.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0, twist=np.radians(2.0), airfoil=af0),
            vt.WingSection(y_frac=0.4, chord=1.7, twist=np.radians(0.5), airfoil=af1),
            vt.WingSection(y_frac=0.7, chord=1.1, twist=np.radians(-1.0), airfoil=af2),
            vt.WingSection(y_frac=1.0, chord=0.8, twist=np.radians(-3.0), airfoil=af3),
        ],
    )


def _fragile_wing() -> vt.LiftingSurface:
    """Four smooth sections with one chord kink (a fragile break structure).

    Section parameter changes add new breaks here, so the mesh stations jump
    within a difference step: the check skips those components with a reason.
    """
    return vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sweep_le=np.radians(10.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0, twist=np.radians(2.0),
                           airfoil=_linear_section()),
            vt.WingSection(y_frac=0.4, chord=1.6, twist=np.radians(0.0),
                           airfoil=_linear_section()),
            vt.WingSection(y_frac=0.7, chord=1.2, twist=np.radians(-1.0),
                           airfoil=_linear_section()),
            vt.WingSection(y_frac=1.0, chord=0.8, twist=np.radians(-2.0),
                           airfoil=_linear_section()),
        ],
    )


def _flapped_wing() -> vt.LiftingSurface:
    """Two sections with an undeflected control (a control-surface break)."""
    wing = _plain_wing()
    wing.controls = [vt.ControlSurface(name="flap", eta_start=0.3, eta_end=0.7, hinge_x_c=0.75,
                                       deflection=0.0)]
    return wing


def _zle_wing() -> vt.LiftingSurface:
    """Three sections with explicit x_le and z_le (projected-span convention)."""
    return vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0, x_le=0.0, z_le=0.0, airfoil=_linear_section()),
            vt.WingSection(y_frac=0.5, chord=1.5, x_le=0.4, z_le=0.3, airfoil=_linear_section()),
            vt.WingSection(y_frac=1.0, chord=1.0, x_le=0.9, z_le=0.8, airfoil=_linear_section()),
        ],
    )


def _failed(out: dict) -> dict:
    """Return the failed components and the reasons of a check_gradients result."""
    bad = {k: v for k, v in out["details"].items() if v["compared"] and not v["passed"]}
    return {"reasons": out["reasons"], "failed": bad}


def test_port_parity(float64_precision) -> None:
    """The torch port equals the numpy builder to 1e-14 for every surface feature."""
    from ventorum.geometry.lattice import _surface_mesh_eta, surface_edge_geometry
    from ventorum.sensitivity.planform import build_planform, planform_tensors

    import torch

    fin = vt.LiftingSurface(
        name="Fin", semi_span=2.0, is_symmetric=False, dihedral=np.radians(90.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0, airfoil=_linear_section()),
            vt.WingSection(y_frac=1.0, chord=0.6, airfoil=_linear_section()),
        ])
    twin = vt.LiftingSurface(
        name="Twin", semi_span=1.5, is_symmetric=False,
        position=np.array([3.0, 0.8, 0.2]),
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.8, airfoil=_linear_section()),
            vt.WingSection(y_frac=1.0, chord=0.5, airfoil=_linear_section()),
        ])
    configs = [
        ("plain", vt.Aircraft(surfaces=[_plain_wing()])),
        ("swept", vt.Aircraft(surfaces=[_swept_wing()])),
        ("kinked", vt.Aircraft(surfaces=[_kinked_wing()])),
        ("flapped", vt.Aircraft(surfaces=[_flapped_wing()])),
        ("zle", vt.Aircraft(surfaces=[_zle_wing()])),
        ("fin", vt.Aircraft(surfaces=[fin])),
        ("twin", vt.Aircraft(surfaces=[twin, twin.mirrored("TwinL")])),
        ("wing-tail", vt.Aircraft(surfaces=[_plain_wing(), vt.LiftingSurface(
            name="Tail", semi_span=2.0, position=np.array([4.0, 0.0, 0.5]),
            sections=[vt.WingSection(y_frac=0.0, chord=0.8, airfoil=_linear_section()),
                      vt.WingSection(y_frac=1.0, chord=0.5, airfoil=_linear_section())])])),
    ]
    st = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=2)
    for case_id, ac in configs:
        for collocation in ("vlm", "llt"):
            geom = build_planform(ac, st, collocation)
            ref_semi = max(s.semi_span for s in ac.surfaces)
            tensors = planform_tensors(geom, "cpu", torch.float64, differentiable=False)
            for surf in ac.surfaces:
                tr = geom.surfaces[surf.name]
                n_sp = tr.n_sp
                eta_e, _ = _surface_mesh_eta(surf, st, collocation, ref_semi)
                assert eta_e.shape == (n_sp + 1,), (case_id, surf.name)
                geo = surface_edge_geometry(surf, eta_e)
                nn = tensors.nodes[surf.name]
                le_t, te_t, eta_t = nn["le"], nn["te"], nn["eta"]
                assert le_t.shape == geo["le"].shape, (case_id, surf.name)
                d_eta = float(np.max(np.abs(eta_t.detach().cpu().numpy() - eta_e)))
                d_le = float(np.max(np.abs(le_t.detach().cpu().numpy() - geo["le"])))
                d_te = float(np.max(np.abs(te_t.detach().cpu().numpy() - geo["te"])))
                assert d_eta <= 1e-14, (case_id, collocation, surf.name, d_eta)
                assert d_le <= 1e-14, (case_id, collocation, surf.name, d_le)
                assert d_te <= 1e-14, (case_id, collocation, surf.name, d_te)


def _check_case(ac, cond, st, float64_precision) -> None:
    """Run check_gradients with wrt planform and assert it passes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "CD", "Cm"),
                                 wrt=("planform",), n_directions=0, rtol=1e-6, seed=0)
    assert out["groups"]["planform"]["compared"] > 0, _failed(out)
    assert out["pass"], _failed(out)


def test_planform_richardson_vlm(float64_precision, cpu_device) -> None:
    """Richardson checks of each planform parameter (1e-6) on the VLM parity configurations."""
    beta = np.radians(3.0)
    cases = []
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cases.append((ac, vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta),
                  vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=4)))
    ac = vt.Aircraft(surfaces=[_swept_wing()])
    cases.append((ac, vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta),
                  vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=2)))
    wing = _plain_wing()
    tail = vt.LiftingSurface(
        name="Tail", semi_span=2.0, position=np.array([4.0, 0.0, 0.5]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.8, airfoil=_linear_section()),
                  vt.WingSection(y_frac=1.0, chord=0.5, airfoil=_linear_section())])
    ac = vt.Aircraft(name="WingTail", surfaces=[wing, tail])
    cases.append((ac, vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta),
                  vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=2)))
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    ac.compute_reference_values()
    cases.append((ac, vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=beta,
                                        h=0.3 * float(ac.c_ref)),
                  vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=2)))
    for ac, cond, st in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = vt.gradients(ac, cond, st, outputs=("CL", "CD", "Cm"),
                               wrt=("planform",), axes="body")
        assert res.forward_mismatch <= 1e-12, res.forward_mismatch
        assert res.symmetric is False
        _check_case(ac, cond, st, float64_precision)


def test_planform_richardson_linear(float64_precision, cpu_device) -> None:
    """Richardson check of each planform parameter (1e-6) for the linear lifting line."""
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
    st = vt.SolverSettings(solver_type="linear", n_panels=12)
    _check_case(ac, cond, st, float64_precision)


def test_planform_richardson_breaks(float64_precision, cpu_device) -> None:
    """Richardson checks (1e-6) with section breaks, control breaks and explicit x_le/z_le."""
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=2)
    _check_case(vt.Aircraft(surfaces=[_kinked_wing()]), cond, st, float64_precision)
    _check_case(vt.Aircraft(surfaces=[_flapped_wing()]), cond, st, float64_precision)
    _check_case(vt.Aircraft(surfaces=[_zle_wing()]), cond, st, float64_precision)


def test_planform_topology_jump_skips(float64_precision, cpu_device) -> None:
    """A fragile break structure skips with a reason instead of failing.

    Section changes of the fragile wing add new mesh snaps, so the stations
    jump within the difference step. The check passes with skipped
    components that name the jump; the surface parameters still compare.
    """
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(vt.Aircraft(surfaces=[_fragile_wing()]), cond, st,
                                 outputs=("CL", "CD", "Cm"), wrt=("planform",),
                                 n_directions=0, rtol=1e-6, seed=0)
    assert out["pass"], _failed(out)
    assert out["groups"]["planform"]["compared"] > 0
    jumps = [k for k, v in out["details"].items()
             if not v["compared"] and "jump" in v.get("reason", "")]
    assert len(jumps) > 0, out["groups"]
    assert any("sec[1]" in k for k in jumps), jumps


def test_translation_identity_planform(float64_precision, cpu_device) -> None:
    """A uniform position shift equals the sum of the node gradients (1e-9 of the largest).

    Per surface, dJ/dposition equals the summed node gradients of that surface.
    Over all surfaces, the position gradients plus the ref_point gradient sum
    to zero in x and z (the translation rule), also with a second surface.
    """
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(2.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    for ac in (vt.Aircraft(surfaces=[_plain_wing()], ref_point=np.array([0.5, 0.0, 0.1])),
               vt.Aircraft(surfaces=[_plain_wing(), vt.LiftingSurface(
                   name="Tail", semi_span=2.0, position=np.array([4.0, 0.0, 0.5]),
                   sections=[vt.WingSection(y_frac=0.0, chord=0.8, airfoil=_linear_section()),
                             vt.WingSection(y_frac=1.0, chord=0.5, airfoil=_linear_section())])],
                   ref_point=np.array([0.5, 0.0, 0.1]))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_p = vt.gradients(ac, cond, st, outputs=("CL", "CDi", "Cm"),
                                 wrt=("planform", "reference"), axes="body")
            res_n = vt.gradients(ac, cond, st, outputs=("CL", "CDi", "Cm"),
                                 wrt=("nodes", "reference"), axes="body")
        for o in ("CL", "CDi", "Cm"):
            total = np.zeros(3)
            biggest = 0.0
            for surf in ac.surfaces:
                jp = np.asarray(res_p.jacobian[o]["planform"][surf.name]["position"], dtype=float)
                jn = res_n.jacobian[o]["nodes"][surf.name]
                node_sum = (np.asarray(jn["le"], dtype=float).sum(axis=0)
                            + np.asarray(jn["te"], dtype=float).sum(axis=0))
                biggest = max(biggest, float(np.max(np.abs(jn["le"]))),
                              float(np.max(np.abs(jn["te"]))), float(np.max(np.abs(jp))))
                assert float(np.max(np.abs(jp - node_sum))) <= 1e-9 * biggest, (o, surf.name)
                total += jp
            total += np.asarray(res_p.jacobian[o]["reference"]["ref_point"], dtype=float)
            assert float(abs(total[0])) <= 1e-9 * biggest, (o, total)
            assert float(abs(total[2])) <= 1e-9 * biggest, (o, total)


def test_nodes_planform_refusal(float64_precision) -> None:
    """Requesting nodes and planform together raises ValueError (the same freedom)."""
    ac = vt.Aircraft(surfaces=[_plain_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="planform"):
            vt.gradients(ac, cond, st, outputs=("CL",), wrt=("nodes", "planform"))
        with pytest.raises(ValueError, match="planform"):
            vt.gradient_vjp(ac, cond, st, seeds={"CL": 1.0}, wrt=("nodes", "planform"))
        with pytest.raises(ValueError, match="Unknown input"):
            vt.gradients(ac, cond, st, outputs=("CL",), wrt=("camber",))


def test_planform_sections_consistency(float64_precision, cpu_device) -> None:
    """The planform airfoil incidence agrees with the sections view; combined wrt works."""
    from ventorum.sensitivity.check import _section_weights

    ac = vt.Aircraft(surfaces=[_kinked_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_sec = vt.gradients(ac, cond, st, outputs=("CL", "CD", "Cm"),
                               wrt=("sections",), axes="body")
        res_both = vt.gradients(ac, cond, st, outputs=("CL", "CD", "Cm"),
                                wrt=("planform", "sections"), axes="body")
    ac2 = ac.clone()
    ac2.compute_reference_values()
    from ventorum.solvers.factory import make_solver, resolve_solver_type

    solver = make_solver(resolve_solver_type(st.solver_type))
    lat0 = solver.build(ac2, st, cond)
    base_props = {"a0": np.asarray(lat0.a0, dtype=float),
                  "alpha_L0": np.asarray(lat0.alpha_L0, dtype=float),
                  "Cd0": np.asarray(lat0.Cd0, dtype=float),
                  "Cm0": np.asarray(lat0.Cm0, dtype=float)}
    for o in ("CL", "CD", "Cm"):
        a = np.asarray(res_sec.jacobian[o]["sections"]["alpha_L0"], dtype=float)
        b = np.asarray(res_both.jacobian[o]["sections"]["alpha_L0"], dtype=float)
        scale = float(np.max(np.abs(a)))
        assert float(np.max(np.abs(a - b))) <= 1e-12 * scale, o
        for surf in ac.surfaces:
            for j in range(len(surf.sections)):
                w = _section_weights(ac, solver, st, cond, surf.name, j, base_props)["alpha_L0"]
                expect = float(np.asarray(
                    res_both.jacobian[o]["sections"]["alpha_L0"], dtype=float).reshape(-1) @ w)
                got = float(res_both.jacobian[o]["planform"][surf.name]["sections"][j]["alpha_L0"])
                assert abs(got - expect) <= 1e-9 * max(scale, 1e-300), (o, surf.name, j)


def test_planform_kernels_equal_reference(float64_precision, cpu_device) -> None:
    """The compiled VJP kernels give the planform gradients of the reference copy to 1e-12."""
    from ventorum.sensitivity import kernels as K

    old = K.get_forced_backend()
    try:
        ac = vt.Aircraft(surfaces=[_swept_wing()])
        cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(2.0))
        st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = vt.gradients(ac, cond, st, outputs=("CL", "CD", "Cm"),
                               wrt=("planform", "alpha"), method="reference")
            K.set_vjp_backend("numpy")
            new = vt.gradients(ac, cond, st, outputs=("CL", "CD", "Cm"),
                               wrt=("planform", "alpha"), method="kernels")
        assert new.partials == "kernels:numpy"
        assert new.forward_mismatch <= 1e-12
        for o in ("CL", "CD", "Cm"):
            for surf in ac.surfaces:
                for key in ("semi_span", "sweep_le", "dihedral", "incidence"):
                    a = float(ref.jacobian[o]["planform"][surf.name][key])
                    b = float(new.jacobian[o]["planform"][surf.name][key])
                    assert abs(a - b) <= 1e-12 * max(abs(a), 1e-300), (o, key)
                a = np.asarray(ref.jacobian[o]["planform"][surf.name]["position"], dtype=float)
                b = np.asarray(new.jacobian[o]["planform"][surf.name]["position"], dtype=float)
                assert float(np.max(np.abs(a - b))) <= 1e-12 * float(np.max(np.abs(a))), o
    finally:
        K.set_vjp_backend(old)


def test_howto_planform_block_runs(float64_precision) -> None:
    """Prove that the planform how-to example runs as written."""
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs" / "user" / "how_to_gradients.md").read_text(
        encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    plan = [b for b in blocks if "planform" in b]
    assert len(plan) >= 1, "the guide must have a planform Python example"
    namespace: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(plan[0], "docs/user/how_to_gradients.md", "exec"), namespace)  # noqa: S102
    res = namespace["presult"]
    assert res.jacobian["CL"]["planform"]["Wing"]["semi_span"] != 0.0
