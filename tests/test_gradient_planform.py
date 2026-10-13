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


def _section(y: float, chord: float, twist_deg: float = 0.0, x_le=None, z_le=None,
             airfoil=None) -> vt.WingSection:
    return vt.WingSection(y_frac=y, chord=chord, twist=np.radians(twist_deg), x_le=x_le,
                          z_le=z_le, airfoil=airfoil if airfoil is not None else _linear_section())


def _cranked_wing() -> vt.LiftingSurface:
    """Three sections, 25 deg sweep, chord and twist kink at y_frac 0.35."""
    return vt.LiftingSurface(
        name="W", semi_span=6.0, sweep_le=np.radians(25.0), dihedral=np.radians(4.0),
        incidence=np.radians(2.0), position=np.array([1.0, 0.0, 0.3]),
        sections=[_section(0.0, 3.0, 2.0), _section(0.35, 2.2, 1.0), _section(1.0, 0.9, -2.0)])


def _coincident_wing() -> vt.LiftingSurface:
    """Three sections with explicit x_le and z_le; the break at 0.5 lies on a mesh edge."""
    return vt.LiftingSurface(
        name="W", semi_span=4.0,
        sections=[_section(0.0, 2.0, 1.0, x_le=0.0, z_le=0.0),
                  _section(0.5, 1.5, 0.0, x_le=0.5, z_le=0.25),
                  _section(1.0, 1.0, -1.0, x_le=1.2, z_le=0.9)])


def _production_cl(ac: vt.Aircraft, cond, st) -> float:
    return float(vt.analyze(ac.clone(), cond, st.clone()).totals.CL)


def test_planform_linear_auto_spacing_cranked(float64_precision, cpu_device) -> None:
    """Regression: the linear lifting line uses the lifting-line rule of the auto spacing.

    The chain used the vortex-lattice spacing rule for the lifting line, so the
    port stations differed from the production stations and the forward check
    raised. Now the forward check passes and every planform parameter of a
    swept cranked wing matches the Richardson differences (1e-6).
    """
    from ventorum.geometry.lattice import _surface_mesh_eta
    from ventorum.sensitivity.planform import build_planform

    ac = vt.Aircraft(surfaces=[_cranked_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
    st = vt.SolverSettings(solver_type="linear", n_panels=16)
    geom = build_planform(ac, st, "llt")
    eta_e, _ = _surface_mesh_eta(ac.surfaces[0], st, "llt", 6.0)
    assert np.array_equal(geom.surfaces["W"].eta_base, eta_e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CL", "CD", "Cm"), wrt=("planform",))
    assert res.forward_mismatch <= 1e-12, res.forward_mismatch
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "CD", "Cm"),
                                 wrt=("planform",), n_directions=0, rtol=1e-6, seed=0)
    assert out["pass"], _failed(out)
    assert out["groups"]["planform"]["compared"] > 0, out["groups"]
    assert not any("jump" in str(v.get("reason")) for v in out["details"].values())
    # Some components do not resolve to rtol/10 in check_gradients (the lifting
    # line differences have a larger noise); compare them here with one
    # Richardson step at 1e-6.
    jac = res.jacobian["CL"]["planform"]["W"]
    setters = {
        "semi_span": (lambda a, v: setattr(a.surfaces[0], "semi_span", v), 6.0,
                      jac["semi_span"]),
        "sweep_le": (lambda a, v: setattr(a.surfaces[0], "sweep_le", v), np.radians(25.0),
                     jac["sweep_le"]),
        "dihedral": (lambda a, v: setattr(a.surfaces[0], "dihedral", v), np.radians(4.0),
                     jac["dihedral"]),
        "sec[1].y_frac": (lambda a, v: setattr(a.surfaces[0].sections[1], "y_frac", v), 0.35,
                          jac["sections"][1]["y_frac"]),
        "sec[1].chord": (lambda a, v: setattr(a.surfaces[0].sections[1], "chord", v), 2.2,
                         jac["sections"][1]["chord"]),
    }
    h = 1e-3
    for key, (setter, x0, g) in setters.items():
        def cl_at(v: float, _s=setter) -> float:
            a = ac.clone()
            _s(a, v)
            return _production_cl(a, cond, st)

        d1 = (cl_at(x0 + h) - cl_at(x0 - h)) / (2.0 * h)
        d2 = (cl_at(x0 + 0.5 * h) - cl_at(x0 - 0.5 * h)) / h
        ref = (4.0 * d2 - d1) / 3.0
        assert abs(float(g) - ref) <= 1e-6 * abs(ref), (key, float(g), ref)


def test_planform_break_count_limit(float64_precision, cpu_device) -> None:
    """Regression: the station trace applies the break-count limit of the numpy builder.

    Five sections with distinct airfoils give three breaks; with five panels
    the limit max(1, 5 // 3) = 1 keeps the plain spacing. The trace snapped
    the breaks anyway and the chain refused to run.
    """
    from ventorum.geometry.lattice import _surface_mesh_eta
    from ventorum.sensitivity.planform import build_planform

    def af(name: str, a_l0: float, cd0: float = 0.008) -> vt.LinearAirfoil:
        return vt.LinearAirfoil(name=name, a0=2.0 * np.pi, alpha_L0=np.radians(a_l0),
                                Cd0=cd0, Cm0=-0.05)

    wing = vt.LiftingSurface(
        name="W", semi_span=5.0, sweep_le=np.radians(-8.0),
        sections=[_section(0.0, 2.0, 1.0, airfoil=af("a", -2.0)),
                  _section(0.2, 1.9, 0.5, airfoil=af("b", -1.0)),
                  _section(0.5, 1.5, 0.0, airfoil=af("c", -3.0)),
                  _section(0.8, 1.2, -1.0, airfoil=af("d", -2.0, 0.01)),
                  _section(1.0, 0.6, -3.0, airfoil=af("e", -1.5))])
    ac = vt.Aircraft(surfaces=[wing])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=5, n_chord=1)
    geom = build_planform(ac, st, "vlm")
    eta_e, eta_mid = _surface_mesh_eta(wing, st, "vlm", 5.0)
    assert np.array_equal(geom.surfaces["W"].eta_base, eta_e)
    assert np.array_equal(geom.surfaces["W"].mid_base, eta_mid)
    assert all(s is None for s in geom.surfaces["W"].snap_sec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CL",), wrt=("planform",))
    assert res.forward_mismatch <= 1e-12, res.forward_mismatch


def test_planform_control_on_break_edge(float64_precision) -> None:
    """The control-limit snap of the trace equals the numpy rule when a limit is on a moved edge."""
    from ventorum.geometry.lattice import _surface_mesh_eta
    from ventorum.sensitivity.planform import build_planform

    wing = _cranked_wing()
    st = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=2)
    geom0 = build_planform(vt.Aircraft(surfaces=[wing]), st, "vlm")
    tr = geom0.surfaces["W"]
    i = next(k for k, s in enumerate(tr.snap_sec) if s is not None)
    wing.controls = [vt.ControlSurface(name="flap", eta_start=float(tr.eta_base[i]),
                                       eta_end=0.8, hinge_x_c=0.75, deflection=0.0)]
    geom = build_planform(vt.Aircraft(surfaces=[wing]), st, "vlm")
    eta_e, eta_mid = _surface_mesh_eta(wing, st, "vlm", 6.0)
    assert np.array_equal(geom.surfaces["W"].eta_base, eta_e)
    assert np.array_equal(geom.surfaces["W"].mid_base, eta_mid)


def test_planform_coincident_break(float64_precision, cpu_device) -> None:
    """Regression: a break exactly on a mesh edge gives the derivative of the moved branch.

    Production moves the edge for any nonzero y_frac step, so central
    differences converge to the derivative of the edge-mean branch. The
    gradient matches them (1e-6), a note names the section, and
    check_gradients compares that component.
    """
    from ventorum.sensitivity.planform import build_planform

    ac = vt.Aircraft(surfaces=[_coincident_wing()])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=2)
    geom = build_planform(ac, st, "vlm")
    assert geom.coincident == [("W", 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CL", "CD", "Cm"), wrt=("planform",))
    assert res.forward_mismatch <= 1e-12, res.forward_mismatch
    assert any("section 1" in n and "lies exactly on a mesh edge" in n for n in res.notes), res.notes
    g = float(res.jacobian["CL"]["planform"]["W"]["sections"][1]["y_frac"])

    def cl_at(v: float) -> float:
        a = ac.clone()
        a.surfaces[0].sections[1].y_frac = v
        return _production_cl(a, cond, st)

    fd = {h: (cl_at(0.5 + h) - cl_at(0.5 - h)) / (2.0 * h) for h in (2e-4, 1e-4)}
    ref = (4.0 * fd[1e-4] - fd[2e-4]) / 3.0
    assert abs(g - ref) <= 1e-6 * abs(ref), (g, ref)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "CD", "Cm"),
                                 wrt=("planform",), n_directions=0, rtol=1e-6, seed=0)
    assert out["pass"], _failed(out)
    for o in ("CL", "CD", "Cm"):
        assert out["details"][f"{o}:planform.W.sec[1].y_frac"]["compared"], o


def test_planform_mirror_pair_auto_refs(float64_precision, cpu_device) -> None:
    """Mirror pairs: a note and skips with automatic references; all pass with explicit ones."""
    twin = vt.LiftingSurface(
        name="Twin", semi_span=1.5, is_symmetric=False, position=np.array([3.0, 0.8, 0.2]),
        sweep_le=np.radians(10.0), sections=[_section(0.0, 0.8, 1.0), _section(1.0, 0.5, -1.0)])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=2)
    ac = vt.Aircraft(surfaces=[twin, twin.mirrored("TwinL")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vt.gradients(ac, cond, st, outputs=("CL",), wrt=("planform",))
        out = vt.check_gradients(ac, cond, st, outputs=("CL", "Cm"), wrt=("planform",),
                                 n_directions=0, rtol=1e-6, seed=0)
    assert any("mirror pair" in n for n in res.notes), res.notes
    pf = [v for k, v in out["details"].items() if k.startswith("planform.")]
    assert pf and all(not v["compared"] and "mirror pair" in v["reason"] for v in pf)
    assert out["groups"]["planform"]["compared"] == 0
    ac_ex = ac.clone()
    ac_ex.S_ref, ac_ex.b_ref, ac_ex.c_ref = 2.0, 4.0, 0.65
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_ex = vt.gradients(ac_ex, cond, st, outputs=("CL",), wrt=("planform",))
        out_ex = vt.check_gradients(ac_ex, cond, st, outputs=("CL", "Cm"), wrt=("planform",),
                                    n_directions=0, rtol=1e-6, seed=0)
    assert not any("mirror pair" in n for n in res_ex.notes)
    assert out_ex["pass"], _failed(out_ex)
    # With explicit reference values every planform component is compared,
    # except a component whose reference is not resolved by the differences.
    # check_gradients skips it (mode "skipped", with its error bound): the
    # owner rule accepts the skip when the round-off bound is larger than
    # rtol |value|. The count depends on the platform (one or two components
    # on the CI machines), so no count is fixed here. A skipped component
    # with a smaller bound must still agree with the central difference to
    # rtol relative.
    rtol = 1e-6
    for key, v in out_ex["details"].items():
        if ":planform." not in key or v["compared"]:
            continue
        assert v["mode"] == "skipped" and "bound" in v, (key, v)
        if v["bound"] <= rtol * abs(v["ref"]):
            assert abs(v["grad"] - v["ref"]) <= rtol * abs(v["ref"]), (key, v)
    assert out_ex["groups"]["planform"]["compared"] > 0


def test_planform_refuses_unsorted_sections(float64_precision) -> None:
    """The port refuses sections that are not ordered by y_frac (the numpy builder sorts)."""
    from ventorum.sensitivity.planform import build_planform

    wing = _cranked_wing()
    wing.sections = [wing.sections[0], wing.sections[2], wing.sections[1]]
    st = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=2)
    with pytest.raises(ValueError, match="ordered by y_frac"):
        build_planform(vt.Aircraft(surfaces=[wing]), st, "vlm")
    with pytest.raises(ValueError, match="collocation"):
        build_planform(vt.Aircraft(surfaces=[_cranked_wing()]), st, "linear")
