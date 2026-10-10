# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests of the agent tools of the P1 features: control surfaces, node
displacements, the trim solver and the layer-1 numerical error bars.

Every test compares a tool with the Python API, never with the tool
itself. The tool payloads round some values (wing analysis: 6 decimals
for CL, 7 for CDi); the reference then uses the same documented
rounding, so the comparison still checks the full input path at
rel 1e-12. The new tools (trim, error bars, undeformed nodes) return
full-precision values.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import ventorum as vt
from ventorum.agent import (
    batch_evaluate,
    call_tool,
    error_bars,
    get_tool_schemas,
    ground_effect,
    mesh_convergence,
    polar_sweep,
    stability_derivatives,
    trim,
    undeformed_nodes,
    wing_analysis,
)
from ventorum.agent.schemas import (
    CONTROL_PROPS,
    NODE_DISPLACEMENT_PROPS,
    SURFACE_PROPS,
    build_aircraft_from_spec,
)

FAST = {"n_panels": 8}
RECT = {"span_m": 10.0, "chord_m": 1.0}
WING_TAIL = {
    "surfaces": [
        {"name": "wing", "span_m": 12.0, "root_chord_m": 1.8, "tip_chord_m": 1.0},
        {"name": "tail", "span_m": 4.0, "chord_m": 0.7, "position_m": [4.6, 0.0, 0.3],
         "controls": [{"name": "elevator", "hinge_x_c": 0.7}]},
    ],
    "ref_point_m": [0.45, 0.0, 0.0],
}


def strict_json(payload):
    """The payload must be strict JSON (no NaN, no infinity)."""
    return json.loads(json.dumps(payload, allow_nan=False))


def assert_error(payload, kind, text=None):
    assert payload["status"] == "error", payload
    assert payload["error"]["type"] == kind, payload
    if text is not None:
        assert text in payload["error"]["message"], payload["error"]["message"]
    strict_json(payload)


def _python_wing_tail():
    wing = vt.LiftingSurface(
        name="Wing", semi_span=6.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.8), vt.WingSection(y_frac=1.0, chord=1.0)])
    tail = vt.LiftingSurface(
        name="tail", semi_span=2.0, position=np.array([4.6, 0.0, 0.3]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.7), vt.WingSection(y_frac=1.0, chord=0.7)],
        controls=[vt.ControlSurface(name="elevator", hinge_x_c=0.7)])
    return vt.Aircraft(name="Aircraft", surfaces=[wing, tail], ref_point=np.array([0.45, 0.0, 0.0]))


# 1. Controls through the tool equal the Python API ────────────────────────────

def test_controls_key_equals_python_api():
    """A wing with a flap (and a tail with an elevator) through the tool."""
    wing = {"span_m": 12.0, "root_chord_m": 1.6, "tip_chord_m": 0.8,
            "controls": [{"name": "flap", "eta_start": 0.2, "eta_end": 0.8,
                           "hinge_x_c": 0.75, "deflection_deg": 5.0}]}
    tail = {"name": "tail", "span_m": 4.0, "chord_m": 0.7, "position_m": [4.6, 0.0, 0.3],
            "controls": [{"name": "elevator", "hinge_x_c": 0.7, "deflection_deg": -2.0}]}
    spec = {"surfaces": [wing, tail], "ref_point_m": [0.45, 0.0, 0.0]}
    cond = {"V_inf_m_s": 40.0, "alpha_deg": 4.0}
    p = wing_analysis(spec, cond, {"n_panels": 12}, "summary")
    strict_json(p)
    assert p["status"] == "success", p

    w = vt.LiftingSurface(
        name="Wing", semi_span=6.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.6), vt.WingSection(y_frac=1.0, chord=0.8)],
        controls=[vt.ControlSurface(name="flap", eta_start=0.2, eta_end=0.8,
                                    hinge_x_c=0.75, deflection=np.radians(5.0))])
    t = vt.LiftingSurface(
        name="tail", semi_span=2.0, position=np.array([4.6, 0.0, 0.3]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.7), vt.WingSection(y_frac=1.0, chord=0.7)],
        controls=[vt.ControlSurface(name="elevator", hinge_x_c=0.7, deflection=np.radians(-2.0))])
    ac = vt.Aircraft(name="Aircraft", surfaces=[w, t], ref_point=np.array([0.45, 0.0, 0.0]))
    ref = vt.analyze(ac, vt.FlightCondition(V_inf=40.0, alpha=np.radians(4.0)),
                     vt.SolverSettings(n_panels=12))
    m = p["metrics"]
    assert m["CL"] == pytest.approx(round(ref.totals.CL, 6), rel=1e-12)
    assert m["CDi"] == pytest.approx(round(ref.totals.CDi, 7), rel=1e-12)
    assert m["Cm"] == pytest.approx(round(ref.totals.Cm, 6), rel=1e-12)
    # The deflection is not ignored: the lift differs from the clean wing.
    clean = wing_analysis({"surfaces": [{k: v for k, v in wing.items() if k != "controls"}, tail]},
                          cond, {"n_panels": 12}, "summary")
    assert clean["status"] == "success", clean
    assert m["CL"] != pytest.approx(clean["metrics"]["CL"])


# 2. Aileron on a symmetric wing and on a mirror pair ──────────────────────────

def test_aileron_on_mirror_pair_and_symmetric_surface():
    """symmetric false gives Cl < 0 for a positive deflection, as in docs/user/geometry.md."""
    ctrl = [{"name": "aileron", "eta_start": 0.6, "eta_end": 1.0,
             "deflection_deg": 5.0, "symmetric": False}]
    cond = {"alpha_deg": 4.0}
    p = wing_analysis({**RECT, "controls": ctrl}, cond, {"n_panels": 12}, "summary")
    strict_json(p)
    assert p["status"] == "success", p
    assert p["metrics"]["Cl"] < 0

    wing = vt.LiftingSurface(
        name="Wing", semi_span=5.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
        controls=[vt.ControlSurface(name="aileron", eta_start=0.6, eta_end=1.0,
                                    deflection=np.radians(5.0), symmetric=False)])
    ref = vt.analyze(vt.Aircraft(surfaces=[wing]),
                     vt.FlightCondition(alpha=np.radians(4.0)), vt.SolverSettings(n_panels=12))
    assert p["metrics"]["Cl"] == pytest.approx(round(ref.totals.Cl, 6), rel=1e-12)

    half = {"name": "right", "semi_span_m": 5.0, "chord_m": 1.0,
            "symmetric": False, "mirror": True, "controls": ctrl}
    q = wing_analysis(half, cond, {"n_panels": 12}, "summary")
    strict_json(q)
    assert q["status"] == "success", q
    right = vt.LiftingSurface(
        name="right", semi_span=5.0, is_symmetric=False,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
        controls=[vt.ControlSurface(name="aileron", eta_start=0.6, eta_end=1.0,
                                    deflection=np.radians(5.0), symmetric=False)])
    pair = vt.Aircraft(surfaces=[right, right.mirrored(name="left")])
    ref2 = vt.analyze(pair, vt.FlightCondition(alpha=np.radians(4.0)),
                      vt.SolverSettings(n_panels=12))
    assert q["metrics"]["Cl"] == pytest.approx(round(ref2.totals.Cl, 6), rel=1e-12)


# 3. Control input errors ──────────────────────────────────────────────────────

def test_controls_input_errors():
    """Bad controls are invalid_input that names the key."""
    cases = [
        ({"name": "f", "deflection_deg": 31.0}, "deflection_deg"),
        ({"name": "f", "eta_start": 0.8, "eta_end": 0.2}, "eta_start"),
        ({"name": "f", "hinge_x_c": 1.0}, "hinge_x_c"),
        ({"deflection_deg": 2.0}, "'name'"),
        ([{"name": "f"}, {"name": "f"}], "'name'"),
        ({"name": "f", "deflection": 5.0}, "deflection_deg"),  # the hint names deflection_deg
    ]
    for ctrl, key in cases:
        spec = {**RECT, "controls": ctrl if isinstance(ctrl, list) else [ctrl]}
        assert_error(wing_analysis(spec, None, FAST, "summary"), "invalid_input", key)
    assert_error(wing_analysis({**RECT, "controls": None}, None, FAST, "summary"),
                 "invalid_input", "'controls' is null")


# 4. Controls in every geometry tool ───────────────────────────────────────────

def test_controls_in_every_geometry_tool():
    """One deflected flap runs in every tool that takes a geometry."""
    flap = {**RECT, "controls": [{"name": "flap", "deflection_deg": 5.0}]}
    assert polar_sweep(flap, 0.0, 4.0, 2.0, None, FAST, "standard")["status"] == "success"
    assert ground_effect(flap, [1.5], alpha_deg=4.0, settings=FAST)["status"] == "success"
    assert stability_derivatives(flap, {"alpha_deg": 4.0}, settings=FAST)["status"] == "success"
    assert batch_evaluate([flap, dict(RECT)], {"alpha_deg": 4.0}, "max_CL", FAST)["status"] == "success"
    m = mesh_convergence(flap, {"alpha_deg": 4.0}, panel_counts=[8, 10],
                         spacing_schemes=["half-cosine"], ref_n_panels=20)
    assert m["status"] == "success", m
    got = polar_sweep(flap, 0.0, 4.0, 2.0, None, FAST, "standard")
    ref = polar_sweep(dict(RECT), 0.0, 4.0, 2.0, None, FAST, "standard")
    cl = {r["alpha_deg"]: r["CL"] for r in got["polar_table"] if r["status"] == "ok"}
    cl0 = {r["alpha_deg"]: r["CL"] for r in ref["polar_table"] if r["status"] == "ok"}
    assert cl[4.0] != pytest.approx(cl0[4.0])


# 5. Undeformed nodes round trip ───────────────────────────────────────────────

def test_undeformed_nodes_round_trip():
    """The tool stations and nodes equal undeformed_nodes in Python."""
    from ventorum.geometry import undeformed_nodes as py_undeformed_nodes

    u = undeformed_nodes(dict(RECT), FAST)
    strict_json(u)
    assert u["status"] == "success", u
    assert u["settings_used"]["n_panels"] == 8
    surf = u["surfaces"]["Wing"]
    assert surf["n_edges"] == 9 == len(surf["eta"])

    ac = build_aircraft_from_spec(dict(RECT))
    ref = py_undeformed_nodes(ac, vt.SolverSettings(n_panels=8), solver="vlm")["Wing"]
    np.testing.assert_allclose(np.array(surf["eta"]), ref["eta"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.array(surf["le_m"]), ref["le"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.array(surf["te_m"]), ref["te"], rtol=1e-12, atol=0)

    # A small heave that grows with eta, with eta given, equals the Python result.
    # undeformed_nodes gives node positions; the displacements are the deltas.
    eta = np.array(surf["eta"])
    dz = (0.02 * eta)[:, None] * np.array([0.0, 0.0, 1.0])
    disp = {"le_m": dz.tolist(), "te_m": dz.tolist(), "eta": eta.tolist()}
    p = wing_analysis({**RECT, "node_displacements": disp}, {"alpha_deg": 4.0}, FAST, "summary")
    strict_json(p)
    assert p["status"] == "success", p
    wing = vt.LiftingSurface(
        name="Wing", semi_span=5.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
        node_displacements=vt.NodeDisplacements(le=dz, te=dz, eta=eta))
    py = vt.analyze(vt.Aircraft(surfaces=[wing]), vt.FlightCondition(alpha=np.radians(4.0)),
                    vt.SolverSettings(n_panels=8))
    m = p["metrics"]
    assert m["CL"] == pytest.approx(round(py.totals.CL, 6), rel=1e-12)
    assert m["CDi"] == pytest.approx(round(py.totals.CDi, 7), rel=1e-12)
    assert m["Cm"] == pytest.approx(round(py.totals.Cm, 6), rel=1e-12)


# 6. Node displacement errors ──────────────────────────────────────────────────

def test_node_displacements_errors():
    """Bad displacements give the documented error types."""
    zeros = lambda n: [[0.0, 0.0, 0.0]] * n  # noqa: E731
    # Wrong count: 5 edges, the mesh with n_panels 8 has 9.
    bad = {**RECT, "node_displacements": {"le_m": zeros(5), "te_m": zeros(5)}}
    assert_error(wing_analysis(bad, None, FAST, "summary"), "invalid_input")
    # le_m and te_m of different lengths.
    bad = {**RECT, "node_displacements": {"le_m": zeros(9), "te_m": zeros(8)}}
    assert_error(wing_analysis(bad, None, FAST, "summary"), "invalid_input", "le_m")
    # An item with 2 numbers.
    le = zeros(9)
    le[0] = [0.0, 0.0]
    bad = {**RECT, "node_displacements": {"le_m": le, "te_m": zeros(9)}}
    assert_error(wing_analysis(bad, None, FAST, "summary"), "invalid_input", "le_m[0]")
    # A station list of another solver: eta of the lifting line on a swept wing.
    from ventorum.geometry import undeformed_nodes as py_undeformed_nodes

    swept = {"span_m": 10.0, "chord_m": 1.0, "sweep_le_deg": 30.0}
    ac = build_aircraft_from_spec(swept)
    eta_llt = py_undeformed_nodes(ac, vt.SolverSettings(n_panels=16), solver="linear")["Wing"]["eta"]
    bad = {**swept, "node_displacements": {"le_m": zeros(17), "te_m": zeros(17),
                                           "eta": eta_llt.tolist()}}
    assert_error(wing_analysis(bad, None, {"n_panels": 16}, "summary"), "invalid_input", "eta")
    # The Fourier solver takes no node displacements.
    assert_error(undeformed_nodes(dict(RECT), {"solver": "fourier", "n_panels": 8}),
                 "invalid_method", "Fourier")


# 7. Trim equals the Python trim ───────────────────────────────────────────────

def test_trim_tool_equals_python_trim():
    """ventorum_trim with CL_target 0.5 equals vt.trim."""
    t = trim(dict(WING_TAIL), 0.5, "elevator", settings={"n_panels": 8})
    strict_json(t)
    assert t["status"] == "success", t
    assert t["trim_status"] == "trimmed" and t["trimmed"] is True
    assert t["device"] == "cpu" and t["precision"] == "float64"

    ref = vt.trim(_python_wing_tail(), vt.FlightCondition(alpha=np.radians(5.0)),
                  vt.SolverSettings(n_panels=8), CL_target=0.5, pitch_control="elevator")
    assert ref.status == "trimmed"
    assert t["alpha_deg"] == pytest.approx(ref.alpha_deg, rel=1e-9)
    assert t["deflections_deg"]["elevator"] == pytest.approx(ref.deflections_deg["elevator"], rel=1e-9)
    assert t["CL"] == pytest.approx(ref.CL, rel=1e-9)
    assert t["Cm"] == pytest.approx(ref.Cm, rel=1e-9)
    assert abs(t["Cm"]) < 1e-6
    assert t["trust"]["rating"] in ("HIGH", "MODERATE", "LOW", "UNRELIABLE")


# 8. A missed target is success with a status ──────────────────────────────────

def test_trim_tool_not_reached_is_success_with_status():
    """CL_target 3.0 is success with trimmed false and the Not trimmed summary."""
    t = trim(dict(WING_TAIL), 3.0, "elevator", settings={"n_panels": 8})
    strict_json(t)
    assert t["status"] == "success", t
    assert t["trimmed"] is False
    assert t["trim_status"] in ("not_converged", "alpha_limit", "control_limit")
    assert len(t["notes"]) > 0
    assert t["executive_summary"].startswith("[Ventorum RESULT] Not trimmed:")
    assert_error(trim(dict(WING_TAIL), 0.5, "no_such_control", settings=FAST),
                 "invalid_input", "no_such_control")
    assert_error(trim(dict(WING_TAIL), 0.5, "elevator", settings={"solver": "fourier", "n_panels": 8}),
                 "invalid_input", "Fourier")


# 9. Error bars equal the Python error bars ────────────────────────────────────

def test_error_bars_tool_equals_python():
    """The six bars equal numerical_error_bars in Python."""
    e = error_bars(dict(RECT), {"alpha_deg": 4.0}, {"n_panels": 16}, "body", "standard")
    strict_json(e)
    assert e["status"] == "success", e
    assert e["levels"] == [16, 11, 8]
    assert e["status_of_bars"] == "numerical_only"
    assert e["device"] in ("cpu", "gpu") and e["precision"] in ("float32", "float64")

    wing = vt.LiftingSurface(
        name="Wing", semi_span=5.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)])
    ref = vt.numerical_error_bars(vt.Aircraft(surfaces=[wing]),
                                  condition=vt.FlightCondition(alpha=np.radians(4.0)),
                                  settings=vt.SolverSettings(n_panels=16), axes="body")
    assert set(e["error_bars"]["bars"]) == {"CL", "CD", "CY", "Cl", "Cm", "Cn"}
    for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
        got = e["error_bars"]["bars"][name]
        want = ref.bars[name].to_dict()
        assert got["value"] == pytest.approx(want["value"], rel=1e-12)
        if want["interval_low"] is None:
            assert got["interval_low"] is None and got["interval_high"] is None
        else:
            assert got["interval_low"] == pytest.approx(want["interval_low"], rel=1e-12)
            assert got["interval_high"] == pytest.approx(want["interval_high"], rel=1e-12)
        assert got["layers"]["numerical"]["state"] == want["layers"]["numerical"]["state"]
    assert_error(error_bars(dict(RECT), None, {"n_panels": 16}, "all"), "invalid_input", "axes")
    assert_error(error_bars(dict(RECT), None, {"n_panels": 12}), "invalid_input", "16")


# 10. Work budget of the new tools ─────────────────────────────────────────────

def test_new_tools_work_budget(monkeypatch):
    """A trim and an error-bar call above the budget are refused before any solve."""
    import ventorum.agent.tools as tools

    def _fail(*args, **kwargs):
        raise AssertionError("no solve must run after a budget refusal")

    monkeypatch.setattr(tools.vt, "analyze", _fail)
    big = {**RECT, "controls": [{"name": "flap", "deflection_deg": 2.0}]}
    t = trim(big, 0.5, "flap", settings={"n_panels": 250, "n_chord": 8})
    assert_error(t, "invalid_input", "budget")
    assert "max_iterations" in t["error"]["message"]

    from ventorum.agent.schemas import MAX_CALL_WORK

    monkeypatch.setattr(tools, "MAX_CALL_WORK", 1000)
    assert MAX_CALL_WORK > 1000
    e = error_bars(dict(RECT), {"alpha_deg": 4.0}, {"n_panels": 16})
    assert_error(e, "invalid_input", "budget")
    assert "fewer panels" in e["error"]["message"]


# 11. Schema export of the new keys and tools ──────────────────────────────────

def test_new_schemas_export():
    """All four formats export the new tools; nested arrays keep items in Gemini."""
    for fmt in ("openai", "anthropic", "gemini", "mcp"):
        schemas = get_tool_schemas(fmt)
        assert len(schemas) == 11
        strict_json(schemas)
    gemini = {t["name"]: t for t in get_tool_schemas("gemini")}
    wing_props = gemini["ventorum_wing_analysis"]["parameters"]["properties"]["wing"]["properties"]
    le_m = wing_props["node_displacements"]["properties"]["le_m"]
    assert "items" in le_m and le_m["items"]["type"] == "array"
    assert "items" in le_m["items"] and le_m["items"]["items"]["type"] == "number"
    mcp = {t["name"]: t for t in get_tool_schemas("mcp")}
    assert mcp["ventorum_trim"]["inputSchema"]["required"] == ["wing", "CL_target", "pitch_control"]
    assert mcp["ventorum_error_bars"]["inputSchema"]["properties"]["axes"]["enum"] == ["body", "stability",
                                                                                       "wind"]
    # Every new key has a description with its unit.
    units = ("[m]", "[deg]", "[-]", "degrees", "metres", "coefficient", "integer", "boolean", "string")
    new_keys: dict[str, dict] = dict(CONTROL_PROPS)
    new_keys.update(NODE_DISPLACEMENT_PROPS)
    new_keys["controls"] = SURFACE_PROPS["controls"]
    new_keys["node_displacements"] = SURFACE_PROPS["node_displacements"]
    trim_props = next(t for t in get_tool_schemas("mcp")
                      if t["name"] == "ventorum_trim")["inputSchema"]["properties"]
    for key in ("CL_target", "pitch_control", "roll_control", "yaw_control",
                "alpha_bounds_deg", "max_iterations"):
        new_keys[key] = trim_props[key]
    for key, prop in new_keys.items():
        assert "description" in prop, key
        assert any(u in prop["description"] for u in units), (key, prop["description"])


# 12. Explicit null in the new keys ────────────────────────────────────────────

def test_explicit_null_in_new_keys():
    """An explicit null for each new key is invalid_input that names the key."""
    base = {"wing": dict(WING_TAIL), "CL_target": 0.5, "pitch_control": "elevator",
            "settings": dict(FAST)}
    for key in ("CL_target", "pitch_control", "alpha_bounds_deg", "max_iterations"):
        assert_error(call_tool("ventorum_trim", {**base, key: None}), "invalid_input", key)
    assert_error(call_tool("ventorum_wing_analysis",
                           {"wing": {**RECT, "controls": None}, "settings": dict(FAST)}),
                 "invalid_input", "controls")
    assert_error(call_tool("ventorum_wing_analysis",
                           {"wing": {**RECT, "node_displacements": None}, "settings": dict(FAST)}),
                 "invalid_input", "node_displacements")


# Review: the new tools follow the method check, the mesh cap and the output fields of the others ─

def test_trim_and_error_bars_refuse_methods_like_the_other_tools():
    """Body-axis wake in ground effect is invalid_input and Fourier in ground effect is invalid_method,
    as in ventorum_wing_analysis."""
    ground = {"alpha_deg": 3.0, "h_m": 2.0}
    body = {"n_panels": 16, "wake_alignment": "body"}
    assert_error(wing_analysis(dict(WING_TAIL), ground, body), "invalid_input")
    assert_error(trim(dict(WING_TAIL), 0.5, "elevator", flight_condition=ground, settings=body),
                 "invalid_input")
    assert_error(error_bars(dict(RECT), ground, body), "invalid_input")
    fourier = {"n_panels": 16, "solver": "fourier"}
    assert_error(wing_analysis(dict(RECT), ground, fourier), "invalid_method")
    assert_error(error_bars(dict(RECT), ground, fourier), "invalid_method")


def test_error_bars_and_trim_carry_the_common_output_fields():
    """The error-bar result states its axes, condition and settings; trim with axes 'all' gives the three sets."""
    e = error_bars(dict(RECT), {"alpha_deg": 4.0}, {"n_panels": 16}, axes="stability")
    strict_json(e)
    assert e["status"] == "success", e
    assert e["axes"] == "stability"
    assert e["condition_used"]["alpha_deg"] == 4.0
    assert e["settings_used"]["n_panels"] == 16
    t = trim(dict(WING_TAIL), 0.5, "elevator", settings={"n_panels": 8}, axes="all")
    strict_json(t)
    assert t["trimmed"] is True
    assert set(t["moments"]) == {"body", "stability", "wind"}
    assert t["moments"]["body"]["Cm"] == pytest.approx(t["Cm"], abs=1e-6)  # the sets are rounded to 6 decimals
