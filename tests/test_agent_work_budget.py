# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Work budget of the agent tools.

Each tool estimates its total work before it solves anything. The work
of one solve of a lattice with N panels is N^2 units. A call above
MAX_CALL_WORK is refused with type invalid_input before any solve.
"""

from __future__ import annotations

from ventorum.agent import (
    batch_evaluate,
    call_tool,
    ground_effect,
    mesh_convergence,
    polar_sweep,
    stability_derivatives,
    wing_analysis,
)
from ventorum.agent.schemas import MAX_CALL_WORK, MAX_TOTAL_PANELS

FAST = {"n_panels": 8}
RECT = {"span_m": 10.0, "chord_m": 1.0}


def test_mesh_convergence_surface_override_is_limited():
    """A surface n_panels that exceeds the mesh limit at one level is refused."""
    assert MAX_TOTAL_PANELS == 4000
    wing = {**RECT, "n_panels": 300}
    p = call_tool(
        "ventorum_mesh_convergence",
        {"wing": wing, "panel_counts": [8, 12], "ref_n_panels": 160},
    )
    assert p["status"] == "error", p
    assert p["error"]["type"] == "invalid_input", p
    assert "4000" in p["error"]["message"], p["error"]["message"]


def test_large_sweep_is_refused_before_solving(monkeypatch):
    """A polar sweep above the budget is refused and runs no solve."""
    import ventorum.agent.tools as tools

    def _fail(*args, **kwargs):
        raise AssertionError("no solve must run after a budget refusal")

    monkeypatch.setattr(tools.vt, "analyze", _fail)
    wing = dict(RECT)
    settings = {"n_panels": 250, "n_chord": 8}
    p = polar_sweep(wing, 0.0, 12.0, 1.0, None, settings, "summary")
    assert p["status"] == "error", p
    assert p["error"]["type"] == "invalid_input", p
    assert "budget" in p["error"]["message"].lower(), p["error"]["message"]


def test_ground_effect_heights_count_in_the_budget(monkeypatch):
    """More heights mean more work: one height passes, many heights are refused."""
    import ventorum.agent.tools as tools

    settings = {"n_panels": 125, "n_chord": 8}
    ok = ground_effect(dict(RECT), [2.0], alpha_deg=4.0, settings=settings)
    assert ok["status"] == "success", ok

    def _fail(*args, **kwargs):
        raise AssertionError("no solve must run after a budget refusal")

    monkeypatch.setattr(tools.vt, "analyze", _fail)
    heights = [2.0 + 0.5 * k for k in range(20)]
    p = ground_effect(dict(RECT), heights, alpha_deg=4.0, settings=settings)
    assert p["status"] == "error", p
    assert p["error"]["type"] == "invalid_input", p
    assert "height" in p["error"]["message"].lower(), p["error"]["message"]


def test_message_names_the_budget_and_the_estimate():
    """The refusal message gives the estimate, the budget and how to reduce the work."""
    settings = {"n_panels": 250, "n_chord": 8}
    p = polar_sweep(dict(RECT), 0.0, 12.0, 1.0, None, settings, "summary")
    assert p["status"] == "error", p
    msg = p["error"]["message"]
    assert str(int(MAX_CALL_WORK)) in msg, msg
    assert "estimate" in msg.lower(), msg
    assert "budget" in msg.lower(), msg
    assert "fewer" in msg.lower(), msg


def test_normal_calls_pass():
    """Each tool with the inputs of its existing tests passes the budget."""
    assert wing_analysis(dict(RECT), None, dict(FAST), "summary")["status"] == "success"
    assert polar_sweep(dict(RECT), 0.0, 8.0, 2.0, None, dict(FAST), "summary")["status"] == "success"
    assert ground_effect(dict(RECT), [0.6, 1.0, 1.5], alpha_deg=4.0,
                         settings=dict(FAST))["status"] == "success"
    assert stability_derivatives(dict(RECT), None, settings=dict(FAST))["status"] == "success"
    assert batch_evaluate([dict(RECT), {"span_m": 6.0, "chord_m": 1.0}], None, "max_CL",
                          dict(FAST))["status"] == "success"
    p = mesh_convergence(
        {"span_m": 10.0, "root_chord_m": 1.5, "tip_chord_m": 1.0},
        {"V_inf_m_s": 45.0, "alpha_deg": 4.0},
        panel_counts=[12, 20],
        spacing_schemes=["half-cosine"],
        ref_n_panels=30,
    )
    assert p["status"] == "success", p
