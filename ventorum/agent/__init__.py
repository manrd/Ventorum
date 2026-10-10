# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Provide Ventorum tools for AI agents.

Tools with strict JSON inputs (units in the key names, unknown keys refused)
and JSON-safe outputs, for LLM function calling and the Model Context
Protocol (MCP). See :mod:`ventorum.agent.schemas` for the input rules.

Every tool returns ``"status": "success"`` or ``"status": "error"`` with
``"error": {"type", "message"}``; the types are ``invalid_input``,
``ground_strike``, ``invalid_method`` and ``internal``.
"""

from __future__ import annotations

from ventorum.agent.dispatcher import call_tool
from ventorum.agent.mcp_server import handle_message, handle_request, run_mcp_server
from ventorum.agent.schemas import (
    AGENT_TOOL_DEFINITIONS,
    InputError,
    build_aircraft_from_spec,
    build_flight_condition_from_spec,
    get_tool_schemas,
)
from ventorum.agent.tools import (
    batch_evaluate,
    error_bars,
    ground_effect,
    mesh_convergence,
    polar_sweep,
    stability_derivatives,
    trim,
    undeformed_nodes,
    wing_analysis,
)

__all__ = [
    "AGENT_TOOL_DEFINITIONS",
    "InputError",
    "batch_evaluate",
    "build_aircraft_from_spec",
    "build_flight_condition_from_spec",
    "call_tool",
    "error_bars",
    "get_tool_schemas",
    "ground_effect",
    "handle_message",
    "handle_request",
    "mesh_convergence",
    "polar_sweep",
    "run_mcp_server",
    "stability_derivatives",
    "trim",
    "undeformed_nodes",
    "wing_analysis",
]
