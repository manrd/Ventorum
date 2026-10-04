# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Command-line interface of the agent tools.

    python -m ventorum.agent --list
    python -m ventorum.agent --schemas mcp
    python -m ventorum.agent --tool ventorum_wing_analysis --args '{"wing": {...}}'
    python -m ventorum.agent --mcp
"""

from __future__ import annotations

import argparse
import json
import sys

from ventorum.agent.dispatcher import call_tool
from ventorum.agent.mcp_server import run_mcp_server
from ventorum.agent.schemas import AGENT_TOOL_DEFINITIONS, get_tool_schemas


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments. If None, the arguments of the process are
        used.

    Returns
    -------
    int
        Exit status: 0 on success, 1 if the tool returns an error.
    """
    parser = argparse.ArgumentParser(prog="python -m ventorum.agent",
                                     description="Ventorum tools for AI agents.")
    parser.add_argument("--list", action="store_true", help="List the tools and exit.")
    parser.add_argument("--schemas", choices=["openai", "anthropic", "gemini", "mcp"],
                        help="Print the tool schemas in this format and exit.")
    parser.add_argument("--tool", "-t", help="Tool to run, for example ventorum_wing_analysis.")
    parser.add_argument("--args", "-a", default="{}", help="Tool arguments as a JSON object.")
    parser.add_argument("--mcp", action="store_true", help="Run the MCP server on stdin and stdout.")
    ns = parser.parse_args(argv)

    if ns.mcp:
        run_mcp_server()
        return 0
    if ns.list:
        for tool in AGENT_TOOL_DEFINITIONS:
            print(f"{tool['name']}: {tool['description']}")
        return 0
    if ns.schemas:
        print(json.dumps(get_tool_schemas(format=ns.schemas), indent=2))
        return 0
    if ns.tool:
        result = call_tool(ns.tool, ns.args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result.get("status") == "success" else 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
