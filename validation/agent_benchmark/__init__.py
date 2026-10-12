# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Agent benchmark of the Ventorum MCP interface.

The harness measures how well an LLM agent uses the Ventorum tools through
the Model Context Protocol (MCP). It has these parts:

* :mod:`.mcp_client`: a stdio MCP client that starts the Ventorum server.
* :mod:`.adapters`: model adapters (OpenAI chat completions, Anthropic
  Messages and a scripted fake model for tests).
* :mod:`.tasks`: the task set, its variants and the prompts.
* :mod:`.references`: the golden calls that give the reference values at
  run time, and the self-test against the library API.
* :mod:`.scoring`: the strict score, the provenance check and the error kinds.
* :mod:`.runner`: the command line (``run``, ``report``, ``references``,
  ``selftest``).
* :mod:`.client_track`: scripts for existing MCP clients (track B).

The harness uses only the Python standard library and Ventorum. The task
files hold no reference values: the harness computes them at run time.
The run outputs go to a folder outside the repository.
"""

from __future__ import annotations

HARNESS_VERSION = "1.0.0"
LOG_SCHEMA_VERSION = 1
