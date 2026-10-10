# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Sphinx configuration of the Ventorum documentation.

Build the HTML site from the repository root with::

    sphinx-build -W --keep-going -b html docs docs/_build/html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT))

import ventorum  # noqa: E402

project = "Ventorum"
author = "Manuel Alejandro Rodriguez Diaz, PhD"
copyright = "2026, Manuel Alejandro Rodriguez Diaz"
version = release = ventorum.__version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# The package re-exports its main classes at the top level, so a type name can
# point to two equal targets. That ambiguity is harmless.
suppress_warnings = ["ref.python"]

# MyST: Markdown with math and admonitions.
myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# API reference from the NumPy-style docstrings.
autosummary_generate = True
# NVIDIA Warp is the optional extra "gpu". It is not installed on the CI
# documentation runners, so the GPU modules import a mock of it.
autodoc_mock_imports = ["warp"]
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

html_theme = "pydata_sphinx_theme"
html_title = "Ventorum"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_with_keys": False,
    "show_toc_level": 2,
}


def _write_tool_reference(app) -> None:
    """Write the reference of the agent tools from the live MCP schemas.

    The page is generated at each build, so it cannot differ from the
    schemas that the MCP server sends.
    """
    from ventorum.agent import get_tool_schemas

    lines = [
        "# Tool reference",
        "",
        "This page is generated from `ventorum.agent.get_tool_schemas(format=\"mcp\")` at each build.",
        "",
    ]
    for tool in get_tool_schemas(format="mcp"):
        lines += [f"## `{tool['name']}`", "", tool["description"], "", "Input schema:", "", "```json",
                  json.dumps(tool["inputSchema"], indent=2), "```", ""]
    out = DOCS / "agent" / "tool_reference.md"
    text = "\n".join(lines)
    if not out.exists() or out.read_text(encoding="utf-8") != text:
        out.write_text(text, encoding="utf-8")


def setup(app):
    app.connect("builder-inited", _write_tool_reference)
