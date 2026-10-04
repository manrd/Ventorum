# Ventorum

Ventorum is a Python package for the aerodynamic analysis of wings and aircraft in low-speed flow. It calculates spanwise loads, lift, drag and moments with lifting-line and vortex-lattice methods, in free air and near flat ground.

:::{warning}
**Work in progress.** Ventorum is under active development and is not ready for production use. The interfaces and the results can change between versions, and the validation against test data is not complete. Do not use the results for design decisions or certification without an independent check.
:::

| Part | For | Content |
| --- | --- | --- |
| [User manual](user/index) | Engineers and students | Installation, first analysis, geometry, solvers, ground effect, results |
| [Theory manual](theory/index) | Engineers and researchers | Equations, assumptions, limits and references of each model |
| [Agent guide](agent/index) | AI agents and their developers | MCP server, tool calls, strict inputs and outputs |
| [Software description](design/index) | Developers | Architecture, data flow, design decisions |
| [Verification and validation](vv/index) | Everyone | Comparison with theory, ground-effect study |
| [API reference](api/index) | Python users | Classes and functions, generated from the docstrings |
| [Contributing](contributing/index) | Developers | Documentation and code standards |

```{toctree}
:hidden:
:maxdepth: 2

user/index
theory/index
agent/index
design/index
vv/index
api/index
contributing/index
```
