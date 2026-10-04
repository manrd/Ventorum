# Rules for AI coding agents

This file gives the working rules of the Ventorum repository to every AI coding agent (Claude Code, OpenCode and others). Read it before you change anything. A task from the owner can narrow these rules; it never removes the rules marked **always** or **never**.

## The project

Ventorum is a set of low-order aerodynamic solvers in Python: Lanchester–Prandtl lifting-line models (Ventorum-LLT), vortex-lattice models (Ventorum-VLM) and, later, 3D panel models (Ventorum-3DP). It is a work in progress. The package is `ventorum`. The current scope is incompressible flow (Mach below 0.3), in free air and in ground effect.

Three properties define it: high computational efficiency with no manual tuning (all CPU cores, GPUs, autotuning), state-of-the-art models that are verified and credited, and honest results (each result states how far it can be trusted).

## Your role

- Do the scoped task you are given: implementation, debugging, tests, documentation.
- **Never** make a design decision on your own. If the task needs one (a new dependency, a change of method, a change of a public interface, a change of a result), stop and ask the owner.
- **Never** delete or disable a feature, a solver, a kernel, a test or a file because it looks unused, unverified or slow. Report it to the owner instead. Unverified code is code to verify, not code to delete.
- **Never** skip, delete or weaken GPU work because your environment has no GPU. GPU use is a core principle: development can happen on the CPU, but GPU tests must always follow shortly. Leave GPU code and GPU tests ready to run on a machine with a GPU.
- **macOS**: the owner cannot test on a Mac, so macOS problems wait until the general development is done. Do not spend the task on them. When you see a possible macOS problem, record it: add an entry to `docs/user/performance_limits.md` (if it costs speed) or a GitHub issue with the label `macos`, and mention it in your report. Do not try to fix it unless the task asks for it.
- **Never** remove a kernel backend (Numba, Cython, numpy, PyTorch) because it is slower in one case or on one machine. Each backend that wins somewhere stays; the tuner measures which one wins.

## Gates before every commit (always)

Run these commands. All must pass, and the verification report must not change unless the task is to change a result:

```bash
pip install -e .[dev,docs]
ruff check .
python -m pytest -q
python validation/run_verification.py /tmp/verification_check.md   # compare with docs/verification_report.md (ignore the run-time line)
sphinx-build -W --keep-going -b html docs docs/_build/html
```

- A change that moves a verification number needs a reason in the commit message and the owner's approval.
- Every bug you fix gets a regression test.
- A new kernel or backend must agree with the numpy reference kernel to round-off; add a parity test (see `tests/test_kernels.py`).
- Run all scripts in `examples/` when you change a public interface.

## Code standards (always)

- **Language** of comments, docstrings, documents and commit messages: simple technical English that follows ASD-STE100 (short sentences, active voice, one instruction per sentence).
- **Never** use the em dash character (Unicode U+2014) anywhere: code, comments, documents or messages.
- **Docstrings**: NumPy docstring standard, checked by ruff. Give the unit of every physical quantity: m, m/s, rad, deg, kg/m^3, N.
- **Style**: PEP 8, checked by `ruff check .`. Match the style of the code around your change.
- **Credit**: each model names its sources and their authors, in its docstring and in the theory manual. Use the name "Lanchester–Prandtl lifting-line theory".
- **Numbers in documents**: produce them with a script from the code. Never type a result by hand.
- **Units and axes**: SI units. Geometry axes x aft, y right, z up. Moments as in AVL: Cl > 0 right wing down, Cm > 0 nose up, Cn > 0 nose right.

## Copyright and licences (never)

- **Never** take code from another tool, whatever its licence: no copy, no translation, no close paraphrase. This applies to all aerodynamic tools (for example AVL, XFLR5, Flow5, OpenVSP/VSPAERO, MachUpX, AeroSandbox) and to any other software.
- **Never** read the source code of another tool to find out how it does something. Work from the published method (papers, books, reports), and cite that source in the docstring and the theory manual.
- Some code is the same in every correct implementation (a textbook formula, a standard algorithm). That is acceptable when you wrote it from the published method yourself.
- You may run another tool as a black box (inputs and outputs only) when the task asks for a comparison.
- **Never** add a dependency without the owner's approval. Approved dependencies must be free to use; keep their versions bounded in `pyproject.toml`.

## Data and security (never)

- This repository is public. Treat everything you write as public.
- **Never** add credentials, API keys, tokens, machine identifiers or personal data to the repository, to logs or to test data.
- **Never** add unpublished results, paper drafts, private planning documents or reference data whose licence is not clear. The owner adds reference data.
- **Never** add task cards (`T-NNNN-*.md`, `TASK_CARD.md`), the backlog or the decision log, study scripts or study reports. They stay in the owner's private folder. The folders `tasks/` and `studies/` must not exist in this repository; `.gitignore` excludes them. Before every commit, read `git diff --cached --stat` and remove any such file.

## Repository map

| Path | Content |
| --- | --- |
| `ventorum/core` | Data types, constants, errors, trust score |
| `ventorum/geometry` | Lattice, spanwise spacing, mesh convergence |
| `ventorum/aero` | Vortex kernels (`vortex.py` numpy reference, `vortex_numba.py` compiled), assembly, loads, polars |
| `ventorum/solvers` | Verified solvers: VLM, linear and nonlinear lifting line, Fourier |
| `ventorum/hardware` | Hardware detection, machine profile, tuner (`ventorum-tune`) |
| `ventorum/utils/parallel.py` | Two-level parallel plan (cases in parallel, kernel threads per case) |
| `ventorum/ground_effect` | Ground-effect analysis and sweeps |
| `ventorum/agent` | Tool interface for AI agents and the MCP server |
| `ventorum/legacy` | The original solvers and acceleration paths (Numba, Cython, PyTorch GPU, autotuner), kept to study and port their optimisations. Some legacy solvers have known physics defects; see `tests/test_legacy.py`. Do not delete anything here. |
| `tests/` | Tests; `test_analytical_verification.py` holds the theory checks |
| `validation/` | Verification script and reference data rules |
| `docs/` | Sphinx documentation (user manual, theory, agent guide, software description, V&V, API) |

## Delegated tasks (task cards)

Task cards and the backlog are private: they are not in this repository. When the owner gives you a task card (in a worktree, the file `TASK_CARD.md`, which git ignores):

1. Read this file and the task card completely before you start.
2. Work only on the branch that the card names, and change only the files that the card allows.
3. The decisions in the card are fixed. Do not change them. If the card is not clear, or the task needs a decision that the card does not give, stop and write the question in the report. Do not guess.
4. Write the tests that the card asks for, then the code. Run the gates (above) and the acceptance checks of the card.
5. Fill in the "Completion report" section of the card, honestly: what you did, the results of each check, and what you could not do. Do not commit the card: commit only your work.
6. If a check fails and you cannot fix it inside the scope, report it. Do not weaken a test, a tolerance or a check to make it pass.

The full procedure is in `docs/contributing/agent_tasks.md`.

## Commits

- One logical change per commit, with a message that says what changed and why, in ASD-STE100 English.
- Do not push to `main` and do not open pull requests unless the owner asks.
