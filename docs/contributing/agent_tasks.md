# Delegating work to other AI agents

Ventorum can use other AI coding agents (for example free models in OpenCode) for scoped implementation and debugging work. The owner and Claude keep all design decisions. Claude writes the task card and reviews the result. This page gives the procedure. The rules for all agents are in `AGENTS.md` at the root of the repository.

## When to delegate

| Delegate | Do not delegate |
| --- | --- |
| Work with an exact specification and objective checks: a kernel with a parity test, a port of a known algorithm, new tests, docstrings, plots, file readers and writers, command-line tools | Design decisions, new physics or new methods, numerical schemes where the error is subtle, changes to verification results |
| Work that is large compared with its review: much code, simple to check | Small changes: the task card costs more than the change |
| Work in one area with clear file limits | Changes across many modules or public interfaces |

A task is ready for delegation when its acceptance checks can find every likely error without a deep review.

## The procedure

1. **Owner: select the work.** Tell Claude: "Prepare a task card for: <the work>".
2. **Claude: prepare the card.** Claude asks the owner for any open design decision, then writes the card `T-NNNN-<name>.md` from the card template. The cards, the template and the backlog are in the owner's private task folder, not in this repository. Claude creates the branch `agent/T-NNNN-<name>` and a worktree for it, and copies the card into the worktree as `TASK_CARD.md`. Git ignores that file, so the card is never committed. The card contains everything the agent needs: fixed decisions, allowed files, exact interfaces, the published method with its source, tests with names and tolerances, and the commands to run.
3. **Owner: hand over.** Give the agent the worktree on that branch, no other files and no credentials, and this prompt:

   > Read AGENTS.md and TASK_CARD.md completely. Do exactly the task in the card, on the branch it names. Follow every rule in AGENTS.md. When you finish, fill in the completion report in the card and commit your work. Do not commit the card.

4. **Agent: do the task.** It writes the tests and the code, runs the checks, fills in the completion report and commits its work on the branch.
5. **Owner: request the review.** Tell Claude: "Review T-NNNN" (the branch is in the card).
6. **Claude: review** (below). Claude copies the filled-in card back to the private task folder. Then Claude gives a verdict:
   - **Accepted**: Claude merges the branch into the development branch.
   - **Small fixes**: Claude fixes small problems itself when that is cheaper than a rework round, then merges.
   - **Rework**: Claude writes the problems in the review section of the card. The owner gives the agent the same prompt again. After two rework rounds, Claude does the task itself.

## Review checklist (Claude)

The review must find the same problems that Claude would avoid when it writes the code. It is cheaper than writing the code because it reads the difference, not the whole code base, and because the checks run automatically.

1. **Scope**: `git diff --stat` against the base commit. Every changed file is in the allowed list. No test, tolerance or check was weakened or deleted.
2. **Gates**: run all gates of AGENTS.md again; do not trust the report. The verification report must be unchanged.
3. **Acceptance checks**: run each check of the card, and compare with the report.
4. **Specification**: read the difference against the card: interfaces, units, edge cases, errors, method and its source.
5. **Numerics**: the parts of the card that need judgement (for example near-singular cases, tolerances, thread safety). Add a quick independent check where the card names a risk.
6. **Provenance**: no code from other tools; no unusual copyright or licence text; the sources are papers or books.
7. **Standards**: ASD-STE100 language, no em dash, NumPy docstrings with units, PEP 8.
8. **Report**: the completion report is complete and honest.

## Rules for the task card (Claude)

- Write every decision into the card. The agent must never need to guess.
- Name the allowed files. Everything else is out of scope.
- Give the interfaces exactly (names, arguments, return values, units).
- Give the method from its published source, with the reference. Never point to another tool's code.
- Give the tests with their names and tolerances. A test that the agent designs alone checks less.
- Give the commands of the acceptance checks and the expected results.
- Name the risks that the review will check, so that the agent checks them too.
- Do not put private planning content, unpublished results or reference data in a card: the agent's provider may keep it.
