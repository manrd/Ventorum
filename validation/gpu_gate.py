# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Run the GPU checks of the phase gate and write a short Markdown log.

Usage
-----
python validation/gpu_gate.py <log.md> [--float64] [--example-timeout SECONDS]
[--allow-no-gpu] [--dry-run]

The steps run in this order: an environment check
(``ventorum.gpu.available()``), all GPU tests (``python -m pytest -q -m
gpu``), and every ``examples/[0-9]*.py`` with ``VENTORUM_DEVICE=gpu``.
With ``--float64`` the examples run a second time with
``VENTORUM_GPU_PRECISION=float64``. Each step runs in a subprocess with
``sys.executable``, so a crash in one step does not stop the script.

Exit codes: 0 when every step passes, 1 when a step fails or times out,
2 when there is no GPU and ``--allow-no-gpu`` is not given. With
``--dry-run`` the script prints the commands in order and exits 0
without running them.

The log holds the date and time (local, ISO 8601), the Ventorum version
and the git commit (short hash), the Python, NumPy, Warp and PyTorch
versions (or "not installed"), the GPU model name and the CUDA version
when they are known, one table row per step (name, result, time [s],
last output line), and a final "GPU gate: PASS" or "GPU gate: FAIL (n
failures)" line. It holds no host name, no user name and no home path;
paths are relative to the repository root.

Testing only: when the environment variable
``VENTORUM_GPU_GATE_FORCE_NO_GPU`` is ``1``, the script takes the no-GPU
path even on a machine with a GPU. Tests use this or patch
``ventorum.gpu.available`` and call ``main()`` in-process with a fake
runner; see ``tests/test_gpu_gate.py``.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import importlib.util
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ventorum import gpu

#: Root of the source tree that holds this script.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Timeout [s] of each example step, unless --example-timeout says otherwise.
EXAMPLE_TIMEOUT_S = 600.0

#: Testing only: "1" forces the no-GPU path (see the module docstring).
FORCE_NO_GPU_ENV = "VENTORUM_GPU_GATE_FORCE_NO_GPU"

#: Sentence written to the log when the gate runs without a GPU.
NO_GPU_NOTICE = "no GPU: the GPU tests skip; this log is NOT GPU evidence"

#: Longest last-output line kept in the log, in characters.
LAST_LINE_MAX = 300


@dataclass
class Step:
    """One check of the gate: its command and its environment."""

    name: str
    argv: list[str]
    env: dict[str, str]
    timeout: float | None


@dataclass
class StepResult:
    """Outcome of one step: pass, fail or timeout, with its run time."""

    name: str
    status: str
    seconds: float
    last_line: str


def list_examples() -> list[str]:
    """Return the example scripts ``examples/[0-9]*.py``, sorted, relative to the root."""
    names = sorted(path.name for path in (REPO_ROOT / "examples").glob("[0-9]*.py") if path.is_file())
    return [f"examples/{name}" for name in names]


def build_plan(include_float64: bool, example_timeout: float = EXAMPLE_TIMEOUT_S) -> list[Step]:
    """Return the steps of the gate in run order: pytest, examples, float64 examples."""
    plan = [
        Step(
            name="pytest -m gpu",
            argv=[sys.executable, "-m", "pytest", "-q", "-m", "gpu"],
            env={"MPLBACKEND": "Agg"},
            timeout=None,
        )
    ]
    for relative in list_examples():
        plan.append(
            Step(
                name=relative,
                argv=[sys.executable, relative],
                env={"VENTORUM_DEVICE": "gpu", "MPLBACKEND": "Agg"},
                timeout=example_timeout,
            )
        )
    if include_float64:
        for relative in list_examples():
            plan.append(
                Step(
                    name=f"{relative} [float64]",
                    argv=[sys.executable, relative],
                    env={
                        "VENTORUM_DEVICE": "gpu",
                        "VENTORUM_GPU_PRECISION": "float64",
                        "MPLBACKEND": "Agg",
                    },
                    timeout=example_timeout,
                )
            )
    return plan


def describe(step: Step) -> str:
    """Return the shell-like command of *step* for --dry-run."""
    env_text = " ".join(f"{key}={value}" for key, value in sorted(step.env.items()))
    command = " ".join(step.argv)
    if step.timeout is not None:
        command += f" (timeout {step.timeout:g} s)"
    return f"{env_text} {command}" if env_text else command


def _last_line(text: str) -> str:
    """Return the last non-empty line of *text*, cleaned for the log table."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    last = lines[-1] if lines else "-"
    last = last.replace(str(REPO_ROOT), ".")
    try:
        last = last.replace(str(Path.home()), "~")
    except (OSError, RuntimeError):
        pass
    last = last.replace("|", "/")
    return last[:LAST_LINE_MAX]


def run_step(step: Step) -> StepResult:
    """Run *step* in a subprocess and return its result."""
    env = dict(os.environ)
    env.update(step.env)
    # Import Ventorum from this tree, so the log's git commit is the code that ran.
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH")) if p)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            step.argv, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=step.timeout
        )
    except subprocess.TimeoutExpired as exc:
        # The captured output of a timeout is bytes, also with text=True.
        out = "\n".join(
            part.decode("utf-8", "replace") if isinstance(part, bytes) else (part or "")
            for part in (exc.stdout, exc.stderr)
        )
        return StepResult(step.name, "timeout", time.perf_counter() - start, _last_line(out))
    except OSError as exc:
        return StepResult(step.name, "fail", time.perf_counter() - start, f"cannot start: {exc}")
    seconds = time.perf_counter() - start
    status = "pass" if proc.returncode == 0 else "fail"
    output = proc.stdout if proc.stdout and proc.stdout.strip() else proc.stderr
    return StepResult(step.name, status, seconds, _last_line(output or "-"))


def _package_version(distribution: str, module: str) -> str:
    """Return the version of *distribution*, or "not installed"."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        if importlib.util.find_spec(module) is not None:
            return "installed (unknown version)"
        return "not installed"


def _gpu_lines() -> list[str]:
    """Return the GPU model and CUDA lines of the log header."""
    if not gpu.available():
        return ["- GPU: not available"]
    name = gpu.info().get("name") or "unknown"
    cuda = "unknown"
    try:
        import torch

        cuda = torch.version.cuda or "unknown"
    except Exception:  # noqa: BLE001 - torch may be missing or broken
        cuda = "not installed" if importlib.util.find_spec("torch") is None else "unknown"
    return [f"- GPU: {name}", f"- CUDA (PyTorch build): {cuda}"]


def _git_commit() -> str:
    """Return the short hash of the checked-out commit, or "unknown".

    A tree with uncommitted changes to tracked files is marked, so the log
    does not claim a commit that is not the code that ran.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if proc.returncode != 0 or not proc.stdout.strip():
        return "unknown"
    commit = proc.stdout.strip()
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return commit
    return f"{commit} (with uncommitted changes)" if dirty else commit


def _numpy_version() -> str:
    """Return the NumPy version, or "not installed"."""
    try:
        import numpy

        return numpy.__version__
    except Exception:  # noqa: BLE001 - numpy may be missing or broken
        return "not installed"


def header_lines() -> list[str]:
    """Return the header lines of the log (no machine identifiers)."""
    now = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [
        f"# GPU gate log - {now}",
        "",
        f"- Ventorum version: {importlib.metadata.version('ventorum')}",
        f"- Git commit: {_git_commit()}",
        f"- Python: {platform.python_version()}",
        f"- NumPy: {_numpy_version()}",
        f"- Warp: {_package_version('warp-lang', 'warp')}",
        f"- PyTorch: {_package_version('torch', 'torch')}",
        *_gpu_lines(),
        "",
    ]
    return lines


def render_log(results: list[StepResult], notices: list[str]) -> str:
    """Return the full log text for *results* with the extra *notices*."""
    lines = header_lines()
    lines += [f"- {notice}" for notice in notices]
    if notices:
        lines.append("")
    lines += [
        "| step | result | time [s] | last output line |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(f"| {result.name} | {result.status} | {result.seconds:.1f} | {result.last_line} |")
    lines.append("")
    failures = sum(1 for result in results if result.status != "pass")
    if failures == 0:
        lines.append("GPU gate: PASS")
    else:
        lines.append(f"GPU gate: FAIL ({failures} failures)")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="gpu_gate.py",
        description="Run the GPU checks of the phase gate and write a Markdown log.",
    )
    parser.add_argument("log", nargs="?", help="Markdown log to write (not needed with --dry-run)")
    parser.add_argument("--float64", action="store_true", help="run the examples again in float64")
    parser.add_argument(
        "--example-timeout", type=float, default=EXAMPLE_TIMEOUT_S, help="timeout [s] of each example"
    )
    parser.add_argument(
        "--allow-no-gpu", action="store_true", help="run the steps even without a GPU (not GPU evidence)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands in order and exit 0 without running them"
    )
    args = parser.parse_args(argv)
    if args.log is None and not args.dry_run:
        parser.error("the log path <log.md> is required without --dry-run.")
    if args.example_timeout is not None and args.example_timeout <= 0:
        parser.error("--example-timeout must be positive.")
    return args


def main(argv: list[str] | None = None, runner=None) -> int:
    """Run the gate and return its exit code (0 pass, 1 failure, 2 no GPU)."""
    args = parse_args(argv)
    plan = build_plan(args.float64, args.example_timeout)
    if args.dry_run:
        for step in plan:
            print(describe(step))
        return 0
    run = runner if runner is not None else run_step
    forced = os.environ.get(FORCE_NO_GPU_ENV, "").strip() == "1"
    if forced:
        gpu_ok, reason = False, f"{FORCE_NO_GPU_ENV}=1 forces the no-GPU path (testing only)."
    else:
        gpu_ok, reason = gpu.available(), gpu.unavailable_reason()
    notices: list[str] = []
    if not gpu_ok:
        notices.append(f"GPU check: not available: {reason or 'unknown reason'}")
        if not args.allow_no_gpu:
            lines = header_lines() + [f"- {notices[0]}", "", "GPU gate: NOT RUN (no GPU)", ""]
            Path(args.log).write_text("\n".join(lines), encoding="utf-8")
            return 2
        notices.append(NO_GPU_NOTICE)
    results = [run(step) for step in plan]
    Path(args.log).write_text(render_log(results, notices), encoding="utf-8")
    return 0 if all(result.status == "pass" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
