# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Interactive terminal progress bar and status display for Ventorum benchmarks.
"""

from __future__ import annotations

import sys
import threading
import time

_STDOUT_LOCK = threading.RLock()


class ProgressBar:
    """A clean, real-time terminal progress bar with ETA, item counters,
    and status text. Thread-safe across multiple concurrent callers.
    """

    def __init__(
        self,
        total: int,
        title: str = "Progress",
        unit: str = "runs",
        bar_width: int = 28,
        stream=sys.stdout,
    ):
        self.total = max(1, total)
        self.title = title
        self.unit = unit
        self.bar_width = bar_width
        self.stream = stream
        self.current = 0
        self.start_time = time.perf_counter()
        self.last_update_time = 0.0
        self._is_finished = False
        self._lock = threading.Lock()

    def update(self, step: int = 1, status: str = ""):
        """Advance progress by `step` items and refresh the terminal display."""
        with self._lock:
            self.current = min(self.total, self.current + step)
            self._render(status)

    def set_current(self, current: int, status: str = ""):
        """Set absolute progress count and refresh."""
        with self._lock:
            self.current = min(self.total, max(0, current))
            self._render(status)

    def _render(self, status: str = ""):
        now = time.perf_counter()
        elapsed = now - self.start_time
        fraction = self.current / self.total
        percent = fraction * 100.0

        # Bar characters (ASCII-safe for cp1252 / standard Windows CMD / PowerShell)
        filled_len = int(round(self.bar_width * fraction))
        bar = "=" * max(0, filled_len - 1) + (">" if filled_len > 0 and filled_len < self.bar_width else "=" if filled_len == self.bar_width else "")
        bar = bar.ljust(self.bar_width, " ")

        # Rate & ETA
        if self.current > 0:
            rate = elapsed / self.current
            remaining_items = self.total - self.current
            eta_seconds = remaining_items * rate
            eta_str = f"{eta_seconds:.1f}s" if eta_seconds < 60 else f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s"
        else:
            eta_str = "--"

        elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed//60)}m {int(elapsed%60)}s"

        # Truncate status if needed
        status_disp = f" | {status}" if status else ""
        if len(status_disp) > 35:
            status_disp = status_disp[:32] + "..."

        line = (
            f"\r{self.title:<22} [{bar}] {self.current}/{self.total} {self.unit} "
            f"({percent:>5.1f}%) | {elapsed_str} (ETA: {eta_str}){status_disp}"
        )

        with _STDOUT_LOCK:
            self.stream.write(f"\r\033[K{line}")
            self.stream.flush()

    def finish(self, final_msg: str = ""):
        """Finish the progress bar and print final elapsed time."""
        with self._lock:
            if not self._is_finished:
                self._is_finished = True
                total_elapsed = time.perf_counter() - self.start_time
                msg = f" - {final_msg}" if final_msg else ""
                elapsed_str = f"{total_elapsed:.2f}s"
                with _STDOUT_LOCK:
                    self.stream.write(
                        f"\r\033[K{self.title:<22} [COMPLETED] {self.total}/{self.total} {self.unit} "
                        f"in {elapsed_str}{msg}\n"
                    )
                    self.stream.flush()


class MultiInstanceProgressTracker:
    """Thread-safe coordinated progress display for multiple concurrent Ventorum instances.
    
    Displays completed instances, active instances, and total throughput without
    terminal line collisions.
    """

    def __init__(self, total_instances: int, title: str = "Parallel Instances"):
        self.total = max(1, total_instances)
        self.title = title
        self.completed = 0
        self.active_cases: set[str] = set()
        self.start_time = time.perf_counter()
        self._lock = threading.Lock()
        self._is_finished = False

    def on_instance_start(self, case_name: str):
        with self._lock:
            self.active_cases.add(case_name)
            self._render()

    def on_instance_finish(self, case_name: str):
        with self._lock:
            self.active_cases.discard(case_name)
            self.completed += 1
            self._render()

    def _render(self):
        now = time.perf_counter()
        elapsed = now - self.start_time
        fraction = self.completed / self.total
        percent = fraction * 100.0
        bar_width = 24
        filled_len = int(round(bar_width * fraction))
        bar = "=" * max(0, filled_len - 1) + (">" if filled_len > 0 and filled_len < bar_width else "=" if filled_len == bar_width else "")
        bar = bar.ljust(bar_width, " ")

        active_str = f"Active: {len(self.active_cases)}"
        line = (
            f"\r{self.title:<20} [{bar}] {self.completed}/{self.total} cases "
            f"({percent:>5.1f}%) | {elapsed:.1f}s | {active_str}"
        )
        with _STDOUT_LOCK:
            sys.stdout.write(f"\r\033[K{line}")
            sys.stdout.flush()

    def finish(self, msg: str = ""):
        with self._lock:
            if not self._is_finished:
                self._is_finished = True
                elapsed = time.perf_counter() - self.start_time
                with _STDOUT_LOCK:
                    sys.stdout.write(
                        f"\r\033[K{self.title:<20} [COMPLETED] {self.completed}/{self.total} cases "
                        f"in {elapsed:.2f}s {msg}\n"
                    )
                    sys.stdout.flush()


class BenchmarkTracker:
    """Manages multi-stage benchmark suites with real-time overall and stage progress."""

    def __init__(self, stages: list[tuple[str, int, str]]):
        """
        stages: list of (stage_name, total_runs, unit)
        """
        self.stages = stages
        self.stage_index = 0
        self.total_stages = len(stages)
        self.total_runs = sum(count for _, count, _ in stages)
        self.completed_runs = 0
        self.start_time = time.perf_counter()

    def start_stage(self, stage_idx: int) -> ProgressBar:
        self.stage_index = stage_idx
        name, count, unit = self.stages[stage_idx]
        header = f"\n>>> [{stage_idx + 1}/{self.total_stages}] {name} ({count} {unit})"
        print(header)
        return ProgressBar(total=count, title=f"Stage {stage_idx + 1}/{self.total_stages}", unit=unit)
