# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests of the BLAS thread policy of the parallel plan (review round 5, C1).

The BLAS thread count is a setting of the whole process. The workers of a
parallel run enter and leave ``blas_single_thread`` at different times;
the count inside must stay at one thread while any worker is inside, and
the count of each BLAS library must come back after the run.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import ventorum as vt
from ventorum.utils import parallel as par

threadpoolctl = pytest.importorskip("threadpoolctl")


def _blas_state() -> dict[str, int]:
    """Return the BLAS thread count of each loaded BLAS library, by file path."""
    return {d["filepath"]: d["num_threads"] for d in threadpoolctl.threadpool_info() if d["user_api"] == "blas"}


def _limited_counts() -> set[int]:
    """Return the thread counts of the BLAS libraries that the context of Ventorum limits."""
    known = {lc.filepath for lc in par._blas_controller().select(user_api="blas").lib_controllers}
    return {n for fp, n in _blas_state().items() if fp in known}


@pytest.fixture
def blas_two_threads():
    """Load the BLAS libraries, set two threads for the test, and restore the counts after it."""
    wing = vt.LiftingSurface(semi_span=2.0, sections=[vt.WingSection(y_frac=0.0, chord=1.0),
                                                      vt.WingSection(y_frac=1.0, chord=1.0)])
    vt.analyze(wing, alpha_deg=2.0, solver="vlm", n_panels=4)
    if not _blas_state() or par._blas_controller() is None:
        pytest.skip("no BLAS library or no threadpoolctl")
    limiter = threadpoolctl.ThreadpoolController().limit(limits=2, user_api="blas")
    try:
        yield _blas_state()
    finally:
        limiter.restore_original_limits()


def test_overlapping_contexts_keep_one_thread(blas_two_threads):
    """Worker B enters while A is inside, and A leaves first: B keeps one BLAS thread."""
    before = blas_two_threads
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    seen: dict[str, set[int]] = {}

    def worker_a():
        with par.blas_single_thread():
            a_in.set()
            b_in.wait(5)
        a_out.set()

    def worker_b():
        a_in.wait(5)
        with par.blas_single_thread():
            b_in.set()
            a_out.wait(5)
            seen["b_after_a_left"] = _limited_counts()

    ta, tb = threading.Thread(target=worker_a), threading.Thread(target=worker_b)
    ta.start()
    tb.start()
    ta.join()
    tb.join()
    assert seen["b_after_a_left"] == {1}
    assert _blas_state() == before


def test_blas_threads_restored_after_parallel_sweep(blas_two_threads):
    before = blas_two_threads
    wing = vt.LiftingSurface(name="w", semi_span=4.0, sections=[vt.WingSection(y_frac=0.0, chord=1.0),
                                                              vt.WingSection(y_frac=1.0, chord=0.6)])
    for _ in range(3):
        vt.analyze_sweep(wing, np.arange(-4.0, 12.0, 0.5), solver="vlm", n_panels=12, n_jobs=8)
    after = _blas_state()
    assert {fp: after[fp] for fp in before} == before


def test_nested_context_restores_once(blas_two_threads):
    before = blas_two_threads
    with par.blas_single_thread():
        with par.blas_single_thread():
            assert _limited_counts() == {1}
        assert _limited_counts() == {1}
    assert _blas_state() == before
