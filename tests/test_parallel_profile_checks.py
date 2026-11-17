# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the machine-profile value checks of the parallel plan.

A tuned worker or thread count that is not a finite number (NaN,
infinity, a boolean, a text that is not a number) must give the default
plan with one RuntimeWarning, not an exception or a nonsense plan.
"""

from __future__ import annotations

import math
import warnings

import pytest

from ventorum.hardware import profile as prof
from ventorum.utils import parallel as par


@pytest.fixture
def tuned_batch(monkeypatch):
    """Replace the tuned batch setting of the profile; clear the warning record."""
    monkeypatch.setattr(par, "_warned", set())
    box: dict = {}

    def fake(*keys, default=None):
        if keys and keys[0] == "batch":
            return box.get("value", default)
        return default

    monkeypatch.setattr(prof, "tuned_setting", fake)
    return box


def _default_plan(n_tasks: int, n_panels: int) -> tuple[int, int]:
    return par._default_plan(n_tasks, n_panels, par.cpu_cores())


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "many", [2], {"a": 1}])
def test_invalid_workers_give_the_default_plan_with_one_warning(tuned_batch, bad):
    """A worker count that is not a finite number gives the built-in default plan."""
    tuned_batch["value"] = {"workers": bad, "threads": 1}
    with pytest.warns(RuntimeWarning, match="batch.small.workers"):
        plan = par.plan_parallel(8, 100)
    assert plan.source == "default"
    assert (plan.workers, plan.threads_per_worker) == _default_plan(8, 100)
    # The second call gives no second warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = par.plan_parallel(8, 100)
    assert again == plan


@pytest.mark.parametrize("bad", [math.nan, math.inf, False, "x", None])
def test_invalid_threads_give_the_default_threads(tuned_batch, bad):
    """A thread count that is not valid gives the cores divided by the workers."""
    tuned_batch["value"] = {"workers": 2, "threads": bad}
    cores = par.cpu_cores()
    if bad is None:
        plan = par.plan_parallel(8, 100)
    else:
        with pytest.warns(RuntimeWarning, match="batch.small.threads"):
            plan = par.plan_parallel(8, 100)
    w = max(1, min(2, 8, cores))
    assert plan.source == "profile"
    assert plan.workers == w
    assert plan.threads_per_worker == max(1, cores // w)


def test_valid_profile_values_are_kept(tuned_batch):
    """Valid values keep the profile plan (also a numeric text and a float)."""
    cores = par.cpu_cores()
    tuned_batch["value"] = {"workers": 1, "threads": 1}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan = par.plan_parallel(8, 100)
        assert plan == par.ParallelPlan(1, 1, "profile")
        tuned_batch["value"] = {"workers": "1", "threads": 1.0}
        assert par.plan_parallel(8, 100) == par.ParallelPlan(1, 1, "profile")
    tuned_batch["value"] = {"workers": 0}
    assert par.plan_parallel(8, 100).source == "default"
    assert cores >= 1
