# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Regression tests of the GPU settings: profile checks, environment values, auto precision.

- A lattice cost model with a repeated panel count, or with times that are
  not finite, is ignored with one warning; the dispatch then uses the
  built-in rule (before, the interpolation divided by zero).
- Profile values of ``min_work`` and ``min_cases`` that are not finite
  numbers give the defaults with one warning.
- An unknown ``VENTORUM_DEVICE`` or ``VENTORUM_GPU_PRECISION`` value gives
  one warning, and ``"auto"`` is used.
- ``"auto"`` precision is the precision stored in the profile, else float32.
"""

from __future__ import annotations

import warnings

import pytest

from ventorum import gpu
from ventorum.hardware import profile as profile_mod


def _good_model(n=(100, 1000)):
    return {"n": list(n),
            "cpu": {"k": [1, 33], "t": [[1e-3, 3e-2], [1e-2, 3e-1]]},
            "gpu": {"k": [1, 33], "t": [[5e-3, 6e-3], [6e-3, 7e-3]]}}


@pytest.fixture
def profile_settings(monkeypatch):
    """Replace the machine profile with the settings dict that the test fills in."""
    settings: dict = {}

    def fake(*keys, default=None):
        node = settings
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    monkeypatch.setattr(profile_mod, "tuned_setting", fake)
    monkeypatch.setattr(gpu, "_precision", "float32")
    monkeypatch.setattr(gpu, "_warned", set())
    return settings


def _runtime_warnings(record):
    return [str(w.message) for w in record if issubclass(w.category, RuntimeWarning)]


def test_valid_cost_model_is_used(profile_settings):
    profile_settings["gpu"] = {"model": {"vlm": {"float32": _good_model()}}}
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert gpu.cost_model("vlm") == _good_model()
    assert not _runtime_warnings(record)


@pytest.mark.parametrize("bad", [
    _good_model(n=(100, 100)),                       # repeated panel count
    _good_model(n=(1000, 100)),                      # decreasing panel count
    _good_model(n=(100, float("inf"))),              # count that is not finite
    {**_good_model(), "cpu": {"k": [1, 33], "t": [[float("inf"), 3e-2], [1e-2, 3e-1]]}},  # time inf
    {**_good_model(), "gpu": {"k": [1, 33], "t": [[float("nan"), 6e-3], [6e-3, 7e-3]]}},  # time nan
    {**_good_model(), "gpu": {"k": [1, float("inf")], "t": [[5e-3, 6e-3], [6e-3, 7e-3]]}},  # batch inf
])
def test_invalid_cost_model_is_ignored_with_one_warning(profile_settings, bad):
    profile_settings["gpu"] = {"model": {"vlm": {"float32": bad}}}
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert gpu.cost_model("vlm") is None
        assert gpu.cost_model("vlm") is None
    msgs = _runtime_warnings(record)
    assert len(msgs) == 1 and "gpu.model.vlm.float32" in msgs[0]


def test_repeated_panel_count_falls_back_to_the_built_in_rule(profile_settings, monkeypatch):
    """The auto dispatch with a broken model uses min_work and min_cases (no ZeroDivisionError)."""
    profile_settings["gpu"] = {"model": {"vlm": {"float32": _good_model(n=(100, 100))}}}
    monkeypatch.setattr(gpu, "_device", "auto")
    monkeypatch.setattr(gpu, "available", lambda: True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        use, reason = gpu.use_gpu(1.0, "vlm", n_cases=1, n_panels=50, return_reason=True)
    assert use is False
    assert reason.startswith("auto: work 1 < min_work")


@pytest.mark.parametrize("key, value", [("min_work", float("inf")), ("min_work", float("nan")),
                                        ("min_work", "x"), ("min_cases", float("inf")),
                                        ("min_cases", "x"), ("min_cases", True)])
def test_invalid_threshold_values_give_the_defaults(profile_settings, key, value):
    profile_settings["gpu"] = {key: {"vlm": value}}
    func = gpu.min_work if key == "min_work" else gpu.min_cases
    default = (gpu.DEFAULT_MIN_WORK if key == "min_work" else gpu.DEFAULT_MIN_CASES)["vlm"]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert func("vlm") == default
        assert func("vlm") == default
    msgs = _runtime_warnings(record)
    assert len(msgs) == 1 and f"gpu.{key}.vlm" in msgs[0]


def test_valid_threshold_values_are_used(profile_settings):
    profile_settings["gpu"] = {"min_work": {"vlm": 1.5e5}, "min_cases": {"vlm": 7}}
    assert gpu.min_work("vlm") == 1.5e5
    assert gpu.min_cases("vlm") == 7


@pytest.mark.parametrize("name, allowed", [("VENTORUM_DEVICE", gpu.DEVICES),
                                           ("VENTORUM_GPU_PRECISION", gpu.PRECISIONS)])
def test_unknown_environment_value_warns_and_uses_auto(monkeypatch, name, allowed):
    monkeypatch.setenv(name, "cuda")
    with pytest.warns(RuntimeWarning, match=rf"'cuda'.*{name}") as record:
        assert gpu._env(name, allowed, "auto") == "auto"
    assert len(record) == 1
    assert str(allowed) in str(record[0].message)


@pytest.mark.parametrize("value", ["gpu", " CPU ", "auto"])
def test_known_environment_value_gives_no_warning(monkeypatch, value):
    monkeypatch.setenv("VENTORUM_DEVICE", value)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert gpu._env("VENTORUM_DEVICE", gpu.DEVICES, "auto") == value.strip().lower()


def test_auto_precision_is_the_profile_precision(profile_settings, monkeypatch):
    monkeypatch.setattr(gpu, "_precision", "auto")
    assert gpu.get_precision() == gpu.AUTO_PRECISION == "float32"
    profile_settings["gpu"] = {"precision": "float64"}
    assert gpu.get_precision() == "float64"
    profile_settings["gpu"] = {"precision": "half"}
    assert gpu.get_precision() == "float32"
    monkeypatch.setattr(gpu, "_precision", "float32")
    profile_settings["gpu"] = {"precision": "float64"}
    assert gpu.get_precision() == "float32"
