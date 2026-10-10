# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests of the stability derivatives in the selected axes.

The derivatives follow the ``axes`` choice. They are taken in the axes
of the reference flight condition (alpha_0, beta_0), held fixed while
alpha and beta change (B. Etkin and L. D. Reid, Dynamics of Flight:
Stability and Control, 3rd ed., Wiley, 1996, chapter on the equations
of motion and the stability derivatives). The rotation R(alpha_0,
beta_0) is constant, so each derivative vector in the selected axes is
the rotation of the body-axis derivative vector. Force derivatives
(CL_alpha, CY_beta) do not change. Static margin and neutral point use
body-axis Cm_alpha and do not depend on the axes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ventorum.agent import stability_derivatives, wing_analysis
from ventorum.core.axes import rotation_body_to_stability, rotation_body_to_wind

FAST = {"n_panels": 8}
RECT = {"span_m": 10.0, "chord_m": 1.0}


def test_body_axes_unchanged():
    """With axes='body' (and without axes) the output keeps the old values."""
    cond = {"alpha_deg": 4.0, "beta_deg": 2.0}
    body = stability_derivatives(RECT, cond, settings=FAST, axes="body")
    default = stability_derivatives(RECT, cond, settings=FAST)
    assert body["status"] == "success", body
    assert default["status"] == "success", default
    assert body["axes"] == "body"
    assert default["axes"] == "body"
    assert body["stability_derivatives"] == default["stability_derivatives"]
    assert body["stability_assessment"] == default["stability_assessment"]
    assert body["base_point"] == default["base_point"]
    # Body values are central differences of the body-axis moments.
    base = wing_analysis(RECT, cond, FAST, "summary", axes="body")
    assert base["status"] == "success", base
    d = body["stability_derivatives"]
    assert d["CL_alpha_per_rad"] == pytest.approx(
        (wing_analysis(RECT, {"alpha_deg": 4.5, "beta_deg": 2.0}, FAST, "summary",
                        axes="body")["metrics"]["CL"]
         - wing_analysis(RECT, {"alpha_deg": 3.5, "beta_deg": 2.0}, FAST, "summary",
                         axes="body")["metrics"]["CL"]) / np.radians(1.0),
        abs=5e-5)
    # The old note is gone; the new note names the axes.
    assert not any("The derivatives are body-axis moments per rad" in n for n in body["notes"])
    assert any("'body'" in n for n in body["notes"])


def test_derivatives_are_rotated_body_derivatives(monkeypatch):
    """At alpha 6 deg and beta 4 deg the selected derivatives are R times body."""
    import math as _math

    from ventorum.agent import tools

    def _noround(x, n=6):
        if x is None:
            return None
        v = float(x)
        return v if _math.isfinite(v) else None

    monkeypatch.setattr(tools, "rnd", _noround)
    cond = {"alpha_deg": 6.0, "beta_deg": 4.0}
    body = stability_derivatives(RECT, cond, settings=FAST, axes="body")
    stab = stability_derivatives(RECT, cond, settings=FAST, axes="stability")
    wind = stability_derivatives(RECT, cond, settings=FAST, axes="wind")
    assert body["status"] == "success", body
    assert stab["status"] == "success", stab
    assert wind["status"] == "success", wind
    a0, b0 = np.radians(6.0), np.radians(4.0)
    r_bs = rotation_body_to_stability(a0)
    r_bw = rotation_body_to_wind(a0, b0)
    db = body["stability_derivatives"]
    ds = stab["stability_derivatives"]
    dw = wind["stability_derivatives"]
    # Force derivatives do not change with the axes.
    assert ds["CL_alpha_per_rad"] == pytest.approx(db["CL_alpha_per_rad"], abs=1e-12)
    assert dw["CL_alpha_per_rad"] == pytest.approx(db["CL_alpha_per_rad"], abs=1e-12)
    assert ds["CY_beta_per_rad"] == pytest.approx(db["CY_beta_per_rad"], abs=1e-12)
    assert dw["CY_beta_per_rad"] == pytest.approx(db["CY_beta_per_rad"], abs=1e-12)
    # Stability Cm_alpha is the body value (middle row of R_bs is [0, 1, 0]).
    assert ds["Cm_alpha_per_rad"] == pytest.approx(db["Cm_alpha_per_rad"], abs=1e-12)
    # Stability Cl_beta and Cn_beta use only the reported body pair.
    assert ds["Cl_beta_per_rad"] == pytest.approx(
        r_bs[0, 0] * db["Cl_beta_per_rad"] + r_bs[0, 2] * db["Cn_beta_per_rad"], abs=1e-12)
    assert ds["Cn_beta_per_rad"] == pytest.approx(
        r_bs[2, 0] * db["Cl_beta_per_rad"] + r_bs[2, 2] * db["Cn_beta_per_rad"], abs=1e-12)
    # Wind needs the full body vectors: rebuild them from unrounded point moments.
    cases = {"a+": (6.5, 4.0), "a-": (5.5, 4.0), "b+": (6.0, 5.0), "b-": (6.0, 3.0)}
    mom = {}
    for key, (aa, bb) in cases.items():
        p = wing_analysis(RECT, {"alpha_deg": aa, "beta_deg": bb}, FAST, "summary",
                          axes="body")
        assert p["status"] == "success", p
        mom[key] = p["metrics"]
    da, dbb = np.radians(1.0), np.radians(2.0)
    d_m_da = np.array([(mom["a+"]["Cl"] - mom["a-"]["Cl"]) / da,
                       (mom["a+"]["Cm"] - mom["a-"]["Cm"]) / da,
                       (mom["a+"]["Cn"] - mom["a-"]["Cn"]) / da])
    d_m_db = np.array([(mom["b+"]["Cl"] - mom["b-"]["Cl"]) / dbb,
                       (mom["b+"]["Cm"] - mom["b-"]["Cm"]) / dbb,
                       (mom["b+"]["Cn"] - mom["b-"]["Cn"]) / dbb])
    exp_wind_a = r_bw @ d_m_da
    exp_wind_b = r_bw @ d_m_db
    assert dw["Cm_alpha_per_rad"] == pytest.approx(float(exp_wind_a[1]), abs=1e-12)
    assert dw["Cl_beta_per_rad"] == pytest.approx(float(exp_wind_b[0]), abs=1e-12)
    assert dw["Cn_beta_per_rad"] == pytest.approx(float(exp_wind_b[2]), abs=1e-12)


def test_stability_axes_against_finite_differences():
    """Stability Cl_beta and Cn_beta match differences of stability moments."""
    cond = {"alpha_deg": 6.0, "beta_deg": 0.0}
    got = stability_derivatives(RECT, cond, settings=FAST, axes="stability")
    assert got["status"] == "success", got
    d = got["stability_derivatives"]
    hi = wing_analysis(RECT, {"alpha_deg": 6.0, "beta_deg": 1.0}, FAST, "summary",
                       axes="stability")
    lo = wing_analysis(RECT, {"alpha_deg": 6.0, "beta_deg": -1.0}, FAST, "summary",
                       axes="stability")
    assert hi["status"] == "success", hi
    assert lo["status"] == "success", lo
    db = np.radians(2.0)
    cl_b = (hi["metrics"]["Cl"] - lo["metrics"]["Cl"]) / db
    cn_b = (hi["metrics"]["Cn"] - lo["metrics"]["Cn"]) / db
    # Same step and same fixed axes, so only rounding separates the two.
    assert d["Cl_beta_per_rad"] == pytest.approx(cl_b, abs=1e-4)
    assert d["Cn_beta_per_rad"] == pytest.approx(cn_b, abs=1e-4)


def test_all_returns_three_sets():
    """axes='all' returns the derivatives of the three sets; main is body."""
    cond = {"alpha_deg": 6.0, "beta_deg": 4.0}
    all_axes = stability_derivatives(RECT, cond, settings=FAST, axes="all")
    assert all_axes["status"] == "success", all_axes
    assert all_axes["axes"] == "all"
    by_axes = all_axes["derivatives_by_axes"]
    assert set(by_axes) == {"body", "stability", "wind"}
    main = all_axes["stability_derivatives"]
    for key, val in by_axes["body"].items():
        assert main[key] == pytest.approx(val) if isinstance(val, float) else main[key] == val
    for name in ("body", "stability", "wind"):
        single = stability_derivatives(RECT, cond, settings=FAST, axes=name)
        assert single["status"] == "success", (name, single)
        for key in ("Cm_alpha_per_rad", "Cl_beta_per_rad", "Cn_beta_per_rad",
                    "CL_alpha_per_rad", "CY_beta_per_rad"):
            assert by_axes[name][key] == pytest.approx(
                single["stability_derivatives"][key])
    # The base point keeps the body values and adds the three moment sets.
    assert "moments" in all_axes["base_point"]
    assert set(all_axes["base_point"]["moments"]) == {"body", "stability", "wind"}


def test_static_margin_does_not_depend_on_axes():
    """Neutral point and static margin are equal in every set."""
    cond = {"alpha_deg": 4.0, "beta_deg": 2.0}
    outs = {}
    for name in ("body", "stability", "wind", "all"):
        p = stability_derivatives(RECT, cond, settings=FAST, axes=name)
        assert p["status"] == "success", (name, p)
        outs[name] = p["stability_derivatives"]
    for name in ("stability", "wind", "all"):
        assert outs[name]["neutral_point_x_m"] == pytest.approx(outs["body"]["neutral_point_x_m"])
        assert outs[name]["static_margin_fraction"] == pytest.approx(
            outs["body"]["static_margin_fraction"])
    # The margin uses the body-axis Cm_alpha.
    d = outs["body"]
    assert d["static_margin_fraction"] == pytest.approx(
        -d["Cm_alpha_per_rad"] / d["CL_alpha_per_rad"], abs=1e-5)
    assert math.isfinite(d["static_margin_fraction"])
