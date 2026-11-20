# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""The theory text of the 3D panel Trefftz rule states the rule as a method choice with a measured order."""

from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs" / "theory"


def test_panel3d_trefftz_text_does_not_claim_second_order():
    text = (DOCS / "panel3d.md").read_text(encoding="utf-8")
    assert "from second order to first order" not in text
    assert "does not claim second-order spanwise convergence" in text
    assert "panel centroids" in text and "method choice" in text
    assert "observed order" in text


def test_ventorum_3dp_trefftz_text_points_to_the_measured_order():
    text = (DOCS / "ventorum_3dp.md").read_text(encoding="utf-8")
    assert "using the spacing parameter of Multhopp 1950 and Lan 1974" not in text
    assert "measured spanwise order between 1 and 2" in text
