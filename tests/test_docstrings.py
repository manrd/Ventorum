# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Rules of AGENTS.md that ruff does not check (review round 5, C12 and C14).

* Every public parameter or attribute with a physical name has a
  description with its unit in the docstring.
* Every dependency in ``pyproject.toml`` has bounded versions.
"""

from __future__ import annotations

import inspect
import re
import tomllib
from pathlib import Path

import ventorum as vt

#: Names of physical quantities. A public parameter with one of these
#: names must have a description with a unit.
PHYSICAL = re.compile(
    r"^(alpha|beta|phi|h|height|heights|V_inf|V|rho|chord|semi_span|span|twist|sweep|sweep_le|"
    r"dihedral|incidence|x_le|z_le|position|S_ref|b_ref|c_ref|ref_point|alpha_deg|alpha_deg_range|"
    r"alphas|alphas_deg|phis_deg|h_ref|phi_deg|beta_deg)$"
)
#: A unit in the description: a bracket [..], or a unit word.
UNIT = re.compile(r"\[|\bm/s\b|\brad\b|\bdeg|degree|radian|\bm\b|metre|meter|kg|\bm\^2|dimensionless|fraction")


def _missing_units() -> list[str]:
    missing = []
    for name in sorted(dir(vt)):
        obj = getattr(vt, name)
        if name.startswith("_") or not callable(obj):
            continue
        doc = inspect.getdoc(obj) or ""
        try:
            params = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            continue
        for p in params:
            if not PHYSICAL.match(p):
                continue
            m = re.search(rf"^{re.escape(p)}\b[^\n]*\n((?:[ \t]+[^\n]*\n?)*)", doc, re.M)
            if m is None:
                missing.append(f"{name}({p}): not documented")
            elif not UNIT.search(m.group(0)):
                missing.append(f"{name}({p}): no unit")
    return missing


def test_public_parameters_documented():
    missing = _missing_units()
    assert not missing, "Public parameters without a description or a unit:\n" + "\n".join(missing)


def _requirements() -> list[str]:
    data = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    reqs = list(data["build-system"]["requires"]) + list(data["project"]["dependencies"])
    for group in data["project"].get("optional-dependencies", {}).values():
        reqs.extend(group)
    return reqs


def test_dependency_versions_are_bounded():
    """Each requirement has an upper bound (``<`` or ``==``)."""
    unbounded = [r for r in _requirements() if "<" not in r and "==" not in r]
    assert not unbounded, f"Requirements without an upper bound: {unbounded}"
