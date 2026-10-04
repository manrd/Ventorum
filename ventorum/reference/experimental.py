# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Experimental datasets with mandatory provenance.

A dataset is a JSON file. It is refused unless every provenance field is
present and not empty, so a number cannot enter the validation without a
traceable source. See ``validation/experimental/TEMPLATE.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "title": "Short name of the case.",
    "source": "Full citation (authors, title, report series and number, year).",
    "report_id": "Report or paper identifier, e.g. 'NASA TN D-926'.",
    "table_or_figure": "Exact table or figure number the values come from.",
    "page": "Page number(s) in the source.",
    "transcription": "How the values were taken: 'table' or 'digitized figure' (with tool and estimated reading error).",
    "configuration": "Geometry: planform, aspect ratio, taper, sweep, dihedral, twist, airfoil, chord, span.",
    "conditions": "Test conditions: Reynolds number, Mach number, transition, ground board, mounting.",
    "corrections": "Wind-tunnel corrections applied in the source (or 'none stated').",
    "data_columns": "Names and units of the data columns, in order.",
    "data": "Rows of values, each with one value per column.",
}


class ProvenanceError(ValueError):
    """A dataset without complete provenance."""


def check_dataset(d: dict[str, Any]) -> None:
    """Raise :class:`ProvenanceError` if a provenance field is missing or empty."""
    missing = [k for k in REQUIRED_FIELDS if k not in d or d[k] in (None, "", [], {})]
    if missing:
        raise ProvenanceError(f"Dataset '{d.get('title', '?')}' is missing provenance fields: {', '.join(missing)}.")
    placeholders = [k for k in REQUIRED_FIELDS if isinstance(d[k], str) and d[k].strip().upper().startswith("TODO")]
    if placeholders:
        raise ProvenanceError(f"Dataset '{d['title']}' has placeholder values in: {', '.join(placeholders)}.")
    n_col = len(d["data_columns"])
    bad = [i for i, row in enumerate(d["data"]) if len(row) != n_col]
    if bad:
        raise ProvenanceError(f"Dataset '{d['title']}': rows {bad} do not have {n_col} values.")


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load a dataset JSON file and check its provenance."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    check_dataset(d)
    return d


def load_all(folder: str | Path) -> list[dict[str, Any]]:
    """All datasets in a folder (``*.json`` except the template)."""
    out = []
    for p in sorted(Path(folder).glob("*.json")):
        if p.name.upper().startswith("TEMPLATE"):
            continue
        out.append(load_dataset(p))
    return out
