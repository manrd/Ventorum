# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""The experimental-data loader must refuse data without full provenance."""

import json
from pathlib import Path

import pytest

from ventorum.reference.experimental import (
    REQUIRED_FIELDS,
    ProvenanceError,
    check_dataset,
    load_all,
    load_dataset,
)

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "validation" / "experimental" / "TEMPLATE.json"


def _complete() -> dict:
    """A dataset with every provenance field filled (loader test only, not real data)."""
    d = {k: f"value for {k}" for k in REQUIRED_FIELDS}
    d["data_columns"] = ["alpha_deg", "CL"]
    d["data"] = [[0.0, 0.0], [2.0, 0.1]]
    return d


def test_template_is_refused():
    with pytest.raises(ProvenanceError):
        load_dataset(TEMPLATE)


def test_complete_dataset_is_accepted():
    check_dataset(_complete())


@pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
def test_missing_or_empty_field_is_refused(field):
    d = _complete()
    del d[field]
    with pytest.raises(ProvenanceError):
        check_dataset(d)
    d = _complete()
    d[field] = "" if isinstance(d[field], str) else []
    with pytest.raises(ProvenanceError):
        check_dataset(d)


def test_placeholder_is_refused():
    d = _complete()
    d["page"] = "TODO page number"
    with pytest.raises(ProvenanceError):
        check_dataset(d)


def test_row_length_is_checked():
    d = _complete()
    d["data"].append([4.0])
    with pytest.raises(ProvenanceError):
        check_dataset(d)


def test_load_all_skips_template(tmp_path):
    (tmp_path / "TEMPLATE.json").write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "case.json").write_text(json.dumps(_complete()), encoding="utf-8")
    out = load_all(tmp_path)
    assert len(out) == 1 and out[0]["data_columns"] == ["alpha_deg", "CL"]


def test_repository_datasets_have_provenance():
    for d in load_all(REPO / "validation" / "experimental"):
        check_dataset(d)
