# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""The public repository holds no private planning IDs (rule of AGENTS.md).

Task IDs (``T-`` and four digits), decision IDs (``D-`` and two digits) and
plan items (``P1-`` or ``P2-`` and two digits) belong to the owner's private
planning files. A public file gives the reason in plain words instead.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Private ID patterns. "NASA TN D-3767" (a report number) is not one.
PRIVATE_ID = re.compile(r"(?<!TN )\b(T-\d{4}|D-\d{2}|P[12]-\d{2})\b")

#: Folders that keep historical text unchanged.
EXCLUDED = ("ventorum/legacy/",)

#: This file names the patterns.
SELF = "tests/test_no_private_ids.py"

#: File types that are text.
TEXT_SUFFIXES = {".py", ".md", ".pyx", ".toml", ".cfg", ".txt", ".yml", ".yaml", ".json", ".rst", ".ini"}


def _tracked_files() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available or this is not a git checkout")
    return [line for line in out.stdout.splitlines() if line]


def test_no_private_ids_in_tracked_files():
    """No tracked text file outside the legacy package names a private ID."""
    hits = []
    for rel in _tracked_files():
        if rel == SELF or rel.startswith(EXCLUDED) or Path(rel).suffix not in TEXT_SUFFIXES:
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), start=1):
            m = PRIVATE_ID.search(line)
            if m:
                hits.append(f"{rel}:{n}: {m.group(0)}")
    assert not hits, "Private planning IDs in public files:\n" + "\n".join(hits)
