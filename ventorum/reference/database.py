# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Reference cases in Markdown with one TOML block: format, parser and data types.

A reference case is a file ``<case-id>.md``. It holds free Markdown text for
people, and exactly one fenced code block with the info string ``toml``. Only
that block is read, with :mod:`tomllib` (Python standard library).

Tables of the TOML block:

* ``[identity]``: ``id`` (equals the file name), ``title``, ``entry_version``.
* ``[source]``: ``citation``, ``doi_or_url`` (optional), ``location`` (page,
  table or figure), ``type`` (``experiment``, ``cfd`` or ``theory``),
  ``grade`` (``A``, ``B`` or ``C``; required for ``experiment`` and ``cfd``,
  absent for ``theory``), ``synthetic`` (boolean, default false).
* ``[geometry]``: the aircraft, in the structure that
  :func:`ventorum.agent.schemas.build_aircraft_from_spec` reads.
* ``[conditions]``: ``mach``, ``reynolds``, ``beta_deg`` [deg], ``h_m``
  (optional height [m]), ``corrections``.
* ``[[results]]`` (one or more): ``quantity`` (``CL``, ``CD``, ``CDi``,
  ``Cm``, ``CY``, ``Cl``, ``Cn``), ``alpha_deg`` [deg] (list), ``values``
  (list, same length), ``uncertainty`` (number or list, optional),
  ``uncertainty_kind`` (``absolute`` or ``relative``; required with
  ``uncertainty``), ``moment_point`` (for moments; list of 3 values [m]),
  ``h_m`` (optional list of heights [m], one for each point, each > 0),
  ``height_point`` (optional point [x, y, z] [m] whose height ``h_m`` gives;
  when missing, the height is the height of the moment reference point of
  the solve; refused without a ``h_m`` list), ``extraction``
  (``table`` or ``digitised``), ``digitising_error`` (required
  when digitised).
* ``[quality]`` (optional): ``notes``.

Grades: A is an experiment with stated uncertainty and documented
corrections; B is an experiment without stated uncertainty, or high-order
CFD with a grid study; C is digitised plots or one unchecked source. A
result of another low-order tool is never a reference. A theory case has
no grade: it is verification, not validation.

The parser refuses a file with a missing required field, a value that starts
with ``TODO``, lists of different lengths, an unknown quantity, a grade
on a theory case, a ``h_m`` list with a length different from ``alpha_deg``,
a ``h_m`` value that is not > 0, a ``height_point`` without 3 values, or a
``height_point`` without a ``h_m`` list, and names the file and the field.
"""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Quantities that a reference result can hold.
QUANTITIES = ("CL", "CD", "CDi", "Cm", "CY", "Cl", "Cn")
#: Quantities that need a moment reference point.
MOMENT_QUANTITIES = ("Cm", "Cl", "Cn")
#: Allowed source types.
SOURCE_TYPES = ("experiment", "cfd", "theory")
#: Allowed source grades (experiments and CFD; a theory case has none).
GRADES = ("A", "B", "C")
#: Largest Mach number of the present scope. Cases above it are listed as
#: outside the envelope and are not run.
MACH_LIMIT = 0.3

_FENCE = re.compile(r"^```toml[ \t]*\r?\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


class ReferenceError(ValueError):
    """A reference case file that cannot be used (it names the file and the field)."""


@dataclass(slots=True)
class ResultEntry:
    """One measured or computed series of a reference case.

    Parameters
    ----------
    quantity : str
        One of ``CL``, ``CD``, ``CDi``, ``Cm``, ``CY``, ``Cl``, ``Cn``.
    alpha_deg : list[float]
        Angles of attack [deg].
    values : list[float]
        Reference values at ``alpha_deg``.
    abs_uncertainty : list[float] or None
        Absolute uncertainty of each value (relative uncertainties are
        converted with ``abs(value)``). None when the source states none.
    extraction : str
        ``table`` or ``digitised``.
    digitising_error : float or None
        Estimated reading error (required when digitised).
    moment_point : list[float] or None
        Moment reference point [x, y, z] in metres (required for moments).
    h_m : list[float] or None
        Height [m] of each point above the ground, one for each angle of
        ``alpha_deg`` (each value must be > 0). None means the height of
        ``[conditions] h_m`` (free air when that is also None).
    height_point : list[float] or None
        Point [x, y, z] of the model in metres whose height ``h_m`` gives.
        None means the moment reference point of the solve (``moment_point``
        when the result has one, else ``Aircraft.ref_point``).
    """

    quantity: str = "CL"
    alpha_deg: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    abs_uncertainty: list[float] | None = None
    extraction: str = "table"
    digitising_error: float | None = None
    moment_point: list[float] | None = None
    h_m: list[float] | None = None
    height_point: list[float] | None = None


@dataclass(slots=True)
class ReferenceCase:
    """One parsed reference case.

    Parameters
    ----------
    file : str
        File name of the case.
    id : str
        Case identifier (equals the file name without ``.md``).
    title : str
        Short name of the case.
    entry_version : int
        Version of the entry.
    citation : str
        Full citation of the source.
    doi_or_url : str or None
        DOI or URL of the source, when one exists.
    location : str
        Page, table or figure the values come from.
    source_type : str
        ``experiment``, ``cfd`` or ``theory``.
    grade : str or None
        ``A``, ``B`` or ``C`` for an experiment or CFD case. None for a
        theory case (a theory case has no grade).
    synthetic : bool
        True for a made-up case (marked, never counted in statistics).
    aircraft : Aircraft
        Geometry built from the ``[geometry]`` table.
    mach : float
        Mach number of the test.
    reynolds : float
        Reynolds number of the test.
    beta_deg : float
        Sideslip angle [deg].
    h_m : float or None
        Height [m] of the moment reference point above the ground, if in
        ground effect.
    corrections : str
        Corrections applied in the source (or ``none stated``).
    results : list[ResultEntry]
        Reference series of the case.
    notes : str
        Quality notes.
    """

    file: str = ""
    id: str = ""
    title: str = ""
    entry_version: int = 1
    citation: str = ""
    doi_or_url: str | None = None
    location: str = ""
    source_type: str = "experiment"
    grade: str | None = None
    synthetic: bool = False
    aircraft: Any = None
    mach: float = 0.0
    reynolds: float = 0.0
    beta_deg: float = 0.0
    h_m: float | None = None
    corrections: str = ""
    results: list[ResultEntry] = field(default_factory=list)
    notes: str = ""

    @property
    def in_envelope(self) -> bool:
        """True when the case is inside the present scope (Mach at most 0.3)."""
        return self.mach <= MACH_LIMIT


def _fail(name: str, field_path: str, reason: str) -> ReferenceError:
    return ReferenceError(f"{name}: field '{field_path}': {reason}.")


def _required(data: dict[str, Any], key: str, where: str, name: str) -> Any:
    if not isinstance(data, dict) or key not in data or data[key] is None:
        raise _fail(name, f"{where}.{key}", "missing required field")
    return data[key]


def _text(value: Any, field_path: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(name, field_path, f"must be a non-empty string, got {value!r}")
    return value


def _number(value: Any, field_path: str, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _fail(name, field_path, f"must be a finite number, got {value!r}")
    out = float(value)
    if minimum is not None and out < minimum:
        raise _fail(name, field_path, f"must be >= {minimum:g}, got {out:g}")
    return out


def _number_list(value: Any, field_path: str, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise _fail(name, field_path, f"must be a non-empty list of numbers, got {value!r}")
    return [_number(v, f"{field_path}[{i}]", name) for i, v in enumerate(value)]


def _check_todo(value: Any, field_path: str, name: str) -> None:
    """Refuse any string value that starts with TODO, naming the file and the field."""
    if isinstance(value, str):
        if value.lstrip().startswith("TODO"):
            raise _fail(name, field_path, f"value starts with TODO ({value!r})")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_todo(item, f"{field_path}.{key}", name)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _check_todo(item, f"{field_path}[{i}]", name)


def _parse_result(data: Any, index: int, name: str) -> ResultEntry:
    """Parse one ``[[results]]`` table."""
    where = f"results[{index}]"
    if not isinstance(data, dict):
        raise _fail(name, where, f"must be a table, got {data!r}")
    quantity = _required(data, "quantity", where, name)
    if quantity not in QUANTITIES:
        raise _fail(name, f"{where}.quantity", f"unknown quantity {quantity!r}")
    alpha = _number_list(_required(data, "alpha_deg", where, name), f"{where}.alpha_deg", name)
    values = _number_list(_required(data, "values", where, name), f"{where}.values", name)
    if len(alpha) != len(values):
        raise _fail(
            name,
            where,
            f"lists of different lengths: 'alpha_deg' has {len(alpha)} values, "
            f"'values' has {len(values)} values",
        )
    extraction = _required(data, "extraction", where, name)
    if extraction not in ("table", "digitised"):
        raise _fail(name, f"{where}.extraction", f"must be 'table' or 'digitised', got {extraction!r}")
    digitising_error = None
    if extraction == "digitised":
        digitising_error = _number(
            _required(data, "digitising_error", where, name), f"{where}.digitising_error", name, minimum=0.0
        )
    elif "digitising_error" in data and data["digitising_error"] is not None:
        digitising_error = _number(data["digitising_error"], f"{where}.digitising_error", name, minimum=0.0)
    moment_point = None
    if quantity in MOMENT_QUANTITIES:
        raw = _required(data, "moment_point", where, name)
        moment_point = _number_list(raw, f"{where}.moment_point", name)
        if len(moment_point) != 3:
            raise _fail(name, f"{where}.moment_point", "must hold 3 values [x, y, z] in metres")
    elif "moment_point" in data and data["moment_point"] is not None:
        moment_point = _number_list(data["moment_point"], f"{where}.moment_point", name)
        if len(moment_point) != 3:
            raise _fail(name, f"{where}.moment_point", "must hold 3 values [x, y, z] in metres")
    abs_uncertainty = None
    h_m = None
    if "h_m" in data and data["h_m"] is not None:
        h_m = _number_list(data["h_m"], f"{where}.h_m", name)
        if len(h_m) != len(alpha):
            raise _fail(
                name,
                where,
                f"lists of different lengths: 'alpha_deg' has {len(alpha)} values, "
                f"'h_m' has {len(h_m)} values",
            )
        for i, h in enumerate(h_m):
            if h <= 0.0:
                raise _fail(name, f"{where}.h_m[{i}]", f"must be > 0, got {h:g}")
    height_point = None
    if "height_point" in data and data["height_point"] is not None:
        if h_m is None:
            raise _fail(name, f"{where}.height_point", "present without a 'h_m' list")
        height_point = _number_list(data["height_point"], f"{where}.height_point", name)
        if len(height_point) != 3:
            raise _fail(name, f"{where}.height_point", "must hold 3 values [x, y, z] in metres")
    if "uncertainty" in data and data["uncertainty"] is not None:
        kind = _required(data, "uncertainty_kind", where, name)
        if kind not in ("absolute", "relative"):
            raise _fail(name, f"{where}.uncertainty_kind", f"must be 'absolute' or 'relative', got {kind!r}")
        raw = data["uncertainty"]
        unc = _number_list(raw, f"{where}.uncertainty", name) if isinstance(raw, list) else [_number(raw, f"{where}.uncertainty", name)] * len(values)
        if len(unc) != len(values):
            raise _fail(
                name,
                where,
                f"lists of different lengths: 'uncertainty' has {len(unc)} values, "
                f"'values' has {len(values)} values",
            )
        if any(u < 0.0 for u in unc):
            raise _fail(name, f"{where}.uncertainty", "uncertainty must be >= 0")
        abs_uncertainty = list(unc) if kind == "absolute" else [abs(v) * u for v, u in zip(values, unc)]
    elif "uncertainty_kind" in data and data["uncertainty_kind"] is not None:
        raise _fail(name, f"{where}.uncertainty_kind", "present without 'uncertainty'")
    return ResultEntry(
        quantity=quantity,
        alpha_deg=alpha,
        values=values,
        abs_uncertainty=abs_uncertainty,
        extraction=extraction,
        digitising_error=digitising_error,
        moment_point=moment_point,
        h_m=h_m,
        height_point=height_point,
    )


def load_case(path: str | Path) -> ReferenceCase:
    """Load one reference case file (``<case-id>.md``).

    Parameters
    ----------
    path : str or Path
        Path of the case file.

    Returns
    -------
    ReferenceCase
        The parsed case, with the geometry built.

    Raises
    ------
    ReferenceError
        If the file has no (or more than one) ``toml`` block, a required
        field is missing, a value starts with ``TODO``, two lists have
        different lengths, the quantity is unknown, a theory case has a
        grade, a ``h_m`` list has a length different from ``alpha_deg`` or a
        value that is not > 0, ``height_point`` has not 3 values, or
        ``height_point`` is present without a ``h_m`` list. The message
        names the file and the field.
    """
    from ventorum.agent.schemas import build_aircraft_from_spec

    path = Path(path)
    name = path.name
    text = path.read_text(encoding="utf-8")
    blocks = _FENCE.findall(text)
    if len(blocks) != 1:
        raise ReferenceError(
            f"{name}: the file must hold exactly one fenced code block with the info string "
            f"'toml', found {len(blocks)}."
        )
    try:
        data = tomllib.loads(blocks[0])
    except tomllib.TOMLDecodeError as exc:
        raise ReferenceError(f"{name}: the TOML block is not valid TOML: {exc}.") from exc
    _check_todo(data, "case", name)

    identity = _required(data, "identity", "case", name)
    case_id = _text(_required(identity, "id", "identity", name), "identity.id", name)
    if case_id != path.stem:
        raise _fail(name, "identity.id", f"{case_id!r} does not equal the file name {path.stem!r}")
    title = _text(_required(identity, "title", "identity", name), "identity.title", name)
    entry_version = _required(identity, "entry_version", "identity", name)
    if isinstance(entry_version, bool) or not isinstance(entry_version, int):
        raise _fail(name, "identity.entry_version", f"must be an integer, got {entry_version!r}")

    source = _required(data, "source", "case", name)
    citation = _text(_required(source, "citation", "source", name), "source.citation", name)
    location = _text(_required(source, "location", "source", name), "source.location", name)
    source_type = _required(source, "type", "source", name)
    if source_type not in SOURCE_TYPES:
        raise _fail(name, "source.type", f"must be one of {list(SOURCE_TYPES)}, got {source_type!r}")
    grade = None
    if source_type == "theory":
        if "grade" in source:
            raise _fail(
                name,
                "source.grade",
                "theory cases have no grade: a theory case is verification, not validation",
            )
    else:
        grade = _required(source, "grade", "source", name)
        if grade not in GRADES:
            raise _fail(name, "source.grade", f"must be one of {list(GRADES)}, got {grade!r}")
    doi_or_url = source.get("doi_or_url")
    if doi_or_url is not None and not isinstance(doi_or_url, str):
        raise _fail(name, "source.doi_or_url", f"must be a string, got {doi_or_url!r}")
    synthetic = source.get("synthetic", False)
    if not isinstance(synthetic, bool):
        raise _fail(name, "source.synthetic", f"must be a boolean, got {synthetic!r}")

    geometry = _required(data, "geometry", "case", name)
    if not isinstance(geometry, dict):
        raise _fail(name, "geometry", f"must be a table, got {geometry!r}")
    try:
        aircraft = build_aircraft_from_spec(geometry, where="geometry")
    except ValueError as exc:
        raise _fail(name, "geometry", str(exc)) from exc

    conditions = _required(data, "conditions", "case", name)
    mach = _number(_required(conditions, "mach", "conditions", name), "conditions.mach", name, minimum=0.0)
    reynolds = _number(_required(conditions, "reynolds", "conditions", name), "conditions.reynolds", name, minimum=0.0)
    if reynolds <= 0.0:
        raise _fail(name, "conditions.reynolds", "must be > 0")
    beta_deg = _number(_required(conditions, "beta_deg", "conditions", name), "conditions.beta_deg", name)
    corrections = _text(_required(conditions, "corrections", "conditions", name), "conditions.corrections", name)
    h_m = None
    if "h_m" in conditions and conditions["h_m"] is not None:
        h_m = _number(conditions["h_m"], "conditions.h_m", name, minimum=0.0)
        if h_m <= 0.0:
            raise _fail(name, "conditions.h_m", "must be > 0")

    raw_results = _required(data, "results", "case", name)
    if not isinstance(raw_results, list) or not raw_results:
        raise _fail(name, "results", "must hold at least one [[results]] table")
    results = [_parse_result(item, i, name) for i, item in enumerate(raw_results)]

    notes = ""
    if "quality" in data and data["quality"] is not None:
        quality = data["quality"]
        if not isinstance(quality, dict):
            raise _fail(name, "quality", f"must be a table, got {quality!r}")
        notes = quality.get("notes", "")
        if not isinstance(notes, str):
            raise _fail(name, "quality.notes", f"must be a string, got {notes!r}")

    return ReferenceCase(
        file=name,
        id=case_id,
        title=title,
        entry_version=entry_version,
        citation=citation,
        doi_or_url=doi_or_url,
        location=location,
        source_type=source_type,
        grade=grade,
        synthetic=synthetic,
        aircraft=aircraft,
        mach=mach,
        reynolds=reynolds,
        beta_deg=beta_deg,
        h_m=h_m,
        corrections=corrections,
        results=results,
        notes=notes,
    )


def load_database(folder: str | Path) -> list[ReferenceCase]:
    """Load all reference cases (``*.md``) of a folder, in file order.

    Parameters
    ----------
    folder : str or Path
        Folder with the case files.

    Returns
    -------
    list[ReferenceCase]
        The parsed cases.

    Raises
    ------
    ReferenceError
        If any file cannot be used. The message names the file and the field.
    """
    return [load_case(p) for p in sorted(Path(folder).glob("*.md"))]
