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
  ``uncertainty``), ``moment_point`` (for moments and moment derivatives;
  list of 3 values [m]),
  ``h_m`` (optional list of heights [m], one for each point, each > 0),
  ``height_point`` (optional point [x, y, z] [m] whose height ``h_m`` gives;
  when missing, the height is the height of the moment reference point of
  the solve; refused without a ``h_m`` list), ``extraction``
  (``table`` or ``digitised``), ``digitising_error`` (required
  when digitised).
* ``[quality]`` (optional): ``notes``.

Derivative quantities, abscissas and slope entries:

* ``quantity`` can also be a static derivative ``<C>_<v>``: ``<C>`` is
  ``CL``, ``CD``, ``CY``, ``Cl``, ``Cm`` or ``Cn`` and ``<v>`` is ``alpha``
  or ``beta`` (for example ``Cl_beta``). A derivative quantity needs
  ``derivative_unit`` (``1/deg`` or ``1/rad``, the unit of the stored
  values). A moment derivative needs ``moment_point``, and takes ``axes``
  (``body``, ``stability`` or ``wind``; default ``stability``, as NACA
  stability-tunnel data). ``derivative_unit`` and ``axes`` are refused on
  other quantities.
* A result has either ``alpha_deg`` (list) or ``CL`` (list of lift
  coefficients). With ``CL``, the runner finds the angle of attack where
  Ventorum gives that lift coefficient and evaluates the quantity there; a
  lift coefficient beyond the Ventorum range of the alpha sweep is "not
  reached", never extrapolated.
* A slope entry has ``slope_of = "CL"``, ``CL_range = [lo, hi]`` (lo < hi)
  and exactly one value: the slope d(quantity)/dCL over that range. It has
  no ``alpha_deg`` or ``CL`` list. For a derivative quantity,
  ``derivative_unit`` is the unit of the quantity (for example
  d(Cl_beta [1/deg])/dCL).
* Signs follow Ventorum: Cl > 0 right wing down, beta > 0 for wind from the
  right. A source in yaw angle psi = -beta stores the converted value and
  says so in the case text and in ``[quality] notes``.

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
It also refuses a derivative quantity without ``derivative_unit``, an
unknown derivative variable, a result with both ``alpha_deg`` and ``CL``, a
slope entry with more than one value or without ``CL_range``, and a
``CL_range`` with lo >= hi.
"""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ventorum.reference.derivatives import (
    DEFAULT_DERIVATIVE_AXES,
    DERIVATIVE_AXES,
    DERIVATIVE_COEFFICIENTS,
    DERIVATIVE_UNITS,
    DERIVATIVE_VARIABLES,
    is_derivative,
    split_derivative,
)

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
        One of ``CL``, ``CD``, ``CDi``, ``Cm``, ``CY``, ``Cl``, ``Cn``, or a
        derivative quantity ``<C>_<v>`` with ``<C>`` in ``CL``, ``CD``,
        ``CY``, ``Cl``, ``Cm``, ``Cn`` and ``<v>`` in ``alpha``, ``beta``.
    alpha_deg : list[float]
        Angles of attack [deg]. Empty when the result uses the ``CL``
        abscissa or is a slope entry.
    values : list[float]
        Reference values at ``alpha_deg`` (or at ``CL``; one value for a
        slope entry).
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
    derivative_unit : str or None
        Unit of the stored values of a derivative quantity: ``"1/deg"`` or
        ``"1/rad"``. For a slope entry it is the unit of the quantity, for
        example d(Cl_beta [1/deg])/dCL. None for other quantities.
    axes : str or None
        Axes of a moment derivative: ``"body"``, ``"stability"`` (default)
        or ``"wind"``. None for quantities that are not derivatives.
    CL : list[float] or None
        Lift coefficients of the points (the abscissa instead of
        ``alpha_deg``). None when the result uses ``alpha_deg``.
    slope_of : str or None
        ``"CL"`` for a slope entry: the single value is the slope
        d(quantity)/dCL over ``CL_range``. None otherwise.
    CL_range : list[float] or None
        Lift range [lo, hi] of a slope entry (lo < hi).
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
    derivative_unit: str | None = None
    axes: str | None = None
    CL: list[float] | None = None
    slope_of: str | None = None
    CL_range: list[float] | None = None


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
    derivative = is_derivative(quantity)
    if quantity not in QUANTITIES and not derivative:
        parts = split_derivative(quantity)
        if parts is not None and parts[0] in DERIVATIVE_COEFFICIENTS:
            raise _fail(
                name,
                f"{where}.quantity",
                f"unknown variable {parts[1]!r} in quantity {quantity!r} "
                f"(allowed: {', '.join(DERIVATIVE_VARIABLES)})",
            )
        raise _fail(name, f"{where}.quantity", f"unknown quantity {quantity!r}")
    derivative_unit = None
    axes = None
    if derivative:
        derivative_unit = _required(data, "derivative_unit", where, name)
        if derivative_unit not in DERIVATIVE_UNITS:
            raise _fail(
                name,
                f"{where}.derivative_unit",
                f"must be one of {list(DERIVATIVE_UNITS)}, got {derivative_unit!r}",
            )
        axes = data.get("axes", DEFAULT_DERIVATIVE_AXES)
        if axes not in DERIVATIVE_AXES:
            raise _fail(name, f"{where}.axes", f"must be one of {list(DERIVATIVE_AXES)}, got {axes!r}")
    else:
        for key in ("derivative_unit", "axes"):
            if key in data and data[key] is not None:
                raise _fail(name, f"{where}.{key}", "is only for derivative quantities <C>_<v>")
    values = _number_list(_required(data, "values", where, name), f"{where}.values", name)
    has_alpha = "alpha_deg" in data and data["alpha_deg"] is not None
    has_cl = "CL" in data and data["CL"] is not None
    alpha: list[float] = []
    cl_list = None
    slope_of = None
    cl_range = None
    if "slope_of" in data and data["slope_of"] is not None:
        slope_of = data["slope_of"]
        if slope_of != "CL":
            raise _fail(name, f"{where}.slope_of", f"must be 'CL', got {slope_of!r}")
        if has_alpha or has_cl:
            raise _fail(
                name,
                where,
                "a slope entry ('slope_of') takes no 'alpha_deg' or 'CL' list: it uses 'CL_range'",
            )
        cl_range = _number_list(_required(data, "CL_range", where, name), f"{where}.CL_range", name)
        if len(cl_range) != 2:
            raise _fail(name, f"{where}.CL_range", "must hold 2 values [lo, hi]")
        if cl_range[0] >= cl_range[1]:
            raise _fail(
                name, f"{where}.CL_range", f"lo must be < hi, got [{cl_range[0]:g}, {cl_range[1]:g}]"
            )
        if len(values) != 1:
            raise _fail(
                name,
                f"{where}.values",
                f"a slope entry holds exactly one value, got {len(values)} values",
            )
        n_points = 1
        points_key = "values"
    else:
        if "CL_range" in data and data["CL_range"] is not None:
            raise _fail(name, f"{where}.CL_range", "present without 'slope_of'")
        if has_alpha and has_cl:
            raise _fail(name, where, "give either 'alpha_deg' or 'CL', not both")
        if has_cl:
            cl_list = _number_list(data["CL"], f"{where}.CL", name)
            abscissa_key, abscissa = "CL", cl_list
        else:
            if not has_alpha:
                raise _fail(name, f"{where}.alpha_deg", "missing required field (give 'alpha_deg' or 'CL')")
            alpha = _number_list(data["alpha_deg"], f"{where}.alpha_deg", name)
            abscissa_key, abscissa = "alpha_deg", alpha
        if len(abscissa) != len(values):
            raise _fail(
                name,
                where,
                f"lists of different lengths: '{abscissa_key}' has {len(abscissa)} values, "
                f"'values' has {len(values)} values",
            )
        n_points = len(values)
        points_key = abscissa_key
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
    moment_quantity = quantity in MOMENT_QUANTITIES or (
        derivative and split_derivative(quantity)[0] in MOMENT_QUANTITIES
    )
    if moment_quantity:
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
        if len(h_m) != n_points:
            first = f"'{points_key}' has {n_points} values" if slope_of is None else "a slope entry has 1 value"
            raise _fail(
                name,
                where,
                f"lists of different lengths: {first}, 'h_m' has {len(h_m)} values",
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
        derivative_unit=derivative_unit,
        axes=axes,
        CL=cl_list,
        slope_of=slope_of,
        CL_range=cl_range,
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
        ``height_point`` is present without a ``h_m`` list. Also if a
        derivative quantity has no ``derivative_unit``, a derivative
        variable is unknown, a result has both ``alpha_deg`` and ``CL``, a
        slope entry has more than one value or no ``CL_range``, or
        ``CL_range`` has lo >= hi. The message names the file and the field.
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
        # The case format keeps one geometry table: one surface, or several with "surfaces".
        key = "aircraft" if "surfaces" in geometry else "wing"
        aircraft = build_aircraft_from_spec(**{key: geometry}, where="geometry")
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
