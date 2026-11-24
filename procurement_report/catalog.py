"""Read and validate the fixed catalogue (``catalog.json``).

Catalogue parsing never touches a ledger file and never reads a summary file.
Summary-file verification is performed by :mod:`procurement_report.verify`
only for slices that could take part in a query.

Failure containment
-------------------
* A catalogue entry that cannot even be identified (no usable ``slice_id``) is
  a configuration error.
* An identifiable slice that violates a rule (wrong ledger version, unknown
  grouping dimension, reversed interval, not declared complete, ...) is kept
  with ``valid=False`` and a reason.  It is never used.
* A missing, corrupt, duplicated or digest-mismatched summary *file* marks
  that slice unusable without failing the whole run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dates import parse_day
from .model import (
    ConfigError,
    Dataset,
    DimensionSchema,
    LedgerDay,
    SummarySlice,
    is_dimension_value,
)

_DAY_KEYS = {"day", "readable", "status", "file"}
_SLICE_KEYS = {
    "slice_id",
    "file",
    "start",
    "end",
    "group_dims",
    "ledger_version",
    "complete",
    "digest",
}


def _fail(where: str, message: str) -> None:
    raise ConfigError(f"{where}: {message}")


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(where, "expected a JSON object")
    return value


def _require_keys(
    obj: dict[str, Any], required: set[str], optional: set[str], where: str
) -> None:
    missing = required - obj.keys()
    extra = obj.keys() - required - optional
    if missing:
        _fail(where, f"missing key(s): {', '.join(sorted(missing))}")
    if extra:
        _fail(where, f"unknown key(s): {', '.join(sorted(extra))}")


def _resolve(root: Path, relative: Any, where: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail(where, "file must be a non-empty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        _fail(where, "file escapes the data directory")
    return candidate


def _read_json(path: Path, where: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _fail(where, f"cannot read {path.name}: {exc.strerror or exc}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        _fail(where, f"invalid JSON in {path.name}: {exc.msg} (line {exc.lineno})")


def load_catalog(data_dir: str | Path) -> Dataset:
    root = Path(data_dir).resolve()
    catalog_path = root / "catalog.json"
    if not catalog_path.is_file():
        raise ConfigError(f"{catalog_path}: missing catalogue")
    raw = _object(_read_json(catalog_path, "catalog.json"), "catalog.json")
    _require_keys(
        raw, {"ledger_version", "dimensions", "ledger", "summaries"}, set(), "catalog.json"
    )

    ledger_version = raw["ledger_version"]
    if not isinstance(ledger_version, str) or not ledger_version:
        _fail("catalog.json.ledger_version", "must be a non-empty string")

    dimensions = _parse_dimensions(raw["dimensions"], "catalog.json.dimensions")
    ledger_start, ledger_end, days = _parse_ledger(
        root, _object(raw["ledger"], "catalog.json.ledger")
    )

    entries = raw["summaries"]
    if not isinstance(entries, list):
        _fail("catalog.json.summaries", "must be a list")
    slices: list[SummarySlice] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(entries):
        where = f"catalog.json.summaries[{i}]"
        slice_obj = _parse_summary_entry(root, entry, ledger_version, dimensions, where)
        if slice_obj.slice_id in seen_ids:
            _fail(where, f"duplicate summary slice_id {slice_obj.slice_id!r}")
        seen_ids.add(slice_obj.slice_id)
        slices.append(slice_obj)

    return Dataset(
        data_dir=root,
        ledger_version=ledger_version,
        dimensions=dimensions,
        ledger_start=ledger_start,
        ledger_end=ledger_end,
        ledger_days=days,
        summaries=slices,
    )


def _parse_dimensions(raw: Any, where: str) -> DimensionSchema:
    obj = _object(raw, where)
    _require_keys(obj, {"order"}, {"domains"}, where)
    order = obj["order"]
    if (
        not isinstance(order, list)
        or not order
        or any(not isinstance(d, str) or not d for d in order)
        or len(set(order)) != len(order)
    ):
        _fail(f"{where}.order", "must be a non-empty list of unique dimension names")

    domains_raw = obj.get("domains")
    domains: dict[str, frozenset[Any]] | None = None
    if domains_raw is not None:
        domains_obj = _object(domains_raw, f"{where}.domains")
        unknown = set(domains_obj) - set(order)
        if unknown:
            _fail(f"{where}.domains", f"unknown dimension(s): {', '.join(sorted(unknown))}")
        domains = {}
        for dim, values in domains_obj.items():
            if not isinstance(values, list):
                _fail(f"{where}.domains.{dim}", "must be a list")
            seen: set[Any] = set()
            for value in values:
                if not is_dimension_value(value):
                    _fail(
                        f"{where}.domains.{dim}",
                        "values must be non-null strings or integers",
                    )
                if value in seen:
                    _fail(f"{where}.domains.{dim}", f"duplicate value {value!r}")
                seen.add(value)
            domains[dim] = frozenset(seen)
    return DimensionSchema(tuple(order), domains)


def _parse_ledger(
    root: Path, obj: dict[str, Any]
) -> tuple[Any, Any, dict[Any, LedgerDay]]:
    _require_keys(obj, {"start", "end", "days"}, set(), "catalog.json.ledger")
    start = parse_day(obj["start"], "catalog.json.ledger.start")
    end = parse_day(obj["end"], "catalog.json.ledger.end")
    if start > end:
        _fail("catalog.json.ledger", "start must not be after end")
    raw_days = obj["days"]
    if not isinstance(raw_days, list):
        _fail("catalog.json.ledger.days", "must be a list")

    days: dict[Any, LedgerDay] = {}
    for i, entry in enumerate(raw_days):
        where = f"catalog.json.ledger.days[{i}]"
        item = _object(entry, where)
        missing = _DAY_KEYS - item.keys()
        if missing:
            _fail(where, f"missing key(s): {', '.join(sorted(missing))}")
        day = parse_day(item["day"], f"{where}.day")
        if day < start or day >= end:
            _fail(where, f"{item['day']} lies outside the half-open ledger range [start, end)")
        if day in days:
            _fail(where, f"duplicate ledger day {item['day']}")
        readable = item["readable"]
        if not isinstance(readable, bool):
            _fail(f"{where}.readable", "must be a boolean")
        status = item["status"]
        if not isinstance(status, str) or not status:
            _fail(f"{where}.status", "must be a non-empty string")
        days[day] = LedgerDay(day, readable, status, _resolve(root, item["file"], f"{where}.file"))
    return start, end, days


def _parse_summary_entry(
    root: Path,
    entry: Any,
    ledger_version: str,
    dimensions: DimensionSchema,
    where: str,
) -> SummarySlice:
    """Parse one catalogue entry, degrading slice-level problems to invalid.

    Only an entry that cannot even be identified (not an object, no usable
    ``slice_id``) is a catalogue error.  Everything else -- a malformed file
    path, a missing field, an unknown extra field, a bad date, a wrong ledger
    version -- marks just this slice unusable, so one bad row cannot stop the
    report.
    """

    obj = _object(entry, where)

    slice_id = obj.get("slice_id")
    if not isinstance(slice_id, str) or not slice_id:
        _fail(f"{where}.slice_id", "must be a non-empty string")

    # Best-effort identity/position, used for diagnostics even when invalid.
    try:
        file = _resolve(root, obj.get("file"), f"{where}.file")
    except ConfigError:
        file = root

    start: Any = None
    end: Any = None
    try:
        start = parse_day(obj.get("start"), f"{where}.start")
        end = parse_day(obj.get("end"), f"{where}.end")
    except ConfigError:
        pass

    raw_dims = obj.get("group_dims")
    group_dims: tuple[str, ...] = ()
    if isinstance(raw_dims, list) and all(isinstance(d, str) and d for d in raw_dims):
        group_dims = tuple(d for d in dimensions.order if d in set(raw_dims))

    version = obj.get("ledger_version") if isinstance(obj.get("ledger_version"), str) else ""
    complete = obj.get("complete") is True
    digest = obj.get("digest") if isinstance(obj.get("digest"), str) else ""

    def invalid(reason: str) -> SummarySlice:
        return SummarySlice(
            slice_id=slice_id,
            file=file,
            start=start,
            end=end,
            group_dims=group_dims,
            ledger_version=version,
            complete=complete,
            digest=digest,
            valid=False,
            invalid_reason=reason,
        )

    missing = _SLICE_KEYS - obj.keys()
    if missing:
        return invalid(f"missing catalogue key(s): {', '.join(sorted(missing))}")
    extra = obj.keys() - _SLICE_KEYS
    if extra:
        return invalid(f"unknown catalogue key(s): {', '.join(sorted(extra))}")

    if not isinstance(obj["file"], str) or not obj["file"]:
        return invalid("file must be a non-empty relative path")
    if start is None or end is None:
        return invalid("invalid start/end date")
    if start >= end:
        return invalid("slice interval must contain at least one whole day")

    if not isinstance(raw_dims, list) or any(not isinstance(d, str) or not d for d in raw_dims):
        return invalid("group_dims must be a list of non-empty dimension names")
    if len(raw_dims) != len(set(raw_dims)):
        return invalid("group_dims contains a duplicate dimension")
    unknown = set(raw_dims) - dimensions.names
    if unknown:
        return invalid(f"unknown grouping dimension(s): {', '.join(sorted(unknown))}")

    if not isinstance(obj["ledger_version"], str) or not obj["ledger_version"]:
        return invalid("ledger_version must be a non-empty string")
    if obj["ledger_version"] != ledger_version:
        return invalid(
            f"ledger version {obj['ledger_version']!r} does not match fixed version {ledger_version!r}"
        )
    if not isinstance(obj["complete"], bool):
        return invalid("complete must be a boolean")
    if not complete:
        return invalid("slice is not declared complete for its whole date interval")
    if not isinstance(obj["digest"], str) or not digest.startswith("sha256:"):
        return invalid("digest must be a sha256:... string")

    return SummarySlice(
        slice_id=slice_id,
        file=file,
        start=start,
        end=end,
        group_dims=group_dims,
        ledger_version=obj["ledger_version"],
        complete=True,
        digest=digest,
        valid=True,
        invalid_reason=None,
    )
