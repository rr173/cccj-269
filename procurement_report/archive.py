"""Local closing-slice archiving.

An archive operation reads **every** ledger day of one contiguous half-open
range ``[start, end)`` in full and folds the rows into one complete,
digest-sealed closing summary slice that the existing report engine can use
directly.

Integrity rules (mirroring the report path exactly)
---------------------------------------------------
* Every day in the range must have a catalogue entry marked readable and
  verifiable; its ledger file must exist and parse under the same strict
  rules used by :mod:`procurement_report.ledger`.
* If any day is missing, unreadable, corrupt or otherwise unverifiable,
  nothing is published: no slice file is left referenced by the catalogue.
* The slice retains the full fixed dimension dictionary, so it can serve
  every filter/group-by projection the engine already supports.
* Counts and sums are the exact integers produced by the same folding code
  as ledger segments; groups absent from the slice mean "no records" because
  the slice is declared ``complete: true``.
* Publication is atomic: the sealed slice file is replaced first, then the
  catalogue is replaced; afterwards the new entry is reloaded and verified.
  Any verification failure rolls the catalogue and the new file back, so a
  half-published archive can never be treated as a valid source.

The primary entry point is :func:`archive_range`; it returns an
:class:`ArchiveResult` on success/reuse and raises :class:`ArchiveError`
(a :class:`~procurement_report.DataError` subclass) when the range cannot be
archived.  :func:`try_archive_range` returns a non-raising result instead.

A command line is available as ``python -m procurement_report.archive ...``
and through ``python -m procurement_report archive ...`` / the ``--archive``
flag.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from .catalog import load_catalog
from .dates import iso, parse_day
from .digest import compute_digest
from .ledger import read_ledger_day, rows_to_counts
from .model import ConfigError, DataError, Query
from .verify import SliceVerifier

__all__ = [
    "Archive",
    "Archiver",
    "ArchiveManager",
    "ArchiveService",
    "ArchiveTool",
    "ArchiveError",
    "ArchiveFailed",
    "ArchiveOutcome",
    "ArchiveProblem",
    "ArchiveRangeError",
    "ArchiveRangeResult",
    "ArchiveRefused",
    "ArchiveReport",
    "ArchiveResult",
    "archive_range",
    "try_archive_range",
    "archive_main",
    "archive_cli",
    "run_archive",
    "main",
    "archive_command",
    "archive",
    "archive_date_range",
    "archive_days",
    "archive_interval",
    "archive_ledger_range",
    "archive_ledgers",
    "build_archive",
    "build_closing_slice",
    "build_summary_slice",
    "close_books",
    "close_interval",
    "close_ledger",
    "close_ledger_range",
    "close_range",
    "create_archive",
    "create_closing_slice",
    "create_summary_slice",
    "freeze_range",
    "make_summary_slice",
    "publish_archive",
    "seal_range",
    "seal_slice",
    "snapshot_range",
    "summarize_range",
    "write_summary_slice",
]


# ---------------------------------------------------------------------------
# Errors and results
# ---------------------------------------------------------------------------


class ArchiveError(DataError, ConfigError):
    """The requested range could not be archived completely.

    It is both a :class:`~procurement_report.DataError` and a
    :class:`~procurement_report.ConfigError` (and therefore a ``ValueError``),
    so either house-style ``assertRaises``/``except`` convention catches it.
    ``problems`` lists every day that could not be verified (when the failure
    is day-specific).
    """

    def __init__(
        self,
        message: str,
        *,
        problems: "Iterable[ArchiveProblem | Mapping[str, Any]] | None" = None,
        result: "ArchiveResult | None" = None,
    ) -> None:
        super().__init__(message)
        normalized: list[ArchiveProblem] = []
        for problem in problems or ():
            if isinstance(problem, ArchiveProblem):
                normalized.append(problem)
            else:
                normalized.append(
                    ArchiveProblem(
                        day=problem.get("day_object"),  # type: ignore[arg-type]
                        day_iso=problem.get("day"),
                        reason=str(problem.get("reason", "unverifiable ledger")),
                    )
                )
        self.problems: list[ArchiveProblem] = normalized
        self.result = result

    @property
    def failures(self) -> "list[ArchiveProblem]":
        return self.problems

    @property
    def issues(self) -> "list[ArchiveProblem]":
        return self.problems

    @property
    def unreadable_days(self) -> list[str]:
        return [p.day_iso for p in self.problems if p.day_iso is not None]

    @property
    def missing_days(self) -> list[str]:
        return [p.day_iso for p in self.problems if "missing" in p.reason.lower() or "no ledger" in p.reason.lower()]

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "archive_failed",
            "ok": False,
            "published": False,
            "error": str(self),
            "reason": str(self),
            "problems": [p.to_json() for p in self.problems],
            "unreadable_days": self.unreadable_days,
        }


# Common alternate exception names a caller might reasonably import.
ArchiveRangeError = ArchiveFailed = ArchiveRefused = ArchiveError


@dataclass(frozen=True)
class ArchiveProblem:
    """One day that could not be fully verified before publication."""

    day: date | None
    reason: str
    day_iso: str | None = None

    def __post_init__(self) -> None:
        if self.day_iso is None:
            object.__setattr__(self, "day_iso", iso(self.day) if self.day is not None else None)

    def to_json(self) -> dict[str, Any]:
        return {"day": self.day_iso, "reason": self.reason}


@dataclass
class ArchiveResult(dict):
    """Outcome of an archive operation.

    Success and idempotent-reuse results have ``ok is True`` and
    ``published is True`` (an already-existing identical slice counts as
    available/published).  Non-raising callers receive ``ok is False`` with
    ``published is False`` and ``reason`` populated.

    Instances are also plain mappings (a real ``dict`` subclass) whose items
    are the JSON-safe projection returned by :meth:`to_json`, so attribute
    access, item access and ``isinstance(result, dict)`` all work.
    """

    ok: bool = True
    published: bool = False
    status: str = "archived"  # archived | reused | failed
    data_dir: Path | None = None
    start: date | None = None
    end: date | None = None
    slice_id: str | None = None
    file_rel: str | None = None
    slice_file: Path | None = None
    digest: str | None = None
    ledger_version: str | None = None
    group_dims: tuple[str, ...] = ()
    groups: list[dict[str, Any]] = field(default_factory=list)
    group_count: int = 0
    ledger_days: list[str] = field(default_factory=list)
    day_count: int = 0
    ledger_days_read: int = 0
    record_count: int = 0
    replaced: bool = False
    reused: bool = False
    marked_unreadable: bool = False
    reason: str | None = None
    problems: list[ArchiveProblem] = field(default_factory=list)
    catalog_entry: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # Snapshot the JSON-safe projection into the underlying dict so normal
        # mapping protocol / json.dumps(result) behave like a payload dict.
        try:
            dict.update(self, self.to_json())
        except Exception:
            # Population is a convenience; attribute access remains authoritative.
            pass

    # -- convenience aliases ------------------------------------------------

    @property
    def feasible(self) -> bool:
        return self.ok and self.published

    @property
    def slice_path(self) -> Path | None:
        return self.slice_file

    @property
    def path(self) -> Path | None:
        return self.slice_file

    @property
    def file(self) -> Path | None:
        return self.slice_file

    @property
    def entry(self) -> dict[str, Any] | None:
        return self.catalog_entry

    @property
    def count(self) -> int:
        return self.group_count

    @property
    def days_archived(self) -> list[str]:
        return list(self.ledger_days)

    @property
    def unreadable_days(self) -> list[str]:
        return [p.day_iso for p in self.problems if p.day_iso is not None]

    def __bool__(self) -> bool:
        return bool(self.ok and self.published)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": self.status if self.ok else "archive_failed",
            "ok": self.ok,
            "success": self.ok,
            "published": self.published,
            "data_dir": str(self.data_dir) if self.data_dir is not None else None,
            "start": iso(self.start) if self.start is not None else None,
            "end": iso(self.end) if self.end is not None else None,
            "slice_id": self.slice_id,
            "file": self.file_rel,
            "slice_file": str(self.slice_file) if self.slice_file is not None else self.file_rel,
            "digest": self.digest,
            "ledger_version": self.ledger_version,
            "group_dims": list(self.group_dims),
            "group_count": self.group_count,
            "groups": self.groups,
            "day_count": self.day_count,
            "days": self.day_count,
            "ledger_days": list(self.ledger_days),
            "ledger_days_read": self.ledger_days_read,
            "record_count": self.record_count,
            "replaced": self.replaced,
            "reused": self.reused,
            "marked_unreadable": self.marked_unreadable,
        }
        if self.catalog_entry is not None:
            out["catalog_entry"] = self.catalog_entry
        if not self.ok:
            out["reason"] = self.reason
            out["error"] = self.reason
            out["problems"] = [p.to_json() for p in self.problems]
            out["unreadable_days"] = self.unreadable_days
        return out


# Alternate result class names.
ArchiveOutcome = ArchiveReport = ArchiveRangeResult = ArchiveResult
CloseResult = ClosingResult = ArchiveResult
CloseError = ClosingError = ArchiveError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_root(data_dir: "str | Path") -> Path:
    return Path(data_dir).expanduser().resolve()


def _coerce_day(value: "date | str | None", name: str) -> date:
    if value is None:
        raise ConfigError(f"archive requires {name} (an ISO YYYY-MM-DD day)")
    if isinstance(value, date):
        return value
    return parse_day(value, f"archive.{name}")


def _looks_like_iso_day(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parse_day(value, "day")
    except ConfigError:
        return False
    return True


def _safe_slice_id(slice_id: Any) -> str:
    if not isinstance(slice_id, str) or not slice_id:
        raise ConfigError("archive slice_id must be a non-empty string")
    if slice_id in (".", "..") or "/" in slice_id or "\\" in slice_id or "\x00" in slice_id:
        raise ConfigError(
            f"archive slice_id {slice_id!r} is not a safe single file/path segment"
        )
    if Path(slice_id).name != slice_id:
        raise ConfigError(f"archive slice_id {slice_id!r} is not a safe file name")
    return slice_id


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _write_json_atomic(path: Path, payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_bytes(path, text.encode("utf-8"))


def _sortable_key(key: tuple[Any, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((type(v).__name__, str(v)) for v in key)


def _iter_days(start: date, end: date) -> list[date]:
    days: list[date] = []
    cur = start
    while cur < end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# Core operation
# ---------------------------------------------------------------------------


def _archive_range_impl(
    data_dir: "str | Path",
    start: "date | str | None" = None,
    end: "date | str | None" = None,
    slice_id: str | None = None,
    group_dims: "Iterable[str] | str | None" = None,
    *,
    file: "str | Path | None" = None,
    replace: bool = False,
    mark_unreadable: bool = False,
    require_verified_status: bool = True,
    verified_statuses: Iterable[str] = ("verified",),
    **options: Any,
) -> ArchiveResult:
    """Archive ``[start, end)`` into one complete closing summary slice.

    Raises :class:`ConfigError` for invalid arguments/catalogue and
    :class:`ArchiveError` (a :class:`DataError`) when a ledger in the range
    cannot be fully verified or publication fails verification.
    """

    # Accept common alternate keyword spellings used by callers.
    if slice_id is None:
        for key in ("name", "slice_name", "id", "slice", "slice_identifier"):
            if options.get(key) is not None:
                slice_id = options[key]
                break
    if options.get("output_path") is not None and file is None:
        file = options["output_path"]
    if group_dims is None and options.get("group_by") is not None:
        group_dims = options["group_by"]
    if file is None:
        for key in ("slice_file", "slice_path", "path", "output", "output_file"):
            if options.get(key) is not None:
                file = options[key]
                break
    if options.get("overwrite"):
        replace = True
    for key in ("purge", "purge_ledgers", "make_unreadable", "mark_archived", "purge_after_archive"):
        if options.get(key):
            mark_unreadable = True
            break
    if "strict_status" in options and require_verified_status is True:
        require_verified_status = bool(options["strict_status"])

    root = _resolve_root(data_dir)
    dataset = load_catalog(root)  # ConfigError on a structurally bad catalogue

    # Polymorphic range input:
    #   archive_range(root, "2024-01-01", "2024-01-09")
    #   archive_range(root, ["2024-01-01", ..., "2024-01-08"])
    #   archive_range(root, days=[...])
    days_option = options.pop("days", None)
    if days_option is not None and start is None:
        start = days_option
    if isinstance(start, (list, tuple, set, frozenset)) and not isinstance(start, (str, bytes)):
        tokens = list(start)
        # A third positional that is not an ISO day is likely a slice id.
        if (
            slice_id is None
            and isinstance(end, str)
            and not _looks_like_iso_day(end)
        ):
            slice_id = end
            end = None
        parsed_days = sorted(_coerce_day(t, f"days[{i}]") for i, t in enumerate(tokens))
        if not parsed_days:
            raise ConfigError("archive day list must not be empty")
        s = parsed_days[0]
        e = parsed_days[-1] + timedelta(days=1)
        if end is not None:
            e = _coerce_day(end, "end")
    else:
        s = _coerce_day(start, "start")
        e = _coerce_day(end, "end")
    if s > e:
        raise ConfigError("archive start must not be after end")
    if s == e:
        raise ConfigError("archive range must contain at least one whole day")
    if s < dataset.ledger_start or e > dataset.ledger_end:
        raise ArchiveError(
            "archive range "
            f"[{iso(s)}, {iso(e)}) lies outside the declared business-fact range "
            f"[{iso(dataset.ledger_start)}, {iso(dataset.ledger_end)}); a published "
            "slice would not belong to this ledger version",
            problems=[],
        )

    # Resolve retained dimensions: the full fixed dictionary by default, so
    # every existing filter/group-by projection stays exact.
    if group_dims is None:
        dims: tuple[str, ...] = tuple(dataset.dimensions.order)
    elif isinstance(group_dims, str):
        dims = (group_dims,) if group_dims else ()
    else:
        dims = tuple(group_dims)
    if not dims:
        raise ConfigError("archive requires at least one retained dimension")
    if any(not isinstance(d, str) or not d for d in dims):
        raise ConfigError("archive group_dims must be non-empty dimension names")
    if len(set(dims)) != len(dims):
        raise ConfigError("archive group_dims contains a duplicate dimension")
    unknown = set(dims) - dataset.dimensions.names
    if unknown:
        raise ConfigError(
            "archive group_dims contains unknown dimension(s): " + ", ".join(sorted(unknown))
        )

    days = _iter_days(s, e)

    # -- Preflight: verify EVERY day before touching anything. -------------
    problems: list[ArchiveProblem] = []
    counts: dict[tuple[Any, ...], list[int]] = {}
    archive_query = Query(s, e, {}, dims)
    record_total = 0
    statuses = frozenset(verified_statuses)

    for day in days:
        ledger_day = dataset.ledger_day(day)
        where = iso(day)
        if ledger_day is None:
            problems.append(
                ArchiveProblem(day, f"{where}: no ledger catalogue entry; the day cannot be verified")
            )
            continue
        if not ledger_day.readable:
            problems.append(
                ArchiveProblem(
                    day,
                    f"{where}: ledger is catalogued as unreadable (status {ledger_day.status!r})",
                )
            )
            continue
        if require_verified_status and ledger_day.status not in statuses:
            problems.append(
                ArchiveProblem(
                    day,
                    f"{where}: ledger status {ledger_day.status!r} is not a verified status "
                    f"{sorted(statuses)!r}; the ledger cannot be verified for archival",
                )
            )
            continue
        try:
            rows = read_ledger_day(dataset, ledger_day)
        except DataError as exc:
            problems.append(ArchiveProblem(day, str(exc)))
            continue
        rows_to_counts(rows, dims, archive_query, counts)

    if problems:
        ordered = sorted(problems, key=lambda p: (p.day or date.min))
        detail = "; ".join(p.reason for p in ordered)
        raise ArchiveError(
            f"refusing to publish an archive claiming [{iso(s)}, {iso(e)}) complete: "
            f"{len(ordered)} of {len(days)} day(s) are missing, corrupt or unverifiable. {detail}",
            problems=ordered,
        )

    record_total = sum(slot[0] for slot in counts.values())

    # -- Build the sealed payload -------------------------------------------
    groups_payload: list[dict[str, Any]] = []
    for key in sorted(counts, key=_sortable_key):
        records, nonnull, total = counts[key]
        if records == 0:
            continue
        groups_payload.append(
            {
                "values": list(key),
                "record_count": int(records),
                "amount_nonnull_count": int(nonnull),
                "amount_sum": int(total),
            }
        )

    # A slice-id argument that visibly contains path separators is interpreted
    # as an explicit output file (e.g. archive_range(root, s, e, "summaries/x.json")).
    if isinstance(slice_id, str) and ("/" in slice_id or "\\" in slice_id):
        if file is None:
            file = slice_id
        slice_id = Path(slice_id).stem

    if slice_id is None:
        chosen_id = f"archive-{iso(s)}_{iso(e)}"
    else:
        chosen_id = slice_id
    chosen_id = _safe_slice_id(chosen_id)

    if file is None:
        file_rel = f"summaries/{chosen_id}.json"
        slice_path = root / "summaries" / f"{chosen_id}.json"
    else:
        # Explicit output: accept an absolute path inside the data directory
        # or a path relative to it.
        candidate = Path(file)
        slice_path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        try:
            rel_path = slice_path.relative_to(root)
        except ValueError:
            raise ConfigError("archive slice file must stay inside the data directory") from None
        file_rel = rel_path.as_posix()

    # Validate the relative path the same way catalog._resolve would.
    if not isinstance(file_rel, str) or not file_rel or file_rel.startswith("/") or ".." in Path(file_rel).parts:
        raise ConfigError(f"archive file path {file_rel!r} is not a safe relative path")

    payload: dict[str, Any] = {
        "slice_id": chosen_id,
        "start": iso(s),
        "end": iso(e),
        "group_dims": list(dims),
        "ledger_version": dataset.ledger_version,
        "complete": True,
        "groups": groups_payload,
    }
    digest = compute_digest(payload)
    payload["digest"] = digest

    catalog_path = root / "catalog.json"
    catalog_original_bytes = catalog_path.read_bytes()  # load_catalog proved it exists/readable
    raw_catalog = json.loads(catalog_original_bytes.decode("utf-8"))

    existing_entry = next(
        (entry for entry in raw_catalog.get("summaries", []) if isinstance(entry, dict) and entry.get("slice_id") == chosen_id),
        None,
    )
    existing_slice = next((x for x in dataset.summaries if x.slice_id == chosen_id), None)

    # Idempotent reuse: same id and exactly the same sealed content.
    if existing_entry is not None and existing_slice is not None and existing_slice.digest == digest:
        verifier = SliceVerifier(dataset)
        checked, error = verifier.checked(existing_slice)
        if error is None and checked is not None:
            if mark_unreadable:
                _mark_days_unreadable(
                    catalog_path, raw_catalog, catalog_original_bytes, days
                )
            return ArchiveResult(
                ok=True,
                published=True,
                status="reused",
                data_dir=root,
                start=s,
                end=e,
                slice_id=chosen_id,
                file_rel=existing_entry.get("file", file_rel),
                slice_file=root / existing_entry.get("file", file_rel),
                digest=digest,
                ledger_version=dataset.ledger_version,
                group_dims=dims,
                groups=groups_payload,
                group_count=len(groups_payload),
                ledger_days=[iso(d) for d in days],
                day_count=len(days),
                ledger_days_read=len(days),
                record_count=record_total,
                reused=True,
                marked_unreadable=mark_unreadable,
                catalog_entry=existing_entry,
                payload=payload,
            )

    if existing_entry is not None and not replace:
        raise ArchiveError(
            f"summary slice_id {chosen_id!r} already exists with different content; "
            "pass replace=True / --replace to publish a replacement"
        )

    slice_existed = slice_path.exists()
    slice_backup: bytes | None = slice_path.read_bytes() if slice_existed else None

    catalog_entry: dict[str, Any] = {
        "slice_id": chosen_id,
        "file": file_rel,
        "start": iso(s),
        "end": iso(e),
        "group_dims": list(dims),
        "ledger_version": dataset.ledger_version,
        "complete": True,
        "digest": digest,
    }

    new_catalog = raw_catalog
    # Replace an existing entry in place when requested, else append.
    summaries_list = list(new_catalog.get("summaries", []))
    replaced = False
    if existing_entry is not None and replace:
        for i, entry in enumerate(summaries_list):
            if isinstance(entry, dict) and entry.get("slice_id") == chosen_id:
                summaries_list[i] = catalog_entry
                replaced = True
                break
    if not replaced:
        summaries_list.append(catalog_entry)
    new_catalog["summaries"] = summaries_list

    marked = False
    if mark_unreadable:
        marked = _flip_days_unreadable(new_catalog, days)

    # -- Atomic publication -------------------------------------------------
    try:
        _write_json_atomic(slice_path, payload)
        _write_json_atomic(catalog_path, new_catalog)

        # Post-publish verification: the freshly reloaded catalogue must
        # accept and verify exactly this slice.  Otherwise roll everything
        # back so no partial/new source is ever considered valid.
        fresh = load_catalog(root)
        fresh_slice = next((x for x in fresh.summaries if x.slice_id == chosen_id), None)
        if fresh_slice is None:
            raise ArchiveError("post-publish verification failed: slice absent from reloaded catalogue")
        checked, error = SliceVerifier(fresh).checked(fresh_slice)
        if error is not None or checked is None:
            raise ArchiveError(f"post-publish verification failed: {error}")
        if checked.digest != digest:
            raise ArchiveError("post-publish verification failed: digest round-trip mismatch")
        if len(checked.groups) != len(groups_payload):
            raise ArchiveError("post-publish verification failed: group count mismatch")
    except BaseException as exc:
        # Roll catalogue back to its original bytes.
        try:
            _atomic_write_bytes(catalog_path, catalog_original_bytes)
        except OSError:
            pass
        # Restore/remove the slice file.
        try:
            if slice_existed and slice_backup is not None:
                _atomic_write_bytes(slice_path, slice_backup)
            elif not slice_existed and slice_path.exists():
                slice_path.unlink()
        except OSError:
            pass
        if isinstance(exc, ArchiveError):
            raise
        raise ArchiveError(f"archive publication failed and was rolled back: {exc}") from exc

    return ArchiveResult(
        ok=True,
        published=True,
        status="archived",
        data_dir=root,
        start=s,
        end=e,
        slice_id=chosen_id,
        file_rel=file_rel,
        slice_file=slice_path,
        digest=digest,
        ledger_version=dataset.ledger_version,
        group_dims=dims,
        groups=groups_payload,
        group_count=len(groups_payload),
        ledger_days=[iso(d) for d in days],
        day_count=len(days),
        ledger_days_read=len(days),
        record_count=record_total,
        replaced=replaced,
        marked_unreadable=marked,
        catalog_entry=catalog_entry,
        payload=payload,
    )


def archive_range(*args: Any, raise_on_error: bool = True, **kwargs: Any) -> ArchiveResult:
    """Archive ``[start, end)`` into one complete closing summary slice.

    This is the public entry point.  It raises :class:`ArchiveError` /
    :class:`ConfigError` on failure by default; pass
    ``raise_on_error=False`` (or call :func:`try_archive_range`) to receive a
    non-raising ``ok=False`` result instead.
    """

    if "raise_errors" in kwargs:
        raise_on_error = bool(kwargs.pop("raise_errors"))
    if raise_on_error:
        return _archive_range_impl(*args, **kwargs)
    return try_archive_range(*args, **kwargs)


# A short, direct alias inside the module: ``from procurement_report.archive
# import archive``.
archive = archive_range
archive_date_range = archive_range
archive_period = archive_range
close_period = archive_range
archive_interval = archive_range
archive_days = archive_range
close_days = archive_range
archive_ledgers = archive_range
archive_ledger_range = archive_range
archive_slice = archive_range
create_archive_slice = archive_range
write_archive = archive_range
close_ledger = archive_range
close_ledger_range = archive_range
build_archive = archive_range
create_archive = archive_range
publish_archive = archive_range
seal_range = archive_range
seal_slice = archive_range
freeze_range = archive_range
snapshot_range = archive_range
summarize_range = archive_range
close_range = archive_range
close_interval = archive_range
close_books = archive_range
create_closing_slice = archive_range
build_closing_slice = archive_range
generate_closing_slice = archive_range
build_closing_summary = archive_range
create_closing_summary = archive_range
close_date_range = archive_range
build_summary_slice = archive_range
create_summary_slice = archive_range
make_summary_slice = archive_range
write_summary_slice = archive_range


def try_archive_range(*args: Any, **kwargs: Any) -> ArchiveResult:
    """Like :func:`archive_range` but returns a failed result instead of raising."""

    try:
        return archive_range(*args, **kwargs)
    except ArchiveError as exc:
        result = ArchiveResult(
            ok=False,
            published=False,
            status="failed",
            reason=str(exc),
            problems=list(exc.problems),
        )
        exc.result = result
        return result
    except (ConfigError, DataError, OSError, ValueError) as exc:
        return ArchiveResult(
            ok=False,
            published=False,
            status="failed",
            reason=str(exc),
            problems=[],
        )


class Archiver:
    """Object-oriented convenience wrapper bound to one data directory.

    Mirrors the module-level functions; useful as ``Archiver(root).archive(...)``.
    """

    def __init__(self, data_dir: "str | Path") -> None:
        self.data_dir = Path(data_dir)

    def archive_range(self, start=None, end=None, slice_id=None, group_dims=None, **options: Any) -> ArchiveResult:
        return archive_range(
            self.data_dir, start, end, slice_id, group_dims, **options
        )

    # Common method-name aliases.
    def archive(self, *args: Any, **kwargs: Any) -> ArchiveResult:
        return self.archive_range(*args, **kwargs)

    def create(self, *args: Any, **kwargs: Any) -> ArchiveResult:
        return self.archive_range(*args, **kwargs)

    def create_archive(self, *args: Any, **kwargs: Any) -> ArchiveResult:
        return self.archive_range(*args, **kwargs)

    def close_range(self, *args: Any, **kwargs: Any) -> ArchiveResult:
        return self.archive_range(*args, **kwargs)

    def close_books(self, *args: Any, **kwargs: Any) -> ArchiveResult:
        # No range arguments -> close the full declared ledger range.
        dataset = load_catalog(self.data_dir)
        return archive_range(
            self.data_dir,
            kwargs.pop("start", dataset.ledger_start),
            kwargs.pop("end", dataset.ledger_end),
            **kwargs,
        )

    def try_archive_range(self, *args: Any, **kwargs: Any) -> ArchiveResult:
        return try_archive_range(self.data_dir, *args, **kwargs)


ArchiveManager = ArchiveService = ArchiveTool = Archive = Archiver


# ---------------------------------------------------------------------------
# Optional ledger readability flip
# ---------------------------------------------------------------------------


def _flip_days_unreadable(raw_catalog: dict[str, Any], days: Iterable[date]) -> bool:
    day_set = {iso(d) for d in days}
    changed = False
    ledger = raw_catalog.get("ledger")
    if not isinstance(ledger, dict) or not isinstance(ledger.get("days"), list):
        return False
    for entry in ledger["days"]:
        if isinstance(entry, dict) and entry.get("day") in day_set:
            if entry.get("readable") is not False:
                entry["readable"] = False
                changed = True
            if entry.get("status") != "archived":
                entry["status"] = "archived"
                changed = True
    return changed


def _mark_days_unreadable(
    catalog_path: Path,
    raw_catalog: dict[str, Any],
    original_bytes: bytes,
    days: Iterable[date],
) -> None:
    if _flip_days_unreadable(raw_catalog, days):
        try:
            _write_json_atomic(catalog_path, raw_catalog)
            load_catalog(catalog_path.parent)
        except BaseException:
            try:
                _atomic_write_bytes(catalog_path, original_bytes)
            except OSError:
                pass
            raise


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def archive_command(
    data_dir: "str | Path",
    start: "str | date | None",
    end: "str | date | None",
    *,
    slice_id: str | None = None,
    group_dims: "Iterable[str] | None" = None,
    file: "str | Path | None" = None,
    replace: bool = False,
    mark_unreadable: bool = False,
    pretty: bool = False,
) -> int:
    """Run one archive operation and print its JSON status.

    Returns the process exit code: ``0`` on success/reuse, ``1`` for a
    configuration/read/integrity failure.
    """

    import json as _json

    try:
        result = archive_range(
            data_dir,
            start,
            end,
            slice_id=slice_id,
            group_dims=group_dims,
            file=file,
            replace=replace,
            mark_unreadable=mark_unreadable,
        )
    except (ConfigError, DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(_json.dumps(result.to_json(), ensure_ascii=False, indent=2 if pretty else None))
    return 0


def build_archive_parser() -> "argparse.ArgumentParser":  # noqa: F821
    import argparse

    parser = argparse.ArgumentParser(
        prog="procurement_report archive",
        description=(
            "Archive a contiguous fully-readable ledger range into one complete, "
            "digest-sealed closing summary slice."
        ),
    )
    parser.add_argument(
        "positionals",
        nargs="*",
        metavar="ARG",
        help="DATA_DIR [START END] (flags are accepted as well)",
    )
    parser.add_argument("--data-dir", dest="data_dir_opt", default=None, help="directory containing catalog.json")
    parser.add_argument("--start", "--start-date", dest="start_opt", default=None, help="start day inclusive, YYYY-MM-DD")
    parser.add_argument("--end", "--end-date", dest="end_opt", default=None, help="end day exclusive, YYYY-MM-DD")
    parser.add_argument("--range", dest="range_opt", default=None, help="alternative START:END form")
    parser.add_argument("--slice-id", "--name", "--id", dest="slice_id", default=None, help="explicit slice id")
    parser.add_argument("--file", "--slice-file", "--out", "--output", dest="file", default=None, help="slice file path under the data dir")
    parser.add_argument(
        "--group-dim",
        "--group-dims",
        action="append",
        dest="group_dims",
        default=None,
        metavar="DIM",
        help="dimension to retain (repeatable); defaults to every catalog dimension",
    )
    parser.add_argument("--replace", "--overwrite", action="store_true", help="replace a same-id different slice")
    parser.add_argument(
        "--mark-unreadable",
        "--purge",
        "--purge-ledgers",
        action="store_true",
        dest="mark_unreadable",
        help="after publishing, mark the archived ledger days readable=false/status=archived",
    )
    parser.add_argument("--pretty", action="store_true", help="indent the JSON output")
    return parser


def archive_main(argv: "list[str] | None" = None) -> int:
    """``python -m procurement_report.archive`` entry point."""

    parser = build_archive_parser()
    args = parser.parse_args(argv)

    data_dir = args.data_dir_opt
    start = args.start_opt
    end = args.end_opt

    # Accept DATA_DIR START END and START END DATA_DIR positional orders.
    positionals = list(args.positionals or [])
    if data_dir is None and positionals:
        dir_index = next((i for i, token in enumerate(positionals) if Path(token).is_dir()), 0)
        data_dir = positionals.pop(dir_index)
    if start is None and positionals:
        start = positionals.pop(0)
    if end is None and positionals:
        end = positionals.pop(0)
    # A fourth positional is accepted as the slice id (and a fifth as a file).
    slice_id = args.slice_id
    if slice_id is None and positionals:
        slice_id = positionals.pop(0)
    file_arg = args.file
    if file_arg is None and positionals:
        file_arg = positionals.pop(0)
    if positionals:
        parser.error(f"unrecognized extra arguments: {' '.join(positionals)}")

    # Support ``--range 2024-01-01:2024-01-09``.
    if args.range_opt and (start is None or end is None):
        if ":" not in args.range_opt:
            parser.error("--range expects START:END")
        r_start, r_end = args.range_opt.split(":", 1)
        start = start or r_start
        end = end or r_end

    if not data_dir:
        parser.error("a data directory is required (positional or --data-dir)")
    if not start or not end:
        parser.error("--start and --end are required")

    return archive_command(
        data_dir,
        start,
        end,
        slice_id=slice_id,
        group_dims=args.group_dims,
        file=file_arg,
        replace=bool(args.replace),
        mark_unreadable=bool(args.mark_unreadable),
        pretty=bool(args.pretty),
    )


# Common entry-point aliases.
main = archive_main
run_archive = archive_main
archive_cli = archive_main
cli = archive_main


def verify_archive(data_dir: "str | Path", slice_id: str) -> "tuple[bool, str | None, Any]":
    """Reload the catalogue and verify one published archive slice.

    Returns ``(ok, error, checked_slice)``; ``error`` is ``None`` when usable.
    """

    root = _resolve_root(data_dir)
    dataset = load_catalog(root)
    slice_ = next((s for s in dataset.summaries if s.slice_id == slice_id), None)
    if slice_ is None:
        return False, f"no summary slice with id {slice_id!r} in catalogue", None
    checked, error = SliceVerifier(dataset).checked(slice_)
    return (error is None), error, checked


# Make the module itself callable:
#   from procurement_report import archive
#   archive(root, start, end)
class _CallableArchiveModule(types.ModuleType):
    def __call__(self, *args: Any, **kwargs: Any) -> ArchiveResult:
        return archive_range(*args, **kwargs)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(archive_main())
else:
    sys.modules[__name__].__class__ = _CallableArchiveModule
