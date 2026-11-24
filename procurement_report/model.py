"""Core data types for the offline procurement report tool.

Conventions
-----------
* Every date is a whole calendar day.  Every interval is half-open:
  ``[start, end)``.  This includes the catalogue's ledger range, query ranges
  and summary-slice ranges, so adjacent intervals can be concatenated without
  off-by-one errors.
* Money is an integer number of cents (``amount_sum`` / ``amount``).  Python
  ints are arbitrary precision, so negative values and values above 2**63 are
  supported exactly.
* Missing ledger files are never treated as zero.  A readable ledger day is
  read all-or-nothing.
* A summary group that is absent means "no records" only when the slice is
  explicitly marked complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping


class ConfigError(ValueError):
    """The input catalogue or the query is structurally invalid."""


class DataError(RuntimeError):
    """A source selected by a plan could not be read completely."""


# ---------------------------------------------------------------------------
# Dimensions


def is_dimension_value(value: Any) -> bool:
    """Dimension values are non-null JSON scalars.

    ``bool`` is rejected deliberately: in JSON ``true`` equals 1 in Python's
    comparison rules and would make grouping/filtering ambiguous.
    """

    return isinstance(value, (str, int)) and not isinstance(value, bool)


@dataclass(frozen=True)
class DimensionSchema:
    """The fixed dimension dictionary of the single ledger version."""

    order: tuple[str, ...]
    domains: Mapping[str, frozenset[Any]] | None = None

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self.order)

    def value_allowed(self, dim: str, value: Any) -> bool:
        if not is_dimension_value(value):
            return False
        if self.domains is None:
            return True
        domain = self.domains.get(dim)
        return True if domain is None else value in domain


# ---------------------------------------------------------------------------
# Ledger


@dataclass(frozen=True)
class LedgerDay:
    day: date
    readable: bool
    status: str
    file: Path


@dataclass(frozen=True)
class LedgerRow:
    department: Any
    category: Any
    amount: int | None


# ---------------------------------------------------------------------------
# Closing summaries


@dataclass(frozen=True)
class SummaryGroup:
    """One aggregate group of a summary slice.

    ``values`` is aligned with ``SummarySlice.group_dims``.  Counts and sums are
    exact integers; the sum may be negative and may exceed 64 bits.
    """

    values: tuple[Any, ...]
    record_count: int
    amount_nonnull_count: int
    amount_sum: int


@dataclass(frozen=True)
class SummarySlice:
    slice_id: str
    file: Path
    start: date | None
    end: date | None
    group_dims: tuple[str, ...]
    ledger_version: str
    complete: bool
    digest: str
    valid: bool
    invalid_reason: str | None
    groups: tuple[SummaryGroup, ...] = ()

    def interval(self) -> tuple[date, date] | None:
        if self.start is None or self.end is None or self.start >= self.end:
            return None
        return self.start, self.end

    def touches(self, start: date, end: date) -> bool:
        """Whether any date this entry claims overlaps ``[start, end)``.

        Works even for structurally invalid entries where the interval is
        reversed, so diagnostics can still locate them.
        """

        if self.start is None or self.end is None:
            return False
        lo, hi = sorted((self.start, self.end))
        return lo < end and hi > start

    def compatible_with(self, query: "Query") -> bool:
        """Whether the slice is legal for ``query`` (after validity checks).

        Requirements from the task:

        * the slice's *whole* date interval lies inside the query; a slice
          that pokes out of either end cannot be used (it cannot be split);
        * every dimension touched by a filter is retained by the slice, even
          when the whitelist happens to contain the entire dictionary;
        * every requested output grouping dimension is retained.
        """

        span = self.interval()
        if not self.valid or span is None:
            return False
        start, end = span
        if start < query.start or end > query.end:
            return False
        dims = set(self.group_dims)
        if not set(query.filters).issubset(dims):
            return False
        if not set(query.group_by).issubset(dims):
            return False
        return True


# ---------------------------------------------------------------------------
# Dataset and query


@dataclass
class Dataset:
    data_dir: Path
    ledger_version: str
    dimensions: DimensionSchema
    ledger_start: date
    ledger_end: date
    ledger_days: dict[date, LedgerDay]
    summaries: list[SummarySlice]

    def ledger_day(self, day: date) -> LedgerDay | None:
        return self.ledger_days.get(day)


@dataclass(frozen=True)
class Query:
    start: date
    end: date
    filters: Mapping[str, frozenset[Any]] = field(default_factory=dict)
    group_by: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "filters": {
                dim: sorted(values, key=sort_value)
                for dim, values in sorted(self.filters.items())
            },
            "group_by": list(self.group_by),
        }


def sort_value(value: Any) -> tuple[str, str]:
    """Deterministic ordering for mixed str/int dimension values."""

    return (type(value).__name__, str(value))


def normalize_query(
    start: date | str,
    end: date | str,
    filters: Mapping[str, Any] | None = None,
    group_by: Iterable[str] | str = (),
) -> Query:
    from .dates import parse_day

    qs = parse_day(start, "query.start") if isinstance(start, str) else start
    qe = parse_day(end, "query.end") if isinstance(end, str) else end
    if not isinstance(qs, date) or not isinstance(qe, date):
        raise ConfigError("query start/end must be dates")
    if qs > qe:
        raise ConfigError("query.start must not be after query.end")

    normalized_filters: dict[str, frozenset[Any]] = {}
    for dim, raw in (filters or {}).items():
        if not isinstance(dim, str) or not dim:
            raise ConfigError("filter dimension names must be non-empty strings")
        if isinstance(raw, (str, int)) and not isinstance(raw, bool):
            values = [raw]
        else:
            try:
                values = list(raw)
            except TypeError as exc:
                raise ConfigError(
                    f"filter {dim!r} must be a value or an iterable of values"
                ) from exc
        for value in values:
            if not is_dimension_value(value):
                raise ConfigError(
                    f"filter {dim!r} contains unsupported value {value!r}; "
                    "dimension values must be non-null strings or integers"
                )
        normalized_filters[dim] = frozenset(values)

    if isinstance(group_by, str):
        group_dims = (group_by,) if group_by else ()
    else:
        group_dims = tuple(group_by)
    if any(not isinstance(d, str) or not d for d in group_dims):
        raise ConfigError("group_by entries must be non-empty dimension names")
    if len(set(group_dims)) != len(group_dims):
        raise ConfigError("group_by contains a duplicate dimension")

    return Query(qs, qe, normalized_filters, group_dims)
