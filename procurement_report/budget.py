"""Budget-vs-actual reconciliation over the offline procurement warehouse.

This module layers a second question on top of :mod:`procurement_report`: for
every department, category and *budget effective interval*, how much did we
actually spend, how much was budgeted, what is the difference, and are we
within budget?

Inputs
------
* the existing data directory (catalogue, ledgers, closing summaries), read
  with exactly the same integrity, filtering and exact-integer semantics as
  the plain report;
* a **budget rule file that lives outside the data directory**.  Each rule is
  ``(department, category, start, end, daily_limit)`` with a half-open
  ``[start, end)`` effective interval.

Rules for the same ``(department, category)`` pair must not overlap in dates.
A date not covered by any rule is **unbudgeted**: its budget is ``null``, not
zero.  An explicitly configured ``daily_limit`` of 0 is a real zero budget.

Segmentation
------------
The query interval is split at every budget effective boundary.  Each segment
is reconciled on its own:

* the actual amount is attributed by *effective day*.  A ledger day belongs to
  exactly the segment that contains it; a summary slice may only be used when
  its **whole** interval lies inside one segment for every group it reports;
* the segment budget is ``daily_limit * number_of_days`` (after clipping the
  segment to the query);
* ``actual > budget`` is over budget, otherwise within budget.

A summary slice that straddles a budget boundary cannot be split: its rows are
already aggregated, and neither day-count nor proportional proration is exact.
Such a slice is rejected, and if no other legal set of sources can cover the
query, the result is ``cannot_reconcile`` rather than a complete-looking table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping
from .catalog import load_catalog
from .dates import iso, parse_day
from .ledger import passes, read_ledger_day
from .model import (
    ConfigError,
    Dataset,
    Query,
    is_dimension_value,
    normalize_query,
    sort_value,
)
from .planner import LedgerSegment, Plan, Planner, SliceSegment
from .verify import SliceVerifier

_RULE_FILE_KEYS = {"rules"}
_RULE_KEYS = {"department", "category", "start", "end", "daily_limit"}
_OPTIONAL_RULE_KEYS = {"rule_id"}


# ---------------------------------------------------------------------------
# Budget rules


@dataclass(frozen=True)
class BudgetRule:
    department: Any
    category: Any
    start: date
    end: date
    daily_limit: int
    rule_id: str
    position: int

    def describe(self) -> str:
        return (
            f"{self.rule_id} (department={self.department!r}, category={self.category!r}, "
            f"[{iso(self.start)}, {iso(self.end)}), daily_limit={self.daily_limit})"
        )


@dataclass(frozen=True)
class BudgetBook:
    path: Path
    rules: tuple[BudgetRule, ...]
    by_pair: Mapping[tuple[Any, Any], tuple[BudgetRule, ...]]

    def boundaries(self, pair: tuple[Any, Any]) -> tuple[date, ...]:
        out: set[date] = set()
        for rule in self.by_pair.get(pair, ()):
            out.add(rule.start)
            out.add(rule.end)
        return tuple(sorted(out))


def _budget_fail(where: str, message: str) -> None:
    raise ConfigError(f"{where}: {message}")


def _budget_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _budget_fail(where, "expected a JSON object")
    return value


def _validate_rule_value(dataset: Dataset, dim: str, value: Any, where: str) -> None:
    if not is_dimension_value(value):
        _budget_fail(where, "must be a non-null string or integer")
    if not dataset.dimensions.value_allowed(dim, value):
        _budget_fail(where, f"{value!r} is not in the {dim} dimension dictionary")


def load_budget(path: str | Path, dataset: Dataset) -> BudgetBook:
    """Load and strictly validate the budget rule file.

    The file must live outside the business data directory: budget rules are a
    separate input, never part of the warehouse being reconciled.
    """

    resolved = Path(path).resolve()
    try:
        resolved.relative_to(dataset.data_dir)
    except ValueError:
        pass
    else:
        raise ConfigError(
            "budget rules must be supplied from outside the data directory "
            f"({dataset.data_dir}); got {resolved}"
        )

    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"budget file {resolved}: cannot read: {exc.strerror or exc}"
        ) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"budget file {resolved}: invalid JSON: {exc.msg} (line {exc.lineno})"
        ) from exc

    obj = _budget_object(raw, "budget")
    missing = _RULE_FILE_KEYS - obj.keys()
    if missing:
        _budget_fail("budget", f"missing key(s): {', '.join(sorted(missing))}")
    extra = obj.keys() - _RULE_FILE_KEYS
    if extra:
        _budget_fail("budget", f"unknown key(s): {', '.join(sorted(extra))}")
    entries = obj["rules"]
    if not isinstance(entries, list):
        _budget_fail("budget.rules", "must be a list")

    rules: list[BudgetRule] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(entries):
        where = f"budget.rules[{i}]"
        item = _budget_object(entry, where)
        missing = _RULE_KEYS - item.keys()
        if missing:
            _budget_fail(where, f"missing key(s): {', '.join(sorted(missing))}")
        extra = item.keys() - _RULE_KEYS - _OPTIONAL_RULE_KEYS
        if extra:
            _budget_fail(where, f"unknown key(s): {', '.join(sorted(extra))}")

        _validate_rule_value(dataset, "department", item["department"], f"{where}.department")
        _validate_rule_value(dataset, "category", item["category"], f"{where}.category")

        start = parse_day(item["start"], f"{where}.start")
        end = parse_day(item["end"], f"{where}.end")
        if start >= end:
            _budget_fail(where, "effective interval must contain at least one whole day")

        limit = item["daily_limit"]
        if not isinstance(limit, int) or isinstance(limit, bool):
            _budget_fail(f"{where}.daily_limit", "must be an integer number of cents")
        if limit < 0:
            _budget_fail(f"{where}.daily_limit", "must not be negative")

        rule_id = item.get("rule_id", f"rules[{i}]")
        if not isinstance(rule_id, str) or not rule_id:
            _budget_fail(f"{where}.rule_id", "must be a non-empty string")
        if rule_id in seen_ids:
            _budget_fail(f"{where}.rule_id", f"duplicate rule_id {rule_id!r}")
        seen_ids.add(rule_id)

        rules.append(
            BudgetRule(
                department=item["department"],
                category=item["category"],
                start=start,
                end=end,
                daily_limit=limit,
                rule_id=rule_id,
                position=i,
            )
        )

    grouped: dict[tuple[Any, Any], list[BudgetRule]] = {}
    for rule in rules:
        grouped.setdefault((rule.department, rule.category), []).append(rule)
    for pair, pair_rules in grouped.items():
        pair_rules.sort(key=lambda r: (r.start, r.end, r.position))
        for previous, current in zip(pair_rules, pair_rules[1:]):
            if current.start < previous.end:
                raise ConfigError(
                    "budget rules overlap for the same department and category "
                    f"({pair[0]!r}, {pair[1]!r}): {previous.describe()} and "
                    f"{current.describe()}"
                )

    return BudgetBook(
        path=resolved,
        rules=tuple(rules),
        by_pair={pair: tuple(items) for pair, items in grouped.items()},
    )


# ---------------------------------------------------------------------------
# Segments


@dataclass(frozen=True)
class BudgetSegment:
    department: Any
    category: Any
    start: date
    end: date
    rule: BudgetRule | None

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    @property
    def pair(self) -> tuple[Any, Any]:
        return self.department, self.category

    @property
    def budget_amount(self) -> int | None:
        if self.rule is None:
            return None
        return self.rule.daily_limit * self.days


def partition(
    book: BudgetBook, pair: tuple[Any, Any], start: date, end: date
) -> list[BudgetSegment]:
    """Split ``[start, end)`` for ``pair`` at every rule boundary.

    Returns configured segments (each carrying its rule) and unbudgeted gaps
    (``rule is None``).  Rules are clipped to the requested interval; an
    empty interval yields no segments.
    """

    segments: list[BudgetSegment] = []
    cursor = start
    for rule in book.by_pair.get(pair, ()):
        lo = max(rule.start, start)
        hi = min(rule.end, end)
        if lo >= hi:
            continue
        if lo > cursor:
            segments.append(BudgetSegment(pair[0], pair[1], cursor, lo, None))
        segments.append(BudgetSegment(pair[0], pair[1], lo, hi, rule))
        cursor = hi
    if cursor < end:
        segments.append(BudgetSegment(pair[0], pair[1], cursor, end, None))
    return segments


# ---------------------------------------------------------------------------
# Result


@dataclass
class BudgetResult:
    feasible: bool
    query: Query
    budget_rule_count: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    ledger_days_read: int = 0
    ledger_day_list: list[str] = field(default_factory=list)
    summary_slices_used: list[str] = field(default_factory=list)
    reachable_boundary: str | None = None
    last_covered_day: str | None = None
    failure_reason: str | None = None
    rejected_slices: list[dict[str, str]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": "ok" if self.feasible else "cannot_reconcile",
            "query": self.query.to_json(),
            "budget": {"rule_count": self.budget_rule_count},
            "ledger_days_read": self.ledger_days_read,
            "ledger_days": self.ledger_day_list,
            "summary_slices_used": self.summary_slices_used,
            "coverage": self.segments,
        }
        if self.feasible:
            out["rows"] = self.rows
            out["row_count"] = len(self.rows)
        else:
            out["reachable_boundary"] = self.reachable_boundary
            out["last_covered_day"] = self.last_covered_day
            out["failure_reason"] = self.failure_reason
        if self.rejected_slices:
            out["rejected_slices"] = self.rejected_slices
        return out


# ---------------------------------------------------------------------------
# Reconciliation


def reconcile_with_dataset(
    dataset: Dataset,
    start: date | str,
    end: date | str,
    budget: str | Path | BudgetBook,
    filters: Mapping[str, Any] | None = None,
) -> BudgetResult:
    book = budget if isinstance(budget, BudgetBook) else load_budget(budget, dataset)
    # The reconciliation is always keyed by department and category: budget
    # rules are defined on exactly this pair, and a slice that does not retain
    # both dimensions cannot be split into budget rows.
    query = normalize_query(start, end, filters, ("department", "category"))
    unsupported = set(query.filters) - {"department", "category"}
    if unsupported:
        raise ConfigError(
            "budget reconciliation only supports department/category filters; got: "
            + ", ".join(sorted(unsupported))
        )

    verifier = SliceVerifier(dataset)
    gate = _make_slice_gate(query, book)
    plan = Planner(dataset, query, verifier, gate=gate).build()

    result = BudgetResult(
        feasible=plan.feasible,
        query=query,
        budget_rule_count=len(book.rules),
        reachable_boundary=iso(plan.reachable_boundary) if plan.reachable_boundary else None,
        last_covered_day=iso(plan.last_covered_day) if plan.last_covered_day else None,
        failure_reason=plan.failure_reason,
        rejected_slices=[
            {"slice_id": slice_id, "reason": reason} for slice_id, reason in plan.rejected
        ],
    )

    if not plan.feasible:
        result.segments = [_coverage_entry(segment) for segment in plan.segments]
        if _has_budget_rejection(plan.rejected):
            note = (
                " Note: otherwise valid summary slices were excluded because their "
                "intervals straddle a budget effective-date boundary. Their amounts are "
                "aggregated across segments and are not split by day count or prorated, "
                "so they cannot fill this gap. The reconciliation is reported as "
                "incomplete rather than as a complete table."
            )
            result.failure_reason = (result.failure_reason or "") + note
        return result

    counts = _execute(dataset, query, book, plan, result)
    result.ledger_days_read = len(plan.ledger_days)
    result.rows = _build_budget_rows(query, book, counts)
    return result


def reconcile(
    data_dir: str | Path,
    start: date | str,
    end: date | str,
    budget: str | Path,
    filters: Mapping[str, Any] | None = None,
) -> BudgetResult:
    dataset = load_catalog(data_dir)
    return reconcile_with_dataset(dataset, start, end, budget, filters)


# -- slice gate -------------------------------------------------------------


def _make_slice_gate(query: Query, book: BudgetBook):
    """Reject any slice whose reported group straddles a budget boundary.

    The check is per group: a slice may span a boundary for a pair that has no
    records in it (the group is simply absent, contributing nothing), while a
    present group whose pair is cut by the boundary cannot be used because its
    single aggregate cannot be attributed to either side.
    """

    def gate(slice_) -> str | None:
        # ``query.group_by`` always includes both dimensions, so compatibility
        # already guarantees the slice retains them.
        index = {dim: i for i, dim in enumerate(slice_.group_dims)}
        for group in slice_.groups:
            if group.record_count == 0:
                # An explicit zero group in a complete slice contributes no
                # records and no amount, so its pair cannot need splitting.
                continue
            values = {
                "department": group.values[index["department"]],
                "category": group.values[index["category"]],
            }
            if not passes(values, query):
                continue
            pair = (values["department"], values["category"])
            for boundary in book.boundaries(pair):
                if slice_.start < boundary < slice_.end:
                    return (
                        f"slice interval [{iso(slice_.start)}, {iso(slice_.end)}) straddles "
                        f"the budget boundary {iso(boundary)} for department "
                        f"{pair[0]!r} category {pair[1]!r}; its aggregated amount cannot be "
                        "attributed to one effective interval and will not be prorated"
                    )
        return None

    return gate


def _has_budget_rejection(rejected: Iterable[tuple[str, str]]) -> bool:
    return any("straddles the budget boundary" in reason for _, reason in rejected)


# -- execution --------------------------------------------------------------


def _coverage_entry(segment: LedgerSegment | SliceSegment) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "source": segment.source,
        "start": iso(segment.start),
        "end": iso(segment.end),
        "days": segment.day_count,
    }
    if isinstance(segment, SliceSegment):
        entry["slice_id"] = segment.slice_.slice_id
    else:
        entry["ledger_days"] = [iso(day) for day in segment.days]
    return entry


def _add(counts: dict[tuple[Any, ...], list[int]], key, records, nonnull, total) -> None:
    slot = counts.get(key)
    if slot is None:
        counts[key] = [records, nonnull, total]
    else:
        slot[0] += records
        slot[1] += nonnull
        slot[2] += total


def _execute(
    dataset: Dataset,
    query: Query,
    book: BudgetBook,
    plan: Plan,
    result: BudgetResult,
) -> dict[tuple[Any, ...], list[int]]:
    counts: dict[tuple[Any, ...], list[int]] = {}
    partitions: dict[tuple[Any, Any], list[BudgetSegment]] = {}

    def partition_for(pair: tuple[Any, Any]) -> list[BudgetSegment]:
        cached = partitions.get(pair)
        if cached is None:
            cached = partition(book, pair, query.start, query.end)
            partitions[pair] = cached
        return cached

    def segment_containing(
        pair: tuple[Any, Any], lo: date, hi: date
    ) -> BudgetSegment:
        for segment in partition_for(pair):
            if segment.start <= lo and hi <= segment.end:
                return segment
        raise AssertionError(
            "internal error: source interval cannot be attributed to one budget segment"
        )

    for segment in plan.segments:
        entry = _coverage_entry(segment)
        result.segments.append(entry)
        if isinstance(segment, SliceSegment):
            slice_ = segment.slice_
            index = {dim: i for i, dim in enumerate(slice_.group_dims)}
            result.summary_slices_used.append(slice_.slice_id)
            for group in slice_.groups:
                if group.record_count == 0:
                    # A complete slice's explicit empty group has no records
                    # and no amount, so it needs no budget attribution.
                    continue
                values = {
                    "department": group.values[index["department"]],
                    "category": group.values[index["category"]],
                }
                if not passes(values, query):
                    continue
                pair = (values["department"], values["category"])
                contained = segment_containing(pair, segment.start, segment.end)
                _add(
                    counts,
                    (*pair, contained.start, contained.end),
                    group.record_count,
                    group.amount_nonnull_count,
                    group.amount_sum,
                )
        else:
            ledger_days: list[str] = []
            for day in segment.days:
                ledger_day = dataset.ledger_day(day)
                assert ledger_day is not None
                for row in read_ledger_day(dataset, ledger_day):
                    values = {"department": row.department, "category": row.category}
                    if not passes(values, query):
                        continue
                    pair = (row.department, row.category)
                    contained = segment_containing(pair, day, day + timedelta(days=1))
                    is_counted = 0 if row.amount is None else 1
                    _add(
                        counts,
                        (*pair, contained.start, contained.end),
                        1,
                        is_counted,
                        row.amount if row.amount is not None else 0,
                    )
                ledger_days.append(iso(day))
            entry["ledger_days"] = ledger_days
            result.ledger_day_list.extend(ledger_days)
    return counts


# -- rows -------------------------------------------------------------------


def _build_budget_rows(
    query: Query,
    book: BudgetBook,
    counts: Mapping[tuple[Any, ...], list[int]],
) -> list[dict[str, Any]]:
    pairs: set[tuple[Any, Any]] = set()
    for pair in book.by_pair:
        values = {"department": pair[0], "category": pair[1]}
        if passes(values, query):
            pairs.add(pair)
    for key in counts:
        pairs.add((key[0], key[1]))

    rows: list[dict[str, Any]] = []
    for pair in sorted(pairs, key=lambda p: (sort_value(p[0]), sort_value(p[1]))):
        for segment in partition(book, pair, query.start, query.end):
            records, nonnull, total = counts.get(
                (*pair, segment.start, segment.end), (0, 0, 0)
            )
            budget = segment.budget_amount
            if budget is None:
                status = "unbudgeted"
                difference = None
            else:
                difference = budget - total
                status = "over_budget" if total > budget else "within_budget"
            rows.append(
                {
                    "group": {"department": pair[0], "category": pair[1]},
                    "segment": {
                        "start": iso(segment.start),
                        "end": iso(segment.end),
                        "days": segment.days,
                    },
                    "rule_id": segment.rule.rule_id if segment.rule is not None else None,
                    "budget_daily_limit": (
                        segment.rule.daily_limit if segment.rule is not None else None
                    ),
                    "budget_amount": budget,
                    "actual_amount": total,
                    "record_count": records,
                    "amount_nonnull_count": nonnull,
                    "difference": difference,
                    "status": status,
                }
            )
    return rows
