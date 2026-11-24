"""Execute a plan and produce exact grouped results.

Only the sources named by the plan are read: ledger days chosen by the
planner, and summary files that were already verified.  No unselected ledger
day is ever opened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dates import iso
from .ledger import build_rows, merge_counts, passes, read_ledger_day, rows_to_counts
from .model import Dataset, Query, SummarySlice
from .planner import LedgerSegment, Plan, SliceSegment


@dataclass
class ReportResult:
    feasible: bool
    query: Query
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
            "status": "ok" if self.feasible else "cannot_produce",
            "query": self.query.to_json(),
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


def execute(dataset: Dataset, query: Query, plan: Plan) -> ReportResult:
    result = ReportResult(
        feasible=plan.feasible,
        query=query,
        reachable_boundary=iso(plan.reachable_boundary) if plan.reachable_boundary else None,
        last_covered_day=iso(plan.last_covered_day) if plan.last_covered_day else None,
        failure_reason=plan.failure_reason,
        rejected_slices=[
            {"slice_id": slice_id, "reason": reason} for slice_id, reason in plan.rejected
        ],
    )

    if not plan.feasible:
        # Report the legal partial cover for diagnosis only.  Nothing is
        # read: the caller asked whether the table can be produced, and the
        # answer is no.
        result.segments = [
            {
                "source": segment.source,
                "start": iso(segment.start),
                "end": iso(segment.end),
                "days": segment.day_count,
                **(
                    {"slice_id": segment.slice_.slice_id}
                    if isinstance(segment, SliceSegment)
                    else {"ledger_days": [iso(d) for d in segment.days]}
                ),
            }
            for segment in plan.segments
        ]
        return result

    counts: dict[tuple[Any, ...], list[int]] = {}
    group_dims = query.group_by

    for segment in plan.segments:
        entry: dict[str, Any] = {
            "source": segment.source,
            "start": iso(segment.start),
            "end": iso(segment.end),
            "days": segment.day_count,
        }
        if isinstance(segment, SliceSegment):
            entry["slice_id"] = segment.slice_.slice_id
            _fold_slice(segment.slice_, query, counts)
            result.summary_slices_used.append(segment.slice_.slice_id)
        elif isinstance(segment, LedgerSegment):
            day_strings: list[str] = []
            for day in segment.days:
                ledger_day = dataset.ledger_day(day)
                assert ledger_day is not None
                rows_to_counts(read_ledger_day(dataset, ledger_day), group_dims, query, counts)
                day_strings.append(iso(day))
            entry["ledger_days"] = day_strings
            result.ledger_day_list.extend(day_strings)
        result.segments.append(entry)

    result.ledger_days_read = len(plan.ledger_days)
    if plan.feasible:
        result.rows = build_rows(counts, group_dims)
    return result


def _fold_slice(
    slice_: SummarySlice,
    query: Query,
    counts: dict[tuple[Any, ...], list[int]],
) -> None:
    """Project a verified slice's groups onto the query's grouping dimensions.

    Planning guarantees the slice retains every filtered dimension and every
    requested grouping dimension, so filtering and re-grouping are exact.
    A group absent from a complete slice means no records and adds nothing.
    """

    slice_dims = slice_.group_dims
    group_dims = query.group_by
    index = {dim: i for i, dim in enumerate(slice_dims)}
    group_indices = [index[d] for d in group_dims]

    partial: dict[tuple[Any, ...], list[int]] = {}
    for group in slice_.groups:
        values = {dim: group.values[i] for i, dim in enumerate(slice_dims)}
        if not passes(values, query):
            continue
        key = tuple(group.values[i] for i in group_indices)
        slot = partial.get(key)
        if slot is None:
            slot = [0, 0, 0]
            partial[key] = slot
        slot[0] += group.record_count
        slot[1] += group.amount_nonnull_count
        slot[2] += group.amount_sum
    merge_counts(counts, partial)
