"""Coverage planning.

The query is a half-open interval of calendar days.  Each day must be served by
exactly one source:

* a **readable ledger day**, read whole; or
* a **verified summary slice** whose whole interval lies inside the query and
  inside the declared business-fact range, and which retains all filtered and
  requested grouping dimensions.

A slice is indivisible: choosing it covers its entire interval, and no day in
that interval may be served again by another slice or by the ledger.  Two
overlapping slices therefore can never both be chosen; we never subtract one
overlapping aggregate from another.

Modelling
---------
Let the query days be indexed ``0 .. n-1``.  A plan is a left-to-right walk of
a frontier.  At frontier ``i``:

* one readable ledger day moves the frontier to ``i + 1`` at a cost of one
  ledger read;
* a selected slice starting exactly at ``i`` moves it to ``j`` (its exclusive
  end index) at zero ledger cost.

The frontier construction is complete: any legal disjoint set of slices plus
ledger days covering ``[0, n)`` corresponds to exactly one such walk, and vice
versa.  This is why "pick a priority and then report missing days" is wrong and
is not what this module does.

Two dynamic programs are run, both on the verified candidate set:

1. **Reachability / farthest boundary** -- ``reach(i)`` is the farthest
   boundary reachable from ``i``.  If the query end is reached, a plan exists.
   Otherwise ``reach(0)`` is the requested farthest left-reachable date
   boundary and the reason for getting stuck is derived from it.
2. **Minimum ledger reads** -- ``cost(i)`` is the minimum number of ledger days
   that must be read to cover ``[i, n)``, with a deterministic plan as a
   tie-break.  Summary verification is free; no ledger is read to evaluate
   either DP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from .dates import iso
from .model import ConfigError, Dataset, Query, SummarySlice
from .verify import SliceVerifier


@dataclass(frozen=True)
class SliceSegment:
    source: str  # "summary"
    slice_: SummarySlice
    start: date
    end: date

    @property
    def day_count(self) -> int:
        return (self.end - self.start).days


@dataclass(frozen=True)
class LedgerSegment:
    source: str  # "ledger"
    start: date
    end: date  # exclusive
    days: tuple[date, ...]

    @property
    def day_count(self) -> int:
        return len(self.days)


Segment = SliceSegment | LedgerSegment


@dataclass
class Plan:
    feasible: bool
    segments: list[Segment] = field(default_factory=list)
    ledger_days: list[date] = field(default_factory=list)
    slices: list[SummarySlice] = field(default_factory=list)
    reachable_boundary: date | None = None
    last_covered_day: date | None = None
    failure_reason: str | None = None
    rejected: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ledger_days_read(self) -> int:
        return len(self.ledger_days)


class Planner:
    def __init__(self, dataset: Dataset, query: Query, verifier: SliceVerifier):
        self.dataset = dataset
        self.query = query
        self.verifier = verifier
        self._validate_query_dims()

        self._n = (query.end - query.start).days
        self._exclusive: dict[str, int] = {}
        self._slices_at: dict[int, list[SummarySlice]] = {}
        self._readable: list[bool] = [False] * self._n
        self.rejected: list[tuple[str, str]] = []

        for i in range(self._n):
            day = query.start + timedelta(days=i)
            ledger_day = dataset.ledger_day(day)
            # Only a day the trustworthy catalogue marks readable can serve
            # the query from the ledger.  A missing catalogue entry means the
            # day is outside the declared business-fact range, not "zero".
            self._readable[i] = bool(ledger_day and ledger_day.readable)

        self._collect_candidates()

    # -- setup -------------------------------------------------------------

    def _validate_query_dims(self) -> None:
        known = self.dataset.dimensions.names
        unknown_filters = set(self.query.filters) - known
        if unknown_filters:
            raise ConfigError(
                "query filters unknown dimension(s): " + ", ".join(sorted(unknown_filters))
            )
        unknown_groups = set(self.query.group_by) - known
        if unknown_groups:
            raise ConfigError(
                "query groups by unknown dimension(s): " + ", ".join(sorted(unknown_groups))
            )

    def _collect_candidates(self) -> None:
        for slice_ in self.dataset.summaries:
            if not slice_.valid:
                # Report only slices that touch the query; an unrelated broken
                # slice elsewhere in the warehouse is not relevant noise.
                if slice_.touches(self.query.start, self.query.end):
                    self.rejected.append(
                        (slice_.slice_id, slice_.invalid_reason or "invalid catalogue entry")
                    )
                continue
            if not slice_.compatible_with(self.query):
                continue
            # The declared business-fact range is the universe the fixed
            # ledger version describes.  A slice claiming dates outside it
            # cannot belong to that version, so it is unusable rather than
            # trusted.
            span = slice_.interval()
            if span is None:  # pragma: no cover - compatible_with excludes this
                continue
            if span[0] < self.dataset.ledger_start or span[1] > self.dataset.ledger_end:
                self.rejected.append(
                    (
                        slice_.slice_id,
                        "slice interval lies outside the declared business-fact range "
                        f"[{iso(self.dataset.ledger_start)}, {iso(self.dataset.ledger_end)})",
                    )
                )
                continue
            checked, error = self.verifier.checked(slice_)
            if error is not None:
                self.rejected.append((slice_.slice_id, error))
                continue
            start_index = (checked.start - self.query.start).days  # type: ignore[operator]
            self._exclusive[checked.slice_id] = (checked.end - self.query.start).days  # type: ignore[operator]
            self._slices_at.setdefault(start_index, []).append(checked)

        for group in self._slices_at.values():
            group.sort(key=lambda s: (s.slice_id,))

    def verified_slices(self) -> dict[str, SummarySlice]:
        return {
            s.slice_id: s
            for slices in self._slices_at.values()
            for s in slices
        }

    # -- dynamic programs --------------------------------------------------

    def _reach_table(self) -> list[int]:
        m = self._n
        reach = list(range(m + 1))
        for i in range(m - 1, -1, -1):
            best = i
            if self._readable[i]:
                best = max(best, reach[i + 1])
            for slice_ in self._slices_at.get(i, ()):
                best = max(best, reach[self._exclusive[slice_.slice_id]])
            reach[i] = best
        return reach

    def _farthest_prefix(self, reach: list[int]) -> tuple[list[Segment], list[date], list[SummarySlice]]:
        """A legal prefix that reaches the farthest boundary.

        Used only for diagnosing an infeasible query; it is not a report.
        """

        by_id = self.verified_slices()
        entries: list[tuple[str, Any, int]] = []
        i = 0
        target = reach[0]
        while i < target:
            if self._readable[i] and reach[i + 1] == target:
                entries.append(("L", i, i + 1))
                i += 1
                continue
            advanced = False
            for slice_ in self._slices_at.get(i, ()):
                j = self._exclusive[slice_.slice_id]
                if reach[j] == target:
                    entries.append(("S", slice_.slice_id, j))
                    i = j
                    advanced = True
                    break
            if not advanced:  # pragma: no cover - reach table guarantees a step
                break
        materialised = self._materialise(
            tuple((kind, payload) for kind, payload, _ in entries), by_id
        )
        return materialised

    def build(self) -> Plan:
        if self._n == 0:
            return Plan(
                feasible=True,
                reachable_boundary=self.query.end,
                last_covered_day=None,
                rejected=self.rejected,
            )

        reach = self._reach_table()
        reached = reach[0]
        if reached < self._n:
            boundary = self.query.start + timedelta(days=reached)
            segments, ledger_days, slices = self._farthest_prefix(reach)
            return Plan(
                feasible=False,
                segments=segments,
                ledger_days=ledger_days,
                slices=slices,
                reachable_boundary=boundary,
                last_covered_day=boundary - timedelta(days=1) if reached > 0 else None,
                failure_reason=self._failure_reason(boundary),
                rejected=self.rejected,
            )

        plan_entries = self._minimum_reads()
        segments, ledger_days, slices = self._materialise(plan_entries)
        self._self_check(segments)
        return Plan(
            feasible=True,
            segments=segments,
            ledger_days=ledger_days,
            slices=slices,
            reachable_boundary=self.query.end,
            last_covered_day=self.query.end - timedelta(days=1),
            rejected=self.rejected,
        )

    def _self_check(self, segments: list[Segment]) -> None:
        """Fail loudly if the plan does not tile the query exactly once."""

        cursor = self.query.start
        for segment in segments:
            if segment.start != cursor:
                raise AssertionError(
                    f"internal error: coverage gap/overlap at {iso(segment.start)}"
                )
            if segment.end <= segment.start:
                raise AssertionError("internal error: empty coverage segment")
            if isinstance(segment, SliceSegment):
                if (segment.start, segment.end) != (
                    segment.slice_.start,
                    segment.slice_.end,
                ):
                    raise AssertionError("internal error: summary segment was truncated")
            else:
                expected = (segment.end - segment.start).days
                if len(segment.days) != expected:
                    raise AssertionError("internal error: ledger segment is not contiguous")
                for offset, day in enumerate(segment.days):
                    if day != segment.start + timedelta(days=offset):
                        raise AssertionError("internal error: ledger segment is not contiguous")
            cursor = segment.end
        if cursor != self.query.end:
            raise AssertionError("internal error: coverage does not reach the query end")

    def _minimum_reads(self) -> tuple[tuple[str, Any], ...]:
        """Minimise ledger days read, returning an ordered plan.

        ``dp[i]`` is the minimum number of ledger days needed to cover
        ``[i, n)``; ``choice[i]`` records how the optimum was reached.  Only
        costs and back-pointers are stored, so the DP is O(n) in memory.

        Ties are broken deterministically: prefer a summary slice over a
        ledger day, then the slice with the smallest ``slice_id``, then the
        shortest next step.  Any optimum is allowed; determinism only makes
        the output reproducible.
        """

        m = self._n
        INF: int | None = None
        cost: list[int | None] = [INF] * (m + 1)
        choice: list[tuple[str, Any, int] | None] = [None] * (m + 1)
        cost[m] = 0

        for i in range(m - 1, -1, -1):
            best_cost: int | None = None
            best_choice: tuple[str, Any, int] | None = None
            best_rank: tuple[Any, ...] | None = None

            if self._readable[i] and cost[i + 1] is not None:
                candidate_cost = cost[i + 1] + 1  # type: ignore[operator]
                rank = (1, "", 1)
                best_cost, best_choice, best_rank = candidate_cost, ("L", i, i + 1), rank

            for slice_ in self._slices_at.get(i, ()):
                j = self._exclusive[slice_.slice_id]
                if cost[j] is None:
                    continue
                candidate_cost = cost[j]  # type: ignore[assignment]
                rank = (0, slice_.slice_id, j - i)
                if (
                    best_cost is None
                    or candidate_cost < best_cost
                    or (candidate_cost == best_cost and best_rank is not None and rank < best_rank)
                ):
                    best_cost, best_choice, best_rank = candidate_cost, ("S", slice_.slice_id, j), rank

            if best_choice is not None:
                cost[i] = best_cost
                choice[i] = best_choice

        assert choice[0] is not None, "reachable end but no cost plan: inconsistent DP"

        entries: list[tuple[str, Any]] = []
        i = 0
        while i < m:
            step = choice[i]
            assert step is not None
            kind, payload, j = step
            entries.append((kind, payload))
            assert j > i
            i = j
        return tuple(entries)

    def _materialise(
        self,
        entries: tuple[tuple[str, Any], ...],
        by_id: dict[str, SummarySlice] | None = None,
    ) -> tuple[list[Segment], list[date], list[SummarySlice]]:
        by_id = by_id if by_id is not None else self.verified_slices()
        segments: list[Segment] = []
        ledger_days: list[date] = []
        slices: list[SummarySlice] = []

        pending_ledger: list[date] = []

        def flush_ledger() -> None:
            if not pending_ledger:
                return
            days = tuple(pending_ledger)
            segments.append(LedgerSegment("ledger", days[0], days[-1] + timedelta(days=1), days))
            pending_ledger.clear()

        for kind, payload in entries:
            if kind == "L":
                day = self.query.start + timedelta(days=payload)
                ledger_days.append(day)
                pending_ledger.append(day)
            else:
                flush_ledger()
                slice_ = by_id[payload]
                assert slice_.start is not None and slice_.end is not None
                segments.append(SliceSegment("summary", slice_, slice_.start, slice_.end))
                slices.append(slice_)
        flush_ledger()
        return segments, ledger_days, slices

    # -- failure explanation ----------------------------------------------

    def _failure_reason(self, boundary: date) -> str:
        """Explain the first day that cannot be covered.

        Every prefix of a legal cover is itself a legal partial cover, so the
        plan reported for an infeasible query is the *prefix* of some legal
        cover (empty when even the first day has no source).  At the boundary
        day no usable slice starts and the ledger is not readable.  A slice
        starting earlier cannot rescue the situation because its days are
        already served by the prefix; slices cannot be split and overlapping
        aggregates may not be netted.
        """

        if boundary == self.query.start:
            head = (
                "empty coverage reaches the left endpoint "
                f"{iso(self.query.start)}: no source can serve the first day"
            )
        else:
            head = (
                "the farthest prefix ends at the boundary "
                f"{iso(boundary)} (last covered day {iso(boundary - timedelta(days=1))})"
            )

        ledger_day = self.dataset.ledger_day(boundary)
        if ledger_day is None:
            ledger_why = (
                f"{iso(boundary)} has no ledger catalogue entry "
                f"(declared business-fact range is "
                f"[{iso(self.dataset.ledger_start)}, {iso(self.dataset.ledger_end)}))"
            )
        elif not ledger_day.readable:
            ledger_why = f"{iso(boundary)} has no readable ledger (status {ledger_day.status!r})"
        else:
            # Unreachable when this is truly the maximal boundary; kept as a
            # defensive explanation.
            ledger_why = f"{iso(boundary)} has a readable ledger but the prefix cannot advance"

        index = (boundary - self.query.start).days
        starting = self._slices_at.get(index, ())
        if starting:
            slice_why = (
                "usable compatible slices begin there ("
                + ", ".join(s.slice_id for s in starting)
                + ") but none can be chained to the query end"
            )
        else:
            slice_why = "no usable compatible summary slice begins there"

        return (
            f"{head}; cannot continue past it: {slice_why}, and {ledger_why}. "
            "A slice starting earlier cannot be reused: its days are already covered, "
            "slices cannot be split, and overlapping slices may not be subtracted."
        )
