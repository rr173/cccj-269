"""Local retention planning for daily ledgers.

A procurement warehouse eventually reaches the end of its retention period for
old per-day ledgers.  Before deleting anything we must be able to prove that
every report query in an agreed *future query workload* can still be produced
**exactly** from the sources that will remain: untouched daily ledgers plus
already-published, complete, legal and indivisible closing summary slices.

This module only *plans*.  It never deletes a ledger, never rewrites
``catalog.json`` and never touches a summary file.  A plan is a set of dates
that are safe to purge together with the dates that must be kept.

Semantics
---------
* Inputs: an existing data directory, a cutoff day ``before`` and a workload
  JSON file that lives **outside** the data directory.
* A daily ledger may be a cleanup *candidate* only when the catalogue marks it
  readable **and** its day is strictly before ``before`` (``day < before``).
* For every workload query we must exhibit a legal cover (the same coverage
  model as :mod:`procurement_report.planner`) that uses only:
  - verified, complete, compatible summary slices (indivisible, whole interval), and
  - daily ledgers that remain readable after the plan.
* Only if *all* workloads are proven does the plan become safe.  Otherwise the
  plan reports the earliest workload that cannot be proven and the date
  boundary at which its cover gets stuck.

Maximal cleaning, jointly
-------------------------
The objective is to clean as many candidate days as possible while a *single*
retained set keeps **every** workload provable.  Because feasibility only gets
harder as more days are cleaned, the search can treat the cleaned set as a
monotone constraint:

* a greedy reverse-delete sweeps candidate days in several deterministic
  orders, adding each to the cleaned set whenever every workload stays
  provable -- each sweep is inclusion-maximal (no skipped day could be added);
* for moderate candidate sets, branch and bound (seeded by those sweeps and
  bounded by a node budget) improves on them, proving maximum cardinality
  unless the budget runs out.

Every state visited is feasible, so the reported plan is always safe; the
budget can only cost *maximality*, never correctness.

Crucially this is a single global retained set verified against *all*
workloads together.  A day one workload can avoid is never cleaned if another
workload still needs it; we do not union per-workload "cleanable" sets.

Missing data is never treated as empty: a day that no slice covers and whose
ledger is not readable makes its workload unprovable, exactly as the report
engine would refuse to produce that query.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

from .catalog import load_catalog
from .dates import iso, parse_day
from .model import ConfigError, Dataset, Query, normalize_query
from .planner import Plan, Planner
from .verify import SliceVerifier

__all__ = [
    "CleanupWorkload",
    "WorkloadPlan",
    "CleanupPlanResult",
    "load_workloads",
    "plan_cleanup",
    "plan_cleanup_with_dataset",
    "cleanup_command",
    "cleanup_main",
    "build_cleanup_parser",
    "main",
]


# ---------------------------------------------------------------------------
# Workload file
# ---------------------------------------------------------------------------

#: Top-level object keys that may carry the list of future queries.  Exactly
#: one of them is expected; a bare JSON list is also accepted.
_WORKLOAD_CONTAINER_KEYS = ("queries", "workloads", "reports", "requests", "items")

_ENTRY_REQUIRED = {"start", "end"}
_ENTRY_OPTIONAL = {"query_id", "id", "name", "filters", "group_by"}


@dataclass(frozen=True)
class CleanupWorkload:
    """One future report query that must keep working after cleanup."""

    index: int
    workload_id: str
    query: Query

    def describe(self) -> str:
        return f"{self.workload_id} ({self.query.to_json()})"

    def to_json(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "index": self.index,
            "query": self.query.to_json(),
        }


def _workload_fail(where: str, message: str) -> None:
    raise ConfigError(f"{where}: {message}")


def _workload_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _workload_fail(where, "expected a JSON object")
    return value


def load_workloads(path: str | Path, dataset: Dataset) -> tuple[CleanupWorkload, ...]:
    """Load and strictly validate the query workload file.

    Like the budget rule file, the workload must live outside the data
    directory: it is an input to the retention decision, not part of the
    warehouse being reasoned about.
    """

    resolved = Path(path).resolve()
    try:
        resolved.relative_to(dataset.data_dir)
    except ValueError:
        pass
    else:
        raise ConfigError(
            "cleanup workload must be supplied from outside the data directory "
            f"({dataset.data_dir}); got {resolved}"
        )

    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"workload file {resolved}: cannot read: {exc.strerror or exc}"
        ) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"workload file {resolved}: invalid JSON: {exc.msg} (line {exc.lineno})"
        ) from exc

    if isinstance(raw, list):
        entries = raw
    else:
        obj = _workload_object(raw, "workload")
        containers = [key for key in _WORKLOAD_CONTAINER_KEYS if key in obj]
        if len(containers) != 1:
            _workload_fail(
                "workload",
                "expected exactly one of the keys "
                + ", ".join(_WORKLOAD_CONTAINER_KEYS)
                + "; found "
                + (", ".join(containers) if containers else "none"),
            )
        container = containers[0]
        extra = obj.keys() - {container}
        if extra:
            _workload_fail("workload", f"unknown key(s): {', '.join(sorted(extra))}")
        entries = obj[container]
        if not isinstance(entries, list):
            _workload_fail(f"workload.{container}", "must be a list")

    known_dims = dataset.dimensions.names
    workloads: list[CleanupWorkload] = []
    seen_ids: set[str] = set()

    for i, entry in enumerate(entries):
        where = f"workload.queries[{i}]"
        item = _workload_object(entry, where)
        missing = _ENTRY_REQUIRED - item.keys()
        if missing:
            _workload_fail(where, f"missing key(s): {', '.join(sorted(missing))}")
        extra = item.keys() - _ENTRY_REQUIRED - _ENTRY_OPTIONAL
        if extra:
            _workload_fail(where, f"unknown key(s): {', '.join(sorted(extra))}")

        raw_id = item.get("query_id", item.get("id", item.get("name")))
        if raw_id is None:
            workload_id = f"queries[{i}]"
        elif not isinstance(raw_id, str) or not raw_id:
            _workload_fail(f"{where}.query_id", "must be a non-empty string")
        else:
            workload_id = raw_id
        if workload_id in seen_ids:
            _workload_fail(where, f"duplicate query_id {workload_id!r}")
        seen_ids.add(workload_id)

        filters = item.get("filters") or {}
        if not isinstance(filters, Mapping):
            _workload_fail(f"{where}.filters", "must be a JSON object")
        group_by = item.get("group_by") or ()

        try:
            query = normalize_query(item["start"], item["end"], filters, group_by)
        except ConfigError as exc:
            raise ConfigError(f"{where}: {exc}") from exc

        unknown_filters = set(query.filters) - known_dims
        if unknown_filters:
            _workload_fail(
                where,
                "query filters unknown dimension(s): "
                + ", ".join(sorted(unknown_filters)),
            )
        unknown_groups = set(query.group_by) - known_dims
        if unknown_groups:
            _workload_fail(
                where,
                "query groups by unknown dimension(s): "
                + ", ".join(sorted(unknown_groups)),
            )

        workloads.append(CleanupWorkload(i, workload_id, query))

    return tuple(workloads)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class WorkloadPlan:
    """Proof state and chosen sources for one workload query."""

    workload: CleanupWorkload
    feasible: bool
    coverage: list[dict[str, Any]] = field(default_factory=list)
    ledger_days: list[date] = field(default_factory=list)
    candidate_days_used: list[date] = field(default_factory=list)
    summary_slices_used: list[str] = field(default_factory=list)
    reachable_boundary: date | None = None
    last_covered_day: date | None = None
    failure_reason: str | None = None
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "workload_id": self.workload.workload_id,
            "index": self.workload.index,
            "status": "ok" if self.feasible else "unprovable",
            "query": self.workload.query.to_json(),
        }
        if self.feasible:
            out["sources"] = self.coverage
            out["summary_slices_used"] = list(self.summary_slices_used)
            out["ledger_days"] = [iso(d) for d in self.ledger_days]
            out["ledger_days_read"] = len(self.ledger_days)
            # Candidate ledger days this workload's chosen cover still reads.
            # These are a subset of the plan's global must-retain set.
            out["candidate_ledger_days"] = [iso(d) for d in self.candidate_days_used]
        else:
            out["boundary"] = iso(self.reachable_boundary) if self.reachable_boundary else None
            out["reachable_boundary"] = out["boundary"]
            out["last_covered_day"] = (
                iso(self.last_covered_day) if self.last_covered_day else None
            )
            out["failure_reason"] = self.failure_reason
            out["reason"] = self.failure_reason
        if self.rejected:
            out["rejected_slices"] = [
                {"slice_id": slice_id, "reason": reason}
                for slice_id, reason in self.rejected
            ]
        return out


@dataclass
class CleanupPlanResult:
    feasible: bool
    before: date
    data_dir: Path | None
    candidate_days: list[date] = field(default_factory=list)
    cleanable_days: list[date] = field(default_factory=list)
    must_retain_days: list[date] = field(default_factory=list)
    workloads: list[WorkloadPlan] = field(default_factory=list)
    unprovable: WorkloadPlan | None = None
    workload_count: int = 0

    @property
    def safe(self) -> bool:
        return self.feasible

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": "ok" if self.feasible else "cannot_clean",
            "can_clean": self.feasible,
            "safe": self.feasible,
            "before": iso(self.before),
            "data_dir": str(self.data_dir) if self.data_dir is not None else None,
            "workload_count": self.workload_count,
            "candidate_dates": [iso(d) for d in self.candidate_days],
        }
        if self.feasible:
            out["cleanable_dates"] = [iso(d) for d in self.cleanable_days]
            out["must_retain_dates"] = [iso(d) for d in self.must_retain_days]
            out["cleanable_count"] = len(self.cleanable_days)
            out["must_retain_count"] = len(self.must_retain_days)
            out["workloads"] = [plan.to_json() for plan in self.workloads]
        else:
            proven = [plan.workload.workload_id for plan in self.workloads if plan.feasible]
            out["proven_workloads"] = proven
            if self.unprovable is not None:
                out["unprovable_workload"] = self.unprovable.to_json()
        return out


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _cleaned_dataset(dataset: Dataset, cleaned_days: Iterable[date]) -> Dataset:
    """A view of ``dataset`` with the given ledger days marked unavailable."""

    cleaned = set(cleaned_days)
    ledger_days = dict(dataset.ledger_days)
    for day in cleaned:
        ledger_day = ledger_days.get(day)
        if ledger_day is not None:
            ledger_days[day] = replace(ledger_day, readable=False, status="cleaned")
    return replace(dataset, ledger_days=ledger_days)


def _unprovable_plan(workload: CleanupWorkload, plan: Plan) -> WorkloadPlan:
    return WorkloadPlan(
        workload=workload,
        feasible=False,
        reachable_boundary=plan.reachable_boundary,
        last_covered_day=plan.last_covered_day,
        failure_reason=plan.failure_reason,
        rejected=list(plan.rejected),
    )


def _proof_plan(
    dataset: Dataset,
    workload: CleanupWorkload,
    candidate_set: set[date],
    verifier: SliceVerifier,
) -> WorkloadPlan:
    """Build the concrete, usable cover for one workload on ``dataset``."""

    plan = Planner(dataset, workload.query, verifier).build()
    if not plan.feasible:
        return _unprovable_plan(workload, plan)

    segments: list[dict[str, Any]] = []
    for segment in plan.segments:
        entry: dict[str, Any] = {
            "source": segment.source,
            "start": iso(segment.start),
            "end": iso(segment.end),
            "days": segment.day_count,
        }
        if segment.source == "summary":
            entry["slice_id"] = segment.slice_.slice_id  # type: ignore[attr-defined]
        else:
            entry["ledger_days"] = [iso(d) for d in segment.days]  # type: ignore[attr-defined]
        segments.append(entry)

    return WorkloadPlan(
        workload=workload,
        feasible=True,
        coverage=segments,
        ledger_days=list(plan.ledger_days),
        candidate_days_used=sorted(d for d in plan.ledger_days if d in candidate_set),
        summary_slices_used=[s.slice_id for s in plan.slices],
        rejected=list(plan.rejected),
    )


def _first_failure(
    dataset: Dataset,
    workloads: Iterable[CleanupWorkload],
    cleaned_days: set[date],
    verifier: SliceVerifier,
) -> WorkloadPlan | None:
    """The earliest workload that cannot be covered with ``cleaned_days`` removed.

    Returns ``None`` when every workload is provable.  ``verifier`` is reused
    across calls: slice verification depends only on slice content and the
    fixed dimension dictionary, never on ledger readability.
    """

    cleaned = _cleaned_dataset(dataset, cleaned_days)
    for workload in workloads:
        plan = Planner(cleaned, workload.query, verifier).build()
        if not plan.feasible:
            return _unprovable_plan(workload, plan)
    return None


def _all_provable(
    dataset: Dataset,
    workloads: Iterable[CleanupWorkload],
    cleaned_days: set[date],
    verifier: SliceVerifier,
) -> bool:
    return _first_failure(dataset, workloads, cleaned_days, verifier) is None


def _reverse_delete(
    dataset: Dataset,
    workloads: tuple[CleanupWorkload, ...],
    candidates: list[date],
    order: Iterable[date],
    verifier: SliceVerifier,
) -> set[date]:
    """Greedily clean days in ``order``, keeping every workload provable.

    Feasibility is downward-closed in the cleaned set, so the result is
    inclusion-maximal: no day that was skipped could be added to the final set.
    Used to seed the exact search with a strong, always-valid lower bound.
    """

    cleaned: set[date] = set()
    candidate_set = set(candidates)
    for day in order:
        if day not in candidate_set:  # pragma: no cover - orders come from candidates
            continue
        trial = cleaned | {day}
        if _all_provable(dataset, workloads, trial, verifier):
            cleaned.add(day)
    return cleaned


#: Branch and bound is attempted up to this many candidate days.  Beyond it we
#: rely only on several deterministic greedy passes (each inclusion-maximal,
#: hence never over-cleaning).  The node budget bounds the exact attempt.
_EXACT_MAX_CANDIDATES = 24
_EXACT_NODE_BUDGET = 200_000


def _maximise_cleanable(
    dataset: Dataset,
    workloads: tuple[CleanupWorkload, ...],
    candidates: list[date],
    verifier: SliceVerifier,
) -> set[date]:
    """Largest verified cleanable set among ``candidates``.

    Objective: maximise ``|cleaned|`` subject to *every* workload remaining
    provable on the cleaned catalogue.  This is one joint optimisation, never
    a per-workload union.

    * A greedy reverse-delete in several orders (always valid, always
      inclusion-maximal) seeds the incumbent.
    * For up to ``_EXACT_MAX_CANDIDATES`` days, branch and bound improves on
      it, proving maximal cardinality unless the node budget is exhausted.
    """

    if not candidates:
        return set()

    best: set[date] = set()
    orders = [
        list(candidates),
        list(reversed(candidates)),
        # Days appearing in fewer workload query intervals are cheaper to
        # clean; try them first so they are not blocked by later choices.
        sorted(candidates, key=lambda d: (_workload_span_count(d, workloads), d)),
    ]
    for order in orders:
        trial = _reverse_delete(dataset, workloads, candidates, order, verifier)
        if len(trial) > len(best):
            best = trial

    if len(candidates) <= _EXACT_MAX_CANDIDATES:
        best = _exact_max_cleanable(dataset, workloads, candidates, verifier, best)
    return best


def _workload_span_count(day: date, workloads: Iterable[CleanupWorkload]) -> int:
    return sum(1 for workload in workloads if workload.query.start <= day < workload.query.end)


def _exact_max_cleanable(
    dataset: Dataset,
    workloads: tuple[CleanupWorkload, ...],
    candidates: list[date],
    verifier: SliceVerifier,
    seed: set[date],
) -> set[date]:
    """Branch and bound over candidate days, maximising the cleaned count.

    ``seed`` is an already-feasible incumbent from the greedy passes.  Every
    visited state is feasible, so the returned set is always a valid plan even
    if the node budget cuts the search short.
    """

    best: set[date] = set(seed)
    best_size = len(best)
    n = len(candidates)
    nodes = 0

    def explore(index: int, cleaned: set[date]) -> None:
        nonlocal best, best_size, nodes
        nodes += 1
        if nodes > _EXACT_NODE_BUDGET:
            return

        if len(cleaned) > best_size:
            best, best_size = set(cleaned), len(cleaned)
        if len(cleaned) + (n - index) <= best_size:
            return
        if index >= n:
            return

        # Try cleaning the current day first: deeper cleaning is what we want.
        trial = cleaned | {candidates[index]}
        if _all_provable(dataset, workloads, trial, verifier):
            explore(index + 1, trial)
        explore(index + 1, cleaned)

    explore(0, set())
    return best


def plan_cleanup_with_dataset(
    dataset: Dataset,
    before: date | str,
    workload_path: str | Path,
) -> CleanupPlanResult:
    """Plan a safe cleanup for ``dataset``; see the module docstring."""

    before_day = before if isinstance(before, date) else parse_day(before, "cleanup.before")
    workloads = load_workloads(workload_path, dataset)

    candidate_days = sorted(
        day
        for day, ledger_day in dataset.ledger_days.items()
        if ledger_day.readable and day < before_day
    )
    candidate_set = set(candidate_days)

    # A ledger day outside every workload's query interval is never selected
    # by any of those queries, so cleaning it cannot affect feasibility.  Only
    # candidates that some workload could actually read take part in the
    # search; the rest are cleanable by construction.
    relevant = [
        day
        for day in candidate_days
        if any(workload.query.start <= day < workload.query.end for workload in workloads)
    ]
    relevant_set = set(relevant)
    always_cleanable = candidate_set - relevant_set

    # A slice file is verified once and reused across every trial: its content
    # and the fixed dimension dictionary never change here.
    verifier = SliceVerifier(dataset)

    # Nothing cleaned must already prove every workload; the first workload
    # that cannot even be produced from the current sources bounds the answer.
    failure = _first_failure(dataset, workloads, set(), verifier)
    if failure is not None:
        # Prove the workloads that already precede the failure so the report
        # can state exactly where the workload order first stops working.
        proven = [
            _proof_plan(dataset, workload, candidate_set, verifier)
            for workload in workloads[: failure.workload.index]
        ]
        return CleanupPlanResult(
            feasible=False,
            before=before_day,
            data_dir=dataset.data_dir,
            candidate_days=candidate_days,
            workloads=proven,
            unprovable=failure,
            workload_count=len(workloads),
        )

    # A single global retained set is optimised against every workload at
    # once.  Relevance-filtered candidate days left over are proved safe to
    # clean together, along with everything no workload can read.
    cleaned = _maximise_cleanable(dataset, workloads, relevant, verifier) | always_cleanable

    # Defensive: the incumbent is always feasible, but re-prove the final set
    # against all workloads as a whole before reporting it safe.
    failure = _first_failure(dataset, workloads, cleaned, verifier)
    if failure is not None:  # pragma: no cover - search only keeps feasible sets
        return CleanupPlanResult(
            feasible=False,
            before=before_day,
            data_dir=dataset.data_dir,
            candidate_days=candidate_days,
            unprovable=failure,
            workload_count=len(workloads),
        )

    cleanable = sorted(cleaned)
    retained = sorted(candidate_set - cleaned)

    # Report each workload's concrete sources against the final cleaned view.
    final_dataset = _cleaned_dataset(dataset, cleaned)
    workload_plans = [
        _proof_plan(final_dataset, workload, candidate_set, verifier)
        for workload in workloads
    ]

    return CleanupPlanResult(
        feasible=True,
        before=before_day,
        data_dir=dataset.data_dir,
        candidate_days=candidate_days,
        cleanable_days=cleanable,
        must_retain_days=retained,
        workloads=workload_plans,
        workload_count=len(workloads),
    )


def plan_cleanup(
    data_dir: str | Path,
    before: date | str,
    workload_path: str | Path,
) -> CleanupPlanResult:
    """Load the catalogue and plan a safe cleanup (never modifies anything)."""

    dataset = load_catalog(data_dir)
    return plan_cleanup_with_dataset(dataset, before, workload_path)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def cleanup_command(
    data_dir: str | Path,
    before: str | date | None,
    workload_path: str | Path | None,
    *,
    pretty: bool = False,
) -> int:
    """Run one cleanup plan and print its JSON.

    Exit status: ``0`` when a safe plan was produced, ``2`` when no safe plan
    exists (``cannot_clean``) and ``1`` for a configuration/read error.
    """

    if before is None:
        print("error: --before is required (an ISO YYYY-MM-DD day)", file=sys.stderr)
        return 1
    if workload_path is None:
        print(
            "error: --workload is required (a JSON file outside the data directory)",
            file=sys.stderr,
        )
        return 1

    try:
        result = plan_cleanup(data_dir, before, workload_path)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.to_json(), ensure_ascii=False, indent=2 if pretty else None))
    return 0 if result.feasible else 2


def build_cleanup_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        prog="procurement_report cleanup",
        description=(
            "Plan which daily ledgers before a cutoff may be purged while every "
            "future query in a workload file remains exactly producible. Never "
            "deletes or modifies anything."
        ),
    )
    parser.add_argument(
        "positionals",
        nargs="*",
        metavar="ARG",
        help="DATA_DIR (flags are accepted as well)",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir_opt",
        default=None,
        help="directory containing catalog.json",
    )
    parser.add_argument(
        "--before",
        "--cutoff",
        "--purge-before",
        dest="before",
        default=None,
        help="cleanup cutoff day (exclusive), YYYY-MM-DD",
    )
    parser.add_argument(
        "--workload",
        "--workloads",
        "--workload-file",
        "--queries",
        dest="workload",
        default=None,
        help="query workload JSON file outside the data directory",
    )
    parser.add_argument("--pretty", action="store_true", help="indent the JSON output")
    return parser


def cleanup_main(argv: "list[str] | None" = None) -> int:
    """``python -m procurement_report.cleanup`` entry point."""

    parser = build_cleanup_parser()
    args = parser.parse_args(argv)

    data_dir = args.data_dir_opt
    positionals = list(args.positionals or [])
    if data_dir is None and positionals:
        dir_index = next(
            (i for i, token in enumerate(positionals) if Path(token).is_dir()), 0
        )
        data_dir = positionals.pop(dir_index)
    if positionals:
        parser.error(f"unrecognized extra arguments: {' '.join(positionals)}")
    if not data_dir:
        parser.error("a data directory is required (positional or --data-dir)")

    return cleanup_command(
        data_dir,
        args.before,
        args.workload,
        pretty=bool(args.pretty),
    )


main = cleanup_main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cleanup_main())
