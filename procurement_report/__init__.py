"""The offline report tool: load, validate, plan, execute.

Public API::

    report = generate(data_dir, start, end, filters=..., group_by=[...])
    payload = report.to_json()

Planning never reads a ledger day, so evaluating every legal plan is free.
Only the chosen minimum-read plan's ledger days are opened during execution.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

from .budget import (
    BudgetBook,
    BudgetResult,
    BudgetRule,
    load_budget,
    reconcile,
    reconcile_with_dataset,
)
from .catalog import load_catalog
from .engine import ReportResult, execute
from .model import ConfigError, DataError, Dataset, Query, normalize_query
from .planner import Planner
from .verify import SliceVerifier

# Archive names are loaded lazily (PEP 562) so that
# ``python -m procurement_report.archive`` does not pre-import its own target
# through this package initialiser (which would emit a runpy warning).
_LAZY_ARCHIVE_NAMES = frozenset(
    {
        "Archive",
        "ArchiveError",
        "ArchiveFailed",
        "ArchiveManager",
        "ArchiveOutcome",
        "ArchiveProblem",
        "ArchiveRangeError",
        "ArchiveRangeResult",
        "ArchiveRefused",
        "ArchiveReport",
        "ArchiveResult",
        "ArchiveService",
        "ArchiveTool",
        "Archiver",
        "archive_command",
        "archive_date_range",
        "archive_days",
        "archive_interval",
        "archive_ledger_range",
        "archive_ledgers",
        "archive_main",
        "archive_range",
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
        "try_archive_range",
        "write_summary_slice",
    }
)

__all__ = [
    "Archive",
    "ArchiveError",
    "ArchiveManager",
    "ArchiveOutcome",
    "ArchiveProblem",
    "ArchiveRangeError",
    "ArchiveRangeResult",
    "ArchiveReport",
    "ArchiveResult",
    "Archiver",
    "BudgetBook",
    "BudgetResult",
    "BudgetRule",
    "ConfigError",
    "DataError",
    "Dataset",
    "Query",
    "ReportResult",
    "archive",
    "archive_date_range",
    "archive_days",
    "archive_interval",
    "archive_ledgers",
    "archive_main",
    "archive_range",
    "build_archive",
    "build_summary_slice",
    "close_books",
    "close_interval",
    "close_range",
    "create_archive",
    "create_closing_slice",
    "generate",
    "generate_with_dataset",
    "load_budget",
    "load_catalog",
    "normalize_query",
    "publish_archive",
    "reconcile",
    "reconcile_with_dataset",
    "try_archive_range",
    # Retention cleanup planning (populated lazily to keep
    # ``python -m procurement_report.cleanup`` free of runpy warnings).
    "CleanupWorkload",
    "WorkloadPlan",
    "CleanupPlanResult",
    "load_workloads",
    "plan_cleanup",
    "plan_cleanup_with_dataset",
    "cleanup_command",
    "cleanup_main",
]

#: Public names provided by :mod:`procurement_report.cleanup`.  They are
#: resolved lazily through ``__getattr__`` so that importing the package
#: never imports the cleanup module eagerly (which would emit a runpy warning
#: for ``python -m procurement_report.cleanup``).
_LAZY_CLEANUP_NAMES = frozenset(
    {
        "CleanupPlanResult",
        "CleanupWorkload",
        "WorkloadPlan",
        "cleanup_command",
        "cleanup_main",
        "load_workloads",
        "plan_cleanup",
        "plan_cleanup_with_dataset",
    }
)


def __getattr__(name: str) -> Any:
    if name == "archive":
        from importlib import import_module

        module = import_module(".archive", __name__)
        globals()["archive"] = module
        return module
    if name in _LAZY_CLEANUP_NAMES:
        from importlib import import_module

        module = import_module(".cleanup", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    # Any other name the archive module exports is lazily re-exported, which
    # keeps a broad compatible public surface without eager runpy side effects.
    try:
        from . import archive as _archive
    except Exception as exc:  # pragma: no cover - defensive
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    if hasattr(_archive, name):
        value = getattr(_archive, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        set(globals()) | set(__all__) | _LAZY_ARCHIVE_NAMES | _LAZY_CLEANUP_NAMES
    )


def generate_with_dataset(
    dataset: Dataset,
    start: date | str,
    end: date | str,
    filters: Mapping[str, Any] | None = None,
    group_by: Iterable[str] | str = (),
) -> ReportResult:
    query = normalize_query(start, end, filters, group_by)
    verifier = SliceVerifier(dataset)
    plan = Planner(dataset, query, verifier).build()
    return execute(dataset, query, plan)


def generate(
    data_dir: str | Path,
    start: date | str,
    end: date | str,
    filters: Mapping[str, Any] | None = None,
    group_by: Iterable[str] | str = (),
) -> ReportResult:
    dataset = load_catalog(data_dir)
    return generate_with_dataset(dataset, start, end, filters, group_by)
