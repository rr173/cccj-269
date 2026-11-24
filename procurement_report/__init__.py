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

from .catalog import load_catalog
from .engine import ReportResult, execute
from .model import ConfigError, DataError, Dataset, Query, normalize_query
from .planner import Planner
from .verify import SliceVerifier

__all__ = [
    "ConfigError",
    "DataError",
    "Dataset",
    "Query",
    "ReportResult",
    "generate",
    "generate_with_dataset",
    "load_catalog",
    "normalize_query",
]


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
