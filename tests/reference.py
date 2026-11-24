"""Independent reference implementation used by the tests.

This module deliberately does **not** import the planner or engine.  It
brute-forces every legal selection of summary slices and ledger days, then
compares the chosen plan's ledger-read count against the true minimum and
compares grouped results against a direct row-by-row recomputation.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any


def all_days(start: str, end: str) -> list[str]:
    cur, stop = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while cur < stop:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def brute_force_min_reads(catalog: dict[str, Any], root: Path, query: dict[str, Any]) -> int | None:
    """Minimum number of ledger days over all legal covers, or None.

    Legal cover: a set of whole compatible slices with pairwise disjoint
    intervals, plus a readable ledger day for every remaining day.  No
    overlap is allowed, so this really is exhaustive over the structural
    choices.
    """

    qdays = all_days(query["start"], query["end"])
    qset = set(qdays)
    ledger_readable = {
        entry["day"] for entry in catalog["ledger"]["days"] if entry["readable"]
    }

    candidates = []
    for entry in catalog["summaries"]:
        days = all_days(entry["start"], entry["end"])
        if not set(days) <= qset:
            continue
        dims = set(entry["group_dims"])
        if not set(query.get("filters", {})) <= dims:
            continue
        if not set(query.get("group_by", [])) <= dims:
            continue
        if entry["ledger_version"] != catalog["ledger_version"] or not entry["complete"]:
            continue
        path = root / entry["file"]
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("digest") != entry["digest"]:
            continue
        candidates.append((entry["slice_id"], days))

    best: int | None = None
    # Exhaustive over subsets of candidates (candidate counts are tiny).
    for mask in range(1 << len(candidates)):
        covered: set[str] = set()
        ok = True
        for bit, (_, days) in enumerate(candidates):
            if not mask & (1 << bit):
                continue
            dset = set(days)
            if covered & dset:
                ok = False
                break
            covered |= dset
        if not ok:
            continue
        rest = qset - covered
        if not rest <= ledger_readable:
            continue
        if best is None or len(rest) < best:
            best = len(rest)
    return best


def direct_recompute(
    catalog: dict[str, Any],
    root: Path,
    query: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compute grouped truth from the readable ledgers only.

    The test datasets always keep the ledgers that a correct plan would use,
    so this is an independent oracle for the reported numbers.
    """

    group_by = list(query.get("group_by", []))
    filters = query.get("filters", {})
    out: dict[str, dict[str, Any]] = {}
    for day in all_days(query["start"], query["end"]):
        entry = next(
            (e for e in catalog["ledger"]["days"] if e["day"] == day and e["readable"]), None
        )
        if entry is None:
            continue
        path = root / entry["file"]
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            values = {"department": row["department"], "category": row["category"]}
            if any(values[dim] not in allowed for dim, allowed in filters.items()):
                continue
            key = json.dumps([values[d] for d in group_by], sort_keys=True)
            slot = out.setdefault(
                key,
                {
                    "group": {d: values[d] for d in group_by},
                    "record_count": 0,
                    "amount_nonnull_count": 0,
                    "amount_sum": 0,
                },
            )
            slot["record_count"] += 1
            if row["amount"] is not None:
                slot["amount_nonnull_count"] += 1
                slot["amount_sum"] += row["amount"]
    return out
