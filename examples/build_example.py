"""Build the bundled example warehouse.

    python3 examples/build_example.py /tmp/procurement-demo

The example mirrors the motivating situation: a January report whose daily
ledgers were partly purged, with several generations of closing summaries,
including overlapping ones and one slice carrying the wrong ledger version.

Every summary slice is *derived from the hidden daily rows*, so the example is
internally truthful: the tool's answer over the whole range must equal a
direct aggregation of those rows.  ``tests/test_example.py`` asserts exactly
that.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.fixtures import build_dataset  # noqa: E402

DEPARTMENTS = ("north", "south")
CATEGORIES = ("steel", "glass")
START = "2024-01-01"
END = "2024-01-16"


def row(department: str, category: str, amount: int | None = None) -> dict:
    return {"department": department, "category": category, "amount": amount}


#: Hidden truth for every day.  Some of these days are later purged from disk.
HIDDEN: dict[str, list[dict]] = {
    "2024-01-01": [row("north", "steel", 100), row("south", "glass", None)],
    "2024-01-02": [row("north", "steel", 250), row("south", "glass", 40)],
    "2024-01-03": [row("north", "steel", 1000), row("south", "steel", 60)],
    "2024-01-04": [row("north", "steel", 1000), row("south", "glass", 70)],
    "2024-01-05": [row("north", "glass", 700), row("south", "steel", 80)],
    "2024-01-06": [row("north", "steel", 1000), row("south", "glass", 80)],
    "2024-01-07": [row("north", "steel", 1000), row("south", "steel", 90)],
    "2024-01-08": [row("north", "glass", None), row("south", "glass", None)],
    "2024-01-09": [row("north", "glass", -25), row("south", "steel", 10**19)],
    "2024-01-10": [row("north", "glass", 5)],
    "2024-01-11": [row("north", "steel", 400), row("south", "glass", 10)],
    "2024-01-12": [row("north", "steel", 10), row("south", "glass", 10)],
    "2024-01-13": [row("north", "glass", 10), row("south", "steel", 10)],
    "2024-01-14": [row("south", "glass", 10)],
    "2024-01-15": [row("north", "glass"), row("south", "steel", 10)],
}

#: Days that still have a readable daily ledger on disk.
READABLE = {"2024-01-01", "2024-01-02", "2024-01-09", "2024-01-10"}


def aggregate(days: Iterable[str], dims: Sequence[str]) -> list[dict]:
    """Truthful groups for the half-open day interval formed by ``days``."""

    counts: dict[tuple, list[int]] = {}
    for day in days:
        for record in HIDDEN[day]:
            key = tuple(record[dim] for dim in dims)
            slot = counts.setdefault(key, [0, 0, 0])
            slot[0] += 1
            if record["amount"] is not None:
                slot[1] += 1
                slot[2] += record["amount"]
    return [
        {
            "values": list(key),
            "record_count": values[0],
            "amount_nonnull_count": values[1],
            "amount_sum": values[2],
        }
        for key, values in counts.items()
    ]


def days_between(start: str, end: str) -> list[str]:
    from datetime import date, timedelta

    cur, stop = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while cur < stop:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def _slice(slice_id: str, start: str, end: str, dims: Sequence[str], **extra) -> dict:
    return {
        "slice_id": slice_id,
        "start": start,
        "end": end,
        "group_dims": list(dims),
        "groups": aggregate(days_between(start, end), dims),
        **extra,
    }


def build_summaries() -> list[dict]:
    return [
        # Current closing summary covering the purged block Jan 3..Jan 8.
        _slice("jan03-08-v7", "2024-01-03", "2024-01-09", ("department", "category")),
        # An earlier generation with an overlapping range.  Legal on its own,
        # but never combined with the slice above: the shared days would be
        # counted twice and may not be netted out.
        _slice("jan05-12-v7-old", "2024-01-05", "2024-01-13", ("department", "category")),
        # Late-month summary for the archived tail.
        _slice("jan11-16-v7", "2024-01-11", "2024-01-16", ("department", "category")),
        # Wrong ledger version: rejected, never used.
        _slice(
            "jan09-16-v6", "2024-01-09", "2024-01-16",
            ("department", "category"), version="v6",
        ),
        # Right version but department-only: unusable for a category query.
        _slice("jan03-08-deptonly", "2024-01-03", "2024-01-09", ("department",)),
    ]


def main(target: str | None = None) -> Path:
    root = Path(target or "/tmp/procurement-demo")
    days: dict[str, object] = {
        day: (rows if day in READABLE else False) for day, rows in HIDDEN.items()
    }
    build_dataset(
        root,
        ledger_version="v7",
        ledger_start=START,
        ledger_end=END,
        departments=DEPARTMENTS,
        categories=CATEGORIES,
        days=days,
        summaries=build_summaries(),
    )
    return root


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else None))
