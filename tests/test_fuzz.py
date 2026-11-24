"""Randomised differential tests against exhaustive search.

Each scenario builds a small warehouse from a hidden per-day truth table:

* every day has hidden ledger rows;
* some days expose those rows as a readable ledger, some are catalogued
  unreadable, some have no catalogue entry at all;
* summary slices are generated from the same hidden rows, so a slice that
  covers a purged day is arithmetically truthful.

Then the test checks that

* feasibility and the minimum ledger-read count equal an exhaustive search
  over all disjoint slice subsets;
* the returned coverage covers every query day exactly once;
* grouped results equal a direct aggregation of the hidden rows.
"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from procurement_report import generate

from tests.fixtures import build_dataset
from tests.reference import all_days, brute_force_min_reads

BASE = date(2024, 3, 1)
DEPTS = ("dA", "dB")
CATS = ("cX", "cY")


def _day(offset: int) -> str:
    return (BASE + timedelta(days=offset)).isoformat()


def _random_rows(rng: random.Random) -> list[dict]:
    out = []
    for _ in range(rng.randrange(0, 4)):
        out.append(
            {
                "department": rng.choice(DEPTS),
                "category": rng.choice(CATS),
                "amount": rng.choice([None, rng.randrange(-50, 50)]),
            }
        )
    return out


class FuzzTests(unittest.TestCase):
    def test_random_scenarios(self) -> None:
        rng = random.Random(20241124)
        for case in range(200):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                self._run_case(rng, Path(tmp) / "data", case)

    def _run_case(self, rng: random.Random, root: Path, case: int) -> None:
        n = rng.randrange(1, 9)
        hidden = {i: _random_rows(rng) for i in range(n)}

        days: dict[str, object] = {}
        for i in range(n):
            roll = rng.random()
            if roll < 0.2:
                days[_day(i)] = None  # inside the range, no catalogue entry
            elif roll < 0.55:
                days[_day(i)] = False  # catalogued, unreadable
            else:
                days[_day(i)] = hidden[i]  # readable ledger rows

        specs = []
        versions = ["v1", "v1", "v1", "v0"]
        for s in range(rng.randrange(0, 6)):
            start_i = rng.randrange(0, n)
            end_i = rng.randrange(start_i + 1, n + 1)
            dims = rng.choice([["department"], ["category"], ["department", "category"]])
            # Drop some groups randomly: absence in a complete slice means no
            # records, and a dropped group must not reappear in the report.
            groups = self._groups_from_hidden(hidden, start_i, end_i, dims)
            specs.append(
                {
                    "slice_id": f"s{s}",
                    "start": _day(start_i),
                    "end": _day(end_i),
                    "group_dims": dims,
                    "groups": groups,
                    "version": rng.choice(versions),
                }
            )

        build_dataset(
            root,
            ledger_start=_day(0),
            ledger_end=_day(n),
            days=days,
            summaries=specs,
        )
        catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))

        qstart = rng.randrange(0, n)
        qend = rng.randrange(qstart + 1, n + 1)
        group_by = rng.choice([[], ["department"], ["category"], ["department", "category"]])
        filters: dict[str, list[str]] = {}
        if rng.random() < 0.4:
            dim = rng.choice(["department", "category"])
            pool = DEPTS if dim == "department" else CATS
            filters[dim] = [rng.choice(pool)]
        if rng.random() < 0.15 and "department" not in filters:
            filters["department"] = list(DEPTS)

        query = {
            "start": _day(qstart),
            "end": _day(qend),
            "group_by": group_by,
            "filters": filters,
        }
        expected_plan = brute_force_min_reads(catalog, root, query)
        result = generate(
            root, query["start"], query["end"], filters=filters, group_by=group_by
        )

        self.assertEqual(
            result.feasible,
            expected_plan is not None,
            f"case {case}: feasibility mismatch (query={query}, filters={filters})",
        )
        if expected_plan is None:
            self.assertEqual(result.rows, [])
            self.assertIsNotNone(result.failure_reason)
            return

        self.assertEqual(result.ledger_days_read, expected_plan)
        self._check_cover(result, query)

        truth = self._reference_rows(hidden, qstart, qend, group_by, filters)
        got = {}
        for row in result.rows:
            key = tuple(row["group"][d] for d in group_by)
            got[key] = row
        self.assertEqual(set(got), set(truth))
        for key, (records, nonnull, total) in truth.items():
            row = got[key]
            self.assertEqual(row["record_count"], records)
            self.assertEqual(row["amount_nonnull_count"], nonnull)
            self.assertEqual(row["amount_sum"], total)

    def _check_cover(self, result, query: dict) -> None:
        covered: dict[str, str] = {}
        for seg in result.segments:
            for day in all_days(seg["start"], seg["end"]):
                self.assertNotIn(day, covered, "day covered twice")
                covered[day] = seg["source"]
        self.assertEqual(sorted(covered), all_days(query["start"], query["end"]))
        ledger_days = [d for d, src in covered.items() if src == "ledger"]
        self.assertEqual(sorted(ledger_days), sorted(result.ledger_day_list))

    def _groups_from_hidden(self, hidden, start_i, end_i, dims) -> list[dict]:
        counts: dict[tuple, list[int]] = {}
        for i in range(start_i, end_i):
            for row in hidden[i]:
                key = tuple(row[d] for d in dims)
                slot = counts.setdefault(key, [0, 0, 0])
                slot[0] += 1
                if row["amount"] is not None:
                    slot[1] += 1
                    slot[2] += row["amount"]
        return [
            {
                "values": list(key),
                "record_count": v[0],
                "amount_nonnull_count": v[1],
                "amount_sum": v[2],
            }
            for key, v in counts.items()
        ]

    def _reference_rows(self, hidden, qstart, qend, group_by, filters):
        counts: dict[tuple, list[int]] = {}
        for i in range(qstart, qend):
            for row in hidden[i]:
                values = {"department": row["department"], "category": row["category"]}
                if any(values[d] not in allowed for d, allowed in filters.items()):
                    continue
                key = tuple(values[d] for d in group_by)
                slot = counts.setdefault(key, [0, 0, 0])
                slot[0] += 1
                if row["amount"] is not None:
                    slot[1] += 1
                    slot[2] += row["amount"]
        return {k: tuple(v) for k, v in counts.items() if v[0] > 0}


if __name__ == "__main__":
    unittest.main()
