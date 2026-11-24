"""Randomised differential tests for budget reconciliation.

Each case builds a fully readable warehouse with a hidden per-day truth table
and a random, non-overlapping set of budget rules.  The reconciliation must
equal a direct recomputation of:

* actual amounts, attributed by effective day;
* ``budget_amount = daily_limit * segment days``;
* ``difference`` and ``status``, including ``unbudgeted`` gaps.
"""

from __future__ import annotations

import random
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from procurement_report import ConfigError, reconcile

from tests.fixtures import build_dataset, write_budget

BASE = date(2024, 5, 1)
DEPTS = ("dA", "dB")
CATS = ("cX", "cY")


def _day(offset: int) -> str:
    return (BASE + timedelta(days=offset)).isoformat()


def _random_rules(rng: random.Random, n: int) -> list[dict]:
    rules: list[dict] = []
    for department in DEPTS:
        for category in CATS:
            # Build a random chain of adjacent/non-adjacent non-overlapping
            # intervals.  Sometimes leave gaps (unbudgeted).
            cursor = 0
            index = 0
            while cursor < n:
                if rng.random() < 0.3:
                    cursor += 1  # gap: no rule
                    continue
                length = rng.randrange(1, n - cursor + 1)
                rules.append(
                    {
                        "rule_id": f"{department}-{category}-{index}",
                        "department": department,
                        "category": category,
                        "start": _day(cursor),
                        "end": _day(cursor + length),
                        "daily_limit": rng.choice([0, 1, 7, 50, 10**6]),
                    }
                )
                cursor += length
                index += 1
    return rules


class BudgetFuzzTests(unittest.TestCase):
    def test_random_rules_match_direct_recomputation(self) -> None:
        rng = random.Random(20241124)
        for case in range(150):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                self._run_case(rng, Path(tmp) / "data", Path(tmp) / "budget.json", case)

    def _run_case(self, rng: random.Random, root: Path, budget_path: Path, case: int) -> None:
        n = rng.randrange(1, 10)
        hidden = {
            i: [
                {
                    "department": rng.choice(DEPTS),
                    "category": rng.choice(CATS),
                    "amount": rng.choice([None, rng.randrange(-20, 40)]),
                }
                for _ in range(rng.randrange(0, 5))
            ]
            for i in range(n)
        }
        rules = _random_rules(rng, n)
        write_budget(budget_path, rules)
        build_dataset(
            root,
            ledger_start=_day(0),
            ledger_end=_day(n),
            days={_day(i): rows for i, rows in hidden.items()},
        )

        qstart = rng.randrange(0, n)
        qend = rng.randrange(qstart + 1, n + 1)
        filters: dict[str, list[str]] = {}
        if rng.random() < 0.4:
            dim = rng.choice(["department", "category"])
            pool = DEPTS if dim == "department" else CATS
            filters[dim] = [rng.choice(pool)]

        result = reconcile(root, _day(qstart), _day(qend), budget_path, filters=filters)
        self.assertTrue(result.feasible, f"case {case}: unexpectedly infeasible")

        expected = self._reference(hidden, rules, qstart, qend, filters)
        got = {
            (
                row["group"]["department"],
                row["group"]["category"],
                row["segment"]["start"],
                row["segment"]["end"],
            ): row
            for row in result.rows
        }
        self.assertEqual(set(got), set(expected), f"case {case}: row set mismatch")
        for key, (actual, budget, status) in expected.items():
            row = got[key]
            self.assertEqual(row["actual_amount"], actual, f"case {case}: {key}")
            self.assertEqual(row["budget_amount"], budget, f"case {case}: {key}")
            self.assertEqual(row["status"], status, f"case {case}: {key}")
            if budget is None:
                self.assertIsNone(row["difference"])
            else:
                self.assertEqual(row["difference"], budget - actual)

    def _reference(self, hidden, rules, qstart, qend, filters):
        by_pair: dict[tuple, list[dict]] = {}
        for rule in rules:
            by_pair.setdefault((rule["department"], rule["category"]), []).append(rule)

        def allowed(pair):
            values = {"department": pair[0], "category": pair[1]}
            return all(values[d] in vals for d, vals in filters.items())

        pairs = {pair for pair in by_pair if allowed(pair)}
        for i in range(qstart, qend):
            for row in hidden.get(i, []):
                pair = (row["department"], row["category"])
                if allowed(pair):
                    pairs.add(pair)

        out = {}
        for pair in pairs:
            for start, end, rule in self._partition(by_pair.get(pair, []), qstart, qend):
                actual = 0
                for i in range(start, end):
                    for row in hidden.get(i, []):
                        if (row["department"], row["category"]) != pair:
                            continue
                        if row["amount"] is not None:
                            actual += row["amount"]
                if rule is None:
                    out[(pair[0], pair[1], _day(start), _day(end))] = (actual, None, "unbudgeted")
                else:
                    budget = rule["daily_limit"] * (end - start)
                    status = "over_budget" if actual > budget else "within_budget"
                    out[(pair[0], pair[1], _day(start), _day(end))] = (actual, budget, status)
        return out

    def _partition(self, rules, qstart, qend):
        rules = sorted(rules, key=lambda r: r["start"])
        segments = []
        cursor = qstart
        for rule in rules:
            lo = max(self._off(rule["start"]), qstart)
            hi = min(self._off(rule["end"]), qend)
            if lo >= hi:
                continue
            if lo > cursor:
                segments.append((cursor, lo, None))
            segments.append((lo, hi, rule))
            cursor = hi
        if cursor < qend:
            segments.append((cursor, qend, None))
        return segments

    @staticmethod
    def _off(day: str) -> int:
        return (date.fromisoformat(day) - BASE).days


class OverlapFuzzTests(unittest.TestCase):
    def test_random_overlapping_rules_rejected(self) -> None:
        rng = random.Random(99)
        for case in range(80):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "data"
                build_dataset(
                    root,
                    ledger_start=_day(0),
                    ledger_end=_day(5),
                    days={_day(i): [] for i in range(5)},
                )
                # Both rules contain the shared day ``d``, so they genuinely
                # overlap on dates for the same department/category pair.
                d = rng.randrange(0, 5)
                start = rng.randrange(0, d + 1)
                end = rng.randrange(d + 1, 6)
                os_ = rng.randrange(0, d + 1)
                oe = rng.randrange(d + 1, 6)
                rules = [
                    {"department": "dA", "category": "cX",
                     "start": _day(start), "end": _day(end), "daily_limit": 1},
                    {"department": "dA", "category": "cX",
                     "start": _day(os_), "end": _day(oe), "daily_limit": 2},
                ]
                path = write_budget(Path(tmp) / "budget.json", rules)
                with self.assertRaises(ConfigError):
                    reconcile(root, _day(0), _day(5), path)


if __name__ == "__main__":
    unittest.main()
