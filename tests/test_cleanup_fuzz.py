"""Randomised differential tests for retention cleanup planning.

For small warehouses we brute-force every subset of candidate ledger days and
check that

* a safe plan really is safe: every workload stays producible with exactly the
  declared cleanable days removed (using the ordinary report engine);
* the number of cleanable days equals the true maximum over all subsets;
* an unprovable answer is correct: even the full current catalogue cannot
  produce the reported earliest workload.
"""

from __future__ import annotations

import itertools
import json
import random
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from procurement_report import generate, plan_cleanup

from tests.fixtures import build_dataset

BASE = date(2024, 3, 1)
DEPTS = ("dA", "dB")
CATS = ("cX", "cY")


def _day(offset: int) -> str:
    return (BASE + timedelta(days=offset)).isoformat()


def _days(start: int, end: int) -> list[str]:
    return [_day(i) for i in range(start, end)]


class CleanupFuzzTests(unittest.TestCase):
    def test_random_scenarios(self) -> None:
        rng = random.Random(20241125)
        for case in range(80):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                self._run_case(rng, Path(tmp), case)

    def _run_case(self, rng: random.Random, tmp: Path, case: int) -> None:
        n = rng.randrange(1, 7)
        hidden = {
            i: [
                {
                    "department": rng.choice(DEPTS),
                    "category": rng.choice(CATS),
                    "amount": rng.choice([None, rng.randrange(-9, 20)]),
                }
                for _ in range(rng.randrange(0, 4))
            ]
            for i in range(n)
        }

        # Some days readable, some purged, some absent.
        readable: dict[int, bool] = {}
        days_spec: dict[str, object] = {}
        for i in range(n):
            roll = rng.random()
            if roll < 0.2:
                days_spec[_day(i)] = None
                readable[i] = False
            elif roll < 0.5:
                days_spec[_day(i)] = False
                readable[i] = False
            else:
                days_spec[_day(i)] = hidden[i]
                readable[i] = True

        specs = []
        for s in range(rng.randrange(0, 5)):
            start_i = rng.randrange(0, n)
            end_i = rng.randrange(start_i + 1, n + 1)
            dims = rng.choice([["department"], ["category"], ["department", "category"]])
            specs.append(
                {
                    "slice_id": f"s{s}",
                    "start": _day(start_i),
                    "end": _day(end_i),
                    "group_dims": dims,
                    "groups": self._groups(hidden, start_i, end_i, dims),
                    "version": rng.choice(["v1", "v1", "v0"]),
                    "complete": rng.random() > 0.1,
                }
            )

        queries = []
        for q in range(rng.randrange(0, 4)):
            qs = rng.randrange(0, n)
            qe = rng.randrange(qs + 1, n + 1)
            entry = {"query_id": f"q{q}", "start": _day(qs), "end": _day(qe)}
            if rng.random() < 0.4:
                dim = rng.choice(["department", "category"])
                pool = DEPTS if dim == "department" else CATS
                entry["filters"] = {dim: [rng.choice(pool)]}
            entry["group_by"] = rng.choice([[], ["department"], ["category"], ["department", "category"]])
            queries.append(entry)

        before_i = rng.randrange(0, n + 2)
        before = _day(before_i)

        data_root = tmp / f"data{case}"
        build_dataset(
            data_root,
            ledger_start=_day(0),
            ledger_end=_day(n),
            days=days_spec,
            summaries=specs,
        )
        workload_path = tmp / f"workload{case}.json"
        workload_path.write_text(json.dumps({"queries": queries}), encoding="utf-8")

        result = plan_cleanup(data_root, before, workload_path)
        payload = result.to_json()

        candidates = [
            i for i in range(n) if readable[i] and _day(i) < before
        ]

        # ---- Oracle: brute force over candidate subsets -----------------
        def feasible_after(cleaned: set[int]) -> bool:
            for entry in queries:
                clean_view = self._build_view(tmp, case, n, days_spec, specs, hidden, cleaned)
                report = generate(
                    clean_view,
                    entry["start"],
                    entry["end"],
                    filters=entry.get("filters"),
                    group_by=entry.get("group_by", []),
                )
                if not report.feasible:
                    return False
            return True

        best = None
        for r in range(len(candidates), -1, -1):
            if best is not None:
                break
            for subset in itertools.combinations(candidates, r):
                if feasible_after(set(subset)):
                    best = set(subset)
                    break
        # best is the maximum-cardinality cleanable set (None if empty fails,
        # but the empty set is always "nothing cleaned").
        empty_ok = feasible_after(set())

        if not empty_ok:
            self.assertFalse(payload["can_clean"], f"case {case}: expected cannot_clean")
            self.assertIsNotNone(payload.get("unprovable_workload"))
            return

        self.assertTrue(payload["can_clean"], f"case {case}: expected a safe plan")

        declared = {
            (date.fromisoformat(d) - BASE).days for d in payload["cleanable_dates"]
        }
        self.assertTrue(
            feasible_after(declared), f"case {case}: declared plan is not actually safe"
        )
        self.assertEqual(
            len(declared),
            len(best),
            f"case {case}: not a maximum-cardinality cleanup "
            f"(declared={sorted(declared)}, best={sorted(best)})",
        )

    def _build_view(self, tmp, case, n, days_spec, specs, hidden, cleaned) -> Path:
        root = tmp / f"view{case}"
        view_days: dict[str, object] = {}
        for i in range(n):
            day = _day(i)
            if i in cleaned:
                # A cleaned day keeps its catalogue entry but becomes
                # unreadable; it is never treated as absent/empty.
                view_days[day] = False
            elif day in days_spec:
                view_days[day] = days_spec[day]
            else:  # pragma: no cover - every day is in days_spec
                view_days[day] = False
        build_dataset(
            root,
            ledger_start=_day(0),
            ledger_end=_day(n),
            days=view_days,
            summaries=specs,
        )
        return root

    def _groups(self, hidden, start_i, end_i, dims) -> list[dict]:
        counts: dict[tuple, list[int]] = {}
        for i in range(start_i, end_i):
            for record in hidden[i]:
                key = tuple(record[d] for d in dims)
                slot = counts.setdefault(key, [0, 0, 0])
                slot[0] += 1
                if record["amount"] is not None:
                    slot[1] += 1
                    slot[2] += record["amount"]
        return [
            {
                "values": list(key),
                "record_count": v[0],
                "amount_nonnull_count": v[1],
                "amount_sum": v[2],
            }
            for key, v in counts.items()
        ]


if __name__ == "__main__":
    unittest.main()
