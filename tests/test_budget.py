"""Tests for budget-vs-actual reconciliation.

Run with:  python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from procurement_report import ConfigError, DataError, generate, reconcile
from procurement_report.cli import main

from tests.fixtures import build_dataset, write_budget


def rows(*spec):
    return [{"department": d, "category": c, "amount": a} for d, c, a in spec]


def group(values, records, nonnull, total):
    return {
        "values": list(values),
        "record_count": records,
        "amount_nonnull_count": nonnull,
        "amount_sum": total,
    }


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "data"
        self.budget_dir = Path(self._tmp.name) / "budgets"
        self.budget_dir.mkdir()

    def build(self, **kwargs) -> Path:
        kwargs.setdefault("ledger_start", "2024-01-01")
        kwargs.setdefault("ledger_end", "2024-01-11")
        return build_dataset(self.root, **kwargs)

    def budget(self, rules, name: str = "rules.json") -> Path:
        return write_budget(self.budget_dir / name, rules)

    def by_key(self, result) -> dict[tuple, dict]:
        return {
            (
                row["group"]["department"],
                row["group"]["category"],
                row["segment"]["start"],
                row["segment"]["end"],
            ): row
            for row in result.rows
        }


class RuleValidationTests(Base):
    def setUp(self) -> None:
        super().setUp()
        self.build(days={})

    def load(self, rules):
        from procurement_report import load_budget, load_catalog

        dataset = load_catalog(self.root)
        return load_budget(self.budget(rules), dataset)

    def test_overlapping_rules_same_pair_rejected(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            self.load(
                [
                    {"department": "dA", "category": "cX", "start": "2024-01-01", "end": "2024-01-05"},
                    {"department": "dA", "category": "cX", "start": "2024-01-04", "end": "2024-01-08"},
                ]
            )
        self.assertIn("overlap", str(caught.exception))

    def test_adjacent_rules_are_allowed(self) -> None:
        book = self.load(
            [
                {"department": "dA", "category": "cX", "start": "2024-01-01", "end": "2024-01-05", "daily_limit": 1},
                {"department": "dA", "category": "cX", "start": "2024-01-05", "end": "2024-01-08", "daily_limit": 2},
            ]
        )
        self.assertEqual(len(book.rules), 2)

    def test_overlap_across_different_pairs_is_fine(self) -> None:
        book = self.load(
            [
                {"department": "dA", "category": "cX", "start": "2024-01-01", "end": "2024-01-08"},
                {"department": "dA", "category": "cY", "start": "2024-01-01", "end": "2024-01-08"},
                {"department": "dB", "category": "cX", "start": "2024-01-01", "end": "2024-01-08"},
            ]
        )
        self.assertEqual(len(book.rules), 3)

    def test_bad_rules_are_config_errors(self) -> None:
        cases = [
            [{"department": "dA", "category": "zz", "start": "2024-01-01", "end": "2024-01-05"}],
            [{"department": "dA", "category": "cX", "start": "2024-01-05", "end": "2024-01-05"}],
            [{"department": "dA", "category": "cX", "start": "2024-01-01", "end": "2024-01-05", "daily_limit": -1}],
            [{"department": "dA", "category": "cX", "start": "2024-01-01", "end": "2024-01-05", "daily_limit": 1.5}],
            [{"department": "dA", "category": "cX", "start": "2024-01-01", "end": "2024-01-05", "daily_limit": True}],
            [{"department": True, "category": "cX", "start": "2024-01-01", "end": "2024-01-05"}],
        ]
        for i, rules in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ConfigError):
                self.load(rules)

    def test_budget_file_inside_data_dir_is_rejected(self) -> None:
        path = self.root / "budget.json"
        write_budget(path, [])
        with self.assertRaises(ConfigError) as caught:
            reconcile(self.root, "2024-01-01", "2024-01-02", path)
        self.assertIn("outside the data directory", str(caught.exception))

    def test_missing_budget_file_is_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            reconcile(self.root, "2024-01-01", "2024-01-02", self.budget_dir / "nope.json")


class SegmentTests(Base):
    def test_boundary_split_and_per_segment_budget(self) -> None:
        # Rule changes on Jan 4.  Query Jan 1..Jan 7, all ledger readable.
        self.build(
            days={
                "2024-01-01": rows(("dA", "cX", 10)),
                "2024-01-02": rows(("dA", "cX", 10)),
                "2024-01-03": rows(("dA", "cX", 10)),
                "2024-01-04": rows(("dA", "cX", 4)),
                "2024-01-05": rows(("dA", "cX", 4)),
                "2024-01-06": rows(("dA", "cX", 4)),
            },
        )
        book = self.budget(
            [
                {"rule_id": "a", "department": "dA", "category": "cX",
                 "start": "2024-01-01", "end": "2024-01-04", "daily_limit": 5},
                {"rule_id": "b", "department": "dA", "category": "cX",
                 "start": "2024-01-04", "end": "2024-01-07", "daily_limit": 5},
            ]
        )
        result = reconcile(self.root, "2024-01-01", "2024-01-07", book)
        self.assertTrue(result.feasible)
        got = self.by_key(result)
        first = got[("dA", "cX", "2024-01-01", "2024-01-04")]
        second = got[("dA", "cX", "2024-01-04", "2024-01-07")]
        self.assertEqual(first["actual_amount"], 30)
        self.assertEqual(first["budget_amount"], 15)
        self.assertEqual(first["status"], "over_budget")
        self.assertEqual(second["actual_amount"], 12)
        self.assertEqual(second["budget_amount"], 15)
        self.assertEqual(second["status"], "within_budget")
        self.assertEqual(second["difference"], 3)

    def test_query_clipped_to_partial_segment(self) -> None:
        # Rule spans Jan 1..Jan 10 (90 budget), query only Jan 4..Jan 7: the
        # segment is clipped and the budget is 3 days * 10 = 30.
        self.build(
            days={day: rows(("dA", "cX", 1)) for day in
                  ("2024-01-04", "2024-01-05", "2024-01-06")},
        )
        book = self.budget(
            [{"department": "dA", "category": "cX",
              "start": "2024-01-01", "end": "2024-01-10", "daily_limit": 10}]
        )
        result = reconcile(self.root, "2024-01-04", "2024-01-07", book)
        row = result.rows[0]
        self.assertEqual(row["segment"], {"start": "2024-01-04", "end": "2024-01-07", "days": 3})
        self.assertEqual(row["budget_amount"], 30)
        self.assertEqual(row["actual_amount"], 3)

    def test_uncovered_dates_are_unbudgeted_not_zero(self) -> None:
        self.build(days={day: rows(("dA", "cX", 3)) for day in
                         ("2024-01-01", "2024-01-02", "2024-01-03")})
        book = self.budget(
            [{"department": "dA", "category": "cX",
              "start": "2024-01-02", "end": "2024-01-04", "daily_limit": 100}]
        )
        result = reconcile(self.root, "2024-01-01", "2024-01-04", book)
        got = self.by_key(result)
        unbudgeted = got[("dA", "cX", "2024-01-01", "2024-01-02")]
        configured = got[("dA", "cX", "2024-01-02", "2024-01-04")]
        self.assertEqual(unbudgeted["status"], "unbudgeted")
        self.assertIsNone(unbudgeted["budget_amount"])
        self.assertIsNone(unbudgeted["difference"])
        self.assertEqual(unbudgeted["actual_amount"], 3)
        self.assertEqual(configured["budget_amount"], 200)
        self.assertEqual(configured["actual_amount"], 6)

    def test_explicit_zero_limit_is_a_real_zero_budget(self) -> None:
        self.build(days={"2024-01-01": rows(("dA", "cX", 0))})
        book = self.budget(
            [{"department": "dA", "category": "cX",
              "start": "2024-01-01", "end": "2024-01-02", "daily_limit": 0}]
        )
        result = reconcile(self.root, "2024-01-01", "2024-01-02", book)
        row = result.rows[0]
        self.assertEqual(row["status"], "within_budget")
        self.assertEqual(row["budget_amount"], 0)
        self.assertIsNotNone(row["difference"])

    def test_zero_actual_with_zero_budget_is_within(self) -> None:
        self.build(days={"2024-01-01": rows(("dA", "cY", 0))})
        book = self.budget(
            [{"department": "dA", "category": "cX",
              "start": "2024-01-01", "end": "2024-01-02", "daily_limit": 0}]
        )
        result = reconcile(self.root, "2024-01-01", "2024-01-02", book)
        got = self.by_key(result)
        self.assertEqual(got[("dA", "cX", "2024-01-01", "2024-01-02")]["actual_amount"], 0)
        self.assertEqual(got[("dA", "cX", "2024-01-01", "2024-01-02")]["status"], "within_budget")

    def test_configured_pair_without_records_gets_a_zero_actual_row(self) -> None:
        self.build(days={"2024-01-01": [], "2024-01-02": []})
        book = self.budget(
            [{"department": "dA", "category": "cX",
              "start": "2024-01-01", "end": "2024-01-03", "daily_limit": 7}]
        )
        result = reconcile(self.root, "2024-01-01", "2024-01-03", book)
        rows = self.by_key(result)
        row = rows[("dA", "cX", "2024-01-01", "2024-01-03")]
        self.assertEqual(row["record_count"], 0)
        self.assertEqual(row["actual_amount"], 0)
        self.assertEqual(row["budget_amount"], 14)
        self.assertEqual(row["status"], "within_budget")

    def test_actual_is_attributed_by_effective_day(self) -> None:
        # Rows on the boundary day belong to the later segment.
        self.build(
            days={
                "2024-01-01": [],
                "2024-01-02": [],
                "2024-01-03": rows(("dA", "cX", 5)),
                "2024-01-04": rows(("dA", "cX", 50)),
                "2024-01-05": [],
            },
            summaries=[],
        )
        book = self.budget(
            [
                {"rule_id": "old", "department": "dA", "category": "cX",
                 "start": "2024-01-01", "end": "2024-01-04", "daily_limit": 1},
                {"rule_id": "new", "department": "dA", "category": "cX",
                 "start": "2024-01-04", "end": "2024-01-06", "daily_limit": 1},
            ]
        )
        result = reconcile(self.root, "2024-01-01", "2024-01-06", book)
        got = self.by_key(result)
        self.assertEqual(got[("dA", "cX", "2024-01-01", "2024-01-04")]["actual_amount"], 5)
        self.assertEqual(got[("dA", "cX", "2024-01-04", "2024-01-06")]["actual_amount"], 50)


class SourceIntegrityTests(Base):
    def test_straddling_slice_is_rejected_and_reconciliation_fails(self) -> None:
        # Only a summary covers the purged days, and it spans the Jan 4 budget
        # boundary.  It cannot be split, so no complete table may be emitted.
        self.build(
            days={"2024-01-01": rows(("dA", "cX", 1))},
            summaries=[
                {
                    "slice_id": "s1", "start": "2024-01-02", "end": "2024-01-06",
                    "group_dims": ["department", "category"],
                    "groups": [group(("dA", "cX"), 4, 4, 40)],
                }
            ],
        )
        book = self.budget(
            [
                {"rule_id": "a", "department": "dA", "category": "cX",
                 "start": "2024-01-02", "end": "2024-01-04", "daily_limit": 1},
                {"rule_id": "b", "department": "dA", "category": "cX",
                 "start": "2024-01-04", "end": "2024-01-06", "daily_limit": 1},
            ]
        )
        result = reconcile(self.root, "2024-01-01", "2024-01-06", book)
        self.assertFalse(result.feasible)
        payload = result.to_json()
        self.assertEqual(payload["status"], "cannot_reconcile")
        self.assertNotIn("rows", payload)
        self.assertIn("cannot be split", payload["failure_reason"])
        self.assertIn("straddles the budget boundary", json.dumps(payload["rejected_slices"]))

    def test_slice_aligned_to_budget_boundary_is_usable(self) -> None:
        self.build(
            days={"2024-01-01": rows(("dA", "cX", 1))},
            summaries=[
                {
                    "slice_id": "s1", "start": "2024-01-02", "end": "2024-01-04",
                    "group_dims": ["department", "category"],
                    "groups": [group(("dA", "cX"), 2, 2, 20)],
                },
                {
                    "slice_id": "s2", "start": "2024-01-04", "end": "2024-01-06",
                    "group_dims": ["department", "category"],
                    "groups": [group(("dA", "cX"), 2, 2, 60)],
                },
            ],
        )
        book = self.budget(
            [
                {"rule_id": "a", "department": "dA", "category": "cX",
                 "start": "2024-01-02", "end": "2024-01-04", "daily_limit": 5},
                {"rule_id": "b", "department": "dA", "category": "cX",
                 "start": "2024-01-04", "end": "2024-01-06", "daily_limit": 5},
            ]
        )
        result = reconcile(self.root, "2024-01-01", "2024-01-06", book)
        self.assertTrue(result.feasible)
        got = self.by_key(result)
        self.assertEqual(got[("dA", "cX", "2024-01-02", "2024-01-04")]["actual_amount"], 20)
        self.assertEqual(got[("dA", "cX", "2024-01-04", "2024-01-06")]["actual_amount"], 60)
        self.assertEqual(got[("dA", "cX", "2024-01-04", "2024-01-06")]["status"], "over_budget")

    def test_slice_spanning_boundary_for_inactive_pair_is_usable(self) -> None:
        # The slice spans Jan 4 but only reports category cY, which has no
        # budget rules in the query; cX groups do not appear at all.
        self.build(
            days={},
            summaries=[
                {
                    "slice_id": "s1", "start": "2024-01-01", "end": "2024-01-06",
                    "group_dims": ["department", "category"],
                    "groups": [group(("dA", "cY"), 5, 5, 50)],
                }
            ],
        )
        book = self.budget(
            [
                {"rule_id": "a", "department": "dA", "category": "cX",
                 "start": "2024-01-01", "end": "2024-01-04", "daily_limit": 1},
                {"rule_id": "b", "department": "dA", "category": "cX",
                 "start": "2024-01-04", "end": "2024-01-06", "daily_limit": 1},
            ]
        )
        result = reconcile(self.root, "2024-01-01", "2024-01-06", book)
        self.assertTrue(result.feasible)
        got = self.by_key(result)
        self.assertEqual(got[("dA", "cY", "2024-01-01", "2024-01-06")]["actual_amount"], 50)

    def test_slice_missing_a_dimension_is_rejected(self) -> None:
        self.build(
            days={},
            summaries=[
                {
                    "slice_id": "s1", "start": "2024-01-01", "end": "2024-01-06",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 5, 5, 50)],
                }
            ],
        )
        book = self.budget(
            [{"department": "dA", "category": "cX",
              "start": "2024-01-01", "end": "2024-01-06", "daily_limit": 1}]
        )
        result = reconcile(self.root, "2024-01-01", "2024-01-06", book)
        # The slice does not retain category, so it is incompatible and never
        # used; without any other source the reconciliation cannot complete.
        self.assertFalse(result.feasible)
        self.assertNotIn("s1", result.summary_slices_used)
        self.assertEqual(result.rows, [])

    def test_exact_big_and_negative_amounts(self) -> None:
        big = 2**70 + 1
        self.build(days={"2024-01-01": rows(("dA", "cX", big), ("dA", "cX", -big + 10))})
        book = self.budget(
            [{"department": "dA", "category": "cX",
              "start": "2024-01-01", "end": "2024-01-02", "daily_limit": 5}]
        )
        result = reconcile(self.root, "2024-01-01", "2024-01-02", book)
        row = result.rows[0]
        self.assertEqual(row["actual_amount"], 10)
        self.assertEqual(row["difference"], -5)
        self.assertEqual(row["status"], "over_budget")

    def test_unreadable_source_still_raises_data_error(self) -> None:
        self.build(days={"2024-01-01": rows(("dA", "cX", 1))})
        (self.root / "ledger" / "2024-01-01.jsonl").unlink()
        book = self.budget(
            [{"department": "dA", "category": "cX",
              "start": "2024-01-01", "end": "2024-01-02", "daily_limit": 1}]
        )
        with self.assertRaises(DataError):
            reconcile(self.root, "2024-01-01", "2024-01-02", book)

    def test_filters_are_applied_to_budget_rows(self) -> None:
        self.build(
            days={day: rows(("dA", "cX", 2), ("dB", "cY", 5)) for day in ("2024-01-01",)}
        )
        book = self.budget(
            [
                {"department": "dA", "category": "cX",
                 "start": "2024-01-01", "end": "2024-01-02", "daily_limit": 1},
                {"department": "dB", "category": "cY",
                 "start": "2024-01-01", "end": "2024-01-02", "daily_limit": 100},
            ]
        )
        result = reconcile(self.root, "2024-01-01", "2024-01-02", book,
                           filters={"department": ["dA"]})
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0]["group"], {"department": "dA", "category": "cX"})

    def test_plain_report_unchanged(self) -> None:
        self.build(days={"2024-01-01": rows(("dA", "cX", 1))})
        result = generate(self.root, "2024-01-01", "2024-01-02", group_by=["department"])
        self.assertEqual(result.rows[0]["amount_sum"], 1)


class CliBudgetTests(Base):
    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(list(argv))
            except SystemExit as exc:  # argparse errors
                code = int(exc.code)
        return code, out.getvalue(), err.getvalue()

    def test_budget_cli_ok(self) -> None:
        self.build(days={"2024-01-01": rows(("dA", "cX", 3))})
        book = self.budget(
            [{"department": "dA", "category": "cX",
              "start": "2024-01-01", "end": "2024-01-02", "daily_limit": 2}]
        )
        code, out, _ = self.run_cli(
            str(self.root), "--start", "2024-01-01", "--end", "2024-01-02",
            "--budget", str(book),
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["rows"][0]["status"], "over_budget")

    def test_budget_cli_cannot_reconcile_exit_three(self) -> None:
        self.build(days={})
        book = self.budget(
            [{"department": "dA", "category": "cX",
              "start": "2024-01-01", "end": "2024-01-02", "daily_limit": 2}]
        )
        code, out, _ = self.run_cli(
            str(self.root), "--start", "2024-01-01", "--end", "2024-01-03",
            "--budget", str(book),
        )
        self.assertEqual(code, 3)
        self.assertEqual(json.loads(out)["status"], "cannot_reconcile")

    def test_budget_cli_bad_file_exit_one(self) -> None:
        self.build(days={})
        code, _, err = self.run_cli(
            str(self.root), "--start", "2024-01-01", "--end", "2024-01-02",
            "--budget", str(self.budget_dir / "missing.json"),
        )
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_budget_cli_rejects_group_by(self) -> None:
        self.build(days={})
        book = self.budget([])
        code, _, err = self.run_cli(
            str(self.root), "--start", "2024-01-01", "--end", "2024-01-02",
            "--budget", str(book), "--group-by", "department",
        )
        self.assertEqual(code, 2)
        self.assertIn("--group-by", err)


if __name__ == "__main__":
    unittest.main()
