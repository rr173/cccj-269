"""Smoke test for the command-line interface."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from procurement_report.cli import main

from tests.fixtures import build_dataset


def rows(*spec):
    return [{"department": d, "category": c, "amount": a} for d, c, a in spec]


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = build_dataset(
            Path(self._tmp.name) / "data",
            ledger_start="2024-01-01",
            ledger_end="2024-01-04",
            days={
                "2024-01-01": rows(("dA", "cX", 5)),
                "2024-01-02": [],
                "2024-01-03": [],
            },
        )

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_ok_exit_zero_and_json(self) -> None:
        code, out, _ = self.run_cli(
            str(self.root), "--start", "2024-01-01", "--end", "2024-01-03",
            "--group-by", "department",
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["rows"][0]["amount_sum"], 5)

    def test_infeasible_exit_two(self) -> None:
        root = build_dataset(
            Path(self._tmp.name) / "infeasible",
            ledger_start="2024-01-01",
            ledger_end="2024-01-04",
            days={"2024-01-01": rows(("dA", "cX", 5)), "2024-01-02": False, "2024-01-03": False},
            summaries=[
                {
                    "slice_id": "later",
                    "start": "2024-01-03",
                    "end": "2024-01-04",
                    "group_dims": ["department"],
                    "groups": [],
                }
            ],
        )
        code, out, _ = self.run_cli(
            str(root), "--start", "2024-01-02", "--end", "2024-01-04",
            "--group-by", "department",
        )
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertEqual(payload["status"], "cannot_produce")
        self.assertEqual(payload["reachable_boundary"], "2024-01-02")

    def test_bad_filter_syntax_exit_one(self) -> None:
        code, _, err = self.run_cli(
            str(self.root), "--start", "2024-01-01", "--end", "2024-01-02",
            "--filter", "department",
        )
        self.assertEqual(code, 1)
        self.assertIn("DIM=VALUE", err)

    def test_integer_filter_value(self) -> None:
        root = build_dataset(
            Path(self._tmp.name) / "ints",
            ledger_start="2024-01-01",
            ledger_end="2024-01-02",
            departments=(7,),
            categories=(9,),
            days={"2024-01-01": [{"department": 7, "category": 9, "amount": 1}]},
        )
        code, out, _ = self.run_cli(
            str(root), "--start", "2024-01-01", "--end", "2024-01-02",
            "--filter", "department=7", "--group-by", "department",
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["rows"][0]["group"], {"department": 7})


if __name__ == "__main__":
    unittest.main()
