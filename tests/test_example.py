"""The bundled example is internally truthful: verify end to end."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from procurement_report import generate

SCRIPT = Path(__file__).resolve().parent.parent / "examples" / "build_example.py"


def load_example_module():
    spec = importlib.util.spec_from_file_location("build_example", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.example = load_example_module()

    def test_full_range_matches_hidden_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self.example.main(str(Path(tmp) / "demo"))
            result = generate(
                root, "2024-01-01", "2024-01-16",
                group_by=["department", "category"],
            )
            self.assertTrue(result.feasible)
            expected = self._expected(["department", "category"])
            self.assertEqual(set(self._keys(result)), set(expected))
            for key, (records, nonnull, total) in expected.items():
                row = {tuple(r["group"][d] for d in ["department", "category"]): r
                       for r in result.rows}[key]
                self.assertEqual(row["record_count"], records)
                self.assertEqual(row["amount_nonnull_count"], nonnull)
                self.assertEqual(row["amount_sum"], total)
            # Only the four surviving daily ledgers are charged.
            self.assertEqual(
                result.ledger_days_read, len(self.example.READABLE)
            )

    def test_filtered_query_matches_hidden_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self.example.main(str(Path(tmp) / "demo"))
            result = generate(
                root, "2024-01-01", "2024-01-16",
                filters={"category": ["steel"]}, group_by=["department"],
            )
            self.assertTrue(result.feasible)
            expected: dict[tuple, list[int]] = {}
            for rows in self.example.HIDDEN.values():
                for row in rows:
                    if row["category"] != "steel":
                        continue
                    slot = expected.setdefault((row["department"],), [0, 0, 0])
                    slot[0] += 1
                    if row["amount"] is not None:
                        slot[1] += 1
                        slot[2] += row["amount"]
            got = {
                (r["group"]["department"],): (
                    r["record_count"], r["amount_nonnull_count"], r["amount_sum"]
                )
                for r in result.rows
            }
            self.assertEqual(got, {k: tuple(v) for k, v in expected.items()})
            # The overlapping alternate slice must never be combined in.
            self.assertNotIn("jan05-12-v7-old", result.summary_slices_used)

    def _expected(self, dims):
        counts: dict[tuple, list[int]] = {}
        for rows in self.example.HIDDEN.values():
            for row in rows:
                key = tuple(row[d] for d in dims)
                slot = counts.setdefault(key, [0, 0, 0])
                slot[0] += 1
                if row["amount"] is not None:
                    slot[1] += 1
                    slot[2] += row["amount"]
        return {k: tuple(v) for k, v in counts.items()}

    def _keys(self, result):
        return [
            tuple(r["group"][d] for d in ["department", "category"])
            for r in result.rows
        ]


if __name__ == "__main__":
    unittest.main()
