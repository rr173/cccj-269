"""Focused tests for edge cases and format contracts.

Run with:  python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from procurement_report import ConfigError, DataError, generate
from procurement_report.digest import compute_digest

from tests.fixtures import build_dataset


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
        self.root = Path(self._tmp.name) / "data"
        self.addCleanup(self._tmp.cleanup)

    def build(self, **kwargs) -> Path:
        kwargs.setdefault("ledger_start", "2024-01-01")
        kwargs.setdefault("ledger_end", "2024-01-11")
        return build_dataset(self.root, **kwargs)


class QueryContractTests(Base):
    def test_empty_query_interval_is_zero_rows_and_no_reads(self) -> None:
        self.build(days={})
        result = generate(self.root, "2024-01-05", "2024-01-05", group_by=["department"])
        self.assertTrue(result.feasible)
        self.assertEqual(result.rows, [])
        self.assertEqual(result.ledger_days_read, 0)
        self.assertEqual(result.segments, [])

    def test_reversed_query_is_a_config_error(self) -> None:
        self.build(days={})
        with self.assertRaises(ConfigError):
            generate(self.root, "2024-01-05", "2024-01-01")

    def test_unknown_filter_dimension_is_a_config_error(self) -> None:
        self.build(days={})
        with self.assertRaises(ConfigError):
            generate(self.root, "2024-01-01", "2024-01-02", filters={"region": ["x"]})

    def test_unknown_group_dimension_is_a_config_error(self) -> None:
        self.build(days={})
        with self.assertRaises(ConfigError):
            generate(self.root, "2024-01-01", "2024-01-02", group_by=["region"])

    def test_no_group_by_rolls_up_everything(self) -> None:
        self.build(days={"2024-01-01": rows(("dA", "cX", 1), ("dB", "cY", 2))})
        result = generate(self.root, "2024-01-01", "2024-01-02")
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0]["group"], {})
        self.assertEqual(result.rows[0]["record_count"], 2)
        self.assertEqual(result.rows[0]["amount_sum"], 3)

    def test_integer_dimension_values_round_trip(self) -> None:
        self.build(
            days={"2024-01-01": [{"department": 7, "category": 9, "amount": 5}]},
            departments=(7, 8),
            categories=(9, 10),
        )
        result = generate(
            self.root, "2024-01-01", "2024-01-02",
            filters={"department": [7]}, group_by=["category"],
        )
        self.assertEqual(result.rows[0]["group"], {"category": 9})
        self.assertEqual(result.rows[0]["amount_sum"], 5)


class CatalogueContractTests(Base):
    def test_missing_catalogue_is_config_error(self) -> None:
        self.root.mkdir(parents=True)
        with self.assertRaises(ConfigError):
            generate(self.root, "2024-01-01", "2024-01-02")

    def test_slice_outside_business_fact_range_is_rejected(self) -> None:
        self.build(
            days={},
            ledger_end="2024-01-06",
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-10",  # extends past the declared range
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 9, 9, 9)],
                }
            ],
        )
        result = generate(self.root, "2024-01-01", "2024-01-10", group_by=["department"])
        self.assertFalse(result.feasible)
        self.assertTrue(any("outside" in r["reason"] for r in result.rejected_slices))

    def test_slice_file_interval_mismatch_is_unusable(self) -> None:
        self.build(
            days={},
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-06",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 5, 5, 5)],
                    # On-disk payload claims a different interval.
                    "file_payload": {
                        "slice_id": "s1",
                        "start": "2024-01-01",
                        "end": "2024-01-05",
                        "group_dims": ["department"],
                        "ledger_version": "v1",
                        "complete": True,
                        "groups": [group(("dA",), 4, 4, 4)],
                    },
                }
            ],
        )
        result = generate(self.root, "2024-01-01", "2024-01-06", group_by=["department"])
        self.assertFalse(result.feasible)
        self.assertTrue(any("interval" in r["reason"] for r in result.rejected_slices))

    def test_incomplete_slice_is_unusable(self) -> None:
        self.build(
            days={},
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-06",
                    "group_dims": ["department"],
                    "complete": False,
                    "groups": [group(("dA",), 5, 5, 5)],
                }
            ],
        )
        result = generate(self.root, "2024-01-01", "2024-01-06", group_by=["department"])
        self.assertFalse(result.feasible)

    def test_digest_is_content_addressed(self) -> None:
        payload = {
            "slice_id": "s1",
            "start": "2024-01-01",
            "end": "2024-01-02",
            "group_dims": ["department"],
            "ledger_version": "v1",
            "complete": True,
            "groups": [],
        }
        first = compute_digest(payload)
        reordered = dict(reversed(list(payload.items())))
        self.assertEqual(first, compute_digest(reordered))
        changed = dict(payload)
        changed["end"] = "2024-01-03"
        self.assertNotEqual(first, compute_digest(changed))

    def test_missing_catalogue_file_key_is_an_error(self) -> None:
        self.build(days={})
        payload = json.loads((self.root / "catalog.json").read_text(encoding="utf-8"))
        del payload["summaries"]
        (self.root / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ConfigError):
            generate(self.root, "2024-01-01", "2024-01-02")


class ResultShapeTests(Base):
    def test_json_counts_and_status(self) -> None:
        self.build(days={"2024-01-01": rows(("dA", "cX", None))})
        result = generate(self.root, "2024-01-01", "2024-01-02", group_by=["department"])
        payload = result.to_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["row_count"], 1)
        self.assertEqual(payload["rows"][0]["amount_sum"], 0)
        self.assertEqual(payload["rows"][0]["amount_nonnull_count"], 0)
        self.assertIsNone(payload["rows"][0]["average_amount"])
        self.assertEqual(payload["coverage"][0]["source"], "ledger")

    def test_infeasible_json_has_boundary_and_reason(self) -> None:
        self.build(days={})
        result = generate(self.root, "2024-01-01", "2024-01-03", group_by=["department"])
        payload = result.to_json()
        self.assertEqual(payload["status"], "cannot_produce")
        self.assertNotIn("rows", payload)
        self.assertEqual(payload["reachable_boundary"], "2024-01-01")
        self.assertIsNone(payload["last_covered_day"])
        self.assertIn("no usable compatible summary slice", payload["failure_reason"])

    def test_zero_groups_are_not_emitted_but_other_groups_are(self) -> None:
        self.build(
            days={},
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-06",
                    "group_dims": ["department", "category"],
                    "groups": [
                        group(("dA", "cX"), 5, 5, 5),
                        group(("dB", "cY"), 0, 0, 0),
                    ],
                }
            ],
        )
        result = generate(self.root, "2024-01-01", "2024-01-06", group_by=["department"])
        self.assertEqual([row["group"]["department"] for row in result.rows], ["dA"])

    def test_numeric_group_values_render_deterministically(self) -> None:
        self.build(
            days={"2024-01-01": [
                {"department": "dB", "category": "cX", "amount": 1},
                {"department": "dA", "category": "cX", "amount": 1},
            ]},
        )
        first = generate(self.root, "2024-01-01", "2024-01-02", group_by=["department"]).rows
        second = generate(self.root, "2024-01-01", "2024-01-02", group_by=["department"]).rows
        self.assertEqual(first, second)
        self.assertEqual([r["group"]["department"] for r in first], ["dA", "dB"])

    def test_ledger_value_outside_dictionary_is_an_error(self) -> None:
        self.build(
            days={"2024-01-01": [{"department": "zz", "category": "cX", "amount": 1}]},
        )
        with self.assertRaises(DataError):
            generate(self.root, "2024-01-01", "2024-01-02", group_by=["department"])


if __name__ == "__main__":
    unittest.main()
