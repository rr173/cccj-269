"""End-to-end tests for the offline procurement report tool.

Run with:  python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from procurement_report import DataError, generate

from tests.fixtures import build_dataset
from tests.reference import brute_force_min_reads, direct_recompute


def rows(*specs: tuple[str, str, int | None]) -> list[dict]:
    return [{"department": a, "category": b, "amount": c} for a, b, c in specs]


def group(values: tuple, record_count: int, nonnull: int, total: int) -> dict:
    return {
        "values": list(values),
        "record_count": record_count,
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
        kwargs.setdefault("ledger_end", "2024-01-16")
        return build_dataset(self.root, **kwargs)

    def catalog(self) -> dict:
        return json.loads((self.root / "catalog.json").read_text(encoding="utf-8"))

    def row_map(self, result) -> dict[str, dict]:
        out = {}
        for row in result.rows:
            key = json.dumps(
                [row["group"][dim] for dim in result.query.group_by], sort_keys=True
            )
            out[key] = row
        return out


# ---------------------------------------------------------------------------
# Coverage and minimality


class CoverageTests(Base):
    def test_picks_whole_summary_slice_over_ledger(self) -> None:
        # 12 days; two adjacent complete slices cover [1,7) and [7,13);
        # ledgers exist for all days.  Reading only the 3 tail days is optimal.
        self.build(
            days={day: rows(("dA", "cX", 100)) for day in ("2024-01-14", "2024-01-15")},
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-07",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 6, 6, 600)],
                },
                {
                    "slice_id": "s2",
                    "start": "2024-01-07",
                    "end": "2024-01-13",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 6, 6, 600)],
                },
            ],
        )
        result = generate(self.root, "2024-01-01", "2024-01-13", group_by=["department"])
        self.assertTrue(result.feasible)
        self.assertEqual(result.ledger_days_read, 0)
        self.assertEqual(result.summary_slices_used, ["s1", "s2"])
        self.assertEqual(
            self.row_map(result)[json.dumps(["dA"], sort_keys=True)]["amount_sum"],
            1200,
        )

    def test_overlapping_slices_do_not_permit_netting(self) -> None:
        # s1=[1,6), s2=[4,10), ledger unreadable from 6..9.  s1 and s2 overlap
        # so they cannot both be used, and ledger days 6..9 are unavailable:
        # no plan.  Farthest boundary is 6 (i.e. after 2024-01-05).
        self.build(
            days={"2024-01-01": rows(("dA", "cX", 1))},
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-06",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 5, 5, 5)],
                },
                {
                    "slice_id": "s2",
                    "start": "2024-01-04",
                    "end": "2024-01-10",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 6, 6, 6)],
                },
            ],
        )
        result = generate(self.root, "2024-01-01", "2024-01-10", group_by=["department"])
        self.assertFalse(result.feasible)
        self.assertEqual(result.reachable_boundary, "2024-01-06")
        self.assertEqual(result.last_covered_day, "2024-01-05")
        self.assertEqual(result.rows, [])
        self.assertIn("cannot continue", result.failure_reason)

    def test_empty_cover_reaches_left_endpoint(self) -> None:
        self.build(days={})
        result = generate(self.root, "2024-01-01", "2024-01-05", group_by=["department"])
        self.assertFalse(result.feasible)
        self.assertEqual(result.reachable_boundary, "2024-01-01")
        self.assertIsNone(result.last_covered_day)

    def test_minimum_reads_beats_greedy_longest_slice(self) -> None:
        # A long slice [1,10) is unusable (wrong grouping dims for filters);
        # short slices cover [1,7) and [7,10) leaving only the tail unreadable.
        # This checks the DP does not greedily force one big slice.
        self.build(
            days={},
            summaries=[
                {
                    "slice_id": "big",
                    "start": "2024-01-01",
                    "end": "2024-01-11",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 10, 10, 100)],
                },
                {
                    "slice_id": "a",
                    "start": "2024-01-03",
                    "end": "2024-01-07",
                    "group_dims": ["department", "category"],
                    "groups": [group(("dA", "cX"), 4, 4, 40)],
                },
                {
                    "slice_id": "b",
                    "start": "2024-01-07",
                    "end": "2024-01-11",
                    "group_dims": ["department", "category"],
                    "groups": [group(("dA", "cX"), 4, 4, 40)],
                },
            ],
        )
        # filter includes category -> only a/b are compatible; they cover
        # [Jan 3, Jan 11) and the two leading days need ledger, which is
        # unreadable -> no plan from the left except the empty cover.
        result = generate(
            self.root,
            "2024-01-01",
            "2024-01-11",
            filters={"category": ["cX"]},
            group_by=["category"],
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.reachable_boundary, "2024-01-01")

    def test_tie_selects_a_valid_minimum(self) -> None:
        # Two alternative single-slice covers, both zero ledger reads.
        self.build(
            days={},
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-06",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 5, 5, 5)],
                },
                {
                    "slice_id": "s2",
                    "start": "2024-01-01",
                    "end": "2024-01-06",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 5, 5, 5)],
                },
            ],
        )
        result = generate(self.root, "2024-01-01", "2024-01-06", group_by=["department"])
        self.assertTrue(result.feasible)
        self.assertEqual(result.ledger_days_read, 0)
        self.assertIn(result.summary_slices_used, (["s1"], ["s2"]))

    def test_exhaustive_minimum_matches_reference(self) -> None:
        days_spec = {
            "2024-01-04": rows(("dA", "cX", 7)),
            "2024-01-09": rows(("dA", "cX", 9)),
        }
        self.build(
            days=days_spec,
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-05",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 4, 4, 4)],
                },
                {
                    "slice_id": "s2",
                    "start": "2024-01-05",
                    "end": "2024-01-10",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 5, 5, 5)],
                },
                {
                    "slice_id": "s3",
                    "start": "2024-01-03",
                    "end": "2024-01-09",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 6, 6, 6)],
                },
            ],
        )
        query = {"start": "2024-01-01", "end": "2024-01-10", "group_by": []}
        expected = brute_force_min_reads(self.catalog(), self.root, query)
        result = generate(self.root, query["start"], query["end"])
        self.assertTrue(result.feasible)
        self.assertEqual(result.ledger_days_read, expected)
        self.assertEqual(expected, 0)


# ---------------------------------------------------------------------------
# Slice legality


class SliceLegalityTests(Base):
    def test_slice_must_lie_inside_query(self) -> None:
        self.build(
            days={},
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-06",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 5, 5, 5)],
                }
            ],
        )
        result = generate(self.root, "2024-01-02", "2024-01-06", group_by=["department"])
        self.assertFalse(result.feasible)

    def test_filter_dimension_must_be_retained_even_for_full_whitelist(self) -> None:
        # The slice groups only by department.  A category filter containing
        # every dictionary value still means category was "touched" and must
        # be retained, so the slice is not usable.
        self.build(
            days={},
            departments=("dA",),
            categories=("cX", "cY"),
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-06",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 5, 5, 5)],
                }
            ],
        )
        without = generate(self.root, "2024-01-01", "2024-01-06", group_by=["department"])
        self.assertTrue(without.feasible)
        with_filter = generate(
            self.root,
            "2024-01-01",
            "2024-01-06",
            filters={"category": ["cX", "cY"]},
            group_by=["department"],
        )
        self.assertFalse(with_filter.feasible)

    def test_group_dimension_must_be_retained(self) -> None:
        self.build(
            days={},
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-06",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 5, 5, 5)],
                }
            ],
        )
        result = generate(self.root, "2024-01-01", "2024-01-06", group_by=["category"])
        self.assertFalse(result.feasible)

    def test_wrong_ledger_version_is_unusable(self) -> None:
        self.build(
            days={},
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-06",
                    "group_dims": ["department"],
                    "version": "v0",
                    "groups": [group(("dA",), 5, 5, 5)],
                }
            ],
        )
        result = generate(self.root, "2024-01-01", "2024-01-06", group_by=["department"])
        self.assertFalse(result.feasible)
        self.assertIn("s1", [r["slice_id"] for r in result.rejected_slices])

    def test_bad_unused_slice_does_not_break_valid_plan(self) -> None:
        # s_bad is outside the query (or incompatible); its file is missing.
        # It must not be verified, and the plan must succeed via ledger days.
        self.build(
            days={
                "2024-01-01": rows(("dA", "cX", 3)),
                **{day: [] for day in ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06")},
            },
            summaries=[
                {
                    "slice_id": "s_bad",
                    "start": "2024-01-10",
                    "end": "2024-01-12",
                    "group_dims": ["department"],
                    "drop_file": True,
                }
            ],
        )
        result = generate(self.root, "2024-01-01", "2024-01-06", group_by=["department"])
        self.assertTrue(result.feasible)
        self.assertEqual(result.ledger_days_read, 5)
        self.assertNotIn("s_bad", [r["slice_id"] for r in result.rejected_slices])

    def test_used_slice_with_missing_file_is_rejected(self) -> None:
        self.build(
            days={},
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-06",
                    "group_dims": ["department"],
                    "drop_file": True,
                }
            ],
        )
        result = generate(self.root, "2024-01-01", "2024-01-06", group_by=["department"])
        self.assertFalse(result.feasible)
        self.assertTrue(any("unreadable" in r["reason"] for r in result.rejected_slices))

    def test_duplicate_group_makes_slice_unusable(self) -> None:
        self.build(
            days={},
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-06",
                    "group_dims": ["department"],
                    "groups": [
                        group(("dA",), 3, 3, 30),
                        group(("dA",), 2, 2, 20),
                    ],
                }
            ],
        )
        result = generate(self.root, "2024-01-01", "2024-01-06", group_by=["department"])
        self.assertFalse(result.feasible)
        self.assertTrue(any("duplicate group" in r["reason"] for r in result.rejected_slices))

    def test_illegal_count_relations_rejected(self) -> None:
        for bad in (
            group(("dA",), 2, 5, 50),  # nonnull > records
            group(("dA",), 2, 0, 9),  # zero nonnull but nonzero sum
            group(("dA",), -1, 0, 0),  # negative records
        ):
            with self.subTest(bad=bad):
                self.build(
                    days={},
                    summaries=[
                        {
                            "slice_id": "s1",
                            "start": "2024-01-01",
                            "end": "2024-01-06",
                            "group_dims": ["department"],
                            "groups": [bad],
                        }
                    ],
                )
                result = generate(self.root, "2024-01-01", "2024-01-06", group_by=["department"])
                self.assertFalse(result.feasible)

    def test_digest_mismatch_rejected(self) -> None:
        self.build(
            days={},
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-06",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 5, 5, 5)],
                }
            ],
        )
        path = self.root / "summaries" / "s1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["groups"][0]["amount_sum"] = 999
        path.write_text(json.dumps(payload), encoding="utf-8")
        result = generate(self.root, "2024-01-01", "2024-01-06", group_by=["department"])
        self.assertFalse(result.feasible)
        self.assertTrue(any("digest" in r["reason"] for r in result.rejected_slices))


# ---------------------------------------------------------------------------
# Exactness and averages


class ExactnessTests(Base):
    def test_null_amounts_negative_and_big_sums(self) -> None:
        big = 2**70 + 12345
        self.build(
            days={
                "2024-01-01": rows(
                    ("dA", "cX", big),
                    ("dA", "cX", -big + 8),
                    ("dA", "cY", None),
                ),
                "2024-01-02": rows(("dA", "cX", None), ("dA", "cX", 4)),
            },
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-02",
                    "end": "2024-01-03",
                    "group_dims": ["department", "category"],
                    "groups": [group(("dA", "cX"), 2, 1, 4)],
                }
            ],
        )
        result = generate(
            self.root, "2024-01-01", "2024-01-03", group_by=["department", "category"]
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.ledger_days_read, 1)
        truth = direct_recompute(
            self.catalog(),
            self.root,
            {"start": "2024-01-01", "end": "2024-01-03", "group_by": ["department", "category"]},
        )
        got = self.row_map(result)
        self.assertEqual(len(got), len(truth))
        for key, expected in truth.items():
            row = got[key]
            self.assertEqual(row["record_count"], expected["record_count"])
            self.assertEqual(row["amount_nonnull_count"], expected["amount_nonnull_count"])
            self.assertEqual(row["amount_sum"], expected["amount_sum"])

    def test_group_with_only_null_amounts_has_no_average(self) -> None:
        self.build(days={"2024-01-01": rows(("dA", "cX", None))})
        result = generate(self.root, "2024-01-01", "2024-01-02", group_by=["department"])
        self.assertEqual(result.rows[0]["amount_nonnull_count"], 0)
        self.assertIsNone(result.rows[0]["average_amount"])

    def test_empty_result_is_zero_rows(self) -> None:
        self.build(days={"2024-01-01": []})
        result = generate(self.root, "2024-01-01", "2024-01-02", group_by=["department"])
        self.assertTrue(result.feasible)
        self.assertEqual(result.rows, [])
        self.assertEqual(result.to_json()["row_count"], 0)

    def test_average_is_not_mean_of_slice_averages(self) -> None:
        # Slice A: 1 value 10 (avg 10); Slice B: 3 values 100 (avg 100).
        # True average is 310/4 = 77.5, not (10+100)/2 = 55.
        self.build(
            days={},
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-02",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 1, 1, 10)],
                },
                {
                    "slice_id": "s2",
                    "start": "2024-01-02",
                    "end": "2024-01-05",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 3, 3, 300)],
                },
            ],
        )
        result = generate(self.root, "2024-01-01", "2024-01-05", group_by=["department"])
        row = result.rows[0]
        self.assertEqual(row["amount_sum"], 310)
        self.assertEqual(row["amount_nonnull_count"], 4)
        self.assertEqual(row["average_amount_exact"], "155/2")
        self.assertEqual(row["average_amount"], 77.5)

    def test_filters_are_applied_to_slices(self) -> None:
        self.build(
            days={},
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-06",
                    "group_dims": ["department", "category"],
                    "groups": [
                        group(("dA", "cX"), 3, 3, 30),
                        group(("dA", "cY"), 2, 2, 20),
                        group(("dB", "cX"), 5, 5, 50),
                    ],
                }
            ],
        )
        result = generate(
            self.root,
            "2024-01-01",
            "2024-01-06",
            filters={"category": ["cX"]},
            group_by=["department"],
        )
        got = {row["group"]["department"]: row for row in result.rows}
        self.assertEqual(got["dA"]["record_count"], 3)
        self.assertEqual(got["dB"]["record_count"], 5)
        self.assertEqual(set(got), {"dA", "dB"})


# ---------------------------------------------------------------------------
# Cost discipline


class CostDisciplineTests(Base):
    def test_unselected_ledger_files_are_not_read(self) -> None:
        # Catalogued readable but unreadable on disk: if the planner opened it,
        # the query would raise DataError.  It must never be touched.
        self.build(
            days={
                "2024-01-01": rows(("dA", "cX", 1)),
                "2024-01-02": rows(("dA", "cX", 1)),
            },
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-03",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 2, 2, 2)],
                }
            ],
        )
        (self.root / "ledger" / "2024-01-01.jsonl").write_text("{corrupt", encoding="utf-8")
        (self.root / "ledger" / "2024-01-02.jsonl").write_text("{corrupt", encoding="utf-8")
        result = generate(self.root, "2024-01-01", "2024-01-03", group_by=["department"])
        self.assertTrue(result.feasible)
        self.assertEqual(result.ledger_days_read, 0)

    def test_missing_readable_ledger_is_an_error_not_zero(self) -> None:
        self.build(days={"2024-01-01": rows(("dA", "cX", 1))})
        (self.root / "ledger" / "2024-01-01.jsonl").unlink()
        with self.assertRaises(DataError):
            generate(self.root, "2024-01-01", "2024-01-02", group_by=["department"])

    def test_unreadable_status_is_not_zero(self) -> None:
        self.build(days={"2024-01-01": False})
        result = generate(self.root, "2024-01-01", "2024-01-02", group_by=["department"])
        self.assertFalse(result.feasible)
        self.assertEqual(result.reachable_boundary, "2024-01-01")


if __name__ == "__main__":
    unittest.main()
