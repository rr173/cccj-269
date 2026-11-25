"""Tests for local retention cleanup planning.

Run with:  python3 -m unittest discover -s tests -t .

The operation must

* only ever *plan*: never delete a ledger, never rewrite the catalogue;
* treat only currently readable ledgers before ``before`` as candidates;
* prove every workload with complete, legal, indivisible slices or ledgers
  that remain readable;
* report a safe plan only when *all* workloads are proven, otherwise the
  earliest unprovable workload plus its boundary;
* maximise the cleanable dates under one joint retained set (no per-workload
  union, no guessing, never treat missing data as zero).
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from procurement_report import (
    ConfigError,
    DataError,
    cleanup_main,
    generate,
    load_catalog,
    plan_cleanup,
)
from procurement_report.cli import main as cli_main

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


def days_between(start: str, end: str) -> list[str]:
    cur, stop = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while cur < stop:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def aggregate(hidden, start: str, end: str, dims) -> list[dict]:
    counts: dict[tuple, list[int]] = {}
    for day in days_between(start, end):
        for record in hidden[day]:
            key = tuple(record[d] for d in dims)
            slot = counts.setdefault(key, [0, 0, 0])
            slot[0] += 1
            if record["amount"] is not None:
                slot[1] += 1
                slot[2] += record["amount"]
    return [group(key, v[0], v[1], v[2]) for key, v in counts.items()]


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.root = self.dir / "data"
        self.workload_dir = self.dir / "workloads"
        self.workload_dir.mkdir()

    def build(self, **kwargs) -> Path:
        kwargs.setdefault("ledger_start", "2024-01-01")
        kwargs.setdefault("ledger_end", "2024-01-05")
        return build_dataset(self.root, **kwargs)

    def workload(self, queries, name: str = "workload.json") -> Path:
        path = self.workload_dir / name
        path.write_text(json.dumps({"queries": queries}), encoding="utf-8")
        return path

    def catalog_bytes(self) -> bytes:
        return (self.root / "catalog.json").read_bytes()


# ---------------------------------------------------------------------------
# Candidate selection


class CandidateTests(Base):
    def test_only_readable_days_before_cutoff_are_candidates(self) -> None:
        self.build(
            days={
                "2024-01-01": rows(("dA", "cX", 1)),
                "2024-01-02": rows(("dA", "cX", 2)),
                "2024-01-03": False,  # catalogued unreadable: never a candidate
                "2024-01-04": rows(("dA", "cX", 4)),
            },
            summaries=[
                {
                    "slice_id": "every",
                    "start": "2024-01-01",
                    "end": "2024-01-04",
                    "group_dims": ["department", "category"],
                    "groups": aggregate(
                        {day: rows(("dA", "cX", 1)) for day in days_between("2024-01-01", "2024-01-04")},
                        "2024-01-01",
                        "2024-01-04",
                        ["department", "category"],
                    ),
                }
            ],
        )
        path = self.workload([{"query_id": "q", "start": "2024-01-01", "end": "2024-01-04"}])
        result = plan_cleanup(self.root, "2024-01-03", path)
        self.assertEqual(result.to_json()["candidate_dates"], ["2024-01-01", "2024-01-02"])

    def test_cutoff_is_exclusive(self) -> None:
        self.build(days={day: [] for day in days_between("2024-01-01", "2024-01-05")})
        path = self.workload([{"query_id": "q", "start": "2024-01-01", "end": "2024-01-05"}])
        result = plan_cleanup(self.root, "2024-01-03", path)
        self.assertEqual(
            result.to_json()["candidate_dates"], ["2024-01-01", "2024-01-02"]
        )

    def test_no_candidates_still_proves_workloads(self) -> None:
        self.build(days={day: [] for day in days_between("2024-01-01", "2024-01-03")})
        path = self.workload([{"query_id": "q", "start": "2024-01-01", "end": "2024-01-03"}])
        result = plan_cleanup(self.root, "2024-01-01", path)
        payload = result.to_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["candidate_dates"], [])
        self.assertEqual(payload["cleanable_dates"], [])


# ---------------------------------------------------------------------------
# Planning safety and maximality


class PlanningTests(Base):
    def test_summary_frees_its_whole_range(self) -> None:
        hidden = {day: rows(("dA", "cX", i)) for i, day in enumerate(days_between("2024-01-01", "2024-01-03"))}
        self.build(
            days=hidden,
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-03",
                    "group_dims": ["department", "category"],
                    "groups": aggregate(hidden, "2024-01-01", "2024-01-03", ["department", "category"]),
                }
            ],
        )
        path = self.workload(
            [{"query_id": "q", "start": "2024-01-01", "end": "2024-01-03", "group_by": ["department"]}]
        )
        result = plan_cleanup(self.root, "2024-01-05", path)
        payload = result.to_json()
        self.assertEqual(payload["cleanable_dates"], ["2024-01-01", "2024-01-02"])
        self.assertEqual(payload["must_retain_dates"], [])
        self.assertEqual(payload["workloads"][0]["summary_slices_used"], ["s1"])

    def test_ledger_only_day_is_retained(self) -> None:
        self.build(
            days={
                "2024-01-01": rows(("dA", "cX", 1)),
                "2024-01-02": rows(("dA", "cX", 2)),
            },
            summaries=[
                {
                    "slice_id": "s1",
                    "start": "2024-01-01",
                    "end": "2024-01-02",
                    "group_dims": ["department", "category"],
                    "groups": [group(("dA", "cX"), 1, 1, 1)],
                }
            ],
        )
        path = self.workload([{"query_id": "q", "start": "2024-01-01", "end": "2024-01-03"}])
        result = plan_cleanup(self.root, "2024-01-05", path)
        payload = result.to_json()
        # Jan 1 is served by a slice; Jan 2 has no slice and must stay readable.
        self.assertEqual(payload["cleanable_dates"], ["2024-01-01"])
        self.assertEqual(payload["must_retain_dates"], ["2024-01-02"])

    def test_joint_retention_not_per_workload_union(self) -> None:
        # W2 (department only) can use the department-only slice; W1 needs the
        # category.  A per-workload "cleanable" union would clean Jan 1 twice
        # and then fail; the joint must retain it for W1.
        self.build(
            days={
                "2024-01-01": rows(("dA", "cX", 1)),
                "2024-01-02": rows(("dA", "cY", 2)),
            },
            summaries=[
                {
                    "slice_id": "dept_jan1",
                    "start": "2024-01-01",
                    "end": "2024-01-02",
                    "group_dims": ["department"],
                    "groups": [group(("dA",), 1, 1, 1)],
                }
            ],
        )
        path = self.workload(
            [
                {
                    "query_id": "needs_category",
                    "start": "2024-01-01",
                    "end": "2024-01-02",
                    "filters": {"category": ["cX"]},
                    "group_by": ["department"],
                },
                {"query_id": "dept_only", "start": "2024-01-01", "end": "2024-01-02", "group_by": ["department"]},
            ]
        )
        result = plan_cleanup(self.root, "2024-01-05", path)
        payload = result.to_json()
        self.assertEqual(payload["must_retain_dates"], ["2024-01-01"])
        self.assertEqual(payload["cleanable_dates"], ["2024-01-02"])

    def test_purged_days_outside_queries_are_cleanable(self) -> None:
        self.build(days={day: [] for day in days_between("2024-01-01", "2024-01-05")})
        path = self.workload([{"query_id": "q", "start": "2024-01-01", "end": "2024-01-03"}])
        result = plan_cleanup(self.root, "2024-01-05", path)
        payload = result.to_json()
        # Jan 1 and Jan 2 are inside the only query and have no slice, so they
        # must stay.  Jan 3 and Jan 4 are outside every query: never read by
        # any workload, hence cleanable.
        self.assertEqual(payload["must_retain_dates"], ["2024-01-01", "2024-01-02"])
        self.assertEqual(payload["cleanable_dates"], ["2024-01-03", "2024-01-04"])

    def test_cleaned_view_matches_documented_semantics(self) -> None:
        # Every cleanable date really can be gone: build a catalogue with
        # those days unreadable and re-prove every workload with the engine.
        self.build(
            days={day: [] for day in days_between("2024-01-01", "2024-01-05")},
        )
        path = self.workload([{"query_id": "q", "start": "2024-01-01", "end": "2024-01-04"}])
        result = plan_cleanup(self.root, "2024-01-05", path)
        payload = result.to_json()
        for day in payload["cleanable_dates"]:
            entry = next(
                e for e in json.loads(self.catalog_bytes())["ledger"]["days"] if e["day"] == day
            )
            entry["readable"] = False
            entry["status"] = "cleaned"
        # Write the would-be cleaned catalogue to a fresh directory and query.
        cleaned_root = self.dir / "cleaned"
        build_dataset(
            cleaned_root,
            ledger_start="2024-01-01",
            ledger_end="2024-01-05",
            days={day: (False if day in payload["cleanable_dates"] else []) for day in days_between("2024-01-01", "2024-01-05")},
        )
        report = generate(cleaned_root, "2024-01-01", "2024-01-04")
        self.assertTrue(report.feasible)


# ---------------------------------------------------------------------------
# Unprovable workloads


class UnprovableTests(Base):
    def test_missing_ledger_is_never_zero(self) -> None:
        self.build(
            days={
                "2024-01-01": rows(("dA", "cX", 1)),
                "2024-01-02": None,  # no catalogue entry at all
                "2024-01-03": rows(("dA", "cX", 3)),
            }
        )
        path = self.workload([{"query_id": "gap", "start": "2024-01-01", "end": "2024-01-03"}])
        result = plan_cleanup(self.root, "2024-01-05", path)
        payload = result.to_json()
        self.assertEqual(payload["status"], "cannot_clean")
        self.assertFalse(payload["can_clean"])
        self.assertEqual(payload["unprovable_workload"]["workload_id"], "gap")
        self.assertEqual(payload["unprovable_workload"]["boundary"], "2024-01-02")
        self.assertEqual(payload["unprovable_workload"]["last_covered_day"], "2024-01-01")

    def test_earliest_unprovable_workload_is_reported(self) -> None:
        self.build(
            days={
                "2024-01-01": rows(("dA", "cX", 1)),
                "2024-01-02": False,
            }
        )
        path = self.workload(
            [
                {"query_id": "fine", "start": "2024-01-01", "end": "2024-01-02"},
                {"query_id": "second_bad", "start": "2024-01-01", "end": "2024-01-03"},
                {"query_id": "third_bad", "start": "2024-01-02", "end": "2024-01-03"},
            ]
        )
        payload = plan_cleanup(self.root, "2024-01-05", path).to_json()
        self.assertEqual(payload["proven_workloads"], ["fine"])
        self.assertEqual(payload["unprovable_workload"]["workload_id"], "second_bad")
        self.assertEqual(payload["unprovable_workload"]["boundary"], "2024-01-02")

    def test_unusable_slice_does_not_help(self) -> None:
        # Wrong-version slice: even though its range would cover the day, it
        # is not usable and cannot prove the workload.
        self.build(
            days={"2024-01-01": rows(("dA", "cX", 1)), "2024-01-02": False},
            summaries=[
                {
                    "slice_id": "bad",
                    "start": "2024-01-02",
                    "end": "2024-01-03",
                    "group_dims": ["department", "category"],
                    "version": "v0",
                    "groups": [group(("dA", "cX"), 1, 1, 1)],
                }
            ],
        )
        path = self.workload([{"query_id": "q", "start": "2024-01-01", "end": "2024-01-03"}])
        payload = plan_cleanup(self.root, "2024-01-05", path).to_json()
        self.assertEqual(payload["status"], "cannot_clean")
        self.assertEqual(payload["unprovable_workload"]["boundary"], "2024-01-02")

    def test_empty_workload_is_trivially_safe(self) -> None:
        self.build(days={day: [] for day in days_between("2024-01-01", "2024-01-04")})
        path = self.workload([])
        payload = plan_cleanup(self.root, "2024-01-05", path).to_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["cleanable_dates"], ["2024-01-01", "2024-01-02", "2024-01-03"])


# ---------------------------------------------------------------------------
# Input validation


class ValidationTests(Base):
    def setUp(self) -> None:
        super().setUp()
        self.build(days={day: [] for day in days_between("2024-01-01", "2024-01-04")})

    def test_workload_inside_data_dir_is_rejected(self) -> None:
        path = self.root / "workload.json"
        path.write_text(json.dumps({"queries": []}), encoding="utf-8")
        with self.assertRaises(ConfigError) as caught:
            plan_cleanup(self.root, "2024-01-05", path)
        self.assertIn("outside the data directory", str(caught.exception))

    def test_missing_workload_file_is_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            plan_cleanup(self.root, "2024-01-05", self.workload_dir / "nope.json")

    def test_bad_workload_shapes_are_config_errors(self) -> None:
        cases = [
            {"queries": [{"start": "2024-01-01"}]},
            {"queries": [{"start": "2024-01-03", "end": "2024-01-01"}]},
            {"queries": [{"start": "2024-01-01", "end": "2024-01-02", "group_by": ["region"]}]},
            {"queries": [{"start": "2024-01-01", "end": "2024-01-02", "filters": {"region": ["x"]}}]},
            {"queries": [{"start": "2024-01-01", "end": "2024-01-02", "extra": 1}]},
            {"queries": [{"start": "2024-01-01", "end": "2024-01-02", "query_id": "a"},
                         {"start": "2024-01-01", "end": "2024-01-02", "query_id": "a"}]},
            {"queries": [], "extra": []},
            {"nothing": []},
        ]
        for i, payload in enumerate(cases):
            with self.subTest(case=i):
                path = self.workload_dir / f"bad{i}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ConfigError):
                    plan_cleanup(self.root, "2024-01-05", path)

    def test_workload_accepts_bare_list(self) -> None:
        path = self.workload_dir / "list.json"
        path.write_text(
            json.dumps([{"start": "2024-01-01", "end": "2024-01-02"}]), encoding="utf-8"
        )
        payload = plan_cleanup(self.root, "2024-01-05", path).to_json()
        self.assertEqual(payload["status"], "ok")

    def test_bad_cutoff_is_config_error(self) -> None:
        path = self.workload([])
        with self.assertRaises(ConfigError):
            plan_cleanup(self.root, "2024-13-01", path)


# ---------------------------------------------------------------------------
# Read-only guarantee


class ReadOnlyTests(Base):
    def test_planning_never_changes_the_directory(self) -> None:
        self.build(
            days={day: rows(("dA", "cX", 1)) for day in days_between("2024-01-01", "2024-01-04")},
            summaries=[],
        )
        path = self.workload([{"query_id": "q", "start": "2024-01-01", "end": "2024-01-03"}])
        before_catalog = self.catalog_bytes()
        before_ledgers = {
            p.name: p.read_bytes() for p in (self.root / "ledger").glob("*.jsonl")
        }
        before_summaries = {
            p.name: p.read_bytes() for p in (self.root / "summaries").glob("*.json")
        }
        plan_cleanup(self.root, "2024-01-05", path)
        self.assertEqual(self.catalog_bytes(), before_catalog)
        self.assertEqual(
            {p.name: p.read_bytes() for p in (self.root / "ledger").glob("*.jsonl")},
            before_ledgers,
        )
        self.assertEqual(
            {p.name: p.read_bytes() for p in (self.root / "summaries").glob("*.json")},
            before_summaries,
        )

    def test_catalog_and_ledger_days_unchanged_after_unprovable(self) -> None:
        self.build(days={"2024-01-01": rows(("dA", "cX", 1)), "2024-01-02": False})
        path = self.workload([{"query_id": "q", "start": "2024-01-01", "end": "2024-01-03"}])
        before = self.catalog_bytes()
        plan_cleanup(self.root, "2024-01-05", path)
        self.assertEqual(self.catalog_bytes(), before)
        self.assertTrue((self.root / "ledger" / "2024-01-01.jsonl").is_file())


# ---------------------------------------------------------------------------
# Command line


class CleanupCliTests(Base):
    def run_cleanup(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = cleanup_main(list(argv))
            except SystemExit as exc:
                code = int(exc.code)
        return code, out.getvalue(), err.getvalue()

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = cli_main(list(argv))
            except SystemExit as exc:
                code = int(exc.code)
        return code, out.getvalue(), err.getvalue()

    def test_cli_ok_exit_zero(self) -> None:
        self.build(days={day: [] for day in days_between("2024-01-01", "2024-01-04")})
        path = self.workload([{"query_id": "q", "start": "2024-01-01", "end": "2024-01-03"}])
        code, out, _ = self.run_cleanup(str(self.root), "--before", "2024-01-05", "--workload", str(path))
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["status"], "ok")

    def test_cli_cannot_clean_exit_two(self) -> None:
        self.build(days={"2024-01-01": False})
        path = self.workload([{"query_id": "q", "start": "2024-01-01", "end": "2024-01-02"}])
        code, out, _ = self.run_cleanup(str(self.root), "--before", "2024-01-05", "--workload", str(path))
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertEqual(payload["status"], "cannot_clean")

    def test_cli_missing_workload_exit_one(self) -> None:
        self.build(days={})
        code, _, err = self.run_cleanup(str(self.root), "--before", "2024-01-05")
        self.assertEqual(code, 1)
        self.assertIn("--workload", err)

    def test_cli_bad_workload_file_exit_one(self) -> None:
        self.build(days={})
        bad = self.workload_dir / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        code, _, err = self.run_cleanup(str(self.root), "--before", "2024-01-05", "--workload", str(bad))
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_subcommand_forms(self) -> None:
        self.build(days={day: [] for day in days_between("2024-01-01", "2024-01-04")})
        path = self.workload([{"query_id": "q", "start": "2024-01-01", "end": "2024-01-03"}])
        code, out, _ = self.run_cli("cleanup", str(self.root), "--before", "2024-01-05", "--workload", str(path))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["status"], "ok")
        code2, out2, _ = self.run_cli(str(self.root), "cleanup", "--before", "2024-01-05", "--workload", str(path))
        self.assertEqual(code2, 0)
        self.assertEqual(json.loads(out2)["status"], "ok")


# ---------------------------------------------------------------------------
# Existing report/budget contracts are untouched


class RegressionTests(Base):
    def test_plain_report_and_budget_still_work(self) -> None:
        self.build(days={"2024-01-01": rows(("dA", "cX", 5))})
        result = generate(self.root, "2024-01-01", "2024-01-02", group_by=["department"])
        self.assertTrue(result.feasible)
        self.assertEqual(result.rows[0]["amount_sum"], 5)


if __name__ == "__main__":
    unittest.main()
