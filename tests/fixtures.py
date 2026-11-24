"""Fixture builder for tests and examples.

Writes a local-directory simulation of the procurement warehouse:

    <root>/catalog.json
    <root>/ledger/<day>.jsonl
    <root>/summaries/<slice>.json

Digests are computed with the library's own canonical scheme.  Tests can
tamper with a slice file after building to simulate a corrupt/missing slice.
"""

from __future__ import annotations

import json
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from procurement_report.digest import compute_digest


def day_list(start: str, end: str) -> list[str]:
    """Whole days in the half-open interval [start, end)."""

    cur, stop = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while cur < stop:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_dataset(
    root: str | Path,
    *,
    ledger_version: str = "v1",
    ledger_start: str,
    ledger_end: str,
    departments: Sequence[str] = ("dA", "dB"),
    categories: Sequence[str] = ("cX", "cY"),
    days: Mapping[str, Any] | None = None,
    summaries: Iterable[Mapping[str, Any]] = (),
    write_domains: bool = True,
) -> Path:
    """Build a dataset directory.

    ``days`` maps an ISO day to either:

    * ``True``  – readable ledger with no records (empty file);
    * ``False`` – catalogued but unreadable;
    * a list of ledger rows (dicts) – readable with those rows;
    * ``None``  – no catalogue entry at all for that day (the day is within
      the declared range but has no status record);
    * omitted from the mapping – catalogued but unreadable.

    ``summaries`` entries support:

    * ``slice_id``, ``start``, ``end``, ``group_dims``, ``groups``;
    * ``version`` (defaults to the dataset version) and ``complete`` (default True);
    * ``file_payload`` – replace the on-disk payload (defaults are derived);
    * ``drop_file`` – write the catalogue entry but no summary file;
    * ``digest`` – override the catalogue digest.

    Returns ``root`` as a Path.
    """

    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    (root / "ledger").mkdir(parents=True)
    (root / "summaries").mkdir(parents=True)

    day_entries: list[dict[str, Any]] = []
    for day in day_list(ledger_start, ledger_end):
        if days is None:
            spec: Any = False
        elif day not in days:
            spec = False
        else:
            spec = days[day]
            if spec is None:
                # Explicitly no catalogue entry for this day.
                continue
        rel = f"ledger/{day}.jsonl"
        path = root / rel
        if isinstance(spec, bool):
            if spec:
                path.write_text("", encoding="utf-8")
            day_entries.append(
                {"day": day, "readable": spec, "status": "verified" if spec else "purged", "file": rel}
            )
        else:
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in spec),
                encoding="utf-8",
            )
            day_entries.append({"day": day, "readable": True, "status": "verified", "file": rel})

    dimensions: dict[str, Any] = {"order": ["department", "category"]}
    if write_domains:
        dimensions["domains"] = {
            "department": list(departments),
            "category": list(categories),
        }

    summary_entries: list[dict[str, Any]] = []
    for spec in summaries:
        slice_id = spec["slice_id"]
        rel = spec.get("file", f"summaries/{slice_id}.json")
        payload: dict[str, Any] = spec.get(
            "file_payload",
            {
                "slice_id": slice_id,
                "start": spec["start"],
                "end": spec["end"],
                "group_dims": list(spec["group_dims"]),
                "ledger_version": spec.get("version", ledger_version),
                "complete": spec.get("complete", True),
                "groups": list(spec.get("groups", [])),
            },
        )
        payload = dict(payload)
        if "digest" not in payload:
            payload["digest"] = compute_digest(payload)
        if not spec.get("drop_file"):
            write_json(root / rel, payload)

        summary_entries.append(
            {
                "slice_id": slice_id,
                "file": rel,
                "start": spec["start"],
                "end": spec["end"],
                "group_dims": list(spec["group_dims"]),
                "ledger_version": spec.get("catalog_version", spec.get("version", ledger_version)),
                "complete": spec.get("catalog_complete", spec.get("complete", True)),
                "digest": spec.get("digest", payload["digest"]),
            }
        )

    write_json(
        root / "catalog.json",
        {
            "ledger_version": ledger_version,
            "dimensions": dimensions,
            "ledger": {"start": ledger_start, "end": ledger_end, "days": day_entries},
            "summaries": summary_entries,
        },
    )
    return root
