"""Command-line interface.

Example::

    python -m procurement_report.cli /path/to/data \\
        --start 2024-01-01 --end 2024-01-31 \\
        --filter department=dA --filter department=dB \\
        --group-by department --group-by category

The report is printed as JSON.  Exit status is 0 when a report was produced,
2 when no legal cover exists, and 1 for a configuration/read error.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import ConfigError, DataError, generate


def _filters(pairs: list[str]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"--filter expects DIM=VALUE, got {pair!r}")
        dim, raw = pair.split("=", 1)
        dim = dim.strip()
        if not dim:
            raise ConfigError(f"--filter expects DIM=VALUE, got {pair!r}")
        value: Any = raw
        # Allow integer dictionary codes to be expressed without quotes.
        if raw.lstrip("-").isdigit():
            value = int(raw)
        out.setdefault(dim, []).append(value)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="procurement_report",
        description="Offline exact report over summary slices and surviving ledgers.",
    )
    parser.add_argument("data_dir", help="directory containing catalog.json")
    parser.add_argument("--start", required=True, help="query start day (inclusive), YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="query end day (exclusive), YYYY-MM-DD")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="DIM=VALUE",
        help="whitelist filter; repeat for multiple values",
    )
    parser.add_argument(
        "--group-by",
        action="append",
        default=[],
        metavar="DIM",
        help="output grouping dimension; repeat for multiple dimensions",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="indent the JSON output"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = generate(
            args.data_dir,
            args.start,
            args.end,
            filters=_filters(args.filter),
            group_by=args.group_by,
        )
    except (ConfigError, DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = result.to_json()
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if result.feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
