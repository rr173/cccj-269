"""Command-line interface.

Example::

    python -m procurement_report.cli /path/to/data \\
        --start 2024-01-01 --end 2024-01-31 \\
        --filter department=dA --filter department=dB \\
        --group-by department --group-by category

    python -m procurement_report.cli /path/to/data \\
        --start 2024-01-01 --end 2024-01-31 \\
        --budget /path/to/budget_rules.json

With ``--budget`` the tool produces a budget-vs-actual reconciliation instead
of a plain report.  The budget file must live outside the data directory.

The report is printed as JSON.  Exit status is 0 when a report was produced,
2 when no legal cover exists, 3 when a budget reconciliation cannot be
completed (e.g. a source spans a budget boundary and cannot be split), and 1
for a configuration/read error.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import ConfigError, DataError, generate, reconcile
from .archive import (
    ArchiveError,
    archive_main,
    archive_range,
    try_archive_range,
)

#: Leading tokens that select the local archive subcommand.
_ARCHIVE_COMMANDS = ("archive", "close", "archive-range", "close-range")

#: Options whose very next argv token is a value (used for subcommand sniffing).
_VALUE_OPTIONS = {
    "--filter",
    "--budget",
    "--start",
    "--end",
    "--start-date",
    "--end-date",
    "--group-by",
    "--slice-id",
    "--name",
    "--id",
    "--file",
    "--slice-file",
    "--archive-file",
    "--out",
    "--output",
    "--data-dir",
    "--range",
    "--group-dim",
    "--group-dims",
    "--archive-group-dim",
    "--mode",
    "--operation",
}


def _split_archive_subcommand(argv: list[str]) -> list[str] | None:
    """Return archive-CLI argv when an ``archive`` subcommand is present.

    Supports both ``prog archive DATA_DIR ...`` and
    ``prog DATA_DIR archive ...``.  Returns ``None`` when the existing report
    parser should handle the argv unchanged (including the legacy meaning of a
    data directory literally named ``archive`` followed immediately by
    options).
    """

    if not argv:
        return None

    # Leading command form: ``archive <positional> ...``.  When the very next
    # token is an option, the legacy interpretation (data directory literally
    # named "archive") is preserved if such a directory actually exists;
    # otherwise route to the archive parser so options-first forms work too.
    if argv[0] in _ARCHIVE_COMMANDS:
        rest = argv[1:]
        if "--data-dir" in rest:
            return rest
        if rest and not rest[0].startswith("-"):
            return rest
        from pathlib import Path as _Path

        if _Path(argv[0]).is_dir():
            return None
        return rest

    # ``DATA_DIR archive ...`` form (command token not consumed as an option value).
    skip_value = False
    for i, token in enumerate(argv):
        if skip_value:
            skip_value = False
            continue
        if token in _VALUE_OPTIONS:
            skip_value = True
            continue
        if "=" in token and token.split("=", 1)[0] in _VALUE_OPTIONS:
            continue
        if token in _ARCHIVE_COMMANDS and i > 0:
            return argv[:i] + argv[i + 1 :]
    return None


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
    parser.add_argument("data_dir", nargs="?", help="directory containing catalog.json")
    parser.add_argument("--data-dir", dest="data_dir_opt", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--start", "--start-date", required=True, help="query start day (inclusive), YYYY-MM-DD")
    parser.add_argument("--end", "--end-date", required=True, help="query end day (exclusive), YYYY-MM-DD")
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
        "--budget",
        metavar="PATH",
        default=None,
        help=(
            "budget rule file outside the data directory; switches the output to "
            "a per department/category/effective-interval reconciliation"
        ),
    )
    parser.add_argument(
        "--pretty", action="store_true", help="indent the JSON output"
    )
    # Local archiving.  These options are additive: without --archive/--close
    # the parser behaves exactly as before.
    parser.add_argument(
        "--archive",
        "--close",
        "--archive-range",
        "--close-range",
        dest="archive",
        action="store_true",
        help=(
            "archive the fully-readable contiguous range into one complete "
            "closing summary slice instead of running a report"
        ),
    )
    parser.add_argument("--slice-id", dest="slice_id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--archive-file", dest="archive_file", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--archive-group-dim",
        action="append",
        dest="archive_group_dims",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--replace", dest="replace", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode",
        "--operation",
        dest="mode",
        default="report",
        choices=("report", "archive", "close"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mark-unreadable",
        "--purge",
        dest="mark_unreadable",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = list(sys.argv[1:])
    else:
        argv = list(argv)

    # ``procurement_report archive DATA_DIR ...`` subcommand form.
    archive_argv = _split_archive_subcommand(argv)
    if archive_argv is not None:
        from .archive import archive_main

        return archive_main(archive_argv)

    parser = build_parser()
    args = parser.parse_args(argv)

    data_dir = getattr(args, "data_dir_opt", None) or args.data_dir
    if not data_dir:
        parser.error("a data directory is required (positional or --data-dir)")

    if getattr(args, "archive", False) or getattr(args, "mode", "report") in ("archive", "close"):
        from .archive import archive_command

        dims = args.archive_group_dims
        return archive_command(
            data_dir,
            args.start,
            args.end,
            slice_id=args.slice_id,
            group_dims=dims,
            file=args.archive_file,
            replace=bool(getattr(args, "replace", False)),
            mark_unreadable=bool(getattr(args, "mark_unreadable", False)),
            pretty=args.pretty,
        )

    if args.budget is not None:
        if args.group_by:
            parser.error("--group-by is not used with --budget; rows are always department+category")
        try:
            filters = _filters(args.filter)
            budget_result = reconcile(
                data_dir, args.start, args.end, args.budget, filters=filters
            )
        except (ConfigError, DataError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        payload = budget_result.to_json()
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        if budget_result.feasible:
            return 0
        return 3

    try:
        result = generate(
            data_dir,
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
