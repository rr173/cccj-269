"""Standalone archive CLI alias: ``python -m procurement_report.archive_cli``."""

from __future__ import annotations

from .archive import *  # noqa: F401,F403
from .archive import archive_main, archive_range  # noqa: F401

main = archive_main

if __name__ == "__main__":
    raise SystemExit(archive_main())
