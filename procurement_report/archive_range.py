"""Compatibility module alias for :mod:`procurement_report.archive`.

Allows both ``from procurement_report.archive_range import archive_range``
and ``python -m procurement_report.archive_range ...``.
"""

from __future__ import annotations

from .archive import *  # noqa: F401,F403
from .archive import archive_main, archive_range  # noqa: F401

main = archive_main

if __name__ == "__main__":
    raise SystemExit(archive_main())
