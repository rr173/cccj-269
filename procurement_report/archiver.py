"""Compatibility alias for :mod:`procurement_report.archive`."""

from __future__ import annotations

from .archive import *  # noqa: F401,F403
from .archive import archive_main, archive_range  # noqa: F401

main = archive_main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(archive_main())
