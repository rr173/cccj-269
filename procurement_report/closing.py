"""Compatibility alias for :mod:`procurement_report.archive` ("close the books")."""

from __future__ import annotations

from .archive import *  # noqa: F401,F403
from .archive import archive_main, archive_range  # noqa: F401

main = archive_main
close_range = archive_range

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(archive_main())
