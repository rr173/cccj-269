#!/usr/bin/env python3
"""Root-level convenience wrapper for local archiving.

Equivalent to ``python -m procurement_report.archive ...``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from procurement_report.archive import archive_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(archive_main())
