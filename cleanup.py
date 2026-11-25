#!/usr/bin/env python3
"""Root-level convenience wrapper for retention cleanup planning.

Equivalent to ``python -m procurement_report.cleanup ...``.  This only plans;
it never deletes a ledger or modifies the data directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from procurement_report.cleanup import cleanup_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(cleanup_main())
