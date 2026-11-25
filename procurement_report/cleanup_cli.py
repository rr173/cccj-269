"""Standalone cleanup CLI: ``python -m procurement_report.cleanup``.

Also usable as ``python -m procurement_report cleanup ...``.
"""

from __future__ import annotations

from .cleanup import *  # noqa: F401,F403
from .cleanup import cleanup_main, plan_cleanup  # noqa: F401

main = cleanup_main

if __name__ == "__main__":
    raise SystemExit(cleanup_main())
