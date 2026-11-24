"""Allow ``python -m procurement_report ...``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
