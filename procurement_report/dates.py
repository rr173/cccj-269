"""Strict date helpers.  Only whole calendar days in ISO form are accepted."""

from __future__ import annotations

import re
from datetime import date

from .model import ConfigError

_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_day(value: object, where: str) -> date:
    if not isinstance(value, str) or not _ISO_DAY.match(value):
        raise ConfigError(f"{where}: expected an ISO calendar day YYYY-MM-DD, got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"{where}: invalid calendar day {value!r}") from exc


def iso(value: date) -> str:
    return value.isoformat()
