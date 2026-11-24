"""Read ledger days and fold them into query-level groups.

A ledger file is JSON Lines, one row per record::

    {"department": "d1", "category": "c1", "amount": 125}
    {"department": "d1", "category": "c2", "amount": null}

Rules enforced here:

* ``department`` and ``category`` are present, non-null dimension values from
  the fixed dictionary;
* ``amount`` is present, and is either ``null`` or an integer (may be
  negative and may exceed 64 bits);
* extra row keys are ignored: they cannot affect the two dimensions or the
  amount, and requiring an exact schema would turn harmless additions into
  hard failures;
* a malformed file fails the whole read.  There is no partial reading: a day
  is either fully consumed or the query fails with ``DataError``.  A missing
  file on a day catalogued as readable is an error, never zero rows.  An
  absence of a file is never interpreted as "no records".
"""

from __future__ import annotations

import json
from math import gcd
from typing import Any, Iterable, Mapping

from .model import DataError, Dataset, LedgerDay, LedgerRow, Query, is_dimension_value

_ROW_KEYS = {"department", "category", "amount"}


def exact_average(total: int, count: int) -> int | str:
    """Exact mean as an int when divisible, else a reduced ``"num/den"`` string.

    Floats are not used: sums may exceed 2**53 (and 2**63), and the report is
    required to be exact.
    """

    if count <= 0:
        raise ValueError("average is undefined for an empty population")
    common = gcd(abs(total), count)
    numerator, denominator = total // common, count // common
    if denominator == 1:
        return numerator
    return f"{numerator}/{denominator}"


def merge_counts(target: dict[tuple[Any, ...], list[int]], source: Mapping[tuple[Any, ...], Iterable[int]]) -> None:
    """Additively merge ``(records, nonnull, sum)`` triples."""

    for key, triple in source.items():
        records, nonnull, total = triple
        slot = target.get(key)
        if slot is None:
            target[key] = [records, nonnull, total]
        else:
            slot[0] += records
            slot[1] += nonnull
            slot[2] += total


def rows_to_counts(
    rows: Iterable[LedgerRow],
    group_dims: tuple[str, ...],
    query: Query,
    target: dict[tuple[Any, ...], list[int]],
) -> None:
    for row in rows:
        values = {"department": row.department, "category": row.category}
        if not passes(values, query):
            continue
        key = tuple(values[d] for d in group_dims)
        slot = target.get(key)
        if slot is None:
            slot = [0, 0, 0]
            target[key] = slot
        slot[0] += 1
        if row.amount is not None:
            slot[1] += 1
            slot[2] += row.amount


def passes(values: Mapping[str, Any], query: Query) -> bool:
    for dim, allowed in query.filters.items():
        if values[dim] not in allowed:
            return False
    return True


def build_rows(
    counts: Mapping[tuple[Any, ...], Iterable[int]], group_dims: tuple[str, ...]
) -> list[dict[str, Any]]:
    """One output row per group that actually contains records.

    * A group with ``record_count == 0`` is not emitted: an absent group in a
      complete slice, and an explicitly zero group, both mean "no records".
    * When no group has any records, the result is zero rows: that is the
      "no data" case, so no average exists.
    * A group with rows but no non-null amount has ``average_amount = null``
      (and ``average_amount_exact = null``): a mean over zero values is
      undefined, not zero.
    * The mean is derived only from the exact ``amount_sum`` and
      ``amount_nonnull_count``; averages of averages are never combined.

    Two representations are emitted:

    * ``average_amount`` -- a JSON number (float) for convenience.  It can
      lose precision when the sum is very large or the quotient is not
      binary-exact; ``null`` if it overflows.
    * ``average_amount_exact`` -- the exact value as an integer when
      divisible, otherwise a reduced ``"numerator/denominator"`` string.
      This is the authoritative value.  Floats cannot represent sums above
      2**53 exactly, and the report is required to stay exact.
    """

    rows: list[dict[str, Any]] = []
    for key in sorted(counts, key=_sortable_key):
        records, nonnull, total = counts[key]
        if records == 0:
            continue
        if nonnull == 0:
            exact: int | str | None = None
            approximate: float | None = None
        else:
            exact = exact_average(total, nonnull)
            try:
                approximate = total / nonnull
            except OverflowError:
                approximate = None
        rows.append(
            {
                "group": {dim: value for dim, value in zip(group_dims, key)},
                "record_count": records,
                "amount_nonnull_count": nonnull,
                "amount_sum": total,
                "average_amount": approximate,
                "average_amount_exact": exact,
            }
        )
    return rows


def _sortable_key(key: tuple[Any, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((type(v).__name__, str(v)) for v in key)


def read_ledger_day(dataset: Dataset, day: LedgerDay) -> list[LedgerRow]:
    """Read one whole ledger day.  All-or-nothing; no partial results."""

    if not day.readable:
        raise DataError(f"{day.day.isoformat()}: ledger is catalogued as unreadable")
    try:
        text = day.file.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataError(
            f"{day.day.isoformat()}: readable ledger file cannot be read: {exc.strerror or exc}"
        ) from exc

    rows: list[LedgerRow] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        where = f"{day.day.isoformat()}:{lineno}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"{where}: invalid JSON: {exc.msg}") from exc
        if not isinstance(raw, dict):
            raise DataError(f"{where}: expected a JSON object")
        missing = _ROW_KEYS - raw.keys()
        if missing:
            raise DataError(f"{where}: missing key(s): {', '.join(sorted(missing))}")
        amount = raw["amount"]
        if amount is not None and (not isinstance(amount, int) or isinstance(amount, bool)):
            raise DataError(f"{where}: amount must be null or an integer")
        if not is_dimension_value(raw["department"]):
            raise DataError(f"{where}: department must be a non-null string or integer")
        if not is_dimension_value(raw["category"]):
            raise DataError(f"{where}: category must be a non-null string or integer")
        if not dataset.dimensions.value_allowed("department", raw["department"]):
            raise DataError(f"{where}: department {raw['department']!r} is not in the dimension dictionary")
        if not dataset.dimensions.value_allowed("category", raw["category"]):
            raise DataError(f"{where}: category {raw['category']!r} is not in the dimension dictionary")
        rows.append(LedgerRow(raw["department"], raw["category"], amount))
    return rows
