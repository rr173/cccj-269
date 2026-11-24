"""Lazy verification of summary-slice files.

Verification is performed only for slices the planner might use, and its cost
is explicitly *not* a ledger read: reading a summary file is free.  There is no
side effect on the read budget.

A slice file is usable only if all of the following hold:

* the file exists, is valid JSON and matches the catalogue entry structurally;
* its declared interval, grouping dimensions, ledger version, completeness and
  digest equal the catalogue's;
* the stored digest matches the digest recomputed from the payload;
* every group row has valid values, counts and amount relations;
* no group values are duplicated (same group twice would allow double
  counting).

Absent groups are meaningful: because the catalogue declares the slice
complete for its whole interval, a group that does not appear in the file had
no records.  The file is never asked to enumerate empty groups, and a group
with ``record_count == 0`` is simply a no-op.

Any failure makes that slice unusable.  It does not abort the run.
"""

from __future__ import annotations

import json
from typing import Any

from .dates import parse_day
from .digest import digest_matches
from .model import Dataset, SummaryGroup, SummarySlice, is_dimension_value

_REQUIRED_FILE_KEYS = {
    "slice_id",
    "start",
    "end",
    "group_dims",
    "ledger_version",
    "complete",
    "groups",
    "digest",
}
_GROUP_KEYS = {"values", "record_count", "amount_nonnull_count", "amount_sum"}


class SliceVerifier:
    """Caches verification results so each slice file is read at most once."""

    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._cache: dict[str, tuple[SummarySlice, str | None]] = {}

    def checked(self, slice_: SummarySlice) -> tuple[SummarySlice, str | None]:
        """Return ``(slice, error)``.  ``error`` is None when usable."""

        if slice_.slice_id in self._cache:
            return self._cache[slice_.slice_id]
        result = self._verify(slice_)
        self._cache[slice_.slice_id] = result
        return result

    # -- internals ---------------------------------------------------------

    def _verify(self, slice_: SummarySlice) -> tuple[SummarySlice, str | None]:
        if not slice_.valid:
            return slice_, slice_.invalid_reason or "invalid catalogue entry"

        try:
            text = slice_.file.read_text(encoding="utf-8")
        except OSError as exc:
            return slice_, f"summary file unreadable: {exc.strerror or exc}"
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return slice_, f"summary file is not valid JSON: {exc.msg}"

        if not isinstance(payload, dict):
            return slice_, "summary file must contain a JSON object"
        missing = _REQUIRED_FILE_KEYS - payload.keys()
        if missing:
            return slice_, f"summary file missing key(s): {', '.join(sorted(missing))}"

        if payload["slice_id"] != slice_.slice_id:
            return slice_, "slice_id does not match the catalogue"
        try:
            start = parse_day(payload["start"], "summary.start")
            end = parse_day(payload["end"], "summary.end")
        except Exception as exc:  # ConfigError subclasses ValueError
            return slice_, f"invalid summary interval: {exc}"
        if (start, end) != (slice_.start, slice_.end):
            return slice_, "summary interval does not match the catalogue"
        if start >= end:
            return slice_, "summary interval must contain at least one whole day"

        raw_dims = payload["group_dims"]
        if (
            not isinstance(raw_dims, list)
            or any(not isinstance(d, str) or not d for d in raw_dims)
            or len(set(raw_dims)) != len(raw_dims)
        ):
            return slice_, "summary group_dims is malformed"
        # The catalogue fixes *which* dimensions are retained; the file's
        # order only fixes the order of each group's ``values`` array.
        if set(raw_dims) != set(slice_.group_dims):
            return slice_, "summary group_dims does not match the catalogue"
        group_dims = tuple(raw_dims)

        if payload["ledger_version"] != slice_.ledger_version:
            return slice_, "summary ledger_version does not match the catalogue"
        if not isinstance(payload["complete"], bool) or payload["complete"] != slice_.complete:
            return slice_, "summary complete flag does not match the catalogue"
        if not slice_.complete:
            return slice_, "slice is not declared complete"

        if not digest_matches(payload, slice_.digest):
            return slice_, "summary digest does not match its contents"
        # If the file carries its own digest field it must agree with the
        # catalogue; a differing field signals a substituted or stale file.
        if "digest" in payload and payload["digest"] != slice_.digest:
            return slice_, "summary file digest field disagrees with the catalogue"

        if not isinstance(payload["groups"], list):
            return slice_, "summary groups must be a list"

        groups: list[SummaryGroup] = []
        seen: set[tuple[Any, ...]] = set()
        width = len(group_dims)
        for i, raw in enumerate(payload["groups"]):
            where = f"summary groups[{i}]"
            if not isinstance(raw, dict):
                return slice_, f"{where}: expected an object"
            missing = _GROUP_KEYS - raw.keys()
            if missing:
                return slice_, f"{where}: missing key(s): {', '.join(sorted(missing))}"
            values = raw["values"]
            if not isinstance(values, list) or len(values) != width:
                return slice_, f"{where}: values must have {width} entries"
            for j, value in enumerate(values):
                if not is_dimension_value(value):
                    return slice_, f"{where}: values[{j}] must be a non-null string or integer"
                if not self._dataset.dimensions.value_allowed(group_dims[j], value):
                    return slice_, f"{where}: values[{j}]={value!r} is not in the dimension dictionary"
            key = tuple(values)
            if key in seen:
                return slice_, f"{where}: duplicate group {key!r}"
            seen.add(key)

            counts = (raw["record_count"], raw["amount_nonnull_count"], raw["amount_sum"])
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in counts):
                return slice_, f"{where}: counts and amount_sum must be integers"
            record_count, nonnull, total = counts
            if record_count < 0:
                return slice_, f"{where}: record_count must be non-negative"
            if nonnull < 0:
                return slice_, f"{where}: amount_nonnull_count must be non-negative"
            if nonnull > record_count:
                return slice_, f"{where}: amount_nonnull_count must not exceed record_count"
            if nonnull == 0 and total != 0:
                return slice_, f"{where}: amount_sum must be 0 when amount_nonnull_count is 0"
            groups.append(SummaryGroup(key, record_count, nonnull, total))

        result = SummarySlice(
            slice_id=slice_.slice_id,
            file=slice_.file,
            start=slice_.start,
            end=slice_.end,
            group_dims=group_dims,
            ledger_version=slice_.ledger_version,
            complete=slice_.complete,
            digest=slice_.digest,
            valid=True,
            invalid_reason=None,
            groups=tuple(groups),
        )
        return result, None
