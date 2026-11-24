"""Canonical, content-addressed sealing for summary slices.

A slice's digest is the SHA-256 of the canonical JSON of its payload with the
``digest`` field removed.  Canonical form: UTF-8, sorted keys, no insignificant
whitespace, non-ASCII kept as UTF-8.  This makes the digest independent of
formatting and of the digest field's own value.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DIGEST_PREFIX = "sha256:"


def canonical_payload(payload: dict[str, Any]) -> bytes:
    body = {key: value for key, value in payload.items() if key != "digest"}
    return json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def compute_digest(payload: dict[str, Any]) -> str:
    return DIGEST_PREFIX + hashlib.sha256(canonical_payload(payload)).hexdigest()


def digest_matches(payload: dict[str, Any], expected: str) -> bool:
    if not isinstance(expected, str) or not expected.startswith(DIGEST_PREFIX):
        return False
    return compute_digest(payload) == expected


def seal_file(path: str | Path) -> str:
    """Compute the digest over a summary file and write the digest back.

    Used by the example/test fixture builders.  Production catalogues are
    expected to carry the digest out of band (in ``catalog.json``).
    """

    file_path = Path(path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("summary file must contain a JSON object")
    digest = compute_digest(payload)
    payload["digest"] = digest
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return digest
