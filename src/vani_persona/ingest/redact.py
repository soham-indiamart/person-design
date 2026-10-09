"""PII redaction applied at ingest, before any feature is computed or stored.

Patterns are conservative and India-specific. Redaction runs first so that no
downstream artifact (features, reports, model files) can contain contact details.
"""

from __future__ import annotations

import hashlib
import os
import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    # GSTIN: 2 digits + PAN (5 letters, 4 digits, 1 letter) + 3 chars
    ("GSTIN", re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]{3}\b", re.IGNORECASE)),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)),
    # Indian mobile numbers, optionally prefixed with +91 / 91 / 0, allowing separators
    ("PHONE", re.compile(r"(?:\+?91[\s-]?|0)?[6-9](?:[\s-]?\d){9}\b")),
    # Any remaining long digit run (account numbers, pincodes are 6 digits and kept)
    ("NUMBER", re.compile(r"\b\d{7,}\b")),
]


def redact(text: str) -> tuple[str, int]:
    """Return (redacted_text, number_of_redactions)."""
    count = 0
    for label, pattern in _PATTERNS:
        text, n = pattern.subn(f"<{label}>", text)
        count += n
    return text, count


def pseudonymize(raw_id: str | None, salt: str | None = None) -> str | None:
    """Salted SHA-256 of a seller identifier. The salt must be kept out of the repo."""
    if raw_id is None or str(raw_id).strip() == "":
        return None
    salt = salt if salt is not None else os.environ.get("VANI_PSEUDONYM_SALT", "")
    return hashlib.sha256(f"{salt}:{raw_id}".encode()).hexdigest()[:16]
