"""Stdlib-only secret redaction + value truncation for transcript lines."""

from typing import Any


def redact(value: Any) -> Any:
    """Return value unchanged (skeleton — real redaction logic not yet implemented)."""
    return value


def truncate(value: Any, max_len: int = 8192) -> Any:
    """Return value unchanged (skeleton — real truncation logic not yet implemented)."""
    return value
