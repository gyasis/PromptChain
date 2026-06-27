"""Stdlib-only secret redaction + value truncation for transcript lines."""

import re
from typing import Any

_REDACTED = "***REDACTED***"

# Key-name pattern: case-insensitive search anywhere in the key name.
# Bias to over-redact (D5): matches partial names too (e.g. "token_count").
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|authorization|bearer)"
)

# Value patterns that identify a string as a secret (searched, not full-match).
_SECRET_VALUE_PATTERNS = [
    # OpenAI-style API keys: sk-<10+ alphanumeric/dash/underscore chars>
    re.compile(r"sk-[A-Za-z0-9_-]{10,}"),
    # Bearer authentication header tokens
    re.compile(r"Bearer\s+\S{4,}"),
    # JSON Web Tokens: eyJ<base64url>.<base64url>.<base64url>
    re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    # Long hexadecimal strings ≥32 chars (SHA hashes, API keys, GUIDs stripped of dashes)
    re.compile(r"[A-Fa-f0-9]{32,}"),
]

# Broad base64-ish character set for long-token detection (≥32 chars).
# Only flagged as secret if the matched token has mixed char classes (see
# _is_mixed_entropy) to avoid masking low-entropy repeated strings like 'y'*5000.
_LONG_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_-]{32,}")


def _is_mixed_entropy(s: str) -> bool:
    """Return True if *s* contains at least one uppercase, one lowercase, and one digit.

    This guards against false-positives on long repeated-character strings
    (e.g. ``'y' * 5000``) that are not secrets.
    """
    return (
        any(c.isupper() for c in s)
        and any(c.islower() for c in s)
        and any(c.isdigit() for c in s)
    )


def _looks_like_secret(s: str) -> bool:
    """Return True if string *s* appears to be a secret credential."""
    for pattern in _SECRET_VALUE_PATTERNS:
        if pattern.search(s):
            return True
    # Long base64-ish token with mixed char classes (upper + lower + digit required)
    m = _LONG_TOKEN_RE.search(s)
    if m and _is_mixed_entropy(m.group(0)):
        return True
    return False


def redact(value: Any) -> Any:
    """Recursively redact secrets from *value*, returning a new structure.

    Two mechanisms (both applied; bias to over-redact per D5):

    * **Key-name match** — for a dict, any key whose name matches
      ``(?i)(api[_-]?key|token|secret|password|authorization|bearer)``
      has its value replaced with ``"***REDACTED***"`` without recursing
      into it.
    * **Value-pattern match** — a string value that looks like a secret
      (``sk-…`` API keys, ``Bearer …`` headers, JWTs, long high-entropy
      hex ≥ 32 chars, or base64-ish ≥ 32 chars with mixed char classes)
      is replaced with ``"***REDACTED***"``.

    Scalars (int, float, bool, None) are returned unchanged.
    """
    if isinstance(value, dict):
        result: dict = {}
        for k, v in value.items():
            if isinstance(k, str) and _SENSITIVE_KEY_RE.search(k):
                result[k] = _REDACTED
            else:
                result[k] = redact(v)
        return result
    if isinstance(value, (list, tuple)):
        processed = [redact(item) for item in value]
        return type(value)(processed)
    if isinstance(value, str):
        return _REDACTED if _looks_like_secret(value) else value
    # int, float, bool, None — pass through unchanged
    return value


def truncate(value: Any, max_len: int = 8192) -> Any:
    """Cap a string *value* at *max_len* characters.

    If *value* is a :class:`str` longer than *max_len*, returns the first
    *max_len* characters followed by ``'…[truncated N chars]'`` where *N* is
    the number of characters removed.  The result is always JSON-serialisable.

    Non-string values are returned unchanged (including dicts and lists —
    the emitter applies truncation per-field before the dict is assembled).
    """
    if isinstance(value, str) and len(value) > max_len:
        removed = len(value) - max_len
        return value[:max_len] + f"…[truncated {removed} chars]"
    return value
