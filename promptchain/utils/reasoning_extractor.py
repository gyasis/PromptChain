"""Reasoning extractor — pull a model's INTERNAL reasoning from an LLM response.

Profile-driven WITH graceful fallback (the ``model_catalog`` design principle):
we try the catalog profile's declared surface first, then fall back to checking
*every* known surface, so an unknown or mis-profiled model still yields its
reasoning when it's actually present (and a plain model yields nothing).

Known surfaces (``profile["extract"]``):
  * ``reasoning_content`` — litellm reasoning models (OpenAI o-series, deepseek
    via cloud) surface ``message.reasoning_content``.
  * ``field``            — ollama deepseek-r1 with ``think=true`` →
    ``message.thinking``.
  * ``think_tag``        — ``<think>…</think>`` embedded in ``message.content``.

The extractor reads from either a dict message or a litellm message object
(incl. ``model_dump()``), and returns the reasoning text plus a cleaned content
string (with any ``<think>`` block removed) so the visible answer stays clean.
"""
import re
from typing import Any, Dict, Optional, Tuple

# <think>…</think> (closed) and an unclosed trailing <think> (streaming/truncated).
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>(.*)\Z", re.DOTALL | re.IGNORECASE)

_REASONING_KEYS = ("reasoning_content", "thinking", "reasoning")


def _get(message: Any, key: str) -> Optional[Any]:
    """Read ``key`` from a dict OR an object (litellm message), incl. model_dump."""
    if message is None:
        return None
    if isinstance(message, dict):
        return message.get(key)
    val = getattr(message, key, None)
    if val is not None:
        return val
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        try:
            d = dump()
            if isinstance(d, dict):
                return d.get(key)
        except Exception:
            return None
    return None


def _content_str(message: Any) -> str:
    c = _get(message, "content")
    if isinstance(c, str):
        return c
    return "" if c is None else str(c)


def _strip_think_tags(content: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (reasoning, cleaned_content) for ``<think>`` content, else (None, None)."""
    if not content or "<think" not in content.lower():
        return None, None
    blocks = _THINK_RE.findall(content)
    if blocks:
        reasoning = "\n".join(b.strip() for b in blocks if b.strip())
        cleaned = _THINK_RE.sub("", content).strip()
        return (reasoning or None), cleaned
    # Unclosed <think> (truncated / mid-stream): treat the remainder as reasoning.
    m = _OPEN_THINK_RE.search(content)
    if m:
        reasoning = m.group(1).strip()
        cleaned = content[: m.start()].strip()
        return (reasoning or None), cleaned
    return None, None


def extract_reasoning(
    message: Any, profile: Optional[Dict] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Extract a model's internal reasoning from an LLM response message.

    Args:
        message: dict or litellm message object (the assistant turn).
        profile: catalog reasoning profile (``{"extract": ..., "tags": ...}``).
                 None / ``extract="none"`` → pure graceful fallback.

    Returns:
        ``(reasoning_text, cleaned_content)``:
          * ``reasoning_text`` — the internal reasoning, or None if absent.
          * ``cleaned_content`` — content with any ``<think>`` block removed,
            or None when the content was unchanged (caller may ignore).
    """
    profile = profile or {}
    extract = profile.get("extract", "none")

    # 1) Profile-preferred surface first.
    if extract == "reasoning_content":
        rc = _get(message, "reasoning_content")
        if isinstance(rc, str) and rc.strip():
            return rc.strip(), None
    elif extract == "field":
        for k in ("thinking", "reasoning"):
            v = _get(message, k)
            if isinstance(v, str) and v.strip():
                return v.strip(), None
    elif extract == "think_tag":
        r, cleaned = _strip_think_tags(_content_str(message))
        if r:
            return r, cleaned

    # 2) Graceful fallback — try EVERY surface regardless of profile.
    for key in _REASONING_KEYS:
        v = _get(message, key)
        if isinstance(v, str) and v.strip():
            return v.strip(), None
    return _strip_think_tags(_content_str(message))
