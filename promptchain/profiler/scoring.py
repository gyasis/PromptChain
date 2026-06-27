"""scoring — deterministic auto-scorers for probe replies (F2).

score_response(item, reply) -> (correct: bool, raw: dict)
where reply == {"text": str, "usage": {...}, "latency_ms": float}.
Dispatches on item.dimension. See tests/test_profiler_scoring.py for the
exact rules mirrored here.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

from promptchain.profiler.item_bank import ProbeItem


def _try_parse_json(text: str) -> Tuple[bool, Any]:
    """Attempt to parse a JSON object out of `text`.

    First tries the whole string; if that fails, tries the first {...} span.
    Returns (ok, parsed_or_None).
    """
    try:
        return True, json.loads(text)
    except (ValueError, TypeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return True, json.loads(match.group(0))
        except (ValueError, TypeError):
            pass
    return False, None


def score_response(item: ProbeItem, reply: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Score one probe reply. Returns (correct, raw_values)."""
    text = reply.get("text", "")
    dim = item.dimension
    meta = item.meta or {}

    if dim == "instruction_following":
        correct = text.strip() == meta.get("expect")
        return bool(correct), {"text": text.strip()}

    if dim == "tool_call_reliability":
        ok, parsed = _try_parse_json(text)
        if not ok or not isinstance(parsed, dict):
            return False, {"parsed": None}
        tool = parsed.get("tool")
        correct = tool == meta.get("expect_tool")
        return bool(correct), {"tool": tool}

    if dim == "format_sensitivity":
        if meta.get("format") == "json":
            try:
                json.loads(text)
                return True, {"format": "json"}
            except (ValueError, TypeError):
                return False, {"format": "json"}
        # default: non-empty text passes
        return bool(text.strip()), {}

    if dim == "reasoning_depth":
        answer = str(meta.get("answer", ""))
        correct = answer != "" and answer in text
        return bool(correct), {"answer": answer}

    if dim == "latency_throughput":
        latency_ms = reply.get("latency_ms")
        threshold = meta.get("max_latency_ms")
        if threshold is not None and latency_ms is not None:
            correct = latency_ms <= threshold
        else:
            correct = True
        return bool(correct), {"latency_ms": latency_ms}

    # sensible default for any other dimension: non-empty text is "correct"
    return bool(text.strip()), {"text": text}
