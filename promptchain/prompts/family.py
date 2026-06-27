"""F3 Dynamic Prompt Layer — model-family detection + format-only adaptation.

Implements ``contracts/generator-api.md`` (family_of, adapt_format) and the
data-model FamilyAdapter. ``adapt_format`` adjusts FORMAT ONLY — it never drops
or alters the base content (FR-003): every input part's text survives verbatim
as a substring of the result. Per-family *variants* over a ``default`` fallback
mirror opencode ``routing.ts`` (a per-family framing preamble over ``default.txt``):
known families get a concise, family-appropriate FORMAT preamble; ``default`` and
unknown families are the identity transform.
"""
from __future__ import annotations

from typing import List

_KNOWN = {"anthropic", "openai", "google", "qwen", "llama", "default"}

def family_of(model_id: str) -> str:
    """Map a model id to its family stem.

    Handles bare ids, ``provider/model`` prefixes, and ``:tag`` suffixes →
    one of ``anthropic|openai|google|qwen|llama|default``.
    """
    if not model_id:
        return "default"

    stem = model_id.strip().lower()
    if "/" in stem:  # strip provider prefix, e.g. "ollama/qwen3-coder:30b"
        stem = stem.rsplit("/", 1)[-1]
    if ":" in stem:  # strip a trailing tag, e.g. "qwen3-coder:30b"
        stem = stem.split(":", 1)[0]

    if "claude" in stem or "anthropic" in stem:
        return "anthropic"
    if stem.startswith("gpt") or stem.startswith("o1") or "openai" in stem:
        return "openai"
    if "gemini" in stem:
        return "google"
    if "qwen" in stem:
        return "qwen"
    if "llama" in stem:
        return "llama"
    return "default"


def adapt_format(parts: List[str], family: str) -> List[str]:
    """Format-only normalization of prompt ``parts`` for a model family (D3).

    Returns a new list, identity for ALL families. The seam exists (``family_of``
    is wired) but adapt_format injects NO content — in particular NO prescriptive
    output-format directive.

    Why identity (origin 2026-06-27): an earlier version prepended a per-family
    ``FORMAT: …`` preamble. The offline live smoke showed it MEASURABLY HARMED a
    weak model (llama3.2:1b regurgitated the prompt structure in ALL-CAPS instead
    of solving the task → 0% vs 60% for the bare base). A family adapter must
    normalize how the prompt is FRAMED, never dictate the model's OUTPUT format —
    so until a genuinely helpful (non-prescriptive) per-family framing is designed
    and validated against weak models, this is the safe identity transform.
    """
    return list(parts)
