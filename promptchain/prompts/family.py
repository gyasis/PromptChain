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

# Per-family FORMAT preamble (format guidance only — no task content). Distinct
# per family so two same-tier models of different families get measurably
# different prompts (SC-002), while every base/optional part stays verbatim.
_FAMILY_PREAMBLE = {
    "anthropic": "FORMAT: structure your reasoning and outputs with XML-style tags "
                 "(e.g. <plan>…</plan>); keep tool calls native.",
    "openai": "FORMAT: structure sections with Markdown headers; emit concise, "
              "single-purpose tool calls.",
    "google": "FORMAT: use clear Markdown sections and explicit step labels; "
              "prefer compact, well-delimited blocks.",
    "qwen": "FORMAT: use plain headed sections; keep instructions terse and "
            "imperative; one action per step.",
    "llama": "FORMAT: use plain ALL-CAPS section labels and short imperative "
             "lines; avoid nested structure.",
}


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

    Known families get a concise family-specific FORMAT preamble prepended (a
    per-family *variant*); ``default`` and any unknown family are the identity
    transform. Returns a new list. Every original part's text is preserved
    verbatim — the preamble is additive framing, never a rewrite of content.
    """
    preamble = _FAMILY_PREAMBLE.get(family)
    if not preamble:  # "default" and unknown families → identity
        return list(parts)
    return [preamble, *parts]
