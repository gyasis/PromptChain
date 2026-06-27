"""F3 Dynamic Prompt Layer — token measurement + budget fitting.

Implements ``contracts/prompt-layout.md`` (D5 budget enforcement) and
``generator-api.md``: the static ``base`` is NEVER dropped or truncated;
optional modules are dropped in ascending ``drop_priority`` until the assembly
fits the target budget.
"""
from __future__ import annotations

from typing import List, Tuple

from promptchain.prompts.tiers import OptionalModule


def measure(text: str) -> int:
    """Token count of ``text`` (tiktoken cl100k_base; ``len // 4`` fallback).

    Mirrors ``DynamicTUIPromptGenerator.get_token_estimate``. ``measure("")`` is 0.
    """
    try:
        import tiktoken  # type: ignore[import-untyped]

        return max(0, len(tiktoken.get_encoding("cl100k_base").encode(text)))
    except Exception:
        return max(0, len(text) // 4)


def _assemble(base: str, kept: List[OptionalModule]) -> str:
    """Join the base with the kept optional modules' text (blank-line separated)."""
    if not kept:
        return base
    return "\n\n".join([base, *(m.text for m in kept)])


def fit_to_budget(
    base: str,
    optional: List[OptionalModule],
    *,
    target_max: int,
    hard_cap: int,
) -> Tuple[str, List[str]]:
    """Assemble ``base`` + as many optional modules as fit ``target_max``.

    The base is always present in full (never dropped/truncated). Optional
    modules are dropped greedily in ascending ``drop_priority`` (lowest first)
    while the assembly measures over ``target_max`` and modules remain.

    If ``base`` alone exceeds ``hard_cap``, all optional modules are dropped and
    the base-only assembly is returned (parity floor never truncated).

    Returns ``(assembled, dropped_keys)`` where ``dropped_keys`` lists the
    dropped module keys in the order dropped (ascending ``drop_priority``).
    """
    ordered = sorted(optional, key=lambda m: m.drop_priority)  # drop-first first

    if measure(base) > hard_cap:
        return _assemble(base, []), [m.key for m in ordered]

    kept = list(ordered)
    dropped: List[str] = []
    while kept and measure(_assemble(base, kept)) > target_max:
        shed = kept.pop(0)  # lowest drop_priority among those remaining
        dropped.append(shed.key)

    return _assemble(base, kept), dropped
