"""F3 Dynamic Prompt Layer — prompt tiers + per-tier optional modules.

Implements the contract in ``specs/014-dynamic-prompt-layer/data-model.md``
(PromptTier, OptionalModule) and ``contracts/generator-api.md`` (select_tier,
modules_for_tier). All pure + deterministic.

- ``PromptTier`` — the three prompt strengths a model can be served (D2).
- ``OptionalModule`` — a droppable section; lower ``drop_priority`` is dropped
  FIRST under budget pressure (D5).
- ``select_tier`` — maps a capability profile to a tier.
- ``modules_for_tier`` — the optional modules that belong to a tier, ordered by
  ``drop_priority`` ascending (drop-first first).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, FrozenSet, List, Optional, Set, Union

if TYPE_CHECKING:  # avoid a hard import cycle / runtime cost
    from promptchain.profiler.jacket import CapabilityProfile


class PromptTier(str, Enum):
    """Prompt strength served to a model, by measured capability (D2)."""

    CORE = "CORE"
    EXTENDED = "EXTENDED"
    TINY = "TINY"


@dataclass
class OptionalModule:
    """A droppable prompt section. Lower ``drop_priority`` is dropped first."""

    key: str
    text: str
    drop_priority: int
    tiers: Union[Set[PromptTier], FrozenSet[PromptTier]]


# --------------------------------------------------------------------------- #
# Per-tier optional-module registry.
#
# EXTENDED carries the richest set; "examples" is the lowest drop_priority so it
# is shed first, then "extended_guidance". CORE = base + essentials (no
# examples, no extended_guidance). TINY = the most reduced protocol (no optional
# modules at all → <= CORE count).
# --------------------------------------------------------------------------- #
_MODULES: List[OptionalModule] = [
    OptionalModule(
        key="examples",
        text=(
            "EXAMPLES:\n"
            "- Worked example: THINK → restate goal → PLAN tasks → ACT one tool "
            "at a time → OBSERVE the real result before the next step."
        ),
        drop_priority=1,
        tiers=frozenset({PromptTier.EXTENDED}),
    ),
    OptionalModule(
        key="extended_guidance",
        text=(
            "EXTENDED GUIDANCE:\n"
            "- Prefer a dedicated tool over a raw shell command; gather context "
            "(read/search) before editing; verify every change with real output."
        ),
        drop_priority=2,
        tiers=frozenset({PromptTier.EXTENDED}),
    ),
]


# --------------------------------------------------------------------------- #
# TINY tier = a SIMPLER REPLACEMENT base (Goose `tiny_model_system` precedent;
# the brief: "the tiny variant may even swap the loop protocol").
#
# Origin 2026-06-27: a weak model (llama3.2:1b) served the full ~748-token
# foundation scores ~60% on trivial tasks but ANY *addition* to that base makes it
# WORSE (a family preamble → −60%, a focusing directive in the floor → −40%): the
# full foundation is already at the 1B model's ceiling, so additive-only F3 cannot
# help it. The only way to help a weak model is a *shorter, simpler* prompt. So
# the TINY tier SWAPS the full foundation for this short base. Consequently the
# verbatim-static-base invariant (SC-003) holds for CORE/EXTENDED only — TINY is
# the sanctioned exception (decided with the user 2026-06-27).
#
# TINY_DIRECTIVE is the focusing tail (also the stable marker the A/B eval uses to
# tell the f3 arm from the static-base arm). It must NOT appear in the full
# foundation. It avoids prescribing an OUTPUT FORMAT (that is what harmed the
# model) — it steers toward ANSWERING, not reformatting.
# --------------------------------------------------------------------------- #
TINY_DIRECTIVE = (
    "Answer directly and minimally: for a coding task, give the minimal code AND "
    "its final printed result/value. Do not restate the task, do not explain your "
    "steps, do not add section headers — just produce the answer."
)

TINY_BASE_PROMPT = (
    "You are a focused coding assistant. Do exactly ONE task and stop.\n\n"
    "TASK:\n{objective}\n\n" + TINY_DIRECTIVE
)


def select_tier(profile: "Optional[CapabilityProfile]") -> PromptTier:
    """Map a capability profile to a prompt tier (D2).

    None profile → CORE (safe default). Otherwise, by ``profile.capability``:
    ``>= 0.66`` → EXTENDED, ``0.33 <= c < 0.66`` → CORE, ``< 0.33`` → TINY.
    """
    if profile is None:
        return PromptTier.CORE

    capability = profile.capability
    if capability >= 0.66:
        return PromptTier.EXTENDED
    if capability >= 0.33:
        return PromptTier.CORE
    return PromptTier.TINY


def modules_for_tier(tier: PromptTier) -> List[OptionalModule]:
    """Optional modules for ``tier``, sorted by ``drop_priority`` ascending."""
    selected = [m for m in _MODULES if tier in m.tiers]
    return sorted(selected, key=lambda m: m.drop_priority)
