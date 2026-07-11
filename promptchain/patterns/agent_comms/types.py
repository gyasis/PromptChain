"""Core data types for agent-communication patterns.

A message stream is a plain ``list[Msg]``; a shared blackboard is ``MeshContext``
(G9); an accessibility graph is ``AccessibilityGate`` (G11). Deliberately tiny —
these are records, not machinery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class Msg:
    """One line in a conversation. ``str(msg)`` renders ``"Role: content"``."""

    role: str
    content: str

    def __str__(self) -> str:  # noqa: D401
        return f"{self.role}: {self.content}"


# A transcript is just a list of Msg — no wrapper class needed.
Transcript = List[Msg]

# The kickoff/system voice; excluded from "who has spoken" counts.
FACILITATOR = "Facilitator"


def render(transcript: Transcript) -> str:
    """The transcript as a single string, ready to feed an agent."""
    return "\n".join(str(m) for m in transcript) if transcript else "(open the discussion)"


def speakers(transcript: Transcript) -> Set[str]:
    """Distinct agent names that have spoken (excludes the facilitator)."""
    return {m.role for m in transcript if m.role != FACILITATOR}


@dataclass
class MeshContext:
    """G9 — the shared blackboard: state read/written out-of-band of the message
    stream. Selectors and routers READ it; tools WRITE it."""

    vars: Dict[str, Any] = field(default_factory=dict)
    participants: List[str] = field(default_factory=list)


@dataclass
class AccessibilityGate:
    """G11 — a relationship graph gating who may reach whom.

    >>> g = AccessibilityGate({"A": {"B"}, "B": {"A", "C"}, "C": {"B"}})
    >>> g.can_reach("A", "B"), g.can_reach("A", "C")
    (True, False)
    """

    allowed: Dict[str, Set[str]]

    def can_reach(self, src: str, dst: str) -> bool:
        return dst in self.allowed.get(src, set())
