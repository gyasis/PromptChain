"""G12 termination / loop-break — pure stop predicates.

A stop predicate is ``(transcript, ctx) -> bool``; the group loop ends the moment
any of them is True. The hard, un-disableable cap is the group's ``max_turns``
(enforced by :class:`ExternalLoop`'s ``max_iters`` guard) — these predicates add the
*semantic* stops on top.
"""
from __future__ import annotations

from .types import speakers


def max_turns(n: int):
    """Stop after ``n`` agent turns (facilitator lines don't count)."""
    return lambda transcript, ctx=None: len([m for m in transcript if m.role != "Facilitator"]) >= n


def quorum(k: int):
    """Stop once ``k`` distinct agents have spoken (everyone weighed in)."""
    return lambda transcript, ctx=None: len(speakers(transcript)) >= k


def stop_when(substring: str):
    """Stop when the last message contains ``substring`` (e.g. an ``AGREE`` token)."""
    s = substring.lower()
    return lambda transcript, ctx=None: bool(transcript) and s in transcript[-1].content.lower()


def jaccard_repeat(threshold: float = 0.85):
    """Stop when an agent repeats itself (word-overlap of the last two turns >
    ``threshold``) — the TinyTroupe-style runaway-chatter guard."""
    def _jac(a: str, b: str) -> float:
        A, B = set(a.lower().split()), set(b.lower().split())
        return len(A & B) / len(A | B) if (A | B) else 0.0

    def stop(transcript, ctx=None) -> bool:
        msgs = [m for m in transcript if m.role != "Facilitator"]
        return len(msgs) >= 2 and _jac(msgs[-1].content, msgs[-2].content) > threshold

    return stop
