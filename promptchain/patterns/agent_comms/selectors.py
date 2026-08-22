"""G4 speaker-selection menu + G5 FSM — pure Callables.

A selector has the signature::

    selector(last, agents, transcript, ctx) -> Agent | None   (may be async)

Returning ``None`` ends the group loop. Only ``llm_auto`` calls a model — and it is
**async** because it runs inside an :class:`ExternalLoop` step; the group engine
awaits it automatically.
"""
from __future__ import annotations

import random as _random

from .types import render


def round_robin():
    """Strict cyclic order = the order agents were listed."""
    i = {"n": -1}

    def pick(last, agents, transcript, ctx=None):
        i["n"] = (i["n"] + 1) % len(agents)
        return agents[i["n"]]

    return pick


def random_pick(seed: int = 0):
    """Random next speaker (excluding the current one), seeded for reproducibility."""
    rng = _random.Random(seed)

    def pick(last, agents, transcript, ctx=None):
        pool = [a for a in agents if a is not last] or list(agents)
        return rng.choice(pool)

    return pick


def manual(order):
    """Deterministic explicit order (a stand-in for a human picking next). Ends
    the loop when the list is exhausted."""
    q = list(order)

    def pick(last, agents, transcript, ctx=None):
        if not q:
            return None
        name = q.pop(0)
        return next((a for a in agents if a.name == name), None)

    return pick


def llm_auto(judge):
    """A judge agent NAMES who speaks next — the ONLY model-driven selector.
    Async: the group engine awaits it inside the loop."""
    async def pick(last, agents, transcript, ctx=None):
        names = ", ".join(a.name for a in agents)
        q = (f"Given the discussion, reply with ONLY the name of who should speak "
             f"next ({names}) to move us forward.\n\n" + render(transcript))
        choice = (await judge.respond_async(q)).strip()
        for a in agents:
            if a.name.lower() in choice.lower():
                return a
        return agents[0]

    return pick


def by_capability(default=None):
    """STATIC capability-based selection: pick the agent whose declared
    ``capabilities`` best match the current topic (the last message). Deterministic
    (keyword overlap, no model) — this is how the static orchestrator picks order by
    *what agents can do*. Ties / no match fall back to ``default`` (round-robin).

    >>> A = agent("A", "...", capabilities=["risk", "security"])
    >>> group(agents, by_capability())      # routes each turn to the fitting skill
    """
    fallback = default or round_robin()

    def pick(last, agents, transcript, ctx=None):
        topic = (transcript[-1].content if transcript else "").lower()
        best, best_score = None, 0
        for a in agents:
            score = sum(1 for cap in getattr(a, "capabilities", []) if cap.lower() in topic)
            if score > best_score:
                best, best_score = a, score
        return best if best is not None else fallback(last, agents, transcript, ctx)

    return pick


def custom(fn):
    """Wrap any ``fn(last, agents, transcript, ctx)`` as a selector."""
    return fn


def fsm(graph, inner):
    """G5 — constrain the candidate set to legal successors, then delegate to an
    inner selector.

    >>> sel = fsm({"A": ["B"], "B": ["C"], "C": ["A"]}, round_robin())
    """
    def pick(last, agents, transcript, ctx=None):
        if last is None:
            return agents[0]
        legal_names = graph.get(last.name, [])
        legal = [a for a in agents if a.name in legal_names] or list(agents)
        return inner(last, legal, transcript, ctx)

    return pick
