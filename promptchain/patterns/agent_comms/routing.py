"""G7 handoff / condition-based routing — pure Callables.

A *rule* is ``(query, ctx) -> Agent | None``; a *router* tries rules in priority
order and falls back. ``keyword_rule`` / ``cond_rule`` are model-free (safe inside a
group loop via :func:`sel_from_router`). ``llm_rule`` calls a model synchronously —
use it for standalone routing, not inside an ``ExternalLoop`` step (use
:func:`~promptchain.patterns.agent_comms.selectors.llm_auto` there instead).
"""
from __future__ import annotations

from typing import List


def keyword_rule(words, target):
    """Route to ``target`` if any keyword appears in the query."""
    ws = [w.lower() for w in words]
    return lambda query, ctx=None: target if any(w in query.lower() for w in ws) else None


def cond_rule(predicate, target):
    """Route to ``target`` if ``predicate(query, ctx)`` is truthy (e.g. a blackboard check)."""
    return lambda query, ctx=None: target if predicate(query, ctx) else None


def llm_rule(judge, question, target):
    """Route to ``target`` if the judge answers yes to ``question`` about the query.
    Synchronous — standalone use only (not inside a group loop)."""
    def rule(query, ctx=None):
        ans = judge.respond(f"{question}\n\nText: {query}\n\nAnswer yes or no only.").strip().lower()
        return target if ans.startswith("y") else None

    return rule


def router(rules: List, fallback):
    """Priority chain: first rule to return a target wins; else ``fallback``.

    >>> route = router([keyword_rule(["data"], C), keyword_rule(["risk"], A)], B)
    >>> route("what does the data say?")   # -> C
    """
    def route(query, ctx=None):
        for rule in rules:
            t = rule(query, ctx)
            if t is not None:
                return t
        return fallback

    return route


def sel_from_router(route):
    """Adapt a router into a G4 selector — routes on the LAST message's content."""
    def pick(last, agents, transcript, ctx=None):
        query = transcript[-1].content if transcript else ""
        target = route(query, ctx)
        return target if hasattr(target, "name") else None

    return pick
