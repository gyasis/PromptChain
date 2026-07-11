"""The pattern factories — G1 dyad · G2 pipeline · G3 group · G6 broadcast · G8 nested.

The engine is :class:`Group`, driven by :class:`ExternalLoop` (deterministic, with an
un-disableable iteration guard = the G12 hard cap). ``dyad`` is just a group over two
agents. ``pipeline`` and ``broadcast`` are straight-line (inherently bounded).
Everything here is STATIC control flow: a Callable selector/stop decides flow, never a model.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Callable, List, Optional

from promptchain.utils.external_loop import ExternalLoop

from .selectors import round_robin
from .types import Msg, MeshContext, Transcript, render


class Group:
    """G3 — a shared thread; each round a ``selector`` picks the next speaker until
    a stop predicate fires or ``max_turns`` is hit. Runs synchronously to the caller;
    async internally (workers use ``say_async`` under the ExternalLoop)."""

    def __init__(self, agents, selector=None, term: Optional[List[Callable]] = None,
                 on_turn: Optional[Callable] = None, max_turns: int = 8):
        self.agents = list(agents)
        self.selector = selector or round_robin()
        self.term = list(term or [])
        self.on_turn = on_turn
        self.max_turns = max_turns

    def run(self, goal: str, ctx: Optional[MeshContext] = None) -> Transcript:
        ctx = ctx or MeshContext(participants=[a.name for a in self.agents])
        t: Transcript = [Msg("Facilitator", goal)]
        state = {"last": None}

        async def step(it: int, st: dict) -> bool:
            nxt = self.selector(st["last"], self.agents, t, ctx)
            if inspect.isawaitable(nxt):          # llm_auto is async
                nxt = await nxt
            if nxt is None:
                return False                      # selector ended the conversation
            await nxt.say_async(t, ctx)
            st["last"] = nxt
            if self.on_turn:
                self.on_turn(t[-1], ctx)
            return not any(stop(t, ctx) for stop in self.term)

        ExternalLoop(max_iters=self.max_turns).run_sync(step, state)
        return t


def group(agents, selector=None, term=None, on_turn=None, max_turns: int = 8) -> Group:
    """A moderated group (G3). See :class:`Group`."""
    return Group(agents, selector, term, on_turn, max_turns)


def dyad(a, b, goal: str = "Begin.", term=None, max_turns: int = 6) -> Transcript:
    """G1 — a 1:1 exchange = a group over two agents, alternating."""
    return Group([a, b], round_robin(), term, max_turns=max_turns).run(goal)


class _Pipeline:
    def __init__(self, agents):
        self.agents = list(agents)

    def run(self, seed: str) -> str:
        """G2 — each agent transforms the previous one's output (carryover)."""
        out = seed
        for a in self.agents:
            out = a.respond(out)
        return out


def pipeline(agents) -> _Pipeline:
    """G2 — sequential pipeline with carryover."""
    return _Pipeline(agents)


class _Broadcast:
    def __init__(self, agents, synthesizer=None):
        self.agents = list(agents)
        self.synthesizer = synthesizer

    def run(self, payload: str):
        """G6 — fan the payload out to all agents at once, then (optionally) a
        synthesizer merges the replies. Returns ``(replies, verdict)``."""
        async def _fanout():
            async def one(a):
                return Msg(a.name, await a.respond_async(payload))
            return await asyncio.gather(*[one(a) for a in self.agents])

        replies = asyncio.run(_fanout())
        verdict = None
        if self.synthesizer is not None:
            verdict = self.synthesizer.respond(render(replies))
        return replies, verdict


def broadcast(agents, synthesizer=None) -> _Broadcast:
    """G6 — broadcast fan-out / fan-in."""
    return _Broadcast(agents, synthesizer)


def as_agent(name: str, inner_run: Callable, summarizer):
    """G8 — wrap a whole group/callable as ONE agent. Outsiders see only the
    summarized result; the inner discussion stays hidden (chains-are-functions).

    ``inner_run`` is a zero-arg callable returning a transcript (or any object)."""
    class _Nested:
        def __init__(self):
            self.name = name

        def _summary(self) -> str:
            inner = inner_run()
            txt = render(inner) if isinstance(inner, list) else str(inner)
            return summarizer.respond(txt)

        def say(self, transcript: Transcript, ctx=None) -> str:
            s = self._summary()
            transcript.append(Msg(self.name, s))
            return s

        async def say_async(self, transcript: Transcript, ctx=None) -> str:
            return self.say(transcript, ctx)

    return _Nested()
