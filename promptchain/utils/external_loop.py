"""External, deterministic loop controller for PromptChain pipelines.

`ExternalLoop` is the *external* analog of `AgenticStepProcessor`'s **internal**
loop: it re-runs a caller-supplied step (typically a `PromptChain` pass) over a
worklist — or until a condition — with a **REQUIRED, default-on guard** that bounds
the loop by a maximum iteration count (always on) and an optional wall-clock budget.
That guard is the external twin of `AgenticStepProcessor.max_internal_steps`.

Custom breakers stack on top, reusing the **same** ``(iteration, state) ->
(should_stop, reason)`` shape PromptChain uses for its per-pass ``chainbreakers`` —
so "a chainbreaker breaks out of the loop" works at the loop level, over a guard
that can never be disabled.

Why this exists: `PromptChain` is linear (each instruction fires once), and the only
built-in loop is the *agentic* one (model-driven, requires a tool-calling model).
`ExternalLoop` gives a **deterministic, bounded** loop for fixed-shape work — any
model, no tools, explicit termination.

See issue #8.

Example
-------
    from promptchain.utils.external_loop import ExternalLoop

    async def step(it, state):
        state["results"] = state.get("results", 0) + 1
        return state["results"] < 100        # keep looping until 100 (or the guard)

    state = await ExternalLoop(max_iters=10, max_seconds=5).run(step)
    # state["_stopped"] == "max_iters=10"; state["_iterations"] == 10
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

# A loop breaker mirrors PromptChain's chainbreaker shape, applied to the LOOP:
#   (iteration_number, state) -> (should_stop, reason)
LoopBreaker = Callable[[int, dict], "tuple[bool, str]"]

# A step performs ONE pass: it mutates `state` and returns True to keep looping,
# or False when work is naturally exhausted.
LoopStep = Callable[[int, dict], Awaitable[bool]]


class ExternalLoop:
    """A deterministic external loop with a required, default-on guard.

    :param max_iters: hard cap on iterations. **Always on; cannot be disabled.**
        The external twin of ``AgenticStepProcessor.max_internal_steps``.
    :param max_seconds: optional wall-clock budget; the loop exits once exceeded
        (checked after each pass — a long pass may overshoot, since a running step
        is never interrupted).
    :param breakers: optional extra :data:`LoopBreaker` callables, checked **after**
        the required guard.

    The loop **always loops** (``while True``); termination comes from (a) the
    required guard, (b) a custom breaker, or (c) the step returning ``False``
    (work exhausted). On exit, ``state["_stopped"]`` holds the reason and
    ``state["_iterations"]`` the pass count.
    """

    def __init__(
        self,
        *,
        max_iters: int = 100_000,
        max_seconds: float | None = None,
        breakers: tuple[LoopBreaker, ...] = (),
    ) -> None:
        if max_iters < 1:
            raise ValueError("max_iters must be >= 1 (the guard cannot be disabled)")
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be > 0 when set")
        self.max_iters = max_iters
        self.max_seconds = max_seconds
        self.breakers = tuple(breakers)

    def _guard(self, it: int, t0: float) -> tuple[bool, str]:
        """The REQUIRED default-on guard: iteration cap (always) + optional time budget."""
        if it >= self.max_iters:
            return True, f"max_iters={self.max_iters}"
        if self.max_seconds is not None and (time.monotonic() - t0) >= self.max_seconds:
            return True, f"max_seconds={self.max_seconds}"
        return False, ""

    async def run(self, step: LoopStep, state: dict | None = None) -> dict:
        """Run ``step(iteration, state)`` repeatedly until the guard, a breaker, or
        natural exhaustion (``step`` returns ``False``) stops the loop.

        ``iteration`` is 1-based. ``state`` is created if not supplied. Returns
        ``state`` with ``_stopped`` (reason) and ``_iterations`` (count) set.
        """
        state = {} if state is None else state
        t0 = time.monotonic()
        it = 0
        while True:
            it += 1
            more = await step(it, state)
            stop, reason = self._guard(it, t0)              # required guard FIRST
            if not stop:
                for brk in self.breakers:                   # then custom breakers
                    s, r = brk(it, state)
                    if s:
                        stop, reason = True, r
                        break
            if stop or not more:
                state["_stopped"] = reason or "exhausted"
                state["_iterations"] = it
                return state

    def run_sync(self, step: LoopStep, state: dict | None = None) -> dict:
        """Synchronous convenience wrapper around :meth:`run`."""
        return asyncio.run(self.run(step, state))


async def over_worklist(
    items,
    handler: Callable[[object, dict], Awaitable[None]],
    *,
    max_iters: int = 100_000,
    max_seconds: float | None = None,
    breakers: tuple[LoopBreaker, ...] = (),
) -> dict:
    """Run ``handler(item, state)`` once per item under an :class:`ExternalLoop`.

    ``handler`` is an async callable that processes ONE item and typically writes
    into ``state["results"]``. Items are consumed from a worklist; the same
    bounded-guard guarantees apply. Returns the populated ``state``
    (``state["results"]`` is a dict, ``state["worklist"]`` the remaining items).
    """
    state: dict = {"worklist": list(items), "results": {}}

    async def step(it: int, st: dict) -> bool:
        if not st["worklist"]:
            return False
        item = st["worklist"].pop(0)
        await handler(item, st)
        return bool(st["worklist"])

    loop = ExternalLoop(max_iters=max_iters, max_seconds=max_seconds, breakers=breakers)
    return await loop.run(step, state)
