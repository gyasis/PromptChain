---
id: recipe-external-loop
title: Bounded external loop (ExternalLoop, over_worklist)
when: You need to repeat a fixed-shape step until a condition, over a worklist, or under a hard iteration/time cap, with any model and no tool calling.
api_version: 0.10.0+
---

# External Loop

`PromptChain` is linear: each instruction fires once. `ExternalLoop` re-runs a step you supply
(often one `PromptChain` pass) with a guard you cannot turn off. It is deterministic: no
model decides when to stop.

## Which loop?

| Need | Use |
|---|---|
| Repeat a step N times or until a flag, with a hard cap | `ExternalLoop` (this recipe) |
| Run a handler once per item in a list | `over_worklist` (this recipe) |
| The model chooses when to stop and calls tools | `AgenticStepProcessor` (`recipe-agentic-step`) |
| Generate code until a test command exits 0 | `MicroPromptChain` or `RalphChain` (`recipe-test-loop`) |
| Research, build until verified, then critique | `AutoResearch` (`recipe-autoresearch`) |

## Offline example

```python
import asyncio
from promptchain.utils.external_loop import ExternalLoop, over_worklist

async def step(it: int, state: dict) -> bool:
    # it is 1-based. Return True to keep looping, False when the work is done.
    state["total"] = state.get("total", 0) + it
    return state["total"] < 10

async def main():
    state = await ExternalLoop(max_iters=50, max_seconds=5).run(step)
    print(state["_stopped"], state["_iterations"], state["total"])   # exhausted 4 10

    # A breaker has the same shape as a PromptChain chainbreaker: (it, state) -> (stop, reason)
    def too_big(it, st):
        return st.get("total", 0) > 5, "total>5"

    state = await ExternalLoop(max_iters=50, breakers=(too_big,)).run(step)
    print(state["_stopped"], state["_iterations"])                   # total>5 3

    # The guard wins: max_iters stops a step that never says "done".
    async def forever(it, st):
        return True
    state = await ExternalLoop(max_iters=3).run(forever)
    print(state["_stopped"], state["_iterations"])                   # max_iters=3 3

    # One handler call per item; results go in state["results"].
    async def handler(item, st):
        st["results"][item] = item.upper()

    state = await over_worklist(["a", "b", "c"], handler, max_iters=10)
    print(state["results"], state["_stopped"], state["_iterations"])
    # {'a': 'A', 'b': 'B', 'c': 'C'} exhausted 3

asyncio.run(main())

# Sync callers (no event loop running): run_sync wraps asyncio.run.
async def forever(it, st):
    return True

print(ExternalLoop(max_iters=2).run_sync(forever)["_stopped"])       # max_iters=2
```

## Contract

| Item | Value |
|---|---|
| `ExternalLoop(*, max_iters=100_000, max_seconds=None, breakers=())` | All kwargs are keyword-only. `max_iters < 1` or `max_seconds <= 0` raises `ValueError`. |
| `await loop.run(step, state=None)` | Returns the `state` dict. A `None` state becomes `{}`. |
| `step` | `async (iteration: int, state: dict) -> bool`. `True` = keep going. |
| breaker | `(iteration, state) -> (should_stop: bool, reason: str)`. Plain function, not async. |
| `over_worklist(items, handler, *, max_iters, max_seconds, breakers)` | `handler` is `async (item, state) -> None`. State starts as `{"worklist": [...], "results": {}}`. A handler that raises propagates out of `over_worklist`, and you lose every result so far (the state is not returned). When the cap or a breaker stops the run early, the unprocessed items stay in `state["worklist"]`. If `len(items) == max_iters`, the run ends by the cap on the last item and reports `max_iters=N`, not `exhausted`. |
| Result keys | `state["_stopped"]` (reason) and `state["_iterations"]` (count). |

`_stopped` values: `"max_iters=N"`, `"max_seconds=S"`, a breaker's reason string, or `"exhausted"`
when the step returned `False`.

Each pass runs the step, then the always-on guard, then your breakers in order. The step
always runs at least once.

## Using it with a PromptChain

```python
from promptchain import PromptChain
from promptchain.utils.external_loop import ExternalLoop

chain = PromptChain(models=["openai/gpt-4o-mini"], instructions=["Improve: {input}"])

async def step(it, state):
    state["draft"] = await chain.process_prompt_async(state["draft"])
    return "DONE" not in state["draft"]

# state = await ExternalLoop(max_iters=5).run(step, {"draft": "first draft"})
```

## Common mistakes

- **Positional args to the constructor.** `ExternalLoop(10)` raises `TypeError`. Write `ExternalLoop(max_iters=10)`.
- **Step returns `None`.** `None` is falsy, so the loop stops after pass 1 with `_stopped == "exhausted"`. Return `True` explicitly to continue.
- **Sync step.** `run` awaits the step. A plain `def` step fails with `TypeError: object bool can't be used in 'await' expression`.
- **Expecting `max_seconds` to interrupt a pass.** The loop checks the time budget after each pass. It never cancels a running pass, so put a timeout inside the step.
- **Treating the default cap as a limit.** `max_iters` defaults to 100,000. Set a value that fits the task.
- **Reading the return as a list.** `over_worklist` returns the state dict. The per-item output is `state["results"]`.
- **`run_sync` inside an `async def`.** `ExternalLoop.run_sync` calls `asyncio.run` and raises in a running loop. Use `await loop.run(...)`. (`MicroPromptChain.run_sync` and friends are safe there; `ExternalLoop.run_sync` is not.)
