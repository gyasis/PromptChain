# ExternalLoop — a deterministic external loop for PromptChain

`ExternalLoop` (`promptchain.utils.external_loop`) re-runs a step — typically a whole
`PromptChain` pass — over a worklist, or until a condition, with a **required,
default-on guard** that bounds the loop by a maximum iteration count (always on) and
an optional wall-clock budget.

It is the **external, deterministic analog of `AgenticStepProcessor`'s *internal*
loop**: the guard is the external twin of `max_internal_steps`, and custom breakers
reuse the same `(iteration, state) -> (should_stop, reason)` shape PromptChain uses
for its per-pass `chainbreakers`.

```python
from promptchain.utils.external_loop import ExternalLoop, over_worklist
```

## Why it exists

A `PromptChain` is **linear** — each instruction fires once, left → right. The only
loop built into the library is the *agentic* one inside `AgenticStepProcessor`, which
is model-driven and requires a tool-calling-capable model. There was no **deterministic,
bounded** loop for re-running a fixed-shape chain over many inputs. `ExternalLoop` fills
that gap: any model, no tools, explicit termination, can't run away.

## The required guard (cannot be disabled)

| Guard | Default | Behaviour |
|---|---|---|
| `max_iters` | `100_000` | Hard cap on iterations. **Always on** — `max_iters < 1` raises `ValueError`. The external twin of `AgenticStepProcessor.max_internal_steps`. |
| `max_seconds` | `None` | Optional wall-clock budget. Checked after each pass (a running pass is never interrupted, so a long pass may overshoot). |

The loop **always loops** (`while True`); it stops when (a) the required guard fires,
(b) a custom breaker fires, or (c) the `step` returns `False` (work naturally
exhausted). On exit, `state["_stopped"]` holds the reason and `state["_iterations"]`
the pass count.

## Wrapping a 6-step sequential chain

The classic shape: an inner **sequential** `PromptChain` does the per-item work; the
**`ExternalLoop` `state` wraps around it**, threading across iterations.

```python
import json
from promptchain import PromptChain
from promptchain.utils.external_loop import ExternalLoop

# ---- inner work: a 6-step SEQUENTIAL chain (one full pass per document) ----
def make_distill_chain() -> PromptChain:
    return PromptChain(
        models=["openai/gpt-4o-mini"] * 5,   # 5 STRING steps need 5 slots; the function step needs none
        instructions=[
            "Extract the key points from this document as a bullet list:\n{input}",   # 1
            "Group these bullet points into 2-4 themes:\n{input}",                     # 2
            "Order the themes from most to least important:\n{input}",                 # 3
            "Write a short paragraph summary from these ordered themes:\n{input}",     # 4
            "Tighten that paragraph to exactly TWO sentences:\n{input}",               # 5
            lambda two: json.dumps({"summary": two.strip()}),                          # 6 (function, no slot)
        ],
        store_steps=True,
    )

# ---- outer loop: ExternalLoop threads STATE across documents ----
async def distill_all(docs: list[str]) -> dict:
    state = {"queue": list(docs), "results": [], "done": 0}   # the state that WRAPS the chain

    async def step(it: int, st: dict) -> bool:
        if not st["queue"]:
            return False                                       # natural exhaustion → loop ends
        doc = st["queue"].pop(0)
        out = await make_distill_chain().process_prompt_async(doc)   # ONE full 6-step pass
        st["results"].append(json.loads(out))                  # accumulate into the wrapping state
        st["done"] += 1
        return bool(st["queue"])                               # keep looping while docs remain

    # always loops; the REQUIRED guard exits on 50 iterations OR 120s, whichever first
    return await ExternalLoop(max_iters=50, max_seconds=120).run(step, state)
```

For `distill_all([docA, docB, docC])`:

| iter | `step` does | `state` after |
|---|---|---|
| 1 | pop docA → 6-step pass | `results=[A]`, `done=1`, queue=[B,C] → `True` |
| 2 | pop docB → 6-step pass | `results=[A,B]`, `done=2`, queue=[C] → `True` |
| 3 | pop docC → 6-step pass | `results=[A,B,C]`, `done=3`, queue=[] → `False` |
| — | loop exits | `_stopped="exhausted"`, `_iterations=3` |

The **same `state` dict** is threaded through every iteration; the chain is re-created
and run *inside* each pass; results accumulate in `state`.

## Worklist convenience

`over_worklist` packages the common "run a handler once per item" pattern:

```python
async def handler(item, state):
    state["results"][item] = await make_distill_chain().process_prompt_async(item)

state = await over_worklist(docs, handler, max_iters=50, max_seconds=120)
# state["results"] is a dict keyed by item; same guard guarantees apply.
```

## Refine-until-good (the other way state wraps)

Instead of a worklist, re-feed the chain's output as its next input and exit on a
**custom breaker** (quality gate). The required guard is still the backstop.

```python
def good_enough(it, st):
    return (st.get("score", 0) >= 0.9, f"score={st.get('score')}")

async def step(it, st):
    st["draft"] = await refine_chain().process_prompt_async(st.get("draft", st["seed"]))
    st["score"] = score(st["draft"])      # your scoring function
    return True                            # the breaker / guard decide when to stop

state = await ExternalLoop(max_iters=6, breakers=(good_enough,)).run(step, {"seed": prompt})
```

## Two levels of break, one contract

- **Inside a pass** → PromptChain's native `chainbreakers` (e.g. abort a node's chain
  early on an empty/garbled step).
- **Across passes** → `ExternalLoop` breakers + the required guard.

Both use `(n, x) -> (stop, reason, ...)`, so "a chainbreaker breaks out of the loop"
works at the loop level — over a guard that can never be disabled.

## ExternalLoop vs AgenticStepProcessor

| | `ExternalLoop` (this) | `AgenticStepProcessor` |
|---|---|---|
| Loop driver | **deterministic** (your `step`) | the model (ReAct/TAO) |
| Bound | `max_iters` (always) + `max_seconds` | `max_internal_steps` |
| Model | **any** (or none) | tool-calling-capable required |
| Tools | not required | the point |
| Order | fixed | model-chosen |
| Use when | fixed-shape work over N inputs / refine-to-threshold | open-ended reasoning, model picks tools |

## API reference

### `class ExternalLoop(*, max_iters=100_000, max_seconds=None, breakers=())`
- `max_iters: int` — required iteration cap (≥ 1).
- `max_seconds: float | None` — optional wall-clock budget (> 0 when set).
- `breakers: tuple[LoopBreaker, ...]` — extra `(iter, state) -> (stop, reason)`, checked after the guard.
- `async run(step, state=None) -> dict` — run until guard / breaker / `step` returns `False`. Sets `state["_stopped"]` + `state["_iterations"]`.
- `run_sync(step, state=None) -> dict` — `asyncio.run` wrapper.

### `async over_worklist(items, handler, *, max_iters=100_000, max_seconds=None, breakers=()) -> dict`
Run `handler(item, state)` once per item under an `ExternalLoop`. Returns `state` with `results` (dict) and remaining `worklist`.

### Types
- `LoopStep = Callable[[int, dict], Awaitable[bool]]` — one pass; returns `True` to keep looping, `False` when exhausted.
- `LoopBreaker = Callable[[int, dict], tuple[bool, str]]` — loop-level exit predicate.

## See also
- `docs/circular_runs.md` — the manual iterative/circular pattern `ExternalLoop` formalizes with a bounded, required guard.
- `docs/chainbreaker_guide.md` — the per-pass `chainbreakers` whose contract the loop breakers reuse.
- `docs/agentic_step_processor.md` — the internal-loop counterpart.
- `examples/external_loop_example.py` — runnable (no API key needed for the default demo).
- `tests/test_external_loop.py` — semantics, deterministic, no LLM.
- `docs/external_loop_infographic.html` — one-page visual overview (open in a browser).
