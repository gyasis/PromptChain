---
id: recipe-cascade
title: Verify-then-escalate cascade (cheap model first, frontier only on failure)
when: A step can often be answered by a cheap model or function, an output check can tell good from bad, and you want to pay for the expensive tier only when the check fails.
api_version: 0.10.0+
---

# Cascade

`cascade()` returns one plain Callable chain step. It runs tier 0 on the step input, calls
`check(step_input, output)`, and returns the first output that passes. On a failure it runs the
next tier on the **same original input**. It uses no model slot, so a chain made only of cascade
steps takes `models=[]`.

Module: `promptchain.utils.cascade` (`cascade`, `decision_check`, `CascadeResult`,
`TierAttempt`, `CascadeExhausted`). The tier type is the same `Branch` as `route_on`: a
`PromptChain` or a Callable step (sync or async, text in, text out).

## Runnable offline example

```python
import asyncio
from promptchain import PromptChain
from promptchain.utils.cascade import cascade, CascadeExhausted

async def cheap(state: str) -> str:      # stands in for a small model
    return "I think the answer is maybe 41"

async def frontier(state: str) -> str:   # stands in for the expensive model
    return "ANSWER: 42"

def check(step_input: str, output: str):  # (input, output) -> bool or (bool, reason)
    ok = output.startswith("ANSWER:")
    return ok, "has ANSWER prefix" if ok else "missing ANSWER prefix"

step = cascade([cheap, frontier], check=check, name="answer", labels=["cheap", "frontier"])

async def main():
    chain = PromptChain(models=[], instructions=[step])
    print(await chain.process_prompt_async("What is 6*7?"))   # ANSWER: 42
    r = step.last_result
    print(r.answered_by, r.escalated)                          # frontier True
    for a in r.attempts:
        print(a.tier, a.label, a.passed, a.reason)

asyncio.run(main())
```

Real tiers: `cascade([PromptChain(models=["ollama/small"], instructions=["{input}"]),
PromptChain(models=["openai/gpt-4.1"], instructions=["{input}"])], check=..., name="answer")`.

## Cascade vs two-tier vs `route_on`

| need | use |
|---|---|
| Try cheap, verify the OUTPUT, escalate on a bad output | `cascade` |
| `AgenticStepProcessor` picks a model per call by task complexity (heuristic, does not inspect output) | `fallback_model` two-tier routing in the agentic step |
| Pick a branch BEFORE running anything, from a measured label or confidence | `decision` + `route_on` |

## The check contract

- Signature `check(step_input: str, output: str)`, in that order. `step_input` is the text
  the cascade step received. `output` is the tier's text.
- Return `True`/`False`, or `(bool, reason)`. The reason lands on `TierAttempt.reason`.
- The check may be sync or `async`.
- **A check that raises propagates out of the cascade.** A broken verifier is never read as
  pass or fail, and no later tier runs. Catch inside the check if you want a verdict.
- A **tier** that raises is different: the cascade stores the exception on `TierAttempt.error`
  (`"ExcType: message"`) and the cascade escalates.

Ready-made check: `decision_check(backend, instructions=..., accept=0.9)`. It asks one boolean
question through a decision backend (see `recipe-decision-step`) with the state
`<input>...</input><output>...</output>`. The output passes when the answer is True and the
measured confidence is at least `accept`.

```python
from promptchain.utils.decision_step import SystemOneBackend
from promptchain.utils.cascade import decision_check

check = decision_check(SystemOneBackend(model="tev1:4b"),
                       instructions="Does the output fully answer the input?", accept=0.9)
```

Test-based check: run the output in an executor and pass on exit code 0.

```python
from promptchain.utils.test_loop_chain import LocalExecutor   # trusted code only, no sandbox

def tests_pass(step_input: str, output: str):
    with LocalExecutor() as ex:
        ex.write_file("solution.py", output)
        ex.write_file("test_solution.py", "from solution import f\ndef test_f():\n    assert f(2) == 4\n")
        res = ex.run("python -m pytest -q test_solution.py")
    return res.exit_code == 0, res.output[-200:]
```

Do not use `CoVeVerifier` as a check: it verifies tool calls only and passes a plain text answer.

## Reading the result

`step.last_result` is a `CascadeResult` after every run:

| field | meaning |
|---|---|
| `name`, `output` | cascade name; accepted text (`None` if exhausted) |
| `answered_by`, `tier` | label and index of the passing tier (`None` if exhausted) |
| `attempts` | list of `TierAttempt(tier, label, passed, reason, elapsed_s, output, error)` |
| `escalated` | property: more than one tier ran |
| `to_dict()` | JSON-serialisable dict, includes `escalated` |

Under concurrent calls of one step, `last_result` holds the run that finished last. Pass
`on_result=callable` to receive each `CascadeResult` in turn. Each tier run also logs one
`logger.info` line on `promptchain.utils.cascade`.

## When every tier fails

- `on_exhausted=<Branch>` runs on the original input and the cascade returns its text.
- Without it, the cascade raises `CascadeExhausted`. `exc.result` holds every tier run, so tier
  errors (an expired key, a timeout) stay visible there and in `str(exc)`.

## Defaults and limits

- Labels default to `chain.models[0]` (or the dict `name`) for a `PromptChain`, else the
  callable's `__name__`; override with `labels=[...]`.
- The step returns plain text, so it composes with any next step.
- `decision_check` needs a reachable backend; the offline example above needs none.
