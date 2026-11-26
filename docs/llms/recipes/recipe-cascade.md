---
id: recipe-cascade
title: Verify-then-escalate cascade (cheap model first, frontier only on failure)
when: A step can often be answered by a cheap model or function, an output check can tell good from bad, and you want to pay for the expensive tier only when the check fails.
api_version: 0.6.1+
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
PromptChain(models=["ollama_chat/glm-5.3:cloud"], instructions=["{input}"])], check=..., name="answer")`.

## Cascade vs two-tier vs `route_on`

| need | use |
|---|---|
| Try cheap, verify the OUTPUT, escalate on a bad output | `cascade` |
| `AgenticStepProcessor` picks a model per call by task complexity (heuristic, does not inspect output) | `fallback_model` two-tier routing in the agentic step |
| Pick a branch BEFORE running anything, from a measured label or confidence | `decision` + `route_on` |

## The check contract

- Signature `check(step_input: str, output: str)`, in that order. `step_input` is the text
  the cascade step received. `output` is the tier's text.
- Return a `bool`, or a 2-item tuple/list `(bool, reason)` whose first item is a `bool`. Any
  other return value (a string, `None`, an int) raises `TypeError`. The reason lands on
  `TierAttempt.reason`.
- The check may be sync or `async`.
- **A check that raises propagates out of the cascade.** A broken verifier is never read as
  pass or fail, and no later tier runs. The cascade first stores the failure on the tier run
  (`error="check raised: ..."`), ends the run with no answer (`last_result`, `on_result`),
  and re-raises. Catch inside the check if you want a verdict.
- A **tier** that raises is different: the cascade stores the exception on `TierAttempt.error`
  (`"ExcType: message"`) and on `TierAttempt.exception`, then moves on to the next tier. A tier that returns
  `None` also escalates, with `error` set.

Ready-made check: `decision_check(backend, *, instructions, accept, min_probability=None)`.
It asks one boolean question through a decision backend (see `recipe-decision-step`) with the
state `<input>...</input><output>...</output>`; the cascade escapes `<` and `>` inside both fields to
`&lt;` and `&gt;`. Without `min_probability`, the output passes when the answer is True and
the measured **confidence** is at least `accept`.

Confidence is not the probability. A backend that gives only p(True) yields an entropy
confidence. Examples: p=0.9 gives 0.53, p=0.95 gives 0.71, p=0.97 gives 0.81.
With `accept=0.9` the output then needs p >= 0.988.

To threshold on the probability itself, pass `min_probability`.
The output then passes when the answer is True and p(True) is at least that value.
The check ignores `accept` in that mode. `accept` and `min_probability` must be numbers within
0..1; the function rejects booleans. A malformed backend reply raises `DecisionError`.

```python
from promptchain.utils.decision_step import SystemOneBackend
from promptchain.utils.cascade import decision_check

backend = SystemOneBackend(model="tev1:4b")
by_confidence = decision_check(backend, instructions="Does the output fully answer the input?",
                               accept=0.9)
by_probability = decision_check(backend, instructions="Does the output fully answer the input?",
                                accept=0.9, min_probability=0.95)
```

Test-based check: extract the code from the model text, run the tests in an executor, and
pass on exit code 0. The check is async and runs the blocking executor in a worker thread so
the event loop stays free. `LocalExecutor` runs on the host with no sandbox: use it for
**trusted code only**. For untrusted model output use `DockerExecutor` (same `write_file` and
`run` interface; see `promptchain.utils.docker_executor`).

```python
import asyncio
from promptchain.utils.test_loop_chain import LocalExecutor, extract_code   # trusted code only

TESTS = "from solution import f\ndef test_f():\n    assert f(2) == 4\n"

def _run_tests(code: str):
    with LocalExecutor() as ex:
        ex.write_file("solution.py", code)
        ex.write_file("test_solution.py", TESTS)
        res = ex.run("python -m pytest -q test_solution.py")
    return res.exit_code == 0, res.output[-200:]

async def tests_pass(step_input: str, output: str):
    code = extract_code(output, "python")          # drops the code fence and prose
    return await asyncio.to_thread(_run_tests, code)
```

Do not use `CoVeVerifier` as a check: it verifies tool calls only and passes a plain text answer.

## Reading the result

Pass `on_result=callable`: it receives one `CascadeResult` per run, also when the run ends in
`CascadeExhausted`, in a check error, or in an `on_exhausted` answer. An `on_result` exception
does not discard the answer; the cascade logs it. Each tier run logs one line on
`promptchain.utils.cascade` (`warning` with traceback when a tier raises).

```python
results = []
step = cascade([cheap, frontier], check=check, name="answer", on_result=results.append)
```

`step.last_result` holds the latest `CascadeResult` (reset to `None` at the start of each
run). Under concurrent calls of one step it holds whichever run finished last, so concurrent
callers read `on_result` instead.

| field | meaning |
|---|---|
| `name`, `output` | cascade name; the returned text (`None` when the run produced no answer) |
| `answered_by`, `tier` | label and index of the passing tier; `answered_by="on_exhausted"` with `tier=None` for a fallback answer |
| `attempts` | list of `TierAttempt(tier, label, passed, reason, elapsed_s, output, error, exception)` |
| `escalated` | property: more than one tier ran |
| `to_dict()` | JSON-serialisable dict, includes `escalated`, omits `exception` |

## When every tier fails

- `on_exhausted=<Branch>` runs on the original input and the cascade returns its text.
- Without it, the cascade raises `CascadeExhausted`, chained (`__cause__`) to the last tier
  exception. `exc.result` holds every tier run, so tier errors (an expired key, a timeout)
  stay visible there, in `exc.result.attempts[i].exception` and in `str(exc)`.
- The cascade records the `on_exhausted` answer: `answered_by="on_exhausted"`, `output` set.

## Defaults and limits

- Labels default to `chain.models[0]` (or the dict `name`) for a `PromptChain`, else the
  callable's `__name__`; override with `labels=[...]`.
- The step returns plain text, so it composes with any next step.
- `decision_check` needs a reachable backend; the offline example above needs none.
