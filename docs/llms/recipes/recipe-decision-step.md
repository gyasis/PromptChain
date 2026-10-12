---
id: recipe-decision-step
title: Typed decision step + route on measured confidence
when: A chain must branch, gate or classify on a closed question (pick a label, yes/no, a score) and you want a real probability, not a model's self-reported confidence.
api_version: 0.10.0+
---

# Decision Step

`decision()` asks typed questions about the step input and gets back a **measured**
distribution + confidence from a decision backend. `route_on()` branches on it. Both are
plain Callable steps: they consume **no model slot**, so a chain made only of them takes
`models=[]`.

```python
import asyncio
from promptchain import PromptChain
from promptchain.utils.decision_step import decision, route_on, SystemOneBackend, hunch_backend

QUESTIONS = {
    "dept": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "charges, invoices, refunds",
                          "technical": "bugs, errors, outages",
                          "account": "login, password, profile"}},
    "urgent": {"type": "boolean", "instructions": "Does the customer convey urgency?"},
}

async def main():
    chain = PromptChain(
        models=[],                                   # no string instructions -> no models
        instructions=[
            decision(QUESTIONS, backend=SystemOneBackend(model="tev1:4b"), name="triage"),
            route_on("dept", accept=0.9, review=0.6,  # REQUIRED: set per task from an eval
                     on_accept={"billing": billing_chain,      # a PromptChain or a Callable
                                "technical": tech_step,
                                "account": account_step},
                     on_review=send_to_human,
                     on_escalate=escalate),
        ],
    )
    print(await chain.process_prompt_async("I was charged twice, please refund."))

asyncio.run(main())
```

## Question types
| type | `criteria` | answer `value` |
|---|---|---|
| `choice` | dict `{label: description}`, >= 2 | the chosen label |
| `boolean` | — | `True` / `False` |
| `score` | list of levels, >= 2 | expected value over levels |

## Backends (swap by config, same reply shape)
| backend | what | when |
|---|---|---|
| `SystemOneBackend(model="tev1:4b")` | Ollama >= 0.35 `/v1/systemone`; LM fine-tuned to emit one option letter (2–24 options) | **default** — best accuracy + calibration measured |
| `hunch_backend(model=..., vote=6)` | `hunch` CLI, next-token logprobs from an ordinary LM | second opinion; use `vote` if you threshold on it |
| `CliBackend([...], model=..., model_env=...)` | any CLI with `<cmd> - -q <json> --raw` | custom backends |

The model is **required** — there is no env-var default (R-PC4).

## The decision record
`decision()` returns a JSON string (chain steps return strings):
```json
{"decision": "triage", "backend": "systemone", "model": "tev1:4b", "elapsed_s": 0.71,
 "state": "<the step input>",
 "answers": {"dept": {"type": "choice", "value": "billing", "confidence": 0.975,
                      "probabilities": {"billing": 0.98, "technical": 0.01, "account": 0.01}}}}
```
`state` is carried so `route_on` hands the ORIGINAL text to the chosen branch.

## Where this comes from (Jev-style decisions)
This is the **Jev** decision model brought into PromptChain: a closed question goes to a
decision engine, and the probability comes from the model's internals (the next-token
distribution over option letters, from a model trained for it (`tev1`) or an ordinary LM (`hunch`)), not from a number the model writes. PromptChain is the **caller**; the
engines (`tev1` via `/v1/systemone`, the `hunch` CLI) don't know PromptChain exists. Hosted
Jev (Vercel) is retired, and there is deliberately no backend for it.

## Binary, multiclass, or score?
- **`boolean`**: a yes/no gate ("is this urgent?"). Chance is 50%.
- **`choice`**: **multiclass, exactly one** of N mutually exclusive labels. Chance is 1/N.
  It is *not* multi-label: if several things can be true at once, ask several `boolean`
  questions in the same `decision()` step (each is answered independently, one call).
- **`score`**: ordered levels (`["low","medium","high"]`); the value is the expected level.

`confidence` is the engine's own number when it sends one (`tev1` does). Only when an engine
returns probabilities without a confidence does the step derive it as `1 - entropy / log(N)`
(0 = flat spread, 1 = certain, so a 50/50 boolean is 0.0). Either way, chance falls as N grows,
so measure thresholds per task rather than reusing one across label counts.

## How many labels? (measured)
Label count is the difficulty axis, not the domain (`tev1:4b`, external banking77 sets):

| labels | accuracy | ECE |
|---|---|---|
| 6 | 98.3% | .005 |
| 20 | 87.5% | .098 |

More labels mean more confusable neighbours and probability spread thinner, so fewer items
clear `accept` and calibration degrades. They save calls and route more finely. Write a real
description for each label in `criteria`: it is what the engine scores against.

## Nested decision trees
A `route_on` branch may be a PromptChain that starts with its own `decision()`, so a
hierarchy is just composition: each level keeps N small, gets its own thresholds, and a
low-confidence root escalates before deeper work runs.

```python
billing_tree = PromptChain(models=[], instructions=[
    decision({"issue": {"type": "choice", "instructions": "Which billing issue?",
                        "criteria": {"refund": "wants money back",
                                     "invoice": "needs a document"}}},
             backend=SystemOneBackend(model="tev1:4b"), name="billing_issue"),
    route_on("issue", accept=0.9, review=0.6,
             on_accept={"refund": refund_step, "invoice": invoice_step},
             on_review=send_to_human, on_escalate=escalate),
])
# root: route_on("dept", ..., on_accept={"billing": billing_tree, ...})
```

Costs: errors multiply down a path (0.98 × 0.98 ≈ 0.96), each level adds a call (~0.7 s on
`tev1:4b`), and a wrong parent sends the item into the wrong subtree. **Unmeasured:** that a
2-level tree beats one flat 20-way question. Run both through `labdecide` before relying on
it. Tested: `test_nested_decision_tree_routes_into_a_second_decision_chain`.

## Common failures + fix
- **`DecisionError: answer 'sales' is not one of [...]`** — the backend returned a label outside
  `criteria`. This is raised on purpose; never mapped to a guess.
- **`route_on(...) must follow a decision step that asks it`** — `route_on` must come directly
  after a `decision` step that includes that question.
- **Everything lands in `accept` at confidence ~1.0** — saturated backend (plain `hunch` on a
  small model). Use `SystemOneBackend` or `hunch_backend(..., vote=6)`.
- **Choosing thresholds** — measure the task first (harness-lab `labdecide` +
  `labdecide-report.py --reliability`) and read the cut from the reliability table. Coverage at
  a fixed 0.9 swung 73–100% across domains in the bake-off.
