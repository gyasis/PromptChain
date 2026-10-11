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
| `SystemOneBackend(model="tev1:4b")` | Ollama >= 0.35 `/v1/systemone`, real classifier heads | **default** — best accuracy + calibration measured |
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
