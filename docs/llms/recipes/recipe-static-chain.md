---
id: recipe-static-chain
title: Static chain — pure-Python pipeline, zero LLM calls
when: You want PromptChain's chaining + observability + history machinery, but the work is purely deterministic Python (parsing, scoring, transformation). No LLM step.
api_version: 0.6.1+
---

# Static Chain

A `PromptChain` made entirely of Python functions. Useful as:
- A pre-/post-processor stage that gets MLflow tracking for free
- Deterministic pipeline experiments where you want the chain abstraction without paying for LLM calls
- An A/B baseline against an LLM-driven version of the same flow

## The pattern: `models=[]`

If every instruction is a callable (or `AgenticStepProcessor` carrying its own model), the chain needs **zero** entries in `models`. PromptChain only counts string instructions when matching models to slots.

```python
import asyncio
import json
import re
from promptchain import PromptChain


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def split_sentences(text: str) -> str:
    """Split into a JSON list of sentences."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return json.dumps(sents)


def score(text: str) -> str:
    """Return a JSON envelope with sentence count and char count."""
    data = json.loads(text)
    return json.dumps({"count": len(data), "chars": sum(len(s) for s in data)})


async def main():
    chain = PromptChain(
        models=[],                 # ← no models needed, all instructions are functions
        instructions=[normalize, split_sentences, score],
        store_steps=True,          # populate chain.step_outputs for inspection
        verbose=True,
    )
    result = await chain.process_prompt_async("Hello.   World!  How are you?")
    print(result)
    print(chain.step_outputs)      # see each stage's intermediate output


asyncio.run(main())
```

## Why use PromptChain instead of plain function composition?

Honest answer: **you don't always need to.** Choose this pattern when you want one of:

1. **Chainbreakers** — early termination on a condition:
   ```python
   def stop_if_too_short(step_no, current_output):
       if len(current_output) < 10:
           return (True, "too short", current_output)
       return (False, "", current_output)

   chain = PromptChain(models=[], instructions=[...], chainbreakers=[stop_if_too_short])
   ```
2. **Observability** — every step gets MLflow-tracked when `init_mlflow()` is called.
3. **Mixing LLM and non-LLM later** — start static, swap one function for a string instruction when you're ready to add an LLM stage. Code structure doesn't change.
4. **Step storage** — `store_steps=True` populates `chain.step_outputs` for debugging without needing to instrument the functions.

## Common failures + fix

- **`ValueError: Number of models (1) must match number of non-function/non-agentic instructions (0)`** — you passed `models=["openai/gpt-4o-mini"]` but every instruction is a callable. Use `models=[]`.
- **Function raises** — propagates and stops the chain (no automatic recovery). Wrap in try/except inside the function or use a chainbreaker.
- **`async def`-only function** — supported but the framework will run it inside its event loop. Prefer plain `def` for static functions; the chain itself is async.
