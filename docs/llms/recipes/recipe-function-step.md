---
id: recipe-function-step
title: Mix Python functions into a chain
when: You need deterministic post-processing (dedup, parsing, validation) between LLM calls.
api_version: 0.6.1+
---

# Function Step

A Python function is a valid `instruction`. It must accept a single string and return a string. **It does NOT consume a model slot.**

```python
import asyncio
import json
from promptchain import PromptChain


def to_uppercase_json(text: str) -> str:
    """Wrap whatever the LLM returned in a JSON envelope, uppercased."""
    return json.dumps({"value": text.upper()})


async def main():
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],   # ONE model for the ONE string instruction
        instructions=[
            "Give me a single short fact about the moon.",
            to_uppercase_json,           # python step — no model slot
        ],
        verbose=True,
    )
    result = await chain.process_prompt_async("start")
    print(result)


asyncio.run(main())
```

## Common failures + fix

- **`ValueError: Number of models (2) must match number of non-function/non-agentic instructions (1)`** — you passed two models because you have two instructions, but only the string instruction needs a model. Pass `models=["openai/gpt-4o-mini"]` (one entry).
- **Function returns a dict / list** — the next instruction's `{input}` substitution will get `repr()` of the object. Always return a string (use `json.dumps` if you need structured data).
- **Function raises an exception** — propagates and stops the chain. Wrap in try/except inside the function and return an error string if you want the chain to continue.
