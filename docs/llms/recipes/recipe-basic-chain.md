---
id: recipe-basic-chain
title: Basic single-step PromptChain
when: One LLM step, string in / string out. The "hello world" of PromptChain.
api_version: 0.6.1+
---

# Basic Chain

```python
import asyncio
from promptchain import PromptChain

async def main():
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Summarise this in 3 bullets:\n\n{input}"],
        verbose=True,
    )
    result = await chain.process_prompt_async("Long article text here…")
    print(result)

asyncio.run(main())
```

## Sync version

```python
from promptchain import PromptChain

chain = PromptChain(
    models=["openai/gpt-4o-mini"],
    instructions=["Summarise this in 3 bullets:\n\n{input}"],
)
print(chain.process_prompt("Long article text here…"))
```

## Common failures + fix

- **`ValueError: Number of models (1) must match…`** — your instruction count doesn't match `models` length. Either pass one model per string instruction, or pass a single-element `models` list (auto-expands).
- **`litellm.BadRequestError: model='gpt-4o' not found`** — missing provider prefix. Use `"openai/gpt-4o"`.
- **Empty output / silent hang** — forgot `await` on `process_prompt_async`. The result is a coroutine, not a string.
