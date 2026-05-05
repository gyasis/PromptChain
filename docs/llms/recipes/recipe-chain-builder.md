---
id: recipe-chain-builder
title: ChainBuilder fluent API (agent-facing, self-writing chains)
when: You want to build, modify, or clone a `ChainDefinition` programmatically — especially when the AGENT itself is constructing a chain via tool calls. Bridge to the "watertight" north-star (PromptChain writes its own PromptChain code).
api_version: 0.6.1+
---

# ChainBuilder

`ChainBuilder` is **the agent-facing API** for self-writing chains. It is purpose-built so an LLM can call `create_chain` / `modify_chain` / `clone_chain` as registered tools, and the resulting chain is validated, versioned, and saved to disk.

**Source:** `promptchain/utils/chain_builder.py:40`

## Two ways to use it

### A. Fluent builder (human / direct code use)

```python
from promptchain.utils.chain_builder import ChainBuilder
from promptchain.utils.chain_models import ChainMode

chain_def = (
    ChainBuilder("research-and-summarize", version="v1.0", mode=ChainMode.STRICT)
        .description("Research a topic, then produce a 5-bullet summary")
        .llm_model("openai/gpt-4o-mini")
        .tags("research", "summary")
        .created_by("agent:claude-opus-4-7")
        .add_prompt("Research the topic and list 10 key facts: {input}")
        .add_prompt("Summarise the facts as 5 numbered bullets: {input}")
        .guardrails(max_steps=10, timeout_seconds=120)
        .build()                  # → ChainDefinition
)

# Save to disk via ChainFactory
from promptchain.utils.chain_factory import ChainFactory
factory = ChainFactory()
path = factory.save(chain_def)
print(f"Saved to {path}")
```

### B. Static tool methods (LLM agent calling via tool-call)

```python
from promptchain.utils.chain_builder import (
    ChainBuilder,
    get_chain_builder_tools,        # OpenAI function-schema list
    get_chain_builder_functions,    # name → callable mapping
)
from promptchain import PromptChain

# Register the chain-builder tools on a PromptChain so the LLM can self-author
chain = PromptChain(
    models=["openai/gpt-4o"],
    instructions=["The user wants a workflow. Build it via tools: {input}"],
)
for fn in get_chain_builder_functions().values():
    chain.register_tool_function(fn)
# (Or pass the schemas via add_tools; see PROMPTCHAIN_FOR_LLMS.md §7)

# The LLM can now emit:
#   create_chain(name="my-flow",
#                steps=[{"type":"prompt","content":"…"},
#                       {"type":"agentic","objective":"…","max_steps":5}])
# and a validated, versioned ChainDefinition is saved to disk.
```

## The 4 step types

| step `type` | required fields | when to use |
|---|---|---|
| `"prompt"` | `content` (template, may include `{input}`) | LLM call with a string template |
| `"chain"` | `chain_id` (e.g. `"my-chain:v1.0"` or `"my-chain"`) | Nest another saved chain as a step |
| `"function"` | `function_name` (must be registered on the executor) | Pure-Python step |
| `"agentic"` | `objective`, optional `max_steps`, optional `tools` | Internal reasoning loop. **REQUIRES `mode="hybrid"`** — auto-switched by the builder. |

## STRICT vs HYBRID mode

- `ChainMode.STRICT` (default) — rejects agentic steps. Used when you want a fully predictable, guardrailed workflow.
- `ChainMode.HYBRID` — allows agentic steps. The builder will auto-switch to HYBRID if you call `add_agentic(...)`.

## Static tool methods return `dict`

Unlike the fluent API which raises on error, the static methods (`create_chain`, `modify_chain`, `clone_chain`) return:

```python
# success
{"success": True, "vin": "<vin>", "model": "...", "version": "v1.0", "path": "/path/to/saved", "steps_count": 3}

# failure
{"success": False, "error": "Validation failed: ..."}
```

This is intentional — the LLM gets a structured response it can reason about, instead of having to recover from a Python exception.

## Common failures + fix

- **`ValueError: Chain must have at least one step`** at `.build()` — you forgot any `.add_*` call. At least one step is required.
- **Agentic step without HYBRID mode** — the builder logs a warning and switches the mode for you. If you want strict behaviour, drop the agentic step.
- **`Validation failed: …`** in the dict response — `ChainFactory.validate()` rejected something. Common: bad `prompt_id` reference, malformed `chain_id`, forbidden_patterns hit. Read the errors list and fix.
- **Forgetting `mode="hybrid"`** in the static `create_chain` form when adding an agentic step — wraps to a validation error. Pass `mode="hybrid"`.
- **Treating `version` as semver-major** — `modify_chain` auto-increments the *patch* (last segment). For major bumps, build a new chain with a new name.

## Where to read the source-of-truth

- `promptchain/utils/chain_builder.py` — class and static tool methods
- `promptchain/utils/chain_models.py:20-87` — `StepType`, `ChainMode`, `ChainStepDefinition`, `Guardrails`, `ChainDefinition`
- `promptchain/utils/chain_factory.py` — `ChainFactory.save`, `.resolve`, `.validate`
