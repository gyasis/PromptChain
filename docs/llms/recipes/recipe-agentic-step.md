---
id: recipe-agentic-step
title: Agentic step with internal reasoning loop
when: You need the LLM to plan, call tools, and iterate until an objective is met — without manually coding the loop.
api_version: 0.6.1+
---

# Agentic Step

`AgenticStepProcessor` runs an internal LLM loop with tool access. It's a single instruction in the parent chain — but inside, it can make many LLM calls. It does **NOT** consume a model slot in the parent chain.

```python
import asyncio
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.promptchaining import PromptChain


def list_files(directory: str) -> str:
    """List files in a directory."""
    import os
    return "\n".join(os.listdir(directory))


async def main():
    agentic = AgenticStepProcessor(
        objective="List the files in /tmp and return a one-line summary of how many are .log files.",
        max_internal_steps=8,
        model_name="openai/gpt-4o-mini",
        history_mode="progressive",      # recommended over "minimal" for multi-hop reasoning
    )

    chain = PromptChain(
        models=[],                       # no model slot needed; agentic step carries its own
        instructions=[agentic],
    )
    chain.register_tool_function(list_files)   # tool available inside the agentic loop

    result = await chain.process_prompt_async("start")
    print(result)


asyncio.run(main())
```

## Why no model in `models=[]`?

Because the *only* instruction is an `AgenticStepProcessor` and it has its own `model_name`. String instructions are the only kind that consume parent-chain model slots.

## Enabling cost / token / safety phases

`AgenticStepProcessor` supports four research-backed phases. All default to `False`:

```python
agentic = AgenticStepProcessor(
    objective="…",
    model_name="gemini/gemini-2.5-pro",
    fallback_model="gemini/gemini-1.5-flash-8b",
    enable_two_tier_routing=True,    # Phase 1: route simple sub-tasks to fallback (~60-70% cost cut)
    enable_blackboard=True,          # Phase 2: structured state instead of growing history (~72% token cut)
    enable_cove=True,                # Phase 3: pre-execution tool-call verification
    enable_checkpointing=True,       # Phase 3: stuck-state detection + rollback
    enable_tao_loop=True,            # Phase 4: explicit Think-Act-Observe phases
    enable_dry_run=True,             # Phase 4: predict tool output before executing
)
```

See `examples/two_tier_routing_demo.py` for measured cost numbers.

## Common failures + fix

- **`objective="hi please find me X"`** — the LLM treats the objective as the goal definition; write it as `"Find X and return Y as a JSON list"`, not as a conversational message.
- **Loop runs forever / hits step cap with no answer** — `max_internal_steps` too low for the task, OR the model can't see the right tool. Check `chain.local_tool_functions` and `chain.local_tools`.
- **History explodes in multi-agent flow** — won't happen. `AgenticStepProcessor` history is internally isolated by design (see `agentic_step_processor.py:142-199`). Only `final_answer` is exposed to downstream agents.
