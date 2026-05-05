---
id: recipe-multi-agent-router
title: Multi-agent system with router mode
when: You have multiple specialist agents and want an LLM router to pick the right one per turn.
api_version: 0.6.1+
---

# Multi-Agent Router

```python
import asyncio
from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain   # NOT in top-level — submodule import


async def main():
    researcher = PromptChain(
        models=["openai/gpt-4o"],
        instructions=["Research and return facts about: {input}"],
    )
    writer = PromptChain(
        models=["openai/gpt-4o"],
        instructions=["Write a polished paragraph about: {input}"],
    )

    agent_chain = AgentChain(
        agents={"researcher": researcher, "writer": writer},
        agent_descriptions={
            "researcher": "Use when the user wants facts gathered.",
            "writer":     "Use when the user wants a polished draft.",
        },
        router={
            "models": ["openai/gpt-4o-mini"],
            "instructions": [None, "Choose ONE of: researcher, writer. User: {input}"],
            "decision_prompt_template": "For input '{input}', return only the agent name.",
        },
        execution_mode="router",   # default; other options: "pipeline", "round_robin", "broadcast"
        verbose=True,
    )

    result = await agent_chain.process_input("Find recent papers on graph neural networks.")
    print(result)


asyncio.run(main())
```

## The four execution modes

| `execution_mode` | What it does |
|---|---|
| `"router"` | Picks ONE agent per turn (router = LLM, dict, or callable) |
| `"pipeline"` | All agents run sequentially; output of N feeds N+1 |
| `"round_robin"` | Cycle through agents, one per turn |
| `"broadcast"` | All agents run in parallel; results synthesized — **REQUIRES** `synthesizer_config={"model": "...", "prompt": "..."}` |

## Per-agent history (token saver)

```python
agent_chain = AgentChain(
    agents={...},
    agent_descriptions={...},
    router={...},
    agent_history_configs={
        "terminal_agent": {"enabled": False},   # save 30-60% tokens on a terminal-style agent
        "writer":         {"enabled": True, "max_tokens": 8000, "truncation_strategy": "keep_last"},
    },
)
```

## Common failures + fix

- **`from promptchain import AgentChain`** → `ImportError`. Use `from promptchain.utils.agent_chain import AgentChain`.
- **Calling `chain.process_prompt_async(...)`** on `AgentChain` → method doesn't exist. Use `await agent_chain.process_input(...)`.
- **`broadcast` mode without `synthesizer_config`** → `ValueError`. Required for that mode only.
- **`agent_descriptions` keys don't match `agents` keys** → `ValueError` at construction time.
