---
id: recipe-models-and-tool-calling
title: Model selection — providers, tool-calling support, and the LiteLLM string format
when: You're choosing which model to put in `models=[...]`, `model_name=`, or `fallback_model=`. Picking the wrong one (e.g. an Ollama model without function-calling for an agentic flow) silently breaks tool calls.
api_version: 0.6.1+
---

# Models & Tool-Calling Cheat-Sheet

PromptChain delegates LLM calls to **LiteLLM**, so every model string is `"<provider>/<model>"`. The capability that matters most for PromptChain is **tool-calling support** — without it, `AgenticStepProcessor`, MCP tools, and `register_tool_function` all degrade to plain prompting.

## A. The string format (LiteLLM convention)

```python
"openai/gpt-4o-mini"                       # OpenAI
"anthropic/claude-3-5-sonnet-20241022"     # Anthropic
"gemini/gemini-2.5-pro"                    # Google Gemini (NOT "google/...")
"ollama/llama3.1:8b"                       # Ollama (local)
"ollama/qwen2.5:7b"                        # Ollama
"openrouter/<model_id>"                    # OpenRouter
"groq/<model_id>"                          # Groq
```

**Common mistake:** dropping the provider prefix (`"gpt-4o"` instead of `"openai/gpt-4o"`). Always provider-prefix.

## B. Tool-calling capability matrix

This table reflects what's stable and broadly true. **Always verify against the model's current docs** — capability flips are common (e.g. an Ollama model gaining function-calling in a point release).

| Provider | Tool-calling reliability | Notes |
|---|---|---|
| **OpenAI** (`openai/gpt-4o`, `gpt-4o-mini`, `gpt-5-*`, `o4-mini`) | ⭐⭐⭐ Strongest | The reference implementation. The repo's own comment: `"Using OpenAI for reliable tool calling"`. Pick OpenAI when the agent must reliably call tools. |
| **Anthropic** (`anthropic/claude-3-5-sonnet`, `claude-3-5-haiku`, `claude-opus-*`) | ⭐⭐⭐ Strong | Native tool use. Slightly different schema under the hood; LiteLLM normalises it. |
| **Gemini** (`gemini/gemini-2.5-pro`, `gemini-1.5-flash-8b`, `gemini-2.0-flash-exp`) | ⭐⭐ Good but variable | `gemini-2.5-pro` and `1.5-flash-8b` support tool calling; some experimental models do not. The repo uses Gemini for cost-saving (`fallback_model="gemini/gemini-1.5-flash-8b"` is 33x cheaper than 2.5-pro). |
| **Ollama** (`ollama/llama3.1`, `ollama/qwen2.5`, `ollama/mistral-nemo`, etc.) | ⭐ Model-dependent | **Most Ollama models DO NOT support tool-calling.** Models that do (as of late 2025): `llama3.1`, `qwen2.5`, `mistral-nemo`, `command-r`. Models that don't (or only partially): older `llama2`, `phi`, `gemma:2b`, etc. |
| **LocalAI / Llamacpp** | ⭐ Same as Ollama | Capability tracks the underlying GGUF model, not the runtime. |
| **OpenRouter** | ⭐⭐ Depends on routed model | If OpenRouter routes to OpenAI/Anthropic/Gemini, capability matches that. |

## C. What the repo's own examples use (provable, ground truth)

Mined from `examples/*.py` — these are the model strings the package's own canonical examples actually run:

```text
openai/gpt-4o
openai/gpt-4o-mini
openai/gpt-4
openai/gpt-5-mini
openai/gpt-5-nano
openai/o4-mini
openai/gpt-4.1-mini-2025-04-14   # default in promptchain/utils/chain_builder.py
gemini/gemini-2.0-flash-exp
gemini/gemini-2.5-pro
gemini/gemini-1.5-flash-8b       # cheapest Gemini, 33x cheaper than 2.5-pro
```

**No Ollama models appear in `examples/`** — the package supports them via `OllamaModelManager` but doesn't ship Ollama examples. If you're using Ollama, see §E below.

## D. Picking a model — decision tree

```
Does this step / chain register tool-calling functions, MCP tools, or use AgenticStepProcessor with tools?
├─ YES → Pick a tool-calling-capable model:
│        ├─ Reliability matters most  → openai/gpt-4o or openai/gpt-4o-mini
│        ├─ Cost matters most         → gemini/gemini-1.5-flash-8b ($0.0375/1M)
│        ├─ Big context window needed → gemini/gemini-2.5-pro (1M ctx) or anthropic/claude-3-5-sonnet (200K)
│        └─ Local / offline           → ollama/llama3.1 or ollama/qwen2.5 (verify model card says "tools")
└─ NO (pure string-template chain, no tools) →
         Any model works. Optimize for cost: openai/gpt-4o-mini or gemini/gemini-1.5-flash-8b.
```

## E. Using Ollama — extra wiring

Ollama models need a local Ollama server and (usually) an explicit base URL when not on `localhost:11434`. PromptChain has `OllamaModelManager` for VRAM-aware load/unload between steps.

```python
from promptchain import PromptChain
from promptchain.utils.model_management import ModelProvider

chain = PromptChain(
    models=["ollama/llama3.1:8b"],
    instructions=["Reply tersely: {input}"],
    model_management={
        "enabled": True,
        "provider": "ollama",
        "config": {"base_url": "http://localhost:11434"},
    },
    auto_unload_models=True,    # unload between steps to free VRAM
)
```

**Ollama tool-calling sanity check:** before relying on it, confirm the model card on the Ollama site lists "Tools" in its capabilities, or run a quick smoke test:

```bash
ollama run llama3.1 "Call this function: get_weather('Paris')"
```
If the response is a structured tool call, you're good. If it's a plain English answer pretending to call the function, the model doesn't support real tool-calling — pick a different one.

## F. Two-tier routing — the right pairings

`AgenticStepProcessor` Phase 1 (`enable_two_tier_routing=True`) needs a `model_name` (primary) and `fallback_model` (cheap). Best-tested pairings from the repo:

| Primary | Fallback | Cost ratio | Both tool-calling? |
|---|---|---|---|
| `gemini/gemini-2.5-pro` | `gemini/gemini-1.5-flash-8b` | 33x | ✅ |
| `openai/gpt-5-mini` | `openai/gpt-5-nano` | 5x | ✅ |
| `openai/o4-mini` | `openai/gpt-5-nano` | 22x | ✅ (o4-mini = reasoning model) |
| `anthropic/claude-3-5-sonnet` | `anthropic/claude-3-5-haiku` | ~12x | ✅ |

Mixing providers in a two-tier setup works but requires both API keys present.

## G. Required env vars per provider

| Provider | Env vars |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GOOGLE_API_KEY` (NOT `GEMINI_API_KEY` — LiteLLM uses `GOOGLE_API_KEY`) |
| Ollama | none (local server); `OLLAMA_HOST` if non-default |
| OpenRouter | `OPENROUTER_API_KEY` |
| Groq | `GROQ_API_KEY` |

## Common failures + fix

- **`litellm.BadRequestError: model='gpt-4o' not found`** — missing provider prefix. Use `"openai/gpt-4o"`.
- **Tool-call function never executes, agent just hallucinates a response in plain text** — the chosen model doesn't support tool-calling. Switch to one of the ⭐⭐⭐ entries in §B.
- **Two-tier routing produces garbage on simple sub-tasks** — your `fallback_model` is too weak (e.g. `phi:2b`). Use a fallback that still supports the language complexity of the task; `gpt-4o-mini` and `gemini-1.5-flash-8b` are safe defaults.
- **Ollama model loaded but inference is slow/OOM** — enable `model_management` with `auto_unload_models=True` so models swap out between steps.
- **`API key not found`** for Gemini despite `GEMINI_API_KEY` being set — LiteLLM expects `GOOGLE_API_KEY`. Rename it (or `export GOOGLE_API_KEY="$GEMINI_API_KEY"` in your shell).

## Where to read the source-of-truth

- `promptchain/utils/model_management.py:27` — `ModelProvider` enum (local-model providers only)
- `promptchain/utils/ollama_model_manager.py:22` — `OllamaModelManager`
- `examples/two_tier_routing_demo.py` — measured cost numbers + working primary/fallback pairings
- LiteLLM provider list — `https://docs.litellm.ai/docs/providers` (canonical capability data)
