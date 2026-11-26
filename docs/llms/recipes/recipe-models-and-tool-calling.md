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
"ollama_chat/glm-5.3-flash:cloud"                       # OpenAI
"ollama_chat/glm-5.3-flash:cloud"     # Anthropic
"gemini/gemini-2.5-pro"                    # Google Gemini (NOT "google/...")
"ollama/llama3.1:8b"                       # Ollama (local)
"ollama/qwen2.5:7b"                        # Ollama
"openrouter/<model_id>"                    # OpenRouter
"groq/<model_id>"                          # Groq
```

**Common mistake:** dropping the provider prefix (`"ollama_chat/glm-5.3-flash:cloud"` instead of `"ollama_chat/glm-5.3-flash:cloud"`). Always provider-prefix.

## B. Tool-calling capability matrix

This table reflects what's stable and broadly true. **Always verify against the model's current docs** — capability flips are common (e.g. an Ollama model gaining function-calling in a point release).

| Provider | Tool-calling reliability | Notes |
|---|---|---|
| **OpenAI** (`ollama_chat/glm-5.3-flash:cloud`, `ollama_chat/glm-5.3-flash:cloud`, `ollama_chat/glm-5.3-flash:cloud*`, `ollama_chat/glm-5.3-flash:cloud`) | ⭐⭐⭐ Strongest | The reference implementation. The repo's own comment: `"Using OpenAI for reliable tool calling"`. Pick OpenAI when the agent must reliably call tools. |
| **Anthropic** (`ollama_chat/glm-5.3-flash:cloud`, `ollama_chat/glm-5.3-flash:cloud`, `claude-opus-*`) | ⭐⭐⭐ Strong | Native tool use. Slightly different schema under the hood; LiteLLM normalises it. |
| **Gemini** (`gemini/gemini-2.5-pro`, `gemini-1.5-flash-8b`, `gemini-2.0-flash-exp`) | ⭐⭐ Good but variable | `gemini-2.5-pro` and `1.5-flash-8b` support tool calling; some experimental models do not. The repo uses Gemini for cost-saving (`fallback_model="gemini/gemini-1.5-flash-8b"` is 33x cheaper than 2.5-pro). |
| **Ollama** (`ollama/llama3.1`, `ollama/qwen2.5`, `ollama/mistral-nemo`, etc.) | ⭐ Model-dependent | **Most Ollama models DO NOT support tool-calling.** Models that do (as of late 2025): `llama3.1`, `qwen2.5`, `mistral-nemo`, `command-r`. Models that don't (or only partially): older `llama2`, `phi`, `gemma:2b`, etc. |
| **LocalAI / Llamacpp** | ⭐ Same as Ollama | Capability tracks the underlying GGUF model, not the runtime. |
| **OpenRouter** | ⭐⭐ Depends on routed model | If OpenRouter routes to OpenAI/Anthropic/Gemini, capability matches that. |

## C. What the repo's own examples use (provable, ground truth)

Mined from `examples/*.py` — these are the model strings the package's own canonical examples actually run:

```text
ollama_chat/glm-5.3-flash:cloud
ollama_chat/glm-5.3-flash:cloud
ollama_chat/glm-5.3-flash:cloud
ollama_chat/glm-5.3-flash:cloud
ollama_chat/glm-5.3-flash:cloud
ollama_chat/glm-5.3-flash:cloud
ollama_chat/glm-5.3-flash:cloud   # default in promptchain/utils/chain_builder.py
gemini/gemini-2.0-flash-exp
gemini/gemini-2.5-pro
gemini/gemini-1.5-flash-8b       # cheapest Gemini, 33x cheaper than 2.5-pro
```

**No Ollama models appear in `examples/`** — the package supports them via `OllamaModelManager` but doesn't ship Ollama examples. If you're using Ollama, see §E below.

## C2. Naming models once: the models file

PromptChain names no model. Instead of passing a model to every component, list them in one YAML file the user owns: `$PROMPTCHAIN_MODELS_CONFIG`, else `~/.promptchain/models.yaml`. Write a template with every role and empty values using `promptchain models init` (or `python -m promptchain.utils.model_roles init`); `promptchain models show` prints each role, its resolved model, its endpoint and where it came from. The file is YAML (PyYAML is a dependency). A missing file is an empty config; a malformed file raises an error that names the file.

```yaml
default: <provider/model>             # required roles only
endpoints:                            # optional: where a model runs
  gov-ollama:   {api_base: "http://<host>:<port>/<ollama-lane-path>"}                       # litellm ollama_chat/<tag>
  gov-llamacpp: {api_base: "http://<host>:<port>/<llamacpp-lane-path>/v1", provider: openai} # OpenAI-compatible; model = alias
  local:        {api_base: "http://localhost:11434"}
roles:
  supervisor: <provider/model>                                    # plain string: provider/model as usual
  state_agent: {model: <tag>, endpoint: gov-ollama}              # a role on the governor Ollama lane
  judge:       {model: <alias>, endpoint: gov-llamacpp}          # a role on the llamacpp lane
```

Precedence for one component: **explicit argument > `roles.<role>` > `default` (required roles only) > error**. An optional role never reads `default`: its feature stays off until the role is set or an argument is passed, so setting `default` cannot start extra paid calls. A required role that nothing supplies raises `ValueError` naming the parameter, the role key and the file path.

A role value (and `default`) is a string or a mapping `{model, endpoint}`. An endpoint is `api_base` plus an optional `provider` (passed to litellm as `custom_llm_provider`); an unknown endpoint name is an error that names the role and the endpoint. Endpoints hold no `num_ctx` or `max_tokens`. An explicit model argument carries no endpoint. In code: `resolve_model_spec(role, explicit, required=, param=)` returns `ModelSpec(model, api_base, extra_kwargs, source)`; `.as_model()` gives a value for `PromptChain(models=[...])` (a `{"name", "params"}` dict when an endpoint applies); `resolve_model(...)` returns the string only. The CLI and TUI apply the file's `default` (and `config.json`'s `default_model`) by model name only; they do not apply endpoints. `lightrag` and `lightrag_embedding` also take the name only.

| Role | Need | Component and parameter |
|---|---|---|
| `agent` | required | agent_comms LLMAgent / agent() / create_from_template |
| `orchestrator` | required | agent_comms orchestrator() / captain() |
| `supervisor` | required | AgentChain supervisor_model, OrchestratorSupervisor model_name |
| `synthesizer` | required | AgentChain synthesizer_config['model'] |
| `history_summarizer` | required | HistorySummarizer model |
| `state_agent` | required | StateAgent / create_state_agent model_name |
| `prompt_engineer` | required | PromptEngineer evaluator_model |
| `chain_builder` | required | DynamicChainBuilder model step |
| `chain` | required | ChainExecutor / ChainFactory / ChainBuilder llm_model |
| `agentic_step` | required | PromptChain agentic step with no models or model_name |
| `router` | required | YAML orchestration.router.model (router mode) |
| `lightrag` | required | LightRAGConfig.model_name |
| `lightrag_embedding` | required | LightRAGConfig.embedding_model |
| `summarizer` | optional | AgenticStepProcessor / ContextManagementConfig summarizer_model |
| `prompt_improver` | optional | PromptEngineer improver_model (falls back to the evaluator) |
| `judge` | optional | BranchingConfig.judge_model |
| `multi_hop_synthesizer` | optional | MultiHopConfig.synthesizer_model |

## D. Picking a model — decision tree

```
Does this step / chain register tool-calling functions, MCP tools, or use AgenticStepProcessor with tools?
├─ YES → Pick a tool-calling-capable model:
│        ├─ Reliability matters most  → ollama_chat/glm-5.3-flash:cloud or ollama_chat/glm-5.3-flash:cloud
│        ├─ Cost matters most         → gemini/gemini-1.5-flash-8b ($0.0375/1M)
│        ├─ Big context window needed → gemini/gemini-2.5-pro (1M ctx) or ollama_chat/glm-5.3-flash:cloud (200K)
│        └─ Local / offline           → ollama/llama3.1 or ollama/qwen2.5 (verify model card says "tools")
└─ NO (pure string-template chain, no tools) →
         Any model works. Optimize for cost: ollama_chat/glm-5.3-flash:cloud or gemini/gemini-1.5-flash-8b.
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
| `ollama_chat/glm-5.3-flash:cloud` | `ollama_chat/glm-5.3-flash:cloud` | 5x | ✅ |
| `ollama_chat/glm-5.3-flash:cloud` | `ollama_chat/glm-5.3-flash:cloud` | 22x | ✅ (ollama_chat/glm-5.3-flash:cloud = reasoning model) |
| `ollama_chat/glm-5.3-flash:cloud` | `ollama_chat/glm-5.3-flash:cloud` | ~12x | ✅ |

Mixing providers in a two-tier setup works but requires both API keys present.

## G. Health-probe before you start

Run **`python scripts/check_keys.py`** to verify every configured key works AND list which models your account actually has access to. Output lands in `scripts/scratch/api-status.md` (gitignored). Use `--no-probe` for a free env-presence check that doesn't burn quota.

This is the fastest fix for "I picked `ollama_chat/glm-5.3-flash:cloud` and got 'model not found'" — the probe lists what your key actually sees.

## H. Required env vars per provider

| Provider | Env vars |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GOOGLE_API_KEY` (NOT `GEMINI_API_KEY` — LiteLLM uses `GOOGLE_API_KEY`) |
| Ollama | none (local server); `OLLAMA_HOST` if non-default |
| OpenRouter | `OPENROUTER_API_KEY` |
| Groq | `GROQ_API_KEY` |

## Common failures + fix

- **`litellm.BadRequestError: model='ollama_chat/glm-5.3-flash:cloud' not found`** — missing provider prefix. Use `"ollama_chat/glm-5.3-flash:cloud"`.
- **Tool-call function never executes, agent just hallucinates a response in plain text** — the chosen model doesn't support tool-calling. Switch to one of the ⭐⭐⭐ entries in §B.
- **Two-tier routing produces garbage on simple sub-tasks** — your `fallback_model` is too weak (e.g. `phi:2b`). Use a fallback that still supports the language complexity of the task; `ollama_chat/glm-5.3-flash:cloud` and `gemini-1.5-flash-8b` are safe defaults.
- **Ollama model loaded but inference is slow/OOM** — enable `model_management` with `auto_unload_models=True` so models swap out between steps.
- **`API key not found`** for Gemini despite `GEMINI_API_KEY` being set — LiteLLM expects `GOOGLE_API_KEY`. Rename it (or `export GOOGLE_API_KEY="$GEMINI_API_KEY"` in your shell).

## Where to read the source-of-truth

- `promptchain/utils/model_management.py:27` — `ModelProvider` enum (local-model providers only)
- `promptchain/utils/ollama_model_manager.py:22` — `OllamaModelManager`
- `examples/two_tier_routing_demo.py` — measured cost numbers + working primary/fallback pairings
- LiteLLM provider list — `https://docs.litellm.ai/docs/providers` (canonical capability data)
- `scripts/check_keys.py` — probe every configured provider, list available models per your keys
