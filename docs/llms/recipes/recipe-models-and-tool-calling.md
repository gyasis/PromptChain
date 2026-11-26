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
"ollama_chat/glm-5.3:cloud"                # Ollama (cloud tag, large)
"ollama_chat/glm-5.3-flash:cloud"          # Ollama (cloud tag, fast)
"ollama_chat/gemma4:12b"                   # Ollama (local)
"openai/<model_id>"                        # OpenAI
"anthropic/<model_id>"                     # Anthropic
"gemini/gemini-2.5-pro"                    # Google Gemini (NOT "google/...")
"openrouter/<model_id>"                    # OpenRouter
"groq/<model_id>"                          # Groq
```

**Common mistake:** dropping the provider prefix (`"glm-5.3:cloud"` instead of `"ollama_chat/glm-5.3:cloud"`). Always provider-prefix.

## B. Tool-calling capability matrix

This table reflects what's stable and broadly true. **Always verify against the model's current docs** — capability flips are common (e.g. an Ollama model gaining function-calling in a point release).

| Provider | Tool-calling reliability | Notes |
|---|---|---|
| **OpenAI** (`openai/<model_id>`) | ⭐⭐⭐ Strongest | The reference implementation. The repo's own comment: `"Using OpenAI for reliable tool calling"`. Pick OpenAI when the agent must reliably call tools. |
| **Anthropic** (`anthropic/<model_id>`) | ⭐⭐⭐ Strong | Native tool use. Slightly different schema under the hood; LiteLLM normalises it. |
| **Gemini** (`gemini/gemini-2.5-pro`, `gemini-2.5-flash-lite`, `gemini-2.5-flash`) | ⭐⭐ Good but variable | `gemini-2.5-pro` and `gemini-2.5-flash` support tool calling; some experimental models do not. The repo uses Gemini for cost-saving (`fallback_model="gemini/gemini-2.5-flash-lite"` is the cheap tier; check the provider price page). |
| **Ollama** (`ollama_chat/glm-5.3:cloud`, `ollama_chat/glm-5.3-flash:cloud`, `ollama_chat/gemma4:12b`, etc.) | ⭐ Model-dependent | **Not every Ollama model supports tool-calling.** Read the model's `capabilities` (the generated catalog, or `/api/show`) and use only models that list `tools`. Embedding models and older small models often do not. |
| **LocalAI / Llamacpp** | ⭐ Same as Ollama | Capability tracks the underlying GGUF model, not the runtime. |
| **OpenRouter** | ⭐⭐ Depends on routed model | If OpenRouter routes to OpenAI/Anthropic/Gemini, capability matches that. |

## C. What the repo's own examples use (provable, ground truth)

The package's examples name a model explicitly (there is no library default). Typical strings:

```text
ollama_chat/glm-5.3-flash:cloud
gemini/gemini-2.5-flash
gemini/gemini-2.5-pro
```

`ollama_chat/<tag>` (cloud and local Ollama tags) is the form the examples use. The model-manager helpers (`OllamaModelManager`) are optional; see §E below.

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
│        ├─ Reliability matters most  → a model whose catalog capabilities list `tools` (e.g. ollama_chat/glm-5.3:cloud)
│        ├─ Cost matters most         → gemini/gemini-2.5-flash-lite
│        ├─ Big context window needed → gemini/gemini-2.5-pro (1M ctx) or ollama_chat/glm-5.3:cloud (1M ctx, per the catalog)
│        └─ Local / offline           → ollama_chat/gemma4:12b (catalog lists tools, thinking, vision)
└─ NO (pure string-template chain, no tools) →
         Any model works. Optimize for cost: ollama_chat/glm-5.3-flash:cloud, a local model, or gemini/gemini-2.5-flash-lite.
```

## E. Using Ollama — extra wiring

Ollama models need a local Ollama server and (usually) an explicit base URL when not on `localhost:11434`. PromptChain has `OllamaModelManager` for VRAM-aware load/unload between steps.

```python
from promptchain import PromptChain
from promptchain.utils.model_management import ModelProvider

chain = PromptChain(
    models=["ollama_chat/gemma4:12b"],
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
ollama run gemma4:12b "Call this function: get_weather('Paris')"
```
If the response is a structured tool call, you're good. If it's a plain English answer pretending to call the function, the model doesn't support real tool-calling — pick a different one.

## F. Two-tier routing — the right pairings

`AgenticStepProcessor` Phase 1 (`enable_two_tier_routing=True`) needs a `model_name` (primary) and `fallback_model` (cheaper or faster). Example pairings (check each model's capabilities first):

| Primary | Fallback | Both tool-calling? |
|---|---|---|
| `gemini/gemini-2.5-pro` | `gemini/gemini-2.5-flash-lite` | ✅ |
| `ollama_chat/glm-5.3:cloud` | `ollama_chat/glm-5.3-flash:cloud` | ✅ (catalog: both list tools and thinking) |
| `ollama_chat/glm-5.3:cloud` | `ollama_chat/gemma4:12b` | ✅ (fallback runs locally; catalog lists tools) |

Mixing providers in a two-tier setup works but requires both API keys present.

## G. Health-probe before you start

Run **`python scripts/check_keys.py`** to verify every configured key works AND list which models your account actually has access to. Output lands in `scripts/scratch/api-status.md` (gitignored). Use `--no-probe` for a free env-presence check that doesn't burn quota.

This is the fastest fix for "I picked a model tag and got 'model not found'" — the probe lists what your key actually sees.

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

- **`litellm.BadRequestError: model='glm-5.3:cloud' not found`** — missing provider prefix. Use `"ollama_chat/glm-5.3:cloud"`.
- **Tool-call function never executes, agent just hallucinates a response in plain text** — the chosen model doesn't support tool-calling. Switch to one of the ⭐⭐⭐ entries in §B.
- **Two-tier routing produces garbage on simple sub-tasks** — your `fallback_model` is too weak (e.g. a model under 2B parameters). Use a fallback that still supports the language complexity of the task; a same-family smaller model such as `ollama_chat/glm-5.3-flash:cloud` is a reasonable starting point.
- **Ollama model loaded but inference is slow/OOM** — enable `model_management` with `auto_unload_models=True` so models swap out between steps.
- **`API key not found`** for Gemini despite `GEMINI_API_KEY` being set — LiteLLM expects `GOOGLE_API_KEY`. Rename it (or `export GOOGLE_API_KEY="$GEMINI_API_KEY"` in your shell).

## Where to read the source-of-truth

- `promptchain/utils/model_management.py:27` — `ModelProvider` enum (local-model providers only)
- `promptchain/utils/ollama_model_manager.py:22` — `OllamaModelManager`
- `examples/two_tier_routing_demo.py` — measured cost numbers + working primary/fallback pairings
- LiteLLM provider list — `https://docs.litellm.ai/docs/providers` (canonical capability data)
- `scripts/check_keys.py` — probe every configured provider, list available models per your keys
