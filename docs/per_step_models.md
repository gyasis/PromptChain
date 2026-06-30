# Per-Step Models — give each chain step its own model (and params)

PromptChain lets **every model-step run on its own model**. You do this by passing a
`models=[...]` list that lines up 1:1 with your model-instruction steps. As the chain
executes, an internal `model_index` counter walks the list — step 1 uses `models[0]`,
step 2 uses `models[1]`, and so on.

This is the mechanism behind `models=["openai/gpt-4o", "anthropic/claude-opus-4-8", ...]`.
It's distinct from `model_management` (which governs **VRAM load/unload** of Ollama models
between steps — see [`model_management.md`](model_management.md)). Per-step models is purely
*which model answers which step*.

## The shape

```python
from promptchain import PromptChain

chain = PromptChain(
    models=[
        "openai/gpt-4o",                          # step 1's model (plain string)
        {                                          # step 2: its own model AND its own params
            "name": "anthropic/claude-opus-4-8",
            "params": {"temperature": 0.2, "max_tokens": 2000},
        },
        "ollama/qwen2.5:7b",                       # step 3's model
    ],
    instructions=[
        "Brainstorm ideas about: {input}",         # → openai/gpt-4o
        "Critique the ideas: {input}",             # → claude-opus-4-8 (temp 0.2)
        "Write the final code: {input}",           # → ollama/qwen2.5:7b
    ],
)
```

Each entry in `models` can be:

- a **plain string** — `"openai/gpt-4o"` → that step runs on this model with default params.
- a **dict** — `{"name": "...", "params": {...}}` → that step gets its **own model and its
  own model_params** (temperature, max_tokens, etc.). The params are passed through to the
  LiteLLM call for that step only.

## The three rules

1. **Count must match.** `len(models)` must equal the number of **model steps** — i.e.
   instructions that are *not* a Python function, *not* an `AgenticStepProcessor`, and *not*
   a `ChainCall`. Those step types are skipped in the count (functions run no LLM; agentic
   steps set their model separately via `model_name`). Mismatch raises a `ValueError` at
   construction time.

2. **One model → repeated for all.** If you pass a **single** model but multiple model
   instructions, PromptChain auto-expands it so every step uses that one model:

   ```python
   PromptChain(
       models=["openai/gpt-4o-mini"],             # one model
       instructions=["Step 1: {input}", "Step 2: {input}", "Step 3: {input}"],
   )  # → all three steps run on gpt-4o-mini
   ```

3. **Order is execution order.** The model list is consumed in sequence by `model_index`,
   which resets at the start of each `process_prompt` / `process_prompt_async` run. The Nth
   model step always pulls the Nth entry — so keep `models` and `instructions` aligned.

## Agentic steps set their model separately

An `AgenticStepProcessor` does **not** draw from the `models` list. It carries its own model
(via `model_name=`, plus `fallback_model=` for two-tier routing). That's why agentic steps are
excluded from the count in rule 1. A mixed chain looks like:

```python
chain = PromptChain(
    models=["openai/gpt-4o"],                      # ONE entry — for the single string step below
    instructions=[
        "Summarize: {input}",                      # ← consumes models[0]
        AgenticStepProcessor(                       # ← brings its OWN model, NOT from models[]
            objective="Research and verify the summary",
            model_name="gemini/gemini-2.5-pro",
            fallback_model="gemini/gemini-1.5-flash-8b",
        ),
    ],
)
```

## Source of truth (code)

- Constructor — `promptchain/utils/promptchaining.py:286-324`
  - per-entry string/dict handling: `:308-314`
  - single-model auto-repeat: `:303-305`
  - count validation + `ValueError`: `:318-324`
- Per-step selection at execution — `promptchain/utils/promptchaining.py:1669-1671`
  ```python
  model = self.models[self.model_index]
  step_model_params = self.model_params[self.model_index]
  self.model_index += 1   # advance to the next model step
  ```
- `model_index` reset per run — `:481-483` (`reset_model_index`), called at `:751`.

## Related

- [`recipe-models-and-tool-calling`](../recipes) — *which* model string to pick per step
  (provider prefixes, tool-calling capability matrix).
- [`model_management.md`](model_management.md) — VRAM-aware load/unload of Ollama models
  between steps (a different concern from per-step model selection).
- Two-tier routing (`AgenticStepProcessor` `enable_two_tier_routing=True`) — per-step primary
  + cheap fallback model inside an agentic step.
