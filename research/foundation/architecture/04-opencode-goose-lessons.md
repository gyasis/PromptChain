# opencode + Goose — model-agnostic harness lessons (corpus addition)

opencode (`sst/opencode`) + Goose (`block/goose`) are open-source, **model-agnostic** harnesses — the
closest precedents to PromptChain — so their prompts are the best references for the portable base, the
per-family adapter, and weak/local-model handling. Saved to `research/system-prompts/{opencode,goose}/`.

## They validate our design
- **Per-family prompt variants** (opencode `routing.ts` → anthropic/gpt/gemini/beast/codex/kimi/default.txt by
  model-ID) → Constraint C. They go *beyond* a format adapter — family-tuned variants over a `default.txt`
  fallback. **`default.txt` ≈ our model-agnostic core.**
- **Tiered core vs tiny** (Goose `system.md` vs `tiny_model_system.md`) → Constraint B; the tiny variant even
  swaps the loop protocol (`$`-prefixed shell instead of tool calls) for the smallest models.
- **Turn-context + budget throttle** (Goose `<turn-context>`: time/cwd/compaction/turn-budget + "as budget gets
  low, become more direct, batch calls, assume") → Constraint D + per-turn goal re-injection.
- **Lazy hint injection** (Goose loads `.goosehints`/AGENTS.md only when the agent touches a subdir) →
  Constraint A progressive disclosure.
- **Modular assembly** (Goose MiniJinja: mode as a template var, keyed-extras `IndexMap` idempotent injections,
  per-extension `## Name / ### Instructions` blocks, user-overridable templates) → our static-base + dynamic
  add-ons, validated (DSPy-style composition; a template engine is a viable build).
- **Planner/Executor fresh-context** (both) → big-brother + Ralph. **Subagent "cannot spawn subagents" +
  `max_turns`** → no-nested-spawn + turn budget. **Compaction re-read** → Document-&-Clear. **Cache opt**
  (hour-truncated timestamp + stable ordering) → token economy. **Unicode-tag sanitization** → injection safety.

## NEW capability to ADOPT — Toolshim (tool-calling fallback for non-native models)
Goose `toolshim.rs`: many local/weak models lack native function-calling. Two paths:
- **A — prompt-inject:** append tool schemas + *"specify a tool call as JSON `{name, arguments}`, one at a
  time"* to the system prompt; parse JSON from the model's TEXT output; re-serialize tool history as plain text
  (non-native APIs reject `tool_use`/`tool_result` blocks).
- **B — interpreter:** a second cheap model (Ollama structured-output) extracts the tool call from the text.

→ Essential for PromptChain's local models. It's a per-model adapter concern (Constraint C) + a feature.
**Add `tool_mode ∈ {native, shim_prompt, shim_interpreter}` to the jacket;** the Model Profiler's probe detects
native-tool-calling support and sets it. The `<tools>` section renders differently for shim models.

## How this changes Phase 4 / the design
- The **core foundation** we assemble = opencode `default.txt`-style portable base.
- The **adapter** may render family variants (not just format tweaks), with `default` as fallback.
- Add **`tool_mode`** to the jacket; `<tools>` renders the JSON-in-text protocol for shim models.
- A Jinja-style template engine is a viable build for static-base + keyed dynamic modules.
