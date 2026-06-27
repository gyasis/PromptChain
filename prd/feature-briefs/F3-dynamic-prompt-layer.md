# Feature Brief F3 — Dynamic Prompt Layer (assemble the prompt through the formulas)

> **Purpose:** the locked, self-contained input for `/speckit-specify`. Everything below is DECIDED —
> formalize it, don't re-open the architecture. Source: PRD §7 · design
> `research/foundation/architecture/04-opencode-goose-lessons.md` + `00-design-constraints.md` (constraints A–D).
> **Epic order: build LAST (reads F2's profile/jacket; sits on the already-built static foundation).**

## Hand to /speckit-specify (the framing)
"Build the dynamic half of the two-layer prompt: a generator that takes (objective, tools, model),
reads the Model Profiler's jacket, and assembles the right prompt PER MODEL — tier, family adapter,
toolshim, token-budget compression, and weak-model Document-&-Clear longevity — keeping the existing
model-agnostic static base intact and the split purely additive."

## Context & dependencies
- **Depends on F2** — F3 reads the profile/jacket (tier, budget, family_adapter, tool_mode, compress@, max_turns, spawn_temp, role) F2 produces.
- **Built already (do not rebuild):** the **static base** foundation prompt is wired into `TUI_FOUNDATION_PROMPT` (`promptchain/prompts/tui_dynamic.py`). The loops (`ExternalLoop`, MicroPromptChain/RalphChain) and `ContextDistiller`/`HistorySummarizer` exist. F3 is the **dynamic add-on layer** on top.
- **Integration seam:** `DynamicTUIPromptGenerator.generate()` / `BasePromptBuilder` (`agentic_step_processor.py:1006`). The static base stays intact; the dynamic split is additive.

## Decisions ALREADY MADE (locked — do not re-research)
1. **Two layers.** Static base = model-agnostic parity floor (BUILT). Dynamic = per-model additions chosen at `generate()` time, driven by F2's jacket. F3 builds only the dynamic half.
2. **Three model-aware axes** at `generate(objective, tools, model, budget)`: pick **TIER** (strength, Constraint B) + **family ADAPTER** (Constraint C) → attach only needed modules → assemble → **measure → trim to a 300–1,000-token base** (hard cap ~1,500; Constraint A).
3. **Tiering = CORE / EXTENDED / TINY** by profile (Goose `system.md` vs `tiny_model_system.md` precedent; the tiny variant may even swap the loop protocol). Strong→extended, weak→core, smallest→tiny.
4. **Family adapter** = model-keyed module that adjusts FORMAT only (opencode `routing.ts` → anthropic/gpt/gemini/… variants over a `default.txt` fallback ≈ our agnostic core). LiteLLM normalizes transport, NOT content — content portability is on us.
5. **Toolshim** (Goose) for non-tool-calling models: add **`tool_mode ∈ {native, shim_prompt, shim_interpreter}`** to the jacket; `<tools>` renders the JSON-in-text protocol for shim models; re-serialize tool history as plain text for non-native APIs. (F2's probe detects native tool-calling and sets `tool_mode`.)
6. **Token-economy assembly** (Constraint A): one-line imperative rules, no examples in the always-on base (examples = optional module), use `get_token_estimate()` to budget, **drop optional modules when over cap** (enforced, not hoped).
7. **Weak-model longevity = Document-&-Clear** (Constraint D): at **~60% context** dump plan/decisions/progress to disk (`PROGRESS.md`/`todo.md`), clear, resume from the doc; `<turn-context>` + goal re-injection per turn; target **≥10 task-turns** then reset; escalate to big-brother **only on stall**. Document-&-Clear is primary; lossy auto-compaction is fallback. The **loop owns this**, not the model's tokens.

## In scope (PRD §7 — each a candidate sub-spec)
- **Tiering** (CORE/EXTENDED/TINY by profile) · **family adapter** (format normalization + per-family variants, `default` fallback) · **toolshim** (`tool_mode` rendering) · **token-budget assembly** (300–1,000 cap, compression, drop-optional-over-cap) · **weak-model longevity** (`<turn-context>` + goal re-injection + Document-&-Clear at ~60% via the loop).
- The integration at `DynamicTUIPromptGenerator.generate()` / `BasePromptBuilder` so the split is additive over the intact static base.

## Out of scope (do NOT let specify wander)
- Building the profiler / the jacket math — that's **F2** (F3 only READS the jacket).
- Changing the already-built static base foundation prompt, or the loops' internals.
- Rewiring dev-kid to call PromptChain's loop (PRD decision #14 — a separate, later effort).

## User stories (priority — refine in specify)
- **US1 (P1, MVP):** `generate(objective, tools, model)` returns a profile-appropriate prompt (different TIER + family ADAPTER per model) **within the 300–1,000-token budget**, static base intact.
- **US2 (P2):** toolshim — non-tool-calling models get the JSON-in-text `<tools>` rendering per `tool_mode`.
- **US3 (P3):** weak-model longevity — the loop runs Document-&-Clear at ~60%, `<turn-context>` + goal re-injection, ≥10 turns then reset, escalate-on-stall only.

## Acceptance / success criteria (PRD §7)
- `generate(objective, tools, model)` returns a **measurably different**, **budget-compliant** prompt per model (different tier/adapter/shim), static base unchanged.
- **A/B measurably improves a weak model's task completion vs the static base alone** on a small eval.
- **OPEN ITEM to define during F3 planning:** the small **weak-vs-strong model eval set** (the A/B benchmark) is not yet specified — define it as part of this feature's spec (e.g. N scoped coding tasks at a couple of token tiers).

## Guardrails
- Test-first (Constitution III). Budget-capped + measured (Constitution V). Additive over the intact static base (Constitution VII — no speculative abstraction; build only the five components above).
