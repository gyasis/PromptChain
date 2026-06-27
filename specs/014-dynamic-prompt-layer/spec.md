# Feature Specification: Dynamic Prompt Layer (drive assembly through the formulas)

**Feature Branch**: `014-dynamic-prompt-layer`
**Created**: 2026-06-27
**Status**: Draft
**Input**: User description: "Build the dynamic half of the two-layer prompt: a generator that takes (objective, tools, model), reads the Model Profiler's jacket, and assembles the right prompt PER MODEL — tier, family adapter, toolshim, token-budget compression, and weak-model Document-&-Clear longevity — keeping the existing model-agnostic static base intact and the split purely additive."

## Overview

The Dynamic Prompt Layer is the **dynamic half** of the two-layer prompt model. The **static
base** (a model-agnostic parity floor, already built and wired into the foundation prompt) is the
guidance every agent always reasons from. F3 adds, on top of that intact base, the **per-model
choices**: how strong a prompt the model needs, what format its model family expects, whether it
can call tools natively or needs a text-based shim, how to fit all of it inside a token budget,
and — for weak models — how the loop keeps the model productive across many turns without drowning
in its own context.

Those choices are not guessed. They are **read from the capability profile / "jacket"** the Model
Profiler (F2) produces per model (tier, token budget, execution mode, compression threshold,
max-turns, role, spawn temperature, escalation flag, tool mode, and an optional compiled system
prompt). F3 is the consumer that turns that measured profile into an actual assembled prompt at
`generate(objective, tools, model)` time, keeping the split **purely additive** over the static
base so nothing model-agnostic regresses.

The value: a weak local model gets a prompt shaped to its limits (and a loop that documents-and-
clears to stay coherent), a strong model gets a leaner extended prompt, and a non-tool-calling
model still gets working tool use — all from the **same** generator, the **same** static base, and
the model's own measured profile.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Assemble a profile-appropriate, budget-compliant prompt per model (Priority: P1) — MVP

A caller supplies an objective, the tools actually loaded for the turn, and a model id. The
generator reads that model's profile/jacket, selects a prompt **tier** (CORE / EXTENDED / TINY)
appropriate to the model's measured capability, applies the **family adapter** that normalizes the
prompt's format for that model's family (with a `default` fallback for unknown families), assembles
the static base plus the chosen optional modules, **measures** the result and **trims** it to a
300–1,000-token base (hard cap ~1,500) by dropping optional modules when over budget. Two models
with materially different profiles receive **measurably different** prompts; the static base is
present verbatim in both.

**Why this priority**: This is the MVP. Without per-model assembly that is both differentiated and
budget-compliant — while leaving the static base intact — F3 delivers nothing. Tiering + family
adapter + token-budget assembly is the irreducible core; toolshim (US2) and longevity (US3) layer
on top.

**Independent Test**: Given two seeded profiles (e.g., a strong model and a weak/tiny model) and a
fixed objective + tool set, call `generate()` for each and confirm: (a) each output contains the
static base verbatim, (b) the two outputs differ in tier and/or family-adapter framing, (c) each
output's measured token count is within the configured budget / under the hard cap, and (d) when
forced over budget, optional modules are dropped (never the static base) in the documented order.

**Acceptance Scenarios**:

1. **Given** a model id with a persisted profile and a set of loaded tools, **When** `generate()`
   is called, **Then** the returned prompt contains the static base verbatim plus the optional
   modules selected for that profile's tier, and its measured token count is within the configured
   budget (target 300–1,000, hard cap ~1,500).
2. **Given** two model ids whose profiles map to different tiers (e.g., EXTENDED vs TINY), **When**
   `generate()` is called for each with the same objective and tools, **Then** the two prompts are
   measurably different (different tier content and/or family-adapter framing).
3. **Given** a model whose family is recognized, **When** `generate()` is called, **Then** the
   family adapter applies that family's format variant; **and** given an unrecognized family, the
   `default` adapter is used and assembly still succeeds.
4. **Given** an assembled prompt that exceeds the hard cap, **When** budget enforcement runs,
   **Then** optional modules are dropped in the documented priority order until the prompt fits,
   and the static base is never dropped.
5. **Given** a model with **no profile** or a **null jacket**, **When** `generate()` is called,
   **Then** assembly falls back to `recommended_tier` + `budget_tokens` (null jacket) or to a safe
   default tier (no profile) and returns a valid prompt without raising.

---

### User Story 2 - Toolshim for non-tool-calling models (Priority: P2)

A model that cannot call tools natively (its jacket records a shim `tool_mode`) still needs to use
tools. The generator reads `tool_mode ∈ {native, shim_prompt, shim_interpreter}` from the jacket
and, for the shim modes, renders the tools as a **`<tools>` JSON-in-text protocol** the model can
emit by writing text, and **re-serializes tool-call history as plain text** so a non-native API
receives a readable transcript rather than native tool-call objects. Native-mode models pass their
tools through unchanged with no shim text added.

**Why this priority**: It is what makes the harness work for the long tail of weaker / older /
non-tool-calling models, which is the whole point of "fit the harness to the model." It is P2
because US1 already yields a usable, differentiated, budget-compliant prompt for tool-calling
models; the shim extends coverage to non-tool-calling ones.

**Independent Test**: Given a jacket with `tool_mode = shim_prompt` (or `shim_interpreter`) and a
set of tools, confirm the rendered prompt includes a `<tools>` block describing those tools in the
text protocol and that a provided tool-call history is rendered as plain text; given
`tool_mode = native` (or absent), confirm no `<tools>` shim block is added and the tools are passed
through unchanged.

**Acceptance Scenarios**:

1. **Given** a jacket with a shim `tool_mode` and loaded tools, **When** `generate()` is called,
   **Then** the prompt contains a `<tools>` JSON-in-text protocol block enumerating those tools.
2. **Given** a shim `tool_mode` and a prior tool-call/result history, **When** the history is
   prepared for the model, **Then** it is re-serialized as plain text (no native tool-call objects).
3. **Given** `tool_mode = native` or an absent/null `tool_mode`, **When** `generate()` is called,
   **Then** no shim block is rendered and native tool passing is used (the missing-jacket default
   is native, never broken tool rendering).

---

### User Story 3 - Weak-model longevity via Document-&-Clear in the loop (Priority: P3)

For weak / small models, raw context grows until the model loses the thread. The loop (not the
model's token window) owns longevity: each turn it injects a `<turn-context>` block that
**re-injects the goal**; at roughly the jacket's compression threshold (~60% of effective context)
it runs **Document-&-Clear** — dumping the plan / decisions / progress to disk (a progress doc),
clearing the working context, and resuming from the doc; it targets **≥10 task-turns** before a
reset and **escalates to a bigger model only on stall** (no progress), not on every compression.
Document-&-Clear is primary; lossy auto-compaction is the fallback when the doc path is unavailable.

**Why this priority**: This is the longevity optimization for the weakest models — it depends on
US1's tiering/jacket reading and is the most involved (it lives in the loop, with disk side
effects). It is lowest priority because US1+US2 already deliver per-model prompts; US3 makes weak
models last across a long task.

**Independent Test**: Given a loop driving a weak-tier model with a fake context-usage signal,
confirm: a `<turn-context>` block with the goal is present each turn; when usage crosses the
compression threshold, a progress doc is written and the working context is reduced/cleared and
then resumed from the doc; the run sustains ≥10 simulated task-turns before a reset; escalation is
triggered only when a stall (no-progress) condition is detected, not merely on compression.

**Acceptance Scenarios**:

1. **Given** a weak-tier model in the loop, **When** each turn runs, **Then** a `<turn-context>`
   block re-injecting the goal is included for that turn.
2. **Given** context usage reaches the jacket's compression threshold, **When** the longevity step
   runs, **Then** plan/decisions/progress are persisted to a progress doc, the working context is
   cleared, and execution resumes from the doc.
3. **Given** a task in progress that is still making progress, **When** turns accumulate, **Then**
   the loop sustains ≥10 task-turns before a reset and does **not** escalate.
4. **Given** a stall (no measurable progress across the configured window) **and** the jacket
   permits escalation, **When** the stall is detected, **Then** the loop escalates to a bigger
   model; **and** when the doc path is unavailable, the loop falls back to lossy auto-compaction.

---

### Edge Cases

- **No profile for the model**: assembly MUST fall back to a safe default tier (equivalent to
  today's static-base behavior) and return a valid prompt — never raise.
- **Null jacket on an existing profile**: assembly MUST fall back to the profile's
  `recommended_tier` + `budget_tokens` (per the F2 profile-schema contract).
- **Over the hard cap after dropping every optional module**: the static base + objective + tool
  inventory MUST be preserved (the parity floor is never dropped); the over-cap condition is
  surfaced rather than silently corrupting the prompt.
- **Unknown model family**: the `default` family adapter MUST be used (assembly still succeeds).
- **`tool_mode` native or missing**: NO shim is rendered; tools pass through natively.
- **Empty objective**: `generate()` MUST reject an empty/blank objective (the existing generate
  contract requires a non-empty objective).
- **Jacket forbids escalation (`escalate = false`) under a stall**: the loop MUST respect the
  jacket (no escalation) while still resetting / continuing from the progress doc.
- **Document-&-Clear disk path unavailable**: the loop MUST fall back to lossy auto-compaction
  rather than failing the run.

## Requirements *(mandatory)*

### Functional Requirements

**Tiering, family adapter & budget assembly (US1, P1)**

- **FR-001**: The system MUST, at `generate(objective, tools, model)` time, read the target model's
  persisted profile/jacket from the profile store and select a prompt **tier** of CORE / EXTENDED /
  TINY from the profile's measured capability / recommended tier.
- **FR-002**: The tier mapping MUST be documented and deterministic (e.g., strongest → EXTENDED,
  mid → CORE, smallest → TINY), driven by the profile's capability / `recommended_tier`.
- **FR-003**: The system MUST apply a **family adapter** keyed by the model's family that adjusts
  **format only** (per-family variants over a `default` fallback), without altering the semantic
  content of the static base.
- **FR-004**: The system MUST keep the existing **static base** foundation prompt intact and emit
  it **verbatim** in every assembled prompt; the dynamic layer only ADDS optional modules — it MUST
  NOT modify or omit the static base.
- **FR-005**: The system MUST measure the assembled prompt's token count (via the existing token
  estimator) and **trim to a 300–1,000-token base** (hard cap ~1,500) by **dropping optional
  modules** — never the static base — in a documented priority order when over budget.
- **FR-006**: For two models whose profiles materially differ, the system MUST produce **measurably
  different** prompts (different tier content and/or family-adapter framing), comparable by diffing
  the rendered outputs.
- **FR-007**: When no profile exists for the model the system MUST fall back to a safe default tier;
  when the profile's jacket is null it MUST fall back to `recommended_tier` + `budget_tokens`. In
  neither case may `generate()` raise.

**Toolshim (US2, P2)**

- **FR-008**: The system MUST read `tool_mode ∈ {native, shim_prompt, shim_interpreter}` from the
  jacket and render tools accordingly: **native** → tools passed through unchanged with no shim
  text; **shim_prompt / shim_interpreter** → a `<tools>` **JSON-in-text protocol** block
  enumerating the loaded tools.
- **FR-009**: For shim modes, the system MUST **re-serialize tool-call history as plain text** so a
  non-native API receives a textual transcript rather than native tool-call objects.
- **FR-010**: When `tool_mode` is absent or null, the system MUST default to **native** (no shim),
  so a missing jacket never breaks tool rendering.

**Weak-model longevity — Document-&-Clear (US3, P3)**

- **FR-011**: The loop MUST inject a `<turn-context>` block that **re-injects the goal** on each
  turn for weak-tier models.
- **FR-012**: At the jacket's compression threshold (~60% of effective context), the loop MUST run
  **Document-&-Clear**: persist plan / decisions / progress to a progress doc on disk, clear the
  working context, and resume from the doc. The **loop** owns this — it is not delegated to the
  model's token window.
- **FR-013**: The loop MUST target **≥10 task-turns** before a reset and MUST **escalate to a
  bigger model only on a stall** (no measurable progress), subject to the jacket's `escalate`
  flag — not on every compression.
- **FR-014**: Document-&-Clear MUST be the **primary** longevity mechanism; lossy auto-compaction
  is the **fallback** when the progress-doc path is unavailable.

**Determinism & measurability (test-first, Constitution III & V)**

- **FR-015**: Given the same `(objective, tools, model, profile)`, `generate()` MUST be
  **deterministic** (identical output) so assembled prompts are reproducible and unit-testable
  offline with seeded profiles (no live model required).
- **FR-016**: The per-model **difference**, **budget-compliance**, **static-base-intact**, and
  **toolshim** behaviors MUST each be validated by unit tests against seeded profiles/jackets; the
  **A/B weak-model improvement** (SC-007) is validated by a small eval set defined during planning.

### Key Entities

- **Prompt tier**: one of CORE / EXTENDED / TINY — the strength band of the assembled prompt,
  selected from the model's measured capability. TINY may use a reduced protocol.
- **Family adapter**: a model-family-keyed module that normalizes the prompt's **format** (per-
  family variants over a `default` fallback); content-portability layer over the agnostic base.
- **Optional module**: an addable prompt section (e.g., examples, extended guidance, shim block)
  that is included per tier and **dropped first** when over budget — the static base is never an
  optional module.
- **Toolshim rendering**: the `<tools>` JSON-in-text protocol + plain-text tool-history
  serialization used when `tool_mode` is a shim mode.
- **Progress doc**: the on-disk plan / decisions / progress artifact written at the compression
  threshold and resumed from (the Document-&-Clear substrate).
- **Jacket (read-only here)**: the F2-produced per-model config the generator consumes —
  {tier, budget_tokens, mode, spawn_temp, compress_at, max_turns, role, escalate, tool_mode,
  optional system_prompt}; F3 reads it and tolerates nulls.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For any model id, `generate(objective, tools, model)` returns a prompt within the
  configured token budget (target 300–1,000, hard cap ~1,500) with the static base present verbatim.
- **SC-002**: Two models with materially different profiles receive **measurably different** prompts
  (different tier and/or family adapter), verifiable by diffing the rendered outputs.
- **SC-003**: The static base is **unchanged** by F3 — the foundation prompt's content appears
  verbatim and is byte-identical to its pre-F3 form (no regression to the model-agnostic floor).
- **SC-004**: A non-tool-calling model (shim `tool_mode`) receives a `<tools>` JSON-in-text protocol
  and plain-text tool history; a native model receives unchanged native tools.
- **SC-005**: A null/missing jacket falls back to `recommended_tier` + `budget_tokens` (null jacket)
  or a safe default (no profile) **without error**.
- **SC-006**: A weak-tier model in the loop runs Document-&-Clear at the compression threshold,
  re-injects the goal each turn via `<turn-context>`, sustains **≥10 task-turns** before a reset,
  and escalates only on a stall.
- **SC-007**: On the small eval set defined during planning, a weak model's task-completion rate is
  **measurably higher** with the F3 per-model prompt than with the static base alone (the A/B win).
- **SC-008**: `generate()` is **deterministic** for fixed inputs (reproducible prompts), validated
  by offline unit tests with seeded profiles.

## Assumptions

- **A1**: F2's **profile/jacket schema is frozen** (`specs/013-model-profiler/contracts/
  profile-schema.md`). F3 reads it defensively (unknown fields ignored, missing optional fields
  treated as null/default) and adds **no** required field; any new profile field would be additive
  and optional only.
- **A2**: The **static base** foundation prompt is already built and wired (`TUI_FOUNDATION_PROMPT`
  in `promptchain/prompts/tui_dynamic.py`); F3 does not author or change it.
- **A3**: The integration seam is `DynamicTUIPromptGenerator.generate()` / the `BasePromptBuilder`
  surface consumed at `agentic_step_processor.py:1006`; the split is additive over the intact base.
- **A4**: The loops (`ExternalLoop` / MicroPromptChain / RalphChain) and
  `ContextDistiller` / `HistorySummarizer` already exist and host the US3 longevity behavior; F3
  adds the Document-&-Clear policy to the loop rather than rebuilding the loop.
- **A5**: A **real local model is reachable offline** (LAN ollama) for the live smoke and the A/B
  eval acceptance path; no paid/cloud credentials are required. The deterministic core (SC-001..005,
  SC-008) is fully testable offline with seeded profiles and no live model.
- **A6**: `tool_mode` is available on (or derivable from) the jacket per F2; if it is not present on
  a given profile, F3 treats it as `native` (FR-010).

## Dependencies

- **Depends on F2** (Model Profiler) — F3 reads the persisted profile/jacket
  (`~/.promptchain/model_profiles.json`) for tier, budget, family/tool_mode, compress_at, max_turns,
  escalate, and the optional compiled system prompt.
- **Depends on F1 / the static base** — the dynamic layer is additive over the intact
  model-agnostic foundation; F1's transcript emitter remains the observability surface.
- **Builds on the existing loops** — US3's Document-&-Clear lives in the existing loop +
  distiller/summarizer machinery.

## Out of Scope

- Building the profiler or the jacket math — that is **F2**; F3 only **reads** the profile/jacket.
- Changing the already-built **static base** foundation prompt, or the loops' internals beyond
  adding the Document-&-Clear longevity policy.
- Rewiring **dev-kid** to call PromptChain's loop (PRD decision #14 — a separate, later effort).
- SIO-side CLI surfaces — they ship in the SIO repository, not here (same rule as F1/F2).

---

## Gap-check vs PRD §7 (source of truth)

| PRD §7 element | Covered by |
|---|---|
| Tiering (CORE/EXTENDED/TINY by profile, Constraint B) | US1 · FR-001, FR-002 |
| Family adapter (format normalization + per-family variants, `default` fallback, Constraint C) | US1 · FR-003 |
| Toolshim (JSON tool-call fallback for non-tool-calling models, Goose) | US2 · FR-008, FR-009, FR-010 |
| Token-economy assembly (budget-cap 300–1,000, compression, Constraint A) | US1 · FR-005, FR-006 · SC-001 |
| Weak-model longevity (`<turn-context>` + goal re-injection, Document-&-Clear at ~60%, Constraint D) | US3 · FR-011..FR-014 · SC-006 |
| Integration seam at `DynamicTUIPromptGenerator.generate()` / `BasePromptBuilder` (`agentic_step_processor.py:1006`), static base intact, additive | A2, A3 · FR-004 · SC-003 |
| Acceptance: profile-appropriate prompt (different tier/adapter/shim per model) within budget | SC-001, SC-002, SC-004 |
| Acceptance: A/B measurably improves weak-model task completion vs static base alone | SC-007 (+ eval set defined in planning) |

**No gaps identified.** The one PRD-flagged open item — the small weak-vs-strong A/B eval set — is
intentionally deferred to F3 planning per the brief (tracked as SC-007 + FR-016).
