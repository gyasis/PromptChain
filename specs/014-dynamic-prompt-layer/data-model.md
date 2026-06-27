# Phase 1 Data Model — Dynamic Prompt Layer (F3)

F3 is assembly logic; its "data" is mostly small value objects + the (read-only) F2 profile/jacket
it consumes. No new persistence except US3's on-disk progress doc.

---

## PromptTier (enum-like)

The F3-owned prompt-strength axis.

| Value | Meaning | Selected when (default thresholds, D2) |
|---|---|---|
| `EXTENDED` | Strong model: agnostic base + the optional modules (examples, extended guidance) | `capability ≥ 0.66` |
| `CORE` | Mid model: agnostic base + essentials only | `0.33 ≤ capability < 0.66` |
| `TINY` | Weakest/smallest model: reduced protocol (may swap to a simpler loop protocol) | `capability < 0.33` |

- Selection input: `CapabilityProfile.capability` (primary); `recommended_tier` as a hint/override.
- Deterministic, pure function: `select_tier(profile) -> PromptTier` (FR-001/002/015).
- No profile available → default **CORE** (safe parity floor, FR-007).

## OptionalModule

An addable prompt section that is included per tier and is the unit dropped under budget pressure.

| Field | Type | Notes |
|---|---|---|
| `key` | str | stable id (e.g. `examples`, `extended_guidance`, `toolshim`) |
| `text` | str | the rendered section |
| `drop_priority` | int | lower = dropped first when over budget (D5 order: examples → extended_guidance → toolshim) |
| `tiers` | set[PromptTier] | tiers that include this module by default |

- The **static base**, the **objective**, and the **live tool inventory** are NOT OptionalModules —
  they are the parity floor and are never dropped (SC-003).

## FamilyAdapter

Format-only normalization keyed by model family (D3).

| Field | Type | Notes |
|---|---|---|
| `family` | str | `anthropic` / `openai` / `google` / `qwen` / `llama` / `default` |
| `apply(prompt_parts) -> prompt_parts` | fn | adjusts FORMAT only (delimiters/role-framing); never changes base semantics |

- `family_of(model_id) -> str` derives the family by parsing `model_id` (strip `provider/`, match
  stems); unknown → `default`.

## ToolMode (enum-like) + ToolShim

| Value | Behavior |
|---|---|
| `native` | tools passed through unchanged; no shim text (DEFAULT when jacket has no `tool_mode`) |
| `shim_prompt` | render a `<tools>` JSON-in-text protocol block; tool history re-serialized as plain text |
| `shim_interpreter` | same `<tools>` protocol, interpreter-style invocation framing |

ToolShim renderers (pure, D4):
- `resolve_tool_mode(jacket) -> ToolMode` (jacket.tool_mode or `native`).
- `render_tools_block(tools) -> str` — the `<tools>` JSON-in-text enumeration (shim modes only).
- `serialize_history_plaintext(history) -> str` — native tool-call objects → readable plain text.

## ProgressDoc (US3, on-disk)

The Document-&-Clear artifact written at the compression threshold.

| Field | Type | Notes |
|---|---|---|
| `path` | str | `PROGRESS.md` / `todo.md` under the caller-provided working dir (temp dir in tests) |
| `goal` | str | the task objective (re-injected each resume) |
| `plan` | list[str] | the current plan/steps |
| `decisions` | list[str] | decisions made so far |
| `progress` | list[str] | completed items / current state |

- Written, then the working context is cleared and resumed from this doc (FR-012/014).

## LongevityState (US3, in-memory)

Tracks the loop's longevity bookkeeping (pure decisions, D6).

| Field | Type | Notes |
|---|---|---|
| `turns` | int | task-turns since last reset; target ≥ 10 before a reset (FR-013) |
| `last_progress_signal` | any | for stall detection (no measurable progress across a window) |
| `stalled` | bool | computed; drives escalate-on-stall (subject to `jacket.escalate`) |
| `compress_at` | float | from the jacket (~0.60); threshold for Document-&-Clear |

## A/B Eval entities (SC-007, D7)

| Entity | Fields | Notes |
|---|---|---|
| `EvalTask` | `id`, `objective`, `tools`, `check(output) -> bool` | a scoped, programmatically-checkable coding task (N=5) |
| `EvalArm` | `name` (`f3` / `static_base`), `prompt_fn` | which prompt the arm uses |
| `EvalResult` | `arm`, `task_id`, `passed`, `tokens` | one task×arm outcome |
| `EvalReport` | `per_arm_completion_rate`, `delta` | aggregate; the A/B win = `f3 − static_base` |

- Deterministic scoring/aggregation unit-tested with a **fake model**; the real win is the offline
  live smoke (LAN ollama).

---

## Consumed (read-only) F2 entities

From `specs/013-model-profiler/contracts/profile-schema.md` — F3 reads these, never writes them
(except the one additive `tool_mode` field on `Jacket`):

- **`CapabilityProfile`** — `model_id`, `capability`, `recommended_tier`, `budget_tokens`,
  `effective_context`, `degradation_turn`, `skills{}`, `jacket` (may be null), `schema_version`.
- **`Jacket`** — `tier`, `budget_tokens`, `mode`, `spawn_temp`, `compress_at`, `max_turns`, `role`,
  `escalate`, `system_prompt`, **+ NEW optional `tool_mode`**. F3 tolerates a null jacket (→ fall
  back to `recommended_tier` + `budget_tokens`) and an absent profile (→ default CORE tier).
