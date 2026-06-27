# Phase 1 Data Model — Model Profiler

All structures are in-process Python dataclasses (the only on-disk artifact is the JSON profile
store, contract in `contracts/profile-schema.md`). Field names below are the canonical ones.

## ProbeItem

One calibrated probe task targeting one skill dimension.

| Field | Type | Notes |
|---|---|---|
| `item_id` | str | stable id within the bank |
| `dimension` | str | one of the locked skill dimensions (see Skill dimensions) |
| `a` | float | discrimination (>0); higher = sharper item |
| `b` | float | difficulty (on the θ scale) |
| `c` | float | guessing (lower asymptote, ∈ [0,1)) |
| `prompt` | str | the probe prompt sent to the model (isolated session) |
| `scorer` | str / callable ref | which auto-scorer grades the response → correct/incorrect (+ raw values) |
| `meta` | dict | optional (expected answer, format spec, turn budget for degradation items, …) |

**Validation**: `a > 0`, `0 ≤ c < 1`, `dimension` in the known set. Calibration-quality filter
(applied when building/accepting a bank): response variance ≥ 1%, accuracy ≤ 95%, point-biserial
≥ 0.1 (uncalibrated/empty bank → refuse to score, FR edge case).

## ItemBank

| Field | Type | Notes |
|---|---|---|
| `items` | list[ProbeItem] | calibrated items |
| `calibrated` | bool | False → capability scoring refuses (raises) |

Helpers: `by_dimension(dim)`, `filter_quality(...)`, `synthetic(...)` (build a bank with KNOWN
(a,b,c) for the math tests / MVP).

## ProbeResponse

One administered item against the target model.

| Field | Type | Notes |
|---|---|---|
| `item_id` | str | |
| `correct` | bool | scored outcome (the IRT response) |
| `raw` | dict | dimension-specific raw values: degradation turn, usable context tokens, latency ms, format-broke flag, tool-call validity, … |
| `transcript_path` | str | the F1 transcript file written for this trial |
| `error` | str \| None | set if the trial itself failed (recorded, excluded from θ̂ or retried) |

## SkillEstimate

Per-dimension ability result.

| Field | Type | Notes |
|---|---|---|
| `dimension` | str | |
| `theta` | float | θ̂ (EAP, or WLE at extremes) |
| `se` | float | standard error `1/√(ΣI_i)` at θ̂ |
| `n_items` | int | items administered (bounded by SE stop rule / hard max) |
| `capability` | float | `C = σ(θ̂) ∈ [0,1]` |
| `low_precision` | bool | True if SE never reached τ within the item cap |

## Jacket

The per-model derived configuration (the F3-facing artifact).

| Field | Type | Derivation |
|---|---|---|
| `tier` | str | `lean`/`standard`/`rich + few-shot`/`max-rich` from Ω band |
| `budget_tokens` | int | from Ω band (1–2k / 4k / 8k / 16k), clamped by effective ctx |
| `mode` | str | `single-shot` / `single-shot+retry` / `heavy-loop` / `escalate` from Ω band |
| `spawn_temp` | float | `1 − C` (weaker models delegate more) |
| `compress_at` | float | ctx fraction from Ω band (0.85 / 0.75 / 0.60 / 0.50) |
| `max_turns` | int | `degradation_turn − 1` (reset just before the cliff) |
| `role` | str | `planner` / `executor` / `both` from capability thresholds |
| `escalate` | bool | True if `P_route ≥ α OR Ω < 0.25 OR SE > 0.4` |
| `system_prompt` | str \| None | optional GEPA-compiled per-model prompt (US3; None otherwise) |

## CapabilityProfile

The persisted per-model record (the unit in `model_profiles.json`).

| Field | Type | Notes |
|---|---|---|
| `model_id` | str | the key |
| `skills` | dict[str, SkillEstimate] | per-dimension θ̂/SE/C |
| `capability` | float | aggregate C (weighted over the capability dimensions) |
| `omega` | float \| None | composite Ω (US2; None until computed) |
| `calibration_k` | float \| None | K = 1 − ECE (US2) |
| `cost_penalty_f` | float \| None | F (US2) |
| `jacket` | Jacket \| None | derived config (US2) |
| `recommended_tier` | str | available from US1 (capability → tier) even before full Ω |
| `budget_tokens` | int | available from US1 (capability + ctx → budget) |
| `effective_context` | int \| None | measured usable context ceiling |
| `degradation_turn` | int \| None | measured degradation point |
| `n_observations` | int | telemetry sessions folded in via EWMA (US3) |
| `created_ts` / `updated_ts` | str | ISO 8601 |
| `schema_version` | int | profile-store schema (starts at 1) |

## Relationships

```
ItemBank ──contains──> ProbeItem
ModelProfiler.run_probe(model_id, bank) ──administers──> ProbeResponse (per item, isolated F1 transcript)
   responses ──IRT/CAT──> SkillEstimate (per dimension)
   SkillEstimate + cost inputs ──composite──> Ω, K, F ──> Jacket
   {SkillEstimate, Ω, Jacket, …} ──> CapabilityProfile ──persist──> model_profiles.json
   new telemetry ──EWMA──> CapabilityProfile (refine; no-op if none)
   model × {jacket} grid ──Δθ──> best-fit Jacket (US3, reuses sio experiment/GEPA)
```

## Skill dimensions (locked)

`instruction_following`, `tool_call_reliability`, `reasoning_depth`, `degradation_turn`,
`effective_context`, `format_sensitivity`, `latency_throughput`. The capability score `C`
aggregates the *ability* dimensions (instruction/tool/reasoning/format); `degradation_turn`,
`effective_context`, `latency_throughput` contribute raw values to the jacket (max_turns,
budget, F) rather than θ̂.
