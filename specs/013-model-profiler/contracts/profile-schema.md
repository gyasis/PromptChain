# Contract: `model_profiles.json` — the persisted profile store (consumed by F3)

The on-disk artifact F2 produces and F3 reads. One file, a JSON object keyed by model id. Like
F1's transcript schema, this is a **lockable contract**: changing a field after F3 builds against
it is a breaking change — **additive optional fields only** without a `schema_version` bump.

## File

- Path: `~/.promptchain/model_profiles.json` (base dir configurable; mirrors F1's
  `~/.promptchain/transcripts/`).
- Shape: `{ "schema_version": 1, "profiles": { "<model_id>": <CapabilityProfile>, ... } }`.
- Writes are idempotent per model id (re-probe overwrites that model's record only) and atomic
  (write-temp + rename) so a crash never leaves a half-written store.

## `CapabilityProfile` record

```json
{
  "model_id": "ollama/qwen3-coder:30b",
  "capability": 0.78,
  "recommended_tier": "standard",
  "budget_tokens": 4000,
  "effective_context": 28000,
  "degradation_turn": 9,
  "omega": 0.46,
  "calibration_k": 0.82,
  "cost_penalty_f": 0.21,
  "skills": {
    "instruction_following": { "dimension": "instruction_following", "theta": 1.1, "se": 0.28,
                               "n_items": 12, "capability": 0.75, "low_precision": false },
    "tool_call_reliability": { "dimension": "tool_call_reliability", "theta": 0.6, "se": 0.29,
                               "n_items": 14, "capability": 0.65, "low_precision": false },
    "reasoning_depth":       { "dimension": "reasoning_depth", "theta": 0.9, "se": 0.30,
                               "n_items": 11, "capability": 0.71, "low_precision": false }
  },
  "jacket": {
    "tier": "standard", "budget_tokens": 4000, "mode": "single-shot+retry",
    "spawn_temp": 0.22, "compress_at": 0.75, "max_turns": 8, "role": "both",
    "escalate": false, "system_prompt": null
  },
  "n_observations": 0,
  "created_ts": "2026-06-27T13:10:00.000000+00:00",
  "updated_ts": "2026-06-27T13:10:00.000000+00:00",
  "schema_version": 1
}
```

## Field guarantees

- **Always present after US1**: `model_id`, `capability`, `recommended_tier`, `budget_tokens`,
  `skills` (≥1 ability dimension with `theta`, `se`, `capability`), `created_ts`, `updated_ts`,
  `schema_version`.
- **Present after US2**: `omega`, `calibration_k`, `cost_penalty_f`, `jacket`.
- **Present after US3 / over time**: `n_observations` increments via EWMA refine; `jacket.system_prompt`
  may carry a GEPA-compiled prompt; otherwise `null`.
- **Optional / nullable**: `effective_context`, `degradation_turn`, `omega`, `calibration_k`,
  `cost_penalty_f`, `jacket` (null until US2 computed). F3 MUST tolerate nulls and fall back to
  `recommended_tier` + `budget_tokens` when `jacket` is null.

## Forward-compatibility

- `schema_version` starts at `1`. New REQUIRED fields → version bump; new OPTIONAL fields may be
  added at the same version. F3 reads defensively (unknown fields ignored; missing optional fields
  treated as null/default).
