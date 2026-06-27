# Phase 1 Data Model: SIO Output Integration

The feature ships no database; the "data model" is the in-memory config object, the transcript
file, and the event-record line. The authoritative line schema is
[`contracts/transcript-schema.md`](./contracts/transcript-schema.md).

## Entities

### TranscriptEmitterConfig
Configuration for the observer (all fields have defaults; emission is OFF by default).

| Field | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | `False` | Opt-in (FR-005). Also settable via `PROMPTCHAIN_TRANSCRIPTS_ENABLED`. |
| `base_dir` | path | `~/.promptchain/transcripts` | Global default, configurable (FR-006). |
| `project` | str | cwd basename | Path segment `<project>`. |
| `max_files` | int | e.g. 500 | Whole-file rotation cap by count (FR-007). |
| `max_bytes` | int | e.g. 200 MB | Optional total-size cap by mtime (FR-007). |
| `max_value_len` | int | e.g. 8192 | Truncation threshold for field values (FR-014). |

### Transcript (file)
- One per run: `<base_dir>/<project>/<session_id>.jsonl`.
- Append-only, ordered; opens with `chain_start`, closes with a terminal event.
- `session_id` = `chain_id` from `CHAIN_START` metadata (uuid4 fallback).

### EventRecord (one JSONL line)
- Common envelope `{type, ts, session_id}` + per-type fields (see contract).
- Produced by mapping `ExecutionEvent` (`event_type`, `timestamp`, `metadata`) → a line dict, then
  redact → truncate → `json.dumps`.
- No new event types are introduced; `ExecutionEventType` is unchanged.

## Lifecycle / state transitions (per run)

```
CHAIN_START → write chain_start (resolve path, ensure dir)
   ↓
[ STEP_* / MODEL_CALL_* / TOOL_CALL_* / FUNCTION_CALL_* ]  → append step / model_call / tool_call / tool_result
   ↓
CHAIN_END  → append chain_end (stop_reason=completed, outcome=success)   ─┐
CHAIN_ERROR→ append chain_error (stop_reason=error|limit, outcome=error) ─┴ terminal, then rotation sweep
```

- A crash mid-run leaves all prior lines valid; the terminal event records the failure when the
  chain reports `CHAIN_ERROR` (FR-003).
- Rotation runs after the terminal write: delete oldest files by mtime until under `max_files`
  / `max_bytes`.

## Validation rules

- Every emitted line MUST `json.loads` cleanly (contract invariant 2; enforced by the contract test).
- `model` MUST be non-empty on a completed run's `model_call` line(s) (SC-001).
- Secret-shaped values MUST NOT appear unredacted in any line (SC-006).
- With emission disabled, NO file is created and NO emit path runs (US3 AS1).
