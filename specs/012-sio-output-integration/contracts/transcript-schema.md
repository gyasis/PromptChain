# Contract: PromptChain JSONL Transcript Line Schema (LOCKED — F2 + SIO depend on this)

This is the **lockable artifact** of F1. The Model Profiler (F2) derives `model_used` and token
totals from it; the SIO harness adapter (`sio/harnesses/promptchain.py`) maps it to SIO's
`session_metrics` / `flow_events` / `error_records`. Changing a field after lock is a breaking
change for downstream work — additive optional fields only.

## File-level invariants

1. **One file per run**: `~/.promptchain/transcripts/<project>/<session_id>.jsonl` (base dir configurable).
2. **One event per line**; every line is **valid standalone JSON** (`json.loads(line)` succeeds).
3. **First line** is `type: "chain_start"`. **Last line** is a terminal event
   (`chain_end` or `chain_error`) — present on both success and failure (no partial-only file).
4. Lines are append-only and ordered by emission time.

## Common envelope (every line)

```json
{
  "type": "chain_start | step | model_call | tool_call | tool_result | chain_end | chain_error",
  "ts": "2026-06-27T08:53:00.123456+00:00",   // ISO 8601, event.timestamp.isoformat()
  "session_id": "<chain_id>"                    // stable per run; SIO native session id
}
```

`type`, `ts`, `session_id` are REQUIRED on every line (FR-002).

## Per-type line shapes

### `chain_start` (first line, required)
```json
{ "type": "chain_start", "ts": "...", "session_id": "abc123",
  "project": "PromptChain", "schema_version": 1 }
```

### `step`
```json
{ "type": "step", "ts": "...", "session_id": "abc123",
  "phase": "start | end | skipped", "instruction_index": 0,
  "execution_time_ms": 12.4 }
```

### `model_call` (carries the model — F2 linchpin)
```json
{ "type": "model_call", "ts": "...", "session_id": "abc123",
  "phase": "start | end | error",
  "model": "ollama/qwen3-coder:30b",          // = model_used for F2; non-empty on completed runs
  "call_id": "model-…",
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 },
  "execution_time_ms": 0,
  "error": null }
```

### `tool_call`
```json
{ "type": "tool_call", "ts": "...", "session_id": "abc123",
  "call_id": "tool-…", "tool_name": "build_until_tests_pass",
  "arguments": { ... } }                        // redacted + truncated
```

### `tool_result`
```json
{ "type": "tool_result", "ts": "...", "session_id": "abc123",
  "call_id": "tool-…", "tool_name": "build_until_tests_pass",
  "status": "ok | error",
  "result": "…",                                // redacted + truncated; "…[truncated N chars]" if capped
  "error": null }
```

### `chain_end` (terminal, success)
```json
{ "type": "chain_end", "ts": "...", "session_id": "abc123",
  "stop_reason": "completed",
  "total_tokens": 0, "execution_time_ms": 0,
  "outcome": "success",
  "correction_count": 0, "positive_signal_count": 0, "sidechain_count": 0 }  // OPTIONAL rich signals; emit 0 in F1
}
```

### `chain_error` (terminal, failure)
```json
{ "type": "chain_error", "ts": "...", "session_id": "abc123",
  "stop_reason": "error | limit",
  "error": "…",                                 // redacted + truncated
  "outcome": "error" }
```

## Derivable-by-SIO guarantees (FR-002, FR-004)

From a transcript alone, a consumer MUST be able to derive:
- **`model_used`** — from the `model` field on `model_call` lines (non-empty for a completed run).
- **token totals** — from `usage` on `model_call` and/or `total_tokens` on `chain_end`.
- **tool sequence** — the ordered `tool_call` → `tool_result` pairs (for `sio flows`).
- **error / stop_reason** — from `status: "error"` lines and the terminal `stop_reason`/`outcome`.

## Forward-compatibility

- `schema_version` starts at `1`. New REQUIRED fields require a version bump; new OPTIONAL fields
  may be added at the same version.
- Rich-signal fields (`correction_count`, `positive_signal_count`, `sidechain_count`) are OPTIONAL
  and emitted as `0` in F1 (FR-013); F2+ may populate them without a version bump.

## Security

- All `arguments`, `result`, `error`, and any message content are **redacted** (key-name + pattern
  based, over-redact) BEFORE serialization (FR-009), and oversized values **truncated** (FR-014),
  while keeping each line valid JSON.
