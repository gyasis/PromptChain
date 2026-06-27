# Phase 0 Research: SIO Output Integration — JSONL Transcript Emitter

All decisions below are grounded in the existing code (`promptchain/observability/mlflow_observer.py`,
`promptchain/utils/execution_events.py`, `promptchain/utils/promptchaining.py`,
`promptchain/utils/test_loop_chain.py`) and the locked F1 brief / PRD §5.

## D1 — Attachment mechanism

- **Decision**: Attach via the public `PromptChain.register_callback(callback, event_filter)` API
  (`promptchaining.py:620`), exactly like `MLflowObserver`. The emitter exposes a single
  `handle_event(event: ExecutionEvent)` callback.
- **Rationale**: Brief decision #4 (attach via public API, do not edit the event system or emit
  sites). `register_callback` already supports an optional `event_filter` set, so the emitter can
  subscribe to all event types or a subset.
- **Alternatives rejected**: Monkey-patching emit sites, or a decorator on `process_prompt_async`
  — both edit the chain (forbidden, non-additive).

## D2 — Async safety (resolves the `run_coro_blocking` import-surface tension)

- **Decision**: Register the emitter as an **async callback** (`async def handle_event`). The
  `CallbackManager` already awaits async callbacks (`register_callback` docstring: "Callbacks can
  be either synchronous or asynchronous"), so the emitter never needs to bridge loops in the hot
  path. The actual JSONL append is a fast synchronous file write performed inside the async
  callback. `run_coro_blocking` is therefore **not imported in the normal path**, keeping the
  module's imports to stdlib + `execution_events`.
- **If a synchronous flush is ever required** (e.g. a final flush from a sync context): import
  `run_coro_blocking` from `promptchain.utils.test_loop_chain` lazily at that call site only.
- **Rationale**: Brief decision #5 forbids bare `asyncio.run` in a running loop and says reuse
  `run_coro_blocking`; brief decision #3 caps imports at stdlib + `execution_events`. Registering
  an async callback satisfies both — the await is handled by the CallbackManager, and no
  loop-bridging helper is needed in steady state.
- **Alternatives rejected**: A sync callback that calls `run_coro_blocking` on every event (adds a
  cross-module import to `test_loop_chain` and unnecessary thread-offload overhead); a background
  writer thread + queue (more complexity than one-file-per-run append needs for F1).

## D3 — Event → transcript-line mapping

- **Decision**: Subscribe to the lifecycle event types and emit one JSONL line per relevant event:

  | ExecutionEventType | Transcript line `type` |
  |---|---|
  | `CHAIN_START` | `chain_start` (first line) |
  | `STEP_START` / `STEP_END` / `STEP_SKIPPED` | `step` |
  | `MODEL_CALL_START` / `MODEL_CALL_END` / `MODEL_CALL_ERROR` | `model_call` (carries `model`, token usage) |
  | `TOOL_CALL_START` | `tool_call` (name + arguments) |
  | `TOOL_CALL_END` | `tool_result` (result + status) |
  | `TOOL_CALL_ERROR` | `tool_result` (status=error) |
  | `FUNCTION_CALL_*` | `tool_result` / `tool_call` (functions are tools) |
  | `CHAIN_END` | `chain_end` (terminal, success — stop_reason) |
  | `CHAIN_ERROR` | `chain_error` (terminal, failure — stop_reason + error) |

- **Rationale**: These are the event types `MLflowObserver.handle_event` already switches on
  (`mlflow_observer.py:165–188`), so the metadata is known to be populated. The spec requires
  `chain_start` first and a terminal event on both success and failure (FR-003).
- **Source of fields**: `event.event_type`, `event.timestamp` (ISO via `.isoformat()`), and
  `event.metadata` which already carries `chain_id`, `model_name`, `usage`, `total_tokens`,
  `tool_name`, `arguments`, `result`, `error`, `call_id`, `execution_time_ms`
  (confirmed in `mlflow_observer.py` reads). The emitter is purely additive — no event changes.

## D4 — Session id & project derivation (the file path)

- **Decision**: `session_id` = `event.metadata["chain_id"]` (already used by MLflowObserver as the
  run name) captured at `CHAIN_START`; fall back to a uuid4 if absent. `<project>` = basename of
  the current working directory at run start. Final path:
  `~/.promptchain/transcripts/<project>/<session_id>.jsonl` (base dir configurable).
- **Rationale**: Brief decision #1 (global default path) + FR-006. `chain_id` gives a stable
  per-run id that downstream SIO uses as the native session id (FR-004 / SC-002).
- **Alternatives rejected**: repo-local `./.promptchain/` (brief explicitly rejects); timestamp-only
  ids (collide under concurrency; `chain_id` is already unique per run).

## D5 — Secret redaction

- **Decision**: A stdlib-only `_transcript_redaction.py` that walks a value (dict/list/str)
  recursively and redacts before serialization, using BOTH:
  (a) **key-name** matching — keys matching `(?i)(api[_-]?key|token|secret|password|authorization|bearer)`
  have their values replaced with `"***REDACTED***"`; and
  (b) **pattern** matching — values matching known secret shapes (e.g. `sk-…`, `Bearer …`, long
  high-entropy hex/base64 runs) are masked. Bias to over-redact (FR-009).
- **Rationale**: Brief decision #2; applied to tool args, tool results, and messages before write.
- **Alternatives rejected**: importing an external redaction lib (violates stdlib-only); regex on
  the final JSON string only (misses structured-key cases).

## D6 — Truncation of oversized values

- **Decision**: Cap individual field values (notably tool `result`) to a configurable max length
  (default ~8 KB), replacing the tail with a `…[truncated N chars]` marker, keeping the line's JSON
  valid (FR-014). Applied after redaction, before `json.dumps`.
- **Rationale**: Token-economy / Constitution V; brief In-Scope "truncate large results".

## D7 — Rotation (bounded output)

- **Decision**: Whole-file rotation by mtime — after writing, if the transcript directory exceeds a
  configurable cap (max file count and/or total bytes), delete the oldest files by mtime until under
  the cap (FR-007). One file per run means rotation never corrupts an in-progress transcript.
- **Rationale**: Brief decision #6; economy-first (no unbounded growth). Whole-file (not within-file)
  keeps each transcript a complete, mineable session for SIO.
- **Alternatives rejected**: `RotatingFileHandler` (designed for a single growing log, not
  one-file-per-run sessions); compaction/within-file truncation (would damage a session SIO mines).

## D8 — Opt-in / configuration

- **Decision**: Off by default. Enable via constructor flag and/or env var
  (`PROMPTCHAIN_TRANSCRIPTS_ENABLED`), mirroring `MLflowObserver`'s `PROMPTCHAIN_MLFLOW_ENABLED`
  pattern. Config surface: `enabled`, `base_dir`, `max_files`/`max_bytes`, `max_value_len`.
- **Rationale**: FR-005; zero default overhead (US3 AS1).

## D9 — No MLflow / no SIO import (verification)

- **Decision**: Module top-level imports are stdlib + `..utils.execution_events` only. A test
  asserts the module imports and emits with `mlflow` uninstalled and performs no `import sio`
  (SC-005). The SIO adapter is delivered separately in the SIO repo.
- **Rationale**: Brief decision #3 + #7; FR-008.

## Open questions

- None blocking. The exact `<project>` derivation (cwd basename vs. an explicit config value) is an
  implementation detail noted in the spec Assumptions; default = cwd basename.
