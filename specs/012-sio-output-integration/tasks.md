# Tasks: SIO Output Integration — JSONL Transcript Emitter (F1)

**Input**: Design documents from `specs/012-sio-output-integration/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/transcript-schema.md, quickstart.md
**Tests**: INCLUDED — Constitution III (Test-First) is NON-NEGOTIABLE. Tests are written FIRST and MUST FAIL before implementation.

## Format

`- [ ] [TaskID] [P?] [W#] [Story?] [agent] [file-ownership] Description`

- **[P]**: parallelizable — different file, no incomplete-task dependency.
- **[W#]**: execution wave (Development Orchestration Protocol, CLAUDE.md). Same-file tasks share an owner and run sequentially within a wave.
- **[US#]**: user story from spec.md.
- File ownership in `[...]` = the EXCLUSIVE file that task edits.

## File ownership map (one owner per file per wave)

| File | Role |
|---|---|
| `promptchain/observability/transcript_emitter.py` | the observer (core, event→line, path, rotation, config) — NEW |
| `promptchain/observability/_transcript_redaction.py` | redaction + truncation helpers (stdlib only) — NEW |
| `promptchain/observability/__init__.py` | additive export of `TranscriptEmitter` |
| `tests/test_transcript_schema_contract.py` | the LOCKED schema contract test |
| `tests/test_transcript_emitter_unit.py` | event→line mapping, redaction, truncation, rotation, config |
| `tests/test_transcript_emitter_integration.py` | register_callback() on a real chain run + error path |
| `tests/test_transcript_emitter_performance.py` | <2% overhead enabled-vs-disabled |

**Out of scope (do NOT create):** the SIO harness adapter `sio/harnesses/promptchain.py` — it lives in the SIO repo (`~/Documents/code/SIO`). F1 delivers only the schema contract it consumes. Also forbidden: any edit to the loops, `execution_events.py`, `promptchaining.py` emit sites, `mlflow_observer.py`, dev-kid, or the micro-agent fork.

---

## Phase 1: Setup (Shared Infrastructure) — Wave 1

**Purpose**: Importable skeletons so test-first tests fail on assertions, not collection errors.

- [x] T001 [W1] [python-pro] [promptchain/observability/transcript_emitter.py] Create skeleton `TranscriptEmitter` + `TranscriptEmitterConfig` (constructor signature per data-model.md, `async def handle_event(self, event)` no-op, `enabled=False` default) — imports limited to stdlib (`json, os, pathlib, datetime, re, threading`) + `..utils.execution_events`. No mlflow, no sio.
- [x] T002 [P] [W1] [python-pro] [promptchain/observability/_transcript_redaction.py] Create skeleton `redact(value)` and `truncate(value, max_len)` (stdlib only) returning input unchanged for now.
- [x] T003 [P] [W1] [python-pro] [promptchain/observability/__init__.py] Additively export `TranscriptEmitter` (and `TranscriptEmitterConfig`) — do not remove existing exports.

**Checkpoint**: `from promptchain.observability import TranscriptEmitter` resolves.

---

## Phase 2: Foundational (Blocking) — Wave 2

**Purpose**: The locked line-schema constants every test + implementation depends on.

- [x] T004 [W2] [python-pro] [promptchain/observability/transcript_emitter.py] Encode the schema contract as constants/builders: the `ExecutionEventType → line.type` map (research D3), `SCHEMA_VERSION = 1`, the common envelope builder `{type, ts (isoformat), session_id}`, and per-type field assembly per `contracts/transcript-schema.md`. (Same file as T001 → sequential.)

**⚠️ CRITICAL**: No user-story implementation begins until T004 is complete.

---

## Phase 3: User Story 1 — Structured transcript (Priority: P1) 🎯 MVP — Waves 3–4

**Goal**: Any PromptChain run writes an append-only JSONL transcript (chain_start first, instruction/tool events, terminal event last; one event/line; non-empty model).

**Independent Test**: Run a small chain with one model instruction + one tool call; assert the JSONL file exists, every line is valid JSON, contains `chain_start`/`tool_call`/`tool_result`/terminal and a non-empty `model`.

### Tests for US1 (write FIRST — MUST FAIL) — Wave 3

- [x] T005 [P] [W3] [US1] [tests/test_transcript_schema_contract.py] Contract test: every line `json.loads`-valid; first line `chain_start`; last line terminal (`chain_end`/`chain_error`); required envelope `{type, ts, session_id}` on every line. MUST FAIL.
- [x] T006 [P] [W3] [US1] [tests/test_transcript_emitter_unit.py] Unit test: feed synthetic `ExecutionEvent`s (chain_start, model_call, tool_call, tool_result, chain_end, chain_error) → assert correct line `type` + fields (model, usage, tool_name, arguments, result/status, stop_reason) per contract. MUST FAIL.
- [x] T007 [P] [W3] [US1] [tests/test_transcript_emitter_integration.py] Integration test: `chain.register_callback(emitter.handle_event)`, run a real chain to completion → transcript at `~/.promptchain/transcripts/<project>/<session_id>.jsonl` with expected ordered lines + non-empty `model`; and an erroring run still closes with a terminal `chain_error` (no partial-only file). MUST FAIL.

### Implementation for US1 — Wave 4

- [x] T008 [W4] [US1] [python-pro] [promptchain/observability/transcript_emitter.py] Implement `handle_event`: map event→line (T004), resolve path (`session_id`=`chain_id` w/ uuid4 fallback, `<project>`=cwd basename), ensure dir, append one JSON line per event (one file per run). (Sequential on transcript_emitter.py.)
- [x] T009 [W4] [US1] [python-pro] [promptchain/observability/transcript_emitter.py] Guarantee a terminal event on both `CHAIN_END` (stop_reason=completed/outcome=success) and `CHAIN_ERROR` (stop_reason=error|limit/outcome=error) — FR-003. (Sequential.)
- [x] T010 [W4] [US1] [python-pro] [promptchain/observability/transcript_emitter.py] Make `handle_event` an async callback that the CallbackManager awaits (research D2 — no `run_coro_blocking` in the hot path; no bare `asyncio.run`). (Sequential.)

**Checkpoint**: US1 green — MVP transcript is produced and contract test passes.

---

## Phase 4: User Story 2 — SIO-derivable transcript (Priority: P2) — Wave 5

**Goal**: A transcript dir is mineable by SIO as first-class sessions. F1 scope = the schema makes `model_used`, token totals, tool sequence, and error/stop_reason derivable from the transcript alone (the adapter itself ships in the SIO repo).

**Independent Test**: From N transcripts, a generic reader derives exactly N sessions and recovers each one's ordered tool sequence and non-empty model.

### Tests for US2 (write FIRST — MUST FAIL) — Wave 5

- [ ] T011 [P] [W5] [US2] [tests/test_transcript_schema_contract.py] Contract test (SIO-derivable guarantees): from a transcript, derive `model_used` (from `model_call.model`), token totals (`usage`/`chain_end.total_tokens`), the ordered `tool_call`→`tool_result` sequence (paired by `call_id`), and error/stop_reason. MUST FAIL. (Sequential after T005 — same file.)

### Implementation for US2 — Wave 5

- [ ] T012 [W5] [US2] [python-pro] [promptchain/observability/transcript_emitter.py] Ensure `model_call` lines carry non-empty `model` + `usage`, and `tool_call`/`tool_result` carry matching `call_id` so the tool sequence is recoverable; `chain_end` carries `total_tokens` + `stop_reason` (FR-002/FR-004 derivability). (Sequential on transcript_emitter.py.)

**Checkpoint**: US1 + US2 green — transcripts are mineable; SIO adapter (separate repo) can consume them.

---

## Phase 5: User Story 3 — Bounded, opt-in, low-overhead, no MLflow (Priority: P3) — Waves 6–7

**Goal**: Off by default; redact secrets + truncate oversized values; whole-file rotation by mtime; <2% overhead; works with mlflow uninstalled and never imports sio.

**Independent Test**: Small rotation cap across ≥100 runs stays bounded; secret-shaped args redacted; enabled-vs-disabled <2%; import + emit with `mlflow` uninstalled.

### Tests for US3 (write FIRST — MUST FAIL) — Wave 6

- [ ] T013 [P] [W6] [US3] [tests/test_transcript_emitter_unit.py] Unit test: `redact()` masks key-name matches (`api_key`/`token`/`secret`/`authorization`/`password`/`bearer`) AND pattern matches (`sk-…`, `Bearer …`, long high-entropy runs), recursively, biased to over-redact. MUST FAIL. (Sequential after T006 — same file.)
- [ ] T014 [W6] [US3] [python-pro] [tests/test_transcript_emitter_unit.py] Unit test: `truncate()` caps oversized values with `…[truncated N chars]`, line stays valid JSON (FR-014). MUST FAIL. (Sequential, same file.)
- [ ] T015 [W6] [US3] [python-pro] [tests/test_transcript_emitter_unit.py] Unit test: whole-file rotation by mtime keeps dir within `max_files`/`max_bytes` across ≥100 runs (FR-007). MUST FAIL. (Sequential, same file.)
- [ ] T016 [P] [W6] [US3] [tests/test_transcript_emitter_integration.py] Test: disabled-by-default writes NOTHING and runs no emit path (US3 AS1); emitter imports + emits with `mlflow` uninstalled; module performs no `import sio` (SC-005). MUST FAIL. (Sequential after T007 — same file.)
- [ ] T017 [P] [W6] [US3] [tests/test_transcript_emitter_performance.py] Performance test: end-to-end wall-clock with emission enabled is <2% over disabled (SC-003). MUST FAIL.

### Implementation for US3 — Wave 7

- [ ] T018 [P] [W7] [US3] [python-pro] [promptchain/observability/_transcript_redaction.py] Implement `redact()` (key-name + pattern, recursive, over-redact).
- [ ] T019 [W7] [US3] [python-pro] [promptchain/observability/_transcript_redaction.py] Implement `truncate(value, max_len)` keeping JSON validity. (Sequential after T018 — same file.)
- [ ] T020 [W7] [US3] [python-pro] [promptchain/observability/transcript_emitter.py] Wire redact+truncate into line assembly (applied to arguments/result/error/messages before `json.dumps`); add opt-in config (`enabled` flag + `PROMPTCHAIN_TRANSCRIPTS_ENABLED`); guarantee zero work when disabled. (Sequential on transcript_emitter.py.)
- [ ] T021 [W7] [US3] [python-pro] [promptchain/observability/transcript_emitter.py] Implement whole-file rotation by mtime after the terminal write (`max_files`/`max_bytes`). (Sequential.)

**Checkpoint**: All user stories green.

---

## Phase 6: Polish & Cross-Cutting — Wave 8

- [ ] T022 [P] [W8] [python-pro] [promptchain/observability/transcript_emitter.py] Emit optional rich-signal fields (`correction_count`/`positive_signal_count`/`sidechain_count`) as `0` + `schema_version` on the relevant lines (FR-013). (Sequential on transcript_emitter.py.)
- [ ] T023 [P] [W8] [python-pro] [tests/test_transcript_schema_contract.py] Add a guard test asserting `transcript_emitter`'s module-level imports contain no `mlflow`/`sio` (emit-not-reuse, FR-008). (Sequential after T011 — same file.)
- [ ] T024 [W8] Run the full suite + the offline live smoke per quickstart.md (`OLLAMA_API_BASE=http://192.168.0.159:11434 PROMPTCHAIN_LOOP_MODEL=ollama/qwen3-coder:30b`, `use_docker=False`); confirm SC-001..SC-006.

---

## Dependencies & execution order

- **Wave 1 (T001–T003)** → skeletons. T002/T003 [P]; T001 owns transcript_emitter.py.
- **Wave 2 (T004)** → schema constants. Blocks all stories.
- **Wave 3 (T005–T007)** → US1 failing tests, all [P] (distinct files).
- **Wave 4 (T008–T010)** → US1 impl, sequential (all transcript_emitter.py).
- **Wave 5 (T011–T012)** → US2 test + impl.
- **Wave 6 (T013–T017)** → US3 failing tests (unit-file ones sequential; T016/T017 parallel to them).
- **Wave 7 (T018–T021)** → US3 impl. T018/T019 own _transcript_redaction.py; T020/T021 own transcript_emitter.py (the two source files are parallel to each other).
- **Wave 8 (T022–T024)** → polish + full verification.

**Story independence**: US1 is the MVP and stands alone. US2 adds derivability guarantees on top of US1's schema. US3 is production hygiene layered on US1. Each story's tests can run independently.

## Implementation strategy (MVP first)

1. Ship **US1** (Waves 1–4) → a real, mineable transcript exists. That is the lockable artifact F2 depends on — lock it here.
2. Layer **US2** derivability guarantees, then **US3** hygiene.
3. dev-kid (`~/.local/bin/dev-kid`) is the BUILDER of this tasks.md — do NOT modify dev-kid or the micro-agent fork (PRD decision #14). The test exit code is the success bar.
