# Feature Specification: SIO Output Integration — JSONL Transcript Emitter

**Feature Branch**: `012-sio-output-integration`
**Created**: 2026-06-27
**Status**: Draft
**Input**: PRD `prd/adaptive_prompting_system_prd.md` §5 (F1, locked first) · brief `prd/feature-briefs/F1-sio-output-integration.md` · design `research/foundation/architecture/03-sio-integration.md`

> **Context:** F1 of the Adaptive Prompting System. PromptChain must emit each run as a
> structured JSONL transcript that SIO ("CO") can mine as a 7th harness — WITHOUT reusing SIO's
> code and WITHOUT depending on MLflow. This feature is locked first because the Model Profiler
> (F2) consumes this telemetry; the transcript line schema must stabilize before downstream work.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A PromptChain run produces a structured transcript (Priority: P1)

When any `PromptChain` runs, it writes an append-only JSONL transcript capturing the run as an
ordered sequence of events: chain start/end, each instruction, each tool call and its result,
model(s) used, token counts, timings, and the final outcome — one event per line.

**Why this priority**: This IS the lockable artifact everything downstream depends on. With only
this story shipped, the run output is already observable and mineable by generic tooling — a
viable MVP.

**Independent Test**: Run a small chain; assert a JSONL file exists whose lines are each valid
standalone JSON, contain the expected event types in order, and include model/tool/outcome fields.

**Acceptance Scenarios**:

1. **Given** a `PromptChain` with one model instruction, **When** `process_prompt_async` runs to
   completion, **Then** a JSONL transcript exists with a `chain_start` line first, the instruction
   event(s), and a terminal `chain_end` line carrying the outcome.
2. **Given** a chain that calls a tool, **When** it runs, **Then** the transcript contains a
   `tool_call` line (name + arguments) and a `tool_result` line (result + status) for that call.
3. **Given** a run that errors, **When** it fails, **Then** the transcript still closes with a
   terminal event recording the error — no truncated/partial-only file.
4. **Given** a completed run, **When** the transcript is read, **Then** a non-empty `model` value
   (the basis for F2's `model_used`) is present on the relevant event(s).

---

### User Story 2 - SIO can mine PromptChain transcripts (Priority: P2)

A SIO **harness adapter** lets `sio mine` / `sio search` / `sio flows` read PromptChain JSONL
transcripts as first-class sessions, the same way SIO reads its other harnesses — invoked as
`--agent promptchain`.

**Why this priority**: The point of emitting is to be mined. Depends on US1's schema being stable.

**Independent Test**: Point the adapter at a directory of transcripts; assert SIO enumerates the
sessions and recovers their events / tool sequences.

**Acceptance Scenarios**:

1. **Given** a directory of PromptChain transcripts, **When** the SIO adapter lists sessions,
   **Then** each run appears as exactly one session with its native id, model, and timestamp.
2. **Given** a transcript with tool calls, **When** SIO mines it, **Then** the tool sequence is
   recovered for flow discovery.

---

### User Story 3 - Bounded, opt-in, low-overhead, no MLflow (Priority: P3)

Emission is opt-in (off by default), writes to a configurable location, is bounded and rotated
(whole-file caps by modification time), adds negligible overhead, and introduces no MLflow
dependency.

**Why this priority**: Production hygiene (Constitution V — Token Economy / economy-first). Important
but not required to prove the schema.

**Independent Test**: Configure a small rotation cap; generate many runs; assert old transcripts
rotate and total size stays bounded; assert emission imports and runs with `mlflow` uninstalled.

**Acceptance Scenarios**:

1. **Given** emission disabled (default), **When** a chain runs, **Then** no transcript is written
   and no emission overhead path is taken.
2. **Given** a rotation cap, **When** it is exceeded, **Then** the oldest transcripts are removed
   and the directory stays within the cap.
3. **Given** `mlflow` is not installed, **When** emission is enabled and a chain runs, **Then**
   the transcript is still produced.

### Edge Cases

- **Concurrent runs** writing simultaneously — one file per run, so lines never interleave/corrupt.
- **Streaming / partial output** — events flush incrementally; a crash leaves the prior lines valid
  and the terminal event records the failure.
- **Very large tool results** — capped/truncated in the transcript (FR-014, token economy) without
  breaking the line's JSON validity.
- **Secrets in tool args/results/messages** — redacted before writing (see FR-009; resolved, not a
  clarification).
- **Rich signals absent** (`correction_count` / `positive_signal_count` / `sidechain_count`) — emitted
  as 0 and marked optional in the schema; not required for F1.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST emit each run as an append-only JSONL transcript driven by the existing
  event system (`ExecutionEvent` / `CallbackManager`), one event per line.
- **FR-002**: Each event line MUST be valid standalone JSON and include at minimum: event type, ISO
  timestamp, and a run/session id; relevant lines MUST additionally include model name, instruction
  index, tool name + arguments, tool result + status, token counts, and the final outcome including
  a stop-reason (normal completion vs. error vs. limit reached), such that SIO can derive
  `model_used`, token totals, the tool sequence, and the error/stop-reason from the transcript alone.
- **FR-003**: The transcript MUST begin with a `chain_start` event and MUST record a terminal event
  on both success and failure (no partial-only files).
- **FR-004**: System MUST provide a SIO harness adapter (shape: `sio/harnesses/promptchain.py`,
  living in the SIO repository) that exposes PromptChain transcripts to `sio mine` / `sio suggest` /
  `sio flows` / `sio search` as `--agent promptchain`, mapping the transcript directory to SIO's
  `session_metrics` / `flow_events` / `error_records`.
- **FR-005**: Emission MUST be opt-in (disabled by default) and configurable via an enable flag and
  an output directory.
- **FR-006**: The default output location MUST be the global path
  `~/.promptchain/transcripts/<project>/<session_id>.jsonl` (configurable), NOT repo-local.
- **FR-007**: Output MUST be bounded and rotated by whole-file caps (size and/or count, evicting by
  modification time) per economy-first policy.
- **FR-008**: Emission MUST NOT require MLflow and MUST NOT import SIO's code (emit-not-reuse); the
  emitter depends only on the standard library plus `promptchain.utils.execution_events`.
- **FR-009**: The emitter MUST redact secrets (API keys / tokens) from tool arguments, tool results,
  and messages BEFORE writing — using both pattern-based and key-name-based detection, biased to
  over-redact.
- **FR-010**: The emitter MUST attach via PromptChain's public `register()` API (mirroring the
  reference `MLflowObserver`) and MUST NOT modify the event system or the emit sites.
- **FR-011**: Emission MUST be async-safe — the emitter callback is asynchronous and MUST reuse the
  existing `run_coro_blocking` helper rather than calling `asyncio.run` inside a running loop.
- **FR-012**: Emission MUST add less than 2% wall-clock overhead to a run versus emission disabled.
- **FR-013**: Optional rich-signal fields absent in F1 MUST be emitted as 0 and marked optional in
  the schema, so the contract is forward-compatible with later signal detection.
- **FR-014**: The emitter MUST cap/truncate oversized field values (notably large tool results)
  before writing, keeping the line's JSON valid and the transcript bounded per economy-first policy.

### Key Entities

- **Transcript**: one run's append-only JSONL file; identified by a run/session id; an ordered
  sequence of event records, opening with `chain_start` and closing with a terminal event.
- **Event record**: a single JSON line — `{type, ts, session_id, …type-specific fields}` — the
  locked line schema that downstream tooling depends on.
- **SIO harness adapter**: reads a transcript directory → enumerates sessions and recovers their
  events / tool sequences for SIO; lives in the SIO repo and never imports PromptChain internals.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A run with ≥1 model instruction and ≥1 tool call yields a transcript whose lines are
  100% valid JSON and contain `chain_start`, `tool_call`, `tool_result`, a terminal event, and a
  non-empty `model` value.
- **SC-002**: `sio mine` over a directory of N transcripts enumerates exactly N sessions and
  recovers their tool sequences.
- **SC-003**: With emission enabled, end-to-end run wall-clock increases by less than 2% versus
  emission disabled.
- **SC-004**: Under a configured rotation cap, the transcript directory stays within the cap across
  ≥100 runs.
- **SC-005**: The emitter imports and produces a transcript with `mlflow` uninstalled and with no
  import of SIO's code.
- **SC-006**: A run whose tool arguments or results contain a secret-shaped value (e.g. an API key)
  produces a transcript in which that value is redacted.

## Assumptions

- The existing event system already carries everything the emitter needs in event `metadata`
  (`usage`, `tool_name`, `arguments`, `result`, `error`, `model_name`, `call_id`, `total_tokens`,
  `execution_time_ms`), so the emitter is purely additive — no chain/event-system changes.
- "Project" in the default path is derived from the run context (e.g. working directory basename);
  the exact derivation is an implementation detail left to planning.
- The SIO harness adapter is delivered in the SIO repository (`~/Documents/code/SIO`) and is out of
  scope for PromptChain's own test suite beyond the schema contract it consumes.

## Out of Scope

- Any change to the loops (`test_loop_chain.py` / `ralph_chain.py` / `autoresearch.py`),
  `execution_events.py`, dev-kid, or the micro-agent fork.
- MLflow anything; the DSPy `optimize` / `experiment` "optimization-as-a-service" layer (F2+).
- Rich-signal *detection* logic — F1 only reserves the optional fields (emit 0 when absent).
