# Feature Specification: SIO Output Integration — JSONL Transcript Emitter

**Feature Branch**: `012-sio-output-integration`
**Created**: 2026-06-27
**Status**: Draft
**Input**: PRD `prd/adaptive_prompting_system_prd.md` §5 (F1, locked first) · design `research/foundation/architecture/03-sio-integration.md`

> **Context:** F1 of the Adaptive Prompting System. PromptChain must emit its runs as
> structured JSONL that SIO ("CO") can mine, WITHOUT reusing SIO's code and WITHOUT depending
> on MLflow (which may be dropped). This is locked first because the Model Profiler (F2) consumes
> this telemetry; the output schema must stabilize before downstream work.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A PromptChain run produces a structured transcript (Priority: P1)

When any `PromptChain` runs, it writes an append-only JSONL transcript capturing the run as a
sequence of events: chain start/end, each instruction, each tool call and its result, model(s)
used, token counts, timings, and the final outcome — one event per line.

**Why this priority**: This IS the lockable artifact everything downstream depends on. With only
this story shipped, the run output is already observable and mineable by generic tooling — a
viable MVP.

**Independent Test**: Run a small chain; assert a JSONL file exists whose lines are valid JSON,
contain the expected event types in order, and include model/tool/outcome fields.

**Acceptance Scenarios**:

1. **Given** a `PromptChain` with one model instruction, **When** `process_prompt_async` runs to
   completion, **Then** a JSONL transcript exists with a `chain_start` line, the instruction
   event(s), and a `chain_end` line carrying the outcome.
2. **Given** a chain that calls a tool, **When** it runs, **Then** the transcript contains a
   `tool_call` line (name + arguments) and a `tool_result` line (result + status) for that call.
3. **Given** a run that errors, **When** it fails, **Then** the transcript still closes with a
   terminal event recording the error (no truncated/partial-only file).

---

### User Story 2 - SIO can mine PromptChain transcripts (Priority: P2)

A SIO **harness adapter** lets `sio mine` / `sio search` / `sio flows` read PromptChain JSONL
transcripts as first-class sessions, the same way SIO reads other harnesses.

**Why this priority**: The point of emitting is to be mined. Depends on US1's schema being stable.

**Independent Test**: Point the adapter at a directory of transcripts; assert SIO enumerates the
sessions and returns their events/tool-sequences.

**Acceptance Scenarios**:

1. **Given** a directory of PromptChain transcripts, **When** the SIO adapter lists sessions,
   **Then** each run appears as one session with its native id, model, and timestamp.
2. **Given** a transcript with tool calls, **When** SIO mines it, **Then** the tool sequence is
   recovered for flow discovery.

---

### User Story 3 - Bounded, opt-in, low-overhead, no MLflow (Priority: P3)

Emission is configurable (opt-in path), bounded and rotated (size/count caps), adds negligible
overhead, and introduces no MLflow dependency.

**Why this priority**: Production hygiene (Constitution V — Token Economy / economy-first memory).
Important but not required to prove the schema.

**Independent Test**: Configure a small rotation cap; generate many runs; assert old transcripts
rotate and total size stays bounded; assert no `mlflow` import is required for emission.

**Acceptance Scenarios**:

1. **Given** emission disabled, **When** a chain runs, **Then** no transcript is written and there
   is no overhead path taken.
2. **Given** a rotation cap, **When** it is exceeded, **Then** oldest transcripts are removed and
   the directory stays within the cap.

### Edge Cases

- Concurrent runs writing transcripts simultaneously — no interleaved/corrupted lines (one file
  per run, or append-safe).
- A streaming run / partial output — events flush incrementally; a crash leaves valid prior lines.
- Very large tool results — capped/truncated in the transcript (token economy) without breaking JSON.
- Secrets in tool args/results — [NEEDS CLARIFICATION: redaction policy for secrets in transcripts?].
- Storage location default — [NEEDS CLARIFICATION: repo-local `./.promptchain/transcripts/` vs global `~/.promptchain/transcripts/`?].

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST emit each run as an append-only JSONL transcript via the existing event
  system (`ExecutionEvent` / `callback_manager`), one event per line.
- **FR-002**: Each event line MUST be valid standalone JSON and include at minimum: event type,
  ISO timestamp, and a run/session id; relevant lines MUST include model name, instruction index,
  tool name + arguments, tool result + status, token counts, and final outcome.
- **FR-003**: System MUST record a terminal event on both success and failure (no partial-only files).
- **FR-004**: System MUST provide a SIO harness adapter (shape: `sio/harnesses/promptchain.py`)
  that exposes PromptChain transcripts to `sio mine` / `sio search` / `sio flows`.
- **FR-005**: Emission MUST be opt-in and configurable (enable flag + output directory).
- **FR-006**: Output MUST be bounded and rotated (size and/or count caps) per economy-first policy.
- **FR-007**: Emission MUST NOT require MLflow and MUST NOT import SIO's code (emit-not-reuse).
- **FR-008**: Emission MUST be async-safe and add <2% overhead to a run.

### Key Entities

- **Transcript**: one run's append-only JSONL file; identified by a run/session id; ordered events.
- **Event record**: a single JSON line — {type, ts, session_id, …type-specific fields}.
- **SIO harness adapter**: reads a transcript directory → enumerates sessions + their events for SIO.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A run with ≥1 model instruction and ≥1 tool call yields a transcript whose lines are
  100% valid JSON and contain `chain_start`, `tool_call`, `tool_result`, and a terminal event.
- **SC-002**: `sio mine` over a directory of N transcripts enumerates exactly N sessions and
  recovers their tool sequences.
- **SC-003**: With emission enabled, end-to-end run wall-clock increases by <2% vs disabled.
- **SC-004**: Under a configured rotation cap, transcript directory size stays within the cap
  across ≥100 runs; emission works with `mlflow` uninstalled.
