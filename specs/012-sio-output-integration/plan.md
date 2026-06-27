# Implementation Plan: SIO Output Integration — JSONL Transcript Emitter

**Branch**: `012-sio-output-integration` | **Date**: 2026-06-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/012-sio-output-integration/spec.md`

## Summary

Add a self-contained `TranscriptEmitter` observer under `promptchain/observability/` that listens
to the existing `CallbackManager` event stream and writes each PromptChain run as one append-only
JSONL transcript (one event per line) at `~/.promptchain/transcripts/<project>/<session_id>.jsonl`.
It mirrors the existing `MLflowObserver` plugin pattern (attach via the public
`register_callback()` API; no changes to the event system or emit sites), but imports only the
standard library plus `promptchain.utils.execution_events`. The emitter redacts secrets and
truncates oversized values before writing, is opt-in (off by default), and keeps the directory
bounded via whole-file rotation by mtime. The **locked JSONL line schema** is the deliverable
contract that the Model Profiler (F2) and the SIO harness adapter consume; the adapter itself
ships in the SIO repo and is out of scope here.

## Technical Context

**Language/Version**: Python 3.10+ (repo CI runs 3.10 and 3.12; current active 3.12.11)
**Primary Dependencies**: standard library only (`json`, `os`, `pathlib`, `datetime`, `re`,
`threading`) + `promptchain.utils.execution_events`. NO `mlflow`, NO `sio` import.
**Storage**: append-only JSONL files at `~/.promptchain/transcripts/<project>/<session_id>.jsonl`
(configurable base dir); one file per run.
**Testing**: pytest (flat `tests/test_*.py`, mirroring the existing `test_observability_*` /
`test_mlflow_observer` files).
**Target Platform**: Linux/macOS — a cross-platform Python library.
**Project Type**: single (library).
**Performance Goals**: <2% wall-clock overhead with emission enabled vs disabled.
**Constraints**: opt-in (off by default); bounded + rotated output; secrets redacted; oversized
values truncated; async-safe; works with `mlflow` uninstalled; no SIO import.
**Scale/Scope**: one transcript per run; rotation keeps the directory bounded across ≥100 runs.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | How this feature satisfies it |
|---|---|---|
| I. Library-First | ✅ | Self-contained observer module; minimal deps (stdlib + event types); independently testable; clear single purpose. |
| II. Observable Systems | ✅ | This feature *is* the structured-JSONL observability surface the constitution calls for. |
| III. Test-First (NON-NEGOTIABLE) | ✅ | Plan mandates contract + unit + integration tests written first (red), then implementation (green). |
| IV. Integration Testing | ✅ | A contract test pins the JSONL line schema (the public API for F2/SIO); an integration test verifies `register_callback()` wiring on a real chain run. |
| V. Token Economy & Performance | ✅ | Bounded + rotated output, truncation of large values, <2% overhead, opt-in so default cost is zero. |
| VI. Async-First Design | ✅ | Emitter registers as an async callback (the CallbackManager already awaits async callbacks); no `asyncio.run` inside a running loop. |
| VII. Simplicity & Maintainability | ✅ | One observer module + small redaction helper; no speculative features (rich-signal detection deferred to F2+; only the schema slots are reserved). |

**Result**: PASS — no violations. Complexity Tracking left empty.

## Project Structure

### Documentation (this feature)

```text
specs/012-sio-output-integration/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── transcript-schema.md   # The LOCKED JSONL line schema (F2 + SIO consume this)
├── checklists/
│   └── requirements.md  # spec quality checklist (from /speckit-specify)
└── tasks.md             # /speckit-tasks output (NOT created here)
```

### Source Code (repository root)

```text
promptchain/observability/
├── transcript_emitter.py     # NEW — TranscriptEmitter observer (attach via register_callback)
├── _transcript_redaction.py  # NEW — secret redaction + value truncation helpers (stdlib only)
├── mlflow_observer.py        # reference pattern — UNCHANGED
└── __init__.py               # export TranscriptEmitter (additive)

tests/
├── test_transcript_schema_contract.py   # contract: every line valid JSON + locked schema (write FIRST)
├── test_transcript_emitter_unit.py      # redaction, truncation, rotation, path resolution
├── test_transcript_emitter_integration.py  # register_callback() on a real chain → transcript; error path
└── test_transcript_emitter_performance.py  # <2% overhead enabled-vs-disabled
```

The SIO harness adapter (`sio/harnesses/promptchain.py`) lives in `~/Documents/code/SIO` and is NOT
part of this repo's source tree — only the schema contract it consumes is delivered here.

**Structure Decision**: Single-project library layout. The new code is two modules under the
existing `promptchain/observability/` package (so it sits beside `mlflow_observer.py`, its
reference), plus four flat pytest files matching the repo's existing `test_observability_*`
naming convention. No changes to `promptchain/utils/execution_events.py`, `promptchaining.py` emit
sites, the loops, dev-kid, or the micro-agent fork.

## Complexity Tracking

> No Constitution violations — section intentionally empty.
