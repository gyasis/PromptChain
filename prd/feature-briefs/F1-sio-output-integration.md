# Feature Brief F1 — SIO Output Integration (JSONL Transcript Emitter)

> **Purpose of this doc:** the locked, self-contained input for `/speckit-specify`. Everything
> below is DECIDED — specify should formalize it into spec.md, not re-research or re-open it.
> Source: PRD `prd/adaptive_prompting_system_prd.md` §5 · design `research/foundation/architecture/03-sio-integration.md`.
> **Epic order: F1 FIRST (locked) → F2 → F3.** F2 consumes this transcript schema.

## Hand to /speckit-specify (the framing)
"PromptChain must emit every run as an append-only JSONL transcript that SIO can mine as a 7th
harness — WITHOUT importing SIO's code and WITHOUT any MLflow dependency. Add a transcript-emitter
observer (mirroring the existing MLflowObserver) plus a SIO harness adapter. The schema is the
lockable artifact F2 depends on."

## Context & dependencies
- **Built already (do not rebuild):** the event system — `promptchain/utils/execution_events.py`
  (`ExecutionEventType`, `ExecutionEvent.to_dict()`), `execution_callback.py` (`CallbackManager.register/emit`),
  and `promptchain/observability/mlflow_observer.py` (the reference observer pattern). `PromptChain`
  exposes `self.callback_manager` + public `register(callback, event_filter)` (promptchaining.py:648).
- Event `metadata` already carries `usage`, `tool_name`, `arguments`, `result`, `error`, `model_name`,
  `call_id`, `total_tokens`, `execution_time_ms` → the emitter is **purely additive**, no chain changes.
- **Depends on:** nothing (F1 is first). **Blocks:** F2 (Model Profiler reads `session_metrics.model_used` derived from these transcripts).

## Decisions ALREADY MADE (locked — do not re-research)
1. **Storage default = global** `~/.promptchain/transcripts/<project>/<session_id>.jsonl` (configurable). NOT repo-local.
2. **Secrets = REDACT** API keys/tokens from tool args/results/messages BEFORE writing (pattern + key-name based; over-redact).
3. **Emit-not-reuse:** emitter imports only stdlib + `promptchain.utils.execution_events`. **No `mlflow`. No `sio` import.**
4. **Attach via the public `register()` API**, like MLflowObserver. Do not edit the event system or emit sites.
5. **Async-safe:** emitter callback is `async def`; never bare `asyncio.run` in a running loop — reuse `run_coro_blocking`.
6. **One file per run** (concurrent runs never interleave). **Bounded + rotated** (whole-file, by mtime).
7. **The SIO adapter lives in the SIO repo** (`~/Documents/code/SIO`, shape `sio/harnesses/promptchain.py`) — PromptChain never imports SIO.

## In scope
- A `TranscriptEmitter` observer under `promptchain/observability/` (write JSONL, redact, truncate large results, rotate, opt-in).
- A SIO harness adapter (`sio/harnesses/promptchain.py`) mapping the transcript dir → SIO `session_metrics` / `flow_events` / `error_records`, registered so `sio mine|search|flows --agent promptchain` work.
- A locked **line schema** (the contract): every line valid standalone JSON; `chain_start` first, terminal event last; must let SIO derive `model_used` + token totals + tool sequence + error/stop-reason.

## Out of scope (do NOT let specify wander into)
- Any change to the loops, `execution_events.py`, dev-kid, or the micro-agent fork.
- MLflow anything. The DSPy `optimize`/`experiment` "optimization-as-a-service" layer (rides on the surface later, F2+).
- Rich signal detection (`correction_count`/`positive_signal_count`/`sidechain_count`) — emit 0 when absent; schema marks them optional.

## User stories (priority)
- **US1 (P1, MVP):** a run produces a structured JSONL transcript (chain_start, instruction/tool events, terminal event; one event/line).
- **US2 (P2):** SIO mines PromptChain transcripts as first-class sessions (enumerate + recover tool sequences).
- **US3 (P3):** bounded, opt-in, <2% overhead, no MLflow.

## Acceptance / success criteria (measurable)
- Run with ≥1 model instruction + ≥1 tool call → transcript lines 100% valid JSON containing `chain_start`, `tool_call`, `tool_result`, terminal event, and a non-empty `model` (= `model_used`).
- `sio mine` over N transcripts → exactly N sessions, tool sequences recovered.
- Emission-on vs off: <2% wall-clock delta. Rotation cap holds across ≥100 runs. Emitter imports + runs with `mlflow` uninstalled.

## Guardrails
- Test-first (Constitution III). Observability is the feature (Constitution II). Bounded output (Constitution V / economy-first).

## Note on existing artifact
A hand-drafted spec exists at `specs/012-sio-output-integration/spec.md` (+ plan/research/data-model/contracts/tasks were drafted then discarded). **Decide:** reuse `012` (run `/speckit-plan` + `/speckit-tasks` on it) OR `rm -rf specs/012-...` and let `/speckit-specify` regenerate F1 fresh from THIS brief. Either way the locked decisions above apply.
