# Decision (LOCKED) — Integrate SIO into PromptChain via JSONL session transcripts

**Decision (2026-06-27, user):** INTEGRATE SIO *into* PromptChain — don't just call it externally and
don't just reuse code. The **primary integration surface = PromptChain emits JSONL session transcripts
that SIO mines.** This is **MLflow-independent** on purpose (we may drop MLflow); JSONL transcripts become
the durable telemetry standard, and we keep it that way. Plus optimization-as-a-service + other SIO
techniques on top.

## Why JSONL transcripts (not MLflow)
- MLflow may be dropped → JSONL transcripts are the durable, portable, human-auditable telemetry surface.
- **SIO is built to add harnesses:** `sio/harnesses/__init__.py` — "adding a new harness is bounded to
  writing ONE adapter module." PromptChain becomes the **7th SIO harness** (after claude/codex/goose/
  opencode/gemini/aider).
- Transcripts replay + pair naturally with Document-&-Clear (state already lives on disk).

## Build (two thin pieces — this is "dealing with PromptChain's output first")
1. **PromptChain JSONL transcript emitter.** The TUI writes one JSON object per line to
   `~/.promptchain/transcripts/<project>/<session_id>.jsonl`, mirroring the Claude-Code transcript shape
   SIO already parses (minimal new parser). Each line carries enough to derive SIO's `session_metrics`:
   - **message:** role · content · `model` · timestamp · token usage (input/output/cache_read/cache_create) · stop_reason
   - **tool events:** `tool_use {name, input}` · `tool_result {output, is_error}`
   - **signals:** errors · user corrections · positive signals · sidechains
   - **session:** id · cwd/project · start/end · cost_usd
2. **SIO PromptChain harness adapter** — `sio/harnesses/promptchain.py`: register the transcript dir +
   field mapping so `sio mine --agent promptchain` fills `session_metrics` (incl. **`model_used`**) +
   `flow_events` + `error_records` + `behavior_invocations`.

## SIO `session_metrics` target (verified schema — what the transcript must let SIO derive)
`session_id · file_path · total_input/output/cache_read/cache_create tokens · cache_hit_ratio · cost_usd ·
session_duration_seconds · message_count · tool_call_count · error_count · correction_count ·
positive_signal_count · sidechain_count · stop_reason_distribution · `**`model_used`**` · mined_at`.
→ `model_used` is the linchpin for the Model Profiler.

## On top of the transcript surface
- **Optimization-as-a-service:** `sio optimize --optimizer gepa` (`SIO_TASK_LM=<target>`) compiles jackets;
  `sio experiment` A/Bs them with `config_hash`.
- **Other SIO techniques:** `sio flows` (per-model winning tool-sequences) · `sio suggest` (rule
  suggestions) · `sio velocity`/`trend` (per-model error trends) · `sio scan` (mine).
- The Model Profiler **reads** SIO's `session_metrics` (the measured ledger) AND **writes** its 10-cast
  probe trials *as transcripts* (so `sio mine` ingests them). One surface, both directions.

## Net / build order
PromptChain's session **OUTPUT = JSONL transcripts in SIO-mineable schema** (the locked, MLflow-independent
telemetry standard). SIO gains a PromptChain harness adapter. **Build order:** emitter + adapter FIRST (the
telemetry foundation), THEN the probe harness + `model_prompt_generator` DSPy module (the Model Profiler).
This supersedes the infographic's "build a SIO sink" framing — it's a first-class integration, not a sink.
