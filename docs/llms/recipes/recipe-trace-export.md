---
id: recipe-trace-export
title: Export a run as TRL-shaped training data
when: You want the prompts and completions of a PromptChain run as JSONL for supervised fine-tuning or for offline reward filtering.
api_version: 0.6.1+
---

# Trace Export

`promptchain.utils.trace_export` records what a chain sent to the model and what came back,
and writes it in the dataset shapes that Hugging Face TRL reads. The existing transcript
(`TranscriptEmitter`) and MLflow layers record counts and timings, not text. This module is
the only place that captures the prompt and completion text.

## Pick the capture path

| Your run | Use | Output record |
|---|---|---|
| `PromptChain` with plain steps or tool calls | `TraceRecorder` registered as a callback | `{"prompt", "completion", "tools"?, "model", "step", "reward"?}` |
| `AgenticStepProcessor` | `records_from_agentic(...)` after the run | `{"messages", "tools"?, "reward"?}` |

`AgenticStepProcessor` runs its own loop and emits no `MODEL_CALL_*` events. A recorder on the
chain sees nothing from an agentic step. Use `records_from_agentic` for those.

## Export a chain run (runs offline)

The example patches `run_model_async` so it needs no key and no network. Remove the patch to
record a real run.

```python
import asyncio
from unittest.mock import AsyncMock, patch

from promptchain import PromptChain
from promptchain.utils.trace_export import TraceRecorder, write_jsonl


async def main():
    chain = PromptChain(
        models=["openai/gpt-4o-mini"] * 2,
        instructions=["Summarize the following text. {input}", "Shorten the following text. {input}"],
    )
    recorder = TraceRecorder()
    chain.register_callback(recorder.handle_event)

    fake = AsyncMock(side_effect=[
        {"role": "assistant", "content": "A long summary."},
        {"role": "assistant", "content": "Short."},
    ])
    with patch.object(PromptChain, "run_model_async", fake):
        await chain.process_prompt_async("Some document text")

    records = recorder.records(reward=lambda rec: 1.0 if rec["step"] == 2 else None)
    print(write_jsonl(records, "traces.jsonl"))   # -> 2


asyncio.run(main())
```

Each line of `traces.jsonl` is one model call:

```json
{"prompt": [{"role": "user", "content": "Summarize the following text.\n\nSome document text"}],
 "completion": [{"role": "assistant", "content": "A long summary."}],
 "model": "openai/gpt-4o-mini", "step": 1}
```

Rules for the recorder:

- Register it **before** the run, with `chain.register_callback(recorder.handle_event)`.
- Use **one recorder per run.** It pairs each start event with its end event by step number.
  Two concurrent runs that share a recorder mix their records. Call `recorder.clear()` to reuse it.
- The recorder drops a model call that raises and logs a warning. It never writes a half record.
- A tool-calling step gives two records: the call that asked for the tool, then the follow-up
  call whose prompt holds the tool results.
- The recorder converts `tool_calls[].function.arguments` from the JSON string the API returns
  into a dict, which is the shape TRL chat templates expect. A string that is not valid JSON
  stays a string, and the recorder logs a warning.
- `records(reward=fn)` calls `fn(record)` for each record. A float goes into a `reward`
  column. `None` leaves the column out for that record.

## Export an agentic step (runs offline)

```python
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.trace_export import records_from_agentic, write_jsonl

history = [
    {"role": "system", "content": "You are a file assistant."},
    {"role": "user", "content": "How many .log files are in /tmp?"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "count_files", "arguments": "{\"pattern\": \"*.log\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "count_files", "content": "3"},
    {"role": "assistant", "content": "There are 3 .log files."},
]

records = records_from_agentic(history, reward=1.0)
write_jsonl(records, "agentic.jsonl")

# With a real processor, pass the object. It reads .internal_history after the run:
#   processor = AgenticStepProcessor(objective="...", model_name="openai/gpt-4o-mini")
#   ... run it inside a chain ...
#   records = records_from_agentic(processor, tools=chain.tools, reward=1.0)
```

The result is one `{"messages": [...], "tools"?: [...], "reward"?: float}` record per call.

## Load it for training

```python
from datasets import load_dataset

chain_data = load_dataset("json", data_files="traces.jsonl", split="train")
agentic_data = load_dataset("json", data_files="agentic.jsonl", split="train")
```

`datasets` is not a PromptChain dependency. Install it with `pip install datasets`.

## How this fits TRL

Checked against the TRL docs (`docs/source/sft_trainer.md`, `dataset_formats.md`,
`grpo_trainer.md`, through context7, library `/huggingface/trl`, 2026-11-26):

| TRL format | Our output | Fit |
|---|---|---|
| Conversational prompt-completion: `{"prompt": [msgs], "completion": [assistant msg]}` | `TraceRecorder` records | Direct. `SFTTrainer` applies the chat template. |
| Conversational language modeling: `{"messages": [...]}` | `records_from_agentic` records | Direct. |
| `tools` column, JSON schemas, with `tool_calls` in messages and `tool` role messages | `tools` plus the converted `tool_calls` | Direct. The `tools` value is the OpenAI-style schema list PromptChain already holds. |
| GRPO `reward_funcs` receive every extra dataset column as a keyword argument | `reward`, `model` and `step` columns | Partial. See below. |

Limits you should know about:

- These are **SFT shapes.** A trace is one sampled completion, not a preference pair. DPO and
  KTO need chosen/rejected or labelled data. This module does not build them.
- `reward` is an **extra column.** GRPO scores completions the trainer generates itself, so a
  stored `reward` does not feed GRPO as a reward. Use it for offline filtering before SFT
  (`data.filter(lambda r: r["reward"] >= 1.0)`), and a reward function can read the column as a
  keyword argument if you pass the data as a GRPO dataset.
- The extra columns `model` and `step` are scalars. If your trainer rejects unknown columns,
  drop them with `data.remove_columns(["model", "step"])`.
- A record whose assistant `content` is `None` (a tool call) next to records with string
  content can make `datasets` infer a nullable string column. If your loader complains, load
  the file with an explicit `features=` argument.

## What is and is not captured

Captured: every `PromptChain` model call made while the recorder is registered. That includes the
tool follow-up call, with the exact messages sent and the assistant message received.

Not captured:

- Agentic step internals as events. Use `records_from_agentic`.
- Calls made with no callback registered. `PromptChain` adds the text to event metadata
  only when at least one callback exists, so an unobserved run pays nothing.
- Token usage and timings. The transcript and MLflow layers keep those.

## Redaction

`redact_secrets=True` is the default. It runs `promptchain.observability._transcript_redaction.redact`
on prompts, completions and agentic messages.

- A string value that looks like a secret (`sk-...` keys, `Bearer ...` headers, JWTs, long hex
  or mixed-case base64-like tokens) becomes `***REDACTED***`. **The whole string value
  changes**, not only the secret. A long prompt with one key inside becomes
  `***REDACTED***`. This favours safety over data yield. Review a sample of the output.
- A dict key named like `api_key`, `token`, `secret`, `password`, `authorization` or `bearer`
  has its value replaced.
- The `tools` schema column is **not** redacted, because a parameter named `token` would
  otherwise replace the whole parameter schema. Do not put secrets in tool descriptions.
- `redact_secrets=False` keeps everything. Use it only for data that never leaves your machine.

The new `prompt_messages`, `tools` and `response_message` event keys carry the raw, unredacted
text to every registered callback. They are separate from the `messages` and `response` keys
that `MLflowObserver` reads, so MLflow logs no new artifacts. Any callback you write yourself
receives them unredacted.

## Cross-references

- [recipe-observability-on](recipe-observability-on.md) for the MLflow and transcript layers.
- [recipe-agentic-step](recipe-agentic-step.md) for `AgenticStepProcessor`.
