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
- The recorder pairs each start event with its end event on the `trace_call_id` key that
  `PromptChain` adds to the model-call events. Concurrent runs can share a recorder safely.
  Events that lack the id (from another emitter) pair on step number, which is **unsafe under
  concurrency**: overlapping calls can swap completions. The recorder detects an overlap on that
  fallback key, drops the affected records and logs a warning. Call `recorder.clear()` to reuse
  a recorder.
- `CallbackManager.emit` skips an event that arrives while another emit is still running on the
  same chain object. Two concurrent runs on **one chain object** can therefore lose events and
  yield fewer records. Use one chain object per concurrent run. The recorder never mispairs
  the records it keeps.
- The recorder skips and logs an end event whose response has a `warning` key, or has no
  content and no tool calls.
- Every message keeps only `role`, `content`, `tool_calls`, `tool_call_id` and `name`, and
  drops keys whose value is `None`. Usage, refusal and reasoning fields do not reach the file.
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

The result is one `{"messages": [...], "tools"?: [...], "reward"?: float, "redactions": int}`
record per call. Know what it represents:

- `records_from_agentic` raises `ValueError` on an empty history.
- After a failed `run_async`, `processor.internal_history` holds the partial history of the
  failed run, not the previous run's. That history ends mid-conversation. Check that the last
  message is a final assistant answer before you train on it.
- The record is the whole internal conversation. With `history_mode="minimal"` the model saw
  only the system message, the objective and the latest tool exchange on each step. The record
  then shows the trajectory, not the exact prompt at each step. With `"progressive"` or
  `"kitchen_sink"` the model saw the accumulated history, and the record is closer to it.

## Load it for training

```python
from datasets import load_dataset

chain_data = load_dataset("json", data_files="traces.jsonl", split="train")
agentic_data = load_dataset("json", data_files="agentic.jsonl", split="train")
```

`datasets` is not a PromptChain dependency. Install it with `pip install "datasets>=4.7"`.

Tool-call `arguments` differ in shape from one tool to the next, and so do the `tools` schemas.
Version 4.7 and later of `datasets` stores such free-form values with the `Json` type, so
mixed rows load. On an older version, the load can fail on conflicting struct types. Then store
the free-form columns as JSON strings before you write the file (for example
`rec["tools"] = json.dumps(rec["tools"])`) and `json.loads` them in a `map` step before
training. This recipe ran only the `>=4.7` path.

## How this fits TRL

Based on the TRL docs for `SFTTrainer`, the dataset formats page and `GRPOTrainer`:

| TRL format | Our output | Fit |
|---|---|---|
| Conversational prompt-completion: `{"prompt": [msgs], "completion": [assistant msg]}` | `TraceRecorder` records | Direct. `SFTTrainer` applies the chat template. |
| Conversational language modeling: `{"messages": [...]}` | `records_from_agentic` records | Direct. |
| `tools` column, JSON schemas, with `tool_calls` in messages and `tool` role messages | `tools` plus the converted `tool_calls` | Direct. The `tools` value is the OpenAI-style schema list PromptChain already holds. |
| GRPO `reward_funcs` receive every extra dataset column as a keyword argument | `reward`, `model`, `step` and `redactions` columns | Partial. See below. |

Limits you should know about:

- These are **SFT shapes.** A trace is one sampled completion, not a preference pair. DPO and
  KTO need chosen/rejected or labelled data. This module does not build them.
- `reward` is an **extra column.** GRPO scores completions the trainer generates itself, so a
  stored `reward` does not feed GRPO as a reward. Use it for offline filtering before SFT
  (`data.filter(lambda r: r["reward"] >= 1.0)`), and a reward function can read the column as a
  keyword argument if you pass the data as a GRPO dataset.
- The extra columns `model`, `step` and `redactions` are scalars. If your trainer rejects
  unknown columns, drop them with `data.remove_columns(["model", "step", "redactions"])`.
- A record whose assistant `content` is `None` (a tool call) next to records with string
  content can make `datasets` infer a nullable string column. If your loader complains, load
  the file with an explicit `features=` argument.

## What is and is not captured

Captured: every `PromptChain` model call that runs while the recorder listens. That includes the
tool follow-up call, with the exact messages sent and the assistant message received.

Not captured:

- Agentic step internals as events. Use `records_from_agentic`.
- Calls made with no callback registered. `PromptChain` adds the text to event metadata
  only when at least one callback exists, so an unobserved run pays nothing.
- Token usage and timings. The transcript and MLflow layers keep those.

## Redaction

`redact_secrets=True` is the default. The recorder replaces **only the matched span** and
keeps the text around it. It stores the number of replacements in a `redactions` column.
Pass `drop_redacted_completions=True` to `TraceRecorder` to drop records whose completion needed a
replacement. A model trained on such a completion learns to emit `***REDACTED***`. Redacted
prompts are less harmful, so the option leaves them in.

Caught:

- `sk-` keys, `Bearer` tokens, JWTs, AWS `AKIA` key ids, Slack `xoxa-`, `xoxb-` and `xoxp-`
  tokens, GitHub `ghp_`-style and `github_pat_` tokens, Hugging Face `hf_` tokens.
- The whole value of a dict key that is exactly `api_key`, `apikey`, `api-key`,
  `authorization`, `password`, `passwd`, `secret`, `token`, `access_token`, `refresh_token`,
  `client_secret` or `private_key` (any case). That covers parsed tool-call arguments.
  The recorder keeps keys such as `max_tokens` and `token_count`.

Never redacted: values of `id` and `tool_call_id` keys, file paths, git SHAs, UUIDs, base64
data URLs, and the `tools` schema column.

Known false negatives, which you must handle yourself:

- Plain-language secrets ("my password is hunter2") and secrets with no known prefix.
- A secret inside a JSON string, under a key name, such as the text `"api_key": "abc"` in a
  tool result. The key match works on dict keys, not on text.
- Any other provider's key format.

Known false positives: a string that looks like a listed key, for example `sk-` followed by ten
or more letters that is not a key. Review a sample of the output before you train.

`redact_secrets=False` keeps everything. Use it only for data that never leaves your machine.

The new `prompt_messages`, `tools`, `response_message` and `trace_call_id` event keys carry
raw, unredacted text to every registered callback. They differ from the `messages` and
`response` keys that `MLflowObserver` reads. They also differ from the `call_id` key that the
transcript and MLflow layers read. Those layers log nothing new. Any callback you write yourself receives
the raw text.

## Cross-references

- [recipe-observability-on](recipe-observability-on.md) for the MLflow and transcript layers.
- [recipe-agentic-step](recipe-agentic-step.md) for `AgenticStepProcessor`.
