# Quickstart: SIO Output Integration — JSONL Transcript Emitter

## Enable emission on a chain

```python
from promptchain import PromptChain
from promptchain.observability import TranscriptEmitter

chain = PromptChain(models=["ollama/qwen3-coder:30b"], instructions=["{input}"])

emitter = TranscriptEmitter(enabled=True)          # or env: PROMPTCHAIN_TRANSCRIPTS_ENABLED=true
chain.register_callback(emitter.handle_event)      # public API — mirrors MLflowObserver

await chain.process_prompt_async("write a function")
# → ~/.promptchain/transcripts/<project>/<session_id>.jsonl
```

Default is OFF — with no emitter registered (or `enabled=False`) nothing is written and there is
no overhead path.

## Inspect a transcript

```bash
ls ~/.promptchain/transcripts/<project>/
head -1 ~/.promptchain/transcripts/<project>/<session_id>.jsonl   # → {"type":"chain_start", ...}
python -c 'import json,sys; [json.loads(l) for l in open(sys.argv[1])]' <file>   # all lines valid JSON
```

## Mine with SIO (after the SIO adapter ships in the SIO repo)

```bash
sio search "pattern" --agent promptchain
sio mine  --agent promptchain
sio flows --agent promptchain
```

## Validate against the success criteria

| Check | How |
|---|---|
| SC-001 valid-JSON + required types + non-empty model | run a chain w/ a tool, assert lines parse and contain `chain_start`/`tool_call`/`tool_result`/terminal + `model` |
| SC-003 <2% overhead | benchmark enabled vs disabled |
| SC-004 bounded | set a small `max_files`, run ≥100, assert dir stays within cap |
| SC-005 no mlflow / no sio | `pip uninstall mlflow`; import + emit; assert no `import sio` |
| SC-006 redaction | pass a tool arg containing `sk-…`; assert it's redacted in the file |

## Run the tests

```bash
cd /home/gyasis/Documents/PromptChain.wt-epic-adaptive-prompting
python -m pytest tests/test_transcript_schema_contract.py tests/test_transcript_emitter_unit.py \
                 tests/test_transcript_emitter_integration.py -q
```

Live (offline, real model on the Mac Studio ollama):

```bash
OLLAMA_API_BASE=http://192.168.0.159:11434 PROMPTCHAIN_LOOP_MODEL=ollama/qwen3-coder:30b \
  python -m pytest tests/test_transcript_emitter_integration.py -q
```
