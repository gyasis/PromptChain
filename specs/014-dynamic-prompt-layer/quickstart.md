# Quickstart — Dynamic Prompt Layer (F3)

How to assemble a per-model prompt, and how to run the offline live smoke / A/B.

## 1. Assemble a per-model prompt (US1)

```python
from promptchain.prompts import DynamicModelPromptGenerator

# Model bound at construction (drop-in BasePromptBuilder path); reads ~/.promptchain/model_profiles.json
gen = DynamicModelPromptGenerator(model="ollama/qwen3-coder:30b")

tools = [{"function": {"name": "file_read", "description": "Read a file"}}]
prompt = gen.generate(objective="Summarize file X", tools=tools)
# → static base VERBATIM + family-adapted framing + tier optional modules, trimmed to budget
print(gen.get_token_estimate("Summarize file X", tools))   # within 300–1,000 (hard cap 1500)

# Direct/test use: override the model per call (keyword-only)
weak = gen.generate("Summarize file X", tools, model="ollama/llama3.2:1b")
assert weak != prompt          # measurably different (tier/family)  — SC-002
```

Fallbacks (SC-005): a model with a **null jacket** uses `recommended_tier` + `budget_tokens`; a
model with **no profile** uses the default CORE tier. Neither raises.

## 2. Drop it in at the agentic seam

`DynamicModelPromptGenerator` is a `BasePromptBuilder`, so it slots in wherever
`DynamicTUIPromptGenerator` is used (the `prompt_builder` consumed at
`agentic_step_processor.py:1006`) — `generate(objective, tools, context)` unchanged; the static base
stays intact and additive.

## 3. Toolshim for a non-tool-calling model (US2)

```python
from promptchain.profiler.jacket import Jacket
# a jacket whose tool_mode marks a non-native model:
jk = Jacket(tier="standard", budget_tokens=1000, mode="single-shot+retry", spawn_temp=0.5,
            compress_at=0.6, max_turns=8, role="both", tool_mode="shim_prompt")
# generate() renders a <tools> JSON-in-text block + plain-text tool history; native models do not.
```

## 4. Weak-model longevity (US3)

```python
from promptchain.prompts.longevity import DocumentAndClear, build_turn_context

dac = DocumentAndClear(compress_at=0.60, min_turns=10, jacket=jk)
ctx = build_turn_context("Solve task X", turn=3)        # <turn-context> re-injecting the goal
if dac.should_compress(context_usage_fraction=0.62):    # ≥ 60%
    history = dac.document_and_clear(working_dir="/path/to/work", state={...})  # writes PROGRESS.md, clears, resumes
# escalate only on a real stall, and only if the jacket allows it:
if dac.should_escalate(stalled=dac.is_stalled(progress_signals)):
    ...  # hand off to a bigger model
```

## 5. Run the tests (test-first gate)

```bash
# F3 unit + integration (red first, then green):
python -m pytest tests/test_dynamic_prompt_*.py tests/test_profiler_jacket_toolmode.py -q
# No F1/F2 regression:
python -m pytest tests/test_profiler_*.py tests/test_transcript_*.py -q
```

## 6. Offline live smoke + A/B (SC-007)

Real model, no secrets (mirrors F2's smoke). litellm prefix `ollama/<model>`:

```bash
OLLAMA_API_BASE=http://192.168.0.159:11434 \
PYTHONPATH=/home/gyasis/Documents/PromptChain.wt-epic-adaptive-prompting \
python specs/014-dynamic-prompt-layer/scripts/ab_smoke.py --weak ollama/llama3.2:1b
# → per-arm completion rate; expect F3 ≥ static-base on the weak model (the A/B win)
```

(The deterministic A/B harness — task set, scoring, aggregation — is unit-tested with a fake model;
this script is the live demonstration.)
