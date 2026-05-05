# Medgemma Hybrid Clinical Demo (2026-05-05)

## Intent

Prove the **hybrid pattern** documented in `docs/llms/recipes/recipe-hybrid-chain.md` and `docs/llms/PROMPTCHAIN_FOR_LLMS.md §4`:

- **Step 1 (AGENTIC):** `openai/gpt-4o-mini` autonomously calls retrieval tools (`get_labs`, `get_history`) to assemble a structured case JSON
- **Step 2 (SEQUENTIAL):** `ollama/medgemma:4b` (local, specialist) produces the clinical reasoning report from the structured JSON

Both run inside a SINGLE `PromptChain`. Step 2 sees only step 1's `final_answer`, not its internal reasoning trace (history isolation).

## Expected output

A markdown clinical report with 4 sections:
1. Primary diagnosis (expected: inferior wall STEMI)
2. Top 3 differentials
3. Immediate management plan with doses (each dose flagged `[VERIFY]` if model is uncertain)
4. Time-sensitive intervention (expected: primary PCI / fibrinolysis)

## Triggered by

Ad-hoc demo built during 2026-05-05 conversation about the two operating modes (sequential vs agentic vs hybrid).

## Required env

- `OPENAI_API_KEY` (for the agentic retrieval step)
- Ollama running locally with `medgemma:4b` pulled (`ollama pull medgemma:4b`)

## Run

```bash
# From repo root, with .env loaded and MLflow tracking on:
bash scripts/observe.sh runs/2026-05-05_medgemma-clinical-demo
```

Output will be tee'd to `output.log` (gitignored).

## What success looks like

- Agentic step makes 2 tool calls (get_labs + get_history), each returning JSON
- Step 2's input is the combined JSON
- medgemma synthesis matches the expected sections
- MLflow records both LLM calls + the 2 tool invocations
- Total latency: ~30-60s (gpt-4o-mini retrieval ~5-10s + medgemma synthesis ~30-50s)

## What could go wrong

- gpt-4o-mini hallucinates the JSON structure instead of calling tools → agentic step's `objective` not strict enough; tighten it
- medgemma includes doses without `[VERIFY]` flags → expected for textbook STEMI doses (ASA 325 mg, etc.); flagged ones are the safety bonus
- Agentic step iterates more than expected → check tool return values are parseable JSON

## ⚠️ Known failure observed on first runs (2026-05-05)

**Both** the original objective AND a strengthened "you MUST call both tools" version produced the SAME failure mode:

- Run 1 (loose objective): agentic step returned `{}` (1 completion token), `tool_calls=None`
- Run 2 (strict "MUST call both"): agentic step returned `{"labs": null, "history": null}` (14 tokens), `tool_calls=None`

In both cases, `chain.register_tool_function(get_labs)` and `register_tool_function(get_history)` were called on the parent `PromptChain` BUT gpt-4o-mini never emitted tool calls. Step 2 (medgemma) still produced a plausible clinical report — but only because the *case complaint* was embedded in step 2's prompt, not because step 1 actually retrieved data.

**Hypothesis (to investigate):** local tool functions registered on the parent `PromptChain` may not automatically propagate to the `AgenticStepProcessor`'s internal LLM calls under the v0.6.0+ `DynamicPromptGenerator` default. The agentic step may need its own tool list, OR the prompt builder may not be including tool schemas for the agentic loop.

**Next steps:**
1. Read `agentic_step_processor.py` to confirm whether `chain.local_tool_functions` is read by the agentic loop, OR whether tools must be passed via `AgenticStepProcessor(... tools=[...])`.
2. Check `promptchain/prompts/dynamic.py` — `DynamicPromptGenerator` should be reading registered tools from somewhere; if it's not, that's the bug.
3. If confirmed: file a `gyasis/PromptChain` issue for the propagation gap, and update `recipe-agentic-step.md` + `recipe-hybrid-chain.md` with the workaround.

This is **the closed loop working as designed** — the demo surfaced a real bug. Logged as the first real entry in `docs/llms/FEEDBACK_LOG.md`.

The two run outputs are preserved:
- `output.log` — the most recent run (run 2, strict objective)
- `output.log.run2-stronger-objective` — copy of run 2 for audit
- (Run 1's log was overwritten by run 2; the failure was identical in shape — agentic step returned a stub JSON without tool calls.)
