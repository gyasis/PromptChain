---
id: recipe-hybrid-chain
title: Hybrid chain — agentic step (open-ended) + sequential prompts (deterministic)
when: Part of the work is open-ended (gather data, can't pre-script the steps) and part is deterministic (synthesise the gathered data into a structured output). Mix both modes in one PromptChain.
api_version: 0.6.1+
---

# Hybrid Chain

The most common production pattern. Two operating modes inside ONE chain:

- **Step 1 — Agentic** (`AgenticStepProcessor`) for open-ended work where the model decides which tools to call and when. Needs a tool-calling-capable model.
- **Step 2 — Sequential prompt** for deterministic synthesis. Any model. Often a cheaper / specialised one.

The agentic step's *only output* (its `final_answer`) is the input to step 2. The agentic step's internal reasoning history is **isolated** — it does NOT pollute step 2's context. (See `agentic_step_processor.py:142-199` for the rationale.)

## Pattern

```python
import asyncio
import json
from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


# ---- domain tools (called by the agentic step) ----

def get_labs(patient_id: str) -> str:
    """Fetch latest labs for a patient_id. Returns JSON."""
    # Real impl would query an EHR
    return json.dumps({
        "patient_id": patient_id,
        "troponin_I": 4.2,           # ng/mL — elevated
        "creatinine": 1.0,
        "potassium": 4.1,
        "INR": 1.0,
    })


def get_history(patient_id: str) -> str:
    """Fetch relevant medical history for a patient_id. Returns JSON."""
    return json.dumps({
        "patient_id": patient_id,
        "age": 45,
        "sex": "M",
        "history": ["hypertension", "smoker"],
        "medications": ["lisinopril 20mg daily"],
        "allergies": ["NKDA"],
    })


async def main():
    chain = PromptChain(
        models=["ollama/medgemma:4b"],     # ONE model for the ONE string instruction below
        instructions=[
            # Step 1 — AGENTIC: gpt-4o-mini autonomously calls retrieval tools
            AgenticStepProcessor(
                objective=(
                    "For patient_id mentioned in the input, retrieve labs AND history via the available tools. "
                    "Return ONE JSON object combining vitals, labs, and history. No commentary."
                ),
                model_name="openai/gpt-4o-mini",   # tool-calling capable; cheap
                max_internal_steps=6,
                history_mode="progressive",
            ),

            # Step 2 — SEQUENTIAL: medgemma synthesis with EXPLICIT EMPTY TOOL SCOPE.
            # The (instruction, []) tuple form tells PromptChain: this step
            # gets ZERO tools. Isolates medgemma from tool schemas it would
            # otherwise try to use inappropriately. See PROMPTCHAIN_FOR_LLMS.md §17.
            (
                (
                    "You are a clinical reasoning assistant. Given the structured case data below, produce:\n"
                    "1. Primary diagnosis with reasoning.\n"
                    "2. Top 3 differentials.\n"
                    "3. Immediate management plan WITH doses (mg, route, frequency).\n"
                    "4. For every drug dose listed, append [VERIFY] if you are not 100% sure of the standard adult dose.\n\n"
                    "Case data (from upstream retrieval step): {input}"
                ),
                [],   # ← EMPTY tool scope — medgemma sees zero tools
            ),
        ],
        verbose=True,
    )

    # Register tools — both visible to step 1 (default scope), neither to step 2 (scoped empty)
    chain.register_tool_function(get_labs)
    chain.register_tool_function(get_history)
    chain.add_tools([...])  # schemas for both — see recipe-tool-calling-local.md for the format

    # The user-facing question only mentions the patient_id and presentation;
    # the agentic step is responsible for fetching the rest.
    user_input = (
        "Patient_id=PT-1234 presents to the ED with crushing substernal chest pain "
        "radiating to the left arm, diaphoresis, SOB. ECG: ST-elevation in II, III, aVF. "
        "Pull labs + history and produce a clinical reasoning report."
    )
    result = await chain.process_prompt_async(user_input)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

## Why this is the right shape

- **Open-ended retrieval** (which patient? which tools? in what order?) → AGENTIC. The model figures out it needs `get_history(PT-1234)` AND `get_labs(PT-1234)` and calls both — possibly in parallel.
- **Deterministic synthesis** (always: dx, differentials, plan, doses) → SEQUENTIAL prompt. Same shape every time. Cheaper local model is enough.
- **Token economy** — the agentic step's internal history is isolated; medgemma sees only the structured JSON, not the agent's tool-call traces. This is the multi-agent isolation pattern (§3 in `recipe-agentic-step.md`) applied to a single chain.
- **Right model for each job** — `gpt-4o-mini` for tool calling (reliable schemas), `medgemma:4b` for clinical reasoning (specialist). Neither could play both roles well.

## Running with observability

```bash
bash scripts/observe.sh runs/2026-05-05_medgemma-clinical-demo
```

MLflow captures: each LLM call, tool invocations, latency per step. SIO can mine the run log later to find recurring failure modes.

## Two gotchas surfaced by the live demo, both now resolved

1. **You MUST call `chain.add_tools([schemas])` AS WELL AS `chain.register_tool_function(func)`.** Without the schema, the LLM has no idea the tool exists → agentic step returns `{}` or `{"key": null}`. See `recipe-tool-calling-local.md` for the two-call pattern. v0.6.1's symmetric validator will raise if call order is wrong (register all funcs first, then add_tools once).
2. **Tool-weak specialist models (medgemma, similar) in synthesis step would previously hallucinate tool calls because tools were chain-scoped.** **FIXED 2026-05-06**: per-step tool scoping landed. Wrap the synthesis instruction as `(prompt, [])` and the framework filters tools to an empty list before the LLM call — no Python-function workaround needed. See §17 in `PROMPTCHAIN_FOR_LLMS.md`.

Use the updated pattern below — the demo at `scripts/runs/2026-05-05_medgemma-clinical-demo/run.py` shows it working end-to-end.

## Common failures + fix

- **Agentic step's model isn't tool-calling-capable** — silently degrades to plain text answer. Switch `model_name=` to one of the ⭐⭐⭐ entries in `recipe-models-and-tool-calling.md` §B.
- **Step 2's prompt doesn't reference `{input}`** — medgemma synthesises from nothing. Always include `{input}` placeholder where you want the upstream output.
- **Adding the synthesis model into `models=[...]` BUT also using a string instruction for it** — count carefully. ONE string instruction needs ONE model slot. The AgenticStepProcessor does NOT consume a model slot.
- **Agentic step takes too long** — the loop is iterating. Lower `max_internal_steps`, or switch from `history_mode="progressive"` to `"minimal"`, or enable Phase 2 `enable_blackboard=True` (see `recipe-advanced-agentic.md`).
- **Synthesis model hallucinates a dose** — that's why the prompt asks for `[VERIFY]` flags. For higher safety, wrap step 2 in its own `AgenticStepProcessor` with `enable_cove=True` — CoVe will second-check each claim before commit.

## Variant — full safety stack

For high-stakes domains (clinical, legal, financial), wrap the deterministic synthesis in an agentic step too, with CoVe on:

```python
synthesis_with_cove = AgenticStepProcessor(
    objective="Produce dx + management plan with doses. Verify every claim before final answer.",
    model_name="ollama/medgemma:4b",       # not tool-calling — but agentic loop here is just for CoVe
    enable_cove=True,
    cove_confidence_threshold=0.75,
)
```

Note: medgemma:4b doesn't support tool-calling, so the agentic loop here can't use external verification tools — but CoVe's *self-verification* sub-prompt still works (it just uses the same model to challenge its own output before commit).
