"""
Hybrid PromptChain demo — agentic retrieval (gpt-4o-mini) + sequential
synthesis (ollama/medgemma:4b).

Pattern documented in:
  docs/llms/recipes/recipe-hybrid-chain.md
  docs/llms/PROMPTCHAIN_FOR_LLMS.md §4 (two operating modes)

Run:
  bash scripts/observe.sh runs/2026-05-05_medgemma-clinical-demo

What it proves:
  - Sequential prompt + agentic step coexist in ONE PromptChain
  - Agentic step has its own model_name (gpt-4o-mini) and tool surface
  - Sequential step uses a different model (local medgemma:4b)
  - Agentic step's internal reasoning history is isolated from the
    synthesis step (only its final_answer is passed forward)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Make this script runnable from anywhere — add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.observability import init_mlflow

init_mlflow()


# ---------- domain tools (the agentic step calls these) ----------

def get_labs(patient_id: str) -> str:
    """Fetch latest labs for a patient_id. Returns JSON."""
    fixtures = {
        "PT-1234": {
            "patient_id": "PT-1234",
            "troponin_I_ng_per_mL": 4.2,
            "creatinine_mg_per_dL": 1.0,
            "potassium_mEq_per_L": 4.1,
            "INR": 1.0,
            "drawn_at": "2026-05-05T12:30:00Z",
        }
    }
    return json.dumps(fixtures.get(patient_id, {"error": f"no labs for {patient_id}"}))


def get_history(patient_id: str) -> str:
    """Fetch relevant medical history for a patient_id. Returns JSON."""
    fixtures = {
        "PT-1234": {
            "patient_id": "PT-1234",
            "age": 45,
            "sex": "M",
            "history": ["hypertension", "active smoker", "hyperlipidemia"],
            "medications": ["lisinopril 20 mg daily", "atorvastatin 40 mg nightly"],
            "allergies": ["NKDA"],
        }
    }
    return json.dumps(fixtures.get(patient_id, {"error": f"no history for {patient_id}"}))


# ---------- the chain ----------

async def main() -> str:
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set — agentic step needs it.", file=sys.stderr)
        sys.exit(2)

    chain = PromptChain(
        # ONE model slot — for the ONE string instruction in step 2.
        # The AgenticStepProcessor brings its own model_name; no slot needed.
        models=["ollama/medgemma:4b"],
        instructions=[
            # Step 1 — AGENTIC: open-ended retrieval
            # NOTE: Objective is INTENTIONALLY explicit about MUST-call-tools because
            # an earlier version with looser wording caused gpt-4o-mini to short-circuit
            # to `{}` without calling either tool. See docs/llms/FEEDBACK_LOG.md.
            AgenticStepProcessor(
                objective=(
                    "You have access to two tools: get_labs(patient_id) and "
                    "get_history(patient_id). For the patient_id mentioned in the input, "
                    "you MUST call BOTH get_labs AND get_history. After both tool results "
                    "are in, combine them into ONE JSON object with the patient's labs and "
                    "history merged. ONLY THEN return the combined JSON as your final answer. "
                    "DO NOT produce a final answer before both tool calls have completed."
                ),
                model_name="openai/gpt-4o-mini",   # tool-calling capable
                max_internal_steps=6,
                history_mode="progressive",
            ),
            # Step 2 — SEQUENTIAL: deterministic clinical synthesis
            (
                "You are a clinical reasoning assistant. Given the structured case data "
                "(JSON) below and the original presenting complaint, produce a markdown "
                "report with EXACTLY these sections:\n\n"
                "1. **Primary diagnosis** — single sentence + 2-3 sentences of reasoning\n"
                "2. **Top 3 differentials** — bullet list, one line each\n"
                "3. **Immediate management plan** — numbered list with doses (mg, route, frequency)\n"
                "4. **Time-sensitive intervention** — one paragraph\n\n"
                "For ANY drug dose you list, append `[VERIFY]` if you are not 100% sure of "
                "the standard adult dose. This safety flag is required.\n\n"
                "Original complaint: 45 y/o M, crushing substernal chest pain radiating to "
                "left arm + jaw, diaphoresis, SOB, ECG ST-elevation in II/III/aVF.\n\n"
                "Structured case data (from upstream retrieval): {input}"
            ),
        ],
        verbose=True,
    )

    chain.register_tool_function(get_labs)
    chain.register_tool_function(get_history)

    user_input = (
        "Patient_id=PT-1234 — pull labs and history, then produce a clinical reasoning report."
    )
    result = await chain.process_prompt_async(user_input)
    print("\n" + "=" * 70)
    print("FINAL OUTPUT")
    print("=" * 70)
    print(result)
    return result


if __name__ == "__main__":
    asyncio.run(main())
