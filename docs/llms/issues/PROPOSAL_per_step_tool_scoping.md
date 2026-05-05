# Proposal: Per-step tool scoping in PromptChain

> **Draft GitHub issue body.** `gh` CLI auth was broken when this was drafted (`GH_TOKEN` invalid), so the proposal was committed here for the user to paste into a real issue at `https://github.com/gyasis/PromptChain/issues/new` once auth is restored.
>
> **Suggested title:** `Per-step tool scoping (so tool-weak specialist models can be used in mixed chains without seeing tools they shouldn't)`
> **Suggested labels:** `enhancement`, `api-design`, `multi-model`

---

## Problem

`PromptChain.add_tools(...)` registers tools at **chain scope**. Every LLM step in the chain is given the full tool list at completion time, regardless of whether that step actually needs tools.

This is fine for homogeneous chains (one model family throughout). It breaks for **mixed chains** that pair a tool-strong model (e.g. `openai/gpt-4o-mini`) for retrieval with a **tool-weak specialist** (e.g. `ollama/medgemma:4b`, code-specialised models, or any small instruction-tuned model that hasn't been heavily RLHF'd on tool refusal) for synthesis. The specialist sees tool schemas in its prompt context and tries to use them inappropriately — typically emitting hallucinated tool calls (`assessment`, `generate_report`, etc.) instead of producing the expected prose output.

## Reproduction

Live demo committed at `scripts/runs/2026-05-05_medgemma-clinical-demo/` and walked through in `docs/llms/FEEDBACK_LOG.md` (2026-05-05 entries).

Minimal repro:

```python
from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

chain = PromptChain(
    models=["ollama/medgemma:4b"],   # tool-weak specialist for synthesis
    instructions=[
        AgenticStepProcessor(
            objective="Use get_labs and get_history tools, then return JSON.",
            model_name="openai/gpt-4o-mini",
        ),
        "Synthesise this clinical case data: {input}",   # medgemma should not see tools
    ],
)
chain.register_tool_function(get_labs); chain.register_tool_function(get_history)
chain.add_tools([
    {...get_labs schema...},
    {...get_history schema...},
])
```

When step 2 runs, `litellm.acompletion()` is called with `tools=[get_labs, get_history]` even though step 2 is a plain synthesis prompt. medgemma sees the tools and tries to use them.

## Current workaround (works but isn't elegant)

Implement step 2 as a **Python function** that calls `litellm.acompletion(...)` directly without a `tools=` parameter. Function-step instructions don't see chain-scoped tools because they aren't LLM calls themselves.

```python
def medgemma_synthesize(case_json: str) -> str:
    resp = await litellm.acompletion(
        model="ollama/medgemma:4b",
        messages=[{"role": "user", "content": prompt_with(case_json)}],
        # No tools= — medgemma stays isolated.
    )
    return resp.choices[0].message.content

chain = PromptChain(models=[], instructions=[agentic_retrieval, medgemma_synthesize])
```

This works but breaks the "string instruction == clean LLM step" abstraction. The user pushed back on this approach — they want medgemma to be a first-class chain step.

## Proposed API

Allow each instruction to declare its tool requirement. Three sketches, in order of breaking-ness:

### Option A — tuple-based, additive

```python
chain = PromptChain(
    models=["ollama/medgemma:4b"],
    instructions=[
        (agentic_retrieval, ["get_labs", "get_history"]),   # only these tools
        ("Synthesise: {input}", []),                          # NO tools
    ],
)
```

Backward-compatible: if an instruction is a bare string/Callable/AgenticStepProcessor (no tuple), behave as today (chain-scoped tools).

### Option B — explicit kwarg on string instructions

Wrap string instructions in a small `Prompt` class:

```python
from promptchain import Prompt
chain = PromptChain(
    models=["ollama/medgemma:4b"],
    instructions=[
        agentic_retrieval,
        Prompt("Synthesise: {input}", tool_scope=[]),    # explicit empty
    ],
)
```

Cleaner type contract, slightly more breaking (introduces a new public class).

### Option C — declare on `AgenticStepProcessor` only

Just add `tool_scope: List[str]` to `AgenticStepProcessor.__init__`. Sequential string steps would need to explicitly pass through `Prompt(...)` (Option B-lite). Limits the scope of the change.

## Acceptance criteria

- [ ] When an instruction declares `tool_scope=[]`, `litellm.acompletion()` for that step is called WITHOUT a `tools=` parameter (verified via verbose log).
- [ ] When an instruction declares `tool_scope=["a", "b"]`, ONLY those two schemas are passed.
- [ ] When an instruction does NOT declare a scope, current behaviour is preserved (all chain-scoped tools visible).
- [ ] `AgenticStepProcessor` gets the same scoping (it already accepts `available_tools` at runtime — just needs to be filtered before the call).
- [ ] Existing examples in `examples/` continue to run unchanged.
- [ ] One new test in `tests/test_chain_with_tool_scoping.py` covers all three states above.
- [ ] One new example demonstrating the medgemma + gpt-4o-mini hybrid pattern, replacing the Python-function workaround in `scripts/runs/2026-05-05_medgemma-clinical-demo/`.

## Why this matters

Multi-model chains are the production pattern (cost optimization via two-tier routing, capability optimization via specialist + generalist pairing). Without per-step tool scoping, the framework forces users to choose between:

- All steps must use a tool-strong model (cost — gpt-4o for everything when gpt-4o-mini would do)
- Wrap weak-tool steps in Python functions (loses the abstraction the framework provides)
- Don't use specialists at all (loses domain expertise — medical, legal, code)

This API change unlocks the hybrid pattern as a first-class citizen.

## Related

- Discussion in `docs/llms/FEEDBACK_LOG.md` (2026-05-05 entries — the live debug session that surfaced this)
- Working pattern in `scripts/runs/2026-05-05_medgemma-clinical-demo/run.py` (current Python-function workaround)
- Doc anti-pattern #14 in `docs/llms/PROMPTCHAIN_FOR_LLMS.md §9`
