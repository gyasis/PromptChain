# Feedback Log

Audit trail of LLM mistakes observed when an agent uses PromptChain, and the doc / recipe / package fix that followed. Powers the closed-loop "agent makes mistake → user runs `sio scan` → user updates this and the relevant doc → next agent doesn't repeat the mistake".

Append entries newest-at-top.

## Format

```
## YYYY-MM-DD — <one-line title>
- **Symptom:** what the agent did wrong
- **Root cause:** why
- **Fix:** what was changed (file path + commit, or "added anti-pattern #N to PROMPTCHAIN_FOR_LLMS.md §9")
- **Source signal:** SIO query, transcript link, or "user-observed"
```

---

## 2026-05-05 — RESOLUTION + correction of the entry below

The original "tool propagation" hypothesis was **WRONG**. After reading `promptchain/utils/promptchaining.py` lines 469 (`add_tools`), 558 (`register_tool_function`), and 1244 (where `available_tools` is passed to `AgenticStepProcessor.run_async`), the actual contract is:

- **`register_tool_function(func)`** registers the IMPLEMENTATION (function callable) in `chain.local_tool_functions`.
- **`add_tools([schema_dict, ...])`** registers the OpenAI-format SCHEMA in `chain.local_tools`.
- **The LLM only sees what's in `chain.local_tools`.** If you only call `register_tool_function`, the function is dispatchable but the LLM has no schema for it → it sees zero tools → it shortcuts.
- v0.6.1 validates symmetrically (call register first for all funcs, then add_tools at end) — earlier versions were silent about the mismatch.

**The earlier failure was a documentation bug, not a library bug.** The LLM-targeted recipes (`recipe-agentic-step.md`, `recipe-hybrid-chain.md`, `recipe-tool-calling-local.md`) all said "just call `register_tool_function`" and skipped `add_tools`. That's the bug — in the docs I wrote, not in the package. **Anti-pattern #13 has been corrected** to reflect this in `PROMPTCHAIN_FOR_LLMS.md §9`.

**Meta-lesson logged separately:** I jumped to "library bug" too fast. Must always rule out pipeline/syntax errors in MY code before filing a library issue. (See "Meta-lesson" entry below.)

After the schema fix, the agentic step DID call both tools and returned proper structured JSON. The chain ran end-to-end successfully (run preserved in the run dir).

---

## 2026-05-05 — Real library gap: PromptChain has no per-step tool scoping

- **Symptom:** Once tools were correctly registered (see entry above), the hybrid chain progressed to step 2 (medgemma:4b synthesis). medgemma is a tool-WEAK specialist medical model. With chain-scoped tools still visible to its prompt, it tried to emit a hallucinated `assessment` tool call instead of producing a prose clinical report.
- **Root cause:** **PromptChain `add_tools()` registers tools at chain scope, not step scope.** All registered tools are visible to every LLM step in the chain. For multi-model chains where some models are tool-weak specialists (medgemma, code-focused models, etc.), there is no way to say "step 2 should NOT see the tools that step 1 needed."
- **Fix (workaround in demo):** Implemented step 2 as a Python function (`medgemma_synthesize`) that calls medgemma directly via `litellm.acompletion(...)` WITHOUT a `tools=` parameter. Function steps don't see chain-scoped tools, so medgemma stays isolated. Demo now runs clean: gpt-4o-mini retrieves via tools (step 1), medgemma synthesises a proper STEMI assessment (step 2).
- **Real fix (proposed for the package):** Add `step_tool_scope` (or similar) to `PromptChain` so each instruction can declare which tools it needs/allows. e.g. `instructions=[(agentic_step, ["get_labs", "get_history"]), (synthesis_prompt, [])]`. File as a `gyasis/PromptChain` issue.
- **Source signal:** Live demo at `scripts/runs/2026-05-05_medgemma-clinical-demo/`. Working version preserved in git history.

---

## 2026-05-05 — META-LESSON: pipeline errors vs library bugs

When a chain misbehaves, the agent MUST distinguish:
- **(a) Pipeline / syntax / call-order errors in MY code** — wrong import path, missed config call, bad prompt structure, model that can't do the task. These are 90%+ of perceived "bugs."
- **(b) Real library bugs** — the framework's contracts are documented and I followed them, but the behaviour doesn't match. These are rare.

**The agent must rule out (a) before claiming (b).** Steps:
1. Read the actual API contract in source (don't trust your own recipes — they may be wrong).
2. Look at canonical examples in `examples/` to see how the library is *actually* used.
3. Check verbose logs to see what the LLM was actually given (was it actually given the tools? what was in the prompt?).
4. Only after 1-3 rule out misuse should you investigate the library.

**Concrete failures from today:**
- "Tool propagation broken" → was actually missing `add_tools()` call → MY pipeline error
- "medgemma emits weird JSON" → was actually overcomplicated prompt → MY pipeline error
- "medgemma emits fake tool call" → was chain-scoped tool exposure on a tool-weak model → BORDERLINE (workaround possible; per-step tool-scoping is a real package gap)

I claimed (b) for all three at first. The first two were (a). Only the third is genuinely (b).

This entry exists so future-me does not make the same shortcut.

---

## 2026-05-05 — Original (now corrected) entry — kept for the audit trail

(Below is the original entry. Reading it after the resolution above is instructive — it shows the agent jumping to a wrong conclusion about a "library bug" when the actual cause was a missing `add_tools()` call in the demo and in the recipes.)

- **Symptom:** In the hybrid demo (`scripts/runs/2026-05-05_medgemma-clinical-demo/run.py`), the `AgenticStepProcessor` step's LLM (`openai/gpt-4o-mini`) returned `{}` (run 1) and then `{"labs": null, "history": null}` (run 2, after tightening the objective) without ever emitting `tool_calls`. The downstream medgemma synthesis still produced a plausible clinical report — but only because the case complaint was embedded in step 2's prompt; step 1's actual retrieval did nothing.
- **Root cause (CLAIMED, WRONG):** Hypothesis: local tool functions registered via `chain.register_tool_function()` on the parent `PromptChain` may not automatically propagate to the `AgenticStepProcessor`'s internal LLM calls under the v0.6.0+ `DynamicPromptGenerator` default.
- **ACTUAL root cause:** I forgot to call `chain.add_tools(schemas)`. Without it, the LLM has no tool schema to consult — even though the function is registered as dispatchable. Anti-pattern #13 corrected.
- **Source signal:** Direct observation while running `bash scripts/observe.sh runs/2026-05-05_medgemma-clinical-demo` during 2026-05-05 conversation.

## 2026-05-05 — Bootstrap entry (no real failures yet)

- **Symptom:** None — this is the seed entry created when the LLM-usability layer was built.
- **Root cause:** N/A.
- **Fix:** Established `docs/llms/PROMPTCHAIN_FOR_LLMS.md` (Layer 1), `docs/llms/recipes/` (Layer 2), `~/.claude/skills/promptchain.md` (Layer 3). Anti-pattern catalog in `PROMPTCHAIN_FOR_LLMS.md` §9 is pre-seeded with 12 mistakes the *author* anticipated; real entries here will refine or replace those.
- **Source signal:** N/A — bootstrap.

> **Process for the next entry:**
> 1. Claude Code writes some PromptChain code, makes a mistake.
> 2. User runs `sio scan` to surface the failure pattern.
> 3. User picks the most-impactful finding.
> 4. User updates `PROMPTCHAIN_FOR_LLMS.md` §9 (anti-patterns), or adds/edits a recipe, or files a real `gyasis/PromptChain` issue if the package itself is the problem.
> 5. User adds an entry to this log with what changed.
