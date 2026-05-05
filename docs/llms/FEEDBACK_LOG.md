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

## 2026-05-05 — Agentic step in hybrid chain returns stub JSON instead of calling registered tools

- **Symptom:** In the hybrid demo (`scripts/runs/2026-05-05_medgemma-clinical-demo/run.py`), the `AgenticStepProcessor` step's LLM (`openai/gpt-4o-mini`) returned `{}` (run 1) and then `{"labs": null, "history": null}` (run 2, after tightening the objective) without ever emitting `tool_calls`. The downstream medgemma synthesis still produced a plausible clinical report — but only because the case complaint was embedded in step 2's prompt; step 1's actual retrieval did nothing.
- **Root cause:** UNCONFIRMED. Hypothesis: local tool functions registered via `chain.register_tool_function()` on the parent `PromptChain` may not automatically propagate to the `AgenticStepProcessor`'s internal LLM calls under the v0.6.0+ `DynamicPromptGenerator` default. The agentic loop may need an explicit tool list OR the dynamic prompt builder may not be reading the parent's `local_tool_functions`.
- **Fix:** PARTIAL. Added anti-pattern #13 to `PROMPTCHAIN_FOR_LLMS.md` §9 ("don't assume parent-chain tools auto-propagate to AgenticStepProcessor"). Updated `recipe-hybrid-chain.md` with a "Known issue" callout. Real fix requires reading `agentic_step_processor.py` + `promptchain/prompts/dynamic.py` to confirm the propagation contract — pending follow-up. If confirmed broken, file a `gyasis/PromptChain` issue and update recipes with the workaround.
- **Source signal:** Direct observation while running `bash scripts/observe.sh runs/2026-05-05_medgemma-clinical-demo` during 2026-05-05 conversation. Both run outputs preserved in the run dir.

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
