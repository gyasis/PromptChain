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
