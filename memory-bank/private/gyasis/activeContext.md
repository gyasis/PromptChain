# Active Context

**Last Updated**: 2026-06-27 15:43:59

## Current Focus
**F3 (Dynamic Prompt Layer / 014) — COMPLETE & GREEN.** Built via spec-kit → dev-kid, test-first,
24/24 tasks across 11 waves + 1 corrective wave. Tests: **F3 77, F1+F2 88, all 165 pass, no regression.**

### Per-US deliverables (`promptchain/prompts/`)
- **US1 (P1, MVP):** `DynamicModelPromptGenerator` (`model_dynamic.py`) — a `BasePromptBuilder`
  drop-in that reads the F2 profile/jacket via an injected/default store, picks a CORE/EXTENDED/TINY
  tier (`tiers.py`), applies a family adapter (`family.py`), and trims to budget (`budget.py`).
  Per-model differentiated, budget-compliant, deterministic. Null/missing jacket → fallback.
- **US2 (P2):** toolshim (`toolshim.py`) — `<tools>` JSON-in-text + plain-text history for shim
  models; native passthrough. **One additive F2 field: `Jacket.tool_mode`** (optional, F2 66 green).
- **US3 (P3):** `longevity.py` — `<turn-context>` + goal re-injection; `DocumentAndClear`
  (write progress doc at ~60%, clear+resume, ≥10 turns, escalate-on-stall respecting jacket, lossy
  fallback).
- **A/B eval (SC-007):** `eval_ab.py` (direct-answer task set + run_ab) + `scripts/ab_smoke.py`.

### Key decisions (the hard-won ones — see spec.md FR-004/SC-003/SC-007)
- **Static base verbatim (SC-003) holds for CORE/EXTENDED ONLY.** **TINY swaps to a short
  `TINY_BASE_PROMPT`** (Goose tiny_model_system). Why: the live smoke caught a REAL regression —
  additive-only F3 over the full 748-tok foundation made `llama3.2:1b` WORSE (−40% to −60%); the
  foundation is at a 1B model's ceiling, so the only way to help is a *simpler* prompt.
- **family.adapt_format is non-prescriptive identity** — an earlier per-family `FORMAT:` preamble
  made the 1B model echo prompt structure in ALL-CAPS instead of solving. Never dictate output format.
- **Native TINY omits the tool inventory** — a weak model reaches for advertised tools and fails.
- **The eval is direct-answer framed** — "write-and-run" is an invalid single-shot proxy.
- **SC-007 result** (offline live smoke vs LAN `ollama/llama3.2:1b`): **f3 40% vs static_base 30%
  (+10%; diagnostic +20%)**. Magnitude noisy (N=5×1B) but sign consistent; qualitative win robust
  (f3 answers on-task, base echoes `<work_loop>`/`<safety>`).

### Next
Merge `014-dynamic-prompt-layer` into `epic/adaptive-prompting` (`--no-ff`). With F1+F2+F3 all green,
the epic→main merge can then be considered.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

## Recent Changes
```
 .claude/activity_stream.md | 1 +
 1 file changed, 1 insertion(+)
```

## Modified Files
.claude/activity_stream.md
memory-bank/private/gyasis/activeContext.md

## Next Actions
- Continue implementation
- Run tests
- Create checkpoint
