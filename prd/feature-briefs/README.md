# Feature Briefs — Adaptive Prompting epic (the inputs to /speckit-specify)

One locked, self-contained brief per feature. Each is the document you **hand to
`/speckit-specify`** so it formalizes a tight spec instead of re-researching or guessing — the
PRD's per-feature scope + the design docs, distilled, with the already-made decisions pinned.

| Order | Brief | Goal | Depends on |
|---|---|---|---|
| 1 | `F1-sio-output-integration.md` | JSONL transcript emitter + SIO harness adapter (lock the schema) | — |
| 2 | `F2-model-profiler.md` | CAT/IRT probe → capability profile + jacket | F1 (transcript schema) |
| 3 | `F3-dynamic-prompt-layer.md` | per-model prompt assembly (tier/adapter/shim/budget/Document-&-Clear) | F2 (the jacket) |

**Build each fully before the next** (PRD §10): `/speckit-specify` (off the brief) → `/speckit-plan`
→ `/speckit-tasks` → dev-kid builds → merge to `epic/adaptive-prompting`. Then the next feature.

Source of truth: `prd/adaptive_prompting_system_prd.md` (§5/§6/§7) + `research/foundation/architecture/00–06`.
The briefs distill those; if a brief and the PRD ever disagree, the PRD wins.
