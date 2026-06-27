# Component E — The Model Profiler ("inter-harness model analyst")

The unifying meta-layer. Constraints A–D each have a per-model knob (budget, tier, family-adapter,
compress-threshold, max-turns, subagent-spawn propensity, escalation). Today those would be guessed.
The **Model Profiler** *measures* a model and emits a **config profile (the "formula")** that drives all
of them — and keeps refining it from real session telemetry. PromptChain runs many models, so this is
the thing that makes "right-size the prompt to the model" **data-driven, not hand-tuned**.

Prior art (for a later smart-research pass, not deep-research): model routing / RouteLLM, model cards,
capability elicitation, adaptive inference, LLM-as-router. Adjacent in-house: SIO telemetry, `sio
experiment` cohorts, DSPy per-model compilation, PromptChain's token-accounting + MLflow observability.

## Inputs
- `model id` + optional published facts (context window, params, known benchmark scores).
- If facts are thin → run the probe trial (below). Otherwise seed from facts, then refine from telemetry.

## The probe: the "10-cast isolated trial"
When we lack benchmarks, run **~10 controlled, ISOLATED probe tasks** (each a fresh session, no shared
context — so results aren't contaminated). Each probe targets one dimension and is auto-scored:
1. **Instruction-following / structure** — honors XML/format, one-in_progress, stop conditions?
2. **Tool-call reliability** — valid calls, correct schema, parallelism, "never name tools"?
3. **Reasoning depth** — multi-hop planning quality on a small task.
4. **Degradation point** — run a multi-turn task; detect the turn where quality drops / it forgets / relapses.
5. **Effective context ceiling** — usable context (not advertised) before "lost in the middle".
6. **Format sensitivity** — does it break on JSON vs markdown vs XML?
7. **Latency / throughput** — from timing telemetry.
Collect every probe's history + outcome → score each dimension ∈ [0,1] (+ raw values for turns/ctx).

## The formula → the config profile (per model)
Let `cap = w·(instruction, tool, reasoning)` ∈ [0,1] (weighted capability score),
`deg` = measured degradation turn, `ctx_eff` = measured usable context.

| Knob (drives) | Formula |
|---|---|
| `prompt_tier` (B) | `extended` if `cap ≥ θ_tier` else `core` |
| `prompt_budget` tokens (A) | `clamp(0.03–0.05 × ctx_eff, 300, 1500)` |
| `format_adapter` (C) | from format-sensitivity + model family |
| `compress_threshold` ctx% (D) | `0.50 + 0.25·cap` → fragile models Document-&-Clear earlier |
| `max_turns_before_reset` (D) | `deg − 1` (reset just before the cliff; ≥10 is the goal) |
| `subagent_spawn_temp` | `1 − cap` → weaker models delegate MORE / take smaller task slices |
| `escalate_on_stall_turns` | small for weak (escalate sooner), large/∞ for strong |
| `role` | `planner` (big-brother) if `cap ≥ θ_planner`, else `executor`, else `both` |

So a strong model → extended prompt, big budget, low spawn-temp (does more itself), planner role,
late compression. A "gemma-12b" → core prompt, ~300–1k tokens, higher spawn-temp (slices work),
executor role, Document-&-Clear at ~55%, reset by ~turn 6–9, escalate on stall.

## Continuous collection (the "login" the user means)
Every real session logs outcome metrics (turns survived, tool-call error rate, completion success,
tokens, reset count) → the profile is updated with an **EWMA** so it sharpens with use. This is
telemetry-driven; reuse PromptChain's token accounting + activity-log JSONL + MLflow. SIO can mine it.

## Where it lives / reuse
- `ModelProfiler` component; profiles persisted (e.g. `~/.promptchain/model_profiles.json` or a table).
- The **foundation generator**, **ExternalLoop**, and **OrchestratorSupervisor** all READ the profile to
  set their knobs — so the profile is the single source of per-model config.
- Reuse: token accounting, activity logger, MLflow observer, two-tier routing, `sio experiment` cohorts.
- Optional: a profile can trigger a **DSPy/GEPA optimization** pass to compile a model-specific prompt.

## Net
The Profiler closes the loop: **measure → derive the formula → drive Constraints A–D → collect telemetry
→ refine.** It is the "inter-harness model analyst" — the brain that configures how each model is prompted,
looped, compressed, and delegated, from evidence instead of guesses. Feeds Phase 5 (a later build phase
after the foundation prompt + modular/loop/subagency land).
