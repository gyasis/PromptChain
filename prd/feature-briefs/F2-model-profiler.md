# Feature Brief F2 — Model Profiler (auto model-capability assessment)

> **Purpose:** the locked, self-contained input for `/speckit-specify`. Everything below is DECIDED —
> formalize it, don't re-derive the math or re-open the approach. Source: PRD §6 · design
> `research/foundation/architecture/01-model-profiler.md` + `02-model-profiler-math-and-sio.md`.
> **Epic order: build AFTER F1 (consumes F1's transcript schema), BEFORE F3 (which reads F2's profile).**

## Hand to /speckit-specify (the framing)
"Build the Model Profiler: assess any model via a cheap adaptive probe → a persisted capability
profile (per-skill ability + recommended prompt tier + best-fit 'jacket') that downstream prompt
assembly reads. The math is established psychometrics (IRT 3PL + CAT), NOT a hand-waved score. It
reuses SIO for telemetry/scoring/optimization and writes its probe runs AS F1 transcripts."

## Context & dependencies
- **Depends on F1** — the profiler reads SIO `session_metrics` derived from PromptChain transcripts
  (`model_used` is the linchpin) AND its probe harness **writes its 10 trials as transcripts** so `sio mine` ingests them. F1's schema must be locked first.
- **Blocks F3** — F3's dynamic assembly reads the profile/jacket this produces.
- **Reuse (exists, do not rebuild):** SIO `session_metrics`/`error_records`/`flow_events` (`~/.sio/sio.db`), `sio experiment` cohorts + `config_hash`, `sio optimize --optimizer gepa` (`SIO_TASK_LM=<target>`), PromptChain token accounting + activity-log JSONL.

## Decisions ALREADY MADE (locked — do not re-research the math)
1. **Capability via IRT + CAT.** The "10-cast isolated trial" IS Computerized Adaptive Testing over an IRT bank.
   - **3PL item model:** `P_i(θ)=c_i+(1−c_i)·σ(a_i(θ−b_i))`.
   - **CAT loop:** pick next probe by max **Fisher information**; estimate `θ̂` via **EAP** (WLE for all-right/all-wrong); **stop when `SE(θ̂)≤τ`, τ=0.3** (~10–30 items).
   - **Capability `C=σ(θ̂) ∈ [0,1]`.** Pre-calibrate the item bank's (a,b,c) across many models ONCE (filter: variance≥1%, acc≤95%, point-biserial≥0.1).
2. **Probe dimensions (each an isolated fresh session, auto-scored):** instruction-following/structure, tool-call reliability, reasoning depth, degradation turn, effective context ceiling, format sensitivity, latency/throughput.
3. **Composite Ω → the jacket grid.** `K=1−ECE` (or the weighted calibration form), `F=(latency/latency_max)·(cost/cost_ref)`, **`Ω=0.7·C·K − 0.3·F`**. Ω/θ̂/SE map to {tier, budget, mode (single-shot vs heavy-loop vs escalate), spawn_temp=1−C, compress@}. Escalate if `P_route≥α OR Ω<0.25 OR SE>0.4`.
4. **Two-sided (model × jacket).** IRT is symmetric: fix model, vary jacket → measures **jacket fit** via the interaction term `δ_{model×jacket}`; ship the jacket with **max Δθ (lift) per model**. The (model×jacket) grid = the experiment sweep (`sio experiment` A/B + `sio optimize` GEPA search).
5. **The jacket** = per-model compiled config `{tier, budget, family_adapter, tool_mode, modules, compress@, max_turns, spawn_temp, role}` (+ optional GEPA-compiled per-model system prompt). Never hand-tuned — the measured best fit.
6. **Continuous refine:** every real session updates the profile via **EWMA**; optional GEPA re-compile.
7. **Persist** profiles to `~/.promptchain/model_profiles.json` (or a table). The foundation generator / ExternalLoop / OrchestratorSupervisor READ it (single source of per-model config).

## In scope (PRD §6 — "2 new builds + quality-of-life")
- **The probe harness** — runs the CAT/IRT 10-cast trial as isolated sessions, auto-scores each dimension, computes θ̂/Ω, emits each trial **as an F1 transcript** (so `sio mine` tags `model_used`). Reuse the loops where a probe is build-until-pass.
- **A `model_prompt_generator` DSPy module** — per-model error/flow profile → jacket (distinct from SIO's `suggestion_generator`).
- **Quality-of-life:** `sio profile --model X` scorecard; `--model` filter on `velocity`/`flows`/`suggest` (the JOIN); a `model` column/convention on experiments; backfill NULL `model_used` rows.

## Out of scope (do NOT let specify wander)
- The dynamic prompt **assembly** that USES the profile — that's **F3**.
- Re-implementing SIO's telemetry/scoring/GEPA/experiment — reuse them; only add the probe harness, the DSPy module, and the `--model` quality-of-life filters.
- Modifying the loops / foundation prompt / dev-kid / micro-agent fork.

## User stories (priority — refine in specify)
- **US1 (P1, MVP):** given a model id, run the ~10-cast offline probe against a real local model → a persisted profile {per-skill θ̂, capability C, recommended tier, budget}.
- **US2 (P2):** the composite Ω + jacket selection (tier/budget/mode/spawn/compress) from θ̂/K/F.
- **US3 (P3):** two-sided model×jacket fit (Δθ) via `sio experiment`/GEPA; EWMA refine from telemetry.

## Acceptance / success criteria (PRD §6)
- Given a model id → a **persisted profile** {per-skill θ, recommended tier, best jacket}.
- The ten-trial probe runs **offline against a real local model, reproducibly**.
- **Math validated by unit tests against known item parameters** (3PL P_i, Fisher info, EAP/WLE θ̂, Ω) — synthetic items with known (a,b,c) recover the expected θ̂ within tolerance.
- Probe runs appear as transcripts that `sio mine --agent promptchain` ingests with `model_used` set.

## Guardrails
- Test-first (Constitution III) — the math gets unit tests against known parameters before the harness.
- Probes are isolated (no shared context — results uncontaminated). Economy: CAT keeps probes cheap (stop at SE≤0.3).
