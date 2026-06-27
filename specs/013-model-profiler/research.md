# Phase 0 Research — Model Profiler

All method choices are **already decided** in the locked brief (`prd/feature-briefs/F2-model-
profiler.md`) and PRD §6 + design docs `01-model-profiler.md`, `02-model-profiler-math-and-sio.md`.
This file records the decisions, rationale, and the few implementation choices left to settle.
There were **no NEEDS CLARIFICATION** markers in the spec.

## D1 — Capability measurement: IRT 3PL scored by CAT (LOCKED)

- **Decision**: Model a probe item's success with the 3-parameter logistic model
  `P_i(θ) = c_i + (1−c_i)·σ(a_i(θ−b_i))`, `σ(x)=1/(1+e^{−x})`. Estimate model ability θ̂ via
  **EAP** (expected a posteriori over a θ grid with a N(0,1) prior); use **WLE** (Warm's
  weighted likelihood) as the fallback for all-correct / all-incorrect patterns where EAP is
  prior-dominated / ML is undefined. Select each next item by **maximum Fisher information**
  `I_i(θ̂)=a_i²(P_i−c_i)²(1−P_i)/[P_i(1−c_i)²]`. **Stop** when `SE(θ̂)=1/√(ΣI_i) ≤ τ`, τ=0.3.
  Capability `C=σ(θ̂) ∈ [0,1]`.
- **Rationale**: This is textbook Computerized Adaptive Testing. It recovers ability in ~10–30
  high-information items instead of a full benchmark (≥90% fewer calls) — the "cheap probe"
  requirement. The estimators have closed forms testable against synthetic known-parameter items
  (SC-004 / FR-016).
- **Alternatives considered**: Rasch/1PL (no discrimination/guessing — too coarse for varied LLM
  probes); raw accuracy averaging (the "hand-waved score" the brief explicitly rejects);
  Bradley-Terry/Elo (kept as a *cross-check* for pairwise jacket comparisons, not the primary
  ability scorer).

## D2 — Implementation libs for the math (settled here)

- **Decision**: Use **numpy** for vectorized P_i / I_i over the item array and the θ grid, and
  **scipy** only where it simplifies (e.g. `scipy.special.expit` for a stable σ, trapezoid
  integration for EAP). Both are already installed (numpy 2.4.4, scipy 1.17.1). Keep the EAP grid
  explicit (e.g. θ ∈ [−4, 4], 81 points) so the estimator is deterministic and unit-testable.
- **Rationale**: Deterministic, dependency-light, already present; an explicit grid makes EAP a
  pure function (reproducible profiles, SC-002). No new dependency added.
- **Alternatives**: a third-party IRT package (`girth`, `py-irt`) — rejected (heavier dep, less
  control over the WLE-fallback edge case, harder to pin determinism for tests).

## D3 — Composite Ω and the jacket bands (LOCKED)

- **Decision**: `K = 1 − ECE` (calibration confidence; the weighted
  `0.4(1−ECE)+0.35·SC+0.25·e^{−SemEntropy}` form is supported but the MVP uses `1−ECE` with the
  others optional), `F = (latency/latency_max)·(cost/cost_ref)`, **`Ω = 0.7·C·K − 0.3·F`**.
  Map (Ω, θ̂, SE) → jacket per the locked band table (lean/standard/rich/max-rich → tier, budget,
  mode, spawn=1−C, compress@). **Escalate** if `P_route ≥ α OR Ω < 0.25 OR SE > 0.4`.
- **Rationale**: One composite answers every downstream knob (single-shot vs heavy-loop vs
  escalate, spawn propensity, compression). The bands are the locked PRD mapping.
- **Alternatives**: per-knob independent heuristics — rejected (the brief locks the single
  composite; independent knobs drift and aren't jointly testable).

## D4 — Two-sided model×jacket fit (LOCKED, US3)

- **Decision**: Use the symmetric two-facet logit
  `logit P(success_i) = a_i·(θ_model + γ_jacket + δ_{model×jacket} − b_item)`. Jacket fit on a
  model = the ability lift `Δθ_{M,J} = (θ_M+γ_J+δ_{M,J}) − (θ_M+γ_base)` vs a baseline jacket on
  the same bank; ship the max-Δθ jacket per model. The (model×jacket) grid IS the experiment
  sweep — reuse `sio experiment` A/B + `sio optimize --optimizer gepa` (`SIO_TASK_LM=<target>`)
  to search jacket space, NOT a reimplementation.
- **Rationale**: IRT is symmetric (subjects↔items); the same bank + scorer measure both model
  ability and jacket fit. Reusing SIO's experiment/GEPA avoids rebuilding optimization.
- **Alternatives**: bespoke A/B harness — rejected (SIO already does cohort A/B with config-hash
  versioning; the brief mandates reuse).

## D5 — Continuous refinement: EWMA (LOCKED, US3)

- **Decision**: After each real session, update the stored per-skill estimate with an EWMA
  `x ← (1−λ)·x + λ·obs` (λ small, e.g. 0.2). "No new telemetry" → no-op.
- **Rationale**: Cheap, stable, sharpens the probe-seeded profile with use without re-probing.
- **Alternatives**: full re-probe each session (too expensive); Bayesian online IRT update
  (heavier; EWMA is the locked choice and adequate for the QoL refinement).

## D6 — Persistence location & shape (settled here)

- **Decision**: `~/.promptchain/model_profiles.json` — a single JSON object keyed by model id,
  one `CapabilityProfile` record per model (base dir configurable, mirrors F1's
  `~/.promptchain/transcripts/`). Upsert is idempotent by model id (re-probe overwrites that
  model's record; other records untouched).
- **Rationale**: Matches the brief ("`~/.promptchain/model_profiles.json` or a table") and the F1
  convention; a flat JSON keyed by model id is the simplest store F3 can read. Economy-first:
  re-probe is idempotent, no duplicate records.
- **Alternatives**: SQLite table — deferred (YAGNI for dozens of models; JSON is inspectable and
  F3-readable; a table can come later without changing the public API).

## D7 — Probe trials AS F1 transcripts (LOCKED — the F1↔F2 contract)

- **Decision**: Each probe trial runs as an isolated PromptChain run with the **F1
  `TranscriptEmitter` attached**, so the trial is written as a schema-valid F1 transcript whose
  `model_call` lines carry `model` (= `model_used` for SIO). The profiler reads model-attributed
  telemetry that SIO derives from these. The profiler MUST NOT add a required transcript field;
  any extra it wants (e.g. a probe-trial tag) is **additive + optional** at `schema_version: 1`.
- **Rationale**: This is the whole point of building F1 first — `sio mine --agent promptchain`
  ingests the probe runs and attributes them. Honors the frozen schema (additive-only).
- **Alternatives**: a separate bespoke probe log — rejected (defeats F1 reuse; SIO wouldn't
  ingest it).

## D8 — Async tests without pytest-asyncio (settled here)

- **Decision**: The repo has **no `pytest-asyncio`**; `@pytest.mark.asyncio` silently no-ops.
  New async-driving tests use plain `def test_*` + `asyncio.run(coro)`. The pure-math tests are
  synchronous anyway.
- **Rationale**: Documented gotcha from the F1 build; avoids silently-skipped async tests.
