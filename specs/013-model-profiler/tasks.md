---
description: "Task list for Model Profiler (F2)"
---

# Tasks: Model Profiler (auto model-capability assessment)

**Input**: Design documents from `specs/013-model-profiler/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/profile-schema.md,
contracts/profiler-api.md, quickstart.md

**Tests**: INCLUDED and TEST-FIRST — Constitution III (NON-NEGOTIABLE) + SC-004/FR-016/17 require
the math be validated against known item parameters BEFORE implementation. Every story's test
wave runs and FAILS (red) before its implementation wave (green).

**Organization**: Tasks grouped by user story (US1 P1 MVP → US2 P2 → US3 P3). Wave tags `[W#]`
encode execution order; **impl waves run strictly after their test waves** (dev-kid orchestrate
collapses test→impl ordering, so the plan is regrouped after orchestration — see plan.md).

## Format: `[ID] [P?] [W#] [Story] [agent] [file] Description`

- **[P]**: parallelizable (different file, no incomplete deps within the wave)
- **[W#]**: wave (executes fully, in order)
- **[Story]**: US1 / US2 / US3 (none for Setup / Foundational / Polish)

## Path Conventions

Single-project library. New code under `promptchain/profiler/`; flat tests `tests/test_profiler_*.py`.

---

## Phase 1: Setup (Shared Infrastructure)

- [x] T001 [W1] [python-pro] [promptchain/profiler/] Create the `promptchain/profiler/` package: an `__init__.py` and empty module stubs `irt.py`, `cat.py`, `item_bank.py`, `scoring.py`, `composite.py`, `jacket.py`, `probe.py`, `store.py`, `prompt_generator.py` so `import promptchain.profiler` resolves; confirm `numpy`, `scipy`, `dspy` import in this env.

**Checkpoint**: package imports cleanly.

---

## Phase 2: Foundational (Blocking Prerequisites)

**⚠️ CRITICAL**: the data structures + item bank are used by ALL user stories.

### Tests (W2) — write FIRST, must FAIL

- [ ] T002 [P] [W2] [python-pro] [tests/test_profiler_item_bank.py] FAILING tests for `ProbeItem` validation (`a>0`, `0≤c<1`, known `dimension`), `ItemBank.synthetic(...)` producing items with KNOWN (a,b,c), `ItemBank.filter_quality(...)` (variance≥1%, accuracy≤95%, point-biserial≥0.1), `by_dimension(...)`, and that an uncalibrated/empty bank is flagged (`calibrated=False`).

### Implementation (W3)

- [ ] T003 [W3] [python-pro] [promptchain/profiler/item_bank.py] Implement `ProbeItem`, `ItemBank` (`synthetic`, `filter_quality`, `by_dimension`, `calibrated`) per data-model.md → T002 green.
- [ ] T004 [P] [W3] [python-pro] [promptchain/profiler/jacket.py] Implement the dataclasses `SkillEstimate`, `ProbeResponse`, `Jacket`, `CapabilityProfile` with `to_dict`/`from_dict` per data-model.md + contracts/profile-schema.md (defaults/nullable fields; `schema_version`). Round-trips exercised by US1/US2 tests downstream.

**Checkpoint**: structures + item bank ready.

---

## Phase 3: User Story 1 — Profile a model with a cheap offline probe (P1) 🎯 MVP

**Goal**: model id → persisted profile {per-skill θ̂, capability C, recommended tier, budget};
each of ~10 trials emitted as an F1 transcript carrying `model`.

**Independent Test**: run the probe end-to-end against a fake model → a profile is persisted with
per-skill θ̂/C/tier/budget; a repeat run is materially equivalent; trial count bounded by SE stop;
trial transcripts are F1-schema-valid and carry `model`.

### Tests (W4) — write FIRST, must FAIL

- [ ] T005 [P] [W4] [python-pro] [tests/test_profiler_irt.py] FAILING tests (FR-016): `prob_3pl` matches closed-form values; `fisher_information` matches closed form; `estimate_theta_eap` and `estimate_theta_wle` recover a KNOWN θ from synthetic responses to known-(a,b,c) items within tolerance; WLE used for all-correct/all-incorrect.
- [ ] T006 [P] [W4] [python-pro] [tests/test_profiler_cat.py] FAILING tests: `select_next_item` picks the max-Fisher-info unused item at θ̂; `standard_error` = `1/√(ΣI)`; `cat_should_stop` stops at SE≤τ (after `min_items`) and at `max_items` cap.
- [ ] T007 [P] [W4] [python-pro] [tests/test_profiler_scoring.py] FAILING tests: each per-dimension auto-scorer maps a sample model response → correct/incorrect (+ raw value) deterministically (instruction-following, tool-call validity, reasoning, format-sensitivity, degradation turn, effective context, latency).
- [ ] T008 [P] [W4] [python-pro] [tests/test_profiler_probe_integration.py] FAILING end-to-end test using an INJECTED fake `model_runner` (no live calls): `ModelProfiler.run_probe` produces a persisted `CapabilityProfile` (per-skill θ̂/C, `recommended_tier`, `budget_tokens`); a second run is materially equivalent (SC-002); item count bounded (SC-003); each trial wrote an F1-schema-valid transcript whose `model_call` lines carry `model` (SC-005, FR-007); uncalibrated bank raises.

### Implementation

- [ ] T009 [P] [W5] [python-pro] [promptchain/profiler/irt.py] Implement `prob_3pl`, `fisher_information`, `estimate_theta_eap` (EAP over a fixed θ grid, N(0,1) prior), `estimate_theta_wle`, `estimate_theta` (EAP→WLE at extremes) with numpy/scipy → T005 green.
- [ ] T010 [P] [W5] [python-pro] [promptchain/profiler/cat.py] Implement `select_next_item`, `standard_error`, `cat_should_stop` (τ=0.3, min/max items) → T006 green.
- [ ] T011 [P] [W5] [python-pro] [promptchain/profiler/scoring.py] Implement the per-dimension auto-scorers (deterministic; return correct/incorrect + raw) → T007 green.
- [ ] T012 [W6] [python-pro] [promptchain/profiler/store.py] Implement the profile store: atomic load/save of `~/.promptchain/model_profiles.json` (configurable base dir), idempotent upsert by `model_id`, `get_profile` (US1 portions only; refine/jacket_fit added in US3).
- [ ] T013 [W6] [python-pro] [promptchain/profiler/probe.py] Implement the probe harness `ModelProfiler.run_probe_async`/`run_probe`: drive the CAT loop, run each selected item as an ISOLATED PromptChain session with the F1 `TranscriptEmitter` attached (trial → F1 transcript carrying `model`), score it, update per-dimension θ̂/SE, stop at SE≤τ; compute capability C → `recommended_tier` + `budget_tokens` (US1); persist via `store`. `model_runner` injectable for tests. Raise on uncalibrated bank. (depends on T009–T012, T003, T004)
- [ ] T014 [W6] [python-pro] [promptchain/profiler/__init__.py] Export the public API (`ModelProfiler`, `CapabilityProfile`, `Jacket`, `SkillEstimate`, `run_probe` convenience) per contracts/profiler-api.md → T008 green. (depends on T013)

**Checkpoint**: MVP — `python -m pytest tests/test_profiler_irt.py tests/test_profiler_cat.py tests/test_profiler_scoring.py tests/test_profiler_probe_integration.py -q` green.

---

## Phase 4: User Story 2 — Composite Ω + jacket derivation (P2)

**Goal**: from θ̂ + calibration + cost → composite Ω → a jacket {tier, budget, mode, spawn,
compress@, max_turns, role, escalate}.

**Independent Test**: with known θ̂/K/F inputs, Ω matches the formula, jacket knobs fall in the
documented bands, escalation fires exactly when a condition holds.

### Tests (W7) — write FIRST, must FAIL

- [ ] T015 [P] [W7] [python-pro] [tests/test_profiler_composite.py] FAILING tests (FR-017): `capability`=σ(θ̂); `calibration_k`=1−ECE; `cost_penalty`=F formula; `omega`=0.7·C·K−0.3·F; `derive_jacket` maps each Ω band → the documented {tier, budget, mode, spawn=1−C, compress@}; escalate=True iff `P_route≥α OR Ω<0.25 OR SE>0.4`. Also `Jacket` to_dict/from_dict round-trip.

### Implementation

- [ ] T016 [W8] [python-pro] [promptchain/profiler/composite.py] Implement `capability`, `calibration_k`, `cost_penalty`, `omega`, `derive_jacket` (band table + escalation) per research D3 / data-model.md → T015 green.
- [ ] T017 [W8] [python-pro] [promptchain/profiler/probe.py] Wire `derive_jacket` into `run_probe` so the persisted profile carries `omega`/`calibration_k`/`cost_penalty_f`/`jacket` (additive to the US1 record; nulls tolerated when inputs absent). (depends on T016; same file as T013 → not [P])

**Checkpoint**: US1 + US2 — profile now includes Ω + jacket.

---

## Phase 5: User Story 3 — Two-sided fit + EWMA refine + DSPy jacket generator (P3)

**Goal**: fix a model, vary the jacket → Δθ lift → best jacket per model; EWMA-refine the profile
from telemetry; a DSPy `model_prompt_generator` compiles the jacket.

**Independent Test**: higher-Δθ jacket selected; a profile updated from new telemetry moves toward
the evidence by the EWMA amount; refine is a no-op with no telemetry.

### Tests (W9) — write FIRST, must FAIL

- [ ] T018 [P] [W9] [python-pro] [tests/test_profiler_refine.py] FAILING tests: `refine(model_id, session_metrics, lam)` EWMA-updates the stored estimate toward new evidence by the expected amount and is a no-op for empty metrics (FR-013); `jacket_fit(model_id, jackets, baseline)` computes Δθ per jacket vs baseline and returns the max-lift jacket (FR-012); graceful no-op/return-probe-jacket when SIO experiment/optimize unavailable.

### Implementation

- [ ] T019 [W10] [python-pro] [promptchain/profiler/store.py] Add `refine` (EWMA update; no-op on empty) and `jacket_fit` (two-sided Δθ vs baseline → best) to the store/`ModelProfiler` → T018 green. (depends on T012; same file as T012 → not [P])
- [ ] T020 [P] [W10] [python-pro] [promptchain/profiler/prompt_generator.py] Implement the DSPy `model_prompt_generator` module (per-model error/flow profile → jacket / optional system prompt), distinct from SIO's `suggestion_generator`; degrade gracefully (return the probe-derived jacket) when DSPy LM / SIO optimize is unavailable.

**Checkpoint**: all three stories functional.

---

## Phase 6: Polish & Cross-Cutting

- [ ] T021 [P] [W11] [python-pro] [promptchain/profiler/__init__.py] Final public-API review + module docstrings; ensure `refine`/`jacket_fit` exported.
- [ ] T022 [W11] [python-pro] [—] Run the full profiler suite `python -m pytest tests/test_profiler_*.py -q` green; then the OFFLINE LIVE SMOKE from quickstart.md against LAN ollama (`OLLAMA_API_BASE=http://192.168.0.159:11434`, `PYTHONPATH=<worktree>`, model `ollama/qwen3-coder:30b`) — a real cheap probe → persisted profile. Record the result.

---

## Dependencies & Execution Order

- **W1 Setup** → **W2 Foundational tests** → **W3 Foundational impl** → **W4 US1 tests** →
  **W5 US1 math impl** → **W6 US1 harness impl** → **W7 US2 tests** → **W8 US2 impl** →
  **W9 US3 tests** → **W10 US3 impl** → **W11 Polish**.
- Test waves (W2/W4/W7/W9) MUST be red before their impl waves (W3/W5+W6/W8/W10). This ordering is
  the post-orchestrate regroup (dev-kid orchestrate alone would merge test+impl into one wave).
- US1 (P1) is the MVP and is independently shippable after W6.

### Parallel opportunities

- W4: T005/T006/T007/T008 all [P] (separate test files).
- W5: T009/T010/T011 all [P] (irt.py / cat.py / scoring.py — separate files).
- W3: T003/T004 [P] (item_bank.py / jacket.py).
- W10: T019/T020 [P] (store.py / prompt_generator.py).

## Implementation Strategy

1. W1–W3 foundation. 2. W4→W6 = MVP (US1); STOP + validate. 3. W7–W8 add US2. 4. W9–W10 add US3.
5. W11 polish + live smoke. Commit once per wave (durable checkpoint).

## Notes

- `pytest-asyncio` NOT installed → async-driving tests use `def test_*` + `asyncio.run(...)`.
- Probe trials conform to the FROZEN F1 schema (additive optional fields only).
- The SIO-side CLI surfaces (`sio profile`, `--model` filters) are OUT OF SCOPE (ship in SIO repo).
