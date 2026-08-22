# Feature Specification: Model Profiler (auto model-capability assessment)

**Feature Branch**: `013-model-profiler`
**Created**: 2026-06-27
**Status**: Draft
**Input**: User description: "Build the Model Profiler (F2 of the Adaptive Prompting epic): assess any model via a cheap adaptive probe → a persisted capability profile (per-skill ability + recommended prompt tier + best-fit 'jacket') that downstream prompt assembly reads. The math is established psychometrics (IRT 3PL + CAT), NOT a hand-waved score. It reuses SIO for telemetry/scoring/optimization and writes its probe runs AS F1 transcripts."

## Overview

The Model Profiler answers a single question that every other part of the adaptive-prompting
system needs answered: **how capable is this model, and how should the harness be shaped to fit
it?** Today that shaping (prompt tier, token budget, single-shot vs. heavy-loop, sub-agent
spawn propensity, when to compress, when to escalate) would be guessed. The Profiler *measures*
a model with a cheap adaptive probe and emits a **persisted capability profile + recommended
configuration ("jacket")** that downstream prompt assembly (F3) reads as the single source of
per-model configuration.

The measurement is **established psychometrics — Item Response Theory (3PL) scored via
Computerized Adaptive Testing (CAT)** — not a hand-waved heuristic. This makes "right-size the
prompt to the model" data-driven and reproducible.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Profile a model with a cheap offline probe (Priority: P1) — MVP

A user (or an upstream component) supplies a model identifier. The Profiler runs a short,
adaptive series of isolated probe trials against that model, scores each, and produces a
**persisted capability profile** containing, per skill dimension, an ability estimate (θ̂), an
overall capability score (C ∈ [0,1]), a recommended prompt tier, and a token budget. Re-running
the probe against the same model reproduces the same profile (within tolerance) and is cheap —
it stops as soon as the ability estimate is precise enough, typically far fewer trials than a
full benchmark.

**Why this priority**: This is the MVP. Without a persisted profile keyed by model id, nothing
downstream (F3, the foundation generator, the loops) has anything to read. The adaptive probe +
the IRT scoring + persistence is the irreducible core of the feature.

**Independent Test**: Given a model id and a calibrated item bank, run the probe end-to-end
against a real local model offline and confirm a profile is written to the profile store with a
per-skill θ̂, a capability C, a recommended tier, and a budget; confirm a second run produces a
materially equivalent profile; confirm the trial count is bounded by the precision stop rule.

**Acceptance Scenarios**:

1. **Given** a model id and a calibrated probe item bank, **When** the probe runs, **Then** a
   persisted profile is created containing per-skill ability estimates, a capability score, a
   recommended prompt tier, and a token budget.
2. **Given** a probe in progress, **When** the standard error of the ability estimate falls to
   or below the precision threshold, **Then** the probe stops administering further items.
3. **Given** the IRT/CAT scoring routines, **When** they are evaluated against synthetic items
   with known parameters and a known true ability, **Then** the recovered ability estimate
   matches the expected value within a stated tolerance.
4. **Given** a completed probe, **When** its trials are inspected, **Then** each of the ~10
   trials was recorded as a transcript that carries the model identifier and is ingestible by
   the downstream telemetry miner.

---

### User Story 2 - Derive the composite fit score and the jacket (Priority: P2)

From the per-skill ability estimates plus calibration confidence and a cost/latency penalty, the
Profiler computes a single **composite fit score (Ω)** and maps it (together with the ability
estimate and its precision) to a **jacket**: a concrete per-model configuration —
{prompt tier, token budget, execution mode (single-shot / single-shot+retry / heavy loop /
escalate-to-bigger-model), sub-agent spawn propensity, compression threshold, max turns before
reset, role}. The jacket includes an explicit escalation decision.

**Why this priority**: The raw ability estimate alone is not actionable; the jacket is the
artifact the rest of the harness consumes. It is P2 because US1 already delivers a usable
profile (tier + budget) for a first integration; the full composite + mode/spawn/compress
knobs sharpen it.

**Independent Test**: Given a profile with known θ̂, calibration, and cost inputs, confirm Ω is
computed by the stated formula, the jacket knobs fall in the documented bands for that Ω range,
and the escalation flag is set when any escalation condition is met.

**Acceptance Scenarios**:

1. **Given** a capability score, a calibration confidence, and a cost/latency penalty, **When**
   the composite is computed, **Then** Ω equals the stated weighted combination of those inputs.
2. **Given** a computed Ω, ability estimate, and precision, **When** the jacket is derived,
   **Then** the tier, budget, execution mode, spawn propensity, and compression threshold fall
   within the band documented for that Ω range.
3. **Given** an Ω below the escalation floor, OR a routing probability above the escalation
   threshold, OR a precision worse than the escalation ceiling, **When** the jacket is derived,
   **Then** the jacket's mode is "escalate to a bigger model".

---

### User Story 3 - Two-sided model×jacket fit and continuous refinement (Priority: P3)

Because the underlying model is symmetric in model-ability and jacket-quality, the same probe
bank can fix a model and vary the jacket to measure **which jacket fits which model** (the
interaction "lift", Δθ). The Profiler can sweep a model×jacket grid, pick the jacket with the
greatest lift per model, and — as real sessions accumulate telemetry — **continuously refine**
each model's profile with an exponentially-weighted update so the profile sharpens with use.

**Why this priority**: This is the optimization layer ("the harness fits the model" in its full
form) plus the feedback loop. It depends on US1 + US2 and on accumulated telemetry, so it is
lowest priority and the most reuse-heavy (it leans on the existing experiment/optimization
engine rather than new measurement code).

**Independent Test**: Given two jackets evaluated on the same model and bank, confirm the lift
(Δθ) is computed as the ability difference vs. a baseline jacket and the higher-lift jacket is
selected; given a profile and a stream of new session outcomes, confirm the exponentially-
weighted update moves the stored estimate toward the new evidence by the expected amount.

**Acceptance Scenarios**:

1. **Given** a model and two candidate jackets scored on the same bank, **When** fit is
   computed, **Then** the per-jacket lift Δθ is the ability difference relative to the baseline
   jacket and the jacket with the maximum lift is recommended for that model.
2. **Given** a stored profile and a new session outcome, **When** the refinement update runs,
   **Then** the stored estimate is updated by the exponentially-weighted rule toward the new
   evidence.

---

### Edge Cases

- **All-correct or all-incorrect probe run**: the standard ability estimator is undefined at the
  extremes; the Profiler MUST fall back to the weighted-likelihood estimator (or a bounded
  estimate) and still produce a finite profile.
- **Item bank not calibrated / empty**: the Profiler MUST refuse to produce a capability score
  from uncalibrated items and surface a clear, actionable error rather than emitting a bogus θ̂.
- **Model unreachable / probe trial errors**: a failed trial MUST be recorded (as a terminal
  transcript) and either retried or excluded; one failed trial MUST NOT corrupt the estimate or
  crash the whole probe.
- **Precision threshold never reached within the item budget**: the probe MUST stop at a hard
  maximum item count and emit the best estimate so far, flagged as low-precision.
- **No telemetry yet for a model (US3)**: refinement MUST be a no-op that leaves the probe-
  derived profile intact rather than degrading it.
- **Backfilling historical telemetry**: rows with a missing model identifier MUST be handled
  without double-counting once the model is later known.

## Requirements *(mandatory)*

### Functional Requirements

**Probe & capability (US1, P1)**

- **FR-001**: The system MUST run an adaptive probe over a calibrated item bank, selecting each
  next item by maximum information at the current ability estimate.
- **FR-002**: The system MUST estimate per-skill model ability (θ̂) and its standard error from
  probe responses using a 3-parameter item model, and MUST stop administering items once the
  standard error reaches or falls below the configured precision threshold (default 0.3).
- **FR-003**: The system MUST fall back to the weighted-likelihood ability estimator for all-
  correct / all-incorrect response patterns so that a finite estimate is always produced.
- **FR-004**: The system MUST derive a capability score C ∈ [0,1] from the ability estimate, and
  a recommended prompt tier and token budget from the measured capability and effective context.
- **FR-005**: The probe MUST cover the locked skill dimensions (instruction-following/structure,
  tool-call reliability, reasoning depth, degradation turn, effective context ceiling, format
  sensitivity, latency/throughput), with each trial auto-scored.
- **FR-006**: Each probe trial MUST be an **isolated session** (no shared context across trials)
  so trial results are uncontaminated.
- **FR-007**: Each probe trial MUST be emitted as a transcript conforming to the **frozen F1
  transcript schema** (one file per trial, carrying the model identifier on its model-call lines)
  so the telemetry miner ingests it with the model attributed. The Profiler MUST NOT introduce a
  new required transcript field; any new field it needs MUST be additive and optional.
- **FR-008**: The system MUST persist profiles keyed by model id to a profile store, and MUST
  produce a materially equivalent profile on a repeat probe of the same model (reproducibility).

**Composite & jacket (US2, P2)**

- **FR-009**: The system MUST compute the composite fit score Ω from the capability score, a
  calibration-confidence term, and a cost/latency penalty term, by the stated weighted formula.
- **FR-010**: The system MUST derive a jacket — {tier, token budget, execution mode, sub-agent
  spawn propensity, compression threshold, max-turns-before-reset, role} — from Ω, the ability
  estimate, and its precision, with knob values falling in the documented bands per Ω range.
- **FR-011**: The system MUST set the jacket's mode to "escalate to a bigger model" when the
  routing probability meets the escalation threshold, OR Ω is below the escalation floor, OR the
  precision is worse than the escalation ceiling.

**Two-sided fit & refinement (US3, P3)**

- **FR-012**: The system MUST be able to fix a model and vary the jacket over the same item bank
  to compute each jacket's fit as the ability lift (Δθ) relative to a baseline jacket, and MUST
  recommend the maximum-lift jacket per model.
- **FR-013**: The system MUST update a stored profile from new session telemetry using an
  exponentially-weighted moving update, and MUST treat "no new telemetry" as a no-op.

**Quality-of-life & telemetry attribution**

- **FR-014**: The system MUST expose a per-model scorecard summarizing a model's profile
  (per-skill θ̂, capability, recommended tier, jacket).
- **FR-015**: The system MUST support attributing telemetry to a model (a model filter / join on
  the existing telemetry), including backfilling records whose model attribution is missing,
  without double-counting.

### Math-correctness Requirements (test-first, Constitution III)

- **FR-016**: The 3-parameter item probability, the item information function, and the ability
  estimators (expected-a-posteriori and weighted-likelihood) MUST be validated by unit tests
  against synthetic items with known parameters: a known true ability MUST be recovered within a
  stated tolerance, and information/probability MUST match closed-form expected values.
- **FR-017**: The composite Ω and the jacket-band mapping MUST be validated by unit tests with
  known inputs producing the documented outputs.

### Key Entities

- **Probe item**: one calibrated probe task targeting one skill dimension; carries calibration
  parameters (discrimination, difficulty, guessing) and an auto-scoring rule.
- **Item bank**: the calibrated collection of probe items, calibrated once across many models and
  filtered for quality (sufficient variance, not near-ceiling accuracy, adequate discrimination).
- **Probe trial / response**: one administered item against the target model in an isolated
  session — its scored outcome plus raw values (e.g., degradation turn, usable context); also a
  transcript on disk.
- **Capability profile**: the persisted per-model record — per-skill ability estimates and
  precisions, capability score, composite Ω, recommended tier, budget, and the jacket; updated
  over time by refinement.
- **Jacket**: the per-model configuration derived from the profile — {tier, budget, execution
  mode, spawn propensity, compression threshold, max turns, role} (+ optional escalation flag).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Given a model id and a calibrated item bank, the Profiler produces a persisted
  profile containing per-skill ability, a capability score, a recommended tier, and a budget.
- **SC-002**: The probe runs **offline against a real local model, reproducibly** — a repeat run
  on the same model yields a materially equivalent profile.
- **SC-003**: The probe is **cheap**: it stops as soon as the precision threshold is met,
  typically on the order of 10–30 items rather than a full benchmark.
- **SC-004**: The math is **validated by unit tests against known item parameters** — synthetic
  items with known parameters recover the expected ability estimate within tolerance, and the
  item-probability / information / composite formulas match their closed-form expected values.
- **SC-005**: Probe runs **appear as transcripts that the telemetry miner ingests with the model
  attributed** (each trial transcript carries the model identifier and is schema-valid).
- **SC-006** (US2): A profile with known inputs yields the documented Ω and a jacket whose knobs
  fall in the documented bands, with escalation triggered exactly when an escalation condition
  holds.
- **SC-007** (US3): For a model with two candidate jackets, the higher-lift jacket is selected;
  and a profile updated from new telemetry moves toward the new evidence by the expected amount.

## Assumptions

- **A1**: F1's transcript schema is **frozen** (`specs/012-sio-output-integration/contracts/
  transcript-schema.md`). The Profiler conforms to it and only adds optional fields if needed.
- **A2**: A **calibrated item bank** is available (or a small seeded/synthetic bank suffices for
  the MVP); full multi-model calibration is a one-time, out-of-band activity and is not re-run by
  this feature each probe.
- **A3**: A **real local model is reachable offline** for the live probe (e.g., a LAN model
  endpoint); no paid/cloud credentials are required for the acceptance path.
- **A4**: The existing telemetry/scoring/optimization/experiment engine (SIO) is the reuse
  substrate; this feature does not reimplement telemetry storage, scoring, optimization search, or
  experiment cohorts.

## Dependencies

- **Depends on F1** (transcript schema + emitter) — the probe writes trials as F1 transcripts and
  reads model-attributed telemetry derived from them.
- **Blocks F3** (dynamic prompt assembly) — F3 reads the profile/jacket this feature produces.
- **Reuses SIO** — `session_metrics` / `error_records` / `flow_events`, experiment cohorts +
  config-hash versioning, and the optimization search; plus PromptChain token accounting and the
  activity-log JSONL.

## Out of Scope

- The dynamic prompt **assembly** that consumes the profile/jacket — that is **F3**.
- Re-implementing the telemetry store, the scoring engine, the optimization (GEPA) search, or the
  experiment-cohort machinery — these are **reused** from SIO, not rebuilt here.
- The SIO-side CLI surfaces (e.g. a `sio profile` subcommand, `--model` filters on SIO commands)
  ship in the **SIO repository**, not in this repository; this feature delivers only what
  PromptChain emits/derives (the probe harness, the IRT/CAT + Ω math, the jacket derivation, the
  per-model DSPy jacket-generator module, and the profile store).
- Modifying the existing loops, the foundation prompt, dev-kid, or the micro-agent fork.
