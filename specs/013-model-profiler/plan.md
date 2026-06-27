# Implementation Plan: Model Profiler (auto model-capability assessment)

**Branch**: `013-model-profiler` | **Date**: 2026-06-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/013-model-profiler/spec.md`

## Summary

Add a self-contained `promptchain/profiler/` package that assesses a model with a cheap adaptive
probe and emits a **persisted per-model capability profile + jacket** that downstream prompt
assembly (F3) reads. The measurement is established psychometrics: a 3-parameter (3PL) Item
Response Theory item model, scored by Computerized Adaptive Testing (CAT) — pick the next probe
by maximum Fisher information, estimate ability θ̂ by EAP (with a WLE fallback at the all-right /
all-wrong extremes), and stop when the standard error SE(θ̂) ≤ τ (default 0.3). From θ̂ the
profiler derives capability `C = σ(θ̂)`, a composite fit score `Ω = 0.7·C·K − 0.3·F`, and a
**jacket** (per-model config: tier, budget, mode, spawn propensity, compress threshold, max
turns, role, escalation). The probe harness runs each trial as an **isolated PromptChain
session** and emits it as an **F1 transcript** (conforming to the frozen schema) so `sio mine`
ingests it with `model_used` set. Profiles persist to `~/.promptchain/model_profiles.json` and
refine over time via EWMA from telemetry; a DSPy `model_prompt_generator` module compiles the
jacket. **SIO-side CLI surfaces (`sio profile`, `--model` filters) ship in the SIO repo and are
out of scope here** — this feature delivers only what PromptChain emits/derives.

## Technical Context

**Language/Version**: Python 3.10+ (repo CI runs 3.10 and 3.12; current active 3.12.11)
**Primary Dependencies**: `numpy` (2.4.x, IRT/CAT vector math) + `scipy` (1.17.x, the EAP
quadrature / optimization helpers) — both already installed; `dspy` (3.2.x, already installed)
for the `model_prompt_generator` module only; `promptchain` itself (PromptChain runs the probe
sessions) and `promptchain.observability.transcript_emitter` (F1) to emit trial transcripts.
NO `sio` import (SIO is a downstream consumer, reached only via the transcripts/DB it reads).
**Storage**: `~/.promptchain/model_profiles.json` (one record per model id; base dir
configurable, mirrors F1's `~/.promptchain/transcripts/` convention). Probe trial transcripts
land in the F1 transcript dir.
**Testing**: pytest, flat `tests/test_profiler_*.py` (mirrors F1's flat `tests/test_transcript_*`
layout — NOT contract/integration/unit subdirs). `pytest-asyncio` is NOT installed; async-driving
tests use plain `def test_*` + `asyncio.run(...)`.
**Target Platform**: Linux/macOS — a cross-platform Python library.
**Project Type**: single (library).
**Performance Goals**: the probe stops at SE ≤ 0.3 → typically 10–30 items (≥90% fewer than a
full benchmark); pure-math routines vectorized so a probe's compute cost is dominated by the
model calls, not the scoring.
**Constraints**: probe trials are isolated (no shared context); offline against a real local
model (LAN ollama) with no paid credentials on the acceptance path; math is deterministic +
unit-tested against known parameters; conforms to F1's FROZEN transcript schema (additive
optional fields only); economy-first (CAT keeps probes cheap; profiles content-keyed by model id,
idempotent re-probe).
**Scale/Scope**: dozens of models; one profile record per model; a small calibrated/seeded item
bank (a synthetic bank with known parameters suffices for the MVP + the math tests).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | How this feature satisfies it |
|---|---|---|
| I. Library-First | ✅ | Self-contained `promptchain/profiler/` package; pure-math core (`irt`, `cat`, `composite`) independently testable with zero model calls; clear single purpose. |
| II. Observable Systems | ✅ | Each probe trial is emitted as an F1 transcript (structured JSONL); the profile + jacket are inspectable artifacts; reuses the existing event/telemetry surface. |
| III. Test-First (NON-NEGOTIABLE) | ✅ | The math (`irt`, `cat`, `composite`) gets unit tests against KNOWN item parameters FIRST (red), then implementation (green) — this IS an explicit acceptance criterion (SC-004 / FR-016/17). Harness + store likewise test-first. |
| IV. Integration Testing | ✅ | An integration test runs the probe end-to-end (against a stub/fake model) and asserts trial transcripts are F1-schema-valid + carry `model`, and a profile is persisted + reproducible — the F1↔F2 contract boundary. |
| V. Token Economy & Performance | ✅ | CAT stops at SE ≤ τ (fewest items); isolated cheap probes; idempotent re-probe (no needless re-measurement); vectorized math. |
| VI. Async-First Design | ✅ | The probe harness drives async PromptChain runs; public API offers sync wrappers over the async core (matching the repo's dual-interface pattern), using `asyncio.run` only when not already in a loop. |
| VII. Simplicity & Maintainability | ✅ | Pure-math modules with no I/O; the DSPy jacket-generator + GEPA/experiment reuse are isolated behind US3 (P3) and degrade gracefully if absent; no SIO reimplementation; YAGNI on the calibration pipeline (synthetic bank for MVP). |

**Result**: PASS — no violations. Complexity Tracking left empty.

## Project Structure

### Documentation (this feature)

```text
specs/013-model-profiler/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   ├── profile-schema.md     # the persisted model_profiles.json record (consumed by F3)
│   └── profiler-api.md       # the public ModelProfiler / probe / scoring API surface
├── checklists/
│   └── requirements.md       # spec quality checklist (from /speckit.specify)
└── tasks.md             # /speckit.tasks output (NOT created here)
```

### Source Code (repository root)

```text
promptchain/
└── profiler/
    ├── __init__.py            # public API: ModelProfiler, Jacket, CapabilityProfile, run_probe
    ├── irt.py                 # 3PL P_i(θ), Fisher info I_i(θ), EAP + WLE estimators  [pure math]
    ├── cat.py                 # CAT loop: max-info item selection, SE stop rule        [pure math]
    ├── item_bank.py           # ProbeItem, ItemBank, calibration-quality filter, synthetic/seed bank
    ├── scoring.py             # per-dimension auto-scorers (instruction/tool/reasoning/format/…)
    ├── composite.py           # C, K (calibration), F (cost penalty), Ω; jacket band-mapping + escalation
    ├── jacket.py              # Jacket + CapabilityProfile dataclasses (+ to/from dict)
    ├── probe.py               # the probe harness: isolated PromptChain sessions per item, emits F1 transcripts, drives CAT
    ├── store.py               # persist/load model_profiles.json; idempotent upsert; EWMA refine; two-sided Δθ fit
    └── prompt_generator.py    # DSPy model_prompt_generator module (per-model profile → jacket)  [US3]

tests/
├── test_profiler_irt.py            # FR-016: 3PL P_i, Fisher info, EAP/WLE recover known θ within tol
├── test_profiler_cat.py            # CAT next-item selection + SE≤τ stop rule + max-item cap
├── test_profiler_item_bank.py      # calibration filter, synthetic bank with known params
├── test_profiler_scoring.py        # per-dimension scorers map outcomes → [0,1] correctly
├── test_profiler_composite.py      # FR-017: Ω formula, jacket bands per Ω range, escalation triggers
├── test_profiler_probe_integration.py  # end-to-end probe (fake model) → F1-valid trial transcripts + persisted profile + reproducibility
└── test_profiler_refine.py         # EWMA refine update + two-sided Δθ jacket selection
```

**Structure Decision**: Single-project library. New code is isolated under
`promptchain/profiler/` (mirrors how F1 isolated everything under
`promptchain/observability/`). The pure-math core (`irt`, `cat`, `composite`, `item_bank`,
`scoring`) has no I/O and no model calls, so it is fully unit-testable with synthetic
known-parameter fixtures — satisfying the test-first acceptance criterion without any live model.
The harness (`probe`), persistence (`store`), and DSPy module (`prompt_generator`) sit on top and
are exercised by integration tests with a fake/stub model plus one offline live smoke against LAN
ollama.

## Phasing → User-Story mapping (drives tasks + waves)

- **US1 (P1, MVP)**: `irt` + `cat` + `item_bank` + `scoring` + `probe` + `store` (persist) →
  model id → persisted profile {per-skill θ̂, C, tier, budget}; trials emitted as F1 transcripts.
- **US2 (P2)**: `composite` (Ω, K, F) + `jacket` derivation + escalation.
- **US3 (P3)**: `store` EWMA refine + two-sided Δθ fit + `prompt_generator` (DSPy), reusing SIO
  experiment/GEPA where present (graceful no-op when absent / no telemetry).

## Complexity Tracking

> No constitution violations — section intentionally empty.
