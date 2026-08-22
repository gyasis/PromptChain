# Tasks 015 — Technique Graduation Pipeline

Ordered, test-first. Each task is independently reviewable. `[P]` = parallelizable.

## Phase 0 — the gate (the turnstile first)
- T001 `promptchain/experiments/gate.py`: `run_gate(base_fn, treatment_fn, scenarios, reps=3, held_out) -> ValidityReport`.
  Wraps `promptchain.validity`: technique_fired, no_regression, above_noise(N>=3), harness_faithful,
  compare_paired_binary/mcnemar. Returns a structured report; `ValiditySuite.raise_if_failed()` on demand.
- T002 [P] Tests: gate PASSES a genuinely-better treatment; gate FAILS (a) a no-op treatment
  (negative control), (b) an in-noise treatment, (c) a treatment that regresses a base-correct item.
- T003 [P] `promptchain/experiments/README.md`: the pilot -> validate -> graduate runbook.

## Phase 1 — repurpose the bench (do no harm)
- T010 Strip the always-on RAG `LogicVerifier` / `GeminiReasoningAugmentor` / adaptive-learning core from
  `enhanced_agentic_step_processor.py`; add a module-level warning pointing at
  `docs/ADVERSARIAL_ANALYSIS_SUMMARY.md`. Preserve the memo/interrupt/context-distiller layer as a
  graduation candidate cohort (untouched, still tested).
- T011 Add `experiments: dict[str, cfg]` (default `{}`, all OFF) as the flagged-technique host on the bench.
- T012 [P] Update/retire the tests that only asserted the removed core; keep the memo/steering tests green.

## Phase 2 — the graduation surface (production, additive)
- T020 `AgenticStepProcessor`: add opt-in `verify: VerifyHook | None = None` (+ `okf`, `caller`) — default
  None = zero behavior change. Define the `VerifyHook` protocol (dominance-gated: may only replace on a
  provable win; else return the base call).
- T021 [P] Tests: default construction unchanged; a passing verify replaces on a base-fail; verify NEVER
  regresses a base-correct call (monotonicity assertion, mirrors `no_regression`).

## Phase 3 — the first pilot (prove the pipeline end-to-end)
- T030 `promptchain/experiments/bench.py`: pilot "local-caller dominance-gate verify" on the bench over a
  held-out slice of a tool-calling scenario set (local caller via governor; emit-not-execute).
- T031 Run `run_gate` on the pilot. Attach the ValidityReport + before/after numbers.
- T032 IF green -> graduate the technique to `AgenticStepProcessor(verify=...)` in a PR that links the
  report (FR-7). IF in-noise -> record it, leave on the bench. Either way the PIPELINE is validated.

## Definition of done
- Gate rejects a null technique (negative control passes) and accepts a real one.
- One technique either graduates (with an attached passing report) or is provably parked — demonstrating
  both branches of the turnstile.
- `AgenticStepProcessor` default behavior is byte-unchanged for existing users.
