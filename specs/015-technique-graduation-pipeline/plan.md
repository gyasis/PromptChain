# Plan 015 — Technique Graduation Pipeline

Implements spec.md. Approach: minimal, additive, test-first. No new subsystem — wire the existing
`promptchain.validity` + local callers into a repurposed bench and a thin graduation surface.

## Architecture

```
promptchain/utils/agentic_step_processor.py        (PRODUCTION — graduation target)
  + AgenticStepProcessor(..., verify: VerifyHook | None = None, okf=None, caller=None)   # additive, opt-in, default None
  + a VerifyHook protocol: (emitted_call, context) -> maybe-replacement-call   (dominance-gated)

promptchain/utils/enhanced_agentic_step_processor.py   (BENCH — experimental)
  - REMOVE always-on RAG LogicVerifier / Gemini augmentor / adaptive-learning core (audit-rejected)
  + module-level warning -> docs/ADVERSARIAL_ANALYSIS_SUMMARY.md
  + experiments: dict[str, cfg] — each an opt-in, default-OFF flagged technique host

promptchain/experiments/                              (NEW — the graduation harness)
  gate.py     : run_gate(base_fn, treatment_fn, scenarios, reps>=3) -> ValidityReport
                (wraps promptchain.validity: technique_fired, no_regression, above_noise,
                 harness_faithful, compare_paired_binary/mcnemar; ValiditySuite.raise_if_failed)
  bench.py    : pilot a flagged technique on EnhancedAgenticStepProcessor over a held-out slice
  README.md   : the pilot->validate->graduate runbook
```

## Key decisions

- **Additive + default-None** on `AgenticStepProcessor` — zero behavior change for existing users (FR-6).
- **Dominance gate** (FR-4): `verify` may only return a replacement call that passes a check the base
  call failed; otherwise it returns the base call unchanged. Monotonic by construction — cannot regress.
- **Local-first** (FR-5): the bench's default verifier is a `RawCaller`/`LlamaCppCaller`/`MLXCaller` through
  the governor; paid APIs are opt-in only. OKF injects static verification knowledge (no RAG-per-call).
- **Gate is mandatory before graduation** (FR-3/FR-7): `run_gate` returns a report; graduation PR must
  attach a passing report + the before/after numbers. A held-out split is required (FR-3) so a technique
  can't overfit the scenarios it was tuned on.
- **Negative control** (success criterion 3): a deliberately-null technique must FAIL the gate — proving
  the gate rejects no-ops, not just rubber-stamps.

## Risks / mitigations

- *Bench rot* (the thing that killed enhanced_ASP): mitigated because the bench now has a defined output
  (graduations) and a gate; a technique that never graduates is logged + dropped, not left disabled forever.
- *Gate gaming* (tuning to the eval): mitigated by the held-out split (FR-3) + `above_noise` N≥3 + McNemar.
- *Cost creep*: local-first default (FR-5); paid callers require an explicit flag.
