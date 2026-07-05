# Spec 015 — Technique Graduation Pipeline (experimental bench → validity gate → production param)

Status: Draft · Created 2026-07-05 · Issues #49 (bench), #50 (graduation target)

## 1. Problem

Agentic-tool-call *enhancement* techniques (verify-before-execute, augment, repair, re-plan) keep being
authored directly into production and claimed to help without proof. The canonical failure is
`enhanced_agentic_step_processor.py`: a "10x improvement" pitch (RAG `LogicVerifier` + `GeminiReasoningAugmentor`)
that its **own** 5-agent adversarial audit rejected the *same day* (`docs/ADVERSARIAL_ANALYSIS_SUMMARY.md`:
DO NOT DEPLOY — +356% tokens, +81% latency, +456% cost, 5 CVSS 7.2–9.1 findings, fail-open approves
destructive tools). It shipped anyway as a prototype, was never re-measured, is disabled in every test, and
has zero production callers. The technique wasn't necessarily bad — it **skipped validation**.

The 2026-07 session built the missing rails: `promptchain.validity` (experiment-validity assertions),
runtime-agnostic local callers (`RawCaller`/`LlamaCppCaller`/`MLXCaller`), OKF knowledge injection, and the
emit-not-execute / dominance-gate discipline. This spec turns those rails into a **process**.

## 2. Goal

A two-stage lifecycle for agentic techniques so that **nothing reaches production unproven**:

- **`EnhancedAgenticStepProcessor` = the experimental bench.** A technique is piloted here behind an
  opt-in flag. Messy, experimental, safe to be wrong.
- **`AgenticStepProcessor` = production.** A technique becomes an **additive, opt-in parameter** here
  **only after** it passes the validity gate on the bench.
- **`promptchain.validity` = the turnstile.** The promotion criterion is a passing experiment, not a claim.

```
[pilot technique] --opt-in flag--> EnhancedAgenticStepProcessor (bench)
   -> validity experiment: technique_fired + no_regression + above_noise(N>=3, held-out)
                           + harness_faithful + McNemar (paired pass/fail)
   -> PROVEN?  no  -> stays on the bench / dropped; logged, NEVER ships
              yes  -> GRADUATE: add as an additive param on AgenticStepProcessor
                   -> the bench flag retires / becomes a thin alias
```

## 3. Functional requirements

- **FR-1 (bench, opt-in).** Every experimental technique on `EnhancedAgenticStepProcessor` is behind an
  explicit, default-OFF flag (e.g. `experiments={"local_verify": {...}}`). Default construction = base behavior.
- **FR-2 (repurpose, don't resurrect).** Strip the audit-rejected *always-on* RAG/Gemini core. Keep the shell
  as the flagged experiment host. A visible module-level warning points at `docs/ADVERSARIAL_ANALYSIS_SUMMARY.md`.
- **FR-3 (the gate is code, not vibes).** A promotion runs `promptchain.validity` on a held-out scenario set:
  `technique_fired` (not a no-op), `no_regression` (never worsens a base-correct item), `above_noise`
  (N≥3 reps, delta beyond base variance), `harness_faithful` (a known model reproduces its known score),
  and `compare_paired_binary`/`mcnemar` for the paired pass/fail delta. A `ValiditySuite` must pass
  (`raise_if_failed`) before graduation.
- **FR-4 (monotonic by construction).** A bench technique may only *replace* the base tool-call when it
  **provably passes a check the base call failed** (dominance gate); ties go to the base. Emit-not-execute:
  the harness runs tools, the step only emits — bounding blast radius and call count.
- **FR-5 (local-first).** Verification/augmentation defaults to a **local caller** (RawCaller/LlamaCpp/MLX)
  via the governor, not a paid per-call API — directly answering the +456% cost rejection. OKF-injected
  knowledge (`okf_agentic_context`/`okf_reader_tool`) replaces RAG-per-call where knowledge is static.
- **FR-6 (graduation = additive param).** A proven technique is added to `AgenticStepProcessor` as an
  opt-in parameter/hook (no subclass). The salvageable memo/interrupt/context-distiller layer graduates the
  same way — each still passes the bench + gate first.
- **FR-7 (provenance).** Each graduation records: the experiment (scenarios, N, seeds), the validity report,
  and the before/after numbers, linked from the issue. No graduation without a recorded passing experiment.

## 4. Non-goals

- Reviving the always-on RAG-verify + Gemini-augment core as-is (rejected; re-derive only via FR-3).
- Any "Nx improvement" claim that has not passed the FR-3 gate on a held-out set.
- Authoring a technique directly on `AgenticStepProcessor` (must transit the bench).

## 5. Success criteria

- The bench hosts ≥1 flagged technique with a runnable validity experiment.
- A green experiment graduates exactly one technique to an `AgenticStepProcessor` param, with the validity
  report attached — proving the pipeline end-to-end.
- A red experiment blocks graduation (demonstrated with a deliberately-null technique — negative control).

## 6. First pilot (the end-to-end proof)

Pilot **"local-caller dominance-gate verify"**: on the bench, an emitted tool-call is re-checked by a local
caller; the checked call replaces the base call ONLY if it passes a validity check the base failed
(FR-4). Run the FR-3 experiment on a held-out slice of a tool-calling scenario set (e.g. the existing
tool-eval-bench scenarios). If it passes → graduate as `AgenticStepProcessor(verify=...)`. If it's within
noise → it stays on the bench. Either outcome validates the *pipeline*.

## Refs
Issues #49/#50 · `docs/ADVERSARIAL_ANALYSIS_SUMMARY.md` · `promptchain.validity` (validity_suite/validity_stats)
· `promptchain/utils/{raw_caller,llama_cpp_caller,mlx_caller,okf_loader}.py` · plan.md · tasks.md
