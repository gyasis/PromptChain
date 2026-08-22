# promptchain.experiments — the technique graduation harness

Spec: `specs/015-technique-graduation-pipeline/`. Issues: #49 (bench), #50 (graduation target).

Agentic-technique enhancements earn their way into production. Pilot on the **bench**
(`EnhancedAgenticStepProcessor`), prove with the **gate** (`promptchain.validity`), and only then
**graduate** to an additive, opt-in `AgenticStepProcessor` param. Nothing ships unproven — the discipline
whose absence sank the original enhanced_ASP ("10x" claimed, never measured; see
`docs/ADVERSARIAL_ANALYSIS_SUMMARY.md`).

```
[pilot technique] --opt-in flag--> EnhancedAgenticStepProcessor (bench)
   -> run_gate(...) : technique_fired + no_regression + above_noise(N>=3, held-out) + harness_faithful + McNemar
   -> PROVEN?  no  -> stays on the bench / dropped; logged, NEVER ships
              yes  -> GRADUATE: add as an additive AgenticStepProcessor param (attach the report)
```

## The gate

```python
from promptchain.experiments import run_gate, split

tune, held_out = split(scenarios, frac=0.5)     # gate on data the technique was NOT tuned on (FR-3)

rep = run_gate(
    base_fn,          # (scenario) -> output   : the base step
    treatment_fn,     # (scenario) -> output   : base + the piloted technique
    held_out,
    score_fn=lambda s, o: is_correct(s, o),     # (scenario, output) -> bool
    reps=3,                                      # >= 3 (Henderson/Dodge/Card); the gate enforces this
    known_fn=None, expected_score=None,          # optional positive control (harness_faithful)
)

print(rep.summary())
if rep.passed:            # True only if EVERY validity check passed
    graduate(...)         # add the technique as an AgenticStepProcessor param; attach rep.summary() to the PR
else:
    rep.raise_if_failed() # or keep it on the bench
```

`run_gate` returns a `GateReport`:
- `.passed` — True iff `technique_fired` AND `no_regression` AND `above_noise` AND (if a control was given)
  `harness_faithful` AND a significant paired improvement (McNemar).
- `.summary()` — the base→treatment scores, verdict, and every check.
- `.raise_if_failed()` — blocks graduation with an `AssertionError` unless the technique is proven.

## Why each check (what it stops)
- **technique_fired** — treatment identical to base = the technique never ran (a non-result, not evidence it fails).
- **no_regression** — never worsens an item the base got right (monotonicity; ties → base).
- **above_noise** — the delta must exceed run-to-run variance over N≥3 reps; n=1 proves nothing.
- **harness_faithful** — a known control must reproduce its known score, or the rig is broken.
- **McNemar** — the correct significance test for paired pass/fail (not a t-test); small effects need real n.

## Graduation checklist (FR-7)
A graduation PR must attach: the held-out scenarios, `reps`/seeds, `rep.summary()` (the passing report), and
the before/after numbers. No graduation without a recorded passing experiment.
