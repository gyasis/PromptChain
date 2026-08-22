# Experiment Validity — OKF knowledge bundle

The agent-readable instruction set for validating ML/LLM experiments before trusting a score.
Pairs with the executable checks in `promptchain.validity`. Read [validation-workflow](/validation-workflow.md) first.

## Procedural checks (`checks/`)
- [technique-fired](/checks/technique-fired.md) — did the intervention actually run?
- [no-regression](/checks/no-regression.md) — did it break correct baseline items?
- [harness-faithful](/checks/harness-faithful.md) — does a known-good model reproduce its known score?
- [above-noise](/checks/above-noise.md) — is the delta bigger than run-to-run variance?

## Statistical measures (`tests/`)
- [mcnemar](/tests/mcnemar.md) — the correct test for paired pass/fail
- [wilson-ci](/tests/wilson-ci.md) — confidence interval for a pass-rate
- [holm-bonferroni](/tests/holm-bonferroni.md) — correct for comparing many arms
- [power-mde](/tests/power-mde.md) — can the eval even detect this effect?
