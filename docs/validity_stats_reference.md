# validity_stats — statistical inference reference

The formulas behind `promptchain/utils/validity_stats.py`. Pairs with `validity_suite` (procedural
checks). Use this to pick the RIGHT measure for your data shape so any inference you draw is sound.

## Decision guide — which test?

```text
outcome type      pairing        -> correct significance test
----------------  -------------  --------------------------------------------
pass/fail (binary) PAIRED         -> McNemar's test         <-- OUR case; use compare_paired_binary()
pass/fail (binary) unpaired       -> chi-square / Fisher's exact
score (continuous) PAIRED         -> paired t / Wilcoxon signed-rank (if non-normal)
score (continuous) unpaired       -> Welch's t / Mann-Whitney U (if non-normal)
3+ arms, paired    (omnibus)      -> Friedman test, then post-hoc + correction
```
Always pair a p-value with an **effect size** and a **confidence interval**; when comparing K arms,
apply a **multiple-comparison correction**; before trusting a null, check **power/MDE**.

## Significance tests

| Measure | Formula | Use when | Prevents |
|---|---|---|---|
| **McNemar** | χ² = (\|n01−n10\|−1)² / (n01+n10), df=1 | paired pass/fail (same items) | t-test on paired binary → overstated significance (Dietterich 1998) |
| Welch's t | t = (x̄₁−x̄₂)/√(s₁²/n₁+s₂²/n₂) | unpaired continuous, unequal var | Student's-t → inflated Type-I error |
| Wilcoxon signed-rank | signed ranks of paired diffs | paired ordinal / non-normal | paired-t assuming normality |
| Mann-Whitney U | rank-sum of two groups | unpaired non-normal | t-test on skewed data |
| Friedman | ranks across K paired arms | 3+ arms omnibus | many pairwise tests with no global check |

## Effect size (magnitude — report ALONGSIDE p)

| Measure | Formula | Note |
|---|---|---|
| **Cohen's h** (proportions) | h = 2·asin(√p₁) − 2·asin(√p₂) | 0.2 small · 0.5 med · 0.8 large; accounts for the ceiling (90→95 ≠ 50→55) |
| Cohen's d (continuous) | (x̄₁−x̄₂)/pooled_sd | 0.2 / 0.5 / 0.8 |
| Odds ratio | (p₁/(1−p₁))/(p₂/(1−p₂)) | "likelihood multiplier" |

## Uncertainty (never report a bare mean)

| Measure | Note |
|---|---|
| **Wilson score CI** | gold standard for pass-RATES; Wald interval gives impossible >100%/<0% near the edges |
| Bootstrap CI | assumption-free (percentile of resamples); for custom metrics with no closed form |
| Standard error | SE = s/√N — the CI/significance building block |

## Multiple comparisons (K arms/scenarios)

P(≥1 false positive) = 1 − (1−α)^K. Correct it:
- **Holm-Bonferroni** — controls Family-Wise Error Rate (want *zero* false positives). Step-down.
- Benjamini-Hochberg (BH) — controls False Discovery Rate (tolerate a small % of false positives for more power; exploratory).

## Power & sample size

- **Power (1−β)**: probability of detecting a real effect. Target ≥ 0.80.
- **Minimum Detectable Effect**: MDE ≈ (z_{α/2}+z_β)·√(2p(1−p)/n). If your delta < MDE, the eval is too small to see it.
- **Type-M error (Card 2020)**: underpowered evals *exaggerate* the effects that happen to pass p<0.05 — so an underpowered "significant" result is untrustworthy in both existence AND magnitude.

## Reliability (when a judge/metric is in the loop)

- **Cohen's κ** = (p_o − p_e)/(1 − p_e) — agreement corrected for chance (judge vs gold, or two judges). Relevant when using an LLM-as-critic: if κ ≈ 0 the judge is no better than chance.
- ICC(2,1)/(3,1) — absolute agreement across raters (detects a consistently harsher judge). (scipy/pingouin.)

## Priority for the library
MUST: McNemar · Wilson CI · Holm-Bonferroni · MDE/power · Cohen's h/d.
NICE: bootstrap CI · Wilcoxon/Mann-Whitney/Friedman · Cohen's κ / ICC.

## References
Dietterich (1998) *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*;
Card et al. (2020) *With Little Power Comes Great Responsibility*; Cohen (1988) *Statistical Power Analysis*;
Demsar (2006) *Statistical Comparisons of Classifiers over Multiple Data Sets*; Dodge et al. (2019) *Show Your Work*.
