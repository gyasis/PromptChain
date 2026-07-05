---
type: StatisticalTest
title: Holm-Bonferroni (multiple comparisons)
description: Correct p-values when comparing many arms/scenarios, else a chance winner looks significant.
tags: [statistics, multiple-comparisons]
---

Comparing K arms inflates the false-positive rate: P(≥1 false positive) = 1 − (1−α)^K. **Holm-Bonferroni**
(step-down) controls the family-wise error rate. Use it for any multi-arm leaderboard or prompt sweep.
(Benjamini-Hochberg controls the false-discovery rate for exploratory work.)

Function: `promptchain.validity.holm_bonferroni(pvalues, alpha=0.05)`.
