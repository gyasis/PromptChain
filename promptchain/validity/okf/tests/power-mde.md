---
type: StatisticalTest
title: Power & minimum detectable effect
description: Can your eval set even detect the effect you're claiming? Underpowered evals lie both ways.
tags: [statistics, power, sample-size]
---

Before trusting a delta (or a null), ask whether the eval is big enough to see it. Minimum Detectable
Effect: MDE ≈ (z_{α/2}+z_β)·√(2p(1−p)/n). If your delta < MDE, the set is too small. Underpowered evals
produce Type-M (magnitude) errors — "significant" results are exaggerated (Card et al. 2020).

Function: `promptchain.validity.min_detectable_effect(n, p_baseline)`.
