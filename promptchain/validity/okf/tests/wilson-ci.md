---
type: StatisticalTest
title: Wilson score confidence interval (proportions)
description: The correct CI for a pass-RATE; never report a bare pass-rate without one.
tags: [statistics, confidence-interval, proportions]
---

A pass-rate is an estimate with uncertainty — always report a CI. Use the **Wilson score interval** for
proportions; the naive Wald interval (p ± 1.96·SE) gives impossible bounds (<0 or >1) near 0%/100%.

Function: `promptchain.validity.wilson_ci(passes, n)`.
