---
type: StatisticalTest
title: McNemar's test (paired pass/fail)
description: The correct significance test for two arms scored on the SAME items with binary outcomes.
tags: [statistics, paired, binary]
---

When comparing base vs treatment on the SAME scenarios with pass/fail outcomes (our usual shape), the
correct test is **McNemar's**, not a t-test (which ignores the pairing and overstates significance —
Dietterich 1998). χ² = (|n01−n10|−1)²/(n01+n10), df=1, over the discordant pairs (n01 = base wrong &
treatment right; n10 = the reverse).

Function: `promptchain.validity.mcnemar(base_correct, treatment_correct)` or the one-call
`compare_paired_binary(...)` (McNemar + Wilson CIs + Cohen's h + STRONG/WEAK verdict).
