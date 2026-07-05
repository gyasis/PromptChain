---
type: Assertion
title: Above noise (N-run significance) — the ground floor
description: n=1 proves nothing; a delta inside the base's own run-to-run variance is not a result.
tags: [validity, statistics, significance]
---

The FIRST and most fundamental check. Never conclude from a single run: run the same experiment N≥3–5
times to get a mean AND a variance, and check the BASE's own variance first — a "−5" delta is meaningless
if the base swings ±5. Case: "strategies hurt 84→79→75" was temp-0.2 noise; the base itself flipped
scenarios. Grounded in Henderson (Deep RL that Matters), Dodge (Show Your Work), Card (With Little Power).

Function: `promptchain.validity.above_noise(control_scores, treatment_scores)`. For paired pass/fail use
[mcnemar](/tests/mcnemar.md), not a t-test.
