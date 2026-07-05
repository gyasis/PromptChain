---
type: Assertion
title: Harness faithful (positive control)
description: A known-good model must reproduce its known score through your eval rig, else the rig is broken.
tags: [validity, harness, positive-control]
---

Before trusting ANY number, run a known-good model through the SAME rig. If it doesn't reproduce its
known score within tolerance, the **harness is broken** and every result is suspect. Case: glm-5.2 scored
70 natively but 40 through a naked eval server — the rig capped everyone at 40, so "the technique doesn't
help" was really "the rig is broken". Grounded in Breck et al. ML Test Score (Infra Test 2).

Function: `promptchain.validity.harness_faithful(known_score, expected_score)`.
