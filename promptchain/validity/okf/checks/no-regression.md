---
type: Assertion
title: No regression on correct baseline items
description: An intervention must not make items the baseline got right worse; aggregate gains can hide regressions.
tags: [validity, regression]
---

A technique can raise the aggregate while silently breaking items the baseline got right. Compare
item-level correctness, not averages: count items where base=correct AND treatment=wrong. Case: a critique
that turned a correct restraint into a hallucination — the "helper" LOWERED a correct base.

Function: `promptchain.validity.no_regression(control_correct, treatment_correct)`. See TFX ModelValidator.
