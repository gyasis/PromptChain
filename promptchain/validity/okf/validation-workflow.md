---
type: Workflow
title: Validate an experiment before concluding
description: The order to run validity checks in; an experiment number is a hypothesis, not a conclusion.
tags: [validity, experiments, meta]
---

An experiment's aggregate score is a **hypothesis to verify, not a conclusion**. Before reporting
"technique X scored Y", run these — if any fails, investigate; do NOT conclude.

1. **[above-noise](/checks/above-noise.md)** (ground floor) — N≥3–5 reps; a delta inside the base's own
   variance is nothing. n=1 proves nothing.
2. **[harness-faithful](/checks/harness-faithful.md)** — a known-good model must reproduce its known
   score through your rig, else the rig is broken and all numbers are suspect.
3. **[technique-fired](/checks/technique-fired.md)** — treatment byte-identical to control = the
   intervention never ran (a non-result, not "no effect").
4. **[no-regression](/checks/no-regression.md)** — the intervention must not make correct base items worse.
5. **no-silent-defaults** — a parse/exception fallback must not quietly "pass".
6. Pick the CORRECT statistical test for the data shape: paired pass/fail → [mcnemar](/tests/mcnemar.md);
   report an effect size + a [wilson-ci](/tests/wilson-ci.md); comparing many arms → [holm-bonferroni](/tests/holm-bonferroni.md);
   before trusting a null, check [power-mde](/tests/power-mde.md).

Runnable: `promptchain.validity` (`ValiditySuite`, `compare_paired_binary`, ...).
