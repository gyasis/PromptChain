---
type: Assertion
title: Technique actually fired (no-op detection)
description: Treatment output identical to control means the intervention never ran — a non-result.
tags: [validity, no-op]
---

If the treatment arm's per-item outputs are (near-)identical to control, the technique **did not fire**
(a config/wiring bug, a gate that never triggered). That is a NON-result — not evidence the technique
has no effect. Case: a gated critique whose gate never fired → treatment byte-identical to base → looked
like "critique fails" but the critique simply never ran. Related: [validation-workflow](/validation-workflow.md).

Function: `promptchain.validity.technique_fired(control_outputs, treatment_outputs)`.
Deeper: also assert the intervention's STEP fired via the pipeline trace (PromptChain `trail`/`step_outputs`),
and use a "strength knob" (`monotonic`) — a flat metric across the knob means a no-op.
