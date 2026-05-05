---
id: recipe-advanced-agentic
title: AgenticStepProcessor advanced — Phase 1-4 cost/token/safety/transparency knobs
when: You're using `AgenticStepProcessor` and want to control cost, token budget, error rate, or reasoning transparency.
api_version: 0.6.1+
---

# Advanced AgenticStepProcessor

`AgenticStepProcessor` ships four research-backed feature phases. All are **off by default** for backward compatibility. Turn them on independently or together.

**Source:** `promptchain/utils/agentic_step_processor.py:135` (constructor signature) and `examples/two_tier_routing_demo.py` (canonical end-to-end demo).

## Phase 1 — Two-Tier Routing (cost saver, ~60-70%)

Route simple sub-tasks to a cheap fallback model; keep the primary model for hard ones.

```python
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

agentic = AgenticStepProcessor(
    objective="…",
    model_name="openai/gpt-4o",                # primary (expensive)
    fallback_model="openai/gpt-4o-mini",       # cheap fallback
    enable_two_tier_routing=True,              # ✅ Phase 1
)
```

**How it routes:** an internal classifier inspects each sub-task and picks `model_name` for complex ones, `fallback_model` for simple ones (file listing, summarisation of small text, etc.). Customise by subclassing and overriding `_classify_task_complexity()`.

**Measured savings:** 60-70% on typical workloads (Gemini 2.5 Pro + Flash-8B = 33x cost ratio). See `examples/two_tier_routing_demo.py:52-108`.

## Phase 2 — Blackboard Architecture (token saver, ~72%)

Replace linear chat history with a structured "blackboard" of facts/observations/plans. Constant ~1000-token state instead of indefinitely growing context.

```python
agentic = AgenticStepProcessor(
    objective="…",
    enable_blackboard=True,                    # ✅ Phase 2
)
```

**Measured:** 71.7% token reduction (39,334 → 11,125 tokens across 10 internal iterations). LRU eviction + snapshot/rollback support built in.

⚠️ **Side effect:** changes the format of tool results the LLM sees (structured slot-fill instead of raw stringified tool output). Re-validate your prompts when toggling.

## Phase 3 — Safety: CoVe + Checkpointing (~80% error reduction)

Two complementary safety mechanisms:
- **Chain of Verification (CoVe)** — pre-execution validation of every tool call; assigns a confidence score; rejects calls below `cove_confidence_threshold`.
- **Epistemic Checkpointing** — detects "stuck" states (same tool 3+ times) and rolls back to a known-good snapshot.

```python
agentic = AgenticStepProcessor(
    objective="…",
    enable_cove=True,                          # ✅ Phase 3a
    cove_confidence_threshold=0.7,             # 0-1; 0.7 is the documented default
    enable_checkpointing=True,                 # ✅ Phase 3b
)
```

**Measured:** 80% error reduction (5 → 1) and 100% dangerous-operation prevention. Cost: ~5-10% overhead (CoVe issues 1 extra LLM call per tool, but uses the cheap model when two-tier routing is on, so it's ~5% of overall cost).

## Phase 4 — TAO Loop + Dry Runs (transparent reasoning)

Make the implicit ReAct loop explicit: separate Think / Act / Observe phases, and predict tool outputs before execution.

```python
agentic = AgenticStepProcessor(
    objective="…",
    enable_tao_loop=True,                      # ✅ Phase 4a
    enable_dry_run=True,                       # ✅ Phase 4b
)
```

**TAO loop output:**
```
THINK:   "I need to search for patterns first"
ACT:     search_files()           [predicted: 'find 10-15 .py files']
OBSERVE: 'Found 15 .py files'     [accuracy: 0.9]
…
```

Dry runs track prediction-vs-reality accuracy over time — useful for catching when an agent's mental model has drifted from actual tool behaviour.

**Cost:** <15% overhead with all Phase 4 features on.

## Full stack — all phases combined

The stack the user's global CLAUDE.md prefers (`enable_blackboard=True` and `enable_cove=True`) is the safety+token-savings sweet spot. Everything turned on:

```python
agentic = AgenticStepProcessor(
    objective="Comprehensive analysis of project architecture",
    max_internal_steps=15,

    # Phase 1
    model_name="gemini/gemini-2.5-pro",
    fallback_model="gemini/gemini-1.5-flash-8b",
    enable_two_tier_routing=True,

    # Phase 2
    enable_blackboard=True,

    # Phase 3
    enable_cove=True,
    cove_confidence_threshold=0.7,
    enable_checkpointing=True,

    # Phase 4
    enable_tao_loop=True,
    enable_dry_run=True,

    verbose=True,
)
```

**Combined measurement (from `examples/two_tier_routing_demo.py:434-457`):**
- Cost: 86% reduction ($1.25 → $0.17 per 1M tokens)
- Errors: 80% reduction
- Tokens: 71.7% reduction
- Transparency: 100% reasoning visibility

## History modes (orthogonal to the 4 phases)

```python
history_mode="minimal"      # last assistant + tool results only (cheapest, may deprecate)
history_mode="progressive"  # accumulate all assistant messages + tool results (RECOMMENDED)
history_mode="kitchen_sink" # keep everything (debug only)
```

## Spec 011 — `prompt_builder` and `workflow_pattern` (v0.6.0+)

The default prompt produced by `AgenticStepProcessor` is now dynamically built from the agent's *actually-registered* tools, not a hardcoded TUI scaffold. To use the legacy ReAct scaffold:

```python
from promptchain.prompts.legacy_tui import LegacyTUIPromptGenerator

agentic = AgenticStepProcessor(
    objective="…",
    prompt_builder=LegacyTUIPromptGenerator(),    # opt-in legacy
    workflow_pattern="react",                      # vs default "standard"
)
```

Or use the alias `TUIAgenticStepProcessor` if you want the old behaviour with no kwargs:

```python
from promptchain.utils.agentic_step_processor import TUIAgenticStepProcessor   # if exported
```

## Common failures + fix

- **Two-tier routing doesn't activate** — `fallback_model` is `None`. Pass it explicitly.
- **Blackboard breaks your prompt** — Phase 2 changes tool-result format. Re-test after enabling.
- **CoVe rejects every tool call** — `cove_confidence_threshold=0.7` is too high for your task; lower to 0.5 and observe.
- **Agent stuck and checkpointing doesn't help** — `max_internal_steps` is too low (rollback consumes budget). Bump to 15-20.
- **TAO + dry-run output is noisy** — don't combine with `verbose=True` unless you need the full trace; the loop already structures the output.
