# Implementation Plan: Dynamic Prompt Layer (drive assembly through the formulas)

**Branch**: `014-dynamic-prompt-layer` | **Date**: 2026-06-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/014-dynamic-prompt-layer/spec.md`

## Summary

Add the **dynamic half** of the two-layer prompt as a set of small, additive modules under
`promptchain/prompts/`, plus one model-aware generator that ties them together. At assembly time
the generator reads the target model's **F2 capability profile / jacket** (via F2's existing
`ModelProfiler`/`store` — no new persistence), then: (1) maps the profile's measured capability to
a **prompt tier** of CORE / EXTENDED / TINY; (2) applies a **family adapter** (format-only,
per-family variants over a `default` fallback) derived from the `model_id`; (3) renders a
**toolshim** when the jacket's `tool_mode` is a shim mode (`<tools>` JSON-in-text + plain-text tool
history), else passes tools natively; (4) **measures and trims** the assembled prompt to a
300–1,000-token base (hard cap ~1,500) by dropping **optional** modules — never the static base;
and (5) for weak tiers, a **Document-&-Clear** longevity policy the loop runs (`<turn-context>` +
goal re-injection each turn, dump plan/progress to disk at ~60% context, clear, resume; ≥10
task-turns before reset; escalate only on stall).

The **static base** (`TUI_FOUNDATION_PROMPT` + `DynamicTUIPromptGenerator`) is reused **by
composition and left byte-identical** — the split is purely additive at the `BasePromptBuilder`
seam consumed in `agentic_step_processor.py:1006`. The new generator is a drop-in
`BasePromptBuilder` that binds its model at construction (the protocol's `generate(objective,
tools, context)` has no `model` parameter), with an optional `model=` keyword override for direct
use/tests. The **one** additive touch to F2 is an **optional** `tool_mode` field on the `Jacket`
dataclass (sanctioned additive-optional field; no `schema_version` bump, no math change). A small
**A/B eval harness** + tiny task set validates SC-007 (weak-model completion improves vs the static
base) — deterministic parts unit-tested with a fake model, the real A/B run is an offline live
smoke against LAN ollama (mirroring F2).

## Technical Context

**Language/Version**: Python 3.10+ (repo CI runs 3.10 and 3.12; current active 3.12.11).
**Primary Dependencies**: `promptchain.prompts` (the static base `TUI_FOUNDATION_PROMPT` +
`DynamicTUIPromptGenerator`, reused by composition) and `promptchain.profiler` (F2 — read the
profile/jacket via `ModelProfiler.get_profile` / the store; read-only consumer). `tiktoken`
(already used by `DynamicTUIPromptGenerator.get_token_estimate` and
`execution_history_manager.py`) for budget measurement, with the `len//4` fallback already present.
The existing loops (`promptchain/utils/external_loop.py` `ExternalLoop`, `ralph_chain.py`
`RalphChain`, `history_summarizer.py` `HistorySummarizer`, `context_distiller.py`) host US3. NO
`sio` import; NO new third-party dependency.
**Storage**: none new. F3 **reads** `~/.promptchain/model_profiles.json` through F2's store; US3's
Document-&-Clear writes a per-task **progress doc** (e.g. `PROGRESS.md`/`todo.md`) to a
caller-provided working dir (a temp dir in tests).
**Testing**: pytest, **flat** `tests/test_dynamic_prompt_*.py` (mirrors F1's `test_transcript_*` /
F2's `test_profiler_*` flat layout — NOT contract/integration/unit subdirs). `pytest-asyncio` is
NOT installed; any async-driving test uses plain `def test_*` + `asyncio.run(...)`.
**Target Platform**: Linux/macOS — cross-platform Python library.
**Project Type**: single (library).
**Performance Goals**: `generate()` is synchronous, **deterministic**, and cheap (one token
estimate + string assembly, no model call). Assembled base stays within 300–1,000 tokens (hard cap
~1,500). Per-model differentiation + budget compliance are verifiable by diffing/​measuring rendered
strings offline.
**Constraints**: static base **byte-identical** (parity floor never regresses, never dropped under
budget pressure); F3 is a **read-only** consumer of the FROZEN F2 profile/jacket schema (defensive
reads; only an additive-optional `tool_mode` field added); null/missing jacket → documented
fallback, never raise; offline-deterministic core with no live model on the unit-test path; the
real A/B win is shown via an offline live smoke (LAN ollama, no paid credentials).
**Scale/Scope**: a handful of model families (anthropic / openai / google / qwen / llama / default);
three prompt tiers; one generator; one longevity policy; a tiny A/B eval set (a few scoped coding
tasks at a couple of token tiers).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | How this feature satisfies it |
|---|---|---|
| I. Library-First | ✅ | Small self-contained modules under `promptchain/prompts/` (`tiers`, `family`, `toolshim`, `budget`, `longevity`, `model_dynamic`); each pure-function core is independently testable with zero model calls; one clear purpose (per-model assembly). |
| II. Observable Systems | ✅ | The assembled prompt + the chosen tier/family/tool_mode are inspectable, machine-readable outputs; the progress doc is a durable on-disk artifact; F1 transcripts remain the run-time observability surface (unchanged). |
| III. Test-First (NON-NEGOTIABLE) | ✅ | Every module gets unit tests FIRST (red): tier mapping, family derivation, toolshim rendering, budget trimming (static base never dropped), generate() determinism + per-model difference + budget compliance + null-jacket fallback, and the longevity policy — then implementation (green). Test waves strictly precede impl waves. |
| IV. Integration Testing | ✅ | An integration test wires the new `DynamicModelPromptGenerator` through the `BasePromptBuilder` seam (the contract boundary `agentic_step_processor.py` consumes) and asserts static-base-intact + drop-in conformance; an F2↔F3 read test asserts a real persisted profile/jacket is consumed correctly (incl. null jacket). |
| V. Token Economy & Performance | ✅ | The whole feature IS token economy: measured assembly, 300–1,000 budget, drop-optional-over-cap, weak-model Document-&-Clear so context never grows unbounded. Uses the existing tiktoken estimator; no extra model calls. |
| VI. Async-First Design | ✅ | `generate()` is pure/sync by nature (no I/O); the US3 longevity policy exposes async-friendly hooks that the existing async loops call, with sync wrappers where a sync path is offered (matching the repo dual-interface pattern). |
| VII. Simplicity & Maintainability | ✅ | Reuse over rebuild: composes the existing static base + loops rather than re-authoring them; one additive-optional field on F2's `Jacket`; family/tier/tool_mode are tiny pure mappings; YAGNI on the eval (a tiny task set, not a benchmark suite). No speculative abstractions — only the five PRD §7 components. |

**Result**: PASS — no violations. Complexity Tracking left empty.

## Project Structure

### Documentation (this feature)

```text
specs/014-dynamic-prompt-layer/
├── plan.md              # This file
├── research.md          # Phase 0 output (decisions: tier mapping, family source, tool_mode source, D&C seam, eval set)
├── data-model.md        # Phase 1 output (PromptTier, FamilyAdapter, ToolShim, OptionalModule, ProgressDoc, A/B eval entities)
├── quickstart.md        # Phase 1 output (how to call generate() + run the offline live smoke / A/B)
├── contracts/
│   ├── generator-api.md      # DynamicModelPromptGenerator + tier/family/toolshim/longevity public surface
│   └── prompt-layout.md      # the assembled-prompt section order + budget/drop-order + <tools>/<turn-context> shapes
├── checklists/
│   └── requirements.md       # spec quality checklist (from /speckit.specify)
└── tasks.md             # /speckit.tasks output (NOT created here)
```

### Source Code (repository root)

```text
promptchain/
├── prompts/
│   ├── __init__.py           # (edit) export DynamicModelPromptGenerator + tier/family/toolshim/longevity helpers
│   ├── base.py               # (unchanged) BasePromptBuilder Protocol
│   ├── tui_dynamic.py        # (unchanged) TUI_FOUNDATION_PROMPT + DynamicTUIPromptGenerator — reused by composition
│   ├── tiers.py              # NEW: PromptTier {CORE,EXTENDED,TINY}; map profile capability/recommended_tier → tier; per-tier optional-module set  [pure]
│   ├── family.py             # NEW: family-from-model_id; per-family FORMAT adapter over a `default` fallback (format-only)  [pure]
│   ├── toolshim.py           # NEW: resolve tool_mode; render <tools> JSON-in-text; re-serialize tool history as plain text; native passthrough  [pure]
│   ├── budget.py             # NEW: measure (tiktoken) + drop-optional-over-cap trimming; static base never dropped; hard cap  [pure]
│   ├── longevity.py          # NEW (US3): <turn-context>+goal builder; DocumentAndClear policy (threshold trigger, write progress doc, clear+resume, turn/stall tracking, escalate-on-stall)
│   └── model_dynamic.py      # NEW: DynamicModelPromptGenerator — BasePromptBuilder; reads F2 profile (model bound at __init__, optional model= override); composes static base + tier + family + toolshim + budget
└── profiler/
    └── jacket.py             # (edit) add OPTIONAL `tool_mode: Optional[str] = None` to Jacket + to_dict/from_dict — the ONE additive-optional F2 touch

tests/
├── test_dynamic_prompt_tiers.py      # capability/recommended_tier → CORE/EXTENDED/TINY thresholds; per-tier module sets (TINY may swap protocol)
├── test_dynamic_prompt_family.py     # family derivation from model_id; `default` fallback for unknown; format-only (no semantic-base change)
├── test_dynamic_prompt_toolshim.py   # tool_mode resolution (jacket→native default); <tools> render for shim; native passthrough; plain-text history
├── test_dynamic_prompt_budget.py     # token measure; drop optional modules in order over cap; static base + objective + tools never dropped; hard cap honored
├── test_dynamic_prompt_generate.py   # US1: per-model difference; budget-compliant; static base VERBATIM (byte-identical); null/missing jacket fallback; determinism; BasePromptBuilder drop-in conformance
├── test_dynamic_prompt_longevity.py  # US3: <turn-context> each turn; D&C at threshold writes doc + clears + resumes; ≥10 turns; escalate only on stall (respects jacket.escalate); disk-unavailable → lossy fallback
├── test_dynamic_prompt_eval.py       # SC-007: A/B harness scoring + aggregation deterministic with a fake model (real A/B is the live smoke)
└── test_profiler_jacket_toolmode.py  # the additive Jacket.tool_mode field round-trips (to_dict/from_dict) + F2 defaults stay green
```

**Structure Decision**: Single-project library. New code is isolated under `promptchain/prompts/`
(where the static base + the existing builders already live — F3 is the *dynamic builder* sibling
of the static one), mirroring how F1 isolated `observability/` and F2 isolated `profiler/`. The
pure cores (`tiers`, `family`, `toolshim`, `budget`) have no I/O and no model calls → fully
unit-testable offline. `model_dynamic` (the generator) and `longevity` (US3) sit on top; the
generator reads F2's profile via the existing store, and the longevity policy is exercised with a
fake context-usage signal + a temp dir. The single F2 edit (`jacket.py` optional `tool_mode`) is
additive-optional and covered by its own round-trip test so F2's 66 tests stay green.

## Phasing → User-Story mapping (drives tasks + waves)

- **US1 (P1, MVP)**: `tiers` + `family` + `budget` + `model_dynamic` → `generate(objective, tools,
  model)` returns a profile-appropriate, budget-compliant prompt that **differs per model** (tier +
  family), with the **static base verbatim** and **null/missing-jacket fallback**. (Reads F2 via the
  existing store.)
- **US2 (P2)**: `toolshim` (tool_mode resolution, `<tools>` JSON-in-text, plain-text history) +
  the additive `Jacket.tool_mode` field, wired into `model_dynamic`.
- **US3 (P3)**: `longevity` (Document-&-Clear policy: `<turn-context>` + goal re-injection, dump@~60%,
  clear+resume, ≥10 turns, escalate-on-stall, lossy fallback), integrated at the existing loop seam.
- **A/B eval (SC-007, spans US1)**: a tiny eval harness + task set; deterministic scoring unit-tested
  with a fake model; the real weak-vs-static win demonstrated by an **offline live smoke** (LAN ollama).

## Key design decisions carried into Phase 0 (research.md expands)

1. **`generate()` signature vs the protocol** — the `BasePromptBuilder.generate(objective, tools,
   context)` has no `model` param (third positional is `context`). F3 **binds the model at
   construction** (`DynamicModelPromptGenerator(model=..., store=...)`) for a clean drop-in, plus an
   optional **keyword** `model=` override on `generate()` for direct/tests use. The spec's
   `generate(objective, tools, model)` is the conceptual surface; `model` is keyword-only in code.
2. **Tier taxonomy is F3-owned** — the jacket's tier band is `lean/standard/rich/max-rich` (a
   budget/mode axis). F3 maps the profile's **capability C** (and `recommended_tier`) onto its own
   **CORE/EXTENDED/TINY** prompt-strength axis with documented thresholds; the jacket's
   `budget_tokens` drives the budget axis. Two separate, documented mappings.
3. **`family` is derived from `model_id`** (string parse → anthropic/openai/google/qwen/llama/…,
   `default` fallback) — pure F3 logic, no F2 field, format-only.
4. **`tool_mode` source** — read `jacket.tool_mode` (the new optional field); default **native** when
   absent (FR-010). Populating it richly from probe detection is a future F2 refinement (out of
   scope here); F3 only consumes it + lets tests/jackets set it.
5. **Document-&-Clear seam** — a self-contained policy object the loop calls (not a loop rewrite):
   pure decision functions (`<turn-context>` build, threshold trigger, turn/stall tracking) + an
   I/O method that writes the progress doc and returns the cleared/resumed history; lossy
   `HistorySummarizer` is the documented fallback.

## Complexity Tracking

> No constitution violations — section intentionally empty.
