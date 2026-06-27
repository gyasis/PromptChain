---
description: "Task list for Dynamic Prompt Layer (F3)"
---

# Tasks: Dynamic Prompt Layer (drive assembly through the formulas)

**Input**: Design documents from `specs/014-dynamic-prompt-layer/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/generator-api.md,
contracts/prompt-layout.md, quickstart.md

**Tests**: INCLUDED and TEST-FIRST — Constitution III (NON-NEGOTIABLE). Every story's test wave is
written and FAILS (red) before its implementation wave (green).

**Organization**: Tasks grouped by user story (US1 P1 MVP → US2 P2 → US3 P3 → A/B eval). Wave tags
`[W#]` encode execution order; **impl waves run strictly after their test waves** (dev-kid
orchestrate collapses test→impl ordering, so the plan is regrouped after orchestration by the
`[W#]` tags — see plan.md + handoff gotcha #1).

## Format: `[ID] [P?] [W#] [Story] [agent] [file] Description`

- **[P]**: parallelizable (different file, no incomplete deps within the wave)
- **[W#]**: wave (executes fully, in order)
- **[Story]**: US1 / US2 / US3 (none for Setup / Eval / Polish)

## Path Conventions

Single-project library. New code under `promptchain/prompts/`; flat tests
`tests/test_dynamic_prompt_*.py` (+ `tests/test_profiler_jacket_toolmode.py` for the one additive
F2 field). The static base (`tui_dynamic.py`) and F2 (`profiler/`) are reused, not rebuilt.

---

## Phase 1: Setup (Shared Infrastructure)

- [x] T001 [W1] [python-pro] [promptchain/prompts/] Create empty module stubs `tiers.py`, `family.py`, `toolshim.py`, `budget.py`, `longevity.py`, `model_dynamic.py` under `promptchain/prompts/` so `import promptchain.prompts.<mod>` resolves; confirm the existing static base (`from promptchain.prompts import TUI_FOUNDATION_PROMPT, DynamicTUIPromptGenerator, BasePromptBuilder`) and F2 (`from promptchain.profiler import ModelProfiler, Jacket, CapabilityProfile`) import cleanly in this env. Do NOT yet edit `__init__.py` exports.

**Checkpoint**: new modules import as empty; static base + profiler reachable.

> **Phase 2 Foundational**: none beyond setup. F3's shared value types (`PromptTier`,
> `OptionalModule`) are small and live in `tiers.py`, created within US1; later stories depend on
> US1's generator rather than on a separate shared foundation.

---

## Phase 3: User Story 1 — Profile-appropriate, budget-compliant prompt per model (P1) 🎯 MVP

**Goal**: `generate(objective, tools, model)` reads the F2 profile/jacket → selects CORE/EXTENDED/
TINY tier → applies the family adapter → assembles over the intact static base → measures + trims
to a 300–1,000-token base (hard cap ~1,500, dropping optional modules, never the base). Two models
with different profiles get measurably different prompts; null/missing jacket falls back; output is
deterministic.

**Independent Test**: with two seeded profiles (strong vs tiny) + a fixed objective/tools, call
`generate()` for each → static base appears VERBATIM in both; outputs differ in tier/family; each
is within budget; forcing over-cap drops optional modules in order (never the base); null jacket →
`recommended_tier`+`budget_tokens`, no profile → CORE; identical inputs → identical output.

### Tests (W2) — write FIRST, must FAIL

- [x] T002 [P] [W2] [US1] [python-pro] [tests/test_dynamic_prompt_tiers.py] FAILING tests: `select_tier(profile)` maps capability to `PromptTier` by the D2 thresholds (`≥0.66→EXTENDED`, `0.33–0.66→CORE`, `<0.66... <0.33→TINY`), `recommended_tier` honored as hint, no-profile → CORE; `modules_for_tier(tier)` returns the documented `OptionalModule` set per tier (EXTENDED has `examples`+`extended_guidance`; CORE essentials only; TINY reduced) with correct `drop_priority` ordering.
- [x] T003 [P] [W2] [US1] [python-pro] [tests/test_dynamic_prompt_family.py] FAILING tests: `family_of(model_id)` derives `anthropic/openai/google/qwen/llama` from representative ids (incl. `provider/` prefixes) and `default` for unknown; `adapt_format(parts, family)` changes FORMAT only — the static-base substring is preserved unchanged for every family (incl. `default` = no-op).
- [x] T004 [P] [W2] [US1] [python-pro] [tests/test_dynamic_prompt_budget.py] FAILING tests: `measure(text)` returns a non-negative int (tiktoken or `len//4`); `fit_to_budget(base, optional, target_max, hard_cap)` drops optional modules in ascending `drop_priority` until within budget, NEVER drops `base`, returns `(assembled, dropped_keys)`; when the base alone exceeds `hard_cap` it returns the base + flags over-cap (no truncation of the base).
- [x] T005 [P] [W2] [US1] [python-pro] [tests/test_dynamic_prompt_generate.py] FAILING tests for `DynamicModelPromptGenerator` (US1 acceptance): static base VERBATIM in output (SC-003); two seeded profiles → measurably different prompts (SC-002); output within budget / under hard cap (SC-001); null jacket → `recommended_tier`+`budget_tokens`, no profile → CORE, neither raises (SC-005); deterministic for fixed inputs (SC-008); conforms to `BasePromptBuilder` (`generate(objective, tools, context)` + `get_token_estimate`) — drop-in; empty objective rejected. Profiles seeded via an injected store/fixture (no live model).

### Implementation (W3) — pure modules, parallel

- [ ] T006 [P] [W3] [US1] [python-pro] [promptchain/prompts/tiers.py] Implement `PromptTier` (CORE/EXTENDED/TINY), `OptionalModule` dataclass, the per-tier module registry, `select_tier(profile)` (D2 thresholds; no-profile→CORE), `modules_for_tier(tier)` → T002 green.
- [ ] T007 [P] [W3] [US1] [python-pro] [promptchain/prompts/family.py] Implement `family_of(model_id)` (strip `provider/`, match stems, `default` fallback) and `adapt_format(parts, family)` (FORMAT-only per-family variants over `default`) → T003 green.
- [ ] T008 [P] [W3] [US1] [python-pro] [promptchain/prompts/budget.py] Implement `measure(text)` (reuse tiktoken `cl100k_base`, `len//4` fallback) and `fit_to_budget(...)` (drop optional by `drop_priority`, never the base, over-cap flag) → T004 green.

### Implementation (W4) — the generator (depends on W3)

- [ ] T009 [W4] [US1] [python-pro] [promptchain/prompts/model_dynamic.py] Implement `DynamicModelPromptGenerator` (`BasePromptBuilder`): `__init__(*, model=None, store=None, base_generator=None, store_path=None)`; `generate(objective, tools, context=None, *, model=None)` → resolve model (kwarg→ctor→None), load the F2 profile via the store (`get_profile`; tolerate None), `select_tier`, compose the static base from `base_generator` (default `DynamicTUIPromptGenerator`) VERBATIM, `adapt_format` for the family, attach `modules_for_tier`, then `fit_to_budget` (effective budget = `min(jacket.budget_tokens or 1000, 1000)`, hard cap 1500); `get_token_estimate`. Null jacket → `recommended_tier`+`budget_tokens`; no profile → CORE. Deterministic. → T005 green.
- [ ] T010 [W4] [US1] [python-pro] [promptchain/prompts/__init__.py] Export `DynamicModelPromptGenerator` + `PromptTier`, `select_tier`, `family_of` from `promptchain.prompts` per contracts/generator-api.md (additive to `__all__`; do not remove existing exports).

**Checkpoint**: US1 green — per-model, budget-compliant prompts with the static base intact; F3 is independently usable as a `BasePromptBuilder`.

---

## Phase 4: User Story 2 — Toolshim for non-tool-calling models (P2)

**Goal**: read `tool_mode` from the jacket; shim modes render a `<tools>` JSON-in-text protocol +
plain-text tool history; native (or absent) passes tools through unchanged.

**Independent Test**: a jacket with `tool_mode=shim_prompt`/`shim_interpreter` → `<tools>` block +
plain-text history; `native`/absent → no shim block, native passthrough; the additive
`Jacket.tool_mode` field round-trips through `to_dict`/`from_dict` and F2's existing tests stay green.

### Tests (W5) — write FIRST, must FAIL

- [ ] T011 [P] [W5] [US2] [python-pro] [tests/test_profiler_jacket_toolmode.py] FAILING test: `Jacket` accepts an optional `tool_mode` (default `None`); `to_dict`/`from_dict` round-trip it; a jacket dict WITHOUT `tool_mode` still loads (backward-compat) — asserting the field is additive + optional.
- [ ] T012 [P] [W5] [US2] [python-pro] [tests/test_dynamic_prompt_toolshim.py] FAILING tests: `resolve_tool_mode(jacket)` → `jacket.tool_mode` or `native` (None/absent→native); `render_tools_block(tools)` emits the `<tools>` JSON-in-text protocol enumerating tools (shim modes); `serialize_history_plaintext(history)` turns native tool-call objects into readable plain text; and `generate()` with a shim jacket includes the `<tools>` block while a native jacket does not (SC-004, mutually exclusive).

### Implementation (W6)

- [ ] T013 [P] [W6] [US2] [python-pro] [promptchain/profiler/jacket.py] Add OPTIONAL `tool_mode: Optional[str] = None` to the `Jacket` dataclass + `to_dict` (emit it) + `from_dict` (`d.get("tool_mode")`). No change to `derive_jacket`/math; `schema_version` unchanged → T011 green; F2's 66 tests stay green.
- [ ] T014 [P] [W6] [US2] [python-pro] [promptchain/prompts/toolshim.py] Implement `resolve_tool_mode`, `render_tools_block` (`<tools>` per contracts/prompt-layout.md), `serialize_history_plaintext` → T012 (helpers) green.
- [ ] T015 [W6] [US2] [python-pro] [promptchain/prompts/model_dynamic.py] Wire toolshim into `generate()`: when `resolve_tool_mode` is a shim mode, render the `<tools>` block (as the tool inventory, replacing the native `AVAILABLE TOOLS`/`MCP TOOLS` blocks) and expose plain-text history serialization; native keeps the existing passthrough → T012 (generate) green. (depends on T013, T014, T009)

**Checkpoint**: US2 green — non-tool-calling models get working tool use; native unchanged; F2 intact.

---

## Phase 5: User Story 3 — Weak-model longevity via Document-&-Clear (P3)

**Goal**: the loop runs `<turn-context>` + goal re-injection each turn; at ~60% context it
Documents-&-Clears (write progress doc → clear → resume); ≥10 task-turns before reset; escalate
only on stall (respecting `jacket.escalate`); lossy `HistorySummarizer` is the fallback.

**Independent Test**: with a fake context-usage signal + a temp dir — `<turn-context>` present each
turn; crossing the threshold writes the progress doc + clears + resumes from it; ≥10 turns
sustained before a reset; escalation only on a stall and only when `jacket.escalate`; doc-path
unavailable → lossy fallback.

### Tests (W7) — write FIRST, must FAIL

- [ ] T016 [P] [W7] [US3] [python-pro] [tests/test_dynamic_prompt_longevity.py] FAILING tests: `build_turn_context(goal, turn=...)` emits a `<turn-context>` block re-injecting the goal (FR-011); `DocumentAndClear.should_compress(usage)` true at ≥`compress_at` (~0.60); `document_and_clear(working_dir, state)` writes `PROGRESS.md`/`todo.md` (plan/decisions/progress), clears working context, returns the doc-seeded resumed history (temp dir) (FR-012); a simulated run sustains ≥10 task-turns before a reset (FR-013); `is_stalled`/`should_escalate` escalate ONLY on a no-progress stall AND only when `jacket.escalate` (no escalate when forbidden); when `working_dir` is not writable it falls back to `HistorySummarizer` (FR-014).

### Implementation (W8)

- [ ] T017 [W8] [US3] [python-pro] [promptchain/prompts/longevity.py] Implement `build_turn_context(...)`, `DocumentAndClear` (`should_compress`, `is_stalled`, `should_escalate` respecting `jacket.escalate`, `document_and_clear` writing the progress doc per contracts/prompt-layout.md + clear/resume, lossy `HistorySummarizer` fallback). Pure decisions deterministic; only `document_and_clear` does I/O → T016 green.
- [ ] T018 [W8] [US3] [python-pro] [promptchain/prompts/__init__.py] Export `DocumentAndClear`, `build_turn_context` (additive to `__all__`).

**Checkpoint**: US3 green — weak models sustain long tasks via Document-&-Clear; the loop owns longevity.

---

## Phase 6: A/B eval set + offline live smoke (SC-007, D7)

**Goal**: a tiny weak-vs-static A/B harness — N=5 programmatically-scored coding tasks × 2 budget
tiers; deterministic scoring/aggregation unit-tested with a fake model; the real "weak model
improves vs the static base alone" claim shown by an offline live smoke (LAN ollama).

### Tests (W9) — write FIRST, must FAIL

- [ ] T019 [P] [W9] [python-pro] [tests/test_dynamic_prompt_eval.py] FAILING tests for the A/B harness: the eval set has N=5 `EvalTask`s each with a deterministic `check(output)`; the runner scores each task×arm (`f3` vs `static_base`) with an INJECTED fake model (scripted pass/fail), aggregates per-arm completion rate, and computes `delta = f3 − static_base`; aggregation is deterministic.

### Implementation (W10)

- [ ] T020 [P] [W10] [python-pro] [promptchain/prompts/eval_ab.py] Implement `EvalTask`/`EvalArm`/`EvalResult`/`EvalReport`, the 5-task set, and the runner (inject a model runner; `f3` arm uses `DynamicModelPromptGenerator`, `static_base` arm uses `DynamicTUIPromptGenerator`) → T019 green.
- [ ] T021 [W10] [python-pro] [specs/014-dynamic-prompt-layer/scripts/ab_smoke.py] Add the offline live-smoke script (`--weak <model>`): run the A/B against a real LAN ollama weak model, print per-arm completion + delta. No secrets; `OLLAMA_API_BASE`/`PYTHONPATH` per quickstart.

**Checkpoint**: A/B harness green offline; live smoke ready to run.

---

## Phase 7: Polish & Verification

- [ ] T022 [W11] [python-pro] [tests/] Run the full F3 suite `python -m pytest tests/test_dynamic_prompt_*.py tests/test_profiler_jacket_toolmode.py -q` (all green) AND the no-regression suite `python -m pytest tests/test_profiler_*.py tests/test_transcript_*.py -q` (F1+F2 still green, esp. F2's 66).
- [ ] T023 [W11] [python-pro] [specs/014-dynamic-prompt-layer/] Run the offline live smoke (`ab_smoke.py` vs a weak LAN ollama model) and record the per-arm completion + delta in quickstart.md / the memory-bank checkpoint (SC-007 evidence). If the LAN model is unreachable, note it and rely on the deterministic harness.
- [ ] T024 [W11] [python-pro] [memory-bank/] Update the memory-bank (progress.md + activeContext.md): F3 complete + green; summarize per-US deliverables, the one additive F2 field (`Jacket.tool_mode`), and the SC-007 smoke result.

**Checkpoint**: F3 fully green, no F1/F2 regression, evidence recorded → ready to merge `--no-ff` into `epic/adaptive-prompting`.

---

## Dependencies & execution order

- **W1** (setup) → **W2** (US1 tests, FAIL) → **W3** (US1 pure impl) → **W4** (US1 generator) →
  **W5** (US2 tests, FAIL) → **W6** (US2 impl) → **W7** (US3 tests, FAIL) → **W8** (US3 impl) →
  **W9** (eval tests, FAIL) → **W10** (eval impl) → **W11** (polish/verify).
- **Story independence**: US1 is the MVP and standalone. US2 wires into US1's generator (T015 dep
  T009). US3 (longevity) is largely independent of the generator. The A/B eval exercises US1 (+US2/3
  when present).
- **The one F2 touch**: T013 adds the optional `Jacket.tool_mode` field — additive, covered by T011,
  F2 tests stay green.

## Parallel execution examples

- **W2**: T002, T003, T004, T005 in parallel (4 distinct test files).
- **W3**: T006, T007, T008 in parallel (tiers / family / budget — distinct files).
- **W5**: T011, T012 in parallel. **W6**: T013, T014 in parallel, then T015 (same-file as T009).
- **W9/W10**: T019 then T020 (eval), T021 sequential (smoke script).

## Implementation strategy

- **MVP = US1 (Phase 3)**: ship a per-model, budget-compliant, static-base-intact generator first;
  it is independently valuable and verifiable offline with seeded profiles.
- **Incremental**: US2 (toolshim) → US3 (longevity) → A/B eval, each test-first, committed per wave
  (full-autonomous sprint cadence), stopping only on a failed red→green gate or for the live smoke.
- **Verify**: every wave's tests FAIL before its impl; final no-regression run guarantees F1+F2
  intact; the offline live smoke is the SC-007 evidence.
