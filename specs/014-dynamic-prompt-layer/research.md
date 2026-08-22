# Phase 0 Research — Dynamic Prompt Layer (F3)

All "NEEDS CLARIFICATION" from the technical context are resolved below. F3 is the assembly layer
that **reads** F2's frozen profile/jacket; the decisions here are about *how F3 maps that profile
into an assembled prompt*, not about re-opening F2's math or F1's emitter.

---

## D1 — `generate()` signature reconciled with `BasePromptBuilder`

**Decision**: Ship `DynamicModelPromptGenerator` that **binds its model + profile source at
construction** and conforms to `BasePromptBuilder.generate(objective, tools, context=None)`
verbatim (so it is a drop-in at `agentic_step_processor.py:1006`). Add an optional **keyword-only**
`model=` override on `generate()` for direct use and tests. The spec/brief shorthand
`generate(objective, tools, model)` maps to `generate(objective, tools, context=None, *, model=None)`
in code, resolving the model as: explicit `model=` kwarg → construction-time model → `None`
(→ safe default tier).

**Rationale**: The protocol's third positional argument is `context`, not `model`; a positional
`model` would silently bind to `context`. Binding the model at construction matches how a processor
/ agent is configured for a specific model and keeps the drop-in seam unchanged. Keyword override
keeps unit tests ergonomic (one generator, many models) without violating the protocol.

**Alternatives considered**: (a) add a positional `model` to `BasePromptBuilder` — rejected: a
breaking change to a frozen 011-era contract used by other builders. (b) a separate non-protocol
method `generate_for_model(...)` — rejected: duplicates the surface and complicates the drop-in.

---

## D2 — Prompt-tier taxonomy (CORE / EXTENDED / TINY) is F3-owned and separate from the jacket band

**Decision**: F3 owns a **prompt-strength** axis with three tiers — **EXTENDED** (strong models,
fuller guidance + optional modules), **CORE** (mid models, the agnostic base + essentials),
**TINY** (weakest/smallest models, a reduced protocol; may swap to a simpler loop protocol). Map
from the profile's **capability `C ∈ [0,1]`** with documented thresholds (default: `C ≥ 0.66 →
EXTENDED`, `0.33 ≤ C < 0.66 → CORE`, `C < 0.33 → TINY`), with `recommended_tier` as a tiebreak/override
hint. The jacket's own `tier` (`lean/standard/rich/max-rich`) is a **budget/mode** axis and is NOT
the prompt tier; F3 uses the jacket's `budget_tokens` for the budget axis only.

**Rationale**: Goose precedent (Constraint B) distinguishes `system.md` vs `tiny_model_system.md`
— a prompt-strength choice, not a token-budget band. Keeping the two axes separate avoids
conflating "how much guidance" with "how many tokens" and keeps the mapping testable with seeded
capabilities. Thresholds are constants so the mapping is deterministic (FR-002/FR-015).

**Alternatives considered**: reuse the jacket's `lean/standard/rich/max-rich` directly as the
prompt tier — rejected: those bands encode budget+mode, conflate axes, and give four tiers where
the design calls for three strength tiers with a distinct TINY protocol.

---

## D3 — `family` derived from `model_id`; adapter changes FORMAT only

**Decision**: Derive the model **family** by parsing `model_id` (strip a `provider/` prefix, match
known stems): `anthropic`/`claude*` → anthropic, `gpt*`/`o1*`/`openai*` → openai, `gemini*` →
google, `qwen*` → qwen, `llama*`/`meta*` → llama, else → **`default`**. The family adapter adjusts
**format only** (e.g., section delimiters / role-framing conventions a family parses best) layered
over the agnostic base; it MUST NOT change the base's semantic content. Unknown family → `default`
(≈ the agnostic core).

**Rationale**: opencode `routing.ts` keys format variants by family over a `default.txt`; LiteLLM
normalizes transport, not content, so content portability is on us (Constraint C). Deriving family
from `model_id` keeps F3 a pure reader (no new F2 field) and is trivially testable. Format-only
guarantees the static-base-intact invariant (SC-003).

**Alternatives considered**: add a `family` field to the F2 jacket — rejected: unnecessary F2 surface
growth; `model_id` already carries the family. Per-model (not per-family) variants — rejected:
explodes config; family-level + `default` fallback is the documented design.

---

## D4 — `tool_mode` source: optional Jacket field, default native

**Decision**: Read `tool_mode ∈ {native, shim_prompt, shim_interpreter}` from the jacket via a new
**optional** field `Jacket.tool_mode: Optional[str] = None` (added to `to_dict`/`from_dict`). When
absent/null → **native** (FR-010). This is the single additive-optional touch to F2 (no
`schema_version` bump, no change to `derive_jacket`'s math — it simply leaves `tool_mode` unset).

**Rationale**: The handoff/PRD explicitly sanction "additive + optional only" profile fields. The
brief says the jacket carries `tool_mode`; the built F2 `Jacket` lacks it, so F3 adds the field so
it round-trips through the store and so tests/jackets can set a shim mode. Populating it from probe
detection ("does this model call tools natively?") is a **future F2 refinement** — out of F3's
scope (don't re-open F2's probe). F3 only consumes it.

**Alternatives considered**: (a) derive `tool_mode` from the `tool_call_reliability` skill — rejected:
reliability ≠ API support; a model can support native calls yet be unreliable. (b) read it only from
a raw dict, never touch the dataclass — rejected: `Jacket.from_dict` drops unknown keys, so it would
never round-trip; the dataclass field is the clean, contract-respecting path.

---

## D5 — Token budget: measure with tiktoken, drop optional modules, never the base

**Decision**: Budget target **300–1,000 tokens**, hard cap **~1,500**. Measure the assembled prompt
with the existing tiktoken estimator (`len//4` fallback already present). When over the effective
budget (the smaller of the jacket's `budget_tokens` clamp and the 1,000/1,500 caps), **drop optional
modules in a documented priority order** (examples first, then extended-guidance, then the toolshim
text only if a non-shim path is viable) until it fits. The **static base + objective + live tool
inventory** are the parity floor and are **never dropped**; if even the floor exceeds the hard cap,
surface the over-cap condition rather than corrupt the prompt.

**Rationale**: Constraint A — one-line imperative rules, examples are optional, enforce the cap
(don't hope). Reusing the existing estimator avoids a new dependency and matches
`DynamicTUIPromptGenerator.get_token_estimate`. The never-drop-the-floor rule guarantees SC-003 even
under budget pressure.

**Alternatives considered**: summarize/compress module text to fit — rejected: lossy + nondeterministic
at assembly time; dropping whole optional modules is deterministic and testable. A hard truncation of
the final string — rejected: would cut the static base and break parity.

---

## D6 — Document-&-Clear seam: a policy object the loop calls (no loop rewrite)

**Decision**: Implement US3 as a self-contained `longevity` module: pure decision functions
(`build_turn_context(goal, …)` → the `<turn-context>` block with goal re-injection; a threshold
check against the jacket's `compress_at` ~0.60; turn-count + stall tracking; an escalate decision
respecting `jacket.escalate`) plus one I/O method `document_and_clear(working_dir, state)` that
writes the **progress doc** (`PROGRESS.md`/`todo.md`), clears the working context, and returns the
resumed (doc-seeded) history. The existing loops (`ExternalLoop`, `RalphChain`) call this policy;
`RalphChain` already resets context per pass and has stagnation detection, so the policy plugs into
that pattern. Lossy `HistorySummarizer` is the documented **fallback** when the doc path is
unavailable (FR-014).

**Rationale**: Constraint D — the **loop** owns longevity, not the model's tokens. A policy object
keeps US3 additive (no rewrite of loop internals), makes every decision a pure, offline-testable
function (fake context-usage signal + temp dir), and reuses the already-built reset/stagnation +
summarizer machinery rather than rebuilding it (Principle VII).

**Alternatives considered**: bake Document-&-Clear directly into `ExternalLoop.run` — rejected:
couples the policy to one loop and is harder to test in isolation. Rely solely on
`HistorySummarizer` (lossy) — rejected: the design makes Document-&-Clear primary and lossy
compaction only the fallback.

---

## D7 — The A/B eval set (the PRD-flagged open item, SC-007)

**Decision**: Define a **tiny** eval set: **N = 5 scoped, deterministically-checkable coding tasks**
(e.g., "write a function that returns the Nth Fibonacci number", "parse this CSV line", each with a
programmatic pass check), run at **two token tiers** (a tight budget and a generous budget). The A/B
harness runs each task twice — once with the F3 per-model prompt, once with the static base alone —
and reports per-arm completion rate. **Deterministic parts** (task set, scoring, aggregation) are
unit-tested with a **fake model** (scripted pass/fail). The **real** "weak model improves vs static
base" claim is demonstrated by an **offline live smoke** against a weak LAN ollama model (mirroring
F2's offline live smoke), not asserted in CI.

**Rationale**: SC-007 needs a real model to show a real win, which can't be a deterministic unit
test; F2 set the precedent (deterministic core in CI + one offline live smoke for the live claim).
Keeping the set tiny + programmatically scored honors YAGNI and keeps the smoke cheap and offline.

**Alternatives considered**: a large benchmark suite — rejected: scope/cost, not needed to show a
directional win. An LLM-judge scorer — rejected: nondeterministic; programmatic pass checks are
reproducible.

---

## Resolved unknowns summary

| Unknown | Resolution |
|---|---|
| How does `generate()` get the model given the protocol? | D1 — model bound at construction + optional `model=` kwarg |
| CORE/EXTENDED/TINY vs jacket bands | D2 — F3-owned capability→tier mapping; jacket band is budget/mode only |
| Where does `family` come from? | D3 — derived from `model_id`; `default` fallback; format-only |
| Where does `tool_mode` come from? | D4 — optional `Jacket.tool_mode`; default native |
| How is the budget enforced? | D5 — tiktoken measure + drop optional modules; never the base |
| Where does Document-&-Clear live? | D6 — a policy object the existing loops call; lossy fallback |
| What is the A/B eval set? | D7 — 5 programmatically-scored tasks × 2 budgets; live smoke for the real win |
