# PRD — Adaptive Prompting System ("the harness fits the model")

**Status:** Draft → ready for Spec Kit decomposition
**Branch at origin:** `feat/test-loop-chain`
**Owner doc set:** `research/foundation/architecture/00-06` + `research/foundation/foundation-prompt.md`
**Execution pipeline:** this PRD → `specify` (spec.md) → `/plan` (plan.md) → `/tasks` (tasks.md) → **dev-kid** autonomous build
**Ephemeral marker:** keep until all three pillars (F1–F3) ship to `main`; then graduate the design docs into `specs/` and archive this PRD.

> **Purpose of this PRD:** a single consolidating brief that recaps every architectural
> decision, the design, AND what's already built (the loops), so Spec Kit has all the
> material to generate spec.md / plan.md / tasks.md without re-deriving anything.

---

## 1. Context / Origin (2026-06-27)

This session designed a complete **adaptive prompting system** for PromptChain whose
**north star is "the harness fits the model," not "the model fits the harness."** We
produced a full design (`research/foundation/architecture/00-06`) and a grounded
foundation prompt (Phase 4), then **built the substrate** (iterate-until-tests-pass loops +
tool-forge) and wired it + the static foundation into the TUI.

The **core adaptive modules remain design-only**. The design surface is too large to
hand-build inline, so it runs through the repo's **spec-driven pipeline**
(`.specify/` + Constitution → Spec Kit → dev-kid). This PRD feeds that pipeline.

### Honest build status (so dev-kid does NOT rebuild what exists)

| Component | Design doc | Status |
|---|---|---|
| Tool-forge loops — `MicroPromptChain` (A), `RalphChain` (B), `AutoResearch` | 05, 06 | ✅ **BUILT + live-verified** (`14bd138`, `8492b51`, `502955a`) |
| `run_sync` event-loop-safety guard (`run_coro_blocking`) | — | ✅ **BUILT** (`cba0090`) |
| Loop tools registered for the TUI (`build_until_tests_pass`, `multi_agent_build`) | — | ✅ **BUILT** (`95a6bc0`) |
| Grounded foundation prompt wired into `TUI_FOUNDATION_PROMPT` (static base) | Phase 4 | ✅ **BUILT** (`65b07d2`) |
| **SIO output integration** (JSONL transcript emitter + SIO harness adapter) | **03** | ❌ **NOT built → F1** |
| **Model Profiler** (IRT/CAT probe → ability θ → Ω fit + model×prompt "jacket") | **01, 02** | ❌ **NOT built → F2** |
| **Dynamic prompt layer** (per-model tiering, family adapters, toolshim, compression, Document-&-Clear) | **04 + constraint D** | ❌ **NOT built → F3** |

The three ❌ rows are the work this PRD specifies. F1 is locked **first** (the user's
explicit "deal with PromptChain's output first" instruction — everything downstream mines it).

---

## 2. North star & the two-layer model

- **Inversion thesis:** the prompt/harness adapts to each model's strengths and limits, rather
  than forcing every model through one Claude-shaped prompt.
- **Two layers:** (1) a **static base** = the model-agnostic foundation (parity floor; built +
  wired); (2) a **dynamic split** = per-model additions chosen at `generate()` time by the
  **Model Profiler**, driven by the formulas. F2 + F3 build layer 2.
- **The four constraints (doc 00):** **A** token economy · **B** heterogeneous models /
  big-brother planner · **C** cross-model normalization / family adapter · **D** weak-model
  longevity / Document-&-Clear.

---

## 3. Architectural decisions & design rationale (recap — the "why")

These are the decisions made this session that the specs must honor:

1. **Inversion is the unifying thesis.** Every module exists to make the harness fit the model
   (tiering, adapters, toolshim, the profiler). Don't regress to one-prompt-fits-all.
2. **Two loops, not one (the micro-agent fork).** The repo's `micro-agent` is a fork that
   *added* the MA loop on top of the original. We reproduce **both**: engine **A**
   (`MicroPromptChain`) = the original single-agent generate→test→repair; engine **B**
   (`RalphChain`) = the fork's multi-agent state machine (librarian→artisan→critic→testing).
   A is the substrate; B explodes A's generator into the staged pipeline.
3. **The Ralph founding principle = fresh context every iteration.** Each outer pass discards
   accumulated context; only the structured test result (+ critic notes) threads forward. This
   is *why* weak/local models converge — each pass is a clean, bounded task, not a growing
   transcript. (Verified in the fork's `ralph-machine.ts` + `iteration-manager.ts`.)
4. **"Never invent a tool" was re-scoped, not deleted.** Ground everyone (never fake a tool
   call or its output); the **planner/strong tier may BUILD** a real tool via the chain-builder/
   loop. The loop's **test-pass IS the anti-hallucination guard** — only proven code/tools exit.
   So tool authoring is gated by the **test**, not the model role (even weak models can forge a
   tool by brute-force iteration). The loops ARE the Karpathy/Ralph tool-forge.
5. **Language-agnostic by exit code.** The universal pass signal is the test command's exit code
   (0 = pass); swap `image` + `test_command` for python/rust/go/node. Sandbox via
   `DockerExecutor`; `LocalExecutor` for trusted/CI + docker-less unit tests.
6. **Heterogeneous models per role (Constraint B).** `RalphChain` takes a model per stage
   (librarian cheap, artisan strong, critic medium) — the fork's table. This is the concrete
   "big-brother" mechanism and the seed for F2's tiering.
7. **Verification must be REAL (anti-lying).** A loop that fakes a PASS is worse than none.
   We proved it with **independent re-runs** (separate subprocess) + **negative controls**
   (wrong code must FAIL), live against a real model. dev-kid's "independently re-run the test"
   discipline carries into F2's probe runner.
8. **Async-safety is a first-class concern.** Tools are `async def` (awaited natively);
   `run_sync` is guarded by `run_coro_blocking` (detects a running loop → offloads to a
   worker-thread loop) so the "asyncio.run inside a running loop" crash can never occur.
9. **Sandbox hygiene.** `PYTHONDONTWRITEBYTECODE=1` in both executors (a same-size source
   rewritten within one second must not re-use a stale `.pyc`). Found + fixed live.
10. **SIO integration = emit, don't reuse.** PromptChain emits JSONL transcripts SIO can mine;
    we do NOT import SIO's code and do NOT depend on MLflow (which may be dropped). Lock the
    output schema FIRST (F1) because F2 consumes it.
11. **Model Profiler = psychometrics.** IRT 3PL `P_i(θ)=c_i+(1−c_i)·σ(a_i(θ−b_i))`, CAT for
    cheap adaptive probing, ability **θ** per skill (EAP/WLE), Fisher info; a fit score **Ω**
    and a **two-sided model×prompt "jacket"** grid (which prompt shape fits which model);
    Bradley-Terry/Elo for pairwise prompt comparison; SIO telemetry seeds priors.
12. **Dynamic layer is profiler-driven.** F3 uses F2's profile to pick tier (CORE/EXTENDED/
    TINY), family adapter (Constraint C), toolshim (Goose) for non-tool-calling models, token-
    budget compression (Constraint A), and Document-&-Clear longevity (Constraint D).
13. **Foundation prompt = model-agnostic dense-XML, merged with TUI specifics.** Static base
    (`<objective>/<work_loop>/<tools>/<editing>/<safety>/<response>`) + dynamic tool tail; the
    only runtime slot is `{objective}`; the live AVAILABLE TOOLS block is appended from the real
    registry (never advertises a tool that isn't loaded). ~700-token base (300–1,000 target).
14. **Execution discipline (this PRD).** Big design ⇒ in-repo PRD → Spec Kit → dev-kid.
    **dev-kid is the BUILDER here, not the target** — rewiring dev-kid to call PromptChain's loop
    instead of the TS `ma-loop` is a *separate, later* effort; this PRD does not touch dev-kid or
    the micro-agent fork.
15. **Infographic theme** = `promptchain-dark` (standing default for PromptChain infographics).

---

## 4. What we've built so far — the loops (detailed recap)

The substrate F2/F3 build on. All on `feat/test-loop-chain`; 19 offline tests + a live,
anti-lying smoke (real `qwen3-coder:30b` on the Mac Studio ollama) all green.

- **`MicroPromptChain` (engine A)** — `promptchain/utils/test_loop_chain.py`. Single agent:
  generate `target_file` → run `test_command` in a sandbox → on fail, thread the test output
  back and retry, bounded by `max_iterations`/`max_seconds`. Built on `ExternalLoop` (the outer
  while), `DockerExecutor`/`LocalExecutor` (sandbox, +`write_file`), and a single-instruction
  `PromptChain` (generator). Returns `LoopResult{result, winning_code, iterations, attempts}`.
- **`RalphChain` (engine B, the MA loop)** — `promptchain/utils/ralph_chain.py`. Per iteration:
  librarian (gather) → artisan (write) → critic (review) → testing, then **fresh-context reset**;
  per-role models; entropy breaker (identical failures N× → STAGNATED). Reuses A's sandbox/test
  contract via shared helpers (`extract_code`/`truncate_tail`/`make_generator`/`resolve_executor`).
- **`AutoResearch`** — `promptchain/utils/autoresearch.py`. research → build-until-verified
  (MicroPromptChain) → optional critic gate, as a registerable tool. The "sandboxed exec +
  success-bar check" the external autoresearch system had left as a TODO.
- **`run_coro_blocking`** — the async footgun guard (decision #8).
- **TUI tools** — `promptchain/cli/tools/library/loop_tools.py`: `build_until_tests_pass`
  (A) + `multi_agent_build` (B), async, model from `PROMPTCHAIN_LOOP_MODEL`/arg; registry 32→34.
- **Foundation prompt** — `promptchain/prompts/tui_dynamic.py` `TUI_FOUNDATION_PROMPT` (decision #13).

---

## 5. Feature F1 — SIO output integration (LOCK FIRST)  · design: `03`

**Goal:** every PromptChain run emits a structured **JSONL transcript** SIO ("CO") can mine —
without reusing SIO's code and without depending on MLflow.

**Scope:** a transcript **emitter** on the existing event system (`ExecutionEvent` /
`callback_manager`) → append-only JSONL per run (model, instructions, tool calls + results,
tokens, timings, outcome; one event/line); a **SIO harness adapter** (`sio/harnesses/promptchain.py`
shape) so `sio mine|suggest|flows|search` work on PromptChain sessions; bounded/rotated output
path (Constitution V).

**Acceptance:** a run produces valid JSONL; `sio mine`/`sio search` ingest it; no MLflow dep;
emitter is async-safe and <2% overhead.

---

## 6. Feature F2 — Model Profiler (auto model-capability assessment)  · design: `01`, `02`

**Goal:** assess a model → a capability profile that drives prompt assembly ("inter-harness
model analyst").

**Scope:** a **probe runner** (the "ten isolated trials" per skill — codegen, instruction-
following, tool-calling, long-context, repair — reusing the loops where the probe is build-
until-pass); **IRT/CAT scoring** (3PL, θ per skill via EAP/WLE, Fisher-info CAT to keep probes
cheap); the **Ω fit score** and the **two-sided model×jacket grid** (which prompt shape fits
which model; Bradley-Terry/Elo for pairwise); **SIO-fed priors** from F1 telemetry; persisted
profiles.

**Acceptance:** given a model id → persisted profile {per-skill θ, recommended tier, best
jacket}; the ten-trial probe runs offline against a real local model, reproducibly; math
validated by unit tests against known item parameters.

---

## 7. Feature F3 — Dynamic prompt layer (drive assembly through the formulas)  · design: `04` + constraint D

**Goal:** the generator that **uses F2** to assemble the right prompt per model — the dynamic
half of the two-layer model.

**Scope (each a candidate sub-spec):** **Tiering** (CORE/EXTENDED/TINY by profile, Constraint B);
**family adapter** (format normalization + per-family variants, Constraint C); **toolshim**
(JSON tool-call fallback for non-tool-calling models, Goose); **token-economy assembly**
(budget-cap 300–1,000 tokens, compression, Constraint A); **weak-model longevity** (`<turn-
context>` + goal re-injection, **Document-&-Clear** at ~60% context, Constraint D);
**integration seam** at `DynamicTUIPromptGenerator.generate()` / `BasePromptBuilder`
(`agentic_step_processor.py:1006`) so the static base stays intact and the split is additive.

**Acceptance:** `generate(objective, tools, model)` returns a profile-appropriate prompt
(different tier/adapter/shim per model) within budget; A/B measurably improves weak-model task
completion vs the static base alone.

---

## 8. Constitution alignment (`.specify/memory/constitution.md`)

- **II Observable** — F1 IS observability; F2/F3 expose profiles + chosen tier as machine-readable surfaces.
- **III Test-First / IV Integration** — every feature lands tests-first (the loops set the precedent: offline fakes + real sandbox + anti-lying live checks).
- **V Token Economy** — F3 budget-capped; F1 transcripts bounded/rotated.
- **VI Async-First** — async-first modules + guarded sync wrappers (`run_coro_blocking`).
- **VII Simplicity** — build only F1→F3 as specified; no speculative abstractions.

---

## 9. Success criteria

1. F1 transcripts mineable by `sio` with no MLflow dependency.
2. F2 returns a reproducible capability profile for a real local model from a ten-trial probe.
3. F3 assembles a measurably different, budget-compliant prompt per model, improving a weak
   model's completion rate on a small eval vs the static base.
4. Built loops/foundation untouched (no rebuild); each feature ships test-first per the Constitution.

---

## 10. Execution pipeline (what happens after this PRD)

1. **Spec Kit** — for each feature F1→F3 (in that order), `specify` → `specs/<NNN>-<slug>/spec.md`,
   then `/plan` → `plan.md`, `/tasks` → `tasks.md`. **F1 first.**
2. **dev-kid** — execute the generated `tasks.md` autonomously (tier-escalation loop), with each
   feature's tests as the success bar. (dev-kid is the *builder*; not modified by this PRD.)
3. Land each feature on its own branch, test-first.

---

## 11. References

- Design docs: `research/foundation/architecture/00-design-constraints.md` … `06-test-loop-chain.md`,
  `research/foundation/architecture/README.md`, `research/foundation/foundation-prompt.md`.
- Built substrate: `promptchain/utils/{test_loop_chain,ralph_chain,autoresearch}.py`,
  `promptchain/cli/tools/library/loop_tools.py`, `promptchain/prompts/tui_dynamic.py`.
- Spec-kit: `.specify/templates/*`, `.specify/memory/constitution.md`; existing specs `001–011`.
- Prompt corpus + synthesis: `research/system-prompts/MANIFEST.md`,
  `research/foundation/{dossier,best-foundations,gap-analysis}.md`.
- Commits (this session): `14bd138` Micro · `502955a` AutoResearch · `8492b51` Ralph ·
  `cba0090` async guard · `95a6bc0` TUI tools · `65b07d2` foundation wiring.
