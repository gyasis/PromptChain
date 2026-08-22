# Feature Request — Durable, Verified TUI Agent Loop

**Status:** Draft / Proposed
**Date:** 2026-07-06
**Author:** Gyasi Sutton (design session w/ Claude)
**Area:** TUI agent · `AgenticStepProcessor` · `ExternalLoop`
**Related code:** `promptchain/utils/agentic_step_processor.py`, `promptchain/utils/external_loop.py`, TUI session store (`~/.promptchain/sessions/`, SQLite + JSONL)

---

## 1. Summary

Improve the TUI agent by upgrading its `AgenticStepProcessor` (ASP) reasoning loop along two axes:

1. **Gap 1 — Durability & non-degradation over long sessions.** ASP is a *finite* inner loop (`max_internal_steps`, default 5) whose context *accumulates* and whose state is in-RAM. Long objectives hit the step ceiling and/or degrade from context rot, and a crash loses everything. Add an **outer loop (internal, `ExternalLoop`-shaped) that restarts ASP with a fresh context seeded from a durable state artifact.**
2. **Gap 2 — A quantifiable "task complete" gate.** ASP stops when the *model declares itself done*. Replace that with an **external, falsifiable verification gate** as the loop's stop condition — do not trust the doer's self-report.

**Explicitly deferred:** an autonomy-slider / constraint contract (originally "gap 3") — noted as future work, not in this feature.

### 1.1 Governing principle — PRINCIPLES, not implementation (our-way mandate) · READ FIRST

We are **not** porting the Pi framework (or any external harness). Pi, Ralph, Karpathy/AutoResearch, Observational Memory, `pi-fork`, etc. are **sources of principles only**. Every principle we adopt must be realized in **PromptChain's own primitives** — `PromptChain` static chains, `AgenticStepProcessor` (ASP), `ExternalLoop`, `ExecutionHistoryManager`, `AgentChain`, `chainbreakers`, and `Callable` function-steps — in the idiomatic PromptChain way. No Pi code, no Pi extension model; just the ideas, expressed as our chains and loops.

Where a principle doesn't fit our current primitives, we **change the primitive deliberately** (e.g. how ASP bounds a step, how `ExternalLoop` reseeds context) rather than bolt on a foreign harness. **§13 maps each principle to its PromptChain realization and lists the concrete gaps this requires us to close.** Context management is the core of this work — and closing it requires our loops to **erase and reseed** context between outer iterations, not only compress it in place (see §13 Gap A).

---

## 2. Background — findings from the loop comparison

We compared ASP against two well-known agentic loops to see which principles we already have and which are missing.

### Altitude correction (the crux)
The three loops are **not at the same level**:

- **ASP** = an **inner loop**. One node in a chain: `LLM → tool? → observe → objective complete? → loop/exit`, bounded by `max_internal_steps`.
- **Ralph loop** (Huntley) = an **outer loop**. A shell `while` around a *fresh agent process* each iteration; state lives on disk (files, TODO, git).
- **Karpathy's loop** (AutoResearch) = an **outer loop** too, but a *scientific* one: edit → run a **time-boxed, measured** experiment → **keep/discard on a real metric** → repeat.

ASP's true peer for those is **`ExternalLoop`** (`max_iters` / `max_seconds` / `breakers`) wrapping ASP — not ASP alone.

> **Caveat on Karpathy's loop:** its concrete machinery is domain-specific — it optimizes ML training scripts against a built-in metric (loss/accuracy). The TUI agent is *not* doing ML experiment search, so **we borrow only the transferable principle** — *an external, falsifiable gate + keep/discard + never trust self-report* — **not** the AutoResearch optimizer wholesale.

### Comparison

```
DIMENSION            ASP                    RALPH (Huntley)        KARPATHY (AutoResearch)
─────────────────    ───────────────────    ───────────────────    ───────────────────────
loop location        inner (in-context)     outer (shell)          outer (script-driven)
context per iter     ACCUMULATES            RESETS (fresh)         fresh-ish per experiment
                     (+summarize/state)     every iteration
state / memory       in-RAM, ephemeral      filesystem+TODO+git    training script + logs
                     (never persisted)      (durable)              (durable)
stop condition       LLM self-declares      PRD items all done     stopping-criteria + metric
verification gate    CoVe (pre-exec tool    none built-in          FALSIFIABLE: measured
                     sanity check)          (git diff = audit)     keep/discard on a metric
autonomy control     max_internal_steps     infinite / manual kill autonomy slider + time-box
```

### Shared principles (already present in ASP)
- ReAct primitive: act → observe → decide continue/stop.
- Bounded iteration (`max_internal_steps`).
- Tool use inside the loop (MCPHelper + registered functions).
- A *pre-execution* verification wrinkle (CoVe) that neither Ralph nor Karpathy has — ASP verifies a tool call's assumptions/risk **before** running it.

### The gaps (relevant to the TUI)
- **Gap 1 (vs Ralph):** no fresh-context reset, no durable on-disk state. ASP fights context rot by *compressing* (summarization, `use_state_management`), which is the opposite of Ralph's *discard-and-rebuild-from-disk*. Long objectives still degrade; crashes lose state.
- **Gap 2 (vs Karpathy):** no falsifiable outcome gate. ASP's stop = "the model thinks it's done." CoVe checks whether a *tool call looks safe*, not whether the *outcome actually passed a measurable bar*.

---

## 3. Scope

| Item | In / Out |
|---|---|
| Gap 1 — durable, non-degrading outer loop around ASP | **IN** |
| Gap 2 — quantifiable verification gate as stop condition | **IN** |
| Gap 3 — autonomy slider / constraint contract | **OUT (future)** |
| Import Karpathy's ML-experiment optimizer wholesale | **OUT** (borrow principle only) |

---

## 4. Gap 1 design — durable, non-degrading loop

### 4.1 Two problems hide inside "gap 1" — do not conflate them
- **Problem A — the step ceiling.** ASP breaks out after `max_internal_steps`. A restart loop **fully solves this.**
- **Problem B — gap 1 proper: context rot + durability.** Context accumulates and degrades; state is ephemeral.

**Restarting ASP raises the ceiling but does not, by itself, solve B.** It solves B *only* with the two additions below.

### 4.2 Why a naive restart fails
Everything hinges on **what is handed from iteration N to N+1**:
- **Re-inject old history** → context keeps climbing across restarts → rot gets *worse*.
- **Fresh but stateless** → the fresh iteration has amnesia → re-does or forgets prior work.

Ralph's real trick is **"restart with a fresh context that bootstraps from durable external state."** The restart is necessary but not sufficient — the **state handoff** is the trick.

### 4.3 The design
```
ASP iter N  --> writes a distilled STATE artifact (done / next / key facts)
   --break (step ceiling or verifier says "not done")-->
ASP iter N+1  starts FRESH, seeded ONLY with (objective + state artifact)
```

**Two rules make the restart actually solve gap 1:**
1. **Reset to a genuinely fresh context.** Construct a *new* ASP (or clear `internal_history`) seeded with `objective + compact state` — **not** the old message list. (A fresh ASP already starts with empty `internal_history`, so this is the natural default; the discipline is *not* to feed old history back in.)
2. **Externalize progress into a state artifact** the fresh iteration reads.

### 4.4 "Internal" is fine — the process boundary was incidental
Ralph uses a fresh OS process only because it's a shell loop. What matters is that the **message array sent to the LLM starts clean** — fully achievable in-process. So "outer loop but still internal" is legitimately Ralph-shaped; no subprocess required.

### 4.5 In-RAM vs on-disk splits gap 1 in half
- Handoff state **in RAM** → solves **context rot** (the long-session-degrades half). ✅
- Handoff state **persisted to disk** → also solves **crash durability**. ✅

**Leverage what exists:** the TUI already persists sessions to **SQLite + JSONL** (`~/.promptchain/sessions/`; current branch `fix/tui-history-persistence`). This is a ready-made durability substrate for the state artifact — the durable half of gap 1 does **not** need new storage infrastructure.

### 4.6 Reuse existing machinery for the handoff
ASP already ships `use_state_management` (structured state, claimed 70–80% token reduction) and `enable_summarization`. Today these compress context *in place*. **Repurpose them as the reset handoff:** at each break, dump ASP's structured state → seed the fresh ASP from it. The compression tool becomes the continuity tool (low new invention).

**Handoff FORMAT (research upgrade #1) — a deterministic observation *list*, not a free-form summary.** Per the research brief (`docs/research/true_agent_autonomy.md`, §4 P4 + the primary source): free-form LLM summarization compounds into a *"summary of a summary"* decay across cycles, whereas a **deterministic list of atomic observations** (each = event / decision / bug / constraint) does not decay. So the state artifact should be a structured, append/prune-able **list of observations** (later distilled into stable *reflections*) — the Observational-Memory (OM) pattern — **not** a re-summarized prose blob.

### 4.7 Alternative architecture (Option B) — continuous single-session + deterministic compaction
The research surfaced a *second* valid school (the "True Agent Autonomy" video's own design) that is the **inverse** of the fresh-reset approach above:

- **Option A (default, this design):** *reset to stay sharp* — fresh ASP per iteration, bootstrapped from the state artifact (classic Ralph).
- **Option B (alternative):** *never reset, but compact deterministically* — one continuous session whose context never resets, kept alive by an Observational-Memory layer (observer → reflector → dropper) that distills the running history into a bounded observation list *in place*.

Both defeat context rot; they differ on WHERE the boundary sits. Option A is simpler to build on ASP (a new instance is already fresh) and gets crash-durability for free (the artifact is persisted). Option B avoids re-seeding cost but needs the full OM machinery and an answer to the **in-turn compaction problem** (a single ASP step whose own tool output overflows the budget → the cut lands mid-step). **Recommendation: build Option A first; harvest Option B's OM as the *source of the handoff format* (upgrade #1), not a full alternative for v1.**

---

## 5. Gap 2 design — quantifiable verification gate (the stop condition)

The stop condition must answer **"is this task actually complete?"** with a real check — **not** "is a breaker function callable" and **not** "the doer says finished."

### 5.1 The hard part is sourcing the metric, not the gate mechanism
Karpathy/AutoResearch get a metric for free (ML loss/accuracy) and force a falsifiable bar up front. For a general TUI task, **there is no metric lying around — we must manufacture the measurable bar.**

### 5.2 Two rules that make the gate trustworthy
- **Rule 1 — the acceptance bar is an INPUT, defined before the loop runs.** Never let the doer author its own passing criteria (Goodhart / grading its own homework). Acceptance criteria enter the loop as config.
- **Rule 2 — prefer a real check over a judge; fall back only when forced; log which tier certified it.**

### 5.3 The Verifier ladder (climb as high as the task allows)
```
1. EXECUTABLE      run tests / cmd → exit 0 · endpoint 200 · metric ≥ threshold
                   (autoresearch's Docker gate is a ready-made tier-1 for code tasks)
2. DETERMINISTIC   predicate on artifacts: file exists · parses · schema-valid ·
   PREDICATE       grep finds symbol · row_count > N
3. INDEPENDENT     a DIFFERENT model instance, given a fixed RUBRIC, scores output
   RUBRIC-JUDGE    vs explicit criteria → number + verdict
                   (adversarial: prompted to find why it's NOT done; default FAIL
                    on uncertainty)
─────────────────
✗ NEVER            "the doer says it's finished"   ← what we are removing
```
Tiers 1–2 are objective. Tier 3 is still *quantifiable* (a rubric score) but soft — use only when nothing executable exists, keep the judge **independent + adversarial**, and **emit which tier certified "done"** so "verified" never silently means "an LLM vibed it."

### 5.4 Upgrade the breaker, don't rebuild the loop
`ExternalLoop` breakers are already `(step_no, output) -> (stop, reason, output)`. Keep the socket; change what plugs in: a trivial predicate → a **Verifier** that runs the ladder and returns `{pass, score, reason, feedback}`.

### 5.5 The gate's output is also the steering signal (keep/discard)
```
ASP iter --> VERIFIER
   ├─ PASS (bar met) .................. STOP: success
   ├─ FAIL, improving ................. CONTINUE: feed verifier feedback as
   │                                    next iter's input (steer it)
   └─ FAIL, regressing / plateaued .... DISCARD (revert?) → stop or escalate
```
The failing verifier's diff ("missing X; metric 0.6 vs 0.8 bar") becomes the next iteration's instruction — **which is also the gap-1 state handoff.** The two gaps fuse here.

### 5.6 Distinguish two stop reasons (never conflate)
- **Success stop** — the Verifier's bar was met. (The new gate.)
- **Give-up stop** — `max_iters` / `max_seconds` / no-improvement plateau. (`ExternalLoop` already owns these.)

A give-up is a *failure with a reason*, not a success. Collapsing them re-introduces "declare victory by exhaustion."

### 5.7 Typed verification triggers + Boolean exit gate (research upgrades #3, #4)
**Upgrade #3 — typed triggers.** The verifier ladder (§5.3) answers *how* to check; the research adds a taxonomy of *when/why* a check fires, which routes to the right tier: **syntactic** (parse errors, schema violations), **semantic** (failed assertions/tests), **epistemic** (high uncertainty / contradiction), **strategic** (reward drop, repeated tool failure), **social** (user pushback). Tier-1/2 cover syntactic+semantic; tier-3 covers epistemic/strategic; social is the human-override channel.

**Upgrade #4 — Boolean exit gate on a real oracle.** Make the success-stop concrete: completion is declared **only** when `state["all_stages_passed"]` (or equivalent) is set by an **external oracle** — a real build / test / Docker run — never by a model's opinion. Corpus framing: *"Docker doesn't care how confident the LLM was — it only cares if the container runs."* This is the canonical tier-1 shape; wire the Verifier to set that flag and gate the loop's success-stop on it.

---

## 6. How gaps 1 & 2 compose into one loop

```
ExternalLoop (outer: max_iters / max_seconds / breaker=Verifier)
   └─ fresh ASP each iteration  (gap 1: fresh context + state handoff)
        └─ does work, writes STATE artifact
   └─ Verifier runs the ladder  (gap 2: quantifiable "done?" gate)
        ├─ PASS      → success-stop
        ├─ FAIL      → verifier feedback becomes the NEXT iter's seed
        │              (== the durable state handoff)
        └─ budget/plateau → give-up-stop (failure + reason)
```

The verifier feedback IS the durable handoff carried into the next fresh-context iteration — so building gap 2 also supplies the "what to carry forward" content that gap 1 needs.

---

## 7. Proposed components (shapes, not code)

- **`AcceptanceContract`** — supplied up front: `{criteria, verifier_spec, threshold}`. The loop's definition of "done."
- **`Verifier`** — resolves the highest available ladder tier, runs it, returns `{pass, score, reason, feedback, tier_used}`. Logs `tier_used`.
- **Ladder resolvers** — executable-check runner (optionally delegate to autoresearch `ar` for code tasks), deterministic-predicate evaluator, independent adversarial rubric-judge.
- **Fresh-ASP factory** — builds a new ASP per outer iteration seeded with `objective + state`, never the prior history.
- **State artifact / handoff** — reuses `use_state_management` / `enable_summarization`; persisted via the existing TUI SQLite+JSONL session store for crash durability.
- **Wiring** — Verifier installed as an `ExternalLoop` breaker; outer caps (`max_iters`, `max_seconds`) retained for give-up stops.

---

## 8. Guardrails & operating principles (imported from the True Agent Autonomy research)

Cross-cutting rules the research corroborated — fold into the loop regardless of Option A/B:

- **Upgrade #5 — re-inject the global objective every iteration (anti state-drift).** The named failure *state drift* = the agent forgets the original goal and over-optimizes a sub-task. Mitigation: re-inject the top-level objective into every fresh ASP iteration's seed/system prompt. Cheap, and it composes with the §4 handoff.
- **Upgrade #6 — one-writer-per-module (parallel-write safety).** If iterations or sub-agents ever write in parallel, enforce a single writer per file/module via isolated git worktrees — the named fix for *parallel write conflicts*. (We already operate this way; make it explicit in the loop's contract.)
- **Upgrade #7 — sandbox tier for real-world actions + "verification-infra scales with autonomy."** Any consequential/irreversible action (shell, network, file delete, sending anything) runs behind a sandbox / dry-run gate. Guiding principle: **verification infrastructure — not trust — is what should grow as autonomy rises** (the "L3 production ceiling": beyond it, "silent drift" appears unless a real verification pipeline backs it). Target L3-style *bounded* autonomy, not L5 "never stops."

> **Provenance:** upgrades #1–#7 are folded in from `docs/research/true_agent_autonomy.md` (§7 mapping). #1–#2 live in §4, #3–#4 in §5, #5–#7 here.

---

## 9. Open questions

1. **State artifact location & format** — ✅ **RESOLVED (§14, §14.1, §14.2):** DB is the single write-everything source of truth; OKF is a derived, distilled projection (the in-context tiered memory, transient → short-term → long-term) reconciled by `cadastre`; retrieval-on-demand for erased detail; transient tier = per-topic concept files + `log.md`.
2. **Authoring acceptance contracts in the TUI UX** — how does a user declare `{criteria, verifier, threshold}` per task? A slash command? A per-agent default? Inferred + confirmed?
3. **Judge model selection** for tier-3 (must be independent from the doer; adversarial prompt; default-fail on uncertainty).
4. **Revert semantics** on a "regressing" verdict — do we snapshot/rollback artifacts, and using what (git? session DB checkpoint?).
5. **Handoff fidelity** — how lossy is the state summary; what's the minimum "done/next/facts" schema that keeps a fresh iteration correct.
6. **"Inject entropy to avoid plateau" (comment #11) — revisit via `/paired-debate`.** A commenter proposed injecting entropy so a long run doesn't plateau/converge prematurely (maps to temperature/top-p diversity or periodic fresh re-seeding). Unproven, and possibly redundant with our erase+reseed (which already injects a fresh perspective each iteration). **Deferred to a later paired-debate** — do not build without it.

---

## 10. Out of scope / future

- **Gap 3 — autonomy slider / constraint contract** (instructions / constraints / stopping-criteria as first-class registers; graduated human checkpoints). Deferred to a later feature. The research already hands us a blueprint when we pick this up: the L1–L5 autonomy taxonomies + Karpathy's slider, a Pydantic **constraint contract** (allowed tools + token/$ budget + max depth), and **escalation triggers** (high-stakes / uncertainty). See `docs/research/true_agent_autonomy.md` §5.
- **SOUL / persistent identity anchor (deferred).** A persona/spec the agent re-inherits to resist drift (prior art: SoulSpec / `soul.md`; ContextEcho persona-drift benchmark). Deferred: for a *task* agent, re-injecting the *objective* each iteration (upgrade #5 / R-MEM5) already covers drift; persona-persistence is lower-value here. Revisit only if identity drift shows up in practice.
- **Full Karpathy ML optimizer** — not applicable to the TUI agent; principle borrowed, machinery not.
- **autoresearch (`ar`) delegation** — noted as a ready-made tier-1 verifier for code-producing tasks; optional integration, not required for v1.

---

## 11. Relationship to existing code

- `promptchain/utils/agentic_step_processor.py` — the inner loop; `max_internal_steps`, `history_mode`, `use_state_management`, `enable_summarization`, CoVe.
- `promptchain/utils/external_loop.py` — the outer-loop primitive (`max_iters`, `max_seconds`, `breakers`) — where both gaps land.
- TUI session persistence (`~/.promptchain/sessions/`, SQLite + JSONL) — durability substrate for the gap-1 state artifact.
- Existing ASP docs: `docs/agentic_step_processor*.md`, `docs/architecture/ARCHITECTURE_VISION.md`.

---

## 12. Next steps

1. Review / refine this design (esp. §9 open questions).
2. **The user will run `/speckit-specify`** to promote this doc into a numbered speckit spec (`specs/016-*`) for the task breakdown — this design doc is the spec source (do not auto-run speckit).
3. Prototype: `ExternalLoop` + fresh-ASP factory + a tier-2 deterministic Verifier as the thinnest end-to-end slice, then add tier-1 (executable) and tier-3 (judge).

---

## 13. Realizing these principles in PromptChain (mapping + gaps)

Per §1.1 — every borrowed principle is expressed in PromptChain's own primitives. This section is the translation layer and the gap list.

### 13.1 Principle → PromptChain realization (our idiom)

| Principle (source) | PromptChain realization |
|---|---|
| Long-run persistence | `ExternalLoop` + a fresh-ASP factory |
| Context reset (stay sharp) | `ExternalLoop` reseeds a NEW ASP; erase `internal_history` |
| Distilled durable memory | `ExecutionHistoryManager` → an observation-list artifact |
| Bounded work unit | one ASP per iteration, break on context health |
| Verification gate | Verifier as a `chainbreaker` / `Callable`, separate model |
| Operating protocol | a `PromptChain` static chain of typed steps |
| Blocking subagent | nested `Callable` chain — **its own model**, distilled return |
| Async subagent | `AgentChain` parallel — **different-model** adversarial review |
| Scored / decaying memory | importance + recency weighting in the artifact |

### 13.2 Gaps between how our agents work today and these principles

- **Gap A — context ERASE + reseed. ✅ DECIDED (2026-07-06): YES — erase + reseed (Option A).** Today PromptChain manages context by **compression-in-place**: `ExecutionHistoryManager` truncation (`oldest_first` / `keep_last`) + ASP `enable_summarization` / `use_state_management`. **None of these erase and start fresh.** To realize the reset principle, `ExternalLoop` **must build a new ASP each outer iteration with empty `internal_history`**, carrying forward ONLY the persisted observation-list artifact. *Erased:* the ASP's accumulated internal message history. *Survives:* the objective + the distilled observation list (in the session store). This is the opposite of our current keep-and-compress default — a deliberate, designed erasure, and it is now the v1 direction.
- **Gap B — inner step boundary. ✅ DECIDED (2026-07-06): context-health break (not a bigger count).** `max_internal_steps=5` is a fixed, short, **count-based** cap. Two problems: (1) 5 is too short for a real unit of autonomous work; (2) simply raising it worsens context rot (more steps → more accumulation before any reset). The locked fix: make the inner boundary **context-health-based** — ASP works until its context reaches ~50% of budget (the "first half of the window" sweet spot), then breaks to compact→handoff→reset — and get total length from the **outer loop** (many fresh iterations), not a big inner count. ASP already has `max_context_tokens` (currently warning-only) → **promote it to a break trigger**; keep `max_internal_steps` as a safety backstop (raised modestly, not the primary boundary).
- **Gap C — observation-list artifact + importance/decay.** We have summarization but not a deterministic observation-**list** with importance scores + recency decay (Generative Agents model). Build it on `ExecutionHistoryManager` structured entries (add `importance` + `timestamp`; retention = importance+recency weighted, not `oldest_first`). Feeds Gap A's handoff.
- **Gap D — subagent return convention.** PromptChain composes subagents as nested chains / `AgentChain`; add the convention that a subagent returns a **distilled** result (Result / Output / Evidence / Learnings) rather than raw tool noise — so delegation keeps the parent context clean (the `pi-fork` principle, our way).
- **Gap E — keep-going ≠ verifier.** State explicitly that `ExternalLoop`'s continuation mechanism and the Verifier breaker are **separate components** (the video's autonomy extension was only a turn-continuation nudge, not a verifier). Mostly covered in §5.7 / §8; recorded here for the mapping.

> The verified comment-suggestions (protocol, blocking/async delegation, scored memory, task-meaning decoding, verifier-separation, in-turn compaction) are all realized **through this mapping** — not as Pi features, but as PromptChain chains/loops.

### 13.3 PromptChain advantage — heterogeneous models per subagent (no multiplexer)

A structural edge over tmux-style harnesses (Pi's `interactive-subagents`, Claude Code sub-sessions, etc.): those spin each subagent up as a **separate CLI process of the *same* configured model** in its own pane. PromptChain instead composes subagents **in-process** as nested `PromptChain` chains (a `Callable` step) or `AgentChain` agents — each with its **own model**, via per-step `models[i]` or per-agent config (R-PC4: per-step models are explicit, never defaulted). So **one autonomous run can mix models and providers with NO multiplexer / tmux**:

- **Doer** subagent → a fast/cheap model (the bulk of the work).
- **Adversarial reviewer / critic** subagent → a stronger or simply *different* model/provider.
- **Searcher / summarizer** subagents → whatever tier fits, per task (cf. the local-first escalation ladder).

**This directly strengthens gap 2's doer/critic separation:** a critic running a *different model* is more genuinely independent than a same-model self-check — it does not share the doer's blind spots. So the **tier-3 judge (§5.3)** and the **async reviewer (Gap D)** should default to a **different model than the doer** — which PromptChain makes trivial (one entry in the `models` list). Heterogeneous, mixed-provider subagents — with no separate processes and no single-model constraint — are a first-class PromptChain capability this design should lean on throughout.

---

## 14. Memory architecture — tiered OKF + permanent store (context carries the WHAT, never the HOW)

**Founding principle — "keep the WHAT, erase the HOW."** A PromptChain task decomposes into subtasks; each subtask is handled by a **nested** loop / ASP / subagent with its **own** ephemeral context. When a subtask is judged complete, that working context is **erased** — the parent receives **only the distilled output** ("done + result"), never the subtask's internal steps. Erasure is **by construction via nesting**: the child's context simply never enters the parent's. So the parent context stays **lean and goal-directed** — the *main goal* + the *completed-subtask outputs*, and nothing else.

> Origin (user, 2026-07-06): *"I don't need to know HOW something was done. I just need to know that it was done and completed — the output — and that goes into context, which is going toward the goal."*

**The permanent store = write-everything, out-of-context-by-default.** Everything PromptChain does — including every erased HOW — is persisted to the **permanent database store** (`~/.promptchain/sessions/`, SQLite + JSONL; the *raw complete record*, answers **how**) and distilled into the **long-term OKF bundle** (structured cross-session knowledge + index; answers **what / that it was done**). It is the complete, durable, queryable memory — but it is **not in context by default.**

**Retrieval-on-demand — the only re-entry path for erased detail.** Normally the HOW stays dormant. On a trigger — an error, a *"wait, what happened here?"* moment, or a later step that genuinely needs a prior detail — the agent **queries the permanent store for that specific instance** and pulls *just that* slice back into context. Nothing else brings erased detail back (no bulk reload).

```text
        ┌─────────────────────────── CONTEXT (working) ───────────────────────────┐
        │  main goal  +  completed-subtask OUTPUTS only   (lean, forward-looking)  │
        └───────▲───────────────────────────────────────────────────▲─────────────┘
   distilled    │ output only ("done + result")          retrieval- │ on-demand
   output       │  (the HOW is dropped here)              (error / "what happened?")
        ┌───────┴────────┐   ┌───────────────┐   ┌───────────────┐   │
        │ nested ASP /   │   │ nested ASP /  │   │ … subtask N   │   │
        │ subtask 1      │   │ subtask 2     │   │               │   │
        │ own context →  │   │ own context → │   │               │   │
        │ ERASED         │   │ ERASED        │   │               │   │
        └───────┬────────┘   └───────┬───────┘   └───────┬───────┘   │
                └──── write-everything (the full HOW) ────┘           │
                                     ▼                                │
        ┌──────────────── PERMANENT STORE (out of context) ──────────┴──────────┐
        │  DB: raw complete record (SQLite+JSONL)  ·  OKF: distilled knowledge  │
        │  queryable, durable — dormant until retrieval-on-demand pulls a slice │
        └───────────────────────────────────────────────────────────────────────┘
```

**Tier reconciliation (with §13 + the prior OKF tiers):**
- **Transient (working)** = a subtask's own nested context → **erased** on completion (the HOW).
- **Short-term** = the parent run's resident context: goal + completed outputs + this-run reflections (the WHAT, in-flight).
- **Long-term / permanent** = the **DB** (raw everything) + the **OKF** long-term bundle (distilled, cross-session).
- **Pointers** = the OKF `index.md` over the store — the map retrieval-on-demand walks.

**Mapping to PromptChain primitives:**
- A **subtask = a nested `PromptChain` / ASP** (`Callable` sub-chain or `AgentChain` agent) with its own context. The **nesting boundary IS the erasure boundary** — Gap A's erase applies at *every* nesting level, not just the outer loop.
- The subagent **returns the distilled output only** (Gap D) — literally "keep the WHAT, drop the HOW."
- **Write-through:** every subtask/step result is persisted to the permanent store as it happens (out of context).
- **Retrieval-on-demand** = a `Callable` "recall" step that queries the DB/OKF for a specific past instance, only when triggered.

**Rules (R-MEM):**
- **R-MEM1** — Context holds the *goal + completed outputs* only; never a subtask's internal steps.
- **R-MEM2** — Erasure = the nesting boundary. A completed nested unit's context is discarded; only its distilled output returns to the parent.
- **R-MEM3** — Write-everything to the permanent store (observability: nothing is lost). Content-hash dedup **before** any paid embedding of distilled concepts (economy — the $5K scar).
- **R-MEM4** — Retrieval-on-demand is the *only* re-entry path for erased detail; triggered by error / uncertainty / explicit need, scoped to the specific instance, never a bulk reload.
- **R-MEM5** — The main goal is always resident (re-injected each iteration; ties to upgrade #5, anti state-drift).

**Why this is the whole point:** context *cost* becomes a function of the goal + open outputs, not of everything the agent has ever done — while losing nothing (it's all in the store, recoverable on demand). **Ephemeral by default, complete by record.**

### 14.1 DB and OKF — one source of truth, one derived view (✅ RESOLVED 2026-07-06)

There is **one permanent source of truth — the DB** (`~/.promptchain/sessions/`, write-everything, raw, complete). **OKF is a *derived projection* over it** — the distilled, structured, navigable "what / that it was done" layer (reflections + stable concepts + `index.md`), produced by the reflector and reconciled by **`cadastre`** (R-OKF3: OKF is passive serialization, `cadastre` is the engine). OKF is **not** a second independent store and never holds anything the DB doesn't.

- **DB** = source of truth · raw **HOW** · cheap write-everything (no embedding of raw rows).
- **OKF** = derived **WHAT** · embedded/indexed for retrieval (content-hash dedup **before** embed — R-MEM3 / economy).
- **Retrieval-on-demand:** walk the OKF `index.md` to locate the relevant "what," then drill to the DB rows for the "how" only when an error/uncertainty needs it.
- **Rebuildable:** because OKF is derived, it can be regenerated from the DB if it drifts or is lost — the DB stays authoritative.

### 14.2 Transient-tier granularity (✅ RESOLVED 2026-07-06)

The **transient (working) tier = per-topic concept files + a running `log.md`** — **not** one file per observation. Fast observation writes would explode into thousands of tiny files; per-topic files + `log.md` match both OM's "per-topic markdown + index" shape and OKF's `index.md` / `log.md` conventions, and stay conformant (R-OKF2). **Finer granularity — a concept in its own file — happens only on promotion to long-term**, when a distilled reflection earns a stable OKF concept. So: coarse + append-friendly while transient; proper OKF concepts once promoted.
