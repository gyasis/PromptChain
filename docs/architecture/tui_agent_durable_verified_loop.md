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

## 8. Open questions

1. **State artifact location** — in-RAM only (rot fix) vs persisted to the TUI session DB (rot + crash durability). Default: persist, reusing the existing session store.
2. **Authoring acceptance contracts in the TUI UX** — how does a user declare `{criteria, verifier, threshold}` per task? A slash command? A per-agent default? Inferred + confirmed?
3. **Judge model selection** for tier-3 (must be independent from the doer; adversarial prompt; default-fail on uncertainty).
4. **Revert semantics** on a "regressing" verdict — do we snapshot/rollback artifacts, and using what (git? session DB checkpoint?).
5. **Handoff fidelity** — how lossy is the state summary; what's the minimum "done/next/facts" schema that keeps a fresh iteration correct.

---

## 9. Out of scope / future

- **Gap 3 — autonomy slider / constraint contract** (instructions / constraints / stopping-criteria as first-class registers; graduated human checkpoints). Deferred to a later feature.
- **Full Karpathy ML optimizer** — not applicable to the TUI agent; principle borrowed, machinery not.
- **autoresearch (`ar`) delegation** — noted as a ready-made tier-1 verifier for code-producing tasks; optional integration, not required for v1.

---

## 10. Relationship to existing code

- `promptchain/utils/agentic_step_processor.py` — the inner loop; `max_internal_steps`, `history_mode`, `use_state_management`, `enable_summarization`, CoVe.
- `promptchain/utils/external_loop.py` — the outer-loop primitive (`max_iters`, `max_seconds`, `breakers`) — where both gaps land.
- TUI session persistence (`~/.promptchain/sessions/`, SQLite + JSONL) — durability substrate for the gap-1 state artifact.
- Existing ASP docs: `docs/agentic_step_processor*.md`, `docs/architecture/ARCHITECTURE_VISION.md`.

---

## 11. Next steps

1. Review / refine this design (esp. §8 open questions).
2. Optionally promote to a numbered speckit spec (`specs/016-*`) via `/speckit-specify` for the task breakdown.
3. Prototype: `ExternalLoop` + fresh-ASP factory + a tier-2 deterministic Verifier as the thinnest end-to-end slice, then add tier-1 (executable) and tier-3 (judge).
