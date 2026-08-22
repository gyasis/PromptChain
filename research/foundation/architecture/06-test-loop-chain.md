# Component — TestLoopChain (PromptChain-native "iterate-until-tests-pass" engine)

The concrete engine behind tool-forging / autoresearch / Karpathy loops: a reusable PromptChain that
generates code and iterates **until a language's test suite passes**, in a Docker sandbox. Reverse-
engineered from the micro-agent loops, then made **language-agnostic** (the TS originals are TS-locked).

## TWO loops to capture, not one (user, 2026-06-27)
The user's `~/Documents/code/micro-agent` is a **fork**. The fork **ADDED the "MA loop"** (the Ralph Loop
2026 multi-agent state machine) on top of — not replacing — **the original micro-agent loop**. They are two
distinct engines and PromptChain should reproduce **both**, at two complexity tiers:

| | **(A) Original micro-agent loop** | **(B) The MA loop (what the fork added)** |
|---|---|---|
| Shape | single agent: generate → run test → on fail feed error back → regenerate, until pass | multi-agent state machine: **librarian → artisan → critic → testing → adversarial → completion**, then **fresh-context reset** for the next iteration |
| Context | one growing thread (simple) | **fresh context every outer iteration** (only structured `testResults` threaded forward) — the Ralph principle |
| Breakers | max retries / test pass | entropy/stagnation detector, budget enforcer, context-usage reset (40%), max_iterations |
| Code | classic Builder.io `micro-agent` RGR loop | `src/state-machine/ralph-machine.ts` (the XState machine) + `lifecycle/iteration-manager.ts` (the outer fresh-reset loop) + `state-machine/ralph-orchestrator.ts` |
| PromptChain form | **MicroPromptChain** — lean: 1 model + TestRunner + ExternalLoop. Cheap; great for weak/local models + tool-forging | **RalphChain / MA-PromptChain** — `AgenticStepProcessor` sub-chains (librarian/artisan/critic/adversarial) under `OrchestratorSupervisor`, each ExternalLoop pass a *fresh* turn, threading only the failure render |

**MicroPromptChain ≈ the TestLoopChain spec below** (it already IS the simple loop). **RalphChain** wraps the
same TestRunner/Sandbox but swaps the single generator for the 5-state multi-agent pipeline + fresh-reset
outer loop. Build A first (it's the substrate); B is A's generator stage exploded into the state machine.

## The founding principle — the **Ralph loop** (fresh context every iteration)
The fork's files are named `ralph-*` on purpose. The principle, verified in the code:
- `iteration-manager.ts:4` — *"the core Ralph Loop principle: fresh LLM context each iteration."*
- `ralph-machine.ts:7-9` — the state machine runs **once** (librarian→…→`completion` is `type:'final'`) then
  **"the machine is destroyed and a fresh instance is created for the next iteration."** Both `TESTS_PASS`
  and `TESTS_FAIL` land in `completion`; the **outer** manager decides pass→stop vs fail→spawn-fresh.
- README — *"named after the iterative testing approach: Fresh Context Reset — each iteration destroys and
  recreates the AI context to prevent pollution"* + entropy circuit-breaker + budget.

So the loop is **brute-force toward a falsifiable goal (tests pass), discarding accumulated context each
pass** (after Geoffrey Huntley's "Ralph Wiggum" loop). Token waste and context pollution can't compound;
only the structured failure is carried forward. **This is exactly why it rescues weak/local models** — each
iteration is a clean bounded task, not an ever-growing transcript — and it's directly mimickable: ExternalLoop
= the outer `while`, each `step` = a fresh `AgenticStepProcessor` turn, `state["failure_context"]` = the one
thing threaded forward.

## The endgame — dev-kid drops the TS micro-agent and calls the PromptChain loop (user, 2026-06-27)
Once MicroPromptChain + RalphChain work, **rewire dev-kid's `tier_runner.py` to invoke the PromptChain loop
instead of spawning `ma-loop run` (the TS micro-agent)**. dev-kid keeps its outer Python **tier-escalation**
(the model ladder) but its inner engine becomes a PromptChain — so the loop is Python-native, language-
agnostic (any test framework, sandboxed), and reuses PromptChain primitives instead of a Node subprocess.
The TUI "micro-PromptChain" work feeds this: the same RalphChain that powers the TUI is the engine dev-kid
calls. Net: **one loop engine (PromptChain), two consumers (the TUI + dev-kid).**

## What dev-kid's loop actually is (reverse-engineered)
A **two-level nested loop** (`~/Documents/code/dev-kid`):
- **Outer — Python tier escalation** (`cli/sentinel/tier_runner.py:282` `run_tiered()`): walks up to 11 cost
  tiers from `ralph-tiers.json` (all-local → groq → cerebras → claude-handoff → … → openai-max). Per tier:
  budget/key guards → write tier model config → spawn the inner loop as `ma-loop run <file> -o <obj> -t
  <test_cmd>` → parse metrics → **independently re-run the test to verify PASS** (anti-lying) → PASS return,
  FAIL escalate. → this is dev-kid's **model-escalation ladder** (= our Constraint-B big-brother escalation).
- **Inner — TypeScript XState machine** (`src/state-machine/ralph-orchestrator.ts`, the "micro-agent"/"mini"):
  per iteration up to `max_iterations`: **librarian** (build context + thread prior `testResults`) → **artisan**
  (generate/edit the target file, sees prior failures) → **critic** (review) → **testing** (`test-executor.ts`
  runs the test cmd → parse to `ralph-test-json`; PASS → exit0; FAIL → `TESTS_FAIL` event carrying structured
  failures → back to librarian). Fresh context per iteration + threaded test results. Stop: pass / max_iters /
  budget.

**Module decomposition (what we mimic):** ContextBuilder(Librarian) · CodeGenerator(Artisan) ·
CodeReviewer(Critic) · TestRunner(test-executor) · TestResultParser · FailureRenderer · Orchestrator(loop) +
Sandbox.

**Language coupling (what to fix):** hardcoded `-f cargo` flag (`tier_runner.py:694`), Rust-only `cargo
--message-format=json` error attribution, TS/vitest assumptions, host (not sandboxed) execution, and the inner
runtime is Node.js-only (opaque subprocess).

## The PromptChain-native design — `TestLoopChain`
Mirror the module decomposition; replace Node/XState with Python + **ExternalLoop**; replace host execution
with **DockerExecutor**; make the test-runner **pluggable per language**.

```
TestLoopChain
  ├─ ContextBuilder   (≈Librarian)
  ├─ CodeGenerator    (≈Artisan — prompt includes prior failure)
  ├─ CodeReviewer     (optional ≈Critic)
  ├─ TestRunner       (PLUGGABLE protocol ≈test-executor): Pytest · Cargo · Vitest/Jest · GoTest …
  ├─ TestResultParser (per-runner stdout → TestResult)
  ├─ FailureRenderer  (TestResult → next-iteration prompt section)
  └─ ExternalLoop  +  DockerExecutor   (loop driver + sandbox — both ALREADY EXIST)
```

**Per-iteration step (ExternalLoop `step`):** build prompt (objective + current code + prior failure) → LLM
generates code → optional review → **DockerExecutor**: write code file → install deps → run **that language's**
test cmd → parse → PASS? `state["winning_code"]=code; return False(stop)` : `state["failure_context"]=render(
result); return True(continue)`. **Breakers:** `max_iters`, `max_seconds`, budget, no-improvement(patience).

**Language-agnostic TestRunner protocol** (the key win over dev-kid): one instance per language —
`image` (python:3.12 / rust:1.78 / node:20 / golang) · `install_deps_cmd` · `test_cmd` · `parse_output()`.
`detect_runner(target, root)` reads pyproject/Cargo.toml/package.json/go.mod. **Write code in ANY language,
run THAT language's test framework in Docker** — not locked to one TS lib.

**Wrappable as a tool** — `run_test_loop(objective, target_file, language, max_iterations)` → a tool the parent
chain (or autoresearch, or the tool-forge) calls. Returns `{result: PASS|FAIL|BUDGET|NO_IMPROVEMENT,
winning_code, iterations, stopped_by, attempts}`.

## Why this "brings it all together"
- **It IS the Karpathy/Ralph tool-forge** from `05-...md`: "build tool X → test → iterate until pass" = run
  `TestLoopChain` with the tool's test as the success bar. The loop's pass-gate is the grounding guardrail.
- **It's the missing inner loop for autoresearch.** autoresearch (`dataengineer/autoresearch/engine/run_real.py`)
  has DockerExecutor + ExternalLoop available but uses neither — it `for _ in range(MAX_TICKS)` and **terminates
  after one Docker run without feeding failures back**. Dropping `TestLoopChain` in gives it dev-kid-style
  iterative test-repair.
- **It's a reusable PromptChain-as-a-tool** — the engine for every iterate-until-pass loop (tool creation,
  sub-chain authoring, autoresearch, codegen).
- **The outer tier-escalation = our heterogeneous-model ladder** (Constraint B): dev-kid's `tier_runner` walks
  cheap→expensive models on FAIL; we already have `ralph-tiers.json` + the ollama-cloud escalation ladder. The
  TestLoopChain's breaker can escalate the model on stall (big-brother), reusing that ladder.

## Build delta (≈500 lines Python, 0 TS)
New: TestRunner protocol + `detect_runner` (~50) · per-language parsers (Pytest/Cargo/Vitest/Jest/Go ~250) ·
FailureRenderer (~30) · generation prompt builder (~40) · budget/no-improvement breakers (~30) · TestLoopChain
wiring (~60). Extend: `DockerExecutor.write_file()` + language images (~30). **Reuse unchanged:** `ExternalLoop`
+ `DockerExecutor` (+ `AgenticStepProcessor` for richer orchestration). **Better than dev-kid:** language-
agnostic, sandboxed, no Node subprocess, no TS lock-in.

## Branch note
Can live on this branch (it's a PromptChain library component) or branch off — it's independent of the
foundation-prompt work. Recommended: its own branch `feat/test-loop-chain` since it's a self-contained tool.
