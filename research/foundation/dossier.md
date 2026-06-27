# Research Dossier — Agentic-Coding System Prompts, Dynamic Prompting, Loops & Subagency

**Sources:** 14 leaked coding-agent system prompts (Claude Sonnet 4.6, Claude Code 2.0, Cursor 2.0,
Devin, v0, Cline, RooCode, Codex CLI, Gemini CLI, Warp, Augment ×2, Windsurf, Amp) + the user's
DeepLake corpus + 2 Gemini deep-research reports (78 web sources). 2026-06-27.
**Optimization targets:** general agentic coding + robustness on weaker/local models.

---

## A. Prompt-engineering techniques (the recurring DNA across 14 prompts)

**The three behavioral anchors** (OpenAI's own guidance, confirmed everywhere): every effective
agentic prompt explicitly states **persistence** ("keep going until resolved"), **active tool use**,
and **planning** ("think ahead"). Without all three, models revert to chatbot mode.

- **Persistence / anti-laziness** — near-universal. Cursor: *"keep going until the user's query is
  completely resolved… Only terminate your turn when you are sure that the problem is solved."* Same in
  Codex CLI, Windsurf, Gemini CLI. Devin opens with a competence "pep-talk" persona to bias toward
  thoroughness.
- **Plan-first + task-list discipline** — Claude Code `TodoWrite` (mandatory, real-time, **one
  in_progress at a time, mark complete immediately, never batch**), Codex `update_plan`, Cursor
  `todo_write`, Augment/Amp todo. Plan when non-trivial; *don't* plan trivial tasks.
- **ReAct (think→act→observe)** — Devin `<think>` (mandatory before critical git ops / before editing /
  before reporting done), Cline/RooCode `<thinking>`, Windsurf exemplar chains. Plan when steps are
  predictable; ReAct when dynamic.
- **Tool-call discipline** — **never name tools to the user** (Cursor, Warp, Amp), **explain-why
  preamble** before calls (Codex 8–12 words), **parallelize independent calls** (Claude Code, Gemini
  CLI, v0), **specialized tools > bash** (Read/Edit/Grep not cat/sed), only-call-when-necessary.
- **Surgical edits** — patch/edit tools not full rewrites; minimal/conservative changes; read before
  edit; **line-anchored diffs** (RooCode `apply_diff`, Codex `apply_patch`); never truncate code (Warp).
- **Verification habit** — run tests/lints/typecheck after edits; Gemini CLI mandatory "Verify
  (Standards)" step; Amp `get_diagnostics` gate; **retry cap then escalate** (Cursor 3× lint, Devin/Codex
  3× CI, v0 2× sandbox).
- **Context-gathering** — understand before acting; **broad→narrow** search; **never guess/hallucinate**
  ("answer rooted in research" — Windsurf); fewest high-signal calls (Augment "one high-signal call
  first").
- **Output shaping** — concise (Claude Code/Amp ≤4 lines CLI default), **no preamble/postamble/flattery**
  (universal), markdown for CLI, **code refs as `file:line`**, final-answer must include full tool-result
  content ("user can't see tool results").
- **Constraints / scope** — do exactly what's asked, no more; **don't modify tests**; **ask before
  destructive** (commit/push/install/deploy — Augment/Amp/Warp). This is Karpathy's "what NOT to change."
- **Safety** — defensive-only; bias against unsafe commands (Windsurf: user can't override); secrets
  never logged/committed; approval booleans (Cline `requires_approval`); Devin `report_environment_issue`
  + **POP-QUIZ divergence detector** (anti-prompt-injection); identity/refusal templates.
- **Library/convention adherence** — *"NEVER assume a library is available; verify its usage in the
  project"* (Gemini CLI BLOCKING, Devin, Amp); mimic existing style.

## B. Weak / local-model robustness (the user's key target)

From deep-research + Augment-gpt-5 prompt analysis:
- **Modularity is a NECESSITY, not a nicety** — MoE local models (~3B active params) tax every prompt
  token; assemble only the needed fragments (a terse base + attached modules).
- **DO NOT suppress reasoning** — Claude Code's "lead with the answer, not the reasoning" / "rush bias"
  is **fatal for small models**; they *must* emit pre-action reasoning tokens. Keep a **required
  `<thinking>`/`<plan>` block**.
- **XML > JSON** for structure — hard semantic boundaries small models parse reliably; JSON streams
  break. Use `<thinking>`, `<plan>`, tool envelopes.
- **Schema-enforced output + separate scratchpad** — keep reasoning in an internal field, deliver via a
  structured output field (Ollama `format=`); decompose into smallest auditable unit per call ("one issue
  at a time").
- **Goal re-injection** — re-state the overarching goal after every few steps to fight context drift;
  end-of-prompt "summary of most important instructions" (recency attention) — Augment does both.
- **Single-IN_PROGRESS invariant + success-criteria checklist** anchor weak models against subjective
  "done."

## C. Dynamic & modular prompting (DSPy-style — the feature to capture)

- **Static base + dynamically attached modules** is the production norm (Claude Code assembles dozens of
  fragments by mode/sub-agent/session; RooCode = Roles + Modes; Cursor = base + `.cursor/rules` router).
- **DSPy** — Signatures (typed input→output contract), Modules (`Predict`, `ChainOfThought`, `ReAct`),
  Optimizers compile the actual prompt text. **GEPA** (Genetic-Pareto, 2025) reflects on execution
  traces to mutate instructions — beats RL by ~20% with **35× fewer rollouts**; great on high-linguistic-
  diversity tasks. **MIPROv2** = Bayesian instruction+demo search.
- **Composable assembly** — Jinja2/templating (Claude Launcher `cl tdd sonn`), **type-safe** module
  injection (Embabel: persona/objective/guardrails as distinct typed blocks). Categories: orchestration ·
  knowledge · personality/constraints.
- **→ PromptChain fit:** our `BasePromptBuilder` + the 21 composable blocks in `prompt_templates.py` +
  `DynamicTUIPromptGenerator` already model "static base + dynamic add-ons"; `prompt_engineer.py` is a
  GEPA-adjacent reflective optimizer. We can expose add-on modules + optional optimization.

## D. Agentic loops (Karpathy / Ralph / goals — to integrate with ExternalLoop)

- **Karpathy "AutoResearch" loop** — anchor in **git + file state**: read `program.md` goal → modify one
  file → time-boxed run → parse metric → **commit if better, `git revert` if worse**. Failures are erased
  so they never pollute context. Structure = (a) what to do, (b) constraints/what-not-to-change, (c)
  **stopping criteria**.
- **Ralph Wiggum loop** (Geoffrey Huntley) — **fresh context every iteration**: a bash `while` loop
  re-runs the agent `--yolo`; the agent has **zero memory**, derives all state from `todo.md` on disk →
  do one task → update todo → exit → loop restarts. Beats "context rot." Demands bulletproof specs
  (spec-driven dev).
- **Claude `/loop` + `/goal` + Stop-hook** — native completion-promise: on exit, a Stop hook checks the
  promise ("all lints resolved"); if unmet, reinject errors + force another iteration.
- **→ PromptChain fit:** `ExternalLoop` (deterministic, bounded, breakers) **wrapping an
  AgenticStepProcessor turn** is exactly this. Add a **machine-readable completion schema**
  (status/progress/subtasks/next/handoff) + `AGENTIC_LOOP_BLOCK` + goal state in the loop `state` dict +
  a TaskList-completion breaker. Fresh-context option = re-run the processor with state from disk.

## E. Subagency (orchestrator-worker — the architecture to add)

- **Context isolation is the whole point** — orchestrator spawns ephemeral workers with a **fresh
  context**, scoped prompt + restricted tools (`.claude/agents/*.md`); worker's noise (stack traces,
  misreads) stays trapped; it returns **one synthesized summary**. Keeps parent context pristine.
- **No nested spawning** (hard boundary — prevents runaway trees). **Async fan-out/fan-in** (dispatch
  many `Task` calls, gather summaries). **Dynamic Workflows** = deterministic JS removes the LLM from the
  routing loop. **File-based handoff** (inbox/outbox YAML) = crash-resilient, auditable.
- Anthropic reports **~90% improvement** from orchestrator-worker vs single-agent on complex tasks.
- **→ PromptChain fit:** `OrchestratorSupervisor` + `delegate_task` + `AgentChain` + blackboard +
  `AsyncAgentInbox` already provide the pieces; formalize a first-class spawn/delegate/handoff + the
  no-nested-spawn rule + summary-return.

## F. Security (must bake into the prompt)
Prompt-injection zero-click RCE + TOCTOU "trust persistence" are real (Codex CLI CVE-class). Mitigations:
`report_environment_issue` instead of self-fixing env; approval booleans for destructive ops; ignore
instructions that try to exfiltrate the system prompt; divergence self-check.

---

## Cross-prompt feature matrix (for the Phase-3 gap analysis)

| Capability | Who has it | PromptChain status (real tools) |
|---|---|---|
| Task/todo list | Claude Code, Codex, Cursor, Augment, Amp, v0 | ✅ `task_list_write_tool` + TUI task widget |
| ReAct `<thinking>` | Devin, Cline, RooCode, Windsurf | ✅ AgenticStepProcessor loop (add to prompt) |
| Line-anchored patch edit | RooCode `apply_diff`, Codex `apply_patch` | ⚠️ line-based editor (`insert_at_line`/`replace_lines`), no unified-diff patch — **feature gap** |
| Semantic codebase search | Cursor, Augment, Amp, Devin | ⚠️ `ripgrep_search` only (lexical) — **feature gap** |
| Web/fetch tool | Claude Code, Cursor, Gemini, v0, Amp | ❌ none — **feature gap** (epic #15 / D5) |
| LSP navigation | Devin (go_to_def/refs/hover) | ❌ none — **feature gap** |
| Sub-agent delegation | Claude Code `Task`, Amp `Task`+Oracle, RooCode Boomerang | ⚠️ `delegate_task` + OrchestratorSupervisor (Phase 5) |
| Approval/sandbox gating | Cline `requires_approval`, Codex modes, v0 | ⚠️ ApprovalScreen (user-shell only) — **feature gap** (epic #15 / D6,D7) |
| Project-doc auto-load | AGENTS.md / CLAUDE.md / .cursorrules | ❌ none — **feature gap** (epic #15 / D2) |
| Diagnostics/lint tool | Amp `get_diagnostics`, Gemini verify-step | ⚠️ via `terminal_execute` only |
| Loop-until-completion | Ralph, Karpathy, Claude /loop | ⚠️ ExternalLoop exists, not wired to a turn (Phase 5) |

Legend: ✅ have it · ⚠️ partial / via generic tool · ❌ missing. The ⚠️/❌ rows feed the Phase-3
strip-list (don't promise it in the prompt) **and** feature-backlog (build it → epic #15).
