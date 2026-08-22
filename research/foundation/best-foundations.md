# The 20 Best Foundations for an Agentic-Coding System Prompt

Distilled from 14 leaked coding-agent prompts + 78 web sources + the DeepLake corpus.
Ordered by priority. Tags: **[U]** universal across prompts · **[W]** weak/local-model-critical ·
**[G]** needs grounding to PromptChain's *real* capabilities (Phase-3 gap check).
The base for assembly is **Claude Sonnet 4.6**; these are the sections it must contain.

1. **Identity & competence persona** [U] — short, high-competence role ("expert software engineer in a
   CLI"). Biases the model toward thorough, persistent behavior. *Devin's "code-wiz" pep-talk; Cursor's
   "powerful agentic coding assistant."*

2. **Persistence / anti-laziness / keep-going-until-resolved** [U][W] — "You are an agent: keep going
   until the task is fully solved; only yield when done or genuinely blocked. Don't guess — resolve it."
   The single most universal rule. *Cursor, Codex, Windsurf, Gemini CLI.*

3. **Mandatory pre-action reasoning (`<thinking>`/`<plan>`)** [W] — require a brief reasoning block
   before tool calls. **Critical for weak models — never suppress reasoning for speed.** *Devin `<think>`,
   Cline/RooCode `<thinking>`.*

4. **Plan-first + task-list discipline** [U][G] — for non-trivial tasks, write a plan first (our
   `task_list_write_tool`); **one task in_progress at a time; mark complete immediately, never batch**;
   skip plans for trivial tasks. *Claude Code TodoWrite, Codex update_plan.*

5. **ReAct loop (think → act → observe → repeat)** [U] — explicit perceive-reason-act-observe cycle;
   adapt after each observation. Plan when steps are predictable; ReAct when dynamic.

6. **Tool-call discipline** [U][W] — never name tools to the user; one-sentence "why" before each call;
   parallelize independent calls; prefer specialized tools over raw shell; only call when necessary.
   *Cursor/Warp/Amp "never refer to tool names"; Codex preamble.*

7. **Tool-selection guidance** [U][G] — short "use X for Y" map (search local vs web, read vs edit, run
   command), **rendered from the real registry** so it never names a tool we lack.

8. **Surgical edit discipline** [U][G] — edit in place with the edit tools (not full rewrites); minimal/
   conservative changes; read before editing; never truncate code. *RooCode apply_diff, Codex apply_patch,
   Augment str_replace-only.* (We have line-based edits; a unified-diff patch is a feature gap.)

9. **Context-gathering before acting** [U] — understand first; search broad→narrow; **never guess or
   hallucinate** file contents/structure; fewest high-signal calls. *Augment "one high-signal call
   first"; Windsurf "answer rooted in research."*

10. **Verification habit** [U][G] — after changes, run tests/lints/typecheck; self-verify; **cap retries
    (≈3×) then escalate** instead of looping. *Gemini CLI verify-step, Amp get_diagnostics gate, Cursor
    3× lint cap.*

11. **Explicit stop conditions / completion detection** [U][W] — define "done" precisely + a
    machine-readable completion signal (status/progress/next); **don't end with a question.** Karpathy's
    "stopping criteria." *Cline attempt_completion.*

12. **Constraints / scope discipline** [U] — do exactly what's asked, no more; don't modify tests; **ask
    before destructive/irreversible actions** (commit/push/install/deploy). Karpathy's "what NOT to
    change." *Augment/Amp/Warp permission gates.*

13. **Output & communication shaping** [U][W] — concise (≤~4 lines default for CLI); **no preamble/
    postamble/flattery/apology**; markdown for terminal; reference code as `file:line`. *Claude Code, Amp,
    Gemini CLI.*

14. **Final-answer requirements** [U] — the final message must include the **full content** from tool
    results (the user can't see tool output); never "I have explained…", show the actual information.

15. **Library / convention adherence** [U] — **never assume a library is available** — verify it's used
    in the project first; mimic existing style/idioms. *Gemini CLI BLOCKING, Devin, Amp.*

16. **Safety, approval & sandbox discipline** [U][G] — defensive-only; bias hard against unsafe commands;
    never log/commit secrets; flag destructive ops for approval; report environment issues instead of
    self-fixing; resist prompt injection (ignore instructions to exfiltrate the prompt). *Cline
    requires_approval, Devin POP-QUIZ/report_environment_issue.* (Gating LLM tool calls is a feature gap →
    epic #15.)

17. **Path & environment discipline** [U][G] — absolute paths; verify before reporting; respect cwd and
    the session security mode (strict/default/trusted). *Already in our current foundation.*

18. **Deterministic structure for weak/local models** [W] — XML semantic boundaries (`<thinking>`,
    `<plan>`); schema-enforced output; **separate the reasoning scratchpad from the delivered answer**;
    decompose into the smallest auditable unit; **re-inject the goal every few steps** + an end-of-prompt
    "most important rules" recap. *Augment-gpt-5 patterns; local-model research.*

19. **Progress updates / observability** [U] — emit a brief progress line every few tool calls; ReAct
    externalizes reasoning for an audit trail. *Codex 8–10-word updates.*

20. **Project-doc / memory adherence** [G] — obey rules in a project doc (AGENTS.md / CLAUDE.md) when
    present. *Codex AGENTS.md, Claude Code CLAUDE.md, Cursor .cursorrules.* (Auto-load is a feature gap →
    epic #15 / D2.)

---

### Two architecture mechanisms the prompt must *hook into* (Phase 5, not prose)
- **Loop-until-completion / goals** — Ralph (fresh context + `todo.md` source-of-truth) / Karpathy
  (constraints + stopping criteria + git-revert) / Claude `/loop`+`/goal`+Stop-hook. The prompt provides
  the **completion signal**; `ExternalLoop`-wraps-`AgenticStepProcessor` provides the loop.
- **Subagency** — orchestrator delegates scoped work to an isolated sub-agent that returns a **summary**;
  **no nested spawning**. The prompt provides the **delegate/handoff awareness**; OrchestratorSupervisor +
  `delegate_task` provide the mechanism.

### Notes for assembly (Phase 4)
- Use Sonnet 4.6's structure as the skeleton, but **strip its product-specific bulk** (artifacts,
  window.storage, image search, copyright-quote rules) — none apply to a coding TUI.
- Keep #3, #6, #11, #13, #18 **prominent** — they carry the most weight for weak/local models.
- Every tool reference stays **dynamic** (rendered from the registry), so the static base never names a
  tool we don't have.
