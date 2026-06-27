# PRD — Codex Harness Parity: Close the Chat-Session-Loop Gaps

**Status:** Open · **Created:** 2026-06-27 · **Owner:** gyasis
**Tracking epic:** [#15](https://github.com/gyasis/PromptChain/issues/15) · **Infographic:** `docs/codex-harness-gap-analysis.html`
**Global PRD pointer:** `~/dev/prd/` entry `codex_harness_parity` (references this file)
**Ephemeral marker:** delete / archive when epic #15 is closed (all 8 child issues resolved).

---

## 1. Context / Origin

Produced via `/smart-research` (2026-06-27): two parallel research agents reverse-engineered the
**canonical chat-session loop** of modern agentic coding harnesses — OpenAI **Codex** (`codex-rs`),
**Claude Code**, **opencode**, **aider** — across **14 dimensions**, then mapped the PromptChain
Textual TUI against it with file:line evidence. The full visual writeup is the committed infographic
`docs/codex-harness-gap-analysis.html`.

**Headline finding:** PromptChain implements the *entire* harness loop shape — **0/14 dimensions
missing, 13 partial, 1 ahead** (Reliability). The gaps are *depth within dimensions* and cluster on
**safety & control** (ungated LLM tool calls, no true turn-cancel, sandbox not auto-enforced) plus a
few cheap parity wins (project-doc auto-load, diff cells, `/compact`).

This PRD turns that analysis into a tracked, issue-backed plan to reach full parity.

## 2. The 14-dimension scorecard

| # | Dimension | Status | Gap | Issue |
|---|-----------|--------|-----|-------|
| D1 | Input & composition | PARTIAL | no image/paste input | — (low) |
| D2 | Context assembly | PARTIAL | no project-doc auto-load (AGENTS.md/CLAUDE.md) | [#19](https://github.com/gyasis/PromptChain/issues/19) |
| D3 | Model call & streaming | PARTIAL | no reasoning-effort control | [#21](https://github.com/gyasis/PromptChain/issues/21) |
| D4 | Agentic tool loop | PARTIAL | parallel tools run sequentially | [#23](https://github.com/gyasis/PromptChain/issues/23) |
| D5 | Tools | PARTIAL | no web tool; line-based (not patch) edits | [#22](https://github.com/gyasis/PromptChain/issues/22), [#23](https://github.com/gyasis/PromptChain/issues/23) |
| D6 | Approvals & permissions | PARTIAL | **LLM tool calls ungated** | [#16](https://github.com/gyasis/PromptChain/issues/16) |
| D7 | Sandboxing & isolation | PARTIAL | sandbox agent-chosen, not enforced | [#17](https://github.com/gyasis/PromptChain/issues/17) (builds on #6) |
| D8 | Interrupt & steering | PARTIAL | **no true turn-cancel** | [#18](https://github.com/gyasis/PromptChain/issues/18) |
| D9 | Rendering & UX | PARTIAL | no diff/exec cell | [#20](https://github.com/gyasis/PromptChain/issues/20) |
| D10 | Context-window mgmt | PARTIAL | no manual `/compact` / `/context` | [#21](https://github.com/gyasis/PromptChain/issues/21) |
| D11 | Persistence & resume | PARTIAL | no checkpoints / `/undo` | [#23](https://github.com/gyasis/PromptChain/issues/23) |
| D12 | Multi-agent / routing | PARTIAL | sequential only, no parallel agents | [#23](https://github.com/gyasis/PromptChain/issues/23) |
| D13 | **Reliability** | **HAS ▲** | ahead of Codex (turn timeout + retry + finally) | — done |
| D14 | Observability | PARTIAL | no OTEL | [#23](https://github.com/gyasis/PromptChain/issues/23) |

## 3. Work items (prioritized)

### HIGH — safety & control (the real divergence from Codex)
- **[#16] D6 — Gate LLM-invoked tool calls** with approval modes (`suggest` / `auto-edit` /
  `full-auto`), reusing `ApprovalScreen`. *Accept:* write/exec calls prompt in `suggest`; mode in status bar.
  Files: `app.py` agent tool loop (~3320–3429), `cli/tui/approval_screen.py`.
- **[#17] D7 — Auto-enforce the sandbox by policy.** Under `full-auto`, route exec + file-write
  through the existing `DockerExecutor` (built in #6) instead of relying on the model to choose it.
  Files: `docker_executor.py`, tool-registry routing.
- **[#18] D8 — True turn-cancel on Ctrl+C / Esc** → `Task.cancel()` + `request_interrupt()`.
  Complements the shipped turn-timeout (timeout = backstop; this = user control).
  Files: `app.py:626`, `agentic_step_processor.request_interrupt()`.

### MED — high value, low effort
- **[#19] D2 — Auto-load project docs** (AGENTS.md / CLAUDE.md / README) into context at session start.
- **[#20] D9 — Diff + exec-output render cells** styled with the new blend theme.
- **[#21] D3/D10 — reasoning-effort knob + `/compact` + `/context`** (auto-distillation already exists).
- **[#22] D5 — Web fetch / search tool** (the one missing whole tool category).

### LOW — longer horizon
- **[#23] D4/D5/D11/D12/D14 — Polish:** safe parallel tools, structured-patch edits,
  checkpoints/`/undo`, parallel agents, OTEL export. Split out when picked up.

## 4. Already done / ahead
- **D13 Reliability** — per-turn timeout (`PROMPTCHAIN_TURN_TIMEOUT`, `asyncio.wait_for`) + classified
  retry/backoff + router fallback + guaranteed `finally` cleanup. Shipped on `fix/tui-mcp-freeze`
  (commit `c3d32ec`). Codex has no documented per-turn timeout — PromptChain is more turn-resilient here.

## 5. Related existing issues
- **#6** — native DockerExecutor (the executor D7 builds on; D7 is the *enforcement* layer).
- **#2** — AgenticStepProcessor hardcodes the SWE/TUI system prompt + tool inventory; related to D2
  context-grounding (auto-loaded project docs help, but #2 is a deeper grounding fix).

## 6. Success criteria
Epic #15 closed = all 8 child issues resolved and the 14-dimension scorecard shows no HIGH/MED gaps
(LOW items may remain as tracked follow-ups). Re-run the `/smart-research` comparison to confirm.

## 7. References
- Infographic: `docs/codex-harness-gap-analysis.html` (this repo)
- Method: `/smart-research` skill (parallel multi-source research → 14-dimension synthesis)
- Codex: developers.openai.com/codex (cli · approvals · sandboxing); codex-rs architecture
- Claude Code internals; opencode/aider docs
