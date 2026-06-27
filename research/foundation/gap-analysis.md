# Phase 3 — Gap Analysis: Strip-List + Feature-Backlog

For every collected leaked prompt, tag content that is **harness/coder-specific** (references functions,
tools, or capabilities PromptChain doesn't have) or **one-model idiom** (Constraint C). Two outputs:
- **STRIP-LIST** — do NOT put it in our base prompt (no real backing / would mislead the model).
- **FEATURE-BACKLOG** — worth BUILDING into the PromptChain TUI → issues under **epic #15**.
PromptChain's real surface = the **32 registered tools** (file/edit/sandbox/ripgrep/terminal/task/
blackboard/delegate/path) — anything outside that is a gap.

---

## A. STRIP-LIST — keep out of the base prompt

| Strip | Seen in | Why it's stripped | Keep instead |
|---|---|---|---|
| **Product bulk** — artifacts, `window.storage`, image_search, copyright 15-word rule, `/v1/messages`, computer-use sandbox idioms, skill mount paths | Claude Sonnet 4.6 | Anthropic *product* features, irrelevant to a coding TUI (~95% of the 99KB) | nothing |
| **Tool-name-specific edit idioms** — `apply_patch` / `str_replace_editor` / `edit_file` diff markers / `replace_in_file` SEARCH-REPLACE | Codex, Augment, Cursor, Cline | We have **line-based** edits (`insert_at_line`/`replace_lines`), not those tools | the *surgical-edit principle* (F8), rendered against our real edit tools |
| **Semantic codebase search** — `codebase_search` / `codebase-retrieval` | Cursor, Augment, Amp, Devin | We only have **lexical `ripgrep_search`** | "search local files" → ripgrep (dynamic tool map) |
| **LSP navigation** — `go_to_definition`/`references`/`hover_symbol` | Devin | not present | nothing |
| **Web / browser** — `web_search`/`web_fetch`/`browser_action`/Playwright suite | Claude, Cursor, Gemini, v0, Cline, Amp | not present (MCP only if loaded) | dynamic MCP section only |
| **One-model idioms** — Anthropic reminder tags, "respond GPT-4.1" identity misdirection, model-specific XML reminder styles | Claude, Windsurf | violates Constraint C (don't bake one model's dialect into the base) | the **family adapter** sets format per model |
| **Sub-agent tool specifics** — Claude `Task`, Amp `Oracle`/`Task`, RooCode Boomerang | Claude Code, Amp, RooCode | we have `delegate_task` but a different shape; subagency is Phase 5 | the *delegation principle*, wired to our real mechanism later |
| **Project-doc references** — "obey AGENTS.md / CLAUDE.md / .cursorrules" | Codex, Claude, Cursor, Amp | we don't auto-load project docs yet | add only after the feature ships (↓) |
| **Deployment / design idioms** — Fly.io deploy, `expose_port`, v0 design-token/color rules, Supabase-first | Devin, v0 | out of scope for a general coding TUI | nothing |
| **Verbose few-shot example blocks** | Cursor, Gemini CLI, v0 | token cost (Constraint A) — examples are a *module*, not always-on base | optional "examples" module for strong models only |

**Net:** the base keeps the **principles** (the 20 foundations) and drops the **harness-specific nouns**.
Every tool reference in the shipped prompt is rendered dynamically from our real registry — so the base
can never name a tool we don't have.

---

## B. FEATURE-BACKLOG — build into the PromptChain TUI (→ epic #15)

| Feature (from the gap) | Value | Issue |
|---|---|---|
| **Structured / unified-diff patch edit** (apply_patch-style) | matches what every top agent uses; cleaner than line ops | #15 · D5 |
| **Web fetch / search tool** | the one whole tool category we lack | **#22** · D5 |
| **Semantic codebase search** | beyond lexical ripgrep | new (epic #15) |
| **LSP navigation** (def/refs/hover) | type-accurate edits (Devin-class) | new (epic #15) |
| **Gate LLM tool calls** (suggest/auto-edit/full-auto) | safety — biggest divergence from Codex | **#16** · D6 |
| **Auto-enforce sandbox** (route exec through DockerExecutor) | safety | **#17** · D7 |
| **Project-doc auto-load** (AGENTS.md/CLAUDE.md) | cheap parity win; *then* the prompt may reference it | **#19** · D2 |
| **Diff / exec render cells** | distinct framing in the TUI | **#20** · D9 |
| **get_diagnostics / lint tool** | first-class verification (Amp-class) | new (epic #15) |
| **Loop-until-completion + goals + Document-&-Clear** | the loop architecture (Ralph/Karpathy/Claude) | new · Phase 5 |
| **Subagency** (orchestrator → isolated worker → summary) | ~90% gain on complex tasks; big-brother/lean-executor | new · Phase 5 |
| **Model Profiler + SIO integration** (transcript emitter + harness adapter + probe harness + DSPy jacket module) | the whole "harness fits the model" engine | new · Phase 5 |

These feed the existing epic **#15** (Codex-parity) + the new Phase-5 builds. Several are already filed
(#16–#22).

---

## C. What STAYS — the model-agnostic base

The **20 foundations**, written portably and **grounded to PromptChain's real 32 tools** (tools rendered
dynamically), are the base content. The strip-list removes harness-idiom + non-existent-tool references;
the feature-backlog captures what to build so the prompt can *honestly* grow into those capabilities. The
base never promises what the harness can't do — and the dynamic layer + Model Profiler make it fit each
model. **This is the input to Phase 4 (assemble the Sonnet-4.6-based foundation prompt).**
