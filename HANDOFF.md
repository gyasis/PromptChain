# Rich-output TUI work — handoff (worktree `feat/tui-rich-output-ui`)

Goal: make the TUI render the agent's **tool calls (#1)** and **internal reasoning
(#2)** as persistent collapsible gutter-bar sections (the Codex-lean mockup),
instead of collapsing every turn down to just the final answer.

## How to run THIS worktree's code (not the global install)
The editable `promptchain` is pinned to the **main** repo dir. Use the launcher:
```
.claude/worktrees/tui-rich-ui/dev-tui.sh [--dev] [--model openai/gpt-4.1-mini]
```
It self-locates → always runs the worktree's code, and sources the Ollama-Cloud key.

## Done this session
- `model_catalog.py` (on `feat/tui-rich-output`, the base) — live catalog of 71
  models (Mac-Studio-local + Ollama-Cloud + curated OpenAI/Gemini), each with
  routing (api_base/api_key_env) + a best-effort reasoning profile + graceful
  fallback. Verified live.
- `dev-tui.sh` launcher.

## SESSION UPDATE (2026-06-27) — #1 RESOLVED, #2 BUILT + VERIFIED

**#1 is NOT a router-mode gap — it already works in the committed code.** Proven
at three levels: (a) static trace — `run_chat_turn_async`'s router branch DOES
forward `streaming_callback` → `process_prompt_async` → `instruction.run_async`
→ `_stream_event`; (b) an isolated repro of the AgentChain(router)→PromptChain→
processor path streamed `tool_call`/`tool_result`; (c) a **headless Textual pilot
of the REAL app** (`app.run_test()` + `handle_user_message`) showed 2 visible
`role-tool` widgets + 1 `role-think`. The previous "still not rendering" reading
was almost certainly the **editable-install gotcha** (see below): the test
imported the *main-repo* `promptchain`, not this worktree's edited code. The
fixes that actually closed #1 (streaming_callback wired at app.py:3443/4180 +
by-identity `remove_message(processing_msg)`) are already committed in `eccd2c7`.

> **TESTING GOTCHA (critical):** `pip install -e` pins `promptchain` to the MAIN
> repo. Running a script with `python /path/to/script.py` puts the *script's* dir
> on `sys.path`, so `import promptchain` resolves to the main repo — NOT this
> worktree. To test worktree code: `sys.path.insert(0, "<worktree>")` at the top
> of the script BEFORE importing promptchain (or run `python -m` from the worktree
> root). `dev-tui.sh` works because `-m` runs from the worktree cwd.

**#2 BUILT + VERIFIED.** Collapsible reasoning + tool blocks with the full
lifecycle (stream live full → 3.5s idle dwell → auto-collapse to a one-line
summary → click to expand → new delta re-expands). 14/14 deterministic lifecycle
checks pass against the real widgets; real-LLM pilot confirms live events flow
into blocks and auto-collapse. See "## #2 IMPLEMENTATION (done)" below.

## #1 status — RESOLVED (history below; was wrongly suspected as a router gap)
Diagnosis so far:
- The agentic processor **DOES** emit the event:
  `agentic_step_processor.py:1479  self._stream_event("tool_call", ...)` (and an
  `error` event on timeout). `_stream_event` only fires if `self.streaming_callback`
  is set (line 556).
- `_streaming_callback` (app.py:1485) already turns `tool_call`/`tool_result`/
  `thinking` into `role="system"` messages with `metadata.event_type=...`, and
  `chat_view.MessageItem` (chat_view.py:115-122) gives those the **amber role-tool
  gutter** via metadata; `detail_visible=True` by default.
- **Bug found + fixed (but still not rendering):** the agent-chain processors at
  `app.py:3442` and `4178` wired only `progress_callback`, NOT `streaming_callback`.
  Added `streaming_callback=self._streaming_callback` to both. **Yet tool sections
  still don't appear** in a live test (gpt-4.1-mini, `read /tmp/pc_term_probe.txt`).

### Next debugging step (do this FIRST, fresh context)
Confirm whether `_streaming_callback` is actually invoked for `tool_call`:
1. Launch `dev-tui.sh --dev`, run a tool prompt, then check the dev log
   (`~/.promptchain/sessions/default/debug_*.log`) for `Executing tool:` (proves
   line 1474/1479 reached) — if present, the processor emits but the callback/render
   is the gap.
2. Add a temporary `logger.warning("STREAM %s", event_type)` at the top of
   `_streaming_callback` → relaunch → see if `tool_call` arrives.
   - **If it does NOT arrive:** the *running* processor isn't the 3442/4178 instance.
     The chat path is AgentChain (router) → PromptChain → AgenticStepProcessor;
     trace which processor instance handles the turn and whether AgentChain forwards
     `streaming_callback`. (Grep `register_callback`, `AgentChain`, and how the
     per-agent PromptChain is invoked at runtime — app.py ~3660/3699 has a SECOND
     path that DOES set streaming_callback; the agent may use that or yet another.)
   - **If it DOES arrive but nothing shows:** the message is added then hidden/removed.
     Check `detail_visible` at runtime and whether a turn-end sweep removes
     `metadata.streaming==True` messages.

## Changes made in this worktree (uncommitted → being committed now)
- `app.py`: `_add_tool_section()` helper + TOOL_CALL_START/END observability hook
  (covers MCP-tool + PromptChain-loop tools, which emit observability events);
  `streaming_callback` added to the two agentic processor constructions.
- `dev-tui.sh`, this `HANDOFF.md`.

## #2 (not started) — collapsible reasoning
- Build a reasoning **extractor** keyed off the model's catalog reasoning profile
  (`field` / `think_tag` / `reasoning_content` / `none`) with graceful fallback,
  fed by deepseek-r1 (`message.thinking`) / qwq (`<think>`) etc.
- A collapsible Textual widget: stream reasoning in → collapse on answer → click to
  expand. The `thinking` _stream_event path + the same render plumbing as #1.

## UPDATE — Ctrl+T toggle ruled OUT as the cause (live test)
Pressed Ctrl+T live: it fires ("Thinking/tool detail hidden" → "shown"), so the
binding works AND can hide detail. BUT toggling to **shown** revealed nothing —
so the tool-call messages are **never created**. `_streaming_callback` is not
firing for `tool_call` in the running path.

Strong suspect: **router mode**. This session runs AgentChain in `execution_mode=router`
(cheap gpt-4o-mini router + the agent). The per-agent PromptChain processor got
`streaming_callback` wired (app.py:3442/4178), but the **router execution path**
likely doesn't forward the inner AgenticStepProcessor's stream events to the TUI's
`_streaming_callback`. NEXT: trace how AgentChain (router) invokes the agent's
PromptChain at runtime and whether streaming_callback survives that hop; OR test in
single-agent (non-router) mode to confirm the events render there.

## #2 UX SPEC (user, explicit) — the thinking/tool block lifecycle
The collapsible reasoning (and tool) blocks must behave like this:
1. **Stream live, FULL** — reasoning streams in fully visible as the model generates.
2. **Dwell ~3–4s** — once complete, stay FULLY on screen for 3–4 seconds so it can
   be read.
3. **Auto-collapse** — then collapse to a single **truncated line / rich bullet**
   (one-line summary, e.g. `▸ reasoning · N steps · click to expand`).
4. **Expand on demand** — click/Ctrl-toggle re-opens it to the FULL reasoning
   rendered as **rich bullet points**.
Flow: full stream → 3–4s full dwell → auto-collapse to truncated bullet → expandable.
Same collapse-after-dwell applies to the #1 tool-call sections.

Implementation notes:
- The 3–4s dwell + auto-collapse = a `set_timer(3.5, collapse)` after the block's
  final delta (cancel/restart the timer on each new delta so it only fires once the
  stream is idle).
- Collapsed state = a one-line summary widget; expanded = the full rich-bullet body.
  Toggle on click (`on_click`) and/or a Ctrl key. ListView doesn't support nested
  Collapsible widgets (see chat_view note ~line 281), so do collapse by swapping the
  item's rendered content / `display`, not a Textual `Collapsible`.
- "rich bullet points" = render the reasoning steps via the ChatMarkdown path
  (clean headings, `•` bullets, no boxes) — reuse the #1 render plumbing.

## #2 IMPLEMENTATION (done — 2026-06-27)

Design: reuse `MessageItem` with a `metadata["block"]` shape (no nested Textual
`Collapsible`, per the ListView constraint). One **reasoning block per turn**
(accumulates `thinking` lines → `▸ reasoning · N steps · click to expand`) and one
**tool block per tool call** (call + result, amber `role-tool` → `▸ ⚙ <tool> ·
click to expand`).

Files:
- `chat_view.py` — `MessageItem._render_block()` (collapsed = one `Text` line;
  expanded = `Group` of `• `-prefixed rich lines + header), `on_click` toggles
  `collapsed` + `refresh(layout=True)`; `ChatView.item_for()` / `refresh_block()`.
- `app.py` — block state + helpers `_begin_turn_blocks` / `_append_reasoning_line`
  / `_start_tool_block` / `_append_tool_result` / `_restart_block_dwell` /
  `_collapse_block`. Dwell = `set_timer(3.5, …)` re-armed on each delta and
  **generation-guarded** (`metadata["dwell_gen"]`) so only the latest arming
  collapses. `_streaming_callback` thinking/tool_call/tool_result now route into
  blocks; the observability `_add_tool_section` routes through the same helpers so
  PromptChain-loop / MCP tools get the identical lifecycle. Turn reset wired in
  `handle_user_message`.

Verified headless: `scratchpad/pilot_unit.py` (14/14 deterministic lifecycle
checks) + `scratchpad/pilot_blocks.py` (real LLM → blocks created + auto-collapse).

### Reasoning extractor (the OTHER half of #2 — DONE 2026-06-27)

Pulls a reasoning model's INTERNAL reasoning and feeds it into the reasoning
block as `thinking` stream events. Profile-driven (model_catalog) WITH graceful
all-surfaces fallback.

- `promptchain/utils/reasoning_extractor.py` — pure `extract_reasoning(message,
  profile) -> (reasoning, cleaned_content)`. Surfaces: `reasoning_content`
  (o-series / deepseek-cloud / gpt-oss), `thinking` field (ollama deepseek-r1
  `think=true`), `<think>…</think>` tags (qwq / qwen3), incl. unclosed-tag
  (streaming) handling. Reads dicts AND litellm objects (via `model_dump`).
- `agentic_step_processor.py` — `reasoning_profile` ctor param + `_emit_reasoning()`
  called after each LLM response (both standard + TAO paths): emits `thinking` +
  strips `<think>` from content so the answer stays clean. No-op for plain models.
- `app.py` — `_reasoning_profile_for(model_name)` (via `infer_reasoning_profile`)
  passed to the processor at both construction sites; the profile's `enable`
  params (`think=True` / `reasoning_effort`) merged into the model params so the
  model actually emits reasoning.
- `model_catalog.py` — gpt-oss reclassified to `reasoning_content` (verified live).

Verified: `scratchpad/test_extractor.py` (14/14), `test_emit_reasoning.py` (6/6),
and a LIVE call to `gpt-oss:20b` on Ollama Cloud — real reasoning extracted, AND
it exercised the graceful fallback (catalog guessed `think_tag`; the model used
`reasoning_content`; fallback caught it). `enable` params and the openai-compat
shim route are the only cloud-routing concern; the TUI applies them automatically.

Interaction note: Ctrl+T keeps its existing **hide/show all detail** behavior;
per-block expand/collapse is **click**. (Spec said "click/Ctrl-toggle re-opens" —
click is wired; a Ctrl-to-expand-all binding can be added if wanted, but it would
overload the current Ctrl+T hide binding.)

## RICH-OUTPUT REVAMP (2026-06-27) — shipped + backlog

**Core principle:** the TUI is the "output agent" — it RECOGNISES tool output and
styles it ITSELF (syntax-highlight code, color diffs, italic reasoning). It does
NOT prompt the LLM to format anything (non-deterministic, token-wasteful).

### Shipped (committed on feat/tui-rich-output-ui)
- #1 persistent tool sections; #2 collapsible reasoning/tool blocks (stream→dwell→
  collapse→expand); reasoning extractor (gpt-oss/qwq/deepseek reasoning_content/
  `<think>` → block) + cloud-model routing (`_model_call_params` via model_catalog).
- Welcome banner ordering fix; robust agentic-path (empty instruction_chain still
  streams blocks); cohesive monochrome glyph set (▸ ✓ ✗ ⚠ ⚙ ◇ ★ ↳ ◌ — no emoji);
  ALL gray backgrounds → black canvas #0b0e14; bottom padding on message containers.
- Reasoning = ★ italic word-for-word; tool blocks render syntax-highlighted CODE
  (file reads, line numbers, transparent bg) + colored +/- diffs; tool blocks stay
  expanded (code is content), reasoning dwell-collapses. Truncation lifted
  (processor 500→4000, TUI no 300 cap).
- Agent Activity dock panel LIFECYCLE fixed: `finalize_turn()` clears+hides it when
  a turn ends (no more stuck `[..........] waiting…`); kept only if real TodoWrite
  tasks remain unfinished; shows live during processing.

### Tooling
- `tmux-manager` skill + `~/.local/bin/tmux-nav` (find/whoami/ls/goto/capture/send/
  run/wait/keys + `restart` with process-death verification + `kill`). Visibility-
  first (HITL): drive the REAL visible pane, never headless-only. Saved to memory.
- TESTING GOTCHA: editable install resolves to MAIN repo; a script must
  `sys.path.insert(0, "<worktree>")` to import worktree code. `dev-tui.sh` is fine.

### Backlog — CHAT rendering (the general-chat richness)
1. **diffs-on-edits** (IN PROGRESS) — `file_edit`/`replace_lines`/`file_write` should
   return a unified diff (useful for the agent too) → `_render_diff` colors it.
   `file_edit(path, old_text, new_text)` currently returns a plain success string.
2. **section labels + per-call metadata** — dim uppercase `FUNCTION / TOOL CALL`,
   `SUBTASKS`, `AGENTICSTEPPROCESSOR — INTERNAL REASONING`; `142ms · ok`,
   `step 3/5 · 2 tools · 3.4s` (mockup Image #4).
3. **inline SUBTASKS block** — the `≡ Tasks · 2/4 · click to expand` todo list IN the
   chat (with +/v/o/- markers, strikethrough skipped). This is the useful half of the
   old dock panel, moved inline.

### Backlog — DOCK PANEL vision (live agent-status surface)
The dock panel (TaskListWidget) is now a clean, auto-clearing LIVE surface. Build it
into a three-part status board:
1. **Live activity** while working — steps + sub-agents working (with spinners) +
   loop progress (e.g. Ralph loop `8/10` with a spinner). Needs the loop/sub-agent
   runtime to EMIT progress events the panel subscribes to.
2. **RECAP** — a rolling COMPRESSED summary of the session (files touched, tasks
   done, decisions) shown in the dock, AND reused as the **compaction seed**: when
   history is summarized for smaller models, compaction builds on the recap instead
   of re-summarizing raw turns (continuity + efficiency). Build on PromptChain's
   existing summarization (`history_summarizer.py`, `ExecutionHistoryManager`,
   processor `enable_summarization`) — one rolling summary that both renders + feeds
   compaction.
3. Auto-clear/shrink when idle (DONE).
