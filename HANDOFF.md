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

## #1 status — BLOCKED on a render/propagation gap (the live mystery)
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
