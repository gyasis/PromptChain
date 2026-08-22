# Design — `/task`: launch subagents for a task in the PromptChain TUI

**Status:** DRAFT for approval (no code yet). **Repo:** `~/Documents/PromptChain`.
**Depends on:** `promptchain.patterns.agent_comms` (v0.10.0, on `main`).

## 1. Goal

A TUI command that hands a **task** to the `agent_comms` **orchestrator**, which
launches the participant **subagents**, routes the work by capability, streams each
subagent's turn into the UI live, and commits the final synthesis to the chat. In one
line: *type a task → watch a manager agent drive a team of subagents → get a result.*

```
/task "triage this stack trace and propose a fix"
   Boss (orchestrator) → launches [analyst, coder, reviewer]
   routes each open question to whoever can answer it (by capability)
   each turn streams into the right-side dock; final synthesis → chat
```

## 2. User-facing surface

Reuse the existing pattern-command arg parser (`_parse_pattern_command`, `app.py:2938`):

```
/task "<goal>" [--agents a,b,c] [--authority select|steer|full] [--rounds N] [--static]
```

- `"<goal>"` — the task (required).
- `--agents a,b,c` — which subagents to launch. **Default:** all configured session
  agents (`self.session.agents`). Each becomes an `agent_comms` participant.
- `--authority` — orchestrator authority (default `steer`): `select` (pick next) ·
  `steer` (+per-turn directive — best for weaker subagents) · `full` (+final synthesis).
- `--rounds N` — max turns (hard cap; default 8).
- `--static` — use the static orchestrator (`group()` + `by_capability()`) instead of
  the agentic manager. Cheap/deterministic; no manager model calls.

## 3. Flow

1. `handle_command` sees `/task` → `_handle_task_command(command)` (new async method).
2. Parse args; resolve the subagent roster from `self.session.agents` (or `--agents`).
3. Build `agent_comms` participants from each session agent (name, persona/model,
   `capabilities` — see §6). Build the orchestrator:
   - agentic (default): `Orchestrator(name="Coordinator", authority=..., max_rounds=...)`
   - static (`--static`): `group(participants, by_capability(), term=[...])`
4. **Run without blocking the event loop** — `await orch.run_group_async(participants,
   goal, ctx)`, wrapped in `asyncio.wait_for(..., timeout=turn_timeout)` exactly like the
   existing `run_chat_turn_async` call (`app.py:4386-4404`). **Never** call the sync
   `Orchestrator.run_group()` / `Group.run()` — both do `asyncio.run()` internally and
   will raise `RuntimeError: asyncio.run() cannot be called from a running event loop`.
5. Each completed subagent turn fires an `on_turn(msg, ctx)` callback → surfaced live
   (§5). The final synthesis (authority `full`) → committed to `ChatView` as the answer.

## 4. Where it hooks in (concrete)

| Piece | Location |
|---|---|
| Dispatch | `handle_command` if/elif — add `elif command.startswith("/task"): await self._handle_task_command(command)` before the catch-all `else` at **`app.py:2926-2929`** |
| Handler | new `async def _handle_task_command(self, command)` beside `_handle_branch_pattern` (~`app.py:2998`); reuse `_parse_pattern_command` (`app.py:2938`) |
| Run | `await orch.run_group_async(...)` inside that handler (mirrors `handle_user_message` awaiting `run_chat_turn_async`, `app.py:4386`) |
| Autocomplete | add `/task` to the InputWidget command list (same file family as the if/elif chain) |

## 5. Live display — reuse the existing subagent surface

The TUI **already** shows sub-agents: AgentChain router mode emits `plan_agent_start/
complete` → `_orchestration_callback` (`app.py:1728`) → `_add_task_internal_step(...)`
→ **`TaskListWidget`** (the right dock). We reuse that exact visual language:

- Define a sync `_on_agent_task_turn(self, msg, ctx)` that does
  `self._safe_call_ui(lambda: self._add_task_internal_step("tool_call", f"{msg.role} ▸ {msg.content}"))`.
- `_safe_call_ui` (`app.py:393-410`) marshals the update onto the UI thread safely —
  the sanctioned mechanism for any callback that might fire off-thread.
- Pass it as the orchestrator's `on_turn`. So subagent turns land in the dock just like
  today's router sub-agents; the final synthesis commits to `ChatView`.

## 6. Subagent roster + capabilities

- **Roster source:** `self.session.agents` (already managed by `/agent create|use|list`,
  `app.py:2702-2894`) — each has `model_name` etc. `/task` turns these into participants.
- **Capabilities:** `by_capability()` and the agentic manager both route by an agent's
  declared `capabilities`. Session agents don't carry that field today → **add an optional
  `capabilities` field to the session-agent record** (settable via `/agent create ...
  --capabilities risk,security`), defaulting to `[]` (falls back to round-robin / manager
  judgment). Small, additive.

## 7. Prerequisites (small package changes — must land first)

1. **Merge `main` (agent_comms v0.10.0) into the working branch.** The package is on
   `main`; the TUI branch (`fix/tui-history-persistence`) doesn't have it yet, so
   `from promptchain.patterns.agent_comms import ...` currently `ImportError`s there.
2. **Add an `on_turn` hook to `Orchestrator`** (mirror `Group`'s): accept `on_turn` in
   `__init__`, and call `self.on_turn(t[-1], ctx)` inside `_prep`'s `step()` right after
   `await nxt.say_async(...)`. Today only `Group` has `on_turn`; the agentic orchestrator
   needs it for live turn surfacing. (~5 lines, additive.)
3. **(Phase 2) Forward `streaming_callback` through `LLMAgent`** into its
   `AgenticStepProcessor` so we can stream *inside* a subagent's turn (thinking / tool
   calls), not just per-completed-turn. Today `LLMAgent` threads no streaming callback →
   turn-level granularity only.

## 8. Phasing

- **Phase 1 (MVP):** `/task` → agentic orchestrator over session agents, `--authority`/
  `--rounds`/`--static` flags, **turn-level** display in the dock, synthesis to chat.
  Needs prereqs 1 + 2.
- **Phase 2:** intra-turn streaming (prereq 3), `--capabilities` wiring on `/agent
  create`, a capability-aware `--static` default, optional `captain()` nesting for
  sub-teams.

## 9. Decisions (LOCKED 2026-07-12)

1. **Roster default = ALL session agents.** `/task` launches every configured
   `self.session.agents`; `--agents a,b,c` narrows.
2. **Result = final synthesis in CHAT, full transcript in the DOCK** (not chat). Rationale:
   putting every subagent turn in the chat log would bloat the conversation's token
   window; the dock keeps the transcript out of the chat's context budget.
3. **Default orchestrator = AGENTIC manager** (`authority="steer"`); `--static`
   (`group()` + `by_capability()`) is the cheap/deterministic opt-out.
4. **Doc home = repo `docs/`** (`docs/agent_comms_tui_task_command.md`), committed on the
   implementation branch alongside the code.

## 10. Risks / gotchas (already accounted for)

- `asyncio.run()` trap → use `run_group_async` only (§3.4).
- Long tasks blocking the UI → `asyncio.wait_for` timeout, same as existing turns.
- Off-thread UI mutation → `_safe_call_ui` (§5).
- Agentic orchestrator cost → each turn is an ASP loop + a manager decision; `--static`
  and `--rounds` bound it; `--authority select` is the cheapest agentic option.
