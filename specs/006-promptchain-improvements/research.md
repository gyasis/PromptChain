# Research: PromptChain Comprehensive Improvements
**Branch**: `006-promptchain-improvements` | **Phase**: 0 — Research
**Date**: 2026-02-24

---

## Summary of Findings

All NEEDS CLARIFICATION items are resolved below. Decisions draw from direct
codebase inspection, industry references (AG2, Claude Code, Gemini CLI) cited
in `IMPROVEMENT_ROADMAP.md`, and PromptChain's existing architectural patterns.

---

## Decision Log

### D-001: Gemini MCP Parameter Name Corrections

**Decision**: Fix three parameter mismatches in
`promptchain/utils/enhanced_agentic_step_processor.py`.
- Line 645: `error_context` → `error_message` (for `gemini_debug`)
- Line 564: Remove unsupported `num_ideas` (for `gemini_brainstorm`)
- Line 575: `question` → `prompt` (for `ask_gemini`)

**Rationale**: Confirmed by `IMPROVEMENT_ROADMAP.md` BUG-017/018/019. MCP tool
schemas are the source of truth; Python call-sites must match them exactly.

**Alternatives considered**: Patching MCP server schemas — rejected because
PromptChain does not own the Gemini MCP server.

---

### D-002: Event Loop Conflict Resolution

**Decision**: Use the existing `promptchain/cli/utils/event_loop_manager.py`
module (`run_async_in_context`, `make_sync_from_async`) throughout all TUI
pattern-command handlers instead of ad-hoc `asyncio.run()` calls.

**Rationale**: The module is already present (created in a prior fix), correctly
detects CLI vs. TUI context, and avoids nested loop errors from Textual's
running loop. The issue is incomplete adoption across all pattern commands.

**Alternatives considered**: `nest_asyncio` — rejected because it patches the
global event loop, creates subtle race conditions, and is not needed when proper
context detection is used.

---

### D-003: JSON Output Parser Fallback

**Decision**: The `json_output_parser.py` already has `try/except` blocks.
The remaining gap is the top-level `extract()` method on line 92 which re-raises
raw `Exception`. Wrap in a documented fallback that logs the raw string and
returns the configured `default` value rather than propagating.

**Rationale**: Silent data loss is worse than partial data. A logged fallback
satisfies FR-003 and SC-001 without breaking callers that check for `None`.

**Alternatives considered**: Raising a typed `JSONParseError` — rejected as it
would break existing callers that do not catch it.

---

### D-004: MLflow Queue Bounded Shutdown

**Decision**: The `promptchain/observability/queue.py` already implements a
`flush(timeout=10.0)` method added in the previous fix cycle. The remaining
work is to ensure `shutdown()` calls `flush(timeout=timeout)` before
`worker.join()` and that the worker thread respects the `_shutdown` flag on
timeout rather than blocking in `queue.join()` indefinitely.

**Rationale**: Confirmed working pattern in `queue.py` lines 122-188. The
`flush()` method correctly polls with 0.1 s intervals; `shutdown()` must use
it rather than raw `queue.join()`.

**Alternatives considered**: Thread daemon flag — rejected because it silently
drops in-flight events on process exit.

---

### D-005: Observability Config Caching

**Decision**: `promptchain/observability/config.py` already implements
file-mtime-based cache invalidation (`_load_yaml_config` lines 75-140). The
remaining work is to ensure all public accessor functions call only
`_load_yaml_config()` and never open the file directly.

**Rationale**: mtime-based invalidation is a well-established OS-level pattern
with no external dependencies. Already partially implemented; needs audit to
confirm all access points route through the cached loader.

**Alternatives considered**: inotify / watchdog — over-engineered for config
that changes rarely.

---

### D-006: Verification Result Deep Copy

**Decision**: Add `copy.deepcopy()` before returning a cached
`VerificationResult` from `self.verification_cache` in
`enhanced_agentic_step_processor.py` line 190.

**Rationale**: The `VerificationResult` dataclass contains mutable fields.
Returning a reference from the cache allows callers to modify the cached entry,
causing BUG-009 cache corruption. `deepcopy` is the minimal, correct fix.

**Alternatives considered**: Making `VerificationResult` frozen/immutable —
rejected because it would require refactoring all mutation sites.

---

### D-007: Context Distillation Wiring

**Decision**: `ContextDistiller` already exists in
`promptchain/utils/execution_history_manager.py` (line 519). The work is to
wire it into `AgenticStepProcessor._execute_step()` so it is automatically
invoked when token usage reaches the 70 % threshold.

**Rationale**: Avoids reimplementing existing code. The `ContextDistiller` API
(`should_distill()`, `distill()`) maps directly to FR-007.

**Implementation approach**: Inject a `ContextDistiller` instance into
`AgenticStepProcessor.__init__()` as an optional parameter. Check
`should_distill()` at the start of each thought cycle before the LLM call.

**Alternatives considered**: Background thread — deferred to FR-010's Janitor
Agent; the inline check is simpler and sufficient for FR-007.

---

### D-008: MemoStore Integration

**Decision**: `promptchain/utils/memo_store.py` is already implemented with
SQLite persistence and bag-of-words cosine similarity fallback. The work is to:
1. Wire it into `AgenticStepProcessor` context injection (via `inject_relevant_memos`)
2. Auto-save successful task completions as memos
3. Expose configuration through `AgenticStepProcessor.__init__()`

**Rationale**: The helper function `inject_relevant_memos()` is already defined.
Integration follows the AG2 Teachability pattern from the IMPROVEMENT_ROADMAP.

**Embedding strategy**: Default bag-of-words fallback is sufficient for MVP.
Pluggable `embedding_function` parameter supports future upgrade to OpenAI or
local embeddings without breaking changes.

---

### D-009: InterruptQueue Integration

**Decision**: `promptchain/utils/interrupt_queue.py` is fully implemented with
`InterruptQueue`, `InterruptHandler`, and all interrupt types. The work is to:
1. Integrate `InterruptHandler` into `AgenticStepProcessor` thought cycle
2. Wire TUI input handler to call `submit_interrupt()` on `/interrupt` commands
3. Integrate with `GlobalOverrideSignal` in the message bus

**Rationale**: The queue is implemented and isolated. The integration gap is
calling `check_and_handle_interrupt()` between agentic steps and propagating
the result to the TUI.

---

### D-010: Micro-Checkpoint Strategy

**Decision**: Use the existing `promptchain/utils/checkpoint_manager.py`
(`Checkpoint`, `CheckpointManager` classes) for micro-checkpoints saved after
each tool call. Checkpoints are in-memory only (per spec assumption), stored in
a per-session dict keyed by `(agent_id, step_number)`.

**Rationale**: Existing infrastructure prevents creating a new storage system.
The ephemeral design satisfies the spec assumption that micro-checkpoints do not
persist across sessions.

**Alternatives considered**: SQLite persistence — deferred to a future iteration
as noted in spec assumptions.

---

### D-011: PubSubBus Design

**Decision**: Extend `promptchain/cli/communication/message_bus.py` with a
`PubSubBus` class that:
- Maintains a `Dict[str, List[Callable]]` of topic → subscriber callbacks
- Uses `asyncio.gather()` for concurrent fan-out to all subscribers on publish
- Exposes `subscribe(topic, callback)`, `unsubscribe(topic, callback)`,
  `publish(topic, payload)` (async), and `publish_sync(topic, payload)` (sync
  wrapper)

**Rationale**: AG2's topic-based pub/sub is the target pattern. Extending the
existing bus avoids breaking current `MessageBus` consumers and satisfies
FR-016/FR-017.

**Alternatives considered**: Third-party pub/sub library (e.g., PyPubSub) —
rejected to keep dependency count low; the required feature set is simple.

---

### D-012: AsyncAgentInbox Design

**Decision**: Implement `AsyncAgentInbox` as a thin wrapper around
`asyncio.PriorityQueue`. Priority levels: 0 = interrupt/override,
1 = normal message, 2 = background. Expose `async send(message)` and
`async receive()` methods with a non-blocking `try_receive()` fallback.

**Rationale**: `asyncio.PriorityQueue` is stdlib, zero dependencies, and
natively supports the priority ordering required by FR-016. The thin wrapper
adds PromptChain-specific message typing without reinventing queueing.

**Alternatives considered**: `janus` (sync/async queue) — rejected as an
external dependency for minimal added value.

---

### D-013: Testing Strategy

**Decision**: Apply TDD (constitution III) for all changes. Test order:
1. Unit tests for each new/modified class in isolation
2. Integration tests wiring components together
3. Contract tests for public API stability

**Key test files**:
- `tests/unit/test_interrupt_queue_integration.py`
- `tests/unit/test_memo_store_integration.py`
- `tests/unit/test_context_distiller_wiring.py`
- `tests/unit/test_pubsub_bus.py`
- `tests/unit/test_async_agent_inbox.py`
- `tests/integration/test_006_bug_fixes.py`
- `tests/integration/test_006_steering_flow.py`

---

### D-014: Backward Compatibility

**Decision**: All new parameters added to `__init__()` methods carry `Optional`
type annotations with `None` defaults. Sync wrapper methods remain unmodified
in signature. Existing callers with no changes continue to work identically.

**Rationale**: Constitution VI (Async-First Design) requires backward
compatibility. The spec assumptions explicitly state sync interfaces continue
to work via wrapper methods.

---

## Resolved Unknowns

| # | Unknown | Resolution |
|---|---------|------------|
| 1 | Are any bug fixes already partially implemented? | Yes — queue flush (D-004), config cache (D-005), event loop manager (D-002), context distiller (D-007), memo store (D-008), interrupt queue (D-009) are all present. Work is integration/wiring, not greenfield. |
| 2 | Which embedding approach for MemoStore? | Bag-of-words fallback (already implemented), pluggable via `embedding_function` (D-008). |
| 3 | Is CheckpointManager usable for micro-checkpoints? | Yes (D-010). |
| 4 | PubSubBus: new file or extend existing? | Extend existing MessageBus file (D-011). |
| 5 | AsyncAgentInbox: asyncio.PriorityQueue sufficient? | Yes (D-012). |
| 6 | Janitor Agent scope? | Background asyncio Task monitoring ExecutionHistoryManager; creates a separate class `JanitorAgent` in `promptchain/utils/janitor_agent.py`. |
