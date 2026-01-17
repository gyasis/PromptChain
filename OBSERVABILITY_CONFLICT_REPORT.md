# Observability Conflict Analysis Report

**Date:** 2026-01-10
**Branch:** 005-mlflow-observability
**Status:** CRITICAL - MLflow tracking completely inactive

---

## Executive Summary

**FINDING:** The new MLflow observability system (spec 005) is **completely disabled** via the ghost pattern, while the old callback system remains **fully active**. Both systems are present but **zero MLflow tracking occurs**.

**ROOT CAUSE:** The ghost pattern (line 17 in `ghost.py`) evaluates `is_enabled()` at module import time, which returns `False` because:
1. Environment variable `PROMPTCHAIN_MLFLOW_ENABLED` is not set (defaults to `False`)
2. MLflow is not installed
3. Ghost pattern returns identity decorator, making all `@track_*` decorators no-ops

**IMPACT:** User reports "0/0 activities" in MLflow because decorators are never actually wrapping functions.

---

## System Architecture Comparison

### OLD SYSTEM (v0.4.1 - Callback-based)
**Status:** ✓ ACTIVE
**Files:**
- `promptchain/utils/execution_callback.py` (CallbackManager)
- `promptchain/utils/execution_history_manager.py`

**Integration Points:**
- Line 38: `promptchaining.py` imports `CallbackManager`
- Line 234: Initializes `self.callback_manager = CallbackManager()`
- Line 522-1790: 50+ `callback_manager.emit()` calls throughout chain execution
- Line 946: Passed to `AgenticStepProcessor` for observability

**Current Activity:**
- ✓ Captures 6 event types: `CHAIN_START`, `STEP_START`, `MODEL_CALL_START`, `MODEL_CALL_END`, `STEP_END`, `CHAIN_END`
- ✓ Fully integrated into `PromptChain.process_prompt_async()`
- ✓ Active in production CLI usage

### NEW SYSTEM (spec 005 - MLflow decorators)
**Status:** ✗ INACTIVE (ghost pattern active)
**Files:**
- `promptchain/observability/decorators.py` (@track_llm_call, @track_task, @track_routing, @track_session)
- `promptchain/observability/ghost.py` (conditional decorator pattern)
- `promptchain/observability/config.py` (environment-based configuration)
- `promptchain/observability/queue.py` (background logging queue)
- `promptchain/observability/context.py` (nested run tracking)

**Integration Points:**
- Line 14: `promptchaining.py` imports `track_llm_call`
- Line 1834-1837: `@track_llm_call` decorator on `run_model_async()`
- Line 197: `main.py` has `@track_session()` decorator on `_launch_tui()`
- Line 201: `init_mlflow()` called at TUI startup
- Line 247: `shutdown_mlflow()` called at TUI shutdown

**Current Activity:**
- ✗ Ghost pattern returns identity decorator (line 68-71 in `ghost.py`)
- ✗ Decorators do NOT wrap functions (confirmed: `run_model_async` has no closure)
- ✗ No MLflow runs created
- ✗ Background queue never receives events

---

## Conflict Evidence

### 1. Environment Variables (ROOT CAUSE)
```
PROMPTCHAIN_MLFLOW_ENABLED: NOT SET (defaults to False)
MLFLOW_TRACKING_URI: NOT SET
PROMPTCHAIN_MLFLOW_EXPERIMENT: NOT SET
```

**Impact:** `ghost.py` line 17 evaluates `_ENABLED = is_enabled()` → `False` at module import time.

### 2. MLflow Installation (BLOCKER)
```
MLflow is NOT installed
```

**Impact:** `ghost.py` line 23-26 forces `_ENABLED = False` when MLflow import fails.

### 3. Ghost Pattern State
```python
# From ghost.py
_ENABLED = is_enabled()  # False
_MLFLOW_AVAILABLE = False  # MLflow not installed

# From conditional_decorator()
if _ENABLED:  # Never true
    return tracking_decorator
else:
    return make_ghost_decorator()  # Always returns this (identity decorator)
```

**Result:** All `@track_*` decorators become no-ops.

### 4. Decorator Wrapping Test
```
run_model_async.__name__: run_model_async
run_model_async.__wrapped__ exists: False
run_model_async has closure: NO (ghost pattern active)
```

**Proof:** Function is NOT wrapped - decorator was replaced by identity function.

### 5. Callback System Activity
```
Callback events captured: 6
   - CHAIN_START
   - STEP_START
   - MODEL_CALL_START
   - MODEL_CALL_END
   - STEP_END
   - CHAIN_END
```

**Proof:** Old system is fully functional and capturing events.

---

## Line-by-Line Conflict Analysis

### File: `promptchain/utils/promptchaining.py`

**Line 14:** `from promptchain.observability import track_llm_call`
- **Status:** Import succeeds, but returns ghost decorator
- **Conflict:** No conflict (decorator is identity function)

**Line 38:** `from .execution_callback import CallbackManager, CallbackFunction`
- **Status:** Active import
- **Conflict:** No conflict (separate system)

**Line 234:** `self.callback_manager = CallbackManager()`
- **Status:** Active initialization
- **Conflict:** **POTENTIAL**: If MLflow were enabled, both systems would track same events

**Line 1834-1837:** `@track_llm_call(...)` on `run_model_async()`
- **Status:** Decorator applied but is identity function (no-op)
- **Evidence:** Function has no closure, no `__wrapped__` attribute
- **Conflict:** **DESIGN INTENT VIOLATED**: Decorator intended to wrap but is ghosted

**Lines 522-1790:** 50+ `callback_manager.emit()` calls
- **Status:** All active and firing
- **Conflict:** **REDUNDANCY**: If MLflow enabled, would duplicate tracking

### File: `promptchain/cli/main.py`

**Line 19:** `from promptchain.observability import track_session, init_mlflow, shutdown_mlflow`
- **Status:** Imports succeed, `track_session()` is ghost decorator

**Line 197:** `@track_session()` on `_launch_tui()`
- **Status:** Decorator applied but is identity function (no-op)
- **Conflict:** Session tracking never occurs

**Line 201:** `init_mlflow()`
- **Status:** Called but exits early (line 678-680 in `decorators.py`)
- **Evidence:** `is_enabled()` returns `False`, logs "MLflow tracking disabled"

**Line 247:** `shutdown_mlflow()`
- **Status:** Called but exits early (line 717-718 in `decorators.py`)

---

## Why MLflow Shows "0/0 Activities"

**Execution Flow:**

```
1. User runs `promptchain`
   ↓
2. CLI imports observability module
   ↓
3. ghost.py evaluates _ENABLED = is_enabled() → False (env var not set)
   ↓
4. conditional_decorator() returns make_ghost_decorator() for ALL decorators
   ↓
5. @track_llm_call becomes identity function: def ghost(func): return func
   ↓
6. run_model_async is NOT wrapped (confirmed: no closure)
   ↓
7. init_mlflow() exits early (disabled check at line 678)
   ↓
8. Chain executes with callbacks ONLY
   ↓
9. Decorators never execute (they ARE the original functions)
   ↓
10. No MLflow runs created
   ↓
11. User sees "0/0 activities"
```

**Proof Points:**
- Decorator wrapping test confirms no closure (ghost pattern active)
- Callback system captures 6 events (old system works)
- MLflow experiment check fails (no module named 'mlflow')
- Environment check shows `PROMPTCHAIN_MLFLOW_ENABLED: NOT SET`

---

## Interference Analysis

### Current State (Both systems present, MLflow disabled)
**Interference:** ✗ NONE
**Reason:** Ghost pattern makes decorators completely transparent (identity functions). No performance overhead, no conflicts.

### If MLflow Were Enabled (PROMPTCHAIN_MLFLOW_ENABLED=true + MLflow installed)
**Interference:** ✓ LIKELY
**Predicted Conflicts:**

1. **Event Duplication:**
   - CallbackManager emits `MODEL_CALL_START` (line 871)
   - `@track_llm_call` decorator tracks same call (line 1834)
   - **Result:** Same execution tracked twice in different systems

2. **Resource Consumption:**
   - CallbackManager: In-memory event queue + ExecutionHistoryManager
   - MLflow: Background queue + HTTP requests to tracking server
   - **Result:** 2x memory overhead, 2x processing

3. **Execution Order Ambiguity:**
   - Decorator wraps function (executes before/after)
   - CallbackManager emits from within function (executes during)
   - **Question:** Which system's timestamps are authoritative?

4. **Token Counting Conflicts:**
   - ExecutionHistoryManager uses tiktoken for token limits
   - MLflow decorator extracts tokens from LiteLLM response
   - **Question:** Are both systems tracking same token counts?

5. **Error Handling Conflicts:**
   - CallbackManager emits error events
   - Decorator logs exceptions to MLflow then re-raises
   - **Result:** Same error logged twice

---

## Registration Conflicts

### ExecutionHistoryManager ↔ @track_llm_call

**Overlap:** Both track LLM execution metadata

```python
# ExecutionHistoryManager (callback-based)
- Entry types: "user_input", "agent_output", "tool_call", "tool_result"
- Token counting: tiktoken-based truncation
- Storage: In-memory deque with max_tokens/max_entries limits
- Integration: Receives events from CallbackManager

# @track_llm_call (decorator-based)
- Metrics: execution_time_seconds, prompt_tokens, completion_tokens, total_tokens
- Parameters: model, temperature, max_tokens, etc.
- Storage: MLflow backend (HTTP to tracking server)
- Integration: Wraps function directly
```

**Potential Conflict:** If MLflow enabled, same token counts logged twice:
1. ExecutionHistoryManager adds entry from `MODEL_CALL_END` callback
2. Decorator extracts `result.usage.total_tokens` and logs to MLflow

### CallbackManager Events ↔ Decorator Tracking

**Overlap:** Both track execution lifecycle

```python
# CallbackManager Events
CHAIN_START → @track_session
STEP_START → (no decorator equivalent)
MODEL_CALL_START → @track_llm_call (before execution)
MODEL_CALL_END → @track_llm_call (after execution)
STEP_END → (no decorator equivalent)
CHAIN_END → @track_session
TOOL_CALL → @track_tool
ROUTING_DECISION → @track_routing

# Decorator Coverage
@track_session() → Wraps entire CLI session
@track_llm_call() → Wraps run_model_async()
@track_task() → Wraps task operations
@track_routing() → Wraps agent routing
@track_tool() → Wraps MCP tool calls
```

**Design Difference:**
- **Callbacks:** Event-driven, emitted at specific points during execution
- **Decorators:** Function-level, wrap entire function execution

**Conflict Scenario:** If both enabled:
```python
# User calls PromptChain
@track_session()  # Decorator starts MLflow run
def _launch_tui():
    chain = PromptChain()
    chain.callback_manager.emit(CHAIN_START)  # Callback fires

    @track_llm_call()  # Decorator starts nested run
    async def run_model_async():
        callback_manager.emit(MODEL_CALL_START)  # Callback fires
        result = await acompletion()
        callback_manager.emit(MODEL_CALL_END)  # Callback fires
        return result  # Decorator logs metrics

    chain.callback_manager.emit(CHAIN_END)  # Callback fires
# Decorator logs session metrics
```

**Result:** Same execution tracked by 2 systems with different granularity.

---

## Evidence-Based Recommendations

### Option 1: Enable MLflow (Activate New System)
**Steps:**
1. `pip install mlflow`
2. `export PROMPTCHAIN_MLFLOW_ENABLED=true`
3. `mlflow ui --backend-store-uri sqlite:///mlruns.db` (start server)

**Expected Outcome:**
- Decorators will wrap functions (ghost pattern disabled)
- MLflow runs will be created
- Callback system still active → **DUAL TRACKING** (both systems run)

**Risk:** Resource overhead, event duplication, unclear which system is authoritative.

**Verdict:** ⚠️ NOT RECOMMENDED (without callback system removal)

---

### Option 2: Disable Callback System (Unify on MLflow)
**Steps:**
1. Enable MLflow (as above)
2. Remove CallbackManager initialization (line 234 in `promptchaining.py`)
3. Remove all `callback_manager.emit()` calls (50+ locations)
4. Update `AgenticStepProcessor` to not require `callback_manager` (line 946)
5. Update `ExecutionHistoryManager` to consume MLflow runs instead of callbacks

**Expected Outcome:**
- Single tracking system (MLflow only)
- No event duplication
- Clean architecture

**Risk:**
- Breaking change (callbacks are part of v0.4.1 API)
- ExecutionHistoryManager integration needs redesign
- Users with custom callbacks will break

**Effort:** HIGH (50+ callback sites to remove, API redesign)

**Verdict:** ✓ RECOMMENDED for long-term (breaking change in v0.5.0)

---

### Option 3: Bridge Systems (Hybrid Approach)
**Steps:**
1. Create `MLflowCallbackAdapter` that implements `CallbackFunction`
2. Register adapter with `CallbackManager` to forward events to MLflow
3. Remove `@track_*` decorators from PromptChain methods
4. Let callbacks be the single source of truth

**Implementation:**
```python
# promptchain/observability/callback_adapter.py
class MLflowCallbackAdapter:
    """Adapter that converts CallbackManager events to MLflow tracking."""

    def __call__(self, event: ExecutionEvent):
        if event.event_type == ExecutionEventType.MODEL_CALL_START:
            self._start_llm_run(event)
        elif event.event_type == ExecutionEventType.MODEL_CALL_END:
            self._end_llm_run(event)
        # ... etc

# promptchain/cli/main.py
def _launch_tui(...):
    from promptchain.observability.callback_adapter import MLflowCallbackAdapter

    init_mlflow()
    adapter = MLflowCallbackAdapter()

    # All chains in session will track to MLflow via callbacks
    chain.register_callback(adapter)
```

**Expected Outcome:**
- Single tracking system (callbacks)
- MLflow receives events via adapter
- No decorator overhead
- Backward compatible (callbacks remain)

**Risk:**
- Adapter needs to maintain state (map callback events to MLflow runs)
- Nested run tracking becomes complex

**Effort:** MEDIUM (1 adapter class, minimal changes)

**Verdict:** ✓ RECOMMENDED for short-term (backward compatible)

---

### Option 4: Disable New System (Status Quo)
**Steps:**
1. Remove `@track_*` decorators from codebase
2. Remove `promptchain/observability/` package
3. Document that v0.4.1 uses callbacks only

**Expected Outcome:**
- Single tracking system (callbacks)
- No MLflow dependency
- Clean removal of unused code

**Risk:**
- Loses spec 005 work
- No MLflow integration

**Verdict:** ✗ NOT RECOMMENDED (throws away completed work)

---

## Recommended Action Plan

### Phase 1: Immediate Fix (Enable MLflow with Adapter)
**Goal:** Get MLflow tracking working WITHOUT removing callbacks

```bash
# 1. Install MLflow
pip install mlflow

# 2. Enable in environment
export PROMPTCHAIN_MLFLOW_ENABLED=true
export MLFLOW_TRACKING_URI=http://localhost:5000

# 3. Start MLflow server
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

**Code Changes:**
```python
# promptchain/observability/callback_adapter.py (NEW FILE)
"""Adapter that bridges CallbackManager events to MLflow tracking."""

from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType
from promptchain.observability.queue import queue_log_metric, queue_log_param, queue_set_tag
from promptchain.observability.context import run_context
import time

class MLflowCallbackAdapter:
    """Convert callback events to MLflow operations."""

    def __init__(self):
        self._run_stack = []  # Track nested runs
        self._call_start_times = {}  # Track execution time

    def __call__(self, event: ExecutionEvent):
        """Handle callback event and log to MLflow."""
        if event.event_type == ExecutionEventType.CHAIN_START:
            self._run_stack.append(run_context("chain_execution").__enter__())

        elif event.event_type == ExecutionEventType.MODEL_CALL_START:
            call_id = id(event)  # Unique ID for this call
            self._call_start_times[call_id] = time.time()
            # Log model params from event metadata
            if hasattr(event, 'metadata') and 'model' in event.metadata:
                queue_log_param("model", event.metadata['model'])

        elif event.event_type == ExecutionEventType.MODEL_CALL_END:
            call_id = id(event)
            if call_id in self._call_start_times:
                duration = time.time() - self._call_start_times.pop(call_id)
                queue_log_metric("execution_time_seconds", duration)

            # Extract token counts from event metadata
            if hasattr(event, 'metadata'):
                if 'prompt_tokens' in event.metadata:
                    queue_log_metric("prompt_tokens", float(event.metadata['prompt_tokens']))
                if 'completion_tokens' in event.metadata:
                    queue_log_metric("completion_tokens", float(event.metadata['completion_tokens']))
                if 'total_tokens' in event.metadata:
                    queue_log_metric("total_tokens", float(event.metadata['total_tokens']))

        elif event.event_type == ExecutionEventType.CHAIN_END:
            if self._run_stack:
                self._run_stack.pop().__exit__(None, None, None)
```

```python
# promptchain/cli/main.py (MODIFY)
@track_session()  # Keep decorator for session-level tracking
def _launch_tui(...):
    init_mlflow()

    try:
        # ... existing setup ...

        # Register MLflow adapter for callback bridging
        from promptchain.observability.callback_adapter import MLflowCallbackAdapter
        adapter = MLflowCallbackAdapter()

        # This will be set on all chains created in session
        # (requires passing to PromptChain constructor)
        app = PromptChainApp(
            ...,
            mlflow_adapter=adapter  # Pass to TUI
        )
```

**Testing:**
```bash
# Run diagnostic again
python test_observability_conflict.py

# Expected output:
# MLflow tracking is ENABLED
# MLflow runs: 1 (session run created)
# Callback system is ACTIVE
# Adapter is bridging events to MLflow
```

---

### Phase 2: Validation (Verify Dual Tracking)
**Goal:** Confirm both systems work without conflicts

**Test Script:**
```python
import asyncio
from promptchain.utils.promptchaining import PromptChain
from promptchain.observability import init_mlflow, shutdown_mlflow

async def test_dual_tracking():
    init_mlflow()

    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Respond with 'test'"]
    )

    # Register callback tracker
    callback_events = []
    chain.register_callback(lambda e: callback_events.append(e.event_type))

    # Execute
    result = await chain.process_prompt_async("test")

    shutdown_mlflow()

    print(f"Callback events: {len(callback_events)}")
    print(f"MLflow runs: (check UI)")

asyncio.run(test_dual_tracking())
```

**Validation Checklist:**
- [ ] Callback events captured (old system works)
- [ ] MLflow runs created (new system works)
- [ ] No exceptions during execution
- [ ] Token counts match between systems
- [ ] Execution times reasonable (<5% overhead)

---

### Phase 3: Long-Term Unification (v0.5.0)
**Goal:** Decide on single tracking system

**Decision Matrix:**

| Factor | Keep Callbacks | Keep MLflow | Use Adapter |
|--------|---------------|-------------|-------------|
| Backward compatibility | ✓ High | ✗ Breaking | ✓ High |
| Industry standard | ✗ Custom | ✓ MLflow | ✓ MLflow |
| UI/visualization | ✗ None | ✓ Built-in | ✓ Built-in |
| Dependency weight | ✓ Zero | ✗ MLflow | ✗ MLflow |
| Code complexity | ✓ Low | ✓ Low | ✗ Medium |
| Performance overhead | ✓ Minimal | ✗ HTTP calls | ✗ HTTP calls |

**Recommendation:** **Use Adapter** (Option 3)
- Keeps backward compatibility
- Enables MLflow UI
- Single source of truth (callbacks)
- Minimal code changes

**Future Cleanup (v0.6.0):**
- Once adapter is stable, consider removing decorator system entirely
- Keep callbacks as primary tracking mechanism
- MLflow becomes optional backend (via adapter)

---

## Appendix: File Locations

### Active System (Callbacks)
```
promptchain/utils/execution_callback.py         - CallbackManager
promptchain/utils/execution_events.py           - Event types
promptchain/utils/execution_history_manager.py  - Token-aware history
promptchain/utils/promptchaining.py:234         - CallbackManager init
promptchain/utils/promptchaining.py:522-1790    - 50+ emit() calls
```

### Inactive System (MLflow Decorators)
```
promptchain/observability/__init__.py           - Public API
promptchain/observability/decorators.py         - @track_* decorators
promptchain/observability/ghost.py:17           - _ENABLED = False
promptchain/observability/config.py:12          - DEFAULT_MLFLOW_ENABLED = False
promptchain/observability/queue.py              - Background logger (unused)
promptchain/observability/context.py            - Run tracking (unused)
promptchain/utils/promptchaining.py:14          - Import (ghost)
promptchain/utils/promptchaining.py:1834        - Decorator (no-op)
promptchain/cli/main.py:19                      - Import (ghost)
promptchain/cli/main.py:197                     - Decorator (no-op)
```

### Integration Points
```
promptchain/cli/main.py:201                     - init_mlflow() (exits early)
promptchain/cli/main.py:247                     - shutdown_mlflow() (exits early)
```

---

## Appendix: Ghost Pattern Mechanics

The ghost pattern is why MLflow tracking is completely transparent when disabled:

```python
# At module import time (promptchain/observability/ghost.py)
_ENABLED = is_enabled()  # Evaluates ONCE when module loads

# In conditional_decorator()
def conditional_decorator(tracking_decorator):
    if _ENABLED:  # Never true (env var not set, MLflow not installed)
        return tracking_decorator  # Would return actual decorator
    else:
        return make_ghost_decorator()  # Returns this

def make_ghost_decorator():
    def ghost(func):
        return func  # Identity function - does nothing
    return ghost

# Result when applied
@track_llm_call(...)  # Becomes @ghost
async def run_model_async():  # Which becomes: def ghost(func): return func
    pass

# So effectively:
run_model_async = ghost(run_model_async) = run_model_async
# No wrapping, no overhead, no tracking
```

**Evidence this is active:**
- `run_model_async.__closure__` is `None` (no decorator closure)
- `run_model_async.__wrapped__` doesn't exist (no functools.wraps preservation)
- Execution test shows decorators have no effect

---

## Conclusion

**Current State:**
- MLflow system is **completely disabled** (ghost pattern + missing env var + no MLflow)
- Callback system is **fully operational** (capturing 6 event types)
- NO conflicts because decorators are identity functions
- User sees "0/0 activities" because MLflow never initializes

**Root Cause:**
- `PROMPTCHAIN_MLFLOW_ENABLED` environment variable not set (defaults to `False`)
- MLflow not installed
- Ghost pattern correctly returns no-op decorators when disabled

**Recommended Fix:**
1. **Short-term:** Install MLflow + set env var + implement adapter (Option 3)
2. **Long-term:** Keep adapter pattern, make MLflow optional backend (v0.5.0)
3. **Cleanup:** Consider removing decorator code if adapter proves sufficient (v0.6.0)

**Why User Sees Zero Tracking:**
The ghost pattern is working **exactly as designed** - when MLflow is disabled, decorators become transparent. The issue is that the user **expected** MLflow to be active, but didn't set the required environment variables or install dependencies.

**Next Steps:**
1. Set `PROMPTCHAIN_MLFLOW_ENABLED=true`
2. Install MLflow (`pip install mlflow`)
3. Start MLflow server (`mlflow ui`)
4. Implement callback adapter to bridge systems
5. Verify both systems track without conflicts
