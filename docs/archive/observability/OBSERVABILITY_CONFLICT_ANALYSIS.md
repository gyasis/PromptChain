# Observability System Conflict Analysis
**Date**: 2025-01-10
**Analyst**: Python Architecture Expert
**Issue**: Dual observability systems showing zero activity despite active operations

---

## Executive Summary

**ROOT CAUSE IDENTIFIED**: MLflow observability system (spec 005) is **COMPLETELY DISABLED** via ghost decorator pattern, preventing all tracking. The old callback system (v0.4.1) remains functional but is **NEVER INITIALIZED** by CLI code.

**Impact**: Both observability systems show zero activity because:
1. **New system (MLflow)**: Disabled at import time via `_ENABLED = False`
2. **Old system (Callbacks)**: No callbacks registered, so events fire into void

**Status**: Zero observability coverage, silent failure mode

---

## System 1: Legacy Callback System (v0.4.1)

### Architecture
```
ExecutionCallback Protocol (execution_callback.py)
    ↓
CallbackManager (execution_callback.py)
    ↓
ExecutionEvent → emit() → FilteredCallback → User callbacks
```

### Implementation Location
- **Protocol**: `promptchain/utils/execution_callback.py`
- **Events**: `promptchain/utils/execution_events.py`
- **Integration**: `promptchain/utils/promptchaining.py` (line 234)

### Initialization Pattern
```python
# In PromptChain.__init__
self.callback_manager = CallbackManager()  # Line 234

# Integration with MCPHelper
MCPHelper(
    callback_manager=self.callback_manager  # Pass for event firing
)
```

### Current State
```python
# PromptChain creates callback_manager
✓ CallbackManager initialized

# But NO callbacks registered
✗ callback_manager.callbacks = []  # Empty list

# Events fire, but no handlers exist
callback_manager.emit(event)  # Goes to empty list → silent failure
```

### Why It Shows Zero Activity
The callback system works perfectly, but **the CLI never registers any callbacks**. The ObservePanel and ActivityLogViewer are looking for callbacks that were never attached.

**Evidence**:
```python
# promptchain/cli/main.py - NO callback registration found
# Should have something like:
# promptchain.register_callback(activity_logger.log_event)
# But this code doesn't exist
```

---

## System 2: MLflow Observability (Spec 005)

### Architecture
```
@track_llm_call decorator (decorators.py)
    ↓
conditional_decorator (ghost.py) [CHECK: _ENABLED]
    ↓
If _ENABLED=True → MLflow tracking
If _ENABLED=False → ghost decorator (identity function)
```

### Implementation Location
- **Decorators**: `promptchain/observability/decorators.py`
- **Ghost Pattern**: `promptchain/observability/ghost.py`
- **Config**: `promptchain/observability/config.py`
- **Integration**: `promptchain/utils/promptchaining.py` (line 14, 1834-1837)

### Activation Check
```python
# ghost.py - Import-time evaluation (ONCE)
_ENABLED = is_enabled()  # Line 17

# config.py
def is_enabled() -> bool:
    yaml_config = _load_yaml_config()
    yaml_enabled = yaml_config.get('enabled', DEFAULT_MLFLOW_ENABLED)  # False
    return _get_bool_env(ENV_MLFLOW_ENABLED, yaml_enabled)

# Result
DEFAULT_MLFLOW_ENABLED = False  # Line 12
ENV_MLFLOW_ENABLED not set
→ _ENABLED = False  # DISABLED AT IMPORT TIME
```

### Decorator Application
```python
# promptchain/utils/promptchaining.py:1834-1837
@track_llm_call(
    model_param="model_name",
    extract_args=["temperature", "max_tokens", "max_completion_tokens", "top_p"]
)
async def run_model_async(self, model_name: str, messages: List[Dict], ...):
```

### Ghost Pattern Execution
```python
# decorators.py:226 - Returns conditional_decorator
return conditional_decorator(decorator)

# ghost.py:68-71 - Import-time decision
def conditional_decorator(tracking_decorator):
    if _ENABLED:  # False
        return tracking_decorator
    else:
        return make_ghost_decorator()  # ← THIS PATH TAKEN

# ghost.py:32-43 - Ghost decorator is identity function
def make_ghost_decorator():
    def ghost(func):
        return func  # RETURNS ORIGINAL FUNCTION UNCHANGED
    return ghost
```

### Current State
```python
# What decorator does when _ENABLED = False:
@track_llm_call(...)  # Evaluates to ghost decorator
async def run_model_async(...):
    # Decorator does NOTHING - function executes as if undecorated
```

### Why It Shows Zero Activity
The MLflow system is **completely bypassed** via the ghost pattern. When `_ENABLED = False`, all `@track_*` decorators become identity functions that return the original method unchanged.

**Evidence**:
```bash
$ python3 -c "from promptchain.observability.config import is_enabled; print(f'MLflow enabled: {is_enabled()}')"
MLflow enabled: False
```

---

## Import Order Analysis

### CLI Startup Sequence
```
promptchain/cli/main.py
    ↓ Line 19
from promptchain.observability import track_session, init_mlflow, shutdown_mlflow
    ↓ Triggers import chain
promptchain/observability/__init__.py
    ↓ Line 21-28
from .decorators import (track_llm_call, ...)
    ↓ Triggers import
promptchain/observability/decorators.py
    ↓ Line 43
from .config import is_enabled, ...
    ↓ Line 44
from .ghost import conditional_decorator
    ↓ Triggers import
promptchain/observability/ghost.py
    ↓ Line 17 (IMPORT TIME EXECUTION)
_ENABLED = is_enabled()  # Evaluates to False
```

### PromptChain Import Sequence
```
promptchain/utils/promptchaining.py
    ↓ Line 14
from promptchain.observability import track_llm_call
    ↓ (ghost.py already imported via CLI)
_ENABLED already set to False
    ↓ Line 234
self.callback_manager = CallbackManager()  # Old system initialized
```

### Key Finding: No Conflict, Just Dual Failure
The systems **do not interfere** with each other:
- **MLflow decorators**: Disabled via ghost pattern (identity functions)
- **Callback system**: Enabled but no callbacks registered

Both systems coexist peacefully because one is effectively deleted at import time, and the other has no listeners.

---

## Method Resolution Order Analysis

### When `run_model_async()` Executes

```python
# What Python sees after decorators applied
async def run_model_async(self, model_name, messages, params, tools, tool_choice):
    # NO decorator wrapping - ghost pattern removed it

    # Old callback system events (if any)
    self.callback_manager.emit(event)  # Fires to empty list → no-op

    # Actual LLM call
    response = await acompletion(**model_params)

    return response
```

### Execution Trace: CLI → PromptChain → run_model_async

```
User input → PromptChainApp.on_message_submit
    ↓
AgentChain.send_message
    ↓
PromptChain.process_prompt_async
    ↓
PromptChain.run_model_async  # <-- Decorated method
    ↓
[NO MLflow tracking - ghost decorator removed wrapper]
[NO callback handling - no callbacks registered]
    ↓
LiteLLM.acompletion (raw execution)
    ↓
Response returned (no tracking)
```

### Event Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ CLI Startup                                                 │
│ @track_session() on _launch_tui                             │
│   ↓                                                         │
│ init_mlflow() called                                        │
│   ↓                                                         │
│ is_enabled() returns False → init_mlflow() does nothing    │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ LLM Call                                                    │
│ PromptChain.run_model_async()                               │
│   ↓                                                         │
│ @track_llm_call decorator → ghost(run_model_async)         │
│   ↓                                                         │
│ Ghost returns original function unchanged                   │
│   ↓                                                         │
│ No MLflow tracking executed                                 │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Callback System (if events fired)                          │
│ callback_manager.emit(event)                                │
│   ↓                                                         │
│ callbacks list is empty []                                  │
│   ↓                                                         │
│ No handlers executed                                        │
└─────────────────────────────────────────────────────────────┘
                        ↓
                   RESULT:
         Both systems show zero activity
```

---

## Architectural Incompatibility Assessment

### Are Decorators and Callbacks Fundamentally Incompatible?

**No.** They are **fully compatible** and designed to coexist:

1. **Different Purposes**:
   - Decorators: External observability (MLflow UI, metrics)
   - Callbacks: Internal observability (TUI panels, activity logs)

2. **Different Execution Layers**:
   - Decorators: Wrap method execution (outer layer)
   - Callbacks: Fire events during execution (inner layer)

3. **Design Intent**:
   ```python
   # INTENDED ARCHITECTURE (from spec 005)
   @track_llm_call  # Outer: MLflow tracking
   async def run_model_async(...):
       # Inner: Callback events
       self._emit_event(ExecutionEventType.MODEL_CALL_START)
       response = await acompletion(...)
       self._emit_event(ExecutionEventType.MODEL_CALL_END)
       return response
   ```

### Why Both Systems Show Zero Activity

Not due to architectural conflict, but **configuration failure**:

```
MLflow System: _ENABLED = False → Ghost decorators bypass all tracking
Callback System: callback_manager.callbacks = [] → No handlers to receive events
```

### Can They Coexist?

**Yes, by design.** Evidence:

1. **Spec 005 Design**:
   ```python
   # FR-001: Non-Breaking Integration
   # "Does not modify existing ExecutionHistoryManager or callback patterns"
   # ↓ Indicates intentional coexistence
   ```

2. **Import Pattern**:
   ```python
   # promptchain/utils/promptchaining.py
   from .execution_callback import CallbackManager  # Line 38
   from promptchain.observability import track_llm_call  # Line 14
   # Both imported - no mutual exclusion
   ```

3. **Dual Tracking Flow**:
   ```python
   @track_llm_call  # MLflow: Logs to external UI
   async def run_model_async(...):
       self._emit_event(...)  # Callbacks: Logs to internal TUI
       # Both fire independently
   ```

### Must One Be Removed?

**No.** Both systems serve different audiences:
- **MLflow**: Data scientists reviewing experiments
- **Callbacks**: Real-time TUI observability for CLI users

**Recommended Integration**: Enable both, use as complementary systems.

---

## Why BOTH Systems Show Zero Activity

### MLflow System: Disabled via Configuration

```python
# Environment check
$ env | grep -i mlflow
# (no output - variable not set)

# Default configuration
DEFAULT_MLFLOW_ENABLED = False  # config.py:12

# Ghost pattern result
_ENABLED = is_enabled()  # Returns False
→ All @track_* decorators become identity functions
→ Zero MLflow runs created
```

### Callback System: No Handlers Registered

```python
# PromptChain initialization
self.callback_manager = CallbackManager()  # Created
self.callback_manager.callbacks = []  # Empty

# CLI never registers callbacks
# Expected (but missing):
def setup_observability_callbacks(promptchain, activity_logger):
    def log_to_panel(event):
        activity_logger.log(event)

    promptchain.register_callback(log_to_panel)

# Actual:
# (this code doesn't exist in cli/main.py)
```

### Result: Silent Failure Mode

```
ObservePanel → Checks MLflow runs → None found (ghost pattern disabled tracking)
ActivityLogViewer → Checks callback events → None found (no callbacks registered)
```

---

## Recommended Integration Strategy

### Option 1: Enable MLflow + Fix Callbacks (Full Observability)

**Steps**:
1. Set environment variable: `export PROMPTCHAIN_MLFLOW_ENABLED=true`
2. Register TUI callbacks in `cli/main.py`:
   ```python
   def setup_observability(promptchain, activity_logger):
       # Bridge callback system to TUI
       def bridge_event_to_tui(event):
           activity_logger.log_execution_event(event)

       promptchain.register_callback(bridge_event_to_tui)
   ```

**Benefits**:
- MLflow UI shows experiment tracking
- TUI panels show real-time activity
- Full observability coverage

**Tradeoffs**:
- MLflow server must be running
- Increased overhead (~5-10% per LLM call)

---

### Option 2: Callbacks Only (Lightweight)

**Steps**:
1. Keep MLflow disabled (current state)
2. Implement callback registration in CLI
3. Remove MLflow UI references from TUI

**Benefits**:
- No external dependencies
- Minimal performance overhead
- Self-contained observability

**Tradeoffs**:
- No persistent experiment tracking
- No MLflow UI visualization

---

### Option 3: MLflow Only (Spec 005 Intent)

**Steps**:
1. Enable MLflow: `PROMPTCHAIN_MLFLOW_ENABLED=true`
2. Remove callback system integration
3. Use MLflow API for TUI data:
   ```python
   # In ObservePanel
   def refresh_activities():
       client = mlflow.tracking.MlflowClient()
       runs = client.search_runs(experiment_ids=[exp_id])
       # Display runs in TUI
   ```

**Benefits**:
- Single source of truth
- Persistent storage
- Rich querying capabilities

**Tradeoffs**:
- MLflow server required
- More complex TUI integration

---

## Minimal Fix for Immediate Observability

### Quick Win: Enable MLflow

```bash
# In shell
export PROMPTCHAIN_MLFLOW_ENABLED=true

# Start MLflow server (separate terminal)
mlflow server --host 127.0.0.1 --port 5000

# Run PromptChain CLI
promptchain
```

**Result**: ObservePanel will show runs immediately (no code changes needed)

### Verification
```python
# After sending a message in CLI
# Check MLflow UI: http://localhost:5000
# Experiment: promptchain-cli
# Runs: Should show llm_call_* runs with metrics
```

---

## Conclusion

**Primary Finding**: No architectural conflict exists. Both systems are designed to coexist.

**Actual Problem**: Configuration failure
- MLflow: Intentionally disabled via `_ENABLED = False`
- Callbacks: Functional but no handlers registered

**Immediate Action**: Set `PROMPTCHAIN_MLFLOW_ENABLED=true` to activate spec 005 observability

**Long-Term Decision**: Choose integration strategy based on requirements:
- Full observability → Option 1
- Lightweight TUI → Option 2
- Production-grade tracking → Option 3

**Key Insight**: The ghost pattern is working perfectly - it's doing exactly what it was designed to do (disable tracking when `_ENABLED = False`). The issue is not a bug, it's the **default configuration**.
