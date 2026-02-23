# Observability Data Flow Architecture

## Executive Summary

The TUI Observability Panel and Activity Logs are empty because **two separate systems exist with NO connection**:

1. **MLflow Observability** (Wave 4-US1): Working, tracks LLM calls to MLflow
2. **TUI Display Widgets**: Working, but receive NO data from decorators

**The missing piece**: Event bridge connecting decorator events → TUI widgets.

---

## Data Flow Comparison

### CURRENT STATE (Broken)

```
User Action (sends message in CLI)
         ↓
PromptChain.process_prompt()
         ↓
@track_llm_call decorator wraps run_model_async()
         ↓
Background Queue receives event
         ↓
         ├─→ MLflow Storage (✅ WORKING)
         │   - Metrics: execution_time, tokens
         │   - Params: model, temperature
         │
         └─→ TUI Widgets (❌ NO CONNECTION)
             - ObservePanel: self._entries = [] (empty)
             - ActivityLogViewer: searcher.grep_logs() returns []


Result: MLflow has data, TUI shows nothing.
```

### REQUIRED STATE (Fixed)

```
User Action (sends message in CLI)
         ↓
PromptChain.process_prompt()
         ↓
@track_llm_call decorator wraps run_model_async()
         ↓
         ├─→ Background Queue
         │         ↓
         │   MLflow Storage (✅ existing)
         │
         └─→ TUI Event Bridge (🆕 NEEDED)
                   ↓
         ┌─────────┴──────────┐
         ↓                    ↓
   ObservePanel          ActivityLogger
   .log_llm_request()    .log_activity()
         ↓                    ↓
   TUI updates           JSONL + SQLite
   (real-time)                ↓
                      ActivityLogViewer
                      .load_activities()


Result: Both MLflow AND TUI have data.
```

---

## Component Interaction Map

### Layer 1: Execution Layer (User Code)

```python
# File: promptchain/utils/promptchain.py
class PromptChain:
    @track_llm_call(model_param="model_name", extract_args=["temperature"])
    async def run_model_async(self, model_name, messages, **kwargs):
        # Decorated method automatically tracked
        response = await litellm.acompletion(...)
        return response
```

**Current Behavior**:
- Decorator captures execution
- Sends to background queue
- ❌ NO TUI notification

### Layer 2: Observability Layer (Tracking)

```python
# File: promptchain/observability/decorators.py
def track_llm_call(model_param="model_name", extract_args=None):
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Start nested run
            with run_context(f"llm_call_{model_name}"):
                # Log to MLflow via queue
                queue_log_param("model", model_name)
                queue_log_metric("execution_time_seconds", execution_time)

                # ❌ MISSING: TUI notification
                # tui_observer.on_llm_call(model_name, messages, response)

                return result
        return async_wrapper
    return decorator
```

**Current Behavior**:
- Queues MLflow operations
- Returns result
- ❌ TUI widgets never notified

### Layer 3: Queue Layer (Processing)

```python
# File: promptchain/observability/queue.py
class BackgroundLogger:
    def _worker(self):
        while not self.shutdown_flag.is_set():
            operation, args, kwargs = self.queue.get(timeout=0.1)

            # Execute MLflow operation
            operation(*args, **kwargs)  # e.g., log_metric(key, value)

            # ❌ MISSING: TUI event emission
            # for observer in self.observers:
            #     observer.on_queue_processed(operation, args, kwargs)

            self.queue.task_done()
```

**Current Behavior**:
- Processes queue items
- Writes to MLflow
- ❌ No observer pattern

### Layer 4: TUI Layer (Display)

#### ObservePanel (Real-time View)

```python
# File: promptchain/cli/tui/observe_panel.py
class ObservePanel(Container):
    def __init__(self):
        self._entries: List[dict] = []  # ← PROBLEM: Always empty

    def log_entry(self, entry_type, content, metadata=None):
        # Manual call required - no auto-population
        self._entries.append({...})
        self._update_display()

    # ❌ MISSING: Event subscription
    # def subscribe_to_queue(self, background_logger):
    #     background_logger.add_observer(self)
    #
    # def on_queue_processed(self, operation, args, kwargs):
    #     if operation.__name__ == "log_metric":
    #         self.log_entry("llm-response", f"Metric: {args[0]}={args[1]}")
```

**Current Behavior**:
- Has methods to display data
- ❌ Never receives data automatically
- Only manually called in ONE place (app.py line 1013)

#### ActivityLogViewer (Search View)

```python
# File: promptchain/cli/tui/activity_log_viewer.py
class ActivityLogViewer(Container):
    def __init__(self, session_name, log_dir, db_path):
        # Reads from ActivityLogger storage
        self.searcher = ActivitySearcher(
            session_name=session_name,
            log_dir=log_dir,  # ← PROBLEM: ActivityLogger not populated
            db_path=db_path
        )

    def load_activities(self, pattern=".*", limit=50):
        # Searches ActivityLogger's JSONL files
        self.activities = self.searcher.grep_logs(
            pattern=pattern,
            agent_name=...,
            activity_type=...
        )  # ← Returns [] because ActivityLogger has no data
```

**Current Behavior**:
- Searches ActivityLogger storage
- ❌ ActivityLogger never receives decorator events
- Returns empty results

---

## Integration Points - Where to Connect

### Integration Point 1: Decorator → Observer

**Location**: `promptchain/observability/decorators.py`

**Change Needed**:
```python
def track_llm_call(model_param="model_name", extract_args=None):
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # ... existing MLflow logging ...

            # NEW: Notify TUI observer
            from .tui_bridge import get_tui_observer
            tui_observer = get_tui_observer()
            if tui_observer:
                tui_observer.on_llm_call_start(model_name, messages)

            result = await func(*args, **kwargs)

            if tui_observer:
                tui_observer.on_llm_call_complete(model_name, result, execution_time)

            return result
        return async_wrapper
    return decorator
```

### Integration Point 2: Queue → Observer Pattern

**Location**: `promptchain/observability/queue.py`

**Change Needed**:
```python
class BackgroundLogger:
    def __init__(self, maxsize=1000):
        self.queue = queue.Queue(maxsize=maxsize)
        self.observers: List[QueueObserver] = []  # NEW

    def add_observer(self, observer: QueueObserver):
        """Register observer for queue events."""
        self.observers.append(observer)

    def _worker(self):
        while not self.shutdown_flag.is_set():
            operation, args, kwargs = self.queue.get(timeout=0.1)

            # Execute operation
            operation(*args, **kwargs)

            # NEW: Notify observers
            for observer in self.observers:
                try:
                    observer.on_queue_event(operation.__name__, args, kwargs)
                except Exception as e:
                    logger.error(f"Observer error: {e}")

            self.queue.task_done()
```

### Integration Point 3: App → ObservePanel

**Location**: `promptchain/cli/tui/app.py`

**Change Needed**:
```python
class PromptChainApp(App):
    def on_mount(self):
        # ... existing setup ...

        # NEW: Connect ObservePanel to observability system
        if self.verbose_mode and self.observe_panel:
            from promptchain.observability.tui_bridge import TUIObserver
            tui_observer = TUIObserver(observe_panel=self.observe_panel)

            # Register with background queue
            from promptchain.observability.queue import _background_logger
            _background_logger.add_observer(tui_observer)
```

### Integration Point 4: App → ActivityLogger

**Location**: `promptchain/cli/session_manager.py`

**Change Needed**:
```python
class SessionManager:
    def create_session(self, name, **kwargs):
        # ... existing session creation ...

        # NEW: Auto-initialize ActivityLogger
        from promptchain.cli.activity_logger import ActivityLogger

        session_dir = self.sessions_dir / session.id
        activity_log_dir = session_dir / "activity_logs"
        activity_db_path = session_dir / "activities.db"

        session._activity_logger = ActivityLogger(
            session_name=session.name,
            log_dir=activity_log_dir,
            db_path=activity_db_path,
            enable_console=False
        )

        # NEW: Register ActivityLogger with observability system
        from promptchain.observability.tui_bridge import get_activity_logger_bridge
        bridge = get_activity_logger_bridge()
        bridge.register_logger(session._activity_logger)

        return session
```

---

## Missing Components to Build

### Component 1: TUI Observer Interface

**File**: `promptchain/observability/tui_bridge.py` (NEW)

```python
"""Bridge between observability system and TUI widgets."""

from typing import Optional, Protocol

class QueueObserver(Protocol):
    """Observer interface for background queue events."""

    def on_queue_event(self, operation_name: str, args: tuple, kwargs: dict):
        """Called when queue processes an event."""
        ...


class TUIObserver:
    """Connects background queue events to TUI ObservePanel."""

    def __init__(self, observe_panel):
        self.observe_panel = observe_panel

    def on_queue_event(self, operation_name: str, args: tuple, kwargs: dict):
        """Forward queue events to ObservePanel."""
        if operation_name == "log_metric":
            key, value = args[:2]
            self.observe_panel.log_entry("info", f"Metric: {key}={value}")

        elif operation_name == "log_param":
            key, value = args[:2]
            self.observe_panel.log_entry("info", f"Param: {key}={value}")


class ActivityLoggerBridge:
    """Connects observability events to ActivityLogger."""

    def __init__(self):
        self.loggers = []

    def register_logger(self, activity_logger):
        self.loggers.append(activity_logger)

    def on_llm_call(self, model_name: str, execution_time: float, **kwargs):
        """Log LLM call to all registered ActivityLoggers."""
        for logger in self.loggers:
            logger.log_activity(
                activity_type="agent_output",
                content=f"LLM call to {model_name} ({execution_time:.2f}s)",
                metadata=kwargs
            )


# Singleton instances
_tui_observer: Optional[TUIObserver] = None
_activity_logger_bridge = ActivityLoggerBridge()


def get_tui_observer() -> Optional[TUIObserver]:
    """Get global TUI observer instance."""
    return _tui_observer


def set_tui_observer(observer: TUIObserver):
    """Set global TUI observer."""
    global _tui_observer
    _tui_observer = observer


def get_activity_logger_bridge() -> ActivityLoggerBridge:
    """Get global ActivityLogger bridge."""
    return _activity_logger_bridge
```

### Component 2: Enhanced Decorators

**File**: `promptchain/observability/decorators.py` (MODIFY)

Add TUI notifications to existing decorators:

```python
def track_llm_call(model_param="model_name", extract_args=None):
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # ... existing MLflow logging ...

            # NEW: Notify TUI
            tui_observer = get_tui_observer()
            if tui_observer:
                tui_observer.on_queue_event(
                    "llm_call_start",
                    (model_name, messages),
                    {}
                )

            result = await func(*args, **kwargs)

            # NEW: Notify TUI on completion
            if tui_observer:
                tui_observer.on_queue_event(
                    "llm_call_complete",
                    (model_name, execution_time),
                    {"tokens": total_tokens}
                )

            # NEW: Notify ActivityLogger
            activity_bridge = get_activity_logger_bridge()
            activity_bridge.on_llm_call(
                model_name=model_name,
                execution_time=execution_time,
                tokens=total_tokens
            )

            return result
        return async_wrapper
    return decorator
```

---

## Implementation Phases

### Phase 1: Enable Observability (5 min)
1. Set `DEFAULT_MLFLOW_ENABLED = True` in config.py
2. OR add startup check in main.py with helpful error
3. Verify background queue starts

**Verification**: `python diagnostic_observability.py` passes config check

### Phase 2: Build TUI Bridge (30 min)
1. Create `promptchain/observability/tui_bridge.py`
2. Implement `TUIObserver` class
3. Add observer pattern to `BackgroundLogger`
4. Register observer in `PromptChainApp.on_mount()`

**Verification**: ObservePanel shows real-time entries during LLM calls

### Phase 3: Connect ActivityLogger (30 min)
1. Auto-initialize ActivityLogger in SessionManager
2. Create `ActivityLoggerBridge` in tui_bridge.py
3. Hook decorators to bridge
4. Verify JSONL files populated

**Verification**: ActivityLogViewer shows searchable activity history

### Phase 4: Complete Integration (15 min)
1. Add @track_task decorator to TaskController
2. Add @track_routing to AgentChain router
3. Ensure all execution paths logged
4. Add auto-refresh to ActivityLogViewer

**Verification**: All agent interactions visible in both panels

---

## Quick Fixes Summary

### Fix 1: Enable Observability
```bash
# In .env or environment
export PROMPTCHAIN_MLFLOW_ENABLED=true
```

### Fix 2: Create Event Bridge
```python
# NEW FILE: promptchain/observability/tui_bridge.py
class TUIObserver:
    def on_queue_event(self, operation_name, args, kwargs):
        # Forward to ObservePanel
        ...
```

### Fix 3: Wire TUI to Bridge
```python
# promptchain/cli/tui/app.py
def on_mount(self):
    tui_observer = TUIObserver(self.observe_panel)
    _background_logger.add_observer(tui_observer)
```

### Fix 4: Auto-init ActivityLogger
```python
# promptchain/cli/session_manager.py
def create_session(self, name, **kwargs):
    session._activity_logger = ActivityLogger(...)
    get_activity_logger_bridge().register_logger(session._activity_logger)
```

---

## Files to Create/Modify

### CREATE
- `promptchain/observability/tui_bridge.py` - Event bridge between systems

### MODIFY
- `promptchain/observability/queue.py` - Add observer pattern
- `promptchain/observability/decorators.py` - Add TUI notifications
- `promptchain/cli/tui/app.py` - Register observers on mount
- `promptchain/cli/session_manager.py` - Auto-init ActivityLogger
- `promptchain/observability/config.py` - Enable by default or add checks

---

## Testing Checklist

- [ ] Background queue worker thread starts
- [ ] ObservePanel receives decorator events
- [ ] ActivityLogger captures all agent interactions
- [ ] ActivityLogViewer displays searchable history
- [ ] Real-time updates during LLM execution
- [ ] No performance degradation (<5ms overhead)
- [ ] Works with observability disabled (graceful degradation)
