# TUI Observability Panel & Activity Logs - Empty Data Diagnosis

## Problem Statement

The TUI displays two observability features that show **NO DATA**:
1. **Observability Panel** (Ctrl+O): Shows "0/0 activities"
2. **Activity Logs Tab**: Empty log viewer

This is NOT an MLflow tracking issue but a **TUI data flow disconnect**.

---

## Architecture Analysis

### Data Flow: Decorator → Queue → Storage → TUI Display

```
┌─────────────────────────────────────────────────────────────────┐
│                    OBSERVATION DATA SOURCES                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  @track_llm_call       @track_task        @track_routing       │
│         │                   │                    │              │
│         └───────────────────┴────────────────────┘              │
│                             │                                   │
│                    Background Queue                             │
│                      (queue.py)                                 │
│                             │                                   │
│                   ┌─────────┴──────────┐                        │
│                   │                    │                        │
│              MLflow Storage      ActivityLogger                 │
│              (local/remote)       (JSONL+SQLite)                │
│                   │                    │                        │
└───────────────────┼────────────────────┼────────────────────────┘
                    │                    │
                    ↓                    ↓
          ┌──────────────────┐  ┌──────────────────┐
          │  ObservePanel    │  │ ActivityLogViewer│
          │  (real-time TUI) │  │ (search TUI)     │
          └──────────────────┘  └──────────────────┘
               ❌ EMPTY             ❌ EMPTY
```

---

## Root Cause Analysis

### 1. **TWO SEPARATE DATA SYSTEMS**

#### System 1: MLflow Observability (Wave 4 - US1)
- **Purpose**: LLM tracking for performance monitoring
- **Storage**: MLflow tracking server
- **Access**: Background queue → MLflow API
- **Status**: ✅ **WORKING** (decorators functional)

#### System 2: TUI Activity Display
- **Purpose**: Real-time verbose debugging for developers
- **Storage**: TUI widget state (in-memory)
- **Access**: Direct method calls on widgets
- **Status**: ❌ **NOT CONNECTED** to decorators

### 2. **CRITICAL MISSING LINK**

**ObservePanel** and **ActivityLogViewer** are **NOT** subscribed to the background queue or MLflow events.

```python
# File: promptchain/cli/tui/observe_panel.py
class ObservePanel(Container):
    def __init__(self):
        self._entries: List[dict] = []  # Internal state only

    def log_entry(self, entry_type, content, metadata=None):
        # Manual call required - no automatic subscription
        self._entries.append(entry)
        self._update_display()
```

**Problem**: The decorators (@track_llm_call, @track_task) write to MLflow, but ObservePanel has NO WAY to listen to those events.

---

## Disconnection Points

### Point 1: ObservePanel Never Receives Decorator Events

**Current Flow** (broken):
```
@track_llm_call(...)
async def run_model_async(self, model_name, messages):
    # This ONLY logs to MLflow via queue
    queue_log_metric("execution_time_seconds", execution_time)
    queue_log_param("model", model_name)
    # ObservePanel NEVER NOTIFIED ❌
```

**Expected Flow**:
```
@track_llm_call(...)
async def run_model_async(self, model_name, messages):
    # Should ALSO notify TUI widgets
    observe_panel.log_llm_request(model_name, prompt_preview)  # MISSING
    observe_panel.log_llm_response(model_name, response, tokens)  # MISSING
```

### Point 2: ActivityLogViewer Reads ActivityLogger, NOT MLflow

**Current State**:
- ActivityLogger (separate from MLflow) writes to JSONL + SQLite
- ActivityLogViewer reads from ActivityLogger storage
- **Gap**: Nothing connects @track_* decorators to ActivityLogger

**Evidence**:
```python
# File: promptchain/cli/tui/activity_log_viewer.py
class ActivityLogViewer(Container):
    def __init__(self, session_name, log_dir, db_path):
        # Reads from ActivityLogger's JSONL/SQLite
        self.searcher = ActivitySearcher(
            session_name=session_name,
            log_dir=log_dir,  # ← ActivityLogger's files
            db_path=db_path   # ← ActivityLogger's DB
        )
```

**Problem**: MLflow tracking (from decorators) and ActivityLogger are SEPARATE SYSTEMS.

---

## Configuration Issue

### Observability DISABLED by Default

**File**: `promptchain/observability/config.py`
```python
DEFAULT_MLFLOW_ENABLED = False  # ← Observability OFF by default
ENV_MLFLOW_ENABLED = "PROMPTCHAIN_MLFLOW_ENABLED"  # ← Correct env var
```

**User must set**:
```bash
export PROMPTCHAIN_MLFLOW_ENABLED=true  # Not OBSERVABILITY_ENABLED
```

---

## Integration Gap: What's Missing

### 1. **Event Bridge: MLflow → TUI**

The background queue processes decorator events but has NO HOOKS for TUI widgets:

```python
# File: promptchain/observability/queue.py
class BackgroundLogger:
    def _worker(self):
        while not self.shutdown_flag.is_set():
            operation, args, kwargs = self.queue.get(timeout=0.1)
            operation(*args, **kwargs)  # Executes MLflow logging
            # ❌ NO TUI NOTIFICATION HERE
```

**Needed**:
```python
# Pseudo-code for event bridge
class BackgroundLogger:
    def _worker(self):
        while not self.shutdown_flag.is_set():
            operation, args, kwargs = self.queue.get()
            operation(*args, **kwargs)

            # NEW: Notify TUI widgets
            if self.tui_observer:  # Event handler
                self.tui_observer.on_metric_logged(...)
                self.tui_observer.on_param_logged(...)
```

### 2. **ObservePanel Hook: Direct Calls in PromptChain**

Currently, only **one place** manually logs to ObservePanel:

```python
# File: promptchain/cli/tui/app.py (line 1013+)
if self.verbose_mode and self.observe_panel:
    if status_type == "tool_call":
        self.observe_panel.log_entry("tool-call", f"[Step {current_step}] {status}")
    elif status_type == "tool_result":
        self.observe_panel.log_entry("tool-result", f"[Step {current_step}] {status}")
    # ... etc
```

**Needed**: Hook all LLM calls, task operations, routing decisions to ObservePanel.

### 3. **ActivityLogger Integration: Automatic Capture**

ActivityLogger exists but is **NOT automatically connected** to AgentChain or PromptChain execution:

```python
# File: promptchain/cli/models/session.py
class Session:
    _activity_logger: Any = field(default=None)  # Lazy init

    @property
    def activity_logger(self):
        return self._activity_logger  # May be None
```

**Needed**: Auto-initialize ActivityLogger for all sessions and capture all agent interactions.

---

## Verification Steps

### Diagnostic Script Results

Run: `python diagnostic_observability.py`

**Expected Failures**:
1. ❌ **Config check**: `PROMPTCHAIN_MLFLOW_ENABLED=false` (default)
2. ❌ **Queue check**: Worker thread not started (observability disabled)
3. ❌ **TUI integration**: Widgets exist but have NO DATA SOURCE

**After enabling** (`PROMPTCHAIN_MLFLOW_ENABLED=true`):
1. ✅ Background queue processes items
2. ✅ MLflow records metrics
3. ❌ **TUI still empty** (no event bridge)

---

## Architecture Diagrams

### Current Reality

```
┌──────────────────────────┐      ┌─────────────────────────┐
│   @track_* Decorators    │      │   TUI Widgets           │
│                          │      │                         │
│  @track_llm_call         │      │  - ObservePanel         │
│  @track_task             │      │  - ActivityLogViewer    │
│  @track_routing          │      │                         │
│         │                │      │  NO CONNECTION ❌       │
│         ↓                │      │                         │
│  Background Queue        │      │  Empty displays         │
│         │                │      └─────────────────────────┘
│         ↓                │
│  MLflow Storage          │
│  (metrics, params)       │
└──────────────────────────┘
```

### Required Architecture

```
┌──────────────────────────────────────────────────────────┐
│                @track_* Decorators                       │
│                                                          │
│  @track_llm_call  @track_task  @track_routing           │
│         │              │              │                  │
│         └──────────────┴──────────────┘                  │
│                        │                                 │
│                        ↓                                 │
│              Background Queue (queue.py)                 │
│                        │                                 │
│           ┌────────────┼────────────┐                    │
│           ↓            ↓            ↓                    │
│   ┌──────────┐  ┌──────────┐  ┌──────────────┐         │
│   │  MLflow  │  │ Activity │  │  TUI Event   │         │
│   │ Storage  │  │  Logger  │  │   Bridge     │← NEW   │
│   │          │  │  (JSONL) │  │              │         │
│   └──────────┘  └──────────┘  └──────┬───────┘         │
│                                       │                  │
└───────────────────────────────────────┼──────────────────┘
                                        ↓
                        ┌───────────────────────────┐
                        │    TUI Widgets            │
                        │                           │
                        │  ObservePanel.log_entry() │
                        │  ActivityLogViewer.load() │
                        └───────────────────────────┘
```

---

## Quick Diagnostic Commands

### 1. Check if observability is enabled
```bash
python -c "from promptchain.observability.config import is_enabled; print(f'Enabled: {is_enabled()}')"
```

### 2. Check background queue status
```bash
python -c "from promptchain.observability.queue import _background_logger, get_queue_size; print(f'Worker alive: {_background_logger.worker.is_alive() if _background_logger.worker else False}'); print(f'Queue size: {get_queue_size()}')"
```

### 3. Check ActivityLogger integration
```bash
# Start CLI and check
promptchain --session test
# In TUI: Ctrl+L (toggle activity logs)
# Expected: "Activity logging is not enabled for this session" ← Confirms not connected
```

### 4. Check ObservePanel integration
```bash
# Start CLI with verbose mode
promptchain --session test --verbose
# In TUI: Ctrl+O (toggle observe panel)
# Expected: Panel opens but shows NO ENTRIES ← Confirms no decorator integration
```

---

## Solution Requirements

### Immediate Fixes Needed

1. **Enable Observability by Default** (or document clearly)
   - Change `DEFAULT_MLFLOW_ENABLED = True` OR
   - Add startup check with helpful error message

2. **Connect ActivityLogger to Session Lifecycle**
   - Auto-initialize ActivityLogger for all sessions
   - Hook into AgentChain execution to capture agent interactions

3. **Create TUI Event Bridge**
   - Add callback mechanism to background queue
   - Notify ObservePanel when metrics/params logged
   - Notify ActivityLogger when agent operations occur

4. **Hook Decorators to ObservePanel**
   - In @track_llm_call: Call observe_panel.log_llm_request/response
   - In @track_task: Call observe_panel.log_tool_call/result
   - In @track_routing: Call observe_panel.log_reasoning

5. **Wire ActivityLogViewer to Real Data**
   - Ensure ActivityLogger is capturing all @track_* events
   - Verify searcher.grep_logs() returns actual data
   - Add auto-refresh when observe panel receives new entries

---

## Files Requiring Changes

1. `promptchain/observability/queue.py` - Add TUI event callbacks
2. `promptchain/observability/decorators.py` - Add observe_panel notifications
3. `promptchain/cli/session_manager.py` - Auto-init ActivityLogger
4. `promptchain/cli/tui/app.py` - Wire event bridge to widgets
5. `promptchain/observability/config.py` - Enable by default or add checks
6. `promptchain/cli/activity_logger.py` - Integrate with decorator events

---

## Testing Strategy

### Phase 1: Enable Observability
```bash
export PROMPTCHAIN_MLFLOW_ENABLED=true
promptchain --session test --verbose
```

**Expected**: Background queue starts, MLflow logs metrics.

### Phase 2: Connect Event Bridge
- Create `TUIObserver` class that receives queue events
- Register observer with background logger
- Forward events to ObservePanel

**Expected**: ObservePanel shows real-time entries as LLM calls execute.

### Phase 3: Wire ActivityLogger
- Hook ActivityLogger into AgentChain lifecycle
- Capture all agent interactions automatically
- Verify ActivityLogViewer shows data

**Expected**: Activity Logs tab shows searchable history.

---

## Key Insight

**The observability infrastructure EXISTS and WORKS:**
- ✅ Decorators track LLM calls
- ✅ Background queue processes events
- ✅ MLflow receives metrics

**The TUI widgets are ISOLATED:**
- ❌ No connection to decorator events
- ❌ No subscription to background queue
- ❌ No integration with ActivityLogger

**Solution**: Build the missing event bridge between observability system and TUI display layer.
