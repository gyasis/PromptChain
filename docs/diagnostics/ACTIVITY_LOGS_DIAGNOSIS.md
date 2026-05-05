# Activity Logs Empty (0/0) Issue - Root Cause Analysis

**Date**: 2026-01-10
**Status**: DIAGNOSED - Missing ActivityLogger Connection

---

## Executive Summary

The Activity Logs panel shows "0/0 activities" because the `ActivityLogger` is **NOT being passed to `AgentChain`** during initialization in the TUI. The infrastructure exists and works correctly, but there's a missing link in the connection chain.

---

## What is Activity Logs?

**Purpose**: Comprehensive agent interaction logging separate from chat history.

**Data Source**:
- **JSONL File**: `~/.promptchain/sessions/<session>/activity_logs/activities.jsonl`
- **SQLite Database**: `~/.promptchain/sessions/<session>/activities.db`

**What It Logs**:
- User inputs
- Agent outputs
- Tool calls and results
- Reasoning steps
- Router decisions
- Errors
- System messages

**How It Works**:
1. `ActivityLogger` writes structured JSONL entries to file
2. `ActivityLogger` inserts metadata to SQLite for fast queries
3. `ActivitySearcher` reads from both JSONL (full content) and SQLite (fast filtering)
4. `ActivityLogViewer` displays the results in TUI

---

## Current Architecture

### Data Flow (INTENDED)

```
User Input
    ↓
TUI App → AgentChain (with activity_logger parameter)
                ↓
    AgentChain.run_chat_turn_async()
                ↓
    activity_logger.start_interaction_chain()
    activity_logger.log_activity(type="user_input", ...)
                ↓
    [Agent processes input]
                ↓
    activity_logger.log_activity(type="agent_output", ...)
    activity_logger.log_activity(type="tool_call", ...)
                ↓
    JSONL File ← ActivityLogger → SQLite DB
                ↓
    ActivitySearcher reads from both
                ↓
    ActivityLogViewer displays in TUI
```

### Data Flow (ACTUAL - BROKEN)

```
User Input
    ↓
TUI App → AgentChain (❌ NO activity_logger parameter!)
                ↓
    AgentChain.run_chat_turn_async()
                ↓
    ❌ activity_logger is None - no logging happens
                ↓
    JSONL File: EMPTY
    SQLite DB: EMPTY (or only has tables, no data)
                ↓
    ActivitySearcher finds 0 activities
                ↓
    ActivityLogViewer shows "0/0 activities"
```

---

## Root Cause

### Location: `promptchain/cli/tui/app.py`

**Lines 2838-2861**: `_get_or_create_agent_chain()` method

```python
# Multi-agent router mode
self.agent_chain = AgentChain(
    agents=agents_dict,
    agent_descriptions=agent_descriptions,
    execution_mode="router",
    router=router_config,
    default_agent=orchestration.default_agent,
    auto_include_history=orchestration.auto_include_history,
    agent_history_configs=self._build_history_configs(),
    verbose=False,
    # ❌ MISSING: activity_logger=self.session.activity_logger
)

# Single-agent mode
self.agent_chain = AgentChain(
    agents=agents_dict,
    agent_descriptions=agent_descriptions,
    execution_mode="router",
    router=self._build_router_config(),
    default_agent=active_agent_name,
    auto_include_history=True,
    agent_history_configs=self._build_history_configs(),
    verbose=False,
    # ❌ MISSING: activity_logger=self.session.activity_logger
)
```

**AgentChain Signature** (`promptchain/utils/agent_chain.py`, line 199-203):

```python
# ✅ NEW: ActivityLogger integration for comprehensive agent activity logging
# Captures ALL agent interactions without affecting chat history or token usage
self.activity_logger = kwargs.get("activity_logger", None)
if self.activity_logger:
    logger.info("ActivityLogger enabled for comprehensive agent activity tracking")
```

**The Gap**:
- `AgentChain` accepts `activity_logger` via `kwargs`
- `Session` has `activity_logger` properly initialized by `SessionManager`
- TUI `app.py` creates `AgentChain` but **never passes** `activity_logger` parameter
- Result: `AgentChain.activity_logger` is always `None`

---

## Why ObservePanel Works But Activity Logs Doesn't

### ObservePanel (WORKING)
- Uses `CallbackManager` events
- Receives events via `register_callback()` during agent execution
- No dependency on `ActivityLogger`
- Source: In-memory event stream

### Activity Logs (BROKEN)
- Uses `ActivityLogger` for persistent storage
- Requires `activity_logger` parameter passed to `AgentChain`
- Source: JSONL file + SQLite database
- Dependency: `AgentChain` must have `activity_logger` to write logs

---

## Evidence

### 1. Session Has ActivityLogger
```python
# promptchain/cli/session_manager.py:478
session._activity_logger = ActivityLogger(
    session_name=session.name,
    log_dir=activity_log_dir,
    db_path=session_dir / "activities.db",
    enable_console=False
)
```

### 2. AgentChain Accepts ActivityLogger
```python
# promptchain/utils/agent_chain.py:199-203
self.activity_logger = kwargs.get("activity_logger", None)
if self.activity_logger:
    logger.info("ActivityLogger enabled for comprehensive agent activity tracking")
```

### 3. AgentChain Uses ActivityLogger
```python
# promptchain/utils/agent_chain.py:1004-1007
if self.activity_logger:
    chain_id = self.activity_logger.start_interaction_chain()
    self.activity_logger.log_activity(
        activity_type="user_input",
        agent_name=None,
        content={"input": user_input},
    )
```

### 4. TUI Never Passes ActivityLogger
```python
# promptchain/cli/tui/app.py:2838-2861
self.agent_chain = AgentChain(
    agents=agents_dict,
    # ... other parameters ...
    verbose=False,
    # ❌ MISSING: activity_logger=self.session.activity_logger
)
```

---

## Fix Recommendation

### Fix Location: `promptchain/cli/tui/app.py`

**Line 2838-2847** (Multi-agent mode):
```python
self.agent_chain = AgentChain(
    agents=agents_dict,
    agent_descriptions=agent_descriptions,
    execution_mode="router",
    router=router_config,
    default_agent=orchestration.default_agent,
    auto_include_history=orchestration.auto_include_history,
    agent_history_configs=self._build_history_configs(),
    verbose=False,
    activity_logger=self.session.activity_logger  # ✅ ADD THIS LINE
)
```

**Line 2852-2861** (Single-agent mode):
```python
self.agent_chain = AgentChain(
    agents=agents_dict,
    agent_descriptions=agent_descriptions,
    execution_mode="router",
    router=self._build_router_config(),
    default_agent=active_agent_name,
    auto_include_history=True,
    agent_history_configs=self._build_history_configs(),
    verbose=False,
    activity_logger=self.session.activity_logger  # ✅ ADD THIS LINE
)
```

---

## Expected Outcome

After fix:
1. `AgentChain` receives `activity_logger` on initialization
2. During `run_chat_turn_async()`, logging code activates:
   - `activity_logger.start_interaction_chain()` creates chain ID
   - `activity_logger.log_activity()` writes to JSONL + SQLite
3. `ActivitySearcher.grep_logs()` finds activities in JSONL
4. `ActivitySearcher.get_statistics()` returns counts > 0
5. `ActivityLogViewer` displays "Showing X/Y activities"

---

## Testing Plan

### Before Fix
```bash
# Launch TUI
promptchain --session test-activity-logs

# Send message
> Hello, test message

# Press Ctrl+L to toggle Activity Logs
# Expected: "Showing 0/0 activities"
```

### After Fix
```bash
# Launch TUI
promptchain --session test-activity-logs

# Send message
> Hello, test message

# Press Ctrl+L to toggle Activity Logs
# Expected: "Showing 2/2 activities" (user_input + agent_output)

# Click "Stats" button
# Expected: Shows statistics with activities_by_type, activities_by_agent

# Search for "test"
# Expected: Shows matching activities
```

---

## Related Files

**ActivityLogger**:
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/activity_logger.py`

**ActivitySearcher**:
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/activity_searcher.py`

**ActivityLogViewer**:
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/tui/activity_log_viewer.py`

**AgentChain**:
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agent_chain.py` (lines 199-203, 1004-1007)

**TUI App**:
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/tui/app.py` (lines 2838-2861)

---

## Summary

**What**: Activity Logs shows 0/0 activities
**Why**: `activity_logger` parameter not passed to `AgentChain` during initialization
**Where**: `promptchain/cli/tui/app.py`, lines 2838-2861
**Fix**: Add `activity_logger=self.session.activity_logger` to both `AgentChain()` constructor calls
**Impact**: TWO-LINE FIX to enable full activity logging infrastructure
