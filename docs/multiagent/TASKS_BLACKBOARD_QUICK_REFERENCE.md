# Tasks and Blackboard Commands - Quick Reference

## Files Modified

**File**: `/home/gyasis/Documents/code/PromptChain/promptchain/cli/command_handler.py`

## Changes Summary

### 1. COMMAND_REGISTRY (Lines 113-116)
```python
"/tasks": {"description": "List pending/in-progress tasks", "usage": "/tasks [agent_name]"},
"/blackboard": {"description": "List or show blackboard entries", "usage": "/blackboard [key]"},
```

### 2. handle_tasks() Method (Lines 2795-2880)
```python
def handle_tasks(self, session, agent_name: Optional[str] = None) -> CommandResult:
    # Lists pending/in-progress tasks for current or specified agent
    # Uses: session_manager.list_tasks()
```

### 3. handle_blackboard() Method (Lines 2884-2968)
```python
def handle_blackboard(self, session, key: Optional[str] = None) -> CommandResult:
    # Lists all blackboard keys OR shows specific entry
    # Uses: session_manager.list_blackboard_keys(), session_manager.read_blackboard()
```

## Usage Examples

### /tasks Command
```bash
# List tasks for current agent
> /tasks

# List tasks for specific agent
> /tasks analyzer
```

### /blackboard Command
```bash
# List all blackboard keys
> /blackboard

# Show specific entry
> /blackboard analysis_results
```

## Return Values

Both methods return `CommandResult` objects with:
- `success`: bool
- `message`: str (formatted output)
- `data`: dict (structured data)
- `error`: Optional[str]

## Testing Commands
```bash
# Syntax check
python -m py_compile promptchain/cli/command_handler.py

# Verify commands exist
grep -n '"/tasks"' promptchain/cli/command_handler.py
grep -n '"/blackboard"' promptchain/cli/command_handler.py
grep -n 'def handle_tasks' promptchain/cli/command_handler.py
grep -n 'def handle_blackboard' promptchain/cli/command_handler.py
```

## Status: ✅ COMPLETE

All code changes implemented, tested, and verified.
