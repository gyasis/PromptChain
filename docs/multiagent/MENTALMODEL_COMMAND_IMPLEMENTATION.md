# Mental Model CLI Command Implementation

## Summary

Successfully implemented the `/mentalmodel` CLI command for PromptChain, enabling users to view an agent's mental model including specializations, capabilities, known agents, and recent task history.

## Changes Made

### 1. Command Handler (`promptchain/cli/command_handler.py`)

**Added:**
- New command entry in `COMMAND_REGISTRY` (line 117-118):
  ```python
  "/mentalmodel": {"description": "Show agent's mental model", "usage": "/mentalmodel"}
  ```

- New handler method `handle_mentalmodel()` (lines 2974-3068):
  - Retrieves current agent's mental model from session
  - Displays specializations with proficiency levels
  - Lists known agents with their capabilities
    - Shows recent task history (last 5 tasks)
  - Creates default mental model if none exists

**Key Implementation Details:**
- Uses `session.active_agent` to get current agent name
- Accesses `session_manager._mental_model_manager` for mental model data
- Returns structured `CommandResult` with both formatted message and data dict
- Handles edge cases (no specializations, no known agents, no task history)

### 2. TUI Integration (`promptchain/cli/tui/app.py`)

**Added:**
- Command routing in `handle_command()` method (lines 1707-1716):
  ```python
  elif command.startswith("/mentalmodel"):
      if self.session:
          result = self.command_handler.handle_mentalmodel(self.session)
          msg = Message(role="system", content=result.message)
          chat_view.add_message(msg)
      else:
          chat_view.add_message(Message(role="system", content="No active session"))
  ```

- Updated help text in `_get_help_text()` (line 342):
  ```python
  "  [bold]/mentalmodel[/bold] - Show agent's mental model and capabilities\n"
  ```

### 3. Tool Registry Enhancement (`promptchain/cli/tools/registry.py`)

**Added:**
- New `COLLABORATION` category to `ToolCategory` enum (line 36):
  ```python
  COLLABORATION = "collaboration"
  ```

This was required because `delegation_tools.py` was already using `ToolCategory.COLLABORATION` but the enum value didn't exist.

## Output Format

Example output from `/mentalmodel`:

```
Mental Model for agent: default

Specializations:
  - code_analysis (proficiency: 0.50)
    Capabilities: code_search, ripgrep_search, file_read
  - testing (proficiency: 0.50)
    Capabilities: terminal_execute, test_runner

Known Agents:
  - reviewer: code_review, testing
  - coder: code_generation, implementation

Recent Tasks (last 5):
  ✓ code_analysis - success
  ✗ debugging - failed
  ✓ code_review - success
```

## Data Structure

The command returns structured data:

```python
{
    "agent_name": "default",
    "specializations": [
        {
            "type": "code_analysis",
            "proficiency": 0.5,
            "capabilities": ["code_search", "ripgrep_search", "file_read"],
            "experience": 0
        }
    ],
    "known_agents": {
        "reviewer": ["code_review", "testing"],
        "coder": ["code_generation", "implementation"]
    },
    "recent_tasks": [
        {
            "task_type": "code_analysis",
            "success": True,
            "capabilities_used": ["code_analysis", "code_review"],
            "timestamp": 1764364214.757064
        }
    ]
}
```

## Testing

Created and executed standalone test (`test_mentalmodel_command.py`) that verified:
- Command handler successfully retrieves mental model data
- Output formatting matches specification
- Specializations display correctly with proficiency
- Known agents list properly
- Task history shows with success/failure indicators
- Default mental model creation works when none exists

All tests passed successfully.

## Files Modified

1. `/home/gyasis/Documents/code/PromptChain/promptchain/cli/command_handler.py`
   - Added command registry entry
   - Added `handle_mentalmodel()` method

2. `/home/gyasis/Documents/code/PromptChain/promptchain/cli/tui/app.py`
   - Added command routing in `handle_command()`
   - Updated help text

3. `/home/gyasis/Documents/code/PromptChain/promptchain/cli/tools/registry.py`
   - Added `COLLABORATION` to `ToolCategory` enum

## Dependencies

The implementation relies on existing mental models infrastructure:
- `promptchain.cli.models.mental_models.MentalModel`
- `promptchain.cli.models.mental_models.MentalModelManager`
- `promptchain.cli.models.mental_models.create_default_model`
- `promptchain.cli.models.mental_models.SpecializationType`

## Usage

Users can now run `/mentalmodel` in the CLI to view:
1. Current agent's specializations and proficiency levels
2. Associated capabilities for each specialization
3. Other agents the current agent knows about
4. Recent task history (last 5 tasks) with success/failure status

This enables agents to have self-awareness and helps users understand agent capabilities for better task delegation.

## Integration Points

The command integrates with:
- **Session Management**: Accesses current session and active agent
- **Mental Models System**: Uses MentalModelManager for data retrieval
- **Command Handler**: Follows standard CommandResult pattern
- **TUI**: Displays as system message in chat view
- **Help System**: Documented in `/help commands`

## Future Enhancements

Potential improvements:
1. Add filtering options (e.g., `/mentalmodel --specialization code_analysis`)
2. Support for other agents (e.g., `/mentalmodel <agent_name>`)
3. Visualization of proficiency trends over time
4. Export mental model to JSON file
5. Comparison between multiple agents' mental models
