# Hierarchical Step Numbering Implementation

## Problem Statement

When multiple `AgenticStepProcessor` instances are called in a single PromptChain execution, all internal steps showed "[Step 1]" in the Observability Panel, making it impossible to distinguish which processor the steps belonged to.

## Solution Overview

Implemented hierarchical step numbering in format: `{processor_call_number}.{internal_step_number}`

**Examples:**
- First AgenticStepProcessor (15 steps): "Step 1.1", "Step 1.2", ..., "Step 1.15"
- Second AgenticStepProcessor (10 steps): "Step 2.1", "Step 2.2", ..., "Step 2.10"
- Third processor: "Step 3.1", "Step 3.2", etc.

## Implementation Details

### Files Modified

1. **`promptchain/cli/tui/app.py`**:
   - Added processor call tracking: `processor_call_count`, `last_step_number`, `processor_completed`
   - Modified `_reasoning_progress_callback()` to detect new processor instances
   - Reset counters at the start of each user message
   - Format hierarchical step numbers for ObservePanel logging

2. **`promptchain/cli/tui/observe_panel.py`**:
   - Updated `log_reasoning()` signature to accept `Union[int, str]` for step parameter
   - Supports both integer (legacy) and hierarchical string formats

### Detection Logic

A new AgenticStepProcessor instance is detected when:

1. **Step goes backward**: `current_step < last_step_number` (e.g., 3 → 1)
2. **First call ever**: `current_step == 1 and last_step_number == 0`
3. **After completion**: `current_step == 1 and processor_completed == True`

When detected:
- Increment `processor_call_count`
- Reset `processor_completed` flag
- Show reasoning progress widget

### Completion Tracking

The `processor_completed` flag is set when:
- Callback receives `status == "Complete"`

This allows the next `current_step == 1` to be recognized as a new processor rather than a continuation.

### Counter Reset

All counters are reset at the start of each user message in `handle_user_message()`:
```python
self.processor_call_count = 0
self.last_step_number = 0
self.processor_completed = False
```

This ensures each conversation turn starts fresh with processor 1.

## Code Changes

### app.py `__init__()`

```python
# Track AgenticStepProcessor call number for hierarchical step display (1.1, 2.1, etc.)
self.processor_call_count = 0
self.last_step_number = 0  # Track last seen step to detect new processor instances
self.processor_completed = False  # Track if last processor completed
```

### app.py `_reasoning_progress_callback()`

```python
def _reasoning_progress_callback(self, current_step: int, max_steps: int, status: str) -> None:
    """Progress callback for AgenticStepProcessor reasoning updates (T052, T054)."""
    if not self.reasoning_progress:
        return

    # Detect new AgenticStepProcessor instance
    if (current_step < self.last_step_number or
        (current_step == 1 and self.last_step_number == 0) or
        (current_step == 1 and self.processor_completed)):
        self.processor_call_count += 1
        self.processor_completed = False
        self.reasoning_progress.show_progress()

    self.last_step_number = current_step

    # Track completion for next processor detection
    if status == "Complete":
        self.processor_completed = True

    # Format hierarchical step number: {processor_call}.{internal_step}
    hierarchical_step = f"{self.processor_call_count}.{current_step}"

    # Update ObservePanel with hierarchical numbering
    if self.verbose_mode and self.observe_panel:
        # ... log entries with hierarchical_step ...
```

### observe_panel.py `log_reasoning()`

```python
def log_reasoning(self, step: Union[int, str], thought: str) -> None:
    """Log a reasoning step.

    Args:
        step: Step number or hierarchical step (e.g., "1.5" for processor 1, step 5)
        thought: Reasoning content to log
    """
    thought_preview = thought[:150] + "..." if len(thought) > 150 else thought
    self.log_entry(
        "reasoning",
        f"THINK [Step {step}]: {thought_preview}"
    )
```

## Testing

### Test Files

1. **`test_hierarchical_steps.py`**: Basic test with 2 processors
2. **`test_hierarchical_multi_step.py`**: Advanced test verifying multi-step execution

### Test Results

```bash
$ python test_hierarchical_steps.py
[Step 1.1] Reasoning...
[Step 1.1] Calling: simple_tool
[Step 1.1] Synthesizing results...
[Step 1.1] Complete
[Step 2.1] Reasoning...
[Step 2.1] Calling: simple_tool
[Step 2.1] Synthesizing results...
[Step 2.1] Complete

Total processor calls: 2
✅ SUCCESS: Hierarchical step numbering detected!
```

## Behavior Notes

### Single Internal Step Processors

Most AgenticStepProcessor instances complete in their first internal iteration when the objective is simple and tools return sufficient information. This results in all steps showing as `X.1` for each processor:

- Processor 1: 1.1, 1.1, 1.1, 1.1 (Reasoning, Calling, Synthesizing, Complete)
- Processor 2: 2.1, 2.1, 2.1, 2.1 (Reasoning, Calling, Synthesizing, Complete)

This is CORRECT behavior - each processor's first internal step (`step_num=0` → `current_step=1`) has multiple callback invocations for different phases.

### Multi-Step Processors

When a processor requires multiple internal iterations (complex objectives needing multiple tool calls), you'll see:

- Processor 1: 1.1, 1.2, 1.3, 1.4, ... (multiple internal steps)
- Processor 2: 2.1, 2.2, 2.3, ... (multiple internal steps)

### Callback Frequency

The `progress_callback` is NOT called once per internal iteration. It's called at key phases within each iteration:
- At start: "Reasoning..."
- During tool execution: "Calling: tool_name"
- After tools: "Synthesizing results..."
- On completion: "Complete"

## Impact

### User Benefits

1. **Clear Processor Identification**: Users can now see which processor is executing (1.x, 2.x, 3.x)
2. **Execution Progress**: Within each processor, see step progression (x.1, x.2, x.3)
3. **Debugging Aid**: Easier to trace issues to specific processor calls

### Observability Panel

The hierarchical format appears in all ObservePanel entries:
```
[Step 1.1] THINK: Analyzing user request...
[Step 1.2] TOOL-CALL: search_files("pattern")
[Step 1.3] TOOL-RESULT: Found 5 matches...
[Step 2.1] THINK: Processing previous results...
```

## Backwards Compatibility

- The change is fully backwards compatible
- Single processor chains still work (show 1.1, 1.2, 1.3...)
- ObservePanel's `log_reasoning()` accepts both `int` and `str`
- No changes required to AgenticStepProcessor itself

## Future Enhancements

Potential improvements:
1. Display processor objective in step label: `[Processor 1 "Analysis" - Step 1.1]`
2. Color-code different processors in ObservePanel
3. Add processor execution summary (total steps, duration) when switching processors
4. Persist processor metadata in session history for post-execution analysis

## Related Files

- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/tui/app.py`
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/tui/observe_panel.py`
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agentic_step_processor.py`
- `/home/gyasis/Documents/code/PromptChain/test_hierarchical_steps.py`
- `/home/gyasis/Documents/code/PromptChain/test_hierarchical_multi_step.py`
