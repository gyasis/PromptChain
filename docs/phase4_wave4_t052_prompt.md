# T052: Step-by-Step Output Streaming for AgenticStepProcessor

## Objective
Implement real-time streaming of AgenticStepProcessor internal reasoning steps to TUI, updating progress widget as steps execute.

## Context Files
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/tui/app.py` (integrate streaming)
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/tui/widgets/reasoning_progress.py` (T051 widget)
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agentic_step_processor.py` (add callbacks)
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/promptchaining.py` (T049 integration)

## Requirements

### Part 1: Add Progress Callbacks to AgenticStepProcessor

**Modify `promptchain/utils/agentic_step_processor.py`**:

```python
from typing import Callable, Optional


class AgenticStepProcessor:
    def __init__(
        self,
        # ... existing params
        progress_callback: Optional[Callable[[int, str], None]] = None
    ):
        """
        Args:
            progress_callback: Callback for step progress updates
                Signature: callback(step_num: int, activity: str)
        """
        self.progress_callback = progress_callback

    def set_progress_callback(self, callback: Callable[[int, str], None]):
        """Set callback for progress updates after initialization."""
        self.progress_callback = callback

    async def process_async(self, user_input: str) -> str:
        """Process with progress callbacks."""
        # Notify start
        if self.progress_callback:
            self.progress_callback(0, f"Starting: {self.objective}")

        for step_num in range(1, self.max_internal_steps + 1):
            # Notify step start
            if self.progress_callback:
                self.progress_callback(step_num, f"Step {step_num}: Reasoning...")

            # Execute LLM call
            response = await self._execute_reasoning_step(user_input, step_num)

            # Notify step completion
            if self.progress_callback:
                activity = self._extract_step_summary(response)
                self.progress_callback(step_num, f"Completed: {activity}")

            # Check completion
            if self._is_objective_complete(response):
                # Notify completion
                if self.progress_callback:
                    self.progress_callback(
                        self.max_internal_steps,
                        "Objective achieved"
                    )
                return self._extract_final_synthesis(response)

        # Max steps reached
        if self.progress_callback:
            self.progress_callback(
                self.max_internal_steps,
                "Max steps reached"
            )

        # Handle exhaustion...

    def _extract_step_summary(self, llm_response: str) -> str:
        """Extract brief summary of what this step accomplished.

        Args:
            llm_response: Raw LLM response from reasoning step

        Returns:
            One-line summary of step activity
        """
        # Simple heuristic: first sentence or tool call description
        lines = llm_response.strip().split('\n')

        # Look for tool calls
        if 'tool_call' in llm_response.lower():
            # Extract tool name
            import re
            match = re.search(r"'name':\s*'([^']+)'", llm_response)
            if match:
                return f"Called tool: {match.group(1)}"

        # First non-empty line
        for line in lines:
            if line.strip() and len(line) < 100:
                return line.strip()

        # Fallback: truncated first line
        return (lines[0][:80] + "...") if lines else "Processing..."
```

### Part 2: Connect PromptChain to Progress Widget

**Modify `promptchain/utils/promptchaining.py`** (T049 method):

```python
async def _execute_agentic_step(self, processor, step_index):
    """Execute AgenticStepProcessor with TUI progress streaming."""
    # Set up progress callback if TUI callback exists
    if hasattr(self, 'agentic_progress_callback'):
        def progress_callback(step_num: int, activity: str):
            """Forward progress to TUI callback."""
            self.agentic_progress_callback(
                processor.objective,
                processor.max_internal_steps,
                step_num,
                activity
            )

        processor.set_progress_callback(progress_callback)

    # Execute processor
    result = await processor.process_async(self.current_input)

    return result
```

### Part 3: TUI Integration

**Extend `promptchain/cli/tui/app.py`**:

```python
from promptchain.cli.tui.widgets.reasoning_progress import ReasoningProgressWidget


class PromptChainTUI(App):
    def __init__(self):
        super().__init__()
        self.reasoning_widget: Optional[ReasoningProgressWidget] = None

    def _setup_agent_chain_callbacks(self):
        """Set up callbacks for agent chain progress updates."""
        # Set agentic progress callback on each agent's chain
        for agent_name, agent in self.agent_chain.agents.items():
            agent.agentic_progress_callback = self._handle_agentic_progress

    def _handle_agentic_progress(
        self,
        objective: str,
        max_steps: int,
        current_step: int,
        activity: str
    ):
        """Handle progress updates from AgenticStepProcessor.

        Args:
            objective: Reasoning objective
            max_steps: Maximum reasoning steps
            current_step: Current step number (0 = start, >0 = step)
            activity: Description of current activity
        """
        # Initialize widget on first progress update
        if current_step == 0:
            if not self.reasoning_widget:
                self.reasoning_widget = ReasoningProgressWidget()
                self.mount(self.reasoning_widget, before=self.query_one("#chat-area"))

            self.reasoning_widget.start_reasoning(objective, max_steps)

        # Update step progress
        elif current_step > 0:
            if self.reasoning_widget:
                self.reasoning_widget.update_step(current_step, activity)

                # Check if completed
                if current_step >= max_steps or "achieved" in activity.lower():
                    # Mark complete and schedule removal
                    self.set_timer(
                        2.0,
                        lambda: self._complete_reasoning(activity)
                    )

    def _complete_reasoning(self, final_message: str):
        """Complete reasoning and remove widget."""
        if self.reasoning_widget:
            self.reasoning_widget.complete_reasoning(final_message)
            self.set_timer(1.5, self._remove_reasoning_widget)

    def _remove_reasoning_widget(self):
        """Remove reasoning widget from display."""
        if self.reasoning_widget:
            self.reasoning_widget.remove()
            self.reasoning_widget = None

    async def _handle_agent_execution(self, user_input: str):
        """Execute agent with progress streaming."""
        try:
            # Set up callbacks before execution
            self._setup_agent_chain_callbacks()

            # Execute agent chain
            result = await self.agent_chain.process_input_async(user_input)

            # Display result (reasoning widget auto-updates)
            self._display_agent_response(result)

        except Exception as e:
            # Error handling...
            if self.reasoning_widget:
                self.reasoning_widget.fail_reasoning(str(e))
```

### Part 4: Thread-Safe Updates

**Handle Textual's Thread Safety Requirements**:

```python
from textual import work


class PromptChainTUI(App):
    def _handle_agentic_progress(
        self,
        objective: str,
        max_steps: int,
        current_step: int,
        activity: str
    ):
        """Handle progress updates (may come from async background thread)."""
        # Use Textual's work decorator for thread-safe UI updates
        self._update_progress_ui(objective, max_steps, current_step, activity)

    @work(exclusive=False, thread=False)
    async def _update_progress_ui(
        self,
        objective: str,
        max_steps: int,
        current_step: int,
        activity: str
    ):
        """Thread-safe UI update method."""
        # Initialize widget on first update
        if current_step == 0:
            if not self.reasoning_widget:
                self.reasoning_widget = ReasoningProgressWidget()
                await self.mount(self.reasoning_widget, before=self.query_one("#chat-area"))

            self.reasoning_widget.start_reasoning(objective, max_steps)

        # Update step
        elif current_step > 0:
            if self.reasoning_widget:
                self.reasoning_widget.update_step(current_step, activity)
```

### Success Criteria
- Real-time progress updates stream to TUI
- Reasoning widget updates as steps execute
- No blocking or UI freezing
- Clean widget lifecycle (create, update, remove)
- Thread-safe UI updates
- Works with multiple concurrent agentic steps

## Testing

**Manual Test Flow**:
1. Start TUI: `promptchain`
2. Create agent with agentic_step in YAML config
3. Send query requiring multi-hop reasoning
4. Observe:
   - Progress widget appears
   - Steps update in real-time
   - Activity descriptions change
   - Widget removes after completion

**Integration Test** (T046 extended):
```python
@pytest.mark.asyncio
async def test_progress_streaming():
    """Progress callbacks stream to widget."""
    progress_updates = []

    def mock_callback(obj, max_s, step, act):
        progress_updates.append((step, act))

    processor = AgenticStepProcessor(
        objective="Test streaming",
        max_internal_steps=3,
        model_name="openai/gpt-4o-mini",
        progress_callback=lambda step, act: mock_callback("obj", 3, step, act)
    )

    await processor.process_async("Test input")

    # Verify updates received
    assert len(progress_updates) >= 2  # At least start + steps
    assert progress_updates[0][0] == 0  # First update is start
```

## Validation
- Manual TUI test with agentic workflows
- Progress widget updates smoothly
- No UI blocking or lag
- Clean error recovery

## Deliverable
- Progress callback integration in `agentic_step_processor.py`
- Callback forwarding in `promptchaining.py`
- TUI streaming in `app.py`
- Thread-safe UI update handling
- Working real-time progress display
