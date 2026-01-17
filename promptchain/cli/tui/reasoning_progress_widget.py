"""Reasoning progress widget for displaying AgenticStepProcessor execution.

Shows real-time progress during multi-hop reasoning:
- Current step count
- Tool calls being made
- Brief status text
- Clean, minimal display
"""

from typing import List, Optional
from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.widgets import Static


class ReasoningProgressWidget(Container):
    """Widget displaying AgenticStepProcessor reasoning progress.

    Shows step count, tool calls, and status while AgenticStepProcessor
    is actively executing internal reasoning loop.

    Clean, minimal design without spinners or checkboxes.
    """

    DEFAULT_CSS = """
    ReasoningProgressWidget {
        display: none;
        height: auto;
        max-height: 12;
        border: none;
        background: transparent;
        padding: 0;
        margin: 0;
    }

    ReasoningProgressWidget.visible {
        display: block;
    }

    #progress-header {
        height: 1;
        color: #888888;
    }

    #progress-steps {
        height: auto;
        max-height: 8;
        padding: 0;
        background: transparent;
        overflow-y: auto;
        scrollbar-size: 1 1;
    }

    .step-line {
        height: 1;
        color: #666666;
    }

    .step-line.current {
        color: #aaaaaa;
    }

    .step-line.tool-call {
        color: #888888;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize reasoning progress widget."""
        super().__init__(*args, **kwargs)
        self._current_step = 0
        self._max_steps = 0
        self._steps: List[str] = []  # Track step history
        self._entry_counter = 0  # Running counter for unique entry numbers
        self._max_displayed_steps = 20  # Show last N steps (scrollable, so can show more)

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("", id="progress-header")
        yield ScrollableContainer(id="progress-steps")

    def update_progress(
        self,
        current_step: int,
        max_steps: int,
        objective: str = "",
        status: str = ""
    ) -> None:
        """Update progress display.

        Args:
            current_step: Current reasoning step (1-indexed)
            max_steps: Maximum internal steps
            objective: Objective being pursued (unused - too verbose)
            status: Current status message or tool call info
        """
        self._current_step = current_step
        self._max_steps = max_steps

        # Update header with step count
        header = self.query_one("#progress-header", Static)
        header.update(f"Step {current_step}/{max_steps}")

        # Add step to history if status provided
        if status:
            # Increment entry counter for unique numbering
            self._entry_counter += 1
            # Format with running entry number (not step number)
            step_text = self._format_step(self._entry_counter, status)
            self._steps.append(step_text)

            # Rebuild step display
            self._update_step_display()

    def _format_step(self, step_num: int, status: str) -> str:
        """Format a step for display.

        Args:
            step_num: Step number
            status: Status text

        Returns:
            Formatted step string
        """
        # Truncate long status
        if len(status) > 60:
            status = status[:57] + "..."

        # Detect tool calls
        if "tool" in status.lower() or "calling" in status.lower():
            return f"  {step_num}. [tool] {status}"
        else:
            return f"  {step_num}. {status}"

    def _update_step_display(self) -> None:
        """Update the step display area."""
        steps_container = self.query_one("#progress-steps", ScrollableContainer)

        # Clear existing
        steps_container.remove_children()

        # Show last N steps
        display_steps = self._steps[-self._max_displayed_steps:]

        for i, step_text in enumerate(display_steps):
            is_current = (i == len(display_steps) - 1)
            is_tool = "[tool]" in step_text

            # Create step line
            step_static = Static(step_text.replace("[tool] ", ""))

            # Apply styling
            step_static.add_class("step-line")
            if is_current:
                step_static.add_class("current")
            if is_tool:
                step_static.add_class("tool-call")

            steps_container.mount(step_static)

        # Auto-scroll to show latest steps
        steps_container.scroll_end(animate=False)

    def add_tool_call(self, tool_name: str, step_num: Optional[int] = None) -> None:
        """Add a tool call to the display.

        Args:
            tool_name: Name of tool being called
            step_num: Step number (uses current if not provided)
        """
        step = step_num or self._current_step
        self._steps.append(f"  {step}. [tool] {tool_name}")
        self._update_step_display()

    def show_progress(self) -> None:
        """Show the progress widget."""
        self.add_class("visible")

    def hide_progress(self) -> None:
        """Hide the progress widget."""
        self.remove_class("visible")

    def reset(self) -> None:
        """Reset progress to initial state."""
        self._current_step = 0
        self._max_steps = 0
        self._steps = []
        self._entry_counter = 0  # Reset entry counter

        # Clear header
        header = self.query_one("#progress-header", Static)
        header.update("")

        # Clear steps
        steps_container = self.query_one("#progress-steps", ScrollableContainer)
        steps_container.remove_children()

        self.hide_progress()
