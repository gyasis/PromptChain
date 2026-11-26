# T051: Reasoning Step Progress Display Widget

## Objective
Create Textual widget to display real-time progress of AgenticStepProcessor multi-hop reasoning steps in the TUI.

## Context Files
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/tui/app.py` (integrate into)
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/tui/status_bar.py` (reference pattern)
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agentic_step_processor.py` (data source)

## Requirements

### Widget File Location
`/home/gyasis/Documents/code/PromptChain/promptchain/cli/tui/widgets/reasoning_progress.py`

### Widget Design

**Visual Layout**:
```
┌─ Reasoning Progress ──────────────────────────────────────┐
│                                                            │
│ Objective: Find and analyze authentication patterns       │
│                                                            │
│ Progress: [████████░░░░░░░░░░░░░] 3/8 steps               │
│                                                            │
│ Current Step:                                              │
│ ⟳ Analyzing search results from authentication.py...      │
│                                                            │
│ Completed Steps:                                           │
│ ✓ Step 1: Initial file search (2.3s)                      │
│ ✓ Step 2: Read auth/authentication.py (1.1s)              │
│ ⟳ Step 3: Analyze patterns (in progress...)               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static, ProgressBar, Label
from textual.reactive import reactive
from rich.text import Text
from typing import Optional, List
import time


class ReasoningProgressWidget(Container):
    """Display progress of AgenticStepProcessor multi-hop reasoning.

    Attributes:
        objective: Current reasoning objective
        total_steps: Maximum number of reasoning steps
        current_step: Current step number (1-indexed)
        step_history: List of completed step descriptions
        current_activity: What the processor is currently doing
        start_time: When reasoning started
    """

    DEFAULT_CSS = """
    ReasoningProgressWidget {
        height: auto;
        border: solid $primary;
        padding: 1;
        margin: 1 0;
    }

    ReasoningProgressWidget .objective {
        color: $text;
        text-style: bold;
        margin-bottom: 1;
    }

    ReasoningProgressWidget .progress-bar {
        margin: 1 0;
    }

    ReasoningProgressWidget .current-step {
        color: $accent;
        margin: 1 0;
    }

    ReasoningProgressWidget .step-history {
        color: $text-muted;
        margin-top: 1;
    }

    ReasoningProgressWidget .completed-step {
        color: $success;
    }

    ReasoningProgressWidget .active-step {
        color: $accent;
        text-style: italic;
    }
    """

    objective: reactive[Optional[str]] = reactive(None)
    total_steps: reactive[int] = reactive(0)
    current_step: reactive[int] = reactive(0)
    current_activity: reactive[Optional[str]] = reactive(None)
    is_active: reactive[bool] = reactive(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_history: List[tuple[str, float]] = []  # (description, duration)
        self.start_time: Optional[float] = None
        self.step_start_time: Optional[float] = None

    def compose(self) -> ComposeResult:
        """Create widget structure."""
        yield Static("", classes="objective", id="objective-label")
        yield ProgressBar(total=100, show_eta=False, classes="progress-bar")
        yield Static("", classes="current-step", id="current-activity")
        yield Static("", classes="step-history", id="step-history")

    def start_reasoning(self, objective: str, max_steps: int):
        """Initialize reasoning progress tracking.

        Args:
            objective: Reasoning objective description
            max_steps: Maximum number of internal steps
        """
        self.objective = objective
        self.total_steps = max_steps
        self.current_step = 0
        self.current_activity = "Initializing reasoning..."
        self.step_history = []
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.is_active = True

        # Update display
        self._update_display()

    def update_step(self, step_num: int, activity: str):
        """Update current reasoning step.

        Args:
            step_num: Current step number (1-indexed)
            activity: Description of current activity
        """
        if not self.is_active:
            return

        # Record previous step completion
        if self.current_step > 0 and self.step_start_time:
            duration = time.time() - self.step_start_time
            prev_activity = self.current_activity or f"Step {self.current_step}"
            self.step_history.append((prev_activity, duration))

        # Update to new step
        self.current_step = step_num
        self.current_activity = activity
        self.step_start_time = time.time()

        self._update_display()

    def complete_reasoning(self, final_message: str = "Reasoning complete"):
        """Mark reasoning as complete.

        Args:
            final_message: Completion message to display
        """
        if not self.is_active:
            return

        # Record final step
        if self.step_start_time:
            duration = time.time() - self.step_start_time
            self.step_history.append((self.current_activity or "Final step", duration))

        self.current_step = self.total_steps
        self.current_activity = final_message
        self.is_active = False

        self._update_display()

    def fail_reasoning(self, error_message: str):
        """Mark reasoning as failed.

        Args:
            error_message: Error description
        """
        if not self.is_active:
            return

        self.current_activity = f"❌ Failed: {error_message}"
        self.is_active = False

        self._update_display()

    def _update_display(self):
        """Update widget display elements."""
        # Update objective
        objective_label = self.query_one("#objective-label", Static)
        objective_label.update(f"Objective: {self.objective or 'N/A'}")

        # Update progress bar
        progress_bar = self.query_one(ProgressBar)
        if self.total_steps > 0:
            progress_pct = (self.current_step / self.total_steps) * 100
            progress_bar.update(progress=progress_pct)

            # Add step counter
            progress_text = f"Progress: {self.current_step}/{self.total_steps} steps"
            if self.start_time and self.is_active:
                elapsed = time.time() - self.start_time
                progress_text += f" ({elapsed:.1f}s elapsed)"
            objective_label.update(
                f"Objective: {self.objective or 'N/A'}\n{progress_text}"
            )

        # Update current activity
        activity_label = self.query_one("#current-activity", Static)
        if self.current_activity:
            activity_text = Text()
            if self.is_active:
                activity_text.append("⟳ ", style="bold cyan")
            activity_text.append(self.current_activity)
            activity_label.update(activity_text)

        # Update step history
        history_label = self.query_one("#step-history", Static)
        if self.step_history:
            history_text = Text()
            history_text.append("Completed Steps:\n", style="bold")

            for idx, (step_desc, duration) in enumerate(self.step_history, 1):
                history_text.append(f"✓ Step {idx}: ", style="bold green")
                history_text.append(f"{step_desc} ", style="")
                history_text.append(f"({duration:.1f}s)\n", style="dim")

            history_label.update(history_text)

    def watch_is_active(self, is_active: bool):
        """React to active state changes."""
        if not is_active:
            # Dim widget when inactive
            self.styles.opacity = 0.7


# Helper function for TUI integration
def create_reasoning_progress_widget() -> ReasoningProgressWidget:
    """Factory function to create reasoning progress widget."""
    return ReasoningProgressWidget()
```

### Integration with TUI

**In `promptchain/cli/tui/app.py`**:

```python
from promptchain.cli.tui.widgets.reasoning_progress import ReasoningProgressWidget

class PromptChainTUI(App):
    def __init__(self):
        super().__init__()
        self.reasoning_widget: Optional[ReasoningProgressWidget] = None

    def show_reasoning_progress(self, objective: str, max_steps: int):
        """Display reasoning progress widget."""
        # Create widget if doesn't exist
        if not self.reasoning_widget:
            self.reasoning_widget = ReasoningProgressWidget()

        # Mount widget above chat area
        self.mount(self.reasoning_widget, before=self.query_one("#chat-area"))

        # Initialize
        self.reasoning_widget.start_reasoning(objective, max_steps)

    def update_reasoning_step(self, step_num: int, activity: str):
        """Update reasoning step progress."""
        if self.reasoning_widget:
            self.reasoning_widget.update_step(step_num, activity)

    def complete_reasoning(self, message: str = "Complete"):
        """Mark reasoning as complete and remove widget."""
        if self.reasoning_widget:
            self.reasoning_widget.complete_reasoning(message)

            # Remove widget after brief delay
            self.set_timer(2.0, lambda: self._remove_reasoning_widget())

    def _remove_reasoning_widget(self):
        """Remove reasoning widget from display."""
        if self.reasoning_widget:
            self.reasoning_widget.remove()
            self.reasoning_widget = None
```

### Success Criteria
- Widget displays objective, progress, and step history
- Real-time updates during reasoning
- Clean visual integration with TUI
- Auto-removal on completion
- Smooth animations and transitions

## Testing

**Manual Test**:
```python
# In TUI app
app.show_reasoning_progress("Test reasoning", 5)
app.update_reasoning_step(1, "Searching files...")
await asyncio.sleep(1)
app.update_reasoning_step(2, "Analyzing results...")
await asyncio.sleep(1)
app.complete_reasoning("Analysis complete")
```

## Validation
- Visual inspection in TUI
- Progress updates render correctly
- Widget removes cleanly
- No visual artifacts

## Deliverable
- `reasoning_progress.py` widget implementation
- Integration hooks in `app.py`
- CSS styling for professional appearance
- Clean lifecycle management
