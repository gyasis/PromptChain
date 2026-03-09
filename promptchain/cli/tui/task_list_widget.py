"""Task list widget for displaying agent task progress.

Shows real-time task tracking during complex multi-step operations:
- Current task status (pending, in_progress, completed)
- Progress bar
- Task count
- EXPANDABLE tasks to see internal operations (LLM calls, tool calls, steps)
- Subtask support
- Clean, grayscale design
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.message import Message as TextualMessage
from textual.widgets import Static


@dataclass
class TaskInternalStep:
    """Internal step within a task (LLM call, tool call, etc.)."""

    step_type: str  # "thinking", "tool_call", "tool_result", "llm_call", "error"
    content: str
    timestamp: str = ""


@dataclass
class ExpandableTask:
    """Task with internal steps and subtasks."""

    task_id: str
    content: str
    active_form: str
    status: str  # "pending", "in_progress", "completed", "skipped"
    progress: float = 0.0
    internal_steps: List[TaskInternalStep] = field(default_factory=list)
    subtasks: List["ExpandableTask"] = field(default_factory=list)
    expanded: bool = False
    max_visible_steps: int = 5  # Show last N steps when expanded


class TaskListWidget(Container):
    """Widget displaying agent task list progress with expandable details.

    Shows task list with status indicators while agent works
    through multi-step tasks.

    Features:
    - Expandable tasks to see internal operations
    - Internal step tracking (thinking, tool calls, LLM calls)
    - Subtask hierarchy support
    - Real-time updates during processing
    - Grayscale design matching the TUI theme.
    """

    DEFAULT_CSS = """
    TaskListWidget {
        display: none;
        height: auto;
        max-height: 20;
        border: none;
        background: transparent;
        padding: 0;
        margin: 0;
    }

    TaskListWidget.visible {
        display: block;
    }

    #task-header {
        height: 1;
        color: #888888;
    }

    #task-list-container {
        height: auto;
        max-height: 15;
        padding: 0;
        background: transparent;
        overflow-y: auto;
        scrollbar-size: 1 1;
    }

    .task-line {
        height: 1;
        color: #666666;
    }

    .task-line.pending {
        color: #666666;
    }

    .task-line.in-progress {
        color: #aaaaaa;
    }

    .task-line.completed {
        color: #444444;
    }

    .task-line.expandable {
        color: #aaaaaa;
    }

    .internal-step {
        height: 1;
        color: #555555;
        padding-left: 4;
    }

    .internal-step.thinking {
        color: #777777;
    }

    .internal-step.tool-call {
        color: #5588aa;
    }

    .internal-step.tool-result {
        color: #558855;
    }

    .internal-step.error {
        color: #aa5555;
    }

    .subtask-line {
        height: 1;
        color: #555555;
        padding-left: 4;
    }

    #task-progress-bar {
        height: 1;
        color: #666666;
        margin-top: 0;
    }
    """

    class TaskToggled(TextualMessage):
        """Event when task expand/collapse is toggled."""

        def __init__(self, task_id: str, expanded: bool):
            super().__init__()
            self.task_id = task_id
            self.expanded = expanded

    def __init__(self, *args, **kwargs) -> None:
        """Initialize task list widget."""
        super().__init__(*args, **kwargs)
        self._task_manager = None
        # Track expandable tasks with internal steps
        self._expandable_tasks: Dict[str, ExpandableTask] = {}
        # Track which task is currently in_progress for step tracking
        self._current_task_id: Optional[str] = None
        # Refresh timer to update task counts in real-time
        self._refresh_timer: Optional[Any] = None

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("", id="task-header")
        yield ScrollableContainer(id="task-list-container")
        yield Static("", id="task-progress-bar")

    def on_mount(self) -> None:
        """Start periodic refresh timer when widget is mounted."""
        # Refresh every 0.5 seconds to update task counts in real-time
        self._refresh_timer = self.set_interval(0.5, self._periodic_refresh)

    def _periodic_refresh(self) -> None:
        """Periodically refresh the display to catch task count changes."""
        # Only refresh if widget is visible and has a task manager
        if self.has_class("visible") and self._task_manager:
            self.refresh_display()

    def set_task_manager(self, manager) -> None:
        """Set the task list manager to display.

        Args:
            manager: TaskListManager instance
        """
        self._task_manager = manager
        self._sync_expandable_tasks()
        self.refresh_display()

    def _sync_expandable_tasks(self) -> None:
        """Sync expandable task state with task manager."""
        if not self._task_manager or not self._task_manager.current_list:
            return

        task_list = self._task_manager.current_list

        for i, task in enumerate(task_list.tasks):
            task_id = f"task_{i}"

            if task_id not in self._expandable_tasks:
                # Create new expandable task
                self._expandable_tasks[task_id] = ExpandableTask(
                    task_id=task_id,
                    content=task.content,
                    active_form=task.active_form,
                    status=task.status.value,
                )
            else:
                # Update existing task
                et = self._expandable_tasks[task_id]
                et.content = task.content
                et.active_form = task.active_form
                et.status = task.status.value

            # Track current in_progress task
            if task.status.value == "in_progress":
                self._current_task_id = task_id

    def add_internal_step(
        self, step_type: str, content: str, task_id: Optional[str] = None
    ) -> None:
        """Add an internal step to a task.

        Args:
            step_type: Type of step ("thinking", "tool_call", "tool_result", "llm_call", "error")
            content: Step content/description
            task_id: Task ID to add step to (defaults to current in_progress task)
        """
        target_id = task_id or self._current_task_id

        # If no target task exists, create a default "agent_processing" task
        # This allows tracking internal steps even without a formal TaskListManager
        if not target_id or target_id not in self._expandable_tasks:
            target_id = "agent_processing"
            if target_id not in self._expandable_tasks:
                self._expandable_tasks[target_id] = ExpandableTask(
                    task_id=target_id,
                    content="Processing request",
                    active_form="Processing...",
                    status="in_progress",
                    expanded=True,  # Auto-expand to show steps
                )
            self._current_task_id = target_id

        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Truncate content for display
        display_content = content[:80] + "..." if len(content) > 80 else content

        step = TaskInternalStep(
            step_type=step_type, content=display_content, timestamp=timestamp
        )

        self._expandable_tasks[target_id].internal_steps.append(step)

        # Auto-expand task with new internal step if in_progress
        if self._expandable_tasks[target_id].status == "in_progress":
            self._expandable_tasks[target_id].expanded = True

        self.refresh_display()

    def toggle_task(self, task_id: str) -> None:
        """Toggle task expanded state.

        Args:
            task_id: ID of task to toggle
        """
        if task_id in self._expandable_tasks:
            self._expandable_tasks[task_id].expanded = not self._expandable_tasks[
                task_id
            ].expanded
            self.post_message(
                self.TaskToggled(task_id, self._expandable_tasks[task_id].expanded)
            )
            self.refresh_display()

    def expand_current_task(self) -> None:
        """Expand the current in_progress task."""
        if self._current_task_id and self._current_task_id in self._expandable_tasks:
            self._expandable_tasks[self._current_task_id].expanded = True
            self.refresh_display()

    def collapse_all_tasks(self) -> None:
        """Collapse all tasks."""
        for task in self._expandable_tasks.values():
            task.expanded = False
        self.refresh_display()

    def refresh_display(self) -> None:
        """Refresh the task list display from the manager or expandable tasks."""
        # Check if we have a task manager OR expandable tasks (from streaming)
        has_task_manager = self._task_manager and self._task_manager.current_list
        has_expandable_tasks = bool(self._expandable_tasks)

        if not has_task_manager and not has_expandable_tasks:
            self.hide_task_list()
            return

        # Update task lines
        task_container = self.query_one("#task-list-container", ScrollableContainer)
        task_container.remove_children()

        if has_task_manager:
            # Use task manager tasks
            self._sync_expandable_tasks()
            task_list = self._task_manager.current_list

            # Update header with expand hint
            header = self.query_one("#task-header", Static)
            completed = task_list.completed_count
            total = task_list.total_count
            header.update(
                f"[bold]Tasks ({completed}/{total})[/bold] [dim](click to expand)[/dim]"
            )

            for i, task in enumerate(task_list.tasks):
                task_id = f"task_{i}"
                expandable = self._expandable_tasks.get(task_id)
                self._render_task(
                    task_container,
                    task_id,
                    expandable,
                    task.status.value,
                    task.content,
                    task.active_form,
                )
        else:
            # Display from expandable tasks (streaming mode without task manager)
            header = self.query_one("#task-header", Static)
            in_progress_count = sum(
                1 for t in self._expandable_tasks.values() if t.status == "in_progress"
            )
            completed_count = sum(
                1 for t in self._expandable_tasks.values() if t.status == "completed"
            )
            total = len(self._expandable_tasks)
            header.update(f"[bold]Agent Activity ({completed_count}/{total})[/bold]")

            for expandable in self._expandable_tasks.values():
                self._render_task(
                    task_container,
                    expandable.task_id,
                    expandable,
                    expandable.status,
                    expandable.content,
                    expandable.active_form,
                )

        # Update progress bar
        self._update_progress_bar()

        # Auto-scroll to show current in_progress task
        task_container.scroll_end(animate=False)
        self.show_task_list()

    def _render_task(
        self,
        container: ScrollableContainer,
        task_id: str,
        expandable: Optional[ExpandableTask],
        status: str,
        content: str,
        active_form: str,
    ) -> None:
        """Render a single task with its internal steps.

        Args:
            container: Container to mount widgets to
            task_id: Task ID
            expandable: ExpandableTask with internal steps (may be None)
            status: Task status
            content: Task content
            active_form: Active form text
        """
        has_steps = expandable and len(expandable.internal_steps) > 0
        is_expanded = expandable and expandable.expanded

        if status == "completed":
            prefix = "[bold]+[/bold]"
            text = f"[dim]{content}[/dim]"
            css_class = "completed"
        elif status == "in_progress":
            # Show expand indicator for in_progress tasks
            expand_icon = "v" if is_expanded else ">"
            prefix = f"[bold]{expand_icon}[/bold]"
            text = f"[bold]{active_form}[/bold]"
            if has_steps and expandable is not None:
                text += f" [dim]({len(expandable.internal_steps)} steps)[/dim]"
            css_class = "in-progress"
        elif status == "skipped":
            prefix = "[dim]-[/dim]"
            text = f"[dim]{content} (skipped)[/dim]"
            css_class = "completed"
        else:  # pending
            prefix = "[dim]o[/dim]"
            text = content
            css_class = "pending"

        task_static = Static(f"  {prefix} {text}")
        task_static.add_class("task-line")
        task_static.add_class(css_class)
        if has_steps:
            task_static.add_class("expandable")
        container.mount(task_static)

        # Show internal steps if expanded
        if is_expanded and has_steps and expandable is not None:
            # Show last N steps
            visible_steps = expandable.internal_steps[-expandable.max_visible_steps :]

            for step in visible_steps:
                step_prefix = self._get_step_prefix(step.step_type)
                # Escape content to prevent Rich markup conflicts
                safe_content = escape_markup(step.content)
                step_text = f"    {step_prefix} {safe_content}"
                if step.timestamp:
                    step_text += f" [dim]{step.timestamp}[/dim]"

                step_static = Static(step_text)
                step_static.add_class("internal-step")
                step_static.add_class(step.step_type.replace("_", "-"))
                container.mount(step_static)

            # Show "more steps" indicator if truncated
            if len(expandable.internal_steps) > expandable.max_visible_steps:
                hidden_count = (
                    len(expandable.internal_steps) - expandable.max_visible_steps
                )
                more_static = Static(f"    [dim]... {hidden_count} more steps[/dim]")
                more_static.add_class("internal-step")
                container.mount(more_static)

        # Show subtasks if any
        if expandable and expandable.subtasks:
            for subtask in expandable.subtasks:
                sub_prefix = self._get_status_prefix(subtask.status)
                subtask_static = Static(f"    {sub_prefix} {subtask.content}")
                subtask_static.add_class("subtask-line")
                container.mount(subtask_static)

    def _update_progress_bar(self) -> None:
        """Update the progress bar based on internal step activity.

        For streaming mode (without task manager), shows step count rather than
        percentage since we don't know total steps in advance.
        """
        progress_bar = self.query_one("#task-progress-bar", Static)

        if self._task_manager and self._task_manager.current_list:
            # Use task manager progress
            task_list = self._task_manager.current_list
            progress_pct = task_list.progress_percentage
            filled = int(progress_pct / 10)
            bar = "[" + "#" * filled + "." * (10 - filled) + "]"
            progress_bar.update(f"  {bar} {progress_pct:.0f}%")
        elif self._expandable_tasks:
            # For streaming mode: show step count instead of misleading percentage
            current_task = (
                self._expandable_tasks.get(self._current_task_id)
                if self._current_task_id
                else None
            )
            if current_task and current_task.internal_steps:
                step_count = len(current_task.internal_steps)
                # Animated progress indicator based on step count
                filled = min(step_count, 10)  # Cap at 10 for visual
                bar = "[" + "#" * filled + "." * (10 - filled) + "]"
                progress_bar.update(f"  {bar} {step_count} steps")
            else:
                progress_bar.update("  [..........] waiting...")
        else:
            progress_bar.update("")

    def _get_step_prefix(self, step_type: str) -> str:
        """Get display prefix for step type."""
        prefixes = {
            "thinking": "[dim]🧠[/dim]",
            "tool_call": "[cyan]🔧[/cyan]",
            "tool_result": "[green]✓[/green]",
            "llm_call": "[blue]💬[/blue]",
            "error": "[red]⚠[/red]",
        }
        return prefixes.get(step_type, "[dim]•[/dim]")

    def _get_status_prefix(self, status: str) -> str:
        """Get display prefix for status."""
        prefixes = {
            "completed": "[bold]+[/bold]",
            "in_progress": "[bold]>[/bold]",
            "pending": "[dim]o[/dim]",
            "skipped": "[dim]-[/dim]",
        }
        return prefixes.get(status, "[dim]•[/dim]")

    def show_task_list(self) -> None:
        """Show the task list widget."""
        self.add_class("visible")

    def hide_task_list(self) -> None:
        """Hide the task list widget."""
        self.remove_class("visible")

    def mark_processing_complete(self) -> None:
        """Mark the default processing task as completed.

        Called when agent finishes processing to update the task status.
        """
        if "agent_processing" in self._expandable_tasks:
            self._expandable_tasks["agent_processing"].status = "completed"
            self._expandable_tasks["agent_processing"].content = "Request completed"
            self._expandable_tasks["agent_processing"].active_form = "Completed"
            self._current_task_id = None
            self.refresh_display()

    def clear(self) -> None:
        """Clear the task list display."""
        # Clear header
        header = self.query_one("#task-header", Static)
        header.update("")

        # Clear tasks
        task_container = self.query_one("#task-list-container", ScrollableContainer)
        task_container.remove_children()

        # Clear progress bar
        progress_bar = self.query_one("#task-progress-bar", Static)
        progress_bar.update("")

        # Clear internal state
        self._expandable_tasks.clear()
        self._current_task_id = None

        self.hide_task_list()
