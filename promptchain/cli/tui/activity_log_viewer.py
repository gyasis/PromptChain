"""ActivityLogViewer widget for displaying and searching agent activity logs in TUI.

Phase 5: TUI Integration for Activity Logging
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from ..activity_searcher import ActivitySearcher


class ActivityLogItem(ListItem):
    """A single activity item in the log viewer.

    Features:
    - Timestamp display
    - Agent name highlight
    - Activity type indicator
    - Content preview (first 100 chars)
    - Click to expand full content
    """

    def __init__(self, activity: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activity = activity
        self.expanded = reactive(False)

    def compose(self) -> ComposeResult:
        """Compose the activity item."""
        # Extract activity data
        timestamp = self.activity.get('timestamp', 'unknown')
        agent_name = self.activity.get('agent_name', 'system')
        activity_type = self.activity.get('activity_type', 'unknown')
        content = self.activity.get('content', {})

        # Format timestamp (HH:MM:SS)
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%H:%M:%S")
        except (ValueError, TypeError):
            time_str = timestamp[:8] if len(timestamp) >= 8 else timestamp

        # Format content preview
        if isinstance(content, dict):
            content_str = str(content)
        else:
            content_str = str(content)

        preview = content_str[:100] + "..." if len(content_str) > 100 else content_str

        # Create rich text with colors
        text = Text()
        text.append(f"[{time_str}] ", style="dim cyan")
        text.append(f"{activity_type}", style="bold yellow")
        text.append(f" - ", style="dim")
        text.append(f"{agent_name}", style="bold green")
        text.append(f"\n{preview}", style="white")

        if self.expanded:
            text.append(f"\n\n[Full Content]\n{content_str}", style="dim white")

        yield Static(text)

    def on_click(self, event: events.Click):
        """Toggle expansion when clicked."""
        self.expanded = not self.expanded
        self.refresh(recompose=True)
        event.stop()


class ActivityLogViewer(Container):
    """Activity log viewer widget with search and filtering.

    Phase 5: TUI Integration

    Features:
    - Real-time activity display
    - Search with regex patterns
    - Filter by agent, type
    - Statistics display
    - Keyboard shortcuts (Enter to search, Escape to clear)
    - Auto-refresh during agent execution
    """

    DEFAULT_CSS = """
    ActivityLogViewer {
        border: none;
        background: transparent;
        height: 100%;
    }

    #log-header {
        height: 3;
        background: transparent;
        padding: 0 1;
    }

    #log-title {
        width: 1fr;
        padding: 0 1;
        color: #888888;
    }

    #log-stats {
        color: #666666;
        padding: 0 1;
    }

    #search-container {
        height: 3;
        background: transparent;
        padding: 0 1;
    }

    #search-input {
        width: 1fr;
    }

    #filter-container {
        height: 3;
        background: transparent;
        padding: 0 1;
    }

    #agent-filter {
        width: 1fr;
        margin-right: 1;
    }

    #type-filter {
        width: 1fr;
    }

    #log-list {
        height: 1fr;
        background: transparent;
    }

    #log-footer {
        height: 1;
        background: transparent;
        padding: 0 1;
        color: #666666;
    }

    .search-btn {
        min-width: 12;
        background: transparent;
        color: #888888;
        border: none;
    }

    .search-btn:hover {
        background: $surface;
    }

    .clear-btn {
        min-width: 10;
        background: transparent;
        color: #888888;
        border: none;
    }

    .clear-btn:hover {
        background: $surface;
    }
    """

    def __init__(
        self,
        session_name: str,
        log_dir: Path,
        db_path: Path,
        *args,
        **kwargs
    ):
        """Initialize activity log viewer.

        Args:
            session_name: Name of the session
            log_dir: Path to activity logs directory
            db_path: Path to activities database
        """
        super().__init__(*args, **kwargs)
        self.session_name = session_name
        self.log_dir = log_dir
        self.db_path = db_path

        # Create ActivitySearcher
        self.searcher = ActivitySearcher(
            session_name=session_name,
            log_dir=log_dir,
            db_path=db_path
        )

        # Current search state
        self.current_pattern = ""
        self.current_agent_filter = ""
        self.current_type_filter = ""

        # Activities cache
        self.activities: List[dict] = []
        self.total_activities = 0

        # Auto-refresh task
        self.refresh_task: Optional[asyncio.Task] = None
        self.auto_refresh_enabled = False

    def compose(self) -> ComposeResult:
        """Compose the activity log viewer."""
        yield Vertical(
            # Header with title and stats
            Horizontal(
                Label("Activity Logs", id="log-title"),
                Label("", id="log-stats"),
                id="log-header"
            ),

            # Search input
            Horizontal(
                Input(placeholder="Search pattern (regex)...", id="search-input"),
                Button("Search", id="search-btn", classes="search-btn"),
                Button("Clear", id="clear-btn", classes="clear-btn"),
                id="search-container"
            ),

            # Filter inputs
            Horizontal(
                Input(placeholder="Agent filter...", id="agent-filter"),
                Input(placeholder="Type filter...", id="type-filter"),
                Button("Stats", id="stats-btn", classes="search-btn"),
                id="filter-container"
            ),

            # Activity list
            ListView(id="log-list"),

            # Footer with help
            Label(
                "Enter: Search | Escape: Clear | Ctrl+R: Refresh | Ctrl+L: Toggle Log View",
                id="log-footer"
            )
        )

    def on_mount(self) -> None:
        """Initialize viewer on mount."""
        # Load initial activities
        self.load_activities()

        # Focus search input
        search_input = self.query_one("#search-input", Input)
        search_input.focus()

    def load_activities(self, pattern: str = ".*", limit: int = 50):
        """Load activities from searcher.

        Args:
            pattern: Search pattern (default: .* for all)
            limit: Maximum results to return
        """
        try:
            # Search activities
            self.activities = self.searcher.grep_logs(
                pattern=pattern,
                agent_name=self.current_agent_filter or None,
                activity_type=self.current_type_filter or None,
                max_results=limit
            )

            # Get total count
            stats = self.searcher.get_statistics()
            self.total_activities = stats['total_activities']

            # Update stats display
            self.update_stats_display()

            # Update list view
            self.update_list_view()

        except Exception as e:
            # Show error in stats
            stats_label = self.query_one("#log-stats", Label)
            stats_label.update(f"Error: {str(e)}")

    def update_stats_display(self):
        """Update statistics display in header."""
        # Check if widget is mounted (has been composed)
        try:
            stats_label = self.query_one("#log-stats", Label)
        except Exception:
            # Widget not mounted yet, skip update
            return

        showing = len(self.activities)
        total = self.total_activities

        stats_text = f"Showing {showing}/{total} activities"

        if self.current_pattern:
            stats_text += f" | Pattern: {self.current_pattern[:20]}"

        if self.current_agent_filter:
            stats_text += f" | Agent: {self.current_agent_filter}"

        if self.current_type_filter:
            stats_text += f" | Type: {self.current_type_filter}"

        stats_label.update(stats_text)

    def update_list_view(self):
        """Update list view with current activities."""
        # Check if widget is mounted (has been composed)
        try:
            log_list = self.query_one("#log-list", ListView)
        except Exception:
            # Widget not mounted yet, skip update
            return

        # Clear existing items
        log_list.clear()

        # Add activity items
        for activity in self.activities:
            log_list.append(ActivityLogItem(activity))

        # Auto-scroll to bottom (most recent)
        if len(self.activities) > 0:
            log_list.scroll_end(animate=False)

    @on(Button.Pressed, "#search-btn")
    def handle_search(self):
        """Handle search button press."""
        self.perform_search()

    @on(Button.Pressed, "#clear-btn")
    def handle_clear(self):
        """Handle clear button press."""
        # Clear search inputs
        search_input = self.query_one("#search-input", Input)
        agent_filter = self.query_one("#agent-filter", Input)
        type_filter = self.query_one("#type-filter", Input)

        search_input.value = ""
        agent_filter.value = ""
        type_filter.value = ""

        # Clear filters
        self.current_pattern = ""
        self.current_agent_filter = ""
        self.current_type_filter = ""

        # Reload all activities
        self.load_activities()

    @on(Button.Pressed, "#stats-btn")
    def handle_stats(self):
        """Handle stats button press - show statistics dialog."""
        try:
            stats = self.searcher.get_statistics()

            # Format stats message
            stats_text = Text()
            stats_text.append("Activity Statistics\n\n", style="bold cyan")
            stats_text.append(f"Total Activities: {stats['total_activities']}\n", style="white")
            stats_text.append(f"Total Chains: {stats['total_chains']}\n", style="white")
            stats_text.append(f"Active Chains: {stats['active_chains']}\n", style="white")
            stats_text.append(f"Total Errors: {stats['total_errors']}\n", style="white")
            stats_text.append(f"Avg Chain Depth: {stats['avg_chain_depth']:.1f}\n\n", style="white")

            stats_text.append("Activities by Type:\n", style="bold yellow")
            for activity_type, count in stats['activities_by_type'].items():
                stats_text.append(f"  - {activity_type}: {count}\n", style="dim white")

            stats_text.append("\nActivities by Agent:\n", style="bold green")
            for agent_name, count in stats['activities_by_agent'].items():
                agent_display = agent_name or "system"
                stats_text.append(f"  - {agent_display}: {count}\n", style="dim white")

            # Show in stats label (temporary display)
            stats_label = self.query_one("#log-stats", Label)
            stats_label.update(f"Stats: {stats['total_activities']} total, {stats['total_errors']} errors")

        except Exception as e:
            stats_label = self.query_one("#log-stats", Label)
            stats_label.update(f"Stats Error: {str(e)}")

    @on(Input.Submitted, "#search-input")
    def handle_search_enter(self, event: Input.Submitted):
        """Handle Enter key in search input."""
        self.perform_search()

    def perform_search(self):
        """Perform search with current input values."""
        # Get input values
        search_input = self.query_one("#search-input", Input)
        agent_filter = self.query_one("#agent-filter", Input)
        type_filter = self.query_one("#type-filter", Input)

        # Update current filters
        self.current_pattern = search_input.value or ".*"
        self.current_agent_filter = agent_filter.value
        self.current_type_filter = type_filter.value

        # Load activities with filters
        self.load_activities(pattern=self.current_pattern)

    def on_key(self, event: events.Key) -> None:
        """Handle keyboard shortcuts."""
        if event.key == "escape":
            # Clear search
            self.handle_clear()
            event.stop()
        elif event.key == "ctrl+r":
            # Refresh activities
            self.load_activities(pattern=self.current_pattern or ".*")
            event.stop()

    def enable_auto_refresh(self, interval: float = 2.0):
        """Enable auto-refresh during agent execution.

        Args:
            interval: Refresh interval in seconds
        """
        if not self.auto_refresh_enabled:
            self.auto_refresh_enabled = True
            self.refresh_task = asyncio.create_task(self._auto_refresh_loop(interval))

    def disable_auto_refresh(self):
        """Disable auto-refresh."""
        self.auto_refresh_enabled = False
        if self.refresh_task:
            self.refresh_task.cancel()
            self.refresh_task = None

    async def _auto_refresh_loop(self, interval: float):
        """Auto-refresh loop for real-time activity streaming.

        Args:
            interval: Refresh interval in seconds
        """
        while self.auto_refresh_enabled:
            try:
                # Reload activities
                self.load_activities(pattern=self.current_pattern or ".*")
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue on error
                await asyncio.sleep(interval)
