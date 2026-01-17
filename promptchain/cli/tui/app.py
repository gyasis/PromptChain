"""Main Textual application for PromptChain CLI."""

import json
import logging
import os
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

# Module logger for debug tracing (enabled by --dev flag)
logger = logging.getLogger(__name__)

import pyperclip
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Static

from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain

from ..agentic import TaskController, TaskControllerConfig, TaskStatus
from ..error_handler import ErrorHandler, RetryConfig
from ..models import Message, Session
from ..session_manager import SessionManager
from ..shell_executor import ShellCommandParser, ShellExecutor
from ..utils.file_context_manager import FileContextManager
from ..utils.output_formatter import OutputFormatter
from .activity_log_viewer import ActivityLogViewer
from .chat_view import ChatView
from .input_widget import InputWidget
from .observe_panel import ObservePanel
from .reasoning_progress_widget import ReasoningProgressWidget
from .autocomplete_popup import AutocompletePopup
from .status_bar import StatusBar
from .task_list_widget import TaskListWidget
from .token_bar import TokenBar

if TYPE_CHECKING:
    from ..models import Config
    from ..models.workflow import WorkflowState


class PromptChainApp(App):
    """Main PromptChain CLI application.

    Features:
    - Chat interface for conversation
    - Input widget for user messages
    - Status bar showing session info
    - Command handling (/exit, /help, etc.)
    - Session persistence
    """

    CSS = """
    /* Minimal TUI Theme - Inherits terminal background */
    Screen {
        background: transparent;
    }

    #chat-container {
        height: 1fr;
        background: transparent;
    }

    #chat-header {
        height: 1;
        background: transparent;
        padding: 0 1;
        align: right middle;
    }

    #chat-title {
        width: 1fr;
        padding: 0 1;
    }

    .copy-btn {
        min-width: 16;
        background: transparent;
        color: #666666;
        border: none;
    }

    .copy-btn:hover {
        background: $surface;
        color: #888888;
    }


    ChatView {
        height: 1fr;
        border: none;
        background: transparent;
        padding: 0;
    }

    InputWidget {
        height: 3;
        border: none;
        background: transparent;
        padding: 0;
    }

    StatusBar {
        dock: bottom;
        height: 1;
        background: transparent;
        color: #666666;
        border: none;
    }

    Header {
        display: none;
    }

    Footer {
        background: transparent;
        color: #666666;
        border: none;
    }

    .copy-feedback {
        color: #888888;
    }

    .system-message {
        padding: 0;
        color: #666666;
        text-style: italic;
    }

    /* Autocomplete popup styling */
    AutocompletePopup {
        layer: autocomplete;
        dock: bottom;
        offset: 0 -4;
        width: 60;
        height: auto;
        max-height: 12;
        background: $surface;
        border: solid $primary;
        padding: 0 1;
        display: none;
    }
    AutocompletePopup.visible {
        display: block;
    }
    """

    # Define layers for proper z-ordering (autocomplete popup above content)
    LAYERS = ["base", "autocomplete"]

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+d", "quit", "Quit"),
        ("ctrl+l", "toggle_log_view", "Activity Logs"),
        ("ctrl+o", "toggle_observe", "Observe"),  # T118: Toggle observe panel
    ]

    def __init__(
        self,
        session_name: str = "default",
        sessions_dir: Optional[Path] = None,
        config: Optional["Config"] = None,
        verbose_mode: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize the PromptChain app.

        Args:
            session_name: Name of session to load/create
            sessions_dir: Directory for session storage
            config: Configuration object (T151-T153)
            verbose_mode: Enable verbose observability mode (T118)
        """
        super().__init__(*args, **kwargs)
        self.session_name = session_name
        self.verbose_mode = verbose_mode  # T118: Verbose observability mode

        # Store config (T151-T153)
        from ..models import Config

        self.config = config if config else Config.load_or_create_default()

        # Initialize session manager
        if sessions_dir is None:
            sessions_dir = Path.home() / ".promptchain" / "sessions"
        self.session_manager = SessionManager(sessions_dir=sessions_dir)

        # Initialize file context manager for @syntax support (User Story 4)
        self.file_context_manager = FileContextManager()

        # Initialize shell executor for !command support (User Story 5: T123)
        self.shell_executor = ShellExecutor()
        self.shell_parser = ShellCommandParser()
        self.shell_mode = False  # Track if !! shell mode is active

        # Session will be loaded/created in on_mount
        self.session: Optional[Session] = None

        # AgentChain instance - handles both single-agent and multi-agent router mode (T037)
        self.agent_chain: Optional[AgentChain] = None

        # Legacy PromptChain for backward compatibility with single-agent mode
        self.single_agent_chain: Optional[PromptChain] = None

        # Initialize comprehensive error handler (T141)
        self.error_handler = ErrorHandler(
            retry_config=RetryConfig(
                max_attempts=3,
                base_delay=1.0,
                max_delay=30.0,
            ),
            debug_mode=config.debug_mode if config else False,
        )

        # Activity log viewer (Phase 5)
        self.log_viewer: Optional[ActivityLogViewer] = None
        self.log_viewer_visible = False

        # Track last displayed agent for switch detection (T042)
        self.last_displayed_agent: Optional[str] = None

        # Reasoning progress widget (T051)
        self.reasoning_progress: Optional[ReasoningProgressWidget] = None

        # Task list widget for displaying agent task progress
        self.task_list_widget: Optional[TaskListWidget] = None

        # Observe panel for verbose mode (T118)
        self.observe_panel: Optional[ObservePanel] = None

        # MLflow observer for optional observability (T118++)
        self._mlflow_observer = None

        # Task controller for agentic loop control (task completion, handoffs)
        task_controller_config = TaskControllerConfig(
            max_internal_steps=self.config.agentic.default_max_internal_steps,
            max_failed_attempts=self.config.agentic.task_completion_threshold,
            max_steps_without_progress=self.config.agentic.user_input_threshold,
            handoff_enabled=self.config.agentic.handoff_enabled,
        )
        self.task_controller = TaskController(config=task_controller_config)

        # REACT Loop: Async input queue for mid-execution user input
        import asyncio
        self.user_input_queue: asyncio.Queue = asyncio.Queue()
        self.is_processing = False  # Track if agent is currently processing

        # Cumulative API token tracking (from LiteLLM responses)
        self.cumulative_prompt_tokens = 0
        self.cumulative_completion_tokens = 0

        # Track AgenticStepProcessor call number for hierarchical step display (1.1, 2.1, etc.)
        self.processor_call_count = 0
        self.last_step_number = 0  # Track last seen step to detect new processor instances
        self.processor_completed = False  # Track if last processor completed
        self.last_displayed_step = None  # Track last hierarchical step shown to avoid repetition

        # Store main thread ID for thread-safe UI updates
        self._main_thread_id = threading.get_ident()

    def _safe_call_ui(self, callback: Callable[[], None]) -> None:
        """Thread-safe UI update helper.

        Calls the callback directly if already on the main thread,
        or uses call_from_thread if on a worker thread.

        Args:
            callback: Zero-argument callable to execute on the main thread
        """
        try:
            if threading.get_ident() == self._main_thread_id:
                # Already on main thread - call directly
                callback()
            else:
                # On worker thread - use call_from_thread
                self.call_from_thread(callback)
        except Exception as e:
            logger.debug(f"_safe_call_ui error: {e}")

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)
        yield Container(
            Vertical(
                Horizontal(
                    Static("", id="chat-title"),
                    id="chat-header",
                ),
                ChatView(id="chat-view"),
                id="chat-container",
            ),
            TaskListWidget(id="task-list-widget"),
            ReasoningProgressWidget(id="reasoning-progress"),
            ObservePanel(id="observe-panel"),  # T118: Verbose observability panel
            InputWidget(id="input-widget"),
            TokenBar(id="token-bar"),  # Real-time token usage display
            AutocompletePopup(id="autocomplete-popup"),  # Slash command autocomplete
        )
        yield StatusBar(id="status-bar")
        yield Footer()

    def _handle_error(self, error: Exception, context: str = "") -> Message:
        """Global error handler for graceful crash recovery (T141).

        Uses comprehensive ErrorHandler for classification, user-friendly messages,
        and recovery hints. Also logs errors to session for debugging (T143).

        Args:
            error: The exception that occurred
            context: Additional context about what was being done

        Returns:
            Message object with formatted error for display and logging
        """
        # Use error handler to classify and create context
        error_context = self.error_handler.classify_error(error, context)

        # Track error in history
        self.error_handler.track_error(error_context)

        # Create message from context
        message = self.error_handler.create_error_message(error_context)

        # Add recovery hint if available
        if error_context.recovery_hint:
            message.content += f"\n\n[dim]Recovery suggestion:\n{error_context.recovery_hint}[/dim]"

        return message

    def _get_help_text(self, topic: Optional[str] = None) -> str:
        """Get formatted help text, optionally for specific topic (T154-T156).

        Args:
            topic: Help topic (commands, shell, files, shortcuts, config)

        Returns:
            Formatted help text with Rich markup
        """
        if topic == "commands":
            return (
                "[bold]Slash Commands[/bold]\n\n"
                "  [bold]/exit[/bold] - Save session and exit\n"
                "  [bold]/help[/bold] \\[topic] - Show help (topics: commands, shell, files, shortcuts, config)\n"
                "  [bold]/session[/bold] - Show session info\n"
                "  [bold]/agent[/bold] - Show active agent info\n"
                "  [bold]/mentalmodel[/bold] - Show agent's mental model and capabilities\n"
                "  [bold]/clear[/bold] - Clear chat history\n"
                "  [bold]/cache clear[/bold] - Clear Python __pycache__ directories\n\n"
                "[dim]Tip: Use Tab to autocomplete slash commands[/dim]"
            )
        elif topic == "shell":
            return (
                "[bold]Shell Commands[/bold]\n\n"
                "[bold]Single Command Execution:[/bold]\n"
                "  [bold]!command[/bold] - Execute shell command\n"
                "  Examples:\n"
                "    !ls -la\n"
                "    !git status\n"
                "    !python script.py\n\n"
                "[bold]Shell Mode:[/bold]\n"
                "  [bold]!![/bold] - Toggle shell mode (all input becomes commands)\n"
                "  Type !! again to exit shell mode\n\n"
                "[dim]Shell commands show formatted output with exit codes and timing[/dim]"
            )
        elif topic == "files":
            return (
                "[bold]File References[/bold]\n\n"
                "[bold]Include File Content:[/bold]\n"
                "  [bold]@filename[/bold] - Include single file\n"
                "  [bold]@path/to/file.py[/bold] - Include file with path\n\n"
                "[bold]Include Directory Listings:[/bold]\n"
                "  [bold]@directory/[/bold] - List directory contents\n\n"
                "[bold]Glob Patterns:[/bold]\n"
                "  [bold]@*.py[/bold] - Include all Python files\n"
                "  [bold]@src/**/*.ts[/bold] - Include all TypeScript files in src/\n\n"
                "[dim]File content is injected before sending to the LLM[/dim]"
            )
        elif topic == "shortcuts":
            return (
                "[bold]Keyboard Shortcuts[/bold]\n\n"
                "[bold]Input Shortcuts:[/bold]\n"
                "  [bold]Enter[/bold] - Submit message\n"
                "  [bold]Shift+Enter[/bold] - Insert newline (multi-line input)\n"
                "  [bold]Tab[/bold] - Autocomplete slash commands\n"
                "  [bold]Up/Down[/bold] - Navigate command history\n"
                "  [bold]Ctrl+C[/bold] - Cancel current input\n\n"
                "[bold]App Shortcuts:[/bold]\n"
                "  [bold]Ctrl+D[/bold] - Exit application\n"
                "  [bold]Ctrl+L[/bold] - Toggle Activity Logs (view agent activities)\n\n"
                "[bold]Message Selection:[/bold]\n"
                "  [bold]Click[/bold] - Select/deselect message\n"
                "  [bold]c[/bold] - Copy focused message (when selected)\n"
            )
        elif topic == "config":
            return (
                "[bold]Configuration[/bold]\n\n"
                "[bold]Config File Location:[/bold]\n"
                f"  {self.config.get_default_path()}\n\n"
                "[bold]Current Settings:[/bold]\n"
                f"  Default Model: [bold]{self.config.default_model}[/bold]\n"
                f"  Default Agent: [bold]{self.config.default_agent}[/bold]\n"
                f"  Max Displayed Messages: [bold]{self.config.ui.max_displayed_messages}[/bold]\n"
                f"  Lazy Load Agents: [bold]{self.config.performance.lazy_load_agents}[/bold]\n"
                f"  History Max Tokens: [bold]{self.config.performance.history_max_tokens}[/bold]\n\n"
                "[dim]Edit config.json to customize settings[/dim]"
            )
        else:
            # General help - show overview
            return (
                "[bold]PromptChain CLI Help[/bold]\n\n"
                "[bold]Available Help Topics:[/bold]\n"
                "  [bold]/help commands[/bold] - Slash commands reference\n"
                "  [bold]/help shell[/bold] - Shell command integration\n"
                "  [bold]/help files[/bold] - File reference syntax\n"
                "  [bold]/help shortcuts[/bold] - Keyboard shortcuts\n"
                "  [bold]/help config[/bold] - Configuration settings\n\n"
                "[bold]Quick Start:[/bold]\n"
                "  - Type messages to chat with the active agent\n"
                "  - Use @file.txt to include file content\n"
                "  - Use !command to run shell commands\n"
                "  - Press Tab to autocomplete commands\n"
                "  - Press Up/Down to navigate history\n\n"
                "[dim]Type [bold]/help \\[topic][/bold] for detailed information[/dim]"
            )

    def action_toggle_log_view(self):
        """Toggle activity log viewer visibility (Ctrl+L) - Phase 5.

        Shows/hides the ActivityLogViewer widget for interactive log browsing.
        Creates the widget on first toggle if not already initialized.
        """
        if not self.session or not self.session.activity_logger:
            # Show message if activity logging not enabled
            chat_view = self.query_one("#chat-view", ChatView)
            from ..models import Message

            error_msg = Message(
                role="system",
                content="[italic]Activity logging is not enabled for this session.[/italic]\n"
                        "[dim]Activity logs are only available when ActivityLogger is configured.[/dim]"
            )
            chat_view.add_message(error_msg)
            return

        # Create log viewer on first toggle if not exists
        if not self.log_viewer:
            session_dir = self.session_manager.sessions_dir / self.session.id
            self.log_viewer = ActivityLogViewer(
                session_name=self.session.name,
                log_dir=session_dir / "activity_logs",
                db_path=session_dir / "activities.db",
                id="log-viewer"
            )
            self.log_viewer.display = False  # Hidden initially

            # Mount the widget dynamically
            container = self.query_one(Container)
            container.mount(self.log_viewer)

        # Toggle visibility
        self.log_viewer_visible = not self.log_viewer_visible
        self.log_viewer.display = self.log_viewer_visible

        if self.log_viewer_visible:
            # Refresh activities when showing
            self.log_viewer.load_activities()

            # Show feedback message
            chat_view = self.query_one("#chat-view", ChatView)
            from ..models import Message

            feedback_msg = Message(
                role="system",
                content="[bold]Activity Logs opened[/bold]\n"
                        "[dim]Press Ctrl+L again to close, or use search/filters in the log view[/dim]"
            )
            chat_view.add_message(feedback_msg)

    def action_toggle_observe(self):
        """Toggle observe panel visibility (Ctrl+O) - T118.

        Shows/hides the ObservePanel widget for detailed execution observability.
        """
        if self.observe_panel:
            self.observe_panel.toggle_panel()

            # Show feedback in chat
            chat_view = self.query_one("#chat-view", ChatView)
            from ..models import Message

            is_visible = "visible" in self.observe_panel.classes
            if is_visible:
                feedback_msg = Message(
                    role="system",
                    content="[bold]Observe Panel opened[/bold]\n"
                            "[dim]Shows detailed tool calls, LLM requests, and execution steps.[/dim]\n"
                            "[dim]Press Ctrl+O again to close, or use --verbose flag to auto-open[/dim]"
                )
            else:
                feedback_msg = Message(
                    role="system",
                    content="[dim]Observe Panel closed[/dim]"
                )
            chat_view.add_message(feedback_msg)

    def on_key(self, event) -> None:
        """Handle global key events (T147).

        Ctrl+C: Cancel current operation without exiting
        """
        if event.key == "ctrl+c":
            # Prevent default Ctrl+C behavior (which exits the app)
            event.prevent_default()
            try:
                # Clear input widget
                input_widget = self.query_one("#input-widget", InputWidget)
                input_widget.clear()
                # Show cancellation message (only if ChatView is fully mounted)
                chat_view = self.query_one("#chat-view", ChatView)
                if chat_view.is_attached:
                    from ..models import Message
                    cancel_msg = Message(role="system", content="[dim]Operation cancelled (Ctrl+C)[/dim]")
                    chat_view.add_message(cancel_msg)
            except Exception as e:
                # Ignore errors during early startup
                logger.debug(f"Ctrl+C handler error (likely early startup): {e}")

    async def on_mount(self):
        """Handle app mount - load/create session and display welcome message."""
        logger.debug("on_mount: START")

        # Import tools registry to get tool count
        from ..tools import registry
        logger.debug("on_mount: Tools registry imported")

        # Try to load existing session, or create new one
        is_new_session = False
        try:
            logger.debug(f"on_mount: Loading session '{self.session_name}'")
            self.session = self.session_manager.load_session(self.session_name)
            assert self.session is not None  # Type narrowing for mypy
            self.title = f"PromptChain - {self.session.name}"
            logger.debug(f"on_mount: Session loaded - {len(self.session.messages)} messages")

            # Load existing messages into chat view
            chat_view = self.query_one("#chat-view", ChatView)
            chat_view.load_messages(self.session.messages)
            logger.debug("on_mount: Messages loaded into chat view")

        except ValueError:
            # Session doesn't exist, create it
            logger.debug(f"on_mount: Creating new session '{self.session_name}'")
            is_new_session = True
            self.session = self.session_manager.create_session(
                name=self.session_name, working_directory=Path.cwd()
            )
            assert self.session is not None  # Type narrowing for mypy
            self.title = f"PromptChain - {self.session.name}"
            logger.debug("on_mount: New session created")

        # Display welcome message (for both new AND existing sessions)
        from ..models import Message

        # Get active agent and model info
        active_agent = self.session.agents.get(self.session.active_agent)
        active_model = active_agent.model_name if active_agent else self.session.default_model

        # Get tool count
        tool_count = len(registry.list_tools())

        # Get agent's tools if available
        agent_tool_count = len(active_agent.tools) if active_agent and active_agent.tools else tool_count

        session_status = "New session created" if is_new_session else "Session loaded"

        # Wire config.performance.history_max_tokens to session
        self.session.history_max_tokens = self.config.performance.history_max_tokens

        welcome_msg = Message(
            role="system",
            content=(
                f"[bold]Welcome to PromptChain CLI[/bold]\n\n"
                f"[dim]Session:[/dim] {self.session.name} [dim]({session_status})[/dim]\n"
                f"[dim]Active Agent:[/dim] {self.session.active_agent}\n"
                f"[dim]Model:[/dim] [bold]{active_model}[/bold]\n"
                f"[dim]Tools Loaded:[/dim] [bold]{agent_tool_count}/{tool_count}[/bold] available\n"
                f"[dim]Working Directory:[/dim] {self.session.working_directory}\n\n"
                "[dim]Type your message and press Enter to chat.[/dim]\n"
                "[dim]Commands: /exit, /help, /session, /agent[/dim]\n"
                "[dim]Shortcuts: Ctrl+C or Ctrl+D to exit[/dim]"
            ),
        )
        chat_view = self.query_one("#chat-view", ChatView)
        chat_view.add_message(welcome_msg)

        # Update status bar with active agent's model and router mode indicator (T039, T061)
        status_bar = self.query_one("#status-bar", StatusBar)
        active_agent = self.session.agents.get(self.session.active_agent)
        active_model = active_agent.model_name if active_agent else self.session.default_model

        # Determine if router mode is active (T039)
        orchestration = self.session.orchestration_config
        is_router_mode = (
            orchestration
            and orchestration.execution_mode == "router"
            and len(self.session.agents) > 1
        )

        # Collect MCP server status (T069)
        mcp_server_status = [
            {"id": server.id, "state": server.state} for server in self.session.mcp_servers
        ]

        # Get history token stats for status bar
        history_stats = self.session.history_manager.get_statistics()

        status_bar.update_session_info(
            session_name=self.session.name,
            active_agent=self.session.active_agent,
            model_name=active_model,
            message_count=len(self.session.messages),
            session_state=self.session.state,
            router_mode=is_router_mode,  # T039: Show router mode indicator
            mcp_servers=mcp_server_status,  # T069: MCP server status display
            token_count=history_stats.get("total_tokens", 0),
            max_tokens=self.session.history_max_tokens,
        )

        # Set focus to input widget
        input_widget = self.query_one("#input-widget", InputWidget)
        input_widget.focus()

        # Capture reasoning progress widget reference (T051)
        self.reasoning_progress = self.query_one("#reasoning-progress", ReasoningProgressWidget)

        # Capture task list widget reference
        self.task_list_widget = self.query_one("#task-list-widget", TaskListWidget)

        # T118: Capture observe panel reference and show if verbose mode
        self.observe_panel = self.query_one("#observe-panel", ObservePanel)
        if self.verbose_mode:
            self.observe_panel.show_panel()
            self.observe_panel.log_info("Verbose observability mode enabled")

            # T118++: Connect ObservePanel to CallbackManager (primary observability system)
            await self._setup_callback_bridge()

        # Re-initialize error handler with session ID for JSONL logging (T143)
        logger.debug("on_mount: Initializing error handler")
        self.error_handler = ErrorHandler(
            retry_config=RetryConfig(
                max_attempts=3,
                base_delay=1.0,
                max_delay=30.0,
            ),
            debug_mode=self.config.debug_mode if self.config else False,
            session_id=self.session.id,
            sessions_dir=self.session_manager.sessions_dir,
        )

        # Auto-connect default MCP servers from config
        logger.debug("on_mount: Auto-connecting MCP servers...")
        await self._auto_connect_default_mcp_servers()
        logger.debug("on_mount: MCP servers connected")

        # Initialize PromptChain for active agent (T031-T032)
        logger.debug("on_mount: Initializing agent chain...")
        self._initialize_agent_chain()
        logger.debug("on_mount: COMPLETE")

    async def _setup_callback_bridge(self) -> None:
        """Connect CallbackManager to ObservePanel (primary observability system).

        This bridges internal callback events to the TUI without requiring
        any external dependencies (no MLflow needed).

        T118++: Primary observability architecture - callbacks first, MLflow optional.
        """
        from promptchain.utils.execution_events import ExecutionEventType

        # Get current agent's chain
        active_agent = self.session.agents.get(self.session.active_agent)
        if not active_agent:
            logger.debug("No active agent - callback bridge not established")
            return

        # Get the chain (could be PromptChain or AgentChain)
        chain = getattr(active_agent, 'chain', None) or getattr(active_agent, 'agent_chain', None)
        if not chain or not hasattr(chain, 'register_callback'):
            logger.debug("Agent has no callback manager - bridge not established")
            return

        # Define callback for LLM and tool events
        def observability_callback(event):
            """Bridge callback events to ObservePanel"""
            try:
                if event.event_type == ExecutionEventType.MODEL_CALL_START:
                    # LLM call started
                    model = event.data.get("model_name", "unknown")
                    messages = event.data.get("messages", [])
                    prompt_preview = str(messages[-1] if messages else "")[:100] if messages else "Empty prompt"
                    self.observe_panel.log_entry(
                        "llm-request",
                        f"[{model}] {prompt_preview}..."
                    )

                elif event.event_type == ExecutionEventType.MODEL_CALL_END:
                    # LLM call completed
                    model = event.data.get("model_name", "unknown")
                    usage = event.data.get("usage", {})
                    response = event.data.get("response", "")
                    response_preview = str(response)[:100] if response else "No response"

                    # Format token info
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    token_info = f"({prompt_tokens}p + {completion_tokens}c = {total_tokens}t)"

                    self.observe_panel.log_entry(
                        "llm-response",
                        f"[{model}] {response_preview}... {token_info}"
                    )

                elif event.event_type == ExecutionEventType.TOOL_CALL_START:
                    # Tool call started
                    tool_name = event.data.get("tool_name", "unknown")
                    args_preview = str(event.data.get("arguments", ""))[:50]
                    self.observe_panel.log_entry("tool-call", f"Calling: {tool_name}({args_preview}...)")

                elif event.event_type == ExecutionEventType.TOOL_CALL_END:
                    # Tool call completed
                    tool_name = event.data.get("tool_name", "unknown")
                    result = event.data.get("result", "")
                    result_preview = str(result)[:100] if result else "No result"
                    self.observe_panel.log_entry("tool-result", f"{tool_name}: {result_preview}...")

                elif event.event_type == ExecutionEventType.STEP_START:
                    # Chain step started
                    step_num = event.data.get("step_index", "?")
                    self.observe_panel.log_info(f"Step {step_num} started")

                elif event.event_type == ExecutionEventType.STEP_END:
                    # Chain step completed
                    step_num = event.data.get("step_index", "?")
                    self.observe_panel.log_info(f"Step {step_num} completed")

            except Exception as e:
                logger.error(f"Error in observability callback: {e}")

        # Register callback with chain
        try:
            chain.register_callback(observability_callback)
            logger.info("✓ Callback bridge established: CallbackManager → ObservePanel (primary observability)")
            self.observe_panel.log_info("✓ Connected to CallbackManager (primary observability)")
        except Exception as e:
            logger.error(f"Failed to establish callback bridge: {e}")
            self.observe_panel.log_info(f"⚠ Callback bridge failed: {e}")

        # T118++: Optionally activate MLflow observer plugin
        try:
            from promptchain.observability import MLflowObserver

            mlflow_observer = MLflowObserver(
                experiment_name=f"promptchain-{self.session.name}",
                tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )

            if mlflow_observer.is_available():
                chain.register_callback(mlflow_observer.handle_event)
                logger.info("✓ MLflow observer plugin activated (optional observability)")
                self.observe_panel.log_info("✓ MLflow observer connected (tracking enabled)")

                # Store reference for cleanup
                self._mlflow_observer = mlflow_observer
            else:
                logger.debug("MLflow observer not available (optional - install with: pip install mlflow)")

        except ImportError:
            logger.debug("MLflow observer plugin not available (optional)")
        except Exception as e:
            logger.warning(f"MLflow observer initialization failed: {e}")

    async def _auto_connect_default_mcp_servers(self) -> None:
        """Auto-connect default MCP servers from config on startup.

        Reads default servers from config.mcp.default_servers and:
        1. Adds them to session.mcp_servers if not already present
        2. Connects to servers with auto_connect=True

        This ensures MCP servers persist across sessions without manual setup.
        """
        if not self.session or not self.config:
            return

        # Check if MCP config has default servers
        if not hasattr(self.config, 'mcp') or not self.config.mcp.auto_connect_on_start:
            return

        from ..models.mcp_config import MCPServerConfig
        from ..utils.mcp_manager import MCPManager

        chat_view = self.query_one("#chat-view", ChatView)

        for server_def in self.config.mcp.default_servers:
            server_id = server_def.get("id")
            if not server_id:
                continue

            # Check if server already exists in session
            existing = next((s for s in self.session.mcp_servers if s.id == server_id), None)

            if not existing:
                # Add server to session
                server = MCPServerConfig(
                    id=server_id,
                    type=server_def.get("type", "stdio"),
                    command=server_def.get("command", ""),
                    args=server_def.get("args", []),
                    auto_connect=server_def.get("auto_connect", True),
                )
                self.session.mcp_servers.append(server)
                existing = server

            # Connect if auto_connect is enabled
            if existing.auto_connect and existing.state != "connected":
                try:
                    mcp_manager = MCPManager(self.session)
                    success = await mcp_manager.connect_server(server_id)

                    if success:
                        tools = existing.discovered_tools
                        # Show brief status in welcome (grayscale)
                        from ..models import Message
                        connect_msg = Message(
                            role="system",
                            content=f"[dim]MCP '{server_id}' connected ({len(tools)} tools)[/dim]"
                        )
                        chat_view.add_message(connect_msg)
                except Exception as e:
                    # Log error but don't crash startup
                    from ..models import Message
                    error_msg = Message(
                        role="system",
                        content=f"[dim]MCP '{server_id}' connection failed: {str(e)[:50]}[/dim]"
                    )
                    chat_view.add_message(error_msg)

    def show_reasoning_progress(self, objective: str, max_steps: int) -> None:
        """Show reasoning progress widget with initial state (T052).

        Args:
            objective: The objective being pursued
            max_steps: Maximum number of internal steps
        """
        if not self.reasoning_progress:
            return

        self.reasoning_progress.update_progress(
            current_step=0,
            max_steps=max_steps,
            objective=objective,
            status="Starting..."
        )
        self.reasoning_progress.show_progress()

    def update_reasoning_step(self, step_num: int, status: str) -> None:
        """Update current reasoning step (T052).

        Args:
            step_num: Current step number (1-indexed)
            status: Current status message
        """
        if not self.reasoning_progress:
            return

        # Get current max_steps from widget state
        max_steps = self.reasoning_progress._max_steps or 10

        self.reasoning_progress.update_progress(
            current_step=step_num,
            max_steps=max_steps,
            status=status
        )

    def hide_reasoning_progress(self) -> None:
        """Hide reasoning progress widget (T052)."""
        if self.reasoning_progress:
            self.reasoning_progress.hide_progress()
            self.reasoning_progress.reset()

    def _refresh_task_list_widget(self) -> None:
        """Refresh task list widget from global task list manager.

        Fetches the current task list state from the task_list_tool's
        global manager and updates the TUI widget display.
        """
        if not self.task_list_widget:
            return

        try:
            # Import the task list manager accessor
            from ..tools.library.task_list_tool import get_task_list_manager

            manager = get_task_list_manager()
            self.task_list_widget.set_task_manager(manager)
        except Exception:
            # Silently fail if task list module not available
            pass

    def _log_reasoning_step(
        self,
        step_num: int,
        step_content: str,
        tool_calls: Optional[List] = None
    ) -> None:
        """Log individual reasoning step to session history (T053).

        Args:
            step_num: Step number (1-indexed)
            step_content: Content/status of the step
            tool_calls: Optional list of tool calls made in this step
        """
        if not self.session:
            return

        from datetime import datetime

        # Create log entry for session JSONL
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "reasoning_step",
            "session_id": self.session.id,
            "agent_name": self.session.active_agent,
            "step_number": step_num,
            "content_preview": step_content[:200],
            "had_tool_calls": len(tool_calls) > 0 if tool_calls else False
        }

        # Log to session manager's JSONL
        # TODO: Implement _append_to_jsonl method in SessionManager
        # self.session_manager._append_to_jsonl(self.session.id, log_entry)

        # Also log to ExecutionHistoryManager if available
        if hasattr(self.session, 'history_manager'):
            self.session.history_manager.add_entry(
                entry_type="reasoning_step",
                content=f"Step {step_num}: {step_content}",
                metadata={
                    "step_number": step_num,
                    "tool_calls": tool_calls
                },
                source="agentic_step_processor"
            )

    def _check_agentic_completion(self, result: str, objective: str) -> bool:
        """Check if AgenticStepProcessor successfully completed objective (T054).

        Uses TaskController for enhanced status detection including:
        - TASK_COMPLETE: Objective achieved
        - TASK_BLOCKED: Waiting for user input
        - TASK_REQUIRES_HANDOFF: Route to another agent
        - TASK_IN_PROGRESS: Continue working

        Returns:
            True if objective completed, False if exhausted early
        """
        # Use TaskController for enhanced detection
        status = self.task_controller.detect_status(result)

        if status == TaskStatus.COMPLETE:
            return True

        # AgenticStepProcessor sets completion flag on agent_chain (fallback)
        if hasattr(self.agent_chain, 'last_processor_completed'):
            return self.agent_chain.last_processor_completed

        # Fallback: heuristic check based on result content
        if status == TaskStatus.UNKNOWN and len(result) > 100:
            # Simple keyword check - if result mentions objective keywords
            objective_words = set(objective.lower().split())
            result_words = set(result.lower().split())
            overlap = len(objective_words & result_words)
            return overlap > len(objective_words) * 0.3  # 30% keyword match

        return False

    def _check_task_status(self, result: str, step_number: int) -> TaskStatus:
        """Check task status using TaskController.

        Updates the task state and returns the detected status.

        Args:
            result: Agent response text
            step_number: Current step number

        Returns:
            Detected TaskStatus
        """
        self.task_controller.update_state(result, step_number)
        return self.task_controller.task_state.status

    async def _handle_handoff(self, result: str) -> Optional[str]:
        """Handle handoff to another agent when TASK_REQUIRES_HANDOFF detected.

        Args:
            result: Agent response containing handoff signal

        Returns:
            Response from target agent if handoff successful, None otherwise
        """
        if not self.config.agentic.handoff_enabled:
            return None

        handoff_context = self.task_controller.get_handoff_context()
        target_agent = handoff_context.get("handoff_target")

        if not target_agent:
            return None

        # Check if target agent exists
        if target_agent not in self.session.agents:
            # Log warning and return None
            return None

        # Switch to target agent
        try:
            self.switch_active_agent(target_agent)

            # Get the agent chain
            agent_chain = self._get_or_create_agent_chain()
            if not agent_chain:
                return None

            # Execute handoff with context (REACT v0.4.3)
            continuation_prompt = handoff_context.get("continuation_prompt", result)
            handoff_response = await agent_chain.run_chat_turn_async(
                continuation_prompt,
                user_input_queue=self.user_input_queue,
                streaming_callback=self._streaming_callback
            )

            return handoff_response

        except Exception as e:
            # Log error but don't crash
            return None

    def _get_user_return_message(self) -> Optional[Message]:
        """Generate message when returning control to user.

        Returns:
            Message with context for user, or None if not needed
        """
        context = self.task_controller.get_user_return_context()

        if context.get("status") == TaskStatus.BLOCKED.value:
            questions = context.get("questions", [])
            if questions:
                content = (
                    "[italic]I need your input to continue:[/italic]\n\n" +
                    "\n".join(f"- {q}" for q in questions[:5])  # Limit to 5 questions
                )
                return Message(role="system", content=content)

        return None

    def _log_completion(self, objective: str, completed: bool, result: str) -> None:
        """Log reasoning completion status (T054).

        Args:
            objective: The objective that was being pursued
            completed: Whether objective was successfully completed
            result: Final result content
        """
        if not self.session:
            return

        completion_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "reasoning_completion",
            "session_id": self.session.id,
            "agent_name": self.session.active_agent,
            "objective": objective,
            "completed": completed,
            "result_length": len(result)
        }

        # Log to session history.jsonl
        session_dir = self.session_manager.sessions_dir / self.session.id
        log_file = session_dir / "history.jsonl"

        try:
            # Ensure session directory exists
            session_dir.mkdir(parents=True, exist_ok=True)

            # Append to JSONL
            with open(log_file, "a") as f:
                json.dump(completion_entry, f)
                f.write("\n")
        except Exception as log_error:
            # If logging fails, don't crash - just log to stderr
            import sys
            print(
                f"ERROR: Failed to log completion to {log_file}: {log_error}",
                file=sys.stderr,
            )

    def _reasoning_progress_callback(self, current_step: int, max_steps: int, status: str) -> None:
        """Progress callback for AgenticStepProcessor reasoning updates (T052, T054).

        Args:
            current_step: Current step number (1-indexed)
            max_steps: Maximum number of steps
            status: Current status message
        """
        if not self.reasoning_progress:
            return

        # Detect new AgenticStepProcessor instance:
        # - If step goes backward (e.g., 3 -> 1), new processor started
        # - If step is 1 and last was 0 (first ever call), new processor
        # - If step is 1 and previous processor completed, new processor started
        new_processor_detected = False

        # Debug: Log every callback invocation
        logger.debug(f"[STEP TRACKING] Callback invoked: current_step={current_step}, "
                    f"last_step={self.last_step_number}, processor_completed={self.processor_completed}, "
                    f"processor_call_count={self.processor_call_count}, status={status[:50]}")

        if (current_step < self.last_step_number or
            (current_step == 1 and self.last_step_number == 0) or
            (current_step == 1 and self.processor_completed)):
            self.processor_call_count += 1
            self.processor_completed = False  # Reset completion flag
            new_processor_detected = True
            self.reasoning_progress.show_progress()
            logger.debug(f"[STEP TRACKING] ✓ New processor detected! Count: {self.processor_call_count}, "
                        f"current_step: {current_step}, last_step: {self.last_step_number}")

        self.last_step_number = current_step

        # Track completion for next processor detection
        if status == "Complete":
            self.processor_completed = True
            logger.debug(f"[STEP TRACKING] Processor {self.processor_call_count} completed at step {current_step}")

        # Format hierarchical step number: {processor_call}.{internal_step}
        hierarchical_step = f"{self.processor_call_count}.{current_step}"
        logger.debug(f"[STEP TRACKING] Formatted step: {hierarchical_step}, status: {status}")

        # Update progress display
        self.reasoning_progress.update_progress(
            current_step=current_step,
            max_steps=max_steps,
            objective="",  # Could be populated from AgenticStepProcessor if needed
            status=status
        )

        # T118: Log to observe panel in verbose mode with hierarchical numbering
        if self.verbose_mode and self.observe_panel:
            # Improved display: Only show step number for FIRST callback of each unique step
            # Subsequent status updates within same step shown as indented sub-items
            is_new_step = (self.last_displayed_step != hierarchical_step)

            if is_new_step:
                # First callback for this step - show full step prefix
                self.last_displayed_step = hierarchical_step
                display_prefix = f"[Step {hierarchical_step}]"
            else:
                # Status update within same step - show as sub-item
                display_prefix = "  └─"

            # Detect entry type from status message
            status_lower = status.lower()
            if "calling" in status_lower or "tool" in status_lower:
                # Extract tool name if possible
                tool_name = status.replace("Calling:", "").replace("Calling", "").strip()
                self.observe_panel.log_entry("tool-call", f"{display_prefix} {status}")
            elif "synthesizing" in status_lower or "result" in status_lower:
                self.observe_panel.log_entry("tool-result", f"{display_prefix} {status}")
            elif "reasoning" in status_lower or "thinking" in status_lower:
                if is_new_step:
                    self.observe_panel.log_reasoning(hierarchical_step, status)
                else:
                    self.observe_panel.log_entry("reasoning", f"{display_prefix} {status}")
            elif "complete" in status_lower:
                self.observe_panel.log_info(f"{display_prefix} {status}")
            else:
                self.observe_panel.log_entry("info", f"{display_prefix} {status}")

        # T053: Log each reasoning step to session history
        if self.session:
            self._log_reasoning_step(
                step_num=current_step,
                step_content=f"Status: {status}",
                tool_calls=None  # Could be enhanced to track actual tool calls
            )

        # Refresh task list widget (may have been updated by tool calls)
        self._refresh_task_list_widget()

        # T054: Hide widget when complete
        if status == "Complete":
            # Give user a moment to see completion before hiding
            self.set_timer(1.0, lambda: self.reasoning_progress.hide_progress())

        # T055: Handle max steps exhaustion
        if current_step == max_steps and status != "Complete":
            # Update status to show partial completion
            self.reasoning_progress.update_progress(
                current_step=current_step,
                max_steps=max_steps,
                status="Max steps reached - partial result"
            )
            # Hide after delay
            self.set_timer(2.0, lambda: self.reasoning_progress.hide_progress())

    def _add_task_internal_step(self, step_type: str, content: str) -> None:
        """Add an internal step to the current task in TaskListWidget.

        Provides Claude Code-style visibility into agent operations.

        Args:
            step_type: Type of step ("thinking", "tool_call", "tool_result", "llm_call", "error")
            content: Step content/description
        """
        try:
            def add_step():
                try:
                    task_list = self.query_one("#task-list-widget", TaskListWidget)
                    task_list.add_internal_step(step_type, content)
                except Exception:
                    pass  # Widget may not be ready
            self._safe_call_ui(add_step)
        except Exception:
            pass  # Silently ignore errors

    # REACT Loop: Streaming callback for real-time agent output
    def _streaming_callback(self, event_type: str, content: str) -> None:
        """Handle streaming events from AgenticStepProcessor.

        Displays real-time updates in the chat view as the agent thinks and acts.

        Args:
            event_type: Type of event ("thinking", "tool_call", "tool_result", "answer", "error", "user_input", "tokens")
            content: Event content to display (for "tokens", JSON with prompt_tokens/completion_tokens)
        """
        try:
            chat_view = self.query_one("#chat-view", ChatView)
            from ..models import Message

            # Format based on event type + add to task internal steps
            if event_type == "thinking":
                # Show thinking in dim/italic
                # Escape brackets to prevent Rich markup conflicts
                safe_content = content.replace('[', '\\[').replace(']', '\\]')
                msg = Message(
                    role="system",
                    content=f"[dim italic]🧠 {safe_content}[/dim italic]",
                    metadata={"streaming": True, "event_type": "thinking"}
                )
                # Track in task list (Claude Code style)
                self._add_task_internal_step("thinking", content)
            elif event_type == "tool_call":
                # Show tool call being made
                # Escape brackets to prevent Rich markup conflicts
                safe_content = content.replace('[', '\\[').replace(']', '\\]')
                msg = Message(
                    role="system",
                    content=f"[cyan]🔧 Calling: {safe_content}[/cyan]",
                    metadata={"streaming": True, "event_type": "tool_call"}
                )
                # Track in task list (Claude Code style)
                self._add_task_internal_step("tool_call", content)
            elif event_type == "tool_result":
                # Show tool result (truncated for display)
                preview = content[:300] + "..." if len(content) > 300 else content
                # Escape brackets to prevent Rich markup conflicts
                safe_preview = preview.replace('[', '\\[').replace(']', '\\]')
                msg = Message(
                    role="system",
                    content=f"[green]✓ {safe_preview}[/green]",
                    metadata={"streaming": True, "event_type": "tool_result"}
                )
                # Track in task list (Claude Code style)
                self._add_task_internal_step("tool_result", preview)
            elif event_type == "error":
                # Show errors in red
                # Escape brackets to prevent Rich markup conflicts
                safe_content = content.replace('[', '\\[').replace(']', '\\]')
                msg = Message(
                    role="system",
                    content=f"[red]⚠️ {safe_content}[/red]",
                    metadata={"streaming": True, "event_type": "error"}
                )
                # Track in task list (Claude Code style)
                self._add_task_internal_step("error", content)
            elif event_type == "user_input":
                # Show that user input was received during execution
                # Escape brackets to prevent Rich markup conflicts
                preview = content[:100] + "..." if len(content) > 100 else content
                safe_preview = preview.replace('[', '\\[').replace(']', '\\]')
                msg = Message(
                    role="system",
                    content=f"[yellow]📥 User input received: {safe_preview}[/yellow]",
                    metadata={"streaming": True, "event_type": "user_input"}
                )
            elif event_type == "tokens":
                # Real-time token usage update - update token bar and status bar
                logger.debug(f"Token event received: {content[:100]}")
                try:
                    import json
                    token_data = json.loads(content)
                    self.cumulative_prompt_tokens = token_data.get("prompt_tokens", 0)
                    self.cumulative_completion_tokens = token_data.get("completion_tokens", 0)
                    logger.debug(f"Token update: prompt={self.cumulative_prompt_tokens}, completion={self.cumulative_completion_tokens}")

                    # Get history stats for context window display
                    history_tokens = 0
                    max_history = self.session.history_max_tokens if hasattr(self.session, 'history_max_tokens') else 64000
                    try:
                        history_stats = self.session.history_manager.get_statistics()
                        history_tokens = history_stats.get("total_tokens", 0)
                    except Exception as e:
                        logger.debug(f"Failed to get history stats: {e}")

                    # Update both token bar and status bar (thread-safe)
                    def update_token_displays():
                        try:
                            # Update dedicated token bar
                            token_bar = self.query_one("#token-bar", TokenBar)
                            token_bar.update_tokens(
                                api_prompt_tokens=self.cumulative_prompt_tokens,
                                api_completion_tokens=self.cumulative_completion_tokens,
                                history_tokens=history_tokens,
                                max_history_tokens=max_history,
                            )
                            logger.debug("Token bar updated successfully")
                        except Exception as e:
                            logger.debug(f"Token bar update failed: {e}")

                        try:
                            # Also update status bar
                            status_bar = self.query_one("#status-bar", StatusBar)
                            status_bar.update_session_info(
                                api_prompt_tokens=self.cumulative_prompt_tokens,
                                api_completion_tokens=self.cumulative_completion_tokens,
                            )
                        except Exception as e:
                            logger.debug(f"Status bar update failed: {e}")

                    self._safe_call_ui(update_token_displays)
                except Exception as e:
                    logger.debug(f"Token event processing failed: {e}")
                return  # Don't add a chat message for token events
            else:
                # Default for other events (including "answer")
                return  # Don't show answer here, it will be displayed normally

            # Add message to chat view (thread-safe)
            self._safe_call_ui(lambda: chat_view.add_message(msg))

        except Exception as e:
            # Don't crash on streaming errors
            logger.warning(f"Streaming callback error: {e}")

    # Autocomplete message handlers
    def on_input_widget_autocomplete_request(
        self, event: InputWidget.AutocompleteRequest
    ) -> None:
        """Handle request to show/update autocomplete popup."""
        popup = self.query_one("#autocomplete-popup", AutocompletePopup)
        popup.update_suggestions(event.suggestions)
        # Keep focus on input widget
        input_widget = self.query_one("#input-widget", InputWidget)
        input_widget.focus()

    def on_input_widget_autocomplete_navigate(
        self, event: InputWidget.AutocompleteNavigate
    ) -> None:
        """Handle autocomplete navigation (up/down)."""
        popup = self.query_one("#autocomplete-popup", AutocompletePopup)
        if event.direction == "up":
            popup.navigate_up()
        else:
            popup.navigate_down()

    def on_input_widget_autocomplete_select(
        self, event: InputWidget.AutocompleteSelect
    ) -> None:
        """Handle autocomplete selection."""
        popup = self.query_one("#autocomplete-popup", AutocompletePopup)
        command = popup.select_current()
        if command:
            input_widget = self.query_one("#input-widget", InputWidget)
            input_widget.set_text_from_autocomplete(command)
            popup.hide()
            input_widget.focus()

    def on_input_widget_autocomplete_dismiss(
        self, event: InputWidget.AutocompleteDismiss
    ) -> None:
        """Handle autocomplete dismiss."""
        popup = self.query_one("#autocomplete-popup", AutocompletePopup)
        popup.hide()
        # Refocus input widget
        input_widget = self.query_one("#input-widget", InputWidget)
        input_widget.focus()

    async def on_input_widget_message_submitted(self, event: InputWidget.MessageSubmitted):
        """Handle user message submission.

        Args:
            event: Message submitted event with content
        """
        content = event.content

        # REACT Loop: If processing, add to queue for mid-execution input
        if self.is_processing:
            # Add to queue for agent to process
            await self.user_input_queue.put(content)

            # Show feedback that input was queued
            chat_view = self.query_one("#chat-view", ChatView)
            from ..models import Message
            queued_msg = Message(
                role="user",
                content=f"{content}\n[dim](queued for agent)[/dim]",
                metadata={"queued": True}
            )
            chat_view.add_message(queued_msg)
            return

        # Check for shell mode toggle (!!) first (User Story 5: T125)
        if self.shell_parser.is_shell_mode_toggle(content):
            await self.toggle_shell_mode()
            return

        # Check for shell commands (! prefix) or shell mode active (User Story 5: T123)
        if self.shell_mode or self.shell_parser.is_shell_command(content):
            await self.handle_shell_command(content)
        # Check for slash commands
        elif content.startswith("/"):
            await self.handle_command(content)
        else:
            await self.handle_user_message(content)

    async def handle_command(self, command: str):
        """Handle slash commands.

        Args:
            command: Command string (e.g., "/exit", "/help")
        """
        chat_view = self.query_one("#chat-view", ChatView)

        if command == "/exit":
            # Save session and exit
            if self.session:
                self.session_manager.save_session(self.session)
                from ..models import Message

                goodbye_msg = Message(
                    role="system", content=f"Session '{self.session.name}' saved. Goodbye!"
                )
                chat_view.add_message(goodbye_msg)
            self.exit()

        elif command.startswith("/help"):
            # Topic-specific help (T154-T156)
            parts = command.split()
            topic = parts[1] if len(parts) > 1 else None
            help_content = self._get_help_text(topic)

            from ..models import Message

            help_msg = Message(role="system", content=help_content)
            chat_view.add_message(help_msg)

        elif command == "/session":
            from ..models import Message

            if self.session:
                session_msg = Message(
                    role="system",
                    content=(
                        f"Session Information:\n"
                        f"  Name: {self.session.name}\n"
                        f"  ID: {self.session.id}\n"
                        f"  State: {self.session.state}\n"
                        f"  Working Directory: {self.session.working_directory}\n"
                        f"  Messages: {len(self.session.messages)}\n"
                        f"  Active Agent: {self.session.active_agent}\n"
                        f"  Default Model: {self.session.default_model}"
                    ),
                )
                chat_view.add_message(session_msg)
            else:
                error_msg = Message(
                    role="system",
                    content="No active session"
                )
                chat_view.add_message(error_msg)

        elif command.startswith("/cache"):
            # Handle cache commands (T-cache: Python pycache clearing)
            from ..models import Message
            import shutil

            parts = command.split()
            subcommand = parts[1] if len(parts) > 1 else "help"

            if subcommand == "clear":
                # Clear Python __pycache__ directories
                import os
                from pathlib import Path

                # Get project root (where promptchain package is)
                project_root = Path(__file__).parent.parent.parent.parent
                cleared_count = 0

                for pycache_dir in project_root.rglob("__pycache__"):
                    try:
                        shutil.rmtree(pycache_dir)
                        cleared_count += 1
                    except Exception:
                        pass

                # Also clear .pyc files
                pyc_count = 0
                for pyc_file in project_root.rglob("*.pyc"):
                    try:
                        pyc_file.unlink()
                        pyc_count += 1
                    except Exception:
                        pass

                cache_msg = Message(
                    role="system",
                    content=(
                        f"Cache cleared!\n"
                        f"  - Removed {cleared_count} __pycache__ directories\n"
                        f"  - Removed {pyc_count} .pyc files\n\n"
                        f"Tip: Run with PYTHONDONTWRITEBYTECODE=1 to prevent caching"
                    ),
                )
                chat_view.add_message(cache_msg)
            else:
                help_msg = Message(
                    role="system",
                    content=(
                        "Cache Commands:\n"
                        "  /cache clear  - Clear Python __pycache__ directories\n\n"
                        "Tip: Set PYTHONDONTWRITEBYTECODE=1 to prevent bytecode caching"
                    ),
                )
                chat_view.add_message(help_msg)

        elif command.startswith("/mcp"):
            # Handle MCP server commands
            from ..models import Message
            from ..models.mcp_config import MCPServerConfig

            parts = command.split()
            subcommand = parts[1] if len(parts) > 1 else "help"

            # Pre-defined MCP servers for easy setup
            PRESET_MCP_SERVERS = {
                "gemini": {
                    "id": "gemini",
                    "type": "stdio",
                    "command": "uv",
                    "args": ["run", "--directory", "/home/gyasis/Documents/code/gemini-mcp", "fastmcp", "run"],
                    "description": "Gemini AI with web search (gemini_research), code review, brainstorming"
                },
                "playwright": {
                    "id": "playwright",
                    "type": "stdio",
                    "command": "npx",
                    "args": ["@anthropic/mcp-playwright"],
                    "description": "Browser automation and web scraping"
                },
                "filesystem": {
                    "id": "filesystem",
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@anthropic/mcp-filesystem", "."],
                    "description": "File system operations"
                }
            }

            if subcommand == "add":
                # /mcp add <preset-name> OR /mcp add <id> <command> [args...]
                if len(parts) < 3:
                    error_msg = Message(
                        role="system",
                        content="Usage: /mcp add <preset> OR /mcp add <id> <command> [args...]\n"
                                f"Available presets: {', '.join(PRESET_MCP_SERVERS.keys())}"
                    )
                    chat_view.add_message(error_msg)
                else:
                    preset_or_id = parts[2]

                    if preset_or_id in PRESET_MCP_SERVERS:
                        # Use preset
                        preset = PRESET_MCP_SERVERS[preset_or_id]
                        server = MCPServerConfig(
                            id=preset["id"],
                            type=preset["type"],
                            command=preset["command"],
                            args=preset["args"],
                            auto_connect=True
                        )
                    elif len(parts) >= 4:
                        # Custom server: /mcp add <id> <command> [args...]
                        server = MCPServerConfig(
                            id=preset_or_id,
                            type="stdio",
                            command=parts[3],
                            args=parts[4:] if len(parts) > 4 else [],
                            auto_connect=True
                        )
                    else:
                        error_msg = Message(
                            role="system",
                            content=f"Unknown preset '{preset_or_id}'. Use a preset or provide: /mcp add <id> <command> [args...]"
                        )
                        chat_view.add_message(error_msg)
                        return

                    # Check if already exists
                    existing = [s for s in self.session.mcp_servers if s.id == server.id]
                    if existing:
                        error_msg = Message(
                            role="system",
                            content=f"MCP server '{server.id}' already exists. Use /mcp connect {server.id} to connect."
                        )
                        chat_view.add_message(error_msg)
                    else:
                        self.session.mcp_servers.append(server)
                        add_msg = Message(
                            role="system",
                            content=f"Added MCP server '{server.id}'\n"
                                    f"  Command: {server.command} {' '.join(server.args)}\n\n"
                                    f"Use /mcp connect {server.id} to connect and discover tools."
                        )
                        chat_view.add_message(add_msg)

            elif subcommand == "connect":
                # /mcp connect <server-id>
                if len(parts) < 3:
                    error_msg = Message(
                        role="system",
                        content="Usage: /mcp connect <server-id>"
                    )
                    chat_view.add_message(error_msg)
                else:
                    server_id = parts[2]
                    server = next((s for s in self.session.mcp_servers if s.id == server_id), None)

                    if not server:
                        error_msg = Message(
                            role="system",
                            content=f"MCP server '{server_id}' not found. Use /mcp add to add it first."
                        )
                        chat_view.add_message(error_msg)
                    else:
                        connect_msg = Message(
                            role="system",
                            content=f"Connecting to '{server_id}'..."
                        )
                        chat_view.add_message(connect_msg)

                        try:
                            from ..utils.mcp_manager import MCPManager
                            mcp_manager = MCPManager(self.session)
                            success = await mcp_manager.connect_server(server_id)

                            if success:
                                tools = server.discovered_tools
                                success_msg = Message(
                                    role="system",
                                    content=f"Connected to '{server_id}'!\n"
                                            f"Discovered {len(tools)} tools:\n" +
                                            "\n".join(f"  - {t}" for t in tools[:10]) +
                                            (f"\n  ... and {len(tools) - 10} more" if len(tools) > 10 else "")
                                )
                                chat_view.add_message(success_msg)
                            else:
                                error_msg = Message(
                                    role="system",
                                    content=f"Failed to connect to '{server_id}': {server.error_message}"
                                )
                                chat_view.add_message(error_msg)
                        except Exception as e:
                            error_msg = Message(
                                role="system",
                                content=f"Error connecting to '{server_id}': {str(e)}"
                            )
                            chat_view.add_message(error_msg)

            elif subcommand == "list":
                # /mcp list
                if not self.session.mcp_servers:
                    list_msg = Message(
                        role="system",
                        content="No MCP servers configured.\n\n"
                                "Add one with:\n"
                                "  /mcp add gemini     - Gemini AI with web search\n"
                                "  /mcp add playwright - Browser automation\n"
                                "  /mcp add filesystem - File operations"
                    )
                else:
                    lines = ["MCP Servers:\n"]
                    for server in self.session.mcp_servers:
                        status = {"connected": "✓", "disconnected": "○", "error": "✗"}.get(server.state, "?")
                        tool_count = len(server.discovered_tools)
                        lines.append(f"  {status} {server.id} - {server.state}")
                        if tool_count > 0:
                            lines.append(f"      {tool_count} tools available")
                    list_msg = Message(role="system", content="\n".join(lines))
                chat_view.add_message(list_msg)

            else:
                # Show help
                help_msg = Message(
                    role="system",
                    content=(
                        "MCP Server Commands:\n\n"
                        "  /mcp add <preset>     - Add preset server (gemini, playwright, filesystem)\n"
                        "  /mcp add <id> <cmd>   - Add custom server\n"
                        "  /mcp connect <id>     - Connect to server and discover tools\n"
                        "  /mcp list             - List configured servers\n\n"
                        "Example:\n"
                        "  /mcp add gemini       - Add Gemini MCP with web search\n"
                        "  /mcp connect gemini   - Connect and discover tools"
                    ),
                )
                chat_view.add_message(help_msg)

        elif command.startswith("/agent"):
            # Handle agent commands
            from ..models import Message

            parts = command.split()
            subcommand = parts[1] if len(parts) > 1 else "info"

            if subcommand == "info" or len(parts) == 1:
                # Show current agent info
                if self.session:
                    active = self.session.active_agent or "default"
                    model = self.session.default_model
                    agent_count = len(self.session.agents) + 1  # +1 for default
                    agent_msg = Message(
                        role="system",
                        content=(
                            f"[bold]Active Agent:[/bold] {active}\n"
                            f"[bold]Model:[/bold] {model}\n"
                            f"[bold]Total Agents:[/bold] {agent_count}"
                        ),
                    )
                    chat_view.add_message(agent_msg)
                else:
                    chat_view.add_message(Message(role="system", content="No active session"))

            elif subcommand == "list":
                # List all agents
                if self.session:
                    lines = ["[bold]Available Agents:[/bold]\n"]
                    # Default agent
                    active_marker = " [green](active)[/green]" if not self.session.active_agent else ""
                    lines.append(f"  [bold]default[/bold] - {self.session.default_model}{active_marker}")
                    # Custom agents
                    for name, agent in self.session.agents.items():
                        model = agent.get("model", "unknown") if isinstance(agent, dict) else getattr(agent, "model", "unknown")
                        active_marker = " [green](active)[/green]" if self.session.active_agent == name else ""
                        lines.append(f"  [bold]{name}[/bold] - {model}{active_marker}")
                    chat_view.add_message(Message(role="system", content="\n".join(lines)))
                else:
                    chat_view.add_message(Message(role="system", content="No active session"))

            elif subcommand == "create":
                # Create new agent: /agent create <name> --model <model>
                if len(parts) < 3:
                    chat_view.add_message(Message(
                        role="system",
                        content="Usage: /agent create <name> --model <model>\nExample: /agent create coder --model openai/gpt-4"
                    ))
                else:
                    agent_name = parts[2]
                    # Parse --model flag
                    model = self.session.default_model if self.session else "openai/gpt-4.1-mini-2025-04-14"
                    for i, p in enumerate(parts):
                        if p == "--model" and i + 1 < len(parts):
                            model = parts[i + 1]
                            break

                    if self.session:
                        self.session.agents[agent_name] = {"model": model, "description": ""}
                        chat_view.add_message(Message(
                            role="system",
                            content=f"Created agent '[bold]{agent_name}[/bold]' with model {model}"
                        ))
                    else:
                        chat_view.add_message(Message(role="system", content="No active session"))

            elif subcommand == "use":
                # Switch to agent: /agent use <name>
                if len(parts) < 3:
                    chat_view.add_message(Message(
                        role="system",
                        content="Usage: /agent use <name>\nExample: /agent use coder"
                    ))
                else:
                    agent_name = parts[2]
                    if self.session:
                        if agent_name == "default":
                            self.session.active_agent = None
                            chat_view.add_message(Message(
                                role="system",
                                content="Switched to [bold]default[/bold] agent"
                            ))
                        elif agent_name in self.session.agents:
                            self.session.active_agent = agent_name
                            chat_view.add_message(Message(
                                role="system",
                                content=f"Switched to agent '[bold]{agent_name}[/bold]'"
                            ))
                        else:
                            chat_view.add_message(Message(
                                role="system",
                                content=f"Agent '{agent_name}' not found. Use /agent list to see available agents."
                            ))
                    else:
                        chat_view.add_message(Message(role="system", content="No active session"))

            elif subcommand == "delete":
                # Delete agent: /agent delete <name>
                if len(parts) < 3:
                    chat_view.add_message(Message(
                        role="system",
                        content="Usage: /agent delete <name>"
                    ))
                else:
                    agent_name = parts[2]
                    if self.session:
                        if agent_name == "default":
                            chat_view.add_message(Message(
                                role="system",
                                content="Cannot delete the default agent"
                            ))
                        elif agent_name in self.session.agents:
                            del self.session.agents[agent_name]
                            if self.session.active_agent == agent_name:
                                self.session.active_agent = None
                            chat_view.add_message(Message(
                                role="system",
                                content=f"Deleted agent '[bold]{agent_name}[/bold]'"
                            ))
                        else:
                            chat_view.add_message(Message(
                                role="system",
                                content=f"Agent '{agent_name}' not found"
                            ))
                    else:
                        chat_view.add_message(Message(role="system", content="No active session"))

            else:
                # Unknown subcommand - show help
                chat_view.add_message(Message(
                    role="system",
                    content=(
                        "[bold]Agent Commands:[/bold]\n"
                        "  /agent            - Show current agent info\n"
                        "  /agent list       - List all agents\n"
                        "  /agent create <name> --model <model> - Create new agent\n"
                        "  /agent use <name> - Switch to agent\n"
                        "  /agent delete <name> - Delete agent"
                    )
                ))

        elif command.startswith("/mentalmodel"):
            # Handle mental model command
            from ..models import Message

            if self.session:
                result = self.command_handler.handle_mentalmodel(self.session)
                msg = Message(role="system", content=result.message)
                chat_view.add_message(msg)
            else:
                chat_view.add_message(Message(role="system", content="No active session"))

        elif command.startswith("/branch"):
            await self._handle_branch_pattern(command)

        elif command.startswith("/expand"):
            await self._handle_expand_pattern(command)

        elif command.startswith("/multihop"):
            await self._handle_multihop_pattern(command)

        elif command.startswith("/hybrid"):
            await self._handle_hybrid_pattern(command)

        elif command.startswith("/sharded"):
            await self._handle_sharded_pattern(command)

        elif command.startswith("/speculate"):
            await self._handle_speculate_pattern(command)

        elif command == "/patterns":
            await self._handle_patterns_help()

        else:
            from ..models import Message

            error_msg = Message(
                role="system",
                content=f"Unknown command: {command}\nType /help for available commands.",
            )
            chat_view.add_message(error_msg)

    def _parse_pattern_command(self, command: str) -> dict:
        """Parse pattern command into query and options.

        Handles syntax like:
            /branch "query text" --count=3 --mode=hybrid
            /expand "query" --strategies=semantic,synonym --max=5

        Returns:
            Dict with 'query', 'options', 'error' keys
        """
        import shlex

        parts = command.split(None, 1)  # Split command from args
        if len(parts) < 2:
            return {
                "query": None,
                "options": {},
                "error": "Missing query. Usage: /pattern \"query\" [options]",
            }

        args_str = parts[1]

        # Extract quoted query
        if not (args_str.startswith('"') or args_str.startswith("'")):
            return {
                "query": None,
                "options": {},
                "error": 'Query must be quoted. Usage: /pattern "query" [options]',
            }

        try:
            # Use shlex to handle quoted strings and flags
            parsed = shlex.split(args_str)
            query = parsed[0]

            # Parse --key=value flags
            options = {}
            for arg in parsed[1:]:
                if arg.startswith("--"):
                    if "=" in arg:
                        key, value = arg[2:].split("=", 1)
                        # Convert numeric values
                        try:
                            if "." in value:
                                options[key] = float(value)
                            else:
                                options[key] = int(value)
                        except ValueError:
                            # Handle comma-separated lists
                            if "," in value:
                                options[key] = value.split(",")
                            else:
                                options[key] = value
                    else:
                        options[arg[2:]] = True

            return {"query": query, "options": options, "error": None}
        except Exception as e:
            return {"query": None, "options": {}, "error": f"Parse error: {str(e)}"}

    async def _handle_branch_pattern(self, command: str):
        """Handle /branch pattern command."""
        from ..models import Message
        from promptchain.patterns.executors import (
            execute_branch,
            PatternNotAvailableError,
        )

        chat_view = self.query_one("#chat-view", ChatView)

        # Parse command
        parsed = self._parse_pattern_command(command)
        if parsed["error"]:
            error_msg = Message(role="system", content=f"Error: {parsed['error']}")
            chat_view.add_message(error_msg)
            return

        query = parsed["query"]
        options = parsed["options"]

        # Show processing message
        processing_msg = Message(
            role="system", content=f"🔍 Generating branching hypotheses for: {query}"
        )
        chat_view.add_message(processing_msg)

        try:
            # Execute pattern with session MessageBus/Blackboard if available
            message_bus = (
                getattr(self.session, "message_bus", None) if self.session else None
            )
            blackboard = (
                getattr(self.session, "blackboard", None) if self.session else None
            )

            result = await execute_branch(
                query=query,
                count=options.get("count", 3),
                mode=options.get("mode", "hybrid"),
                deeplake_path=options.get("deeplake_path"),
                verbose=options.get("verbose", False),
                message_bus=message_bus,
                blackboard=blackboard,
            )

            # Format results
            if result["success"]:
                hypotheses = result.get("hypotheses", [])
                content_lines = [f"✅ Generated {len(hypotheses)} hypotheses:\n"]
                for i, hyp in enumerate(hypotheses, 1):
                    content_lines.append(f"{i}. {hyp}")
                content_lines.append(
                    f"\n⏱️ Execution time: {result.get('execution_time_ms', 0):.0f}ms"
                )

                result_msg = Message(role="assistant", content="\n".join(content_lines))
                chat_view.add_message(result_msg)

                # Add to session history
                if self.session:
                    self.session.messages.append(Message(role="user", content=command))
                    self.session.messages.append(result_msg)
            else:
                error_msg = Message(
                    role="system",
                    content=f"❌ Error: {result.get('error', 'Unknown error')}",
                )
                chat_view.add_message(error_msg)

        except PatternNotAvailableError as e:
            error_msg = Message(
                role="system",
                content=f"❌ Pattern not available: {str(e)}\n\nInstall with: pip install promptchain[hybridrag]",
            )
            chat_view.add_message(error_msg)
        except Exception as e:
            error_msg = Message(
                role="system", content=f"❌ Error executing pattern: {str(e)}"
            )
            chat_view.add_message(error_msg)

    async def _handle_expand_pattern(self, command: str):
        """Handle /expand pattern command."""
        from ..models import Message
        from promptchain.patterns.executors import (
            execute_expand,
            PatternNotAvailableError,
        )

        chat_view = self.query_one("#chat-view", ChatView)

        # Parse command
        parsed = self._parse_pattern_command(command)
        if parsed["error"]:
            error_msg = Message(role="system", content=f"Error: {parsed['error']}")
            chat_view.add_message(error_msg)
            return

        query = parsed["query"]
        options = parsed["options"]

        # Show processing message
        processing_msg = Message(
            role="system", content=f"🔄 Expanding query: {query}"
        )
        chat_view.add_message(processing_msg)

        try:
            # Execute pattern with session MessageBus/Blackboard if available
            message_bus = (
                getattr(self.session, "message_bus", None) if self.session else None
            )
            blackboard = (
                getattr(self.session, "blackboard", None) if self.session else None
            )

            result = await execute_expand(
                query=query,
                strategies=options.get("strategies", ["semantic", "synonym"]),
                max_expansions=options.get("max", 5),
                verbose=options.get("verbose", False),
                message_bus=message_bus,
                blackboard=blackboard,
            )

            # Format results
            if result["success"]:
                expansions = result.get("expansions", [])
                content_lines = [f"✅ Generated {len(expansions)} query expansions:\n"]
                for i, exp in enumerate(expansions, 1):
                    content_lines.append(f"{i}. {exp}")
                content_lines.append(
                    f"\n⏱️ Execution time: {result.get('execution_time_ms', 0):.0f}ms"
                )

                result_msg = Message(role="assistant", content="\n".join(content_lines))
                chat_view.add_message(result_msg)

                # Add to session history
                if self.session:
                    self.session.messages.append(Message(role="user", content=command))
                    self.session.messages.append(result_msg)
            else:
                error_msg = Message(
                    role="system",
                    content=f"❌ Error: {result.get('error', 'Unknown error')}",
                )
                chat_view.add_message(error_msg)

        except PatternNotAvailableError as e:
            error_msg = Message(
                role="system",
                content=f"❌ Pattern not available: {str(e)}\n\nInstall with: pip install promptchain[hybridrag]",
            )
            chat_view.add_message(error_msg)
        except Exception as e:
            error_msg = Message(
                role="system", content=f"❌ Error executing pattern: {str(e)}"
            )
            chat_view.add_message(error_msg)

    async def _handle_multihop_pattern(self, command: str):
        """Handle /multihop pattern command."""
        from ..models import Message
        from promptchain.patterns.executors import (
            execute_multihop,
            PatternNotAvailableError,
        )

        chat_view = self.query_one("#chat-view", ChatView)

        # Parse command
        parsed = self._parse_pattern_command(command)
        if parsed["error"]:
            error_msg = Message(role="system", content=f"Error: {parsed['error']}")
            chat_view.add_message(error_msg)
            return

        query = parsed["query"]
        options = parsed["options"]

        # Show processing message
        processing_msg = Message(
            role="system", content=f"🔗 Running multi-hop retrieval for: {query}"
        )
        chat_view.add_message(processing_msg)

        try:
            # Execute pattern with session MessageBus/Blackboard if available
            message_bus = (
                getattr(self.session, "message_bus", None) if self.session else None
            )
            blackboard = (
                getattr(self.session, "blackboard", None) if self.session else None
            )

            result = await execute_multihop(
                query=query,
                max_hops=options.get("max_hops", 3),
                mode=options.get("mode", "hybrid"),
                deeplake_path=options.get("deeplake_path"),
                verbose=options.get("verbose", False),
                message_bus=message_bus,
                blackboard=blackboard,
            )

            # Format results
            if result["success"]:
                documents = result.get("documents", result.get("results", []))
                content_lines = [f"✅ Retrieved {len(documents)} documents:\n"]
                for i, doc in enumerate(documents[:10], 1):  # Show first 10
                    doc_preview = str(doc)[:100] + "..." if len(str(doc)) > 100 else str(doc)
                    content_lines.append(f"{i}. {doc_preview}")
                if len(documents) > 10:
                    content_lines.append(f"\n... and {len(documents) - 10} more")
                content_lines.append(
                    f"\n⏱️ Execution time: {result.get('execution_time_ms', 0):.0f}ms"
                )

                result_msg = Message(role="assistant", content="\n".join(content_lines))
                chat_view.add_message(result_msg)

                # Add to session history
                if self.session:
                    self.session.messages.append(Message(role="user", content=command))
                    self.session.messages.append(result_msg)
            else:
                error_msg = Message(
                    role="system",
                    content=f"❌ Error: {result.get('error', 'Unknown error')}",
                )
                chat_view.add_message(error_msg)

        except PatternNotAvailableError as e:
            error_msg = Message(
                role="system",
                content=f"❌ Pattern not available: {str(e)}\n\nInstall with: pip install promptchain[hybridrag]",
            )
            chat_view.add_message(error_msg)
        except Exception as e:
            error_msg = Message(
                role="system", content=f"❌ Error executing pattern: {str(e)}"
            )
            chat_view.add_message(error_msg)

    async def _handle_hybrid_pattern(self, command: str):
        """Handle /hybrid pattern command."""
        from ..models import Message
        from promptchain.patterns.executors import (
            execute_hybrid,
            PatternNotAvailableError,
        )

        chat_view = self.query_one("#chat-view", ChatView)

        # Parse command
        parsed = self._parse_pattern_command(command)
        if parsed["error"]:
            error_msg = Message(role="system", content=f"Error: {parsed['error']}")
            chat_view.add_message(error_msg)
            return

        query = parsed["query"]
        options = parsed["options"]

        # Show processing message
        processing_msg = Message(
            role="system", content=f"🔀 Running hybrid search for: {query}"
        )
        chat_view.add_message(processing_msg)

        try:
            # Execute pattern with session MessageBus/Blackboard if available
            message_bus = (
                getattr(self.session, "message_bus", None) if self.session else None
            )
            blackboard = (
                getattr(self.session, "blackboard", None) if self.session else None
            )

            result = await execute_hybrid(
                query=query,
                fusion_method=options.get("fusion", "rrf"),
                top_k=options.get("top_k", 10),
                deeplake_path=options.get("deeplake_path"),
                verbose=options.get("verbose", False),
                message_bus=message_bus,
                blackboard=blackboard,
            )

            # Format results
            if result["success"]:
                results_data = result.get("results", [])
                content_lines = [f"✅ Found {len(results_data)} results:\n"]
                for i, res in enumerate(results_data[:10], 1):  # Show first 10
                    res_preview = str(res)[:100] + "..." if len(str(res)) > 100 else str(res)
                    content_lines.append(f"{i}. {res_preview}")
                if len(results_data) > 10:
                    content_lines.append(f"\n... and {len(results_data) - 10} more")
                content_lines.append(
                    f"\n⏱️ Execution time: {result.get('execution_time_ms', 0):.0f}ms"
                )

                result_msg = Message(role="assistant", content="\n".join(content_lines))
                chat_view.add_message(result_msg)

                # Add to session history
                if self.session:
                    self.session.messages.append(Message(role="user", content=command))
                    self.session.messages.append(result_msg)
            else:
                error_msg = Message(
                    role="system",
                    content=f"❌ Error: {result.get('error', 'Unknown error')}",
                )
                chat_view.add_message(error_msg)

        except PatternNotAvailableError as e:
            error_msg = Message(
                role="system",
                content=f"❌ Pattern not available: {str(e)}\n\nInstall with: pip install promptchain[hybridrag]",
            )
            chat_view.add_message(error_msg)
        except Exception as e:
            error_msg = Message(
                role="system", content=f"❌ Error executing pattern: {str(e)}"
            )
            chat_view.add_message(error_msg)

    async def _handle_sharded_pattern(self, command: str):
        """Handle /sharded pattern command."""
        from ..models import Message
        from promptchain.patterns.executors import (
            execute_sharded,
            PatternNotAvailableError,
        )

        chat_view = self.query_one("#chat-view", ChatView)

        # Parse command
        parsed = self._parse_pattern_command(command)
        if parsed["error"]:
            error_msg = Message(role="system", content=f"Error: {parsed['error']}")
            chat_view.add_message(error_msg)
            return

        query = parsed["query"]
        options = parsed["options"]

        # Validate required shards parameter
        if "shards" not in options:
            error_msg = Message(
                role="system",
                content="Error: --shards parameter is required.\nUsage: /sharded \"query\" --shards=shard1,shard2",
            )
            chat_view.add_message(error_msg)
            return

        # Show processing message
        shards = options["shards"] if isinstance(options["shards"], list) else [options["shards"]]
        processing_msg = Message(
            role="system",
            content=f"🗂️  Searching across {len(shards)} shards for: {query}",
        )
        chat_view.add_message(processing_msg)

        try:
            # Execute pattern with session MessageBus/Blackboard if available
            message_bus = (
                getattr(self.session, "message_bus", None) if self.session else None
            )
            blackboard = (
                getattr(self.session, "blackboard", None) if self.session else None
            )

            result = await execute_sharded(
                query=query,
                shard_paths=shards,
                aggregation_method=options.get("aggregation", "rrf"),
                top_k=options.get("top_k", 10),
                verbose=options.get("verbose", False),
                message_bus=message_bus,
                blackboard=blackboard,
            )

            # Format results
            if result["success"]:
                results_data = result.get("results", [])
                content_lines = [f"✅ Found {len(results_data)} results across shards:\n"]
                for i, res in enumerate(results_data[:10], 1):  # Show first 10
                    res_preview = str(res)[:100] + "..." if len(str(res)) > 100 else str(res)
                    content_lines.append(f"{i}. {res_preview}")
                if len(results_data) > 10:
                    content_lines.append(f"\n... and {len(results_data) - 10} more")
                content_lines.append(
                    f"\n⏱️ Execution time: {result.get('execution_time_ms', 0):.0f}ms"
                )

                result_msg = Message(role="assistant", content="\n".join(content_lines))
                chat_view.add_message(result_msg)

                # Add to session history
                if self.session:
                    self.session.messages.append(Message(role="user", content=command))
                    self.session.messages.append(result_msg)
            else:
                error_msg = Message(
                    role="system",
                    content=f"❌ Error: {result.get('error', 'Unknown error')}",
                )
                chat_view.add_message(error_msg)

        except PatternNotAvailableError as e:
            error_msg = Message(
                role="system",
                content=f"❌ Pattern not available: {str(e)}\n\nInstall with: pip install promptchain[hybridrag]",
            )
            chat_view.add_message(error_msg)
        except Exception as e:
            error_msg = Message(
                role="system", content=f"❌ Error executing pattern: {str(e)}"
            )
            chat_view.add_message(error_msg)

    async def _handle_speculate_pattern(self, command: str):
        """Handle /speculate pattern command."""
        from ..models import Message
        from promptchain.patterns.executors import (
            execute_speculate,
            PatternNotAvailableError,
        )

        chat_view = self.query_one("#chat-view", ChatView)

        # Parse command
        parsed = self._parse_pattern_command(command)
        if parsed["error"]:
            error_msg = Message(role="system", content=f"Error: {parsed['error']}")
            chat_view.add_message(error_msg)
            return

        context = parsed["query"]  # For speculate, query is actually context
        options = parsed["options"]

        # Show processing message
        processing_msg = Message(
            role="system", content=f"⚡ Running speculative execution..."
        )
        chat_view.add_message(processing_msg)

        try:
            # Execute pattern with session MessageBus/Blackboard if available
            message_bus = (
                getattr(self.session, "message_bus", None) if self.session else None
            )
            blackboard = (
                getattr(self.session, "blackboard", None) if self.session else None
            )

            result = await execute_speculate(
                context=context,
                min_confidence=options.get("min_confidence", 0.7),
                prefetch_count=options.get("prefetch", 3),
                deeplake_path=options.get("deeplake_path"),
                verbose=options.get("verbose", False),
                message_bus=message_bus,
                blackboard=blackboard,
            )

            # Format results
            if result["success"]:
                predictions = result.get("predictions", result.get("results", []))
                content_lines = [f"✅ Generated {len(predictions)} speculative predictions:\n"]
                for i, pred in enumerate(predictions, 1):
                    pred_preview = str(pred)[:100] + "..." if len(str(pred)) > 100 else str(pred)
                    content_lines.append(f"{i}. {pred_preview}")
                content_lines.append(
                    f"\n⏱️ Execution time: {result.get('execution_time_ms', 0):.0f}ms"
                )

                result_msg = Message(role="assistant", content="\n".join(content_lines))
                chat_view.add_message(result_msg)

                # Add to session history
                if self.session:
                    self.session.messages.append(Message(role="user", content=command))
                    self.session.messages.append(result_msg)
            else:
                error_msg = Message(
                    role="system",
                    content=f"❌ Error: {result.get('error', 'Unknown error')}",
                )
                chat_view.add_message(error_msg)

        except PatternNotAvailableError as e:
            error_msg = Message(
                role="system",
                content=f"❌ Pattern not available: {str(e)}\n\nInstall with: pip install promptchain[hybridrag]",
            )
            chat_view.add_message(error_msg)
        except Exception as e:
            error_msg = Message(
                role="system", content=f"❌ Error executing pattern: {str(e)}"
            )
            chat_view.add_message(error_msg)

    async def _handle_patterns_help(self):
        """Show pattern commands help."""
        from ..models import Message

        chat_view = self.query_one("#chat-view", ChatView)

        help_content = """Pattern Commands:

🌳 /branch "query" [--count=N] [--mode=local|global|hybrid]
   Generate branching hypotheses for exploration

🔄 /expand "query" [--strategies=semantic,synonym] [--max=N]
   Expand query with variations

🔗 /multihop "query" [--max-hops=N] [--mode=hybrid]
   Multi-hop retrieval with reasoning chains

🔀 /hybrid "query" [--fusion=rrf|linear|borda] [--top-k=N]
   Hybrid search combining dense and sparse retrieval

🗂️  /sharded "query" --shards=shard1,shard2 [--aggregation=rrf]
   Search across multiple sharded indexes

⚡ /speculate "context" [--min-confidence=0.7] [--prefetch=N]
   Speculative execution with prefetching

📖 /patterns
   Show this help message

Examples:
  /branch "quantum computing applications" --count=5
  /expand "machine learning" --strategies=semantic,synonym
  /multihop "what caused the 2008 financial crisis" --max-hops=3
"""

        help_msg = Message(role="system", content=help_content)
        chat_view.add_message(help_msg)

    async def toggle_shell_mode(self):
        """Toggle shell mode on/off (User Story 5: T125).

        When shell mode is active:
        - Commands execute without ! prefix
        - Status bar shows SHELL MODE indicator
        - !! toggles it off
        """
        self.shell_mode = not self.shell_mode

        chat_view = self.query_one("#chat-view", ChatView)
        from ..models import Message

        if self.shell_mode:
            mode_msg = Message(
                role="system",
                content="🔧 Shell mode activated. All input will be executed as shell commands.\nType !! to exit shell mode.",
            )
            # Update status bar to show shell mode
            status_bar = self.query_one("#status-bar", StatusBar)
            # TODO: Add shell mode indicator to status bar in future enhancement
        else:
            mode_msg = Message(
                role="system",
                content="💬 Chat mode activated. Back to normal conversation.",
            )

        chat_view.add_message(mode_msg)

    async def handle_shell_command(self, content: str):
        """Handle shell command execution (User Story 5: T123-T124).

        Args:
            content: User input (with or without ! prefix)
        """
        # Extract command (remove ! if present)
        if self.shell_mode:
            # In shell mode, treat input as-is
            command = content
        else:
            # Extract command without ! prefix
            command = self.shell_parser.extract_command(content)

        # Display user command in chat
        chat_view = self.query_one("#chat-view", ChatView)
        from ..models import Message

        user_msg = Message(
            role="user",
            content=f"!{command}" if not content.startswith("!") else content,
        )
        chat_view.add_message(user_msg)

        # Show processing indicator with spinner
        processing_msg = Message(
            role="system",
            content=f"Executing: {command}",
            metadata={"is_processing": True},
        )
        chat_view.add_message(processing_msg)

        # Start spinner on the processing message
        if len(chat_view) > 0:
            last_item = chat_view.children[-1]
            if hasattr(last_item, "is_processing"):
                last_item.is_processing = True
                last_item.start_spinner()

        try:
            # Execute command with working directory from session
            if not self.session:
                return
            working_directory = str(self.session.working_directory)
            result = await self.shell_executor.execute_shell_command(
                command, working_directory=working_directory
            )

            # Stop spinner and remove processing message
            if len(chat_view) > 0:
                last_item = chat_view.children[-1]
                if hasattr(last_item, "is_processing") and last_item.is_processing:
                    last_item.stop_spinner()

            chat_view.messages.pop()  # Remove processing indicator
            if len(chat_view) > 0:
                chat_view.pop()

            # Format output using OutputFormatter for better readability
            formatted_output = OutputFormatter.format_shell_output(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.return_code,
                execution_time=result.execution_time,
                timed_out=result.timed_out,
                error_message=result.error_message,
            )

            output_msg = Message(role="system", content=formatted_output)
            chat_view.add_message(output_msg)

        except Exception as e:
            # Stop spinner and remove processing indicator
            if len(chat_view) > 0:
                last_item = chat_view.children[-1]
                if hasattr(last_item, "is_processing") and last_item.is_processing:
                    last_item.stop_spinner()

            if (
                chat_view.messages
                and chat_view.messages[-1].role == "system"
                and "Executing" in chat_view.messages[-1].content
            ):
                chat_view.messages.pop()
                if len(chat_view) > 0:
                    chat_view.pop()

            # Use global error handler for user-friendly messages (T141-T142)
            error_msg = self._handle_error(e, context="executing shell command")
            chat_view.add_message(error_msg)

    def _initialize_agent_chain(self):
        """Initialize PromptChain or AgentChain based on orchestration mode (T037, T058-T059).

        Two modes supported:
        1. Single-agent mode: Uses lazy-loaded PromptChain instances (T148)
        2. Multi-agent router mode: Uses AgentChain with intelligent routing (T037)

        Router mode activates when:
        - session.orchestration_config.execution_mode == "router"
        - Multiple agents configured in session

        NOTE: This is now a legacy initialization method. AgentChain is created
        lazily via _get_or_create_agent_chain() in T038.
        """
        if not self.session:
            return

        # Check if router mode should be enabled
        orchestration = self.session.orchestration_config
        num_agents = len(self.session.agents)
        use_router_mode = (
            orchestration
            and orchestration.execution_mode == "router"
            and num_agents > 1
        )

        if use_router_mode:
            # Multi-agent router mode (T037) - now handled by _get_or_create_agent_chain()
            pass
        else:
            # Single-agent mode with lazy loading (T148) - now handled by _get_or_create_agent_chain()
            pass

    def _build_history_configs(self) -> Dict[str, Dict[str, Any]]:
        """Build per-agent history configurations for AgentChain (T075).

        Extracts history_config from each agent in the session and converts
        to dict format for AgentChain's agent_history_configs parameter.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping of agent names to history config dicts
        """
        if not self.session:
            return {}

        history_configs = {}

        for agent_name, agent in self.session.agents.items():
            if agent.history_config is not None:
                # Convert HistoryConfig object to dict
                history_configs[agent_name] = agent.history_config.to_dict()

        return history_configs

    def _build_router_config(self) -> dict:
        """Build router configuration for AgentChain (T039).

        Returns router config dict with models, instructions, and decision prompt template.
        Uses session's orchestration_config.router_model if available, otherwise defaults
        to "openai/gpt-4o-mini" for fast, cost-effective routing.

        Returns:
            dict: Router configuration for AgentChain
        """
        if not self.session or not self.session.orchestration_config:
            # Fallback to default router config with task classification
            return {
                "models": ["openai/gpt-4o-mini"],
                "instructions": [None, "{input}"],
                "decision_prompt_templates": {
                    "single_agent_dispatch": """
TASK CLASSIFICATION AND ROUTING:

User request: {user_input}

STEP 1: Classify the request type:
- CONVERSATIONAL: Greetings (hello, hi, hey), self-identification (who/what are you, what can you do)
- SIMPLE_QUERY: Direct factual questions not requiring tools
- TASK_ORIENTED: Code changes, file operations, searches, analysis
- PATTERN_BASED: Complex reasoning needing multi-hop/branching analysis

STEP 2: Determine response approach:

For CONVERSATIONAL/SIMPLE_QUERY:
- Set refined_query to instruct agent to respond WITHOUT using any tools
- Example refined_query: "Respond conversationally without tools: [original query]"

For TASK_ORIENTED/PATTERN_BASED:
- Choose most appropriate agent from available agents
- Keep refined_query as original or clarify if needed

Available agents:
{agent_details}

Conversation history:
{history}

Return JSON:
{{"chosen_agent": "agent_name", "refined_query": "query with tool usage instructions if needed"}}

IMPORTANT: For conversational queries, ALWAYS prefix refined_query with "Respond conversationally without tools: "
                    """
                }
            }

        orchestration = self.session.orchestration_config
        router_config = orchestration.router_config

        # Use router model from config, or default to gpt-4o-mini
        router_model = router_config.model if router_config else "openai/gpt-4o-mini"

        # Use decision prompt template from config, or default with task classification
        decision_prompt = (
            router_config.decision_prompt_template
            if router_config and router_config.decision_prompt_template
            else """
TASK CLASSIFICATION AND ROUTING:

User request: {user_input}

STEP 1: Classify the request type:
- CONVERSATIONAL: Greetings (hello, hi, hey), self-identification (who/what are you, what can you do)
- SIMPLE_QUERY: Direct factual questions not requiring tools
- TASK_ORIENTED: Code changes, file operations, searches, analysis
- PATTERN_BASED: Complex reasoning needing multi-hop/branching analysis

STEP 2: Determine response approach:

For CONVERSATIONAL/SIMPLE_QUERY:
- Set refined_query to instruct agent to respond WITHOUT using any tools
- Example refined_query: "Respond conversationally without tools: [original query]"

For TASK_ORIENTED/PATTERN_BASED:
- Choose most appropriate agent from available agents
- Keep refined_query as original or clarify if needed

Available agents:
{agent_details}

Conversation history:
{history}

Return JSON:
{{"chosen_agent": "agent_name", "refined_query": "query with tool usage instructions if needed"}}

IMPORTANT: For conversational queries, ALWAYS prefix refined_query with "Respond conversationally without tools: "
            """
        )

        return {
            "models": [router_model],
            "instructions": [None, "{input}"],  # Pass-through instructions
            "decision_prompt_templates": {
                "single_agent_dispatch": decision_prompt
            }
        }

    def _get_or_create_agent_chain(self) -> Optional[AgentChain]:
        """Build or return cached AgentChain instance for current session (T038).

        This method creates a single unified AgentChain that handles both:
        - Router mode: Multiple agents with intelligent routing (when multiple agents exist)
        - Single-agent mode: Single agent wrapped in AgentChain for consistency

        Returns:
            AgentChain with router mode if multiple agents exist,
            AgentChain wrapping single PromptChain if only one agent,
            None if no agents configured
        """
        # Return existing AgentChain if already created
        if self.agent_chain is not None:
            return self.agent_chain

        if not self.session or not self.session.agents:
            return None

        # Build agents dict for AgentChain
        agents_dict = {}
        agent_descriptions = {}

        # Import global tool registry with all 19 registered tools
        from ..tools import registry

        # Get all tools as OpenAI function schemas
        all_tool_schemas = registry.get_openai_schemas()

        # Get all tool metadata for function registration
        all_tools = [registry.get(tool_name) for tool_name in registry.list_tools()]

        # Build MCP server configs from connected session servers
        mcp_server_configs = []
        if self.session.mcp_servers:
            for server in self.session.mcp_servers:
                if server.state == "connected":
                    mcp_config = {
                        "id": server.id,
                        "type": server.type,
                        "command": server.command,
                        "args": server.args,
                    }
                    if server.url:
                        mcp_config["url"] = server.url
                    mcp_server_configs.append(mcp_config)

        for agent_name, agent in self.session.agents.items():
            # Determine agent type: AgenticStepProcessor or simple PromptChain
            if agent.instruction_chain and len(agent.instruction_chain) > 0:
                # Import AgenticStepProcessor for complex reasoning agents
                from promptchain.utils.agentic_step_processor import AgenticStepProcessor

                objective = agent.instruction_chain[0] if isinstance(agent.instruction_chain[0], str) else str(agent.instruction_chain[0])

                # Determine history mode from agent's history_config
                history_mode = "progressive"  # Default for multi-hop reasoning
                if agent.history_config and not agent.history_config.enabled:
                    history_mode = "minimal"  # Terminal agents

                # Create PromptChain with tools using agentic config
                chain = PromptChain(
                    models=[{
                        "name": agent.model_name,
                        "params": {"max_completion_tokens": agent.max_completion_tokens}
                    }],
                    instructions=[
                        AgenticStepProcessor(
                            objective=objective,
                            max_internal_steps=self.config.agentic.default_max_internal_steps,
                            model_name=agent.model_name,
                            history_mode=history_mode or self.config.agentic.history_mode,
                            progress_callback=self._reasoning_progress_callback,  # T052: Real-time progress updates
                        )
                    ],
                    verbose=False,
                    mcp_servers=mcp_server_configs if mcp_server_configs else None,
                )

                # Add tool schemas FIRST so LLM knows what tools are available
                chain.add_tools(all_tool_schemas)

                # Then register the actual tool functions for execution
                for tool_meta in all_tools:
                    chain.register_tool_function(tool_meta.function)

                agents_dict[agent_name] = chain
            else:
                # Simple PromptChain for basic agents
                chain = PromptChain(
                    models=[{
                        "name": agent.model_name,
                        "params": {"max_completion_tokens": agent.max_completion_tokens}
                    }],
                    instructions=["{input}"],  # Simple pass-through
                    verbose=False,
                    mcp_servers=mcp_server_configs if mcp_server_configs else None,
                )

                # Add tool schemas FIRST, then register functions
                chain.add_tools(all_tool_schemas)
                for tool_meta in all_tools:
                    chain.register_tool_function(tool_meta.function)

                agents_dict[agent_name] = chain

            agent_descriptions[agent_name] = agent.description or f"{agent_name} agent"

        # Determine execution mode
        orchestration = self.session.orchestration_config
        num_agents = len(self.session.agents)

        if num_agents > 1 and orchestration and orchestration.execution_mode == "router":
            # Multi-agent router mode
            router_config = self._build_router_config()

            self.agent_chain = AgentChain(
                agents=agents_dict,
                agent_descriptions=agent_descriptions,
                execution_mode="router",
                router=router_config,
                default_agent=orchestration.default_agent,
                auto_include_history=orchestration.auto_include_history,
                agent_history_configs=self._build_history_configs(),
                verbose=False,
                activity_logger=self.session.activity_logger,  # ✅ FIX: Enable activity logging
            )
        else:
            # Single-agent mode: Wrap single agent in AgentChain for consistency
            active_agent_name = self.session.active_agent or list(agents_dict.keys())[0]

            self.agent_chain = AgentChain(
                agents=agents_dict,
                agent_descriptions=agent_descriptions,
                execution_mode="router",  # Use router mode even for single agent
                router=self._build_router_config(),
                default_agent=active_agent_name,
                auto_include_history=True,
                agent_history_configs=self._build_history_configs(),
                verbose=False,
                activity_logger=self.session.activity_logger,  # ✅ FIX: Enable activity logging
            )

        return self.agent_chain


    def add_agent_chain(self, agent_name: str, agent_model: str):
        """Add a new agent's PromptChain dynamically (T058).

        Called when a new agent is created via /agent create command.

        Args:
            agent_name: Name of the new agent
            agent_model: Model name for the agent
        """
        if not hasattr(self, "agent_chains"):
            self.agent_chains = {}

        self.agent_chains[agent_name] = PromptChain(
            models=[{
                "name": agent_model,
                "params": {"max_completion_tokens": 16000}
            }],
            instructions=["{input}"],
            verbose=False
        )

    def switch_active_agent(self, agent_name: str):
        """Switch to a different active agent (T059, T061).

        Updates the active agent reference to route messages correctly
        and updates the status bar display.

        Args:
            agent_name: Name of agent to switch to
        """
        if not self.session:
            return

        # Verify agent exists
        if agent_name not in self.session.agents:
            raise ValueError(f"Agent '{agent_name}' not found in session")

        # Update session active agent
        self.session.active_agent = agent_name

        # Reset agent_chain to force re-creation with new active agent
        self.agent_chain = None

        # Update status bar to show new active agent (T061)
        active_agent = self.session.agents.get(agent_name)
        if active_agent:
            status_bar = self.query_one("#status-bar", StatusBar)
            status_bar.update_session_info(
                active_agent=agent_name, model_name=active_agent.model_name
            )

    async def handle_user_message(self, content: str):
        """Handle user message (non-command) with AgentChain integration (T032-T034).

        Args:
            content: User message content
        """
        if not self.session:
            return  # Guard against None session

        # Reset processor call counter for new user message
        self.processor_call_count = 0
        self.last_step_number = 0
        self.processor_completed = False
        self.last_displayed_step = None

        # Inject file context for @syntax references (User Story 4: T096-T098)
        working_directory = Path(self.session.working_directory)
        content_with_files = self.file_context_manager.inject_file_context(
            message=content, working_directory=working_directory
        )

        # Add user message to session and chat view (use original content for display)
        self.session.add_message(role="user", content=content)

        chat_view = self.query_one("#chat-view", ChatView)
        chat_view.add_message(self.session.messages[-1])

        # Update status bar message count
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.update_session_info(message_count=len(self.session.messages))

        # T091: Check for active workflow and integrate with AgenticStepProcessor
        workflow = self.session_manager.load_workflow(self.session.id)
        if workflow and not workflow.is_completed:
            await self._handle_workflow_message(content_with_files, workflow)
            return

        try:
            # REACT Loop: Set processing flag to enable mid-execution input
            self.is_processing = True

            # Show processing indicator with animated spinner
            from ..models import Message

            active_agent_name = self.session.active_agent
            if not active_agent_name:
                self.is_processing = False  # Reset before early return
                return  # Guard against None agent name

            active_agent = self.session.agents.get(active_agent_name)
            model_name = active_agent.model_name if active_agent else self.session.default_model

            # Create a processing message with plain text (animation handled by MessageItem)
            processing_msg = Message(
                role="system",
                content=f"Processing with {active_agent_name} ({model_name})...",
                metadata={"is_processing": True},  # Flag for MessageItem to animate
            )
            chat_view.add_message(processing_msg)

            # Start spinner animation on the last message item
            if len(chat_view) > 0:
                last_item = chat_view.children[-1]
                if hasattr(last_item, "is_processing"):
                    last_item.is_processing = True
                    last_item.start_spinner()

            # Get formatted history for context (T034)
            history = self.session.history_manager.get_formatted_history(
                format_style="chat", max_tokens=6000  # Leave room for response
            )

            # Prepare input with history context (use content_with_files for LLM)
            if history:
                # Add history as context before the current message
                full_input = f"Previous conversation:\n{history}\n\nUser: {content_with_files}"
            else:
                full_input = content_with_files

            # Enable auto-refresh for activity logs during execution (Phase 5)
            if self.log_viewer and self.log_viewer_visible:
                self.log_viewer.enable_auto_refresh(interval=2.0)

            try:
                # T038-T040: Use unified AgentChain for both single-agent and router modes
                agent_chain = self._get_or_create_agent_chain()

                if not agent_chain:
                    # Handle no agents configured
                    error_msg = Message(
                        role="system",
                        content="[bold]Error: No agents configured for this session[/bold]"
                    )
                    chat_view.add_message(error_msg)
                    return

                # Execute via AgentChain (works for both single-agent and router modes)
                # T044: Router failure handling with fallback to default agent
                # REACT Loop (v0.4.3): Pass queue/callback for mid-execution interaction
                async def _generate_response():
                    return await agent_chain.run_chat_turn_async(
                        content_with_files,
                        user_input_queue=self.user_input_queue,
                        streaming_callback=self._streaming_callback
                    )

                try:
                    response_content = await self.error_handler.handle_with_retry(
                        _generate_response, f"generating response via AgentChain"
                    )
                except (TimeoutError, ValueError, KeyError, json.JSONDecodeError) as e:
                    # T044: Router failed - fallback to default agent
                    default_agent_name = (
                        self.session.orchestration_config.default_agent
                        if self.session.orchestration_config
                        else None
                    )

                    # If no valid default, use first available agent
                    if not default_agent_name or default_agent_name not in self.session.agents:
                        default_agent_name = list(self.session.agents.keys())[0]

                    # Display fallback notification
                    # Escape brackets in exception message to prevent Rich markup conflicts
                    safe_error = str(e).replace('[', '\\[').replace(']', '\\]')
                    fallback_msg = Message(
                        role="system",
                        content=f"[italic]Router failed ({type(e).__name__}: {safe_error}), using fallback: {default_agent_name}[/italic]"
                    )
                    chat_view.add_message(fallback_msg)

                    # Execute with fallback agent directly
                    fallback_chain = self.session.agents[default_agent_name]
                    response_content = await fallback_chain.process_prompt_async(content_with_files)

                    # Log router failure to JSONL (T044)
                    self.session_manager.log_router_failure(
                        session_id=self.session.id,
                        error_type=type(e).__name__,
                        reason=str(e),
                        user_query=message_text,
                        fallback_agent=default_agent_name
                    )

                    # Update active agent tracking for response formatting
                    active_agent_name = default_agent_name
                    selected_agent = self.session.agents.get(default_agent_name)
                    model_name = selected_agent.model_name if selected_agent else ""

                    # Update status bar to reflect fallback agent
                    status_bar = self.query_one("#status-bar", StatusBar)
                    status_bar.update_session_info(
                        active_agent=default_agent_name,
                        model_name=model_name,
                        last_agent_switch=default_agent_name,
                    )

                # T040: Track selected agent for router decision display
                # T043: Log router decision to JSONL
                if agent_chain.last_selected_agent:
                    selected_agent_name = agent_chain.last_selected_agent

                    # T043: Log router decision with all available agents
                    self.session_manager.log_router_decision(
                        session_id=self.session.id,
                        user_query=content_with_files,
                        selected_agent=selected_agent_name,
                        rationale=getattr(agent_chain, 'last_routing_rationale', None),
                        all_agents=list(self.session.agents.keys()),
                    )

                    # T042: Detect agent switch and display visual indicator
                    if self.last_displayed_agent and selected_agent_name != self.last_displayed_agent:
                        # Show switch notification in chat
                        switch_message = f"[dim]→ Router switched to agent: [bold]{selected_agent_name}[/bold][/dim]"
                        await self._display_system_message(switch_message)

                    # Update tracking for next message
                    self.last_displayed_agent = selected_agent_name

                    # Check if agent actually changed from active agent
                    if selected_agent_name != active_agent_name:
                        status_bar = self.query_one("#status-bar", StatusBar)
                        selected_agent = self.session.agents.get(selected_agent_name)
                        selected_model = selected_agent.model_name if selected_agent else ""

                        # T041: Update status bar with selected agent
                        status_bar.update_session_info(
                            active_agent=selected_agent_name,
                            model_name=selected_model,
                            last_agent_switch=selected_agent_name,
                            selected_agent=selected_agent_name,  # T041: Show router selection
                        )

                        # Update active_agent_name for response formatting
                        active_agent_name = selected_agent_name
                        model_name = selected_model
                    else:
                        # T041: Update status bar even if same agent (for router icon)
                        status_bar = self.query_one("#status-bar", StatusBar)
                        status_bar.update_session_info(
                            selected_agent=selected_agent_name,
                        )

                # T054: Check for agentic completion and handle exhaustion
                # Enhanced with TaskController for handoff and task status detection
                if agent_chain.last_selected_agent:
                    selected_agent = self.session.agents.get(agent_chain.last_selected_agent)
                    if selected_agent and selected_agent.instruction_chain:
                        # Extract objective from first instruction if it's a string
                        objective = ""
                        if selected_agent.instruction_chain:
                            first_instr = selected_agent.instruction_chain[0]
                            objective = first_instr if isinstance(first_instr, str) else str(first_instr)

                        # Get step count from processor if available
                        step_count = getattr(agent_chain, 'last_step_count', 1)
                        max_steps = self.config.agentic.default_max_internal_steps

                        # Use TaskController for enhanced status detection
                        task_status = self._check_task_status(response_content, step_count)

                        # Check completion status
                        completed = self._check_agentic_completion(response_content, objective)

                        # Handle different task statuses
                        if task_status == TaskStatus.REQUIRES_HANDOFF:
                            # Attempt handoff to another agent
                            handoff_target = self.task_controller.extract_handoff_target(response_content)
                            if handoff_target and handoff_target in self.session.agents:
                                # Display handoff notification
                                handoff_msg = Message(
                                    role="system",
                                    content=f"[dim]→ Handing off to agent:[/dim] [bold]{handoff_target}[/bold]"
                                )
                                chat_view.add_message(handoff_msg)

                                # Perform handoff
                                handoff_response = await self._handle_handoff(response_content)
                                if handoff_response:
                                    # Append handoff response to original response
                                    response_content = f"{response_content}\n\n---\n\n**[{handoff_target}]:**\n{handoff_response}"

                        elif task_status == TaskStatus.BLOCKED:
                            # Show user questions from TaskController
                            user_return_msg = self._get_user_return_message()
                            if user_return_msg:
                                chat_view.add_message(user_return_msg)

                        # Handle max steps exhaustion
                        if hasattr(agent_chain, 'last_processor_exhausted') and agent_chain.last_processor_exhausted:
                            # Log exhaustion event (T053, T055)
                            self.session_manager.log_agentic_exhaustion(
                                session_id=self.session.id,
                                agent_name=agent_chain.last_selected_agent,
                                objective=objective,
                                max_steps=max_steps,
                                steps_completed=step_count,
                                partial_result=response_content
                            )

                            # Also log to ExecutionHistoryManager
                            if hasattr(self.session, 'history_manager'):
                                self.session.history_manager.add_exhaustion_entry(
                                    objective=objective,
                                    max_steps=max_steps,
                                    steps_completed=step_count,
                                    partial_result=response_content,
                                    source="agentic_step_processor"
                                )

                            # Display warning to user with threshold info
                            from ..models import Message as MsgModel
                            warning_msg = MsgModel(
                                role="system",
                                content=(
                                    f"[italic]Reasoning reached max steps ({max_steps}) - objective may be incomplete[/italic]\n\n"
                                    "Suggestions:\n"
                                    "  - Increase max_internal_steps in agent config\n"
                                    "  - Simplify the objective\n"
                                    "  - Break into smaller sub-objectives"
                                )
                            )
                            chat_view.add_message(warning_msg)

                        # Visual completion feedback (T054)
                        if completed:
                            # Reset task controller for next task
                            self.task_controller.reset()

                            # Log successful completion
                            completion_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "event_type": "reasoning_completion",
                                "session_id": self.session.id,
                                "agent_name": agent_chain.last_selected_agent,
                                "objective": objective,
                                "completed": True,
                                "result_length": len(response_content)
                            }
                            # TODO: Implement _append_to_jsonl method in SessionManager
                            # self.session_manager._append_to_jsonl(self.session.id, completion_entry)

                # Create assistant message
                # Note: Don't use OutputFormatter.format_assistant_message() here
                # because chat_view.py already handles agent_name/model_name formatting
                # in its render() method with proper Rich Text styling
                response_msg = Message(
                    role="assistant",
                    content=response_content,  # Use raw content, not formatted
                    agent_name=active_agent_name,
                    model_name=model_name,
                )

                # Add to session and display (remove processing indicator by replacing it)
                self.session.add_message(
                    role="assistant",
                    content=response_content,  # Store original, not formatted
                    metadata={
                        "agent_name": response_msg.agent_name,
                        "model_name": response_msg.model_name,
                    },
                )

                # Refresh task list widget after agent response (may have updated tasks)
                self._refresh_task_list_widget()

                # T087: Update workflow state after agent response
                workflow_update = self.session_manager.update_workflow_on_message(
                    self.session, self.session.messages[-1]
                )

                # Display workflow progress update if step completed (T090)
                if workflow_update:
                    progress_msg = Message(
                        role="system",
                        content=(
                            f"[bold]Workflow step completed:[/bold] {workflow_update['step_description']}\n"
                            f"[dim]Workflow progress: {workflow_update['progress_percentage']:.0f}% "
                            f"({workflow_update['completed_count']}/{workflow_update['total_steps']} steps)[/dim]"
                        )
                    )
                    # Add progress message to chat (will be displayed after response)
                    self.session.add_message(
                        role="system",
                        content=progress_msg.content,
                        metadata={"workflow_update": True}
                    )

                # Stop spinner and remove processing message
                if len(chat_view) > 0:
                    last_item = chat_view.children[-1]
                    if hasattr(last_item, "is_processing") and last_item.is_processing:
                        last_item.stop_spinner()

                chat_view.messages.pop()  # Remove the processing indicator
                if len(chat_view) > 0:
                    chat_view.pop()

                # Add actual response
                chat_view.add_message(response_msg)

                # Display workflow progress message if step completed (T090)
                if workflow_update:
                    chat_view.add_message(progress_msg)

                # Extract and update API token usage from agent chain
                if agent_chain:
                    # Get the last used agent's token stats
                    selected_name = agent_chain.last_selected_agent or active_agent_name
                    if selected_name and selected_name in agent_chain.agents:
                        agent_prompt_chain = agent_chain.agents[selected_name]
                        if hasattr(agent_prompt_chain, 'total_prompt_tokens'):
                            # Update cumulative totals from the PromptChain
                            self.cumulative_prompt_tokens = agent_prompt_chain.total_prompt_tokens
                            self.cumulative_completion_tokens = agent_prompt_chain.total_completion_tokens

                # Update status bar with token counts
                history_stats = self.session.history_manager.get_statistics()
                status_bar = self.query_one("#status-bar", StatusBar)
                status_bar.update_session_info(
                    message_count=len(self.session.messages),
                    token_count=history_stats.get("total_tokens", 0),
                    max_tokens=self.session.history_max_tokens,
                    api_prompt_tokens=self.cumulative_prompt_tokens,
                    api_completion_tokens=self.cumulative_completion_tokens,
                )

            finally:
                # REACT Loop: Reset processing flag
                self.is_processing = False
                # Clear any remaining items in the queue
                while not self.user_input_queue.empty():
                    try:
                        self.user_input_queue.get_nowait()
                    except:
                        break

                # Disable auto-refresh after execution (Phase 5)
                if self.log_viewer:
                    self.log_viewer.disable_auto_refresh()
                    # Final refresh to show complete execution
                    if self.log_viewer_visible:
                        self.log_viewer.load_activities()

                # Mark task list processing as complete
                try:
                    task_list = self.query_one("#task-list-widget", TaskListWidget)
                    task_list.mark_processing_complete()
                except Exception:
                    pass  # Widget may not be ready

        except Exception as e:
            # REACT Loop: Reset processing flag on error
            self.is_processing = False
            # Mark task list processing as complete (even on error)
            try:
                task_list = self.query_one("#task-list-widget", TaskListWidget)
                task_list.mark_processing_complete()
            except Exception:
                pass
            # Clear queue
            while not self.user_input_queue.empty():
                try:
                    self.user_input_queue.get_nowait()
                except:
                    break

            # Use global error handler for user-friendly messages (T141-T142)
            error_msg = self._handle_error(e, context="generating response")

            # Log error to session for debugging (T143)
            self.session.add_message(
                role="system", content=error_msg.content, metadata=error_msg.metadata
            )

            # Stop spinner and remove processing indicator if it exists
            if len(chat_view) > 0:
                last_item = chat_view.children[-1]
                if hasattr(last_item, "is_processing") and last_item.is_processing:
                    last_item.stop_spinner()

            if (
                chat_view.messages
                and chat_view.messages[-1].role == "system"
                and (
                    "Processing" in chat_view.messages[-1].content
                    or "⏳" in chat_view.messages[-1].content
                )
            ):
                chat_view.messages.pop()
                if len(chat_view) > 0:
                    chat_view.pop()

            chat_view.add_message(error_msg)

        # Update status bar
        status_bar.update_session_info(message_count=len(self.session.messages))

    async def _handle_workflow_message(self, content: str, workflow: "WorkflowState"):
        """Handle message when workflow is active (T091).

        Integrates workflow objective with AgenticStepProcessor for automatic
        step-by-step execution guided by the current workflow step.

        Args:
            content: User message with file context injected
            workflow: Active workflow state
        """
        from ..models import Message
        from promptchain.utils.agentic_step_processor import AgenticStepProcessor

        chat_view = self.query_one("#chat-view", ChatView)
        status_bar = self.query_one("#status-bar", StatusBar)

        # Get current step
        current_step = workflow.current_step
        if not current_step:
            # No more steps - workflow complete
            completion_msg = Message(
                role="system",
                content="[bold]Workflow completed![/bold]\n"
                        f"All {len(workflow.steps)} steps finished."
            )
            chat_view.add_message(completion_msg)
            return

        # Mark step as in_progress
        if current_step.status == "pending":
            current_step.mark_in_progress(self.session.active_agent)
            self.session_manager.save_workflow(self.session.id, workflow)

        # Show workflow context
        workflow_context_msg = Message(
            role="system",
            content=(
                f"[bold]Workflow Active:[/bold] {workflow.objective}\n"
                f"[dim]Current step ({workflow.current_step_index + 1}/{len(workflow.steps)}):[/dim] {current_step.description}\n"
                f"[dim]Progress: {workflow.progress_percentage:.0f}%[/dim]"
            )
        )
        chat_view.add_message(workflow_context_msg)

        # Update status bar to show workflow progress (T091)
        status_bar.update_session_info(
            message_count=len(self.session.messages),
            workflow_objective=workflow.objective,
            workflow_progress=workflow.progress_percentage,
            workflow_step=f"{workflow.current_step_index + 1}/{len(workflow.steps)}",
        )

        # Get active agent model
        active_agent = self.session.agents.get(self.session.active_agent)
        model_name = active_agent.model_name if active_agent else self.session.default_model

        # Show processing indicator
        processing_msg = Message(
            role="system",
            content=f"Processing workflow step with {self.session.active_agent} ({model_name})...",
            metadata={"is_processing": True},
        )
        chat_view.add_message(processing_msg)

        # Start spinner on processing message
        if len(chat_view) > 0:
            last_item = chat_view.children[-1]
            if hasattr(last_item, "is_processing"):
                last_item.is_processing = True
                last_item.start_spinner()

        try:
            # T091: Create AgenticStepProcessor with workflow step objective
            # The objective is derived from the current workflow step, not the overall workflow
            # Use agentic config for max_internal_steps (default: 15)
            max_steps = self.config.agentic.default_max_internal_steps
            agentic_processor = AgenticStepProcessor(
                objective=current_step.description,
                max_internal_steps=max_steps,
                model_name=model_name,
                history_mode=self.config.agentic.history_mode,  # Use config history mode
                progress_callback=self._reasoning_progress_callback,  # T052: Progress updates
            )

            # Create workflow-aware PromptChain with AgenticStepProcessor
            workflow_chain = PromptChain(
                models=[{
                    "name": model_name,
                    "params": {"max_completion_tokens": 16000}
                }],
                instructions=[
                    f"Working on workflow: {workflow.objective}\nCurrent step: {current_step.description}",
                    agentic_processor,  # Multi-hop reasoning for current step
                    "Summarize step completion: {input}"
                ],
                verbose=False,
            )

            # Show reasoning progress for workflow step
            self.show_reasoning_progress(
                objective=current_step.description,
                max_steps=max_steps
            )

            # Execute workflow step with agentic reasoning
            response_content = await workflow_chain.process_prompt_async(content)

            # T091: Track AgenticStepProcessor internal steps as metadata
            # This allows us to see what the processor actually did
            agentic_metadata = {
                "steps_executed": getattr(agentic_processor, "steps_executed", 0),
                "total_tools_called": getattr(agentic_processor, "total_tools_called", 0),
                "total_tokens_used": getattr(agentic_processor, "total_tokens_used", 0),
                "execution_time_ms": getattr(agentic_processor, "execution_time_ms", 0),
                "max_steps_reached": getattr(agentic_processor, "max_steps_reached", False),
            }

            # Hide reasoning progress
            self.hide_reasoning_progress()

            # Stop spinner and remove processing indicator
            if len(chat_view) > 0:
                last_item = chat_view.children[-1]
                if hasattr(last_item, "is_processing") and last_item.is_processing:
                    last_item.stop_spinner()

            chat_view.messages.pop()  # Remove processing indicator
            if len(chat_view) > 0:
                chat_view.pop()

            # Create assistant response
            response_msg = Message(
                role="assistant",
                content=response_content,
                agent_name=self.session.active_agent,
                model_name=model_name,
            )

            # Add to session and display with agentic metadata
            self.session.add_message(
                role="assistant",
                content=response_content,
                metadata={
                    "agent_name": self.session.active_agent,
                    "model_name": model_name,
                    "workflow_step": workflow.current_step_index,
                    "agentic_processor": agentic_metadata,  # T091: Track processor metrics
                },
            )

            chat_view.add_message(response_msg)

            # Refresh task list widget after workflow response
            self._refresh_task_list_widget()

            # T091: Update workflow state after agent response
            workflow_update = self.session_manager.update_workflow_on_message(
                self.session, self.session.messages[-1]
            )

            # Display workflow progress update if step completed (T090)
            if workflow_update:
                progress_msg = Message(
                    role="system",
                    content=(
                        f"[bold]Workflow step completed:[/bold] {workflow_update['step_description']}\n"
                        f"[dim]Workflow progress: {workflow_update['progress_percentage']:.0f}% "
                        f"({workflow_update['completed_count']}/{workflow_update['total_steps']} steps)[/dim]"
                    )
                )
                chat_view.add_message(progress_msg)

                # Update status bar with new progress (T091)
                status_bar.update_session_info(
                    workflow_progress=workflow_update['progress_percentage'],
                    workflow_step=f"{workflow_update['completed_count']}/{workflow_update['total_steps']}",
                )

                # Check if workflow is now complete
                workflow = self.session_manager.load_workflow(self.session.id)
                if workflow and workflow.is_completed:
                    completion_msg = Message(
                        role="system",
                        content=(
                            f"[bold]Workflow '{workflow.objective}' completed successfully![/bold]\n"
                            f"[dim]All {len(workflow.steps)} steps finished.[/dim]"
                        )
                    )
                    chat_view.add_message(completion_msg)

                    # Clear workflow from status bar (T091)
                    status_bar.update_session_info(
                        workflow_objective=None,
                        workflow_progress=None,
                        workflow_step=None,
                    )

        except Exception as e:
            # Hide reasoning progress on error
            self.hide_reasoning_progress()

            # Stop spinner and remove processing indicator
            if len(chat_view) > 0:
                last_item = chat_view.children[-1]
                if hasattr(last_item, "is_processing") and last_item.is_processing:
                    last_item.stop_spinner()

            if (
                chat_view.messages
                and chat_view.messages[-1].role == "system"
                and "Processing" in chat_view.messages[-1].content
            ):
                chat_view.messages.pop()
                if len(chat_view) > 0:
                    chat_view.pop()

            # Use global error handler
            error_msg = self._handle_error(e, context="executing workflow step")
            chat_view.add_message(error_msg)

            # Mark step as failed
            if current_step:
                current_step.mark_failed(str(e))
                self.session_manager.save_workflow(self.session.id, workflow)

        # Update status bar message count
        status_bar.update_session_info(message_count=len(self.session.messages))

    async def _display_system_message(self, message: str) -> None:
        """Display system message in chat with styling (T042).

        Args:
            message: Rich markup formatted message
        """
        chat_view = self.query_one("#chat-view", ChatView)
        system_msg = Message(role="system", content=message)
        chat_view.add_message(system_msg)

    def on_exception(self, event) -> None:
        """Global exception handler for Textual app (T141).

        Catches all unhandled exceptions and provides graceful recovery.
        Overrides Textual's default exception handling.

        Args:
            event: Exception event from Textual
        """
        error = event.exception if hasattr(event, "exception") else event

        # Handle error with comprehensive error handler
        error_msg = self._handle_error(error, "application error")

        # Try to display error in chat view
        try:
            chat_view = self.query_one("#chat-view", ChatView)
            chat_view.add_message(error_msg)

            # Save session state before potential crash
            if self.session:
                try:
                    self.session_manager.save_session(self.session)
                    recovery_msg = Message(
                        role="system",
                        content="[dim]Session saved before error. You can continue or restart.[/dim]",
                    )
                    chat_view.add_message(recovery_msg)
                except Exception as save_error:
                    # If save fails, show error but don't crash
                    save_error_msg = Message(
                        role="system",
                        content=f"[dim]Warning: Could not save session: {str(save_error)}[/dim]",
                    )
                    chat_view.add_message(save_error_msg)

        except Exception as display_error:
            # If we can't even display the error, log it
            self.error_handler.logger.critical(
                f"Critical error - could not display error message: {str(display_error)}"
            )
            # Let Textual handle it (will exit)
            raise

        # Prevent default Textual behavior (which exits the app)
        # Allow app to continue running
        event.prevent_default()

    async def on_exit(self):
        """Handle app exit - save session and cleanup observers."""
        try:
            # Cleanup MLflow observer if active
            if hasattr(self, '_mlflow_observer'):
                try:
                    self._mlflow_observer.shutdown()
                    logger.debug("MLflow observer shutdown complete")
                except Exception as e:
                    logger.warning(f"Error during MLflow observer shutdown: {e}")

            # Save session
            if self.session:
                self.session_manager.save_session(self.session)
        except Exception as e:
            # Log error but don't prevent exit
            self.error_handler.logger.error(f"Error saving session on exit: {str(e)}")
