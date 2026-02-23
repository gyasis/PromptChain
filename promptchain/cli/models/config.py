"""Configuration model for PromptChain CLI.

This module defines the Config data model for managing CLI settings
including default models, UI preferences, and performance tuning.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class UIConfig:
    """UI/UX configuration settings.

    Attributes:
        max_displayed_messages: Maximum messages shown in chat view (pagination)
        show_line_numbers: Whether to show line numbers in input widget
        theme: Color theme (default, dark, light)
        animation_fps: Frames per second for spinner animations
    """

    max_displayed_messages: int = 100
    show_line_numbers: bool = False
    theme: str = "default"
    animation_fps: int = 10


@dataclass
class PerformanceConfig:
    """Performance tuning settings.

    Attributes:
        lazy_load_agents: Whether to lazy load agents on startup
        history_max_tokens: Maximum tokens for conversation history
        cache_enabled: Whether to enable session caching
    """

    lazy_load_agents: bool = True
    history_max_tokens: int = 128000  # Default for large context models
    cache_enabled: bool = True


@dataclass
class MCPServerDefaults:
    """Default MCP server configuration.

    Defines an MCP server that should be auto-connected on session start.

    Attributes:
        id: Server identifier
        command: Command to run the server
        args: Command arguments
        auto_connect: Whether to auto-connect on session start
        description: Human-readable description
    """

    id: str
    command: str
    args: list = field(default_factory=list)
    auto_connect: bool = True
    description: str = ""


@dataclass
class MCPConfig:
    """MCP (Model Context Protocol) configuration settings.

    Defines default MCP servers that are auto-connected on session start.

    Attributes:
        default_servers: List of default MCP servers to auto-connect
        auto_connect_on_start: Whether to auto-connect servers when CLI starts
    """

    default_servers: list = field(default_factory=lambda: [
        # Gemini MCP server with web search enabled by default
        {
            "id": "gemini",
            "command": "uv",
            "args": ["run", "--directory", "/home/gyasis/Documents/code/gemini-mcp", "fastmcp", "run"],
            "auto_connect": True,
            "description": "Gemini AI with web search (gemini_research), code review, brainstorming"
        }
    ])
    auto_connect_on_start: bool = True


@dataclass
class AgenticConfig:
    """Agentic workflow configuration settings.

    Controls the AgenticStepProcessor behavior and agentic loop workflow
    for task completion, handoffs, and user interaction thresholds.

    Attributes:
        default_max_internal_steps: Maximum internal reasoning steps per task (default: 15)
        task_completion_threshold: Retry attempts before asking user for input (default: 3)
        handoff_enabled: Enable handoff to other agents when task requires (default: True)
        completion_check_enabled: Check if task is complete after each step (default: True)
        user_input_threshold: Max steps without progress before returning to user (default: 5)
        history_mode: Default history mode for agentic steps (progressive/minimal/full)
    """

    default_max_internal_steps: int = 15
    task_completion_threshold: int = 3
    handoff_enabled: bool = True
    completion_check_enabled: bool = True
    user_input_threshold: int = 5
    history_mode: str = "progressive"


@dataclass
class ContextManagementConfig:
    """Smart context management configuration.

    Controls automatic history summarization and ephemeral tool execution
    to prevent context window overflow in long agentic sessions.

    Attributes:
        # Summarization settings
        summarizer_model: LiteLLM model for history summarization (cheap/fast)
        summarize_every_n_iterations: Trigger summarization after N iterations
        summarize_token_threshold: Trigger summarization when tokens exceed this % of max
        max_summary_tokens: Maximum tokens for summary output
        preserve_last_n_turns: Number of recent turns to preserve verbatim

        # Ephemeral execution settings
        ephemeral_enabled: Whether to enable ephemeral execution for heavy tools
        ephemeral_file_threshold_kb: Files larger than this use ephemeral execution
        ephemeral_response_threshold_kb: Responses larger than this get summarized
        ephemeral_timeout_seconds: Maximum execution time for ephemeral tools
        capture_full_errors: Whether to include full error details in results
    """

    # Summarization settings
    summarizer_model: str = "openai/gpt-4.1-mini-2025-04-14"
    summarize_every_n_iterations: int = 5
    summarize_token_threshold: float = 0.7  # 70% of max
    max_summary_tokens: int = 500
    preserve_last_n_turns: int = 2

    # Ephemeral execution settings
    ephemeral_enabled: bool = True
    ephemeral_file_threshold_kb: int = 10
    ephemeral_response_threshold_kb: int = 5
    ephemeral_timeout_seconds: int = 300
    capture_full_errors: bool = True


@dataclass
class Config:
    """PromptChain CLI configuration.

    Manages user preferences, default settings, and performance tuning.

    Attributes:
        default_model: Default LiteLLM model string
        default_agent: Default agent name to use
        sessions_dir: Directory for session storage
        debug_mode: Enable debug mode with full tracebacks (T141)
        ui: UI configuration
        performance: Performance configuration
        agentic: Agentic workflow configuration
        mcp: MCP server configuration
        context_management: Smart context management for preventing overflow
        metadata: Extensible storage for additional settings
    """

    default_model: str = "openai/gpt-4.1-mini-2025-04-14"  # Better function calling, 1M context ($0.40/1M vs gpt-4 $30/1M)
    default_agent: str = "default"
    sessions_dir: Optional[str] = None
    debug_mode: bool = False  # T141: Debug mode for error handler
    ui: UIConfig = field(default_factory=UIConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)  # Default MCP servers
    context_management: ContextManagementConfig = field(default_factory=ContextManagementConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Path) -> "Config":
        """Load configuration from JSON file (T152).

        Args:
            config_path: Path to config.json

        Returns:
            Config: Loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                data = json.load(f)

            # Validate and construct nested objects
            ui_data = data.pop("ui", {})
            perf_data = data.pop("performance", {})
            agentic_data = data.pop("agentic", {})
            mcp_data = data.pop("mcp", {})
            context_mgmt_data = data.pop("context_management", {})

            return cls(
                **data,
                ui=UIConfig(**ui_data),
                performance=PerformanceConfig(**perf_data),
                agentic=AgenticConfig(**agentic_data),
                mcp=MCPConfig(**mcp_data),
                context_management=ContextManagementConfig(**context_mgmt_data),
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        except TypeError as e:
            raise ValueError(f"Invalid config structure: {e}")

    def save(self, config_path: Path) -> None:
        """Save configuration to JSON file (T151).

        Args:
            config_path: Path to save config.json
        """
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict with nested objects
        data = asdict(self)

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

    def validate(self) -> None:
        """Validate configuration values (T153).

        Raises:
            ValueError: If any configuration value is invalid
        """
        # Validate default_model format
        if not self.default_model or "/" not in self.default_model:
            raise ValueError(
                f"default_model must be in format 'provider/model', got: {self.default_model}"
            )

        # Validate UI settings
        if self.ui.max_displayed_messages < 10:
            raise ValueError(
                f"ui.max_displayed_messages must be >= 10, got: {self.ui.max_displayed_messages}"
            )

        if self.ui.animation_fps < 1 or self.ui.animation_fps > 60:
            raise ValueError(f"ui.animation_fps must be 1-60, got: {self.ui.animation_fps}")

        if self.ui.theme not in ["default", "dark", "light"]:
            raise ValueError(f"ui.theme must be default/dark/light, got: {self.ui.theme}")

        # Validate performance settings
        if self.performance.history_max_tokens < 1000:
            raise ValueError(
                f"performance.history_max_tokens must be >= 1000, got: {self.performance.history_max_tokens}"
            )

        # Validate agentic settings
        if self.agentic.default_max_internal_steps < 1 or self.agentic.default_max_internal_steps > 50:
            raise ValueError(
                f"agentic.default_max_internal_steps must be 1-50, got: {self.agentic.default_max_internal_steps}"
            )

        if self.agentic.task_completion_threshold < 1:
            raise ValueError(
                f"agentic.task_completion_threshold must be >= 1, got: {self.agentic.task_completion_threshold}"
            )

        if self.agentic.user_input_threshold < 1:
            raise ValueError(
                f"agentic.user_input_threshold must be >= 1, got: {self.agentic.user_input_threshold}"
            )

        if self.agentic.history_mode not in ["progressive", "minimal", "full"]:
            raise ValueError(
                f"agentic.history_mode must be progressive/minimal/full, got: {self.agentic.history_mode}"
            )

        # Validate context management settings
        cm = self.context_management
        if cm.summarize_every_n_iterations < 1:
            raise ValueError(
                f"context_management.summarize_every_n_iterations must be >= 1, got: {cm.summarize_every_n_iterations}"
            )

        if cm.summarize_token_threshold < 0.1 or cm.summarize_token_threshold > 1.0:
            raise ValueError(
                f"context_management.summarize_token_threshold must be 0.1-1.0, got: {cm.summarize_token_threshold}"
            )

        if cm.max_summary_tokens < 100 or cm.max_summary_tokens > 2000:
            raise ValueError(
                f"context_management.max_summary_tokens must be 100-2000, got: {cm.max_summary_tokens}"
            )

        if cm.preserve_last_n_turns < 0 or cm.preserve_last_n_turns > 10:
            raise ValueError(
                f"context_management.preserve_last_n_turns must be 0-10, got: {cm.preserve_last_n_turns}"
            )

        if cm.ephemeral_file_threshold_kb < 1:
            raise ValueError(
                f"context_management.ephemeral_file_threshold_kb must be >= 1, got: {cm.ephemeral_file_threshold_kb}"
            )

        if cm.ephemeral_timeout_seconds < 10 or cm.ephemeral_timeout_seconds > 3600:
            raise ValueError(
                f"context_management.ephemeral_timeout_seconds must be 10-3600, got: {cm.ephemeral_timeout_seconds}"
            )

    @classmethod
    def get_default_path(cls) -> Path:
        """Get default config file path.

        Returns:
            Path: ~/.promptchain/config.json
        """
        return Path.home() / ".promptchain" / "config.json"

    @classmethod
    def load_or_create_default(cls) -> "Config":
        """Load config or create default if doesn't exist.

        Returns:
            Config: Loaded or default configuration
        """
        config_path = cls.get_default_path()

        if config_path.exists():
            try:
                config = cls.load(config_path)
                config.validate()
                return config
            except (FileNotFoundError, ValueError):
                # Fall back to default if load fails
                pass

        # Create default config
        config = cls()
        config.validate()
        return config
