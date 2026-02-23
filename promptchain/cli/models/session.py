"""Session model for PromptChain CLI.

This module defines the Session data model representing a persistent conversation
instance with full state management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Session:
    """Represents a persistent conversation instance.

    A session maintains conversation state across CLI invocations, including
    active agents, working directory context, and conversation history.

    V2 Schema (002-cli-orchestration):
    - Adds orchestration_config for AgentChain integration
    - Adds mcp_servers for Model Context Protocol support
    - Adds workflow_state for multi-session objective tracking
    - Adds schema_version for migration tracking

    Attributes:
        # V1 Schema (existing fields)
        id: Unique session identifier (UUID)
        name: Human-readable session name (1-64 chars, alphanumeric+dashes)
        created_at: Session creation timestamp (Unix seconds)
        last_accessed: Last access timestamp (Unix seconds, auto-updated)
        working_directory: Directory context for file references and shell commands
        active_agent: Currently active agent name (None = default agent)
        default_model: LLM model for default agent (LiteLLM format)
        auto_save_enabled: Whether auto-save is enabled
        auto_save_interval: Auto-save interval in seconds (60-600)
        metadata: Extensible storage for session-specific data
        agents: Dictionary of agent configurations {agent_name: Agent}
        messages: List of conversation messages

        # V2 Schema (orchestration fields)
        orchestration_config: AgentChain execution mode and router settings
        mcp_servers: List of MCP server configurations
        workflow_state: Multi-session workflow progress tracking
        schema_version: Schema version for migration tracking (default: "2.0")
    """

    # V1 Schema fields (existing)
    id: str
    name: str
    created_at: float
    last_accessed: float
    working_directory: Path
    active_agent: Optional[str] = None
    default_model: str = "openai/gpt-4.1-mini-2025-04-14"
    auto_save_enabled: bool = True
    auto_save_interval: int = 120  # 2 minutes
    history_max_tokens: int = 128000  # Max tokens for history manager (from config)
    metadata: Dict[str, Any] = field(default_factory=dict)
    agents: Dict[str, Any] = field(default_factory=dict)  # Will be Dict[str, Agent]
    messages: list = field(default_factory=list)  # Will be List[Message]
    _history_manager: Any = field(
        default=None, init=False, repr=False
    )  # ExecutionHistoryManager (lazy init)

    # Activity logging (Phase 2 integration)
    _activity_logger: Any = field(
        default=None, init=False, repr=False
    )  # ActivityLogger (lazy init, captures ALL agent interactions)

    # V2 Schema fields (orchestration)
    orchestration_config: Optional[Any] = None  # OrchestrationConfig (lazy import to avoid circular)
    mcp_servers: List[Any] = field(default_factory=list)  # List[MCPServerConfig]
    workflow_state: Optional[Any] = None  # WorkflowState (lazy import)
    schema_version: str = "2.0"

    # Security context for path boundary checking
    security_context_data: Optional[Dict[str, Any]] = None  # Serialized SecurityContext

    def __setattr__(self, name, value):
        """Override setattr to add validation for auto_save_interval."""
        if name == "auto_save_interval":
            if not (60 <= value <= 600):
                raise ValueError(f"auto_save_interval must be 60-600 seconds, got: {value}")
        super().__setattr__(name, value)

    def __post_init__(self):
        """Validate session attributes after initialization."""
        # Validate auto_save_interval
        if not (60 <= self.auto_save_interval <= 600):
            raise ValueError(
                f"auto_save_interval must be 60-600 seconds, got: {self.auto_save_interval}"
            )

        # Validate name format
        if not self.name or len(self.name) > 64:
            raise ValueError(f"Session name must be 1-64 characters, got: {len(self.name)}")

        if not self.name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Session name must be alphanumeric+dashes/underscores: {self.name}")

        # Ensure working_directory is Path object
        if not isinstance(self.working_directory, Path):
            self.working_directory = Path(self.working_directory)

        # Validate working directory exists
        if not self.working_directory.exists():
            raise ValueError(f"Working directory does not exist: {self.working_directory}")

        # Initialize auto-save tracking attributes (T085-T086)
        if not hasattr(self, "messages_since_save"):
            self.messages_since_save = 0
        if not hasattr(self, "last_save_time"):
            self.last_save_time = self.created_at
        if not hasattr(self, "autosave_message_interval"):
            self.autosave_message_interval = 5  # Default: save every 5 messages
        if not hasattr(self, "autosave_time_interval"):
            self.autosave_time_interval = 120  # Default: save every 2 minutes

        # V2 Schema: Initialize orchestration_config if None (T011)
        if self.orchestration_config is None:
            # Lazy import to avoid circular dependencies
            from .orchestration_config import OrchestrationConfig

            self.orchestration_config = OrchestrationConfig()

        # Validate schema_version format
        if not self.schema_version or not self.schema_version[0].isdigit():
            raise ValueError(f"Invalid schema_version format: {self.schema_version}")

    @property
    def state(self) -> str:
        """Get current session state.

        Returns:
            str: Session state (Active, Paused, Archived)
        """
        now = datetime.now().timestamp()

        # Active: accessed within last hour
        if now - self.last_accessed < 3600:
            return "Active"

        # Paused: accessed within last 24 hours
        elif now - self.last_accessed < 86400:
            return "Paused"

        # Archived: not accessed in >24 hours
        else:
            return "Archived"

    def update_access_time(self):
        """Update last_accessed timestamp to current time."""
        self.last_accessed = datetime.now().timestamp()

    @property
    def history_manager(self):
        """Get or create ExecutionHistoryManager for this session (T033).

        Lazy initialization ensures history manager is only created when needed.
        Uses session's history_max_tokens setting (from config.performance.history_max_tokens).

        Returns:
            ExecutionHistoryManager: History manager for this session
        """
        if self._history_manager is None:
            # Import here to avoid circular import
            from promptchain.utils.execution_history_manager import ExecutionHistoryManager

            # Create history manager with session's configured max tokens
            self._history_manager = ExecutionHistoryManager(
                max_tokens=self.history_max_tokens,
                max_entries=50,
                truncation_strategy="oldest_first",
            )

            # Populate history from existing messages
            for msg in self.messages:
                self._history_manager.add_entry(
                    entry_type=msg.role,
                    content=msg.content,
                    source=msg.agent_name or "user",
                    metadata=msg.metadata,
                )

        return self._history_manager

    @property
    def security_context(self):
        """Get SecurityContext for this session.

        Lazily initializes the SecurityContext from stored data or creates new one.
        Updates the global security context when accessed.

        Returns:
            SecurityContext: Security context for path boundary checking
        """
        from ..security_context import SecurityContext, set_security_context

        if self.security_context_data:
            # Restore from stored data
            ctx = SecurityContext.from_dict(self.security_context_data)
        else:
            # Create new with session's working directory
            ctx = SecurityContext(working_directory=str(self.working_directory))

        # Update global context
        set_security_context(ctx)
        return ctx

    def save_security_context(self):
        """Save current security context to session data for persistence."""
        from ..security_context import get_security_context

        ctx = get_security_context()
        self.security_context_data = ctx.to_dict()

    @property
    def activity_logger(self):
        """Get ActivityLogger for this session (Phase 3).

        ActivityLogger is initialized by SessionManager during session creation/load.
        This property provides access to the logger for AgentChain integration.

        Returns:
            ActivityLogger: Activity logger for this session, or None if not initialized

        Note:
            ActivityLogger captures ALL agent interactions without consuming tokens.
            It's automatically initialized when sessions are created or loaded.
        """
        return self._activity_logger

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation history (T025).

        Args:
            role: Message role ('user', 'assistant', 'system', 'tool')
            content: Message content (must be non-empty)
            metadata: Optional metadata for the message

        Raises:
            ValueError: If content is empty or whitespace-only
        """
        from .message import Message  # Avoid circular import

        # Validate content is non-empty
        if not content or not content.strip():
            raise ValueError("Message content must be non-empty")

        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().timestamp(),
            metadata=metadata or {},
        )
        self.messages.append(message)
        self.update_access_time()

        # Track messages since save for auto-save (T085)
        if not hasattr(self, "messages_since_save"):
            self.messages_since_save = 0
        self.messages_since_save += 1

        # Also add to history manager if it exists (T033)
        if self._history_manager is not None:
            self._history_manager.add_entry(
                entry_type=role,
                content=content,
                source=metadata.get("agent_name", "user") if metadata else "user",
                metadata=metadata or {},
            )

    def check_autosave(self, session_manager):
        """Check if auto-save should trigger and execute if needed (T085-T086).

        Args:
            session_manager: SessionManager instance to use for saving

        Auto-save triggers when:
        - 5 or more messages added since last save (configurable via autosave_message_interval)
        - 2 minutes passed since last save (configurable via auto_save_interval)
        """
        # Check if auto-save is enabled (support both naming conventions)
        autosave_enabled = getattr(self, "autosave_enabled", self.auto_save_enabled)
        if not autosave_enabled:
            return

        # Initialize tracking attributes if not present
        if not hasattr(self, "messages_since_save"):
            self.messages_since_save = 0
        if not hasattr(self, "last_save_time"):
            self.last_save_time = self.created_at
        if not hasattr(self, "autosave_message_interval"):
            self.autosave_message_interval = 5  # Default: save every 5 messages
        if not hasattr(self, "autosave_time_interval"):
            self.autosave_time_interval = 120  # Default: save every 2 minutes

        current_time = datetime.now().timestamp()

        # Check message count threshold
        message_threshold_met = self.messages_since_save >= self.autosave_message_interval

        # Check time threshold
        time_threshold_met = (current_time - self.last_save_time) >= self.autosave_time_interval

        # Trigger save if either threshold met
        if message_threshold_met or time_threshold_met:
            try:
                session_manager.save_session(self)
                self.messages_since_save = 0
                self.last_save_time = current_time
            except Exception as e:
                # Log error but don't crash the session
                import logging

                logging.error(f"Auto-save failed: {e}")

    def on_exit(self, session_manager):
        """Perform final save on session exit (T086).

        Args:
            session_manager: SessionManager instance to use for saving
        """
        if self.auto_save_enabled:
            try:
                session_manager.save_session(self)
            except Exception as e:
                import logging

                logging.error(f"Final save on exit failed: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for SQLite storage.

        V2 Schema includes orchestration_config, mcp_servers, workflow_state, and schema_version.

        Returns:
            Dict[str, Any]: Session data as dictionary
        """
        # V1 fields
        result = {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "working_directory": str(self.working_directory),
            "active_agent": self.active_agent,
            "default_model": self.default_model,
            "auto_save_enabled": int(self.auto_save_enabled),
            "auto_save_interval": self.auto_save_interval,
            "metadata_json": str(self.metadata),  # JSON serialization in SessionManager
        }

        # V2 fields (orchestration)
        result["schema_version"] = self.schema_version
        result["orchestration_config"] = (
            self.orchestration_config.to_dict() if self.orchestration_config else None
        )
        result["mcp_servers"] = [server.to_dict() for server in self.mcp_servers]
        result["workflow_state"] = (
            self.workflow_state.to_dict() if self.workflow_state else None
        )

        # Security context
        result["security_context_data"] = self.security_context_data

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create session from dictionary.

        Handles both V1 and V2 schema for backward compatibility.

        Args:
            data: Session data dictionary from SQLite

        Returns:
            Session: Reconstructed session object
        """
        # Lazy imports to avoid circular dependencies
        from .mcp_config import MCPServerConfig
        from .orchestration_config import OrchestrationConfig
        from .workflow import WorkflowState

        # V1 fields
        session = cls(
            id=data["id"],
            name=data["name"],
            created_at=data["created_at"],
            last_accessed=data["last_accessed"],
            working_directory=Path(data["working_directory"]),
            active_agent=data.get("active_agent"),
            default_model=data.get("default_model", "openai/gpt-4.1-mini-2025-04-14"),
            auto_save_enabled=bool(data.get("auto_save_enabled", 1)),
            auto_save_interval=data.get("auto_save_interval", 120),
            metadata=data.get("metadata", {}),
        )

        # V2 fields (orchestration) - only if present in data
        schema_version = data.get("schema_version", "1.0")
        session.schema_version = schema_version

        if data.get("orchestration_config"):
            session.orchestration_config = OrchestrationConfig.from_dict(
                data["orchestration_config"]
            )

        if data.get("mcp_servers"):
            session.mcp_servers = [
                MCPServerConfig.from_dict(server_data) for server_data in data["mcp_servers"]
            ]

        if data.get("workflow_state"):
            session.workflow_state = WorkflowState.from_dict(data["workflow_state"])

        # Security context
        if data.get("security_context_data"):
            session.security_context_data = data["security_context_data"]

        return session
