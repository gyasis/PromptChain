"""Agent configuration model for PromptChain CLI.

This module defines the Agent data model representing an AI agent configuration
with model-agnostic support via LiteLLM, plus extended orchestration features
for AgentChain integration (instruction chains, history configs, tool access).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass
class HistoryConfig:
    """Token-efficient history configuration for individual agents.

    Controls how conversation history is managed for each agent to optimize
    token usage and context relevance. Different agent types benefit from
    different history strategies (e.g., terminal agents don't need history).

    Attributes:
        enabled: Whether to include conversation history (default: True)
        max_tokens: Maximum tokens for history context (100-16000 range)
        max_entries: Maximum number of history entries (messages, tool calls)
        truncation_strategy: How to truncate when limits exceeded
        include_types: Filter history by entry types (e.g., ["user_input", "agent_output"])
        exclude_sources: Exclude entries from specific sources
    """

    enabled: bool = True
    max_tokens: int = 4000
    max_entries: int = 20
    truncation_strategy: Literal["oldest_first", "keep_last"] = "oldest_first"
    include_types: Optional[List[str]] = None
    exclude_sources: Optional[List[str]] = None

    def __post_init__(self):
        """Validate history configuration parameters."""
        # Allow max_tokens=0 and max_entries=0 when history is disabled (terminal mode)
        if self.enabled:
            if self.max_tokens < 100 or self.max_tokens > 16000:
                raise ValueError(
                    f"max_tokens must be between 100-16000, got: {self.max_tokens}"
                )

            if self.max_entries < 1 or self.max_entries > 200:
                raise ValueError(
                    f"max_entries must be between 1-200, got: {self.max_entries}"
                )
        else:
            # When disabled, max_tokens and max_entries must be 0
            if self.max_tokens != 0 or self.max_entries != 0:
                raise ValueError(
                    f"When enabled=False, max_tokens and max_entries must be 0, "
                    f"got max_tokens={self.max_tokens}, max_entries={self.max_entries}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "enabled": self.enabled,
            "max_tokens": self.max_tokens,
            "max_entries": self.max_entries,
            "truncation_strategy": self.truncation_strategy,
            "include_types": self.include_types,
            "exclude_sources": self.exclude_sources,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryConfig":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            max_tokens=data.get("max_tokens", 4000),
            max_entries=data.get("max_entries", 20),
            truncation_strategy=data.get("truncation_strategy", "oldest_first"),
            include_types=data.get("include_types"),
            exclude_sources=data.get("exclude_sources"),
        )

    @classmethod
    def for_agent_type(cls, agent_type: str) -> "HistoryConfig":
        """Create default history config for specific agent type (T076).

        Different agent types have different history requirements:
        - terminal: No history (saves 100% tokens)
        - coder: Moderate history (4000 tokens, 20 entries)
        - researcher: Full history (8000 tokens, 50 entries)
        - analyst: Full history (8000 tokens, 50 entries)
        - default: Coder-level settings

        Args:
            agent_type: Agent type identifier ("terminal", "coder", "researcher", etc.)

        Returns:
            HistoryConfig: Optimized configuration for the agent type

        Examples:
            >>> terminal_config = HistoryConfig.for_agent_type("terminal")
            >>> assert terminal_config.enabled is False
            >>> coder_config = HistoryConfig.for_agent_type("coder")
            >>> assert coder_config.max_tokens == 4000
        """
        type_configs = {
            "terminal": {
                "enabled": False,
                "max_tokens": 0,
                "max_entries": 0,
            },
            "coder": {
                "enabled": True,
                "max_tokens": 4000,
                "max_entries": 20,
                "truncation_strategy": "oldest_first",
            },
            "researcher": {
                "enabled": True,
                "max_tokens": 8000,
                "max_entries": 50,
                "truncation_strategy": "oldest_first",
            },
            "analyst": {
                "enabled": True,
                "max_tokens": 8000,
                "max_entries": 50,
                "truncation_strategy": "oldest_first",
            },
        }

        # Get config for type, fallback to coder defaults
        config_dict: Dict[str, Any] = type_configs.get(  # type: ignore[assignment]
            agent_type.lower(), type_configs["coder"]
        )
        return cls.from_dict(config_dict)


@dataclass
class Agent:
    """Represents an AI agent configuration.

    An agent is a specialized AI assistant with a specific model and optional
    description. Agents are model-agnostic and support ANY LiteLLM-compatible
    model (OpenAI, Anthropic, Ollama, local models, etc.).

    Attributes:
        name: Agent name (1-32 chars, alphanumeric+dashes)
        model_name: LiteLLM model string (e.g., "openai/gpt-4", "ollama/llama2")
        description: Optional agent description (max 256 chars)
        created_at: Agent creation timestamp (Unix seconds)
        last_used: Last usage timestamp (Unix seconds, None if never used)
        usage_count: Number of times agent has been used
        metadata: Extensible storage for agent-specific data
        max_completion_tokens: Maximum output tokens for LLM responses (default: 16000)
    """

    # Core agent identity (existing v1 fields)
    name: str
    model_name: str
    description: str = ""
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    last_used: Optional[float] = None
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_completion_tokens: int = (
        16000  # Maximum tokens for agent output (LiteLLM parameter)
    )

    # NEW: Orchestration fields (v2 schema for AgentChain integration)
    instruction_chain: List[Union[str, Dict[str, Any]]] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    history_config: Optional[HistoryConfig] = None

    def __post_init__(self):
        """Validate agent attributes after initialization."""
        # Validate name format
        if not self.name or len(self.name) > 32:
            raise ValueError(
                f"Agent name must be 1-32 characters, got: {len(self.name)}"
            )

        if not self.name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                f"Agent name must be alphanumeric+dashes/underscores: {self.name}"
            )

        # Validate model_name format (provider/model-name)
        if not self.model_name or "/" not in self.model_name:
            raise ValueError(
                f"model_name must be in format 'provider/model-name' "
                f"(e.g., 'openai/gpt-4'), got: {self.model_name}"
            )

        # Validate description length
        if len(self.description) > 256:
            raise ValueError(
                f"Description must be ≤256 characters, got: {len(self.description)}"
            )

    def update_usage(self):
        """Update usage statistics when agent is used."""
        self.usage_count += 1
        self.last_used = datetime.now().timestamp()

    @property
    def is_terminal_agent(self) -> bool:
        """Check if this is a terminal/execution agent with disabled history.

        Terminal agents don't need conversation history for token efficiency.

        Returns:
            bool: True if history is explicitly disabled
        """
        return self.history_config is not None and not self.history_config.enabled

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for SQLite storage.

        Returns:
            Dict[str, Any]: Agent data as dictionary
        """
        return {
            "name": self.name,
            "model_name": self.model_name,
            "description": self.description,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "usage_count": self.usage_count,
            "metadata_json": str(self.metadata),  # JSON serialization in SessionManager
            "max_completion_tokens": self.max_completion_tokens,
            "instruction_chain": self.instruction_chain,
            "tools": self.tools,
            "history_config": (
                self.history_config.to_dict() if self.history_config else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        """Create agent from dictionary.

        Args:
            data: Agent data dictionary from SQLite

        Returns:
            Agent: Reconstructed agent object
        """
        history_config_data = data.get("history_config")
        history_config = (
            HistoryConfig.from_dict(history_config_data)
            if history_config_data
            else None
        )

        return cls(
            name=data["name"],
            model_name=data["model_name"],
            description=data.get("description", ""),
            created_at=data.get("created_at", datetime.now().timestamp()),
            last_used=data.get("last_used"),
            usage_count=data.get("usage_count", 0),
            metadata=data.get("metadata", {}),
            max_completion_tokens=data.get("max_completion_tokens", 16000),
            instruction_chain=data.get("instruction_chain", []),
            tools=data.get("tools", []),
            history_config=history_config,
        )

    def __str__(self) -> str:
        """Human-readable agent representation.

        Returns:
            str: Formatted agent string
        """
        status = "[ACTIVE]" if self.last_used is not None else ""
        usage_info = (
            f"used {self.usage_count} times" if self.usage_count > 0 else "not yet used"
        )

        if self.description:
            return f"{self.name} ({self.model_name}) - {self.description} ({usage_info}) {status}"
        else:
            return f"{self.name} ({self.model_name}) ({usage_info}) {status}"
