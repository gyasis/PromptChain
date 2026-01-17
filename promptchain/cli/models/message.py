"""Message model for PromptChain CLI conversation history.

This module defines the Message data model representing a single conversation
exchange with JSONL storage format.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Message:
    """Represents a conversation exchange.

    A message captures a single turn in the conversation, including the role
    (user, assistant, system, tool), content, and optional metadata.

    Attributes:
        role: Message role ('user', 'assistant', 'system', 'tool')
        content: Message content text
        timestamp: Message creation timestamp (Unix seconds)
        metadata: Optional metadata (file references, command executed, etc.)
        agent_name: Name of agent that generated this message (for assistant messages)
        model_name: Model used to generate this message (for assistant messages)
    """

    role: str
    content: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_name: Optional[str] = None
    model_name: Optional[str] = None

    VALID_ROLES = {"user", "assistant", "system", "tool"}

    def __post_init__(self):
        """Validate message attributes after initialization."""
        # Validate role
        if self.role not in self.VALID_ROLES:
            raise ValueError(
                f"Invalid message role: {self.role}. "
                f"Must be one of: {', '.join(self.VALID_ROLES)}"
            )

        # Validate content is not empty
        if not self.content or not isinstance(self.content, str):
            raise ValueError("Message content must be a non-empty string")

    def to_jsonl(self) -> str:
        """Convert message to JSONL format for storage.

        JSONL format stores one JSON object per line, making it easy to append
        messages without reading the entire file.

        Returns:
            str: JSON string representation (single line, no trailing newline)
        """
        data = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

        # Add optional fields only if present
        if self.agent_name:
            data["agent_name"] = self.agent_name

        if self.model_name:
            data["model_name"] = self.model_name

        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "Message":
        """Create message from JSONL line.

        Args:
            line: JSON string from JSONL file

        Returns:
            Message: Reconstructed message object
        """
        data = json.loads(line.strip())

        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now().timestamp()),
            metadata=data.get("metadata", {}),
            agent_name=data.get("agent_name"),
            model_name=data.get("model_name"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary.

        Returns:
            Dict[str, Any]: Message data as dictionary
        """
        result = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

        if self.agent_name:
            result["agent_name"] = self.agent_name

        if self.model_name:
            result["model_name"] = self.model_name

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary.

        Args:
            data: Message data dictionary

        Returns:
            Message: Reconstructed message object
        """
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now().timestamp()),
            metadata=data.get("metadata", {}),
            agent_name=data.get("agent_name"),
            model_name=data.get("model_name"),
        )

    def __str__(self) -> str:
        """Human-readable message representation.

        Returns:
            str: Formatted message string
        """
        timestamp_str = datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")

        prefix = f"[{timestamp_str}] {self.role.upper()}"

        if self.agent_name:
            prefix += f" ({self.agent_name})"

        return f"{prefix}: {self.content[:100]}{'...' if len(self.content) > 100 else ''}"
