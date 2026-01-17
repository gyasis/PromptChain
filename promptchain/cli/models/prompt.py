"""
Data models for prompt management system.

This module defines the data structures for prompts, strategies, and agent templates
used by the CLI prompt management system.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class Prompt:
    """Represents a prompt template.

    Attributes:
        id: Unique identifier (filename without extension)
        content: Full prompt text content
        category: Directory category (e.g., 'agents', 'custom')
        description: Brief description extracted from file
        path: Full file path to prompt file
        strategies: List of compatible strategy IDs
    """
    id: str
    content: str
    category: str
    description: Optional[str] = None
    path: str = ""
    strategies: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """String representation for display."""
        desc = self.description or "No description"
        return f"{self.id} ({self.category}): {desc}"


@dataclass
class Strategy:
    """Represents a prompt strategy (instruction prefix).

    Strategies are reusable instruction templates that can be prepended
    to base prompts to modify behavior (e.g., 'concise', 'detailed').

    Attributes:
        id: Unique identifier (filename without extension)
        name: Display name for the strategy
        prompt: Strategy instruction text
        description: Brief description of strategy behavior
    """
    id: str
    name: str
    prompt: str
    description: Optional[str] = None

    def __str__(self) -> str:
        """String representation for display."""
        desc = self.description or "No description"
        return f"{self.name} ({self.id}): {desc}"


@dataclass
class AgentTemplate:
    """Represents a pre-configured agent template.

    Templates provide quick agent setup with predefined prompts,
    models, and strategies for common use cases.

    Attributes:
        name: Template display name
        model: LLM model name (e.g., 'gpt-4', 'claude-3-opus')
        prompt_id: ID of associated prompt template
        strategy: Optional strategy ID to apply
        description: Template description
        tags: Searchable tags for categorization
    """
    name: str
    model: str
    prompt_id: str
    description: str
    strategy: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """String representation for display."""
        tags_str = ", ".join(self.tags) if self.tags else "no tags"
        return f"{self.name} ({self.model}): {self.description} [{tags_str}]"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "model": self.model,
            "prompt_id": self.prompt_id,
            "strategy": self.strategy,
            "description": self.description,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentTemplate":
        """Create AgentTemplate from dictionary.

        Args:
            data: Dictionary with template data

        Returns:
            AgentTemplate instance
        """
        return cls(
            name=data["name"],
            model=data["model"],
            prompt_id=data["prompt_id"],
            description=data["description"],
            strategy=data.get("strategy"),
            tags=data.get("tags", [])
        )
