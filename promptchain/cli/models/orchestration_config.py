"""Orchestration configuration models for AgentChain execution.

This module defines configuration models for AgentChain execution modes,
router decision-making, and global history management settings.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


# Default router decision prompt template
DEFAULT_ROUTER_PROMPT = """User query: {user_input}

Available agents:
{agent_details}

Conversation history:
{history}

Based on the user's request and conversation context, choose the most appropriate agent.
Consider agent specializations and the nature of the task.

Return JSON response:
{{"chosen_agent": "agent_name", "refined_query": "optional refined query"}}
"""


@dataclass
class RouterConfig:
    """Router decision prompt and model configuration.

    RouterConfig controls how AgentChain makes routing decisions when in
    router execution mode, including model selection and decision prompts.

    Attributes:
        model: Fast LLM model for routing decisions (default: gpt-4o-mini)
        decision_prompt_template: Jinja2 template for routing decisions
        timeout_seconds: Maximum time for routing decision (1-60 seconds)
    """

    model: str = "openai/gpt-4o-mini"
    decision_prompt_template: str = DEFAULT_ROUTER_PROMPT
    timeout_seconds: int = 10

    def __post_init__(self):
        """Validate router configuration parameters."""
        if not self.model or len(self.model) < 3:
            raise ValueError(f"Invalid model name: {self.model}")

        if self.timeout_seconds < 1 or self.timeout_seconds > 60:
            raise ValueError(f"timeout_seconds must be 1-60, got: {self.timeout_seconds}")

        # Validate prompt template has required variables
        required_vars = ["{user_input}", "{agent_details}"]
        for var in required_vars:
            if var not in self.decision_prompt_template:
                raise ValueError(
                    f"decision_prompt_template missing required variable: {var}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage.

        Returns:
            Dict[str, Any]: Router config as dictionary
        """
        return {
            "model": self.model,
            "decision_prompt_template": self.decision_prompt_template,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouterConfig":
        """Create from dictionary.

        Args:
            data: Router config dictionary

        Returns:
            RouterConfig: Reconstructed router config
        """
        return cls(
            model=data.get("model", "openai/gpt-4o-mini"),
            decision_prompt_template=data.get("decision_prompt_template", DEFAULT_ROUTER_PROMPT),
            timeout_seconds=data.get("timeout_seconds", 10),
        )


@dataclass
class OrchestrationConfig:
    """AgentChain execution mode and router settings.

    OrchestrationConfig defines how AgentChain manages multiple agents,
    including execution mode, default fallback agent, and global history settings.

    Attributes:
        execution_mode: Agent coordination mode (router/pipeline/round-robin/broadcast)
        default_agent: Fallback agent name if routing fails
        router_config: Router configuration (required if execution_mode="router")
        auto_include_history: Global setting for conversation history inclusion
    """

    execution_mode: Literal["router", "pipeline", "round-robin", "broadcast"] = "router"
    default_agent: Optional[str] = None
    router_config: Optional[RouterConfig] = None
    auto_include_history: bool = True

    def __post_init__(self):
        """Validate orchestration configuration."""
        # Router mode requires router config
        if self.execution_mode == "router" and self.router_config is None:
            self.router_config = RouterConfig()  # Use default router config

        # Validate execution mode
        valid_modes = ["router", "pipeline", "round-robin", "broadcast"]
        if self.execution_mode not in valid_modes:
            raise ValueError(
                f"execution_mode must be one of {valid_modes}, got: {self.execution_mode}"
            )

    def validate_default_agent(self, available_agents: list[str]):
        """Validate that default_agent references an existing agent.

        Args:
            available_agents: List of available agent names

        Raises:
            ValueError: If default_agent not in available_agents
        """
        if self.default_agent and self.default_agent not in available_agents:
            raise ValueError(
                f"default_agent '{self.default_agent}' not found in available agents: {available_agents}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage.

        Returns:
            Dict[str, Any]: Orchestration config as dictionary
        """
        return {
            "execution_mode": self.execution_mode,
            "default_agent": self.default_agent,
            "router_config": self.router_config.to_dict() if self.router_config else None,
            "auto_include_history": self.auto_include_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestrationConfig":
        """Create from dictionary.

        Args:
            data: Orchestration config dictionary

        Returns:
            OrchestrationConfig: Reconstructed orchestration config
        """
        router_config = None
        if data.get("router_config"):
            router_config = RouterConfig.from_dict(data["router_config"])

        return cls(
            execution_mode=data.get("execution_mode", "router"),
            default_agent=data.get("default_agent"),
            router_config=router_config,
            auto_include_history=data.get("auto_include_history", True),
        )

    def __str__(self) -> str:
        """Human-readable configuration representation.

        Returns:
            str: Formatted config string
        """
        mode_emoji = {
            "router": "🔀",
            "pipeline": "⚡",
            "round-robin": "🔄",
            "broadcast": "📢",
        }
        emoji = mode_emoji.get(self.execution_mode, "⚙️")

        default_info = f" (default: {self.default_agent})" if self.default_agent else ""
        history_status = "enabled" if self.auto_include_history else "disabled"

        return f"{emoji} {self.execution_mode}{default_info} | history: {history_status}"
