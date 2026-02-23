# agent_execution_result.py
"""Dataclass for comprehensive execution metadata from AgentChain."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List


@dataclass
class AgentExecutionResult:
    """Complete execution metadata from AgentChain.

    This dataclass captures comprehensive execution information when
    AgentChain.process_input() is called with return_metadata=True.

    Attributes:
        response: The final response string from the agent
        agent_name: Name of the agent that generated the response
        execution_time_ms: Total execution time in milliseconds
        start_time: Timestamp when execution started
        end_time: Timestamp when execution completed
        router_decision: Optional dictionary containing router decision details
        router_steps: Number of routing/orchestrator reasoning steps executed
                     (When OrchestratorSupervisor is used, this reflects the number
                     of output_reasoning_step tool calls during multi-hop reasoning)
        fallback_used: Whether fallback mechanism was used
        agent_execution_metadata: Optional metadata from the executing agent
        tools_called: List of tool call details (dicts with name, args, results)
        total_tokens: Total tokens consumed (if available)
        prompt_tokens: Prompt tokens consumed (if available)
        completion_tokens: Completion tokens consumed (if available)
        cache_hit: Whether the response came from cache
        cache_key: Optional cache key used
        errors: List of error messages encountered during execution
        warnings: List of warning messages encountered during execution
    """

    # Response data
    response: str
    agent_name: str

    # Execution metadata
    execution_time_ms: float
    start_time: datetime
    end_time: datetime

    # Routing information
    router_decision: Optional[Dict[str, Any]] = None
    router_steps: int = 0
    fallback_used: bool = False

    # Agent execution details
    agent_execution_metadata: Optional[Dict[str, Any]] = None
    tools_called: List[Dict[str, Any]] = field(default_factory=list)

    # Token usage
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # Cache information
    cache_hit: bool = False
    cache_key: Optional[str] = None

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and serialization.

        Returns:
            Dictionary representation with ISO formatted timestamps
        """
        return {
            "response": self.response,
            "agent_name": self.agent_name,
            "execution_time_ms": self.execution_time_ms,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "router_decision": self.router_decision,
            "router_steps": self.router_steps,
            "fallback_used": self.fallback_used,
            "agent_execution_metadata": self.agent_execution_metadata,
            "tools_called_count": len(self.tools_called),
            "tools_called": self.tools_called,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hit": self.cache_hit,
            "cache_key": self.cache_key,
            "errors_count": len(self.errors),
            "errors": self.errors,
            "warnings_count": len(self.warnings),
            "warnings": self.warnings
        }

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to a summary dictionary with key metrics only.

        Returns:
            Condensed dictionary with essential execution metrics
        """
        return {
            "agent_name": self.agent_name,
            "execution_time_ms": self.execution_time_ms,
            "router_steps": self.router_steps,
            "fallback_used": self.fallback_used,
            "tools_called_count": len(self.tools_called),
            "total_tokens": self.total_tokens,
            "cache_hit": self.cache_hit,
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "response_length": len(self.response)
        }
