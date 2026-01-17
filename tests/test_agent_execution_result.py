# test_agent_execution_result.py
"""Comprehensive tests for AgentExecutionResult dataclass and AgentChain metadata return."""

import pytest
import asyncio
from datetime import datetime
from promptchain.utils.agent_execution_result import AgentExecutionResult
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.promptchaining import PromptChain


class TestAgentExecutionResultDataclass:
    """Tests for the AgentExecutionResult dataclass itself."""

    def test_dataclass_creation_minimal(self):
        """Test creating AgentExecutionResult with minimal required fields."""
        start = datetime.now()
        end = datetime.now()

        result = AgentExecutionResult(
            response="Test response",
            agent_name="test_agent",
            execution_time_ms=100.5,
            start_time=start,
            end_time=end
        )

        assert result.response == "Test response"
        assert result.agent_name == "test_agent"
        assert result.execution_time_ms == 100.5
        assert result.start_time == start
        assert result.end_time == end
        # Check defaults
        assert result.router_decision is None
        assert result.router_steps == 0
        assert result.fallback_used is False
        assert result.agent_execution_metadata is None
        assert result.tools_called == []
        assert result.total_tokens is None
        assert result.cache_hit is False
        assert result.errors == []
        assert result.warnings == []

    def test_dataclass_creation_full(self):
        """Test creating AgentExecutionResult with all fields populated."""
        start = datetime.now()
        end = datetime.now()

        result = AgentExecutionResult(
            response="Full response",
            agent_name="full_agent",
            execution_time_ms=250.75,
            start_time=start,
            end_time=end,
            router_decision={"chosen_agent": "full_agent", "confidence": 0.95},
            router_steps=3,
            fallback_used=True,
            agent_execution_metadata={"step_count": 5},
            tools_called=[{"name": "calculator", "args": {"expr": "2+2"}, "result": "4"}],
            total_tokens=150,
            prompt_tokens=100,
            completion_tokens=50,
            cache_hit=True,
            cache_key="cache_123",
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"]
        )

        assert result.response == "Full response"
        assert result.agent_name == "full_agent"
        assert result.execution_time_ms == 250.75
        assert result.router_decision == {"chosen_agent": "full_agent", "confidence": 0.95}
        assert result.router_steps == 3
        assert result.fallback_used is True
        assert result.agent_execution_metadata == {"step_count": 5}
        assert len(result.tools_called) == 1
        assert result.tools_called[0]["name"] == "calculator"
        assert result.total_tokens == 150
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.cache_hit is True
        assert result.cache_key == "cache_123"
        assert len(result.errors) == 2
        assert len(result.warnings) == 1

    def test_to_dict_serialization(self):
        """Test to_dict() serialization method."""
        start = datetime.fromisoformat("2025-10-04T10:30:00")
        end = datetime.fromisoformat("2025-10-04T10:30:01")

        result = AgentExecutionResult(
            response="Dict test",
            agent_name="dict_agent",
            execution_time_ms=1000.0,
            start_time=start,
            end_time=end,
            router_steps=2,
            tools_called=[{"name": "tool1"}],
            errors=["error1"],
            warnings=["warn1"]
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["response"] == "Dict test"
        assert result_dict["agent_name"] == "dict_agent"
        assert result_dict["execution_time_ms"] == 1000.0
        assert result_dict["start_time"] == "2025-10-04T10:30:00"
        assert result_dict["end_time"] == "2025-10-04T10:30:01"
        assert result_dict["router_steps"] == 2
        assert result_dict["tools_called_count"] == 1
        assert result_dict["errors_count"] == 1
        assert result_dict["warnings_count"] == 1
        # Check that lists are included
        assert "tools_called" in result_dict
        assert "errors" in result_dict
        assert "warnings" in result_dict

    def test_to_summary_dict(self):
        """Test to_summary_dict() for condensed metrics."""
        start = datetime.now()
        end = datetime.now()

        result = AgentExecutionResult(
            response="Summary test response",
            agent_name="summary_agent",
            execution_time_ms=500.0,
            start_time=start,
            end_time=end,
            router_steps=1,
            tools_called=[{"name": "tool1"}, {"name": "tool2"}],
            total_tokens=200,
            cache_hit=True,
            errors=["error1"]
        )

        summary = result.to_summary_dict()

        assert isinstance(summary, dict)
        assert summary["agent_name"] == "summary_agent"
        assert summary["execution_time_ms"] == 500.0
        assert summary["router_steps"] == 1
        assert summary["tools_called_count"] == 2
        assert summary["total_tokens"] == 200
        assert summary["cache_hit"] is True
        assert summary["errors_count"] == 1
        assert summary["response_length"] == len("Summary test response")
        # Summary should NOT include full lists
        assert "tools_called" not in summary
        assert "errors" not in summary


class TestAgentChainMetadataReturn:
    """Tests for AgentChain.process_input() with return_metadata parameter."""

    @pytest.fixture
    def simple_agent_chain(self):
        """Create a simple AgentChain for testing."""
        # Create a simple agent
        simple_agent = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Respond with: {input}"],
            verbose=False
        )

        # Create AgentChain in pipeline mode (simplest for testing)
        agent_chain = AgentChain(
            agents={"simple": simple_agent},
            agent_descriptions={"simple": "A simple test agent"},
            execution_mode="pipeline",
            verbose=False
        )

        return agent_chain

    @pytest.mark.asyncio
    async def test_backward_compatibility_default_returns_string(self, simple_agent_chain):
        """Test that default behavior (return_metadata=False) returns string."""
        result = await simple_agent_chain.process_input("Hello")

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_return_metadata_true_returns_dataclass(self, simple_agent_chain):
        """Test that return_metadata=True returns AgentExecutionResult."""
        result = await simple_agent_chain.process_input("Hello", return_metadata=True)

        assert isinstance(result, AgentExecutionResult)
        assert isinstance(result.response, str)
        assert result.agent_name is not None
        assert result.execution_time_ms >= 0
        assert isinstance(result.start_time, datetime)
        assert isinstance(result.end_time, datetime)

    @pytest.mark.asyncio
    async def test_metadata_execution_time_calculation(self, simple_agent_chain):
        """Test that execution time is calculated correctly."""
        result = await simple_agent_chain.process_input("Test timing", return_metadata=True)

        assert isinstance(result, AgentExecutionResult)
        # Execution time should be positive
        assert result.execution_time_ms > 0
        # End time should be after start time
        assert result.end_time > result.start_time
        # Manual calculation should match
        manual_calc = (result.end_time - result.start_time).total_seconds() * 1000
        assert abs(result.execution_time_ms - manual_calc) < 1.0  # Allow 1ms tolerance

    @pytest.mark.asyncio
    async def test_metadata_contains_response(self, simple_agent_chain):
        """Test that metadata contains the actual response."""
        result = await simple_agent_chain.process_input("Test response", return_metadata=True)

        assert isinstance(result, AgentExecutionResult)
        assert result.response is not None
        assert len(result.response) > 0
        # Response should be accessible
        assert isinstance(result.response, str)

    @pytest.mark.asyncio
    async def test_metadata_agent_name_populated(self, simple_agent_chain):
        """Test that agent_name is populated in metadata."""
        result = await simple_agent_chain.process_input("Test agent", return_metadata=True)

        assert isinstance(result, AgentExecutionResult)
        # In pipeline mode with one agent, should be "simple"
        assert result.agent_name in ["simple", "unknown"]

    @pytest.mark.asyncio
    async def test_metadata_errors_list_initialized(self, simple_agent_chain):
        """Test that errors list is properly initialized (empty on success)."""
        result = await simple_agent_chain.process_input("Test errors", return_metadata=True)

        assert isinstance(result, AgentExecutionResult)
        assert isinstance(result.errors, list)
        # Should be empty on successful execution
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_metadata_warnings_list_initialized(self, simple_agent_chain):
        """Test that warnings list is properly initialized."""
        result = await simple_agent_chain.process_input("Test warnings", return_metadata=True)

        assert isinstance(result, AgentExecutionResult)
        assert isinstance(result.warnings, list)

    @pytest.mark.asyncio
    async def test_metadata_to_dict_integration(self, simple_agent_chain):
        """Test that AgentExecutionResult can be serialized to dict."""
        result = await simple_agent_chain.process_input("Test dict", return_metadata=True)

        assert isinstance(result, AgentExecutionResult)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "response" in result_dict
        assert "agent_name" in result_dict
        assert "execution_time_ms" in result_dict
        assert "start_time" in result_dict
        assert "end_time" in result_dict


class TestMetadataFieldPopulation:
    """Tests for specific metadata field population (TODOs in implementation)."""

    @pytest.mark.asyncio
    async def test_router_steps_initialized(self):
        """Test that router_steps is initialized to 0."""
        simple_agent = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Echo: {input}"],
            verbose=False
        )

        agent_chain = AgentChain(
            agents={"echo": simple_agent},
            agent_descriptions={"echo": "Echo agent"},
            execution_mode="pipeline",
            verbose=False
        )

        result = await agent_chain.process_input("Test", return_metadata=True)

        assert isinstance(result, AgentExecutionResult)
        assert result.router_steps == 0  # Should be 0 for pipeline mode

    @pytest.mark.asyncio
    async def test_fallback_used_initialized(self):
        """Test that fallback_used is initialized to False."""
        simple_agent = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Respond: {input}"],
            verbose=False
        )

        agent_chain = AgentChain(
            agents={"test": simple_agent},
            agent_descriptions={"test": "Test agent"},
            execution_mode="pipeline",
            verbose=False
        )

        result = await agent_chain.process_input("Test", return_metadata=True)

        assert isinstance(result, AgentExecutionResult)
        assert result.fallback_used is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
