"""
Comprehensive tests for AgenticStepProcessor metadata tracking.

Tests cover:
- StepExecutionMetadata dataclass creation and serialization
- AgenticStepResult dataclass creation and serialization
- Backward compatibility (return_metadata=False)
- Metadata return (return_metadata=True)
- Step tracking across multiple steps
- Tool call tracking within steps
- Token counting accuracy
- Error handling and tracking
"""

import pytest
import asyncio
from typing import List, Dict, Any
from datetime import datetime

from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.agentic_step_result import StepExecutionMetadata, AgenticStepResult


class TestStepExecutionMetadata:
    """Test StepExecutionMetadata dataclass."""

    def test_creation_with_defaults(self):
        """Test creating StepExecutionMetadata with default values."""
        metadata = StepExecutionMetadata(step_number=1)

        assert metadata.step_number == 1
        assert metadata.tool_calls == []
        assert metadata.tokens_used == 0
        assert metadata.execution_time_ms == 0.0
        assert metadata.clarification_attempts == 0
        assert metadata.error is None

    def test_creation_with_all_fields(self):
        """Test creating StepExecutionMetadata with all fields populated."""
        tool_calls = [
            {"name": "search", "args": {"query": "test"}, "result": "found", "time_ms": 100},
            {"name": "compute", "args": {"x": 5}, "result": "25", "time_ms": 50}
        ]

        metadata = StepExecutionMetadata(
            step_number=2,
            tool_calls=tool_calls,
            tokens_used=500,
            execution_time_ms=250.5,
            clarification_attempts=1,
            error="Test error"
        )

        assert metadata.step_number == 2
        assert len(metadata.tool_calls) == 2
        assert metadata.tokens_used == 500
        assert metadata.execution_time_ms == 250.5
        assert metadata.clarification_attempts == 1
        assert metadata.error == "Test error"

    def test_to_dict_conversion(self):
        """Test converting StepExecutionMetadata to dictionary."""
        tool_calls = [{"name": "test_tool", "args": {}, "result": "ok", "time_ms": 10}]
        metadata = StepExecutionMetadata(
            step_number=1,
            tool_calls=tool_calls,
            tokens_used=100,
            execution_time_ms=50.0
        )

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert result["step_number"] == 1
        assert result["tool_calls_count"] == 1
        assert result["tool_calls"] == tool_calls
        assert result["tokens_used"] == 100
        assert result["execution_time_ms"] == 50.0
        assert result["clarification_attempts"] == 0
        assert result["error"] is None


class TestAgenticStepResult:
    """Test AgenticStepResult dataclass."""

    def test_creation_minimal(self):
        """Test creating AgenticStepResult with minimal fields."""
        result = AgenticStepResult(
            final_answer="Test answer",
            total_steps=3,
            max_steps_reached=False,
            objective_achieved=True
        )

        assert result.final_answer == "Test answer"
        assert result.total_steps == 3
        assert result.max_steps_reached is False
        assert result.objective_achieved is True
        assert result.steps == []
        assert result.total_tools_called == 0
        assert result.total_tokens_used == 0

    def test_creation_complete(self):
        """Test creating AgenticStepResult with all fields."""
        steps = [
            StepExecutionMetadata(step_number=1, tokens_used=100),
            StepExecutionMetadata(step_number=2, tokens_used=150)
        ]

        result = AgenticStepResult(
            final_answer="Complete answer",
            total_steps=2,
            max_steps_reached=True,
            objective_achieved=False,
            steps=steps,
            total_tools_called=5,
            total_tokens_used=250,
            total_execution_time_ms=500.0,
            history_mode="progressive",
            max_internal_steps=10,
            model_name="gpt-4.1-mini-2025-04-14",
            errors=["Error 1"],
            warnings=["Warning 1"]
        )

        assert result.final_answer == "Complete answer"
        assert result.total_steps == 2
        assert len(result.steps) == 2
        assert result.total_tools_called == 5
        assert result.total_tokens_used == 250
        assert result.history_mode == "progressive"
        assert result.model_name == "gpt-4.1-mini-2025-04-14"
        assert len(result.errors) == 1
        assert len(result.warnings) == 1

    def test_to_dict_conversion(self):
        """Test converting AgenticStepResult to dictionary."""
        steps = [StepExecutionMetadata(step_number=1)]
        result = AgenticStepResult(
            final_answer="Test",
            total_steps=1,
            max_steps_reached=False,
            objective_achieved=True,
            steps=steps,
            total_tools_called=2,
            errors=["error1", "error2"]
        )

        dict_result = result.to_dict()

        assert isinstance(dict_result, dict)
        assert dict_result["final_answer"] == "Test"
        assert dict_result["total_steps"] == 1
        assert dict_result["total_tools_called"] == 2
        assert dict_result["errors_count"] == 2
        assert dict_result["errors"] == ["error1", "error2"]
        assert len(dict_result["steps"]) == 1

    def test_to_summary_dict(self):
        """Test converting AgenticStepResult to summary dictionary."""
        result = AgenticStepResult(
            final_answer="Test answer with long content",
            total_steps=3,
            max_steps_reached=True,
            objective_achieved=False,
            total_tools_called=7,
            total_tokens_used=500,
            total_execution_time_ms=1000.0,
            errors=["e1", "e2"],
            warnings=["w1"]
        )

        summary = result.to_summary_dict()

        assert isinstance(summary, dict)
        assert summary["total_steps"] == 3
        assert summary["tools_called"] == 7
        assert summary["execution_time_ms"] == 1000.0
        assert summary["objective_achieved"] is False
        assert summary["max_steps_reached"] is True
        assert summary["tokens_used"] == 500
        assert summary["errors_count"] == 2
        assert summary["warnings_count"] == 1
        # Summary should not include the full answer
        assert "final_answer" not in summary


class TestAgenticStepProcessorMetadata:
    """Test AgenticStepProcessor metadata tracking functionality."""

    @pytest.fixture
    def mock_llm_runner_simple(self):
        """Create a simple mock LLM runner that returns final answer immediately."""
        async def runner(messages, tools, tool_choice):
            return {"role": "assistant", "content": "Final answer"}
        return runner

    @pytest.fixture
    def mock_llm_runner_with_tools(self):
        """Create a mock LLM runner that makes tool calls."""
        call_count = 0

        async def runner(messages, tools, tool_choice):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: request a tool
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "search_tool", "arguments": '{"query": "test"}'}
                        }
                    ]
                }
            else:
                # Second call: return final answer
                return {"role": "assistant", "content": "Final answer after tool use"}

        return runner

    @pytest.fixture
    def mock_tool_executor(self):
        """Create a mock tool executor."""
        async def executor(tool_call):
            return "Tool execution result"
        return executor

    @pytest.mark.asyncio
    async def test_backward_compatibility_returns_string(self, mock_llm_runner_simple, mock_tool_executor):
        """Test that return_metadata=False returns just a string (backward compatible)."""
        processor = AgenticStepProcessor(
            objective="Test objective",
            max_internal_steps=3
        )

        result = await processor.run_async(
            initial_input="test input",
            available_tools=[],
            llm_runner=mock_llm_runner_simple,
            tool_executor=mock_tool_executor,
            return_metadata=False  # Explicit False
        )

        assert isinstance(result, str)
        assert result == "Final answer"

    @pytest.mark.asyncio
    async def test_default_returns_string(self, mock_llm_runner_simple, mock_tool_executor):
        """Test that default behavior returns string (backward compatible)."""
        processor = AgenticStepProcessor(
            objective="Test objective",
            max_internal_steps=3
        )

        result = await processor.run_async(
            initial_input="test input",
            available_tools=[],
            llm_runner=mock_llm_runner_simple,
            tool_executor=mock_tool_executor
            # return_metadata not specified, defaults to False
        )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_metadata_return_structure(self, mock_llm_runner_simple, mock_tool_executor):
        """Test that return_metadata=True returns AgenticStepResult."""
        processor = AgenticStepProcessor(
            objective="Test objective",
            max_internal_steps=3,
            model_name="test-model"
        )

        result = await processor.run_async(
            initial_input="test input",
            available_tools=[],
            llm_runner=mock_llm_runner_simple,
            tool_executor=mock_tool_executor,
            return_metadata=True
        )

        assert isinstance(result, AgenticStepResult)
        assert result.final_answer == "Final answer"
        assert result.total_steps >= 1
        assert result.model_name == "test-model"
        assert isinstance(result.steps, list)
        assert result.total_execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_step_tracking_single_step(self, mock_llm_runner_simple, mock_tool_executor):
        """Test that single-step execution is tracked correctly."""
        processor = AgenticStepProcessor(
            objective="Single step test",
            max_internal_steps=5
        )

        result = await processor.run_async(
            initial_input="input",
            available_tools=[],
            llm_runner=mock_llm_runner_simple,
            tool_executor=mock_tool_executor,
            return_metadata=True
        )

        assert len(result.steps) == 1
        assert result.steps[0].step_number == 1
        assert result.total_steps == 1
        assert result.max_steps_reached is False

    @pytest.mark.asyncio
    async def test_tool_call_tracking(self, mock_llm_runner_with_tools, mock_tool_executor):
        """Test that tool calls are tracked correctly."""
        processor = AgenticStepProcessor(
            objective="Tool tracking test",
            max_internal_steps=5
        )

        result = await processor.run_async(
            initial_input="input",
            available_tools=[{"type": "function", "function": {"name": "search_tool"}}],
            llm_runner=mock_llm_runner_with_tools,
            tool_executor=mock_tool_executor,
            return_metadata=True
        )

        # Should have 2 steps (one with tool call, one with final answer)
        assert len(result.steps) >= 1
        assert result.total_tools_called >= 1

        # First step should have tool calls
        first_step = result.steps[0]
        assert len(first_step.tool_calls) >= 1

        # Check tool call structure
        tool_call = first_step.tool_calls[0]
        assert "name" in tool_call
        assert "result" in tool_call
        assert "time_ms" in tool_call

    @pytest.mark.asyncio
    async def test_token_counting(self, mock_llm_runner_simple, mock_tool_executor):
        """Test that token counting works correctly."""
        processor = AgenticStepProcessor(
            objective="Token counting test",
            max_internal_steps=3
        )

        result = await processor.run_async(
            initial_input="A longer input string to test token counting",
            available_tools=[],
            llm_runner=mock_llm_runner_simple,
            tool_executor=mock_tool_executor,
            return_metadata=True
        )

        assert result.total_tokens_used > 0
        assert all(step.tokens_used >= 0 for step in result.steps)

    @pytest.mark.asyncio
    async def test_execution_time_tracking(self, mock_llm_runner_simple, mock_tool_executor):
        """Test that execution times are tracked."""
        processor = AgenticStepProcessor(
            objective="Timing test",
            max_internal_steps=2
        )

        result = await processor.run_async(
            initial_input="input",
            available_tools=[],
            llm_runner=mock_llm_runner_simple,
            tool_executor=mock_tool_executor,
            return_metadata=True
        )

        assert result.total_execution_time_ms > 0
        assert all(step.execution_time_ms >= 0 for step in result.steps)

        # Total time should be >= sum of step times
        step_times_sum = sum(step.execution_time_ms for step in result.steps)
        assert result.total_execution_time_ms >= step_times_sum * 0.9  # Allow some variance

    @pytest.mark.asyncio
    async def test_history_modes_in_metadata(self):
        """Test that history mode is captured in metadata."""
        for mode in ["minimal", "progressive", "kitchen_sink"]:
            processor = AgenticStepProcessor(
                objective="History mode test",
                max_internal_steps=2,
                history_mode=mode
            )

            async def simple_runner(messages, tools, tool_choice):
                return {"role": "assistant", "content": "Answer"}

            async def simple_executor(tool_call):
                return "Result"

            result = await processor.run_async(
                initial_input="test",
                available_tools=[],
                llm_runner=simple_runner,
                tool_executor=simple_executor,
                return_metadata=True
            )

            assert result.history_mode == mode

    @pytest.mark.asyncio
    async def test_error_tracking(self):
        """Test that errors are tracked in metadata."""
        processor = AgenticStepProcessor(
            objective="Error tracking test",
            max_internal_steps=3
        )

        async def failing_runner(messages, tools, tool_choice):
            raise ValueError("Test error")

        async def simple_executor(tool_call):
            return "Result"

        result = await processor.run_async(
            initial_input="test",
            available_tools=[],
            llm_runner=failing_runner,
            tool_executor=simple_executor,
            return_metadata=True
        )

        assert len(result.errors) > 0
        assert any("Error" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_comprehensive_metadata_tracking(self):
        """Test comprehensive metadata tracking across multiple steps."""
        processor = AgenticStepProcessor(
            objective="Multi-step test",
            max_internal_steps=3,
            model_name="test-model",
            history_mode="progressive"
        )

        call_count = [0]
        async def multi_step_runner(messages, tools, tool_choice):
            call_count[0] += 1
            # Step 1: Call tool
            if call_count[0] == 1:
                return {
                    "role": "assistant",
                    "tool_calls": [{"id": "call_1", "function": {"name": "tool1", "arguments": '{"x": 1}'}}]
                }
            # Step 2: After tool result, return final answer
            else:
                return {"role": "assistant", "content": "Final answer based on tool results"}

        async def simple_executor(tool_call):
            return "Tool execution result"

        result = await processor.run_async(
            initial_input="test input",
            available_tools=[
                {"type": "function", "function": {"name": "tool1"}},
                {"type": "function", "function": {"name": "tool2"}}
            ],
            llm_runner=multi_step_runner,
            tool_executor=simple_executor,
            return_metadata=True
        )

        # Verify comprehensive metadata
        assert isinstance(result, AgenticStepResult)
        assert result.final_answer == "Final answer based on tool results"
        assert result.total_steps >= 1
        assert result.total_tools_called >= 1
        assert result.total_tokens_used > 0
        assert result.total_execution_time_ms > 0
        assert result.model_name == "test-model"
        assert result.history_mode == "progressive"
        assert result.max_internal_steps == 3

        # Verify step-level metadata
        assert len(result.steps) >= 1
        first_step = result.steps[0]
        assert first_step.step_number == 1
        assert len(first_step.tool_calls) >= 1
        assert first_step.execution_time_ms > 0

        # Verify tool call metadata
        tool_call = first_step.tool_calls[0]
        assert "name" in tool_call
        assert "result" in tool_call
        assert "time_ms" in tool_call


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
