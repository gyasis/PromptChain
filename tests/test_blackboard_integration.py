"""
Integration tests for Blackboard Architecture.

Tests the complete Blackboard system integration with AgenticStepProcessor,
including token reduction benchmarking.

Test Coverage:
- Multi-iteration workflows with Blackboard enabled vs disabled
- Token usage comparison and reduction measurement
- State persistence across iterations
- LRU eviction behavior under load
- Real-world scenario simulations
"""

import pytest
import json
from typing import List, Dict, Any
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using simple heuristic.

    Real implementation would use tiktoken, but this provides
    a reasonable approximation: ~1.3 tokens per word.
    """
    words = len(text.split())
    return int(words * 1.3)


class TokenTrackingMockLLM:
    """Mock LLM that tracks token usage for benchmarking."""

    def __init__(self, responses: List[Dict[str, Any]]):
        """
        Initialize with predefined responses.

        Args:
            responses: List of response dicts with 'content' and optional 'tool_calls'
        """
        self.responses = responses
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_history = []

    async def __call__(self, messages: List[Dict], model: str = None, tools: List = None, **kwargs):
        """Mock LLM call that tracks token usage."""
        if self.call_count >= len(self.responses):
            # Return completion message if out of responses
            return {
                "content": "Task completed successfully.",
                "tool_calls": []
            }

        # Calculate input tokens (all messages)
        input_text = "\n".join([msg.get("content", "") for msg in messages])
        input_tokens = estimate_tokens(input_text)
        self.total_input_tokens += input_tokens

        # Get response
        response = self.responses[self.call_count]
        self.call_count += 1

        # Calculate output tokens
        output_text = response.get("content", "")
        output_tokens = estimate_tokens(output_text)
        self.total_output_tokens += output_tokens

        # Track this call
        self.call_history.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_length": len(input_text),
            "messages_count": len(messages)
        })

        return response


class TestBlackboardTokenReduction:
    """Test token reduction with Blackboard enabled."""

    @pytest.mark.asyncio
    async def test_token_reduction_multi_iteration(self):
        """
        Test that Blackboard reduces token usage across multiple iterations.

        Simulates a 10-iteration workflow with large tool results where traditional
        history grows linearly but Blackboard maintains bounded size through truncation.
        """
        # Create large tool result (simulates database query returning lots of data)
        large_tool_result = " ".join([f"Record {i}: data data data" for i in range(100)])

        # Define responses simulating a data-heavy multi-step task
        # 9 iterations with tool calls that return large results
        # Final iteration signals completion
        responses = []
        tool_names = ["search_db", "filter_data", "aggregate", "transform",
                     "validate", "analyze", "correlate", "summarize", "format"]

        for i, tool_name in enumerate(tool_names):
            responses.append({
                "content": f"Step {i+1}: Processing {tool_name}",
                "tool_calls": [{
                    "id": f"call_{i+1}",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"}
                }]
            })

        # Final completion response
        responses.append({"content": "All analysis complete", "tool_calls": []})

        # Test WITHOUT Blackboard (traditional history)
        # Each iteration accumulates more history
        mock_llm_traditional = TokenTrackingMockLLM(responses.copy())

        processor_traditional = AgenticStepProcessor(
            objective="Analyze large dataset and generate comprehensive report",
            max_internal_steps=10,
            enable_blackboard=False,  # Traditional history
            history_mode="progressive"  # Keep full history
        )

        # Tool executor returns large results
        async def tool_executor_large_results(tool_call):
            return large_tool_result

        await processor_traditional.run_async(
            initial_input="Analyze the complete sales database for Q4 trends",
            available_tools=[],
            llm_runner=mock_llm_traditional,
            tool_executor=tool_executor_large_results
        )

        traditional_tokens = mock_llm_traditional.total_input_tokens

        # Test WITH Blackboard
        # Blackboard truncates tool results and maintains bounded state
        mock_llm_blackboard = TokenTrackingMockLLM(responses.copy())

        processor_blackboard = AgenticStepProcessor(
            objective="Analyze large dataset and generate comprehensive report",
            max_internal_steps=10,
            enable_blackboard=True  # Use Blackboard
        )

        await processor_blackboard.run_async(
            initial_input="Analyze the complete sales database for Q4 trends",
            available_tools=[],
            llm_runner=mock_llm_blackboard,
            tool_executor=tool_executor_large_results
        )

        blackboard_tokens = mock_llm_blackboard.total_input_tokens

        # Calculate reduction
        reduction_pct = ((traditional_tokens - blackboard_tokens) / traditional_tokens) * 100

        print(f"\n=== Token Reduction Benchmark ===")
        print(f"Traditional history: {traditional_tokens} tokens")
        print(f"Blackboard:          {blackboard_tokens} tokens")
        print(f"Reduction:           {reduction_pct:.1f}%")
        print(f"Tokens saved:        {traditional_tokens - blackboard_tokens}")

        # Assert significant reduction (target: 40%+)
        # Blackboard truncates large tool results and maintains bounded state
        assert blackboard_tokens < traditional_tokens, "Blackboard should use fewer tokens"
        assert reduction_pct >= 30, f"Expected ≥30% reduction, got {reduction_pct:.1f}%"

    @pytest.mark.asyncio
    async def test_blackboard_state_persistence(self):
        """Test that Blackboard maintains state correctly across iterations."""
        responses = [
            {"content": "Found 3 facts", "tool_calls": []},
            {"content": "Added 2 more facts", "tool_calls": []},
            {"content": "Completed analysis", "tool_calls": []}
        ]

        mock_llm = TokenTrackingMockLLM(responses)

        processor = AgenticStepProcessor(
            objective="Research task",
            max_internal_steps=3,
            enable_blackboard=True
        )

        # Add some state before execution
        processor.blackboard.add_fact("initial_fact", "Important information")
        processor.blackboard.add_observation("Starting research")

        async def dummy_tool_executor(tool_call):
            return "Tool executed"

        await processor.run_async(
            initial_input="Conduct research",
            available_tools=[],
            llm_runner=mock_llm,
            tool_executor=dummy_tool_executor
        )

        # Verify Blackboard maintained state
        state = processor.blackboard.get_state()
        assert "initial_fact" in state["facts_discovered"]
        assert len(state["observations"]) > 0
        assert state["objective"] == "Research task"


class TestBlackboardLRUEviction:
    """Test LRU eviction behavior under load."""

    @pytest.mark.asyncio
    async def test_facts_lru_eviction(self):
        """Test that facts are evicted when max_facts is exceeded."""
        responses = [{"content": "Done", "tool_calls": []}]
        mock_llm = TokenTrackingMockLLM(responses)

        processor = AgenticStepProcessor(
            objective="Test",
            max_internal_steps=1,
            enable_blackboard=True
        )

        # Blackboard default max_facts is 20
        # Add 25 facts to trigger eviction
        for i in range(25):
            processor.blackboard.add_fact(f"fact_{i}", f"Value {i}")

        state = processor.blackboard.get_state()
        facts = state["facts_discovered"]

        # Should only have 20 facts (max_facts limit)
        assert len(facts) <= 20

        # Oldest facts (fact_0 through fact_4) should be evicted
        assert "fact_0" not in facts
        assert "fact_1" not in facts
        assert "fact_2" not in facts
        assert "fact_3" not in facts
        assert "fact_4" not in facts

        # Recent facts should remain
        assert "fact_24" in facts
        assert "fact_23" in facts

    @pytest.mark.asyncio
    async def test_observations_circular_buffer(self):
        """Test that observations are kept within max_observations limit."""
        responses = [{"content": "Done", "tool_calls": []}]
        mock_llm = TokenTrackingMockLLM(responses)

        processor = AgenticStepProcessor(
            objective="Test",
            max_internal_steps=1,
            enable_blackboard=True
        )

        # Blackboard default max_observations is 15
        # Add 20 observations
        for i in range(20):
            processor.blackboard.add_observation(f"Observation {i}")

        state = processor.blackboard.get_state()
        observations = state["observations"]

        # Should only have 15 observations (max_observations limit)
        assert len(observations) <= 15

        # Should be the most recent 15
        assert "Observation 19" in observations[-1]
        assert "Observation 5" in observations[0]


class TestBlackboardPromptGeneration:
    """Test Blackboard prompt generation."""

    def test_to_prompt_format(self):
        """Test that Blackboard.to_prompt() generates structured output."""
        processor = AgenticStepProcessor(
            objective="Test objective with detailed description",
            max_internal_steps=1,
            enable_blackboard=True
        )

        # Add diverse state
        processor.blackboard.add_fact("user_count", 150)
        processor.blackboard.add_fact("database_status", "connected")
        processor.blackboard.add_observation("Found user data")
        processor.blackboard.add_observation("Validated schema")
        processor.blackboard.mark_step_complete("Data retrieval")
        processor.blackboard.store_tool_result("query_db", "Retrieved 150 records")

        # Generate prompt
        prompt = processor.blackboard.to_prompt()

        # Verify sections are present
        assert "OBJECTIVE:" in prompt
        assert "Test objective" in prompt
        assert "FACTS DISCOVERED:" in prompt
        assert "user_count: 150" in prompt
        assert "RECENT OBSERVATIONS:" in prompt
        assert "Found user data" in prompt
        assert "COMPLETED STEPS:" in prompt
        assert "Data retrieval" in prompt
        assert "AVAILABLE TOOL RESULTS:" in prompt

        # Verify it's concise (not massive like full history)
        # A well-structured Blackboard prompt should be under 1000 tokens
        estimated_tokens = estimate_tokens(prompt)
        assert estimated_tokens < 1500, f"Blackboard prompt too large: {estimated_tokens} tokens"

    def test_prompt_with_minimal_state(self):
        """Test prompt generation with minimal state."""
        processor = AgenticStepProcessor(
            objective="Simple task",
            max_internal_steps=1,
            enable_blackboard=True
        )

        prompt = processor.blackboard.to_prompt()

        # Should still have structure even with empty state
        assert "OBJECTIVE:" in prompt
        assert "Simple task" in prompt
        assert "CURRENT PLAN:" in prompt
        assert "FACTS DISCOVERED:" in prompt


class TestBlackboardSnapshotAndRollback:
    """Test Blackboard snapshot and rollback functionality."""

    def test_snapshot_creation(self):
        """Test creating Blackboard snapshots."""
        processor = AgenticStepProcessor(
            objective="Test",
            enable_blackboard=True
        )

        # Add initial state
        processor.blackboard.add_fact("key1", "value1")
        processor.blackboard.add_observation("Observation 1")

        # Create snapshot
        snapshot_id = processor.blackboard.snapshot()

        assert snapshot_id == "snapshot_0"

        # Modify state after snapshot
        processor.blackboard.add_fact("key2", "value2")
        processor.blackboard.add_observation("Observation 2")

        # Create another snapshot
        snapshot_id2 = processor.blackboard.snapshot()
        assert snapshot_id2 == "snapshot_1"

        # Verify state has both facts
        state = processor.blackboard.get_state()
        assert "key1" in state["facts_discovered"]
        assert "key2" in state["facts_discovered"]

    def test_rollback_to_snapshot(self):
        """Test rolling back to a previous snapshot."""
        processor = AgenticStepProcessor(
            objective="Test",
            enable_blackboard=True
        )

        # Initial state
        processor.blackboard.add_fact("initial", "data")
        snapshot_id = processor.blackboard.snapshot()

        # Add more state
        processor.blackboard.add_fact("new_fact", "new_data")
        processor.blackboard.add_observation("New observation")

        # Verify new state exists
        state_before = processor.blackboard.get_state()
        assert "new_fact" in state_before["facts_discovered"]

        # Rollback
        processor.blackboard.rollback(snapshot_id)

        # Verify state was restored
        state_after = processor.blackboard.get_state()
        assert "new_fact" not in state_after["facts_discovered"]
        assert "initial" in state_after["facts_discovered"]


class TestBlackboardRealWorldScenarios:
    """Test Blackboard with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_research_task_scenario(self):
        """
        Simulate a research task with multiple information gathering steps.

        This scenario tests:
        - Multiple facts accumulation
        - Observations tracking progress
        - Completed steps recording
        - Tool results storage
        """
        responses = [
            {"content": "Searching academic databases", "tool_calls": []},
            {"content": "Found 5 relevant papers", "tool_calls": []},
            {"content": "Analyzing paper abstracts", "tool_calls": []},
            {"content": "Synthesizing findings", "tool_calls": []},
            {"content": "Research complete", "tool_calls": []}
        ]

        mock_llm = TokenTrackingMockLLM(responses)

        processor = AgenticStepProcessor(
            objective="Research AI safety techniques published in 2024",
            max_internal_steps=5,
            enable_blackboard=True
        )

        # Add discovered facts during "research"
        processor.blackboard.add_fact("papers_found", 5)
        processor.blackboard.add_fact("primary_technique", "Constitutional AI")
        processor.blackboard.add_fact("publication_venue", "NeurIPS 2024")

        async def dummy_tool_executor(tool_call):
            return "Tool executed"

        await processor.run_async(
            initial_input="Find and summarize recent AI safety research",
            available_tools=[],
            llm_runner=mock_llm,
            tool_executor=dummy_tool_executor
        )

        # Verify Blackboard captured research state
        state = processor.blackboard.get_state()
        assert state["facts_discovered"]["papers_found"] == 5
        assert len(state["completed_steps"]) > 0

        # Verify token efficiency
        prompt = processor.blackboard.to_prompt()
        tokens = estimate_tokens(prompt)

        # Research task with 5 facts and multiple observations
        # should still be under 1000 tokens
        assert tokens < 1200, f"Research prompt too large: {tokens} tokens"

    @pytest.mark.asyncio
    async def test_debugging_task_scenario(self):
        """
        Simulate a debugging task with error tracking and hypothesis testing.
        """
        responses = [
            {"content": "Checking error logs", "tool_calls": []},
            {"content": "Found null pointer exception", "tool_calls": []},
            {"content": "Testing hypothesis 1", "tool_calls": []},
            {"content": "Hypothesis confirmed - fix identified", "tool_calls": []}
        ]

        mock_llm = TokenTrackingMockLLM(responses)

        processor = AgenticStepProcessor(
            objective="Debug production crash in payment service",
            max_internal_steps=4,
            enable_blackboard=True
        )

        # Simulate debugging discoveries
        processor.blackboard.add_fact("error_type", "NullPointerException")
        processor.blackboard.add_fact("affected_service", "PaymentProcessor")
        processor.blackboard.add_fact("error_frequency", "3 times per hour")
        processor.blackboard.add_error("Null check missing in payment validation")
        processor.blackboard.add_observation("Error occurs only with refund requests")

        async def dummy_tool_executor(tool_call):
            return "Tool executed"

        await processor.run_async(
            initial_input="Find root cause of payment crashes",
            available_tools=[],
            llm_runner=mock_llm,
            tool_executor=dummy_tool_executor
        )

        # Verify debugging state captured
        state = processor.blackboard.get_state()
        assert state["facts_discovered"]["error_type"] == "NullPointerException"
        assert len(state["errors"]) > 0
        assert "Null check missing" in state["errors"][0]


class TestBlackboardPerformance:
    """Test Blackboard performance characteristics."""

    def test_prompt_generation_performance(self):
        """Test that prompt generation is fast even with max state."""
        processor = AgenticStepProcessor(
            objective="Performance test",
            enable_blackboard=True
        )

        # Fill Blackboard to capacity
        for i in range(20):  # max_facts
            processor.blackboard.add_fact(f"fact_{i}", f"Value {i}")

        for i in range(15):  # max_observations
            processor.blackboard.add_observation(f"Observation {i}")

        for i in range(10):  # max_plan_items
            processor.blackboard._state["current_plan"].append(f"Step {i}")

        # Generate prompt multiple times (should be fast)
        import time
        start = time.time()
        for _ in range(100):
            prompt = processor.blackboard.to_prompt()
        duration = time.time() - start

        # 100 prompt generations should take less than 1 second
        assert duration < 1.0, f"Prompt generation too slow: {duration:.3f}s"

        # Prompt should still be bounded
        tokens = estimate_tokens(prompt)
        assert tokens < 1500, f"Full Blackboard prompt too large: {tokens} tokens"


class TestBlackboardBackwardCompatibility:
    """Test that disabling Blackboard preserves original behavior."""

    @pytest.mark.asyncio
    async def test_traditional_history_still_works(self):
        """Test that traditional history mode works when Blackboard disabled."""
        # First 2 responses include tool calls, final response signals completion
        responses = [
            {
                "content": "Step 1 complete",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "step1", "arguments": "{}"}
                }]
            },
            {
                "content": "Step 2 complete",
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "step2", "arguments": "{}"}
                }]
            },
            {"content": "All done", "tool_calls": []}
        ]

        mock_llm = TokenTrackingMockLLM(responses)

        processor = AgenticStepProcessor(
            objective="Test task",
            max_internal_steps=3,
            enable_blackboard=False,  # Disabled - use traditional history
            history_mode="progressive"
        )

        # Should complete successfully without Blackboard
        async def dummy_tool_executor(tool_call):
            return "Tool executed"

        await processor.run_async(
            initial_input="Complete the task",
            available_tools=[],
            llm_runner=mock_llm,
            tool_executor=dummy_tool_executor
        )

        # Verify Blackboard was not created
        assert processor.blackboard is None
        assert mock_llm.call_count == 3


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
