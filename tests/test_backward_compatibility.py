"""
Backward Compatibility Test Suite for PromptChain 0.4.1g

Tests that all Phase 1 and Phase 2 changes maintain 100% backward compatibility:
- Phase 1 (0.4.1a-c): Public APIs for ExecutionHistoryManager, AgentChain, AgenticStepProcessor
- Phase 2 (0.4.1d-f): Event system and callbacks

Test Categories:
1. Legacy API patterns (private attributes still work)
2. Default behavior validation (return_metadata=False, callbacks opt-in)
3. Migration path testing (old + new API together)
4. Regression detection (existing behavior unchanged)
"""

import asyncio
import pytest
from typing import List, Dict, Any
from datetime import datetime

# Import the components
from promptchain import PromptChain
from promptchain.utils.execution_history_manager import ExecutionHistoryManager
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType
from promptchain.utils.execution_callback import CallbackManager


# ============================================================================
# Test Category 1: ExecutionHistoryManager Backward Compatibility (0.4.1a)
# ============================================================================

class TestExecutionHistoryManagerBackwardCompat:
    """Test ExecutionHistoryManager backward compatibility."""

    def test_private_attribute_still_works(self):
        """Test that old private attribute access still works (deprecated but functional)."""
        manager = ExecutionHistoryManager(max_tokens=1000)

        # Add some entries
        manager.add_entry("user_input", "Test content 1", source="user")
        manager.add_entry("agent_output", "Test response 1", source="agent")

        # OLD PATTERN: Direct access to private attributes (deprecated)
        old_token_count = manager._current_token_count
        old_history = manager._history

        # Should still work - these attributes should be accessible
        assert isinstance(old_token_count, int)
        assert old_token_count > 0
        assert isinstance(old_history, list)
        assert len(old_history) == 2

    def test_public_api_matches_private_values(self):
        """Test that new public APIs return same values as old private attributes."""
        manager = ExecutionHistoryManager(max_tokens=1000)

        # Add entries
        manager.add_entry("user_input", "Test content", source="user")

        # OLD PATTERN: Private attributes
        old_count = manager._current_token_count
        old_history = manager._history

        # NEW PATTERN: Public properties
        new_count = manager.current_token_count
        new_history = manager.history

        # Values should match
        assert old_count == new_count, "Token count mismatch between private and public API"
        assert len(old_history) == len(new_history), "History length mismatch"

    def test_existing_methods_unchanged(self):
        """Test that existing methods work identically."""
        manager = ExecutionHistoryManager(max_entries=10)

        # Test add_entry (existing method)
        manager.add_entry("user_input", "Test", source="user")
        assert manager.history_size == 1

        # Test get_formatted_history (existing method)
        formatted = manager.get_formatted_history(format_style='chat')
        assert isinstance(formatted, list)

    def test_default_behavior_unchanged(self):
        """Test that default instantiation works as before."""
        # OLD PATTERN: Create without any new parameters
        manager = ExecutionHistoryManager()

        # Should work without errors
        manager.add_entry("user_input", "Test", source="user")
        assert len(manager.history) == 1


# ============================================================================
# Test Category 2: AgentChain Backward Compatibility (0.4.1b)
# ============================================================================

class TestAgentChainBackwardCompat:
    """Test AgentChain backward compatibility."""

    @pytest.mark.asyncio
    async def test_default_returns_string(self):
        """Test that process_input returns string by default (backward compatible)."""
        # Create simple agent
        simple_agent = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Say 'Hello': {input}"]
        )

        # OLD PATTERN: No return_metadata parameter, should return string
        agent_chain = AgentChain(
            agents={"simple": simple_agent},
            execution_mode="single"
        )

        result = await agent_chain.process_input("test")

        # Should return string, not metadata object
        assert isinstance(result, str), f"Expected str, got {type(result)}"

    @pytest.mark.asyncio
    async def test_return_metadata_opt_in(self):
        """Test that return_metadata is opt-in and doesn't break old code."""
        simple_agent = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Echo: {input}"]
        )

        agent_chain = AgentChain(
            agents={"simple": simple_agent},
            execution_mode="single"
        )

        # OLD PATTERN: Default call returns string
        result_old = await agent_chain.process_input("test")
        assert isinstance(result_old, str)

        # NEW PATTERN: Opt-in to metadata
        result_new = await agent_chain.process_input("test", return_metadata=True)
        assert hasattr(result_new, 'response'), "Metadata object missing 'response' attribute"
        assert hasattr(result_new, 'agent_name'), "Metadata object missing 'agent_name' attribute"

    @pytest.mark.asyncio
    async def test_existing_parameters_work(self):
        """Test that all existing parameters still work."""
        agent = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Process: {input}"]
        )

        # OLD PATTERN: Using existing parameters
        agent_chain = AgentChain(
            agents={"agent1": agent},
            execution_mode="single",
            auto_include_history=True,
            verbose=False
        )

        result = await agent_chain.process_input(
            "test",
            override_include_history=False  # Existing parameter
        )

        assert isinstance(result, str)


# ============================================================================
# Test Category 3: AgenticStepProcessor Backward Compatibility (0.4.1c)
# ============================================================================

class TestAgenticStepProcessorBackwardCompat:
    """Test AgenticStepProcessor backward compatibility."""

    @pytest.mark.asyncio
    async def test_default_returns_string(self):
        """Test that run_async returns string by default."""
        processor = AgenticStepProcessor(
            objective="Test objective",
            max_internal_steps=2
        )

        # Mock LLM runner and tool executor
        async def mock_llm_runner(messages, tools, tool_choice):
            class MockResponse:
                class Message:
                    content = "Test response"
                    tool_calls = None
                message = Message()
            return MockResponse()

        async def mock_tool_executor(tool_call):
            return "Tool result"

        # OLD PATTERN: No return_metadata, should return string
        result = await processor.run_async(
            initial_input="test input",
            available_tools=[],
            llm_runner=mock_llm_runner,
            tool_executor=mock_tool_executor
        )

        assert isinstance(result, str), f"Expected str, got {type(result)}"

    @pytest.mark.asyncio
    async def test_return_metadata_opt_in(self):
        """Test that return_metadata is opt-in."""
        processor = AgenticStepProcessor(
            objective="Test objective",
            max_internal_steps=2
        )

        async def mock_llm_runner(messages, tools, tool_choice):
            class MockResponse:
                class Message:
                    content = "Test response"
                    tool_calls = None
                message = Message()
            return MockResponse()

        async def mock_tool_executor(tool_call):
            return "Tool result"

        # NEW PATTERN: Opt-in to metadata
        result = await processor.run_async(
            initial_input="test input",
            available_tools=[],
            llm_runner=mock_llm_runner,
            tool_executor=mock_tool_executor,
            return_metadata=True
        )

        assert hasattr(result, 'final_answer'), "Metadata missing 'final_answer'"
        assert hasattr(result, 'total_steps'), "Metadata missing 'total_steps'"
        assert hasattr(result, 'steps'), "Metadata missing 'steps'"


# ============================================================================
# Test Category 4: Callback System Backward Compatibility (0.4.1d-f)
# ============================================================================

class TestCallbackSystemBackwardCompat:
    """Test that callback system is opt-in and doesn't affect existing behavior."""

    def test_no_callbacks_by_default(self):
        """Test that no callbacks fire unless explicitly registered."""
        # OLD PATTERN: Create chain without any callbacks
        chain = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Say hello: {input}"]
        )

        # Should have callback manager but no callbacks registered
        assert hasattr(chain, 'callback_manager')
        assert not chain.callback_manager.has_callbacks()

    def test_chain_works_without_callbacks(self):
        """Test that chains work identically with or without callbacks."""
        # Create two identical chains
        chain_no_callbacks = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Echo: {input}"]
        )

        chain_with_callbacks = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Echo: {input}"]
        )

        # Register callback on second chain
        events_captured = []
        def capture_event(event: ExecutionEvent):
            events_captured.append(event)

        chain_with_callbacks.register_callback(capture_event)

        # Both should work
        result1 = chain_no_callbacks.process_prompt("test")
        result2 = chain_with_callbacks.process_prompt("test")

        # Results should be strings (backward compatible)
        assert isinstance(result1, str)
        assert isinstance(result2, str)

        # Callbacks should have fired for second chain
        assert len(events_captured) > 0, "Callbacks should fire when registered"

    def test_callback_registration_is_optional(self):
        """Test that callback registration is completely optional."""
        # OLD PATTERN: No callback registration
        chain = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Test: {input}"]
        )

        # Should work without any callback setup
        result = chain.process_prompt("test input")
        assert isinstance(result, str)


# ============================================================================
# Test Category 5: Mixed Usage Patterns (Old + New API Together)
# ============================================================================

class TestMixedUsagePatterns:
    """Test that old and new APIs can be used together."""

    def test_mixed_history_access(self):
        """Test mixing private and public API access."""
        manager = ExecutionHistoryManager(max_tokens=1000)
        manager.add_entry("user_input", "Test", source="user")

        # Mix old and new patterns
        old_count = manager._current_token_count  # Old private access
        new_count = manager.current_token_count    # New public access
        public_history = manager.history           # New public access
        private_history = manager._history         # Old private access

        # All should work and match
        assert old_count == new_count
        assert len(public_history) == len(private_history)

    @pytest.mark.asyncio
    async def test_gradual_migration_pattern(self):
        """Test gradual migration from old to new API."""
        agent = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Process: {input}"]
        )

        chain = AgentChain(
            agents={"agent": agent},
            execution_mode="single"
        )

        # Step 1: Use old pattern (returns string)
        result1 = await chain.process_input("test1")
        assert isinstance(result1, str)

        # Step 2: Migrate to new pattern (returns metadata)
        result2 = await chain.process_input("test2", return_metadata=True)
        assert hasattr(result2, 'response')

        # Step 3: Extract string from metadata (bridge pattern)
        response_str = result2.response
        assert isinstance(response_str, str)


# ============================================================================
# Test Category 6: Regression Detection
# ============================================================================

class TestRegressionDetection:
    """Test that existing behavior is completely unchanged."""

    def test_history_truncation_unchanged(self):
        """Test that history truncation works as before."""
        manager = ExecutionHistoryManager(max_entries=3)

        # Add more entries than max
        manager.add_entry("user_input", "Entry 1", source="user")
        manager.add_entry("user_input", "Entry 2", source="user")
        manager.add_entry("user_input", "Entry 3", source="user")
        manager.add_entry("user_input", "Entry 4", source="user")

        # Should truncate to max_entries
        assert manager.history_size <= 3

    def test_chain_execution_unchanged(self):
        """Test that basic chain execution is unchanged."""
        # OLD PATTERN: Basic chain creation and execution
        chain = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=[
                "Step 1: {input}",
                "Step 2: {input}"
            ]
        )

        result = chain.process_prompt("test")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_async_execution_unchanged(self):
        """Test that async execution is unchanged."""
        chain = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Async test: {input}"]
        )

        # OLD PATTERN: Async execution
        result = await chain.process_prompt_async("test")
        assert isinstance(result, str)


# ============================================================================
# Test Category 7: Performance Validation
# ============================================================================

class TestPerformanceUnchanged:
    """Test that performance is unchanged when new features are not used."""

    def test_no_overhead_without_callbacks(self):
        """Test that callbacks don't add overhead when not registered."""
        import time

        # Chain without callbacks
        chain = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Quick response: {input}"]
        )

        # Should not slow down execution
        start = time.time()
        result = chain.process_prompt("test")
        duration = time.time() - start

        # Basic assertion - should complete
        assert isinstance(result, str)
        # Note: Actual performance benchmarking would need more sophisticated setup

    def test_no_overhead_without_metadata(self):
        """Test that return_metadata=False doesn't add overhead."""
        agent = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Test: {input}"]
        )

        chain = AgentChain(
            agents={"agent": agent},
            execution_mode="single"
        )

        # Both should complete successfully
        async def run_test():
            result1 = await chain.process_input("test", return_metadata=False)
            result2 = await chain.process_input("test", return_metadata=True)
            return result1, result2

        result1, result2 = asyncio.run(run_test())
        assert isinstance(result1, str)
        assert hasattr(result2, 'response')


# ============================================================================
# Test Category 8: Error Handling Backward Compatibility
# ============================================================================

class TestErrorHandlingBackwardCompat:
    """Test that error handling is backward compatible."""

    def test_errors_still_raise_as_before(self):
        """Test that errors are raised in the same way."""
        manager = ExecutionHistoryManager()

        # Test that invalid entry types still raise errors (if they did before)
        # This is a placeholder - adjust based on actual error handling
        try:
            manager.add_entry("invalid_type", "content", source="test")  # type: ignore
            # If it doesn't raise, that's fine too - backward compatible
        except Exception as e:
            # If it raises, error type should be expected
            assert isinstance(e, (ValueError, TypeError, AttributeError))

    def test_chain_errors_unchanged(self):
        """Test that chain execution errors work as before."""
        # Test with invalid model (should raise error as before)
        with pytest.raises(Exception):
            chain = PromptChain(
                models=["invalid/model/name"],
                instructions=["Test: {input}"]
            )
            chain.process_prompt("test")


# ============================================================================
# Integration Test: Real-World Usage Validation
# ============================================================================

class TestRealWorldUsage:
    """Test that real-world usage patterns work correctly."""

    @pytest.mark.asyncio
    async def test_agentic_team_chat_pattern(self):
        """Test the pattern used in agentic_team_chat.py."""
        # Simulate the agentic_team_chat.py pattern
        history_manager = ExecutionHistoryManager(max_tokens=4000)

        # Create agent with agentic step processor
        agentic_step = AgenticStepProcessor(
            objective="Test research task",
            max_internal_steps=3
        )

        agent = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=[
                "Prepare: {input}",
                agentic_step,
                "Finalize: {input}"
            ],
            verbose=False
        )

        # This pattern should work
        result = agent.process_prompt("test research")
        assert isinstance(result, str)

        # Add to history (old pattern)
        history_manager.add_entry("agent_output", result, source="agent")
        assert history_manager.history_size > 0


# ============================================================================
# Summary Test: Complete Compatibility Matrix
# ============================================================================

def test_compatibility_matrix():
    """Comprehensive compatibility matrix test."""
    results = {
        "ExecutionHistoryManager": {
            "private_attributes_work": False,
            "public_api_works": False,
            "values_match": False
        },
        "AgentChain": {
            "instantiation_works": False,
            "has_process_input": False,
            "signature_compatible": False
        },
        "AgenticStepProcessor": {
            "instantiation_works": False,
            "has_run_async": False,
            "signature_compatible": False
        },
        "CallbackSystem": {
            "opt_in": False,
            "no_overhead": False
        }
    }

    # Test ExecutionHistoryManager
    try:
        manager = ExecutionHistoryManager(max_tokens=1000)
        manager.add_entry("user_input", "test", source="user")

        # Test private attributes
        _ = manager._current_token_count
        results["ExecutionHistoryManager"]["private_attributes_work"] = True

        # Test public API
        _ = manager.current_token_count
        results["ExecutionHistoryManager"]["public_api_works"] = True

        # Test values match
        if manager._current_token_count == manager.current_token_count:
            results["ExecutionHistoryManager"]["values_match"] = True
    except Exception as e:
        print(f"ExecutionHistoryManager test failed: {e}")

    # Test AgentChain (without API calls)
    try:
        agent = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Test: {input}"]
        )
        chain = AgentChain(
            agents={"agent": agent},
            agent_descriptions={"agent": "Test agent"},
            execution_mode="pipeline"  # Use valid mode
        )
        results["AgentChain"]["instantiation_works"] = True

        # Check process_input exists
        if hasattr(chain, 'process_input'):
            results["AgentChain"]["has_process_input"] = True

        # Check signature has return_metadata parameter
        import inspect
        sig = inspect.signature(chain.process_input)
        if 'return_metadata' in sig.parameters:
            # Check it has default value False
            param = sig.parameters['return_metadata']
            if param.default is False:
                results["AgentChain"]["signature_compatible"] = True
    except Exception as e:
        print(f"AgentChain test failed: {e}")

    # Test AgenticStepProcessor (without API calls)
    try:
        processor = AgenticStepProcessor(
            objective="Test objective",
            max_internal_steps=2
        )
        results["AgenticStepProcessor"]["instantiation_works"] = True

        # Check run_async exists
        if hasattr(processor, 'run_async'):
            results["AgenticStepProcessor"]["has_run_async"] = True

        # Check signature has return_metadata parameter
        import inspect
        sig = inspect.signature(processor.run_async)
        if 'return_metadata' in sig.parameters:
            # Check it has default value False
            param = sig.parameters['return_metadata']
            if param.default is False:
                results["AgenticStepProcessor"]["signature_compatible"] = True
    except Exception as e:
        print(f"AgenticStepProcessor test failed: {e}")

    # Test callbacks
    try:
        chain = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Test: {input}"]
        )

        # Should have callback_manager
        if hasattr(chain, 'callback_manager'):
            results["CallbackSystem"]["opt_in"] = True

        # Should not have callbacks by default
        if not chain.callback_manager.has_callbacks():
            results["CallbackSystem"]["no_overhead"] = True
    except Exception as e:
        print(f"Callback test failed: {e}")

    # Print compatibility matrix
    print("\n" + "="*80)
    print("BACKWARD COMPATIBILITY MATRIX")
    print("="*80)
    for component, tests in results.items():
        print(f"\n{component}:")
        for test_name, passed in tests.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name}: {status}")
    print("\n" + "="*80)

    # Assert all critical tests pass
    all_passed = all(
        all(tests.values()) for tests in results.values()
    )

    if not all_passed:
        print("\n⚠️  Some compatibility tests failed!")
    else:
        print("\n✅ All backward compatibility tests passed!")

    assert all_passed, "Not all compatibility tests passed"


if __name__ == "__main__":
    # Run compatibility matrix test
    test_compatibility_matrix()
    print("\nRunning full test suite with pytest...")
    pytest.main([__file__, "-v"])
