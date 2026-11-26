"""
Production Validation Test Suite for Observability Features (v0.4.1)

This module provides comprehensive end-to-end validation of observability improvements
in production-like scenarios with real LLM calls and complete feature integration.

Test Coverage:
- Complete chains with callbacks and metadata
- AgentChain with router and event tracking
- MCP integration with event logging
- Error scenarios with graceful degradation
- Performance validation and overhead measurement
- Long-running sessions with memory management
"""

import pytest
import asyncio
import json
import time
import os
from typing import List, Dict, Any
from datetime import datetime
from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType
from promptchain.utils.execution_callback import CallbackManager
from promptchain.utils.execution_history_manager import ExecutionHistoryManager


# The root conftest mocks the model unless PC_LIVE_TESTS=1; only a live run needs a key.
pytestmark = pytest.mark.skipif(
    os.getenv("PC_LIVE_TESTS") == "1" and not os.getenv("OPENAI_API_KEY"),
    reason="OpenAI API key required for live production validation"
)

LIVE = os.getenv("PC_LIVE_TESTS") == "1"

# A real gpt-3.5 call takes roughly this long; the "<10%" overhead targets are
# 10% of it. Against the instant mock a ratio is meaningless, so the mocked runs
# assert the same budget as absolute milliseconds per call.
REFERENCE_CALL_MS = 500.0
OVERHEAD_BUDGET_MS = REFERENCE_CALL_MS * 0.10


def single_agent_chain(name, chain, **kwargs):
    """AgentChain that runs one agent directly (pipeline mode with a single agent)."""
    return AgentChain(
        agents={name: chain},
        agent_descriptions={name: f"{name} agent"},
        execution_mode="pipeline",
        **kwargs,
    )


def history_entries(history_manager):
    """Entries held by an ExecutionHistoryManager, parsed from its JSON rendering."""
    return json.loads(history_manager.get_formatted_history(format_style="full_json"))


class ProductionCallbackTracker:
    """Track callbacks for production validation."""

    def __init__(self):
        self.events: List[ExecutionEvent] = []
        self.event_types: List[str] = []

    def callback(self, event: ExecutionEvent):
        """Record callback events."""
        self.events.append(event)
        self.event_types.append(event.event_type.name)

    def get_event_count(self, event_type: ExecutionEventType) -> int:
        """Count events of a specific type."""
        return sum(1 for e in self.events if e.event_type == event_type)

    def get_metadata_values(self, key: str) -> List[Any]:
        """Extract metadata values for a given key."""
        return [e.metadata.get(key) for e in self.events if key in e.metadata]


@pytest.mark.integration
@pytest.mark.production
class TestProductionChainValidation:
    """End-to-end validation with real LLM calls."""

    @pytest.mark.asyncio
    async def test_basic_chain_with_callbacks_and_metadata(self):
        """Test complete chain execution with callbacks via AgentChain."""
        tracker = ProductionCallbackTracker()

        # Create chain
        chain = PromptChain(
            models=["openai/gpt-3.5-turbo"],
            instructions=[
                "Analyze this number: {input}",
                "Is it prime? Answer yes or no: {input}"
            ],
            verbose=True
        )

        # Register callback
        chain.register_callback(tracker.callback)

        # Create AgentChain for metadata support
        agent_chain = single_agent_chain("analyzer", chain, verbose=True)

        # Execute with metadata
        result = await agent_chain.process_input("17", return_metadata=True)

        # Validate execution
        assert result is not None
        assert result.response is not None
        assert len(result.response) > 0

        # Validate metadata
        assert result.agent_name == "analyzer"
        assert result.execution_time_ms > 0

        # Validate callbacks fired
        assert len(tracker.events) >= 4  # At least START, 2 steps, END
        assert tracker.get_event_count(ExecutionEventType.CHAIN_START) >= 1
        assert tracker.get_event_count(ExecutionEventType.CHAIN_END) >= 1
        assert tracker.get_event_count(ExecutionEventType.STEP_END) >= 2

    @pytest.mark.asyncio
    async def test_async_chain_with_event_tracking(self):
        """Test async chain execution with comprehensive event tracking."""
        tracker = ProductionCallbackTracker()

        chain = PromptChain(
            models=["openai/gpt-3.5-turbo"],
            instructions=[
                "What is 5 + 3?",
                "Now multiply {input} by 2"
            ],
            verbose=True
        )

        chain.register_callback(tracker.callback)

        result = await chain.process_prompt_async("test")

        # Validate execution
        assert result is not None

        # Check event sequence
        event_sequence = tracker.event_types
        assert "CHAIN_START" in event_sequence
        assert "CHAIN_END" in event_sequence
        assert event_sequence.index("CHAIN_START") < event_sequence.index("CHAIN_END")

    def test_chain_with_execution_history_integration(self):
        """Test chain with ExecutionHistoryManager integration."""
        tracker = ProductionCallbackTracker()
        history_manager = ExecutionHistoryManager(max_tokens=2000, max_entries=20)

        chain = PromptChain(
            models=["openai/gpt-3.5-turbo"],
            instructions=["Summarize in one word: {input}"],
            verbose=True
        )

        chain.register_callback(tracker.callback)

        # Track in history
        user_input = "Machine learning is a subset of artificial intelligence"
        history_manager.add_entry("user_input", user_input, source="test")

        result = chain.process_prompt(user_input)

        history_manager.add_entry("agent_output", result, source="chain")

        # Validate history - the entries, in order, and the chat rendering
        entries = history_entries(history_manager)
        assert [e["type"] for e in entries] == ["user_input", "agent_output"]
        chat_lines = history_manager.get_formatted_history(format_style='chat').splitlines()
        assert chat_lines[0].startswith("User:")
        assert any(line.startswith("Agent (chain):") for line in chat_lines[1:])

        # Validate callbacks still work
        assert len(tracker.events) >= 2


@pytest.mark.integration
@pytest.mark.production
class TestProductionAgentChainValidation:
    """Validate AgentChain in production scenarios."""

    @pytest.mark.asyncio
    async def test_agent_chain_with_router_and_callbacks(self, mock_llm):
        """Test multi-agent system with router and event tracking."""
        tracker = ProductionCallbackTracker()

        # Create specialized agents
        analyzer_agent = PromptChain(
            models=["openai/gpt-3.5-turbo"],
            instructions=["Analyze the mathematical expression: {input}"],
            verbose=True
        )

        solver_agent = PromptChain(
            models=["openai/gpt-3.5-turbo"],
            instructions=["Solve this: {input}"],
            verbose=True
        )

        # Create router configuration
        router_config = {
            "models": ["openai/gpt-3.5-turbo"],
            "instructions": [None, "{input}"],
            "decision_prompt_templates": {
                "single_agent_dispatch": """
Based on the user request: {user_input}

Available agents:
{agent_details}

Choose the most appropriate agent and return JSON:
{{"chosen_agent": "agent_name", "refined_query": "optional refined query"}}
                """
            }
        }

        # Create multi-agent system
        agent_chain = AgentChain(
            agents={
                "analyzer": analyzer_agent,
                "solver": solver_agent
            },
            agent_descriptions={
                "analyzer": "Analyzes mathematical expressions",
                "solver": "Solves mathematical problems"
            },
            execution_mode="router",
            router=router_config,
            verbose=True
        )

        # Register callback on both agents
        analyzer_agent.register_callback(tracker.callback)
        solver_agent.register_callback(tracker.callback)

        # The mock model answers the router prompt with a routing decision.
        if mock_llm is not None:
            mock_llm.reply = lambda kwargs: (
                '{"chosen_agent": "solver", "refined_query": "What is 7 times 8?"}'
                if "Available agents" in str(kwargs.get("messages"))
                else "56"
            )

        # Execute with router (process_input is async)
        result = await agent_chain.process_input("What is 7 times 8?")

        # Validate execution
        assert result is not None
        assert len(tracker.events) > 0
        if mock_llm is not None:
            assert result == "56"

        # Should have routed to solver
        assert tracker.get_event_count(ExecutionEventType.CHAIN_START) >= 1


@pytest.mark.integration
@pytest.mark.production
class TestProductionErrorHandling:
    """Validate error scenarios and graceful degradation."""

    def test_callback_exception_does_not_break_chain(self):
        """Test that callback exceptions don't break chain execution."""

        def failing_callback(event: ExecutionEvent):
            """Callback that always raises an exception."""
            raise ValueError("Intentional callback failure")

        tracker = ProductionCallbackTracker()

        chain = PromptChain(
            models=["openai/gpt-3.5-turbo"],
            instructions=["Echo: {input}"],
            verbose=True
        )

        # Register both failing and working callbacks
        chain.register_callback(failing_callback)
        chain.register_callback(tracker.callback)

        # Should complete despite failing callback
        result = chain.process_prompt("test message")

        assert result is not None
        # Working callback should still receive events
        assert len(tracker.events) > 0

    @pytest.mark.asyncio
    async def test_metadata_extraction_with_minimal_model(self):
        """Test metadata works even with minimal model responses."""
        chain = PromptChain(
            models=["openai/gpt-3.5-turbo"],
            instructions=["Say 'ok': {input}"],
            verbose=True
        )

        agent_chain = single_agent_chain("simple", chain, verbose=True)

        result = await agent_chain.process_input("test", return_metadata=True)

        # Even minimal execution should have metadata
        assert result is not None
        assert result.response is not None
        assert result.execution_time_ms > 0


@pytest.mark.performance
@pytest.mark.production
class TestProductionPerformance:
    """Validate performance characteristics and overhead."""

    def test_callback_overhead_measurement(self):
        """Measure overhead of callbacks vs baseline (target: <10% of a real call)."""
        calls = 3 if LIVE else 30

        def time_per_call(chain):
            chain.process_prompt("warm-up")  # exclude one-time setup from the timing
            start = time.perf_counter()
            for i in range(calls):
                chain.process_prompt(f"test {i}")
            return (time.perf_counter() - start) / calls

        # Baseline without callbacks
        chain_baseline = PromptChain(
            models=["openai/gpt-3.5-turbo"],
            instructions=["Echo: {input}"],
            verbose=False
        )
        baseline_time = time_per_call(chain_baseline)

        # With callbacks
        tracker = ProductionCallbackTracker()
        chain_with_callbacks = PromptChain(
            models=["openai/gpt-3.5-turbo"],
            instructions=["Echo: {input}"],
            verbose=False
        )
        chain_with_callbacks.register_callback(tracker.callback)
        callback_time = time_per_call(chain_with_callbacks)

        overhead_ms = (callback_time - baseline_time) * 1000
        overhead_percent = (overhead_ms / (baseline_time * 1000)) * 100

        print(f"\nPerformance Results (per call):")
        print(f"  Baseline: {baseline_time * 1000:.2f}ms")
        print(f"  With callbacks: {callback_time * 1000:.2f}ms")
        print(f"  Overhead: {overhead_ms:.2f}ms ({overhead_percent:.1f}%)")

        if LIVE:
            # Real model latency dominates the baseline, so the ratio is meaningful.
            assert overhead_percent < 10, f"Callback overhead too high: {overhead_percent:.1f}%"
        else:
            # The mocked model returns instantly, so the baseline is ~1ms and a ratio
            # only measures the callback machinery. Hold the absolute cost per call
            # to the same budget: 10% of a real call.
            assert overhead_ms < OVERHEAD_BUDGET_MS, (
                f"Callback overhead too high: {overhead_ms:.2f}ms per call "
                f"(budget {OVERHEAD_BUDGET_MS:.0f}ms)"
            )

        # Verify callbacks actually fired
        assert len(tracker.events) > 0

    @pytest.mark.asyncio
    async def test_metadata_extraction_performance(self):
        """Test that metadata extraction has minimal impact."""
        calls = 3 if LIVE else 30
        chain = PromptChain(
            models=["openai/gpt-3.5-turbo"],
            instructions=["Respond: {input}"],
            verbose=False
        )

        agent_chain = single_agent_chain("responder", chain, verbose=False)
        await agent_chain.process_input("warm-up")  # exclude one-time setup from the timing

        # Without metadata
        start = time.perf_counter()
        for _ in range(calls):
            await agent_chain.process_input("test")
        without_metadata = (time.perf_counter() - start) / calls

        # With metadata
        start = time.perf_counter()
        for _ in range(calls):
            await agent_chain.process_input("test", return_metadata=True)
        with_metadata = (time.perf_counter() - start) / calls

        overhead_ms = (with_metadata - without_metadata) * 1000
        metadata_overhead = (overhead_ms / (without_metadata * 1000)) * 100

        print(f"\nMetadata Performance (per call):")
        print(f"  Without metadata: {without_metadata * 1000:.2f}ms")
        print(f"  With metadata: {with_metadata * 1000:.2f}ms")
        print(f"  Overhead: {overhead_ms:.2f}ms ({metadata_overhead:.1f}%)")

        if LIVE:
            assert metadata_overhead < 10, f"Metadata overhead too high: {metadata_overhead:.1f}%"
        else:
            # Instant mocked model: see test_callback_overhead_measurement.
            assert overhead_ms < OVERHEAD_BUDGET_MS, (
                f"Metadata overhead too high: {overhead_ms:.2f}ms per call "
                f"(budget {OVERHEAD_BUDGET_MS:.0f}ms)"
            )


@pytest.mark.integration
@pytest.mark.production
class TestProductionLongRunning:
    """Validate long-running sessions and memory management."""

    def test_long_session_with_history_management(self):
        """Test multiple interactions with proper history management."""
        tracker = ProductionCallbackTracker()
        history_manager = ExecutionHistoryManager(
            max_tokens=4000,
            max_entries=50,
            truncation_strategy="oldest_first"
        )

        chain = PromptChain(
            models=["openai/gpt-3.5-turbo"],
            instructions=["Respond briefly: {input}"],
            verbose=True
        )

        chain.register_callback(tracker.callback)

        # Simulate multiple interactions
        interactions = [
            "What is 2+2?",
            "What is 5*3?",
            "What is 10-7?",
            "What is 8/2?",
            "What is 3^2?"
        ]

        for user_input in interactions:
            history_manager.add_entry("user_input", user_input, source="user")
            result = chain.process_prompt(user_input)
            history_manager.add_entry("agent_output", result, source="chain")

        # Validate session state (use _history internal attribute)
        assert len(history_manager._history) == 10  # 5 user + 5 assistant

        # Validate all callbacks fired
        assert len(tracker.events) >= 10  # At least 2 events per interaction

        # Check history is properly formatted
        formatted_history = history_manager.get_formatted_history(format_style='chat')
        assert len(history_entries(history_manager)) == 10
        assert formatted_history.count("User:") == 5
        assert formatted_history.count("Agent (chain):") == 5

    @pytest.mark.asyncio
    async def test_concurrent_chains_with_callbacks(self):
        """Test multiple concurrent chains with independent callback tracking."""
        tracker1 = ProductionCallbackTracker()
        tracker2 = ProductionCallbackTracker()

        chain1 = PromptChain(
            models=["openai/gpt-3.5-turbo"],
            instructions=["Math: {input}"],
            verbose=False
        )
        chain1.register_callback(tracker1.callback)

        chain2 = PromptChain(
            models=["openai/gpt-3.5-turbo"],
            instructions=["Science: {input}"],
            verbose=False
        )
        chain2.register_callback(tracker2.callback)

        # Run concurrently
        results = await asyncio.gather(
            chain1.process_prompt_async("What is 5+5?"),
            chain2.process_prompt_async("What is H2O?")
        )

        # Both should complete
        assert all(r is not None for r in results)

        # Each tracker should have independent events
        assert len(tracker1.events) > 0
        assert len(tracker2.events) > 0

        # Trackers should be independent
        assert tracker1.events != tracker2.events


@pytest.mark.integration
@pytest.mark.production
class TestProductionFeatureIntegration:
    """Validate complete feature integration."""

    @pytest.mark.asyncio
    async def test_all_features_together(self):
        """Test chain with ALL observability features enabled simultaneously."""
        # Setup all components
        tracker = ProductionCallbackTracker()
        history_manager = ExecutionHistoryManager(max_tokens=2000, max_entries=20)

        # Create chain with multiple steps
        chain = PromptChain(
            models=["openai/gpt-3.5-turbo", "openai/gpt-3.5-turbo"],
            instructions=[
                "First, analyze: {input}",
                "Then, conclude: {input}"
            ],
            store_steps=True,
            verbose=True
        )

        # Register callback
        chain.register_callback(tracker.callback)

        # Create AgentChain for metadata support
        agent_chain = single_agent_chain("analyzer", chain, verbose=True)

        # Execute with metadata and history tracking
        user_input = "What is the capital of France?"
        history_manager.add_entry("user_input", user_input, source="test")

        result = await agent_chain.process_input(user_input, return_metadata=True)

        history_manager.add_entry("agent_output", result.response, source="chain")

        # Validate ALL features work together

        # 1. Basic execution
        assert result.response is not None

        # 2. Metadata extraction
        assert result.execution_time_ms > 0
        assert result.agent_name == "analyzer"

        # 3. Callbacks
        assert len(tracker.events) >= 4
        assert tracker.get_event_count(ExecutionEventType.CHAIN_START) >= 1

        # 4. Step storage
        # step_outputs also holds the initial input as step_0
        model_steps = [o for o in chain.step_outputs.values() if o["type"] == "model"]
        assert len(model_steps) == 2

        # 5. History management
        history = history_entries(history_manager)
        assert [e["type"] for e in history] == ["user_input", "agent_output"]

        # 6. Event metadata correlation
        models_in_events = [e.model_name for e in tracker.events]
        assert any(m is not None for m in models_in_events)

        print("\n✅ ALL FEATURES VALIDATED TOGETHER:")
        print(f"  - Execution: {len(result.response)} chars")
        print(f"  - Metadata: agent={result.agent_name}, time={result.execution_time_ms}ms")
        print(f"  - Callbacks: {len(tracker.events)} events")
        print(f"  - Steps: {len(model_steps)} stored")
        print(f"  - History: {len(history)} entries")


if __name__ == "__main__":
    """Run production validation tests."""
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "-m", "production"
    ])
