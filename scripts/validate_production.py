#!/usr/bin/env python3
"""
Production Validation Script for Observability Features (v0.4.1)

This script validates all observability improvements in production-like scenarios.
Runs comprehensive tests with real LLM calls to ensure production readiness.
"""

import asyncio
import time
import sys
import os
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType
from promptchain.utils.execution_history_manager import ExecutionHistoryManager


class ValidationTracker:
    """Track validation results."""

    def __init__(self):
        self.events: List[ExecutionEvent] = []
        self.tests_passed = 0
        self.tests_failed = 0

    def callback(self, event: ExecutionEvent):
        """Record callback events."""
        self.events.append(event)

    def assert_true(self, condition: bool, test_name: str):
        """Assert condition is true."""
        if condition:
            print(f"  ✅ {test_name}")
            self.tests_passed += 1
        else:
            print(f"  ❌ {test_name}")
            self.tests_failed += 1

    def get_event_count(self, event_type: ExecutionEventType) -> int:
        """Count events of a specific type."""
        return sum(1 for e in self.events if e.event_type == event_type)


async def test_basic_callbacks():
    """Test 1: Basic chain with callbacks."""
    print("\n" + "="*70)
    print("TEST 1: Basic Chain with Callbacks")
    print("="*70)

    tracker = ValidationTracker()

    chain = PromptChain(
        models=["openai/gpt-3.5-turbo"],
        instructions=["Analyze: {input}", "Summarize: {input}"],
        verbose=False
    )

    chain.register_callback(tracker.callback)

    result = await chain.process_prompt_async("What is 2+2?")

    tracker.assert_true(result is not None, "Execution completed")
    tracker.assert_true(len(tracker.events) > 0, "Callbacks fired")
    tracker.assert_true(
        tracker.get_event_count(ExecutionEventType.CHAIN_START) >= 1,
        "CHAIN_START event fired"
    )
    tracker.assert_true(
        tracker.get_event_count(ExecutionEventType.CHAIN_END) >= 1,
        "CHAIN_END event fired"
    )

    return tracker


async def test_agent_metadata():
    """Test 2: AgentChain with metadata."""
    print("\n" + "="*70)
    print("TEST 2: AgentChain with Metadata")
    print("="*70)

    tracker = ValidationTracker()

    chain = PromptChain(
        models=["openai/gpt-3.5-turbo"],
        instructions=["Answer: {input}"],
        verbose=False
    )

    chain.register_callback(tracker.callback)

    agent_chain = AgentChain(
        agents={"responder": chain},
        agent_descriptions={"responder": "Responds to queries"},
        execution_mode="pipeline",  # Use pipeline with single agent
        verbose=False
    )

    result = await agent_chain.process_input("What is AI?", return_metadata=True)

    tracker.assert_true(result is not None, "Execution completed")
    tracker.assert_true(result.response is not None, "Got response")
    tracker.assert_true(result.agent_name == "responder", "Agent name in metadata")
    tracker.assert_true(result.execution_time_ms > 0, "Execution time tracked")
    tracker.assert_true(len(tracker.events) > 0, "Callbacks fired")

    return tracker


async def test_agentic_step_metadata():
    """Test 3: AgenticStepProcessor with metadata (via PromptChain integration)."""
    print("\n" + "="*70)
    print("TEST 3: AgenticStepProcessor with Metadata")
    print("="*70)

    tracker = ValidationTracker()

    def search_tool(query: str) -> str:
        """Simulated search tool."""
        return f"Search results for '{query}': Found relevant information."

    agentic_step = AgenticStepProcessor(
        objective="Research the topic",
        max_internal_steps=2,
        model_name="openai/gpt-3.5-turbo",
        history_mode="minimal"
    )

    # Create chain with agentic step
    chain = PromptChain(
        models=["openai/gpt-3.5-turbo"],  # Need a model for final step
        instructions=[
            agentic_step,
            "Summarize findings: {input}"
        ],
        verbose=False
    )

    chain.register_callback(tracker.callback)
    chain.register_tool_function(search_tool)
    chain.add_tools([{
        "type": "function",
        "function": {
            "name": "search_tool",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }])

    result = await chain.process_prompt_async("Research renewable energy")

    tracker.assert_true(result is not None, "Got result")
    tracker.assert_true(len(result) > 0, "Result has content")
    tracker.assert_true(len(tracker.events) > 0, "Callbacks fired")

    return tracker


async def test_history_integration():
    """Test 4: ExecutionHistoryManager integration."""
    print("\n" + "="*70)
    print("TEST 4: ExecutionHistoryManager Integration")
    print("="*70)

    tracker = ValidationTracker()
    history_manager = ExecutionHistoryManager(max_tokens=2000, max_entries=20)

    chain = PromptChain(
        models=["openai/gpt-3.5-turbo"],
        instructions=["Reply: {input}"],
        verbose=False
    )

    chain.register_callback(tracker.callback)

    user_input = "Hello AI"
    history_manager.add_entry("user_input", user_input, source="test")

    result = await chain.process_prompt_async(user_input)

    history_manager.add_entry("agent_output", result, source="chain")

    # Get raw history list
    history = history_manager.get_history()
    formatted = history_manager.get_formatted_history(format_style='chat')

    tracker.assert_true(len(history_manager._history) == 2, "History has 2 entries")
    tracker.assert_true(len(history) == 2, "get_history() returns 2 entries")
    tracker.assert_true(history[0]['type'] == 'user_input', "First entry is user_input")
    tracker.assert_true(history[1]['type'] == 'agent_output', "Second entry is agent_output")
    tracker.assert_true(len(formatted) > 0, "Formatted history has content")
    tracker.assert_true(len(tracker.events) > 0, "Callbacks fired")

    return tracker


async def test_performance_overhead():
    """Test 5: Performance overhead measurement."""
    print("\n" + "="*70)
    print("TEST 5: Performance Overhead Measurement")
    print("="*70)

    tracker = ValidationTracker()

    # Baseline without callbacks
    chain_baseline = PromptChain(
        models=["openai/gpt-3.5-turbo"],
        instructions=["Echo: {input}"],
        verbose=False
    )

    start = time.time()
    for i in range(3):
        await chain_baseline.process_prompt_async(f"test {i}")
    baseline_time = time.time() - start

    # With callbacks
    chain_with_callbacks = PromptChain(
        models=["openai/gpt-3.5-turbo"],
        instructions=["Echo: {input}"],
        verbose=False
    )
    chain_with_callbacks.register_callback(tracker.callback)

    start = time.time()
    for i in range(3):
        await chain_with_callbacks.process_prompt_async(f"test {i}")
    callback_time = time.time() - start

    if baseline_time > 0:
        overhead_percent = ((callback_time - baseline_time) / baseline_time) * 100
    else:
        overhead_percent = 0

    print(f"\n  Baseline time: {baseline_time:.2f}s")
    print(f"  With callbacks: {callback_time:.2f}s")
    print(f"  Overhead: {overhead_percent:.1f}%")

    tracker.assert_true(overhead_percent < 20, f"Overhead acceptable ({overhead_percent:.1f}% < 20%)")
    tracker.assert_true(len(tracker.events) > 0, "Callbacks fired")

    return tracker


async def test_all_features_together():
    """Test 6: All features working together."""
    print("\n" + "="*70)
    print("TEST 6: All Features Together")
    print("="*70)

    tracker = ValidationTracker()
    history_manager = ExecutionHistoryManager(max_tokens=2000, max_entries=20)

    chain = PromptChain(
        models=["openai/gpt-3.5-turbo", "openai/gpt-3.5-turbo"],
        instructions=["Analyze: {input}", "Conclude: {input}"],
        store_steps=True,
        verbose=False
    )

    chain.register_callback(tracker.callback)

    agent_chain = AgentChain(
        agents={"analyzer": chain},
        agent_descriptions={"analyzer": "Analyzes input"},
        execution_mode="pipeline",  # Use pipeline with single agent
        verbose=False
    )

    user_input = "What is quantum computing?"
    history_manager.add_entry("user_input", user_input, source="test")

    result = await agent_chain.process_input(user_input, return_metadata=True)

    history_manager.add_entry("agent_output", result.response, source="chain")

    tracker.assert_true(result.response is not None, "Execution completed")
    tracker.assert_true(result.execution_time_ms > 0, "Metadata present")
    tracker.assert_true(len(tracker.events) > 0, "Callbacks fired")
    tracker.assert_true(len(chain.step_outputs) >= 2, "Steps stored (expected >= 2)")
    tracker.assert_true(len(history_manager._history) == 2, "History tracked")

    print(f"\n  Response length: {len(result.response)} chars")
    print(f"  Execution time: {result.execution_time_ms:.1f}ms")
    print(f"  Callbacks: {len(tracker.events)} events")
    print(f"  Steps stored: {len(chain.step_outputs)}")
    print(f"  History entries: {len(history_manager._history)}")

    return tracker


async def run_all_tests():
    """Run all production validation tests."""
    print("\n" + "="*80)
    print("PRODUCTION VALIDATION - OBSERVABILITY FEATURES v0.4.1")
    print("="*80)

    all_trackers = []

    # Run tests
    all_trackers.append(await test_basic_callbacks())
    all_trackers.append(await test_agent_metadata())
    all_trackers.append(await test_agentic_step_metadata())
    all_trackers.append(await test_history_integration())
    all_trackers.append(await test_performance_overhead())
    all_trackers.append(await test_all_features_together())

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    total_passed = sum(t.tests_passed for t in all_trackers)
    total_failed = sum(t.tests_failed for t in all_trackers)
    total_tests = total_passed + total_failed

    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {total_passed} ✅")
    print(f"Failed: {total_failed} ❌")

    if total_failed == 0:
        print("\n🎉 ALL PRODUCTION VALIDATION TESTS PASSED!")
        print("✅ Observability features are PRODUCTION READY")
        return 0
    else:
        print(f"\n⚠️  {total_failed} tests failed")
        print("❌ Review failures before production deployment")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
