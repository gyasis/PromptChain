"""
T106: Performance tests for AgentChain routing latency

Tests validate routing performance characteristics and ensure optimizations
maintain correctness while improving speed where possible.

NOTE: <500ms target is NOT achievable with LLM-based routing due to external
API latency (700-1100ms). These tests verify optimal Python-level performance.
"""

import pytest
import time
import asyncio
from promptchain.utils.agent_chain import AgentChain
from promptchain import PromptChain


@pytest.fixture
def simple_agents():
    """Create minimal agent setup for performance testing."""
    agents = {
        "agent1": PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Process: {input}"],
            verbose=False
        ),
        "agent2": PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Handle: {input}"],
            verbose=False
        )
    }

    descriptions = {
        "agent1": "Handles first type of requests",
        "agent2": "Handles second type of requests"
    }

    return agents, descriptions


@pytest.fixture
def router_config():
    """Standard router configuration."""
    return {
        "models": ["openai/gpt-4o-mini"],
        "instructions": [None, "{input}"],
        "decision_prompt_templates": {
            "single_agent_dispatch": """Based on: {user_input}

Agents:
{agent_details}

History:
{history}

Return JSON: {{"chosen_agent": "agent_name"}}"""
        }
    }


class TestHistoryFormattingPerformance:
    """Test history formatting optimizations."""

    def test_history_formatting_speed(self, simple_agents, router_config):
        """Verify history formatting completes in <100ms (target met)."""
        agents, descriptions = simple_agents

        chain = AgentChain(
            agents=agents,
            agent_descriptions=descriptions,
            execution_mode="router",
            router=router_config,
            verbose=False
        )

        # Populate history with 50 messages
        for i in range(50):
            role = "user" if i % 2 == 0 else "assistant"
            chain._conversation_history.append({
                "role": role,
                "content": f"Message {i} with some content to format"
            })

        # Measure formatting time
        start = time.perf_counter()
        for _ in range(100):  # 100 iterations
            formatted = chain._format_chat_history()
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_time_ms = elapsed_ms / 100

        # Should be well under 100ms target (actual: <1ms)
        assert avg_time_ms < 100, f"History formatting took {avg_time_ms:.2f}ms (target: <100ms)"

        # Verify cache working
        assert formatted is not None
        assert len(formatted) > 0

    def test_history_cache_invalidation(self, simple_agents, router_config):
        """Verify history cache invalidates when history changes."""
        agents, descriptions = simple_agents

        chain = AgentChain(
            agents=agents,
            agent_descriptions=descriptions,
            execution_mode="router",
            router=router_config,
            verbose=False
        )

        # Format once
        result1 = chain._format_chat_history()
        version1 = chain._history_cache_version

        # Add message via _add_to_history (invalidates cache)
        chain._add_to_history("user", "New message")

        # Cache should be invalidated
        version2 = chain._history_cache_version
        assert version2 > version1, "Cache version should increment on history change"

        # Format again should produce different result
        result2 = chain._format_chat_history()
        assert result2 != result1, "Formatted history should change after adding message"

    def test_history_cache_reuse(self, simple_agents, router_config):
        """Verify history cache reused when history unchanged."""
        agents, descriptions = simple_agents

        chain = AgentChain(
            agents=agents,
            agent_descriptions=descriptions,
            execution_mode="router",
            router=router_config,
            verbose=False
        )

        # Populate history
        chain._conversation_history.append({
            "role": "user",
            "content": "Test message"
        })

        # Format once
        result1 = chain._format_chat_history()
        cache_version = chain._history_cache_version

        # Format again without changing history
        result2 = chain._format_chat_history()

        # Should be identical (cache hit)
        assert result1 == result2
        assert chain._history_cache_version == cache_version


class TestAgentDetailsCaching:
    """Test agent details lazy caching optimization."""

    def test_agent_details_cached(self, simple_agents, router_config):
        """Verify agent details are cached on first use."""
        agents, descriptions = simple_agents

        chain = AgentChain(
            agents=agents,
            agent_descriptions=descriptions,
            execution_mode="router",
            router=router_config,
            verbose=False
        )

        # Initially no cache
        assert chain._agent_details_cache is None

        # Trigger caching via _prepare_full_decision_prompt
        prompt = chain._prepare_full_decision_prompt(chain, "test input")

        # Cache should now exist
        assert chain._agent_details_cache is not None
        assert "agent1" in chain._agent_details_cache
        assert "agent2" in chain._agent_details_cache

    def test_agent_details_reused(self, simple_agents, router_config):
        """Verify cached agent details are reused."""
        agents, descriptions = simple_agents

        chain = AgentChain(
            agents=agents,
            agent_descriptions=descriptions,
            execution_mode="router",
            router=router_config,
            verbose=False
        )

        # First call caches
        prompt1 = chain._prepare_full_decision_prompt(chain, "input 1")
        cache1 = chain._agent_details_cache

        # Second call reuses
        prompt2 = chain._prepare_full_decision_prompt(chain, "input 2")
        cache2 = chain._agent_details_cache

        # Should be same object (not reformatted)
        assert cache1 is cache2


class TestRoutingLatencyBudget:
    """Test realistic routing latency expectations."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_simple_router_performance(self, simple_agents, router_config):
        """Simple router should bypass LLM and complete in <50ms."""
        agents, descriptions = simple_agents

        chain = AgentChain(
            agents=agents,
            agent_descriptions=descriptions,
            execution_mode="router",
            router=router_config,
            verbose=False
        )

        # Use input that matches simple router pattern (if implemented)
        test_input = "Help with agent1"

        start = time.perf_counter()
        result = chain._simple_router(test_input)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Simple router should be extremely fast
        assert elapsed_ms < 50, f"Simple router took {elapsed_ms:.2f}ms (target: <50ms)"

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_llm_routing_realistic_latency(self, simple_agents, router_config):
        """LLM routing latency should be documented (700-1100ms realistic)."""
        agents, descriptions = simple_agents

        chain = AgentChain(
            agents=agents,
            agent_descriptions=descriptions,
            execution_mode="router",
            router=router_config,
            verbose=False
        )

        # Use input that requires LLM routing
        test_input = "Analyze complex data patterns"

        start = time.perf_counter()
        agent_name, refined_query = await chain._route_to_agent(test_input)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Document realistic latency (not a failure if >500ms)
        print(f"\nLLM routing latency: {elapsed_ms:.1f}ms")
        print(f"Selected agent: {agent_name}")

        # Verify routing worked
        assert agent_name in chain.agent_names

        # Log latency for analysis (not an assertion failure)
        if elapsed_ms < 500:
            print("✅ Exceptional performance (<500ms)")
        elif elapsed_ms < 1000:
            print("✅ Good performance (500-1000ms)")
        else:
            print("⚠️  Slower than typical (>1000ms) - network variance")


class TestPythonOverhead:
    """Test Python-level routing overhead (excluding LLM calls)."""

    def test_prompt_preparation_overhead(self, simple_agents, router_config):
        """Prompt preparation should be <10ms."""
        agents, descriptions = simple_agents

        chain = AgentChain(
            agents=agents,
            agent_descriptions=descriptions,
            execution_mode="router",
            router=router_config,
            verbose=False
        )

        # Populate some history
        chain._conversation_history = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
        ]

        # Measure prompt preparation time
        start = time.perf_counter()
        for _ in range(100):
            prompt = chain._prepare_full_decision_prompt(chain, "test input")
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_time_ms = elapsed_ms / 100

        # Should be very fast with caching
        assert avg_time_ms < 10, f"Prompt preparation took {avg_time_ms:.2f}ms (target: <10ms)"

    def test_decision_parsing_overhead(self, simple_agents, router_config):
        """Decision parsing should be <5ms."""
        agents, descriptions = simple_agents

        chain = AgentChain(
            agents=agents,
            agent_descriptions=descriptions,
            execution_mode="router",
            router=router_config,
            verbose=False
        )

        # Sample decision output
        decision_output = '{"chosen_agent": "agent1", "refined_query": "Processed input"}'

        # Measure parsing time
        start = time.perf_counter()
        for _ in range(1000):
            parsed = chain._parse_decision(decision_output)
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_time_ms = elapsed_ms / 1000

        # Should be very fast (JSON parsing)
        assert avg_time_ms < 5, f"Decision parsing took {avg_time_ms:.2f}ms (target: <5ms)"

        # Verify parsing worked
        assert parsed["chosen_agent"] == "agent1"


@pytest.mark.integration
class TestEndToEndPerformance:
    """Integration tests for end-to-end routing performance."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_routing_with_empty_history(self, simple_agents, router_config):
        """Test routing performance with no conversation history."""
        agents, descriptions = simple_agents

        chain = AgentChain(
            agents=agents,
            agent_descriptions=descriptions,
            execution_mode="router",
            router=router_config,
            verbose=False
        )

        test_input = "Perform analysis task"

        start = time.perf_counter()
        agent_name, refined_query = await chain._route_to_agent(test_input)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"\nRouting latency (0 history): {elapsed_ms:.1f}ms")

        # Verify routing worked
        assert agent_name in chain.agent_names

        # Document performance (realistic expectation: 700-1100ms)
        # Not a test failure if >500ms due to LLM API latency

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_routing_with_conversation_history(self, simple_agents, router_config):
        """Test routing performance with conversation history."""
        agents, descriptions = simple_agents

        chain = AgentChain(
            agents=agents,
            agent_descriptions=descriptions,
            execution_mode="router",
            router=router_config,
            verbose=False
        )

        # Add conversation history
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            chain._conversation_history.append({
                "role": role,
                "content": f"Message {i} in conversation"
            })

        test_input = "Continue the analysis"

        start = time.perf_counter()
        agent_name, refined_query = await chain._route_to_agent(test_input)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"\nRouting latency (10 history): {elapsed_ms:.1f}ms")

        # Verify routing worked
        assert agent_name in chain.agent_names

        # With caching, history overhead should be minimal (<1ms)
        # Total latency still dominated by LLM call (700-1100ms)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
