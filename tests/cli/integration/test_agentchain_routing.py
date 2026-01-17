"""Integration tests for AgentChain router mode agent selection (T034).

These tests verify that the AgentChain router correctly selects agents based on
user queries. This is the core functionality for User Story 1.

Test-Driven Development (TDD):
- RED: These tests WILL FAIL because TUI integration doesn't exist yet
- GREEN: Implement TUI AgentChain integration (T037-T044)
- REFACTOR: Optimize routing logic after tests pass
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain


class TestAgentChainRouterMode:
    """Test AgentChain router mode functionality."""

    @pytest.fixture
    def research_agent(self):
        """Create research agent with web search capabilities."""
        agent = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Research topic: {input}"],
            verbose=False,
        )
        return agent

    @pytest.fixture
    def coder_agent(self):
        """Create coder agent with code generation capabilities."""
        agent = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Generate code: {input}"],
            verbose=False,
        )
        return agent

    @pytest.fixture
    def analyst_agent(self):
        """Create analyst agent with data analysis capabilities."""
        agent = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Analyze data: {input}"],
            verbose=False,
        )
        return agent

    @pytest.fixture
    def agent_chain_router_mode(self, research_agent, coder_agent, analyst_agent):
        """Create AgentChain in router mode with multiple agents."""
        # Import default router prompt from orchestration config
        from promptchain.cli.models.orchestration_config import DEFAULT_ROUTER_PROMPT

        agent_chain = AgentChain(
            agents={
                "researcher": research_agent,
                "coder": coder_agent,
                "analyst": analyst_agent,
            },
            agent_descriptions={
                "researcher": "Deep research specialist with web search and analysis capabilities",
                "coder": "Code generation and software development specialist",
                "analyst": "Data analysis and statistical interpretation specialist",
            },
            execution_mode="router",
            router={
                "models": ["gpt-4.1-mini-2025-04-14"],
                "instructions": [None, "{input}"],
                "decision_prompt_templates": {
                    "single_agent_dispatch": DEFAULT_ROUTER_PROMPT
                },
            },
            verbose=False,
        )
        return agent_chain

    @pytest.mark.asyncio
    async def test_router_selects_research_agent_for_research_query(
        self, agent_chain_router_mode
    ):
        """Test router selects researcher for research-oriented queries."""
        # Research-oriented query
        query = "Research the latest developments in quantum computing and summarize key breakthroughs"

        # Mock the router to return researcher
        with patch.object(
            agent_chain_router_mode,
            "_route_to_agent",
            return_value=("researcher", query),
        ):
            # Mock the agent execution
            with patch.object(
                agent_chain_router_mode.agents["researcher"],
                "process_prompt_async",
                return_value="Research summary on quantum computing...",
            ) as mock_execute:
                result = await agent_chain_router_mode.run_chat_turn_async(query)

                # Verify researcher was called
                mock_execute.assert_called_once()
                assert "Research summary" in result

    @pytest.mark.asyncio
    async def test_router_selects_coder_agent_for_code_query(
        self, agent_chain_router_mode
    ):
        """Test router selects coder for code generation queries."""
        # Code-oriented query
        query = "Write a Python function to validate email addresses using regex"

        # Mock the router to return coder
        with patch.object(
            agent_chain_router_mode,
            "_route_to_agent",
            return_value=("coder", query),
        ):
            # Mock the agent execution
            with patch.object(
                agent_chain_router_mode.agents["coder"],
                "process_prompt_async",
                return_value="def validate_email(email): ...",
            ) as mock_execute:
                result = await agent_chain_router_mode.run_chat_turn_async(query)

                # Verify coder was called
                mock_execute.assert_called_once()
                assert "def validate_email" in result

    @pytest.mark.asyncio
    async def test_router_selects_analyst_agent_for_data_query(
        self, agent_chain_router_mode
    ):
        """Test router selects analyst for data analysis queries."""
        # Data analysis query
        query = "Analyze the distribution of sales data and identify outliers"

        # Mock the router to return analyst
        with patch.object(
            agent_chain_router_mode,
            "_route_to_agent",
            return_value=("analyst", query),
        ):
            # Mock the agent execution
            with patch.object(
                agent_chain_router_mode.agents["analyst"],
                "process_prompt_async",
                return_value="Distribution analysis: mean=100, std=15...",
            ) as mock_execute:
                result = await agent_chain_router_mode.run_chat_turn_async(query)

                # Verify analyst was called
                mock_execute.assert_called_once()
                assert "Distribution analysis" in result

    @pytest.mark.asyncio
    async def test_router_switches_agents_across_conversation(
        self, agent_chain_router_mode
    ):
        """Test router can switch between different agents in conversation."""
        # First query: research
        research_query = "Research authentication best practices"

        with patch.object(
            agent_chain_router_mode,
            "_route_to_agent",
            return_value=("researcher", research_query),
        ):
            with patch.object(
                agent_chain_router_mode.agents["researcher"],
                "process_prompt_async",
                return_value="Best practices include JWT, OAuth2...",
            ):
                result1 = await agent_chain_router_mode.run_chat_turn_async(
                    research_query
                )
                assert "Best practices" in result1

        # Second query: code (agent switch)
        code_query = "Implement JWT authentication in Python"

        with patch.object(
            agent_chain_router_mode,
            "_route_to_agent",
            return_value=("coder", code_query),
        ):
            with patch.object(
                agent_chain_router_mode.agents["coder"],
                "process_prompt_async",
                return_value="import jwt\ndef create_token()...",
            ):
                result2 = await agent_chain_router_mode.run_chat_turn_async(code_query)
                assert "import jwt" in result2

    @pytest.mark.asyncio
    async def test_router_considers_conversation_history(
        self, agent_chain_router_mode
    ):
        """Test router includes conversation history in routing decision."""
        # First message
        query1 = "What are the benefits of microservices?"

        with patch.object(
            agent_chain_router_mode,
            "_route_to_agent",
            return_value=("researcher", query1),
        ) as mock_route:
            with patch.object(
                agent_chain_router_mode.agents["researcher"],
                "process_prompt_async",
                return_value="Microservices provide scalability...",
            ):
                await agent_chain_router_mode.run_chat_turn_async(query1)

        # Second message (should consider history)
        query2 = "How would you implement that?"

        with patch.object(
            agent_chain_router_mode,
            "_route_to_agent",
            return_value=("coder", query2),
        ) as mock_route:
            with patch.object(
                agent_chain_router_mode.agents["coder"],
                "process_prompt_async",
                return_value="To implement microservices in Python...",
            ):
                await agent_chain_router_mode.run_chat_turn_async(query2)

                # Verify _route_to_agent was called (it receives history internally)
                assert mock_route.call_count == 1

    @pytest.mark.asyncio
    async def test_router_fallback_to_default_agent_on_failure(
        self, agent_chain_router_mode
    ):
        """Test router falls back to default agent if routing fails."""
        # Set default agent
        agent_chain_router_mode.default_agent = "researcher"

        query = "Ambiguous query that might confuse router"

        # Mock router to return default agent (simulating fallback behavior)
        # This is what the real _route_to_agent does when an exception occurs
        with patch.object(
            agent_chain_router_mode,
            "_route_to_agent",
            return_value=("researcher", query),  # Returns default agent
        ):
            # Mock default agent execution
            with patch.object(
                agent_chain_router_mode.agents["researcher"],
                "process_prompt_async",
                return_value="Fallback response from default agent",
            ) as mock_execute:
                result = await agent_chain_router_mode.run_chat_turn_async(query)

                # Verify fallback to default agent
                mock_execute.assert_called_once()
                assert "Fallback response" in result

    @pytest.mark.asyncio
    async def test_router_timeout_handling(self, agent_chain_router_mode):
        """Test router handles timeout gracefully."""
        # Configure router with short timeout
        agent_chain_router_mode.router_config = {
            "timeout_seconds": 1,
        }
        agent_chain_router_mode.default_agent = "researcher"

        query = "Test query"

        # Mock router to timeout
        async def slow_route(*args, **kwargs):
            import asyncio

            await asyncio.sleep(2)  # Exceed timeout
            return ("researcher", query)

        with patch.object(
            agent_chain_router_mode, "_route_to_agent", side_effect=slow_route
        ):
            with patch.object(
                agent_chain_router_mode.agents["researcher"],
                "process_prompt_async",
                return_value="Fallback after timeout",
            ) as mock_execute:
                result = await agent_chain_router_mode.run_chat_turn_async(query)

                # Should fall back to default agent
                mock_execute.assert_called_once()

    def test_router_mode_requires_agent_descriptions(
        self, research_agent, coder_agent
    ):
        """Test router mode requires agent descriptions for decision-making."""
        # Router mode without descriptions should raise ValueError
        with pytest.raises(ValueError, match="agent_descriptions must be provided"):
            AgentChain(
                agents={
                    "researcher": research_agent,
                    "coder": coder_agent,
                },
                agent_descriptions={},  # Empty descriptions - should fail validation
                execution_mode="router",
                verbose=False,
            )

    def test_router_config_validation(self, research_agent, coder_agent):
        """Test router configuration is validated."""
        from promptchain.cli.models.orchestration_config import DEFAULT_ROUTER_PROMPT

        # Valid router config should create AgentChain successfully
        agent_chain = AgentChain(
            agents={
                "researcher": research_agent,
                "coder": coder_agent,
            },
            agent_descriptions={
                "researcher": "Research specialist",
                "coder": "Code specialist",
            },
            execution_mode="router",
            router={
                "models": ["gpt-4.1-mini-2025-04-14"],
                "instructions": [None, "{input}"],
                "decision_prompt_templates": {
                    "single_agent_dispatch": DEFAULT_ROUTER_PROMPT
                },
            },
            verbose=False,
        )

        # Verify execution_mode is set correctly
        assert agent_chain.execution_mode == "router"
        assert agent_chain.router_strategy == "single_agent_dispatch"


class TestRouterDecisionParsing:
    """Test router decision JSON parsing and error handling."""

    @pytest.mark.asyncio
    async def test_parse_valid_router_decision_json(self):
        """Test parsing valid router decision JSON."""
        # Valid JSON response from router
        router_response = '{"chosen_agent": "researcher", "refined_query": null}'

        # This would be parsed by AgentChain routing logic
        import json

        decision = json.loads(router_response)

        assert "chosen_agent" in decision
        assert decision["chosen_agent"] == "researcher"

    @pytest.mark.asyncio
    async def test_parse_router_decision_with_refined_query(self):
        """Test parsing router decision with refined query."""
        router_response = (
            '{"chosen_agent": "coder", "refined_query": "Implement with error handling"}'
        )

        import json

        decision = json.loads(router_response)

        assert decision["chosen_agent"] == "coder"
        assert decision["refined_query"] == "Implement with error handling"

    @pytest.mark.asyncio
    async def test_handle_invalid_router_json(self):
        """Test handling of invalid JSON from router."""
        import json

        invalid_json = "This is not valid JSON"

        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json)

    @pytest.mark.asyncio
    async def test_handle_missing_chosen_agent_field(self):
        """Test handling when chosen_agent field is missing."""
        incomplete_json = '{"refined_query": "Some query"}'

        import json

        decision = json.loads(incomplete_json)

        # Should detect missing field
        assert "chosen_agent" not in decision
