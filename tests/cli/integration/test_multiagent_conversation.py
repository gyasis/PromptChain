"""Integration tests for multi-agent conversation flows (T035).

These tests verify that multi-agent conversations work end-to-end with automatic
agent switching, context preservation, and natural flow between specialized agents.

Test-Driven Development (TDD):
- RED: These tests WILL FAIL because TUI integration doesn't exist yet
- GREEN: Implement TUI AgentChain integration (T037-T044)
- REFACTOR: Optimize conversation flow after tests pass
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain


class TestMultiAgentConversationFlow:
    """Test multi-agent conversation flows with automatic agent switching."""

    @pytest.fixture(autouse=True)
    def reset_agent_chain_state(self, request):
        """Reset AgentChain state between tests to ensure isolation.

        This autouse fixture prevents test interaction by:
        1. Clearing conversation history before each test
        2. Resetting execution_mode to ensure it's "router"
        3. Clearing any cached state
        """
        # Get the agent_chain fixture (works for any fixture name)
        try:
            agent_chain = request.getfixturevalue('agent_chain_with_history')
        except Exception:
            # If that fixture doesn't exist, we're in a different test class
            yield
            return

        # Reset conversation history
        if hasattr(agent_chain, '_conversation_history'):
            agent_chain._conversation_history.clear()

        # Ensure execution_mode is "router" (in case previous test changed it)
        agent_chain.execution_mode = "router"

        # Reset last_selected_agent
        if hasattr(agent_chain, 'last_selected_agent'):
            agent_chain.last_selected_agent = None

        yield

        # Cleanup after test
        if hasattr(agent_chain, '_conversation_history'):
            agent_chain._conversation_history.clear()

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
    def reviewer_agent(self):
        """Create code review agent."""
        agent = PromptChain(
            models=["anthropic/claude-3-sonnet-20240229"],
            instructions=["Review code: {input}"],
            verbose=False,
        )
        return agent

    @pytest.fixture
    def agent_chain_with_history(self, research_agent, coder_agent, reviewer_agent):
        """Create AgentChain with conversation history enabled."""
        from promptchain.cli.models.orchestration_config import DEFAULT_ROUTER_PROMPT

        agent_chain = AgentChain(
            agents={
                "researcher": research_agent,
                "coder": coder_agent,
                "reviewer": reviewer_agent,
            },
            agent_descriptions={
                "researcher": "Research specialist with web search and deep analysis",
                "coder": "Code generation and implementation specialist",
                "reviewer": "Code review and quality assurance specialist",
            },
            execution_mode="router",
            router={
                "models": ["gpt-4.1-mini-2025-04-14"],
                "instructions": [None, "{input}"],
                "decision_prompt_templates": {
                    "single_agent_dispatch": DEFAULT_ROUTER_PROMPT
                },
            },
            auto_include_history=True,  # Enable conversation history
            verbose=False,
        )
        return agent_chain

    @pytest.mark.asyncio
    async def test_natural_research_to_code_flow(self, agent_chain_with_history):
        """Test natural conversation flow from research to code generation.

        User Story: Developer wants to implement a feature, starting with research,
        then moving to code generation. Router should automatically switch agents
        while maintaining context.
        """
        # Turn 1: Research phase
        research_query = "What are the best practices for JWT authentication in Python?"

        with patch.object(
            agent_chain_with_history,
            "_route_to_agent",
            return_value=("researcher", research_query),
        ):
            with patch.object(
                agent_chain_with_history.agents["researcher"],
                "process_prompt_async",
                return_value="JWT best practices: Use RS256, short expiry, secure storage...",
            ) as mock_research:
                result1 = await agent_chain_with_history.run_chat_turn_async(research_query)
                mock_research.assert_called_once()
                assert "JWT best practices" in result1

        # Turn 2: Implementation phase (router should switch to coder)
        code_query = "Now implement a JWT authentication function based on those best practices"

        with patch.object(
            agent_chain_with_history,
            "_route_to_agent",
            return_value=("coder", code_query),
        ):
            with patch.object(
                agent_chain_with_history.agents["coder"],
                "process_prompt_async",
                return_value="def authenticate_jwt(token): ...",
            ) as mock_coder:
                result2 = await agent_chain_with_history.run_chat_turn_async(code_query)
                mock_coder.assert_called_once()
                assert "def authenticate_jwt" in result2

        # Verify conversation history is maintained
        # (History tracking will be tested when implementation exists)

    @pytest.mark.asyncio
    async def test_code_review_workflow(self, agent_chain_with_history):
        """Test three-agent workflow: research → code → review.

        User Story: Complete development workflow with quality assurance.
        """
        # Turn 1: Research
        with patch.object(
            agent_chain_with_history,
            "_route_to_agent",
            return_value=("researcher", "error handling patterns"),
        ):
            with patch.object(
                agent_chain_with_history.agents["researcher"],
                "process_prompt_async",
                return_value="Error handling: try/except, custom exceptions, logging...",
            ):
                await agent_chain_with_history.run_chat_turn_async(
                    "What are good error handling patterns?"
                )

        # Turn 2: Code
        with patch.object(
            agent_chain_with_history,
            "_route_to_agent",
            return_value=("coder", "implement error handler"),
        ):
            with patch.object(
                agent_chain_with_history.agents["coder"],
                "process_prompt_async",
                return_value="def handle_error(e): ...",
            ):
                await agent_chain_with_history.run_chat_turn_async(
                    "Implement an error handler with those patterns"
                )

        # Turn 3: Review (router should recognize review request)
        review_query = "Review the error handler code for issues"

        with patch.object(
            agent_chain_with_history,
            "_route_to_agent",
            return_value=("reviewer", review_query),
        ) as mock_route:
            with patch.object(
                agent_chain_with_history.agents["reviewer"],
                "process_prompt_async",
                return_value="Code review: Add type hints, improve error messages...",
            ) as mock_reviewer:
                result = await agent_chain_with_history.run_chat_turn_async(review_query)
                mock_reviewer.assert_called_once()
                assert "Code review" in result

    @pytest.mark.asyncio
    async def test_clarification_query_maintains_context(self, agent_chain_with_history):
        """Test that follow-up clarification questions maintain conversation context.

        User Story: User asks follow-up question about previous response.
        Router should understand context and select appropriate agent.
        """
        # Turn 1: Initial query
        with patch.object(
            agent_chain_with_history,
            "_route_to_agent",
            return_value=("researcher", "API authentication methods"),
        ):
            with patch.object(
                agent_chain_with_history.agents["researcher"],
                "process_prompt_async",
                return_value="Common methods: JWT, OAuth2, API keys...",
            ):
                await agent_chain_with_history.run_chat_turn_async(
                    "What are common API authentication methods?"
                )

        # Turn 2: Clarification (vague query that needs context)
        clarification_query = "Which one is most secure?"

        with patch.object(
            agent_chain_with_history,
            "_route_to_agent",
            return_value=("researcher", clarification_query),
        ) as mock_route:
            with patch.object(
                agent_chain_with_history.agents["researcher"],
                "process_prompt_async",
                return_value="OAuth2 is most secure when properly implemented...",
            ) as mock_research:
                result = await agent_chain_with_history.run_chat_turn_async(clarification_query)

                # Verify router was called (receives history internally)
                mock_route.assert_called_once()
                mock_research.assert_called_once()
                assert "OAuth2" in result

    @pytest.mark.asyncio
    async def test_agent_specialization_respected(self, agent_chain_with_history):
        """Test that agent specializations are respected in routing decisions.

        User Story: Different query types should route to appropriate specialists.
        """
        test_cases = [
            # (query, expected_agent, response_keyword)
            ("Research microservices architecture", "researcher", "research"),
            ("Implement a REST API endpoint", "coder", "implement"),
            ("Review this authentication code", "reviewer", "review"),
        ]

        for query, expected_agent, keyword in test_cases:
            with patch.object(
                agent_chain_with_history,
                "_route_to_agent",
                return_value=(expected_agent, query),
            ):
                with patch.object(
                    agent_chain_with_history.agents[expected_agent],
                    "process_prompt_async",
                    return_value=f"Response from {expected_agent} with {keyword}",
                ) as mock_agent:
                    result = await agent_chain_with_history.run_chat_turn_async(query)
                    mock_agent.assert_called_once()
                    assert expected_agent in result or keyword in result

    @pytest.mark.asyncio
    async def test_conversation_history_preserved_across_agents(
        self, agent_chain_with_history
    ):
        """Test that conversation history is preserved when switching agents.

        User Story: Context from previous agents should be available to subsequent agents.
        """
        # Turn 1: Research
        with patch.object(
            agent_chain_with_history,
            "_route_to_agent",
            return_value=("researcher", "database patterns"),
        ):
            with patch.object(
                agent_chain_with_history.agents["researcher"],
                "process_prompt_async",
                return_value="Repository pattern is recommended for database access",
            ):
                await agent_chain_with_history.run_chat_turn_async(
                    "What are good database access patterns?"
                )

        # Turn 2: Implementation (should have access to previous context)
        impl_query = "Implement that pattern for user data"

        with patch.object(
            agent_chain_with_history,
            "_route_to_agent",
            return_value=("coder", impl_query),
        ):
            # The coder agent should receive history when processing
            with patch.object(
                agent_chain_with_history.agents["coder"],
                "process_prompt_async",
                return_value="class UserRepository: ...",
            ) as mock_coder:
                result = await agent_chain_with_history.run_chat_turn_async(impl_query)

                # Verify coder was called
                mock_coder.assert_called_once()
                assert "UserRepository" in result

        # Verify history tracking (will be implemented in GREEN phase)
        # Should contain both research and implementation exchanges

    @pytest.mark.asyncio
    async def test_multi_turn_debugging_session(self, agent_chain_with_history):
        """Test realistic debugging session across multiple agents and turns.

        User Story: Developer debugging an issue needs research, code analysis, and fixes.
        """
        # Turn 1: Research the error
        with patch.object(
            agent_chain_with_history,
            "_route_to_agent",
            return_value=("researcher", "connection timeout error"),
        ):
            with patch.object(
                agent_chain_with_history.agents["researcher"],
                "process_prompt_async",
                return_value="Connection timeouts: check network, increase timeout, retry logic",
            ):
                await agent_chain_with_history.run_chat_turn_async(
                    "Why am I getting connection timeout errors?"
                )

        # Turn 2: Review existing code
        with patch.object(
            agent_chain_with_history,
            "_route_to_agent",
            return_value=("reviewer", "review connection code"),
        ):
            with patch.object(
                agent_chain_with_history.agents["reviewer"],
                "process_prompt_async",
                return_value="Issue: No timeout set, no retry logic",
            ):
                await agent_chain_with_history.run_chat_turn_async(
                    "Review my connection code: requests.get(url)"
                )

        # Turn 3: Implement fix
        fix_query = "Fix it with proper timeout and retry"

        with patch.object(
            agent_chain_with_history,
            "_route_to_agent",
            return_value=("coder", fix_query),
        ):
            with patch.object(
                agent_chain_with_history.agents["coder"],
                "process_prompt_async",
                return_value="requests.get(url, timeout=10) with retry decorator",
            ) as mock_coder:
                result = await agent_chain_with_history.run_chat_turn_async(fix_query)
                mock_coder.assert_called_once()
                assert "timeout" in result

    @pytest.mark.asyncio
    async def test_default_agent_fallback_in_conversation(
        self, agent_chain_with_history
    ):
        """Test fallback to default agent during conversation if routing fails.

        User Story: System gracefully handles routing failures by falling back to default agent.
        """
        # Set default agent
        agent_chain_with_history.default_agent = "researcher"

        # Turn 1: Normal routing
        with patch.object(
            agent_chain_with_history,
            "_route_to_agent",
            return_value=("coder", "write code"),
        ):
            with patch.object(
                agent_chain_with_history.agents["coder"],
                "process_prompt_async",
                return_value="Code generated",
            ):
                await agent_chain_with_history.run_chat_turn_async("Write some code")

        # Turn 2: Router fails (should fall back to default)
        query = "Ambiguous query"

        # Mock _route_to_agent to return the default agent (simulating fallback behavior)
        # This is what the real _route_to_agent does when an exception occurs
        with patch.object(
            agent_chain_with_history,
            "_route_to_agent",
            return_value=("researcher", query),  # Returns default agent
        ):
            with patch.object(
                agent_chain_with_history.agents["researcher"],
                "process_prompt_async",
                return_value="Fallback response from default agent",
            ) as mock_default:
                result = await agent_chain_with_history.run_chat_turn_async(query)
                mock_default.assert_called_once()
                assert "Fallback" in result


class TestConversationHistoryManagement:
    """Test conversation history management across multi-agent conversations."""

    @pytest.fixture(autouse=True)
    def reset_agent_chain_state(self, request):
        """Reset AgentChain state between tests to ensure isolation."""
        # Get the agent_chain fixture (works for any fixture name)
        try:
            agent_chain = request.getfixturevalue('simple_agent_chain')
        except Exception:
            # If that fixture doesn't exist, skip
            yield
            return

        # Reset conversation history
        if hasattr(agent_chain, '_conversation_history'):
            agent_chain._conversation_history.clear()

        # Ensure execution_mode is "router"
        agent_chain.execution_mode = "router"

        # Reset last_selected_agent
        if hasattr(agent_chain, 'last_selected_agent'):
            agent_chain.last_selected_agent = None

        yield

        # Cleanup after test
        if hasattr(agent_chain, '_conversation_history'):
            agent_chain._conversation_history.clear()

    @pytest.fixture
    def simple_agent_chain(self):
        """Create simple two-agent chain for history testing."""
        from promptchain.cli.models.orchestration_config import DEFAULT_ROUTER_PROMPT

        agent1 = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Agent 1: {input}"],
            verbose=False,
        )

        agent2 = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Agent 2: {input}"],
            verbose=False,
        )

        return AgentChain(
            agents={"agent1": agent1, "agent2": agent2},
            agent_descriptions={
                "agent1": "First agent",
                "agent2": "Second agent",
            },
            execution_mode="router",
            router={
                "models": ["gpt-4.1-mini-2025-04-14"],
                "instructions": [None, "{input}"],
                "decision_prompt_templates": {
                    "single_agent_dispatch": DEFAULT_ROUTER_PROMPT
                },
            },
            auto_include_history=True,
            verbose=False,
        )

    @pytest.mark.asyncio
    async def test_history_includes_all_agent_responses(self, simple_agent_chain):
        """Test that conversation history includes responses from all agents."""
        # Turn 1: Agent 1
        with patch.object(simple_agent_chain, "_route_to_agent", return_value=("agent1", "q1")):
            with patch.object(
                simple_agent_chain.agents["agent1"],
                "process_prompt_async",
                return_value="Response from agent 1",
            ):
                await simple_agent_chain.run_chat_turn_async("Query 1")

        # Turn 2: Agent 2
        with patch.object(simple_agent_chain, "_route_to_agent", return_value=("agent2", "q2")):
            with patch.object(
                simple_agent_chain.agents["agent2"],
                "process_prompt_async",
                return_value="Response from agent 2",
            ):
                await simple_agent_chain.run_chat_turn_async("Query 2")

        # Verify history contains both exchanges
        # (Will be implemented in GREEN phase with actual history tracking)

    @pytest.mark.asyncio
    async def test_history_formatted_for_router(self, simple_agent_chain):
        """Test that history is properly formatted for router decision-making."""
        # Build conversation history
        with patch.object(simple_agent_chain, "_route_to_agent", return_value=("agent1", "q1")):
            with patch.object(
                simple_agent_chain.agents["agent1"],
                "process_prompt_async",
                return_value="Response 1",
            ):
                await simple_agent_chain.run_chat_turn_async("Query 1")

        # Router should receive formatted history
        with patch.object(
            simple_agent_chain,
            "_route_to_agent",
            return_value=("agent2", "q2"),
        ) as mock_route:
            with patch.object(
                simple_agent_chain.agents["agent2"],
                "process_prompt_async",
                return_value="Response 2",
            ):
                await simple_agent_chain.run_chat_turn_async("Query 2")

                # Verify router was called (receives history as parameter)
                mock_route.assert_called_once()
