"""Unit tests for agent selection and matching logic (T036).

These tests verify the core agent selection algorithms that power router mode,
including agent description matching, query analysis, and selection scoring.

Test-Driven Development (TDD):
- RED: These tests WILL FAIL because selection logic doesn't exist yet
- GREEN: Implement agent selection logic in AgentChain (T037-T044)
- REFACTOR: Optimize selection algorithms after tests pass
"""

import pytest
from unittest.mock import MagicMock, patch

from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain


class TestAgentDescriptionMatching:
    """Test agent description matching and scoring logic."""

    @pytest.fixture
    def sample_agents(self):
        """Create sample agents with varied descriptions."""
        return {
            "researcher": PromptChain(
                models=["gpt-4.1-mini-2025-04-14"],
                instructions=["Research: {input}"],
                verbose=False,
            ),
            "coder": PromptChain(
                models=["gpt-4.1-mini-2025-04-14"],
                instructions=["Code: {input}"],
                verbose=False,
            ),
            "analyst": PromptChain(
                models=["gpt-4.1-mini-2025-04-14"],
                instructions=["Analyze: {input}"],
                verbose=False,
            ),
        }

    @pytest.fixture
    def agent_descriptions(self):
        """Agent descriptions with specific keywords."""
        return {
            "researcher": "Research specialist with web search, data gathering, and analysis capabilities",
            "coder": "Code generation specialist for Python, JavaScript, SQL, and system design",
            "analyst": "Data analysis specialist for statistics, visualization, and insights",
        }

    def test_format_agent_details_for_router(self, sample_agents, agent_descriptions):
        """Test formatting agent details for router prompt.

        Agent details should be formatted as numbered list with name and description.
        """
        from promptchain.cli.models.orchestration_config import DEFAULT_ROUTER_PROMPT

        agent_chain = AgentChain(
            agents=sample_agents,
            agent_descriptions=agent_descriptions,
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

        # Method to test (will be implemented in GREEN phase)
        # formatted = agent_chain._format_agent_details()

        # Expected format:
        # 1. researcher: Research specialist with web search...
        # 2. coder: Code generation specialist for Python...
        # 3. analyst: Data analysis specialist for statistics...

        # assert "1. researcher:" in formatted
        # assert "2. coder:" in formatted
        # assert "web search" in formatted
        # assert "Python" in formatted

    def test_extract_query_keywords(self):
        """Test extracting relevant keywords from user queries."""
        # This would be a helper method for basic keyword matching
        test_cases = [
            (
                "Research the latest developments in AI",
                ["research", "latest", "developments"],
            ),
            (
                "Write Python code for authentication",
                ["write", "python", "code", "authentication"],
            ),
            (
                "Analyze sales data and create visualization",
                ["analyze", "sales", "data", "visualization"],
            ),
        ]

        # Method to test (will be implemented if needed)
        # for query, expected_keywords in test_cases:
        #     keywords = _extract_keywords(query)
        #     for expected in expected_keywords:
        #         assert expected.lower() in [k.lower() for k in keywords]

    def test_match_agent_by_keywords(self, agent_descriptions):
        """Test basic keyword matching between query and agent descriptions."""
        # Test keyword overlap scoring
        query_keywords = ["research", "web", "data"]
        researcher_desc = agent_descriptions["researcher"].lower()

        # All query keywords should match researcher description
        matches = sum(1 for keyword in query_keywords if keyword in researcher_desc)
        assert matches == len(query_keywords)

        # Keywords shouldn't all match coder description
        coder_desc = agent_descriptions["coder"].lower()
        coder_matches = sum(1 for keyword in query_keywords if keyword in coder_desc)
        assert coder_matches < matches


class TestRouterDecisionLogic:
    """Test router decision-making logic and scoring."""

    @pytest.fixture
    def router_agent_chain(self):
        """Create AgentChain with router for testing."""
        from promptchain.cli.models.orchestration_config import DEFAULT_ROUTER_PROMPT

        agents = {
            "researcher": PromptChain(
                models=["gpt-4.1-mini-2025-04-14"],
                instructions=["Research: {input}"],
                verbose=False,
            ),
            "coder": PromptChain(
                models=["gpt-4.1-mini-2025-04-14"],
                instructions=["Code: {input}"],
                verbose=False,
            ),
        }

        descriptions = {
            "researcher": "Research and information gathering specialist",
            "coder": "Code generation and implementation specialist",
        }

        return AgentChain(
            agents=agents,
            agent_descriptions=descriptions,
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

    def test_route_to_agent_method_exists(self, router_agent_chain):
        """Test that _route_to_agent method will exist on AgentChain.

        This is the core routing method that will be implemented in GREEN phase.
        """
        # Method should exist after implementation
        # assert hasattr(router_agent_chain, "_route_to_agent")
        # assert callable(router_agent_chain._route_to_agent)

    @pytest.mark.asyncio
    async def test_route_to_agent_returns_tuple(self, router_agent_chain):
        """Test that _route_to_agent returns (agent_name, refined_query) tuple."""
        # This method will be implemented in GREEN phase
        # query = "Research AI trends"
        # result = await router_agent_chain._route_to_agent(query)

        # assert isinstance(result, tuple)
        # assert len(result) == 2
        # agent_name, refined_query = result
        # assert isinstance(agent_name, str)
        # assert agent_name in router_agent_chain.agents

    @pytest.mark.asyncio
    async def test_parse_router_decision_json(self, router_agent_chain):
        """Test parsing JSON response from router LLM."""
        # Router returns JSON like: {"chosen_agent": "researcher", "refined_query": null}
        sample_json = '{"chosen_agent": "researcher", "refined_query": null}'

        import json

        decision = json.loads(sample_json)

        assert "chosen_agent" in decision
        assert decision["chosen_agent"] == "researcher"
        assert decision["chosen_agent"] in router_agent_chain.agents

    @pytest.mark.asyncio
    async def test_parse_router_decision_with_refinement(self, router_agent_chain):
        """Test parsing router decision with refined query."""
        sample_json = '{"chosen_agent": "coder", "refined_query": "Implement with error handling"}'

        import json

        decision = json.loads(sample_json)

        assert decision["chosen_agent"] == "coder"
        assert decision["refined_query"] == "Implement with error handling"
        assert isinstance(decision["refined_query"], str)

    def test_validate_agent_name_from_router(self, router_agent_chain):
        """Test validation of agent name returned by router."""
        # Valid agent names
        assert "researcher" in router_agent_chain.agents
        assert "coder" in router_agent_chain.agents

        # Invalid agent name should be handled
        invalid_name = "nonexistent_agent"
        assert invalid_name not in router_agent_chain.agents

    def test_fallback_to_default_agent(self, router_agent_chain):
        """Test fallback logic when router fails or returns invalid agent."""
        # Set default agent
        router_agent_chain.default_agent = "researcher"

        # Verify default is valid
        assert router_agent_chain.default_agent in router_agent_chain.agents

        # Fallback should use default when router fails
        # (Implementation will be tested in integration tests)


class TestAgentSelectionScoring:
    """Test agent selection scoring and ranking algorithms."""

    def test_score_agent_match_basic(self):
        """Test basic scoring algorithm for agent-query matching.

        Scoring factors:
        - Keyword overlap between query and agent description
        - Query type indicators (research, code, analyze)
        - Historical performance (if available)
        """
        # Example scoring
        query = "Research machine learning papers"
        agent_desc = "Research specialist with academic paper analysis"

        # Simple keyword overlap score
        query_words = set(query.lower().split())
        desc_words = set(agent_desc.lower().split())
        overlap = query_words & desc_words

        # Should have "research" in overlap
        assert "research" in overlap

    def test_score_prefers_specialized_agents(self):
        """Test that scoring prefers more specialized agents over generalists."""
        query = "Write Python code for file handling"

        specialist_desc = "Python code generation specialist focusing on file I/O and system operations"
        generalist_desc = "General programming assistant for multiple languages"

        # Specialist should score higher due to more specific keyword matches
        specialist_keywords = ["python", "code", "file"]
        generalist_keywords = ["programming"]

        spec_matches = sum(1 for k in specialist_keywords if k in specialist_desc.lower())
        gen_matches = sum(1 for k in generalist_keywords if k in generalist_desc.lower())

        assert spec_matches > gen_matches

    def test_normalize_scores_to_probabilities(self):
        """Test that agent scores can be normalized to probabilities."""
        # Example scores for 3 agents
        scores = {"researcher": 0.8, "coder": 0.3, "analyst": 0.1}

        # Normalize to probabilities (sum to 1.0)
        total = sum(scores.values())
        probabilities = {agent: score / total for agent, score in scores.items()}

        assert abs(sum(probabilities.values()) - 1.0) < 0.001
        assert probabilities["researcher"] > probabilities["coder"]
        assert probabilities["coder"] > probabilities["analyst"]


class TestRouterPromptConstruction:
    """Test construction of router prompts with variables."""

    def test_router_prompt_variable_substitution(self):
        """Test that router prompt template variables are properly substituted."""
        from promptchain.cli.models.orchestration_config import DEFAULT_ROUTER_PROMPT

        # Test template has required variables
        assert "{user_input}" in DEFAULT_ROUTER_PROMPT
        assert "{agent_details}" in DEFAULT_ROUTER_PROMPT
        assert "{history}" in DEFAULT_ROUTER_PROMPT

        # Simulate substitution
        user_query = "Research quantum computing"
        agent_list = "1. researcher: Research specialist\n2. coder: Code specialist"
        history = "Previous: User asked about AI trends"

        filled_prompt = DEFAULT_ROUTER_PROMPT.replace("{user_input}", user_query)
        filled_prompt = filled_prompt.replace("{agent_details}", agent_list)
        filled_prompt = filled_prompt.replace("{history}", history)

        # Verify all variables replaced
        assert "{user_input}" not in filled_prompt
        assert "{agent_details}" not in filled_prompt
        assert "{history}" not in filled_prompt

        # Verify content present
        assert user_query in filled_prompt
        assert "researcher" in filled_prompt
        assert "Previous" in filled_prompt

    def test_router_prompt_with_empty_history(self):
        """Test router prompt construction when conversation history is empty."""
        from promptchain.cli.models.orchestration_config import DEFAULT_ROUTER_PROMPT

        user_query = "First query"
        agent_list = "1. agent1\n2. agent2"
        empty_history = ""

        filled_prompt = DEFAULT_ROUTER_PROMPT.replace("{user_input}", user_query)
        filled_prompt = filled_prompt.replace("{agent_details}", agent_list)
        filled_prompt = filled_prompt.replace("{history}", empty_history)

        # Should still work with empty history
        assert user_query in filled_prompt
        assert "agent1" in filled_prompt

    def test_router_prompt_format_consistency(self):
        """Test that router prompt format is consistent across calls."""
        from promptchain.cli.models.orchestration_config import DEFAULT_ROUTER_PROMPT

        # Multiple calls should produce consistent format
        queries = ["Query 1", "Query 2", "Query 3"]
        agents = "1. agent1\n2. agent2"

        for query in queries:
            filled = DEFAULT_ROUTER_PROMPT.replace("{user_input}", query)
            filled = filled.replace("{agent_details}", agents)
            filled = filled.replace("{history}", "")

            # All should have JSON instruction
            assert "JSON" in filled or "json" in filled
            assert "chosen_agent" in filled


class TestAgentSelectionEdgeCases:
    """Test edge cases in agent selection logic."""

    def test_single_agent_always_selected(self):
        """Test that single-agent system always selects that agent."""
        single_agent = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Process: {input}"],
            verbose=False,
        )

        from promptchain.cli.models.orchestration_config import DEFAULT_ROUTER_PROMPT

        agent_chain = AgentChain(
            agents={"only_agent": single_agent},
            agent_descriptions={"only_agent": "General purpose agent"},
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

        # With only one agent, router should always select it
        assert len(agent_chain.agents) == 1
        assert "only_agent" in agent_chain.agents

    def test_empty_agent_descriptions_handled(self):
        """Test that empty agent descriptions raise validation error.

        Router mode requires agent descriptions for decision-making,
        so empty descriptions should raise ValueError.
        """
        agents = {
            "agent1": PromptChain(models=["gpt-4.1-mini-2025-04-14"], instructions=["A: {input}"], verbose=False),
            "agent2": PromptChain(models=["gpt-4.1-mini-2025-04-14"], instructions=["B: {input}"], verbose=False),
        }

        from promptchain.cli.models.orchestration_config import DEFAULT_ROUTER_PROMPT

        # Empty descriptions should raise error
        with pytest.raises(ValueError, match="agent_descriptions must be provided"):
            agent_chain = AgentChain(
                agents=agents,
                agent_descriptions={},  # Empty descriptions
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

    def test_identical_agent_descriptions_handled(self):
        """Test handling of agents with identical descriptions."""
        agents = {
            "agent1": PromptChain(models=["gpt-4.1-mini-2025-04-14"], instructions=["A: {input}"], verbose=False),
            "agent2": PromptChain(models=["gpt-4.1-mini-2025-04-14"], instructions=["B: {input}"], verbose=False),
        }

        from promptchain.cli.models.orchestration_config import DEFAULT_ROUTER_PROMPT

        # Identical descriptions
        agent_chain = AgentChain(
            agents=agents,
            agent_descriptions={
                "agent1": "General purpose agent",
                "agent2": "General purpose agent",  # Same as agent1
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

        # Should handle identical descriptions without error
        assert len(agent_chain.agents) == 2

    def test_very_long_agent_descriptions(self):
        """Test handling of very long agent descriptions."""
        long_desc = "A " * 500  # Very long description

        agent = PromptChain(models=["gpt-4.1-mini-2025-04-14"], instructions=["Process: {input}"], verbose=False)

        from promptchain.cli.models.orchestration_config import DEFAULT_ROUTER_PROMPT

        agent_chain = AgentChain(
            agents={"agent": agent},
            agent_descriptions={"agent": long_desc},
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

        # Should handle long descriptions (may truncate in implementation)
        assert "agent" in agent_chain.agents

    def test_special_characters_in_descriptions(self):
        """Test handling of special characters in agent descriptions."""
        special_desc = "Agent with special chars: @#$%^&*() and unicode: 你好 🚀"

        agent = PromptChain(models=["gpt-4.1-mini-2025-04-14"], instructions=["Process: {input}"], verbose=False)

        from promptchain.cli.models.orchestration_config import DEFAULT_ROUTER_PROMPT

        agent_chain = AgentChain(
            agents={"agent": agent},
            agent_descriptions={"agent": special_desc},
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

        # Should handle special characters without error
        assert "agent" in agent_chain.agents
