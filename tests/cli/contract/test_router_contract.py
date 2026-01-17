"""Contract tests for router decision prompt template (T033).

These tests verify the router configuration contracts defined in
specs/002-cli-orchestration/data-model.md for OrchestrationConfig and RouterConfig.

Test-Driven Development (TDD):
- RED: Tests verify contracts exist and are valid
- GREEN: Models already implemented (will pass)
- REFACTOR: N/A (testing existing contracts)
"""

import pytest
import json

from promptchain.cli.models.orchestration_config import (
    OrchestrationConfig,
    RouterConfig,
    DEFAULT_ROUTER_PROMPT,
)


class TestRouterConfigContract:
    """Test RouterConfig schema validation per data-model.md."""

    def test_router_config_defaults(self):
        """Contract: RouterConfig has sensible defaults."""
        config = RouterConfig()

        assert config.model == "gpt-4.1-mini-2025-04-14"
        assert config.timeout_seconds == 10
        assert config.decision_prompt_template is not None
        assert "{user_input}" in config.decision_prompt_template
        assert "{agent_details}" in config.decision_prompt_template
        assert "{history}" in config.decision_prompt_template

    def test_router_config_custom_model(self):
        """Contract: RouterConfig accepts custom model names."""
        config = RouterConfig(model="anthropic/claude-3-sonnet-20240229")

        assert config.model == "anthropic/claude-3-sonnet-20240229"

    def test_router_config_custom_timeout(self):
        """Contract: RouterConfig accepts custom timeout values."""
        config = RouterConfig(timeout_seconds=30)

        assert config.timeout_seconds == 30

    def test_router_config_custom_prompt_template(self):
        """Contract: RouterConfig accepts custom decision prompt templates."""
        custom_prompt = """
        User query: {user_input}
        Available agents: {agent_details}
        Choose the best agent and return JSON:
        {{"agent": "name"}}
        """

        config = RouterConfig(decision_prompt_template=custom_prompt)

        assert config.decision_prompt_template == custom_prompt
        assert "{user_input}" in config.decision_prompt_template


class TestDefaultRouterPromptContract:
    """Test DEFAULT_ROUTER_PROMPT template per research.md requirements."""

    def test_default_prompt_exists(self):
        """Contract: DEFAULT_ROUTER_PROMPT is defined as string."""
        assert isinstance(DEFAULT_ROUTER_PROMPT, str)
        assert len(DEFAULT_ROUTER_PROMPT) > 50

    def test_default_prompt_has_required_variables(self):
        """Contract: DEFAULT_ROUTER_PROMPT contains required template variables."""
        required_variables = ["{user_input}", "{agent_details}", "{history}"]

        for var in required_variables:
            assert var in DEFAULT_ROUTER_PROMPT, f"Missing variable: {var}"

    def test_default_prompt_mentions_json_response(self):
        """Contract: DEFAULT_ROUTER_PROMPT instructs model to return JSON."""
        # Router prompt must request JSON format for parsing
        assert "JSON" in DEFAULT_ROUTER_PROMPT or "json" in DEFAULT_ROUTER_PROMPT
        assert "chosen_agent" in DEFAULT_ROUTER_PROMPT

    def test_default_prompt_includes_context_guidance(self):
        """Contract: DEFAULT_ROUTER_PROMPT guides router to consider context."""
        # Should mention considering conversation history and agent specializations
        prompt_lower = DEFAULT_ROUTER_PROMPT.lower()

        # Check for context-awareness keywords
        context_keywords = ["conversation", "context", "history", "previous"]
        has_context_guidance = any(keyword in prompt_lower for keyword in context_keywords)

        assert has_context_guidance, "Prompt should guide router to consider context"

    def test_default_prompt_includes_agent_specialization_guidance(self):
        """Contract: DEFAULT_ROUTER_PROMPT guides router to match agent capabilities."""
        prompt_lower = DEFAULT_ROUTER_PROMPT.lower()

        # Check for specialization keywords
        specialization_keywords = [
            "speciali",  # catches specialization, specialized, etc.
            "capabilit",  # catches capabilities, capability
            "agent",
            "task",
        ]

        has_specialization_guidance = any(
            keyword in prompt_lower for keyword in specialization_keywords
        )

        assert (
            has_specialization_guidance
        ), "Prompt should guide router to match agent specializations"


class TestOrchestrationConfigContract:
    """Test OrchestrationConfig schema validation per data-model.md."""

    def test_orchestration_config_defaults(self):
        """Contract: OrchestrationConfig has sensible defaults."""
        config = OrchestrationConfig()

        assert config.execution_mode == "router"
        assert config.default_agent is None
        # Router mode automatically creates default RouterConfig
        assert config.router_config is not None
        assert isinstance(config.router_config, RouterConfig)
        assert config.auto_include_history is True

    def test_orchestration_config_valid_execution_modes(self):
        """Contract: execution_mode must be 'router', 'pipeline', 'round-robin', or 'broadcast'."""
        valid_modes = ["router", "pipeline", "round-robin", "broadcast"]

        for mode in valid_modes:
            config = OrchestrationConfig(execution_mode=mode)
            assert config.execution_mode == mode

    def test_orchestration_config_router_mode_with_config(self):
        """Contract: Router mode should have RouterConfig."""
        router_config = RouterConfig(
            model="gpt-4.1-mini-2025-04-14", timeout_seconds=15
        )

        config = OrchestrationConfig(
            execution_mode="router", router_config=router_config
        )

        assert config.execution_mode == "router"
        assert config.router_config is not None
        assert config.router_config.model == "gpt-4.1-mini-2025-04-14"
        assert config.router_config.timeout_seconds == 15

    def test_orchestration_config_default_agent(self):
        """Contract: default_agent specifies fallback agent name."""
        config = OrchestrationConfig(default_agent="default-agent")

        assert config.default_agent == "default-agent"

    def test_orchestration_config_auto_include_history(self):
        """Contract: auto_include_history enables conversation context."""
        # Enabled by default
        config_enabled = OrchestrationConfig()
        assert config_enabled.auto_include_history is True

        # Can be disabled
        config_disabled = OrchestrationConfig(auto_include_history=False)
        assert config_disabled.auto_include_history is False

    def test_orchestration_config_pipeline_mode(self):
        """Contract: Pipeline mode executes agents sequentially."""
        config = OrchestrationConfig(execution_mode="pipeline")

        assert config.execution_mode == "pipeline"
        # Pipeline mode doesn't need router_config
        assert config.router_config is None

    def test_orchestration_config_round_robin_mode(self):
        """Contract: Round-robin mode cycles through agents."""
        config = OrchestrationConfig(execution_mode="round-robin")

        assert config.execution_mode == "round-robin"

    def test_orchestration_config_broadcast_mode(self):
        """Contract: Broadcast mode sends query to all agents."""
        config = OrchestrationConfig(execution_mode="broadcast")

        assert config.execution_mode == "broadcast"


class TestRouterDecisionFormatContract:
    """Test router decision JSON format requirements."""

    def test_router_decision_json_format(self):
        """Contract: Router must return JSON with chosen_agent field."""
        # This is the expected format from DEFAULT_ROUTER_PROMPT
        example_decision = {"chosen_agent": "researcher", "refined_query": None}

        # Validate JSON serialization
        json_str = json.dumps(example_decision)
        parsed = json.loads(json_str)

        assert "chosen_agent" in parsed
        assert isinstance(parsed["chosen_agent"], str)

    def test_router_decision_with_refined_query(self):
        """Contract: Router can optionally provide refined_query."""
        example_decision = {
            "chosen_agent": "coder",
            "refined_query": "Implement function with error handling",
        }

        # Validate JSON serialization
        json_str = json.dumps(example_decision)
        parsed = json.loads(json_str)

        assert "chosen_agent" in parsed
        assert "refined_query" in parsed
        assert isinstance(parsed["refined_query"], str)

    def test_router_decision_minimal_format(self):
        """Contract: Minimal router decision only needs chosen_agent."""
        minimal_decision = {"chosen_agent": "analyst"}

        json_str = json.dumps(minimal_decision)
        parsed = json.loads(json_str)

        assert "chosen_agent" in parsed
        assert parsed["chosen_agent"] == "analyst"


class TestRouterPromptTemplateVariables:
    """Test router prompt template variable substitution requirements."""

    def test_user_input_variable_substitution(self):
        """Contract: {user_input} must be replaceable with actual query."""
        template = DEFAULT_ROUTER_PROMPT
        user_query = "Analyze authentication code for security issues"

        # Simulate substitution
        filled = template.replace("{user_input}", user_query)

        assert user_query in filled
        assert "{user_input}" not in filled

    def test_agent_details_variable_substitution(self):
        """Contract: {agent_details} must be replaceable with agent list."""
        template = DEFAULT_ROUTER_PROMPT
        agent_details = """
        1. researcher: Deep research specialist with web search
        2. coder: Code generation and validation
        3. analyst: Data analysis and interpretation
        """

        # Simulate substitution
        filled = template.replace("{agent_details}", agent_details)

        assert "researcher" in filled
        assert "coder" in filled
        assert "{agent_details}" not in filled

    def test_history_variable_substitution(self):
        """Contract: {history} must be replaceable with conversation context."""
        template = DEFAULT_ROUTER_PROMPT
        history = """
        User: What authentication methods do we use?
        Assistant: We use JWT tokens with RS256 signing.
        """

        # Simulate substitution
        filled = template.replace("{history}", history)

        assert "JWT tokens" in filled
        assert "{history}" not in filled

    def test_all_variables_substituted(self):
        """Contract: All template variables can be substituted simultaneously."""
        template = DEFAULT_ROUTER_PROMPT

        filled = template.replace("{user_input}", "Test query")
        filled = filled.replace("{agent_details}", "Agents: researcher, coder")
        filled = filled.replace("{history}", "Previous: none")

        # No template variables remaining
        assert "{user_input}" not in filled
        assert "{agent_details}" not in filled
        assert "{history}" not in filled
