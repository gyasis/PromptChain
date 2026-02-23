"""Unit tests for agent template configuration validation (T094-T097).

Tests template structure, history configs, instruction chains, and metadata
for all four agent templates: researcher, coder, analyst, terminal.
"""

import pytest

from promptchain.cli.models.agent_config import Agent, HistoryConfig
from promptchain.cli.utils.agent_templates import (
    AGENT_TEMPLATES,
    ANALYST_TEMPLATE,
    CODER_TEMPLATE,
    RESEARCHER_TEMPLATE,
    TERMINAL_TEMPLATE,
    create_from_template,
    get_template_info,
    list_templates,
    validate_template_name,
)


# ============================================================================
# T094: Researcher Template Configuration Validation
# ============================================================================


class TestResearcherTemplate:
    """Test suite for researcher template configuration (T094)."""

    def test_researcher_template_exists(self):
        """Researcher template must be registered in AGENT_TEMPLATES."""
        assert "researcher" in AGENT_TEMPLATES
        assert AGENT_TEMPLATES["researcher"] == RESEARCHER_TEMPLATE

    def test_researcher_template_name(self):
        """Researcher template must have correct name and display_name."""
        assert RESEARCHER_TEMPLATE.name == "researcher"
        assert RESEARCHER_TEMPLATE.display_name == "Researcher"

    def test_researcher_template_description(self):
        """Researcher template must have meaningful description."""
        description = RESEARCHER_TEMPLATE.description
        assert len(description) > 20
        assert "research" in description.lower()
        assert "AgenticStepProcessor" in description

    def test_researcher_template_model(self):
        """Researcher template must use gpt-4 model."""
        assert RESEARCHER_TEMPLATE.model == "openai/gpt-4"

    def test_researcher_template_instruction_chain(self):
        """Researcher template must have 3-step instruction chain with agentic step."""
        chain = RESEARCHER_TEMPLATE.instruction_chain
        assert len(chain) == 3

        # Step 1: Initial analysis (string)
        assert isinstance(chain[0], str)
        assert "{input}" in chain[0]
        assert "research" in chain[0].lower()

        # Step 2: AgenticStepProcessor (dict)
        assert isinstance(chain[1], dict)
        assert chain[1]["type"] == "agentic_step"
        assert chain[1]["max_internal_steps"] == 8
        assert "research" in chain[1]["objective"].lower()
        assert "web_search" in chain[1]["tools"]

        # Step 3: Synthesis (string)
        assert isinstance(chain[2], str)
        assert "{input}" in chain[2]
        assert "synthesize" in chain[2].lower() or "report" in chain[2].lower()

    def test_researcher_template_tools(self):
        """Researcher template must have web_search tool."""
        assert "web_search" in RESEARCHER_TEMPLATE.tools

    def test_researcher_template_history_config(self):
        """Researcher template must have high token limit (8000 tokens, 50 entries)."""
        config = RESEARCHER_TEMPLATE.history_config
        assert isinstance(config, HistoryConfig)
        assert config.enabled is True
        assert config.max_tokens == 8000
        assert config.max_entries == 50
        assert config.truncation_strategy == "oldest_first"

    def test_researcher_template_metadata(self):
        """Researcher template must have metadata with category and complexity."""
        metadata = RESEARCHER_TEMPLATE.metadata
        assert metadata["category"] == "research"
        assert metadata["complexity"] == "high"
        assert metadata["token_usage"] == "high"


# ============================================================================
# T095: Coder Template Configuration Validation
# ============================================================================


class TestCoderTemplate:
    """Test suite for coder template configuration (T095)."""

    def test_coder_template_exists(self):
        """Coder template must be registered in AGENT_TEMPLATES."""
        assert "coder" in AGENT_TEMPLATES
        assert AGENT_TEMPLATES["coder"] == CODER_TEMPLATE

    def test_coder_template_name(self):
        """Coder template must have correct name and display_name."""
        assert CODER_TEMPLATE.name == "coder"
        assert CODER_TEMPLATE.display_name == "Coder"

    def test_coder_template_description(self):
        """Coder template must have meaningful description."""
        description = CODER_TEMPLATE.description
        assert len(description) > 20
        assert "code" in description.lower()
        assert "generation" in description.lower() or "file" in description.lower()

    def test_coder_template_model(self):
        """Coder template must use gpt-4 model."""
        assert CODER_TEMPLATE.model == "openai/gpt-4"

    def test_coder_template_instruction_chain(self):
        """Coder template must have 3-step instruction chain with validation step."""
        chain = CODER_TEMPLATE.instruction_chain
        assert len(chain) == 3

        # Step 1: Requirements analysis (string)
        assert isinstance(chain[0], str)
        assert "{input}" in chain[0]
        assert "coding" in chain[0].lower() or "requirements" in chain[0].lower()

        # Step 2: Code generation (string)
        assert isinstance(chain[1], str)
        assert "{input}" in chain[1]
        assert "implementation" in chain[1].lower() or "generate" in chain[1].lower()

        # Step 3: AgenticStepProcessor for validation (dict)
        assert isinstance(chain[2], dict)
        assert chain[2]["type"] == "agentic_step"
        assert chain[2]["max_internal_steps"] == 5
        assert "validate" in chain[2]["objective"].lower() or "test" in chain[2]["objective"].lower()
        assert "execute_code" in chain[2]["tools"] or "run_tests" in chain[2]["tools"]

    def test_coder_template_tools(self):
        """Coder template must have file ops and code execution tools."""
        tools = CODER_TEMPLATE.tools
        assert "mcp_filesystem_read" in tools
        assert "mcp_filesystem_write" in tools
        assert "execute_code" in tools

    def test_coder_template_history_config(self):
        """Coder template must have moderate token limit (4000 tokens, 20 entries)."""
        config = CODER_TEMPLATE.history_config
        assert isinstance(config, HistoryConfig)
        assert config.enabled is True
        assert config.max_tokens == 4000
        assert config.max_entries == 20
        assert config.truncation_strategy == "oldest_first"

    def test_coder_template_metadata(self):
        """Coder template must have metadata with category and complexity."""
        metadata = CODER_TEMPLATE.metadata
        assert metadata["category"] == "development"
        assert metadata["complexity"] == "high"
        assert metadata["token_usage"] == "medium"


# ============================================================================
# T096: Analyst Template Configuration Validation
# ============================================================================


class TestAnalystTemplate:
    """Test suite for analyst template configuration (T096)."""

    def test_analyst_template_exists(self):
        """Analyst template must be registered in AGENT_TEMPLATES."""
        assert "analyst" in AGENT_TEMPLATES
        assert AGENT_TEMPLATES["analyst"] == ANALYST_TEMPLATE

    def test_analyst_template_name(self):
        """Analyst template must have correct name and display_name."""
        assert ANALYST_TEMPLATE.name == "analyst"
        assert ANALYST_TEMPLATE.display_name == "Analyst"

    def test_analyst_template_description(self):
        """Analyst template must have meaningful description."""
        description = ANALYST_TEMPLATE.description
        assert len(description) > 20
        assert "data" in description.lower() or "analysis" in description.lower()
        assert "insights" in description.lower() or "statistical" in description.lower()

    def test_analyst_template_model(self):
        """Analyst template must use gpt-4 model."""
        assert ANALYST_TEMPLATE.model == "openai/gpt-4"

    def test_analyst_template_instruction_chain(self):
        """Analyst template must have 4-step analysis workflow."""
        chain = ANALYST_TEMPLATE.instruction_chain
        assert len(chain) == 4

        # All steps should be strings (no agentic step)
        for step in chain:
            assert isinstance(step, str)
            assert "{input}" in step

        # Verify logical analysis flow
        assert "understand" in chain[0].lower() or "objectives" in chain[0].lower()
        assert "load" in chain[1].lower() or "explore" in chain[1].lower()
        assert "statistical" in chain[2].lower() or "analysis" in chain[2].lower()
        assert "visualizations" in chain[3].lower() or "report" in chain[3].lower()

    def test_analyst_template_tools(self):
        """Analyst template must have data analysis tools."""
        tools = ANALYST_TEMPLATE.tools
        assert "mcp_filesystem_read" in tools
        assert "data_analysis" in tools

    def test_analyst_template_history_config(self):
        """Analyst template must have high token limit (8000 tokens, 50 entries)."""
        config = ANALYST_TEMPLATE.history_config
        assert isinstance(config, HistoryConfig)
        assert config.enabled is True
        assert config.max_tokens == 8000
        assert config.max_entries == 50
        assert config.truncation_strategy == "oldest_first"

    def test_analyst_template_metadata(self):
        """Analyst template must have metadata with category and complexity."""
        metadata = ANALYST_TEMPLATE.metadata
        assert metadata["category"] == "analysis"
        assert metadata["complexity"] == "medium"
        assert metadata["token_usage"] == "medium-high"


# ============================================================================
# T097: Terminal Template Configuration Validation
# ============================================================================


class TestTerminalTemplate:
    """Test suite for terminal template configuration (T097)."""

    def test_terminal_template_exists(self):
        """Terminal template must be registered in AGENT_TEMPLATES."""
        assert "terminal" in AGENT_TEMPLATES
        assert AGENT_TEMPLATES["terminal"] == TERMINAL_TEMPLATE

    def test_terminal_template_name(self):
        """Terminal template must have correct name and display_name."""
        assert TERMINAL_TEMPLATE.name == "terminal"
        assert TERMINAL_TEMPLATE.display_name == "Terminal"

    def test_terminal_template_description(self):
        """Terminal template must have meaningful description."""
        description = TERMINAL_TEMPLATE.description
        assert len(description) > 20
        assert "terminal" in description.lower() or "execution" in description.lower()
        assert "fast" in description.lower()
        assert "60%" in description  # Token savings claim

    def test_terminal_template_model(self):
        """Terminal template must use fast gpt-3.5-turbo model."""
        assert TERMINAL_TEMPLATE.model == "openai/gpt-3.5-turbo"

    def test_terminal_template_instruction_chain(self):
        """Terminal template must have single-step direct pass-through."""
        chain = TERMINAL_TEMPLATE.instruction_chain
        assert len(chain) == 1

        # Single step: direct pass-through
        assert isinstance(chain[0], str)
        assert chain[0] == "{input}"  # No additional processing

    def test_terminal_template_tools(self):
        """Terminal template must have shell execution and file ops tools."""
        tools = TERMINAL_TEMPLATE.tools
        assert "execute_shell" in tools
        assert "mcp_filesystem_read" in tools
        assert "mcp_filesystem_write" in tools

    def test_terminal_template_history_config(self):
        """Terminal template must have history DISABLED (0 tokens, 0 entries)."""
        config = TERMINAL_TEMPLATE.history_config
        assert isinstance(config, HistoryConfig)
        assert config.enabled is False
        assert config.max_tokens == 0
        assert config.max_entries == 0

    def test_terminal_template_metadata(self):
        """Terminal template must have metadata with category and response_time."""
        metadata = TERMINAL_TEMPLATE.metadata
        assert metadata["category"] == "execution"
        assert metadata["complexity"] == "low"
        assert metadata["token_usage"] == "minimal"
        assert metadata["response_time"] == "fast"


# ============================================================================
# Cross-Template Validation
# ============================================================================


class TestAllTemplates:
    """Test suite for cross-template validation."""

    def test_all_four_templates_registered(self):
        """All four templates must be registered."""
        assert len(AGENT_TEMPLATES) == 4
        assert "researcher" in AGENT_TEMPLATES
        assert "coder" in AGENT_TEMPLATES
        assert "analyst" in AGENT_TEMPLATES
        assert "terminal" in AGENT_TEMPLATES

    def test_all_templates_have_required_fields(self):
        """All templates must have required fields."""
        required_fields = ["name", "display_name", "description", "model", "instruction_chain", "tools", "history_config", "metadata"]

        for template_name, template in AGENT_TEMPLATES.items():
            for field in required_fields:
                assert hasattr(template, field), f"Template '{template_name}' missing field '{field}'"

    def test_all_templates_have_unique_names(self):
        """All templates must have unique names."""
        names = [t.name for t in AGENT_TEMPLATES.values()]
        assert len(names) == len(set(names))

    def test_all_templates_have_non_empty_instruction_chains(self):
        """All templates must have at least one instruction."""
        for template_name, template in AGENT_TEMPLATES.items():
            assert len(template.instruction_chain) >= 1, f"Template '{template_name}' has empty instruction chain"

    def test_all_templates_have_valid_models(self):
        """All templates must have valid LiteLLM model strings."""
        for template_name, template in AGENT_TEMPLATES.items():
            assert "/" in template.model, f"Template '{template_name}' has invalid model format: {template.model}"

    def test_terminal_is_only_template_with_disabled_history(self):
        """Only terminal template should have history disabled."""
        for template_name, template in AGENT_TEMPLATES.items():
            if template_name == "terminal":
                assert not template.history_config.enabled
            else:
                assert template.history_config.enabled

    def test_researcher_and_analyst_have_highest_token_limits(self):
        """Researcher and analyst should have 8000 token limits."""
        assert RESEARCHER_TEMPLATE.history_config.max_tokens == 8000
        assert ANALYST_TEMPLATE.history_config.max_tokens == 8000

    def test_coder_has_moderate_token_limit(self):
        """Coder should have 4000 token limit."""
        assert CODER_TEMPLATE.history_config.max_tokens == 4000

    def test_terminal_uses_fastest_model(self):
        """Terminal should use gpt-3.5-turbo for speed."""
        assert TERMINAL_TEMPLATE.model == "openai/gpt-3.5-turbo"
        # All others should use gpt-4
        assert RESEARCHER_TEMPLATE.model == "openai/gpt-4"
        assert CODER_TEMPLATE.model == "openai/gpt-4"
        assert ANALYST_TEMPLATE.model == "openai/gpt-4"


# ============================================================================
# Template Creation and Utility Functions
# ============================================================================


class TestCreateFromTemplate:
    """Test suite for create_from_template() function."""

    def test_create_researcher_agent_from_template(self):
        """Create researcher agent from template."""
        agent = create_from_template("researcher", "my-researcher")

        assert isinstance(agent, Agent)
        assert agent.name == "my-researcher"
        assert agent.model_name == "openai/gpt-4"
        assert "research" in agent.description.lower()
        assert len(agent.instruction_chain) == 3
        assert agent.history_config.max_tokens == 8000
        assert agent.metadata["template"] == "researcher"

    def test_create_coder_agent_from_template(self):
        """Create coder agent from template."""
        agent = create_from_template("coder", "python-dev")

        assert isinstance(agent, Agent)
        assert agent.name == "python-dev"
        assert agent.model_name == "openai/gpt-4"
        assert "code" in agent.description.lower()
        assert len(agent.instruction_chain) == 3
        assert agent.history_config.max_tokens == 4000

    def test_create_analyst_agent_from_template(self):
        """Create analyst agent from template."""
        agent = create_from_template("analyst", "data-analyst")

        assert isinstance(agent, Agent)
        assert agent.name == "data-analyst"
        assert agent.model_name == "openai/gpt-4"
        assert "analysis" in agent.description.lower()
        assert len(agent.instruction_chain) == 4
        assert agent.history_config.max_tokens == 8000

    def test_create_terminal_agent_from_template(self):
        """Create terminal agent from template."""
        agent = create_from_template("terminal", "bash-exec")

        assert isinstance(agent, Agent)
        assert agent.name == "bash-exec"
        assert agent.model_name == "openai/gpt-3.5-turbo"
        assert "terminal" in agent.description.lower() or "execution" in agent.description.lower()
        assert len(agent.instruction_chain) == 1
        assert agent.history_config.enabled is False

    def test_create_with_model_override(self):
        """Create agent with custom model override."""
        agent = create_from_template("researcher", "claude-researcher", model_override="anthropic/claude-3-opus")

        assert agent.model_name == "anthropic/claude-3-opus"
        # Other properties should remain from template
        assert agent.history_config.max_tokens == 8000

    def test_create_with_description_override(self):
        """Create agent with custom description override."""
        custom_desc = "Custom ML research specialist"
        agent = create_from_template("researcher", "ml-researcher", description_override=custom_desc)

        assert agent.description == custom_desc
        # Other properties should remain from template
        assert agent.history_config.max_tokens == 8000

    def test_create_from_invalid_template_raises_error(self):
        """Creating from non-existent template should raise ValueError."""
        with pytest.raises(ValueError, match="Template 'invalid' not found"):
            create_from_template("invalid", "test-agent")

    def test_created_agent_has_timestamp(self):
        """Created agent should have creation timestamp."""
        agent = create_from_template("researcher", "test")
        assert agent.created_at > 0

    def test_created_agent_instruction_chain_is_copy(self):
        """Created agent should have independent instruction chain copy."""
        agent1 = create_from_template("researcher", "agent1")
        agent2 = create_from_template("researcher", "agent2")

        # Modify agent1's chain
        agent1.instruction_chain.append("extra step")

        # agent2 should not be affected
        assert len(agent1.instruction_chain) == 4
        assert len(agent2.instruction_chain) == 3


class TestListTemplates:
    """Test suite for list_templates() function."""

    def test_list_templates_returns_all_templates(self):
        """list_templates() should return info for all 4 templates."""
        templates = list_templates()
        assert len(templates) == 4

    def test_list_templates_contains_required_fields(self):
        """Each template info should contain required fields."""
        templates = list_templates()
        required_fields = ["name", "display_name", "description", "model", "complexity", "token_usage"]

        for template_info in templates:
            for field in required_fields:
                assert field in template_info

    def test_list_templates_has_correct_names(self):
        """Template list should include all template names."""
        templates = list_templates()
        names = {t["name"] for t in templates}
        assert names == {"researcher", "coder", "analyst", "terminal"}


class TestGetTemplateInfo:
    """Test suite for get_template_info() function."""

    def test_get_researcher_template_info(self):
        """Get detailed info for researcher template."""
        info = get_template_info("researcher")
        assert info is not None
        assert info["name"] == "researcher"
        assert info["instruction_steps"] == 3
        assert info["history_enabled"] is True
        assert info["max_tokens"] == 8000
        assert "web_search" in info["tools"]

    def test_get_terminal_template_info(self):
        """Get detailed info for terminal template."""
        info = get_template_info("terminal")
        assert info is not None
        assert info["name"] == "terminal"
        assert info["instruction_steps"] == 1
        assert info["history_enabled"] is False
        assert info["max_tokens"] == 0

    def test_get_invalid_template_info_returns_none(self):
        """Getting info for non-existent template should return None."""
        info = get_template_info("invalid")
        assert info is None


class TestValidateTemplateName:
    """Test suite for validate_template_name() function."""

    def test_validate_existing_template_names(self):
        """Validating existing template names should return True."""
        assert validate_template_name("researcher") is True
        assert validate_template_name("coder") is True
        assert validate_template_name("analyst") is True
        assert validate_template_name("terminal") is True

    def test_validate_invalid_template_name(self):
        """Validating non-existent template should return False."""
        assert validate_template_name("invalid") is False
        assert validate_template_name("") is False
        assert validate_template_name("data_scientist") is False
