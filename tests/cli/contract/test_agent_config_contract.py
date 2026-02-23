"""Contract test for Agent schema validation (T032).

This test verifies that the Agent dataclass includes all required fields for
User Story 1 (Intelligent Multi-Agent Conversations) and adheres to the schema
defined in specs/002-cli-orchestration/data-model.md.

Test Strategy:
- Validate required fields exist with correct types
- Test schema validation rules (name format, model format, description length)
- Verify serialization/deserialization preserves all orchestration fields
- Ensure backward compatibility with v1 schema (basic agent creation)

RED Phase: Test should FAIL if orchestration fields are missing or invalid
GREEN Phase: Test should PASS after implementation ensures schema correctness
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from promptchain.cli.models.agent_config import Agent, HistoryConfig


class TestAgentSchemaContract:
    """Contract tests for Agent dataclass schema (T032)."""

    def test_agent_has_required_core_fields(self):
        """Verify Agent has all core identity fields (v1 schema)."""
        # Create minimal agent
        agent = Agent(
            name="test-agent",
            model_name="gpt-4.1-mini-2025-04-14"
        )

        # Core v1 fields
        assert hasattr(agent, "name")
        assert hasattr(agent, "model_name")
        assert hasattr(agent, "description")
        assert hasattr(agent, "created_at")
        assert hasattr(agent, "last_used")
        assert hasattr(agent, "usage_count")
        assert hasattr(agent, "metadata")

        # Verify types
        assert isinstance(agent.name, str)
        assert isinstance(agent.model_name, str)
        assert isinstance(agent.description, str)
        assert isinstance(agent.created_at, float)
        assert agent.last_used is None or isinstance(agent.last_used, float)
        assert isinstance(agent.usage_count, int)
        assert isinstance(agent.metadata, dict)

    def test_agent_has_required_orchestration_fields(self):
        """Verify Agent has all orchestration fields (v2 schema) for US1."""
        # Create agent with orchestration fields
        agent = Agent(
            name="research-agent",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Research specialist for multi-agent routing",
            instruction_chain=["Analyze: {input}", "Summarize: {input}"],
            tools=["web_search", "file_reader"],
            history_config=HistoryConfig(max_tokens=8000, max_entries=50)
        )

        # NEW: Orchestration fields (v2 schema)
        assert hasattr(agent, "description")
        assert hasattr(agent, "instruction_chain")
        assert hasattr(agent, "tools")
        assert hasattr(agent, "history_config")

        # Verify types
        assert isinstance(agent.description, str)
        assert isinstance(agent.instruction_chain, list)
        assert isinstance(agent.tools, list)
        assert isinstance(agent.history_config, HistoryConfig) or agent.history_config is None

    def test_description_field_used_by_router(self):
        """Verify description field exists and has correct constraints for router."""
        # Description is critical for router agent selection in US1
        agent = Agent(
            name="coder",
            model_name="anthropic/claude-3-sonnet",
            description="Specialized in code generation and debugging"
        )

        assert agent.description == "Specialized in code generation and debugging"
        assert len(agent.description) <= 256  # Per spec: max 256 characters

    def test_description_field_validation_max_length(self):
        """Verify description rejects strings exceeding 256 characters."""
        long_description = "x" * 257

        with pytest.raises(ValueError, match="Description must be ≤256 characters"):
            Agent(
                name="test",
                model_name="gpt-4.1-mini-2025-04-14",
                description=long_description
            )

    def test_instruction_chain_field_type(self):
        """Verify instruction_chain accepts strings and dict configurations."""
        # String instructions (simple prompts)
        agent1 = Agent(
            name="simple",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=["Analyze: {input}"]
        )
        assert agent1.instruction_chain == ["Analyze: {input}"]

        # Mixed instructions (strings + function refs + AgenticStep configs)
        agent2 = Agent(
            name="complex",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[
                "First step: {input}",
                {"type": "function", "name": "validate_data"},
                {"type": "agentic_step", "objective": "Research topic"}
            ]
        )
        assert len(agent2.instruction_chain) == 3
        assert isinstance(agent2.instruction_chain[1], dict)
        assert isinstance(agent2.instruction_chain[2], dict)

    def test_tools_field_type(self):
        """Verify tools field accepts list of tool names."""
        agent = Agent(
            name="tooluser",
            model_name="gpt-4.1-mini-2025-04-14",
            tools=["mcp_filesystem_read", "mcp_web_search", "local_calculator"]
        )

        assert isinstance(agent.tools, list)
        assert all(isinstance(tool, str) for tool in agent.tools)
        assert "mcp_filesystem_read" in agent.tools

    def test_history_config_field_integration(self):
        """Verify history_config integrates properly with HistoryConfig dataclass."""
        # Create agent with custom history config
        history_cfg = HistoryConfig(
            enabled=True,
            max_tokens=6000,
            max_entries=30,
            truncation_strategy="keep_last",
            include_types=["user_input", "agent_output"],
            exclude_sources=["debug"]
        )

        agent = Agent(
            name="researcher",
            model_name="gpt-4.1-mini-2025-04-14",
            history_config=history_cfg
        )

        assert agent.history_config == history_cfg
        assert agent.history_config.max_tokens == 6000
        assert agent.history_config.truncation_strategy == "keep_last"

    def test_history_config_optional_none(self):
        """Verify history_config can be None (uses AgentChain defaults)."""
        agent = Agent(
            name="default",
            model_name="gpt-4.1-mini-2025-04-14",
            history_config=None
        )

        assert agent.history_config is None

    def test_is_terminal_agent_property(self):
        """Verify is_terminal_agent property correctly identifies disabled history."""
        # Terminal agent (history disabled for token efficiency)
        terminal_cfg = HistoryConfig(enabled=False, max_tokens=0, max_entries=0)
        terminal_agent = Agent(
            name="terminal",
            model_name="openai/gpt-3.5-turbo",
            history_config=terminal_cfg
        )

        assert terminal_agent.is_terminal_agent is True

        # Regular agent (history enabled)
        regular_agent = Agent(
            name="coder",
            model_name="gpt-4.1-mini-2025-04-14",
            history_config=HistoryConfig(enabled=True)
        )

        assert regular_agent.is_terminal_agent is False

    def test_agent_serialization_preserves_orchestration_fields(self):
        """Verify to_dict() includes all orchestration fields for storage."""
        agent = Agent(
            name="full-agent",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Complete agent with all fields",
            instruction_chain=["Step 1: {input}", "Step 2: {input}"],
            tools=["tool1", "tool2"],
            history_config=HistoryConfig(max_tokens=5000, max_entries=25)
        )

        agent_dict = agent.to_dict()

        # Verify all orchestration fields present
        assert "description" in agent_dict
        assert "instruction_chain" in agent_dict
        assert "tools" in agent_dict
        assert "history_config" in agent_dict

        # Verify values preserved
        assert agent_dict["description"] == "Complete agent with all fields"
        assert agent_dict["instruction_chain"] == ["Step 1: {input}", "Step 2: {input}"]
        assert agent_dict["tools"] == ["tool1", "tool2"]
        assert agent_dict["history_config"]["max_tokens"] == 5000

    def test_agent_deserialization_reconstructs_orchestration_fields(self):
        """Verify from_dict() correctly reconstructs Agent with orchestration fields."""
        agent_data: Dict[str, Any] = {
            "name": "restored-agent",
            "model_name": "anthropic/claude-3-opus",
            "description": "Restored from storage",
            "created_at": datetime.now().timestamp(),
            "last_used": None,
            "usage_count": 0,
            "metadata": {},
            "instruction_chain": ["Instruction 1", "Instruction 2"],
            "tools": ["web_search", "calculator"],
            "history_config": {
                "enabled": True,
                "max_tokens": 7000,
                "max_entries": 40,
                "truncation_strategy": "oldest_first",
                "include_types": None,
                "exclude_sources": None
            }
        }

        agent = Agent.from_dict(agent_data)

        # Verify orchestration fields reconstructed
        assert agent.name == "restored-agent"
        assert agent.description == "Restored from storage"
        assert agent.instruction_chain == ["Instruction 1", "Instruction 2"]
        assert agent.tools == ["web_search", "calculator"]
        assert agent.history_config is not None
        assert agent.history_config.max_tokens == 7000
        assert agent.history_config.max_entries == 40

    def test_agent_name_validation(self):
        """Verify agent name follows validation rules from spec."""
        # Valid names
        valid_names = ["coder", "research-agent", "agent_1", "analyst-2"]
        for name in valid_names:
            agent = Agent(name=name, model_name="gpt-4.1-mini-2025-04-14")
            assert agent.name == name

        # Invalid: too long (>32 chars)
        with pytest.raises(ValueError, match="must be 1-32 characters"):
            Agent(name="a" * 33, model_name="gpt-4.1-mini-2025-04-14")

        # Invalid: empty
        with pytest.raises(ValueError, match="must be 1-32 characters"):
            Agent(name="", model_name="gpt-4.1-mini-2025-04-14")

        # Invalid: special characters (not alphanumeric+dash/underscore)
        with pytest.raises(ValueError, match="must be alphanumeric"):
            Agent(name="agent@name", model_name="gpt-4.1-mini-2025-04-14")

    def test_agent_model_name_validation(self):
        """Verify model_name follows LiteLLM format validation."""
        # Valid model names (provider/model format)
        valid_models = [
            "gpt-4.1-mini-2025-04-14",
            "anthropic/claude-3-sonnet-20240229",
            "ollama/llama2",
            "google/gemini-pro"
        ]
        for model in valid_models:
            agent = Agent(name="test", model_name=model)
            assert agent.model_name == model

        # Invalid: missing provider prefix
        with pytest.raises(ValueError, match="must be in format 'provider/model-name'"):
            Agent(name="test", model_name="gpt-4.1-mini-2025-04-14")

        # Invalid: empty
        with pytest.raises(ValueError, match="must be in format 'provider/model-name'"):
            Agent(name="test", model_name="")

    def test_backward_compatibility_v1_agents(self):
        """Verify agents created without orchestration fields still work (v1 compat)."""
        # V1-style agent (only core fields, no orchestration)
        v1_agent = Agent(
            name="legacy",
            model_name="openai/gpt-3.5-turbo"
        )

        # Orchestration fields should exist with defaults
        assert v1_agent.description == ""
        assert v1_agent.instruction_chain == []
        assert v1_agent.tools == []
        assert v1_agent.history_config is None

        # Should serialize/deserialize without errors
        v1_dict = v1_agent.to_dict()
        restored = Agent.from_dict(v1_dict)
        assert restored.name == "legacy"
        assert restored.instruction_chain == []

    def test_default_values_for_optional_fields(self):
        """Verify optional fields have sensible defaults."""
        agent = Agent(
            name="minimal",
            model_name="gpt-4.1-mini-2025-04-14"
        )

        # Defaults from dataclass field() declarations
        assert agent.description == ""
        assert agent.instruction_chain == []
        assert agent.tools == []
        assert agent.history_config is None
        assert agent.usage_count == 0
        assert agent.last_used is None
        assert agent.metadata == {}

    def test_roundtrip_serialization_preserves_all_data(self):
        """Verify complete roundtrip serialization/deserialization."""
        original = Agent(
            name="roundtrip-test",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Testing serialization roundtrip",
            instruction_chain=[
                "Step 1: {input}",
                {"type": "function", "name": "validate"},
                "Step 3: {input}"
            ],
            tools=["tool_a", "tool_b", "tool_c"],
            history_config=HistoryConfig(
                enabled=True,
                max_tokens=9000,
                max_entries=60,
                truncation_strategy="keep_last",
                include_types=["user_input", "agent_output"],
                exclude_sources=["system", "debug"]
            )
        )

        # Serialize to dict
        agent_dict = original.to_dict()

        # Deserialize back to Agent
        restored = Agent.from_dict(agent_dict)

        # Verify all fields match
        assert restored.name == original.name
        assert restored.model_name == original.model_name
        assert restored.description == original.description
        assert restored.instruction_chain == original.instruction_chain
        assert restored.tools == original.tools
        assert restored.history_config.enabled == original.history_config.enabled
        assert restored.history_config.max_tokens == original.history_config.max_tokens
        assert restored.history_config.include_types == original.history_config.include_types
        assert restored.history_config.exclude_sources == original.history_config.exclude_sources


class TestHistoryConfigContract:
    """Contract tests for HistoryConfig dataclass (T032)."""

    def test_history_config_has_required_fields(self):
        """Verify HistoryConfig has all required fields per spec."""
        config = HistoryConfig()

        # Required fields
        assert hasattr(config, "enabled")
        assert hasattr(config, "max_tokens")
        assert hasattr(config, "max_entries")
        assert hasattr(config, "truncation_strategy")
        assert hasattr(config, "include_types")
        assert hasattr(config, "exclude_sources")

        # Verify types
        assert isinstance(config.enabled, bool)
        assert isinstance(config.max_tokens, int)
        assert isinstance(config.max_entries, int)
        assert isinstance(config.truncation_strategy, str)
        assert config.include_types is None or isinstance(config.include_types, list)
        assert config.exclude_sources is None or isinstance(config.exclude_sources, list)

    def test_history_config_default_values(self):
        """Verify HistoryConfig defaults match spec."""
        config = HistoryConfig()

        assert config.enabled is True
        assert config.max_tokens == 4000  # Default for coder-level agents
        assert config.max_entries == 20
        assert config.truncation_strategy == "oldest_first"
        assert config.include_types is None
        assert config.exclude_sources is None

    def test_history_config_max_tokens_validation(self):
        """Verify max_tokens range validation (100-16000)."""
        # Valid range
        valid_values = [100, 4000, 8000, 16000]
        for val in valid_values:
            config = HistoryConfig(enabled=True, max_tokens=val)
            assert config.max_tokens == val

        # Invalid: too low
        with pytest.raises(ValueError, match="max_tokens must be between 100-16000"):
            HistoryConfig(enabled=True, max_tokens=99)

        # Invalid: too high
        with pytest.raises(ValueError, match="max_tokens must be between 100-16000"):
            HistoryConfig(enabled=True, max_tokens=16001)

    def test_history_config_max_entries_validation(self):
        """Verify max_entries range validation (1-200)."""
        # Valid range
        valid_values = [1, 20, 100, 200]
        for val in valid_values:
            config = HistoryConfig(enabled=True, max_entries=val)
            assert config.max_entries == val

        # Invalid: too low
        with pytest.raises(ValueError, match="max_entries must be between 1-200"):
            HistoryConfig(enabled=True, max_entries=0)

        # Invalid: too high
        with pytest.raises(ValueError, match="max_entries must be between 1-200"):
            HistoryConfig(enabled=True, max_entries=201)

    def test_history_config_disabled_requires_zero_limits(self):
        """Verify disabled history requires max_tokens=0 and max_entries=0."""
        # Valid: disabled with zero limits
        config = HistoryConfig(enabled=False, max_tokens=0, max_entries=0)
        assert config.enabled is False

        # Invalid: disabled but non-zero max_tokens
        with pytest.raises(ValueError, match="max_tokens and max_entries must be 0"):
            HistoryConfig(enabled=False, max_tokens=4000, max_entries=0)

        # Invalid: disabled but non-zero max_entries
        with pytest.raises(ValueError, match="max_tokens and max_entries must be 0"):
            HistoryConfig(enabled=False, max_tokens=0, max_entries=20)

    def test_history_config_truncation_strategy_values(self):
        """Verify truncation_strategy accepts valid literal values."""
        # Valid strategies
        valid_strategies = ["oldest_first", "keep_last"]
        for strategy in valid_strategies:
            config = HistoryConfig(truncation_strategy=strategy)
            assert config.truncation_strategy == strategy

    def test_history_config_for_agent_type_factory(self):
        """Verify for_agent_type() factory creates appropriate defaults."""
        # Terminal agent (history disabled)
        terminal_config = HistoryConfig.for_agent_type("terminal")
        assert terminal_config.enabled is False
        assert terminal_config.max_tokens == 0
        assert terminal_config.max_entries == 0

        # Coder agent (moderate history)
        coder_config = HistoryConfig.for_agent_type("coder")
        assert coder_config.enabled is True
        assert coder_config.max_tokens == 4000
        assert coder_config.max_entries == 20
        assert coder_config.truncation_strategy == "oldest_first"

        # Researcher agent (full history)
        researcher_config = HistoryConfig.for_agent_type("researcher")
        assert researcher_config.enabled is True
        assert researcher_config.max_tokens == 8000
        assert researcher_config.max_entries == 50

        # Analyst agent (full history)
        analyst_config = HistoryConfig.for_agent_type("analyst")
        assert analyst_config.enabled is True
        assert analyst_config.max_tokens == 8000
        assert analyst_config.max_entries == 50

    def test_history_config_serialization(self):
        """Verify HistoryConfig serialization for storage."""
        config = HistoryConfig(
            enabled=True,
            max_tokens=6000,
            max_entries=35,
            truncation_strategy="keep_last",
            include_types=["user_input", "agent_output"],
            exclude_sources=["debug"]
        )

        config_dict = config.to_dict()

        assert config_dict["enabled"] is True
        assert config_dict["max_tokens"] == 6000
        assert config_dict["max_entries"] == 35
        assert config_dict["truncation_strategy"] == "keep_last"
        assert config_dict["include_types"] == ["user_input", "agent_output"]
        assert config_dict["exclude_sources"] == ["debug"]

    def test_history_config_deserialization(self):
        """Verify HistoryConfig deserialization from storage."""
        config_data = {
            "enabled": True,
            "max_tokens": 7000,
            "max_entries": 45,
            "truncation_strategy": "oldest_first",
            "include_types": ["user_input"],
            "exclude_sources": ["system"]
        }

        config = HistoryConfig.from_dict(config_data)

        assert config.enabled is True
        assert config.max_tokens == 7000
        assert config.max_entries == 45
        assert config.truncation_strategy == "oldest_first"
        assert config.include_types == ["user_input"]
        assert config.exclude_sources == ["system"]
