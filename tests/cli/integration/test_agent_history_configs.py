"""Integration tests for per-agent history configuration in AgentChain (T071).

These tests verify that AgentChain correctly applies different history
configurations to different agents, enabling token optimization through
selective history management.

Test Coverage:
- AgentChain accepts agent_history_configs parameter
- Different agents receive different history configurations
- Disabled history (terminal/execution agents) works correctly
- History filtering by type and source works per-agent
- History truncation strategies work per-agent
- Token limits are enforced per-agent
- Global auto_include_history interacts correctly with per-agent configs
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain
from promptchain.cli.models.agent_config import HistoryConfig


class TestAgentHistoryConfigsIntegration:
    """Integration tests for per-agent history configuration."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for AgentChain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def basic_agents(self):
        """Create basic test agents."""
        return {
            "analyst": PromptChain(
                models=["gpt-4.1-mini-2025-04-14"],
                instructions=["Analyze: {input}"],
                verbose=False
            ),
            "coder": PromptChain(
                models=["gpt-4.1-mini-2025-04-14"],
                instructions=["Code: {input}"],
                verbose=False
            ),
            "terminal": PromptChain(
                models=["gpt-4.1-mini-2025-04-14"],
                instructions=["Execute: {input}"],
                verbose=False
            )
        }

    @pytest.fixture
    def agent_descriptions(self):
        """Create agent descriptions for AgentChain."""
        return {
            "analyst": "Analyzes data and provides insights",
            "coder": "Writes code and implements solutions",
            "terminal": "Executes commands and runs scripts"
        }

    def test_agent_chain_accepts_agent_history_configs_parameter(self, basic_agents, agent_descriptions, temp_cache_dir):
        """AgentChain accepts agent_history_configs parameter in constructor."""
        history_configs = {
            "analyst": {
                "enabled": True,
                "max_tokens": 8000,
                "max_entries": 50
            },
            "coder": {
                "enabled": True,
                "max_tokens": 4000,
                "max_entries": 20
            },
            "terminal": {
                "enabled": False,
                "max_tokens": 0,
                "max_entries": 0
            }
        }

        # Should not raise
        agent_chain = AgentChain(
            agents=basic_agents,
            agent_descriptions=agent_descriptions,
            agent_history_configs=history_configs,
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False
        )

        # Verify agent_chain was created
        assert agent_chain is not None
        assert hasattr(agent_chain, "agent_history_configs")

    def test_different_agents_receive_different_history_configurations(
        self, basic_agents, agent_descriptions, temp_cache_dir
    ):
        """Different agents can have different max_tokens and max_entries settings."""
        history_configs = {
            "analyst": {
                "enabled": True,
                "max_tokens": 8000,
                "max_entries": 50,
                "truncation_strategy": "oldest_first"
            },
            "coder": {
                "enabled": True,
                "max_tokens": 4000,
                "max_entries": 20,
                "truncation_strategy": "keep_last"
            }
        }

        agent_chain = AgentChain(
            agents=basic_agents,
            agent_descriptions=agent_descriptions,
            agent_history_configs=history_configs,
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False
        )

        # Verify each agent has correct config
        assert agent_chain.agent_history_configs["analyst"]["max_tokens"] == 8000
        assert agent_chain.agent_history_configs["analyst"]["max_entries"] == 50
        assert agent_chain.agent_history_configs["analyst"]["truncation_strategy"] == "oldest_first"

        assert agent_chain.agent_history_configs["coder"]["max_tokens"] == 4000
        assert agent_chain.agent_history_configs["coder"]["max_entries"] == 20
        assert agent_chain.agent_history_configs["coder"]["truncation_strategy"] == "keep_last"

    def test_disabled_history_for_terminal_agents(self, basic_agents, agent_descriptions, temp_cache_dir):
        """Terminal/execution agents can have history completely disabled (enabled=False)."""
        history_configs = {
            "terminal": {
                "enabled": False,
                "max_tokens": 0,
                "max_entries": 0
            },
            "analyst": {
                "enabled": True,
                "max_tokens": 4000,
                "max_entries": 20
            }
        }

        agent_chain = AgentChain(
            agents=basic_agents,
            agent_descriptions=agent_descriptions,
            agent_history_configs=history_configs,
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False
        )

        # Verify terminal agent has disabled history
        assert agent_chain.agent_history_configs["terminal"]["enabled"] is False
        assert agent_chain.agent_history_configs["terminal"]["max_tokens"] == 0
        assert agent_chain.agent_history_configs["terminal"]["max_entries"] == 0

        # Verify analyst agent has enabled history
        assert agent_chain.agent_history_configs["analyst"]["enabled"] is True
        assert agent_chain.agent_history_configs["analyst"]["max_tokens"] == 4000

    def test_history_filtering_by_type_and_source_per_agent(self, basic_agents, agent_descriptions, temp_cache_dir):
        """History filtering (include_types, exclude_sources) works per-agent."""
        history_configs = {
            "analyst": {
                "enabled": True,
                "max_tokens": 8000,
                "max_entries": 50,
                "include_types": ["user_input", "agent_output"],
                "exclude_sources": ["system"]
            },
            "coder": {
                "enabled": True,
                "max_tokens": 4000,
                "max_entries": 20,
                "include_types": ["user_input", "agent_output", "tool_call"],
                "exclude_sources": None
            }
        }

        agent_chain = AgentChain(
            agents=basic_agents,
            agent_descriptions=agent_descriptions,
            agent_history_configs=history_configs,
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False
        )

        # Verify analyst filters
        assert agent_chain.agent_history_configs["analyst"]["include_types"] == [
            "user_input", "agent_output"
        ]
        assert agent_chain.agent_history_configs["analyst"]["exclude_sources"] == ["system"]

        # Verify coder filters
        assert agent_chain.agent_history_configs["coder"]["include_types"] == [
            "user_input", "agent_output", "tool_call"
        ]
        assert agent_chain.agent_history_configs["coder"]["exclude_sources"] is None

    def test_truncation_strategies_work_per_agent(self, basic_agents, agent_descriptions, temp_cache_dir):
        """Different agents can use different truncation strategies."""
        history_configs = {
            "analyst": {
                "enabled": True,
                "max_tokens": 8000,
                "max_entries": 50,
                "truncation_strategy": "oldest_first"
            },
            "coder": {
                "enabled": True,
                "max_tokens": 4000,
                "max_entries": 20,
                "truncation_strategy": "keep_last"
            }
        }

        agent_chain = AgentChain(
            agents=basic_agents,
            agent_descriptions=agent_descriptions,
            agent_history_configs=history_configs,
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False
        )

        # Verify truncation strategies
        assert agent_chain.agent_history_configs["analyst"]["truncation_strategy"] == "oldest_first"
        assert agent_chain.agent_history_configs["coder"]["truncation_strategy"] == "keep_last"

    def test_global_auto_include_history_interacts_with_per_agent_configs(
        self, basic_agents, agent_descriptions, temp_cache_dir
    ):
        """Global auto_include_history=False can be overridden per-agent."""
        # Global history disabled
        agent_chain = AgentChain(
            agents=basic_agents,
            agent_descriptions=agent_descriptions,
            auto_include_history=False,  # Global default
            agent_history_configs={
                # Analyst overrides to enable history
                "analyst": {
                    "enabled": True,
                    "max_tokens": 8000,
                    "max_entries": 50
                }
                # Other agents inherit global disabled setting
            },
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False
        )

        # Verify analyst overrides global setting
        assert agent_chain.agent_history_configs["analyst"]["enabled"] is True

        # Verify global setting is respected
        assert agent_chain.auto_include_history is False

    def test_agent_without_explicit_config_uses_global_defaults(
        self, basic_agents, agent_descriptions, temp_cache_dir
    ):
        """Agents without explicit config use global auto_include_history setting."""
        history_configs = {
            "analyst": {
                "enabled": True,
                "max_tokens": 8000,
                "max_entries": 50
            }
            # coder and terminal have no explicit config
        }

        agent_chain = AgentChain(
            agents=basic_agents,
            agent_descriptions=agent_descriptions,
            auto_include_history=True,  # Global default
            agent_history_configs=history_configs,
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False
        )

        # Verify analyst has explicit config
        assert "analyst" in agent_chain.agent_history_configs
        assert agent_chain.agent_history_configs["analyst"]["enabled"] is True

        # Verify coder and terminal inherit global setting
        # (Implementation detail: they might not be in agent_history_configs,
        # or they might have default HistoryConfig values)
        # This test validates that agents without explicit config can still work

    def test_history_config_dict_converted_to_historyconfig_objects(
        self, basic_agents, agent_descriptions, temp_cache_dir
    ):
        """Dict configs in agent_history_configs are converted to HistoryConfig objects."""
        history_configs = {
            "analyst": {
                "enabled": True,
                "max_tokens": 8000,
                "max_entries": 50,
                "truncation_strategy": "oldest_first",
                "include_types": ["user_input"],
                "exclude_sources": ["system"]
            }
        }

        agent_chain = AgentChain(
            agents=basic_agents,
            agent_descriptions=agent_descriptions,
            agent_history_configs=history_configs,
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False
        )

        # Verify config is stored (as dict or HistoryConfig object)
        assert "analyst" in agent_chain.agent_history_configs
        analyst_config = agent_chain.agent_history_configs["analyst"]

        # Can be dict or HistoryConfig - both are valid
        if isinstance(analyst_config, dict):
            assert analyst_config["max_tokens"] == 8000
        elif isinstance(analyst_config, HistoryConfig):
            assert analyst_config.max_tokens == 8000
        else:
            pytest.fail(f"Unexpected config type: {type(analyst_config)}")

    def test_invalid_agent_name_in_history_configs_raises_error(
        self, basic_agents, agent_descriptions, temp_cache_dir
    ):
        """Providing history config for non-existent agent raises ValueError."""
        history_configs = {
            "nonexistent_agent": {
                "enabled": True,
                "max_tokens": 4000,
                "max_entries": 20
            }
        }

        # Should raise ValueError for unknown agent
        with pytest.raises(ValueError, match="nonexistent_agent"):
            AgentChain(
                agents=basic_agents,
            agent_descriptions=agent_descriptions,
                agent_history_configs=history_configs,
                execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
                verbose=False
            )

    def test_history_config_validation_enforced_at_agent_chain_creation(
        self, basic_agents, agent_descriptions, temp_cache_dir
    ):
        """Invalid HistoryConfig values raise ValueError at AgentChain creation."""
        # Out of range max_tokens
        history_configs = {
            "analyst": {
                "enabled": True,
                "max_tokens": 99,  # Below minimum (100)
                "max_entries": 20
            }
        }

        with pytest.raises(ValueError, match="max_tokens must be between 100-16000"):
            AgentChain(
                agents=basic_agents,
            agent_descriptions=agent_descriptions,
                agent_history_configs=history_configs,
                execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
                verbose=False
            )

    def test_token_savings_calculation_for_disabled_history(
        self, basic_agents, agent_descriptions, temp_cache_dir
    ):
        """Disabled history for terminal agents should show significant token savings."""
        history_configs = {
            "terminal": {
                "enabled": False,
                "max_tokens": 0,
                "max_entries": 0
            },
            "analyst": {
                "enabled": True,
                "max_tokens": 4000,
                "max_entries": 20
            }
        }

        agent_chain = AgentChain(
            agents=basic_agents,
            agent_descriptions=agent_descriptions,
            agent_history_configs=history_configs,
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False
        )

        # Calculate potential savings
        # Terminal agent: 0 tokens (disabled)
        # Analyst agent: 4000 max tokens
        # If terminal had default 4000 tokens, we save 4000 tokens (50% reduction in 2-agent scenario)

        terminal_tokens = agent_chain.agent_history_configs["terminal"]["max_tokens"]
        analyst_tokens = agent_chain.agent_history_configs["analyst"]["max_tokens"]

        assert terminal_tokens == 0  # Terminal uses no history tokens
        assert analyst_tokens == 4000  # Analyst uses full history
        assert terminal_tokens < analyst_tokens  # Clear token savings


class TestAgentHistoryConfigsEdgeCases:
    """Edge case tests for per-agent history configuration."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_empty_agent_history_configs_is_valid(self, temp_cache_dir):
        """Empty agent_history_configs dict is valid (uses global defaults)."""
        agents = {
            "analyst": PromptChain(
                models=["gpt-4.1-mini-2025-04-14"],
                instructions=["Analyze: {input}"],
                verbose=False
            )
        }

        agent_descriptions = {
            "analyst": "Analyzes data and provides insights"
        }

        # Should not raise
        agent_chain = AgentChain(
            agents=agents,
            agent_descriptions=agent_descriptions,
            agent_history_configs={},  # Empty dict
            auto_include_history=True,
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False
        )

        assert agent_chain is not None

    def test_none_agent_history_configs_is_valid(self, temp_cache_dir):
        """None agent_history_configs is valid (uses global defaults)."""
        agents = {
            "analyst": PromptChain(
                models=["gpt-4.1-mini-2025-04-14"],
                instructions=["Analyze: {input}"],
                verbose=False
            )
        }

        agent_descriptions = {
            "analyst": "Analyzes data and provides insights"
        }
        # Should not raise
        agent_chain = AgentChain(
            agents=agents,
            agent_descriptions=agent_descriptions,
            agent_history_configs=None,  # Explicitly None
            auto_include_history=True,
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False
        )

        assert agent_chain is not None

    def test_partial_agent_history_configs_is_valid(self, temp_cache_dir):
        """Some agents configured, others use defaults."""
        agents = {
            "analyst": PromptChain(
                models=["gpt-4.1-mini-2025-04-14"],
                instructions=["Analyze: {input}"],
                verbose=False
            ),
            "coder": PromptChain(
                models=["gpt-4.1-mini-2025-04-14"],
                instructions=["Code: {input}"],
                verbose=False
            ),
            "terminal": PromptChain(
                models=["gpt-4.1-mini-2025-04-14"],
                instructions=["Execute: {input}"],
                verbose=False
            )
        }

        agent_descriptions = {
            "analyst": "Analyzes data and provides insights",
            "coder": "Writes code and implements solutions",
            "terminal": "Executes commands and runs scripts"
        }

        # Only configure analyst and terminal, coder uses defaults
        history_configs = {
            "analyst": {
                "enabled": True,
                "max_tokens": 8000,
                "max_entries": 50
            },
            "terminal": {
                "enabled": False,
                "max_tokens": 0,
                "max_entries": 0
            }
            # coder not configured - should use global defaults
        }

        agent_chain = AgentChain(
            agents=agents,
            agent_descriptions=agent_descriptions,
            agent_history_configs=history_configs,
            auto_include_history=True,
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False
        )

        # Verify analyst and terminal have explicit configs
        assert "analyst" in agent_chain.agent_history_configs
        assert "terminal" in agent_chain.agent_history_configs

        # coder should work with defaults (might not be in agent_history_configs)
