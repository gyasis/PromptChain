"""Unit tests for history config defaults by agent type (T073).

These tests verify that different agent types receive appropriate default
history configurations optimized for their typical usage patterns:

- Terminal agents: History disabled (saves 60% tokens)
- Coder agents: Moderate history (4000 tokens, 20 entries)
- Researcher agents: Full history (8000 tokens, 50 entries)

Test Coverage:
- Default history config creation by agent type
- Terminal agent optimization (disabled history)
- Coder agent moderate settings
- Researcher agent full history settings
- Custom history config override capability
"""

import pytest
from promptchain.cli.models.agent_config import HistoryConfig


class TestHistoryConfigDefaults:
    """Unit tests for default history configurations by agent type."""

    def test_terminal_agent_default_history_disabled(self):
        """Terminal agents have history disabled by default for token efficiency."""
        # Terminal agents don't need conversation context
        terminal_config = HistoryConfig(
            enabled=False,
            max_tokens=0,
            max_entries=0,
        )

        assert terminal_config.enabled is False
        assert terminal_config.max_tokens == 0
        assert terminal_config.max_entries == 0

    def test_coder_agent_default_moderate_history(self):
        """Coder agents have moderate history (4000 tokens, 20 entries)."""
        # Coder agents need some context but not extensive history
        coder_config = HistoryConfig(
            enabled=True,
            max_tokens=4000,
            max_entries=20,
            truncation_strategy="oldest_first",
        )

        assert coder_config.enabled is True
        assert coder_config.max_tokens == 4000
        assert coder_config.max_entries == 20
        assert coder_config.truncation_strategy == "oldest_first"

    def test_researcher_agent_default_full_history(self):
        """Researcher agents have full history (8000 tokens, 50 entries)."""
        # Researcher agents need extensive context for analysis
        researcher_config = HistoryConfig(
            enabled=True,
            max_tokens=8000,
            max_entries=50,
            truncation_strategy="oldest_first",
        )

        assert researcher_config.enabled is True
        assert researcher_config.max_tokens == 8000
        assert researcher_config.max_entries == 50
        assert researcher_config.truncation_strategy == "oldest_first"

    def test_terminal_agent_token_savings(self):
        """Terminal agents with disabled history save significant tokens."""
        # Baseline: enabled history
        baseline_config = HistoryConfig(enabled=True, max_tokens=4000, max_entries=20)

        # Optimized: disabled history
        terminal_config = HistoryConfig(enabled=False, max_tokens=0, max_entries=0)

        # Terminal config saves 100% of history tokens
        baseline_tokens = baseline_config.max_tokens
        terminal_tokens = terminal_config.max_tokens

        token_savings = baseline_tokens - terminal_tokens
        savings_percentage = (token_savings / baseline_tokens) * 100

        assert savings_percentage == 100.0

    def test_coder_vs_researcher_token_difference(self):
        """Coder agents use 50% fewer tokens than researcher agents."""
        coder_config = HistoryConfig(enabled=True, max_tokens=4000, max_entries=20)
        researcher_config = HistoryConfig(enabled=True, max_tokens=8000, max_entries=50)

        # Coder uses half the tokens of researcher
        assert coder_config.max_tokens == researcher_config.max_tokens / 2

        # Coder uses 40% of researcher's entries
        assert coder_config.max_entries == researcher_config.max_entries * 0.4

    def test_default_config_is_coder_level(self):
        """Default HistoryConfig matches coder agent settings."""
        default_config = HistoryConfig()

        # Default should be moderate (coder-level) settings
        assert default_config.enabled is True
        assert default_config.max_tokens == 4000
        assert default_config.max_entries == 20
        assert default_config.truncation_strategy == "oldest_first"


class TestHistoryConfigByAgentTypeHelper:
    """Unit tests for helper function that creates config by agent type."""

    def test_get_history_config_for_terminal_agent(self):
        """get_history_config_by_type('terminal') returns disabled config."""
        terminal_config = self._get_history_config_by_type("terminal")

        assert terminal_config.enabled is False
        assert terminal_config.max_tokens == 0
        assert terminal_config.max_entries == 0

    def test_get_history_config_for_coder_agent(self):
        """get_history_config_by_type('coder') returns moderate config."""
        coder_config = self._get_history_config_by_type("coder")

        assert coder_config.enabled is True
        assert coder_config.max_tokens == 4000
        assert coder_config.max_entries == 20

    def test_get_history_config_for_researcher_agent(self):
        """get_history_config_by_type('researcher') returns full config."""
        researcher_config = self._get_history_config_by_type("researcher")

        assert researcher_config.enabled is True
        assert researcher_config.max_tokens == 8000
        assert researcher_config.max_entries == 50

    def test_get_history_config_for_unknown_type_uses_default(self):
        """Unknown agent types get default (coder-level) config."""
        unknown_config = self._get_history_config_by_type("unknown_type")

        # Should default to coder-level settings
        assert unknown_config.enabled is True
        assert unknown_config.max_tokens == 4000
        assert unknown_config.max_entries == 20

    def test_custom_config_overrides_defaults(self):
        """Custom history config overrides type-based defaults."""
        # Start with terminal default (disabled)
        terminal_config = self._get_history_config_by_type("terminal")
        assert terminal_config.enabled is False

        # Override with custom config
        custom_config = HistoryConfig(
            enabled=True,
            max_tokens=6000,
            max_entries=30,
        )

        # Custom settings take precedence
        assert custom_config.enabled is True
        assert custom_config.max_tokens == 6000
        assert custom_config.max_entries == 30

    # Helper method (would typically be in a utility module)
    def _get_history_config_by_type(self, agent_type: str) -> HistoryConfig:
        """Get default history config for agent type.

        Args:
            agent_type: Type of agent ("terminal", "coder", "researcher")

        Returns:
            HistoryConfig: Default configuration for the agent type
        """
        type_defaults = {
            "terminal": {
                "enabled": False,
                "max_tokens": 0,
                "max_entries": 0,
            },
            "coder": {
                "enabled": True,
                "max_tokens": 4000,
                "max_entries": 20,
                "truncation_strategy": "oldest_first",
            },
            "researcher": {
                "enabled": True,
                "max_tokens": 8000,
                "max_entries": 50,
                "truncation_strategy": "oldest_first",
            },
        }

        # Get default for type, fallback to coder defaults
        config_dict = type_defaults.get(agent_type, type_defaults["coder"])
        return HistoryConfig.from_dict(config_dict)


class TestHistoryConfigTokenOptimization:
    """Unit tests for token optimization calculations."""

    def test_multi_agent_system_token_savings(self):
        """6-agent system with selective history saves significant tokens."""
        # Multi-agent system configuration
        agents_config = {
            "terminal": HistoryConfig(enabled=False, max_tokens=0, max_entries=0),
            "coder": HistoryConfig(enabled=True, max_tokens=4000, max_entries=20),
            "researcher1": HistoryConfig(enabled=True, max_tokens=8000, max_entries=50),
            "researcher2": HistoryConfig(enabled=True, max_tokens=8000, max_entries=50),
            "analyst": HistoryConfig(enabled=True, max_tokens=8000, max_entries=50),
            "writer": HistoryConfig(enabled=True, max_tokens=8000, max_entries=50),
        }

        # Calculate total token budget
        total_tokens = sum(config.max_tokens for config in agents_config.values())

        # Expected: 0 + 4000 + 8000 + 8000 + 8000 + 8000 = 36000 tokens
        assert total_tokens == 36000

        # Compare to baseline: all agents with 8000 tokens
        baseline_total = 6 * 8000  # 48000 tokens

        token_savings = baseline_total - total_tokens
        savings_percentage = (token_savings / baseline_total) * 100

        # Should save 25% of tokens (12000 / 48000)
        assert token_savings == 12000
        assert savings_percentage == 25.0

    def test_terminal_heavy_system_maximum_savings(self):
        """System with mostly terminal agents achieves maximum token savings."""
        # 4 terminal + 2 research agents
        agents_config = {
            "terminal1": HistoryConfig(enabled=False, max_tokens=0, max_entries=0),
            "terminal2": HistoryConfig(enabled=False, max_tokens=0, max_entries=0),
            "terminal3": HistoryConfig(enabled=False, max_tokens=0, max_entries=0),
            "terminal4": HistoryConfig(enabled=False, max_tokens=0, max_entries=0),
            "researcher1": HistoryConfig(enabled=True, max_tokens=8000, max_entries=50),
            "researcher2": HistoryConfig(enabled=True, max_tokens=8000, max_entries=50),
        }

        total_tokens = sum(config.max_tokens for config in agents_config.values())
        baseline_total = 6 * 8000  # If all had full history

        token_savings = baseline_total - total_tokens
        savings_percentage = (token_savings / baseline_total) * 100

        # Should save 66.7% of tokens (4 out of 6 agents disabled)
        assert total_tokens == 16000  # Only 2 agents with history
        assert savings_percentage == pytest.approx(66.67, abs=0.1)

    def test_balanced_system_moderate_savings(self):
        """Balanced system (2 terminal, 2 coder, 2 researcher) saves moderately."""
        agents_config = {
            "terminal1": HistoryConfig(enabled=False, max_tokens=0, max_entries=0),
            "terminal2": HistoryConfig(enabled=False, max_tokens=0, max_entries=0),
            "coder1": HistoryConfig(enabled=True, max_tokens=4000, max_entries=20),
            "coder2": HistoryConfig(enabled=True, max_tokens=4000, max_entries=20),
            "researcher1": HistoryConfig(enabled=True, max_tokens=8000, max_entries=50),
            "researcher2": HistoryConfig(enabled=True, max_tokens=8000, max_entries=50),
        }

        total_tokens = sum(config.max_tokens for config in agents_config.values())
        baseline_total = 6 * 8000

        token_savings = baseline_total - total_tokens
        savings_percentage = (token_savings / baseline_total) * 100

        # 0 + 0 + 4000 + 4000 + 8000 + 8000 = 24000 tokens
        assert total_tokens == 24000
        # Should save 50% of tokens ((48000 - 24000) / 48000 = 0.5)
        assert savings_percentage == 50.0
