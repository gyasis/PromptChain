"""Contract tests for HistoryConfig schema validation (T070).

These tests verify that the HistoryConfig dataclass enforces all validation
rules as specified in the data model schema. Contract tests ensure the public
API and constraints remain stable across refactorings.

Test Coverage:
- Valid configurations with different settings
- Boundary conditions for max_tokens and max_entries
- Validation errors for out-of-range values
- Disabled history mode validation
- Dictionary serialization/deserialization
"""

import pytest
from promptchain.cli.models.agent_config import HistoryConfig


class TestHistoryConfigValidation:
    """Contract tests for HistoryConfig validation rules."""

    def test_default_history_config_is_valid(self):
        """Default HistoryConfig values pass validation."""
        config = HistoryConfig()

        assert config.enabled is True
        assert config.max_tokens == 4000
        assert config.max_entries == 20
        assert config.truncation_strategy == "oldest_first"
        assert config.include_types is None
        assert config.exclude_sources is None

    def test_custom_history_config_within_bounds(self):
        """Custom HistoryConfig within valid bounds passes validation."""
        config = HistoryConfig(
            enabled=True,
            max_tokens=8000,
            max_entries=50,
            truncation_strategy="keep_last",
            include_types=["user_input", "agent_output"],
            exclude_sources=["system"],
        )

        assert config.enabled is True
        assert config.max_tokens == 8000
        assert config.max_entries == 50
        assert config.truncation_strategy == "keep_last"
        assert config.include_types == ["user_input", "agent_output"]
        assert config.exclude_sources == ["system"]

    def test_max_tokens_minimum_boundary_valid(self):
        """max_tokens=100 (minimum valid value) passes validation."""
        config = HistoryConfig(max_tokens=100)

        assert config.max_tokens == 100

    def test_max_tokens_maximum_boundary_valid(self):
        """max_tokens=16000 (maximum valid value) passes validation."""
        config = HistoryConfig(max_tokens=16000)

        assert config.max_tokens == 16000

    def test_max_entries_minimum_boundary_valid(self):
        """max_entries=1 (minimum valid value) passes validation."""
        config = HistoryConfig(max_entries=1)

        assert config.max_entries == 1

    def test_max_entries_maximum_boundary_valid(self):
        """max_entries=200 (maximum valid value) passes validation."""
        config = HistoryConfig(max_entries=200)

        assert config.max_entries == 200

    def test_max_tokens_below_minimum_fails(self):
        """max_tokens < 100 raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be between 100-16000"):
            HistoryConfig(max_tokens=99)

    def test_max_tokens_above_maximum_fails(self):
        """max_tokens > 16000 raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be between 100-16000"):
            HistoryConfig(max_tokens=16001)

    def test_max_entries_below_minimum_fails(self):
        """max_entries < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_entries must be between 1-200"):
            HistoryConfig(max_entries=0)

    def test_max_entries_above_maximum_fails(self):
        """max_entries > 200 raises ValueError."""
        with pytest.raises(ValueError, match="max_entries must be between 1-200"):
            HistoryConfig(max_entries=201)

    def test_disabled_history_requires_zero_limits(self):
        """When enabled=False, max_tokens and max_entries must be 0."""
        config = HistoryConfig(enabled=False, max_tokens=0, max_entries=0)

        assert config.enabled is False
        assert config.max_tokens == 0
        assert config.max_entries == 0

    def test_disabled_history_with_nonzero_max_tokens_fails(self):
        """enabled=False with max_tokens > 0 raises ValueError."""
        with pytest.raises(
            ValueError, match="When enabled=False, max_tokens and max_entries must be 0"
        ):
            HistoryConfig(enabled=False, max_tokens=4000, max_entries=0)

    def test_disabled_history_with_nonzero_max_entries_fails(self):
        """enabled=False with max_entries > 0 raises ValueError."""
        with pytest.raises(
            ValueError, match="When enabled=False, max_tokens and max_entries must be 0"
        ):
            HistoryConfig(enabled=False, max_tokens=0, max_entries=20)

    def test_disabled_history_with_both_nonzero_fails(self):
        """enabled=False with both limits > 0 raises ValueError."""
        with pytest.raises(
            ValueError, match="When enabled=False, max_tokens and max_entries must be 0"
        ):
            HistoryConfig(enabled=False, max_tokens=4000, max_entries=20)


class TestHistoryConfigSerialization:
    """Contract tests for HistoryConfig dictionary conversion."""

    def test_to_dict_preserves_all_fields(self):
        """to_dict() returns dict with all field values."""
        config = HistoryConfig(
            enabled=True,
            max_tokens=8000,
            max_entries=50,
            truncation_strategy="keep_last",
            include_types=["user_input"],
            exclude_sources=["system"],
        )

        data = config.to_dict()

        assert data == {
            "enabled": True,
            "max_tokens": 8000,
            "max_entries": 50,
            "truncation_strategy": "keep_last",
            "include_types": ["user_input"],
            "exclude_sources": ["system"],
        }

    def test_to_dict_with_none_optional_fields(self):
        """to_dict() includes None for optional fields."""
        config = HistoryConfig()

        data = config.to_dict()

        assert data["include_types"] is None
        assert data["exclude_sources"] is None

    def test_from_dict_creates_valid_config(self):
        """from_dict() creates HistoryConfig from dictionary."""
        data = {
            "enabled": True,
            "max_tokens": 6000,
            "max_entries": 30,
            "truncation_strategy": "keep_last",
            "include_types": ["agent_output"],
            "exclude_sources": ["debug"],
        }

        config = HistoryConfig.from_dict(data)

        assert config.enabled is True
        assert config.max_tokens == 6000
        assert config.max_entries == 30
        assert config.truncation_strategy == "keep_last"
        assert config.include_types == ["agent_output"]
        assert config.exclude_sources == ["debug"]

    def test_from_dict_applies_defaults_for_missing_fields(self):
        """from_dict() uses defaults for missing optional fields."""
        data = {}

        config = HistoryConfig.from_dict(data)

        assert config.enabled is True
        assert config.max_tokens == 4000
        assert config.max_entries == 20
        assert config.truncation_strategy == "oldest_first"
        assert config.include_types is None
        assert config.exclude_sources is None

    def test_round_trip_serialization_preserves_data(self):
        """to_dict() → from_dict() round trip preserves all data."""
        original = HistoryConfig(
            enabled=False,
            max_tokens=0,
            max_entries=0,
            truncation_strategy="oldest_first",
            include_types=None,
            exclude_sources=None,
        )

        data = original.to_dict()
        restored = HistoryConfig.from_dict(data)

        assert restored.enabled == original.enabled
        assert restored.max_tokens == original.max_tokens
        assert restored.max_entries == original.max_entries
        assert restored.truncation_strategy == original.truncation_strategy
        assert restored.include_types == original.include_types
        assert restored.exclude_sources == original.exclude_sources


class TestHistoryConfigTruncationStrategies:
    """Contract tests for truncation strategy validation."""

    def test_oldest_first_strategy_valid(self):
        """truncation_strategy='oldest_first' is valid."""
        config = HistoryConfig(truncation_strategy="oldest_first")

        assert config.truncation_strategy == "oldest_first"

    def test_keep_last_strategy_valid(self):
        """truncation_strategy='keep_last' is valid."""
        config = HistoryConfig(truncation_strategy="keep_last")

        assert config.truncation_strategy == "keep_last"


class TestHistoryConfigFiltering:
    """Contract tests for history filtering configuration."""

    def test_include_types_filters_history_entries(self):
        """include_types specifies entry types to include."""
        config = HistoryConfig(include_types=["user_input", "agent_output", "tool_call"])

        assert len(config.include_types) == 3
        assert "user_input" in config.include_types
        assert "agent_output" in config.include_types
        assert "tool_call" in config.include_types

    def test_exclude_sources_filters_by_source(self):
        """exclude_sources specifies sources to exclude."""
        config = HistoryConfig(exclude_sources=["system", "debug"])

        assert len(config.exclude_sources) == 2
        assert "system" in config.exclude_sources
        assert "debug" in config.exclude_sources

    def test_combined_include_exclude_filters(self):
        """include_types and exclude_sources can be used together."""
        config = HistoryConfig(
            include_types=["user_input", "agent_output"],
            exclude_sources=["system"],
        )

        assert config.include_types == ["user_input", "agent_output"]
        assert config.exclude_sources == ["system"]

    def test_empty_filter_lists_allowed(self):
        """Empty lists for filters are valid."""
        config = HistoryConfig(include_types=[], exclude_sources=[])

        assert config.include_types == []
        assert config.exclude_sources == []
