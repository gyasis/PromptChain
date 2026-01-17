"""Unit tests for blackboard tools.

Tests cover:
- write_to_blackboard: String values, dict/list values (JSON), validation
- read_from_blackboard: Existing keys, non-existent keys
- list_blackboard_keys: All keys, empty blackboard
- delete_blackboard_entry: Successful deletion, non-existent key

Note: The blackboard tools have a design issue - they call session_manager
methods that require session_id, but the tools don't have access to it.
For testing, we mock the session_manager to provide a default session_id.
"""

import json
from unittest.mock import Mock, MagicMock, patch
import pytest

from promptchain.cli.tools.library import blackboard_tools
from promptchain.cli.models.blackboard import BlackboardEntry


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager with blackboard support.

    This mock implements the session manager interface that blackboard
    tools expect, but with a simplified session_id handling.
    """
    mock_sm = Mock()

    # Storage for blackboard entries (keyed by key, not session_id for simplicity)
    mock_sm._blackboard_storage = {}

    def mock_write_blackboard(key, value, written_by):
        """Mock write_blackboard that doesn't require session_id."""
        if key in mock_sm._blackboard_storage:
            # Update existing entry
            entry = mock_sm._blackboard_storage[key]
            entry.update(value, written_by)
            return entry
        else:
            # Create new entry
            entry = BlackboardEntry.create(key=key, value=value, written_by=written_by)
            mock_sm._blackboard_storage[key] = entry
            return entry

    def mock_read_blackboard(key):
        """Mock read_blackboard that doesn't require session_id."""
        return mock_sm._blackboard_storage.get(key)

    def mock_list_blackboard_keys():
        """Mock list_blackboard_keys that doesn't require session_id."""
        return list(mock_sm._blackboard_storage.keys())

    def mock_delete_blackboard_entry(key):
        """Mock delete_blackboard_entry that doesn't require session_id."""
        if key in mock_sm._blackboard_storage:
            del mock_sm._blackboard_storage[key]
            return True
        return False

    # Assign mock methods
    mock_sm.write_blackboard = Mock(side_effect=mock_write_blackboard)
    mock_sm.read_blackboard = Mock(side_effect=mock_read_blackboard)
    mock_sm.list_blackboard_keys = Mock(side_effect=mock_list_blackboard_keys)
    mock_sm.delete_blackboard_entry = Mock(side_effect=mock_delete_blackboard_entry)

    return mock_sm


@pytest.fixture(autouse=True)
def setup_session_manager(mock_session_manager):
    """Automatically set up mock session manager for all tests."""
    blackboard_tools.set_session_manager(mock_session_manager)
    yield
    # Reset after each test
    blackboard_tools._session_manager = None


class TestWriteToBlackboard:
    """Test write_to_blackboard function."""

    def test_write_string_value(self, mock_session_manager):
        """Test writing a simple string value."""
        result = blackboard_tools.write_to_blackboard(
            key="test_key",
            value="test_value",
            written_by="test_agent"
        )

        assert "Blackboard entry 'test_key' written by test_agent (v1)" == result
        mock_session_manager.write_blackboard.assert_called_once_with(
            key="test_key",
            value="test_value",
            written_by="test_agent"
        )

    def test_write_dict_value(self, mock_session_manager):
        """Test writing a dictionary value (JSON serialization)."""
        dict_value = {"score": 0.95, "status": "complete", "items": [1, 2, 3]}

        result = blackboard_tools.write_to_blackboard(
            key="analysis_results",
            value=dict_value,
            written_by="analyzer_agent"
        )

        assert "Blackboard entry 'analysis_results' written by analyzer_agent (v1)" == result

        # Verify the value was passed correctly
        call_args = mock_session_manager.write_blackboard.call_args
        assert call_args.kwargs["value"] == dict_value

    def test_write_list_value(self, mock_session_manager):
        """Test writing a list value (JSON serialization)."""
        list_value = ["item1", "item2", "item3"]

        result = blackboard_tools.write_to_blackboard(
            key="shared_context",
            value=list_value,
            written_by="coordinator"
        )

        assert "Blackboard entry 'shared_context' written by coordinator (v1)" == result

        call_args = mock_session_manager.write_blackboard.call_args
        assert call_args.kwargs["value"] == list_value

    def test_write_numeric_value(self, mock_session_manager):
        """Test writing numeric values (int, float)."""
        # Integer
        result = blackboard_tools.write_to_blackboard(
            key="step_count",
            value=42,
            written_by="orchestrator"
        )
        assert "Blackboard entry 'step_count' written by orchestrator (v1)" == result

        # Float
        result = blackboard_tools.write_to_blackboard(
            key="confidence_score",
            value=0.87,
            written_by="evaluator"
        )
        assert "Blackboard entry 'confidence_score' written by evaluator (v1)" == result

    def test_write_boolean_value(self, mock_session_manager):
        """Test writing boolean values."""
        result = blackboard_tools.write_to_blackboard(
            key="processing_complete",
            value=True,
            written_by="processor"
        )

        assert "Blackboard entry 'processing_complete' written by processor (v1)" == result

    def test_write_nested_json(self, mock_session_manager):
        """Test writing nested JSON structures."""
        nested_value = {
            "metadata": {
                "timestamp": 1234567890,
                "author": "agent1"
            },
            "results": [
                {"id": 1, "score": 0.9},
                {"id": 2, "score": 0.8}
            ]
        }

        result = blackboard_tools.write_to_blackboard(
            key="complex_data",
            value=nested_value,
            written_by="complex_agent"
        )

        assert "Blackboard entry 'complex_data' written by complex_agent (v1)" == result

    def test_write_update_existing_key(self, mock_session_manager):
        """Test updating an existing key increments version."""
        # First write
        blackboard_tools.write_to_blackboard(
            key="counter",
            value=1,
            written_by="agent1"
        )

        # Second write (update)
        result = blackboard_tools.write_to_blackboard(
            key="counter",
            value=2,
            written_by="agent2"
        )

        # Version should be incremented to 2
        assert "(v2)" in result
        assert "agent2" in result

    def test_write_empty_key_validation(self, mock_session_manager):
        """Test validation: empty key should return error."""
        result = blackboard_tools.write_to_blackboard(
            key="",
            value="test",
            written_by="agent"
        )

        assert result == "Error: Key cannot be empty"
        mock_session_manager.write_blackboard.assert_not_called()

    def test_write_whitespace_key_validation(self, mock_session_manager):
        """Test validation: whitespace-only key should return error."""
        result = blackboard_tools.write_to_blackboard(
            key="   ",
            value="test",
            written_by="agent"
        )

        assert result == "Error: Key cannot be empty"
        mock_session_manager.write_blackboard.assert_not_called()

    def test_write_empty_written_by_validation(self, mock_session_manager):
        """Test validation: empty written_by should return error."""
        result = blackboard_tools.write_to_blackboard(
            key="test_key",
            value="test",
            written_by=""
        )

        assert result == "Error: written_by cannot be empty"
        mock_session_manager.write_blackboard.assert_not_called()

    def test_write_whitespace_written_by_validation(self, mock_session_manager):
        """Test validation: whitespace-only written_by should return error."""
        result = blackboard_tools.write_to_blackboard(
            key="test_key",
            value="test",
            written_by="   "
        )

        assert result == "Error: written_by cannot be empty"
        mock_session_manager.write_blackboard.assert_not_called()

    def test_write_none_value(self, mock_session_manager):
        """Test writing None value is allowed."""
        result = blackboard_tools.write_to_blackboard(
            key="nullable_field",
            value=None,
            written_by="agent"
        )

        assert "Blackboard entry 'nullable_field' written by agent (v1)" == result


class TestReadFromBlackboard:
    """Test read_from_blackboard function."""

    def test_read_existing_string_value(self, mock_session_manager):
        """Test reading an existing string value."""
        # Write first
        blackboard_tools.write_to_blackboard(
            key="test_key",
            value="test_value",
            written_by="writer_agent"
        )

        # Read
        result = blackboard_tools.read_from_blackboard(key="test_key")

        assert "Blackboard['test_key'] (v1, by writer_agent):" in result
        assert "test_value" in result
        mock_session_manager.read_blackboard.assert_called_once_with("test_key")

    def test_read_existing_dict_value(self, mock_session_manager):
        """Test reading an existing dictionary value with JSON formatting."""
        dict_value = {"score": 0.95, "status": "complete"}

        blackboard_tools.write_to_blackboard(
            key="results",
            value=dict_value,
            written_by="analyzer"
        )

        result = blackboard_tools.read_from_blackboard(key="results")

        assert "Blackboard['results'] (v1, by analyzer):" in result
        # Should be formatted as JSON with indentation
        assert '"score": 0.95' in result or '"score":0.95' in result
        assert '"status": "complete"' in result or '"status":"complete"' in result

    def test_read_existing_list_value(self, mock_session_manager):
        """Test reading an existing list value with JSON formatting."""
        list_value = ["item1", "item2", "item3"]

        blackboard_tools.write_to_blackboard(
            key="items",
            value=list_value,
            written_by="collector"
        )

        result = blackboard_tools.read_from_blackboard(key="items")

        assert "Blackboard['items'] (v1, by collector):" in result
        assert "item1" in result
        assert "item2" in result
        assert "item3" in result

    def test_read_numeric_value(self, mock_session_manager):
        """Test reading numeric values."""
        blackboard_tools.write_to_blackboard(
            key="count",
            value=42,
            written_by="counter"
        )

        result = blackboard_tools.read_from_blackboard(key="count")

        assert "Blackboard['count'] (v1, by counter):" in result
        assert "42" in result

    def test_read_boolean_value(self, mock_session_manager):
        """Test reading boolean values."""
        blackboard_tools.write_to_blackboard(
            key="is_complete",
            value=True,
            written_by="checker"
        )

        result = blackboard_tools.read_from_blackboard(key="is_complete")

        assert "Blackboard['is_complete'] (v1, by checker):" in result
        assert "True" in result

    def test_read_updated_entry_shows_new_version(self, mock_session_manager):
        """Test reading updated entry shows incremented version."""
        # Write initial
        blackboard_tools.write_to_blackboard(
            key="counter",
            value=1,
            written_by="agent1"
        )

        # Update
        blackboard_tools.write_to_blackboard(
            key="counter",
            value=2,
            written_by="agent2"
        )

        # Read should show v2 and agent2
        result = blackboard_tools.read_from_blackboard(key="counter")

        assert "(v2, by agent2)" in result
        assert "2" in result

    def test_read_non_existent_key(self, mock_session_manager):
        """Test reading a non-existent key returns error message."""
        result = blackboard_tools.read_from_blackboard(key="non_existent_key")

        assert result == "No entry found for key 'non_existent_key'"
        mock_session_manager.read_blackboard.assert_called_once_with("non_existent_key")

    def test_read_after_delete(self, mock_session_manager):
        """Test reading after deletion returns not found."""
        # Write and delete
        blackboard_tools.write_to_blackboard(
            key="temp_key",
            value="temp_value",
            written_by="temp_agent"
        )
        blackboard_tools.delete_blackboard_entry(key="temp_key")

        # Read should return not found
        result = blackboard_tools.read_from_blackboard(key="temp_key")

        assert "No entry found for key 'temp_key'" == result

    def test_read_none_value(self, mock_session_manager):
        """Test reading None value."""
        blackboard_tools.write_to_blackboard(
            key="null_field",
            value=None,
            written_by="agent"
        )

        result = blackboard_tools.read_from_blackboard(key="null_field")

        assert "Blackboard['null_field'] (v1, by agent):" in result
        assert "None" in result


class TestListBlackboardKeys:
    """Test list_blackboard_keys function."""

    def test_list_empty_blackboard(self, mock_session_manager):
        """Test listing keys when blackboard is empty."""
        result = blackboard_tools.list_blackboard_keys()

        assert result == "Blackboard is empty"
        mock_session_manager.list_blackboard_keys.assert_called_once()

    def test_list_single_key(self, mock_session_manager):
        """Test listing with a single key."""
        blackboard_tools.write_to_blackboard(
            key="only_key",
            value="value",
            written_by="agent"
        )

        result = blackboard_tools.list_blackboard_keys()

        assert "Blackboard keys (1): only_key" == result

    def test_list_multiple_keys(self, mock_session_manager):
        """Test listing multiple keys."""
        # Write multiple entries
        blackboard_tools.write_to_blackboard(
            key="key1",
            value="value1",
            written_by="agent1"
        )
        blackboard_tools.write_to_blackboard(
            key="key2",
            value="value2",
            written_by="agent2"
        )
        blackboard_tools.write_to_blackboard(
            key="key3",
            value="value3",
            written_by="agent3"
        )

        result = blackboard_tools.list_blackboard_keys()

        assert "Blackboard keys (3):" in result
        assert "key1" in result
        assert "key2" in result
        assert "key3" in result

    def test_list_keys_are_sorted(self, mock_session_manager):
        """Test that keys are returned in sorted order."""
        # Write in non-alphabetical order
        blackboard_tools.write_to_blackboard(key="zebra", value="z", written_by="agent")
        blackboard_tools.write_to_blackboard(key="apple", value="a", written_by="agent")
        blackboard_tools.write_to_blackboard(key="mango", value="m", written_by="agent")

        result = blackboard_tools.list_blackboard_keys()

        # Keys should be sorted alphabetically
        assert "Blackboard keys (3): apple, mango, zebra" == result

    def test_list_after_deletion(self, mock_session_manager):
        """Test listing after deleting some keys."""
        # Write three keys
        blackboard_tools.write_to_blackboard(key="key1", value="v1", written_by="agent")
        blackboard_tools.write_to_blackboard(key="key2", value="v2", written_by="agent")
        blackboard_tools.write_to_blackboard(key="key3", value="v3", written_by="agent")

        # Delete one
        blackboard_tools.delete_blackboard_entry(key="key2")

        # List should show remaining keys
        result = blackboard_tools.list_blackboard_keys()

        assert "Blackboard keys (2):" in result
        assert "key1" in result
        assert "key2" not in result
        assert "key3" in result

    def test_list_after_delete_all(self, mock_session_manager):
        """Test listing after deleting all keys."""
        # Write and delete
        blackboard_tools.write_to_blackboard(key="temp", value="v", written_by="agent")
        blackboard_tools.delete_blackboard_entry(key="temp")

        result = blackboard_tools.list_blackboard_keys()

        assert result == "Blackboard is empty"


class TestDeleteBlackboardEntry:
    """Test delete_blackboard_entry function."""

    def test_delete_existing_entry(self, mock_session_manager):
        """Test deleting an existing entry."""
        # Write first
        blackboard_tools.write_to_blackboard(
            key="to_delete",
            value="value",
            written_by="agent"
        )

        # Delete
        result = blackboard_tools.delete_blackboard_entry(key="to_delete")

        assert result == "Blackboard entry 'to_delete' deleted"
        mock_session_manager.delete_blackboard_entry.assert_called_once_with("to_delete")

    def test_delete_non_existent_entry(self, mock_session_manager):
        """Test deleting a non-existent entry."""
        result = blackboard_tools.delete_blackboard_entry(key="non_existent")

        assert result == "No entry found for key 'non_existent'"
        mock_session_manager.delete_blackboard_entry.assert_called_once_with("non_existent")

    def test_delete_then_read_returns_not_found(self, mock_session_manager):
        """Test that reading after deletion returns not found."""
        # Write, delete, then try to read
        blackboard_tools.write_to_blackboard(key="temp", value="v", written_by="agent")
        blackboard_tools.delete_blackboard_entry(key="temp")

        result = blackboard_tools.read_from_blackboard(key="temp")

        assert "No entry found for key 'temp'" == result

    def test_delete_then_list_excludes_key(self, mock_session_manager):
        """Test that listing after deletion excludes the deleted key."""
        # Write two keys
        blackboard_tools.write_to_blackboard(key="keep", value="v1", written_by="agent")
        blackboard_tools.write_to_blackboard(key="delete", value="v2", written_by="agent")

        # Delete one
        blackboard_tools.delete_blackboard_entry(key="delete")

        result = blackboard_tools.list_blackboard_keys()

        assert "keep" in result
        assert "delete" not in result

    def test_delete_twice_returns_not_found(self, mock_session_manager):
        """Test deleting the same key twice."""
        # Write and delete
        blackboard_tools.write_to_blackboard(key="once", value="v", written_by="agent")

        result1 = blackboard_tools.delete_blackboard_entry(key="once")
        assert "deleted" in result1

        # Delete again
        result2 = blackboard_tools.delete_blackboard_entry(key="once")
        assert "No entry found" in result2


class TestSessionManagerIntegration:
    """Test integration with session manager."""

    def test_get_session_manager_not_initialized(self):
        """Test error when session manager not initialized."""
        # Reset session manager
        blackboard_tools._session_manager = None

        with pytest.raises(RuntimeError, match="Session manager not initialized"):
            blackboard_tools.get_session_manager()

    def test_set_session_manager(self, mock_session_manager):
        """Test setting session manager."""
        blackboard_tools.set_session_manager(mock_session_manager)

        sm = blackboard_tools.get_session_manager()
        assert sm is mock_session_manager

    def test_write_without_session_manager_raises_error(self):
        """Test write raises error when session manager not set."""
        blackboard_tools._session_manager = None

        with pytest.raises(RuntimeError, match="Session manager not initialized"):
            blackboard_tools.write_to_blackboard(
                key="test",
                value="value",
                written_by="agent"
            )

    def test_read_without_session_manager_raises_error(self):
        """Test read raises error when session manager not set."""
        blackboard_tools._session_manager = None

        with pytest.raises(RuntimeError, match="Session manager not initialized"):
            blackboard_tools.read_from_blackboard(key="test")

    def test_list_without_session_manager_raises_error(self):
        """Test list raises error when session manager not set."""
        blackboard_tools._session_manager = None

        with pytest.raises(RuntimeError, match="Session manager not initialized"):
            blackboard_tools.list_blackboard_keys()

    def test_delete_without_session_manager_raises_error(self):
        """Test delete raises error when session manager not set."""
        blackboard_tools._session_manager = None

        with pytest.raises(RuntimeError, match="Session manager not initialized"):
            blackboard_tools.delete_blackboard_entry(key="test")


class TestJSONSerializationRoundTrip:
    """Test JSON serialization/deserialization works correctly."""

    def test_dict_serialization_roundtrip(self, mock_session_manager):
        """Test dict values can be written and read back correctly."""
        original_value = {
            "name": "test",
            "count": 42,
            "active": True,
            "tags": ["a", "b", "c"]
        }

        # Write
        blackboard_tools.write_to_blackboard(
            key="test_dict",
            value=original_value,
            written_by="agent"
        )

        # Read back
        result = blackboard_tools.read_from_blackboard(key="test_dict")

        # Verify all fields are present in output
        assert "name" in result
        assert "test" in result
        assert "count" in result
        assert "42" in result
        assert "active" in result
        assert "tags" in result

    def test_list_serialization_roundtrip(self, mock_session_manager):
        """Test list values can be written and read back correctly."""
        original_value = [1, 2, 3, "four", True, None]

        blackboard_tools.write_to_blackboard(
            key="test_list",
            value=original_value,
            written_by="agent"
        )

        result = blackboard_tools.read_from_blackboard(key="test_list")

        # Verify list elements are present
        assert "1" in result
        assert "2" in result
        assert "3" in result
        assert "four" in result

    def test_nested_structure_serialization(self, mock_session_manager):
        """Test nested structures serialize correctly."""
        nested = {
            "level1": {
                "level2": {
                    "level3": ["deep", "value"]
                }
            }
        }

        blackboard_tools.write_to_blackboard(
            key="nested",
            value=nested,
            written_by="agent"
        )

        result = blackboard_tools.read_from_blackboard(key="nested")

        assert "level1" in result
        assert "level2" in result
        assert "level3" in result
        assert "deep" in result


class TestVersionTracking:
    """Test version tracking across updates."""

    def test_initial_version_is_one(self, mock_session_manager):
        """Test that initial write creates version 1."""
        blackboard_tools.write_to_blackboard(
            key="versioned",
            value="v1",
            written_by="agent1"
        )

        result = blackboard_tools.read_from_blackboard(key="versioned")
        assert "(v1, by agent1)" in result

    def test_version_increments_on_update(self, mock_session_manager):
        """Test that updates increment version."""
        # Initial write
        blackboard_tools.write_to_blackboard(
            key="versioned",
            value="v1",
            written_by="agent1"
        )

        # Update 1
        blackboard_tools.write_to_blackboard(
            key="versioned",
            value="v2",
            written_by="agent2"
        )

        # Update 2
        blackboard_tools.write_to_blackboard(
            key="versioned",
            value="v3",
            written_by="agent3"
        )

        result = blackboard_tools.read_from_blackboard(key="versioned")
        assert "(v3, by agent3)" in result

    def test_different_keys_have_independent_versions(self, mock_session_manager):
        """Test that different keys have independent version tracking."""
        # Write key1 twice
        blackboard_tools.write_to_blackboard(key="key1", value="a", written_by="agent")
        blackboard_tools.write_to_blackboard(key="key1", value="b", written_by="agent")

        # Write key2 once
        blackboard_tools.write_to_blackboard(key="key2", value="x", written_by="agent")

        result1 = blackboard_tools.read_from_blackboard(key="key1")
        result2 = blackboard_tools.read_from_blackboard(key="key2")

        assert "(v2, by agent)" in result1
        assert "(v1, by agent)" in result2
