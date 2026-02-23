"""
Unit tests for CheckpointManager class (Epistemic Checkpointing).

Tests cover:
- Checkpoint creation and retrieval
- Stuck state detection (same tool 3+ times)
- Rollback functionality
- Tool history tracking
- Statistics and metrics
- Edge cases
"""

import pytest
from promptchain.utils.checkpoint_manager import CheckpointManager, Checkpoint


class TestCheckpointManagerInitialization:
    """Test CheckpointManager initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        manager = CheckpointManager()

        assert manager.stuck_threshold == 3
        assert len(manager.checkpoints) == 0
        assert manager.tool_history == []

    def test_init_with_custom_threshold(self):
        """Test initialization with custom stuck threshold."""
        manager = CheckpointManager(stuck_threshold=5)

        assert manager.stuck_threshold == 5


class TestCheckpointCreation:
    """Test checkpoint creation."""

    def test_create_single_checkpoint(self):
        """Test creating a single checkpoint."""
        manager = CheckpointManager()

        checkpoint_id = manager.create_checkpoint(
            iteration=0,
            blackboard_snapshot="snapshot_0",
            confidence=0.9
        )

        assert checkpoint_id == "cp_0"
        assert manager.get_checkpoint_count() == 1

    def test_create_multiple_checkpoints(self):
        """Test creating multiple checkpoints."""
        manager = CheckpointManager()

        cp1 = manager.create_checkpoint(0, "snapshot_0", 1.0)
        cp2 = manager.create_checkpoint(1, "snapshot_1", 0.9)
        cp3 = manager.create_checkpoint(2, "snapshot_2", 0.8)

        assert cp1 == "cp_0"
        assert cp2 == "cp_1"
        assert cp3 == "cp_2"
        assert manager.get_checkpoint_count() == 3

    def test_checkpoint_stores_tool_history(self):
        """Test that checkpoint stores copy of tool history."""
        manager = CheckpointManager()

        # Execute some tools
        manager.record_tool_execution("search")
        manager.record_tool_execution("analyze")

        # Create checkpoint
        checkpoint_id = manager.create_checkpoint(0, "snapshot_0", 1.0)

        # Execute more tools
        manager.record_tool_execution("write")

        # Retrieve checkpoint
        checkpoint = manager.get_checkpoint(checkpoint_id)

        # Checkpoint should have only first 2 tools
        assert len(checkpoint.tool_history) == 2
        assert checkpoint.tool_history == ["search", "analyze"]

        # Current history should have all 3 tools
        assert len(manager.get_tool_history()) == 3


class TestCheckpointRetrieval:
    """Test checkpoint retrieval."""

    def test_get_checkpoint_by_id(self):
        """Test retrieving checkpoint by ID."""
        manager = CheckpointManager()

        manager.create_checkpoint(0, "snapshot_0", 0.9)
        manager.create_checkpoint(1, "snapshot_1", 0.8)

        checkpoint = manager.get_checkpoint("cp_1")

        assert checkpoint is not None
        assert checkpoint.checkpoint_id == "cp_1"
        assert checkpoint.iteration == 1
        assert checkpoint.blackboard_snapshot == "snapshot_1"
        assert checkpoint.confidence == 0.8

    def test_get_nonexistent_checkpoint(self):
        """Test retrieving non-existent checkpoint."""
        manager = CheckpointManager()

        checkpoint = manager.get_checkpoint("cp_99")

        assert checkpoint is None

    def test_get_latest_checkpoint(self):
        """Test getting most recent checkpoint."""
        manager = CheckpointManager()

        manager.create_checkpoint(0, "snapshot_0", 1.0)
        manager.create_checkpoint(1, "snapshot_1", 0.9)
        manager.create_checkpoint(2, "snapshot_2", 0.8)

        latest = manager.get_latest_checkpoint()

        assert latest is not None
        assert latest.checkpoint_id == "cp_2"
        assert latest.iteration == 2

    def test_get_latest_checkpoint_when_empty(self):
        """Test getting latest checkpoint when none exist."""
        manager = CheckpointManager()

        latest = manager.get_latest_checkpoint()

        assert latest is None


class TestToolHistoryTracking:
    """Test tool execution history tracking."""

    def test_record_single_tool(self):
        """Test recording a single tool execution."""
        manager = CheckpointManager()

        manager.record_tool_execution("search_database")

        history = manager.get_tool_history()
        assert len(history) == 1
        assert history[0] == "search_database"

    def test_record_multiple_tools(self):
        """Test recording multiple tool executions."""
        manager = CheckpointManager()

        manager.record_tool_execution("search")
        manager.record_tool_execution("analyze")
        manager.record_tool_execution("write")

        history = manager.get_tool_history()
        assert len(history) == 3
        assert history == ["search", "analyze", "write"]

    def test_get_recent_tools(self):
        """Test getting last N tool calls."""
        manager = CheckpointManager()

        for i in range(10):
            manager.record_tool_execution(f"tool_{i}")

        recent = manager.get_recent_tools(3)

        assert len(recent) == 3
        assert recent == ["tool_7", "tool_8", "tool_9"]

    def test_get_recent_tools_when_fewer_exist(self):
        """Test getting recent tools when fewer than N exist."""
        manager = CheckpointManager()

        manager.record_tool_execution("tool_1")
        manager.record_tool_execution("tool_2")

        recent = manager.get_recent_tools(5)

        assert len(recent) == 2
        assert recent == ["tool_1", "tool_2"]

    def test_get_recent_tools_empty_history(self):
        """Test getting recent tools with empty history."""
        manager = CheckpointManager()

        recent = manager.get_recent_tools(5)

        assert recent == []


class TestStuckStateDetection:
    """Test stuck state detection."""

    def test_not_stuck_with_varied_tools(self):
        """Test that varied tool usage is not stuck."""
        manager = CheckpointManager(stuck_threshold=3)

        manager.record_tool_execution("search")
        manager.record_tool_execution("analyze")
        manager.record_tool_execution("write")

        assert manager.is_stuck() is False

    def test_stuck_with_repeated_tool(self):
        """Test stuck detection when same tool called 3+ times."""
        manager = CheckpointManager(stuck_threshold=3)

        manager.record_tool_execution("search")
        manager.record_tool_execution("search")
        manager.record_tool_execution("search")

        assert manager.is_stuck() is True

    def test_not_stuck_below_threshold(self):
        """Test not stuck when repetition below threshold."""
        manager = CheckpointManager(stuck_threshold=3)

        manager.record_tool_execution("search")
        manager.record_tool_execution("search")

        # Only 2 repetitions, threshold is 3
        assert manager.is_stuck() is False

    def test_stuck_with_custom_threshold(self):
        """Test stuck detection with custom threshold."""
        manager = CheckpointManager(stuck_threshold=5)

        # Call same tool 5 times
        for _ in range(5):
            manager.record_tool_execution("search")

        assert manager.is_stuck() is True

    def test_not_stuck_with_interleaved_tools(self):
        """Test that interleaved tools don't trigger stuck state."""
        manager = CheckpointManager(stuck_threshold=3)

        manager.record_tool_execution("search")
        manager.record_tool_execution("analyze")
        manager.record_tool_execution("search")
        manager.record_tool_execution("analyze")
        manager.record_tool_execution("search")

        # Last 3 calls: search, analyze, search (not 3 of same)
        assert manager.is_stuck() is False

    def test_stuck_detects_most_common_tool(self):
        """Test stuck state checks most common tool in recent window."""
        manager = CheckpointManager(stuck_threshold=3)

        manager.record_tool_execution("search")
        manager.record_tool_execution("write")
        manager.record_tool_execution("search")
        manager.record_tool_execution("write")
        manager.record_tool_execution("search")

        # Last 3 calls: search, write, search - only 2 searches
        assert manager.is_stuck() is False

        # Add one more search
        manager.record_tool_execution("search")

        # Last 3 calls: write, search, search - still only 2 searches
        assert manager.is_stuck() is False

        # Add another search
        manager.record_tool_execution("search")

        # Last 3 calls: search, search, search - 3 searches!
        assert manager.is_stuck() is True


class TestRollbackCheckpointRetrieval:
    """Test rollback checkpoint selection."""

    def test_rollback_with_multiple_checkpoints(self):
        """Test rollback returns checkpoint 2 steps back."""
        manager = CheckpointManager()

        manager.create_checkpoint(0, "snapshot_0", 1.0)
        manager.create_checkpoint(1, "snapshot_1", 0.9)
        manager.create_checkpoint(2, "snapshot_2", 0.8)
        manager.create_checkpoint(3, "snapshot_3", 0.7)

        rollback_cp = manager.get_rollback_checkpoint()

        # Should return checkpoint 2 steps back (cp_2, not cp_3)
        assert rollback_cp is not None
        assert rollback_cp.checkpoint_id == "cp_2"
        assert rollback_cp.iteration == 2

    def test_rollback_with_one_checkpoint(self):
        """Test rollback with only 1 checkpoint."""
        manager = CheckpointManager()

        manager.create_checkpoint(0, "snapshot_0", 1.0)

        rollback_cp = manager.get_rollback_checkpoint()

        # Should return the only checkpoint
        assert rollback_cp is not None
        assert rollback_cp.checkpoint_id == "cp_0"

    def test_rollback_with_no_checkpoints(self):
        """Test rollback when no checkpoints exist."""
        manager = CheckpointManager()

        rollback_cp = manager.get_rollback_checkpoint()

        assert rollback_cp is None


class TestToolFrequency:
    """Test tool frequency counting."""

    def test_get_tool_frequency(self):
        """Test getting tool execution frequency."""
        manager = CheckpointManager()

        manager.record_tool_execution("search")
        manager.record_tool_execution("search")
        manager.record_tool_execution("analyze")
        manager.record_tool_execution("search")
        manager.record_tool_execution("write")

        frequency = manager.get_tool_frequency()

        assert frequency["search"] == 3
        assert frequency["analyze"] == 1
        assert frequency["write"] == 1

    def test_get_tool_frequency_empty(self):
        """Test tool frequency with no executions."""
        manager = CheckpointManager()

        frequency = manager.get_tool_frequency()

        assert frequency == {}


class TestReset:
    """Test checkpoint manager reset."""

    def test_reset_clears_state(self):
        """Test that reset clears all checkpoints and history."""
        manager = CheckpointManager()

        # Add some state
        manager.create_checkpoint(0, "snapshot_0", 1.0)
        manager.create_checkpoint(1, "snapshot_1", 0.9)
        manager.record_tool_execution("search")
        manager.record_tool_execution("analyze")

        # Reset
        manager.reset()

        # Verify everything cleared
        assert manager.get_checkpoint_count() == 0
        assert len(manager.get_tool_history()) == 0
        assert manager.get_tool_frequency() == {}
        assert manager.get_latest_checkpoint() is None


class TestStatistics:
    """Test statistics gathering."""

    def test_get_stats_empty(self):
        """Test stats with empty manager."""
        manager = CheckpointManager(stuck_threshold=3)

        stats = manager.get_stats()

        assert stats["checkpoint_count"] == 0
        assert stats["tool_execution_count"] == 0
        assert stats["stuck_threshold"] == 3
        assert stats["recent_tools"] == []
        assert stats["tool_frequency"] == {}
        assert "latest_checkpoint" not in stats

    def test_get_stats_with_data(self):
        """Test stats with checkpoints and tool executions."""
        manager = CheckpointManager(stuck_threshold=3)

        manager.create_checkpoint(0, "snapshot_0", 0.9)
        manager.create_checkpoint(1, "snapshot_1", 0.8)
        manager.record_tool_execution("search")
        manager.record_tool_execution("analyze")
        manager.record_tool_execution("search")

        stats = manager.get_stats()

        assert stats["checkpoint_count"] == 2
        assert stats["tool_execution_count"] == 3
        assert stats["stuck_threshold"] == 3
        assert len(stats["recent_tools"]) == 3
        assert stats["tool_frequency"]["search"] == 2
        assert stats["tool_frequency"]["analyze"] == 1
        assert stats["latest_checkpoint"]["id"] == "cp_1"
        assert stats["latest_checkpoint"]["iteration"] == 1
        assert stats["latest_checkpoint"]["confidence"] == 0.8


class TestReprMethod:
    """Test string representation."""

    def test_repr_empty(self):
        """Test __repr__ with empty manager."""
        manager = CheckpointManager(stuck_threshold=3)

        repr_str = repr(manager)

        assert "CheckpointManager" in repr_str
        assert "checkpoints=0" in repr_str
        assert "tools_executed=0" in repr_str
        assert "stuck_threshold=3" in repr_str

    def test_repr_with_data(self):
        """Test __repr__ with data."""
        manager = CheckpointManager(stuck_threshold=5)

        manager.create_checkpoint(0, "snapshot_0", 1.0)
        manager.create_checkpoint(1, "snapshot_1", 0.9)
        manager.record_tool_execution("search")
        manager.record_tool_execution("analyze")
        manager.record_tool_execution("write")

        repr_str = repr(manager)

        assert "CheckpointManager" in repr_str
        assert "checkpoints=2" in repr_str
        assert "tools_executed=3" in repr_str
        assert "stuck_threshold=5" in repr_str


class TestCheckpointDataclass:
    """Test Checkpoint dataclass."""

    def test_checkpoint_creation(self):
        """Test creating Checkpoint instance."""
        checkpoint = Checkpoint(
            checkpoint_id="cp_0",
            iteration=0,
            blackboard_snapshot="snapshot_0",
            tool_history=["search", "analyze"],
            confidence=0.85
        )

        assert checkpoint.checkpoint_id == "cp_0"
        assert checkpoint.iteration == 0
        assert checkpoint.blackboard_snapshot == "snapshot_0"
        assert len(checkpoint.tool_history) == 2
        assert checkpoint.confidence == 0.85

    def test_checkpoint_default_values(self):
        """Test Checkpoint with default values."""
        checkpoint = Checkpoint(
            checkpoint_id="cp_1",
            iteration=1,
            blackboard_snapshot="snapshot_1"
        )

        assert checkpoint.tool_history == []
        assert checkpoint.confidence == 1.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_stuck_detection_with_insufficient_history(self):
        """Test stuck detection when history length < threshold."""
        manager = CheckpointManager(stuck_threshold=5)

        manager.record_tool_execution("search")
        manager.record_tool_execution("search")

        # Only 2 executions, threshold is 5
        assert manager.is_stuck() is False

    def test_checkpoint_with_empty_tool_history(self):
        """Test creating checkpoint with no tool history."""
        manager = CheckpointManager()

        checkpoint_id = manager.create_checkpoint(0, "snapshot_0", 1.0)
        checkpoint = manager.get_checkpoint(checkpoint_id)

        assert checkpoint.tool_history == []

    def test_multiple_stuck_detections(self):
        """Test that stuck state can be detected multiple times."""
        manager = CheckpointManager(stuck_threshold=3)

        # First stuck state
        for _ in range(3):
            manager.record_tool_execution("search")

        assert manager.is_stuck() is True

        # Continue with same tool
        manager.record_tool_execution("search")

        # Still stuck
        assert manager.is_stuck() is True

    def test_recovery_from_stuck_state(self):
        """Test that stuck state clears with different tools."""
        manager = CheckpointManager(stuck_threshold=3)

        # Get stuck
        for _ in range(3):
            manager.record_tool_execution("search")

        assert manager.is_stuck() is True

        # Execute different tools
        manager.record_tool_execution("analyze")
        manager.record_tool_execution("write")

        # Now last 3 calls are: search, analyze, write (not stuck)
        assert manager.is_stuck() is False


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
