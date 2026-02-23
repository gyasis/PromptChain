"""
Unit tests for Blackboard class.

Tests cover:
- Initialization
- State management (facts, observations, plan)
- LRU eviction
- Snapshot/rollback functionality
- Prompt generation
- Edge cases
"""

import pytest
from promptchain.utils.blackboard import Blackboard


class TestBlackboardInitialization:
    """Test Blackboard initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        bb = Blackboard(objective="Test objective")

        assert bb.objective == "Test objective"
        assert bb.max_plan_items == 10
        assert bb.max_facts == 20
        assert bb.max_observations == 15

        state = bb.get_state()
        assert state["objective"] == "Test objective"
        assert state["current_plan"] == []
        assert state["completed_steps"] == []
        assert state["facts_discovered"] == {}
        assert state["confidence"] == 1.0
        assert state["observations"] == []
        assert state["errors"] == []
        assert state["tool_results"] == {}
        assert state["iteration_count"] == 0

    def test_init_with_custom_limits(self):
        """Test initialization with custom limits."""
        bb = Blackboard(
            objective="Custom limits test",
            max_plan_items=5,
            max_facts=10,
            max_observations=8
        )

        assert bb.max_plan_items == 5
        assert bb.max_facts == 10
        assert bb.max_observations == 8


class TestFactManagement:
    """Test fact storage and LRU eviction."""

    def test_add_single_fact(self):
        """Test adding a single fact."""
        bb = Blackboard(objective="Test", max_facts=5)
        bb.add_fact("fact1", "value1")

        facts = bb.get_facts()
        assert facts == {"fact1": "value1"}

    def test_add_multiple_facts(self):
        """Test adding multiple facts."""
        bb = Blackboard(objective="Test", max_facts=5)
        bb.add_fact("fact1", "value1")
        bb.add_fact("fact2", "value2")
        bb.add_fact("fact3", "value3")

        facts = bb.get_facts()
        assert len(facts) == 3
        assert facts["fact1"] == "value1"
        assert facts["fact2"] == "value2"
        assert facts["fact3"] == "value3"

    def test_lru_eviction(self):
        """Test LRU eviction when max_facts is reached."""
        bb = Blackboard(objective="Test", max_facts=3)

        # Add 3 facts (at limit)
        bb.add_fact("fact1", "value1")
        bb.add_fact("fact2", "value2")
        bb.add_fact("fact3", "value3")

        facts = bb.get_facts()
        assert len(facts) == 3

        # Add 4th fact - should evict fact1 (oldest)
        bb.add_fact("fact4", "value4")

        facts = bb.get_facts()
        assert len(facts) == 3
        assert "fact1" not in facts  # Evicted
        assert "fact2" in facts
        assert "fact3" in facts
        assert "fact4" in facts

    def test_fact_types(self):
        """Test storing different fact types."""
        bb = Blackboard(objective="Test")

        bb.add_fact("string_fact", "string value")
        bb.add_fact("int_fact", 42)
        bb.add_fact("list_fact", [1, 2, 3])
        bb.add_fact("dict_fact", {"key": "value"})

        facts = bb.get_facts()
        assert facts["string_fact"] == "string value"
        assert facts["int_fact"] == 42
        assert facts["list_fact"] == [1, 2, 3]
        assert facts["dict_fact"] == {"key": "value"}


class TestObservationManagement:
    """Test observation storage and sliding window."""

    def test_add_single_observation(self):
        """Test adding a single observation."""
        bb = Blackboard(objective="Test", max_observations=5)
        bb.add_observation("First observation")

        state = bb.get_state()
        assert len(state["observations"]) == 1
        assert state["observations"][0] == "First observation"

    def test_observation_sliding_window(self):
        """Test sliding window when max_observations is exceeded."""
        bb = Blackboard(objective="Test", max_observations=3)

        bb.add_observation("obs1")
        bb.add_observation("obs2")
        bb.add_observation("obs3")

        state = bb.get_state()
        assert len(state["observations"]) == 3

        # Add 4th observation - should keep only last 3
        bb.add_observation("obs4")

        state = bb.get_state()
        assert len(state["observations"]) == 3
        assert "obs1" not in state["observations"]
        assert state["observations"] == ["obs2", "obs3", "obs4"]


class TestPlanManagement:
    """Test plan storage and management."""

    def test_update_plan(self):
        """Test updating the current plan."""
        bb = Blackboard(objective="Test", max_plan_items=5)

        plan = ["Step 1", "Step 2", "Step 3"]
        bb.update_plan(plan)

        current_plan = bb.get_plan()
        assert current_plan == ["Step 1", "Step 2", "Step 3"]

    def test_plan_truncation(self):
        """Test plan truncation when exceeding max_plan_items."""
        bb = Blackboard(objective="Test", max_plan_items=3)

        plan = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]
        bb.update_plan(plan)

        # Should keep only last 3 items
        current_plan = bb.get_plan()
        assert len(current_plan) == 3
        assert current_plan == ["Step 3", "Step 4", "Step 5"]


class TestStepCompletion:
    """Test step completion tracking."""

    def test_mark_step_complete(self):
        """Test marking steps as complete."""
        bb = Blackboard(objective="Test")

        bb.mark_step_complete("Completed step 1")
        bb.mark_step_complete("Completed step 2")

        state = bb.get_state()
        assert len(state["completed_steps"]) == 2
        assert "Completed step 1" in state["completed_steps"]
        assert "Completed step 2" in state["completed_steps"]


class TestErrorTracking:
    """Test error tracking."""

    def test_add_error(self):
        """Test adding errors."""
        bb = Blackboard(objective="Test")

        bb.add_error("Error 1")
        bb.add_error("Error 2")

        state = bb.get_state()
        assert len(state["errors"]) == 2
        assert state["errors"] == ["Error 1", "Error 2"]

    def test_clear_errors(self):
        """Test clearing errors."""
        bb = Blackboard(objective="Test")

        bb.add_error("Error 1")
        bb.add_error("Error 2")

        bb.clear_errors()

        state = bb.get_state()
        assert len(state["errors"]) == 0


class TestToolResults:
    """Test tool result storage."""

    def test_store_tool_result(self):
        """Test storing tool results."""
        bb = Blackboard(objective="Test")

        bb.store_tool_result("search_tool", "Found 10 results")
        bb.store_tool_result("calculate_tool", "Result: 42")

        state = bb.get_state()
        assert state["tool_results"]["search_tool"] == "Found 10 results"
        assert state["tool_results"]["calculate_tool"] == "Result: 42"

    def test_tool_result_truncation(self):
        """Test tool result truncation for long results."""
        bb = Blackboard(objective="Test")

        long_result = "A" * 1000
        bb.store_tool_result("long_tool", long_result, truncate_at=100)

        state = bb.get_state()
        result = state["tool_results"]["long_tool"]

        # Should be truncated to 100 chars + truncation message
        assert len(result) < len(long_result)
        assert "truncated" in result
        assert result.startswith("A" * 100)


class TestConfidenceTracking:
    """Test confidence level tracking."""

    def test_update_confidence(self):
        """Test updating confidence level."""
        bb = Blackboard(objective="Test")

        bb.update_confidence(0.8)
        assert bb.get_state()["confidence"] == 0.8

        bb.update_confidence(0.5)
        assert bb.get_state()["confidence"] == 0.5

    def test_confidence_bounds(self):
        """Test confidence is bounded to [0.0, 1.0]."""
        bb = Blackboard(objective="Test")

        # Test upper bound
        bb.update_confidence(1.5)
        assert bb.get_state()["confidence"] == 1.0

        # Test lower bound
        bb.update_confidence(-0.5)
        assert bb.get_state()["confidence"] == 0.0


class TestIterationTracking:
    """Test iteration counter."""

    def test_increment_iteration(self):
        """Test iteration counter increments correctly."""
        bb = Blackboard(objective="Test")

        assert bb.get_state()["iteration_count"] == 0

        bb.increment_iteration()
        assert bb.get_state()["iteration_count"] == 1

        bb.increment_iteration()
        assert bb.get_state()["iteration_count"] == 2


class TestPromptGeneration:
    """Test to_prompt() method."""

    def test_empty_blackboard_prompt(self):
        """Test prompt generation with empty blackboard."""
        bb = Blackboard(objective="Test objective")

        prompt = bb.to_prompt()

        assert "OBJECTIVE: Test objective" in prompt
        assert "CURRENT PLAN:" in prompt
        assert "COMPLETED STEPS:" in prompt
        assert "FACTS DISCOVERED:" in prompt
        assert "RECENT OBSERVATIONS:" in prompt
        assert "CONFIDENCE: 1.00" in prompt
        assert "ITERATION: 0" in prompt

    def test_populated_blackboard_prompt(self):
        """Test prompt generation with populated blackboard."""
        bb = Blackboard(objective="Research task")

        bb.update_plan(["Step 1: Research", "Step 2: Analyze"])
        bb.add_fact("database", "PostgreSQL")
        bb.add_observation("Executed search")
        bb.mark_step_complete("Research completed")
        bb.store_tool_result("search_tool", "Found 5 results")
        bb.update_confidence(0.85)
        bb.increment_iteration()

        prompt = bb.to_prompt()

        # Verify all sections are present
        assert "OBJECTIVE: Research task" in prompt
        assert "1. Step 1: Research" in prompt
        assert "2. Step 2: Analyze" in prompt
        assert "✓ Research completed" in prompt
        assert "database: PostgreSQL" in prompt
        assert "→ Executed search" in prompt
        assert "CONFIDENCE: 0.85" in prompt
        assert "ITERATION: 1" in prompt
        assert "search_tool" in prompt

    def test_prompt_limits_history(self):
        """Test that prompt only includes recent items."""
        bb = Blackboard(
            objective="Test",
            max_plan_items=2,
            max_facts=2,
            max_observations=2
        )

        # Add more items than limits
        bb.update_plan(["P1", "P2", "P3", "P4"])
        bb.add_fact("f1", "v1")
        bb.add_fact("f2", "v2")
        bb.add_fact("f3", "v3")
        bb.add_observation("o1")
        bb.add_observation("o2")
        bb.add_observation("o3")

        prompt = bb.to_prompt()

        # Should only show recent items
        assert "P3" in prompt
        assert "P4" in prompt
        assert "P1" not in prompt

        # Facts respect LRU so last 2 facts should be f2 and f3
        assert "f2: v2" in prompt
        assert "f3: v3" in prompt

        # Observations: last 2
        assert "o2" in prompt
        assert "o3" in prompt

    def test_prompt_with_errors(self):
        """Test prompt includes errors when present."""
        bb = Blackboard(objective="Test")

        bb.add_error("Connection timeout")
        bb.add_error("Invalid input")

        prompt = bb.to_prompt()

        assert "ERRORS ENCOUNTERED:" in prompt
        assert "⚠ Connection timeout" in prompt
        assert "⚠ Invalid input" in prompt


class TestSnapshotAndRollback:
    """Test snapshot and rollback functionality."""

    def test_create_snapshot(self):
        """Test creating a snapshot."""
        bb = Blackboard(objective="Test")

        bb.add_fact("fact1", "value1")
        snapshot_id = bb.snapshot()

        assert snapshot_id == "snapshot_0"
        assert bb.get_snapshot_count() == 1

    def test_multiple_snapshots(self):
        """Test creating multiple snapshots."""
        bb = Blackboard(objective="Test")

        snapshot1 = bb.snapshot()
        snapshot2 = bb.snapshot()
        snapshot3 = bb.snapshot()

        assert snapshot1 == "snapshot_0"
        assert snapshot2 == "snapshot_1"
        assert snapshot3 == "snapshot_2"
        assert bb.get_snapshot_count() == 3

    def test_rollback_to_snapshot(self):
        """Test rolling back to a previous snapshot."""
        bb = Blackboard(objective="Test")

        # Initial state
        bb.add_fact("fact1", "value1")
        bb.add_observation("obs1")
        snapshot_id = bb.snapshot()

        # Modify state
        bb.add_fact("fact2", "value2")
        bb.add_observation("obs2")
        bb.update_confidence(0.5)

        # Verify modified state
        state = bb.get_state()
        assert len(state["facts_discovered"]) == 2
        assert len(state["observations"]) == 2
        assert state["confidence"] == 0.5

        # Rollback
        bb.rollback(snapshot_id)

        # Verify state restored
        state = bb.get_state()
        assert len(state["facts_discovered"]) == 1
        assert len(state["observations"]) == 1
        assert state["confidence"] == 1.0

    def test_rollback_invalid_snapshot(self):
        """Test rollback with invalid snapshot ID."""
        bb = Blackboard(objective="Test")

        with pytest.raises(ValueError):
            bb.rollback("snapshot_99")

    def test_snapshot_isolation(self):
        """Test that snapshot is isolated from current state."""
        bb = Blackboard(objective="Test")

        bb.add_fact("fact1", "value1")
        snapshot_id = bb.snapshot()

        # Modify state
        bb.add_fact("fact1", "value2_modified")

        # Rollback should restore original value
        bb.rollback(snapshot_id)

        state = bb.get_state()
        assert state["facts_discovered"]["fact1"] == "value1"


class TestReprAndString:
    """Test string representations."""

    def test_repr(self):
        """Test __repr__ output."""
        bb = Blackboard(objective="This is a very long objective that should be truncated")

        repr_str = repr(bb)

        assert "Blackboard" in repr_str
        assert "objective=" in repr_str
        assert "facts=" in repr_str
        assert "observations=" in repr_str
        assert "completed_steps=" in repr_str
        assert "snapshots=" in repr_str


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_plan_update(self):
        """Test updating with empty plan."""
        bb = Blackboard(objective="Test")

        bb.update_plan([])

        plan = bb.get_plan()
        assert plan == []

    def test_get_state_is_copy(self):
        """Test that get_state returns a copy, not reference."""
        bb = Blackboard(objective="Test")

        bb.add_fact("fact1", "value1")

        state1 = bb.get_state()
        state1["facts_discovered"]["fact2"] = "value2"  # Modify copy

        state2 = bb.get_state()

        # Original state should not be affected
        assert "fact2" not in state2["facts_discovered"]

    def test_get_facts_is_copy(self):
        """Test that get_facts returns a copy."""
        bb = Blackboard(objective="Test")

        bb.add_fact("fact1", "value1")

        facts1 = bb.get_facts()
        facts1["fact2"] = "value2"  # Modify copy

        facts2 = bb.get_facts()

        # Original facts should not be affected
        assert "fact2" not in facts2

    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        bb = Blackboard(objective="Test")

        bb.add_fact("unicode_fact", "Hello 世界 🌍")
        bb.add_observation("Observation with émojis 🎉")

        prompt = bb.to_prompt()

        assert "Hello 世界 🌍" in prompt
        assert "Observation with émojis 🎉" in prompt


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
