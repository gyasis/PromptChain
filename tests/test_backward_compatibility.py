"""
Backward Compatibility Test

Ensures that AgenticStepProcessor works correctly with all new features disabled.
This validates that existing code won't break when upgrading.
"""

import pytest
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


class TestBackwardCompatibility:
    """Test that existing code continues to work."""

    def test_initialization_with_defaults(self):
        """Test that AgenticStepProcessor can be initialized with default parameters."""
        processor = AgenticStepProcessor(
            objective="Test objective"
        )

        # Verify all new features are disabled by default
        assert processor.enable_blackboard is False
        assert processor.enable_cove is False
        assert processor.enable_checkpointing is False
        assert processor.enable_tao_loop is False
        assert processor.enable_dry_run is False

        # Verify blackboard and managers are not initialized
        assert processor.blackboard is None
        assert processor.cove_verifier is None
        assert processor.checkpoint_manager is None
        assert processor.dry_run_predictor is None

    def test_initialization_with_phase2_enabled(self):
        """Test Blackboard initialization."""
        processor = AgenticStepProcessor(
            objective="Test objective",
            enable_blackboard=True
        )

        assert processor.enable_blackboard is True
        assert processor.blackboard is not None
        assert processor.blackboard.objective == "Test objective"

    def test_initialization_with_phase3_enabled(self):
        """Test CoVe and Checkpointing initialization."""
        processor = AgenticStepProcessor(
            objective="Test objective",
            enable_cove=True,
            enable_checkpointing=True,
            cove_confidence_threshold=0.8
        )

        assert processor.enable_cove is True
        assert processor.enable_checkpointing is True
        assert processor.cove_confidence_threshold == 0.8
        # CoVe verifier is lazy-initialized
        assert processor.cove_verifier is None
        # Checkpoint manager should be initialized
        assert processor.checkpoint_manager is not None

    def test_initialization_with_phase4_enabled(self):
        """Test TAO Loop and Dry Run initialization."""
        processor = AgenticStepProcessor(
            objective="Test objective",
            enable_tao_loop=True,
            enable_dry_run=True
        )

        assert processor.enable_tao_loop is True
        assert processor.enable_dry_run is True
        # Dry run predictor is lazy-initialized
        assert processor.dry_run_predictor is None

    def test_initialization_with_all_features_enabled(self):
        """Test initialization with all features enabled."""
        processor = AgenticStepProcessor(
            objective="Complex research task",
            max_internal_steps=10,
            # Phase 2: Blackboard
            enable_blackboard=True,
            # Phase 3: Safety & Reliability
            enable_cove=True,
            enable_checkpointing=True,
            cove_confidence_threshold=0.7,
            # Phase 4: TAO Loop
            enable_tao_loop=True,
            enable_dry_run=True
        )

        # Verify all features are enabled
        assert processor.enable_blackboard is True
        assert processor.enable_cove is True
        assert processor.enable_checkpointing is True
        assert processor.enable_tao_loop is True
        assert processor.enable_dry_run is True

        # Verify components are initialized (or will be lazy-initialized)
        assert processor.blackboard is not None
        assert processor.checkpoint_manager is not None


class TestPhaseInteractions:
    """Test interactions between different phases."""

    def test_blackboard_with_checkpointing(self):
        """Test that Blackboard and Checkpointing work together."""
        processor = AgenticStepProcessor(
            objective="Test",
            enable_blackboard=True,
            enable_checkpointing=True
        )

        # Both should be initialized
        assert processor.blackboard is not None
        assert processor.checkpoint_manager is not None

        # Test snapshot creation
        snapshot_id = processor.blackboard.snapshot()
        assert snapshot_id == "snapshot_0"

        # Test checkpoint creation
        checkpoint_id = processor.checkpoint_manager.create_checkpoint(
            iteration=0,
            blackboard_snapshot=snapshot_id,
            confidence=0.9
        )
        assert checkpoint_id == "cp_0"

    def test_tao_with_blackboard(self):
        """Test that TAO Loop and Blackboard work together."""
        processor = AgenticStepProcessor(
            objective="Test",
            enable_blackboard=True,
            enable_tao_loop=True
        )

        # Both should be configured
        assert processor.enable_blackboard is True
        assert processor.enable_tao_loop is True
        assert processor.blackboard is not None

    def test_all_phases_together(self):
        """Test that all phases can be enabled simultaneously."""
        processor = AgenticStepProcessor(
            objective="Test",
            enable_blackboard=True,
            enable_cove=True,
            enable_checkpointing=True,
            enable_tao_loop=True,
            enable_dry_run=True
        )

        # All features enabled
        assert processor.enable_blackboard is True
        assert processor.enable_cove is True
        assert processor.enable_checkpointing is True
        assert processor.enable_tao_loop is True
        assert processor.enable_dry_run is True


class TestParameterValidation:
    """Test parameter validation and edge cases."""

    def test_confidence_threshold_bounds(self):
        """Test that confidence threshold is validated."""
        # Valid threshold
        processor = AgenticStepProcessor(
            objective="Test",
            enable_cove=True,
            cove_confidence_threshold=0.5
        )
        assert processor.cove_confidence_threshold == 0.5

        # Edge cases
        processor_min = AgenticStepProcessor(
            objective="Test",
            enable_cove=True,
            cove_confidence_threshold=0.0
        )
        assert processor_min.cove_confidence_threshold == 0.0

        processor_max = AgenticStepProcessor(
            objective="Test",
            enable_cove=True,
            cove_confidence_threshold=1.0
        )
        assert processor_max.cove_confidence_threshold == 1.0

    def test_blackboard_limits(self):
        """Test Blackboard size limits."""
        processor = AgenticStepProcessor(
            objective="Test",
            enable_blackboard=True
        )

        # Check default limits
        assert processor.blackboard.max_plan_items == 10
        assert processor.blackboard.max_facts == 20
        assert processor.blackboard.max_observations == 15

    def test_checkpoint_stuck_threshold(self):
        """Test checkpoint manager stuck threshold."""
        processor = AgenticStepProcessor(
            objective="Test",
            enable_checkpointing=True
        )

        # Default threshold should be 3
        assert processor.checkpoint_manager.stuck_threshold == 3


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
