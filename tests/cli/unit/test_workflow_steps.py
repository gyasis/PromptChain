"""Unit tests for WorkflowStep state transitions and WorkflowState progress tracking.

Tests cover:
- Initial state verification
- Valid state transitions (pending → in_progress → completed/failed)
- Timestamp behavior and validation
- WorkflowState step advancement and progress calculation
- Edge cases and error handling
"""

import pytest
from datetime import datetime
from time import sleep
from promptchain.cli.models.workflow import WorkflowStep, WorkflowState


class TestWorkflowStepTransitions:
    """Unit tests for WorkflowStep state machine transitions."""

    def test_initial_state(self):
        """Unit: New step starts in pending state with all fields unset."""
        step = WorkflowStep(description="Test step")

        assert step.status == "pending"
        assert step.agent_name is None
        assert step.started_at is None
        assert step.completed_at is None
        assert step.result is None
        assert step.error_message is None
        assert step.retry_count == 0

    def test_pending_to_in_progress(self):
        """Unit: mark_in_progress() transitions from pending and sets timestamp."""
        step = WorkflowStep(description="Test step")
        start_time = datetime.now().timestamp()

        step.mark_in_progress(agent_name="test_agent")

        assert step.status == "in_progress"
        assert step.agent_name == "test_agent"
        assert step.started_at is not None
        assert step.started_at >= start_time
        assert step.completed_at is None

    def test_in_progress_to_completed(self):
        """Unit: mark_completed() transitions to completed and sets result."""
        step = WorkflowStep(description="Test step")
        step.mark_in_progress(agent_name="test_agent")

        # Small delay to ensure completed_at > started_at
        sleep(0.01)

        step.mark_completed(result="Successfully completed test")

        assert step.status == "completed"
        assert step.result == "Successfully completed test"
        assert step.completed_at is not None
        assert step.completed_at > step.started_at
        assert step.error_message is None  # Error cleared on success

    def test_in_progress_to_failed(self):
        """Unit: mark_failed() transitions to failed and sets error message."""
        step = WorkflowStep(description="Test step")
        step.mark_in_progress(agent_name="test_agent")

        # Small delay to ensure completed_at > started_at
        sleep(0.01)

        step.mark_failed(error="Test error occurred")

        assert step.status == "failed"
        assert step.error_message == "Test error occurred"
        assert step.completed_at is not None
        assert step.completed_at > step.started_at
        assert step.retry_count == 1

    def test_failed_step_increments_retry_count(self):
        """Unit: Each failure increments retry_count."""
        step = WorkflowStep(description="Test step")

        step.mark_in_progress(agent_name="agent1")
        step.mark_failed(error="First failure")
        assert step.retry_count == 1

        step.mark_in_progress(agent_name="agent2")
        step.mark_failed(error="Second failure")
        assert step.retry_count == 2

    def test_step_reset(self):
        """Unit: reset() returns step to pending state, preserving retry_count."""
        step = WorkflowStep(description="Test step")
        step.mark_in_progress(agent_name="test_agent")
        step.mark_failed(error="Test error")

        original_retry_count = step.retry_count
        step.reset()

        assert step.status == "pending"
        assert step.started_at is None
        assert step.completed_at is None
        assert step.result is None
        assert step.error_message is None
        assert step.retry_count == original_retry_count  # Preserved across reset

    def test_timestamp_ordering(self):
        """Unit: Timestamps maintain chronological order (started_at < completed_at)."""
        step = WorkflowStep(description="Test step")

        step.mark_in_progress(agent_name="test_agent")
        start_time = step.started_at

        sleep(0.02)  # Ensure measurable time difference

        step.mark_completed(result="Done")

        assert step.started_at == start_time  # Unchanged
        assert step.completed_at > step.started_at

    def test_step_serialization(self):
        """Unit: to_dict() and from_dict() preserve all state."""
        step = WorkflowStep(description="Test step")
        step.mark_in_progress(agent_name="test_agent")
        step.mark_completed(result="Test result")

        step_dict = step.to_dict()
        reconstructed = WorkflowStep.from_dict(step_dict)

        assert reconstructed.description == step.description
        assert reconstructed.status == step.status
        assert reconstructed.agent_name == step.agent_name
        assert reconstructed.started_at == step.started_at
        assert reconstructed.completed_at == step.completed_at
        assert reconstructed.result == step.result
        assert reconstructed.retry_count == step.retry_count

    def test_invalid_transition_pending_to_completed(self):
        """Unit: Attempting pending → completed should document current behavior.

        NOTE: Current implementation ALLOWS this transition (no state machine enforcement).
        This test documents the behavior for future enhancement with proper validation.
        When state machine validation is added, this test should expect an exception.
        """
        step = WorkflowStep(description="Test step")
        assert step.status == "pending"

        # Current behavior: mark_completed() works from any state
        step.mark_completed(result="Skipped to completed")

        # Documents current behavior (no validation)
        assert step.status == "completed"
        assert step.result == "Skipped to completed"
        assert step.completed_at is not None
        # Note: started_at remains None since mark_in_progress was never called

    def test_invalid_transition_completed_to_in_progress(self):
        """Unit: Attempting completed → in_progress should document current behavior.

        NOTE: Current implementation ALLOWS this transition (no state machine enforcement).
        This test documents the behavior for future enhancement with proper validation.
        When state machine validation is added, this test should expect an exception.
        """
        step = WorkflowStep(description="Test step")
        step.mark_in_progress(agent_name="agent1")
        step.mark_completed(result="First completion")

        assert step.status == "completed"
        original_completed_at = step.completed_at

        # Current behavior: mark_in_progress() works from any state
        sleep(0.01)
        step.mark_in_progress(agent_name="agent2")

        # Documents current behavior (no immutability enforcement)
        assert step.status == "in_progress"
        assert step.agent_name == "agent2"  # Overwrites previous agent
        assert step.started_at > original_completed_at  # Updates timestamp

    def test_state_immutability_after_completion(self):
        """Unit: Completed step should document current mutability behavior.

        NOTE: Current implementation allows modification after completion.
        This test documents the behavior for future enhancement with immutability.
        When immutability is enforced, subsequent mark_* calls should raise exceptions.
        """
        step = WorkflowStep(description="Test step")
        step.mark_in_progress(agent_name="agent1")
        step.mark_completed(result="Original result")

        assert step.status == "completed"
        original_result = step.result
        original_completed_at = step.completed_at

        # Current behavior: Can mark as failed after completion
        sleep(0.01)
        step.mark_failed(error="Modified after completion")

        # Documents current behavior (no immutability)
        assert step.status == "failed"  # Status changed from completed
        assert step.error_message == "Modified after completion"
        assert step.completed_at > original_completed_at  # Timestamp updated
        assert step.retry_count == 1  # Retry count incremented

    def test_state_immutability_after_failure(self):
        """Unit: Failed step should document current mutability behavior.

        NOTE: Current implementation allows modification after failure (except via reset()).
        This test documents the behavior for future enhancement with immutability.
        """
        step = WorkflowStep(description="Test step")
        step.mark_in_progress(agent_name="agent1")
        step.mark_failed(error="Original failure")

        assert step.status == "failed"
        original_error = step.error_message
        original_retry_count = step.retry_count

        # Current behavior: Can mark as completed after failure
        sleep(0.01)
        step.mark_completed(result="Modified after failure")

        # Documents current behavior (no immutability)
        assert step.status == "completed"  # Status changed from failed
        assert step.result == "Modified after failure"
        assert step.error_message is None  # Error cleared by mark_completed
        assert step.retry_count == original_retry_count  # Preserved

    def test_timestamp_immutability_on_state_change(self):
        """Unit: started_at timestamp should remain unchanged after initial setting.

        Tests that once started_at is set, it persists through state changes.
        """
        step = WorkflowStep(description="Test step")

        step.mark_in_progress(agent_name="agent1")
        original_started_at = step.started_at

        sleep(0.01)

        # Complete the step
        step.mark_completed(result="Done")
        assert step.started_at == original_started_at  # Unchanged

        # Mark as failed (documents current behavior allowing this)
        step.mark_failed(error="Later failure")
        assert step.started_at == original_started_at  # Still unchanged

        # Mark in progress again (documents current behavior)
        sleep(0.01)
        step.mark_in_progress(agent_name="agent2")

        # started_at is now updated (documents current behavior)
        assert step.started_at > original_started_at

    def test_error_message_cleared_on_successful_completion(self):
        """Unit: mark_completed() clears any previous error message."""
        step = WorkflowStep(description="Test step")
        step.mark_in_progress(agent_name="agent1")
        step.mark_failed(error="Initial failure")

        assert step.error_message == "Initial failure"

        # Reset and retry
        step.reset()
        step.mark_in_progress(agent_name="agent1")
        step.mark_completed(result="Success after retry")

        assert step.error_message is None
        assert step.result == "Success after retry"


class TestWorkflowStateProgress:
    """Unit tests for WorkflowState step advancement and progress tracking."""

    def test_workflow_initial_state(self):
        """Unit: New workflow starts with no steps and 0% progress."""
        workflow = WorkflowState(objective="Test workflow")

        assert workflow.objective == "Test workflow"
        assert len(workflow.steps) == 0
        assert workflow.current_step_index == 0
        assert workflow.progress_percentage == 0.0
        assert workflow.is_completed is False
        assert workflow.is_failed is False

    def test_add_step_updates_workflow(self):
        """Unit: add_step() creates new step and updates timestamp."""
        workflow = WorkflowState(objective="Test workflow")
        initial_updated_at = workflow.updated_at

        sleep(0.01)
        step = workflow.add_step(description="Step 1", agent_name="agent1")

        assert len(workflow.steps) == 1
        assert step.description == "Step 1"
        assert step.agent_name == "agent1"
        assert step.status == "pending"
        assert workflow.updated_at > initial_updated_at

    def test_step_advancement(self):
        """Unit: advance_step() increments current_step_index."""
        workflow = WorkflowState(objective="Test workflow")
        workflow.add_step("Step 1")
        workflow.add_step("Step 2")
        workflow.add_step("Step 3")

        assert workflow.current_step_index == 0
        assert workflow.current_step.description == "Step 1"

        workflow.advance_step()
        assert workflow.current_step_index == 1
        assert workflow.current_step.description == "Step 2"

        workflow.advance_step()
        assert workflow.current_step_index == 2
        assert workflow.current_step.description == "Step 3"

    def test_step_advancement_boundary(self):
        """Unit: advance_step() stops at last step."""
        workflow = WorkflowState(objective="Test workflow")
        workflow.add_step("Step 1")
        workflow.add_step("Step 2")

        workflow.advance_step()  # Now at index 1 (last step)
        workflow.advance_step()  # Should not advance beyond last step

        assert workflow.current_step_index == 1
        assert workflow.current_step.description == "Step 2"

    def test_current_step_returns_none_when_complete(self):
        """Unit: current_step returns None when index exceeds step count."""
        workflow = WorkflowState(objective="Test workflow")
        workflow.add_step("Only step")

        workflow.current_step_index = 1  # Beyond last step

        assert workflow.current_step is None

    def test_progress_calculation_partial(self):
        """Unit: progress_percentage reflects completed steps (2/4 = 50%)."""
        workflow = WorkflowState(objective="Test workflow")
        step1 = workflow.add_step("Step 1")
        step2 = workflow.add_step("Step 2")
        step3 = workflow.add_step("Step 3")
        step4 = workflow.add_step("Step 4")

        # Mark 2 steps as completed
        step1.mark_in_progress("agent1")
        step1.mark_completed("Done")
        step2.mark_in_progress("agent1")
        step2.mark_completed("Done")

        assert workflow.progress_percentage == 50.0

    def test_progress_calculation_all_completed(self):
        """Unit: progress_percentage = 100% when all steps completed."""
        workflow = WorkflowState(objective="Test workflow")
        step1 = workflow.add_step("Step 1")
        step2 = workflow.add_step("Step 2")

        step1.mark_in_progress("agent1")
        step1.mark_completed("Done 1")
        step2.mark_in_progress("agent1")
        step2.mark_completed("Done 2")

        assert workflow.progress_percentage == 100.0

    def test_progress_calculation_no_steps(self):
        """Unit: progress_percentage = 0% for workflow with no steps."""
        workflow = WorkflowState(objective="Test workflow")

        assert workflow.progress_percentage == 0.0

    def test_is_completed_true(self):
        """Unit: is_completed = True when all steps have completed status."""
        workflow = WorkflowState(objective="Test workflow")
        step1 = workflow.add_step("Step 1")
        step2 = workflow.add_step("Step 2")

        step1.mark_in_progress("agent1")
        step1.mark_completed("Done 1")
        step2.mark_in_progress("agent1")
        step2.mark_completed("Done 2")

        assert workflow.is_completed is True

    def test_is_completed_false_with_pending(self):
        """Unit: is_completed = False when any step is pending."""
        workflow = WorkflowState(objective="Test workflow")
        step1 = workflow.add_step("Step 1")
        step2 = workflow.add_step("Step 2")  # Still pending

        step1.mark_in_progress("agent1")
        step1.mark_completed("Done")

        assert workflow.is_completed is False

    def test_is_completed_false_with_no_steps(self):
        """Unit: is_completed = False for workflow with no steps."""
        workflow = WorkflowState(objective="Test workflow")

        assert workflow.is_completed is False

    def test_is_failed_true(self):
        """Unit: is_failed = True when any step has failed status."""
        workflow = WorkflowState(objective="Test workflow")
        step1 = workflow.add_step("Step 1")
        step2 = workflow.add_step("Step 2")

        step1.mark_in_progress("agent1")
        step1.mark_failed("Test error")

        assert workflow.is_failed is True

    def test_is_failed_false(self):
        """Unit: is_failed = False when no steps have failed."""
        workflow = WorkflowState(objective="Test workflow")
        step1 = workflow.add_step("Step 1")
        step2 = workflow.add_step("Step 2")

        step1.mark_in_progress("agent1")
        step1.mark_completed("Done")

        assert workflow.is_failed is False

    def test_mark_workflow_completed(self):
        """Unit: mark_completed() sets workflow completion timestamp."""
        workflow = WorkflowState(objective="Test workflow")
        initial_updated_at = workflow.updated_at

        assert workflow.completed_at is None

        sleep(0.01)
        workflow.mark_completed()

        assert workflow.completed_at is not None
        assert workflow.completed_at > initial_updated_at
        assert workflow.updated_at == workflow.completed_at

    def test_workflow_serialization(self):
        """Unit: to_dict() and from_dict() preserve complete workflow state."""
        workflow = WorkflowState(objective="Test workflow")
        step1 = workflow.add_step("Step 1", agent_name="agent1")
        step2 = workflow.add_step("Step 2", agent_name="agent2")

        step1.mark_in_progress("agent1")
        step1.mark_completed("Done")
        workflow.advance_step()

        workflow_dict = workflow.to_dict()
        reconstructed = WorkflowState.from_dict(workflow_dict)

        assert reconstructed.objective == workflow.objective
        assert len(reconstructed.steps) == len(workflow.steps)
        assert reconstructed.current_step_index == workflow.current_step_index
        assert reconstructed.created_at == workflow.created_at
        assert reconstructed.updated_at == workflow.updated_at
        assert reconstructed.steps[0].status == "completed"
        assert reconstructed.steps[1].status == "pending"

    def test_workflow_validation_rejects_invalid_objective(self):
        """Unit: __post_init__ validates objective length (1-512 chars)."""
        # Empty objective
        with pytest.raises(ValueError, match="Objective must be 1-512 characters"):
            WorkflowState(objective="")

        # Objective too long
        with pytest.raises(ValueError, match="Objective must be 1-512 characters"):
            WorkflowState(objective="x" * 513)

    def test_workflow_validation_rejects_invalid_step_index(self):
        """Unit: __post_init__ validates current_step_index is in range."""
        workflow = WorkflowState(objective="Test workflow")
        workflow.add_step("Step 1")
        workflow.add_step("Step 2")

        # Index 2 is out of range for 2 steps (valid: 0, 1)
        with pytest.raises(ValueError, match="current_step_index.*out of range"):
            WorkflowState(
                objective="Test workflow",
                steps=workflow.steps,
                current_step_index=2
            )

    def test_workflow_string_representation(self):
        """Unit: __str__ provides human-readable workflow summary."""
        workflow = WorkflowState(objective="Test workflow")
        step1 = workflow.add_step("Step 1")
        step2 = workflow.add_step("Step 2")

        step1.mark_in_progress("agent1")
        step1.mark_completed("Done")

        workflow_str = str(workflow)

        assert "Test workflow" in workflow_str
        assert "50.0%" in workflow_str  # Progress
        assert "Step 1/2" in workflow_str  # Current step
        assert "IN PROGRESS" in workflow_str  # Status
