"""Contract test for WorkflowState schema validation (T082).

This test verifies that the WorkflowState and WorkflowStep dataclasses include all
required fields for multi-session objective tracking and adhere to the schema
defined in specs/002-cli-orchestration/workflow-state.md.

Test Strategy:
- Validate required fields exist with correct types
- Test schema validation rules (objective length, step_index bounds, status literals)
- Verify serialization/deserialization preserves all workflow fields
- Test state transitions (pending → in_progress → completed/failed)
- Validate progress calculation and current step detection

RED Phase: Test should FAIL if workflow fields are missing or validation fails
GREEN Phase: Test should PASS after implementation ensures schema correctness
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from promptchain.cli.models.workflow import WorkflowState, WorkflowStep


class TestWorkflowStepSchemaContract:
    """Contract tests for WorkflowStep dataclass schema (T082)."""

    def test_workflow_step_has_required_fields(self):
        """Verify WorkflowStep has all required fields with correct types."""
        # Create minimal step
        step = WorkflowStep(description="Design authentication schema")

        # Required fields
        assert hasattr(step, "description")
        assert hasattr(step, "status")
        assert hasattr(step, "agent_name")
        assert hasattr(step, "started_at")
        assert hasattr(step, "completed_at")
        assert hasattr(step, "result")
        assert hasattr(step, "error_message")
        assert hasattr(step, "retry_count")

        # Verify types
        assert isinstance(step.description, str)
        assert step.status in ["pending", "in_progress", "completed", "failed"]
        assert step.agent_name is None or isinstance(step.agent_name, str)
        assert step.started_at is None or isinstance(step.started_at, float)
        assert step.completed_at is None or isinstance(step.completed_at, float)
        assert step.result is None or isinstance(step.result, str)
        assert step.error_message is None or isinstance(step.error_message, str)
        assert isinstance(step.retry_count, int)

    def test_workflow_step_default_values(self):
        """Verify WorkflowStep defaults match spec."""
        step = WorkflowStep(description="Test step")

        # Default values
        assert step.status == "pending"
        assert step.agent_name is None
        assert step.started_at is None
        assert step.completed_at is None
        assert step.result is None
        assert step.error_message is None
        assert step.retry_count == 0

    def test_workflow_step_valid_creation(self):
        """Validate minimal valid WorkflowStep creation."""
        step = WorkflowStep(description="Implement user authentication")

        assert step.description == "Implement user authentication"
        assert step.status == "pending"
        assert step.retry_count == 0

    def test_workflow_step_status_literal_validation(self):
        """Verify status field only accepts valid literal values."""
        # Valid statuses
        valid_statuses = ["pending", "in_progress", "completed", "failed"]
        for status in valid_statuses:
            step = WorkflowStep(description="Test", status=status)
            assert step.status == status

    def test_workflow_step_mark_in_progress_transition(self):
        """Verify WorkflowStep pending → in_progress transition."""
        step = WorkflowStep(description="Create database models")
        assert step.status == "pending"
        assert step.agent_name is None
        assert step.started_at is None

        # Mark in progress
        before_time = datetime.now().timestamp()
        step.mark_in_progress(agent_name="coder")
        after_time = datetime.now().timestamp()

        # Verify state changes
        assert step.status == "in_progress"
        assert step.agent_name == "coder"
        assert step.started_at is not None
        assert before_time <= step.started_at <= after_time

    def test_workflow_step_mark_completed_transition(self):
        """Verify WorkflowStep in_progress → completed transition."""
        step = WorkflowStep(description="Run tests")
        step.mark_in_progress(agent_name="test-runner")
        assert step.status == "in_progress"

        # Mark completed
        before_time = datetime.now().timestamp()
        step.mark_completed(result="All 42 tests passed")
        after_time = datetime.now().timestamp()

        # Verify state changes
        assert step.status == "completed"
        assert step.result == "All 42 tests passed"
        assert step.completed_at is not None
        assert before_time <= step.completed_at <= after_time
        assert step.error_message is None

    def test_workflow_step_mark_failed_transition(self):
        """Verify WorkflowStep in_progress → failed transition."""
        step = WorkflowStep(description="Deploy to production")
        step.mark_in_progress(agent_name="deployer")
        assert step.status == "in_progress"
        assert step.retry_count == 0

        # Mark failed
        before_time = datetime.now().timestamp()
        step.mark_failed(error="Database connection timeout")
        after_time = datetime.now().timestamp()

        # Verify state changes
        assert step.status == "failed"
        assert step.error_message == "Database connection timeout"
        assert step.completed_at is not None
        assert before_time <= step.completed_at <= after_time
        assert step.retry_count == 1

    def test_workflow_step_reset_transition(self):
        """Verify WorkflowStep failed → pending reset for retry."""
        step = WorkflowStep(description="Process data")
        step.mark_in_progress(agent_name="processor")
        step.mark_failed(error="Out of memory")

        assert step.status == "failed"
        assert step.retry_count == 1
        retry_count_before_reset = step.retry_count

        # Reset for retry
        step.reset()

        # Verify state reset (but retry_count preserved)
        assert step.status == "pending"
        assert step.started_at is None
        assert step.completed_at is None
        assert step.result is None
        assert step.error_message is None
        assert step.retry_count == retry_count_before_reset  # Preserved across reset

    def test_workflow_step_serialization_to_dict(self):
        """Verify WorkflowStep.to_dict() includes all fields."""
        step = WorkflowStep(
            description="Analyze code quality",
            status="in_progress",
            agent_name="analyzer",
            retry_count=2
        )
        step.started_at = 1234567890.0

        step_dict = step.to_dict()

        # Verify all fields present
        assert "description" in step_dict
        assert "status" in step_dict
        assert "agent_name" in step_dict
        assert "started_at" in step_dict
        assert "completed_at" in step_dict
        assert "result" in step_dict
        assert "error_message" in step_dict
        assert "retry_count" in step_dict

        # Verify values
        assert step_dict["description"] == "Analyze code quality"
        assert step_dict["status"] == "in_progress"
        assert step_dict["agent_name"] == "analyzer"
        assert step_dict["started_at"] == 1234567890.0
        assert step_dict["retry_count"] == 2

    def test_workflow_step_deserialization_from_dict(self):
        """Verify WorkflowStep.from_dict() round-trip integrity."""
        step_data: Dict[str, Any] = {
            "description": "Write documentation",
            "status": "completed",
            "agent_name": "writer",
            "started_at": 1234567890.0,
            "completed_at": 1234567950.0,
            "result": "Documentation updated with 15 new sections",
            "error_message": None,
            "retry_count": 0
        }

        step = WorkflowStep.from_dict(step_data)

        # Verify all fields reconstructed
        assert step.description == "Write documentation"
        assert step.status == "completed"
        assert step.agent_name == "writer"
        assert step.started_at == 1234567890.0
        assert step.completed_at == 1234567950.0
        assert step.result == "Documentation updated with 15 new sections"
        assert step.error_message is None
        assert step.retry_count == 0

    def test_workflow_step_roundtrip_serialization(self):
        """Verify WorkflowStep serialization roundtrip preserves all data."""
        original = WorkflowStep(
            description="Refactor authentication logic",
            status="failed",
            agent_name="refactorer",
            retry_count=3
        )
        original.started_at = 1234567890.0
        original.completed_at = 1234567920.0
        original.error_message = "Type mismatch in credentials"

        # Serialize to dict
        step_dict = original.to_dict()

        # Deserialize back
        restored = WorkflowStep.from_dict(step_dict)

        # Verify all fields match
        assert restored.description == original.description
        assert restored.status == original.status
        assert restored.agent_name == original.agent_name
        assert restored.started_at == original.started_at
        assert restored.completed_at == original.completed_at
        assert restored.result == original.result
        assert restored.error_message == original.error_message
        assert restored.retry_count == original.retry_count


class TestWorkflowStateSchemaContract:
    """Contract tests for WorkflowState dataclass schema (T082)."""

    def test_workflow_state_has_required_fields(self):
        """Verify WorkflowState has all required fields with correct types."""
        # Create minimal workflow
        workflow = WorkflowState(
            objective="Implement JWT authentication",
            steps=[
                WorkflowStep(description="Design schema"),
                WorkflowStep(description="Create models")
            ]
        )

        # Required fields
        assert hasattr(workflow, "objective")
        assert hasattr(workflow, "steps")
        assert hasattr(workflow, "current_step_index")
        assert hasattr(workflow, "created_at")
        assert hasattr(workflow, "updated_at")
        assert hasattr(workflow, "completed_at")
        assert hasattr(workflow, "metadata")

        # Verify types
        assert isinstance(workflow.objective, str)
        assert isinstance(workflow.steps, list)
        assert all(isinstance(step, WorkflowStep) for step in workflow.steps)
        assert isinstance(workflow.current_step_index, int)
        assert isinstance(workflow.created_at, float)
        assert isinstance(workflow.updated_at, float)
        assert workflow.completed_at is None or isinstance(workflow.completed_at, float)
        assert isinstance(workflow.metadata, dict)

    def test_workflow_state_valid_creation(self):
        """Validate minimal valid WorkflowState creation."""
        state = WorkflowState(
            objective="Implement authentication",
            steps=[
                WorkflowStep(description="Design schema", status="pending"),
                WorkflowStep(description="Create models", status="pending")
            ],
            current_step_index=0
        )

        assert state.objective == "Implement authentication"
        assert len(state.steps) == 2
        assert state.current_step_index == 0
        assert state.steps[0].description == "Design schema"
        assert state.steps[1].description == "Create models"

    def test_workflow_state_objective_length_validation(self):
        """Verify objective field validation (1-512 characters)."""
        # Valid: within bounds
        valid_objective = "Build a REST API with authentication"
        state = WorkflowState(objective=valid_objective, steps=[])
        assert state.objective == valid_objective

        # Invalid: empty objective
        with pytest.raises(ValueError, match="Objective must be 1-512 characters"):
            WorkflowState(objective="", steps=[])

        # Invalid: too long (>512 chars)
        long_objective = "x" * 513
        with pytest.raises(ValueError, match="Objective must be 1-512 characters"):
            WorkflowState(objective=long_objective, steps=[])

    def test_workflow_state_step_index_bounds_validation(self):
        """Verify current_step_index bounds validation."""
        steps = [
            WorkflowStep(description="Step 1"),
            WorkflowStep(description="Step 2"),
            WorkflowStep(description="Step 3")
        ]

        # Valid: index within range
        for i in range(len(steps)):
            state = WorkflowState(
                objective="Test workflow",
                steps=steps,
                current_step_index=i
            )
            assert state.current_step_index == i

        # Invalid: negative index
        with pytest.raises(ValueError, match="current_step_index .* out of range"):
            WorkflowState(
                objective="Test workflow",
                steps=steps,
                current_step_index=-1
            )

        # Invalid: index beyond steps length
        with pytest.raises(ValueError, match="current_step_index .* out of range"):
            WorkflowState(
                objective="Test workflow",
                steps=steps,
                current_step_index=3  # Out of bounds for 3 steps (0-2 valid)
            )

    def test_workflow_state_empty_steps_allows_index_zero(self):
        """Verify empty steps list allows current_step_index=0."""
        # Edge case: empty workflow should allow index 0
        state = WorkflowState(objective="Empty workflow", steps=[])
        assert state.current_step_index == 0
        assert len(state.steps) == 0

    def test_workflow_state_progress_calculation_empty(self):
        """Verify progress percentage for empty workflow (0%)."""
        state = WorkflowState(objective="Empty workflow", steps=[])
        assert state.progress_percentage == 0.0

    def test_workflow_state_progress_calculation_partial(self):
        """Verify progress percentage: 0/5 = 0%, 3/5 = 60%, 5/5 = 100%."""
        steps = [
            WorkflowStep(description="Step 1", status="pending"),
            WorkflowStep(description="Step 2", status="pending"),
            WorkflowStep(description="Step 3", status="pending"),
            WorkflowStep(description="Step 4", status="pending"),
            WorkflowStep(description="Step 5", status="pending")
        ]
        state = WorkflowState(objective="Test progress", steps=steps)

        # 0/5 completed = 0%
        assert state.progress_percentage == 0.0

        # 3/5 completed = 60%
        steps[0].status = "completed"
        steps[1].status = "completed"
        steps[2].status = "completed"
        assert state.progress_percentage == 60.0

        # 5/5 completed = 100%
        steps[3].status = "completed"
        steps[4].status = "completed"
        assert state.progress_percentage == 100.0

    def test_workflow_state_current_step_detection(self):
        """Verify current_step property points to correct step."""
        steps = [
            WorkflowStep(description="Step 1"),
            WorkflowStep(description="Step 2"),
            WorkflowStep(description="Step 3")
        ]
        state = WorkflowState(
            objective="Test current step",
            steps=steps,
            current_step_index=1  # Should point to "Step 2"
        )

        current = state.current_step
        assert current is not None
        assert current.description == "Step 2"
        assert current == steps[1]

    def test_workflow_state_current_step_none_when_complete(self):
        """Verify current_step returns None when workflow advanced beyond last step."""
        steps = [
            WorkflowStep(description="Step 1", status="completed"),
            WorkflowStep(description="Step 2", status="completed")
        ]
        state = WorkflowState(
            objective="Completed workflow",
            steps=steps,
            current_step_index=0
        )

        # Advance through all steps
        state.current_step_index = 1
        assert state.current_step is not None
        assert state.current_step.description == "Step 2"

        # Manually set index beyond last step (simulating workflow completion)
        # Note: advance_step() won't go beyond last index, so we test boundary behavior
        state.current_step_index = len(steps)

        # After all steps, current_step should be None
        assert state.current_step is None

    def test_workflow_state_is_completed_property(self):
        """Verify is_completed property when all steps are completed."""
        steps = [
            WorkflowStep(description="Step 1", status="pending"),
            WorkflowStep(description="Step 2", status="pending")
        ]
        state = WorkflowState(objective="Test completion", steps=steps)

        # Not completed yet
        assert state.is_completed is False

        # Complete first step
        steps[0].status = "completed"
        assert state.is_completed is False

        # Complete second step
        steps[1].status = "completed"
        assert state.is_completed is True

    def test_workflow_state_is_failed_property(self):
        """Verify is_failed property when any step fails."""
        steps = [
            WorkflowStep(description="Step 1", status="completed"),
            WorkflowStep(description="Step 2", status="in_progress"),
            WorkflowStep(description="Step 3", status="pending")
        ]
        state = WorkflowState(objective="Test failure", steps=steps)

        # No failures yet
        assert state.is_failed is False

        # Fail step 2
        steps[1].status = "failed"
        assert state.is_failed is True

    def test_workflow_state_add_step_method(self):
        """Verify add_step() adds new step and updates timestamp."""
        state = WorkflowState(objective="Growing workflow", steps=[])
        assert len(state.steps) == 0

        before_time = datetime.now().timestamp()
        added_step = state.add_step(description="New step", agent_name="coder")
        after_time = datetime.now().timestamp()

        # Verify step added
        assert len(state.steps) == 1
        assert state.steps[0] == added_step
        assert added_step.description == "New step"
        assert added_step.agent_name == "coder"

        # Verify updated_at timestamp updated
        assert before_time <= state.updated_at <= after_time

    def test_workflow_state_advance_step_method(self):
        """Verify advance_step() moves to next step."""
        steps = [
            WorkflowStep(description="Step 1"),
            WorkflowStep(description="Step 2"),
            WorkflowStep(description="Step 3")
        ]
        state = WorkflowState(
            objective="Sequential workflow",
            steps=steps,
            current_step_index=0
        )

        # Advance to step 2
        before_time = datetime.now().timestamp()
        state.advance_step()
        after_time = datetime.now().timestamp()

        assert state.current_step_index == 1
        assert state.current_step.description == "Step 2"
        assert before_time <= state.updated_at <= after_time

        # Advance to step 3
        state.advance_step()
        assert state.current_step_index == 2
        assert state.current_step.description == "Step 3"

        # Cannot advance beyond last step
        state.advance_step()
        assert state.current_step_index == 2  # Stays at last step

    def test_workflow_state_mark_completed_method(self):
        """Verify mark_completed() sets completion timestamp."""
        state = WorkflowState(
            objective="Finishable workflow",
            steps=[WorkflowStep(description="Only step", status="completed")]
        )
        assert state.completed_at is None

        before_time = datetime.now().timestamp()
        state.mark_completed()
        after_time = datetime.now().timestamp()

        # Verify completion timestamp set
        assert state.completed_at is not None
        assert before_time <= state.completed_at <= after_time
        assert state.updated_at == state.completed_at

    def test_workflow_state_serialization_to_dict(self):
        """Verify WorkflowState.to_dict() round-trip integrity."""
        state = WorkflowState(
            objective="Serialization test workflow",
            steps=[
                WorkflowStep(description="Step 1", status="completed"),
                WorkflowStep(description="Step 2", status="in_progress")
            ],
            current_step_index=1,
            metadata={"priority": "high", "tags": ["backend", "api"]}
        )

        state_dict = state.to_dict()

        # Verify all fields present
        assert "objective" in state_dict
        assert "steps" in state_dict
        assert "current_step_index" in state_dict
        assert "created_at" in state_dict
        assert "updated_at" in state_dict
        assert "completed_at" in state_dict
        assert "metadata" in state_dict

        # Verify values
        assert state_dict["objective"] == "Serialization test workflow"
        assert len(state_dict["steps"]) == 2
        assert state_dict["steps"][0]["description"] == "Step 1"
        assert state_dict["steps"][1]["description"] == "Step 2"
        assert state_dict["current_step_index"] == 1
        assert state_dict["metadata"]["priority"] == "high"

    def test_workflow_state_nested_serialization(self):
        """Verify WorkflowState.to_dict() with nested WorkflowSteps serialization."""
        state = WorkflowState(
            objective="Nested serialization test",
            steps=[
                WorkflowStep(description="Build API", status="completed", agent_name="coder"),
                WorkflowStep(description="Write tests", status="in_progress", agent_name="tester"),
                WorkflowStep(description="Deploy", status="pending")
            ],
            current_step_index=1
        )

        state_dict = state.to_dict()

        # Verify nested steps serialized
        assert isinstance(state_dict["steps"], list)
        assert len(state_dict["steps"]) == 3

        # Verify first step dict
        step1_dict = state_dict["steps"][0]
        assert step1_dict["description"] == "Build API"
        assert step1_dict["status"] == "completed"
        assert step1_dict["agent_name"] == "coder"

        # Verify second step dict
        step2_dict = state_dict["steps"][1]
        assert step2_dict["description"] == "Write tests"
        assert step2_dict["status"] == "in_progress"
        assert step2_dict["agent_name"] == "tester"

    def test_workflow_state_deserialization_from_dict(self):
        """Verify WorkflowState.from_dict() reconstructs workflow correctly."""
        state_data: Dict[str, Any] = {
            "objective": "Deserialization test workflow",
            "steps": [
                {
                    "description": "Analyze requirements",
                    "status": "completed",
                    "agent_name": "analyst",
                    "started_at": 1234567890.0,
                    "completed_at": 1234567950.0,
                    "result": "Requirements documented",
                    "error_message": None,
                    "retry_count": 0
                },
                {
                    "description": "Design architecture",
                    "status": "in_progress",
                    "agent_name": "architect",
                    "started_at": 1234567960.0,
                    "completed_at": None,
                    "result": None,
                    "error_message": None,
                    "retry_count": 0
                }
            ],
            "current_step_index": 1,
            "created_at": 1234567800.0,
            "updated_at": 1234567960.0,
            "completed_at": None,
            "metadata": {"project": "AuthService", "version": "1.0"}
        }

        state = WorkflowState.from_dict(state_data)

        # Verify all fields reconstructed
        assert state.objective == "Deserialization test workflow"
        assert len(state.steps) == 2
        assert state.current_step_index == 1
        assert state.created_at == 1234567800.0
        assert state.updated_at == 1234567960.0
        assert state.completed_at is None
        assert state.metadata["project"] == "AuthService"

        # Verify nested steps reconstructed
        step1 = state.steps[0]
        assert step1.description == "Analyze requirements"
        assert step1.status == "completed"
        assert step1.agent_name == "analyst"
        assert step1.result == "Requirements documented"

        step2 = state.steps[1]
        assert step2.description == "Design architecture"
        assert step2.status == "in_progress"
        assert step2.agent_name == "architect"

    def test_workflow_state_roundtrip_serialization(self):
        """Verify WorkflowState complete roundtrip serialization/deserialization."""
        original = WorkflowState(
            objective="Full roundtrip test workflow",
            steps=[
                WorkflowStep(description="Step A", status="completed", agent_name="agent1"),
                WorkflowStep(description="Step B", status="in_progress", agent_name="agent2"),
                WorkflowStep(description="Step C", status="pending")
            ],
            current_step_index=1,
            metadata={"env": "production", "region": "us-west"}
        )
        original.steps[0].started_at = 1234567890.0
        original.steps[0].completed_at = 1234567920.0
        original.steps[1].started_at = 1234567930.0

        # Serialize to dict
        state_dict = original.to_dict()

        # Deserialize back
        restored = WorkflowState.from_dict(state_dict)

        # Verify all fields match
        assert restored.objective == original.objective
        assert len(restored.steps) == len(original.steps)
        assert restored.current_step_index == original.current_step_index
        assert restored.metadata == original.metadata

        # Verify nested steps match
        for i in range(len(original.steps)):
            assert restored.steps[i].description == original.steps[i].description
            assert restored.steps[i].status == original.steps[i].status
            assert restored.steps[i].agent_name == original.steps[i].agent_name
            assert restored.steps[i].started_at == original.steps[i].started_at
            assert restored.steps[i].completed_at == original.steps[i].completed_at

    def test_workflow_state_string_representation(self):
        """Verify __str__() provides human-readable workflow status."""
        # In-progress workflow
        state = WorkflowState(
            objective="Test workflow representation",
            steps=[
                WorkflowStep(description="Step 1", status="completed"),
                WorkflowStep(description="Step 2", status="in_progress"),
                WorkflowStep(description="Step 3", status="pending")
            ],
            current_step_index=1
        )

        str_repr = str(state)

        # Verify key information present
        assert "Test workflow representation" in str_repr
        assert "33.3%" in str_repr  # 1/3 completed
        assert "Step 2/3" in str_repr
        assert "IN PROGRESS" in str_repr

        # Completed workflow
        for step in state.steps:
            step.status = "completed"
        state.mark_completed()

        str_repr = str(state)
        assert "100.0%" in str_repr
        assert "COMPLETED" in str_repr

        # Failed workflow
        state.steps[1].status = "failed"
        str_repr = str(state)
        assert "FAILED" in str_repr
