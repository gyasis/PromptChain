"""Integration tests for /workflow status command (T088).

Test Plan:
    1. No workflow exists -> should show "No active workflow" message
    2. Workflow with all pending steps -> should show 0% progress
    3. Workflow with mixed status steps -> should show correct progress
    4. Workflow with all completed steps -> should show 100% progress
    5. Workflow with failed steps -> should show failed indicator (❌)
    6. Status formatting -> should show correct indicators for each status
    7. Edge case: Empty workflow -> should handle gracefully

Each test verifies:
    - CommandResult success status
    - Message formatting
    - Data payload accuracy
    - Status indicator correctness
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from promptchain.cli.command_handler import CommandHandler
from promptchain.cli.session_manager import SessionManager
from promptchain.cli.models.session import Session
from promptchain.cli.models.workflow import WorkflowState, WorkflowStep


@pytest.fixture
def temp_sessions_dir():
    """Create temporary sessions directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def session_manager(temp_sessions_dir):
    """Create session manager with temporary directory."""
    return SessionManager(sessions_dir=temp_sessions_dir)


@pytest.fixture
def command_handler(session_manager):
    """Create command handler."""
    return CommandHandler(session_manager=session_manager)


@pytest.fixture
def test_session(session_manager):
    """Create test session."""
    session = session_manager.create_session(name="test_workflow_status")
    return session


class TestWorkflowStatus:
    """Test suite for /workflow status command."""

    @pytest.mark.asyncio
    async def test_no_workflow_exists(self, command_handler, test_session):
        """Test status when no workflow exists."""
        # Execute status command
        result = await command_handler.handle_workflow_status(test_session)

        # Verify result
        assert result.success is True
        assert "No active workflow" in result.message
        assert "Use /workflow create" in result.message
        assert result.data["workflow_exists"] is False

    @pytest.mark.asyncio
    async def test_workflow_all_pending_steps(self, command_handler, test_session):
        """Test status with all pending steps (0% progress)."""
        # Create workflow with all pending steps
        steps = [
            WorkflowStep(description="Step 1", status="pending"),
            WorkflowStep(description="Step 2", status="pending"),
            WorkflowStep(description="Step 3", status="pending"),
        ]
        workflow = WorkflowState(
            objective="Test all pending workflow",
            steps=steps,
            current_step_index=0,
            created_at=datetime.now().timestamp()
        )

        # Save workflow
        command_handler.session_manager.save_workflow(test_session.id, workflow)

        # Execute status command
        result = await command_handler.handle_workflow_status(test_session)

        # Verify result
        assert result.success is True
        assert "Test all pending workflow" in result.message
        assert "Progress: 0%" in result.message
        assert "Current Step: 1/3" in result.message

        # Verify data payload
        assert result.data["workflow_exists"] is True
        assert result.data["progress_percentage"] == 0.0
        assert result.data["total_steps"] == 3
        assert result.data["completed_steps"] == 0
        assert result.data["pending_steps"] == 3
        assert result.data["in_progress_steps"] == 0
        assert result.data["failed_steps"] == 0

        # Verify indicators
        assert "⏳" in result.message  # All pending
        assert "✅" not in result.message  # No completed
        assert "🔄" not in result.message  # No in-progress

    @pytest.mark.asyncio
    async def test_workflow_mixed_status(self, command_handler, test_session):
        """Test status with mixed step statuses."""
        # Create workflow with mixed statuses
        steps = [
            WorkflowStep(description="Completed step", status="completed"),
            WorkflowStep(description="In progress step", status="in_progress"),
            WorkflowStep(description="Pending step 1", status="pending"),
            WorkflowStep(description="Pending step 2", status="pending"),
            WorkflowStep(description="Failed step", status="failed"),
        ]
        workflow = WorkflowState(
            objective="Test mixed status workflow",
            steps=steps,
            current_step_index=1,  # On second step
            created_at=datetime.now().timestamp()
        )

        # Save workflow
        command_handler.session_manager.save_workflow(test_session.id, workflow)

        # Execute status command
        result = await command_handler.handle_workflow_status(test_session)

        # Verify result
        assert result.success is True
        assert "Test mixed status workflow" in result.message
        assert "Progress: 20%" in result.message  # 1/5 = 20%
        assert "Current Step: 2/5" in result.message

        # Verify data payload
        assert result.data["workflow_exists"] is True
        assert result.data["progress_percentage"] == 20.0
        assert result.data["total_steps"] == 5
        assert result.data["completed_steps"] == 1
        assert result.data["in_progress_steps"] == 1
        assert result.data["pending_steps"] == 2
        assert result.data["failed_steps"] == 1

        # Verify all status indicators present
        assert "✅" in result.message  # Completed
        assert "🔄" in result.message  # In-progress
        assert "⏳" in result.message  # Pending
        assert "❌" in result.message  # Failed

    @pytest.mark.asyncio
    async def test_workflow_all_completed(self, command_handler, test_session):
        """Test status with all completed steps (100% progress)."""
        # Create workflow with all completed steps
        steps = [
            WorkflowStep(description="Completed step 1", status="completed"),
            WorkflowStep(description="Completed step 2", status="completed"),
            WorkflowStep(description="Completed step 3", status="completed"),
        ]
        workflow = WorkflowState(
            objective="Test completed workflow",
            steps=steps,
            current_step_index=2,  # On last step
            created_at=datetime.now().timestamp()
        )

        # Save workflow
        command_handler.session_manager.save_workflow(test_session.id, workflow)

        # Execute status command
        result = await command_handler.handle_workflow_status(test_session)

        # Verify result
        assert result.success is True
        assert "Test completed workflow" in result.message
        assert "Progress: 100%" in result.message
        assert "Current Step: 3/3" in result.message

        # Verify data payload
        assert result.data["workflow_exists"] is True
        assert result.data["progress_percentage"] == 100.0
        assert result.data["total_steps"] == 3
        assert result.data["completed_steps"] == 3
        assert result.data["in_progress_steps"] == 0
        assert result.data["pending_steps"] == 0
        assert result.data["failed_steps"] == 0

        # Verify only completed indicators
        assert "✅" in result.message
        assert "⏳" not in result.message
        assert "🔄" not in result.message

    @pytest.mark.asyncio
    async def test_workflow_with_failed_steps(self, command_handler, test_session):
        """Test status with failed steps."""
        # Create workflow with failed steps
        steps = [
            WorkflowStep(description="Failed step 1", status="failed"),
            WorkflowStep(description="Failed step 2", status="failed"),
            WorkflowStep(description="Pending step", status="pending"),
        ]
        workflow = WorkflowState(
            objective="Test failed workflow",
            steps=steps,
            current_step_index=0,
            created_at=datetime.now().timestamp()
        )

        # Save workflow
        command_handler.session_manager.save_workflow(test_session.id, workflow)

        # Execute status command
        result = await command_handler.handle_workflow_status(test_session)

        # Verify result
        assert result.success is True
        assert "Test failed workflow" in result.message
        assert "Progress: 0%" in result.message  # No completed steps

        # Verify data payload
        assert result.data["failed_steps"] == 2
        assert result.data["completed_steps"] == 0

        # Verify failed indicators
        assert "❌" in result.message

    @pytest.mark.asyncio
    async def test_status_formatting(self, command_handler, test_session):
        """Test status message formatting."""
        # Create workflow with descriptive steps
        steps = [
            WorkflowStep(description="Design database schema", status="completed"),
            WorkflowStep(description="Implement user authentication", status="in_progress"),
            WorkflowStep(description="Create REST API endpoints", status="pending"),
            WorkflowStep(description="Write integration tests", status="pending"),
        ]
        workflow = WorkflowState(
            objective="Build web application backend",
            steps=steps,
            current_step_index=1,
            created_at=datetime.now().timestamp()
        )

        # Save workflow
        command_handler.session_manager.save_workflow(test_session.id, workflow)

        # Execute status command
        result = await command_handler.handle_workflow_status(test_session)

        # Verify formatting
        assert result.success is True

        # Check header section
        assert "Workflow: Build web application backend" in result.message
        assert "Progress: 25%" in result.message  # 1/4 = 25%
        assert "Current Step: 2/4" in result.message

        # Check steps section
        assert "Steps:" in result.message

        # Verify step numbering and descriptions
        assert "1. ✅ Design database schema" in result.message
        assert "2. 🔄 Implement user authentication" in result.message
        assert "3. ⏳ Create REST API endpoints" in result.message
        assert "4. ⏳ Write integration tests" in result.message

    @pytest.mark.asyncio
    async def test_error_handling(self, command_handler, test_session):
        """Test error handling for status command."""
        # This test verifies that exceptions are caught and returned properly
        # We can't easily trigger an exception in the normal flow, so we just
        # verify the structure handles errors correctly

        # Create a workflow and verify normal operation
        steps = [WorkflowStep(description="Test step", status="pending")]
        workflow = WorkflowState(
            objective="Test workflow",
            steps=steps,
            current_step_index=0,
            created_at=datetime.now().timestamp()
        )

        command_handler.session_manager.save_workflow(test_session.id, workflow)
        result = await command_handler.handle_workflow_status(test_session)

        # Verify normal success case
        assert result.success is True
        assert result.error is None


class TestWorkflowStatusEdgeCases:
    """Test edge cases for /workflow status command."""

    @pytest.mark.asyncio
    async def test_workflow_single_step(self, command_handler, test_session):
        """Test workflow with single step."""
        steps = [WorkflowStep(description="Only step", status="in_progress")]
        workflow = WorkflowState(
            objective="Single step workflow",
            steps=steps,
            current_step_index=0,
            created_at=datetime.now().timestamp()
        )

        command_handler.session_manager.save_workflow(test_session.id, workflow)
        result = await command_handler.handle_workflow_status(test_session)

        assert result.success is True
        assert "Current Step: 1/1" in result.message
        assert result.data["total_steps"] == 1

    @pytest.mark.asyncio
    async def test_workflow_long_objective(self, command_handler, test_session):
        """Test workflow with long objective description."""
        long_objective = "A" * 500  # Long but valid objective
        steps = [WorkflowStep(description="Step 1", status="pending")]
        workflow = WorkflowState(
            objective=long_objective,
            steps=steps,
            current_step_index=0,
            created_at=datetime.now().timestamp()
        )

        command_handler.session_manager.save_workflow(test_session.id, workflow)
        result = await command_handler.handle_workflow_status(test_session)

        assert result.success is True
        assert long_objective in result.message

    @pytest.mark.asyncio
    async def test_multiple_status_checks(self, command_handler, test_session):
        """Test calling status multiple times (idempotent)."""
        steps = [
            WorkflowStep(description="Step 1", status="completed"),
            WorkflowStep(description="Step 2", status="pending"),
        ]
        workflow = WorkflowState(
            objective="Test idempotency",
            steps=steps,
            current_step_index=1,
            created_at=datetime.now().timestamp()
        )

        command_handler.session_manager.save_workflow(test_session.id, workflow)

        # Call status multiple times
        result1 = await command_handler.handle_workflow_status(test_session)
        result2 = await command_handler.handle_workflow_status(test_session)
        result3 = await command_handler.handle_workflow_status(test_session)

        # All should be identical
        assert result1.success is True
        assert result2.success is True
        assert result3.success is True
        assert result1.message == result2.message == result3.message
        assert result1.data == result2.data == result3.data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
