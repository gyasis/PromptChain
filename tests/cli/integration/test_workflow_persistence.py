"""Integration test for workflow persistence across session restarts (T083).

This test suite verifies that workflow state persists correctly across session
save/load cycles using SQLite database storage. Tests include:

1. Workflow survives session save/load cycle
2. Workflow progress persists (step status, indices)
3. Workflow completion state persists
4. Multiple workflows in database don't interfere
5. Workflow metadata persists (timestamps, custom metadata)
6. Deleted workflow stays deleted after reload
7. Empty workflow handling (no workflow in session)

Test Strategy:
- Create workflows in sessions
- Save sessions to SQLite
- Load sessions from SQLite
- Verify WorkflowState fields persist correctly
- Test edge cases (empty, deleted, multiple sessions)
- Use temp directories for isolation
"""

import pytest
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

from promptchain.cli.session_manager import SessionManager
from promptchain.cli.models.workflow import WorkflowState, WorkflowStep


@pytest.fixture
def temp_sessions_dir():
    """Create temporary sessions directory for test isolation."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def session_manager(temp_sessions_dir):
    """Create SessionManager with temporary directory."""
    return SessionManager(sessions_dir=temp_sessions_dir)


class TestWorkflowPersistence:
    """Integration tests for workflow persistence across session restarts (T083)."""

    def test_workflow_survives_session_reload(self, session_manager):
        """Test: Workflow state persists across session save/load cycle.

        Given: Session with workflow containing objective and 3 steps
        When: Session is saved to SQLite and reloaded
        Then: Workflow objective, steps, and structure are preserved
        """
        # Create session with workflow
        session = session_manager.create_session("test-persist", Path.cwd())
        workflow = WorkflowState(
            objective="Test objective for persistence",
            steps=[
                WorkflowStep(description="Step 1: Initialize"),
                WorkflowStep(description="Step 2: Execute"),
                WorkflowStep(description="Step 3: Validate"),
            ],
        )
        session_manager.save_workflow(session.id, workflow)

        # Save session to disk
        session_manager.save_session(session)

        # Reload session from SQLite
        loaded_session = session_manager.load_session("test-persist")
        loaded_workflow = session_manager.load_workflow(loaded_session.id)

        # Verify workflow structure preserved
        assert loaded_workflow is not None
        assert loaded_workflow.objective == "Test objective for persistence"
        assert len(loaded_workflow.steps) == 3
        assert loaded_workflow.steps[0].description == "Step 1: Initialize"
        assert loaded_workflow.steps[1].description == "Step 2: Execute"
        assert loaded_workflow.steps[2].description == "Step 3: Validate"

    def test_workflow_progress_persists(self, session_manager):
        """Test: Workflow step progress persists across session restart.

        Given: Workflow with step 0 completed, step 1 in_progress, step 2 pending
        When: Session is saved and reloaded
        Then: current_step_index and step statuses are preserved
        """
        # Create session with partially completed workflow
        session = session_manager.create_session("test-progress", Path.cwd())
        workflow = WorkflowState(
            objective="Multi-step workflow with progress",
            steps=[
                WorkflowStep(description="Completed step"),
                WorkflowStep(description="In-progress step"),
                WorkflowStep(description="Pending step"),
            ],
            current_step_index=1,  # Currently on step 1
        )

        # Mark step 0 as completed
        workflow.steps[0].mark_completed(result="Step 1 done")
        # Mark step 1 as in_progress
        workflow.steps[1].mark_in_progress(agent_name="test-agent")

        session_manager.save_workflow(session.id, workflow)
        session_manager.save_session(session)

        # Reload and verify progress
        loaded_session = session_manager.load_session("test-progress")
        loaded_workflow = session_manager.load_workflow(loaded_session.id)

        assert loaded_workflow.current_step_index == 1
        assert loaded_workflow.steps[0].status == "completed"
        assert loaded_workflow.steps[0].result == "Step 1 done"
        assert loaded_workflow.steps[1].status == "in_progress"
        assert loaded_workflow.steps[1].agent_name == "test-agent"
        assert loaded_workflow.steps[2].status == "pending"

    def test_workflow_completion_state_persists(self, session_manager):
        """Test: Completed workflow state persists correctly.

        Given: Workflow with all steps completed
        When: Session is saved and reloaded
        Then: is_complete is True and progress is 100%
        """
        # Create fully completed workflow
        session = session_manager.create_session("test-complete", Path.cwd())
        workflow = WorkflowState(
            objective="Completed workflow",
            steps=[
                WorkflowStep(description="Step 1"),
                WorkflowStep(description="Step 2"),
                WorkflowStep(description="Step 3"),
            ],
        )

        # Mark all steps as completed
        for i, step in enumerate(workflow.steps):
            step.mark_completed(result=f"Step {i+1} result")

        workflow.mark_completed()  # Mark workflow as completed

        session_manager.save_workflow(session.id, workflow)
        session_manager.save_session(session)

        # Reload and verify completion
        loaded_session = session_manager.load_session("test-complete")
        loaded_workflow = session_manager.load_workflow(loaded_session.id)

        assert loaded_workflow.is_completed is True
        assert loaded_workflow.progress_percentage == 100.0
        assert loaded_workflow.completed_at is not None
        # Verify all steps completed
        for i, step in enumerate(loaded_workflow.steps):
            assert step.status == "completed"
            assert step.result == f"Step {i+1} result"

    def test_multiple_workflows_in_database(self, session_manager):
        """Test: Multiple sessions with workflows don't interfere.

        Given: 3 sessions with different workflows
        When: Each session is saved and reloaded
        Then: Each session loads its correct workflow without interference
        """
        # Create 3 sessions with different workflows
        workflows_data = [
            ("session-1", "Build API", ["Design", "Implement", "Test"]),
            ("session-2", "Write docs", ["Research", "Draft", "Review", "Publish"]),
            ("session-3", "Deploy app", ["Setup", "Configure"]),
        ]

        for session_name, objective, step_descriptions in workflows_data:
            session = session_manager.create_session(session_name, Path.cwd())
            workflow = WorkflowState(
                objective=objective,
                steps=[WorkflowStep(description=desc) for desc in step_descriptions],
            )
            session_manager.save_workflow(session.id, workflow)
            session_manager.save_session(session)

        # Reload each session and verify correct workflow
        for session_name, expected_objective, expected_steps in workflows_data:
            loaded_session = session_manager.load_session(session_name)
            loaded_workflow = session_manager.load_workflow(loaded_session.id)

            assert loaded_workflow is not None
            assert loaded_workflow.objective == expected_objective
            assert len(loaded_workflow.steps) == len(expected_steps)
            for i, expected_desc in enumerate(expected_steps):
                assert loaded_workflow.steps[i].description == expected_desc

    def test_workflow_metadata_persists(self, session_manager):
        """Test: Workflow metadata fields persist correctly.

        Given: Workflow with custom metadata and timestamps
        When: Session is saved and reloaded
        Then: created_at, updated_at, and custom metadata are preserved
        """
        # Create workflow with metadata
        session = session_manager.create_session("test-metadata", Path.cwd())
        workflow = WorkflowState(
            objective="Workflow with metadata",
            steps=[WorkflowStep(description="Test step")],
            metadata={
                "project": "PromptChain",
                "version": "1.0.0",
                "tags": ["testing", "persistence"],
            },
        )

        # Capture timestamps before save
        created_at = workflow.created_at
        updated_at = workflow.updated_at

        session_manager.save_workflow(session.id, workflow)
        session_manager.save_session(session)

        # Reload and verify metadata
        loaded_session = session_manager.load_session("test-metadata")
        loaded_workflow = session_manager.load_workflow(loaded_session.id)

        assert loaded_workflow.created_at == created_at
        assert loaded_workflow.updated_at == updated_at
        assert loaded_workflow.metadata == {
            "project": "PromptChain",
            "version": "1.0.0",
            "tags": ["testing", "persistence"],
        }

    def test_deleted_workflow_stays_deleted(self, session_manager):
        """Test: Deleted workflow remains deleted after session reload.

        Given: Session with workflow that is then deleted
        When: Session is saved and reloaded
        Then: load_workflow returns None (workflow is deleted)
        """
        # Create session with workflow
        session = session_manager.create_session("test-delete", Path.cwd())
        workflow = WorkflowState(
            objective="Workflow to be deleted",
            steps=[WorkflowStep(description="Some step")],
        )
        session_manager.save_workflow(session.id, workflow)
        session_manager.save_session(session)

        # Delete workflow from database
        import sqlite3

        conn = sqlite3.connect(session_manager.db_path)
        try:
            conn.execute("DELETE FROM workflow_states WHERE session_id = ?", (session.id,))
            conn.commit()
        finally:
            conn.close()

        # Save session again (without workflow)
        session_manager.save_session(session)

        # Reload and verify workflow is None
        loaded_session = session_manager.load_session("test-delete")
        loaded_workflow = session_manager.load_workflow(loaded_session.id)

        assert loaded_workflow is None

    def test_empty_workflow_handling(self, session_manager):
        """Test: Session with no workflow loads correctly.

        Given: Session created without a workflow
        When: Session is saved and reloaded
        Then: load_workflow returns None gracefully
        """
        # Create session without workflow
        session = session_manager.create_session("test-empty", Path.cwd())
        session_manager.save_session(session)

        # Reload and verify no workflow
        loaded_session = session_manager.load_session("test-empty")
        loaded_workflow = session_manager.load_workflow(loaded_session.id)

        assert loaded_workflow is None

    def test_workflow_step_timestamps_persist(self, session_manager):
        """Test: WorkflowStep timestamps persist correctly.

        Given: Workflow with steps having started_at and completed_at times
        When: Session is saved and reloaded
        Then: Step timestamps are preserved
        """
        # Create workflow with timestamped steps
        session = session_manager.create_session("test-timestamps", Path.cwd())
        workflow = WorkflowState(
            objective="Workflow with timestamped steps",
            steps=[
                WorkflowStep(description="Step 1"),
                WorkflowStep(description="Step 2"),
            ],
        )

        # Mark step 0 completed (has timestamps)
        workflow.steps[0].mark_completed(result="Done")
        # Mark step 1 in progress (has started_at)
        workflow.steps[1].mark_in_progress(agent_name="test-agent")

        # Capture timestamps
        step0_started = workflow.steps[0].started_at
        step0_completed = workflow.steps[0].completed_at
        step1_started = workflow.steps[1].started_at

        session_manager.save_workflow(session.id, workflow)
        session_manager.save_session(session)

        # Reload and verify timestamps
        loaded_session = session_manager.load_session("test-timestamps")
        loaded_workflow = session_manager.load_workflow(loaded_session.id)

        # Step 0 timestamps (completed step)
        assert loaded_workflow.steps[0].started_at == step0_started
        assert loaded_workflow.steps[0].completed_at == step0_completed

        # Step 1 timestamps (in-progress step)
        assert loaded_workflow.steps[1].started_at == step1_started
        assert loaded_workflow.steps[1].completed_at is None

    def test_workflow_error_state_persists(self, session_manager):
        """Test: Failed workflow step state persists.

        Given: Workflow with failed step containing error message
        When: Session is saved and reloaded
        Then: Failed status and error message are preserved
        """
        # Create workflow with failed step
        session = session_manager.create_session("test-error", Path.cwd())
        workflow = WorkflowState(
            objective="Workflow with error",
            steps=[
                WorkflowStep(description="Step that failed"),
                WorkflowStep(description="Pending step"),
            ],
        )

        # Mark step 0 as failed
        workflow.steps[0].mark_failed(error="Database connection timeout")

        session_manager.save_workflow(session.id, workflow)
        session_manager.save_session(session)

        # Reload and verify error state
        loaded_session = session_manager.load_session("test-error")
        loaded_workflow = session_manager.load_workflow(loaded_session.id)

        assert loaded_workflow.steps[0].status == "failed"
        assert loaded_workflow.steps[0].error_message == "Database connection timeout"
        assert loaded_workflow.steps[0].retry_count == 1
        assert loaded_workflow.is_failed is True

    def test_workflow_current_step_property_after_reload(self, session_manager):
        """Test: current_step property works after reload.

        Given: Workflow with current_step_index = 2
        When: Session is saved and reloaded
        Then: current_step property returns correct step
        """
        # Create workflow with specific current step
        session = session_manager.create_session("test-current-step", Path.cwd())
        workflow = WorkflowState(
            objective="Workflow with current step",
            steps=[
                WorkflowStep(description="Step 1"),
                WorkflowStep(description="Step 2"),
                WorkflowStep(description="Step 3"),
                WorkflowStep(description="Step 4"),
            ],
            current_step_index=2,  # Currently on step 3
        )

        session_manager.save_workflow(session.id, workflow)
        session_manager.save_session(session)

        # Reload and verify current_step property
        loaded_session = session_manager.load_session("test-current-step")
        loaded_workflow = session_manager.load_workflow(loaded_session.id)

        assert loaded_workflow.current_step_index == 2
        assert loaded_workflow.current_step is not None
        assert loaded_workflow.current_step.description == "Step 3"
