"""Integration test for workflow step tracking and auto-update logic (T090 + T087).

This test verifies that the workflow tracking system:
1. Detects when agent outputs contain completion keywords
2. Automatically marks steps as completed
3. Advances workflow to next step
4. Updates database with new workflow state
5. Returns progress information for display

Test Strategy:
- Create workflow with multiple steps
- Simulate agent responses with completion keywords
- Verify step transitions and database persistence
- Validate progress tracking and visual feedback
"""

import pytest
from pathlib import Path
from unittest.mock import Mock

from promptchain.cli.session_manager import SessionManager
from promptchain.cli.models import Session, Message
from promptchain.cli.models.workflow import WorkflowState, WorkflowStep


@pytest.fixture
def temp_sessions_dir(tmp_path):
    """Create temporary sessions directory."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return sessions_dir


@pytest.fixture
def session_manager(temp_sessions_dir):
    """Create SessionManager with temporary directory."""
    return SessionManager(sessions_dir=temp_sessions_dir)


@pytest.fixture
def test_session(session_manager, tmp_path):
    """Create test session with workflow."""
    session = session_manager.create_session(
        name="test-workflow-tracking",
        working_directory=tmp_path
    )

    # Create workflow with 3 steps
    workflow = WorkflowState(
        objective="Build authentication system",
        steps=[
            WorkflowStep(description="Design database schema"),
            WorkflowStep(description="Create User model"),
            WorkflowStep(description="Implement login endpoint"),
        ],
        current_step_index=0
    )

    session_manager.save_workflow(session.id, workflow)
    return session


class TestWorkflowStepTracking:
    """Integration tests for workflow step tracking (T090 + T087)."""

    def test_detect_step_completion_with_keyword(self, session_manager, test_session):
        """Unit: detect_step_transition() identifies completion keywords.

        Given: Workflow with pending step
        When: Agent output contains "completed" keyword
        Then: Step is marked as completed
        And: Workflow advances to next step
        And: Database is updated
        """
        # Create agent message with completion keyword
        agent_output = "I have completed the database schema design. The users table includes email, password_hash, and created_at columns."

        # Detect step transition
        completed_step = session_manager.detect_step_transition(test_session, agent_output)

        # Verify step was detected and marked completed
        assert completed_step is not None
        assert completed_step.description == "Design database schema"
        assert completed_step.status == "completed"
        assert completed_step.result == agent_output[:200]  # First 200 chars
        assert completed_step.completed_at is not None

        # Verify workflow advanced
        updated_workflow = session_manager.load_workflow(test_session.id)
        assert updated_workflow.current_step_index == 1
        assert updated_workflow.steps[0].status == "completed"
        assert updated_workflow.steps[1].status == "pending"

    def test_detect_step_completion_with_done_keyword(self, session_manager, test_session):
        """Unit: detect_step_transition() recognizes 'done' as completion signal.

        Given: Workflow with pending step
        When: Agent output contains "done" keyword
        Then: Step is marked as completed
        """
        agent_output = "Done! The schema is ready for implementation."

        completed_step = session_manager.detect_step_transition(test_session, agent_output)

        assert completed_step is not None
        assert completed_step.status == "completed"

    def test_detect_step_completion_with_checkmark(self, session_manager, test_session):
        """Unit: detect_step_transition() recognizes checkmark emoji as completion.

        Given: Workflow with pending step
        When: Agent output contains "✅" or "✓"
        Then: Step is marked as completed
        """
        agent_output = "✅ Database schema design is complete"

        completed_step = session_manager.detect_step_transition(test_session, agent_output)

        assert completed_step is not None
        assert completed_step.status == "completed"

    def test_no_detection_without_keywords(self, session_manager, test_session):
        """Unit: detect_step_transition() returns None when no completion keywords.

        Given: Workflow with pending step
        When: Agent output has no completion keywords
        Then: No step transition detected
        And: Workflow state unchanged
        """
        agent_output = "I'm working on the database schema. Here are my initial thoughts..."

        completed_step = session_manager.detect_step_transition(test_session, agent_output)

        assert completed_step is None

        # Verify workflow unchanged
        workflow = session_manager.load_workflow(test_session.id)
        assert workflow.current_step_index == 0
        assert workflow.steps[0].status == "pending"

    def test_no_detection_when_no_workflow(self, session_manager, test_session):
        """Unit: detect_step_transition() returns None when no workflow exists.

        Given: Session with no active workflow
        When: Agent output contains completion keywords
        Then: No step transition detected
        """
        # Create session without workflow
        session_no_workflow = session_manager.create_session(
            name="no-workflow-session",
            working_directory=Path("/tmp")
        )

        agent_output = "Task completed successfully!"

        completed_step = session_manager.detect_step_transition(
            session_no_workflow, agent_output
        )

        assert completed_step is None

    def test_sequential_step_completion(self, session_manager, test_session):
        """Integration: Multiple steps complete in sequence.

        Given: Workflow with 3 steps
        When: Agent responses trigger 3 consecutive completions
        Then: All steps marked completed in order
        And: Progress advances from 0% to 100%
        """
        # Complete first step
        agent_output_1 = "Database schema completed with all required tables."
        step_1 = session_manager.detect_step_transition(test_session, agent_output_1)
        assert step_1.description == "Design database schema"

        workflow = session_manager.load_workflow(test_session.id)
        assert workflow.current_step_index == 1
        assert workflow.progress_percentage == pytest.approx(33.33, rel=0.1)

        # Complete second step
        agent_output_2 = "User model is done with password hashing implemented."
        step_2 = session_manager.detect_step_transition(test_session, agent_output_2)
        assert step_2.description == "Create User model"

        workflow = session_manager.load_workflow(test_session.id)
        assert workflow.current_step_index == 2
        assert workflow.progress_percentage == pytest.approx(66.67, rel=0.1)

        # Complete third step
        agent_output_3 = "✅ Login endpoint completed with JWT authentication."
        step_3 = session_manager.detect_step_transition(test_session, agent_output_3)
        assert step_3.description == "Implement login endpoint"

        workflow = session_manager.load_workflow(test_session.id)
        assert workflow.current_step_index == 2  # Stays at last step
        assert workflow.progress_percentage == 100.0

    def test_update_workflow_on_message_with_assistant_message(
        self, session_manager, test_session
    ):
        """Integration: update_workflow_on_message() processes assistant messages.

        Given: Workflow with pending step
        When: Assistant message contains completion keyword
        Then: Workflow is updated
        And: Progress info is returned for display
        """
        # Create assistant message with completion keyword
        assistant_msg = Message(
            role="assistant",
            content="I have successfully completed the database schema design."
        )

        # Update workflow on message
        progress_info = session_manager.update_workflow_on_message(
            test_session, assistant_msg
        )

        # Verify progress info returned
        assert progress_info is not None
        assert progress_info["step_description"] == "Design database schema"
        assert progress_info["progress_percentage"] == pytest.approx(33.33, rel=0.1)
        assert progress_info["completed_count"] == 1
        assert progress_info["total_steps"] == 3

    def test_update_workflow_on_message_ignores_user_messages(
        self, session_manager, test_session
    ):
        """Unit: update_workflow_on_message() ignores non-assistant messages.

        Given: Workflow with pending step
        When: User message contains completion keywords
        Then: No workflow update occurs
        And: None is returned
        """
        user_msg = Message(
            role="user",
            content="Please mark the task as completed when you're done."
        )

        progress_info = session_manager.update_workflow_on_message(
            test_session, user_msg
        )

        assert progress_info is None

        # Verify workflow unchanged
        workflow = session_manager.load_workflow(test_session.id)
        assert workflow.current_step_index == 0
        assert workflow.steps[0].status == "pending"

    def test_update_workflow_on_message_returns_none_when_no_completion(
        self, session_manager, test_session
    ):
        """Unit: update_workflow_on_message() returns None when no step completes.

        Given: Workflow with pending step
        When: Assistant message has no completion keywords
        Then: None is returned
        And: Workflow state unchanged
        """
        assistant_msg = Message(
            role="assistant",
            content="I'm still working on the database schema design."
        )

        progress_info = session_manager.update_workflow_on_message(
            test_session, assistant_msg
        )

        assert progress_info is None

    def test_case_insensitive_keyword_matching(self, session_manager, test_session):
        """Unit: Completion keywords are case-insensitive.

        Given: Workflow with pending step
        When: Agent output contains "COMPLETED", "Done", "FINISHED"
        Then: Step transitions are detected
        """
        # Test uppercase
        agent_output_1 = "COMPLETED the schema design."
        step_1 = session_manager.detect_step_transition(test_session, agent_output_1)
        assert step_1 is not None

        # Reload workflow and advance to next step
        workflow = session_manager.load_workflow(test_session.id)
        assert workflow.current_step_index == 1

        # Test mixed case
        agent_output_2 = "User model is Done!"
        step_2 = session_manager.detect_step_transition(test_session, agent_output_2)
        assert step_2 is not None

    def test_persistence_across_sessions(self, session_manager, test_session):
        """Integration: Workflow state persists across session reloads.

        Given: Workflow with completed steps
        When: Session is saved and reloaded
        Then: Workflow state is preserved
        And: Completed steps remain completed
        """
        # Complete first step
        agent_output = "Database schema design completed."
        session_manager.detect_step_transition(test_session, agent_output)

        # Save session
        session_manager.save_session(test_session)

        # Reload session
        reloaded_session = session_manager.load_session(test_session.id)

        # Load workflow from reloaded session
        workflow = session_manager.load_workflow(reloaded_session.id)

        # Verify state preserved
        assert workflow.current_step_index == 1
        assert workflow.steps[0].status == "completed"
        assert workflow.steps[0].result is not None
        assert workflow.steps[1].status == "pending"

    def test_result_truncation_to_200_chars(self, session_manager, test_session):
        """Unit: Step result is truncated to first 200 characters.

        Given: Agent output longer than 200 characters
        When: Step is marked completed
        Then: Result contains only first 200 characters
        """
        long_output = "Completed! " + ("x" * 300)  # 311 characters total

        completed_step = session_manager.detect_step_transition(test_session, long_output)

        assert completed_step is not None
        assert len(completed_step.result) == 200
        assert completed_step.result == long_output[:200]

    def test_multiple_keywords_in_message(self, session_manager, test_session):
        """Unit: Detection works when multiple keywords present.

        Given: Agent output with multiple completion keywords
        When: detect_step_transition() is called
        Then: Step is detected and marked completed
        """
        agent_output = "Task completed! ✅ I'm done with the schema design, it's finished now."

        completed_step = session_manager.detect_step_transition(test_session, agent_output)

        assert completed_step is not None
        assert completed_step.status == "completed"
