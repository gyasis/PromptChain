"""Integration tests for /workflow list command (T093).

Tests workflow listing across sessions with progress tracking.
"""

import pytest
import time
from pathlib import Path
from promptchain.cli.session_manager import SessionManager
from promptchain.cli.command_handler import CommandHandler
from promptchain.cli.models.workflow import WorkflowState, WorkflowStep


@pytest.fixture
def session_manager(tmp_path):
    """Create session manager with temporary database."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    db_path = sessions_dir / "test_sessions.db"
    return SessionManager(sessions_dir=sessions_dir, db_path=db_path)


@pytest.fixture
def command_handler(session_manager):
    """Create command handler with session manager."""
    return CommandHandler(session_manager)


def test_list_no_workflows(command_handler):
    """Test /workflow list with no workflows in database."""
    import asyncio
    result = asyncio.run(command_handler.handle_workflow_list())

    assert result.success is True
    assert "No workflows found" in result.message
    assert result.data["workflows"] == []
    assert result.data["count"] == 0


def test_list_single_workflow(session_manager, command_handler):
    """Test /workflow list with one active workflow."""
    # Create session with workflow
    session = session_manager.create_session(name="project_a")

    workflow = WorkflowState(
        objective="Implement user authentication",
        steps=[
            WorkflowStep(description="Design database schema"),
            WorkflowStep(description="Implement login endpoint"),
            WorkflowStep(description="Add JWT tokens"),
        ],
        current_step_index=0
    )

    session_manager.save_workflow(session.id, workflow)

    # List workflows
    import asyncio
    result = asyncio.run(command_handler.handle_workflow_list())

    assert result.success is True
    assert "All Workflows" in result.message
    assert "[project_a]" in result.message
    assert "Implement user authentication" in result.message
    assert "0%" in result.message  # No completed steps
    assert "🔄" in result.message  # Active workflow indicator


def test_list_multiple_workflows(session_manager, command_handler):
    """Test /workflow list with multiple workflows across sessions."""
    # Create first session with active workflow
    session1 = session_manager.create_session(name="project_a")
    workflow1 = WorkflowState(
        objective="Implement user authentication",
        steps=[
            WorkflowStep(description="Design schema", status="completed"),
            WorkflowStep(description="Implement endpoint"),
        ]
    )
    session_manager.save_workflow(session1.id, workflow1)

    # Create second session with completed workflow
    session2 = session_manager.create_session(name="project_b")
    workflow2 = WorkflowState(
        objective="Build dashboard",
        steps=[
            WorkflowStep(description="Create components", status="completed"),
            WorkflowStep(description="Add charts", status="completed"),
        ]
    )
    workflow2.mark_completed()
    session_manager.save_workflow(session2.id, workflow2)

    # List workflows
    import asyncio
    result = asyncio.run(command_handler.handle_workflow_list())

    assert result.success is True
    assert result.data["count"] == 2

    # Check both workflows appear in message
    assert "[project_a]" in result.message
    assert "[project_b]" in result.message
    assert "Implement user authentication" in result.message
    assert "Build dashboard" in result.message

    # Check one active, one complete
    assert "🔄" in result.message  # Active indicator
    assert "✅" in result.message  # Complete indicator


def test_list_ordering_most_recent_first(session_manager, command_handler):
    """Test workflows are ordered by creation date (most recent first)."""
    # Create first workflow (older)
    session1 = session_manager.create_session(name="old_project")
    workflow1 = WorkflowState(objective="Old task", steps=[])
    time.sleep(0.01)  # Ensure time difference
    session_manager.save_workflow(session1.id, workflow1)

    # Create second workflow (newer)
    session2 = session_manager.create_session(name="new_project")
    workflow2 = WorkflowState(objective="New task", steps=[])
    session_manager.save_workflow(session2.id, workflow2)

    # List workflows
    workflows = session_manager.list_all_workflows()

    # Most recent should be first
    assert workflows[0]["session_name"] == "new_project"
    assert workflows[1]["session_name"] == "old_project"


def test_list_progress_calculation(session_manager, command_handler):
    """Test progress percentage calculation is accurate."""
    session = session_manager.create_session(name="test_session")

    # Create workflow with 50% completion (2 out of 4 steps)
    workflow = WorkflowState(
        objective="Test workflow",
        steps=[
            WorkflowStep(description="Step 1", status="completed"),
            WorkflowStep(description="Step 2", status="completed"),
            WorkflowStep(description="Step 3", status="pending"),
            WorkflowStep(description="Step 4", status="pending"),
        ]
    )
    session_manager.save_workflow(session.id, workflow)

    # Get workflows
    workflows = session_manager.list_all_workflows()

    assert len(workflows) == 1
    assert workflows[0]["progress"] == 50.0
    assert workflows[0]["completed_count"] == 2
    assert workflows[0]["step_count"] == 4


def test_list_with_active_and_complete_mix(session_manager, command_handler):
    """Test listing workflows with mix of active and completed statuses."""
    # Create 3 sessions: 2 active, 1 complete
    session1 = session_manager.create_session(name="active_1")
    workflow1 = WorkflowState(
        objective="Active workflow 1",
        steps=[WorkflowStep(description="Step 1")]
    )
    session_manager.save_workflow(session1.id, workflow1)

    session2 = session_manager.create_session(name="active_2")
    workflow2 = WorkflowState(
        objective="Active workflow 2",
        steps=[WorkflowStep(description="Step 1")]
    )
    session_manager.save_workflow(session2.id, workflow2)

    session3 = session_manager.create_session(name="complete_1")
    workflow3 = WorkflowState(
        objective="Completed workflow",
        steps=[WorkflowStep(description="Step 1", status="completed")]
    )
    workflow3.mark_completed()
    session_manager.save_workflow(session3.id, workflow3)

    # List workflows
    import asyncio
    result = asyncio.run(command_handler.handle_workflow_list())

    assert result.success is True
    assert result.data["count"] == 3

    # Count status indicators in message
    active_count = result.message.count("🔄")
    complete_count = result.message.count("✅")

    assert active_count == 2  # 2 active workflows
    assert complete_count == 1  # 1 completed workflow


def test_list_empty_steps_workflow(session_manager, command_handler):
    """Test workflow with no steps shows 0% progress."""
    session = session_manager.create_session(name="test_session")

    workflow = WorkflowState(
        objective="Workflow without steps",
        steps=[]  # No steps defined
    )
    session_manager.save_workflow(session.id, workflow)

    workflows = session_manager.list_all_workflows()

    assert len(workflows) == 1
    assert workflows[0]["progress"] == 0.0
    assert workflows[0]["step_count"] == 0
    assert workflows[0]["completed_count"] == 0
