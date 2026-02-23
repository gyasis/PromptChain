"""Integration tests for /workflow resume command (T089, T084).

Tests verify workflow resume functionality including:
- No workflow scenario
- Active workflow scenario
- Completed workflow scenario
- Message injection into session (T084)
- Resume message format (T084)
- System message context restoration (T084)
- Multiple resume scenarios (T084)
- Session persistence (T084)
- Timestamp validation (T084)

Total: 17 tests (8 from T089, 9 from T084)
"""

import asyncio
import sqlite3
import tempfile
from pathlib import Path

import pytest

from promptchain.cli.command_handler import CommandHandler
from promptchain.cli.models import Session
from promptchain.cli.models.workflow import WorkflowState, WorkflowStep
from promptchain.cli.session_manager import SessionManager


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sessions.db"
        sessions_dir = Path(tmpdir) / "sessions"
        yield db_path, sessions_dir


@pytest.fixture
def session_manager(temp_db):
    """Create session manager with temporary database."""
    db_path, sessions_dir = temp_db
    return SessionManager(sessions_dir=sessions_dir, db_path=db_path)


@pytest.fixture
def test_session(session_manager, tmp_path):
    """Create test session."""
    session = session_manager.create_session(
        name="test-workflow-resume",
        working_directory=tmp_path,
        default_model="openai/gpt-4"
    )
    return session


@pytest.fixture
def command_handler(session_manager):
    """Create command handler."""
    return CommandHandler(session_manager)


# ===== Test 1: Resume with No Workflow =====

@pytest.mark.asyncio
async def test_resume_no_workflow(command_handler, test_session):
    """Test /workflow resume when no workflow exists."""
    result = await command_handler.handle_workflow_resume(test_session)

    assert result.success is False
    assert "No workflow to resume" in result.message
    assert "Use /workflow create" in result.message
    assert result.error == "No workflow found"


# ===== Test 2: Resume with Active Workflow (No Steps Completed) =====

@pytest.mark.asyncio
async def test_resume_active_workflow_no_completed_steps(
    command_handler, session_manager, test_session
):
    """Test /workflow resume with active workflow at step 0."""
    # Create workflow with 5 steps (all pending)
    steps = [
        WorkflowStep(description="Step 1: Design database schema", status="pending"),
        WorkflowStep(description="Step 2: Create models", status="pending"),
        WorkflowStep(description="Step 3: Write tests", status="pending"),
        WorkflowStep(description="Step 4: Implement logic", status="pending"),
        WorkflowStep(description="Step 5: Deploy", status="pending"),
    ]

    workflow = WorkflowState(
        objective="Build authentication system",
        steps=steps,
        current_step_index=0
    )

    session_manager.save_workflow(test_session.id, workflow)

    # Resume workflow
    result = await command_handler.handle_workflow_resume(test_session)

    assert result.success is True
    assert "Resuming workflow: Build authentication system" in result.message
    assert "Progress: 0% (0/5 steps)" in result.message
    assert "Current step: Step 1: Design database schema" in result.message
    assert "Let's continue with: Step 1: Design database schema" in result.message

    # Verify data
    assert result.data["current_step"]["description"] == "Step 1: Design database schema"
    assert result.data["completed_count"] == 0
    assert result.data["workflow"]["current_step_index"] == 0


# ===== Test 3: Resume with Active Workflow (Some Steps Completed) =====

@pytest.mark.asyncio
async def test_resume_active_workflow_partial_completion(
    command_handler, session_manager, test_session
):
    """Test /workflow resume with partially completed workflow."""
    # Create workflow with 5 steps (2 completed, 1 in progress, 2 pending)
    steps = [
        WorkflowStep(description="Step 1: Design database schema", status="completed"),
        WorkflowStep(description="Step 2: Create models", status="completed"),
        WorkflowStep(description="Step 3: Write tests", status="pending"),
        WorkflowStep(description="Step 4: Implement logic", status="pending"),
        WorkflowStep(description="Step 5: Deploy", status="pending"),
    ]

    workflow = WorkflowState(
        objective="Build authentication system",
        steps=steps,
        current_step_index=2  # Currently at step 3
    )

    session_manager.save_workflow(test_session.id, workflow)

    # Resume workflow
    result = await command_handler.handle_workflow_resume(test_session)

    assert result.success is True
    assert "Resuming workflow: Build authentication system" in result.message
    assert "Progress: 40% (2/5 steps)" in result.message
    assert "Current step: Step 3: Write tests" in result.message

    # Verify completed steps listed
    assert "✅ Step 1: Design database schema" in result.message
    assert "✅ Step 2: Create models" in result.message

    # Verify data
    assert result.data["current_step"]["description"] == "Step 3: Write tests"
    assert result.data["completed_count"] == 2
    assert result.data["workflow"]["current_step_index"] == 2


# ===== Test 4: Resume with Completed Workflow =====

@pytest.mark.asyncio
async def test_resume_completed_workflow(
    command_handler, session_manager, test_session
):
    """Test /workflow resume when workflow is already completed."""
    # Create workflow with all steps completed
    steps = [
        WorkflowStep(description="Step 1: Design database schema", status="completed"),
        WorkflowStep(description="Step 2: Create models", status="completed"),
        WorkflowStep(description="Step 3: Write tests", status="completed"),
    ]

    workflow = WorkflowState(
        objective="Build authentication system",
        steps=steps,
        current_step_index=2  # Last valid index (all steps completed)
    )

    session_manager.save_workflow(test_session.id, workflow)

    # Resume workflow
    result = await command_handler.handle_workflow_resume(test_session)

    assert result.success is True
    assert "Workflow already complete" in result.message
    assert "All 3 steps finished" in result.message

    # Verify workflow data included
    assert result.data["workflow"]["objective"] == "Build authentication system"
    assert len(result.data["workflow"]["steps"]) == 3


# ===== Test 5: Verify Message Injection =====

@pytest.mark.asyncio
async def test_resume_message_injection(
    command_handler, session_manager, test_session
):
    """Test that resume injects system message into session."""
    # Create workflow
    steps = [
        WorkflowStep(description="Step 1: Design database schema", status="completed"),
        WorkflowStep(description="Step 2: Create models", status="pending"),
    ]

    workflow = WorkflowState(
        objective="Build authentication system",
        steps=steps,
        current_step_index=1
    )

    session_manager.save_workflow(test_session.id, workflow)

    # Record initial message count
    initial_message_count = len(test_session.messages)

    # Resume workflow
    result = await command_handler.handle_workflow_resume(test_session)

    assert result.success is True

    # Verify message was added
    assert len(test_session.messages) == initial_message_count + 1

    # Verify message content and role
    injected_message = test_session.messages[-1]
    assert injected_message.role == "system"
    assert "Resuming workflow: Build authentication system" in injected_message.content
    assert "Current step: Step 2: Create models" in injected_message.content


# ===== Test 6: Verify Message Format =====

@pytest.mark.asyncio
async def test_resume_message_format(
    command_handler, session_manager, test_session
):
    """Test that resume message contains all required information."""
    # Create workflow with specific steps
    steps = [
        WorkflowStep(description="Configure database connection", status="completed"),
        WorkflowStep(description="Create user model", status="completed"),
        WorkflowStep(description="Implement password hashing", status="pending"),
        WorkflowStep(description="Add JWT token generation", status="pending"),
    ]

    workflow = WorkflowState(
        objective="User authentication with JWT",
        steps=steps,
        current_step_index=2
    )

    session_manager.save_workflow(test_session.id, workflow)

    # Resume workflow
    result = await command_handler.handle_workflow_resume(test_session)

    assert result.success is True

    # Verify message structure
    message = result.message

    # Should contain objective
    assert "Resuming workflow: User authentication with JWT" in message

    # Should contain progress
    assert "Progress: 50% (2/4 steps)" in message

    # Should contain current step
    assert "Current step: Implement password hashing" in message

    # Should list completed steps
    assert "Completed steps:" in message
    assert "✅ Configure database connection" in message
    assert "✅ Create user model" in message

    # Should have continuation prompt
    assert "Let's continue with: Implement password hashing" in message


# ===== Test 7: Error Handling =====

@pytest.mark.asyncio
async def test_resume_error_handling(command_handler, test_session):
    """Test error handling when workflow load fails."""
    # Create invalid session (no database entry)
    invalid_session = Session(
        id="invalid-session-id",
        name="invalid",
        created_at=0.0,
        last_accessed=0.0,
        working_directory=Path.cwd()
    )

    # Try to resume with invalid session
    result = await command_handler.handle_workflow_resume(invalid_session)

    # Should handle gracefully (no workflow found)
    assert result.success is False
    assert "No workflow to resume" in result.message


# ===== Test 8: Resume Multiple Times =====

@pytest.mark.asyncio
async def test_resume_multiple_times(
    command_handler, session_manager, test_session
):
    """Test that resume can be called multiple times safely."""
    # Create workflow
    steps = [
        WorkflowStep(description="Step 1", status="completed"),
        WorkflowStep(description="Step 2", status="pending"),
    ]

    workflow = WorkflowState(
        objective="Test objective",
        steps=steps,
        current_step_index=1
    )

    session_manager.save_workflow(test_session.id, workflow)

    # Resume multiple times
    result1 = await command_handler.handle_workflow_resume(test_session)
    result2 = await command_handler.handle_workflow_resume(test_session)
    result3 = await command_handler.handle_workflow_resume(test_session)

    # All should succeed
    assert result1.success is True
    assert result2.success is True
    assert result3.success is True

    # Verify each resume injected a message
    system_messages = [m for m in test_session.messages if m.role == "system"]
    assert len(system_messages) == 3

    # All messages should have same content (workflow state unchanged)
    assert system_messages[0].content == system_messages[1].content
    assert system_messages[1].content == system_messages[2].content


# ===== Test 9: System Message Injection Verification =====

@pytest.mark.asyncio
async def test_system_message_injection_contains_objective(
    command_handler, session_manager, test_session
):
    """Test that injected system message contains workflow objective (T084)."""
    # Create workflow with 2 steps, mark first complete
    steps = [
        WorkflowStep(description="Initialize database", status="completed"),
        WorkflowStep(description="Create tables", status="pending"),
    ]

    workflow = WorkflowState(
        objective="Database migration workflow",
        steps=steps,
        current_step_index=1
    )

    session_manager.save_workflow(test_session.id, workflow)

    # Resume workflow
    result = await command_handler.handle_workflow_resume(test_session)

    assert result.success is True

    # Verify system message added
    system_messages = [m for m in test_session.messages if m.role == "system"]
    assert len(system_messages) == 1

    system_msg = system_messages[0]
    assert system_msg.role == "system"
    assert "Database migration workflow" in system_msg.content
    assert "Resuming workflow:" in system_msg.content


# ===== Test 10: Resume Message Full Context Validation =====

@pytest.mark.asyncio
async def test_resume_message_contains_all_context_elements(
    command_handler, session_manager, test_session
):
    """Test that resume message contains all required context elements (T084)."""
    # Create workflow with some steps completed
    steps = [
        WorkflowStep(description="Setup environment", status="completed"),
        WorkflowStep(description="Install dependencies", status="completed"),
        WorkflowStep(description="Configure settings", status="pending"),
        WorkflowStep(description="Run tests", status="pending"),
    ]

    workflow = WorkflowState(
        objective="Project initialization",
        steps=steps,
        current_step_index=2
    )

    session_manager.save_workflow(test_session.id, workflow)

    # Resume workflow
    result = await command_handler.handle_workflow_resume(test_session)

    assert result.success is True

    message = result.message

    # Verify all required elements present
    assert "Resuming workflow: Project initialization" in message  # Objective
    assert "Progress: 50% (2/4 steps)" in message  # Progress percentage
    assert "Current step: Configure settings" in message  # Current step
    assert "Completed steps:" in message  # Completed steps section
    assert "✅ Setup environment" in message  # First completed step
    assert "✅ Install dependencies" in message  # Second completed step


# ===== Test 11: Resume After Multiple Completions =====

@pytest.mark.asyncio
async def test_resume_after_multiple_step_completions(
    command_handler, session_manager, test_session
):
    """Test resume with multiple completed steps (T084)."""
    # Create workflow with 5 steps, mark first 3 complete
    steps = [
        WorkflowStep(description="Design API endpoints", status="completed"),
        WorkflowStep(description="Implement models", status="completed"),
        WorkflowStep(description="Write unit tests", status="completed"),
        WorkflowStep(description="Integration tests", status="pending"),
        WorkflowStep(description="Deploy to staging", status="pending"),
    ]

    workflow = WorkflowState(
        objective="API development workflow",
        steps=steps,
        current_step_index=3  # At step 4 (0-indexed)
    )

    session_manager.save_workflow(test_session.id, workflow)

    # Resume workflow
    result = await command_handler.handle_workflow_resume(test_session)

    assert result.success is True

    # Verify system message lists all 3 completed steps
    system_msg = test_session.messages[-1]
    assert system_msg.role == "system"
    assert "✅ Design API endpoints" in system_msg.content
    assert "✅ Implement models" in system_msg.content
    assert "✅ Write unit tests" in system_msg.content

    # Verify current step is step 4
    assert "Current step: Integration tests" in system_msg.content

    # Verify progress calculation
    assert "Progress: 60% (3/5 steps)" in system_msg.content


# ===== Test 12: Resume Message Format Validation =====

@pytest.mark.asyncio
async def test_resume_message_format_structure(
    command_handler, session_manager, test_session
):
    """Test resume message format matches expected structure (T084)."""
    # Create simple workflow
    steps = [
        WorkflowStep(description="Step A", status="completed"),
        WorkflowStep(description="Step B", status="pending"),
    ]

    workflow = WorkflowState(
        objective="Format validation workflow",
        steps=steps,
        current_step_index=1
    )

    session_manager.save_workflow(test_session.id, workflow)

    # Resume workflow
    result = await command_handler.handle_workflow_resume(test_session)

    assert result.success is True

    message = result.message

    # Check all required format sections
    assert "Resuming workflow:" in message
    assert "Progress:" in message
    assert "Current step:" in message
    assert "Completed steps:" in message
    assert "Let's continue with:" in message

    # Verify structure order (sections appear in expected sequence)
    resuming_idx = message.index("Resuming workflow:")
    progress_idx = message.index("Progress:")
    current_idx = message.index("Current step:")
    completed_idx = message.index("Completed steps:")
    continue_idx = message.index("Let's continue with:")

    assert resuming_idx < progress_idx < current_idx < completed_idx < continue_idx


# ===== Test 13: Session Message Count Increases =====

@pytest.mark.asyncio
async def test_resume_increases_message_count(
    command_handler, session_manager, test_session
):
    """Test that resume increases session message count by exactly 1 (T084)."""
    # Create workflow
    steps = [
        WorkflowStep(description="First step", status="completed"),
        WorkflowStep(description="Second step", status="pending"),
    ]

    workflow = WorkflowState(
        objective="Message count test",
        steps=steps,
        current_step_index=1
    )

    session_manager.save_workflow(test_session.id, workflow)

    # Record initial state
    initial_count = len(test_session.messages)
    initial_messages = list(test_session.messages)

    # Resume workflow
    result = await command_handler.handle_workflow_resume(test_session)

    assert result.success is True

    # Verify count increased by exactly 1
    assert len(test_session.messages) == initial_count + 1

    # Verify new message is at end
    new_message = test_session.messages[-1]
    assert new_message not in initial_messages
    assert new_message.role == "system"


# ===== Test 14: Resume Multiple Times Adds Multiple Messages =====

@pytest.mark.asyncio
async def test_resume_multiple_times_adds_system_messages(
    command_handler, session_manager, test_session
):
    """Test that each resume adds a new system message (T084)."""
    # Create workflow
    steps = [
        WorkflowStep(description="Task 1", status="completed"),
        WorkflowStep(description="Task 2", status="pending"),
    ]

    workflow = WorkflowState(
        objective="Multiple resume test",
        steps=steps,
        current_step_index=1
    )

    session_manager.save_workflow(test_session.id, workflow)

    initial_count = len(test_session.messages)

    # Resume multiple times
    await command_handler.handle_workflow_resume(test_session)
    await command_handler.handle_workflow_resume(test_session)
    await command_handler.handle_workflow_resume(test_session)

    # Verify 3 messages added
    assert len(test_session.messages) == initial_count + 3

    # Verify all are system messages
    new_messages = test_session.messages[initial_count:]
    assert all(m.role == "system" for m in new_messages)

    # Verify all contain workflow context
    for msg in new_messages:
        assert "Multiple resume test" in msg.content
        assert "Task 2" in msg.content


# ===== Test 15: Resume Context Survives Save/Load =====

@pytest.mark.asyncio
async def test_resume_context_survives_session_persistence(
    session_manager, command_handler, temp_db
):
    """Test that system message persists after save/load (T084)."""
    db_path, sessions_dir = temp_db

    # Create initial session
    session = session_manager.create_session(
        name="persist-test",
        working_directory=Path.cwd(),
        default_model="openai/gpt-4"
    )

    # Create workflow
    steps = [
        WorkflowStep(description="Persist step 1", status="completed"),
        WorkflowStep(description="Persist step 2", status="pending"),
    ]

    workflow = WorkflowState(
        objective="Persistence test workflow",
        steps=steps,
        current_step_index=1
    )

    session_manager.save_workflow(session.id, workflow)

    # Resume workflow (adds system message)
    result = await command_handler.handle_workflow_resume(session)
    assert result.success is True

    # Verify message added
    original_message_count = len(session.messages)
    assert original_message_count > 0

    system_message = session.messages[-1]
    assert system_message.role == "system"
    assert "Persistence test workflow" in system_message.content

    # Save session to disk
    session_manager.save_session(session)

    # Create new session manager (simulates new process)
    new_session_manager = SessionManager(sessions_dir=sessions_dir, db_path=db_path)

    # Load session
    loaded_session = new_session_manager.load_session(session.id)

    # Verify message count matches
    assert len(loaded_session.messages) == original_message_count

    # Verify system message still present
    loaded_system_message = loaded_session.messages[-1]
    assert loaded_system_message.role == "system"
    assert "Persistence test workflow" in loaded_system_message.content
    assert loaded_system_message.content == system_message.content


# ===== Test 16: Resume with No Completed Steps Still Adds System Message =====

@pytest.mark.asyncio
async def test_resume_no_completed_steps_still_injects_message(
    command_handler, session_manager, test_session
):
    """Test that resume adds system message even with no completed steps (T084)."""
    # Create workflow with no completed steps
    steps = [
        WorkflowStep(description="First task", status="pending"),
        WorkflowStep(description="Second task", status="pending"),
    ]

    workflow = WorkflowState(
        objective="No completion workflow",
        steps=steps,
        current_step_index=0
    )

    session_manager.save_workflow(test_session.id, workflow)

    initial_count = len(test_session.messages)

    # Resume workflow
    result = await command_handler.handle_workflow_resume(test_session)

    assert result.success is True

    # Verify message added
    assert len(test_session.messages) == initial_count + 1

    # Verify system message content
    system_msg = test_session.messages[-1]
    assert system_msg.role == "system"
    assert "No completion workflow" in system_msg.content
    assert "Progress: 0% (0/2 steps)" in system_msg.content
    assert "Current step: First task" in system_msg.content


# ===== Test 17: Resume Message Timestamp Validation =====

@pytest.mark.asyncio
async def test_resume_message_has_valid_timestamp(
    command_handler, session_manager, test_session
):
    """Test that injected system message has valid timestamp (T084)."""
    import time

    # Create workflow
    steps = [
        WorkflowStep(description="Task", status="pending"),
    ]

    workflow = WorkflowState(
        objective="Timestamp test",
        steps=steps,
        current_step_index=0
    )

    session_manager.save_workflow(test_session.id, workflow)

    # Record time before resume
    before_time = time.time()

    # Small delay to ensure timestamp difference
    await asyncio.sleep(0.01)

    # Resume workflow
    result = await command_handler.handle_workflow_resume(test_session)

    await asyncio.sleep(0.01)

    # Record time after resume
    after_time = time.time()

    assert result.success is True

    # Get injected message
    system_msg = test_session.messages[-1]

    # Verify timestamp is within expected range (timestamp is already a float)
    msg_timestamp = system_msg.timestamp
    assert before_time <= msg_timestamp <= after_time
