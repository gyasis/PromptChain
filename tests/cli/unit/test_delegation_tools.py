#!/usr/bin/env python3
"""
Unit tests for delegation tools module.

Tests the task delegation, querying, and help request functionality
for multi-agent communication (US2, US6).

DESIGN PRINCIPLES:
- Mock session manager to avoid database dependencies
- Fast, isolated tests
- Comprehensive edge case coverage
- Test both success and error paths
"""

import json
import pytest
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Import delegation tools and dependencies
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from promptchain.cli.tools.library import delegation_tools
from promptchain.cli.models.task import Task, TaskPriority, TaskStatus


# ============================================================================
# MOCK SESSION MANAGER
# ============================================================================

class MockSessionManager:
    """Mock session manager for testing without database."""

    def __init__(self):
        self.tasks = []
        self.session = Mock()
        self.session.current_agent = "test_agent"

    def create_task(
        self,
        description: str,
        source_agent: str,
        target_agent: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        context: Optional[Dict] = None
    ) -> Task:
        """Create and store a mock task."""
        task = Task.create(
            description=description,
            source_agent=source_agent,
            target_agent=target_agent,
            priority=priority,
            context=context or {}
        )
        self.tasks.append(task)
        return task

    def list_tasks(
        self,
        target_agent: Optional[str] = None,
        status: Optional[TaskStatus] = None
    ) -> List[Task]:
        """List tasks with optional filtering."""
        filtered_tasks = self.tasks

        if target_agent:
            filtered_tasks = [t for t in filtered_tasks if t.target_agent == target_agent]

        if status:
            filtered_tasks = [t for t in filtered_tasks if t.status == status]

        return filtered_tasks

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[Dict] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Update task status."""
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        task.status = status
        if result:
            task.result = result
        if error_message:
            task.error_message = error_message

    def add_task(self, task: Task) -> None:
        """Add task to storage."""
        self.tasks.append(task)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_session_manager():
    """Create mock session manager."""
    return MockSessionManager()


@pytest.fixture(autouse=True)
def inject_session_manager(mock_session_manager):
    """Auto-inject mock session manager before each test."""
    delegation_tools.set_session_manager(mock_session_manager)
    yield
    # Clean up after test
    delegation_tools._session_manager = None


# ============================================================================
# TEST: delegate_task
# ============================================================================

class TestDelegateTask:
    """Tests for delegate_task function."""

    def test_delegate_task_success_low_priority(self, mock_session_manager):
        """Test successful task delegation with low priority."""
        result = delegation_tools.delegate_task(
            description="Analyze code for bugs",
            target_agent="code_reviewer",
            source_agent="orchestrator",
            priority="low"
        )

        # Verify success message
        assert "Task delegated:" in result
        assert "code_reviewer" in result
        assert "(priority: low)" in result

        # Verify task created in session
        tasks = mock_session_manager.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].description == "Analyze code for bugs"
        assert tasks[0].target_agent == "code_reviewer"
        assert tasks[0].source_agent == "orchestrator"
        assert tasks[0].priority == TaskPriority.LOW
        assert tasks[0].status == TaskStatus.PENDING

    def test_delegate_task_success_medium_priority(self, mock_session_manager):
        """Test successful task delegation with medium priority (default)."""
        result = delegation_tools.delegate_task(
            description="Run test suite",
            target_agent="test_runner",
            source_agent="ci_agent"
        )

        assert "Task delegated:" in result
        assert "test_runner" in result
        assert "(priority: medium)" in result

        tasks = mock_session_manager.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].priority == TaskPriority.MEDIUM

    def test_delegate_task_success_high_priority(self, mock_session_manager):
        """Test successful task delegation with high priority."""
        result = delegation_tools.delegate_task(
            description="Emergency: Fix production bug",
            target_agent="hotfix_agent",
            source_agent="monitor",
            priority="high"
        )

        assert "Task delegated:" in result
        assert "hotfix_agent" in result
        assert "(priority: high)" in result

        tasks = mock_session_manager.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].priority == TaskPriority.HIGH

    def test_delegate_task_with_context_preservation(self, mock_session_manager):
        """Test task delegation preserves context data."""
        context = {
            "file_path": "/path/to/file.py",
            "line_number": 42,
            "error_type": "NullPointerException"
        }

        result = delegation_tools.delegate_task(
            description="Debug exception",
            target_agent="debugger",
            source_agent="error_handler",
            context=context
        )

        assert "Task delegated:" in result

        tasks = mock_session_manager.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].context == context

    def test_delegate_task_empty_context(self, mock_session_manager):
        """Test task delegation with no context (defaults to empty dict)."""
        result = delegation_tools.delegate_task(
            description="Simple task",
            target_agent="worker",
            source_agent="manager"
        )

        tasks = mock_session_manager.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].context == {}

    def test_delegate_task_self_delegation_error(self, mock_session_manager):
        """Test that self-delegation is prevented."""
        result = delegation_tools.delegate_task(
            description="Do something",
            target_agent="same_agent",
            source_agent="same_agent"
        )

        assert "Error: Cannot delegate task to self" in result
        assert "same_agent" in result

        # Verify no task was created
        tasks = mock_session_manager.list_tasks()
        assert len(tasks) == 0

    def test_delegate_task_empty_description_error(self, mock_session_manager):
        """Test that empty description is rejected."""
        result = delegation_tools.delegate_task(
            description="   ",  # Whitespace only
            target_agent="worker",
            source_agent="manager"
        )

        assert "Error: Task description cannot be empty" in result

        # Verify no task was created
        tasks = mock_session_manager.list_tasks()
        assert len(tasks) == 0

    def test_delegate_task_invalid_priority_error(self, mock_session_manager):
        """Test that invalid priority is handled."""
        result = delegation_tools.delegate_task(
            description="Some task",
            target_agent="worker",
            source_agent="manager",
            priority="ultra_mega_high"  # Invalid
        )

        assert "Error: Invalid priority" in result
        assert "ultra_mega_high" in result
        assert "low/medium/high" in result

    def test_delegate_task_id_preview_truncation(self, mock_session_manager):
        """Test that task ID is truncated in response for readability."""
        result = delegation_tools.delegate_task(
            description="Test task",
            target_agent="worker",
            source_agent="manager"
        )

        # Response should have truncated ID (8 chars + "...")
        assert "..." in result

        # Full ID should be stored in database
        tasks = mock_session_manager.list_tasks()
        assert len(tasks) == 1
        assert len(tasks[0].task_id) == 36  # Full UUID


# ============================================================================
# TEST: get_pending_tasks
# ============================================================================

class TestGetPendingTasks:
    """Tests for get_pending_tasks function."""

    def test_get_pending_tasks_empty_queue(self, mock_session_manager):
        """Test querying pending tasks when none exist."""
        result = delegation_tools.get_pending_tasks(agent_name="worker")

        assert "No pending tasks for agent 'worker'" in result

    def test_get_pending_tasks_single_task(self, mock_session_manager):
        """Test querying pending tasks with one task."""
        # Create task
        mock_session_manager.create_task(
            description="Review code",
            source_agent="manager",
            target_agent="reviewer",
            priority=TaskPriority.HIGH
        )

        result = delegation_tools.get_pending_tasks(agent_name="reviewer")

        assert "Pending tasks for 'reviewer' (1):" in result
        assert "[high]" in result
        assert "Review code" in result

    def test_get_pending_tasks_multiple_tasks(self, mock_session_manager):
        """Test querying pending tasks with multiple tasks."""
        # Create multiple tasks
        mock_session_manager.create_task(
            description="Task 1: Urgent fix",
            source_agent="manager",
            target_agent="worker",
            priority=TaskPriority.HIGH
        )
        mock_session_manager.create_task(
            description="Task 2: Run tests",
            source_agent="manager",
            target_agent="worker",
            priority=TaskPriority.MEDIUM
        )
        mock_session_manager.create_task(
            description="Task 3: Update docs",
            source_agent="manager",
            target_agent="worker",
            priority=TaskPriority.LOW
        )

        result = delegation_tools.get_pending_tasks(agent_name="worker")

        assert "Pending tasks for 'worker' (3):" in result
        assert "[high]" in result
        assert "Task 1: Urgent fix" in result
        assert "[medium]" in result
        assert "Task 2: Run tests" in result
        assert "[low]" in result
        assert "Task 3: Update docs" in result

    def test_get_pending_tasks_filters_by_agent(self, mock_session_manager):
        """Test that pending tasks are filtered by target agent."""
        # Create tasks for different agents
        mock_session_manager.create_task(
            description="Task for agent A",
            source_agent="manager",
            target_agent="agent_a",
            priority=TaskPriority.MEDIUM
        )
        mock_session_manager.create_task(
            description="Task for agent B",
            source_agent="manager",
            target_agent="agent_b",
            priority=TaskPriority.MEDIUM
        )

        # Query agent_a
        result_a = delegation_tools.get_pending_tasks(agent_name="agent_a")
        assert "agent_a" in result_a
        assert "Task for agent A" in result_a
        assert "Task for agent B" not in result_a

        # Query agent_b
        result_b = delegation_tools.get_pending_tasks(agent_name="agent_b")
        assert "agent_b" in result_b
        assert "Task for agent B" in result_b
        assert "Task for agent A" not in result_b

    def test_get_pending_tasks_excludes_completed_tasks(self, mock_session_manager):
        """Test that only pending tasks are returned, not completed ones."""
        # Create pending task
        task1 = mock_session_manager.create_task(
            description="Pending task",
            source_agent="manager",
            target_agent="worker",
            priority=TaskPriority.MEDIUM
        )

        # Create completed task
        task2 = mock_session_manager.create_task(
            description="Completed task",
            source_agent="manager",
            target_agent="worker",
            priority=TaskPriority.MEDIUM
        )
        task2.status = TaskStatus.COMPLETED

        result = delegation_tools.get_pending_tasks(agent_name="worker")

        # Should only show pending task
        assert "Pending tasks for 'worker' (1):" in result
        assert "Pending task" in result
        assert "Completed task" not in result

    def test_get_pending_tasks_description_truncation(self, mock_session_manager):
        """Test that long descriptions are truncated for readability."""
        long_description = "This is a very long task description that exceeds fifty characters and should be truncated"

        mock_session_manager.create_task(
            description=long_description,
            source_agent="manager",
            target_agent="worker",
            priority=TaskPriority.MEDIUM
        )

        result = delegation_tools.get_pending_tasks(agent_name="worker")

        # Should be truncated with "..."
        assert "..." in result
        assert long_description not in result  # Full text shouldn't appear

    def test_get_pending_tasks_error_handling(self, mock_session_manager):
        """Test error handling when session manager fails."""
        # Make list_tasks raise an exception
        mock_session_manager.list_tasks = Mock(side_effect=Exception("Database error"))

        result = delegation_tools.get_pending_tasks(agent_name="worker")

        assert "Error retrieving tasks:" in result
        assert "Database error" in result


# ============================================================================
# TEST: update_task_status
# ============================================================================

class TestUpdateTaskStatus:
    """Tests for update_task_status function."""

    def test_update_status_pending_to_in_progress(self, mock_session_manager):
        """Test status transition from pending to in_progress."""
        # Create task
        task = mock_session_manager.create_task(
            description="Test task",
            source_agent="manager",
            target_agent="worker"
        )
        task_id = task.task_id

        # Update to in_progress
        result = delegation_tools.update_task_status(
            task_id=task_id,
            status="in_progress"
        )

        assert "status updated to 'in_progress'" in result
        assert task_id[:8] in result

        # Verify status changed
        updated_task = mock_session_manager.get_task(task_id)
        assert updated_task.status == TaskStatus.IN_PROGRESS

    def test_update_status_in_progress_to_completed(self, mock_session_manager):
        """Test status transition from in_progress to completed with result."""
        # Create and start task
        task = mock_session_manager.create_task(
            description="Test task",
            source_agent="manager",
            target_agent="worker"
        )
        task.status = TaskStatus.IN_PROGRESS
        task_id = task.task_id

        # Complete with result
        result_data = {
            "files_processed": 5,
            "errors_found": 2,
            "status": "success"
        }

        result = delegation_tools.update_task_status(
            task_id=task_id,
            status="completed",
            result=result_data
        )

        assert "status updated to 'completed'" in result

        # Verify status and result
        updated_task = mock_session_manager.get_task(task_id)
        assert updated_task.status == TaskStatus.COMPLETED
        assert updated_task.result == result_data

    def test_update_status_to_failed_with_error_message(self, mock_session_manager):
        """Test status transition to failed with error message."""
        # Create task
        task = mock_session_manager.create_task(
            description="Test task",
            source_agent="manager",
            target_agent="worker"
        )
        task_id = task.task_id

        # Fail task
        result = delegation_tools.update_task_status(
            task_id=task_id,
            status="failed",
            error_message="Connection timeout after 30 seconds"
        )

        assert "status updated to 'failed'" in result

        # Verify status and error message
        updated_task = mock_session_manager.get_task(task_id)
        assert updated_task.status == TaskStatus.FAILED
        assert updated_task.error_message == "Connection timeout after 30 seconds"

    def test_update_status_invalid_task_id(self, mock_session_manager):
        """Test handling of non-existent task ID."""
        result = delegation_tools.update_task_status(
            task_id="non-existent-id",
            status="completed"
        )

        assert "Error: Task 'non-existent-id' not found" in result

    def test_update_status_invalid_status_value(self, mock_session_manager):
        """Test handling of invalid status value."""
        # Create task
        task = mock_session_manager.create_task(
            description="Test task",
            source_agent="manager",
            target_agent="worker"
        )

        result = delegation_tools.update_task_status(
            task_id=task.task_id,
            status="super_done"  # Invalid
        )

        assert "Error: Invalid status 'super_done'" in result
        assert "in_progress/completed/failed" in result

    def test_update_status_completed_without_result(self, mock_session_manager):
        """Test that completed status can be set without result (optional)."""
        task = mock_session_manager.create_task(
            description="Test task",
            source_agent="manager",
            target_agent="worker"
        )

        result = delegation_tools.update_task_status(
            task_id=task.task_id,
            status="completed"
            # No result provided
        )

        assert "status updated to 'completed'" in result

        updated_task = mock_session_manager.get_task(task.task_id)
        assert updated_task.status == TaskStatus.COMPLETED
        assert updated_task.result is None

    def test_update_status_error_handling(self, mock_session_manager):
        """Test error handling when update fails."""
        task = mock_session_manager.create_task(
            description="Test task",
            source_agent="manager",
            target_agent="worker"
        )

        # Make update_task_status raise an exception
        mock_session_manager.update_task_status = Mock(
            side_effect=Exception("Database write failed")
        )

        result = delegation_tools.update_task_status(
            task_id=task.task_id,
            status="completed"
        )

        assert "Error updating task status:" in result
        assert "Database write failed" in result


# ============================================================================
# TEST: request_help
# ============================================================================

class TestRequestHelp:
    """Tests for request_help function."""

    @patch('promptchain.cli.tools.library.delegation_tools.registry')
    def test_request_help_with_capability_match(self, mock_registry, mock_session_manager):
        """Test help request finds agent with matching capabilities."""
        # Mock tool with capabilities
        mock_tool = Mock()
        mock_tool.allowed_agents = ["debugger_agent", "code_expert"]

        mock_registry.discover_capabilities.return_value = [mock_tool]

        result = delegation_tools.request_help(
            session_manager=mock_session_manager,
            help_request="Need help debugging authentication flow",
            required_capabilities=["debugging", "code_analysis"],
            priority="high"
        )

        # Verify capability discovery was called
        mock_registry.discover_capabilities.assert_called_once_with(
            capability_filter=["debugging", "code_analysis"]
        )

        # Verify result
        assert result["assigned_agent"] == "debugger_agent"
        assert result["is_broadcast"] is False
        assert result["status"] == "pending"
        assert "assigned to debugger_agent" in result["message"]

        # Verify task was created
        tasks = mock_session_manager.tasks
        assert len(tasks) == 1
        assert "[HELP REQUEST]" in tasks[0].description
        assert tasks[0].priority == TaskPriority.HIGH

    @patch('promptchain.cli.tools.library.delegation_tools.registry')
    def test_request_help_broadcast_when_no_match(self, mock_registry, mock_session_manager):
        """Test help request broadcasts when no capable agent found."""
        # No matching tools
        mock_registry.discover_capabilities.return_value = []

        result = delegation_tools.request_help(
            session_manager=mock_session_manager,
            help_request="Need help with obscure legacy system",
            required_capabilities=["legacy_system_expertise"],
            priority="medium"
        )

        # Should broadcast to all agents
        assert result["assigned_agent"] == "broadcast"
        assert result["is_broadcast"] is True
        assert "broadcast to all agents" in result["message"]

        # Verify task created with broadcast target
        tasks = mock_session_manager.tasks
        assert len(tasks) == 1
        assert tasks[0].target_agent == "broadcast"

    @patch('promptchain.cli.tools.library.delegation_tools.registry')
    def test_request_help_excludes_requesting_agent(self, mock_registry, mock_session_manager):
        """Test that requesting agent is excluded from assignment."""
        # Set current agent
        mock_session_manager.session.current_agent = "agent_a"

        # Mock tools with agent_a and agent_b
        mock_tool = Mock()
        mock_tool.allowed_agents = ["agent_a", "agent_b"]
        mock_registry.discover_capabilities.return_value = [mock_tool]

        result = delegation_tools.request_help(
            session_manager=mock_session_manager,
            help_request="Need help",
            required_capabilities=["some_capability"]
        )

        # Should assign to agent_b, not agent_a (requesting agent)
        assert result["assigned_agent"] == "agent_b"
        assert result["assigned_agent"] != "agent_a"

    def test_request_help_without_capabilities(self, mock_session_manager):
        """Test help request without capability requirements (broadcasts)."""
        result = delegation_tools.request_help(
            session_manager=mock_session_manager,
            help_request="General question about architecture",
            required_capabilities=None
        )

        # Should broadcast when no capabilities specified
        assert result["assigned_agent"] == "broadcast"
        assert result["is_broadcast"] is True

    def test_request_help_priority_handling(self, mock_session_manager):
        """Test that priority is correctly set for help requests."""
        # Test low priority
        result_low = delegation_tools.request_help(
            session_manager=mock_session_manager,
            help_request="Low priority question",
            priority="low"
        )
        task_low = mock_session_manager.tasks[-1]
        assert task_low.priority == TaskPriority.LOW

        # Test medium priority
        result_med = delegation_tools.request_help(
            session_manager=mock_session_manager,
            help_request="Medium priority question",
            priority="medium"
        )
        task_med = mock_session_manager.tasks[-1]
        assert task_med.priority == TaskPriority.MEDIUM

        # Test high priority (default for help requests)
        result_high = delegation_tools.request_help(
            session_manager=mock_session_manager,
            help_request="Urgent help needed",
            priority="high"
        )
        task_high = mock_session_manager.tasks[-1]
        assert task_high.priority == TaskPriority.HIGH

    def test_request_help_context_preservation(self, mock_session_manager):
        """Test that context is preserved in help request."""
        context_data = {
            "file": "auth.py",
            "line": 42,
            "error": "NullPointerException"
        }

        result = delegation_tools.request_help(
            session_manager=mock_session_manager,
            help_request="Debug this error",
            context=context_data
        )

        # Verify context stored in task
        task = mock_session_manager.tasks[-1]
        assert task.context["type"] == "help_request"
        assert task.context["original_context"] == context_data

    def test_request_help_task_format(self, mock_session_manager):
        """Test that help request task has correct format."""
        result = delegation_tools.request_help(
            session_manager=mock_session_manager,
            help_request="Need assistance",
            required_capabilities=["capability_1", "capability_2"]
        )

        task = mock_session_manager.tasks[-1]

        # Description should have [HELP REQUEST] prefix
        assert task.description.startswith("[HELP REQUEST]")
        assert "Need assistance" in task.description

        # Context should have help request metadata
        assert task.context["type"] == "help_request"
        assert task.context["required_capabilities"] == ["capability_1", "capability_2"]


# ============================================================================
# TEST: request_help_tool (wrapper)
# ============================================================================

class TestRequestHelpTool:
    """Tests for request_help_tool wrapper function."""

    @patch('promptchain.cli.tools.library.delegation_tools.request_help')
    def test_request_help_tool_success(self, mock_request_help, mock_session_manager):
        """Test successful help request via tool wrapper."""
        mock_request_help.return_value = {
            "task_id": "abc123",
            "assigned_agent": "helper",
            "status": "pending"
        }

        result = delegation_tools.request_help_tool(
            help_request="Need help",
            required_capabilities=["capability_1"]
        )

        # Verify request_help was called with session manager
        mock_request_help.assert_called_once()
        call_args = mock_request_help.call_args
        assert call_args[0][0] == mock_session_manager  # First arg is session manager
        assert call_args[0][1] == "Need help"

        # Verify JSON response
        result_dict = json.loads(result)
        assert result_dict["task_id"] == "abc123"
        assert result_dict["assigned_agent"] == "helper"

    def test_request_help_tool_error_handling(self, mock_session_manager):
        """Test error handling in tool wrapper."""
        # Make session manager unavailable
        delegation_tools._session_manager = None

        result = delegation_tools.request_help_tool(
            help_request="Need help"
        )

        result_dict = json.loads(result)
        assert "error" in result_dict
        assert result_dict["status"] == "failed"


# ============================================================================
# TEST: Session Manager Injection
# ============================================================================

class TestSessionManagerInjection:
    """Tests for session manager injection pattern."""

    def test_set_session_manager(self):
        """Test setting session manager."""
        mock_sm = MockSessionManager()
        delegation_tools.set_session_manager(mock_sm)

        retrieved_sm = delegation_tools.get_session_manager()
        assert retrieved_sm is mock_sm

    def test_get_session_manager_not_initialized(self):
        """Test error when session manager not initialized."""
        # Clear session manager
        delegation_tools._session_manager = None

        with pytest.raises(RuntimeError, match="Session manager not initialized"):
            delegation_tools.get_session_manager()


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
