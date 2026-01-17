"""Integration test for /workflow create command (T086).

This test verifies that the /workflow create command:
1. Accepts an objective from the user
2. Uses LLM to generate 5-7 actionable steps
3. Creates WorkflowState with generated steps
4. Persists workflow to session database
5. Returns formatted status with all steps

Test Strategy:
- Use mock LiteLLM completion to avoid API calls
- Verify database persistence via session_manager.load_workflow()
- Validate WorkflowState structure and step formatting
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from promptchain.cli.command_handler import CommandHandler
from promptchain.cli.session_manager import SessionManager
from promptchain.cli.models import Session
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
    """Create test session."""
    return session_manager.create_session(
        name="test-workflow-session",
        working_directory=tmp_path
    )


@pytest.fixture
def command_handler(session_manager):
    """Create CommandHandler instance."""
    return CommandHandler(session_manager=session_manager)


class TestWorkflowCreateCommand:
    """Integration tests for /workflow create command (T086)."""

    @pytest.mark.asyncio
    async def test_workflow_create_with_llm_generated_steps(
        self, command_handler, test_session
    ):
        """Integration: /workflow create generates steps via LLM and persists to DB.

        Given: User objective "Implement user authentication system"
        When: handle_workflow_create() is called with objective
        Then: LLM generates 5-7 steps
        And: WorkflowState is created with generated steps
        And: Workflow is persisted to session database
        And: CommandResult contains formatted status
        """
        # Mock LiteLLM completion response
        mock_llm_response = [
            "Design database schema for users table",
            "Create User model with password hashing",
            "Implement login endpoint with JWT",
            "Add JWT token generation and validation",
            "Create user registration endpoint",
            "Write unit tests for authentication",
            "Add API documentation for auth endpoints"
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = json.dumps(mock_llm_response)

        with patch("litellm.completion", return_value=mock_response):
            # Execute workflow create command
            result = await command_handler.handle_workflow_create(
                session=test_session,
                objective="Implement user authentication system"
            )

        # Verify CommandResult success
        assert result.success is True
        assert "Workflow created: 7 steps" in result.message
        assert result.data["objective"] == "Implement user authentication system"
        assert result.data["step_count"] == 7
        assert len(result.data["steps"]) == 7

        # Verify workflow persisted to database
        loaded_workflow = command_handler.session_manager.load_workflow(test_session.id)
        assert loaded_workflow is not None
        assert loaded_workflow.objective == "Implement user authentication system"
        assert len(loaded_workflow.steps) == 7

        # Verify all steps are pending
        for step in loaded_workflow.steps:
            assert step.status == "pending"
            assert step.agent_name is None
            assert step.started_at is None
            assert step.completed_at is None

        # Verify step descriptions match LLM output
        for i, step in enumerate(loaded_workflow.steps):
            assert step.description == mock_llm_response[i]

    @pytest.mark.asyncio
    async def test_workflow_create_with_markdown_json_response(
        self, command_handler, test_session
    ):
        """Integration: Handle LLM response with markdown code blocks.

        Given: LLM returns JSON wrapped in ```json code blocks
        When: handle_workflow_create() parses response
        Then: Markdown formatting is stripped and JSON is extracted
        And: Workflow is created successfully
        """
        mock_llm_response = """```json
[
    "Step 1: Design API schema",
    "Step 2: Implement endpoints",
    "Step 3: Write tests",
    "Step 4: Deploy to staging",
    "Step 5: Monitor and optimize"
]
```"""

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = mock_llm_response

        with patch("litellm.completion", return_value=mock_response):
            result = await command_handler.handle_workflow_create(
                session=test_session,
                objective="Build REST API"
            )

        # Verify successful parsing and creation
        assert result.success is True
        assert result.data["step_count"] == 5

        # Verify workflow in database
        loaded_workflow = command_handler.session_manager.load_workflow(test_session.id)
        assert len(loaded_workflow.steps) == 5
        assert loaded_workflow.steps[0].description == "Step 1: Design API schema"

    @pytest.mark.asyncio
    async def test_workflow_create_caps_steps_at_10(
        self, command_handler, test_session
    ):
        """Integration: Workflow creation caps steps at 10 maximum.

        Given: LLM returns 15 steps
        When: handle_workflow_create() processes response
        Then: Only first 10 steps are included
        And: Workflow is created with 10 steps
        """
        # Mock LLM response with 15 steps
        mock_llm_response = [f"Step {i+1}" for i in range(15)]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = json.dumps(mock_llm_response)

        with patch("litellm.completion", return_value=mock_response):
            result = await command_handler.handle_workflow_create(
                session=test_session,
                objective="Complex multi-phase project"
            )

        # Verify capped at 10 steps
        assert result.success is True
        assert result.data["step_count"] == 10

        loaded_workflow = command_handler.session_manager.load_workflow(test_session.id)
        assert len(loaded_workflow.steps) == 10
        assert loaded_workflow.steps[9].description == "Step 10"

    @pytest.mark.asyncio
    async def test_workflow_create_handles_invalid_json(
        self, command_handler, test_session
    ):
        """Integration: Gracefully handle invalid JSON from LLM.

        Given: LLM returns invalid JSON or non-array response
        When: handle_workflow_create() attempts to parse
        Then: CommandResult indicates failure
        And: Error message explains JSON parsing issue
        """
        # Mock invalid JSON response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "This is not valid JSON"

        with patch("litellm.completion", return_value=mock_response):
            result = await command_handler.handle_workflow_create(
                session=test_session,
                objective="Test invalid response"
            )

        # Verify error handling
        assert result.success is False
        assert "Failed to create workflow" in result.message
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_workflow_create_uses_active_agent_model(
        self, command_handler, test_session, session_manager
    ):
        """Integration: Uses active agent's model for step generation.

        Given: Session has active agent with specific model
        When: handle_workflow_create() is called
        Then: LLM completion uses active agent's model
        And: Workflow is created successfully
        """
        # Create custom agent with different model
        from promptchain.cli.models.agent_config import Agent
        import time

        custom_agent = Agent(
            name="planner",
            model_name="openai/gpt-4o",
            description="Planning specialist",
            created_at=time.time()
        )
        test_session.agents["planner"] = custom_agent
        test_session.active_agent = "planner"
        session_manager.save_session(test_session)

        # Mock LLM response
        mock_llm_response = ["Step 1", "Step 2", "Step 3"]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = json.dumps(mock_llm_response)

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            result = await command_handler.handle_workflow_create(
                session=test_session,
                objective="Plan project architecture"
            )

            # Verify correct model was used
            call_args = mock_completion.call_args
            assert call_args[1]["model"] == "openai/gpt-4o"

        assert result.success is True

    @pytest.mark.asyncio
    async def test_workflow_create_output_formatting(
        self, command_handler, test_session
    ):
        """Integration: Verify formatted output contains status icons and progress.

        Given: Workflow created with 6 steps
        When: handle_workflow_create() returns result
        Then: Message includes workflow objective
        And: Message includes progress percentage (0%)
        And: Message includes all steps with pending icons (⬜)
        """
        mock_llm_response = [
            "Design architecture",
            "Implement core logic",
            "Add error handling",
            "Write documentation",
            "Create tests",
            "Deploy to production"
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = json.dumps(mock_llm_response)

        with patch("litellm.completion", return_value=mock_response):
            result = await command_handler.handle_workflow_create(
                session=test_session,
                objective="Build robust API service"
            )

        # Verify formatted output structure
        assert result.success is True
        message = result.message

        # Check for key formatting elements
        assert "Workflow: Build robust API service" in message
        assert "Progress: 0%" in message
        assert "(0/6 steps completed)" in message

        # Verify all steps shown with pending icon
        for i, step_desc in enumerate(mock_llm_response, 1):
            assert f"⬜ Step {i}: {step_desc} (pending)" in message

    @pytest.mark.asyncio
    async def test_workflow_create_handles_llm_exception(
        self, command_handler, test_session
    ):
        """Integration: Handle LLM API failures gracefully.

        Given: LiteLLM completion raises exception (API error, timeout, etc.)
        When: handle_workflow_create() is called
        Then: CommandResult indicates failure
        And: Error message contains exception details
        """
        with patch(
            "litellm.completion",
            side_effect=Exception("API rate limit exceeded")
        ):
            result = await command_handler.handle_workflow_create(
                session=test_session,
                objective="Test API failure"
            )

        # Verify error handling
        assert result.success is False
        assert "Failed to create workflow" in result.message
        assert "API rate limit exceeded" in result.error
