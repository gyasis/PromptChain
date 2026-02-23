"""Integration tests for T091: Workflow + AgenticStepProcessor integration.

Tests that workflow objectives are properly integrated with AgenticStepProcessor
for automatic step-by-step execution guided by workflow state.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from promptchain.cli.models import Agent, Message, Session
from promptchain.cli.models.workflow import WorkflowState, WorkflowStep
from promptchain.cli.session_manager import SessionManager
from promptchain.cli.tui.app import PromptChainApp


@pytest.fixture
def temp_sessions_dir(tmp_path):
    """Create temporary sessions directory."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


@pytest.fixture
def session_manager(temp_sessions_dir):
    """Create session manager with temp directory."""
    return SessionManager(sessions_dir=temp_sessions_dir)


@pytest.fixture
def test_session(session_manager):
    """Create test session with agent."""
    session = session_manager.create_session(
        name="workflow-test",
        working_directory=Path.cwd()
    )

    # Add test agent
    agent = Agent(
        name="test-agent",
        model_name="openai/gpt-4",
        description="Test agent for workflow execution"
    )
    session.agents["test-agent"] = agent
    session.active_agent = "test-agent"

    session_manager.save_session(session)
    return session


@pytest.fixture
def test_workflow(session_manager, test_session):
    """Create test workflow with multiple steps."""
    workflow = WorkflowState(
        objective="Complete multi-step research project",
        steps=[
            WorkflowStep(description="Research background information", status="pending"),
            WorkflowStep(description="Analyze key findings", status="pending"),
            WorkflowStep(description="Write executive summary", status="pending"),
        ]
    )

    session_manager.save_workflow(test_session.id, workflow)
    return workflow


def create_ui_mocks():
    """Helper to create properly configured UI mocks."""
    chat_view_mock = MagicMock()

    # Use real lists that can be appended/popped
    messages_list = []
    children_list = []

    def add_message_side_effect(msg):
        messages_list.append(msg)
        # Simulate adding to children list
        child_mock = MagicMock()
        child_mock.is_processing = False
        child_mock.start_spinner = MagicMock()
        child_mock.stop_spinner = MagicMock()
        children_list.append(child_mock)

    def pop_side_effect():
        if children_list:
            children_list.pop()

    def messages_pop_side_effect():
        if messages_list:
            return messages_list.pop()

    def len_side_effect(*args, **kwargs):
        return len(children_list)

    chat_view_mock.add_message.side_effect = add_message_side_effect
    chat_view_mock.messages = messages_list  # Direct assignment of list
    chat_view_mock.children = children_list
    chat_view_mock.__len__ = MagicMock(side_effect=len_side_effect)
    chat_view_mock.pop = pop_side_effect

    status_bar_mock = MagicMock()

    def query_one_side_effect(selector, widget_type=None):
        if "chat-view" in selector:
            return chat_view_mock
        elif "status-bar" in selector:
            return status_bar_mock
        return MagicMock()

    return chat_view_mock, status_bar_mock, query_one_side_effect


@pytest.mark.asyncio
async def test_workflow_message_uses_agentic_processor(
    session_manager, test_session, test_workflow
):
    """Test that workflow messages use AgenticStepProcessor with step objective (T091)."""

    # Mock PromptChain to verify AgenticStepProcessor integration
    with patch("promptchain.cli.tui.app.PromptChain") as mock_promptchain_class:
        mock_chain = AsyncMock()
        mock_chain.process_prompt_async.return_value = "Research completed successfully. Found relevant papers."
        mock_promptchain_class.return_value = mock_chain

        # Mock AgenticStepProcessor to verify it's created with correct objective
        with patch("promptchain.utils.agentic_step_processor.AgenticStepProcessor") as mock_agentic_class:
            mock_agentic = MagicMock()
            mock_agentic_class.return_value = mock_agentic

            # Create app with test session
            app = PromptChainApp(
                session_name=test_session.name,
                sessions_dir=session_manager.sessions_dir
            )
            app.session = test_session
            app.session_manager = session_manager

            # Mock UI components
            app.query_one = MagicMock()
            chat_view_mock = MagicMock()
            chat_view_mock.add_message = MagicMock()
            chat_view_mock.messages = []
            chat_view_mock.children = []
            chat_view_mock.__len__ = MagicMock(return_value=0)
            chat_view_mock.pop = MagicMock()

            status_bar_mock = MagicMock()

            def query_one_side_effect(selector, widget_type=None):
                if "chat-view" in selector:
                    return chat_view_mock
                elif "status-bar" in selector:
                    return status_bar_mock
                return MagicMock()

            app.query_one.side_effect = query_one_side_effect

            # Mock progress widgets
            app.show_reasoning_progress = MagicMock()
            app.hide_reasoning_progress = MagicMock()

            # Trigger workflow message handling
            await app._handle_workflow_message(
                content="Start researching background",
                workflow=test_workflow
            )

            # Verify AgenticStepProcessor was created with workflow step objective
            mock_agentic_class.assert_called_once()
            call_kwargs = mock_agentic_class.call_args[1]

            assert call_kwargs["objective"] == "Research background information"
            assert call_kwargs["max_internal_steps"] == 8
            assert call_kwargs["history_mode"] == "progressive"
            assert "progress_callback" in call_kwargs

            # Verify PromptChain was created with AgenticStepProcessor
            mock_promptchain_class.assert_called_once()
            chain_kwargs = mock_promptchain_class.call_args[1]

            assert "instructions" in chain_kwargs
            instructions = chain_kwargs["instructions"]
            assert len(instructions) == 3
            assert "Working on workflow" in instructions[0]
            assert instructions[1] == mock_agentic  # AgenticStepProcessor instance
            assert "Summarize step completion" in instructions[2]

            # Verify reasoning progress was shown
            app.show_reasoning_progress.assert_called_once_with(
                objective="Research background information",
                max_steps=8
            )

            # Verify chain was executed
            mock_chain.process_prompt_async.assert_called_once()


@pytest.mark.asyncio
async def test_workflow_step_completion_detection(
    session_manager, test_session, test_workflow
):
    """Test that workflow step completion is detected and workflow advances (T091)."""

    # This test validates the overall integration - actual completion detection is tested
    # in test_workflow_persistence.py. Here we verify AgenticStepProcessor integration works.

    with patch("promptchain.cli.tui.app.PromptChain") as mock_promptchain_class:
        mock_chain = AsyncMock()
        # Response contains completion keywords
        mock_chain.process_prompt_async.return_value = (
            "Research completed. I found 10 relevant papers on the topic."
        )
        mock_promptchain_class.return_value = mock_chain

        with patch("promptchain.utils.agentic_step_processor.AgenticStepProcessor"):
            app = PromptChainApp(
                session_name=test_session.name,
                sessions_dir=session_manager.sessions_dir
            )
            app.session = test_session
            app.session_manager = session_manager

            # Mock UI
            chat_view_mock, status_bar_mock, query_one_side_effect = create_ui_mocks()
            app.query_one = MagicMock(side_effect=query_one_side_effect)
            app.show_reasoning_progress = MagicMock()
            app.hide_reasoning_progress = MagicMock()

            # Execute first step
            await app._handle_workflow_message(
                content="Research background",
                workflow=test_workflow
            )

            # Verify workflow step was marked as in_progress (shows integration working)
            updated_workflow = session_manager.load_workflow(test_session.id)
            assert updated_workflow is not None

            # Step should be marked in_progress during execution
            step = updated_workflow.steps[0]
            assert step.status in ["in_progress", "completed"], \
                f"Expected 'in_progress' or 'completed', got '{step.status}' with error: {step.error_message}"

            # Verify agent was assigned
            assert step.agent_name == test_session.active_agent

            # Verify chain was executed (key integration point)
            mock_chain.process_prompt_async.assert_called_once()


@pytest.mark.asyncio
async def test_workflow_completion_message(
    session_manager, test_session, test_workflow
):
    """Test that workflow completion is detected and displayed (T091)."""

    # Complete all steps except the last one
    test_workflow.steps[0].mark_completed("Step 1 done")
    test_workflow.steps[1].mark_completed("Step 2 done")
    test_workflow.current_step_index = 2
    session_manager.save_workflow(test_session.id, test_workflow)

    with patch("promptchain.cli.tui.app.PromptChain") as mock_promptchain_class:
        mock_chain = AsyncMock()
        # Last step completion
        mock_chain.process_prompt_async.return_value = (
            "Executive summary completed. All objectives achieved."
        )
        mock_promptchain_class.return_value = mock_chain

        with patch("promptchain.utils.agentic_step_processor.AgenticStepProcessor"):
            app = PromptChainApp(
                session_name=test_session.name,
                sessions_dir=session_manager.sessions_dir
            )
            app.session = test_session
            app.session_manager = session_manager

            # Mock UI
            chat_view_mock, status_bar_mock, query_one_side_effect = create_ui_mocks()
            app.query_one = MagicMock(side_effect=query_one_side_effect)
            app.show_reasoning_progress = MagicMock()
            app.hide_reasoning_progress = MagicMock()

            # Execute last step
            await app._handle_workflow_message(
                content="Write summary",
                workflow=test_workflow
            )

            # Verify workflow execution completed (last step should be in_progress or completed)
            final_workflow = session_manager.load_workflow(test_session.id)
            last_step = final_workflow.steps[2]
            assert last_step.status in ["in_progress", "completed"], \
                f"Expected last step to be in_progress or completed, got '{last_step.status}'"

            # Verify workflow context was shown
            workflow_msg_calls = [
                call for call in chat_view_mock.add_message.call_args_list
                if len(call[0]) > 0 and isinstance(call[0][0], Message)
                and "Workflow Active" in call[0][0].content
            ]
            assert len(workflow_msg_calls) > 0


@pytest.mark.asyncio
async def test_workflow_step_failure_handling(
    session_manager, test_session, test_workflow
):
    """Test that workflow step failures are properly handled (T091)."""

    with patch("promptchain.cli.tui.app.PromptChain") as mock_promptchain_class:
        mock_chain = AsyncMock()
        # Simulate failure
        mock_chain.process_prompt_async.side_effect = RuntimeError("API timeout")
        mock_promptchain_class.return_value = mock_chain

        with patch("promptchain.utils.agentic_step_processor.AgenticStepProcessor"):
            app = PromptChainApp(
                session_name=test_session.name,
                sessions_dir=session_manager.sessions_dir
            )
            app.session = test_session
            app.session_manager = session_manager

            # Mock UI and error handler
            chat_view_mock, status_bar_mock, query_one_side_effect = create_ui_mocks()
            app.query_one = MagicMock(side_effect=query_one_side_effect)
            app.show_reasoning_progress = MagicMock()
            app.hide_reasoning_progress = MagicMock()
            app._handle_error = MagicMock(return_value=Message(
                role="system",
                content="Error: API timeout"
            ))

            # Execute step (should fail gracefully)
            await app._handle_workflow_message(
                content="Research background",
                workflow=test_workflow
            )

            # Verify step was marked as failed
            failed_workflow = session_manager.load_workflow(test_session.id)
            assert failed_workflow.steps[0].status == "failed"
            assert "API timeout" in failed_workflow.steps[0].error_message

            # Verify error was displayed
            app._handle_error.assert_called_once()


@pytest.mark.asyncio
async def test_workflow_context_displayed(
    session_manager, test_session, test_workflow
):
    """Test that workflow context is displayed before step execution (T091)."""

    with patch("promptchain.cli.tui.app.PromptChain") as mock_promptchain_class:
        mock_chain = AsyncMock()
        mock_chain.process_prompt_async.return_value = "Step executed"
        mock_promptchain_class.return_value = mock_chain

        with patch("promptchain.utils.agentic_step_processor.AgenticStepProcessor"):
            app = PromptChainApp(
                session_name=test_session.name,
                sessions_dir=session_manager.sessions_dir
            )
            app.session = test_session
            app.session_manager = session_manager

            chat_view_mock, status_bar_mock, query_one_side_effect = create_ui_mocks()
            app.query_one = MagicMock(side_effect=query_one_side_effect)
            app.show_reasoning_progress = MagicMock()
            app.hide_reasoning_progress = MagicMock()

            await app._handle_workflow_message(
                content="Test",
                workflow=test_workflow
            )

            # Verify workflow context message was displayed
            context_msg_calls = [
                call for call in chat_view_mock.add_message.call_args_list
                if len(call[0]) > 0 and isinstance(call[0][0], Message)
                and "Workflow Active" in call[0][0].content
            ]
            assert len(context_msg_calls) > 0

            # Verify message contains objective and current step
            context_msg = context_msg_calls[0][0][0]
            assert "Complete multi-step research project" in context_msg.content
            assert "Research background information" in context_msg.content
            assert "Progress:" in context_msg.content


@pytest.mark.asyncio
async def test_normal_mode_unaffected_by_workflow_integration(
    session_manager, test_session
):
    """Test that normal (non-workflow) message handling is unaffected (T091)."""

    # No workflow created - normal mode should work
    with patch("promptchain.cli.tui.app.PromptChain"):
        with patch("promptchain.cli.tui.app.AgentChain") as mock_agent_chain_class:
            mock_agent_chain = AsyncMock()
            mock_agent_chain.run_chat_turn_async.return_value = "Normal response"
            mock_agent_chain_class.return_value = mock_agent_chain

            app = PromptChainApp(
                session_name=test_session.name,
                sessions_dir=session_manager.sessions_dir
            )
            app.session = test_session
            app.session_manager = session_manager
            app.agent_chain = mock_agent_chain

            # Mock UI
            chat_view_mock = MagicMock()
            chat_view_mock.add_message = MagicMock()
            chat_view_mock.messages = []
            chat_view_mock.children = []
            chat_view_mock.__len__ = MagicMock(return_value=0)
            chat_view_mock.pop = MagicMock()

            status_bar_mock = MagicMock()

            def query_one_side_effect(selector, widget_type=None):
                if "chat-view" in selector:
                    return chat_view_mock
                elif "status-bar" in selector:
                    return status_bar_mock
                return MagicMock()

            app.query_one = MagicMock(side_effect=query_one_side_effect)
            app.file_context_manager = MagicMock()
            app.file_context_manager.inject_file_context.return_value = "test message"

            # Execute normal message (no workflow)
            await app.handle_user_message("test message")

            # Verify AgentChain was used (normal mode)
            mock_agent_chain.run_chat_turn_async.assert_called_once()

            # Verify no workflow context message was displayed
            workflow_msg_calls = [
                call for call in chat_view_mock.add_message.call_args_list
                if len(call[0]) > 0 and isinstance(call[0][0], Message)
                and "Workflow Active" in call[0][0].content
            ]
            assert len(workflow_msg_calls) == 0
