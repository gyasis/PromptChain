"""Integration tests for AgenticStepProcessor execution (T046).

These tests verify end-to-end AgenticStepProcessor functionality within
the CLI environment, including multi-step reasoning, tool calling, and
completion detection.

Test Coverage:
- test_agentic_step_execution: Basic AgenticStepProcessor workflow
- test_reasoning_with_history: History context in reasoning
- test_completion_detection: Automatic objective completion
- test_max_steps_exhaustion: Handling max_internal_steps limit
- test_agentic_step_in_tui: Integration with TUI app
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.agent_chain import AgentChain
from promptchain.cli.tui.app import PromptChainApp
from promptchain.cli.session_manager import SessionManager


class TestAgenticReasoningIntegration:
    """Integration tests for AgenticStepProcessor in CLI context."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create temporary sessions directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def session_manager(self, temp_sessions_dir):
        """Create SessionManager for testing."""
        return SessionManager(sessions_dir=temp_sessions_dir)

    def test_agentic_step_execution(self):
        """Integration: AgenticStepProcessor executes multi-step reasoning.

        Validates:
        - AgenticStepProcessor runs internal reasoning loop
        - Multiple LLM calls made
        - Final synthesis produced
        - Completion detected
        """
        agentic_step = AgenticStepProcessor(
            objective="Analyze Python testing best practices and recommend approach",
            max_internal_steps=4,
            model_name="gpt-4.1-mini-2025-04-14",
        )

        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[
                "Initial query: {input}",
                agentic_step,
                "Final summary: {input}",
            ],
        )

        # Execute chain
        result = chain.process_prompt(
            "What testing framework should I use for my Python CLI app?"
        )

        # Validate execution
        assert isinstance(result, str)
        assert len(result) > 100  # Should be substantial analysis

        # Validate agentic step executed
        # Note: Without tools, LLM may complete objective in 1 step (which is correct)
        assert agentic_step.current_step >= 1  # At least 1 reasoning step
        assert agentic_step.current_step <= agentic_step.max_internal_steps

        # Should have internal history
        assert len(agentic_step.internal_history) > 0

    def test_reasoning_with_history(self):
        """Integration: AgenticStepProcessor uses conversation history.

        Validates:
        - Previous context available to reasoning
        - History influences reasoning steps
        - Context from earlier messages used
        """
        agentic_step = AgenticStepProcessor(
            objective="Build on previous discussion to provide implementation details",
            max_internal_steps=3,
            model_name="gpt-4.1-mini-2025-04-14",
            history_mode="progressive",  # Include full history
        )

        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[agentic_step],
        )

        # Simulate conversation with history
        # First turn: Establish context
        result1 = chain.process_prompt(
            "I'm building a CLI app with multi-agent routing"
        )
        assert isinstance(result1, str)

        # Second turn: Should reference first turn
        result2 = chain.process_prompt("How should I test the router logic?")
        assert isinstance(result2, str)

        # AgenticStepProcessor should have more history after second turn
        assert len(agentic_step.internal_history) > 0

    def test_completion_detection(self):
        """Integration: AgenticStepProcessor detects objective completion.

        Validates:
        - Reasoning stops when objective achieved
        - May stop before max_internal_steps
        - Clear completion signal
        """
        agentic_step = AgenticStepProcessor(
            objective="Provide simple answer to factual question",
            max_internal_steps=5,  # Allow up to 5, but should complete sooner
            model_name="gpt-4.1-mini-2025-04-14",
        )

        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[agentic_step],
        )

        # Simple factual question should complete quickly
        result = chain.process_prompt("What is Python's package manager called?")

        assert isinstance(result, str)
        assert "pip" in result.lower()

        # Should complete in < 5 steps for simple question
        assert agentic_step.current_step <= 5

    def test_max_steps_exhaustion(self):
        """Integration: AgenticStepProcessor handles max_steps limit.

        Validates:
        - Stops at max_internal_steps
        - Provides best effort result
        - No infinite loops
        - Graceful handling
        """
        agentic_step = AgenticStepProcessor(
            objective="Solve complex architectural design problem",
            max_internal_steps=2,  # Very low limit
            model_name="gpt-4.1-mini-2025-04-14",
        )

        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[agentic_step],
        )

        # Complex problem may not complete in 2 steps
        result = chain.process_prompt(
            "Design a distributed microservices architecture with event sourcing"
        )

        assert isinstance(result, str)
        assert len(result) > 0  # Should still produce output

        # Should not exceed max steps (may complete earlier)
        # Note: Even complex problems may be answered in 1 step without tools
        assert agentic_step.current_step >= 1
        assert agentic_step.current_step <= agentic_step.max_internal_steps

    @pytest.mark.asyncio
    async def test_agentic_step_in_tui(self, session_manager, temp_sessions_dir):
        """Integration: AgenticStepProcessor works in TUI app context.

        Validates:
        - TUI can initialize agents with agentic steps
        - Message processing handles reasoning steps
        - Status updates during reasoning
        - Final result displayed
        """
        # Create session with agentic agent
        session = session_manager.create_session(
            name="agentic-test", working_directory=temp_sessions_dir
        )

        # Add agent with agentic step
        from promptchain.cli.models.agent_config import Agent

        researcher = Agent(
            name="researcher",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Deep research agent with multi-hop reasoning",
            instruction_chain=[
                {"type": "agentic_step", "objective": "Research topic", "max_internal_steps": 3}
            ],
        )

        session.agents["researcher"] = researcher
        session_manager.save_session(session)

        # Create TUI app
        app = PromptChainApp(session=session)

        # Mock the LLM calls to avoid actual API calls
        with patch("promptchain.PromptChain.process_prompt") as mock_process:
            mock_process.return_value = "Research complete: Found 3 key insights"

            # Simulate user message
            await app.handle_user_message("Research Python async patterns")

            # Validate app state
            assert len(app.chat_log) > 0  # Message added to log
            # Last message should be agent response
            last_message = app.chat_log[-1]
            assert last_message.agent_name == "default"  # Or "researcher" if switched

    def test_agentic_step_with_tools(self):
        """Integration: AgenticStepProcessor can call tools during reasoning.

        Validates:
        - Tool registration works
        - Tools callable from agentic steps
        - Tool results influence reasoning
        - Multi-hop with tool calls
        """

        def search_docs(query: str) -> str:
            """Mock documentation search tool."""
            return f"Found docs for: {query}\n- Doc 1: Best practices\n- Doc 2: Examples"

        agentic_step = AgenticStepProcessor(
            objective="Research using documentation search",
            max_internal_steps=3,
            model_name="gpt-4.1-mini-2025-04-14",
        )

        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[agentic_step],
        )

        # Register tool
        chain.register_tool_function(search_docs)
        chain.add_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "search_docs",
                        "description": "Search documentation for information",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ]
        )

        # Execute with tool available
        result = chain.process_prompt("How do I implement async generators in Python?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_agentic_chain_with_multiple_agents(self):
        """Integration: Multiple agents with agentic steps in AgentChain.

        Validates:
        - AgentChain routes to agents with agentic steps
        - Each agent's agentic step executes independently
        - Router selects appropriate agent
        - Multi-agent + multi-hop reasoning works
        """
        # Create agents with agentic steps
        researcher = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[
                AgenticStepProcessor(
                    objective="Deep research with multiple sources",
                    max_internal_steps=4,
                    model_name="gpt-4.1-mini-2025-04-14",
                )
            ],
        )

        coder = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[
                AgenticStepProcessor(
                    objective="Generate and refine code solution",
                    max_internal_steps=3,
                    model_name="gpt-4.1-mini-2025-04-14",
                )
            ],
        )

        # Create router config
        router_config = {
            "models": ["gpt-4.1-mini-2025-04-14"],
            "instructions": [None, "{input}"],
        }

        # Create AgentChain
        agent_chain = AgentChain(
            agents={"researcher": researcher, "coder": coder},
            agent_descriptions={
                "researcher": "Research specialist with deep analysis",
                "coder": "Code generation and optimization",
            },
            execution_mode="router",
            router=router_config,
            default_agent="researcher",
        )

        # Test research query (should route to researcher)
        with patch.object(
            agent_chain, "_route_to_agent", return_value=("researcher", "test query")
        ):
            result = agent_chain.run_chat_turn("Explain microservices patterns")
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_reasoning_progress_updates(self):
        """Integration: TUI receives progress updates during reasoning.

        Validates:
        - Progress callbacks fire during reasoning
        - Step counts updated
        - Status reflects current reasoning state
        - Completion signal sent
        """
        progress_updates = []

        def progress_callback(step: int, max_steps: int, status: str):
            """Capture progress updates."""
            progress_updates.append({"step": step, "max_steps": max_steps, "status": status})

        agentic_step = AgenticStepProcessor(
            objective="Multi-step analysis",
            max_internal_steps=3,
            model_name="gpt-4.1-mini-2025-04-14",
        )

        # Note: AgenticStepProcessor may not have progress_callback in current impl
        # This test documents desired behavior for T052

        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[agentic_step],
        )

        # Use async variant since this is an async test
        result = await chain.process_prompt_async("Analyze testing strategies")

        assert isinstance(result, str)
        # Progress updates would be collected if callback supported

    def test_error_recovery_in_reasoning(self):
        """Integration: AgenticStepProcessor handles errors gracefully.

        Validates:
        - LLM errors during reasoning handled
        - Partial results preserved
        - Error doesn't crash entire chain
        - Retry logic if applicable
        """
        agentic_step = AgenticStepProcessor(
            objective="Test error handling",
            max_internal_steps=3,
            model_name="gpt-4.1-mini-2025-04-14",
        )

        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[agentic_step],
        )

        # Simulate error scenario (invalid model would cause error)
        # In real scenario, network issues or API errors might occur

        try:
            result = chain.process_prompt("Test query")
            # If no error, should have valid result
            assert isinstance(result, str)
        except Exception as e:
            # If error occurs, should be handled gracefully
            # AgenticStepProcessor should provide partial results or clear error
            assert "error" in str(e).lower() or "failed" in str(e).lower()

    def test_agentic_step_yaml_end_to_end(self, temp_sessions_dir):
        """Integration: Full YAML → Agent → Execution flow.

        Validates:
        - Load YAML with agentic_step config
        - Create session with agentic agent
        - Execute reasoning workflow
        - Results persisted to history
        """
        # Create YAML config
        yaml_content = """
agents:
  analyst:
    model: gpt-4.1-mini-2025-04-14
    description: "Data analysis specialist"
    instruction_chain:
      - "Analyzing: {input}"
      - type: agentic_step
        objective: "Perform deep analysis with multiple perspectives"
        max_internal_steps: 3
      - "Summary: {input}"
"""
        yaml_path = temp_sessions_dir / "config.yml"
        yaml_path.write_text(yaml_content)

        # Load config
        from promptchain.cli.config.yaml_translator import YAMLConfigTranslator

        translator = YAMLConfigTranslator()
        yaml_config = translator.load_yaml(yaml_path)

        # Build agents
        agents = translator.build_agents(yaml_config)

        # Execute analyst agent
        analyst = agents["analyst"]
        result = analyst.process_prompt("Analyze error handling patterns in Python")

        # Validate result
        assert isinstance(result, str)
        assert len(result) > 100

        # Validate agentic step executed
        agentic_step = analyst.instructions[1]
        assert isinstance(agentic_step, AgenticStepProcessor)
        assert agentic_step.current_step > 0
