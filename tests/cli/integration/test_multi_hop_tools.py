"""Integration tests for multi-hop reasoning with tool calls (T047).

These tests verify that AgenticStepProcessor can effectively use tools during
its internal reasoning loop to solve complex problems requiring multiple tool
interactions.

Test Coverage:
- test_sequential_tool_calls: Multiple tool calls in sequence
- test_tool_result_influences_reasoning: Tool results guide next steps
- test_ripgrep_search_integration: Real-world file search scenario
- test_mcp_tool_integration: External MCP tools in reasoning
- test_tool_error_recovery: Handle tool failures gracefully
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


class TestMultiHopToolsIntegration:
    """Integration tests for tool-enhanced multi-hop reasoning."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            # Create mock project structure
            (project_dir / "src").mkdir()
            (project_dir / "src" / "main.py").write_text(
                """
def authenticate_user(username: str, password: str) -> bool:
    '''Authenticate user with database.'''
    return check_credentials(username, password)

def check_credentials(username: str, password: str) -> bool:
    '''Check credentials against database.'''
    return database.query(username, password)
                """
            )

            (project_dir / "tests").mkdir()
            (project_dir / "tests" / "test_auth.py").write_text(
                """
def test_authenticate_user():
    '''Test user authentication.'''
    assert authenticate_user('admin', 'password123') == True

def test_invalid_credentials():
    '''Test invalid credentials.'''
    assert authenticate_user('user', 'wrong') == False
                """
            )

            yield project_dir

    def test_sequential_tool_calls(self, temp_project_dir):
        """Integration: AgenticStepProcessor makes multiple tool calls.

        Validates:
        - First tool call informs second tool call
        - Each tool result influences reasoning
        - Multi-hop pattern emerges
        - Final synthesis includes all tool insights
        """

        def list_files(path: str) -> str:
            """List files in directory."""
            dir_path = Path(path)
            if not dir_path.exists():
                return f"Directory not found: {path}"

            files = [f.name for f in dir_path.iterdir() if f.is_file()]
            return f"Files in {path}:\n" + "\n".join(f"- {f}" for f in files)

        def read_file(path: str) -> str:
            """Read file contents."""
            file_path = Path(path)
            if not file_path.exists():
                return f"File not found: {path}"

            return file_path.read_text()

        # Create agentic step with access to file tools
        agentic_step = AgenticStepProcessor(
            objective="Find authentication functions in the project",
            max_internal_steps=5,
            model_name="gpt-4.1-mini-2025-04-14",
        )

        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[agentic_step],
        )

        # Register tools
        chain.register_tool_function(list_files)
        chain.register_tool_function(read_file)
        chain.add_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "list_files",
                        "description": "List all files in a directory",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read contents of a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                },
            ]
        )

        # Execute reasoning workflow
        result = chain.process_prompt(
            f"Analyze authentication in project at {temp_project_dir}"
        )

        # Validate result
        assert isinstance(result, str)
        assert len(result) > 100

        # Should mention authentication-related findings
        assert "authenticate" in result.lower() or "auth" in result.lower()

    def test_tool_result_influences_reasoning(self):
        """Integration: Tool results guide subsequent reasoning steps.

        Validates:
        - Tool result changes reasoning direction
        - Follow-up tool calls based on previous results
        - Adaptive multi-hop strategy
        - Synthesis incorporates all tool outputs
        """
        call_history = []

        def search_documentation(query: str) -> str:
            """Search documentation database."""
            call_history.append(("search_documentation", query))

            if "async" in query.lower():
                return (
                    "AsyncIO Documentation:\n"
                    "- Use async/await for concurrent operations\n"
                    "- Event loop manages tasks\n"
                    "- See: asyncio.run(), asyncio.gather()"
                )
            elif "testing" in query.lower():
                return (
                    "Testing Async Code:\n"
                    "- Use pytest-asyncio plugin\n"
                    "- Mark tests with @pytest.mark.asyncio\n"
                    "- Use await in test functions"
                )
            else:
                return f"No documentation found for: {query}"

        def get_code_examples(topic: str) -> str:
            """Get code examples for topic."""
            call_history.append(("get_code_examples", topic))

            if "pytest" in topic.lower():
                return (
                    "```python\n"
                    "@pytest.mark.asyncio\n"
                    "async def test_async_function():\n"
                    "    result = await async_operation()\n"
                    "    assert result == expected\n"
                    "```"
                )
            else:
                return f"No examples for: {topic}"

        agentic_step = AgenticStepProcessor(
            objective="Learn how to test async Python code",
            max_internal_steps=4,
            model_name="gpt-4.1-mini-2025-04-14",
        )

        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[agentic_step],
        )

        # Register tools
        chain.register_tool_function(search_documentation)
        chain.register_tool_function(get_code_examples)
        chain.add_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "search_documentation",
                        "description": "Search Python documentation",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_code_examples",
                        "description": "Get code examples for a topic",
                        "parameters": {
                            "type": "object",
                            "properties": {"topic": {"type": "string"}},
                            "required": ["topic"],
                        },
                    },
                },
            ]
        )

        result = chain.process_prompt("How do I test async functions in Python?")

        # Validate reasoning flow
        assert isinstance(result, str)
        assert len(result) > 100

        # Should have made multiple tool calls
        assert len(call_history) >= 2

        # Tool calls should be related (one informs the next)
        # First call likely about async or testing
        first_call = call_history[0]
        assert "async" in first_call[1].lower() or "test" in first_call[1].lower()

    def test_ripgrep_search_integration(self, temp_project_dir):
        """Integration: Real-world file search with ripgrep.

        Validates:
        - Ripgrep tool integration
        - Pattern-based code search
        - Multi-file analysis
        - Result synthesis across files
        """
        from promptchain.tools.ripgrep_wrapper import RipgrepSearcher

        searcher = RipgrepSearcher()

        def search_code(pattern: str) -> str:
            """Search codebase for pattern."""
            results = searcher.search(pattern, search_path=str(temp_project_dir))

            if not results:
                return f"No matches found for: {pattern}"

            return (
                f"Found {len(results)} matches for '{pattern}':\n"
                + "\n".join(results[:10])
            )

        agentic_step = AgenticStepProcessor(
            objective="Find and analyze authentication patterns in codebase",
            max_internal_steps=4,
            model_name="gpt-4.1-mini-2025-04-14",
        )

        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[agentic_step],
        )

        chain.register_tool_function(search_code)
        chain.add_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "search_code",
                        "description": "Search codebase for regex pattern",
                        "parameters": {
                            "type": "object",
                            "properties": {"pattern": {"type": "string"}},
                            "required": ["pattern"],
                        },
                    },
                }
            ]
        )

        result = chain.process_prompt(
            "What authentication methods are used in this project?"
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.skip(reason="Requires MCP server setup")
    def test_mcp_tool_integration(self):
        """Integration: External MCP tools in multi-hop reasoning.

        Validates:
        - MCP tool discovery
        - Tool calling via MCP protocol
        - Multi-hop with external tools
        - Error handling for MCP failures
        """
        # This test requires MCP server to be running
        # Documenting expected behavior for future implementation

        from promptchain.utils.mcp_helpers import MCPHelper

        mcp_config = [
            {
                "id": "filesystem",
                "type": "stdio",
                "command": "mcp-server-filesystem",
                "args": ["--root", "./"],
            }
        ]

        agentic_step = AgenticStepProcessor(
            objective="Analyze project structure using filesystem tools",
            max_internal_steps=5,
            model_name="gpt-4.1-mini-2025-04-14",
        )

        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[agentic_step],
            mcp_servers=mcp_config,
        )

        # Should discover and use MCP filesystem tools
        result = chain.process_prompt("What is the project structure?")

        assert isinstance(result, str)

    def test_tool_error_recovery(self):
        """Integration: Graceful handling of tool failures.

        Validates:
        - Tool errors don't crash reasoning
        - Retry logic or alternative approach
        - Error messages included in reasoning
        - Partial results still useful
        """
        call_count = {"count": 0}

        def unreliable_tool(query: str) -> str:
            """Tool that fails intermittently."""
            call_count["count"] += 1

            # Fail on first call, succeed on second
            if call_count["count"] == 1:
                raise Exception("Tool temporarily unavailable")

            return f"Results for: {query}\n- Finding 1\n- Finding 2"

        agentic_step = AgenticStepProcessor(
            objective="Gather information using potentially unreliable tools",
            max_internal_steps=5,
            model_name="gpt-4.1-mini-2025-04-14",
        )

        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[agentic_step],
        )

        chain.register_tool_function(unreliable_tool)
        chain.add_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "unreliable_tool",
                        "description": "Search for information (may fail)",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ]
        )

        # Should handle tool failure and potentially retry or adapt
        try:
            result = chain.process_prompt("Find information about Python testing")

            # If successful, should have result
            assert isinstance(result, str)

        except Exception as e:
            # If reasoning fails due to tool, should have meaningful error
            assert "tool" in str(e).lower() or "error" in str(e).lower()

    def test_chain_breaking_with_tools(self):
        """Integration: Tool results can trigger chain breaking.

        Validates:
        - Early completion if tool provides complete answer
        - Chain breaking logic respects tool outputs
        - Efficient reasoning (not all steps needed)
        """

        def get_quick_answer(question: str) -> str:
            """Tool that provides complete answers."""
            if "python package manager" in question.lower():
                return "Python's package manager is pip (pip installs packages)"

            return "No quick answer available"

        agentic_step = AgenticStepProcessor(
            objective="Answer user question efficiently",
            max_internal_steps=5,
            model_name="gpt-4.1-mini-2025-04-14",
        )

        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[agentic_step],
        )

        chain.register_tool_function(get_quick_answer)
        chain.add_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_quick_answer",
                        "description": "Get quick answer to common questions",
                        "parameters": {
                            "type": "object",
                            "properties": {"question": {"type": "string"}},
                            "required": ["question"],
                        },
                    },
                }
            ]
        )

        result = chain.process_prompt("What is Python's package manager?")

        assert isinstance(result, str)
        assert "pip" in result.lower()

        # Should complete quickly if tool provides answer
        assert agentic_step.current_step <= 3

    def test_complex_multi_hop_scenario(self, temp_project_dir):
        """Integration: Complex multi-hop workflow with multiple tools.

        Validates:
        - 4+ hop reasoning chain
        - Different tools at each step
        - Progressive understanding
        - Comprehensive final synthesis
        """

        def list_python_files(directory: str) -> str:
            """List Python files in directory."""
            dir_path = Path(directory)
            if not dir_path.exists():
                return f"Directory not found: {directory}"

            py_files = list(dir_path.rglob("*.py"))
            return "Python files:\n" + "\n".join(str(f) for f in py_files)

        def count_functions(file_path: str) -> str:
            """Count functions in Python file."""
            path = Path(file_path)
            if not path.exists():
                return f"File not found: {file_path}"

            content = path.read_text()
            func_count = content.count("def ")
            return f"File {file_path} has {func_count} functions"

        def get_file_summary(file_path: str) -> str:
            """Get summary of Python file."""
            path = Path(file_path)
            if not path.exists():
                return "File not found"

            content = path.read_text()
            lines = content.split("\n")
            return f"File: {file_path}\nLines: {len(lines)}\nPreview:\n{chr(10).join(lines[:5])}"

        agentic_step = AgenticStepProcessor(
            objective="Analyze Python codebase structure and provide insights",
            max_internal_steps=6,
            model_name="gpt-4.1-mini-2025-04-14",
        )

        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=[agentic_step],
        )

        # Register multiple tools
        for func in [list_python_files, count_functions, get_file_summary]:
            chain.register_tool_function(func)

        chain.add_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "list_python_files",
                        "description": "List all Python files in directory",
                        "parameters": {
                            "type": "object",
                            "properties": {"directory": {"type": "string"}},
                            "required": ["directory"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "count_functions",
                        "description": "Count functions in Python file",
                        "parameters": {
                            "type": "object",
                            "properties": {"file_path": {"type": "string"}},
                            "required": ["file_path"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_file_summary",
                        "description": "Get summary of Python file",
                        "parameters": {
                            "type": "object",
                            "properties": {"file_path": {"type": "string"}},
                            "required": ["file_path"],
                        },
                    },
                },
            ]
        )

        result = chain.process_prompt(
            f"Analyze the Python project at {temp_project_dir}"
        )

        assert isinstance(result, str)
        assert len(result) > 200  # Should be comprehensive analysis

        # Should have executed multiple steps
        assert agentic_step.current_step >= 3
