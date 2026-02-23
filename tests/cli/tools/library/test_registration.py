"""
Unit tests for library tool registration with CLI registry.

Tests that all 14 library tools (FileOperations, RipgrepSearcher, TerminalTool)
are properly registered with the PromptChain CLI tool registry and can be
discovered/executed by agents.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from promptchain.cli.tools import registry, ToolCategory


class TestLibraryToolRegistration:
    """Tests for library tool registration in CLI registry."""

    def test_all_library_tools_registered(self):
        """Test that all 14 library tools are registered."""
        expected_tools = [
            # File operations (12 tools)
            "file_read",
            "file_write",
            "file_edit",
            "file_append",
            "file_delete",
            "list_directory",
            "create_directory",
            "read_file_range",
            "insert_at_line",
            "replace_lines",
            "insert_after_pattern",
            "insert_before_pattern",
            # Code search
            "ripgrep_search",
            # Terminal
            "terminal_execute"
        ]

        all_tools = registry.list_tools()

        for tool_name in expected_tools:
            assert tool_name in all_tools, f"Tool '{tool_name}' not found in registry"

    def test_library_tools_in_utility_category(self):
        """Test that library tools are in the UTILITY category."""
        utility_tools = registry.get_by_category(ToolCategory.UTILITY)
        utility_tool_names = [t.name for t in utility_tools]

        library_tools = [
            "file_read", "file_write", "file_edit", "file_append",
            "file_delete", "list_directory", "create_directory",
            "read_file_range", "insert_at_line", "replace_lines",
            "insert_after_pattern", "insert_before_pattern",
            "ripgrep_search", "terminal_execute"
        ]

        for tool_name in library_tools:
            assert tool_name in utility_tool_names, \
                f"Library tool '{tool_name}' not in UTILITY category"

    def test_file_read_metadata(self):
        """Test file_read tool metadata."""
        tool = registry.get("file_read")

        assert tool is not None
        assert tool.name == "file_read"
        assert tool.category == ToolCategory.UTILITY
        assert "read" in tool.description.lower()
        assert "path" in tool.parameters

        # Check parameter
        path_param = tool.parameters["path"]
        assert path_param.required is True
        assert path_param.type == "string"

        # Check tags
        assert "file" in tool.tags
        assert "read" in tool.tags

    def test_file_write_metadata(self):
        """Test file_write tool metadata."""
        tool = registry.get("file_write")

        assert tool is not None
        assert tool.name == "file_write"
        assert "write" in tool.description.lower()

        # Check required parameters
        assert "path" in tool.parameters
        assert "content" in tool.parameters

        path_param = tool.parameters["path"]
        assert path_param.required is True

        content_param = tool.parameters["content"]
        assert content_param.required is True

    def test_ripgrep_search_metadata(self):
        """Test ripgrep_search tool metadata."""
        tool = registry.get("ripgrep_search")

        assert tool is not None
        assert tool.name == "ripgrep_search"
        assert "search" in tool.description.lower() or "ripgrep" in tool.description.lower()

        # Check required parameter
        assert "query" in tool.parameters
        query_param = tool.parameters["query"]
        assert query_param.required is True

        # Check optional parameters
        assert "search_path" in tool.parameters
        assert "case_sensitive" in tool.parameters
        assert "include_patterns" in tool.parameters
        assert "exclude_patterns" in tool.parameters

        # Check tags
        assert "search" in tool.tags

    def test_terminal_execute_metadata(self):
        """Test terminal_execute tool metadata."""
        tool = registry.get("terminal_execute")

        assert tool is not None
        assert tool.name == "terminal_execute"
        assert "terminal" in tool.description.lower() or "command" in tool.description.lower()

        # Check required parameter
        assert "command" in tool.parameters
        command_param = tool.parameters["command"]
        assert command_param.required is True

        # Check tags
        assert "terminal" in tool.tags or "shell" in tool.tags

    def test_openai_schema_generation(self):
        """Test that library tools generate valid OpenAI schemas."""
        library_tools = [
            "file_read", "file_write", "file_edit",
            "ripgrep_search", "terminal_execute"
        ]

        for tool_name in library_tools:
            tool = registry.get(tool_name)
            schema = tool.to_openai_schema()

            # Validate OpenAI schema structure
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

            function = schema["function"]
            assert function["name"] == tool_name
            assert function["parameters"]["type"] == "object"
            assert "properties" in function["parameters"]
            assert "required" in function["parameters"]

    @patch('promptchain.cli.tools.library.registration._file_ops')
    def test_file_read_execution(self, mock_file_ops):
        """Test that file_read can be executed via registry."""
        mock_file_ops.file_read.return_value = "file content here"

        result = registry.execute("file_read", path="test.txt")

        mock_file_ops.file_read.assert_called_once_with("test.txt")
        assert "file content here" in result

    @patch('promptchain.cli.tools.library.registration._file_ops')
    def test_file_write_execution(self, mock_file_ops):
        """Test that file_write can be executed via registry."""
        mock_file_ops.file_write.return_value = "✅ File written successfully"

        result = registry.execute(
            "file_write",
            path="output.txt",
            content="test content"
        )

        mock_file_ops.file_write.assert_called_once_with("output.txt", "test content")
        assert "✅" in result

    @patch('promptchain.cli.tools.library.registration._terminal')
    def test_terminal_execute_execution(self, mock_terminal):
        """Test that terminal_execute can be executed via registry."""
        mock_terminal.return_value = "command output"

        result = registry.execute("terminal_execute", command="ls -la")

        mock_terminal.assert_called_once_with("ls -la")
        assert "command output" in result

    def test_library_tools_have_examples(self):
        """Test that key library tools have usage examples."""
        tools_with_examples = [
            "file_read",
            "file_write",
            "file_edit",
            "ripgrep_search",
            "terminal_execute"
        ]

        for tool_name in tools_with_examples:
            tool = registry.get(tool_name)
            assert len(tool.examples) > 0, \
                f"Tool '{tool_name}' has no examples"

    def test_library_tools_have_tags(self):
        """Test that all library tools are properly tagged."""
        library_tools = [
            "file_read", "file_write", "list_directory",
            "ripgrep_search", "terminal_execute"
        ]

        for tool_name in library_tools:
            tool = registry.get(tool_name)
            assert len(tool.tags) > 0, \
                f"Tool '{tool_name}' has no tags"

    def test_registry_can_filter_by_file_tag(self):
        """Test that file tools can be found by 'file' tag."""
        file_tools = registry.get_by_tags(["file"])
        file_tool_names = [t.name for t in file_tools]

        # File operations should be found
        assert "file_read" in file_tool_names
        assert "file_write" in file_tool_names
        assert "file_edit" in file_tool_names

    def test_registry_can_filter_by_search_tag(self):
        """Test that search tools can be found by 'search' tag."""
        search_tools = registry.get_by_tags(["search"])
        search_tool_names = [t.name for t in search_tools]

        assert "ripgrep_search" in search_tool_names

    def test_parameter_validation_for_file_read(self):
        """Test parameter validation for file_read."""
        tool = registry.get("file_read")

        # Valid parameters
        tool.validate_parameters({"path": "test.txt"})

        # Invalid: missing required 'path'
        with pytest.raises(Exception):  # ToolValidationError
            tool.validate_parameters({})

    def test_parameter_validation_for_file_write(self):
        """Test parameter validation for file_write."""
        tool = registry.get("file_write")

        # Valid parameters
        tool.validate_parameters({
            "path": "output.txt",
            "content": "test content"
        })

        # Invalid: missing required parameters
        with pytest.raises(Exception):
            tool.validate_parameters({"path": "output.txt"})  # Missing content

        with pytest.raises(Exception):
            tool.validate_parameters({"content": "test"})  # Missing path

    def test_parameter_validation_for_ripgrep(self):
        """Test parameter validation for ripgrep_search."""
        tool = registry.get("ripgrep_search")

        # Valid parameters
        tool.validate_parameters({
            "query": "test",
            "search_path": "src/",
            "case_sensitive": True
        })

        # Invalid: missing required 'query'
        with pytest.raises(Exception):
            tool.validate_parameters({"search_path": "src/"})

    def test_parameter_validation_for_terminal(self):
        """Test parameter validation for terminal_execute."""
        tool = registry.get("terminal_execute")

        # Valid parameters
        tool.validate_parameters({"command": "ls -la"})

        # Invalid: missing required 'command'
        with pytest.raises(Exception):
            tool.validate_parameters({})

    def test_file_operations_count(self):
        """Test that we have exactly 12 file operation tools."""
        file_op_tools = [
            "file_read", "file_write", "file_edit", "file_append",
            "file_delete", "list_directory", "create_directory",
            "read_file_range", "insert_at_line", "replace_lines",
            "insert_after_pattern", "insert_before_pattern"
        ]

        all_tools = registry.list_tools()

        for tool_name in file_op_tools:
            assert tool_name in all_tools, f"File operation '{tool_name}' not registered"

        # Verify count
        assert len(file_op_tools) == 12, "Expected exactly 12 file operation tools"

    def test_total_library_tools_count(self):
        """Test that we have exactly 14 library tools total."""
        library_tools = [
            # File operations (12)
            "file_read", "file_write", "file_edit", "file_append",
            "file_delete", "list_directory", "create_directory",
            "read_file_range", "insert_at_line", "replace_lines",
            "insert_after_pattern", "insert_before_pattern",
            # Search (1)
            "ripgrep_search",
            # Terminal (1)
            "terminal_execute"
        ]

        all_tools = registry.list_tools()

        registered_count = sum(1 for t in library_tools if t in all_tools)
        assert registered_count == 14, f"Expected 14 library tools, found {registered_count}"

    def test_pattern_tools_have_boolean_parameters(self):
        """Test that pattern insertion tools have first_match boolean parameter."""
        pattern_tools = ["insert_after_pattern", "insert_before_pattern"]

        for tool_name in pattern_tools:
            tool = registry.get(tool_name)
            assert "first_match" in tool.parameters
            first_match_param = tool.parameters["first_match"]
            assert first_match_param.type == "boolean"
            assert first_match_param.default is True

    def test_line_tools_have_integer_parameters(self):
        """Test that line-based tools have integer line number parameters."""
        tool = registry.get("read_file_range")
        assert "start_line" in tool.parameters
        assert "end_line" in tool.parameters
        assert tool.parameters["start_line"].type == "integer"
        assert tool.parameters["end_line"].type == "integer"

        tool = registry.get("insert_at_line")
        assert "line_number" in tool.parameters
        assert tool.parameters["line_number"].type == "integer"

    def test_ripgrep_array_parameters(self):
        """Test that ripgrep has properly defined array parameters."""
        tool = registry.get("ripgrep_search")

        # Check include_patterns
        assert "include_patterns" in tool.parameters
        include_param = tool.parameters["include_patterns"]
        assert include_param.type == "array"
        assert include_param.items is not None
        assert include_param.items.type == "string"

        # Check exclude_patterns
        assert "exclude_patterns" in tool.parameters
        exclude_param = tool.parameters["exclude_patterns"]
        assert exclude_param.type == "array"
        assert exclude_param.items is not None
