"""Contract tests for shell command execution (T114).

These tests verify the core shell execution functionality:
- Command execution with stdout/stderr capture
- Return code handling
- Basic error handling
"""

import pytest
import asyncio
from pathlib import Path


class TestShellExecution:
    """Test shell command execution functionality."""

    @pytest.fixture
    def shell_executor(self):
        """Create ShellExecutor instance (to be implemented)."""
        try:
            from promptchain.cli.shell_executor import ShellExecutor
            return ShellExecutor()
        except ImportError:
            pytest.skip("ShellExecutor not yet implemented (will be in T119)")

    @pytest.mark.asyncio
    async def test_execute_command(self, shell_executor):
        """Contract: Execute shell command and capture output.

        Given: Simple shell command (echo "test")
        When: Executing the command
        Then: Returns result with stdout captured

        Validates:
        - Command execution works
        - Output captured correctly
        - Return code is 0 for success
        """
        result = await shell_executor.execute_shell_command('echo "test"')

        # Should have stdout
        assert result.stdout is not None
        assert "test" in result.stdout

        # Should have return code 0 (success)
        assert result.return_code == 0

        # stderr should be empty or None
        assert not result.stderr or result.stderr.strip() == ""

    @pytest.mark.asyncio
    async def test_capture_stdout(self, shell_executor):
        """Contract: Capture stdout from command.

        Given: Command that writes to stdout
        When: Executing the command
        Then: All stdout is captured

        Validates:
        - Full stdout capture
        - Multi-line output preserved
        - No truncation
        """
        result = await shell_executor.execute_shell_command('echo "line1"; echo "line2"')

        assert "line1" in result.stdout
        assert "line2" in result.stdout

        # Both lines should be present
        lines = result.stdout.strip().split('\n')
        assert len(lines) >= 2

    @pytest.mark.asyncio
    async def test_capture_stderr(self, shell_executor):
        """Contract: Capture stderr from command.

        Given: Command that writes to stderr
        When: Executing the command
        Then: stderr is captured separately

        Validates:
        - stderr captured separately from stdout
        - Error messages preserved
        - Return code reflects error
        """
        # Command that writes to stderr (using bash)
        result = await shell_executor.execute_shell_command('echo "error" >&2; exit 1')

        # Should capture stderr
        assert result.stderr is not None
        assert "error" in result.stderr

        # Should have non-zero return code
        assert result.return_code != 0

    @pytest.mark.asyncio
    async def test_return_code(self, shell_executor):
        """Contract: Return code reflects command success/failure.

        Given: Commands with different return codes
        When: Executing commands
        Then: Return codes captured correctly

        Validates:
        - Success (0) captured
        - Failure (non-zero) captured
        - Different error codes distinguished
        """
        # Success case
        success_result = await shell_executor.execute_shell_command('exit 0')
        assert success_result.return_code == 0

        # Failure case
        failure_result = await shell_executor.execute_shell_command('exit 1')
        assert failure_result.return_code == 1

        # Different error code
        custom_error = await shell_executor.execute_shell_command('exit 42')
        assert custom_error.return_code == 42

    @pytest.mark.asyncio
    async def test_working_directory(self, shell_executor, tmp_path):
        """Contract: Command executes in specified working directory.

        Given: Working directory specified
        When: Executing command (pwd)
        Then: Command runs in that directory

        Validates:
        - Working directory support
        - Path resolution
        - Directory context preserved
        """
        result = await shell_executor.execute_shell_command(
            'pwd',
            working_directory=str(tmp_path)
        )

        # Output should contain the temp directory path
        assert str(tmp_path) in result.stdout

    @pytest.mark.asyncio
    async def test_command_with_arguments(self, shell_executor):
        """Contract: Commands with arguments execute correctly.

        Given: Command with multiple arguments
        When: Executing the command
        Then: All arguments processed correctly

        Validates:
        - Argument parsing
        - Spaces in arguments
        - Special characters handled
        """
        result = await shell_executor.execute_shell_command('echo "hello world" test')

        assert "hello world" in result.stdout
        assert "test" in result.stdout

    @pytest.mark.asyncio
    async def test_environment_variables(self, shell_executor):
        """Contract: Environment variables accessible in commands.

        Given: Command using environment variable
        When: Executing the command
        Then: Environment variable expanded

        Validates:
        - Environment variable access
        - Variable expansion
        - Shell behavior preserved
        """
        result = await shell_executor.execute_shell_command('echo $HOME')

        # Should have some output (HOME should be set)
        assert result.stdout.strip() != ""
        assert result.return_code == 0

    @pytest.mark.asyncio
    async def test_command_not_found(self, shell_executor):
        """Contract: Non-existent commands handled gracefully.

        Given: Command that doesn't exist
        When: Executing the command
        Then: Returns error with non-zero return code

        Validates:
        - Command not found detection
        - Error message in stderr
        - Non-zero return code
        """
        result = await shell_executor.execute_shell_command('nonexistentcommand12345')

        # Should have non-zero return code
        assert result.return_code != 0

        # Should have error message (either in stderr or stdout depending on shell)
        error_output = (result.stderr or "") + (result.stdout or "")
        assert "not found" in error_output.lower() or "command" in error_output.lower()

    @pytest.mark.asyncio
    async def test_long_output(self, shell_executor):
        """Contract: Long output captured completely.

        Given: Command with long output
        When: Executing the command
        Then: All output captured without truncation

        Validates:
        - No output truncation
        - Large output handling
        - Memory efficiency
        """
        # Generate ~100 lines of output
        result = await shell_executor.execute_shell_command(
            'for i in {1..100}; do echo "Line $i"; done'
        )

        # Should have all lines
        lines = result.stdout.strip().split('\n')
        assert len(lines) >= 100

        # First and last lines should be present
        assert "Line 1" in result.stdout
        assert "Line 100" in result.stdout

    @pytest.mark.asyncio
    async def test_multiline_command(self, shell_executor):
        """Contract: Multi-line commands execute correctly.

        Given: Command with multiple statements
        When: Executing the command
        Then: All statements execute in order

        Validates:
        - Multi-line support
        - Statement sequencing
        - Combined output capture
        """
        command = '''
        echo "first"
        echo "second"
        echo "third"
        '''
        result = await shell_executor.execute_shell_command(command)

        # Should have all outputs in order
        assert "first" in result.stdout
        assert "second" in result.stdout
        assert "third" in result.stdout

        # Should maintain order
        first_pos = result.stdout.index("first")
        second_pos = result.stdout.index("second")
        third_pos = result.stdout.index("third")

        assert first_pos < second_pos < third_pos
