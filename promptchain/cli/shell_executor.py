"""Shell command execution with timeout and output capture.

This module provides shell command execution for the PromptChain CLI.
Supports:
- Async command execution
- Timeout with configurable duration
- Stdout and stderr capture
- Return code handling
- Process cleanup on timeout

User Story 5: Shell Command Execution (T119-T122)
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ShellResult:
    """Result of shell command execution.

    Attributes:
        stdout: Standard output from command
        stderr: Standard error from command
        return_code: Process exit code
        execution_time: Time taken to execute in seconds
        timed_out: Whether command exceeded timeout
        process_killed: Whether process was forcibly terminated
        error_message: Error message if timeout or failure occurred
    """

    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    timed_out: bool = False
    process_killed: bool = False
    error_message: Optional[str] = None


class ShellExecutor:
    """Execute shell commands with timeout and output capture.

    Features:
    - Async subprocess execution
    - Configurable timeout (default 30s)
    - Stdout/stderr capture
    - Process cleanup on timeout
    - Working directory support
    - Environment variable support

    Example:
        executor = ShellExecutor()
        result = await executor.execute_shell_command('ls -la', timeout=5.0)
        print(result.stdout)
    """

    def __init__(self, default_timeout: float = 30.0):
        """Initialize shell executor.

        Args:
            default_timeout: Default timeout in seconds (default: 30.0)
        """
        self.default_timeout = default_timeout

    async def execute_shell_command(
        self,
        command: str,
        timeout: Optional[float] = None,
        working_directory: Optional[str] = None,
        env: Optional[dict] = None,
    ) -> ShellResult:
        """Execute shell command with timeout.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds (None disables timeout)
            working_directory: Working directory for command
            env: Environment variables dict

        Returns:
            ShellResult with output, return code, and timing info

        Example:
            result = await executor.execute_shell_command(
                'echo "test"',
                timeout=5.0,
                working_directory='/tmp'
            )
        """
        # Use default timeout if not specified
        if timeout is None:
            timeout = self.default_timeout

        # Convert working directory to Path if provided
        cwd = Path(working_directory) if working_directory else None

        start_time = time.time()
        timed_out = False
        process_killed = False
        error_message = None
        stdout_data = ""
        stderr_data = ""
        return_code = -1

        try:
            # Create subprocess with explicit bash shell for proper expansion
            # Use /bin/bash -c to ensure brace expansion and other bash features work
            process = await asyncio.create_subprocess_exec(
                "/bin/bash",
                "-c",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            # Wait for completion with timeout
            if timeout and timeout > 0:
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(), timeout=timeout
                    )
                    stdout_data = stdout_bytes.decode("utf-8", errors="replace")
                    stderr_data = stderr_bytes.decode("utf-8", errors="replace")
                    return_code = process.returncode if process.returncode is not None else -1

                except asyncio.TimeoutError:
                    # Timeout occurred - kill process
                    timed_out = True
                    error_message = f"Command timed out after {timeout} seconds"

                    # Try to capture partial output before killing
                    try:
                        # Read any available output (non-blocking)
                        if process.stdout:
                            stdout_bytes = await asyncio.wait_for(
                                process.stdout.read(), timeout=0.5
                            )
                            stdout_data = stdout_bytes.decode("utf-8", errors="replace")
                    except (asyncio.TimeoutError, Exception):
                        pass

                    # Kill the process
                    try:
                        process.kill()
                        await process.wait()
                        process_killed = True
                    except ProcessLookupError:
                        # Process already terminated
                        pass

                    # Ensure return_code is set from process if available
                    return_code = process.returncode if process.returncode is not None else -1
            else:
                # No timeout - wait indefinitely
                stdout_bytes, stderr_bytes = await process.communicate()
                stdout_data = stdout_bytes.decode("utf-8", errors="replace")
                stderr_data = stderr_bytes.decode("utf-8", errors="replace")
                return_code = process.returncode if process.returncode is not None else -1

        except Exception as e:
            error_message = f"Error executing command: {str(e)}"
            return_code = -1

        execution_time = time.time() - start_time

        return ShellResult(
            stdout=stdout_data,
            stderr=stderr_data,
            return_code=return_code,
            execution_time=execution_time,
            timed_out=timed_out,
            process_killed=process_killed,
            error_message=error_message,
        )


class ShellCommandParser:
    """Parse shell commands from user input.

    Detects:
    - ! prefix for shell commands
    - !! for shell mode toggle
    - Extracts command without prefix

    Example:
        parser = ShellCommandParser()
        if parser.is_shell_command("!ls -la"):
            command = parser.extract_command("!ls -la")  # "ls -la"
    """

    def is_shell_command(self, text: str) -> bool:
        """Check if text is a shell command (starts with !).

        Args:
            text: Text to check

        Returns:
            True if text starts with ! (after stripping whitespace)

        Note: !! is considered a shell command (returns True) because it's
        used to toggle shell mode, which is still a shell-related command.
        """
        stripped = text.strip()
        return stripped.startswith("!")

    def is_shell_mode_toggle(self, text: str) -> bool:
        """Check if text is shell mode toggle (exactly !!).

        Args:
            text: Text to check

        Returns:
            True if text is exactly !! (after stripping whitespace)
        """
        stripped = text.strip()
        return stripped == "!!"

    def extract_command(self, text: str) -> str:
        """Extract command without ! prefix.

        Args:
            text: Shell command text starting with !

        Returns:
            Command string without ! prefix

        Example:
            extract_command("!ls -la") -> "ls -la"
            extract_command("  !echo test") -> "echo test"
        """
        # Strip leading whitespace first
        stripped = text.lstrip()
        if stripped.startswith("!"):
            # Remove ! and return rest (preserving any trailing whitespace)
            return stripped[1:]
        return stripped
