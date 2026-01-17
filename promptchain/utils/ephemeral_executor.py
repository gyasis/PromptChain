"""Ephemeral Tool Executor for PromptChain.

Executes heavy tool operations (docker, uv, code execution, large file reads)
in isolated contexts and returns only summarized results to save tokens.

This prevents context window explosion from verbose tool outputs while
preserving critical information like errors, success/failure status.
"""

import asyncio
import fnmatch
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class EphemeralResult:
    """Result of an ephemeral tool execution.

    Attributes:
        success: Whether the operation completed successfully
        summary: Concise summary of the result (max ~500 tokens)
        error_details: Full error details if operation failed
        execution_time: Time taken in seconds
        original_output_tokens: Estimated tokens in original output (for metrics)
        tool_name: Name of the tool that was executed
        was_ephemeral: Whether this was executed ephemerally
    """

    success: bool
    summary: str
    error_details: Optional[str] = None
    execution_time: float = 0.0
    original_output_tokens: int = 0
    tool_name: str = ""
    was_ephemeral: bool = True


# Default patterns for heavy tools that should use ephemeral execution
DEFAULT_HEAVY_TOOL_PATTERNS: Set[str] = {
    # Docker operations
    "docker_*",
    "mcp_*docker*",
    # UV/package management
    "uv_*",
    "pip_*",
    "mcp_*uv*",
    # Code execution
    "execute_code",
    "run_code",
    "code_execute",
    "sandbox_execute",
    "sandbox_run",
    # Shell/terminal
    "terminal_execute",
    "run_shell",
    "shell_execute",
    "bash_execute",
    # File operations (for large files)
    "file_read",
    "read_file",
    # API calls
    "api_call",
    "http_request",
    "fetch_url",
}

# Size thresholds for ephemeral execution
DEFAULT_FILE_SIZE_THRESHOLD_KB = 10
DEFAULT_RESPONSE_SIZE_THRESHOLD_KB = 5


def _estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation).

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // 4  # Rough: ~4 chars per token


def _truncate_output(output: str, max_chars: int = 1000) -> str:
    """Truncate output while preserving beginning and end.

    Args:
        output: Output to truncate
        max_chars: Maximum characters to keep

    Returns:
        Truncated output with indicator
    """
    if len(output) <= max_chars:
        return output

    half = max_chars // 2
    return f"{output[:half]}\n\n... [{len(output) - max_chars} chars truncated] ...\n\n{output[-half:]}"


class EphemeralToolExecutor:
    """Executes heavy tools in isolated context, returns summarized results.

    Heavy operations like docker builds, package installations, and large
    file reads are executed normally, but their output is summarized before
    being returned to the main conversation context.

    This dramatically reduces token usage while preserving critical information.

    Attributes:
        timeout: Maximum execution time per tool (seconds)
        heavy_tool_patterns: Set of glob patterns matching heavy tools
        file_size_threshold_kb: Files larger than this use ephemeral
        response_size_threshold_kb: Responses larger than this get summarized
        summarize_success: Whether to summarize successful results
        capture_full_errors: Whether to include full error details
    """

    def __init__(
        self,
        timeout: int = 300,
        heavy_tool_patterns: Optional[Set[str]] = None,
        file_size_threshold_kb: int = DEFAULT_FILE_SIZE_THRESHOLD_KB,
        response_size_threshold_kb: int = DEFAULT_RESPONSE_SIZE_THRESHOLD_KB,
        summarize_success: bool = True,
        capture_full_errors: bool = True,
    ):
        """Initialize the ephemeral executor.

        Args:
            timeout: Maximum execution time per tool in seconds
            heavy_tool_patterns: Glob patterns for heavy tools (defaults to DEFAULT_HEAVY_TOOL_PATTERNS)
            file_size_threshold_kb: Files larger than this use ephemeral execution
            response_size_threshold_kb: Responses larger than this get summarized
            summarize_success: Whether to summarize successful outputs
            capture_full_errors: Whether to include full error details in results
        """
        self.timeout = timeout
        self.heavy_tool_patterns = heavy_tool_patterns or DEFAULT_HEAVY_TOOL_PATTERNS
        self.file_size_threshold_kb = file_size_threshold_kb
        self.response_size_threshold_kb = response_size_threshold_kb
        self.summarize_success = summarize_success
        self.capture_full_errors = capture_full_errors

        # Metrics tracking
        self._total_executions = 0
        self._total_tokens_saved = 0

    def is_heavy_tool(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a tool should use ephemeral execution.

        Args:
            tool_name: Name of the tool
            args: Tool arguments (used to check file sizes, etc.)

        Returns:
            True if tool should be executed ephemerally
        """
        # Check against patterns
        for pattern in self.heavy_tool_patterns:
            if fnmatch.fnmatch(tool_name.lower(), pattern.lower()):
                return True

        # Check for file operations with large files
        if args and tool_name.lower() in ("file_read", "read_file"):
            file_path = args.get("path") or args.get("file_path") or args.get("file")
            if file_path:
                try:
                    import os
                    file_size_kb = os.path.getsize(file_path) / 1024
                    if file_size_kb > self.file_size_threshold_kb:
                        logger.debug(f"File {file_path} ({file_size_kb:.1f}KB) exceeds threshold")
                        return True
                except (OSError, TypeError):
                    pass  # File doesn't exist or path invalid

        return False

    def _summarize_docker_output(self, output: str, command: str = "") -> str:
        """Summarize Docker command output.

        Args:
            output: Raw Docker output
            command: Docker command that was run

        Returns:
            Summarized output
        """
        lines = output.strip().split("\n")

        # Extract key information
        summary_parts = []

        # Check for success indicators
        if "successfully built" in output.lower():
            # Extract image ID
            for line in reversed(lines):
                if "successfully built" in line.lower():
                    summary_parts.append(f"[SUCCESS] {line.strip()}")
                    break
            # Look for image tag
            for line in lines:
                if "successfully tagged" in line.lower():
                    summary_parts.append(line.strip())
                    break

        elif "created" in output.lower() or "started" in output.lower():
            summary_parts.append(f"[SUCCESS] Container operation completed")
            # Get container ID if present
            for line in lines[-5:]:
                if re.match(r'^[a-f0-9]{12,64}$', line.strip()):
                    summary_parts.append(f"Container ID: {line.strip()[:12]}")
                    break

        elif "error" in output.lower() or "failed" in output.lower():
            summary_parts.append("[FAILED] Docker operation failed")
            # Extract error message
            for line in lines:
                if "error" in line.lower():
                    summary_parts.append(line.strip()[:200])
                    break

        else:
            # Generic summary
            summary_parts.append(f"[COMPLETED] Docker command finished ({len(lines)} lines output)")

        return "\n".join(summary_parts) if summary_parts else _truncate_output(output, 500)

    def _summarize_uv_output(self, output: str, command: str = "") -> str:
        """Summarize UV/pip output.

        Args:
            output: Raw UV/pip output
            command: Command that was run

        Returns:
            Summarized output
        """
        lines = output.strip().split("\n")
        summary_parts = []

        # Count installations
        installed = [l for l in lines if "installed" in l.lower() or "adding" in l.lower()]
        errors = [l for l in lines if "error" in l.lower() or "failed" in l.lower()]

        if errors:
            summary_parts.append(f"[FAILED] Installation encountered errors")
            for error in errors[:3]:  # Show first 3 errors
                summary_parts.append(f"  - {error.strip()[:150]}")

        elif installed:
            summary_parts.append(f"[SUCCESS] Installed {len(installed)} package(s)")
            # Show a few package names
            for line in installed[:5]:
                pkg_match = re.search(r'([\w-]+)(?:==|>=|<=|~=|@)', line)
                if pkg_match:
                    summary_parts.append(f"  - {pkg_match.group(1)}")

        elif "resolved" in output.lower():
            summary_parts.append("[SUCCESS] Dependencies resolved")

        else:
            summary_parts.append(f"[COMPLETED] UV/pip command finished ({len(lines)} lines)")

        return "\n".join(summary_parts)

    def _summarize_code_output(self, output: str, code: str = "") -> str:
        """Summarize code execution output.

        Args:
            output: Raw execution output
            code: Code that was executed

        Returns:
            Summarized output
        """
        lines = output.strip().split("\n") if output else []

        # Check for errors
        error_indicators = ["error", "exception", "traceback", "failed"]
        has_error = any(ind in output.lower() for ind in error_indicators)

        if has_error:
            # Extract the error
            error_lines = []
            in_traceback = False
            for line in lines:
                if "traceback" in line.lower():
                    in_traceback = True
                if in_traceback:
                    error_lines.append(line)
                elif any(ind in line.lower() for ind in error_indicators):
                    error_lines.append(line)

            if error_lines:
                return f"[FAILED] Code execution error:\n{chr(10).join(error_lines[-10:])}"
            else:
                return f"[FAILED] Execution failed. Output:\n{_truncate_output(output, 500)}"

        else:
            # Success - summarize output
            if len(output) < 500:
                return f"[SUCCESS] Output:\n{output}"
            else:
                return f"[SUCCESS] Execution completed ({len(lines)} lines, {len(output)} chars):\n{_truncate_output(output, 400)}"

    def _summarize_file_read(self, content: str, file_path: str = "") -> str:
        """Summarize file read output.

        Args:
            content: File content
            file_path: Path to the file

        Returns:
            Summarized content
        """
        lines = content.split("\n")
        chars = len(content)

        # Get file extension for context
        ext = file_path.split(".")[-1].lower() if "." in file_path else ""

        summary_parts = [
            f"[FILE READ] {file_path}",
            f"Size: {chars} chars, {len(lines)} lines",
        ]

        # For code files, show structure
        if ext in ("py", "js", "ts", "java", "go", "rs", "cpp", "c"):
            # Count functions/classes
            func_count = len(re.findall(r'\bdef\s+\w+|function\s+\w+|\bfn\s+\w+', content))
            class_count = len(re.findall(r'\bclass\s+\w+', content))
            if func_count or class_count:
                summary_parts.append(f"Structure: {class_count} classes, {func_count} functions")

        # Show preview
        preview_lines = lines[:10]
        preview = "\n".join(preview_lines)
        if len(lines) > 10:
            preview += f"\n... [{len(lines) - 10} more lines]"

        summary_parts.append(f"Preview:\n{preview}")

        return "\n".join(summary_parts)

    def _summarize_generic(self, output: str, tool_name: str) -> str:
        """Generic summarization for unrecognized tools.

        Args:
            output: Tool output
            tool_name: Name of the tool

        Returns:
            Summarized output
        """
        output_tokens = _estimate_tokens(output)

        if output_tokens < 100:
            return output  # Small enough, return as-is

        # Check for error indicators
        if any(ind in output.lower() for ind in ["error", "failed", "exception"]):
            return f"[COMPLETED WITH ERRORS] {tool_name}:\n{_truncate_output(output, 600)}"

        lines = output.strip().split("\n")
        return f"[SUCCESS] {tool_name} completed ({len(lines)} lines, ~{output_tokens} tokens):\n{_truncate_output(output, 400)}"

    def summarize_result(self, output: str, tool_name: str, args: Optional[Dict[str, Any]] = None) -> str:
        """Summarize tool output based on tool type.

        Args:
            output: Raw tool output
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Summarized output
        """
        tool_lower = tool_name.lower()

        # Docker operations
        if "docker" in tool_lower:
            command = args.get("command", "") if args else ""
            return self._summarize_docker_output(output, command)

        # UV/pip operations
        if any(pkg in tool_lower for pkg in ("uv_", "pip_", "install", "sync")):
            command = args.get("command", "") if args else ""
            return self._summarize_uv_output(output, command)

        # Code execution
        if any(ex in tool_lower for ex in ("execute", "run_code", "sandbox")):
            code = args.get("code", "") if args else ""
            return self._summarize_code_output(output, code)

        # File read
        if "file" in tool_lower and "read" in tool_lower:
            path = args.get("path", args.get("file_path", "")) if args else ""
            return self._summarize_file_read(output, path)

        # Generic fallback
        return self._summarize_generic(output, tool_name)

    async def execute(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_executor: Callable[[str, Dict[str, Any]], Any],
        force_ephemeral: bool = False,
    ) -> EphemeralResult:
        """Execute a tool, optionally in ephemeral mode.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            tool_executor: Async callable that executes the actual tool
            force_ephemeral: Force ephemeral execution even if tool not in heavy list

        Returns:
            EphemeralResult with summarized output
        """
        start_time = time.time()
        use_ephemeral = force_ephemeral or self.is_heavy_tool(tool_name, tool_args)

        try:
            # Execute the tool with timeout
            if asyncio.iscoroutinefunction(tool_executor):
                coro = tool_executor(tool_name, tool_args)
            else:
                # Wrap sync function
                coro = asyncio.to_thread(tool_executor, tool_name, tool_args)

            result = await asyncio.wait_for(coro, timeout=self.timeout)
            execution_time = time.time() - start_time

            # Convert result to string
            output = str(result) if result else ""
            original_tokens = _estimate_tokens(output)

            # Summarize if ephemeral
            if use_ephemeral and self.summarize_success:
                summary = self.summarize_result(output, tool_name, tool_args)
                summary_tokens = _estimate_tokens(summary)
                tokens_saved = original_tokens - summary_tokens

                self._total_executions += 1
                self._total_tokens_saved += tokens_saved

                logger.info(
                    f"Ephemeral execution of {tool_name}: "
                    f"{original_tokens} -> {summary_tokens} tokens "
                    f"(saved {tokens_saved})"
                )

                return EphemeralResult(
                    success=True,
                    summary=summary,
                    execution_time=execution_time,
                    original_output_tokens=original_tokens,
                    tool_name=tool_name,
                    was_ephemeral=True,
                )
            else:
                # Return full output for non-ephemeral tools
                return EphemeralResult(
                    success=True,
                    summary=output,
                    execution_time=execution_time,
                    original_output_tokens=original_tokens,
                    tool_name=tool_name,
                    was_ephemeral=False,
                )

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Tool {tool_name} timed out after {self.timeout}s"
            logger.error(error_msg)

            return EphemeralResult(
                success=False,
                summary=f"[TIMEOUT] {error_msg}",
                error_details=error_msg if self.capture_full_errors else None,
                execution_time=execution_time,
                tool_name=tool_name,
                was_ephemeral=use_ephemeral,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Tool {tool_name} failed: {error_msg}")

            summary = f"[FAILED] {tool_name}: {error_msg[:200]}"
            error_details = error_msg if self.capture_full_errors else None

            return EphemeralResult(
                success=False,
                summary=summary,
                error_details=error_details,
                execution_time=execution_time,
                tool_name=tool_name,
                was_ephemeral=use_ephemeral,
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics.

        Returns:
            Dict with total_executions and total_tokens_saved
        """
        return {
            "total_executions": self._total_executions,
            "total_tokens_saved": self._total_tokens_saved,
        }

    def reset_metrics(self) -> None:
        """Reset execution metrics."""
        self._total_executions = 0
        self._total_tokens_saved = 0
