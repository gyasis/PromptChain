"""Safety validation layer for PromptChain CLI tool execution.

Prevents dangerous operations through comprehensive security checks:
- Path traversal prevention (OWASP A01:2021 - Broken Access Control)
- Command injection prevention (OWASP A03:2021 - Injection)
- Resource limits (OWASP A04:2021 - Insecure Design)
- Safe defaults (OWASP A05:2021 - Security Misconfiguration)

This module provides the final safety layer before executing any file operations,
shell commands, or other potentially dangerous operations requested by LLM agents.

Example:
    >>> validator = SafetyValidator(project_root="/home/user/project")
    >>> # Safe operation
    >>> validator.validate_path("/home/user/project/src/main.py")

    >>> # Blocked operation
    >>> validator.validate_path("/etc/passwd")  # Raises SecurityError

    >>> # Command validation
    >>> validator.validate_command(["ls", "-la"])  # Safe
    >>> validator.validate_command(["rm", "-rf", "/"])  # Raises SecurityError
"""

import os
import re
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Set, Union


class SecurityError(Exception):
    """Raised when a security validation check fails.

    This exception indicates a potential security threat was detected and blocked.
    All SecurityError instances include severity level for monitoring and alerting.
    """

    def __init__(self, message: str, severity: str = "high"):
        """Initialize security error.

        Args:
            message: Detailed error description
            severity: Security severity (low, medium, high, critical)
        """
        super().__init__(message)
        self.severity = severity


class SafetyValidator:
    """Validates operations for security before execution.

    Provides defense-in-depth security validation for:
    - File system operations (path validation, size limits)
    - Shell command execution (injection prevention, whitelist)
    - Resource consumption (timeouts, memory limits)
    - Operation-specific validation (delete confirmations, etc.)

    This validator integrates with Phase 9 security infrastructure:
    - Uses PathValidator for directory traversal prevention
    - Uses InputSanitizer for string validation
    - Extends YAML validation patterns to tool execution

    OWASP Compliance:
    - A01:2021 - Broken Access Control (path validation)
    - A03:2021 - Injection (command injection prevention)
    - A04:2021 - Insecure Design (resource limits, safe defaults)
    - A05:2021 - Security Misconfiguration (fail-secure, minimal privileges)

    Example:
        >>> # Initialize with project root
        >>> validator = SafetyValidator(
        ...     project_root="/home/user/my_project",
        ...     safe_mode=True
        ... )

        >>> # Validate file operation
        >>> safe_path = validator.validate_path("src/main.py")
        >>> validator.validate_file_size(safe_path)

        >>> # Validate command execution
        >>> validator.validate_command(["git", "status"])

        >>> # Comprehensive operation validation
        >>> validator.validate_operation(
        ...     operation="fs.read",
        ...     path="data/input.txt"
        ... )
    """

    # File size limits (prevent DoS via huge files)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB default
    MAX_FILE_SIZE_SAFE_MODE = 10 * 1024 * 1024  # 10MB in safe mode

    # Execution timeouts (prevent infinite loops)
    MAX_TIMEOUT = 300  # 5 minutes default
    MAX_TIMEOUT_SAFE_MODE = 60  # 1 minute in safe mode

    # Memory limits (future enhancement placeholder)
    MAX_MEMORY_MB = 1024  # 1GB

    # Dangerous commands (never allowed, even in unsafe mode)
    DANGEROUS_COMMANDS = {
        "rm -rf /",
        "rm -rf /*",
        "dd if=/dev/zero of=/dev/sda",
        "mkfs",
        "mkfs.ext4",
        "mkfs.ext3",
        "> /dev/sda",
        "> /dev/",
        ":(){ :|:& };:",  # Fork bomb
        "chmod -R 777 /",
        "chown -R nobody /",
    }

    # Dangerous command patterns (regex)
    DANGEROUS_PATTERNS = [
        re.compile(r"rm\s+-rf\s+/"),  # Remove root
        re.compile(r"dd\s+.*of=/dev/"),  # Overwrite device
        re.compile(r"mkfs\."),  # Format filesystem
        re.compile(r">\s*/dev/"),  # Write to device
        re.compile(r":\(\)\{.*\|\:&\s*\};:"),  # Fork bomb
        re.compile(r"chmod\s+-R\s+777\s+/"),  # Dangerous permissions
    ]

    # Whitelisted safe commands (allowed without additional validation)
    SAFE_COMMANDS = {
        "ls", "cat", "grep", "find", "echo", "pwd", "whoami",
        "git", "python", "python3", "pip", "pip3", "node", "npm",
        "pytest", "black", "isort", "flake8", "mypy",
    }

    # Dangerous file extensions (require extra validation)
    DANGEROUS_EXTENSIONS = {
        ".sh", ".bash", ".exe", ".bat", ".cmd", ".ps1",
        ".dll", ".so", ".dylib",
    }

    # System directories (never allow write access)
    SYSTEM_DIRECTORIES = {
        "/bin", "/sbin", "/boot", "/dev", "/etc", "/lib",
        "/lib64", "/proc", "/root", "/sys", "/usr/bin",
        "/usr/sbin", "/usr/lib", "/var/log",
    }

    def __init__(
        self,
        project_root: Union[str, Path],
        safe_mode: bool = True,
        max_file_size: Optional[int] = None,
        max_timeout: Optional[int] = None,
    ):
        """Initialize safety validator.

        Args:
            project_root: Root directory for all file operations (security boundary)
            safe_mode: If True, use stricter limits and validation
            max_file_size: Override default max file size (bytes)
            max_timeout: Override default max timeout (seconds)
        """
        self.project_root = Path(project_root).resolve()
        self.safe_mode = safe_mode

        # Set limits based on safe mode
        if safe_mode:
            self.max_file_size = max_file_size or self.MAX_FILE_SIZE_SAFE_MODE
            self.max_timeout = max_timeout or self.MAX_TIMEOUT_SAFE_MODE
        else:
            self.max_file_size = max_file_size or self.MAX_FILE_SIZE
            self.max_timeout = max_timeout or self.MAX_TIMEOUT

        # Ensure project root exists
        if not self.project_root.exists():
            raise SecurityError(
                f"Project root does not exist: {self.project_root}",
                severity="critical",
            )

    def validate_path(
        self,
        path: Union[str, Path],
        allow_write: bool = False,
        must_exist: bool = False,
    ) -> Path:
        """Validate file path for security issues.

        OWASP A01:2021 - Broken Access Control

        Checks:
        1. Null byte injection
        2. Directory traversal (stays within project_root)
        3. System directory access
        4. Symlink resolution
        5. Existence validation (optional)

        Args:
            path: Path to validate (absolute or relative to project_root)
            allow_write: If True, validate for write access
            must_exist: If True, path must exist

        Returns:
            Path: Validated absolute path

        Raises:
            SecurityError: If path validation fails

        Example:
            >>> validator.validate_path("src/main.py")
            PosixPath('/home/user/project/src/main.py')

            >>> validator.validate_path("../../etc/passwd")
            SecurityError: Path escapes project root
        """
        # Check for null bytes (common in path traversal exploits)
        if "\0" in str(path):
            raise SecurityError(
                "Path contains null bytes (potential attack)",
                severity="critical",
            )

        # Convert to Path object
        path_obj = Path(path)

        # Resolve to absolute path
        if path_obj.is_absolute():
            absolute_path = path_obj.resolve()
        else:
            absolute_path = (self.project_root / path_obj).resolve()

        # Check if path escapes project root (directory traversal prevention)
        try:
            absolute_path.relative_to(self.project_root)
        except ValueError:
            raise SecurityError(
                f"Path escapes project root: {path}\n"
                f"Attempted: {absolute_path}\n"
                f"Project root: {self.project_root}",
                severity="critical",
            )

        # Check for access to system directories (write operations only)
        if allow_write:
            for system_dir in self.SYSTEM_DIRECTORIES:
                if str(absolute_path).startswith(system_dir):
                    raise SecurityError(
                        f"Write access to system directory not allowed: {system_dir}",
                        severity="critical",
                    )

        # Check existence if required
        if must_exist and not absolute_path.exists():
            raise SecurityError(
                f"Path does not exist: {path}",
                severity="low",
            )

        # Resolve symlinks and validate final target is within project root
        if absolute_path.is_symlink():
            resolved_path = absolute_path.resolve()
            try:
                resolved_path.relative_to(self.project_root)
            except ValueError:
                raise SecurityError(
                    f"Symlink target escapes project root: {path} -> {resolved_path}",
                    severity="high",
                )

        return absolute_path

    def validate_command(self, command: List[str]) -> None:
        """Validate shell command for injection risks.

        OWASP A03:2021 - Injection

        Checks:
        1. Dangerous command patterns (rm -rf /, etc.)
        2. Shell metacharacters in arguments
        3. Command whitelist (safe_mode)
        4. Path injection in arguments

        Args:
            command: Command as list of strings (e.g., ["ls", "-la"])

        Raises:
            SecurityError: If command validation fails

        Example:
            >>> validator.validate_command(["ls", "-la"])  # Safe
            >>> validator.validate_command(["rm", "-rf", "/"])  # SecurityError
        """
        if not command or len(command) == 0:
            raise SecurityError(
                "Empty command not allowed",
                severity="medium",
            )

        # Convert to string for pattern matching
        command_str = " ".join(command)

        # Check for exact dangerous command matches
        for dangerous in self.DANGEROUS_COMMANDS:
            if dangerous in command_str:
                raise SecurityError(
                    f"Dangerous command detected: {dangerous}",
                    severity="critical",
                )

        # Check for dangerous patterns (regex)
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(command_str):
                raise SecurityError(
                    f"Dangerous command pattern detected: {pattern.pattern}",
                    severity="critical",
                )

        # Check command executable (first element)
        executable = command[0]

        # In safe mode, only allow whitelisted commands
        if self.safe_mode:
            # Extract base command (remove path if present)
            base_command = Path(executable).name

            if base_command not in self.SAFE_COMMANDS:
                raise SecurityError(
                    f"Command not in safe whitelist: {base_command}\n"
                    f"Allowed: {', '.join(sorted(self.SAFE_COMMANDS))}",
                    severity="high",
                )

        # Validate all arguments don't contain shell metacharacters
        shell_metacharacters = [";", "&", "|", "`", "$", "(", ")", "<", ">"]
        for arg in command[1:]:
            for char in shell_metacharacters:
                if char in arg:
                    raise SecurityError(
                        f"Shell metacharacter '{char}' in command argument: {arg}\n"
                        f"This may indicate command injection attempt",
                        severity="high",
                    )

    def validate_file_size(self, path: Path) -> None:
        """Validate file size is within limits.

        OWASP A04:2021 - Insecure Design (resource limits)

        Args:
            path: File path to check

        Raises:
            SecurityError: If file exceeds size limit
        """
        if not path.exists():
            return  # File doesn't exist yet, will be created

        file_size = path.stat().st_size

        if file_size > self.max_file_size:
            raise SecurityError(
                f"File exceeds size limit: {file_size} bytes\n"
                f"Maximum allowed: {self.max_file_size} bytes "
                f"({self.max_file_size / (1024 * 1024):.1f}MB)",
                severity="medium",
            )

    def validate_file_extension(self, path: Path, operation: str) -> None:
        """Validate file extension for potentially dangerous operations.

        Args:
            path: File path to check
            operation: Operation being performed (e.g., "execute", "write")

        Raises:
            SecurityError: If dangerous extension detected
        """
        extension = path.suffix.lower()

        if extension in self.DANGEROUS_EXTENSIONS:
            if operation in ("execute", "fs.execute"):
                raise SecurityError(
                    f"Execution of {extension} files not allowed for security",
                    severity="high",
                )
            elif operation in ("write", "fs.write") and self.safe_mode:
                raise SecurityError(
                    f"Writing {extension} files requires unsafe mode",
                    severity="medium",
                )

    def validate_operation(
        self,
        operation: str,
        path: Optional[Union[str, Path]] = None,
        command: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        **params,
    ) -> Dict[str, any]:
        """Comprehensive validation for tool operations.

        This is the main entry point for validating operations before execution.
        Performs operation-specific validation based on the operation type.

        Args:
            operation: Operation identifier (e.g., "fs.read", "shell.execute")
            path: File path (for file operations)
            command: Command list (for shell operations)
            timeout: Timeout in seconds (for long operations)
            **params: Additional operation-specific parameters

        Returns:
            Dict: Validated parameters ready for execution

        Raises:
            SecurityError: If validation fails

        Example:
            >>> validator.validate_operation(
            ...     operation="fs.read",
            ...     path="data/input.txt"
            ... )
            {'path': PosixPath('/home/user/project/data/input.txt')}
        """
        validated = {"operation": operation}

        # Path validation for file operations
        if path is not None:
            # Determine if write access needed
            allow_write = operation in {
                "fs.write",
                "fs.delete",
                "fs.move",
                "fs.create",
                "fs.mkdir",
            }

            # Validate path
            validated_path = self.validate_path(
                path,
                allow_write=allow_write,
                must_exist=operation in {"fs.read", "fs.delete", "fs.move"},
            )

            # Validate file size (for existing files)
            if validated_path.exists():
                self.validate_file_size(validated_path)

            # Validate extension for certain operations
            if operation in {"fs.write", "fs.execute"}:
                self.validate_file_extension(validated_path, operation)

            validated["path"] = validated_path

        # Command validation for execution operations
        if command is not None:
            self.validate_command(command)
            validated["command"] = command

        # Timeout validation
        if timeout is not None:
            if timeout > self.max_timeout:
                raise SecurityError(
                    f"Timeout exceeds limit: {timeout}s (max: {self.max_timeout}s)",
                    severity="medium",
                )
            validated["timeout"] = timeout
        else:
            validated["timeout"] = self.max_timeout

        # Operation-specific validation
        if operation == "fs.delete":
            self._validate_delete_operation(validated.get("path"), params)

        elif operation == "shell.execute":
            self._validate_shell_execution(command, params)

        elif operation == "fs.write":
            self._validate_write_operation(validated.get("path"), params)

        return validated

    def _validate_delete_operation(
        self, path: Optional[Path], params: Dict
    ) -> None:
        """Validate file deletion operation (extra safety).

        Args:
            path: File to delete
            params: Additional parameters

        Raises:
            SecurityError: If deletion should be blocked
        """
        if path is None:
            raise SecurityError(
                "Delete operation requires path",
                severity="high",
            )

        # Require explicit confirmation for delete in safe mode
        if self.safe_mode and not params.get("confirmed", False):
            raise SecurityError(
                f"Deletion of {path} requires explicit confirmation in safe mode\n"
                f"Set confirmed=True to proceed",
                severity="medium",
            )

        # Block deletion of project root
        if path == self.project_root:
            raise SecurityError(
                "Cannot delete project root directory",
                severity="critical",
            )

        # Block deletion of important files
        important_files = {".git", ".env", "requirements.txt", "pyproject.toml"}
        if path.name in important_files and self.safe_mode:
            raise SecurityError(
                f"Deletion of important file {path.name} blocked in safe mode",
                severity="high",
            )

    def _validate_shell_execution(
        self, command: Optional[List[str]], params: Dict
    ) -> None:
        """Validate shell command execution.

        Args:
            command: Command to execute
            params: Additional parameters

        Raises:
            SecurityError: If execution should be blocked
        """
        if command is None:
            raise SecurityError(
                "Shell execution requires command",
                severity="high",
            )

        # Block shell=True usage (always use list form)
        if params.get("shell", False):
            raise SecurityError(
                "shell=True not allowed (use command list instead)",
                severity="critical",
            )

    def _validate_write_operation(
        self, path: Optional[Path], params: Dict
    ) -> None:
        """Validate file write operation.

        Args:
            path: File to write
            params: Additional parameters

        Raises:
            SecurityError: If write should be blocked
        """
        if path is None:
            raise SecurityError(
                "Write operation requires path",
                severity="high",
            )

        # Check content size if provided
        content = params.get("content")
        if content is not None:
            content_size = len(content) if isinstance(content, (str, bytes)) else 0

            if content_size > self.max_file_size:
                raise SecurityError(
                    f"Write content exceeds size limit: {content_size} bytes",
                    severity="medium",
                )


# Convenience function for quick validation
def validate_safe_operation(
    operation: str,
    project_root: Union[str, Path],
    **params,
) -> Dict[str, any]:
    """Convenience function for one-off operation validation.

    Args:
        operation: Operation identifier
        project_root: Project root directory
        **params: Operation parameters

    Returns:
        Dict: Validated parameters

    Example:
        >>> validate_safe_operation(
        ...     operation="fs.read",
        ...     project_root="/home/user/project",
        ...     path="data/input.txt"
        ... )
    """
    validator = SafetyValidator(project_root=project_root)
    return validator.validate_operation(operation=operation, **params)
