"""Input sanitization utilities for PromptChain CLI.

Provides defense-in-depth sanitization for user inputs:
- Path validation (directory traversal prevention)
- String sanitization (injection prevention)
- Instruction chain validation (code execution prevention)

OWASP A01:2021 - Broken Access Control
OWASP A03:2021 - Injection
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class SanitizationError(Exception):
    """Raised when input sanitization fails."""

    def __init__(self, message: str, severity: str = "high"):
        """Initialize sanitization error.

        Args:
            message: Error description
            severity: Security severity level (low, medium, high, critical)
        """
        super().__init__(message)
        self.severity = severity


class PathValidator:
    """Validates file paths to prevent directory traversal attacks.

    OWASP A01:2021 - Broken Access Control

    Example:
        >>> validator = PathValidator(base_dir="/home/user/sessions")
        >>> safe_path = validator.validate_path("session-123/messages.jsonl")
        >>> # Raises SanitizationError if path escapes base_dir
    """

    # Dangerous path components
    DANGEROUS_PATH_COMPONENTS = [
        "..",  # Parent directory
        "~",  # Home directory
        "/etc",  # System config
        "/root",  # Root home
        "/var",  # System var
        "/usr",  # System usr
        "/sys",  # System sys
        "/proc",  # System proc
    ]

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """Initialize path validator.

        Args:
            base_dir: Base directory for relative path validation
        """
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    def validate_path(
        self,
        path: Union[str, Path],
        must_exist: bool = False,
        allow_absolute: bool = False,
    ) -> Path:
        """Validate path for security issues.

        Args:
            path: Path to validate
            must_exist: If True, path must exist
            allow_absolute: If True, allow absolute paths

        Returns:
            Path: Sanitized absolute path

        Raises:
            SanitizationError: If path validation fails
        """
        path_obj = Path(path)

        # Check for null bytes (directory traversal prevention)
        if "\0" in str(path):
            raise SanitizationError(
                "Path contains null bytes",
                severity="critical",
            )

        # Check absolute path policy
        if path_obj.is_absolute() and not allow_absolute:
            raise SanitizationError(
                "Absolute paths not allowed",
                severity="high",
            )

        # Check for dangerous components (only for relative paths or when not allowing absolute)
        if not allow_absolute or not path_obj.is_absolute():
            path_str = str(path)
            for dangerous in self.DANGEROUS_PATH_COMPONENTS:
                if dangerous in path_str:
                    raise SanitizationError(
                        f"Path contains dangerous component: {dangerous}",
                        severity="critical",
                    )

        # Resolve to absolute path
        if self.base_dir:
            # Relative to base directory
            absolute_path = (self.base_dir / path_obj).resolve()

            # Ensure path stays within base_dir (directory traversal prevention)
            try:
                absolute_path.relative_to(self.base_dir)
            except ValueError:
                raise SanitizationError(
                    f"Path escapes base directory: {path}",
                    severity="critical",
                )
        else:
            # Resolve to absolute
            absolute_path = path_obj.resolve()

        # Check existence if required
        if must_exist and not absolute_path.exists():
            raise SanitizationError(
                f"Path does not exist: {path}",
                severity="low",
            )

        return absolute_path

    def validate_filename(self, filename: str) -> str:
        """Validate filename (no directory components).

        Args:
            filename: Filename to validate

        Returns:
            str: Validated filename

        Raises:
            SanitizationError: If validation fails
        """
        # Check for null bytes
        if "\0" in filename:
            raise SanitizationError(
                "Filename contains null bytes",
                severity="critical",
            )

        # Check for directory separators
        if "/" in filename or "\\" in filename:
            raise SanitizationError(
                "Filename contains directory separators",
                severity="high",
            )

        # Check for dangerous names
        if filename in (".", "..", "~"):
            raise SanitizationError(
                f"Dangerous filename: {filename}",
                severity="high",
            )

        # Check length
        if len(filename) > 255:
            raise SanitizationError(
                "Filename too long (max: 255 chars)",
                severity="medium",
            )

        return filename


class InputSanitizer:
    """Sanitizes various user inputs for security.

    Provides centralized sanitization for:
    - Agent instruction chains (code execution prevention)
    - Session names (injection prevention)
    - Model names (validation)
    - Tool names (whitelist validation)

    OWASP A03:2021 - Injection Prevention
    """

    # Allowed characters in session/agent names
    NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

    # Allowed model prefixes (whitelist)
    ALLOWED_MODEL_PREFIXES = [
        "openai/",
        "anthropic/",
        "ollama/",
        "gemini/",
        "gpt-",  # OpenAI shorthand
        "claude-",  # Anthropic shorthand
    ]

    # Dangerous patterns in instruction chains
    DANGEROUS_INSTRUCTION_PATTERNS = [
        re.compile(r"__import__"),  # Python imports
        re.compile(r"exec\s*\("),  # Python exec
        re.compile(r"eval\s*\("),  # Python eval
        re.compile(r"compile\s*\("),  # Code compilation
        re.compile(r"os\.(system|popen|exec)"),  # OS commands
        re.compile(r"subprocess\."),  # Subprocess module
        re.compile(r"open\s*\(.*,\s*['\"]w"),  # File writes
    ]

    # Maximum lengths
    MAX_NAME_LENGTH = 64
    MAX_DESCRIPTION_LENGTH = 500
    MAX_INSTRUCTION_LENGTH = 10000

    def validate_name(
        self, name: str, context: str = "name", max_length: int = MAX_NAME_LENGTH
    ) -> str:
        """Validate name (session, agent, etc.).

        Args:
            name: Name to validate
            context: Context for error message
            max_length: Maximum length

        Returns:
            str: Validated name

        Raises:
            SanitizationError: If validation fails
        """
        # Check type
        if not isinstance(name, str):
            raise SanitizationError(
                f"{context} must be string, got {type(name).__name__}",
                severity="high",
            )

        # Check length
        if not (1 <= len(name) <= max_length):
            raise SanitizationError(
                f"{context} length must be 1-{max_length} chars",
                severity="medium",
            )

        # Check format
        if not self.NAME_PATTERN.match(name):
            raise SanitizationError(
                f"{context} must contain only alphanumeric, dash, underscore",
                severity="high",
            )

        return name

    def validate_model_name(self, model: str) -> str:
        """Validate model name.

        Args:
            model: Model name to validate

        Returns:
            str: Validated model name

        Raises:
            SanitizationError: If validation fails
        """
        if not isinstance(model, str):
            raise SanitizationError(
                f"Model must be string, got {type(model).__name__}",
                severity="high",
            )

        # Check length
        if len(model) > 100:
            raise SanitizationError(
                "Model name too long (max: 100 chars)",
                severity="medium",
            )

        # Check against whitelist
        if not any(model.startswith(prefix) for prefix in self.ALLOWED_MODEL_PREFIXES):
            raise SanitizationError(
                f"Invalid model prefix. Allowed: {self.ALLOWED_MODEL_PREFIXES}",
                severity="high",
            )

        return model

    def validate_instruction_chain(
        self, instructions: List[Union[str, Dict]]
    ) -> List[Union[str, Dict]]:
        """Validate instruction chain for dangerous patterns.

        Args:
            instructions: List of instruction strings/dicts

        Returns:
            List[Union[str, Dict]]: Validated instructions

        Raises:
            SanitizationError: If validation fails
        """
        if not isinstance(instructions, list):
            raise SanitizationError(
                f"Instruction chain must be list, got {type(instructions).__name__}",
                severity="high",
            )

        # Check collection size
        if len(instructions) > 20:
            raise SanitizationError(
                "Instruction chain too long (max: 20 steps)",
                severity="medium",
            )

        validated: List[Union[str, Dict]] = []
        for i, instruction in enumerate(instructions):
            if isinstance(instruction, str):
                # Validate string instruction
                validated_instruction = self._validate_instruction_string(
                    instruction, f"instruction[{i}]"
                )
                validated.append(validated_instruction)

            elif isinstance(instruction, dict):
                # Validate dict instruction (agentic steps, etc.)
                validated_instruction = self._validate_instruction_dict(
                    instruction, f"instruction[{i}]"
                )
                validated.append(validated_instruction)

            else:
                raise SanitizationError(
                    f"Invalid instruction type at index {i}: {type(instruction).__name__}",
                    severity="high",
                )

        return validated

    def _validate_instruction_string(self, instruction: str, context: str) -> str:
        """Validate string instruction.

        Args:
            instruction: Instruction string
            context: Context for error message

        Returns:
            str: Validated instruction

        Raises:
            SanitizationError: If validation fails
        """
        # Check length
        if len(instruction) > self.MAX_INSTRUCTION_LENGTH:
            raise SanitizationError(
                f"{context} too long: {len(instruction)} chars (max: {self.MAX_INSTRUCTION_LENGTH})",
                severity="medium",
            )

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_INSTRUCTION_PATTERNS:
            if pattern.search(instruction):
                raise SanitizationError(
                    f"Dangerous pattern in {context}: {pattern.pattern}",
                    severity="critical",
                )

        return instruction

    def _validate_instruction_dict(self, instruction: Dict, context: str) -> Dict:
        """Validate dict instruction (agentic step, etc.).

        Args:
            instruction: Instruction dictionary
            context: Context for error message

        Returns:
            Dict: Validated instruction

        Raises:
            SanitizationError: If validation fails
        """
        # Check required fields
        if "type" not in instruction:
            raise SanitizationError(
                f"{context} missing required field: type",
                severity="high",
            )

        # Validate type field
        allowed_types = {"agentic_step", "function"}
        if instruction["type"] not in allowed_types:
            raise SanitizationError(
                f"Invalid instruction type in {context}: {instruction['type']}. Allowed: {allowed_types}",
                severity="high",
            )

        # Validate agentic_step specific fields
        if instruction["type"] == "agentic_step":
            if "objective" in instruction:
                self._validate_instruction_string(
                    instruction["objective"], f"{context}.objective"
                )

            if "max_internal_steps" in instruction:
                if not isinstance(instruction["max_internal_steps"], int):
                    raise SanitizationError(
                        f"{context}.max_internal_steps must be integer",
                        severity="medium",
                    )
                # Allow up to 50 internal steps for complex agentic tasks
                if not (1 <= instruction["max_internal_steps"] <= 50):
                    raise SanitizationError(
                        f"{context}.max_internal_steps must be 1-50",
                        severity="medium",
                    )

        return instruction

    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """Sanitize general string input.

        Args:
            value: String to sanitize
            max_length: Maximum length

        Returns:
            str: Sanitized string

        Raises:
            SanitizationError: If validation fails
        """
        if not isinstance(value, str):
            raise SanitizationError(
                f"Value must be string, got {type(value).__name__}",
                severity="medium",
            )

        # Check length
        if len(value) > max_length:
            raise SanitizationError(
                f"String too long: {len(value)} chars (max: {max_length})",
                severity="medium",
            )

        # Strip control characters (except newlines and tabs)
        sanitized = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", value)

        return sanitized


# Convenience instances
path_validator = PathValidator()
input_sanitizer = InputSanitizer()
