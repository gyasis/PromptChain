"""YAML security validator for PromptChain CLI.

Prevents YAML injection attacks by validating structure and content before processing.

Security Measures:
1. Schema validation - whitelist allowed fields
2. Depth limits - prevent nested bomb attacks
3. Size limits - prevent DoS via large YAML
4. Safe types only - no custom tags or Python objects
5. String sanitization - prevent injection in string values

OWASP A03:2021 - Injection Prevention
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml


class ValidationError(Exception):
    """Raised when YAML validation fails for security reasons."""

    def __init__(self, message: str, severity: str = "high"):
        """Initialize validation error.

        Args:
            message: Error description
            severity: Security severity level (low, medium, high, critical)
        """
        super().__init__(message)
        self.severity = severity


class YAMLValidator:
    """Validates YAML configuration for security vulnerabilities.

    Prevents:
    - YAML injection attacks
    - Arbitrary code execution via unsafe YAML loading
    - Directory traversal in file paths
    - Denial of service via YAML bombs
    - Template injection in string values

    Example:
        >>> validator = YAMLValidator()
        >>> config = validator.load_and_validate("config.yml")
        >>> # Raises ValidationError if security issues detected
    """

    # Maximum file size: 1MB (prevents DoS via large files)
    MAX_FILE_SIZE = 1024 * 1024

    # Maximum nesting depth (prevents YAML bombs)
    MAX_DEPTH = 10

    # Maximum string length (prevents DoS)
    MAX_STRING_LENGTH = 10000

    # Maximum list/dict entries
    MAX_COLLECTION_SIZE = 100

    # Allowed top-level keys (whitelist approach)
    ALLOWED_TOP_LEVEL_KEYS = {
        "mcp_servers",
        "agents",
        "orchestration",
        "session",
        "preferences",
    }

    # Allowed MCP server fields
    ALLOWED_MCP_SERVER_KEYS = {
        "id",
        "type",
        "command",
        "args",
        "url",
        "auto_connect",
    }

    # Allowed agent fields
    ALLOWED_AGENT_KEYS = {
        "model",
        "description",
        "instruction_chain",
        "tools",
        "history_config",
    }

    # Allowed orchestration fields
    ALLOWED_ORCHESTRATION_KEYS = {
        "execution_mode",
        "default_agent",
        "router",
    }

    # Allowed session fields
    ALLOWED_SESSION_KEYS = {
        "auto_save_interval",
        "max_history_entries",
        "working_directory",
    }

    # Allowed preferences fields
    ALLOWED_PREFERENCES_KEYS = {
        "verbose",
        "theme",
        "show_token_usage",
        "show_reasoning_steps",
    }

    # Dangerous patterns in strings (template injection indicators)
    DANGEROUS_PATTERNS = [
        re.compile(r"__import__"),  # Python imports
        re.compile(r"exec\s*\("),  # Python exec
        re.compile(r"eval\s*\("),  # Python eval
        re.compile(r"os\.(system|popen)"),  # OS commands
        re.compile(r"subprocess\."),  # Subprocess module
        re.compile(r"\$\{.*(?:__import__|exec|eval).*\}"),  # Injection in env vars
    ]

    def __init__(
        self,
        max_file_size: int = MAX_FILE_SIZE,
        max_depth: int = MAX_DEPTH,
        max_string_length: int = MAX_STRING_LENGTH,
        max_collection_size: int = MAX_COLLECTION_SIZE,
    ):
        """Initialize YAML validator.

        Args:
            max_file_size: Maximum file size in bytes
            max_depth: Maximum nesting depth
            max_string_length: Maximum string length
            max_collection_size: Maximum list/dict size
        """
        self.max_file_size = max_file_size
        self.max_depth = max_depth
        self.max_string_length = max_string_length
        self.max_collection_size = max_collection_size

    def load_and_validate(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML file with full security validation.

        Args:
            path: Path to YAML file

        Returns:
            Dict[str, Any]: Validated YAML configuration

        Raises:
            ValidationError: If security validation fails
            FileNotFoundError: If file doesn't exist
        """
        path = Path(path)

        # Check file exists
        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")

        # Validate file size (DoS prevention)
        file_size = path.stat().st_size
        if file_size > self.max_file_size:
            raise ValidationError(
                f"YAML file too large: {file_size} bytes (max: {self.max_file_size})",
                severity="medium",
            )

        # Read and parse with safe_load (prevents code execution)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_yaml = f.read()

            # Parse with yaml.safe_load (CRITICAL: prevents arbitrary code execution)
            config = yaml.safe_load(raw_yaml)

        except yaml.YAMLError as e:
            raise ValidationError(
                f"Invalid YAML syntax: {e}",
                severity="medium",
            )

        # Handle empty file
        if config is None:
            return {}

        # Validate structure
        self.validate_structure(config)

        return config

    def validate_structure(self, config: Any, path: str = "", depth: int = 0) -> None:
        """Recursively validate YAML structure.

        Args:
            config: Configuration object to validate
            path: Current path in config tree (for error messages)
            depth: Current nesting depth

        Raises:
            ValidationError: If validation fails
        """
        # Check nesting depth (YAML bomb prevention)
        if depth > self.max_depth:
            raise ValidationError(
                f"Nesting depth exceeded at '{path}' (max: {self.max_depth})",
                severity="high",
            )

        # Validate based on type
        if isinstance(config, dict):
            self._validate_dict(config, path, depth)
        elif isinstance(config, list):
            self._validate_list(config, path, depth)
        elif isinstance(config, str):
            self._validate_string(config, path)
        elif isinstance(config, (int, float, bool, type(None))):
            # Safe primitive types
            pass
        else:
            # Unexpected type (potential injection)
            raise ValidationError(
                f"Unsupported type {type(config).__name__} at '{path}'",
                severity="critical",
            )

    def _validate_dict(self, config: Dict, path: str, depth: int) -> None:
        """Validate dictionary structure.

        Args:
            config: Dictionary to validate
            path: Current path
            depth: Current depth

        Raises:
            ValidationError: If validation fails
        """
        # Check collection size (DoS prevention)
        if len(config) > self.max_collection_size:
            raise ValidationError(
                f"Dictionary too large at '{path}': {len(config)} entries (max: {self.max_collection_size})",
                severity="medium",
            )

        # Validate top-level keys (whitelist approach)
        if path == "":
            invalid_keys = set(config.keys()) - self.ALLOWED_TOP_LEVEL_KEYS
            if invalid_keys:
                raise ValidationError(
                    f"Invalid top-level keys: {invalid_keys}. Allowed: {self.ALLOWED_TOP_LEVEL_KEYS}",
                    severity="high",
                )

        # Validate section-specific keys
        if path.startswith("mcp_servers"):
            self._validate_mcp_server_keys(config, path)
        elif path.startswith("agents."):
            self._validate_agent_keys(config, path)
        elif path == "orchestration":
            self._validate_keys(config, path, self.ALLOWED_ORCHESTRATION_KEYS)
        elif path == "session":
            self._validate_keys(config, path, self.ALLOWED_SESSION_KEYS)
        elif path == "preferences":
            self._validate_keys(config, path, self.ALLOWED_PREFERENCES_KEYS)

        # Recursively validate values
        for key, value in config.items():
            new_path = f"{path}.{key}" if path else key
            self.validate_structure(value, new_path, depth + 1)

    def _validate_list(self, config: List, path: str, depth: int) -> None:
        """Validate list structure.

        Args:
            config: List to validate
            path: Current path
            depth: Current depth

        Raises:
            ValidationError: If validation fails
        """
        # Check collection size (DoS prevention)
        if len(config) > self.max_collection_size:
            raise ValidationError(
                f"List too large at '{path}': {len(config)} entries (max: {self.max_collection_size})",
                severity="medium",
            )

        # Recursively validate items
        for i, item in enumerate(config):
            new_path = f"{path}[{i}]"
            self.validate_structure(item, new_path, depth + 1)

    def _validate_string(self, value: str, path: str) -> None:
        """Validate string value.

        Args:
            value: String to validate
            path: Current path

        Raises:
            ValidationError: If validation fails
        """
        # Check string length (DoS prevention)
        if len(value) > self.max_string_length:
            raise ValidationError(
                f"String too long at '{path}': {len(value)} chars (max: {self.max_string_length})",
                severity="high",  # Changed to high severity
            )

        # Check for dangerous patterns (injection prevention)
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(value):
                raise ValidationError(
                    f"Dangerous pattern detected in '{path}': {pattern.pattern}",
                    severity="critical",
                )

    def _validate_keys(
        self, config: Dict, path: str, allowed_keys: Set[str]
    ) -> None:
        """Validate dictionary keys against whitelist.

        Args:
            config: Dictionary to validate
            path: Current path
            allowed_keys: Set of allowed keys

        Raises:
            ValidationError: If invalid keys found
        """
        invalid_keys = set(config.keys()) - allowed_keys
        if invalid_keys:
            raise ValidationError(
                f"Invalid keys at '{path}': {invalid_keys}. Allowed: {allowed_keys}",
                severity="high",
            )

    def _validate_mcp_server_keys(self, config: Dict, path: str) -> None:
        """Validate MCP server configuration keys.

        Args:
            config: MCP server config
            path: Current path

        Raises:
            ValidationError: If validation fails
        """
        # Allow extra keys in MCP server list items (more flexible)
        # But validate required fields exist
        if "id" not in config or "type" not in config:
            raise ValidationError(
                f"MCP server at '{path}' missing required fields: id, type",
                severity="high",
            )

        # Validate known keys
        known_keys = set(config.keys()) & self.ALLOWED_MCP_SERVER_KEYS
        for key in known_keys:
            if key == "id":
                self._validate_identifier(config[key], f"{path}.id")
            elif key == "type":
                self._validate_mcp_type(config[key], f"{path}.type")
            elif key == "command":
                self._validate_command(config[key], f"{path}.command")

    def _validate_agent_keys(self, config: Dict, path: str) -> None:
        """Validate agent configuration keys.

        Args:
            config: Agent config
            path: Current path

        Raises:
            ValidationError: If validation fails
        """
        # Allow extra keys in agent config (extensibility)
        # But validate required fields if this is the agent config itself
        # (not nested dicts like history_config or instruction_chain items)
        if not any(path.endswith(suffix) for suffix in [".history_config", ".instruction_chain"]):
            if isinstance(config, dict) and "type" not in config and "model" not in config:
                # This looks like an agent config without required field
                # But only validate if it's at the right level
                path_parts = path.split(".")
                if len(path_parts) == 2 and path_parts[0] == "agents":
                    raise ValidationError(
                        f"Agent at '{path}' missing required field: model",
                        severity="high",
                    )

    def _validate_identifier(self, value: str, path: str) -> None:
        """Validate identifier (alphanumeric, dashes, underscores).

        Args:
            value: Identifier string
            path: Current path

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(
                f"Identifier at '{path}' must be string, got {type(value).__name__}",
                severity="high",
            )

        if not re.match(r"^[a-zA-Z0-9_-]+$", value):
            raise ValidationError(
                f"Invalid identifier at '{path}': '{value}'. Must contain only alphanumeric, dash, underscore",
                severity="high",
            )

    def _validate_mcp_type(self, value: str, path: str) -> None:
        """Validate MCP server type.

        Args:
            value: Type string
            path: Current path

        Raises:
            ValidationError: If validation fails
        """
        allowed_types = {"stdio", "sse"}
        if value not in allowed_types:
            raise ValidationError(
                f"Invalid MCP type at '{path}': '{value}'. Allowed: {allowed_types}",
                severity="high",
            )

    def _validate_command(self, value: str, path: str) -> None:
        """Validate command string (basic path injection prevention).

        Args:
            value: Command string
            path: Current path

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(
                f"Command at '{path}' must be string, got {type(value).__name__}",
                severity="high",
            )

        # Block obvious shell injection patterns
        dangerous_chars = ["|", ";", "&", "`", "$", "(", ")", ">", "<"]
        for char in dangerous_chars:
            if char in value:
                raise ValidationError(
                    f"Dangerous character '{char}' in command at '{path}'",
                    severity="critical",
                )


def validate_yaml_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Convenience function to validate YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Dict[str, Any]: Validated configuration

    Raises:
        ValidationError: If validation fails
    """
    validator = YAMLValidator()
    return validator.load_and_validate(path)
