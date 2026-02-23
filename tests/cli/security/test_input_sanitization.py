"""Security tests for input sanitization (T109).

Tests path validation, string sanitization, and injection prevention
for user-provided inputs.

OWASP A01:2021 - Broken Access Control
OWASP A03:2021 - Injection
"""

import pytest
from pathlib import Path

from promptchain.cli.security.input_sanitizer import (
    InputSanitizer,
    PathValidator,
    SanitizationError,
)


class TestPathValidation:
    """Test path traversal attack prevention."""

    def test_directory_traversal_blocked(self, tmp_path):
        """Test that directory traversal attempts are blocked."""
        validator = PathValidator(base_dir=tmp_path)

        # Attempts to escape base directory
        dangerous_paths = [
            "../../../etc/passwd",
            "../../root/.ssh/id_rsa",
            "../..",
            "./../../../etc",
            "session/../../../etc/passwd",
        ]

        for path in dangerous_paths:
            with pytest.raises(SanitizationError) as exc_info:
                validator.validate_path(path)

            assert exc_info.value.severity == "critical"

    def test_absolute_path_policy(self, tmp_path):
        """Test absolute path allow/deny policy."""
        # Create validator without base_dir for absolute path testing
        validator = PathValidator()

        # Absolute paths blocked by default
        with pytest.raises(SanitizationError):
            validator.validate_path("/tmp/test.txt", allow_absolute=False)

        # Absolute paths allowed when enabled (using a safe path)
        result = validator.validate_path("/tmp/test.txt", allow_absolute=True)
        assert result.is_absolute()

    def test_null_byte_injection_blocked(self, tmp_path):
        """Test that null byte injection is blocked."""
        validator = PathValidator(base_dir=tmp_path)

        # Null byte to bypass path checks
        dangerous_path = "safe.txt\x00../../etc/passwd"

        with pytest.raises(SanitizationError) as exc_info:
            validator.validate_path(dangerous_path)

        assert exc_info.value.severity == "critical"
        assert "null byte" in str(exc_info.value).lower()

    def test_dangerous_path_components_blocked(self, tmp_path):
        """Test that dangerous path components are blocked for relative paths."""
        validator = PathValidator(base_dir=tmp_path)

        # Dangerous components in relative paths
        dangerous_relative_paths = [
            "../../../etc/shadow",
            "sessions/../../root/.bash_history",
            "~/../../etc/passwd",
            "../var/log/auth.log",
        ]

        for path in dangerous_relative_paths:
            with pytest.raises(SanitizationError):
                validator.validate_path(path)

    def test_safe_relative_paths_allowed(self, tmp_path):
        """Test that safe relative paths are allowed."""
        validator = PathValidator(base_dir=tmp_path)

        safe_paths = [
            "sessions/session-123",
            "logs/activity.jsonl",
            "config.yml",
            "agents/researcher.json",
        ]

        for path in safe_paths:
            result = validator.validate_path(path)
            assert result.is_relative_to(tmp_path)

    def test_path_stays_within_base_dir(self, tmp_path):
        """Test that resolved path stays within base directory."""
        validator = PathValidator(base_dir=tmp_path)

        # Create subdirectory
        subdir = tmp_path / "sessions"
        subdir.mkdir()

        # Safe path within base
        safe_path = validator.validate_path("sessions/test.txt")
        assert safe_path.is_relative_to(tmp_path)

        # Attempt to escape via symlink trick (should fail)
        with pytest.raises(SanitizationError):
            validator.validate_path("sessions/../../etc/passwd")

    def test_filename_validation(self, tmp_path):
        """Test filename-only validation (no directories)."""
        validator = PathValidator()

        # Valid filenames
        valid_names = ["session.db", "config.yml", "data-2024.json"]
        for name in valid_names:
            result = validator.validate_filename(name)
            assert result == name

        # Invalid filenames (contain directory separators)
        invalid_names = [
            "../config.yml",
            "sessions/test.db",
            "../../etc/passwd",
            "/etc/shadow",
        ]
        for name in invalid_names:
            with pytest.raises(SanitizationError):
                validator.validate_filename(name)

    def test_filename_length_limit(self):
        """Test that excessively long filenames are blocked."""
        validator = PathValidator()

        # Too long (> 255 chars)
        long_name = "A" * 300 + ".txt"

        with pytest.raises(SanitizationError) as exc_info:
            validator.validate_filename(long_name)

        assert "too long" in str(exc_info.value).lower()

    def test_dangerous_filename_patterns(self):
        """Test that dangerous filenames are blocked."""
        validator = PathValidator()

        dangerous_names = [".", "..", "~"]

        for name in dangerous_names:
            with pytest.raises(SanitizationError):
                validator.validate_filename(name)


class TestInputSanitization:
    """Test input sanitization for various user inputs."""

    def test_name_validation(self):
        """Test session/agent name validation."""
        sanitizer = InputSanitizer()

        # Valid names
        valid_names = ["my-session", "agent_1", "test-123", "a", "A1-_"]
        for name in valid_names:
            result = sanitizer.validate_name(name)
            assert result == name

        # Invalid names
        invalid_names = [
            "my session",  # Space
            "test@123",  # Special char
            "a" * 100,  # Too long
            "",  # Empty
            "test/path",  # Path separator
            "../../etc",  # Traversal
        ]
        for name in invalid_names:
            with pytest.raises(SanitizationError):
                sanitizer.validate_name(name)

    def test_model_name_validation(self):
        """Test model name validation (whitelist approach)."""
        sanitizer = InputSanitizer()

        # Valid models
        valid_models = [
            "openai/gpt-4",
            "anthropic/claude-3-opus",
            "ollama/llama2",
            "gemini/gemini-pro",
            "gpt-4",
            "claude-3-sonnet",
        ]
        for model in valid_models:
            result = sanitizer.validate_model_name(model)
            assert result == model

        # Invalid models (not in whitelist)
        invalid_models = [
            "malicious/model",
            "../../etc/passwd",
            "os.system",
            "A" * 200,  # Too long
        ]
        for model in invalid_models:
            with pytest.raises(SanitizationError):
                sanitizer.validate_model_name(model)

    def test_instruction_chain_validation(self):
        """Test instruction chain validation."""
        sanitizer = InputSanitizer()

        # Valid instruction chains
        valid_chains = [
            ["Analyze: {input}"],
            ["Step 1: {input}", "Step 2: summarize"],
            [{"type": "agentic_step", "objective": "Research", "max_internal_steps": 5}],
            ["String instruction", {"type": "agentic_step", "objective": "Task"}],
        ]
        for chain in valid_chains:
            result = sanitizer.validate_instruction_chain(chain)
            assert len(result) == len(chain)

        # Invalid instruction chains
        invalid_chains = [
            ["__import__('os').system('ls')"],  # Code execution
            ["exec('print(1)')"],  # Exec
            ["eval('1+1')"],  # Eval
            ["subprocess.call(['ls'])"],  # Subprocess
            ["A" * 15000],  # Too long
        ]
        for chain in invalid_chains:
            with pytest.raises(SanitizationError):
                sanitizer.validate_instruction_chain(chain)

    def test_instruction_dict_validation(self):
        """Test dict instruction validation."""
        sanitizer = InputSanitizer()

        # Valid agentic_step
        valid_dict = {
            "type": "agentic_step",
            "objective": "Research topic",
            "max_internal_steps": 8,
        }
        result = sanitizer.validate_instruction_chain([valid_dict])
        assert len(result) == 1

        # Invalid type
        with pytest.raises(SanitizationError):
            sanitizer.validate_instruction_chain([{"type": "malicious"}])

        # Missing type
        with pytest.raises(SanitizationError):
            sanitizer.validate_instruction_chain([{"objective": "test"}])

        # Invalid max_internal_steps
        with pytest.raises(SanitizationError):
            sanitizer.validate_instruction_chain([
                {"type": "agentic_step", "max_internal_steps": 100}  # Too high
            ])

    def test_string_sanitization(self):
        """Test general string sanitization."""
        sanitizer = InputSanitizer()

        # Strip control characters
        dirty_string = "Hello\x00World\x1f!"
        clean_string = sanitizer.sanitize_string(dirty_string)
        assert "\x00" not in clean_string
        assert "\x1f" not in clean_string

        # Preserve newlines and tabs
        string_with_whitespace = "Line1\nLine2\tTabbed"
        result = sanitizer.sanitize_string(string_with_whitespace)
        assert "\n" in result
        assert "\t" in result

        # Enforce length limit
        long_string = "A" * 2000
        with pytest.raises(SanitizationError):
            sanitizer.sanitize_string(long_string, max_length=1000)

    def test_injection_pattern_detection(self):
        """Test detection of common injection patterns."""
        sanitizer = InputSanitizer()

        # Code execution patterns
        dangerous_strings = [
            "__import__('os')",
            "exec('code')",
            "eval('expression')",
            "compile('code', 'file', 'exec')",
            "os.system('ls')",
            "os.exec('ls')",
            "subprocess.call(['ls'])",
            "open('/etc/passwd', 'w')",
        ]

        for dangerous in dangerous_strings:
            with pytest.raises(SanitizationError) as exc_info:
                sanitizer._validate_instruction_string(dangerous, "test")

            assert exc_info.value.severity == "critical"

    def test_collection_size_limits(self):
        """Test that large instruction chains are blocked."""
        sanitizer = InputSanitizer()

        # Too many instructions
        large_chain = ["Step {i}" for i in range(25)]  # Exceeds max=20

        with pytest.raises(SanitizationError):
            sanitizer.validate_instruction_chain(large_chain)

    def test_type_validation(self):
        """Test that incorrect types are rejected."""
        sanitizer = InputSanitizer()

        # Name must be string
        with pytest.raises(SanitizationError):
            sanitizer.validate_name(123)

        # Model must be string
        with pytest.raises(SanitizationError):
            sanitizer.validate_model_name(["openai/gpt-4"])

        # Instruction chain must be list
        with pytest.raises(SanitizationError):
            sanitizer.validate_instruction_chain("not a list")


class TestSecurityDefenseInDepth:
    """Test defense-in-depth security patterns."""

    def test_multiple_validation_layers(self, tmp_path):
        """Test that multiple validation layers catch attacks."""
        # Path validation + filename validation
        path_validator = PathValidator(base_dir=tmp_path)

        # Attack attempt 1: Directory traversal
        with pytest.raises(SanitizationError):
            path_validator.validate_path("../../../etc/passwd")

        # Attack attempt 2: Null byte injection
        with pytest.raises(SanitizationError):
            path_validator.validate_filename("safe.txt\x00../../etc/passwd")

        # Attack attempt 3: Absolute path to sensitive file
        with pytest.raises(SanitizationError):
            path_validator.validate_path("/etc/shadow", allow_absolute=False)

    def test_fail_securely(self):
        """Test that validation failures don't leak information."""
        sanitizer = InputSanitizer()

        # Invalid input should raise exception, not return partial data
        try:
            sanitizer.validate_name("../../etc/passwd")
            assert False, "Should have raised exception"
        except SanitizationError as e:
            # Error message should not contain sensitive path info
            error_msg = str(e).lower()
            assert "alphanumeric" in error_msg or "invalid" in error_msg

    def test_whitelist_approach(self):
        """Test that whitelist approach is used for critical validations."""
        sanitizer = InputSanitizer()

        # Model names: whitelist of allowed prefixes
        allowed_prefix = "openai/gpt-4"
        result = sanitizer.validate_model_name(allowed_prefix)
        assert result == allowed_prefix

        # Anything else is blocked
        with pytest.raises(SanitizationError):
            sanitizer.validate_model_name("unknown/model")

        # Instruction types: whitelist of allowed types
        with pytest.raises(SanitizationError):
            sanitizer.validate_instruction_chain([{"type": "unknown_type"}])
