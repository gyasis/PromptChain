"""Tests for SafetyValidator - Basic validation logic.

Tests core validation functionality:
- Path validation
- Command validation
- File size validation
- Operation validation
- Safe/unsafe mode differences

For attack vector tests, see: tests/cli/tools/security/test_safety_attacks.py
"""

import pytest
import tempfile
from pathlib import Path

from promptchain.cli.tools.safety import (
    SafetyValidator,
    SecurityError,
    validate_safe_operation,
)


class TestSafetyValidatorInit:
    """Test SafetyValidator initialization."""

    def test_init_with_valid_project_root(self, tmp_path):
        """Test initialization with valid project root."""
        validator = SafetyValidator(project_root=tmp_path)

        assert validator.project_root == tmp_path.resolve()
        assert validator.safe_mode is True

    def test_init_safe_mode_limits(self, tmp_path):
        """Test that safe mode uses stricter limits."""
        safe_validator = SafetyValidator(project_root=tmp_path, safe_mode=True)
        unsafe_validator = SafetyValidator(project_root=tmp_path, safe_mode=False)

        # Safe mode should have lower limits
        assert safe_validator.max_file_size < unsafe_validator.max_file_size
        assert safe_validator.max_timeout < unsafe_validator.max_timeout

    def test_init_custom_limits(self, tmp_path):
        """Test initialization with custom limits."""
        validator = SafetyValidator(
            project_root=tmp_path,
            max_file_size=1024 * 1024,  # 1MB
            max_timeout=30,  # 30 seconds
        )

        assert validator.max_file_size == 1024 * 1024
        assert validator.max_timeout == 30

    def test_init_nonexistent_project_root(self, tmp_path):
        """Test initialization with nonexistent project root fails."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(SecurityError) as exc_info:
            SafetyValidator(project_root=nonexistent)

        assert exc_info.value.severity == "critical"
        assert "does not exist" in str(exc_info.value)


class TestPathValidation:
    """Test path validation functionality."""

    def test_validate_safe_relative_path(self, tmp_path):
        """Test validation of safe relative path."""
        validator = SafetyValidator(project_root=tmp_path)

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        validated = validator.validate_path("test.txt", must_exist=True)

        assert validated == test_file
        assert validated.is_relative_to(tmp_path)

    def test_validate_safe_absolute_path(self, tmp_path):
        """Test validation of safe absolute path within project root."""
        validator = SafetyValidator(project_root=tmp_path)

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        validated = validator.validate_path(test_file, must_exist=True)

        assert validated == test_file

    def test_validate_nested_path(self, tmp_path):
        """Test validation of nested directory path."""
        validator = SafetyValidator(project_root=tmp_path)

        # Create nested structure
        nested_dir = tmp_path / "src" / "utils"
        nested_dir.mkdir(parents=True)
        test_file = nested_dir / "helper.py"
        test_file.write_text("# test")

        validated = validator.validate_path("src/utils/helper.py", must_exist=True)

        assert validated == test_file

    def test_validate_nonexistent_path_allowed(self, tmp_path):
        """Test that nonexistent paths are allowed when must_exist=False."""
        validator = SafetyValidator(project_root=tmp_path)

        # Should not raise for nonexistent file
        validated = validator.validate_path("new_file.txt", must_exist=False)

        assert validated == tmp_path / "new_file.txt"

    def test_validate_nonexistent_path_blocked(self, tmp_path):
        """Test that nonexistent paths are blocked when must_exist=True."""
        validator = SafetyValidator(project_root=tmp_path)

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_path("nonexistent.txt", must_exist=True)

        assert exc_info.value.severity == "low"
        assert "does not exist" in str(exc_info.value)

    def test_symlink_within_project(self, tmp_path):
        """Test that symlinks within project are allowed."""
        validator = SafetyValidator(project_root=tmp_path)

        # Create target and symlink
        target = tmp_path / "target.txt"
        target.write_text("content")

        symlink = tmp_path / "link.txt"
        symlink.symlink_to(target)

        # Should validate successfully
        validated = validator.validate_path(symlink)

        assert validated.resolve() == target


class TestCommandValidation:
    """Test command validation functionality."""

    def test_validate_safe_command(self, tmp_path):
        """Test validation of safe whitelisted command."""
        validator = SafetyValidator(project_root=tmp_path)

        # Should not raise
        validator.validate_command(["ls", "-la"])
        validator.validate_command(["git", "status"])
        validator.validate_command(["pytest", "tests/"])

    def test_validate_empty_command(self, tmp_path):
        """Test that empty commands are blocked."""
        validator = SafetyValidator(project_root=tmp_path)

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_command([])

        assert exc_info.value.severity == "medium"

    def test_validate_command_safe_mode_whitelist(self, tmp_path):
        """Test that safe mode enforces command whitelist."""
        safe_validator = SafetyValidator(project_root=tmp_path, safe_mode=True)

        # Safe command should pass
        safe_validator.validate_command(["ls", "-la"])

        # Unsafe command should fail
        with pytest.raises(SecurityError) as exc_info:
            safe_validator.validate_command(["unknown_command"])

        assert exc_info.value.severity == "high"
        assert "not in safe whitelist" in str(exc_info.value)

    def test_validate_command_unsafe_mode(self, tmp_path):
        """Test that unsafe mode allows non-whitelisted commands."""
        unsafe_validator = SafetyValidator(project_root=tmp_path, safe_mode=False)

        # Should allow non-whitelisted commands (but still block dangerous ones)
        unsafe_validator.validate_command(["custom_script.sh"])


class TestFileSizeValidation:
    """Test file size validation."""

    def test_validate_small_file(self, tmp_path):
        """Test that small files pass validation."""
        validator = SafetyValidator(project_root=tmp_path)

        # Create small file (1KB)
        small_file = tmp_path / "small.txt"
        small_file.write_text("x" * 1024)

        # Should not raise
        validator.validate_file_size(small_file)

    def test_validate_large_file_blocked(self, tmp_path):
        """Test that oversized files are blocked."""
        # Use very low limit for testing
        validator = SafetyValidator(
            project_root=tmp_path,
            max_file_size=1024,  # 1KB limit
        )

        # Create file exceeding limit (2KB)
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * 2048)

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_file_size(large_file)

        assert exc_info.value.severity == "medium"
        assert "exceeds size limit" in str(exc_info.value)

    def test_validate_nonexistent_file_allowed(self, tmp_path):
        """Test that nonexistent files don't trigger size validation."""
        validator = SafetyValidator(project_root=tmp_path)

        # Should not raise for nonexistent file
        nonexistent = tmp_path / "new.txt"
        validator.validate_file_size(nonexistent)


class TestFileExtensionValidation:
    """Test file extension validation."""

    def test_dangerous_extension_execute_blocked(self, tmp_path):
        """Test that executing files with dangerous extensions is blocked."""
        validator = SafetyValidator(project_root=tmp_path)

        script_file = tmp_path / "script.sh"

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_file_extension(script_file, "execute")

        assert exc_info.value.severity == "high"

    def test_dangerous_extension_write_blocked_safe_mode(self, tmp_path):
        """Test that writing dangerous extensions is blocked in safe mode."""
        safe_validator = SafetyValidator(project_root=tmp_path, safe_mode=True)

        script_file = tmp_path / "script.sh"

        with pytest.raises(SecurityError) as exc_info:
            safe_validator.validate_file_extension(script_file, "write")

        assert exc_info.value.severity == "medium"

    def test_dangerous_extension_write_allowed_unsafe_mode(self, tmp_path):
        """Test that writing dangerous extensions is allowed in unsafe mode."""
        unsafe_validator = SafetyValidator(project_root=tmp_path, safe_mode=False)

        script_file = tmp_path / "script.sh"

        # Should not raise in unsafe mode
        unsafe_validator.validate_file_extension(script_file, "write")

    def test_safe_extension_always_allowed(self, tmp_path):
        """Test that safe extensions are always allowed."""
        validator = SafetyValidator(project_root=tmp_path)

        safe_file = tmp_path / "data.txt"

        # Should not raise for any operation
        validator.validate_file_extension(safe_file, "execute")
        validator.validate_file_extension(safe_file, "write")


class TestOperationValidation:
    """Test comprehensive operation validation."""

    def test_validate_read_operation(self, tmp_path):
        """Test validation of read operation."""
        validator = SafetyValidator(project_root=tmp_path)

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        validated = validator.validate_operation(
            operation="fs.read",
            path="test.txt",
        )

        assert validated["operation"] == "fs.read"
        assert validated["path"] == test_file
        assert "timeout" in validated

    def test_validate_write_operation(self, tmp_path):
        """Test validation of write operation."""
        validator = SafetyValidator(project_root=tmp_path)

        validated = validator.validate_operation(
            operation="fs.write",
            path="output.txt",
            content="test content",
        )

        assert validated["operation"] == "fs.write"
        assert validated["path"] == tmp_path / "output.txt"

    def test_validate_delete_operation_requires_confirmation(self, tmp_path):
        """Test that delete operations require confirmation in safe mode."""
        safe_validator = SafetyValidator(project_root=tmp_path, safe_mode=True)

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Without confirmation should fail
        with pytest.raises(SecurityError) as exc_info:
            safe_validator.validate_operation(
                operation="fs.delete",
                path="test.txt",
            )

        assert "confirmation" in str(exc_info.value).lower()

        # With confirmation should pass
        validated = safe_validator.validate_operation(
            operation="fs.delete",
            path="test.txt",
            confirmed=True,
        )

        assert validated["path"] == test_file

    def test_validate_shell_operation(self, tmp_path):
        """Test validation of shell execution operation."""
        validator = SafetyValidator(project_root=tmp_path)

        validated = validator.validate_operation(
            operation="shell.execute",
            command=["ls", "-la"],
        )

        assert validated["command"] == ["ls", "-la"]

    def test_validate_timeout_limit(self, tmp_path):
        """Test that excessive timeouts are blocked."""
        validator = SafetyValidator(
            project_root=tmp_path,
            max_timeout=60,
        )

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_operation(
                operation="shell.execute",
                command=["ls", "-la"],  # Use safe command
                timeout=120,  # Exceeds limit
            )

        assert "timeout exceeds limit" in str(exc_info.value).lower()

    def test_validate_shell_no_shell_true(self, tmp_path):
        """Test that shell=True is blocked."""
        validator = SafetyValidator(project_root=tmp_path)

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_operation(
                operation="shell.execute",
                command=["ls"],
                shell=True,
            )

        assert exc_info.value.severity == "critical"
        assert "shell=True" in str(exc_info.value)


class TestConvenienceFunction:
    """Test convenience function."""

    def test_validate_safe_operation(self, tmp_path):
        """Test validate_safe_operation convenience function."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        validated = validate_safe_operation(
            operation="fs.read",
            project_root=tmp_path,
            path="test.txt",
        )

        assert validated["path"] == test_file
        assert validated["operation"] == "fs.read"


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_file_read_workflow(self, tmp_path):
        """Test complete file read validation workflow."""
        validator = SafetyValidator(project_root=tmp_path)

        # Create test file
        test_file = tmp_path / "data" / "input.txt"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("test data")

        # Validate read operation
        validated = validator.validate_operation(
            operation="fs.read",
            path="data/input.txt",
        )

        # Should pass all validations
        assert validated["path"].exists()
        assert validated["path"].is_file()

    def test_file_write_workflow(self, tmp_path):
        """Test complete file write validation workflow."""
        validator = SafetyValidator(project_root=tmp_path)

        # Validate write operation
        validated = validator.validate_operation(
            operation="fs.write",
            path="output/results.txt",
            content="output data",
        )

        # Should pass validation even if directory doesn't exist yet
        assert validated["path"] == tmp_path / "output" / "results.txt"

    def test_command_execution_workflow(self, tmp_path):
        """Test complete command execution validation workflow."""
        validator = SafetyValidator(project_root=tmp_path)

        # Validate git status command
        validated = validator.validate_operation(
            operation="shell.execute",
            command=["git", "status"],
            timeout=30,
        )

        assert validated["command"] == ["git", "status"]
        assert validated["timeout"] == 30
