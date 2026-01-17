"""Unit tests for shell command parsing (T118).

These tests verify command parsing logic:
- ! prefix detection
- !! shell mode toggle detection
- Command string extraction
- Parsing edge cases
"""

import pytest


class TestShellParser:
    """Test shell command parsing logic."""

    @pytest.fixture
    def shell_parser(self):
        """Create shell parser instance (to be implemented)."""
        try:
            from promptchain.cli.shell_executor import ShellCommandParser
            return ShellCommandParser()
        except ImportError:
            pytest.skip("ShellCommandParser not yet implemented (will be in T123)")

    def test_detect_shell_command(self, shell_parser):
        """Unit: Detect ! prefix as shell command.

        Given: String starting with !
        When: Checking if shell command
        Then: Returns True

        Validates:
        - ! prefix detection
        - Single exclamation mark
        - Not confused with other punctuation
        """
        assert shell_parser.is_shell_command("!ls -la") is True
        assert shell_parser.is_shell_command("!echo test") is True
        assert shell_parser.is_shell_command("!") is True  # Edge case

    def test_detect_non_shell_command(self, shell_parser):
        """Unit: Regular text not detected as shell command.

        Given: String without ! prefix
        When: Checking if shell command
        Then: Returns False

        Validates:
        - Regular text distinguished
        - No false positives
        - Exclamation mid-string ignored
        """
        assert shell_parser.is_shell_command("hello world") is False
        assert shell_parser.is_shell_command("ls -la") is False  # No !
        assert shell_parser.is_shell_command("wow! amazing") is False  # ! not at start

    def test_detect_shell_mode_toggle(self, shell_parser):
        """Unit: Detect !! as mode toggle.

        Given: String that is exactly !!
        When: Checking if toggle
        Then: Returns True

        Validates:
        - Double-bang detection
        - Exact match required
        - Whitespace handling
        """
        assert shell_parser.is_shell_mode_toggle("!!") is True
        assert shell_parser.is_shell_mode_toggle("!!  ") is True  # Trailing whitespace
        assert shell_parser.is_shell_mode_toggle("  !!") is True  # Leading whitespace

    def test_not_shell_mode_toggle(self, shell_parser):
        """Unit: Non-!! strings not detected as toggle.

        Given: Various strings
        When: Checking if toggle
        Then: Returns False

        Validates:
        - Single ! not toggle
        - Text after !! not toggle
        - Other punctuation not toggle
        """
        assert shell_parser.is_shell_mode_toggle("!") is False
        assert shell_parser.is_shell_mode_toggle("! !") is False
        assert shell_parser.is_shell_mode_toggle("!! ls") is False  # Has command after
        assert shell_parser.is_shell_mode_toggle("!!!") is False  # Too many

    def test_extract_command_string(self, shell_parser):
        """Unit: Extract command without ! prefix.

        Given: Shell command with !
        When: Extracting command
        Then: Returns command without !

        Validates:
        - ! prefix removed
        - Rest of string preserved
        - Whitespace maintained
        """
        assert shell_parser.extract_command("!ls -la") == "ls -la"
        assert shell_parser.extract_command("!echo hello") == "echo hello"
        assert shell_parser.extract_command("!git status") == "git status"

    def test_extract_command_with_multiple_exclamations(self, shell_parser):
        """Unit: Handle multiple ! in command.

        Given: Command with ! in text (!echo "wow!")
        When: Extracting command
        Then: Only first ! removed

        Validates:
        - Only leading ! removed
        - Internal ! preserved
        - String content intact
        """
        assert shell_parser.extract_command('!echo "wow!"') == 'echo "wow!"'
        assert shell_parser.extract_command('!echo "!!!"') == 'echo "!!!"'

    def test_extract_command_preserves_whitespace(self, shell_parser):
        """Unit: Whitespace in command preserved.

        Given: Command with various whitespace
        When: Extracting command
        Then: Whitespace preserved exactly

        Validates:
        - Leading spaces after ! kept
        - Internal spacing maintained
        - No trimming or normalization
        """
        assert shell_parser.extract_command("!  ls") == "  ls"
        assert shell_parser.extract_command("!echo  'double  space'") == "echo  'double  space'"

    def test_extract_command_multiline(self, shell_parser):
        """Unit: Multi-line commands extracted correctly.

        Given: Command with newlines
        When: Extracting command
        Then: Newlines preserved

        Validates:
        - Multi-line support
        - Newline preservation
        - Complex commands work
        """
        multiline = """!echo "line1"
echo "line2"
echo "line3" """

        extracted = shell_parser.extract_command(multiline)
        assert extracted.startswith('echo "line1"')
        assert "line2" in extracted
        assert "line3" in extracted

    def test_extract_empty_command(self, shell_parser):
        """Unit: Handle just ! with no command.

        Given: String that is just "!"
        When: Extracting command
        Then: Returns empty string

        Validates:
        - Empty command handled
        - No crash on edge case
        - Returns "" not None
        """
        assert shell_parser.extract_command("!") == ""
        assert shell_parser.extract_command("!  ") == "  "  # Just whitespace

    def test_parse_command_with_quotes(self, shell_parser):
        """Unit: Quoted strings in commands parsed.

        Given: Command with single/double quotes
        When: Parsing command
        Then: Quotes preserved in extraction

        Validates:
        - Quote preservation
        - No quote stripping
        - Shell will handle quotes
        """
        assert shell_parser.extract_command('!echo "hello world"') == 'echo "hello world"'
        assert shell_parser.extract_command("!echo 'single quotes'") == "echo 'single quotes'"

    def test_parse_command_with_special_chars(self, shell_parser):
        """Unit: Special characters in commands preserved.

        Given: Command with special chars ($, &, |, etc.)
        When: Parsing command
        Then: All characters preserved

        Validates:
        - Special chars not escaped
        - Shell metacharacters preserved
        - Pipes, redirects, etc. intact
        """
        assert shell_parser.extract_command("!echo $HOME") == "echo $HOME"
        assert shell_parser.extract_command("!ls | grep test") == "ls | grep test"
        assert shell_parser.extract_command("!echo 'test' > file.txt") == "echo 'test' > file.txt"

    def test_is_shell_command_with_whitespace(self, shell_parser):
        """Unit: Whitespace before ! handled correctly.

        Given: Strings with leading whitespace
        When: Checking if shell command
        Then: Leading space ignored for detection

        Validates:
        - Whitespace tolerance
        - User typing flexibility
        - Trim before checking
        """
        assert shell_parser.is_shell_command("  !ls") is True
        assert shell_parser.is_shell_command("\t!echo") is True
        assert shell_parser.is_shell_command("   !!") is True

    def test_extract_command_with_whitespace_prefix(self, shell_parser):
        """Unit: Extract command strips leading whitespace.

        Given: Command with leading whitespace
        When: Extracting command
        Then: Whitespace before ! removed

        Validates:
        - Clean extraction
        - No extra whitespace in result
        - Command ready to execute
        """
        assert shell_parser.extract_command("  !ls") == "ls"
        assert shell_parser.extract_command("\t!echo test") == "echo test"
