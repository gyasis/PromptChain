"""Contract tests for file reference parsing (T091).

These tests define the contract for how the CLI should parse file and directory
references using @ syntax (e.g., @README.md, @src/).
"""

import pytest
from pathlib import Path
import tempfile
import shutil


class TestFileReferenceParsing:
    """Test file and directory reference parsing with @ syntax."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory with test files."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create test files
        (temp_dir / "README.md").write_text("# Test Project\n\nThis is a test.")
        (temp_dir / "config.json").write_text('{"setting": "value"}')

        # Create src directory with files
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("def main():\n    print('Hello')")
        (src_dir / "utils.py").write_text("def helper():\n    return 42")

        # Create nested directory
        nested = src_dir / "models"
        nested.mkdir()
        (nested / "user.py").write_text("class User:\n    pass")

        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def file_reference_parser(self):
        """Create FileReferenceParser instance (to be implemented)."""
        try:
            from promptchain.cli.utils.file_reference_parser import FileReferenceParser
            return FileReferenceParser()
        except ImportError:
            pytest.skip("FileReferenceParser not yet implemented (will be in T096)")

    def test_parse_single_file(self, file_reference_parser, temp_project_dir):
        """Contract: Parse single file reference @README.md.

        Given: User message with @README.md reference
        When: Parser processes the message
        Then: Returns parsed reference with file path

        Validates:
        - @ syntax recognized
        - File path resolved correctly
        - Reference type identified as 'file'
        """
        user_input = "Please analyze @README.md and summarize it"

        parsed = file_reference_parser.parse_references(
            user_input,
            working_directory=temp_project_dir
        )

        # Should find one reference
        assert len(parsed.references) == 1

        ref = parsed.references[0]
        assert ref.type == "file"
        assert ref.original_text == "@README.md"
        assert ref.resolved_path == temp_project_dir / "README.md"
        assert ref.exists is True

        # Message should be returned with reference preserved
        assert "@README.md" in parsed.message

    def test_parse_multiple_files(self, file_reference_parser, temp_project_dir):
        """Contract: Parse multiple file references in one message.

        Given: User message with multiple @ references
        When: Parser processes the message
        Then: Returns all parsed references

        Validates:
        - Multiple references detected
        - Each resolved independently
        - Order preserved
        """
        user_input = "Compare @README.md with @config.json and @src/main.py"

        parsed = file_reference_parser.parse_references(
            user_input,
            working_directory=temp_project_dir
        )

        # Should find three references
        assert len(parsed.references) == 3

        # Verify each reference
        assert parsed.references[0].original_text == "@README.md"
        assert parsed.references[1].original_text == "@config.json"
        assert parsed.references[2].original_text == "@src/main.py"

        # All should exist
        assert all(ref.exists for ref in parsed.references)

        # All should be files
        assert all(ref.type == "file" for ref in parsed.references)

    def test_parse_directory(self, file_reference_parser, temp_project_dir):
        """Contract: Parse directory reference @src/.

        Given: User message with directory reference
        When: Parser processes the message
        Then: Returns parsed reference with directory type

        Validates:
        - Directory syntax (@src/) recognized
        - Type identified as 'directory'
        - Files within directory discovered
        """
        user_input = "What files are in @src/?"

        parsed = file_reference_parser.parse_references(
            user_input,
            working_directory=temp_project_dir
        )

        # Should find one reference
        assert len(parsed.references) == 1

        ref = parsed.references[0]
        assert ref.type == "directory"
        assert ref.original_text == "@src/"
        assert ref.resolved_path == temp_project_dir / "src"
        assert ref.exists is True

        # Directory should have discovered files
        assert hasattr(ref, 'files')
        assert len(ref.files) >= 2  # At least main.py and utils.py

    def test_parse_nested_path(self, file_reference_parser, temp_project_dir):
        """Contract: Parse nested file path @src/models/user.py.

        Given: User message with nested path reference
        When: Parser processes the message
        Then: Returns parsed reference with correct path

        Validates:
        - Nested paths resolved correctly
        - Path separators handled
        - File existence verified
        """
        user_input = "Review @src/models/user.py for issues"

        parsed = file_reference_parser.parse_references(
            user_input,
            working_directory=temp_project_dir
        )

        assert len(parsed.references) == 1

        ref = parsed.references[0]
        assert ref.type == "file"
        assert ref.resolved_path == temp_project_dir / "src" / "models" / "user.py"
        assert ref.exists is True

    def test_parse_nonexistent_file(self, file_reference_parser, temp_project_dir):
        """Contract: Handle nonexistent file reference gracefully.

        Given: User message with reference to nonexistent file
        When: Parser processes the message
        Then: Returns reference marked as not found

        Validates:
        - Nonexistent files don't cause errors
        - Reference still parsed
        - exists=False flag set
        """
        user_input = "Check @nonexistent.txt please"

        parsed = file_reference_parser.parse_references(
            user_input,
            working_directory=temp_project_dir
        )

        assert len(parsed.references) == 1

        ref = parsed.references[0]
        assert ref.original_text == "@nonexistent.txt"
        assert ref.exists is False
        assert hasattr(ref, 'error') or hasattr(ref, 'not_found')

    def test_parse_absolute_path(self, file_reference_parser, temp_project_dir):
        """Contract: Parse absolute file path reference.

        Given: User message with absolute path
        When: Parser processes the message
        Then: Returns parsed reference with absolute path

        Validates:
        - Absolute paths supported
        - Working directory not prepended
        - Path resolved correctly
        """
        readme_path = temp_project_dir / "README.md"
        user_input = f"Analyze @{readme_path}"

        parsed = file_reference_parser.parse_references(
            user_input,
            working_directory=temp_project_dir
        )

        assert len(parsed.references) == 1

        ref = parsed.references[0]
        assert ref.resolved_path == readme_path
        assert ref.exists is True

    def test_parse_no_references(self, file_reference_parser, temp_project_dir):
        """Contract: Handle messages with no file references.

        Given: User message without @ references
        When: Parser processes the message
        Then: Returns empty reference list

        Validates:
        - No false positives
        - Message returned unchanged
        - Empty references list
        """
        user_input = "What is the weather today?"

        parsed = file_reference_parser.parse_references(
            user_input,
            working_directory=temp_project_dir
        )

        assert len(parsed.references) == 0
        assert parsed.message == user_input

    def test_parse_reference_with_spaces(self, file_reference_parser, temp_project_dir):
        """Contract: Handle file names with spaces.

        Given: User message with reference to file with spaces
        When: Parser processes the message
        Then: Returns correctly parsed reference

        Validates:
        - Spaces in file names handled
        - Quote handling (if required)
        - Path resolution works
        """
        # Create file with spaces
        spaced_file = temp_project_dir / "my document.txt"
        spaced_file.write_text("Content with spaces")

        user_input = "Read @my document.txt please"

        parsed = file_reference_parser.parse_references(
            user_input,
            working_directory=temp_project_dir
        )

        assert len(parsed.references) == 1

        ref = parsed.references[0]
        assert ref.resolved_path == spaced_file
        assert ref.exists is True

    def test_parse_relative_parent_path(self, file_reference_parser, temp_project_dir):
        """Contract: Handle relative paths with parent directory (..).

        Given: User message with ../ in path
        When: Parser processes the message
        Then: Returns resolved absolute path

        Validates:
        - Parent directory references handled
        - Path normalized correctly
        - Security checks (no path traversal outside project)
        """
        # From src/ directory, reference parent's README
        user_input = "Check @../README.md from src context"

        # Set working directory to src/
        parsed = file_reference_parser.parse_references(
            user_input,
            working_directory=temp_project_dir / "src"
        )

        assert len(parsed.references) == 1

        ref = parsed.references[0]
        assert ref.resolved_path == temp_project_dir / "README.md"
        assert ref.exists is True

    def test_parse_glob_pattern(self, file_reference_parser, temp_project_dir):
        """Contract: Handle glob patterns in references.

        Given: User message with glob pattern (e.g., @*.py)
        When: Parser processes the message
        Then: Returns reference with matched files

        Validates:
        - Glob patterns recognized
        - Multiple files matched
        - Type identified as 'glob' or 'pattern'
        """
        user_input = "Review all @src/*.py files"

        parsed = file_reference_parser.parse_references(
            user_input,
            working_directory=temp_project_dir
        )

        # Should find glob pattern reference
        assert len(parsed.references) == 1

        ref = parsed.references[0]
        assert ref.type in ["glob", "pattern", "directory"]
        assert hasattr(ref, 'matched_files') or hasattr(ref, 'files')

        # Should match at least main.py and utils.py
        matched = getattr(ref, 'matched_files', None) or getattr(ref, 'files', [])
        assert len(matched) >= 2
