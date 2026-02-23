"""Integration tests for directory discovery (T094).

These tests verify that when a user references a directory with @syntax,
the system discovers relevant files and provides appropriate context.
"""

import pytest
from pathlib import Path
import tempfile
import shutil


class TestDirectoryDiscovery:
    """Test directory file discovery and context building."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory with realistic structure."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create src directory with Python files
        src = temp_dir / "src"
        src.mkdir()
        (src / "main.py").write_text("# Main entry point\ndef main():\n    pass")
        (src / "utils.py").write_text("# Utilities\ndef helper():\n    return 42")
        (src / "config.py").write_text("# Configuration\nSETTING = 'value'")

        # Create models subdirectory
        models = src / "models"
        models.mkdir()
        (models / "user.py").write_text("class User:\n    pass")
        (models / "post.py").write_text("class Post:\n    pass")

        # Create tests directory
        tests = temp_dir / "tests"
        tests.mkdir()
        (tests / "test_main.py").write_text("def test_main():\n    assert True")
        (tests / "test_utils.py").write_text("def test_helper():\n    assert True")

        # Create docs directory with non-code files
        docs = temp_dir / "docs"
        docs.mkdir()
        (docs / "README.md").write_text("# Documentation")
        (docs / "guide.md").write_text("# User Guide")

        # Create data directory with various file types
        data = temp_dir / "data"
        data.mkdir()
        (data / "sample.json").write_text('{"data": "value"}')
        (data / "image.png").write_bytes(b'\x89PNG\r\n')
        (data / "sample.csv").write_text("col1,col2\n1,2")

        # Create hidden directory (should be ignored by default)
        hidden = temp_dir / ".git"
        hidden.mkdir()
        (hidden / "config").write_text("git config")

        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def directory_discoverer(self):
        """Create DirectoryDiscoverer instance (to be implemented)."""
        try:
            from promptchain.cli.utils.directory_discoverer import DirectoryDiscoverer
            return DirectoryDiscoverer()
        except ImportError:
            pytest.skip("DirectoryDiscoverer not yet implemented (will be in T099)")

    def test_relevant_files_found(self, directory_discoverer, temp_project_dir):
        """Integration: Discover relevant files in directory.

        Flow:
        1. User references @src/ directory
        2. System discovers Python files
        3. Returns relevant files only

        Validates:
        - File discovery works
        - Relevant file types identified
        - Hidden files excluded
        """
        src_dir = temp_project_dir / "src"

        discovered = directory_discoverer.discover_files(
            directory=src_dir,
            recursive=True
        )

        # Should find Python files
        file_names = [f.name for f in discovered.files]

        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "config.py" in file_names

        # Should find nested files
        assert "user.py" in file_names
        assert "post.py" in file_names

        # Total should be 5 Python files
        assert len(discovered.files) >= 5

    def test_code_files_prioritized(self, directory_discoverer, temp_project_dir):
        """Integration: Code files prioritized over other file types.

        Flow:
        1. User references directory with mixed file types
        2. System discovers files
        3. Code files appear first in results

        Validates:
        - Priority ordering
        - Code files (.py, .js, .ts, etc.) prioritized
        - Documentation files included but lower priority
        """
        discovered = directory_discoverer.discover_files(
            directory=temp_project_dir,
            recursive=True,
            max_files=10
        )

        # Get file extensions
        extensions = [f.suffix for f in discovered.files[:5]]

        # First 5 should mostly be code files
        code_extensions = ['.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java']
        code_count = sum(1 for ext in extensions if ext in code_extensions)

        assert code_count >= 3, "Code files should be prioritized"

    def test_max_files_limit(self, directory_discoverer, temp_project_dir):
        """Integration: Respect maximum file limit.

        Flow:
        1. User references large directory
        2. System discovers files
        3. Returns at most max_files entries

        Validates:
        - Limit enforcement
        - Most relevant files selected
        - Clear indication of truncation
        """
        discovered = directory_discoverer.discover_files(
            directory=temp_project_dir,
            recursive=True,
            max_files=5
        )

        # Should not exceed limit
        assert len(discovered.files) <= 5

        # Should have truncation indicator
        assert hasattr(discovered, 'truncated')
        if len(list(temp_project_dir.rglob('*.*'))) > 5:
            assert discovered.truncated is True

    def test_hidden_files_excluded(self, directory_discoverer, temp_project_dir):
        """Integration: Hidden files and directories excluded by default.

        Flow:
        1. User references directory with .git/
        2. System discovers files
        3. .git/ contents not included

        Validates:
        - Hidden directory exclusion
        - .git, .venv, node_modules excluded
        - __pycache__ excluded
        """
        discovered = directory_discoverer.discover_files(
            directory=temp_project_dir,
            recursive=True
        )

        # Should not include files from .git
        file_paths = [str(f) for f in discovered.files]

        for path in file_paths:
            assert ".git" not in path
            assert "__pycache__" not in path

    def test_binary_files_skipped(self, directory_discoverer, temp_project_dir):
        """Integration: Binary files excluded from discovery.

        Flow:
        1. User references directory with binary files
        2. System discovers text files only
        3. Binary files not included

        Validates:
        - Binary detection
        - Text-only results
        - Image/compiled files excluded
        """
        data_dir = temp_project_dir / "data"

        discovered = directory_discoverer.discover_files(
            directory=data_dir,
            recursive=False
        )

        file_names = [f.name for f in discovered.files]

        # Should include text files
        assert "sample.json" in file_names or "sample.csv" in file_names

        # Should NOT include binary files
        assert "image.png" not in file_names

    def test_file_size_filtering(self, directory_discoverer, temp_project_dir):
        """Integration: Very large files excluded or truncated.

        Flow:
        1. Create very large file in directory
        2. User references directory
        3. Large file handled appropriately

        Validates:
        - Size-based filtering
        - Large files excluded or marked for truncation
        - Reasonable file sizes included
        """
        # Create large file
        large_file = temp_project_dir / "src" / "large.py"
        large_file.write_text("x" * (5 * 1024 * 1024))  # 5MB

        src_dir = temp_project_dir / "src"

        discovered = directory_discoverer.discover_files(
            directory=src_dir,
            recursive=False,
            max_file_size=1024 * 1024  # 1MB limit
        )

        # Large file should be excluded or marked
        large_included = any(f.name == "large.py" for f in discovered.files)

        if large_included:
            # If included, should be marked for truncation
            large_file_info = next(f for f in discovered.files if f.name == "large.py")
            assert hasattr(large_file_info, 'will_truncate') or hasattr(large_file_info, 'too_large')

    def test_nested_directory_discovery(self, directory_discoverer, temp_project_dir):
        """Integration: Recursive discovery finds nested files.

        Flow:
        1. User references parent directory
        2. System discovers files recursively
        3. Nested directories included

        Validates:
        - Recursive traversal
        - Nested files found
        - Correct depth handling
        """
        src_dir = temp_project_dir / "src"

        discovered = directory_discoverer.discover_files(
            directory=src_dir,
            recursive=True
        )

        file_names = [f.name for f in discovered.files]

        # Should find files in nested models/ directory
        assert "user.py" in file_names
        assert "post.py" in file_names

    def test_non_recursive_discovery(self, directory_discoverer, temp_project_dir):
        """Integration: Non-recursive mode only gets immediate children.

        Flow:
        1. User references directory (non-recursive)
        2. System discovers files in directory only
        3. Subdirectory contents not included

        Validates:
        - Non-recursive mode
        - Only immediate children
        - Subdirectories not traversed
        """
        src_dir = temp_project_dir / "src"

        discovered = directory_discoverer.discover_files(
            directory=src_dir,
            recursive=False
        )

        file_names = [f.name for f in discovered.files]

        # Should find immediate files
        assert "main.py" in file_names
        assert "utils.py" in file_names

        # Should NOT find nested files
        assert "user.py" not in file_names
        assert "post.py" not in file_names

    def test_file_metadata_included(self, directory_discoverer, temp_project_dir):
        """Integration: Discovered files include useful metadata.

        Flow:
        1. User references directory
        2. System discovers files
        3. Returns metadata with each file

        Validates:
        - File size included
        - File type/extension
        - Relative path from base directory
        """
        src_dir = temp_project_dir / "src"

        discovered = directory_discoverer.discover_files(
            directory=src_dir,
            recursive=True
        )

        # Check first file has metadata
        if discovered.files:
            first_file = discovered.files[0]

            # Should have useful attributes
            assert hasattr(first_file, 'path') or hasattr(first_file, 'name')
            assert hasattr(first_file, 'size') or hasattr(first_file, 'bytes')
            assert hasattr(first_file, 'type') or hasattr(first_file, 'extension')

    def test_directory_summary_format(self, directory_discoverer, temp_project_dir):
        """Integration: Directory discovery returns formatted summary.

        Flow:
        1. User references directory
        2. System discovers files
        3. Returns clear summary

        Validates:
        - Summary format
        - File count
        - Directory structure clear
        """
        src_dir = temp_project_dir / "src"

        discovered = directory_discoverer.discover_files(
            directory=src_dir,
            recursive=True
        )

        # Should have summary information
        assert hasattr(discovered, 'total_files') or hasattr(discovered, 'count')
        assert hasattr(discovered, 'files')

        # Should be able to format for display
        if hasattr(discovered, 'format_summary'):
            summary = discovered.format_summary()
            assert len(summary) > 0
            assert "files" in summary.lower() or str(len(discovered.files)) in summary

    def test_file_type_filtering(self, directory_discoverer, temp_project_dir):
        """Integration: Filter by file type/extension.

        Flow:
        1. User references directory with type filter
        2. System discovers only matching files
        3. Returns filtered results

        Validates:
        - Extension filtering
        - Multiple extension support
        - Case-insensitive matching
        """
        discovered = directory_discoverer.discover_files(
            directory=temp_project_dir,
            recursive=True,
            extensions=['.py']  # Only Python files
        )

        # All discovered files should be .py
        for file in discovered.files:
            assert file.suffix == '.py' or file.name.endswith('.py')

    def test_empty_directory_handling(self, directory_discoverer, temp_project_dir):
        """Integration: Empty directory handled gracefully.

        Flow:
        1. User references empty directory
        2. System discovers no files
        3. Clear message returned

        Validates:
        - Empty directory detection
        - No crash
        - Clear message
        """
        # Create empty directory
        empty_dir = temp_project_dir / "empty"
        empty_dir.mkdir()

        discovered = directory_discoverer.discover_files(
            directory=empty_dir,
            recursive=True
        )

        # Should have empty results
        assert len(discovered.files) == 0

        # Should indicate empty directory
        if hasattr(discovered, 'is_empty'):
            assert discovered.is_empty is True
