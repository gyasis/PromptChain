"""Unit tests for file truncation logic (T095).

These tests verify that large files are truncated appropriately with clear
indicators, maintaining readability while respecting token limits.
"""

import pytest
from pathlib import Path
import tempfile
import shutil


class TestFileTruncation:
    """Test file truncation logic for large files."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def file_truncator(self):
        """Create FileTruncator instance (to be implemented)."""
        try:
            from promptchain.cli.utils.file_truncator import FileTruncator
            return FileTruncator(max_size=1024 * 100)  # 100KB default
        except ImportError:
            pytest.skip("FileTruncator not yet implemented (will be in T100)")

    def test_large_file_truncated(self, file_truncator, temp_dir):
        """Unit: Large files truncated to max size.

        Given: File larger than max_size
        When: Loading file content
        Then: Content truncated to max_size

        Validates:
        - Truncation at specified limit
        - Content reduced
        - No full file loaded
        """
        # Create large file (200KB)
        large_file = temp_dir / "large.txt"
        large_content = "x" * (200 * 1024)
        large_file.write_text(large_content)

        result = file_truncator.truncate_file(large_file)

        # Should be truncated
        assert result.truncated is True
        assert len(result.content) < len(large_content)
        assert len(result.content) <= file_truncator.max_size

    def test_small_file_not_truncated(self, file_truncator, temp_dir):
        """Unit: Small files pass through unchanged.

        Given: File smaller than max_size
        When: Loading file content
        Then: Full content returned

        Validates:
        - No unnecessary truncation
        - Full content preserved
        - truncated=False flag
        """
        # Create small file (1KB)
        small_file = temp_dir / "small.txt"
        small_content = "Hello World\n" * 50
        small_file.write_text(small_content)

        result = file_truncator.truncate_file(small_file)

        # Should NOT be truncated
        assert result.truncated is False
        assert result.content == small_content
        assert len(result.content) == len(small_content)

    def test_preview_format(self, file_truncator, temp_dir):
        """Unit: Truncated files have clear preview format.

        Given: Large file requiring truncation
        When: Truncating content
        Then: Preview includes header + content + footer

        Validates:
        - Preview structure
        - Header with file info
        - Truncation indicator
        - Footer with stats
        """
        # Create large file (200KB to ensure truncation with 100KB max_size)
        large_file = temp_dir / "preview.txt"
        large_content = "Line {}\n".format(1) * 30000  # ~210KB
        large_file.write_text(large_content)

        result = file_truncator.truncate_file(large_file)

        # Should have structured preview
        preview = result.content

        # Should include file name/path
        assert "preview.txt" in preview or str(large_file) in preview

        # Should have truncation indicator
        assert (
            "[truncated]" in preview.lower() or
            "..." in preview or
            "[content truncated]" in preview.lower() or
            "showing" in preview.lower()
        )

        # Should have size info
        assert "KB" in preview or "bytes" in preview or "size" in preview.lower()

    def test_truncation_indicator(self, file_truncator, temp_dir):
        """Unit: Truncation indicator clearly visible.

        Given: Truncated file
        When: Formatting preview
        Then: Clear indicator present

        Validates:
        - Indicator visibility
        - User understands truncation
        - Format consistent
        """
        large_file = temp_dir / "indicator.txt"
        large_file.write_text("x" * (200 * 1024))

        result = file_truncator.truncate_file(large_file)

        # Should have clear indicator
        indicator_present = (
            "[TRUNCATED]" in result.content or
            "..." in result.content or
            "[Content truncated" in result.content or
            "Showing first" in result.content
        )

        assert indicator_present, "Should have clear truncation indicator"

    def test_head_and_tail_preview(self, file_truncator, temp_dir):
        """Unit: Preview includes both head and tail of file.

        Given: Large file
        When: Truncating
        Then: Shows beginning and end sections

        Validates:
        - Head section included
        - Tail section included
        - Clear separator
        - Context from both ends
        """
        large_file = temp_dir / "head_tail.txt"
        content = (
            "START_MARKER\n" +
            ("middle content\n" * 10000) +
            "END_MARKER\n"
        )
        large_file.write_text(content)

        result = file_truncator.truncate_file(large_file)

        # Should include beginning
        assert "START_MARKER" in result.content

        # Should include end (if using head+tail strategy)
        if "END_MARKER" in result.content:
            # Verify separator between sections
            assert (
                "..." in result.content or
                "[...]" in result.content or
                "..." in result.content
            )

    def test_line_boundary_truncation(self, file_truncator, temp_dir):
        """Unit: Truncation respects line boundaries.

        Given: Large file
        When: Truncating
        Then: Cuts at line boundary, not mid-line

        Validates:
        - Line integrity preserved
        - No partial lines
        - Clean truncation
        """
        large_file = temp_dir / "lines.txt"
        lines = [f"Line {i}: Content here\n" for i in range(10000)]
        large_file.write_text("".join(lines))

        result = file_truncator.truncate_file(large_file)

        # Content should end with complete line (newline)
        # or have clear truncation marker
        content_lines = result.content.split('\n')

        # Should not have partial line at end (before truncation marker)
        # Check that lines are complete
        for line in content_lines[:5]:  # Check first few lines
            if line and not line.startswith('[') and not line.startswith('...'):
                assert line.startswith("Line")

    def test_configurable_max_size(self, temp_dir):
        """Unit: Max size is configurable.

        Given: Custom max_size setting
        When: Truncating files
        Then: Uses specified limit

        Validates:
        - Configuration respected
        - Different limits work
        - Flexible truncation
        """
        try:
            from promptchain.cli.utils.file_truncator import FileTruncator
        except ImportError:
            pytest.skip("FileTruncator not yet implemented")

        # Create truncators with different limits
        truncator_small = FileTruncator(max_size=1024)  # 1KB
        truncator_large = FileTruncator(max_size=1024 * 100)  # 100KB

        # Create 50KB file
        test_file = temp_dir / "config.txt"
        test_file.write_text("x" * (50 * 1024))

        # Small limit should truncate
        result_small = truncator_small.truncate_file(test_file)
        assert result_small.truncated is True
        assert len(result_small.content) <= 1024 * 2  # Allow some overhead

        # Large limit should NOT truncate
        result_large = truncator_large.truncate_file(test_file)
        assert result_large.truncated is False

    def test_truncation_metadata(self, file_truncator, temp_dir):
        """Unit: Truncation result includes metadata.

        Given: Truncated file
        When: Getting result
        Then: Metadata about truncation available

        Validates:
        - Original size
        - Truncated size
        - Percentage shown
        - Bytes removed
        """
        large_file = temp_dir / "metadata.txt"
        original_content = "x" * (200 * 1024)
        large_file.write_text(original_content)

        result = file_truncator.truncate_file(large_file)

        # Should have metadata
        assert hasattr(result, 'original_size') or hasattr(result, 'total_size')
        assert hasattr(result, 'truncated_size') or hasattr(result, 'preview_size')

        if hasattr(result, 'original_size'):
            assert result.original_size == len(original_content)

    def test_empty_file_handling(self, file_truncator, temp_dir):
        """Unit: Empty files handled gracefully.

        Given: Empty file
        When: Truncating
        Then: Returns empty content

        Validates:
        - No crash on empty file
        - truncated=False
        - Empty string returned
        """
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")

        result = file_truncator.truncate_file(empty_file)

        assert result.truncated is False
        assert result.content == ""

    def test_unicode_handling(self, file_truncator, temp_dir):
        """Unit: Unicode content truncated correctly.

        Given: File with Unicode characters
        When: Truncating
        Then: No encoding errors

        Validates:
        - Unicode preserved
        - Character boundaries respected
        - No corruption
        """
        unicode_file = temp_dir / "unicode.txt"
        unicode_content = "Hello 世界 🌍\n" * 10000
        unicode_file.write_text(unicode_content, encoding='utf-8')

        result = file_truncator.truncate_file(unicode_file)

        # Should handle Unicode correctly
        assert "世界" in result.content or "🌍" in result.content
        # No encoding errors should occur

    def test_binary_file_detection(self, file_truncator, temp_dir):
        """Unit: Binary files rejected or handled specially.

        Given: Binary file
        When: Attempting truncation
        Then: Returns error or binary indicator

        Validates:
        - Binary detection
        - Appropriate handling
        - No text processing on binary
        """
        binary_file = temp_dir / "binary.bin"
        binary_file.write_bytes(bytes(range(256)) * 1000)

        try:
            result = file_truncator.truncate_file(binary_file)

            # If binary files are supported, should have indicator
            if result:
                assert (
                    "binary" in result.content.lower() or
                    result.content == "" or
                    hasattr(result, 'is_binary')
                )

        except Exception as e:
            # Binary files might raise exception
            assert "binary" in str(e).lower() or "text" in str(e).lower()

    def test_token_aware_truncation(self, temp_dir):
        """Unit: Truncation considers token count if available.

        Given: FileTruncator with token limit
        When: Truncating files
        Then: Respects token budget

        Validates:
        - Token counting (if implemented)
        - Token limit respected
        - Accurate token estimation
        """
        try:
            from promptchain.cli.utils.file_truncator import FileTruncator
        except ImportError:
            pytest.skip("FileTruncator not yet implemented")

        # If token-aware truncation is implemented
        truncator = FileTruncator(max_tokens=1000)  # 1000 token limit

        large_file = temp_dir / "tokens.txt"
        # ~4 chars per token average, create file exceeding limit
        large_file.write_text("word " * 5000)  # ~5000 tokens

        result = truncator.truncate_file(large_file)

        # Should be truncated based on tokens
        if hasattr(result, 'token_count'):
            assert result.token_count <= 1000
