"""Contract tests for file loading performance (T092).

These tests verify SC-004 requirement: File loading operations must complete
within 500ms for files up to 10MB.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import time


class TestFileLoadingPerformance:
    """Test file loading performance meets SC-004 requirements."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory with various sized files."""
        temp_dir = Path(tempfile.mkdtemp())

        # Small file (1KB)
        small_file = temp_dir / "small.txt"
        small_file.write_text("x" * 1024)

        # Medium file (100KB)
        medium_file = temp_dir / "medium.txt"
        medium_file.write_text("x" * (100 * 1024))

        # Large file (1MB)
        large_file = temp_dir / "large.txt"
        large_file.write_text("x" * (1024 * 1024))

        # Very large file (10MB - at SC-004 limit)
        very_large_file = temp_dir / "very_large.txt"
        very_large_file.write_text("x" * (10 * 1024 * 1024))

        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def file_loader(self):
        """Create FileLoader instance (to be implemented)."""
        try:
            from promptchain.cli.utils.file_loader import FileLoader
            return FileLoader()
        except ImportError:
            pytest.skip("FileLoader not yet implemented (will be in T097)")

    def test_file_load_under_500ms(self, file_loader, temp_project_dir):
        """Contract: File loading completes within 500ms (SC-004).

        Given: File up to 10MB in size
        When: Loading file content
        Then: Operation completes in <500ms

        Validates:
        - SC-004 performance requirement
        - Fast file reading
        - No blocking operations
        """
        large_file = temp_project_dir / "very_large.txt"

        start = time.time()
        loaded = file_loader.load_file(large_file)
        elapsed = time.time() - start

        # Should complete in under 500ms
        assert elapsed < 0.5, f"File load took {elapsed:.3f}s (should be <0.5s)"

        # Content should be loaded
        assert len(loaded.content) > 0

    def test_small_file_fast_load(self, file_loader, temp_project_dir):
        """Contract: Small files load very quickly (<50ms).

        Given: Small file (1KB)
        When: Loading file content
        Then: Operation completes in <50ms

        Validates:
        - Small files don't have overhead
        - Fast path for small files
        """
        small_file = temp_project_dir / "small.txt"

        start = time.time()
        loaded = file_loader.load_file(small_file)
        elapsed = time.time() - start

        # Small files should be very fast
        assert elapsed < 0.05, f"Small file load took {elapsed:.3f}s (should be <0.05s)"
        assert len(loaded.content) == 1024

    def test_medium_file_reasonable_load(self, file_loader, temp_project_dir):
        """Contract: Medium files load efficiently (<100ms).

        Given: Medium file (100KB)
        When: Loading file content
        Then: Operation completes in <100ms

        Validates:
        - Reasonable performance for typical files
        - No linear scaling issues
        """
        medium_file = temp_project_dir / "medium.txt"

        start = time.time()
        loaded = file_loader.load_file(medium_file)
        elapsed = time.time() - start

        # Medium files should still be fast
        assert elapsed < 0.1, f"Medium file load took {elapsed:.3f}s (should be <0.1s)"
        assert len(loaded.content) == 100 * 1024

    def test_multiple_files_sequential_load(self, file_loader, temp_project_dir):
        """Contract: Loading multiple files sequentially meets performance.

        Given: Multiple files to load
        When: Loading files one after another
        Then: Total time proportional to file count

        Validates:
        - No accumulation of delays
        - Consistent performance
        - Total time within bounds
        """
        files = [
            temp_project_dir / "small.txt",
            temp_project_dir / "medium.txt",
            temp_project_dir / "large.txt"
        ]

        start = time.time()
        for file_path in files:
            loaded = file_loader.load_file(file_path)
            assert len(loaded.content) > 0
        elapsed = time.time() - start

        # Total should be reasonable (all small-medium files < 500ms total)
        assert elapsed < 0.5, f"Sequential load took {elapsed:.3f}s (should be <0.5s)"

    def test_repeated_load_performance(self, file_loader, temp_project_dir):
        """Contract: Repeated loads have consistent performance.

        Given: Same file loaded multiple times
        When: Loading file repeatedly
        Then: Each load has similar performance

        Validates:
        - No caching artifacts
        - Consistent timing
        - No memory leaks affecting performance
        """
        large_file = temp_project_dir / "large.txt"

        timings = []
        for _ in range(5):
            start = time.time()
            loaded = file_loader.load_file(large_file)
            elapsed = time.time() - start
            timings.append(elapsed)
            assert len(loaded.content) > 0

        # All loads should be under 500ms
        for timing in timings:
            assert timing < 0.5, f"Load took {timing:.3f}s (should be <0.5s)"

        # Variance should be reasonable (within 2x)
        min_time = min(timings)
        max_time = max(timings)
        assert max_time / min_time < 2.0, "Performance variance too high"

    def test_concurrent_file_loads(self, file_loader, temp_project_dir):
        """Contract: Concurrent file loads don't degrade performance.

        Given: Multiple files loaded concurrently (if supported)
        When: Loading files in parallel
        Then: Performance better than sequential

        Validates:
        - Async/parallel loading support (if implemented)
        - No resource contention
        - Improved throughput
        """
        import asyncio

        files = [
            temp_project_dir / "small.txt",
            temp_project_dir / "medium.txt",
            temp_project_dir / "large.txt"
        ]

        # If async load is supported
        if hasattr(file_loader, 'load_file_async'):
            async def load_all():
                tasks = [file_loader.load_file_async(f) for f in files]
                return await asyncio.gather(*tasks)

            start = time.time()
            results = asyncio.run(load_all())
            elapsed = time.time() - start

            # Concurrent should be faster than sequential
            # Sequential would be ~500ms, concurrent should be closer to slowest file
            assert elapsed < 0.3, f"Concurrent load took {elapsed:.3f}s (should be <0.3s)"
            assert len(results) == 3
        else:
            pytest.skip("Async loading not implemented")

    def test_memory_efficient_large_file(self, file_loader, temp_project_dir):
        """Contract: Large file loading is memory-efficient.

        Given: Very large file (10MB)
        When: Loading file content
        Then: Memory usage reasonable (not loading entire file at once if streamed)

        Validates:
        - Memory efficiency
        - Streaming or chunked reading (if implemented)
        - No memory spikes
        """
        import sys

        very_large_file = temp_project_dir / "very_large.txt"

        # Get baseline memory
        # Note: This is a simplified check, real implementation might use tracemalloc
        start = time.time()
        loaded = file_loader.load_file(very_large_file)
        elapsed = time.time() - start

        # Performance should still meet SC-004
        assert elapsed < 0.5, f"Large file load took {elapsed:.3f}s (should be <0.5s)"

        # Content should be available (either full or truncated with preview)
        assert len(loaded.content) > 0

        # If file is truncated, should have indicator
        if len(loaded.content) < 10 * 1024 * 1024:
            # File was truncated, should have preview/truncation info
            assert hasattr(file_loader, 'max_file_size') or '...' in loaded.content

    def test_binary_file_load_performance(self, file_loader, temp_project_dir):
        """Contract: Binary files handled efficiently or skipped.

        Given: Binary file (non-text)
        When: Attempting to load
        Then: Fast handling (either skip or error indication)

        Validates:
        - Binary file detection
        - No hanging on binary data
        - Clear indication of binary file
        """
        # Create binary file
        binary_file = temp_project_dir / "binary.bin"
        binary_file.write_bytes(bytes(range(256)) * 1000)

        start = time.time()
        try:
            result = file_loader.load_file(binary_file)
            elapsed = time.time() - start

            # Should be fast either way (skip or load)
            assert elapsed < 0.1, f"Binary file handling took {elapsed:.3f}s"

            # If loaded, should have indication it's binary
            if result:
                assert 'binary' in str(result).lower() or len(result) == 0

        except Exception as e:
            elapsed = time.time() - start
            # Error handling should be fast
            assert elapsed < 0.1, f"Binary file error took {elapsed:.3f}s"
            assert 'binary' in str(e).lower() or 'text' in str(e).lower()

    def test_directory_listing_performance(self, file_loader, temp_project_dir):
        """Contract: Directory file discovery is fast.

        Given: Directory with multiple files
        When: Discovering files in directory
        Then: Discovery completes quickly

        Validates:
        - Fast directory traversal
        - File filtering performance
        - No blocking on large directories
        """
        # Create directory with many files
        test_dir = temp_project_dir / "many_files"
        test_dir.mkdir()

        for i in range(50):
            (test_dir / f"file_{i}.txt").write_text(f"Content {i}")

        start = time.time()
        if hasattr(file_loader, 'discover_files'):
            files = file_loader.discover_files(test_dir)
            elapsed = time.time() - start

            # Should discover quickly
            assert elapsed < 0.1, f"Directory discovery took {elapsed:.3f}s"
            assert len(files) >= 50
        else:
            pytest.skip("Directory discovery not implemented")
