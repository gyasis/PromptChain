"""File loader with performance optimization.

This module handles efficient file loading with binary detection,
encoding handling, and performance requirements (SC-004: <500ms for 10MB).
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class LoadedFile:
    """Result of loading a file.

    Attributes:
        path: File path
        content: File content (text)
        size: File size in bytes
        load_time: Time taken to load (seconds)
        is_binary: Whether file is binary
        encoding: Detected encoding
        error: Error message if loading failed
    """

    path: Path
    content: str
    size: int
    load_time: float
    is_binary: bool = False
    encoding: str = "utf-8"
    error: Optional[str] = None


class FileLoader:
    """Efficient file loader with binary detection.

    Performance requirement (SC-004):
    - Load files up to 10MB in <500ms
    """

    def __init__(self, max_file_size: int = 10 * 1024 * 1024):
        """Initialize file loader.

        Args:
            max_file_size: Maximum file size to load (bytes)
        """
        self.max_file_size = max_file_size

    def load_file(self, file_path: Path) -> LoadedFile:
        """Load file content synchronously.

        Args:
            file_path: Path to file

        Returns:
            LoadedFile: Loaded file result
        """
        file_path = Path(file_path)
        start_time = time.time()

        try:
            # Get file size
            size = file_path.stat().st_size

            # Check if file is too large
            if size > self.max_file_size:
                return LoadedFile(
                    path=file_path,
                    content=f"[File too large: {size / (1024*1024):.1f}MB]",
                    size=size,
                    load_time=time.time() - start_time,
                    error=f"File exceeds {self.max_file_size / (1024*1024):.0f}MB limit",
                )

            # Check if binary
            is_binary = self._is_binary_file(file_path)

            if is_binary:
                return LoadedFile(
                    path=file_path,
                    content=f"[Binary file: {file_path.name}]",
                    size=size,
                    load_time=time.time() - start_time,
                    is_binary=True,
                    error="Binary file cannot be loaded as text",
                )

            # Load text content
            content = file_path.read_text(encoding="utf-8", errors="replace")

            load_time = time.time() - start_time

            return LoadedFile(
                path=file_path,
                content=content,
                size=size,
                load_time=load_time,
                is_binary=False,
                encoding="utf-8",
            )

        except Exception as e:
            return LoadedFile(
                path=file_path,
                content="",
                size=0,
                load_time=time.time() - start_time,
                error=str(e),
            )

    async def load_file_async(self, file_path: Path) -> LoadedFile:
        """Load file content asynchronously.

        Args:
            file_path: Path to file

        Returns:
            LoadedFile: Loaded file result
        """
        # Run synchronous load in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load_file, file_path)

    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary.

        Args:
            file_path: Path to file

        Returns:
            bool: True if binary, False if text
        """
        try:
            # Read first 8KB to check for binary content
            with open(file_path, "rb") as f:
                chunk = f.read(8192)

            # Check for null bytes (common in binary files)
            if b"\x00" in chunk:
                return True

            # Check for high percentage of non-text bytes
            text_chars = sum(1 for byte in chunk if 32 <= byte <= 126 or byte in (9, 10, 13))
            if len(chunk) > 0:
                text_ratio = text_chars / len(chunk)
                if text_ratio < 0.7:  # Less than 70% text characters
                    return True

            return False

        except Exception:
            # If we can't read, assume it might be binary
            return True

    def discover_files(self, directory: Path) -> List[Path]:
        """Discover files in directory.

        Args:
            directory: Directory path

        Returns:
            List[Path]: List of file paths
        """
        directory = Path(directory)
        files = []

        try:
            for item in directory.rglob("*"):
                if item.is_file():
                    files.append(item)
        except Exception:
            pass

        return files
