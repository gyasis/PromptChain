"""File reference parser for PromptChain CLI.

This module handles parsing and resolving @file.txt and @directory/ references
from user messages, including intelligent truncation and directory discovery.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class FileReference:
    """Represents a file or directory reference.

    Attributes:
        path: Absolute path to file or directory
        is_directory: True if this is a directory reference
        content: File content (None for directories)
        files: Discovered files (directories only)
        size_bytes: Original file size
        truncated: True if content was truncated
        preview_lines: Number of lines in preview (if truncated)
        modified_at: File modification time (Unix timestamp)
        error: Error message if reference failed
    """

    path: Path
    is_directory: bool
    content: Optional[str] = None
    files: Optional[list[Path]] = None
    size_bytes: int = 0
    truncated: bool = False
    preview_lines: Optional[int] = None
    modified_at: Optional[float] = None
    error: Optional[str] = None

    def to_message_context(self) -> str:
        """Convert reference to context string for LLM.

        Returns:
            str: Formatted content for inclusion in prompt

        Format (file):
            --- FILE: src/main.py (5.4 KB, modified: 2024-11-16) ---
            import os
            ...
            --- END FILE ---

        Format (directory):
            --- DIRECTORY: src/ (15 files) ---
            - main.py
            - utils.py
            ...
            --- END DIRECTORY ---

        Format (truncated):
            --- FILE: large.log (500 KB, TRUNCATED, showing first 500 + last 100 lines) ---
            ...
            --- END FILE ---

        Format (error):
            --- FILE: nonexistent.txt (ERROR: File not found) ---
        """
        if self.error:
            return f"--- FILE: {self.path} (ERROR: {self.error}) ---"

        # Format file size
        size_str = self._format_file_size(self.size_bytes)

        # Format modification time
        if self.modified_at:
            mod_time = datetime.fromtimestamp(self.modified_at).strftime("%Y-%m-%d")
        else:
            mod_time = "unknown"

        if self.is_directory:
            # Directory format
            file_count = len(self.files) if self.files else 0
            header = f"--- DIRECTORY: {self.path} ({file_count} files) ---"

            if self.files:
                file_list = "\n".join(f"- {f.name}" for f in self.files)
                return f"{header}\n{file_list}\n--- END DIRECTORY ---"
            else:
                return f"{header}\n(empty directory)\n--- END DIRECTORY ---"

        else:
            # File format
            truncation_info = ""
            if self.truncated and self.preview_lines:
                truncation_info = f", TRUNCATED, showing ~{self.preview_lines} lines"

            header = (
                f"--- FILE: {self.path} ({size_str}, modified: {mod_time}{truncation_info}) ---"
            )

            if self.content:
                return f"{header}\n{self.content}\n--- END FILE ---"
            else:
                return f"{header}\n(empty file)\n--- END FILE ---"

    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format.

        Args:
            size_bytes: File size in bytes

        Returns:
            str: Formatted size (e.g., "5.4 KB", "1.2 MB")
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


class FileReferenceParser:
    """Parses @file.txt and @directory/ references from messages.

    Responsibilities:
    - Detect @syntax in user messages
    - Resolve relative paths to absolute paths
    - Read file contents with size limits
    - Discover relevant files in directories
    - Handle permission/access errors gracefully
    """

    def __init__(
        self,
        working_directory: Path,
        max_file_size: int = 100 * 1024,  # 100KB default
        preview_lines_head: int = 500,
        preview_lines_tail: int = 100,
    ):
        """Initialize file reference parser.

        Args:
            working_directory: Session working directory for relative paths
            max_file_size: Max file size before truncation (bytes)
            preview_lines_head: Lines to include from file start (large files)
            preview_lines_tail: Lines to include from file end (large files)
        """
        self.working_directory = working_directory
        self.max_file_size = max_file_size
        self.preview_lines_head = preview_lines_head
        self.preview_lines_tail = preview_lines_tail

    # Methods will be implemented in subsequent tasks
    # This is the skeleton class structure for Phase 2 Foundation
