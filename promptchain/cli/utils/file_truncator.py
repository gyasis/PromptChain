"""File truncation for large files.

This module handles truncating large files to manageable sizes with
clear indicators and preview formatting.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TruncationResult:
    """Result of file truncation.

    Attributes:
        content: Truncated content (or full if not truncated)
        truncated: Whether file was truncated
        original_size: Original file size in bytes
        truncated_size: Size after truncation
        preview_size: Size of preview content
        total_size: Alias for original_size
    """

    content: str
    truncated: bool
    original_size: int
    truncated_size: int = 0
    preview_size: int = 0
    total_size: int = 0  # Alias for original_size

    def __post_init__(self):
        """Set aliases after initialization."""
        if self.total_size == 0:
            self.total_size = self.original_size
        if self.preview_size == 0:
            self.preview_size = len(self.content)
        if self.truncated_size == 0:
            self.truncated_size = len(self.content)


class FileTruncator:
    """Truncates large files with clear preview formatting.

    Features:
    - Head and tail preview
    - Clear truncation indicators
    - Line boundary respect
    - File size information
    """

    def __init__(
        self,
        max_size: int = 100 * 1024,  # 100KB default
        max_tokens: Optional[int] = None,
        head_lines: int = 50,
        tail_lines: int = 20,
    ):
        """Initialize file truncator.

        Args:
            max_size: Maximum file size in bytes
            max_tokens: Maximum tokens (if token-aware)
            head_lines: Number of lines to show from start
            tail_lines: Number of lines to show from end
        """
        self.max_size = max_size
        self.max_tokens = max_tokens
        self.head_lines = head_lines
        self.tail_lines = tail_lines

    def truncate_file(self, file_path: Path) -> TruncationResult:
        """Truncate file if needed.

        Args:
            file_path: Path to file

        Returns:
            TruncationResult: Truncation result
        """
        file_path = Path(file_path)

        try:
            # Get file size
            original_size = file_path.stat().st_size

            # Load content
            content = file_path.read_text(encoding="utf-8", errors="replace")

            # Check if truncation needed
            if len(content) <= self.max_size:
                return TruncationResult(
                    content=content, truncated=False, original_size=original_size
                )

            # Truncate with preview format
            truncated_content = self._create_preview(
                content=content, file_path=file_path, original_size=original_size
            )

            return TruncationResult(
                content=truncated_content,
                truncated=True,
                original_size=original_size,
                truncated_size=len(truncated_content),
            )

        except Exception as e:
            # If error, return error message
            return TruncationResult(
                content=f"[Error loading file: {e}]", truncated=False, original_size=0
            )

    def _create_preview(self, content: str, file_path: Path, original_size: int) -> str:
        """Create preview format with head and tail.

        Args:
            content: Full file content
            file_path: Path to file
            original_size: Original size in bytes

        Returns:
            str: Formatted preview
        """
        lines = content.split("\n")

        # For very large files with few lines (like 'x' repeated),
        # truncate by character count instead
        if len(lines) < self.head_lines + self.tail_lines:
            # Truncate by bytes to ensure preview is shorter
            head = content[: self.max_size // 2]
            tail = content[-self.max_size // 4 :] if len(content) > self.max_size else ""
        else:
            # Get head lines
            head = "\n".join(lines[: self.head_lines])
            # Get tail lines
            tail = "\n".join(lines[-self.tail_lines :])

        # Format preview
        size_mb = original_size / (1024 * 1024)
        size_kb = original_size / 1024

        size_str = f"{size_mb:.1f}MB" if size_mb >= 1 else f"{size_kb:.0f}KB"

        preview_parts = [
            f"[File: {file_path.name}]",
            f"[Size: {size_str} ({original_size} bytes)]",
            f"[Content truncated]",
            "",
            head,
        ]

        if tail and len(content) > self.max_size:
            preview_parts.extend(["", "[...]", "", tail])

        preview_parts.append("")
        preview_parts.append("[End File]")

        return "\n".join(preview_parts)
