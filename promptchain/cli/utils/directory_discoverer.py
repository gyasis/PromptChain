"""Directory file discovery with intelligent filtering.

This module discovers relevant files in directories, prioritizing code files
and excluding hidden/binary files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

# File extensions to prioritize (code files)
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".cpp",
    ".c",
    ".h",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".sql",
    ".r",
    ".m",
    ".mm",
}

# Directories to exclude
EXCLUDED_DIRS = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "target",
    "out",
    "bin",
}

# Binary file extensions to skip
BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".pyc",
    ".pyo",
    ".class",
    ".o",
    ".obj",
}


@dataclass
class FileInfo:
    """Information about a discovered file.

    Attributes:
        path: File path
        name: File name
        suffix: File extension
        size: File size in bytes
        bytes: Alias for size
        type: File type classification
        extension: Alias for suffix
        is_code: Whether this is a code file
        will_truncate: Whether file will need truncation
        too_large: Whether file exceeds size limit
    """

    path: Path
    name: str
    suffix: str
    size: int
    type: str = "other"
    is_code: bool = False
    will_truncate: bool = False
    too_large: bool = False

    @property
    def bytes(self) -> int:
        """Alias for size."""
        return self.size

    @property
    def extension(self) -> str:
        """Alias for suffix."""
        return self.suffix


@dataclass
class DiscoveredFiles:
    """Result of directory discovery.

    Attributes:
        files: List of discovered files
        truncated: Whether results were truncated
        total_files: Total number of files
        count: Alias for total_files
        is_empty: Whether no files were found
    """

    files: List[FileInfo] = field(default_factory=list)
    truncated: bool = False
    total_files: int = 0
    is_empty: bool = False

    def __post_init__(self):
        """Set computed attributes."""
        if self.total_files == 0:
            self.total_files = len(self.files)
        self.is_empty = len(self.files) == 0

    @property
    def count(self) -> int:
        """Alias for total_files."""
        return self.total_files

    def format_summary(self) -> str:
        """Format discovery summary.

        Returns:
            str: Human-readable summary
        """
        if self.is_empty:
            return "No files found"

        summary = f"Found {self.total_files} files"
        if self.truncated:
            summary += f" (showing first {len(self.files)})"

        return summary


class DirectoryDiscoverer:
    """Discovers relevant files in directories.

    Features:
    - Code file prioritization
    - Hidden directory exclusion
    - Binary file filtering
    - Size-based filtering
    - Recursive discovery
    """

    def __init__(self):
        """Initialize directory discoverer."""
        pass

    def discover_files(
        self,
        directory: Path,
        recursive: bool = True,
        max_files: Optional[int] = None,
        max_file_size: Optional[int] = None,
        extensions: Optional[List[str]] = None,
    ) -> DiscoveredFiles:
        """Discover files in directory.

        Args:
            directory: Directory to search
            recursive: Whether to search recursively
            max_files: Maximum number of files to return
            max_file_size: Maximum file size in bytes
            extensions: File extensions to include (e.g., ['.py'])

        Returns:
            DiscoveredFiles: Discovery results
        """
        directory = Path(directory)

        if not directory.exists() or not directory.is_dir():
            return DiscoveredFiles(is_empty=True)

        # Collect files
        all_files: List[FileInfo] = []

        if recursive:
            pattern = directory.rglob("*")
        else:
            pattern = directory.glob("*")

        for item in pattern:
            # Skip directories
            if not item.is_file():
                continue

            # Skip if in excluded directory
            if self._is_in_excluded_dir(item):
                continue

            # Skip binary files
            if item.suffix in BINARY_EXTENSIONS:
                continue

            # Filter by extension if specified
            if extensions and item.suffix not in extensions:
                continue

            # Get file info
            try:
                size = item.stat().st_size

                # Skip if too large
                if max_file_size and size > max_file_size:
                    # Still add but mark as too large
                    file_info = FileInfo(
                        path=item,
                        name=item.name,
                        suffix=item.suffix,
                        size=size,
                        type=self._classify_file(item),
                        is_code=item.suffix in CODE_EXTENSIONS,
                        too_large=True,
                    )
                    all_files.append(file_info)
                    continue

                file_info = FileInfo(
                    path=item,
                    name=item.name,
                    suffix=item.suffix,
                    size=size,
                    type=self._classify_file(item),
                    is_code=item.suffix in CODE_EXTENSIONS,
                )
                all_files.append(file_info)

            except Exception:
                # Skip files we can't read
                continue

        # Sort files: code files first, then by name
        all_files.sort(key=lambda f: (not f.is_code, f.name))

        # Apply max_files limit
        truncated = False
        if max_files and len(all_files) > max_files:
            all_files = all_files[:max_files]
            truncated = True

        return DiscoveredFiles(files=all_files, truncated=truncated, total_files=len(all_files))

    def _is_in_excluded_dir(self, file_path: Path) -> bool:
        """Check if file is in excluded directory.

        Args:
            file_path: File path

        Returns:
            bool: True if in excluded directory
        """
        for part in file_path.parts:
            if part in EXCLUDED_DIRS:
                return True
        return False

    def _classify_file(self, file_path: Path) -> str:
        """Classify file by type.

        Args:
            file_path: File path

        Returns:
            str: File type classification
        """
        suffix = file_path.suffix

        if suffix in CODE_EXTENSIONS:
            return "code"
        elif suffix in {".md", ".txt", ".rst", ".adoc"}:
            return "documentation"
        elif suffix in {".json", ".yaml", ".yml", ".toml", ".xml", ".ini", ".cfg"}:
            return "config"
        elif suffix in {".csv", ".tsv", ".dat"}:
            return "data"
        else:
            return "other"
