"""File reference parser for @syntax in user messages.

This module parses file and directory references from user messages,
resolves paths, and validates existence.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ParsedReference:
    """Represents a parsed file or directory reference.

    Attributes:
        type: Reference type ('file', 'directory', 'glob', 'pattern')
        original_text: Original @ reference text
        resolved_path: Resolved absolute path
        exists: Whether the path exists
        error: Error message if resolution failed
        files: List of files (for directory/glob references)
        matched_files: List of matched files (for glob patterns)
    """

    type: str
    original_text: str
    resolved_path: Path
    exists: bool
    error: Optional[str] = None
    not_found: bool = False
    files: List[Path] = field(default_factory=list)
    matched_files: List[Path] = field(default_factory=list)


@dataclass
class ParsedMessage:
    """Result of parsing a message for file references.

    Attributes:
        message: Original message text
        references: List of parsed references
    """

    message: str
    references: List[ParsedReference] = field(default_factory=list)


class FileReferenceParser:
    """Parser for file and directory references using @ syntax.

    Supports:
    - Single files: @README.md
    - Multiple files: @file1.txt @file2.txt
    - Directories: @src/
    - Nested paths: @src/models/user.py
    - Absolute paths: @/absolute/path/file.txt
    - Glob patterns: @src/*.py
    - Files with spaces: @my document.txt
    """

    # Regex pattern to match @ references
    # Matches: @filename, @path/to/file, @directory/, @pattern*.ext
    # Supports spaces in filenames if there's an extension: @my document.txt
    # Two alternatives: with spaces (requires extension) OR without spaces
    REFERENCE_PATTERN = re.compile(
        r"@([A-Za-z0-9_\-./\\*]+(?:\s+[A-Za-z0-9_\-./\\*]+)+\.[a-zA-Z0-9]+|[A-Za-z0-9_\-./\\*]+)"
    )

    def __init__(self):
        """Initialize file reference parser."""
        pass

    def parse_references(self, message: str, working_directory: Path) -> ParsedMessage:
        """Parse file references from message.

        Args:
            message: User message text
            working_directory: Base directory for relative paths

        Returns:
            ParsedMessage: Parsed message with references
        """
        working_directory = Path(working_directory)
        references = []

        # Find all @ references
        matches = self.REFERENCE_PATTERN.finditer(message)

        for match in matches:
            ref_text = match.group(1)
            original_text = f"@{ref_text}"

            # Resolve path
            resolved_path = self._resolve_path(ref_text, working_directory)

            # Determine type and check existence
            ref = self._create_reference(original_text, resolved_path)
            references.append(ref)

        return ParsedMessage(message=message, references=references)

    def _resolve_path(self, ref_text: str, working_directory: Path) -> Path:
        """Resolve reference text to absolute path.

        Args:
            ref_text: Reference text (without @)
            working_directory: Base directory

        Returns:
            Path: Resolved absolute path
        """
        # Handle absolute paths
        if ref_text.startswith("/"):
            return Path(ref_text)

        # Handle relative paths
        resolved = working_directory / ref_text

        # Normalize path (resolve .., ., etc.)
        try:
            resolved = resolved.resolve()
        except Exception:
            # If resolution fails, return as-is
            pass

        return resolved

    def _create_reference(self, original_text: str, resolved_path: Path) -> ParsedReference:
        """Create ParsedReference from path.

        Args:
            original_text: Original @ reference text
            resolved_path: Resolved path

        Returns:
            ParsedReference: Parsed reference object
        """
        # Check if path exists
        exists = resolved_path.exists()

        # Determine type
        ref_type = "file"
        files = []
        matched_files = []
        error = None
        not_found = False

        if exists:
            if resolved_path.is_dir():
                ref_type = "directory"
                # Discover files in directory
                try:
                    files = list(resolved_path.iterdir())
                except Exception as e:
                    error = f"Cannot read directory: {e}"
            elif resolved_path.is_file():
                ref_type = "file"
        else:
            # Check for glob patterns (but not if ends with /)
            if ("*" in original_text or "?" in original_text) and not original_text.endswith("/"):
                ref_type = "glob"
                # Try glob matching
                try:
                    parent = resolved_path.parent
                    pattern = resolved_path.name
                    if parent.exists():
                        matched_files = list(parent.glob(pattern))
                        if matched_files:
                            exists = True
                except Exception as e:
                    error = f"Glob pattern failed: {e}"
            else:
                not_found = True
                error = f"File not found: {resolved_path}"

        return ParsedReference(
            type=ref_type,
            original_text=original_text,
            resolved_path=resolved_path,
            exists=exists,
            error=error,
            not_found=not_found,
            files=files,
            matched_files=matched_files,
        )
