"""File context manager for injecting file content into prompts.

This module handles parsing file references, loading content, and
formatting context for LLM consumption.
"""

from pathlib import Path
from typing import Optional

from .directory_discoverer import DirectoryDiscoverer
from .file_loader import FileLoader
from .file_reference_parser import FileReferenceParser
from .file_truncator import FileTruncator


class FileContextManager:
    """Manages file context injection into prompts.

    Coordinates:
    - File reference parsing
    - Content loading
    - Truncation
    - Directory discovery
    - Context formatting
    """

    def __init__(
        self,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        truncate_at: int = 100 * 1024,  # 100KB
    ):
        """Initialize file context manager.

        Args:
            max_file_size: Maximum file size to load
            truncate_at: Size at which to truncate files
        """
        self.parser = FileReferenceParser()
        self.loader = FileLoader(max_file_size=max_file_size)
        self.truncator = FileTruncator(max_size=truncate_at)
        self.discoverer = DirectoryDiscoverer()

    def inject_file_context(self, message: str, working_directory: Path) -> str:
        """Inject file content into message.

        Args:
            message: User message with @ references
            working_directory: Base directory for resolving paths

        Returns:
            str: Enhanced message with file content
        """
        working_directory = Path(working_directory)

        # Parse references
        parsed = self.parser.parse_references(message, working_directory)

        # If no references, return original message
        if not parsed.references:
            return message

        # Build context for each reference
        context_parts = [message, ""]  # Start with original message

        for ref in parsed.references:
            if not ref.exists:
                # File not found
                context_parts.append(f"\n[File not found: {ref.original_text}]\n")
                continue

            if ref.type == "file":
                # Load and format file content
                file_context = self._format_file_context(ref.resolved_path)
                context_parts.append(file_context)

            elif ref.type == "directory":
                # Format directory listing
                dir_context = self._format_directory_context(ref.resolved_path)
                context_parts.append(dir_context)

            elif ref.type == "glob":
                # Format matched files
                glob_context = self._format_glob_context(ref)
                context_parts.append(glob_context)

        return "\n".join(context_parts)

    def _format_file_context(self, file_path: Path) -> str:
        """Format file content for context.

        Args:
            file_path: Path to file

        Returns:
            str: Formatted file content
        """
        # Load file
        loaded = self.loader.load_file(file_path)

        # Check for errors
        if loaded.error:
            if loaded.is_binary:
                return f"\n[Binary file: {file_path.name}]\n"
            else:
                return f"\n[Error loading {file_path.name}: {loaded.error}]\n"

        # Check if truncation needed
        if len(loaded.content) > self.truncator.max_size:
            truncated = self.truncator.truncate_file(file_path)
            return f"\n{truncated.content}\n"

        # Format with file markers
        return self._wrap_file_content(file_path, loaded.content)

    def _wrap_file_content(self, file_path: Path, content: str) -> str:
        """Wrap file content with clear markers.

        Args:
            file_path: File path
            content: File content

        Returns:
            str: Wrapped content
        """
        return f"""
[File: {file_path.name}]
{content}
[End File]
"""

    def _format_directory_context(self, dir_path: Path) -> str:
        """Format directory listing for context.

        Args:
            dir_path: Directory path

        Returns:
            str: Formatted directory listing
        """
        # Discover files
        discovered = self.discoverer.discover_files(
            directory=dir_path, recursive=True, max_files=20  # Limit for context
        )

        if discovered.is_empty:
            return f"\n[Directory: {dir_path.name} - Empty]\n"

        # Format file list
        lines = [f"\n[Directory: {dir_path.name}]"]
        lines.append(f"Files found: {discovered.total_files}")

        if discovered.truncated:
            lines.append(f"(showing first {len(discovered.files)} files)")

        lines.append("")

        for file_info in discovered.files:
            size_kb = file_info.size / 1024
            lines.append(f"  - {file_info.name} ({size_kb:.1f}KB) [{file_info.type}]")

        lines.append("[End Directory]\n")

        return "\n".join(lines)

    def _format_glob_context(self, ref) -> str:
        """Format glob pattern results.

        Args:
            ref: ParsedReference with matched_files

        Returns:
            str: Formatted glob results
        """
        if not ref.matched_files:
            return f"\n[Pattern: {ref.original_text} - No matches]\n"

        lines = [f"\n[Pattern: {ref.original_text}]"]
        lines.append(f"Matched {len(ref.matched_files)} files:")
        lines.append("")

        for file_path in ref.matched_files[:10]:  # Limit to 10
            lines.append(f"  - {file_path.name}")

        if len(ref.matched_files) > 10:
            lines.append(f"  ... and {len(ref.matched_files) - 10} more")

        lines.append("[End Pattern]\n")

        return "\n".join(lines)
