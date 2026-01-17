"""
Efficient File Editing Tools

Line-based and pattern-based file editing that doesn't require reading
entire files into memory. Uses Python file I/O for reliable multi-line
content handling (avoids sed newline issues with TerminalTool).

These tools are optimized for large files where reading the entire file
would be wasteful.
"""

from pathlib import Path
from typing import Optional
import re


class EfficientFileEditor:
    """
    Efficient file editing operations that work line-by-line or pattern-based.

    All methods preserve file content and handle multi-line insertions properly.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize efficient file editor.

        Args:
            verbose: Whether to print verbose output for debugging.
        """
        self.verbose = verbose

    def insert_at_line(self, path: str, line_number: int, content: str) -> str:
        """
        Insert content at a specific line number.

        Args:
            path: Path to the file to edit
            line_number: Line number to insert at (1-indexed, content goes BEFORE this line)
            content: Content to insert (can be multi-line)

        Returns:
            str: Success message, or error message

        Example:
            >>> editor = EfficientFileEditor()
            >>> result = editor.insert_at_line("main.py", 10, "    # New comment\\n")
        """
        try:
            file_path = Path(path)

            if not file_path.exists():
                return f"❌ Error: File does not exist: {path}"

            # Read file lines
            lines = file_path.read_text(encoding='utf-8').splitlines(keepends=True)

            # Insert content at specified line (before that line)
            insert_pos = line_number - 1  # Convert to 0-indexed
            if insert_pos < 0:
                insert_pos = 0
            elif insert_pos > len(lines):
                insert_pos = len(lines)

            # Ensure content ends with newline if it doesn't already
            if content and not content.endswith('\n'):
                content += '\n'

            lines.insert(insert_pos, content)

            # Write back
            file_path.write_text(''.join(lines), encoding='utf-8')

            return f"✅ Content inserted at line {line_number} in: {path}"

        except Exception as e:
            return f"❌ Exception in insert_at_line: {e}"

    def replace_lines(self, path: str, start_line: int, end_line: int, new_content: str) -> str:
        """
        Replace a range of lines with new content.

        Args:
            path: Path to the file to edit
            start_line: First line to replace (1-indexed, inclusive)
            end_line: Last line to replace (inclusive)
            new_content: New content to replace the lines with (can be multi-line)

        Returns:
            str: Success message, or error message

        Example:
            >>> editor = EfficientFileEditor()
            >>> result = editor.replace_lines("config.py", 5, 7, "# New config\\nDEBUG = True\\n")
        """
        try:
            file_path = Path(path)

            if not file_path.exists():
                return f"❌ Error: File does not exist: {path}"

            # Read file lines
            lines = file_path.read_text(encoding='utf-8').splitlines(keepends=True)

            # Convert to 0-indexed
            start_idx = start_line - 1
            end_idx = end_line  # end_line is inclusive, so we don't subtract 1 for slicing

            # Validate range
            if start_idx < 0 or end_idx > len(lines):
                return f"❌ Error: Line range {start_line}-{end_line} is out of bounds (file has {len(lines)} lines)"

            # Ensure content ends with newline if it doesn't already
            if new_content and not new_content.endswith('\n'):
                new_content += '\n'

            # Replace the range
            lines[start_idx:end_idx] = [new_content]

            # Write back
            file_path.write_text(''.join(lines), encoding='utf-8')

            return f"✅ Replaced lines {start_line}-{end_line} in: {path}"

        except Exception as e:
            return f"❌ Exception in replace_lines: {e}"

    def insert_after_pattern(self, path: str, pattern: str, content: str, first_match: bool = True) -> str:
        """
        Insert content after a pattern match.

        Args:
            path: Path to the file to edit
            pattern: Pattern to search for (regex supported)
            content: Content to insert after the pattern (can be multi-line)
            first_match: If True, only insert after first match; if False, after all matches

        Returns:
            str: Success message, or error message

        Example:
            >>> editor = EfficientFileEditor()
            >>> result = editor.insert_after_pattern("main.py", "^def main", "    # Function body\\n")
        """
        try:
            file_path = Path(path)

            if not file_path.exists():
                return f"❌ Error: File does not exist: {path}"

            # Read file lines
            lines = file_path.read_text(encoding='utf-8').splitlines(keepends=True)

            # Compile regex pattern
            try:
                regex = re.compile(pattern)
            except re.error as e:
                return f"❌ Error: Invalid regex pattern: {e}"

            # Ensure content ends with newline
            if content and not content.endswith('\n'):
                content += '\n'

            # Find matches and insert
            matches_found = 0
            offset = 0  # Track offset as we insert

            for i, line in enumerate(lines[:]):  # Iterate over copy
                if regex.search(line):
                    matches_found += 1
                    # Insert after this line (i + offset + 1)
                    lines.insert(i + offset + 1, content)
                    offset += 1

                    if first_match:
                        break

            if matches_found == 0:
                return f"❌ Error: Pattern '{pattern}' not found in file"

            # Write back
            file_path.write_text(''.join(lines), encoding='utf-8')

            match_type = "first match" if first_match else f"{matches_found} matches"
            return f"✅ Content inserted after '{pattern}' ({match_type}) in: {path}"

        except Exception as e:
            return f"❌ Exception in insert_after_pattern: {e}"

    def insert_before_pattern(self, path: str, pattern: str, content: str, first_match: bool = True) -> str:
        """
        Insert content before a pattern match.

        Args:
            path: Path to the file to edit
            pattern: Pattern to search for (regex supported)
            content: Content to insert before the pattern (can be multi-line)
            first_match: If True, only insert before first match; if False, before all matches

        Returns:
            str: Success message, or error message

        Example:
            >>> editor = EfficientFileEditor()
            >>> result = editor.insert_before_pattern("main.py", "^if __name__", "# Entry point\\n")
        """
        try:
            file_path = Path(path)

            if not file_path.exists():
                return f"❌ Error: File does not exist: {path}"

            # Read file lines
            lines = file_path.read_text(encoding='utf-8').splitlines(keepends=True)

            # Compile regex pattern
            try:
                regex = re.compile(pattern)
            except re.error as e:
                return f"❌ Error: Invalid regex pattern: {e}"

            # Ensure content ends with newline
            if content and not content.endswith('\n'):
                content += '\n'

            # Find matches and insert
            matches_found = 0
            offset = 0  # Track offset as we insert

            for i, line in enumerate(lines[:]):  # Iterate over copy
                if regex.search(line):
                    matches_found += 1
                    # Insert before this line (i + offset)
                    lines.insert(i + offset, content)
                    offset += 1

                    if first_match:
                        break

            if matches_found == 0:
                return f"❌ Error: Pattern '{pattern}' not found in file"

            # Write back
            file_path.write_text(''.join(lines), encoding='utf-8')

            match_type = "first match" if first_match else f"{matches_found} matches"
            return f"✅ Content inserted before '{pattern}' ({match_type}) in: {path}"

        except Exception as e:
            return f"❌ Exception in insert_before_pattern: {e}"


# --- Standalone Tool Functions for AgenticStepProcessor ---

_efficient_editor = None

def _get_efficient_editor():
    """Lazy initialization of shared EfficientFileEditor instance."""
    global _efficient_editor
    if _efficient_editor is None:
        _efficient_editor = EfficientFileEditor(verbose=False)
    return _efficient_editor


def insert_at_line(path: str, line_number: int, content: str) -> str:
    """Insert content at specific line. LLM-friendly standalone function."""
    return _get_efficient_editor().insert_at_line(path, line_number, content)


def replace_lines(path: str, start_line: int, end_line: int, new_content: str) -> str:
    """Replace line range with new content. LLM-friendly standalone function."""
    return _get_efficient_editor().replace_lines(path, start_line, end_line, new_content)


def insert_after_pattern(path: str, pattern: str, content: str, first_match: bool = True) -> str:
    """Insert content after pattern match. LLM-friendly standalone function."""
    return _get_efficient_editor().insert_after_pattern(path, pattern, content, first_match)


def insert_before_pattern(path: str, pattern: str, content: str, first_match: bool = True) -> str:
    """Insert content before pattern match. LLM-friendly standalone function."""
    return _get_efficient_editor().insert_before_pattern(path, pattern, content, first_match)


# --- Example Usage and Testing ---
if __name__ == "__main__":
    import tempfile
    import shutil

    # Create temporary directory for testing
    test_dir = Path(tempfile.mkdtemp(prefix="efficient_edit_test_"))
    print(f"Testing EfficientFileEditor in: {test_dir}\n")

    try:
        editor = EfficientFileEditor(verbose=True)

        # Create test file
        test_file = test_dir / "test.py"
        initial_content = """# Test Python File
import os
import sys

def main():
    print("Hello World")
    return 0

if __name__ == "__main__":
    main()
"""
        test_file.write_text(initial_content, encoding='utf-8')

        print("--- Initial Content ---")
        print(test_file.read_text())

        print("\n--- Test 1: insert_at_line (add comment at line 4) ---")
        result = editor.insert_at_line(str(test_file), 4, "# This is a new comment\n")
        print(result)
        print(test_file.read_text())

        print("\n--- Test 2: insert_after_pattern (add code after 'def main') ---")
        result = editor.insert_after_pattern(str(test_file), r"^def main", "    # Function documentation\n")
        print(result)
        print(test_file.read_text())

        print("\n--- Test 3: insert_before_pattern (add comment before if __name__) ---")
        result = editor.insert_before_pattern(str(test_file), r"^if __name__", "# Entry point\n")
        print(result)
        print(test_file.read_text())

        print("\n--- Test 4: replace_lines (replace lines 7-8 with new code) ---")
        new_code = """    name = input("Enter your name: ")
    print(f"Hello {name}")
"""
        result = editor.replace_lines(str(test_file), 8, 9, new_code)
        print(result)
        print(test_file.read_text())

        print("\n--- Test 5: insert_at_line with proper indentation (add if/then block) ---")
        if_block = """    if name:
        print("Valid name!")
    else:
        print("Empty name!")
"""
        result = editor.insert_at_line(str(test_file), 11, if_block)
        print(result)
        print(test_file.read_text())

        print("\n✅ All Tests Complete!")

    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"\nCleaned up: {test_dir}")
