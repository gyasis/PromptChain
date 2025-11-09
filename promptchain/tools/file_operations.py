"""
File Operations Wrapper

Pseudo-tool wrappers that match LLM training data expectations while
using the robust TerminalTool underneath. Prevents tool hallucination
by providing the exact tool names LLMs expect from their training.

Pattern inspired by ripgrep_wrapper.py - lightweight wrappers around
powerful tools with comprehensive error handling.
"""

from pathlib import Path
from typing import Optional
import sys


class FileOperations:
    """
    Lightweight wrapper providing LLM-expected file operation tools.

    All operations delegate to TerminalTool for consistency, security,
    and centralized logging. This prevents LLM hallucination of tools
    like file_read, file_write, etc. that don't exist.

    Designed to be used with AgenticStepProcessor and PromptChain agents.
    """

    def __init__(self, terminal_tool=None, verbose: bool = False):
        """
        Initialize FileOperations wrapper.

        Args:
            terminal_tool: Optional TerminalTool instance to use. If None,
                          creates a new instance with default settings.
            verbose: Whether to print verbose output for debugging.
        """
        self.verbose = verbose

        # Import here to avoid circular dependency
        if terminal_tool is None:
            from promptchain.tools.terminal.terminal_tool import TerminalTool
            self.terminal = TerminalTool(verbose=verbose)
        else:
            self.terminal = terminal_tool

    def file_read(self, path: str) -> str:
        """
        Read the contents of a file.

        Args:
            path: Path to the file to read (relative or absolute)

        Returns:
            str: File contents, or error message if read fails

        Example:
            >>> ops = FileOperations()
            >>> content = ops.file_read("config.json")
        """
        try:
            # Validate path exists
            file_path = Path(path)
            if not file_path.exists():
                return f"❌ Error: File does not exist: {path}"

            if not file_path.is_file():
                return f"❌ Error: Path is not a file: {path}"

            # Use cat command via terminal tool (TerminalTool is callable, returns string)
            output = self.terminal(f"cat {self._quote_path(path)}")
            return output

        except Exception as e:
            return f"❌ Exception in file_read: {e}"

    def file_write(self, path: str, content: str) -> str:
        """
        Write content to a file, creating it if it doesn't exist.
        Overwrites existing files completely.

        Args:
            path: Path to the file to write (relative or absolute)
            content: Content to write to the file

        Returns:
            str: Success message with path, or error message

        Example:
            >>> ops = FileOperations()
            >>> result = ops.file_write("output.txt", "Hello World")
        """
        try:
            file_path = Path(path)

            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Use Python's file writing directly - more reliable than shell heredoc
            # which TerminalTool collapses into single-line commands
            file_path.write_text(content, encoding='utf-8')

            # Verify file was created
            if file_path.exists():
                size = file_path.stat().st_size
                return f"✅ File written successfully: {path} ({size} bytes)"
            else:
                return f"❌ File write succeeded but file not found: {path}"

        except Exception as e:
            return f"❌ Exception in file_write: {e}"

    def file_edit(self, path: str, old_text: str, new_text: str) -> str:
        """
        Edit a file by replacing old_text with new_text using sed.

        Args:
            path: Path to the file to edit
            old_text: Text to find and replace
            new_text: Text to replace with

        Returns:
            str: Success message, or error message

        Example:
            >>> ops = FileOperations()
            >>> result = ops.file_edit("config.py", "DEBUG = False", "DEBUG = True")
        """
        try:
            file_path = Path(path)

            if not file_path.exists():
                return f"❌ Error: File does not exist: {path}"

            # Escape special characters for sed
            old_escaped = old_text.replace("/", "\\/").replace("&", "\\&")
            new_escaped = new_text.replace("/", "\\/").replace("&", "\\&")

            # Use sed for in-place replacement
            command = f"sed -i 's/{old_escaped}/{new_escaped}/g' {self._quote_path(path)}"

            self.terminal(command)
            return f"✅ File edited successfully: {path}"

        except Exception as e:
            return f"❌ Exception in file_edit: {e}"

    def file_append(self, path: str, content: str) -> str:
        """
        Append content to the end of a file.

        Args:
            path: Path to the file to append to
            content: Content to append

        Returns:
            str: Success message, or error message

        Example:
            >>> ops = FileOperations()
            >>> result = ops.file_append("log.txt", "New log entry\\n")
        """
        try:
            file_path = Path(path)

            # Create file if it doesn't exist
            if not file_path.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.touch()

            # Use Python's append mode - more reliable than shell heredoc
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)

            size = file_path.stat().st_size
            return f"✅ Content appended successfully: {path} ({size} bytes total)"

        except Exception as e:
            return f"❌ Exception in file_append: {e}"

    def file_delete(self, path: str) -> str:
        """
        Delete a file.

        Args:
            path: Path to the file to delete

        Returns:
            str: Success message, or error message

        Example:
            >>> ops = FileOperations()
            >>> result = ops.file_delete("temp.txt")
        """
        try:
            file_path = Path(path)

            if not file_path.exists():
                return f"❌ Error: File does not exist: {path}"

            if not file_path.is_file():
                return f"❌ Error: Path is not a file (use delete_directory for directories): {path}"

            # Use rm command
            command = f"rm {self._quote_path(path)}"

            self.terminal(command)
            return f"✅ File deleted successfully: {path}"

        except Exception as e:
            return f"❌ Exception in file_delete: {e}"

    def list_directory(self, path: str = ".") -> str:
        """
        List contents of a directory.

        Args:
            path: Path to directory to list (defaults to current directory)

        Returns:
            str: Directory listing, or error message

        Example:
            >>> ops = FileOperations()
            >>> listing = ops.list_directory("src/")
        """
        try:
            dir_path = Path(path)

            if not dir_path.exists():
                return f"❌ Error: Directory does not exist: {path}"

            if not dir_path.is_dir():
                return f"❌ Error: Path is not a directory: {path}"

            # Use ls -lah for detailed listing
            command = f"ls -lah {self._quote_path(path)}"

            output = self.terminal(command)
            return output

        except Exception as e:
            return f"❌ Exception in list_directory: {e}"

    def create_directory(self, path: str) -> str:
        """
        Create a directory and all parent directories.

        Args:
            path: Path to directory to create

        Returns:
            str: Success message, or error message

        Example:
            >>> ops = FileOperations()
            >>> result = ops.create_directory("data/output/results")
        """
        try:
            # Use mkdir -p to create parent directories
            command = f"mkdir -p {self._quote_path(path)}"

            self.terminal(command)

            dir_path = Path(path)
            if dir_path.exists():
                return f"✅ Directory created successfully: {path}"
            else:
                return f"❌ mkdir succeeded but directory not found: {path}"

        except Exception as e:
            return f"❌ Exception in create_directory: {e}"

    def read_file_range(self, path: str, start_line: int, end_line: int) -> str:
        """
        Read a specific range of lines from a file.

        Args:
            path: Path to the file to read
            start_line: First line to read (1-indexed)
            end_line: Last line to read (inclusive)

        Returns:
            str: Requested lines, or error message

        Example:
            >>> ops = FileOperations()
            >>> lines = ops.read_file_range("main.py", 10, 20)
        """
        try:
            file_path = Path(path)

            if not file_path.exists():
                return f"❌ Error: File does not exist: {path}"

            if not file_path.is_file():
                return f"❌ Error: Path is not a file: {path}"

            # Use sed to extract line range
            command = f"sed -n '{start_line},{end_line}p' {self._quote_path(path)}"

            output = self.terminal(command)
            return output

        except Exception as e:
            return f"❌ Exception in read_file_range: {e}"

    def insert_at_line(self, path: str, line_number: int, content: str) -> str:
        """
        Insert content at a specific line number (efficient for large files).

        Args:
            path: Path to the file to edit
            line_number: Line number to insert at (1-indexed, content goes BEFORE this line)
            content: Content to insert

        Returns:
            str: Success message, or error message

        Example:
            >>> ops = FileOperations()
            >>> result = ops.insert_at_line("main.py", 10, "    # New comment\\n")
        """
        try:
            file_path = Path(path)

            if not file_path.exists():
                return f"❌ Error: File does not exist: {path}"

            # Escape content for sed (replace single quotes)
            safe_content = content.replace("'", "'\\''")

            # Use sed to insert at specific line (inserts BEFORE the line number)
            # sed '10i\content' inserts before line 10
            command = f"sed -i '{line_number}i\\{safe_content}' {self._quote_path(path)}"

            self.terminal(command)
            return f"✅ Content inserted at line {line_number} in: {path}"

        except Exception as e:
            return f"❌ Exception in insert_at_line: {e}"

    def replace_lines(self, path: str, start_line: int, end_line: int, new_content: str) -> str:
        """
        Replace a range of lines with new content (efficient for large files).

        Args:
            path: Path to the file to edit
            start_line: First line to replace (1-indexed, inclusive)
            end_line: Last line to replace (inclusive)
            new_content: New content to replace the lines with

        Returns:
            str: Success message, or error message

        Example:
            >>> ops = FileOperations()
            >>> result = ops.replace_lines("config.py", 5, 7, "# New config section\\n")
        """
        try:
            file_path = Path(path)

            if not file_path.exists():
                return f"❌ Error: File does not exist: {path}"

            # Escape content for sed
            safe_content = new_content.replace("'", "'\\''")

            # Use sed to delete lines and insert new content
            # First delete the range, then insert new content at start position
            command = f"sed -i '{start_line},{end_line}d' {self._quote_path(path)} && sed -i '{start_line}i\\{safe_content}' {self._quote_path(path)}"

            self.terminal(command)
            return f"✅ Replaced lines {start_line}-{end_line} in: {path}"

        except Exception as e:
            return f"❌ Exception in replace_lines: {e}"

    def insert_after_pattern(self, path: str, pattern: str, content: str, first_match: bool = True) -> str:
        """
        Insert content after a pattern match (efficient for large files).

        Args:
            path: Path to the file to edit
            pattern: Pattern to search for (regex supported)
            content: Content to insert after the pattern
            first_match: If True, only insert after first match; if False, after all matches

        Returns:
            str: Success message, or error message

        Example:
            >>> ops = FileOperations()
            >>> result = ops.insert_after_pattern("main.py", "^def main", "    # Function body\\n")
        """
        try:
            file_path = Path(path)

            if not file_path.exists():
                return f"❌ Error: File does not exist: {path}"

            # Escape content for sed
            safe_content = content.replace("'", "'\\''").replace("/", "\\/")
            safe_pattern = pattern.replace("/", "\\/")

            # Use sed to insert after pattern
            # /pattern/a\content inserts after matching line
            if first_match:
                # Only first match: 0,/pattern/
                command = f"sed -i '0,/{safe_pattern}/a\\{safe_content}' {self._quote_path(path)}"
            else:
                # All matches
                command = f"sed -i '/{safe_pattern}/a\\{safe_content}' {self._quote_path(path)}"

            self.terminal(command)
            match_type = "first match" if first_match else "all matches"
            return f"✅ Content inserted after '{pattern}' ({match_type}) in: {path}"

        except Exception as e:
            return f"❌ Exception in insert_after_pattern: {e}"

    def insert_before_pattern(self, path: str, pattern: str, content: str, first_match: bool = True) -> str:
        """
        Insert content before a pattern match (efficient for large files).

        Args:
            path: Path to the file to edit
            pattern: Pattern to search for (regex supported)
            content: Content to insert before the pattern
            first_match: If True, only insert before first match; if False, before all matches

        Returns:
            str: Success message, or error message

        Example:
            >>> ops = FileOperations()
            >>> result = ops.insert_before_pattern("main.py", "^if __name__", "# Entry point\\n")
        """
        try:
            file_path = Path(path)

            if not file_path.exists():
                return f"❌ Error: File does not exist: {path}"

            # Escape content for sed
            safe_content = content.replace("'", "'\\''").replace("/", "\\/")
            safe_pattern = pattern.replace("/", "\\/")

            # Use sed to insert before pattern
            # /pattern/i\content inserts before matching line
            if first_match:
                # Only first match
                command = f"sed -i '0,/{safe_pattern}/i\\{safe_content}' {self._quote_path(path)}"
            else:
                # All matches
                command = f"sed -i '/{safe_pattern}/i\\{safe_content}' {self._quote_path(path)}"

            self.terminal(command)
            match_type = "first match" if first_match else "all matches"
            return f"✅ Content inserted before '{pattern}' ({match_type}) in: {path}"

        except Exception as e:
            return f"❌ Exception in insert_before_pattern: {e}"

    def _quote_path(self, path: str) -> str:
        """
        Properly quote a file path for shell commands.

        Args:
            path: Path to quote

        Returns:
            str: Quoted path safe for shell
        """
        # Use double quotes and escape special characters
        return f'"{path}"'


# --- Standalone Tool Functions for AgenticStepProcessor ---
# These are convenience functions that match exact LLM expectations

_file_ops = None

def _get_file_ops():
    """Lazy initialization of shared FileOperations instance."""
    global _file_ops
    if _file_ops is None:
        _file_ops = FileOperations(verbose=False)
    return _file_ops


def file_read(path: str) -> str:
    """Read file contents. LLM-friendly standalone function."""
    return _get_file_ops().file_read(path)


def file_write(path: str, content: str) -> str:
    """Write file contents. LLM-friendly standalone function."""
    return _get_file_ops().file_write(path, content)


def file_edit(path: str, old_text: str, new_text: str) -> str:
    """Edit file by replacing text. LLM-friendly standalone function."""
    return _get_file_ops().file_edit(path, old_text, new_text)


def file_append(path: str, content: str) -> str:
    """Append to file. LLM-friendly standalone function."""
    return _get_file_ops().file_append(path, content)


def file_delete(path: str) -> str:
    """Delete file. LLM-friendly standalone function."""
    return _get_file_ops().file_delete(path)


def list_directory(path: str = ".") -> str:
    """List directory contents. LLM-friendly standalone function."""
    return _get_file_ops().list_directory(path)


def create_directory(path: str) -> str:
    """Create directory. LLM-friendly standalone function."""
    return _get_file_ops().create_directory(path)


def read_file_range(path: str, start_line: int, end_line: int) -> str:
    """Read file line range. LLM-friendly standalone function."""
    return _get_file_ops().read_file_range(path, start_line, end_line)


def insert_at_line(path: str, line_number: int, content: str) -> str:
    """Insert content at specific line. LLM-friendly standalone function."""
    return _get_file_ops().insert_at_line(path, line_number, content)


def replace_lines(path: str, start_line: int, end_line: int, new_content: str) -> str:
    """Replace line range with new content. LLM-friendly standalone function."""
    return _get_file_ops().replace_lines(path, start_line, end_line, new_content)


def insert_after_pattern(path: str, pattern: str, content: str, first_match: bool = True) -> str:
    """Insert content after pattern match. LLM-friendly standalone function."""
    return _get_file_ops().insert_after_pattern(path, pattern, content, first_match)


def insert_before_pattern(path: str, pattern: str, content: str, first_match: bool = True) -> str:
    """Insert content before pattern match. LLM-friendly standalone function."""
    return _get_file_ops().insert_before_pattern(path, pattern, content, first_match)


# --- Example Usage ---
if __name__ == "__main__":
    import tempfile
    import shutil

    # Create temporary directory for testing
    test_dir = Path(tempfile.mkdtemp(prefix="file_ops_test_"))
    print(f"Testing FileOperations in: {test_dir}")

    try:
        # Create FileOperations with TerminalTool that doesn't require permission (for testing)
        from promptchain.tools.terminal.terminal_tool import TerminalTool
        terminal = TerminalTool(require_permission=False, verbose=False)
        ops = FileOperations(terminal_tool=terminal, verbose=True)

        # Test file_write
        print("\n--- Test file_write ---")
        test_file = test_dir / "test.txt"
        result = ops.file_write(str(test_file), "Hello World\nLine 2\nLine 3")
        print(result)

        # Test file_read
        print("\n--- Test file_read ---")
        content = ops.file_read(str(test_file))
        print(content)

        # Test file_append
        print("\n--- Test file_append ---")
        result = ops.file_append(str(test_file), "\nAppended line")
        print(result)

        # Test read_file_range
        print("\n--- Test read_file_range ---")
        lines = ops.read_file_range(str(test_file), 2, 3)
        print(f"Lines 2-3:\n{lines}")

        # Test file_edit
        print("\n--- Test file_edit ---")
        result = ops.file_edit(str(test_file), "World", "Universe")
        print(result)
        content = ops.file_read(str(test_file))
        print(f"After edit:\n{content}")

        # Test create_directory
        print("\n--- Test create_directory ---")
        nested_dir = test_dir / "nested/deep/path"
        result = ops.create_directory(str(nested_dir))
        print(result)

        # Test list_directory
        print("\n--- Test list_directory ---")
        listing = ops.list_directory(str(test_dir))
        print(listing)

        # Test file_delete
        print("\n--- Test file_delete ---")
        result = ops.file_delete(str(test_file))
        print(result)

        # Test standalone functions
        print("\n--- Test standalone functions ---")
        test_file2 = test_dir / "standalone.txt"
        result = file_write(str(test_file2), "Standalone test")
        print(result)
        content = file_read(str(test_file2))
        print(f"Content: {content}")

    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"\nCleaned up: {test_dir}")
