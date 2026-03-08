"""
Library Tool Registration for PromptChain CLI

Registers PromptChain's core library tools with the CLI tool registry:
- FileOperations (12 file manipulation methods)
- RipgrepSearcher (code search)
- TerminalTool (terminal command execution)

These tools enable AI agents to perform file operations, code search,
and terminal commands through the unified CLI tool registry system.

Token Optimization: By registering library tools, we make them accessible
to agents via the registry, achieving token efficiency and consistent tooling.
"""

import json
from typing import Dict, List, Optional

from promptchain.cli.tools import ParameterSchema, ToolCategory, registry
# Import library tool classes
from promptchain.tools.file_operations import FileOperations
from promptchain.tools.ripgrep_wrapper import RipgrepSearcher
from promptchain.tools.terminal.terminal_tool import TerminalTool

# Import task list tool
from .task_list_tool import (get_task_list_manager, task_list_clear,
                             task_list_get, task_list_write)

# Create shared instances for all registered tools
_file_ops = FileOperations(verbose=False)
_ripgrep = None  # Lazy init (requires ripgrep installed)
_terminal = TerminalTool(require_permission=False, verbose=False)


# ============================================================================
# FILE OPERATIONS TOOLS (12 tools)
# ============================================================================


@registry.register(
    category=ToolCategory.UTILITY,
    description=(
        "FILE READ: Read the ENTIRE contents of a file.\n\n"
        "USE WHEN:\n"
        "- You need the complete file contents\n"
        "- File is small enough to process entirely\n"
        "- Analyzing entire file structure\n\n"
        "DO NOT USE WHEN:\n"
        "- File is very large (use read_file_range instead)\n"
        "- You only need specific lines (use read_file_range)\n"
        "- Searching for patterns (use ripgrep_search)\n\n"
        "SECURITY: Respects session security mode for paths outside working directory."
    ),
    parameters={
        "path": {
            "type": "string",
            "required": True,
            "description": "Path to the file to read (relative or absolute)",
        }
    },
    tags=["file", "read", "io", "filesystem"],
    examples=[
        "file_read('config.json')",
        "file_read('/etc/hosts')",
        "file_read('src/main.py')",
    ],
    capabilities=["file_read", "io"],
)
def file_read(path: str) -> str:
    """Read file contents (CLI registry wrapper)."""
    return _file_ops.file_read(path)


@registry.register(
    category=ToolCategory.UTILITY,
    description=(
        "FILE WRITE: CREATE or OVERWRITE a file with content.\n\n"
        "USE WHEN:\n"
        "- Creating new files from scratch\n"
        "- Completely replacing file contents\n"
        "- Writing generated code, configs, or data\n\n"
        "DO NOT USE WHEN:\n"
        "- You want to ADD to end of file (use file_append)\n"
        "- You want to REPLACE specific text (use file_edit)\n"
        "- You want to MODIFY specific lines (use replace_lines)\n\n"
        "WARNING: OVERWRITES existing files! Use file_read first if unsure.\n"
        "SECURITY: Respects session security mode. Creates parent directories automatically."
    ),
    parameters={
        "path": {
            "type": "string",
            "required": True,
            "description": "Path to the file to write (relative or absolute)",
        },
        "content": {
            "type": "string",
            "required": True,
            "description": "Content to write to the file",
        },
    },
    tags=["file", "write", "io", "filesystem", "create"],
    examples=[
        "file_write('output.txt', 'Hello World')",
        "file_write('data/results.json', json_data)",
        "file_write('/tmp/test.txt', 'Test content\\nLine 2')",
    ],
    capabilities=["file_write", "io"],
)
def file_write(path: str, content: str) -> str:
    """Write file contents (CLI registry wrapper)."""
    return _file_ops.file_write(path, content)


@registry.register(
    category=ToolCategory.UTILITY,
    description=(
        "FILE EDIT: Find and REPLACE TEXT in a file (sed-style).\n\n"
        "USE WHEN:\n"
        "- Changing specific text strings (like renaming variables)\n"
        "- Updating version numbers or config values\n"
        "- Making exact text substitutions\n\n"
        "DO NOT USE WHEN:\n"
        "- Modifying specific line numbers (use replace_lines)\n"
        "- Inserting new content (use insert_at_line or insert_after_pattern)\n"
        "- Replacing entire file (use file_write)\n"
        "- Text appears multiple times but you only want first match\n\n"
        "NOTE: Replaces ALL occurrences. Uses sed internally."
    ),
    parameters={
        "path": {
            "type": "string",
            "required": True,
            "description": "Path to the file to edit",
        },
        "old_text": {
            "type": "string",
            "required": True,
            "description": "Text to find and replace (exact match)",
        },
        "new_text": {
            "type": "string",
            "required": True,
            "description": "Text to replace with",
        },
    },
    tags=["file", "edit", "modify", "replace", "filesystem"],
    examples=[
        "file_edit('config.py', 'DEBUG = False', 'DEBUG = True')",
        "file_edit('README.md', 'v1.0', 'v2.0')",
        "file_edit('settings.json', '\"production\"', '\"development\"')",
    ],
    capabilities=["file_write", "file_edit", "io"],
)
def file_edit(path: str, old_text: str, new_text: str) -> str:
    """Edit file by text replacement (CLI registry wrapper)."""
    return _file_ops.file_edit(path, old_text, new_text)


@registry.register(
    category=ToolCategory.UTILITY,
    description=(
        "FILE APPEND: ADD content to the END of a file.\n\n"
        "USE WHEN:\n"
        "- Adding new entries to log files\n"
        "- Appending rows to CSV/data files\n"
        "- Adding new sections at end of documents\n"
        "- Incrementally building output\n\n"
        "DO NOT USE WHEN:\n"
        "- You want to REPLACE file contents (use file_write)\n"
        "- You want to INSERT in the middle (use insert_at_line)\n"
        "- You want to INSERT after a pattern (use insert_after_pattern)\n\n"
        "NOTE: Creates file if it doesn't exist. Always adds at the END."
    ),
    parameters={
        "path": {
            "type": "string",
            "required": True,
            "description": "Path to the file to append to",
        },
        "content": {
            "type": "string",
            "required": True,
            "description": "Content to append to the file",
        },
    },
    tags=["file", "append", "io", "filesystem", "log"],
    examples=[
        "file_append('log.txt', 'New log entry\\n')",
        "file_append('data.csv', 'new,row,data\\n')",
        "file_append('notes.md', '\\n## New Section\\nContent here')",
    ],
    capabilities=["file_write", "io"],
)
def file_append(path: str, content: str) -> str:
    """Append to file (CLI registry wrapper)."""
    return _file_ops.file_append(path, content)


@registry.register(
    category=ToolCategory.UTILITY,
    description=(
        "FILE DELETE: PERMANENTLY REMOVE a file.\n\n"
        "USE WHEN:\n"
        "- Cleaning up temporary files\n"
        "- Removing outdated or obsolete files\n"
        "- Deleting generated/cached files\n\n"
        "DO NOT USE WHEN:\n"
        "- Deleting directories (not supported - use terminal_execute)\n"
        "- You might need the file later (consider backup first)\n\n"
        "WARNING: PERMANENT deletion - cannot be undone!\n"
        "SECURITY: Respects session security mode. Fails safely if file doesn't exist."
    ),
    parameters={
        "path": {
            "type": "string",
            "required": True,
            "description": "Path to the file to delete",
        }
    },
    tags=["file", "delete", "remove", "filesystem"],
    examples=[
        "file_delete('temp.txt')",
        "file_delete('/tmp/cache.json')",
        "file_delete('old_data.csv')",
    ],
    capabilities=["file_delete", "io"],
)
def file_delete(path: str) -> str:
    """Delete file (CLI registry wrapper)."""
    return _file_ops.file_delete(path)


@registry.register(
    category=ToolCategory.UTILITY,
    description=(
        "LIST DIRECTORY: List contents of ONE directory (ls -lah style).\n\n"
        "USE WHEN:\n"
        "- Viewing files in a specific directory\n"
        "- Checking file sizes, permissions, timestamps\n"
        "- Understanding directory structure\n\n"
        "DO NOT USE WHEN:\n"
        "- Searching for files by pattern (use find_paths)\n"
        "- Searching file contents (use ripgrep_search)\n"
        "- Recursively listing subdirectories (use find_paths)\n\n"
        "NOTE: Lists ONE directory only. Shows permissions, size, modification time."
    ),
    parameters={
        "path": {
            "type": "string",
            "default": ".",
            "description": "Path to directory to list (defaults to current directory)",
        }
    },
    tags=["directory", "list", "filesystem", "ls"],
    examples=[
        "list_directory('.')",
        "list_directory('src/')",
        "list_directory('/home/user/projects')",
    ],
    capabilities=["file_read", "directory_listing"],
)
def list_directory(path: str = ".") -> str:
    """List directory contents (CLI registry wrapper)."""
    return _file_ops.list_directory(path)


@registry.register(
    category=ToolCategory.UTILITY,
    description=(
        "CREATE DIRECTORY: Create directory with parents (mkdir -p behavior).\n\n"
        "USE WHEN:\n"
        "- Creating new directories for output\n"
        "- Setting up project structure\n"
        "- Creating nested directory paths\n\n"
        "DO NOT USE WHEN:\n"
        "- Creating files (use file_write)\n"
        "- Directory already exists (safe to call, but unnecessary)\n\n"
        "NOTE: Creates all parent directories automatically. Safe if directory exists."
    ),
    parameters={
        "path": {
            "type": "string",
            "required": True,
            "description": "Path to directory to create (can be nested)",
        }
    },
    tags=["directory", "create", "filesystem", "mkdir"],
    examples=[
        "create_directory('data/output/results')",
        "create_directory('/tmp/workspace')",
        "create_directory('logs/2025/01')",
    ],
    capabilities=["file_write", "directory_management"],
)
def create_directory(path: str) -> str:
    """Create directory (CLI registry wrapper)."""
    return _file_ops.create_directory(path)


@registry.register(
    category=ToolCategory.UTILITY,
    description=(
        "READ FILE RANGE: Read SPECIFIC LINE NUMBERS from a file.\n\n"
        "USE WHEN:\n"
        "- Reading specific sections of large files\n"
        "- Examining code around specific line numbers\n"
        "- Efficient partial file reading (saves tokens)\n"
        "- Checking specific function/class definitions\n\n"
        "DO NOT USE WHEN:\n"
        "- You need the entire file (use file_read)\n"
        "- Searching for patterns (use ripgrep_search)\n"
        "- File is small enough to read entirely\n\n"
        "NOTE: 1-indexed line numbers. Efficient for large files."
    ),
    parameters={
        "path": {
            "type": "string",
            "required": True,
            "description": "Path to the file to read",
        },
        "start_line": {
            "type": "integer",
            "required": True,
            "description": "First line to read (1-indexed)",
        },
        "end_line": {
            "type": "integer",
            "required": True,
            "description": "Last line to read (inclusive)",
        },
    },
    tags=["file", "read", "range", "lines", "filesystem"],
    examples=[
        "read_file_range('main.py', 10, 20)",
        "read_file_range('log.txt', 1, 100)",
        "read_file_range('data.csv', 500, 600)",
    ],
    capabilities=["file_read", "io"],
)
def read_file_range(path: str, start_line: int, end_line: int) -> str:
    """Read file line range (CLI registry wrapper)."""
    return _file_ops.read_file_range(path, start_line, end_line)


@registry.register(
    category=ToolCategory.UTILITY,
    description=(
        "INSERT AT LINE: Insert content at a SPECIFIC LINE NUMBER.\n\n"
        "USE WHEN:\n"
        "- Adding content at a known line number\n"
        "- Inserting imports at top of file (line 1)\n"
        "- Adding code at specific locations\n\n"
        "DO NOT USE WHEN:\n"
        "- You know the pattern to insert after (use insert_after_pattern)\n"
        "- You want to replace lines (use replace_lines)\n"
        "- You want to append at end (use file_append)\n\n"
        "NOTE: Content inserted BEFORE the specified line. 1-indexed."
    ),
    parameters={
        "path": {
            "type": "string",
            "required": True,
            "description": "Path to the file to edit",
        },
        "line_number": {
            "type": "integer",
            "required": True,
            "description": "Line number to insert at (1-indexed, content goes before this line)",
        },
        "content": {
            "type": "string",
            "required": True,
            "description": "Content to insert (include \\n for newlines)",
        },
    },
    tags=["file", "insert", "edit", "filesystem"],
    examples=[
        "insert_at_line('main.py', 10, '    # New comment\\n')",
        "insert_at_line('config.json', 5, '  \"new_key\": \"value\",\\n')",
        "insert_at_line('README.md', 1, '# Title\\n\\n')",
    ],
    capabilities=["file_write", "file_edit", "io"],
)
def insert_at_line(path: str, line_number: int, content: str) -> str:
    """Insert content at line (CLI registry wrapper)."""
    return _file_ops.insert_at_line(path, line_number, content)


@registry.register(
    category=ToolCategory.UTILITY,
    description=(
        "REPLACE LINES: Replace a LINE RANGE with new content.\n\n"
        "USE WHEN:\n"
        "- Replacing entire functions or code blocks\n"
        "- Updating sections you know the line numbers of\n"
        "- Efficient edits to specific line ranges\n\n"
        "DO NOT USE WHEN:\n"
        "- Replacing specific text strings (use file_edit)\n"
        "- Inserting without removing (use insert_at_line)\n"
        "- You don't know the line numbers (use ripgrep_search first)\n\n"
        "NOTE: Deletes lines start_line to end_line (inclusive), then inserts new_content."
    ),
    parameters={
        "path": {
            "type": "string",
            "required": True,
            "description": "Path to the file to edit",
        },
        "start_line": {
            "type": "integer",
            "required": True,
            "description": "First line to replace (1-indexed, inclusive)",
        },
        "end_line": {
            "type": "integer",
            "required": True,
            "description": "Last line to replace (inclusive)",
        },
        "new_content": {
            "type": "string",
            "required": True,
            "description": "New content to replace the lines with",
        },
    },
    tags=["file", "replace", "edit", "filesystem"],
    examples=[
        "replace_lines('config.py', 5, 7, '# New config section\\n')",
        "replace_lines('main.py', 20, 25, 'def new_function():\\n    pass\\n')",
        "replace_lines('README.md', 10, 15, '## Updated Section\\n\\nNew content\\n')",
    ],
    capabilities=["file_write", "file_edit", "io"],
)
def replace_lines(path: str, start_line: int, end_line: int, new_content: str) -> str:
    """Replace line range (CLI registry wrapper)."""
    return _file_ops.replace_lines(path, start_line, end_line, new_content)


@registry.register(
    category=ToolCategory.UTILITY,
    description=(
        "INSERT AFTER PATTERN: Insert content AFTER a regex pattern match.\n\n"
        "USE WHEN:\n"
        "- Adding code after specific functions ('^def main')\n"
        "- Inserting after import blocks ('import .*')\n"
        "- Adding content after sections in markdown\n\n"
        "DO NOT USE WHEN:\n"
        "- You know the exact line number (use insert_at_line)\n"
        "- You want to insert BEFORE the pattern (use insert_before_pattern)\n"
        "- You want to REPLACE content (use file_edit or replace_lines)\n\n"
        "NOTE: Supports regex. Use first_match=False to insert after ALL matches."
    ),
    parameters={
        "path": {
            "type": "string",
            "required": True,
            "description": "Path to the file to edit",
        },
        "pattern": {
            "type": "string",
            "required": True,
            "description": "Pattern to search for (regex supported, e.g., '^def main', 'import .*')",
        },
        "content": {
            "type": "string",
            "required": True,
            "description": "Content to insert after the pattern",
        },
        "first_match": {
            "type": "boolean",
            "default": True,
            "description": "If true, insert only after first match; if false, after all matches",
        },
    },
    tags=["file", "insert", "pattern", "regex", "filesystem"],
    examples=[
        "insert_after_pattern('main.py', '^def main', '    # Function body\\n', first_match=True)",
        "insert_after_pattern('config.py', 'import .*', '\\n# Additional imports\\n', first_match=False)",
        "insert_after_pattern('README.md', '^## Installation', '\\nDetailed steps...\\n')",
    ],
    capabilities=["file_write", "file_edit", "io", "regex"],
)
def insert_after_pattern(
    path: str, pattern: str, content: str, first_match: bool = True
) -> str:
    """Insert after pattern (CLI registry wrapper)."""
    return _file_ops.insert_after_pattern(path, pattern, content, first_match)


@registry.register(
    category=ToolCategory.UTILITY,
    description=(
        "INSERT BEFORE PATTERN: Insert content BEFORE a regex pattern match.\n\n"
        "USE WHEN:\n"
        "- Adding content before __main__ block ('^if __name__')\n"
        "- Inserting headers before sections\n"
        "- Adding docstrings before functions\n\n"
        "DO NOT USE WHEN:\n"
        "- You know the exact line number (use insert_at_line)\n"
        "- You want to insert AFTER the pattern (use insert_after_pattern)\n"
        "- You want to REPLACE content (use file_edit or replace_lines)\n\n"
        "NOTE: Supports regex. Use first_match=False to insert before ALL matches."
    ),
    parameters={
        "path": {
            "type": "string",
            "required": True,
            "description": "Path to the file to edit",
        },
        "pattern": {
            "type": "string",
            "required": True,
            "description": "Pattern to search for (regex supported)",
        },
        "content": {
            "type": "string",
            "required": True,
            "description": "Content to insert before the pattern",
        },
        "first_match": {
            "type": "boolean",
            "default": True,
            "description": "If true, insert only before first match; if false, before all matches",
        },
    },
    tags=["file", "insert", "pattern", "regex", "filesystem"],
    examples=[
        "insert_before_pattern('main.py', '^if __name__', '# Entry point\\n', first_match=True)",
        "insert_before_pattern('test.py', '^def test_', '\\n# Test case\\n', first_match=False)",
        "insert_before_pattern('README.md', '^## License', '\\n---\\n')",
    ],
    capabilities=["file_write", "file_edit", "io", "regex"],
)
def insert_before_pattern(
    path: str, pattern: str, content: str, first_match: bool = True
) -> str:
    """Insert before pattern (CLI registry wrapper)."""
    return _file_ops.insert_before_pattern(path, pattern, content, first_match)


# ============================================================================
# RIPGREP SEARCH TOOL
# ============================================================================


def _get_ripgrep() -> RipgrepSearcher:
    """Lazy initialization of RipgrepSearcher (requires ripgrep installed)."""
    global _ripgrep
    if _ripgrep is None:
        try:
            _ripgrep = RipgrepSearcher()
        except Exception as e:
            # Return a mock that explains the error with clear installation instructions
            error_msg = str(e)

            class MockRipgrep:
                def search(self, *args, **kwargs):
                    return (
                        f"❌ RIPGREP NOT INSTALLED\n\n"
                        f"Error: {error_msg}\n\n"
                        f"Install ripgrep to enable code search:\n"
                        f"  Ubuntu/Debian: sudo apt install ripgrep\n"
                        f"  macOS: brew install ripgrep\n"
                        f"  Cargo: cargo install ripgrep\n\n"
                        f"ALTERNATIVE: Use terminal_execute('grep -r \"pattern\" .') for basic search, "
                        f"or find_paths('*.py') to find files by pattern."
                    )

            _ripgrep = MockRipgrep()  # type: ignore[assignment]
    return _ripgrep  # type: ignore[return-value]


@registry.register(
    category=ToolCategory.UTILITY,
    description=(
        "RIPGREP SEARCH: Search LOCAL FILE CONTENTS (NOT the internet!).\n\n"
        "USE WHEN:\n"
        "- Searching for code patterns (functions, classes, variables)\n"
        "- Finding TODOs, FIXMEs, error messages in code\n"
        "- Locating usages of specific strings/patterns\n"
        "- Searching across large codebases (very fast)\n\n"
        "DO NOT USE WHEN:\n"
        "- Searching the web/internet (this is LOCAL files only!)\n"
        "- Finding files by NAME (use find_paths instead)\n"
        "- You need file metadata (use path_info or list_directory)\n\n"
        "NOTE: Uses ripgrep (rg). Much faster than grep. Supports regex."
    ),
    parameters={
        "query": {
            "type": "string",
            "required": True,
            "description": "Pattern to search for (regex by default, or literal if fixed_strings=true)",
        },
        "search_path": {
            "type": "string",
            "default": ".",
            "description": "Directory or file to search within (defaults to current directory)",
        },
        "case_sensitive": {
            "type": "boolean",
            "description": "Force case-sensitive (true) or case-insensitive (false). Default: smart case",
        },
        "include_patterns": {
            "type": "array",
            "items": ParameterSchema(
                name="pattern", type="string", description="Glob pattern"
            ),
            "description": "List of glob patterns for files to include (e.g., ['*.py', '*.js'])",
        },
        "exclude_patterns": {
            "type": "array",
            "items": ParameterSchema(
                name="pattern", type="string", description="Glob pattern"
            ),
            "description": "List of glob patterns for files to exclude (e.g., ['*.test.py', '*/node_modules/*'])",
        },
        "fixed_strings": {
            "type": "boolean",
            "default": False,
            "description": "Treat query as literal string (not regex)",
        },
        "word_regexp": {
            "type": "boolean",
            "default": False,
            "description": "Match only whole words",
        },
        "multiline": {
            "type": "boolean",
            "default": False,
            "description": "Allow matches to span multiple lines",
        },
    },
    tags=["search", "code", "ripgrep", "rg", "grep", "find"],
    examples=[
        "ripgrep_search('def main', search_path='src/')",
        "ripgrep_search('TODO', include_patterns=['*.py'], case_sensitive=False)",
        "ripgrep_search('class.*Model', search_path='app/', exclude_patterns=['*test*'])",
    ],
    capabilities=["code_search", "text_search", "regex"],
)
def ripgrep_search(
    query: str,
    search_path: str = ".",
    case_sensitive: Optional[bool] = None,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    fixed_strings: bool = False,
    word_regexp: bool = False,
    multiline: bool = False,
) -> str:
    """Search code with ripgrep (CLI registry wrapper)."""
    try:
        rg = _get_ripgrep()
        results = rg.search(
            query=query,
            search_path=search_path,
            case_sensitive=case_sensitive,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            fixed_strings=fixed_strings,
            word_regexp=word_regexp,
            multiline=multiline,
        )

        if not results:
            return f"No matches found for: {query}"

        # Format results nicely
        result_str = f"🔍 Found {len(results)} matches for '{query}':\n\n"
        result_str += "\n".join(results)
        return result_str
    except Exception as e:
        return f"❌ Error in ripgrep_search: {e}"


# ============================================================================
# TERMINAL TOOL
# ============================================================================


@registry.register(
    category=ToolCategory.UTILITY,
    description=(
        "TERMINAL EXECUTE: Run shell commands with REAL SYSTEM ACCESS.\n\n"
        "USE WHEN:\n"
        "- Running git commands (status, commit, push)\n"
        "- Installing packages (pip, npm, apt)\n"
        "- Running scripts (python script.py)\n"
        "- System operations not covered by other tools\n\n"
        "DO NOT USE WHEN:\n"
        "- Reading files (use file_read or read_file_range)\n"
        "- Writing files (use file_write or file_edit)\n"
        "- Listing directories (use list_directory)\n"
        "- Searching code (use ripgrep_search)\n\n"
        "SECURITY: Commands run with real system access. "
        "Respects session security mode for path-related commands."
    ),
    parameters={
        "command": {
            "type": "string",
            "required": True,
            "description": "Shell command to execute (e.g., 'ls -la', 'git status', 'python script.py')",
        }
    },
    tags=["terminal", "shell", "command", "execute", "bash"],
    examples=[
        "terminal_execute('ls -la')",
        "terminal_execute('git status')",
        "terminal_execute('python --version')",
        "terminal_execute('cat config.json')",
    ],
    capabilities=["shell_execute", "system_command"],
)
def terminal_execute(command: str) -> str:
    """Execute terminal command (CLI registry wrapper)."""
    try:
        # TerminalTool is callable - returns formatted output string
        result = _terminal(command)
        return result
    except Exception as e:
        return f"❌ Error in terminal_execute: {e}"


# ============================================================================
# TASK LIST TOOL
# ============================================================================


@registry.register(
    category=ToolCategory.UTILITY,
    description="""Create or update a task list for tracking progress on complex multi-step tasks.

Use this tool when:
1. A task requires 3 or more distinct steps
2. You need to track progress on a complex query
3. The user provides multiple tasks to complete
4. You want to show the user your progress

IMPORTANT RULES:
- Only ONE task should be "in_progress" at a time
- Mark tasks as "completed" IMMEDIATELY after finishing
- Always provide both "content" and "activeForm" for each task

Task status values:
- "pending": Task not yet started
- "in_progress": Currently working on this task (max ONE at a time)
- "completed": Task finished successfully""",
    parameters={
        "todos": {
            "type": "array",
            "required": True,
            "description": "Array of task objects with content, status, and activeForm fields",
            "items": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Task description (imperative form)",
                    },
                    "status": {
                        "type": "string",
                        "description": "pending|in_progress|completed",
                    },
                    "activeForm": {
                        "type": "string",
                        "description": "Task in present continuous form",
                    },
                },
            },
        }
    },
    tags=["task", "todo", "progress", "tracking", "planning"],
    examples=[
        'task_list_write([{"content": "Search codebase", "status": "completed", "activeForm": "Searching codebase"}, {"content": "Analyze results", "status": "in_progress", "activeForm": "Analyzing results"}])'
    ],
    capabilities=["task_management", "progress_tracking"],
)
def task_list_write_tool(todos: list) -> str:
    """Create or update task list (CLI registry wrapper)."""
    try:
        return task_list_write(json.dumps(todos))
    except Exception as e:
        return f"Error updating task list: {e}"


# ============================================================================
# REGISTRATION SUMMARY
# ============================================================================


def register_all_tools():
    """
    Register all library tools with the CLI registry.

    This function is called automatically when the module is imported,
    but can also be called explicitly to re-register tools.

    Returns:
        List[str]: Names of registered tools
    """
    registered_tools = [
        # File operations (12 tools)
        "file_read",
        "file_write",
        "file_edit",
        "file_append",
        "file_delete",
        "list_directory",
        "create_directory",
        "read_file_range",
        "insert_at_line",
        "replace_lines",
        "insert_after_pattern",
        "insert_before_pattern",
        # Code search (1 tool)
        "ripgrep_search",
        # Terminal (1 tool)
        "terminal_execute",
        # Task list (1 tool)
        "task_list_write_tool",
    ]

    return registered_tools


# Module-level registration info
__all__ = [
    # File operations
    "file_read",
    "file_write",
    "file_edit",
    "file_append",
    "file_delete",
    "list_directory",
    "create_directory",
    "read_file_range",
    "insert_at_line",
    "replace_lines",
    "insert_after_pattern",
    "insert_before_pattern",
    # Code search
    "ripgrep_search",
    # Terminal
    "terminal_execute",
    # Task list
    "task_list_write_tool",
    # Task list helper exports
    "get_task_list_manager",
    "task_list_write",
    "task_list_get",
    "task_list_clear",
    # Registration function
    "register_all_tools",
]

# Tool metadata for documentation
TOOL_SUMMARY = """
PromptChain Library Tools - Registered with CLI

Total Tools: 15 (12 file ops + 1 search + 1 terminal + 1 task list)
Token Impact: Makes core library tools accessible via registry

Tools:
1. file_read - Read complete file contents
2. file_write - Write/create files
3. file_edit - Find and replace text
4. file_append - Append to files
5. file_delete - Delete files
6. list_directory - List directory contents
7. create_directory - Create directories
8. read_file_range - Read specific line ranges
9. insert_at_line - Insert at line number
10. replace_lines - Replace line ranges
11. insert_after_pattern - Insert after regex match
12. insert_before_pattern - Insert before regex match
13. ripgrep_search - Fast code search
14. terminal_execute - Run shell commands
15. task_list_write_tool - Track progress on multi-step tasks

Category: UTILITY
Tags: file, filesystem, search, terminal, io, task, progress

Use Cases:
- File reading/writing/editing
- Code search and pattern matching
- Directory management
- Terminal command execution
- Log file manipulation
- Configuration file updates
- Multi-step task tracking and progress display
"""
