# Contract: FileReferenceParser

**Component**: `promptchain.cli.file_reference_parser.FileReferenceParser`
**Purpose**: Parse and resolve file/directory references from user messages
**Integration Points**: Session working directory, PromptChain message content

---

## Public API Contract

### Class Definition

```python
class FileReferenceParser:
    """
    Parses @file.txt and @directory/ references from messages.

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
        preview_lines_tail: int = 100
    ):
        """
        Initialize file reference parser.

        Args:
            working_directory: Session working directory for relative paths
            max_file_size: Max file size before truncation (bytes)
            preview_lines_head: Lines to include from file start (large files)
            preview_lines_tail: Lines to include from file end (large files)
        """
```

---

## Core Methods

### Method: parse_message

**Purpose**: Extract all file references from a message

**Signature**:
```python
def parse_message(self, message: str) -> tuple[str, list[FileReference]]:
    """
    Parse message for file references.

    Args:
        message: User message text

    Returns:
        tuple[str, list[FileReference]]:
            - Processed message (with references resolved)
            - List of FileReference objects

    Parsing Rules:
        - Detect @path syntax (e.g., @src/main.py, @docs/)
        - Resolve relative paths against working_directory
        - Validate file/directory existence
        - Read file contents (with truncation if needed)
        - For directories, discover relevant files

    Example:
        message = "Analyze @src/main.py and @tests/"
        processed, refs = parser.parse_message(message)
        # refs = [
        #   FileReference(path=Path("src/main.py"), content="...", is_directory=False),
        #   FileReference(path=Path("tests/"), files=[...], is_directory=True)
        # ]
    """
```

**Contract Guarantees**:
- All @references detected and resolved
- Invalid references generate warnings (not errors)
- File content truncated if >max_file_size
- Directory references discover relevant files (not all files)
- Binary files detected and skipped (metadata only)

**Example**:
```python
parser = FileReferenceParser(working_directory=Path.cwd())
message = "Review @README.md and explain the project"

processed, refs = parser.parse_message(message)
# processed = "Review @README.md and explain the project"
# refs = [FileReference(path=Path("README.md"), content="# Project...")]
```

---

### Method: resolve_reference

**Purpose**: Resolve single @reference to FileReference object

**Signature**:
```python
def resolve_reference(self, reference: str) -> FileReference | None:
    """
    Resolve @reference to FileReference.

    Args:
        reference: Path string from @syntax (e.g., "src/main.py")

    Returns:
        FileReference: Resolved file reference
        None: If file/directory doesn't exist or inaccessible

    Resolution Process:
        1. Resolve relative to working_directory
        2. Check existence and permissions
        3. Determine if file or directory
        4. Load content (files) or discover files (directories)
        5. Apply size/truncation limits

    Example:
        ref = parser.resolve_reference("src/main.py")
        # FileReference(
        #   path=Path("/home/user/project/src/main.py"),
        #   content="import os\n...",
        #   is_directory=False,
        #   size_bytes=5432,
        #   truncated=False
        # )
    """
```

**Contract Guarantees**:
- Returns None for non-existent paths (graceful)
- Returns None for permission errors (graceful)
- File content always truncated if >max_file_size
- Directory file discovery limited to relevant files

---

### Method: load_file_content

**Purpose**: Read file content with intelligent truncation

**Signature**:
```python
def load_file_content(self, path: Path) -> tuple[str, bool]:
    """
    Load file content with size-based truncation.

    Args:
        path: Absolute path to file

    Returns:
        tuple[str, bool]:
            - File content (possibly truncated)
            - Whether content was truncated

    Truncation Strategy:
        - Files ≤max_file_size: Read entirely
        - Files >max_file_size: Preview mode
          - First `preview_lines_head` lines (default: 500)
          - Separator: "... [truncated N bytes] ..."
          - Last `preview_lines_tail` lines (default: 100)

    Binary File Detection:
        - Check magic bytes for binary signatures
        - Return metadata string instead of content

    Example (small file):
        content, truncated = parser.load_file_content(Path("small.txt"))
        # content = "Full file content...", truncated = False

    Example (large file):
        content, truncated = parser.load_file_content(Path("large.log"))
        # content = "First 500 lines...\n... [truncated 500KB] ...\nLast 100 lines"
        # truncated = True
    """
```

**Contract Guarantees**:
- Never reads more than max_file_size bytes into memory
- Always indicates if truncation occurred
- Binary files return metadata: "Binary file: {filename} ({size})"
- UTF-8 decoding errors handled gracefully (try latin-1 fallback)

**Performance**: <500ms for files <10MB (SC-004)

---

### Method: discover_directory_files

**Purpose**: Find relevant files in a directory

**Signature**:
```python
def discover_directory_files(
    self,
    directory: Path,
    max_files: int = 20
) -> list[Path]:
    """
    Discover relevant files in directory.

    Args:
        directory: Directory path
        max_files: Maximum files to return

    Returns:
        list[Path]: Relevant file paths (sorted by relevance)

    Relevance Heuristics:
        - Prioritize common code files (.py, .js, .ts, .md, .txt)
        - Skip hidden files/directories (starting with .)
        - Skip common ignore patterns (node_modules, __pycache__, .git)
        - Sort by file extension priority, then alphabetically

    Example:
        files = parser.discover_directory_files(Path("src/"))
        # [Path("src/main.py"), Path("src/utils.py"), ...]
        # (max 20 files, code files prioritized)
    """
```

**Contract Guarantees**:
- Never returns more than max_files
- Respects .gitignore patterns (if present)
- Non-recursive by default (single directory level)
- Fast discovery (no content reading, just listing)

---

## FileReference Data Structure

**Contract**:
```python
@dataclass
class FileReference:
    """Represents a file or directory reference."""

    path: Path                    # Absolute path
    is_directory: bool            # True if directory
    content: str | None = None    # File content (None for directories)
    files: list[Path] | None = None  # Discovered files (directories only)
    size_bytes: int = 0           # Original file size
    truncated: bool = False       # True if content was truncated
    preview_lines: int | None = None  # Lines in preview (if truncated)
    modified_at: float | None = None  # File mtime
    error: str | None = None      # Error message if ref failed

    def to_message_context(self) -> str:
        """
        Convert reference to context string for LLM.

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
            - config.py
            ...
            --- END DIRECTORY ---

        Format (truncated):
            --- FILE: large.log (500 KB, TRUNCATED, showing first 500 + last 100 lines) ---
            ...
            --- END FILE ---
        """
```

**Example Usage**:
```python
ref = FileReference(
    path=Path("src/main.py"),
    is_directory=False,
    content="import os\n...",
    size_bytes=5432,
    truncated=False,
    modified_at=1700000000.0
)

context = ref.to_message_context()
# --- FILE: src/main.py (5.4 KB, modified: 2024-11-16) ---
# import os
# ...
# --- END FILE ---
```

---

## Integration with PromptChain

**Message Augmentation**:
```python
def augment_message_with_references(
    message: str,
    references: list[FileReference]
) -> str:
    """
    Combine message with file reference context.

    Args:
        message: Original user message
        references: Parsed file references

    Returns:
        str: Augmented message for LLM

    Example:
        original = "Analyze @src/main.py"
        refs = [FileReference(...)]
        augmented = augment_message_with_references(original, refs)

        # Result:
        # Analyze @src/main.py
        #
        # Referenced Files:
        # --- FILE: src/main.py (5.4 KB) ---
        # import os
        # ...
        # --- END FILE ---
    """
```

**Usage in CLI**:
```python
# User types: "Review @README.md"
raw_message = "Review @README.md"

# Parse references
parser = FileReferenceParser(session.working_directory)
processed_message, refs = parser.parse_message(raw_message)

# Augment message for LLM
llm_message = augment_message_with_references(processed_message, refs)

# Send to PromptChain
response = await agent_chain.process_prompt(llm_message)
```

---

## Error Handling Contracts

### FileReferenceError

```python
class FileReferenceError(Exception):
    """Base class for file reference errors."""
    pass

class FileNotFoundError(FileReferenceError):
    """Raised when referenced file doesn't exist."""
    def __init__(self, path: Path):
        super().__init__(f"File not found: {path}")

class FilePermissionError(FileReferenceError):
    """Raised when file is not readable."""
    def __init__(self, path: Path):
        super().__init__(f"Permission denied: {path}")

class FileTooLargeError(FileReferenceError):
    """Raised when file exceeds size limits (non-fatal, use truncation)."""
    def __init__(self, path: Path, size: int):
        super().__init__(f"File too large: {path} ({size} bytes)")
```

**Graceful Degradation**:
```python
# Errors don't crash, just log warning and continue
ref = resolve_reference("nonexistent.txt")
# ref = FileReference(path=..., error="File not found: nonexistent.txt")

# UI displays warning but continues
logger.warning(f"Failed to load {ref.path}: {ref.error}")
```

---

## Testing Contract

### Contract Tests

```python
# tests/cli/contract/test_file_reference_parser.py

def test_parse_message_single_file():
    """Contract: Single file reference parsed correctly."""
    parser = FileReferenceParser(Path.cwd())
    # Create test file
    test_file = Path("test.txt")
    test_file.write_text("Hello world")

    message = "Analyze @test.txt"
    processed, refs = parser.parse_message(message)

    assert len(refs) == 1
    assert refs[0].path == test_file.absolute()
    assert refs[0].content == "Hello world"
    assert not refs[0].is_directory

def test_parse_message_directory():
    """Contract: Directory reference discovers files."""
    parser = FileReferenceParser(Path.cwd())
    # Create test directory
    test_dir = Path("testdir")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "file1.py").write_text("import os")
    (test_dir / "file2.py").write_text("import sys")

    message = "Review @testdir/"
    processed, refs = parser.parse_message(message)

    assert len(refs) == 1
    assert refs[0].is_directory
    assert len(refs[0].files) >= 2

def test_file_truncation_large_file():
    """Contract: Large files truncated correctly (SC-004)."""
    parser = FileReferenceParser(Path.cwd(), max_file_size=1000)
    # Create large file
    large_file = Path("large.txt")
    large_file.write_text("line\n" * 1000)  # ~5KB

    content, truncated = parser.load_file_content(large_file)

    assert truncated
    assert "truncated" in content.lower()
    assert len(content) < 5000  # Smaller than original

def test_performance_file_load():
    """Contract: File loading meets performance target (SC-004)."""
    parser = FileReferenceParser(Path.cwd())
    # Create 5MB file
    medium_file = Path("medium.txt")
    medium_file.write_text("x" * (5 * 1024 * 1024))

    start = time.perf_counter()
    content, truncated = parser.load_file_content(medium_file)
    duration = time.perf_counter() - start

    assert duration < 0.5, f"Load took {duration}s, exceeds 500ms (SC-004)"

def test_binary_file_detection():
    """Contract: Binary files handled gracefully."""
    parser = FileReferenceParser(Path.cwd())
    # Create binary file
    binary_file = Path("test.bin")
    binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")

    content, truncated = parser.load_file_content(binary_file)

    assert "Binary file" in content
    assert "test.bin" in content

def test_graceful_error_handling():
    """Contract: Errors don't crash, just warn."""
    parser = FileReferenceParser(Path.cwd())

    message = "Analyze @nonexistent.txt"
    processed, refs = parser.parse_message(message)

    # Should still return reference with error
    assert len(refs) == 1
    assert refs[0].error is not None
    assert "not found" in refs[0].error.lower()
```

---

## Configuration Options

**User Configuration** (future):
```python
# ~/.promptchain/config.json
{
  "file_references": {
    "max_file_size": 102400,  // 100KB
    "preview_lines_head": 500,
    "preview_lines_tail": 100,
    "ignore_patterns": [
      "node_modules/",
      "__pycache__/",
      ".git/",
      "*.pyc"
    ],
    "binary_extensions": [".bin", ".exe", ".jpg", ".png"]
  }
}
```

---

## Security Considerations

**Path Traversal Prevention**:
```python
def resolve_path_safely(reference: str, working_dir: Path) -> Path:
    """
    Resolve path with security checks.

    Security Measures:
        - Reject absolute paths starting with /
        - Reject .. parent directory traversal
        - Ensure resolved path is within working_dir
        - Follow symlinks but validate target

    Raises:
        SecurityError: If path escapes working directory
    """
    # Reject absolute paths
    if reference.startswith('/'):
        raise SecurityError("Absolute paths not allowed")

    # Resolve relative to working dir
    full_path = (working_dir / reference).resolve()

    # Ensure within working dir
    if not full_path.is_relative_to(working_dir):
        raise SecurityError(f"Path escapes working directory: {reference}")

    return full_path
```

---

## Backward Compatibility

**Version 1.0 (Initial)**:
- `@syntax` for file references is stable
- FileReference data structure is frozen
- Truncation algorithm stable (no surprise behavior changes)

**Future Extensions**:
- Glob pattern support: `@src/**/*.py` (optional)
- Git integration: `@git:main:src/file.py` (optional)
- Remote files: `@https://...` (optional, security review needed)
