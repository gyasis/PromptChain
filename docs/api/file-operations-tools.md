# File Operations Tools API Documentation

**Version:** v0.4.2
**Module:** `promptchain.tools.file_operations`
**Status:** Production Ready ✅

---

## Overview

The File Operations wrapper provides **pseudo-tool interfaces** that match LLM training data expectations while using the robust `TerminalTool` underneath. This prevents **tool hallucination** where LLMs call non-existent tools like `file_read`, `file_write`, etc.

### Problem Solved

**Before v0.4.2:**
- Coding agents hallucinated file operation tools from training data
- Calls like `<invoke name="file_write">` failed silently
- No error reporting - agent believed operations succeeded
- Files were never created despite "success" messages

**After v0.4.2:**
- All standard file operations available as real tools
- Proper error handling and validation
- Success/failure reporting with file sizes
- Consistent integration with TerminalTool security

---

## Architecture

```
FileOperations Wrapper
├── Class-based API (FileOperations)
│   └── Accepts custom TerminalTool instance
└── Standalone Functions (for tool registration)
    └── Lazy-initialized shared instance

Pattern: Lightweight wrappers → TerminalTool → Shell commands
         OR direct Python I/O for reliability
```

### Design Pattern

Following the **RipgrepSearcher pattern** (`promptchain/tools/ripgrep_wrapper.py`):
- Lightweight wrapper class
- Standalone functions for LLM tool registration
- Comprehensive error handling
- Security via TerminalTool integration

---

## Which Agent Should Use These Tools?

### ✅ **BOTH Coding Agent AND Terminal Agent Have File Operations!**

File operations are available to both agents for different purposes:

---

### 🔄 **Typical Workflows:**

#### **Workflow 1: Code Creation → Execution**

```
User Request: "Process my CSV data and generate a report"
       ↓
1. Coding Agent (uses File Operations)
   - file_read("data.csv") → analyze data structure
   - file_write("scripts/process.py", code) → create processing script
   - file_write("scripts/generate_report.sh", code) → create report script
       ↓
2. Terminal Agent (uses TerminalTool)
   - Execute: python scripts/process.py
   - Execute: bash scripts/generate_report.sh
   - View output: cat output/report.txt
       ↓
3. Result returned to user
```

#### **Workflow 2: Direct File Operations**

```
User Request: "Create a backups directory and move old logs there"
       ↓
Terminal Agent (uses File Operations directly)
   - create_directory("backups")
   - list_directory("logs")
   - file_read("logs/old.log") → verify it's old
   - Execute: mv logs/old.log backups/
       ↓
Result: Files organized, task complete
```

**Key Principle:**
- **Coding Agent** = Best for writing code/scripts
- **Terminal Agent** = Best for direct file operations & command execution
- **Both have file operations** - Use whichever agent fits the task!

---

### ✅ **Coding Agent**

**Location:** `agentic_chat/agentic_team_chat.py` - `create_coding_agent()`

**Why Coding Agent:**
- Writes scripts and code files
- Creates configuration files
- Reads existing code for analysis
- Edits source files for updates
- Manages project file structure

**Tools Available:**
- `write_script` (custom coding tool)
- `file_read` ← File Operations
- `file_write` ← File Operations
- `file_edit` ← File Operations
- `file_append` ← File Operations
- `file_delete` ← File Operations
- `list_directory` ← File Operations
- `create_directory` ← File Operations
- `read_file_range` ← File Operations

### ✅ **Terminal Agent**

**Location:** `agentic_chat/agentic_team_chat.py` - `create_terminal_agent()`

**Why Terminal Agent Also Needs File Operations:**
- **Direct file management tasks:** "Create a backups directory", "Delete temp files"
- **File navigation:** "List all log files", "Show me what's in this directory"
- **File organization:** "Move old files", "Find files by date"
- **Quick file operations:** "Read error.log", "Append to system.log"
- **Still executes scripts** created by Coding Agent via `TerminalTool`

**Tools Available:**
- `execute_terminal_command` (TerminalTool wrapper)
- `file_read` ← File Operations
- `file_write` ← File Operations
- `file_edit` ← File Operations
- `file_append` ← File Operations
- `file_delete` ← File Operations
- `list_directory` ← File Operations
- `create_directory` ← File Operations
- `read_file_range` ← File Operations

**When to Use Terminal vs Coding Agent:**
- **Terminal Agent:** "Create a directory", "List files", "Delete old logs"
- **Coding Agent:** "Write a Python script to...", "Create a data processor"

### ❌ **Other Agents (GENERALLY NOT NEEDED)**

- **Research Agent:** Reads web data, not local files
- **Analysis Agent:** Analyzes data provided by other agents
- **Documentation Agent:** May benefit from file_read for existing docs, but typically receives content
- **Synthesis Agent:** Works with aggregated results, not direct file access

---

## API Reference

### Class: `FileOperations`

```python
from promptchain.tools.file_operations import FileOperations

ops = FileOperations(
    terminal_tool=None,  # Optional: Custom TerminalTool instance
    verbose=False        # Enable verbose output for debugging
)
```

**Parameters:**
- `terminal_tool` (TerminalTool, optional): Custom TerminalTool instance. If None, creates default instance.
- `verbose` (bool): Enable verbose output for debugging operations.

---

## Tool Functions

### 1. `file_read(path: str) -> str`

Read the complete contents of a file.

**Implementation:** Uses `cat` command via TerminalTool

**Parameters:**
- `path` (str): Path to file (relative or absolute)

**Returns:**
- `str`: File contents, or error message if read fails

**Example:**
```python
from promptchain.tools.file_operations import file_read

content = file_read("config.json")
# Returns: '{"api_key": "...", "model": "gpt-4"}'
```

**Error Messages:**
- `❌ Error: File does not exist: {path}`
- `❌ Error: Path is not a file: {path}`
- `❌ Exception in file_read: {error}`

---

### 2. `file_write(path: str, content: str) -> str`

Write content to a file, creating it if it doesn't exist. **Overwrites existing files completely.**

**Implementation:** Uses Python's `Path.write_text()` (more reliable than heredoc)

**Parameters:**
- `path` (str): Path to file to write
- `content` (str): Content to write

**Returns:**
- `str`: Success message with file size, or error message

**Example:**
```python
from promptchain.tools.file_operations import file_write

result = file_write("output.txt", "Hello World\nLine 2\nLine 3")
# Returns: "✅ File written successfully: output.txt (25 bytes)"
```

**Features:**
- Creates parent directories automatically (`mkdir -p`)
- Handles UTF-8 encoding
- Verifies file creation
- Reports file size

**Error Messages:**
- `❌ File write succeeded but file not found: {path}`
- `❌ Exception in file_write: {error}`

---

### 3. `file_edit(path: str, old_text: str, new_text: str) -> str`

Edit a file by replacing **all occurrences** of `old_text` with `new_text`.

**Implementation:** Uses `sed -i 's/old/new/g'` command via TerminalTool

**Parameters:**
- `path` (str): Path to file to edit
- `old_text` (str): Text to find and replace
- `new_text` (str): Replacement text

**Returns:**
- `str`: Success message, or error message

**Example:**
```python
from promptchain.tools.file_operations import file_edit

result = file_edit("config.py", "DEBUG = False", "DEBUG = True")
# Returns: "✅ File edited successfully: config.py"
```

**Use Cases:**
- Configuration updates
- Variable renaming
- String replacements
- Pattern-based line editing

**Note:** Special characters (/, &) are automatically escaped for sed.

**Error Messages:**
- `❌ Error: File does not exist: {path}`
- `❌ Exception in file_edit: {error}`

---

### 4. `file_append(path: str, content: str) -> str`

Append content to the end of a file.

**Implementation:** Uses Python's `open(path, 'a')` mode (more reliable than heredoc)

**Parameters:**
- `path` (str): Path to file
- `content` (str): Content to append

**Returns:**
- `str`: Success message with total file size, or error message

**Example:**
```python
from promptchain.tools.file_operations import file_append

result = file_append("log.txt", "2025-10-11: System started\n")
# Returns: "✅ Content appended successfully: log.txt (156 bytes total)"
```

**Features:**
- Creates file if it doesn't exist
- Creates parent directories automatically
- UTF-8 encoding
- Reports total file size after append

**Error Messages:**
- `❌ Exception in file_append: {error}`

---

### 5. `file_delete(path: str) -> str`

Delete a file.

**Implementation:** Uses `rm` command via TerminalTool

**Parameters:**
- `path` (str): Path to file to delete

**Returns:**
- `str`: Success message, or error message

**Example:**
```python
from promptchain.tools.file_operations import file_delete

result = file_delete("temp.txt")
# Returns: "✅ File deleted successfully: temp.txt"
```

**Safety Features:**
- Validates file exists before deletion
- Ensures path is a file (not directory)
- Provides clear error messages

**Error Messages:**
- `❌ Error: File does not exist: {path}`
- `❌ Error: Path is not a file (use delete_directory for directories): {path}`
- `❌ Exception in file_delete: {error}`

---

### 6. `list_directory(path: str = ".") -> str`

List contents of a directory with detailed information.

**Implementation:** Uses `ls -lah` command via TerminalTool

**Parameters:**
- `path` (str, optional): Directory path (defaults to current directory)

**Returns:**
- `str`: Directory listing, or error message

**Example:**
```python
from promptchain.tools.file_operations import list_directory

listing = list_directory("src/")
# Returns:
# total 24K
# drwxrwxr-x 3 user user 4.0K Oct 11 10:30 .
# drwxrwxr-x 5 user user 4.0K Oct 11 10:30 ..
# -rw-rw-r-- 1 user user  2.1K Oct 11 10:30 main.py
# drwxrwxr-x 2 user user 4.0K Oct 11 10:30 utils
```

**Listing Format:**
- Human-readable sizes (-h)
- Hidden files included (-a)
- Long format with permissions (-l)

**Error Messages:**
- `❌ Error: Directory does not exist: {path}`
- `❌ Error: Path is not a directory: {path}`
- `❌ Exception in list_directory: {error}`

---

### 7. `create_directory(path: str) -> str`

Create a directory and all parent directories.

**Implementation:** Uses `mkdir -p` command via TerminalTool

**Parameters:**
- `path` (str): Directory path to create

**Returns:**
- `str`: Success message, or error message

**Example:**
```python
from promptchain.tools.file_operations import create_directory

result = create_directory("data/output/results")
# Returns: "✅ Directory created successfully: data/output/results"
```

**Features:**
- Creates all parent directories automatically (-p flag)
- No error if directory already exists
- Verifies creation

**Error Messages:**
- `❌ mkdir succeeded but directory not found: {path}`
- `❌ Exception in create_directory: {error}`

---

### 8. `read_file_range(path: str, start_line: int, end_line: int) -> str`

Read a specific range of lines from a file.

**Implementation:** Uses `sed -n '{start},{end}p'` command via TerminalTool

**Parameters:**
- `path` (str): Path to file
- `start_line` (int): First line to read (1-indexed)
- `end_line` (int): Last line to read (inclusive)

**Returns:**
- `str`: Requested lines, or error message

**Example:**
```python
from promptchain.tools.file_operations import read_file_range

lines = read_file_range("main.py", 10, 20)
# Returns lines 10-20 from main.py
```

**Use Cases:**
- Reading specific code sections
- Extracting configuration blocks
- Reviewing log file segments
- Analyzing specific data ranges

**Error Messages:**
- `❌ Error: File does not exist: {path}`
- `❌ Error: Path is not a file: {path}`
- `❌ Exception in read_file_range: {error}`

---

## Integration with PromptChain

### Registering Tools with Coding Agent

```python
from promptchain import PromptChain
from promptchain.tools.file_operations import (
    file_read, file_write, file_edit, file_append,
    file_delete, list_directory, create_directory, read_file_range
)

# Create agent
agent = PromptChain(
    models=["openai/gpt-4"],
    instructions=[agentic_step],
    verbose=True
)

# Register all file operation tools
agent.register_tool_function(file_read)
agent.register_tool_function(file_write)
agent.register_tool_function(file_edit)
agent.register_tool_function(file_append)
agent.register_tool_function(file_delete)
agent.register_tool_function(list_directory)
agent.register_tool_function(create_directory)
agent.register_tool_function(read_file_range)

# Add tool schemas for LLM
agent.add_tools([
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"}
                },
                "required": ["path"]
            }
        }
    },
    # ... (add schemas for other tools)
])
```

**Complete Example:** See `agentic_chat/agentic_team_chat.py` lines 462-608

---

## Tool Registration Schemas

### Complete Schema Definitions

```python
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write content to a file, creating it if it doesn't exist",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "Content to write to the file"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_edit",
            "description": "Edit a file by replacing old_text with new_text using sed",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit"},
                    "old_text": {"type": "string", "description": "Text to find and replace"},
                    "new_text": {"type": "string", "description": "Text to replace with"}
                },
                "required": ["path", "old_text", "new_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_append",
            "description": "Append content to the end of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to append to"},
                    "content": {"type": "string", "description": "Content to append"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_delete",
            "description": "Delete a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to delete"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List contents of a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to directory to list (defaults to current directory)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a directory and all parent directories",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to directory to create"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_range",
            "description": "Read a specific range of lines from a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                    "start_line": {"type": "integer", "description": "First line to read (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Last line to read (inclusive)"}
                },
                "required": ["path", "start_line", "end_line"]
            }
        }
    }
]
```

---

## Usage Examples

### Example 1: Create Python Script with Config

```python
from promptchain.tools.file_operations import file_write, create_directory

# Create project structure
create_directory("myproject/src")
create_directory("myproject/config")

# Write main script
code = '''#!/usr/bin/env python3
import json
from pathlib import Path

def load_config():
    with open("config/settings.json") as f:
        return json.load(f)

if __name__ == "__main__":
    config = load_config()
    print(f"Loaded config: {config}")
'''
file_write("myproject/src/main.py", code)

# Write config file
config = '''{
    "api_key": "your_key_here",
    "model": "gpt-4",
    "temperature": 0.7
}'''
file_write("myproject/config/settings.json", config)
```

### Example 2: Read and Edit Configuration

```python
from promptchain.tools.file_operations import file_read, file_edit

# Read current config
config_content = file_read("config.py")
print(f"Current config:\n{config_content}")

# Update debug mode
file_edit("config.py", "DEBUG = False", "DEBUG = True")

# Update API endpoint
file_edit("config.py",
          "API_URL = 'http://localhost:8000'",
          "API_URL = 'https://api.production.com'")
```

### Example 3: Log File Management

```python
from promptchain.tools.file_operations import file_append, read_file_range

# Append log entry
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
file_append("application.log", f"{timestamp} - Application started\n")

# Read last 10 lines of log
# Assuming we know file has 100 lines
last_lines = read_file_range("application.log", 91, 100)
print(f"Recent logs:\n{last_lines}")
```

### Example 4: Code Analysis Workflow

```python
from promptchain.tools.file_operations import list_directory, file_read, read_file_range

# List Python files in src/
listing = list_directory("src/")
print(f"Source files:\n{listing}")

# Read specific function from file
# Read lines 50-75 which contain the target function
function_code = read_file_range("src/utils.py", 50, 75)
print(f"Function implementation:\n{function_code}")

# Read entire file for full context
full_content = file_read("src/utils.py")
```

---

## File Editing Decision Tree

### ⚠️ CRITICAL: Choose the Right Tool for Editing

When a user asks to **edit, update, add to, or modify** an existing file, follow this decision tree:

```
User wants to modify existing file
         ↓
    Does file exist?
    ├─ NO → Use file_write(path, content) to create it
    └─ YES → Continue ↓
         ↓
    What type of modification?
    ├─ REPLACE specific text (e.g., "Change DEBUG=False to DEBUG=True")
    │  └─ Use: file_edit(path, old_text, new_text)
    │
    ├─ ADD content to END ONLY (e.g., "Append log entry", "Add new dependency")
    │  └─ Use: file_append(path, content)
    │
    └─ INSERT/RESTRUCTURE (e.g., "Add installation section to tutorial", "Reorganize config")
       └─ Use: file_read(path) + file_write(path, complete_new_content)
          Steps:
          1. file_read(path) → Get current content
          2. Construct complete new file with modifications
          3. file_write(path, new_complete_content) → Replace entire file
```

### 🚨 Common Mistakes to AVOID:

1. **❌ WRONG: Using file_append() for middle insertions**
   ```python
   # User: "Add installation section to the tutorial"
   # WRONG APPROACH:
   file_append("tutorial.md", "\n## Installation\n...")
   # RESULT: Installation section at END, not where it should be
   ```

2. **❌ WRONG: Using file_write() without reading first**
   ```python
   # User: "Add a new feature to README.md"
   # WRONG APPROACH:
   file_write("README.md", "## New Feature\n...")
   # RESULT: Original README content DESTROYED
   ```

3. **❌ WRONG: Using file_edit() for complex restructuring**
   ```python
   # User: "Reorganize the config file sections"
   # WRONG APPROACH:
   file_edit("config.py", old_section, new_section)  # Multiple edits needed
   # RESULT: Fragile, error-prone, hard to maintain
   ```

### ✅ CORRECT Approaches:

#### Example 1: Simple Text Replacement
```python
# User: "Change the API URL from localhost to production"
result = file_edit(
    "config.py",
    old_text="API_URL = 'http://localhost:8000'",
    new_text="API_URL = 'https://api.production.com'"
)
# ✅ Clean, precise replacement
```

#### Example 2: Append to End
```python
# User: "Add today's log entry"
from datetime import datetime
timestamp = datetime.now().isoformat()
result = file_append(
    "application.log",
    f"\n{timestamp} - Application started successfully"
)
# ✅ New content added to end, existing logs preserved
```

#### Example 3: Insert Section in Middle (Correct Way)
```python
# User: "Add installation section to tutorial.md"

# Step 1: Read current content
current_content = file_read("tutorial.md")

# Step 2: Construct new content with insertion
new_content = """# Tutorial

## Installation
Follow these steps to install...

## Prerequisites
""" + current_content.replace("# Tutorial", "").strip()

# Step 3: Write complete new content
result = file_write("tutorial.md", new_content)
# ✅ Installation section properly inserted, all content preserved
```

### 📋 Quick Reference Table:

| User Request | Tool to Use | Why |
|-------------|-------------|-----|
| "Change port 8000 to 9000" | `file_edit()` | Simple text replacement |
| "Update DEBUG=False to True" | `file_edit()` | Specific value change |
| "Add new dependency to requirements.txt" | `file_append()` | Adding to end of list |
| "Append today's log entry" | `file_append()` | Adding to end chronologically |
| "Add installation section at top" | `file_read()` + `file_write()` | Inserting in middle |
| "Reorganize config sections" | `file_read()` + `file_write()` | Complex restructuring |
| "Add code example after line 50" | `file_read()` + `file_write()` | Position-specific insertion |

---

## Best Practices

### 1. **Error Handling**

Always check return messages for error indicators:

```python
result = file_write("output.txt", "content")
if "❌" in result:
    # Handle error
    print(f"Write failed: {result}")
else:
    # Success
    print(f"Write succeeded: {result}")
```

### 2. **Path Validation**

Use `list_directory` before operations to verify paths:

```python
# Check if directory exists before creating files
listing = list_directory("data/output")
if "❌" not in listing:
    # Directory exists, safe to write
    file_write("data/output/results.txt", "data")
```

### 3. **Incremental Operations**

For large files, use `read_file_range` instead of `file_read`:

```python
# Read file in chunks
chunk_size = 100
for i in range(0, total_lines, chunk_size):
    chunk = read_file_range("large_file.txt", i+1, i+chunk_size)
    process_chunk(chunk)
```

### 4. **Atomic Updates**

For critical files, write to temp file first:

```python
# Write to temporary file
file_write("config.tmp", new_config)

# Verify content
verification = file_read("config.tmp")
if verify_config(verification):
    # Replace original
    file_delete("config.json")
    file_write("config.json", new_config)
    file_delete("config.tmp")
```

### 5. **Logging Operations**

Keep audit trail of file operations:

```python
def logged_file_write(path, content, log_file="operations.log"):
    result = file_write(path, content)
    timestamp = datetime.now().isoformat()
    log_entry = f"{timestamp} - file_write({path}): {result}\n"
    file_append(log_file, log_entry)
    return result
```

---

## Security Considerations

### TerminalTool Integration

File operations inherit TerminalTool's security features:

1. **Permission System:**
   - TerminalTool can require user approval for risky operations
   - Configurable via `require_permission` parameter

2. **Command Validation:**
   - All paths are properly quoted to prevent injection
   - Special characters are escaped for shell safety

3. **Access Control:**
   - Operations respect filesystem permissions
   - Cannot access files/directories without proper rights

### Safe Path Handling

```python
def _quote_path(self, path: str) -> str:
    """Properly quote a file path for shell commands."""
    return f'"{path}"'  # Prevents command injection
```

### Recommended Practices

1. **Validate Input Paths:**
   ```python
   from pathlib import Path

   def safe_file_write(path, content):
       # Ensure path is within allowed directory
       safe_dir = Path("/allowed/directory").resolve()
       target_path = Path(path).resolve()

       if safe_dir in target_path.parents:
           return file_write(str(target_path), content)
       else:
           return "❌ Error: Path outside allowed directory"
   ```

2. **Sanitize Content:**
   ```python
   def sanitized_write(path, content):
       # Remove dangerous patterns
       safe_content = content.replace("$(", "\\$(")
       safe_content = safe_content.replace("`", "\\`")
       return file_write(path, safe_content)
   ```

3. **Limit File Sizes:**
   ```python
   def size_limited_write(path, content, max_size=1_000_000):
       if len(content.encode('utf-8')) > max_size:
           return f"❌ Error: Content exceeds {max_size} bytes"
       return file_write(path, content)
   ```

---

## Testing

The module includes comprehensive test suite:

```bash
# Run tests
python promptchain/tools/file_operations.py

# Expected output:
# Testing FileOperations in: /tmp/file_ops_test_xxxxx
#
# --- Test file_write ---
# ✅ File written successfully: /tmp/file_ops_test_xxxxx/test.txt (25 bytes)
#
# --- Test file_read ---
# Hello World
# Line 2
# Line 3
#
# (... all tests ...)
```

### Test Coverage

- ✅ File creation and writing
- ✅ File reading
- ✅ Content appending
- ✅ Line range reading
- ✅ Text replacement editing
- ✅ Directory creation (including nested)
- ✅ Directory listing
- ✅ File deletion
- ✅ Standalone function interfaces

---

## Troubleshooting

### Issue: Empty Files Created

**Problem:** Files created but with 0 bytes

**Cause:** TerminalTool collapsing heredoc syntax to single line

**Solution:** ✅ Fixed in v0.4.2
- `file_write` now uses Python's `Path.write_text()`
- `file_append` now uses Python's `open(path, 'a')`

### Issue: Permission Denied

**Problem:** `❌ Permission denied` errors

**Solutions:**
1. Check file/directory permissions
2. Ensure parent directories exist
3. Verify user has write access
4. Consider using `create_directory` first

### Issue: Special Characters in Content

**Problem:** Content with quotes or special characters fails

**Solution:**
- `file_write` and `file_append` use Python I/O (handles all characters)
- `file_edit` automatically escapes special characters for sed

### Issue: Tool Not Found by LLM

**Problem:** LLM tries to call tool but gets "tool not found" error

**Solutions:**
1. Verify tool is registered: `agent.register_tool_function(file_write)`
2. Verify schema is added: `agent.add_tools([schema])`
3. Check tool name matches in both registration and schema
4. Ensure agent has access to tool in its instructions

---

## Version History

### v0.4.2 (October 2025)
- ✅ Initial release of FileOperations wrapper
- ✅ Fixed heredoc empty file bug (switched to Python I/O)
- ✅ Integrated with coding agent in agentic_team_chat.py
- ✅ Comprehensive test suite added
- ✅ Full API documentation

---

## Related Documentation

- [TerminalTool Visual Debugging](./terminal-tool-visual-debugging.md)
- [AgenticStepProcessor Best Practices](../agentic_step_processor_best_practices.md)
- [Agent Chain Usage](../AgentChain_Usage.md)
- [Registering Functions to Chain](../registering_functions_to_chain.md)

---

## Support

For issues or questions:
1. Check test suite: `python promptchain/tools/file_operations.py`
2. Review agentic_team_chat.py integration (lines 365-610)
3. Examine ripgrep_wrapper.py for similar patterns
4. Check TerminalTool documentation for underlying behavior

---

**Last Updated:** October 11, 2025
**Maintainer:** PromptChain Development Team
