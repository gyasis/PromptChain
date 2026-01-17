# PRD: PromptChain CLI Internal Tools Ecosystem

**Version:** 1.0
**Date:** 2025-01-18
**Status:** Planning Phase
**Priority:** High (Post User Story 5 Integration)

---

## Executive Summary

Expand PromptChain CLI with a comprehensive suite of internal tools that agents can use autonomously, similar to Claude Code, Cursor, and Aider. These tools will enable the CLI to perform file operations, code execution, testing, Git operations, and more without requiring external dependencies or manual user intervention.

## Problem Statement

Currently, PromptChain CLI has limited built-in tools:
- ✅ Shell command execution (`!command`)
- ✅ File reference syntax (`@file.py`)
- ⚠️ Limited grep functionality (library level)
- ❌ No Git integration
- ❌ No test runner integration
- ❌ No code analysis tools
- ❌ No project scaffolding tools

**Competitive Gap:** Claude Code, Cursor, and Aider provide rich internal tooling that makes them more powerful for autonomous agent workflows.

## Goals & Success Metrics

### Primary Goals
1. **Agent Autonomy**: Enable agents to perform complex development tasks without user intervention
2. **Developer Experience**: Provide intuitive, fast, and reliable tools
3. **Extensibility**: Create a plugin architecture for adding custom tools
4. **Performance**: All tools should execute in <500ms for typical operations

### Success Metrics
- **Tool Coverage**: 20+ internal tools implemented
- **Usage Rate**: 50%+ of agent interactions use at least one tool
- **Error Rate**: <5% tool execution failures
- **Performance**: 95th percentile response time <500ms

## Research Findings: Competitive Analysis

### Claude Code Tools
- **File Operations**: Read, Write, Edit, Search
- **Code Execution**: Run tests, execute scripts
- **Git Integration**: Commit, PR creation, branch management
- **Planning Mode**: Analyze without changes
- **Sub-Agents**: Parallel task execution

### Cursor Tools
- **AI-Powered Editor**: Intelligent code completion
- **Git Operations**: Repository management, branching
- **.cursorrules**: Project-specific customization
- **Automated Testing**: Test generation and execution
- **Debugging**: Error log analysis
- **GitHub Actions**: CI/CD integration
- **MCP Support**: Model Context Protocol integration

### Aider Tools
- **Terminal-Based**: CLI pair programming
- **Git Integration**: Auto-commits with descriptive messages
- **Git Ingest**: Convert repos to single files
- **Execution Loops**: Run until desired output

### Key Takeaways
1. **Git integration is essential** - All major tools have it
2. **Test execution is standard** - Automated testing support
3. **File operations are foundational** - Read/write/search
4. **Execution loops are powerful** - Iterative refinement
5. **MCP adoption is growing** - Extensibility through protocols

---

## Proposed Tool Categories

### 1. File System Operationslets 
**Priority**: P0 (Critical)

#### Tools
- **`fs.read(path, encoding='utf-8')`** - Read file contents
  - Line range support
  - Binary file detection
  - Size limits with truncation

- **`fs.write(path, content, mode='overwrite')`** - Write file
  - Modes: overwrite, append, create_only
  - Atomic writes with backup
  - Permission handling

- **`fs.edit(path, old_text, new_text, count=1)`** - Edit file (like Edit tool)
  - Context-aware replacements
  - Regex support
  - Undo capability

- **`fs.search(pattern, path='.', type='content')`** - Search files
  - Types: content (grep), filename (find), both
  - Regex support
  - Glob patterns
  - Exclude patterns (.gitignore aware)

- **`fs.tree(path='.', depth=3, exclude=[])`** - Directory tree
  - Configurable depth
  - Exclude patterns
  - Size summaries

- **`fs.stat(path)`** - File/directory metadata
  - Size, permissions, timestamps
  - File type detection

- **`fs.copy(src, dst)`** - Copy files/directories
- **`fs.move(src, dst)`** - Move/rename files
- **`fs.delete(path, confirm=True)`** - Delete with safety checks
- **`fs.mkdir(path, parents=True)`** - Create directories
- **`fs.chmod(path, mode)`** - Change permissions
- **`fs.exists(path)`** - Check existence

**Implementation Notes:**
- Wrap Python `pathlib`, `os`, `shutil`
- Add safety checks (don't delete outside project)
- Support async operations
- Cache frequently accessed files

---

### 2. Code Analysis & Search
**Priority**: P0 (Critical)

#### Tools
- **`code.grep(pattern, path='.', context=0)`** - Enhanced grep
  - Regex patterns
  - Context lines (before/after)
  - Type filtering (by extension)
  - Exclude patterns
  - Count-only mode

- **`code.ast_search(query, path='.', lang='python')`** - AST-based search
  - Find functions, classes, imports
  - Semantic search (not just text)
  - Language-specific parsing

- **`code.symbols(path)`** - Extract code symbols
  - Functions, classes, variables
  - Docstrings
  - Type annotations

- **`code.imports(path)`** - Analyze imports
  - Detect unused imports
  - Suggest missing imports
  - Dependency tree

- **`code.lint(path, rules=[])`** - Code linting
  - Configurable rules
  - Auto-fix suggestions

- **`code.format(path, style='pep8')`** - Code formatting
  - Python: black, autopep8
  - JavaScript: prettier
  - Multiple languages

**Implementation Notes:**
- Use `ast` module for Python AST parsing
- Integrate `tree-sitter` for multi-language support
- Consider `ripgrep` wrapper for performance
- Cache parse results

---

### 3. Git Operations
**Priority**: P1 (High)

#### Tools
- **`git.status()`** - Repository status
  - Modified, staged, untracked files
  - Current branch
  - Commit behind/ahead count

- **`git.diff(file=None, staged=False)`** - Show changes
  - File-specific or all changes
  - Staged vs unstaged
  - Context lines

- **`git.log(n=10, file=None)`** - Commit history
  - Limit number of commits
  - File-specific history
  - Author, date, message

- **`git.commit(message, files='.')`** - Create commit
  - Auto-generate messages (AI-powered)
  - Stage specific files
  - Amend support

- **`git.branch(name=None, switch=False)`** - Branch operations
  - List branches
  - Create new branch
  - Switch branches

- **`git.checkout(branch_or_file)`** - Checkout branch/file
- **`git.pull(remote='origin', branch=None)`** - Pull changes
- **`git.push(remote='origin', branch=None)`** - Push changes
- **`git.merge(branch)`** - Merge branches
- **`git.stash(action='save', name=None)`** - Stash changes
- **`git.blame(file, line=None)`** - Show line history
- **`git.show(commit)`** - Show commit details

**Implementation Notes:**
- Use `GitPython` or subprocess to `git` CLI
- Safety checks (don't push without confirmation)
- AI-generated commit messages
- Conflict detection and guidance

---

### 4. Testing & Execution
**Priority**: P1 (High)

#### Tools
- **`test.run(path='.', pattern='test_*.py')`** - Run tests
  - Pytest integration
  - Filter by pattern
  - Verbose output
  - Coverage reports

- **`test.watch(path='.')`** - Watch mode (re-run on changes)
- **`test.coverage(path='.')`** - Coverage analysis
- **`test.generate(file)`** - Generate test stubs

- **`exec.run(command, timeout=30, cwd=None)`** - Execute command
  - Already implemented as `!command`
  - Enhance with better output capture
  - Streaming support

- **`exec.python(code, timeout=10)`** - Execute Python code
  - Sandboxed execution
  - Capture stdout/stderr
  - Return value capture

- **`exec.script(path, args=[])`** - Run script file
- **`exec.loop(command, condition, max_iter=10)`** - Execution loop
  - Run until condition met
  - Aider-style iterative execution

**Implementation Notes:**
- Use `pytest` for Python tests
- Sandbox execution for safety
- Capture output in real-time
- Handle timeouts gracefully

---

### 5. Project Management
**Priority**: P2 (Medium)

#### Tools
- **`project.init(name, template='python')`** - Initialize project
  - Templates: python, javascript, rust, etc.
  - Setup directory structure
  - Create config files

- **`project.scaffold(component, name)`** - Generate boilerplate
  - Components: class, function, test, module
  - Follow project conventions

- **`project.deps(action='list')`** - Dependency management
  - Actions: list, add, remove, update
  - Package managers: pip, npm, cargo

- **`project.config(key, value=None)`** - Manage configuration
  - Read/write .env, config files
  - Secrets management (warn about committing)

- **`project.clean()`** - Clean build artifacts
  - Remove __pycache__, dist/, build/
  - Configurable patterns

**Implementation Notes:**
- Template system with Jinja2
- Detect project type automatically
- Integration with package managers

---

### 6. Documentation & Analysis
**Priority**: P2 (Medium)

#### Tools
- **`docs.generate(path, format='markdown')`** - Generate docs
  - Extract docstrings
  - Create API documentation
  - Support multiple formats

- **`docs.readme(action='create')`** - README management
  - Generate from project structure
  - Update sections

- **`analyze.complexity(path)`** - Code complexity metrics
  - Cyclomatic complexity
  - Line counts
  - Maintainability index

- **`analyze.deps(path)`** - Dependency analysis
  - Import graph
  - Circular dependency detection
  - Unused dependency detection

- **`analyze.security(path)`** - Security scan
  - Known vulnerabilities
  - Unsafe patterns
  - Secrets detection

**Implementation Notes:**
- Use `radon` for complexity metrics
- Integrate security scanners (bandit, safety)
- Generate reports in multiple formats

---

### 7. Database Operations (Optional)
**Priority**: P3 (Low)

#### Tools
- **`db.query(sql, params=[])`** - Execute SQL query
- **`db.schema(table=None)`** - Show database schema
- **`db.migrate(action, name=None)`** - Database migrations
- **`db.seed(file)`** - Load seed data

**Implementation Notes:**
- Support SQLite, PostgreSQL, MySQL
- Safety checks (read-only by default)
- Transaction support

---

## Architecture Design

### Tool Registration System

```python
from promptchain.cli.tools import ToolRegistry

# Core tool registry
registry = ToolRegistry()

# Register tool with metadata
@registry.register(
    category="filesystem",
    description="Read file contents",
    parameters={
        "path": {"type": "string", "required": True},
        "encoding": {"type": "string", "default": "utf-8"}
    }
)
def fs_read(path: str, encoding: str = "utf-8") -> str:
    """Read file contents."""
    # Implementation
    pass
```

### Tool Execution Flow

```
User/Agent Request
    ↓
Tool Registry Lookup
    ↓
Parameter Validation
    ↓
Permission Check (safety)
    ↓
Tool Execution
    ↓
Result Formatting
    ↓
Return to Agent
```

### Safety & Sandboxing

```python
class ToolExecutor:
    def __init__(self, safe_mode: bool = True):
        self.safe_mode = safe_mode
        self.project_root = Path.cwd()

    def execute(self, tool_name: str, **kwargs):
        # Validate paths are within project
        if self.safe_mode:
            for key, value in kwargs.items():
                if isinstance(value, (str, Path)):
                    self._validate_path(value)

        # Execute tool
        tool = registry.get(tool_name)
        return tool(**kwargs)

    def _validate_path(self, path: Path):
        """Ensure path is within project root."""
        abs_path = path.resolve()
        if not abs_path.is_relative_to(self.project_root):
            raise SecurityError(f"Path {path} is outside project")
```

---

## Integration Points

### 1. Agent Integration
Agents can call tools directly:
```python
result = agent_chain.call_tool("fs.read", path="main.py")
result = agent_chain.call_tool("git.commit", message="Fix bug", files=".")
```

### 2. Command Syntax
Users can invoke tools explicitly:
```
# Tool command syntax
#fs.read main.py
#git.status
#test.run tests/
```

### 3. Auto-Tool Selection
Agent analyzes request and selects tools automatically:
```
User: "Find all TODO comments in Python files"
Agent: Uses code.grep("TODO", path=".", type="python")
```

---

## Implementation Phases

### Phase 1: Core File Operations (2 weeks)
- fs.read, fs.write, fs.edit
- fs.search (enhanced grep)
- fs.tree, fs.stat
- Safety validation layer

### Phase 2: Code Analysis (2 weeks)
- code.grep with context
- code.symbols extraction
- code.imports analysis
- AST-based search

### Phase 3: Git Integration (2 weeks)
- git.status, git.diff, git.log
- git.commit with AI messages
- git.branch, git.checkout
- git.push, git.pull with safety

### Phase 4: Testing & Execution (1 week)
- test.run with pytest
- exec.loop for iterative execution
- test.coverage integration

### Phase 5: Project Management (1 week)
- project.init with templates
- project.scaffold
- project.deps integration

### Phase 6: Documentation & Analysis (1 week)
- docs.generate
- analyze.complexity
- analyze.security

---

## Technical Requirements

### Dependencies
```toml
# Add to pyproject.toml
[tool.poetry.dependencies]
GitPython = "^3.1.0"        # Git operations
tree-sitter = "^0.20.0"     # Multi-language parsing
pytest = "^7.0.0"           # Testing framework
radon = "^6.0.0"            # Code complexity
bandit = "^1.7.0"           # Security scanning
jinja2 = "^3.1.0"           # Template engine
```

### File Structure
```
promptchain/cli/tools/
├── __init__.py
├── registry.py          # Tool registration system
├── executor.py          # Tool execution engine
├── safety.py            # Safety validation
├── filesystem/
│   ├── __init__.py
│   ├── read.py
│   ├── write.py
│   ├── search.py
│   └── tree.py
├── code/
│   ├── __init__.py
│   ├── grep.py
│   ├── ast_search.py
│   └── symbols.py
├── git/
│   ├── __init__.py
│   ├── status.py
│   ├── commit.py
│   └── diff.py
├── testing/
│   ├── __init__.py
│   ├── runner.py
│   └── coverage.py
└── project/
    ├── __init__.py
    ├── init.py
    └── scaffold.py
```

---

## Testing Strategy

### Unit Tests
- Each tool has comprehensive unit tests
- Mock external dependencies (Git, filesystem)
- Test error handling and edge cases

### Integration Tests
- Test tool chains (e.g., search → edit → commit)
- Test safety validation
- Test performance under load

### Performance Tests
- Benchmark each tool
- Ensure <500ms p95 latency
- Test with large codebases

---

## Security Considerations

### Path Traversal Prevention
```python
def validate_path(path: Path, project_root: Path) -> Path:
    """Ensure path is within project."""
    abs_path = path.resolve()
    if not abs_path.is_relative_to(project_root):
        raise SecurityError("Path outside project")
    return abs_path
```

### Command Injection Prevention
```python
def safe_exec(command: List[str], **kwargs):
    """Execute command safely."""
    # Use list form, not string
    # Validate command whitelist
    # Set timeouts
    # Capture output securely
    pass
```

### Secrets Detection
```python
def scan_for_secrets(content: str) -> List[str]:
    """Detect potential secrets in content."""
    patterns = [
        r"api_key\s*=\s*['\"]([^'\"]+)['\"]",
        r"password\s*=\s*['\"]([^'\"]+)['\"]",
        # More patterns...
    ]
    # Warn before committing
```

---

## Success Criteria

### Functional
- ✅ 20+ tools implemented across 6 categories
- ✅ All tools have comprehensive tests
- ✅ Safety validation prevents dangerous operations
- ✅ Tool registry supports plugins

### Performance
- ✅ p95 latency <500ms
- ✅ Memory usage <100MB per tool execution
- ✅ Support 1000+ file operations/min

### User Experience
- ✅ Intuitive tool naming and parameters
- ✅ Clear error messages
- ✅ Helpful auto-suggestions
- ✅ Works offline (no external API dependencies)

---

## Future Enhancements

### Plugin System
Allow users to create custom tools:
```python
# ~/.promptchain/plugins/my_tool.py
from promptchain.cli.tools import tool

@tool(category="custom", description="My custom tool")
def my_tool(arg: str) -> str:
    return f"Processed: {arg}"
```

### MCP Integration
Support Model Context Protocol for extensibility:
```python
# Connect to external MCP servers
mcp_config = {
    "filesystem": "mcp://filesystem-server",
    "database": "mcp://database-server"
}
```

### AI-Powered Tool Chaining
Agent automatically chains tools:
```
User: "Fix all linting errors and commit"
Agent:
  1. code.lint(path=".", auto_fix=True)
  2. git.commit(message="Fix linting errors")
```

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Security vulnerabilities | High | Medium | Comprehensive safety checks, sandboxing |
| Performance degradation | Medium | Low | Benchmarking, caching, async execution |
| Tool complexity overwhelms users | Medium | Medium | Clear documentation, auto-suggestions |
| External dependency failures | Low | Low | Graceful degradation, offline support |

---

## Appendix: Tool Reference

### Complete Tool List (Planned)

**File System (12 tools)**
- fs.read, fs.write, fs.edit, fs.search, fs.tree, fs.stat
- fs.copy, fs.move, fs.delete, fs.mkdir, fs.chmod, fs.exists

**Code Analysis (6 tools)**
- code.grep, code.ast_search, code.symbols
- code.imports, code.lint, code.format

**Git Operations (12 tools)**
- git.status, git.diff, git.log, git.commit
- git.branch, git.checkout, git.pull, git.push
- git.merge, git.stash, git.blame, git.show

**Testing (4 tools)**
- test.run, test.watch, test.coverage, test.generate

**Execution (4 tools)**
- exec.run, exec.python, exec.script, exec.loop

**Project (5 tools)**
- project.init, project.scaffold, project.deps
- project.config, project.clean

**Documentation (5 tools)**
- docs.generate, docs.readme, analyze.complexity
- analyze.deps, analyze.security

**Total: 48 tools across 7 categories**

---

## Approval & Sign-Off

**Document Owner:** PromptChain Development Team
**Last Updated:** 2025-01-18
**Status:** ✅ Ready for Review
**Next Steps:** Begin Phase 1 implementation after User Story 5 completion
